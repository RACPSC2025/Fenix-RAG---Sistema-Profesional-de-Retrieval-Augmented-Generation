"""
Grafo LangGraph principal de Fénix RAG.

Arquitectura del grafo:
                    ┌──────────────────────┐
                    │      __start__       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   document_router    │  ← clasifica archivos subidos
                    └──────┬──────┬────────┘
                   uploads │      │ sin uploads
                           │      │
              ┌────────────▼──┐   │
              │   ingestion   │   │
              └────────┬──────┘   │
                       │          │
              ┌────────▼──────────▼──┐
              │      retrieval       │  ← Multi-Query Fusion + dedup
              └────────────┬─────────┘
                           │
              ┌────────────▼─────────┐
              │        grade         │  ← CRAG → escribe crag_route
              └──────┬───────────────┘
      correct / max  │   \ ambiguous|incorrect (+ retry guard)
      retries agot.  │
              ┌──────▼───────┐   ┌──────────────────────┐
              │  generation  │   │  retrieval (retry)   │
              └──────┬───────┘   └──────────────────────┘
                     │
              ┌──────▼───────┐
              │  reflection  │  ← escribe reflection_route (NO route)
              └──────┬───┬───┘
   reflection_route  │   │ reflection_route
         == "END"    │   │ == "retrieval"
                     │   │
              ┌──────▼─┐  └─▼──────────────────┐
              │ __end__ │   │ retrieval (retry) │
              └─────────┘   └───────────────────┘
                             (máx. max_iterations)

ROUTING — CAMPOS AISLADOS POR RESPONSABILIDAD:
  route_after_router     → lee `route`           (document_router)
  route_after_ingestion  → lee `error`           (ingestion_node)
  route_after_grading    → lee `crag_route`      (grade_documents_node)
  route_after_generation → lee `messages`        (generation_node / ToolNode)
  route_after_reflection → lee `reflection_route`(reflection_node)

  Ninguna función de routing lee el mismo campo que otra.
  Esto elimina toda posibilidad de colisión de estado entre nodos.

Herramientas disponibles (via wrapped_tool_node con sync memory):
  - ingest_pdf, ingest_excel, ingest_word, ingest_image_pdf
  - semantic_search, hybrid_search, article_lookup
  - list_indexed_documents
  - save_context, retrieve_context, list_context_keys, clear_context
  - load_skill, search_skills, list_available_profiles [Fase 10]
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes.all_nodes import (
    document_router_node,
    generation_node,
    ingestion_node,
    reflection_node,
    retrieval_node,
    supervisor_node,
)
from src.agent.skills.crag import grade_documents_node, route_after_grading
from src.agent.state import AgentState, initial_state
from src.agent.tools.ingest_tools import (
    ingest_excel,
    ingest_image_pdf,
    ingest_pdf,
    ingest_word,
    list_indexed_documents,
)
from src.agent.tools.memory_tools import (
    clear_context,
    list_context_keys,
    retrieve_context,
    save_context,
)
from src.agent.tools.search_tools import (
    article_lookup,
    hybrid_search,
    semantic_search,
)
from src.agent.tools.skill_tools import (
    list_available_profiles,
    load_skill,
    search_skills,
)
from src.config.logging import get_logger

log = get_logger(__name__)

ALL_TOOLS = [
    # Ingestion tools (5)
    ingest_pdf,
    ingest_excel,
    ingest_word,
    ingest_image_pdf,
    list_indexed_documents,
    # Search tools (3)
    semantic_search,
    hybrid_search,
    article_lookup,
    # Memory tools (4)
    save_context,
    retrieve_context,
    list_context_keys,
    clear_context,
    # Skill tools (3) — Fase 10
    load_skill,
    search_skills,
    list_available_profiles,
]


# ─── Funciones de routing condicional ─────────────────────────────────────────

def route_after_router(state: AgentState) -> Literal["ingestion", "retrieval"]:
    """Lee `route` — escrito por document_router_node."""
    route = state.get("route", "retrieval")
    return "ingestion" if route == "ingestion" else "retrieval"


def route_after_ingestion(state: AgentState) -> Literal["retrieval", "__end__"]:
    """Lee `error` e `ingested_documents` — escritos por ingestion_node."""
    if state.get("error") and not state.get("ingested_documents"):
        return END
    return "retrieval"


def route_after_reflection(
    state: AgentState,
) -> Literal["retrieval", "__end__"]:
    """
    Lee `reflection_route` — escrito exclusivamente por reflection_node.

    NO lee `route` para evitar colisión con:
      - CRAG (escribe `crag_route`)
      - supervisor (escribe `route`)

    "END"       → respuesta aprobada o iteraciones agotadas
    "retrieval" → score bajo, active_query reformulada disponible
    """
    reflection_route = state.get("reflection_route", "END")
    if reflection_route == "END":
        return END
    return "retrieval"


def route_after_generation(
    state: AgentState,
) -> Literal["tools", "reflection"]:
    """
    Lee `messages` — si el último mensaje tiene tool_calls → ToolNode.
    Patrón ReAct estándar de LangGraph.
    """
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
    return "reflection"


# ─── Constructor del grafo ────────────────────────────────────────────────────

def build_graph(
    with_tools: bool = True,
    checkpointer: Any | None = None,
) -> StateGraph:
    """
    Construye y compila el StateGraph principal de Fénix RAG.

    Args:
        with_tools: Si True, agrega wrapped_tool_node con sync de memoria.
        checkpointer: Checkpointer para persistencia entre invocaciones.

    Returns:
        StateGraph compilado listo para graph.invoke().
    """
    builder = StateGraph(AgentState)

    # ── Nodos ─────────────────────────────────────────────────────────────────
    builder.add_node("document_router", document_router_node)
    builder.add_node("ingestion", ingestion_node)
    builder.add_node("retrieval", retrieval_node, retry=3)
    builder.add_node("grade", grade_documents_node)
    builder.add_node("generation", generation_node, retry=2)
    builder.add_node("reflection", reflection_node)
    builder.add_node("supervisor", supervisor_node)

    # Context Window Manager — comprime historial si supera el umbral
    from src.agent.middleware.context_window_manager import context_manager_node  # noqa: PLC0415
    builder.add_node("context_manager", context_manager_node)

    if with_tools:
        from src.agent.tools.memory_tools import get_memory_store, get_context_metrics  # noqa: PLC0415

        # _tool_node se define UNA sola vez — evita GC pressure en alta concurrencia
        _tool_node = ToolNode(tools=ALL_TOOLS)

        def wrapped_tool_node(state: AgentState) -> dict:
            """
            Envuelve ToolNode con sync bidireccional de session_memory.

            1. sync_from_state: PostgreSQL (via checkpointer) → in-memory store
            2. _tool_node.invoke: tools mutan el in-memory store
            3. sync_to_state: in-memory store → estado del grafo → PostgreSQL
            """
            store = get_memory_store()
            session_id = state.get("session_id", "default")
            store.sync_from_state(session_id, state.get("session_memory", {}))

            result = _tool_node.invoke(state)

            updated_memory = store.sync_to_state(session_id)
            if isinstance(result, dict):
                result["session_memory"] = updated_memory
            return result

        builder.add_node("tools", wrapped_tool_node)

    # ── Edges lineales ────────────────────────────────────────────────────────
    builder.add_edge(START, "document_router")
    builder.add_edge("retrieval", "grade")   # CRAG evalúa ANTES de generation

    # ── Edges condicionales ───────────────────────────────────────────────────
    builder.add_conditional_edges(
        "document_router",
        route_after_router,
        {"ingestion": "ingestion", "retrieval": "retrieval"},
    )

    builder.add_conditional_edges(
        "ingestion",
        route_after_ingestion,
        {"retrieval": "retrieval", END: END},
    )

    # CRAG: route_after_grading lee `crag_route` (via crag.py)
    # correct / max_retries_agotados → generation
    # ambiguous | incorrect (con retries) → retrieval
    builder.add_conditional_edges(
        "grade",
        route_after_grading,          # definida en crag.py, lee crag_route
        {"generation": "generation", "retrieval": "retrieval"},
    )

    if with_tools:
        builder.add_conditional_edges(
            "generation",
            route_after_generation,
            {"tools": "tools", "reflection": "reflection"},
        )
        builder.add_edge("tools", "generation")
    else:
        builder.add_edge("generation", "reflection")

    # Reflection → context_manager → END/retrieval
    # El context_manager comprime el historial si es necesario, luego se ruta
    builder.add_edge("reflection", "context_manager")

    builder.add_conditional_edges(
        "context_manager",
        route_after_reflection,  # Sigue leyendo reflection_route
        {"retrieval": "retrieval", END: END},
    )

    # ── Compilar ──────────────────────────────────────────────────────────────
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    try:
        from langgraph.cache.memory import InMemoryCache  # noqa: PLC0415
        compile_kwargs["cache"] = InMemoryCache()
    except ImportError:
        pass

    graph = builder.compile(**compile_kwargs)

    log.info(
        "graph_compiled",
        nodes=list(builder.nodes.keys()) if hasattr(builder, "nodes") else "unknown",
        with_tools=with_tools,
        with_checkpointer=checkpointer is not None,
    )
    return graph


# ─── Singleton ────────────────────────────────────────────────────────────────

_graph: StateGraph | None = None


def get_graph(
    with_tools: bool = True,
    checkpointer: Any | None = None,
    force_rebuild: bool = False,
) -> StateGraph:
    """Retorna la instancia singleton del grafo compilado."""
    global _graph  # noqa: PLW0603
    if _graph is None or force_rebuild:
        _graph = build_graph(with_tools=with_tools, checkpointer=checkpointer)
    return _graph


# ─── API de alto nivel ────────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    uploaded_files: list[str] | None = None,
    session_id: str = "",
    max_iterations: int = 2,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Ejecuta el agente Fénix RAG para una query del usuario.

    Returns:
        Dict con final_answer, sources, reflection, iteration_count,
        retrieval_strategy, doc_quality, grade_score, generation_mode,
        crag_retry_count, ingested_files.
    """
    graph = get_graph()
    state = initial_state(
        user_query=user_query,
        session_id=session_id,
        uploaded_files=uploaded_files,
        max_iterations=max_iterations,
    )

    invoke_config = config or {}
    if session_id and "configurable" not in invoke_config:
        invoke_config["configurable"] = {"thread_id": session_id}

    log.info(
        "agent_run_start",
        query=user_query[:80],
        session=session_id,
        files=len(uploaded_files or []),
    )

    try:
        final_state = graph.invoke(state, config=invoke_config)
    except Exception as exc:
        log.error("agent_run_failed", error=str(exc))
        return {
            "final_answer": "Ocurrió un error al procesar tu consulta. Por favor intenta de nuevo.",
            "sources": [],
            "error": str(exc),
            "iteration_count": 0,
        }

    answer = (
        final_state.get("final_answer")
        or final_state.get("draft_answer")
        or "No encontré información relevante para responder tu pregunta."
    )

    log.info(
        "agent_run_complete",
        answer_len=len(answer),
        iterations=final_state.get("iteration_count", 0),
        sources=len(final_state.get("sources", [])),
        doc_quality=final_state.get("doc_quality", ""),
        grade_score=final_state.get("grade_score", 0.0),
        generation_mode=final_state.get("generation_mode", ""),
        crag_retry_count=final_state.get("crag_retry_count", 0),
    )

    return {
        "final_answer": answer,
        "sources": final_state.get("sources", []),
        "reflection": final_state.get("reflection"),
        "iteration_count": final_state.get("iteration_count", 0),
        "retrieval_strategy": final_state.get("retrieval_strategy", ""),
        "doc_quality": final_state.get("doc_quality", ""),
        "grade_score": final_state.get("grade_score", 0.0),
        "generation_mode": final_state.get("generation_mode", ""),
        "crag_retry_count": final_state.get("crag_retry_count", 0),
        "ingested_files": [
            p["source_path"] for p in final_state.get("ingestion_plans", [])
        ],
    }


async def arun_agent(
    user_query: str,
    uploaded_files: list[str] | None = None,
    session_id: str = "",
    max_iterations: int = 2,
) -> dict[str, Any]:
    """Versión async de run_agent para uso en FastAPI / endpoints async."""
    graph = get_graph()
    state = initial_state(
        user_query=user_query,
        session_id=session_id,
        uploaded_files=uploaded_files,
        max_iterations=max_iterations,
    )
    config = {"configurable": {"thread_id": session_id}} if session_id else {}

    try:
        final_state = await graph.ainvoke(state, config=config)
    except Exception as exc:
        log.error("agent_arun_failed", error=str(exc))
        return {"final_answer": "Error procesando la consulta.", "sources": [], "error": str(exc)}

    answer = (
        final_state.get("final_answer")
        or final_state.get("draft_answer")
        or "No encontré información relevante."
    )
    return {
        "final_answer": answer,
        "sources": final_state.get("sources", []),
        "reflection": final_state.get("reflection"),
        "iteration_count": final_state.get("iteration_count", 0),
        "doc_quality": final_state.get("doc_quality", ""),
        "grade_score": final_state.get("grade_score", 0.0),
        "generation_mode": final_state.get("generation_mode", ""),
    }
