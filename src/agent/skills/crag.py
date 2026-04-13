"""
CRAG — Corrective RAG.

Evalúa la calidad de los documentos recuperados y decide:
  - CORRECT  (>0.7): documentos relevantes → generar respuesta directamente
  - AMBIGUOUS (0.3-0.7): parcialmente relevantes → re-retrieve con query reformulada
  - INCORRECT (<0.3): no relevantes → descartar y re-retrieve con step-back query

Sin dependencia de búsqueda web: cuando los docs son insuficientes,
se reescribe la query (rewriting o step-back) y se reintenta el retrieval.

CONTRATO DE ESTADO:
  Escribe → `crag_route`, `crag_retry_count`, `doc_quality`, `grade_score`
  NO escribe → `route` (ese campo es de document_router y supervisor)
  route_after_grading lee `crag_route`, NO `route`

PROTECCIÓN ANTI-LOOP:
  MAX_CRAG_RETRIES = 2. Si crag_retry_count >= MAX_CRAG_RETRIES, fuerza
  crag_route = "generation" con los docs actuales, aunque sean subóptimos.
  Esto garantiza que el grafo siempre termina.

Uso en el grafo:
    from src.agent.skills.crag import grade_documents_node, route_after_grading

    builder.add_edge("retrieval", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grading,
        {"generation": "generation", "retrieval": "retrieval"},
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from src.config.logging import get_logger
from src.config.providers import get_llm

log = get_logger(__name__)

# Máximo de reintentos antes de forzar generation con los docs actuales.
MAX_CRAG_RETRIES: int = 2


# ─── Structured output para el grader ────────────────────────────────────────

class DocumentGrade(BaseModel):
    """Resultado del grading de documentos recuperados."""

    quality: str = Field(description="correct | ambiguous | incorrect")
    score: float = Field(ge=0.0, le=1.0, description="Relevancia general (0-1).")
    reasoning: str = Field(description="Breve justificación del score.")


# ─── Prompts ─────────────────────────────────────────────────────────────────

GRADER_SYSTEM = (
    "Eres un evaluador experto de documentos para sistemas RAG.\n"
    "Tu tarea es determinar si los documentos recuperados son suficientes "
    "para responder la consulta del usuario.\n\n"
    "Criterios:\n"
    "- **correct** (score > 0.7): documentos directamente relevantes y completos.\n"
    "- **ambiguous** (score 0.3-0.7): parcialmente relevantes, información incompleta.\n"
    "- **incorrect** (score < 0.3): no relevantes o fuera de contexto.\n\n"
    "Retorna SOLO el JSON con quality, score y reasoning."
)


# ─── Función principal de grading ────────────────────────────────────────────

def grade_documents(
    query: str,
    documents: list,
) -> DocumentGrade:
    """
    Evalúa la calidad de los documentos recuperados.

    Args:
        query: Consulta original del usuario.
        documents: Lista de Documents recuperados.

    Returns:
        DocumentGrade con quality, score y reasoning.
    """
    if not documents:
        return DocumentGrade(
            quality="incorrect",
            score=0.0,
            reasoning="No se recuperaron documentos.",
        )

    docs_summary = "\n---\n".join(
        f"[Doc {i + 1}] (source: {doc.metadata.get('source', '?')}, "
        f"chunk: {doc.metadata.get('chunk_index', '?')})\n"
        f"{doc.page_content[:500]}"
        for i, doc in enumerate(documents[:5])
    )

    llm = get_llm(temperature=0)
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_SYSTEM),
        ("human", "{input}"),
    ])
    chain = grader_prompt | llm.with_structured_output(DocumentGrade)

    try:
        grade: DocumentGrade = chain.invoke({
            "input": f"Consulta: {query}\n\nDocumentos recuperados:\n{docs_summary}"
        })
    except Exception as exc:
        log.warning("crag_grading_failed", error=str(exc))
        return DocumentGrade(
            quality="ambiguous",
            score=0.5,
            reasoning=f"Error en grading: {exc}. Reintentando.",
        )

    log.info(
        "crag_grade_complete",
        quality=grade.quality,
        score=grade.score,
        docs_count=len(documents),
    )
    return grade


# ─── Query rewriting ─────────────────────────────────────────────────────────

def rewrite_query_for_reretrieval(
    query: str,
    grade: DocumentGrade,
    documents: list | None = None,
) -> str:
    """
    Reescribe la query para un nuevo intento de retrieval.

    - ambiguous: reformula con más detalle técnico.
    - incorrect: genera step-back query (más general).
    """
    llm = get_llm(temperature=0)

    if grade.quality == "ambiguous":
        prompt = (
            "Los documentos recuperados son parcialmente relevantes. "
            "Reformula la siguiente consulta para ser más específica y técnica.\n\n"
            f"Consulta original: {query}\n\n"
            "Consulta reformulada (SOLO la consulta, sin explicaciones):"
        )
    else:
        prompt = (
            "Los documentos recuperados no son relevantes. "
            "Genera una consulta más amplia y general sobre el tema de fondo.\n\n"
            f"Consulta específica: {query}\n\n"
            "Consulta general (SOLO la consulta, sin explicaciones):"
        )

    try:
        response = llm.invoke(prompt)
        result = response.content.strip()
        log.debug(
            "crag_query_rewritten",
            original=query[:60],
            rewritten=result[:60],
            quality=grade.quality,
        )
        return result
    except Exception as exc:
        log.warning("crag_rewrite_failed", error=str(exc))
        return query


# ─── Nodo del grafo ───────────────────────────────────────────────────────────

def grade_documents_node(state: dict) -> dict:
    """
    Nodo de grading para el grafo LangGraph.

    CONTRATO:
      Escribe `crag_route` (no `route`) para no colisionar con reflection_node.
      Incrementa `crag_retry_count` en cada re-retrieval.
      Si crag_retry_count >= MAX_CRAG_RETRIES, fuerza generation.

    Args:
        state: AgentState con active_query, retrieval_results, crag_retry_count.

    Returns:
        Dict con doc_quality, grade_score, crag_route, crag_retry_count,
        y opcionalmente active_query reescrita.
    """
    query = state.get("active_query") or state.get("user_query", "")
    docs = state.get("retrieval_results", [])
    retry_count = state.get("crag_retry_count", 0)

    grade = grade_documents(query, docs)

    result: dict[str, object] = {
        "doc_quality": grade.quality,
        "grade_score": grade.score,
    }

    if grade.quality == "correct":
        result["crag_route"] = "generation"
        log.info(
            "crag_decision",
            decision="generation",
            quality=grade.quality,
            score=grade.score,
            retry_count=retry_count,
        )

    elif retry_count >= MAX_CRAG_RETRIES:
        # Reintentos agotados → forzar generation con lo que hay.
        # Es preferible responder con docs imperfectos que loopearse.
        result["crag_route"] = "generation"
        log.warning(
            "crag_max_retries_reached",
            retry_count=retry_count,
            max_retries=MAX_CRAG_RETRIES,
            quality=grade.quality,
            score=grade.score,
            action="forcing_generation_with_current_docs",
        )

    else:
        # Ambiguous o incorrect con retries disponibles → reescribir y reintentar
        rewritten = rewrite_query_for_reretrieval(query, grade, docs)
        result["active_query"] = rewritten
        result["crag_route"] = "retrieval"
        result["crag_retry_count"] = retry_count + 1
        log.info(
            "crag_decision",
            decision="retrieval_retry",
            quality=grade.quality,
            score=grade.score,
            retry_count=retry_count + 1,
            rewritten_query=rewritten[:60],
        )

    return result


# ─── Edge condicional ─────────────────────────────────────────────────────────

def route_after_grading(state: dict) -> str:
    """
    Edge condicional post-grading.

    Lee `crag_route` — campo exclusivo del CRAG grader.
    NO lee `route` para evitar colisión con reflection_node y supervisor.

    Returns:
        "generation" | "retrieval"
    """
    return state.get("crag_route", "generation")
