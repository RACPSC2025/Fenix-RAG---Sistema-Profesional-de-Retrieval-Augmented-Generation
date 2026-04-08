"""
Nodos individuales del grafo LangGraph — implementaciones reales.

Cada nodo es una función pura que recibe AgentState y retorna un dict
con las actualizaciones al estado.
"""

from __future__ import annotations

from typing import Any

from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def document_router_node(state: AgentState) -> dict:
    """Clasifica archivos subidos y decide ruta de ingestión."""
    from src.agent.skills.document_classifier import DocumentClassifierSkill  # noqa: PLC0415

    uploaded = state.get("uploaded_files", [])
    if not uploaded:
        return {"route": "retrieval", "ingestion_plans": []}

    classifier = DocumentClassifierSkill()
    plans = []

    for fpath in uploaded:
        plan = classifier.classify(fpath)
        plans.append(plan)

    log.info(
        "document_router_complete",
        files=len(uploaded),
        plans=[p.get("loader_type", "?") for p in plans],
    )

    return {
        "route": "ingestion",
        "ingestion_plans": plans,
    }


def ingestion_node(state: AgentState) -> dict:
    """Ejecuta IngestionPipeline e indexa chunks a Chroma."""
    from src.ingestion.pipeline import IngestionPipeline  # noqa: PLC0415

    plans = state.get("ingestion_plans", [])
    if not plans:
        return {"error": "No hay planes de ingestión", "ingested_documents": []}

    pipeline = IngestionPipeline()
    ingested = []

    for plan in plans:
        source_path = plan.get("source_path", "")
        if not source_path:
            continue

        result = pipeline.ingest_file(source_path)
        ingested.append({
            "source_path": source_path,
            "success": result.success,
            "chunk_count": result.chunk_count,
            "page_count": result.page_count,
            "loader_used": result.loader_used,
            "errors": result.errors,
        })

        if result.success:
            log.info(
                "ingestion_success",
                file=source_path,
                chunks=result.chunk_count,
                pages=result.page_count,
            )
        else:
            log.error(
                "ingestion_failed",
                file=source_path,
                errors=result.errors,
            )

    return {
        "ingested_documents": ingested,
        "error": None if any(i["success"] for i in ingested) else "Todos los archivos fallaron",
    }


def retrieval_node(state: AgentState) -> dict:
    """Ejecuta EnsembleRetriever con context enrichment."""
    from src.retrieval.base import RetrievalQuery  # noqa: PLC0415
    from src.retrieval.ensemble import get_ensemble_retriever  # noqa: PLC0415
    from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

    query_text = state.get("active_query") or state.get("user_query", "")
    if not query_text:
        return {"retrieval_results": [], "retrieval_strategy": "none"}

    vs = get_vector_store()
    if not vs.is_initialized:
        vs.open_or_create()

    # Ensemble con context enrichment habilitado
    ensemble = get_ensemble_retriever(
        vector_store=vs,
        use_context_enrichment=True,
        context_window_size=2,
    )

    query = RetrievalQuery(text=query_text)
    result = ensemble.retrieve(query)

    log.info(
        "retrieval_complete",
        query=query_text[:60],
        docs=len(result.documents),
        strategy=result.strategy if hasattr(result, "strategy") else "auto",
    )

    return {
        "retrieval_results": result.documents,
        "retrieval_strategy": result.strategy if hasattr(result, "strategy") else "auto",
    }


def generation_node(state: AgentState) -> dict:
    """Genera respuesta con contexto de retrieval_results."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: PLC0415
    from src.config.providers import get_llm  # noqa: PLC0415

    query = state.get("active_query") or state.get("user_query", "")
    docs = state.get("retrieval_results", [])

    if not docs:
        return {
            "draft_answer": "No encontré documentos relevantes para responder tu consulta.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'N/A')} | Chunk: {d.metadata.get('chunk_index', '?')}]\n{d.page_content}"
        for d in docs
    )

    from src.agent.prompts.system import GENERATION_PROMPT  # noqa: PLC0415

    llm = get_llm()
    messages = [
        SystemMessage(content=GENERATION_PROMPT),
        HumanMessage(content=f"Consulta: {query}\n\nContexto:\n{context}"),
    ]

    response = llm.invoke(messages)

    # Extraer fuentes de los documentos
    sources = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source", ""),
            "article": d.metadata.get("article_number", ""),
            "page": str(d.metadata.get("page", "")),
        })

    return {
        "draft_answer": response.content,
        "sources": sources,
    }


def reflection_node(state: AgentState) -> dict:
    """Auto-evaluación de la respuesta generada."""
    from src.agent.skills.answer_validator import AnswerValidatorSkill  # noqa: PLC0415
    from src.agent.state import ReflectionOutput  # noqa: PLC0415

    draft = state.get("draft_answer", "")
    query = state.get("user_query", "")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 2)

    # Validación rule-based primero (sin costo de LLM)
    validator = AnswerValidatorSkill()
    validation = validator.validate(draft, query)

    if validation.is_valid:
        return {
            "final_answer": draft,
            "reflection": ReflectionOutput(
                score=validation.score,
                is_grounded=True,
                has_hallucination=False,
                cites_source=validation.cites_source,
                feedback="Respuesta válida",
                reformulated_query="",
            ),
            "route": "END",
            "iteration_count": iteration + 1,
        }

    # Si no es válida y quedan iteraciones → reformular
    if iteration < max_iter:
        reformulated = validator.suggest_reformulation(query, draft)
        return {
            "active_query": reformulated,
            "reflection": ReflectionOutput(
                score=validation.score,
                is_grounded=False,
                has_hallucination=validation.has_hallucination,
                cites_source=validation.cites_source,
                feedback=validation.feedback,
                reformulated_query=reformulated,
            ),
            "route": "retrieval",
            "iteration_count": iteration + 1,
        }

    # Iteraciones agotadas → usar borrador con advertencia
    return {
        "final_answer": draft + "\n\n⚠️ Nota: Esta respuesta puede estar incompleta.",
        "reflection": ReflectionOutput(
            score=validation.score,
            is_grounded=False,
            has_hallucination=False,
            cites_source=validation.cites_source,
            feedback="Iteraciones agotadas",
            reformulated_query="",
        ),
        "route": "END",
        "iteration_count": iteration + 1,
    }


def supervisor_node(state: AgentState) -> dict:
    """Supervisor pattern: coordina subagentes especializados."""
    # Por ahora, solo valida la ruta — implementación completa en Fase futura
    return {"route": state.get("route", "retrieval")}


__all__ = [
    "document_router_node",
    "ingestion_node",
    "retrieval_node",
    "generation_node",
    "reflection_node",
    "supervisor_node",
]
