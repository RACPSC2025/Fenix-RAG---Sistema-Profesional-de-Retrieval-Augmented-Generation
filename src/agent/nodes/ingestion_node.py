"""
Ingestion node — ejecuta IngestionPipeline e indexa chunks a Chroma.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def ingestion_node(state: AgentState) -> dict:
    """
    Ejecuta IngestionPipeline e indexa chunks a Chroma.

    Procesa cada plan de ingestión generado por el document_router.
    Retorna resultados de ingestión por archivo.
    """
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
