"""
API Routes — Admin.

Endpoints de métricas, analytics y estado del sistema.
Solo para uso interno / dashboard de administración.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import DB
from src.api.schemas import (
    ErrorResponse,
    QualityMetricsResponse,
    StrategyPerformanceItem,
    TopQueryItem,
)
from src.config.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Métricas de calidad ──────────────────────────────────────────────────────

@router.get(
    "/metrics/quality",
    response_model=QualityMetricsResponse,
    responses={500: {"model": ErrorResponse}},
)
async def get_quality_metrics(
    days: int = Query(default=7, ge=1, le=90, description="Días a analizar"),
    db: AsyncSession = DB,
):
    """
    Métricas de calidad del retrieval para los últimos N días.

    - **avg_reflection_score**: Promedio de score de reflexión (0-1).
    - **avg_iterations**: Ciclos promedio de reflection.
    - **reformulated_pct**: Porcentaje de queries reformuladas.
    - **low_score_pct**: Porcentaje de respuestas con score < 0.7.
    """
    from src.persistence.repositories.query_repo import get_quality_metrics  # noqa: PLC0415

    metrics = await get_quality_metrics(db, days=days)

    return QualityMetricsResponse(
        total_queries=metrics.get("total_queries", 0),
        avg_reflection_score=metrics.get("avg_reflection_score", 0.0),
        avg_iterations=metrics.get("avg_iterations", 0.0),
        reformulated_pct=metrics.get("reformulated_pct", 0.0),
        low_score_pct=metrics.get("low_score_pct", 0.0),
        period_days=days,
    )


# ── Rendimiento por estrategia ───────────────────────────────────────────────

@router.get(
    "/metrics/strategies",
    response_model=list[StrategyPerformanceItem],
)
async def get_strategy_performance(
    db: AsyncSession = DB,
):
    """
    Rendimiento de cada estrategia de retrieval.

    Compara: vector, bm25, hybrid, hierarchical, full.
    """
    from src.persistence.repositories.query_repo import get_strategy_performance  # noqa: PLC0415

    results = await get_strategy_performance(db)

    return [
        StrategyPerformanceItem(
            strategy=r.get("strategy", ""),
            total_queries=r.get("total_queries", 0),
            avg_score=r.get("avg_score", 0.0),
            avg_time_ms=r.get("avg_time_ms", 0.0),
            avg_iterations=r.get("avg_iterations", 0.0),
        )
        for r in results
    ]


# ── Queries más frecuentes ───────────────────────────────────────────────────

@router.get(
    "/queries/top",
    response_model=list[TopQueryItem],
)
async def get_top_queries(
    limit: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = DB,
):
    """
    Queries más frecuentes del corpus.

    Útil para entender qué pregunta más el usuario y optimizar retrieval.
    """
    from src.persistence.repositories.query_repo import get_top_queries  # noqa: PLC0415

    results = await get_top_queries(db, limit=limit)

    return [
        TopQueryItem(
            query=r.get("query", ""),
            frequency=r.get("frequency", 0),
            avg_score=r.get("avg_score", 0.0),
        )
        for r in results
    ]


# ── Estado del vector store ──────────────────────────────────────────────────

@router.get(
    "/vector-store",
)
async def get_vector_store_status():
    """Estado del vector store ChromaDB."""
    from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

    vs = get_vector_store()

    return {
        "initialized": vs.is_initialized,
        "collection_name": vs.collection_name if hasattr(vs, "collection_name") else "unknown",
        "document_count": vs.count() if vs.is_initialized else 0,
        "persist_dir": str(vs.persist_dir) if hasattr(vs, "persist_dir") else "unknown",
    }
