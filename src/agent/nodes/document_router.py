"""
Document router node — clasifica archivos subidos y decide ruta de ingestión.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def document_router_node(state: AgentState) -> dict:
    """
    Clasifica archivos subidos y decide ruta de ingestión.

    Si no hay archivos, retorna ruta directa a retrieval.
    Si hay archivos, clasifica cada uno y retorna planes de ingestión.
    """
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
