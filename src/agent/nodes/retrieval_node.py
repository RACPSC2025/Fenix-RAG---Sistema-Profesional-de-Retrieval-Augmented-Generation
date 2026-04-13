"""
Retrieval node — Multi-Query Fusion + Context Enrichment.

FIX APLICADO:
  Versión anterior usaba `timer.update()` y `timer.to_state()` sin el
  context manager `with node_timer(state, "retrieval") as timer:`.
  Esto causaba NameError en runtime → el nodo crasheaba en cada invocación
  → LangGraph lo reintentaba 3 veces (retry=3) → fallaba las 3 → el grafo
  terminaba sin retrieval_results. Bug silencioso de alto impacto.
"""

from __future__ import annotations

from src.agent.metrics import node_timer
from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def retrieval_node(state: AgentState) -> dict:
    """
    Ejecuta EnsembleRetriever con Multi-Query Fusion + Context Enrichment.

    1. QueryTransformer genera variantes: [original, rewritten, step-back]
    2. Retrieval con cada variante del query
    3. Deduplicación por source::chunk_index
    4. Context Enrichment aplicado por el ensemble
    """
    from src.agent.skills.query_transformer import QueryTransformer  # noqa: PLC0415
    from src.retrieval.base import RetrievalQuery  # noqa: PLC0415
    from src.retrieval.ensemble import get_ensemble_retriever  # noqa: PLC0415
    from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

    with node_timer(state, "retrieval") as timer:
        query_text = state.get("active_query") or state.get("user_query", "")

        if not query_text:
            timer.update(docs_count=0, extra={"reason": "empty_query"})
            return {
                "retrieval_results": [],
                "retrieval_strategy": "none",
                **timer.to_state(),
            }

        vs = get_vector_store()
        if not vs.is_initialized:
            vs.open_or_create()

        ensemble = get_ensemble_retriever(
            vector_store=vs,
            use_context_enrichment=True,
            context_window_size=2,
        )

        # Paso 1: QueryTransformer genera variantes
        transformer = QueryTransformer()
        variants = transformer.transform_all(query_text)

        log.debug(
            "retrieval_query_variants",
            original=query_text[:60],
            variants=len(variants),
            queries=[v[:60] for v in variants],
        )

        # Paso 2: Retrieval con cada variante + deduplicación
        seen_ids: set[str] = set()
        all_docs = []
        total_before_dedup = 0

        for variant in variants:
            result = ensemble.retrieve(RetrievalQuery(text=variant))
            total_before_dedup += len(result.documents)
            for doc in result.documents:
                doc_id = (
                    f"{doc.metadata.get('source', '')}"
                    f"::{doc.metadata.get('chunk_index', '')}"
                )
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        timer.update(
            docs_count=len(all_docs),
            extra={
                "variants": len(variants),
                "total_before_dedup": total_before_dedup,
                "dedup_removed": total_before_dedup - len(all_docs),
                "query": query_text[:80],
            },
        )

        log.info(
            "retrieval_complete",
            query=query_text[:60],
            docs_retrieved=len(all_docs),
            variants=len(variants),
            dedup_removed=total_before_dedup - len(all_docs),
        )

        return {
            "retrieval_results": all_docs,
            "retrieval_strategy": "multi_query_fusion",
            **timer.to_state(),
        }
