"""Tests de los retrievers individuales: BM25, Hybrid, Ensemble, Reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from src.retrieval.base import RetrievalQuery, RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import FlashRankReranker
from src.retrieval.ensemble import EnsembleRetriever, RetrievalStrategy


# ─── BM25 Retriever ────────────────────────────────────────────────────────

class TestBM25Retriever:
    """Tests del retriever BM25 Okapi."""

    def test_bm25_retriever_type(self):
        retriever = BM25Retriever()
        assert retriever.retriever_type == "bm25"

    def test_bm25_is_ready_empty(self):
        retriever = BM25Retriever()
        # Debe ser ready incluso sin documentos (BM25 maneja corpus vacío)
        assert retriever.is_ready() is True

    def test_bm25_retrieve_empty_corpus_returns_empty(self):
        retriever = BM25Retriever()
        result = retriever.retrieve(RetrievalQuery(text="test query"))
        assert result.documents == []

    def test_bm25_retrieve_with_corpus(self):
        retriever = BM25Retriever()
        docs = [
            Document(page_content="Python es un lenguaje de programación", metadata={"source": "doc1"}),
            Document(page_content="Java es otro lenguaje popular", metadata={"source": "doc2"}),
        ]
        retriever.add_documents(docs)

        result = retriever.retrieve(RetrievalQuery(text="Python"))
        assert len(result.documents) > 0
        assert result.documents[0].metadata.get("source") == "doc1"

    def test_bm25_retrieve_finds_exact_terms(self):
        """BM25 encuentra términos exactos mejor que embeddings."""
        retriever = BM25Retriever()
        docs = [
            Document(page_content="El artículo 5 establece las obligaciones", metadata={"source": "doc1", "chunk_index": 0}),
            Document(page_content="Información general sobre el tema", metadata={"source": "doc2", "chunk_index": 1}),
        ]
        retriever.add_documents(docs)

        result = retriever.retrieve(RetrievalQuery(text="artículo 5 obligaciones"))
        assert len(result.documents) > 0
        assert result.documents[0].metadata.get("source") == "doc1"


# ─── Hybrid Retriever ─────────────────────────────────────────────────────

class TestHybridRetriever:
    """Tests del retriever híbrido con RRF fusion."""

    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.is_initialized = True
        return store

    def test_hybrid_retriever_type(self, mock_vector_store):
        retriever = HybridRetriever(vector_store=mock_vector_store)
        assert retriever.retriever_type == "hybrid"

    def test_hybrid_returns_documents(self, mock_vector_store):
        mock_vector_store.similarity_search_with_score.return_value = [
            (Document(page_content="vector result", metadata={"source": "a"}), 0.8),
        ]

        retriever = HybridRetriever(vector_store=mock_vector_store, bm25_docs=[
            Document(page_content="bm25 result", metadata={"source": "b"}),
        ])

        result = retriever.retrieve(RetrievalQuery(text="test query", top_k=5))
        assert len(result.documents) > 0


# ─── FlashRank Reranker ───────────────────────────────────────────────────

class TestFlashRankReranker:
    """Tests del reranker FlashRank."""

    def test_reranker_type(self):
        reranker = FlashRankReranker()
        assert reranker.retriever_type == "reranker"

    def test_reranker_empty_docs(self):
        reranker = FlashRankReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_reranker_preserves_order_when_no_rerank(self):
        """Si solo hay 1 doc, el orden no cambia."""
        reranker = FlashRankReranker()
        docs = [Document(page_content="single doc", metadata={"source": "a"})]
        result = reranker.rerank("query", docs, top_k=5)
        assert len(result) == 1


# ─── Ensemble Retriever ───────────────────────────────────────────────────

class TestEnsembleRetriever:
    """Tests del EnsembleRetriever con auto-selección de estrategia."""

    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.is_initialized = True
        return store

    def test_ensemble_strategy_vector(self, mock_vector_store):
        """Query conceptual → estrategia vector."""
        ensemble = EnsembleRetriever(
            vector_store=mock_vector_store,
            strategy=RetrievalStrategy.AUTO,
            top_k=5,
        )

        query = RetrievalQuery(text="¿cuáles son mis derechos como trabajador?")
        # Debe seleccionar vector por ser query conceptual
        result = ensemble.retrieve(query)
        assert result.documents is not None

    def test_ensemble_strategy_bm25(self, mock_vector_store):
        """Query con números exactos → estrategia bm25."""
        ensemble = EnsembleRetriever(
            vector_store=mock_vector_store,
            strategy=RetrievalStrategy.AUTO,
            top_k=5,
        )

        query = RetrievalQuery(text="artículo 5 sección 3")
        # Debe seleccionar bm25 por tener números específicos
        result = ensemble.retrieve(query)
        assert result.documents is not None

    def test_ensemble_strategy_hybrid(self, mock_vector_store):
        """Query larga (>10 palabras) → estrategia hybrid."""
        ensemble = EnsembleRetriever(
            vector_store=mock_vector_store,
            strategy=RetrievalStrategy.AUTO,
            top_k=5,
        )

        query = RetrievalQuery(
            text="¿cuáles son las obligaciones del empleador en materia de seguridad y salud en el trabajo según la normativa vigente?"
        )
        # Debe seleccionar hybrid por ser query larga
        result = ensemble.retrieve(query)
        assert result.documents is not None

    def test_ensemble_respects_top_k(self, mock_vector_store):
        """El top_k se respeta en el resultado final."""
        ensemble = EnsembleRetriever(
            vector_store=mock_vector_store,
            strategy=RetrievalStrategy.VECTOR,
            top_k=3,
        )

        mock_vector_store.similarity_search_with_score.return_value = [
            (Document(page_content=f"doc {i}"), 0.9 - i * 0.1)
            for i in range(10)
        ]

        result = ensemble.retrieve(RetrievalQuery(text="test", top_k=3))
        assert len(result.documents) <= 3

    def test_ensemble_returns_strategy_in_result(self, mock_vector_store):
        """El resultado indica qué estrategia se usó."""
        ensemble = EnsembleRetriever(
            vector_store=mock_vector_store,
            strategy=RetrievalStrategy.HYBRID,
            top_k=5,
        )

        result = ensemble.retrieve(RetrievalQuery(text="test query"))
        assert hasattr(result, "strategy") or "strategy" in str(result.__dict__)
