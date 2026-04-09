"""Tests del ContextEnrichmentWindow — retrieval con chunks vecinos."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from src.retrieval.context_enricher import (
    ContextEnrichmentWindow,
    EnrichedRetriever,
    get_context_enricher,
)


# ─── ContextEnrichmentWindow ────────────────────────────────────────────────

class TestContextEnrichmentWindow:
    """Tests del enriquecimiento de contexto."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock del vector store."""
        store = MagicMock()
        store.is_initialized = True
        return store

    @pytest.fixture
    def enricher(self, mock_vector_store):
        return ContextEnrichmentWindow(
            vector_store=mock_vector_store,
            window_size=2,
        )

    def test_enrich_empty_docs(self, enricher):
        assert enricher.enrich([]) == []

    def test_enrich_single_doc_no_neighbors(self, enricher):
        doc = Document(
            page_content="test content",
            metadata={"source": "test.pdf", "chunk_index": 0},
        )
        result = enricher.enrich([doc])
        assert len(result) == 1
        # No neighbors found → retorna el original
        assert result[0].page_content == "test content"

    def test_enrich_with_neighbors(self, enricher):
        docs = [
            Document(
                page_content="chunk 0",
                metadata={"source": "test.pdf", "chunk_index": 0},
            ),
        ]

        # Mock: vector store returns neighbors
        enricher._vector_store.get.return_value = {
            "documents": ["chunk 0", "chunk 1", "chunk 2"],
            "metadatas": [
                {"source": "test.pdf", "chunk_index": 0},
                {"source": "test.pdf", "chunk_index": 1},
                {"source": "test.pdf", "chunk_index": 2},
            ],
        }

        result = enricher.enrich(docs)
        assert len(result) == 1
        assert result[0].metadata["context_enriched"] is True
        assert result[0].metadata["context_chunks_count"] == 3

    def test_overlap_detection(self, enricher):
        """Detecta y elimina overlap entre chunks."""
        text_a = "This is some text that ends here"
        text_b = "that ends hereand continues with new content"

        result = enricher._detect_and_remove_overlap(text_a, text_b)
        assert result is not None
        assert "that ends here" in result
        # Should not duplicate the overlap
        assert result.count("that ends here") == 1

    def test_no_overlap_returns_none(self, enricher):
        """Si no hay overlap, retorna None."""
        text_a = "completely different text"
        text_b = "another unrelated text"
        assert enricher._detect_and_remove_overlap(text_a, text_b) is None

    def test_dedup_by_source_chunk_index(self, enricher):
        """No duplica documentos con mismo source::chunk_index."""
        doc1 = Document(
            page_content="content",
            metadata={"source": "test.pdf", "chunk_index": 0},
        )
        doc2 = Document(
            page_content="content",
            metadata={"source": "test.pdf", "chunk_index": 0},  # mismo ID
        )

        result = enricher.enrich([doc1, doc2])
        # Should deduplicate
        assert len(result) == 1

    def test_fallback_on_vector_store_error(self, enricher):
        """Si el vector store falla, retorna el doc original."""
        enricher._vector_store.get.side_effect = Exception("DB error")

        doc = Document(
            page_content="content",
            metadata={"source": "test.pdf", "chunk_index": 0},
        )
        result = enricher.enrich([doc])
        assert len(result) == 1
        assert result[0].page_content == "content"

    def test_concatenate_single_doc(self, enricher):
        """Un solo chunk se retorna tal cual."""
        docs = [Document(page_content="single chunk", metadata={})]
        result = enricher._concatenate_with_dedup(docs)
        assert result == "single chunk"

    def test_concatenate_with_separator(self, enricher):
        """Chunks sin overlap se concatenan con separador."""
        docs = [
            Document(page_content="first", metadata={}),
            Document(page_content="second", metadata={}),
        ]
        result = enricher._concatenate_with_dedup(docs)
        assert "first" in result
        assert "second" in result


# ─── EnrichedRetriever ──────────────────────────────────────────────────────

class TestEnrichedRetriever:
    """Tests del wrapper EnrichedRetriever."""

    @pytest.fixture
    def mock_base_retriever(self):
        retriever = MagicMock()
        result = MagicMock()
        result.documents = [Document(page_content="test", metadata={"source": "a", "chunk_index": 0})]
        retriever.retrieve.return_value = result
        return retriever

    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.is_initialized = True
        return store

    def test_enriched_retriever_returns_enriched_docs(self, mock_base_retriever, mock_vector_store):
        enriched = EnrichedRetriever(mock_base_retriever, mock_vector_store)
        result = enriched.retrieve("query")
        # Debería llamar al base retriever
        mock_base_retriever.retrieve.assert_called_once()
