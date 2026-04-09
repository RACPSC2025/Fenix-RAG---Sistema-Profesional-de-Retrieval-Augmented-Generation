"""Tests del ParentChildRetriever — busca en hijos, retorna padres."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document
from langchain.storage import InMemoryByteStore

from src.retrieval.parent_child import (
    ParentChildRetriever,
    get_parent_child_retriever,
)


# ─── ParentChildRetriever ─────────────────────────────────────────────────

class TestParentChildRetriever:
    """Tests del retrieval parent-child."""

    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.is_initialized = True
        return store

    @pytest.fixture
    def parent_store(self):
        return InMemoryByteStore()

    @pytest.fixture
    def retriever(self, mock_vector_store, parent_store):
        return ParentChildRetriever(
            child_vector_store=mock_vector_store,
            parent_store=parent_store,
            top_k=5,
        )

    def test_retriever_type(self, retriever):
        assert retriever.retriever_type == "parent_child"

    def test_is_ready(self, retriever):
        assert retriever.is_ready() is True

    def test_add_documents_stores_parents_and_children(self, retriever):
        parent = Document(
            page_content="Full article content",
            metadata={"doc_id": "parent-1"},
        )
        child = Document(
            page_content="Child chunk",
            metadata={"parent_id": "parent-1", "chunk_index": 0},
        )

        result = retriever.add_documents([parent], [child])

        assert result["parents_stored"] == 1
        assert result["children_indexed"] == 1
        assert "parent-1" in retriever._child_to_parent

    def test_retrieve_returns_parent_when_child_matches(self, retriever):
        parent = Document(
            page_content="Full article content",
            metadata={"doc_id": "parent-1"},
        )
        retriever._parent_store.mset([("parent-1", parent)])
        retriever._child_to_parent["test.pdf::0"] = "parent-1"

        # Mock: vector store returns child
        retriever._child_store.similarity_search.return_value = [
            Document(
                page_content="Child chunk",
                metadata={"source": "test.pdf", "chunk_index": 0},
            ),
        ]

        from src.retrieval.base import RetrievalQuery
        result = retriever._retrieve(RetrievalQuery(text="query"))

        assert len(result) == 1
        assert result[0].page_content == "Full article content"
        assert result[0].metadata["parent_child"] is True

    def test_retrieve_deduplicates_same_parent(self, retriever):
        parent = Document(
            page_content="Full article",
            metadata={"doc_id": "parent-1"},
        )
        retriever._parent_store.mset([("parent-1", parent)])
        retriever._child_to_parent["test.pdf::0"] = "parent-1"
        retriever._child_to_parent["test.pdf::1"] = "parent-1"

        retriever._child_store.similarity_search.return_value = [
            Document(
                page_content="Child 0",
                metadata={"source": "test.pdf", "chunk_index": 0},
            ),
            Document(
                page_content="Child 1",
                metadata={"source": "test.pdf", "chunk_index": 1},
            ),
        ]

        from src.retrieval.base import RetrievalQuery
        result = retriever._retrieve(RetrievalQuery(text="query"))

        # Both children point to same parent → should deduplicate
        assert len(result) == 1

    def test_retrieve_returns_empty_when_no_children(self, retriever):
        retriever._child_store.similarity_search.return_value = []

        from src.retrieval.base import RetrievalQuery
        result = retriever._retrieve(RetrievalQuery(text="query"))

        assert result == []

    def test_retrieve_handles_missing_parent_id(self, retriever):
        """Si un hijo no tiene parent_id, intenta match por metadata."""
        retriever._child_store.similarity_search.return_value = [
            Document(
                page_content="Child",
                metadata={"source": "test.pdf", "chunk_index": 0},
            ),
        ]

        from src.retrieval.base import RetrievalQuery
        result = retriever._retrieve(RetrievalQuery(text="query"))

        # No parent found → returns empty
        assert result == []

    def test_find_parent_by_metadata(self, retriever):
        parent = Document(
            page_content="Parent",
            metadata={
                "doc_id": "parent-1",
                "article_number": "2.2.4.6.1",
                "source": "test.pdf",
            },
        )
        child = Document(
            page_content="Child",
            metadata={
                "article_number": "2.2.4.6.1",
                "source": "test.pdf",
                "chunk_index": 0,
            },
        )

        result = retriever._find_parent_by_metadata(child, [parent])
        assert result == "parent-1"

    def test_doc_id_from_metadata(self, retriever):
        doc = Document(
            page_content="test",
            metadata={"doc_id": "custom-id-123"},
        )
        assert retriever._doc_id(doc) == "custom-id-123"

    def test_doc_id_from_source_chunk(self, retriever):
        doc = Document(
            page_content="test",
            metadata={"source": "test.pdf", "chunk_index": 5},
        )
        assert retriever._doc_id(doc) == "test.pdf::chunk_5"

    def test_doc_id_from_content_hash(self, retriever):
        doc = Document(
            page_content="unique content for hashing",
            metadata={},
        )
        result = retriever._doc_id(doc)
        assert isinstance(result, str)
        assert len(result) > 0
