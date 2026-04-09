"""Tests del rethinking generation — two-pass answer generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from src.agent.skills.rethinking import (
    generate_with_rethinking,
    rethinking_generation_node,
    _build_context,
    _extract_sources,
)


# ─── generate_with_rethinking ──────────────────────────────────────────────

class TestGenerateWithRethinking:
    """Tests de la generación de dos pasadas."""

    def test_empty_docs_returns_fallback(self):
        answer, sources = generate_with_rethinking("test query", [])
        assert "No se encontraron documentos" in answer
        assert sources == []

    @patch("src.agent.skills.rethinking.get_llm")
    def test_two_pass_generation(self, mock_get_llm):
        mock_llm = MagicMock()
        # Pass 1: identification
        mock_llm.invoke.side_effect = [
            MagicMock(content="Key passage found in doc 1"),  # Pass 1
            MagicMock(content="Final answer based on passage"),  # Pass 2
        ]
        mock_get_llm.return_value = mock_llm

        docs = [
            Document(
                page_content="Important content about the topic",
                metadata={"source": "doc.pdf", "chunk_index": 0},
            ),
        ]

        answer, sources = generate_with_rethinking("test query", docs)

        assert mock_llm.invoke.call_count == 2
        assert answer == "Final answer based on passage"

    @patch("src.agent.skills.rethinking.get_llm")
    def test_handles_llm_error_gracefully(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm

        docs = [
            Document(
                page_content="content",
                metadata={"source": "doc.pdf"},
            ),
        ]

        with pytest.raises(Exception):
            generate_with_rethinking("test query", docs)


# ─── _build_context ───────────────────────────────────────────────────────

class TestBuildContext:
    """Tests de construcción del contexto."""

    def test_builds_context_with_metadata(self):
        docs = [
            Document(
                page_content="Content 1",
                metadata={
                    "source": "doc.pdf",
                    "chunk_index": 0,
                    "article_number": "5",
                },
            ),
            Document(
                page_content="Content 2",
                metadata={
                    "source": "doc.pdf",
                    "chunk_index": 1,
                    "section_number": "3.2",
                },
            ),
        ]

        context = _build_context(docs)

        assert "Content 1" in context
        assert "Content 2" in context
        assert "Artículo: 5" in context
        assert "Sección: 3.2" in context

    def test_handles_empty_docs(self):
        assert _build_context([]) == ""

    def test_handles_missing_metadata(self):
        docs = [
            Document(
                page_content="Content",
                metadata={},
            ),
        ]

        context = _build_context(docs)
        assert "Content" in context


# ─── _extract_sources ─────────────────────────────────────────────────────

class TestExtractSources:
    """Tests de extracción de fuentes."""

    def test_extracts_article_references(self):
        answer = "According to Article 5, the requirements are..."
        docs = [Document(page_content="content", metadata={})]

        sources = _extract_sources(answer, docs)
        assert any(s.get("type") == "article" for s in sources)

    def test_extracts_section_references(self):
        answer = "As stated in Section 3.2, the process is..."
        docs = [Document(page_content="content", metadata={})]

        sources = _extract_sources(answer, docs)
        assert any(s.get("type") == "section" for s in sources)

    def test_falls_back_to_doc_metadata_when_no_explicit_refs(self):
        answer = "This is a general answer without specific references."
        docs = [
            Document(
                page_content="content",
                metadata={"source": "doc.pdf", "article_number": "5"},
            ),
        ]

        sources = _extract_sources(answer, docs)
        assert len(sources) > 0

    def test_deduplicates_sources(self):
        answer = "Article 5 says X. As Article 5 also states..."
        docs = [Document(page_content="content", metadata={})]

        sources = _extract_sources(answer, docs)
        # Should deduplicate
        article_refs = [s for s in sources if s.get("type") == "article"]
        assert len(article_refs) <= 1


# ─── rethinking_generation_node ───────────────────────────────────────────

class TestRethinkingGenerationNode:
    """Tests del nodo de generación con rethinking."""

    @patch("src.agent.skills.rethinking.generate_with_rethinking")
    def test_node_returns_answer_and_sources(self, mock_rethinking):
        mock_rethinking.return_value = (
            "Answer text",
            [{"type": "article", "value": "5"}],
        )

        state = {
            "user_query": "test",
            "retrieval_results": [Document(page_content="content")],
        }

        result = rethinking_generation_node(state)

        assert result["draft_answer"] == "Answer text"
        assert result["sources"] == [{"type": "article", "value": "5"}]

    def test_node_handles_empty_docs(self):
        state = {
            "user_query": "test",
            "retrieval_results": [],
        }

        result = rethinking_generation_node(state)

        assert "No encontré documentos" in result["draft_answer"]
        assert result["sources"] == []
