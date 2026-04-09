"""Tests del AdaptiveChunker — chunking adaptativo por tipo de documento."""

from __future__ import annotations

import pytest

from langchain_core.documents import Document

from src.ingestion.processors.adaptive_chunker import (
    AdaptiveChunker,
    detect_document_type,
    DocumentTypeResult,
    DOCUMENT_SEPARATORS,
)


# ─── detect_document_type ────────────────────────────────────────────────

class TestDetectDocumentType:
    """Tests de detección de tipo de documento."""

    def test_detects_documentation(self):
        text = "This function takes a parameter and returns a value. The class implements the interface."
        result = detect_document_type(text)
        assert result.doc_type == "documentation"

    def test_detects_api_docs(self):
        text = "This API endpoint accepts a request body and returns a 200 response. The header must contain the API key."
        result = detect_document_type(text)
        assert result.doc_type == "api_docs"

    def test_detects_architecture(self):
        text = "The microservice architecture uses a component-based design. Each layer handles different responsibilities."
        result = detect_document_type(text)
        assert result.doc_type == "architecture"

    def test_detects_contract(self):
        text = "The parties agree to the following terms and conditions. Clause 1: The contractor shall..."
        result = detect_document_type(text)
        assert result.doc_type == "contract"

    def test_detects_policy(self):
        text = "This policy defines the procedure for access control. The responsible party shall ensure compliance."
        result = detect_document_type(text)
        assert result.doc_type == "policy"

    def test_returns_standard_when_no_signal(self):
        text = "This is a random text with no specific patterns."
        result = detect_document_type(text)
        assert result.doc_type == "standard"
        assert result.confidence == 0.0

    def test_confidence_between_0_and_1(self):
        text = "function test() { return value; }"
        result = detect_document_type(text)
        assert 0.0 <= result.confidence <= 1.0

    def test_indicators_populated(self):
        text = "This function takes a parameter and returns a value."
        result = detect_document_type(text)
        # Should have found some indicators
        if result.doc_type != "standard":
            assert len(result.indicators) > 0


# ─── DOCUMENT_SEPARATORS ──────────────────────────────────────────────────

class TestDocumentSeparators:
    """Tests de los separadores por tipo."""

    def test_all_types_have_separators(self):
        assert len(DOCUMENT_SEPARATORS) > 0
        for doc_type, separators in DOCUMENT_SEPARATORS.items():
            assert len(separators) > 0
            # All separator lists should end with ["", " "] for fallback
            assert separators[-1] == ""
            assert separators[-2] == " "

    def test_documentation_separators(self):
        seps = DOCUMENT_SEPARATORS["documentation"]
        assert "\n## " in seps
        assert "\n### " in seps

    def test_api_docs_separators(self):
        seps = DOCUMENT_SEPARATORS["api_docs"]
        assert "\nEndpoint" in seps or "\nResponse" in seps


# ─── AdaptiveChunker ─────────────────────────────────────────────────────

class TestAdaptiveChunker:
    """Tests del chunker adaptativo."""

    def test_chunk_returns_list(self):
        docs = [
            Document(
                page_content="Some content for testing purposes.\n\n## Section 1\n\nMore content here.\n\n### Subsection 1.1\n\nDetailed information.",
                metadata={"source": "test.md"},
            ),
        ]

        result = AdaptiveChunker.chunk(docs, chunk_size=100, chunk_overlap=20)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunks_have_metadata(self):
        docs = [
            Document(
                page_content="Content with some text to chunk properly.",
                metadata={"source": "test.md"},
            ),
        ]

        result = AdaptiveChunker.chunk(docs, chunk_size=50, chunk_overlap=10)

        for chunk in result:
            assert "document_type" in chunk.metadata

    def test_detect_and_chunk_alias(self):
        docs = [
            Document(
                page_content="Test content for alias check.",
                metadata={"source": "test.md"},
            ),
        ]

        result1 = AdaptiveChunker.chunk(docs, chunk_size=50)
        result2 = AdaptiveChunker.detect_and_chunk(docs)

        # Both should return lists of Documents
        assert isinstance(result1, list)
        assert isinstance(result2, list)
