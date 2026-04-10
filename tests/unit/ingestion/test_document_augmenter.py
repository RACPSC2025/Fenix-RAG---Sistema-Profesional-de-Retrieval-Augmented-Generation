"""Tests del DocumentAugmenter — generación de preguntas por chunk."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from langchain_core.documents import Document

from src.ingestion.processors.document_augmenter import (
    augment_documents,
    augment_documents_async,
    augment_and_index,
    _generate_questions,
)


# ─── augment_documents ─────────────────────────────────────────────────────

class TestAugmentDocuments:
    """Tests de augmentación de documentos."""

    def test_returns_originals_plus_questions(self):
        docs = [
            Document(
                page_content="JWT is used for authentication.",
                metadata={"source": "auth.pdf", "chunk_index": 0},
            ),
        ]

        with patch("src.ingestion.processors.document_augmenter._generate_questions") as mock_gen:
            mock_gen.return_value = ["What is JWT?", "How is authentication implemented?"]

            result = augment_documents(docs, questions_per_chunk=2)

            # Should have 1 original + 2 questions = 3 docs
            assert len(result) == 3
            # Original should be unchanged
            assert result[0].page_content == "JWT is used for authentication."
            # Questions should have augmented metadata
            assert result[1].metadata["is_augmented_question"] is True
            assert result[1].metadata["augmentation_type"] == "question"

    def test_handles_empty_docs(self):
        result = augment_documents([])
        assert result == []

    def test_continues_on_question_generation_error(self):
        docs = [
            Document(page_content="Content 1", metadata={"chunk_index": 0}),
            Document(page_content="Content 2", metadata={"chunk_index": 1}),
        ]

        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Generation error")
            return ["Question about content 2"]

        with patch("src.ingestion.processors.document_augmenter._generate_questions", side_effect=mock_generate):
            result = augment_documents(docs, questions_per_chunk=1)

        # Should have 2 originals + 1 question (second chunk only)
        assert len(result) >= 2

    def test_preserves_original_metadata(self):
        docs = [
            Document(
                page_content="Content",
                metadata={
                    "source": "doc.pdf",
                    "chunk_index": 0,
                    "article_number": "5",
                },
            ),
        ]

        with patch("src.ingestion.processors.document_augmenter._generate_questions") as mock_gen:
            mock_gen.return_value = ["Question?"]

            result = augment_documents(docs, questions_per_chunk=1)

            question_doc = result[1]  # First question
            assert question_doc.metadata["source"] == "doc.pdf"
            assert question_doc.metadata["parent_chunk_index"] == 0


# ─── _generate_questions ─────────────────────────────────────────────────

class TestGenerateQuestions:
    """Tests de generación de preguntas."""

    @patch("src.ingestion.processors.document_augmenter.get_llm")
    def test_generates_questions(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="What is the purpose?\nHow does it work?\nWho is responsible?"
        )
        mock_get_llm.return_value = mock_llm

        doc = Document(page_content="Some content about a topic.")
        questions = _generate_questions(doc, mock_llm, n=3)

        assert len(questions) == 3
        assert "purpose" in questions[0].lower()

    @patch("src.ingestion.processors.document_augmenter.get_llm")
    def test_limits_to_n_questions(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Q1?\nQ2?\nQ3?\nQ4?\nQ5?"
        )
        mock_get_llm.return_value = mock_llm

        doc = Document(page_content="Content.")
        questions = _generate_questions(doc, mock_llm, n=2)

        assert len(questions) <= 2

    @patch("src.ingestion.processors.document_augmenter.get_llm")
    def test_cleans_question_numbering(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="1. What is this?\n2. How does it work?\n- Why is it important?"
        )
        mock_get_llm.return_value = mock_llm

        doc = Document(page_content="Content.")
        questions = _generate_questions(doc, mock_llm, n=3)

        # Should strip numbering
        for q in questions:
            assert not q.startswith("1.")
            assert not q.startswith("-")

    @patch("src.ingestion.processors.document_augmenter.get_llm")
    def test_filters_empty_questions(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Valid question?\n\n\nToo short\nAnother valid question?"
        )
        mock_get_llm.return_value = mock_llm

        doc = Document(page_content="Content.")
        questions = _generate_questions(doc, mock_llm, n=3)

        # Should filter out empty and too-short questions
        assert all(len(q) > 10 for q in questions)


# ─── augment_and_index ─────────────────────────────────────────────────────

class TestAugmentAndIndex:
    """Tests de augmentación + indexación."""

    @patch("src.ingestion.processors.document_augmenter.augment_documents")
    def test_returns_stats(self, mock_augment):
        mock_augment.return_value = [
            Document(page_content="original", metadata={}),
            Document(page_content="Q1?", metadata={"is_augmented_question": True}),
        ]

        mock_vs = MagicMock()

        result = augment_and_index(
            chunks=[Document(page_content="original", metadata={})],
            vector_store=mock_vs,
            questions_per_chunk=1,
        )

        assert result["original_chunks"] == 1
        assert result["questions_added"] == 1
        assert result["total_indexed"] == 2
        mock_vs.add_documents.assert_called_once()


# ─── Async Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAugmentDocumentsAsync:
    """Tests de augmentación async con concurrencia."""

    async def test_async_returns_augmented_docs(self):
        """augment_documents_async retorna originales + preguntas."""
        chunks = [
            Document(
                page_content="Content about authentication.",
                metadata={"source": "doc.pdf", "chunk_index": 0},
            ),
        ]

        with patch("src.ingestion.processors.document_augmenter._generate_questions_async") as mock_gen:
            mock_gen.return_value = ["What is auth?", "How does auth work?"]

            result = await augment_documents_async(chunks, questions_per_chunk=2)

            assert len(result) == 3  # 1 original + 2 questions
            assert result[0].page_content == "Content about authentication."
            assert result[1].metadata["is_augmented_question"] is True
            assert result[2].metadata["is_augmented_question"] is True

    async def test_async_handles_errors_gracefully(self):
        """Si un chunk falla, continúa con los demás."""
        chunks = [
            Document(page_content="Chunk 1", metadata={"chunk_index": 0}),
            Document(page_content="Chunk 2", metadata={"chunk_index": 1}),
        ]

        call_count = 0

        async def mock_generate(chunk, llm, n):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("LLM error")
            return ["Question about chunk 2"]

        with patch("src.ingestion.processors.document_augmenter._generate_questions_async", side_effect=mock_generate):
            result = await augment_documents_async(chunks, questions_per_chunk=1)

            # Should have 2 originals + 1 question (second chunk only)
            assert len(result) >= 2

    async def test_async_uses_semaphore_for_concurrency(self):
        """El semáforo limita la concurrencia."""
        chunks = [
            Document(page_content=f"Chunk {i}", metadata={"chunk_index": i})
            for i in range(10)
        ]

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_generate(chunk, llm, n):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

            await asyncio.sleep(0.01)  # Simular latencia

            async with lock:
                current_concurrent -= 1

            return [f"Question about {chunk.page_content}"]

        with patch("src.ingestion.processors.document_augmenter._generate_questions_async", side_effect=mock_generate):
            result = await augment_documents_async(
                chunks, questions_per_chunk=1, max_concurrency=3
            )

            # Max concurrent should be <= 3
            assert max_concurrent <= 3
            assert len(result) == 20  # 10 originals + 10 questions


@pytest.mark.asyncio
class TestAugmentAndIndexAsync:
    """Tests de augment_and_index_async."""

    async def test_returns_stats(self):
        """augment_and_index_async retorna stats correctas."""
        chunks = [Document(page_content="original", metadata={})]
        mock_vs = MagicMock()

        with patch("src.ingestion.processors.document_augmenter.augment_documents_async") as mock_augment:
            mock_augment.return_value = [
                Document(page_content="original", metadata={}),
                Document(page_content="Q1?", metadata={"is_augmented_question": True}),
            ]

            result = await augment_and_index_async(
                chunks=chunks,
                vector_store=mock_vs,
                questions_per_chunk=1,
            )

            assert result["original_chunks"] == 1
            assert result["questions_added"] == 1
            assert result["total_indexed"] == 2
            mock_vs.add_documents.assert_called_once()
