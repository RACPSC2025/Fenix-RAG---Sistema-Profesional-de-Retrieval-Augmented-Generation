"""Tests del generation_node con Re2 condicional."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from src.agent.nodes.all_nodes import generation_node


class TestGenerationNodeConditionalRe2:
    """Tests de la generacion condicional basada en CRAG grade_score."""

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(
                page_content="Relevant content about the topic",
                metadata={"source": "doc.pdf", "chunk_index": 0},
            ),
        ]

    def test_generation_no_docs(self):
        """Sin documentos → mensaje informativo."""
        state = {
            "user_query": "test query",
            "retrieval_results": [],
        }

        result = generation_node(state)

        assert "No encontré documentos" in result["draft_answer"]
        assert result["sources"] == []
        assert result["generation_mode"] == "no_docs"

    @patch("src.agent.nodes.all_nodes.generate_direct")
    def test_direct_generation_when_score_high(self, mock_direct, sample_docs):
        """grade_score > 0.8 → generacion directa (1 llamada LLM)."""
        mock_direct.return_value = ("Direct answer", [{"source": "doc.pdf"}])

        state = {
            "user_query": "test query",
            "retrieval_results": sample_docs,
            "grade_score": 0.95,
        }

        result = generation_node(state)

        mock_direct.assert_called_once()
        assert result["generation_mode"] == "direct"
        assert result["draft_answer"] == "Direct answer"

    @patch("src.agent.nodes.all_nodes.generate_with_rethinking")
    def test_rethinking_when_score_medium(self, mock_rethinking, sample_docs):
        """grade_score 0.5-0.8 → Re2 dos pasadas (2 llamadas LLM)."""
        mock_rethinking.return_value = ("Rethinking answer", [{"source": "doc.pdf"}])

        state = {
            "user_query": "test query",
            "retrieval_results": sample_docs,
            "grade_score": 0.65,
        }

        result = generation_node(state)

        mock_rethinking.assert_called_once()
        assert result["generation_mode"] == "rethinking"
        assert result["draft_answer"] == "Rethinking answer"

    @patch("src.agent.nodes.all_nodes.generate_with_rethinking")
    def test_rethinking_with_warning_when_score_low(self, mock_rethinking, sample_docs):
        """grade_score < 0.5 → Re2 + advertencia."""
        mock_rethinking.return_value = ("Marginal answer", [{"source": "doc.pdf"}])

        state = {
            "user_query": "test query",
            "retrieval_results": sample_docs,
            "grade_score": 0.25,
        }

        result = generation_node(state)

        mock_rethinking.assert_called_once()
        assert result["generation_mode"] == "rethinking_low_confidence"
        assert "Marginal answer" in result["draft_answer"]
        assert "relevancia limitada" in result["draft_answer"]
        assert "0.25" in result["draft_answer"]

    @patch("src.agent.nodes.all_nodes.generate_direct")
    def test_default_score_uses_direct(self, mock_direct, sample_docs):
        """Sin grade_score (default 0.0) → deberia usar rethinking, no direct."""
        mock_direct.return_value = ("Answer", [])

        state = {
            "user_query": "test query",
            "retrieval_results": sample_docs,
            # grade_score not set → defaults to 0.0
        }

        # Since grade_score defaults to 0.0, it should use rethinking, not direct
        with patch("src.agent.nodes.all_nodes.generate_with_rethinking") as mock_rethinking:
            mock_rethinking.return_value = ("Answer", [])
            result = generation_node(state)

            mock_direct.assert_not_called()
            mock_rethinking.assert_called_once()
            assert result["generation_mode"] == "rethinking_low_confidence"

    @patch("src.agent.nodes.all_nodes.generate_direct")
    def test_boundary_score_0_8_uses_direct(self, mock_direct, sample_docs):
        """grade_score exactly 0.8 → edge case. Should be direct (> 0.8 is direct)."""
        mock_direct.return_value = ("Answer", [])

        state = {
            "user_query": "test query",
            "retrieval_results": sample_docs,
            "grade_score": 0.8,  # Exactly at boundary
        }

        # 0.8 is NOT > 0.8, so should fall to elif (0.5-0.8)
        with patch("src.agent.nodes.all_nodes.generate_with_rethinking") as mock_rethinking:
            mock_rethinking.return_value = ("Answer", [])
            result = generation_node(state)

            mock_direct.assert_not_called()
            mock_rethinking.assert_called_once()

    @patch("src.agent.nodes.all_nodes.generate_direct")
    def test_boundary_score_0_81_uses_direct(self, mock_direct, sample_docs):
        """grade_score 0.81 → should use direct."""
        mock_direct.return_value = ("Direct answer", [])

        state = {
            "user_query": "test query",
            "retrieval_results": sample_docs,
            "grade_score": 0.81,
        }

        result = generation_node(state)

        mock_direct.assert_called_once()
        assert result["generation_mode"] == "direct"
