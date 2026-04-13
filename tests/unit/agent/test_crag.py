"""
Tests del CRAG skill — Corrective RAG grading.

Cubre los contratos de estado post-fix:
  - grade_documents_node escribe `crag_route` (no `route`)
  - route_after_grading lee `crag_route` (no `route`)
  - crag_retry_count se incrementa en cada re-retrieval
  - MAX_CRAG_RETRIES fuerza generation cuando se agotan los reintentos
  - doc_quality se persiste en el estado (ahora existe en AgentState)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agent.skills.crag import (
    MAX_CRAG_RETRIES,
    DocumentGrade,
    grade_documents,
    grade_documents_node,
    rewrite_query_for_reretrieval,
    route_after_grading,
)


# ─── grade_documents ─────────────────────────────────────────────────────────

class TestGradeDocuments:

    def test_empty_documents_returns_incorrect(self):
        grade = grade_documents("test query", [])
        assert grade.quality == "incorrect"
        assert grade.score == 0.0

    @patch("src.agent.skills.crag.get_llm")
    def test_grading_calls_llm_with_chat_prompt(self, mock_get_llm):
        """Verifica que usa ChatPromptTemplate (no str directo)."""
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = DocumentGrade(
            quality="correct", score=0.85, reasoning="Relevant"
        )
        mock_llm.with_structured_output.return_value = mock_chain
        mock_get_llm.return_value = mock_llm

        docs = [Document(page_content="content", metadata={"source": "doc.pdf", "chunk_index": 0})]
        grade = grade_documents("test query", docs)

        mock_get_llm.assert_called_once_with(temperature=0)
        mock_chain.invoke.assert_called_once()
        assert grade.quality == "correct"
        assert grade.score == 0.85

    @patch("src.agent.skills.crag.get_llm")
    def test_grading_fallback_ambiguous_on_llm_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM timeout")
        mock_llm.with_structured_output.return_value = mock_chain
        mock_get_llm.return_value = mock_llm

        docs = [Document(page_content="content", metadata={"source": "doc.pdf"})]
        grade = grade_documents("test query", docs)

        assert grade.quality == "ambiguous"
        assert grade.score == 0.5

    def test_limits_to_5_docs_for_grading(self):
        """El grader no debe enviar más de 5 docs al LLM."""
        with patch("src.agent.skills.crag.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = DocumentGrade(
                quality="correct", score=0.9, reasoning="ok"
            )
            mock_llm.with_structured_output.return_value = mock_chain
            mock_get_llm.return_value = mock_llm

            # 10 docs — solo deben pasar 5 al LLM
            docs = [
                Document(page_content=f"content {i}", metadata={"source": f"doc{i}.pdf"})
                for i in range(10)
            ]
            grade_documents("query", docs)

            call_args = mock_chain.invoke.call_args[0][0]["input"]
            # El summary solo debe mencionar hasta Doc 5
            assert "Doc 5" in call_args
            assert "Doc 6" not in call_args


# ─── rewrite_query_for_reretrieval ───────────────────────────────────────────

class TestRewriteQuery:

    @patch("src.agent.skills.crag.get_llm")
    def test_rewrite_for_ambiguous_adds_detail(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Rewritten specific query")
        mock_get_llm.return_value = mock_llm

        grade = DocumentGrade(quality="ambiguous", score=0.5, reasoning="Partial")
        result = rewrite_query_for_reretrieval("original query", grade)

        assert result == "Rewritten specific query"
        # Verifica que el prompt menciona "específica" (para ambiguous)
        prompt_used = mock_llm.invoke.call_args[0][0]
        assert "específica" in prompt_used

    @patch("src.agent.skills.crag.get_llm")
    def test_rewrite_for_incorrect_uses_stepback(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Broader general query")
        mock_get_llm.return_value = mock_llm

        grade = DocumentGrade(quality="incorrect", score=0.1, reasoning="Not relevant")
        result = rewrite_query_for_reretrieval("specific query", grade)

        assert result == "Broader general query"
        # Verifica que el prompt menciona "amplia" (para incorrect / step-back)
        prompt_used = mock_llm.invoke.call_args[0][0]
        assert "amplia" in prompt_used

    @patch("src.agent.skills.crag.get_llm")
    def test_rewrite_fallback_returns_original_on_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm

        grade = DocumentGrade(quality="ambiguous", score=0.5, reasoning="test")
        result = rewrite_query_for_reretrieval("original query", grade)

        assert result == "original query"


# ─── grade_documents_node — CONTRATO DE ESTADO ───────────────────────────────

class TestGradeDocumentsNode:
    """
    Tests del contrato de estado post-fix.

    INVARIANTE CLAVE: el nodo escribe `crag_route`, NUNCA `route`.
    Cualquier test que verifique `result["route"]` está verificando el bug
    antiguo — estos tests verifican el comportamiento correcto.
    """

    @patch("src.agent.skills.crag.grade_documents")
    def test_correct_writes_crag_route_generation(self, mock_grade):
        """correct → crag_route="generation", NO route="generation"."""
        mock_grade.return_value = DocumentGrade(quality="correct", score=0.85, reasoning="Good")

        state = {"user_query": "test", "retrieval_results": [Document(page_content="c")]}
        result = grade_documents_node(state)

        assert result["crag_route"] == "generation"
        assert "route" not in result, "El nodo NO debe escribir al campo `route`"
        assert result["doc_quality"] == "correct"
        assert result["grade_score"] == 0.85

    @patch("src.agent.skills.crag.grade_documents")
    def test_ambiguous_writes_crag_route_retrieval(self, mock_grade):
        """ambiguous con retries disponibles → crag_route="retrieval"."""
        mock_grade.return_value = DocumentGrade(quality="ambiguous", score=0.5, reasoning="Partial")

        with patch("src.agent.skills.crag.rewrite_query_for_reretrieval") as mock_rewrite:
            mock_rewrite.return_value = "rewritten query"
            state = {
                "user_query": "test",
                "retrieval_results": [Document(page_content="c")],
                "crag_retry_count": 0,
            }
            result = grade_documents_node(state)

        assert result["crag_route"] == "retrieval"
        assert "route" not in result, "El nodo NO debe escribir al campo `route`"
        assert result["active_query"] == "rewritten query"
        assert result["crag_retry_count"] == 1

    @patch("src.agent.skills.crag.grade_documents")
    def test_incorrect_writes_crag_route_retrieval(self, mock_grade):
        """incorrect con retries disponibles → crag_route="retrieval"."""
        mock_grade.return_value = DocumentGrade(quality="incorrect", score=0.1, reasoning="Bad")

        with patch("src.agent.skills.crag.rewrite_query_for_reretrieval") as mock_rewrite:
            mock_rewrite.return_value = "step-back query"
            state = {
                "user_query": "test",
                "retrieval_results": [Document(page_content="c")],
                "crag_retry_count": 0,
            }
            result = grade_documents_node(state)

        assert result["crag_route"] == "retrieval"
        assert result["crag_retry_count"] == 1

    @patch("src.agent.skills.crag.grade_documents")
    def test_max_retries_forces_generation(self, mock_grade):
        """
        Con crag_retry_count >= MAX_CRAG_RETRIES, fuerza generation
        aunque los docs sean ambiguous — protección anti-loop.
        """
        mock_grade.return_value = DocumentGrade(quality="ambiguous", score=0.4, reasoning="Partial")

        state = {
            "user_query": "test",
            "retrieval_results": [Document(page_content="c")],
            "crag_retry_count": MAX_CRAG_RETRIES,  # límite alcanzado
        }
        result = grade_documents_node(state)

        assert result["crag_route"] == "generation", (
            "Con max retries alcanzados debe forzar generation aunque docs sean ambiguous"
        )
        # NO debe incrementar el contador ni reescribir la query
        assert "active_query" not in result
        # El contador no se modifica cuando se fuerza generation
        assert result.get("crag_retry_count", MAX_CRAG_RETRIES) == MAX_CRAG_RETRIES

    @patch("src.agent.skills.crag.grade_documents")
    def test_max_retries_forces_generation_on_incorrect(self, mock_grade):
        """También funciona con incorrect cuando retries agotados."""
        mock_grade.return_value = DocumentGrade(quality="incorrect", score=0.1, reasoning="Bad")

        state = {
            "user_query": "test",
            "retrieval_results": [Document(page_content="c")],
            "crag_retry_count": MAX_CRAG_RETRIES + 1,  # más del límite
        }
        result = grade_documents_node(state)

        assert result["crag_route"] == "generation"
        assert "active_query" not in result

    @patch("src.agent.skills.crag.grade_documents")
    def test_retry_count_increments_correctly(self, mock_grade):
        """Cada re-retrieval incrementa crag_retry_count en 1."""
        mock_grade.return_value = DocumentGrade(quality="ambiguous", score=0.5, reasoning="P")

        with patch("src.agent.skills.crag.rewrite_query_for_reretrieval") as mock_rewrite:
            mock_rewrite.return_value = "rewritten"

            # Primer intento
            state = {"user_query": "t", "retrieval_results": [Document(page_content="c")], "crag_retry_count": 0}
            r1 = grade_documents_node(state)
            assert r1["crag_retry_count"] == 1

            # Segundo intento
            state["crag_retry_count"] = 1
            r2 = grade_documents_node(state)
            assert r2["crag_retry_count"] == 2

    def test_empty_docs_returns_incorrect_and_crag_route_retrieval(self):
        """Sin documentos → incorrect → crag_route="retrieval" (retry_count=0 < MAX)."""
        state = {"user_query": "test", "retrieval_results": [], "crag_retry_count": 0}
        result = grade_documents_node(state)

        assert result["doc_quality"] == "incorrect"
        assert result["crag_route"] == "retrieval"
        assert "route" not in result

    def test_doc_quality_is_always_in_result(self):
        """doc_quality debe estar siempre en el resultado — ya existe en AgentState."""
        state = {"user_query": "test", "retrieval_results": [], "crag_retry_count": 0}
        result = grade_documents_node(state)
        assert "doc_quality" in result
        assert result["doc_quality"] in ("correct", "ambiguous", "incorrect")

    def test_grade_score_is_always_in_result(self):
        """grade_score debe estar siempre presente."""
        state = {"user_query": "test", "retrieval_results": [], "crag_retry_count": 0}
        result = grade_documents_node(state)
        assert "grade_score" in result
        assert isinstance(result["grade_score"], float)


# ─── route_after_grading — lee crag_route ────────────────────────────────────

class TestRouteAfterGrading:
    """
    Tests del edge condicional post-fix.

    INVARIANTE: route_after_grading lee `crag_route`, NO `route`.
    """

    def test_reads_crag_route_generation(self):
        state = {"crag_route": "generation"}
        assert route_after_grading(state) == "generation"

    def test_reads_crag_route_retrieval(self):
        state = {"crag_route": "retrieval"}
        assert route_after_grading(state) == "retrieval"

    def test_default_is_generation_when_crag_route_missing(self):
        """Sin crag_route en estado → default generation (safe)."""
        state = {}
        assert route_after_grading(state) == "generation"

    def test_ignores_route_field(self):
        """
        El campo `route` no debe afectar el resultado.
        Verifica aislamiento del contrato de estado.
        """
        # route dice "retrieval" pero crag_route dice "generation"
        state = {"route": "retrieval", "crag_route": "generation"}
        assert route_after_grading(state) == "generation"

    def test_ignores_route_field_inverse(self):
        # route dice "generation" pero crag_route dice "retrieval"
        state = {"route": "generation", "crag_route": "retrieval"}
        assert route_after_grading(state) == "retrieval"
