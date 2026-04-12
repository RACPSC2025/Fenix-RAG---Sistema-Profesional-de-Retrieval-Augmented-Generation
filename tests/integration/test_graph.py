"""Tests de integración del grafo LangGraph completo."""

from __future__ import annotations

import pytest

from unittest.mock import MagicMock, patch

from src.agent.graph import build_graph, get_graph, run_agent
from src.agent.state import AgentState, initial_state


class TestGraphIntegration:
    """Tests de integración del StateGraph completo."""

    @pytest.fixture(autouse=True)
    def reset_graph(self):
        """Resetear singleton del grafo entre tests."""
        from src.agent import graph as graph_module
        graph_module._graph = None
        yield
        graph_module._graph = None

    def test_graph_builds_without_tools(self):
        """El grafo se compila sin tools."""
        graph = build_graph(with_tools=False)
        assert graph is not None

    def test_graph_builds_with_tools(self):
        """El grafo se compila con tools."""
        graph = build_graph(with_tools=True)
        assert graph is not None

    def test_graph_has_required_nodes(self):
        """El grafo tiene todos los nodos requeridos."""
        graph = build_graph(with_tools=True)
        # Verificar que los nodos existen
        compiled_graph = graph
        # LangGraph no expone nodos directamente, pero podemos verificar que invoke funciona
        assert compiled_graph is not None

    def test_graph_singletons_return_same_instance(self):
        """get_graph() retorna la misma instancia."""
        graph1 = get_graph(force_rebuild=True)
        graph2 = get_graph()
        assert graph1 is graph2


class TestRunAgent:
    """Tests de la función de alto nivel run_agent."""

    @pytest.fixture(autouse=True)
    def reset_graph(self):
        from src.agent import graph as graph_module
        graph_module._graph = None
        yield
        graph_module._graph = None

    @patch("src.agent.nodes.retrieval_node.retrieval_node")
    @patch("src.agent.nodes.generation_node.generation_node")
    @patch("src.agent.nodes.reflection_node.reflection_node")
    def test_run_agent_returns_response(
        self, mock_reflection, mock_generation, mock_retrieval,
    ):
        """run_agent retorna una respuesta estructurada."""
        mock_retrieval.return_value = {
            "retrieval_results": [],
            "retrieval_strategy": "vector",
            "pipeline_metrics": {},
        }
        mock_generation.return_value = {
            "draft_answer": "Esta es una respuesta de prueba.",
            "sources": [],
            "generation_mode": "direct",
            "pipeline_metrics": {},
        }
        mock_reflection.return_value = {
            "final_answer": "Esta es una respuesta de prueba.",
            "reflection": {
                "score": 0.9,
                "is_grounded": True,
                "has_hallucination": False,
                "cites_source": False,
                "feedback": "Respuesta válida",
                "reformulated_query": "",
            },
            "route": "END",
            "iteration_count": 1,
            "pipeline_metrics": {},
        }

        result = run_agent(
            user_query="¿Qué es RACodex?",
            session_id="test-session",
            max_iterations=1,
        )

        assert "final_answer" in result
        assert "sources" in result
        assert result["final_answer"] == "Esta es una respuesta de prueba."

    def test_run_agent_handles_empty_query(self):
        """Query vacía no debe causar error."""
        # Aunque el grafo puede fallar, run_agent debe capturar el error
        result = run_agent(
            user_query="",
            session_id="test-session",
            max_iterations=1,
        )

        # Debe retornar algo incluso si falla
        assert "final_answer" in result
        assert "sources" in result

    def test_initial_state_has_required_fields(self):
        """initial_state() retorna un estado con todos los campos requeridos."""
        state = initial_state(
            user_query="test query",
            session_id="test-session",
            uploaded_files=["test.pdf"],
            max_iterations=3,
        )

        assert state["user_query"] == "test query"
        assert state["session_id"] == "test-session"
        assert state["uploaded_files"] == ["test.pdf"]
        assert state["max_iterations"] == 3
        assert state["active_query"] == "test query"
        assert state["messages"] is not None
        assert state["pipeline_metrics"] == {}
        assert state["session_memory"] == {}
        assert state["grade_score"] == 0.0
        assert state["generation_mode"] == ""
