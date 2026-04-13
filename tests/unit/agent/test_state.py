"""
Tests del AgentState — validación de estado del grafo.

Cubre los campos nuevos post-fix:
  - crag_route, crag_retry_count, doc_quality (CRAG)
  - reflection_route (reflection)
  - Aislamiento entre campos de routing
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agent.state import AgentState, initial_state


class TestAgentState:
    """Tests de inicialización y estructura del estado."""

    def test_initial_state_has_messages(self) -> None:
        state = initial_state(user_query="¿qué dice el artículo 5?")
        assert "messages" in state
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)

    def test_initial_state_has_user_query(self) -> None:
        state = initial_state(user_query="test query")
        assert state["user_query"] == "test query"
        assert state["active_query"] == "test query"

    def test_initial_state_has_session_id(self) -> None:
        state = initial_state(user_query="test", session_id="session-123")
        assert state["session_id"] == "session-123"

    def test_initial_state_has_empty_collections(self) -> None:
        state = initial_state(user_query="test")
        assert state["uploaded_files"] == []
        assert state["ingestion_plans"] == []
        assert state["ingested_documents"] == []
        assert state["retrieval_results"] == []
        assert state["sources"] == []

    def test_initial_state_has_base_defaults(self) -> None:
        state = initial_state(user_query="test")
        assert state["draft_answer"] == ""
        assert state["final_answer"] == ""
        assert state["reflection"] is None
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 2
        assert state["error"] is None
        assert state["route"] == ""
        assert state["session_memory"] == {}
        assert state["grade_score"] == 0.0
        assert state["generation_mode"] == ""
        assert state["pipeline_metrics"] == {}
        assert state["retrieval_strategy"] == ""

    # ── Campos CRAG (nuevos) ──────────────────────────────────────────────────

    def test_initial_state_has_crag_route(self) -> None:
        """crag_route debe inicializarse vacío, no con un valor por defecto."""
        state = initial_state(user_query="test")
        assert "crag_route" in state
        assert state["crag_route"] == ""

    def test_initial_state_has_crag_retry_count_zero(self) -> None:
        """crag_retry_count debe comenzar en 0 para el conteo de reintentos."""
        state = initial_state(user_query="test")
        assert "crag_retry_count" in state
        assert state["crag_retry_count"] == 0

    def test_initial_state_has_doc_quality(self) -> None:
        """doc_quality debe estar en el estado para que LangGraph lo persista."""
        state = initial_state(user_query="test")
        assert "doc_quality" in state
        assert state["doc_quality"] == ""

    # ── Campos Reflection (nuevos) ────────────────────────────────────────────

    def test_initial_state_has_reflection_route(self) -> None:
        """reflection_route debe inicializarse vacío."""
        state = initial_state(user_query="test")
        assert "reflection_route" in state
        assert state["reflection_route"] == ""

    # ── Aislamiento de routing ────────────────────────────────────────────────

    def test_routing_fields_are_independent(self) -> None:
        """
        Los tres campos de routing deben ser independientes.
        Ninguno debe afectar a los otros en el estado inicial.
        """
        state = initial_state(user_query="test")
        # Todos vacíos al inicio
        assert state["route"] == ""
        assert state["crag_route"] == ""
        assert state["reflection_route"] == ""

    def test_crag_route_not_aliased_to_route(self) -> None:
        """
        crag_route y route son campos distintos — no deben ser el mismo objeto.
        Si se modifica uno, el otro no debe cambiar.
        """
        state = initial_state(user_query="test")
        state["crag_route"] = "generation"
        assert state["route"] == "", (
            "Modificar crag_route no debe afectar a route"
        )

    def test_reflection_route_not_aliased_to_route(self) -> None:
        state = initial_state(user_query="test")
        state["reflection_route"] = "END"
        assert state["route"] == "", (
            "Modificar reflection_route no debe afectar a route"
        )

    def test_reflection_route_not_aliased_to_crag_route(self) -> None:
        state = initial_state(user_query="test")
        state["reflection_route"] = "retrieval"
        assert state["crag_route"] == "", (
            "Modificar reflection_route no debe afectar a crag_route"
        )

    # ── AgentState TypedDict coverage ────────────────────────────────────────

    def test_agent_state_has_all_required_keys(self) -> None:
        """
        Verifica que initial_state produce todos los campos requeridos.
        Cualquier campo en AgentState que no esté en initial_state causará
        KeyError en runtime cuando LangGraph intente acceder a él.
        """
        state = initial_state(user_query="test")
        required_keys = [
            "messages", "session_id", "user_query", "active_query",
            "uploaded_files", "ingestion_plans", "ingested_documents",
            "retrieval_results", "retrieval_strategy",
            # Skill Pack [Fase 10]
            "active_profile",
            # CRAG
            "doc_quality", "grade_score", "crag_route", "crag_retry_count",
            # Generación
            "draft_answer", "final_answer", "sources", "generation_mode",
            # Reflexión
            "reflection", "reflection_route", "iteration_count", "max_iterations",
            # Memoria y metadata
            "session_memory", "error", "route", "pipeline_metrics",
        ]
        for key in required_keys:
            assert key in state, f"Campo requerido faltante en initial_state: '{key}'"

    def test_initial_state_with_uploaded_files(self) -> None:
        state = initial_state(
            user_query="test",
            uploaded_files=["/path/to/file.pdf"],
        )
        assert state["uploaded_files"] == ["/path/to/file.pdf"]

    def test_initial_state_custom_max_iterations(self) -> None:
        state = initial_state(user_query="test", max_iterations=5)
        assert state["max_iterations"] == 5

    # ── Skill Pack [Fase 10] ──────────────────────────────────────────────────

    def test_initial_state_active_profile_default_empty(self) -> None:
        """active_profile debe comenzar en '' para que generation_node use default."""
        state = initial_state(user_query="test")
        assert "active_profile" in state
        assert state["active_profile"] == ""

    def test_initial_state_active_profile_explicit(self) -> None:
        state = initial_state(user_query="test", active_profile="ai-rag-engineer")
        assert state["active_profile"] == "ai-rag-engineer"
