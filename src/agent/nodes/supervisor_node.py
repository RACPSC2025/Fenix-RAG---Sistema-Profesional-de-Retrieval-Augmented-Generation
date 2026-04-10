"""
Supervisor node — coordina subagentes especializados.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor pattern: coordina subagentes especializados.

    Por ahora, solo valida la ruta — implementación completa en Fase futura.
    """
    return {"route": state.get("route", "retrieval")}
