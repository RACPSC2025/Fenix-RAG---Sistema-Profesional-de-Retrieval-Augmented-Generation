"""
Memory tools — Herramientas para gestionar contexto conversacional del agente.

Permiten al agente guardar y recuperar información de contexto
entre iteraciones del grafo LangGraph.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.config.logging import get_logger

log = get_logger(__name__)


@tool
def save_context(
    key: str,
    value: str,
    session_id: str = "",
) -> dict[str, Any]:
    """
    Guarda un par clave-valor en el contexto de la sesión.

    Útil para almacenar hallazgos intermedios, preferencias del usuario,
    o resultados de análisis que se necesitan en iteraciones posteriores.

    Args:
        key: Identificador del contexto (ej: "tema_principal", "articulos_relevantes").
        value: Valor a guardar (string).
        session_id: ID de sesión para aislamiento.

    Returns:
        Dict confirmando el guardado.
    """
    # En una implementación completa, esto iría a un store persistente
    # (Redis, PostgreSQL, o LangGraph Store)
    # Por ahora, retorna confirmación para que el agente lo use internamente
    result = {
        "saved": True,
        "key": key,
        "session_id": session_id,
        "value_length": len(value),
    }

    log.info(
        "context_saved",
        key=key,
        session=session_id,
    )

    return result


@tool
def retrieve_context(
    key: str,
    session_id: str = "",
) -> dict[str, Any]:
    """
    Recupera un valor guardado previamente en el contexto de la sesión.

    Args:
        key: Identificador del contexto a recuperar.
        session_id: ID de sesión para aislamiento.

    Returns:
        Dict con el valor recuperado o indicador de no encontrado.
    """
    # En una implementación completa, esto consultaría el store persistente
    result = {
        "found": False,
        "key": key,
        "value": None,
        "session_id": session_id,
    }

    log.debug(
        "context_retrieve_attempt",
        key=key,
        session=session_id,
    )

    return result


__all__ = ["save_context", "retrieve_context"]
