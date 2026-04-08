"""
Nodes del grafo LangGraph.

Todos los nodos están implementados individualmente en este paquete.
El grafo principal (graph.py) los importa desde all_nodes.

Estado actual: los nodos individuales están pendientes de implementación.
Por ahora, los placeholders retornan el estado sin modificar.
"""

# TODO: Implementar nodos individuales en sus archivos correspondientes.
# Por ahora, este __init__.py exporta stubs para que los imports no fallen.

from typing import Any


def document_router_node(state: dict) -> dict:
    """Clasifica archivos subidos y decide ruta de ingestión."""
    return state


def ingestion_node(state: dict) -> dict:
    """Ejecuta IngestionPipeline e indexa chunks a Chroma."""
    return state


def retrieval_node(state: dict) -> dict:
    """Ejecuta EnsembleRetriever con estrategia auto-seleccionada."""
    return state


def generation_node(state: dict) -> dict:
    """Genera respuesta con contexto de retrieval_results."""
    return state


def reflection_node(state: dict) -> dict:
    """Auto-evaluación de la respuesta generada."""
    return state


def supervisor_node(state: dict) -> dict:
    """Supervisor pattern: coordina subagentes especializados."""
    return state


__all__ = [
    "document_router_node",
    "ingestion_node",
    "retrieval_node",
    "generation_node",
    "reflection_node",
    "supervisor_node",
]
