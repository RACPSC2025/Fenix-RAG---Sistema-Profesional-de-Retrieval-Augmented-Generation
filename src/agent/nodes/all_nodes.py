"""
Bundle de todos los nodos del grafo LangGraph.

Este módulo re-exporta todos los nodos desde el paquete `nodes`.
Permite el import que usa graph.py:

    from src.agent.nodes.all_nodes import (
        document_router_node,
        generation_node,
        ...
    )
"""

from src.agent.nodes import (
    document_router_node,
    generation_node,
    ingestion_node,
    reflection_node,
    retrieval_node,
    supervisor_node,
)

__all__ = [
    "document_router_node",
    "ingestion_node",
    "retrieval_node",
    "generation_node",
    "reflection_node",
    "supervisor_node",
]
