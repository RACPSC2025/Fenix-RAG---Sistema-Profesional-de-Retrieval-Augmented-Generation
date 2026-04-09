"""
Parent-Child Retrieval — Index small, retrieve large.

Arquitectura:
  1. Los chunks hijos (incisos, sub-secciones) se indexan en Chroma
     para búsqueda precisa por semántica o BM25.
  2. Los documentos padres (artículos completos, secciones completas)
     se almacenan en un ByteStore separado.
  3. Al recuperar un hijo, se retorna el padre completo al LLM.

Esto resuelve el problema más común en RAG legal:
  - Decreto 1072 Art. 2.2.4.6.8 tiene 5 parágrafos y 12 numerales.
  - El usuario pregunta por el artículo completo, no el numeral 3.
  - Con chunking normal, el LLM ve fragmentos sin contexto.
  - Con Parent-Child, el LLM ve el artículo completo.

Uso:
    from src.retrieval.parent_child import ParentChildRetriever

    pcr = ParentChildRetriever(
        child_vector_store=vector_store,
        parent_store=InMemoryByteStore(),
    )
    results = pcr.retrieve("¿Qué dice el artículo 2.2.4.6.8?")
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain.storage import InMemoryByteStore

from src.config.logging import get_logger
from src.retrieval.base import BaseRetriever, RetrievalQuery, RetrievalResult

log = get_logger(__name__)


class ParentChildRetriever(BaseRetriever):
    """
    Retriever que busca en hijos pero retorna padres completos.

    Flujo:
      1. Indexar: add_documents(parents, children) — padres en store, hijos en Chroma
      2. Buscar: retrieve(query) — similarity search en hijos → lookup padres
      3. Retornar: documentos padre con metadata del hijo que matcheó
    """

    def __init__(
        self,
        child_vector_store: Any,
        parent_store: InMemoryByteStore | None = None,
        top_k: int = 5,
    ) -> None:
        """
        Args:
            child_vector_store: Chroma VectorStore con chunks hijos indexados.
            parent_store: Almacén de documentos padre. None = crea uno nuevo.
            top_k: Número de padres a retornar.
        """
        self._child_store = child_vector_store
        self._parent_store = parent_store or InMemoryByteStore()
        self._top_k = top_k

        # Cache de mapping hijo→padre: {child_id: parent_id}
        self._child_to_parent: dict[str, str] = {}

    @property
    def retriever_type(self) -> str:
        return "parent_child"

    def is_ready(self) -> bool:
        return self._child_store.is_initialized

    # ── Indexación ────────────────────────────────────────────────────────────

    def add_documents(
        self,
        parent_docs: list[Document],
        child_docs: list[Document],
    ) -> dict[str, int]:
        """
        Indexa documentos parent-child.

        Args:
            parent_docs: Documentos completos (padres).
            child_docs: Chunks pequeños (hijos).

        Cada hijo debe tener en metadata:
            - "parent_id": ID del documento padre
            - o "article_number" / "section_number" para matching automático

        Returns:
            Dict con {"parents_stored": N, "children_indexed": M}
        """
        # 1. Almacenar padres en el byte store
        for doc in parent_docs:
            parent_id = self._doc_id(doc)
            self._parent_store.mset([(parent_id, doc)])

        # 2. Indexar hijos en Chroma
        self._child_store.add_documents(child_docs)

        # 3. Construir mapping hijo→padre
        for child in child_docs:
            child_id = self._doc_id(child)
            parent_id = child.metadata.get("parent_id")

            # Fallback: matching por número de artículo/sección
            if parent_id is None:
                parent_id = self._find_parent_by_metadata(child, parent_docs)

            if parent_id:
                self._child_to_parent[child_id] = parent_id

        log.info(
            "parent_child_indexed",
            parents=len(parent_docs),
            children=len(child_docs),
            mappings=len(self._child_to_parent),
        )

        return {
            "parents_stored": len(parent_docs),
            "children_indexed": len(child_docs),
        }

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, query: RetrievalQuery) -> list[Document]:
        """Busca hijos, retorna padres completos."""
        # 1. Búsqueda en hijos
        child_results = self._child_store.similarity_search(
            query.text,
            k=self._top_k * 3,  # Fetch more children for dedup
            filters=query.filters,
        )

        if not child_results:
            return []

        # 2. Para cada hijo, encontrar su padre y deduplicar
        seen_parents: set[str] = set()
        parent_docs: list[Document] = []

        for child in child_results:
            child_id = self._doc_id(child)
            parent_id = self._child_to_parent.get(child_id)

            # Si no hay mapping, intentar por metadata
            if parent_id is None:
                parent_id = child.metadata.get("parent_id")

            if parent_id is None or parent_id in seen_parents:
                continue

            seen_parents.add(parent_id)

            # Recuperar padre del store
            parent = self._parent_store.mget([parent_id])
            if parent and parent[0] is not None:
                parent_doc = parent[0]
                # Enriquecer metadata con info del hijo que matcheó
                parent_doc.metadata.update({
                    "parent_child": True,
                    "matched_child_source": child.metadata.get("source", ""),
                    "matched_child_chunk": child.metadata.get("chunk_index", ""),
                    "relevance_score": child.metadata.get("relevance_score"),
                })
                parent_docs.append(parent_doc)

            if len(parent_docs) >= self._top_k:
                break

        log.debug(
            "parent_child_retrieved",
            children_searched=len(child_results),
            parents_returned=len(parent_docs),
        )

        return parent_docs

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_parent_by_metadata(
        self,
        child: Document,
        parent_docs: list[Document],
    ) -> str | None:
        """Fallback: encuentra padre por número de artículo/sección en metadata."""
        child_article = child.metadata.get("article_number")
        child_section = child.metadata.get("section_number")
        child_source = child.metadata.get("source", "")

        for pdoc in parent_docs:
            p_article = pdoc.metadata.get("article_number")
            p_section = pdoc.metadata.get("section_number")
            p_source = pdoc.metadata.get("source", "")

            # Match por artículo + fuente
            if child_article and child_article == p_article and child_source == p_source:
                return self._doc_id(pdoc)

            # Match por sección + fuente
            if child_section and child_section == p_section and child_source == p_source:
                return self._doc_id(pdoc)

        return None

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Genera ID único para un documento basado en metadata o content hash."""
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("id")
        if doc_id:
            return str(doc_id)

        # Fallback: hash del contenido + fuente
        source = doc.metadata.get("source", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        if source and chunk_idx != "":
            return f"{source}::chunk_{chunk_idx}"

        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc.page_content[:200]))


def get_parent_child_retriever(
    child_vector_store: Any,
    parent_store: InMemoryByteStore | None = None,
    top_k: int = 5,
) -> ParentChildRetriever:
    """Factory para ParentChildRetriever."""
    return ParentChildRetriever(
        child_vector_store=child_vector_store,
        parent_store=parent_store,
        top_k=top_k,
    )
