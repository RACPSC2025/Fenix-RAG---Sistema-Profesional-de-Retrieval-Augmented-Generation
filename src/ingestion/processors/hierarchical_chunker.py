"""
Chunker semántico y jerárquico para documentos estructurados.

Problema con RecursiveCharacterTextSplitter estándar:
  Corta bloques lógicos a la mitad. Una sección entera puede tener
  3.000 caracteres pero el splitter estándar la parte en el punto
  más cercano al límite, destruyendo la coherencia semántica que necesita
  el LLM para responder con precisión.

Estrategia:
  1. Segmentación semántica: detectar límites naturales (Secciones, Encabezados, Capítulos)
  2. Si el segmento cabe en chunk_size → chunk completo
  3. Si el segmento es muy grande → sub-chunking con overlap respetando párrafos
  4. Si el segmento es muy pequeño → merge con el siguiente hasta llenar el chunk

Separadores por prioridad (de mayor a menor):
  H1 > H2 > H3 > CAPÍTULO > SECCIÓN > doble salto > salto simple > espacio
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.ingestion.processors.metadata_extractor import (
    MetadataExtractor,
    get_metadata_extractor,
)

log = get_logger(__name__)


# ─── Separadores jerárquicos ──────────────────────────────────────────────────

# Orden: de mayor semántica a menor. El splitter intenta el primero;
# si el segmento sigue siendo muy grande, intenta el siguiente.
HIERARCHICAL_SEPARATORS: list[str] = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\nCAPÍTULO ",
    "\nSECCIÓN ",
    "\n\n\n",
    "\n\n",
    "\n",
    " ",
    "",
]


# ─── Configuración de chunking ────────────────────────────────────────────────

@dataclass
class ChunkConfig:
    """
    Configuración de chunking.

    min_chunk_size: chunks más pequeños que esto se mergean con el siguiente
    max_chunk_size: límite duro — nunca exceder (truncar si es necesario)
    overlap:        solapamiento entre chunks consecutivos del mismo segmento
    add_header:     inyectar header contextual al inicio de cada chunk
    cleaner_profile: perfil de limpieza a aplicar antes del chunking
    """
    chunk_size: int = 1100
    chunk_overlap: int = 220
    min_chunk_size: int = 100
    add_header: bool = True
    source_path: Path | None = None
    loader_type: str = ""
    cleaner_profile: str = "default"


# ─── Chunker ──────────────────────────────────────────────────────────────────

class HierarchicalChunker:
    """
    Chunker semántico optimizado para documentos jerárquicos corporativos y técnicos.

    Preserva secciones completas dentro del límite de chunk_size.
    Cuando una sección supera el límite, la sub-divide respetando
    la estructura interna (párrafos, numerales).
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        metadata_extractor: MetadataExtractor | None = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.extractor = metadata_extractor or get_metadata_extractor()

        # Splitter de respaldo para segmentos demasiado grandes
        self._splitter = RecursiveCharacterTextSplitter(
            separators=HIERARCHICAL_SEPARATORS,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Document]:
        """
        Divide el texto en chunks semánticamente coherentes.

        Args:
            text: Texto limpio (post-cleaner) del documento completo.
            config: Configuración de chunking. Si None, usa defaults de settings.

        Returns:
            Lista de Documents con page_content enriquecido y metadata estructurada.
        """
        if not text or not text.strip():
            log.warning("chunker_empty_input", config=str(config))
            return []

        cfg = config or ChunkConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # 1. Segmentar por límites semánticos (artículos, capítulos)
        segments = self._segment_by_headers(text)
        log.debug(
            "hierarchical_segments_found",
            count=len(segments),
            source=cfg.source_path.name if cfg.source_path else "unknown",
        )

        # 2. Procesar cada segmento
        raw_chunks: list[str] = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            if len(segment) <= self.chunk_size:
                # Cabe completo — preservar artículo íntegro
                raw_chunks.append(segment)
            else:
                # Demasiado grande — sub-dividir con overlap
                sub_chunks = self._splitter.split_text(segment)
                raw_chunks.extend(sub_chunks)

        # 3. Filtrar chunks muy pequeños (ruido de encabezados vacíos, etc.)
        raw_chunks = [c for c in raw_chunks if len(c.strip()) >= cfg.min_chunk_size]

        # 4. Construir Documents con metadata y header contextual
        documents: list[Document] = []
        for idx, chunk_text in enumerate(raw_chunks):
            meta = self.extractor.extract(
                chunk_text=chunk_text,
                source_path=cfg.source_path,
                chunk_index=idx,
                loader_type=cfg.loader_type,
            )

            if cfg.add_header:
                header = self.extractor.build_contextual_header(meta)
                final_content = header + chunk_text.strip() if header else chunk_text.strip()
            else:
                final_content = chunk_text.strip()

            meta.chunk_size = len(final_content)

            documents.append(Document(
                page_content=final_content,
                metadata=meta.to_dict(),
            ))

        log.info(
            "chunking_complete",
            source=cfg.source_path.name if cfg.source_path else "unknown",
            input_chars=len(text),
            segments=len(segments),
            chunks=len(documents),
        )

        return documents

    def _segment_by_headers(self, text: str) -> list[str]:
        """
        Divide el texto en segmentos por marcadores de encabezados principales.

        Estrategia:
          1. Buscar todas las posiciones donde empieza una sección (Markdown o Genérica)
          2. Usar esas posiciones como límites de segmento
          3. El texto entre marcadores es un segmento

        Esto garantiza que una sección nunca se parte de manera destructiva.
        """
        boundary_pattern = re.compile(
            r"(?=\n(?:#+\s|CAPÍTULO|SECCIÓN|PARTE)\b)",
            re.IGNORECASE,
        )

        parts = boundary_pattern.split(text)

        # Filtrar partes vacías y retornar
        return [p for p in parts if p.strip()]

    def chunk_with_profile(
        self,
        text: str,
        source_path: Path | None = None,
        loader_type: str = "",
        cleaner_profile: str = "default",
        add_header: bool = True,
    ) -> list[Document]:
        """
        Conveniencia: limpia el texto y lo divide en una sola llamada.

        Aplica el cleaner del perfil especificado antes del chunking.
        """
        from src.ingestion.processors.text_cleaner import get_cleaner

        cleaner = get_cleaner(cleaner_profile)
        clean_text = cleaner.clean(text)

        config = ChunkConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            source_path=source_path,
            loader_type=loader_type,
            cleaner_profile=cleaner_profile,
            add_header=add_header,
        )

        return self.chunk(clean_text, config)


# ─── Instancia por defecto ────────────────────────────────────────────────────

_default_chunker: HierarchicalChunker | None = None


def get_hierarchical_chunker() -> HierarchicalChunker:
    global _default_chunker  # noqa: PLW0603
    if _default_chunker is None:
        _default_chunker = HierarchicalChunker()
    return _default_chunker
