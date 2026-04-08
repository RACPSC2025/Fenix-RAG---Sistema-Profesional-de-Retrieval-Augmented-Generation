"""
Chunk repository — CRUD para chunks en PostgreSQL.

Usado para auditoría y correlación de chunks del vector store
con el registro relacional.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.persistence.models import Chunk

log = get_logger(__name__)


async def create_chunk(
    db: AsyncSession,
    *,
    document_id: uuid.UUID,
    chunk_index: int,
    content_preview: str,
    chroma_id: Optional[str] = None,
    article_number: Optional[str] = None,
    page_number: Optional[int] = None,
) -> Chunk:
    """Registra un chunk en la BD para correlación con Chroma."""
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=chunk_index,
        content_preview=content_preview[:500],
        chroma_id=chroma_id,
        article_number=article_number,
        page_number=page_number,
    )
    db.add(chunk)
    return chunk


async def get_chunks_by_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
) -> list[Chunk]:
    """Lista chunks de un documento."""
    stmt = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def count_chunks_by_document(
    db: AsyncSession,
    document_id: uuid.UUID,
) -> int:
    """Cuenta chunks de un documento."""
    from sqlalchemy import func  # noqa: PLC0415

    stmt = select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    result = await db.execute(stmt)
    return result.scalar() or 0
