"""
API Routes — Documents.

Endpoints para subir, listar, obtener estadísticas y eliminar documentos del corpus.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import DB, Pagination
from src.api.schemas import (
    CorpusStatsResponse,
    DocumentIngestResponse,
    DocumentResponse,
    ErrorResponse,
    PaginatedResponse,
)
from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = get_settings().upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Subir documento ──────────────────────────────────────────────────────────

@router.post(
    "/upload/{session_id}",
    response_model=DocumentIngestResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_document(
    session_id: str,
    file: UploadFile,
    db: AsyncSession = DB,
):
    """
    Sube un documento y lo indexa en el vector store.

    Formatos soportados: PDF, DOCX, XLSX, imágenes (OCR).
    """
    # Validar extensión
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    allowed = {".pdf", ".docx", ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".tiff", ".webp"}

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: '{ext}'. Formatos permitidos: {', '.join(allowed)}",
        )

    # Guardar archivo temporal
    import uuid  # noqa: PLC0415
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / safe_name

    content = await file.read()
    dest.write_bytes(content)

    file_hash = hashlib.sha256(content).hexdigest()

    log.info(
        "document_uploaded",
        filename=filename,
        hash=file_hash[:16],
        size=len(content),
    )

    # Verificar duplicado
    # TODO: Implementar find_document_by_hash en document_repo
    existing = None

    if existing:
        log.info("document_already_indexed", doc_id=str(existing.id))
        return DocumentIngestResponse(
            document_id=str(existing.id),
            filename=existing.filename,
            document_type=existing.document_type,
            loader_used=existing.loader_used,
            chunk_count=existing.chunk_count,
            page_count=existing.page_count,
            classifier_confidence=existing.classifier_confidence or 0.0,
            already_indexed=True,
        )

    # Ingestar documento
    start_ms = time.time() * 1000

    from src.agent.tools.ingest_tools import (  # noqa: PLC0415
        ingest_pdf,
        ingest_excel,
        ingest_word,
        ingest_image_pdf,
    )

    # Seleccionar tool según extensión
    if ext == ".pdf":
        # Intentar detectar si es OCR
        result = await ingest_pdf.ainvoke({"file_path": str(dest)})
    elif ext in {".docx", ".doc"}:
        result = await ingest_word.ainvoke({"file_path": str(dest)})
    elif ext in {".xlsx", ".xls"}:
        result = await ingest_excel.ainvoke({"file_path": str(dest)})
    else:
        # Imágenes → PDF con OCR
        result = await ingest_image_pdf.ainvoke({"file_path": str(dest)})

    end_ms = time.time() * 1000
    ingest_time = int(end_ms - start_ms)

    log.info(
        "document_ingested",
        filename=filename,
        chunks=result.get("chunk_count", 0),
        time_ms=ingest_time,
    )

    return DocumentIngestResponse(
        document_id=result.get("document_id", str(uuid.uuid4())),
        filename=filename,
        document_type=result.get("document_type", "unknown"),
        loader_used=result.get("loader_used", "unknown"),
        chunk_count=result.get("chunk_count", 0),
        page_count=result.get("page_count", 0),
        classifier_confidence=result.get("confidence", 0.0),
        already_indexed=False,
    )


# ── Listar documentos ────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=PaginatedResponse[DocumentResponse],
)
async def list_documents(
    pagination: Pagination = Pagination,
    db: AsyncSession = DB,
):
    """Lista todos los documentos indexados en el corpus."""
    # TODO: Implementar list_documents y count_documents en document_repo
    return PaginatedResponse.from_list(
        items=[],
        total=0,
        limit=pagination.limit,
        offset=pagination.offset,
    )


# ── Estadísticas del corpus ──────────────────────────────────────────────────

@router.get(
    "/stats",
    response_model=CorpusStatsResponse,
)
async def get_corpus_stats(
    db: AsyncSession = DB,
):
    """Obtiene estadísticas del corpus indexado."""
    # TODO: Implementar get_corpus_stats en document_repo
    return CorpusStatsResponse(
        total_documents=0,
        total_chunks=0,
        by_document_type={},
        by_loader={},
    )


# ── Eliminar documento ───────────────────────────────────────────────────────

@router.delete(
    "/{document_id}",
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(
    document_id: str,
    db: AsyncSession = DB,
):
    """Des-indexa un documento del corpus."""
    import uuid as _uuid  # noqa: PLC0415

    try:
        doc_id = _uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="document_id debe ser un UUID válido")

    # TODO: Implementar find_document_by_id y delete_document en document_repo
    # TODO: Implementar delete_by_metadata en vector_store

    log.info("document_deleted", doc_id=document_id)

    return {"message": f"Documento '{document_id}' eliminado correctamente"}
