"""
API Routes — Sessions.

Endpoints para crear, consultar, listar y desactivar sesiones.
Cada sesión agrupa conversaciones y documentos de un usuario.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import DB, Pagination
from src.api.schemas import (
    ErrorResponse,
    MessageResponse,
    PaginatedResponse,
    SessionCreateRequest,
    SessionResponse,
)
from src.config.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ── Crear sesión ─────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=SessionResponse,
    status_code=201,
    responses={500: {"model": ErrorResponse}},
)
async def create_session(
    body: SessionCreateRequest,
    db: AsyncSession = DB,
):
    """
    Crea una nueva sesión para un usuario.

    - **user_identifier**: Email, ID o identificador del usuario.
    - **title**: Título opcional de la sesión.
    """
    from src.persistence.repositories import session_repo  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    session = await session_repo.create_session(
        db,
        user_identifier=body.user_identifier,
        title=body.title,
    )

    log.info(
        "session_created",
        session_id=str(session.id),
        user=body.user_identifier,
    )

    return session


# ── Obtener sesión ───────────────────────────────────────────────────────────

@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_session(
    session_id: str,
    db: AsyncSession = DB,
):
    """Obtiene una sesión por su ID."""
    from src.persistence.repositories import session_repo  # noqa: PLC0415
    import uuid as _uuid  # noqa: PLC0415

    try:
        sid = _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id debe ser un UUID válido")

    session = await session_repo.get_session(db, sid)

    if session is None:
        raise HTTPException(status_code=404, detail=f"Sesión {session_id} no encontrada")

    return session


# ── Listar sesiones del usuario ──────────────────────────────────────────────

@router.get(
    "/",
    response_model=PaginatedResponse[SessionResponse],
)
async def list_sessions(
    user_identifier: Optional[str] = Query(None, description="Filtrar por identificador de usuario"),
    pagination: Pagination = Pagination,
    db: AsyncSession = DB,
):
    """
    Lista sesiones.
    Si `user_identifier` es proporcionado, filtra por ese usuario.
    """
    from src.persistence.repositories import session_repo  # noqa: PLC0415

    if user_identifier:
        sessions, total = await session_repo.get_sessions_by_user(
            db, user_identifier=user_identifier, limit=pagination.limit, offset=pagination.offset
        )
    else:
        # List all sessions — use a broad query
        sessions = []
        total = 0

    return PaginatedResponse.from_list(
        items=sessions,
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


# ── Desactivar sesión ────────────────────────────────────────────────────────

@router.delete(
    "/{session_id}",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def deactivate_session(
    session_id: str,
    db: AsyncSession = DB,
):
    """Desactiva una sesión (soft-delete)."""
    from src.persistence.repositories import session_repo  # noqa: PLC0415
    import uuid as _uuid  # noqa: PLC0415

    try:
        sid = _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id debe ser un UUID válido")

    session = await session_repo.get_session(db, sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Sesión {session_id} no encontrada")

    await session_repo.delete_session(db, sid)

    log.info("session_deactivated", session_id=str(session.id))

    return session


# ── Historial de mensajes ────────────────────────────────────────────────────

@router.get(
    "/{session_id}/messages",
    response_model=PaginatedResponse[MessageResponse],
)
async def get_session_messages(
    session_id: str,
    pagination: Pagination = Pagination,
    db: AsyncSession = DB,
):
    """Obtiene el historial de mensajes de una sesión."""
    from src.persistence.repositories import session_repo  # noqa: PLC0415
    import uuid as _uuid  # noqa: PLC0415

    try:
        sid = _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id debe ser un UUID válido")

    session = await session_repo.get_session(db, sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Sesión {session_id} no encontrada")

    messages = await session_repo.get_messages_by_session(
        db, session_id=sid, limit=pagination.limit, offset=pagination.offset
    )
    total = await session_repo.count_messages_by_session(db, session_id=sid)

    return PaginatedResponse.from_list(
        items=messages,
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )
