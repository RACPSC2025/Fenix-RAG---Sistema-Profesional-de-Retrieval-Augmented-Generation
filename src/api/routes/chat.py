"""
API Routes — Chat.

Endpoints de consulta al agente RAG:
  - POST /chat/         → Consulta síncrona (espera respuesta completa)
  - POST /chat/stream   → Streaming via Server-Sent Events (SSE)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import DB
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ErrorResponse,
    ReflectionInfo,
    SourceReference,
)
from src.config.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Chat síncrono ─────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def chat(
    body: ChatRequest,
    db: AsyncSession = DB,
):
    """
    Consulta síncrona al agente RAG.

    - **session_id**: UUID de la sesión activa.
    - **query**: Pregunta del usuario (3-4000 caracteres).
    - **max_iterations**: Ciclos máximos de reflexión (1-4).

    Espera a que el agente complete toda la cadena de retrieval →
    generation → reflection y retorna la respuesta final con fuentes.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    # Validar sesión activa
    from src.persistence.repositories.session_repo import get_session  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    try:
        sid = uuid.UUID(body.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id debe ser un UUID válido")

    session = await get_session(db, sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Sesión '{body.session_id}' no encontrada")
    if not session.is_active:
        raise HTTPException(status_code=410, detail=f"Sesión '{body.session_id}' desactivada")

    # Ejecutar agente
    start_ms = time.time() * 1000

    from src.agent.graph import run_agent  # noqa: PLC0415

    result = run_agent(
        user_query=body.query,
        session_id=body.session_id,
        max_iterations=body.max_iterations,
        config={"configurable": {"thread_id": body.session_id}},
    )

    end_ms = time.time() * 1000
    response_time = int(end_ms - start_ms)

    # Parsear sources
    sources = [
        SourceReference(
            source=s.get("source", ""),
            article=s.get("article", ""),
            page=s.get("page", ""),
        )
        for s in (result.get("sources") or [])
    ]

    # Parsear reflection
    reflection = None
    if result.get("reflection"):
        ref = result["reflection"]
        reflection = ReflectionInfo(
            score=ref.get("score", 0.0),
            is_grounded=ref.get("is_grounded", True),
            has_hallucination=ref.get("has_hallucination", False),
            cites_source=ref.get("cites_source", False),
            feedback=ref.get("feedback", ""),
        )

    # Guardar mensajes en BD
    await _save_chat_messages(
        db=db,
        session_id=body.session_id,
        user_query=body.query,
        assistant_answer=result.get("final_answer", ""),
        sources=sources,
        retrieval_strategy=result.get("retrieval_strategy", ""),
        reflection_score=reflection.score if reflection else None,
        iteration_count=result.get("iteration_count", 0),
        response_time_ms=response_time,
    )

    log.info(
        "chat_complete",
        session=body.session_id,
        query_len=len(body.query),
        answer_len=len(result.get("final_answer", "")),
        time_ms=response_time,
    )

    return ChatResponse(
        message_id=str(uuid.uuid4()),
        session_id=body.session_id,
        answer=result.get("final_answer", ""),
        sources=sources,
        retrieval_strategy=result.get("retrieval_strategy", ""),
        iteration_count=result.get("iteration_count", 0),
        reflection=reflection,
        response_time_ms=response_time,
    )


# ── Chat streaming (SSE) ─────────────────────────────────────────────────────

@router.post(
    "/stream",
    response_model=None,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def chat_stream(
    body: ChatRequest,
    db: AsyncSession = DB,
):
    """
    Consulta con streaming via Server-Sent Events.

    Retorna tokens del LLM conforme se generan, seguido de un evento
    `done` con fuentes y metadata completa.

    Format de cada chunk SSE:
        data: {"type": "token", "content": "El"}
        data: {"type": "token", "content": " artículo"}
        data: {"type": "done", "content": "", "metadata": {...}}
    """
    from fastapi import HTTPException  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    try:
        sid = uuid.UUID(body.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id debe ser un UUID válido")

    from src.persistence.repositories.session_repo import get_session  # noqa: PLC0415
    session = await get_session(db, sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Sesión '{body.session_id}' no encontrada")
    if not session.is_active:
        raise HTTPException(status_code=410, detail=f"Sesión '{body.session_id}' desactivada")

    return StreamingResponse(
        _stream_agent_response(body, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_agent_response(
    body: ChatRequest,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """Genera eventos SSE del agente."""
    import uuid  # noqa: PLC0415

    start_ms = time.time() * 1000

    try:
        from src.agent.graph import get_graph  # noqa: PLC0415
        from src.agent.state import initial_state  # noqa: PLC0415

        graph = get_graph()
        state = initial_state(
            user_query=body.query,
            session_id=body.session_id,
            max_iterations=body.max_iterations,
        )

        config = {"configurable": {"thread_id": body.session_id}}

        # Usar stream del grafo para obtener actualizaciones parciales
        final_state = None
        for update in graph.stream(state, config=config):
            # Verificar si hay un draft_answer disponible
            if isinstance(update, dict):
                if "generation" in update:
                    draft = update["generation"].get("draft_answer", "")
                    if draft:
                        chunk = ChatStreamChunk(type="token", content=draft)
                        yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
                final_state = update

        # Si no obtuvimos estado final por streaming, hacer invoke completo
        if final_state is None:
            final_state = graph.invoke(state, config=config)

        end_ms = time.time() * 1000
        response_time = int(end_ms - start_ms)

        # Construir metadata final
        sources = [
            SourceReference(
                source=s.get("source", ""),
                article=s.get("article", ""),
                page=s.get("page", ""),
            )
            for s in (final_state.get("sources") or [])
        ]

        metadata = {
            "session_id": body.session_id,
            "retrieval_strategy": final_state.get("retrieval_strategy", ""),
            "iteration_count": final_state.get("iteration_count", 0),
            "sources": [s.model_dump() for s in sources],
            "response_time_ms": response_time,
        }

        done_chunk = ChatStreamChunk(type="done", content="", metadata=metadata)
        yield f"data: {json.dumps(done_chunk.model_dump(), ensure_ascii=False)}\n\n"

        # Guardar en BD (fire and forget — no bloquear el stream)
        asyncio.create_task(
            _save_chat_messages(
                db=db,
                session_id=body.session_id,
                user_query=body.query,
                assistant_answer=final_state.get("final_answer", ""),
                sources=sources,
                retrieval_strategy=final_state.get("retrieval_strategy", ""),
                iteration_count=final_state.get("iteration_count", 0),
                response_time_ms=response_time,
            )
        )

    except Exception as exc:
        log.error("chat_stream_error", error=str(exc))
        error_chunk = ChatStreamChunk(
            type="error",
            content="Error procesando la consulta.",
            metadata={"error": str(exc)},
        )
        yield f"data: {json.dumps(error_chunk.model_dump(), ensure_ascii=False)}\n\n"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _save_chat_messages(
    db: AsyncSession,
    *,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    sources: list[SourceReference],
    retrieval_strategy: str,
    iteration_count: int,
    response_time_ms: int,
    reflection_score: float | None = None,
) -> None:
    """
    Guarda el par de mensajes (user + assistant) en la BD.

    Se ejecuta después de completar la respuesta.
    """
    import uuid  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    from src.persistence.models import Message  # noqa: PLC0415

    now = datetime.now(timezone.utc)

    # Mensaje del usuario
    user_msg = Message(
        id=uuid.uuid4(),
        session_id=uuid.UUID(session_id),
        role="user",
        content=user_query,
        created_at=now,
    )
    db.add(user_msg)

    # Mensaje del asistente
    assistant_msg = Message(
        id=uuid.uuid4(),
        session_id=uuid.UUID(session_id),
        role="assistant",
        content=assistant_answer,
        sources=[s.model_dump() for s in sources] if sources else None,
        retrieval_strategy=retrieval_strategy,
        iteration_count=iteration_count,
        response_time_ms=response_time_ms,
        created_at=now,
    )
    db.add(assistant_msg)

    # Actualizar contador de mensajes en la sesión
    from src.persistence.repositories import session_repo  # noqa: PLC0415

    await session_repo.increment_session_counters(
        db, session_id=uuid.UUID(session_id), messages_delta=2
    )

    log.info(
        "messages_saved",
        session=session_id,
        user_len=len(user_query),
        assistant_len=len(assistant_answer),
    )
