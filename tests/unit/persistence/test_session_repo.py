"""Tests del session_repo — repositorio de sesiones y mensajes."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.persistence.repositories import session_repo


# ─── create_session ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCreateSession:
    """Tests de creación de sesiones."""

    async def test_creates_session(self):
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.flush = AsyncMock()

        session = await session_repo.create_session(
            mock_db,
            user_identifier="test@example.com",
            title="Test Session",
        )

        assert session.user_identifier == "test@example.com"
        assert session.title == "Test Session"
        assert session.is_active is True
        mock_db.add.assert_called_once()


# ─── get_session ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGetSession:
    """Tests de obtención de sesiones."""

    async def test_returns_session_when_found(self):
        mock_db = AsyncMock(spec=AsyncSession)
        session_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock(
            id=session_id,
            user_identifier="test@example.com",
            is_active=True,
        )
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await session_repo.get_session(mock_db, session_id)

        assert result is not None
        mock_db.execute.assert_called_once()

    async def test_returns_none_when_not_found(self):
        mock_db = AsyncMock(spec=AsyncSession)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await session_repo.get_session(mock_db, uuid.uuid4())

        assert result is None


# ─── deactivate_session ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestDeactivateSession:
    """Tests de desactivación de sesiones."""

    async def test_deactivates_session(self):
        mock_db = AsyncMock(spec=AsyncSession)

        mock_session = MagicMock(is_active=True)
        result = await session_repo.deactivate_session(mock_db, mock_session)

        assert mock_session.is_active is False
        assert result is True

    async def test_does_not_deactivate_already_inactive(self):
        mock_db = AsyncMock(spec=AsyncSession)

        mock_session = MagicMock(is_active=False)
        result = await session_repo.deactivate_session(mock_db, mock_session)

        assert result is False
