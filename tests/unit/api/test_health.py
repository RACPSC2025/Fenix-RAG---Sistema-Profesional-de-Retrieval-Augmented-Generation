"""Tests de los endpoints de Health."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Crea la app FastAPI para tests."""
    with patch("src.api.main.create_app") as mock_create:
        from src.api.main import create_app
        return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthEndpoint:
    """Tests del endpoint /health."""

    @patch("src.api.routes.health.check_db_health")
    @patch("src.api.routes.health.check_vector_store_health")
    @patch("src.api.routes.health.check_bedrock_health")
    def test_health_healthy(self, mock_bedrock, mock_vector, mock_db, client):
        mock_db.return_value = {"status": "ok"}
        mock_vector.return_value = {"status": "ok"}
        mock_bedrock.return_value = {"bedrock": True, "region": "us-east-1"}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @patch("src.api.routes.health.check_db_health")
    def test_health_unhealthy_db(self, mock_db, client):
        mock_db.return_value = {"status": "error", "detail": "Connection refused"}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("degraded", "unhealthy")


class TestReadinessEndpoint:
    """Tests del endpoint /health/ready."""

    def test_readiness_returns_200(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200


class TestLivenessEndpoint:
    """Tests del endpoint /health/live."""

    def test_liveness_returns_200(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
