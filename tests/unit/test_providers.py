"""Tests del providers — LLM y embeddings factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.providers import (
    get_llm,
    get_embeddings,
    check_bedrock_connectivity,
    clear_provider_cache,
)


class TestGetLLM:
    """Tests del factory de LLM."""

    def setup_method(self):
        clear_provider_cache()

    @patch("src.config.providers.get_settings")
    @patch("langchain_aws.ChatBedrock")
    def test_returns_llm_instance(self, mock_bedrock, mock_settings):
        mock_settings.return_value = MagicMock(
            aws_region="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="",
            bedrock_llm_model="amazon.nova-pro-v1:0",
            llm_temperature=0.0,
            llm_max_tokens=4096,
        )

        llm = get_llm()

        mock_bedrock.assert_called_once()
        assert llm is not None

    @patch("src.config.providers.get_settings")
    @patch("langchain_aws.ChatBedrock")
    def test_caches_result(self, mock_bedrock, mock_settings):
        mock_settings.return_value = MagicMock(
            aws_region="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_session_token="",
            bedrock_llm_model="amazon.nova-pro-v1:0",
            llm_temperature=0.0,
            llm_max_tokens=4096,
        )

        llm1 = get_llm()
        llm2 = get_llm()

        # Misma instancia por lru_cache
        assert llm1 is llm2

    @patch("src.config.providers.get_settings")
    @patch("langchain_aws.ChatBedrock")
    def test_large_context_uses_different_model(self, mock_bedrock, mock_settings):
        mock_settings.return_value = MagicMock(
            aws_region="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_session_token="",
            bedrock_llm_model="amazon.nova-pro-v1:0",
            bedrock_llm_large_ctx_model="amazon.nova-lite-v1:0",
            llm_temperature=0.0,
            llm_max_tokens=4096,
        )

        clear_provider_cache()
        llm = get_llm(large_context=True)

        # Debería usar el modelo de contexto largo
        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["model_id"] == "amazon.nova-lite-v1:0"


class TestGetEmbeddings:
    """Tests del factory de embeddings."""

    def setup_method(self):
        clear_provider_cache()

    @patch("src.config.providers.get_settings")
    @patch("langchain_aws.BedrockEmbeddings")
    def test_returns_embeddings_instance(self, mock_embeddings, mock_settings):
        mock_settings.return_value = MagicMock(
            aws_region="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_session_token="",
            bedrock_embeddings_model="amazon.titan-embed-text-v2:0",
        )

        embeddings = get_embeddings()

        mock_embeddings.assert_called_once()
        assert embeddings is not None


class TestCheckBedrockConnectivity:
    """Tests de verificación de conectividad Bedrock."""

    @patch("src.config.providers.get_settings")
    @patch("boto3.client")
    def test_returns_ok_on_success(self, mock_boto_client, mock_settings):
        mock_settings.return_value = MagicMock(
            aws_region="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_session_token="",
        )

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [{"modelId": "model1"}, {"modelId": "model2"}]
        }
        mock_boto_client.return_value = mock_client

        result = check_bedrock_connectivity()

        assert result["bedrock"] is True
        assert result["models_available"] == 2
        assert result["error"] is None

    @patch("src.config.providers.get_settings")
    @patch("boto3.client")
    def test_returns_error_on_failure(self, mock_boto_client, mock_settings):
        mock_settings.return_value = MagicMock(
            aws_region="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_session_token="",
        )

        mock_boto_client.side_effect = Exception("Connection error")

        result = check_bedrock_connectivity()

        assert result["bedrock"] is False
        assert result["error"] is not None
