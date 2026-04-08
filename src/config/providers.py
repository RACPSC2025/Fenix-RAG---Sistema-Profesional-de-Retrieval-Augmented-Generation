"""
Providers — factory de LLM, embeddings y verificación de conectividad.

Todos los módulos obtienen sus proveedores desde aquí con lazy init:

    from src.config.providers import get_llm, get_embeddings

    llm = get_llm()                      # Amazon Nova Pro via Bedrock
    embeddings = get_embeddings()         # Titan Embed v2 via Bedrock

Las instancias se cachean tras la primera creación (singleton por proceso).
En producción, esto evita recrear conexiones de Bedrock en cada request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)


# ─── LLM ──────────────────────────────────────────────────────────────────────

def _create_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    *,
    large_context: bool = False,
) -> BaseChatModel:
    """
    Crea una instancia de LLM conectada a AWS Bedrock.

    Args:
        model: Override del modelo. Si None, usa el de settings.
        temperature: Override de temperatura. Si None, usa el de settings.
        max_tokens: Override de max_tokens. Si None, usa el de settings.
        large_context: Si True, usa el modelo de contexto largo.

    Returns:
        ChatBedrock instance lista para invocar.
    """
    from langchain_aws import ChatBedrock  # noqa: PLC0415

    settings = get_settings()

    if large_context:
        model_name = model or settings.bedrock_llm_large_ctx_model
    else:
        model_name = model or settings.bedrock_llm_model

    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens or settings.llm_max_tokens

    # Configurar region y credentials
    bedrock_kwargs: dict[str, Any] = {
        "model_id": model_name,
        "region_name": settings.aws_region,
        "temperature": temp,
        "max_tokens": tokens,
    }

    # Credentials opcionales (si no están en el profile de AWS)
    if settings.aws_access_key_id:
        bedrock_kwargs["aws_access_key_id"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        bedrock_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_session_token:
        bedrock_kwargs["aws_session_token"] = settings.aws_session_token

    llm = ChatBedrock(**bedrock_kwargs)

    log.info(
        "llm_created",
        model=model_name,
        temperature=temp,
        max_tokens=tokens,
        region=settings.aws_region,
    )

    return llm


@lru_cache(maxsize=2)
def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    *,
    large_context: bool = False,
) -> BaseChatModel:
    """
    Retorna una instancia cacheada del LLM.

    Gracias a @lru_cache(maxsize=2), se mantienen dos instancias:
    una para el modelo estándar (Nova Pro) y otra para contexto largo (Nova Lite).

    Args:
        model: Override del modelo.
        temperature: Override de temperatura.
        max_tokens: Override de max_tokens.
        large_context: Si True, usa el modelo de contexto largo.

    Returns:
        ChatBedrock instance cacheada.
    """
    return _create_llm(model=model, temperature=temperature, max_tokens=max_tokens, large_context=large_context)


# ─── Embeddings ───────────────────────────────────────────────────────────────

def _create_embeddings(
    model: str | None = None,
) -> Embeddings:
    """
    Crea una instancia de embeddings conectada a AWS Bedrock.

    Args:
        model: Override del modelo de embeddings. Si None, usa el de settings.

    Returns:
        BedrockEmbeddings instance lista para usar.
    """
    from langchain_aws import BedrockEmbeddings  # noqa: PLC0415

    settings = get_settings()
    model_name = model or settings.bedrock_embeddings_model

    bedrock_kwargs: dict[str, Any] = {
        "model_id": model_name,
        "region_name": settings.aws_region,
    }

    if settings.aws_access_key_id:
        bedrock_kwargs["aws_access_key_id"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        bedrock_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_session_token:
        bedrock_kwargs["aws_session_token"] = settings.aws_session_token

    embeddings = BedrockEmbeddings(**bedrock_kwargs)

    log.info(
        "embeddings_created",
        model=model_name,
        region=settings.aws_region,
    )

    return embeddings


@lru_cache(maxsize=1)
def get_embeddings(model: str | None = None) -> Embeddings:
    """
    Retorna una instancia cacheada de embeddings.

    Args:
        model: Override del modelo.

    Returns:
        BedrockEmbeddings instance cacheada.
    """
    return _create_embeddings(model=model)


# ─── Conectividad ─────────────────────────────────────────────────────────────

def check_bedrock_connectivity() -> dict[str, Any]:
    """
    Verifica que AWS Bedrock sea accesible.

    Usa una llamada mínima (ListFoundationModels) para probar
    la conexión sin consumir tokens ni generar contenido.

    Returns:
        Dict con:
            - bedrock: True/False
            - region: región configurada
            - error: mensaje de error si falla
            - models_available: número de modelos disponibles (si éxito)
    """
    import boto3  # noqa: PLC0415

    settings = get_settings()

    try:
        bedrock_client = boto3.client(
            "bedrock",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            aws_session_token=settings.aws_session_token or None,
        )

        response = bedrock_client.list_foundation_models()
        models = response.get("modelSummaries", [])

        log.info(
            "bedrock_connectivity_ok",
            region=settings.aws_region,
            models=len(models),
        )

        return {
            "bedrock": True,
            "region": settings.aws_region,
            "models_available": len(models),
            "error": None,
        }

    except Exception as exc:
        log.warning(
            "bedrock_connectivity_failed",
            region=settings.aws_region,
            error=str(exc),
        )

        return {
            "bedrock": False,
            "region": settings.aws_region,
            "error": str(exc),
            "models_available": 0,
        }


def clear_provider_cache() -> None:
    """
    Limpia la cache de providers.

    Útil en tests para recrear instancias con configuraciones diferentes.
    """
    get_llm.cache_clear()
    get_embeddings.cache_clear()
