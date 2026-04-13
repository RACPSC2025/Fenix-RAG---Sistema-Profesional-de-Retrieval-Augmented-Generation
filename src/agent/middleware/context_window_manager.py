"""
Context Window Manager — Sub-tarea 15.0.

Gestiona el crecimiento del historial de mensajes (`messages`) en sesiones largas.

PROBLEMA QUE RESUELVE:
  El campo `messages: Annotated[list[BaseMessage], add_messages]` en AgentState
  es acumulativo por diseño de LangGraph — nunca se trunca automáticamente.
  En sesiones largas (10+ turnos con documentos extensos), este campo puede
  consumir 15,000-40,000 tokens por invocación, superando cualquier ganancia
  de otras optimizaciones de tokens.

ESTRATEGIA: Summary Buffer Memory
  1. Contar tokens del historial en cada invocación.
  2. Si supera `CONTEXT_THRESHOLD`, resumir los mensajes más viejos con un LLM.
  3. Reemplazarlos con un SystemMessage de resumen usando `RemoveMessage`.
  4. Mantener siempre los últimos `KEEP_LAST_N` mensajes intactos (contexto inmediato).

INTEGRACIÓN EN EL GRAFO:
  - Nodo `context_manager` insertado DESPUÉS de `reflection` y ANTES de `__end__`.
  - También se llama al inicio de cada invocación desde `run_agent` para limpiar
    el contexto antes de que el grafo procese la siguiente query.

FEATURE FLAG:
  Controlado por `settings.enable_context_window_manager` (default: True).
  Sin riesgo — si falla, el grafo continúa con el historial completo.

PATRÓN LANGGRAPH:
  RemoveMessage es el operador oficial de LangGraph para eliminar mensajes
  del historial. Junto con `add_messages`, permite control granular del estado
  de conversación sin hackear el operador acumulativo.

Referencias:
  - LangGraph docs: "How to delete messages"
  - LangGraph memory/summary.ipynb — ConversationSummaryBufferMemory pattern
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage

from src.config.logging import get_logger
from src.config.providers import get_llm

log = get_logger(__name__)

# ─── Constantes de configuración ──────────────────────────────────────────────

# Umbral de tokens antes de activar la compresión.
# 8,000 tokens ≈ ~6,000 palabras — más que suficiente para contexto inmediato.
# Con Sonnet 3.5 (200k ctx), este umbral es conservador y deja ~190k para docs.
CONTEXT_THRESHOLD_TOKENS: int = 8_000

# Cuántos mensajes recientes se preservan SIN resumir.
# Los últimos 4 turnos (8 mensajes: 4 human + 4 AI) siempre se mantienen intactos.
# Esto garantiza que el LLM tenga acceso al contexto conversacional inmediato.
KEEP_LAST_N_MESSAGES: int = 8

# Máximo de tokens que puede ocupar el resumen generado.
# Si el resumen es demasiado largo, no tiene sentido comprimir.
SUMMARY_MAX_TOKENS: int = 500

# Prompt para el LLM que genera el resumen.
SUMMARY_PROMPT = (
    "Eres un asistente que resume conversaciones de manera concisa y precisa.\n\n"
    "Resume la siguiente conversación en un párrafo compacto. "
    "Preserva: decisiones tomadas, documentos mencionados, queries previas "
    "y respuestas clave. Omite saludos, confirmaciones y texto de relleno.\n\n"
    "Límite estricto: máximo 150 palabras. Solo el resumen, sin introducción.\n\n"
    "Conversación a resumir:\n{conversation}"
)


# ─── Conteo de tokens ─────────────────────────────────────────────────────────

def count_tokens_in_messages(messages: list[BaseMessage]) -> int:
    """
    Cuenta tokens aproximados en la lista de mensajes.

    Usa tiktoken si está disponible (más preciso).
    Fallback: estimación por caracteres (1 token ≈ 4 chars).

    Args:
        messages: Lista de BaseMessage del AgentState.

    Returns:
        Estimación del número de tokens.
    """
    total_chars = sum(
        len(msg.content) if isinstance(msg.content, str) else
        sum(len(str(block)) for block in msg.content)
        for msg in messages
    )

    try:
        import tiktoken  # noqa: PLC0415
        # cl100k_base es compatible con Claude (aproximación)
        enc = tiktoken.get_encoding("cl100k_base")
        total_text = " ".join(
            msg.content if isinstance(msg.content, str) else str(msg.content)
            for msg in messages
        )
        return len(enc.encode(total_text))
    except (ImportError, Exception):
        # Fallback: estimación chars / 4
        return total_chars // 4


# ─── Generación del resumen ───────────────────────────────────────────────────

def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    """
    Formatea mensajes en texto legible para el LLM resumidor.

    Args:
        messages: Lista de mensajes a resumir.

    Returns:
        Texto formateado con prefijos de rol.
    """
    lines: list[str] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        role = type(msg).__name__.replace("Message", "").upper()
        # Truncar mensajes muy largos (docs, tool outputs) para el prompt de resumen
        if len(content) > 800:
            content = content[:800] + "... [truncado]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def generate_conversation_summary(
    messages: list[BaseMessage],
    llm: Any | None = None,
) -> str:
    """
    Genera un resumen conciso de los mensajes dados.

    Args:
        messages: Lista de mensajes a resumir (los más viejos).
        llm: LLM instance. None = usa el default de providers.

    Returns:
        Resumen en texto plano. Empty string si falla.
    """
    if not messages:
        return ""

    llm = llm or get_llm(temperature=0)
    conversation_text = _format_messages_for_summary(messages)

    prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()
        log.info(
            "context_summary_generated",
            messages_summarized=len(messages),
            summary_len=len(summary),
            summary_tokens=len(summary) // 4,
        )
        return summary
    except Exception as exc:
        log.warning("context_summary_failed", error=str(exc))
        return ""  # Fallback: no comprimir si falla


# ─── Función principal ────────────────────────────────────────────────────────

def manage_context_window(
    state: dict,
    threshold_tokens: int = CONTEXT_THRESHOLD_TOKENS,
    keep_last_n: int = KEEP_LAST_N_MESSAGES,
    llm: Any | None = None,
) -> dict:
    """
    Evalúa y comprime el historial de mensajes si supera el umbral.

    FLUJO:
      1. Contar tokens del historial actual.
      2. Si está bajo el umbral → no hacer nada (return {}).
      3. Separar mensajes en: [viejos] + [últimos keep_last_n].
      4. Resumir los viejos con el LLM.
      5. Construir lista de RemoveMessage para los viejos.
      6. Insertar SystemMessage de resumen al inicio del historial comprimido.
      7. Retornar dict con los cambios — LangGraph aplica add_messages.

    Args:
        state: AgentState dict con campo `messages`.
        threshold_tokens: Tokens antes de activar compresión.
        keep_last_n: Mensajes recientes a preservar sin resumir.
        llm: LLM instance para generar resúmenes.

    Returns:
        Dict vacío (sin cambios) o dict con `messages` conteniendo
        RemoveMessage ops + nuevo SystemMessage de resumen.
    """
    from src.config.settings import get_settings  # noqa: PLC0415
    settings = get_settings()

    # Feature flag — default True
    if not getattr(settings, "enable_context_window_manager", True):
        return {}

    messages: list[BaseMessage] = state.get("messages", [])

    if not messages:
        return {}

    # Paso 1: Contar tokens actuales
    current_tokens = count_tokens_in_messages(messages)

    log.debug(
        "context_window_check",
        messages_count=len(messages),
        estimated_tokens=current_tokens,
        threshold=threshold_tokens,
    )

    # Paso 2: ¿Está bajo el umbral? No hacer nada.
    if current_tokens <= threshold_tokens:
        return {}

    # Paso 3: Separar historial
    # Si hay pocos mensajes, no hay nada que comprimir
    if len(messages) <= keep_last_n:
        log.debug(
            "context_window_skip",
            reason="not_enough_messages_to_compress",
            total=len(messages),
            keep_last_n=keep_last_n,
        )
        return {}

    messages_to_summarize = messages[:-keep_last_n]
    messages_to_keep = messages[-keep_last_n:]

    log.info(
        "context_window_compression_start",
        total_messages=len(messages),
        to_summarize=len(messages_to_summarize),
        to_keep=len(messages_to_keep),
        current_tokens=current_tokens,
        threshold=threshold_tokens,
    )

    # Paso 4: Generar resumen de los mensajes viejos
    summary_text = generate_conversation_summary(messages_to_summarize, llm=llm)

    if not summary_text:
        # Si el resumen falla, no comprimir — mejor historial completo que vacío
        log.warning(
            "context_window_compression_aborted",
            reason="summary_generation_failed",
        )
        return {}

    # Paso 5: Construir ops de eliminación para mensajes viejos
    # LangGraph usa el id del mensaje para identificar cuál eliminar.
    # Si el mensaje no tiene id (mensajes creados manualmente), RemoveMessage
    # lo ignora silenciosamente — safe by design.
    remove_ops = [
        RemoveMessage(id=msg.id)
        for msg in messages_to_summarize
        if hasattr(msg, "id") and msg.id is not None
    ]

    # Paso 6: Crear SystemMessage de resumen
    summary_message = SystemMessage(
        content=(
            f"[RESUMEN DE CONVERSACIÓN ANTERIOR]\n"
            f"{summary_text}\n"
            f"[FIN DEL RESUMEN — Los últimos {keep_last_n} mensajes siguen completos]"
        )
    )

    # Calcular tokens estimados post-compresión para logging
    kept_tokens = count_tokens_in_messages(messages_to_keep)
    summary_tokens = len(summary_text) // 4
    tokens_saved = current_tokens - kept_tokens - summary_tokens

    log.info(
        "context_window_compression_complete",
        messages_removed=len(remove_ops),
        messages_kept=len(messages_to_keep),
        tokens_before=current_tokens,
        tokens_after=kept_tokens + summary_tokens,
        tokens_saved=tokens_saved,
        compression_ratio=round(tokens_saved / max(current_tokens, 1), 2),
    )

    # Paso 7: Retornar cambios
    # add_messages aplicará: primero las ops de remove, luego agrega el summary.
    return {
        "messages": remove_ops + [summary_message],
        "context_compressed": True,
        "context_tokens_saved": tokens_saved,
    }


# ─── Nodo del grafo ───────────────────────────────────────────────────────────

def context_manager_node(state: dict) -> dict:
    """
    Nodo LangGraph para gestión del contexto de conversación.

    Posición en el grafo:
      reflection → context_manager → __end__
                                  ↑
                    Se ejecuta después de cada ciclo completo.

    También puede llamarse como middleware antes de run_agent
    para limpiar el estado antes de una nueva query.

    Args:
        state: AgentState completo.

    Returns:
        Dict con cambios de estado (mensajes comprimidos) o vacío.
    """
    result = manage_context_window(state)

    if result:
        log.info(
            "context_manager_node_compressed",
            tokens_saved=result.get("context_tokens_saved", 0),
        )
    else:
        log.debug("context_manager_node_noop")

    return result


# ─── Métricas de contexto ─────────────────────────────────────────────────────

def get_context_metrics(state: dict) -> dict[str, Any]:
    """
    Retorna métricas del estado actual del contexto.

    Útil para el TokenCounter (15.1) y para debugging.

    Args:
        state: AgentState dict.

    Returns:
        Dict con message_count, estimated_tokens, needs_compression.
    """
    messages = state.get("messages", [])
    token_count = count_tokens_in_messages(messages)

    return {
        "message_count": len(messages),
        "estimated_tokens": token_count,
        "needs_compression": token_count > CONTEXT_THRESHOLD_TOKENS,
        "context_compressed": state.get("context_compressed", False),
        "tokens_saved_this_session": state.get("context_tokens_saved", 0),
    }
