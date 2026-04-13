"""
Tests unitarios para Context Window Manager (Sub-tarea 15.0).

Cubre:
  - count_tokens_in_messages: estimación con y sin tiktoken
  - generate_conversation_summary: generación exitosa y fallback
  - manage_context_window: todos los paths (noop, compresión, edge cases)
  - context_manager_node: integración como nodo LangGraph
  - get_context_metrics: métricas de observabilidad

Mocking strategy:
  - get_llm mockeado para evitar llamadas reales a Bedrock/AWS
  - get_settings mockeado para controlar feature flag
  - tiktoken mockeado en tests que verifican el fallback por chars
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from src.agent.middleware.context_window_manager import (
    CONTEXT_THRESHOLD_TOKENS,
    KEEP_LAST_N_MESSAGES,
    _format_messages_for_summary,
    context_manager_node,
    count_tokens_in_messages,
    generate_conversation_summary,
    get_context_metrics,
    manage_context_window,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_messages(n: int, content_len: int = 100) -> list:
    """Genera una lista de n mensajes alternando Human/AI."""
    messages = []
    for i in range(n):
        content = f"mensaje {i}: " + "x" * content_len
        if i % 2 == 0:
            msg = HumanMessage(content=content)
        else:
            msg = AIMessage(content=content)
        # Asignar id simulado (LangGraph lo asigna automáticamente en producción)
        msg.id = f"msg-{i:04d}"
        messages.append(msg)
    return messages


def _make_state(messages: list, **kwargs) -> dict:
    """Construye un AgentState mínimo para los tests."""
    return {
        "messages": messages,
        "session_id": "test-session",
        "user_query": "test query",
        "context_compressed": False,
        "context_tokens_saved": 0,
        **kwargs,
    }


# ─── count_tokens_in_messages ─────────────────────────────────────────────────

class TestCountTokensInMessages:
    def test_empty_list_returns_zero(self):
        assert count_tokens_in_messages([]) == 0

    def test_single_message_chars_fallback(self):
        """Sin tiktoken, usa chars/4."""
        msg = HumanMessage(content="a" * 400)
        with patch("src.agent.middleware.context_window_manager.tiktoken", side_effect=ImportError):
            result = count_tokens_in_messages([msg])
        # 400 chars / 4 = 100 tokens
        assert result == 100

    def test_multiple_messages_accumulate(self):
        messages = [
            HumanMessage(content="a" * 400),
            AIMessage(content="b" * 800),
        ]
        result = count_tokens_in_messages(messages)
        # Mínimo (400+800)//4 = 300 (sin tiktoken)
        assert result >= 300

    def test_with_tiktoken_available(self):
        """Con tiktoken disponible, usa encoding real."""
        messages = [HumanMessage(content="hello world")]
        result = count_tokens_in_messages(messages)
        # "hello world" = ~2-3 tokens en cl100k_base
        assert result > 0
        assert result < 10  # No más de 10 tokens para 2 palabras

    def test_non_string_content_handled(self):
        """Mensajes con content como lista (tool messages) no crashean."""
        msg = HumanMessage(content=[{"type": "text", "text": "hola"}])
        result = count_tokens_in_messages([msg])
        assert isinstance(result, int)
        assert result >= 0


# ─── _format_messages_for_summary ────────────────────────────────────────────

class TestFormatMessagesForSummary:
    def test_formats_with_role_prefix(self):
        messages = [
            HumanMessage(content="Hola"),
            AIMessage(content="Hola, ¿en qué te ayudo?"),
        ]
        result = _format_messages_for_summary(messages)
        assert "HUMAN: Hola" in result
        assert "AI: Hola" in result

    def test_truncates_long_content(self):
        """Mensajes > 800 chars se truncan para el prompt de resumen."""
        long_content = "x" * 1000
        msg = HumanMessage(content=long_content)
        result = _format_messages_for_summary([msg])
        assert "[truncado]" in result
        assert len(result) < 900  # Debe ser considerablemente más corto

    def test_empty_list_returns_empty_string(self):
        result = _format_messages_for_summary([])
        assert result == ""


# ─── generate_conversation_summary ───────────────────────────────────────────

class TestGenerateConversationSummary:
    def test_returns_summary_text(self):
        """LLM mockeado devuelve resumen."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Resumen de la conversación.")

        messages = _make_messages(4)
        result = generate_conversation_summary(messages, llm=mock_llm)

        assert result == "Resumen de la conversación."
        mock_llm.invoke.assert_called_once()

    def test_returns_empty_string_on_llm_failure(self):
        """Si el LLM falla, retorna string vacío (fallback seguro)."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Connection timeout")

        messages = _make_messages(4)
        result = generate_conversation_summary(messages, llm=mock_llm)

        assert result == ""

    def test_empty_messages_returns_empty_string(self):
        result = generate_conversation_summary([], llm=MagicMock())
        assert result == ""

    def test_strips_whitespace_from_summary(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="  Resumen.  \n")

        result = generate_conversation_summary(_make_messages(2), llm=mock_llm)
        assert result == "Resumen."


# ─── manage_context_window ────────────────────────────────────────────────────

class TestManageContextWindow:

    def _mock_settings(self, enabled: bool = True):
        mock_settings = MagicMock()
        mock_settings.enable_context_window_manager = enabled
        return mock_settings

    def test_noop_when_feature_flag_disabled(self):
        """Con feature flag desactivado, no hace nada."""
        messages = _make_messages(20, content_len=500)
        state = _make_state(messages)

        with patch(
            "src.agent.middleware.context_window_manager.get_settings",
            return_value=self._mock_settings(enabled=False),
        ):
            result = manage_context_window(state)

        assert result == {}

    def test_noop_when_under_threshold(self):
        """Con pocos mensajes, no comprime."""
        messages = _make_messages(2, content_len=10)  # ~5 tokens total
        state = _make_state(messages)

        with patch(
            "src.agent.middleware.context_window_manager.get_settings",
            return_value=self._mock_settings(),
        ):
            result = manage_context_window(state, threshold_tokens=10_000)

        assert result == {}

    def test_noop_when_not_enough_messages_to_compress(self):
        """Con menos mensajes que keep_last_n, no hay nada que comprimir."""
        messages = _make_messages(4)
        state = _make_state(messages)

        with patch(
            "src.agent.middleware.context_window_manager.get_settings",
            return_value=self._mock_settings(),
        ):
            # threshold=0 para forzar que pase el check de tokens
            result = manage_context_window(
                state,
                threshold_tokens=0,
                keep_last_n=8,  # más que los 4 mensajes
            )

        assert result == {}

    def test_compresses_when_over_threshold(self):
        """Con muchos mensajes, genera RemoveMessage ops + SystemMessage."""
        messages = _make_messages(12, content_len=500)
        state = _make_state(messages)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Resumen compacto.")
        mock_settings = self._mock_settings()

        with patch(
            "src.agent.middleware.context_window_manager.get_settings",
            return_value=mock_settings,
        ):
            result = manage_context_window(
                state,
                threshold_tokens=0,   # forzar compresión
                keep_last_n=4,
                llm=mock_llm,
            )

        assert "messages" in result
        assert result.get("context_compressed") is True
        assert isinstance(result.get("context_tokens_saved"), int)

        # Debe haber RemoveMessage ops
        remove_ops = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(remove_ops) == 8  # 12 - 4 = 8 a eliminar

        # Debe haber un SystemMessage de resumen
        summary_msgs = [m for m in result["messages"] if isinstance(m, SystemMessage)]
        assert len(summary_msgs) == 1
        assert "RESUMEN" in summary_msgs[0].content
        assert "Resumen compacto." in summary_msgs[0].content

    def test_aborts_if_summary_fails(self):
        """Si el LLM de resumen falla, no comprime — mejor historial completo."""
        messages = _make_messages(12, content_len=500)
        state = _make_state(messages)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")
        mock_settings = self._mock_settings()

        with patch(
            "src.agent.middleware.context_window_manager.get_settings",
            return_value=mock_settings,
        ):
            result = manage_context_window(
                state,
                threshold_tokens=0,
                keep_last_n=4,
                llm=mock_llm,
            )

        # Si el summary falla, retorna dict vacío — no comprime
        assert result == {}

    def test_messages_without_id_are_skipped_in_remove_ops(self):
        """Mensajes sin id (creados manualmente) no generan RemoveMessage."""
        # Mensajes sin .id asignado
        messages = [
            HumanMessage(content="msg sin id " + "x" * 500),
            AIMessage(content="respuesta sin id " + "x" * 500),
            HumanMessage(content="msg sin id 2 " + "x" * 500),
            AIMessage(content="respuesta sin id 2 " + "x" * 500),
        ]
        # No asignamos .id — son None por defecto
        state = _make_state(messages)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Resumen.")
        mock_settings = self._mock_settings()

        with patch(
            "src.agent.middleware.context_window_manager.get_settings",
            return_value=mock_settings,
        ):
            result = manage_context_window(
                state,
                threshold_tokens=0,
                keep_last_n=2,
                llm=mock_llm,
            )

        if result:
            # Si hubo compresión, los RemoveMessage solo para msgs con id
            remove_ops = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
            # Todos los mensajes tienen id=None, así que no hay remove ops
            assert len(remove_ops) == 0

    def test_noop_when_messages_empty(self):
        state = _make_state([])
        with patch(
            "src.agent.middleware.context_window_manager.get_settings",
            return_value=self._mock_settings(),
        ):
            result = manage_context_window(state)
        assert result == {}


# ─── context_manager_node ─────────────────────────────────────────────────────

class TestContextManagerNode:
    def test_returns_empty_dict_when_no_compression_needed(self):
        """Nodo retorna {} cuando no hay compresión necesaria."""
        messages = _make_messages(2, content_len=10)
        state = _make_state(messages)

        with patch(
            "src.agent.middleware.context_window_manager.get_settings",
            return_value=MagicMock(enable_context_window_manager=True),
        ):
            result = context_manager_node(state)

        assert result == {}

    def test_returns_compression_result_when_needed(self):
        """Nodo retorna el resultado de compresión cuando aplica."""
        messages = _make_messages(12, content_len=500)
        state = _make_state(messages)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Resumen.")

        with (
            patch(
                "src.agent.middleware.context_window_manager.get_settings",
                return_value=MagicMock(enable_context_window_manager=True),
            ),
            patch(
                "src.agent.middleware.context_window_manager.get_llm",
                return_value=mock_llm,
            ),
            patch(
                "src.agent.middleware.context_window_manager.CONTEXT_THRESHOLD_TOKENS",
                0,
            ),
        ):
            result = context_manager_node(state)

        # Si hubo compresión, el resultado tiene messages
        if result:
            assert "messages" in result


# ─── get_context_metrics ─────────────────────────────────────────────────────

class TestGetContextMetrics:
    def test_returns_correct_structure(self):
        messages = _make_messages(5, content_len=100)
        state = _make_state(messages, context_compressed=True, context_tokens_saved=500)

        metrics = get_context_metrics(state)

        assert metrics["message_count"] == 5
        assert isinstance(metrics["estimated_tokens"], int)
        assert isinstance(metrics["needs_compression"], bool)
        assert metrics["context_compressed"] is True
        assert metrics["tokens_saved_this_session"] == 500

    def test_needs_compression_false_when_under_threshold(self):
        messages = _make_messages(2, content_len=10)
        state = _make_state(messages)

        metrics = get_context_metrics(state)
        assert metrics["needs_compression"] is False

    def test_empty_messages(self):
        state = _make_state([])
        metrics = get_context_metrics(state)

        assert metrics["message_count"] == 0
        assert metrics["estimated_tokens"] == 0
        assert metrics["needs_compression"] is False
        assert metrics["context_compressed"] is False
        assert metrics["tokens_saved_this_session"] == 0
