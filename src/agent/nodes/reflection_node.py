"""
Reflection node — auto-evaluación de la respuesta generada.
"""

from __future__ import annotations

from src.agent.state import AgentState, ReflectionOutput
from src.config.logging import get_logger

log = get_logger(__name__)


def reflection_node(state: AgentState) -> dict:
    """
    Auto-evaluación de la respuesta generada.

    Validación rule-based primero (sin costo de LLM).
    Si es válida → final_answer con route=END.
    Si no es válida y quedan iteraciones → reformular y re-retrieval.
    Si iteraciones agotadas → borrador con advertencia.
    """
    from src.agent.skills.answer_validator import AnswerValidatorSkill  # noqa: PLC0415

    draft = state.get("draft_answer", "")
    query = state.get("user_query", "")
    docs = state.get("retrieval_results", [])
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 2)

    # Validación rule-based primero (sin costo de LLM)
    validator = AnswerValidatorSkill()
    validation = validator.validate(draft, docs, query)

    if validation.is_valid:
        return {
            "final_answer": draft,
            "reflection": ReflectionOutput(
                score=validation.confidence,
                is_grounded=validation.is_valid,
                has_hallucination=False,
                cites_source=True,
                feedback="Respuesta válida",
                reformulated_query="",
            ),
            "route": "END",
            "iteration_count": iteration + 1,
        }

    # Si no es válida y quedan iteraciones → reformular
    if iteration < max_iter:
        # Usar query_transformer para reformular
        from src.agent.skills.query_transformer import QueryTransformer  # noqa: PLC0415
        transformer = QueryTransformer()
        reformulated = transformer.rewrite(query)

        return {
            "active_query": reformulated,
            "reflection": ReflectionOutput(
                score=validation.confidence,
                is_grounded=False,
                has_hallucination=bool(validation.violations),
                cites_source=False,
                feedback=validation.violations[0] if validation.violations else "Respuesta inválida",
                reformulated_query=reformulated,
            ),
            "route": "retrieval",
            "iteration_count": iteration + 1,
        }

    # Iteraciones agotadas → usar borrador con advertencia
    return {
        "final_answer": draft + "\n\n⚠️ Nota: Esta respuesta puede estar incompleta.",
        "reflection": ReflectionOutput(
            score=validation.confidence,
            is_grounded=False,
            has_hallucination=False,
            cites_source=False,
            feedback="Iteraciones agotadas",
            reformulated_query="",
        ),
        "route": "END",
        "iteration_count": iteration + 1,
    }
