"""
Reflection node — auto-evaluación de la respuesta generada.

CONTRATO DE ESTADO:
  Escribe → `reflection_route`, `reflection`, `final_answer`, `iteration_count`
  NO escribe → `route` (ese campo es de document_router y supervisor)
  route_after_reflection en graph.py lee `reflection_route`, NO `route`

Los tres paths de retorno (válido, retry, agotado) escriben a `reflection_route`.
Esto aísla la decisión de reflexión del campo `route` que usan CRAG y supervisor,
eliminando el riesgo de colisión de estado.
"""

from __future__ import annotations

from src.agent.metrics import node_timer
from src.agent.state import AgentState, ReflectionOutput
from src.config.logging import get_logger

log = get_logger(__name__)


def reflection_node(state: AgentState) -> dict:
    """
    Auto-evaluación de la respuesta generada.

    Validación rule-based primero (sin costo de LLM).
    Si es válida        → final_answer con reflection_route="END".
    Si no válida+retry  → reformular con reflection_route="retrieval".
    Si iteraciones end  → borrador con advertencia y reflection_route="END".
    """
    from src.agent.skills.answer_validator import AnswerValidatorSkill  # noqa: PLC0415
    from src.agent.skills.query_transformer import QueryTransformer  # noqa: PLC0415

    with node_timer(state, "reflection") as timer:
        draft = state.get("draft_answer", "")
        query = state.get("user_query", "")
        docs = state.get("retrieval_results", [])
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 2)

        validator = AnswerValidatorSkill()
        validation = validator.validate(draft, docs, query)

        # ── Path 1: respuesta válida ───────────────────────────────────────
        if validation.is_valid:
            timer.update(extra={
                "validation_score": validation.confidence,
                "reflection_route": "END",
                "reason": "valid_response",
            })
            return {
                "final_answer": draft,
                "reflection": ReflectionOutput(
                    score=validation.confidence,
                    is_grounded=True,
                    has_hallucination=False,
                    cites_source=True,
                    feedback="Respuesta válida",
                    reformulated_query="",
                ),
                "reflection_route": "END",      # ← campo exclusivo de reflection
                "iteration_count": iteration + 1,
                **timer.to_state(),
            }

        # ── Path 2: no válida, quedan iteraciones → reformular ─────────────
        if iteration < max_iter:
            transformer = QueryTransformer()
            reformulated = transformer.rewrite(query)

            timer.update(extra={
                "validation_score": validation.confidence,
                "reflection_route": "retrieval",
                "reason": "invalid_response_retry",
                "violations": validation.violations[:3] if validation.violations else [],
            })
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
                "reflection_route": "retrieval",  # ← campo exclusivo de reflection
                "iteration_count": iteration + 1,
                **timer.to_state(),
            }

        # ── Path 3: iteraciones agotadas → borrador con advertencia ────────
        timer.update(extra={
            "validation_score": validation.confidence,
            "violations": len(validation.violations),
            "reflection_route": "END",
            "reason": "iterations_exhausted",
        })
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
            "reflection_route": "END",          # ← campo exclusivo de reflection
            "iteration_count": iteration + 1,
            **timer.to_state(),
        }
