"""
Generation node — Re2 condicional basado en CRAG grade_score.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def generation_node(state: AgentState) -> dict:
    """
    Genera respuesta con Re2 condicional basado en CRAG grade_score.

    - grade_score > 0.8: Generación directa (1 llamada LLM, bajo costo)
    - grade_score 0.5-0.8: Re2 dos pasadas (2 llamadas LLM, mayor precisión)
    - grade_score < 0.5: Re2 + advertencia (docs marginales, máxima precaución)
    """
    from src.agent.skills.rethinking import generate_direct, generate_with_rethinking  # noqa: PLC0415

    query = state.get("active_query") or state.get("user_query", "")
    docs = state.get("retrieval_results", [])
    grade_score = state.get("grade_score", 0.0)

    if not docs:
        return {
            "draft_answer": "No encontré documentos relevantes para responder tu consulta.",
            "sources": [],
            "generation_mode": "no_docs",
        }

    # Decidir modo de generación basado en CRAG grade_score
    if grade_score > 0.8:
        # Documentos claramente relevantes → generación directa (1 llamada LLM)
        answer, sources = generate_direct(query, docs)
        generation_mode = "direct"

        log.info(
            "generation_direct",
            query=query[:60],
            grade_score=grade_score,
            llm_calls=1,
        )

    elif grade_score >= 0.5:
        # Documentos parcialmente relevantes → Re2 dos pasadas (2 llamadas LLM)
        answer, sources = generate_with_rethinking(query, docs)
        generation_mode = "rethinking"

        log.info(
            "generation_rethinking",
            query=query[:60],
            grade_score=grade_score,
            llm_calls=2,
        )

    else:
        # Documentos marginales → Re2 + advertencia
        answer, sources = generate_with_rethinking(query, docs)
        answer = (
            f"{answer}\n\n"
            f"⚠️ *Nota: Los documentos recuperados tienen relevancia limitada "
            f"(score: {grade_score:.2f}). Verifica la información con fuentes adicionales.*"
        )
        generation_mode = "rethinking_low_confidence"

        log.warning(
            "generation_low_confidence",
            query=query[:60],
            grade_score=grade_score,
            llm_calls=2,
        )

    return {
        "draft_answer": answer,
        "sources": sources,
        "generation_mode": generation_mode,
    }
