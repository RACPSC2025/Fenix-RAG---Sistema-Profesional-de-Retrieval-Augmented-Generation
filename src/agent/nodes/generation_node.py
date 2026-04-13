"""
Generation node — Re2 condicional + Skill Pack dinámico (Fase 10).

Flujo del nodo:
  1. Leer `active_profile` del estado (viene de initial_state o API en Fase 11)
  2. `SkillRegistry.load_pack(profile)` → INDEX.md como string
  3. Si el INDEX.md tiene contenido → crear SystemMessage y pasar como extra_system
  4. Si está vacío o falla → extra_system=None → degradación graciosa

DISEÑO DEL SYSTEM PROMPT CON PACK:
  El skill pack se inyecta como extra_system ANTES del GENERATION_PROMPT base.
  Esto se maneja en rethinking.py:
    [extra_system: skill pack INDEX.md]  ← rol y especialización (si existe)
    [GENERATION_PROMPT]                  ← restricciones anti-alucinación
    [HumanMessage: contexto + pregunta]  ← query del usuario

  El pack NUNCA puede sobreescribir las reglas anti-alucinación porque viene
  primero en la lista de mensajes.

RESOLUCIÓN DEL PERFIL:
  1. `state["active_profile"]` — pasado por initial_state() o API (Fase 11)
  2. Si está vacío → SkillRegistry.default_profile ("general-dev")
  3. Si el perfil no existe o falla la carga → fallback a default_profile

RE2 CONDICIONAL (se mantiene intacto):
  - grade_score > 0.8 → generate_direct (1 llamada LLM)
  - grade_score 0.5-0.8 → generate_with_rethinking (2 llamadas LLM)
  - grade_score < 0.5 → generate_with_rethinking + advertencia
"""

from __future__ import annotations

from src.agent.metrics import node_timer
from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def _load_skill_pack_for_state(state: AgentState) -> tuple[str, str]:
    """
    Carga el INDEX.md del perfil activo para este estado.

    Args:
        state: AgentState con campo active_profile.

    Returns:
        Tuple (profile_name, pack_content).
        pack_content puede ser "" si el pack no existe o falla la carga.
    """
    from src.agent.skills.registry import get_skill_registry  # noqa: PLC0415

    registry = get_skill_registry()

    # Resolución del perfil: estado → default del registry
    profile = state.get("active_profile", "") or registry.default_profile

    try:
        pack_content = registry.load_pack(profile)
        if not pack_content:
            log.warning(
                "skill_pack_empty",
                profile=profile,
                fallback=registry.default_profile,
            )
            # Intentar fallback al default si el perfil solicitado está vacío
            if profile != registry.default_profile:
                pack_content = registry.load_pack(registry.default_profile)
                profile = registry.default_profile
    except Exception as exc:
        log.warning("skill_pack_load_failed", profile=profile, error=str(exc))
        pack_content = ""

    return profile, pack_content


def generation_node(state: AgentState) -> dict:
    """
    Genera respuesta con:
      - Skill Pack dinámico inyectado como system prompt (Fase 10)
      - Re2 condicional basado en CRAG grade_score

    Modos de generación:
      - grade_score > 0.8 → directo (1 llamada LLM)
      - grade_score 0.5-0.8 → Re2 (2 llamadas LLM)
      - grade_score < 0.5 → Re2 + advertencia (docs marginales)
    """
    from langchain_core.messages import SystemMessage  # noqa: PLC0415
    from src.agent.skills.rethinking import (  # noqa: PLC0415
        generate_direct,
        generate_with_rethinking,
    )

    with node_timer(state, "generation") as timer:
        query = state.get("active_query") or state.get("user_query", "")
        docs = state.get("retrieval_results", [])
        grade_score = state.get("grade_score", 0.0)

        # ── Cargar Skill Pack (Fase 10) ───────────────────────────────────────
        profile_name, pack_content = _load_skill_pack_for_state(state)

        # System message adicional del pack (vacío si falla la carga)
        pack_system_message: SystemMessage | None = None
        if pack_content:
            pack_system_message = SystemMessage(content=pack_content)
            log.info(
                "skill_pack_injected",
                profile=profile_name,
                pack_size=len(pack_content),
                grade_score=grade_score,
            )

        # ── Sin documentos ────────────────────────────────────────────────────
        if not docs:
            timer.update(extra={
                "generation_mode": "no_docs",
                "grade_score": grade_score,
                "skill_profile": profile_name,
            })
            return {
                "draft_answer": "No encontré documentos relevantes para responder tu consulta.",
                "sources": [],
                "generation_mode": "no_docs",
                "active_profile": profile_name,
                **timer.to_state(),
            }

        # ── Re2 condicional por grade_score ───────────────────────────────────
        if grade_score > 0.8:
            # Documentos claramente relevantes → generación directa (1 llamada LLM)
            answer, sources = generate_direct(
                query, docs, extra_system=pack_system_message
            )
            generation_mode = "direct"
            llm_calls = 1

        elif grade_score >= 0.5:
            # Documentos parcialmente relevantes → Re2 dos pasadas (2 llamadas LLM)
            answer, sources = generate_with_rethinking(
                query, docs, extra_system=pack_system_message
            )
            generation_mode = "rethinking"
            llm_calls = 2

        else:
            # Documentos marginales → Re2 + advertencia
            answer, sources = generate_with_rethinking(
                query, docs, extra_system=pack_system_message
            )
            answer = (
                f"{answer}\n\n"
                f"⚠️ *Nota: Los documentos recuperados tienen relevancia limitada "
                f"(score: {grade_score:.2f}). Verifica la información con fuentes adicionales.*"
            )
            generation_mode = "rethinking_low_confidence"
            llm_calls = 2

        timer.update(
            docs_count=len(sources),
            extra={
                "generation_mode": generation_mode,
                "grade_score": grade_score,
                "llm_calls": llm_calls,
                "skill_profile": profile_name,
                "pack_injected": bool(pack_content),
            },
        )

        return {
            "draft_answer": answer,
            "sources": sources,
            "generation_mode": generation_mode,
            "active_profile": profile_name,
            **timer.to_state(),
        }
