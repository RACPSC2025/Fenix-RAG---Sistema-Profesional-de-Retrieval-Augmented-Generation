"""
Skill tools — Herramientas para que el agente cargue skills on-demand.

Fase 10, sub-tareas 10.7.

El agente ReAct puede invocar estas tools cuando necesita conocimiento
especializado que no está en el INDEX.md del pack activo:
  - `load_skill` — carga el contenido completo de un archivo .md del pack
  - `search_skills` — busca archivos .md por keyword en nombre o contenido

Patrón de uso esperado en el agente:
  User: "¿Cómo implemento CCR reversible en mi RAG?"
  Agent → search_skills("ai-rag-engineer", "CCR")
       → ["Rag_Mastery/stage7-memory.md", "advanced_compression.md"]
  Agent → load_skill("ai-rag-engineer", "Rag_Mastery/stage7-memory.md")
       → contenido completo del archivo con el patrón CCR

Diseño:
  - Siguen exactamente el patrón de memory_tools.py y search_tools.py:
    @tool, docstring descriptivo, try/except con fallback, retorno dict
  - No tienen estado interno — todo se delega a SkillRegistry singleton
  - El active_profile se pasa explícitamente para no depender del estado
    del grafo (las tools son stateless por diseño de LangGraph)
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.config.logging import get_logger

log = get_logger(__name__)


@tool
def load_skill(
    profile: str,
    skill_file: str,
) -> dict[str, Any]:
    """
    Carga el contenido completo de un archivo de skill dentro de un pack.

    Usar cuando el agente necesita conocimiento específico que no está
    en el INDEX.md del perfil activo. Por ejemplo, para leer el patrón
    completo de CRAG, o los detalles de LangGraph checkpointing.

    Args:
        profile: Nombre del pack de skills (ej: "ai-rag-engineer", "general-dev").
                 Usar el mismo valor que `active_profile` del estado.
        skill_file: Nombre del archivo .md a cargar
                    (ej: "crag.md", "Rag_Mastery/stage7-memory.md").
                    Usar search_skills primero si no conoces el nombre exacto.

    Returns:
        Dict con {found, profile, skill_file, content, size_chars}.
    """
    try:
        from src.agent.skills.registry import get_skill_registry  # noqa: PLC0415

        registry = get_skill_registry()
        content = registry.load_skill(profile, skill_file)

        if content:
            log.info(
                "skill_loaded_via_tool",
                profile=profile,
                file=skill_file,
                size=len(content),
            )
            return {
                "found": True,
                "profile": profile,
                "skill_file": skill_file,
                "content": content,
                "size_chars": len(content),
            }

        log.debug(
            "skill_not_found_via_tool",
            profile=profile,
            file=skill_file,
        )
        return {
            "found": False,
            "profile": profile,
            "skill_file": skill_file,
            "content": "",
            "size_chars": 0,
            "hint": f"Usa search_skills('{profile}', '<keyword>') para encontrar archivos disponibles.",
        }

    except Exception as exc:
        log.error("load_skill_tool_failed", profile=profile, file=skill_file, error=str(exc))
        return {
            "found": False,
            "profile": profile,
            "skill_file": skill_file,
            "content": "",
            "size_chars": 0,
            "error": str(exc),
        }


@tool
def search_skills(
    profile: str,
    query: str,
) -> dict[str, Any]:
    """
    Busca archivos de skill dentro de un pack por keyword.

    Busca en el nombre del archivo Y en los primeros 500 caracteres
    del contenido. Útil para descubrir qué skills están disponibles
    antes de cargar una con load_skill.

    Args:
        profile: Nombre del pack de skills (ej: "ai-rag-engineer", "general-dev").
        query: Término de búsqueda (case-insensitive, substring match).
               Ejemplos: "CRAG", "langraph", "testing", "async", "FastAPI".

    Returns:
        Dict con {found, profile, query, matches, total}.
        `matches` es lista de rutas relativas al directorio del pack.
    """
    try:
        from src.agent.skills.registry import get_skill_registry  # noqa: PLC0415

        registry = get_skill_registry()
        matches = registry.search_skills(profile, query)

        log.info(
            "skills_searched_via_tool",
            profile=profile,
            query=query,
            total_matches=len(matches),
        )

        return {
            "found": len(matches) > 0,
            "profile": profile,
            "query": query,
            "matches": matches,
            "total": len(matches),
            "hint": (
                f"Usa load_skill('{profile}', '<archivo>') para cargar el contenido completo."
                if matches else
                f"No se encontraron skills con '{query}' en el pack '{profile}'. "
                f"Prueba con un término más general."
            ),
        }

    except Exception as exc:
        log.error("search_skills_tool_failed", profile=profile, query=query, error=str(exc))
        return {
            "found": False,
            "profile": profile,
            "query": query,
            "matches": [],
            "total": 0,
            "error": str(exc),
        }


@tool
def list_available_profiles() -> dict[str, Any]:
    """
    Lista todos los perfiles de skill packs disponibles en el sistema.

    Usar cuando el usuario pregunta qué perfiles están disponibles
    o cuando se necesita validar que un perfil existe antes de usarlo.

    Returns:
        Dict con {profiles, total, default_profile}.
    """
    try:
        from src.agent.skills.registry import get_skill_registry  # noqa: PLC0415

        registry = get_skill_registry()
        profiles = registry.available_profiles
        default = registry.default_profile

        return {
            "profiles": profiles,
            "total": len(profiles),
            "default_profile": default,
        }

    except Exception as exc:
        log.error("list_profiles_tool_failed", error=str(exc))
        return {
            "profiles": [],
            "total": 0,
            "default_profile": "general-dev",
            "error": str(exc),
        }


__all__ = ["load_skill", "search_skills", "list_available_profiles"]
