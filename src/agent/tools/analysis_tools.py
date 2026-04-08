"""
Analysis tools — Herramientas de análisis especializado para el agente ReAct.

Estas tools permiten al agente realizar análisis estructurados sobre
documentos y textos, más allá de la búsqueda y generación estándar.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.config.logging import get_logger

log = get_logger(__name__)


@tool
def specialized_analysis(
    query: str,
    analysis_type: str = "general",
) -> dict[str, Any]:
    """
    Realiza un análisis especializado sobre la query del usuario.

    Tipos de análisis:
      - "general": Análisis estándar con contexto retrieval
      - "compare": Comparación entre múltiples documentos o secciones
      - "extract": Extracción de información específica (fechas, montos, plazos)
      - "summarize": Resumen estructurado del tema consultado

    Args:
        query: Consulta o tema a analizar.
        analysis_type: Tipo de análisis (general, compare, extract, summarize).

    Returns:
        Dict con el análisis estructurado y los hallazgos clave.
    """
    from src.config.providers import get_llm  # noqa: PLC0415

    llm = get_llm()

    analysis_prompts = {
        "general": (
            f"Realiza un análisis detallado del siguiente tema:\n\n{query}\n\n"
            "Proporciona: contexto, puntos clave, implicaciones relevantes."
        ),
        "compare": (
            f"Compara y contrasta los siguientes aspectos del tema:\n\n{query}\n\n"
            "Proporciona: similitudes, diferencias, conclusiones."
        ),
        "extract": (
            f"Extrae información específica del siguiente tema:\n\n{query}\n\n"
            "Proporciona: fechas, montos, plazos, responsables citados."
        ),
        "summarize": (
            f"Genera un resumen estructurado de:\n\n{query}\n\n"
            "Proporciona: resumen ejecutivo, puntos clave, referencias."
        ),
    }

    prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])

    response = llm.invoke(prompt)

    result = {
        "analysis_type": analysis_type,
        "query": query,
        "findings": response.content,
    }

    log.info(
        "analysis_completed",
        type=analysis_type,
        query=query[:60],
    )

    return result


@tool
def extract_key_points(
    text: str,
    max_points: int = 5,
) -> dict[str, Any]:
    """
    Extrae los puntos clave de un texto largo.

    Útil para resúmenes de documentos, artículos o secciones completas.

    Args:
        text: Texto a analizar.
        max_points: Número máximo de puntos a extraer.

    Returns:
        Dict con lista de puntos clave y contexto resumido.
    """
    from src.config.providers import get_llm  # noqa: PLC0415

    llm = get_llm()

    prompt = (
        f"Extrae los {max_points} puntos más importantes del siguiente texto:\n\n"
        f"{text[:5000]}\n\n"
        "Retorna cada punto como un ítem de lista conciso."
    )

    response = llm.invoke(prompt)

    points = [
        line.strip().lstrip("-•* ")
        for line in response.content.split("\n")
        if line.strip() and line.strip()[0] in "-•*"
    ]

    return {
        "key_points": points[:max_points],
        "total_points": len(points),
        "source_length": len(text),
    }


__all__ = ["specialized_analysis", "extract_key_points"]
