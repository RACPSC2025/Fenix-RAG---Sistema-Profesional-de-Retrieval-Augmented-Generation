"""
Rethinking (Re2) Generation — Two-pass answer generation.

Flujo de dos lecturas:
  1. **Primera lectura:** el LLM lee el contexto e identifica los pasajes
     clave que responden la consulta.
  2. **Segunda lectura:** el LLM genera la respuesta final usando los
     pasajes identificados como guía, citando fuentes específicas.

Beneficio: mejora la precisión en respuestas que requieren conectar
información dispersa en múltiples documentos.

Uso:
    from src.agent.skills.rethinking import generate_with_rethinking

    answer, sources = generate_with_rethinking(
        query="¿Cuáles son los requisitos del sistema de gestión?",
        documents=retrieved_docs,
    )
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.logging import get_logger
from src.config.providers import get_llm

log = get_logger(__name__)


# ─── Prompts ─────────────────────────────────────────────────────────────────

PASSAGE_IDENTIFICATION_PROMPT = SystemMessage(
    content=(
        "Eres un experto analizando documentos para encontrar información relevante.\n"
        "Tu única tarea es identificar los pasajes EXACTOS del contexto que responden "
        "la consulta del usuario.\n\n"
        "Reglas:\n"
        "- Extrae los fragmentos relevantes textualmente\n"
        "- Indica la fuente de cada fragmento\n"
        "- No generes una respuesta completa, solo identifica los pasajes\n"
        "- Si no hay información relevante, indícalo explícitamente"
    )
)

ANSWER_GENERATION_PROMPT = SystemMessage(
    content=(
        "Eres un asistente experto y preciso.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. Responde ÚNICAMENTE basándote en los documentos proporcionados.\n"
        "2. SIEMPRE cita la fuente exacta de donde obtienes la información.\n"
        "3. Si la información NO está en el contexto, indícalo explícitamente.\n"
        "4. NUNCA inventes o extrapoles más allá del texto.\n"
        "5. Si hay ambigüedad, indica las distintas interpretaciones posibles.\n\n"
        "FORMATO DE RESPUESTA:\n"
        "- Respuesta directa y concisa\n"
        "- Cita textual o parafraseada con fuente\n"
        "- Advertencia de limitaciones si aplica"
    )
)


# ─── Función principal ───────────────────────────────────────────────────────

def generate_with_rethinking(
    query: str,
    documents: list[Document],
    llm: Any | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """
    Genera respuesta con Re-Reading de dos pasadas.

    Args:
        query: Consulta del usuario.
        documents: Documentos recuperados para el contexto.
        llm: LLM instance. None = usa el default de providers.

    Returns:
        Tuple de (respuesta_final, lista_de_fuentes).
    """
    if not documents:
        return "No se encontraron documentos relevantes para esta consulta.", []

    llm = llm or get_llm(temperature=0)

    # Construir contexto completo
    context = _build_context(documents)

    # ── Primera lectura: identificar pasajes clave ────────────────────────
    passage_response = llm.invoke([
        PASSAGE_IDENTIFICATION_PROMPT,
        HumanMessage(content=f"Consulta: {query}\n\nContexto:\n{context}"),
    ])
    key_passages = passage_response.content.strip()

    log.debug(
        "rethinking_pass1_complete",
        passages_len=len(key_passages),
        docs_count=len(documents),
    )

    # ── Segunda lectura: generar respuesta final ──────────────────────────
    answer_response = llm.invoke([
        ANSWER_GENERATION_PROMPT,
        HumanMessage(content=(
            f"Consulta: {query}\n\n"
            f"Pasajes relevantes identificados:\n{key_passages}\n\n"
            f"Contexto completo de referencia:\n{context}"
        )),
    ])
    final_answer = answer_response.content.strip()

    # Extraer fuentes mencionadas en la respuesta
    sources = _extract_sources(final_answer, documents)

    log.info(
        "rethinking_generation_complete",
        query=query[:60],
        answer_len=len(final_answer),
        sources_count=len(sources),
    )

    return final_answer, sources


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_context(documents: list[Document]) -> str:
    """Construye el contexto concatenando documentos con metadata."""
    parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "N/A")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        article = doc.metadata.get("article_number", "")
        section = doc.metadata.get("section_number", "")

        header_parts = [f"Fuente: {source}"]
        if article:
            header_parts.append(f"Artículo: {article}")
        if section:
            header_parts.append(f"Sección: {section}")
        header_parts.append(f"Chunk: {chunk_idx}")

        header = " | ".join(header_parts)
        parts.append(f"[Documento {i + 1} — {header}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def _extract_sources(
    answer: str,
    documents: list[Document],
) -> list[dict[str, str]]:
    """
    Extrae fuentes mencionadas en la respuesta.

    Busca patrones como "Artículo X", "Sección X", nombres de fuente, etc.
    """
    sources: list[dict[str, str]] = []
    seen: set[str] = set()

    # Patrones de extracción
    patterns = [
        (r"[Aa]rt[íi]culo\s+([\d.]+[\w]*)", "article"),
        (r"[Ss]ecci[óo]n\s+([\d.]+[\w]*)", "section"),
        (r"[Cc]ap[íi]tulo\s+([\w\s]+?)(?:\n|$)", "chapter"),
    ]

    for pattern, field_type in patterns:
        matches = re.findall(pattern, answer)
        for match in matches:
            key = f"{field_type}:{match.strip()}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "type": field_type,
                    "value": match.strip(),
                })

    # Si no se encontraron fuentes explícitas, usar las de los documentos
    if not sources:
        for doc in documents[:3]:
            source_key = doc.metadata.get("source", "")
            if source_key and source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "source": source_key,
                    "article": doc.metadata.get("article_number", ""),
                    "page": str(doc.metadata.get("page", "")),
                })

    return sources


# ─── Integración con nodo del grafo ──────────────────────────────────────────

def rethinking_generation_node(state: dict) -> dict:
    """
    Nodo de generación con Rethinking para el grafo LangGraph.

    Args:
        state: AgentState con user_query, retrieval_results.

    Returns:
        Dict con draft_answer, sources.
    """
    query = state.get("active_query") or state.get("user_query", "")
    docs = state.get("retrieval_results", [])

    if not docs:
        return {
            "draft_answer": "No encontré documentos relevantes para responder tu consulta.",
            "sources": [],
        }

    answer, sources = generate_with_rethinking(query, docs)

    return {
        "draft_answer": answer,
        "sources": sources,
    }
