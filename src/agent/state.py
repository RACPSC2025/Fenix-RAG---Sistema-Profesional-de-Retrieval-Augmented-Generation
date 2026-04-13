"""
Estado central del agente LangGraph para Fénix RAG.

Principios de diseño:
  - `AgentState` es un TypedDict — LangGraph lo serializa/deserializa en cada
    paso del grafo. Todo campo debe ser JSON-serializable.
  - Los mensajes usan `Annotated[list, add_messages]` — el operador `add_messages`
    acumula sin reemplazar (patrón estándar de LangGraph para chat history).
  - Los campos no acumulativos usan el operador por defecto de reemplazo.
  - Separación clara entre estado de ingestion, retrieval, generación y reflexión.

Ciclo de vida del estado en el grafo:
  1. `__start__` → rellena `messages` con la query del usuario
  2. `document_router` → rellena `ingestion_plan`
  3. `ingestion_node` → rellena `ingested_documents`
  4. `retrieval_node` → rellena `retrieval_results`
  5. `grade_node` → rellena `doc_quality`, `grade_score`, `crag_route`
  6. `generation_node` → rellena `draft_answer`, `generation_mode`
  7. `reflection_node` → rellena `reflection`, `reflection_route`
  8. Si reflection_route == "retrieval" → vuelve a retrieval con `active_query`
  9. Si reflection_route == "END" → `__end__` con `final_answer`

CAMPOS DE ROUTING — AISLADOS POR RESPONSABILIDAD:
  `route`            → solo supervisor_node y document_router (legacy)
  `crag_route`       → exclusivo de grade_documents_node → route_after_grading
  `reflection_route` → exclusivo de reflection_node → route_after_reflection
  Los tres campos son distintos para evitar colisión de estado entre nodos.

PROTECCIÓN ANTI-LOOP:
  `crag_retry_count` se incrementa en cada re-retrieval disparado por CRAG.
  Cuando alcanza MAX_CRAG_RETRIES (en crag.py), el grader fuerza
  `crag_route = "generation"` aunque los docs sean subóptimos.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from typing_extensions import TypedDict


# ─── Plan de ingestion ────────────────────────────────────────────────────────

class IngestionPlan(TypedDict):
    """
    Plan de procesamiento para un documento.

    Lo produce DocumentClassifierSkill antes de la ingestion.
    Define qué loader usar, qué cleaner aplicar y si se necesita OCR.
    """
    loader_type: str            # "pymupdf" | "ocr" | "docling" | "word" | "excel"
    cleaner_profile: str        # "technical" | "contract" | "ocr_output" | "default"
    requires_ocr: bool
    document_type: str          # tipo genérico del documento
    source_path: str            # path absoluto al archivo
    mime_type: str
    confidence: float           # confianza de la clasificación [0.0 - 1.0]
    reasoning: str              # explicación del classifier


# ─── Resultado de reflexión ───────────────────────────────────────────────────

class ReflectionOutput(TypedDict):
    """Resultado del nodo de reflexión sobre la respuesta generada."""
    score: float                # calidad [0.0 - 1.0]
    is_grounded: bool           # ¿la respuesta está fundamentada en los docs?
    has_hallucination: bool     # ¿hay información inventada?
    cites_source: bool          # ¿menciona el artículo/fuente?
    feedback: str               # qué mejorar en la próxima iteración
    reformulated_query: str     # query reformulada para re-retrieval (si aplica)


# ─── Estado principal del agente ──────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Estado completo del agente Fénix RAG.

    Todos los campos son opcionales excepto `messages`.
    LangGraph reemplaza campos a menos que usen un operador acumulativo.

    Convención de nombres:
      `*_plan`        — planes/decisiones tomadas por skills
      `*_results`     — resultados de operaciones de retrieval
      `draft_*`       — respuesta en construcción (pre-reflexión)
      `final_*`       — respuesta aprobada post-reflexión
      `crag_*`        — campos exclusivos del CRAG grading
      `reflection_*`  — campos exclusivos del nodo de reflexión
    """

    # ── Mensajes (acumulativos) ───────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Contexto de la sesión ─────────────────────────────────────────────────
    session_id: str
    user_query: str             # query original del usuario (sin modificar)
    active_query: str           # query actual (puede ser reformulada)

    # ── Skill Pack activo [Fase 10] ───────────────────────────────────────────
    active_profile: str         # nombre del pack activo ("general-dev", etc.)
                                # "" → generation_node usa default del registry

    # ── Documentos del usuario ────────────────────────────────────────────────
    uploaded_files: list[str]
    ingestion_plans: list[IngestionPlan]
    ingested_documents: list[Document]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_results: list[Document]
    retrieval_strategy: str

    # ── CRAG — campos propios, NO compartidos con reflection ni supervisor ─────
    doc_quality: str            # "correct" | "ambiguous" | "incorrect"
    grade_score: float          # score CRAG (0.0-1.0), leído por generation_node
    crag_route: str             # "generation"|"retrieval" — leído por route_after_grading
    crag_retry_count: int       # reintentos CRAG — protege contra loops infinitos

    # ── Generación ────────────────────────────────────────────────────────────
    draft_answer: str
    final_answer: str
    sources: list[dict[str, str]]
    generation_mode: str        # "direct"|"rethinking"|"rethinking_low_confidence"|"no_docs"

    # ── Reflexión — campos propios, NO compartidos con CRAG ni supervisor ──────
    reflection: ReflectionOutput | None
    reflection_route: str       # "END"|"retrieval" — leído por route_after_reflection
    iteration_count: int
    max_iterations: int

    # ── Memoria de sesión (persistida via checkpointer) ──────────────────────
    session_memory: dict[str, str]

    # ── Metadata del grafo ────────────────────────────────────────────────────
    error: str | None
    route: str                  # legacy: document_router y supervisor_node únicamente

    # ── Métricas del pipeline (observabilidad) ───────────────────────────────
    pipeline_metrics: dict[str, dict]

    # ── Managed values (LangGraph internal) ──────────────────────────────────
    remaining_steps: RemainingSteps


# ─── Estado inicial ───────────────────────────────────────────────────────────

def initial_state(
    user_query: str,
    session_id: str = "",
    uploaded_files: list[str] | None = None,
    max_iterations: int = 2,
    active_profile: str = "",
) -> dict[str, Any]:
    """
    Construye el estado inicial para una nueva invocación del agente.

    Args:
        user_query: Pregunta del usuario.
        session_id: ID de sesión para trazabilidad.
        uploaded_files: Paths a archivos que el usuario quiere consultar.
        max_iterations: Máximo de ciclos reflection → re-retrieval.
        active_profile: Skill pack a usar. "" → generation_node resuelve
                        con el default del SkillRegistry ("general-dev").
                        La API puede pasar el perfil explícito si el usuario
                        lo seleccionó en el frontend (Fase 11).

    Returns:
        Dict compatible con AgentState para pasar a graph.invoke().
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=user_query)],
        "session_id": session_id,
        "user_query": user_query,
        "active_query": user_query,
        "uploaded_files": uploaded_files or [],
        "ingestion_plans": [],
        "ingested_documents": [],
        "retrieval_results": [],
        "retrieval_strategy": "",
        "draft_answer": "",
        "final_answer": "",
        "sources": [],
        "reflection": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "session_memory": {},
        "error": None,
        "route": "",
        # Skill Pack [Fase 10]
        "active_profile": active_profile,
        # CRAG — inicialización explícita obligatoria
        "doc_quality": "",
        "grade_score": 0.0,
        "crag_route": "",
        "crag_retry_count": 0,
        # Reflection — inicialización explícita obligatoria
        "reflection_route": "",
        "generation_mode": "",
        # Métricas
        "pipeline_metrics": {},
    }
