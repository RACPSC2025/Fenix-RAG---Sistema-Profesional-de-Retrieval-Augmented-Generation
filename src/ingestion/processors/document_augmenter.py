"""
Document Augmentation — Genera preguntas por chunk durante la ingestión.

Cada chunk del documento se augmenta con 3-5 preguntas generadas por LLM
que ese chunk podría responder. Estas preguntas se indexan junto al chunk
original, mejorando el match entre la query del usuario y el contenido.

Beneficio: la query del usuario ("¿qué pasa si no cumplo?") tiene más
probabilidad de coincidir con una pregunta pregenerada
("¿Cuáles son las consecuencias del incumplimiento?") que con el texto
legal crudo del chunk.

Uso en el pipeline de ingestión:
    from src.ingestion.processors.document_augmenter import augment_documents

    chunks = chunker.chunk(docs)
    augmented = augment_documents(chunks, llm=llm, questions_per_chunk=3)
    vector_store.add_documents(augmented)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.logging import get_logger
from src.config.providers import get_llm

log = get_logger(__name__)


# ─── Prompt ──────────────────────────────────────────────────────────────────

QUESTION_GENERATION_PROMPT = SystemMessage(
    content=(
        "Eres un experto generando preguntas que los usuarios podrían hacer "
        "sobre un fragmento de documento.\n\n"
        "Tu tarea:\n"
        "- Genera preguntas realistas que un usuario haría basándose en este contenido\n"
        "- Usa lenguaje natural, no terminología técnica excesiva\n"
        "- Cada pregunta debe ser respondible SOLO con el fragmento proporcionado\n"
        "- Retorna UNA pregunta por línea, sin numeración ni prefijos"
    )
)


# ─── Función principal ───────────────────────────────────────────────────────

def augment_documents(
    chunks: list[Document],
    llm=None,
    questions_per_chunk: int = 3,
) -> list[Document]:
    """
    Augmenta cada chunk con preguntas generadas por LLM.

    Para cada chunk original:
      1. Genera N preguntas que ese chunk puede responder
      2. Crea N documentos "hijo" con la pregunta como page_content
         y metadata apuntando al chunk padre

    Args:
        chunks: Lista de documentos chunkeados.
        llm: LLM instance. None = usa el default.
        questions_per_chunk: Número de preguntas por chunk (3-5 recomendado).

    Returns:
        Lista original de chunks + documentos de preguntas augmentadas.
        Los chunks originales se mantienen intactos.
    """
    llm = llm or get_llm(temperature=0.5)
    augmented: list[Document] = list(chunks)  # Copia de los originales

    for i, chunk in enumerate(chunks):
        try:
            questions = _generate_questions(
                chunk, llm, questions_per_chunk
            )
            for q_text in questions:
                augmented.append(Document(
                    page_content=q_text,
                    metadata={
                        **chunk.metadata,
                        "augmentation_type": "question",
                        "parent_chunk_index": chunk.metadata.get("chunk_index", i),
                        "is_augmented_question": True,
                    },
                ))

            log.debug(
                "chunk_augmented",
                chunk_index=chunk.metadata.get("chunk_index", i),
                questions_generated=len(questions),
            )

        except Exception as exc:
            log.warning(
                "chunk_augmentation_failed",
                chunk_index=chunk.metadata.get("chunk_index", i),
                error=str(exc),
            )
            # Continúa sin augmentar este chunk — no bloquea la ingestión

    log.info(
        "document_augmentation_complete",
        original_chunks=len(chunks),
        total_documents=len(augmented),
        questions_added=len(augmented) - len(chunks),
    )

    return augmented


# ─── Versión Async con concurrencia ──────────────────────────────────────────

async def augment_documents_async(
    chunks: list[Document],
    llm=None,
    questions_per_chunk: int = 3,
    max_concurrency: int = 5,
) -> list[Document]:
    """
    Augmenta cada chunk con preguntas generadas por LLM — versión async.

    Usa asyncio.gather con semáforo para procesar chunks en paralelo
    sin saturar el LLM. Para 400 chunks con max_concurrency=5:
    ~80 batches de 5 llamadas simultáneas vs 400 llamadas secuenciales.

    Args:
        chunks: Lista de documentos chunkeados.
        llm: LLM instance. None = usa el default.
        questions_per_chunk: Número de preguntas por chunk (3-5 recomendado).
        max_concurrency: Máximas llamadas LLM simultáneas (default 5).

    Returns:
        Lista original de chunks + documentos de preguntas augmentadas.
    """
    llm = llm or get_llm(temperature=0.5)
    semaphore = asyncio.Semaphore(max_concurrency)

    log.info(
        "async_augmentation_start",
        chunks=len(chunks),
        max_concurrency=max_concurrency,
        estimated_batches=len(chunks) // max_concurrency + 1,
    )

    async def _augment_one_chunk(chunk: Document, index: int) -> list[Document]:
        """Augmenta un solo chunk con semáforo."""
        async with semaphore:
            try:
                questions = await _generate_questions_async(
                    chunk, llm, questions_per_chunk
                )
                return [
                    Document(
                        page_content=q_text,
                        metadata={
                            **chunk.metadata,
                            "augmentation_type": "question",
                            "parent_chunk_index": chunk.metadata.get("chunk_index", index),
                            "is_augmented_question": True,
                        },
                    )
                    for q_text in questions
                ]
            except Exception as exc:
                log.warning(
                    "async_chunk_augmentation_failed",
                    chunk_index=chunk.metadata.get("chunk_index", index),
                    error=str(exc),
                )
                return []

    # Ejecutar todas las augmentaciones en paralelo con semáforo
    tasks = [
        _augment_one_chunk(chunk, i)
        for i, chunk in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aplanar resultados (cada resultado es una lista de Documents)
    all_questions: list[Document] = []
    for result in results:
        if isinstance(result, Exception):
            log.warning("async_augment_exception", error=str(result))
        elif isinstance(result, list):
            all_questions.extend(result)

    augmented = list(chunks) + all_questions

    log.info(
        "async_augmentation_complete",
        original_chunks=len(chunks),
        total_documents=len(augmented),
        questions_added=len(all_questions),
    )

    return augmented


# ─── Helpers síncronos ────────────────────────────────────────────────────────

def _generate_questions(
    chunk: Document,
    llm,
    n: int,
) -> list[str]:
    """Genera N preguntas para un chunk dado."""
    prompt = (
        f"Genera {n} preguntas realistas que un usuario podría hacer "
        f"sobre el siguiente fragmento:\n\n"
        f"{chunk.page_content[:1500]}\n\n"
        f"Retorna exactamente {n} preguntas, una por línea."
    )

    response = llm.invoke([
        QUESTION_GENERATION_PROMPT,
        HumanMessage(content=prompt),
    ])

    # Parsear respuestas: una por línea, filtrar vacías
    questions = [
        q.strip().lstrip("0123456789.-*? ")
        for q in response.content.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ]

    return questions[:n]


async def _generate_questions_async(
    chunk: Document,
    llm,
    n: int,
) -> list[str]:
    """Genera N preguntas para un chunk dado — versión async."""
    prompt = (
        f"Genera {n} preguntas realistas que un usuario podría hacer "
        f"sobre el siguiente fragmento:\n\n"
        f"{chunk.page_content[:1500]}\n\n"
        f"Retorna exactamente {n} preguntas, una por línea."
    )

    response = await llm.ainvoke([
        QUESTION_GENERATION_PROMPT,
        HumanMessage(content=prompt),
    ])

    questions = [
        q.strip().lstrip("0123456789.-*? ")
        for q in response.content.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ]

    return questions[:n]


# ─── Integración con pipeline de ingestión ───────────────────────────────────

async def augment_and_index_async(
    chunks: list[Document],
    vector_store,
    llm=None,
    questions_per_chunk: int = 3,
    max_concurrency: int = 5,
) -> dict[str, int]:
    """
    Augmenta chunks async y los indexa en el vector store.

    Args:
        chunks: Documentos chunkeados originales.
        vector_store: Chroma VectorStore para indexar.
        llm: LLM Instance. None = default.
        questions_per_chunk: Preguntas por chunk.
        max_concurrency: Máximas llamadas LLM simultáneas.

    Returns:
        Dict con {"original_chunks": N, "questions_added": M, "total_indexed": T}
    """
    augmented = await augment_documents_async(
        chunks,
        llm=llm,
        questions_per_chunk=questions_per_chunk,
        max_concurrency=max_concurrency,
    )

    originals = [d for d in augmented if not d.metadata.get("is_augmented_question")]
    questions = [d for d in augmented if d.metadata.get("is_augmented_question")]

    vector_store.add_documents(augmented)

    return {
        "original_chunks": len(originals),
        "questions_added": len(questions),
        "total_indexed": len(augmented),
    }

def augment_and_index(
    chunks: list[Document],
    vector_store,
    llm=None,
    questions_per_chunk: int = 3,
) -> dict[str, int]:
    """
    Augmenta chunks y los indexa en el vector store.

    Args:
        chunks: Documentos chunkeados originales.
        vector_store: Chroma VectorStore para indexar.
        llm: LLM Instance. None = default.
        questions_per_chunk: Preguntas por chunk.

    Returns:
        Dict con {"original_chunks": N, "questions_added": M, "total_indexed": T}
    """
    augmented = augment_documents(chunks, llm=llm, questions_per_chunk=questions_per_chunk)

    # Separar originales de augmentados para stats
    originals = [d for d in augmented if not d.metadata.get("is_augmented_question")]
    questions = [d for d in augmented if d.metadata.get("is_augmented_question")]

    # Indexar todo junto
    vector_store.add_documents(augmented)

    return {
        "original_chunks": len(originals),
        "questions_added": len(questions),
        "total_indexed": len(augmented),
    }
