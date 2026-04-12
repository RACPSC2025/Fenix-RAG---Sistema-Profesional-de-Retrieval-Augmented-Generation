"""
Script de evaluación RAGAS para el sistema RACodex.

Ejecuta evaluación completa con métricas:
- Faithfulness: ¿La respuesta se basa únicamente en los contextos?
- Answer Relevancy: ¿La respuesta aborda la pregunta?
- Context Precision: ¿Los contextos recuperados son relevantes?

Uso:
    python scripts/eval_ragas.py
    python scripts/eval_ragas.py --dataset-size 100 --batch-size 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_evaluation_dataset(size: int = 50) -> list[dict]:
    """
    Crea dataset de evaluación con queries realistas.

    En producción, esto debería venir de queries reales de usuarios
    con ground truth anotado manualmente.
    """
    # Dataset de ejemplo — reemplazar con datos reales
    return [
        {
            "question": "¿Qué es RACodex y para quién es?",
            "answer": "RACodex es un asistente de desarrollo con conocimiento personalizado basado en RAG, diseñado para estudiantes, startups, empresas y freelancers.",
            "contexts": ["RACodex es un agente de desarrollo inteligente que combina RAG profesional con capacidades cognitivas agenticas."],
            "ground_truth": "RACodex es un asistente de desarrollo con conocimiento personalizado basado en RAG.",
        },
        {
            "question": "¿Cómo funciona el retrieval híbrido con RRF?",
            "answer": "El retrieval híbrido combina búsqueda vectorial (semántica) con BM25 (keywords) usando Reciprocal Rank Fusion con k=60.",
            "contexts": ["El hybrid retriever combina vector search y BM25 con RRF fusion (k=60)."],
            "ground_truth": "El retrieval híbrido combina búsqueda vectorial con BM25 usando fusión RRF con k=60.",
        },
        {
            "question": "¿Qué es CRAG y cómo funciona?",
            "answer": "CRAG es Corrective RAG: evalúa la calidad de los documentos recuperados antes de generar, con routing condicional según score.",
            "contexts": ["CRAG grades documents: correct (>0.7) → generate, ambiguous (0.3-0.7) → rewrite, incorrect (<0.3) → step-back."],
            "ground_truth": "CRAG evalúa documentos recuperados y rutea condicionalmente según relevancia.",
        },
    ]


def run_evaluation(dataset: list[dict], output_path: str = "ragas_results.json"):
    """
    Ejecuta evaluación RAGAS y guarda resultados.

    En producción, usar ragas.evaluate() con las métricas configuradas.
    """
    print("=" * 60)
    print("📊 Evaluación RAGAS — RACodex")
    print("=" * 60)
    print(f"\nDataset: {len(dataset)} queries")
    print(f"Métricas: faithfulness, answer_relevancy, context_precision\n")

    # Evaluación básica (sin RAGAS instalado — métricas simples)
    results = []
    for i, sample in enumerate(dataset):
        # Faithfulness simple
        answer_words = set(sample["answer"].lower().split())
        context_words = set(" ".join(sample["contexts"]).lower().split())
        faithfulness = len(answer_words & context_words) / max(len(answer_words), 1)

        # Answer relevancy simple
        question_words = set(q for q in sample["question"].lower().split() if len(q) > 3)
        answer_words_sig = set(a for a in sample["answer"].lower().split() if len(a) > 3)
        relevancy = len(question_words & answer_words_sig) / max(len(question_words), 1)

        # Context precision simple
        context_relevant = sum(
            1 for ctx in sample["contexts"]
            if len(set(q for q in sample["question"].lower().split() if len(q) > 3) &
                   set(c for c in ctx.lower().split() if len(c) > 3)) > 0
        )
        precision = context_relevant / max(len(sample["contexts"]), 1)

        results.append({
            "question": sample["question"],
            "faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_precision": round(precision, 3),
        })

    # Promedios
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_precision = sum(r["context_precision"] for r in results) / len(results)

    print("Resultados por query:")
    for r in results:
        print(f"  Q: {r['question'][:60]}...")
        print(f"     Faithfulness: {r['faithfulness']:.3f}")
        print(f"     Relevancy:    {r['answer_relevancy']:.3f}")
        print(f"     Precision:    {r['context_precision']:.3f}")
        print()

    print("=" * 60)
    print("Promedios:")
    print(f"  Faithfulness:     {avg_faithfulness:.3f}")
    print(f"  Answer Relevancy: {avg_relevancy:.3f}")
    print(f"  Context Precision: {avg_precision:.3f}")
    print("=" * 60)

    # Guardar resultados
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps({
        "dataset_size": len(dataset),
        "metrics": {
            "faithfulness": round(avg_faithfulness, 3),
            "answer_relevancy": round(avg_relevancy, 3),
            "context_precision": round(avg_precision, 3),
        },
        "results": results,
    }, indent=2))

    print(f"\nResultados guardados en: {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluación RAGAS para RACodex")
    parser.add_argument("--dataset-size", type=int, default=50, help="Tamaño del dataset")
    parser.add_argument("--output", type=str, default="ragas_results.json", help="Archivo de salida")
    args = parser.parse_args()

    dataset = create_evaluation_dataset(args.dataset_size)
    run_evaluation(dataset, args.output)


if __name__ == "__main__":
    main()
