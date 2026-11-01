"""Faz uma pergunta ao RAG pelo terminal e mostra a resposta com as avaliações citadas.

Uso:
    python scripts/ask.py "quais as principais reclamações sobre entrega?"
    python scripts/ask.py "o que dizem dos móveis?" --category moveis_decoracao --sentiment negativo
    python scripts/ask.py "como está a embalagem?" --all-evidence

Requer o índice (scripts/build_index.py) e OPENAI_API_KEY no .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import DEFAULT_MODEL, E5Embedder  # noqa: E402
from src.llm import OpenAIClient  # noqa: E402
from src.rag_pipeline import RagPipeline, format_evidence  # noqa: E402
from src.retrieval import DEFAULT_K, Retriever, SearchFilters  # noqa: E402
from src.vector_store import DEFAULT_COLLECTION, open_collection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="número máximo de evidências")
    parser.add_argument("--sentiment", choices=["negativo", "neutro", "positivo"], action="append")
    parser.add_argument("--category", action="append", help="categoria do produto (pode repetir)")
    parser.add_argument("--model", default=None, help="modelo da OpenAI (padrão: OPENAI_MODEL ou gpt-4.1-mini)")
    parser.add_argument("--persist-dir", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--all-evidence", action="store_true", help="mostra também as evidências não citadas")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    collection = open_collection(args.persist_dir, DEFAULT_MODEL, DEFAULT_COLLECTION)
    pipeline = RagPipeline(Retriever(collection, E5Embedder()), OpenAIClient(model=args.model))
    filters = SearchFilters(
        sentiments=tuple(args.sentiment) if args.sentiment else None,
        categories=tuple(args.category) if args.category else None,
    )
    response = pipeline.ask(args.question, k=args.k, filters=filters)

    print(f"\nPergunta: {response.question}\n")
    print(response.answer)
    if not response.answered:
        print(f"\n(recusada na etapa: {response.refused_at}; motivo: {response.refusal_reason})")

    evidences = response.retrieval.evidences if args.all_evidence else response.citations
    if evidences:
        numbers = {id(e): i for i, e in enumerate(response.retrieval.evidences, 1)}
        print("\nAvaliações citadas:" if not args.all_evidence else "\nAvaliações encontradas:")
        for evidence in evidences:
            print("  " + format_evidence(numbers[id(evidence)], evidence).replace("\n", "\n  "))
    if response.invalid_citations:
        print(f"\nAtenção: a resposta citou números inexistentes: {list(response.invalid_citations)}")

    best = response.retrieval.best_similarity
    print(f"\nFiltros: {response.retrieval.filters.sentiments or 'nenhum'} | "
          f"melhor similaridade: {best:.3f} | " if best is not None else "\n", end="")
    if response.model:
        print(f"modelo: {response.model} | tokens: {response.input_tokens} + {response.output_tokens}")
    else:
        print("LLM não chamado")


if __name__ == "__main__":
    main()
