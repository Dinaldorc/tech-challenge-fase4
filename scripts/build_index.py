"""Gera os embeddings das avaliações e monta o índice vetorial no ChromaDB.

Uso:
    python scripts/prepare_data.py     # gera data/processed/reviews_docs.parquet, se ainda não existir
    python scripts/build_index.py      # base completa (42.370 documentos, cerca de 11 min na CPU)

    # Índice só da amostra, usado pelo experimento de scripts/compare_search.py:
    python scripts/build_index.py --input data/sample/reviews_sample.csv --persist-dir chroma_db_sample

O padrão é a base completa porque o corte de recusa da busca (src/retrieval.py) foi calibrado nela.
Na primeira execução, o modelo de embeddings (~1,1 GB) é baixado do Hugging Face e fica em cache.

Saída:
    chroma_db/   índice persistente (não versionado, reproduzível)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import DEFAULT_MODEL, E5Embedder  # noqa: E402
from src.preprocessing import load_documents  # noqa: E402
from src.vector_store import DEFAULT_BATCH_SIZE, DEFAULT_COLLECTION, index_documents, open_collection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "reviews_docs.parquet")
    parser.add_argument("--persist-dir", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cpu ou cuda (padrão: automático)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--reset", action="store_true", help="apaga a coleção antes de indexar")
    return parser.parse_args()


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Arquivo não encontrado: {args.input}\nGere a base com: python scripts/prepare_data.py")
    docs = load_documents(args.input)
    print(f"Documentos lidos de {args.input.name}: {fmt(len(docs))}")

    print(f"Carregando o modelo {args.model}...")
    embedder = E5Embedder(args.model, device=args.device)
    print(f"  dimensão {embedder.dimension}, dispositivo {embedder.device}")

    collection = open_collection(args.persist_dir, args.model, args.collection, reset=args.reset)

    start = time.perf_counter()

    def report(done: int, total: int) -> None:
        elapsed = time.perf_counter() - start
        print(f"\r  indexados {fmt(done)}/{fmt(total)} ({elapsed:.0f} s)", end="", flush=True)

    written = index_documents(collection, docs, embedder, args.batch_size, on_batch=report)
    elapsed = time.perf_counter() - start
    print(f"\nGravados {fmt(written)} documentos em {elapsed:.1f} s ({written / elapsed:.0f} por segundo)")
    print(f"Coleção '{args.collection}' com {fmt(collection.count())} documentos em {args.persist_dir}")


if __name__ == "__main__":
    main()
