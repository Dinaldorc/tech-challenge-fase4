"""Gera a base de documentos do RAG a partir dos CSVs da Olist.

Uso:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --sample-size 3000 --seed 42

Saídas:
    data/processed/reviews_docs.parquet   base completa (não versionada, reproduzível)
    data/sample/reviews_sample.csv        amostra estratificada por nota (versionada)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import build_documents, load_raw_tables, sample_documents  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "reviews_docs.parquet")
    parser.add_argument("--sample-output", type=Path, default=ROOT / "data" / "sample" / "reviews_sample.csv")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> None:
    args = parse_args()
    tables = load_raw_tables(args.raw_dir)
    reviews = tables["reviews"]

    docs = build_documents(**tables)
    sample = sample_documents(docs, args.sample_size, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(args.output, index=False)
    sample.to_csv(args.sample_output, index=False, encoding="utf-8")

    unique_reviews = reviews["review_id"].nunique()
    print(f"Linhas em order_reviews:           {fmt(len(reviews))}")
    print(f"Avaliações distintas (review_id):  {fmt(unique_reviews)}")
    print(f"Documentos com título ou texto:    {fmt(len(docs))} ({len(docs) / unique_reviews:.1%})")
    print(f"  de baixa informação:             {fmt(int(docs['low_information'].sum()))}")
    print(f"  pedidos sem itens:               {fmt(int((docs['product_category'] == 'sem_itens').sum()))}")
    print("Status da entrega:")
    for status, count in docs["delivery_status"].value_counts().items():
        print(f"  {status:<31}  {fmt(count)}")
    print(f"\nBase completa: {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")
    print(f"Amostra ({fmt(len(sample))} documentos): "
          f"{args.sample_output.relative_to(ROOT) if args.sample_output.is_relative_to(ROOT) else args.sample_output}")


if __name__ == "__main__":
    main()
