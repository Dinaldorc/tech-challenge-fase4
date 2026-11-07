"""Classifica as avaliações por tema com o LLM e valida contra uma referência independente.

Uso:
    python scripts/classify_themes.py --sample 300 --output piloto.csv   # piloto
    python scripts/classify_themes.py                                     # reclamações, base inteira
    python scripts/classify_themes.py --temas elogios                     # elogios, base inteira

Requer a base (scripts/prepare_data.py) e OPENAI_API_KEY no .env. Chama a API: veja o custo no fim.

Validação: um tema de cada lista é comparado com uma referência independente, uma expressão regular.
Nas reclamações, o não recebimento ("não recebi", "não chegou", "ainda não", "cadê"); nos elogios, a
entrega no prazo ou rápida ("antes do prazo", "no prazo", "entrega rápida"). As divergências são
listadas para leitura, porque nenhuma das duas é perfeita: a expressão regular não entende o sentido,
e o LLM pode errar.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import DEFAULT_MODEL, build_chat_model  # noqa: E402
from src.preprocessing import load_documents, normalize_for_comparison  # noqa: E402
from src.themes import DEFAULT_BATCH_SIZE, TAXONOMIES, build_classifier, parse_batch  # noqa: E402

# Referência independente de um tema de cada lista: (tema, expressão regular sobre o texto normalizado).
REFERENCES = {
    "reclamacoes": ("nao_recebimento", re.compile(r"\bnao (recebi|chegou|foi entregue|veio)\b|\bainda nao\b|\bcade\b")),
    "elogios": ("entrega_prazo", re.compile(
        r"\b(antes|dentro|no) do prazo\b|\bno prazo\b|\bantes do previsto\b"
        r"|\b(entrega|chegou|chegaram|entregue|veio)\w* (super |muito |bem )?(rapid|rapida|rapido|cedo)\w*"
        r"|\brapidez\b|\bpontual\w*")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temas", choices=sorted(TAXONOMIES), default="reclamacoes",
                        help="lista de temas: reclamações (avaliações negativas) ou elogios (positivas)")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "reviews_docs.parquet")
    parser.add_argument("--output", type=Path, default=None, help="padrão: data/themes/<arquivo da lista>")
    parser.add_argument("--sample", type=int, default=None, help="classifica só uma amostra aleatória (piloto)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default=None, help=f"modelo da OpenAI (padrão: OPENAI_MODEL ou {DEFAULT_MODEL})")
    return parser.parse_args()


def classify(classifier, texts: list[str], batch_size: int, concurrency: int, taxonomy) -> tuple[list[list[str]], dict]:
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    outputs = classifier.batch(batches, config={"max_concurrency": concurrency})
    themes, usage = [], Counter()
    for batch, output in zip(batches, outputs):
        batch_themes, missing = parse_batch(output, len(batch), taxonomy)
        if missing:  # uma segunda tentativa para o lote incompleto
            output = classifier.invoke(batch)
            batch_themes, missing = parse_batch(output, len(batch), taxonomy)
        usage.update({k: v for k, v in (output["raw"].usage_metadata or {}).items() if isinstance(v, int)})
        usage["missing"] += missing
        themes.extend(batch_themes)
    usage["calls"] = len(batches)
    return themes, usage


def validate(docs: pd.DataFrame, theme: str, pattern: re.Pattern) -> None:
    reference = docs["text"].map(lambda t: bool(pattern.search(normalize_for_comparison(t))))
    llm = docs["themes"].map(lambda t: theme in t)
    both = int((reference & llm).sum())
    print(f"\n## Validação do tema {theme} contra a expressão regular")
    print(f"  expressão regular: {int(reference.sum())} | LLM: {int(llm.sum())} | os dois: {both}")
    print(f"  do que a expressão marca, o LLM também marca: {both / max(reference.sum(), 1):.0%}")
    print(f"  do que o LLM marca, a expressão também marca: {both / max(llm.sum(), 1):.0%}")
    for title, mask in [("Só a expressão regular marcou", reference & ~llm), ("Só o LLM marcou", llm & ~reference)]:
        sample = docs[mask].head(8)
        print(f"\n  {title} ({int(mask.sum())}):")
        for row in sample.itertuples():
            print(f"    [{';'.join(row.themes)}] {row.text[:110]}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    taxonomy = TAXONOMIES[args.temas]
    output_path = args.output or ROOT / "data" / "themes" / taxonomy.output_file
    docs = load_documents(args.input)
    docs = docs[(docs["sentiment"] == taxonomy.sentiment) & ~docs["low_information"]].reset_index(drop=True)
    if args.sample:
        docs = docs.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    print(f"{taxonomy.reviews.capitalize()} a classificar: {len(docs)}")

    chat_model = build_chat_model(model=args.model)
    start = time.perf_counter()
    themes, usage = classify(build_classifier(chat_model, taxonomy), docs["text"].tolist(), args.batch_size,
                             args.concurrency, taxonomy)
    elapsed = time.perf_counter() - start
    docs["themes"] = themes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs[["review_id"]].assign(themes=docs["themes"].map(";".join)).to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nModelo: {chat_model.model_name} | chamadas: {usage['calls']} | tempo: {elapsed:.0f} s")
    print(f"Tokens: {usage['input_tokens']:,} de entrada + {usage['output_tokens']:,} de saída".replace(",", "."))
    print(f"Por avaliação: {usage['input_tokens'] / len(docs):.0f} + {usage['output_tokens'] / len(docs):.1f} tokens"
          f" | sem resposta após nova tentativa: {usage['missing']}")
    counts = Counter(theme for row in themes for theme in row)
    print("\n## Temas (uma avaliação pode ter mais de um)")
    for theme in taxonomy.themes:
        print(f"  {theme:<27} {counts[theme]:>6}  {counts[theme] / len(docs):>6.1%}")
    print(f"  temas por avaliação, em média: {np.mean([len(t) for t in themes]):.2f}")
    validate(docs, *REFERENCES[taxonomy.name])
    print(f"\nResultado salvo em {output_path}")


if __name__ == "__main__":
    main()
