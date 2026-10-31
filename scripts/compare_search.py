"""Compara a busca semântica (E5) com a busca lexical (BM25), com e sem filtro de sentimento.

Reproduz o experimento do README ("Achado 1"). As duas buscas rodam sobre a amostra versionada, então
o E5 usa um índice só da amostra, que precisa ser gerado antes:

    python scripts/build_index.py --input data/sample/reviews_sample.csv --persist-dir chroma_db_sample
    python scripts/compare_search.py
    python scripts/compare_search.py --question "produto veio quebrado" --k 5

O BM25 é a linha de base padrão em recuperação de informação: um TF-IDF em que repetir uma palavra
no mesmo texto tem retorno decrescente e textos longos não ganham vantagem só pelo tamanho. Ele
roda sobre o texto normalizado (minúsculas, sem acentos e pontuação), separado por espaços.

A relevância de cada resultado é julgada lendo os textos impressos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import DEFAULT_MODEL, E5Embedder  # noqa: E402
from src.preprocessing import load_documents, normalize_for_comparison  # noqa: E402
from src.vector_store import DEFAULT_COLLECTION, open_collection  # noqa: E402

DEFAULT_QUESTION = "entrega atrasada, não chegou no prazo"
# Comentários que descrevem o problema sem repetir as palavras da pergunta.
DEFAULT_PROBES = ["Demorou para entregar", "Atraso na entrega."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "sample" / "reviews_sample.csv")
    parser.add_argument("--persist-dir", type=Path, default=ROOT / "chroma_db_sample")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--probe", action="append", help="texto exato cuja posição será mostrada")
    return parser.parse_args()


def print_results(title: str, rows: list[tuple[int, str]]) -> None:
    print(f"\n## {title}: notas {[score for score, _ in rows]}")
    for position, (score, text) in enumerate(rows, 1):
        print(f"  {position:2d} [{score}] {text[:110]}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    probes = args.probe or DEFAULT_PROBES

    # Mesmo universo nas duas buscas: a amostra sem os textos de baixa informação.
    docs = load_documents(args.input)
    docs = docs[~docs["low_information"]].reset_index(drop=True)
    negative = (docs["sentiment"] == "negativo").to_numpy()

    # ---- BM25 -----------------------------------------------------------------------------------
    bm25 = BM25Okapi([text.split() for text in docs["text_norm"]])
    query_tokens = normalize_for_comparison(args.question).split()
    bm25_scores = bm25.get_scores(query_tokens)

    def bm25_top(mask: np.ndarray) -> list[tuple[int, str]]:
        candidates = np.flatnonzero(mask)
        order = candidates[np.argsort(-bm25_scores[candidates], kind="stable")[: args.k]]
        return list(zip(docs["review_score"].iloc[order].tolist(), docs["text"].iloc[order].tolist()))

    # ---- E5 -------------------------------------------------------------------------------------
    embedder = E5Embedder()
    collection = open_collection(args.persist_dir, DEFAULT_MODEL, DEFAULT_COLLECTION)
    query_vector = embedder.embed_query(args.question)

    def e5_search(where: dict, n: int) -> list[tuple[int, str]]:
        result = collection.query(query_embeddings=[query_vector], n_results=n, where=where)
        return [(meta["review_score"], text) for meta, text in zip(result["metadatas"][0], result["documents"][0])]

    no_filter = {"low_information": False}
    with_filter = {"$and": [{"low_information": False}, {"sentiment": "negativo"}]}

    print(f'Pergunta: "{args.question}"  ({len(docs)} documentos)')
    print_results("BM25, sem filtro", bm25_top(np.ones(len(docs), dtype=bool)))
    print_results("E5, sem filtro", e5_search(no_filter, args.k))
    print_results("BM25 + sentimento negativo", bm25_top(negative))
    print_results("E5 + sentimento negativo", e5_search(with_filter, args.k))

    # ---- Diagnóstico ----------------------------------------------------------------------------
    print("\n## Diagnóstico")
    zero = int((bm25_scores == 0).sum())
    print(f"Documentos com pontuação BM25 zero (nenhuma palavra em comum): {zero} ({zero / len(docs):.0%})")

    e5_ranking = [text for _, text in e5_search(no_filter, len(docs))]
    for probe in probes:
        matches = np.flatnonzero(docs["text"].to_numpy() == probe)
        if len(matches) == 0:
            print(f'"{probe}": não está na amostra')
            continue
        score = bm25_scores[matches[0]]
        bm25_rank = int((bm25_scores > score).sum()) + 1
        tie = " (empatado com todos os de pontuação zero)" if score == 0 else ""
        print(f'"{probe}": BM25 posição {bm25_rank}, pontuação {score:.2f}{tie}; '
              f"E5 posição {e5_ranking.index(probe) + 1}")

    share = {token: float(np.mean([token in text.split() for text in docs["text_norm"]])) for token in query_tokens}
    print("Peso (IDF) dos termos da pergunta no BM25 e % de documentos que os contêm:")
    for token in dict.fromkeys(query_tokens):
        print(f"  {token:<10} IDF {bm25.idf.get(token, 0):5.2f}   presente em {share[token]:.1%}")


if __name__ == "__main__":
    main()
