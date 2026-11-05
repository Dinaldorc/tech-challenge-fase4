"""Experimento: o agrupamento automático dos embeddings serve para contar temas das reclamações?

Agrupa as avaliações negativas (sem as de baixa informação) com K-Means e HDBSCAN e, se instalado,
BERTopic, usando os embeddings E5 já gravados no índice. Depois:

- mede a silhueta (quão separados estão os grupos);
- mostra os termos que mais distinguem cada grupo (c-TF-IDF);
- valida contra uma referência independente de não recebimento (expressão regular): o grupo que
  melhor representa o tema deveria reunir a maior parte dessas avaliações.

Resultado (README, "Frequência por tema"): as reclamações não formam grupos separados (silhueta
perto de zero) e o não recebimento fica espalhado por vários grupos. O agrupamento serviu para
descobrir os temas, e a contagem passou a ser feita por classificação (scripts/classify_themes.py).

Uso:
    python scripts/compare_clustering.py
    pip install bertopic   # opcional, só para incluir o BERTopic no experimento
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import DEFAULT_MODEL  # noqa: E402
from src.preprocessing import normalize_for_comparison  # noqa: E402
from src.vector_store import open_vector_store  # noqa: E402

SEED = 7
NOT_RECEIVED = re.compile(r"\bnao (recebi|chegou|foi entregue|veio)\b|\bainda nao\b|\bcade\b")
STOP = ["de", "do", "da", "o", "a", "e", "que", "em", "um", "uma", "os", "as", "no", "na", "com", "para", "por",
        "meu", "minha", "foi", "ja", "me", "se", "mais", "muito", "ate", "pois", "esta", "ao", "eu", "mas", "pelo",
        "pela", "dos", "das", "nem", "isso", "ele", "ela", "tem", "ter", "sem", "so", "como", "porem", "produto", "nao"]


class StoredVectorsOnly(Embeddings):
    """Só leitura: os vetores já estão no índice, então o modelo não precisa ser carregado."""

    model_name = DEFAULT_MODEL

    def embed_documents(self, texts):
        raise RuntimeError("não usado")

    def embed_query(self, text):
        raise RuntimeError("não usado")


def load_negatives(persist_dir: Path) -> tuple[np.ndarray, list[str]]:
    store = open_vector_store(persist_dir, StoredVectorsOnly())
    where = {"$and": [{"sentiment": "negativo"}, {"low_information": False}]}
    vectors, texts, offset = [], [], 0
    while True:
        part = store.get(where=where, include=["embeddings", "documents"], limit=2000, offset=offset)
        if not part["ids"]:
            break
        vectors.append(np.asarray(part["embeddings"], dtype=np.float32))
        texts += part["documents"]
        offset += 2000
    return np.vstack(vectors), [normalize_for_comparison(t) for t in texts]


def top_terms(docs: list[str], labels: np.ndarray, n: int = 6) -> dict[int, str]:
    """Termos que mais distinguem cada grupo (c-TF-IDF: o TF-IDF calculado por grupo)."""
    counts = CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=5).fit(docs)
    matrix = counts.transform(docs)
    groups = sorted(set(labels) - {-1})
    per_group = np.vstack([np.asarray(matrix[labels == g].sum(axis=0)).ravel() for g in groups])
    scores = (per_group / per_group.sum(axis=1, keepdims=True)) * np.log(1 + per_group.sum() / (1 + per_group.sum(axis=0)))
    vocab = counts.get_feature_names_out()
    return {g: ", ".join(vocab[np.argsort(-scores[i])[:n]]) for i, g in enumerate(groups)}


def best_match(labels: np.ndarray, truth: np.ndarray) -> str:
    best = None
    for g in sorted(set(labels) - {-1}):
        inside = labels == g
        precision = (truth & inside).sum() / inside.sum()
        recall = (truth & inside).sum() / truth.sum()
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if best is None or f1 > best[3]:
            best = (g, precision, recall, f1, int(inside.sum()))
    g, p, r, f1, size = best
    return f"grupo {g} ({size} avaliações): precisão {p:.0%}, cobertura {r:.0%}, F1 {f1:.2f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    X, docs = load_negatives(ROOT / "chroma_db")
    truth = np.array([bool(NOT_RECEIVED.search(d)) for d in docs])
    sample = np.random.default_rng(SEED).choice(len(X), min(4000, len(X)), replace=False)
    print(f"Avaliações negativas: {len(X)} | referência de não recebimento: {truth.sum()} ({truth.mean():.1%})")

    print("\n## K-Means")
    print(f"{'k':>4}{'silhueta':>10}  melhor grupo para não recebimento")
    for k in [5, 8, 12, 20, 30]:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        sil = silhouette_score(X[sample], labels[sample], metric="cosine")
        print(f"{k:>4}{sil:>10.4f}  {best_match(labels, truth)}")
    labels = KMeans(n_clusters=12, n_init=10, random_state=SEED).fit_predict(X)
    print("\n  Termos dos grupos com k = 12:")
    terms = top_terms(docs, labels)
    for g in np.argsort(-np.bincount(labels)):
        print(f"  {g:>3} {np.bincount(labels)[g]:>6}  {terms[g]}")

    print("\n## HDBSCAN sobre PCA (768 -> 50 dimensões)")
    Z = PCA(n_components=50, random_state=SEED).fit_transform(X)
    for min_size in [25, 50, 100]:
        labels = HDBSCAN(min_cluster_size=min_size, min_samples=10).fit_predict(Z)
        print(f"  tamanho mínimo {min_size}: {labels.max() + 1} grupos, {np.mean(labels == -1):.0%} sem grupo")

    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN as HDBSCANLib
        from umap import UMAP
    except ImportError:
        print("\n## BERTopic: não instalado (pip install bertopic) — etapa ignorada")
        return
    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
        hdbscan_model=HDBSCANLib(min_cluster_size=50, metric="euclidean", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=5),
    )
    labels = np.array(model.fit_transform(docs, embeddings=X)[0])
    print(f"\n## BERTopic (tamanho mínimo 50): {labels.max() + 1} temas, {np.mean(labels == -1):.0%} sem tema")
    print(f"  melhor tema para não recebimento: {best_match(labels, truth)}")
    for _, row in model.get_topic_info().head(12).iterrows():
        if row.Topic != -1:
            print(f"  {row.Topic:>3} {row.Count:>6}  {', '.join(w for w, _ in model.get_topic(row.Topic)[:6])}")


if __name__ == "__main__":
    main()
