from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.embeddings import E5Embedder, as_passage, as_query
from src.preprocessing import load_documents
from src.vector_store import (
    EmbeddingModelMismatchError,
    document_metadata,
    index_documents,
    iter_batches,
    open_collection,
)
from tests.fakes import FakeEmbedder

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "reviews_sample.csv"


@pytest.fixture(scope="module")
def docs() -> pd.DataFrame:
    return load_documents(SAMPLE).head(30)


# ---- Embeddings ---------------------------------------------------------------------------------


def test_e5_prefixes():
    assert as_passage("chegou quebrado") == "passage: chegou quebrado"
    assert as_query("reclamações de entrega") == "query: reclamações de entrega"


class _RecordingModel:
    def __init__(self):
        self.inputs: list[str] = []
        self.kwargs: dict = {}

    def encode(self, texts, **kwargs):
        self.inputs.extend(texts)
        self.kwargs = kwargs
        return np.ones((len(texts), 4)) / 2


def _embedder_with(model) -> E5Embedder:
    # Evita o __init__ para não baixar o modelo: testa só como o embedder chama o SentenceTransformer.
    embedder = object.__new__(E5Embedder)
    embedder.model_name, embedder.batch_size, embedder._model = "fake", 16, model
    return embedder


def test_e5_embedder_applies_prefixes_and_normalizes():
    model = _RecordingModel()
    embedder = _embedder_with(model)

    documents = embedder.embed_documents(["produto ótimo", "não chegou"])
    query = embedder.embed_query("atraso")

    assert model.inputs == ["passage: produto ótimo", "passage: não chegou", "query: atraso"]
    assert model.kwargs["normalize_embeddings"] is True
    assert len(documents) == 2 and len(query) == 4


# ---- Metadados ----------------------------------------------------------------------------------


def test_document_metadata_types(docs):
    metadata = document_metadata(docs.iloc[0])
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
    assert isinstance(metadata["review_score"], int)
    assert isinstance(metadata["low_information"], bool)
    assert metadata["review_date_int"] == int(metadata["review_date"].replace("-", ""))


def test_document_metadata_omits_missing_delay_and_fills_order_status(docs):
    row = docs.iloc[0].copy()
    row["delay_days"] = pd.NA
    row["order_status"] = np.nan
    metadata = document_metadata(row)
    assert "delay_days" not in metadata
    assert metadata["order_status"] == "desconhecido"


def test_iter_batches():
    assert [list(batch) for batch in iter_batches(list(range(5)), 2)] == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError):
        list(iter_batches([1], 0))


# ---- Índice -------------------------------------------------------------------------------------


def test_index_is_idempotent_and_keeps_raw_text(tmp_path, docs):
    embedder = FakeEmbedder()
    collection = open_collection(tmp_path, embedder.model_name)

    assert index_documents(collection, docs, embedder, batch_size=7) == len(docs)
    index_documents(collection, docs, embedder, batch_size=7)  # segunda vez: atualiza, não duplica

    assert collection.count() == len(docs)
    first = docs.iloc[0]
    stored = collection.get(ids=[first["review_id"]])
    # O prefixo "passage:" só entra no embedding; o texto guardado é o que o cliente escreveu.
    assert stored["documents"] == [first["text"]]
    assert stored["metadatas"][0]["review_score"] == int(first["review_score"])
    assert max(len(call) for call in embedder.calls) == 7


def test_index_skips_empty_text(tmp_path, docs):
    embedder = FakeEmbedder()
    collection = open_collection(tmp_path, embedder.model_name)
    with_empty = docs.head(3).copy()
    with_empty.loc[with_empty.index[0], "text"] = "   "
    assert index_documents(collection, with_empty, embedder) == 2


def test_collection_uses_cosine_and_finds_same_text(tmp_path, docs):
    embedder = FakeEmbedder()
    collection = open_collection(tmp_path, embedder.model_name)
    index_documents(collection, docs, embedder)
    assert collection.configuration["hnsw"]["space"] == "cosine"

    target = docs.iloc[3]
    result = collection.query(query_embeddings=[embedder.vector(as_passage(target["text"]))], n_results=1)
    assert result["ids"][0][0] == target["review_id"]
    assert result["distances"][0][0] == pytest.approx(0.0, abs=1e-5)


def test_metadata_filters(tmp_path, docs):
    embedder = FakeEmbedder()
    collection = open_collection(tmp_path, embedder.model_name)
    index_documents(collection, docs, embedder)

    negatives = collection.get(where={"sentiment": "negativo"})
    assert len(negatives["ids"]) == int((docs["sentiment"] == "negativo").sum())
    assert all(meta["review_score"] <= 2 for meta in negatives["metadatas"])


def test_reopening_with_another_model_fails(tmp_path, docs):
    collection = open_collection(tmp_path, "fake-model")
    index_documents(collection, docs.head(2), FakeEmbedder())

    with pytest.raises(EmbeddingModelMismatchError):
        open_collection(tmp_path, "outro-modelo")

    rebuilt = open_collection(tmp_path, "outro-modelo", reset=True)
    assert rebuilt.count() == 0
