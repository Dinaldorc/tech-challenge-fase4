from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.embeddings import E5Embeddings, as_passage, as_query
from src.preprocessing import load_documents
from src.vector_store import (
    EmbeddingModelMismatchError,
    document_metadata,
    count_documents,
    index_documents,
    iter_batches,
    SEARCH_EF,
    open_vector_store,
    to_documents,
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


def _embedder_with(model) -> E5Embeddings:
    # Evita o __init__ para não baixar o modelo: testa só como o embedder chama o SentenceTransformer.
    embedder = object.__new__(E5Embeddings)
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


def test_e5_is_a_langchain_embeddings_component():
    assert issubclass(E5Embeddings, Embeddings)


# ---- Metadados ----------------------------------------------------------------------------------


def test_document_metadata_types(docs):
    metadata = document_metadata(docs.iloc[0])
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
    assert isinstance(metadata["review_score"], int)
    assert isinstance(metadata["low_information"], bool)
    assert metadata["purchase_date_int"] == int(metadata["purchase_date"].replace("-", ""))
    assert metadata["purchase_date"] <= metadata["review_date"]  # a compra vem antes da avaliação


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


# ---- Documentos do LangChain -------------------------------------------------------------------


def test_to_documents_uses_review_id_and_raw_text(docs):
    documents = to_documents(docs.head(3))
    first = docs.iloc[0]
    assert all(isinstance(document, Document) for document in documents)
    assert documents[0].id == first["review_id"]
    assert documents[0].page_content == first["text"]
    assert documents[0].metadata == document_metadata(first)


# ---- Índice -------------------------------------------------------------------------------------


def test_index_is_idempotent_and_keeps_raw_text(tmp_path, docs):
    embedder = FakeEmbedder()
    vector_store = open_vector_store(tmp_path, embedder)

    assert index_documents(vector_store, docs, batch_size=7) == len(docs)
    index_documents(vector_store, docs, batch_size=7)  # segunda vez: atualiza, não duplica

    assert count_documents(vector_store) == len(docs)
    first = docs.iloc[0]
    stored = vector_store.get(ids=[first["review_id"]])
    # O prefixo "passage:" só entra no embedding; o texto guardado é o que o cliente escreveu.
    assert stored["documents"] == [first["text"]]
    assert stored["metadatas"][0]["review_score"] == int(first["review_score"])
    assert max(len(call) for call in embedder.calls) == 7


def test_index_skips_empty_text(tmp_path, docs):
    vector_store = open_vector_store(tmp_path, FakeEmbedder())
    with_empty = docs.head(3).copy()
    with_empty.loc[with_empty.index[0], "text"] = "   "
    assert index_documents(vector_store, with_empty) == 2


def test_index_uses_cosine_and_langchain_returns_distance(tmp_path, docs):
    embedder = FakeEmbedder()
    vector_store = open_vector_store(tmp_path, embedder)
    index_documents(vector_store, docs)
    assert vector_store._collection.configuration["hnsw"]["space"] == "cosine"

    target = docs.iloc[3]
    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        embedder.vector(as_passage(target["text"])), k=1
    )
    document, score = results[0]
    assert document.id == target["review_id"]
    # Apesar do nome do método, o valor é a distância (0 para o texto idêntico), e não a similaridade.
    # A busca (src/retrieval.py) converte com 1 - distância; este teste avisa se o LangChain mudar.
    assert score == pytest.approx(0.0, abs=1e-5)


def test_metadata_filters(tmp_path, docs):
    vector_store = open_vector_store(tmp_path, FakeEmbedder())
    index_documents(vector_store, docs)

    negatives = vector_store.get(where={"sentiment": "negativo"})
    assert len(negatives["ids"]) == int((docs["sentiment"] == "negativo").sum())
    assert all(meta["review_score"] <= 2 for meta in negatives["metadatas"])


def test_reopening_with_another_model_fails(tmp_path, docs):
    vector_store = open_vector_store(tmp_path, FakeEmbedder())
    index_documents(vector_store, docs.head(2))

    other = FakeEmbedder(model_name="outro-modelo")
    with pytest.raises(EmbeddingModelMismatchError):
        open_vector_store(tmp_path, other)

    rebuilt = open_vector_store(tmp_path, other, reset=True)
    assert count_documents(rebuilt) == 0


def test_index_uses_near_exact_search_and_fixes_old_indexes(tmp_path):
    vector_store = open_vector_store(tmp_path, FakeEmbedder())
    assert vector_store._collection.configuration["hnsw"]["ef_search"] == SEARCH_EF
    # Um índice criado com o valor padrão é ajustado ao ser aberto.
    vector_store._collection.modify(configuration={"hnsw": {"ef_search": 100}})
    reopened = open_vector_store(tmp_path, FakeEmbedder())
    assert reopened._collection.configuration["hnsw"]["ef_search"] == SEARCH_EF

