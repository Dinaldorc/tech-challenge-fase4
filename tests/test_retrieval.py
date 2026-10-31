from datetime import date
from pathlib import Path

import pytest

from src.preprocessing import load_documents, normalize_for_comparison
from src.retrieval import (
    Evidence,
    Retriever,
    SearchFilters,
    build_where,
    deduplicate,
    infer_sentiment,
)
from src.vector_store import index_documents, open_collection
from tests.fakes import FakeEmbedder

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "reviews_sample.csv"

# ---- Inferência do sentimento ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Quais as principais reclamações sobre entrega?", ("negativo",)),
        ("os clientes reclamam do rastreamento?", ("negativo",)),
        ("Que problemas aparecem nos relógios?", ("negativo",)),
        ("cliente INSATISFEITO com a cor", ("negativo",)),
        ("Quais elogios os clientes fazem?", ("positivo",)),
        ("pontos positivos da loja", ("positivo",)),
        # Perguntas de julgamento pedem os dois lados: sem filtro.
        ("Os clientes ficaram satisfeitos?", None),
        ("os clientes recomendariam a loja?", None),
        ("a qualidade é ruim?", None),
        # Ambígua ou neutra: sem filtro.
        ("pontos positivos e negativos da entrega", None),
        ("o que os clientes acham da embalagem?", None),
    ],
)
def test_infer_sentiment(question, expected):
    assert infer_sentiment(question) == expected


# ---- Filtros -----------------------------------------------------------------------------------


def test_build_where_default_excludes_low_information():
    assert build_where(SearchFilters()) == {"low_information": False}
    assert build_where(SearchFilters(include_low_information=True)) is None


def test_build_where_combines_filters():
    where = build_where(
        SearchFilters(
            sentiments=("negativo",),
            categories=("moveis_decoracao", "cama_mesa_banho"),
            date_from=date(2018, 1, 1),
            date_to=date(2018, 6, 30),
        )
    )
    assert where == {
        "$and": [
            {"low_information": False},
            {"sentiment": "negativo"},
            {"product_category": {"$in": ["moveis_decoracao", "cama_mesa_banho"]}},
            {"review_date_int": {"$gte": 20180101}},
            {"review_date_int": {"$lte": 20180630}},
        ]
    }


# ---- Repetições --------------------------------------------------------------------------------


def test_deduplicate_groups_normalized_texts_and_keeps_the_best():
    candidates = [
        Evidence("a", "Não recebi o produto", 0.93, {}),
        Evidence("b", "Produto quebrado", 0.90, {}),
        Evidence("c", "nao recebi o produto!!", 0.92, {}),
        Evidence("d", "NÃO RECEBI O PRODUTO", 0.91, {}),
    ]
    result = deduplicate(candidates)
    assert [e.review_id for e in result] == ["a", "b"]
    assert result[0].duplicate_ids == ("c", "d")
    assert result[0].mentions == 3


# ---- Busca -------------------------------------------------------------------------------------


@pytest.fixture()
def retriever(tmp_path):
    """Índice pequeno com textos controlados e um embedder em que pergunta = documento idêntico."""
    docs = load_documents(SAMPLE).head(6).copy()
    rows = [
        ("Atraso na entrega", "negativo", 1, False),
        ("atraso na entrega!", "negativo", 2, False),
        ("Entrega no prazo", "positivo", 5, False),
        ("Produto quebrado", "negativo", 1, False),
        ("ok", "positivo", 5, True),
        ("Ótimo produto", "positivo", 5, False),
    ]
    for i, (text, sentiment, score, low_info) in enumerate(rows):
        docs.loc[docs.index[i], ["text", "sentiment", "review_score", "low_information"]] = [
            text, sentiment, score, low_info,
        ]
    docs["text_norm"] = docs["text"].map(normalize_for_comparison)

    embedder = FakeEmbedder(symmetric=True)
    collection = open_collection(tmp_path, embedder.model_name)
    index_documents(collection, docs, embedder)
    return Retriever(collection, embedder, min_similarity=0.0)


def test_search_infers_sentiment_and_groups_repetitions(retriever):
    result = retriever.search("reclamações: Atraso na entrega", k=5)
    assert result.filters.sentiments == ("negativo",)
    assert all(e.metadata["sentiment"] == "negativo" for e in result.evidences)
    # "Atraso na entrega" e "atraso na entrega!" viram uma evidência só.
    texts = [normalize_for_comparison(e.text) for e in result.evidences]
    assert texts.count("atraso na entrega") == 1
    assert next(e for e in result.evidences if normalize_for_comparison(e.text) == "atraso na entrega").mentions == 2


def test_search_respects_explicit_filters(retriever):
    result = retriever.search("reclamações", k=5, filters=SearchFilters(sentiments=("positivo",)))
    assert result.filters.sentiments == ("positivo",)
    assert {e.metadata["sentiment"] for e in result.evidences} == {"positivo"}

    by_score = retriever.search("reclamações", k=5, filters=SearchFilters(review_scores=(5,)))
    assert by_score.filters.sentiments is None  # nota escolhida pelo usuário: não infere sentimento
    assert {e.metadata["review_score"] for e in by_score.evidences} == {5}


def test_search_excludes_low_information_by_default(retriever):
    assert "ok" not in [e.text for e in retriever.search("ok", k=10).evidences]
    with_low = retriever.search("ok", k=10, filters=SearchFilters(include_low_information=True))
    assert with_low.evidences[0].text == "ok"


def test_threshold_removes_weak_evidence_and_signals_refusal(retriever):
    exact = retriever.search("Produto quebrado", k=3, infer_filters=False)
    assert exact.evidences[0].text == "Produto quebrado"
    assert exact.best_similarity == pytest.approx(1.0, abs=1e-5)

    retriever.min_similarity = 0.999
    strict = retriever.search("Produto quebrado", k=3, infer_filters=False)
    assert [e.text for e in strict.evidences] == ["Produto quebrado"]

    unrelated = retriever.search("qual a capital da França?", k=3, infer_filters=False)
    assert not unrelated.has_evidence
    assert unrelated.best_similarity is not None and unrelated.best_similarity < 0.999


def test_search_respects_k(retriever):
    assert len(retriever.search("produto", k=2, infer_filters=False).evidences) == 2


def test_evidence_is_immutable():
    evidence = Evidence("a", "texto", 0.9, {})
    with pytest.raises(AttributeError):
        evidence.text = "outro"  # type: ignore[misc]


def test_search_widens_window_when_repetitions_fill_it(tmp_path):
    # 40 cópias do mesmo texto ocupariam toda a janela inicial (k * 4 = 12 candidatos).
    docs = load_documents(SAMPLE).head(45).copy()
    docs["text"] = ["Não recebi o produto"] * 40 + [f"Comentário diferente {i}" for i in range(5)]
    docs["low_information"] = False
    embedder = FakeEmbedder(symmetric=True)
    collection = open_collection(tmp_path, embedder.model_name)
    index_documents(collection, docs, embedder)

    result = Retriever(collection, embedder, min_similarity=-1.0).search("Não recebi o produto", k=3, infer_filters=False)
    assert len(result.evidences) == 3
    assert result.evidences[0].text == "Não recebi o produto"
    assert result.evidences[0].mentions == 40
