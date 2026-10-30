import pandas as pd
import pytest

from src.preprocessing import (
    add_delivery_status,
    build_documents,
    clean_text,
    is_low_information,
    join_title_and_message,
    load_documents,
    normalize_for_comparison,
    sample_documents,
    score_to_sentiment,
)

# ---- Texto -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Produto\r\nchegou   quebrado", "Produto chegou quebrado"),
        ("PÉSSIMO!!!!!!!", "PÉSSIMO!!!"),
        ("não recebi...", "não recebi..."),
        ("ok,,, valeu", "ok, valeu"),
        ("  Ótimo 😍  ", "Ótimo 😍"),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_preserves_negation_case_and_accents():
    assert clean_text("NÃO chegou") == "NÃO chegou"


def test_normalize_for_comparison_unifies_spellings():
    spellings = ["Ótimo!", "otimo", "ÓTIMO", "  Otimo!!! "]
    assert {normalize_for_comparison(s) for s in spellings} == {"otimo"}


@pytest.mark.parametrize("text", [".", "ok", "10", "*****", "ok ok", "100%", ""])
def test_low_information(text):
    assert is_low_information(normalize_for_comparison(text))


@pytest.mark.parametrize("text", ["bom", "não recebi", "Ótimo produto"])
def test_not_low_information(text):
    assert not is_low_information(normalize_for_comparison(text))


@pytest.mark.parametrize(
    ("title", "message", "expected"),
    [
        ("Ruim", "Veio quebrado", "Ruim. Veio quebrado"),
        ("Ruim!", "Veio quebrado", "Ruim! Veio quebrado"),
        ("", "Veio quebrado", "Veio quebrado"),
        ("Não recebi", "", "Não recebi"),
    ],
)
def test_join_title_and_message(title, message, expected):
    assert join_title_and_message(title, message) == expected


def test_score_to_sentiment():
    assert [score_to_sentiment(s) for s in range(1, 6)] == ["negativo", "negativo", "neutro", "positivo", "positivo"]


# ---- Tabelas sintéticas ------------------------------------------------------------------------


def _orders() -> pd.DataFrame:
    ts = pd.Timestamp
    return pd.DataFrame(
        {
            "order_id": ["o_prazo", "o_atraso", "o_enviado", "o_cancel", "o_sem_itens", "o_sem_cat"],
            "order_status": ["delivered", "delivered", "shipped", "canceled", "unavailable", "delivered"],
            "order_purchase_timestamp": [ts("2018-01-01")] * 6,
            "order_approved_at": [ts("2018-01-01")] * 6,
            "order_delivered_carrier_date": [ts("2018-01-02"), ts("2018-01-02"), ts("2018-01-02"), None, None,
                                             ts("2018-01-02")],
            "order_delivered_customer_date": [ts("2018-01-10 15:00"), ts("2018-01-25"), None, None, None,
                                              ts("2018-01-05")],
            "order_estimated_delivery_date": [ts("2018-01-10")] * 6,
        }
    )


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o_prazo", "o_prazo", "o_atraso", "o_enviado", "o_cancel", "o_sem_cat"],
            "order_item_id": [2, 1, 1, 1, 1, 1],
            "product_id": ["p_moveis", "p_beleza", "p_beleza", "p_beleza", "p_beleza", "p_sem_cat"],
        }
    )


def _products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["p_beleza", "p_moveis", "p_sem_cat"],
            "product_category_name": ["beleza_saude", "moveis_decoracao", None],
        }
    )


def _reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r2", "r3", "r4", "r5", "r6", "r7"],
            "order_id": ["o_prazo", "o_atraso", "o_enviado", "o_enviado", "o_cancel", "o_sem_itens", "o_sem_cat",
                         "o_prazo"],
            "review_score": [5, 1, 1, 2, 1, 1, 4, 5],
            "review_comment_title": [None, "Atrasou", "Atrasou", "Não recebi", None, None, "", None],
            "review_comment_message": ["Chegou\r\nno prazo", "Demorou demais!!!!!", "Demorou demais!!!!!", None,
                                       "Cancelaram sem avisar", "Pago e nunca enviado", "ok", None],
            "review_creation_date": [pd.Timestamp("2018-02-01")] * 8,
            "review_answer_timestamp": [pd.Timestamp("2018-02-02")] * 8,
        }
    )


@pytest.fixture
def docs() -> pd.DataFrame:
    return build_documents(_reviews(), _orders(), _items(), _products()).set_index("review_id")


def test_delivery_status():
    status = add_delivery_status(_orders()).set_index("order_id")["delivery_status"]
    assert status.to_dict() == {
        "o_prazo": "no_prazo",  # entregue no dia previsto, mesmo à tarde
        "o_atraso": "atrasado",
        "o_enviado": "enviado_nao_recebido",
        "o_cancel": "cancelado_indisponivel",
        "o_sem_itens": "cancelado_indisponivel",
        "o_sem_cat": "no_prazo",
    }


def test_one_document_per_review_id(docs):
    assert docs.index.is_unique
    assert docs.loc["r2", "order_id"] == "o_atraso"  # primeira ocorrência


def test_reviews_without_text_are_dropped(docs):
    assert "r7" not in docs.index


def test_title_only_reviews_are_kept(docs):
    assert docs.loc["r3", "text"] == "Não recebi"


def test_document_text_is_lightly_cleaned(docs):
    assert docs.loc["r1", "text"] == "Chegou no prazo"
    assert docs.loc["r2", "text"] == "Atrasou. Demorou demais!!!"
    assert docs.loc["r2", "text_norm"] == "atrasou demorou demais"


def test_metadata(docs):
    assert docs.loc["r2", "delivery_status"] == "atrasado"
    assert docs.loc["r2", "delay_days"] == 15
    assert docs.loc["r2", "sentiment"] == "negativo"
    assert docs.loc["r6", "low_information"]
    assert not docs.loc["r1", "low_information"]


def test_categories(docs):
    assert docs.loc["r1", "product_category"] == "beleza_saude"  # categoria do primeiro item
    assert docs.loc["r5", "product_category"] == "sem_itens"
    assert docs.loc["r6", "product_category"] == "desconhecida"


def test_sample_keeps_score_proportions():
    docs = pd.DataFrame({"review_id": [f"r{i}" for i in range(100)], "review_score": [1] * 20 + [5] * 80})
    sample = sample_documents(docs, 10, seed=1)
    assert sample["review_score"].value_counts().to_dict() == {5: 8, 1: 2}


def test_csv_round_trip(docs, tmp_path):
    path = tmp_path / "docs.csv"
    docs.reset_index().to_csv(path, index=False)
    loaded = load_documents(path).set_index("review_id")
    assert loaded.loc["r2", "delay_days"] == 15
    assert loaded.loc["r3", "review_comment_message"] == ""
    assert loaded.loc["r2", "order_purchase_timestamp"] == pd.Timestamp("2018-01-01")
