from datetime import date

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from src.analytics import THEME_LABELS, AnalyticsFilters, ThemeAnalytics
from src.themes import PROMPT, THEMES, BatchThemes, ReviewThemes, describe_themes, format_reviews, parse_batch

# ---- Classificação -----------------------------------------------------------------------------


def test_every_theme_has_a_definition_and_a_label():
    assert set(THEMES) == set(THEME_LABELS)
    assert all(definition for definition in THEMES.values())
    assert {"outro", "sem_reclamacao"} <= set(THEMES)


def test_format_reviews_numbers_each_review_on_one_line():
    text = format_reviews(["Não recebi\no produto", "Veio quebrado"])
    assert text == "[1] Não recebi o produto\n[2] Veio quebrado"


def test_prompt_lists_all_themes_and_rules():
    messages = PROMPT.invoke({"themes": describe_themes(), "reviews": "[1] x"}).to_messages()
    assert all(f"- {theme}:" in messages[0].content for theme in THEMES)
    assert "mais de um tema" in messages[0].content
    assert "pseudônimos de lojas" in messages[0].content
    assert messages[1].content == "Avaliações:\n\n[1] x"


def test_parse_batch_keeps_order_and_counts_missing():
    parsed = BatchThemes(items=[
        ReviewThemes(number=2, themes=["atraso", "nao_recebimento", "atraso"]),
        ReviewThemes(number=1, themes=["defeito_qualidade"]),
    ])
    themes, missing = parse_batch({"raw": AIMessage(content=""), "parsed": parsed}, size=3)
    assert themes == [["defeito_qualidade"], ["atraso", "nao_recebimento"], []]
    assert missing == 1


def test_parse_batch_without_structured_answer():
    themes, missing = parse_batch({"raw": AIMessage(content=""), "parsed": None}, size=2)
    assert themes == [[], []] and missing == 2


def test_schema_rejects_unknown_theme():
    with pytest.raises(ValueError):
        ReviewThemes(number=1, themes=["tema_inventado"])


# ---- Contagens com DuckDB ----------------------------------------------------------------------


@pytest.fixture()
def analytics(tmp_path):
    documents = pd.DataFrame({
        "review_id": ["a", "b", "c", "d"],
        "text": ["Não recebi", "Veio quebrado e atrasou", "Comprei dois, recebi um", "Não chegou, sem resposta"],
        "review_score": [1, 2, 1, 1],
        "product_category": ["moveis", "moveis", "moveis", "beleza"],
        "delivery_status": ["atrasado", "no_prazo", "no_prazo", "atrasado"],
        "review_creation_date": pd.to_datetime(["2018-01-10", "2018-02-10", "2018-02-20", "2018-03-05"]),
        # "d" foi comprada em fevereiro e avaliada em março: o período usa a data da compra.
        "order_purchase_timestamp": pd.to_datetime(["2018-01-03", "2018-02-03", "2018-02-14", "2018-02-25"]),
    })
    documents.to_parquet(tmp_path / "docs.parquet", index=False)
    pd.DataFrame({
        "review_id": ["a", "b", "c", "d"],
        "themes": ["nao_recebimento", "atraso;defeito_qualidade", "entrega_parcial", "atendimento;nao_recebimento"],
    }).to_csv(tmp_path / "themes.csv", index=False)
    return ThemeAnalytics(tmp_path / "themes.csv", tmp_path / "docs.parquet")


def test_theme_counts_are_multilabel_shares(analytics):
    counts = analytics.theme_counts()
    assert analytics.total() == 4
    top = counts.iloc[0]
    assert (top.theme, top.reviews, top.share) == ("nao_recebimento", 2, 0.5)
    assert counts["reviews"].sum() == 6  # uma avaliação pode ter mais de um tema
    assert counts["share"].sum() > 1
    assert top.label == "Não recebimento"


def test_filters_restrict_the_base(analytics):
    moveis = AnalyticsFilters(categories=("moveis",))
    assert analytics.total(moveis) == 3
    counts = analytics.theme_counts(moveis).set_index("theme")
    assert counts.loc["nao_recebimento", "share"] == pytest.approx(1 / 3)
    period = AnalyticsFilters(date_from=date(2018, 2, 1), date_to=date(2018, 2, 28))
    assert analytics.total(period) == 3  # pela data da avaliação seriam 2


def test_theme_by_delivery_status(analytics):
    table = analytics.theme_by("delivery_status").set_index(["value", "theme"])
    assert table.loc[("atrasado", "nao_recebimento"), "share"] == 1.0
    assert table.loc[("no_prazo", "entrega_parcial"), "total"] == 2
    with pytest.raises(ValueError):
        analytics.theme_by("text")


def test_examples_and_monthly(analytics):
    examples = analytics.examples("nao_recebimento")
    assert set(examples["review_id"]) == {"a", "d"}
    monthly = analytics.monthly()
    assert monthly["reviews"].sum() == 6
