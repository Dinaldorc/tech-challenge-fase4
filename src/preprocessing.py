"""Preparo da base de documentos do RAG a partir dos CSVs da Olist.

Implementa as decisões da análise exploratória (notebooks/01_eda_avaliacoes.ipynb, seção 10):

- um documento por ``review_id``, com título + comentário (entra quem tiver qualquer um dos dois);
- sem chunking: cada avaliação é um documento;
- limpeza leve no texto indexado (mantém caixa, acentos, emojis e negações);
- normalização de comparação numa coluna à parte, usada só para deduplicar evidências;
- marcação de baixa informação (".", "ok", "10");
- junções com ``left join``: pedidos sem itens e produtos sem categoria ficam na base;
- status da entrega por etapa do fluxo do pedido.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW_FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
}

ORDER_DATES = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Pedidos sem data de entrega ao cliente: o status depende de até onde o pedido avançou (EDA, seção 8.1).
NO_DELIVERY_DATE_STATUS = {
    "delivered": "entregue_sem_data",
    "shipped": "enviado_nao_recebido",
    "created": "nao_enviado",
    "approved": "nao_enviado",
    "processing": "nao_enviado",
    "invoiced": "nao_enviado",
    "canceled": "cancelado_indisponivel",
    "unavailable": "cancelado_indisponivel",
}

CATEGORY_NO_ITEMS = "sem_itens"
CATEGORY_UNKNOWN = "desconhecida"

DOCUMENT_COLUMNS = [
    "review_id",
    "order_id",
    "text",
    "text_norm",
    "low_information",
    "review_score",
    "sentiment",
    "review_creation_date",
    "review_comment_title",
    "review_comment_message",
    "order_status",
    "delivery_status",
    "delay_days",
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "product_category",
]


# ---- Texto -------------------------------------------------------------------------------------

_LINE_BREAKS = re.compile(r"[\r\n\t]+")
_SPACES = re.compile(r"\s+")
_REPEATED_END_PUNCT = re.compile(r"([!?.])\1{3,}")  # "!!!!!!" -> "!!!"
_REPEATED_SEPARATORS = re.compile(r"([,;:])\1+")  # ",,," -> ","
_HAS_WORD = re.compile(r"[a-z]{3,}")


def clean_text(text: str | None) -> str:
    """Limpeza leve do texto que vai para o índice.

    Remove quebras de linha, espaços extras e pontuação repetida em excesso. Mantém caixa,
    acentos, emojis e negações: o modelo de embeddings foi treinado com texto natural e
    "PÉSSIMO!!!" carrega intensidade.
    """
    if text is None or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _LINE_BREAKS.sub(" ", text)
    text = _REPEATED_END_PUNCT.sub(r"\1\1\1", text)
    text = _REPEATED_SEPARATORS.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


def normalize_for_comparison(text: str | None) -> str:
    """Minúsculas, sem acentos e sem pontuação. Serve só para comparar textos, nunca para indexar."""
    if text is None or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _SPACES.sub(" ", re.sub(r"[^\w\s]", " ", text)).strip()


def is_low_information(text_norm: str) -> bool:
    """Texto sem nenhuma palavra de 3 letras ou mais: ".", "ok", "10", "*****".

    Recebe o texto já normalizado. Marca cerca de 1% dos documentos.
    """
    return not _HAS_WORD.search(text_norm)


def join_title_and_message(title: str, message: str) -> str:
    """Junta título e comentário já limpos num só texto."""
    if not title:
        return message
    if not message:
        return title
    separator = " " if title[-1] in ".!?" else ". "
    return f"{title}{separator}{message}"


def score_to_sentiment(score: int) -> str:
    """Nota 1–2 = negativo, 3 = neutro, 4–5 = positivo (mesma divisão da análise exploratória)."""
    if score <= 2:
        return "negativo"
    if score == 3:
        return "neutro"
    return "positivo"


# ---- Metadados ---------------------------------------------------------------------------------


def add_delivery_status(orders: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ``delay_days`` e ``delivery_status`` aos pedidos.

    Com data de entrega: ``no_prazo`` (atraso <= 0 dias) ou ``atrasado``. Sem data: o grupo do
    fluxo do pedido (``NO_DELIVERY_DATE_STATUS``).
    """
    orders = orders.copy()
    orders["delay_days"] = (
        orders["order_delivered_customer_date"].dt.normalize()
        - orders["order_estimated_delivery_date"].dt.normalize()
    ).dt.days.astype("Int64")
    status = orders["order_status"].map(NO_DELIVERY_DATE_STATUS).fillna("desconhecido")
    # Sem data de entrega, a comparação dá NA: fillna(False) evita que o NA conte como verdadeiro.
    status = status.mask((orders["delay_days"] <= 0).fillna(False), "no_prazo")
    status = status.mask((orders["delay_days"] > 0).fillna(False), "atrasado")
    orders["delivery_status"] = status
    return orders


def order_categories(orders: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Categoria de cada pedido, pelo primeiro item.

    Separa os dois casos de "sem categoria" (EDA, seção 9.1): pedido sem nenhum item
    (``sem_itens``) e produto sem categoria cadastrada (``desconhecida``).
    """
    first_item = items.sort_values(["order_id", "order_item_id"]).drop_duplicates("order_id")
    category = (
        first_item[["order_id", "product_id"]]
        .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        .set_index("order_id")["product_category_name"]
    )
    result = orders[["order_id"]].copy()
    result["product_category"] = result["order_id"].map(category).fillna(CATEGORY_UNKNOWN)
    result.loc[~result["order_id"].isin(items["order_id"]), "product_category"] = CATEGORY_NO_ITEMS
    return result


# ---- Pipeline ----------------------------------------------------------------------------------


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    return {
        "reviews": pd.read_csv(
            raw_dir / RAW_FILES["reviews"], parse_dates=["review_creation_date", "review_answer_timestamp"]
        ),
        "orders": pd.read_csv(raw_dir / RAW_FILES["orders"], parse_dates=ORDER_DATES),
        "items": pd.read_csv(raw_dir / RAW_FILES["items"]),
        "products": pd.read_csv(raw_dir / RAW_FILES["products"]),
    }


def build_documents(
    reviews: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    """Gera a base de documentos do RAG: uma linha por avaliação com título ou comentário."""
    # Mesmo review_id ligado a vários pedidos do mesmo cliente: mantém a primeira ocorrência.
    docs = reviews.drop_duplicates("review_id").copy()

    title = docs["review_comment_title"].map(clean_text)
    message = docs["review_comment_message"].map(clean_text)
    docs["text"] = [join_title_and_message(t, m) for t, m in zip(title, message)]
    docs = docs[docs["text"] != ""].copy()

    docs["text_norm"] = docs["text"].map(normalize_for_comparison)
    docs["low_information"] = docs["text_norm"].map(is_low_information)
    docs["sentiment"] = docs["review_score"].map(score_to_sentiment)

    orders = add_delivery_status(orders)
    order_meta = orders[
        [
            "order_id",
            "order_status",
            "delivery_status",
            "delay_days",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ].merge(order_categories(orders, items, products), on="order_id", how="left")

    docs = docs.merge(order_meta, on="order_id", how="left")
    # Avaliação sem pedido correspondente (não acontece na base atual, mas não derruba o documento).
    docs["delivery_status"] = docs["delivery_status"].fillna("desconhecido")
    docs["product_category"] = docs["product_category"].fillna(CATEGORY_UNKNOWN)

    return docs[DOCUMENT_COLUMNS].sort_values("review_id").reset_index(drop=True)


def sample_documents(docs: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Amostra estratificada por nota, mantendo a proporção de cada nota na base."""
    if n >= len(docs):
        return docs.copy()
    fraction = n / len(docs)
    sample = docs.groupby("review_score", group_keys=False).sample(frac=fraction, random_state=seed)
    return sample.sort_values("review_id").reset_index(drop=True)


def load_documents(path: Path) -> pd.DataFrame:
    """Lê a base gerada por ``scripts/prepare_data.py`` (Parquet ou CSV)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    date_columns = [
        "review_creation_date",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    docs = pd.read_csv(path, parse_dates=date_columns, keep_default_na=False, na_values=[""])
    docs["delay_days"] = docs["delay_days"].astype("Int64")
    for column in ["review_comment_title", "review_comment_message", "text_norm"]:
        docs[column] = docs[column].fillna("")
    return docs
