"""Índice vetorial das avaliações no ChromaDB.

Decisões:

- **Persistência em disco** (``chroma_db/``): o índice é montado uma vez por ``scripts/build_index.py``
  e reaproveitado pela aplicação, sem recalcular embeddings a cada execução. A pasta não é
  versionada, porque o script a recria.
- **Embeddings calculados por nós** (``src/embeddings.py``) e entregues prontos ao Chroma, com a
  coleção criada sem função de embedding própria. Sem isso, o Chroma usaria o modelo padrão dele
  (inglês) e ignoraria os prefixos ``passage:``/``query:`` do E5.
- **Distância cosseno**, a métrica com que o E5 foi treinado.
- **O nome do modelo fica gravado na coleção.** Vetores de modelos diferentes não são comparáveis,
  então reabrir o índice com outro modelo gera erro em vez de devolver resultados sem sentido.
- **Um documento por avaliação**, com ``review_id`` como identificador. A gravação usa ``upsert``:
  rodar a indexação de novo atualiza os documentos em vez de duplicá-los.
- **Metadados** para filtrar a busca (nota, sentimento, status da entrega, categoria, data) e para
  citar a evidência na resposta. Os textos de baixa informação ("ok", ".") são indexados com a
  marca ``low_information``, e a busca decide se os exclui.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import chromadb
import pandas as pd

from src.embeddings import Embedder

DEFAULT_COLLECTION = "olist_reviews"
DISTANCE = "cosine"
MODEL_METADATA_KEY = "embedding_model"
# O Chroma aceita no máximo ~5.400 registros por chamada. Lotes menores mostram o progresso e
# limitam a memória usada pelos embeddings.
DEFAULT_BATCH_SIZE = 256


class EmbeddingModelMismatchError(ValueError):
    """A coleção existente foi indexada com outro modelo de embeddings."""


def document_metadata(row: pd.Series) -> dict[str, str | int | float | bool]:
    """Metadados de um documento no formato aceito pelo Chroma (str, int, float ou bool, sem nulos)."""
    review_date = pd.Timestamp(row["review_creation_date"])
    metadata: dict[str, str | int | float | bool] = {
        "order_id": str(row["order_id"]),
        "review_score": int(row["review_score"]),
        "sentiment": str(row["sentiment"]),
        "low_information": bool(row["low_information"]),
        "order_status": str(row["order_status"]) if pd.notna(row["order_status"]) else "desconhecido",
        "delivery_status": str(row["delivery_status"]),
        "product_category": str(row["product_category"]),
        # Texto para exibir na evidência; inteiro AAAAMMDD para filtrar por período ($gte/$lte),
        # já que o Chroma só compara números.
        "review_date": review_date.strftime("%Y-%m-%d"),
        "review_date_int": int(review_date.strftime("%Y%m%d")),
    }
    # Sem data de entrega não há atraso: o campo fica ausente, em vez de um valor inventado.
    if pd.notna(row["delay_days"]):
        metadata["delay_days"] = int(row["delay_days"])
    return metadata


def iter_batches(items: Sequence, size: int) -> Iterator[Sequence]:
    if size < 1:
        raise ValueError("O tamanho do lote deve ser positivo.")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def open_collection(
    persist_dir: Path,
    embedding_model: str,
    name: str = DEFAULT_COLLECTION,
    reset: bool = False,
) -> chromadb.Collection:
    """Abre (ou cria) a coleção persistente, garantindo que ela usa o mesmo modelo de embeddings."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset and name in {collection.name for collection in client.list_collections()}:
        client.delete_collection(name)

    collection = client.get_or_create_collection(
        name,
        configuration={"hnsw": {"space": DISTANCE}},
        metadata={MODEL_METADATA_KEY: embedding_model},
        embedding_function=None,
    )
    indexed_with = (collection.metadata or {}).get(MODEL_METADATA_KEY)
    if indexed_with != embedding_model:
        raise EmbeddingModelMismatchError(
            f"A coleção '{name}' foi indexada com '{indexed_with}', não com '{embedding_model}'. "
            "Recrie o índice com scripts/build_index.py --reset."
        )
    return collection


def index_documents(
    collection: chromadb.Collection,
    docs: pd.DataFrame,
    embedder: Embedder,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> int:
    """Gera os embeddings e grava os documentos na coleção. Retorna quantos foram gravados."""
    docs = docs[docs["text"].fillna("").str.strip() != ""]
    rows = list(docs.itertuples(index=False))
    columns = list(docs.columns)
    written = 0
    for batch in iter_batches(rows, batch_size):
        records = [pd.Series(row, index=columns) for row in batch]
        texts = [record["text"] for record in records]
        collection.upsert(
            ids=[record["review_id"] for record in records],
            embeddings=embedder.embed_documents(texts),
            documents=texts,
            metadatas=[document_metadata(record) for record in records],
        )
        written += len(records)
        if on_batch is not None:
            on_batch(written, len(rows))
    return written
