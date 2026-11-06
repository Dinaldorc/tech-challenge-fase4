"""Índice vetorial das avaliações, acessado pelo ``Chroma`` do LangChain (``langchain-chroma``).

Decisões:

- **Persistência em disco** (``chroma_db/``): o índice é montado uma vez por ``scripts/build_index.py``
  e reaproveitado pela aplicação, sem recalcular embeddings a cada execução. A pasta não é
  versionada, porque o script a recria.
- **Embeddings do projeto** (``src/embeddings.py``) entregues ao ``Chroma`` do LangChain, que cria a
  coleção sem função de embedding própria do banco. Assim valem os prefixos ``passage:``/``query:``
  do E5, e não o modelo padrão do Chroma (voltado ao inglês).
- **Distância cosseno**, a métrica com que o E5 foi treinado. O LangChain lê essa configuração da
  coleção e converte a distância em similaridade com ``1 - distância``.
- **Busca quase exata** (``ef_search`` = 10.000). O Chroma usa o HNSW, uma busca aproximada: em vez de
  comparar a pergunta com todas as avaliações, percorre um grafo de vizinhos. Com o valor padrão (100),
  a busca recuperava em média 81% dos 32 candidatos corretos e, em 25% das perguntas, errava a melhor
  similaridade, que é o número comparado com o corte de recusa. Com 10.000, o recall das perguntas
  legítimas chega a 99,8%, sem custo de tempo perceptível nesta base (README, "Busca quase exata").
- **O nome do modelo fica gravado na coleção.** Vetores de modelos diferentes não são comparáveis,
  então reabrir o índice com outro modelo gera erro em vez de devolver resultados sem sentido.
- **Um ``Document`` por avaliação**, com ``review_id`` como identificador. A gravação usa ``upsert``:
  rodar a indexação de novo atualiza os documentos em vez de duplicá-los.
- **Metadados** para filtrar a busca (nota, sentimento, status da entrega, categoria, data da compra) e para
  citar a evidência na resposta. Os textos de baixa informação ("ok", ".") são indexados com a
  marca ``low_information``, e a busca decide se os exclui.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_COLLECTION = "olist_reviews"
DISTANCE = "cosine"
SEARCH_EF = 10_000
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
        # Datas como texto, para exibir na evidência, e a da compra também como inteiro AAAAMMDD, para
        # filtrar por período ($gte/$lte), já que o Chroma só compara números. O período usa a data da
        # compra, como pede o enunciado ("período da compra"); a da avaliação fica para exibição.
        "review_date": review_date.strftime("%Y-%m-%d"),
    }
    if pd.notna(row.get("order_purchase_timestamp")):
        purchase = pd.Timestamp(row["order_purchase_timestamp"])
        metadata["purchase_date"] = purchase.strftime("%Y-%m-%d")
        metadata["purchase_date_int"] = int(purchase.strftime("%Y%m%d"))
    # Sem data de entrega não há atraso: o campo fica ausente, em vez de um valor inventado.
    if pd.notna(row["delay_days"]):
        metadata["delay_days"] = int(row["delay_days"])
    return metadata


def to_documents(docs: pd.DataFrame) -> list[Document]:
    """Converte a base de avaliações em ``Document``s do LangChain, ignorando textos vazios."""
    docs = docs[docs["text"].fillna("").str.strip() != ""]
    columns = list(docs.columns)
    documents = []
    for row in docs.itertuples(index=False):
        record = pd.Series(row, index=columns)
        documents.append(Document(page_content=record["text"], metadata=document_metadata(record),
                                  id=record["review_id"]))
    return documents


def iter_batches(items: Sequence, size: int) -> Iterator[Sequence]:
    if size < 1:
        raise ValueError("O tamanho do lote deve ser positivo.")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def open_vector_store(
    persist_dir: Path,
    embeddings: Embeddings,
    name: str = DEFAULT_COLLECTION,
    reset: bool = False,
) -> Chroma:
    """Abre (ou cria) o índice persistente, garantindo que ele usa o mesmo modelo de embeddings."""
    model_name = getattr(embeddings, "model_name", type(embeddings).__name__)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset and name in {collection.name for collection in client.list_collections()}:
        client.delete_collection(name)

    vector_store = Chroma(
        client=client,
        collection_name=name,
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": DISTANCE, "ef_search": SEARCH_EF}},
        collection_metadata={MODEL_METADATA_KEY: model_name},
    )
    indexed_with = (vector_store._collection.metadata or {}).get(MODEL_METADATA_KEY)
    if indexed_with != model_name:
        raise EmbeddingModelMismatchError(
            f"A coleção '{name}' foi indexada com '{indexed_with}', não com '{model_name}'. "
            "Recrie o índice com scripts/build_index.py --reset."
        )
    # Índices criados antes desta configuração: ajusta o ef_search. Vale a partir daqui, desde que
    # nenhuma consulta tenha carregado o índice antes neste processo.
    hnsw = vector_store._collection.configuration.get("hnsw") or {}
    if hnsw.get("ef_search") != SEARCH_EF:
        vector_store._collection.modify(configuration={"hnsw": {"ef_search": SEARCH_EF}})
    return vector_store


def count_documents(vector_store: Chroma) -> int:
    # O Chroma do LangChain não expõe a contagem; a coleção subjacente sim.
    return vector_store._collection.count()


def index_documents(
    vector_store: Chroma,
    docs: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> int:
    """Gera os embeddings e grava os documentos no índice. Retorna quantos foram gravados."""
    documents = to_documents(docs)
    written = 0
    for batch in iter_batches(documents, batch_size):
        vector_store.add_documents(list(batch), ids=[document.id for document in batch])
        written += len(batch)
        if on_batch is not None:
            on_batch(written, len(documents))
    return written
