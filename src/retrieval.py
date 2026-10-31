"""Busca das avaliações que servem de evidência para responder a uma pergunta.

Etapas de uma busca:

1. **Filtros.** O usuário pode escolher nota, sentimento, status da entrega, categoria e período.
   Quando ele não escolhe o sentimento, a busca tenta inferi-lo da pergunta ("reclamações" indica
   negativo, "elogios" indica positivo). O embedding acerta o assunto, mas confunde elogio com
   reclamação (README, Achado 1), e o filtro pela nota do cliente corrige isso.
2. **Similaridade.** A pergunta vira embedding (prefixo ``query:``) e o Chroma devolve os comentários
   mais próximos dentro dos filtros. Os textos de baixa informação ("ok", ".", "*****") ficam fora
   por padrão, porque atraem perguntas fora do tema (Achado 3).
3. **Repetições.** Comentários iguais depois da normalização ("Não recebi o produto" e "nao recebi
   o produto") viram uma única evidência, que guarda quantas vezes o texto apareceu. Assim, as
   evidências enviadas ao LLM não se repetem, e o volume de cada queixa não se perde.
4. **Corte de similaridade.** Evidências abaixo de ``min_similarity`` são descartadas. Se nenhuma
   passar, a busca informa que não há evidência e o sistema deve recusar a pergunta. O valor padrão
   foi calibrado com perguntas dentro e fora do tema (``scripts/calibrate_threshold.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date

import chromadb

from src.embeddings import Embedder
from src.preprocessing import normalize_for_comparison

# Calibrado na base completa (42.370 documentos) com scripts/calibrate_threshold.py: o maior corte
# que não recusa nenhuma pergunta legítima. Na amostra de 3.000, o mesmo critério dá 0,83, porque as
# similaridades sobem com o tamanho da base. Trocar a base exige recalibrar (README, "Busca").
DEFAULT_MIN_SIMILARITY = 0.84
DEFAULT_K = 8
# Busca mais candidatos do que o necessário para sobrar k evidências depois de juntar as repetidas.
# Se as repetições consumirem a janela ("não recebi o produto" aparece 30 vezes), ela dobra até o
# teto, até o último candidato já estar abaixo do corte ou até os documentos acabarem.
CANDIDATE_FACTOR = 4
MAX_CANDIDATES = 512

# Palavras com que a pergunta pede explicitamente um dos lados, comparadas com o texto normalizado
# (minúsculas, sem acentos). Os radicais cobrem as variações: "reclama", "reclamam", "reclamação".
# Ficam de fora palavras de julgamento ("satisfeitos?", "é ruim?", "recomendariam?"): essas
# perguntas pedem os dois lados, e filtrar um deles enviesaria a resposta.
NEGATIVE_INTENT = re.compile(r"\b(reclama\w*|problema\w*|insatisf\w*|negativ\w*|critica\w*|queixa\w*|frustra\w*)\b")
POSITIVE_INTENT = re.compile(r"\b(elogi\w*|positiv\w*)\b")


@dataclass(frozen=True)
class SearchFilters:
    """Filtros de metadados. ``None`` significa "sem filtro" naquele campo."""

    sentiments: tuple[str, ...] | None = None
    review_scores: tuple[int, ...] | None = None
    delivery_statuses: tuple[str, ...] | None = None
    categories: tuple[str, ...] | None = None
    date_from: date | None = None
    date_to: date | None = None
    include_low_information: bool = False


def infer_sentiment(question: str) -> tuple[str, ...] | None:
    """Sentimento indicado pela pergunta, ou ``None`` se ela for neutra ou ambígua."""
    text = normalize_for_comparison(question)
    negative = bool(NEGATIVE_INTENT.search(text))
    positive = bool(POSITIVE_INTENT.search(text))
    if negative and not positive:
        return ("negativo",)
    if positive and not negative:
        return ("positivo",)
    return None


def _condition(field_name: str, values: tuple) -> dict:
    return {field_name: values[0]} if len(values) == 1 else {field_name: {"$in": list(values)}}


def build_where(filters: SearchFilters) -> dict | None:
    """Converte os filtros para a sintaxe ``where`` do Chroma."""
    conditions: list[dict] = []
    if not filters.include_low_information:
        conditions.append({"low_information": False})
    for field_name, values in [
        ("sentiment", filters.sentiments),
        ("review_score", filters.review_scores),
        ("delivery_status", filters.delivery_statuses),
        ("product_category", filters.categories),
    ]:
        if values:
            conditions.append(_condition(field_name, tuple(values)))
    if filters.date_from:
        conditions.append({"review_date_int": {"$gte": int(filters.date_from.strftime("%Y%m%d"))}})
    if filters.date_to:
        conditions.append({"review_date_int": {"$lte": int(filters.date_to.strftime("%Y%m%d"))}})

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@dataclass(frozen=True)
class Evidence:
    """Uma avaliação que sustenta a resposta, com as repetições do mesmo texto agrupadas.

    ``mentions`` conta as repetições entre os candidatos buscados, não na base inteira: é um piso.
    A contagem exata do volume de cada tema fica para as análises quantitativas (DuckDB, fase 2).
    """

    review_id: str
    text: str
    similarity: float
    metadata: dict
    duplicate_ids: tuple[str, ...] = ()

    @property
    def mentions(self) -> int:
        return 1 + len(self.duplicate_ids)


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    filters: SearchFilters
    min_similarity: float
    evidences: list[Evidence] = field(default_factory=list)
    best_similarity: float | None = None  # do melhor candidato, mesmo abaixo do corte

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidences)


def deduplicate(candidates: list[Evidence]) -> list[Evidence]:
    """Junta os textos iguais depois da normalização, mantendo o de maior similaridade."""
    groups: dict[str, Evidence] = {}
    for candidate in candidates:  # chegam em ordem decrescente de similaridade
        key = normalize_for_comparison(candidate.text)
        kept = groups.get(key)
        if kept is None:
            groups[key] = candidate
        else:
            groups[key] = replace(kept, duplicate_ids=kept.duplicate_ids + (candidate.review_id,))
    return list(groups.values())


class Retriever:
    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: Embedder,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ):
        self.collection = collection
        self.embedder = embedder
        self.min_similarity = min_similarity

    def search(
        self,
        question: str,
        k: int = DEFAULT_K,
        filters: SearchFilters | None = None,
        infer_filters: bool = True,
    ) -> RetrievalResult:
        filters = filters or SearchFilters()
        if infer_filters and filters.sentiments is None and filters.review_scores is None:
            filters = replace(filters, sentiments=infer_sentiment(question))

        query_vector = self.embedder.embed_query(question)
        where = build_where(filters)
        n_results = k * CANDIDATE_FACTOR
        while True:
            candidates = self._query(query_vector, n_results, where)
            evidences = [e for e in deduplicate(candidates) if e.similarity >= self.min_similarity]
            exhausted = len(candidates) < n_results
            below_cut = bool(candidates) and candidates[-1].similarity < self.min_similarity
            if len(evidences) >= k or exhausted or below_cut or n_results >= MAX_CANDIDATES:
                break
            n_results = min(n_results * 2, MAX_CANDIDATES)

        best = candidates[0].similarity if candidates else None
        return RetrievalResult(question, filters, self.min_similarity, evidences[:k], best)

    def _query(self, query_vector: list[float], n_results: int, where: dict | None) -> list[Evidence]:
        response = self.collection.query(query_embeddings=[query_vector], n_results=n_results, where=where)
        return [
            Evidence(review_id=review_id, text=text, similarity=1 - distance, metadata=metadata)
            for review_id, text, distance, metadata in zip(
                response["ids"][0], response["documents"][0], response["distances"][0], response["metadatas"][0]
            )
        ]
