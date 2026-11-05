"""Contagens de temas das avaliações negativas, com DuckDB.

Junta a classificação por tema (``data/themes/negative_themes.csv``, gerada por
``scripts/classify_themes.py``) com os metadados da base de documentos e conta por tema, categoria,
período, status da entrega e nota.

Por que DuckDB: as perguntas de frequência são consultas analíticas (agrupar, contar, cruzar), que o
SQL expressa diretamente. O DuckDB lê o CSV e o Parquet no próprio processo, sem servidor, e desdobra
a lista de temas de cada avaliação (``string_split`` + ``unnest``) numa linha por tema. Alternativa
avaliada: ``groupby`` do pandas, que exige explodir a lista à mão e fica mais difícil de ler a cada
cruzamento novo.

Como uma avaliação pode ter mais de um tema, as porcentagens de cada tema são calculadas sobre o
número de avaliações, e somam mais de 100%.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THEMES = ROOT / "data" / "themes" / "negative_themes.csv"
DEFAULT_DOCUMENTS = ROOT / "data" / "processed" / "reviews_docs.parquet"

THEME_LABELS = {
    "nao_recebimento": "Não recebimento",
    "atraso": "Atraso na entrega",
    "entrega_parcial": "Entrega parcial",
    "produto_errado": "Produto errado ou diferente do anunciado",
    "defeito_qualidade": "Defeito ou baixa qualidade",
    "atendimento": "Atendimento sem resposta",
    "troca_devolucao_reembolso": "Troca, devolução ou reembolso",
    "logistica_frete": "Logística e frete",
    "nota_fiscal_cobranca": "Nota fiscal e cobrança",
    "embalagem": "Embalagem",
    "outro": "Outro motivo",
    "sem_reclamacao": "Sem reclamação no texto",
}


@dataclass(frozen=True)
class AnalyticsFilters:
    categories: tuple[str, ...] | None = None
    delivery_statuses: tuple[str, ...] | None = None
    review_scores: tuple[int, ...] | None = None
    date_from: date | None = None
    date_to: date | None = None


class ThemeAnalytics:
    """Consultas de frequência sobre as avaliações negativas classificadas por tema."""

    def __init__(self, themes_path: Path = DEFAULT_THEMES, documents_path: Path = DEFAULT_DOCUMENTS):
        self.connection = duckdb.connect()
        # Uma linha por avaliação negativa classificada, com os metadados usados nos filtros.
        self.connection.execute(f"""
            CREATE VIEW reviews AS
            SELECT d.review_id, d.text, d.review_score, d.product_category, d.delivery_status,
                   CAST(d.review_creation_date AS DATE) AS review_date,
                   string_split(t.themes, ';') AS themes
            FROM read_csv('{Path(themes_path).as_posix()}', header = true, all_varchar = true) AS t
            JOIN read_parquet('{Path(documents_path).as_posix()}') AS d USING (review_id)
        """)
        # Uma linha por par (avaliação, tema).
        self.connection.execute("""
            CREATE VIEW review_themes AS
            SELECT * EXCLUDE (themes), unnest(themes) AS theme FROM reviews
        """)

    @staticmethod
    def _where(filters: AnalyticsFilters | None) -> tuple[str, list]:
        filters = filters or AnalyticsFilters()
        clauses, params = [], []
        for column, values in [("product_category", filters.categories),
                               ("delivery_status", filters.delivery_statuses),
                               ("review_score", filters.review_scores)]:
            if values:
                clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
                params.extend(values)
        if filters.date_from:
            clauses.append("review_date >= ?")
            params.append(filters.date_from)
        if filters.date_to:
            clauses.append("review_date <= ?")
            params.append(filters.date_to)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def total(self, filters: AnalyticsFilters | None = None) -> int:
        where, params = self._where(filters)
        return self.connection.execute(f"SELECT count(*) FROM reviews{where}", params).fetchone()[0]

    def theme_counts(self, filters: AnalyticsFilters | None = None) -> pd.DataFrame:
        """Avaliações por tema, da mais frequente para a menos, com a porcentagem sobre o total filtrado."""
        where, params = self._where(filters)
        frame = self.connection.execute(f"""
            SELECT theme, count(*) AS reviews,
                   count(*) / (SELECT count(*) FROM reviews{where}) AS share
            FROM review_themes{where}
            GROUP BY theme ORDER BY reviews DESC, theme
        """, params + params).df()
        frame.insert(1, "label", frame["theme"].map(THEME_LABELS))
        return frame

    def theme_by(self, column: str, filters: AnalyticsFilters | None = None, top: int | None = None) -> pd.DataFrame:
        """Tabela tema × ``column`` (categoria, status da entrega, nota), com a porcentagem dentro de cada valor."""
        if column not in {"product_category", "delivery_status", "review_score"}:
            raise ValueError(f"Coluna não suportada: {column}")
        where, params = self._where(filters)
        frame = self.connection.execute(f"""
            WITH base AS (SELECT {column} AS value, count(*) AS total FROM reviews{where} GROUP BY 1),
                 counts AS (SELECT {column} AS value, theme, count(*) AS reviews FROM review_themes{where} GROUP BY 1, 2)
            SELECT c.value, c.theme, c.reviews, c.reviews / b.total AS share, b.total
            FROM counts c JOIN base b USING (value)
            ORDER BY b.total DESC, c.reviews DESC
        """, params + params).df()
        if top:
            keep = frame.drop_duplicates("value").head(top)["value"]
            frame = frame[frame["value"].isin(keep)]
        return frame

    def monthly(self, filters: AnalyticsFilters | None = None) -> pd.DataFrame:
        """Avaliações por mês e tema, para acompanhar a evolução."""
        where, params = self._where(filters)
        return self.connection.execute(f"""
            SELECT date_trunc('month', review_date) AS month, theme, count(*) AS reviews
            FROM review_themes{where} GROUP BY 1, 2 ORDER BY 1, 3 DESC
        """, params).df()

    def examples(self, theme: str, filters: AnalyticsFilters | None = None, limit: int = 5) -> pd.DataFrame:
        """Avaliações de um tema, para mostrar como evidência ao lado da contagem."""
        where, params = self._where(filters)
        connector = " AND " if where else " WHERE "
        return self.connection.execute(f"""
            SELECT review_id, text, review_score, product_category, delivery_status, review_date
            FROM review_themes{where}{connector}theme = ? ORDER BY length(text) DESC LIMIT ?
        """, params + [theme, limit]).df()
