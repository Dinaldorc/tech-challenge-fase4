"""Interface do Voice of Customer Intelligence: perguntas em linguagem natural sobre as avaliações da Olist.

Execução:
    streamlit run app.py

Requer o índice (python scripts/build_index.py) e OPENAI_API_KEY no .env.

Duas visões:

- **Perguntar:** a resposta, as avaliações citadas e cada etapa do pipeline (filtros, avaliações
  recuperadas com a similaridade e o corte, contexto enviado ao LLM e decisão final), para que a
  origem de cada resposta possa ser conferida.
- **Panorama:** a frequência de cada tema nas avaliações negativas (classificação de
  ``data/themes/negative_themes.csv``), por categoria, status da entrega e período.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.analytics import DEFAULT_DOCUMENTS, DEFAULT_THEMES, THEME_LABELS, AnalyticsFilters, ThemeAnalytics
from src.llm import build_llm
from src.rag_pipeline import (
    SYSTEM_PROMPT,
    RagPipeline,
    RagResponse,
    asks_for_frequency,
    describe_filters,
    format_int,
)
from src.retrieval import DEFAULT_K, Evidence, SearchFilters, load_retriever
from src.vector_store import count_documents

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "chroma_db"

# Perguntas do enunciado do Tech Challenge, mais uma fora do tema para demonstrar a recusa.
# Cada exemplo tem um rótulo curto para o botão e a pergunta completa, enviada ao clicar.
EXAMPLES = [
    ("Principais problemas", "Quais são os principais problemas relatados pelos clientes?"),
    ("Problemas de entrega", "O que os clientes relatam sobre problemas relacionados à entrega?"),
    ("O que é elogiado", "Quais aspectos da experiência de compra são mais elogiados pelos clientes?"),
    ("Padrões dos insatisfeitos", "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?"),
    ("Avaliações que sustentam", "Quais avaliações sustentam a conclusão de que existem problemas recorrentes "
                                 "relacionados ao prazo de entrega?"),
    ("Reclamações mais frequentes", "Quais reclamações relacionadas à entrega aparecem com maior frequência?"),
    ("Fora do tema (recusa)", "Qual a capital da França?"),
]

DELIVERY_LABELS = {
    "no_prazo": "No prazo",
    "atrasado": "Atrasado",
    "entregue_sem_data": "Entregue, sem data registrada",
    "enviado_nao_recebido": "Enviado, não recebido",
    "nao_enviado": "Não enviado",
    "cancelado_indisponivel": "Cancelado ou indisponível",
    "desconhecido": "Desconhecido",
}
REFUSAL_STAGE = {
    "busca": "na busca: nenhuma avaliação passou do corte de similaridade, e o LLM não foi chamado",
    "llm": "pelo LLM: as avaliações encontradas não respondem à pergunta",
    "citacao": "na conferência das citações: a resposta não citou nenhuma avaliação existente",
}

# Cores dos gráficos, para o tema escuro (.streamlit/config.toml): um único tom, o magenta neon, nas
# barras e na linha (uma série só), e uma escala sequencial do mesmo tom no mapa de calor, do mais
# fechado (pouco) ao mais claro (muito). A escala foi validada para o fundo escuro: o tom mais fraco
# ainda se destaca do fundo (2:1) e os degraus são distinguíveis.
CHART_COLOR = "#ff2e88"
SEQUENTIAL_RANGE = ["#7a2560", "#a8307a", "#d43c8f", "#f866a9", "#ffaed0"]
SURFACE = "#0e0e12"
TEXT = "#f0efec"
SECONDARY_TEXT = "#b9b7c4"
# A partir desta porcentagem, o texto da célula fica escuro (contraste melhor sobre o rosa claro).
DARK_TEXT_FROM = 32
MIN_MONTH_REVIEWS = 50  # meses com menos reclamações que isso oscilam demais em porcentagem
MIN_STATUS_REVIEWS = 30

st.set_page_config(page_title="Voz do Cliente Olist", page_icon=":material/graphic_eq:", layout="wide")


# ---- Recursos carregados uma única vez ---------------------------------------------------------


@st.cache_resource(show_spinner="Carregando a classificação por tema...")
def load_analytics() -> ThemeAnalytics | None:
    """Contagens por tema; ``None`` se a base de documentos ainda não foi gerada (prepare_data.py)."""
    if not (DEFAULT_THEMES.exists() and DEFAULT_DOCUMENTS.exists()):
        return None
    return ThemeAnalytics()


@st.cache_resource(show_spinner="Carregando o modelo de embeddings e o índice...")
def load_pipeline() -> RagPipeline:
    return RagPipeline(load_retriever(INDEX_DIR), build_llm(), load_analytics())


@st.cache_data(show_spinner=False)
def load_filter_options() -> dict:
    vector_store = load_pipeline().retriever.vector_store
    # Em lotes: pedir as 42 mil de uma vez estoura o limite de variáveis do SQLite usado pelo Chroma.
    metadatas, batch = [], 5000
    for offset in range(0, count_documents(vector_store), batch):
        metadatas += vector_store.get(include=["metadatas"], limit=batch, offset=offset)["metadatas"]
    categories = Counter(meta["product_category"] for meta in metadatas)
    dates = [meta["review_date"] for meta in metadatas]
    return {
        "total": len(metadatas),
        "categories": [name for name, _ in categories.most_common()],
        "delivery": sorted({meta["delivery_status"] for meta in metadatas}, key=list(DELIVERY_LABELS).index),
        "date_min": date.fromisoformat(min(dates)),
        "date_max": date.fromisoformat(max(dates)),
    }


def label(category: str) -> str:
    return category.replace("_", " ")


def num(value: float, decimals: int = 3) -> str:
    """Número com vírgula decimal, como no restante da documentação."""
    return f"{value:.{decimals}f}".replace(".", ",")


# ---- Barra lateral: filtros --------------------------------------------------------------------


def sidebar_filters(options: dict) -> tuple[SearchFilters, int]:
    st.sidebar.header("Filtros")
    st.sidebar.caption(
        "Restringem a busca às avaliações com esses metadados. Sem sentimento escolhido, ele é inferido da "
        "pergunta quando ela pede reclamações ou elogios."
    )
    sentiments = st.sidebar.multiselect("Sentimento", ["negativo", "neutro", "positivo"], placeholder="Todos")
    scores = st.sidebar.multiselect("Nota", [1, 2, 3, 4, 5], placeholder="Todas")
    categories = st.sidebar.multiselect("Categoria do produto", options["categories"], format_func=label, placeholder="Todas")
    delivery = st.sidebar.multiselect("Status da entrega", options["delivery"], placeholder="Todos",
                                      format_func=lambda s: DELIVERY_LABELS.get(s, s))
    use_period = st.sidebar.checkbox("Filtrar por período da avaliação")
    date_from = date_to = None
    if use_period:
        period = st.sidebar.date_input("Período", (options["date_min"], options["date_max"]),
                                       min_value=options["date_min"], max_value=options["date_max"],
                                       format="DD/MM/YYYY")
        if isinstance(period, tuple) and len(period) == 2:
            date_from, date_to = period
    k = st.sidebar.slider("Máximo de evidências enviadas ao LLM", 3, 15, DEFAULT_K)
    st.sidebar.divider()
    st.sidebar.caption(f"Base: {options['total']:,} avaliações com texto".replace(",", "."))

    filters = SearchFilters(
        sentiments=tuple(sentiments) or None,
        review_scores=tuple(scores) or None,
        delivery_statuses=tuple(delivery) or None,
        categories=tuple(categories) or None,
        date_from=date_from,
        date_to=date_to,
    )
    return filters, k


# ---- Resposta e evidências ---------------------------------------------------------------------


def evidence_card(number: int, evidence: Evidence) -> None:
    meta = evidence.metadata
    stars = "★" * int(meta["review_score"]) + "☆" * (5 - int(meta["review_score"]))
    delivery = DELIVERY_LABELS.get(meta.get("delivery_status"), meta.get("delivery_status"))
    delay = meta.get("delay_days")
    if delay and delay > 0:
        delivery += f" ({delay} dias)"
    repeated = f" · ao menos {evidence.mentions} avaliações com este mesmo texto" if evidence.mentions > 1 else ""
    with st.container(border=True):
        st.markdown(f"**[{number}]** {stars} · {label(meta['product_category'])} · entrega: {delivery} · "
                    f"{meta['review_date']}{repeated}")
        st.markdown(f"> {evidence.text}")
        if meta.get("theme_example"):
            origin = f"exemplo do tema \"{meta['theme_example']}\", escolhido pela classificação, e não pela similaridade"
        else:
            origin = f"similaridade {num(evidence.similarity)}"
        st.caption(f"{origin} · review_id `{evidence.review_id}`")


def show_frequency_note(response: RagResponse) -> None:
    if not (response.answered and asks_for_frequency(response.question)):
        return
    if response.theme_stats:
        st.info(
            f"Os números vêm da classificação por tema das {format_int(response.theme_stats.total)} avaliações "
            "negativas que atendem aos filtros, e não das avaliações citadas, que são exemplos. O detalhamento "
            "está na visão Panorama.",
            title="Pergunta de frequência: contagem real",
        )
    else:
        st.warning(
            "A contagem por tema existe só para as reclamações. Para esta pergunta, a busca traz os comentários "
            "mais parecidos, e não uma amostra representativa: a resposta mostra exemplos, e não um ranking.",
            title="Pergunta de frequência sem contagem",
        )


def show_answer(response: RagResponse) -> None:
    if response.answered:
        st.markdown(response.answer)
        if response.invalid_citations:
            st.warning(f"A resposta citou números sem avaliação correspondente: {list(response.invalid_citations)}")
        if response.invalid_numbers:
            st.warning(f"A resposta traz porcentagens que não estão na contagem: {', '.join(response.invalid_numbers)}")
        st.subheader(f"Evidências usadas na resposta ({len(response.citations)})")
        st.caption(
            "Este número depende do limite de evidências da barra lateral, e não de quantos clientes "
            "falaram do assunto. A contagem de repetições considera só as cópias exatas encontradas pela "
            "busca: é um mínimo, não o total da base."
        )
        numbers = {id(e): i for i, e in enumerate(response.retrieval.evidences, 1)}
        for evidence in response.citations:
            evidence_card(numbers[id(evidence)], evidence)
    else:
        st.info(f"**{response.answer}**\n\nRecusada {REFUSAL_STAGE.get(response.refused_at, '')}.")
        if response.refused_at == "llm":
            st.caption(f"Justificativa do modelo: {response.refusal_reason}")
        elif response.refusal_reason:
            st.caption(response.refusal_reason)


# ---- Como a resposta foi construída ------------------------------------------------------------


def show_pipeline(response: RagResponse, chosen: SearchFilters) -> None:
    retrieval = response.retrieval
    with st.expander("Como a resposta foi construída", expanded=False):
        steps = st.tabs(["1. Pergunta e filtros", "2. Recuperação", "3. Contexto enviado ao LLM", "4. Decisão"])

        with steps[0]:
            st.markdown(f"**Pergunta:** {response.question}")
            st.markdown(f"**Filtros aplicados:** {describe_filters(retrieval.filters)}")
            inferred = retrieval.filters.sentiments and not chosen.sentiments and not chosen.review_scores
            if inferred:
                st.caption(f"O sentimento '{', '.join(retrieval.filters.sentiments)}' foi inferido da pergunta.")
            st.markdown(
                "**Regras fixas da busca** (valem para toda pergunta): comentários sem conteúdo dos clientes "
                "(\"ok\", \".\", \"*****\") não são pesquisados. A pergunta é convertida em embedding pelo modelo "
                "`multilingual-e5-base`, com o prefixo `query:`."
            )

        with steps[1]:
            st.markdown(
                f"Corte de similaridade: **{num(retrieval.min_similarity, 2)}**. Melhor similaridade encontrada: "
                f"**{num(retrieval.best_similarity)}**." if retrieval.best_similarity is not None
                else "Nenhuma avaliação atende aos filtros."
            )
            selected = {id(e) for e in retrieval.evidences}
            rows = []
            for evidence in retrieval.candidates[: max(len(retrieval.evidences) + 7, 12)]:
                if id(evidence) in selected:
                    status = "enviada ao LLM"
                elif evidence.similarity >= retrieval.min_similarity:
                    status = "acima do corte, fora do limite"
                else:
                    status = "abaixo do corte"
                rows.append({"situação": status, "similaridade": round(evidence.similarity, 3),
                             "nota": evidence.metadata["review_score"], "repetições": evidence.mentions,
                             "categoria": label(evidence.metadata["product_category"]), "texto": evidence.text})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", alt="Candidatos da busca")
                st.caption("Comentários iguais depois da normalização aparecem uma vez, com o número de repetições.")
            if retrieval.evidences and retrieval.evidences[0].metadata.get("theme_example"):
                st.info(
                    "Nenhuma avaliação passou do corte de similaridade. Como a pergunta pede a frequência de "
                    "reclamações, foram usados como evidência exemplos dos temas mais frequentes, escolhidos "
                    "pela classificação por tema."
                )
            if response.theme_stats:
                st.markdown("**Contagem real enviada ao LLM**")
                st.code(response.theme_stats.as_text(), language=None, wrap_lines=True)

        with steps[2]:
            if response.user_prompt:
                st.markdown("**Instruções do sistema**")
                st.code(SYSTEM_PROMPT, language=None, wrap_lines=True)
                st.markdown("**Mensagem com a pergunta e as avaliações**")
                st.code(response.user_prompt, language=None, wrap_lines=True)
            else:
                st.markdown("O LLM não foi chamado: a busca não encontrou evidência acima do corte.")

        with steps[3]:
            if response.answered:
                st.markdown(f"Respondida, com {len(response.citations)} avaliações citadas, todas conferidas.")
            else:
                st.markdown(f"Recusada {REFUSAL_STAGE.get(response.refused_at, '')}.")
                if response.refusal_reason:
                    st.markdown(f"Motivo: {response.refusal_reason}")
            if response.model:
                st.caption(f"Modelo: {response.model} · tokens: {response.input_tokens} de entrada + "
                           f"{response.output_tokens} de saída")


# ---- Página ------------------------------------------------------------------------------------


# ---- Panorama ----------------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=64)
def panorama_data(filters: AnalyticsFilters) -> dict:
    analytics = load_analytics()
    return {
        "total": analytics.total(filters),
        "counts": analytics.theme_counts(filters),
        "by_status": analytics.theme_by("delivery_status", filters),
        "monthly": analytics.monthly(filters),
        "monthly_totals": analytics.monthly_totals(filters),
    }


def ranking_chart(counts: pd.DataFrame) -> alt.Chart:
    data = counts.assign(percent=counts["share"] * 100,
                         text=counts["share"].map(lambda v: f"{v * 100:.1f}%".replace(".", ",")))
    order = data["label"].tolist()
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("percent:Q", title="% das avaliações negativas", axis=alt.Axis(grid=False, tickCount=7)),
        tooltip=[alt.Tooltip("label:N", title="Tema"), alt.Tooltip("reviews:Q", title="Avaliações", format=",d"),
                 alt.Tooltip("text:N", title="% das negativas")],
    )
    bars = base.mark_bar(color=CHART_COLOR, cornerRadiusEnd=4, height=18)
    labels = base.mark_text(align="left", dx=4, color=SECONDARY_TEXT).encode(text="text:N")
    return (bars + labels).properties(height=34 * len(data))


def status_heatmap(by_status: pd.DataFrame, theme_order: list[str]) -> alt.Chart:
    totals = by_status.drop_duplicates("value").set_index("value")["total"]
    statuses = totals[totals >= MIN_STATUS_REVIEWS].sort_values(ascending=False).index.tolist()
    themes = [theme for theme, name in THEME_LABELS.items() if name in theme_order]
    # Todas as combinações tema × status, com 0 onde não há avaliação (em vez de uma célula vazia).
    grid = pd.MultiIndex.from_product([statuses, themes], names=["value", "theme"]).to_frame(index=False)
    data = grid.merge(by_status[["value", "theme", "reviews"]], on=["value", "theme"], how="left").fillna({"reviews": 0})
    data["total"] = data["value"].map(totals)
    data["percent"] = data["reviews"] / data["total"] * 100
    # O "|" vira quebra de linha no rótulo do eixo, para os nomes longos não se sobreporem.
    data["status"] = data["value"].map(lambda v: DELIVERY_LABELS.get(v, v).replace(", ", ",|").replace(" ou ", " ou|"))
    data["label"] = data["theme"].map(THEME_LABELS)
    data["text"] = data["percent"].map(lambda v: f"{v:.0f}%")
    status_order = [DELIVERY_LABELS.get(v, v).replace(", ", ",|").replace(" ou ", " ou|") for v in statuses]
    base = alt.Chart(data).encode(
        x=alt.X("status:N", sort=status_order, title="Status da entrega no pedido",
                axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, '|')", labelLimit=160, labelOverlap=False)),
        y=alt.Y("label:N", sort=theme_order, title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=[alt.Tooltip("label:N", title="Tema"), alt.Tooltip("status:N", title="Entrega"),
                 alt.Tooltip("reviews:Q", title="Avaliações"), alt.Tooltip("total:Q", title="Negativas com esse status"),
                 alt.Tooltip("text:N", title="% das negativas com esse status")],
    )
    cells = base.mark_rect(stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color("percent:Q", title="%", scale=alt.Scale(range=SEQUENTIAL_RANGE, domain=[0, 70], clamp=True))
    )
    text = base.mark_text(fontSize=11).encode(
        text="text:N", color=alt.condition(f"datum.percent >= {DARK_TEXT_FROM}", alt.value(SURFACE), alt.value(TEXT))
    )
    return (cells + text).properties(height=30 * len(theme_order))


def trend_chart(monthly: pd.DataFrame, totals: pd.DataFrame, theme: str) -> tuple[alt.Chart, int]:
    data = totals.merge(monthly[monthly["theme"] == theme], on="month", how="left").fillna({"reviews": 0})
    shown = data[data["total"] >= MIN_MONTH_REVIEWS].copy()
    shown["percent"] = shown["reviews"] / shown["total"] * 100
    shown["text"] = shown["percent"].map(lambda v: f"{v:.1f}%".replace(".", ","))
    chart = alt.Chart(shown).mark_line(color=CHART_COLOR, strokeWidth=2, point=alt.OverlayMarkDef(size=60, color=CHART_COLOR, stroke=SURFACE, strokeWidth=2)).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%m/%Y", grid=False, tickCount=7, labelOverlap=True)),
        y=alt.Y("percent:Q", title=f"% das negativas do mês"),
        tooltip=[alt.Tooltip("month:T", title="Mês", format="%m/%Y"), alt.Tooltip("text:N", title="% das negativas"),
                 alt.Tooltip("reviews:Q", title="Avaliações do tema"), alt.Tooltip("total:Q", title="Negativas no mês")],
    ).properties(height=280)
    return chart, len(data) - len(shown)


def show_panorama(options: dict) -> None:
    if load_analytics() is None:
        st.info("A base de documentos não foi encontrada. Rode `python scripts/prepare_data.py` para ver o Panorama.")
        return
    st.caption(
        "Frequência de cada tema nas avaliações negativas (notas 1 e 2) com texto. Cada avaliação foi classificada "
        "pelo LLM numa lista fechada de temas e pode ter mais de um; por isso as porcentagens somam mais de 100%. "
        "A classificação foi validada contra uma referência independente e contra o status do pedido (README, "
        "\"Frequência por tema\")."
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        categories = st.multiselect("Categoria do produto", options["categories"], format_func=label, key="pan_cat",
                                    placeholder="Todas")
        statuses = st.multiselect("Status da entrega", options["delivery"], key="pan_status", placeholder="Todos",
                                  format_func=lambda v: DELIVERY_LABELS.get(v, v))
        scores = st.pills("Nota", [1, 2], selection_mode="multi", key="pan_scores")
        period = st.date_input("Período da avaliação", (options["date_min"], options["date_max"]), key="pan_period",
                               min_value=options["date_min"], max_value=options["date_max"], format="DD/MM/YYYY")
    date_from, date_to = period if isinstance(period, tuple) and len(period) == 2 else (None, None)
    filters = AnalyticsFilters(tuple(categories) or None, tuple(statuses) or None, tuple(scores) or None, date_from, date_to)
    data = panorama_data(filters)
    if data["total"] == 0:
        st.info("Nenhuma avaliação negativa com esses filtros.")
        return

    counts = data["counts"]
    complaints = counts[counts["theme"] != "sem_reclamacao"]
    top = complaints.iloc[0]
    with st.container(horizontal=True):
        st.metric("Avaliações negativas", format_int(data["total"]), border=True)
        st.metric("Tema mais frequente", top.label, border=True)
        st.metric("Proporção do tema mais frequente", f"{top.share * 100:.1f}%".replace(".", ","), border=True)

    with st.container(border=True):
        st.subheader("Temas das reclamações")
        st.altair_chart(ranking_chart(complaints), width="stretch",
                        alt="Ranking dos temas pela porcentagem das avaliações negativas")
        without = counts.loc[counts["theme"] == "sem_reclamacao", "reviews"].sum()
        if without:
            st.caption(f"Fora do ranking: {without} avaliações negativas cujo texto não traz reclamação "
                       "(por exemplo, elogios dados com nota baixa).")

    with st.container(border=True):
        st.subheader("Tema × status da entrega")
        st.altair_chart(status_heatmap(data["by_status"], complaints["label"].tolist()), width="stretch",
                        alt="Porcentagem de cada tema por status da entrega do pedido")
        st.caption("Cada coluna soma mais de 100%, porque uma avaliação pode ter mais de um tema. "
                   f"Status com menos de {MIN_STATUS_REVIEWS} avaliações ficam de fora.")
    with st.container(border=True):
        st.subheader("Evolução de um tema")
        theme = st.selectbox("Tema", complaints["theme"].tolist(), format_func=THEME_LABELS.get, key="pan_theme")
        chart, hidden = trend_chart(data["monthly"], data["monthly_totals"], theme)
        st.altair_chart(chart, width="stretch", alt=f"Evolução mensal do tema {THEME_LABELS[theme]}")
        if hidden:
            st.caption(f"{hidden} meses com menos de {MIN_MONTH_REVIEWS} avaliações negativas ficam de fora, "
                       "porque a porcentagem oscila demais com poucas avaliações.")

    with st.container(border=True):
        st.subheader(f"Exemplos: {THEME_LABELS[theme]}")
        examples = load_analytics().examples(theme, filters, limit=4)
        for row in examples.itertuples():
            stars = "★" * int(row.review_score) + "☆" * (5 - int(row.review_score))
            st.markdown(f"{stars} · {label(row.product_category)} · entrega: "
                        f"{DELIVERY_LABELS.get(row.delivery_status, row.delivery_status)} · {row.review_date:%d/%m/%Y}")
            st.markdown(f"> {row.text}")


# ---- Página ------------------------------------------------------------------------------------


def show_ask(pipeline: RagPipeline, options: dict) -> None:
    filters, k = sidebar_filters(options)

    st.markdown("**Exemplos:**")
    columns = st.columns(4)
    for i, (short, example) in enumerate(EXAMPLES):
        if columns[i % 4].button(short, key=f"example-{i}", width="stretch", help=example):
            st.session_state["question"] = example
            st.session_state["ask_now"] = True

    with st.form("ask"):
        question = st.text_input("Sua pergunta", key="question",
                                 placeholder="Ex.: o que os clientes reclamam sobre móveis?")
        submitted = st.form_submit_button("Perguntar", type="primary")

    if (submitted or st.session_state.pop("ask_now", False)) and question.strip():
        with st.spinner("Buscando avaliações e gerando a resposta..."):
            st.session_state["response"] = pipeline.ask(question.strip(), k=k, filters=filters)
            st.session_state["chosen_filters"] = filters

    response: RagResponse | None = st.session_state.get("response")
    if response is None:
        return
    show_frequency_note(response)
    show_answer(response)
    show_pipeline(response, st.session_state.get("chosen_filters", SearchFilters()))


def main() -> None:
    st.title(":material/graphic_eq: Voz do Cliente Olist")
    st.caption(
        "Pergunte em linguagem natural sobre as avaliações dos clientes, ou veja a frequência de cada tema. "
        "As respostas se baseiam apenas nas avaliações encontradas, citam cada uma delas e são recusadas "
        "quando não há evidência."
    )

    if not INDEX_DIR.exists():
        st.error("Índice não encontrado. Rode `python scripts/build_index.py` antes de abrir a interface.")
        st.stop()
    try:
        pipeline = load_pipeline()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    options = load_filter_options()
    views = [":material/chat: Perguntar", ":material/bar_chart: Panorama"]
    # O endereço pode abrir direto no Panorama (?visao=panorama), útil para apresentações.
    start = views[1] if st.query_params.get("visao") == "panorama" else views[0]
    view = st.segmented_control("Visão", views, default=start, required=True, key="view",
                                label_visibility="collapsed")
    if view.endswith("Panorama"):
        show_panorama(options)
    else:
        show_ask(pipeline, options)

main()
