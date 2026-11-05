"""Interface do Voice of Customer Intelligence: perguntas em linguagem natural sobre as avaliações da Olist.

Execução:
    streamlit run app.py

Requer o índice (python scripts/build_index.py) e OPENAI_API_KEY no .env.

Além da resposta e das avaliações citadas, a tela mostra cada etapa do pipeline (filtros, avaliações
recuperadas com a similaridade e o corte, contexto enviado ao LLM e decisão final), para que a
origem de cada resposta possa ser conferida.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.llm import build_llm
from src.rag_pipeline import SYSTEM_PROMPT, RagPipeline, RagResponse, asks_for_frequency, describe_filters
from src.retrieval import DEFAULT_K, Evidence, SearchFilters, load_retriever
from src.vector_store import count_documents

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "chroma_db"

# Perguntas do enunciado do Tech Challenge, mais uma fora do tema para demonstrar a recusa.
EXAMPLES = [
    "Quais são os principais problemas relatados pelos clientes?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam que existem problemas recorrentes com o prazo de entrega?",
    "Qual a capital da França?",
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

st.set_page_config(page_title="Voz do Cliente Olist", page_icon="💬", layout="wide")


# ---- Recursos carregados uma única vez ---------------------------------------------------------


@st.cache_resource(show_spinner="Carregando o modelo de embeddings e o índice...")
def load_pipeline() -> RagPipeline:
    return RagPipeline(load_retriever(INDEX_DIR), build_llm())


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
    sentiments = st.sidebar.multiselect("Sentimento", ["negativo", "neutro", "positivo"])
    scores = st.sidebar.multiselect("Nota", [1, 2, 3, 4, 5])
    categories = st.sidebar.multiselect("Categoria do produto", options["categories"], format_func=label)
    delivery = st.sidebar.multiselect("Status da entrega", options["delivery"],
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
        st.caption(f"similaridade {num(evidence.similarity)} · review_id `{evidence.review_id}`")


def show_answer(response: RagResponse) -> None:
    if response.answered:
        st.markdown(response.answer)
        if response.invalid_citations:
            st.warning(f"A resposta citou números sem avaliação correspondente: {list(response.invalid_citations)}")
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
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Comentários iguais depois da normalização aparecem uma vez, com o número de repetições.")

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


def main() -> None:
    st.title("💬 Voz do Cliente Olist")
    st.caption(
        "Pergunte em linguagem natural sobre as avaliações dos clientes. As respostas se baseiam apenas nas "
        "avaliações encontradas, citam cada uma delas e são recusadas quando não há evidência."
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
    filters, k = sidebar_filters(options)

    st.markdown("**Exemplos:**")
    columns = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        if columns[i % 3].button(example, key=f"example-{i}", width="stretch"):
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
    if response.answered and asks_for_frequency(response.question):
        st.warning(
            "Esta pergunta pede frequência ou proporção. A busca traz os comentários mais parecidos com a "
            "pergunta, e não uma amostra representativa: a resposta mostra exemplos, não um ranking. "
            "Exemplo: a busca encontra \"ao menos 98\" cópias de \"não recebi o produto\", mas a base tem "
            "2.127 avaliações que falam em não recebimento, 19,6% das negativas. Esse tipo de número exige "
            "contar sobre a base inteira."
        )
    show_answer(response)
    show_pipeline(response, st.session_state.get("chosen_filters", SearchFilters()))


main()
