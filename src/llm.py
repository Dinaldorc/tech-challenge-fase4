"""Modelo de linguagem que redige a resposta a partir das evidências, como componente do LangChain.

Decisões:

- **OpenAI pelo ``ChatOpenAI`` do ``langchain-openai``, na Responses API, com saída estruturada.**
  ``with_structured_output`` faz o modelo devolver um objeto validado pelo Pydantic (``RagAnswer``), e
  não um texto livre. O código sabe, sem interpretar frases, se o modelo respondeu ou recusou e quais
  evidências citou. Isso permite conferir as citações (``src/rag_pipeline.py``).
- **A mensagem bruta vem junto** (``include_raw=True``): dela saem os tokens usados e o nome exato do
  modelo, mostrados na interface e registrados na avaliação.
- **Temperatura 0.** A mesma pergunta com as mesmas evidências deve gerar a mesma resposta, o que
  torna a avaliação reproduzível. Criatividade não é desejável num sistema que resume evidências.
- **Resposta com tamanho máximo** (``MAX_OUTPUT_TOKENS``). A maior resposta da avaliação usou 285
  tokens. O limite de 1.000 não corta nenhuma resposta normal, mas interrompe cedo um modelo que entre em
  repetição: numa pergunta de teste, o modelo repetiu os números das citações até a resposta ser cortada.
- **Resposta fora do formato vira recusa.** Uma resposta cortada ou malformada não passa na validação
  do Pydantic. Na Responses API, o próprio SDK da OpenAI levanta esse erro antes que o ``include_raw``
  o capture, e a pergunta terminava numa exceção. ``tolerate_format_errors`` converte o erro no mesmo
  formato de uma resposta não estruturada (``parsed`` vazio), que o pipeline recusa com o motivo.
- **Modelo configurável** por ``OPENAI_MODEL`` no ``.env``. O padrão foi escolhido pela avaliação
  descrita no README ("Avaliação e escolha do modelo").
- **O pipeline recebe um ``Runnable``**, e não um cliente específico: trocar a OpenAI por outro
  provedor (o Claude, por exemplo, pelo ``ChatAnthropic``) é trocar o componente montado aqui. Nos
  testes, o mesmo lugar recebe um LLM falso, sem custo.
- **Limite de requisições (erro 429) tratado com novas tentativas, sem modelo reserva.** O SDK da
  OpenAI repete a chamada sozinho (2 vezes por padrão, com espera crescente) em erros de limite, de
  tempo esgotado e de servidor. Não adotamos um *fallback* para outro modelo (``with_fallbacks`` do
  LangChain): a resposta viria de um modelo que não passou pela avaliação, e o modelo reserva da mesma
  conta cairia junto numa falta de crédito ou numa queda da OpenAI. Se as tentativas se esgotam, a
  pergunta termina em erro e pode ser repetida.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 1_000


class RagAnswer(BaseModel):
    """Formato que o LLM deve devolver."""

    answered: bool = Field(description="true se as evidências permitem responder; false para recusar")
    answer: str = Field(description="Resposta em português, com citações [n] após cada afirmação. Vazia se recusar")
    cited: list[int] = Field(description="Números das evidências citadas na resposta")
    refusal_reason: str = Field(description="Se recusar, o motivo em uma frase. Vazio se responder")


def load_env() -> None:
    """Carrega o ``.env`` da raiz do projeto, sem sobrescrever variáveis já definidas."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)


def build_chat_model(model: str | None = None, temperature: float | None = None, api_key: str | None = None):
    """``ChatOpenAI`` configurado pelo ``.env`` (chave, modelo e temperatura)."""
    from langchain_openai import ChatOpenAI  # import tardio: os testes não precisam do SDK configurado

    load_env()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY não definida. Copie .env.example para .env e preencha a chave.")
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
        api_key=key,
        use_responses_api=True,
        max_tokens=MAX_OUTPUT_TOKENS,
    )


def tolerate_format_errors(structured: Runnable, errors: tuple[type[Exception], ...] = (ValidationError,)) -> Runnable:
    """Devolve ``parsed=None`` quando a resposta não pode ser lida no formato estruturado, em vez de falhar."""

    def invoke(messages):
        try:
            return structured.invoke(messages)
        except errors as error:
            raw = AIMessage(content="", response_metadata={"model_name": "desconhecido (resposta fora do formato)"})
            return {"raw": raw, "parsed": None, "parsing_error": error}

    return RunnableLambda(invoke)


def build_llm(model: str | None = None, temperature: float | None = None, api_key: str | None = None) -> Runnable:
    """LLM com saída estruturada: recebe as mensagens e devolve ``{"raw", "parsed", "parsing_error"}``."""
    from openai import LengthFinishReasonError

    chat_model = build_chat_model(model, temperature, api_key)
    structured = chat_model.with_structured_output(RagAnswer, method="json_schema", include_raw=True, strict=True)
    return tolerate_format_errors(structured, (ValidationError, LengthFinishReasonError))
