"""Cliente do LLM que redige a resposta a partir das evidências.

Decisões:

- **OpenAI, pela Responses API, com saída estruturada.** O modelo devolve um objeto validado pelo
  Pydantic (``RagAnswer``), e não um texto livre. O código sabe, sem interpretar frases, se o modelo
  respondeu ou recusou e quais evidências citou. Isso permite conferir as citações (``src/rag_pipeline.py``).
- **Temperatura 0.** A mesma pergunta com as mesmas evidências deve gerar a mesma resposta, o que
  torna a avaliação reproduzível. Criatividade não é desejável num sistema que resume evidências.
- **Modelo configurável** por ``OPENAI_MODEL`` no ``.env``. O padrão foi escolhido pela avaliação
  descrita no README ("Resposta com o LLM").
- **Interface mínima** (``LLMClient``): o pipeline depende só de ``generate``. Os testes usam um
  cliente falso, sem custo, e um provedor alternativo (Claude, fase 2) entra sem mudar o pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.0


class RagAnswer(BaseModel):
    """Formato que o LLM deve devolver."""

    answered: bool = Field(description="true se as evidências permitem responder; false para recusar")
    answer: str = Field(description="Resposta em português, com citações [n] após cada afirmação. Vazia se recusar")
    cited: list[int] = Field(description="Números das evidências citadas na resposta")
    refusal_reason: str = Field(description="Se recusar, o motivo em uma frase. Vazio se responder")


@dataclass(frozen=True)
class LLMOutput:
    parsed: RagAnswer
    model: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, user: str) -> LLMOutput: ...


class OpenAIClient:
    def __init__(self, model: str | None = None, temperature: float | None = None, api_key: str | None = None):
        from openai import OpenAI  # import tardio: os testes não precisam do SDK configurado

        load_env()
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.temperature = temperature if temperature is not None else float(
            os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
        )
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY não definida. Copie .env.example para .env e preencha a chave.")
        self._client = OpenAI(api_key=key)

    def generate(self, system: str, user: str) -> LLMOutput:
        response = self._client.responses.parse(
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            text_format=RagAnswer,
            temperature=self.temperature,
        )
        return LLMOutput(
            parsed=response.output_parsed,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def load_env() -> None:
    """Carrega o ``.env`` da raiz do projeto, sem sobrescrever variáveis já definidas."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
