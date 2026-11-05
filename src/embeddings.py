"""Geração de embeddings das avaliações e das perguntas, como um componente ``Embeddings`` do LangChain.

Modelo: ``intfloat/multilingual-e5-base``, executado localmente com ``sentence-transformers``.

- Multilíngue, com bom desempenho em português, idioma de todas as avaliações.
- Roda na máquina de quem executa o projeto: sem custo por chamada, sem chave de API para indexar
  e sem enviar o texto dos clientes para fora durante a indexação.
- Vetores de 768 dimensões e limite de 512 tokens. Os comentários da Olist têm no máximo 188 tokens
  (mediana de 17), então nenhum é truncado e não há necessidade de chunking.

A família E5 foi treinada com prefixos que dizem ao modelo o papel de cada texto: ``passage:`` para
o conteúdo indexado e ``query:`` para a pergunta. Sem eles a qualidade da busca cai, por isso os
prefixos ficam centralizados aqui e ninguém precisa lembrar deles em outros pontos do código.

Os vetores saem normalizados (norma 1). Assim o produto interno é igual ao cosseno, que é a métrica
configurada no índice (``src/vector_store.py``).

Por que uma subclasse de ``Embeddings`` e não o ``HuggingFaceEmbeddings`` do ``langchain-huggingface``:
os prefixos do E5 ficam explícitos aqui, em vez de escondidos em parâmetros de configuração, e o
projeto não ganha mais uma dependência que fixaria versões do ``sentence-transformers``. Como segue a
interface ``Embeddings``, o componente é aceito por qualquer peça do LangChain, como o ``Chroma``.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
MAX_SEQ_LENGTH = 512
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def as_passage(text: str) -> str:
    return PASSAGE_PREFIX + text


def as_query(text: str) -> str:
    return QUERY_PREFIX + text


class E5Embeddings(Embeddings):
    """Embeddings E5 com os prefixos ``passage:``/``query:`` e vetores normalizados."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 64):
        # Import tardio: os testes e quem só lê o índice não precisam carregar o PyTorch.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = MAX_SEQ_LENGTH

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    @property
    def device(self) -> str:
        return str(self._model.device)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([as_passage(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([as_query(text)])[0]
