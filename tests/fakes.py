import hashlib

import numpy as np
from langchain_core.embeddings import Embeddings

from src.embeddings import as_passage, as_query


class FakeEmbedder(Embeddings):
    """Vetores determinísticos a partir do hash do texto: testa índice e busca sem baixar o modelo.

    Com ``symmetric=True``, a pergunta recebe o mesmo vetor do documento de texto idêntico
    (similaridade 1), o que permite testar a busca com resultados previsíveis.
    """

    def __init__(self, dimension: int = 8, symmetric: bool = False, model_name: str = "fake-model"):
        self.dimension = dimension
        self.symmetric = symmetric
        self.model_name = model_name
        self.calls: list[list[str]] = []

    def vector(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        vector = np.random.default_rng(seed).normal(size=self.dimension)
        return (vector / np.linalg.norm(vector)).tolist()

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [self.vector(as_passage(text)) for text in texts]

    def embed_query(self, text):
        return self.vector(as_passage(text) if self.symmetric else as_query(text))
