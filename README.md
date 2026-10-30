# tech-challenge-fase4
Construir um sistema RAG sobre as avaliações da Olist. Ele recebe perguntas em linguagem natural, como "quais as principais reclamações sobre entrega?", e responde com base nos comentários reais dos clientes.

O enunciado do desafio está em [docs/](docs/).

## Dados

A solução usa o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), com cerca de 100 mil pedidos realizados entre 2016 e 2018. Os CSVs estão versionados em [data/raw/](data/raw/), então basta clonar o repositório, sem precisar baixar nada do Kaggle.

| Arquivo | Conteúdo |
| --- | --- |
| `olist_order_reviews_dataset.csv` | Avaliações: nota, título e comentário |
| `olist_orders_dataset.csv` | Pedidos: status e datas de compra, entrega e entrega estimada |
| `olist_order_items_dataset.csv` | Itens de cada pedido: produto, vendedor, preço e frete |
| `olist_products_dataset.csv` | Produtos e suas categorias |
| `product_category_name_translation.csv` | Tradução das categorias para o inglês |
| `olist_customers_dataset.csv` | Clientes: cidade e estado |
| `olist_order_payments_dataset.csv` | Pagamentos: forma e valor |
| `olist_sellers_dataset.csv` | Vendedores: cidade e estado |

O `olist_geolocation_dataset.csv` (61 MB) não foi versionado, porque a solução não usa coordenadas geográficas. Se precisar dele, baixe-o do Kaggle e coloque-o em `data/raw/`.

## Análise exploratória

O notebook [notebooks/01_eda_avaliacoes.ipynb](notebooks/01_eda_avaliacoes.ipynb) analisa as avaliações antes da construção da base de conhecimento: quantas têm texto, tamanho e qualidade dos comentários, vocabulário (*Bag of Words* e TF-IDF), duplicatas, valores ausentes e quais metadados se associam à insatisfação. Cada achado termina numa decisão de preparo da base, e as escolhas com mais de uma alternativa razoável trazem a comparação que levou à decisão. A última seção resume as técnicas de processamento de texto usadas, onde e por quê.

O notebook está salvo com as saídas e pode ser lido direto no GitHub. Para executá-lo localmente (Python 3.11):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_avaliacoes.ipynb
```

## Preparo da base de documentos

O módulo [src/preprocessing.py](src/preprocessing.py) aplica as decisões da análise exploratória e gera a base que será indexada pelo RAG:

- **Um documento por avaliação** (`review_id`), com título e comentário juntos. Entram as avaliações que têm qualquer um dos dois (42.370 de 98.410). Não há *chunking*: os comentários têm no máximo cerca de 200 caracteres.
- **Limpeza leve** do texto: remove quebras de linha, espaços extras e pontuação repetida em excesso, mantendo caixa, acentos, emojis e negações.
- **Texto normalizado** em coluna separada (`text_norm`: minúsculas, sem acentos, sem pontuação), usado só para identificar comentários repetidos na hora de escolher as evidências.
- **Baixa informação** (`low_information`): textos sem nenhuma palavra de 3 letras ou mais, como ".", "ok" e "10" (cerca de 1% da base). Ficam na base, mas são marcados.
- **Metadados**: nota, sentimento (negativo 1–2, neutro 3, positivo 4–5), status do pedido, status e dias de atraso da entrega, categoria do produto e datas.
  - Status da entrega: `no_prazo`, `atrasado` e, para os pedidos sem data de entrega, `entregue_sem_data`, `enviado_nao_recebido`, `nao_enviado` ou `cancelado_indisponivel`.
  - Categoria: a do primeiro item do pedido, ou `sem_itens` (pedido sem nenhum item) e `desconhecida` (produto sem categoria cadastrada).

Para gerar a base:

```bash
python scripts/prepare_data.py
```

| Saída | Conteúdo |
| --- | --- |
| `data/processed/reviews_docs.parquet` | Base completa. Não é versionada: o script a recria a partir de `data/raw/` |
| [data/sample/reviews_sample.csv](data/sample/reviews_sample.csv) | Amostra de 3.000 documentos, com a mesma proporção de notas da base, para testar o sistema mais rápido. Versionada |

O tamanho da amostra e a semente podem ser alterados com `--sample-size` e `--seed`. Para rodar os testes:

```bash
pytest
```

## Licença dos dados

A base é publicada pela Olist sob [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) e é redistribuída aqui para fins acadêmicos e não comerciais, com atribuição à fonte original.
