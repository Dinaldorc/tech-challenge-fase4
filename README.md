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

O notebook [notebooks/01_eda_avaliacoes.ipynb](notebooks/01_eda_avaliacoes.ipynb) analisa as avaliações antes da construção da base de conhecimento: quantas têm texto, tamanho e qualidade dos comentários, duplicatas, valores ausentes e quais metadados se associam à insatisfação. Cada achado termina numa decisão de preparo da base.

O notebook está salvo com as saídas e pode ser lido direto no GitHub. Para executá-lo localmente (Python 3.11):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_avaliacoes.ipynb
```

## Licença dos dados

A base é publicada pela Olist sob [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) e é redistribuída aqui para fins acadêmicos e não comerciais, com atribuição à fonte original.
