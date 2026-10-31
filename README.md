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

O tamanho da amostra e a semente podem ser alterados com `--sample-size` e `--seed`.

## Embeddings e índice vetorial

Nesta etapa, cada avaliação vira um vetor numérico (*embedding*) que representa o seu significado, e os vetores são gravados num índice que permite encontrar os comentários mais parecidos com uma pergunta. A busca entende significado, não só palavras: "produto chegou quebrado" encontra "Produto veio com defeito", sem nenhuma palavra em comum além de "produto".

- [src/embeddings.py](src/embeddings.py): gera os embeddings com o modelo `intfloat/multilingual-e5-base`.
- [src/vector_store.py](src/vector_store.py): cria e preenche o índice no ChromaDB, com os metadados de cada avaliação.
- [scripts/build_index.py](scripts/build_index.py): monta o índice a partir da base preparada.
- [scripts/compare_search.py](scripts/compare_search.py): reproduz a comparação entre E5 e BM25 descrita no Achado 1.

```bash
python scripts/build_index.py                     # amostra de 3.000 documentos
python scripts/build_index.py --input data/processed/reviews_docs.parquet --reset   # base completa
```

Na primeira execução, o modelo (1,1 GB) é baixado do Hugging Face e fica em cache. O índice é gravado em `chroma_db/`, que não é versionado porque o script o recria.

| | Amostra | Base completa |
| --- | --- | --- |
| Documentos | 3.000 | 42.370 |
| Tempo de indexação (CPU, sem GPU) | 37 s | 10,5 min |
| Tamanho do índice | 16 MB | 205 MB |

A amostra é o padrão do script: ela permite testar o sistema em menos de um minuto. A base completa é a que deve ser usada nas análises.

### Decisões e alternativas avaliadas

**Representação do texto: embedding denso de sentença.**

A primeira escolha é a forma de transformar cada comentário em números. Comparamos as quatro famílias de representação mais usadas em PLN:

| Família | Exemplos | Como representa o texto | Decisão |
| --- | --- | --- | --- |
| Esparsa (lexical) | Bag of Words, TF-IDF, BM25 | Um peso para cada palavra do vocabulário. Dois textos só se aproximam se usarem as mesmas palavras | Usada na análise exploratória (BoW e TF-IDF) e como linha de base da busca (BM25). Sozinha não basta, porque não reconhece sinônimos |
| Vetores de palavra estáticos | Word2Vec, GloVe, FastText | Um vetor fixo por palavra; o texto vira a média dos vetores das palavras | **Descartada**, ver abaixo |
| Modelo de linguagem contextual sem treino para busca | BERTimbau | Um vetor por token, que depende das palavras ao redor | **Descartada**, ver abaixo |
| Embedding de sentença treinado para busca | E5, `text-embedding-3` da OpenAI | Um vetor por texto, treinado para aproximar uma pergunta dos trechos que respondem a ela | **Escolhida** |

Por que descartamos o Word2Vec e o GloVe:

- **Ignoram o contexto.** Cada palavra tem um único vetor, qualquer que seja a frase. "Manga" da camisa e "manga" fruta são o mesmo ponto, e "não chegou" é só a média de "não" com "chegou".
- **A média apaga a ordem e a estrutura.** "O produto chegou, a nota fiscal não" e "a nota fiscal chegou, o produto não" viram exatamente o mesmo vetor.
- **Não foram treinados para busca.** Eles aprendem que palavras usadas em contextos parecidos são parecidas, e "ótimo" e "péssimo" aparecem nos mesmos contextos ("o produto é ___"). Por isso, ficam próximos, o que agrava justamente o problema de polaridade descrito no Achado 1.
- **São tecnicamente superados.** Foram o estado da arte entre 2013 e 2017, antes dos Transformers. Vencer essa linha de base hoje não demonstra nada sobre a qualidade da solução.

Por que descartamos o BERTimbau:

- **É um bom modelo de português, mas não foi treinado para busca.** O treino dele ensina a prever palavras escondidas numa frase, não a aproximar uma pergunta da resposta. Os vetores que ele gera não têm essa propriedade.
- **Seria preciso extrair um vetor por texto.** O BERT gera um vetor por token, e o caminho usual (média dos tokens ou o token `[CLS]`) produz vetores pouco discriminativos, em que textos sem relação ficam próximos. No artigo que propôs o Sentence-BERT (Reimers e Gurevych, 2019), os vetores do BERT sem ajuste fino ficaram abaixo até da média de vetores GloVe em tarefas de similaridade semântica.
- **O ajuste fino exige pares rotulados de pergunta e comentário relevante,** que não temos. Os modelos E5 passaram justamente por esse treino, com milhões de pares e em vários idiomas, inclusive o português.

**Modelo de embeddings: `multilingual-e5-base`, executado localmente.**

| Alternativa | Por que não foi a escolhida |
| --- | --- |
| `text-embedding-3-small` (OpenAI, via API) | Boa qualidade, mas cobra por chamada, exige chave de API já na indexação e envia todos os comentários dos clientes para um serviço externo. O modelo local indexa sem custo, sem chave e sem tirar os dados da máquina. A OpenAI continua no projeto, mas só para gerar a resposta final. A comparação de qualidade entre os dois está prevista na avaliação da fase 2 |
| `multilingual-e5-small` | Cerca de 2 a 3 vezes mais rápido, mas menos preciso. Com 42 mil textos curtos, o modelo `base` já indexa a base inteira em cerca de 10 minutos, uma única vez, então não compensa perder qualidade |
| `multilingual-e5-large` | Mais preciso, mas tem 2,2 GB e é cerca de 2 a 3 vezes mais lento na CPU. Pesaria para quem for executar o projeto num notebook comum |

O E5 tem mais dois cuidados:

- **Prefixos `passage:` e `query:`.** O modelo foi treinado marcando o papel de cada texto, e a documentação dele recomenda manter essa convenção. Os prefixos ficam centralizados em `src/embeddings.py` e não precisam ser lembrados em outros pontos do código.
- **Sem truncamento.** O limite do modelo é de 512 tokens. Os comentários da base têm no máximo 188 tokens (mediana de 17, 99% abaixo de 60), então nenhum é cortado. Isso confirma a decisão de não fazer *chunking*.

**Banco vetorial: ChromaDB, persistido em disco.**

| Alternativa | Por que não foi a escolhida |
| --- | --- |
| FAISS | É só uma biblioteca de busca: não guarda o texto nem os metadados, e não filtra por eles. Tudo isso teria de ser construído à parte |
| Qdrant, pgvector, Pinecone | Exigem um servidor ou uma conta na nuvem. Para um projeto que o professor precisa clonar e executar, isso é atrito sem ganho na escala atual (42 mil vetores) |

O Chroma roda embutido no próprio Python, grava em uma pasta e filtra por metadados na mesma consulta. Se o volume crescer para milhões de avaliações, a troca para um servidor como o Qdrant fica restrita a `src/vector_store.py`.

**Detalhes do índice:**

- **Distância cosseno** com vetores normalizados, a métrica com que o E5 foi treinado.
- **Embeddings calculados pelo projeto** e entregues prontos ao Chroma. Por padrão, o Chroma usaria o próprio modelo, voltado ao inglês, e ignoraria os prefixos do E5.
- **O nome do modelo fica gravado no índice.** Vetores de modelos diferentes não são comparáveis. Se alguém trocar o modelo e reaproveitar o índice antigo, o código para com erro em vez de devolver resultados sem sentido.
- **Indexação idempotente.** Cada avaliação usa o `review_id` como identificador e é gravada com `upsert`. Rodar o script de novo atualiza os documentos em vez de duplicá-los.
- **Metadados** de cada avaliação: nota, sentimento, status do pedido e da entrega, dias de atraso, categoria e data. A data é gravada também como número (`AAAAMMDD`), porque o Chroma só compara números em filtros de período. Os metadados servem para filtrar a busca e para citar a fonte de cada evidência.
- **Textos de baixa informação** ("ok", ".", "*****") entram no índice com a marca `low_information`. Quem decide se eles são excluídos é a busca.

### O que os testes de busca mostraram

Com o índice da amostra, rodamos perguntas reais e olhamos os resultados. Três achados orientam a próxima etapa.

#### Achado 1: a busca acerta o assunto, mas confunde elogio com reclamação

**O experimento.** Fizemos a pergunta "entrega atrasada, não chegou no prazo" aos 2.974 documentos da amostra (sem os de baixa informação) e comparamos os 10 primeiros resultados de quatro buscas:

- **busca lexical com BM25,** a linha de base padrão em recuperação de informação. É um TF-IDF aprimorado: repetir uma palavra no mesmo texto tem retorno decrescente, e textos longos não ganham vantagem só pelo tamanho;
- **busca semântica com os embeddings do E5;**
- cada uma **com e sem o filtro de sentimento negativo** (notas 1 e 2).

Um resultado foi considerado relevante quando o comentário reclama de atraso ou de não entrega. Para reproduzir:

```bash
python scripts/build_index.py
python scripts/compare_search.py
```

| Busca | Notas dos 10 primeiros | Relevantes | O que apareceu |
| --- | --- | --- | --- |
| BM25, sem filtro | 2, 2, 4, 5, 4, 4, 5, 4, 5, 4 | 2 de 10 | Duas reclamações com "entrega atrasada", sete variações de "chegou no prazo" e um elogio que menciona atraso ("Atrasou mas chegou certinho") |
| E5, sem filtro | 3, 5, 1, 5, 5, 3, 5, 5, 5, 5 | 2 de 10 | Oito variações de "entregue no prazo" e "antes do prazo" |
| BM25 + sentimento negativo | 2, 2, 1, 1, 2, 1, 1, 1, 1, 1 | 9 de 10 | Todos repetem palavras da pergunta. O que não é relevante, "Não recebi ainda. MAS está ainda no prazo de entrega", entrou por ter "prazo" e "entrega" |
| E5 + sentimento negativo | 1, 1, 2, 1, 1, 2, 1, 1, 2, 1 | 10 de 10 | Formulações variadas: "Atraso na entrega", "Demorou para entregar", "muita demora na entrega" |

**Por que as duas técnicas erram sem o filtro.** Elogios e reclamações sobre prazo usam praticamente as mesmas palavras. A diferença é que um nega e o outro não.

- **BM25:** representa o texto pelas palavras que ele contém, com peso maior para as raras. "O produto chegou no prazo" compartilha com a pergunta "chegou", "no" e "prazo". A palavra que inverte o sentido, "não", aparece em 23,5% dos comentários e tem o menor peso (IDF 1,18) entre os termos da pergunta. O termo mais específico, "atrasada", tem o maior peso (IDF 6,29), mas aparece em só 0,2% dos comentários: assim que eles se esgotam, sobram os elogios.
- **Embedding (E5):** representa o significado, mas o significado que o modelo aprendeu a aproximar é o **assunto**. O E5 foi treinado para colocar perto uma pergunta e os trechos que tratam dela, e para uma pergunta sobre prazo de entrega tanto o elogio quanto a reclamação tratam do tema. Além disso, o vetor do texto é a média dos vetores de todos os tokens, e a negação é um único token entre vários. Ela pesa pouco no resultado.

**Polaridade não é problema da técnica de representação.** Nenhuma das duas técnicas resolve a polaridade sozinha, e as duas resolvem com o filtro. Trocar o embedding por outra representação não corrigiria o erro. O que corrige é usar uma informação que já existe na base: **a nota que o próprio cliente deu**, um rótulo de sentimento mais confiável do que qualquer inferência a partir do texto. Daí a divisão de trabalho: a representação do texto cuida do **assunto**, e o metadado cuida da **polaridade**.

**Onde o embedding é melhor: entender o sentido, e não só as palavras.** Com o filtro, as duas buscas trazem quase só resultados relevantes, mas não os mesmos:

- **Sinônimos e variações.** O BM25 só encontra comentários que repetem palavras da pergunta. "Demorou para entregar" não tem nenhuma palavra em comum com ela: no BM25, a pontuação é **zero**, empatada com 1.338 comentários sem relação (45% da amostra). No E5, sem filtro algum, esse comentário já aparece na 14ª posição. Com "Atraso na entrega." acontece o mesmo: como o BM25 não reduz as palavras ao radical, "atraso" e "atrasada" são termos diferentes para ele, e o comentário fica na 588ª posição, contra a 3ª no E5.
- **Sentido da frase.** O único erro do BM25 com o filtro, "Não recebi ainda. MAS está ainda no prazo", é uma frase que diz o contrário da pergunta usando as mesmas palavras.

Numa análise de voz do cliente, perder quem escreveu "demorou" em vez de "atrasou" significa subestimar o problema. É por isso que o embedding é a representação principal da busca.

**As duas técnicas são complementares.** O BM25 é forte onde o embedding é fraco: casa termos exatos e raros, como nomes de produto, marcas e códigos, que no embedding se diluem na média com o resto do texto. Sem filtro, o BM25 também pôs nos dois primeiros lugares as duas reclamações que continham "atrasada", a palavra mais específica da pergunta. Isso fundamenta a busca híbrida (E5 + BM25) prevista para a fase 2.

**Consequência para a próxima etapa.** A busca vai combinar a similaridade dos embeddings com filtros de metadados (sentimento, status da entrega, categoria, período), aplicados de três formas:
- **inferidos da pergunta:** "reclamações", "problemas" e "insatisfação" indicam sentimento negativo;
- **escolhidos pelo usuário** na interface;
- **sem filtro**, quando a pergunta é neutra ("o que os clientes dizem da embalagem?"). Nesse caso, a mistura de elogios e reclamações é o que se quer.

O filtro `delivery_status = atrasado` também funciona para esta pergunta: na busca do E5, 9 dos 10 resultados têm nota 1 ou 2. Ele não depende da nota, e sim dos dados do pedido.

**Limitação.** É um experimento com uma pergunta, numa amostra, com relevância julgada manualmente. Serve como evidência qualitativa para orientar o desenho da busca, não como medida de desempenho. A avaliação formal, com um conjunto de 30 a 50 perguntas e métricas como precisão@10 e nDCG@10, fica para a fase 2.

#### Achados 2 e 3

2. **As similaridades do E5 ficam concentradas numa faixa estreita.** Perguntas sobre o tema tiveram similaridade de 0,86 a 0,93 com o melhor resultado. Perguntas sem relação nenhuma ("qual a capital da França?", "como declarar imposto de renda") ficaram entre 0,80 e 0,85. **Consequência:** um limite fixo de similaridade sozinho não basta para o sistema recusar perguntas sem evidência. O limite precisa ser calibrado com exemplos e combinado com a verificação do próprio LLM.
3. **Textos de baixa informação atraem perguntas fora do tema.** O melhor resultado para "receita de bolo de chocolate" foi `*****`. **Consequência:** a busca vai excluir `low_information` por padrão.

Sobre os prefixos: numa comparação rápida com cinco perguntas, tirar o `query:` mudou de 1 a 4 dos 10 primeiros resultados, sem diferença na proporção de resultados com o sentimento esperado (40 de 50 nos dois casos). Cinco perguntas não permitem afirmar ganho nem perda. Por isso mantemos o protocolo recomendado pelo modelo, e a comparação entra na avaliação formal da busca, na fase 2.

## Testes

```bash
pytest
```

Os testes do índice usam um gerador de embeddings falso e determinístico, então rodam em segundos e sem baixar o modelo.

## Licença dos dados

A base é publicada pela Olist sob [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) e é redistribuída aqui para fins acadêmicos e não comerciais, com atribuição à fonte original.
