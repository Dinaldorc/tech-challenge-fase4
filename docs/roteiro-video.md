# Roteiro do vídeo de apresentação

Roteiro para o vídeo de até 7 minutos pedido no enunciado. Ele está cronometrado em cerca de 6 min 40 s, para sobrar margem. Cada trecho traz o que falar e o que mostrar na tela. As falas são uma sugestão: o importante é cobrir os pontos, e não ler palavra por palavra.

## O que o enunciado pede, e onde aparece

| Item exigido no vídeo | Trecho |
| --- | --- |
| O problema de negócio abordado | 1 |
| A arquitetura proposta | 2 |
| A estratégia utilizada para construir o pipeline RAG | 2 e 3 |
| Como ocorre a recuperação das avaliações relevantes | 3 e 4 |
| Uma demonstração prática da solução | 4, 5 e 6 |
| Exemplos de perguntas realizadas | 4, 5 e 6 |
| As evidências recuperadas para fundamentar as respostas | 4 e 5 |
| Os principais desafios e decisões técnicas da equipe | 7 |

## Antes de gravar

- **Abra o app uns minutos antes** (`streamlit run app.py`) e faça uma pergunta qualquer. A primeira pergunta demora cerca de 20 segundos, porque o modelo de embeddings carrega na memória. Depois disso, as respostas saem em 2 a 3 segundos.
- **Deixe abas prontas no navegador,** nesta ordem:
  1. o README no GitHub, no topo;
  2. o `docs/arquitetura.md` no GitHub, no diagrama "Fluxo 2: resposta a uma pergunta";
  3. o app em `http://localhost:8501`;
  4. o app em `http://localhost:8501/?visao=panorama`.
- **Grave em 1920 × 1080,** com o zoom do navegador em 110% a 125%, para os textos ficarem legíveis no vídeo.
- **Deixe a barra lateral do app sem filtros** e o limite de evidências em 8.
- **Plano B:** se a API da OpenAI demorar ou falhar durante a gravação, as respostas reais da avaliação estão em `data/eval/rag_results.csv`, e os exemplos do README mostram respostas completas.

## Trecho 1 — O problema (0:00 a 0:40)

**Tela:** o README no GitHub, no topo.

> Empresas de e-commerce recebem milhares de avaliações escritas, e é nelas que estão os motivos de insatisfação, os elogios e os padrões da experiência de compra. A nota diz se o cliente ficou satisfeito, mas não diz por quê, e ler os comentários um a um não escala. Nosso projeto é um sistema de RAG sobre as 42 mil avaliações com texto da base da Olist: o gestor pergunta em linguagem natural e recebe uma resposta construída só a partir das avaliações reais, citando cada uma delas. E quando a base não tem evidência para responder, o sistema recusa, em vez de inventar.

## Trecho 2 — Arquitetura e base de conhecimento (0:40 a 1:40)

**Tela:** o `docs/arquitetura.md`, no diagrama do fluxo 2. Percorra o diagrama de cima para baixo enquanto fala.

> A solução tem dois fluxos. O primeiro roda uma vez e constrói a base de conhecimento: cada avaliação vira um documento, com título e comentário, e com os metadados do pedido, como nota, categoria e status da entrega. Não dividimos os textos, porque nenhum comentário passa de 188 tokens. Os embeddings são gerados localmente com o modelo multilingual E5, sem custo e sem enviar os dados para fora, e ficam num índice ChromaDB.
>
> O segundo fluxo responde às perguntas. A pergunta vira embedding, a busca recupera as avaliações mais parecidas dentro dos filtros, o contexto é montado com as avaliações numeradas, e o modelo `gpt-4.1-mini` gera a resposta. Tudo isso é uma cadeia do LangChain. E há três barreiras contra respostas sem base: o corte de similaridade da busca, o julgamento do próprio modelo e a conferência automática das citações.

## Trecho 3 — Como a recuperação funciona (1:40 a 2:30)

**Tela:** o README, na seção "Achado 1", na tabela com as quatro buscas.

> Um achado da construção definiu a recuperação. Para a pergunta "entrega atrasada, não chegou no prazo", a busca por significado acertava o assunto, mas trazia elogios como "entregue no prazo": só 2 dos 10 primeiros resultados eram reclamações. O embedding entende o tema, mas não a polaridade. Testamos contra o BM25, a busca por palavras, e ele errava do mesmo jeito. A solução não foi trocar a técnica, foi usar a nota que o próprio cliente deu: com o filtro de sentimento, 10 de 10 resultados passam a ser reclamações. Por isso, quando a pergunta pede reclamações ou elogios, o sentimento é inferido e vira filtro. Além disso, comentários repetidos são agrupados, e o corte de similaridade foi calibrado com 82 perguntas, dentro e fora do tema.

## Trecho 4 — Demonstração: uma pergunta e suas evidências (2:30 a 3:40)

**Tela:** o app, na visão Perguntar. Clique no exemplo **"Problemas de entrega"**.

> Vamos ver na prática. Pergunto "o que os clientes relatam sobre problemas relacionados à entrega?".

**Tela:** a resposta aparece. Aponte para as citações [1], [2] no texto e role até os cartões das avaliações.

> A resposta cita cada afirmação com o número da avaliação, e logo abaixo estão as avaliações citadas, com o texto original, a nota, a categoria, o status da entrega e a data. Qualquer afirmação pode ser conferida na fonte.

**Tela:** abra "Como a resposta foi construída". Mostre a aba **2. Recuperação** e depois a aba **3. Contexto enviado ao LLM**.

> E o sistema mostra como chegou aqui. Esta aba lista as avaliações recuperadas, com a similaridade de cada uma e o corte. Esta outra mostra exatamente o que foi enviado ao modelo: as regras e as avaliações numeradas. Nada fica escondido.

## Trecho 5 — Demonstração: frequência com números reais (3:40 a 4:40)

**Tela:** clique no exemplo **"Reclamações mais frequentes"**.

> O enunciado também pergunta quais reclamações aparecem com maior frequência. A busca por similaridade não sabe contar: ela traz os comentários mais parecidos, e não uma amostra representativa. Por isso classificamos todas as 10.836 avaliações negativas por tema, com o LLM como anotador, e validamos essa classificação de duas formas independentes. Agora a resposta traz a contagem real, como o não recebimento em 32,1% das reclamações. E o código confere se cada porcentagem da resposta existe na contagem.

**Tela:** troque para a aba do app com `?visao=panorama`. Mostre o ranking e depois o mapa de calor.

> Essa contagem também está no Panorama. O mapa de calor cruza os temas com o status da entrega: quando o pedido atrasa, dominam o não recebimento e o atraso; quando ele chega, as reclamações passam a ser sobre o produto. E um achado de negócio: 656 clientes dizem não ter recebido pedidos que o sistema registra como entregues no prazo.

## Trecho 6 — Demonstração: quando o sistema recusa (4:40 a 5:15)

**Tela:** volte à visão Perguntar e clique em **"Fora do tema (recusa)"**.

> E quando não há evidência? "Qual a capital da França?" é recusada já na busca, sem chamar o modelo, porque nenhuma avaliação se parece com a pergunta.

**Tela:** digite "Como trocar o óleo do carro?" e pergunte.

> Esta é mais difícil: uma avaliação diz "quero trocá-lo, como fazer?", então ela passa pela busca. Mas o modelo percebe que o comentário fala de trocar um produto, e não de óleo de carro, e recusa. Na avaliação, o sistema recusou as 36 perguntas fora do tema que testamos.

## Trecho 7 — Desafios e decisões técnicas (5:15 a 6:25)

**Tela:** o `docs/processo.md` no GitHub, na tabela "Descobertas que mudaram o rumo".

> O projeto foi construído em etapas, cada uma numa pull request, e registramos 33 descobertas que mudaram o rumo. Três exemplos. Primeiro: o corte de similaridade, calibrado só com perguntas específicas, recusava perguntas genéricas como "quais as principais reclamações?". Ampliamos as perguntas de calibração e recalibramos. Segundo: tentamos contar os temas agrupando os embeddings, com K-Means e BERTopic, e medimos que não funcionava, porque as reclamações misturam vários temas. O agrupamento ficou para descobrir os temas, e a contagem passou para a classificação. Terceiro: reescrevemos o pipeline com o LangChain sob a regra de não mudar o comportamento, e conferimos que as 89 perguntas de teste devolviam exatamente as mesmas evidências. Foi esse teste que pegou um método do LangChain que devolve a distância no lugar da similaridade.
>
> Na avaliação final, com 95 perguntas, o sistema respondeu 55 das 56 legítimas, recusou todas as fora do tema, não inventou nenhuma citação e não obedeceu a uma instrução maliciosa escondida num comentário.

## Trecho 8 — Fechamento (6:25 a 6:40)

**Tela:** o README no topo, ou o Panorama.

> Em resumo: um sistema que responde com base nas avaliações reais, mostra de onde vem cada resposta, mede a frequência dos problemas e sabe dizer quando não sabe. Código, documentação e todos os experimentos estão no repositório. Obrigado.

## Dicas de gravação

- **Divida as falas entre os membros do grupo** pelos trechos. Os trechos 4, 5 e 6, de demonstração, funcionam melhor com uma pessoa só, para a tela não ficar parada nas trocas.
- **Grave por trechos** e junte depois. Assim, um erro não obriga a refazer tudo.
- **O texto das respostas pode variar um pouco entre execuções.** Os números da contagem não mudam, mas a redação sim: adapte a fala ao que aparecer na tela.
- **Fale os números devagar.** Eles são a prova do que está sendo dito, e quem avalia precisa conseguir acompanhá-los.
- **Cronometre um ensaio** antes da gravação final. Se passar de 7 minutos, os primeiros candidatos a cortar são a segunda pergunta do trecho 6 (a do óleo) e os detalhes do trecho 3.
