# Protocolo de comparação

**Regra:** só comparamos modelos avaliados **nos mesmos dados e com o mesmo gabarito**. Por isso a Tabela 10 do artigo de Loia & Senatore (2014) **não aparece** em nenhuma comparação, tabela ou gráfico deste repositório.

## Por que a Tabela 10 do artigo foi removida

1. **Os tweets são outros.** O artigo avaliou 5000 tweets coletados pelos próprios autores com a Twitter Search API, a partir de buscas por tópicos. Esse conjunto nunca foi publicado, então não há como avaliar nenhum modelo nele.
2. **O gabarito é outro.** No artigo, o "oráculo" foi o classificador automático da API do Sentiment140, ou seja, outro sistema de análise de sentimento. Aqui o gabarito são os rótulos do dataset Sentiment140: no treino, rótulos automáticos derivados dos emoticons dos tweets; no teste, anotação manual. São problemas de dificuldade diferente, e concordar com outro classificador não é o mesmo que acertar o rótulo.
3. **A própria tabela tem inconsistências.** Dois valores de F-measure não batem com a Precision e o Recall da mesma linha (±0.2 positivo: P = 0.88 e R = 0.71 dão F = 0.79, não 0.89; ±0.4 positivo: P = 0.88 e R = 0.58 dão F = 0.70, não 0.79).

Colocar esses números lado a lado com os nossos sugeriria uma comparação que não existe. A diferença refletiria os dados e o gabarito, não a qualidade dos métodos.

Um indício disso: o **baseline corrigido** reproduz os números intermediários do artigo (Eq. 1, Tabelas 3, 4, 6 e 8, conferidos em `tests/test_baseline_paper.py`). Mesmo assim, nos nossos dados ele chega a F = 0.45 em ±0.1, contra os 0.81 publicados. Com o mesmo método e números tão diferentes, a avaliação é que mudou.

## Pelo mesmo motivo: o baseline original foi reavaliado

O resultado antigo do baseline (antes das correções) tinha sido calculado em 1000 tweets aleatórios, outra amostra. Ele foi descartado, e o código antigo (`src/fuzzy_sentic/models/baseline.py` no commit `829e864` da `main`) foi rodado de novo nos mesmos dados dos outros modelos.

## Dados usados por todos os modelos

| Conjunto | Origem | Tweets | Gabarito |
| :--- | :--- | ---: | :--- |
| Amostra do treino | `training.1600000.processed.noemoticon.csv` | 5000 (2500 positivos, 2500 negativos, `random_state=42`) | rótulos por emoticon |
| Teste manual | `testdata.manual.2009.06.14.csv` | 359 (neutros excluídos) | anotação humana |

Os dois arquivos são públicos: https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip. Os 5000 tweets são **nossa** amostra do Sentiment140. O número é igual ao do artigo só para manter a mesma escala; os tweets não são os mesmos.

## Métricas

As mesmas da Seção 6.1 do artigo, aplicadas aos nossos dados. Para cada limiar t ∈ {0.1, 0.2, 0.4, 0.6}:
- **Positivos recuperados:** tweets com polaridade > t; relevantes: rotulados como positivos.
- **Negativos recuperados:** tweets com polaridade < −t; relevantes: rotulados como negativos.
- **Precision, Recall e F-measure** por classe.

## Resultados (F-measure)

| Limiar | Classe | Baseline original | Baseline corrigido | NeoSenticFuzzy |
| :--- | :--- | ---: | ---: | ---: |
| ±0.1 | Positivo | 0.433 | 0.452 | **0.579** |
| ±0.1 | Negativo | 0.315 | 0.341 | **0.481** |
| ±0.2 | Positivo | 0.276 | 0.303 | **0.452** |
| ±0.2 | Negativo | 0.158 | 0.182 | **0.357** |
| ±0.4 | Positivo | 0.089 | 0.107 | **0.259** |
| ±0.4 | Negativo | 0.047 | 0.041 | **0.144** |
| ±0.6 | Positivo | 0.047 | 0.060 | **0.071** |
| ±0.6 | Negativo | 0.013 | 0.014 | **0.043** |

*Amostra de 5000 tweets.* No teste manual, em ±0.1: original 0.616 / 0.357, corrigido 0.673 / 0.419, **Neo 0.737 / 0.669** (positivo / negativo).

O NeoSenticFuzzy tem o maior F-measure em todos os limiares, nos dois conjuntos. A Precision fica parecida entre os três modelos; o ganho vem do Recall. Isso acontece porque o sentido de cada palavra é escolhido pelo contexto e a agregação IOWA deixa de diluir a polaridade em palavras quase neutras.

## Onde estão os números

- `results/comparacao.md`: Precision, Recall e F de todos os modelos nos dois conjuntos.
- `results/table10_<modelo>_<conjunto>.csv`: métricas de cada execução.
- `resultados_execucao/*.png`: gráficos.
