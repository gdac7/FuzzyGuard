# Comparação entre os modelos (mesmos dados)

Ver `COMP.md` para o protocolo.

## Sentiment140: amostra estratificada de 5000 tweets do treino

| Threshold   | Polarity   |   Baseline original P |   Baseline original R |   Baseline original F |   Baseline corrigido P |   Baseline corrigido R |   Baseline corrigido F |   NeoSenticFuzzy P |   NeoSenticFuzzy R |   NeoSenticFuzzy F |
|:------------|:-----------|----------------------:|----------------------:|----------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-------------------:|-------------------:|
| ±0.1        | Positive   |                 0.679 |                 0.318 |                 0.433 |                  0.665 |                  0.342 |                  0.452 |              0.674 |              0.507 |              0.579 |
| ±0.1        | Negative   |                 0.803 |                 0.196 |                 0.315 |                  0.773 |                  0.219 |                  0.341 |              0.747 |              0.355 |              0.481 |
| ±0.2        | Positive   |                 0.746 |                 0.170 |                 0.276 |                  0.749 |                  0.190 |                  0.303 |              0.696 |              0.335 |              0.452 |
| ±0.2        | Negative   |                 0.865 |                 0.087 |                 0.158 |                  0.864 |                  0.102 |                  0.182 |              0.803 |              0.229 |              0.357 |
| ±0.4        | Positive   |                 0.886 |                 0.047 |                 0.089 |                  0.877 |                  0.057 |                  0.107 |              0.801 |              0.155 |              0.259 |
| ±0.4        | Negative   |                 0.882 |                 0.024 |                 0.047 |                  0.869 |                  0.021 |                  0.041 |              0.879 |              0.078 |              0.144 |
| ±0.6        | Positive   |                 0.984 |                 0.024 |                 0.047 |                  0.951 |                  0.031 |                  0.060 |              0.868 |              0.037 |              0.071 |
| ±0.6        | Negative   |                 0.889 |                 0.006 |                 0.013 |                  0.900 |                  0.007 |                  0.014 |              0.948 |              0.022 |              0.043 |

## Sentiment140: conjunto de teste anotado manualmente (sem neutros)

| Threshold   | Polarity   |   Baseline original P |   Baseline original R |   Baseline original F |   Baseline corrigido P |   Baseline corrigido R |   Baseline corrigido F |   NeoSenticFuzzy P |   NeoSenticFuzzy R |   NeoSenticFuzzy F |
|:------------|:-----------|----------------------:|----------------------:|----------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-------------------:|-------------------:|
| ±0.1        | Positive   |                 0.775 |                 0.511 |                 0.616 |                  0.797 |                  0.582 |                  0.673 |              0.788 |              0.692 |              0.737 |
| ±0.1        | Negative   |                 0.851 |                 0.226 |                 0.357 |                  0.860 |                  0.277 |                  0.419 |              0.845 |              0.554 |              0.669 |
| ±0.2        | Positive   |                 0.850 |                 0.280 |                 0.421 |                  0.897 |                  0.335 |                  0.488 |              0.835 |              0.610 |              0.705 |
| ±0.2        | Negative   |                 0.913 |                 0.119 |                 0.210 |                  0.968 |                  0.169 |                  0.288 |              0.890 |              0.367 |              0.520 |
| ±0.4        | Positive   |                 0.900 |                 0.049 |                 0.094 |                  0.917 |                  0.060 |                  0.113 |              0.885 |              0.297 |              0.444 |
| ±0.4        | Negative   |                 1.000 |                 0.045 |                 0.086 |                  1.000 |                  0.068 |                  0.127 |              0.914 |              0.181 |              0.302 |
| ±0.6        | Positive   |                 1.000 |                 0.027 |                 0.053 |                  0.800 |                  0.022 |                  0.043 |              0.875 |              0.038 |              0.074 |
| ±0.6        | Negative   |                 1.000 |                 0.011 |                 0.022 |                  1.000 |                  0.023 |                  0.044 |              1.000 |              0.034 |              0.066 |

## Execução

| Modelo            | Dataset                   |   Tweets |   Polaridade zero (%) |   Tempo (s) |
|:------------------|:--------------------------|---------:|----------------------:|------------:|
| LoiaSenatore_2014 | sentiment140_5000         |     5000 |                   4.6 |        35.2 |
| LoiaSenatore_2014 | sentiment140_teste_manual |      359 |                   1.9 |         2.8 |
| NeoSenticFuzzy    | sentiment140_5000         |     5000 |                   7.6 |       146.6 |
| NeoSenticFuzzy    | sentiment140_teste_manual |      359 |                   3.6 |         7.5 |

## Por que o baseline ativa muito mais emoções

Com α-cut = 0.5, o baseline costuma marcar 8 a 10 emoções por frase ("I am happy": 8;
"I am so angry": 10), enquanto o NeoSenticFuzzy marca de 0 a 3. Possíveis motivos:

1. **Wu-Palmer não separa bem as emoções.** Todos os 24 rótulos da Hourglass ficam no mesmo
   ramo do WordNet (`feeling.n.01`), então qualquer palavra afetiva fica "perto" de quase
   todos eles. Para `happiness`, a similaridade vai de 0.33 a 0.89 e 20 das 24 células já
   passam de 0.5, inclusive Rage, Fear e Disgust. O próprio artigo reconhece o problema
   (Tabela 3: happiness é tão similar a sadness quanto a joy).
2. **A heurística dos opostos garante valores altos.** A âncora de cada par recebe
   max(v, 1 − v) ≥ 0.5. Em todo par que não empata, pelo menos um rótulo passa o α-cut:
   até 12 emoções por palavra só por construção.
3. **No Neo, a similaridade é semântica e mais seletiva.** O cosseno entre embeddings da
   definição da palavra e dos rótulos discrimina melhor ("happy" fica perto de Joy e
   longe de Rage). A normalização com θ = 0.3 zera similaridades fracas, e o lado oposto
   de cada par vai a 0.

O efeito colateral é que o Neo às vezes fica restritivo demais: "I am sad" e "I love this"
não ativam nenhuma emoção acima de 0.5. A Figura 2 (`results/fig2_*.csv`) mostra os dois
extremos: com corte 0.2, o baseline ativa quase todas as emoções em quase todos os tweets,
e o Neo separa positivos de negativos (Joy em 523 positivos × 44 negativos), mas com menos
tweets por emoção.

## Por que o Neo recupera mais tweets, mas ativa menos emoções

Parece contraditório, mas são duas saídas diferentes de cada modelo, calculadas de forma
independente e com limiares diferentes:

| Saída | Vem de | Limiar | Onde aparece |
| :--- | :--- | :--- | :--- |
| Polaridade (−1 a +1) | SentiWordNet | ±t (0.1, 0.2, 0.4, 0.6) | Precision, Recall e F-measure |
| Emoções (matriz 6×4, 0 a 1) | Wu-Palmer (baseline) ou cosseno de embeddings (Neo) | corte 0.2 | Figura 2 / `fig2_*.csv` |

- **Baseline:** polaridade fraca, porque as médias (Eq. 1 sobre todos os sentidos e média de
  todas as palavras da frase) puxam tudo para perto de zero; poucos tweets passam de ±t e o
  recall é baixo. As emoções, ao contrário, são altas em quase tudo, porque o Wu-Palmer dá
  similaridade alta entre qualquer palavra e quase todas as emoções; quase todo tweet passa
  do corte 0.2.
- **NeoSenticFuzzy:** polaridade forte, porque a IOWA dá mais peso à opinião mais forte da
  frase e o sentido de cada palavra é escolhido pelo contexto; mais tweets passam de ±t e o
  recall é maior. As emoções são seletivas, porque o cosseno discrimina melhor e θ zera as
  similaridades fracas; poucos tweets passam do corte.

Na Figura 2, "tweets positivos/negativos" usa só o sinal da polaridade (> 0 ou < 0), sem o
limiar t. Por isso quase todos os tweets entram em algum grupo nos dois modelos, e o tamanho
das barras depende apenas de quantos ativam cada emoção.
