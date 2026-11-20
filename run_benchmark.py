import time
from pathlib import Path

import pandas as pd

from src.fuzzy_sentic.evaluation.data_loader import (
    load_sentiment140_sample,
    load_sentiment140_test,
)
from src.fuzzy_sentic.evaluation.emotions import emotion_counts
from src.fuzzy_sentic.evaluation.metrics import neutral_rate, predict, threshold_metrics
from src.fuzzy_sentic.models.baseline import LoiaSenatoreBaseline
from src.fuzzy_sentic.models.neosentic import NeoSenticFuzzy

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
TRAIN_CSV = DATA_DIR / "training.1600000.processed.noemoticon.csv"
TEST_CSV = DATA_DIR / "testdata.manual.2009.06.14.csv"
DATASET_TITLES = {
    "sentiment140_5000": "Sentiment140: amostra estratificada de 5000 tweets do treino",
    "sentiment140_teste_manual": "Sentiment140: conjunto de teste anotado manualmente (sem neutros)",
}
# Versão do baseline anterior às correções, avaliada nos mesmos dados (ver COMP.md)
REPORT_MODELS = {
    "Baseline original": "LoiaSenatore_original",
    "Baseline corrigido": "LoiaSenatore_2014",
    "NeoSenticFuzzy": "NeoSenticFuzzy",
}
SAMPLE_SIZE = 5000

EMOTIONS_NOTE = """## Por que o baseline ativa muito mais emoções

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
"""


def _comparison_table(columns: dict[str, pd.DataFrame]) -> str:
    keys = ["Threshold", "Polarity"]
    merged = None
    for name, df in columns.items():
        df = df[keys + ["Precision", "Recall", "F-measure"]].rename(
            columns={"Precision": f"{name} P", "Recall": f"{name} R", "F-measure": f"{name} F"}
        )
        merged = df if merged is None else merged.merge(df, on=keys)
    return merged.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print(f"Dataset não encontrado em '{DATA_DIR}/'.")
        print("Baixe https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip e extraia em 'data/'.")
        return
    RESULTS_DIR.mkdir(exist_ok=True)

    datasets = {
        "sentiment140_5000": load_sentiment140_sample(TRAIN_CSV, sample_size=SAMPLE_SIZE),
        "sentiment140_teste_manual": load_sentiment140_test(TEST_CSV),
    }
    # alpha_cut = 0 preserva a tabela emocional completa; os cortes da Figura 2
    # são aplicados depois (para a união por máximo, cortar antes ou depois dá o mesmo).
    models = {
        "LoiaSenatore_2014": LoiaSenatoreBaseline(alpha_cut=0.0),
        "NeoSenticFuzzy": NeoSenticFuzzy(alpha_cut=0.0),
    }

    summary = []
    for model_name, pipeline in models.items():
        for data_name, df in datasets.items():
            start = time.perf_counter()
            polarities, emo_tables = predict(pipeline, df, desc=f"{model_name} / {data_name}")
            elapsed = time.perf_counter() - start

            metrics = threshold_metrics(polarities, df["label"].to_numpy())
            metrics.to_csv(RESULTS_DIR / f"table10_{model_name}_{data_name}.csv", index=False)
            if data_name == "sentiment140_5000":
                emotion_counts(polarities, emo_tables).to_csv(
                    RESULTS_DIR / f"fig2_{model_name}.csv", index=False
                )
            summary.append({
                "Modelo": model_name,
                "Dataset": data_name,
                "Tweets": len(df),
                "Polaridade zero (%)": round(100 * neutral_rate(polarities), 1),
                "Tempo (s)": round(elapsed, 1),
            })
            print(f"\n{model_name} / {data_name}")
            print(metrics.to_string(index=False))

    write_report(pd.DataFrame(summary))


def write_report(summary: pd.DataFrame) -> None:
    report = ["# Comparação entre os modelos (mesmos dados)", "", "Ver `COMP.md` para o protocolo.", ""]
    for data_name, title in DATASET_TITLES.items():
        columns = {
            label: pd.read_csv(path)
            for label, key in REPORT_MODELS.items()
            if (path := RESULTS_DIR / f"table10_{key}_{data_name}.csv").exists()
        }
        report += [f"## {title}", "", _comparison_table(columns), ""]
    report += ["## Execução", "", summary.to_markdown(index=False), "", EMOTIONS_NOTE]
    (RESULTS_DIR / "comparacao.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nRelatório salvo em {RESULTS_DIR / 'comparacao.md'}")


if __name__ == "__main__":
    main()
