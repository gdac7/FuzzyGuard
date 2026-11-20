import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fuzzy_sentic.evaluation.data_loader import (
    load_sentiment140_sample,
    load_sentiment140_test,
)
from src.fuzzy_sentic.evaluation.emotions import emotion_counts
from src.fuzzy_sentic.evaluation.metrics import (
    neutral_rate,
    predict,
    threshold_metrics,
)
from src.fuzzy_sentic.models.baseline import LoiaSenatoreBaseline
from src.fuzzy_sentic.models.neosentic import NeoSenticFuzzy

DATA_DIR = Path("data")
OUT_DIR = Path("resultados_execucao")
TRAIN_CSV = DATA_DIR / "training.1600000.processed.noemoticon.csv"
TEST_CSV = DATA_DIR / "testdata.manual.2009.06.14.csv"


def compare_models(
    models: dict, dataset: pd.DataFrame, split_name: str
) -> tuple[pd.DataFrame, dict]:
    labels = dataset["label"].to_numpy()
    all_metrics = []
    raw_outputs = {}

    for model_name, pipeline in models.items():
        t0 = time.perf_counter()
        polarities, tables = predict(pipeline, dataset, desc=f"{split_name} | {model_name}")
        elapsed = time.perf_counter() - t0

        raw_outputs[model_name] = (polarities, tables)
        df_m = threshold_metrics(polarities, labels)
        df_m.insert(0, "Model", model_name)
        all_metrics.append(df_m)

        neut = neutral_rate(polarities)
        print(
            f"[{split_name}] {model_name}: {len(dataset)} tweets em {elapsed:.2f}s "
            f"({elapsed / len(dataset) * 1000:.2f} ms/tweet) | Taxa neutra (p=0): {neut:.1%}"
        )

    combined = pd.concat(all_metrics, ignore_index=True)
    pivot = combined.pivot(
        index=["Threshold", "Polarity"],
        columns="Model",
        values=["Precision", "Recall", "F-measure"],
    )
    return pivot, raw_outputs


def plot_fmeasure_comparison(pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    f_df = pivot["F-measure"].reset_index()
    labels = f_df["Threshold"] + "\n" + f_df["Polarity"]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f_df["Baseline (2014)"], width, label="Baseline (Loia & Senatore, 2014)")
    ax.bar(x + width / 2, f_df["NeoSenticFuzzy"], width, label="NeoSenticFuzzy (Atualizado)")

    ax.set_ylabel("F-measure")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            "Arquivos do Sentiment140 não encontrados em data/. "
            "Baixe trainingandtestdata.zip e extraia em data/."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando datasets (protocolo docs/COMP.md)...")
    sample_5k = load_sentiment140_sample(TRAIN_CSV, sample_size=5000, random_state=42)
    manual_test = load_sentiment140_test(TEST_CSV)

    print("Inicializando pipelines...")
    models = {
        "Baseline (2014)": LoiaSenatoreBaseline(alpha_cut=0.5),
        "NeoSenticFuzzy": NeoSenticFuzzy(alpha_cut=0.5),
    }

    print("\n=== 1. AMOSTRA ESTRATIFICADA (5.000 TWEETS) ===")
    pivot_5k, outputs_5k = compare_models(models, sample_5k, "Sample5k")
    print("\n" + pivot_5k.to_markdown())
    pivot_5k.to_csv(OUT_DIR / "comparacao_sample5k.csv")
    plot_fmeasure_comparison(
        pivot_5k,
        "F-measure por Limiar e Polaridade — Amostra Sentiment140 (5.000 tweets)",
        OUT_DIR / "fmeasure_sample5k.png",
    )

    print("\n=== 2. TESTE ANOTADO MANUALMENTE (359 TWEETS) ===")
    pivot_test, _ = compare_models(models, manual_test, "ManualTest")
    print("\n" + pivot_test.to_markdown())
    pivot_test.to_csv(OUT_DIR / "comparacao_manual359.csv")
    plot_fmeasure_comparison(
        pivot_test,
        "F-measure por Limiar e Polaridade — Teste Manual Sentiment140 (359 tweets)",
        OUT_DIR / "fmeasure_manual359.png",
    )

    print("\n=== 3. DISTRIBUIÇÃO DE EMOÇÕES (FIGURA 2 — AMOSTRA 5.000 TWEETS) ===")
    for model_name, (pols, tabs) in outputs_5k.items():
        emo_df = emotion_counts(pols, tabs, cuts=(0.2, 0.8))
        slug = "baseline" if "Baseline" in model_name else "neosentic"
        emo_df.to_csv(OUT_DIR / f"emocoes_fig2_{slug}.csv", index=False)
        top_02 = (
            emo_df[emo_df["Cut"] == 0.2]
            .assign(Total=lambda d: d["PositiveTweets"] + d["NegativeTweets"])
            .sort_values("Total", ascending=False)
            .head(6)
        )
        print(f"\nTop 6 emoções ativas (corte 0.2) — {model_name}:")
        print(top_02[["Dimension", "Level", "Emotion", "PositiveTweets", "NegativeTweets"]].to_markdown(index=False))


if __name__ == "__main__":
    main()