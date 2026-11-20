from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
OUT = Path("resultados")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "font.size": 10,
})


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def grouped_bars(ax, models: dict[str, pd.DataFrame], polarity: str, metric: str) -> None:
    thresholds = next(iter(models.values()))["Threshold"].unique()
    x = np.arange(len(thresholds))
    n = len(models)
    width = 0.8 / n
    for i, (name, df) in enumerate(models.items()):
        values = df[df["Polarity"] == polarity].set_index("Threshold").loc[thresholds, metric].to_numpy()
        # 2px de superfície entre barras vizinhas
        bars = ax.bar(
            x + (i - (n - 1) / 2) * width, values, width,
            color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=1.5,
        )
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, thresholds)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Limiar t")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)


def figure_metric(models: dict[str, pd.DataFrame], metric: str, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, polarity, label in zip(axes, ["Positive", "Negative"], ["Tweets positivos", "Tweets negativos"]):
        grouped_bars(ax, models, polarity, metric)
        ax.set_title(label, loc="left")
    axes[0].set_ylabel(metric)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(models), bbox_to_anchor=(0.01, 0.935), loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_precision_recall(models: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for row, metric in enumerate(["Precision", "Recall"]):
        for col, (polarity, label) in enumerate([("Positive", "positivos"), ("Negative", "negativos")]):
            ax = axes[row, col]
            grouped_bars(ax, models, polarity, metric)
            ax.set_title(f"{metric} — tweets {label}", loc="left")
            if col == 0:
                ax.set_ylabel(metric)
    fig.suptitle("Precision e Recall por limiar (Sentiment140, 5000 tweets)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=len(models), bbox_to_anchor=(0.01, 0.965), loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_emotions(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (name, file) in zip(axes, [
        ("Baseline corrigido", "fig2_LoiaSenatore_2014.csv"),
        ("NeoSenticFuzzy", "fig2_NeoSenticFuzzy.csv"),
    ]):
        df = pd.read_csv(RESULTS / file, dtype={"Level": str})
        df = df[(df["Cut"] == 0.2) & (df["Dimension"] == "Pleasantness")]
        x = np.arange(len(df))
        width = 0.4
        for i, (column, label) in enumerate([("PositiveTweets", "Tweets positivos"), ("NegativeTweets", "Tweets negativos")]):
            bars = ax.bar(x + (i - 0.5) * width, df[column], width, color=SERIES[i], label=label,
                          edgecolor=SURFACE, linewidth=1.5)
            ax.bar_label(bars, fontsize=7, color=INK_2, padding=2)
        ax.set_xticks(x, [f"{e}\n({lvl})" for e, lvl in zip(df["Emotion"], df["Level"])])
        ax.tick_params(length=0)
        ax.set_axisbelow(True)
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("Número de tweets")
    fig.suptitle("Emoções da dimensão Pleasantness com corte t = 0.2 (como na Figura 2 do artigo)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, bbox_to_anchor=(0.01, 0.935), loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = {
        "Baseline original": "LoiaSenatore_original",
        "Baseline corrigido": "LoiaSenatore_2014",
        "NeoSenticFuzzy": "NeoSenticFuzzy",
    }
    sample = {label: load(f"table10_{key}_sentiment140_5000.csv") for label, key in models.items()}
    manual = {label: load(f"table10_{key}_sentiment140_teste_manual.csv") for label, key in models.items()}
    figure_metric(sample, "F-measure", "F-measure por limiar (Sentiment140, 5000 tweets)",
                  OUT / "01_fmeasure_5000.png")
    figure_precision_recall(sample, OUT / "02_precision_recall_5000.png")
    figure_metric(manual, "F-measure", "F-measure por limiar (teste anotado manualmente, 359 tweets)",
                  OUT / "03_fmeasure_teste_manual.png")
    figure_emotions(OUT / "04_emocoes_pleasantness.png")
    print(f"Gráficos salvos em {OUT}/")


if __name__ == "__main__":
    main()
