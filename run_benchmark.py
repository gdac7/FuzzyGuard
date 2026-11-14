from pathlib import Path
from src.fuzzy_sentic.evaluation.data_loader import load_sentiment140_sample
from src.fuzzy_sentic.evaluation.metrics import evaluate_thresholds
from src.fuzzy_sentic.models.baseline import LoiaSenatoreBaseline
# from src.fuzzy_sentic.models.neosentic import NeoSenticFuzzy


def main() -> None:
    data_path = Path("data/training.1600000.processed.noemoticon.csv")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"Arquivo não encontrado em '{data_path}'.")
        print("Coloque o CSV do Sentiment140 dentro da pasta 'data/' para executar o benchmark.")
        return

    print("Carregando amostra do Sentiment140...")
    sample_df = load_sentiment140_sample(data_path, sample_size=1000, random_state=42)

    models = {
        "LoiaSenatore_2014": LoiaSenatoreBaseline(alpha_cut=0.5),
        # "NeoSenticFuzzy_2026": NeoSenticFuzzy(alpha_cut=0.5),
    }

    for model_name, pipeline in models.items():
        print(f"\nAvaliando {model_name}...")
        df_results = evaluate_thresholds(pipeline, sample_df)
        print(df_results.to_string(index=False))

        output_csv = results_dir / f"metrics_{model_name}.csv"
        df_results.to_csv(output_csv, index=False)
        print(f"Resultados salvos em: {output_csv}")


if __name__ == "__main__":
    main()