from pathlib import Path
import pandas as pd


def load_sentiment140_sample(
    csv_path: str | Path, sample_size: int = 5000, random_state: int = 42
) -> pd.DataFrame:
    """
    Carrega uma amostra estratificada/aleatória do dataset Sentiment140.
    Coluna 0: target (0 = negativo, 4 = positivo)
    Coluna 5: text (conteúdo do tweet)
    """
    column_names = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(csv_path, encoding="latin-1", names=column_names)

    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = (df["target"] == 4).astype(int)

    return df.sample(n=min(sample_size, len(df)), random_state=random_state)