from pathlib import Path
import pandas as pd

_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]


def _read_sentiment140(csv_path: str | Path) -> pd.DataFrame:
    """Coluna target: 0 = negativo, 2 = neutro (só no teste), 4 = positivo."""
    df = pd.read_csv(csv_path, encoding="latin-1", names=_COLUMNS)
    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = (df["target"] == 4).astype(int)
    return df


def load_sentiment140_sample(
    csv_path: str | Path, sample_size: int = 5000, random_state: int = 42
) -> pd.DataFrame:
    """Amostra estratificada (metade positiva, metade negativa) do conjunto de treino."""
    df = _read_sentiment140(csv_path)
    sample = df.groupby("label").sample(n=sample_size // 2, random_state=random_state)
    return sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_sentiment140_test(csv_path: str | Path) -> pd.DataFrame:
    """Conjunto anotado manualmente, sem os tweets neutros (o artigo os exclui da avaliação)."""
    return _read_sentiment140(csv_path).reset_index(drop=True)
