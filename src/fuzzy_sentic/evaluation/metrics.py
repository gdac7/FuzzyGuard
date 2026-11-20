from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.fuzzy_sentic.config import EVALUATION_THRESHOLDS
from src.fuzzy_sentic.interfaces import SenticPipelineInterface


def predict(
    pipeline: SenticPipelineInterface, dataset: pd.DataFrame, desc: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    polarities, tables = [], []
    for text in tqdm(dataset["text"], desc=desc, leave=False):
        polarity, table = pipeline.analyze_text(str(text))
        polarities.append(polarity)
        tables.append(table)
    return np.array(polarities, dtype=float), np.array(tables)


def _prf(retrieved: np.ndarray, relevant: np.ndarray) -> Tuple[float, float, float]:
    tp = int((retrieved & relevant).sum())
    precision = tp / retrieved.sum() if retrieved.sum() > 0 else 0.0
    recall = tp / relevant.sum() if relevant.sum() > 0 else 0.0
    f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f


def threshold_metrics(
    polarities: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Protocolo da Tabela 10: para t > 0 são recuperados os tweets com polaridade > t
    (positivos) ou < -t (negativos); os relevantes são os rotulados pelo oráculo.
    """
    rows = []
    for t in thresholds or EVALUATION_THRESHOLDS:
        for name, retrieved, relevant in (
            ("Positive", polarities > t, labels == 1),
            ("Negative", polarities < -t, labels == 0),
        ):
            p, r, f = _prf(retrieved, relevant)
            rows.append({
                "Threshold": f"±{t}",
                "Polarity": name,
                "Precision": round(p, 3),
                "Recall": round(r, 3),
                "F-measure": round(f, 3),
            })
    return pd.DataFrame(rows)


def neutral_rate(polarities: np.ndarray) -> float:
    """Fração de textos que o modelo deixa com polaridade exatamente 0."""
    return float(np.isclose(polarities, 0.0).mean())


def evaluate_thresholds(
    pipeline: SenticPipelineInterface,
    dataset: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    polarities, _ = predict(pipeline, dataset)
    return threshold_metrics(polarities, dataset["label"].to_numpy(), thresholds)
