from typing import List, Optional
import numpy as np
import pandas as pd

from src.fuzzy_sentic.config import EVALUATION_THRESHOLDS
from src.fuzzy_sentic.interfaces import SenticPipelineInterface


def evaluate_thresholds(
    pipeline: SenticPipelineInterface,
    dataset: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Avalia Precision, Recall e F-measure variando o limiar t (Tabela 10).
    """
    if thresholds is None:
        thresholds = EVALUATION_THRESHOLDS

    predictions = []
    ground_truth = dataset["label"].to_numpy()

    for text in dataset["text"]:
        pred_polarity, _ = pipeline.analyze_text(str(text))
        predictions.append(pred_polarity)

    predictions_arr = np.array(predictions, dtype=float)
    total_pos_real = int((ground_truth == 1).sum())
    total_neg_real = int((ground_truth == 0).sum())

    results = []
    for t in thresholds:
        # Classe Positiva (polaridade > t)
        pred_pos = predictions_arr > t
        tp_pos = int((pred_pos & (ground_truth == 1)).sum())
        retrieved_pos = int(pred_pos.sum())

        p_pos = tp_pos / retrieved_pos if retrieved_pos > 0 else 0.0
        r_pos = tp_pos / total_pos_real if total_pos_real > 0 else 0.0
        f_pos = (2 * p_pos * r_pos) / (p_pos + r_pos) if (p_pos + r_pos) > 0 else 0.0

        results.append({
            "Threshold": f"±{t}",
            "Polarity": "Positive",
            "Precision": round(p_pos, 3),
            "Recall": round(r_pos, 3),
            "F-measure": round(f_pos, 3),
        })

        # Classe Negativa (polaridade < -t)
        pred_neg = predictions_arr < -t
        tp_neg = int((pred_neg & (ground_truth == 0)).sum())
        retrieved_neg = int(pred_neg.sum())

        p_neg = tp_neg / retrieved_neg if retrieved_neg > 0 else 0.0
        r_neg = tp_neg / total_neg_real if total_neg_real > 0 else 0.0
        f_neg = (2 * p_neg * r_neg) / (p_neg + r_neg) if (p_neg + r_neg) > 0 else 0.0

        results.append({
            "Threshold": f"±{t}",
            "Polarity": "Negative",
            "Precision": round(p_neg, 3),
            "Recall": round(r_neg, 3),
            "F-measure": round(f_neg, 3),
        })

    return pd.DataFrame(results)