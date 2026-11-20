from typing import Iterable
import numpy as np
import pandas as pd

from src.fuzzy_sentic.config import DIMENSIONS, SENTIC_LEVEL_NAMES, SENTIMENT_LEVELS


def emotion_counts(
    polarities: np.ndarray, tables: np.ndarray, cuts: Iterable[float] = (0.2, 0.8)
) -> pd.DataFrame:
    """
    Figura 2: quantos tweets positivos e negativos (pela polaridade do próprio
    modelo) ativam cada nível sentic após o corte t na tabela emocional do texto.
    """
    positive, negative = polarities > 0, polarities < 0
    rows = []
    for cut in cuts:
        active = tables >= cut
        for col, dim in enumerate(DIMENSIONS):
            for row, level in enumerate(SENTIMENT_LEVELS):
                cell = active[:, row, col]
                rows.append({
                    "Cut": cut,
                    "Dimension": dim,
                    "Level": level,
                    "Emotion": SENTIC_LEVEL_NAMES[dim][row],
                    "PositiveTweets": int((cell & positive).sum()),
                    "NegativeTweets": int((cell & negative).sum()),
                })
    return pd.DataFrame(rows)
