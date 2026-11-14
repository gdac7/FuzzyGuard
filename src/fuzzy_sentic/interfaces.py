from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class SenticPipelineInterface(ABC):
    """
    Interface abstrata que garante a interoperabilidade entre o modelo
    baseline e as abordagens modernizadas.
    """

    @abstractmethod
    def __init__(self, alpha_cut: float = 0.5) -> None:
        self.alpha_cut = alpha_cut

    @abstractmethod
    def analyze_text(self, text: str) -> Tuple[float, np.ndarray]:
        """
        Processa um texto completo e retorna sua polaridade global e matriz emocional.

        Args:
            text: Texto de entrada (tweet, review, documento).

        Returns:
            Tupla (total_polarity, text_emo_table):
                - total_polarity (float): escalar no intervalo [-1.0, 1.0].
                - text_emo_table (np.ndarray): matriz (6, 4) representando os
                  6 níveis sentic ao longo das 4 dimensões da Hourglass.
        """
        raise NotImplementedError