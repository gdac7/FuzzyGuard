from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.fuzzy_sentic.config import DIMENSIONS, SENTIC_LEVEL_NAMES
from src.fuzzy_sentic.interfaces import SenticPipelineInterface

SPEAKERS = ("Pessoa A", "Pessoa B")
LABEL_THRESHOLD = 0.1


@dataclass
class Message:
    speaker: str
    text: str


@dataclass
class Classification:
    polarity: float
    label: str
    emotions: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class ConversationResult:
    conversation: Classification
    by_speaker: Dict[str, Classification]
    last_message: Optional[Classification]


def as_sentence(text: str) -> str:
    """Mensagens de chat raramente terminam com pontuação; sem ela, viram uma frase só."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def polarity_label(polarity: float, threshold: float = LABEL_THRESHOLD) -> str:
    if polarity > threshold:
        return "positiva"
    if polarity < -threshold:
        return "negativa"
    return "neutra"


def dominant_emotions(table: np.ndarray, alpha_cut: float) -> List[Tuple[str, float]]:
    cells = [
        (SENTIC_LEVEL_NAMES[dim][row], float(table[row, col]))
        for col, dim in enumerate(DIMENSIONS)
        for row in range(6)
        if table[row, col] > 0 and table[row, col] >= alpha_cut
    ]
    return sorted(cells, key=lambda cell: cell[1], reverse=True)


def classify_texts(pipeline: SenticPipelineInterface, texts: List[str]) -> Classification:
    polarity, table = pipeline.analyze_text(" ".join(as_sentence(t) for t in texts))
    return Classification(
        polarity=polarity,
        label=polarity_label(polarity),
        emotions=dominant_emotions(table, pipeline.alpha_cut),
    )


def classify_conversation(
    pipeline: SenticPipelineInterface, messages: List[Message]
) -> ConversationResult:
    by_speaker = {
        speaker: classify_texts(pipeline, [m.text for m in messages if m.speaker == speaker])
        for speaker in SPEAKERS
        if any(m.speaker == speaker for m in messages)
    }
    return ConversationResult(
        conversation=classify_texts(pipeline, [m.text for m in messages]),
        by_speaker=by_speaker,
        last_message=classify_texts(pipeline, [messages[-1].text]) if messages else None,
    )
