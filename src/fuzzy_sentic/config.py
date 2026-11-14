from typing import Dict, List, Set

# Tabela 1 e Tabela 2: Dimensões afetivas e seus respectivos 6 níveis sentic (+3 a -3)
HOURGLASS_SYNSETS: Dict[str, List[str]] = {
    "Pleasantness": [
        "ecstasy.n.01", "joy.n.01", "serenity.n.01",
        "pensiveness.n.02", "sadness.n.02", "grief.n.01"
    ],
    "Attention": [
        "vigilance.n.01", "anticipation.n.01", "interest.n.01",
        "distraction.n.01", "surprise.n.01", "amazement.n.01"
    ],
    "Sensitivity": [
        "rage.n.01", "anger.n.01", "annoyance.n.01",
        "apprehension.n.01", "fear.n.01", "terror.n.01"
    ],
    "Aptitude": [
        "admiration.n.01", "trust.n.01", "acceptance.n.01",
        "boredom.n.01", "disgust.n.01", "loathing.n.01"
    ]
}

DIMENSIONS: List[str] = list(HOURGLASS_SYNSETS.keys())
SENTIMENT_LEVELS: List[str] = ["+3", "+2", "+1", "-1", "-2", "-3"]

# Léxicos básicos de modificadores do baseline (Seção 5.3)
BASELINE_INTENSIFIERS: Set[str] = {
    "very", "extremely", "really", "too", "always", "simply", "so"
}
BASELINE_NEGATORS: Set[str] = {"not", "never", "no", "n't", "cannot"}

# Limiares padrão utilizados na avaliação da Tabela 10 (Seção 6.1)
EVALUATION_THRESHOLDS: List[float] = [0.1, 0.2, 0.4, 0.6]