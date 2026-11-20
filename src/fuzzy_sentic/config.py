from typing import Dict, List, Set

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

SENTIC_LEVEL_NAMES: Dict[str, List[str]] = {
    dim: [name.split(".")[0].capitalize() for name in synsets]
    for dim, synsets in HOURGLASS_SYNSETS.items()
}

OPPOSITE_PAIRS = [(0, 5), (1, 4), (2, 3)]

# Dimensões em que os opostos são antônimos efetivos e a âncora vem da polaridade
POLARITY_ANCHORED_DIMENSIONS: Set[str] = {"Pleasantness", "Aptitude"}

BASELINE_INTENSIFIERS: Set[str] = {
    "very", "extremely", "really", "too", "always", "simply", "so"
}
BASELINE_NEGATORS: Set[str] = {"not", "never", "no", "n't", "cannot"}

# Sem o WordNet-Affect, cada emoticon vira o substantivo do WordNet que nomeia a emoção.
EMOTICON_EMOTIONS: Dict[str, str] = {
    ":)": "joy", ":-)": "joy", ":]": "joy", "=)": "joy", ":D": "joy", ":-D": "joy",
    "=D": "joy", ";)": "joy", ";-)": "joy", "xD": "joy", "XD": "joy",
    ":(": "sadness", ":-(": "sadness", ":[": "sadness", "=(": "sadness",
    ":'(": "sadness", ":'-(": "sadness",
    ">:(": "anger", ":@": "anger",
    ":O": "surprise", ":-O": "surprise", ":o": "surprise",
    "<3": "love",
    "lol": "amusement",
}

EVALUATION_THRESHOLDS: List[float] = [0.1, 0.2, 0.4, 0.6]
