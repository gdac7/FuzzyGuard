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

NEO_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEO_SPACY_MODEL = "en_core_web_sm"

# Hedges de Zadeh: |p|' = |p| ** (1 / alpha). alpha > 1 intensifica, alpha < 1 atenua.
NEO_HEDGES: Dict[str, float] = {
    "very": 2.0, "extremely": 3.0, "really": 1.5, "so": 1.5, "too": 1.5,
    "totally": 2.0, "absolutely": 2.5, "completely": 2.0, "incredibly": 2.5,
    "super": 2.0, "always": 1.25, "simply": 1.25, "truly": 1.5, "highly": 1.5,
    "quite": 1.25, "pretty": 1.25, "fairly": 0.75, "somewhat": 0.75,
    "rather": 0.8, "slightly": 0.5, "barely": 0.5, "hardly": 0.5, "little": 0.6,
    "bit": 0.6,
}
NEO_NEGATORS: Set[str] = {"not", "n't", "never", "no", "nothing", "nobody", "none", "neither", "nor", "without"}

# Negação assimétrica de Sugeno: S_l(x) = x / (1 + l * (1 - x)).
# Negar um positivo ("not good") atenua pouco; negar um negativo ("not bad") atenua mais.
NEO_SUGENO_LAMBDA_POS = 1.0
NEO_SUGENO_LAMBDA_NEG = 3.0
NEO_SUGENO_LAMBDA_EMOTION = 1.0

# Peso contextual dos sentidos: w_i ∝ (1/i) * exp(cos(frase, glosa_i) / tau)
NEO_SENSE_TEMPERATURE = 0.05

# Fallback para palavras fora do SentiWordNet (gírias): p = clip(k * (cos_pos - cos_neg))
NEO_OOV_SCALE = 2.0
NEO_OOV_MIN_MARGIN = 0.05
NEO_OOV_POSITIVE = "good, great, happy, love, awesome, wonderful, excellent, fun"
NEO_OOV_NEGATIVE = "bad, terrible, sad, hate, awful, horrible, angry, sick"

# Tabela emocional: membership = clip((cos - theta) / (1 - theta), 0, 1)
NEO_EMOTION_THETA = 0.3

# IOWA com quantificador RIM Q(r) = r ** beta (beta < 1 enfatiza as opiniões mais fortes)
NEO_OWA_BETA = 0.5

# T-conorma de Yager para união das emoções: min(1, (a^w + b^w)^(1/w))
NEO_YAGER_W = 3.0
