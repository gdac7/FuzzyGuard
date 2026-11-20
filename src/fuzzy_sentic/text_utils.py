import html
import re

import nltk

from src.fuzzy_sentic.config import EMOTICON_EMOTIONS

_NLTK_RESOURCES = {
    "tokenizers/punkt_tab": "punkt_tab",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
    "corpora/sentiwordnet": "sentiwordnet",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_SPACES_RE = re.compile(r"\s+")

_emoticons = sorted(EMOTICON_EMOTIONS, key=len, reverse=True)
_EMOTICON_RE = re.compile(
    r"(?<!\S)(" + "|".join(re.escape(e) for e in _emoticons) + r")(?!\S)",
    re.IGNORECASE,
)
_EMOTICON_LOOKUP = {e.lower(): emotion for e, emotion in EMOTICON_EMOTIONS.items()}


def ensure_nltk_resources() -> None:
    """Baixa os recursos do NLTK apenas quando ainda não estão instalados."""
    for path, package in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def normalize_tweet(text: str) -> str:
    """
    Limpeza compartilhada pelos dois modelos: entidades HTML, URLs, menções,
    '#' das hashtags e emoticons (trocados pelo nome da emoção).
    """
    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    # O emoticon vira uma frase própria para não entrar no escopo sintático de outra palavra
    text = _EMOTICON_RE.sub(lambda m: f". {_EMOTICON_LOOKUP[m.group(1).lower()]}. ", text)
    return _SPACES_RE.sub(" ", text).strip()
