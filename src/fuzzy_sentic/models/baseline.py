from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from src.fuzzy_sentic.config import (
    BASELINE_INTENSIFIERS,
    BASELINE_NEGATORS,
    DIMENSIONS,
    HOURGLASS_SYNSETS,
    OPPOSITE_PAIRS,
    POLARITY_ANCHORED_DIMENSIONS,
)
from src.fuzzy_sentic.interfaces import SenticPipelineInterface
from src.fuzzy_sentic.text_utils import ensure_nltk_resources, normalize_tweet

_CONTRACTIONS = {"'s": "be", "'m": "be", "'re": "be", "'ve": "have", "n't": "not"}

# "BE" é um verbo cujo lema é "be"; "JJ|RB" aceita adjetivo ou advérbio.
_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("1", ("NN", "BE", "JJ")),
    ("2", ("JJ", "NN", "NN")),
    ("3", ("RB", "JJ|RB", "NN")),
    ("6", ("RB", "RB", "JJ")),
    ("2.1", ("JJ", "NN")),
    ("3.1", ("RB", "JJ|RB")),
    ("4", ("RB", "VB")),
    ("5", ("VB", "NN")),
]

_HOURGLASS = [[wn.synset(name) for name in HOURGLASS_SYNSETS[dim]] for dim in DIMENSIONS]
_FEELING = wn.synset("feeling.n.01")


@lru_cache(maxsize=None)
def wup_jwsl(a: Any, b: Any) -> float:
    """
    Similaridade de Wu & Palmer como calculada pela JWSL usada no artigo.
    Igual à do NLTK com a profundidade do LCS deslocada em +2
    (reproduz a Tabela 3 com erro médio de 0.006).
    """
    lcs = a.lowest_common_hypernyms(b)
    if not lcs:
        return 0.0
    depth = lcs[0].max_depth() + 2
    d1 = a.shortest_path_distance(lcs[0])
    d2 = b.shortest_path_distance(lcs[0])
    return 2.0 * depth / (d1 + d2 + 2.0 * depth)


@lru_cache(maxsize=None)
def _is_affective(synset: Any) -> bool:
    return _FEELING in synset.closure(lambda s: s.hypernyms())


@lru_cache(maxsize=None)
def affective_categories(lemma: str, wn_pos: str) -> Tuple[Any, ...]:
    """
    Substitui o WordNet-Affect: substantivos afetivos (descendentes de
    feeling.n.01) ligados à palavra por derivação, atributo ou pertainym,
    na ordem dos sentidos do WordNet.
    """
    found: List[Any] = []
    for synset in wn.synsets(lemma, pos=wn_pos):
        lemmas = list(synset.lemmas())
        if synset.pos() == wn.ADV:
            lemmas = [p for l in synset.lemmas() for p in l.pertainyms()] + lemmas
        related = [
            r.synset() for l in lemmas for r in l.derivationally_related_forms()
            if r.synset().pos() == wn.NOUN
        ]
        related += [a for a in synset.attributes() if a.pos() == wn.NOUN]
        if synset.pos() == wn.ADJ_SAT:
            for head in synset.similar_tos():
                related += [
                    r.synset() for l in head.lemmas() for r in l.derivationally_related_forms()
                    if r.synset().pos() == wn.NOUN
                ]
        for noun in related:
            if noun not in found and _is_affective(noun):
                found.append(noun)
    return tuple(found)


def _raw_similarity_table(synset: Any) -> np.ndarray:
    table = np.zeros((6, 4))
    for col, labels in enumerate(_HOURGLASS):
        table[:, col] = [wup_jwsl(synset, label) for label in labels]
    return table


@lru_cache(maxsize=None)
def emotional_similarity_table(synset: Any, polarity_sign: int) -> np.ndarray:
    """
    Tabela de similaridade emocional.

    - Palavra sem afinidade emocional (nenhuma similaridade >= 0.5, ex. "home")
      mantém a tabela crua (Tabela 7).
    - Opostos com similaridade idêntica vão a 0.
    - Âncora: em Pleasantness/Aptitude, o rótulo do lado da polaridade da palavra;
      nas demais, o de maior similaridade. A âncora recebe max(v, 1 - v) e o oposto
      o complemento. Essa regra reproduz a Tabela 4 célula a célula.
    """
    raw = _raw_similarity_table(synset)
    if raw.max() < 0.5:
        raw.flags.writeable = False
        return raw

    table = raw.copy()
    for col, dim in enumerate(DIMENSIONS):
        for pos_idx, neg_idx in OPPOSITE_PAIRS:
            v_pos, v_neg = raw[pos_idx, col], raw[neg_idx, col]
            if np.isclose(v_pos, v_neg):
                table[pos_idx, col] = table[neg_idx, col] = 0.0
                continue
            if dim in POLARITY_ANCHORED_DIMENSIONS and polarity_sign != 0:
                anchor_is_pos = polarity_sign > 0
            else:
                anchor_is_pos = v_pos > v_neg
            anchor, opposite = (pos_idx, neg_idx) if anchor_is_pos else (neg_idx, pos_idx)
            value = max(raw[anchor, col], 1.0 - raw[anchor, col])
            table[anchor, col] = value
            table[opposite, col] = 1.0 - value
    table.flags.writeable = False
    return table


class LoiaSenatoreBaseline(SenticPipelineInterface):
    def __init__(self, alpha_cut: float = 0.5) -> None:
        super().__init__(alpha_cut=alpha_cut)
        ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _map_pos_to_wn(tag: str) -> Optional[str]:
        if tag.startswith("NN"):
            return wn.NOUN
        if tag.startswith("VB"):
            return wn.VERB
        if tag.startswith("JJ"):
            return wn.ADJ
        if tag.startswith("RB"):
            return wn.ADV
        return None

    def phase1_pos_tagging_and_filtering(self, sentence: str) -> List[Dict[str, Any]]:
        # O tagger recebe a caixa original ("I" em minúsculo vira substantivo)
        tagged = nltk.pos_tag(word_tokenize(sentence))

        filtered_words = []
        for word, tag in tagged:
            wn_pos = self._map_pos_to_wn(tag)
            lower = word.lower()
            if wn_pos is None or not any(c.isalpha() for c in lower):
                continue
            lemma = _CONTRACTIONS.get(lower) or self.lemmatizer.lemmatize(lower, pos=wn_pos)
            filtered_words.append({
                "word": lower,
                "lemma": lemma,
                "pos": tag,
                "wn_pos": wn_pos,
            })
        return filtered_words

    def _select_noun_synset(self, lemma: str, other_nouns: List[str]) -> Optional[Any]:
        candidates = wn.synsets(lemma, pos=wn.NOUN)
        if not candidates:
            return None
        others = [wn.synsets(n, pos=wn.NOUN) for n in other_nouns]
        others = [o for o in others if o]
        if not others:
            return candidates[0]

        def score(candidate: Any) -> float:
            return sum(max(wup_jwsl(candidate, s) for s in synsets) for synsets in others)

        return max(candidates, key=score)

    def _substantivize(self, index: int, tokens: List[Dict[str, Any]]) -> Optional[Any]:
        """
        Adjetivos, verbos e advérbios recebem a categoria afetiva mais similar
        ao substantivo que os segue; sem substantivo, a primeira.
        Sem categoria afetiva, a palavra fica sem emoção (como "be" e "go").
        """
        token = tokens[index]
        candidates = affective_categories(token["lemma"], token["wn_pos"])
        if not candidates:
            return None
        if index + 1 < len(tokens) and tokens[index + 1].get("synset") is not None \
                and tokens[index + 1]["wn_pos"] == wn.NOUN:
            following = tokens[index + 1]["synset"]
            return max(candidates, key=lambda c: wup_jwsl(c, following))
        return candidates[0]

    def assign_synsets(self, tokens: List[Dict[str, Any]]) -> None:
        nouns = [t["lemma"] for t in tokens if t["wn_pos"] == wn.NOUN]
        noun_index = 0
        for token in tokens:
            if token["wn_pos"] == wn.NOUN:
                others = nouns[:noun_index] + nouns[noun_index + 1:]
                token["synset"] = self._select_noun_synset(token["lemma"], others)
                noun_index += 1
        for i, token in enumerate(tokens):
            if token["wn_pos"] != wn.NOUN:
                token["synset"] = self._substantivize(i, tokens)

    @staticmethod
    def compute_word_polarity(token_info: Dict[str, Any]) -> float:
        senti_synsets = list(swn.senti_synsets(token_info["lemma"], token_info["wn_pos"]))
        if not senti_synsets:
            return 0.0

        numerator = 0.0
        denominator = 0.0
        for i, ssyn in enumerate(senti_synsets, start=1):
            p_i = ssyn.pos_score() - ssyn.neg_score()
            numerator += p_i / i
            denominator += 1.0 / i

        return numerator / denominator if denominator != 0 else 0.0

    def compute_emotional_similarity_table(
        self, token_info: Dict[str, Any], polarity: float
    ) -> np.ndarray:
        synset = token_info.get("synset")
        if synset is None:
            return np.zeros((6, 4))
        return emotional_similarity_table(synset, int(np.sign(polarity))).copy()

    @staticmethod
    def _triangular_mf_polarity(p: float) -> float:
        return float(np.clip((p + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _fuzzy_intensifier(mu: Any) -> Any:
        return np.where(mu >= 0.5, np.sqrt(mu), np.square(mu))

    @staticmethod
    def _fuzzy_negator(mu: Any) -> Any:
        return 1.0 - mu

    def _fuzzy_combinator(self, mu: Any) -> Any:
        return np.sqrt(mu * self._fuzzy_intensifier(mu))

    def _modify(self, mu: Any, modifiers: List[str]) -> Any:
        """
        Aplica a cadeia de modificadores (do mais distante ao mais próximo do alvo).
        "Negador + intensificador" vira o combinador da Eq. (4).
        """
        if modifiers == ["negator", "intensifier"]:
            return self._fuzzy_combinator(mu)
        for modifier in reversed(modifiers):
            if modifier == "intensifier":
                mu = self._fuzzy_intensifier(mu)
            else:
                mu = self._fuzzy_negator(mu)
        return mu

    def _modify_polarity(self, polarity: float, modifiers: List[str]) -> float:
        mu = self._triangular_mf_polarity(polarity)
        return float(2.0 * self._modify(mu, modifiers) - 1.0)

    def _modify_emotions(self, emo_table: np.ndarray, modifiers: List[str]) -> np.ndarray:
        """Células zeradas representam emoção ausente e não são modificadas."""
        active = emo_table > 0
        result = emo_table.copy()
        result[active] = self._modify(emo_table[active], modifiers)
        return result

    @staticmethod
    def _modifier_type(token: Dict[str, Any]) -> Optional[str]:
        if not token["pos"].startswith("RB"):
            return None
        if token["lemma"] in BASELINE_NEGATORS:
            return "negator"
        if token["lemma"] in BASELINE_INTENSIFIERS:
            return "intensifier"
        return None

    @staticmethod
    def _matches(token: Dict[str, Any], tag_class: str) -> bool:
        if tag_class == "BE":
            return token["pos"].startswith("VB") and token["lemma"] == "be"
        return any(token["pos"].startswith(option) for option in tag_class.split("|"))

    def _match_pattern(self, tokens: List[Dict[str, Any]], i: int) -> Optional[Tuple[str, int]]:
        for name, classes in _PATTERNS:
            end = i + len(classes)
            if end <= len(tokens) and all(
                self._matches(tokens[i + k], c) for k, c in enumerate(classes)
            ):
                return name, len(classes)
        return None

    def _pattern_unit(self, words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Polaridade do padrão = soma das polaridades das palavras; um advérbio
        modificador não soma, ele modifica o adjetivo/advérbio/verbo seguinte.
        A tabela emocional é a união fuzzy (máximo) das tabelas das palavras.
        """
        modifiers: List[str] = []
        polarity = 0.0
        emo_table = np.zeros((6, 4))
        i = 0
        while i < len(words):
            kind = self._modifier_type(words[i])
            has_target = i + 1 < len(words) and words[i + 1]["pos"][:2] in ("JJ", "RB", "VB")
            if kind is not None and has_target:
                modifiers.append(kind)
                i += 1
                continue
            word = words[i]
            word_pol, word_emo = word["polarity"], word["emo_table"]
            if modifiers:
                word_pol = self._modify_polarity(word_pol, modifiers)
                word_emo = self._modify_emotions(word_emo, modifiers)
                modifiers = []
            polarity += word_pol
            emo_table = np.maximum(emo_table, word_emo)
            i += 1

        return {
            "polarity": float(np.clip(polarity, -1.0, 1.0)),
            "emo_table": emo_table,
            "label": " ".join(w["word"] for w in words),
        }

    def phase3_pattern_analysis(self, tokens_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        units = []
        i = 0
        while i < len(tokens_data):
            match = self._match_pattern(tokens_data, i)
            length = match[1] if match else 1
            units.append(self._pattern_unit(tokens_data[i:i + length]))
            i += length
        return units

    def phase4_phrase_analysis(
        self, pattern_units: List[Dict[str, Any]]
    ) -> Tuple[float, np.ndarray]:
        non_zero_pols = [
            u["polarity"] for u in pattern_units if not np.isclose(u["polarity"], 0.0)
        ]
        sentence_polarity = float(np.mean(non_zero_pols)) if non_zero_pols else 0.0

        sentence_emo = np.zeros((6, 4))
        for u in pattern_units:
            sentence_emo = np.maximum(sentence_emo, u["emo_table"])

        sentence_emo = np.where(sentence_emo >= self.alpha_cut, sentence_emo, 0.0)
        return sentence_polarity, sentence_emo

    def analyze_text(self, text: str) -> Tuple[float, np.ndarray]:
        sentences = sent_tokenize(normalize_tweet(text))
        sentence_polarities = []
        text_emo_table = np.zeros((6, 4))

        for sent in sentences:
            filtered = self.phase1_pos_tagging_and_filtering(sent)
            self.assign_synsets(filtered)
            for token in filtered:
                token["polarity"] = self.compute_word_polarity(token)
                token["emo_table"] = self.compute_emotional_similarity_table(
                    token, token["polarity"]
                )

            units = self.phase3_pattern_analysis(filtered)
            sent_pol, sent_emo = self.phase4_phrase_analysis(units)

            if not np.isclose(sent_pol, 0.0):
                sentence_polarities.append(sent_pol)

            text_emo_table = np.maximum(text_emo_table, sent_emo)

        total_polarity = float(np.mean(sentence_polarities)) if sentence_polarities else 0.0
        return total_polarity, text_emo_table
