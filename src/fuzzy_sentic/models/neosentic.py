from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import spacy
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

from src.fuzzy_sentic.config import (
    DIMENSIONS,
    HOURGLASS_SYNSETS,
    NEO_EMBEDDING_MODEL,
    NEO_EMOTION_THETA,
    NEO_HEDGES,
    NEO_NEGATORS,
    NEO_OOV_MIN_MARGIN,
    NEO_OOV_NEGATIVE,
    NEO_OOV_POSITIVE,
    NEO_OOV_SCALE,
    NEO_OWA_BETA,
    NEO_SENSE_TEMPERATURE,
    NEO_SPACY_MODEL,
    NEO_SUGENO_LAMBDA_EMOTION,
    NEO_SUGENO_LAMBDA_NEG,
    NEO_SUGENO_LAMBDA_POS,
    NEO_YAGER_W,
    OPPOSITE_PAIRS,
    POLARITY_ANCHORED_DIMENSIONS,
    NEO_ADVERSATIVES,
    NEO_CONCESSIVE_WEIGHT
)
from src.fuzzy_sentic.interfaces import SenticPipelineInterface
from src.fuzzy_sentic.text_utils import ensure_nltk_resources, normalize_tweet

_CONTENT_POS = {"ADJ": wn.ADJ, "VERB": wn.VERB, "NOUN": wn.NOUN, "ADV": wn.ADV, "INTJ": None}

# A negação de um verbo/cópula se estende aos seus complementos ("is not good")
_NEGATION_SCOPE_DEPS = {"acomp", "xcomp", "ccomp", "attr", "dobj", "oprd"}
_MODIFIER_DEPS = {"advmod", "npadvmod", "neg", "det", "preconj"}


def sugeno_attenuation(x: Any, lam: float) -> Any:
    """S_l(x) = N_l(1 - x) = x / (1 + l (1 - x)): complemento de Sugeno do complemento."""
    return x / (1.0 + lam * (1.0 - x))


def owa_weights(n: int, beta: float = NEO_OWA_BETA) -> np.ndarray:
    r = np.arange(n + 1) / n
    return np.diff(r ** beta)


def iowa(values: Iterable[float], beta: float = NEO_OWA_BETA) -> float:
    """IOWA: ordena pelo módulo (força da opinião) e agrega com os pesos OWA."""
    values = [v for v in values if not np.isclose(v, 0.0)]
    if not values:
        return 0.0
    ordered = sorted(values, key=abs, reverse=True)
    return float(np.dot(owa_weights(len(ordered), beta), ordered))


def yager_union(a: np.ndarray, b: np.ndarray, w: float = NEO_YAGER_W) -> np.ndarray:
    return np.minimum(1.0, (a ** w + b ** w) ** (1.0 / w))


class NeoSenticFuzzy(SenticPipelineInterface):
    def __init__(self, alpha_cut: float = 0.5, device: str = "cpu") -> None:
        super().__init__(alpha_cut=alpha_cut)
        ensure_nltk_resources()
        self.nlp = spacy.load(NEO_SPACY_MODEL, disable=["ner"])
        self.encoder = SentenceTransformer(NEO_EMBEDDING_MODEL, device=device)
        self._cache: Dict[str, np.ndarray] = {}

        labels = [wn.synset(name) for dim in DIMENSIONS for name in HOURGLASS_SYNSETS[dim]]
        label_texts = [f"{s.lemmas()[0].name()}: {s.definition()}" for s in labels]
        self._labels = self._embed(label_texts)
        self._prototypes = self._embed([NEO_OOV_POSITIVE, NEO_OOV_NEGATIVE])

    def _embed(self, texts: List[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            vectors = self.encoder.encode(
                missing, batch_size=128, normalize_embeddings=True, show_progress_bar=False
            )
            self._cache.update(zip(missing, vectors))
        return np.array([self._cache[t] for t in texts])

    @staticmethod
    def _modifier_of(child: Any) -> Optional[Tuple[str, float]]:
        lemma = child.lemma_.lower()
        if child.dep_ == "neg" or (child.dep_ in _MODIFIER_DEPS and lemma in NEO_NEGATORS):
            return ("neg", 0.0)
        if child.dep_ in ("advmod", "npadvmod") and lemma in NEO_HEDGES:
            return ("hedge", NEO_HEDGES[lemma])
        return None

    def _collect_modifiers(self, token: Any) -> List[Tuple[str, float]]:
        """Modificadores do token, do mais próximo ao mais distante (ordem de aplicação)."""
        local = []
        for child in token.children:
            modifier = self._modifier_of(child)
            if modifier is not None:
                local.append((abs(child.i - token.i), modifier))
                if modifier[0] == "hedge":
                    for grandchild in child.children:
                        gc_mod = self._modifier_of(grandchild)
                        if gc_mod == ("neg", 0.0):
                            local.append((abs(grandchild.i - token.i), gc_mod))

        modifiers = [m for _, m in sorted(local, key=lambda item: item[0])]

        node = token
        while node.dep_ in _NEGATION_SCOPE_DEPS and node.head is not node:
            node = node.head
            if any(self._modifier_of(c) == ("neg", 0.0) for c in node.children):
                modifiers.append(("neg", 0.0))
        return modifiers

    def _concessive_token_indices(self, span: Any) -> set[int]:
        """
        Identifica os índices (tok.i) dos tokens pertencentes a orações
        concedidas via estrutura de dependências (advcl/mark e conj/cc).
        """
        concessive: set[int] = set()

        for tok in span:
            # Caso 1: Oração subordinada concessiva ("Although X, Y" ou "Y, although X")
            if tok.dep_ == "advcl" and any(
                c.dep_ == "mark" and c.lemma_.lower() in NEO_ADVERSATIVES
                for c in tok.children
            ):
                concessive.update(t.i for t in tok.subtree)

            # Caso 2: Oração coordenada adversativa ("X, but Y" ou "X, however Y")
            elif tok.dep_ in {"conj", "parataxis"} and tok.head is not tok:
                has_head_cc = any(
                    c.dep_ == "cc" and c.lemma_.lower() in NEO_ADVERSATIVES
                    for c in tok.head.children
                )
                has_local_adv = any(
                    c.dep_ in {"cc", "advmod"} and c.lemma_.lower() in NEO_ADVERSATIVES
                    for c in tok.children
                )
                if has_head_cc or has_local_adv:
                    excluded = {t.i for t in tok.subtree}
                    concessive.update(
                        t.i for t in tok.head.subtree if t.i not in excluded
                    )

        return concessive

    def _units_from_span(self, span: Any) -> List[Dict[str, Any]]:
        concessive_ids = self._concessive_token_indices(span)
        units = []

        for token in span:
            lemma = token.lemma_.lower()
            if token.pos_ not in _CONTENT_POS or not token.is_alpha:
                continue
            if self._modifier_of(token) is not None or lemma in NEO_NEGATORS:
                continue

            units.append({
                "word": token.lower_,
                "lemma": lemma,
                "pos": token.pos_,
                "wn_pos": _CONTENT_POS[token.pos_],
                "modifiers": self._collect_modifiers(token),
                "sentence": span.text,
                "weight": NEO_CONCESSIVE_WEIGHT if token.i in concessive_ids else 1.0,
            })

        return units

    def extract_syntactic_units(self, sentence: str) -> List[Dict[str, Any]]:
        return self._units_from_span(self.nlp(sentence))

    def _senses(self, unit: Dict[str, Any]) -> List[Any]:
        if unit["wn_pos"] is None:
            return []
        return list(swn.senti_synsets(unit["lemma"], unit["wn_pos"]))

    @staticmethod
    def _gloss(unit: Dict[str, Any], synset: Any) -> str:
        return f"{unit['lemma']}: {synset.definition()}"

    def compute_unit_polarity(self, unit: Dict[str, Any]) -> float:
        """
        Eq. (1) com o peso de cada sentido ajustado pelo contexto:
        w_i ∝ (1/i) · exp(cos(frase, glosa_i) / tau). Fora do SentiWordNet,
        a polaridade vem da proximidade com protótipos positivo/negativo.
        """
        senses = self._senses(unit)
        if not senses:
            unit["sense"] = None
            vector = self._embed([unit["word"]])[0]
            margin = float(self._prototypes[0] @ vector - self._prototypes[1] @ vector)
            if abs(margin) < NEO_OOV_MIN_MARGIN:
                return 0.0
            return float(np.clip(NEO_OOV_SCALE * margin, -1.0, 1.0))

        sentence = self._embed([unit["sentence"]])[0]
        glosses = self._embed([self._gloss(unit, s.synset) for s in senses])
        sims = glosses @ sentence
        logits = sims / NEO_SENSE_TEMPERATURE - np.log(np.arange(1, len(senses) + 1))
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()

        unit["sense"] = senses[int(np.argmax(weights))].synset
        scores = np.array([s.pos_score() - s.neg_score() for s in senses])
        return float(weights @ scores)

    def compute_emotional_similarity_table(self, unit: Dict[str, Any]) -> np.ndarray:
        sense = unit.get("sense")
        text = self._gloss(unit, sense) if sense is not None else unit["word"]
        sims = self._labels @ self._embed([text])[0]
        table = np.clip((sims - NEO_EMOTION_THETA) / (1.0 - NEO_EMOTION_THETA), 0.0, 1.0)
        table = table.reshape(4, 6).T.copy()

        polarity = unit.get("polarity", 0.0)
        for col, dim in enumerate(DIMENSIONS):
            for pos_idx, neg_idx in OPPOSITE_PAIRS:
                v_pos, v_neg = table[pos_idx, col], table[neg_idx, col]
                if dim in POLARITY_ANCHORED_DIMENSIONS and not np.isclose(polarity, 0.0):
                    anchor_is_pos = polarity > 0
                else:
                    anchor_is_pos = v_pos >= v_neg
                # Emoções opostas não coexistem: o lado que não é âncora vai a 0
                table[neg_idx if anchor_is_pos else pos_idx, col] = 0.0
        return table

    def apply_fuzzy_modifiers(
        self,
        polarity: float,
        emo_table: np.ndarray,
        modifiers: List[Tuple[str, float]],
    ) -> Tuple[float, np.ndarray]:
        emo_table = emo_table.copy()
        for kind, alpha in modifiers:
            sign = np.sign(polarity)
            strength = abs(polarity)
            if kind == "hedge":
                polarity = float(sign * strength ** (1.0 / alpha))
                active = emo_table > 0
                emo_table[active] = emo_table[active] ** (1.0 / alpha)
            else:
                lam = NEO_SUGENO_LAMBDA_POS if polarity > 0 else NEO_SUGENO_LAMBDA_NEG
                polarity = float(-sign * sugeno_attenuation(strength, lam))
                negated = np.zeros_like(emo_table)
                for pos_idx, neg_idx in OPPOSITE_PAIRS:
                    negated[pos_idx] = sugeno_attenuation(emo_table[neg_idx], NEO_SUGENO_LAMBDA_EMOTION)
                    negated[neg_idx] = sugeno_attenuation(emo_table[pos_idx], NEO_SUGENO_LAMBDA_EMOTION)
                emo_table = negated
        return polarity, emo_table

    def _union(self, tables: Iterable[np.ndarray]) -> np.ndarray:
        result = np.zeros((6, 4))
        for table in tables:
            result = yager_union(result, table)
        return result

    def aggregate_phrase(
        self, units: List[Dict[str, Any]]
    ) -> Tuple[float, np.ndarray]:
        polarity = iowa(u["polarity"] for u in units)
        emo = self._union(u["emo_table"] for u in units)
        return polarity, np.where(emo >= self.alpha_cut, emo, 0.0)

    def _analyze_units(self, units: List[Dict[str, Any]]) -> None:
        texts = [u["sentence"] for u in units]
        for u in units:
            texts += [self._gloss(u, s.synset) for s in self._senses(u)] or [u["word"]]
        self._embed(texts)

        for u in units:
            u["polarity"] = self.compute_unit_polarity(u)
        sense_texts = [
            self._gloss(u, u["sense"]) if u["sense"] is not None else u["word"] for u in units
        ]
        self._embed(sense_texts)
        for u in units:
            emo = self.compute_emotional_similarity_table(u)
            pol, emo = self.apply_fuzzy_modifiers(
                u["polarity"], emo, u["modifiers"]
            )
            w = u.get("weight", 1.0)
            u["polarity"] = pol * w
            u["emo_table"] = emo * w

    def analyze_text(self, text: str) -> Tuple[float, np.ndarray]:
        doc = self.nlp(normalize_tweet(text))
        sentences = [self._units_from_span(sent) for sent in doc.sents]
        self._analyze_units([u for units in sentences for u in units])

        results = [self.aggregate_phrase(units) for units in sentences if units]
        total_polarity = iowa(p for p, _ in results)
        return total_polarity, self._union(e for _, e in results)
