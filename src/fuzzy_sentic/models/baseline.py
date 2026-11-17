from typing import Any, Dict, List, Optional, Tuple
import nltk
import re
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
)
from src.fuzzy_sentic.interfaces import SenticPipelineInterface


def _ensure_nltk_resources() -> None:
    """Garante o download silencioso dos léxicos necessários na primeira execução."""
    resources = [
        "punkt", "punkt_tab", "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng", "wordnet", "sentiwordnet", "omw-1.4"
    ]
    for resource in resources:
        nltk.download(resource, quiet=True)


class LoiaSenatoreBaseline(SenticPipelineInterface):
    """
    Implementação de referência do framework proposto por Loia & Senatore (2014).
    """

    def __init__(self, alpha_cut: float = 0.5) -> None:
        super().__init__(alpha_cut=alpha_cut)
        _ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _map_pos_to_wn(tag: str) -> Optional[str]:
        """Converte etiquetas Penn Treebank para categorias gramaticais do WordNet."""
        if tag.startswith("NN"):
            return wn.NOUN
        if tag.startswith("VB"):
            return wn.VERB
        if tag.startswith("JJ"):
            return wn.ADJ
        if tag.startswith("RB"):
            return wn.ADV
        return None

    # ==========================================================================
    # SEÇÃO 5.1: POS Tagging and Filtering
    # ==========================================================================
    def phase1_pos_tagging_and_filtering(self, sentence: str) -> List[Dict[str, Any]]:
        cleaned_sentence = re.sub(r"http\S+|www\.\S+|@\w+", "", sentence.lower())
        tokens = word_tokenize(sentence.lower())
        tagged = nltk.pos_tag(tokens)

        filtered_words = []
        for word, tag in tagged:
            wn_pos = self._map_pos_to_wn(tag)
            if wn_pos is not None:
                lemma = self.lemmatizer.lemmatize(word, pos=wn_pos)
                filtered_words.append({
                    "word": word,
                    "lemma": lemma,
                    "pos": tag,
                    "wn_pos": wn_pos,
                })
        return filtered_words

    @staticmethod
    def _substantivize_synset(token_info: Dict[str, Any]) -> Optional[Any]:
        """Deriva a forma substantiva de adjetivos/verbos/advérbios para cálculo no WordNet."""
        synsets = wn.synsets(token_info["lemma"], pos=token_info["wn_pos"])
        if not synsets:
            return None

        primary_synset = synsets[0]
        if token_info["wn_pos"] == wn.NOUN:
            return primary_synset

        for lemma in primary_synset.lemmas():
            for related in lemma.derivationally_related_forms():
                if related.synset().pos() == wn.NOUN:
                    return related.synset()
        return primary_synset

    # ==========================================================================
    # SEÇÃO 5.2: Term Analysis
    # ==========================================================================
    @staticmethod
    def compute_word_polarity(token_info: Dict[str, Any]) -> float:
        """Calcula a polaridade p_w via SentiWordNet usando a Equação (1)."""
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
        """Calcula a similaridade de Wu & Palmer contra os 24 synsets e aplica heurísticas."""
        table = np.zeros((6, 4))
        word_synset = self._substantivize_synset(token_info)

        if abs(polarity) < 0.01 or word_synset is None or word_synset.pos() != wn.NOUN:
            return table

        for col_idx, dim in enumerate(DIMENSIONS):
            raw_sims = []
            for syn_name in HOURGLASS_SYNSETS[dim]:
                target_syn = wn.synset(syn_name)
                sim = word_synset.wup_similarity(target_syn)
                raw_sims.append(sim if sim is not None else 0.0)

            adjusted = np.array(raw_sims, dtype=float)
            for pos_idx, neg_idx in [(0, 5), (1, 4), (2, 3)]:
                v_pos, v_neg = adjusted[pos_idx], adjusted[neg_idx]

                if np.isclose(v_pos, v_neg):
                    adjusted[pos_idx] = 0.0
                    adjusted[neg_idx] = 0.0
                elif dim in ["Pleasantness", "Aptitude"]:
                    if polarity > 0:
                        adjusted[neg_idx] = (1.0 - v_pos) if v_pos > 0.5 else 0.0
                    else:
                        adjusted[pos_idx] = (1.0 - v_neg) if v_neg > 0.5 else 0.0
                else:
                    if v_pos > v_neg:
                        adjusted[neg_idx] = (1.0 - v_pos) if v_pos > 0.5 else 0.0
                    else:
                        adjusted[pos_idx] = (1.0 - v_neg) if v_neg > 0.5 else 0.0

            table[:, col_idx] = adjusted
        return table

    # ==========================================================================
    # SEÇÃO 5.3: Pattern Analysis
    # ==========================================================================
    @staticmethod
    def _triangular_mf_polarity(p: float) -> float:
        return float(np.clip((p + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _fuzzy_intensifier(mu: Any) -> Any:
        """Equação (2): Modificador linguístico intensificador."""
        return np.where(mu >= 0.5, np.sqrt(mu), np.square(mu))

    @staticmethod
    def _fuzzy_negator(mu: Any) -> Any:
        """Equação (3): Modificador linguístico negador."""
        return 1.0 - mu

    def _fuzzy_combinator(self, mu: Any) -> Any:
        """Equação (4): Combinação Negador + Intensificador."""
        return np.sqrt(mu * self._fuzzy_intensifier(mu))

    def _apply_emotional_modifier(
        self, emo_table: np.ndarray, modifier_type: str, threshold: float = 0.25
    ) -> np.ndarray:
        active_mask = emo_table > threshold
        result = np.zeros_like(emo_table)

        if modifier_type == "intensifier":
            result[active_mask] = self._fuzzy_intensifier(emo_table[active_mask])
        elif modifier_type == "combinator":
            result[active_mask] = self._fuzzy_combinator(emo_table[active_mask])
        elif modifier_type == "negator":
            for col in range(4):
                for pos_idx, neg_idx in [(0, 5), (1, 4), (2, 3)]:
                    if emo_table[pos_idx, col] > threshold:
                        result[neg_idx, col] = emo_table[pos_idx, col]
                    if emo_table[neg_idx, col] > threshold:
                        result[pos_idx, col] = emo_table[neg_idx, col]
        return result

    def phase3_pattern_analysis(self, tokens_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        i = 0
        n = len(tokens_data)
        units = []

        while i < n:
            w1 = tokens_data[i]["lemma"]
            if (
                i + 2 < n
                and w1 in BASELINE_NEGATORS
                and tokens_data[i + 1]["lemma"] in BASELINE_INTENSIFIERS
            ):
                target = tokens_data[i + 2]
                mu_p = self._triangular_mf_polarity(target["polarity"])
                mod_p = self._fuzzy_combinator(mu_p)
                units.append({
                    "polarity": (2.0 * mod_p) - 1.0,
                    "emo_table": self._apply_emotional_modifier(target["emo_table"], "combinator"),
                    "label": f"{w1} {tokens_data[i+1]['word']} {target['word']}",
                })
                i += 3
            elif i + 1 < n and (w1 in BASELINE_INTENSIFIERS or w1 in BASELINE_NEGATORS):
                target = tokens_data[i + 1]
                mu_p = self._triangular_mf_polarity(target["polarity"])
                if w1 in BASELINE_INTENSIFIERS:
                    mod_p = self._fuzzy_intensifier(mu_p)
                    final_emo = self._apply_emotional_modifier(target["emo_table"], "intensifier")
                else:
                    mod_p = self._fuzzy_negator(mu_p)
                    final_emo = self._apply_emotional_modifier(target["emo_table"], "negator")
                units.append({
                    "polarity": (2.0 * mod_p) - 1.0,
                    "emo_table": final_emo,
                    "label": f"{w1} {target['word']}",
                })
                i += 2
            elif (
                i + 1 < n
                and tokens_data[i]["wn_pos"] == wn.ADJ
                and tokens_data[i + 1]["wn_pos"] == wn.NOUN
            ):
                t1, t2 = tokens_data[i], tokens_data[i + 1]
                units.append({
                    "polarity": t1["polarity"] + t2["polarity"],
                    "emo_table": np.maximum(t1["emo_table"], t2["emo_table"]),
                    "label": f"{t1['word']} {t2['word']}",
                })
                i += 2
            else:
                units.append({
                    "polarity": tokens_data[i]["polarity"],
                    "emo_table": tokens_data[i]["emo_table"],
                    "label": tokens_data[i]["word"],
                })
                i += 1

        return units

    # ==========================================================================
    # SEÇÃO 5.4: Phrase Analysis
    # ==========================================================================
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

    # ==========================================================================
    # SEÇÃO 5.5: Text Analysis
    # ==========================================================================
    def analyze_text(self, text: str) -> Tuple[float, np.ndarray]:
        sentences = sent_tokenize(text)
        sentence_polarities = []
        text_emo_table = np.zeros((6, 4))

        for sent in sentences:
            filtered = self.phase1_pos_tagging_and_filtering(sent)
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