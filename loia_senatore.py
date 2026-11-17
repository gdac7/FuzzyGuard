import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Baixa os recursos na primeira execução
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 
                 'averaged_perceptron_tagger_eng', 'wordnet', 'sentiwordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)


# ==============================================================================
# SEÇÃO 4: Linguistic and emotional resources
# ==============================================================================

# Tabela 1 e Tabela 2:
HOURGLASS_SYNSETS = {
    "Pleasantness": ["ecstasy.n.01", "joy.n.01", "serenity.n.01", 
                     "pensiveness.n.02", "sadness.n.02", "grief.n.01"],
    "Attention":    ["vigilance.n.01", "anticipation.n.01", "interest.n.01", 
                     "distraction.n.01", "surprise.n.01", "amazement.n.01"],
    "Sensitivity":  ["rage.n.01", "anger.n.01", "annoyance.n.01", 
                     "apprehension.n.01", "fear.n.01", "terror.n.01"],
    "Aptitude":     ["admiration.n.01", "trust.n.01", "acceptance.n.01", 
                     "boredom.n.01", "disgust.n.01", "loathing.n.01"]
}
DIMENSIONS = list(HOURGLASS_SYNSETS.keys())

INTENSIFIERS = {"very", "extremely", "really", "too", "always", "simply", "so"}
NEGATORS = {"not", "never", "no"}


class LoiaSenatoreBaseline:
    def __init__(self, alpha_cut=0.5):
        self.lemmatizer = WordNetLemmatizer()
        self.alpha_cut = alpha_cut

    def _map_pos_to_wn(self, tag):
        """Converte etiquetas Penn Treebank para categorias do WordNet."""
        if tag.startswith('NN'):
            return wn.NOUN
        elif tag.startswith('VB'):
            return wn.VERB
        elif tag.startswith('JJ'):
            return wn.ADJ
        elif tag.startswith('RB'):
            return wn.ADV
        return None

    # ==========================================================================
    # SEÇÃO 5.1: POS Tagging and filtering
    # ==========================================================================
    def phase1_pos_tagging_and_filtering(self, sentence):
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
                    "wn_pos": wn_pos
                })
        return filtered_words

    def _substantivize_synset(self, token_info):
        """
        Na Seção 5.1, a similaridade no WordNet só é calculada entre substantivos.
        Para adjetivos/verbos/advérbios, busca-se um substantivo derivado relacionado.
        """
        synsets = wn.synsets(token_info["lemma"], pos=token_info["wn_pos"])
        if not synsets:
            return None
        
        primary_synset = synsets[0]
        if token_info["wn_pos"] == wn.NOUN:
            return primary_synset
        
        # Tenta derivar a forma substantiva (ex: happy -> happiness)
        for lemma in primary_synset.lemmas():
            for related in lemma.derivationally_related_forms():
                if related.synset().pos() == wn.NOUN:
                    return related.synset()
        return primary_synset

    # ==========================================================================
    # SEÇÃO 5.2: Term Analysis
    # ==========================================================================
    def compute_word_polarity(self, token_info):
        """Calcula a polaridade p_w via SentiWordNet usando a Equação (1)."""
        senti_synsets = list(swn.senti_synsets(token_info["lemma"], token_info["wn_pos"]))
        if not senti_synsets:
            return 0.0
        
        numerator = 0.0
        denominator = 0.0
        for i, ssyn in enumerate(senti_synsets, start=1):
            # Polaridade líquida do sentido i (positiva - negativa)
            p_i = ssyn.pos_score() - ssyn.neg_score()
            numerator += (p_i / i)
            denominator += (1.0 / i)
            
        return numerator / denominator if denominator != 0 else 0.0

    def compute_emotional_similarity_table(self, token_info, polarity):
        """
        Calcula a similaridade de Wu & Palmer contra os 24 synsets da Hourglass
        e aplica a heurística de antônimos/simetria (Tabelas 3 e 4).
        """
        table = np.zeros((6, 4))  # 6 níveis sentic x 4 dimensões
        word_synset = self._substantivize_synset(token_info)

        if abs(polarity) < 0.01: 
            return table
        
        if word_synset is None or word_synset.pos() != wn.NOUN:
            return table

        for col_idx, dim in enumerate(DIMENSIONS):
            raw_sims = []
            for syn_name in HOURGLASS_SYNSETS[dim]:
                target_syn = wn.synset(syn_name)
                sim = word_synset.wup_similarity(target_syn)
                raw_sims.append(sim if sim is not None else 0.0)
            
            # Heurística de simetria em relação ao eixo Limbo (pares: 0-5, 1-4, 2-3)
            adjusted = np.array(raw_sims, dtype=float)
            for pos_idx, neg_idx in [(0, 5), (1, 4), (2, 3)]:
                v_pos, v_neg = adjusted[pos_idx], adjusted[neg_idx]
                
                if np.isclose(v_pos, v_neg):
                    adjusted[pos_idx] = 0.0
                    adjusted[neg_idx] = 0.0
                elif dim in ["Pleasantness", "Aptitude"]:
                    # Caso especial de antônimos diretos resolvido pela polaridade[cite: 6]
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
    def _triangular_mf_polarity(self, p):
        """Mapeia polaridade p in [-1, 1] para pertinência P in [0, 1] (-1, 1, 1)."""
        return np.clip((p + 1.0) / 2.0, 0.0, 1.0)

    def _fuzzy_intensifier(self, mu):
        """Equação (2): Modificador linguístico intensificador (very, extremely)."""
        return np.where(mu >= 0.5, np.sqrt(mu), np.square(mu))

    def _fuzzy_negator(self, mu):
        """Equação (3): Modificador linguístico negador (not, never)."""
        return 1.0 - mu

    def _fuzzy_combinator(self, mu):
        """Equação (4): Combinação Negador + Intensificador (not very)."""
        return np.sqrt(mu * self._fuzzy_intensifier(mu))

    def _apply_emotional_modifier(self, emo_table, modifier_type, threshold=0.25):
        """Aplica os modificadores seletivamente apenas às células ativas,
        evitando que células residuais/zeradas gerem ruído artificial.
        """
        active_mask = emo_table > threshold
        result = np.zeros_like(emo_table)
        
        if modifier_type == "intensifier":
            result[active_mask] = self._fuzzy_intensifier(emo_table[active_mask])
        elif modifier_type == "combinator":
            result[active_mask] = self._fuzzy_combinator(emo_table[active_mask])
        elif modifier_type == "negator":
            # Inverte a valência emocional entre os níveis positivos (0,1,2) e negativos (5,4,3)
            # mantendo a intensidade modulada pelo negador
            for col in range(4):
                for pos_idx, neg_idx in [(0, 5), (1, 4), (2, 3)]:
                    if emo_table[pos_idx, col] > threshold:
                        result[neg_idx, col] = emo_table[pos_idx, col]
                    if emo_table[neg_idx, col] > threshold:
                        result[pos_idx, col] = emo_table[neg_idx, col]
        return result

    def phase3_pattern_analysis(self, tokens_data):
        """
        Identifica padrões sintáticos (Tabela 5: ex. RB + RB + JJ, RB + JJ, JJ + NN)[cite: 6]
        e aplica os operadores nebulosos sobre polaridade e tabela emocional[cite: 6, 7].
        """
        i = 0
        n = len(tokens_data)
        units = []

        while i < n:
            w1 = tokens_data[i]["lemma"]
            # Padrão Combinado: Negador + Intensificador + Adjetivo/Advérbio (ex: "not very good")[cite: 6, 7]
            if i + 2 < n and w1 in NEGATORS and tokens_data[i+1]["lemma"] in INTENSIFIERS:
                target = tokens_data[i+2]
                mu_p = self._triangular_mf_polarity(target["polarity"])
                mod_p = self._fuzzy_combinator(mu_p)
                final_pol = (2.0 * mod_p) - 1.0
                final_emo = self._apply_emotional_modifier(target["emo_table"], "combinator")
                units.append({
                    "polarity": final_pol, 
                    "emo_table": final_emo, 
                    "label": f"{w1} {tokens_data[i+1]['word']} {target['word']}"
                })
                i += 3
            # Padrão 3.1 (Tabela 5): RB + JJ/RB (Intensificador ou Negador + Alvo)[cite: 6, 7]
            elif i + 1 < n and (w1 in INTENSIFIERS or w1 in NEGATORS):
                target = tokens_data[i+1]
                mu_p = self._triangular_mf_polarity(target["polarity"])
                if w1 in INTENSIFIERS:
                    mod_p = self._fuzzy_intensifier(mu_p)
                    final_emo = self._apply_emotional_modifier(target["emo_table"], "intensifier")
                else:
                    mod_p = self._fuzzy_negator(mu_p)
                    final_emo = self._apply_emotional_modifier(target["emo_table"], "negator")
                final_pol = (2.0 * mod_p) - 1.0
                units.append({
                    "polarity": final_pol, 
                    "emo_table": final_emo, 
                    "label": f"{w1} {target['word']}"
                })
                i += 2
            # Padrão 2 (Tabela 5): JJ + NN (União fuzzy entre adjetivo e substantivo)[cite: 6, 7]
            elif i + 1 < n and tokens_data[i]["wn_pos"] == wn.ADJ and tokens_data[i+1]["wn_pos"] == wn.NOUN:
                t1, t2 = tokens_data[i], tokens_data[i+1]
                final_pol = t1["polarity"] + t2["polarity"]
                final_emo = np.maximum(t1["emo_table"], t2["emo_table"])
                units.append({
                    "polarity": final_pol, 
                    "emo_table": final_emo, 
                    "label": f"{t1['word']} {t2['word']}"
                })
                i += 2
            else:
                units.append({
                    "polarity": tokens_data[i]["polarity"], 
                    "emo_table": tokens_data[i]["emo_table"], 
                    "label": tokens_data[i]["word"]
                })
                i += 1

        return units
    # ==========================================================================
    # SEÇÃO 5.4: Phrase Analysis
    # ==========================================================================
    def phase4_phrase_analysis(self, pattern_units):
        # Polaridade da frase: média aritmética das polaridades não-nulas
        non_zero_pols = [u["polarity"] for u in pattern_units if not np.isclose(u["polarity"], 0.0)]
        sentence_polarity = float(np.mean(non_zero_pols)) if non_zero_pols else 0.0

        # Emoção da frase: união fuzzy (máximo) seguida de alpha-cut
        sentence_emo = np.zeros((6, 4))
        for u in pattern_units:
            sentence_emo = np.maximum(sentence_emo, u["emo_table"])
        
        # Aplica o corte alfa (alpha-cut)
        sentence_emo = np.where(sentence_emo >= self.alpha_cut, sentence_emo, 0.0)
        return sentence_polarity, sentence_emo

    # ==========================================================================
    # SEÇÃO 5.5: Text Analysis
    # ==========================================================================
    def analyze_text(self, text):
        sentences = sent_tokenize(text)
        sentence_polarities = []
        text_emo_table = np.zeros((6, 4))

        for sent in sentences:
            # Executa Fases 1 a 4 por sentença
            filtered = self.phase1_pos_tagging_and_filtering(sent)
            for token in filtered:
                token["polarity"] = self.compute_word_polarity(token)
                token["emo_table"] = self.compute_emotional_similarity_table(token, token["polarity"])
            
            units = self.phase3_pattern_analysis(filtered)
            sent_pol, sent_emo = self.phase4_phrase_analysis(units)
            
            if not np.isclose(sent_pol, 0.0):
                sentence_polarities.append(sent_pol)
            
            # União fuzzy entre as tabelas emocionais de todas as sentenças
            text_emo_table = np.maximum(text_emo_table, sent_emo)

        # Equação (5): Média aritmética das polaridades das sentenças não-nulas
        total_polarity = float(np.mean(sentence_polarities)) if sentence_polarities else 0.0
        return total_polarity, text_emo_table


# ==============================================================================
# EXEMPLO DE EXECUÇÃO (USANDO A SENTENÇA DIDÁTICA DA SEÇÃO 5.1 DO ARTIGO)
# ==============================================================================
if __name__ == "__main__":
    pipeline = LoiaSenatoreBaseline(alpha_cut=0.5)
    
    sample_text = "He is happy to go home. However, the weather is not very good."
    polarity, emotion_matrix = pipeline.analyze_text(sample_text)

    print(f"Texto analisado: '{sample_text}'")
    print(f"Polaridade Global (Eq. 5): {polarity:.4f}\n")
    print("Matriz Emocional Final (6 Níveis Sentic x 4 Dimensões):")
    print(f"{'Nível':<8} | {'Pleasantness':<12} | {'Attention':<10} | {'Sensitivity':<11} | {'Aptitude':<10}")
    print("-" * 62)
    levels = ["+3", "+2", "+1", "-1", "-2", "-3"]
    for idx, lvl in enumerate(levels):
        row = emotion_matrix[idx]
        print(f"{lvl:<8} | {row[0]:<12.3f} | {row[1]:<10.3f} | {row[2]:<11.3f} | {row[3]:<10.3f}")