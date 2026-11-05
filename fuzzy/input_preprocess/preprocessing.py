import nltk
from fuzzy.input_preprocess.nltk_config import start_nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet, sentiwordnet
from nltk.stem import WordNetLemmatizer
from typing import Tuple

start_nltk()

AFFECTIVE_NOUNS = {
    "happy": "happiness.n.02",
    "angry": "anger.n.01",
    "sad": "sadness.n.01",
    "afraid": "fear.n.01",
}

class PreProcessor:
    def __init__(self):
        self.__lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> list[list[Tuple[str, str]]]:
        sentences = []

        for sentence in sent_tokenize(text):
            tokens = self.__tokenize(sentence)
            tags = self.__pos_tagging(tokens)
            filtered = self.__stop_words(tags)
            lemmas = self.__lemmatization(filtered)
            sentences.append(self.__select_synsets(lemmas))

        return sentences
        


    def __tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)
        
    def __pos_tagging(self, tokens: list[str]):
        return pos_tag(tokens)

    def __stop_words(self, tags: list[Tuple[str, str]]) -> list[Tuple[str, str]]:
        keep = {
            "not", "no", "nor", "never", "very", "too", "really",
            "be", "am", "is", "are", "was", "were", "been", "being",
        }
        stop_en = set(stopwords.words('english')) - keep
        result = []
        for word, tag in tags:
            word = "not" if word.lower() == "n't" else word
            if word.isalpha() and word.lower() not in stop_en:
                result.append((word, tag))
    
        return result

    def __to_wordnet_pos(self, tag: str) -> str | None:
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        
        return None
        

    def __lemmatization(self, tags: list[Tuple[str, str]]) -> list[Tuple[str, str]]:
        lemmas = []
        for word, tag in tags:
            wn_pos = self.__to_wordnet_pos(tag)
            if wn_pos is None: continue
            lemma = self.__lemmatizer.lemmatize(word.lower(), pos=wn_pos)
            lemmas.append((lemma, tag))

        return lemmas
    
    def __select_noun_synset(self, noun: str, other_nouns: list[str]):
        candidates = wordnet.synsets(noun, pos=wordnet.NOUN)
        if not candidates:
            return None
        if not other_nouns:
            return candidates[0]

        def score(candidate):
            total = 0.0
            for other in other_nouns:
                other_synsets = wordnet.synsets(other, pos=wordnet.NOUN)
                total += max(
                    (candidate.wup_similarity(s) or 0.0 for s in other_synsets),
                    default=0.0,
                )
            return total

        return max(candidates, key=score)
    
    def __select_synsets(self, lemmas: list[Tuple[str, str]]):
        nouns = [lemma for lemma, tag in lemmas if tag.startswith("N")]
        result = []

        for position, (lemma, tag) in enumerate(lemmas):
            if tag.startswith("N"):
                noun_index = sum(
                    1 for previous, previous_tag in lemmas[:position]
                    if previous_tag.startswith("N")
                )
                other_nouns = nouns[:noun_index] + nouns[noun_index + 1:]
                synset = self.__select_noun_synset(lemma, other_nouns)
            elif lemma in AFFECTIVE_NOUNS:
                synset = wordnet.synset(AFFECTIVE_NOUNS[lemma])
            else:
                synset = None

            result.append((lemma, tag, synset))

        return result

    
    def __wordnet(self, text: str):
        pass

    def __senti_wordnet(self, text: str):
        pass

    def __wu_palmer(self, text: str):
        pass
        
