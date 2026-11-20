import numpy as np
import pytest
from nltk.corpus import wordnet as wn

from src.fuzzy_sentic.models.baseline import (
    LoiaSenatoreBaseline,
    _raw_similarity_table,
    emotional_similarity_table,
)
from src.fuzzy_sentic.text_utils import normalize_tweet

# Valores publicados no artigo (linhas: +3..-3; colunas: Pleasantness, Attention, Sensitivity, Aptitude)
TABLE_3 = np.array([
    [0.89, 0.33, 0.77, 0.70],
    [0.82, 0.70, 0.82, 0.37],
    [0.44, 0.52, 0.58, 0.37],
    [0.50, 0.52, 0.77, 0.60],
    [0.84, 0.70, 0.82, 0.70],
    [0.66, 0.75, 0.77, 0.77],
])
TABLE_4 = np.array([
    [0.89, 0.25, 0.00, 0.70],
    [0.82, 0.00, 0.00, 0.62],
    [0.55, 0.00, 0.22, 0.62],
    [0.44, 0.00, 0.77, 0.37],
    [0.18, 0.00, 0.00, 0.37],
    [0.11, 0.75, 0.00, 0.29],
])
TABLE_7_HAPPY = np.array([
    [0.75, 0.14, 0.00, 0.80],
    [0.80, 0.00, 0.00, 0.57],
    [0.00, 0.00, 0.25, 0.57],
    [0.00, 0.00, 0.75, 0.42],
    [0.20, 0.00, 0.00, 0.42],
    [0.25, 0.85, 0.00, 0.20],
])
# Na Pleasantness de "happy", só Joy e Sadness batem: o WordNet 3.0 tem empates diferentes da JWSL
TABLE_7_REPRODUCIBLE = np.ones((6, 4), dtype=bool)
TABLE_7_REPRODUCIBLE[[0, 2, 3, 5], 0] = False


@pytest.fixture(scope="module")
def model():
    return LoiaSenatoreBaseline(alpha_cut=0.5)


def tok(word, pos, lemma=None, polarity=0.0):
    return {
        "word": word, "lemma": lemma or word, "pos": pos,
        "wn_pos": LoiaSenatoreBaseline._map_pos_to_wn(pos),
        "polarity": polarity, "emo_table": np.zeros((6, 4)),
    }


def test_eq1_polarity_of_happy(model):
    assert model.compute_word_polarity({"lemma": "happy", "wn_pos": wn.ADJ}) == pytest.approx(0.695, abs=1e-3)


def test_eqs_2_to_4_polarity_modifiers(model):
    mu = model._triangular_mf_polarity(0.695)
    assert mu == pytest.approx(0.847, abs=1e-3)
    assert model._fuzzy_intensifier(mu) == pytest.approx(0.92, abs=2e-3)
    assert model._fuzzy_negator(mu) == pytest.approx(0.153, abs=1e-3)
    assert model._fuzzy_combinator(mu) == pytest.approx(0.883, abs=2e-3)


@pytest.mark.parametrize("modifiers, expected", [
    (["intensifier"], [0.89, 0.06]),
    (["negator"], [0.20, 0.75]),
    (["negator", "intensifier"], [0.84, 0.125]),
])
def test_table_6_emotion_modifiers(model, modifiers, expected):
    joy_grief = np.array([0.8, 0.25])
    np.testing.assert_allclose(model._modify_emotions(joy_grief, modifiers), expected, atol=0.01)


def test_table_3_jwsl_wu_palmer():
    raw = _raw_similarity_table(wn.synset("happiness.n.01"))
    np.testing.assert_allclose(raw, TABLE_3, atol=0.02)


def test_table_4_opposite_emotion_heuristic():
    table = emotional_similarity_table(wn.synset("happiness.n.01"), 1)
    np.testing.assert_allclose(table, TABLE_4, atol=0.015)


def test_table_7_happy():
    table = emotional_similarity_table(wn.synset("happiness.n.02"), 1)
    np.testing.assert_allclose(
        table[TABLE_7_REPRODUCIBLE], TABLE_7_HAPPY[TABLE_7_REPRODUCIBLE], atol=0.02
    )


def test_table_7_home_keeps_raw_similarities():
    table = emotional_similarity_table(wn.synset("home.n.01"), 1)
    assert table.max() < 0.5
    assert table.min() > 0.15


def test_table_8_union_and_alpha_cut(model):
    happy = emotional_similarity_table(wn.synset("happiness.n.02"), 1)
    home = emotional_similarity_table(wn.synset("home.n.01"), 1)
    units = [{"polarity": 0.695, "emo_table": happy}, {"polarity": 0.013, "emo_table": home}]
    polarity, table = model.phase4_phrase_analysis(units)
    assert polarity == pytest.approx((0.695 + 0.013) / 2)
    assert table[1, 0] == pytest.approx(0.80, abs=0.01)   # Joy
    assert table[5, 1] == pytest.approx(0.85, abs=0.01)   # Amazement
    assert table[3, 2] == pytest.approx(0.75, abs=0.01)   # Apprehension
    np.testing.assert_allclose(table[:3, 3], [0.80, 0.57, 0.57], atol=0.01)  # Admiration, Trust, Acceptance
    assert np.all((table == 0) | (table >= 0.5))


@pytest.mark.parametrize("tokens, expected", [
    ([tok("movie", "NN"), tok("is", "VBZ", "be"), tok("good", "JJ")], ("1", 3)),
    ([tok("good", "JJ"), tok("movie", "NN"), tok("theater", "NN")], ("2", 3)),
    ([tok("good", "JJ"), tok("movie", "NN")], ("2.1", 2)),
    ([tok("very", "RB"), tok("good", "JJ"), tok("movie", "NN")], ("3", 3)),
    ([tok("very", "RB"), tok("good", "JJ")], ("3.1", 2)),
    ([tok("not", "RB"), tok("like", "VB")], ("4", 2)),
    ([tok("love", "VBP"), tok("movie", "NN")], ("5", 2)),
    ([tok("not", "RB"), tok("very", "RB"), tok("good", "JJ")], ("6", 3)),
    ([tok("go", "VB"), tok("quickly", "RB")], None),
])
def test_table_5_patterns(model, tokens, expected):
    assert model._match_pattern(tokens, 0) == expected


def test_pattern_modifiers_change_polarity(model):
    good = tok("good", "JJ", polarity=0.6)
    assert model._pattern_unit([tok("very", "RB"), good])["polarity"] > 0.6
    assert model._pattern_unit([tok("not", "RB"), good])["polarity"] == pytest.approx(-0.6)
    movie = tok("movie", "NN", polarity=0.1)
    assert model._pattern_unit([good, movie])["polarity"] == pytest.approx(0.7)


def test_happy_to_go_home(model):
    polarity, table = model.analyze_text("He is happy to go home.")
    # be (0.039), happy (0.695) e o padrão 5 "go home" (0.004 + 0.013)
    assert polarity == pytest.approx((0.039 + 0.695 + 0.017) / 3, abs=0.005)
    assert table[0, 0] > 0.8 and table[1, 0] > 0.8


def test_normalize_tweet():
    text = "@bob loving it &amp; more http://t.co/x #win :)"
    assert normalize_tweet(text) == "loving it & more win . joy."
