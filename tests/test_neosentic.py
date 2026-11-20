import numpy as np
import pytest

from src.fuzzy_sentic.models.neosentic import (
    NeoSenticFuzzy,
    iowa,
    owa_weights,
    sugeno_attenuation,
    yager_union,
)


@pytest.fixture(scope="module")
def model():
    return NeoSenticFuzzy(alpha_cut=0.5)


def polarity(model, text):
    return model.analyze_text(text)[0]


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_owa_weights_sum_to_one_and_favor_strongest(n):
    weights = owa_weights(n, beta=0.5)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights) <= 1e-12)


def test_iowa_orders_by_strength_and_ignores_zeros():
    assert iowa([0.4]) == pytest.approx(0.4)
    assert iowa([0.0, 0.0]) == 0.0
    assert iowa([0.05, -0.8, 0.0]) < -0.4


def test_sugeno_attenuation():
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(sugeno_attenuation(x, 0.0), x)
    assert np.all(sugeno_attenuation(x, 2.0) <= x)
    assert sugeno_attenuation(1.0, 5.0) == pytest.approx(1.0)


def test_yager_union_is_between_max_and_one():
    a, b = np.array([0.3, 0.6, 0.0]), np.array([0.5, 0.6, 0.0])
    union = yager_union(a, b)
    assert np.all(union >= np.maximum(a, b)) and np.all(union <= 1.0)


def test_interface_contract(model):
    p, table = model.analyze_text("What a wonderful day! I hate waiting though.")
    assert isinstance(p, float) and -1.0 <= p <= 1.0
    assert table.shape == (6, 4) and table.min() >= 0.0 and table.max() <= 1.0


def test_empty_text(model):
    p, table = model.analyze_text("")
    assert p == 0.0 and not table.any()


def test_negation_and_hedges(model):
    good = polarity(model, "The movie is good.")
    bad = polarity(model, "The movie is bad.")
    assert good > 0 > bad
    assert polarity(model, "The movie is not good.") < 0
    assert polarity(model, "The movie is not bad.") > bad
    assert polarity(model, "The movie is very good.") > good
    assert polarity(model, "The movie is slightly good.") < good

def test_compound_negation_and_hedge(model):
    good = polarity(model, "The movie is good.")
    assert polarity(model, "The movie is not very good.") < 0
    assert polarity(model, "Not very good.") < 0
    assert polarity(model, "The movie is not very good.") < good

    _, emo_table = model.analyze_text("Not very good.")
    assert emo_table[3:, 0].sum() >= emo_table[:3, 0].sum()

def test_negation_scope_through_dependencies(model):
    assert polarity(model, "I don't think it's good.") < 0
    assert polarity(model, "I don't like this movie.") < 0


def test_opposite_emotions_do_not_coexist(model):
    units = model.extract_syntactic_units("I am so happy")
    model._analyze_units(units)
    for unit in units:
        table = unit["emo_table"]
        for pos_idx, neg_idx in [(0, 5), (1, 4), (2, 3)]:
            assert np.all((table[pos_idx] == 0) | (table[neg_idx] == 0))

def test_affective_stopwords_are_preserved(model):
    units = model.extract_syntactic_units("I feel alone.")
    assert any(u["lemma"] == "alone" for u in units)
    assert polarity(model, "I feel well.") > 0
    
def test_adversative_and_concessive_clauses(model):
    assert polarity(model, "The acting is wonderful, but the ending is terrible.") < 0
    assert polarity(model, "Although the acting was wonderful, the ending was terrible.") < 0
    assert polarity(model, "The ending was terrible, although the acting was wonderful.") < 0

def test_oov_contextual_emotion_projection(model):
    units = model.extract_syntactic_units("That joke was really cringeworthy.")
    model._analyze_units(units)

    oov_unit = next(u for u in units if u["sense"] is None)
    assert oov_unit["word"] == "cringeworthy"
    assert "That joke was really cringeworthy." in model._emotional_query_text(oov_unit)
    assert oov_unit["emo_table"].max() > 0.0
    assert oov_unit["emo_table"][3:].max() > oov_unit["emo_table"][:3].max()

def test_negation_of_zero_polarity_with_active_emotion(model):
    emo = np.zeros((6, 4))
    emo[1, 0] = 0.6
    pol, negated_emo = model.apply_fuzzy_modifiers(0.0, emo, [("neg", 0.0)])

    assert pol < 0.0
    assert negated_emo[4, 0] > 0.0
    assert pol == pytest.approx(-negated_emo[4, 0])

def test_negation_scope_over_coordination(model):
    assert polarity(model, "The movie is not good or exciting.") < 0
    assert polarity(model, "The movie is not good, but exciting.") > 0

def test_interjection_uses_sentiwordnet_senses(model):
    units = model.extract_syntactic_units("Alas, the plan failed.")
    model._analyze_units(units)

    alas_unit = next(u for u in units if u["lemma"] == "alas")
    assert alas_unit["pos"] == "INTJ"
    assert alas_unit["sense"] is not None
    assert alas_unit["polarity"] < 0