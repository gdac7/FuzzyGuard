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
