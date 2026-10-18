import copy
import math

import pytest

from services.research_information import prioritize_information, seller_scenario_example


PAYOFF = {"low": {"move": 0, "sell": 10}, "high": {"move": 10, "sell": 0}}
OBS = {"demand": ["low", "high"], "irrelevant": ["same", "same"]}
COST = {"demand": 2, "irrelevant": 0}


def test_explicit_probabilities_analytic_value():
    r = prioritize_information(PAYOFF, OBS, COST, {"low": .5, "high": .5})
    assert r["baseline_score"] == 5
    assert r["perfect_information_bound"] == 5
    assert r["queries"][0]["net_value_bound"] == 3
    assert r["queries"][0]["conditional_actions"] == {"high": "move", "low": "sell"}
    assert r["queries"][1]["gross_value_bound"] == 0
    assert r["recommended_query"] == "demand"


def test_no_probability_is_worst_regret_not_an_expectation():
    r = prioritize_information(PAYOFF, OBS, COST)
    assert r["mode"] == "WORST_REGRET_REDUCTION"
    assert r["baseline_score"] == 10
    assert r["perfect_information_bound"] == 10
    assert r["queries"][0]["net_value_bound"] == 8


@pytest.mark.parametrize("cost", [5, 6, 100])
def test_stop_when_cost_exhausts_information_benefit(cost):
    r = prioritize_information(PAYOFF, {"demand": OBS["demand"]}, {"demand": cost}, {"low": .5, "high": .5})
    assert r["recommended_query"] is None


def test_dominant_action_requires_no_additional_information():
    r = prioritize_information({"a": {"move": 5, "sell": 3}, "b": {"move": 6, "sell": 2}},
                               {"check": ["a", "b"]}, {"check": 0})
    assert r["perfect_information_bound"] == 0
    assert r["recommended_query"] is None


@pytest.mark.parametrize("probs", [{"low": .5}, {"low": .5, "high": .4}, {"low": -1, "high": 2},
                                    {"low": math.nan, "high": 1}, {"low": True, "high": 0}])
def test_invalid_probabilities_rejected(probs):
    with pytest.raises(ValueError):
        prioritize_information(PAYOFF, OBS, COST, probs)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "10"])
def test_nonfinite_or_nonnumeric_payoffs_rejected(bad):
    p = copy.deepcopy(PAYOFF)
    p["low"]["move"] = bad
    with pytest.raises(ValueError):
        prioritize_information(p, OBS, COST)


@pytest.mark.parametrize("cost", [{}, {"demand": -1, "irrelevant": 0}, {"demand": math.inf, "irrelevant": 0}])
def test_costs_are_required_nonnegative_and_finite(cost):
    with pytest.raises(ValueError):
        prioritize_information(PAYOFF, OBS, cost)


def test_common_actions_and_full_observation_shape_required():
    with pytest.raises(ValueError):
        prioritize_information({"x": {"a": 1}, "y": {"b": 1}}, {}, {})
    with pytest.raises(ValueError):
        prioritize_information(PAYOFF, {"q": ["one"]}, {"q": 0})


def test_zero_probability_state_does_not_change_expected_decision():
    r = prioritize_information(PAYOFF, OBS, COST, {"low": 1, "high": 0})
    assert r["baseline_action"] == "sell"
    assert r["perfect_information_bound"] == 0
    assert r["recommended_query"] is None


def test_no_input_or_production_mutation_and_scale_invariance():
    original = copy.deepcopy((PAYOFF, OBS, COST))
    r = prioritize_information(PAYOFF, OBS, COST)
    scaled = prioritize_information({s: {a: v * 10 for a, v in row.items()} for s, row in PAYOFF.items()},
                                    OBS, {q: c * 10 for q, c in COST.items()})
    assert (PAYOFF, OBS, COST) == original
    assert r["production_action_changed"] is False
    assert scaled["recommended_query"] == r["recommended_query"]
    assert scaled["queries"][0]["net_value_bound"] == 10 * r["queries"][0]["net_value_bound"]


@pytest.mark.parametrize("probabilities", [None, {"a": .2, "b": .3, "c": .5}])
def test_refined_signal_never_has_less_gross_information_value(probabilities):
    p = {"a": {"x": 6, "y": 0}, "b": {"x": 1, "y": 4}, "c": {"x": 0, "y": 7}}
    r = prioritize_information(p, {"none": ["z"]*3, "partial": ["a", "z", "z"], "full": ["a", "b", "c"]},
                               {"none": 0, "partial": 0, "full": 0}, probabilities)
    v = {row["query"]: row["gross_value_bound"] for row in r["queries"]}
    assert v["full"] >= v["partial"] >= v["none"] == 0


def test_seller_example_uses_all_three_fixed_actions_and_preserves_loss_values():
    example = seller_scenario_example()
    assert len(example["records"]) == 81
    assert example["probabilities"] is None
    for row in example["records"]:
        assert row["move_quantity"] == 4
        assert len(row["losses"]) == 3
        assert example["payoffs"][row["scenario"]] == {k: -v for k,v in row["losses"].items()}
    result = prioritize_information(example["payoffs"], example["observations"], example["costs"])
    assert len(result["queries"]) == 4
    assert result["mode"] == "WORST_REGRET_REDUCTION"
