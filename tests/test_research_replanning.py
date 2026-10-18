from itertools import product
import copy

import pytest

from services.research_replanning import (
    POLICIES, forecast, plan_value, optimal_plan, run_path,
    observed_regime, fit_policy_selector, select_policy,
)


@pytest.mark.parametrize("prediction", [(0, 0, 0), (1, 3, 2), (6, 0, 1), (2, 2, 2)])
def test_dynamic_program_matches_independent_full_enumeration(prediction):
    stock = 5
    feasible = [p for p in product(range(5), repeat=len(prediction)) if sum(p) <= stock]
    best, plan = optimal_plan(prediction, stock)
    assert best == pytest.approx(max(plan_value(p, prediction, stock) for p in feasible))
    assert sum(plan) <= stock


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("path", [(0,)*6, (6,)*6, (0, 6, 0, 6, 0, 6), (5, 4, 3, 2, 1, 0)])
def test_inventory_and_accounting_conservation(policy, path):
    r = run_path(path, policy, 4, .5, (2,)*14)
    assert r["fulfilled"] + r["discarded"] + r["terminal_stock"] == 12
    assert r["net_value"] == pytest.approx(r["gross_value"] - r["coordination_cost"])
    assert r["changes"] == sum(t["accepted"] for t in r["trace"])
    for row in r["trace"]:
        assert 0 <= row["dispatch"] <= 4
        assert row["observed_prefix"] == (2,)*14 + path[:row["period"]]


@pytest.mark.parametrize("policy", POLICIES)
def test_future_values_cannot_change_already_chosen_actions(policy):
    a = run_path((1, 2, 3, 0, 0, 0), policy, 1, .25, (2,)*14)
    b = run_path((1, 2, 3, 6, 6, 6), policy, 1, .25, (2,)*14)
    # Today's outcome is revealed after today's dispatch.
    for t in range(4):
        assert a["trace"][t]["chosen_plan"] == b["trace"][t]["chosen_plan"]


def training():
    return [{"phase": "TRAIN", "dataset": "test", "fixed": 1., "per_unit": .25, "regime": "STABLE",
             "results": {p: {"net_value": i, "changes": 0} for i, p in enumerate(POLICIES)}}]


def test_freeze_lookup_and_reject_evaluation_data():
    rows = training()
    before = copy.deepcopy(rows)
    selector = fit_policy_selector(rows, 1)
    assert select_policy(selector, "test", 1, .25, (2,)*14) == "THRESHOLD"
    assert select_policy(selector, "test", 1, .25, (1,)*7+(5,)*7) == "THRESHOLD"
    assert rows == before
    rows[0]["phase"] = "EVALUATION"
    with pytest.raises(ValueError):
        fit_policy_selector(rows)


def test_no_rule_extrapolation():
    with pytest.raises(ValueError):
        select_policy(fit_policy_selector(training()), "unknown", 1, .25, (2,)*14)


@pytest.mark.parametrize("history,regime", [((1,)*7+(5,)*7,"RISING"), ((5,)*7+(1,)*7,"FALLING"),
                                          ((2,)*14,"STABLE"), ((0,6)*7,"VARIABLE"), ((1,),"INSUFFICIENT_HISTORY")])
def test_observed_regime(history, regime):
    assert observed_regime(history) == regime


@pytest.mark.parametrize("path", [(1.5,)*6, (-1,)*6, (True,)*6, (1,)*5])
def test_no_rounding_imputation_or_invalid_paths(path):
    with pytest.raises(ValueError):
        run_path(path, "ALWAYS", 0, 0)


def test_zero_change_cost_threshold_matches_always():
    a = run_path((1,5,2,6,0,4), "ALWAYS", 0, 0)
    t = run_path((1,5,2,6,0,4), "THRESHOLD", 0, 0)
    assert a["net_value"] == t["net_value"]
    assert a["trace"] == t["trace"]
