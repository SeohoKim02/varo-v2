import copy
import numpy as np
import pandas as pd
import pytest

from services import forecast_candidate as forecast
from services import candidate_dqn as dqn
from services.learning_third_audit import allocation_problem, forecast_economic_replay, inspect_unmasked_proposal, service_cost_order


def test_third_constant_reference_retained_without_fake_gain():
    r = forecast.select_third_train_only(np.ones(70)*3, np.zeros(70, bool))
    assert r["selected"]["method"] == "EWMA20"
    assert r["reference_retained"] and not r["production_approved"]
    assert r["max_inner_label_index"] == 69


@pytest.mark.parametrize("bad", [np.ones(48), np.ones((70, 1)), np.ones(70)*-1])
def test_third_rejects_ambiguous_training_shape_or_negative_sales(bad):
    with pytest.raises(ValueError):
        forecast.select_third_train_only(bad, np.zeros(bad.shape, bool))


def test_third_requires_uncensored_labels_in_each_inner_window():
    h = np.ones(70)
    mask = np.zeros(70, bool)
    mask[-7:] = True
    assert forecast.select_third_train_only(h, mask)["status"] == "INSUFFICIENT_UNCENSORED_VALIDATION"


def test_third_selector_threshold_is_applied_only_to_past_labels():
    h = np.resize([1, 2, 9, 1, 0, 3, 10], 98)
    original = h.copy()
    a = forecast.select_third_train_only(h[:69], np.zeros(69, bool))
    h[69:] = 9999
    b = forecast.select_third_train_only(h[:69], np.zeros(69, bool))
    assert a == b and a["selected"]["method"] == "SEASONAL7"
    assert a["selected"]["inner_mae"] < a["candidates"][0]["inner_mae"]*.95
    assert np.array_equal(h[:69], original[:69])


@pytest.mark.parametrize("seed", [3, 7, 17, 29, 43])
def test_unmasked_diagnostic_never_executes_invalid_proposal(seed):
    import torch
    p = allocation_problem(4100)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = dqn.TrainedCandidate(dqn._network(), {"seed": seed, "training_decision_fingerprints": []})
    r = inspect_unmasked_proposal(model, p)
    assert not r["evaluation"]["violations"]
    assert r["unsafe_action_applied"] is False
    assert r["invalid_proposals"] <= 1


def test_forecast_economics_freezes_plan_before_realization():
    a = forecast_economic_replay(4, 1, 1)
    b = forecast_economic_replay(4, 9, 1)
    assert a["planned_transfer"] == b["planned_transfer"] == 40
    assert a["planned_expected_net"] == b["planned_expected_net"]
    assert a["replayed_scenario_net"] < b["replayed_scenario_net"]
    assert a["actual_savings"] is None
    assert a["constraint_violations"] == b["constraint_violations"] == 0


@pytest.mark.parametrize("x", [-1, float("nan"), float("inf")])
def test_forecast_economic_bridge_never_imputes_invalid_input(x):
    with pytest.raises(ValueError):
        forecast_economic_replay(x, 1, 1)


def test_vhs_comparator_uses_declared_macro_service_not_raw_quantity():
    from services.candidate_selection import Candidate, AllocationOption, SelectionProblem
    cs = tuple(Candidate(key, "P", "S", key, "FRESH", "L", "EA", {"S": 1, key: 1}, key,
        (AllocationOption(4, cost),)) for key, cost in (("small", 10), ("large", 1)))
    p = SelectionProblem("test", {"S": 4, "small": 4, "large": 8}, {"small": 4, "large": 8}, cs, 1, "SCENARIO")
    assert service_cost_order(p) == ["small", "large"]
