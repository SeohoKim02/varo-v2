from dataclasses import replace
import itertools
import numpy as np
import pytest
import torch
from services.candidate_selection import Candidate, AllocationOption, SelectionProblem, evaluate
from services import candidate_dqn as d


def problem():
    candidates = tuple(Candidate(k, "product", "source", k, "FRESH", "lot", "EA",
                       {"source": 1, k: 1}, k, (AllocationOption(1, 1), AllocationOption(2, 8))) for k in ("A", "B"))
    return SelectionProblem("fixture", {"source": 2, "A": 2, "B": 2}, {"A": 2, "B": 2}, candidates, 2, "SCENARIO")


def test_mask_rejects_double_allocation_and_shared_source_violation():
    env = d.SelectionEnvironment(problem())
    env.step(1)  # allocate 2 to A, source exhausted
    assert env.valid_actions() == [len(env.actions) - 1]
    with pytest.raises(ValueError, match="invalid action"):
        env.step(2)
    assert evaluate(env.problem, env.plan)["violations"] == []


def test_reward_matches_common_finite_objective_on_every_feasible_plan():
    p = problem()
    scale = d.reward_scale(p)
    plans = []
    for a, b in itertools.product((0, 1, 2), repeat=2):
        plan = {k: q for k, q in zip(("A", "B"), (a, b)) if q}
        e = evaluate(p, plan)
        if not e["violations"]:
            plans.append(e)
    for a, b in itertools.product(plans, repeat=2):
        rank = lambda e: e["service"] - scale["cost_weight"] * e["cost"]
        if (a["service"], -a["cost"]) > (b["service"], -b["cost"]):
            assert rank(a) > rank(b)


def test_training_inference_separate_and_rng_restored(tmp_path):
    torch.manual_seed(234)
    before = torch.random.get_rng_state().clone()
    model = d.train([problem()], seed=17, episodes=25)
    assert torch.equal(before, torch.random.get_rng_state())
    params = [x.clone() for x in model.network.parameters()]
    result = d.infer(model, replace(problem(), problem_id="heldout"))
    assert result["evaluation"]["violations"] == []
    assert result["training_overlap"] is True  # A different label is still the same decision instance.
    distinct = replace(problem(), problem_id="heldout", capacities={"source": 3, "A": 2, "B": 2})
    assert d.infer(model, distinct)["training_overlap"] is False
    assert result["optimality_proven"] is False
    assert all(torch.equal(x, y) for x, y in zip(params, model.network.parameters()))
    d.save_candidate(model, tmp_path / "candidate_seed17.pt")
    with pytest.raises(ValueError):
        d.save_candidate(model, tmp_path / "candidate_seed17.pt")
    with pytest.raises(ValueError):
        d.save_candidate(model, tmp_path / "latest.pt")


def test_action_features_do_not_expose_id_or_input_order():
    a = d.SelectionEnvironment(problem())
    b = d.SelectionEnvironment(replace(problem(), candidates=problem().candidates[::-1]))
    assert a.valid_actions() == b.valid_actions()
    assert all(np.array_equal(a.features(i), b.features(i)) for i in a.valid_actions())


def test_zero_route_problem_only_stop_valid():
    e = d.SelectionEnvironment(replace(problem(), max_routes=0))
    assert e.valid_actions() == [len(e.actions) - 1]
    assert e.step(e.valid_actions()[0]) == (0.0, True)
