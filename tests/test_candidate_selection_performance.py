"""Correctness of performance changes, not brittle wall-clock thresholds."""
from dataclasses import asdict, replace
from fractions import Fraction
import itertools
import random

import numpy as np
import pytest

from services import candidate_selection as s
from services.operating_snapshot import signature


def problem(seed=17, metric="LEGACY_RAW_QTY"):
    rng = random.Random(seed)
    needs = {"t1": 6., "t2": 8.}
    candidates = tuple(s.Candidate(str(i), "p", "s", f"t{1+i%2}", "FRESH", "lot", "EA",
        {"stock": 1., f"t{1+i%2}": 1.}, f"t{1+i%2}",
        tuple(s.AllocationOption(q, rng.randint(0, 20)) for q in (1., 2., 3.)),
        vhs=rng.uniform(-2, 2)) for i in range(4))
    return s.SelectionProblem("case", {"stock": 7., **needs}, needs, candidates, 2, "SCENARIO", metric,
                              {"business": "A", "decision_time": "2026-01-01", "risk": "expected"})


@pytest.mark.parametrize("metric", ["LEGACY_RAW_QTY", "MACRO_FULFILLMENT"])
@pytest.mark.parametrize("seed", range(8))
def test_relaxed_bound_dominates_every_finite_plan_with_nonmonotone_costs(seed, metric):
    p = problem(seed, metric)
    upper = s._relaxed_upper_bound(p, p.candidates, p.max_routes, (Fraction(0),) * 3)
    for values in itertools.product(*[(None, *c.options) for c in p.candidates]):
        plan = {c.candidate_id: option.quantity for c, option in zip(p.candidates, values) if option is not None}
        if not s.evaluate(p, plan)["violations"]:
            assert s.exact_objective(p, plan) <= upper


def test_base_signature_identical_to_previous_full_asdict_without_candidates():
    p = problem(); old = asdict(p); old.pop("candidates")
    assert p.base_problem_signature == signature(old)
    # Mappings may be mutated by a caller despite the frozen dataclass. There is
    # no object-identity cache: changed information immediately changes the hash.
    prior = p.base_problem_signature
    p.policy["decision_time"] = "2026-01-02"
    assert p.base_problem_signature != prior


def test_sparse_milp_matrix_is_identical_to_previous_dense_matrix(monkeypatch):
    p = problem(); actual_milp = s.milp; matrices = []
    def inspect(*args, **kwargs):
        matrices.append(kwargs["constraints"][0].A.toarray())
        return actual_milp(*args, **kwargs)
    monkeypatch.setattr(s, "milp", inspect)
    result = s.solve(p)
    items = [(c, o) for c in sorted(p.candidates, key=lambda c: c.candidate_id)
             for o in sorted(c.options, key=lambda o: o.quantity)]
    dense = [[float(ci.candidate_id == c.candidate_id) for ci, _ in items] for c in p.candidates]
    dense += [[o.quantity * c.resources.get(k, 0) for c, o in items] for k in sorted(p.capacities)]
    dense.append([1.] * len(items))
    assert matrices and np.array_equal(matrices[0], np.array(dense))
    assert s.exact_objective(p, result["plan"]) == s.exact_objective(p, s.exhaustive(p))


def test_incremental_greedy_preserves_all_diagnostics_and_avoids_full_trial_evaluation(monkeypatch):
    p = problem(); original = s.evaluate; calls = []
    def counted(*args, **kwargs):
        calls.append(1); return original(*args, **kwargs)
    monkeypatch.setattr(s, "evaluate", counted)
    result = s.solve(p, "PARTIAL")
    assert len(calls) == 2  # Initial state and independent final verification.
    running = {}
    for row in result["trace"]:
        running[row["candidate"]] = row["quantity"]
        assert row["remaining"] == original(p, running)["remaining"]
    assert result["plan"] == running


def test_no_solver_needed_when_relaxed_bound_proves_all_neighbors_nonimproving(monkeypatch):
    p = replace(problem(), candidates=problem().candidates[:1], max_routes=1)
    def unexpected(*args, **kwargs):
        raise AssertionError("non-improving relaxed bound must not call solver")
    monkeypatch.setattr(s, "milp", unexpected)
    result = s.improve_selection(p)
    assert result["solver"]["bound_pruned"] > 0
    assert result["solver"]["residual_milp_calls"] == 0
    assert not result["solver"]["optimality_proven"]
    assert not result["violations"]


@pytest.mark.parametrize("field,value", [("risk", "worst"), ("business", "B"), ("solver_version", "new")])
def test_reused_baseline_never_crosses_information_or_business_boundary(field, value):
    p = problem(); old_plan = s.solve(p, "VARO")["plan"]
    changed = replace(p, policy={**p.policy, field: value})
    with pytest.raises(ValueError, match="cannot be reused"):
        s.protected_selection(changed, baseline_problem=p, baseline_plan=old_plan, time_limit=0)


def test_zero_budget_still_validates_current_baseline_and_never_reuses_changed_stock():
    p = problem(); old_plan = s.solve(p, "VARO")["plan"]
    assert not s.protected_selection(p, baseline_plan=old_plan, time_limit=0)["violations"]
    changed = replace(p, capacities={**p.capacities, "stock": 0})
    with pytest.raises(ValueError, match="cannot be reused"):
        s.protected_selection(changed, baseline_problem=p, baseline_plan=old_plan, time_limit=0)


def test_new_search_matches_exhaustive_with_negative_tie_and_nonmonotone_costs():
    p = problem()
    result = s.improve_selection(p, time_limit=10, neighborhood_limit=100)
    assert s.exact_objective(p, result["plan"]) == s.exact_objective(p, s.exhaustive(p))
