from dataclasses import replace
from fractions import Fraction
import pytest

from services.candidate_selection import (
    AllocationOption as O, Candidate, SelectionProblem as P, evaluate, exact_objective,
    exhaustive, improve_selection, incumbent_compatibility, protected_selection, solve,
)


def slot_problem():
    def candidate(cid, source, target, qty, cost):
        return Candidate(cid, "P", source, target, "CHILLED", "L1", "EA",
                         {source: 1, target: 1}, target, tuple(O(q, cost) for q in range(1, qty + 1)))
    return P("slot", {"S1": 6, "S2": 4, "T1": 5, "T2": 5, "T3": 4},
             {"T1": 5, "T2": 5, "T3": 4},
             (candidate("a", "S1", "T1", 5, 1), candidate("b", "S1", "T2", 5, 2),
              candidate("c", "S2", "T3", 4, 3)), 2, "SCENARIO", "LEGACY_RAW_QTY",
             {"decision_time": "2026-01-01", "information_set_id": "known-at-t0"})


def test_minimal_partial_route_slot_regression_and_real_improvement():
    p = slot_problem()
    whole, partial = solve(p, "VARO"), solve(p, "PARTIAL")
    assert whole["plan"] == {"a": 5, "c": 4}
    assert partial["plan"] == {"a": 5, "b": 1}
    improved = improve_selection(p)
    assert improved["quantity"] == 9
    assert any(t["accepted"] for t in improved["trace"])
    assert not improved["solver"]["optimality_proven"]
    assert all(not t["warm_start_used"] for t in improved["trace"])


def test_two_route_exchange_escapes_one_exchange_trap():
    def c(cid, source, target, qty):
        return Candidate(cid, "P", source, target, "C", "L", "EA", {source: 1, target: 1}, target, (O(qty, 1),))
    p = P("pair", {"a": 4, "b": 4, "x": 4, "y": 4}, {"x": 4, "y": 4},
          (c("ax", "a", "x", 3), c("by", "b", "y", 3), c("ay", "a", "y", 4), c("bx", "b", "x", 4)),
          2, "SCENARIO", "LEGACY_RAW_QTY")
    initial = {"ax": 3, "by": 3}
    assert improve_selection(p, initial_plan=initial, max_removed=1)["quantity"] == 6
    result = improve_selection(p, initial_plan=initial, max_removed=2)
    assert result["quantity"] == 8
    assert any(len(x["removed"]) == 2 and x["accepted"] for x in result["trace"])


def test_protection_preserves_baseline_and_discloses_fallback():
    p = slot_problem(); baseline = solve(p, "VARO"); partial = solve(p, "PARTIAL")
    r = protected_selection(p, baseline_plan=baseline["plan"], challengers={"PARTIAL": partial})
    assert r["plan"] == baseline["plan"] and r["fallback_used"]
    assert r["selected_source"] == "BASELINE_VARO"
    r = protected_selection(p, challengers={"WHOLE_TIE": baseline})
    assert r["baseline_retained"] and not r["fallback_used"]


def test_zero_budget_returns_verified_seed_not_unproven_empty_replacement():
    p = slot_problem(); partial = solve(p, "PARTIAL")
    r = improve_selection(p, time_limit=0)
    assert r["plan"] == partial["plan"] and r["solver"]["termination"] == "TIME_LIMIT"
    assert protected_selection(p, time_limit=0)["quantity"] == 9


@pytest.mark.parametrize("change", ["date", "information", "cap", "unit", "cost", "objective"])
def test_changed_problem_cannot_reuse_incumbent(change):
    p = slot_problem(); old = p
    if change == "date": p = replace(p, problem_id="other-date")
    if change == "information": p = replace(p, policy={**p.policy, "information_set_id": "future"})
    if change == "cap": p = replace(p, capacities={**p.capacities, "S1": 5})
    if change == "unit": p = replace(p, candidates=(replace(p.candidates[0], unit="BOX"), *p.candidates[1:]))
    if change == "cost": p = replace(p, candidates=(replace(p.candidates[0], options=(O(5, 2),)), *p.candidates[1:]))
    if change == "objective": p = replace(p, service_metric="MACRO_FULFILLMENT")
    assert not incumbent_compatibility(p, old, solve(old, "VARO")["plan"])["compatible"]
    with pytest.raises(ValueError, match="cannot be reused"):
        protected_selection(p, baseline_problem=old)


def test_whole_options_are_subset_and_zero_is_implicit():
    p = slot_problem(); old = replace(p, candidates=tuple(replace(c, options=(c.options[-1],)) for c in p.candidates))
    check = incumbent_compatibility(p, old, solve(old, "VARO")["plan"])
    assert check["compatible"] and check["all_old_options_contained"]
    assert not evaluate(p, {})["violations"]


def test_invalid_challenger_never_accepted_even_with_forged_metrics():
    p = slot_problem(); bad = solve(p, "VARO")
    bad["plan"] = {"a": 5, "b": 5}; bad["service"] = 1000000
    with pytest.raises(ValueError, match="independent validation"):
        protected_selection(p, challengers={"bad": bad})


def test_exact_gate_does_not_round_away_service_loss():
    p = slot_problem()
    c = replace(p.candidates[0], options=(O(1, 100), O(0.99999999999, 0)))
    p = replace(p, candidates=(c,))
    assert exact_objective(p, {"a": 1}) > exact_objective(p, {"a": 0.99999999999})
    assert exact_objective(p, {"a": 1})[0] == Fraction(1)


def test_exact_gate_rejects_tolerance_only_undeclared_quantity():
    p = slot_problem()
    assert evaluate(p, {"a": 1.00000000001})["violations"] == ["UNKNOWN_ALLOCATION:a"]
    with pytest.raises(ValueError, match="infeasible"):
        exact_objective(p, {"a": 1.00000000001})


def test_close_declared_fractional_options_keep_their_own_costs():
    p = slot_problem()
    c = replace(p.candidates[0], options=(O(1, 100), O(0.99999999999, 7)))
    p = replace(p, candidates=(c,))
    assert evaluate(p, {"a": 1})["cost"] == 100
    assert evaluate(p, {"a": 0.99999999999})["cost"] == 7
    assert exact_objective(p, {"a": 0.99999999999})[1] == -7


def test_improved_small_problem_matches_enumeration_and_common_evaluator():
    p = slot_problem(); r = improve_selection(p)
    assert exact_objective(p, r["plan"]) == exact_objective(p, exhaustive(p))
    for key, value in evaluate(p, r["plan"]).items():
        assert r[key] == value


def test_challenger_other_pool_is_rejected_even_when_plan_is_valid():
    p = slot_problem(); r = solve(replace(p, candidates=p.candidates[:1]), "VARO")
    with pytest.raises(ValueError, match="signature mismatch"):
        protected_selection(p, challengers={"other": r})
