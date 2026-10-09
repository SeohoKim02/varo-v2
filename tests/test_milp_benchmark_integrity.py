"""T3 MILP benchmark integrity: controlled scenarios A-Z plus contract and compatibility checks."""
import itertools
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp

from services import milp_benchmark_integrity as mbi
from services import milp_benchmark_integrity_validation as mbv
from services import shared_feasibility_selection as sf
from services import suhyup_algorithm_revalidation as rv
from services.optimality_gap_service import build_optimality_settings, clear_optimality_cache, run_optimality_gap

SPEC = {
    sf.SOURCE_SURPLUS: (("source_surplus",), "PROXY", "test surplus rule"),
    sf.TARGET_NEED: (("target_need_7d",), "PROXY", "test need rule"),
    sf.ROUTE_CAPACITY: (sf.ROUTE_CAPACITY_FIELDS, "USER_INPUT", "test route capacity"),
}


def _row(route, qty, cost, source="S1", target="T1", product="P1", source_cap=1000, target_cap=1000, **extra):
    return {"route_id": route, "recommended_qty": qty, "move_cost": cost, "source_id": source, "target_id": target,
            "product_id": product, "route_type": "DIRECT", "dc_id": "", "source_surplus": source_cap,
            "target_need_7d": target_cap, **extra}


def _independent(*specs):
    """Rows on disjoint lanes / groups."""
    return [_row(route, qty, cost, source=f"S{i}", target=f"T{i}", product=f"P{i}") for i, (route, qty, cost) in enumerate(specs)]


def _solve(rows, **kwargs):
    frame = pd.DataFrame(rows)
    result = rv.lexicographic_milp(frame, **kwargs)
    evidence = mbi.lexicographic_solver_evidence(result, frame["recommended_qty"].tolist())
    return result, evidence, _plan(result["selected"])


def _plan(selected):
    frame = pd.DataFrame(selected)
    if frame.empty:
        return {"service": 0.0, "cost": 0.0, "route_ids": []}
    return {"service": float(frame["recommended_qty"].sum()), "cost": float(frame["move_cost"].sum()),
            "route_ids": frame["route_id"].astype(str).tolist()}


def _sig(rows, *, caps_rows=None, constraints=mbv.CONSTRAINTS, cost_basis=mbv.COST_FIXED,
         quantity_basis=mbi.FIXED_BINARY, limit=5, objective=mbi.LEXICOGRAPHIC_OBJECTIVE, date="2026-07-01"):
    caps = sf.caps_from_columns(caps_rows or rows, SPEC)
    return mbi.comparison_signature(dataset="test", date=date, candidates=rows, constraints=constraints,
                                    caps=mbv.effective_caps(caps), cost_basis=cost_basis, quantity_basis=quantity_basis,
                                    selection_limit=limit, objective=objective)


def _caps(rows):
    return sf.caps_from_columns(rows, SPEC)


def _fake_result(status, x=None, fun=None, bound=None, gap=None, message=""):
    return SimpleNamespace(status=status, message=message, x=None if x is None else np.asarray(x, dtype=float),
                           fun=fun, mip_dual_bound=bound, mip_gap=gap, mip_node_count=0, success=status == 0)


# --------------------------------------------------------------------------------------------- A-Z scenarios


def test_a_simple_clear_optimum_is_optimal_enumerated_and_full_scope():
    rows = _independent(("R1", 10, 5), ("R2", 20, 7), ("R3", 30, 9))
    result, evidence, plan = _solve(rows)
    assert evidence["solver_status"] == mbi.OPTIMAL and evidence["optimality_proven"]
    assert plan["service"] == 60 and plan["cost"] == 21
    assert evidence["stage1"]["solver_mip_gap"] == 0.0 and evidence["stage1"]["solver_mip_gap_status"] == mbi.GAP_COMPUTED
    enumeration = mbi.enumerate_lexicographic_optimum(rows, max_routes=5)
    assert (enumeration["optimal_service"], enumeration["optimal_cost"], enumeration["alternative_optima"]) == (60, 21, 1)
    scope = mbi.benchmark_scope(solver_status=evidence["solver_status"], pool_size=3, pool_total=3)
    assert scope["benchmark_scope"] == mbi.FULL_CANDIDATE_MODEL_OPTIMUM and scope["global_optimum_claimable"] is False


def test_b_service_first_beats_a_cheaper_lower_service_plan():
    rows = [_row("HIGH", 50, 100, target_cap=50), _row("LOW", 40, 10, source="S2", target_cap=50)]
    _, evidence, plan = _solve(rows)
    assert plan["route_ids"] == ["HIGH"]
    cheaper = {"service": 40.0, "cost": 10.0, "route_ids": ["LOW"]}
    gap = mbi.decision_gap(cheaper, plan, plan_signature=_sig(rows), reference_signature=_sig(rows),
                           reference_status=evidence["solver_status"])
    assert gap["decision_service_gap"] == 10 and gap["decision_cost_gap"] is None
    assert "SERVICE_NOT_EQUAL" in gap["reason_codes"]   # lower cost is not "better" under a service-first objective


def test_c_equal_service_minimizes_cost():
    rows = [_row("EXPENSIVE", 50, 30, target_cap=50), _row("CHEAP", 50, 20, source="S2", target_cap=50)]
    _, _, plan = _solve(rows)
    assert plan == {"service": 50.0, "cost": 20.0, "route_ids": ["CHEAP"]}


def test_d_route_capacity_is_absent_from_the_milp_and_caught_by_the_independent_check():
    rows = [_row("R1", 50, 10, route_capacity_qty=30)]
    result, evidence, plan = _solve(rows)
    assert plan["route_ids"] == ["R1"]                 # the MILP model has no route capacity row
    check = mbi.independent_check(rv._records(result["selected"]), _caps(rows), max_routes=5)
    assert not check["passed"] and check["route_capacity_excess_qty"] == 20
    gap = mbi.decision_gap(plan, plan, plan_signature=_sig(rows), reference_signature=_sig(rows),
                           reference_status=evidence["solver_status"], plan_feasible=check["passed"])
    assert gap["comparison_status"] == mbi.NOT_COMPARABLE and "INDEPENDENT_CHECK_FAILED" in gap["reason_codes"]


def test_e_shared_source_cap():
    rows = [_row("A", 50, 10, target="T1", source_cap=60), _row("B", 50, 5, target="T2", source_cap=60)]
    result, _, plan = _solve(rows)
    assert plan["route_ids"] == ["B"]
    assert mbi.independent_check(rv._records(result["selected"]), _caps(rows), max_routes=5)["passed"]
    both = mbi.independent_check(rows, _caps(rows), max_routes=5)
    assert not both["passed"] and both["source_excess_qty"] == 40


def test_f_shared_target_cap():
    rows = [_row("A", 50, 10, source="S1", target_cap=60), _row("B", 50, 5, source="S2", target_cap=60)]
    _, _, plan = _solve(rows)
    assert plan["route_ids"] == ["B"]
    assert mbi.independent_check(rows, _caps(rows), max_routes=5)["target_excess_qty"] == 40


def test_g_duplicate_lane_selected_at_most_once():
    rows = [_row("A", 50, 10), _row("B", 50, 10)]   # same product / source / target / route type
    _, _, plan = _solve(rows)
    assert len(plan["route_ids"]) == 1
    assert mbi.independent_check(rows, _caps(rows), max_routes=5)["duplicate_count"] == 1


def test_h_selection_limit():
    rows = _independent(*[(f"R{i}", 10, i) for i in range(7)])
    _, _, plan = _solve(rows)
    assert len(plan["route_ids"]) == 5 and plan["service"] == 50
    assert mbi.independent_check(rows[:6], _caps(rows), max_routes=5)["selection_limit_exceeded"] is True


def _app_records(count):
    return [{"route_id": f"R{i}", "product_id": f"P{i}", "source_id": f"S{i}", "target_id": f"T{i}",
             "recommended_qty": 10, "expected_saving": 100 - i, "varo_final_rank": i + 1} for i in range(count)]


def test_i_candidate_limit_truncation_is_a_restricted_optimum_cut_by_the_evaluated_rank():
    clear_optimality_cache()
    result = run_optimality_gap(_app_records(4), {}, build_optimality_settings(candidate_limit=2, max_routes=5), "cut")
    contract = result["benchmark_integrity"]
    assert contract["candidate_pool_total"] == 4 and contract["candidate_pool_size"] == 2
    assert contract["candidate_pool_truncated"] is True
    assert contract["benchmark_scope"] == mbi.RESTRICTED_CANDIDATE_OPTIMUM
    assert {"CANDIDATE_POOL_TRUNCATED", "CUTOFF_BY_EVALUATED_STRATEGY_RANK"} <= set(contract["reason_codes"])
    assert contract["optimality_claim"] == "OPTIMAL_WITHIN_RESTRICTED_CANDIDATE_POOL"


def test_j_candidate_limit_not_truncating_is_full_input_model_only():
    clear_optimality_cache()
    result = run_optimality_gap(_app_records(4), {}, build_optimality_settings(candidate_limit=None, max_routes=5), "all")
    contract = result["benchmark_integrity"]
    assert contract["candidate_pool_truncated"] is False
    assert contract["benchmark_scope"] == mbi.FULL_CANDIDATE_MODEL_OPTIMUM
    assert "INPUT_POOL_MAY_BE_PRE_CUT" in contract["reason_codes"] and contract["global_optimum_claimable"] is False
    assert contract["solver_status"] == mbi.OPTIMAL and contract["decision_gap_status"] == mbi.GAP_COMPUTED


def test_k_partial_allocation_is_a_different_problem_even_with_equal_results():
    rows = _independent(("R1", 50, 10), ("R2", 50, 20))
    plan = {"service": 100.0, "cost": 30.0, "route_ids": ["R1", "R2"]}
    gap = mbi.decision_gap(plan, plan, plan_signature=_sig(rows, quantity_basis=mbi.PARTIAL_ALLOWED,
                                                           cost_basis=mbv.COST_PARTIAL),
                           reference_signature=_sig(rows), reference_status=mbi.OPTIMAL)
    assert gap["comparison_status"] == mbi.NOT_COMPARABLE
    assert {"DIFFERENT_QUANTITY_BASIS", "DIFFERENT_COST_BASIS"} <= set(gap["reason_codes"])
    split = [{**rows[0], "allocated_qty": 30.0}]
    assert mbi.independent_check(split, _caps(rows), max_routes=5, quantity_basis=mbi.PARTIAL_ALLOWED,
                                 allocated_field="allocated_qty")["passed"]


def test_l_fixed_quantity_selection_rejects_splits_and_fractions():
    rows = _independent(("R1", 50, 10))
    split = [{**rows[0], "allocated_qty": 30.0}]
    fixed = mbi.independent_check(split, _caps(rows), max_routes=5, allocated_field="allocated_qty")
    assert fixed["granularity_violations"] == ["R1"] and not fixed["passed"]
    fraction = mbi.independent_check([{**rows[0], "allocated_qty": 12.5}], _caps(rows), max_routes=5,
                                     quantity_basis=mbi.PARTIAL_ALLOWED, allocated_field="allocated_qty")
    assert fraction["non_integer_allocations"] == ["R1"]


def test_m_time_limit_is_never_optimal(monkeypatch):
    status = mbi.normalize_solver_status(1, "Time limit reached. (HiGHS Status 13: ...)", has_solution=False)
    assert status["solver_status"] == mbi.TIME_LIMIT and status["termination_reason"] == "TIME_LIMIT"
    assert not status["optimality_proven"]
    rows = _independent(("R1", 10, 5), ("R2", 20, 7))
    calls = iter([_fake_result(0, [1, 1], fun=-30.0, bound=-30.0, gap=0.0),
                  _fake_result(1, None, message="Time limit reached.")])
    monkeypatch.setattr(rv, "milp", lambda *a, **k: next(calls))
    result = rv.lexicographic_milp(pd.DataFrame(rows), options={"time_limit": 0.01})
    evidence = mbi.lexicographic_solver_evidence(result, [10, 20])
    assert result["optimal"] is False and result["selected"].empty
    assert evidence["solver_status"] == mbi.TIME_LIMIT and evidence["time_limit_s"] == 0.01
    assert rv._milp_status_label("not_optimal") == "not_optimal"
    scope = mbi.benchmark_scope(solver_status=evidence["solver_status"], pool_size=2, pool_total=2)
    assert scope["benchmark_scope"] == mbi.INSUFFICIENT_DATA and scope["optimality_claim"] == "NO_OPTIMALITY_CLAIM"


def test_n_feasible_but_not_proven_optimal(monkeypatch):
    rows = _independent(("R1", 10, 5), ("R2", 20, 7))
    calls = iter([_fake_result(0, [1, 1], fun=-30.0, bound=-30.0, gap=0.0),
                  _fake_result(1, [1, 1], fun=12.0, bound=10.0, gap=2 / 12, message="Time limit reached.")])
    monkeypatch.setattr(rv, "milp", lambda *a, **k: next(calls))
    result = rv.lexicographic_milp(pd.DataFrame(rows))
    evidence = mbi.lexicographic_solver_evidence(result, [10, 20])
    assert evidence["solver_status"] == mbi.FEASIBLE_NOT_PROVEN_OPTIMAL and result["optimal"] is False
    assert evidence["stage2"]["solver_mip_gap"] == pytest.approx(2 / 12)
    scope = mbi.benchmark_scope(solver_status=evidence["solver_status"], pool_size=2, pool_total=2)
    assert scope["benchmark_scope"] == mbi.FEASIBLE_SOLUTION_ONLY
    plan = _plan(result["selected"])
    gap = mbi.decision_gap(plan, plan, plan_signature=_sig(rows), reference_signature=_sig(rows),
                           reference_status=evidence["solver_status"])
    assert gap["comparison_status"] == mbi.COMPARABLE_REFERENCE_NOT_PROVEN
    assert "REFERENCE_NOT_PROVEN_OPTIMAL" in gap["reason_codes"] and mbi.SOLVER_PROVEN_OPTIMAL not in gap["zero_gap_flags"]


def test_o_infeasible_model_from_the_real_solver():
    result = milp(c=np.array([1.0]), integrality=np.ones(1), bounds=Bounds([0.0], [1.0]),
                  constraints=LinearConstraint(np.array([[1.0]]), 2.0, np.inf))
    status = mbi.normalize_solver_status(result.status, result.message, has_solution=result.x is not None)
    assert status["solver_status"] == mbi.INFEASIBLE and not status["optimality_proven"]


def test_p_missing_bound_is_null_not_zero():
    gap = mbi.solver_mip_gap(100.0, None)
    assert gap["solver_mip_gap"] is None and gap["solver_mip_gap_status"] == mbi.GAP_NOT_COMPUTABLE
    assert gap["reason_codes"] == ["MISSING_BOUND"]
    assert mbi.solver_mip_gap(None, 5.0)["reason_codes"] == ["NO_INCUMBENT"]
    assert mbi.solver_mip_gap(10.0, 8.0, sense="max")["reason_codes"] == ["INCONSISTENT_BOUND"]


def test_q_zero_objective_is_not_reported_as_zero_percent():
    gap = mbi.solver_mip_gap(0.0, 0.0)
    assert gap["solver_mip_gap"] is None and gap["solver_mip_gap_abs"] == 0.0 and gap["reason_codes"] == ["ZERO_OBJECTIVE"]
    empty = {"service": 0.0, "cost": 0.0, "route_ids": []}
    rows = _independent(("R1", 10, 5))
    decision = mbi.decision_gap(empty, empty, plan_signature=_sig(rows), reference_signature=_sig(rows),
                                reference_status=mbi.OPTIMAL)
    assert decision["decision_service_gap_pct"] is None and "ZERO_OBJECTIVE" in decision["reason_codes"]
    clear_optimality_cache()
    records = [{**item, "expected_saving": 0} for item in _app_records(3)]
    contract = run_optimality_gap(records, {}, build_optimality_settings(), "zero")["benchmark_integrity"]
    assert contract["decision_gap_status"] == mbi.GAP_NOT_COMPUTABLE and contract["benchmark_scope"] == mbi.INSUFFICIENT_DATA
    assert {"NO_POSITIVE_OBJECTIVE", "NO_FEASIBLE_CANDIDATES"} <= set(contract["reason_codes"])
    assert contract["solver_mip_gap"] is None and contract["decision_objective_gap_pct"] is None


def test_r_differing_candidate_pools_are_not_comparable():
    rows = _independent(("R1", 10, 5), ("R2", 20, 7))
    plan = {"service": 30.0, "cost": 12.0, "route_ids": ["R1", "R2"]}
    gap = mbi.decision_gap(plan, plan, plan_signature=_sig(rows[:1] + [{**rows[1], "move_cost": 8}]),
                           reference_signature=_sig(rows), reference_status=mbi.OPTIMAL)
    assert gap["comparison_status"] == mbi.NOT_COMPARABLE and "DIFFERENT_CANDIDATE_POOL" in gap["reason_codes"]
    assert gap["decision_service_gap"] is None and gap["decision_cost_gap"] is None


def test_s_differing_constraints_or_cap_values_are_not_comparable():
    rows = _independent(("R1", 10, 5))
    plan = {"service": 10.0, "cost": 5.0, "route_ids": ["R1"]}
    fewer = mbi.decision_gap(plan, plan, plan_signature=_sig(rows, constraints=("SOURCE_CAP", "SELECTION_LIMIT")),
                             reference_signature=_sig(rows), reference_status=mbi.OPTIMAL)
    assert fewer["signature_differences"] == ["constraints"] and "DIFFERENT_CONSTRAINTS" in fewer["reason_codes"]
    other_cap = mbi.decision_gap(plan, plan, plan_signature=_sig(rows, caps_rows=[{**rows[0], "target_need_7d": 6}]),
                                 reference_signature=_sig(rows), reference_status=mbi.OPTIMAL)
    assert other_cap["signature_differences"] == ["caps"] and "DIFFERENT_CAP_VALUES" in other_cap["reason_codes"]
    limit = mbi.decision_gap(plan, plan, plan_signature=_sig(rows, limit=3), reference_signature=_sig(rows),
                             reference_status=mbi.OPTIMAL)
    assert "DIFFERENT_SELECTION_LIMIT" in limit["reason_codes"]


def test_t_differing_cost_basis_is_not_comparable():
    rows = _independent(("R1", 10, 5))
    plan = {"service": 10.0, "cost": 5.0, "route_ids": ["R1"]}
    gap = mbi.decision_gap(plan, plan, plan_signature=_sig(rows, cost_basis="FIXED_PER_TRIP"),
                           reference_signature=_sig(rows), reference_status=mbi.OPTIMAL)
    assert gap["signature_differences"] == ["cost_basis"] and gap["comparison_status"] == mbi.NOT_COMPARABLE


def test_u_proxy_caps_qualify_the_claim():
    rows = _independent(("R1", 10, 5))
    caps = _caps(rows)
    scope = mbi.benchmark_scope(solver_status=mbi.OPTIMAL, pool_size=20, pool_total=36,
                                cap_provenance=[cap.provenance for cap in caps.values()])
    assert scope["scope_flags"] == [mbi.RESTRICTED_CANDIDATE_OPTIMUM, mbi.PROXY_BENCHMARK]
    assert scope["optimality_claim"] == "OPTIMAL_WITHIN_RESTRICTED_CANDIDATE_POOL_UNDER_PROXY_CAPS"
    check = mbi.independent_check(rows, caps, max_routes=5)
    assert check["passed"] and {"SOURCE_SURPLUS=PROXY", "TARGET_NEED=PROXY"} <= set(check["strict_unverifiable_constraints"])


def test_v_strict_actual_with_proxy_caps_has_no_plan_and_no_comparison():
    rows = _independent(("R1", 10, 5), ("R2", 20, 7))
    caps = sf.caps_from_columns(rows, {kind: SPEC[kind] for kind in (sf.SOURCE_SURPLUS, sf.TARGET_NEED)})
    strict = sf.select_shared_feasible(rows, caps, mode=sf.STRICT_ACTUAL, max_routes=5, partial_policy=sf.PARTIAL_NONE)
    assert strict["plan_status"] == sf.INSUFFICIENT_CAP_DATA and strict["selected_ids"] == []
    strict_caps = {slot: cap for slot, cap in caps.items() if cap.strict_accepted}
    gap = mbi.decision_gap({"service": 0.0, "cost": 0.0, "route_ids": []}, {"service": 30.0, "cost": 12.0, "route_ids": ["R1", "R2"]},
                           plan_signature=mbi.comparison_signature(
                               dataset="test", date="2026-07-01", candidates=rows, constraints=mbv.CONSTRAINTS,
                               caps=mbv.effective_caps(strict_caps), cost_basis=mbv.COST_FIXED,
                               quantity_basis=mbi.FIXED_BINARY, selection_limit=5, objective=mbi.LEXICOGRAPHIC_OBJECTIVE),
                           reference_signature=_sig(rows), reference_status=mbi.OPTIMAL)
    assert gap["comparison_status"] == mbi.NOT_COMPARABLE and gap["decision_service_gap"] is None


def test_w_deterministic_tie_break_and_alternative_optima():
    rows = [_row("B", 50, 10, source="S2", target_cap=50), _row("A", 50, 10, source="S1", target_cap=50)]
    first, evidence, plan = _solve(rows)
    second, _, again = _solve(list(reversed(rows)))
    assert plan["route_ids"] == again["route_ids"] == ["A"]      # 1e-7 x route order picks the lower route id
    enumeration = mbi.enumerate_lexicographic_optimum(rows, max_routes=5)
    assert enumeration["alternative_optima"] == 2 and enumeration["tie_break_route_ids"] == ["A"]
    assert enumeration["tie_break_unique"] and "ALTERNATIVE_OPTIMA" in enumeration["reason_codes"]
    assert evidence["stage2_tie_break"] == pytest.approx(1e-7)


def test_x_solver_answer_that_fails_the_independent_check(monkeypatch):
    rows = [_row("A", 50, 10, source="S1", target_cap=60), _row("B", 50, 5, source="S2", target_cap=60)]
    calls = iter([_fake_result(0, [1, 1], fun=-100.0, bound=-100.0, gap=0.0),
                  _fake_result(0, [1, 1], fun=15.0, bound=15.0, gap=0.0)])
    monkeypatch.setattr(rv, "milp", lambda *a, **k: next(calls))
    result = rv.lexicographic_milp(pd.DataFrame(rows))
    assert result["optimal"] is True                      # the solver claims optimality ...
    check = mbi.independent_check(rv._records(result["selected"]), _caps(rows), max_routes=5)
    assert not check["passed"] and check["target_excess_qty"] == 40   # ... the re-check does not
    enumeration = mbi.enumerate_lexicographic_optimum(rows, max_routes=5)
    assert enumeration["optimal_service"] == 50 != _plan(result["selected"])["service"]
    app = mbi.normalize_search_status({"status_code": 0, "message": "ok", "optimal": False, "has_incumbent": True})
    assert app["solver_status"] == mbi.ERROR and app["termination_reason"] == "INDEPENDENT_CHECK_FAILED"


def test_y_solver_mip_gap_and_decision_gap_are_separate_quantities():
    rows = [_row("HIGH", 50, 100, target_cap=50), _row("LOW", 40, 10, source="S2", target_cap=50)]
    _, evidence, plan = _solve(rows)
    assert evidence["stage1"]["solver_mip_gap"] == 0.0 and evidence["stage2"]["solver_mip_gap"] == 0.0
    decision = mbi.decision_gap({"service": 40.0, "cost": 10.0, "route_ids": ["LOW"]}, plan, plan_signature=_sig(rows),
                                reference_signature=_sig(rows), reference_status=evidence["solver_status"])
    assert decision["decision_service_gap_pct"] == pytest.approx(20.0)     # 0 MIP gap, 20% decision gap
    loose = mbi.solver_mip_gap(10.0, 12.0, sense="max")
    assert loose["solver_mip_gap"] == pytest.approx(0.2)
    same = mbi.decision_gap(plan, plan, plan_signature=_sig(rows), reference_signature=_sig(rows),
                            reference_status=mbi.FEASIBLE_NOT_PROVEN_OPTIMAL)
    assert same["decision_service_gap"] == 0 and same["comparison_status"] == mbi.COMPARABLE_REFERENCE_NOT_PROVEN
    contract = mbi.benchmark_contract(solver_mip_gap=0.2, decision_service_gap=0.0)
    assert contract["solver_mip_gap"] == 0.2 and contract["decision_service_gap"] == 0.0


def test_z_equal_totals_with_different_routes():
    rows = _independent(("R1", 50, 10), ("R2", 50, 10))
    _, evidence, reference = _solve(rows, )
    assert reference["route_ids"] == ["R1", "R2"]
    rows = [{**rows[0], "target_id": "T", "target_need_7d": 50}, {**rows[1], "target_id": "T", "product_id": "P0",
                                                                  "target_need_7d": 50}]
    _, evidence, reference = _solve(rows)                      # one slot for two equal-cost candidates
    enumeration = mbi.enumerate_lexicographic_optimum(rows, max_routes=5)
    assert reference["route_ids"] == ["R1"] and enumeration["alternative_optima"] == 2
    other = {"service": 50.0, "cost": 10.0, "route_ids": ["R2"]}
    gap = mbi.decision_gap(other, reference, plan_signature=_sig(rows), reference_signature=_sig(rows),
                           reference_status=evidence["solver_status"], reference_enumeration_proven=True)
    assert gap["zero_gap_flags"][:3] == [mbi.SAME_RESTRICTED_OBJECTIVE, mbi.SAME_SERVICE_AND_COST, mbi.DIFFERENT_ROUTES_SAME_TOTALS]
    assert mbi.SAME_ROUTES not in gap["zero_gap_flags"] and mbi.NOT_PROVEN_GLOBAL in gap["zero_gap_flags"]
    assert {mbi.SOLVER_PROVEN_OPTIMAL, mbi.ENUMERATION_PROVEN_OPTIMAL} <= set(gap["zero_gap_flags"])
    assert gap["decision_service_gap"] == 0 and gap["decision_cost_gap"] == 0


# --------------------------------------------------------------------------------------------- contract / compatibility


def test_lexicographic_milp_keeps_its_keys_and_default_solver_options(monkeypatch):
    seen = []
    real = rv.milp
    monkeypatch.setattr(rv, "milp", lambda *a, **k: seen.append(k["options"]) or real(*a, **k))
    result = rv.lexicographic_milp(pd.DataFrame(_independent(("R1", 10, 5))))
    assert {"stage1_status", "stage2_status", "selected", "optimal"} <= set(result)
    assert seen == [{"presolve": True}, {"presolve": True}]          # unchanged solver call
    strict = rv.lexicographic_milp(pd.DataFrame(_independent(("R1", 10, 5))), options={"mip_rel_gap": 0.0})
    assert strict["diagnostics"]["options"] == {"presolve": True, "mip_rel_gap": 0.0}
    assert rv.lexicographic_milp(pd.DataFrame())["diagnostics"]["stage1"] is None


def test_stage2_preserves_the_stage1_service_exactly():
    rows = [_row("A", 50, 100, target_cap=60), _row("B", 30, 1, source="S2", target_cap=60),
            _row("C", 30, 2, source="S3", target_cap=60)]
    result, evidence, plan = _solve(rows)
    assert evidence["stage1_service"] == 60 and plan["service"] == 60 and plan["cost"] == 3
    assert evidence["service_optimum_exact_by_integrality"] is True


def test_enumeration_matches_brute_force_on_random_instances():
    rng = random.Random(7)
    for _ in range(25):
        frame = pd.DataFrame([
            _row(f"R{i:02d}", rng.choice([10, 20, 30, 50]), rng.choice([5, 7, 9, 11]), source=f"S{rng.randint(1, 3)}",
                 target=f"T{rng.randint(1, 3)}", product="P1", source_cap=rng.choice([30, 50, 80]),
                 target_cap=rng.choice([30, 60, 90])) for i in range(9)])
        # one cap per group, as the benchmark builds them from the inventory
        frame["source_surplus"] = frame.groupby(["source_id", "product_id"])["source_surplus"].transform("first")
        frame["target_need_7d"] = frame.groupby(["target_id", "product_id"])["target_need_7d"].transform("first")
        rows, limit = frame.to_dict("records"), rng.choice([2, 3, 5])
        best, count = None, 0
        for size in range(limit + 1):
            for combo in itertools.combinations(rows, size):
                if rv._selection_violation(list(combo)) is not None:
                    continue
                key = (-sum(r["recommended_qty"] for r in combo), sum(r["move_cost"] for r in combo))
                if best is None or key < best:
                    best, count = key, 1
                elif key == best:
                    count += 1
        enumeration = mbi.enumerate_lexicographic_optimum(rows, max_routes=limit)
        assert (enumeration["optimal_service"], enumeration["optimal_cost"]) == (-best[0], best[1])
        assert enumeration["alternative_optima"] == count
        _, evidence, plan = _solve(rows) if limit == 5 else (None, None, None)
        if plan is not None:
            assert (plan["service"], plan["cost"]) == (-best[0], best[1])


def test_enumeration_refuses_pools_beyond_the_limit():
    rows = _independent(*[(f"R{i}", 10, 1) for i in range(12)])
    result = mbi.enumerate_lexicographic_optimum(rows, max_routes=5, combination_limit=100)
    assert result["status"] == "NOT_ENUMERATED" and result["reason_codes"] == ["ENUMERATION_TOO_LARGE"]


def test_status_normalizer_covers_every_scipy_code():
    assert mbi.normalize_solver_status(0, "ok", has_solution=True)["solver_status"] == mbi.OPTIMAL
    assert mbi.normalize_solver_status(1, "Iteration limit reached.", has_solution=False)["solver_status"] == mbi.NOT_SOLVED
    assert mbi.normalize_solver_status(1, "Time limit reached.", has_solution=True)["solver_status"] == mbi.FEASIBLE_NOT_PROVEN_OPTIMAL
    assert mbi.normalize_solver_status(3)["solver_status"] == mbi.UNBOUNDED
    assert mbi.normalize_solver_status(4, "The problem is unbounded or infeasible.")["termination_reason"] == "UNBOUNDED_OR_INFEASIBLE"
    assert mbi.normalize_solver_status(9)["solver_status"] == mbi.UNKNOWN
    assert mbi.normalize_solver_status(None)["solver_status"] == mbi.NOT_SOLVED
    assert mbi.normalize_solver_status(0, error="boom")["solver_status"] == mbi.ERROR
    assert set(mbi.SOLVER_STATUSES) == {mbi.OPTIMAL, mbi.FEASIBLE_NOT_PROVEN_OPTIMAL, mbi.TIME_LIMIT, mbi.INFEASIBLE,
                                        mbi.UNBOUNDED, mbi.ERROR, mbi.NOT_SOLVED, mbi.UNKNOWN}


def test_search_normalizer_for_the_app_bnb_paths():
    exact = mbi.normalize_search_status({"method": "deterministic exact BnB", "optimal": True, "selected_indices": [0]})
    assert exact["solver_status"] == mbi.OPTIMAL and exact["termination_reason"] == "SEARCH_EXHAUSTED"
    limited = mbi.normalize_search_status({"method": "deterministic limited BnB", "optimal": False, "timed_out": False,
                                           "search_exhausted": True, "selected_indices": [0]})
    assert limited["solver_status"] == mbi.FEASIBLE_NOT_PROVEN_OPTIMAL and limited["termination_reason"] == "LIMITED_SEARCH_MODE"
    timed = mbi.normalize_search_status({"method": "deterministic limited BnB", "optimal": False, "timed_out": True})
    assert timed["termination_reason"] == "TIME_LIMIT"
    assert mbi.normalize_search_status({"available": False, "error": "no scipy"})["solver_status"] == mbi.ERROR


def test_app_gap_result_keeps_existing_keys_and_adds_a_parallel_contract():
    clear_optimality_cache()
    result = run_optimality_gap(_app_records(3), {}, build_optimality_settings(), "keys")
    assert {"settings", "summary_rows", "route_rows", "constraint_rows", "constraint_usage_rows", "excluded_rows",
            "combinations", "gap", "matches", "search", "summary", "metadata"} <= set(result)
    assert result["search"]["status"] == "정확 최적해" and result["gap"]["label"] == "최적성 Gap"
    contract = result["benchmark_integrity"]
    assert set(mbi.CONTRACT_FIELDS) <= set(contract) and contract["contract_version"] == mbi.BENCHMARK_INTEGRITY_VERSION
    assert contract["solver_mip_gap_status"] == mbi.GAP_COMPUTED and contract["decision_objective_gap_pct"] == result["gap"]["gap_pct"]
    cached = run_optimality_gap(_app_records(3), {}, build_optimality_settings(), "keys")
    assert cached["metadata"]["cache_hit"] and cached["benchmark_integrity"] == contract


def test_app_gap_contract_failure_never_breaks_the_gap_result(monkeypatch):
    clear_optimality_cache()

    def boom(_):
        raise RuntimeError("contract failure")

    monkeypatch.setattr(mbi, "app_gap_benchmark_integrity", boom)
    result = run_optimality_gap(_app_records(3), {}, build_optimality_settings(), "isolated")
    assert result["benchmark_integrity"]["contract_status"] == "ERROR" and result["gap"]["available"] is True
    clear_optimality_cache()


def test_contract_never_claims_a_global_optimum_and_keeps_nulls():
    contract = mbi.benchmark_contract(global_optimum_claimable=True, reason_codes=["A", "A", "B"])
    assert contract["global_optimum_claimable"] is False and contract["reason_codes"] == ["A", "B"]
    assert contract["solver_mip_gap"] is None and contract["decision_cost_gap"] is None
    assert mbi.benchmark_scope(solver_status=mbi.OPTIMAL, pool_size=3, pool_total=None)["benchmark_scope"] == mbi.RESTRICTED_CANDIDATE_OPTIMUM


def test_milp_status_label_is_taken_from_the_milp_row():
    assert rv._milp_status_label("optimal") == "optimal"
    assert rv._milp_status_label("feasible") == "not_optimal"
    assert rv._milp_status_label(None) == "not_optimal"


def test_effective_caps_use_the_tightest_source_cap_with_surplus_first_on_ties():
    tie = {(sf.SOURCE_STOCK, ("S1", "P1")): sf.Cap(sf.SOURCE_STOCK, ("S1", "P1"), 50.0, "DIRECT_REAL", "stock"),
           (sf.SOURCE_SURPLUS, ("S1", "P1")): sf.Cap(sf.SOURCE_SURPLUS, ("S1", "P1"), 50.0, "PROXY", "surplus")}
    assert mbv.effective_caps(tie)[("SOURCE_CAP", ("S1", "P1"))].provenance == "PROXY"
    tighter = {**tie, (sf.SOURCE_STOCK, ("S1", "P1")): sf.Cap(sf.SOURCE_STOCK, ("S1", "P1"), 20.0, "DIRECT_REAL", "stock")}
    assert mbv.effective_caps(tighter)[("SOURCE_CAP", ("S1", "P1"))].value == 20.0
    missing = {(sf.ROUTE_CAPACITY, ("R1",)): sf.Cap(sf.ROUTE_CAPACITY, ("R1",), None, "MISSING", "none")}
    assert mbv.effective_caps(missing) == {}


def test_old_benchmark_caps_use_the_generator_need_formula():
    inventory = pd.DataFrame({"store_id": ["A", "B", "C"], "product_id": ["P1"] * 3, "stock_qty": [10, 45, 102]})
    record = {"product_id": "P1", "source_stock": 116.0, "target_stock": 102.0, "target_daily_demand_proxy": 9.0}
    old = mbv.old_benchmark_caps([record], inventory)[0]
    assert old["median_stock"] == 45 and old["target_need_7d"] == 63.0 and old["source_surplus"] == 71.0
    benchmark = max(0.0, 45 + 7 * 9 - 102)
    assert benchmark == 6.0                                   # the 2026-07-16 case: 63 vs 6 under one column name


def _upload():
    stores = pd.DataFrame({"node_id": ["A", "B", "C"], "node_name": ["A", "B", "C"], "node_type": ["STORE"] * 3})
    products = pd.DataFrame({"product_id": ["P1", "P2"], "product_name": ["p1", "p2"], "unit_price": [1000, 1000]})
    inventory = pd.DataFrame({"store_id": ["A", "B", "C", "A", "B", "C"], "product_id": ["P1"] * 3 + ["P2"] * 3,
                              "stock_qty": [300, 20, 10, 200, 15, 5], "sales_qty": [1, 3, 4, 1, 2, 2]})
    routes = pd.DataFrame([{"source_id": s, "target_id": t, "distance_km": 5.0, "estimated_cost": 100.0, "travel_time_min": 10.0}
                           for s in "ABC" for t in "ABC" if s != t])
    return {"stores": stores, "products": products, "inventory": inventory, "routes": routes}


def test_generator_pool_and_lane_pool_lift_only_the_cuts(monkeypatch):
    monkeypatch.delenv("VARO_REAL_DATA_ROOT", raising=False)
    from services import candidate_generator as cg

    upload = _upload()
    full, _ = mbv.generator_pool(upload)
    assert cg.MAX_CANDIDATES == 20                             # restored after the uncut run
    cut, _ = cg.generate_candidates(upload)
    assert full.head(len(cut))["route_id"].tolist() == cut["route_id"].tolist()
    lanes = mbv.lane_pool(upload)
    assert len(lanes) == 4                                     # 2 eligible (source, product) x 2 reachable targets
    key = lambda f: set(zip(f["product_id"].astype(str), f["source_id"].astype(str), f["target_id"].astype(str),
                            f["recommended_qty"].astype(float)))
    assert key(full) <= key(lanes)


def test_claims_audit_does_not_reclassify_an_explicit_non_claim(tmp_path):
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "validation.py").write_text(
        'a = "제한 탐색 결과이므로 확정 최적성 Gap이 아닙니다."\nb = "최적성 Gap 계산"\n', encoding="utf-8")
    audit = mbv.claims_audit(tmp_path)
    hits = audit[audit["location"] == "pages/validation.py"]
    assert hits[hits["line"] == 1]["verdict"].tolist() == ["SAFE"]
    assert hits[hits["line"] == 2]["verdict"].tolist() == ["NEEDS_QUALIFIER"]
    assert (audit[audit["claim_id"] == "P04"]["verdict"] == "UNSUPPORTED").all()
