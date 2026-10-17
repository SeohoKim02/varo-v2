from dataclasses import asdict
import math

import pytest

from services.candidate_selection import AllocationOption, Candidate, SelectionProblem, evaluate, solve
from services.decision_sensitivity_audit import perturb_problem, independently_account, compare_plans, finite_marginals


def problem():
    candidates = tuple(Candidate(cid, "P", "S", target, "FRESH", "L", "EA", {"source:S":1.,target:1.}, target,
                                 tuple(AllocationOption(q,fixed+q) for q in (1.,2.,3.)), vhs=score)
                       for cid,target,fixed,score in (("a","target:A",1,2),("b","target:B",2,1)))
    return SelectionProblem("test",{"source:S":4.,"target:A":3.,"target:B":3.},
                            {"target:A":3.,"target:B":3.},candidates,2,"SCENARIO","LEGACY_RAW_QTY")


@pytest.mark.parametrize("factor",["source_capacity","target_need","selected_route_cost","all_route_cost"])
@pytest.mark.parametrize("change",[-.1,0,.1])
def test_perturbation_preserves_original_and_fixed_candidate_quantities(factor,change):
    p=problem();before=asdict(p)
    changed=perturb_problem(p,factor,change,candidate_id="a")
    assert asdict(p)==before
    assert [c.candidate_id for c in changed.candidates]==[c.candidate_id for c in p.candidates]
    assert [[o.quantity for o in c.options] for c in changed.candidates]==[[o.quantity for o in c.options] for c in p.candidates]
    if factor=="selected_route_cost":assert changed.candidates[1]==p.candidates[1]
    result=solve(changed)
    account=independently_account(changed,result["plan"])
    assert not account["violations"]
    for key in ("service","cost","quantity","vhs"):
        assert account[key]==pytest.approx(result[key])


@pytest.mark.parametrize("value",[math.nan,math.inf,-math.inf,.51,-.51,True])
def test_invalid_perturbation_is_rejected(value):
    with pytest.raises(ValueError):perturb_problem(problem(),"source_capacity",value)


def test_missing_or_unknown_factor_is_not_silently_applied():
    with pytest.raises(ValueError):perturb_problem(problem(),"selected_route_cost",.1,candidate_id="missing")
    with pytest.raises(ValueError):perturb_problem(problem(),"unknown",.1)


def test_changed_constraints_do_not_relabel_infeasible_old_plan_as_regret():
    p=perturb_problem(problem(),"source_capacity",-.5)
    result=compare_plans(p,{"a":3.},{"a":2.})
    assert not result["previous_plan_feasible"]
    assert result["service_difference"] is None and result["cost_difference_at_equal_service"] is None
    assert result["previous_plan_violations"]==["CAP:source:S"]


def test_unequal_service_does_not_become_cost_saving():
    result=compare_plans(problem(),{"a":2.},{"a":3.})
    assert result["service_difference"]==1
    assert result["cost_difference_at_equal_service"] is None


def test_infeasible_new_plan_fails_instead_of_scoring_it():
    with pytest.raises(ValueError):compare_plans(problem(),{}, {"a":3.,"b":3.})


def test_empty_plan_similarity_and_exact_match():
    row=compare_plans(problem(),{}, {})
    assert row["route_jaccard"]==1 and row["exact_plan_same"] and row["equivalent_objective"]


def test_finite_capacity_effect_can_be_zero_then_positive():
    rows=finite_marginals([1,2,3,4],[10,10,18,18])
    assert [x["value_difference"] for x in rows]==[0,8,0]
    assert [x["zero_effect"] for x in rows]==[True,False,True]
    assert not any(x["monotonicity_violation"] for x in rows)


def test_cost_effect_uses_opposite_direction_and_reports_violation():
    rows=finite_marginals([0,10,20],[50,40,45],increasing_resource=False)
    assert rows[0]["difference_per_unit"]==-1
    assert not rows[0]["monotonicity_violation"] and rows[1]["monotonicity_violation"]


@pytest.mark.parametrize("levels,values",[([1],[2]),([1,1],[2,3]),([1,2],[1]),([1,math.nan],[1,2]),([True,2],[1,2])])
def test_invalid_marginal_inputs_raise(levels,values):
    with pytest.raises(ValueError):finite_marginals(levels,values)
