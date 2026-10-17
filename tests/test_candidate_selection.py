from dataclasses import replace
import random
import pytest
from services.candidate_selection import Candidate, AllocationOption as O, SelectionProblem as P, solve, evaluate, exhaustive


def problem():
    return P("small", {"s": 6, "t1": 4, "t2": 4}, {"t1": 4, "t2": 4}, (
        Candidate("a", "p", "s", "t1", "C", "L", "EA", {"s": 1, "t1": 1}, "t1", tuple(O(q, 2+q) for q in range(1,5)), 10),
        Candidate("b", "p", "s", "t2", "C", "L", "EA", {"s": 1, "t2": 1}, "t2", tuple(O(q, 1+q) for q in range(1,5)), 20)), 2, "SCENARIO")


@pytest.mark.parametrize("algo", ["VARO", "PARTIAL", "LOCAL", "VHS", "NO_VHS", "COST_ONLY", "MILP"])
def test_all_selectors_same_caps(algo):
    result = solve(problem(), algo)
    assert not result["violations"] and result["quantity"] <= 6


def test_partial_gains_without_cost_prorating():
    p = problem()
    assert solve(p, "VARO")["quantity"] == 4
    r = solve(p, "PARTIAL")
    assert r["quantity"] == 6 and r["cost"] == 9


def test_milp_matches_exhaustive():
    p = problem()
    a, b = solve(p), evaluate(p, exhaustive(p))
    assert a["solver"]["optimality_proven"]
    assert (a["service"], a["cost"], a["vhs"]) == (b["service"], b["cost"], b["vhs"])


@pytest.mark.parametrize("seed", range(6))
def test_random_small_enumeration(seed):
    rng = random.Random(seed)
    p = problem()
    cs = tuple(replace(c, options=tuple(O(o.quantity, rng.randint(0,10)) for o in c.options), vhs=rng.randint(0,10)) for c in p.candidates)
    p = replace(p, candidates=cs)
    a,b = solve(p), evaluate(p, exhaustive(p))
    assert a["service"] == pytest.approx(b["service"])
    assert a["cost"] == pytest.approx(b["cost"])
    assert a["vhs"] == pytest.approx(b["vhs"])


def test_signature_separation_and_order():
    p = problem(); q = replace(p, candidates=p.candidates[::-1])
    assert p.base_problem_signature == q.base_problem_signature
    assert p.candidate_set_signature == q.candidate_set_signature
    q = replace(p, candidates=p.candidates[:1])
    assert p.base_problem_signature == q.base_problem_signature
    assert p.candidate_set_signature != q.candidate_set_signature


def test_ea_box_equivalent_physical_conclusion():
    p = problem()
    box = replace(p, capacities={k:v/10 for k,v in p.capacities.items()}, needs={k:v/10 for k,v in p.needs.items()},
        candidates=tuple(replace(c, unit="BOX",
                                options=tuple(O(o.quantity/10,o.cost) for o in c.options)) for c in p.candidates))
    a,b = solve(p),solve(box)
    assert a["service"] == pytest.approx(b["service"])
    assert a["cost"] == b["cost"]
    assert a["plan"] == pytest.approx({k:v*10 for k,v in b["plan"].items()})


def test_unknown_cost_is_not_free():
    p = problem(); c = replace(p.candidates[0], options=(O(1,float("nan")),))
    with pytest.raises(ValueError, match="cost"):
        solve(replace(p,candidates=(c,)))


@pytest.mark.parametrize("unit", ["UNKNOWN", "UNDECLARED", "UNCONFIRMED"])
def test_unknown_unit_rejected_even_if_input_verified(unit):
    p=problem()
    with pytest.raises(ValueError,match="state/unit"):
        solve(replace(p,evidence_class="VERIFIED_INPUT",candidates=(replace(p.candidates[0],unit=unit),)))


def test_one_product_conversion_does_not_change_tie_decision():
    p=P("units",{"a":10,"b":10},{"a":10,"b":10},(
        Candidate("a","A","S","A","C","L1","EA",{"a":1},"a",(O(10,1),),2),
        Candidate("b","B","S","B","C","L2","EA",{"b":1},"b",(O(10,1),),3)),1,"SCENARIO")
    box=replace(p,capacities={"a":10,"b":1},needs={"a":10,"b":1},
        candidates=(p.candidates[0],replace(p.candidates[1],unit="BOX",options=(O(1,1),))))
    assert solve(p)["plan"]=={"b":10}
    assert solve(box)["plan"]=={"b":1}


def test_shared_target_and_dc_capacity():
    p=problem(); cs=tuple(replace(c,resources={**c.resources,"dc":2}) for c in p.candidates)
    r=solve(replace(p,candidates=cs,capacities={**p.capacities,"dc":8}))
    assert r["quantity"]==4 and r["resource_used"]["dc"]==8


def test_economic_selector_requires_declared_value():
    with pytest.raises(ValueError,match="evidence"):
        solve(problem(),"ECONOMIC")


def test_no_invalid_plan_is_approved_on_timeout():
    r=solve(problem(), time_limit=0.000001)
    assert not r["violations"] and not r["solver"]["optimality_proven"]
