from dataclasses import replace
from itertools import product
import math

import pytest

from services.research_procurement import ProcurementScenario, build_problem, compare_procurement, order_quantities


def independent_enumeration(s):
    """Integer conservation/revenue enumeration, not MILP variables or rows."""
    values = {}
    for q in range(s.max_order + 1):
        cost = 0 if q == 0 else q * s.purchase_unit_cost + s.purchase_fixed_cost
        if (0 < q < s.minimum_order) or cost > s.purchase_budget or q > s.target_storage:
            continue
        local, joint = -math.inf, -math.inf
        plans = {}
        for moved in range(min(s.source_stock, s.transfer_capacity) + 1):
            early_available = moved + (q if s.arrival_period <= 1 else 0)
            if early_available > s.target_storage:
                continue
            for (pa, da), (p1, d1), (p2, d2) in product(
                    ((10, s.source_normal_demand), (6, s.source_discount_demand)),
                    ((10, s.target_early_demand), (7, s.target_early_demand + 2)),
                    ((10, s.target_late_demand), (7, s.target_late_demand + 2))):
                a_sold = min(s.source_stock - moved, da)
                for early_sold in range(min(early_available, d1) + 1):
                    late_available = early_available - early_sold + (q if s.arrival_period == 2 else 0)
                    if late_available > s.target_storage:
                        continue
                    late_sold = min(late_available, d2)
                    value = pa * a_sold + p1 * early_sold + p2 * late_sold - cost - (s.transfer_fixed_cost if moved else 0)
                    joint = max(joint, value)
                    if moved == 0:
                        local = max(local, value)
                    if value not in plans:
                        plans[value] = {"order": q, "move": moved, "source_sold": a_sold,
                            "early_sold": early_sold, "late_sold": late_sold,
                            "source_price": pa, "early_price": p1, "late_price": p2,
                            "purchase_cost": cost, "transfer_cost": s.transfer_fixed_cost if moved else 0}
        values[q] = {"local": local, "joint": joint, "plan": plans.get(joint)}
    localq = max(values, key=lambda q: (values[q]["local"], -q))
    jointq = max(values, key=lambda q: (values[q]["joint"], -q))
    return {"joint_value": values[jointq]["joint"], "sequential_value": values[localq]["joint"],
            "joint_order": jointq, "sequential_order": localq, "plan": values[jointq]["plan"], "values": values}


@pytest.mark.parametrize("lead,moq,unit,freight", tuple(product((0, 1, 2), (1, 3), (3, 8), (2,))))
def test_existing_solver_matches_independent_enumeration(lead, moq, unit, freight):
    s = ProcurementScenario(arrival_period=lead, minimum_order=moq, purchase_unit_cost=unit, transfer_fixed_cost=freight)
    r, ref = compare_procurement(s), independent_enumeration(s)
    assert r["joint"]["net_value"] == pytest.approx(ref["joint_value"])
    assert r["sequential"]["net_value"] == pytest.approx(ref["sequential_value"])
    assert r["joint"]["order_quantity"] == ref["joint_order"]
    assert r["sequential"]["order_quantity"] == ref["sequential_order"]
    assert r["evidence_class"] == "SCENARIO" and not r["production_approved"]
    assert r["actual_savings"] is None


def test_purchase_cost_is_explicit_and_charged_once():
    r = compare_procurement(ProcurementScenario(purchase_fixed_cost=2))
    for row in r["candidates"]:
        for mode in ("joint", "local"):
            if row[mode] is not None:
                v = row[mode]
                assert v["net_value"] == pytest.approx(v["accounting"]["net_cash_value"] - v["purchase_cost"])


def test_budget_includes_fixed_fee_and_minimum_is_not_package_multiple():
    assert order_quantities(ProcurementScenario(minimum_order=3, purchase_unit_cost=2,
                                               purchase_fixed_cost=3, purchase_budget=11)) == (0, 3, 4)


def test_late_receipt_cannot_supply_early_sale():
    s = ProcurementScenario(source_stock=0, arrival_period=2, target_late_demand=0)
    r = compare_procurement(s)
    assert not [x for x in r["joint"]["sales"] if x["market_id"] == "early"]


@pytest.mark.parametrize("changes", [dict(purchase_unit_cost=None), dict(purchase_budget=math.inf),
    dict(purchase_fixed_cost=-1), dict(arrival_period=3), dict(minimum_order=0),
    dict(max_order=9), dict(source_stock=1.5), dict(transfer_capacity=0), dict(source_stock=True)])
def test_unsupported_or_unknown_input_rejected(changes):
    with pytest.raises(ValueError):
        compare_procurement(replace(ProcurementScenario(), **changes))


def test_unapproved_order_rejected_before_constructing_receipt():
    with pytest.raises(ValueError):
        build_problem(ProcurementScenario(purchase_budget=0), 1)


def test_existing_objects_are_immutable_and_receipt_is_only_hypothetical():
    s = ProcurementScenario()
    p = build_problem(s, 2)
    assert p.evidence_class == "SCENARIO"
    assert p.lots[1].stock == 0
    assert "hypothetical" in p.lineage["receipt"]
    assert p.confirmed_flows[0].period == 1
    assert s == ProcurementScenario()
