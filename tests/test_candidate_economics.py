from dataclasses import replace
import math

import pytest

from services.candidate_economics import Money, PriceOption, solve_economics, solve_snapshot_economics, validate_problem
from services.candidate_economics_experiments import amount, brute_force_two_period, demo_problem


@pytest.mark.parametrize("name, expected", [("normal", 100), ("discount", 80), ("transfer", 118),
    ("mixed", 74), ("no_transfer", 36), ("expiry_block", 36), ("shared_vehicle", 90), ("multiperiod", 58)])
def test_hand_calculation(name, expected):
    r = solve_economics(demo_problem(name))
    assert r["status"] == "OPTIMAL"
    assert r["expected_net_value"] == pytest.approx(expected)
    assert r["constraint_violations"] == 0
    assert r["actual_savings"] is None and not r["production_approved"]


@pytest.mark.parametrize("name", ["normal", "discount", "transfer", "mixed", "no_transfer", "uncertainty"])
def test_independent_complete_enumeration(name):
    p = demo_problem(name)
    assert solve_economics(p)["expected_net_value"] == pytest.approx(brute_force_two_period(p)["net_value"])


def test_mixed_transfer_and_remainder_discount():
    r = solve_economics(demo_problem())
    assert sum(x["quantity"] for x in r["transfers"]) == pytest.approx(4)
    assert next(x for x in r["sales"] if x["market_id"] == "source")["quantity"] == pytest.approx(6)
    assert {x["level"] for x in r["sales"]} == {"normal", "discount_40pct"}


def test_no_double_count_purchase_or_markdown():
    p = demo_problem()
    p = replace(p, common_sunk_cost=Money(amount(999999).field, "CANCELS_COMMON", "already purchased, common"))
    r = solve_economics(p)
    assert r["expected_net_value"] == pytest.approx(6 * 6 + 4 * 10 - 2)
    assert "discount_loss" not in r  # reduced revenue is the only markdown charge


def test_integer_shared_vehicle_cost_once_per_trip():
    p = demo_problem("shared_vehicle")
    assert solve_economics(p)["transport_cost"] == pytest.approx(10)
    p = replace(p, routes=(replace(p.routes[0], vehicle_capacity=6),))
    r = solve_economics(p)
    assert r["vehicles"][0]["count"] == 2
    assert r["transport_cost"] == pytest.approx(20)
    assert r["expected_net_value"] == pytest.approx(80)


def test_expiry_blocks_transport_and_nonexpired_terminal_not_disposed():
    r = solve_economics(demo_problem("expiry_block"))
    assert r["transfers"] == []
    assert sum(x["disposal"] for x in r["inventory"]) == pytest.approx(4)
    r = solve_economics(demo_problem("no_transfer"))
    assert sum(x["disposal"] for x in r["inventory"]) == 0
    assert sum(x["quantity"] for x in r["inventory"] if x["period"] == 1) == pytest.approx(4)


def test_in_transit_cannot_sell_at_both_ends():
    p = demo_problem("multiperiod")
    r = solve_economics(p)
    for t in (0, 1):
        assert sum(x["quantity"] for x in r["inventory"] if x["period"] == t) == 0
    assert sum(x["quantity"] for x in r["sales"]) == 6
    assert sum(x["quantity"] for x in r["transfers"]) == 6


def test_storage_capacity_shared_across_products_at_arrival():
    p = demo_problem("shared_vehicle")
    p = replace(p, storage_capacity={"S": 20, "T": 7})
    r = solve_economics(p)
    assert sum(x["quantity"] for x in r["transfers"]) <= 7
    assert r["expected_net_value"] == pytest.approx(60)


def test_unit_conversion_preserves_economic_decision():
    p = demo_problem()
    lot = p.lots[0]
    converted = replace(p, lots=(replace(lot, unit="BOX", stock=1, quantum=.1, capacity_per_unit=10,
        holding_per_unit_period=amount(0, "BOX"), disposal_per_unit=amount(0, "BOX"), terminal_per_unit=amount(0, "BOX")),),
        markets=tuple(replace(m, unit="BOX", options=tuple(replace(o,
            unit_price=amount(o.unit_price.field.value * 10, "BOX"),
            demand={s: d / 10 for s, d in o.demand.items()}) for o in m.options)) for m in p.markets))
    r, converted_r = solve_economics(p), solve_economics(converted)
    assert r["expected_net_value"] == pytest.approx(converted_r["expected_net_value"])
    assert sum(x["quantity"] for x in r["transfers"]) == pytest.approx(10 * sum(x["quantity"] for x in converted_r["transfers"]))


def test_input_order_deterministic():
    p = demo_problem("shared_vehicle")
    r = solve_economics(p)
    q = replace(p, lots=tuple(reversed(p.lots)), markets=tuple(reversed(p.markets)))
    s = solve_economics(q)
    assert r["problem_signature"] == s["problem_signature"]
    assert r["transfers"] == s["transfers"]
    assert r["expected_net_value"] == s["expected_net_value"]


@pytest.mark.parametrize("treatment", ["UNKNOWN", "BOUNDED", "CANCELS_COMMON"])
def test_missing_differential_cost_never_zero(treatment):
    p = demo_problem()
    p = replace(p, routes=(replace(p.routes[0], fixed_per_vehicle=Money(treatment=treatment, lower=0, upper=20)),))
    r = solve_economics(p)
    assert r["status"] == "INPUT_REQUIRED"
    assert r["expected_net_value"] is None
    assert r["input_issues"][0]["classification"] == treatment


def test_not_applicable_requires_reason():
    p = demo_problem()
    p = replace(p, routes=(replace(p.routes[0], fixed_per_vehicle=Money(treatment="NOT_APPLICABLE")),))
    assert solve_economics(p)["status"] == "INPUT_REQUIRED"
    p = replace(p, routes=(replace(p.routes[0], fixed_per_vehicle=Money(treatment="NOT_APPLICABLE", reason="explicit free shared service")),))
    assert solve_economics(p)["expected_net_value"] == pytest.approx(76)


def test_nonanticipativity_and_risk_comparison():
    p = demo_problem("uncertainty")
    expected = solve_economics(p)
    worst = solve_economics(p, risk="worst_case")
    assert expected["expected_net_value"] == pytest.approx(51)
    assert expected["scenario_net_values"] == pytest.approx({"low": 26, "high": 76})
    assert sum(x["quantity"] for x in expected["transfers"]) == 5
    assert all("scenario" not in row for row in expected["transfers"] + expected["price_levels"])
    assert worst["expected_net_value"] == pytest.approx(30)
    assert worst["worst_net_value"] == pytest.approx(30)
    assert worst["transfers"] == []
    perfect = 0
    for s in p.scenarios:
        q = replace(p, scenarios=(replace(s, probability=1),), markets=tuple(replace(m,
            options=tuple(replace(o, demand={s.scenario_id: o.demand[s.scenario_id]}) for o in m.options)) for m in p.markets))
        perfect += s.probability * solve_economics(q)["expected_net_value"]
    assert perfect == pytest.approx(63)
    assert perfect > expected["expected_net_value"]  # forbidden clairvoyant benchmark, not feasible policy


def test_period_terminal_value_sensitivity():
    p = demo_problem("normal")
    p = replace(p, lots=(replace(p.lots[0], terminal_per_unit=amount(15)),))
    r = solve_economics(p)
    assert r["expected_net_value"] == pytest.approx(150)
    assert not r["sales"]  # deliberately exposes how a strong terminal assumption changes decisions


@pytest.mark.parametrize("change", ["unknown_state", "unit_mismatch", "duplicate_lot", "missing_capacity", "scope", "probability"])
def test_invalid_contract_fail_closed(change):
    p = demo_problem()
    if change == "unknown_state": p = replace(p, lots=(replace(p.lots[0], state="UNKNOWN"),))
    if change == "unit_mismatch": p = replace(p, markets=(replace(p.markets[0], unit="BOX"), p.markets[1]))
    if change == "duplicate_lot": p = replace(p, lots=(p.lots[0], p.lots[0]))
    if change == "missing_capacity": p = replace(p, storage_capacity={"S": 20})
    if change == "scope": p = replace(p, accounting_scope="SEPARATE_SELLERS_UNCONFIRMED")
    if change == "probability": p = replace(p, scenarios=(replace(p.scenarios[0], probability=.8),))
    assert solve_economics(p)["status"] == "INPUT_REQUIRED"


def test_lot_state_not_fungible():
    p = demo_problem("shared_vehicle")
    p = replace(p, lots=(replace(p.lots[0], state="FROZEN"), p.lots[1]),
                markets=(replace(p.markets[0], state="FROZEN"), p.markets[1]))
    r = solve_economics(p)
    assert {row["lot_id"] for row in r["transfers"]} == {"lot_Q"}
    assert r["expected_net_value"] == pytest.approx(40)


def test_sales_quantity_conservation_each_scenario():
    p = demo_problem("uncertainty")
    r = solve_economics(p)
    for s in p.scenarios:
        sold = sum(x["quantity"] for x in r["sales"] if x["scenario"] == s.scenario_id)
        remains = sum(x["quantity"] for x in r["inventory"] if x["scenario"] == s.scenario_id and x["period"] == p.horizon - 1)
        disposed = sum(x["disposal"] for x in r["inventory"] if x["scenario"] == s.scenario_id)
        assert sold + remains + disposed == pytest.approx(sum(l.stock for l in p.lots))


def test_unknown_sunk_cost_does_not_invent_absolute_loss():
    p = demo_problem()
    assert p.common_sunk_cost.field.value is None
    r = solve_economics(p)
    assert r["expected_net_value"] == pytest.approx(74)
    assert r["absolute_loss"] is None


def _snapshot():
    from services.operating_snapshot import OperationalSnapshot, StockPosition
    from services.target_need_contract import TargetNeed, OPERATOR_CONFIRMED_NEED, VERIFIED
    from services import operational_evidence as oe
    position = StockPosition("scenario", "network", "2026-10-16", "S", "P", "CHILLED", "lot_A",
        "EA", 10, 0, 3, 0, 0, "2026-10-19", {"source": "test scenario"})
    need = TargetNeed(OPERATOR_CONFIRMED_NEED, "network", "T", "P", "2026-10-16", 2,
        oe.DIRECT_REAL, dataset_id="scenario", valid_until="2026-10-17", planning_horizon_days=2,
        quantity_unit="EA", validation_status=VERIFIED)
    return OperationalSnapshot("scenario", "network", "2026-10-16", 2, (position,), (need,),
        "STRICT_OPERATOR", {}, {"approved_route_ids": ["S_T"]})


def test_shared_snapshot_adapter_applies_approved_caps():
    result = solve_snapshot_economics(_snapshot(), demo_problem())
    assert result["status"] == "OPTIMAL"
    assert sum(x["quantity"] for x in result["transfers"]) == 2
    assert result["expected_net_value"] == pytest.approx(54)
    assert result["snapshot_signature"]


@pytest.mark.parametrize("field,value", [("reserved", None), ("reserved", 2),
    ("inbound_confirmed", None), ("outbound_confirmed", 1), ("expiry_date", None),
    ("expiry_date", "2026-10-18"), ("lot", None), ("approved_transfer", None)])
def test_snapshot_adapter_unknown_or_unsupported_events_fail_closed(field, value):
    snap = _snapshot()
    snap = replace(snap, positions=(replace(snap.positions[0], **{field: value}),))
    result = solve_snapshot_economics(snap, demo_problem())
    assert result["status"] == "INPUT_REQUIRED"
    assert result["expected_net_value"] is None


def test_scenario_values_cannot_be_presented_as_real_model():
    p = demo_problem()
    p = replace(p, evidence_class="REAL_MODEL", lots=(replace(p.lots[0], approved_departure_qty=10),))
    result = solve_economics(p)
    assert result["status"] == "INPUT_REQUIRED"
    assert any(x["classification"] == "SCENARIO_NOT_REAL_EVIDENCE" for x in result["input_issues"])


def test_cross_dataset_costs_cannot_be_combined():
    p = demo_problem()
    route = p.routes[0]
    fixed = replace(route.fixed_per_vehicle, field=replace(route.fixed_per_vehicle.field, dataset="OTHER_BUSINESS"))
    result = solve_economics(replace(p, routes=(replace(route, fixed_per_vehicle=fixed),)))
    assert result["status"] == "INPUT_REQUIRED"
    assert any(x["classification"] == "CROSS_DATASET_ECONOMICS" for x in result["input_issues"])


def test_cash_bridge_and_period_quantity_with_in_transit():
    p = demo_problem("multiperiod")
    r = solve_economics(p)
    assert r["scenario_cash_components"]["base"] == pytest.approx({
        "external_sales_revenue": 60, "transport_cost": 2, "holding_cost": 0,
        "disposal_cost": 0, "terminal_value": 0, "net_cash_value": 58})
    for t in range(p.horizon):
        on_hand = sum(x["quantity"] for x in r["inventory"] if x["period"] == t)
        transit = sum(x["quantity"] for x in r["in_transit"] if x["period"] == t)
        cumulative_sold = sum(x["sold"] for x in r["market_outcomes"] if x["period"] <= t)
        cumulative_disposed = sum(x["disposal"] for x in r["inventory"] if x["period"] <= t)
        assert on_hand + transit + cumulative_sold + cumulative_disposed == pytest.approx(6)


def test_approved_source_and_target_limits_not_on_hand_assumption():
    p = demo_problem()
    p = replace(p, lots=(replace(p.lots[0], approved_departure_qty=1),),
                receipt_limits={("T", "P", "CHILLED"): 2})
    r = solve_economics(p)
    assert sum(x["quantity"] for x in r["transfers"]) == 1
    assert r["expected_net_value"] == pytest.approx(44)


@pytest.mark.parametrize("field,kind", [("reserved", "RESERVED_OUTBOUND"),
    ("inbound_confirmed", "CONFIRMED_INBOUND"), ("outbound_confirmed", "CONFIRMED_OUTBOUND")])
def test_snapshot_scheduled_flows_reconcile_nonzero_totals(field, kind):
    from services.candidate_economics import ConfirmedFlow
    snap = _snapshot()
    snap = replace(snap, positions=(replace(snap.positions[0], **{field: 1}),))
    p = replace(demo_problem(), confirmed_flows=(ConfirmedFlow("scheduled", "lot_A", "S", 1, 1, kind),))
    r = solve_snapshot_economics(snap, p)
    assert r["status"] == "OPTIMAL"
    assert r["constraint_violations"] == 0
    bad = replace(p, confirmed_flows=(replace(p.confirmed_flows[0], quantity=2),))
    rejected = solve_snapshot_economics(snap, bad)
    assert rejected["status"] == "INPUT_REQUIRED"
    assert any(i["classification"] == "PENDING_FLOW_SCHEDULE_MISMATCH" for i in rejected["input_issues"])
