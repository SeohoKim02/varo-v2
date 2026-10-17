from dataclasses import replace

import pytest

from services.candidate_economics import ConfirmedFlow, CommittedTransfer, DemandMarket, PriceOption, Scenario, Money, solve_economics
from services.candidate_economics_experiments import amount, demo_problem
from services.candidate_economics_extended import history_key
from services.candidate_economics_policies import (RealizedPath, evaluate_policies, evaluate_policy_on_path,
    fair_baselines, observed_prefix, scenario_paths)


def adaptive_problem():
    p = demo_problem("transfer")
    markets = (
        DemandMarket("target", "T", "P", "CHILLED", "EA", 2, (PriceOption("normal", amount(10), {"low": 0, "high": 10}),)),
        DemandMarket("source", "S", "P", "CHILLED", "EA", 2, (PriceOption("normal", amount(4), {"low": 10, "high": 0}),)),
    )
    return replace(p, problem_id="adaptive_signal", horizon=3, routes=(replace(p.routes[0], departure_periods=(0, 1)),),
        markets=markets, scenarios=(Scenario("low", .5, {0: "LOW"}), Scenario("high", .5, {0: "HIGH"})))


def test_scheduled_incoming_is_not_present_before_arrival():
    p = demo_problem("normal")
    p = replace(p, lots=(replace(p.lots[0], stock=0),),
        confirmed_flows=(ConfirmedFlow("receipt", "lot_A", "S", 1, 10, "CONFIRMED_INBOUND"),))
    r = solve_economics(p)
    assert r["status"] == "OPTIMAL"
    assert r["sales"] == []  # source demand was period 0, arrival is period 1
    assert sum(row["quantity"] for row in r["inventory"] if row["period"] == 1) == 10


def test_reserved_outgoing_cannot_be_cancelled_for_profitable_sales():
    p = demo_problem("normal")
    p = replace(p, confirmed_flows=(ConfirmedFlow("order", "lot_A", "S", 1, 4, "RESERVED_OUTBOUND"),))
    r = solve_economics(p)
    assert r["status"] == "OPTIMAL"
    assert sum(row["quantity"] for row in r["sales"]) == 6
    assert r["expected_net_value"] == 60


def test_already_in_transit_not_counted_as_source_stock_and_sunk_freight_not_recharged():
    p = demo_problem("multiperiod")
    p = replace(p, lots=(replace(p.lots[0], stock=0),),
        committed_transfers=(CommittedTransfer("already_departed", "lot_A", "S_T", -1, 6, cost_already_paid=True),))
    r = solve_economics(p)
    assert r["status"] == "OPTIMAL"
    assert r["transport_cost"] == 0
    assert sum(row["quantity"] for row in r["in_transit"] if row["period"] == 0) == 6
    assert sum(row["quantity"] for row in r["sales"]) == 6
    assert r["expected_net_value"] == 60


def test_future_committed_transfer_is_fixed_even_when_optional_transfers_disallowed():
    p = demo_problem("transfer")
    p = replace(p, committed_transfers=(CommittedTransfer("approved", "lot_A", "S_T", 0, 4),))
    r = solve_economics(p, allow_transfers=False)
    assert r["status"] == "OPTIMAL"
    assert r["transfers"] == []
    assert sum(row["quantity"] for row in r["sales"]) == 4
    assert r["expected_net_value"] == 46


def test_handling_processing_period_holding_and_cash_accounting():
    p = demo_problem("transfer")
    p = replace(p, handling_per_capacity={"S": amount(1, "KG"), "T": amount(2, "KG")},
        processing_capacity={("T", 1): 10}, holding_by_period={("lot_A", 0): amount(1)})
    r = solve_economics(p)
    assert r["status"] == "OPTIMAL"
    # T capacity counts inbound + sale = 2 per unit. 5 transferred, 5 held at S.
    assert sum(row["quantity"] for row in r["transfers"]) == 5
    parts = r["accounting_components"]["base"]
    assert parts == pytest.approx({"external_sales_revenue": 60, "transport_cost": 2,
        "handling_cost": 25, "holding_cost": 5, "disposal_cost": 0, "terminal_value": 0,
        "justified_penalty": 0, "net_cash_value": 28})


def test_known_zero_and_unknown_handling_are_different():
    p = demo_problem()
    assert solve_economics(replace(p, handling_per_capacity={"S": amount(0, "KG")}))["expected_net_value"] == 74
    r = solve_economics(replace(p, handling_per_capacity={"S": Money(reason="quote needed")}))
    assert r["status"] == "INPUT_REQUIRED"
    assert r["expected_net_value"] is None


@pytest.mark.parametrize("change", ["duplicate", "forecast", "fraction", "expired", "unit", "period"])
def test_event_and_cost_contracts_fail_closed(change):
    p = demo_problem()
    event = ConfirmedFlow("receipt", "lot_A", "S", 1, 2, "CONFIRMED_INBOUND")
    p = replace(p, confirmed_flows=(event,))
    if change == "duplicate": p = replace(p, confirmed_flows=(event, event))
    if change == "forecast": p = replace(p, confirmed_flows=(replace(event, kind="FORECAST_INBOUND"),))
    if change == "fraction": p = replace(p, confirmed_flows=(replace(event, quantity=.5),))
    if change == "expired": p = replace(p, lots=(replace(p.lots[0], expiry_period=0),))
    if change == "unit": p = replace(p, handling_per_capacity={"S": amount(1, "EA")})
    if change == "period": p = replace(p, period_days=0)
    assert solve_economics(p)["status"] == "INPUT_REQUIRED"


def test_infeasible_irrevocable_order_remains_infeasible_not_cancelled():
    p = demo_problem()
    r = solve_economics(replace(p, confirmed_flows=(ConfirmedFlow("order", "lot_A", "S", 0, 11, "RESERVED_OUTBOUND"),)))
    assert r["status"] == "NO_SOLUTION"
    assert r["expected_net_value"] is None


def test_penalty_requires_evidence_reason_and_is_not_invented():
    p = demo_problem("transfer")
    penalty = amount(3)
    assert solve_economics(replace(p, shortage_penalties={"target": penalty}))["status"] == "INPUT_REQUIRED"
    result = solve_economics(replace(p, shortage_penalties={"target": replace(penalty, reason="SCENARIO: contractual shortfall fee")}))
    assert result["accounting_components"]["base"]["justified_penalty"] == 0


@pytest.mark.parametrize("name", ["normal", "discount", "transfer", "mixed", "no_transfer", "expiry_block", "shared_vehicle", "multiperiod", "uncertainty"])
def test_fair_baselines_same_inputs_and_joint_not_worse_than_feasible_treatments(name):
    p = demo_problem(name)
    report = fair_baselines(p)
    assert report["status"] == "COMPLETED"
    values = [v["expected_net_value"] for v in report["results"].values()]
    assert all(v is not None for v in values)
    assert values[-1] + 1e-6 >= max(values[:-1])
    assert all(v.get("constraint_violations", 0) == 0 for v in report["results"].values())


def test_joint_vs_no_transfer_does_not_fabricate_joint_advantage_over_sequential():
    r = fair_baselines(demo_problem("mixed"))["results"]
    assert r["A"]["expected_net_value"] == 20
    assert r["B"]["expected_net_value"] == 36
    assert r["C"]["expected_net_value"] == r["D"]["expected_net_value"] == r["E"]["expected_net_value"] == 74


def test_multistage_adapts_after_signal_not_before():
    p = adaptive_problem()
    r = solve_economics(p, policy="MULTISTAGE_RECOURSE")
    assert r["status"] == "OPTIMAL"
    assert r["expected_net_value"] == 69
    assert r["constraint_violations"] == 0
    # Wait for period 0 signal; high branch moves ten at t=1, low does not.
    assert {(x["scenario"], x["departure_period"], x["quantity"]) for x in r["scenario_transfers"]} == {("high", 1, 10)}
    assert history_key(p.scenarios[0], 0) == history_key(p.scenarios[1], 0)
    assert history_key(p.scenarios[0], 1) != history_key(p.scenarios[1], 1)


def test_same_history_future_scenarios_have_same_early_actions():
    p = adaptive_problem()
    p = replace(p, scenarios=(Scenario("low", .5, {1: "LOW"}), Scenario("high", .5, {1: "HIGH"})))
    r = solve_economics(p, policy="MULTISTAGE_RECOURSE")
    by_scenario = {s.scenario_id: {(x["route_id"], x["departure_period"]): x["quantity"]
        for x in r["scenario_transfers"] if x["scenario"] == s.scenario_id} for s in p.scenarios}
    assert by_scenario["low"] == by_scenario["high"]


def test_observation_loader_refuses_future_demand():
    p = adaptive_problem()
    prefix = observed_prefix(p, scenario_paths(p)[0], 0)
    assert prefix.demand == {}
    with pytest.raises(ValueError, match="FUTURE_OBSERVATION"):
        prefix.demand_at("target")


@pytest.mark.parametrize("policy", ["OPEN_LOOP", "ROLLING_HORIZON", "MULTISTAGE_RECOURSE"])
def test_realised_future_change_cannot_change_earlier_decisions(policy):
    p = adaptive_problem()
    original = scenario_paths(p)[1]
    changed = replace(original, path_id="changed_future", demand={"target": {"normal": 8}, "source": {"normal": 2}})
    a = evaluate_policy_on_path(p, policy, original)
    b = evaluate_policy_on_path(p, policy, changed)
    assert a["status"] == b["status"] == "COMPLETED"
    assert [x["moves"] for x in a["trajectory"][:2]] == [x["moves"] for x in b["trajectory"][:2]]
    assert [x["prices"] for x in a["trajectory"][:2]] == [x["prices"] for x in b["trajectory"][:2]]


def test_four_policies_same_exogenous_paths_oracle_separate():
    r = evaluate_policies(adaptive_problem())
    assert all(row["status"] == "COMPLETED" for row in r["results"].values())
    assert r["results"]["OPEN_LOOP"]["expected_net_value"] == 48
    for policy in ("ROLLING_HORIZON", "MULTISTAGE_RECOURSE", "PERFECT_INFORMATION"):
        assert r["results"][policy]["expected_net_value"] == 69
    assert not r["results"]["PERFECT_INFORMATION"]["operational_policy"]
    assert not r["production_approved"]


def test_unseen_observation_is_not_silently_replaced_by_oracle():
    p = adaptive_problem()
    path = replace(scenario_paths(p)[0], signals={0: "UNSEEN"})
    r = evaluate_policy_on_path(p, "ROLLING_HORIZON", path)
    assert r["status"] == "INPUT_REQUIRED"
    assert r["net_cash_value"] is None


def test_cross_dataset_new_handling_cost_cannot_be_mixed():
    p = demo_problem()
    cost = amount(1, "KG")
    cost = replace(cost, field=replace(cost.field, dataset="UNRELATED_BUSINESS"))
    r = solve_economics(replace(p, handling_per_capacity={"S": cost}))
    assert r["status"] == "INPUT_REQUIRED"
    assert any(x["classification"] == "CROSS_DATASET_ECONOMICS" for x in r["input_issues"])


@pytest.mark.parametrize("change", ["period", "vehicles"])
def test_immutable_transfer_still_obeys_physical_vehicle_policy(change):
    p = demo_problem("multiperiod")
    e = CommittedTransfer("committed", "lot_A", "S_T", 0, 6)
    if change == "period": p = replace(p, routes=(replace(p.routes[0], departure_periods=(1,)),))
    if change == "vehicles": e = replace(e, vehicle_count=3)
    r = solve_economics(replace(p, committed_transfers=(e,)))
    assert r["status"] == "INPUT_REQUIRED"


@pytest.mark.parametrize("fixed", [{("lot_A", "BAD_ROUTE", 0): 1}, {("lot_A", "S_T", 0): .2},
    {("lot_A", "S_T", 0): float("nan")}, {("lot_A", "S_T", 0): -1}])
def test_invalid_fixed_transfer_is_not_silently_ignored(fixed):
    r = solve_economics(demo_problem(), fixed_transfers=fixed)
    assert r["status"] == "INPUT_REQUIRED"


def test_unselected_price_response_is_not_a_signal_for_sales():
    p = adaptive_problem()
    p = replace(p, routes=(), markets=(DemandMarket("now", "S", "P", "CHILLED", "EA", 0, (
        PriceOption("normal", amount(6), {"low": 5, "high": 5}),
        PriceOption("unused", amount(0), {"low": 0, "high": 100}))),
        DemandMarket("later", "S", "P", "CHILLED", "EA", 2,
            (PriceOption("normal", amount(10), {"low": 0, "high": 10}),))),
        scenarios=(Scenario("low", .5), Scenario("high", .5)))
    r = solve_economics(p, policy="MULTISTAGE_RECOURSE", fixed_prices={("now", 0): "normal"})
    assert r["status"] == "OPTIMAL"
    now = {s.scenario_id: sum(x["quantity"] for x in r["sales"] if x["scenario"] == s.scenario_id and x["market_id"] == "now") for s in p.scenarios}
    assert now["low"] == now["high"]


def _two_lot_future_problem():
    p = adaptive_problem()
    a = replace(p.lots[0], lot_id="soon", stock=5, expiry_period=1)
    b = replace(p.lots[0], lot_id="later", stock=5, expiry_period=3, holding_per_unit_period=amount(1))
    return replace(p, lots=(a, b), routes=(), markets=(
        DemandMarket("now", "S", "P", "CHILLED", "EA", 0,
            (PriceOption("normal", amount(6), {"low": 5, "high": 5}),)),
        DemandMarket("future", "S", "P", "CHILLED", "EA", 2,
            (PriceOption("normal", amount(20), {"low": 0, "high": 10}),))),
        scenarios=(Scenario("low", .5, {0: "LOW"}), Scenario("high", .5, {0: "HIGH"})))


def test_end_of_period_signal_cannot_choose_current_lot_sales_or_inventory():
    p = _two_lot_future_problem()
    r = solve_economics(p, policy="MULTISTAGE_RECOURSE")
    assert r["status"] == "OPTIMAL"
    allocation = {s.scenario_id: {lid: sum(x["quantity"] for x in r["sales"]
        if x["scenario"] == s.scenario_id and x["market_id"] == "now" and x["lot_id"] == lid)
        for lid in ("soon", "later")} for s in p.scenarios}
    assert allocation["low"] == allocation["high"]
    inventories = {s.scenario_id: [(x["lot_id"], x["location"], x["quantity"], x["disposal"])
        for x in r["inventory"] if x["scenario"] == s.scenario_id and x["period"] == 0] for s in p.scenarios}
    assert inventories["low"] == inventories["high"]


@pytest.mark.parametrize("policy", ["OPEN_LOOP", "ROLLING_HORIZON", "MULTISTAGE_RECOURSE"])
def test_playback_does_not_use_end_of_period_signal_for_current_lot_allocation(policy):
    p = _two_lot_future_problem()
    paths = scenario_paths(p)
    a = evaluate_policy_on_path(p, policy, paths[0])
    b = evaluate_policy_on_path(p, policy, paths[1])
    assert a["status"] == b["status"] == "COMPLETED"
    assert a["trajectory"][0]["sales"] == b["trajectory"][0]["sales"]
    first = lambda r: [(x["lot_id"], x["location"], x["quantity"], x["disposal"]) for x in r["trajectory"][0]["inventory"]]
    assert first(a) == first(b)
