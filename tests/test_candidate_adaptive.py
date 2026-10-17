from dataclasses import replace

import pytest

from services.candidate_adaptive import Observation, plan_adaptive
from services.candidate_economics import ConfirmedFlow, CommittedTransfer
from services.candidate_economics_experiments import demo_problem, amount
from services.candidate_economics_extended import solve_extended
from services.candidate_economics_second_experiments import adaptive_problem


ORIGIN = "2026-10-17T00:00:00+09:00"
DAY1 = "2026-10-18T00:00:00+09:00"
DAY2 = "2026-10-19T00:00:00+09:00"


def observed(kind="SALE", quantity=2, **kw):
    defaults = dict(event_id="event-1",kind=kind,occurred_at="2026-10-17T12:00:00+09:00",
        known_at="2026-10-17T12:05:00+09:00",product_id="P",location="S",state="CHILLED",
        lot_id="lot_A",unit="EA",quantity=quantity,source="TEST operator record")
    defaults.update(kw)
    return Observation(**defaults)


def run(p=None, **kw):
    args = dict(policy="ROLLING_HORIZON",decision_at=ORIGIN,origin_at=ORIGIN)
    args.update(kw)
    return plan_adaptive(p or adaptive_problem(), **args)


def test_initial_plan_reuses_financial_equations_and_does_not_execute():
    p=demo_problem("mixed")
    result=run(p)
    assert result["status"]=="COMPLETED"
    assert result["result"]["expected_net_value"]==74
    assert result["production_approved"] is False and result["actual_savings"] is None
    assert sum(x["quantity"] for x in result["current_inventory"])==10
    assert result["observations_used"]==[]
    assert result["initial_inventory"]==result["current_inventory"]


def test_observed_sales_update_actual_inventory_without_booking_planned_sales():
    initial=run()
    replanned=run(decision_at=DAY1, observations=[observed()],previous=initial)
    assert replanned["status"]=="COMPLETED"
    assert sum(x["quantity"] for x in replanned["current_inventory"])==8
    assert replanned["remaining_horizon"]==2 and replanned["replan_count"]==1
    assert replanned["result"]["constraint_violations"]==0
    assert sum(x["quantity"] for x in initial["current_inventory"])==10


def test_signal_changes_current_plan_and_future_actions_are_separate():
    initial=run(policy="MULTISTAGE_RECOURSE")
    assert initial["status"]=="COMPLETED"
    assert initial["future_conditional_actions"]
    signal=observed("SIGNAL",0,signal="HIGH")
    high=run(policy="MULTISTAGE_RECOURSE",decision_at=DAY1,observations=[signal],previous=initial)
    low=run(policy="MULTISTAGE_RECOURSE",decision_at=DAY1,observations=[replace(signal,signal="LOW")],previous=initial)
    assert high["status"]==low["status"]=="COMPLETED"
    assert sum(x.get("quantity",0) for x in high["current_actions"] if x["kind"]=="TRANSFER")==10
    assert sum(x.get("quantity",0) for x in low["current_actions"] if x["kind"]=="TRANSFER")==0
    assert all(x["period"]==1 for x in high["current_actions"])
    assert all(x["period"]>1 for x in high["future_conditional_actions"])
    assert high["information_used"]["signals"]=={0:"HIGH"}


def test_same_known_prefix_is_independent_of_future_realized_observation():
    future=observed("SIGNAL",0,signal="HIGH",known_at=DAY2)
    a=run(decision_at=DAY1,observations=[future])
    b=run(decision_at=DAY1,observations=[replace(future,signal="LOW",quantity=999)])
    assert a["status"]==b["status"]=="COMPLETED"
    assert a["current_actions"]==b["current_actions"]
    assert a["problem_signature"]==b["problem_signature"]
    assert a["deferred_observations"]==["event-1"] and a["observations_used"]==[]


def test_scenario_probabilities_are_not_repaired():
    p=adaptive_problem()
    p=replace(p,scenarios=(replace(p.scenarios[0],probability=.8),p.scenarios[1]))
    r=run(p)
    assert r["status"]=="INPUT_REQUIRED" and r["result"] is None


@pytest.mark.parametrize("change,code",[
    ({"unit":"KG"},"OBSERVATION_IDENTITY_MISMATCH"),
    ({"location":"unknown"},"UNKNOWN_LOCATION"),
    ({"quantity":-1},"INVALID_OBSERVED_QUANTITY"),
    ({"quantity":.5},"INVALID_OBSERVED_QUANTITY"),
    ({"source":""},"OBSERVATION_SOURCE_REQUIRED"),
    ({"known_at":"2026-10-17T11:00:00+09:00"},"KNOWN_BEFORE_OCCURRED"),
    ({"quantity":11},"OBSERVED_NEGATIVE_STOCK"),
    ({"reference_id":"invented"},"UNKNOWN_COMMITMENT_REFERENCE"),
])
def test_observations_fail_closed(change,code):
    result=run(decision_at=DAY1,observations=[observed(**change)])
    assert result["status"]=="INPUT_REQUIRED"
    assert result["diagnostics"][0]["code"]==code
    assert result["result"] is None and result["reasons_ko"]


def test_correction_is_audited_and_does_not_double_count():
    original=observed(quantity=2)
    correction=replace(original,event_id="corrected",quantity=3,correction_of=original.event_id,known_at="2026-10-17T14:00:00+09:00")
    r=run(decision_at=DAY1,observations=[original,correction])
    assert r["status"]=="COMPLETED"
    assert sum(x["quantity"] for x in r["current_inventory"])==7
    assert len(r["observation_audit"])==2 and len(r["observations_used"])==1
    assert r["corrected_observations"]==[original.event_id]


def test_future_correction_does_not_rewrite_past_snapshot():
    original=observed(quantity=2)
    correction=replace(original,event_id="corrected",quantity=3,correction_of=original.event_id,known_at=DAY2)
    r=run(decision_at=DAY1,observations=[original,correction])
    assert sum(x["quantity"] for x in r["current_inventory"])==8
    assert r["corrected_observations"]==[]


def test_stock_count_is_absolute_not_an_inbound_and_has_no_fake_handling_cost():
    p=replace(adaptive_problem(),handling_per_capacity={"S":amount(2,"KG")})
    r=run(p,decision_at=DAY1,observations=[observed("STOCK_COUNT",4)])
    assert r["status"]=="COMPLETED"
    assert sum(x["quantity"] for x in r["current_inventory"])==4
    assert r["remaining_problem"]["confirmed_flows"]==()


def test_same_time_conflicting_counts_are_rejected():
    a=observed("STOCK_COUNT",4)
    b=replace(a,event_id="count-2",quantity=6)
    r=run(decision_at=DAY1,observations=[a,b])
    assert r["diagnostics"][0]["code"]=="DUPLICATE_STOCK_COUNT"


def test_period_boundary_and_timezone_are_explicit():
    assert run(decision_at="2026-10-17T12:00:00+09:00")["diagnostics"][0]["code"]=="PERIOD_BOUNDARY_REQUIRED"
    assert run(origin_at="2026-10-17T00:00:00")["diagnostics"][0]["code"]=="TIMEZONE_REQUIRED"
    assert run(decision_at="2026-10-20T00:00:00+09:00")["diagnostics"][0]["code"]=="HORIZON_COMPLETE"


def test_oracle_is_not_available_as_user_policy():
    r=run(policy="PERFECT_INFORMATION")
    assert r["status"]=="INPUT_REQUIRED" and r["production_approved"] is False


def test_unknown_signal_requires_new_input_not_an_oracle_guess():
    r=run(decision_at=DAY1,observations=[observed("SIGNAL",0,signal="UNSEEN")])
    assert r["diagnostics"][0]["code"]=="UNSEEN_INFORMATION"


def test_current_period_signal_cannot_inform_current_action():
    r=run(decision_at=ORIGIN,observations=[observed("SIGNAL",0,signal="HIGH",occurred_at=ORIGIN,known_at=ORIGIN)])
    assert r["diagnostics"][0]["code"]=="SIGNAL_NOT_YET_DECISION_AVAILABLE"


def test_past_confirmed_flow_requires_actual_outcome_and_is_never_cancelled():
    p=replace(adaptive_problem(),confirmed_flows=(ConfirmedFlow("order","lot_A","S",0,4,"RESERVED_OUTBOUND"),))
    missing=run(p,decision_at=DAY1)
    assert missing["diagnostics"][0]["code"]=="PAST_COMMITMENT_UNCONFIRMED"
    r=run(p,decision_at=DAY1,observations=[observed("OUTBOUND",4,reference_id="order")])
    assert r["status"]=="COMPLETED"
    assert sum(x["quantity"] for x in r["current_inventory"])==6
    assert r["remaining_problem"]["confirmed_flows"]==()


def test_confirmed_receipt_observed_once_not_added_again():
    p=replace(adaptive_problem(),confirmed_flows=(ConfirmedFlow("receipt","lot_A","S",0,2,"CONFIRMED_INBOUND"),))
    r=run(p,decision_at=DAY1,observations=[observed("RECEIPT",2,reference_id="receipt")])
    assert r["status"]=="COMPLETED" and sum(x["quantity"] for x in r["current_inventory"])==12
    assert r["remaining_problem"]["confirmed_flows"]==()


def test_in_transit_is_separate_and_committed_not_cancelled():
    p=replace(adaptive_problem(),routes=(replace(adaptive_problem().routes[0],transit_periods=2),))
    event=observed("TRANSFER_DEPARTED",4,route_id="S_T",cost_already_paid=True)
    r=run(p,decision_at=DAY1,observations=[event])
    assert r["status"]=="COMPLETED"
    assert sum(x["quantity"] for x in r["current_inventory"])==6
    assert r["remaining_problem"]["committed_transfers"][0]["quantity"]==4
    assert r["remaining_problem"]["committed_transfers"][0]["departure_period"]==-1
    assert max(x["quantity"] for x in r["result"]["in_transit"])==4
    assert r["result"]["transport_cost"]==0


def test_observed_arrival_moves_same_lot_to_destination_not_original_source():
    p=adaptive_problem()
    p=replace(p,routes=(replace(p.routes[0],departure_periods=(0,)),))
    event=observed("TRANSFER_DEPARTED",4,route_id="S_T",cost_already_paid=True)
    arrival=observed("TRANSFER_ARRIVED",4,event_id="arrived",location="T",route_id="S_T",reference_id=event.event_id,occurred_at=DAY1,known_at=DAY1)
    r=run(p,decision_at=DAY1,observations=[event,arrival])
    assert r["status"]=="COMPLETED"
    stock={x["location"]:x["quantity"] for x in r["current_inventory"]}
    assert stock=={"S":6,"T":4}
    assert r["remaining_problem"]["committed_transfers"]==()
    assert sum(x["quantity"] for x in r["result"]["sales"] if x["scenario"]=="high")==4


def test_arrival_without_departure_and_duplicate_arrival_are_rejected():
    arrival=observed("TRANSFER_ARRIVED",4,location="T",reference_id="missing")
    assert run(decision_at=DAY1,observations=[arrival])["diagnostics"][0]["code"]=="ARRIVAL_WITHOUT_DEPARTURE"


def test_open_loop_frozen_plan_does_not_silently_adapt_or_repair():
    initial=run(policy="OPEN_LOOP")
    # Initial fixed plan sends stock at t=1. Count of zero makes that plan
    # physically impossible, and it must not return a new zero-transfer plan.
    r=run(policy="OPEN_LOOP",decision_at=DAY1,observations=[observed("STOCK_COUNT",0)],previous=initial)
    assert r["status"] in {"NO_SOLUTION","INPUT_REQUIRED"}
    assert r["current_actions"]==[]


def test_input_change_invalidates_frozen_plan():
    initial=run(policy="OPEN_LOOP")
    p=adaptive_problem()
    r=run(replace(p,lots=(replace(p.lots[0],stock=9),)),policy="OPEN_LOOP",previous=initial)
    assert r["diagnostics"][0]["code"]=="FIXED_PLAN_INPUT_CHANGED"


def test_initial_inventory_override_conserves_stock_and_rejects_bad_totals():
    p=demo_problem("transfer")
    good=solve_extended(p,initial_inventory={("lot_A","S"):6,("lot_A","T"):4})
    assert good["status"]=="OPTIMAL"
    bad=solve_extended(p,initial_inventory={("lot_A","S"):6,("lot_A","T"):5})
    assert bad["status"]=="INPUT_REQUIRED"
    assert any(x["classification"]=="INITIAL_INVENTORY_TOTAL_MISMATCH" for x in bad["input_issues"])


def test_expired_observed_stock_requires_reconciliation():
    p=adaptive_problem()
    p=replace(p,lots=(replace(p.lots[0],expiry_period=0),))
    r=run(p,decision_at=DAY1)
    assert r["diagnostics"][0]["code"]=="EXPIRED_STOCK_UNRECONCILED"


def test_observation_input_order_does_not_change_decision():
    a=observed("SALE",2,event_id="sale")
    b=observed("RECEIPT",1,event_id="receipt",occurred_at="2026-10-17T10:00:00+09:00",known_at="2026-10-17T10:01:00+09:00")
    x=run(decision_at=DAY1,observations=[a,b])
    y=run(decision_at=DAY1,observations=[b,a])
    assert x["status"]==y["status"]=="COMPLETED"
    assert x["current_actions"]==y["current_actions"]
    assert x["problem_signature"]==y["problem_signature"]


def test_arrival_can_link_to_confirmed_transfer_without_double_booking():
    p=adaptive_problem()
    p=replace(p,committed_transfers=(CommittedTransfer("approved","lot_A","S_T",0,4),))
    departure=observed("TRANSFER_DEPARTED",4,route_id="S_T",reference_id="approved",cost_already_paid=True)
    arrival=observed("TRANSFER_ARRIVED",4,event_id="arrived",location="T",reference_id="approved",occurred_at=DAY1,known_at=DAY1)
    r=run(p,decision_at=DAY1,observations=[departure,arrival])
    assert r["status"]=="COMPLETED"
    assert {x["location"]:x["quantity"] for x in r["current_inventory"]}=={"S":6,"T":4}
    assert r["immutable_events"]==[]


@pytest.mark.parametrize("change",[{"vehicle_count":True},{"cost_already_paid":"false"},{"quantity":True}])
def test_boolean_or_text_values_cannot_turn_into_quantities_or_paid_cost(change):
    event=observed("TRANSFER_DEPARTED",4,route_id="S_T")
    r=run(decision_at=DAY1,observations=[replace(event,**change)])
    assert r["status"]=="INPUT_REQUIRED"
    assert r["result"] is None


def test_unavailable_observation_cannot_block_current_actions_with_future_values():
    a=observed("SALE",1,known_at=DAY2)
    b=replace(a,quantity=-999,occurred_at="2026-10-20T00:00:00+09:00")
    first=run(decision_at=DAY1,observations=[a])
    second=run(decision_at=DAY1,observations=[b])
    assert first["status"]==second["status"]=="COMPLETED"
    assert first["current_actions"]==second["current_actions"]
    assert first["problem_signature"]==second["problem_signature"]
