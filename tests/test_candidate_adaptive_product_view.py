"""Display-only regressions: observed locations, scenario weights and day labels."""
from dataclasses import replace

import pytest

from components.candidate_adaptive_view import action_frame
from pages.candidate_review import _plan_day, _product_frame
from services.candidate_adaptive import Observation, plan_adaptive
from services.candidate_economics import solve_economics
from services.candidate_economics_experiments import demo_problem
from services.candidate_economics_second_experiments import adaptive_problem


def context_result(envelope):
    return {**envelope["result"], "adaptive_context": {
        k: v for k, v in envelope.items() if k not in {"result", "diagnostics"}}}


def test_legacy_product_columns_and_values_are_unchanged():
    problem = demo_problem("mixed")
    result = solve_economics(problem)
    assert _product_frame(problem, result).to_dict("records") == [{
        "상품": "P", "배치": "lot_A", "출발 점포": "S", "초기 재고": 10,
        "계획 이동량": 4, "출발 점포 판매 계획": 6, "단위": "EA", "처리 계획": "일부 이동 + 잔여 할인"}]


def test_observed_destination_sale_is_not_mislabelled_as_holding():
    problem = adaptive_problem()
    problem = replace(problem, routes=(replace(problem.routes[0], departure_periods=(0,)),))
    origin = "2026-10-17T00:00:00+09:00"
    decision = "2026-10-18T00:00:00+09:00"
    noon = "2026-10-17T12:00:00+09:00"
    common = dict(product_id="P", state="CHILLED", lot_id="lot_A", unit="EA", source="TEST confirmed record")
    events = [
        Observation("departed", "TRANSFER_DEPARTED", noon, noon, location="S", quantity=4,
                    route_id="S_T", cost_already_paid=True, **common),
        Observation("arrived", "TRANSFER_ARRIVED", decision, decision, location="T", quantity=4,
                    route_id="S_T", reference_id="departed", **common),
        Observation("signal", "SIGNAL", noon, noon, location="S", signal="HIGH", **common),
    ]
    envelope = plan_adaptive(problem, policy="MULTISTAGE_RECOURSE", decision_at=decision,
                             origin_at=origin, observations=events)
    assert envelope["status"] == "COMPLETED"
    result = context_result(envelope)
    row = _product_frame(problem, result).iloc[0]
    assert row["초기 재고"] == row["이번 판단 시 확인 재고"] == 10
    assert row["계획 이동량"] == row["후속 경로 이동량 (구간 합)"] == 0
    assert row["출발 점포 판매 계획"] == 0
    assert row["전체 점포 판매 계획"] == 4
    assert row["처리 계획"] == "정상판매"
    assert result["actual_savings"] is None and result["production_approved"] is False


def dc_problem_and_result(*, observed_at_dc=False):
    problem = demo_problem("transfer")
    first = replace(problem.routes[0], route_id="S_DC", target="DC")
    onward = replace(problem.routes[0], route_id="DC_T", source="DC", departure_periods=(1,))
    problem = replace(problem, horizon=3, routes=(first, onward),
        storage_capacity={"S": 20, "DC": 20, "T": 20},
        markets=(replace(problem.markets[1], period=2),))
    moves = [{"lot_id": "lot_A", "route_id": "DC_T", "departure_period": 1, "quantity": 4}]
    if not observed_at_dc:
        moves.insert(0, {"lot_id": "lot_A", "route_id": "S_DC", "departure_period": 0, "quantity": 4})
    result = {"transfers": moves, "sales": [
        {"scenario": "base", "lot_id": "lot_A", "market_id": "target", "level": "normal", "quantity": 4}],
        "adaptive_context": {"current_period": int(observed_at_dc), "current_inventory": [
            {"lot_id": "lot_A", "location": "S", "quantity": 6 if observed_at_dc else 10, "unit": "EA"},
            {"lot_id": "lot_A", "location": "DC", "quantity": 4 if observed_at_dc else 0, "unit": "EA"}]}}
    return problem, result


def test_dc_second_leg_is_separate_not_double_counted_as_new_goods():
    problem, result = dc_problem_and_result()
    row = _product_frame(problem, result).iloc[0]
    assert row["계획 이동량"] == 4  # Not the eight units counted across two route legs.
    assert row["후속 경로 이동량 (구간 합)"] == 4
    assert row["전체 점포 판매 계획"] == 4
    assert row["처리 계획"] == "재고 이동 + 정상판매"


def test_remaining_dc_departure_is_visible_without_recounting_past_departure():
    problem, result = dc_problem_and_result(observed_at_dc=True)
    row = _product_frame(problem, result).iloc[0]
    assert row["계획 이동량"] == 0  # The earlier confirmed S->DC movement is not a new plan.
    assert row["후속 경로 이동량 (구간 합)"] == 4
    assert row["전체 점포 판매 계획"] == 4
    assert row["처리 계획"] == "재고 이동 + 정상판매"


def test_conditioned_scenario_probabilities_weight_moves_and_all_sales():
    problem = adaptive_problem()
    result = {"transfers": [], "scenario_transfers": [
        {"scenario": "high", "lot_id": "lot_A", "route_id": "S_T", "departure_period": 0, "quantity": 10}],
        "sales": [{"scenario": "high", "lot_id": "lot_A", "market_id": "target", "level": "normal", "quantity": 10}],
        "adaptive_context": {"remaining_problem": {"scenarios": [{"scenario_id": "high", "probability": 1.0}]}}}
    row = _product_frame(problem, result).iloc[0]
    assert row["계획 이동량"] == row["전체 점포 판매 계획"] == 10  # Not original probability .5.


@pytest.mark.parametrize("context,relative,expected", [
    ({}, 0, 1), ({}, 2, 3), ({"current_period": 1}, 0, 2), ({"current_period": 2}, 1, 4)])
def test_solver_relative_day_is_displayed_in_original_planning_calendar(context, relative, expected):
    assert _plan_day({"adaptive_context": context}, relative) == expected


def test_current_action_and_route_detail_use_same_day_after_observation():
    problem = adaptive_problem()
    row = {"kind": "TRANSFER", "period": 1, "lot_id": "lot_A", "route_id": "S_T", "quantity": 4}
    action = action_frame(problem, [row]).iloc[0]
    assert action["계획 날짜 순서"] == _plan_day({"adaptive_context": {"current_period": 1}}, 0) == 2
