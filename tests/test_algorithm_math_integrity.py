"""Cross-strategy mathematical integrity and independence contracts."""
from __future__ import annotations

import math

import pandas as pd

from services.dqn_service import action_reward_matrix
from services.legacy_adapters._local_modules.heuristic_optimizer import add_heuristic_scores
from services.optimality_gap_service import calculate_gap_metrics
from services.vhs_score_engine import apply_auto_vhs, pareto_ranks


def _rows() -> list[dict]:
    return [
        {
            "route_id": "R1", "route_type": "DIRECT", "recommended_qty": 12,
            "suggested_qty": 12, "expected_saving": 120, "estimated_cost": 20,
            "move_cost": 20, "disposal_risk_score": 30, "demand_fit_score": 90,
            "inventory_balance_score": 80, "feasibility_score": 95,
            "promotion_score": 20, "days_to_expiry": 10,
            "final_recommendation": "직접 이동", "reason": "수요 대응 가능",
        },
        {
            "route_id": "R2", "route_type": "VIA_DC", "recommended_qty": 8,
            "suggested_qty": 8, "expected_saving": 90, "estimated_cost": 30,
            "move_cost": 30, "disposal_risk_score": 70, "demand_fit_score": 60,
            "inventory_balance_score": 65, "feasibility_score": 85,
            "promotion_score": 50, "days_to_expiry": 4,
            "final_recommendation": "DC 경유 이동", "reason": "재고 균형 가능",
        },
        {
            "route_id": "R3", "route_type": "DIRECT", "recommended_qty": 5,
            "suggested_qty": 5, "expected_saving": 60, "estimated_cost": 40,
            "move_cost": 40, "disposal_risk_score": 90, "demand_fit_score": 30,
            "inventory_balance_score": 40, "feasibility_score": 70,
            "promotion_score": 80, "days_to_expiry": 2,
            "final_recommendation": "긴급 할인", "reason": "유통기한 위험",
        },
    ]


def test_vhs_is_not_changed_by_greedy_dqn_or_prior_vhs_outputs():
    base = pd.DataFrame(_rows())
    changed = base.copy()
    changed["heuristic_score"] = [0, 100, 50]
    changed["greedy_rank"] = [3, 1, 2]
    changed["dqn_status"] = "정상"
    changed["dqn_reference_score"] = [100, 0, 75]
    changed["confidence_score"] = [0, 100, 20]
    changed["vhs_score"] = [1, 99, 50]
    first = apply_auto_vhs(base).frame.set_index("route_id")["vhs_score"]
    second = apply_auto_vhs(changed, {"status": "정상"}).frame.set_index("route_id")["vhs_score"]
    pd.testing.assert_series_equal(first, second)


def test_greedy_and_dqn_ignore_other_strategy_output_columns():
    base = _rows()
    changed = [dict(row, vhs_score=999, dqn_action="폐기", pareto_rank=99) for row in base]
    first = add_heuristic_scores(pd.DataFrame(base))["route_id"].tolist()
    second = add_heuristic_scores(pd.DataFrame(changed))["route_id"].tolist()
    assert first == second
    assert action_reward_matrix(base) == action_reward_matrix(changed)


def test_greedy_does_not_reward_negative_cost_or_quantity():
    frame = pd.DataFrame([
        dict(_rows()[0], route_id="VALID", estimated_cost=10, suggested_qty=10),
        dict(_rows()[0], route_id="INVALID", estimated_cost=-100, suggested_qty=-10),
    ])
    ranked = add_heuristic_scores(frame)
    assert ranked.iloc[0]["route_id"] == "VALID"
    assert math.isnan(float(ranked.loc[ranked["route_id"] == "INVALID", "_estimated_cost_numeric"].iloc[0]))


def test_pareto_missing_cost_cannot_create_a_false_frontier_member():
    complete = dict(_rows()[0], route_id="COMPLETE")
    incomplete = dict(complete, route_id="MISSING", move_cost=None, estimated_cost=None)
    ranks = pareto_ranks([complete, incomplete])
    assert ranks == [1, 2]


def test_pareto_identical_points_share_frontier_and_are_deterministic():
    items = [dict(_rows()[0], route_id="A"), dict(_rows()[0], route_id="B")]
    assert pareto_ranks(items) == [1, 1]
    assert pareto_ranks(items) == pareto_ranks(items)


def test_optimality_gap_is_not_exact_when_service_levels_differ():
    result = calculate_gap_metrics(
        80, 90, 100, exact=True,
        varo_service=8, greedy_service=10, best_service=10,
    )
    assert result["service_comparable"] is False
    assert result["objective_comparable"] is False
    assert result["label"] == "서비스 비동등 참고 Gap"
    assert "동일 서비스 수준" in result["comparison_warning"]
