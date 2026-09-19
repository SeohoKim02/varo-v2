from __future__ import annotations

import math

from services.pareto_service import (
    active_objectives,
    frontier_indices,
    pareto_layers,
    select_pareto_routes,
)


def _row(route_id: str, qty: float, cost: float, **values):
    suffix = route_id.replace("-", "")
    row = {
        "route_id": route_id,
        "recommended_qty": qty,
        "move_cost": cost,
        "product_id": values.pop("product_id", f"P{suffix}"),
        "source_id": values.pop("source_id", f"S{suffix}"),
        "target_id": values.pop("target_id", f"T{suffix}"),
        "route_type": "DIRECT",
    }
    row.update(values)
    return row


def _ids(result):
    return [row["route_id"] for row in result.selected_rows]


def test_01_dominated_candidate_is_excluded_from_frontier():
    rows = [_row("A", 10, 5), _row("B", 8, 7)]
    assert frontier_indices(rows) == [0]
    assert pareto_layers(rows) == [1, 2]


def test_02_identical_points_share_frontier_in_route_id_order():
    rows = [_row("B", 10, 5), _row("A", 10, 5)]
    assert frontier_indices(rows) == [1, 0]
    assert pareto_layers(rows) == [1, 1]


def test_03_service_is_maximized_and_cost_is_minimized():
    rows = [_row("HIGH", 12, 9), _row("CHEAP", 8, 3), _row("WORSE", 7, 10)]
    assert {_ids(select_pareto_routes(rows, 2))[0], _ids(select_pareto_routes(rows, 2))[1]} == {"HIGH", "CHEAP"}
    assert pareto_layers(rows)[2] > 1


def test_04_nan_cost_is_invalid_and_never_selected():
    rows = [_row("VALID", 4, 2), _row("NAN", 100, math.nan)]
    result = select_pareto_routes(rows, 5)
    assert _ids(result) == ["VALID"]
    assert any(item["route_id"] == "NAN" and "cost" in item["reason"] for item in result.excluded)


def test_05_negative_cost_is_invalid_and_never_selected():
    rows = [_row("VALID", 4, 2), _row("NEG", 100, -1)]
    result = select_pareto_routes(rows, 5)
    assert _ids(result) == ["VALID"]
    assert any(item["route_id"] == "NEG" and "negative" in item["reason"] for item in result.excluded)


def test_06_zero_variance_expected_saving_is_inactive():
    rows = [_row("A", 4, 4, expected_saving=10), _row("B", 5, 5, expected_saving=10)]
    objectives, inactive = active_objectives(rows)
    assert [item.name for item in objectives] == ["service_qty", "transport_cost"]
    assert inactive == ["expected_saving"]


def test_07_all_zero_expected_saving_is_inactive():
    rows = [_row("A", 4, 4, expected_saving=0), _row("B", 5, 5, expected_saving=0)]
    result = select_pareto_routes(rows, 1)
    assert result.active_objectives == ["service_qty", "transport_cost"]
    assert result.inactive_objectives == ["expected_saving"]


def test_08_frontier_is_recalculated_after_each_selection():
    rows = [_row("A", 10, 10), _row("B", 9, 9), _row("C", 8, 8)]
    result = select_pareto_routes(rows, 3)
    assert len(result.trace) == 3
    assert [step["frontier_size"] for step in result.trace] == [3, 2, 1]
    assert [step["step"] for step in result.trace] == [1, 2, 3]


def test_09_shared_source_inventory_is_updated_between_steps():
    rows = [
        _row("A", 7, 1, product_id="P", source_id="S", target_id="T1", source_surplus=10),
        _row("B", 6, 2, product_id="P", source_id="S", target_id="T2", source_surplus=10),
    ]
    result = select_pareto_routes(rows, 5)
    assert _ids(result) == ["A"]
    assert any(item["route_id"] == "B" and "source inventory" in item["reason"] for item in result.excluded)


def test_10_shared_target_need_is_updated_between_steps():
    rows = [
        _row("A", 7, 1, product_id="P", source_id="S1", target_id="T", target_need_7d=10),
        _row("B", 6, 2, product_id="P", source_id="S2", target_id="T", target_need_7d=10),
    ]
    result = select_pareto_routes(rows, 5)
    assert _ids(result) == ["A"]
    assert any(item["route_id"] == "B" and "target demand" in item["reason"] for item in result.excluded)


def test_11_top_k_limit_is_enforced():
    rows = [_row(f"R{index}", 20 - index, 1 + index) for index in range(8)]
    result = select_pareto_routes(rows, 5)
    assert len(result.selected_rows) == 5
    assert len(result.trace) == 5


def test_12_duplicate_route_is_not_selected_twice():
    rows = [
        _row("DUP", 10, 1, product_id="P1", source_id="S1", target_id="T1"),
        _row("DUP", 9, 2, product_id="P2", source_id="S2", target_id="T2"),
    ]
    result = select_pareto_routes(rows, 5)
    assert _ids(result) == ["DUP"]
    assert any(item["reason"] == "duplicate route" for item in result.excluded)


def test_13_repeated_runs_are_deterministic():
    rows = [_row("C", 8, 8), _row("A", 10, 10), _row("B", 9, 9)]
    first = select_pareto_routes(rows, 3)
    second = select_pareto_routes(rows, 3)
    assert first.selected_indices == second.selected_indices
    assert first.trace == second.trace


def test_14_single_candidate_is_selected():
    result = select_pareto_routes([_row("ONLY", 3, 7)], 5)
    assert _ids(result) == ["ONLY"]
    assert result.ranks == [1]
    assert result.trace[0]["distance_to_ideal"] == 0.0


def test_15_all_identical_candidates_use_stable_route_id_tie_break():
    rows = [_row("R3", 5, 5), _row("R1", 5, 5), _row("R2", 5, 5)]
    result = select_pareto_routes(rows, 3)
    assert _ids(result) == ["R1", "R2", "R3"]


def test_16_min_max_normalization_is_invariant_to_cost_scale():
    rows = [_row("A", 10, 100), _row("B", 8, 20), _row("C", 6, 1)]
    scaled = [dict(row, move_cost=row["move_cost"] * 1_000_000) for row in rows]
    assert _ids(select_pareto_routes(rows, 3)) == _ids(select_pareto_routes(scaled, 3))


def test_17_route_feasibility_and_real_capacity_are_respected():
    rows = [
        _row("BLOCKED", 20, 1, route_feasible=False),
        _row("OVER", 15, 2, route_capacity_qty=10),
        _row("OK", 8, 3, route_capacity_qty=10),
    ]
    result = select_pareto_routes(rows, 5)
    assert _ids(result) == ["OK"]
    reasons = {item["route_id"]: item["reason"] for item in result.excluded}
    assert reasons["BLOCKED"] == "route_feasible is false"
    assert reasons["OVER"] == "route capacity exceeded"
