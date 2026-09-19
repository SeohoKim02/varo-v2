import pandas as pd

from services.suhyup_algorithm_revalidation import (
    _selection_violation,
    lexicographic_milp,
    ordered_feasible_selection,
)


def _row(route, qty, cost, source="S1", target="T1", product="P1", source_cap=100, target_cap=100):
    return {
        "route_id": route,
        "recommended_qty": qty,
        "move_cost": cost,
        "source_id": source,
        "target_id": target,
        "product_id": product,
        "route_type": "DIRECT",
        "dc_id": "",
        "source_surplus": source_cap,
        "target_need_7d": target_cap,
    }


def test_shared_selection_prevents_source_and_target_double_allocation():
    rows = pd.DataFrame([
        _row("R1", 60, 10, source_cap=100, target_cap=100),
        _row("R2", 60, 20, source_cap=100, target_cap=100),
    ])
    selected = ordered_feasible_selection(rows, ("move_cost", "route_id"), (True, True))
    assert selected["route_id"].tolist() == ["R1"]
    assert _selection_violation(selected.to_dict("records")) is None


def test_lexicographic_milp_maximizes_service_then_minimizes_cost():
    rows = pd.DataFrame([
        _row("R1", 50, 100, source="S1", target="T1", product="P1"),
        _row("R2", 50, 20, source="S2", target="T2", product="P2"),
        _row("R3", 50, 30, source="S3", target="T3", product="P3"),
        _row("R4", 50, 40, source="S4", target="T4", product="P4"),
        _row("R5", 50, 50, source="S5", target="T5", product="P5"),
        _row("R6", 50, 10, source="S6", target="T6", product="P6"),
    ])
    result = lexicographic_milp(rows)
    assert result["optimal"] is True
    assert set(result["selected"]["route_id"]) == {"R2", "R3", "R4", "R5", "R6"}


def test_lexicographic_milp_enforces_actual_target_need():
    rows = pd.DataFrame([
        _row("BLOCKED", 50, 1, source="S1", target="T1", product="P1", target_cap=6),
        _row("VALID", 50, 20, source="S2", target="T2", product="P2", target_cap=50),
    ])
    result = lexicographic_milp(rows)
    assert result["optimal"] is True
    assert result["selected"]["route_id"].tolist() == ["VALID"]
