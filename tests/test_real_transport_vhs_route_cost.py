from __future__ import annotations

import pandas as pd

from services.vhs_score_engine import apply_auto_vhs


def _frame(real: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route_id": "R1",
                "expected_saving": 100000,
                "estimated_cost": 50000,
                "move_cost": 50000,
                "transport_cost": 50000,
                "distance_km": 100,
                "travel_time_min": 60,
                "recommended_qty": 10,
                "confidence_score": 80,
                "real_transport_applied": real,
            },
            {
                "route_id": "R2",
                "expected_saving": 100000,
                "estimated_cost": 70000,
                "move_cost": 70000,
                "transport_cost": 70000,
                "distance_km": 10,
                "travel_time_min": 10,
                "recommended_qty": 10,
                "confidence_score": 80,
                "real_transport_applied": real,
            },
        ]
    )


def test_real_transport_route_cost_follows_official_cost():
    result = apply_auto_vhs(
        _frame(True)
    ).frame

    scores = (
        result
        .set_index("route_id")[
            "route_cost_score"
        ]
    )

    # Lower verified official transport cost must score better,
    # regardless of the raw distance/time columns.
    assert scores["R1"] > scores["R2"]


def test_legacy_route_cost_behavior_is_preserved():
    result = apply_auto_vhs(
        _frame(False)
    ).frame

    scores = (
        result
        .set_index("route_id")[
            "route_cost_score"
        ]
    )

    # Legacy formula still includes distance/time.
    # R1 has substantially larger distance/time and therefore
    # should be penalized despite its lower nominal cost.
    assert scores["R1"] < scores["R2"]
