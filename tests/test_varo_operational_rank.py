import pandas as pd

from services.vhs_score_engine import _rank_varo_operational


def _rank(rows):
    frame = pd.DataFrame(rows)
    frame["varo_final_rank"] = _rank_varo_operational(frame)
    return frame.sort_values("varo_final_rank").reset_index(drop=True)


def test_varo_operational_rank_prioritizes_service_quantity():
    ranked = _rank(
        [
            {
                "route_id": "LOW18",
                "recommended_qty": 18,
                "move_cost": 50000,
                "vhs_rank": 1,
            },
            {
                "route_id": "FULL50",
                "recommended_qty": 50,
                "move_cost": 100000,
                "vhs_rank": 2,
            },
        ]
    )

    assert ranked.iloc[0]["route_id"] == "FULL50"
    assert int(ranked.iloc[0]["varo_final_rank"]) == 1


def test_varo_operational_rank_minimizes_cost_at_equal_service():
    ranked = _rank(
        [
            {
                "route_id": "EXPENSIVE",
                "recommended_qty": 50,
                "move_cost": 239263,
                "vhs_rank": 1,
            },
            {
                "route_id": "CHEAP",
                "recommended_qty": 50,
                "move_cost": 54400,
                "vhs_rank": 2,
            },
        ]
    )

    assert ranked.iloc[0]["route_id"] == "CHEAP"
    assert int(ranked.iloc[0]["varo_final_rank"]) == 1


def test_varo_operational_rank_uses_vhs_as_final_tiebreak():
    ranked = _rank(
        [
            {
                "route_id": "VHS2",
                "recommended_qty": 50,
                "move_cost": 100000,
                "vhs_rank": 2,
            },
            {
                "route_id": "VHS1",
                "recommended_qty": 50,
                "move_cost": 100000,
                "vhs_rank": 1,
            },
        ]
    )

    assert ranked.iloc[0]["route_id"] == "VHS1"
    assert int(ranked.iloc[0]["varo_final_rank"]) == 1


def test_varo_operational_rank_handles_none_and_empty_frame():
    none_rank = _rank_varo_operational(None)
    empty_rank = _rank_varo_operational(pd.DataFrame())

    assert none_rank.empty
    assert empty_rank.empty
