"""Real-data candidate generation must not depend on fabricated economics."""

from __future__ import annotations

import inspect

import pandas as pd

from services.analysis_pipeline import run_analysis_pipeline
from services.candidate_generator import generate_candidates


def _direct_network_data() -> dict[str, pd.DataFrame]:
    stores = pd.DataFrame(
        [
            {
                "node_id": "A",
                "node_name": "?? A",
                "node_type": "STORE",
                "network_mode": "DIRECT_NETWORK",
            },
            {
                "node_id": "B",
                "node_name": "?? B",
                "node_type": "STORE",
                "network_mode": "DIRECT_NETWORK",
            },
        ]
    )

    products = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "product_name": "???? ??",
            }
        ]
    )

    inventory = pd.DataFrame(
        [
            {"store_id": "A", "product_id": "P1", "stock_qty": 100},
            {"store_id": "B", "product_id": "P1", "stock_qty": 0},
        ]
    )

    # Deliberately huge route cost:
    # legacy fake-price saving logic would reject this candidate.
    routes = pd.DataFrame(
        [
            {
                "source_id": "A",
                "target_id": "B",
                "distance_km": 10.0,
                "estimated_cost": 999_999_999.0,
                "travel_time_min": 20.0,
            },
            {
                "source_id": "B",
                "target_id": "A",
                "distance_km": 10.0,
                "estimated_cost": 999_999_999.0,
                "travel_time_min": 20.0,
            },
        ]
    )

    return {
        "stores": stores,
        "products": products,
        "inventory": inventory,
        "routes": routes,
    }


def test_real_transport_mode_keeps_candidate_without_fake_positive_saving(
    monkeypatch,
):
    monkeypatch.setenv("VARO_REAL_DATA_ROOT", r"C:\varo-real-data")

    frame, info = generate_candidates(_direct_network_data())

    assert info["generated"]
    assert info["real_transport_candidate_mode"] is True
    assert frame is not None
    assert not frame.empty
    assert set(frame["route_type"]) == {"DIRECT"}
    assert (frame["expected_saving"] == 0).all()
    assert set(frame["candidate_cost_status"]) == {"deferred_real_transport"}
    assert set(frame["candidate_economics_status"]) == {
        "actual_price_unavailable"
    }


def test_legacy_mode_still_uses_positive_saving_filter(monkeypatch):
    monkeypatch.delenv("VARO_REAL_DATA_ROOT", raising=False)

    frame, info = generate_candidates(_direct_network_data())

    assert not info["generated"]
    assert info["real_transport_candidate_mode"] is False
    assert info["negative_saving_excluded"] >= 1
    assert frame is None


def test_real_transport_runs_before_greedy_and_legacy_vhs():
    source = inspect.getsource(run_analysis_pipeline)

    real_transport_pos = source.index(
        "candidates = enrich_real_transport(candidates)"
    )
    greedy_pos = source.index(
        'runner.call("heuristic_optimizer", "add_heuristic_scores"'
    )
    legacy_vhs_pos = source.index(
        'runner.call("varo_hybrid_score", "calculate_varo_hybrid_score"'
    )

    assert real_transport_pos < greedy_pos
    assert real_transport_pos < legacy_vhs_pos
