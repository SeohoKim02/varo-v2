from __future__ import annotations

from pathlib import Path

import pandas as pd

from services.real_transport_enrichment import (
    enrich_real_transport,
)


def _write_real_data(root: Path) -> None:
    weight_path = (
        root
        / "14_Product_Unit_Weight"
        / "processed"
        / "suhyup_product_unit_weight_varo_final.csv"
    )

    cost_path = (
        root
        / "13_Transport_Cost_Official"
        / "processed"
        / "suhyup_6_actual_center_route_vehicle_official_cost_matrix.csv"
    )

    center_path = (
        root
        / "09_VARO_Mapping"
        / "suhyup_actual_center_location_master.csv"
    )

    weight_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cost_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    center_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame([
        {
            "product_code": "P_REAL",
            "product_name": "Test product",
            "varo_unit_weight_kg": "10",
            "varo_unit_weight_status": "safe",
            "unit_weight_provenance": "test_actual",
        }
    ]).to_csv(
        weight_path,
        index=False,
        encoding="utf-8-sig",
    )

    rows = []

    for vehicle_class, capacity, rate, provenance in [
        ("1t", 1000, 420, "actual_official_direct"),
        ("2t", 2000, 571, "derived_from_official_upper_class"),
        ("3t", 3000, 571, "derived_from_official_upper_class"),
        ("4t", 4000, 571, "derived_from_official_upper_class"),
        ("5t", 5000, 571, "actual_official_direct"),
    ]:
        rows.append({
            "source_center_code": "A",
            "source_center_name": "A center",
            "target_center_code": "B",
            "target_center_name": "B center",
            "road_distance_km": "153.415",
            "travel_time_min": "124.02",
            "vehicle_capacity_class": vehicle_class,
            "vehicle_capacity_kg": str(capacity),
            "official_cost_per_km_krw": str(rate),
            "cost_provenance": provenance,
            "distance_provenance": "test_osrm",
            "travel_time_provenance": "test_osrm",
        })

    pd.DataFrame(rows).to_csv(
        cost_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        {
            "center_code": "A",
            "center_name": "A center",
        },
        {
            "center_code": "B",
            "center_name": "B center",
        },
        {
            "center_code": "C",
            "center_name": "C center",
        },
    ]).to_csv(
        center_path,
        index=False,
        encoding="utf-8-sig",
    )


def test_disabled_without_environment(monkeypatch):
    monkeypatch.delenv(
        "VARO_REAL_DATA_ROOT",
        raising=False,
    )

    frame = pd.DataFrame([
        {
            "source_id": "S001",
            "target_id": "S002",
            "product_id": "P001",
            "recommended_qty": 10,
            "estimated_cost": 3000,
        }
    ])

    result = enrich_real_transport(frame)

    pd.testing.assert_frame_equal(
        result,
        frame,
    )


def test_real_route_overrides_transport_values(
    tmp_path,
    monkeypatch,
):
    _write_real_data(tmp_path)

    monkeypatch.setenv(
        "VARO_REAL_DATA_ROOT",
        str(tmp_path),
    )

    frame = pd.DataFrame([
        {
            "source_id": "A",
            "target_id": "B",
            "product_id": "P_REAL",
            "recommended_qty": 150,
            "estimated_cost": 999,
            "distance_km": 1,
            "travel_time_min": 1,
        }
    ])

    result = enrich_real_transport(frame)

    row = result.iloc[0]

    assert bool(
        row["real_transport_applied"]
    )

    assert bool(
        row["real_transport_feasible"]
    )

    assert row[
        "real_transport_status"
    ] == "applied"

    assert row[
        "transfer_weight_kg"
    ] == 1500.0

    assert row[
        "vehicle_mix"
    ] == "2tx1"

    assert row[
        "total_capacity_kg"
    ] == 2000

    assert row[
        "distance_km"
    ] == 153.415

    assert row[
        "travel_time_min"
    ] == 124.02

    assert row[
        "estimated_cost"
    ] == 87600

    assert row[
        "transport_cost"
    ] == 87600

    assert row[
        "move_cost"
    ] == 87600


def test_actual_center_without_route_is_not_overwritten(
    tmp_path,
    monkeypatch,
):
    _write_real_data(tmp_path)

    monkeypatch.setenv(
        "VARO_REAL_DATA_ROOT",
        str(tmp_path),
    )

    frame = pd.DataFrame([
        {
            "source_id": "A",
            "target_id": "C",
            "product_id": "P_REAL",
            "recommended_qty": 10,
            "estimated_cost": 777,
            "distance_km": 9,
        }
    ])

    result = enrich_real_transport(frame)

    row = result.iloc[0]

    assert bool(
        row["real_transport_candidate"]
    )

    assert not bool(
        row["real_transport_applied"]
    )

    assert not bool(
        row["real_transport_feasible"]
    )

    assert row[
        "real_transport_status"
    ] == "route_not_available"

    assert row[
        "estimated_cost"
    ] == 777

    assert row[
        "distance_km"
    ] == 9
