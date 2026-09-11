"""Optional real-data transport enrichment for Varo V2 candidates."""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache, reduce
from math import gcd
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_ENV = "VARO_REAL_DATA_ROOT"

WEIGHT_RELATIVE_PATH = Path(
    "14_Product_Unit_Weight",
    "processed",
    "suhyup_product_unit_weight_varo_final.csv",
)

COST_RELATIVE_PATH = Path(
    "13_Transport_Cost_Official",
    "processed",
    "suhyup_6_actual_center_route_vehicle_official_cost_matrix.csv",
)

CENTER_RELATIVE_PATH = Path(
    "09_VARO_Mapping",
    "suhyup_actual_center_location_master.csv",
)


def _id_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]

    return text


def _numeric(value: Any) -> float | None:
    parsed = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(parsed):
        return None

    return float(parsed)


def _configured_root() -> Path | None:
    raw = os.getenv(ROOT_ENV, "").strip()

    if not raw:
        return None

    root = Path(raw)

    if not root.is_dir():
        return None

    return root


@lru_cache(maxsize=4)
def _load_tables(root_text: str):
    root = Path(root_text)

    weight_path = root / WEIGHT_RELATIVE_PATH
    cost_path = root / COST_RELATIVE_PATH
    center_path = root / CENTER_RELATIVE_PATH

    if not (
        weight_path.is_file()
        and cost_path.is_file()
        and center_path.is_file()
    ):
        return None

    weights = pd.read_csv(
        weight_path,
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")

    costs = pd.read_csv(
        cost_path,
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")

    centers = pd.read_csv(
        center_path,
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")

    required_weight = {
        "product_code",
        "varo_unit_weight_kg",
        "varo_unit_weight_status",
    }

    required_cost = {
        "source_center_code",
        "target_center_code",
        "road_distance_km",
        "travel_time_min",
        "vehicle_capacity_class",
        "vehicle_capacity_kg",
        "official_cost_per_km_krw",
        "cost_provenance",
    }

    if not required_weight.issubset(weights.columns):
        return None

    if not required_cost.issubset(costs.columns):
        return None

    if "center_code" not in centers.columns:
        return None

    weight_lookup = {}

    for _, row in weights.iterrows():
        product_id = _id_text(row.get("product_code"))

        if not product_id:
            continue

        weight = _numeric(
            row.get("varo_unit_weight_kg")
        )

        status = str(
            row.get("varo_unit_weight_status", "")
        ).strip().lower()

        weight_lookup[product_id] = {
            "weight_kg": weight,
            "status": status,
            "product_name": row.get(
                "product_name",
                "",
            ),
            "provenance": row.get(
                "unit_weight_provenance",
                "",
            ),
        }

    route_lookup = {}

    route_columns = [
        "source_center_code",
        "source_center_name",
        "target_center_code",
        "target_center_name",
        "road_distance_km",
        "travel_time_min",
        "distance_provenance",
        "travel_time_provenance",
    ]

    route_base = costs[
        [
            column
            for column in route_columns
            if column in costs.columns
        ]
    ].drop_duplicates(
        subset=[
            "source_center_code",
            "target_center_code",
        ]
    )

    for _, row in route_base.iterrows():
        source = _id_text(
            row.get("source_center_code")
        )

        target = _id_text(
            row.get("target_center_code")
        )

        distance = _numeric(
            row.get("road_distance_km")
        )

        travel_time = _numeric(
            row.get("travel_time_min")
        )

        if (
            not source
            or not target
            or distance is None
            or travel_time is None
        ):
            continue

        route_lookup[(source, target)] = {
            "distance_km": distance,
            "travel_time_min": travel_time,
            "distance_provenance": row.get(
                "distance_provenance",
                "",
            ),
            "travel_time_provenance": row.get(
                "travel_time_provenance",
                "",
            ),
        }

    vehicle_rows = (
        costs[
            [
                "vehicle_capacity_class",
                "vehicle_capacity_kg",
                "official_cost_per_km_krw",
                "cost_provenance",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    vehicles = []

    for _, row in vehicle_rows.iterrows():
        capacity = _numeric(
            row.get("vehicle_capacity_kg")
        )

        rate = _numeric(
            row.get("official_cost_per_km_krw")
        )

        if (
            capacity is None
            or rate is None
            or capacity <= 0
            or rate <= 0
        ):
            continue

        provenance = str(
            row.get("cost_provenance", "")
        ).strip()

        vehicles.append({
            "class": str(
                row.get(
                    "vehicle_capacity_class",
                    "",
                )
            ).strip(),
            "capacity_kg": int(round(capacity)),
            "rate_per_km": int(round(rate)),
            "proxy": (
                provenance
                != "actual_official_direct"
            ),
            "provenance": provenance,
        })

    vehicles.sort(
        key=lambda item: item["capacity_kg"]
    )

    actual_centers = {
        _id_text(value)
        for value in centers["center_code"]
        if _id_text(value)
    }

    return {
        "weights": weight_lookup,
        "routes": route_lookup,
        "vehicles": vehicles,
        "actual_centers": actual_centers,
    }


def _optimize_vehicle_mix(
    transfer_weight_kg: float,
    vehicles: list[dict[str, Any]],
):
    if transfer_weight_kg <= 0 or not vehicles:
        return None

    capacities = [
        int(vehicle["capacity_kg"])
        for vehicle in vehicles
    ]

    capacity_gcd = reduce(gcd, capacities)

    if capacity_gcd <= 0:
        return None

    vehicle_units = [
        capacity // capacity_gcd
        for capacity in capacities
    ]

    required_units = int(
        math.ceil(
            transfer_weight_kg
            / capacity_gcd
        )
    )

    max_vehicle_units = max(vehicle_units)

    limit = (
        required_units
        + max_vehicle_units
        - 1
    )

    # Exact-capacity DP:
    # state = (total_rate, vehicle_count, proxy_count, counts)
    dp = [None] * (limit + 1)

    dp[0] = (
        0,
        0,
        0,
        tuple(0 for _ in vehicles),
    )

    for capacity_units in range(1, limit + 1):
        best_state = None

        for index, vehicle in enumerate(vehicles):
            units = vehicle_units[index]

            previous_units = (
                capacity_units - units
            )

            if previous_units < 0:
                continue

            previous = dp[previous_units]

            if previous is None:
                continue

            counts = list(previous[3])
            counts[index] += 1

            candidate = (
                previous[0]
                + vehicle["rate_per_km"],

                previous[1] + 1,

                previous[2]
                + int(vehicle["proxy"]),

                tuple(counts),
            )

            candidate_key = (
                candidate[0],
                candidate[1],
                candidate[2],
            )

            if (
                best_state is None
                or candidate_key
                < (
                    best_state[0],
                    best_state[1],
                    best_state[2],
                )
            ):
                best_state = candidate

        dp[capacity_units] = best_state

    best = None

    for capacity_units in range(
        required_units,
        limit + 1,
    ):
        state = dp[capacity_units]

        if state is None:
            continue

        capacity_kg = (
            capacity_units
            * capacity_gcd
        )

        unused = (
            capacity_kg
            - transfer_weight_kg
        )

        objective = (
            state[0],
            unused,
            state[1],
            state[2],
        )

        if (
            best is None
            or objective < best["objective"]
        ):
            best = {
                "objective": objective,
                "total_rate_per_km": state[0],
                "vehicle_count": state[1],
                "proxy_vehicle_count": state[2],
                "counts": state[3],
                "capacity_kg": capacity_kg,
                "unused_capacity_kg": unused,
            }

    return best


def _quantity(row: pd.Series) -> float | None:
    for column in (
        "recommended_qty",
        "suggested_qty",
        "transfer_qty",
        "quantity",
    ):
        if column not in row.index:
            continue

        value = _numeric(row.get(column))

        if value is not None:
            return value

    return None


def enrich_real_transport(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Apply verified Suhyup transport facts only to matching real IDs."""

    if (
        not isinstance(candidates, pd.DataFrame)
        or candidates.empty
    ):
        return candidates

    root = _configured_root()

    if root is None:
        return candidates

    tables = _load_tables(str(root))

    if tables is None:
        return candidates

    required = {
        "source_id",
        "target_id",
        "product_id",
    }

    if not required.issubset(candidates.columns):
        return candidates

    result = candidates.copy()

    # Candidate frames may infer integer dtype from demo values such
    # as distance_km=1. Real routes contain fractional values such as
    # 153.415 km, so normalize overwrite targets before enrichment.
    for column in (
        "distance_km",
        "travel_time_min",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ).astype("float64")

    defaults = {
        "real_transport_candidate": False,
        "real_transport_applied": False,
        "real_transport_feasible": None,
        "real_transport_status": "not_applicable",
        "transfer_weight_kg": None,
        "unit_weight_kg": None,
        "unit_weight_status": None,
        "unit_weight_provenance": None,
        "vehicle_mix": None,
        "vehicle_count": None,
        "total_capacity_kg": None,
        "unused_capacity_kg": None,
        "capacity_utilization_pct": None,
        "proxy_vehicle_count": None,
        "official_total_rate_per_km_krw": None,
        "transport_cost_provenance": None,
        "real_distance_provenance": None,
        "real_travel_time_provenance": None,
    }

    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    actual_centers = tables["actual_centers"]
    weight_lookup = tables["weights"]
    route_lookup = tables["routes"]
    vehicles = tables["vehicles"]

    for index, row in result.iterrows():
        source = _id_text(
            row.get("source_id")
        )

        target = _id_text(
            row.get("target_id")
        )

        product_id = _id_text(
            row.get("product_id")
        )

        if (
            source not in actual_centers
            or target not in actual_centers
        ):
            continue

        result.at[
            index,
            "real_transport_candidate",
        ] = True

        result.at[
            index,
            "real_transport_feasible",
        ] = False

        route = route_lookup.get(
            (source, target)
        )

        if route is None:
            result.at[
                index,
                "real_transport_status",
            ] = "route_not_available"

            continue

        product_info = weight_lookup.get(
            product_id
        )

        if product_info is None:
            result.at[
                index,
                "real_transport_status",
            ] = "product_weight_not_available"

            continue

        unit_weight = product_info[
            "weight_kg"
        ]

        unit_status = product_info[
            "status"
        ]

        result.at[
            index,
            "unit_weight_kg",
        ] = unit_weight

        result.at[
            index,
            "unit_weight_status",
        ] = unit_status

        result.at[
            index,
            "unit_weight_provenance",
        ] = product_info[
            "provenance"
        ]

        if (
            unit_weight is None
            or unit_weight <= 0
            or unit_status
            not in {"safe", "conditional"}
        ):
            result.at[
                index,
                "real_transport_status",
            ] = "product_weight_insufficient"

            continue

        quantity = _quantity(row)

        if quantity is None or quantity <= 0:
            result.at[
                index,
                "real_transport_status",
            ] = "invalid_transfer_quantity"

            continue

        transfer_weight = (
            quantity * unit_weight
        )

        mix = _optimize_vehicle_mix(
            transfer_weight,
            vehicles,
        )

        if mix is None:
            result.at[
                index,
                "real_transport_status",
            ] = "vehicle_mix_unavailable"

            continue

        distance = route["distance_km"]

        transport_cost = round(
            distance
            * mix["total_rate_per_km"]
        )

        mix_parts = []

        for vehicle, count in zip(
            vehicles,
            mix["counts"],
        ):
            if count <= 0:
                continue

            mix_parts.append(
                f'{vehicle["class"]}x{count}'
            )

        mix_text = "+".join(mix_parts)

        result.at[
            index,
            "distance_km",
        ] = distance

        result.at[
            index,
            "travel_time_min",
        ] = route["travel_time_min"]

        # Keep common cost aliases synchronized so current VHS
        # sees the verified official transport cost.
        result.at[
            index,
            "estimated_cost",
        ] = transport_cost

        result.at[
            index,
            "transport_cost",
        ] = transport_cost

        result.at[
            index,
            "move_cost",
        ] = transport_cost

        result.at[
            index,
            "transfer_weight_kg",
        ] = round(
            transfer_weight,
            3,
        )

        result.at[
            index,
            "vehicle_mix",
        ] = mix_text

        result.at[
            index,
            "vehicle_count",
        ] = mix["vehicle_count"]

        result.at[
            index,
            "total_capacity_kg",
        ] = mix["capacity_kg"]

        result.at[
            index,
            "unused_capacity_kg",
        ] = round(
            mix["unused_capacity_kg"],
            3,
        )

        result.at[
            index,
            "capacity_utilization_pct",
        ] = round(
            transfer_weight
            / mix["capacity_kg"]
            * 100.0,
            2,
        )

        result.at[
            index,
            "proxy_vehicle_count",
        ] = mix[
            "proxy_vehicle_count"
        ]

        result.at[
            index,
            "official_total_rate_per_km_krw",
        ] = mix[
            "total_rate_per_km"
        ]

        result.at[
            index,
            "transport_cost_provenance",
        ] = (
            "official_2026_rate_with_derived_vehicle_mix"
        )

        result.at[
            index,
            "real_distance_provenance",
        ] = route[
            "distance_provenance"
        ]

        result.at[
            index,
            "real_travel_time_provenance",
        ] = route[
            "travel_time_provenance"
        ]

        result.at[
            index,
            "real_transport_status",
        ] = "applied"

        result.at[
            index,
            "real_transport_feasible",
        ] = True

        result.at[
            index,
            "real_transport_applied",
        ] = True

    return result
