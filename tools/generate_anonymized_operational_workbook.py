"""Generate a reproducible, fully anonymized operational validation workbook.

Purpose
-------
Varo V2 is validated end-to-end against a workbook that has the *shape* of a real
operational upload (multi-store, multi-product, two DCs, DIRECT + VIA_DC routes)
but contains no real person, store, or company data. Every identifier and name is
a synthetic placeholder ("가상점포 07", "가상상품 12", "가상물류센터 1") and every
coordinate is a point on an artificial grid, not a real address.

The workbook deliberately mixes three kinds of rows so partial application can be
validated for real:

* normal rows that are kept,
* rows with a *retained* warning (identifier stored as a number),
* rows with a row-scoped error that must be safely excluded, plus the dependent
  rows that must cascade out with them.

It never contains a file-blocking problem, because the point of the exercise is
that the file stays usable after the bad rows are excluded.

Expected results are written next to the workbook as a manifest. They are derived
here *by construction* (from what this script actually wrote), never by calling
the services under test, so the manifest is an independent expectation.

Usage
-----
    python tools/generate_anonymized_operational_workbook.py

Both outputs are deterministic for a fixed seed: running twice produces
byte-identical sheet contents and an identical manifest.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "validation_data"
WORKBOOK_NAME = "varo_v2_anonymized_operational.xlsx"
MANIFEST_NAME = "varo_v2_anonymized_operational_manifest.json"

SEED = 20260901
# Sized from measured pipeline timings (see docs/OPERATIONAL_VALIDATION.md): large
# enough that intake, exclusion and analysis all work on thousands of rows, small
# enough that the whole validation still runs in seconds on one desktop.
STORE_COUNT = 40
PRODUCT_COUNT = 30
PRODUCTS_PER_STORE = 24
MAX_VALID_RECOMMENDATIONS = 60
REFERENCE_DATE = date(2026, 9, 1)

# Two independent DCs. Each store belongs to exactly one DC group, so a problem
# in one DC can never silently remove the other DC's routes.
DC_IDS = ("DC01", "DC02")
DC_NAMES = {"DC01": "가상물류센터 1", "DC02": "가상물류센터 2"}

REGION_NAMES = ("가상권역 A", "가상권역 B", "가상권역 C", "가상권역 D")
CATEGORIES = ("가상신선", "가상냉장", "가상냉동", "가상상온")
STORAGE_BY_CATEGORY = {
    "가상신선": "냉장", "가상냉장": "냉장", "가상냉동": "냉동", "가상상온": "상온",
}
COLD_CATEGORIES = {"가상신선", "가상냉장", "가상냉동"}

NORMAL_TRANSPORT = "일반 탑차"
COLD_TRANSPORT = "냉동/냉장 탑차"

# Road/vehicle cost model (fictional but internally consistent).
FIXED_COST = 3400
COST_PER_KM_NORMAL = 820
COST_PER_KM_COLD = 980
ROAD_FACTOR = 1.25

SHEET_ORDER = ("stores", "products", "inventory", "routes", "v2_recommendations", "config", "README")
# Sheets whose rows are counted as analysis input by services.partial_data.
ANALYSIS_SHEET_KEYS = {
    "stores": "stores", "products": "products", "inventory": "inventory",
    "routes": "routes", "v2_recommendations": "recommendations",
}


# --------------------------------------------------------------------------- #
# Master data
# --------------------------------------------------------------------------- #
def _store_position(index: int) -> tuple[float, float]:
    """Artificial 6x4 grid in kilometres. Not a real-world location."""
    column, row = index % 6, index // 6
    return column * 3.0, row * 4.0


DC_POSITIONS = {"DC01": (7.5, 2.0), "DC02": (7.5, 14.0)}


def _grid_coordinates(x: float, y: float) -> tuple[float, float]:
    """Map the artificial grid to synthetic lat/lon far from any real address."""
    return round(10.0 + y * 0.01, 4), round(10.0 + x * 0.01, 4)


def _distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    raw = math.hypot(a[0] - b[0], a[1] - b[1]) * ROAD_FACTOR
    return round(max(0.6, raw), 1)


def build_stores() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    for dc_index, dc_id in enumerate(DC_IDS):
        x, y = DC_POSITIONS[dc_id]
        latitude, longitude = _grid_coordinates(x, y)
        rows.append({
            "store_id": dc_id, "store_name": DC_NAMES[dc_id], "type": "DC",
            "store_type": "DC", "node_type": "DC",
            "region": REGION_NAMES[dc_index * 2],
            "latitude": latitude, "longitude": longitude,
            "available_start": "07:00", "available_end": "23:00",
            "capacity": 6000, "cold_storage_available": True,
            "node_id": dc_id, "node_name": DC_NAMES[dc_id],
        })
    for index in range(STORE_COUNT):
        store_id = f"S{index + 1:02d}"
        x, y = _store_position(index)
        latitude, longitude = _grid_coordinates(x, y)
        rows.append({
            "store_id": store_id, "store_name": f"가상점포 {index + 1:02d}", "type": "STORE",
            "store_type": "STORE", "node_type": "STORE",
            "region": REGION_NAMES[(index // 6) % len(REGION_NAMES)],
            "latitude": latitude, "longitude": longitude,
            "available_start": "08:00", "available_end": "22:00",
            "capacity": 500, "cold_storage_available": index % 3 != 2,
            "node_id": store_id, "node_name": f"가상점포 {index + 1:02d}",
        })
    names = {row["node_id"]: row["node_name"] for row in rows}
    # First half of the stores is served by DC01, second half by DC02.
    dc_of_store = {
        f"S{index + 1:02d}": DC_IDS[0] if index < STORE_COUNT // 2 else DC_IDS[1]
        for index in range(STORE_COUNT)
    }
    return pd.DataFrame(rows), names, dc_of_store


def build_products(rng: Random) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(PRODUCT_COUNT):
        category = CATEGORIES[index % len(CATEGORIES)]
        unit_cost = rng.randrange(900, 6000, 10)
        rows.append({
            # Numeric SKU codes on purpose: this is the documented id_numeric
            # warning case (Excel stores the code as a number). It is a *retained*
            # warning, not an exclusion.
            "product_id": 100001 + index,
            "product_name": f"가상상품 {index + 1:02d}",
            "category": category,
            "inventory_category": category,
            "storage_type": STORAGE_BY_CATEGORY[category],
            "unit_cost": unit_cost,
            "unit_price": int(unit_cost * 1.7 // 10 * 10),
            "disposal_cost_per_unit": int(unit_cost * 0.35 // 10 * 10),
            "disposal_cost": int(unit_cost * 0.35 // 10 * 10),
            "shelf_life_days": rng.choice([4, 5, 7, 14, 30, 90, 180]),
            "distance_cutline_km": rng.choice([18, 22, 25, 30]),
            "cold_required": category in COLD_CATEGORIES,
            "lead_time_days": rng.choice([1, 1, 2, 3]),
            "order_cost": 25000,
            "holding_rate": 0.18,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def _store_profile(index: int) -> str:
    if index % 4 == 0:
        return "과잉"
    if index % 4 == 1:
        return "부족"
    return "정상"


def build_inventory(
    rng: Random, products: pd.DataFrame, names: dict[str, str], stores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    product_rows = {int(row["product_id"]): row for _, row in products.iterrows()}
    region_of = {str(row["node_id"]): str(row["region"]) for _, row in stores.iterrows()}
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for index in range(STORE_COUNT):
        store_id = f"S{index + 1:02d}"
        profile = _store_profile(index)
        chosen = sorted(rng.sample(range(PRODUCT_COUNT), PRODUCTS_PER_STORE))
        for offset in chosen:
            product = product_rows[100001 + offset]
            product_id = int(product["product_id"])
            daily = rng.randint(2, 12)
            if profile == "과잉":
                stock = daily * rng.randint(18, 30)
                expiry = rng.choice([2, 3, 4, 6, 9, 12])
            elif profile == "부족":
                stock = daily * rng.randint(1, 3)
                expiry = rng.choice([12, 18, 25, 40])
            else:
                stock = daily * rng.randint(6, 12)
                expiry = rng.choice([5, 8, 14, 21, 30])
            expiry = min(expiry, int(product["shelf_life_days"]))
            unit_cost = int(product["unit_cost"])
            record = {
                "inventory_id": f"INV-{store_id}-{product_id}",
                "store_id": store_id,
                "store_name": names[store_id],
                "region": region_of[store_id],
                "product_id": product_id,
                "product_name": product["product_name"],
                "category": product["category"],
                "inventory_category": product["inventory_category"],
                "stock_qty": stock,
                "avg_daily_sales": float(daily),
                "sales_qty": float(daily),
                "sales_7d": daily * 7,
                "sales_30d": daily * 30,
                "dead_stock_qty": max(0, stock - daily * 30),
                "demand_qty": daily * 7,
                "days_to_expiry": expiry,
                "expiry_days": expiry,
                "expiry_date": (REFERENCE_DATE + timedelta(days=expiry)).isoformat(),
                "unit_cost": unit_cost,
                "unit_price": int(product["unit_price"]),
                "disposal_cost_per_unit": int(product["disposal_cost_per_unit"]),
                "demand_std": round(daily * 0.6, 2),
                "lead_time_days": int(product["lead_time_days"]),
                "order_cost": int(product["order_cost"]),
                "holding_cost": int(unit_cost * 0.18),
                "daily_holding_cost": round(unit_cost * 0.18 / 365, 2),
                "cold_required": bool(product["cold_required"]),
                "service_level": 0.95,
                "capacity": 500,
            }
            rows.append(record)
            lookup[(store_id, product_id)] = record
    return pd.DataFrame(rows), lookup


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def build_routes(names: dict[str, str], dc_of_store: dict[str, str]) -> pd.DataFrame:
    positions: dict[str, tuple[float, float]] = {
        f"S{index + 1:02d}": _store_position(index) for index in range(STORE_COUNT)
    }
    positions.update(DC_POSITIONS)
    rows: list[dict[str, Any]] = []

    def add(source: str, target: str, route_type: str, cold: bool) -> None:
        distance = _distance_km(positions[source], positions[target])
        per_km = COST_PER_KM_COLD if cold else COST_PER_KM_NORMAL
        cost = int(round((FIXED_COST + per_km * distance) / 100.0) * 100)
        rows.append({
            "route_id": f"PATH{len(rows) + 1:04d}",
            "from_id": source, "to_id": target,
            "source_id": source, "source_name": names[source],
            "target_id": target, "target_name": names[target],
            "route_type": route_type,
            "distance_km": distance,
            "travel_time_min": int(round(6 + distance * 2.4)),
            "estimated_cost": cost,
            "transport_cost": cost,
            "cost_per_km": per_km,
            "fixed_cost": FIXED_COST,
            "cold_chain_available": cold,
            "available": True,
            "available_start": "08:00", "available_end": "22:00",
            "transport_mode": COLD_TRANSPORT if cold else NORMAL_TRANSPORT,
            "transport_type": COLD_TRANSPORT if cold else NORMAL_TRANSPORT,
        })

    store_ids = [f"S{index + 1:02d}" for index in range(STORE_COUNT)]
    for source in store_ids:
        for target in store_ids:
            if source != target:
                add(source, target, "DIRECT", cold=False)
    # Each store connects only to its own DC, in both directions.
    for store_id in store_ids:
        dc_id = dc_of_store[store_id]
        add(store_id, dc_id, "VIA_DC", cold=True)
        add(dc_id, store_id, "VIA_DC", cold=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
def _route_index(routes: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["source_id"]), str(row["target_id"])): row
        for _, row in routes.iterrows()
    }


def build_recommendations(
    rng: Random, products: pd.DataFrame, inventory_lookup: dict[tuple[str, int], dict[str, Any]],
    routes: pd.DataFrame, names: dict[str, str], dc_of_store: dict[str, str],
    excluded_store_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build valid transfer candidates from real surplus/shortage pairs."""
    product_rows = {int(row["product_id"]): row for _, row in products.iterrows()}
    edges = _route_index(routes)
    surplus_stores = [f"S{i + 1:02d}" for i in range(STORE_COUNT) if _store_profile(i) == "과잉"]
    shortage_stores = [f"S{i + 1:02d}" for i in range(STORE_COUNT) if _store_profile(i) == "부족"]

    pairs: list[tuple[str, str, int]] = []
    for source in surplus_stores:
        for target in shortage_stores:
            if dc_of_store[source] != dc_of_store[target]:
                continue  # VIA_DC only makes sense inside one DC's group
            for product_id in sorted(product_rows):
                if (source, product_id) in inventory_lookup and (target, product_id) in inventory_lookup:
                    pairs.append((source, target, product_id))

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    infeasible_route_ids: list[str] = []
    rng.shuffle(pairs)
    for source, target, product_id in pairs:
        if len(rows) >= MAX_VALID_RECOMMENDATIONS:
            break
        if (source, target, product_id) in seen:
            continue
        source_row = inventory_lookup[(source, product_id)]
        target_row = inventory_lookup[(target, product_id)]
        stock = int(source_row["stock_qty"])
        weekly_need = int(target_row["demand_qty"])
        safety = source_row["demand_std"] * 2.0
        movable = int(min(stock - safety - 1, weekly_need * 2, 60))
        if movable < 5:
            continue
        seen.add((source, target, product_id))

        product = product_rows[product_id]
        cold = bool(product["cold_required"])
        # Alternate DIRECT / VIA_DC so both route types appear for both DCs.
        use_via = len(rows) % 3 == 2
        dc_id = dc_of_store[source] if use_via else None
        if use_via:
            first, second = edges.get((source, dc_id)), edges.get((dc_id, target))
            if first is None or second is None:
                continue
            distance = round(float(first["distance_km"]) + float(second["distance_km"]), 1)
            travel = int(first["travel_time_min"]) + int(second["travel_time_min"])
            cost = int(first["estimated_cost"]) + int(second["estimated_cost"])
            route_type = "VIA_DC"
        else:
            edge = edges.get((source, target))
            if edge is None:
                continue
            distance = float(edge["distance_km"])
            travel = int(edge["travel_time_min"])
            cost = int(edge["estimated_cost"])
            route_type = "DIRECT"

        unit_price = int(product["unit_price"])
        saving = int(movable * unit_price * 0.4) - cost
        if saving <= 0:
            continue
        expiry = int(source_row["days_to_expiry"])
        vhs = round(min(95.0, 45.0 + (30 - min(expiry, 30)) * 1.1 + min(30.0, saving / 12000.0)), 1)
        rows.append({
            "route_id": f"R{len(rows) + 1:04d}",
            "product_id": product_id,
            "product_name": product["product_name"],
            "source_id": source, "source_name": names[source],
            "target_id": target, "target_name": names[target],
            "route_type": route_type,
            "dc_id": dc_id, "dc_name": DC_NAMES[dc_id] if dc_id else None,
            "recommended_qty": movable,
            "transport_type": COLD_TRANSPORT if cold else NORMAL_TRANSPORT,
            "transport_mode": COLD_TRANSPORT if cold else NORMAL_TRANSPORT,
            "estimated_cost": cost, "transport_cost": cost,
            "expected_saving": saving,
            "distance_km": distance, "travel_time_min": travel,
            "vhs_score": vhs,
            "recommendation_grade": "높음" if vhs >= 75 else "보통" if vhs >= 55 else "낮음",
            "confidence_score": round(min(95.0, vhs * 0.95), 1),
            "reason": "출발 점포의 초과 재고를 도착 점포의 부족분으로 이동합니다.",
            "status": "READY",
        })

    # Structurally valid rows that the feasibility gate must block before ranking:
    # the requested quantity is larger than the source store's stock.
    for source, target, product_id in pairs:
        if len(infeasible_route_ids) >= 2:
            break
        if (source, target, product_id) in seen:
            continue
        if dc_of_store[source] != dc_of_store[target]:
            continue
        edge = edges.get((source, target))
        if edge is None:
            continue
        seen.add((source, target, product_id))
        source_row = inventory_lookup[(source, product_id)]
        product = product_rows[product_id]
        route_id = f"R{len(rows) + 1:04d}"
        infeasible_route_ids.append(route_id)
        rows.append({
            "route_id": route_id,
            "product_id": product_id,
            "product_name": product["product_name"],
            "source_id": source, "source_name": names[source],
            "target_id": target, "target_name": names[target],
            "route_type": "DIRECT", "dc_id": None, "dc_name": None,
            "recommended_qty": int(source_row["stock_qty"]) + 500,
            "transport_type": NORMAL_TRANSPORT, "transport_mode": NORMAL_TRANSPORT,
            "estimated_cost": int(edge["estimated_cost"]),
            "transport_cost": int(edge["estimated_cost"]),
            "expected_saving": 90000,
            "distance_km": float(edge["distance_km"]),
            "travel_time_min": int(edge["travel_time_min"]),
            "vhs_score": 70.0, "recommendation_grade": "보통", "confidence_score": 66.0,
            "reason": "출발 점포 재고를 초과하는 요청으로 실행 가능성 검사 대상입니다.",
            "status": "READY",
        })

    # One structurally valid row that references the store whose master row is
    # broken: it must disappear through the reference cascade, not on its own.
    cascade_source = excluded_store_id
    cascade_target = next(
        store for store in shortage_stores
        if store != cascade_source and dc_of_store[store] == dc_of_store[cascade_source]
    )
    cascade_product = next(
        product_id for (store, product_id) in inventory_lookup
        if store == cascade_source and (cascade_target, product_id) in inventory_lookup
    )
    edge = edges[(cascade_source, cascade_target)]
    cascade_route_id = f"R{len(rows) + 1:04d}"
    rows.append({
        "route_id": cascade_route_id,
        "product_id": cascade_product,
        "product_name": product_rows[cascade_product]["product_name"],
        "source_id": cascade_source, "source_name": names[cascade_source],
        "target_id": cascade_target, "target_name": names[cascade_target],
        "route_type": "DIRECT", "dc_id": None, "dc_name": None,
        "recommended_qty": 20, "transport_type": NORMAL_TRANSPORT,
        "transport_mode": NORMAL_TRANSPORT,
        "estimated_cost": int(edge["estimated_cost"]),
        "transport_cost": int(edge["estimated_cost"]),
        "expected_saving": 40000,
        "distance_km": float(edge["distance_km"]),
        "travel_time_min": int(edge["travel_time_min"]),
        "vhs_score": 68.0, "recommendation_grade": "보통", "confidence_score": 64.0,
        "reason": "기준정보 오류 점포를 참조하는 추천으로 연쇄 제외 대상입니다.",
        "status": "READY",
    })
    info = {
        "infeasible_route_ids": infeasible_route_ids,
        "cascade_route_id": cascade_route_id,
        "valid_row_count": len(rows) - len(infeasible_route_ids) - 1,
    }
    return rows, info


# --------------------------------------------------------------------------- #
# Deliberate defects
# --------------------------------------------------------------------------- #
def inject_defects(
    stores: pd.DataFrame, inventory: pd.DataFrame, routes: pd.DataFrame,
    recommendation_rows: list[dict[str, Any]], names: dict[str, str],
    dc_of_store: dict[str, str], excluded_store_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add row-scoped problems only. The file must stay usable after exclusion."""
    plan: dict[str, Any] = {
        "stores": [], "inventory": [], "routes": [], "recommendations": [], "warnings": [],
    }
    # Identifier columns stored as numbers: a *retained* warning on the first data
    # row of each sheet that carries the product code.
    for sheet in ("products", "inventory", "recommendations"):
        plan["warnings"].append({"sheet": sheet, "row": 2, "issue": "id_numeric", "column": "product_id"})

    # --- stores: one master row without a name (row-excludable error) --------- #
    stores = stores.copy()
    store_pos = stores.index[stores["node_id"] == excluded_store_id][0]
    stores.loc[store_pos, "node_name"] = ""
    stores.loc[store_pos, "store_name"] = ""
    plan["stores"].append({
        "row": int(store_pos) + 2, "node_id": excluded_store_id,
        "issue": "missing_required_value", "column": "node_name",
    })

    # --- inventory ------------------------------------------------------------ #
    inventory = inventory.copy()
    inventory["stock_qty"] = inventory["stock_qty"].astype(object)
    inventory["store_id"] = inventory["store_id"].astype(object)

    def pick(offset: int) -> int:
        """A row index that is not the first row and not the broken store."""
        position = offset
        while (
            str(inventory.iloc[position]["store_id"]) == excluded_store_id
            or position < 3
        ):
            position += 1
        return position

    negative_pos = pick(40)
    inventory.iat[negative_pos, inventory.columns.get_loc("stock_qty")] = -12
    plan["inventory"].append({"row": negative_pos + 2, "issue": "negative", "column": "stock_qty"})

    non_numeric_pos = pick(120)
    inventory.iat[non_numeric_pos, inventory.columns.get_loc("stock_qty")] = "십오"
    plan["inventory"].append({"row": non_numeric_pos + 2, "issue": "non_numeric", "column": "stock_qty"})

    multi_pos = pick(200)
    inventory.iat[multi_pos, inventory.columns.get_loc("store_id")] = ""
    inventory.iat[multi_pos, inventory.columns.get_loc("stock_qty")] = -4
    plan["inventory"].append({
        "row": multi_pos + 2, "issue": "missing_id+negative", "column": "store_id, stock_qty",
    })

    # Appended duplicates (exact and conflicting) at the end of the sheet.
    exact_source_pos = pick(60)
    conflict_source_pos = pick(90)
    exact_row = inventory.iloc[exact_source_pos].to_dict()
    conflict_row = inventory.iloc[conflict_source_pos].to_dict()
    conflict_row["stock_qty"] = int(conflict_row["stock_qty"]) + 37
    inventory = pd.concat(
        [inventory, pd.DataFrame([exact_row, conflict_row])], ignore_index=True,
    )
    exact_dup_pos, conflict_dup_pos = len(inventory) - 2, len(inventory) - 1
    plan["inventory"].append({
        "row": exact_dup_pos + 2, "issue": "exact_duplicate", "column": "stock_qty",
        "note": f"원본 {exact_source_pos + 2}행과 동일",
    })
    plan["inventory"].append({
        "row": conflict_dup_pos + 2, "issue": "conflict_duplicate", "column": "stock_qty",
        "note": f"원본 {conflict_source_pos + 2}행과 값 충돌",
    })
    plan["inventory"].append({
        "row": conflict_source_pos + 2, "issue": "conflict_duplicate", "column": "stock_qty",
        "note": f"{conflict_dup_pos + 2}행과 값 충돌",
    })
    # Duplicate findings are warnings: the first row of an exact duplicate pair is
    # kept, every row of a conflicting pair is excluded.
    for row_position, issue in (
        (exact_source_pos, "exact_duplicate"), (exact_dup_pos, "exact_duplicate"),
        (conflict_source_pos, "conflict_duplicate"), (conflict_dup_pos, "conflict_duplicate"),
    ):
        plan["warnings"].append({
            "sheet": "inventory", "row": row_position + 2, "issue": issue, "column": "stock_qty",
        })

    # --- routes --------------------------------------------------------------- #
    routes = routes.copy()
    routes["estimated_cost"] = routes["estimated_cost"].astype(object)
    used_edges = {
        (str(row["source_id"]), str(row["target_id"])) for row in recommendation_rows
    } | {
        (str(row["source_id"]), str(row.get("dc_id"))) for row in recommendation_rows if row.get("dc_id")
    } | {
        (str(row.get("dc_id")), str(row["target_id"])) for row in recommendation_rows if row.get("dc_id")
    }

    def pick_route(offset: int) -> int:
        position = offset
        while True:
            row = routes.iloc[position]
            source, target = str(row["source_id"]), str(row["target_id"])
            if (
                (source, target) not in used_edges
                and excluded_store_id not in (source, target)
                and str(row["route_type"]) == "DIRECT"
            ):
                return position
            position += 1

    negative_route_pos = pick_route(15)
    routes.iat[negative_route_pos, routes.columns.get_loc("distance_km")] = -3.0
    plan["routes"].append({
        "row": negative_route_pos + 2, "issue": "negative", "column": "distance_km",
    })
    bad_cost_pos = pick_route(negative_route_pos + 30)
    routes.iat[bad_cost_pos, routes.columns.get_loc("estimated_cost")] = "미정"
    plan["routes"].append({"row": bad_cost_pos + 2, "issue": "non_numeric", "column": "estimated_cost"})

    # A completely blank row: skipped without shifting the rows below it.
    blank_pos = 250
    blank = pd.DataFrame([{column: None for column in routes.columns}])
    routes = pd.concat(
        [routes.iloc[:blank_pos], blank, routes.iloc[blank_pos:]], ignore_index=True,
    )
    for entry in plan["routes"]:
        if entry["row"] - 2 >= blank_pos:
            entry["row"] += 1
    plan["blank_row"] = {"sheet": "routes", "row": blank_pos + 2}

    # --- recommendations ------------------------------------------------------ #
    rows = [dict(row) for row in recommendation_rows]
    template = rows[0]
    orphan_target = "S99"

    def defect(route_id: str, issue: str, column: str, **overrides: Any) -> None:
        row = dict(template)
        row.update({"route_id": route_id, "reason": "검증용 오류 행입니다."})
        row.update(overrides)
        rows.append(row)
        plan["recommendations"].append({
            "row": len(rows) + 1, "route_id": route_id, "issue": issue, "column": column,
        })

    defect("RX01", "zero_quantity", "recommended_qty", recommended_qty=0)
    via_source = next(row for row in recommendation_rows if row["route_type"] == "VIA_DC")
    defect(
        "RX02", "missing_dc", "dc_id", route_type="VIA_DC", dc_id=None, dc_name=None,
        source_id=via_source["source_id"], source_name=via_source["source_name"],
        target_id=via_source["target_id"], target_name=via_source["target_name"],
        product_id=via_source["product_id"], product_name=via_source["product_name"],
    )
    defect("RX03", "invalid_route_type", "route_type", route_type="경유")
    same_store = template["source_id"]
    defect(
        "RX04", "same_source_target", "target_id",
        target_id=same_store, target_name=names[same_store],
    )
    defect(
        "RX05", "orphan_reference", "target_id",
        target_id=orphan_target, target_name="가상점포 99",
    )
    recommendations = pd.DataFrame(rows)
    return stores, inventory, routes, recommendations, plan


# --------------------------------------------------------------------------- #
# Expected results (derived from what was actually written)
# --------------------------------------------------------------------------- #
def _nonblank_rows(frame: pd.DataFrame) -> list[int]:
    def blank(value: Any) -> bool:
        return value is None or (isinstance(value, float) and value != value) or str(value).strip() == ""
    return [index for index, row in frame.iterrows() if not all(blank(value) for value in row)]


def build_manifest(
    stores: pd.DataFrame, products: pd.DataFrame, inventory: pd.DataFrame,
    routes: pd.DataFrame, recommendations: pd.DataFrame, plan: dict[str, Any],
    rec_info: dict[str, Any], excluded_store_id: str, dc_of_store: dict[str, str],
) -> dict[str, Any]:
    frames = {
        "stores": stores, "products": products, "inventory": inventory,
        "routes": routes, "recommendations": recommendations,
    }
    source_rows = {name: len(_nonblank_rows(frame)) for name, frame in frames.items()}

    # 1) Rows removed by their own row-scoped problem.
    direct_excluded: set[tuple[str, int]] = set()
    for sheet, entries in (
        ("stores", plan["stores"]), ("inventory", plan["inventory"]),
        ("routes", plan["routes"]), ("recommendations", plan["recommendations"]),
    ):
        for entry in entries:
            if entry["issue"] == "orphan_reference":
                continue  # resolved in the cascade pass below
            direct_excluded.add((sheet, int(entry["row"])))

    # 2) Rows removed because the master row they reference is gone.
    cascade_excluded: set[tuple[str, int]] = set()
    for position, row in inventory.reset_index(drop=True).iterrows():
        if ("inventory", position + 2) in direct_excluded:
            continue
        if str(row["store_id"]).strip() == excluded_store_id:
            cascade_excluded.add(("inventory", position + 2))
    for position, row in routes.reset_index(drop=True).iterrows():
        if ("routes", position + 2) in direct_excluded:
            continue
        if excluded_store_id in (str(row["source_id"]).strip(), str(row["target_id"]).strip()):
            cascade_excluded.add(("routes", position + 2))
    surviving_edges = {
        (str(row["source_id"]).strip(), str(row["target_id"]).strip())
        for position, row in routes.reset_index(drop=True).iterrows()
        if ("routes", position + 2) not in direct_excluded | cascade_excluded
    }
    for position, row in recommendations.reset_index(drop=True).iterrows():
        ref = ("recommendations", position + 2)
        if ref in direct_excluded:
            continue
        source, target = str(row["source_id"]).strip(), str(row["target_id"]).strip()
        dc_id = "" if pd.isna(row.get("dc_id")) else str(row.get("dc_id")).strip()
        broken_node = excluded_store_id in (source, target) or target == "S99" or source == "S99"
        route_type = str(row["route_type"]).strip().upper()
        if broken_node:
            cascade_excluded.add(ref)
        elif route_type == "DIRECT" and (source, target) not in surviving_edges:
            cascade_excluded.add(ref)
        elif route_type == "VIA_DC" and (
            (source, dc_id) not in surviving_edges or (dc_id, target) not in surviving_edges
        ):
            cascade_excluded.add(ref)

    excluded = sorted(direct_excluded | cascade_excluded)
    warning_refs = {(entry["sheet"], int(entry["row"])) for entry in plan["warnings"]}
    retained_warnings = sorted(warning_refs - set(excluded))
    excluded_by_table: dict[str, int] = {}
    for sheet, _row in excluded:
        excluded_by_table[sheet] = excluded_by_table.get(sheet, 0) + 1

    node_type = stores["node_type"].astype(str).str.upper()
    route_type_column = routes["route_type"].astype(str).str.upper()
    rec_type = recommendations["route_type"].astype(str).str.upper()
    rec_dc = recommendations["dc_id"].astype(str)

    def dc_route_rows(dc_id: str) -> int:
        return int((
            (routes["source_id"].astype(str) == dc_id) | (routes["target_id"].astype(str) == dc_id)
        ).sum())

    total_rows = sum(source_rows.values())
    return {
        "generator": "tools/generate_anonymized_operational_workbook.py",
        "seed": SEED,
        "reference_date": REFERENCE_DATE.isoformat(),
        "anonymization": {
            "store_name_pattern": "가상점포 NN",
            "dc_name_pattern": "가상물류센터 N",
            "product_name_pattern": "가상상품 NN",
            "region_name_pattern": "가상권역 X",
            "coordinates": "실제 주소가 아닌 인공 격자 좌표 (위도 10.x / 경도 10.x)",
            "contains_personal_data": False,
            "contains_real_company_data": False,
        },
        "scale": {
            "store_count": int((node_type == "STORE").sum()),
            "dc_count": int((node_type == "DC").sum()),
            "product_count": int(len(products)),
            "source_rows_by_table": source_rows,
            "total_source_rows": total_rows,
        },
        "routes": {
            "direct_rows": int((route_type_column == "DIRECT").sum()),
            "via_dc_rows": int((route_type_column == "VIA_DC").sum()),
            "dc01_rows": dc_route_rows("DC01"),
            "dc02_rows": dc_route_rows("DC02"),
        },
        "recommendations": {
            "total_rows": int(len(recommendations)),
            "direct_rows": int((rec_type == "DIRECT").sum()),
            "via_dc_rows": int((rec_type == "VIA_DC").sum()),
            "dc01_rows": int((rec_dc == "DC01").sum()),
            "dc02_rows": int((rec_dc == "DC02").sum()),
            "feasibility_blocked_route_ids": rec_info["infeasible_route_ids"],
            "cascade_excluded_route_id": rec_info["cascade_route_id"],
        },
        "issues": {
            "file_blocking_count": 0,
            "row_excludable_rows": len(direct_excluded),
            "cascade_excluded_rows": len(cascade_excluded),
            "warning_rows": len(warning_refs),
            "retained_warning_rows": len(retained_warnings),
            "retained_warning_detail": (
                "product_id 가 숫자로 저장된 3개 시트(id_numeric)와 완전 중복 쌍의 첫 원본 행은 "
                "경고만 표시하고 적용 데이터에 남깁니다."
            ),
            "injected": plan,
        },
        "expected": {
            "total_rows": total_rows,
            "excluded_rows": len(excluded),
            "applied_rows": total_rows - len(excluded),
            "warning_rows": len(warning_refs),
            "warning_included_rows": len(retained_warnings),
            "excluded_by_table": excluded_by_table,
            "excluded_row_refs": [
                {"source_sheet": sheet, "source_row_number": row} for sheet, row in excluded
            ],
            "retained_warning_row_refs": [
                {"source_sheet": sheet, "source_row_number": row} for sheet, row in retained_warnings
            ],
            "usable_rows_by_table": {
                name: source_rows[name] - excluded_by_table.get(name, 0) for name in source_rows
            },
            "apply_allowed": True,
            "candidate_generation_possible": True,
            "minimum_recommendation_count": 1,
            "orphan_references_after_exclusion": 0,
            "excluded_store_id": excluded_store_id,
        },
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _config_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"key": "distance_cutline_km", "value": 25, "description": "기본 이동 거리 기준"},
        {"key": "available_start", "value": "08:00", "description": "기본 거래 시작 시간"},
        {"key": "available_end", "value": "22:00", "description": "기본 거래 종료 시간"},
    ])


def _readme_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "파일 용도", "description": "Varo V2 운영 형식 검증용 익명화 데이터"},
        {"item": "개인정보", "description": "포함하지 않음 (모든 점포·상품·DC 이름은 가상 값)"},
        {"item": "실제 업체 정보", "description": "포함하지 않음"},
        {"item": "생성 방법", "description": "tools/generate_anonymized_operational_workbook.py (고정 seed)"},
        {"item": "주의", "description": "의도적인 오류 행과 경고 행이 포함되어 있습니다."},
    ])


def generate(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    rng = Random(SEED)
    stores, names, dc_of_store = build_stores()
    products = build_products(rng)
    inventory, inventory_lookup = build_inventory(rng, products, names, stores)
    routes = build_routes(names, dc_of_store)
    # A store whose master row is broken on purpose, to validate the cascade.
    excluded_store_id = "S07"
    recommendation_rows, rec_info = build_recommendations(
        rng, products, inventory_lookup, routes, names, dc_of_store, excluded_store_id,
    )
    stores, inventory, routes, recommendations, plan = inject_defects(
        stores, inventory, routes, recommendation_rows, names, dc_of_store, excluded_store_id,
    )

    manifest = build_manifest(
        stores, products, inventory, routes, recommendations, plan, rec_info,
        excluded_store_id, dc_of_store,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / WORKBOOK_NAME
    manifest_path = output_dir / MANIFEST_NAME
    frames = {
        "stores": stores, "products": products, "inventory": inventory,
        "routes": routes, "v2_recommendations": recommendations,
        "config": _config_frame(), "README": _readme_frame(),
    }
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet in SHEET_ORDER:
            frames[sheet].to_excel(writer, sheet_name=sheet, index=False)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return workbook_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="워크북과 manifest를 쓸 폴더 (기본: validation_data/)",
    )
    args = parser.parse_args()
    workbook_path, manifest_path = generate(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scale, expected = manifest["scale"], manifest["expected"]
    print(f"워크북: {workbook_path}")
    print(f"기대값 manifest: {manifest_path}")
    print(
        f"점포 {scale['store_count']} · DC {scale['dc_count']} · 상품 {scale['product_count']} · "
        f"전체 {expected['total_rows']}행 (적용 {expected['applied_rows']} / 제외 {expected['excluded_rows']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
