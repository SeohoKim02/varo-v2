"""Candidate generation research path (P1): the generator's lanes before its cuts, explicit validity, pool policies.

Research and parallel only.  ``candidate_generator.generate_candidates`` (production) keeps, per eligible (source,
product) inventory row, the one reachable target with the largest need (road distance, or route cost outside the
real-transport mode, breaks ties), then the MAX_CANDIDATES best by candidate_score.  This module enumerates every lane
that loop looks at, with the generator's own arithmetic, and records why a lane is or is not in a pool:

* ``enumerate_universe``  every (source row, target) lane with the generator's need, quantity and score, the target
                          rank inside its row and the production rank of the kept target.  Policy LEGACY_20 reproduces
                          the production output column for column.
* ``price_universe``      move cost from an injected pricer (the official-tariff engine on real data).  A cost that
                          cannot be computed stays NULL with provenance UNKNOWN; it is never 0.
* ``attach_benchmark_caps``  the offline benchmark caps of ``suhyup_algorithm_revalidation`` (PROXY).
* ``validate_universe``   explicit per-lane validity: keys, self move, route evidence, product at source, duplicates,
                          target holds the product, evidenced need, unit, quantity, cost, caps.
* ``policy_pool``         LEGACY_20 / TOP_30 / TOP_40 / GENERATOR_ALL / MULTI_TARGET / ALL_VALID / ALL_LANES.
* ``evaluate_pool``       the unchanged Varo Final order, the T1 selector and the lexicographic MILP on one pool, with
                          T3 comparison signatures and independent checks.
* coverage, diversity and shortage diagnostics with explicit numerators and denominators.

Nothing here changes candidate_generator, the production pool, Varo Final ranks, Top-5, T1 plans, T2 actions or KPIs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import tracemalloc
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from services import candidate_generator as cg
from services import milp_benchmark_integrity as mbi
from services import shared_feasibility_selection as sf
from services import suhyup_algorithm_revalidation as rv
from services.column_aliases import clean_numeric_value

RESEARCH_VERSION = "candidate-generation-research-v0.1"
MAX_ROUTES = rv.MAX_DAILY_ROUTES

# ------------------------------------------------------------------------------------------------ vocabularies

LEGACY_20, TOP_30, TOP_40, GENERATOR_ALL = "LEGACY_20", "TOP_30", "TOP_40", "GENERATOR_ALL"
MULTI_TARGET, ALL_VALID, ALL_LANES = "MULTI_TARGET", "ALL_VALID", "ALL_LANES"
POLICIES = (LEGACY_20, TOP_30, TOP_40, GENERATOR_ALL, MULTI_TARGET, ALL_VALID, ALL_LANES)
GENERATOR_FAMILY = (LEGACY_20, TOP_30, TOP_40, GENERATOR_ALL)
POLICY_SPEC: dict[str, dict[str, Any]] = {
    LEGACY_20: {"family": "P0", "limit": "MAX_CANDIDATES",
                "rule": "production generator: per source-product row the target with the largest need, then the top "
                        "MAX_CANDIDATES by candidate_score"},
    TOP_30: {"family": "P1", "limit": 30, "rule": "production generator order, cut at 30 instead of MAX_CANDIDATES"},
    TOP_40: {"family": "P1", "limit": 40, "rule": "production generator order, cut at 40 instead of MAX_CANDIDATES"},
    GENERATOR_ALL: {"family": "P1", "limit": None, "rule": "every target the generator keeps (one per source-product row), no cut"},
    MULTI_TARGET: {"family": "P2", "limit": None,
                   "rule": "every reachable target whose need is evidenced by its own inventory row (generator need > 0), "
                           "generator quantity rule, several targets per source-product, no cut"},
    ALL_VALID: {"family": "P3", "limit": None, "rule": "every lane that passes validate_universe (VALID), no cut"},
    ALL_LANES: {"family": "P4", "limit": None,
                "rule": "every reachable target per eligible source-product row (the T3 lane pool), unfiltered; lanes "
                        "to a target without the product carry no target cap"},
}
# Selectable pool modes (design for a later, separately approved production option).
POOL_MODES = {
    "LEGACY_20": "production default: generator order, MAX_CANDIDATES cut (unchanged)",
    "EXPANDED": "generator order with an explicit candidate count N (TOP_30, TOP_40) or no cut (GENERATOR_ALL)",
    "ALL_VALID": "every explicitly valid lane, several targets per source-product, no count cut",
}

ROUTE_NETWORK_TABLE, ROUTE_VIA_DC_DERIVED = "ROUTE_NETWORK_TABLE", "ROUTE_VIA_DC_DERIVED"
ROUTE_UNVERIFIED, ROUTE_FORBIDDEN = "ROUTE_UNVERIFIED", "ROUTE_FORBIDDEN"
ROUTE_OK = (ROUTE_NETWORK_TABLE, ROUTE_VIA_DC_DERIVED)
OPERATION_NOT_OBSERVED = "NOT_OBSERVED_IN_INPUT"
ROUTE_FLAG_FIELDS = ("feasible", "route_feasible", "is_feasible", "allowed")

# Cost provenance: the Seller Loss taxonomy as T1/T3 use it; a proxy vehicle class makes a tariff cost PROXY
# (seller_decision_validation).  CONFIG = the generator's distance x 100 default.  UNKNOWN = not computable.
DIRECT_REAL, DERIVED_REAL, PROXY, USER_INPUT, CONFIG, UNKNOWN = (
    "DIRECT_REAL", "DERIVED_REAL", "PROXY", "USER_INPUT", "CONFIG", "UNKNOWN")
COST_PROVENANCE = (DIRECT_REAL, DERIVED_REAL, PROXY, USER_INPUT, CONFIG, UNKNOWN)

VALID = "VALID"
UNIT_FIELDS = ("quantity_unit", "qty_unit", "unit", "uom")
VALIDITY_CHECKS = (
    ("chk_keys", "INVALID_KEY"), ("chk_not_self", "SELF_MOVE"), ("chk_route", None),
    ("chk_product_at_source", "PRODUCT_NOT_AT_SOURCE"), ("chk_unique", "DUPLICATE"),
    ("chk_target_holds_product", "TARGET_PRODUCT_NOT_HELD"), ("chk_need_evidenced", "NEED_NOT_EVIDENCED"),
    ("chk_unit", None), ("chk_qty", "QTY_INVALID"), ("chk_cost", "COST_UNKNOWN"),
    ("chk_source_cap", "SOURCE_CAP_MISSING_OR_ZERO"), ("chk_target_cap", "TARGET_CAP_MISSING_OR_ZERO"),
)
SHORTAGE_CATEGORIES = (
    "IN_POOL", "PRODUCT_NOT_IN_MASTER", "SOURCE_NO_STOCK", "SOURCE_NOT_SURPLUS", "NO_ROUTE", "NO_TARGET_NEED",
    "UNIT_UNKNOWN", "QTY_INVALID", "COST_UNKNOWN", "CAP_MISSING_OR_ZERO", "SAVING_NOT_POSITIVE", "DUPLICATE_ROW",
    "CUT_BY_ONE_TARGET_RULE", "CUT_BY_LIMIT", "NOT_IN_POLICY_RULE",
)
PRODUCTION_COLUMNS = (
    "route_id", "product_id", "product_name", "source_id", "source_name", "target_id", "target_name", "dc_id", "dc_name",
    "route_type", "transport_type", "recommended_qty", "estimated_cost", "expected_saving", "candidate_cost_status",
    "candidate_economics_status", "distance_km", "travel_time_min", "direct_cost", "via_dc_cost", "direct_distance_km",
    "via_dc_distance_km", "vhs_score", "recommendation_grade", "confidence_score", "reason",
)
EPS = 1e-9


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    return value is True or _text(value).lower() in {"true", "1", "yes"}


def lane_key(product: str, source: str, target: str, route_type: str = "DIRECT", dc: str = "") -> str:
    """One physical move: product, source, target, route type and DC (the T1 / MILP duplicate key)."""
    return "/".join((str(product), str(source), str(target), str(route_type or "DIRECT"), str(dc or "")))


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _unit(row: Mapping[str, Any] | None) -> str | None:
    for name in UNIT_FIELDS:
        unit = sf.normalize_unit((row or {}).get(name))
        if unit:
            return unit
    return None


# ------------------------------------------------------------------------------------------------ universe


def real_transport_mode(data: Mapping[str, Any]) -> bool:
    """Same switch as candidate_generator: every store row DIRECT_NETWORK and VARO_REAL_DATA_ROOT set."""
    stores = data.get("stores")
    if not isinstance(stores, pd.DataFrame) or stores.empty or "network_mode" not in stores.columns:
        return False
    direct = stores["network_mode"].fillna("").astype(str).str.strip().str.upper().eq("DIRECT_NETWORK").all()
    return bool(direct and os.environ.get("VARO_REAL_DATA_ROOT", "").strip())


def _route_evidence(routes: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Per directed (source, target) row of the uploaded routes table: distance provenance and explicit flags."""
    if routes is None or routes.empty:
        return {}
    src = _first_column(routes, ("source_id", "from_id", "from_store_id"))
    tgt = _first_column(routes, ("target_id", "to_id", "to_store_id"))
    if src is None or tgt is None:
        return {}
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for row in routes.to_dict("records"):
        s, t = _text(row.get(src)), _text(row.get(tgt))
        if not s or not t or (s, t) in evidence:  # first row wins, as candidate_generator._route_lookup
            continue
        forbidden = next((name for name in ROUTE_FLAG_FIELDS if name in row and _text(row.get(name))
                          and not _truthy(row.get(name)) and _text(row.get(name)).lower() in {"false", "0", "no"}), None)
        evidence[(s, t)] = {"distance_provenance": _text(row.get("distance_provenance")) or "UNDECLARED",
                            "forbidden_flag": forbidden}
    return evidence


def enumerate_universe(data: Mapping[str, Any], *, transport_mode: bool | None = None,
                       snapshot_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Every lane the generator loop looks at, with its arithmetic, before the one-target and MAX_CANDIDATES cuts.

    Returns (lanes, pairs, trace).  ``pairs`` has one row per inventory row at a store node (eligibility and reason);
    ``lanes`` one row per (eligible row, other store) including targets without a route (ROUTE_UNVERIFIED, never in a
    pool); ``trace`` the production step counts.  Raises nothing for a bad upload: empty frames and a reason.
    """
    empty = pd.DataFrame()
    trace: dict[str, Any] = {"generated": False, "reason": None}
    if not cg.can_generate(dict(data)):
        trace["reason"] = "stores/products/inventory/routes 데이터가 부족합니다."
        return empty, empty, trace
    real = real_transport_mode(data) if transport_mode is None else bool(transport_mode)
    stores, inventory = data["stores"], data["inventory"].copy()
    store_ids, dc_id = cg._store_ids_by_type(stores)
    names = cg._name_lookup(stores)
    product_info = cg._product_info(data["products"])
    direct = cg._route_lookup(data["routes"])
    evidence = _route_evidence(data["routes"])
    store_col = _first_column(inventory, ("store_id", "node_id"))
    product_col = _first_column(inventory, ("product_id", "item_id"))
    stock_col = _first_column(inventory, ("stock_qty", "current_stock", "quantity"))
    if not store_ids or not direct or not all((store_col, product_col, stock_col)):
        trace["reason"] = "점포·경로·재고 키가 없어 후보를 생성할 수 없습니다."
        return empty, empty, trace
    expiry_col = _first_column(inventory, ("days_to_expiry", "expiry_days"))
    demand_col = _first_column(inventory, ("avg_daily_sales", "sales_qty", "demand_qty"))

    trace.update({"inventory_rows_input": int(len(inventory)), "store_nodes": len(store_ids), "dc_id": dc_id,
                  "real_transport_mode": real, "max_candidates": cg.MAX_CANDIDATES, "move_cap": cg._MOVE_CAP,
                  "short_expiry_cap": cg._SHORT_EXPIRY_CAP})
    inventory["_inventory_row"] = range(len(inventory))
    inv = inventory[inventory[store_col].astype(str).isin(store_ids)].copy()
    inv["_store"] = inv[store_col].astype(str)
    inv["_product"] = inv[product_col].astype(str)
    inv["_stock"] = inv[stock_col].map(cg._num)
    inv["_expiry"] = inv[expiry_col].map(clean_numeric_value) if expiry_col else None
    inv["_demand"] = inv[demand_col].map(cg._num) if demand_col else 0.0
    stock_by = {(r["_store"], r["_product"]): r["_stock"] for _, r in inv.iterrows()}
    demand_by = {(r["_store"], r["_product"]): r["_demand"] for _, r in inv.iterrows()}
    row_by = {(r["_store"], r["_product"]): r for r in inv.to_dict("records")}
    median_stock = inv.groupby("_product")["_stock"].median().to_dict()
    surplus_range = {p: max(1.0, float(g["_stock"].max() - g["_stock"].min())) for p, g in inv.groupby("_product")}
    trace["inventory_rows_at_store_nodes"] = int(len(inv))

    pairs: list[dict[str, Any]] = []
    lanes: list[dict[str, Any]] = []
    lane_seen: dict[str, int] = {}
    counts = Counter()
    for item in inv.to_dict("records"):
        source, product, stock = item["_store"], item["_product"], item["_stock"]
        expiry = item["_expiry"] if expiry_col else None
        expiry = None if (expiry is None or pd.isna(expiry)) else float(expiry)
        median = median_stock.get(product, stock)
        surplus = stock - median
        pair = {"inventory_row": int(item["_inventory_row"]), "source_id": source, "product_id": product,
                "gen_source_stock": stock, "gen_median_stock": median, "gen_source_surplus_raw": surplus,
                "expiry_days": expiry, "eligibility": "ELIGIBLE", "reachable_targets": 0, "generator_choice_lane": None}
        if product not in product_info:
            pair["eligibility"] = "PRODUCT_NOT_IN_MASTER"
        elif stock <= 0:
            pair["eligibility"] = "SOURCE_NO_STOCK"
        elif not (surplus > 0 or (expiry is not None and expiry <= 7)):
            pair["eligibility"] = "SOURCE_NOT_SURPLUS"
        counts[pair["eligibility"]] += 1
        pairs.append(pair)
        if pair["eligibility"] != "ELIGIBLE":
            continue
        targets, unreachable = [], []
        for target in store_ids:
            if target == source:
                continue
            resolved = cg._resolve_route(source, target, dc_id, direct)
            if resolved is None:
                unreachable.append(target)
                continue
            target_stock = stock_by.get((target, product), median)
            need = max(0.0, median - target_stock) + demand_by.get((target, product), 0.0) * 7
            sort_value = (-float(resolved["route"].get("distance_km") or 0.0) if real
                          else -resolved["route"]["estimated_cost"])
            targets.append((need, sort_value, target, resolved))
        pair["reachable_targets"] = len(targets)
        if not targets:
            counts["route_deferred"] += 1
        targets.sort(reverse=True)  # candidate_generator: need desc, then distance (cost) asc, then target id desc
        source_rule = max(1.0, surplus if surplus > 0 else stock * 0.3)
        move_cap = cg._SHORT_EXPIRY_CAP if (expiry is not None and expiry <= 3) else cg._MOVE_CAP
        source_unit = _unit(item)
        for rank, (need, sort_value, target, resolved) in enumerate(targets, start=1):
            route = resolved["route"]
            route_type = resolved["route_type"]
            dc = dc_id if route_type == "VIA_DC" else ""
            key = lane_key(product, source, target, route_type, dc)
            target_row = row_by.get((target, product))
            target_stock = stock_by.get((target, product), median)
            need_rule = max(1.0, need if need > 0 else source_rule)
            raw_qty = min(source_rule, need_rule, stock, move_cap)
            moved = int(max(1, raw_qty))
            if real:
                cost, saving, unit_price = 0.0, 0.0, None
            else:
                unit_price = product_info[product]["unit_price"]
                cost = route["estimated_cost"] or (route["distance_km"] * 100.0)
                saving = moved * unit_price * cg._DISPOSAL_FRACTION - cost
            first = lane_seen.setdefault(key, int(item["_inventory_row"]))
            direct_leg, via_leg = resolved.get("direct"), resolved.get("via")
            if route_type == "DIRECT":
                route_status = ROUTE_FORBIDDEN if (evidence.get((source, target)) or {}).get("forbidden_flag") else ROUTE_NETWORK_TABLE
                distance_provenance = (evidence.get((source, target)) or {}).get("distance_provenance", "UNDECLARED")
            else:
                legs = [evidence.get((source, dc_id)) or {}, evidence.get((dc_id, target)) or {}]
                route_status = ROUTE_FORBIDDEN if any(leg.get("forbidden_flag") for leg in legs) else ROUTE_VIA_DC_DERIVED
                distance_provenance = "+".join(leg.get("distance_provenance", "UNDECLARED") for leg in legs)
            target_unit = _unit(target_row)
            if source_unit is None and target_unit is None:
                unit_status = "UNDECLARED_SAME_SOURCE"
            elif source_unit is None or target_unit is None:
                unit_status = "UNIT_UNKNOWN"
            else:
                unit_status = "MATCHED" if source_unit == target_unit else "UNIT_MISMATCH"
            distance = route["distance_km"] or 0.0
            lanes.append({
                "snapshot_date": snapshot_date, "candidate_id": key, "lane_key": key,
                "inventory_row": int(item["_inventory_row"]), "first_inventory_row": first,
                "is_duplicate": first != int(item["_inventory_row"]),
                "product_id": product, "product_name": product_info[product]["name"],
                "source_id": source, "source_name": names.get(source, source),
                "target_id": target, "target_name": names.get(target, target),
                "dc_id": dc or None, "dc_name": (names.get(dc_id, dc_id) if dc else None),
                "route_type": route_type, "transport_type": "일반 탑차",
                "gen_source_stock": stock, "gen_median_stock": median, "gen_source_surplus_raw": surplus,
                "gen_surplus_qty_rule": source_rule, "gen_target_stock": target_stock,
                "target_has_inventory_row": target_row is not None,
                "gen_target_demand": demand_by.get((target, product), 0.0), "gen_target_need": need,
                "gen_target_need_qty_rule": need_rule, "need_fabricated": not need > 0,
                "move_cap": move_cap, "qty_limited_by_move_cap": raw_qty > move_cap - EPS and min(source_rule, need_rule, stock) > move_cap,
                "expiry_days": expiry, "recommended_qty": moved,
                "generator_target_rank": rank, "generator_target_count": len(targets), "generator_choice": rank == 1,
                "target_sort_value": sort_value,
                "generator_cost": round(cost, 1), "gen_saving": saving, "expected_saving": round(saving, 1),
                "unit_price": unit_price,
                "saving_excluded": (not real) and saving <= 0,
                "expiry_score": cg._expiry_score(expiry),
                "surplus_score": cg._clamp((surplus / surplus_range.get(product, 1.0)) * 100.0),
                "need_score": cg._clamp((need / max(1.0, median)) * 100.0),
                "route_score": 100.0 if route_type == "DIRECT" else 70.0,
                "distance_score": cg._clamp(100.0 - distance * 5.0),
                "distance_km": route["distance_km"], "travel_time_min": route["travel_time_min"],
                "route_estimated_cost": route["estimated_cost"],
                "direct_cost": (direct_leg or {}).get("estimated_cost"), "via_dc_cost": (via_leg or {}).get("estimated_cost"),
                "direct_distance_km": (direct_leg or {}).get("distance_km"),
                "via_dc_distance_km": (via_leg or {}).get("distance_km"),
                "selected_route_basis": resolved["basis"], "route_status": route_status,
                "route_operational_evidence": OPERATION_NOT_OBSERVED, "distance_provenance": distance_provenance,
                "quantity_unit": source_unit or target_unit, "unit_status": unit_status,
                "real_transport_mode": real,
            })
        for target in unreachable:
            key = lane_key(product, source, target)
            lanes.append({
                "snapshot_date": snapshot_date, "candidate_id": key, "lane_key": key,
                "inventory_row": int(item["_inventory_row"]), "first_inventory_row": lane_seen.setdefault(key, int(item["_inventory_row"])),
                "is_duplicate": lane_seen[key] != int(item["_inventory_row"]),
                "product_id": product, "product_name": product_info[product]["name"], "source_id": source,
                "source_name": names.get(source, source), "target_id": target, "target_name": names.get(target, target),
                "route_type": "DIRECT", "route_status": ROUTE_UNVERIFIED, "generator_choice": False,
                "generator_target_rank": None, "recommended_qty": None, "real_transport_mode": real,
                "target_has_inventory_row": row_by.get((target, product)) is not None,
                "route_operational_evidence": OPERATION_NOT_OBSERVED, "distance_provenance": None,
                "unit_status": None, "saving_excluded": False,
            })
        if targets:
            pair["generator_choice_lane"] = lane_key(product, source, targets[0][2], targets[0][3]["route_type"],
                                                     dc_id if targets[0][3]["route_type"] == "VIA_DC" else "")

    frame = pd.DataFrame(lanes)
    pair_frame = pd.DataFrame(pairs)
    if frame.empty:
        trace.update(reason="이동 후보 lane이 없습니다.", **{f"rows_{k.lower()}": v for k, v in counts.items()})
        return frame, pair_frame, trace
    for column in ("gen_source_stock", "gen_median_stock", "gen_source_surplus_raw", "gen_surplus_qty_rule", "gen_target_stock",
                   "gen_target_demand", "gen_target_need", "gen_target_need_qty_rule", "need_fabricated", "move_cap",
                   "qty_limited_by_move_cap", "expiry_days", "generator_target_count", "target_sort_value", "generator_cost",
                   "gen_saving", "expected_saving", "unit_price", "expiry_score", "surplus_score", "need_score", "route_score",
                   "distance_score", "distance_km", "travel_time_min", "route_estimated_cost", "direct_cost", "via_dc_cost",
                   "direct_distance_km", "via_dc_distance_km", "selected_route_basis", "quantity_unit", "dc_id", "dc_name",
                   "transport_type"):
        if column not in frame.columns:  # a day whose lanes all lack a route
            frame[column] = None

    # production order of the kept targets: dedupe on (product, source, target) in row order, then (legacy mode)
    # positive saving, then a stable sort by candidate_score
    choices = frame[frame["generator_choice"].fillna(False).astype(bool)]
    kept, duplicate_removed, negative = [], 0, 0
    seen_choice: set[tuple[str, str, str]] = set()
    for index, row in choices.iterrows():
        key3 = (row["product_id"], row["source_id"], row["target_id"])
        if key3 in seen_choice:
            duplicate_removed += 1
            continue
        seen_choice.add(key3)
        if row["saving_excluded"]:
            negative += 1
            continue
        kept.append(index)
    # scalar arithmetic and Python round(), exactly as candidate_generator (pandas rounding can differ at halves);
    # saving normalised over the kept targets, as production; other lanes use the same constants, clamped
    savings = [float(frame.at[index, "gen_saving"]) for index in kept] or [0.0]
    s_min, s_max = min(savings), max(savings)
    s_range = max(1.0, s_max - s_min)
    saving_scores, scores = [], []
    for row in frame.to_dict("records"):
        if row["route_status"] not in ROUTE_OK + (ROUTE_FORBIDDEN,):
            saving_scores.append(None)
            scores.append(None)
            continue
        saving_score = cg._clamp((row["gen_saving"] - s_min) / s_range * 100.0)
        saving_scores.append(round(saving_score, 1))
        scores.append(round(cg._clamp(
            0.25 * row["expiry_score"] + 0.20 * row["surplus_score"] + 0.20 * row["need_score"]
            + 0.20 * saving_score + 0.10 * row["route_score"] + 0.05 * row["distance_score"]), 1))
    frame["saving_score"], frame["candidate_score"] = saving_scores, scores
    order = sorted(kept, key=lambda index: frame.at[index, "candidate_score"], reverse=True)  # stable, as production
    frame["generator_rank"] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    for rank, index in enumerate(order, start=1):
        frame.at[index, "generator_rank"] = rank
    frame["generator_route_id"] = [None if rank is None or pd.isna(rank) else f"V2C{int(rank):03d}"
                                   for rank in frame["generator_rank"]]
    frame["in_production_order"] = frame["generator_rank"].notna()
    # a missing generator id is NaN in the frame (truthy), so test for a string explicitly
    frame["route_id"] = [
        gen if isinstance(gen, str) and gen else (f"V2L-{row['product_id']}-{row['source_id']}-{row['target_id']}"
                         + (f"-VIA-{row['dc_id']}" if row.get("route_type") == "VIA_DC" else "")
                         + (f"#DUP{row['inventory_row']}" if row.get("is_duplicate") else "")
                         + ("#NOROUTE" if row.get("route_status") == ROUTE_UNVERIFIED else ""))
        for gen, row in zip(frame["generator_route_id"], frame.to_dict("records"))]
    if frame["route_id"].isna().any() or frame["route_id"].duplicated().any():
        raise AssertionError("candidate route ids must be present and unique")
    frame["recommendation_id"] = frame["route_id"]
    frame["estimated_cost"] = frame["generator_cost"]
    frame["candidate_cost_status"] = "deferred_real_transport" if real else "legacy_route_estimate"
    frame["candidate_economics_status"] = "actual_price_unavailable" if real else "legacy_estimated_saving"
    frame["vhs_score"], frame["recommendation_grade"], frame["confidence_score"] = 50.0, "보통", 50.0
    frame["reason"] = "추천 결과 시트가 없어 재고·경로 데이터로 생성한 V2 기본 후보입니다."
    frame["lineage"] = [
        f"inventory_row={row['inventory_row']}; first_row={row['first_inventory_row']}; "
        f"target_rank_in_row={row.get('generator_target_rank')}/{row.get('generator_target_count')}; "
        f"generator_rank={'NOT_IN_PRODUCTION_ORDER' if pd.isna(row.get('generator_rank')) else int(row['generator_rank'])}"
        for row in frame.to_dict("records")]
    trace.update({
        "generated": True, "reason": None,
        "pairs_eligible": int(counts["ELIGIBLE"]), "pairs_product_not_in_master": int(counts["PRODUCT_NOT_IN_MASTER"]),
        "pairs_no_stock": int(counts["SOURCE_NO_STOCK"]), "pairs_not_surplus": int(counts["SOURCE_NOT_SURPLUS"]),
        "pairs_route_deferred": int(counts["route_deferred"]),
        "lanes_considered": int(len(frame)), "lanes_with_route": int(frame["route_status"].isin(ROUTE_OK).sum()),
        "lanes_route_unverified": int((frame["route_status"] == ROUTE_UNVERIFIED).sum()),
        "lanes_route_forbidden_flag": int((frame["route_status"] == ROUTE_FORBIDDEN).sum()),
        "one_target_rule_kept": int(len(choices)),
        "one_target_rule_dropped": int(frame["route_status"].isin(ROUTE_OK + (ROUTE_FORBIDDEN,)).sum() - len(choices)),
        "duplicate_removed": duplicate_removed, "negative_saving_excluded": negative,
        "generator_pool_before_cut": len(order), "kept_by_max_candidates": min(len(order), cg.MAX_CANDIDATES),
        "cut_by_max_candidates": max(0, len(order) - cg.MAX_CANDIDATES),
        "cut_boundary_tie": bool(len(order) > cg.MAX_CANDIDATES and frame.at[order[cg.MAX_CANDIDATES - 1], "candidate_score"]
                                 == frame.at[order[cg.MAX_CANDIDATES], "candidate_score"]),
    })
    return frame, pair_frame, trace


def production_records(pool: pd.DataFrame) -> pd.DataFrame:
    """The pool in candidate_generator.generate_candidates' output columns (dc_id / dc_name None for DIRECT)."""
    if pool.empty:
        return pd.DataFrame(columns=list(PRODUCTION_COLUMNS))
    frame = pool.copy()
    frame["route_id"] = frame["generator_route_id"].where(frame["generator_route_id"].notna(), frame["route_id"])
    frame["estimated_cost"] = frame["generator_cost"]
    frame["recommended_qty"] = pd.to_numeric(frame["recommended_qty"]).astype(int)  # the generator emits int()
    return frame[list(PRODUCTION_COLUMNS)].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ pricing and caps


Pricer = Callable[[pd.DataFrame], pd.DataFrame]


def real_tariff_pricer(frame: pd.DataFrame) -> pd.DataFrame:
    """The official-tariff engine used by the pipeline and by T1/T3 (needs VARO_REAL_DATA_ROOT)."""
    from services.real_transport_enrichment import enrich_real_transport

    data = frame.copy()
    data["dc_id"] = data["dc_id"].fillna("") if "dc_id" in data.columns else ""
    return enrich_real_transport(data.reset_index(drop=True))


def price_universe(lanes: pd.DataFrame, pricer: Pricer | None = None) -> pd.DataFrame:
    """Move cost at each lane's own quantity, with cost provenance and basis.  Unpriceable lanes keep a NULL cost."""
    if lanes.empty:
        return lanes.copy()
    frame = lanes.reset_index(drop=True).copy()
    routable = frame["route_status"].isin(ROUTE_OK + (ROUTE_FORBIDDEN,)) & frame["recommended_qty"].notna()
    real = bool(frame["real_transport_mode"].fillna(False).astype(bool).any())
    for column in ("move_cost", "cost_provenance", "cost_basis", "cost_source", "real_transport_applied",
                   "real_transport_status", "proxy_vehicle_count", "unit_weight_status", "vehicle_mix",
                   "transport_cost_provenance"):
        frame[column] = None
    if real and pricer is not None and routable.any():
        priced = pricer(frame.loc[routable].reset_index(drop=True))
        positions = frame.index[routable].tolist()
        for column in ("real_transport_applied", "real_transport_status", "proxy_vehicle_count", "unit_weight_status",
                       "vehicle_mix", "transport_cost_provenance"):
            if column in priced.columns:
                frame.loc[positions, column] = priced[column].tolist()
        # the engine's own road distance is kept apart; distance_km stays the generator's routes-table value
        for column in ("distance_km", "travel_time_min"):
            frame[f"tariff_{column}"] = None
            frame.loc[positions, f"tariff_{column}"] = pd.to_numeric(priced[column], errors="coerce").tolist()
        applied = priced.get("real_transport_applied", pd.Series([False] * len(priced))).fillna(False).astype(bool).tolist()
        costs = pd.to_numeric(priced.get("move_cost"), errors="coerce").tolist()
        for position, ok, cost in zip(positions, applied, costs):
            if ok and cost is not None and math.isfinite(cost):
                proxy = (_finite(frame.at[position, "proxy_vehicle_count"]) or 0.0) > 0
                frame.at[position, "move_cost"] = float(cost)
                frame.at[position, "cost_provenance"] = PROXY if proxy else DERIVED_REAL
                frame.at[position, "cost_basis"] = sf.QUANTITY_SPECIFIC
                frame.at[position, "cost_source"] = "official_tariff_x_road_distance_x_derived_unit_weight"
            else:
                frame.at[position, "cost_provenance"] = UNKNOWN
                frame.at[position, "cost_source"] = f"tariff_not_applied:{frame.at[position, 'real_transport_status']}"
    elif real:
        frame.loc[routable, "cost_provenance"] = UNKNOWN
        frame.loc[routable, "cost_source"] = "no_pricer"
    else:
        for position in frame.index[routable]:
            estimate = _finite(frame.at[position, "route_estimated_cost"])
            frame.at[position, "move_cost"] = float(frame.at[position, "generator_cost"])
            frame.at[position, "cost_provenance"] = USER_INPUT if estimate else CONFIG
            frame.at[position, "cost_basis"] = "UNDECLARED"
            frame.at[position, "cost_source"] = "uploaded_route_estimated_cost" if estimate else "distance_km_x_100_default"
    frame.loc[~routable, "cost_provenance"] = UNKNOWN
    frame["move_cost"] = pd.to_numeric(frame["move_cost"], errors="coerce")
    # every cost alias follows move_cost: the generator's real-mode 0.0 must not reach a selector as a free move
    frame["estimated_cost"] = frame["move_cost"]
    frame["cost_per_unit"] = frame["move_cost"] / pd.to_numeric(frame["recommended_qty"], errors="coerce")
    return frame


def attach_benchmark_caps(lanes: pd.DataFrame, inventory_flow: pd.DataFrame) -> pd.DataFrame:
    """Offline benchmark caps (suhyup_algorithm_revalidation.enrich_inventory_constraints, PROXY), unchanged."""
    if lanes.empty:
        return lanes.copy()
    frame = rv.enrich_inventory_constraints(lanes.reset_index(drop=True), inventory_flow)
    for column in ("source_stock", "target_stock", "target_daily_demand_proxy", "median_stock", "source_surplus",
                   "target_need_7d"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("source_surplus", "target_need_7d", "source_stock"):
        frame[f"{column}_unit"] = [unit if status == "MATCHED" else None
                                   for unit, status in zip(frame["quantity_unit"], frame["unit_status"])]
    return frame


def upload_inventory_flow(inventory: pd.DataFrame, date: str) -> pd.DataFrame:
    """An upload inventory sheet in the actual-flow layout the benchmark caps read (stock_qty, outbound = sales_qty)."""
    store = _first_column(inventory, ("store_id", "node_id"))
    product = _first_column(inventory, ("product_id", "item_id"))
    stock = _first_column(inventory, ("stock_qty", "current_stock", "quantity"))
    demand = _first_column(inventory, ("avg_daily_sales", "sales_qty", "demand_qty"))
    return pd.DataFrame({"date": date, "center_code": inventory[store].astype(str),
                         "product_code": inventory[product].astype(str),
                         "stock_qty": pd.to_numeric(inventory[stock], errors="coerce"),
                         "outbound_qty": pd.to_numeric(inventory[demand], errors="coerce") if demand else 0.0})


# ------------------------------------------------------------------------------------------------ validity


def validate_universe(lanes: pd.DataFrame, *, store_ids: Sequence[str], product_ids: Sequence[str]) -> pd.DataFrame:
    """Explicit validity of every lane; the first failing check names the status, every failure is listed.

    Route and DC capacity are not checks here: no quantity-unit capacity exists in the input, so they are listed as
    unchecked for every lane (old and new alike) and left to the T1 selector when a capacity column exists.
    """
    if lanes.empty:
        return lanes.copy()
    frame = lanes.copy()
    stores, products = set(map(str, store_ids)), set(map(str, product_ids))
    qty = pd.to_numeric(frame["recommended_qty"], errors="coerce")
    frame["chk_keys"] = [bool(p) and bool(s) and bool(t) and p in products and s in stores and t in stores
                         for p, s, t in zip(frame["product_id"].map(_text), frame["source_id"].map(_text),
                                            frame["target_id"].map(_text))]
    frame["chk_not_self"] = frame["source_id"].astype(str) != frame["target_id"].astype(str)
    frame["chk_route"] = frame["route_status"].isin(ROUTE_OK)
    source_stock = pd.to_numeric(frame.get("source_stock", frame.get("gen_source_stock")), errors="coerce")
    frame["chk_product_at_source"] = (pd.to_numeric(frame.get("gen_source_stock"), errors="coerce") > 0) & (
        source_stock.isna() | (source_stock > 0))
    frame["chk_unique"] = ~frame["is_duplicate"].fillna(False).astype(bool)
    matched = frame.get("inventory_target_matched", pd.Series([True] * len(frame), index=frame.index))
    frame["chk_target_holds_product"] = frame["target_has_inventory_row"].fillna(False).astype(bool) & matched.fillna(False).astype(bool)
    frame["chk_need_evidenced"] = pd.to_numeric(frame.get("gen_target_need"), errors="coerce") > 0
    frame["chk_unit"] = frame["unit_status"].isin(("UNDECLARED_SAME_SOURCE", "MATCHED"))
    frame["chk_qty"] = qty.notna() & (qty >= 1) & ((qty - qty.round()).abs() <= EPS)
    frame["chk_cost"] = frame["move_cost"].notna() & (frame["move_cost"] >= 0) & (frame["cost_provenance"] != UNKNOWN)
    frame["chk_source_cap"] = pd.to_numeric(frame.get("source_surplus"), errors="coerce") > 0
    frame["chk_target_cap"] = pd.to_numeric(frame.get("target_need_7d"), errors="coerce") > 0
    statuses, flags = [], []
    for row in frame.to_dict("records"):
        failed = []
        for check, code in VALIDITY_CHECKS:
            if not bool(row.get(check)):
                failed.append(code or _text(row.get("route_status") if check == "chk_route" else row.get("unit_status"))
                              or ("UNIT_UNKNOWN" if check == "chk_unit" else check))
        statuses.append(failed[0] if failed else VALID)
        flags.append("|".join(failed) or None)
    frame["validity_status"], frame["validity_flags"] = statuses, flags
    frame["unchecked_constraints"] = "ROUTE_CAPACITY=MISSING|DC_CAPACITY=MISSING|ROUTE_OPERATION=NOT_OBSERVED"
    frame["qty_exceeds_individual_cap"] = qty > pd.concat(
        [pd.to_numeric(frame.get("source_surplus"), errors="coerce"),
         pd.to_numeric(frame.get("target_need_7d"), errors="coerce")], axis=1).min(axis=1) + EPS
    frame["cap_complete"] = frame["chk_source_cap"] & frame["chk_target_cap"]
    frame["provenance"] = [json.dumps({
        "quantity_rule": "generator: int(max(1, min(max(1, surplus | stock x 0.3), max(1, need | surplus), stock, move_cap)))",
        "need_rule": "generator: max(0, median - target_stock) + 7 x target demand (median when the target has no row)",
        "caps": "benchmark PROXY: source_surplus = max(0, stock - median); target_need_7d = max(0, median + 7 x outbound - stock)",
        "cost": f"{row.get('cost_provenance')} / {row.get('cost_basis')} / {row.get('cost_source')}",
        "route": f"{row.get('route_status')} ({row.get('distance_provenance')}); operation {row.get('route_operational_evidence')}",
        "unit": row.get("unit_status")}, ensure_ascii=False) for row in frame.to_dict("records")]
    return frame


# ------------------------------------------------------------------------------------------------ policies


def _usable(frame: pd.DataFrame) -> pd.Series:
    """Lanes a generator-derived pool may hold: reachable the way the generator sees it (a routes row, explicit flags
    ignored as production ignores them), a quantity, not a duplicate row, not saving-excluded.  ALL_VALID adds the
    explicit validity filter; evaluate_pool drops data-invalid lanes before any selection."""
    return (frame["route_status"].isin(ROUTE_OK + (ROUTE_FORBIDDEN,)) & frame["recommended_qty"].notna()
            & ~frame["is_duplicate"].fillna(False).astype(bool) & ~frame["saving_excluded"].fillna(False).astype(bool))


# validity failures that make a lane unselectable on data grounds (cap and need findings stay: the caps decide those)
DATA_EXCLUSIONS = frozenset({"INVALID_KEY", "SELF_MOVE", ROUTE_UNVERIFIED, ROUTE_FORBIDDEN, "PRODUCT_NOT_AT_SOURCE",
                             "DUPLICATE", "UNIT_MISMATCH", "UNIT_UNKNOWN", "QTY_INVALID", "COST_UNKNOWN"})


def selectable_split(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(lanes a selector may see, lanes excluded on data grounds).  A lane without a finite move cost is always
    excluded: the MILP would read a missing cost as 0."""
    if pool.empty:
        return pool.copy(), pool.iloc[0:0].copy()
    cost = pd.to_numeric(pool.get("move_cost", pd.Series([None] * len(pool), index=pool.index)), errors="coerce")
    flags = pool.get("validity_flags", pd.Series([None] * len(pool), index=pool.index))
    data_invalid = flags.map(lambda text: bool(set(str(text).split("|")) & DATA_EXCLUSIONS) if isinstance(text, str) else False)
    excluded = cost.isna() | (cost < 0) | data_invalid
    return pool[~excluded].copy(), pool[excluded].copy()


def policy_limit(policy: str) -> int | None:
    limit = POLICY_SPEC[policy]["limit"]
    return cg.MAX_CANDIDATES if limit == "MAX_CANDIDATES" else limit


def policy_pool(universe: pd.DataFrame, policy: str, *, limit: int | None = None) -> pd.DataFrame:
    """The lanes of one policy, in production order for the generator family and lane-key order otherwise."""
    if policy not in POLICY_SPEC:
        raise ValueError(f"unknown candidate policy {policy!r}")
    if universe.empty:
        return universe.copy()
    frame = universe
    if policy in GENERATOR_FAMILY:
        pool = frame[frame["generator_rank"].notna()].copy()
        pool = pool.sort_values("generator_rank", key=lambda s: s.astype(int), kind="mergesort")
        cut = limit if limit is not None else policy_limit(policy)
        pool = pool.head(cut) if cut is not None else pool
    elif policy == MULTI_TARGET:
        pool = frame[_usable(frame) & frame["target_has_inventory_row"].fillna(False).astype(bool)
                     & (pd.to_numeric(frame["gen_target_need"], errors="coerce") > 0)].copy()
    elif policy == ALL_VALID:
        if "validity_status" not in frame.columns:
            raise ValueError("ALL_VALID needs validate_universe() first")
        pool = frame[_usable(frame) & (frame["validity_status"] == VALID)].copy()
    else:  # ALL_LANES
        pool = frame[_usable(frame)].copy()
    if policy not in GENERATOR_FAMILY:  # production-ordered lanes first, then lane key (input-order independent)
        pool["_order"] = pool["generator_rank"].map(lambda value: 10 ** 9 if value is None or pd.isna(value) else int(value))
        pool = pool.sort_values(["_order", "lane_key"], kind="mergesort").drop(columns="_order")
    pool["policy"] = policy
    pool["policy_rank"] = range(1, len(pool) + 1)
    return pool.reset_index(drop=True)


def pool_membership(universe: pd.DataFrame, pools: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per lane and policy: in the pool, or the reason it is not (cut, rule, validity)."""
    frame = universe[["lane_key", "inventory_row", "route_id"]].copy()
    usable = _usable(universe)
    for policy, pool in pools.items():
        members = set(pool["route_id"].astype(str)) if not pool.empty else set()
        reasons = []
        for row, ok in zip(universe.to_dict("records"), usable.tolist()):
            if row["route_id"] in members:
                reasons.append("IN_POOL")
            elif row.get("is_duplicate"):
                reasons.append("DUPLICATE_ROW")
            elif row.get("route_status") not in ROUTE_OK:
                reasons.append(str(row.get("route_status")))
            elif row.get("saving_excluded"):
                reasons.append("SAVING_NOT_POSITIVE")
            elif policy in GENERATOR_FAMILY:
                reasons.append("CUT_BY_LIMIT" if row.get("generator_rank") is not None and not pd.isna(row.get("generator_rank"))
                               else "CUT_BY_ONE_TARGET_RULE")
            elif policy == ALL_VALID:
                reasons.append(f"INVALID:{row.get('validity_status')}")
            elif policy == MULTI_TARGET:
                reasons.append("NEED_NOT_EVIDENCED" if row.get("target_has_inventory_row") else "TARGET_PRODUCT_NOT_HELD")
            else:
                reasons.append("NOT_IN_POLICY_RULE" if ok else "NOT_USABLE")
        frame[f"pool_{policy}"] = reasons
    return frame


# ------------------------------------------------------------------------------------------------ evaluation


def _measure(action: Callable[[], Any]) -> tuple[Any, float, float]:
    """(result, elapsed ms, tracemalloc peak KB of this call)."""
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    tick = time.perf_counter()
    result = action()
    elapsed = (time.perf_counter() - tick) * 1000.0
    peak = tracemalloc.get_traced_memory()[1] - base
    if started_here:
        tracemalloc.stop()
    return result, round(elapsed, 3), round(max(0, peak) / 1024, 1)


def rank_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """The Varo Final order exactly as the offline revalidation computes it (heuristic scores, then auto VHS)."""
    from services.legacy_adapters._local_modules.heuristic_optimizer import add_heuristic_scores
    from services.vhs_score_engine import apply_auto_vhs

    if pool.empty:
        return pool.copy()
    day = pool.copy()
    day["final_recommendation"] = "재고 이동"
    return apply_auto_vhs(add_heuristic_scores(day)).frame


def _plan_totals(rows: Sequence[Mapping[str, Any]], qty_field: str = "recommended_qty",
                 cost_field: str = "move_cost") -> dict[str, Any]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return {"service": 0.0, "cost": 0.0, "route_ids": [], "lane_keys": []}
    return {"service": float(pd.to_numeric(frame[qty_field], errors="coerce").fillna(0).sum()),
            "cost": float(pd.to_numeric(frame[cost_field], errors="coerce").fillna(0).sum()),
            "route_ids": frame["route_id"].astype(str).tolist(),
            "lane_keys": [lane_key(r.get("product_id"), r.get("source_id"), r.get("target_id"), r.get("route_type") or "DIRECT",
                                   r.get("dc_id") if (r.get("route_type") == "VIA_DC") else "")
                          for r in frame.where(pd.notna(frame), None).to_dict("records")]}


def comparison_signature(records: Sequence[Mapping[str, Any]], caps: sf.CapTable, *, dataset: str, date: str | None,
                         cost_basis: str, quantity_basis: str, max_routes: int | None = MAX_ROUTES) -> dict[str, Any]:
    """T3 comparison signature (milp_benchmark_integrity_validation._signature) for any dataset."""
    from services.milp_benchmark_integrity_validation import CONSTRAINTS, effective_caps

    return mbi.comparison_signature(dataset=dataset, date=date, candidates=records, constraints=CONSTRAINTS,
                                    caps=effective_caps(caps), cost_basis=cost_basis, quantity_basis=quantity_basis,
                                    selection_limit=max_routes, objective=mbi.LEXICOGRAPHIC_OBJECTIVE)


MODEL_COMPONENTS = ("dataset", "date", "constraints", "cost_basis", "quantity_basis", "selection_limit", "objective")


def model_signature(signature: Mapping[str, Any], cap_definition: str) -> str:
    """Everything of a comparison signature except the pool and its cap values, plus the cap formulas: two pools with
    the same model signature run the same model on different candidates (pool sensitivity, not a decision gap)."""
    digests = signature.get("component_digests") or {}
    payload = {name: digests.get(name) for name in MODEL_COMPONENTS}
    payload["cap_definition"] = cap_definition
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def evaluate_pool(pool: pd.DataFrame, *, date: str | None, dataset: str, cap_spec: Mapping[str, tuple],
                  milp_cap_spec: Mapping[str, tuple], max_routes: int = MAX_ROUTES, cost_recompute: Any = None,
                  milp_options: Mapping[str, Any] | None = None, pool_total: int | None = None,
                  enumeration_limit: int = mbi.ENUMERATION_COMBINATION_LIMIT, rank: bool = True) -> dict[str, Any]:
    """Varo Final, Greedy, T1 (three modes) and the lexicographic MILP on one pool, checked and signed.

    The MILP keeps its objective and its fixed 0/1 quantities; MAX_ROUTES is the offline benchmark's 5.  Plans inside
    one pool are compared with the T3 decision gap; a different pool is never a decision gap (see compare_pools).
    """
    timing: dict[str, Any] = {}
    pool, excluded = selectable_split(pool)
    if rank:
        ranked, timing["rank_ms"], timing["rank_peak_kb"] = _measure(lambda: rank_pool(pool))
    else:
        ranked, timing["rank_ms"], timing["rank_peak_kb"] = pool.copy(), 0.0, 0.0
        if not ranked.empty and "varo_final_rank" not in ranked.columns:
            from services.vhs_score_engine import _rank_varo_operational

            ranked["varo_final_rank"] = _rank_varo_operational(ranked)
    records = rv._records(ranked.reset_index(drop=True)) if not ranked.empty else []
    caps = sf.caps_from_columns(records, cap_spec)
    milp_caps = sf.caps_from_columns(records, milp_cap_spec)
    plans: dict[str, dict[str, Any]] = {}
    if records:
        varo, timing["varo_ms"], _ = _measure(lambda: rv.ordered_feasible_selection(ranked, ("varo_final_rank", "route_id"), (True, True)))
        greedy = rv.ordered_feasible_selection(ranked, ("greedy_rank", "route_id"), (True, True)) if "greedy_rank" in ranked.columns else pd.DataFrame()
    else:
        varo, greedy, timing["varo_ms"] = pd.DataFrame(), pd.DataFrame(), 0.0

    def t1(mode: str, partial: str) -> dict[str, Any]:
        return sf.select_shared_feasible(records, caps, mode=mode, max_routes=max_routes, partial_policy=partial,
                                         cost_recompute=cost_recompute)

    (aon, partial_plan, strict), timing["t1_ms"], timing["t1_peak_kb"] = _measure(lambda: (
        t1(sf.BENCHMARK_PROXY, sf.PARTIAL_NONE), t1(sf.BENCHMARK_PROXY, sf.PARTIAL_IF_SAFE), t1(sf.STRICT_ACTUAL, sf.PARTIAL_IF_SAFE)))
    milp, timing["milp_ms"], timing["milp_peak_kb"] = _measure(lambda: rv.lexicographic_milp(ranked, options=milp_options))
    evidence = mbi.lexicographic_solver_evidence(milp, [r["recommended_qty"] for r in records])
    milp_rows = rv._records(milp["selected"]) if not milp["selected"].empty else []
    enumeration, timing["enumeration_ms"], _ = _measure(
        lambda: mbi.enumerate_lexicographic_optimum(records, max_routes=max_routes, combination_limit=enumeration_limit))
    milp_plan = _plan_totals(milp_rows)
    enum_proven = (enumeration.get("status") == "ENUMERATED"
                   and abs(float(enumeration["optimal_service"]) - milp_plan["service"]) <= 1e-9
                   and abs(float(enumeration["optimal_cost"]) - milp_plan["cost"]) <= 1e-6)
    missing_caps = any(cap.value is None for (kind, _), cap in milp_caps.items())
    scope = mbi.benchmark_scope(solver_status=evidence["solver_status"], pool_size=len(records),
                                pool_total=pool_total if pool_total is not None else len(records),
                                cap_provenance=[cap.provenance for cap in milp_caps.values() if cap.value is not None],
                                missing_caps=missing_caps)
    milp_sig = comparison_signature(records, milp_caps, dataset=dataset, date=date, cost_basis="QUANTITY_SPECIFIC_TARIFF_AT_RECOMMENDED_QTY",
                                    quantity_basis=mbi.FIXED_BINARY, max_routes=max_routes)

    def t1_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
        by_id = {sf.candidate_identity(record, index): record for index, record in enumerate(records)}
        return [{**by_id[row["candidate_id"]], "allocated_qty": row["allocated_qty"],
                 "allocated_move_cost": row["allocated_move_cost"]}
                for row in plan["rows"] if row["selection_status"] in (sf.SELECTED, sf.PARTIALLY_SELECTED)]

    raw_plans = {
        "VARO_FINAL": (rv._records(varo) if not varo.empty else [], "recommended_qty", "move_cost", milp_caps, mbi.FIXED_BINARY, None),
        "GREEDY": (rv._records(greedy) if not greedy.empty else [], "recommended_qty", "move_cost", milp_caps, mbi.FIXED_BINARY, None),
        "T1_ALL_OR_NOTHING": (t1_rows(aon), "allocated_qty", "allocated_move_cost", caps, mbi.FIXED_BINARY, aon),
        "T1_PARTIAL": (t1_rows(partial_plan), "allocated_qty", "allocated_move_cost", caps, mbi.PARTIAL_ALLOWED, partial_plan),
        "T1_STRICT_ACTUAL": (t1_rows(strict), "allocated_qty", "allocated_move_cost",
                             {slot: cap for slot, cap in caps.items() if cap.strict_accepted}, mbi.PARTIAL_ALLOWED, strict),
        "MILP": (milp_rows, "recommended_qty", "move_cost", milp_caps, mbi.FIXED_BINARY, None),
    }
    for name, (rows, qty_field, cost_field, plan_caps, basis, t1_plan) in raw_plans.items():
        totals = _plan_totals(rows, qty_field, cost_field)
        allocated = qty_field if qty_field != "recommended_qty" else None
        check = mbi.independent_check(rows, caps, max_routes=max_routes, quantity_basis=basis, allocated_field=allocated)
        cost_basis = ("QUANTITY_SPECIFIC_TARIFF_RECOMPUTED_FOR_PARTIAL" if basis == mbi.PARTIAL_ALLOWED
                      else "QUANTITY_SPECIFIC_TARIFF_AT_RECOMMENDED_QTY")
        signature = comparison_signature(records, plan_caps, dataset=dataset, date=date, cost_basis=cost_basis,
                                         quantity_basis=basis, max_routes=max_routes)
        gap = mbi.decision_gap(totals, milp_plan, plan_signature=signature, reference_signature=milp_sig,
                               reference_status=evidence["solver_status"], reference_enumeration_proven=enum_proven,
                               plan_feasible=check["passed"])
        status = (t1_plan or {}).get("plan_status")
        if status == sf.INSUFFICIENT_CAP_DATA:
            gap.update(comparison_status=mbi.INSUFFICIENT_DATA, decision_gap_status=mbi.GAP_NOT_COMPUTABLE,
                       reason_codes=mbi._unique([*gap["reason_codes"], "PROXY_CAPS", "MISSING_CAP"]))
        plans[name] = {**totals, "selected_count": len(rows), "plan_status": status or ("SELECTED" if rows else "NO_SELECTION"),
                       "partial_count": (t1_plan or {}).get("partial_count"),
                       "unchecked_constraints": (t1_plan or {}).get("unchecked_constraints"),
                       "feasibility_claim": (t1_plan or {}).get("feasibility_claim"),
                       "check": {key: check[key] for key in ("violation_count", "passed", "source_excess_qty", "target_excess_qty",
                                                             "duplicate_count", "selection_limit_exceeded",
                                                             "strict_unverifiable_constraints")},
                       "signature": signature["signature"], "comparison_status": gap["comparison_status"],
                       "decision_gap_status": gap["decision_gap_status"],
                       "decision_service_gap": gap["decision_service_gap"], "decision_cost_gap": gap["decision_cost_gap"],
                       "zero_gap_flags": gap["zero_gap_flags"], "reason_codes": gap["reason_codes"]}
    return {
        "pool_size": len(records), "pool_total": pool_total, "ranked": ranked, "records": records, "plans": plans,
        "excluded_count": int(len(excluded)),
        "excluded": [{"route_id": row.get("route_id"), "lane_key": row.get("lane_key"),
                      "reason": row.get("validity_flags") or "COST_UNKNOWN"}
                     for row in excluded.where(pd.notna(excluded), None).to_dict("records")],
        "milp_evidence": {"solver_status": evidence["solver_status"], "termination_reason": evidence["termination_reason"],
                          "optimality_proven": evidence["optimality_proven"],
                          "stage1_mip_gap": evidence["stage1"]["solver_mip_gap"], "stage2_mip_gap": evidence["stage2"]["solver_mip_gap"],
                          "time_limit_s": evidence["time_limit_s"],
                          "service_optimum_exact_by_integrality": evidence["service_optimum_exact_by_integrality"]},
        "enumeration": {"status": enumeration.get("status"), "combinations": enumeration.get("combinations"),
                        "alternative_optima": enumeration.get("alternative_optima"), "proven": enum_proven},
        "scope": scope, "milp_signature": milp_sig, "missing_caps": missing_caps,
        "cap_complete_share": (sum(1 for r in records if _finite(r.get("source_surplus")) is not None
                                   and _finite(r.get("target_need_7d")) is not None) / len(records)) if records else None,
        "timing": timing,
    }


def compare_pools(candidate: Mapping[str, Any], baseline: Mapping[str, Any], plan: str, *, cap_definition: str,
                  superset: bool | None = None) -> dict[str, Any]:
    """Same plan on two pools.  Cost is compared only at equal service (service first) and only when both pools run the
    same model with every cap present; otherwise the delta is NULL with a reason.  Never a decision gap."""
    left, right = candidate["plans"][plan], baseline["plans"][plan]
    same_model = (model_signature(candidate["milp_signature"], cap_definition)
                  == model_signature(baseline["milp_signature"], cap_definition))
    reasons = []
    if not same_model:
        reasons.append("DIFFERENT_MODEL")
    if candidate.get("missing_caps") or baseline.get("missing_caps"):
        reasons.append("MISSING_CAP")
    if not left["check"]["passed"] or not right["check"]["passed"]:
        reasons.append("INDEPENDENT_CHECK_FAILED")
    if candidate.get("excluded_count") or baseline.get("excluded_count"):
        reasons.append("DATA_INVALID_LANES_EXCLUDED")
    service_equal = abs(left["service"] - right["service"]) <= EPS
    if not service_equal:
        reasons.append("SERVICE_NOT_EQUAL")
    comparable = same_model and not candidate.get("missing_caps") and not baseline.get("missing_caps") \
        and left["check"]["passed"] and right["check"]["passed"] \
        and not candidate.get("excluded_count") and not baseline.get("excluded_count")
    status = ("SAME_MODEL_DIFFERENT_POOL" if comparable else mbi.NOT_COMPARABLE)
    delta = round(left["cost"] - right["cost"], 6) if comparable and service_equal else None
    pct = round(100.0 * delta / right["cost"], 6) if delta is not None and right["cost"] > EPS else None
    monotone = None
    if superset and plan == "MILP" and comparable:
        monotone = left["service"] > right["service"] + EPS or (service_equal and left["cost"] <= right["cost"] + 1e-6)
    return {"pool_comparison_status": status, "service_equal": service_equal,
            "service_delta": round(left["service"] - right["service"], 6), "cost_delta": delta, "cost_delta_pct": pct,
            "superset_of_baseline": superset, "superset_monotone": monotone, "reason_codes": reasons}


# ------------------------------------------------------------------------------------------------ diagnostics


def coverage_metrics(pool: pd.DataFrame, universe: pd.DataFrame, *, reference_lane_keys: Sequence[str] = ()) -> dict[str, Any]:
    """Coverage of a pool against the valid lanes of the universe.  Every ratio is reported with its numerator and
    denominator; a ratio with an empty denominator is NULL."""
    def valid_of(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "validity_status" not in frame.columns:
            return frame.iloc[0:0]
        return frame[frame["validity_status"] == VALID]

    valid, pool_valid = valid_of(universe), valid_of(pool)

    def ratio(num: int, den: int) -> dict[str, Any]:
        return {"numerator": int(num), "denominator": int(den), "value": round(num / den, 6) if den else None}

    def groups(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple]:
        return set(map(tuple, frame[list(columns)].astype(str).itertuples(index=False))) if not frame.empty else set()

    cheapest = 0
    sp_valid = groups(valid, ("source_id", "product_id"))
    pool_keys = set(pool["lane_key"]) if not pool.empty else set()
    if not valid.empty:
        best = valid.sort_values(["cost_per_unit", "lane_key"], kind="mergesort").groupby(["source_id", "product_id"]).head(1)
        cheapest = int(best["lane_key"].isin(pool_keys).sum())
    alternatives = pool.groupby(["source_id", "product_id"])["target_id"].nunique() if not pool.empty else pd.Series(dtype=float)
    duplicates = int(pool.duplicated("lane_key").sum()) if not pool.empty else 0
    strict = 0  # benchmark caps are PROXY: no candidate has strict-accepted source and target caps
    reference = set(reference_lane_keys)
    return {
        "source_product_coverage": ratio(len(groups(pool_valid, ("source_id", "product_id"))), len(sp_valid)),
        "target_product_coverage": ratio(len(groups(pool_valid, ("target_id", "product_id"))),
                                         len(groups(valid, ("target_id", "product_id")))),
        "valid_lane_coverage": ratio(len(pool_valid), len(valid)),
        "products_without_candidate": int(len(set(valid["product_id"].astype(str)) - set(pool["product_id"].astype(str))))
        if not valid.empty else 0,
        "sources_without_candidate": int(len(set(valid["source_id"].astype(str)) - set(pool["source_id"].astype(str))))
        if not valid.empty else 0,
        "targets_without_candidate": int(len(set(valid["target_id"].astype(str)) - set(pool["target_id"].astype(str))))
        if not valid.empty else 0,
        "alternative_targets_per_source_product_mean": round(float(alternatives.mean()), 6) if len(alternatives) else None,
        "alternative_targets_per_source_product_max": int(alternatives.max()) if len(alternatives) else None,
        "cheapest_lane_inclusion": ratio(cheapest, len(sp_valid)),
        "best_known_plan_inclusion": ratio(len(reference & pool_keys), len(reference)),
        "valid_share_benchmark": ratio(len(pool_valid), len(pool)),
        "valid_share_strict_actual": ratio(strict, len(pool)),
        "duplicate_share": ratio(duplicates, len(pool)),
    }


def diversity_metrics(pool: pd.DataFrame) -> dict[str, Any]:
    """Concentration and spread of a pool (reported only; diversity is not an objective)."""
    if pool.empty:
        return {"pool_size": 0}

    def hhi(series: pd.Series) -> float:
        share = series.value_counts(normalize=True)
        return round(float((share ** 2).sum()), 6)

    def stats(values: pd.Series) -> dict[str, Any]:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            return {"min": None, "median": None, "max": None}
        return {"min": round(float(values.min()), 3), "median": round(float(values.median()), 3),
                "max": round(float(values.max()), 3)}

    pairs = pool.groupby(["source_id", "target_id"]).size()
    return {
        "pool_size": int(len(pool)), "sources": int(pool["source_id"].nunique()), "targets": int(pool["target_id"].nunique()),
        "products": int(pool["product_id"].nunique()),
        "by_source": json.dumps(pool["source_id"].astype(str).value_counts().sort_index().to_dict()),
        "by_target": json.dumps(pool["target_id"].astype(str).value_counts().sort_index().to_dict()),
        "source_hhi": hhi(pool["source_id"].astype(str)), "target_hhi": hhi(pool["target_id"].astype(str)),
        "product_max_candidates": int(pool["product_id"].value_counts().max()),
        "source_target_pairs": int(len(pairs)), "source_target_max_repeat": int(pairs.max()),
        "distance_km": stats(pool.get("distance_km")), "move_cost": stats(pool.get("move_cost")),
        "cost_per_unit": stats(pool.get("cost_per_unit")), "candidate_score": stats(pool.get("candidate_score")),
    }


def shortage_diagnosis(pairs: pd.DataFrame, universe: pd.DataFrame, pool: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Why each (source, product) inventory row has no lane in a pool: input infeasibility apart from cuts."""
    if pairs.empty:
        return pd.DataFrame(columns=["inventory_row", "source_id", "product_id", "policy", "category"])
    pool_rows = set(pool["inventory_row"]) if not pool.empty else set()
    lanes_by_row = {row: group for row, group in universe.groupby("inventory_row")} if not universe.empty else {}
    out = []
    for pair in pairs.to_dict("records"):
        row = pair["inventory_row"]
        lanes = lanes_by_row.get(row)
        category = pair["eligibility"] if pair["eligibility"] != "ELIGIBLE" else None
        if category is None and row in pool_rows:
            category = "IN_POOL"
        if category is None:
            routed = lanes[lanes["route_status"].isin(ROUTE_OK)] if lanes is not None else pd.DataFrame()
            usable = routed[~routed["is_duplicate"].astype(bool)] if not routed.empty else routed
            valid = usable[usable.get("validity_status", pd.Series(dtype=str)) == VALID] if not usable.empty else usable
            if routed.empty:
                category = "NO_ROUTE"
            elif usable.empty:
                category = "DUPLICATE_ROW"
            elif not (usable["target_has_inventory_row"].astype(bool) & (pd.to_numeric(usable["gen_target_need"]) > 0)).any():
                category = "NO_TARGET_NEED"
            elif usable["saving_excluded"].astype(bool).all():
                category = "SAVING_NOT_POSITIVE"
            elif "validity_status" in usable.columns and valid.empty:
                statuses = set(usable["validity_status"])
                category = ("UNIT_UNKNOWN" if statuses & {"UNIT_UNKNOWN", "UNIT_MISMATCH"} and not statuses - {"UNIT_UNKNOWN", "UNIT_MISMATCH", "TARGET_PRODUCT_NOT_HELD", "NEED_NOT_EVIDENCED"}
                            else "COST_UNKNOWN" if "COST_UNKNOWN" in statuses and not statuses - {"COST_UNKNOWN", "TARGET_PRODUCT_NOT_HELD", "NEED_NOT_EVIDENCED"}
                            else "QTY_INVALID" if "QTY_INVALID" in statuses and not statuses - {"QTY_INVALID", "TARGET_PRODUCT_NOT_HELD", "NEED_NOT_EVIDENCED"}
                            else "CAP_MISSING_OR_ZERO")
            elif policy in GENERATOR_FAMILY:
                choice = usable[usable["generator_choice"].astype(bool)]
                category = "CUT_BY_LIMIT" if not choice.empty and choice["generator_rank"].notna().any() else "CUT_BY_ONE_TARGET_RULE"
            else:
                category = "NOT_IN_POLICY_RULE"
        out.append({"inventory_row": row, "source_id": pair["source_id"], "product_id": pair["product_id"],
                    "policy": policy, "category": category})
    return pd.DataFrame(out)


CANDIDATE_RECORD_COLUMNS = (
    "snapshot_date", "candidate_id", "route_id", "source_id", "target_id", "product_id", "route_type", "dc_id",
    "recommended_qty", "source_stock", "source_surplus", "target_stock", "target_need_7d", "gen_source_stock",
    "gen_source_surplus_raw", "gen_target_need", "need_fabricated", "distance_km", "move_cost", "cost_per_unit",
    "cost_provenance", "cost_basis", "route_status", "route_operational_evidence", "unit_status", "candidate_score",
    "generator_rank", "generator_target_rank", "generator_target_count", "validity_status", "validity_flags",
    "unchecked_constraints", "lineage", "provenance",
)


def candidate_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Pre-cut candidate record (section 7 fields); a value that is absent or not established stays NULL."""
    data = frame.copy()
    for column in CANDIDATE_RECORD_COLUMNS:
        if column not in data.columns:
            data[column] = None
    return data[list(CANDIDATE_RECORD_COLUMNS)].rename(columns={
        "target_need_7d": "target_need", "move_cost": "transport_cost", "generator_rank": "candidate_rank"})


def frame_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Order-independent digest of selected columns (determinism checks)."""
    data = frame[[c for c in columns if c in frame.columns]].astype(str)
    rows = sorted(map(tuple, data.itertuples(index=False)))
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()[:16]
