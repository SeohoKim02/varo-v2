"""Rule-based V2 candidate generation when an upload has no recommendation sheet.

Operator-oriented (no MILP, no DQN): picks surplus/near-expiry source stores and
needy target stores, builds DIRECT or VIA_DC transfers from the routes data, keeps
only positive-saving candidates, and scores them 0-100 with explainable
components. Results are marked "V2 생성 후보" and never use any DQN value.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from services.column_aliases import clean_numeric_value

MAX_CANDIDATES = 20
_DEFAULT_UNIT_PRICE = 1000.0
_DISPOSAL_FRACTION = 0.5
_MOVE_CAP = 50
_SHORT_EXPIRY_CAP = 20

SCORE_COMPONENTS = (
    "유통기한 긴급도", "초과 재고", "도착지 수요/부족", "예상 절감액", "경로 가능성", "거리 패널티",
)


def _num(value: Any, default: float = 0.0) -> float:
    result = clean_numeric_value(value)
    return default if result is None else result


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _store_ids_by_type(stores: pd.DataFrame) -> tuple[list[str], str | None]:
    type_col = next((c for c in ("node_type", "store_type", "type") if c in stores.columns), None)
    id_col = next((c for c in ("node_id", "store_id", "id") if c in stores.columns), None)
    if id_col is None:
        return [], None
    if type_col is None:
        return [str(v) for v in stores[id_col].dropna()], None
    upper = stores[type_col].astype(str).str.strip().str.upper()
    store_ids = [str(v) for v in stores.loc[upper == "STORE", id_col].dropna()]
    dc_ids = [str(v) for v in stores.loc[upper == "DC", id_col].dropna()]
    return store_ids, (dc_ids[0] if dc_ids else None)


def _name_lookup(stores: pd.DataFrame) -> dict[str, str]:
    id_col = next((c for c in ("node_id", "store_id", "id") if c in stores.columns), None)
    name_col = next((c for c in ("node_name", "store_name", "name") if c in stores.columns), None)
    if id_col is None or name_col is None:
        return {}
    return {str(r[id_col]): str(r[name_col]) for _, r in stores.iterrows() if pd.notna(r.get(id_col))}


def _product_info(products: pd.DataFrame) -> dict[str, dict[str, Any]]:
    id_col = next((c for c in ("product_id", "item_id", "id") if c in products.columns), None)
    name_col = next((c for c in ("product_name", "item_name", "name") if c in products.columns), None)
    if id_col is None:
        return {}
    info: dict[str, dict[str, Any]] = {}
    for _, row in products.iterrows():
        pid = row.get(id_col)
        if pd.isna(pid):
            continue
        info[str(pid)] = {
            "name": str(row.get(name_col)) if name_col and pd.notna(row.get(name_col)) else str(pid),
            "unit_price": _num(row.get("unit_price"), _DEFAULT_UNIT_PRICE) or _DEFAULT_UNIT_PRICE,
        }
    return info


def _route_lookup(routes: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    if routes is None or routes.empty:
        return {}
    src = next((c for c in ("source_id", "from_id", "from_store_id") if c in routes.columns), None)
    tgt = next((c for c in ("target_id", "to_id", "to_store_id") if c in routes.columns), None)
    if src is None or tgt is None:
        return {}
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in routes.iterrows():
        s, t = row.get(src), row.get(tgt)
        if pd.isna(s) or pd.isna(t):
            continue
        lookup.setdefault((str(s), str(t)), {
            "distance_km": _num(row.get("distance_km")),
            "estimated_cost": _num(row.get("estimated_cost")),
            "travel_time_min": _num(row.get("travel_time_min")),
        })
    return lookup


def _expiry_score(days: float | None) -> float:
    if days is None:
        return 30.0
    if days <= 3:
        return 100.0
    if days <= 7:
        return 80.0
    if days <= 14:
        return 55.0
    if days <= 30:
        return 35.0
    return 20.0


def can_generate(data: dict[str, Any]) -> bool:
    return all(
        isinstance(data.get(name), pd.DataFrame) and not data[name].empty
        for name in ("stores", "products", "inventory", "routes")
    )


def _resolve_route(
    source: str, target: str, dc_id: str | None,
    direct: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any] | None:
    direct_leg = direct.get((source, target))
    via = None
    if dc_id and (source, dc_id) in direct and (dc_id, target) in direct:
        a, b = direct[(source, dc_id)], direct[(dc_id, target)]
        via = {
            "distance_km": a["distance_km"] + b["distance_km"],
            "estimated_cost": (a["estimated_cost"] or 0) + (b["estimated_cost"] or 0),
            "travel_time_min": a["travel_time_min"] + b["travel_time_min"],
        }
    if direct_leg and via:
        if (direct_leg["estimated_cost"] or 1e12) <= (via["estimated_cost"] or 1e12):
            return {"route_type": "DIRECT", "route": direct_leg, "direct": direct_leg, "via": via,
                    "basis": "직접 경로 비용이 DC 경유보다 낮아 DIRECT를 선택했습니다."}
        return {"route_type": "VIA_DC", "route": via, "direct": direct_leg, "via": via,
                "basis": "DC 경유 비용이 직접 경로보다 낮아 VIA_DC를 선택했습니다."}
    if direct_leg:
        return {"route_type": "DIRECT", "route": direct_leg, "direct": direct_leg, "via": None,
                "basis": "직접 경로만 가능해 DIRECT 후보로 생성했습니다."}
    if via:
        return {"route_type": "VIA_DC", "route": via, "direct": None, "via": via,
                "basis": "직접 경로가 없어 DC 경유 후보로 생성했습니다."}
    return None


def generate_candidates(data: dict[str, Any]) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    base_stats = {
        "generated": False, "count": 0, "direct_count": 0, "via_dc_count": 0,
        "route_deferred": 0, "negative_saving_excluded": 0, "duplicate_removed": 0,
        "qty_excluded": 0, "candidates": [], "score_components": list(SCORE_COMPONENTS),
        "method": "유통기한·초과재고·도착지 수요·경로 가능성·예상 절감액 기반 rule-based",
    }
    if not can_generate(data):
        return None, {**base_stats, "reason": "stores/products/inventory/routes 데이터가 부족합니다."}
    try:
        stores = data["stores"]
        inventory = data["inventory"].copy()
        store_ids, dc_id = _store_ids_by_type(stores)
        names = _name_lookup(stores)
        product_info = _product_info(data["products"])
        direct = _route_lookup(data["routes"])
        if not store_ids or not direct:
            return None, {**base_stats, "reason": "점포·경로 정보가 없어 후보를 생성할 수 없습니다."}

        store_col = next((c for c in ("store_id", "node_id") if c in inventory.columns), None)
        product_col = next((c for c in ("product_id", "item_id") if c in inventory.columns), None)
        stock_col = next((c for c in ("stock_qty", "current_stock", "quantity") if c in inventory.columns), None)
        if not all((store_col, product_col, stock_col)):
            return None, {**base_stats, "reason": "재고 시트에 store_id/product_id/stock_qty가 없습니다."}
        expiry_col = next((c for c in ("days_to_expiry", "expiry_days") if c in inventory.columns), None)
        demand_col = next((c for c in ("avg_daily_sales", "sales_qty", "demand_qty") if c in inventory.columns), None)

        inv = inventory[inventory[store_col].astype(str).isin(store_ids)].copy()
        inv["_store"] = inv[store_col].astype(str)
        inv["_product"] = inv[product_col].astype(str)
        inv["_stock"] = inv[stock_col].map(_num)
        inv["_expiry"] = inv[expiry_col].map(clean_numeric_value) if expiry_col else None
        inv["_demand"] = inv[demand_col].map(_num) if demand_col else 0.0
        stock_by = {(r["_store"], r["_product"]): r["_stock"] for _, r in inv.iterrows()}
        demand_by = {(r["_store"], r["_product"]): r["_demand"] for _, r in inv.iterrows()}
        median_stock = inv.groupby("_product")["_stock"].median().to_dict()
        surplus_range = {
            p: max(1.0, float(g["_stock"].max() - g["_stock"].min())) for p, g in inv.groupby("_product")
        }

        seen: set[tuple[str, str, str]] = set()
        raw: list[dict[str, Any]] = []
        stats = dict(base_stats)
        for _, item in inv.iterrows():
            source = item["_store"]
            product = item["_product"]
            if product not in product_info:
                continue
            stock = item["_stock"]
            if stock <= 0:
                continue
            expiry = item["_expiry"] if expiry_col else None
            expiry = None if (expiry is None or pd.isna(expiry)) else float(expiry)
            median = median_stock.get(product, stock)
            surplus = stock - median
            is_source = surplus > 0 or (expiry is not None and expiry <= 7)
            if not is_source:
                continue
            # candidate targets: needy (below median) reachable stores
            targets = []
            for target in store_ids:
                if target == source:
                    continue
                resolved = _resolve_route(source, target, dc_id, direct)
                if resolved is None:
                    continue
                target_stock = stock_by.get((target, product), median)
                need = max(0.0, median - target_stock) + demand_by.get((target, product), 0.0) * 7
                targets.append((need, -resolved["route"]["estimated_cost"], target, resolved))
            if not targets:
                stats["route_deferred"] += 1
                continue
            targets.sort(reverse=True)
            need, _negcost, target, resolved = targets[0]
            key = (product, source, target)
            if key in seen:
                stats["duplicate_removed"] += 1
                continue
            seen.add(key)

            target_stock = stock_by.get((target, product), median)
            source_surplus = max(1.0, surplus if surplus > 0 else stock * 0.3)
            target_need = max(1.0, need if need > 0 else source_surplus)
            cap = _SHORT_EXPIRY_CAP if (expiry is not None and expiry <= 3) else _MOVE_CAP
            moved = int(max(1, min(source_surplus, target_need, stock, cap)))
            if moved <= 0:
                stats["qty_excluded"] += 1
                continue

            unit_price = product_info[product]["unit_price"]
            route = resolved["route"]
            cost = route["estimated_cost"] or (route["distance_km"] * 100.0)
            saving = moved * unit_price * _DISPOSAL_FRACTION - cost
            if saving <= 0:
                stats["negative_saving_excluded"] += 1
                continue

            expiry_score = _expiry_score(expiry)
            surplus_score = _clamp((surplus / surplus_range.get(product, 1.0)) * 100.0)
            need_score = _clamp((need / max(1.0, median)) * 100.0)
            route_score = 100.0 if resolved["route_type"] == "DIRECT" else 70.0
            distance = route["distance_km"] or 0.0
            distance_score = _clamp(100.0 - distance * 5.0)
            raw.append({
                "source": source, "target": target, "product": product, "moved": moved,
                "cost": round(cost, 1), "saving": saving, "resolved": resolved,
                "expiry": expiry, "expiry_score": expiry_score, "surplus_score": surplus_score,
                "need_score": need_score, "route_score": route_score, "distance_score": distance_score,
                "target_stock": target_stock, "unit_price": unit_price,
            })

        if not raw:
            return None, {**stats, "reason": "양의 예상 절감액을 가진 후보가 없어 추천을 생성하지 못했습니다."}

        savings = [r["saving"] for r in raw]
        s_min, s_max = min(savings), max(savings)
        s_range = max(1.0, s_max - s_min)
        for r in raw:
            saving_score = _clamp((r["saving"] - s_min) / s_range * 100.0)
            r["candidate_score"] = round(_clamp(
                0.25 * r["expiry_score"] + 0.20 * r["surplus_score"] + 0.20 * r["need_score"]
                + 0.20 * saving_score + 0.10 * r["route_score"] + 0.05 * r["distance_score"]
            ), 1)
            r["saving_score"] = round(saving_score, 1)
        raw.sort(key=lambda r: r["candidate_score"], reverse=True)
        raw = raw[:MAX_CANDIDATES]

        records: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        for index, r in enumerate(raw, start=1):
            resolved = r["resolved"]
            route_type = resolved["route_type"]
            direct_leg, via_leg = resolved.get("direct"), resolved.get("via")
            route_id = f"V2C{index:03d}"
            if route_type == "DIRECT":
                stats["direct_count"] += 1
            else:
                stats["via_dc_count"] += 1
            expiry_txt = f"{int(r['expiry'])}일" if r["expiry"] is not None else "정보 없음"
            score_reason = (
                f"유통기한 {expiry_txt}·초과재고·도착지 수요/부족·예상절감 {r['saving']:,.0f}원 반영 ({route_type})"
            )
            records.append({
                "route_id": route_id, "product_id": r["product"],
                "product_name": product_info[r["product"]]["name"],
                "source_id": r["source"], "source_name": names.get(r["source"], r["source"]),
                "target_id": r["target"], "target_name": names.get(r["target"], r["target"]),
                "dc_id": dc_id if route_type == "VIA_DC" else None,
                "dc_name": names.get(dc_id, dc_id) if (route_type == "VIA_DC" and dc_id) else None,
                "route_type": route_type, "transport_type": "일반 탑차",
                "recommended_qty": r["moved"], "estimated_cost": r["cost"],
                "expected_saving": round(r["saving"], 1),
                "distance_km": r["resolved"]["route"]["distance_km"],
                "travel_time_min": r["resolved"]["route"]["travel_time_min"],
                "direct_cost": (direct_leg or {}).get("estimated_cost"),
                "via_dc_cost": (via_leg or {}).get("estimated_cost"),
                "direct_distance_km": (direct_leg or {}).get("distance_km"),
                "via_dc_distance_km": (via_leg or {}).get("distance_km"),
                "vhs_score": 50.0, "recommendation_grade": "보통", "confidence_score": 50.0,
                "reason": "추천 결과 시트가 없어 재고·경로 데이터로 생성한 V2 기본 후보입니다.",
            })
            details.append({
                "route_id": route_id, "product_name": product_info[r["product"]]["name"],
                "source_name": names.get(r["source"], r["source"]),
                "target_name": names.get(r["target"], r["target"]),
                "route_type": route_type, "transfer_qty": r["moved"],
                "expected_saving": round(r["saving"], 1), "candidate_score": r["candidate_score"],
                "score_reason": score_reason,
                "direct_available": direct_leg is not None,
                "via_dc_available": via_leg is not None,
                "selected_route_basis": resolved["basis"],
                "days_to_expiry_source": r["expiry"],
                "recommendation_source": "V2 생성 후보",
                "dqn_status": "미연결",
            })

        stats.update({
            "generated": True, "count": len(records),
            "candidates": details,
            "reason": "추천 결과 시트가 없어 재고·경로 기반 V2 생성 후보를 만들었습니다.",
        })
        return pd.DataFrame(records), stats
    except Exception as exc:  # never crash the upload
        return None, {**base_stats, "reason": f"후보 생성 중 처리할 수 없는 입력입니다: {type(exc).__name__}"}
