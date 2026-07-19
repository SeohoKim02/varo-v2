"""Immutable inventory transitions for the home logistics simulation.

The service never writes to the uploaded frames.  It uses only quantities that
exist in the workbook or recommendation and records the chosen basis in the
result metadata.  Recommendation order and algorithm scores are not changed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


INVENTORY_TRANSITION_VERSION = "inventory-transition-2026-07-20.1"
INVENTORY_STATES = ("초과재고", "적정재고", "부족재고", "데이터 부족")

_CACHE_LIMIT = 24
_SCENARIO_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_COUNTERS = {
    "inventory_transition_calculations": 0,
    "inventory_transition_cache_hits": 0,
}

_STOCK_FIELDS = ("stock_qty", "current_stock", "inventory_qty", "quantity", "stock")
_EXPLICIT_DEMAND_FIELDS = ("expected_demand_qty", "forecast_demand_qty", "demand_qty")
_SHORTAGE_FIELDS = ("shortage_qty", "demand_shortage", "target_shortage_qty", "unmet_demand")
_MOVABLE_FIELDS = (
    "available_transfer_stock", "transferable_stock", "movable_stock",
    "source_surplus", "surplus_qty", "excess_stock",
)
_RISK_FIELDS = ("disposal_risk_score", "waste_risk_score", "disposal_risk")


def _blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "<na>", "nat"}


def _number(value: object) -> float | None:
    if _blank(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_number(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[float | None, str | None]:
    for field in fields:
        value = _number(record.get(field))
        if value is not None:
            return value, field
    return None, None


def _first_text(record: Mapping[str, Any], fields: Sequence[str], default: str = "") -> str:
    for field in fields:
        value = record.get(field)
        if not _blank(value):
            return str(value).strip()
    return default


def _records(value: object) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return [dict(row) for row in value.to_dict("records")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _product_map(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in _records(data.get("products")):
        product_id = _first_text(record, ("product_id", "item_id", "id"))
        if product_id:
            result.setdefault(product_id, record)
    return result


def _store_map(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in _records(data.get("stores")):
        store_id = _first_text(record, ("node_id", "store_id", "dc_id", "id"))
        if store_id:
            result.setdefault(store_id, record)
    return result


def _state_demand(
    inventory: Mapping[str, Any], product: Mapping[str, Any],
) -> tuple[float | None, str]:
    """Return the stock-state demand without a hidden threshold.

    Explicit demand wins.  Otherwise sellable demand during the remaining
    shelf-life is used and the actual minimum display stock is added when the
    product sheet provides it.  If those fields are unavailable, the existing
    seven-day demand convention used by sensitivity/optimality validation is
    reused.
    """
    explicit, explicit_field = _first_number(inventory, _EXPLICIT_DEMAND_FIELDS)
    if explicit is not None:
        return max(0.0, explicit), str(explicit_field)

    daily, daily_field = _first_number(inventory, ("avg_daily_sales", "sales_qty"))
    expiry, expiry_field = _first_number(inventory, ("days_to_expiry", "expiry_days"))
    display, display_field = _first_number(product, ("min_display_stock", "minimum_display_stock"))
    if daily is not None and expiry is not None:
        demand = max(0.0, daily) * max(0.0, expiry)
        basis = f"{daily_field}×{expiry_field}"
        if display is not None:
            demand += max(0.0, display)
            basis += f"+{display_field}"
        return demand, basis

    seven_day, seven_field = _first_number(inventory, ("demand_forecast_7d", "sales_7d"))
    if seven_day is not None:
        return max(0.0, seven_day), str(seven_field)
    if daily is not None:
        return max(0.0, daily) * 7.0, f"{daily_field}×7"
    monthly, monthly_field = _first_number(inventory, ("sales_30d", "sales_30"))
    if monthly is not None:
        return max(0.0, monthly) / 30.0 * 7.0, f"{monthly_field}/30×7"
    return None, "수요 수량을 계산할 실제 열이 없음"


def _target_shortage_limit(
    inventory: Mapping[str, Any], recommendation: Mapping[str, Any],
) -> tuple[float | None, str]:
    explicit, field = _first_number(inventory, _SHORTAGE_FIELDS)
    if explicit is None:
        explicit, field = _first_number(recommendation, _SHORTAGE_FIELDS)
    if explicit is not None:
        return max(0.0, explicit), str(field)

    stock, stock_field = _first_number(inventory, _STOCK_FIELDS)
    demand, demand_field = _first_number(inventory, ("demand_forecast_7d", "sales_7d"))
    if demand is None:
        daily, daily_field = _first_number(inventory, ("avg_daily_sales", "sales_qty"))
        if daily is not None:
            demand, demand_field = daily * 7.0, f"{daily_field}×7"
    if demand is None:
        monthly, monthly_field = _first_number(inventory, ("sales_30d", "sales_30"))
        if monthly is not None:
            demand, demand_field = monthly / 30.0 * 7.0, f"{monthly_field}/30×7"
    if stock is None or demand is None:
        return None, "재고와 7일 수요를 함께 확인할 수 없음"
    return max(0.0, demand - stock), f"max({demand_field}-{stock_field}, 0)"


def _movable_stock(
    inventory: Mapping[str, Any], recommendation: Mapping[str, Any], state_demand: float | None,
) -> tuple[float | None, str]:
    explicit, field = _first_number(recommendation, _MOVABLE_FIELDS)
    if explicit is None:
        explicit, field = _first_number(inventory, _MOVABLE_FIELDS)
    if explicit is not None:
        return max(0.0, explicit), str(field)

    dead_stock, dead_field = _first_number(inventory, ("dead_stock_qty", "obsolete_stock_qty"))
    if dead_stock is not None:
        stock, _ = _first_number(inventory, _STOCK_FIELDS)
        return max(0.0, min(dead_stock, stock if stock is not None else dead_stock)), str(dead_field)

    stock, stock_field = _first_number(inventory, _STOCK_FIELDS)
    if stock is not None and state_demand is not None:
        return max(0.0, stock - state_demand), f"max({stock_field}-재고상태 수요, 0)"
    return None, "이동 가능 재고를 계산할 실제 열이 없음"


def classify_inventory_state(stock: object, demand: object) -> str:
    """Classify stock by an exact demand boundary; no tolerance is invented."""
    stock_value, demand_value = _number(stock), _number(demand)
    if stock_value is None or demand_value is None:
        return "데이터 부족"
    if stock_value > demand_value + 1e-9:
        return "초과재고"
    if stock_value < demand_value - 1e-9:
        return "부족재고"
    return "적정재고"


def _quantity(value: float, integer_rule: bool) -> float | int:
    clean = max(0.0, value)
    if integer_rule:
        return int(math.floor(clean + 1e-9))
    return round(clean, 3)


def _is_integer_quantity(*values: float | None) -> bool:
    present = [value for value in values if value is not None]
    return bool(present) and all(abs(value - round(value)) <= 1e-9 for value in present)


def _status_metrics(stock: float | None, demand: float | None) -> tuple[float | None, float | None, str]:
    if stock is None or demand is None:
        return None, None, "데이터 부족"
    return max(0.0, demand - stock), max(0.0, stock - demand), classify_inventory_state(stock, demand)


def _improved(before: str, after: str, before_gap: float | None, after_gap: float | None) -> bool | None:
    if "데이터 부족" in {before, after}:
        return None
    if after == "적정재고" and before != "적정재고":
        return True
    if before == after and before_gap is not None and after_gap is not None:
        return after_gap < before_gap - 1e-9
    return False


def build_inventory_baseline(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build copied store-product state records and documented field bases."""
    source = data or {}
    products = _product_map(source)
    stores = _store_map(source)
    records: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in _records(source.get("inventory")):
        row = copy.deepcopy(raw)
        store_id = _first_text(row, ("store_id", "node_id"))
        product_id = _first_text(row, ("product_id", "item_id"))
        if not store_id or not product_id:
            continue
        product = products.get(product_id, {})
        store = stores.get(store_id, {})
        stock, stock_field = _first_number(row, _STOCK_FIELDS)
        demand, demand_basis = _state_demand(row, product)
        shortage, excess, status = _status_metrics(stock, demand)
        risk, risk_field = _first_number(row, _RISK_FIELDS)
        normalized = {
            "store_id": store_id,
            "store_name": _first_text(row, ("store_name", "node_name"), _first_text(store, ("node_name", "store_name"), store_id)),
            "product_id": product_id,
            "product_name": _first_text(row, ("product_name", "item_name"), _first_text(product, ("product_name", "item_name"), product_id)),
            "stock": stock,
            "stock_basis": stock_field or "재고 수량 열 없음",
            "demand": demand,
            "demand_basis": demand_basis,
            "shortage": shortage,
            "excess": excess,
            "status": status,
            "disposal_risk": risk,
            "disposal_risk_basis": risk_field or "폐기 위험 수치 열 없음",
            "raw": row,
        }
        records.append(normalized)
        by_key.setdefault((store_id, product_id), normalized)
    return {
        "version": INVENTORY_TRANSITION_VERSION,
        "records": records,
        "by_key": by_key,
        "metadata": {
            "state_rule": "현재 재고와 실제 수요 기준값을 정확히 비교(초과/동일/미달)",
            "demand_priority": [
                "명시 수요", "일평균 판매량×잔여 유통일+실제 최소 진열재고", "7일 수요", "30일 판매량의 7일 환산",
            ],
            "missing_rule": "필수 실제 열이 없으면 데이터 부족으로 분류하며 값을 생성하지 않음",
        },
    }


def _baseline_record(
    baseline: Mapping[str, Any], recommendation: Mapping[str, Any], endpoint: str,
) -> dict[str, Any] | None:
    product_id = _first_text(recommendation, ("product_id", "item_id"))
    store_fields = ("source_id", "from_store_id", "from_id") if endpoint == "source" else (
        "target_id", "to_store_id", "to_id",
    )
    store_id = _first_text(recommendation, store_fields)
    record = (baseline.get("by_key") or {}).get((store_id, product_id))
    if record:
        return copy.deepcopy(record)

    prefix = "source" if endpoint == "source" else "target"
    stock, stock_field = _first_number(recommendation, (f"{prefix}_stock_qty", f"{prefix}_current_stock"))
    if stock is None:
        return None
    name_fields = (f"{prefix}_name", f"{prefix}_store_name")
    demand, demand_field = _first_number(recommendation, (f"{prefix}_demand_qty",))
    shortage, excess, status = _status_metrics(stock, demand)
    return {
        "store_id": store_id,
        "store_name": _first_text(recommendation, name_fields, store_id),
        "product_id": product_id,
        "product_name": _first_text(recommendation, ("product_name", "item_name"), product_id),
        "stock": stock,
        "stock_basis": stock_field or f"{prefix}_stock_qty",
        "demand": demand,
        "demand_basis": demand_field or "수요 수량 열 없음",
        "shortage": shortage,
        "excess": excess,
        "status": status,
        "disposal_risk": None,
        "disposal_risk_basis": "폐기 위험 수치 열 없음",
        "raw": {
            "stock_qty": stock,
            **({"demand_qty": demand} if demand is not None else {}),
        },
    }


def _node_transition(record: Mapping[str, Any], before_stock: float, delta: float, role: str) -> dict[str, Any]:
    demand = _number(record.get("demand"))
    after_stock = before_stock + delta
    before_shortage, before_excess, before_status = _status_metrics(before_stock, demand)
    after_shortage, after_excess, after_status = _status_metrics(after_stock, demand)
    before_gap = (before_shortage or 0.0) + (before_excess or 0.0) if demand is not None else None
    after_gap = (after_shortage or 0.0) + (after_excess or 0.0) if demand is not None else None
    risk = _number(record.get("disposal_risk"))
    return {
        "store_id": record.get("store_id"),
        "store_name": record.get("store_name"),
        "product_id": record.get("product_id"),
        "product_name": record.get("product_name"),
        "before_stock": round(before_stock, 3),
        "outbound_qty": round(-delta, 3) if role == "source" else 0,
        "inbound_qty": round(delta, 3) if role == "target" else 0,
        "after_stock": round(after_stock, 3),
        "demand_qty": round(demand, 3) if demand is not None else None,
        "demand_basis": record.get("demand_basis"),
        "before_shortage": round(before_shortage, 3) if before_shortage is not None else None,
        "after_shortage": round(after_shortage, 3) if after_shortage is not None else None,
        "before_excess": round(before_excess, 3) if before_excess is not None else None,
        "after_excess": round(after_excess, 3) if after_excess is not None else None,
        "before_disposal_risk": risk,
        "after_disposal_risk": risk if abs(delta) <= 1e-9 else None,
        "risk_after_basis": "재고 이동 후 위험 점수를 재계산할 공식이 없어 값을 생성하지 않음" if abs(delta) > 1e-9 else record.get("disposal_risk_basis"),
        "before_status": before_status,
        "after_status": after_status,
        "change_qty": round(delta, 3),
        "status_improved": _improved(before_status, after_status, before_gap, after_gap),
        "stock_basis": record.get("stock_basis"),
    }


def calculate_inventory_transition(
    data: Mapping[str, Any] | None,
    recommendation: Mapping[str, Any],
    *,
    inventory_overrides: Mapping[tuple[str, str], float] | None = None,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate one constrained DIRECT/VIA_DC transition on copied values."""
    base = copy.deepcopy(dict(baseline)) if baseline is not None else build_inventory_baseline(data)
    rec = copy.deepcopy(dict(recommendation))
    source = _baseline_record(base, rec, "source")
    target = _baseline_record(base, rec, "target")
    route_type = _first_text(rec, ("route_type",), "DIRECT").upper().replace(" ", "_")
    route_type = "VIA_DC" if route_type in {"DC_TRANSFER", "DC_경유", "경유"} else route_type
    route_type = "DIRECT" if route_type in {"DIRECT_TRANSFER", "직접", "직접_이동"} else route_type
    requested, requested_field = _first_number(rec, ("recommended_qty", "suggested_qty", "transfer_qty", "quantity"))
    requested = max(0.0, requested or 0.0)
    overrides = dict(inventory_overrides or {})

    source_key = (str((source or {}).get("store_id") or ""), str((source or {}).get("product_id") or ""))
    target_key = (str((target or {}).get("store_id") or ""), str((target or {}).get("product_id") or ""))
    source_stock = overrides.get(source_key, _number((source or {}).get("stock")))
    target_stock = overrides.get(target_key, _number((target or {}).get("stock")))

    movable, movable_basis = (None, "출발 재고 데이터 없음")
    target_limit, target_limit_basis = (None, "도착 재고 데이터 없음")
    if source is not None:
        raw = copy.deepcopy(dict(source.get("raw") or {}))
        movable, movable_basis = _movable_stock(raw, rec, _number(source.get("demand")))
        if source_stock is not None and movable is not None:
            original_source_stock = _number(source.get("stock"))
            already_moved = max(0.0, float(original_source_stock or source_stock) - float(source_stock))
            movable = max(0.0, movable - already_moved)
            movable = min(max(0.0, source_stock), movable)
    if target is not None:
        raw = copy.deepcopy(dict(target.get("raw") or {}))
        target_limit, target_limit_basis = _target_shortage_limit(raw, rec)
        if target_stock is not None and target_limit is not None:
            original_target_stock = _number(target.get("stock"))
            already_received = max(0.0, float(target_stock) - float(original_target_stock or target_stock))
            target_limit = max(0.0, target_limit - already_received)

    limits = [requested]
    if source_stock is not None:
        limits.append(max(0.0, source_stock))
    if movable is not None:
        limits.append(max(0.0, movable))
    if target_limit is not None:
        limits.append(max(0.0, target_limit))
    complete_inputs = source is not None and target is not None and source_stock is not None and target_stock is not None
    complete_inputs = complete_inputs and movable is not None and target_limit is not None and requested > 0
    integer_rule = _is_integer_quantity(requested, source_stock, target_stock)
    applied = _quantity(min(limits), integer_rule) if complete_inputs else 0
    applied_value = float(applied)

    reasons: list[str] = []
    if source is None or source_stock is None:
        reasons.append("출발 점포의 실제 재고를 확인할 수 없음")
    if target is None or target_stock is None:
        reasons.append("도착 점포의 실제 재고를 확인할 수 없음")
    if movable is None:
        reasons.append("이동 가능 재고를 확인할 수 없음")
    elif movable <= 0:
        reasons.append("현재 상태에서 이동 가능한 출발 재고가 없음")
    if target_limit is None:
        reasons.append("도착점 수요·부족 한도를 확인할 수 없음")
    elif target_limit <= 0:
        reasons.append("현재 상태에서 도착점 부족량이 없음")
    if requested <= 0:
        reasons.append("추천 수량이 0 이하이거나 없음")
    if complete_inputs and applied_value <= 0 and not reasons:
        reasons.append("정수 수량 규칙 적용 후 이동 가능 수량이 없음")

    source_transition = (
        _node_transition(source, float(source_stock), -applied_value, "source")
        if source is not None and source_stock is not None else None
    )
    target_transition = (
        _node_transition(target, float(target_stock), applied_value, "target")
        if target is not None and target_stock is not None else None
    )
    before_total = (float(source_stock) + float(target_stock)) if source_stock is not None and target_stock is not None else None
    after_total = (
        float(source_transition["after_stock"]) + float(target_transition["after_stock"])
        if source_transition and target_transition else None
    )
    inventory_preserved = before_total is not None and after_total is not None and abs(before_total - after_total) <= 1e-9
    direct_balance = abs(applied_value - applied_value) <= 1e-9
    invariants = {
        "source_nonnegative": bool(source_transition and float(source_transition["after_stock"]) >= -1e-9),
        "outbound_within_stock": source_stock is not None and applied_value <= float(source_stock) + 1e-9,
        "outbound_within_movable": movable is not None and applied_value <= movable + 1e-9,
        "inbound_within_shortage": target_limit is not None and applied_value <= target_limit + 1e-9,
        "direct_outbound_equals_inbound": direct_balance if route_type == "DIRECT" else True,
        "via_dc_outbound_equals_inbound": direct_balance if route_type == "VIA_DC" else True,
        "inventory_preserved": inventory_preserved,
        "integer_rule_preserved": (not integer_rule) or abs(applied_value - round(applied_value)) <= 1e-9,
    }
    invariants["all_passed"] = all(invariants.values())

    full_quantity = requested > 0 and abs(applied_value - requested) <= 1e-9
    saving, _ = _first_number(rec, ("expected_saving", "saving_amount", "estimated_saving"))
    executable = applied_value > 0 and invariants["all_passed"]
    status = "실행 가능" if executable and full_quantity else "제약 수량으로 실행" if executable else "현재 상태에서 실행 불가"
    return {
        "version": INVENTORY_TRANSITION_VERSION,
        "recommendation_id": _first_text(rec, ("recommendation_id",), _first_text(rec, ("route_id",))),
        "route_id": _first_text(rec, ("route_id", "recommendation_id")),
        "route_type": route_type,
        "dc_id": _first_text(rec, ("dc_id",)) or None,
        "dc_name": _first_text(rec, ("dc_name",)) or None,
        "product_id": _first_text(rec, ("product_id", "item_id")),
        "product_name": _first_text(rec, ("product_name", "item_name")),
        "requested_quantity": _quantity(requested, integer_rule),
        "applied_quantity": applied,
        "expected_saving": saving if full_quantity and executable else None,
        "original_expected_saving": saving,
        "feasible": executable,
        "status": status,
        "skipped_reason": " / ".join(dict.fromkeys(reasons)) if not executable else None,
        "source": source_transition,
        "target": target_transition,
        "inventory_totals": {
            "before": round(before_total, 3) if before_total is not None else None,
            "after": round(after_total, 3) if after_total is not None else None,
        },
        "invariants": invariants,
        "metadata": {
            "requested_quantity_basis": requested_field or "추천 수량 열 없음",
            "movable_stock": round(movable, 3) if movable is not None else None,
            "movable_stock_basis": movable_basis,
            "target_shortage_limit": round(target_limit, 3) if target_limit is not None else None,
            "target_shortage_basis": target_limit_basis,
            "quantity_rule": "입력 수량이 정수이면 안전 한도의 내림 정수" if integer_rule else "입력 소수 단위 유지(소수 셋째 자리)",
            "partial_saving_rule": "추천 수량 전부 실행 시에만 기존 예상 절감액 표시; 부분 실행 절감액은 생성하지 않음",
            "original_data_mutated": False,
        },
    }


def _final_records(
    baseline: Mapping[str, Any], overrides: Mapping[tuple[str, str], float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in baseline.get("records") or []:
        copied = copy.deepcopy(dict(record))
        key = (str(copied.get("store_id") or ""), str(copied.get("product_id") or ""))
        stock = overrides.get(key, _number(copied.get("stock")))
        copied["stock"] = stock
        shortage, excess, status = _status_metrics(stock, _number(copied.get("demand")))
        copied["shortage"], copied["excess"], copied["status"] = shortage, excess, status
        result.append(copied)
    return result


def run_inventory_scenario(
    data: Mapping[str, Any] | None,
    recommendations: Sequence[Mapping[str, Any]],
    *,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply routes in the received order against one temporary inventory copy."""
    base = build_inventory_baseline(data) if baseline is None else copy.deepcopy(dict(baseline))
    requested = [copy.deepcopy(dict(item)) for item in recommendations]
    overrides: dict[tuple[str, str], float] = {}
    transitions: list[dict[str, Any]] = []
    for recommendation in requested:
        transition = calculate_inventory_transition(
            data, recommendation, inventory_overrides=overrides, baseline=base,
        )
        transitions.append(transition)
        if not transition.get("feasible"):
            continue
        product_id = str(transition.get("product_id") or "")
        source = transition.get("source") or {}
        target = transition.get("target") or {}
        overrides[(str(source.get("store_id") or ""), product_id)] = float(source["after_stock"])
        overrides[(str(target.get("store_id") or ""), product_id)] = float(target["after_stock"])

    executed = [item for item in transitions if item.get("feasible")]
    skipped = [item for item in transitions if not item.get("feasible")]
    moved = sum(float(item.get("applied_quantity") or 0.0) for item in executed)
    excess_reduction = sum(
        max(0.0, float((item.get("source") or {}).get("before_excess") or 0.0) - float((item.get("source") or {}).get("after_excess") or 0.0))
        for item in executed
    )
    shortage_reduction = sum(
        max(0.0, float((item.get("target") or {}).get("before_shortage") or 0.0) - float((item.get("target") or {}).get("after_shortage") or 0.0))
        for item in executed
    )
    known_savings = [item.get("expected_saving") for item in executed if item.get("expected_saving") is not None]
    return {
        "version": INVENTORY_TRANSITION_VERSION,
        "requested_route_count": len(requested),
        "executed_route_count": len(executed),
        "skipped_route_count": len(skipped),
        "skipped_reasons": [
            {"route_id": item.get("route_id"), "reason": item.get("skipped_reason") or "현재 상태에서 실행 불가"}
            for item in skipped
        ],
        "transitions": transitions,
        "baseline_records": copy.deepcopy(base.get("records") or []),
        "final_records": _final_records(base, overrides),
        "kpis": {
            "moved_quantity": round(moved, 3),
            "excess_reduction": round(excess_reduction, 3),
            "shortage_reduction": round(shortage_reduction, 3),
            "expected_saving": round(sum(float(value) for value in known_savings), 3) if known_savings else None,
        },
        "metadata": {
            **copy.deepcopy(base.get("metadata") or {}),
            "application_order": "입력된 추천 순서 유지",
            "temporary_copy_only": True,
        },
    }


def inventory_transition_cache_key(
    data_signature: str, recommendations: Sequence[Mapping[str, Any]], display_mode: str = "single",
) -> str:
    rows = [
        {
            key: item.get(key)
            for key in (
                "recommendation_id", "route_id", "product_id", "source_id", "target_id",
                "route_type", "dc_id", "recommended_qty", "expected_saving",
            )
        }
        for item in recommendations
    ]
    payload = json.dumps(
        {
            "version": INVENTORY_TRANSITION_VERSION,
            "data_signature": str(data_signature or "empty"),
            "display_mode": str(display_mode),
            "routes": rows,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_inventory_scenario(
    data_signature: str,
    data: Mapping[str, Any] | None,
    recommendations: Sequence[Mapping[str, Any]],
    *,
    display_mode: str = "single",
) -> dict[str, Any]:
    """Cache a scenario by data, route IDs, transition version, and display mode."""
    key = inventory_transition_cache_key(data_signature, recommendations, display_mode)
    if key in _SCENARIO_CACHE:
        _COUNTERS["inventory_transition_cache_hits"] += 1
        cached = copy.deepcopy(_SCENARIO_CACHE[key])
        cached.setdefault("performance", {})["cache_hit"] = True
        _SCENARIO_CACHE.move_to_end(key)
        return cached
    started = time.perf_counter()
    _COUNTERS["inventory_transition_calculations"] += 1
    result = run_inventory_scenario(data, recommendations)
    result["performance"] = {
        "cache_hit": False,
        "calculation_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    _SCENARIO_CACHE[key] = copy.deepcopy(result)
    _SCENARIO_CACHE.move_to_end(key)
    while len(_SCENARIO_CACHE) > _CACHE_LIMIT:
        _SCENARIO_CACHE.popitem(last=False)
    return result


def inventory_transition_cache_info() -> dict[str, int]:
    return {**_COUNTERS, "inventory_transition_cache_entries": len(_SCENARIO_CACHE)}


def clear_inventory_transition_cache() -> None:
    _SCENARIO_CACHE.clear()
    for key in _COUNTERS:
        _COUNTERS[key] = 0
