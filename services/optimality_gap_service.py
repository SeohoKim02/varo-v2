"""On-demand constrained optimality-gap analysis for the current Varo result.

The service is deliberately independent from the recommendation pipeline.  It
uses deep-copied in-memory recommendations and workbook frames, never reloads a
workbook, touches DQN artifacts, or mutates Streamlit/session state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any

import pandas as pd


OPTIMALITY_CALCULATION_VERSION = "saving-binary-v1"
CONSTRAINT_VERSION = "shared-feasibility-v1"
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_MAX_ROUTES = 5
DEFAULT_TIME_LIMIT = 3.0
AUTO_EXACT_CANDIDATE_LIMIT = 24
AUTO_EXACT_COMBINATION_LIMIT = 750_000
_CACHE_LIMIT = 16
_RESULT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()

ProgressCallback = Callable[[str, int, int, str], None]

_FALSE_TEXT = {"false", "0", "n", "no", "불가", "불가능", "미통과", "실패", "제외"}
_REJECT_MARKERS = ("이동 불가", "실행 불가", "불가능", "미통과", "실패", "제외", "차단")


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _first_number(record: Mapping[str, Any], names: Sequence[str]) -> tuple[float | None, str | None]:
    for name in names:
        if name in record:
            value = _number(record.get(name))
            if value is not None:
                return value, name
    return None, None


def _identifier(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _rank(item: Mapping[str, Any], names: Sequence[str], fallback: int) -> tuple[float, int]:
    value, _ = _first_number(item, names)
    return (value if value is not None and value > 0 else float("inf"), fallback)


def build_optimality_settings(
    candidate_limit: int | None = DEFAULT_CANDIDATE_LIMIT,
    max_routes: int | None = DEFAULT_MAX_ROUTES,
    search_mode: str = "auto",
    time_limit: float = DEFAULT_TIME_LIMIT,
) -> dict[str, Any]:
    """Normalize UI/test settings into the stable service contract."""
    mode = str(search_mode or "auto").strip().lower().replace("_", "-")
    if mode not in {"auto", "exact-first", "limited"}:
        mode = "auto"
    normalized_limit = None if candidate_limit in (None, 0, "all") else max(1, int(candidate_limit))
    normalized_routes = None if max_routes in (None, 0, "unlimited") else max(1, int(max_routes))
    normalized_time = min(30.0, max(0.05, float(time_limit or DEFAULT_TIME_LIMIT)))
    return {
        "candidate_limit": normalized_limit,
        "max_routes": normalized_routes,
        "search_mode": mode,
        "time_limit": normalized_time,
    }


def _frame_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, pd.DataFrame) or value.empty:
        return []
    return value.where(pd.notna(value), None).to_dict("records")


def _candidate_hash(recommendations: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "recommendation_id", "route_id", "product_id", "source_id", "target_id",
        "route_type", "dc_id", "recommended_qty", "expected_saving", "vhs_rank",
        "greedy_rank", "feasible", "route_feasible", "cutline_passed", "time_window_status",
    )
    payload = [[item.get(field) for field in fields] for item in recommendations]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=False, default=str).encode("utf-8")).hexdigest()


def build_optimality_cache_key(
    data_signature: str,
    settings: Mapping[str, Any],
    recommendations: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "data_signature": str(data_signature or ""),
        "candidate_limit": settings.get("candidate_limit"),
        "max_routes": settings.get("max_routes"),
        "search_mode": settings.get("search_mode"),
        "time_limit": settings.get("time_limit"),
        "constraint_version": CONSTRAINT_VERSION,
        "calculation_version": OPTIMALITY_CALCULATION_VERSION,
        "candidate_hash": _candidate_hash(recommendations),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _explicitly_false(value: Any) -> bool:
    if value is False:
        return True
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in _FALSE_TEXT or any(marker in text for marker in _REJECT_MARKERS)


def _route_maps(data: Mapping[str, Any] | None) -> tuple[dict[str, Mapping[str, Any]], dict[tuple[str, ...], Mapping[str, Any]]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    by_key: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for item in _frame_records((data or {}).get("routes")):
        route_id = _identifier(item.get("route_id"))
        if route_id:
            by_id.setdefault(route_id, item)
        key = (
            _identifier(item.get("source_id") or item.get("from_store_id")),
            _identifier(item.get("target_id") or item.get("to_store_id")),
            _identifier(item.get("route_type")).upper(),
            _identifier(item.get("dc_id")),
        )
        if key[0] and key[1]:
            by_key.setdefault(key, item)
    return by_id, by_key


def prepare_optimality_problem(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare a ranked, immutable binary-candidate problem and exclusions."""
    copied = [copy.deepcopy(dict(item)) for item in recommendations or []]
    ordered = sorted(enumerate(copied), key=lambda pair: _rank(pair[1], ("vhs_rank", "varo_final_rank", "rank"), pair[0]))
    candidate_limit = settings.get("candidate_limit")
    if candidate_limit is not None:
        ordered = ordered[: int(candidate_limit)]
    route_by_id, route_by_key = _route_maps(data)
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for input_index, item in ordered:
        route_id = _identifier(item.get("route_id"))
        recommendation_id = _identifier(item.get("recommendation_id"))
        product_id = _identifier(item.get("product_id"))
        source_id = _identifier(item.get("source_id") or item.get("from_store_id"))
        target_id = _identifier(item.get("target_id") or item.get("to_store_id"))
        route_type = _identifier(item.get("route_type")).upper() or "DIRECT"
        dc_id = _identifier(item.get("dc_id")) if route_type == "VIA_DC" else ""
        qty, _ = _first_number(item, ("recommended_qty", "suggested_qty", "transfer_qty", "quantity"))
        saving, _ = _first_number(item, ("expected_saving", "saving", "estimated_saving"))
        reason = ""
        if not product_id or not source_id or not target_id:
            reason = "상품/출발/도착 식별자 부족"
        elif source_id == target_id:
            reason = "출발 점포와 도착 점포가 같음"
        elif qty is None or qty <= 0:
            reason = "추천 수량이 0 이하이거나 숫자가 아님"
        elif saving is None or saving <= 0:
            reason = "예상 절감액이 0 이하이거나 숫자가 아님"
        else:
            for field in ("feasible", "is_feasible", "route_feasible", "cutline_passed", "time_window_status"):
                if field in item and _explicitly_false(item.get(field)):
                    reason = f"기존 실행 가능성 필드({field}) 미통과"
                    break
        route_key = (source_id, target_id, route_type, dc_id)
        route_record = route_by_id.get(route_id) or route_by_key.get(route_key)
        if not reason and route_record and "feasible" in route_record and _explicitly_false(route_record.get("feasible")):
            reason = "경로 데이터의 feasible 미통과"
        route_capacity, capacity_field = _first_number(
            item, ("route_capacity_qty", "vehicle_capacity_qty", "max_load_qty"),
        )
        if not reason and route_capacity is not None and qty is not None and qty > route_capacity + 1e-9:
            reason = f"{capacity_field} 한도 초과"
        identity = route_id or recommendation_id or f"candidate-{input_index + 1}"
        common = {
            **item,
            "_input_index": input_index,
            "_candidate_index": len(candidates),
            "_identity": identity,
            "_route_id": route_id or identity,
            "_product_id": product_id,
            "_source_id": source_id,
            "_target_id": target_id,
            "_route_type": route_type,
            "_dc_id": dc_id,
            "_qty": float(qty or 0.0),
            "_saving": float(saving or 0.0),
            "_duplicate_key": (product_id, source_id, target_id, route_type, dc_id),
            "_source_key": (source_id, product_id),
            "_target_key": (target_id, product_id),
            "_route_capacity": route_capacity,
        }
        if reason:
            excluded.append({
                "route_id": route_id or identity,
                "recommendation_id": recommendation_id or "-",
                "product_id": product_id or "-", "source_id": source_id or "-",
                "target_id": target_id or "-", "reason": reason,
            })
        else:
            candidates.append(common)
    for index, candidate in enumerate(candidates):
        candidate["_candidate_index"] = index
    return {
        "candidates": candidates,
        "excluded_rows": excluded,
        "input_count": len(copied),
        "ranked_candidate_count": len(ordered),
    }


def _inventory_map(data: Mapping[str, Any] | None) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in _frame_records((data or {}).get("inventory")):
        key = (
            _identifier(item.get("store_id") or item.get("node_id")),
            _identifier(item.get("product_id") or item.get("item_id")),
        )
        if key[0] and key[1]:
            result.setdefault(key, item)
    return result


def _target_shortage(record: Mapping[str, Any]) -> tuple[float | None, str]:
    explicit, field = _first_number(record, ("shortage_qty", "demand_shortage", "target_shortage_qty", "unmet_demand"))
    if explicit is not None:
        return max(0.0, explicit), str(field)
    stock, stock_field = _first_number(record, ("stock_qty", "current_stock", "quantity", "stock"))
    demand, demand_field = _first_number(record, ("demand_qty", "demand_forecast_7d", "sales_7d"))
    if demand is None:
        daily, daily_field = _first_number(record, ("avg_daily_sales", "sales_qty"))
        if daily is not None:
            demand, demand_field = daily * 7.0, f"{daily_field}×7"
    if demand is None:
        monthly, monthly_field = _first_number(record, ("sales_30d", "sales_30"))
        if monthly is not None:
            demand, demand_field = monthly / 30.0 * 7.0, f"{monthly_field}/30×7"
    if stock is None or demand is None:
        return None, "재고와 7일 수요를 함께 확인할 수 없음"
    return max(0.0, demand - stock), f"max({demand_field}-{stock_field}, 0)"


def build_constraint_context(
    candidates: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build only constraints supported by real current inputs."""
    inventory = _inventory_map(data)
    source_caps: dict[tuple[str, str], float] = {}
    target_caps: dict[tuple[str, str], float] = {}
    source_basis: dict[tuple[str, str], str] = {}
    target_basis: dict[tuple[str, str], str] = {}
    for candidate in candidates:
        source_key = candidate["_source_key"]
        target_key = candidate["_target_key"]
        source_record = inventory.get(source_key, {})
        explicit, field = _first_number(
            candidate, ("available_transfer_stock", "transferable_stock", "source_surplus", "surplus_qty", "excess_stock"),
        )
        if explicit is None:
            explicit, field = _first_number(
                source_record, ("available_transfer_stock", "transferable_stock", "source_surplus", "surplus_qty", "excess_stock"),
            )
        if explicit is not None:
            source_caps[source_key] = max(0.0, explicit)
            source_basis[source_key] = str(field)
        elif source_key in inventory:
            stock, stock_field = _first_number(source_record, ("stock_qty", "current_stock", "quantity", "stock"))
            if stock is not None:
                source_caps[source_key] = max(0.0, stock)
                source_basis[source_key] = f"{stock_field} (총 보유 재고 상한)"
        shortage, basis = _target_shortage(inventory.get(target_key, {}))
        if shortage is None:
            shortage, field = _first_number(candidate, ("target_shortage_qty", "shortage_qty", "unmet_demand"))
            basis = str(field or basis)
        if shortage is not None:
            target_caps[target_key] = max(0.0, shortage)
            target_basis[target_key] = basis

    dc_caps: dict[str, float] = {}
    dc_basis: dict[str, str] = {}
    for item in _frame_records((data or {}).get("stores")):
        node_type = _identifier(item.get("node_type") or item.get("store_type")).upper()
        dc_id = _identifier(item.get("dc_id") or item.get("node_id") or item.get("store_id"))
        if node_type == "DC" and dc_id:
            cap, field = _first_number(item, ("dc_capacity_qty", "capacity", "throughput_capacity"))
            if cap is not None:
                dc_caps[dc_id] = max(0.0, cap)
                dc_basis[dc_id] = str(field)
    for candidate in candidates:
        dc_id = candidate["_dc_id"]
        if not dc_id:
            continue
        cap, field = _first_number(candidate, ("dc_capacity_qty", "dc_capacity"))
        if cap is not None:
            dc_caps[dc_id] = max(0.0, cap)
            dc_basis[dc_id] = str(field)

    max_routes = settings.get("max_routes")
    duplicate_keys = {candidate["_duplicate_key"] for candidate in candidates}
    source_groups = {candidate["_source_key"] for candidate in candidates}
    target_groups = {candidate["_target_key"] for candidate in candidates}
    dc_groups = {candidate["_dc_id"] for candidate in candidates if candidate["_dc_id"]}
    route_capacity_count = sum(candidate.get("_route_capacity") is not None for candidate in candidates)
    constraint_rows = [
        {"constraint": "기존 실행 가능성", "status": "적용", "scope": f"후보 {len(candidates)}개", "basis": "추천/경로의 실제 feasibility·cutline·time-window 필드"},
        {"constraint": "출발 점포+상품 재고", "status": "적용" if source_caps else "미적용", "scope": f"{len(source_caps)}/{len(source_groups)}개 그룹", "basis": "실제 이동 가능 재고, 없으면 총 보유 재고 상한" if source_caps else "사용 가능한 재고 컬럼 없음"},
        {"constraint": "도착 점포+상품 부족/수요", "status": "적용" if target_caps else "미적용", "scope": f"{len(target_caps)}/{len(target_groups)}개 그룹", "basis": "실제 부족량 또는 7일 수요-현재 재고" if target_caps else "수요와 재고를 함께 확인할 수 없음"},
        {"constraint": "최대 선택 경로 수", "status": "적용" if max_routes is not None else "미적용", "scope": str(max_routes) if max_routes is not None else "무제한", "basis": "사용자 실행 설정"},
        {"constraint": "동일 경로 중복", "status": "적용", "scope": f"{len(duplicate_keys)}개 고유 조합", "basis": "상품/출발/도착/경로유형/DC가 같은 후보는 최대 1개"},
        {"constraint": "출발=도착 제외", "status": "적용", "scope": "전체 후보", "basis": "후보 준비 단계에서 제외"},
        {"constraint": "경로/차량 용량", "status": "적용" if route_capacity_count else "미적용", "scope": f"후보 {route_capacity_count}개" if route_capacity_count else "0개", "basis": "후보의 실제 route/vehicle/max_load 수량" if route_capacity_count else "현재 파이프라인 후보에 실제 차량 한도 없음"},
        {"constraint": "DC 수용량", "status": "적용" if dc_caps else "미적용", "scope": f"{len(dc_caps)}/{len(dc_groups)}개 DC", "basis": "현재 데이터의 실제 DC capacity" if dc_caps else "DC 용량 컬럼 없음"},
        {"constraint": "수량·절감액 양수", "status": "적용", "scope": "전체 후보", "basis": "0 이하/비수치 후보 준비 단계에서 제외"},
    ]
    return {
        "source_caps": source_caps, "target_caps": target_caps, "dc_caps": dc_caps,
        "source_basis": source_basis, "target_basis": target_basis, "dc_basis": dc_basis,
        "max_routes": int(max_routes) if max_routes is not None else None,
        "constraint_rows": constraint_rows,
    }


def validate_selection(
    selected: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> tuple[bool, str]:
    """Single feasibility function shared by Varo, Greedy, MILP validation and BnB."""
    checks = (
        check_max_routes,
        check_duplicate_routes,
        check_route_capacity,
        check_source_stock,
        check_target_demand,
        check_dc_capacity,
    )
    for check in checks:
        feasible, reason = check(selected, candidates, context)
        if not feasible:
            return False, reason
    return True, "통과"


def check_max_routes(
    selected: Sequence[int], candidates: Sequence[Mapping[str, Any]], context: Mapping[str, Any],
) -> tuple[bool, str]:
    del candidates
    maximum = context.get("max_routes")
    return (False, "최대 선택 경로 수 초과") if maximum is not None and len(selected) > int(maximum) else (True, "통과")


def check_duplicate_routes(
    selected: Sequence[int], candidates: Sequence[Mapping[str, Any]], context: Mapping[str, Any],
) -> tuple[bool, str]:
    del context
    keys = [candidates[index]["_duplicate_key"] for index in selected]
    return (False, "동일 경로 중복") if len(keys) != len(set(keys)) else (True, "통과")


def check_route_capacity(
    selected: Sequence[int], candidates: Sequence[Mapping[str, Any]], context: Mapping[str, Any],
) -> tuple[bool, str]:
    del context
    for index in selected:
        candidate = candidates[index]
        cap = candidate.get("_route_capacity")
        if cap is not None and float(candidate["_qty"]) > float(cap) + 1e-8:
            return False, "경로/차량 용량 초과"
    return True, "통과"


def check_source_stock(
    selected: Sequence[int], candidates: Sequence[Mapping[str, Any]], context: Mapping[str, Any],
) -> tuple[bool, str]:
    source_usage: dict[tuple[str, str], float] = {}
    for index in selected:
        candidate = candidates[index]
        key = candidate["_source_key"]
        source_usage[key] = source_usage.get(key, 0.0) + float(candidate["_qty"])
    for key, used in source_usage.items():
        if key in context["source_caps"] and used > float(context["source_caps"][key]) + 1e-8:
            return False, "출발 점포+상품 재고 초과"
    return True, "통과"


def check_target_demand(
    selected: Sequence[int], candidates: Sequence[Mapping[str, Any]], context: Mapping[str, Any],
) -> tuple[bool, str]:
    target_usage: dict[tuple[str, str], float] = {}
    for index in selected:
        candidate = candidates[index]
        key = candidate["_target_key"]
        target_usage[key] = target_usage.get(key, 0.0) + float(candidate["_qty"])
    for key, used in target_usage.items():
        if key in context["target_caps"] and used > float(context["target_caps"][key]) + 1e-8:
            return False, "도착 점포+상품 부족/수요 초과"
    return True, "통과"


def check_dc_capacity(
    selected: Sequence[int], candidates: Sequence[Mapping[str, Any]], context: Mapping[str, Any],
) -> tuple[bool, str]:
    dc_usage: dict[str, float] = {}
    for index in selected:
        candidate = candidates[index]
        qty = float(candidate["_qty"])
        dc_id = candidate["_dc_id"]
        if dc_id:
            dc_usage[dc_id] = dc_usage.get(dc_id, 0.0) + qty
    for key, used in dc_usage.items():
        if key in context["dc_caps"] and used > float(context["dc_caps"][key]) + 1e-8:
            return False, "DC 수용량 초과"
    return True, "통과"


def build_ordered_feasible_combination(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    strategy: str,
) -> dict[str, Any]:
    """Select in existing VHS or Greedy rank order using the shared validator."""
    rank_fields = ("vhs_rank", "varo_final_rank", "rank") if strategy == "varo" else ("greedy_rank",)
    ordered = sorted(range(len(candidates)), key=lambda index: _rank(candidates[index], rank_fields, index))
    selected: list[int] = []
    skipped: list[dict[str, Any]] = []
    for index in ordered:
        feasible, reason = validate_selection([*selected, index], candidates, context)
        if feasible:
            selected.append(index)
        else:
            skipped.append({"route_id": candidates[index]["_route_id"], "reason": reason})
    return _combination_result(strategy, selected, candidates, skipped)


def _combination_result(
    strategy: str,
    selected: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = [candidates[index] for index in selected]
    return {
        "strategy": strategy,
        "selected_indices": list(selected),
        "selected_ids": [str(item["_identity"]) for item in rows],
        "route_ids": [str(item["_route_id"]) for item in rows],
        "count": len(rows),
        "total_saving": round(sum(float(item["_saving"]) for item in rows), 6),
        "total_qty": round(sum(float(item["_qty"]) for item in rows), 6),
        "skipped": [dict(item) for item in (skipped or [])],
    }


def _constraint_groups(candidates: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> list[tuple[list[int], list[float], float]]:
    groups: list[tuple[list[int], list[float], float]] = []
    max_routes = context.get("max_routes")
    if max_routes is not None:
        groups.append((list(range(len(candidates))), [1.0] * len(candidates), float(max_routes)))
    for map_name, key_name in (("source_caps", "_source_key"), ("target_caps", "_target_key"), ("dc_caps", "_dc_id")):
        for key, cap in context[map_name].items():
            indices = [index for index, item in enumerate(candidates) if item[key_name] == key]
            if indices:
                groups.append((indices, [float(candidates[index]["_qty"]) for index in indices], float(cap)))
    duplicate_groups: dict[tuple[str, ...], list[int]] = {}
    for index, candidate in enumerate(candidates):
        duplicate_groups.setdefault(candidate["_duplicate_key"], []).append(index)
    for indices in duplicate_groups.values():
        if len(indices) > 1:
            groups.append((indices, [1.0] * len(indices), 1.0))
    return groups


def _milp_search(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    time_limit: float,
) -> dict[str, Any]:
    """Lazy SciPy MILP. Only status=optimal is ever reported as exact."""
    started = time.perf_counter()
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except (ImportError, AttributeError) as exc:
        return {"available": False, "error": str(exc), "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
    n = len(candidates)
    if n == 0:
        return {"available": True, "optimal": True, "selected_indices": [], "method": "SciPy MILP", "nodes": 0, "prunes": 0, "elapsed_ms": 0.0}
    rows: list[list[float]] = []
    upper: list[float] = []
    for indices, coefficients, cap in _constraint_groups(candidates, context):
        row = [0.0] * n
        for index, coefficient in zip(indices, coefficients):
            row[index] = coefficient
        rows.append(row)
        upper.append(cap)
    constraints = LinearConstraint(np.asarray(rows), -np.inf, np.asarray(upper)) if rows else None
    try:
        result = milp(
            c=-np.asarray([float(item["_saving"]) for item in candidates]),
            integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
            constraints=constraints,
            options={"time_limit": float(time_limit), "presolve": True},
        )
    except Exception as exc:
        return {"available": True, "optimal": False, "error": str(exc), "method": "SciPy MILP 오류", "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
    selected = [index for index, value in enumerate(result.x if result.x is not None else []) if value > 0.5]
    valid, reason = validate_selection(selected, candidates, context)
    optimal = bool(result.status == 0 and result.success and valid)
    dual_raw = _number(getattr(result, "mip_dual_bound", None))
    upper_bound = -dual_raw if dual_raw is not None else None
    incumbent = sum(float(candidates[index]["_saving"]) for index in selected)
    if upper_bound is not None:
        upper_bound = max(incumbent, upper_bound)
    return {
        "available": True, "optimal": optimal, "selected_indices": selected if valid else [],
        "method": "SciPy MILP (optimal)" if optimal else "SciPy MILP (time/status limited)",
        "status_code": int(result.status), "message": str(result.message),
        "nodes": int(getattr(result, "mip_node_count", 0) or 0), "prunes": 0,
        "upper_bound": upper_bound, "bound_reliable": upper_bound is not None,
        "solver_gap": _number(getattr(result, "mip_gap", None)),
        "validation": reason,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def _bnb_search(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    time_limit: float,
    seeds: Sequence[Mapping[str, Any]],
    force_limited: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + max(0.01, float(time_limit))
    order = sorted(range(len(candidates)), key=lambda index: (-float(candidates[index]["_saving"]), index))
    max_routes = context.get("max_routes")
    best_selected: list[int] = []
    best_value = 0.0
    for seed in seeds:
        indices = list(seed.get("selected_indices") or [])
        valid, _ = validate_selection(indices, candidates, context)
        value = sum(float(candidates[index]["_saving"]) for index in indices)
        if valid and value > best_value + 1e-9:
            best_selected, best_value = indices, value
    slots = len(candidates) if max_routes is None else min(int(max_routes), len(candidates))
    root_upper = sum(sorted((float(item["_saving"]) for item in candidates), reverse=True)[:slots])
    nodes = 0
    prunes = 0
    timed_out = False

    def optimistic(position: int, chosen_count: int, value: float) -> float:
        remaining_slots = len(candidates) if max_routes is None else max(0, int(max_routes) - chosen_count)
        return value + sum(float(candidates[index]["_saving"]) for index in order[position:position + remaining_slots])

    def visit(position: int, selected: list[int], value: float) -> None:
        nonlocal best_selected, best_value, nodes, prunes, timed_out
        nodes += 1
        if nodes % 128 == 0 and time.perf_counter() >= deadline:
            timed_out = True
            return
        upper = optimistic(position, len(selected), value)
        if upper <= best_value + 1e-9:
            prunes += 1
            return
        if position >= len(order):
            if value > best_value + 1e-9:
                best_selected, best_value = list(selected), value
            return
        index = order[position]
        feasible, _ = validate_selection([*selected, index], candidates, context)
        if feasible:
            selected.append(index)
            visit(position + 1, selected, value + float(candidates[index]["_saving"]))
            selected.pop()
        if not timed_out:
            visit(position + 1, selected, value)

    visit(0, [], 0.0)
    exact = not timed_out and not force_limited
    return {
        "available": True, "optimal": exact, "selected_indices": best_selected,
        "method": "deterministic exact BnB" if exact else "deterministic limited BnB",
        "nodes": nodes, "prunes": prunes, "timed_out": timed_out,
        "upper_bound": best_value if exact else max(best_value, root_upper),
        "bound_reliable": True, "search_exhausted": not timed_out,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def estimate_combination_count(n: int, max_routes: int | None) -> int:
    cap = n if max_routes is None else min(n, max_routes)
    total = 1
    for count in range(1, cap + 1):
        total += math.comb(n, count)
        if total > AUTO_EXACT_COMBINATION_LIMIT:
            break
    return total


def search_best_combination(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    settings: Mapping[str, Any],
    varo: Mapping[str, Any],
    greedy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = str(settings.get("search_mode") or "auto")
    time_limit = float(settings.get("time_limit") or DEFAULT_TIME_LIMIT)
    estimate = estimate_combination_count(len(candidates), context.get("max_routes"))
    auto_exact = len(candidates) <= AUTO_EXACT_CANDIDATE_LIMIT and estimate <= AUTO_EXACT_COMBINATION_LIMIT
    attempt_exact = mode == "exact-first" or (mode == "auto" and auto_exact)
    if mode == "limited" or (mode == "auto" and not auto_exact):
        search = _bnb_search(candidates, context, time_limit, (varo, greedy), force_limited=True)
    elif attempt_exact:
        search = _milp_search(candidates, context, time_limit)
        if not search.get("available") or search.get("error"):
            search = _bnb_search(candidates, context, time_limit, (varo, greedy), force_limited=False)
    else:
        search = _bnb_search(candidates, context, time_limit, (varo, greedy), force_limited=True)
    selected = list(search.get("selected_indices") or [])
    if not search.get("optimal"):
        incumbent_value = sum(float(candidates[index]["_saving"]) for index in selected)
        seed = max((varo, greedy), key=lambda item: float(item.get("total_saving") or 0))
        if float(seed.get("total_saving") or 0) > incumbent_value + 1e-9:
            selected = list(seed.get("selected_indices") or [])
            search["incumbent_source"] = f"{seed.get('strategy')} feasible seed"
            if search.get("bound_reliable") and search.get("upper_bound") is not None:
                search["upper_bound"] = max(float(search["upper_bound"]), float(seed.get("total_saving") or 0))
    valid, reason = validate_selection(selected, candidates, context)
    if not valid:
        fallback = max((varo, greedy), key=lambda item: float(item.get("total_saving") or 0))
        selected = list(fallback.get("selected_indices") or [])
        search.update({"optimal": False, "validation": reason, "method": f"{search.get('method')} · feasible seed fallback"})
    best = _combination_result("best", selected, candidates)
    search.update({
        "combination_estimate": estimate,
        "auto_exact_eligible": auto_exact,
        "requested_mode": mode,
        "status": "정확 최적해" if search.get("optimal") else "제한 탐색",
    })
    return best, search


def calculate_gap_metrics(
    varo_saving: float,
    greedy_saving: float,
    best_saving: float,
    *,
    exact: bool,
    upper_bound: float | None = None,
) -> dict[str, Any]:
    optimum = float(best_saving)
    if optimum <= 0:
        return {
            "available": False, "label": "계산 불가", "gap_pct": None,
            "greedy_gap_pct": None, "target_pct": None,
            "reason": "양의 절감액을 가진 실행 가능 조합이 없습니다.",
        }

    def gap(value: float, denominator: float) -> tuple[float, bool]:
        raw = 100.0 * (denominator - float(value)) / denominator
        if -1e-7 <= raw < 0:
            raw = 0.0
        return round(raw, 6), raw < -1e-7

    varo_gap, inconsistent = gap(varo_saving, optimum)
    greedy_gap, greedy_inconsistent = gap(greedy_saving, optimum)
    target = min(100.0, max(0.0, 100.0 * float(varo_saving) / optimum))
    result = {
        "available": True,
        "label": "최적성 Gap" if exact else "참고 Gap",
        "gap_pct": varo_gap,
        "gap_str": f"{varo_gap:.2f}%",
        "greedy_gap_pct": greedy_gap,
        "greedy_gap_str": f"{greedy_gap:.2f}%",
        "target_pct": round(target, 6),
        "target_str": f"{target:.2f}%",
        "formula": "100 × (최선 조합 절감액 - Varo 조합 절감액) / 최선 조합 절감액",
        "inconsistency": inconsistent or greedy_inconsistent,
        "exact": exact,
    }
    if not exact and upper_bound is not None and upper_bound >= optimum and upper_bound > 0:
        lower_gap, _ = gap(varo_saving, optimum)
        upper_gap, _ = gap(varo_saving, float(upper_bound))
        result["certified_gap_range"] = [min(lower_gap, upper_gap), max(lower_gap, upper_gap)]
        result["upper_bound"] = round(float(upper_bound), 6)
    return result


def _route_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_ids = set(left.get("selected_ids") or [])
    right_ids = set(right.get("selected_ids") or [])
    common = left_ids & right_ids
    union = left_ids | right_ids
    return {
        "common_count": len(common), "left_count": len(left_ids), "right_count": len(right_ids),
        "jaccard_pct": round(100.0 * len(common) / len(union), 2) if union else 100.0,
        "varo_coverage_pct": round(100.0 * len(common) / len(left_ids), 2) if left_ids else 100.0,
        "common_ids": sorted(common),
    }


def _constraint_usage_rows(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    best: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = [candidates[index] for index in best.get("selected_indices") or []]
    rows: list[dict[str, Any]] = []
    for label, caps, key_name, basis_map in (
        ("출발 재고", context["source_caps"], "_source_key", context["source_basis"]),
        ("도착 수요", context["target_caps"], "_target_key", context["target_basis"]),
        ("DC 수용량", context["dc_caps"], "_dc_id", context["dc_basis"]),
    ):
        for key, cap in caps.items():
            used = sum(float(item["_qty"]) for item in selected if item[key_name] == key)
            rows.append({
                "constraint": label, "group": " / ".join(key) if isinstance(key, tuple) else str(key),
                "used": round(used, 6), "limit": round(float(cap), 6),
                "remaining": round(float(cap) - used, 6), "basis": basis_map.get(key, "-"),
            })
    return rows


def _route_rows(candidates: Sequence[Mapping[str, Any]], combinations_by_name: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = {
        strategy: set(combination.get("selected_indices") or [])
        for strategy, combination in combinations_by_name.items()
    }
    union = sorted(set().union(*selected.values()) if selected else set())
    for index in union:
        item = candidates[index]
        flags = {strategy: index in indices for strategy, indices in selected.items()}
        chosen_by = [label for strategy, label in (("varo", "Varo"), ("greedy", "Greedy"), ("best", "비교 조합")) if flags.get(strategy)]
        rows.append({
            "recommendation_id": item.get("recommendation_id") or "-",
            "route_id": item["_route_id"], "product_id": item["_product_id"],
            "product_name": item.get("product_name") or item["_product_id"],
            "source_id": item["_source_id"], "target_id": item["_target_id"],
            "route_type": item["_route_type"], "dc_id": item["_dc_id"] or "-",
            "recommended_qty": item["_qty"], "expected_saving": item["_saving"],
            "varo_selected": flags.get("varo", False),
            "greedy_selected": flags.get("greedy", False),
            "best_selected": flags.get("best", False),
            "selection_difference": "모두 선택" if len(chosen_by) == 3 else " · ".join(chosen_by),
        })
    return rows


def run_optimality_gap(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
    data_signature: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the complete on-demand analysis and return a defensive deep copy."""
    normalized = build_optimality_settings(**{
        "candidate_limit": settings.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT),
        "max_routes": settings.get("max_routes", DEFAULT_MAX_ROUTES),
        "search_mode": settings.get("search_mode", "auto"),
        "time_limit": settings.get("time_limit", DEFAULT_TIME_LIMIT),
    })
    recommendation_copy = [copy.deepcopy(dict(item)) for item in recommendations or []]
    data_copy = {
        key: value.copy(deep=True) if isinstance(value, pd.DataFrame) else copy.deepcopy(value)
        for key, value in (data or {}).items()
    }
    cache_key = build_optimality_cache_key(data_signature, normalized, recommendation_copy)
    if cache_key in _RESULT_CACHE:
        cached = copy.deepcopy(_RESULT_CACHE[cache_key])
        cached["metadata"]["cache_hit"] = True
        return cached

    total_stages = 8
    overall_started = time.perf_counter()
    stage_times: dict[str, float] = {}

    def stage(name: str, index: int, detail: str, action: Callable[[], Any]) -> Any:
        if progress_callback:
            progress_callback(name, index - 1, total_stages, detail)
        started = time.perf_counter()
        value = action()
        stage_times[name] = round((time.perf_counter() - started) * 1000.0, 3)
        if progress_callback:
            progress_callback(name, index, total_stages, detail)
        return value

    problem = stage("후보 준비", 1, f"입력 {len(recommendation_copy)}개", lambda: prepare_optimality_problem(recommendation_copy, data_copy, normalized))
    candidates = problem["candidates"]
    context = stage("제약조건 구성", 2, f"유효 후보 {len(candidates)}개", lambda: build_constraint_context(candidates, data_copy, normalized))
    varo = stage("Varo 조합", 3, "VHS 순위 순차 선택", lambda: build_ordered_feasible_combination(candidates, context, "varo"))
    greedy = stage("Greedy 조합", 4, "기존 Greedy 순위 순차 선택", lambda: build_ordered_feasible_combination(candidates, context, "greedy"))
    best, search = stage("최선 조합 탐색", 5, f"{normalized['time_limit']:.1f}초 제한", lambda: search_best_combination(candidates, context, normalized, varo, greedy))
    exact = bool(search.get("optimal"))
    upper_bound = search.get("upper_bound") if search.get("bound_reliable") else None
    gap = stage("Gap·경로 일치", 6, "절감액 및 경로 ID 비교", lambda: calculate_gap_metrics(
        float(varo["total_saving"]), float(greedy["total_saving"]), float(best["total_saving"]),
        exact=exact, upper_bound=_number(upper_bound),
    ))
    matches = {
        "varo_vs_best": _route_match(varo, best),
        "greedy_vs_best": _route_match(greedy, best),
        "varo_vs_greedy": _route_match(varo, greedy),
    }
    combinations_by_name = {"varo": varo, "greedy": greedy, "best": best}
    route_rows = stage("비교표 구성", 7, "조합·제약 표 생성", lambda: _route_rows(candidates, combinations_by_name))
    usage_rows = _constraint_usage_rows(candidates, context, best)
    best_value = float(best["total_saving"] or 0)
    greedy_target = min(100.0, max(0.0, 100.0 * float(greedy["total_saving"]) / best_value)) if best_value > 0 else None
    summary_rows = [
        {"strategy": "Varo", "selected_count": varo["count"], "total_qty": varo["total_qty"], "total_saving": varo["total_saving"], "gap_pct": gap.get("gap_pct"), "target_pct": gap.get("target_pct"), "violation_count": 0 if validate_selection(varo["selected_indices"], candidates, context)[0] else 1},
        {"strategy": "Greedy", "selected_count": greedy["count"], "total_qty": greedy["total_qty"], "total_saving": greedy["total_saving"], "gap_pct": gap.get("greedy_gap_pct"), "target_pct": round(greedy_target, 6) if greedy_target is not None else None, "violation_count": 0 if validate_selection(greedy["selected_indices"], candidates, context)[0] else 1},
        {"strategy": "최적 조합" if exact else "탐색된 최선 조합", "selected_count": best["count"], "total_qty": best["total_qty"], "total_saving": best["total_saving"], "gap_pct": 0.0 if best_value > 0 else None, "target_pct": 100.0 if best_value > 0 else None, "violation_count": 0 if validate_selection(best["selected_indices"], candidates, context)[0] else 1},
    ]
    stage("최종 정리", 8, "캐시·다운로드 데이터 준비", lambda: None)
    result = {
        "settings": normalized,
        "summary_rows": summary_rows,
        "route_rows": route_rows,
        "constraint_rows": copy.deepcopy(context["constraint_rows"]),
        "constraint_usage_rows": usage_rows,
        "excluded_rows": copy.deepcopy(problem["excluded_rows"]),
        "combinations": combinations_by_name,
        "gap": gap, "matches": matches, "search": search,
        "summary": {
            "input_count": problem["input_count"],
            "ranked_candidate_count": problem["ranked_candidate_count"],
            "feasible_candidate_count": len(candidates),
            "excluded_candidate_count": len(problem["excluded_rows"]),
            "applied_constraint_count": sum(item.get("status") == "적용" for item in context["constraint_rows"]),
            "unapplied_constraint_count": sum(item.get("status") == "미적용" for item in context["constraint_rows"]),
            "exclusion_reason_counts": dict(Counter(item["reason"] for item in problem["excluded_rows"])),
            "constraint_violation_count": sum(int(item["violation_count"]) for item in summary_rows),
            "status": search["status"],
        },
        "metadata": {
            "cache_key": cache_key, "cache_hit": False,
            "calculation_ms": round((time.perf_counter() - overall_started) * 1000.0, 3),
            "stage_times_ms": stage_times,
            "calculation_version": OPTIMALITY_CALCULATION_VERSION,
            "constraint_version": CONSTRAINT_VERSION,
            "data_signature": str(data_signature or ""),
            "source_policy": "현재 세션의 추천·워크북 데이터만 사용",
            "original_recommendations_mutated": False,
        },
    }
    _RESULT_CACHE[cache_key] = copy.deepcopy(result)
    _RESULT_CACHE.move_to_end(cache_key)
    while len(_RESULT_CACHE) > _CACHE_LIMIT:
        _RESULT_CACHE.popitem(last=False)
    return copy.deepcopy(result)


def optimality_summary_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result.get("summary_rows") or [])
    return frame.rename(columns={
        "strategy": "전략", "selected_count": "선택 경로 수",
        "total_qty": "총 추천 수량", "total_saving": "총 예상 절감액",
        "gap_pct": "기준 대비 Gap(%)", "target_pct": "목표 달성률(%)",
        "violation_count": "제약 위반 수",
    })


def optimality_routes_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result.get("route_rows") or [])
    return frame.rename(columns={
        "recommendation_id": "recommendation_id",
        "route_id": "route_id", "product_id": "product_id", "product_name": "상품",
        "source_id": "출발 점포", "target_id": "도착 점포", "route_type": "경로 유형",
        "dc_id": "DC", "recommended_qty": "추천 수량", "expected_saving": "예상 절감액",
        "varo_selected": "Varo 선택", "greedy_selected": "Greedy 선택",
        "best_selected": "비교 조합 선택", "selection_difference": "선택 차이",
    })


def optimality_constraints_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result.get("constraint_rows") or []:
        rows.append({"구분": "적용 기준", "제약조건": item.get("constraint"), "상태": item.get("status"), "범위/그룹": item.get("scope"), "사용량": None, "한도": None, "근거": item.get("basis")})
    for item in result.get("constraint_usage_rows") or []:
        rows.append({"구분": "최선 조합 사용량", "제약조건": item.get("constraint"), "상태": "통과", "범위/그룹": item.get("group"), "사용량": item.get("used"), "한도": item.get("limit"), "근거": item.get("basis")})
    for item in result.get("excluded_rows") or []:
        rows.append({"구분": "제외 후보", "제약조건": "후보 제외", "상태": "제외", "범위/그룹": item.get("route_id"), "사용량": None, "한도": None, "근거": item.get("reason")})
    return pd.DataFrame(rows, columns=("구분", "제약조건", "상태", "범위/그룹", "사용량", "한도", "근거"))


def clear_optimality_cache() -> None:
    """Test/maintenance hook."""
    _RESULT_CACHE.clear()
