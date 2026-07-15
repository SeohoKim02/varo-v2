"""On-demand One-at-a-Time sensitivity analysis for Varo V2.

The engine works only with the current in-memory recommendations and workbook
frames.  It never reloads Excel files, trains/loads DQN, runs Pareto search, or
mutates application state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from services.vhs_score_engine import (
    COMPONENTS,
    calculate_weighted_vhs_scores,
    normalize_vhs_values,
    rank_vhs_scores,
    vhs_grade,
)

SENSITIVITY_CALCULATION_VERSION = "oat-v1"
VHS_WEIGHT_VERSION = "auto_distribution_v1"
MAX_ANALYSIS_CANDIDATES = 50
_CACHE_LIMIT = 16
_RESULT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()

VARIABLE_LABELS = {
    "transport_cost": "운송비",
    "distance": "이동 거리",
    "demand": "수요량",
    "disposal_risk": "폐기 위험",
    "vhs_weight": "VHS 가중치",
    "quantity": "추천 수량",
    "shortage": "재고 부족량",
    "promotion": "프로모션 효과",
}
DEFAULT_VARIABLES = (
    "transport_cost", "distance", "demand", "disposal_risk", "vhs_weight",
)
COMPONENT_LABELS = {
    "savings_score": "절감 효과",
    "disposal_risk_score": "폐기 위험",
    "demand_fit_score": "수요 적합",
    "inventory_balance_score": "재고 균형",
    "route_cost_score": "경로 비용",
    "feasibility_score": "실행 가능성",
    "promotion_score": "프로모션 비교",
    "greedy_score": "Greedy 비교",
    "confidence_score": "추천 신뢰도",
    "dqn_reference_score": "DQN 참고",
}

ProgressCallback = Callable[[str, int, int, str], None]


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _first_number(record: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = _number(record.get(name))
        if value is not None:
            return value
    return None


def _series(frame: pd.DataFrame, names: Sequence[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def generate_sensitivity_steps(
    minimum_pct: float = -10.0,
    maximum_pct: float = 10.0,
    step_count: int = 5,
) -> list[float]:
    """Return evenly spaced percentages and always include the 0% baseline."""
    low = float(minimum_pct)
    high = float(maximum_pct)
    if low > high:
        low, high = high, low
    count = max(2, int(step_count))
    interval = (high - low) / (count - 1)
    values = [round(low + interval * index, 6) for index in range(count)]
    values.append(0.0)
    return sorted(set(0.0 if abs(value) < 1e-9 else value for value in values))


def _inventory_context(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
) -> list[dict[str, float | None]]:
    inventory = (data or {}).get("inventory")
    products = (data or {}).get("products")
    inventory_records = (
        inventory.where(pd.notna(inventory), None).to_dict("records")
        if isinstance(inventory, pd.DataFrame) and not inventory.empty else []
    )
    product_records = (
        products.where(pd.notna(products), None).to_dict("records")
        if isinstance(products, pd.DataFrame) and not products.empty else []
    )
    inventory_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in inventory_records:
        store_id = str(item.get("store_id") or item.get("node_id") or "")
        product_id = str(item.get("product_id") or "")
        if store_id and product_id:
            inventory_map.setdefault((store_id, product_id), item)
    product_map = {str(item.get("product_id")): item for item in product_records if item.get("product_id")}

    contexts: list[dict[str, float | None]] = []
    for recommendation in recommendations:
        product_id = str(recommendation.get("product_id") or "")
        source = inventory_map.get((str(recommendation.get("source_id") or ""), product_id), {})
        target = inventory_map.get((str(recommendation.get("target_id") or ""), product_id), {})
        source_stock = _first_number(source, ("stock_qty", "current_stock", "quantity", "stock"))
        target_stock = _first_number(target, ("stock_qty", "current_stock", "quantity", "stock"))
        demand = _first_number(target, ("demand_qty", "demand_forecast_7d", "sales_7d"))
        if demand is None:
            daily = _first_number(target, ("avg_daily_sales", "sales_qty"))
            if daily is not None:
                demand = daily * 7.0
        if demand is None:
            monthly = _first_number(target, ("sales_30d", "sales_30"))
            if monthly is not None:
                demand = monthly / 30.0 * 7.0
        shortage = (
            max(0.0, demand - target_stock)
            if demand is not None and target_stock is not None else None
        )
        max_quantity = source_stock
        if max_quantity is not None and shortage is not None and shortage > 0:
            max_quantity = min(max_quantity, shortage)
        unit_price = (
            _first_number(target, ("unit_price", "price"))
            or _first_number(source, ("unit_price", "price"))
            or _first_number(product_map.get(product_id, {}), ("unit_price", "price"))
        )
        disposal_cost = (
            _first_number(source, ("disposal_cost_per_unit", "disposal_cost"))
            or _first_number(product_map.get(product_id, {}), ("disposal_cost_per_unit", "disposal_cost"))
        )
        contexts.append({
            "source_stock": source_stock,
            "target_stock": target_stock,
            "demand": demand,
            "shortage": shortage,
            "max_quantity": max_quantity,
            "unit_price": unit_price,
            "disposal_cost_per_unit": disposal_cost,
        })
    return contexts


def available_sensitivity_variables(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    weights: Mapping[str, float] | None,
) -> list[str]:
    """Return only variables that have a real input or active VHS weight."""
    items = list(recommendations or [])
    contexts = _inventory_context(items, data)

    def any_value(names: Sequence[str]) -> bool:
        return any(_first_number(item, names) is not None for item in items)

    available: list[str] = []
    if any_value(("estimated_cost", "move_cost", "transport_cost")):
        available.append("transport_cost")
    if any_value(("distance_km", "route_distance_km", "direct_distance_km")):
        available.append("distance")
    if any(context.get("demand") is not None for context in contexts):
        available.append("demand")
    if any_value(("disposal_risk_score", "avoided_disposal_cost")):
        available.append("disposal_risk")
    if any(float(value or 0) > 0 for value in (weights or {}).values()):
        available.append("vhs_weight")
    if any_value(("recommended_qty", "suggested_qty", "quantity")):
        available.append("quantity")
    if any(context.get("shortage") is not None for context in contexts):
        available.append("shortage")
    if any_value(("promotion_effect",)):
        available.append("promotion")
    return available


def build_sensitivity_settings(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    weights: Mapping[str, float] | None,
    *,
    variables: Sequence[str] | None = None,
    minimum_pct: float = -10.0,
    maximum_pct: float = 10.0,
    step_count: int = 5,
    candidate_limit: int | None = 10,
) -> dict[str, Any]:
    available = available_sensitivity_variables(recommendations, data, weights)
    requested = list(variables) if variables is not None else list(DEFAULT_VARIABLES)
    selected = [name for name in requested if name in available]
    total = len(recommendations or [])
    requested_limit = total if candidate_limit is None else max(1, int(candidate_limit))
    applied_limit = min(total, requested_limit, MAX_ANALYSIS_CANDIDATES)
    return {
        "method": "OAT",
        "variables": selected,
        "available_variables": available,
        "minimum_pct": float(minimum_pct),
        "maximum_pct": float(maximum_pct),
        "step_count": max(2, int(step_count)),
        "steps": generate_sensitivity_steps(minimum_pct, maximum_pct, step_count),
        "candidate_limit": applied_limit,
        "candidate_total": total,
        "candidate_limit_applied": total > MAX_ANALYSIS_CANDIDATES and requested_limit > MAX_ANALYSIS_CANDIDATES,
        "calculation_version": SENSITIVITY_CALCULATION_VERSION,
        "weight_version": VHS_WEIGHT_VERSION,
    }


def build_sensitivity_cache_key(
    data_signature: str,
    settings: Mapping[str, Any],
    weights: Mapping[str, float],
    recommendations: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    baseline = [
        {
            "route_id": item.get("route_id"),
            "vhs_score": item.get("vhs_score"),
            "rank": item.get("vhs_rank") or item.get("varo_final_rank") or item.get("rank"),
            "saving": item.get("expected_saving"),
        }
        for item in (recommendations or [])
    ]
    payload = {
        "data_signature": str(data_signature or ""),
        "variables": list(settings.get("variables") or []),
        "minimum_pct": settings.get("minimum_pct"),
        "maximum_pct": settings.get("maximum_pct"),
        "step_count": settings.get("step_count"),
        "steps": list(settings.get("steps") or []),
        "candidate_limit": settings.get("candidate_limit"),
        "weight_version": settings.get("weight_version", VHS_WEIGHT_VERSION),
        "calculation_version": settings.get("calculation_version", SENSITIVITY_CALCULATION_VERSION),
        "weights": {key: round(float(weights.get(key, 0.0) or 0.0), 12) for key in COMPONENTS},
        "baseline": baseline,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_candidates(
    recommendations: Sequence[Mapping[str, Any]], candidate_limit: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(copy.deepcopy(list(recommendations or [])))
    if frame.empty:
        return frame
    if "route_id" not in frame.columns:
        frame["route_id"] = [f"candidate-{index + 1}" for index in range(len(frame))]
    rank = _series(frame, ("vhs_rank", "varo_final_rank", "rank"), default=float("nan"))
    if not rank.notna().all():
        rank = rank_vhs_scores(_series(frame, ("vhs_score", "auto_vhs_score"), 0.0))
    frame["_base_rank"] = rank.astype(int)
    frame["_input_order"] = range(len(frame))
    frame = frame.sort_values(["_base_rank", "_input_order"], kind="mergesort").head(candidate_limit).reset_index(drop=True)
    frame["_base_vhs"] = _series(frame, ("vhs_score", "auto_vhs_score", "recalculated_vhs_score"), 0.0)
    frame["_base_saving"] = _series(frame, ("expected_saving",), 0.0)
    frame["_base_grade"] = [
        str(value) if value is not None and str(value).strip() else vhs_grade(score)
        for value, score in zip(frame.get("recommendation_grade", pd.Series(None, index=frame.index)), frame["_base_vhs"])
    ]
    for component in COMPONENTS:
        neutral = 0.0 if component == "dqn_reference_score" else 50.0
        frame[component] = _series(frame, (component,), neutral).fillna(neutral).clip(0, 100)
    return frame


def _normalized_delta(base_raw: pd.Series, changed_raw: pd.Series, *, higher: bool, neutral: float) -> pd.Series:
    baseline = normalize_vhs_values(base_raw, higher_is_better=higher, neutral=neutral)
    changed = normalize_vhs_values(changed_raw, higher_is_better=higher, neutral=neutral)
    return changed - baseline


def _quantity_with_constraints(
    base_quantity: pd.Series,
    proposed: pd.Series,
    contexts: Sequence[Mapping[str, float | None]],
    shortage_override: pd.Series | None = None,
) -> pd.Series:
    values: list[float] = []
    for position, (base, proposed_value) in enumerate(zip(base_quantity, proposed)):
        value = max(0.0, float(proposed_value))
        context = contexts[position]
        maximum = context.get("source_stock")
        shortage = shortage_override.iloc[position] if shortage_override is not None else context.get("shortage")
        if maximum is not None:
            value = min(value, max(0.0, float(maximum)))
        if shortage is not None:
            value = min(value, max(0.0, float(shortage)))
        if abs(float(base) - round(float(base))) < 1e-9:
            value = float(round(value))
        values.append(value)
    return pd.Series(values, index=base_quantity.index, dtype="float64")


def _saving_for_quantity(
    base_quantity: pd.Series,
    changed_quantity: pd.Series,
    base_saving: pd.Series,
    cost: pd.Series,
    contexts: Sequence[Mapping[str, float | None]],
) -> pd.Series:
    gross_per_unit: list[float] = []
    for qty, saving, move_cost, context in zip(base_quantity, base_saving, cost, contexts):
        if qty > 0:
            gross_per_unit.append((float(saving) + float(move_cost)) / float(qty))
        else:
            gross_per_unit.append(float(context.get("unit_price") or 0.0) * 0.5)
    gross = pd.Series(gross_per_unit, index=base_quantity.index, dtype="float64")
    return changed_quantity * gross - cost


def _renormalized_weight_change(
    weights: Mapping[str, float], component: str, change_pct: float,
) -> tuple[dict[str, float], float]:
    active = [name for name in COMPONENTS if float(weights.get(name, 0.0) or 0.0) > 0]
    result = {name: 0.0 for name in COMPONENTS}
    if component not in active:
        return result, 0.0
    base = float(weights[component])
    requested = max(0.0, base * (1.0 + change_pct / 100.0))
    requested = min(requested, 1.0)
    other_total = sum(float(weights[name]) for name in active if name != component)
    if not other_total:
        result[component] = 1.0
        return result, requested
    selected = min(requested, 1.0 - 1e-12)
    result[component] = selected
    remaining = 1.0 - selected
    for name in active:
        if name != component:
            result[name] = float(weights[name]) / other_total * remaining
    # Correct the last active value for floating point drift.
    correction = 1.0 - sum(result.values())
    result[active[-1]] += correction
    return result, requested


def apply_sensitivity_perturbation(
    frame: pd.DataFrame,
    contexts: Sequence[Mapping[str, float | None]],
    variable: str,
    change_pct: float,
    weights: Mapping[str, float],
    *,
    weight_component: str | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Return changed VHS/saving without modifying ``frame`` or ``weights``."""
    factor = max(0.0, 1.0 + float(change_pct) / 100.0)
    components = frame[list(COMPONENTS)].copy(deep=True)
    base_saving = frame["_base_saving"].astype(float).copy()
    changed_saving = base_saving.copy()
    cost = _series(frame, ("estimated_cost", "move_cost", "transport_cost"), 0.0).fillna(0).clip(lower=0)
    distance = _series(frame, ("distance_km", "route_distance_km", "direct_distance_km"), 0.0).fillna(0).clip(lower=0)
    travel_time = _series(frame, ("travel_time_min", "expected_time_min"), 0.0).fillna(0).clip(lower=0)
    quantity = _series(frame, ("recommended_qty", "suggested_qty", "quantity"), 0.0).fillna(0).clip(lower=0)
    metadata: dict[str, Any] = {"substitutions": [], "weight_component": weight_component}

    base_route_raw = cost * 0.60 + distance * 1000.0 * 0.25 + travel_time * 150.0 * 0.15

    if variable == "vhs_weight":
        changed_weights, requested = _renormalized_weight_change(weights, str(weight_component), change_pct)
        scores = calculate_weighted_vhs_scores(components, changed_weights)
        metadata.update({
            "weights": changed_weights,
            "base_weight": float(weights.get(str(weight_component), 0.0) or 0.0),
            "requested_weight": requested,
            "normalized_weight": changed_weights.get(str(weight_component), 0.0),
        })
        return scores, changed_saving, metadata

    if variable == "transport_cost":
        changed_cost = cost * factor
        changed_saving = base_saving + cost - changed_cost
        changed_route_raw = changed_cost * 0.60 + distance * 1000.0 * 0.25 + travel_time * 150.0 * 0.15
        components["route_cost_score"] = (
            components["route_cost_score"]
            + _normalized_delta(base_route_raw, changed_route_raw, higher=False, neutral=55)
        ).clip(0, 100)
    elif variable == "distance":
        changed_distance = distance * factor
        distance_ratio = changed_distance.divide(distance.where(distance > 0)).fillna(factor)
        changed_cost = cost * distance_ratio
        changed_saving = base_saving + cost - changed_cost
        changed_route_raw = changed_cost * 0.60 + changed_distance * 1000.0 * 0.25 + travel_time * 150.0 * 0.15
        components["route_cost_score"] = (
            components["route_cost_score"]
            + _normalized_delta(base_route_raw, changed_route_raw, higher=False, neutral=55)
        ).clip(0, 100)
    elif variable in {"demand", "shortage"}:
        demand = pd.Series([context.get("demand") for context in contexts], index=frame.index, dtype="float64")
        shortage = pd.Series([context.get("shortage") for context in contexts], index=frame.index, dtype="float64")
        target_stock = pd.Series([context.get("target_stock") for context in contexts], index=frame.index, dtype="float64")
        if variable == "demand":
            changed_demand = (demand * factor).clip(lower=0)
            changed_shortage = (changed_demand - target_stock).clip(lower=0)
            proposed = quantity * factor
            components["demand_fit_score"] = (
                components["demand_fit_score"]
                + _normalized_delta(demand, changed_demand, higher=True, neutral=50)
            ).clip(0, 100)
        else:
            changed_shortage = (shortage * factor).clip(lower=0)
            proposed = quantity * factor
            changed_demand = demand
        changed_quantity = _quantity_with_constraints(quantity, proposed, contexts, changed_shortage)
        changed_saving = _saving_for_quantity(quantity, changed_quantity, base_saving, cost, contexts)
        metadata["minimum_changed_demand"] = float(changed_demand.min(skipna=True))
        metadata["minimum_changed_quantity"] = float(changed_quantity.min(skipna=True))
        components["inventory_balance_score"] = (
            components["inventory_balance_score"]
            + _normalized_delta(quantity, changed_quantity, higher=True, neutral=50)
        ).clip(0, 100)
    elif variable == "quantity":
        changed_quantity = _quantity_with_constraints(quantity, quantity * factor, contexts)
        changed_saving = _saving_for_quantity(quantity, changed_quantity, base_saving, cost, contexts)
        metadata["minimum_changed_quantity"] = float(changed_quantity.min(skipna=True))
        components["inventory_balance_score"] = (
            components["inventory_balance_score"]
            + _normalized_delta(quantity, changed_quantity, higher=True, neutral=50)
        ).clip(0, 100)
    elif variable == "disposal_risk":
        components["disposal_risk_score"] = (components["disposal_risk_score"] * factor).clip(0, 100)
        if "avoided_disposal_cost" in frame.columns:
            avoided = pd.to_numeric(frame["avoided_disposal_cost"], errors="coerce").fillna(0)
            changed_saving = base_saving + avoided * (factor - 1.0)
    elif variable == "promotion":
        promotion = _series(frame, ("promotion_effect",), float("nan"))
        changed_promotion = (promotion * factor).clip(lower=0)
        components["promotion_score"] = (
            components["promotion_score"]
            + _normalized_delta(promotion, changed_promotion, higher=True, neutral=55)
        ).clip(0, 100)
    else:
        raise ValueError(f"지원하지 않는 민감도 변수: {variable}")

    components["savings_score"] = (
        components["savings_score"]
        + _normalized_delta(base_saving, changed_saving, higher=True, neutral=50)
    ).clip(0, 100)
    scores = calculate_weighted_vhs_scores(components, weights)
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("VHS 계산 결과에 유효하지 않은 값이 포함되었습니다.")
    if not all(math.isfinite(float(value)) for value in changed_saving):
        raise ValueError("절감액 계산 결과에 유효하지 않은 값이 포함되었습니다.")
    return scores, changed_saving, metadata


def _scenario_rows(
    frame: pd.DataFrame,
    variable: str,
    change_pct: float,
    scores: pd.Series,
    savings: pd.Series,
    metadata: Mapping[str, Any],
    scenario_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if change_pct == 0:
        scores = frame["_base_vhs"].copy()
        savings = frame["_base_saving"].copy()
        ranks = frame["_base_rank"].copy()
    else:
        ranks = rank_vhs_scores(scores)
    base_top1 = str(frame.sort_values("_base_rank", kind="mergesort").iloc[0]["route_id"])
    changed_top1 = str(frame.loc[ranks.idxmin(), "route_id"])
    base_top3 = set(frame.loc[frame["_base_rank"] <= 3, "route_id"].astype(str))
    changed_top3 = set(frame.loc[ranks <= 3, "route_id"].astype(str))
    overlap = len(base_top3 & changed_top3) / max(1, len(base_top3)) * 100.0
    component = metadata.get("weight_component")
    label = VARIABLE_LABELS[variable]
    if component:
        label = f"{label} · {COMPONENT_LABELS.get(str(component), component)}"
    rows: list[dict[str, Any]] = []
    for position, item in frame.iterrows():
        base_vhs = float(item["_base_vhs"])
        changed_vhs = float(scores.iloc[position])
        base_rank = int(item["_base_rank"])
        changed_rank = int(ranks.iloc[position])
        base_saving = float(item["_base_saving"])
        changed_saving = float(savings.iloc[position])
        changed_grade = vhs_grade(changed_vhs)
        rows.append({
            "scenario_id": scenario_id,
            "variable": label,
            "variable_key": variable,
            "weight_component": component,
            "change_pct": float(change_pct),
            "route_id": str(item.get("route_id") or "-"),
            "product_name": item.get("product_name") or "-",
            "source_name": item.get("source_name") or item.get("source_id") or "-",
            "target_name": item.get("target_name") or item.get("target_id") or "-",
            "base_vhs": base_vhs,
            "changed_vhs": changed_vhs,
            "vhs_delta": round(changed_vhs - base_vhs, 4),
            "base_rank": base_rank,
            "changed_rank": changed_rank,
            "rank_delta": changed_rank - base_rank,
            "base_saving": base_saving,
            "changed_saving": changed_saving,
            "saving_delta": round(changed_saving - base_saving, 4),
            "is_top1": changed_rank == 1,
            "top1_retained": changed_top1 == base_top1,
            "top3_retained": str(item.get("route_id")) in base_top3 & changed_top3,
            "grade_changed": changed_grade != str(item["_base_grade"]),
            "base_grade": str(item["_base_grade"]),
            "changed_grade": changed_grade,
            "status": "계산 완료",
            "error": "",
        })
    scenario = {
        "scenario_id": scenario_id,
        "variable": label,
        "variable_key": variable,
        "weight_component": component,
        "change_pct": float(change_pct),
        "top1_retained": changed_top1 == base_top1,
        "top3_retention_rate": round(overlap, 4),
        "max_rank_change": max(abs(row["rank_delta"]) for row in rows) if rows else 0,
        "total_saving": round(float(savings.sum()), 4),
        "status": "계산 완료",
        "error": "",
        "base_weight": metadata.get("base_weight"),
        "requested_weight": metadata.get("requested_weight"),
        "normalized_weight": metadata.get("normalized_weight"),
        "weight_sum": round(sum((metadata.get("weights") or {}).values()), 12) if metadata.get("weights") else None,
    }
    return rows, scenario


def calculate_sensitivity_stability_score(
    detail_rows: Sequence[Mapping[str, Any]],
    scenario_rows: Sequence[Mapping[str, Any]],
    candidate_count: int,
) -> dict[str, Any]:
    successful_all = [row for row in scenario_rows if row.get("status") == "계산 완료"]
    details_all = [row for row in detail_rows if row.get("status") == "계산 완료"]
    # The 0% rows prove baseline continuity but carry no perturbation signal, so
    # they are deliberately excluded from the stability average.
    successful = [row for row in successful_all if float(row.get("change_pct") or 0) != 0] or successful_all
    details = [row for row in details_all if float(row.get("change_pct") or 0) != 0] or details_all
    top1 = sum(bool(row.get("top1_retained")) for row in successful) / max(1, len(successful)) * 100.0
    top3 = sum(float(row.get("top3_retention_rate") or 0) for row in successful) / max(1, len(successful))
    max_rank_span = max(1, candidate_count - 1)
    rank_movement = sum(abs(float(row.get("rank_delta") or 0)) for row in details) / max(1, len(details))
    rank_stability = max(0.0, 100.0 * (1.0 - rank_movement / max_rank_span))
    vhs_movement = sum(abs(float(row.get("vhs_delta") or 0)) for row in details) / max(1, len(details))
    vhs_stability = max(0.0, 100.0 * (1.0 - vhs_movement / 100.0))
    saving_ratios = [
        min(1.0, abs(float(row.get("saving_delta") or 0)) / (abs(float(row.get("base_saving") or 0)) + 1.0))
        for row in details
    ]
    saving_stability = max(0.0, 100.0 * (1.0 - sum(saving_ratios) / max(1, len(saving_ratios))))
    score = min(100.0, max(0.0,
        top1 * 0.40 + top3 * 0.20 + rank_stability * 0.20
        + vhs_stability * 0.10 + saving_stability * 0.10
    ))
    rating = "매우 안정" if score >= 85 else "안정" if score >= 70 else "조건에 따라 변동" if score >= 50 else "민감"
    return {
        "score": round(score, 2),
        "rating": rating,
        "top1_retention_rate": round(top1, 2),
        "top3_retention_rate": round(top3, 2),
        "rank_stability": round(rank_stability, 2),
        "vhs_stability": round(vhs_stability, 2),
        "saving_stability": round(saving_stability, 2),
    }


def summarize_sensitivity_results(
    detail_rows: Sequence[Mapping[str, Any]],
    scenario_rows: Sequence[Mapping[str, Any]],
    variable_count: int,
    candidate_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stability = calculate_sensitivity_stability_score(detail_rows, scenario_rows, candidate_count)
    completed = [row for row in scenario_rows if row.get("status") == "계산 완료"]
    excluded = [row for row in scenario_rows if row.get("status") != "계산 완료"]
    variable_summaries: list[dict[str, Any]] = []
    for variable in dict.fromkeys(row.get("variable") for row in completed):
        all_scenarios = [row for row in completed if row.get("variable") == variable]
        all_details = [row for row in detail_rows if row.get("variable") == variable and row.get("status") == "계산 완료"]
        scenarios = [row for row in all_scenarios if float(row.get("change_pct") or 0) != 0] or all_scenarios
        details = [row for row in all_details if float(row.get("change_pct") or 0) != 0] or all_details
        average_rank = sum(abs(float(row.get("rank_delta") or 0)) for row in details) / max(1, len(details))
        max_rank = max((abs(int(row.get("rank_delta") or 0)) for row in details), default=0)
        top1 = sum(bool(row.get("top1_retained")) for row in scenarios) / max(1, len(scenarios)) * 100.0
        top3 = sum(float(row.get("top3_retention_rate") or 0) for row in scenarios) / max(1, len(scenarios))
        variable_summaries.append({
            "variable": variable,
            "scenario_count": len(all_scenarios),
            "top1_retention_rate": round(top1, 2),
            "top3_retention_rate": round(top3, 2),
            "average_abs_rank_change": round(average_rank, 4),
            "max_rank_change": max_rank,
            "max_abs_vhs_change": round(max((abs(float(row.get("vhs_delta") or 0)) for row in details), default=0), 4),
            "saving_min": round(min((float(row.get("changed_saving") or 0) for row in details), default=0), 4),
            "saving_max": round(max((float(row.get("changed_saving") or 0) for row in details), default=0), 4),
        })
    variable_summaries.sort(key=lambda row: (-row["average_abs_rank_change"], row["top1_retention_rate"], row["variable"]))
    first_change = next((row for row in completed if not row.get("top1_retained") and float(row.get("change_pct") or 0) != 0), None)
    total_savings = [float(row.get("total_saving") or 0) for row in completed]
    summary = {
        "variable_count": variable_count,
        "scenario_count": len(scenario_rows),
        "completed_scenario_count": len(completed),
        "excluded_scenario_count": len(excluded),
        "candidate_count": candidate_count,
        "max_rank_change": max((int(row.get("max_rank_change") or 0) for row in completed), default=0),
        "most_sensitive_variable": variable_summaries[0]["variable"] if variable_summaries else "-",
        "most_stable_variable": variable_summaries[-1]["variable"] if variable_summaries else "-",
        "first_top1_change": (
            f"{first_change['variable']} {float(first_change['change_pct']):+.1f}%" if first_change else "변경 없음"
        ),
        "total_saving_min": round(min(total_savings), 4) if total_savings else 0.0,
        "total_saving_max": round(max(total_savings), 4) if total_savings else 0.0,
        **stability,
    }
    return summary, variable_summaries


def run_detailed_sensitivity(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    weights: Mapping[str, float],
    settings: Mapping[str, Any],
    data_signature: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run or retrieve a detailed OAT analysis and return a deep copy."""
    cache_key = build_sensitivity_cache_key(data_signature, settings, weights, recommendations)
    if cache_key in _RESULT_CACHE:
        cached = copy.deepcopy(_RESULT_CACHE[cache_key])
        cached["metadata"]["cache_hit"] = True
        return cached

    started = time.perf_counter()
    frame = _prepare_candidates(recommendations, int(settings.get("candidate_limit") or 0))
    if frame.empty:
        return {
            "detail_rows": [], "scenario_rows": [], "variable_summary": [], "weight_rows": [],
            "summary": {}, "settings": dict(settings),
            "metadata": {"cache_key": cache_key, "cache_hit": False, "errors": ["분석 후보가 없습니다."]},
        }
    selected_records = frame.where(pd.notna(frame), None).to_dict("records")
    contexts = _inventory_context(selected_records, data)
    variables = list(settings.get("variables") or [])
    steps = [float(value) for value in (settings.get("steps") or [0.0])]
    active_components = [component for component in COMPONENTS if float(weights.get(component, 0.0) or 0.0) > 0]
    scenario_specs: list[tuple[str, float, str | None]] = []
    for variable in variables:
        if variable == "vhs_weight":
            scenario_specs.extend((variable, step, component) for component in active_components for step in steps)
        else:
            scenario_specs.extend((variable, step, None) for step in steps)

    total = len(scenario_specs)
    if progress_callback:
        progress_callback("기준 결과 준비 중", 0, total, "-")
    detail_rows: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, (variable, step, component) in enumerate(scenario_specs, start=1):
        current_label = VARIABLE_LABELS.get(variable, variable)
        if component:
            current_label += f" · {COMPONENT_LABELS.get(component, component)}"
        if progress_callback:
            progress_callback("변수별 시나리오 계산 중", index - 1, total, current_label)
        scenario_id = f"{variable}:{component or '-'}:{step:+.6f}"
        try:
            if step == 0:
                scores = frame["_base_vhs"].copy()
                savings = frame["_base_saving"].copy()
                metadata = {"weight_component": component}
                if component:
                    metadata.update({
                        "base_weight": float(weights.get(component, 0.0) or 0.0),
                        "requested_weight": float(weights.get(component, 0.0) or 0.0),
                        "normalized_weight": float(weights.get(component, 0.0) or 0.0),
                        "weights": dict(weights),
                    })
            else:
                scores, savings, metadata = apply_sensitivity_perturbation(
                    frame, contexts, variable, step, weights, weight_component=component,
                )
            rows, scenario = _scenario_rows(frame, variable, step, scores, savings, metadata, scenario_id)
            detail_rows.extend(rows)
            scenarios.append(scenario)
        except Exception as exc:  # keep independent scenarios isolated
            error = {"variable": current_label, "change_pct": step, "error": str(exc)}
            errors.append(error)
            scenarios.append({
                "scenario_id": scenario_id, "variable": current_label, "variable_key": variable,
                "weight_component": component, "change_pct": step, "status": "계산 제외", "error": str(exc),
                "top1_retained": False, "top3_retention_rate": 0.0, "max_rank_change": 0,
            })
        if progress_callback:
            progress_callback("변수별 시나리오 계산 중", index, total, current_label)

    if progress_callback:
        progress_callback("순위 변동 분석 중", total, total, "-")
    summary, variable_summary = summarize_sensitivity_results(
        detail_rows, scenarios, len(variables), len(frame),
    )
    if progress_callback:
        progress_callback("안정성 점수 계산 중", total, total, "-")
    weight_rows = [row for row in scenarios if row.get("weight_component")]
    result = {
        "detail_rows": detail_rows,
        "scenario_rows": scenarios,
        "variable_summary": variable_summary,
        "weight_rows": weight_rows,
        "summary": summary,
        "settings": copy.deepcopy(dict(settings)),
        "metadata": {
            "cache_key": cache_key,
            "cache_hit": False,
            "calculation_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "errors": errors,
            "substitution_policy": "기존 VHS 중립값을 유지하고 원본 필드가 없는 변수는 분석 목록에서 제외",
            "candidate_cap": MAX_ANALYSIS_CANDIDATES,
            "calculation_version": SENSITIVITY_CALCULATION_VERSION,
            "weight_version": VHS_WEIGHT_VERSION,
        },
    }
    if progress_callback:
        progress_callback("결과 정리 중", total, total, "-")
    _RESULT_CACHE[cache_key] = copy.deepcopy(result)
    _RESULT_CACHE.move_to_end(cache_key)
    while len(_RESULT_CACHE) > _CACHE_LIMIT:
        _RESULT_CACHE.popitem(last=False)
    return copy.deepcopy(result)


DETAIL_EXPORT_HEADERS = {
    "variable": "변수", "change_pct": "변화율(%)", "route_id": "route_id",
    "product_name": "상품", "source_name": "출발 점포", "target_name": "도착 점포",
    "base_vhs": "기준 VHS", "changed_vhs": "변화 후 VHS", "vhs_delta": "VHS 변화량",
    "base_rank": "기준 순위", "changed_rank": "변화 후 순위", "rank_delta": "순위 변화",
    "base_saving": "기준 예상 절감액", "changed_saving": "변화 후 예상 절감액",
    "saving_delta": "절감액 변화량", "is_top1": "Top1 여부", "top3_retained": "Top3 유지",
    "grade_changed": "추천 등급 변화", "status": "상태", "error": "오류",
}
SUMMARY_EXPORT_HEADERS = {
    "variable": "변수", "scenario_count": "시나리오 수",
    "top1_retention_rate": "Top1 유지율(%)", "top3_retention_rate": "Top3 유지율(%)",
    "average_abs_rank_change": "평균 절대 순위 변화", "max_rank_change": "최대 순위 변화",
    "max_abs_vhs_change": "최대 VHS 변화", "saving_min": "절감액 최소", "saving_max": "절감액 최대",
}


def sensitivity_detail_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result.get("detail_rows") or [])
    columns = [column for column in DETAIL_EXPORT_HEADERS if column in frame.columns]
    return frame[columns].rename(columns=DETAIL_EXPORT_HEADERS) if columns else pd.DataFrame(columns=DETAIL_EXPORT_HEADERS.values())


def sensitivity_summary_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(result.get("variable_summary") or [])
    columns = [column for column in SUMMARY_EXPORT_HEADERS if column in frame.columns]
    return frame[columns].rename(columns=SUMMARY_EXPORT_HEADERS) if columns else pd.DataFrame(columns=SUMMARY_EXPORT_HEADERS.values())


def clear_sensitivity_cache() -> None:
    """Test/maintenance hook; application state clearing does not require this."""
    _RESULT_CACHE.clear()
