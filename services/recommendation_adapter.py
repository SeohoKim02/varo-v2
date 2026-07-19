"""Standard recommendation contract shared by Varo V2 UI and simulations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Mapping, Optional

import pandas as pd

RouteType = Literal["DIRECT", "VIA_DC"]
ROUTE_TYPES = ("DIRECT", "VIA_DC")

ALGORITHM_RESULT_FIELDS = (
    "algorithm_name", "recommendation_id", "route_id", "product", "source",
    "target", "route_type", "quantity", "score", "rank", "expected_savings",
    "feasibility", "confidence", "explanation", "data_signature",
)

RECOMMENDATION_FIELDS = (
    "route_id", "product_id", "product_name", "source_id", "source_name",
    "target_id", "target_name", "dc_id", "dc_name", "route_type",
    "route_type_label", "recommended_qty", "transport_type", "transport_label",
    "distance_km", "expected_time_min", "travel_time_min", "move_cost",
    "estimated_cost", "expected_saving", "vhs_score", "uploaded_vhs_score",
    "recalculated_vhs_score", "vhs_score_source", "greedy_action",
    "greedy_rank", "heuristic_score", "greedy_selected", "greedy_reason",
    "strategy_match", "varo_action", "dqn_action", "dqn_confidence", "dqn_status", "dqn_correction",
    "confidence", "confidence_score", "confidence_level", "confidence_reason",
    "confidence_source", "grade",
    "recommendation_grade", "reason", "status", "rank",
    "direct_distance_km", "direct_time_min", "direct_cost",
    "via_dc_distance_km", "via_dc_time_min", "via_dc_cost",
    "time_window_status", "time_window_reason", "arrival_time",
    "available_window", "cutline_passed", "cutline_reason",
    "distance_cutline_km", "path_reason", "promotion_recommended",
    "promotion_effect", "promotion_transfer_cost", "promotion_net_cost",
    "promotion_reason", "vhs_rank", "greedy_strategy",
    "varo_final_decision", "varo_final_rank", "dqn_reference_score",
    "vhs_vs_greedy_match", "vhs_vs_dqn_match", "final_reason",
    "weight_profile_id", "weight_summary",
    "savings_score", "disposal_risk_score", "demand_fit_score",
    "inventory_balance_score", "route_cost_score", "feasibility_score",
    "promotion_score", "greedy_score", "dqn_reference_score",
    "pareto_rank", "pareto_status", "pareto_reason",
)
NUMERIC_FIELDS = (
    "recommended_qty", "distance_km", "expected_time_min", "travel_time_min",
    "move_cost", "estimated_cost", "expected_saving", "vhs_score",
    "confidence", "confidence_score", "rank", "direct_distance_km",
    "direct_time_min", "direct_cost", "via_dc_distance_km", "via_dc_time_min",
    "via_dc_cost", "uploaded_vhs_score", "recalculated_vhs_score",
    "greedy_rank", "heuristic_score", "dqn_confidence", "dqn_correction",
    "distance_cutline_km", "promotion_effect", "promotion_transfer_cost",
    "promotion_net_cost", "vhs_rank", "varo_final_rank",
    "dqn_reference_score", "savings_score", "disposal_risk_score",
    "demand_fit_score", "inventory_balance_score", "route_cost_score",
    "feasibility_score", "promotion_score", "greedy_score",
    "pareto_rank",
)

_ACTION_ALIASES = {
    "multi_store_transfer": "재고 이동", "transfer": "재고 이동",
    "direct_transfer": "재고 이동", "dc_transfer": "재고 이동",
    "store_transfer": "재고 이동", "relocation": "재고 이동",
    "재배치 이동": "재고 이동", "재배치": "재고 이동", "이동": "재고 이동",
    "discount_sale": "할인", "discount": "할인", "할인 판매": "할인",
    "emergency_discount": "긴급 할인", "urgent_discount": "긴급 할인",
    "one_plus_one": "1+1", "plus_one": "1+1",
    "dispose": "폐기", "discard": "폐기", "waste": "폐기",
    "keep_inventory": "보류", "hold": "보류", "no_action": "보류",
    "maintain": "보류", "모니터링": "보류",
}


@dataclass(frozen=True)
class StandardRecommendation:
    recommendation_id: str = ""
    route_id: str = ""
    product_id: str = ""
    product_name: str = ""
    source_id: str = ""
    source_name: str = ""
    target_id: str = ""
    target_name: str = ""
    dc_id: Optional[str] = None
    dc_name: Optional[str] = None
    route_type: RouteType = "DIRECT"
    route_type_label: Optional[str] = None
    recommended_qty: Optional[float] = None
    transport_type: Optional[str] = None
    transport_label: Optional[str] = None
    distance_km: Optional[float] = None
    expected_time_min: Optional[float] = None
    travel_time_min: Optional[float] = None
    move_cost: Optional[float] = None
    estimated_cost: Optional[float] = None
    expected_saving: Optional[float] = None
    vhs_score: Optional[float] = None
    uploaded_vhs_score: Optional[float] = None
    recalculated_vhs_score: Optional[float] = None
    vhs_score_source: str = "업로드된 사전 계산 VHS"
    greedy_action: Optional[str] = None
    greedy_rank: Optional[float] = None
    heuristic_score: Optional[float] = None
    greedy_selected: bool = False
    greedy_reason: Optional[str] = None
    strategy_match: Optional[bool] = None
    varo_action: Optional[str] = None
    dqn_action: str = "미연결"
    dqn_confidence: Optional[float] = None
    dqn_status: str = "학습 필요"
    dqn_correction: float = 0.0
    confidence: Optional[float] = None
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None
    confidence_reason: Optional[str] = None
    confidence_source: str = "vhs_confidence.add_confidence_columns · DQN 제외"
    grade: Optional[str] = None
    recommendation_grade: Optional[str] = None
    reason: Optional[str] = None
    status: str = "READY"
    rank: Optional[float] = None
    direct_distance_km: Optional[float] = None
    direct_time_min: Optional[float] = None
    direct_cost: Optional[float] = None
    via_dc_distance_km: Optional[float] = None
    via_dc_time_min: Optional[float] = None
    via_dc_cost: Optional[float] = None
    time_window_status: Optional[str] = None
    time_window_reason: Optional[str] = None
    arrival_time: Optional[str] = None
    available_window: Optional[str] = None
    cutline_passed: Optional[str] = None
    cutline_reason: Optional[str] = None
    distance_cutline_km: Optional[float] = None
    path_reason: Optional[str] = None
    promotion_recommended: Optional[str] = None
    promotion_effect: Optional[float] = None
    promotion_transfer_cost: Optional[float] = None
    promotion_net_cost: Optional[float] = None
    promotion_reason: Optional[str] = None
    vhs_rank: Optional[float] = None
    greedy_strategy: Optional[str] = None
    varo_final_decision: Optional[str] = None
    varo_final_rank: Optional[float] = None
    dqn_reference_score: Optional[float] = None
    vhs_vs_greedy_match: Optional[bool] = None
    vhs_vs_dqn_match: Optional[bool] = None
    final_reason: Optional[str] = None
    weight_profile_id: Optional[str] = None
    weight_summary: Optional[str] = None
    savings_score: Optional[float] = None
    disposal_risk_score: Optional[float] = None
    demand_fit_score: Optional[float] = None
    inventory_balance_score: Optional[float] = None
    route_cost_score: Optional[float] = None
    feasibility_score: Optional[float] = None
    promotion_score: Optional[float] = None
    greedy_score: Optional[float] = None
    pareto_rank: Optional[float] = None
    pareto_status: Optional[str] = None
    pareto_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _blank(value) -> bool:
    try:
        return bool(pd.isna(value)) or str(value).strip() == ""
    except (TypeError, ValueError):
        return value is None


def _clean_text(value) -> Optional[str]:
    if _blank(value):
        return None
    return str(value).strip()


def _number(value, field: str) -> Optional[float]:
    if _blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 값을 숫자로 변환할 수 없습니다: {value}") from exc


def _boolean(value, default: bool | None = None) -> bool | None:
    if _blank(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "y", "yes", "예", "일치"}:
        return True
    if text in {"false", "0", "n", "no", "아니오", "불일치"}:
        return False
    return default


def _first(item: Mapping[str, object], *keys: str):
    for key in keys:
        if key in item and not _blank(item.get(key)):
            return item.get(key)
    return None


def normalize_action(value: object, default: str = "비교 불가") -> str:
    text = _clean_text(value)
    if not text:
        return default
    lowered = text.lower()
    if lowered in _ACTION_ALIASES:
        return _ACTION_ALIASES[lowered]
    for token, label in _ACTION_ALIASES.items():
        if token in lowered:
            return label
    return text


def empty_recommendations() -> List[Dict[str, object]]:
    return []


def validate_standard_recommendation(item: Mapping[str, object]) -> None:
    missing = [field for field in RECOMMENDATION_FIELDS if field not in item]
    if missing:
        raise ValueError(f"Missing standard recommendation fields: {missing}")
    for field in ("route_id", "product_id", "product_name", "source_id", "target_id", "route_type"):
        if _blank(item.get(field)):
            raise ValueError(f"{field} 값이 필요합니다.")
    route_type = item.get("route_type")
    if route_type not in ROUTE_TYPES:
        raise ValueError("route_type must be DIRECT or VIA_DC")
    if route_type == "VIA_DC" and (_blank(item.get("dc_id")) or _blank(item.get("dc_name"))):
        raise ValueError("VIA_DC recommendations require dc_id and dc_name")
    qty = item.get("recommended_qty")
    if qty is None or float(qty) <= 0:
        raise ValueError("recommended_qty는 0보다 커야 합니다.")
    for field in ("estimated_cost", "move_cost", "distance_km"):
        value = item.get(field)
        if value is not None and float(value) < 0:
            raise ValueError(f"{field}는 음수일 수 없습니다.")


def normalize_standard_recommendation(item: Mapping[str, object]) -> Dict[str, object]:
    route_type = (_clean_text(_first(item, "route_type")) or "").upper()
    expected_time = _first(item, "expected_time_min", "travel_time_min", "recommended_time_min")
    move_cost = _first(item, "move_cost", "estimated_cost", "transport_cost", "recommended_cost")
    confidence = _first(item, "confidence", "confidence_score", "recommendation_confidence")
    grade = _first(item, "grade", "recommendation_grade", "vhs_grade")
    normalized: dict[str, object] = {
        "recommendation_id": _first(item, "recommendation_id", "route_id"),
        "route_id": _first(item, "route_id", "recommendation_id"),
        "product_id": _first(item, "product_id", "item_id"),
        "product_name": _first(item, "product_name", "item_name"),
        "source_id": _first(item, "source_id", "source_store_id", "from_id"),
        "source_name": _first(item, "source_name", "source_store", "from_store"),
        "target_id": _first(item, "target_id", "target_store_id", "to_id"),
        "target_name": _first(item, "target_name", "target_store", "to_store"),
        "dc_id": _first(item, "dc_id"),
        "dc_name": _first(item, "dc_name", "via_dc_name"),
        "route_type": route_type,
        "route_type_label": "직접 이동" if route_type == "DIRECT" else "DC 경유" if route_type == "VIA_DC" else None,
        "recommended_qty": _first(item, "recommended_qty", "suggested_qty", "transfer_qty"),
        "transport_type": _first(item, "transport_type", "transport_mode"),
        "transport_label": _first(item, "transport_label", "transport_type", "transport_mode"),
        "distance_km": _first(item, "distance_km", "recommended_distance_km"),
        "expected_time_min": expected_time,
        "travel_time_min": expected_time,
        "move_cost": move_cost,
        "estimated_cost": move_cost,
        "expected_saving": _first(item, "expected_saving", "saving_amount", "disposal_avoidance_value"),
        "vhs_score": _first(item, "vhs_score", "vhs", "vhs2"),
        "uploaded_vhs_score": _first(item, "uploaded_vhs_score", "vhs_score"),
        "recalculated_vhs_score": _first(item, "recalculated_vhs_score", "vhs", "vhs2"),
        "vhs_score_source": _first(item, "vhs_score_source") or "업로드된 사전 계산 VHS",
        "greedy_action": normalize_action(_first(item, "greedy_action", "greedy_strategy", "final_recommendation")),
        "greedy_rank": _first(item, "greedy_rank"),
        "heuristic_score": _first(item, "heuristic_score"),
        "greedy_selected": _boolean(_first(item, "greedy_selected", "is_greedy_selected"), False),
        "greedy_reason": _first(item, "greedy_reason"),
        "strategy_match": _boolean(_first(item, "strategy_match"), None),
        "varo_action": normalize_action(_first(item, "varo_action", "vhs_action", "final_recommendation", "greedy_action")),
        "dqn_action": "미연결",
        "dqn_confidence": _first(item, "dqn_confidence"),
        "dqn_status": _first(item, "dqn_status") or "학습 필요",
        "dqn_correction": 0.0,
        "confidence": confidence,
        "confidence_score": confidence,
        "confidence_level": _first(item, "confidence_level"),
        "confidence_reason": _first(item, "confidence_reason"),
        "confidence_source": _first(item, "confidence_source") or "vhs_confidence.add_confidence_columns · DQN 제외",
        "grade": grade,
        "recommendation_grade": grade,
        "reason": _first(item, "reason", "vhs_reason", "transfer_reason", "path_reason"),
        "status": _first(item, "status") or "READY",
        "rank": _first(item, "rank", "recommendation_rank", "vhs_rank", "greedy_rank"),
        "direct_distance_km": _first(item, "direct_distance_km"),
        "direct_time_min": _first(item, "direct_time_min"),
        "direct_cost": _first(item, "direct_cost"),
        "via_dc_distance_km": _first(item, "via_dc_distance_km", "via_distance_km"),
        "via_dc_time_min": _first(item, "via_dc_time_min", "via_time_min"),
        "via_dc_cost": _first(item, "via_dc_cost", "via_cost"),
        "time_window_status": _first(item, "time_window_status", "trade_time_status"),
        "time_window_reason": _first(item, "time_window_reason", "time_reason"),
        "arrival_time": _first(item, "arrival_time"),
        "available_window": _first(item, "available_window"),
        "cutline_passed": _first(item, "cutline_passed", "distance_cutline_status"),
        "cutline_reason": _first(item, "cutline_reason"),
        "distance_cutline_km": _first(item, "distance_cutline_km"),
        "path_reason": _first(item, "path_reason", "transfer_reason"),
        "promotion_recommended": _first(item, "promotion_recommended", "final_decision"),
        "promotion_effect": _first(item, "promotion_effect", "promotion_net_effect"),
        "promotion_transfer_cost": _first(item, "promotion_transfer_cost", "transfer_cost"),
        "promotion_net_cost": _first(item, "promotion_net_cost"),
        "promotion_reason": _first(item, "promotion_reason", "decision_reason"),
        "vhs_rank": _first(item, "vhs_rank"),
        "greedy_strategy": normalize_action(_first(item, "greedy_strategy", "greedy_action")),
        "varo_final_decision": _first(item, "varo_final_decision"),
        "varo_final_rank": _first(item, "varo_final_rank"),
        "dqn_reference_score": _first(item, "dqn_reference_score"),
        "vhs_vs_greedy_match": _boolean(_first(item, "vhs_vs_greedy_match"), None),
        "vhs_vs_dqn_match": _boolean(_first(item, "vhs_vs_dqn_match"), None),
        "final_reason": _first(item, "final_reason"),
        "weight_profile_id": _first(item, "weight_profile_id"),
        "weight_summary": _first(item, "weight_summary"),
        "savings_score": _first(item, "savings_score"),
        "disposal_risk_score": _first(item, "disposal_risk_score"),
        "demand_fit_score": _first(item, "demand_fit_score"),
        "inventory_balance_score": _first(item, "inventory_balance_score"),
        "route_cost_score": _first(item, "route_cost_score"),
        "feasibility_score": _first(item, "feasibility_score"),
        "promotion_score": _first(item, "promotion_score"),
        "greedy_score": _first(item, "greedy_score"),
        "pareto_rank": _first(item, "pareto_rank"),
        "pareto_status": _first(item, "pareto_status"),
        "pareto_reason": _first(item, "pareto_reason"),
    }
    for key in (
        "recommendation_id", "route_id", "product_id", "product_name", "source_id", "source_name", "target_id", "target_name",
        "dc_id", "dc_name", "route_type_label", "transport_type", "transport_label", "greedy_action",
        "varo_action", "dqn_action", "dqn_status", "vhs_score_source", "greedy_reason",
        "confidence_level", "confidence_reason", "confidence_source", "grade",
        "recommendation_grade", "reason", "status", "time_window_status",
        "time_window_reason", "arrival_time", "available_window", "cutline_passed",
        "cutline_reason", "path_reason", "promotion_recommended", "promotion_reason",
        "greedy_strategy", "varo_final_decision", "final_reason",
        "weight_profile_id", "weight_summary",
        "pareto_status", "pareto_reason",
    ):
        normalized[key] = _clean_text(normalized.get(key))
    for key in NUMERIC_FIELDS:
        normalized[key] = _number(normalized.get(key), key)
    if normalized["route_type"] == "DIRECT":
        normalized["dc_id"] = None
        normalized["dc_name"] = None
    validate_standard_recommendation(normalized)
    return normalized


def recommendations_from_dataframe(df: pd.DataFrame) -> List[Dict[str, object]]:
    if df is None or df.empty:
        return []
    recommendations: list[dict[str, object]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, row in df.iterrows():
        route_label = row.get("route_id", f"row {index + 1}")
        try:
            normalized = normalize_standard_recommendation(row.to_dict())
        except ValueError as exc:
            errors.append(f"{route_label}: {exc}")
            continue
        route_id = str(normalized["route_id"])
        if route_id in seen:
            errors.append(f"{route_id}: 중복 route_id입니다.")
            continue
        seen.add(route_id)
        recommendations.append(normalized)
    if errors:
        raise ValueError("추천 결과 변환 오류: " + " / ".join(errors))
    return recommendations


def validate_recommendation_list(recommendations: List[Mapping[str, object]]) -> List[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for item in recommendations:
        route_id = item.get("route_id")
        try:
            validate_standard_recommendation(item)
        except ValueError as exc:
            errors.append(str(exc))
        if route_id in seen:
            errors.append(f"중복 route_id입니다: {route_id}")
        if route_id:
            seen.add(str(route_id))
    return errors


def normalize_algorithm_result(
    item: Mapping[str, object],
    algorithm_name: str,
    data_signature: str | None = None,
) -> Dict[str, object]:
    """Project one algorithm result onto the shared comparison contract.

    This is a read-only projection.  It does not recalculate a score or alter
    the existing recommendation order.
    """
    name = str(algorithm_name or "").strip() or "unknown"
    lowered = name.casefold()
    if "greedy" in lowered:
        score = _first(item, "greedy_score", "heuristic_score")
        rank = _first(item, "greedy_rank", "rank")
        feasibility = _first(item, "greedy_selected", "feasibility_score", "status")
        explanation = _first(item, "greedy_reason", "greedy_strategy", "reason")
    elif "dqn" in lowered:
        score = _first(item, "dqn_reference_score", "dqn_score")
        rank = _first(item, "dqn_rank", "rank")
        feasibility = _first(item, "dqn_status", "status")
        explanation = _first(item, "dqn_action", "final_reason", "reason")
    elif "pareto" in lowered:
        score = _first(item, "pareto_score", "vhs_score")
        rank = _first(item, "pareto_rank", "rank")
        feasibility = _first(item, "pareto_status", "status")
        explanation = _first(item, "pareto_reason", "reason")
    elif "optimal" in lowered or "최적" in name:
        score = _first(item, "optimal_score", "objective_value", "expected_saving")
        rank = _first(item, "optimal_rank", "rank")
        feasibility = _first(item, "optimal_status", "feasible", "status")
        explanation = _first(item, "optimal_reason", "reason")
    else:  # VHS and future ranking adapters use the current standard fields.
        score = _first(item, "vhs_score", "recalculated_vhs_score", "uploaded_vhs_score", "score")
        rank = _first(item, "vhs_rank", "varo_final_rank", "rank")
        feasibility = _first(item, "feasibility_score", "status")
        explanation = _first(item, "reason", "final_reason", "path_reason")
    result: Dict[str, object] = {
        "algorithm_name": name,
        "recommendation_id": _first(item, "recommendation_id", "route_id"),
        "route_id": _first(item, "route_id", "recommendation_id"),
        "product": _first(item, "product_name", "product_id", "item_name", "item_id"),
        "source": _first(item, "source_name", "source_id", "from_store_name", "from_store_id"),
        "target": _first(item, "target_name", "target_id", "to_store_name", "to_store_id"),
        "route_type": _first(item, "route_type"),
        "quantity": _first(item, "recommended_qty", "suggested_qty", "transfer_qty", "quantity"),
        "score": score,
        "rank": rank,
        "expected_savings": _first(item, "expected_saving", "saving_amount", "estimated_saving"),
        "feasibility": feasibility,
        "confidence": _first(item, "confidence", "confidence_score", "dqn_confidence"),
        "explanation": explanation,
        "data_signature": data_signature or _first(item, "data_signature") or "",
    }
    return {field: result.get(field) for field in ALGORITHM_RESULT_FIELDS}


def algorithm_comparison_rows(
    recommendations: List[Mapping[str, object]],
    data_signature: str | None = None,
    algorithms: tuple[str, ...] = ("VHS", "Greedy", "DQN", "Pareto"),
) -> List[Dict[str, object]]:
    """Return comparison rows without sorting or mutating recommendations."""
    return [
        normalize_algorithm_result(item, algorithm, data_signature)
        for item in recommendations
        for algorithm in algorithms
    ]
