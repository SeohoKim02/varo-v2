"""Data-driven VHS scoring and strategy comparison for Varo V2.

This module is deliberately self-contained: it uses only the current V2
candidate frame, never reads historical DQN artifacts, and is deterministic for
the same input data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from services.recommendation_adapter import normalize_action

COMPONENTS = (
    "savings_score",
    "disposal_risk_score",
    "demand_fit_score",
    "inventory_balance_score",
    "route_cost_score",
    "feasibility_score",
    "promotion_score",
    "greedy_score",
    "confidence_score",
    "dqn_reference_score",
)

WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "savings_score": (0.18, 0.35),
    "feasibility_score": (0.12, 0.25),
    "disposal_risk_score": (0.08, 0.20),
    "demand_fit_score": (0.08, 0.20),
    "inventory_balance_score": (0.08, 0.18),
    "route_cost_score": (0.06, 0.16),
    "promotion_score": (0.03, 0.12),
    "greedy_score": (0.03, 0.12),
    "confidence_score": (0.05, 0.15),
    "dqn_reference_score": (0.00, 0.08),
}

BASE_WEIGHTS: dict[str, float] = {
    "savings_score": 0.24,
    "feasibility_score": 0.16,
    "disposal_risk_score": 0.11,
    "demand_fit_score": 0.11,
    "inventory_balance_score": 0.10,
    "route_cost_score": 0.09,
    "promotion_score": 0.05,
    "greedy_score": 0.06,
    "confidence_score": 0.08,
    "dqn_reference_score": 0.00,
}

DQN_READY_STATUSES = {"연결", "정상", "connected", "ok", "ready"}


@dataclass(frozen=True)
class VhsAutoResult:
    frame: pd.DataFrame
    analysis: dict[str, Any]
    comparison_rows: list[dict[str, Any]]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _series(df: pd.DataFrame, names: Sequence[str], default: float | None = None) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _first_text(row: Mapping[str, Any], names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return default


def _normalize_high(values: pd.Series, neutral: float = 50.0) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(neutral, index=values.index, dtype="float64")
    low = float(values.min(skipna=True))
    high = float(values.max(skipna=True))
    if not math.isfinite(low) or not math.isfinite(high) or high == low:
        return pd.Series(neutral, index=values.index, dtype="float64")
    return ((values - low) / (high - low) * 100.0).clip(0, 100).fillna(neutral)


def _normalize_low(values: pd.Series, neutral: float = 50.0) -> pd.Series:
    return 100.0 - _normalize_high(values, neutral=100.0 - neutral)


def _bounded(values: pd.Series, neutral: float = 50.0) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(0, 100).fillna(neutral)


def _column_coverage(df: pd.DataFrame, columns: Sequence[str]) -> float:
    present = [column for column in columns if column in df.columns]
    if not present:
        return 0.0
    usable = 0
    for column in present:
        if pd.to_numeric(df[column], errors="coerce").notna().any() or df[column].notna().any():
            usable += 1
    return usable / max(1, len(columns))


def _variation_score(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    spread = float(clean.max() - clean.min())
    scale = max(abs(float(clean.mean())), 1.0)
    return max(0.0, min(1.0, spread / scale))


def _component_meta(
    df: pd.DataFrame,
    scores: Mapping[str, pd.Series],
    source_columns: Mapping[str, Sequence[str]],
    dqn_enabled: bool,
) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for component in COMPONENTS:
        columns = list(source_columns.get(component, ()))
        present = [column for column in columns if column in df.columns]
        if present and len(df):
            available = pd.DataFrame({
                column: df[column].notna() & df[column].astype(str).str.strip().ne("")
                for column in present
            }).any(axis=1)
            source_missing_rate = float((~available).mean())
        else:
            source_missing_rate = 1.0
        if component == "dqn_reference_score" and not dqn_enabled:
            coverage = 0.0
            used = False
            fallback_reason = "DQN 학습 결과가 없어 최종 VHS에서는 제외했습니다."
            imputation_strategy = "0점 처리 후 가중치 제외"
        else:
            coverage = _column_coverage(df, columns)
            used = coverage > 0 or component in {"feasibility_score", "greedy_score"}
            fallback_reason = "" if used else "입력 컬럼 부족으로 중립값을 사용했습니다."
            imputation_strategy = "사용 가능한 입력을 계산하고 결측 행은 중립값(50) 적용"
        score = scores[component]
        meta[component] = {
            "component": component,
            "used": used,
            "source_columns": present,
            "missing_columns": [column for column in columns if column not in df.columns],
            "coverage": round(float(coverage), 4),
            "missing_rate": round(source_missing_rate, 4),
            "imputation_strategy": imputation_strategy,
            "variance": round(float(pd.to_numeric(score, errors="coerce").var(ddof=0) or 0.0), 4),
            "variation": round(_variation_score(score), 4),
            "fallback_reason": fallback_reason,
        }
    return meta


def _dqn_is_ready(df: pd.DataFrame) -> bool:
    if "dqn_status" not in df.columns:
        return False
    statuses = {str(value).strip().lower() for value in df["dqn_status"].dropna().unique()}
    return bool(statuses & DQN_READY_STATUSES)


def _component_scores(df: pd.DataFrame, dqn_enabled: bool) -> tuple[dict[str, pd.Series], dict[str, Sequence[str]]]:
    source_columns: dict[str, Sequence[str]] = {
        "savings_score": ("expected_saving", "avoided_disposal_cost", "recovered_margin"),
        "disposal_risk_score": ("avoided_disposal_cost", "days_to_expiry", "expiry_days", "disposal_risk_score"),
        "demand_fit_score": ("demand_forecast_7d", "sales_30d", "sales_30", "recovered_margin", "demand_fit_score"),
        "inventory_balance_score": ("recommended_qty", "suggested_qty", "quantity_score", "inventory_balance_score"),
        "route_cost_score": ("estimated_cost", "move_cost", "transport_cost", "distance_km", "travel_time_min", "expected_time_min"),
        "feasibility_score": ("cutline_passed", "time_window_status", "cold_storage_available", "route_type"),
        "promotion_score": ("promotion_effect", "promotion_recommended", "promotion_net_cost", "promotion_transfer_cost"),
        "greedy_score": ("greedy_rank", "heuristic_score", "greedy_selected", "is_greedy_selected"),
        "confidence_score": ("confidence_score", "confidence", "confidence_level"),
        "dqn_reference_score": ("dqn_status", "dqn_action", "dqn_confidence"),
    }

    savings = _normalize_high(
        _series(df, ("expected_saving",), 0).fillna(0)
        + _series(df, ("avoided_disposal_cost",), 0).fillna(0) * 0.25
        + _series(df, ("recovered_margin",), 0).fillna(0) * 0.15
    )
    expiry = _series(df, ("days_to_expiry", "expiry_days"), None)
    disposal_value = _series(df, ("avoided_disposal_cost", "disposal_risk_score"), None)
    disposal = (
        _normalize_high(disposal_value, neutral=50) * 0.6
        + _normalize_low(expiry, neutral=50) * 0.4
    ).clip(0, 100)
    demand = _normalize_high(
        _series(df, ("demand_forecast_7d",), 0).fillna(0)
        + _series(df, ("sales_30d", "sales_30"), 0).fillna(0) * 0.12
        + _series(df, ("recovered_margin",), 0).fillna(0) * 0.08,
        neutral=50,
    )
    inventory_balance = _bounded(_series(df, ("quantity_score", "inventory_balance_score"), None), neutral=50)
    if inventory_balance.eq(50).all():
        inventory_balance = _normalize_high(_series(df, ("recommended_qty", "suggested_qty"), 0), neutral=50)

    route_cost_raw = (
        _series(df, ("estimated_cost", "move_cost", "transport_cost"), 0).fillna(0) * 0.60
        + _series(df, ("distance_km",), 0).fillna(0) * 1000.0 * 0.25
        + _series(df, ("travel_time_min", "expected_time_min"), 0).fillna(0) * 150.0 * 0.15
    )
    route_cost = _normalize_low(route_cost_raw, neutral=55)

    feasibility = pd.Series(78.0, index=df.index, dtype="float64")
    if "cutline_passed" in df.columns:
        text = df["cutline_passed"].astype(str)
        feasibility += text.str.contains("통과|가능|pass|ok|true", case=False, regex=True).map({True: 8, False: 0})
        feasibility -= text.str.contains("불가|실패|fail|false", case=False, regex=True).map({True: 20, False: 0})
    if "time_window_status" in df.columns:
        text = df["time_window_status"].astype(str)
        feasibility += text.str.contains("가능|통과|pass|ok|true", case=False, regex=True).map({True: 6, False: 0})
        feasibility -= text.str.contains("불가|실패|fail|false", case=False, regex=True).map({True: 16, False: 0})
    feasibility = feasibility.clip(0, 100)

    promotion_effect = _series(df, ("promotion_effect",), None)
    promotion = _normalize_high(promotion_effect, neutral=55)
    if "promotion_recommended" in df.columns:
        text = df["promotion_recommended"].astype(str)
        transfer_better = text.str.contains("재배치|이동|transfer|relocation", case=False, regex=True)
        promotion = promotion.where(~transfer_better, (promotion + 8).clip(0, 100))

    greedy_rank = _series(df, ("greedy_rank",), None)
    greedy = (110.0 - greedy_rank.fillna(6) * 10.0).clip(20, 100)
    if "heuristic_score" in df.columns:
        greedy = (_normalize_high(_series(df, ("heuristic_score",), None), neutral=50) * 0.7 + greedy * 0.3).clip(0, 100)
    selected = None
    for column in ("greedy_selected", "is_greedy_selected"):
        if column in df.columns:
            selected = df[column].fillna(False).astype(bool)
            break
    if selected is not None:
        greedy = (greedy + selected.map({True: 10, False: 0})).clip(0, 100)

    confidence = _bounded(_series(df, ("confidence_score", "confidence"), None), neutral=60)

    if dqn_enabled:
        dqn_conf = _bounded(_series(df, ("dqn_confidence",), None), neutral=50)
        dqn_match = []
        for _, row in df.iterrows():
            dqn_action = normalize_action(row.get("dqn_action"), default="")
            baseline = normalize_action(row.get("varo_action") or row.get("greedy_action"), default="")
            dqn_match.append(12.0 if dqn_action and dqn_action == baseline else -8.0)
        dqn_reference = (dqn_conf + pd.Series(dqn_match, index=df.index)).clip(0, 100)
    else:
        dqn_reference = pd.Series(0.0, index=df.index, dtype="float64")

    return {
        "savings_score": savings,
        "disposal_risk_score": disposal,
        "demand_fit_score": demand,
        "inventory_balance_score": inventory_balance,
        "route_cost_score": route_cost,
        "feasibility_score": feasibility,
        "promotion_score": promotion,
        "greedy_score": greedy,
        "confidence_score": confidence,
        "dqn_reference_score": dqn_reference,
    }, source_columns


def _project_to_bounds(raw: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    weights = {
        key: max(bounds[key][0], min(bounds[key][1], float(raw.get(key, 0.0))))
        for key in COMPONENTS
    }
    for _ in range(20):
        total = sum(weights.values())
        diff = 1.0 - total
        if abs(diff) < 1e-10:
            break
        if diff > 0:
            capacity = {key: bounds[key][1] - weights[key] for key in COMPONENTS}
        else:
            capacity = {key: weights[key] - bounds[key][0] for key in COMPONENTS}
        available = {key: value for key, value in capacity.items() if value > 1e-12}
        if not available:
            break
        capacity_sum = sum(available.values())
        for key, value in available.items():
            change = diff * (value / capacity_sum)
            weights[key] = max(bounds[key][0], min(bounds[key][1], weights[key] + change))
    total = sum(weights.values()) or 1.0
    return {key: round(weights[key] / total, 6) for key in COMPONENTS}


def optimize_weights(
    component_scores: Mapping[str, pd.Series],
    component_meta: Mapping[str, Mapping[str, Any]],
    dqn_enabled: bool = False,
) -> dict[str, float]:
    bounds = dict(WEIGHT_BOUNDS)
    if not dqn_enabled:
        bounds["dqn_reference_score"] = (0.0, 0.0)
    raw: dict[str, float] = {}
    for component in COMPONENTS:
        meta = component_meta[component]
        coverage = float(meta.get("coverage") or 0.0)
        variation = float(meta.get("variation") or 0.0)
        if component in {"savings_score", "feasibility_score"}:
            coverage = max(coverage, 0.85)
        if component == "greedy_score":
            coverage = max(coverage, 0.65)
        if component == "dqn_reference_score" and not dqn_enabled:
            raw[component] = 0.0
            continue
        signal = 0.70 + coverage * 0.20 + min(variation, 1.0) * 0.25
        if coverage == 0 and component not in {"savings_score", "feasibility_score", "greedy_score"}:
            signal = 0.45
        raw[component] = BASE_WEIGHTS[component] * signal
    return _project_to_bounds(raw, bounds)


def _grade(score: float) -> str:
    if score >= 80:
        return "최적"
    if score >= 65:
        return "권장"
    if score >= 50:
        return "검토"
    return "보류"


def _top_reason(row: Mapping[str, Any], weights: Mapping[str, float]) -> str:
    pieces = []
    for component in sorted(COMPONENTS, key=lambda key: weights.get(key, 0.0), reverse=True):
        if component == "dqn_reference_score" and weights.get(component, 0.0) == 0:
            continue
        score = _num(row.get(component))
        if score is not None:
            pieces.append((component, score, weights.get(component, 0.0)))
        if len(pieces) >= 3:
            break
    labels = {
        "savings_score": "절감액",
        "feasibility_score": "실현 가능성",
        "disposal_risk_score": "폐기 위험",
        "demand_fit_score": "수요 적합도",
        "inventory_balance_score": "재고 균형",
        "route_cost_score": "이동비용",
        "promotion_score": "프로모션 대안",
        "greedy_score": "Greedy 비교",
        "confidence_score": "신뢰도",
        "dqn_reference_score": "DQN 참고",
    }
    summary = ", ".join(f"{labels.get(name, name)} {score:.1f}" for name, score, _ in pieces)
    return f"{summary}을 종합해 VHS 기준 우선순위를 산정했습니다." if summary else "현재 입력 데이터 기준으로 VHS 우선순위를 산정했습니다."


def _ensure_greedy_rank(frame: pd.DataFrame) -> pd.Series:
    if "greedy_rank" in frame.columns:
        rank = pd.to_numeric(frame["greedy_rank"], errors="coerce")
        if rank.notna().any():
            return rank
    sort_values = _series(frame, ("heuristic_score",), None)
    if sort_values.notna().any():
        return sort_values.rank(method="first", ascending=False)
    saving = _series(frame, ("expected_saving",), 0).fillna(0)
    cost = _series(frame, ("estimated_cost", "move_cost"), 0).fillna(0)
    score = saving - cost
    return score.rank(method="first", ascending=False)


def pareto_ranks(recommendations: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return simple non-dominated layers used only as an auxiliary check."""
    items = list(recommendations or [])
    points = []
    for item in items:
        saving = _num(item.get("expected_saving")) or 0.0
        disposal_risk = _num(item.get("disposal_risk_score")) or 0.0
        demand_fit = _num(item.get("demand_fit_score")) or 0.0
        feasibility = _num(item.get("feasibility_score")) or 0.0
        route_cost = _num(item.get("move_cost") or item.get("estimated_cost")) or 0.0
        points.append((saving, disposal_risk, demand_fit, feasibility, -route_cost))

    remaining = set(range(len(items)))
    ranks = [0] * len(items)
    layer = 1
    while remaining:
        front: list[int] = []
        for candidate in remaining:
            dominated = False
            for other in remaining:
                if candidate == other:
                    continue
                at_least_as_good = all(left >= right for left, right in zip(points[other], points[candidate]))
                strictly_better = any(left > right for left, right in zip(points[other], points[candidate]))
                if at_least_as_good and strictly_better:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        if not front:
            front = [min(remaining)]
        for index in front:
            ranks[index] = layer
            remaining.remove(index)
        layer += 1
    return ranks


def build_strategy_comparison(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sorted_items = sorted(recommendations, key=lambda row: (_num(row.get("vhs_rank") or row.get("varo_final_rank") or row.get("rank")) or 999999))
    computed_ranks = pareto_ranks(sorted_items)
    for item, computed_pareto_rank in zip(sorted_items, computed_ranks):
        pareto_rank = int(_num(item.get("pareto_rank")) or computed_pareto_rank)
        pareto_status = _first_text(
            item, ("pareto_status",), "비지배 후보" if pareto_rank == 1 else "보조 후보",
        )
        pareto_reason = _first_text(
            item,
            ("pareto_reason",),
            "절감액·폐기 위험·수요·경로 비용·실행 가능성의 제한 탐색 비교",
        )
        dqn_status = _first_text(item, ("dqn_status",), "학습 필요")
        dqn_action = _first_text(item, ("dqn_action",), "비교 불가")
        if dqn_action in {"미연결", "학습 필요", "비교 불가"} or dqn_status not in {"연결", "정상", "connected", "ok", "ready"}:
            dqn_action = "비교 불가"
        dqn_reflection = "반영 가능" if dqn_action != "비교 불가" else "반영 안 함"
        vhs_rank = _num(item.get("vhs_rank") or item.get("varo_final_rank") or item.get("rank"))
        greedy_rank = _num(item.get("greedy_rank"))
        rows.append({
            "route_id": item.get("route_id") or item.get("recommendation_id") or "-",
            "상품명": item.get("product_name") or "-",
            "보내는 점포": item.get("source_name") or item.get("source_id") or "-",
            "받는 점포": item.get("target_name") or item.get("target_id") or "-",
            "추천 수량": item.get("recommended_qty"),
            "VHS 순위": int(vhs_rank) if vhs_rank is not None else "-",
            "VHS 점수": item.get("vhs_score"),
            "Greedy 순위": int(greedy_rank) if greedy_rank is not None else "-",
            "Greedy 전략": item.get("greedy_strategy") or item.get("greedy_action") or "비교 불가",
            "DQN 상태": dqn_status,
            "DQN 전략": dqn_action,
            "DQN 반영 여부": dqn_reflection,
            "DQN confidence": item.get("dqn_confidence"),
            "DQN 참고 점수": item.get("dqn_reference_score", 0),
            "Pareto rank": pareto_rank,
            "Pareto 상태": pareto_status,
            "Pareto 판단": pareto_reason,
            "Varo 최종 추천": item.get("varo_final_decision") or ("최종 추천" if vhs_rank == 1 else "후보"),
            "일치 여부": item.get("vhs_vs_greedy_match"),
            "판단 근거": item.get("final_reason") or item.get("reason") or "-",
        })
    return rows


def apply_auto_vhs(
    candidates: pd.DataFrame,
    training_result: Mapping[str, Any] | None = None,
) -> VhsAutoResult:
    if candidates is None or candidates.empty:
        return VhsAutoResult(pd.DataFrame(), {"weights": {}, "weight_rows": []}, [])

    frame = candidates.copy()
    dqn_status = str((training_result or {}).get("status") or "").strip().lower()
    dqn_enabled = dqn_status in DQN_READY_STATUSES or _dqn_is_ready(frame)
    scores, source_columns = _component_scores(frame, dqn_enabled=dqn_enabled)
    for component, values in scores.items():
        frame[component] = _bounded(values, neutral=50 if component != "dqn_reference_score" else 0)

    meta = _component_meta(frame, scores, source_columns, dqn_enabled=dqn_enabled)
    weights = optimize_weights(scores, meta, dqn_enabled=dqn_enabled)
    weighted = pd.Series(0.0, index=frame.index, dtype="float64")
    for component in COMPONENTS:
        weighted += pd.to_numeric(frame[component], errors="coerce").fillna(50) * weights[component]
    frame["auto_vhs_score"] = weighted.clip(0, 100).round(2)
    frame["vhs_score"] = frame["auto_vhs_score"]
    frame["recalculated_vhs_score"] = frame["auto_vhs_score"]
    frame["vhs_rank"] = frame["auto_vhs_score"].rank(method="first", ascending=False).astype(int)
    frame["varo_final_rank"] = frame["vhs_rank"]
    frame["rank"] = frame["vhs_rank"]
    frame["recommendation_grade"] = frame["auto_vhs_score"].apply(_grade)
    frame["grade"] = frame["recommendation_grade"]
    frame["vhs_score_source"] = "VHS 자동 가중치 최적화"
    frame["weight_profile_id"] = "auto_distribution_v1"
    frame["weight_summary"] = ", ".join(f"{key}:{weights[key]:.3f}" for key in COMPONENTS if weights[key] > 0)
    frame["greedy_rank"] = _ensure_greedy_rank(frame)
    frame["greedy_strategy"] = frame.get("greedy_action", pd.Series("재고 이동", index=frame.index)).apply(normalize_action)
    if "greedy_action" not in frame.columns:
        frame["greedy_action"] = frame["greedy_strategy"]
    frame["varo_final_decision"] = frame["vhs_rank"].map(lambda value: "최종 추천" if int(value) == 1 else "후보")
    frame["vhs_vs_greedy_match"] = frame["vhs_rank"].astype(int) == frame["greedy_rank"].fillna(999999).astype(float).astype(int)
    frame["pareto_rank"] = pareto_ranks(frame.where(pd.notna(frame), None).to_dict("records"))
    frame["pareto_status"] = frame["pareto_rank"].map(
        lambda value: "비지배 후보" if int(value) == 1 else "보조 후보"
    )
    frame["pareto_reason"] = (
        "절감액·폐기 위험·수요·경로 비용·실행 가능성의 제한 탐색 비교"
    )
    if "dqn_action" not in frame.columns:
        frame["dqn_action"] = "미연결"
    if "dqn_status" not in frame.columns:
        frame["dqn_status"] = "학습 필요"
    frame["vhs_vs_dqn_match"] = [
        normalize_action(row.get("dqn_action"), default="") == normalize_action(row.get("varo_action") or row.get("greedy_action"), default="")
        if str(row.get("dqn_status") or "") in {"연결", "정상"} else False
        for _, row in frame.iterrows()
    ]
    frame["final_reason"] = [_top_reason(row, weights) for _, row in frame.iterrows()]

    weight_rows = []
    for component in COMPONENTS:
        low, high = WEIGHT_BOUNDS[component]
        if component == "dqn_reference_score" and not dqn_enabled:
            high = 0.0
        item = dict(meta[component])
        item.update({
            "weight": weights[component],
            "min_weight": low,
            "max_weight": high,
            "average_score": round(float(pd.to_numeric(frame[component], errors="coerce").mean()), 2),
        })
        weight_rows.append(item)

    analysis = {
        "calculation_function": "services.vhs_score_engine.apply_auto_vhs",
        "weight_profile_id": "auto_distribution_v1",
        "weights": weights,
        "weight_rows": weight_rows,
        "component_columns": list(COMPONENTS),
        "dqn_included": dqn_enabled,
        "dqn_policy": "DQN 정상 학습/추론 결과가 있을 때만 낮은 비중으로 반영",
        "final_top_route_id": str(frame.sort_values("vhs_rank").iloc[0].get("route_id")),
        "vhs_average": round(float(frame["auto_vhs_score"].mean()), 3),
        "recalculated_average": round(float(frame["auto_vhs_score"].mean()), 3),
        "score_basis": "VHS 자동 가중치 최적화",
        "fallback_components": [
            row["component"] for row in weight_rows if row.get("fallback_reason")
        ],
    }
    return VhsAutoResult(frame, analysis, build_strategy_comparison(frame.where(pd.notna(frame), None).to_dict("records")))
