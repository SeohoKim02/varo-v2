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

from services.dqn_service import dqn_display_status
from services.pareto_analysis import compute_pareto, pareto_summary
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

COMPONENT_LABELS: dict[str, str] = {
    "savings_score": "절감액",
    "feasibility_score": "실행 가능성",
    "disposal_risk_score": "폐기 위험",
    "demand_fit_score": "수요 적합도",
    "inventory_balance_score": "재고 균형",
    "route_cost_score": "이동비용",
    "promotion_score": "프로모션 대안",
    "greedy_score": "Greedy 비교",
    "confidence_score": "신뢰도",
    "dqn_reference_score": "DQN 참고",
}

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
        if component == "dqn_reference_score" and not dqn_enabled:
            coverage = 0.0
            used = False
            fallback_reason = "DQN 학습 결과가 없어 최종 VHS에서는 제외했습니다."
        else:
            coverage = _column_coverage(df, columns)
            used = coverage > 0 or component in {"feasibility_score", "greedy_score"}
            fallback_reason = "" if used else "입력 컬럼 부족으로 중립값을 사용했습니다."
        score = scores[component]
        meta[component] = {
            "component": component,
            "used": used,
            "source_columns": present,
            "missing_columns": [column for column in columns if column not in df.columns],
            "coverage": round(float(coverage), 4),
            "missing_rate": round(float(pd.to_numeric(score, errors="coerce").isna().mean()), 4),
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
    summary = ", ".join(f"{COMPONENT_LABELS.get(name, name)} {score:.1f}" for name, score, _ in pieces)
    return f"{summary}을 종합해 VHS 기준 우선순위를 산정했습니다." if summary else "현재 입력 데이터 기준으로 VHS 우선순위를 산정했습니다."


def _ensure_greedy_rank(frame: pd.DataFrame) -> pd.Series:
    if "greedy_rank" in frame.columns:
        rank = pd.to_numeric(frame["greedy_rank"], errors="coerce")
        if rank.notna().any():
            return rank
    sort_values = _series(frame, ("heuristic_score",), None)
    if sort_values.notna().any():
        return sort_values.rank(method="first", ascending=False)
    saving = pd.to_numeric(frame.get("expected_saving"), errors="coerce").fillna(0)
    cost = pd.to_numeric(frame.get("estimated_cost"), errors="coerce").fillna(0)
    score = saving - cost
    return score.rank(method="first", ascending=False)


def _weighted_scores(recs: Sequence[Mapping[str, Any]], weights: Mapping[str, float]) -> list[float]:
    scores: list[float] = []
    for row in recs:
        total = 0.0
        for component in COMPONENTS:
            value = _num(row.get(component))
            total += (value if value is not None else 50.0) * float(weights.get(component, 0.0))
        scores.append(total)
    return scores


def _ranks_from_scores(scores: Sequence[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(scores)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def weight_sensitivity(
    recommendations: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    delta: float = 0.30,
) -> dict[str, Any]:
    """Perturb each active weight by ±delta (renormalized) and measure Top1 stability.

    Deterministic: recomputes the weighted VHS score from the component scores the
    candidates already carry. Reports Top1 retention rate, average rank movement,
    and the components whose perturbation flips the Top1 recommendation (취약 요소).
    """
    recs = [dict(row) for row in recommendations or []]
    active = [component for component in COMPONENTS if float(weights.get(component, 0.0)) > 0]
    empty = {
        "scenarios": 0,
        "top1_retention_rate": 1.0,
        "rank_volatility": 0.0,
        "fragile_components": [],
        "delta": delta,
        "rows": [],
        "basis": "VHS 가중치 ±30% 섭동 후 Top1 유지·순위 변동 측정 (연구 확장용 검증)",
    }
    if len(recs) < 2 or not active:
        return empty

    base_scores = _weighted_scores(recs, weights)
    base_ranks = _ranks_from_scores(base_scores)
    base_top1 = base_ranks.index(1)

    scenarios = 0
    kept = 0
    rank_changes: list[float] = []
    fragile: list[str] = []
    rows: list[dict[str, Any]] = []
    for component in active:
        holds = 0
        component_changes: list[float] = []
        for factor in (1.0 + delta, max(0.0, 1.0 - delta)):
            perturbed = dict(weights)
            perturbed[component] = float(weights.get(component, 0.0)) * factor
            total = sum(perturbed.values()) or 1.0
            perturbed = {key: value / total for key, value in perturbed.items()}
            scores = _weighted_scores(recs, perturbed)
            ranks = _ranks_from_scores(scores)
            same_top1 = ranks.index(1) == base_top1
            change = sum(abs(ranks[i] - base_ranks[i]) for i in range(len(recs))) / len(recs)
            scenarios += 1
            kept += 1 if same_top1 else 0
            holds += 1 if same_top1 else 0
            rank_changes.append(change)
            component_changes.append(change)
        if holds < 2:
            fragile.append(component)
        rows.append({
            "요소": COMPONENT_LABELS.get(component, component),
            "가중치": round(float(weights.get(component, 0.0)), 3),
            "Top1 유지": {2: "유지", 1: "부분", 0: "변동"}[holds],
            "평균 순위 변동": round(sum(component_changes) / len(component_changes), 2),
        })
    return {
        "scenarios": scenarios,
        "top1_retention_rate": round(kept / scenarios, 4) if scenarios else 1.0,
        "rank_volatility": round(sum(rank_changes) / len(rank_changes), 3) if rank_changes else 0.0,
        "fragile_components": [COMPONENT_LABELS.get(component, component) for component in fragile],
        "delta": delta,
        "rows": rows,
        "basis": "VHS 가중치 ±30% 섭동 후 Top1 유지·순위 변동 측정 (연구 확장용 검증)",
    }


def build_strategy_comparison(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(recommendations, key=lambda row: (_num(row.get("vhs_rank") or row.get("varo_final_rank") or row.get("rank")) or 999999)):
        dqn_status = _first_text(item, ("dqn_status",), "학습 필요")
        dqn_action = _first_text(item, ("dqn_action",), "비교 불가")
        if dqn_action in {"미연결", "학습 필요", "비교 불가"} or dqn_status not in {"연결", "정상", "connected", "ok", "ready"}:
            dqn_action = "비교 불가"
        vhs_rank = _num(item.get("vhs_rank") or item.get("varo_final_rank") or item.get("rank"))
        greedy_rank = _num(item.get("greedy_rank"))
        pareto_rank = _num(item.get("pareto_rank"))
        dqn_comparable = dqn_action != "비교 불가"
        reference_score = _num(item.get("dqn_reference_score")) or 0.0
        # Reflection gate uses the RAW status; the on-screen label is softened after.
        dqn_reflected = (
            dqn_status in {"정상", "연결"} and dqn_comparable and reference_score > 0
        )
        confidence = item.get("confidence_score")
        if confidence is None:
            confidence = item.get("confidence")
        rows.append({
            "route_id": item.get("route_id") or "-",
            "상품명": item.get("product_name") or "-",
            "보내는 점포": item.get("source_name") or item.get("source_id") or "-",
            "받는 점포": item.get("target_name") or item.get("target_id") or "-",
            "경로 방식": str(item.get("route_type") or "-").upper(),
            "추천 수량": item.get("recommended_qty"),
            "예상 절감액": item.get("expected_saving"),
            "추천 등급": item.get("recommendation_grade") or item.get("grade") or "-",
            "VHS 순위": int(vhs_rank) if vhs_rank is not None else "-",
            "VHS 점수": item.get("vhs_score"),
            "Greedy 순위": int(greedy_rank) if greedy_rank is not None else "-",
            "Greedy 전략": item.get("greedy_strategy") or item.get("greedy_action") or "비교 불가",
            "DQN 상태": dqn_display_status(dqn_status),
            "DQN 전략": dqn_action,
            "DQN confidence": item.get("dqn_confidence"),
            "DQN 참고 점수": item.get("dqn_reference_score", 0),
            "DQN 비교 가능": dqn_comparable,
            "DQN 반영 여부": "반영" if dqn_reflected else "미반영",
            "Pareto": item.get("pareto_status") or "-",
            "Pareto 순위": int(pareto_rank) if pareto_rank is not None else "-",
            "신뢰도": confidence,
            "Varo 최종 추천": item.get("varo_final_decision") or ("최종 추천" if vhs_rank == 1 else "후보"),
            "일치 여부": item.get("vhs_vs_greedy_match"),
            "판단 근거": item.get("final_reason") or item.get("reason") or "-",
        })
    return rows


# Core = compact default view rendered as an HTML table (all columns in the DOM).
# Friendly display headers map onto internal build_strategy_comparison keys; the
# full DataFrame keeps every internal column for calculations/tests.
_CORE_COLUMN_MAP = (
    ("순위", "VHS 순위"),
    ("상품", "상품명"),
    ("출발", "보내는 점포"),
    ("도착", "받는 점포"),
    ("경로", "경로 방식"),
    ("수량", "추천 수량"),
    ("예상 절감액", "예상 절감액"),
    ("추천 등급", "추천 등급"),
    ("DQN 상태", "DQN 상태"),
    ("최종 판단", "Varo 최종 추천"),
)
STRATEGY_CORE_COLUMNS = tuple(display for display, _ in _CORE_COLUMN_MAP)

# Detail expander keeps every column, with developer-ish keys renamed and the
# redundant 'DQN 비교 가능' dropped.
_DETAIL_FRIENDLY_NAMES = {
    "route_id": "추천 ID",
    "DQN confidence": "DQN 신뢰도",
    "Pareto": "Pareto 상태",
}
_DETAIL_HIDDEN = ("DQN 비교 가능",)


def _fmt_won(value: Any) -> str:
    number = _num(value)
    return f"{number:,.0f}원" if number is not None else "-"


def _fmt_qty(value: Any) -> str:
    number = _num(value)
    return f"{int(number):,}개" if number is not None else "-"


def build_strategy_core(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact default comparison (10 friendly columns, display-ready values)."""
    core: list[dict[str, Any]] = []
    for row in build_strategy_comparison(recommendations):
        item = {display: row.get(source) for display, source in _CORE_COLUMN_MAP}
        item["예상 절감액"] = _fmt_won(item["예상 절감액"])
        item["수량"] = _fmt_qty(item["수량"])
        core.append(item)
    return core


def build_strategy_detail(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Full detail table for the '상세 비교 보기' expander (friendly column names)."""
    detail: list[dict[str, Any]] = []
    for row in build_strategy_comparison(recommendations):
        detail.append({
            _DETAIL_FRIENDLY_NAMES.get(key, key): value
            for key, value in row.items()
            if key not in _DETAIL_HIDDEN
        })
    return detail


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

    pareto_rows = compute_pareto(frame.where(pd.notna(frame), None).to_dict("records"))
    frame["pareto_rank"] = [row["pareto_rank"] for row in pareto_rows]
    frame["pareto_status"] = [row["pareto_status"] for row in pareto_rows]

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
    records = frame.where(pd.notna(frame), None).to_dict("records")
    analysis["pareto"] = pareto_summary(pareto_rows)
    analysis["weight_sensitivity"] = weight_sensitivity(records, weights)
    return VhsAutoResult(frame, analysis, build_strategy_comparison(records))
