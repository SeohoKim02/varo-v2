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

from services.decision_metrics import ALGORITHM_VERSION
from services.dqn_service import dqn_display_status
from services.pareto_analysis import compute_pareto, pareto_summary
from services.recommendation_adapter import normalize_action

# Each component answers one distinct decision question. Components that made the
# score circular were removed: the Greedy baseline cannot be part of the score it
# is a baseline for, and confidence is derived *from* VHS downstream. The
# promotion comparison is a different action, not a property of this transfer.
COMPONENTS = (
    "net_benefit_score",        # 이 이동이 비용을 빼고도 남는가
    "inventory_balance_score",  # 과잉을 줄이고 부족을 채우는가
    "disposal_risk_score",      # 폐기·노후재고 위험을 줄이는가
    "demand_fit_score",         # 도착 점포가 실제로 필요로 하는가
    "route_cost_score",         # 이동 부담(비용·거리·시간)이 큰가
    "feasibility_score",        # 실행 여건(거리 기준·시간창)이 맞는가
    "demand_risk_score",        # 수요가 흔들려도 유효한가
    "post_move_risk_score",     # 이동 후 새로운 문제가 생기지 않는가
    "dqn_reference_score",      # 선택형 참고 (학습 결과가 있을 때만)
)

COMPONENT_LABELS: dict[str, str] = {
    "net_benefit_score": "예상 순효과",
    "inventory_balance_score": "재고 균형",
    "disposal_risk_score": "폐기 위험",
    "demand_fit_score": "수요 적합도",
    "route_cost_score": "이동 부담",
    "feasibility_score": "실행 여건",
    "demand_risk_score": "수요 안정성",
    "post_move_risk_score": "이동 후 위험",
    "dqn_reference_score": "DQN 참고",
}

WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "net_benefit_score": (0.20, 0.38),
    "inventory_balance_score": (0.10, 0.22),
    "disposal_risk_score": (0.08, 0.20),
    "demand_fit_score": (0.08, 0.20),
    "route_cost_score": (0.06, 0.16),
    "feasibility_score": (0.08, 0.20),
    "demand_risk_score": (0.04, 0.14),
    "post_move_risk_score": (0.04, 0.14),
    "dqn_reference_score": (0.00, 0.08),
}

BASE_WEIGHTS: dict[str, float] = {
    "net_benefit_score": 0.26,
    "inventory_balance_score": 0.14,
    "disposal_risk_score": 0.12,
    "demand_fit_score": 0.12,
    "route_cost_score": 0.10,
    "feasibility_score": 0.12,
    "demand_risk_score": 0.07,
    "post_move_risk_score": 0.07,
    "dqn_reference_score": 0.00,
}

DQN_READY_STATUSES = {"연결", "정상", "connected", "ok", "ready"}

# Components that are always meaningful because they are derived from columns the
# validator already guarantees (net benefit from 절감액/비용, 실행 여건 from the
# route type). The rest fall back to a neutral 50 and a reduced weight when the
# file does not carry their inputs.
_ALWAYS_ACTIVE = frozenset({"net_benefit_score", "feasibility_score"})

# Values below the 5th / above the 95th percentile are pulled to those bounds
# before scaling. Min-max on raw values lets one outlier compress every other
# candidate into a few points; winsorizing keeps the spread meaningful while
# staying deterministic and order-preserving.
WINSOR_LOW, WINSOR_HIGH = 0.05, 0.95

# Deterministic ordering applied after the weighted score. Documented in
# docs/ALGORITHM.md; every key is a real, computed number so the same workbook
# always produces the same ranking regardless of row order.
TIE_BREAK_KEYS: tuple[tuple[str, bool], ...] = (
    ("auto_vhs_score", False),   # 높을수록 우선
    ("net_benefit", False),      # 순효과가 큰 쪽
    ("_route_cost_sort", True),  # 이동 비용이 낮은 쪽
    ("_travel_time_sort", True),  # 이동 시간이 짧은 쪽
    ("_route_simplicity", True),  # DIRECT(0) < VIA_DC(1)
    ("_route_key", True),        # 마지막 안정 키 (route_id)
)


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
    """Winsorized min-max onto 0..100, higher input → higher score.

    ``±inf`` and non-numeric entries become NaN first, so they can never act as
    the scale bound. Candidates outside the 5–95% band are clipped to it rather
    than dropped, which keeps their direction while stopping one extreme value
    from flattening everyone else. A degenerate spread yields the neutral value,
    never a fabricated 0 or 100.
    """
    values = pd.to_numeric(values, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(neutral, index=values.index, dtype="float64")
    low = float(values.quantile(WINSOR_LOW))
    high = float(values.quantile(WINSOR_HIGH))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low, high = float(values.min(skipna=True)), float(values.max(skipna=True))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return pd.Series(neutral, index=values.index, dtype="float64")
    scaled = (values.clip(low, high) - low) / (high - low) * 100.0
    return scaled.clip(0, 100).fillna(neutral)


def _normalize_low(values: pd.Series, neutral: float = 50.0) -> pd.Series:
    """Same scale, inverted: lower input → higher score."""
    return 100.0 - _normalize_high(values, neutral=100.0 - neutral)


def _coverage_score(covered: pd.Series, target: pd.Series, neutral: float = 50.0) -> pd.Series:
    """How much of ``target`` this candidate covers, as 0..100, capped at full.

    An absolute fraction, so it needs no cross-candidate normalization and is
    immune to outliers by construction. Capping at 100 is the point: sending more
    than the destination is short by is not a *better* fit, and the risk of
    over-sending is scored separately by ``post_move_risk_score``.
    """
    covered = pd.to_numeric(covered, errors="coerce")
    target = pd.to_numeric(target, errors="coerce")
    safe = target.where(target > 1e-9)
    return ((covered / safe).clip(0, 1) * 100.0).fillna(neutral)


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
            used = coverage > 0 or component in _ALWAYS_ACTIVE
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
        "net_benefit_score": ("net_benefit", "expected_saving", "estimated_cost"),
        "inventory_balance_score": ("source_movable", "target_shortfall", "recommended_qty", "dead_stock_qty"),
        "disposal_risk_score": ("days_to_expiry", "expiry_days", "dead_stock_qty", "disposal_cost_per_unit"),
        "demand_fit_score": ("target_demand", "target_shortfall", "recommended_qty"),
        "route_cost_score": ("estimated_cost", "move_cost", "transport_cost", "distance_km", "travel_time_min", "expected_time_min"),
        "feasibility_score": ("cutline_passed", "time_window_status", "cold_storage_available", "route_type"),
        "demand_risk_score": ("target_demand_std", "target_demand", "demand_scenario_status"),
        "post_move_risk_score": ("post_move_source_gap", "post_move_target_excess", "source_safety_floor"),
        "dqn_reference_score": ("dqn_status", "dqn_action", "dqn_confidence"),
    }
    quantity = _series(df, ("recommended_qty", "suggested_qty"), None)

    # 1) 순효과 — 절감액이 아니라 "비용을 빼고 남는 금액". 비용은 여기서 이미
    #    반영되므로 route_cost_score는 금액이 아닌 이동 부담(거리·시간)을 본다.
    benefit = _series(df, ("net_benefit",), None)
    if benefit.notna().sum() == 0:
        benefit = _series(df, ("expected_saving",), None) - _series(df, ("estimated_cost", "move_cost"), None)
    net_benefit_score = _normalize_high(benefit, neutral=50)

    # 2) 재고 균형 — 출발지의 이동 가능한 여유 재고 중 이 이동이 실제로 걷어내는
    #    비율. 1.0(전량 해소)에서 상한을 두므로 필요 이상으로 많이 보내도 점수가
    #    더 오르지 않는다. 과다 이동의 위험은 post_move_risk_score가 따로 본다.
    movable = _series(df, ("source_movable",), None)
    shortfall = _series(df, ("target_shortfall",), None)
    inventory_balance = _coverage_score(quantity, movable, neutral=50)
    if movable.notna().sum() == 0:
        inventory_balance = _bounded(_series(df, ("quantity_score",), None), neutral=50)

    # 3) 폐기 위험 — 유통기한이 임박할수록, 악성재고가 많을수록 이동 가치가 크다.
    expiry = _series(df, ("days_to_expiry", "expiry_days"), None)
    dead_stock = _series(df, ("dead_stock_qty",), None)
    disposal = (
        _normalize_low(expiry, neutral=50) * 0.6 + _normalize_high(dead_stock, neutral=50) * 0.4
    ).clip(0, 100)

    # 4) 수요 적합도 — 도착 점포의 부족량을 얼마나 채우는가(최대 100%). 부족량을
    #    넘겨 보내는 것은 적합도를 더 높이지 않는다.
    demand = _coverage_score(quantity, shortfall.where(shortfall > 0), neutral=50)
    if shortfall.notna().sum() == 0:
        demand = _normalize_high(_series(df, ("target_demand", "demand_forecast_7d"), None), neutral=50)

    # 5) 이동 부담 — 거리·시간 중심(금액은 순효과에서 이미 반영). 두 축을 각각
    #    정규화해 단위가 섞이지 않게 한다.
    distance_load = _normalize_low(_series(df, ("distance_km",), None), neutral=55)
    time_load = _normalize_low(_series(df, ("travel_time_min", "expected_time_min"), None), neutral=55)
    cost_load = _normalize_low(_series(df, ("estimated_cost", "move_cost", "transport_cost"), None), neutral=55)
    route_cost = (distance_load * 0.35 + time_load * 0.35 + cost_load * 0.30).clip(0, 100)

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

    # 7) 수요 안정성 — 도착 수요 대비 변동폭이 작을수록 높다. 표준편차 컬럼이
    #    없으면 만들어내지 않고 중립값 50을 쓴다.
    demand_std = _series(df, ("target_demand_std",), None)
    demand_base = _series(df, ("target_demand",), None)
    variability = demand_std / demand_base.where(demand_base.abs() > 1e-9)
    demand_risk = _normalize_low(variability, neutral=50)
    if "demand_scenario_status" in df.columns:
        status = df["demand_scenario_status"].astype(str)
        demand_risk = demand_risk.where(~status.eq("변동 가능성 큼"), (demand_risk - 20).clip(0, 100))
        demand_risk = demand_risk.where(~status.eq("안정"), (demand_risk + 10).clip(0, 100))
    demand_risk = demand_risk.clip(0, 100)

    # 8) 이동 후 위험 — 출발지가 안전재고 아래로 내려가거나 도착지가 과잉이 되는
    #    정도. 두 위험 모두 0이면 100, 클수록 낮아진다.
    source_gap = _series(df, ("post_move_source_gap",), None)
    target_excess = _series(df, ("post_move_target_excess",), None)
    post_move = (
        _normalize_low(source_gap, neutral=50) * 0.5 + _normalize_low(target_excess, neutral=50) * 0.5
    ).clip(0, 100)
    if source_gap.notna().sum() == 0 and target_excess.notna().sum() == 0:
        post_move = pd.Series(50.0, index=df.index, dtype="float64")

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
        "net_benefit_score": net_benefit_score,
        "inventory_balance_score": inventory_balance,
        "disposal_risk_score": disposal,
        "demand_fit_score": demand,
        "route_cost_score": route_cost,
        "feasibility_score": feasibility,
        "demand_risk_score": demand_risk,
        "post_move_risk_score": post_move,
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
        if component in _ALWAYS_ACTIVE:
            coverage = max(coverage, 0.85)
        if component == "dqn_reference_score" and not dqn_enabled:
            raw[component] = 0.0
            continue
        signal = 0.70 + coverage * 0.20 + min(variation, 1.0) * 0.25
        if coverage == 0 and component not in _ALWAYS_ACTIVE:
            signal = 0.45
        raw[component] = BASE_WEIGHTS[component] * signal
    return _project_to_bounds(raw, bounds)


def tie_break_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the sort keys used to break equal VHS scores (never left to row order)."""
    result = frame.copy()
    result["_route_cost_sort"] = _series(result, ("estimated_cost", "move_cost", "transport_cost"), None).fillna(1e18)
    result["_travel_time_sort"] = _series(result, ("travel_time_min", "expected_time_min"), None).fillna(1e18)
    route_type = result.get("route_type", pd.Series("", index=result.index)).astype(str).str.upper()
    result["_route_simplicity"] = route_type.eq("VIA_DC").astype(int)
    result["_route_key"] = result.get("route_id", pd.Series("", index=result.index)).astype(str)
    if "net_benefit" not in result.columns:
        result["net_benefit"] = (
            _series(result, ("expected_saving",), None) - _series(result, ("estimated_cost", "move_cost"), None)
        )
    result["net_benefit"] = pd.to_numeric(result["net_benefit"], errors="coerce").fillna(-1e18)
    return result


def _ranked(frame: pd.DataFrame) -> pd.Series:
    """1-based rank from the score plus the documented tie-break chain."""
    keyed = tie_break_frame(frame)
    columns = [name for name, _ascending in TIE_BREAK_KEYS]
    ascending = [asc for _name, asc in TIE_BREAK_KEYS]
    order = keyed.sort_values(by=columns, ascending=ascending, kind="mergesort").index
    ranks = pd.Series(range(1, len(order) + 1), index=order, dtype=int)
    return ranks.reindex(frame.index)


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


ROBUST = "안정"
ROBUST_REVIEW = "검토 필요"
ROBUST_FRAGILE = "변동 가능성 큼"
ROBUST_UNKNOWN = "계산 불가"


def _robustness_status(base_rank: int, best: int, worst: int, top1_rate: float, count: int) -> str:
    """Classify one candidate from how far its rank moved under perturbation.

    Thresholds scale with the candidate count so "moved 3 places" means something
    different in a 5-candidate and a 60-candidate set. A candidate that is ranked
    first but only holds that rank in some scenarios can never be 안정.
    """
    shift = worst - best
    tight = max(1, round(0.05 * count))
    loose = max(3, round(0.15 * count))
    if base_rank == 1 and top1_rate < 0.8:
        return ROBUST_REVIEW if shift <= loose else ROBUST_FRAGILE
    if shift <= tight:
        return ROBUST
    if shift <= loose:
        return ROBUST_REVIEW
    return ROBUST_FRAGILE


def candidate_robustness(
    recommendations: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    delta: float = 0.30,
) -> dict[str, dict[str, Any]]:
    """Per-candidate rank stability under the same ±delta weight perturbations.

    Returns, keyed by route_id: base rank, Top-1 / Top-3 retention, mean / best /
    worst rank, and a plain status. Uses the component scores the candidates
    already carry, so it adds no algorithm calls and stays deterministic.
    """
    recs = [dict(row) for row in recommendations or []]
    active = [component for component in COMPONENTS if float(weights.get(component, 0.0)) > 0]
    if len(recs) < 2 or not active:
        return {}
    base_ranks = _ranks_from_scores(_weighted_scores(recs, weights))
    observed: list[list[int]] = [[] for _ in recs]
    for component in active:
        for factor in (1.0 + delta, max(0.0, 1.0 - delta)):
            perturbed = dict(weights)
            perturbed[component] = float(weights.get(component, 0.0)) * factor
            total = sum(perturbed.values()) or 1.0
            perturbed = {key: value / total for key, value in perturbed.items()}
            ranks = _ranks_from_scores(_weighted_scores(recs, perturbed))
            for index, rank in enumerate(ranks):
                observed[index].append(rank)

    result: dict[str, dict[str, Any]] = {}
    count = len(recs)
    for index, rec in enumerate(recs):
        ranks = observed[index] or [base_ranks[index]]
        best, worst = min(ranks), max(ranks)
        top1_rate = sum(1 for rank in ranks if rank == 1) / len(ranks)
        top3_rate = sum(1 for rank in ranks if rank <= 3) / len(ranks)
        status = _robustness_status(base_ranks[index], best, worst, top1_rate, count)
        result[str(rec.get("route_id"))] = {
            "base_rank": base_ranks[index],
            "top1_retention": round(top1_rate, 4),
            "top3_retention": round(top3_rate, 4),
            "mean_rank": round(sum(ranks) / len(ranks), 2),
            "best_rank": best,
            "worst_rank": worst,
            "rank_shift": worst - best,
            "scenarios": len(ranks),
            "status": status,
            "detail": f"가중치 변화 {len(ranks)}회 중 순위 {best}~{worst}위",
        }
    return result


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
        "candidate_count": len(recs),
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
    frame["vhs_rank"] = _ranked(frame)
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

    # Per-candidate robustness: the score decides the order, this says how much to
    # trust that order for each candidate. It never reorders anything.
    robustness = candidate_robustness(records, weights)
    route_ids = frame.get("route_id", pd.Series(index=frame.index, dtype=object)).astype(str)
    frame["robustness_status"] = route_ids.map(
        lambda key: (robustness.get(key) or {}).get("status", ROBUST_UNKNOWN)
    )
    frame["robustness_detail"] = route_ids.map(
        lambda key: (robustness.get(key) or {}).get("detail", "")
    )
    frame["algorithm_version"] = ALGORITHM_VERSION
    analysis["candidate_robustness"] = robustness
    analysis["algorithm_version"] = ALGORITHM_VERSION
    records = frame.where(pd.notna(frame), None).to_dict("records")
    return VhsAutoResult(frame, analysis, build_strategy_comparison(records))
