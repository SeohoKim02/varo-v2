"""Recommendation-set stability and confidence status for Varo V2.

Two small, pure summaries that turn existing pipeline outputs into a single,
plain status a user can read at a glance — without exposing weights, formulas,
or DQN internals:

* ``recommendation_stability`` — 안정 / 검토 필요 / 불안정 / 계산 불가, from the VHS
  weight-sensitivity result (Top1 retention, rank volatility, fragile factors).
* ``recommendation_confidence`` — 높음 / 보통 / 낮음 / 계산 불가, from factors that
  are always computable WITHOUT DQN (data completeness, feasibility pass rate,
  Top1 score gap, rank stability, Greedy agreement, Pareto membership). DQN, when
  it is in a normal state, can only *raise* confidence a little; its absence or
  instability never lowers it, because DQN is an optional reference.

Both are deterministic and never fabricate a number: when there is not enough
data to judge, the status is 계산 불가 and the score is None.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from services.feasibility import STATUS_OK as FEASIBLE_OK

STABLE = "안정"
REVIEW = "검토 필요"
UNSTABLE = "불안정"
NOT_COMPUTABLE = "계산 불가"

CONF_HIGH = "높음"
CONF_MEDIUM = "보통"
CONF_LOW = "낮음"
CONF_NONE = "계산 불가"

DQN_READY_STATUSES = {"정상", "연결", "connected", "ok", "ready"}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def recommendation_stability(weight_sensitivity: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize VHS Top1/rank stability into one status.

    Reads the existing ``weight_sensitivity`` result (retention rate, rank
    volatility, fragile components). Fewer than 2 real scenarios → 계산 불가.
    """
    ws = weight_sensitivity or {}
    scenarios = int(ws.get("scenarios") or 0)
    retention = _num(ws.get("top1_retention_rate"))
    volatility = _num(ws.get("rank_volatility"))
    fragile = list(ws.get("fragile_components") or [])

    if scenarios <= 0 or retention is None:
        return {
            "status": NOT_COMPUTABLE,
            "top1_retention_rate": retention,
            "rank_volatility": volatility,
            "fragile_components": fragile,
            "reasons": ["후보나 시나리오가 부족해 안정성을 판단할 수 없습니다."],
        }

    # Mean rank movement grows with the candidate count: in a 60-candidate set the
    # tail always shuffles a little, and that says nothing about the recommendation
    # the user is being shown. The tolerance therefore scales with the set size,
    # while Top-1 retention stays the primary signal.
    candidates = int(ws.get("candidate_count") or 0)
    tolerance = max(0.5, 0.05 * candidates)

    reasons: list[str] = []
    if retention >= 0.999 and (volatility is None or volatility <= tolerance):
        status = STABLE
        reasons.append("가중치를 조정해도 1순위 추천이 유지됩니다.")
    elif retention >= 0.80:
        status = REVIEW
        reasons.append("일부 가중치 변화에서 상위 추천 순위가 바뀔 수 있습니다.")
    else:
        status = UNSTABLE
        reasons.append("작은 가중치 변화에도 1순위 추천이 자주 바뀝니다.")
    if fragile:
        reasons.append(f"{', '.join(fragile[:3])} 비중 변화에 민감합니다.")

    return {
        "status": status,
        "top1_retention_rate": round(retention, 4),
        "rank_volatility": round(volatility, 3) if volatility is not None else None,
        "fragile_components": fragile,
        "reasons": reasons[:3],
    }


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def _completeness(recs: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of key decision fields that are present and finite across recs."""
    fields = ("recommended_qty", "expected_saving", "estimated_cost", "route_type", "vhs_score")
    if not recs:
        return 0.0
    total = 0
    present = 0
    for rec in recs:
        for field_name in fields:
            total += 1
            value = rec.get(field_name)
            if field_name == "route_type":
                if str(value or "").upper() in {"DIRECT", "VIA_DC"}:
                    present += 1
            elif _num(value) is not None:
                present += 1
    return present / total if total else 0.0


def _feasible_pass_rate(recs: Sequence[Mapping[str, Any]]) -> float | None:
    labelled = [r for r in recs if r.get("feasibility_status")]
    if not labelled:
        return None
    ok = sum(1 for r in labelled if r.get("feasibility_status") == FEASIBLE_OK)
    return ok / len(labelled)


def _top_gap(recs: Sequence[Mapping[str, Any]]) -> float | None:
    scores = sorted((s for s in (_num(r.get("vhs_score")) for r in recs) if s is not None), reverse=True)
    if len(scores) < 2:
        return None
    spread = scores[0] - scores[-1]
    if spread <= 0:
        return 0.0
    return max(0.0, min(1.0, (scores[0] - scores[1]) / spread))


def _greedy_alignment(recs: Sequence[Mapping[str, Any]]) -> float | None:
    flags = [r.get("strategy_match") for r in recs if r.get("strategy_match") is not None]
    if not flags:
        return None
    return sum(1 for f in flags if bool(f)) / len(flags)


def _pareto_top_membership(recs: Sequence[Mapping[str, Any]]) -> float | None:
    ranked = [r for r in recs if _num(r.get("vhs_rank")) is not None]
    if not ranked:
        return None
    top = min(ranked, key=lambda r: _num(r.get("vhs_rank")))
    status = str(top.get("pareto_status") or "")
    if not status:
        return None
    return 1.0 if ("비지배" in status or "우수" in status or "front" in status.lower()) else 0.4


def _stability_factor(stability_status: str | None) -> float | None:
    return {
        STABLE: 1.0,
        REVIEW: 0.6,
        UNSTABLE: 0.3,
    }.get(str(stability_status or ""), None)


def recommendation_confidence(
    recommendations: Sequence[Mapping[str, Any]],
    pipeline: Mapping[str, Any] | None = None,
    stability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Overall confidence status for the current recommendation set (DQN-optional).

    Combines only factors that are computable without DQN. DQN, when normal, adds
    a small bonus; its absence never lowers confidence. Returns status/score plus
    up to three plain reasons. No data → 계산 불가 with score None.
    """
    recs = list(recommendations or [])
    if not recs:
        return {"status": CONF_NONE, "score": None, "reasons": ["추천 결과가 없습니다."], "factors": {}}

    pipeline = pipeline or {}
    if stability is None:
        stability = recommendation_stability(pipeline.get("weight_sensitivity_analysis"))
    stability_status = stability.get("status") if isinstance(stability, Mapping) else None

    factors: dict[str, float] = {}
    completeness = _completeness(recs)
    factors["data_completeness"] = completeness
    for name, value in (
        ("feasibility_pass", _feasible_pass_rate(recs)),
        ("top_gap", _top_gap(recs)),
        ("rank_stability", _stability_factor(stability_status)),
        ("greedy_alignment", _greedy_alignment(recs)),
        ("pareto_top", _pareto_top_membership(recs)),
    ):
        if value is not None:
            factors[name] = value

    # Weighted blend over whichever factors were computable (renormalized).
    weights = {
        "data_completeness": 0.30,
        "feasibility_pass": 0.25,
        "top_gap": 0.15,
        "rank_stability": 0.15,
        "greedy_alignment": 0.10,
        "pareto_top": 0.05,
    }
    active = {k: w for k, w in weights.items() if k in factors}
    if not active:
        return {"status": CONF_NONE, "score": None, "reasons": ["신뢰도를 계산할 입력이 부족합니다."], "factors": factors}
    total_w = sum(active.values())
    score01 = sum(factors[k] * (active[k] / total_w) for k in active)

    # Optional DQN bonus: only when DQN is in a normal state and agrees. Never a penalty.
    dqn_result = (pipeline.get("dqn_training_result") if isinstance(pipeline, Mapping) else None) or {}
    dqn_status = str(dqn_result.get("status") or "")
    if dqn_status in DQN_READY_STATUSES:
        agree = [bool(r.get("vhs_vs_dqn_match")) for r in recs if r.get("vhs_vs_dqn_match") is not None]
        if agree and sum(agree) / len(agree) >= 0.5:
            score01 = min(1.0, score01 + 0.03)

    score = round(score01 * 100.0, 1)
    if score >= 75:
        status = CONF_HIGH
    elif score >= 55:
        status = CONF_MEDIUM
    else:
        status = CONF_LOW

    reasons: list[str] = []
    if factors.get("feasibility_pass") is not None and factors["feasibility_pass"] < 1.0:
        reasons.append("일부 후보는 데이터 확인이 필요합니다.")
    if completeness < 0.9:
        reasons.append("일부 입력 값이 비어 있어 신뢰도가 낮아졌습니다.")
    if factors.get("rank_stability") is not None and factors["rank_stability"] < 1.0:
        reasons.append("추천 순위가 가중치 변화에 다소 민감합니다.")
    if not reasons:
        reasons.append("입력이 충분하고 추천 순위가 안정적입니다.")

    return {"status": status, "score": score, "reasons": reasons[:3], "factors": {k: round(v, 3) for k, v in factors.items()}}
