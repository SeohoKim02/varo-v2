"""V2-internal summary builders.

These are simple, stable summaries computed only from Varo V2's own recompute
results (recommendations + VHS provenance). They are NOT the original
``varo_sensitivity``/``vhs_reason`` modules — those stay deferred. DQN values are
never used as a score here; the DQN exclusion is only stated, never quantified.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

V2_SUMMARY_FUNCTIONS = (
    "V2 VHS 중립값 요약",
    "V2 민감도 요약",
    "V2 추천 사유 요약",
)

_GRADE_SCORE = {"낮음": 0, "보통": 1, "높음": 2}
_SCORE_GRADE = {0: "낮음", 1: "보통", 2: "높음"}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # drop NaN


# --------------------------------------------------------------------------- #
# VHS neutral components
# --------------------------------------------------------------------------- #
def vhs_neutral_summary(pipeline: Mapping[str, Any] | None) -> dict[str, Any]:
    vhs = (pipeline or {}).get("vhs_analysis") or {}
    available = list(vhs.get("available_component_columns") or [])
    defaulted = list(vhs.get("defaulted_component_columns") or [])
    penalty_available = list(vhs.get("available_penalty_columns") or [])
    penalty_defaulted = list(vhs.get("defaulted_penalty_columns") or [])
    total = len(available) + len(defaulted)
    return {
        "total_components": total,
        "calculated_components": len(available),
        "neutral_components": len(defaulted),
        "excluded_components": 1,  # vhs_dqn_correction (DQN 보정) — 항상 0으로 제외
        "penalty_calculated": len(penalty_available),
        "penalty_neutral": len(penalty_defaulted),
        "calculated_list": available,
        "neutral_list": defaulted,
        "excluded_list": ["vhs_dqn_correction (DQN 보정, 0 처리)"],
        "neutral_reason": "입력 컬럼 부족 또는 DQN 제외 정책으로 일부 구성요소에 중립값(50)을 적용했습니다.",
        "interpretation": (
            "현재 VHS는 Varo V2 내부 알고리즘으로 재계산된 값입니다. "
            "중립값은 추천 결과가 과도하게 흔들리지 않도록 기준값(50)으로 처리한 값이며, "
            "DQN 관련 구성요소는 현재 추천에 반영하지 않았습니다."
        ),
    }


def vhs_neutral_rows(
    pipeline: Mapping[str, Any] | None,
    recommendations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summary = vhs_neutral_summary(pipeline)
    rows: list[dict[str, Any]] = []
    for rec in recommendations or []:
        uploaded = _num(rec.get("uploaded_vhs_score"))
        recalculated = _num(rec.get("vhs_score"))
        rows.append({
            "route_id": rec.get("route_id"),
            "product_name": rec.get("product_name"),
            "uploaded_vhs": uploaded,
            "recalculated_vhs": recalculated,
            "calculated_components": summary["calculated_components"],
            "neutral_components": summary["neutral_components"],
            "excluded_components": summary["excluded_components"],
            "neutral_reason": "입력 컬럼 부족 / DQN 제외",
            "final_basis": "재계산 VHS (Varo V2 내부 알고리즘)",
            "note": "중립값은 기준값(50)으로 처리되어 순위 급변을 막습니다.",
        })
    return rows


# --------------------------------------------------------------------------- #
# V2 sensitivity summary
# --------------------------------------------------------------------------- #
def _grade_from_gap(gap_ratio: float | None) -> str:
    if gap_ratio is None:
        return "제한적"
    if gap_ratio < 0.10:
        return "높음"
    if gap_ratio < 0.25:
        return "보통"
    return "낮음"


def _metric_grades(values: list[tuple[Any, float | None]]) -> dict[Any, str]:
    numbered = [(rid, v) for rid, v in values if v is not None]
    if len(numbered) < 2:
        return {rid: "제한적" for rid, _ in values}
    sorted_values = sorted(v for _, v in numbered)
    value_range = (sorted_values[-1] - sorted_values[0]) or 1.0
    grades: dict[Any, str] = {}
    for rid, value in values:
        if value is None:
            grades[rid] = "제한적"
            continue
        others = [abs(value - other) for orid, other in numbered if orid != rid]
        gap = min(others) if others else value_range
        grades[rid] = _grade_from_gap(gap / value_range)
    return grades


def _overall_grade(grades: Sequence[str]) -> str:
    scored = [_GRADE_SCORE[g] for g in grades if g in _GRADE_SCORE]
    if not scored:
        return "제한적"
    return _SCORE_GRADE[round(sum(scored) / len(scored))]


def sensitivity_summary(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    recs = list(recommendations or [])
    if not recs:
        return []
    metric_fields = {
        "cost": "estimated_cost",
        "distance": "distance_km",
        "quantity": "recommended_qty",
        "vhs": "vhs_score",
        "savings": "expected_saving",
    }
    grades_by_metric = {
        metric: _metric_grades([(r.get("route_id"), _num(r.get(field))) for r in recs])
        for metric, field in metric_fields.items()
    }
    rows: list[dict[str, Any]] = []
    for rec in recs:
        rid = rec.get("route_id")
        per_metric = {metric: grades_by_metric[metric].get(rid, "제한적") for metric in metric_fields}
        overall = _overall_grade(list(per_metric.values()))
        if overall == "높음":
            note = "다른 후보와 지표 차이가 작아 작은 변화에도 순위가 바뀔 수 있습니다."
        elif overall == "낮음":
            note = "지표 차이가 뚜렷해 순위가 비교적 안정적입니다."
        elif overall == "제한적":
            note = "입력값이 부족해 제한적으로만 평가했습니다."
        else:
            note = "지표 변화에 보통 수준의 영향을 받습니다."
        rows.append({
            "route_id": rid,
            "product_name": rec.get("product_name"),
            "sensitivity_cost": per_metric["cost"],
            "sensitivity_distance": per_metric["distance"],
            "sensitivity_quantity": per_metric["quantity"],
            "sensitivity_vhs": per_metric["vhs"],
            "overall_sensitivity": overall,
            "stability_note": note,
        })
    return rows


# --------------------------------------------------------------------------- #
# V2 recommendation reasons
# --------------------------------------------------------------------------- #
DQN_REASON_NOTE = "DQN은 과거 학습 결과 이상치 가능성으로 현재 판단에 반영하지 않았습니다."
NEUTRAL_CAUTION = "일부 VHS 구성요소는 입력 컬럼 부족으로 중립값이 적용되었습니다."


def recommendation_reason(
    rec: Mapping[str, Any],
    recommendations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sentences: list[str] = []
    vhs = _num(rec.get("vhs_score"))
    saving = _num(rec.get("expected_saving"))
    quantity = _num(rec.get("recommended_qty"))
    route_type = rec.get("route_type")

    vhs_values = sorted(
        v for v in (_num(r.get("vhs_score")) for r in recommendations) if v is not None
    )
    if vhs is not None and vhs_values:
        median = vhs_values[len(vhs_values) // 2]
        if vhs >= median:
            sentences.append("재계산 VHS가 상위권이라 우선 검토할 수 있는 후보입니다.")
        else:
            sentences.append("재계산 VHS는 중하위권이지만 절감액·경로 조건을 함께 검토할 수 있습니다.")
    if saving and saving > 0:
        sentences.append(f"예상 절감액이 약 {saving:,.0f}원으로 재배치 효과가 기대됩니다.")
    if route_type == "VIA_DC":
        if rec.get("cutline_passed") in ("이동 가능", None) and rec.get("time_window_status") in ("가능", None):
            sentences.append("DC 경유 경로이지만 비용·시간·컷라인 기준을 통과했습니다.")
        else:
            sentences.append("DC 경유 경로로 합류 이동을 활용합니다.")
    else:
        sentences.append("직접 이동 경로로 단순한 경로 구성입니다.")
    promotion = str(rec.get("promotion_recommended") or "")
    if "재배치" in promotion:
        sentences.append("프로모션 처리보다 재배치 비용이 낮아 재배치가 유리합니다.")
    if rec.get("strategy_match"):
        sentences.append("Greedy 비교 결과와도 전략이 일치합니다.")
    grade = rec.get("recommendation_grade")
    if grade and quantity:
        sentences.append(f"추천 신뢰도 등급은 '{grade}'이며 이동 수량은 {quantity:,.0f}개입니다.")

    return {
        "route_id": rec.get("route_id"),
        "sentences": sentences[:4] or ["연결된 분석 지표를 기준으로 검토할 수 있는 후보입니다."],
        "caution": NEUTRAL_CAUTION,
        "dqn_note": DQN_REASON_NOTE,
    }


def recommendation_reasons(
    recommendations: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(rec.get("route_id")): recommendation_reason(rec, recommendations)
        for rec in recommendations or []
    }


def recommendation_reason_rows(
    recommendations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in recommendations or []:
        reason = recommendation_reason(rec, recommendations)
        rows.append({
            "route_id": rec.get("route_id"),
            "product_name": rec.get("product_name"),
            "recommendation_reason": " ".join(reason["sentences"]),
            "caution": reason["caution"],
            "dqn_note": reason["dqn_note"],
        })
    return rows
