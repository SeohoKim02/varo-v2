"""Calculation provenance and comparability metadata for Varo V2."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

VHS_COMPONENT_COLUMNS = (
    "disposal_risk_score", "turnover_score", "abc_score", "aging_score",
    "demand_forecast_score", "trend_score", "newsvendor_score", "match_score",
    "service_level_score", "priority_queue_score", "queue_capacity_score",
    "category_balance_score", "safety_stock_score", "transport_lp_score",
    "disposal_avoidance_score", "discount_sensitivity_score", "bottleneck_score",
    "store_capacity_score", "lp_allocation_score", "multiobjective_score",
    "topsis_score", "pareto_score", "assignment_score", "heuristic_score",
    "greedy_score", "eoq_score",
)
VHS_PENALTY_COLUMNS = ("relocation_failure_score", "substitute_conflict_score")
VHS_OUTPUT_COLUMNS = (
    "vhs_raw", "vhs", "vhs_grade", "vhs_action", "vhs_action_icon",
    "vhs_dqn_correction", "vhs_dominant_situation", "vhs_rank",
)


def kpi_sources() -> dict[str, dict[str, str]]:
    return {
        "total_recommended_qty": {
            "source": "analysis_pipeline.recommendations",
            "basis": "업로드 추천 수량을 표준화한 pipeline 후보 합계",
        },
        "active_route_count": {
            "source": "analysis_pipeline.recommendations",
            "basis": "검증과 표준화를 통과한 pipeline 추천 건수",
        },
        "total_expected_saving": {
            "source": "v2_recommendations.expected_saving",
            "basis": "업로드된 사전 계산 절감액 합계",
        },
        "average_vhs_score": {
            "source": "services.vhs_score_engine.apply_auto_vhs",
            "basis": "DQN 보정 없이 재계산한 VHS 평균",
        },
        "data_quality": {
            "source": "services.data_validator.validate_workbook_data",
            "basis": "V2 자체 데이터 검증 상태",
        },
    }


def build_vhs_provenance(
    candidates: pd.DataFrame,
    input_columns: Sequence[str],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    available = [column for column in VHS_COMPONENT_COLUMNS if column in input_columns]
    defaulted = [column for column in VHS_COMPONENT_COLUMNS if column not in input_columns]
    penalties_available = [column for column in VHS_PENALTY_COLUMNS if column in input_columns]
    penalties_defaulted = [column for column in VHS_PENALTY_COLUMNS if column not in input_columns]
    uploaded = pd.to_numeric(candidates.get("uploaded_vhs_score"), errors="coerce")
    recalculated = pd.to_numeric(candidates.get("vhs"), errors="coerce")
    uploaded_average = round(float(uploaded.mean()), 3) if uploaded.notna().any() else None
    recalculated_average = round(float(recalculated.mean()), 3) if recalculated.notna().any() else None
    rows = []
    for index, row in candidates.iterrows():
        uploaded_value = uploaded.loc[index] if index in uploaded.index else None
        recalculated_value = recalculated.loc[index] if index in recalculated.index else None
        rows.append({
            "route_id": row.get("route_id"),
            "product_name": row.get("product_name"),
            "uploaded_vhs_score": None if pd.isna(uploaded_value) else float(uploaded_value),
            "recalculated_vhs_score": None if pd.isna(recalculated_value) else float(recalculated_value),
            "score_difference": (
                None if pd.isna(uploaded_value) or pd.isna(recalculated_value)
                else round(float(recalculated_value - uploaded_value), 1)
            ),
        })
    return {
        "calculation_function": "varo_hybrid_score.calculate_varo_hybrid_score",
        "input_columns": list(input_columns),
        "available_component_columns": available,
        "defaulted_component_columns": defaulted,
        "available_penalty_columns": penalties_available,
        "defaulted_penalty_columns": penalties_defaulted,
        "output_columns": [column for column in VHS_OUTPUT_COLUMNS if column in candidates.columns],
        "uploaded_average": uploaded_average,
        "recalculated_average": recalculated_average,
        "score_basis": "재계산 VHS",
        "uploaded_score_basis": "업로드된 사전 계산 VHS",
        "dqn_correction": 0.0,
        "dqn_correction_used": False,
        "comparison_rows": rows,
        "explanation": (
            f"원본 VHS 구성요소 {len(VHS_COMPONENT_COLUMNS)}개 중 {len(available)}개는 실제 입력을 사용하고 "
            f"{len(defaulted)}개는 원본 함수의 중립값 50을 사용했습니다. "
            f"패널티 입력 {len(penalties_defaulted)}개도 원본 기본 처리 규칙을 따릅니다. "
            "이 때문에 업로드 점수와 재계산 점수가 다를 수 있습니다."
        ),
        "summary": dict(summary),
    }


def build_greedy_provenance(candidates: pd.DataFrame) -> dict[str, Any]:
    selected_mask = candidates.get(
        "is_greedy_selected", pd.Series(False, index=candidates.index)
    ).fillna(False)
    selected = candidates[selected_mask == True]
    selected_route_id = str(selected.iloc[0].get("route_id")) if not selected.empty else None
    strategy_available = candidates.get(
        "greedy_action", pd.Series(index=candidates.index, dtype=object)
    ).notna()
    match = candidates.get(
        "strategy_match", pd.Series(False, index=candidates.index)
    ).fillna(False)
    return {
        "calculation_function": "heuristic_optimizer.add_heuristic_scores",
        "input_columns": ["estimated_cost", "suggested_qty", "final_recommendation", "reason"],
        "score_components": {
            "cost_score": "낮은 비용일수록 최대 40점",
            "quantity_score": "높은 처리 수량일수록 최대 25점",
            "strategy_score": "전략 유형별 5~20점",
            "reason_bonus": "비용·수요·경로 근거 최대 12점",
        },
        "sort_order": [
            "heuristic_score 내림차순",
            "estimated_cost 오름차순",
            "suggested_qty 내림차순",
        ],
        "selected_route_id": selected_route_id,
        "comparison_count": int(len(candidates)),
        "comparable_strategy_count": int(strategy_available.sum()),
        "unavailable_strategy_count": int((~strategy_available).sum()),
        "strategy_match_count": int(match.sum()),
        "strategy_match_rate": round(float(match.mean() * 100), 1) if len(match) else None,
    }


def sanitize_optimality_result(
    raw: Mapping[str, Any] | None,
    input_candidate_count: int,
) -> dict[str, Any]:
    raw = dict(raw or {})
    candidate_frame = raw.pop("df_with_cost", pd.DataFrame())
    candidates_used = int(raw.get("candidates_used") or 0)
    gap = raw.get("gap_pct")
    if gap is not None:
        status = "계산 가능" if candidates_used == input_candidate_count else "비교 가능 후보 기준"
    elif candidates_used == 0:
        status = "입력 컬럼 부족" if input_candidate_count else "비교 불가"
    else:
        status = "비교 불가"
    route_ids = (
        candidate_frame.get("route_id", pd.Series(dtype=object)).astype(str).tolist()
        if isinstance(candidate_frame, pd.DataFrame)
        else []
    )
    raw.update({
        "status": status,
        "calculation_function": "varo_optimality_gap.calculate_optimality_gap",
        "formula": "(Varo TOP-K 비용 - 최소비용 TOP-K) / 최소비용 TOP-K × 100",
        "cost_formula": "이동비용 + 할인손실 + 프로모션 고정비 - 폐기회피효과 - 양의 프로모션 순효과",
        "input_candidate_count": int(input_candidate_count),
        "comparable_candidate_count": candidates_used,
        "selection_size": len(raw.get("varo_idx", [])),
        "varo_route_ids": [
            route_ids[index] for index in raw.get("varo_idx", []) if 0 <= index < len(route_ids)
        ],
        "optimal_route_ids": [
            route_ids[index] for index in raw.get("opt_idx", []) if 0 <= index < len(route_ids)
        ],
        "fallback_used": raw.get("opt_method") not in (None, "-", "milp"),
    })
    return raw


def confidence_provenance(
    candidates: pd.DataFrame,
    removed_columns: Sequence[str],
) -> dict[str, Any]:
    scores = pd.to_numeric(candidates.get("confidence_score"), errors="coerce")
    levels = candidates.get("confidence_level", pd.Series(dtype=object)).value_counts().to_dict()
    return {
        "calculation_function": "vhs_confidence.add_confidence_columns",
        "input_columns": [
            "vhs2", "heuristic_grade", "suggested_qty", "estimated_cost",
            "vhs2_confidence", "demand_status", "promotion_status",
        ],
        "removed_dqn_columns": list(removed_columns),
        "dqn_status_argument": None,
        "dqn_bonus_applied": False,
        "score_range": [
            round(float(scores.min()), 1) if scores.notna().any() else None,
            round(float(scores.max()), 1) if scores.notna().any() else None,
        ],
        "average": round(float(scores.mean()), 1) if scores.notna().any() else None,
        "level_counts": {str(key): int(value) for key, value in levels.items()},
        "level_rules": {"높음": "80 이상", "보통": "60 이상 80 미만", "낮음": "60 미만"},
        "explanation": "DQN 상태와 모델 일치 컬럼을 제거하고 dqn_status=None으로 계산했습니다.",
    }
