"""Shared application-state rules for Varo V2."""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

CANONICAL_DATA_KEYS = (
    "varo_data",
    "varo_validation",
    "varo_recommendations",
    "varo_pipeline_result",
    "analysis_result",
    "pipeline_summary",
    "connected_algorithms",
    "deferred_algorithms",
    "dqn_excluded",
    "selected_route_id",
    "uploaded_filename",
    "data_source_type",
    "upload_report",
    "recommendation_source",
    "data_signature",
    "dqn_training_result",
    "dqn_reflection_mode",
)

TRANSIENT_VIEW_KEYS = (
    "rec_filter_product",
    "rec_filter_source",
    "rec_filter_target",
    "rec_filter_route_type",
    "rec_filter_grade",
    "rec_filter_transport",
    "recommendation_route_select",
    "route_detail_select",
)


def _has_rows(value: Any) -> bool:
    if value is None:
        return False
    empty = getattr(value, "empty", None)
    if empty is not None:
        return not bool(empty)
    try:
        return len(value) > 0
    except TypeError:
        return False


def has_app_data(data: Mapping[str, Any] | None, recommendations: Sequence[Mapping[str, object]] | None) -> bool:
    if not isinstance(data, Mapping):
        return False
    return _has_rows(data.get("stores")) and _has_rows(recommendations)


def default_selected_route_id(recommendations: Sequence[Mapping[str, object]] | None) -> str | None:
    from services.analysis_pipeline import top_recommendations

    top_route = top_recommendations(list(recommendations or []), limit=1)
    if not top_route:
        return None
    route_id = top_route[0].get("route_id")
    return str(route_id) if route_id not in (None, "") else None


def resolve_selected_route_id(
    recommendations: Sequence[Mapping[str, object]] | None,
    selected_route_id: str | None,
) -> str | None:
    route_ids = {str(item.get("route_id")) for item in recommendations or [] if item.get("route_id")}
    if selected_route_id and selected_route_id in route_ids:
        return selected_route_id
    return default_selected_route_id(recommendations)


def build_applied_state_payload(
    data: Mapping[str, Any],
    validation: Any,
    recommendations: Sequence[Mapping[str, object]],
    filename: str,
    source_type: str,
    pipeline_result: Mapping[str, Any] | None = None,
    data_signature: str | None = None,
) -> dict[str, Any]:
    if getattr(validation, "has_errors", False):
        raise ValueError("검증 오류가 있는 데이터는 앱에 적용할 수 없습니다.")
    recommendation_list = [dict(item) for item in recommendations]
    pipeline = dict(pipeline_result or {})
    return {
        "varo_data": dict(data),
        "varo_validation": validation,
        "varo_recommendations": recommendation_list,
        "varo_pipeline_result": pipeline,
        "analysis_result": pipeline,
        "pipeline_summary": pipeline.get("summary", {}),
        "connected_algorithms": pipeline.get("connected_algorithms", []),
        "deferred_algorithms": pipeline.get("deferred_algorithms", []),
        "dqn_excluded": pipeline.get("excluded_dqn_artifacts", {}),
        "selected_route_id": default_selected_route_id(recommendation_list),
        "uploaded_filename": filename,
        "data_source_type": source_type,
        "upload_report": {},
        "recommendation_source": "uploaded",
        "data_signature": data_signature,
    }


def apply_state_payload(state: MutableMapping[str, Any], payload: Mapping[str, Any]) -> None:
    for key in CANONICAL_DATA_KEYS:
        state[key] = payload.get(key)
    for key in TRANSIENT_VIEW_KEYS:
        state.pop(key, None)
    state["simulation_snapshot"] = None
    state["show_all_routes"] = False
    state["home_sim_playing"] = False
    state["simulation_speed"] = "보통"
    state["dqn_training_result"] = None
    state["dqn_reflection_mode"] = "DQN 참고만"
    state["dqn_batch_result"] = None
    state["dqn_comparison_result"] = None
    state["dqn_original_batch_result"] = None
    state["dqn_balanced_batch_result"] = None
    state["dqn_batch_comparison_result"] = None
    state["dqn_sample_diagnosis"] = None
    state["dqn_balanced_files"] = None
    state["dqn_baseline_recommendations"] = None
    state["dqn_baseline_pipeline"] = None
    state["dqn_sample_training_mode"] = "original"
    state["dqn_selected_sample"] = None
    state["sensitivity_settings"] = {}
    state["sensitivity_result"] = None
    state["sensitivity_summary"] = None
    state["sensitivity_data_signature"] = None
    state["sensitivity_is_running"] = False
    state["sensitivity_last_error"] = None
    state["optimality_gap_settings"] = {}
    state["optimality_gap_result"] = None
    state["optimality_gap_data_signature"] = None
    state["optimality_gap_is_running"] = False
    state["optimality_gap_last_error"] = None


def data_status_label(
    data: Mapping[str, Any] | None,
    recommendations: Sequence[Mapping[str, object]] | None,
    validation: Any,
    source_type: str | None,
    pipeline_result: Mapping[str, Any] | None = None,
) -> str:
    status = getattr(validation, "status", None)
    if isinstance(validation, Mapping):
        status = validation.get("status")
    if status == "오류":
        return "검증 오류"
    if not has_app_data(data, recommendations):
        return "데이터 없음"
    pipeline_status = (pipeline_result or {}).get("status")
    if pipeline_status == "success":
        label = "알고리즘 연결됨"
    elif pipeline_status == "partial":
        label = "일부 알고리즘 연결"
    elif source_type == "샘플 추천 데이터":
        label = "샘플 적용됨"
    elif source_type == "업로드된 추천 결과":
        label = "업로드 완료"
    else:
        label = "데이터 적용됨"
    return f"{label} · 주의" if status == "주의" else label


def current_data_status(state: Mapping[str, Any]) -> str:
    return data_status_label(
        state.get("varo_data"),
        state.get("varo_recommendations"),
        state.get("varo_validation"),
        state.get("data_source_type"),
        state.get("varo_pipeline_result"),
    )


def current_result_basis(state: Mapping[str, Any]) -> str:
    pipeline = state.get("analysis_result") or state.get("varo_pipeline_result")
    if isinstance(pipeline, Mapping) and pipeline.get("result_basis"):
        return str(pipeline["result_basis"])
    if has_app_data(state.get("varo_data"), state.get("varo_recommendations")):
        return "업로드된 사전 계산 추천 결과 기준"
    return "알고리즘 미연결"
