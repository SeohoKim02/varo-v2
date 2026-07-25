"""Shared application-state rules for Varo V2."""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from services.analysis_pipeline import top_recommendations

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
    "raw_data",
    "source_metadata",
    "dqn_training_result",
    "dqn_reflection_mode",
    "kakao_map_state",
)

# Widget-backed session keys whose stored value is tied to the *previous*
# dataset. Streamlit keeps a widget's value across reruns, so these must be
# cleared when new data is applied. If left behind they either surface a stale
# filter/selection or (for the home shadow controls) silently re-apply the old
# value over the canonical reset performed in apply_state_payload.
TRANSIENT_VIEW_KEYS = (
    # 추천 실행 페이지: 필터 / 경로 선택
    "rec_filter_product",
    "rec_filter_source",
    "rec_filter_target",
    "rec_filter_route_type",
    "rec_filter_grade",
    "rec_filter_transport",
    "recommendation_route_select",
    # 경로 상세 페이지: 선택된 경로
    "route_detail_select",
    # 홈 시뮬레이션 컨트롤(shadow 키). 남아있으면 아래 simulation_speed /
    # show_all_routes 리셋을 다음 렌더에서 되돌려버린다.
    "home_speed_select",
    "home_show_all",
    # 데이터 관리 페이지: 원본 시트 선택(파일마다 시트 구성이 달라 stale 위험).
    "raw_sheet_select",
)


# Empty defaults for every canonical key (mirrors app_v2.initialize_session_state)
# so 현재 데이터 초기화 leaves the workspace in the exact 데이터 없음 shape.
_CANONICAL_EMPTY: dict[str, Any] = {
    "varo_data": None,
    "varo_validation": None,
    "varo_recommendations": [],
    "varo_pipeline_result": {},
    "analysis_result": {},
    "pipeline_summary": {},
    "connected_algorithms": [],
    "deferred_algorithms": [],
    "dqn_excluded": {},
    "selected_route_id": None,
    "uploaded_filename": None,
    "data_source_type": None,
    "upload_report": {},
    "recommendation_source": "uploaded",
    "data_signature": None,
    "raw_data": {},
    "source_metadata": {},
    "dqn_training_result": None,
    "dqn_reflection_mode": "DQN 참고만",
    "kakao_map_state": None,
}

# Pending-upload keys (kept in step with data_application.PENDING_KEYS + the load
# error/apply error). Defined here to avoid importing data_application (which
# imports us).
_PENDING_KEYS = (
    "pending_varo_data", "pending_varo_validation", "pending_varo_recommendations",
    "pending_uploaded_filename", "pending_data_source_type", "pending_upload_report",
    "pending_raw_data", "pending_source_metadata", "pending_load_error",
    "pending_data_signature", "pending_recommendation_source", "pending_apply_allowed",
    "pending_usable_rows", "pending_excluded_rows", "pending_status", "pending_created_at",
    "data_apply_error",
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
    raw_data: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if getattr(validation, "has_errors", False):
        raise ValueError("검증 오류가 있는 데이터는 앱에 적용할 수 없습니다.")
    recommendation_list = [dict(item) for item in recommendations]
    pipeline = dict(pipeline_result or {})
    return {
        "varo_data": dict(data),
        "raw_data": dict(raw_data or {}),
        "source_metadata": dict(source_metadata or {}),
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


# Canonical keys that may also be bound to an already-instantiated widget on the
# current script run (the DQN 반영 방식 radio uses key="dqn_reflection_mode").
# Streamlit forbids writing a key that is bound to a live widget, so these are
# assigned defensively: the data-management load path (no such widget on screen)
# still resets them normally, while an in-page apply (e.g. 원본 샘플로 학습 in the
# DQN tab) leaves the live widget's value as the user set it instead of crashing.
_WIDGET_BOUND_KEYS = ("dqn_reflection_mode",)


def _assign(state: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set state[key]=value, tolerating only Streamlit's live-widget-key error.

    A plain dict target (used in tests) never raises. Any error other than
    StreamlitAPIException is re-raised so real failures are not swallowed.
    """
    try:
        state[key] = value
    except Exception as exc:  # noqa: BLE001 - narrowed by name check below
        if type(exc).__name__ != "StreamlitAPIException":
            raise


def apply_state_payload(state: MutableMapping[str, Any], payload: Mapping[str, Any]) -> None:
    for key in CANONICAL_DATA_KEYS:
        if key in _WIDGET_BOUND_KEYS:
            _assign(state, key, payload.get(key))
        else:
            state[key] = payload.get(key)
    for key in TRANSIENT_VIEW_KEYS:
        state.pop(key, None)
    state["simulation_snapshot"] = None
    state["show_all_routes"] = False
    state["home_sim_playing"] = False
    state["simulation_speed"] = "보통"
    state["dqn_training_result"] = None
    _assign(state, "dqn_reflection_mode", "DQN 참고만")
    state["kakao_map_state"] = None


def clear_applied_data(state: MutableMapping[str, Any]) -> None:
    """Reset the applied dataset and every derived result to an empty workspace.

    Clears only the in-app applied state (canonical data, analysis result,
    candidate ledger, selected candidate, path detail, simulation) plus any
    pending upload. Never touches the user's original files — only the copy the
    app currently holds. After this the workspace reads as 데이터 없음.
    """
    for key in CANONICAL_DATA_KEYS:
        _assign(state, key, _CANONICAL_EMPTY.get(key))
    for key in TRANSIENT_VIEW_KEYS:
        state.pop(key, None)
    for key in _PENDING_KEYS:
        state.pop(key, None)
    state["simulation_snapshot"] = None
    state["show_all_routes"] = False
    state["home_sim_playing"] = False
    state["simulation_speed"] = "보통"
    state["data_apply_message"] = None


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
