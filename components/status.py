"""Status and badge helpers."""
from __future__ import annotations

import html

_BADGE_VARIANTS = {"neutral", "accent", "success", "warning", "error"}

ROUTE_TYPE_LABELS = {
    "DIRECT": "직접 이동",
    "VIA_DC": "DC 경유",
}

STATUS_LABELS = {
    "READY": "대기",
    "MOVING": "이동 중",
    "AT_DC": "DC 도착",
    "COMPLETED": "완료",
    "PAUSED": "일시정지",
    "ERROR": "오류",
}

STATUS_VARIANTS = {
    "READY": "neutral",
    "MOVING": "accent",
    "AT_DC": "warning",
    "COMPLETED": "success",
    "PAUSED": "neutral",
    "ERROR": "error",
}


def badge_html(label: str, variant: str = "neutral") -> str:
    safe_variant = variant if variant in _BADGE_VARIANTS else "neutral"
    return f'<span class="v2-badge v2-badge-{safe_variant}">{html.escape(str(label))}</span>'


def route_type_badge(route_type: str) -> str:
    if route_type == "DIRECT":
        return badge_html(ROUTE_TYPE_LABELS[route_type], "accent")
    if route_type == "VIA_DC":
        return badge_html(ROUTE_TYPE_LABELS[route_type], "warning")
    return badge_html("경로 오류", "error")


def status_badge(status: str) -> str:
    label = STATUS_LABELS.get(status, "상태 미확인")
    return badge_html(label, STATUS_VARIANTS.get(status, "neutral"))


def data_quality_badge(label: str = "데이터 없음", variant: str = "neutral") -> str:
    return badge_html(label, variant)


def app_status_badges(state) -> list[tuple[str, str]]:
    """The two canonical status chips shown once in the top header.

    데이터(적용 완료/확인 필요) · 분석(완료/실행 필요). Nothing else belongs in the
    header: a user only needs to know whether the data is ready and whether the
    analysis can be run. Learning-model and map-connection state are working
    detail and live on their own detail screens.
    """
    from services.app_state import has_app_data, has_applied_data

    data_ok = has_applied_data(state.get("varo_data"))
    result_ok = has_app_data(state.get("varo_data"), state.get("varo_recommendations"))
    pipeline = state.get("analysis_result") or state.get("varo_pipeline_result") or {}
    calc_ok = result_ok and bool(pipeline.get("connected_algorithms") or pipeline.get("v2_summary_functions"))
    return [
        ("데이터 적용 완료", "success") if data_ok else ("데이터 확인 필요", "warning"),
        ("분석 완료", "success") if calc_ok else ("분석 실행 필요", "warning"),
    ]


def app_status_badges_html(state) -> str:
    return "".join(badge_html(label, variant) for label, variant in app_status_badges(state))
