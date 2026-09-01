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


def _map_key_present() -> bool:
    try:
        import streamlit as st  # local import keeps this module dependency-light
        from services.kakao_service import get_kakao_key_from_sources
        return bool(get_kakao_key_from_sources(st.secrets))
    except Exception:
        return False


def app_status_badges(state) -> list[tuple[str, str]]:
    """The four canonical status chips shown once in the top header.

    데이터(적용 완료/확인 필요) · 추천 계산(계산 완료/확인 필요) ·
    DQN(학습 전/비교 가능/검토 필요) · 지도(연결됨/미연결). Colours: 완료/정상=success,
    확인 필요/검토 필요=warning, 학습 전/미연결=neutral.
    """
    from services.app_state import has_app_data, has_applied_data

    data_ok = has_applied_data(state.get("varo_data"))
    result_ok = has_app_data(state.get("varo_data"), state.get("varo_recommendations"))
    pipeline = state.get("analysis_result") or state.get("varo_pipeline_result") or {}
    calc_ok = result_ok and bool(pipeline.get("connected_algorithms") or pipeline.get("v2_summary_functions"))
    dqn = state.get("dqn_training_result") or {}
    if not dqn:
        dqn_badge = ("DQN 학습 전", "neutral")
    elif str(dqn.get("status")) == "정상":
        dqn_badge = ("DQN 비교 가능", "success")
    else:
        dqn_badge = ("DQN 검토 필요", "warning")
    return [
        ("데이터 적용 완료", "success") if data_ok else ("데이터 확인 필요", "warning"),
        ("추천 계산 완료", "success") if calc_ok else ("추천 확인 필요", "warning"),
        dqn_badge,
        ("지도 연결됨", "success") if _map_key_present() else ("지도 미연결", "neutral"),
    ]


def app_status_badges_html(state) -> str:
    return "".join(badge_html(label, variant) for label, variant in app_status_badges(state))
