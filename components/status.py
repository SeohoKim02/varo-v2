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

# Internal values are intentionally preserved in the analysis/session payload.
# Only user-facing views should pass status text through this mapping.
USER_STATUS_LABELS = {
    "불안정": "데이터 편향 큼",
    "검토 필요": "비교 전 데이터 확인 필요",
    "학습 필요": "학습 후 비교 가능",
    "미연결": "학습 후 비교 가능",
    "실행 환경 필요": "DQN 학습 실행 환경 필요",
    "PyTorch 미설치": "DQN 학습 실행 환경 필요",
    "SDK 미연결": "지도 키 설정 시 표시 가능",
    "DQN 반영 안 함": "최종 추천에는 참고 제외",
    "반영 안 함": "최종 추천에는 참고 제외",
}


def user_status_label(value: object, default: str = "-") -> str:
    """Return submission-friendly copy without changing the stored status."""
    if value in (None, ""):
        return default
    text = str(value)
    return USER_STATUS_LABELS.get(text, text)


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
