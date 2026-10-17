"""Presentation-only diagnostics: retain details in logs, not customer screens."""
import logging
import re

import streamlit as st

logger = logging.getLogger("varo.ui")
logger.setLevel(logging.INFO)

_CODE_TEXT = re.compile(
    r"Traceback|\b\w*(?:Error|Exception)\b|File \"|"
    r"\b[a-zA-Z]+_[a-zA-Z_]+\b|\b[A-Z]{3,}_[A-Z_]+\b|"
    r"```|(?m:^\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+\s*\(|class\s+\w+|return\s+))|"
    r"\b(?:st|pd|np|self)\.[A-Za-z_]\w*|\b[A-Za-z_]\w*\.[A-Za-z_]\w*\(|"
    r"[A-Za-z]:[\\/]|\b\w+\.py(?:\b|:)|\b(?:print|exec|eval)\s*\(|"
    r"(?m:^\s*[A-Za-z_]\w*\s*=\s*[^=])|^\s*[\{\[]\s*['\"]"
)


def report_error(context, error=None, message="화면을 불러오지 못했습니다. 잠시 후 다시 시도하세요. 계속되면 관리자에게 문의하세요."):
    logger.error("%s: %r", context, error, exc_info=isinstance(error, BaseException))
    st.error(user_text(message, "처리 중 문제가 발생했습니다. 입력과 파일 상태를 확인한 뒤 다시 시도하세요. 계속되면 관리자에게 문의하세요."))


def diagnostic_details(context, payload):
    logger.info("%s: %r", context, payload)
    st.caption("기술 진단은 개발자 로그에 기록했습니다. 원본 근거는 결과 다운로드에서 확인할 수 있습니다.")


def user_text(value, fallback="자료를 확인해 주세요"):
    text = str(value or "")
    if _CODE_TEXT.search(text):
        logger.info("Internal display text: %s", text)
        return fallback
    return text or fallback


def analysis_label(value):
    """Name a calculation family without exposing its Python import path."""
    text = str(value or "")
    for markers, label in (
        (("confidence", "sensitivity"), "추천 안정성 분석"),
        (("demand", "safety_stock", "forecast"), "수요·필요 재고 분석"),
        (("inventory", "disposal", "abc", "clustering"), "재고 상태 분석"),
        (("transfer", "route", "transport"), "이동 경로 분석"),
        (("greedy",), "비용·수량 비교"),
        (("pareto",), "여러 기준의 대안 비교"),
        (("dqn",), "학습 결과 참고 비교"),
        (("vhs", "hybrid_score"), "재고 처리 우선순위 분석"),
        (("promotion",), "판매 처리 대안 비교"),
    ):
        if any(marker in text.lower() for marker in markers):
            return label
    return user_text(text, "보조 분석")


FIELD_LABELS = {
    "source_normal_price": "출발 점포 정상 판매가격", "target_normal_price": "도착 점포 정상 판매가격",
    "source_current_stock": "출발 점포 현재 재고", "target_current_stock": "도착 점포 현재 재고",
    "source_daily_demand": "출발 점포 하루 판매수요", "target_daily_demand": "도착 점포 하루 판매수요",
    "remaining_shelf_life_days": "남은 유통기한", "discount_rate": "할인율",
    "promotion_uplift": "할인 판매 증가율", "transfer_cost": "이동비", "decision_qty": "판단 대상 수량",
    "source_id": "출발 점포", "target_id": "도착 점포", "transit_time_days": "이동 소요일",
    "unit_cost": "상품 원가", "disposal_cost": "폐기 비용",
}


def explanation_text(value):
    text = str(value or "")
    for key, label in sorted(FIELD_LABELS.items(), key=lambda pair: -len(pair[0])):
        text = text.replace(key, label)
    return user_text(text, "비교 근거를 확인해 주세요. 필요한 추가 입력과 전략별 예상 손실을 함께 검토하세요.")
