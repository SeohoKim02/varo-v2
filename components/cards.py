"""Common card components for Varo V2."""
from __future__ import annotations

import html
from typing import Iterable, Mapping, Optional

from components.status import badge_html, data_quality_badge, route_type_badge
from components.tables import format_currency, format_number


def _safe(value) -> str:
    if value is None:
        return "-"
    return html.escape(str(value))


def render_page_header(st, title: str, description: str, badge: Optional[str] = None) -> None:
    badge_html_value = f"<div>{badge}</div>" if badge else ""
    st.markdown(
        f"""
        <div class="v2-wrap v2-page-header">
          <div>
            <h1 class="v2-page-title">{_safe(title)}</h1>
            <div class="v2-page-desc">{_safe(description)}</div>
          </div>
          {badge_html_value}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(st, title: str, description: Optional[str] = None, right: Optional[str] = None) -> None:
    desc = f'<div class="v2-section-desc">{_safe(description)}</div>' if description else ""
    right_html = f"<div>{right}</div>" if right else ""
    st.markdown(
        f"""
        <div class="v2-wrap v2-section-header">
          <div>
            <div class="v2-section-title">{_safe(title)}</div>
            {desc}
          </div>
          {right_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    st,
    title: str,
    value: str,
    caption: str = "",
    status: str = "neutral",
    compact: bool = False,
) -> None:
    compact_class = " v2-kpi-card-compact" if compact else ""
    # A short one-line description under the value (shown in both compact and full).
    caption_html = f'<div class="v2-kpi-desc">{_safe(caption)}</div>' if caption else ""
    status_html = f'<div style="margin-top:0.45rem;">{badge_html(status, "neutral")}</div>' if not compact else ""
    st.markdown(
        f"""
        <div class="v2-wrap v2-card v2-kpi-card{compact_class}">
          <div class="v2-kpi-title">{_safe(title)}</div>
          <div class="v2-kpi-value">{_safe(value)}</div>
          {caption_html}
          {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(st, title: str = "데이터가 업로드되지 않았습니다", message: str = "", compact: bool = False) -> None:
    detail = f'<div class="v2-card-caption" style="margin-top:0.35rem;">{_safe(message)}</div>' if message else ""
    compact_class = " v2-empty-state-compact" if compact else ""
    st.markdown(
        f"""
        <div class="v2-wrap v2-empty-state{compact_class}">
          <strong>{_safe(title)}</strong>
          {detail}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_card(st, title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="v2-wrap v2-error-card">
          <strong>{_safe(title)}</strong>
          <div style="margin-top:0.35rem;">{_safe(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _final_action_hint(recommendation: Mapping[str, object]) -> str:
    """One short line only when the final proposal could look at odds with the route."""
    action = str(recommendation.get("varo_action") or "").strip()
    if action in {"보류", "keep_inventory", "hold", "no_action", "maintain"}:
        return "이동 경로는 가능하지만 현재 재고·수요 조건을 함께 고려해 실행 보류로 판단했습니다."
    if action and recommendation.get("route_type") == "DIRECT":
        return "이동비용이 낮고 도착 점포의 수요가 충분해 직접 이동을 권장합니다."
    return ""


def render_recommendation_summary(st, recommendation: Optional[Mapping[str, object]]) -> None:
    """Focused summary: product, route, quantity, saving, final proposal (no internal IDs)."""
    if not recommendation:
        render_empty_state(st, "선택한 추천 후보가 없습니다", compact=True)
        return
    review_route = "DC 경유" if recommendation.get("route_type") == "VIA_DC" else "직접 이동"
    rows = [
        ("검토 경로", review_route),
        ("추천 수량", format_number(recommendation.get("recommended_qty"), "개")),
        ("예상 순효과", format_currency(recommendation.get("net_benefit"))),
        ("추천 안정성", recommendation.get("robustness_status") or "-"),
    ]
    grid = "".join(
        f'<div class="v2-info-item"><span class="v2-card-caption">{_safe(label)}</span><strong>{_safe(value)}</strong></div>'
        for label, value in rows
    )
    hint = _final_action_hint(recommendation)
    hint_html = f'<div class="v2-card-caption" style="margin-top:0.6rem;">{_safe(hint)}</div>' if hint else ""
    source = recommendation.get("source_name") or recommendation.get("source_id") or "-"
    target = recommendation.get("target_name") or recommendation.get("target_id") or "-"
    st.markdown(
        f"""
        <div class="v2-wrap v2-card">
          <div class="v2-card-head">
            <div>
              <div class="v2-card-title">{_safe(recommendation.get("product_name") or "-")}</div>
              <div class="v2-card-caption">{_safe(source)} → {_safe(target)}</div>
            </div>
            <div>{route_type_badge(str(recommendation.get("route_type", "")))}</div>
          </div>
          <div class="v2-recommendation-info">{grid}</div>
          {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_list(
    st,
    title: str,
    metrics: Iterable[str],
    status_label: str = "데이터 없음",
    status_variant: str = "neutral",
) -> None:
    items = "".join(f"<li>{_safe(metric)}</li>" for metric in metrics)
    st.markdown(
        f"""
        <div class="v2-wrap v2-card">
          <div class="v2-card-title">{_safe(title)}</div>
          <ul style="margin:0.4rem 0 0 1.15rem;color:var(--varo-muted);line-height:1.7;">{items}</ul>
          <div style="margin-top:0.7rem;">{data_quality_badge(status_label, status_variant)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
