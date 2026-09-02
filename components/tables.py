"""Table helpers for Varo V2."""
from __future__ import annotations

import html
import math
from typing import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from services.analysis_pipeline import sort_recommendations

ROUTE_TYPE_LABELS = {"DIRECT": "직접 이동", "VIA_DC": "DC 경유"}


def format_currency(value) -> str:
    if value in (None, ""):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    # NaN / inf are missing/invalid data — never render them as "nan원"/"inf원".
    if not math.isfinite(number):
        return "-"
    return f"{number:,.0f}원"


def format_number(value, suffix: str = "") -> str:
    if value in (None, ""):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{int(number):,}{suffix}" if number.is_integer() else f"{number:,.1f}{suffix}"


def route_type_label(route_type: str | None) -> str:
    return ROUTE_TYPE_LABELS.get(route_type, route_type or "-")


def build_recommendation_rows(
    recommendations: Iterable[dict],
    limit: int | None = None,
    include_analysis: bool = False,
    include_route_id: bool = True,
    include_status: bool = True,
    include_vhs: bool = True,
    include_grade: bool = False,
    include_feasibility: bool = False,
) -> list[dict]:
    rows = []
    sorted_items = sort_recommendations(list(recommendations))
    if limit is not None:
        sorted_items = sorted_items[:limit]
    for idx, rec in enumerate(sorted_items, start=1):
        row = {
            "순위": idx,
            "상품": rec.get("product_name") or "-",
            "출발 점포": rec.get("source_name") or rec.get("source_id") or "-",
            "도착 점포": rec.get("target_name") or rec.get("target_id") or "-",
            "경로 유형": route_type_label(rec.get("route_type")),
            "수량": format_number(rec.get("planned_qty") if rec.get("planned_qty") is not None else rec.get("recommended_qty"), "개"),
            # 순효과(절감액 − 이동비용)가 실제 판단 기준이므로 기본 표에는 이 값을 둔다.
            # 절감액·이동비용 각각은 후보 상세에서 그대로 확인할 수 있다.
            "예상 순효과": format_currency(rec.get("net_benefit")),
            "안정성": rec.get("robustness_status") or "-",
        }
        if include_grade:
            row["추천 등급"] = rec.get("recommendation_grade") or "-"
        if include_feasibility:
            row["실행 상태"] = rec.get("feasibility_status") or "추천 가능"
        if include_vhs:
            row["VHS"] = format_number(rec.get("vhs_score"))
        if include_route_id:
            row = {"순위": row.pop("순위"), "route_id": rec.get("route_id") or "-", **row}
        if include_analysis:
            row.update({
                "Greedy": rec.get("greedy_action") or "비교 불가",
                "Varo 추천": rec.get("varo_action") or "-",
                "DQN 상태": rec.get("dqn_action") or "미연결",
            })
        if include_status:
            row["상태"] = rec.get("status") or "READY"
        rows.append(row)
    return rows


def build_top5_rows(recommendations: Iterable[dict]) -> list[dict]:
    return build_recommendation_rows(recommendations, limit=5)


def build_home_top_rows(recommendations: Iterable[dict], limit: int = 5) -> list[dict]:
    """Result-only Top rows for the home dashboard (no VHS/Greedy/DQN/status)."""
    rows = []
    for idx, rec in enumerate(sort_recommendations(list(recommendations))[:limit], start=1):
        rows.append({
            "순위": idx,
            "상품": rec.get("product_name") or "-",
            "출발": rec.get("source_name") or rec.get("source_id") or "-",
            "도착": rec.get("target_name") or rec.get("target_id") or "-",
            "경로": route_type_label(rec.get("route_type")),
            "수량": format_number(rec.get("planned_qty") if rec.get("planned_qty") is not None else rec.get("recommended_qty"), "개"),
            "예상 순효과": format_currency(rec.get("net_benefit")),
        })
    return rows


def render_recommendation_table(rows: list[dict], key: str = "recommendation_table", height: int | None = None) -> None:
    if not rows:
        st.info("표시할 추천 결과가 없습니다.")
        return
    kwargs = {"hide_index": True, "width": "stretch", "key": key}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(pd.DataFrame(rows), **kwargs)


def _cell_text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, bool):
        return "예" if value else "아니오"
    return str(value)


def render_html_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str] | None = None) -> None:
    """Render a compact, fully-in-DOM table (no horizontal virtualization).

    Used for the *default* comparison view so every basic column is readable and
    testable, while the wide full table stays in an st.dataframe expander.
    """
    if not rows:
        st.info("표시할 결과가 없습니다.")
        return
    columns = list(columns or rows[0].keys())
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_cell_text(row.get(column)))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        '<div class="v2-wrap v2-html-table-wrap"><table class="v2-html-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_capped_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str] | None = None,
    limit: int = 10,
    expander_label: str = "전체 결과 보기",
) -> None:
    """Show the first ``limit`` rows on screen; the full set moves into an expander."""
    rows = list(rows)
    if not rows:
        st.info("표시할 결과가 없습니다.")
        return
    render_html_table(rows[:limit], columns)
    if len(rows) > limit:
        with st.expander(f"{expander_label} (전체 {len(rows)}행)", expanded=False):
            render_html_table(rows, columns)
