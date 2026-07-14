"""Table helpers for Varo V2."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

from services.analysis_pipeline import sort_recommendations

ROUTE_TYPE_LABELS = {"DIRECT": "직접 이동", "VIA_DC": "DC 경유"}


def format_currency(value) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):,.0f}원"
    except (TypeError, ValueError):
        return "-"


def format_number(value, suffix: str = "") -> str:
    if value in (None, ""):
        return "-"
    try:
        number = float(value)
        return f"{int(number):,}{suffix}" if number.is_integer() else f"{number:,.1f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def route_type_label(route_type: str | None) -> str:
    return ROUTE_TYPE_LABELS.get(route_type, route_type or "-")


def build_recommendation_rows(
    recommendations: Iterable[dict],
    limit: int | None = None,
    include_analysis: bool = False,
    include_route_id: bool = True,
    include_status: bool = True,
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
            "수량": format_number(rec.get("recommended_qty"), "개"),
            "예상 절감액": format_currency(rec.get("expected_saving")),
            "추천 등급": rec.get("recommendation_grade") or rec.get("grade") or "-",
        }
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
            "수량": format_number(rec.get("recommended_qty"), "개"),
            "예상 절감액": format_currency(rec.get("expected_saving")),
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
