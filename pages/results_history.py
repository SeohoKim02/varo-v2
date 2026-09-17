"""History and result review backed only by real saved DQN artifacts and current results."""
from __future__ import annotations

import html
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_kpi_card, render_page_header, render_section_header
from components.status import badge_html, user_status_label
from components.tables import build_recommendation_rows, render_recommendation_table
from pages.recommendations import _render_downloads, _render_selected_recommendation, _render_selection
from pages.route_detail import _render_route_steps
from services.analysis_pipeline import sort_recommendations
from services.dqn_service import OUTPUT_DIR


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@st.cache_data(show_spinner=False, ttl=20, max_entries=2)
def _saved_dqn_runs(directory: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("dqn_result_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        timestamp_text = str(payload.get("timestamp") or "")
        try:
            timestamp = datetime.fromisoformat(timestamp_text)
        except ValueError:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime)
        mode = str(payload.get("training_mode") or payload.get("variant") or "original")
        reward = _number((payload.get("reward_summary") or {}).get("last"))
        rows.append({
            "id": path.stem,
            "timestamp": timestamp,
            "display_time": timestamp.strftime("%Y-%m-%d %H:%M"),
            "name": str(payload.get("sample_id") or "현재 데이터"),
            "strategy": "DQN 균형형" if mode == "balanced" else "DQN 원본",
            "candidate_count": int(payload.get("candidate_count") or 0),
            "episodes": int(payload.get("episodes") or 0),
            "learning_rate": _number(payload.get("learning_rate")),
            "reward": reward,
            "confidence": _number(payload.get("average_confidence")),
            "status": user_status_label(
                payload.get("final_status") or payload.get("stability_status") or payload.get("status") or "확인 필요"
            ),
            "path": str(path),
        })
    return rows


def _render_history_kpis(rows: list[dict[str, Any]]) -> None:
    successful = sum(1 for row in rows if row["status"] in {"정상", "완료"})
    rewards = [row["reward"] for row in rows if row["reward"] is not None]
    columns = st.columns(3, gap="medium")
    values = (
        ("총 학습 실행", f"{len(rows):,}건", "실제 저장된 결과 파일 기준"),
        ("정상 완료", f"{successful:,}건", "상태가 정상인 실행"),
        ("평균 최종 보상", f"{sum(rewards) / len(rewards):,.3f}" if rewards else "-", "보상이 기록된 실행 기준"),
    )
    for column, (title, value, caption) in zip(columns, values):
        with column:
            render_kpi_card(st, title, value, caption=caption, compact=True)


def _filter_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    dates = [row["timestamp"].date() for row in rows]
    default_start, default_end = min(dates), max(dates)
    controls = st.columns([1.25, 1, 1, 1.45, 0.65], gap="small")
    period = controls[0].date_input(
        "기간",
        value=(default_start, default_end),
        min_value=default_start,
        max_value=max(default_end, date.today()),
        key="history_period",
    )
    strategy = controls[1].selectbox(
        "전략", ["전체 전략", *sorted({row["strategy"] for row in rows})], key="history_strategy"
    )
    status = controls[2].selectbox(
        "상태", ["전체 상태", *sorted({row["status"] for row in rows})], key="history_status"
    )
    search = controls[3].text_input("검색어", placeholder="샘플 또는 실행명", key="history_search")
    controls[4].button("조회", key="history_search_button", type="primary", width="stretch")
    if isinstance(period, (tuple, list)) and len(period) == 2:
        start_date, end_date = period
    else:
        start_date = end_date = period if isinstance(period, date) else default_end
    needle = str(search or "").strip().lower()
    return [
        row for row in rows
        if start_date <= row["timestamp"].date() <= end_date
        and (strategy == "전체 전략" or row["strategy"] == strategy)
        and (status == "전체 상태" or row["status"] == status)
        and (not needle or needle in row["name"].lower() or needle in row["id"].lower())
    ]


def _history_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "실행일시": row["display_time"],
        "실행명": row["name"],
        "전략": row["strategy"],
        "후보수": row["candidate_count"],
        "Episodes": row["episodes"],
        "최종 보상": row["reward"],
        "상태": row["status"],
    } for row in rows])


def _render_selected_history(rows: list[dict[str, Any]]) -> None:
    if not rows:
        render_empty_state(st, "선택할 이력이 없습니다", compact=True)
        return
    by_id = {row["id"]: row for row in rows}
    selected_id = st.selectbox(
        "상세 결과",
        list(by_id),
        format_func=lambda key: f"{by_id[key]['display_time']} · {by_id[key]['strategy']} · {by_id[key]['name']}",
        key="history_detail_select",
    )
    selected = by_id[selected_id]
    metrics = (
        ("실행명", selected["name"]),
        ("전략", selected["strategy"]),
        ("후보 수", f"{selected['candidate_count']:,}건"),
        ("Episodes", f"{selected['episodes']:,}"),
        ("Learning Rate", selected["learning_rate"] if selected["learning_rate"] is not None else "-"),
        ("평균 신뢰도", f"{selected['confidence']:,.2f}%" if selected["confidence"] is not None else "-"),
    )
    rows_html = "".join(
        '<div class="v2-detail-row">'
        f'<span class="v2-card-caption">{html.escape(str(label))}</span>'
        f'<strong>{html.escape(str(value))}</strong></div>'
        for label, value in metrics
    )
    st.markdown(
        '<div class="v2-wrap v2-card">'
        f'<div class="v2-card-head"><div class="v2-card-title">선택 결과</div>{badge_html(selected["status"], "success")}</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )
    try:
        payload = Path(selected["path"]).read_bytes()
    except OSError:
        payload = None
    if payload:
        st.download_button(
            "저장 결과 다운로드",
            data=payload,
            file_name=Path(selected["path"]).name,
            mime="application/json",
            key="history_result_download",
            width="stretch",
        )


def _render_current_result() -> None:
    recommendations = sort_recommendations(st.session_state.get("varo_recommendations") or [])
    render_section_header(st, "현재 시뮬레이션 결과", "현재 적용된 실제 추천 후보와 경로를 확인하고 내려받습니다.")
    if not recommendations:
        render_empty_state(st, "현재 추천 결과가 없습니다", compact=True)
        return
    render_recommendation_table(
        build_recommendation_rows(recommendations, include_route_id=False, include_status=False),
        key="history_current_recommendations",
        height=300,
    )
    selected = _render_selection(recommendations)
    detail_columns = st.columns([1.25, 1], gap="medium")
    with detail_columns[0]:
        _render_selected_recommendation(selected)
    with detail_columns[1]:
        if selected:
            _render_route_steps(selected)
    _render_downloads(recommendations)


def render_results_history_page() -> None:
    render_page_header(
        st,
        "결과 이력",
        "저장된 학습 실행을 찾고 현재 시뮬레이션 결과를 확인하세요.",
    )
    history = _saved_dqn_runs(str(OUTPUT_DIR))
    _render_history_kpis(history)
    render_section_header(st, "결과 검색", "실제 저장된 학습 결과만 조회합니다.")
    filtered = _filter_history(history)
    if not history:
        render_empty_state(st, "저장된 학습 이력이 없습니다", "학습 관리에서 학습을 완료하면 이곳에 표시됩니다.")
    else:
        list_column, detail_column = st.columns([2.35, 1], gap="medium")
        with list_column:
            render_section_header(st, "실행 이력 목록", right=f"총 {len(filtered):,}건")
            if filtered:
                st.dataframe(_history_frame(filtered), hide_index=True, width="stretch", height=360)
            else:
                render_empty_state(st, "검색 조건에 맞는 이력이 없습니다", compact=True)
        with detail_column:
            render_section_header(st, "선택 결과 미리보기")
            _render_selected_history(filtered)

    with st.expander("현재 시뮬레이션 상세", expanded=not bool(history)):
        _render_current_result()
