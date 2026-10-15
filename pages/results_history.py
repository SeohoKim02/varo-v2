"""History and result review backed only by real saved DQN artifacts and current results."""
from __future__ import annotations

import html
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_kpi_card, render_page_header, render_section_header
from components.status import badge_html, user_status_label
from components.user_feedback import logger, report_error, user_text
from components.tables import build_recommendation_rows, render_recommendation_table
from pages.recommendations import _render_downloads, _render_selected_recommendation, _render_selection
from pages.route_detail import _render_route_steps
from services.analysis_pipeline import sort_recommendations
from services.dqn_service import OUTPUT_DIR
from services.simulation_history import (
    STATUS_FAILED,
    get_run_routes,
    history_storage_info,
    list_simulation_runs,
    history_db_path,
)
from components.seller_decisions import current_contract, render_product_results, render_summary


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
            logger.exception("Cannot read saved learning result: %s", path)
            st.warning("일부 학습 결과를 읽지 못했습니다. 파일 접근 권한과 저장 상태를 확인한 뒤 다시 조회하세요.")
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
    # Navigation restores this key. Session state is the sole initializer;
    # supplying value= as well triggers Streamlit's duplicate-default warning.
    maximum = max(default_end, date.today())
    saved = st.session_state.get("history_period", (default_start, default_end))
    if isinstance(saved, date):
        saved = (saved, saved)
    if not isinstance(saved, (tuple, list)) or len(saved) > 2 or any(not isinstance(v, date) for v in saved):
        saved = (default_start, default_end)
    normalized = tuple(sorted(max(default_start, min(maximum, v)) for v in saved))
    if "history_period" not in st.session_state or st.session_state["history_period"] != normalized:
        st.session_state["history_period"] = normalized
    controls = st.columns([1.25, 1, 1, 1.45, 0.65], gap="small")
    period = controls[0].date_input(
        "기간",
        min_value=default_start,
        max_value=maximum,
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
    elif isinstance(period, (tuple, list)):
        st.info("조회할 시작일과 종료일을 모두 선택해 주세요.")
        return []
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
        f'<div class="v2-card-head"><div class="v2-card-title">선택 결과</div>{badge_html(selected["status"], "success" if selected["status"] in {"정상", "완료", "검증 통과"} else "warning")}</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )
    try:
        payload = Path(selected["path"]).read_bytes()
    except OSError as exc:
        report_error("history download", exc, "저장 결과를 읽지 못해 다운로드할 수 없습니다. 파일 접근 권한을 확인하고 다시 조회하세요.")
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


def _scope_label(payload: object, limit: int = 3) -> str:
    try:
        values = [str(value) for value in json.loads(str(payload or "[]"))]
    except (ValueError, TypeError):
        return "-"
    if not values:
        return "-"
    head = " · ".join(values[:limit])
    return head if len(values) <= limit else f"{head} 외 {len(values) - limit}곳"


def _filters_label(payload: object) -> str:
    try:
        values = json.loads(str(payload or "{}"))
    except (ValueError, TypeError):
        return "-"
    parts = [f"{user_text(key, '조회 조건')}: {user_text(value, '확인 필요')}" for key, value in values.items() if value not in (None, "")]
    return " · ".join(parts) if parts else "-"


def _simulation_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "실행일시": str(row.get("created_at") or "")[:19].replace("T", " "),
        "실행 조건": _filters_label(row.get("filters_json")),
        "기존 분석 판단(참고)": user_status_label(row.get("final_strategy")),
        "최종 선택 건수": int(row.get("selected_count") or 0),
        "총 이동수량": _number(row.get("total_moved_quantity")) or 0,
        "예상 절감액": _history_saving_display(row),
        "부족재고 감소": _number(row.get("shortage_reduction")) or 0,
        "서비스 충족률(%)": _number(row.get("service_fill_rate")),
        "상태": user_status_label(row.get("status")),
    } for row in runs])


def _history_saving_display(run: Mapping[str, Any]) -> str:
    value = _number(run.get("total_expected_saving"))
    if "샘플" not in str(run.get("data_source_type") or "") or value is None:
        return "미산정 · 비교 근거 미기록"
    return f"{value:,.0f}원 · 예시 조건"


def _history_cost_display(run: Mapping[str, Any]) -> str:
    value = _number(run.get("total_move_cost"))
    if value is None:
        return "미산정 · 자료 부족"
    if "샘플" in str(run.get("data_source_type") or ""):
        return f"{value:,.0f}원 · 예시 조건"
    # Saved history has no verified currency field; do not infer KRW.
    return f"{value:,.0f} (통화 미확인)"


def _render_simulation_detail(run: dict[str, Any]) -> None:
    metrics = (
        ("실행일시", str(run.get("created_at") or "-")[:19].replace("T", " ")),
        ("상태", user_status_label(run.get("status"))),
        ("데이터", run.get("data_name") or "-"),
        ("데이터 종류", run.get("data_source_type") or "-"),
        ("기준 스냅샷/기간", run.get("snapshot_label") or "데이터에 기준일자 열 없음"),
        ("실행 조건", _filters_label(run.get("filters_json"))),
        ("센터/점포 범위", _scope_label(run.get("node_scope_json"))),
        ("상품 범위", _scope_label(run.get("product_scope_json"))),
        ("후보 수", f"{int(run.get('candidate_count') or 0):,}건"),
        ("조건 적용 후보 수", f"{int(run.get('scoped_candidate_count') or 0):,}건"),
        ("실행 가능 후보 수", f"{int(run.get('feasible_candidate_count') or 0):,}건"),
        ("최종 선택 건수", f"{int(run.get('selected_count') or 0):,}건"),
        ("시뮬레이션 반영 이동량", f"{_number(run.get('total_moved_quantity')) or 0:,.0f}"),
        ("총 예상 이동비용", _history_cost_display(run)),
        ("총 예상 절감액", _history_saving_display(run)),
        ("초과재고 감소", f"{_number(run.get('excess_reduction')) or 0:,.0f}"),
        ("부족재고 감소", f"{_number(run.get('shortage_reduction')) or 0:,.0f}"),
        (
            "서비스 충족률",
            f"{_number(run.get('service_fill_rate')):.1f}%"
            if _number(run.get("service_fill_rate")) is not None else "-",
        ),
        ("기존 분석 판단(참고)", user_status_label(run.get("final_strategy"))),
        ("결과 기준", "재고 이동을 가정한 계산 결과 · 실제 실행 아님"),
    )
    if run.get("error_summary"):
        logger.error("Saved simulation failure %s: %s", run.get("run_id"), run["error_summary"])
        metrics = (*metrics, ("실패 안내", "이전 실행을 완료하지 못했습니다. 입력 자료와 저장 권한을 확인한 뒤 다시 실행하세요."))
    rows_html = "".join(
        '<div class="v2-detail-row">'
        f'<span class="v2-card-caption">{html.escape(str(label))}</span>'
        f'<strong>{html.escape(str(value))}</strong></div>'
        for label, value in metrics
    )
    variant = "success" if str(run.get("status")) in {"완료", "COMPLETED", "SUCCESS"} and not run.get("error_summary") else "warning"
    st.markdown(
        '<div class="v2-wrap v2-card">'
        f'<div class="v2-card-head"><div class="v2-card-title">실행 요약</div>'
        f'{badge_html(user_status_label(run.get("status")), variant)}</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )
    routes = get_run_routes(str(run.get("run_id") or ""))
    st.caption("기존 분석 판단의 할인·보류 등은 상품 처리에 대한 참고 판단입니다. 아래 이동 계획의 실행 방식이나 실제 판매 완료를 뜻하지 않습니다.")
    render_section_header(st, "최종 선택 경로", "계산으로 선택한 이동 계획입니다. 실제 이동 기록이 아닙니다.")
    if not routes:
        render_empty_state(st, "저장된 최종 선택 경로가 없습니다", compact=True)
        return
    st.dataframe(
        pd.DataFrame([{
            "순서": row.get("sequence"),
            "경로 번호": row.get("route_id"),
            "상품": row.get("product_name") or row.get("product_id"),
            "출발": row.get("source_id"),
            "도착": row.get("target_id"),
            "경유 DC": row.get("dc_id") or "-",
            "경로 유형": {"DIRECT": "직접 이동", "VIA_DC": "물류센터 경유"}.get(row.get("route_type"), "경로 확인 필요"),
            "기존 분석 판단(참고)": user_status_label(row.get("action")),
            "추천 수량": _number(row.get("recommended_qty")),
            "적용 수량": _number(row.get("applied_qty")),
            "이동 비용": _number(row.get("move_cost")),
            "예상 절감액": _history_saving_display({"total_expected_saving": row.get("expected_saving"), "data_source_type": run.get("data_source_type")}),
            "VARO 순위": _number(row.get("varo_final_rank")),
            "VHS 순위": _number(row.get("vhs_rank")),
            "Greedy 순위": _number(row.get("greedy_rank")),
            "시뮬레이션 반영": "반영" if row.get("executed") else "건너뜀",
            "건너뛴 사유": user_text(row.get("skipped_reason"), "조건 확인 필요" if not row.get("executed") else "-"),
        } for row in routes]),
        hide_index=True,
        width="stretch",
        height=260,
    )


def _render_simulation_history() -> None:
    runs = list_simulation_runs()
    storage = history_storage_info()
    if not runs:
        render_empty_state(
            st,
            "저장된 시뮬레이션 이력이 없습니다",
            "시뮬레이션 화면에서 실행을 완료하면 이곳에 자동으로 쌓입니다.",
        )
        logger.info("Simulation history storage: %s", storage)
        return
    completed = [row for row in runs if str(row.get("status")) in {"완료", "COMPLETED", "SUCCESS"} and not row.get("error_summary")]
    columns = st.columns(3, gap="medium")
    values = (
        ("총 실행 건수", f"{len(runs):,}건", "로컬에 저장된 시뮬레이션 실행"),
        ("정상 완료", f"{len(completed):,}건", "최종 결과가 확정된 실행"),
        ("실제 절감 성과", "미확인", "계산 이력만으로 실제 성과를 판단하지 않습니다"),
    )
    for column, (title, value, caption) in zip(columns, values):
        with column:
            render_kpi_card(st, title, value, caption=caption, compact=True)

    area = st.container(key="simulation_history_workspace")
    list_column, detail_column = area.columns([2.35, 1], gap="medium")
    with list_column:
        render_section_header(st, "시뮬레이션 실행 목록", right=f"총 {len(runs):,}건")
        st.dataframe(_simulation_frame(runs), hide_index=True, width="stretch", height=330)
    with detail_column:
        render_section_header(st, "선택 실행 상세")
        by_id = {str(row.get("run_id")): row for row in runs}
        selected_id = st.selectbox(
            "실행 선택",
            list(by_id),
            format_func=lambda key: (
                f"{str(by_id[key].get('created_at') or '')[:16].replace('T', ' ')} · "
                f"이동 시뮬레이션 · {user_status_label(by_id[key].get('status'))}"
            ),
            key="simulation_history_select",
        )
    _render_simulation_detail(by_id[selected_id])
    st.caption(
        f"저장된 실행 {storage['run_count']:,}건 · 선택 경로 {storage['route_row_count']:,}행"
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
        "로컬에 저장된 시뮬레이션 실행과 DQN 학습 결과를 구분해 확인하세요.",
    )
    current_tab, simulation_tab, dqn_tab = st.tabs(["현재 상품 결과", "시뮬레이션 이력", "DQN 학습/저장 결과"])
    with current_tab:
        contract = current_contract()
        if contract:
            render_summary(contract)
            render_product_results(contract, "seller_history")
        else:
            st.info("현재 적용된 상품별 분석이 없습니다. 홈에서 데이터를 분석해 주세요.")
    with simulation_tab:
        st.caption("저장된 시뮬레이션은 계산 실행 이력입니다. 판매자의 실제 재고 이동·판매 완료를 의미하지 않습니다.")
        _render_simulation_history()
        with st.expander("판매자 판단·실제 결과 기록 (Shadow Ledger)", expanded=False):
            _render_shadow_history()
        with st.expander("현재 시뮬레이션 상세", expanded=False):
            _render_current_result()
    with dqn_tab:
        _render_dqn_history()


def _render_shadow_history() -> None:
    """Read the existing ledger without creating tables or migrating the database."""
    import sqlite3
    path = history_db_path()
    if not path.exists():
        st.info("저장된 판매자 판단 기록이 없습니다.")
        return
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "seller_shadow_decisions" not in tables:
                st.info("저장된 판매자 판단 기록이 없습니다.")
                return
            rows = db.execute("SELECT decision_at,product_id,source_store_id,data_mode,decision_version,payload_json "
                              "FROM seller_shadow_decisions ORDER BY recorded_at DESC LIMIT 500").fetchall()
            count = db.execute("SELECT COUNT(*) FROM seller_shadow_decisions").fetchone()[0]
            outcomes = db.execute("SELECT COUNT(*) FROM seller_shadow_outcomes").fetchone()[0] if "seller_shadow_outcomes" in tables else 0
        st.caption(f"저장된 판단 {count:,}개 버전 · 실제 결과 기록 {outcomes:,}개 · 최근 500개 표시. 테스트·예시 조건은 운영 실적이 아닙니다.")
        from components.seller_decisions import mode_label
        frame = pd.DataFrame([{"판단일": r[0], "상품": r[1], "출발 점포": r[2], "자료 구분": mode_label(r[3])} for r in rows])
        if not frame.empty:
            st.dataframe(frame, hide_index=True, width="stretch")
            st.download_button("저장된 판매자 판단 JSON", json.dumps([json.loads(r[5]) for r in rows], ensure_ascii=False, default=str).encode("utf-8"),
                               file_name="varo_saved_seller_decisions.json", mime="application/json", key="history_shadow_download")
        else:
            st.info("저장된 판매자 판단 기록이 없습니다.")
    except (sqlite3.Error, OSError, ValueError) as exc:
        report_error("seller history read", exc, "판매자 기록을 읽지 못했습니다. 다른 실행이 끝난 뒤 다시 조회하고, 계속되면 저장 권한을 확인하세요. 기존 기록은 변경하지 않았습니다.")


def _render_dqn_history() -> None:
    history = _saved_dqn_runs(str(OUTPUT_DIR))
    _render_history_kpis(history)
    render_section_header(st, "결과 검색", "실제 저장된 학습 결과만 조회합니다.")
    if history:
        with st.container(border=True, key="history_filters"):
            filtered = _filter_history(history)
    else:
        filtered = []
    if not history:
        render_empty_state(st, "저장된 학습 이력이 없습니다", "학습 관리에서 학습을 완료하면 이곳에 표시됩니다.")
    else:
        history_area = st.container(key="history_workspace")
        list_column, detail_column = history_area.columns([2.35, 1], gap="medium")
        with list_column:
            render_section_header(st, "실행 이력 목록", right=f"총 {len(filtered):,}건")
            if filtered:
                st.dataframe(_history_frame(filtered), hide_index=True, width="stretch", height=360)
            else:
                render_empty_state(st, "검색 조건에 맞는 이력이 없습니다", compact=True)
        with detail_column:
            render_section_header(st, "선택 결과 미리보기")
            _render_selected_history(filtered)
