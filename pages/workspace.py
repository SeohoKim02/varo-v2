"""재고 운영 Workspace — the single screen Varo V2 is used from.

One screen answers the whole question: 어디서 → 어디로 → 무슨 상품을 → 몇 개 →
어떤 경로로 → 왜 → 예상 효과와 위험은 무엇인지.

    왼쪽   작업 패널 (데이터 · 분석 · 필터)
    중앙   점포 / DC / 이동 경로 네트워크
    오른쪽 오늘 권장 이동 (선택된 계획 항목의 결정 정보)
    하단   대안 비교 · 검증 · 실행 이력 · 세부정보

Everything shown here comes from ``services.workspace_view``, which reads the one
execution plan the rest of the app already uses. This page never sorts candidates,
never recomputes a quantity, and never prints a value the data does not contain:
missing logistics facts read 미확보 / 데이터 없음, never 0.
"""
from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from components.analysis_progress import AnalysisProgressView, completion_note
from components.candidate_detail import (
    ledger_record,
    render_excluded_candidates,
    render_quantity_basis,
    render_source_locations,
)
from components.execution_history_panel import render_execution_history_panel
from components.state_banner import render_state_action_card, render_state_summary_card
from components.workspace_network import (
    SCOPE_ALL,
    SCOPE_OPTIONS,
    build_workspace_network,
    plan_edge_options,
)
from services.data_application import run_applied_analysis
from services.execution_history import execution_history_metrics
from services.workspace_view import (
    ALL,
    CHECK_NEEDED,
    NO_DATA,
    action_qty,
    alternatives_for,
    apply_filters,
    build_workspace_view,
    filter_options,
    money_text,
    move_reasons,
    move_risks,
    move_title,
    qty_text,
    route_label,
    store_inventory_states,
    validation_rows,
    whatif_rows,
)

DATA_PAGE = "데이터 관리"
VALIDATION_PAGE = "분석 및 검증"
ROUTE_DETAIL_PAGE = "경로 상세"

_MAX_NETWORK_PICKS = 8
_MAX_MOVE_ROWS = 12
STALE_MESSAGE = "현재 결과가 최신 데이터 기준이 아닙니다. 다시 분석하세요."


def _safe(value: Any) -> str:
    return html.escape(str(value)) if value is not None else "-"


def _navigate(page: str) -> None:
    st.session_state["current_menu"] = page


def _select_move(route_id: str) -> None:
    st.session_state["selected_route_id"] = str(route_id)
    st.session_state["simulation_snapshot"] = None


def _history_confirmed() -> int:
    try:
        metrics = execution_history_metrics()
    except Exception:  # pragma: no cover - storage problems must not break the screen
        return 0
    if not metrics.get("ok"):
        return 0
    try:
        return int(metrics.get("confirmed_items") or 0)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Header — 데이터 상태 · 분석 상태 · 분석 실행
# --------------------------------------------------------------------------- #
def _run_analysis() -> None:
    view = AnalysisProgressView(st)
    succeeded = run_applied_analysis(st.session_state, progress_callback=view.callback)
    view.finish(succeeded, st.session_state.get("analysis_elapsed_seconds"))
    st.rerun()


def _render_header(view: Mapping[str, Any]) -> None:
    # 데이터 상태와 분석 상태는 상단 바가 이미 보여준다. 여기서 한 번 더 배지를 그리면
    # 첫 화면에서 네트워크가 그만큼 아래로 밀려나므로 제목 줄만 남긴다.
    st.markdown(
        '<div class="v2-wrap ws-header">'
        '<div class="ws-header-title">재고 운영 Workspace</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.pop("analysis_completed_notice", None):
        st.success(completion_note(st.session_state.get("analysis_elapsed_seconds")))
    if view.get("stale"):
        st.warning(STALE_MESSAGE)
    if st.session_state.get("analysis_run_error"):
        st.error(st.session_state["analysis_run_error"])


def _render_analysis_button(key: str, label: str = "분석 실행", primary: bool = True) -> bool:
    """The one 분석 실행 action. Returns True when the run was triggered."""
    running = bool(st.session_state.get("analysis_running"))
    if st.button(
        label, key=key, type="primary" if primary else "secondary",
        width="stretch", disabled=running,
    ):
        _run_analysis()
        return True
    return False


# --------------------------------------------------------------------------- #
# Left — 데이터 · 분석 · 필터
# --------------------------------------------------------------------------- #
def _render_left_panel(view: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = view.get("readiness_summary") or {}
    st.markdown('<div class="ws-panel-title">데이터</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="v2-wrap v2-card ws-side-card">'
        f'<div class="ws-side-value">{_safe(view.get("data_status"))}</div>'
        f'<div class="v2-card-caption">{_safe(st.session_state.get("uploaded_filename") or "적용된 파일 없음")}</div>'
        f'<div class="v2-card-caption">{_safe(summary.get("headline"))}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if summary.get("missing"):
        st.caption("미확보: " + ", ".join(summary["missing"]))
    st.button("데이터 관리", key="ws_go_data", width="stretch", on_click=_navigate, args=(DATA_PAGE,))

    st.markdown('<div class="ws-panel-title">분석</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="v2-wrap v2-card ws-side-card">'
        f'<div class="ws-side-value">{_safe(view.get("analysis_status"))}</div>'
        f'<div class="v2-card-caption">{_safe(view.get("plan_message") or "분석을 실행하면 권장 이동이 표시됩니다.")}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    # A result is on screen, so re-running is an explicit "recalculate", never the
    # primary call to action.
    _render_analysis_button("ws_run_analysis_side", "분석 다시 실행", primary=False)

    st.markdown('<div class="ws-panel-title">필터</div>', unsafe_allow_html=True)
    options = filter_options(items)
    # 상품과 "순효과가 있는 이동만"은 매일 쓰는 필터라 그대로 두고, 점포·경로 유형은
    # 가끔 쓰는 조건이라 펼침 안으로 넣는다. 기능은 그대로 유지한다.
    filters = {
        "product": st.selectbox("상품", options["product"], key="ws_filter_product"),
        "only_actionable": st.checkbox("순효과가 있는 이동만", key="ws_only_actionable"),
    }
    narrowed = sum(
        1 for key in ("ws_filter_source", "ws_filter_target", "ws_filter_route_type")
        if str(st.session_state.get(key) or "전체") != "전체"
    )
    label = "점포 · 경로로 좁히기" + (f" ({narrowed}개 적용 중)" if narrowed else "")
    with st.expander(label, expanded=bool(narrowed)):
        filters["source"] = st.selectbox("출발 점포", options["source"], key="ws_filter_source")
        filters["target"] = st.selectbox("도착 점포", options["target"], key="ws_filter_target")
        filters["route_type"] = st.selectbox(
            "경로 유형", options["route_type"], key="ws_filter_route_type",
        )
    return filters


# --------------------------------------------------------------------------- #
# Centre — the network
# --------------------------------------------------------------------------- #
def _background_routes() -> list[dict]:
    routes = (st.session_state.get("varo_data") or {}).get("routes")
    if not isinstance(routes, pd.DataFrame) or routes.empty:
        return []
    return routes.to_dict("records")


def _render_network(view: Mapping[str, Any], drawn: Sequence[Mapping[str, Any]]) -> None:
    head, control = st.columns([2.2, 1.5], gap="small")
    head.markdown('<div class="ws-panel-title">재고 이동 네트워크</div>', unsafe_allow_html=True)
    scope = control.selectbox(
        "표시 범위", list(SCOPE_OPTIONS), key="ws_network_scope", label_visibility="collapsed",
    )
    data = st.session_state.get("varo_data") or {}
    states = store_inventory_states(data, drawn)
    network = build_workspace_network(
        data,
        drawn,
        view.get("selected_route_id"),
        scope=scope,
        store_states=states,
        background_routes=_background_routes() if scope == SCOPE_ALL else (),
    )
    if not network.get("ok"):
        st.markdown(
            f'<div class="v2-wrap ws-network-placeholder">{_safe(network.get("message") or NO_DATA)}</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(network["html"], unsafe_allow_html=True)
    if network.get("message"):
        st.caption(network["message"])
    _render_network_picker(drawn, view.get("selected_route_id"))


FILTER_EMPTY_MESSAGE = "현재 필터 조건에 맞는 이동이 없습니다. 필터를 넓히면 네트워크가 다시 표시됩니다."


def _reset_filters() -> None:
    for key, value in (
        ("ws_filter_product", ALL),
        ("ws_filter_source", ALL),
        ("ws_filter_target", ALL),
        ("ws_filter_route_type", ALL),
        ("ws_only_actionable", False),
    ):
        st.session_state[key] = value


def _render_filtered_out_network() -> None:
    """이동이 하나도 남지 않았을 때는 선 없는 네트워크 대신 안내와 다음 행동을 보여준다."""
    st.markdown('<div class="ws-panel-title">재고 이동 네트워크</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="v2-wrap ws-network-placeholder">{_safe(FILTER_EMPTY_MESSAGE)}</div>',
        unsafe_allow_html=True,
    )
    st.button("필터 초기화", key="ws_reset_filters", on_click=_reset_filters)


def _render_network_picker(items: Sequence[Mapping[str, Any]], selected_id: Any) -> None:
    """Pick a route straight from the network area; the right panel follows."""
    options = plan_edge_options(items)[:_MAX_NETWORK_PICKS]
    if len(options) <= 1:
        return
    st.caption("경로를 선택하면 오른쪽 실행계획이 함께 바뀝니다.")
    per_row = 4
    for start in range(0, len(options), per_row):
        row = options[start:start + per_row]
        columns = st.columns(per_row, gap="small")
        for offset, option in enumerate(row):
            columns[offset].button(
                option["label"],
                key=f"ws_pick_{start + offset}",
                width="stretch",
                type="primary" if option["route_id"] == str(selected_id or "") else "secondary",
                on_click=_select_move,
                args=(option["route_id"],),
            )


# --------------------------------------------------------------------------- #
# Right — 오늘 실행할 이동
# --------------------------------------------------------------------------- #
def _render_execution_panel(view: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> None:
    st.markdown('<div class="ws-panel-title">오늘 권장 이동</div>', unsafe_allow_html=True)
    selected = view.get("selected")
    if not selected:
        st.markdown(
            '<div class="v2-wrap v2-empty-state v2-empty-state-compact">'
            "현재 필터 조건에 맞는 이동이 없습니다. 필터를 넓혀 보세요.</div>",
            unsafe_allow_html=True,
        )
        return

    quantity = action_qty(selected)
    net = selected.get("planned_net_benefit")
    if net is None:
        net = selected.get("net_benefit")
    st.markdown(
        '<div class="v2-wrap v2-card ws-action-card">'
        f'<div class="ws-action-route">{_safe(move_title(selected))}</div>'
        f'<div class="ws-action-product">{_safe(selected.get("product_name") or selected.get("product_id") or "-")}</div>'
        f'<div class="ws-action-qty">{_safe(qty_text(quantity))}</div>'
        '<div class="ws-action-grid">'
        f'<div><span>경로</span><strong>{_safe(route_label(selected))}</strong></div>'
        f'<div><span>예상 순효과</span><strong>{_safe(money_text(net))}</strong></div>'
        f'<div><span>안정성</span><strong>{_safe(selected.get("robustness_status") or CHECK_NEEDED)}</strong></div>'
        f'<div><span>실행 상태</span><strong>{_safe(selected.get("feasibility_status") or "추천 가능")}</strong></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    if selected.get("quantity_adjusted"):
        st.caption(
            f"후보 권장 {qty_text(selected.get('recommended_qty'))} 중 "
            f"{qty_text(quantity)}로, 다른 이동과 재고를 함께 고려해 조정했습니다."
        )

    record = ledger_record(view.get("pipeline"), selected.get("route_id"))
    reasons = move_reasons(selected, record)
    st.markdown('<div class="ws-block-title">이 이동을 권장하는 이유</div>', unsafe_allow_html=True)
    if reasons:
        for line in reasons:
            st.markdown(f"- {line}")
    else:
        st.caption("현재 데이터로 설명할 수 있는 이유가 없습니다.")

    risks = move_risks(selected, view.get("readiness") or [])
    st.markdown('<div class="ws-block-title">위험 · 주의</div>', unsafe_allow_html=True)
    if risks:
        for line in risks:
            st.markdown(f"- {line}")
    else:
        st.caption("현재 확인된 주의 사항이 없습니다.")

    st.button(
        "경로 상세 보기", key="ws_open_route_detail", width="stretch",
        on_click=_navigate, args=(ROUTE_DETAIL_PAGE,),
    )
    _render_move_list(items, view.get("selected_route_id"))


def _render_move_list(items: Sequence[Mapping[str, Any]], selected_id: Any) -> None:
    if len(items) <= 1:
        return
    with st.expander(f"다른 이동 보기 ({len(items)}건)", expanded=False):
        for index, item in enumerate(items[:_MAX_MOVE_ROWS]):
            route_id = str(item.get("route_id") or "")
            label = (
                f"{move_title(item)} · "
                f"{item.get('product_name') or item.get('product_id') or '-'} · "
                f"{qty_text(action_qty(item))}"
            )
            st.button(
                label,
                key=f"ws_move_{index}",
                width="stretch",
                type="primary" if route_id == str(selected_id or "") else "secondary",
                on_click=_select_move,
                args=(route_id,),
            )
        if len(items) > _MAX_MOVE_ROWS:
            st.caption(f"먼저 {_MAX_MOVE_ROWS}건을 표시했습니다. 필터로 범위를 좁혀 보세요.")


# --------------------------------------------------------------------------- #
# Bottom — 대안 비교 · 검증 · 실행 이력 · 세부정보
# --------------------------------------------------------------------------- #
def _render_alternatives_tab(view: Mapping[str, Any]) -> None:
    selected = view.get("selected")
    if not selected:
        st.caption("이동을 선택하면 대안을 비교할 수 있습니다.")
        return
    rows = alternatives_for(
        selected, st.session_state.get("varo_recommendations") or [], view.get("plan"),
    )
    st.caption(
        f"{move_title(selected)} · {selected.get('product_name') or '-'} 기준입니다. "
        "같은 도착 점포를 채우는 다른 이동과, 같은 출발 재고를 보낼 수 있는 다른 이동을 함께 봅니다."
    )
    if rows:
        # 기본 사용자가 실제로 비교하는 열만 남긴다. 내부 식별자와 중간 계산값
        # (route_id · 예상 효과)은 표에서 감춘다.
        display = [
            {key: value for key, value in row.items()
             if key not in ("route_id", "선택", "예상 효과")}
            for row in rows
        ]
        st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch")
    else:
        st.caption("현재 데이터에는 비교할 다른 이동이 없습니다.")
    st.markdown('<div class="ws-block-title">조건이 달라지면</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(whatif_rows(view.get("pipeline"), selected)), hide_index=True, width="stretch")


_VALIDATION_HEADLINES = ("계획 제약", "안전재고", "도착 필요 수량", "추천 안정성")


def _render_validation_tab(view: Mapping[str, Any]) -> None:
    rows = validation_rows(view.get("pipeline"))
    by_item = {str(row.get("검증 항목")): row for row in rows}
    # 검증 결과는 먼저 한 줄로 읽히고, 항목별 설명과 숫자는 펼쳐서 본다.
    headline = [by_item[name] for name in _VALIDATION_HEADLINES if name in by_item]
    if headline:
        columns = st.columns(len(headline), gap="medium")
        for column, row in zip(columns, headline):
            column.markdown(
                '<div class="v2-wrap v2-card ws-kpi">'
                f'<div class="ws-kpi-title">{_safe(row.get("검증 항목"))}</div>'
                f'<div class="ws-check-value">{_safe(row.get("결과"))}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
    plan = view.get("plan") or {}
    issues = (plan.get("validation") or {}).get("issues") or []
    if issues:
        st.warning(" · ".join(str(issue) for issue in issues[:3]))
    with st.expander("검증 상세 보기", expanded=False):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        comparison = (view.get("pipeline") or {}).get("plan_comparison") or {}
        labels = {
            "independent_candidates": "후보를 따로 더한 값",
            "constrained_greedy": "단순 이익 순 계획",
            "vhs_optimized_plan": "현재 실행계획",
        }
        rows = []
        for key, label in labels.items():
            values = comparison.get(key) or {}
            if not values:
                continue
            rows.append({
                "비교 기준": label,
                "이동 건수": values.get("selected_candidates") if values.get("selected_candidates") is not None else NO_DATA,
                "총 이동 수량": qty_text(values.get("total_transfer_qty")),
                "총 이동비용": money_text(values.get("total_cost")),
                "총 순효과": money_text(values.get("total_net_benefit")),
                "안전재고 침범": values.get("safety_stock_violations", NO_DATA),
                "도착 필요량 초과": values.get("destination_overfill_violations", NO_DATA),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("같은 데이터로 계산한 세 결과를 그대로 비교한 값입니다.")
        st.button(
            "분석 및 검증 상세로 이동", key="ws_go_validation", on_click=_navigate, args=(VALIDATION_PAGE,),
        )


def _render_details_tab(view: Mapping[str, Any]) -> None:
    st.markdown('<div class="ws-block-title">데이터 상태</div>', unsafe_allow_html=True)
    readiness = [
        {key: value for key, value in row.items() if key != "grade"}
        for row in view.get("readiness") or []
    ]
    if readiness:
        st.dataframe(pd.DataFrame(readiness), hide_index=True, width="stretch")
    st.caption("미확보 항목은 값을 만들지 않고 미확보로 표시합니다.")

    selected = view.get("selected")
    if selected:
        st.markdown('<div class="ws-block-title">이동 수량 근거</div>', unsafe_allow_html=True)
        record = ledger_record(view.get("pipeline"), selected.get("route_id"))
        render_quantity_basis(st, record)
        render_source_locations(st, record)

    plan = view.get("plan") or {}
    unselected = plan.get("unselected_candidates") or []
    if unselected:
        candidates = {
            str(item.get("route_id")): item
            for item in (st.session_state.get("varo_recommendations") or [])
        }
        rows = []
        for entry in unselected:
            candidate = candidates.get(str(entry.get("route_id"))) or {}
            rows.append({
                "상품": candidate.get("product_name") or "-",
                "출발": candidate.get("source_name") or candidate.get("source_id") or "-",
                "도착": candidate.get("target_name") or candidate.get("target_id") or "-",
                "경로": route_label(candidate),
                "제외 이유": entry.get("reason") or "전체 이동계획에서 다른 이동을 우선했습니다.",
            })
        with st.expander(f"실행계획에 포함되지 않은 이동 ({len(rows)}건)", expanded=False):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    render_excluded_candidates(st, view.get("pipeline"))


def _render_detail_tabs(view: Mapping[str, Any]) -> None:
    tabs = st.tabs(["대안 비교", "검증", "실행 이력", "세부정보"])
    with tabs[0]:
        _render_alternatives_tab(view)
    with tabs[1]:
        _render_validation_tab(view)
    with tabs[2]:
        render_execution_history_panel(view.get("plan") or {})
    with tabs[3]:
        _render_details_tab(view)


# --------------------------------------------------------------------------- #
# Non-result states
# --------------------------------------------------------------------------- #
_PLACEHOLDER_TEXT = {
    "no_data": "현재 적용된 데이터가 없습니다. 데이터를 등록한 뒤 분석을 실행하세요.",
    "analysis_pending": "분석을 실행하면 권장 이동과 네트워크가 여기에 표시됩니다.",
    "stale": "데이터가 바뀌었습니다. 다시 분석하면 네트워크가 갱신됩니다.",
    "no_candidates": "현재 조건에서 실행 가능한 이동이 없습니다.",
}


def _render_waiting_state(view: Mapping[str, Any]) -> None:
    home = view.get("home") or {}
    state_code = str(view.get("state_code") or "")
    needs_run = bool(view.get("analysis_pending") or view.get("stale"))
    if needs_run:
        # The next step is on this screen, so the state card carries no navigation
        # button that would compete with 분석 실행.
        render_state_summary_card(home.get("title"), home.get("short_message"))
        run, manage = st.columns([1.2, 1.2], gap="small")
        with run:
            _render_analysis_button("ws_run_analysis_main")
        manage.button(
            "데이터 관리", key="ws_go_data_waiting", width="stretch",
            on_click=_navigate, args=(DATA_PAGE,),
        )
    else:
        render_state_action_card(home, key="workspace_primary_action")

    placeholder = _PLACEHOLDER_TEXT.get(
        state_code, _PLACEHOLDER_TEXT["stale"] if view.get("stale") else _PLACEHOLDER_TEXT["no_data"]
    )
    if state_code == "no_candidates" and view.get("plan_message"):
        placeholder = str(view["plan_message"])
    st.markdown(
        f'<div class="v2-wrap ws-network-placeholder">{_safe(placeholder)}</div>',
        unsafe_allow_html=True,
    )
    if state_code == "no_candidates":
        render_excluded_candidates(st, view.get("pipeline"))
    with st.expander("데이터 상태 자세히 보기", expanded=False):
        readiness = [
            {key: value for key, value in row.items() if key != "grade"}
            for row in view.get("readiness") or []
        ]
        if readiness:
            st.dataframe(pd.DataFrame(readiness), hide_index=True, width="stretch")
        st.caption("아직 확보되지 않은 값은 미확보로 표시하며 임의로 채우지 않습니다.")


def _render_kpis(view: Mapping[str, Any]) -> None:
    cards = view.get("kpis") or []
    if not cards:
        return
    columns = st.columns(len(cards), gap="medium")
    for column, card in zip(columns, cards):
        column.markdown(
            '<div class="v2-wrap v2-card ws-kpi">'
            f'<div class="ws-kpi-title">{_safe(card["title"])}</div>'
            f'<div class="ws-kpi-value">{_safe(card["value"])}</div>'
            f'<div class="ws-kpi-caption">{_safe(card["caption"])}</div>'
            "</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render_workspace_page() -> None:
    view = build_workspace_view(st.session_state, _history_confirmed())
    _render_header(view)

    if not view.get("ready"):
        _render_waiting_state(view)
        return

    items = list(view.get("plan_items") or [])
    _render_kpis(view)
    view = dict(view)

    # Measured on real 1366 / 1600 / 1920 screens: the centre needs ~54% of the row
    # for the network text to stay above 12px once the SVG is scaled to the column.
    left, centre, right = st.columns([0.95, 2.85, 1.5], gap="medium")
    with left:
        filters = _render_left_panel(view, items)
    visible = apply_filters(items, filters)
    if visible:
        if str(view.get("selected_route_id") or "") not in {
            str(item.get("route_id")) for item in visible
        }:
            view["selected_route_id"] = str(visible[0].get("route_id"))
            view["selected"] = dict(visible[0])
    else:
        # 필터가 모든 이동을 걸러냈다면 오늘 권장할 이동도 없다. 이전 선택을 그대로
        # 두면 필터와 어긋나는 이동이 오른쪽에 남으므로 화면에서만 비운다
        # (session의 선택 id는 유지해 필터를 넓히면 그대로 돌아온다).
        view["selected"] = None
    # One selection for the whole app: the network, the right panel, the bottom
    # tabs and the detail pages all read this single id.
    resolved = str(view.get("selected_route_id") or "")
    if resolved and resolved != str(st.session_state.get("selected_route_id") or ""):
        st.session_state["selected_route_id"] = resolved

    with centre:
        if visible:
            _render_network(view, visible)
        else:
            _render_filtered_out_network()
    with right:
        _render_execution_panel(view, visible)

    _render_detail_tabs(view)


__all__ = ["render_workspace_page", "STALE_MESSAGE", "FILTER_EMPTY_MESSAGE"]
