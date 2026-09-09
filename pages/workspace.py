"""재고 운영 Workspace — the single screen Varo V2 is used from.

One screen answers the whole question: 어디서 → 어디로 → 무슨 상품을 → 몇 개 →
어떤 경로로 → 왜 → 예상 효과와 위험은 무엇인지.

    왼쪽   상태 · 실행 계획 목록 (필터 포함)
    중앙   점포 / DC / 이동 경로 네트워크
    오른쪽 오늘 권장 이동 (선택된 계획 항목의 결정 정보)
    하단   대안 비교 · 세부 정보 · 검증 · 실행 이력

There is exactly **one** selection on this screen: ``selected_route_id``. The plan
list writes it, and the network, the right panel, and every bottom tab read it —
no component keeps a candidate of its own. The list itself is a single scrolling
radio group, so 8 or 50 moves cost the same vertical space and one click to
switch between.

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
from components.execution_history_panel import (
    render_execution_history_panel,
    render_record_plan_action,
)
from components.exports import (
    CSV_MIME,
    EMPTY_PLAN_MESSAGE,
    XLSX_MIME,
    render_download,
)
from components.state_banner import render_state_action_card, render_state_summary_card
from components.tables import render_html_table
from components.workspace_network import (
    SCOPE_ALL,
    SCOPE_OPTIONS,
    build_workspace_network,
)
from services import export_service
from styles import WORKSPACE_MAIN_ROW_KEY
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
    logistics_rows,
    money_text,
    move_detail_rows,
    move_reasons,
    move_risks,
    move_title,
    plan_list_rows,
    qty_text,
    route_label,
    store_inventory_states,
    validation_rows,
    whatif_rows,
)

DATA_PAGE = "데이터 관리"
VALIDATION_PAGE = "분석 및 검증"
# Streamlit turns this into a `st-key-…` class on the three-column row; styles.py
# owns the constant and the matching narrow-desktop rules.
MAIN_ROW_KEY = WORKSPACE_MAIN_ROW_KEY

# The plan list is one scrolling radio group, so the row budget below caps its
# *height*, not the number of selectable moves: 50 moves stay one click away.
# 52 is the measured height of one row (name + caption) in Chrome, not a guess:
# at 62 a five-move plan reserved ~75px of empty scroller under the last row and
# pushed the 내보내기 control that far down the column.
_LIST_ROW_PX = 52
_LIST_MIN_PX = 108
_LIST_MAX_PX = 330
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
        '<div class="ws-header-title">재고 운영</div>'
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
# Left — 상태 · 필터 · 실행 계획 목록
# --------------------------------------------------------------------------- #
def _render_status_block(view: Mapping[str, Any]) -> None:
    """데이터 · 분석 상태 in one card, plus the two screens' worth of actions.

    The top bar already carries the two status chips, so this card adds only what
    the bar cannot: which file is applied, how many logistics facts are still
    미확보, and what the current plan says.
    """
    summary = view.get("readiness_summary") or {}
    # 미확보 항목은 카드 안에서 한 번만 말한다(예전에는 "미확보 2개" 요약과 항목 목록이
    # 따로 두 줄을 차지했다).
    missing = summary.get("missing") or []
    missing_line = (
        f'<div class="v2-card-caption">미확보: {_safe(", ".join(missing))}</div>' if missing else ""
    )
    st.markdown('<div class="ws-panel-title">상태</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="v2-wrap v2-card ws-side-card">'
        f'<div class="ws-side-value">{_safe(view.get("data_status"))}</div>'
        f'<div class="v2-card-caption">{_safe(st.session_state.get("uploaded_filename") or "적용된 파일 없음")}</div>'
        f'<div class="ws-side-line">{_safe(view.get("analysis_status"))} · {_safe(summary.get("headline"))}</div>'
        f"{missing_line}"
        "</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="small")
    with left:
        st.button(
            "데이터 관리", key="ws_go_data", width="stretch",
            on_click=_navigate, args=(DATA_PAGE,),
        )
    with right:
        # A result is on screen, so re-running is an explicit "recalculate",
        # never the primary call to action.
        _render_analysis_button("ws_run_analysis_side", "다시 분석", primary=False)


def _render_filters(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The small filter set that narrows the plan list right below it."""
    options = filter_options(items)
    # 상품과 "순효과가 있는 이동만"은 매일 쓰는 조건이라 바로 보이고, 점포·경로 유형·
    # 주의는 가끔 쓰는 조건이라 펼침 안으로 들어간다. 기능은 그대로 유지한다.
    filters: dict[str, Any] = {
        "product": st.selectbox("상품", options["product"], key="ws_filter_product"),
        "only_actionable": st.checkbox("순효과가 있는 이동만", key="ws_only_actionable"),
    }
    narrowed = sum(
        1 for key in ("ws_filter_source", "ws_filter_target", "ws_filter_route_type")
        if str(st.session_state.get(key) or ALL) != ALL
    ) + (1 if st.session_state.get("ws_only_attention") else 0)
    label = "추가 조건" + (f" ({narrowed}개 적용 중)" if narrowed else "")
    with st.expander(label, expanded=bool(narrowed)):
        filters["source"] = st.selectbox("출발 점포", options["source"], key="ws_filter_source")
        filters["target"] = st.selectbox("도착 점포", options["target"], key="ws_filter_target")
        filters["route_type"] = st.selectbox(
            "경로 유형", options["route_type"], key="ws_filter_route_type",
        )
        filters["only_attention"] = st.checkbox("주의가 필요한 이동만", key="ws_only_attention")
    return filters


def _list_height(count: int) -> int:
    return max(_LIST_MIN_PX, min(_LIST_MAX_PX, count * _LIST_ROW_PX + 16))


def _pick_from_list() -> None:
    route_id = st.session_state.get("ws_plan_pick")
    if route_id:
        _select_move(route_id)


def _render_plan_list(items: Sequence[Mapping[str, Any]], selected_id: Any) -> None:
    """One scrolling list of every move in the current plan.

    A radio group rather than a column of buttons: the height is bounded by the
    container whatever the plan size, the selected row carries a real selected
    control (not only a colour), and switching to any other move is one click.
    """
    rows = plan_list_rows(items, selected_id)
    st.markdown(
        f'<div class="ws-panel-title">실행 계획 <span class="ws-count">{len(rows)}건</span></div>',
        unsafe_allow_html=True,
    )
    if not rows:
        st.markdown(
            '<div class="v2-wrap v2-empty-state v2-empty-state-compact">'
            "현재 조건에 맞는 이동이 없습니다.</div>",
            unsafe_allow_html=True,
        )
        return

    options = [row["route_id"] for row in rows]
    labels = {row["route_id"]: row["label"] for row in rows}
    current = str(selected_id or "")
    index = options.index(current) if current in options else 0
    # The widget value is authoritative only while it still names a visible move;
    # a filter change or a selection made on another screen resets it so `index`
    # (which follows selected_route_id) wins again.
    if st.session_state.get("ws_plan_pick") not in options[index:index + 1]:
        st.session_state.pop("ws_plan_pick", None)
    with st.container(height=_list_height(len(rows)), border=False):
        st.radio(
            "실행 계획 목록",
            options,
            index=index,
            format_func=lambda route_id: labels.get(route_id, route_id),
            captions=[row["caption"] for row in rows],
            key="ws_plan_pick",
            label_visibility="collapsed",
            on_change=_pick_from_list,
        )
    st.caption("선택한 이동이 네트워크·오른쪽 패널·아래 탭에 함께 반영됩니다.")


def _render_export(items: Sequence[Mapping[str, Any]]) -> None:
    """실행 계획 내보내기 — folded away, right where the list it exports lives.

    A download must never cost a page change, but it is also not the everyday
    action, so it sits under the plan list in a folded 내보내기 area rather than
    in the 오늘 권장 이동 panel (that panel carries the decision, not tools).

    What leaves here is the *execution plan* as the list currently shows it —
    현재 필터가 적용된 순서 그대로, 실행 수량은 planned_qty. Nothing is re-sorted
    and nothing is recomputed.

    이 화면의 유일한 실행계획 내보내기 진입점이다. 연구용 상세 결과는 분석 및 검증,
    입력 데이터 문제 목록은 데이터 관리가 맡는다.
    """
    if not items:
        # 빈 파일을 내려주지 않고 이유를 말한다.
        st.caption(EMPTY_PLAN_MESSAGE)
        return
    with st.expander(f"내보내기 ({len(items)}건)", expanded=False):
        left, right = st.columns(2, gap="small")
        render_download(
            left, "CSV",
            lambda: export_service.execution_plan_csv_bytes(items),
            export_service.execution_plan_filename("csv"),
            CSV_MIME, "ws_export_plan_csv", width="stretch",
        )
        render_download(
            right, "Excel",
            lambda: export_service.execution_plan_excel_bytes(items),
            export_service.execution_plan_filename("xlsx"),
            XLSX_MIME, "ws_export_plan_xlsx", width="stretch",
        )
        st.caption(
            "화면에 보이는 실행 계획을 그대로 내려받습니다. 수량은 실행 수량 기준이며, "
            "아직 수집 중인 운송 정보는 빈칸 또는 미확보로 남습니다."
        )


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


FILTER_EMPTY_MESSAGE = "현재 필터 조건에 맞는 이동이 없습니다. 필터를 넓히면 네트워크가 다시 표시됩니다."


def _reset_filters() -> None:
    for key, value in (
        ("ws_filter_product", ALL),
        ("ws_filter_source", ALL),
        ("ws_filter_target", ALL),
        ("ws_filter_route_type", ALL),
        ("ws_only_actionable", False),
        ("ws_only_attention", False),
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
        # 예상 순효과 carries one step of emphasis under 실행 수량 (styles.py).
        f'<div class="ws-action-effect"><span>예상 순효과</span>'
        f'<strong>{_safe(money_text(net))}</strong></div>'
        f'<div><span>안정성</span><strong>{_safe(selected.get("robustness_status") or CHECK_NEEDED)}</strong></div>'
        f'<div><span>실행 상태</span><strong>{_safe(selected.get("feasibility_status") or "추천 가능")}</strong></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    if selected.get("quantity_adjusted"):
        st.caption(
            f"원래 권장 {qty_text(selected.get('recommended_qty'))} 중 "
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

    # 세부 재고·비용 값은 아래 '세부 정보' 탭에 모아 둔다. 오른쪽 패널은 실행 판단에
    # 필요한 것만 크게 보여주는 자리다.
    st.caption("재고·비용 세부 값은 아래 세부 정보 탭에서 확인합니다.")

    # 결정을 내린 자리에서 바로 남길 수 있어야 하므로 기록 버튼 하나만 둔다. 실제
    # 실행 결과를 채우는 form은 아래 실행 이력 탭에 그대로 있다.
    render_record_plan_action(view.get("plan"), key="ws_record_plan")


# --------------------------------------------------------------------------- #
# Bottom — 대안 비교 · 세부 정보 · 검증 · 실행 이력
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
        "같은 구간의 다른 경로(직접 이동 · DC 경유 · 다른 DC), 같은 도착 점포를 채우는 다른 출발점, "
        "같은 출발 재고를 보낼 수 있는 다른 도착점을 함께 봅니다. 상품 대체는 비교하지 않습니다."
    )
    if len(rows) <= 1:
        st.caption("현재 데이터에는 이 상품으로 비교할 다른 이동이 없어 선택한 이동만 표시합니다.")
    if rows:
        # 기본 사용자가 실제로 비교하는 열만 남긴다. 내부 식별자와 중간 계산값
        # (route_id · 예상 효과)은 표에서 감춘다.
        display = [
            {key: value for key, value in row.items()
             if key not in ("route_id", "선택", "예상 효과")}
            for row in rows
        ]
        # The in-DOM table, like 세부 정보 next to it: 수량 · 예상 비용 · 예상 순효과
        # line up on their last digit, the header matches every other table in the
        # product, and the row height is the app's rather than the grid's. The
        # virtualised grid stays where a table is genuinely long (연구용 상세).
        render_html_table(display)
    else:
        st.caption("현재 데이터에는 비교할 다른 이동이 없습니다.")
    st.markdown('<div class="ws-block-title">조건이 달라지면</div>', unsafe_allow_html=True)
    render_html_table(whatif_rows(view.get("pipeline"), selected))


_VALIDATION_HEADLINES = ("계획 제약", "안전재고", "도착 필요 수량", "안정성")


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
            "independent_candidates": "이동을 따로 더한 값",
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
    selected = view.get("selected")
    if selected:
        # Everything the 경로 상세 screen used to be opened for, on this screen.
        st.markdown(
            f'<div class="ws-block-title">{_safe(move_title(selected))} · '
            f'{_safe(selected.get("product_name") or selected.get("product_id") or "-")}</div>',
            unsafe_allow_html=True,
        )
        record = ledger_record(view.get("pipeline"), selected.get("route_id"))
        detail, logistics = st.columns([1.35, 1], gap="medium")
        with detail:
            # An in-DOM table, not a virtualised grid: these are the numbers a
            # user reads off the screen and copies into a work order. The ledger's
            # quantity basis is passed in so this table and the 이동 수량 근거 line
            # below it read the same 출발 현재 재고.
            render_html_table(
                move_detail_rows(selected, (record or {}).get("quantity_basis")),
                ["항목", "값"],
            )
        with logistics:
            st.markdown('<div class="ws-block-title">운송 정보</div>', unsafe_allow_html=True)
            render_html_table(logistics_rows(selected), ["항목", "값", "설명"])
            st.caption("아직 수집 중인 값은 0이 아니라 미확보로 표시합니다.")
        st.markdown('<div class="ws-block-title">이동 수량 근거</div>', unsafe_allow_html=True)
        render_quantity_basis(st, record)
        render_source_locations(st, record)

    st.markdown('<div class="ws-block-title">데이터 상태</div>', unsafe_allow_html=True)
    readiness = [
        {key: value for key, value in row.items() if key != "grade"}
        for row in view.get("readiness") or []
    ]
    if readiness:
        st.dataframe(pd.DataFrame(readiness), hide_index=True, width="stretch")
    st.caption("미확보 항목은 값을 만들지 않고 미확보로 표시합니다.")

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


# 실행 판단 → 세부 → 검증 → 기록 순서. 판단에 쓰는 대안 비교가 먼저, 기록은 마지막.
DETAIL_TABS = ("대안 비교", "세부 정보", "검증", "실행 이력")


def _render_detail_tabs(view: Mapping[str, Any]) -> None:
    tabs = st.tabs(list(DETAIL_TABS))
    with tabs[0]:
        _render_alternatives_tab(view)
    with tabs[1]:
        _render_details_tab(view)
    with tabs[2]:
        _render_validation_tab(view)
    with tabs[3]:
        # 기록 버튼은 오른쪽 패널에 있으므로 여기서는 실제 실행 결과 입력만 맡는다.
        st.caption(
            "오른쪽 오늘 권장 이동의 '이 계획 기록'을 누르면 이 탭에서 실제 실행 결과를 입력할 수 있습니다."
        )
        render_execution_history_panel(view.get("plan") or {}, show_record_action=False)


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

    # Measured on real 1366 / 1600 / 1920 screens: the centre still needs ~48% of
    # the row for the network text to stay above ~10px once the SVG is scaled to
    # the column, and the left column needs ~25% to hold a readable plan list.
    #
    # The keyed container is the CSS hook styles.py uses to re-balance these three
    # columns below ~1450px, where an expanded sidebar would otherwise squeeze the
    # network until its labels drop under 10px (see MAIN_ROW_KEY there).
    with st.container(key=MAIN_ROW_KEY):
        left, centre, right = st.columns([1.25, 2.9, 1.45], gap="medium")
    with left:
        _render_status_block(view)
        filters = _render_filters(items)
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

    # Rendered last in the left column so it can already know which move survived
    # the filters — the list, the network and the right panel are one selection.
    with left:
        _render_plan_list(visible, resolved if visible else None)
        _render_export(visible)

    with centre:
        if visible:
            _render_network(view, visible)
        else:
            _render_filtered_out_network()
    with right:
        _render_execution_panel(view, visible)

    _render_detail_tabs(view)


__all__ = ["render_workspace_page", "STALE_MESSAGE", "FILTER_EMPTY_MESSAGE"]
