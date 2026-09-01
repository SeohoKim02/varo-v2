"""Route detail page for Varo V2."""
from __future__ import annotations

import html

import streamlit as st
import streamlit.components.v1 as components

from components.cards import render_empty_state, render_page_header, render_recommendation_summary, render_section_header
from components.candidate_detail import ledger_record, render_quantity_basis, render_source_locations
from components.tables import format_currency, format_number
from services import upload_quality, v2_summaries
from services.analysis_pipeline import find_recommendation, sort_recommendations
from services.app_state import has_app_data, resolve_selected_route_id
from services.kakao_service import build_kakao_map_html, build_route_payload, get_kakao_key_from_sources
from simulation.route_animation import build_route_legs


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _candidate_detail(route_id: object) -> dict:
    report = st.session_state.get("upload_report") or {}
    return upload_quality.candidate_by_route(report).get(str(route_id), {})


def _reason_detail(route: dict) -> dict:
    reasons = (st.session_state.get("analysis_result") or {}).get("reason_analysis") or {}
    detail = (reasons.get("reasons") or {}).get(str(route.get("route_id")))
    if detail:
        return detail
    return v2_summaries.recommendation_reason(route, st.session_state.get("varo_recommendations") or [])


def _recommendations() -> list[dict]:
    return sort_recommendations(st.session_state.get("varo_recommendations") or [])


def _apply_route_selection() -> None:
    st.session_state["selected_route_id"] = st.session_state.get("route_detail_select")
    st.session_state["simulation_snapshot"] = None


def _route_option_label(recommendations: list[dict], route_id: str) -> str:
    """User-facing label hides the internal route_id: '상품 | 출발 → 도착'."""
    route = find_recommendation(recommendations, route_id)
    if not route:
        return str(route_id)
    source = route.get("source_name") or route.get("source_id") or "-"
    target = route.get("target_name") or route.get("target_id") or "-"
    return f"{route.get('product_name') or '상품'} | {source} → {target}"


def _select_route(recommendations: list[dict]) -> dict | None:
    if not recommendations:
        return None
    options = [str(rec["route_id"]) for rec in recommendations]
    current = resolve_selected_route_id(recommendations, st.session_state.get("selected_route_id"))
    if current != st.session_state.get("selected_route_id"):
        st.session_state["selected_route_id"] = current
    index = options.index(current) if current in options else 0
    if "route_detail_select" in st.session_state and st.session_state.get("route_detail_select") != options[index]:
        st.session_state.pop("route_detail_select", None)
    selected = st.selectbox(
        "경로 선택", options, index=index,
        format_func=lambda route_id: _route_option_label(recommendations, route_id),
        key="route_detail_select",
        on_change=_apply_route_selection,
    )
    return find_recommendation(recommendations, selected)


def _node_name(route: dict, node_id: str) -> str:
    if node_id == str(route.get("source_id")):
        return str(route.get("source_name") or route.get("source_id"))
    if node_id == str(route.get("target_id")):
        return str(route.get("target_name") or route.get("target_id"))
    if node_id == str(route.get("dc_id")):
        return str(route.get("dc_name") or route.get("dc_id"))
    return node_id


def _phase_label(phase: str) -> str:
    return {
        "DIRECT": "출발 점포에서 도착 점포로 직접 이동",
        "TO_DC": "출발 점포에서 DC로 이동",
        "FROM_DC": "DC에서 도착 점포로 이동",
    }.get(phase, "경로 이동")


def _render_steps(route: dict) -> None:
    render_section_header(st, "이동 단계", "")
    for index, leg in enumerate(build_route_legs(route), start=1):
        st.markdown(
            f"""
            <div class="v2-wrap v2-card" style="margin-bottom:0.55rem;">
              <div class="v2-card-title">{index}단계 · {_phase_label(str(leg['phase']))}</div>
              <div class="v2-card-caption">{_node_name(route, str(leg['from_node_id']))} → {_node_name(route, str(leg['to_node_id']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_core_kpis(route: dict) -> None:
    render_section_header(st, "핵심 수치", "")
    row1 = st.columns(3, gap="small")
    row1[0].metric("추천 수량", format_number(route.get("recommended_qty"), "개"))
    row1[1].metric("예상 순효과", format_currency(route.get("net_benefit")))
    row1[2].metric("추천 점수", format_number(route.get("vhs_score"), "점"))
    row2 = st.columns(3, gap="small")
    row2[0].metric("이동 거리", format_number(route.get("distance_km"), "km"))
    row2[1].metric("예상 시간", format_number(route.get("expected_time_min") or route.get("travel_time_min"), "분"))
    row2[2].metric("이동비용", format_currency(route.get("move_cost") or route.get("estimated_cost")))
    st.caption(
        f"예상 절감액 {format_currency(route.get('expected_saving'))}에서 이동비용을 뺀 값이 예상 순효과입니다."
    )


def _render_vhs_components(route: dict) -> None:
    components_list = [
        ("절감 효과", route.get("savings_score")),
        ("폐기 위험", route.get("disposal_risk_score")),
        ("수요 적합도", route.get("demand_fit_score")),
        ("재고 균형", route.get("inventory_balance_score")),
        ("이동 효율", route.get("route_cost_score")),
        ("실행 가능성", route.get("feasibility_score")),
    ]
    if not any(value is not None for _, value in components_list):
        return
    render_section_header(st, "추천 점수 구성", "")
    top = st.columns(3, gap="small")
    bottom = st.columns(3, gap="small")
    for index, (label, value) in enumerate(components_list):
        (top if index < 3 else bottom)[index % 3].metric(label, format_number(value, "점"))
    with st.expander("비교 기준 (Greedy · DQN · Pareto)", expanded=False):
        greedy = route.get("greedy_strategy") or route.get("greedy_action") or "-"
        dqn = route.get("dqn_status") or "학습 전"
        dqn_action = route.get("dqn_action") or "비교 대기"
        pareto = route.get("pareto_status") or "-"
        st.markdown(
            f"- Greedy 기준: {html.escape(str(greedy))}\n"
            f"- DQN: {html.escape(str(dqn))} · {html.escape(str(dqn_action))}\n"
            f"- Pareto: {html.escape(str(pareto))}"
        )


def _render_method_comparison(route: dict) -> None:
    render_section_header(st, "이동 방식 비교", "")
    direct_cost, via_cost = _num(route.get("direct_cost")), _num(route.get("via_dc_cost"))
    direct_time, via_time = _num(route.get("direct_time_min")), _num(route.get("via_dc_time_min"))

    def _verdict(is_direct: bool) -> str:
        marks = []
        if direct_cost is not None and via_cost is not None and direct_cost != via_cost:
            if (is_direct and direct_cost < via_cost) or (not is_direct and via_cost < direct_cost):
                marks.append("비용 우수")
        if direct_time is not None and via_time is not None and direct_time != via_time:
            if (is_direct and direct_time < via_time) or (not is_direct and via_time < direct_time):
                marks.append("시간 우수")
        return " · ".join(marks) if marks else "비교 후보"

    selected_direct = route.get("route_type") == "DIRECT"
    rows = [
        ("직접 이동", route.get("direct_distance_km"), route.get("direct_time_min"), route.get("direct_cost"), _verdict(True), selected_direct),
        ("DC 경유", route.get("via_dc_distance_km"), route.get("via_dc_time_min"), route.get("via_dc_cost"), _verdict(False), not selected_direct),
    ]
    body = []
    for method, dist, minutes, cost, verdict, highlight in rows:
        cls = ' class="v2-row-pick"' if highlight else ""
        body.append(
            f"<tr{cls}><td>{html.escape(method)}</td>"
            f"<td>{html.escape(format_number(dist, 'km'))}</td>"
            f"<td>{html.escape(format_number(minutes, '분'))}</td>"
            f"<td>{html.escape(format_currency(cost))}</td>"
            f"<td>{html.escape(verdict)}</td></tr>"
        )
    st.markdown(
        '<div class="v2-wrap v2-html-table-wrap"><table class="v2-html-table"><thead><tr>'
        "<th>방식</th><th>거리</th><th>예상 시간</th><th>이동비용</th><th>판단</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _condition_state(ok: bool) -> str:
    return "충족" if ok else "확인 필요"


def _render_data_conditions(route: dict) -> None:
    render_section_header(st, "데이터 조건", "")
    distance_ok = _num(route.get("distance_km")) is not None or _num(route.get("distance_cutline_km")) is not None
    time_raw = str(route.get("time_window_status") or "")
    time_ok = bool(time_raw) and "부족" not in time_raw and "없" not in time_raw
    feasible = _num(route.get("feasibility_score"))
    feasible_ok = feasible is None or feasible >= 50
    cols = st.columns(3, gap="small")
    cols[0].metric("거리 조건", _condition_state(distance_ok))
    cols[1].metric("시간 조건", _condition_state(time_ok))
    cols[2].metric("실행 가능성", "가능" if feasible_ok else "확인 필요")
    if not time_ok:
        with st.expander("데이터 확인", expanded=False):
            st.caption("현재 파일에 거래 가능 시간 정보가 없어 거리와 비용 기준으로만 계산했습니다.")


def _pipeline_result() -> dict:
    value = st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result")
    return value if isinstance(value, dict) else {}


def _render_reasons(route: dict) -> None:
    render_section_header(st, "추천 판단 근거", "")
    record = ledger_record(_pipeline_result(), route.get("route_id"))
    detail = _reason_detail(route)
    sentences = (record.get("recommendation_reasons") if record else None) or detail.get("sentences") or [
        route.get("reason") or "추천 사유가 없습니다."
    ]
    for line in sentences[:3]:
        st.markdown(f"- {line}")
    render_quantity_basis(st, record)
    render_source_locations(st, record)


def _render_tech_info(route: dict) -> None:
    with st.expander("기술 정보", expanded=False):
        candidate = _candidate_detail(route.get("route_id"))
        st.markdown(
            f"- 추천 ID: {html.escape(str(route.get('route_id') or '-'))}\n"
            f"- 출발/도착 ID: {html.escape(str(route.get('source_id') or '-'))} → {html.escape(str(route.get('target_id') or '-'))}\n"
            f"- 경유 DC: {html.escape(str(route.get('dc_name') or route.get('dc_id') or '해당 없음'))}\n"
            f"- 시간 조건: {html.escape(str(route.get('time_window_status') or '정보 없음'))}\n"
            f"- 거리 컷라인: {html.escape(format_number(route.get('distance_cutline_km'), 'km'))}"
        )
        if candidate:
            st.caption("경로 선택 근거: " + str(candidate.get("selected_route_basis", "-")))


def _render_kakao_map(route: dict) -> None:
    render_section_header(st, "지도", "")
    key = get_kakao_key_from_sources(st.secrets)
    if not key:
        render_empty_state(st, "지도 키가 설정되면 실제 지도에서 경로를 확인할 수 있습니다.", compact=True)
        return
    payload = build_route_payload(st.session_state.get("varo_data") or {}, route)
    if not payload.ok:
        render_empty_state(st, payload.message or "지도 좌표를 확인할 수 없습니다.", compact=True)
        return
    components.html(build_kakao_map_html(payload, key, height=460), height=480, scrolling=False)


def render_route_detail_page() -> None:
    data = st.session_state.get("varo_data")
    recommendations = _recommendations()
    data_available = has_app_data(data, recommendations)
    render_page_header(st, "경로 상세", "선택한 추천 경로의 이동 조건과 대안을 확인합니다.")
    if not data_available:
        render_empty_state(st, "추천 경로가 선택되면 지도가 표시됩니다.")
        return

    route = _select_route(recommendations)
    if not route:
        render_empty_state(st, "선택한 경로를 찾을 수 없습니다", compact=True)
        return

    render_section_header(st, "선택 경로 요약", "")
    render_recommendation_summary(st, route)

    _render_steps(route)
    _render_core_kpis(route)
    _render_vhs_components(route)
    _render_kakao_map(route)
    _render_method_comparison(route)
    _render_data_conditions(route)
    _render_reasons(route)
    _render_tech_info(route)
