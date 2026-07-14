"""Route detail page for Varo V2."""
from __future__ import annotations

import html

import streamlit as st

from components.cards import render_empty_state, render_page_header, render_section_header
from components.status import badge_html, route_type_badge
from components.tables import format_currency, format_number
from services.analysis_pipeline import find_recommendation, sort_recommendations
from services.app_state import current_data_status, has_app_data, resolve_selected_route_id


def _recommendations() -> list[dict]:
    return sort_recommendations(st.session_state.get("varo_recommendations") or [])


def _apply_route_selection() -> None:
    st.session_state["selected_route_id"] = st.session_state.get("route_detail_select")
    st.session_state["simulation_snapshot"] = None


def _select_route(recommendations: list[dict]) -> dict | None:
    if not recommendations:
        return None
    options = [str(rec["route_id"]) for rec in recommendations]
    rank_by_id = {route_id: index for index, route_id in enumerate(options, start=1)}
    current = resolve_selected_route_id(recommendations, st.session_state.get("selected_route_id"))
    if current != st.session_state.get("selected_route_id"):
        st.session_state["selected_route_id"] = current
    index = options.index(current) if current in options else 0
    if "route_detail_select" in st.session_state and st.session_state.get("route_detail_select") != options[index]:
        st.session_state.pop("route_detail_select", None)
    selected = st.selectbox(
        "추천 후보 선택", options, index=index,
        format_func=lambda route_id: (
            f"{rank_by_id.get(route_id, '-')}순위 · "
            f"{_route_option_label(find_recommendation(recommendations, route_id))}"
        ),
        key="route_detail_select",
        on_change=_apply_route_selection,
    )
    return find_recommendation(recommendations, selected)


def _route_option_label(route: dict | None) -> str:
    if not route:
        return "추천 경로"
    product = route.get("product_name") or "상품"
    source = route.get("source_name") or route.get("source_id") or "-"
    target = route.get("target_name") or route.get("target_id") or "-"
    route_type = "직접 이동" if route.get("route_type") == "DIRECT" else f"DC 경유 · {_route_dc_label(route)}"
    return f"{product} · {source} → {target} · {route_type}"


def _route_dc_label(route: dict) -> str:
    dc_id = str(route.get("dc_id") or "").strip()
    dc_name = str(route.get("dc_name") or "").strip()
    if dc_id and dc_name and dc_id not in dc_name:
        return f"{dc_name} ({dc_id})"
    return dc_name or dc_id or "물류센터"


def _route_description(route: dict) -> str:
    source = route.get("source_name") or route.get("source_id") or "출발 점포"
    target = route.get("target_name") or route.get("target_id") or "도착 점포"
    if route.get("route_type") == "VIA_DC":
        return f"{source}의 재고를 {_route_dc_label(route)}에서 확인한 뒤 {target}로 이동합니다."
    return f"{source}의 재고를 물류센터 경유 없이 {target}로 직접 이동합니다."


def _render_route_summary(route: dict) -> None:
    is_via_dc = route.get("route_type") == "VIA_DC"
    route_type = "DC 경유 (VIA_DC)" if is_via_dc else "직접 이동 (DIRECT)"
    rows = [
        ("상품", route.get("product_name") or route.get("product_id") or "-"),
        ("출발 점포", route.get("source_name") or route.get("source_id") or "-"),
        ("도착 점포", route.get("target_name") or route.get("target_id") or "-"),
        ("경로 유형", route_type),
        ("DC 경유 여부", f"예 · {_route_dc_label(route)}" if is_via_dc else "아니오 · DIRECT"),
        ("추천 수량", format_number(route.get("recommended_qty"), "개")),
        ("예상 절감액", format_currency(route.get("expected_saving"))),
        ("이동 비용", format_currency(route.get("estimated_cost") or route.get("move_cost"))),
        ("이동 거리", format_number(route.get("distance_km"), "km")),
        ("예상 시간", format_number(route.get("expected_time_min") or route.get("travel_time_min"), "분")),
        ("이동 방식", route.get("transport_type") or "일반 트럭"),
        ("경로 설명", _route_description(route)),
    ]
    if is_via_dc:
        rows.insert(5, ("경유 DC", _route_dc_label(route)))
    content = "".join(
        f'<div class="v2-detail-row"><span class="v2-card-caption">{html.escape(label)}</span>'
        f'<strong>{html.escape(str(value))}</strong></div>'
        for label, value in rows
    )
    st.markdown(f'<div class="v2-wrap v2-card">{content}</div>', unsafe_allow_html=True)


def _render_route_steps(route: dict) -> None:
    source = route.get("source_name") or route.get("source_id") or "출발 점포"
    target = route.get("target_name") or route.get("target_id") or "도착 점포"
    if route.get("route_type") == "VIA_DC":
        dc = _route_dc_label(route)
        steps = (
            f"{source}에서 {dc}로 운송",
            f"{dc}에서 도착 물량 확인",
            f"{dc}에서 {target}로 운송",
        )
    else:
        steps = (f"{source}에서 {target}로 직접 운송 (DIRECT)",)
    nodes = [source, target]
    if route.get("route_type") == "VIA_DC":
        nodes = [source, _route_dc_label(route), target]
    flow = '<span class="v2-route-arrow" aria-hidden="true">→</span>'.join(
        f'<span class="v2-route-node{" v2-route-node-dc" if index == 1 and len(nodes) == 3 else ""}">{html.escape(str(node))}</span>'
        for index, node in enumerate(nodes)
    )
    content = "".join(f"<li>{html.escape(str(step))}</li>" for step in steps)
    render_section_header(st, "이동 단계", "선택한 추천이 실제로 거치는 순서를 자체 경로 카드로 표시합니다.")
    st.markdown(f'<div class="v2-wrap v2-route-flow">{flow}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="v2-wrap v2-card"><ol class="v2-route-steps">{content}</ol></div>',
        unsafe_allow_html=True,
    )

def render_route_detail_page() -> None:
    data = st.session_state.get("varo_data")
    recommendations = _recommendations()
    data_available = has_app_data(data, recommendations)
    render_page_header(
        st, "경로 상세", "선택한 추천의 출발·도착 점포와 DC 경유 단계를 확인합니다.",
        badge=badge_html(current_data_status(st.session_state), "accent" if data_available else "neutral"),
    )
    if not data_available:
        render_empty_state(st, "추천 결과가 없습니다")
        return

    route = _select_route(recommendations)
    if not route:
        render_empty_state(st, "선택한 경로를 찾을 수 없습니다", compact=True)
        return
    render_section_header(st, "추천 결과", "", right=route_type_badge(str(route.get("route_type"))))
    _render_route_summary(route)
    _render_route_steps(route)
