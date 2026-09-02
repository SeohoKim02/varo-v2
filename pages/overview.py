"""Home result dashboard for Varo V2."""
from __future__ import annotations

import html
from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_kpi_card, render_page_header, render_section_header
from components.state_banner import render_state_action_card
from components.status import badge_html, route_type_badge
from components.tables import build_home_top_rows, format_currency, format_number, render_recommendation_table
from services.analysis_pipeline import calculate_overview_kpis, sort_recommendations, top_recommendations
from services.app_state import resolve_selected_route_id
from services.execution_plan import planned_recommendations
from services.home_state import READY, build_home_state
from simulation.dynamic_network import (
    build_network_nodes,
    build_route_segments,
    compute_dynamic_layout,
    normalize_route_type,
)

_ROUTE_COLORS = ["#1f766d", "#2d5f9a", "#b28700"]
# Slower, calmer motion than before: one loop of the path in this many seconds.
_SPEED_SECONDS = {"느림": 24.0, "보통": 15.0, "빠름": 9.5}
_MAX_BACKGROUND_ROUTES = 12


def animation_duration_seconds(speed_label: str | None) -> float:
    return _SPEED_SECONDS.get(str(speed_label or "보통"), _SPEED_SECONDS["보통"])


def _safe(value) -> str:
    return html.escape(str(value)) if value is not None else "-"


def _recommendations() -> list[dict]:
    pipeline = st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result") or {}
    if isinstance(pipeline, Mapping) and "execution_plan" in pipeline:
        planned = planned_recommendations(pipeline)
        current = list(st.session_state.get("varo_recommendations") or [])
        current_ids = {str(item.get("route_id") or "") for item in current}
        # A replaced in-memory candidate list makes the cached plan stale.  This
        # guard is mainly for deterministic simulation/test fixtures; normal app
        # data application always refreshes both values atomically.
        if planned and all(str(item.get("route_id") or "") in current_ids for item in planned):
            return planned
        if current_ids:
            return sort_recommendations(current)
        return planned
    return sort_recommendations(st.session_state.get("varo_recommendations") or [])


# Store inventory state → (text color, fill tint). Kept simple for the demo.
_STATE_STYLES: dict[str, tuple[str, str]] = {
    "과잉": ("#b26a1f", "#fdeecb"),
    "부족": ("#b23b3b", "#fbe0e0"),
    "정상": ("#2f7d5b", "#e3f3ea"),
    "이동 대상": ("#2d6fa8", "#e2eefb"),
}


def _store_inventory_states(data: Mapping[str, object], recommendations: list[dict]) -> dict[str, str]:
    """Per-store status (과잉/부족/정상) with recommendation sources marked 이동 대상."""
    states: dict[str, str] = {}
    inventory = (data or {}).get("inventory")
    if isinstance(inventory, pd.DataFrame) and not inventory.empty and "store_id" in inventory.columns:
        stock = pd.to_numeric(inventory.get("stock_qty"), errors="coerce")
        demand = pd.to_numeric(inventory.get("demand_qty"), errors="coerce")
        if demand is None or demand.isna().all():
            demand = pd.to_numeric(inventory.get("sales_30d"), errors="coerce")
        dead = pd.to_numeric(inventory.get("dead_stock_qty"), errors="coerce")
        frame = pd.DataFrame({
            "store_id": inventory["store_id"].astype(str),
            "stock": stock, "demand": demand, "dead": dead,
        })
        grouped = frame.groupby("store_id").sum(min_count=1)
        for store_id, row in grouped.iterrows():
            total_stock = float(row.get("stock") or 0.0)
            total_demand = float(row.get("demand") or 0.0)
            total_dead = float(row.get("dead") or 0.0)
            ratio = total_stock / total_demand if total_demand > 0 else (2.0 if total_stock > 0 else 1.0)
            if (total_stock > 0 and total_dead / total_stock >= 0.30) or ratio >= 1.5:
                states[str(store_id)] = "과잉"
            elif ratio <= 0.7:
                states[str(store_id)] = "부족"
            else:
                states[str(store_id)] = "정상"
    for route in top_recommendations(list(recommendations or []), limit=3):
        source_id = str(route.get("source_id") or "")
        if source_id:
            states[source_id] = "이동 대상"
    return states


def _nodes_from_data(recommendations: list[dict], states: dict[str, str] | None = None) -> list[dict]:
    data = st.session_state.get("varo_data") or {}
    nodes = build_network_nodes(data, recommendations)
    states = states if states is not None else _store_inventory_states(data, recommendations)
    for node in nodes:
        if str(node.get("node_type")).upper() != "DC":
            node["inventory_state"] = states.get(str(node.get("node_id")), "정상")
    return nodes


def _network_routes_from_data() -> list[dict]:
    data = st.session_state.get("varo_data") or {}
    routes = data.get("routes")
    if not isinstance(routes, pd.DataFrame) or routes.empty:
        return []
    return routes.to_dict("records")


def _data_signature() -> str:
    return str(
        st.session_state.get("data_signature")
        or st.session_state.get("uploaded_filename")
        or "empty"
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _layout_cached(nodes: list[dict], sim_routes: list[dict]):
    """Deterministic layout cached by node set + animated route set."""
    return compute_dynamic_layout(nodes, sim_routes)


# --------------------------------------------------------------------------- #
# Navigation + state card (non-result states show one card + one action)
# --------------------------------------------------------------------------- #
def _navigate(page: str, route_id: str | None = None) -> None:
    st.session_state["current_menu"] = page
    if route_id is not None:
        st.session_state["selected_route_id"] = route_id
        st.session_state["simulation_snapshot"] = None


def _render_state_card(home: dict) -> None:
    """A single status card: title, one short message, and one primary action."""
    render_state_action_card(home, key="home_primary_action")


# --------------------------------------------------------------------------- #
# Result KPIs (READY only — never shows 0/"-" placeholders for empty states)
# --------------------------------------------------------------------------- #
def _render_result_kpis(home: dict) -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    pipeline_summary = st.session_state.get("pipeline_summary") or {}
    kpis = pipeline_summary or calculate_overview_kpis(
        recommendations, st.session_state.get("varo_validation")
    )
    pipeline = st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result") or {}
    plan = pipeline.get("execution_plan") if isinstance(pipeline, Mapping) else None
    if not isinstance(plan, Mapping):
        plan = {}
    confidence = home.get("confidence_status") or "계산 불가"
    cards = [
        ("추천 후보", format_number(home.get("recommendation_count")), "오늘 실제로 실행할 이동 수"),
        ("권장 이동 수량", format_number(plan.get("total_transfer_qty", kpis.get("total_recommended_qty"))), "공유 재고를 반영한 실행 수량"),
        ("예상 순효과", format_currency(plan.get("total_net_benefit", kpis.get("total_net_benefit"))), "실행계획의 이동 비용을 뺀 기대 효과"),
        ("추천 신뢰도", str(confidence), "실행 가능성과 순위 안정성 기준"),
        ("데이터 상태", str(home.get("data_status") or "확인 필요"), "현재 사용 중인 데이터"),
    ]
    cols = st.columns(5, gap="medium")
    for idx, (title, value, desc) in enumerate(cards):
        with cols[idx]:
            render_kpi_card(st, title, value, caption=desc, compact=True)


def _render_flow(current_index: int = 3) -> None:
    """A single thin progress row that highlights only the current step."""
    steps = ["엑셀 업로드", "재고 분석", "이동 추천", "결과 확인"]
    cells = '<span class="v2-flow-arrow">→</span>'.join(
        f'<span class="v2-flow-item{" v2-flow-current" if index == current_index else ""}">{_safe(label)}</span>'
        for index, label in enumerate(steps)
    )
    st.markdown(f'<div class="v2-flow-row">{cells}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Simulation (CSS/SMIL — no Python rerun loop)
# --------------------------------------------------------------------------- #
def _transport_style(transport_type: object, route_type: object = None) -> tuple[str, str, str]:
    label = str(transport_type or "")
    route = str(route_type or "")
    if "냉동" in label or "냉장" in label:
        return "#2d6fa8", "#e7f1fb", "냉장"
    if "소형" in label:
        return "#49736b", "#e7f2ef", "소형"
    if "긴급" in label or "DIRECT" in route:
        return "#b2762b", "#fff1dc", "직송"
    return "#596574", "#eef1f4", "트럭"


def _short_label(value: object, limit: int) -> str:
    text = str(value or "-")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _wrap_two_lines(value: object, limit: int) -> list[str]:
    """Split a long node name into up to two centered lines (no mid-word ellipsis)."""
    text = str(value or "-").strip()
    if len(text) <= limit:
        return [text]
    mid = len(text) // 2
    split = -1
    for offset in range(mid):
        for pos in (mid - offset, mid + offset):
            if 0 < pos < len(text) and text[pos] == " ":
                split = pos
                break
        if split != -1:
            break
    if split == -1:
        first, second = text[:limit], text[limit:]
    else:
        first, second = text[:split].strip(), text[split + 1:].strip()
    if len(second) > limit:
        second = second[: max(1, limit - 1)] + "…"
    return [first, second]


def _dc_node_svg(node: Mapping[str, object]) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = str(node.get("node_name") or node.get("node_id"))
    stroke = "#d88378" if node.get("is_recommended") else "#b28700"
    left, top = -width / 2, -height / 2
    return (
        f'<g class="network-node dc-node" transform="translate({x:.2f} {y:.2f})">'
        f'<title>{_safe(name)} · 물류 허브</title>'
        f'<rect x="{left:.2f}" y="{top + 13:.2f}" width="{width:.2f}" height="{height - 13:.2f}" rx="7" fill="#fff8df" stroke="{stroke}" stroke-width="{3 if node.get("is_recommended") else 2.2}" />'
        f'<path d="M {left - 4:.2f} {top + 15:.2f} L 0 {top - 4:.2f} L {-left + 4:.2f} {top + 15:.2f} Z" fill="#f5df98" stroke="{stroke}" stroke-width="2" />'
        '<rect x="-27" y="4" width="20" height="28" rx="2" fill="#ffffff" stroke="#b28700" />'
        '<rect x="7" y="4" width="20" height="28" rx="2" fill="#ffffff" stroke="#b28700" />'
        f'<text class="node-label dc-label" x="0" y="{-height / 2 + 29:.2f}" text-anchor="middle">{_safe(_short_label(name, 20))}</text>'
        f'<text class="node-type" x="0" y="{height / 2 - 8:.2f}" text-anchor="middle">물류센터 · DC</text>'
        '</g>'
    )


def _store_node_svg(node: Mapping[str, object], total_stores: int) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = str(node.get("node_name") or node.get("node_id"))
    emphasized = bool(node.get("is_recommended"))
    show_label = bool(node.get("show_label", True))
    state = str(node.get("inventory_state") or "정상")
    text_color, fill_tint = _STATE_STYLES.get(state, _STATE_STYLES["정상"])
    stroke = "#d88378" if emphasized else (text_color if state != "정상" else "#cbd5df")
    body_fill = "#fff7f5" if emphasized else fill_tint
    limit = 12 if total_stores <= 16 else 9
    left, top = -width / 2, -height / 2
    source = name if show_label else str(node.get("node_id") or "")
    lines = _wrap_two_lines(source, limit)
    if len(lines) == 1:
        label_svg = (
            f'<text class="node-label store-label" x="0" y="{-height * 0.06:.2f}" '
            f'text-anchor="middle">{_safe(lines[0])}</text>'
        )
    else:
        label_svg = (
            '<text class="node-label store-label" text-anchor="middle">'
            f'<tspan x="0" y="{-height * 0.17:.2f}">{_safe(lines[0])}</tspan>'
            f'<tspan x="0" y="{-height * 0.17 + 15:.2f}">{_safe(lines[1])}</tspan>'
            '</text>'
        )
    pill_w = 62.0
    return (
        f'<g class="network-node store-node" transform="translate({x:.2f} {y:.2f})">'
        f'<title>{_safe(name)} · 점포 · 재고 {_safe(state)}</title>'
        f'<rect x="{left:.2f}" y="{top + 10:.2f}" width="{width:.2f}" height="{height - 10:.2f}" rx="7" fill="{body_fill}" stroke="{stroke}" stroke-width="{2.6 if emphasized else 1.3}" />'
        f'<path d="M {left + 8:.2f} {top + 10:.2f} L {left + 16:.2f} {top - 2:.2f} L {-left - 16:.2f} {top - 2:.2f} L {-left - 8:.2f} {top + 10:.2f} Z" fill="{stroke}" opacity="0.82" />'
        f'{label_svg}'
        f'<rect x="{-pill_w / 2:.2f}" y="{height / 2 - 21:.2f}" width="{pill_w:.2f}" height="15.5" rx="7.75" fill="{fill_tint}" stroke="{text_color}" stroke-width="0.9" />'
        f'<text x="0" y="{height / 2 - 9.6:.2f}" text-anchor="middle" fill="{text_color}" font-size="9.5" font-weight="700">{_safe(state)}</text>'
        '</g>'
    )


def _truck_icon(color: str, truck_color: str, truck_soft: str, mode_label: str = "트럭", route_label: str = "") -> str:
    """Small truck centered at the origin so it rides a motion path."""
    return (
        f'<rect x="-18" y="-10" width="25" height="16" rx="3" fill="#ffffff" stroke="{truck_color}" stroke-width="2.2"/>'
        f'<rect x="7" y="-10" width="15" height="16" rx="3" fill="{truck_soft}" stroke="{truck_color}" stroke-width="2.2"/>'
        '<rect x="9" y="-6" width="4.4" height="4.4" rx="1" fill="#ffffff"/>'
        f'<rect x="-15" y="-5" width="12" height="5" rx="1.5" fill="{truck_soft}" opacity="0.9"/>'
        f'<circle cx="-10" cy="8" r="3.8" fill="#ffffff" stroke="{truck_color}" stroke-width="2.2"/>'
        f'<circle cx="15" cy="8" r="3.8" fill="#ffffff" stroke="{truck_color}" stroke-width="2.2"/>'
        f'<rect x="-18" y="-22" width="36" height="10" rx="5" fill="{color}" opacity="0.92"/>'
        f'<text class="vehicle-mode" x="0" y="-14.4" text-anchor="middle" fill="#ffffff">{_safe(mode_label)}</text>'
        f'<text class="vehicle-route" x="0" y="23" text-anchor="middle" fill="{truck_color}">{_safe(route_label)}</text>'
    )


def _segments_points(segments: Sequence[Mapping[str, object]], positions: dict[str, tuple[float, float]]):
    points: list[tuple[float, float]] = []
    for index, seg in enumerate(segments):
        start = positions.get(str(seg["from_node_id"]))
        end = positions.get(str(seg["to_node_id"]))
        if not start or not end:
            return None
        if index == 0:
            points.append(start)
        points.append(end)
    return points if len(points) >= 2 else None


@st.cache_data(show_spinner=False, max_entries=24)
def _network_markup_cached(
    data_signature: str,
    nodes: list[dict], sim_routes: list[dict], all_routes: list[dict],
    playing: bool, speed_seconds: float, show_all: bool, selected_id: str,
) -> dict[str, object]:
    _ = data_signature
    layout = compute_dynamic_layout(nodes, sim_routes)
    if not layout.is_valid:
        return {"ok": False, "errors": layout.errors, "html": ""}

    all_nodes = list(layout.dcs) + list(layout.stores)
    positions = {str(node["node_id"]): (float(node["x"]), float(node["y"])) for node in all_nodes if node}
    canvas = layout.canvas

    background: list[str] = []
    if show_all:
        seen: set[tuple[str, str]] = set()
        for route in all_routes[:_MAX_BACKGROUND_ROUTES]:
            try:
                segments = build_route_segments(route, nodes)
            except ValueError:
                continue
            for seg in segments:
                pair = tuple(sorted((str(seg["from_node_id"]), str(seg["to_node_id"]))))
                if pair in seen or pair[0] == pair[1]:
                    continue
                seen.add(pair)
                s, e = positions.get(pair[0]), positions.get(pair[1])
                if s and e:
                    background.append(
                        f'<line x1="{s[0]:.2f}" y1="{s[1]:.2f}" x2="{e[0]:.2f}" y2="{e[1]:.2f}" '
                        'stroke="#7f8b99" stroke-width="1" stroke-opacity="0.10" />'
                    )

    route_paths: list[str] = []
    vehicles: list[str] = []
    for idx, route in enumerate(sim_routes):
        try:
            segments = build_route_segments(route, nodes)
        except ValueError:
            continue
        points = _segments_points(segments, positions)
        if not points:
            continue
        color = _ROUTE_COLORS[idx % len(_ROUTE_COLORS)]
        is_via = normalize_route_type(route) == "VIA_DC"
        selected = str(route.get("route_id")) == selected_id
        d = "M " + " L ".join(f"{px:.2f} {py:.2f}" for px, py in points)
        dash = ' stroke-dasharray="11 8"' if is_via else ""
        width = 4.0 if selected else 2.8
        opacity = 0.95 if selected else 0.74 if idx == 0 else 0.52
        path_id = f"rp{idx}"
        route_paths.append(
            f'<path id="{path_id}" d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" stroke-linecap="round"{dash} />'
        )
        truck_color, truck_soft, truck_mode = _transport_style(route.get("transport_type"), route.get("route_type"))
        truck = _truck_icon(color, truck_color, truck_soft, truck_mode, f"TOP{idx + 1}")
        if playing:
            vehicles.append(
                f'<g class="v2-vehicle">{truck}'
                f'<animateMotion dur="{speed_seconds:.1f}s" repeatCount="indefinite" rotate="0" '
                f'keyPoints="0;1" keyTimes="0;1" calcMode="linear">'
                f'<mpath xlink:href="#{path_id}"/></animateMotion></g>'
            )
        else:
            sx, sy = points[0]
            vehicles.append(f'<g class="v2-vehicle" transform="translate({sx:.2f} {sy:.2f})">{truck}</g>')

    node_shapes = [_dc_node_svg(node) for node in layout.dcs]
    node_shapes.extend(_store_node_svg(node, len(layout.stores)) for node in layout.stores)

    state_legend = "".join(
        f'<span class="v2-legend-state"><span class="v2-legend-dot" style="background:{fill};border-color:{color};"></span>{state}</span>'
        for state, (color, fill) in _STATE_STYLES.items()
    )
    network_html = (
        '<div class="v2-network-shell">'
        '<div class="v2-network-legend">'
        '<span class="v2-legend-line"></span><span>직접 이동</span>'
        '<span class="v2-legend-line v2-legend-line-dashed"></span><span>DC 경유</span>'
        f'{state_legend}'
        '</div>'
        f'<svg class="v2-network-svg" viewBox="0 0 {canvas["width"]} {canvas["height"]}" '
        'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'preserveAspectRatio="xMidYMid meet" role="img" aria-label="추천 경로 이동 현황">'
        + "".join(background)
        + "".join(route_paths)
        + "".join(node_shapes)
        + "".join(vehicles)
        + "</svg></div>"
    )
    return {"ok": True, "errors": [], "html": network_html}


def _render_network(
    nodes: list[dict], sim_routes: list[dict], all_routes: list[dict],
    playing: bool, speed_seconds: float, show_all: bool,
) -> None:
    if not nodes or not sim_routes:
        st.markdown('<div class="v2-network-placeholder">데이터가 업로드되지 않았습니다</div>', unsafe_allow_html=True)
        return
    result = _network_markup_cached(
        _data_signature(),
        nodes,
        sim_routes,
        all_routes,
        playing,
        float(speed_seconds),
        show_all,
        str(st.session_state.get("selected_route_id") or ""),
    )
    if not result.get("ok"):
        render_empty_state(st, "네트워크를 표시할 수 없습니다", " / ".join(result.get("errors") or []), compact=True)
        return
    st.markdown(str(result.get("html") or ""), unsafe_allow_html=True)


def _render_running_routes(routes: list[dict], states: dict[str, str] | None = None) -> None:
    if not routes:
        st.markdown(
            '<div class="v2-wrap v2-card"><div class="v2-card-title">현재 이동 중</div>'
            '<div class="v2-card-caption">표시할 추천 경로가 없습니다.</div></div>',
            unsafe_allow_html=True,
        )
        return
    playing = bool(st.session_state.get("home_sim_playing"))
    items = []
    for route in routes:
        route_label = "DC 경유" if route.get("route_type") == "VIA_DC" else "직접 이동"
        move_badge = badge_html("이동 중", "accent") if playing else route_type_badge(str(route.get("route_type")))
        items.append(
            '<div class="v2-running-route">'
            f'<strong>{_safe(route.get("product_name"))}</strong>'
            '<div class="v2-running-route-meta">'
            f'<span>{_safe(route.get("source_name") or route.get("source_id"))} → {_safe(route.get("target_name") or route.get("target_id"))}</span>'
            f'<span>방식 {_safe(route_label)}</span>'
            f'<span>수량 {_safe(format_number(route.get("planned_qty") if route.get("planned_qty") is not None else route.get("recommended_qty"), "개"))}</span>'
            f'<span>예상 절감 {_safe(format_currency(route.get("expected_saving")))}</span>'
            '</div>'
            f'<div style="margin-top:0.3rem;">{move_badge}</div>'
            '</div>'
        )
    st.markdown(
        '<div class="v2-wrap v2-card"><div class="v2-card-title">현재 이동 중</div>'
        + "".join(items) + "</div>",
        unsafe_allow_html=True,
    )


def _set_sim_playing(value: bool) -> None:
    st.session_state["home_sim_playing"] = value


def _render_controls() -> None:
    playing = bool(st.session_state.get("home_sim_playing", False))
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.2, 1.5], gap="small")
    c1.button("시작", width="stretch", key="sim_start", disabled=playing,
              on_click=_set_sim_playing, args=(True,))
    c2.button("일시정지", width="stretch", key="sim_pause", disabled=not playing,
              on_click=_set_sim_playing, args=(False,))
    c3.button("다시 시작", width="stretch", key="sim_restart",
              on_click=_set_sim_playing, args=(True,))
    speed_options = ["느림", "보통", "빠름"]
    current_speed = st.session_state.get("simulation_speed", "보통")
    speed = c4.selectbox(
        "속도", speed_options,
        index=speed_options.index(current_speed) if current_speed in speed_options else 1,
        key="home_speed_select", label_visibility="collapsed",
    )
    st.session_state["simulation_speed"] = speed
    show_all = c5.checkbox(
        "전체 경로 보기", value=bool(st.session_state.get("show_all_routes", False)),
        key="home_show_all",
    )
    st.session_state["show_all_routes"] = show_all


def _render_home_top(top_routes: list[dict]) -> None:
    render_section_header(st, "오늘 권장 이동 · 추천 Top 5", "")
    if not top_routes:
        render_empty_state(st, "추천 결과가 없습니다", compact=True)
        return
    render_recommendation_table(build_home_top_rows(top_routes), key="overview_home_top", height=225)


def _render_best_recommendation_card(top: dict, confidence: str | None) -> None:
    """The single top recommendation from the shared ranking + a 상세 보기 action.

    Uses the already-ranked recommendation (no re-sorting here) and hides every
    internal id/score/signature; only what an operator needs to act.
    """
    render_section_header(st, "최우선 추천", "")
    route_label = "DC 경유" if top.get("route_type") == "VIA_DC" else "직접 이동"
    dc_line = ""
    if top.get("route_type") == "VIA_DC" and (top.get("dc_name") or top.get("dc_id")):
        dc_line = f'<span>경유 DC {_safe(top.get("dc_name") or top.get("dc_id"))}</span>'
    reason = ""
    reasons = top.get("recommendation_reasons") or []
    if reasons:
        reason = str(reasons[0])
    elif top.get("reason"):
        reason = str(top.get("reason"))
    reason_html = f'<div class="v2-card-caption" style="margin-top:0.55rem;">{_safe(reason)}</div>' if reason else ""
    st.markdown(
        '<div class="v2-wrap v2-card">'
        '<div class="v2-card-head"><div>'
        f'<div class="v2-card-title">{_safe(top.get("product_name"))}</div>'
        f'<div class="v2-card-caption">{_safe(top.get("source_name") or top.get("source_id"))} → '
        f'{_safe(top.get("target_name") or top.get("target_id"))}</div>'
        f'</div><div>{route_type_badge(str(top.get("route_type")))}</div></div>'
        '<div class="v2-running-route-meta" style="grid-template-columns:repeat(4,minmax(0,1fr));margin-top:0.6rem;">'
        f'<span>방식 {_safe(route_label)}</span>'
        f'{dc_line}'
        f'<span>수량 {_safe(format_number(top.get("planned_qty") if top.get("planned_qty") is not None else top.get("recommended_qty"), "개"))}</span>'
        f'<span>예상 순효과 {_safe(format_currency(top.get("net_benefit")))}</span>'
        f'<span>추천 안정성 {_safe(top.get("robustness_status") or "-")}</span>'
        f'<span>추천 신뢰도 {_safe(confidence or "-")}</span>'
        '</div>'
        f'{reason_html}'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1.4, 4], gap="small")
    cols[0].button(
        "추천 상세 보기",
        key="home_detail_action",
        type="primary",
        width="stretch",
        on_click=_navigate,
        args=("경로 상세", str(top.get("route_id") or "")),
    )


def render_overview_page() -> None:
    render_page_header(st, "Varo 운영 결과", "재고 이동 추천과 예상 절감 효과를 확인합니다.")
    home = build_home_state(st.session_state)

    # Non-result states: one status card + one action, no result KPIs/network/tables.
    if home.get("state_code") != READY:
        _render_state_card(home)
        return

    # A new upload has been inspected while the current result stays in use — a short
    # notice only; the applied result below is unchanged until the user applies it.
    if home.get("pending_notice"):
        st.info("검사 완료된 새 데이터가 있습니다. 데이터 관리에서 적용하세요.")

    data = st.session_state.get("varo_data")
    recommendations = _recommendations()
    _render_result_kpis(home)
    _render_flow()

    selected_route_id = resolve_selected_route_id(recommendations, st.session_state.get("selected_route_id"))
    if selected_route_id != st.session_state.get("selected_route_id"):
        st.session_state["selected_route_id"] = selected_route_id

    top5_routes = recommendations[:5]
    sim_routes = recommendations[:3]
    states = _store_inventory_states(data or {}, recommendations)
    nodes = _nodes_from_data(recommendations, states)
    all_routes = _network_routes_from_data()

    render_section_header(st, "추천 경로 이동 현황", "")
    _render_controls()
    playing = bool(st.session_state.get("home_sim_playing", False))
    speed_seconds = animation_duration_seconds(st.session_state.get("simulation_speed", "보통"))
    show_all = bool(st.session_state.get("show_all_routes", False))
    network_col, routes_col = st.columns([4.3, 1.3], gap="medium")
    with network_col:
        _render_network(nodes, sim_routes, all_routes, playing, speed_seconds, show_all)
    with routes_col:
        _render_running_routes(sim_routes, states)

    _render_home_top(top5_routes)
    top = home.get("top_recommendation")
    if top:
        _render_best_recommendation_card(top, home.get("confidence_status"))
