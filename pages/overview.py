"""Home result dashboard for Varo V2."""
from __future__ import annotations

import html
import math
from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_kpi_card, render_section_header
from components.tables import format_currency, format_number
from services.analysis_pipeline import calculate_overview_kpis, sort_recommendations, top_recommendations
from services.app_state import has_app_data, resolve_selected_route_id
from simulation.dynamic_network import (
    build_network_nodes,
    build_route_segments,
    compute_dynamic_layout,
    normalize_route_type,
)

_DIRECT_ROUTE_COLOR = "#2563a6"
_VIA_DC_ROUTE_COLOR = "#c28a00"
_SPEED_SECONDS = {"느림": 24.0, "보통": 14.0, "빠름": 8.0}
_MAX_BACKGROUND_ROUTES = 10
_ROUTE_LANES = (-54.0, 0.0, 54.0, -27.0, 27.0)


def animation_duration_seconds(speed_label: str | None) -> float:
    return _SPEED_SECONDS.get(str(speed_label or "보통"), _SPEED_SECONDS["보통"])


def _safe(value) -> str:
    return html.escape(str(value)) if value is not None else "-"


def _recommendations() -> list[dict]:
    return sort_recommendations(st.session_state.get("varo_recommendations") or [])


def _nodes_from_data(recommendations: list[dict]) -> list[dict]:
    data = st.session_state.get("varo_data") or {}
    return build_network_nodes(data, recommendations)


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
    return compute_dynamic_layout(nodes, sim_routes, width=1200.0, height=760.0, margin=96.0)


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
def _format_kpi_value(key: str, value) -> str:
    if value is None:
        return "-"
    if key in {"total_recommended_qty", "active_route_count"}:
        return format_number(value)
    if key == "total_expected_saving":
        return format_currency(value)
    if key == "average_vhs_score":
        return format_number(value)
    return str(value)


def _render_kpis() -> None:
    data = st.session_state.get("varo_data")
    recommendations = st.session_state.get("varo_recommendations") or []
    validation = st.session_state.get("varo_validation")
    data_available = has_app_data(data, recommendations)
    pipeline_summary = st.session_state.get("pipeline_summary") or {}
    kpis = (
        pipeline_summary
        if data_available and pipeline_summary
        else calculate_overview_kpis(recommendations, validation) if data_available else {}
    )
    summary = getattr(validation, "summary", {}) if validation else {}
    filename = str(st.session_state.get("uploaded_filename") or "데이터 없음")
    store_count = int(summary.get("store_count") or 0) if data_available else 0
    dc_count = int(summary.get("dc_count") or 0) if data_available else 0
    product_count = int(summary.get("product_count") or 0) if data_available else 0
    columns = st.columns([1.55, 1], gap="medium")
    with columns[0]:
        st.markdown(
            '<div class="v2-wrap v2-card v2-kpi-card v2-kpi-card-compact v2-home-data-card">'
            '<div class="v2-card-caption">현재 데이터 정보</div>'
            f'<div class="v2-kpi-value v2-kpi-value-file" title="{_safe(filename)}">'
            f'{_safe(filename if data_available else "데이터 없음")}</div>'
            '<div class="v2-home-data-stats">'
            f'<span>점포 <strong>{store_count}</strong></span>'
            f'<span>DC <strong>{dc_count}</strong></span>'
            f'<span>상품 <strong>{product_count}</strong></span>'
            f'<span>추천 후보 <strong>{format_number(len(recommendations)) if data_available else "-"}</strong></span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with columns[1]:
        render_kpi_card(
            st,
            "전체 예상 절감액",
            _format_kpi_value("total_expected_saving", kpis.get("total_expected_saving")) if data_available else "-",
            caption=f"추천 후보 {format_number(len(recommendations))}건 기준" if data_available else "데이터 적용 후 계산됩니다",
            compact=True,
        )


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


def _svg_label(value: object, css_class: str, y: float, chars_per_line: int) -> str:
    text = str(value or "-")
    if len(text) <= chars_per_line:
        lines = [text]
    else:
        second = text[chars_per_line: chars_per_line * 2]
        if len(text) > chars_per_line * 2:
            second = second[:-1] + "…"
        lines = [text[:chars_per_line], second]
    start_y = y - 7 if len(lines) == 2 else y
    spans = "".join(
        f'<tspan x="0" y="{start_y + index * 15:.2f}">{_safe(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return f'<text class="{css_class}" text-anchor="middle">{spans}</text>'


def _dc_node_svg(node: Mapping[str, object]) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = str(node.get("node_name") or node.get("node_id"))
    stroke = "#78a9d8" if node.get("is_recommended") else "#b28700"
    left, top = -width / 2, -height / 2
    return (
        f'<g class="network-node dc-node" transform="translate({x:.2f} {y:.2f})">'
        f'<title>{_safe(name)} · 물류 허브</title>'
        f'<rect x="{left:.2f}" y="{top + 13:.2f}" width="{width:.2f}" height="{height - 13:.2f}" rx="7" fill="#fff8df" stroke="{stroke}" stroke-width="{3 if node.get("is_recommended") else 2.2}" />'
        f'<path d="M {left - 4:.2f} {top + 15:.2f} L 0 {top - 4:.2f} L {-left + 4:.2f} {top + 15:.2f} Z" fill="#f5df98" stroke="{stroke}" stroke-width="2" />'
        '<rect x="-27" y="4" width="20" height="28" rx="2" fill="#ffffff" stroke="#b28700" />'
        '<rect x="7" y="4" width="20" height="28" rx="2" fill="#ffffff" stroke="#b28700" />'
        f'{_svg_label(name, "node-label dc-label", -height / 2 + 28, 11)}'
        f'<text class="node-type" x="0" y="{height / 2 - 8:.2f}" text-anchor="middle">물류센터 · DC</text>'
        '</g>'
    )


def _store_node_svg(node: Mapping[str, object], total_stores: int) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = str(node.get("node_name") or node.get("node_id"))
    emphasized = bool(node.get("is_recommended"))
    show_label = bool(node.get("show_label", True))
    stroke = "#78a9d8" if emphasized else "#cbd5df"
    fill = "#f4f8ff" if emphasized else "#ffffff"
    limit = 9 if total_stores <= 16 else 7
    left, top = -width / 2, -height / 2
    label = name if show_label else str(node.get("node_id") or "-")
    return (
        f'<g class="network-node store-node" transform="translate({x:.2f} {y:.2f})">'
        f'<title>{_safe(name)} · 점포</title>'
        f'<rect x="{left:.2f}" y="{top + 10:.2f}" width="{width:.2f}" height="{height - 10:.2f}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="{2.6 if emphasized else 1.3}" />'
        f'<path d="M {left + 8:.2f} {top + 10:.2f} L {left + 16:.2f} {top - 2:.2f} L {-left - 16:.2f} {top - 2:.2f} L {-left - 8:.2f} {top + 10:.2f} Z" fill="{stroke}" opacity="0.82" />'
        f'{_svg_label(label, "node-label store-label", -height * 0.08, limit)}'
        f'<text class="node-type" x="0" y="{height / 2 - 7:.2f}" text-anchor="middle">STORE</text>'
        '</g>'
    )


def _truck_icon(color: str, truck_color: str, truck_soft: str, mode_label: str = "트럭", route_label: str = "") -> str:
    """Small truck centered at the origin so it rides a motion path."""
    return (
        f'<rect x="-21" y="-11" width="29" height="18" rx="3" fill="#ffffff" stroke="{truck_color}" stroke-width="2.3"/>'
        f'<rect x="8" y="-11" width="17" height="18" rx="3" fill="{truck_soft}" stroke="{truck_color}" stroke-width="2.3"/>'
        '<rect x="11" y="-7" width="5" height="5" rx="1" fill="#ffffff"/>'
        f'<rect x="-17" y="-6" width="14" height="6" rx="1.5" fill="{truck_soft}" opacity="0.9"/>'
        f'<circle cx="-12" cy="10" r="4.2" fill="#ffffff" stroke="{truck_color}" stroke-width="2.3"/>'
        f'<circle cx="18" cy="10" r="4.2" fill="#ffffff" stroke="{truck_color}" stroke-width="2.3"/>'
        f'<rect x="-21" y="-25" width="42" height="11" rx="5.5" fill="{color}" opacity="0.94"/>'
        f'<text class="vehicle-mode" x="0" y="-17" text-anchor="middle" fill="#ffffff">{_safe(mode_label)}</text>'
        f'<text class="vehicle-route" x="0" y="34" text-anchor="middle" fill="{truck_color}">{_safe(route_label)}</text>'
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


def _edge_point(
    center: tuple[float, float], toward: tuple[float, float], size: tuple[float, float], padding: float = 7.0,
) -> tuple[float, float]:
    dx, dy = toward[0] - center[0], toward[1] - center[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return center
    half_w, half_h = size[0] / 2 + padding, size[1] / 2 + padding
    tx = half_w / abs(dx) if abs(dx) > 1e-9 else float("inf")
    ty = half_h / abs(dy) if abs(dy) > 1e-9 else float("inf")
    scale = min(tx, ty)
    return center[0] + dx * scale, center[1] + dy * scale


def _route_path_points(
    segments: Sequence[Mapping[str, object]],
    positions: dict[str, tuple[float, float]],
    dimensions: dict[str, tuple[float, float]],
    lane_offset: float = 0.0,
) -> list[tuple[float, float]] | None:
    """Trim paths at node edges and shift concurrent trucks into stable lanes."""
    points = _segments_points(segments, positions)
    if not points:
        return None
    node_ids = [str(segments[0]["from_node_id"])] + [str(segment["to_node_id"]) for segment in segments]
    offsets: list[tuple[float, float]] = []
    for index in range(len(points)):
        previous = points[max(0, index - 1)]
        following = points[min(len(points) - 1, index + 1)]
        dx, dy = following[0] - previous[0], following[1] - previous[1]
        length = math.hypot(dx, dy) or 1.0
        offsets.append((-dy / length * lane_offset, dx / length * lane_offset))
    shifted = [(x + offsets[index][0], y + offsets[index][1]) for index, (x, y) in enumerate(points)]
    start = _edge_point(points[0], points[1], dimensions.get(node_ids[0], (0.0, 0.0)))
    end = _edge_point(points[-1], points[-2], dimensions.get(node_ids[-1], (0.0, 0.0)))
    shifted[0] = (start[0] + offsets[0][0], start[1] + offsets[0][1])
    shifted[-1] = (end[0] + offsets[-1][0], end[1] + offsets[-1][1])
    return [(round(x, 2), round(y, 2)) for x, y in shifted]


def _route_path_d(points: Sequence[tuple[float, float]], bend: float) -> str:
    """Build a gentle quadratic path that keeps shared routes visually distinct."""
    if len(points) < 2:
        return ""
    commands = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for index, (start, end) in enumerate(zip(points, points[1:])):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy) or 1.0
        direction = 1.0 if index % 2 == 0 else -0.72
        control_x = (start[0] + end[0]) / 2 + (-dy / length) * bend * direction
        control_y = (start[1] + end[1]) / 2 + (dx / length) * bend * direction
        commands.append(f"Q {control_x:.2f} {control_y:.2f} {end[0]:.2f} {end[1]:.2f}")
    return " ".join(commands)


def _point_along_polyline(points: Sequence[tuple[float, float]], progress: float) -> tuple[float, float]:
    """Return a stable parked-vehicle position along a multi-segment route."""
    if len(points) < 2:
        return points[0] if points else (0.0, 0.0)
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    target = max(0.0, min(1.0, progress)) * sum(lengths)
    for index, ((start, end), length) in enumerate(zip(zip(points, points[1:]), lengths)):
        if target <= length or index == len(lengths) - 1:
            ratio = 0.0 if length == 0 else min(1.0, target / length)
            return start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio
        target -= length
    return points[-1]


@st.cache_data(show_spinner=False, max_entries=24)
def _network_markup_cached(
    data_signature: str,
    nodes: list[dict], sim_routes: list[dict], all_routes: list[dict],
    playing: bool, speed_seconds: float, show_all: bool, selected_id: str,
) -> dict[str, object]:
    _ = data_signature
    layout = _layout_cached(nodes, sim_routes)
    if not layout.is_valid:
        return {"ok": False, "errors": layout.errors, "html": ""}

    all_nodes = list(layout.dcs) + list(layout.stores)
    positions = {str(node["node_id"]): (float(node["x"]), float(node["y"])) for node in all_nodes if node}
    dimensions = {
        str(node["node_id"]): (float(node["width"]), float(node["height"]))
        for node in all_nodes if node
    }
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
        lane_offset = _ROUTE_LANES[idx % len(_ROUTE_LANES)]
        points = _route_path_points(segments, positions, dimensions, lane_offset)
        if not points:
            continue
        is_via = normalize_route_type(route) == "VIA_DC"
        color = _VIA_DC_ROUTE_COLOR if is_via else _DIRECT_ROUTE_COLOR
        selected = str(route.get("route_id")) == selected_id
        bend_sign = -1.0 if idx % 2 else 1.0
        d = _route_path_d(points, bend_sign * (14.0 + abs(lane_offset) * 0.16))
        dash = ' stroke-dasharray="11 8"' if is_via else ""
        width = 5.2 if selected else 2.4
        opacity = 1.0 if selected else 0.38
        path_id = f"rp{idx}"
        route_paths.append(
            f'<path id="{path_id}" data-lane="{lane_offset:.0f}" d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" stroke-linecap="round"{dash} />'
        )
        truck_color, truck_soft, truck_mode = _transport_style(route.get("transport_type"), route.get("route_type"))
        truck = _truck_icon(color, truck_color, truck_soft, truck_mode, f"TOP{idx + 1}")
        vehicle_class = "v2-vehicle v2-vehicle-selected" if selected else "v2-vehicle v2-vehicle-muted"
        if playing:
            phase = idx * speed_seconds / max(1, len(sim_routes))
            vehicles.append(
                f'<g class="{vehicle_class}">{truck}'
                f'<animateMotion dur="{speed_seconds:.1f}s" begin="-{phase:.1f}s" repeatCount="indefinite" rotate="0" '
                f'keyPoints="0;1" keyTimes="0;1" calcMode="linear">'
                f'<mpath xlink:href="#{path_id}"/></animateMotion></g>'
            )
        else:
            sx, sy = _point_along_polyline(points, 0.22 + idx * 0.19)
            vehicles.append(f'<g class="{vehicle_class}" transform="translate({sx:.2f} {sy:.2f})">{truck}</g>')

    node_shapes = [_dc_node_svg(node) for node in layout.dcs]
    node_shapes.extend(_store_node_svg(node, len(layout.stores)) for node in layout.stores)

    network_html = (
        '<div class="v2-network-shell">'
        '<div class="v2-network-legend" aria-label="네트워크 범례">'
        '<span class="v2-legend-item"><span class="v2-legend-line v2-legend-line-direct"></span>실선: 점포 간 직접 이동</span>'
        '<span class="v2-legend-item"><span class="v2-legend-line v2-legend-line-via"></span>점선: 물류센터 경유</span>'
        '<span class="v2-legend-item"><span class="v2-legend-node v2-legend-store"></span>파란 박스: 점포</span>'
        '<span class="v2-legend-item"><span class="v2-legend-node v2-legend-dc"></span>노란 박스: 물류센터</span>'
        '<span class="v2-legend-item"><span class="v2-legend-truck">🚚</span>차량 아이콘: 이동 경로</span>'
        '</div>'
        f'<svg class="v2-network-svg" viewBox="0 0 {canvas["width"]} {canvas["height"]}" '
        'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'preserveAspectRatio="xMidYMid meet" role="img" aria-label="추천 경로 이동 현황">'
        + "".join(background)
        + "".join(route_paths)
        + "".join(vehicles)
        + "".join(node_shapes)
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


def _simulation_routes(recommendations: list[dict], show_all: bool) -> list[dict]:
    limit = 5 if show_all else 3
    routes = top_recommendations(recommendations, limit=limit)
    selected_id = str(st.session_state.get("selected_route_id") or "")
    selected = next((item for item in recommendations if str(item.get("route_id")) == selected_id), None)
    if selected and all(str(item.get("route_id")) != selected_id for item in routes):
        routes = [selected] + routes[: max(0, limit - 1)]
    return routes


def _set_sim_playing(value: bool) -> None:
    st.session_state["home_sim_playing"] = value


def _render_controls() -> bool:
    playing = bool(st.session_state.get("home_sim_playing", False))
    c1, c2, c3, c4, c5 = st.columns([1.25, 1.1, 0.35, 0.85, 0.9], gap="small")
    c1.button("시뮬레이션 실행", width="stretch", key="sim_start", disabled=playing,
              type="primary", on_click=_set_sim_playing, args=(True,))
    c2.button("다시 실행", width="stretch", key="sim_restart",
              on_click=_set_sim_playing, args=(True,))
    c3.markdown('<div class="v2-speed-label">속도</div>', unsafe_allow_html=True)
    speed_options = ["느림", "보통", "빠름"]
    current_speed = st.session_state.get("simulation_speed", "보통")
    speed = c4.selectbox(
        "시뮬레이션 속도", speed_options,
        index=speed_options.index(current_speed) if current_speed in speed_options else 1,
        key="home_speed_select",
        label_visibility="collapsed",
    )
    st.session_state["simulation_speed"] = speed
    show_all = c5.toggle(
        "전체 경로 보기", value=bool(st.session_state.get("show_all_routes", False)),
        key="home_show_all",
        help="끄면 대표 경로 최대 3개, 켜면 최대 5개와 보조 연결선을 표시합니다.",
    )
    st.session_state["show_all_routes"] = show_all
    return bool(show_all)


def render_overview_page() -> None:
    data = st.session_state.get("varo_data")
    recommendations = _recommendations()
    data_available = has_app_data(data, recommendations)
    _render_kpis()

    if not data_available:
        render_empty_state(
            st, "데이터를 업로드하면 결과가 표시됩니다",
            "왼쪽 메뉴 또는 상단에서 엑셀을 업로드하거나 기본 샘플을 불러와주세요.",
        )
        return

    selected_route_id = resolve_selected_route_id(recommendations, st.session_state.get("selected_route_id"))
    if selected_route_id != st.session_state.get("selected_route_id"):
        st.session_state["selected_route_id"] = selected_route_id

    render_section_header(
        st,
        "재고 이동 시뮬레이션",
        "추천된 재고 이동 경로와 점포·물류센터 관계를 표시합니다.",
    )
    show_all = _render_controls()
    sim_routes = _simulation_routes(recommendations, show_all)
    nodes = _nodes_from_data(recommendations)
    all_routes = _network_routes_from_data()
    playing = bool(st.session_state.get("home_sim_playing", False))
    speed_seconds = animation_duration_seconds(st.session_state.get("simulation_speed", "보통"))
    _render_network(nodes, sim_routes, all_routes, playing, speed_seconds, show_all)
