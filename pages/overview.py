"""Compact home dashboard and data-backed inventory movement simulation."""
from __future__ import annotations

import copy
import html
import math
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_kpi_card, render_section_header
from components.tables import format_currency, format_number
from services.analysis_pipeline import calculate_overview_kpis, sort_recommendations, top_recommendations
from services.app_state import has_app_data, resolve_selected_route_id
from services.inventory_transition_service import (
    INVENTORY_TRANSITION_VERSION,
    cached_inventory_scenario,
    inventory_transition_cache_info,
)
from simulation.dynamic_network import (
    SIMULATION_LAYOUT_VERSION,
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
_STATE_STYLE = {
    "초과재고": ("#fff3d6", "#8a5a00", "#d69b24"),
    "적정재고": ("#e5f5ea", "#24653b", "#67ad7a"),
    "부족재고": ("#e8f1fb", "#245c8d", "#6b9fca"),
    "데이터 부족": ("#eef1f4", "#5d6672", "#aab2bc"),
}
_SIMULATION_COUNTERS = {"layout_calculations": 0, "svg_html_generations": 0}


def animation_duration_seconds(speed_label: str | None) -> float:
    return _SPEED_SECONDS.get(str(speed_label or "보통"), _SPEED_SECONDS["보통"])


def _safe(value: object) -> str:
    return html.escape(str(value)) if value is not None else "-"


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _qty(value: object) -> str:
    number = _number(value)
    if number is None:
        return "데이터 없음"
    if abs(number - round(number)) <= 1e-9:
        return f"{int(round(number)):,}개"
    return f"{number:,.1f}개"


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


def _layout_node_payload(nodes: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Keep view-only inventory labels out of the static layout cache key."""
    fields = (
        "node_id", "node_name", "node_type", "store_id", "store_name", "store_type",
        "dc_id", "dc_name", "region", "latitude", "longitude", "lat", "lon",
        "capacity", "dc_capacity_qty", "throughput_capacity",
    )
    return [{field: node.get(field) for field in fields if field in node} for node in nodes]


@st.cache_data(show_spinner=False, max_entries=8)
def _layout_cached(
    nodes: list[dict], sim_routes: list[dict], layout_version: str = SIMULATION_LAYOUT_VERSION,
):
    """Deterministic structure-only layout shared by every simulation control."""
    _ = layout_version
    _SIMULATION_COUNTERS["layout_calculations"] += 1
    return compute_dynamic_layout(nodes, sim_routes, width=1280.0, height=800.0, margin=112.0)


def simulation_performance_snapshot() -> dict[str, int]:
    """Expose the home interaction counters used by performance tests."""
    transition = inventory_transition_cache_info()
    return {
        "excel_loader_calls": 0,
        "analysis_pipeline_calls": 0,
        "vhs_calls": 0,
        "dqn_training_calls": 0,
        "dqn_model_load_calls": 0,
        **_SIMULATION_COUNTERS,
        **transition,
    }


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
# Inventory view decoration
# --------------------------------------------------------------------------- #
def _record_lookup(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item.get("store_id") or ""), str(item.get("product_id") or "")): dict(item)
        for item in records
    }


def _view_for_transition(
    transition: Mapping[str, Any], role: str, view_mode: str,
    final_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    endpoint = dict(transition.get(role) or {})
    product_id = str(endpoint.get("product_id") or transition.get("product_id") or "")
    store_id = str(endpoint.get("store_id") or "")
    final = final_lookup.get((store_id, product_id), {})
    before_stock = endpoint.get("before_stock")
    after_stock = final.get("stock", endpoint.get("after_stock"))
    before_status = endpoint.get("before_status") or "데이터 부족"
    after_status = final.get("status", endpoint.get("after_status")) or "데이터 부족"
    if view_mode == "이동 전":
        display_stock, status = before_stock, before_status
    else:
        display_stock, status = after_stock, after_status
    movement = endpoint.get("outbound_qty") if role == "source" else endpoint.get("inbound_qty")
    movement_label = (
        f"{'출고' if role == 'source' else '입고'} {_qty(movement)}"
        f" · 수요 {_qty(endpoint.get('demand_qty'))}"
    )
    return {
        "detail": True,
        "role": role,
        "product_id": product_id,
        "product_name": endpoint.get("product_name") or transition.get("product_name") or product_id,
        "display_stock": display_stock,
        "before_stock": before_stock,
        "after_stock": after_stock,
        "demand": endpoint.get("demand_qty"),
        "status": status,
        "before_status": before_status,
        "after_status": after_status,
        "movement_label": movement_label,
        "view_mode": view_mode,
        "feasible": bool(transition.get("feasible")),
    }


def _decorate_nodes(
    nodes: Sequence[Mapping[str, object]],
    scenario: Mapping[str, Any],
    sim_routes: Sequence[Mapping[str, object]],
    view_mode: str,
) -> list[dict[str, object]]:
    transitions = list(scenario.get("transitions") or [])
    baseline = _record_lookup(scenario.get("baseline_records") or [])
    final = _record_lookup(scenario.get("final_records") or [])
    endpoint_views: dict[str, dict[str, Any]] = {}
    for transition in transitions:
        for role in ("source", "target"):
            endpoint = transition.get(role) or {}
            store_id = str(endpoint.get("store_id") or "")
            if store_id and store_id not in endpoint_views:
                endpoint_views[store_id] = _view_for_transition(transition, role, view_mode, final)

    primary_product = str((sim_routes[0] if sim_routes else {}).get("product_id") or "")
    via_by_dc: dict[str, dict[str, float]] = {}
    for transition in transitions:
        if transition.get("route_type") != "VIA_DC":
            continue
        dc_id = str(transition.get("dc_id") or "")
        if not dc_id:
            continue
        stats = via_by_dc.setdefault(dc_id, {"count": 0.0, "quantity": 0.0})
        if transition.get("feasible"):
            stats["count"] += 1
            stats["quantity"] += float(transition.get("applied_quantity") or 0.0)

    decorated: list[dict[str, object]] = []
    for raw_node in nodes:
        node = copy.deepcopy(dict(raw_node))
        node_id = str(node.get("node_id") or "")
        node_type = str(node.get("node_type") or "").upper()
        if node_type == "DC":
            dc_stats = via_by_dc.get(node_id, {"count": 0.0, "quantity": 0.0})
            capacity = next(
                (_number(node.get(field)) for field in ("dc_capacity_qty", "capacity", "throughput_capacity") if _number(node.get(field)) is not None),
                None,
            )
            node["dc_inventory_view"] = {
                "is_via": node_id in via_by_dc,
                "active_count": int(dc_stats["count"]),
                "quantity": dc_stats["quantity"],
                "capacity": capacity,
            }
        else:
            view = endpoint_views.get(node_id)
            if view is None and primary_product:
                before = baseline.get((node_id, primary_product), {})
                after = final.get((node_id, primary_product), before)
                if before:
                    use_after = view_mode != "이동 전"
                    view = {
                        "detail": False,
                        "product_id": primary_product,
                        "product_name": before.get("product_name") or primary_product,
                        "display_stock": after.get("stock") if use_after else before.get("stock"),
                        "before_stock": before.get("stock"),
                        "after_stock": after.get("stock"),
                        "demand": before.get("demand"),
                        "status": after.get("status") if use_after else before.get("status"),
                        "before_status": before.get("status"),
                        "after_status": after.get("status"),
                        "view_mode": view_mode,
                    }
            node["inventory_view"] = view or {
                "detail": False, "status": "데이터 부족", "display_stock": None,
                "product_name": primary_product or "데이터 없음", "view_mode": view_mode,
            }
        decorated.append(node)
    return decorated


# --------------------------------------------------------------------------- #
# SVG helpers
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


def _state_badge(status: str, y: float, width: float = 70.0) -> str:
    background, text_color, border = _STATE_STYLE.get(status, _STATE_STYLE["데이터 부족"])
    return (
        f'<rect class="inventory-status-badge" x="{-width / 2:.2f}" y="{y - 12:.2f}" width="{width:.2f}" height="19" '
        f'rx="9.5" fill="{background}" stroke="{border}"/>'
        f'<text class="inventory-status-text" x="0" y="{y + 2:.2f}" text-anchor="middle" style="fill:{text_color}">{_safe(status)}</text>'
    )


def _dc_node_svg(node: Mapping[str, object]) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = str(node.get("node_name") or node.get("node_id"))
    node_id = str(node.get("node_id") or "-")
    view = dict(node.get("dc_inventory_view") or {})
    active = bool(view.get("is_via"))
    stroke = "#ad7600" if active else "#b9a86b"
    fill = "#fff5ce" if active else "#fffaf0"
    left, top = -width / 2, -height / 2
    via_label = "이번 경로 경유" if active else "경유 아님"
    capacity = _number(view.get("capacity"))
    capacity_label = f"용량 {_qty(capacity)}" if capacity is not None else "용량 데이터 없음"
    return (
        f'<g class="network-node dc-node" data-active="{str(active).lower()}" transform="translate({x:.2f} {y:.2f})">'
        f'<title>{_safe(name)} · {via_label}</title>'
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{3 if active else 1.8}" />'
        f'{_svg_label(name, "node-label dc-label", top + 24, 13)}'
        f'<text class="dc-node-id" x="0" y="{top + 46:.2f}" text-anchor="middle">{_safe(node_id)} · {_safe(via_label)}</text>'
        f'<text class="dc-node-work" x="0" y="{top + 69:.2f}" text-anchor="middle">추천 {int(view.get("active_count") or 0)}건 · {_qty(view.get("quantity"))}</text>'
        f'<text class="dc-node-capacity" x="0" y="{height / 2 - 9:.2f}" text-anchor="middle">{_safe(capacity_label)}</text>'
        '</g>'
    )


def _store_node_svg(node: Mapping[str, object], total_stores: int) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = str(node.get("node_name") or node.get("node_id"))
    emphasized = bool(node.get("is_recommended"))
    show_label = bool(node.get("show_label", True))
    view = dict(node.get("inventory_view") or {})
    detailed = bool(view.get("detail")) and emphasized
    status = str(view.get("status") or "데이터 부족")
    background, _text_color, border = _STATE_STYLE.get(status, _STATE_STYLE["데이터 부족"])
    stroke = border if emphasized else "#c4ccd5"
    fill = background if emphasized else "#ffffff"
    left, top = -width / 2, -height / 2
    label = name if show_label else str(node.get("node_id") or "-")
    title = f"{name} · {status} · 재고 {_qty(view.get('display_stock'))}"

    if not detailed:
        name_y = -8 if height >= 58 else -3
        stock_text = _qty(view.get("display_stock")) if view.get("display_stock") is not None else ""
        return (
            f'<g class="network-node store-node store-node-compact" transform="translate({x:.2f} {y:.2f})">'
            f'<title>{_safe(title)}</title>'
            f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if emphasized else 1.3}" />'
            f'{_svg_label(label, "node-label store-label", name_y, 10 if total_stores <= 16 else 8)}'
            f'{_state_badge(status, height / 2 - 8, 66)}'
            f'<text class="compact-stock" x="{width / 2 - 8:.2f}" y="{top + 15:.2f}" text-anchor="end">{_safe(stock_text)}</text>'
            '</g>'
        )

    product = _short_label(view.get("product_name"), 17)
    before, after = view.get("before_stock"), view.get("after_stock")
    view_mode = str(view.get("view_mode") or "전후 비교")
    if view_mode == "전후 비교":
        stock_line = f"재고 {_qty(before)} → {_qty(after)}"
        state_line = f"{view.get('before_status') or '데이터 부족'} → {view.get('after_status') or '데이터 부족'}"
    elif view_mode == "이동 전":
        stock_line = f"현재 재고 {_qty(before)}"
        state_line = str(view.get("before_status") or "데이터 부족")
    else:
        stock_line = f"이동 후 재고 {_qty(after)}"
        state_line = str(view.get("after_status") or "데이터 부족")
    demand = _number(view.get("demand"))
    stock = _number(view.get("display_stock"))
    ratio = 0.0 if stock is None or demand is None or demand <= 0 else min(1.0, max(0.0, stock / demand))
    bar_width = max(0.0, width - 28.0)
    return (
        f'<g class="network-node store-node store-node-detailed" transform="translate({x:.2f} {y:.2f})">'
        f'<title>{_safe(title)}</title>'
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2.8" />'
        f'{_svg_label(label, "node-label store-label", top + 20, 13)}'
        f'{_state_badge(status, top + 44, 72)}'
        f'<text class="inventory-product" x="0" y="{top + 63:.2f}" text-anchor="middle">{_safe(product)}</text>'
        f'<text class="inventory-stock" x="0" y="{top + 80:.2f}" text-anchor="middle">{_safe(stock_line)}</text>'
        f'<text class="inventory-movement" x="0" y="{top + 96:.2f}" text-anchor="middle">{_safe(view.get("movement_label") or state_line)}</text>'
        f'<text class="inventory-state-change" x="0" y="{top + 112:.2f}" text-anchor="middle">{_safe(state_line)}</text>'
        f'<rect class="inventory-stock-bar-bg" x="{-bar_width / 2:.2f}" y="{height / 2 - 9:.2f}" width="{bar_width:.2f}" height="5" rx="2.5" fill="#dce2e8" />'
        f'<rect class="inventory-stock-bar" x="{-bar_width / 2:.2f}" y="{height / 2 - 9:.2f}" width="{bar_width * ratio:.2f}" height="5" rx="2.5" fill="{border}" />'
        '</g>'
    )


def _truck_icon(
    color: str, truck_color: str, truck_soft: str, mode_label: str = "트럭",
    route_label: str = "", product_label: str = "", quantity_label: str = "",
    stage_label: str = "",
) -> str:
    """Compact truck with user-facing route, product, quantity, and stage text."""
    detail = " · ".join(value for value in (mode_label, route_label, product_label, quantity_label, stage_label) if value)
    return (
        f'<title>{_safe(detail or mode_label)}</title>'
        f'<rect x="-24" y="-11" width="31" height="19" rx="3" fill="#ffffff" stroke="{truck_color}" stroke-width="2.3"/>'
        f'<rect x="7" y="-11" width="18" height="19" rx="3" fill="{truck_soft}" stroke="{truck_color}" stroke-width="2.3"/>'
        '<rect x="11" y="-7" width="5" height="5" rx="1" fill="#ffffff"/>'
        f'<rect x="-19" y="-6" width="15" height="6" rx="1.5" fill="{truck_soft}" opacity="0.9"/>'
        f'<circle cx="-14" cy="11" r="4.2" fill="#ffffff" stroke="{truck_color}" stroke-width="2.3"/>'
        f'<circle cx="18" cy="11" r="4.2" fill="#ffffff" stroke="{truck_color}" stroke-width="2.3"/>'
        f'<rect x="-34" y="-31" width="68" height="14" rx="7" fill="{color}" opacity="0.96"/>'
        f'<text class="vehicle-route" x="0" y="-21" text-anchor="middle" fill="#ffffff">{_safe(route_label or mode_label)}</text>'
        f'<text class="vehicle-mode" x="0" y="28" text-anchor="middle" fill="{truck_color}">{_safe(quantity_label or mode_label)}</text>'
        f'<text class="vehicle-stage" x="0" y="41" text-anchor="middle" fill="{truck_color}">{_safe(stage_label)}</text>'
    )


def _segments_points(segments: Sequence[Mapping[str, object]], positions: dict[str, tuple[float, float]]):
    points: list[tuple[float, float]] = []
    for index, segment in enumerate(segments):
        start = positions.get(str(segment["from_node_id"]))
        end = positions.get(str(segment["to_node_id"]))
        if not start or not end:
            return None
        if index == 0:
            points.append(start)
        points.append(end)
    return points if len(points) >= 2 else None


def _edge_point(
    center: tuple[float, float], toward: tuple[float, float], size: tuple[float, float], padding: float = 9.0,
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


def _via_motion_points(points: Sequence[tuple[float, float]]) -> str:
    if len(points) != 3:
        return "0;1"
    first = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
    second = math.hypot(points[2][0] - points[1][0], points[2][1] - points[1][1])
    midpoint = first / max(first + second, 1e-9)
    return f"0;{midpoint:.4f};{midpoint:.4f};1"


def _route_summary_html(
    sim_routes: Sequence[Mapping[str, object]], scenario: Mapping[str, Any], selected_id: str,
) -> str:
    selected = next((item for item in sim_routes if str(item.get("route_id")) == selected_id), sim_routes[0] if sim_routes else {})
    transition = next(
        (item for item in scenario.get("transitions") or [] if str(item.get("route_id")) == str(selected.get("route_id"))),
        {},
    )
    route_label = "직접 이동" if normalize_route_type(selected) == "DIRECT" else "물류센터 경유"
    applied = transition.get("applied_quantity")
    requested = transition.get("requested_quantity", selected.get("recommended_qty"))
    quantity = _qty(applied) if applied is not None else _qty(requested)
    if applied is not None and requested is not None and float(applied) != float(requested):
        quantity = f"{quantity} / 추천 {_qty(requested)}"
    return (
        '<div class="v2-sim-route-summary">'
        f'<span><small>선택 경로</small><strong>{_safe(route_label)}</strong></span>'
        f'<span><small>이동 상품</small><strong>{_safe(selected.get("product_name") or selected.get("product_id") or "-")}</strong></span>'
        f'<span><small>출발 → 도착</small><strong>{_safe(selected.get("source_name") or selected.get("source_id") or "-")} → {_safe(selected.get("target_name") or selected.get("target_id") or "-")}</strong></span>'
        f'<span><small>이동 수량</small><strong>{_safe(quantity)}</strong></span>'
        f'<span><small>실행 상태</small><strong>{_safe(transition.get("status") or "데이터 부족")}</strong></span>'
        '</div>'
    )


@st.cache_data(show_spinner=False, max_entries=24)
def _network_markup_cached(
    data_signature: str,
    nodes: list[dict], sim_routes: list[dict], all_routes: list[dict],
    playing: bool, speed_seconds: float, show_all: bool, selected_id: str,
    scenario: dict[str, Any] | None = None,
    inventory_view: str = "전후 비교",
    display_mode: str = "단일 경로",
) -> dict[str, object]:
    _ = (data_signature, inventory_view, display_mode, INVENTORY_TRANSITION_VERSION, SIMULATION_LAYOUT_VERSION)
    _SIMULATION_COUNTERS["svg_html_generations"] += 1
    scenario = scenario or {}
    layout = _layout_cached(_layout_node_payload(nodes), sim_routes)
    if not layout.is_valid:
        return {"ok": False, "errors": layout.errors, "html": ""}

    views = {str(node.get("node_id")): node for node in nodes}
    layout_dcs = [{**node, **{key: value for key, value in views.get(str(node.get("node_id")), {}).items() if key in {"dc_inventory_view"}}} for node in layout.dcs]
    layout_stores = [{**node, **{key: value for key, value in views.get(str(node.get("node_id")), {}).items() if key in {"inventory_view"}}} for node in layout.stores]
    all_nodes = layout_dcs + layout_stores
    positions = {str(node["node_id"]): (float(node["x"]), float(node["y"])) for node in all_nodes if node}
    dimensions = {str(node["node_id"]): (float(node["width"]), float(node["height"])) for node in all_nodes if node}
    canvas = layout.canvas

    background: list[str] = []
    if show_all:
        seen: set[tuple[str, str]] = set()
        for route in all_routes[:_MAX_BACKGROUND_ROUTES]:
            try:
                segments = build_route_segments(route, nodes)
            except ValueError:
                continue
            for segment in segments:
                pair = tuple(sorted((str(segment["from_node_id"]), str(segment["to_node_id"]))))
                if pair in seen or pair[0] == pair[1]:
                    continue
                seen.add(pair)
                start, end = positions.get(pair[0]), positions.get(pair[1])
                if start and end:
                    background.append(
                        f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" '
                        'stroke="#7f8b99" stroke-width="1" stroke-opacity="0.10" />'
                    )

    transition_by_route = {str(item.get("route_id")): item for item in scenario.get("transitions") or []}
    route_paths: list[str] = []
    vehicles: list[str] = []
    skip_labels: list[str] = []
    for index, route in enumerate(sim_routes):
        try:
            segments = build_route_segments(route, nodes)
        except ValueError:
            continue
        lane_offset = _ROUTE_LANES[index % len(_ROUTE_LANES)] if len(sim_routes) > 1 else 0.0
        points = _route_path_points(segments, positions, dimensions, lane_offset)
        if not points:
            continue
        is_via = normalize_route_type(route) == "VIA_DC"
        color = _VIA_DC_ROUTE_COLOR if is_via else _DIRECT_ROUTE_COLOR
        selected = str(route.get("route_id")) == selected_id
        transition = transition_by_route.get(str(route.get("route_id")), {})
        feasible = bool(transition.get("feasible", True))
        bend_sign = -1.0 if index % 2 else 1.0
        path_data = _route_path_d(points, bend_sign * (14.0 + abs(lane_offset) * 0.16))
        dash = ' stroke-dasharray="11 8"' if is_via else ""
        width = 5.2 if selected else 3.0
        opacity = 1.0 if selected else 0.52
        if not feasible:
            opacity = 0.24
        path_id = f"rp{index}"
        route_paths.append(
            f'<path id="{path_id}" data-lane="{lane_offset:.0f}" d="{path_data}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" stroke-linecap="round"{dash} />'
        )
        if not feasible:
            px, py = _point_along_polyline(points, 0.5)
            skip_labels.append(
                f'<text class="route-skip-label" x="{px:.2f}" y="{py - 10:.2f}" text-anchor="middle">현재 상태에서 실행 불가</text>'
            )
            disabled_label = "" if display_mode == "단일 경로" else f"추천 {index + 1}"
            disabled_truck = _truck_icon(
                "#8b96a3", "#697481", "#edf0f3", "대기",
                disabled_label, _short_label(route.get("product_name") or route.get("product_id"), 10),
                "0개", "실행 불가",
            )
            vehicles.append(
                f'<g class="v2-vehicle v2-vehicle-disabled" transform="translate({px:.2f} {py:.2f})">{disabled_truck}</g>'
            )
            continue
        truck_color, truck_soft, truck_mode = _transport_style(route.get("transport_type"), route.get("route_type"))
        route_label = "" if display_mode == "단일 경로" else f"추천 {index + 1}"
        product_label = _short_label(route.get("product_name") or route.get("product_id"), 10)
        quantity_label = _qty(transition.get("applied_quantity", route.get("recommended_qty")))
        stage_label = "출고 → DC 경유 → 입고" if is_via else "출고 → 입고"
        truck = _truck_icon(color, truck_color, truck_soft, truck_mode, route_label, product_label, quantity_label, stage_label)
        vehicle_class = "v2-vehicle v2-vehicle-selected" if selected else "v2-vehicle v2-vehicle-muted"
        if playing:
            phase = index * speed_seconds / max(1, len(sim_routes))
            key_points = _via_motion_points(points) if is_via else "0;1"
            key_times = "0;0.43;0.57;1" if is_via else "0;1"
            vehicles.append(
                f'<g class="{vehicle_class}">{truck}'
                f'<animateMotion dur="{speed_seconds:.1f}s" begin="-{phase:.1f}s" repeatCount="indefinite" rotate="0" '
                f'keyPoints="{key_points}" keyTimes="{key_times}" calcMode="linear">'
                f'<mpath xlink:href="#{path_id}"/></animateMotion></g>'
            )
        else:
            parked_x, parked_y = _point_along_polyline(points, 0.28 + index * 0.18)
            vehicles.append(f'<g class="{vehicle_class}" transform="translate({parked_x:.2f} {parked_y:.2f})">{truck}</g>')

    node_shapes = [_dc_node_svg(node) for node in layout_dcs]
    node_shapes.extend(_store_node_svg(node, len(layout_stores)) for node in layout_stores)
    network_html = (
        '<div class="v2-network-shell">'
        + _route_summary_html(sim_routes, scenario, selected_id)
        + '<div class="v2-network-legend" aria-label="네트워크 범례">'
        '<span class="v2-legend-item"><span class="v2-legend-line v2-legend-line-direct"></span>실선: 직접 이동</span>'
        '<span class="v2-legend-item"><span class="v2-legend-line v2-legend-line-via"></span>점선: 물류센터 경유</span>'
        '<span class="v2-legend-item"><span class="v2-state-dot state-excess"></span>초과재고</span>'
        '<span class="v2-legend-item"><span class="v2-state-dot state-normal"></span>적정재고</span>'
        '<span class="v2-legend-item"><span class="v2-state-dot state-shortage"></span>부족재고</span>'
        '<span class="v2-legend-item"><span class="v2-state-dot state-missing"></span>데이터 부족</span>'
        '</div>'
        f'<svg class="v2-network-svg" viewBox="0 0 {canvas["width"]} {canvas["height"]}" '
        'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'preserveAspectRatio="xMidYMid meet" role="img" aria-label="추천 경로 재고 이동 현황">'
        + "".join(background)
        + "".join(route_paths)
        + "".join(skip_labels)
        + "".join(node_shapes)
        + "".join(vehicles)
        + "</svg></div>"
    )
    return {"ok": True, "errors": [], "html": network_html}


def _render_network(
    nodes: list[dict], sim_routes: list[dict], all_routes: list[dict],
    playing: bool, speed_seconds: float, show_all: bool,
    scenario: dict[str, Any], inventory_view: str, display_mode: str, selected_id: str,
) -> None:
    if not nodes or not sim_routes:
        st.markdown('<div class="v2-network-placeholder">표시할 추천 경로가 없습니다</div>', unsafe_allow_html=True)
        return
    result = _network_markup_cached(
        _data_signature(), nodes, sim_routes, all_routes, playing, float(speed_seconds),
        show_all, selected_id, scenario, inventory_view, display_mode,
    )
    if not result.get("ok"):
        render_empty_state(st, "네트워크를 표시할 수 없습니다", " / ".join(result.get("errors") or []), compact=True)
        return
    st.markdown(str(result.get("html") or ""), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Controls and result details
# --------------------------------------------------------------------------- #
def _simulation_routes(
    recommendations: list[dict], show_all: bool = False,
    display_mode: str | None = None, rank_label: str | None = None,
) -> list[dict]:
    _ = show_all  # the toggle affects background lines, never the execution set
    ordered = top_recommendations(recommendations, limit=3)
    mode = display_mode or str(st.session_state.get("home_sim_display_mode") or "단일 경로")
    rank_text = rank_label or str(st.session_state.get("home_sim_route_rank") or "1순위")
    try:
        rank_index = max(0, min(len(ordered) - 1, int(rank_text[0]) - 1))
    except (ValueError, IndexError):
        rank_index = 0
    if mode == "상위 3개":
        return ordered
    return [ordered[rank_index]] if ordered else []


def _set_sim_playing(value: bool) -> None:
    st.session_state["home_sim_playing"] = value
    st.session_state["home_sim_run_nonce"] = int(st.session_state.get("home_sim_run_nonce") or 0) + 1


def _render_controls(recommendations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    playing = bool(st.session_state.get("home_sim_playing", False))
    columns = st.columns([0.92, 1.05, 1.12, 0.92, 0.82, 1.02, 1.08], gap="small")
    rank_options = [f"{index}순위" for index in range(1, min(3, len(recommendations)) + 1)] or ["1순위"]
    current_rank = str(st.session_state.get("home_sim_route_rank") or "1순위")
    if current_rank not in rank_options:
        current_rank = rank_options[0]
        st.session_state["home_sim_route_rank"] = current_rank
    if st.session_state.get("home_sim_route_rank_select") not in rank_options:
        st.session_state["home_sim_route_rank_select"] = current_rank
    rank_label = columns[0].selectbox(
        "표시 경로 선택", rank_options,
        key="home_sim_route_rank_select", label_visibility="collapsed", help="표시할 추천 순위를 선택합니다.",
    )
    display_options = ["단일 경로", "상위 3개"]
    current_display = str(st.session_state.get("home_sim_display_mode") or "단일 경로")
    if current_display not in display_options:
        current_display = display_options[0]
        st.session_state["home_sim_display_mode"] = current_display
    if st.session_state.get("home_sim_display_select") not in display_options:
        st.session_state["home_sim_display_select"] = current_display
    display_mode = columns[1].selectbox(
        "표시 방식", display_options,
        key="home_sim_display_select", label_visibility="collapsed",
    )
    columns[2].button(
        "시뮬레이션 실행", width="stretch", key="sim_start", disabled=playing,
        type="primary", on_click=_set_sim_playing, args=(True,),
    )
    columns[3].button(
        "다시 실행", width="stretch", key="sim_restart",
        on_click=_set_sim_playing, args=(True,),
    )
    speed_options = ["느림", "보통", "빠름"]
    current_speed = str(st.session_state.get("simulation_speed") or "보통")
    if current_speed not in speed_options:
        current_speed = "보통"
        st.session_state["simulation_speed"] = current_speed
    if st.session_state.get("home_speed_select") not in speed_options:
        st.session_state["home_speed_select"] = current_speed
    speed = columns[4].selectbox(
        "시뮬레이션 속도", speed_options,
        key="home_speed_select", label_visibility="collapsed",
    )
    view_options = ["전후 비교", "이동 전", "이동 후"]
    current_view = str(st.session_state.get("home_sim_inventory_view") or "전후 비교")
    if current_view not in view_options:
        current_view = view_options[0]
        st.session_state["home_sim_inventory_view"] = current_view
    if st.session_state.get("home_inventory_view_select") not in view_options:
        st.session_state["home_inventory_view_select"] = current_view
    inventory_view = columns[5].selectbox(
        "재고 표시", view_options,
        key="home_inventory_view_select", label_visibility="collapsed",
    )
    show_all = columns[6].toggle(
        "전체 경로 보기", value=bool(st.session_state.get("show_all_routes", False)),
        key="home_show_all", help="실행 경로는 유지하고 보조 연결선만 표시합니다.",
    )
    st.session_state["home_sim_route_rank"] = rank_label
    st.session_state["home_sim_display_mode"] = display_mode
    st.session_state["simulation_speed"] = speed
    st.session_state["home_sim_inventory_view"] = inventory_view
    st.session_state["show_all_routes"] = show_all
    return {
        "rank_label": rank_label,
        "display_mode": display_mode,
        "speed": speed,
        "inventory_view": inventory_view,
        "show_all": bool(show_all),
    }


def _render_steps(route_type: str, playing: bool, inventory_view: str) -> None:
    steps = (
        ["출고 준비", "물류센터 이동", "물류센터 경유", "도착 점포 이동", "도착 및 입고", "재고 반영 완료"]
        if route_type == "VIA_DC"
        else ["출고 준비", "점포 간 이동", "도착 및 입고", "재고 반영 완료"]
    )
    current = len(steps) if inventory_view == "이동 후" else 2 if playing else 1
    items = []
    for index, label in enumerate(steps, start=1):
        if index < current:
            css_class, marker = "is-complete", "✓"
        elif index == current:
            css_class, marker = "is-current", str(index)
        else:
            css_class, marker = "is-pending", str(index)
        items.append(
            f'<span class="v2-sim-step {css_class}"><b>{marker}</b><span>{_safe(label)}</span></span>'
        )
    st.markdown('<div class="v2-sim-steps">' + "".join(items) + "</div>", unsafe_allow_html=True)


def _render_simulation_kpis(scenario: Mapping[str, Any]) -> None:
    kpis = scenario.get("kpis") or {}
    values = [
        ("이동 수량", _qty(kpis.get("moved_quantity"))),
        ("초과재고 감소량", _qty(kpis.get("excess_reduction"))),
        ("부족재고 감소량", _qty(kpis.get("shortage_reduction"))),
    ]
    if kpis.get("expected_saving") is not None:
        values.append(("예상 절감액", format_currency(kpis.get("expected_saving"))))
    columns = st.columns(len(values), gap="small")
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)


def _render_scenario_status(scenario: Mapping[str, Any], display_mode: str) -> None:
    if display_mode != "상위 3개":
        return
    requested = int(scenario.get("requested_route_count") or 0)
    executed = int(scenario.get("executed_route_count") or 0)
    skipped = int(scenario.get("skipped_route_count") or 0)
    st.caption(f"순차 적용: 요청 {requested}건 · 실행 {executed}건 · 건너뜀 {skipped}건")
    for item in scenario.get("skipped_reasons") or []:
        st.caption(f"{item.get('route_id') or '경로'}: {item.get('reason') or '현재 상태에서 실행 불가'}")


def _render_inventory_basis(scenario: Mapping[str, Any]) -> None:
    with st.expander("재고 상태 계산 기준", expanded=False):
        metadata = scenario.get("metadata") or {}
        st.markdown(
            "- 상태: 현재 재고와 실제 수요 기준값을 비교해 초과·적정·부족을 구분합니다.\n"
            "- 이동 가능 재고: 명시 값이 있으면 우선 사용하고, 없으면 현재 재고에서 잔여 유통기간 수요와 실제 최소 진열재고를 뺀 값만 사용합니다.\n"
            "- 도착 한도: 명시 부족량을 우선하고, 없으면 실제 7일 수요와 현재 재고의 차이를 사용합니다.\n"
            "- 필수 열이 없으면 값을 만들지 않고 데이터 부족 또는 현재 상태에서 실행 불가로 표시합니다."
        )
        performance = scenario.get("performance") or {}
        st.caption(
            f"전이 엔진 {INVENTORY_TRANSITION_VERSION} · 계산 {performance.get('calculation_ms', 0):,.3f}ms"
            f" · 동일 조건 캐시 {'사용' if performance.get('cache_hit') else '미사용'}"
        )
        if metadata.get("temporary_copy_only"):
            st.caption("모든 이동은 복사본에서만 계산되며 적용 데이터는 변경하지 않습니다.")


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
        "추천 경로를 실행했을 때의 실제 재고 변화와 차량 이동 단계를 확인합니다.",
    )
    controls = _render_controls(recommendations)
    sim_routes = _simulation_routes(
        recommendations,
        bool(controls["show_all"]),
        str(controls["display_mode"]),
        str(controls["rank_label"]),
    )
    focus_index = max(0, min(len(recommendations) - 1, int(str(controls["rank_label"])[0]) - 1))
    focus_id = str(recommendations[focus_index].get("route_id") or "") if recommendations else ""
    if controls["display_mode"] == "단일 경로" and sim_routes:
        focus_id = str(sim_routes[0].get("route_id") or "")
    scenario = cached_inventory_scenario(
        _data_signature(), data or {}, sim_routes, display_mode=str(controls["display_mode"]),
    )
    base_nodes = _nodes_from_data(sim_routes)
    nodes = _decorate_nodes(base_nodes, scenario, sim_routes, str(controls["inventory_view"]))
    all_routes = _network_routes_from_data()
    playing = bool(st.session_state.get("home_sim_playing", False))
    speed_seconds = animation_duration_seconds(str(controls["speed"]))
    _render_network(
        nodes, sim_routes, all_routes, playing, speed_seconds, bool(controls["show_all"]),
        scenario, str(controls["inventory_view"]), str(controls["display_mode"]), focus_id,
    )
    focus_route = next((route for route in sim_routes if str(route.get("route_id")) == focus_id), sim_routes[0] if sim_routes else {})
    _render_steps(normalize_route_type(focus_route), playing, str(controls["inventory_view"]))
    _render_simulation_kpis(scenario)
    _render_scenario_status(scenario, str(controls["display_mode"]))
    _render_inventory_basis(scenario)
