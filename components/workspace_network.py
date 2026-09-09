"""Static store/DC/route network for the 재고 운영 Workspace.

This is the decision picture, not an animation: it draws the stores, the DCs and
the moves of the *current execution plan*, and highlights the one move the user
has selected. The layout helpers are reused from
:mod:`simulation.dynamic_network`, so node classification and DIRECT / VIA_DC
segmentation behave exactly as everywhere else in the app.

Deliberate limits, so the picture stays readable and honest:

* Only the plan (or the selected move) is drawn by default — never every
  candidate edge, which would turn into a spaghetti graph.
* An edge is labelled with ``planned_qty`` only. Distance, travel time and real
  transport cost are not collected yet, so they are never printed on an edge.
* Meaning is carried by shape, dash pattern and a written role label as well as
  by colour, and the palette stays at four states.
"""
from __future__ import annotations

import html
import math
from typing import Any, Iterable, Mapping, Sequence

from simulation.dynamic_network import (
    DC,
    build_network_nodes,
    build_route_segments,
    classify_node,
    compute_network_layout,
    normalize_route_type,
)

# The canvas is deliberately narrow and tall. On a 1366px desktop the centre
# column is roughly 640px wide, so a wider viewBox would shrink the SVG text
# below a readable size once the browser scales it down.
CANVAS_WIDTH = 940.0
CANVAS_HEIGHT = 620.0
CANVAS_MARGIN = 78.0

SCOPE_PLAN = "현재 실행계획"
SCOPE_SELECTED = "선택한 이동"
SCOPE_ALL = "전체 네트워크"
SCOPE_OPTIONS = (SCOPE_PLAN, SCOPE_SELECTED, SCOPE_ALL)

# One restrained palette. 선택 경로 / 다른 계획 경로 / 배경 경로 만 구분한다.
COLOR_SELECTED = "#1d6fa3"
COLOR_PLANNED = "#8fa3b5"
COLOR_BACKGROUND = "#d8dee5"
COLOR_LINE = "#cbd5df"

STATE_STYLES: dict[str, tuple[str, str]] = {
    "과잉": ("#b26a1f", "#fdeecb"),
    "부족": ("#b23b3b", "#fbe0e0"),
    "정상": ("#2f7d5b", "#e3f3ea"),
    "이동 대상": ("#2d6fa8", "#e2eefb"),
}

ROLE_SOURCE = "출발"
ROLE_TARGET = "도착"
ROLE_DC = "경유 DC"

_MAX_BACKGROUND_EDGES = 24
_MAX_LABELLED_EDGES = 8

# On-screen legibility budget. The SVG is scaled to the width of the centre
# column, so every size below is multiplied by (column width / CANVAS_WIDTH)
# before anyone reads it. Measured in a real browser: at 1366×768 the centre
# column is ~630px, a 0.67x downscale, so 19.5 user units land at ~13px and the
# 15-unit floor still reads at ~10px. Nothing here may go below that floor —
# a name is shortened rather than shrunk once the floor is reached.
NAME_FONT = 19.5
NAME_FONT_MIN = 15.0
DC_NAME_FONT = 18.5
DC_NAME_FONT_MIN = 15.0
EDGE_FONT = 17.0
EDGE_CHIP_HEIGHT = 23.0
#: Upper bound for the 과잉/부족/정상 pill inside a store node. The pill height
#: (node height × 0.35) usually decides the size; this only stops it growing on
#: very tall nodes.
STATE_PILL_FONT_MAX = 16.5


def _safe(value: Any) -> str:
    return html.escape(str(value)) if value is not None else "-"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _glyph_width(text: str, font: float) -> float:
    """Approximate rendered width: CJK glyphs are full-width, latin about 0.56em."""
    return sum(font * (1.0 if ord(char) > 0x2E80 else 0.56) for char in text)


def _fit_label(value: Any, box_width: float, font: float, min_font: float) -> tuple[str, float]:
    """Largest readable size that keeps a node name inside its own box.

    The size is reduced to ``min_font`` first and only then is the text
    shortened, so a long store or DC name is never painted outside its box —
    which is what used to make 물류센터 names run over the edge labels beside them.
    """
    text = _text(value) or "-"
    inner = max(24.0, box_width - 12.0)
    size = font
    while size > min_font and _glyph_width(text, size) > inner:
        size = round(size - 0.5, 2)
    if _glyph_width(text, size) > inner:
        while len(text) > 1 and _glyph_width(text + "…", size) > inner:
            text = text[:-1]
        text += "…"
    return text, size


def _split_two(text: str) -> tuple[str, str]:
    """Split a name at the space nearest its middle (midpoint if it has none)."""
    spaces = [index for index, char in enumerate(text) if char == " "]
    if spaces:
        cut = min(spaces, key=lambda index: abs(index - len(text) / 2))
        return text[:cut].strip(), text[cut + 1:].strip()
    half = len(text) // 2
    return text[:half], text[half:]


def _fit_dc_label(name: str, box_width: float) -> tuple[list[str], float]:
    """A 물류센터 name on one line, or on two when one line would have to be cut."""
    single, font = _fit_label(name, box_width, DC_NAME_FONT, DC_NAME_FONT_MIN)
    if not single.endswith("…"):
        return [single], font
    first, second = _split_two(_text(name))
    if not first or not second:
        return [single], font
    top, top_font = _fit_label(first, box_width, DC_NAME_FONT, DC_NAME_FONT_MIN)
    bottom, bottom_font = _fit_label(second, box_width, DC_NAME_FONT, DC_NAME_FONT_MIN)
    if top.endswith("…") or bottom.endswith("…"):
        return [single], font
    return [top, bottom], min(top_font, bottom_font)


def _qty_label(item: Mapping[str, Any]) -> str:
    for key in ("planned_qty", "recommended_qty"):
        value = item.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return f"{int(round(number)):,}개"
    return ""


def _dc_size(store_size: tuple[float, float]) -> tuple[float, float]:
    """DC box size. Kept at the size the shared radial layout is tuned for — a
    wider box pushed the DCs into the neighbouring stores on a two-DC network, so
    a long 물류센터 name is wrapped onto two lines instead (see ``_fit_dc_label``)."""
    store_w, store_h = store_size
    return max(150.0, store_w * 1.15), max(72.0, store_h * 1.2)


def _dimensions(count: int) -> tuple[float, float]:
    if count <= 6:
        return 138.0, 60.0
    if count <= 12:
        return 120.0, 54.0
    if count <= 24:
        return 102.0, 48.0
    return 86.0, 42.0


_SINGLE_RING_LIMIT = 16
_CENTRAL_ROW_DC_LIMIT = 3


def _ellipse_perimeter(radius_x: float, radius_y: float) -> float:
    """Ramanujan's approximation — accurate enough to count node slots."""
    a, b = max(radius_x, radius_y), min(radius_x, radius_y)
    return math.pi * (3 * (a + b) - math.sqrt(max(0.0, (3 * a + b) * (a + 3 * b))))


def _dc_grid(dcs: Sequence[Mapping[str, Any]], dc_size: tuple[float, float]) -> tuple[
    dict[str, tuple[float, float]], float, float,
]:
    """Centre the DCs in a compact grid; return positions and its half-extents."""
    dc_w, dc_h = dc_size
    cx, cy = CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2
    dc_list = list(dcs)
    if not dc_list:
        return {}, 0.0, 0.0
    columns = 1 if len(dc_list) <= 3 else 2
    rows = math.ceil(len(dc_list) / columns)
    step_x, step_y = dc_w + 18.0, dc_h + 14.0
    left = cx - (columns - 1) * step_x / 2
    top = cy - (rows - 1) * step_y / 2
    positions: dict[str, tuple[float, float]] = {}
    for index, row in enumerate(dc_list):
        column, line = index % columns, index // columns
        positions[_text(row.get("node_id"))] = (
            round(left + column * step_x, 2), round(top + line * step_y, 2),
        )
    return positions, (columns * step_x) / 2, (rows * step_y) / 2


def ring_capacity(store_size: tuple[float, float], dc_count: int) -> int:
    """How many stores this canvas can hold without nodes touching."""
    return sum(count for _rx, _ry, count in _ring_plan(store_size, dc_count))


def _ring_plan(store_size: tuple[float, float], dc_count: int) -> list[tuple[float, float, int]]:
    """Ring radii (outermost first) and how many stores each one can carry."""
    store_w, store_h = store_size
    dc_w, dc_h = _dc_size((store_w, store_h))
    _positions, half_w, half_h = _dc_grid(
        [{"node_id": f"_{index}"} for index in range(dc_count)], (dc_w, dc_h),
    )
    cx, cy = CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2
    max_rx = cx - (CANVAS_MARGIN + store_w / 2)
    max_ry = cy - (CANVAS_MARGIN + store_h / 2 + 14.0)
    floor_rx = half_w + store_w * 0.8 if dc_count else store_w * 0.8
    floor_ry = half_h + store_h * 1.1 if dc_count else store_h * 1.1

    plan: list[tuple[float, float, int]] = []
    scale = 1.0
    # Each ring steps inward by a whole node, so two rings can never touch.
    step = max(store_w * 1.25 / max_rx, store_h * 1.9 / max_ry) if max_rx and max_ry else 1.0
    while scale > 0:
        radius_x, radius_y = max_rx * scale, max_ry * scale
        if radius_x < floor_rx or radius_y < floor_ry:
            break
        count = int(_ellipse_perimeter(radius_x, radius_y) // (store_w * 1.16))
        if count < 3:
            break
        plan.append((radius_x, radius_y, count))
        scale -= step
    return plan


def _ring_positions(
    stores: Sequence[Mapping[str, Any]],
    dcs: Sequence[Mapping[str, Any]],
    store_size: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Concentric-ring placement for node sets a single ellipse cannot hold.

    The shared radial layout puts every store on one ellipse around one or two
    DCs, which starts to overlap past ~16 stores or with three or more DCs.
    Rather than change that layout (the movement simulation depends on it), the
    workspace lays those cases out itself: DCs in a compact central grid, stores
    on rings outward, each ring holding only as many nodes as its own
    circumference allows.
    """
    store_w, store_h = store_size
    dc_w, dc_h = _dc_size((store_w, store_h))
    cx, cy = CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2

    positions, _half_w, _half_h = _dc_grid(dcs, (dc_w, dc_h))
    store_list = list(stores)
    if not store_list:
        return positions

    index = 0
    for ring, (radius_x, radius_y, capacity) in enumerate(_ring_plan(store_size, len(dcs))):
        if index >= len(store_list):
            break
        count = min(capacity, len(store_list) - index)
        offset = math.pi / max(1, capacity) if ring % 2 else 0.0
        for slot in range(count):
            angle = offset - math.pi / 2 + 2 * math.pi * slot / count
            positions[_text(store_list[index + slot].get("node_id"))] = (
                round(cx + radius_x * math.cos(angle), 2),
                round(cy + radius_y * math.sin(angle), 2),
            )
        index += count
    return positions


def plan_edge_options(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """One selectable entry per plan move — the network's own picker."""
    options: list[dict[str, str]] = []
    for item in items:
        route_id = _text(item.get("route_id"))
        if not route_id:
            continue
        source = _text(item.get("source_name")) or _text(item.get("source_id")) or "-"
        target = _text(item.get("target_name")) or _text(item.get("target_id")) or "-"
        options.append({
            "route_id": route_id,
            "label": f"{source} → {target}",
            "product": _text(item.get("product_name")) or _text(item.get("product_id")) or "-",
            "qty": _qty_label(item),
        })
    return options


def visible_nodes(
    all_nodes: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    """Restrict the drawn node set so the default view stays a decision picture."""
    nodes = [dict(node) for node in all_nodes]
    if scope == SCOPE_ALL or not routes:
        return nodes
    wanted: set[str] = set()
    for route in routes:
        for key in ("source_id", "target_id", "dc_id"):
            value = _text(route.get(key))
            if value:
                wanted.add(value)
        try:
            for segment in build_route_segments(route, nodes):
                wanted.add(_text(segment["from_node_id"]))
                wanted.add(_text(segment["to_node_id"]))
        except ValueError:
            continue
    kept = [node for node in nodes if _text(node.get("node_id")) in wanted]
    return kept or nodes


def _related_node_ids(
    routes: Sequence[Mapping[str, Any]], nodes: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Every node the drawn moves actually touch (source, target, and any DC)."""
    related: set[str] = set()
    for route in routes:
        for key in ("source_id", "target_id", "dc_id"):
            value = _text(route.get(key))
            if value:
                related.add(value)
        try:
            for segment in build_route_segments(route, nodes):
                related.add(_text(segment["from_node_id"]))
                related.add(_text(segment["to_node_id"]))
        except ValueError:
            continue
    return related


def _node_roles(selected: Mapping[str, Any] | None, nodes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if not selected:
        return {}
    roles: dict[str, str] = {}
    source, target = _text(selected.get("source_id")), _text(selected.get("target_id"))
    if source:
        roles[source] = ROLE_SOURCE
    if target:
        roles[target] = ROLE_TARGET
    if normalize_route_type(selected) == "VIA_DC":
        try:
            for segment in build_route_segments(selected, nodes):
                node_id = _text(segment["to_node_id"])
                if segment["phase"] == "TO_DC" and node_id:
                    roles[node_id] = ROLE_DC
        except ValueError:
            pass
    return roles


def _dc_svg(node: Mapping[str, Any], role: str) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = _text(node.get("node_name")) or _text(node.get("node_id"))
    lines, font = _fit_dc_label(name, width)
    # 이름이 이미 물류센터를 말하고 있으면 아래에 같은 말을 한 번 더 적지 않는다.
    show_kind = not any(token in name for token in ("물류센터", "센터", "DC"))
    highlighted = role == ROLE_DC
    stroke = COLOR_SELECTED if highlighted else "#a98a3d"
    left, top = -width / 2, -height / 2
    role_svg = (
        f'<text class="ws-node-role" x="0" y="{top - 8:.2f}" text-anchor="middle">{_safe(role)}</text>'
        if highlighted else ""
    )
    return (
        f'<g class="ws-node ws-node-dc" transform="translate({x:.2f} {y:.2f})">'
        f"<title>{_safe(name)} · 물류센터</title>"
        f'{role_svg}'
        f'<rect x="{left:.2f}" y="{top + 11:.2f}" width="{width:.2f}" height="{height - 11:.2f}" rx="7" '
        f'fill="#fdf7e6" stroke="{stroke}" stroke-width="{2.6 if highlighted else 1.6}" />'
        f'<path d="M {left - 4:.2f} {top + 13:.2f} L 0 {top - 3:.2f} L {-left + 4:.2f} {top + 13:.2f} Z" '
        f'fill="#f3e4b8" stroke="{stroke}" stroke-width="1.6" />'
        + _dc_name_svg(lines, font, height)
        + (
            f'<text class="ws-node-sub" x="0" y="{height / 2 - 9:.2f}" text-anchor="middle">물류센터</text>'
            if show_kind else ""
        )
        + "</g>"
    )


def _dc_name_svg(lines: Sequence[str], font: float, height: float) -> str:
    baseline = -height * 0.02
    if len(lines) == 1:
        return (
            f'<text class="ws-node-name" x="0" y="{baseline:.2f}" text-anchor="middle" '
            f'font-size="{font:.1f}">{_safe(lines[0])}</text>'
        )
    step = font * 1.16
    return "".join(
        f'<text class="ws-node-name" x="0" y="{baseline - step / 2 + index * step:.2f}" '
        f'text-anchor="middle" font-size="{font:.1f}">{_safe(line)}</text>'
        for index, line in enumerate(lines)
    )


def _store_svg(node: Mapping[str, Any], role: str) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = _text(node.get("node_name")) or _text(node.get("node_id"))
    label, font = _fit_label(name, width, NAME_FONT, NAME_FONT_MIN)
    state = _text(node.get("inventory_state")) or "정상"
    text_color, fill_tint = STATE_STYLES.get(state, STATE_STYLES["정상"])
    highlighted = role in (ROLE_SOURCE, ROLE_TARGET)
    stroke = COLOR_SELECTED if highlighted else COLOR_LINE
    left, top = -width / 2, -height / 2
    role_svg = (
        f'<text class="ws-node-role" x="0" y="{top - 8:.2f}" text-anchor="middle">{_safe(role)}</text>'
        if highlighted else ""
    )
    # The state pill scales with the node so its text stays readable instead of
    # sitting at a fixed 9px that vanished once the SVG was scaled to a column.
    # The cap was measured, not guessed: at 1366 with the sidebar expanded the
    # centre column is ~654px, a 0.70x downscale, and the old 15.0 cap landed the
    # pill at 10.4px on screen. The pill box grows with the font (pill_w reads the
    # same value and is still clamped to the node), so nothing overflows.
    pill_h = max(13.0, min(20.0, height * 0.35))
    pill_font = round(min(STATE_PILL_FONT_MAX, pill_h * 0.84), 1)
    pill_w = max(46.0, min(width - 8.0, _glyph_width(state, pill_font) + 16.0))
    pill_top = height / 2 - pill_h - 3.0
    return (
        f'<g class="ws-node ws-node-store" transform="translate({x:.2f} {y:.2f})">'
        f"<title>{_safe(name)} · 점포 · 재고 {_safe(state)}</title>"
        f"{role_svg}"
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" rx="8" '
        f'fill="#ffffff" stroke="{stroke}" stroke-width="{2.6 if highlighted else 1.2}" />'
        f'<text class="ws-node-name" x="0" y="{-height * 0.08:.2f}" text-anchor="middle" '
        f'font-size="{font:.1f}">{_safe(label)}</text>'
        f'<rect x="{-pill_w / 2:.2f}" y="{pill_top:.2f}" width="{pill_w:.2f}" height="{pill_h:.2f}" '
        f'rx="{pill_h / 2:.2f}" fill="{fill_tint}" stroke="{text_color}" stroke-width="0.9" />'
        f'<text x="0" y="{pill_top + pill_h * 0.74:.2f}" text-anchor="middle" fill="{text_color}" '
        f'font-size="{pill_font}" font-weight="700">{_safe(state)}</text>'
        "</g>"
    )


def _edge_line_svg(
    start: tuple[float, float], end: tuple[float, float], *, selected: bool, via_dc: bool,
) -> str:
    (x1, y1), (x2, y2) = start, end
    color = COLOR_SELECTED if selected else COLOR_PLANNED
    width = 3.4 if selected else 1.9
    dash = ' stroke-dasharray="10 7"' if via_dc else ""
    marker = "ws-arrow-selected" if selected else "ws-arrow-planned"
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-opacity="{0.95 if selected else 0.6}"'
        f'{dash} marker-end="url(#{marker})" />'
    )


Box = tuple[float, float, float, float]  # centre x, centre y, half width, half height


def _chip_box(
    start: tuple[float, float], end: tuple[float, float], label: str, position: float,
) -> Box:
    """Chip centre and half-extents at ``position`` along the segment."""
    (x1, y1), (x2, y2) = start, end
    half = max(22.0, _glyph_width(label, EDGE_FONT) / 2 + 9.0)
    return (
        x1 + (x2 - x1) * position, y1 + (y2 - y1) * position, half, EDGE_CHIP_HEIGHT / 2,
    )


def _overlaps(box: Box, placed: Sequence[Box]) -> bool:
    x, y, half_w, half_h = box
    for other_x, other_y, other_half_w, other_half_h in placed:
        if abs(x - other_x) < half_w + other_half_w + 4 and abs(y - other_y) < half_h + other_half_h + 3:
            return True
    return False


def _edge_label_svg(
    start: tuple[float, float], end: tuple[float, float], label: str, *,
    selected: bool, placed: list[Box],
) -> str:
    """Place one ``planned_qty`` chip, sliding it along the edge to clear the node
    boxes and the chips already on screen. A non-selected chip that finds no free
    spot is dropped rather than printed on top of another number; the selected
    move always keeps its own number.
    """
    color = COLOR_SELECTED if selected else COLOR_PLANNED
    box = None
    for position in (0.5, 0.38, 0.62, 0.3, 0.7, 0.24, 0.76):
        candidate = _chip_box(start, end, label, position)
        if not _overlaps(candidate, placed):
            box = candidate
            break
    if box is None:
        if not selected:
            return ""
        box = _chip_box(start, end, label, 0.5)
    placed.append(box)
    x, y, half, _half_h = box
    return (
        f'<rect class="ws-edge-chip" x="{x - half:.2f}" y="{y - EDGE_CHIP_HEIGHT / 2:.2f}" '
        f'width="{half * 2:.2f}" height="{EDGE_CHIP_HEIGHT:.2f}" rx="{EDGE_CHIP_HEIGHT / 2:.2f}" '
        f'fill="#ffffff" stroke="{color}" stroke-width="1" />'
        f'<text class="ws-edge-label" x="{x:.2f}" y="{y + EDGE_CHIP_HEIGHT * 0.19:.2f}" '
        f'text-anchor="middle" fill="{color}">{_safe(label)}</text>'
    )


def _defs() -> str:
    def marker(name: str, color: str) -> str:
        return (
            f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
            f'markerHeight="6.5" orient="auto-start-reverse">'
            f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{color}" /></marker>'
        )

    return "<defs>" + marker("ws-arrow-selected", COLOR_SELECTED) + marker("ws-arrow-planned", COLOR_PLANNED) + "</defs>"


def _legend(has_via_dc: bool) -> str:
    parts = [
        '<span class="ws-legend-item"><span class="ws-legend-line ws-legend-line-selected"></span>선택한 이동</span>',
        '<span class="ws-legend-item"><span class="ws-legend-line"></span>계획된 다른 이동</span>',
    ]
    if has_via_dc:
        parts.append(
            '<span class="ws-legend-item"><span class="ws-legend-line ws-legend-line-dashed"></span>DC 경유</span>'
        )
    parts.extend(
        f'<span class="ws-legend-item"><span class="ws-legend-dot" '
        f'style="background:{fill};border-color:{color};"></span>{state}</span>'
        for state, (color, fill) in STATE_STYLES.items()
        if state != "이동 대상"
    )
    return '<div class="ws-network-legend">' + "".join(parts) + "</div>"


def build_workspace_network(
    data: Mapping[str, Any] | None,
    plan_items: Sequence[Mapping[str, Any]],
    selected_route_id: str | None = None,
    *,
    scope: str = SCOPE_PLAN,
    store_states: Mapping[str, str] | None = None,
    background_routes: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return ``{"ok", "html", "message"}`` for the central network panel."""
    items = [dict(item) for item in plan_items or []]
    selected_id = _text(selected_route_id)
    selected = next((item for item in items if _text(item.get("route_id")) == selected_id), None)

    if scope == SCOPE_SELECTED and selected is not None:
        drawn = [selected]
    else:
        drawn = items

    all_nodes = build_network_nodes(data, items)
    if not all_nodes:
        return {"ok": False, "html": "", "message": "표시할 점포 정보가 없습니다."}
    for node in all_nodes:
        if classify_node(node) != DC:
            node["inventory_state"] = (store_states or {}).get(_text(node.get("node_id")), "정상")

    nodes = visible_nodes(all_nodes, drawn, scope)
    stores = [node for node in nodes if classify_node(node) != DC]
    dcs = [node for node in nodes if classify_node(node) == DC]
    if not stores and not dcs:
        return {"ok": False, "html": "", "message": "표시할 점포 정보가 없습니다."}

    store_w, store_h = _dimensions(len(stores))
    dc_w, dc_h = _dc_size((store_w, store_h))
    # A picture that cannot fit every store legibly shows the ones involved in the
    # plan and says how many it left out — it never draws them on top of each other.
    capacity = ring_capacity((store_w, store_h), len(dcs))
    dropped = 0
    if len(stores) > capacity:
        related = _related_node_ids(drawn, nodes)
        ordered = sorted(stores, key=lambda node: (
            _text(node.get("node_id")) not in related, _text(node.get("node_id")),
        ))
        dropped = len(stores) - capacity
        stores = ordered[:capacity]
        kept = {_text(node.get("node_id")) for node in stores} | {
            _text(node.get("node_id")) for node in dcs
        }
        nodes = [node for node in nodes if _text(node.get("node_id")) in kept]
    # The shared radial layout is tuned for a handful of stores around one or two
    # DCs; past that it starts to overlap, so the workspace lays those cases out
    # itself instead of shipping a crowded picture.
    if len(stores) > _SINGLE_RING_LIMIT or len(dcs) >= _CENTRAL_ROW_DC_LIMIT:
        coordinates = _ring_positions(stores, dcs, (store_w, store_h))
    else:
        positions = compute_network_layout(
            stores, dcs, CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_MARGIN,
            recommended=set(), store_size=(store_w, store_h),
        )
        coordinates = {
            node_id: (value[0], value[1]) for node_id, value in positions.items()
        }
    roles = _node_roles(selected, nodes)

    background: list[str] = []
    if scope == SCOPE_ALL:
        seen: set[tuple[str, str]] = set()
        for route in list(background_routes)[:_MAX_BACKGROUND_EDGES]:
            source = _text(route.get("source_id") or route.get("from_store_id"))
            target = _text(route.get("target_id") or route.get("to_store_id"))
            pair = tuple(sorted((source, target)))
            if not source or not target or source == target or pair in seen:
                continue
            seen.add(pair)
            start, end = coordinates.get(source), coordinates.get(target)
            if start and end:
                background.append(
                    f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" '
                    f'stroke="{COLOR_BACKGROUND}" stroke-width="1" stroke-opacity="0.55" />'
                )

    edges: list[str] = []
    pending_labels: list[tuple[tuple[float, float], tuple[float, float], str, bool]] = []
    has_via_dc = False
    label_all = len(drawn) <= _MAX_LABELLED_EDGES
    for item in drawn:
        try:
            segments = build_route_segments(item, nodes)
        except ValueError:
            continue
        is_selected = _text(item.get("route_id")) == selected_id
        via_dc = normalize_route_type(item) == "VIA_DC"
        has_via_dc = has_via_dc or via_dc
        label = _qty_label(item) if (is_selected or label_all) else ""
        for index, segment in enumerate(segments):
            start = coordinates.get(_text(segment["from_node_id"]))
            end = coordinates.get(_text(segment["to_node_id"]))
            if not start or not end:
                continue
            edges.append(_edge_line_svg(start, end, selected=is_selected, via_dc=via_dc))
            # 두 구간짜리 DC 경유는 두 번째 구간에만 수량을 적어 화면이 겹치지 않게 한다.
            if label and index == len(segments) - 1:
                pending_labels.append((start, end, label, is_selected))

    # Chips are placed after every line so a number is never drawn under a node
    # box or another edge's number; the selected move always keeps its own chip.
    placed: list[Box] = []
    for group, (box_w, box_h) in ((dcs, (dc_w, dc_h)), (stores, (store_w, store_h))):
        for node in group:
            position = coordinates.get(_text(node.get("node_id")))
            if position:
                placed.append((position[0], position[1], box_w / 2 + 3.0, box_h / 2 + 3.0))
    labels = [
        _edge_label_svg(start, end, label, selected=True, placed=placed)
        for start, end, label, is_selected in pending_labels if is_selected
    ]
    labels += [
        _edge_label_svg(start, end, label, selected=False, placed=placed)
        for start, end, label, is_selected in pending_labels if not is_selected
    ]
    edges.extend(part for part in labels if part)

    if not edges and drawn:
        message = "선택한 조건에서 표시할 이동 경로가 없습니다."
    elif dropped:
        message = f"점포가 많아 이동과 관련된 {len(stores)}곳만 표시했습니다 (표시하지 않은 점포 {dropped}곳)."
    else:
        message = ""

    # Node names are fitted to their own box by _fit_label, so no character
    # budget is guessed here any more.
    shapes = [
        _dc_svg(
            {**node, "x": coordinates[_text(node["node_id"])][0], "y": coordinates[_text(node["node_id"])][1],
             "width": dc_w, "height": dc_h},
            roles.get(_text(node.get("node_id")), ""),
        )
        for node in dcs if _text(node.get("node_id")) in coordinates
    ]
    shapes += [
        _store_svg(
            {**node, "x": coordinates[_text(node["node_id"])][0], "y": coordinates[_text(node["node_id"])][1],
             "width": store_w, "height": store_h},
            roles.get(_text(node.get("node_id")), ""),
        )
        for node in stores if _text(node.get("node_id")) in coordinates
    ]

    svg = (
        f'<svg class="ws-network-svg" viewBox="0 0 {CANVAS_WIDTH:.0f} {CANVAS_HEIGHT:.0f}" '
        'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet" role="img" '
        'aria-label="점포와 DC 사이의 오늘 이동 계획">'
        + _defs()
        + "".join(background)
        + "".join(edges)
        + "".join(shapes)
        + "</svg>"
    )
    return {
        "ok": True,
        "message": message,
        "html": f'<div class="v2-wrap ws-network-shell">{_legend(has_via_dc)}{svg}</div>',
    }


__all__ = [
    "ring_capacity",
    "SCOPE_ALL",
    "SCOPE_OPTIONS",
    "SCOPE_PLAN",
    "SCOPE_SELECTED",
    "STATE_STYLES",
    "build_workspace_network",
    "plan_edge_options",
    "visible_nodes",
]
