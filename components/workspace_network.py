"""Static store/DC/route network for the 재고 운영 Workspace.

This is the decision picture, not an animation: it draws the stores, the DCs and
the moves of the *current execution plan*, and highlights the one move the user
has selected. Node classification and DIRECT / VIA_DC segmentation are reused
from :mod:`simulation.dynamic_network`, so they behave exactly as everywhere else
in the app; the geometry comes from :mod:`simulation.flow_layout`, which lays the
plan out as a flow — 내보내는 점포 왼쪽, 경유 물류센터 가운데, 받는 점포 오른쪽.

The picture is built around one idea: **선택한 이동이 먼저 읽히고, 나머지는 맥락으로
남는다.** As a plan grows past a dozen or so moves, showing every name, badge and
number at the same visual weight stops being informative, so this module ranks
what it draws instead of shrinking it:

* the selected move keeps a wider node box, a role label, its 상태 badge and its
  planned_qty at every size — it is never reduced to match its neighbours,
* the other planned moves keep their names and their line, and give up their
  quantity chip first,
* nodes no drawn move touches stay as light context.

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
    normalize_route_type,
)
from simulation.flow_layout import (
    CANVAS_WIDTH,
    DC_BAND,
    FlowLayout,
    clip_to_box,
    compute_flow_layout,
)

SCOPE_PLAN = "현재 실행계획"
SCOPE_SELECTED = "선택한 이동"
SCOPE_ALL = "전체 네트워크"
SCOPE_OPTIONS = (SCOPE_PLAN, SCOPE_SELECTED, SCOPE_ALL)

# One restrained palette. 선택 경로 / 다른 계획 경로 / 배경 경로 만 구분한다.
COLOR_SELECTED = "#1d6fa3"
COLOR_PLANNED = "#8fa3b5"
COLOR_BACKGROUND = "#d8dee5"
COLOR_LINE = "#cbd5df"
COLOR_CONTEXT = "#dfe5ec"

STATE_STYLES: dict[str, tuple[str, str]] = {
    "과잉": ("#b26a1f", "#fdeecb"),
    "부족": ("#b23b3b", "#fbe0e0"),
    "정상": ("#2f7d5b", "#e3f3ea"),
    "이동 대상": ("#2d6fa8", "#e2eefb"),
}

#: A state is never carried by colour alone: each one also has its own outline
#: shape, used wherever the picture is too dense for the written badge.
STATE_MARKERS: dict[str, str] = {
    "과잉": "M 0 -5.2 L 5.2 4.2 L -5.2 4.2 Z",
    "부족": "M -5.2 -4.2 L 5.2 -4.2 L 0 5.2 Z",
    "정상": "M -4.4 -4.4 L 4.4 -4.4 L 4.4 4.4 L -4.4 4.4 Z",
    "이동 대상": "M 0 -5.6 L 5.6 0 L 0 5.6 L -5.6 0 Z",
}

ROLE_SOURCE = "출발"
ROLE_TARGET = "도착"
ROLE_DC = "경유 DC"

#: Emphasis levels. FOCUS is the selected move, PLAN is every other drawn move,
#: CONTEXT is a node no drawn move touches (only reachable from 전체 네트워크).
LEVEL_FOCUS = "focus"
LEVEL_PLAN = "plan"
LEVEL_CONTEXT = "context"

_MAX_BACKGROUND_EDGES = 24
_MAX_LABELLED_EDGES = 8
#: Above this many drawn nodes the written 과잉/부족/정상 badge is kept for the
#: selected move and for the moves competing with it, and every other node states
#: the same thing with its outline marker instead. The number is not a guess: it
#: is the node count at which the store box drops below 58 units high, which is
#: where the badge stops having room to sit beside a full store name.
_BADGE_TEXT_LIMIT = 26

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


def _route_pairs(
    route: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    try:
        segments = build_route_segments(route, nodes)
    except ValueError:
        return []
    return [
        (_text(segment["from_node_id"]), _text(segment["to_node_id"])) for segment in segments
    ]


def visible_nodes(
    all_nodes: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    """Restrict the drawn node set so the default view stays a decision picture."""
    nodes = [dict(node) for node in all_nodes]
    if scope == SCOPE_ALL or not routes:
        return nodes
    wanted = _related_node_ids(routes, nodes)
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
        for start, end in _route_pairs(route, nodes):
            related.add(start)
            related.add(end)
    return related


def _node_roles(
    selected: Mapping[str, Any] | None, nodes: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if not selected:
        return {}
    roles: dict[str, str] = {}
    source, target = _text(selected.get("source_id")), _text(selected.get("target_id"))
    if source:
        roles[source] = ROLE_SOURCE
    if target:
        roles[target] = ROLE_TARGET
    if normalize_route_type(selected) == "VIA_DC":
        for index, (_start, end) in enumerate(_route_pairs(selected, nodes)):
            if index == 0 and end:
                roles[end] = ROLE_DC
    return roles


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def _state_marker_svg(state: str, x: float, y: float) -> str:
    """The state as an outline shape, for nodes too dense to carry the badge."""
    color, fill = STATE_STYLES.get(state, STATE_STYLES["정상"])
    path = STATE_MARKERS.get(state, STATE_MARKERS["정상"])
    return (
        f'<path class="ws-node-marker" transform="translate({x:.2f} {y:.2f})" d="{path}" '
        f'fill="{fill}" stroke="{color}" stroke-width="1.5" stroke-linejoin="round" />'
    )


def _state_pill_svg(state: str, width: float, height: float) -> str:
    # The state pill scales with the node so its text stays readable instead of
    # sitting at a fixed 9px that vanished once the SVG was scaled to a column.
    # The cap was measured, not guessed: at 1366 with the sidebar expanded the
    # centre column is ~654px, a 0.70x downscale, and the old 15.0 cap landed the
    # pill at 10.4px on screen. The pill box grows with the font (pill_w reads the
    # same value and is still clamped to the node), so nothing overflows.
    color, fill = STATE_STYLES.get(state, STATE_STYLES["정상"])
    pill_h = max(13.0, min(20.0, height * 0.35))
    pill_font = round(min(STATE_PILL_FONT_MAX, pill_h * 0.84), 1)
    pill_w = max(46.0, min(width - 8.0, _glyph_width(state, pill_font) + 16.0))
    pill_top = height / 2 - pill_h - 3.0
    return (
        f'<rect x="{-pill_w / 2:.2f}" y="{pill_top:.2f}" width="{pill_w:.2f}" '
        f'height="{pill_h:.2f}" rx="{pill_h / 2:.2f}" fill="{fill}" stroke="{color}" '
        f'stroke-width="0.9" />'
        f'<text x="0" y="{pill_top + pill_h * 0.74:.2f}" text-anchor="middle" fill="{color}" '
        f'font-size="{pill_font}" font-weight="700">{_safe(state)}</text>'
    )


def _store_svg(
    node: Mapping[str, Any], role: str, level: str, *, badge_text: bool,
) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = _text(node.get("node_name")) or _text(node.get("node_id"))
    state = _text(node.get("inventory_state")) or "정상"
    focused = level == LEVEL_FOCUS
    context = level == LEVEL_CONTEXT
    # A written badge takes the bottom of the box, so the name gets the rest;
    # with the outline marker the name may use the full width of the node.
    name_room = width - (0.0 if badge_text else 22.0)
    label, font = _fit_label(name, name_room, NAME_FONT, NAME_FONT_MIN)
    stroke = COLOR_SELECTED if focused else COLOR_CONTEXT if context else COLOR_LINE
    left, top = -width / 2, -height / 2
    # Just above the box, inside the row clearance the layout reserves for it —
    # the old offset put this label on the state badge of the node above.
    role_svg = (
        f'<text class="ws-node-role" x="0" y="{top - 2:.2f}" text-anchor="middle">{_safe(role)}</text>'
        if focused and role else ""
    )
    baseline = -height * 0.08 if badge_text else height * 0.09
    dashed = ' stroke-dasharray="4 3"' if context else ""
    fill = "#fbfcfd" if context else "#ffffff"
    return (
        f'<g class="ws-node ws-node-store ws-node-{level}" transform="translate({x:.2f} {y:.2f})">'
        f"<title>{_safe(name)} · 점포 · 재고 {_safe(state)}</title>"
        f"{role_svg}"
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" rx="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{2.6 if focused else 1.2}"{dashed} />'
        f'<text class="ws-node-name" x="{0.0 if badge_text else 9.0:.2f}" y="{baseline:.2f}" '
        f'text-anchor="middle" style="font-size:{font:.1f}px">{_safe(label)}</text>'
        + (
            _state_pill_svg(state, width, height) if badge_text
            else _state_marker_svg(state, left + 12.0, 0.0)
        )
        + "</g>"
    )


def _dc_name_svg(lines: Sequence[str], font: float, height: float) -> str:
    baseline = -height * 0.02
    if len(lines) == 1:
        return (
            f'<text class="ws-node-name" x="0" y="{baseline:.2f}" text-anchor="middle" '
            f'style="font-size:{font:.1f}px">{_safe(lines[0])}</text>'
        )
    step = font * 1.16
    return "".join(
        f'<text class="ws-node-name" x="0" y="{baseline - step / 2 + index * step:.2f}" '
        f'text-anchor="middle" style="font-size:{font:.1f}px">{_safe(line)}</text>'
        for index, line in enumerate(lines)
    )


def _dc_svg(node: Mapping[str, Any], role: str, *, sub_label: str) -> str:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    name = _text(node.get("node_name")) or _text(node.get("node_id"))
    lines, font = _fit_dc_label(name, width)
    highlighted = role == ROLE_DC
    stroke = COLOR_SELECTED if highlighted else "#a98a3d"
    left, top = -width / 2, -height / 2
    # The DC roof rises above the box, so its role label clears the apex, not the
    # rectangle — at the store offset the two touched.
    role_svg = (
        f'<text class="ws-node-role" x="0" y="{top - 11:.2f}" text-anchor="middle">{_safe(role)}</text>'
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
            f'<text class="ws-node-sub" x="0" y="{height / 2 - 9:.2f}" text-anchor="middle">'
            f'{_safe(sub_label)}</text>'
            if sub_label else ""
        )
        + "</g>"
    )


def _dc_sub_label(name: str, node_id: str, dc_count: int) -> str:
    """What the second line of a 물류센터 box says.

    With more than one DC the picture must let a reader tell DC01 from DC02 even
    when the two names look alike, so the code is written under the name. With a
    single DC the code carries nothing, and the line only repeats 물류센터 when the
    name does not already say it.
    """
    if dc_count > 1 and node_id and node_id.upper() != name.upper():
        return node_id
    return "" if any(token in name for token in ("물류센터", "센터", "DC")) else "물류센터"


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #
def _bow_control(
    start: tuple[float, float], end: tuple[float, float], apex_x: float,
) -> tuple[float, float]:
    """Control point that puts the middle of the bow exactly on ``apex_x``.

    A quadratic passes through ``(P0 + 2C + P1) / 4`` at its midpoint, so solving
    for C is what makes a same-band move leave its column by a known amount
    instead of by a guessed one — the earlier relative bulge pushed the curve off
    the side of the canvas whenever the column already hugged the edge.
    """
    return (
        (4 * apex_x - start[0] - end[0]) / 2,
        (start[1] + end[1]) / 2,
    )


def _edge_line_svg(
    start: tuple[float, float], end: tuple[float, float], *,
    selected: bool, via_dc: bool, control: tuple[float, float] | None,
) -> str:
    """One drawn segment, stopped at both node boxes so its arrowhead is visible.

    A move between two nodes of the same band would otherwise run straight down
    the column through every box between them, so it is bowed out through the
    corridor instead (``control``).
    """
    (x1, y1), (x2, y2) = start, end
    color = COLOR_SELECTED if selected else COLOR_PLANNED
    width = 3.4 if selected else 1.9
    dash = ' stroke-dasharray="10 7"' if via_dc else ""
    marker = "ws-arrow-selected" if selected else "ws-arrow-planned"
    shape = (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"'
        if control is None else
        f'<path fill="none" d="M {x1:.2f} {y1:.2f} Q {control[0]:.2f} '
        f'{control[1]:.2f} {x2:.2f} {y2:.2f}"'
    )
    return (
        f'{shape} stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'stroke-opacity="{0.95 if selected else 0.55}"{dash} marker-end="url(#{marker})" />'
    )


def _ribbon_svg(points: Sequence[tuple[float, float]]) -> str:
    """A single soft band behind a two-segment DC 경유 move.

    Two separate lines through a DC read as two unrelated plans; one ribbon
    running 출발 → DC → 도착 says they are one move without adding a colour.
    """
    if len(points) < 3:
        return ""
    path = " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)
    return (
        f'<path class="ws-edge-ribbon" d="M {path}" fill="none" stroke="{COLOR_SELECTED}" '
        f'stroke-width="11" stroke-opacity="0.16" stroke-linecap="round" stroke-linejoin="round" />'
    )


Box = tuple[float, float, float, float]  # centre x, centre y, half width, half height


def _chip_box(
    start: tuple[float, float], end: tuple[float, float], label: str,
    position: float, offset: float = 0.0,
    control: tuple[float, float] | None = None,
) -> Box:
    """Chip centre and half-extents at ``position`` along the drawn edge, pushed
    ``offset`` units to the side of it.

    ``control`` is the bow of a same-band edge. The chip has to ride the line the
    reader actually sees: measured against the straight chord instead, a bowed
    edge puts its number back inside the column of boxes it was bowed out of.
    """
    (x1, y1), (x2, y2) = start, end
    half = max(22.0, _glyph_width(label, EDGE_FONT) / 2 + 9.0)
    if control is None:
        centre = (x1 + (x2 - x1) * position, y1 + (y2 - y1) * position)
        tangent = (x2 - x1, y2 - y1)
    else:
        rest = 1.0 - position
        centre = (
            rest * rest * x1 + 2 * rest * position * control[0] + position * position * x2,
            rest * rest * y1 + 2 * rest * position * control[1] + position * position * y2,
        )
        tangent = (
            2 * rest * (control[0] - x1) + 2 * position * (x2 - control[0]),
            2 * rest * (control[1] - y1) + 2 * position * (y2 - control[1]),
        )
    length = math.hypot(*tangent) or 1.0
    normal_x, normal_y = -tangent[1] / length, tangent[0] / length
    return (
        centre[0] + normal_x * offset,
        centre[1] + normal_y * offset,
        half,
        EDGE_CHIP_HEIGHT / 2,
    )


def _overlap_area(box: Box, placed: Sequence[Box]) -> float:
    """How much of ``box`` a already-drawn box would cover, with a little clearance.

    Zero means the chip is free-standing. The value is used twice: any candidate
    scoring zero is taken immediately, and when a plan is dense enough that no
    position is free, the selected move's chip takes the least-covered one rather
    than landing blindly in the middle of a node.
    """
    x, y, half_w, half_h = box
    total = 0.0
    for other_x, other_y, other_half_w, other_half_h in placed:
        overlap_x = half_w + other_half_w + 4 - abs(x - other_x)
        overlap_y = half_h + other_half_h + 3 - abs(y - other_y)
        if overlap_x > 0 and overlap_y > 0:
            total += overlap_x * overlap_y
    return total


def _overlaps(box: Box, placed: Sequence[Box]) -> bool:
    return _overlap_area(box, placed) > 0.0


_CHIP_POSITIONS = (0.5, 0.38, 0.62, 0.3, 0.7, 0.24, 0.76, 0.44, 0.56, 0.18, 0.82)
_CHIP_OFFSETS = (0.0, 28.0, -28.0, 54.0, -54.0, 82.0, -82.0)


def _inside(box: Box, bounds: tuple[float, float]) -> bool:
    x, y, half_w, half_h = box
    return (
        x - half_w >= 0 and x + half_w <= bounds[0]
        and y - half_h >= 0 and y + half_h <= bounds[1]
    )


def _clamp_inside(box: Box, bounds: tuple[float, float]) -> Box:
    x, y, half_w, half_h = box
    return (
        min(max(x, half_w), bounds[0] - half_w),
        min(max(y, half_h), bounds[1] - half_h),
        half_w, half_h,
    )


def _edge_label_svg(
    start: tuple[float, float], end: tuple[float, float], label: str, *,
    selected: bool, placed: list[Box], bounds: tuple[float, float],
    control: tuple[float, float] | None = None,
) -> str:
    """Place one ``planned_qty`` chip, sliding it along the edge — and, when the
    edge is too short to hold it, just beside the edge — to clear the node boxes,
    the chips already on screen and the edge of the canvas. A non-selected chip
    that finds no free spot is dropped rather than printed on top of another
    number; the selected move always keeps its own number.
    """
    color = COLOR_SELECTED if selected else COLOR_PLANNED
    # Of every spot that is actually free, take the one nearest the middle of the
    # edge — a number that has drifted to the far end of its line reads as if it
    # belonged to the node it ended up beside.
    middle = _chip_box(start, end, label, 0.5, 0.0, control)
    free: tuple[float, Box] | None = None
    crowded: tuple[float, Box] | None = None
    for offset in _CHIP_OFFSETS:
        for position in _CHIP_POSITIONS:
            candidate = _chip_box(start, end, label, position, offset, control)
            distance = math.dist(candidate[:2], middle[:2])
            covered = _overlap_area(candidate, placed)
            if covered <= 0.0 and _inside(candidate, bounds):
                if free is None or distance < free[0]:
                    free = (distance, candidate)
            elif crowded is None or covered < crowded[0]:
                crowded = (covered, candidate)
    if free is not None:
        box = free[1]
    elif selected and crowded is not None:
        box = _clamp_inside(crowded[1], bounds)
    else:
        return ""
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


def _legend(has_via_dc: bool, marker_states: bool) -> str:
    parts = [
        '<span class="ws-legend-item"><span class="ws-legend-line ws-legend-line-selected"></span>선택한 이동</span>',
        '<span class="ws-legend-item"><span class="ws-legend-line"></span>계획된 다른 이동</span>',
    ]
    if has_via_dc:
        parts.append(
            '<span class="ws-legend-item"><span class="ws-legend-line ws-legend-line-dashed"></span>DC 경유</span>'
        )
    for state, (color, fill) in STATE_STYLES.items():
        if state == "이동 대상":
            continue
        if marker_states:
            parts.append(
                '<span class="ws-legend-item">'
                '<svg class="ws-legend-shape" viewBox="-8 -8 16 16" aria-hidden="true">'
                f'<path d="{STATE_MARKERS[state]}" fill="{fill}" stroke="{color}" '
                'stroke-width="1.5" stroke-linejoin="round" /></svg>'
                f"{state}</span>"
            )
        else:
            parts.append(
                f'<span class="ws-legend-item"><span class="ws-legend-dot" '
                f'style="background:{fill};border-color:{color};"></span>{state}</span>'
            )
    return '<div class="ws-network-legend">' + "".join(parts) + "</div>"


# --------------------------------------------------------------------------- #
# The picture
# --------------------------------------------------------------------------- #
def _drawn_segments(
    drawn: Sequence[Mapping[str, Any]], nodes: Sequence[Mapping[str, Any]], selected_id: str,
) -> list[dict[str, Any]]:
    """Resolve every drawn move once: its node pairs, its type, its quantity."""
    resolved: list[dict[str, Any]] = []
    for item in drawn:
        pairs = _route_pairs(item, nodes)
        if not pairs:
            continue
        resolved.append({
            "route_id": _text(item.get("route_id")),
            "pairs": pairs,
            "via_dc": normalize_route_type(item) == "VIA_DC",
            "selected": _text(item.get("route_id")) == selected_id,
            "label": _qty_label(item),
            "source_id": _text(item.get("source_id")),
            "target_id": _text(item.get("target_id")),
        })
    return resolved


def _half_extents(
    node_id: str, layout: FlowLayout, focus_ids: set[str],
) -> tuple[float, float]:
    is_dc = layout.bands.get(node_id) == DC_BAND
    width = layout.dc_size[0] if is_dc else layout.store_size[0]
    height = layout.dc_size[1] if is_dc else layout.store_size[1]
    if node_id in focus_ids and not is_dc:
        width = layout.focus_width(node_id)
    return width / 2, height / 2


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

    drawn = [selected] if (scope == SCOPE_SELECTED and selected is not None) else items

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

    segments = _drawn_segments(drawn, nodes, selected_id)
    edge_pairs = [pair for segment in segments for pair in segment["pairs"]]
    related = _related_node_ids(drawn, nodes)

    layout = compute_flow_layout(
        [node.get("node_id") for node in stores],
        [node.get("node_id") for node in dcs],
        edge_pairs,
        keep=related,
    )
    coordinates = layout.positions
    dropped = set(layout.dropped)
    if dropped:
        stores = [node for node in stores if _text(node.get("node_id")) not in dropped]

    roles = _node_roles(selected, nodes)
    focus_ids = set(roles)
    store_w, store_h = layout.store_size
    dc_w, dc_h = layout.dc_size
    badge_text = len(coordinates) <= _BADGE_TEXT_LIMIT

    def level(node_id: str) -> str:
        if node_id in focus_ids:
            return LEVEL_FOCUS
        return LEVEL_PLAN if node_id in related else LEVEL_CONTEXT

    # --- background (전체 네트워크 only) ---------------------------------------
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

    # --- edges -----------------------------------------------------------------
    ribbons: list[str] = []
    plan_edges: list[str] = []
    focus_edges: list[str] = []
    pending: list[dict[str, Any]] = []
    has_via_dc = False
    label_all = len(segments) <= _MAX_LABELLED_EDGES
    selected_ends = {
        _text((selected or {}).get("source_id")), _text((selected or {}).get("target_id")),
    } - {""}

    for segment in segments:
        has_via_dc = has_via_dc or segment["via_dc"]
        is_selected = segment["selected"]
        # 선택 이동은 언제나, 선택과 출발/도착을 공유하는 대안은 그 다음, 나머지 수량은
        # 계획이 작을 때만 적는다.
        competing = bool({segment["source_id"], segment["target_id"]} & selected_ends)
        priority = 0 if is_selected else 1 if competing else 2
        show_label = segment["label"] and (is_selected or competing or label_all)

        drawn_points: list[tuple[float, float]] = []
        clipped: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float] | None]] = []
        for index, (from_id, to_id) in enumerate(segment["pairs"]):
            start, end = coordinates.get(from_id), coordinates.get(to_id)
            if not start or not end:
                continue
            from_half = _half_extents(from_id, layout, focus_ids)
            to_half = _half_extents(to_id, layout, focus_ids)
            clipped_start = clip_to_box(start, end, *from_half)
            clipped_end = clip_to_box(end, start, *to_half)
            band = layout.bands.get(from_id, "")
            control = (
                _bow_control(clipped_start, clipped_end, layout.corridor_x(band))
                if band and band == layout.bands.get(to_id) else None
            )
            shape = _edge_line_svg(
                clipped_start, clipped_end, selected=is_selected,
                via_dc=segment["via_dc"], control=control,
            )
            (focus_edges if is_selected else plan_edges).append(shape)
            clipped.append((clipped_start, clipped_end, control))
            if not drawn_points:
                drawn_points.append(clipped_start)
            drawn_points.append(end if index < len(segment["pairs"]) - 1 else clipped_end)
        # 한 이동에는 수량 하나. 두 구간짜리 DC 경유는 둘 중 더 긴 구간에 적는다 —
        # 짧은 구간에는 수량 칩이 통째로 들어가지 않아 옆 점포 위로 밀려난다.
        if show_label and clipped:
            longest = max(clipped, key=lambda part: math.dist(part[0], part[1]))
            pending.append({
                "start": longest[0], "end": longest[1], "control": longest[2],
                "label": segment["label"], "priority": priority,
            })
        if is_selected and segment["via_dc"] and len(drawn_points) >= 3:
            ribbons.append(_ribbon_svg(drawn_points))

    # --- quantity chips --------------------------------------------------------
    # Chips are placed after every line so a number is never drawn under a node
    # box or another edge's number; the selected move always keeps its own chip.
    placed: list[Box] = []
    for node in dcs + stores:
        node_id = _text(node.get("node_id"))
        position = coordinates.get(node_id)
        if position:
            half_w, half_h = _half_extents(node_id, layout, focus_ids)
            placed.append((position[0], position[1], half_w + 3.0, half_h + 3.0))
    chips: list[str] = []
    for chip in sorted(pending, key=lambda row: row["priority"]):
        chips.append(_edge_label_svg(
            chip["start"], chip["end"], chip["label"],
            selected=chip["priority"] == 0, placed=placed, control=chip["control"],
            bounds=(layout.width, layout.height),
        ))

    # --- nodes -----------------------------------------------------------------
    context_shapes: list[str] = []
    plan_shapes: list[str] = []
    focus_shapes: list[str] = []
    for node in dcs:
        node_id = _text(node.get("node_id"))
        if node_id not in coordinates:
            continue
        x, y = coordinates[node_id]
        name = _text(node.get("node_name")) or node_id
        shape = _dc_svg(
            {**node, "x": x, "y": y, "width": dc_w, "height": dc_h},
            roles.get(node_id, ""),
            sub_label=_dc_sub_label(name, node_id, len(dcs)),
        )
        (focus_shapes if node_id in focus_ids else plan_shapes).append(shape)
    for node in stores:
        node_id = _text(node.get("node_id"))
        if node_id not in coordinates:
            continue
        x, y = coordinates[node_id]
        node_level = level(node_id)
        width = layout.focus_width(node_id) if node_level == LEVEL_FOCUS else store_w
        shape = _store_svg(
            {**node, "x": x, "y": y, "width": width, "height": store_h},
            roles.get(node_id, ""), node_level,
            badge_text=badge_text or node_level == LEVEL_FOCUS,
        )
        bucket = (
            focus_shapes if node_level == LEVEL_FOCUS
            else plan_shapes if node_level == LEVEL_PLAN else context_shapes
        )
        bucket.append(shape)

    if not (plan_edges or focus_edges) and drawn:
        message = "선택한 조건에서 표시할 이동 경로가 없습니다."
    elif dropped:
        message = (
            f"점포가 많아 이동과 관련된 {len(stores)}곳만 표시했습니다 "
            f"(표시하지 않은 점포 {len(dropped)}곳)."
        )
    else:
        message = ""

    svg = (
        f'<svg class="ws-network-svg" viewBox="0 0 {layout.width:.0f} {layout.height:.0f}" '
        f'style="aspect-ratio: {layout.width:.0f} / {layout.height:.0f};" '
        'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet" role="img" '
        'aria-label="점포와 DC 사이의 오늘 이동 계획">'
        + _defs()
        + "".join(background)
        + "".join(ribbons)
        + "".join(plan_edges)
        + "".join(focus_edges)
        + "".join(context_shapes)
        + "".join(plan_shapes)
        + "".join(focus_shapes)
        + "".join(part for part in chips if part)
        + "</svg>"
    )
    return {
        "ok": True,
        "message": message,
        "html": (
            f'<div class="v2-wrap ws-network-shell">'
            f'{_legend(has_via_dc, not badge_text)}{svg}</div>'
        ),
    }


__all__ = [
    "SCOPE_ALL",
    "SCOPE_OPTIONS",
    "SCOPE_PLAN",
    "SCOPE_SELECTED",
    "STATE_MARKERS",
    "STATE_STYLES",
    "build_workspace_network",
    "plan_edge_options",
    "visible_nodes",
]
