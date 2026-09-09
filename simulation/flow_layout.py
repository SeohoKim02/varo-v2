"""Deterministic left → right flow layout for the 재고 운영 network picture.

Varo's central picture is a *flow*, not a graph study. The question it answers is
어디서 → (어떤 경유지로) → 어디로 → 몇 개, so the geometry says the same thing:

    왼쪽    재고를 내보내는 점포
    가운데  경유 물류센터
    오른쪽  재고를 받는 점포

That is why this module exists next to — and not instead of — the radial
``compute_network_layout`` in :mod:`simulation.dynamic_network`: the movement
simulation draws one DC with its stores around it and depends on that shape, while
the workspace draws a plan and needs direction. Nothing here is imported by the
simulation, and nothing here feeds a calculation: it only decides *where a box is
drawn*.

Three properties are contracts, not side effects:

* **Deterministic.** The result depends only on the node ids and the edges. The
  same plan always lands in the same place, so re-running the app or picking a
  different move never makes the picture jump.
* **Input-order independent.** Every band starts from a sorted id list and every
  tie is broken by node id, so shuffling the candidate rows cannot move a node.
* **Never overlapping.** Nodes sit on a lane/row grid whose spacing is derived
  from the node box itself, so two boxes cannot be drawn on top of each other at
  any size. When a node set cannot fit legibly the caller is told which nodes
  were left out — the picture never crowds instead.

Density is absorbed in this order: grow the canvas *height* first (height is free
— the SVG is scaled to the width of its column, so a taller viewBox does not
shrink one glyph), then split a band into two lanes, and only then leave nodes
out. Nothing in this module shrinks text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

#: Band names. A store sits in ``SOURCE_BAND`` or ``TARGET_BAND``, a 물류센터 in
#: ``DC_BAND``.
SOURCE_BAND = "source"
DC_BAND = "dc"
TARGET_BAND = "target"

CANVAS_WIDTH = 940.0
#: The canvas a plan too big for its content gets. It is also the height the lane
#: decision below is made against, so it must not be lowered to trim empty space:
#: a shorter default would give a mid-sized plan a second lane it does not need.
MIN_HEIGHT = 620.0
#: Measured ceiling: at 1366 with the sidebar open the centre column is ~646px,
#: so the SVG is drawn at 0.685x and 1130 user units land at ~774px — a panel
#: just taller than that laptop's viewport, which is what the 60-node case costs
#: and the smaller ones never reach. It is the smallest ceiling that still lets
#: two lanes hold 30 stores a side at the row clearance below.
MAX_HEIGHT = 1190.0

SIDE_PAD = 18.0
BAND_GAP = 26.0
LANE_GAP = 12.0
#: Vertical room between two rows. It is not a cosmetic gap: the 출발 / 도착 role
#: label of a selected node is drawn above its box, and at the old 10-16 units it
#: was painted over the state badge of the node above. 22 is that label measured
#: in Chrome, not guessed: at 17.5px (its size on a narrow desktop) its box is
#: 16px tall, which is 19.6 user units above its own box top, so 22 leaves a
#: 2.4-unit margin. Every row carries the clearance because the layout must not
#: change when the selection does.
ROW_GAP = 22.0
#: The same room for a 물류센터, whose roof rises 3 units above its box top and
#: whose role label therefore has to start higher.
DC_ROW_GAP = 34.0
#: Room above the topmost box for that role label.
TOP_PAD = 30.0
#: At or below this many drawn nodes the picture is sized to what it draws rather
#: than to :data:`MIN_HEIGHT`. Measured in Chrome at 1920 before this existed: a
#: two-node move left 207px of empty canvas above it and 222px below (13% of the
#: panel was the picture), and a four-node plan filled a quarter of its card — a
#: plan that small reads as a broken screen, not as a small plan. 10 is where the
#: default canvas stops being mostly empty (a 10-node plan already fills ~64% of
#: it) and it sits well clear of the dense sizes, whose geometry is unchanged:
#: 16 and 24 nodes keep the MIN_HEIGHT canvas, 40 and 60 already outgrew it.
SMALL_NETWORK_NODES = 10
#: Room left above and below a small picture *on top of* the ``TOP_PAD`` the rows
#: already reserve. TOP_PAD alone is the role label's own clearance, so without
#: this the 출발 / 도착 label of a focused node would sit ~8px under the card edge.
SMALL_NETWORK_PAD = 26.0
#: Two lanes per band. A third would push a store box under ~110 units wide,
#: where a Korean store name has to be cut to five characters — leaving context
#: stores out and saying so is more honest than drawing unreadable ones.
MAX_LANES = 2
_SWEEPS = 4


@dataclass(frozen=True)
class FlowLayout:
    """Where every node is drawn, and how much room the picture needs."""

    positions: dict[str, tuple[float, float]]
    bands: dict[str, str]
    width: float
    height: float
    store_size: tuple[float, float]
    dc_size: tuple[float, float]
    lanes: dict[str, int]
    lane_width: dict[str, float]
    inner_edge: dict[str, float] = field(default_factory=dict)
    dropped: tuple[str, ...] = ()

    def corridor_x(self, band: str) -> float:
        """Middle of the empty lane between a band's column and the DC band.

        A move whose two ends sit in the same band would otherwise be drawn as a
        straight line down its own column, through every box between them. It is
        bowed out through here instead — the one part of the canvas that carries
        nothing but moves.
        """
        has_dc = any(band == DC_BAND for band in self.bands.values())
        half_dc = self.dc_size[0] / 2 if has_dc else 0.0
        if band == TARGET_BAND:
            return (self.width / 2 + half_dc + self.inner_edge.get(TARGET_BAND, self.width)) / 2
        return (self.inner_edge.get(SOURCE_BAND, 0.0) + self.width / 2 - half_dc) / 2

    def focus_width(self, node_id: str) -> float:
        """Box width for a node the user has selected.

        The selected move is never shrunk to match its neighbours (a plan is read
        from its own move first), so a focus node is widened into the free space
        its lane already owns. It stays inside the lane, so widening one node can
        never push it onto another.
        """
        band = self.bands.get(node_id, SOURCE_BAND)
        base = self.dc_size[0] if band == DC_BAND else self.store_size[0]
        return round(min(base * 1.35, max(base, self.lane_width.get(band, base) - 8.0)), 2)


def store_dimensions(node_count: int) -> tuple[float, float]:
    """Store box size for a drawn node count.

    The height floor of 52 is measured, not chosen: the 과잉/부족/정상 pill is
    sized from the node (``height × 0.35``, capped), and at 52 the pill font lands
    at 15.3 user units — ~10.2px once the SVG is scaled to the narrowest supported
    centre column. Anything shorter would put a state badge under 10px, so density
    is absorbed by height and lanes instead of by shrinking the box.
    """
    if node_count <= 8:
        return 180.0, 66.0
    if node_count <= 14:
        return 170.0, 62.0
    if node_count <= 26:
        return 160.0, 58.0
    if node_count <= 44:
        return 150.0, 54.0
    return 142.0, 52.0


def dc_dimensions(store_size: tuple[float, float]) -> tuple[float, float]:
    """물류센터 box size — always wider and taller than a store box, so the two
    kinds of node stay distinguishable by shape alone."""
    store_w, store_h = store_size
    return round(max(152.0, store_w * 1.04), 2), round(max(74.0, store_h * 1.18), 2)


def _row_step(store_h: float) -> float:
    return store_h + ROW_GAP


def _rows_that_fit(height: float, box_h: float, step: float, lanes: int) -> int:
    """How many rows of boxes fit between the two role-label margins."""
    stagger = step / 2 if lanes >= 2 else 0.0
    usable = height - box_h - 2 * TOP_PAD - stagger
    if usable < 0:
        return 1
    return int(usable // step) + 1


def _span(rows: int, step: float, lanes: int) -> float:
    return (rows - 1) * step + (step / 2 if lanes >= 2 else 0.0)


def _clean(ids: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in ids:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return sorted(out)


def _sides(
    store_ids: Sequence[str], edges: Sequence[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Split stores into 내보내는 쪽 / 받는 쪽 without ever duplicating a node.

    A store that both gives and receives keeps one identity and is placed on the
    side of its *dominant* role. When the two roles are equal — and for a context
    store no drawn move touches — the node goes to whichever band is currently
    shorter, in sorted id order, so the split stays balanced and deterministic.
    """
    known = set(store_ids)
    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    for source, target in edges:
        if source in known:
            out_degree[source] = out_degree.get(source, 0) + 1
        if target in known:
            in_degree[target] = in_degree.get(target, 0) + 1

    left: list[str] = []
    right: list[str] = []
    undecided: list[str] = []
    for node_id in store_ids:
        gives, takes = out_degree.get(node_id, 0), in_degree.get(node_id, 0)
        if gives > takes:
            left.append(node_id)
        elif takes > gives:
            right.append(node_id)
        else:
            undecided.append(node_id)
    for node_id in undecided:
        (left if len(left) <= len(right) else right).append(node_id)
    return sorted(left), sorted(right)


def _rank_map(*orders: Sequence[str]) -> dict[str, float]:
    ranks: dict[str, float] = {}
    for order in orders:
        last = max(1, len(order) - 1)
        for index, node_id in enumerate(order):
            ranks[node_id] = index / last
    return ranks


def _sweep(
    order: Sequence[str], neighbours: Mapping[str, set[str]], ranks: Mapping[str, float],
) -> list[str]:
    """Barycentre pass: a node moves next to the average height of its partners.

    Plain barycentre ordering, run a fixed number of times with the node id as the
    tie-break. It is not an optimal crossing minimisation and does not try to be —
    it is deterministic, costs nothing at these sizes, and removes the crossings a
    reader actually notices.
    """
    last = max(1, len(order) - 1)
    current = {node_id: index / last for index, node_id in enumerate(order)}
    def key(node_id: str) -> tuple[float, str]:
        partners = [ranks[other] for other in neighbours.get(node_id, ()) if other in ranks]
        return (sum(partners) / len(partners) if partners else current[node_id], node_id)
    return sorted(order, key=key)


def _order_bands(
    left: Sequence[str], dcs: Sequence[str], right: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    successors: dict[str, set[str]] = {}
    predecessors: dict[str, set[str]] = {}
    for source, target in edges:
        successors.setdefault(source, set()).add(target)
        predecessors.setdefault(target, set()).add(source)
    both = {
        node_id: successors.get(node_id, set()) | predecessors.get(node_id, set())
        for node_id in dcs
    }
    left, dcs, right = list(left), list(dcs), list(right)
    for _ in range(_SWEEPS):
        right = _sweep(right, predecessors, _rank_map(left, dcs))
        dcs = _sweep(dcs, both, _rank_map(left, right))
        left = _sweep(left, successors, _rank_map(dcs, right))
    return left, dcs, right


def _place(
    order: Sequence[str], centre_x: float, lanes: int, lane_width: float,
    box_h: float, step: float, height: float,
) -> dict[str, tuple[float, float]]:
    """Lay one band out on a lane/row grid, centred in the canvas.

    Rows are filled across the lanes so that two nodes sharing a row also share a
    barycentre neighbourhood, and the second lane is offset by half a row so an
    edge leaving the outer lane passes *between* two inner boxes instead of over
    one.
    """
    positions: dict[str, tuple[float, float]] = {}
    if not order:
        return positions
    rows = math.ceil(len(order) / lanes)
    top = (height - _span(rows, step, lanes)) / 2
    for index, node_id in enumerate(order):
        row, lane = index // lanes, index % lanes
        # Lanes keep their own column even on a short final row: re-centring the
        # last node would move it half a lane sideways, straight under the
        # half-row-staggered box above it.
        positions[node_id] = (
            round(centre_x + (lane - (lanes - 1) / 2) * lane_width, 2),
            round(top + row * step + (step / 2 if lane % 2 else 0.0), 2),
        )
    return positions


def compute_flow_layout(
    store_ids: Sequence[object],
    dc_ids: Sequence[object],
    edges: Sequence[tuple[str, str]],
    *,
    keep: Iterable[object] = (),
    width: float = CANVAS_WIDTH,
    min_height: float = MIN_HEIGHT,
    max_height: float = MAX_HEIGHT,
) -> FlowLayout:
    """Place stores and 물류센터 on the 출발 → 경유 → 도착 grid.

    ``edges`` are the drawn moves already resolved into node pairs (one pair for a
    직접 이동, two for a DC 경유), and ``keep`` names the nodes that must survive
    when the picture cannot hold everything — in practice every node the current
    plan touches.
    """
    stores = _clean(store_ids)
    dcs = _clean(dc_ids)
    protected = set(_clean(keep))
    pairs = [
        (str(a or "").strip(), str(b or "").strip()) for a, b in edges
        if str(a or "").strip() and str(b or "").strip()
    ]

    store_w, store_h = store_dimensions(len(stores) + len(dcs))
    dc_w, dc_h = dc_dimensions((store_w, store_h))
    step = _row_step(store_h)
    dc_step = dc_h + DC_ROW_GAP

    left, right = _sides(stores, pairs)

    # One column per band for as long as the default canvas can hold it, then a
    # second lane, and only when two lanes at the tallest canvas still cannot hold
    # the band does the picture leave context stores out. Reaching for the second
    # lane before the extra height keeps a mid-sized plan on the screen the
    # workspace already has; the height is spent on the sizes that truly need it.
    tallest = max(len(left), len(right))
    lanes = MAX_LANES
    for limit in (min_height, max_height):
        found = next(
            (
                candidate for candidate in range(1, MAX_LANES + 1)
                if tallest <= candidate * _rows_that_fit(limit, store_h, step, candidate)
            ),
            None,
        )
        if found is not None:
            lanes = found
            break

    capacity = lanes * _rows_that_fit(max_height, store_h, step, lanes)
    dropped: list[str] = []
    for band in (left, right):
        if len(band) <= capacity:
            continue
        band.sort(key=lambda node_id: (node_id not in protected, node_id))
        dropped.extend(band[capacity:])
        del band[capacity:]
        band.sort()

    dc_lanes = 1 if len(dcs) <= _rows_that_fit(max_height, dc_h, dc_step, 1) else 2

    usable = width - 2 * SIDE_PAD
    dc_lane_width = (dc_w + LANE_GAP) if dc_lanes > 1 else dc_w
    dc_span = (dc_lane_width * dc_lanes + 2 * BAND_GAP) if dcs else BAND_GAP
    side_band = (usable - dc_span) / 2
    lane_width = side_band / lanes
    store_w = round(min(store_w, lane_width - LANE_GAP), 2)
    # A single-column band would otherwise sit in the middle of its own half and
    # leave only a narrow corridor for the lines and the quantity chips. Capping
    # the lane pitch pushes the columns out towards the edges of the canvas, which
    # is where the flow reads from anyway, and gives the middle back to the moves.
    lane_width = min(lane_width, store_w * 1.42)
    column = lane_width * lanes

    left_x = SIDE_PAD + column / 2
    right_x = width - SIDE_PAD - column / 2
    centre_x = width / 2

    rows = max(
        math.ceil(len(left) / lanes) if left else 0,
        math.ceil(len(right) / lanes) if right else 0,
    )
    needed = _span(rows, step, lanes) + store_h + 2 * TOP_PAD if rows else 0.0
    if dcs:
        dc_rows = math.ceil(len(dcs) / dc_lanes)
        needed = max(needed, _span(dc_rows, dc_step, dc_lanes) + dc_h + 2 * TOP_PAD)
    # A picture smaller than the default canvas is centred in it, and everything
    # left over becomes empty canvas above and below the plan. Below
    # SMALL_NETWORK_NODES the canvas follows the content instead, so a two- or
    # four-node move fills its card; at every larger size the floor is unchanged,
    # which is what keeps the dense layouts exactly where they were.
    if not needed:
        height = min_height
    elif len(left) + len(right) + len(dcs) <= SMALL_NETWORK_NODES:
        height = min(max_height, needed + 2 * SMALL_NETWORK_PAD)
    else:
        height = min(max_height, max(min_height, needed))
    height = round(height, 2)

    ordered_left, ordered_dcs, ordered_right = _order_bands(left, dcs, right, pairs)
    positions: dict[str, tuple[float, float]] = {}
    positions.update(_place(ordered_left, left_x, lanes, lane_width, store_h, step, height))
    positions.update(_place(ordered_right, right_x, lanes, lane_width, store_h, step, height))
    positions.update(
        _place(ordered_dcs, centre_x, dc_lanes, dc_lane_width or 1.0, dc_h, dc_step, height)
    )

    bands = {node_id: SOURCE_BAND for node_id in ordered_left}
    bands.update({node_id: TARGET_BAND for node_id in ordered_right})
    bands.update({node_id: DC_BAND for node_id in ordered_dcs})

    return FlowLayout(
        positions=positions,
        bands=bands,
        width=width,
        height=height,
        store_size=(store_w, store_h),
        dc_size=(dc_w, dc_h),
        lanes={SOURCE_BAND: lanes, TARGET_BAND: lanes, DC_BAND: dc_lanes},
        lane_width={
            SOURCE_BAND: round(lane_width, 2),
            TARGET_BAND: round(lane_width, 2),
            DC_BAND: round(dc_lane_width, 2),
        },
        inner_edge={
            SOURCE_BAND: round(SIDE_PAD + column, 2),
            TARGET_BAND: round(width - SIDE_PAD - column, 2),
            DC_BAND: round(width / 2, 2),
        },
        dropped=tuple(sorted(dropped)),
    )


def clip_to_box(
    start: tuple[float, float], end: tuple[float, float],
    half_width: float, half_height: float, gap: float = 4.0,
) -> tuple[float, float]:
    """Where the line from ``start`` to ``end`` leaves ``start``'s own box.

    Edges are drawn between box edges rather than between box centres, so an
    arrowhead lands where it can be seen instead of under the node it points at.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return start
    scale = min(
        half_width / abs(dx) if abs(dx) > 1e-6 else math.inf,
        half_height / abs(dy) if abs(dy) > 1e-6 else math.inf,
    )
    length = math.hypot(dx, dy)
    scale = min(1.0, scale + gap / length)
    return round(start[0] + dx * scale, 2), round(start[1] + dy * scale, 2)


__all__ = [
    "CANVAS_WIDTH",
    "DC_BAND",
    "FlowLayout",
    "MAX_HEIGHT",
    "MAX_LANES",
    "MIN_HEIGHT",
    "SMALL_NETWORK_NODES",
    "SMALL_NETWORK_PAD",
    "SOURCE_BAND",
    "TARGET_BAND",
    "clip_to_box",
    "compute_flow_layout",
    "dc_dimensions",
    "store_dimensions",
]
