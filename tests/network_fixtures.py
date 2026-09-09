"""Deterministic node/move sets for testing the workspace network *picture*.

These fixtures exist for one reason: to put the network drawing under sizes and
shapes the bundled sample workbooks do not reach, so density can be measured
instead of guessed.

They are **not** algorithm output and must never be used as if they were. Nothing
here is produced by VHS, Greedy, DQN, MILP or the execution-plan builder, and
nothing here carries a logistics fact Varo has not collected yet: there is no
distance, no travel time, no vehicle capacity, no transport cost and no observed
movement history in any row. Every field present is one the picture actually
reads — the node, the route type, the DC it passes through, and ``planned_qty``.

The real-data and algorithm fixtures live in :mod:`tests.fixtures`; keep the two
apart.
"""
from __future__ import annotations

from typing import Any

#: Sizes the workspace network is measured at, from a plan a user reads at a
#: glance up to the largest one the canvas can hold.
NETWORK_SIZES = (4, 8, 16, 24, 40, 60)

#: Deterministic store names, varying in length on purpose: short ones that fit
#: any box, ordinary four-character ones, and a few long enough to make the
#: renderer choose between shrinking and shortening.
STORE_NAMES = (
    "강남역점", "서초점", "역삼점", "논현점", "신사점", "청담점", "삼성점", "대치점",
    "잠실점", "송파점", "가락점", "문정점", "수서점", "일원점", "개포점", "도곡점",
    "홍대입구점", "합정점", "상수점", "망원점", "연남점", "성산점", "공덕점", "아현점",
    "종로3가점", "을지로입구점", "시청앞점", "광화문점", "서울역서부점", "남영점",
    "용산아이파크몰점", "이촌점", "노량진점", "대방점", "신길점", "영등포시장점",
    "당산점", "선유도점", "양평점", "목동중앙점", "오목교점", "신정네거리점",
    "화곡본동점", "까치산점", "발산점", "마곡나루점", "김포공항점", "송정점",
    "구로디지털단지점", "신대방삼거리점", "봉천점", "서울대입구점", "낙성대점",
    "사당점", "이수점", "동작점", "흑석점", "상도점", "장승배기점", "노들점",
)

DC_NAMES = ("중앙물류센터", "남부물류센터", "서부물류센터", "동부물류센터")

STATES = ("과잉", "부족", "정상")

#: The move shapes the picture has to stay readable under.
SHAPES = ("direct", "via_dc", "mixed", "fan_out", "fan_in")


def store_id(index: int) -> str:
    return f"S{index:03d}"


def dc_id(index: int) -> str:
    return f"DC{index + 1:02d}"


def store_name(index: int) -> str:
    return STORE_NAMES[index % len(STORE_NAMES)]


def nodes(store_count: int, dc_count: int) -> list[dict[str, Any]]:
    """The store and 물류센터 rows the picture draws, in a fixed order."""
    rows: list[dict[str, Any]] = [
        {"node_id": store_id(index), "node_name": store_name(index), "node_type": "STORE"}
        for index in range(store_count)
    ]
    rows += [
        {"node_id": dc_id(index), "node_name": DC_NAMES[index % len(DC_NAMES)], "node_type": "DC"}
        for index in range(dc_count)
    ]
    return rows


def _pairs(store_count: int, move_count: int, shape: str) -> list[tuple[int, int]]:
    """Deterministic (source, target) store indexes for one move shape."""
    half = max(1, store_count // 2)
    if shape == "fan_out":
        # One store supplies many: 여러 도착, 하나의 출발.
        return [(0, half + index % (store_count - half)) for index in range(move_count)]
    if shape == "fan_in":
        # Many stores supply one: 여러 출발, 하나의 도착.
        return [(index % half, store_count - 1) for index in range(move_count)]
    return [
        (index % half, half + index % max(1, store_count - half)) for index in range(move_count)
    ]


def moves(
    store_count: int, dc_count: int, move_count: int, shape: str = "mixed",
) -> list[dict[str, Any]]:
    """Plan rows in the shape the workspace network reads.

    Only the fields the picture uses are present. ``planned_qty`` is a plain
    counter, not a computed quantity — these rows never stand in for a plan.
    """
    rows: list[dict[str, Any]] = []
    for index, (source, target) in enumerate(_pairs(store_count, move_count, shape)):
        if source == target:
            target = (target + 1) % store_count
        if shape == "direct" or not dc_count:
            via = False
        elif shape == "via_dc":
            via = True
        else:
            via = index % 2 == 0
        rows.append({
            "route_id": f"R{index:03d}",
            "source_id": store_id(source), "source_name": store_name(source),
            "target_id": store_id(target), "target_name": store_name(target),
            "product_id": "P01", "product_name": "상품 A",
            "route_type": "VIA_DC" if via else "DIRECT",
            "dc_id": dc_id(index % dc_count) if via else None,
            "planned_qty": 10 + index * 3,
        })
    return rows


def store_states(store_count: int) -> dict[str, str]:
    """A fixed 과잉/부족/정상 label per store, so the badges are measurable."""
    return {store_id(index): STATES[index % len(STATES)] for index in range(store_count)}


def dense_case(
    node_count: int, *, dc_count: int = 1, shape: str = "mixed", move_count: int | None = None,
) -> dict[str, Any]:
    """One measurable network: ``data``, ``items``, ``states`` and a selection.

    ``node_count`` counts everything drawn — stores *and* 물류센터 — because that
    is what the density of the picture depends on.
    """
    store_count = max(2, node_count - dc_count)
    if move_count is None:
        move_count = max(1, store_count // 2)
    items = moves(store_count, dc_count, move_count, shape)
    return {
        "data": {"stores": nodes(store_count, dc_count)},
        "items": items,
        "states": store_states(store_count),
        "selected_route_id": items[0]["route_id"] if items else "",
        "store_count": store_count,
        "dc_count": dc_count,
        "shape": shape,
    }


__all__ = [
    "DC_NAMES",
    "NETWORK_SIZES",
    "SHAPES",
    "STORE_NAMES",
    "dc_id",
    "dense_case",
    "moves",
    "nodes",
    "store_id",
    "store_name",
    "store_states",
]
