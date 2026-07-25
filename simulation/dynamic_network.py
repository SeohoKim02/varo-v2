"""Data-driven network nodes, deterministic layout, and route segments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

DC = "DC"
STORE = "STORE"

ID_KEYS = ("node_id", "store_id", "dc_id", "id")
NAME_KEYS = ("node_name", "store_name", "dc_name", "name")
TYPE_KEYS = ("node_type", "store_type", "type")
REGION_KEYS = ("region", "area", "district", "zone")
LAT_KEYS = ("latitude", "lat")
LON_KEYS = ("longitude", "lon", "lng")


@dataclass(frozen=True)
class DynamicLayout:
    is_valid: bool
    dcs: list[dict[str, object]] = field(default_factory=list)
    stores: list[dict[str, object]] = field(default_factory=list)
    canvas: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def dc(self) -> dict[str, object] | None:
        return self.dcs[0] if self.dcs else None


def _blank(value: object) -> bool:
    return value is None or str(value).strip().lower() in {"", "nan", "none", "<na>", "nat"}


def _value(row: Mapping[str, object], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if not _blank(value):
            return str(value).strip()
    return default


def _number(value: object) -> float | None:
    if _blank(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classify_node(row: Mapping[str, object]) -> str:
    explicit = _value(row, TYPE_KEYS).upper().replace("-", "_").replace(" ", "_")
    if explicit in {"DC", "DISTRIBUTION_CENTER", "WAREHOUSE", "HUB", "물류센터", "센터"}:
        return DC
    if explicit in {"STORE", "SHOP", "RETAIL", "RETAILER", "점포", "매장"}:
        return STORE
    node_id = _value(row, ID_KEYS).upper()
    name = _value(row, NAME_KEYS).upper()
    if node_id.startswith("DC"):
        return DC
    if any(token in name for token in ("물류센터", "물류 센터", "DISTRIBUTION CENTER")):
        return DC
    if name == "DC" or name.startswith("DC ") or name.endswith(" DC") or "센터" in name:
        return DC
    return STORE


def _records(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            return [dict(row) for row in value.to_dict("records")]
        except TypeError:
            pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _normalize(row: Mapping[str, object], index: int) -> dict[str, object]:
    name = _value(row, NAME_KEYS)
    node_id = _value(row, ID_KEYS, name or f"NODE{index + 1:03d}")
    return {
        **dict(row),
        "node_id": node_id,
        "node_name": name or node_id,
        "node_type": classify_node(row),
        "region": _value(row, REGION_KEYS),
        "latitude": _number(next((row.get(key) for key in LAT_KEYS if not _blank(row.get(key))), None)),
        "longitude": _number(next((row.get(key) for key in LON_KEYS if not _blank(row.get(key))), None)),
    }


def build_network_nodes(
    data: Mapping[str, Any] | None,
    recommendations: Iterable[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Normalize store rows and add any route-referenced nodes that are absent."""
    data = data or {}
    source = data.get("stores") if data.get("stores") is not None else data.get("nodes")
    source_rows = _records(source)
    if not source_rows:
        inventory_rows = _records(data.get("inventory"))
        seen_inventory_nodes: set[str] = set()
        for row in inventory_rows:
            node_id = _value(row, ("store_id", "node_id"))
            if node_id and node_id not in seen_inventory_nodes:
                seen_inventory_nodes.add(node_id)
                source_rows.append({
                    "node_id": node_id,
                    "node_name": _value(row, ("store_name", "node_name"), node_id),
                    "node_type": STORE,
                    "region": _value(row, REGION_KEYS),
                })
    nodes = [_normalize(row, index) for index, row in enumerate(source_rows)]
    by_id = {str(node["node_id"]): node for node in nodes}

    def ensure(node_id: object, name: object, node_type: str) -> None:
        if _blank(node_id) and _blank(name):
            return
        resolved = str(node_id).strip() if not _blank(node_id) else str(name).strip()
        if resolved in by_id:
            if node_type == DC:
                by_id[resolved]["node_type"] = DC
            return
        node = _normalize({"node_id": resolved, "node_name": name or resolved, "node_type": node_type}, len(nodes))
        nodes.append(node)
        by_id[resolved] = node

    for route in recommendations:
        ensure(route.get("source_id"), route.get("source_name"), STORE)
        ensure(route.get("target_id"), route.get("target_name"), STORE)
        if not _blank(route.get("dc_id")) or not _blank(route.get("dc_name")):
            ensure(route.get("dc_id"), route.get("dc_name"), DC)
    return nodes


def _sort(nodes: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(
        nodes,
        key=lambda row: (
            _value(row, ("region",)).casefold(),
            _value(row, ("node_id",)).casefold(),
            _value(row, ("node_name",)).casefold(),
        ),
    )


def _recommended_ids(recommendations: Iterable[Mapping[str, object]], limit: int = 3) -> set[str]:
    ids: set[str] = set()
    for route in list(recommendations)[:limit]:
        for key in ("source_id", "target_id", "dc_id"):
            if not _blank(route.get(key)):
                ids.add(str(route[key]))
    return ids


def _dimensions(count: int) -> tuple[float, float]:
    if count <= 8:
        return 164.0, 68.0
    if count <= 16:
        return 142.0, 60.0
    if count <= 30:
        return 116.0, 52.0
    return 92.0, 44.0


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _dc_anchor_positions(count: int, width: float, height: float, margin: float) -> list[tuple[float, float]]:
    """DC anchor points: 1 centered, 2 split left/right, 3+ on a small central ring."""
    cx, cy = width * 0.5, height * 0.5
    inner_w, inner_h = width - 2 * margin, height - 2 * margin
    if count <= 1:
        return [(cx, cy)]
    if count == 2:
        return [(margin + inner_w * 0.27, cy), (margin + inner_w * 0.73, cy)]
    return [
        (cx + inner_w * 0.22 * math.cos(-math.pi / 2 + 2 * math.pi * i / count),
         cy + inner_h * 0.22 * math.sin(-math.pi / 2 + 2 * math.pi * i / count))
        for i in range(count)
    ]


def _split_evenly(items: Sequence[Any], groups: int) -> list[list[Any]]:
    """Deterministically split items into `groups` contiguous near-equal chunks."""
    items = list(items)
    if groups <= 1 or not items:
        return [items] + [[] for _ in range(max(0, groups - 1))]
    per = math.ceil(len(items) / groups)
    return [items[i * per:(i + 1) * per] for i in range(groups)]


def _outward_angle(dcx: float, dcy: float, cx: float, cy: float) -> float | None:
    """Direction from the canvas center out to a DC (arc faces this way). None if centered."""
    dx, dy = dcx - cx, dcy - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


def compute_network_layout(
    stores: Sequence[Mapping[str, object]],
    dcs: Sequence[Mapping[str, object]],
    width: float,
    height: float,
    margin: float,
    recommended: set[str] | None = None,
    store_size: tuple[float, float] = (164.0, 68.0),
) -> dict[str, tuple[float, float, float | None]]:
    """DC-centric radial placement returning ``{node_id: (x, y, angle|None)}``.

    Deterministic (depends only on node counts and sorted order, never on
    latitude/longitude). One DC sits at the center with stores on an ellipse
    around it; two DCs sit left/right with each DC's stores fanned out on the
    outward side; three or more DCs form a small central ring. Every coordinate
    is clamped inside the padding so no node is pushed off the canvas.
    """
    recommended = recommended or set()
    cx, cy = width * 0.5, height * 0.5
    store_w, store_h = store_size
    safe_x = margin + store_w / 2.0
    safe_y = margin + store_h / 2.0 + 6.0
    dc_list = _sort(dcs)
    anchors = _dc_anchor_positions(len(dc_list), width, height, margin)
    positions: dict[str, tuple[float, float, float | None]] = {}
    for row, (ax, ay) in zip(dc_list, anchors):
        positions[_value(row, ("node_id",))] = (round(ax, 2), round(ay, 2), None)

    store_list = _sort(stores)
    groups = _split_evenly(store_list, max(1, len(dc_list)))
    for group_index, group in enumerate(groups):
        if not group:
            continue
        dcx, dcy = anchors[group_index]
        outward = _outward_angle(dcx, dcy, cx, cy)
        if outward is None:  # single central DC → full ellipse
            radius_x = min(cx - safe_x, (width - 2 * margin) * 0.40)
            radius_y = min(cy - safe_y, (height - 2 * margin) * 0.42)
            sweep, base_angle, closed = 2 * math.pi, 0.0, True
        else:  # fan the group out on the DC's outward side
            radius_x = min((width - 2 * margin) * 0.24, abs(dcx - cx) + (width - 2 * margin) * 0.10)
            radius_y = min(cy - safe_y, (height - 2 * margin) * 0.40)
            sweep = math.pi * (1.15 if len(group) > 3 else 0.72)
            base_angle, closed = outward, False
        count = len(group)
        for index, row in enumerate(group):
            if closed:
                angle = base_angle + 2 * math.pi * index / count
            elif count == 1:
                angle = base_angle
            else:
                angle = base_angle - sweep / 2 + sweep * index / (count - 1)
            node_id = _value(row, ("node_id",))
            scale = 0.93 if node_id in recommended else 1.0
            x = _clamp(dcx + radius_x * scale * math.cos(angle), safe_x, width - safe_x)
            y = _clamp(dcy + radius_y * scale * math.sin(angle), safe_y, height - safe_y)
            positions[node_id] = (round(x, 2), round(y, 2), round(angle, 6))
    return positions


def _node_position(
    row: Mapping[str, object], x: float, y: float, width: float, height: float,
    recommended: set[str], show_label: bool, angle: float | None = None,
) -> dict[str, object]:
    node_id = _value(row, ("node_id",))
    return {
        "node_id": node_id,
        "node_name": _value(row, ("node_name",), node_id),
        "node_type": classify_node(row),
        "region": _value(row, ("region",)),
        "x": round(x, 2),
        "y": round(y, 2),
        "width": width,
        "height": height,
        "angle": round(angle, 6) if angle is not None else None,
        "is_recommended": node_id in recommended,
        "show_label": show_label,
        "inventory_state": _value(row, ("inventory_state",)),
    }


def compute_dynamic_layout(
    nodes: Iterable[Mapping[str, object]],
    recommendations: Iterable[Mapping[str, object]] = (),
    width: float = 1200.0,
    height: float = 680.0,
    margin: float = 92.0,
) -> DynamicLayout:
    """Deterministic DC-centric radial layout for any DC/store count."""
    normalized = [_normalize(row, index) for index, row in enumerate(nodes)]
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in normalized:
        if str(row["node_id"]) not in seen:
            seen.add(str(row["node_id"]))
            deduped.append(row)
    dcs = _sort(row for row in deduped if classify_node(row) == DC)
    stores = _sort(row for row in deduped if classify_node(row) == STORE)
    if not dcs:
        return DynamicLayout(False, canvas={"width": width, "height": height}, errors=["네트워크 시뮬레이션에 표시할 DC가 없습니다."])

    recommended = _recommended_ids(recommendations)
    store_width, store_height = _dimensions(len(stores))
    dc_width, dc_height = max(190.0, store_width * 1.32), max(86.0, store_height * 1.38)
    positions = compute_network_layout(
        stores, dcs, width, height, margin, recommended, store_size=(store_width, store_height),
    )

    dc_rows: list[dict[str, object]] = []
    for row in dcs:
        x, y, _ = positions[_value(row, ("node_id",))]
        dc_rows.append(_node_position(row, x, y, dc_width, dc_height, recommended, True))

    store_rows: list[dict[str, object]] = []
    for index, row in enumerate(stores):
        node_id = _value(row, ("node_id",))
        x, y, angle = positions[node_id]
        emphasis = node_id in recommended
        store_rows.append(_node_position(
            row, x, y, store_width + (10 if emphasis else 0), store_height + (6 if emphasis else 0),
            recommended, len(stores) <= 24 or emphasis or index % 2 == 0, angle,
        ))

    warnings = ["점포 수가 많아 추천 관련 라벨을 우선 표시합니다."] if len(stores) >= 31 else []
    return DynamicLayout(
        True,
        dcs=dc_rows,
        stores=store_rows,
        warnings=warnings,
        canvas={
            "width": width,
            "height": height,
            "margin": margin,
            "layout_mode": "radial",
            "dc_count": len(dc_rows),
            "store_count": len(store_rows),
            "ring_count": 1,
        },
    )


def normalize_route_type(route: Mapping[str, object]) -> str:
    raw = str(route.get("route_type") or "").strip().upper().replace("-", "_").replace(" ", "_")
    if raw in {"DIRECT", "DIRECT_TRANSFER", "직접", "직접_이동"}:
        return "DIRECT"
    if raw in {"VIA_DC", "DC_TRANSFER", "DC_경유", "경유"}:
        return "VIA_DC"
    if not raw:
        return "VIA_DC" if not _blank(route.get("dc_id")) or not _blank(route.get("dc_name")) else "DIRECT"
    return raw


def _deterministic_fallback_dc(route: Mapping[str, object], nodes: Sequence[Mapping[str, object]]) -> str | None:
    """Choose an available DC deterministically when VIA_DC omits dc_id/name."""
    dcs = _sort(row for row in nodes if classify_node(row) == DC)
    if not dcs:
        return None
    source_id = _value(route, ("source_id", "from_id", "source_store_id"))
    source = next((_normalize(row, idx) for idx, row in enumerate(nodes) if _value(row, ID_KEYS) == source_id), None)
    if source:
        source_lat, source_lon = _number(source.get("latitude")), _number(source.get("longitude"))
        if source_lat is not None and source_lon is not None:
            candidates: list[tuple[float, str]] = []
            for idx, dc in enumerate(dcs):
                normalized_dc = _normalize(dc, idx)
                dc_lat, dc_lon = _number(normalized_dc.get("latitude")), _number(normalized_dc.get("longitude"))
                if dc_lat is None or dc_lon is None:
                    continue
                distance = math.hypot((source_lat - dc_lat) * 111.0, (source_lon - dc_lon) * 88.0)
                candidates.append((distance, _value(normalized_dc, ID_KEYS)))
            if candidates:
                return min(candidates, key=lambda item: (item[0], item[1]))[1]

        source_region = _value(source, REGION_KEYS)
        if source_region:
            same_region = [
                _value(_normalize(dc, idx), ID_KEYS)
                for idx, dc in enumerate(dcs)
                if _value(_normalize(dc, idx), REGION_KEYS) == source_region
            ]
            if same_region:
                return sorted(same_region)[0]

    return _value(_normalize(dcs[0], 0), ID_KEYS)


def resolve_route_dc_id(route: Mapping[str, object], nodes: Iterable[Mapping[str, object]]) -> str | None:
    node_list = list(nodes)
    dcs = [row for row in node_list if classify_node(row) == DC]
    dc_id = None if _blank(route.get("dc_id")) else str(route.get("dc_id"))
    if dc_id and any(_value(row, ID_KEYS) == dc_id for row in dcs):
        return dc_id
    dc_name = "" if _blank(route.get("dc_name")) else str(route.get("dc_name")).strip()
    for row in dcs:
        if dc_name and _value(row, NAME_KEYS) == dc_name:
            return _value(row, ID_KEYS)
    if len(dcs) == 1:
        return _value(dcs[0], ID_KEYS)
    return dc_id or _deterministic_fallback_dc(route, node_list)


def build_route_segments(
    route: Mapping[str, object], nodes: Iterable[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Return one DIRECT segment or two segments through the row-specific DC."""
    source_id = _value(route, ("source_id", "from_id", "source_store_id"))
    target_id = _value(route, ("target_id", "to_id", "target_store_id"))
    route_type = normalize_route_type(route)
    if not source_id or not target_id:
        raise ValueError("경로의 출발 노드와 도착 노드가 필요합니다.")
    if route_type == "DIRECT":
        return [{"from_node_id": source_id, "to_node_id": target_id, "phase": "DIRECT"}]
    if route_type != "VIA_DC":
        raise ValueError(f"지원하지 않는 route_type입니다: {route.get('route_type')}")
    dc_id = resolve_route_dc_id(route, nodes)
    if not dc_id:
        raise ValueError("DC 경유 경로에 사용할 DC를 확인할 수 없습니다.")
    return [
        {"from_node_id": source_id, "to_node_id": dc_id, "phase": "TO_DC"},
        {"from_node_id": dc_id, "to_node_id": target_id, "phase": "FROM_DC"},
    ]
