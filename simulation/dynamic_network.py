"""Data-driven network nodes, deterministic layout, and route segments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

DC = "DC"
STORE = "STORE"
SIMULATION_LAYOUT_VERSION = "dynamic-layout-2026-07-20.2"

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


def _dc_positions(count: int, width: float, height: float) -> list[tuple[float, float]]:
    cx, cy = width * 0.55, height * 0.45
    if count == 1:
        return [(cx, cy)]
    if count == 2:
        return [(width * 0.40, cy), (width * 0.64, cy)]
    if count == 3:
        return [(width * 0.40, height * 0.43), (width * 0.60, height * 0.43), (cx, height * 0.56)]
    return [
        (cx + width * 0.14 * math.cos(-math.pi / 2 + 2 * math.pi * i / count),
         cy + height * 0.11 * math.sin(-math.pi / 2 + 2 * math.pi * i / count))
        for i in range(count)
    ]


def _coordinate_mode(nodes: Sequence[Mapping[str, object]]) -> bool:
    valid = [
        (_number(row.get("latitude")), _number(row.get("longitude")))
        for row in nodes
    ]
    valid = [(lat, lon) for lat, lon in valid if lat is not None and lon is not None]
    if len(valid) < max(2, math.ceil(len(nodes) * 0.6)):
        return False
    lats, lons = [item[0] for item in valid], [item[1] for item in valid]
    lat_span_km = (max(lats) - min(lats)) * 111.0
    lon_span_km = (max(lons) - min(lons)) * 88.0
    return max(lat_span_km, lon_span_km) >= 1.2


def _geo_positions(
    nodes: Sequence[Mapping[str, object]], width: float, height: float, margin: float,
) -> dict[str, tuple[float, float]]:
    valid = [
        (_number(row.get("latitude")), _number(row.get("longitude")))
        for row in nodes
        if _number(row.get("latitude")) is not None and _number(row.get("longitude")) is not None
    ]
    lat_min, lat_max = min(item[0] for item in valid), max(item[0] for item in valid)
    lon_min, lon_max = min(item[1] for item in valid), max(item[1] for item in valid)
    positions: dict[str, tuple[float, float]] = {}
    safe_x = max(margin, 104.0)
    safe_y = max(margin, 64.0)
    fallback = 0
    for row in _sort(nodes):
        node_id = _value(row, ("node_id",))
        lat, lon = _number(row.get("latitude")), _number(row.get("longitude"))
        if lat is None or lon is None:
            angle = -math.pi / 2 + 2 * math.pi * fallback / max(1, len(nodes))
            positions[node_id] = (width * 0.5 + width * 0.40 * math.cos(angle), height * 0.5 + height * 0.38 * math.sin(angle))
            fallback += 1
            continue
        lon_ratio = (lon - lon_min) / max(lon_max - lon_min, 1e-9)
        lat_ratio = (lat - lat_min) / max(lat_max - lat_min, 1e-9)
        if classify_node(row) == DC:
            positions[node_id] = (width * 0.36 + lon_ratio * width * 0.28, height * 0.37 + (1 - lat_ratio) * height * 0.22)
        else:
            positions[node_id] = (
                safe_x + lon_ratio * (width - safe_x * 2),
                safe_y + (1 - lat_ratio) * (height - safe_y * 2),
            )
    return positions


def _separate(
    nodes: Sequence[Mapping[str, object]], positions: Mapping[str, tuple[float, float]],
    width: float, height: float, margin: float, recommended: set[str],
) -> dict[str, tuple[float, float]]:
    minimum = 170.0 if len(nodes) <= 12 else 145.0 if len(nodes) <= 16 else 78.0 if len(nodes) <= 30 else 48.0
    result: dict[str, tuple[float, float]] = {}
    store_count = sum(classify_node(row) == STORE for row in nodes)
    store_width, store_height = _dimensions(store_count)
    detailed_width = max(194.0, store_width + 24.0) if store_count <= 16 else store_width + 12.0
    detailed_height = max(132.0, store_height + 58.0) if store_count <= 16 else store_height + 24.0
    dc_width, dc_height = max(202.0, store_width * 1.35), max(112.0, store_height * 1.45)
    max_width = max(detailed_width, dc_width)
    max_height = max(detailed_height, dc_height)
    safe_x = max(margin, max_width / 2 + 8.0)
    safe_y = max(margin, max_height / 2 + 8.0)
    placed_sizes: dict[str, tuple[float, float]] = {}

    def card_size(row: Mapping[str, object]) -> tuple[float, float]:
        if classify_node(row) == DC:
            return dc_width, dc_height
        node_id = _value(row, ("node_id",))
        return (detailed_width, detailed_height) if node_id in recommended else (store_width, store_height)

    def clear(candidate_x: float, candidate_y: float, candidate_width: float, candidate_height: float) -> bool:
        for placed_id, (placed_x, placed_y) in result.items():
            placed_width, placed_height = placed_sizes[placed_id]
            horizontal_gap = (candidate_width + placed_width) / 2 + 10.0
            vertical_gap = (candidate_height + placed_height) / 2 + 10.0
            if abs(candidate_x - placed_x) < horizontal_gap and abs(candidate_y - placed_y) < vertical_gap:
                return False
        return True

    ordered_nodes = _sort(row for row in nodes if classify_node(row) == DC) + _sort(
        row for row in nodes if classify_node(row) != DC
    )
    for index, row in enumerate(ordered_nodes):
        node_id = _value(row, ("node_id",))
        base_x, base_y = positions[node_id]
        node_width, node_height = card_size(row)
        x, y = base_x, base_y
        found = False
        for attempt in range(48):
            if (
                all(math.hypot(x - px, y - py) >= minimum for px, py in result.values())
                and clear(x, y, node_width, node_height)
            ):
                found = True
                break
            angle = (index * 2.399963 + attempt * 0.83) % (2 * math.pi)
            radius = minimum * (0.35 + attempt * 0.18)
            x = max(safe_x, min(base_x + radius * math.cos(angle), width - safe_x))
            y = max(safe_y, min(base_y + radius * math.sin(angle), height - safe_y))
        if not found:
            columns = 5
            rows = max(3, math.ceil(len(nodes) / columns))
            fallback_positions = [
                (
                    safe_x + column * (width - 2 * safe_x) / max(1, columns - 1),
                    safe_y + grid_row * (height - 2 * safe_y) / max(1, rows - 1),
                )
                for grid_row in range(rows)
                for column in range(columns)
            ]
            valid_positions = [
                point for point in fallback_positions
                if clear(point[0], point[1], node_width, node_height)
            ]
            if valid_positions:
                x, y = min(valid_positions, key=lambda point: (math.hypot(point[0] - base_x, point[1] - base_y), point[1], point[0]))
        result[node_id] = (round(x, 2), round(y, 2))
        placed_sizes[node_id] = (node_width, node_height)
    return result


def _ring_specs(count: int, width: float, height: float) -> list[tuple[float, float]]:
    if count <= 16:
        return [(width * 0.42, height * 0.37)]
    if count <= 30:
        return [(width * 0.43, height * 0.38), (width * 0.31, height * 0.27)]
    return [(width * 0.44, height * 0.39), (width * 0.34, height * 0.30), (width * 0.25, height * 0.21)]


def _ring_positions(
    stores: Sequence[Mapping[str, object]], recommended: set[str], width: float, height: float,
) -> dict[str, tuple[float, float, float]]:
    """Spread stores around a soft rectangular perimeter instead of a rigid ring."""
    specs = _ring_specs(len(stores), width, height)
    members: list[list[Mapping[str, object]]] = [[] for _ in specs]
    for index, row in enumerate(stores):
        members[index % len(specs)].append(row)
    result: dict[str, tuple[float, float, float]] = {}
    for ring_index, group in enumerate(members):
        radius_x, radius_y = specs[ring_index]
        for index, row in enumerate(group):
            angle = -math.pi / 2 + 2 * math.pi * index / max(1, len(group))
            node_id = _value(row, ("node_id",))
            # A superellipse keeps the central operating area open while making
            # four-to-ten-store layouts feel less radial and more evenly spread.
            exponent = 0.48
            cos_value, sin_value = math.cos(angle), math.sin(angle)
            x_shape = math.copysign(abs(cos_value) ** exponent, cos_value)
            y_shape = math.copysign(abs(sin_value) ** exponent, sin_value)
            scale = 0.95 if node_id in recommended else 1.0
            result[node_id] = (
                round(width * 0.5 + radius_x * scale * x_shape, 2),
                round(height * 0.51 + radius_y * scale * y_shape, 2),
                angle,
            )
    return result


def _small_store_positions(
    stores: Sequence[Mapping[str, object]], width: float, height: float,
) -> dict[str, tuple[float, float, float]]:
    """Keep one-to-three-store samples spacious and readable around the DC."""
    count = len(stores)
    if count == 1:
        points = [(width * 0.20, height * 0.60)]
    elif count == 2:
        # Source/target order is preserved: a short horizontal flow stays below
        # the central DC instead of stretching along the bottom canvas edge.
        points = [(width * 0.18, height * 0.61), (width * 0.86, height * 0.61)]
    else:
        points = [
            (width * 0.17, height * 0.38),
            (width * 0.18, height * 0.72),
            (width * 0.86, height * 0.58),
        ]
    result: dict[str, tuple[float, float, float]] = {}
    for row, (x, y) in zip(stores, points):
        angle = math.atan2(y - height * 0.5, x - width * 0.5)
        result[_value(row, ("node_id",))] = (round(x, 2), round(y, 2), angle)
    return result


def _route_ordered_stores(
    stores: Sequence[Mapping[str, object]], recommendations: Sequence[Mapping[str, object]], limit: int = 5,
) -> list[Mapping[str, object]]:
    """Place Top-route endpoints next to each other before filling remaining slots."""
    by_id = {_value(row, ("node_id",)): row for row in stores}
    ordered: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for route in recommendations[:limit]:
        for key in ("source_id", "target_id"):
            node_id = str(route.get(key) or "").strip()
            if node_id in by_id and node_id not in seen:
                ordered.append(by_id[node_id])
                seen.add(node_id)
    ordered.extend(row for row in _sort(stores) if _value(row, ("node_id",)) not in seen)
    return ordered


def _node_position(
    row: Mapping[str, object], x: float, y: float, width: float, height: float,
    recommended: set[str], show_label: bool, angle: float | None = None,
) -> dict[str, object]:
    node_id = _value(row, ("node_id",))
    return {
        **dict(row),
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
    }


def compute_dynamic_layout(
    nodes: Iterable[Mapping[str, object]],
    recommendations: Iterable[Mapping[str, object]] = (),
    width: float = 1200.0,
    height: float = 680.0,
    margin: float = 78.0,
) -> DynamicLayout:
    """Compute deterministic geographic or ring layout for any DC/store count."""
    recommendation_rows = [dict(row) for row in recommendations]
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

    recommended = _recommended_ids(recommendation_rows, limit=5)
    store_width, store_height = _dimensions(len(stores))
    detailed_width = max(194.0, store_width + 24.0) if len(stores) <= 16 else store_width + 12.0
    detailed_height = max(132.0, store_height + 58.0) if len(stores) <= 16 else store_height + 24.0
    dc_width, dc_height = max(202.0, store_width * 1.35), max(112.0, store_height * 1.45)
    geographic = _coordinate_mode(deduped)
    dc_rows: list[dict[str, object]] = []
    store_rows: list[dict[str, object]] = []

    if geographic:
        base_positions = _geo_positions(deduped, width, height, margin)
        # Geographic coordinates inform store ordering, while DC cards stay in
        # a stable central control zone.  This prevents a wide coordinate range
        # from pushing a hub toward the edge of the canvas.
        for row, position in zip(dcs, _dc_positions(len(dcs), width, height)):
            base_positions[_value(row, ("node_id",))] = position
        ordered_stores = _route_ordered_stores(stores, recommendation_rows)
        if len(ordered_stores) <= 3:
            for node_id, (x, y, _) in _small_store_positions(ordered_stores, width, height).items():
                base_positions[node_id] = (x, y)
        positions = _separate(deduped, base_positions, width, height, margin, recommended)
        for row in dcs:
            x, y = positions[_value(row, ("node_id",))]
            dc_rows.append(_node_position(row, x, y, dc_width, dc_height, recommended, True))
        for index, row in enumerate(stores):
            node_id = _value(row, ("node_id",))
            x, y = positions[node_id]
            emphasis = node_id in recommended
            store_rows.append(_node_position(
                row, x, y, detailed_width if emphasis else store_width, detailed_height if emphasis else store_height,
                recommended, len(stores) <= 30 or emphasis or index % 3 == 0,
            ))
    else:
        for row, (x, y) in zip(dcs, _dc_positions(len(dcs), width, height)):
            dc_rows.append(_node_position(row, x, y, dc_width, dc_height, recommended, True))
        ordered_stores = _route_ordered_stores(stores, recommendation_rows)
        positions = (
            _small_store_positions(ordered_stores, width, height)
            if len(ordered_stores) <= 3
            else _ring_positions(ordered_stores, recommended, width, height)
        )
        for index, row in enumerate(ordered_stores):
            node_id = _value(row, ("node_id",))
            x, y, angle = positions[node_id]
            emphasis = node_id in recommended
            store_rows.append(_node_position(
                row, x, y, detailed_width if emphasis else store_width, detailed_height if emphasis else store_height,
                recommended, len(stores) <= 30 or emphasis or index % 3 == 0, angle,
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
            "layout_mode": "geographic" if geographic else "deterministic",
            "dc_count": len(dc_rows),
            "store_count": len(store_rows),
            "ring_count": len(_ring_specs(len(stores), width, height)) if stores else 0,
            "layout_version": SIMULATION_LAYOUT_VERSION,
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
