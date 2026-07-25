"""Network node layout helpers for Varo V2."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

DC_TYPE = "DC"
STORE_TYPE = "STORE"


@dataclass(frozen=True)
class NodePosition:
    node_id: str
    node_name: str
    node_type: str
    x: float
    y: float
    width: float
    height: float
    angle: Optional[float] = None
    label_anchor: str = "middle"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LayoutResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dc: Optional[Dict[str, object]] = None
    stores: List[Dict[str, object]] = field(default_factory=list)
    canvas: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _node_value(node: Mapping[str, object], *keys: str, default: str = "") -> str:
    for key in keys:
        value = node.get(key)
        if value is not None and value != "":
            return str(value)
    return default


def _node_type(node: Mapping[str, object]) -> str:
    return _node_value(node, "node_type", "type", "store_type").upper()


def _friendly_name(node: Mapping[str, object], fallback: str) -> str:
    return _node_value(node, "node_name", "store_name", "name", default=fallback)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def validate_network_nodes(nodes: Iterable[Mapping[str, object]]) -> tuple[list[Mapping[str, object]], list[str]]:
    node_list = list(nodes)
    dc_nodes = [node for node in node_list if _node_type(node) == DC_TYPE]
    errors: list[str] = []
    if not dc_nodes:
        errors.append("네트워크 시뮬레이션에는 DC가 정확히 1개 필요합니다. 현재 DC가 없습니다.")
    if len(dc_nodes) > 1:
        errors.append(f"네트워크 시뮬레이션에는 DC가 정확히 1개 필요합니다. 현재 {len(dc_nodes)}개입니다.")
    return node_list, errors


def calculate_network_layout(
    nodes: Iterable[Mapping[str, object]],
    width: float = 1000.0,
    height: float = 650.0,
    node_width: float = 156.0,
    node_height: float = 70.0,
    margin: float = 52.0,
) -> LayoutResult:
    """Place exactly one DC at center and stores around it on an ellipse."""
    if width <= 0 or height <= 0:
        return LayoutResult(False, errors=["시뮬레이션 화면 크기는 0보다 커야 합니다."])
    if node_width <= 0 or node_height <= 0:
        return LayoutResult(False, errors=["노드 카드 크기는 0보다 커야 합니다."])

    node_list, errors = validate_network_nodes(nodes)
    if errors:
        return LayoutResult(False, errors=errors, canvas={"width": width, "height": height, "margin": margin})

    dc_node = next(node for node in node_list if _node_type(node) == DC_TYPE)
    stores = [node for node in node_list if _node_type(node) == STORE_TYPE]
    warnings: list[str] = []
    if not stores:
        warnings.append("표시할 점포가 없습니다.")
    elif len(stores) == 1:
        warnings.append("점포가 1개라 단일 연결 형태로 표시합니다.")

    center_x = width / 2.0
    center_y = height / 2.0
    safe_margin_x = max(margin, node_width / 2.0 + 22.0)
    safe_margin_y = max(margin, node_height / 2.0 + 34.0)
    max_radius_x = max(1.0, center_x - safe_margin_x)
    max_radius_y = max(1.0, center_y - safe_margin_y)
    store_count = len(stores)
    density_x = min(1.0, 0.66 + store_count * 0.045) if store_count else 0.66
    density_y = min(1.0, 0.72 + store_count * 0.032) if store_count else 0.72
    radius_x = max_radius_x * density_x
    radius_y = max_radius_y * density_y

    if store_count >= 12:
        warnings.append("점포 수가 많아 카드 간격이 좁아질 수 있습니다.")

    dc_position = NodePosition(
        node_id=_node_value(dc_node, "node_id", "dc_id", "store_id", "id", default="DC"),
        node_name=_friendly_name(dc_node, "DC"),
        node_type=DC_TYPE,
        x=center_x,
        y=center_y,
        width=node_width * 1.20,
        height=node_height * 1.20,
    )

    store_positions: list[NodePosition] = []
    for index, store in enumerate(stores):
        angle = 2.0 * math.pi * index / store_count if store_count else 0.0
        x = center_x + radius_x * math.cos(angle)
        y = center_y + radius_y * math.sin(angle)
        label_anchor = "start" if math.cos(angle) > 0.35 else "end" if math.cos(angle) < -0.35 else "middle"
        store_positions.append(
            NodePosition(
                node_id=_node_value(store, "node_id", "store_id", "id", default=f"S{index + 1}"),
                node_name=_friendly_name(store, f"점포 {index + 1}"),
                node_type=STORE_TYPE,
                x=round(clamp(x, safe_margin_x, width - safe_margin_x), 2),
                y=round(clamp(y, safe_margin_y, height - safe_margin_y), 2),
                width=node_width,
                height=node_height,
                angle=round(angle, 6),
                label_anchor=label_anchor,
            )
        )

    return LayoutResult(
        True,
        warnings=warnings,
        dc=dc_position.to_dict(),
        stores=[position.to_dict() for position in store_positions],
        canvas={
            "width": width,
            "height": height,
            "margin": margin,
            "radius_x": round(radius_x, 2),
            "radius_y": round(radius_y, 2),
            "node_width": node_width,
            "node_height": node_height,
        },
    )


def calculate_ellipse_layout(
    dc_node: Mapping[str, object],
    store_nodes: Iterable[Mapping[str, object]],
    width: float = 1000.0,
    height: float = 650.0,
    margin: float = 52.0,
) -> Dict[str, object]:
    """Compatibility wrapper around the V2 node layout contract."""
    nodes = [{**dict(dc_node), "node_type": DC_TYPE}]
    nodes.extend({**dict(store), "node_type": STORE_TYPE} for store in store_nodes)
    return calculate_network_layout(nodes, width=width, height=height, margin=margin).to_dict()