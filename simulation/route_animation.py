"""Route validation and leg construction for Varo V2 simulation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping

DIRECT = "DIRECT"
VIA_DC = "VIA_DC"
SUPPORTED_ROUTE_TYPES = (DIRECT, VIA_DC)


@dataclass(frozen=True)
class RouteLeg:
    route_id: str
    leg_index: int
    from_node_id: str
    to_node_id: str
    phase: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def validate_route(route: Mapping[str, object]) -> None:
    route_id = route.get("route_id")
    source_id = route.get("source_id")
    target_id = route.get("target_id")
    route_type = route.get("route_type")
    if not route_id:
        raise ValueError("route_id가 필요합니다.")
    if not source_id:
        raise ValueError(f"{route_id}: source_id가 필요합니다.")
    if not target_id:
        raise ValueError(f"{route_id}: target_id가 필요합니다.")
    if route_type not in SUPPORTED_ROUTE_TYPES:
        raise ValueError(f"{route_id}: 지원하지 않는 route_type입니다. {route_type}")
    if route_type == VIA_DC and not route.get("dc_id"):
        raise ValueError(f"{route_id}: VIA_DC 경로에는 dc_id가 필요합니다.")


def build_route_legs(route: Mapping[str, object]) -> List[Dict[str, object]]:
    """Build movement legs from the explicit route_type field."""
    validate_route(route)
    route_id = str(route["route_id"])
    if route["route_type"] == DIRECT:
        return [
            RouteLeg(
                route_id=route_id,
                leg_index=0,
                from_node_id=str(route["source_id"]),
                to_node_id=str(route["target_id"]),
                phase="DIRECT",
            ).to_dict()
        ]
    return [
        RouteLeg(
            route_id=route_id,
            leg_index=0,
            from_node_id=str(route["source_id"]),
            to_node_id=str(route["dc_id"]),
            phase="TO_DC",
        ).to_dict(),
        RouteLeg(
            route_id=route_id,
            leg_index=1,
            from_node_id=str(route["dc_id"]),
            to_node_id=str(route["target_id"]),
            phase="FROM_DC",
        ).to_dict(),
    ]