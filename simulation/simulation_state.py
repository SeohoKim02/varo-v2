"""Simulation state contracts and transitions for Varo V2."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

READY = "READY"
MOVING = "MOVING"
AT_DC = "AT_DC"
COMPLETED = "COMPLETED"
PAUSED = "PAUSED"
ERROR = "ERROR"
SIMULATION_STATUSES = (READY, MOVING, AT_DC, COMPLETED, PAUSED, ERROR)

# One UI tick is approximately 0.35 seconds. VIA_DC uses twice the per-leg step
# so its two legs complete in about the same total time as one DIRECT route.
SPEED_STEPS = {"느림": 0.014, "보통": 0.022, "빠름": 0.035}
DC_DWELL_TICKS = 3


@dataclass(frozen=True)
class RouteRuntimeState:
    route_id: str
    status: str = READY
    current_leg_index: int = 0
    progress: float = 0.0
    paused_from: Optional[str] = None
    dc_wait_ticks: int = 0
    events: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationLog:
    event_id: str
    route_id: str
    message: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationSnapshot:
    routes: List[Dict[str, object]] = field(default_factory=list)
    route_states: Dict[str, RouteRuntimeState] = field(default_factory=dict)
    logs: List[SimulationLog] = field(default_factory=list)
    is_running: bool = False
    speed_label: str = "보통"
    selected_route_id: Optional[str] = None
    show_all_routes: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "routes": self.routes,
            "route_states": {key: value.to_dict() for key, value in self.route_states.items()},
            "logs": [log.to_dict() for log in self.logs],
            "is_running": self.is_running,
            "speed_label": self.speed_label,
            "selected_route_id": self.selected_route_id,
            "show_all_routes": self.show_all_routes,
            "errors": list(self.errors),
        }


def with_route_state(snapshot: SimulationSnapshot, route_id: str, state: RouteRuntimeState) -> SimulationSnapshot:
    states = dict(snapshot.route_states)
    states[route_id] = state
    return replace(snapshot, route_states=states)


def append_log_once(snapshot: SimulationSnapshot, route_id: str, event_key: str, message: str) -> SimulationSnapshot:
    state = snapshot.route_states[route_id]
    if event_key in state.events:
        return snapshot
    logs = [SimulationLog(event_key, route_id, message)] + list(snapshot.logs)
    snapshot = replace(snapshot, logs=logs)
    return with_route_state(snapshot, route_id, replace(state, events=[*state.events, event_key]))