"""Multi-route simulation engine for Varo V2."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Optional

from simulation.route_animation import DIRECT, VIA_DC, build_route_legs, validate_route
from simulation.simulation_state import (
    AT_DC,
    COMPLETED,
    DC_DWELL_TICKS,
    ERROR,
    MOVING,
    PAUSED,
    READY,
    SPEED_STEPS,
    RouteRuntimeState,
    SimulationSnapshot,
    append_log_once,
    with_route_state,
)

MAX_VISIBLE_ROUTES = 5


def select_visible_routes(routes: Iterable[Mapping[str, object]], limit: int = MAX_VISIBLE_ROUTES) -> List[Dict[str, object]]:
    """Return the first N routes without calculating rank or score."""
    if limit < 1:
        return []
    return [dict(route) for route in list(routes)[:limit]]


def validate_routes(routes: Iterable[Mapping[str, object]]) -> List[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for route in routes:
        route_id = str(route.get("route_id", ""))
        if route_id and route_id in seen:
            errors.append(f"중복 route_id가 있습니다: {route_id}")
        if route_id:
            seen.add(route_id)
        try:
            validate_route(route)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def create_simulation(
    routes: Iterable[Mapping[str, object]],
    selected_route_id: Optional[str] = None,
    speed_label: str = "보통",
    show_all_routes: bool = False,
    max_routes: int = MAX_VISIBLE_ROUTES,
) -> SimulationSnapshot:
    visible_routes = select_visible_routes(routes, max_routes)
    errors = validate_routes(visible_routes)
    states: dict[str, RouteRuntimeState] = {}
    for route in visible_routes:
        route_id = str(route.get("route_id", ""))
        if route_id and route_id not in states:
            status = ERROR if any(route_id in error for error in errors) else READY
            states[route_id] = RouteRuntimeState(route_id=route_id, status=status)
    return SimulationSnapshot(
        routes=visible_routes,
        route_states=states,
        is_running=False,
        speed_label=speed_label if speed_label in SPEED_STEPS else "보통",
        selected_route_id=selected_route_id,
        show_all_routes=show_all_routes,
        errors=errors,
    )


def _node_label(route: Mapping[str, object], name_key: str, id_key: str) -> str:
    value = route.get(name_key) or route.get(id_key) or "-"
    return str(value)


def _qty(route: Mapping[str, object]) -> str:
    value = route.get("recommended_qty")
    if value is None:
        return "-"
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except (TypeError, ValueError):
        return str(value)


def _departure_message(route: Mapping[str, object]) -> str:
    product = route.get("product_name") or "상품"
    source = _node_label(route, "source_name", "source_id")
    return f"{source}에서 {product} {_qty(route)}개 출발"


def _dc_arrival_message(route: Mapping[str, object]) -> str:
    product = route.get("product_name") or "상품"
    dc = _node_label(route, "dc_name", "dc_id")
    return f"{product} 운송 차량이 {dc}에 도착"


def _completion_message(route: Mapping[str, object]) -> str:
    product = route.get("product_name") or "상품"
    target = _node_label(route, "target_name", "target_id")
    return f"{target}로 {product} 이동 완료"


def start_simulation(snapshot: SimulationSnapshot) -> SimulationSnapshot:
    if snapshot.errors:
        return replace(snapshot, is_running=False)
    next_snapshot = replace(snapshot, is_running=True)
    for route in next_snapshot.routes:
        route_id = str(route["route_id"])
        state = next_snapshot.route_states[route_id]
        if state.status in (READY, PAUSED):
            resumed_status = state.paused_from if state.paused_from in (MOVING, AT_DC) else MOVING
            next_snapshot = with_route_state(
                next_snapshot,
                route_id,
                replace(state, status=resumed_status, paused_from=None),
            )
            if state.status == READY:
                next_snapshot = append_log_once(next_snapshot, route_id, f"{route_id}:departed", _departure_message(route))
    return next_snapshot


def pause_simulation(snapshot: SimulationSnapshot) -> SimulationSnapshot:
    next_snapshot = replace(snapshot, is_running=False)
    for route in next_snapshot.routes:
        route_id = str(route["route_id"])
        state = next_snapshot.route_states[route_id]
        if state.status in (MOVING, AT_DC):
            next_snapshot = with_route_state(next_snapshot, route_id, replace(state, status=PAUSED, paused_from=state.status))
    return next_snapshot


def restart_simulation(snapshot: SimulationSnapshot) -> SimulationSnapshot:
    return create_simulation(
        snapshot.routes,
        selected_route_id=snapshot.selected_route_id,
        speed_label=snapshot.speed_label,
        show_all_routes=snapshot.show_all_routes,
        max_routes=MAX_VISIBLE_ROUTES,
    )


def select_route(snapshot: SimulationSnapshot, route_id: Optional[str]) -> SimulationSnapshot:
    return replace(snapshot, selected_route_id=route_id)


def set_speed(snapshot: SimulationSnapshot, speed_label: str) -> SimulationSnapshot:
    return replace(snapshot, speed_label=speed_label if speed_label in SPEED_STEPS else "보통")


def set_show_all_routes(snapshot: SimulationSnapshot, show_all_routes: bool) -> SimulationSnapshot:
    return replace(snapshot, show_all_routes=show_all_routes)


def _advance_route(route: Mapping[str, object], state: RouteRuntimeState, step: float) -> RouteRuntimeState:
    if state.status == COMPLETED or state.status in (READY, PAUSED, ERROR):
        return state
    if route.get("route_type") == VIA_DC and state.status == AT_DC:
        if state.dc_wait_ticks < DC_DWELL_TICKS:
            return replace(state, dc_wait_ticks=state.dc_wait_ticks + 1)
        return replace(state, status=MOVING, current_leg_index=1, progress=0.0, dc_wait_ticks=0)
    if state.status != MOVING:
        return state
    progress = min(1.0, state.progress + step)
    if progress < 1.0:
        return replace(state, progress=progress)
    if route.get("route_type") == DIRECT:
        return replace(state, status=COMPLETED, progress=1.0)
    if route.get("route_type") == VIA_DC and state.current_leg_index == 0:
        return replace(state, status=AT_DC, current_leg_index=0, progress=1.0, dc_wait_ticks=0)
    return replace(state, status=COMPLETED, progress=1.0, current_leg_index=1)


def advance_simulation(snapshot: SimulationSnapshot, step: Optional[float] = None) -> SimulationSnapshot:
    """Advance all active routes one UI tick."""
    if not snapshot.is_running or snapshot.errors:
        return snapshot
    step_value = step if step is not None else SPEED_STEPS.get(snapshot.speed_label, SPEED_STEPS["보통"])
    next_snapshot = snapshot
    all_done = True
    for route in next_snapshot.routes:
        route_id = str(route["route_id"])
        before = next_snapshot.route_states[route_id]
        route_step = step_value * (2.0 if route.get("route_type") == VIA_DC else 1.0)
        after = _advance_route(route, before, route_step)
        next_snapshot = with_route_state(next_snapshot, route_id, after)
        if route.get("route_type") == VIA_DC and before.status == MOVING and after.status == AT_DC:
            next_snapshot = append_log_once(next_snapshot, route_id, f"{route_id}:at_dc", _dc_arrival_message(route))
        if before.status != COMPLETED and after.status == COMPLETED:
            next_snapshot = append_log_once(next_snapshot, route_id, f"{route_id}:completed", _completion_message(route))
        if after.status not in (COMPLETED, ERROR):
            all_done = False
    return replace(next_snapshot, is_running=not all_done)


def route_legs_for_snapshot(snapshot: SimulationSnapshot) -> Dict[str, List[Dict[str, object]]]:
    """Build route legs for every valid route in a snapshot."""
    legs: dict[str, list[dict[str, object]]] = {}
    for route in snapshot.routes:
        route_id = str(route.get("route_id", ""))
        if route_id in snapshot.route_states and snapshot.route_states[route_id].status != ERROR:
            legs[route_id] = build_route_legs(route)
    return legs