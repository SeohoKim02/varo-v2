"""Deterministic Pareto frontier and operational route selection.

Frontier mathematics and production selection are deliberately separate:
the former only determines non-dominance, while the latter repeatedly chooses
one compromise route, updates shared feasibility through a callback, and then
recomputes the frontier until the selection limit is reached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


FeasibilityCheck = Callable[[Sequence[Mapping[str, Any]]], bool | tuple[bool, str | None]]


@dataclass(frozen=True)
class ParetoObjective:
    name: str
    direction: str
    fields: tuple[str, ...]
    optional_if_constant: bool = False


@dataclass
class ParetoSelectionResult:
    selected_rows: list[dict[str, Any]] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    ranks: list[int] = field(default_factory=list)
    active_objectives: list[str] = field(default_factory=list)
    inactive_objectives: list[str] = field(default_factory=list)
    compromise_scores: dict[int, float] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)


OBJECTIVES = (
    ParetoObjective("service_qty", "maximize", ("recommended_qty", "suggested_qty", "transfer_qty", "quantity")),
    ParetoObjective("transport_cost", "minimize", ("move_cost", "estimated_cost", "transport_cost", "transfer_cost")),
    ParetoObjective("expected_saving", "maximize", ("expected_saving", "saving", "estimated_saving"), optional_if_constant=True),
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _value(row: Mapping[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        if field not in row:
            continue
        value = _number(row.get(field))
        if value is not None:
            return value
    return None


def _route_id(row: Mapping[str, Any], fallback: int) -> str:
    return str(row.get("route_id") or row.get("recommendation_id") or f"candidate-{fallback + 1}")


def active_objectives(rows: Sequence[Mapping[str, Any]]) -> tuple[list[ParetoObjective], list[str]]:
    """Return objectives supported by this candidate set.

    Service and cost are mandatory.  Optional saving is disabled when missing
    or zero-variance, preventing a constant dimension from affecting distance
    normalization while allowing it to activate on a future dataset.
    """
    active: list[ParetoObjective] = []
    inactive: list[str] = []
    for objective in OBJECTIVES:
        values = [_value(row, objective.fields) for row in rows]
        finite = [value for value in values if value is not None]
        if objective.optional_if_constant and (
            len(finite) != len(rows) or not finite or max(finite) - min(finite) <= 1e-12
        ):
            inactive.append(objective.name)
            continue
        active.append(objective)
    return active, inactive


def _valid_objective_row(row: Mapping[str, Any], objectives: Sequence[ParetoObjective]) -> tuple[bool, str | None]:
    service = _value(row, OBJECTIVES[0].fields)
    cost = _value(row, OBJECTIVES[1].fields)
    if service is None or service <= 0:
        return False, "service quantity is missing or non-positive"
    if cost is None:
        return False, "transport cost is missing/non-finite"
    if cost < 0:
        return False, "transport cost is negative"
    for objective in objectives:
        if _value(row, objective.fields) is None:
            return False, f"{objective.name} is missing/non-finite"
    return True, None


def _desirability(row: Mapping[str, Any], objectives: Sequence[ParetoObjective]) -> tuple[float, ...]:
    values = []
    for objective in objectives:
        value = float(_value(row, objective.fields))
        values.append(value if objective.direction == "maximize" else -value)
    return tuple(values)


def frontier_indices(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[ParetoObjective] | None = None,
) -> list[int]:
    """Return deterministic input indexes of the non-dominated frontier."""
    items = list(rows or [])
    active = list(objectives or active_objectives(items)[0])
    valid = [index for index, row in enumerate(items) if _valid_objective_row(row, active)[0]]
    points = {index: _desirability(items[index], active) for index in valid}
    front: list[int] = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if candidate == other:
                continue
            at_least_as_good = all(left >= right for left, right in zip(points[other], points[candidate]))
            strictly_better = any(left > right for left, right in zip(points[other], points[candidate]))
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda index: (_route_id(items[index], index), index))


def pareto_layers(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return non-dominated layer numbers; invalid rows are ranked last."""
    items = list(rows or [])
    if not items:
        return []
    objectives, _ = active_objectives(items)
    valid = {index for index, row in enumerate(items) if _valid_objective_row(row, objectives)[0]}
    remaining = set(valid)
    ranks = [0] * len(items)
    layer = 1
    while remaining:
        ordered = sorted(remaining)
        local_rows = [items[index] for index in ordered]
        local_front = frontier_indices(local_rows, objectives)
        front = [ordered[index] for index in local_front]
        if not front:
            front = [ordered[0]]
        for index in front:
            ranks[index] = layer
            remaining.remove(index)
        layer += 1
    for index in sorted(set(range(len(items))) - valid, key=lambda value: (_route_id(items[value], value), value)):
        ranks[index] = layer
    return ranks


def compromise_distances(
    rows: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
    objectives: Sequence[ParetoObjective],
) -> dict[int, float]:
    """Euclidean distance to the normalized ideal point (all objectives=1)."""
    selected = list(indexes)
    if not selected:
        return {}
    columns: list[tuple[ParetoObjective, dict[int, float]]] = []
    for objective in objectives:
        raw = {index: float(_value(rows[index], objective.fields)) for index in selected}
        low, high = min(raw.values()), max(raw.values())
        if high - low <= 1e-12:
            continue
        if objective.direction == "maximize":
            normalized = {index: (value - low) / (high - low) for index, value in raw.items()}
        else:
            normalized = {index: (high - value) / (high - low) for index, value in raw.items()}
        columns.append((objective, normalized))
    if not columns:
        return {index: 0.0 for index in selected}
    return {
        index: math.sqrt(sum((1.0 - values[index]) ** 2 for _, values in columns) / len(columns))
        for index in selected
    }


def _explicit_false(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "n", "불가", "실패", "fail", "failed"}


def _default_feasible(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, str | None]:
    route_ids: set[str] = set()
    duplicate_keys: set[tuple[str, ...]] = set()
    source_usage: dict[tuple[str, str], float] = {}
    target_usage: dict[tuple[str, str], float] = {}
    source_caps: dict[tuple[str, str], float] = {}
    target_caps: dict[tuple[str, str], float] = {}
    for position, row in enumerate(rows):
        route_id = _route_id(row, position)
        if route_id in route_ids:
            return False, "duplicate route"
        route_ids.add(route_id)
        for field in ("feasible", "is_feasible", "route_feasible"):
            if field in row and _explicit_false(row.get(field)):
                return False, f"{field} is false"
        product = str(row.get("product_id") or "")
        source = str(row.get("source_id") or row.get("from_store_id") or "")
        target = str(row.get("target_id") or row.get("to_store_id") or "")
        route_type = str(row.get("route_type") or "DIRECT")
        dc_id = str(row.get("dc_id") or "")
        duplicate_key = (product, source, target, route_type, dc_id)
        if duplicate_key in duplicate_keys:
            return False, "duplicate allocation"
        duplicate_keys.add(duplicate_key)
        quantity = float(_value(row, OBJECTIVES[0].fields) or 0.0)
        route_capacity = _value(row, ("route_capacity_qty", "vehicle_capacity_qty", "max_load_qty"))
        if route_capacity is not None and quantity > route_capacity + 1e-9:
            return False, "route capacity exceeded"
        source_key, target_key = (source, product), (target, product)
        source_usage[source_key] = source_usage.get(source_key, 0.0) + quantity
        target_usage[target_key] = target_usage.get(target_key, 0.0) + quantity
        source_cap = _value(row, ("source_surplus", "available_transfer_stock", "transferable_stock"))
        target_cap = _value(row, ("target_need_7d", "target_shortage_qty", "shortage_qty", "unmet_demand"))
        if source_cap is not None:
            source_caps[source_key] = max(0.0, source_cap)
        if target_cap is not None:
            target_caps[target_key] = max(0.0, target_cap)
    if any(value > source_caps[key] + 1e-9 for key, value in source_usage.items() if key in source_caps):
        return False, "source inventory exceeded"
    if any(value > target_caps[key] + 1e-9 for key, value in target_usage.items() if key in target_caps):
        return False, "target demand exceeded"
    return True, None


def _check_feasible(
    rows: Sequence[Mapping[str, Any]],
    feasibility_check: FeasibilityCheck | None,
) -> tuple[bool, str | None]:
    result = (feasibility_check or _default_feasible)(rows)
    if isinstance(result, tuple):
        return bool(result[0]), str(result[1]) if result[1] else None
    return bool(result), None


def select_pareto_routes(
    rows: Sequence[Mapping[str, Any]],
    max_routes: int = 5,
    feasibility_check: FeasibilityCheck | None = None,
) -> ParetoSelectionResult:
    """Repeatedly recompute frontier and choose the closest ideal compromise."""
    items = [dict(row) for row in rows or []]
    objectives, inactive = active_objectives(items)
    result = ParetoSelectionResult(
        ranks=pareto_layers(items),
        active_objectives=[objective.name for objective in objectives],
        inactive_objectives=inactive,
    )
    remaining = list(range(len(items)))
    selected_indices: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    while remaining and len(selected_rows) < max(0, int(max_routes)):
        feasible_pool: list[int] = []
        for index in remaining:
            valid, reason = _valid_objective_row(items[index], objectives)
            if not valid:
                result.excluded.append({"index": index, "route_id": _route_id(items[index], index), "reason": reason})
                continue
            feasible, reason = _check_feasible([*selected_rows, items[index]], feasibility_check)
            if feasible:
                feasible_pool.append(index)
            else:
                result.excluded.append({"index": index, "route_id": _route_id(items[index], index), "reason": reason or "shared constraint"})
        if not feasible_pool:
            break
        local_rows = [items[index] for index in feasible_pool]
        local_front = frontier_indices(local_rows, objectives)
        front = [feasible_pool[position] for position in local_front]
        distances = compromise_distances(items, front, objectives)
        chosen = min(front, key=lambda index: (distances[index], _route_id(items[index], index), index))
        result.compromise_scores[chosen] = round(1.0 / (1.0 + distances[chosen]), 12)
        selected_indices.append(chosen)
        selected_rows.append(items[chosen])
        result.trace.append({
            "step": len(selected_rows),
            "objectives": [
                {"name": objective.name, "direction": objective.direction}
                for objective in objectives
            ],
            "feasible_candidate_count": len(feasible_pool),
            "frontier_size": len(front),
            "frontier_route_ids": [_route_id(items[index], index) for index in front],
            "selected_route_id": _route_id(items[chosen], chosen),
            "selected_objective_values": {
                objective.name: _value(items[chosen], objective.fields)
                for objective in objectives
            },
            "distance_to_ideal": round(distances[chosen], 12),
            "compromise_score": result.compromise_scores[chosen],
        })
        remaining.remove(chosen)
    result.selected_indices = selected_indices
    result.selected_rows = selected_rows
    return result
