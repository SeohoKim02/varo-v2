"""Decision-level metrics shared by the feasibility gate, VHS and the UI.

Everything a transfer decision actually turns on is computed here **once**, from
the applied workbook, so the gate and the score cannot disagree about the same
number:

* ``net_benefit`` — 예상 절감액 − 이동 비용. The gross saving alone says nothing
  about whether a move is worth making; the cost was previously only a separate
  penalty term, which let an expensive move look as good as a cheap one.
* **Quantity decomposition** — how much *can* move (source stock above its safety
  floor), how much *needs* to move (destination shortfall), and which of the two
  actually limited the recommendation.
* **Post-move risk** — what the source and the destination look like *after* the
  move: does the source drop under its safety floor, does the destination end up
  overstocked.
* **Demand scenarios** — 보수적/기준/공격적 수요, derived only from a real
  ``demand_std`` in the data. When the file has no dispersion column the status is
  "계산 불가" and no standard deviation is invented.

Pure and deterministic: same workbook in, same numbers out. No Streamlit, no
randomness, no fabricated columns. Values that cannot be computed stay ``None`` /
``NaN`` and are never silently turned into 0.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd

from services.feasibility import InventoryContext, build_inventory_context

# Bumped whenever the scoring components, their normalization, the hard-constraint
# set, or the tie-break chain change — i.e. whenever the same workbook could
# produce a different ranking. Recorded on every result so a past recommendation
# can be traced back to the logic that produced it. Never shown on the main UI.
ALGORITHM_VERSION = "vhs-2.2"

# One standard deviation either side of the base demand. A wider band would make
# almost every candidate look fragile; a narrower one would never flag anything.
DEMAND_SCENARIO_Z = 1.0
# The destination is treated as over-supplied when the post-move stock exceeds
# this multiple of its measured need. Same constant the feasibility gate uses for
# its soft 과잉 check, so the two cannot drift apart.
OVERSUPPLY_MULTIPLE = 3.0

SCENARIO_STABLE = "안정"
SCENARIO_REVIEW = "확인 필요"
SCENARIO_VOLATILE = "변동 가능성 큼"
SCENARIO_UNKNOWN = "계산 불가"

# Columns this module adds to the candidate frame.
DECISION_COLUMNS = (
    "net_benefit", "net_benefit_computable",
    "source_stock", "source_safety_floor", "source_movable",
    "inventory_floor_value", "inventory_floor_source", "available_to_move",
    "source_reorder_point",
    "target_stock", "target_demand", "target_shortfall", "target_demand_std",
    "target_stock_goal", "target_stock_basis",
    "demand_low", "demand_base", "demand_high", "demand_scenario_status",
    "post_move_source_remaining", "post_move_source_gap",
    "post_move_target_stock", "post_move_target_excess",
    "qty_max_movable", "qty_demand_needed", "qty_limiting_factor", "quantity_limit_reason",
)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_decision_context(data: Mapping[str, Any] | None) -> InventoryContext:
    """Per (store, product) stock / demand / safety / dispersion from inventory."""
    return build_inventory_context(data)


def net_benefit(candidate: Mapping[str, Any]) -> float | None:
    """예상 절감액 − 이동 비용. ``None`` when either side is not computable.

    The cost falls back to ``move_cost`` because the adapter carries the same
    number under both names. A missing cost is *not* treated as zero cost.
    """
    saving = _num(candidate.get("expected_saving"))
    cost = _num(candidate.get("estimated_cost"))
    if cost is None:
        cost = _num(candidate.get("move_cost"))
    if saving is None or cost is None:
        return None
    return saving - cost


def demand_scenarios(base: float | None, std: float | None) -> dict[str, float] | None:
    """보수적 / 기준 / 공격적 수요. ``None`` when the file carries no dispersion."""
    if base is None or std is None or std <= 0:
        return None
    spread = std * DEMAND_SCENARIO_Z
    return {
        "low": max(0.0, base - spread),
        "base": max(0.0, base),
        "high": max(0.0, base + spread),
    }


def scenario_status(quantity: float | None, scenarios: Mapping[str, float] | None) -> str:
    """How well the recommended quantity holds up across the demand scenarios."""
    if quantity is None or not scenarios:
        return SCENARIO_UNKNOWN
    holds = sum(1 for key in ("low", "base", "high") if quantity <= scenarios[key] * OVERSUPPLY_MULTIPLE)
    if holds == 3:
        return SCENARIO_STABLE
    if holds >= 1:
        return SCENARIO_REVIEW
    return SCENARIO_VOLATILE


def quantity_plan(candidate: Mapping[str, Any], context: InventoryContext) -> dict[str, Any]:
    """Split the recommended quantity into its real limits.

    ``qty_max_movable`` is what the source can give up without breaking its own
    safety floor; ``qty_demand_needed`` is what the destination is short by. The
    limiting factor is whichever is smaller — that is the sentence the UI shows.
    """
    source, target = candidate.get("source_id"), candidate.get("target_id")
    product = candidate.get("product_id")
    quantity = _num(candidate.get("recommended_qty"))
    stock = context.source_stock(source, product)
    safety = context.safety_floor(source, product)
    target_stock = context.source_stock(target, product)
    demand = context.target_demand(target, product)

    floor_source = context.inventory_floor_source(source, product)
    movable = context.available_to_move(source, product)
    target_goal = context.target_stock_level(target, product)
    shortfall = None
    target_basis = None
    if target_goal is not None and target_stock is not None:
        shortfall = max(0.0, target_goal - target_stock)
        target_basis = "explicit_target_stock"
    elif demand is not None:
        shortfall = max(0.0, demand - (target_stock or 0.0))
        target_basis = "demand_fallback"

    limits: list[tuple[str, str, float]] = []
    if movable is not None:
        limits.append(("출발 점포 이동 가능량", "source_inventory_floor", movable))
    if shortfall is not None and shortfall > 0:
        target_label = "도착 점포 목표 재고 부족량" if target_goal is not None else "도착 점포 부족량"
        target_code = "target_stock_gap" if target_goal is not None else "target_demand_gap"
        limits.append((target_label, target_code, shortfall))
    limiting = None
    limit_reason = None
    if limits:
        limits.sort(key=lambda item: (item[2], item[0]))
        limiting = limits[0][0]
        limit_reason = limits[0][1]
        if len(limits) == 2 and abs(limits[0][2] - limits[1][2]) < 1e-9:
            limiting = "출발 가능량과 도착 부족량이 같음"
            limit_reason = "equal_source_target_limits"

    remaining = None if stock is None or quantity is None else stock - quantity
    gap = None if remaining is None or safety is None else max(0.0, safety - remaining)
    post_target = None if target_stock is None or quantity is None else target_stock + quantity
    excess = None
    if post_target is not None and target_goal is not None:
        excess = max(0.0, post_target - target_goal)
    elif post_target is not None and demand is not None and demand > 0:
        excess = max(0.0, post_target - demand * OVERSUPPLY_MULTIPLE)

    return {
        "source_stock": stock,
        "source_safety_floor": safety if stock is not None else None,
        "source_movable": movable,
        "inventory_floor_value": safety if stock is not None else None,
        "inventory_floor_source": floor_source,
        "available_to_move": movable,
        "source_reorder_point": context.reorder_point(source, product),
        "target_stock": target_stock,
        "target_demand": demand,
        "target_shortfall": shortfall,
        "target_stock_goal": target_goal,
        "target_stock_basis": target_basis,
        "qty_max_movable": movable,
        "qty_demand_needed": shortfall,
        "qty_limiting_factor": limiting,
        "quantity_limit_reason": limit_reason,
        "post_move_source_remaining": remaining,
        "post_move_source_gap": gap,
        "post_move_target_stock": post_target,
        "post_move_target_excess": excess,
    }


def decision_record(candidate: Mapping[str, Any], context: InventoryContext) -> dict[str, Any]:
    """Every decision metric for one candidate, as a flat dict."""
    plan = quantity_plan(candidate, context)
    quantity = _num(candidate.get("recommended_qty"))
    std = context.demand_std(candidate.get("target_id"), candidate.get("product_id"))
    scenarios = demand_scenarios(plan["target_demand"], std)
    benefit = net_benefit(candidate)
    record = dict(plan)
    record.update({
        "net_benefit": benefit,
        "net_benefit_computable": benefit is not None,
        "target_demand_std": std,
        "demand_low": None if scenarios is None else scenarios["low"],
        "demand_base": None if scenarios is None else scenarios["base"],
        "demand_high": None if scenarios is None else scenarios["high"],
        "demand_scenario_status": scenario_status(quantity, scenarios),
    })
    return record


def annotate_decision_metrics(
    candidates: pd.DataFrame, data: Mapping[str, Any] | None = None,
    context: InventoryContext | None = None,
) -> pd.DataFrame:
    """Attach ``DECISION_COLUMNS`` to a candidate frame (never mutates the input)."""
    if candidates is None or candidates.empty:
        return candidates
    context = context or build_decision_context(data)
    frame = candidates.copy()
    records = [decision_record(row, context) for _, row in frame.iterrows()]
    for column in DECISION_COLUMNS:
        values = [record.get(column) for record in records]
        if column in ("net_benefit_computable",):
            frame[column] = pd.Series(values, index=frame.index).fillna(False).astype(bool)
        elif column in (
            "demand_scenario_status", "qty_limiting_factor", "inventory_floor_source",
            "target_stock_basis", "quantity_limit_reason",
        ):
            frame[column] = pd.Series(values, index=frame.index, dtype=object)
        else:
            frame[column] = pd.to_numeric(pd.Series(values, index=frame.index), errors="coerce")
    return frame


def quantity_basis_text(record: Mapping[str, Any]) -> str | None:
    """One plain sentence explaining the quantity, or ``None`` when unknown."""
    quantity = _num(record.get("recommended_qty"))
    factor = record.get("qty_limiting_factor")
    if quantity is None or not factor:
        return None
    return f"{factor}을 기준으로 {quantity:,.0f}개 이동을 권장합니다."


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate decision metrics for diagnostics and the benchmark report."""
    benefits = [value for value in (_num(row.get("net_benefit")) for row in records) if value is not None]
    statuses: dict[str, int] = {}
    for row in records:
        key = str(row.get("demand_scenario_status") or SCENARIO_UNKNOWN)
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "candidate_count": len(list(records)),
        "net_benefit_total": round(sum(benefits), 2) if benefits else None,
        "net_benefit_median": round(sorted(benefits)[len(benefits) // 2], 2) if benefits else None,
        "net_benefit_computable": len(benefits),
        "demand_scenario_distribution": statuses,
        "scenario_basis": (
            f"수요 표준편차 ±{DEMAND_SCENARIO_Z}σ 기준 보수적/기준/공격적 시나리오 "
            "(표준편차 컬럼이 없으면 계산하지 않음)"
        ),
    }
