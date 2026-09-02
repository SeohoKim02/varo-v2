"""Build one conflict-free inventory transfer plan from ranked candidates.

VHS remains the candidate-value model.  This module runs afterwards and only
decides which already-evaluated moves can be executed together and at what
integer quantity.  It never invents capacity, pack-size, revenue, or penalty
data: the model uses the current candidate saving/cost plus the inventory floor
and destination shortfall already calculated by :mod:`services.decision_metrics`.

SciPy's bundled HiGHS MILP is used when available.  The import is optional so a
deployment without SciPy still returns a deterministic, constraint-safe greedy
plan.  Every result is independently validated before it can reach the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from services.candidate_ledger import ledger_by_route, make_candidate_id
from services.decision_metrics import quantity_plan
from services.feasibility import InventoryContext, build_inventory_context


PLAN_ALGORITHM_VERSION = "execution-plan-1.0"
PLAN_STATUS_READY = "실행 가능"
PLAN_STATUS_PARTIAL = "일부 추천 가능"
PLAN_STATUS_EMPTY = "실행 가능한 이동 없음"
PLAN_STATUS_UNAVAILABLE = "계산 불가"

REASON_TEXT = {
    "shared_source_inventory": "출발 재고가 다른 우선 이동에 배정되었습니다.",
    "destination_fulfilled": "도착 점포의 필요 수량이 다른 이동으로 충족되었습니다.",
    "better_route_selected": "더 효율적인 경로가 선택되었습니다.",
    "lower_plan_value": "전체 이동계획의 예상 효과를 비교해 다른 이동을 우선했습니다.",
    "infeasible": "현재 데이터로 동시에 실행 가능한 이동을 계산할 수 없습니다.",
    "negative_net_benefit": "이동 비용이 예상 효과보다 커 실행계획에서 제외했습니다.",
    "selected": "공유 재고와 필요 수량 범위에서 우선 실행할 이동입니다.",
    "quantity_adjusted": "다른 추천과 재고를 함께 배분해 실행 수량을 조정했습니다.",
}

_EPS = 1e-7
_STABLE_LABELS = ("안정", "높음", "stable", "robust")


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _floor_nonnegative(value: Any) -> int | None:
    number = _num(value)
    if number is None:
        return None
    return max(0, int(math.floor(number + _EPS)))


def _stable_flag(candidate: Mapping[str, Any]) -> float:
    text = " ".join(
        _text(candidate.get(key)).lower()
        for key in ("robustness_status", "demand_scenario_status", "confidence_level")
    )
    return 1.0 if any(label in text for label in _STABLE_LABELS) else 0.0


@dataclass(frozen=True)
class _RouteCatalog:
    stores: set[str]
    products: set[str]
    dcs: set[str]
    direct_pairs: set[tuple[str, str]]
    via_options: set[tuple[str, str, str]]

    def valid(self, source: str, target: str, product: str, route_type: str, dc_id: str) -> bool:
        if source not in self.stores or target not in self.stores or product not in self.products:
            return False
        if route_type == "DIRECT":
            return (source, target) in self.direct_pairs
        if route_type != "VIA_DC" or not dc_id or dc_id not in self.dcs:
            return False
        return (
            (source, target, dc_id) in self.via_options
            or ((source, dc_id) in self.direct_pairs and (dc_id, target) in self.direct_pairs)
        )


def _route_catalog(data: Mapping[str, Any] | None) -> _RouteCatalog:
    data = data if isinstance(data, Mapping) else {}
    stores_frame = data.get("stores")
    products_frame = data.get("products")
    routes_frame = data.get("routes")
    dcs_frame = data.get("dcs")

    stores: set[str] = set()
    dcs: set[str] = set()
    if isinstance(stores_frame, pd.DataFrame) and not stores_frame.empty:
        id_col = _column(stores_frame, "store_id", "node_id", "id")
        type_col = _column(stores_frame, "node_type", "type", "store_type")
        if id_col:
            stores = {_text(value) for value in stores_frame[id_col] if _text(value)}
            if type_col:
                dcs = {
                    _text(row.get(id_col))
                    for _, row in stores_frame.iterrows()
                    if _text(row.get(type_col)).upper() == "DC" and _text(row.get(id_col))
                }
    if isinstance(dcs_frame, pd.DataFrame) and not dcs_frame.empty:
        dc_col = _column(dcs_frame, "dc_id", "node_id", "store_id", "id")
        if dc_col:
            extra = {_text(value) for value in dcs_frame[dc_col] if _text(value)}
            dcs.update(extra)
            stores.update(extra)

    products: set[str] = set()
    if isinstance(products_frame, pd.DataFrame) and not products_frame.empty:
        product_col = _column(products_frame, "product_id", "item_id", "id")
        if product_col:
            products = {_text(value) for value in products_frame[product_col] if _text(value)}

    direct_pairs: set[tuple[str, str]] = set()
    via_options: set[tuple[str, str, str]] = set()
    if isinstance(routes_frame, pd.DataFrame) and not routes_frame.empty:
        source_col = _column(routes_frame, "source_id", "from_store_id", "from_id")
        target_col = _column(routes_frame, "target_id", "to_store_id", "to_id")
        type_col = _column(routes_frame, "route_type")
        dc_col = _column(routes_frame, "dc_id")
        if source_col and target_col:
            for _, row in routes_frame.iterrows():
                source, target = _text(row.get(source_col)), _text(row.get(target_col))
                if not source or not target:
                    continue
                route_type = _text(row.get(type_col)).upper() if type_col else ""
                dc_id = _text(row.get(dc_col)) if dc_col else ""
                if route_type == "VIA_DC" and dc_id:
                    via_options.add((source, target, dc_id))
                    dcs.add(dc_id)
                    stores.add(dc_id)
                else:
                    direct_pairs.add((source, target))
    return _RouteCatalog(stores, products, dcs, direct_pairs, via_options)


@dataclass(frozen=True)
class _Candidate:
    raw: dict[str, Any]
    candidate_id: str
    route_id: str
    source: str
    target: str
    product: str
    route_type: str
    dc_id: str
    recommended_qty: int
    max_qty: int
    source_limit: int
    target_limit: int
    cost: float
    cost_per_unit: float
    saving: float
    saving_per_unit: float
    full_net_benefit: float
    vhs_score: float
    vhs_rank: float
    greedy_rank: float
    stable_flag: float

    @property
    def source_key(self) -> tuple[str, str]:
        return (self.source, self.product)

    @property
    def target_key(self) -> tuple[str, str]:
        return (self.target, self.product)

    @property
    def route_group(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.product)

    @property
    def canonical_key(self) -> tuple[str, ...]:
        return (
            self.source, self.product, self.target, self.route_type,
            self.dc_id or "-", self.route_id, self.candidate_id,
        )

    def planned_values(self, quantity: int) -> tuple[float, float, float]:
        saving = self.saving_per_unit * quantity
        # The candidate supplies one cost/saving estimate for its recommended
        # quantity. With no vehicle-capacity or fixed-cost split in the data, the
        # only non-invented partial-quantity basis is the same observed per-unit
        # rate for both values.
        cost = self.cost_per_unit * quantity
        return saving, cost, saving - cost


def _candidate_id(
    candidate: Mapping[str, Any], ledger: Mapping[str, Mapping[str, Any]], data_signature: str | None,
) -> str:
    record = ledger.get(_text(candidate.get("route_id"))) or {}
    return _text(record.get("candidate_id")) or make_candidate_id(candidate, data_signature)


def _prepare_candidates(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    candidate_ledger: Sequence[Mapping[str, Any]] | None,
    data_signature: str | None,
) -> tuple[list[_Candidate], list[dict[str, Any]], InventoryContext, _RouteCatalog]:
    context = build_inventory_context(data)
    catalog = _route_catalog(data)
    ledger = ledger_by_route(candidate_ledger or [])
    valid: list[_Candidate] = []
    rejected: list[dict[str, Any]] = []

    for raw_item in recommendations or []:
        raw = dict(raw_item)
        route_id = _text(raw.get("route_id"))
        candidate_id = _candidate_id(raw, ledger, data_signature)
        source, target, product = (_text(raw.get(key)) for key in ("source_id", "target_id", "product_id"))
        route_type = _text(raw.get("route_type")).upper()
        dc_id = _text(raw.get("dc_id"))
        recommended = _floor_nonnegative(raw.get("recommended_qty"))
        plan = quantity_plan(raw, context)
        source_limit = _floor_nonnegative(plan.get("available_to_move"))
        # Explicit target_stock wins.  Otherwise keep the candidate generator's
        # established shortage rule (product-median gap + seven-day demand).
        target_limit = _floor_nonnegative(context.planning_shortfall(target, product))
        cost = _num(raw.get("estimated_cost"))
        if cost is None:
            cost = _num(raw.get("move_cost"))
        saving = _num(raw.get("expected_saving"))
        benefit = _num(raw.get("net_benefit"))
        if benefit is None and saving is not None and cost is not None:
            benefit = saving - cost

        reason = None
        if benefit is not None and benefit <= _EPS:
            reason = "negative_net_benefit"
        elif (
            not route_id or not source or not target or not product
            or recommended is None or recommended <= 0
            or source_limit is None or target_limit is None
            or cost is None or cost < 0 or saving is None or saving <= 0
            or not catalog.valid(source, target, product, route_type, dc_id)
        ):
            reason = "infeasible"

        if reason:
            rejected.append({
                "candidate_id": candidate_id,
                "route_id": route_id,
                "reason_code": reason,
                "reason": REASON_TEXT[reason],
            })
            continue

        max_qty = min(recommended, source_limit, target_limit)
        if max_qty <= 0:
            reason = "destination_fulfilled" if target_limit <= 0 else "shared_source_inventory"
            rejected.append({
                "candidate_id": candidate_id,
                "route_id": route_id,
                "reason_code": reason,
                "reason": REASON_TEXT[reason],
            })
            continue

        valid.append(_Candidate(
            raw=raw,
            candidate_id=candidate_id,
            route_id=route_id,
            source=source,
            target=target,
            product=product,
            route_type=route_type,
            dc_id=dc_id,
            recommended_qty=recommended,
            max_qty=max_qty,
            source_limit=source_limit,
            target_limit=target_limit,
            cost=float(cost),
            cost_per_unit=float(cost) / recommended,
            saving=float(saving),
            saving_per_unit=float(saving) / recommended,
            full_net_benefit=float(benefit),
            vhs_score=_num(raw.get("vhs_score")) or 0.0,
            vhs_rank=_num(raw.get("varo_final_rank")) or _num(raw.get("vhs_rank")) or 1_000_000.0,
            greedy_rank=_num(raw.get("greedy_rank")) or 1_000_000.0,
            stable_flag=_stable_flag(raw),
        ))
    valid.sort(key=lambda item: item.canonical_key)
    return valid, rejected, context, catalog


def _group_indices(candidates: Sequence[_Candidate], attr: str) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(getattr(candidate, attr), []).append(index)
    return groups


def _solve_milp(candidates: Sequence[_Candidate], timeout_seconds: float) -> list[int]:
    """Lexicographic MILP: net benefit, relief quantity, stability, then VHS."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    count = len(candidates)
    size = count * 2  # x quantity, y route selected
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: Mapping[int, float], lb: float, ub: float) -> None:
        row = np.zeros(size, dtype=float)
        for index, value in coefficients.items():
            row[index] = value
        rows.append(row)
        lower.append(lb)
        upper.append(ub)

    # x <= max*y, x >= y, and every selected move must have positive net effect.
    for index, candidate in enumerate(candidates):
        add({index: 1.0, count + index: -candidate.max_qty}, -np.inf, 0.0)
        add({index: 1.0, count + index: -1.0}, 0.0, np.inf)
        add(
            {
                index: candidate.saving_per_unit - candidate.cost_per_unit,
                count + index: -_EPS,
            },
            0.0,
            np.inf,
        )

    for indices in _group_indices(candidates, "source_key").values():
        limit = min(candidates[index].source_limit for index in indices)
        add({index: 1.0 for index in indices}, -np.inf, float(limit))
    for indices in _group_indices(candidates, "target_key").values():
        limit = min(candidates[index].target_limit for index in indices)
        add({index: 1.0 for index in indices}, -np.inf, float(limit))
    for indices in _group_indices(candidates, "route_group").values():
        add({count + index: 1.0 for index in indices}, -np.inf, 1.0)

    lb = np.zeros(size, dtype=float)
    ub = np.array([candidate.max_qty for candidate in candidates] + [1.0] * count, dtype=float)
    integrality = np.ones(size, dtype=int)

    objectives = []
    net = np.zeros(size, dtype=float)
    quantity = np.zeros(size, dtype=float)
    stable = np.zeros(size, dtype=float)
    vhs = np.zeros(size, dtype=float)
    for index, candidate in enumerate(candidates):
        net[index] = candidate.saving_per_unit - candidate.cost_per_unit
        quantity[index] = 1.0
        stable[index] = candidate.stable_flag
        vhs[index] = candidate.vhs_score
    objectives.extend((net, quantity, stable, vhs))

    solution = None
    for objective in objectives:
        constraints = LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))
        result = milp(
            -objective,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={"time_limit": max(0.1, float(timeout_seconds)), "presolve": True},
        )
        if not bool(result.success) or result.x is None:
            raise RuntimeError(_text(getattr(result, "message", "optimizer failed")))
        solution = result.x
        optimum = float(objective @ solution)
        tolerance = max(1e-6, abs(optimum) * 1e-9)
        add({index: float(value) for index, value in enumerate(objective) if abs(value) > _EPS}, optimum - tolerance, np.inf)

    if solution is None:
        return [0] * count
    return [max(0, min(candidate.max_qty, int(round(solution[index])))) for index, candidate in enumerate(candidates)]


def _solve_greedy(candidates: Sequence[_Candidate]) -> list[int]:
    """Existing Greedy order with the same shared-resource and route constraints."""
    source_remaining = {candidate.source_key: candidate.source_limit for candidate in candidates}
    target_remaining = {candidate.target_key: candidate.target_limit for candidate in candidates}
    chosen_routes: set[tuple[str, str, str]] = set()
    allocation = {candidate.candidate_id: 0 for candidate in candidates}
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.greedy_rank,
            -item.full_net_benefit,
            -item.vhs_score,
            item.cost,
            0 if item.route_type == "DIRECT" else 1,
            item.candidate_id,
        ),
    )
    for candidate in ordered:
        if candidate.route_group in chosen_routes:
            continue
        quantity = min(
            candidate.max_qty,
            source_remaining.get(candidate.source_key, 0),
            target_remaining.get(candidate.target_key, 0),
        )
        if quantity <= 0:
            continue
        _saving, _cost, benefit = candidate.planned_values(quantity)
        if benefit <= _EPS:
            continue
        allocation[candidate.candidate_id] = int(quantity)
        source_remaining[candidate.source_key] -= int(quantity)
        target_remaining[candidate.target_key] -= int(quantity)
        chosen_routes.add(candidate.route_group)
    return [allocation[candidate.candidate_id] for candidate in candidates]


def _selection_order(candidate: _Candidate) -> tuple[Any, ...]:
    return (
        candidate.vhs_rank,
        -candidate.vhs_score,
        -candidate.full_net_benefit,
        candidate.cost,
        0 if candidate.route_type == "DIRECT" else 1,
        candidate.candidate_id,
    )


def _unselected_reasons(
    candidates: Sequence[_Candidate], quantities: Sequence[int], rejected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [candidate for candidate, qty in zip(candidates, quantities) if qty > 0]
    source_used: dict[tuple[str, str], int] = {}
    target_used: dict[tuple[str, str], int] = {}
    selected_groups = {candidate.route_group for candidate in selected}
    for candidate, qty in zip(candidates, quantities):
        source_used[candidate.source_key] = source_used.get(candidate.source_key, 0) + int(qty)
        target_used[candidate.target_key] = target_used.get(candidate.target_key, 0) + int(qty)

    rows = [dict(row) for row in rejected]
    for candidate, qty in zip(candidates, quantities):
        if qty > 0:
            continue
        if candidate.route_group in selected_groups:
            code = "better_route_selected"
        elif source_used.get(candidate.source_key, 0) >= candidate.source_limit:
            code = "shared_source_inventory"
        elif target_used.get(candidate.target_key, 0) >= candidate.target_limit:
            code = "destination_fulfilled"
        else:
            code = "lower_plan_value"
        rows.append({
            "candidate_id": candidate.candidate_id,
            "route_id": candidate.route_id,
            "reason_code": code,
            "reason": REASON_TEXT[code],
        })
    return sorted(rows, key=lambda row: (_text(row.get("route_id")), _text(row.get("candidate_id"))))


def _plan_id(data_signature: str | None, items: Sequence[Mapping[str, Any]]) -> str:
    payload = "|".join(
        f"{item.get('candidate_id')}:{int(item.get('planned_qty') or 0)}"
        for item in items
    )
    digest = hashlib.sha256(
        f"{PLAN_ALGORITHM_VERSION}|{data_signature or 'nosig'}|{payload}".encode("utf-8")
    ).hexdigest()[:12]
    return f"PLAN-{digest.upper()}"


def _build_items(candidates: Sequence[_Candidate], quantities: Sequence[int]) -> list[dict[str, Any]]:
    selected = sorted(
        ((candidate, int(qty)) for candidate, qty in zip(candidates, quantities) if qty > 0),
        key=lambda pair: _selection_order(pair[0]),
    )
    items: list[dict[str, Any]] = []
    for rank, (candidate, quantity) in enumerate(selected, start=1):
        saving, cost, benefit = candidate.planned_values(quantity)
        adjusted = quantity != candidate.recommended_qty
        item = dict(candidate.raw)
        item.update({
            "candidate_id": candidate.candidate_id,
            "plan_rank": rank,
            "recommended_qty": candidate.recommended_qty,
            "planned_qty": quantity,
            "quantity_adjusted": adjusted,
            "recommended_expected_saving": candidate.saving,
            "planned_expected_saving": round(saving, 6),
            "planned_cost": round(cost, 6),
            "planned_net_benefit": round(benefit, 6),
            # Plan items are action records: these aliases intentionally point to
            # the actually allocated values while recommended_qty stays visible.
            "expected_saving": round(saving, 6),
            "estimated_cost": round(cost, 6),
            "move_cost": round(cost, 6),
            "net_benefit": round(benefit, 6),
            "selection_reason_code": "quantity_adjusted" if adjusted else "selected",
            "selection_reason": REASON_TEXT["quantity_adjusted" if adjusted else "selected"],
            "stability": candidate.raw.get("robustness_status"),
        })
        items.append(item)
    return items


def validate_execution_plan(
    plan: Mapping[str, Any],
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    candidate_ledger: Sequence[Mapping[str, Any]] | None = None,
    data_signature: str | None = None,
) -> dict[str, Any]:
    """Recompute every operational constraint independently from solver output."""
    candidates, _rejected, context, catalog = _prepare_candidates(
        recommendations, data, candidate_ledger, data_signature,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    issues: list[str] = []
    source_used: dict[tuple[str, str], int] = {}
    target_used: dict[tuple[str, str], int] = {}
    route_groups: set[tuple[str, str, str]] = set()
    item_ids: set[str] = set()
    safety_violations = 0
    target_violations = 0
    duplicate_violations = 0

    for item in plan.get("items") or []:
        candidate_id = _text(item.get("candidate_id"))
        quantity_value = _num(item.get("planned_qty"))
        quantity = int(quantity_value) if quantity_value is not None and quantity_value.is_integer() else None
        candidate = by_id.get(candidate_id)
        if not candidate_id or candidate_id in item_ids:
            duplicate_violations += 1
            issues.append("중복 계획 항목이 있습니다.")
        item_ids.add(candidate_id)
        if candidate is None:
            issues.append("원본 후보와 연결되지 않는 계획 항목이 있습니다.")
            continue
        if quantity is None or quantity <= 0:
            issues.append("실행 수량은 1개 이상의 정수여야 합니다.")
            continue
        if any(_text(item.get(key)) != expected for key, expected in (
            ("source_id", candidate.source), ("target_id", candidate.target),
            ("product_id", candidate.product), ("route_type", candidate.route_type),
        )):
            issues.append("계획 항목의 점포·상품·경로가 원본 후보와 다릅니다.")
        if _text(item.get("dc_id")) != candidate.dc_id:
            issues.append("계획 항목의 경유 DC가 원본 후보와 다릅니다.")
        if quantity > candidate.max_qty or quantity > candidate.recommended_qty:
            issues.append("실행 수량이 후보 권장량 또는 실제 가용량을 초과했습니다.")
        if not catalog.valid(candidate.source, candidate.target, candidate.product, candidate.route_type, candidate.dc_id):
            issues.append("존재하지 않는 경로 또는 DC가 포함되었습니다.")
        if candidate.route_group in route_groups:
            duplicate_violations += 1
            issues.append("같은 이동의 경로 대안이 중복 선택되었습니다.")
        route_groups.add(candidate.route_group)
        source_used[candidate.source_key] = source_used.get(candidate.source_key, 0) + quantity
        target_used[candidate.target_key] = target_used.get(candidate.target_key, 0) + quantity
        saving, cost, benefit = candidate.planned_values(quantity)
        if benefit <= _EPS:
            issues.append("순효과가 양수가 아닌 이동이 포함되었습니다.")
        numeric_values = (quantity, saving, cost, benefit, item.get("planned_net_benefit"))
        if any(_num(value) is None for value in numeric_values):
            issues.append("NaN 또는 무한대인 계획 값이 있습니다.")
        for key, expected in (
            ("planned_expected_saving", saving),
            ("planned_cost", cost),
            ("planned_net_benefit", benefit),
        ):
            actual = _num(item.get(key))
            if actual is None or abs(actual - expected) > max(1e-5, abs(expected) * 1e-8):
                issues.append("계획 금액 합계가 배정 수량과 일치하지 않습니다.")

    for key, used in source_used.items():
        limit = context.available_to_move(*key)
        stock = context.source_stock(*key)
        floor = context.safety_floor(*key)
        if limit is None or used > limit + _EPS or (
            stock is not None and floor is not None and stock - used < floor - _EPS
        ):
            safety_violations += 1
            issues.append("출발 재고 하한을 넘는 이동이 있습니다.")
    for key, used in target_used.items():
        shortfall = context.planning_shortfall(*key)
        if shortfall is None or used > shortfall + _EPS:
            target_violations += 1
            issues.append("도착 점포 필요 수량을 초과한 이동이 있습니다.")

    unique_issues = list(dict.fromkeys(issues))
    return {
        "valid": not unique_issues,
        "issues": unique_issues,
        "checked_items": len(plan.get("items") or []),
        "source_group_count": len(source_used),
        "target_group_count": len(target_used),
        "safety_stock_violations": safety_violations,
        "destination_overfill_violations": target_violations,
        "duplicate_plan_violations": duplicate_violations,
    }


def build_execution_plan(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
    candidate_ledger: Sequence[Mapping[str, Any]] | None = None,
    data_signature: str | None = None,
    *,
    strategy: str = "optimized",
    timeout_seconds: float = 5.0,
    optimizer: Callable[[Sequence[_Candidate], float], Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Return a validated optimized plan or a validated constrained fallback."""
    started = time.perf_counter()
    candidates, rejected, _context, _catalog = _prepare_candidates(
        recommendations, data, candidate_ledger, data_signature,
    )
    fallback_used = strategy == "greedy"
    method = "constrained_greedy" if fallback_used else "optimized"
    internal_message = ""

    if not candidates:
        status = (
            PLAN_STATUS_UNAVAILABLE
            if rejected and all(row.get("reason_code") == "infeasible" for row in rejected)
            else PLAN_STATUS_EMPTY
        )
        plan = {
            "plan_id": _plan_id(data_signature, []),
            "algorithm_version": PLAN_ALGORITHM_VERSION,
            "data_signature": data_signature,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_candidates": max(len(recommendations or []), len(candidate_ledger or [])),
            "eligible_candidates": 0,
            "selected_candidates": 0,
            "total_transfer_qty": 0,
            "total_cost": 0.0,
            "total_expected_saving": 0.0,
            "total_net_benefit": 0.0,
            "total_shortage_relief": 0,
            "total_excess_relief": 0,
            "plan_status": status,
            "user_message": (
                "현재 조건에서는 실행 가능한 이동이 없습니다."
                if status == PLAN_STATUS_EMPTY else "현재 데이터로 실행계획을 계산할 수 없습니다."
            ),
            "items": [],
            "unselected_candidates": rejected,
            "method": method,
            "fallback_used": fallback_used,
            "optimization_seconds": round(time.perf_counter() - started, 6),
        }
        validation = validate_execution_plan(
            plan, recommendations, data, candidate_ledger, data_signature,
        )
        plan["validation"] = validation
        plan["validation_seconds"] = 0.0
        return plan

    solve_started = time.perf_counter()
    if strategy == "greedy":
        quantities = _solve_greedy(candidates)
    else:
        try:
            quantities = list((optimizer or _solve_milp)(candidates, timeout_seconds))
            if len(quantities) != len(candidates):
                raise RuntimeError("optimizer returned a wrong-sized allocation")
        except Exception as exc:  # safe product fallback; detail remains internal
            quantities = _solve_greedy(candidates)
            fallback_used = True
            method = "constrained_greedy"
            internal_message = f"{type(exc).__name__}: {exc}"
    optimization_seconds = time.perf_counter() - solve_started

    items = _build_items(candidates, quantities)
    unselected = _unselected_reasons(candidates, quantities, rejected)
    adjusted = sum(1 for item in items if item.get("quantity_adjusted"))
    total_cost = sum(float(item.get("planned_cost") or 0) for item in items)
    total_saving = sum(float(item.get("planned_expected_saving") or 0) for item in items)
    total_net = sum(float(item.get("planned_net_benefit") or 0) for item in items)
    total_qty = sum(int(item.get("planned_qty") or 0) for item in items)
    selected_count = len(items)
    status = PLAN_STATUS_EMPTY
    if selected_count:
        status = PLAN_STATUS_PARTIAL if unselected or adjusted else PLAN_STATUS_READY
    message = (
        f"오늘 실행 가능한 이동 {selected_count}건을 찾았습니다."
        if selected_count else "현재 조건에서는 실행 가능한 이동이 없습니다."
    )
    if selected_count and unselected:
        message += f" 추천 후보 중 {len(unselected)}건은 공유 재고·필요 수량을 함께 고려해 제외했습니다."

    plan = {
        "plan_id": _plan_id(data_signature, items),
        "algorithm_version": PLAN_ALGORITHM_VERSION,
        "data_signature": data_signature,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_candidates": max(len(recommendations or []), len(candidate_ledger or [])),
        "eligible_candidates": len(candidates),
        "selected_candidates": selected_count,
        "adjusted_candidates": adjusted,
        "total_transfer_qty": total_qty,
        "total_cost": round(total_cost, 6),
        "total_expected_saving": round(total_saving, 6),
        "total_net_benefit": round(total_net, 6),
        "total_shortage_relief": total_qty,
        "total_excess_relief": total_qty,
        "plan_status": status,
        "user_message": message,
        "items": items,
        "unselected_candidates": unselected,
        "method": method,
        "fallback_used": fallback_used,
        "optimizer_message": internal_message,
        "optimization_seconds": round(optimization_seconds, 6),
    }
    for item in items:
        item["plan_id"] = plan["plan_id"]

    validation_started = time.perf_counter()
    validation = validate_execution_plan(
        plan, recommendations, data, candidate_ledger, data_signature,
    )
    plan["validation_seconds"] = round(time.perf_counter() - validation_started, 6)
    plan["validation"] = validation

    # Never expose an unvalidated solver result. Retry once with the independent
    # constrained allocator; if that also fails, return 계산 불가 with no actions.
    if not validation["valid"] and strategy != "greedy" and not fallback_used:
        return build_execution_plan(
            recommendations,
            data,
            candidate_ledger,
            data_signature,
            strategy="greedy",
            timeout_seconds=timeout_seconds,
        )
    if not validation["valid"]:
        plan.update({
            "plan_status": PLAN_STATUS_UNAVAILABLE,
            "user_message": "현재 데이터로 실행계획을 계산할 수 없습니다.",
            "items": [],
            "selected_candidates": 0,
            "total_transfer_qty": 0,
            "total_cost": 0.0,
            "total_expected_saving": 0.0,
            "total_net_benefit": 0.0,
            "total_shortage_relief": 0,
            "total_excess_relief": 0,
        })
    plan["total_seconds"] = round(time.perf_counter() - started, 6)
    return plan


def independent_candidate_metrics(
    recommendations: Sequence[Mapping[str, Any]], data: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Show conflicts hidden by summing individually feasible recommendations."""
    context = build_inventory_context(data)
    source_used: dict[tuple[str, str], float] = {}
    target_used: dict[tuple[str, str], float] = {}
    total_qty = total_cost = total_saving = total_net = 0.0
    for item in recommendations or []:
        quantity = _num(item.get("recommended_qty")) or 0.0
        key_source = (_text(item.get("source_id")), _text(item.get("product_id")))
        key_target = (_text(item.get("target_id")), _text(item.get("product_id")))
        source_used[key_source] = source_used.get(key_source, 0.0) + quantity
        target_used[key_target] = target_used.get(key_target, 0.0) + quantity
        cost = _num(item.get("estimated_cost")) or _num(item.get("move_cost")) or 0.0
        saving = _num(item.get("expected_saving")) or 0.0
        total_qty += quantity
        total_cost += cost
        total_saving += saving
        total_net += (_num(item.get("net_benefit")) if _num(item.get("net_benefit")) is not None else saving - cost)
    source_violations = sum(
        1 for key, used in source_used.items()
        if context.available_to_move(*key) is None or used > float(context.available_to_move(*key)) + _EPS
    )
    target_violations = 0
    for key, used in target_used.items():
        shortfall = context.planning_shortfall(*key)
        if shortfall is None or used > float(shortfall) + _EPS:
            target_violations += 1
    return {
        "selected_candidates": len(recommendations or []),
        "total_transfer_qty": round(total_qty, 6),
        "total_cost": round(total_cost, 6),
        "total_expected_saving": round(total_saving, 6),
        "total_net_benefit": round(total_net, 6),
        "total_shortage_relief": round(total_qty, 6),
        "total_excess_relief": round(total_qty, 6),
        "safety_stock_violations": source_violations,
        "destination_overfill_violations": target_violations,
    }


def compare_execution_plans(
    optimized: Mapping[str, Any],
    greedy: Mapping[str, Any],
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Comparable, non-manipulated metrics for independent, Greedy, and VHS plan."""
    independent = independent_candidate_metrics(recommendations, data)

    def metrics(plan: Mapping[str, Any]) -> dict[str, Any]:
        validation = plan.get("validation") or {}
        return {
            key: plan.get(key)
            for key in (
                "selected_candidates", "total_transfer_qty", "total_cost",
                "total_expected_saving", "total_net_benefit",
                "total_shortage_relief", "total_excess_relief",
                "optimization_seconds", "validation_seconds", "total_seconds",
            )
        } | {
            "safety_stock_violations": validation.get("safety_stock_violations", 0),
            "destination_overfill_violations": validation.get("destination_overfill_violations", 0),
        }

    return {
        "independent_candidates": independent,
        "constrained_greedy": metrics(greedy),
        "vhs_optimized_plan": metrics(optimized),
    }


def planned_recommendations(pipeline: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The sole action list shared by home, recommendations, and route detail."""
    if not isinstance(pipeline, Mapping):
        return []
    plan = pipeline.get("execution_plan")
    if not isinstance(plan, Mapping) or not (plan.get("validation") or {}).get("valid"):
        return []
    return [dict(item) for item in (plan.get("items") or [])]


__all__ = [
    "PLAN_ALGORITHM_VERSION",
    "PLAN_STATUS_READY",
    "PLAN_STATUS_PARTIAL",
    "PLAN_STATUS_EMPTY",
    "PLAN_STATUS_UNAVAILABLE",
    "REASON_TEXT",
    "build_execution_plan",
    "validate_execution_plan",
    "independent_candidate_metrics",
    "compare_execution_plans",
    "planned_recommendations",
]
