"""Operational Top-N shared-feasibility selection, computed next to Varo Final (it never replaces it).

Varo Final orders candidates one at a time (recommended_qty desc -> move cost asc -> vhs_rank asc -> route_id), and
the production Top-5 is a slice of that order.  Two moves in the slice can therefore draw on the same source stock or
fill the same target need twice.  This module selects up to N moves that are executable *together*:

* shared caps: per (source, product) the movable surplus (A) and the on-hand stock (C); per (target, product) the
  need (B); per candidate the route/vehicle capacity in quantity units (E); per DC the DC capacity; at most one move per
  physical lane (D); at most N moves (F).
* every cap carries a provenance class (the Seller Loss taxonomy).  STRICT_ACTUAL uses only DIRECT_REAL, DERIVED_REAL,
  USER_INPUT and DERIVED_FROM_USER_INPUT caps, and a candidate whose required cap is missing or only a proxy gets
  INSUFFICIENT_CAP_DATA instead of being called executable.  BENCHMARK_PROXY applies every available cap, PROXY
  included, and says so in the plan.
* the Varo Final key is kept, with the executable quantity under the remaining caps as key 1.  When no shared cap
  binds, the order and the selection are exactly the Varo Final order.  With partial allocation off, the result equals
  the offline ``ordered_feasible_selection`` over the same caps.
* a candidate is split only when that is safe: the candidate allows it, a declared lot size / minimum is respected,
  integer quantities stay integers, and the move cost of the smaller quantity follows from its cost basis.  Otherwise
  it is not split.  A cost is never scaled by quantity unless the basis says PER_UNIT.
* executability and cost comparability are reported apart: the ALLOW_COST_UNKNOWN policy (the pipeline's
  ``quantity_feasibility_view``) allows safe splits whose cost stays NULL, so "a smaller move fits the caps" is visible
  even when its cost cannot be compared.

Nothing here changes recommendations, ranks, Top-5, KPIs or actions.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from services.optimality_gap_service import (
    CONSTRAINT_VERSION,
    DEFAULT_MAX_ROUTES,
    _explicitly_false,
    _first_number,
    _identifier,
    _inventory_map,
    _number,
    build_constraint_context,
)
from services.seller_loss_engine import PROVENANCE_CLASSES, _config, declared_provenance, demand_provenance

SELECTION_VERSION = "shared-feasibility-topn-v0.1"

STRICT_ACTUAL, BENCHMARK_PROXY = "STRICT_ACTUAL", "BENCHMARK_PROXY"
MODES = (STRICT_ACTUAL, BENCHMARK_PROXY)
STRICT_ACCEPTED_PROVENANCE = frozenset({"DIRECT_REAL", "DERIVED_REAL", "USER_INPUT", "DERIVED_FROM_USER_INPUT"})
PARTIAL_NONE, PARTIAL_IF_SAFE, PARTIAL_COST_UNKNOWN = "NONE", "ALLOW_IF_SAFE", "ALLOW_COST_UNKNOWN"
PARTIAL_POLICIES = (PARTIAL_NONE, PARTIAL_IF_SAFE, PARTIAL_COST_UNKNOWN)
NO_COST_COMPARABLE_SELECTION = "NO_COST_COMPARABLE_SELECTION"

SOURCE_SURPLUS, SOURCE_STOCK, TARGET_NEED = "SOURCE_SURPLUS", "SOURCE_AVAILABLE_STOCK", "TARGET_NEED"
ROUTE_CAPACITY, DC_CAPACITY = "ROUTE_CAPACITY", "DC_CAPACITY"
CAP_KINDS = (SOURCE_SURPLUS, SOURCE_STOCK, TARGET_NEED, ROUTE_CAPACITY, DC_CAPACITY)
SOURCE_KINDS = (SOURCE_SURPLUS, SOURCE_STOCK)

SELECTED, PARTIALLY_SELECTED = "SELECTED", "PARTIALLY_SELECTED"
REJECTED_SOURCE_CAP, REJECTED_TARGET_CAP = "REJECTED_SOURCE_CAP", "REJECTED_TARGET_CAP"
REJECTED_ROUTE_CAP, REJECTED_DC_CAP = "REJECTED_ROUTE_CAP", "REJECTED_DC_CAP"
REJECTED_DUPLICATE, REJECTED_SELECTION_LIMIT = "REJECTED_DUPLICATE", "REJECTED_SELECTION_LIMIT"
REJECTED_INFEASIBLE_ROUTE, REJECTED_INVALID_INPUT = "REJECTED_INFEASIBLE_ROUTE", "REJECTED_INVALID_INPUT"
INSUFFICIENT_CAP_DATA, UNIT_MISMATCH, UNKNOWN_UNIT = "INSUFFICIENT_CAP_DATA", "UNIT_MISMATCH", "UNKNOWN_UNIT"
COST_NOT_COMPARABLE = "COST_NOT_COMPARABLE"
CAP_REJECTION = {
    SOURCE_SURPLUS: REJECTED_SOURCE_CAP, SOURCE_STOCK: REJECTED_SOURCE_CAP, TARGET_NEED: REJECTED_TARGET_CAP,
    ROUTE_CAPACITY: REJECTED_ROUTE_CAP, DC_CAPACITY: REJECTED_DC_CAP,
}

FIXED_PER_TRIP, PER_UNIT, QUANTITY_SPECIFIC = "FIXED_PER_TRIP", "PER_UNIT", "QUANTITY_SPECIFIC"
COST_BASES = (FIXED_PER_TRIP, PER_UNIT, QUANTITY_SPECIFIC)

# Field lists are the ones the existing code already reads, so every path sees the same columns.
QTY_FIELDS = ("recommended_qty", "suggested_qty", "transfer_qty", "quantity")          # Varo Final key 1
COST_FIELDS = ("move_cost", "estimated_cost", "transport_cost", "transfer_cost")       # Varo Final key 2
ROUTE_CAPACITY_FIELDS = ("route_capacity_qty", "vehicle_capacity_qty", "max_load_qty")  # shared-feasibility-v1
MOVABLE_FIELDS = ("available_transfer_stock", "transferable_stock", "source_surplus", "surplus_qty", "excess_stock")
STOCK_FIELDS = ("stock_qty", "current_stock", "quantity", "stock")
FEASIBILITY_FLAGS = ("feasible", "is_feasible", "route_feasible", "cutline_passed", "time_window_status")
UNIT_FIELDS = ("quantity_unit", "qty_unit", "unit", "uom")
LOT_FIELDS = ("lot_size_qty", "case_pack_qty", "pack_size_qty")
MINIMUM_FIELDS = ("min_transfer_qty", "min_shipment_qty")
_UNIT_ALIASES = {
    "EA": "EA", "PCS": "EA", "PC": "EA", "UNIT": "EA", "개": "EA", "BOX": "BOX", "CASE": "BOX", "CS": "BOX",
    "박스": "BOX", "KG": "KG", "TON": "TON", "톤": "TON",
}
EPS = 1e-9
INF = float("inf")

CostRecompute = Callable[[Mapping[str, Any], float], "float | None"]


@dataclass(frozen=True)
class Cap:
    """One quantity limit on a group. ``value=None`` documents a missing cap (provenance MISSING)."""

    kind: str
    key: tuple[str, ...]
    value: float | None
    provenance: str
    basis: str
    unit: str | None = None
    conflict: bool = False

    @property
    def strict_accepted(self) -> bool:
        return self.value is not None and self.provenance in STRICT_ACCEPTED_PROVENANCE

    def as_row(self) -> dict[str, Any]:
        return {
            "cap_kind": self.kind, "cap_key": "/".join(self.key),
            "cap_value": None if self.value is None else round(self.value, 6),
            "provenance": self.provenance, "strict_accepted": self.strict_accepted, "basis": self.basis,
            "unit": self.unit or "UNDECLARED", "conflicting_values": self.conflict,
        }


CapTable = dict[tuple[str, tuple[str, ...]], Cap]


def add_cap(caps: CapTable, cap: Cap) -> None:
    """Register a cap. A missing entry never hides a value; two different values keep the smaller one, flagged."""
    if cap.provenance not in PROVENANCE_CLASSES:
        raise ValueError(f"unknown provenance {cap.provenance!r}")
    if cap.value is not None:
        cap = replace(cap, value=max(0.0, float(cap.value)))
    slot = (cap.kind, cap.key)
    current = caps.get(slot)
    if current is None or (current.value is None and cap.value is not None):
        caps[slot] = cap
    elif cap.value is not None and abs(float(current.value) - cap.value) > EPS:
        caps[slot] = replace(current if current.value <= cap.value else cap, conflict=True)


def normalize_unit(value: Any) -> str | None:
    text = _identifier(value)
    if not text or text.lower() in {"nan", "none", "unknown", "undeclared"}:
        return None
    return _UNIT_ALIASES.get(text.upper(), _UNIT_ALIASES.get(text, text.upper()))


def candidate_identity(record: Mapping[str, Any], index: int) -> str:
    return _identifier(record.get("route_id")) or _identifier(record.get("recommendation_id")) or f"candidate-{index + 1}"


def _ids(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    product = _identifier(record.get("product_id") or record.get("item_id"))
    source = _identifier(record.get("source_id") or record.get("from_store_id"))
    target = _identifier(record.get("target_id") or record.get("to_store_id"))
    route_type = _identifier(record.get("route_type")).upper() or "DIRECT"
    dc = _identifier(record.get("dc_id")) if route_type == "VIA_DC" else ""
    return product, source, target, route_type, dc


def cap_slots(record: Mapping[str, Any], index: int) -> dict[str, tuple[str, tuple[str, ...]]]:
    """The cap groups a candidate draws on."""
    product, source, target, _, dc = _ids(record)
    slots = {
        SOURCE_SURPLUS: (SOURCE_SURPLUS, (source, product)), SOURCE_STOCK: (SOURCE_STOCK, (source, product)),
        TARGET_NEED: (TARGET_NEED, (target, product)), ROUTE_CAPACITY: (ROUTE_CAPACITY, (candidate_identity(record, index),)),
    }
    if dc:
        slots[DC_CAPACITY] = (DC_CAPACITY, (dc,))
    return slots


# ------------------------------------------------------------------------------------------------ cap resolution


def caps_from_columns(records: Sequence[Mapping[str, Any]], spec: Mapping[str, tuple[Sequence[str], str, str]]) -> CapTable:
    """Caps carried as candidate columns (e.g. the offline Suhyup benchmark).

    ``spec`` maps a cap kind to (columns, default provenance, basis). A row's ``<column>_provenance`` label overrides
    the default (``actual`` -> DIRECT_REAL, anything containing ``proxy`` -> PROXY).
    """
    caps: CapTable = {}
    for index, record in enumerate(records):
        slots = cap_slots(record, index)
        for kind, (columns, default_provenance, basis) in spec.items():
            if kind not in slots:
                continue
            value, field = _first_number(record, tuple(columns))
            key = slots[kind][1]
            if value is None:
                add_cap(caps, Cap(kind, key, None, "MISSING", f"{'|'.join(columns)} absent"))
                continue
            provenance, _ = declared_provenance(record, str(field), default_provenance)
            unit = normalize_unit(record.get(f"{field}_unit"))
            add_cap(caps, Cap(kind, key, value, provenance, f"{field}: {basis}", unit))
    return caps


def pipeline_caps(records: Sequence[Mapping[str, Any]], uploaded_data: Mapping[str, Any] | None) -> CapTable:
    """shared-feasibility-v1 caps (``optimality_gap_service.build_constraint_context``) with provenance, plus on-hand stock.

    Same cap values as the in-app optimality gap. Provenance comes from ``<column>_provenance`` labels and the
    Seller Loss demand rule: a need computed from a demand proxy is PROXY.
    """
    data = uploaded_data or {}
    default_unit = normalize_unit(_config(data.get("config")).get("quantity_unit"))
    inventory = _inventory_map(data)
    shaped: list[dict[str, Any]] = []
    for record in records:
        product, source, target, route_type, dc = _ids(record)
        shaped.append({**record, "_source_key": (source, product), "_target_key": (target, product), "_dc_id": dc,
                       "_duplicate_key": (product, source, target, route_type, dc),
                       "_route_capacity": _first_number(record, ROUTE_CAPACITY_FIELDS)[0]})
    context = build_constraint_context(shaped, data, {"max_routes": None})
    caps: CapTable = {}

    def unit_of(row: Mapping[str, Any]) -> str | None:
        return normalize_unit(_first_text(row, UNIT_FIELDS)) or default_unit

    for key, value in context["source_caps"].items():
        basis = str(context["source_basis"].get(key) or "")
        field = basis.split(" ")[0]
        row = inventory.get(key, {})
        if field in MOVABLE_FIELDS:
            origin = next((item for item in shaped if item["_source_key"] == key and _number(item.get(field)) is not None), None)
            provenance, _ = declared_provenance(origin if origin is not None else row, field, "USER_INPUT")
            where = "candidate" if origin is not None else "inventory"
            add_cap(caps, Cap(SOURCE_SURPLUS, key, value, provenance, f"{where}.{field} ({CONSTRAINT_VERSION})", unit_of(row)))
            stock, stock_field = _first_number(row, STOCK_FIELDS)
            if stock is not None:
                add_cap(caps, Cap(SOURCE_STOCK, key, stock, declared_provenance(row, str(stock_field), "USER_INPUT")[0],
                                  f"inventory.{stock_field} (on-hand stock)", unit_of(row)))
        else:
            add_cap(caps, Cap(SOURCE_STOCK, key, value, declared_provenance(row, field, "USER_INPUT")[0],
                              f"inventory.{field} (on-hand stock; {CONSTRAINT_VERSION} fallback)", unit_of(row)))
            add_cap(caps, Cap(SOURCE_SURPLUS, key, None, "MISSING",
                              "no movable-stock column; outflow bounded by on-hand stock only"))
    for key, value in context["target_caps"].items():
        basis = str(context["target_basis"].get(key) or "")
        row = inventory.get(key, {})
        if basis.startswith("max("):
            demand = demand_provenance(row)
            _, stock_field = _first_number(row, STOCK_FIELDS)
            stock_provenance = declared_provenance(row, str(stock_field), "USER_INPUT")[0]
            provenance = "PROXY" if demand == "PROXY" else (
                "DERIVED_REAL" if demand == "DERIVED_REAL" and stock_provenance == "DIRECT_REAL" else "DERIVED_FROM_USER_INPUT")
            text = f"inventory {basis} (7-day demand horizon of {CONSTRAINT_VERSION})"
        else:
            origin = row if _number(row.get(basis)) is not None else next(
                (item for item in shaped if item["_target_key"] == key and _number(item.get(basis)) is not None), {})
            provenance, _ = declared_provenance(origin, basis, "USER_INPUT")
            text = f"{basis} (explicit shortage)"
        add_cap(caps, Cap(TARGET_NEED, key, value, provenance, text, unit_of(row)))
    for dc_id, value in context["dc_caps"].items():
        add_cap(caps, Cap(DC_CAPACITY, (dc_id,), value, "USER_INPUT",
                          f"{context['dc_basis'].get(dc_id)} (DC node capacity used as a throughput limit by {CONSTRAINT_VERSION})"))
    for index, record in enumerate(records):
        slots = cap_slots(record, index)
        if (SOURCE_STOCK, slots[SOURCE_STOCK][1]) not in caps:
            add_cap(caps, Cap(SOURCE_STOCK, slots[SOURCE_STOCK][1], None, "MISSING", "no inventory stock row"))
            add_cap(caps, Cap(SOURCE_SURPLUS, slots[SOURCE_SURPLUS][1], None, "MISSING", "no movable-stock column"))
        if (TARGET_NEED, slots[TARGET_NEED][1]) not in caps:
            add_cap(caps, Cap(TARGET_NEED, slots[TARGET_NEED][1], None, "MISSING", "stock and 7-day demand not both available"))
        capacity, field = _first_number(record, ROUTE_CAPACITY_FIELDS)
        if capacity is None:
            add_cap(caps, Cap(ROUTE_CAPACITY, slots[ROUTE_CAPACITY][1], None, "MISSING",
                              "no route/vehicle capacity in quantity units on the candidate"))
        else:
            add_cap(caps, Cap(ROUTE_CAPACITY, slots[ROUTE_CAPACITY][1], capacity,
                              declared_provenance(record, str(field), "USER_INPUT")[0], f"candidate.{field}", unit_of(record)))
    return caps


def _first_text(record: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        text = _identifier(record.get(name))
        if text and text.lower() not in {"nan", "none"}:
            return text
    return ""


# ------------------------------------------------------------------------------------------------ selection


@dataclass
class _Candidate:
    index: int
    record: dict[str, Any]
    candidate_id: str
    route_id: str
    product: str
    source: str
    target: str
    route_type: str
    dc: str
    qty: float | None
    cost: float | None
    cost_basis: str
    cost_basis_declared: bool
    currency: str | None
    unit: str | None
    lot: float | None
    minimum: float | None
    allow_partial: bool
    vhs_rank: float
    original_rank: float | None
    slots: dict[str, tuple[str, tuple[str, ...]]]
    status: str | None = None
    reason: str = ""
    duplicate_of: str | None = None
    unit_status: str = "UNDECLARED"

    @property
    def lane(self) -> tuple[str, ...]:
        return self.product, self.source, self.target, self.route_type, self.dc

    @property
    def tiebreak(self) -> str:
        return json.dumps([*self.lane, self.qty, self.cost], default=str)


@dataclass
class _Evaluation:
    allocated: float
    executable: float
    cost: float | None
    cost_status: str
    partial: bool
    binding: list[Cap]
    blocked: str | None = None
    blocked_status: str | None = None


def _prepare(records: Sequence[Mapping[str, Any]], default_unit: str | None) -> list[_Candidate]:
    prepared: list[_Candidate] = []
    for index, raw in enumerate(records):
        record = dict(raw)
        product, source, target, route_type, dc = _ids(record)
        qty, _ = _first_number(record, QTY_FIELDS)
        cost, _ = _first_number(record, COST_FIELDS)
        basis = _first_text(record, ("transfer_cost_basis", "cost_basis")).upper()
        lot, _ = _first_number(record, LOT_FIELDS)
        minimum, _ = _first_number(record, MINIMUM_FIELDS)
        rank, _ = _first_number(record, ("varo_final_rank",))
        vhs, _ = _first_number(record, ("vhs_rank",))
        candidate = _Candidate(
            index=index, record=record, candidate_id=candidate_identity(record, index),
            route_id=_identifier(record.get("route_id")), product=product, source=source, target=target,
            route_type=route_type, dc=dc, qty=qty, cost=cost, cost_basis=basis or QUANTITY_SPECIFIC,
            cost_basis_declared=bool(basis), currency=_first_text(record, ("currency",)).upper() or None,
            unit=normalize_unit(_first_text(record, UNIT_FIELDS)) or default_unit,
            lot=lot if lot is not None and lot > 0 else None,
            minimum=minimum if minimum is not None and minimum > 0 else None,
            allow_partial=not ("allow_partial" in record and _explicitly_false(record.get("allow_partial"))),
            vhs_rank=vhs if vhs is not None else INF, original_rank=rank, slots=cap_slots(record, index),
        )
        if not product or not source or not target:
            candidate.status, candidate.reason = REJECTED_INVALID_INPUT, "product/source/target id missing"
        elif source == target:
            candidate.status, candidate.reason = REJECTED_INVALID_INPUT, "source equals target"
        elif qty is None or qty <= 0:
            candidate.status, candidate.reason = REJECTED_INVALID_INPUT, "recommended quantity missing or <= 0"
        elif cost is not None and cost < 0:
            candidate.status, candidate.reason = REJECTED_INVALID_INPUT, "negative move cost"
        else:
            for field in FEASIBILITY_FLAGS:
                if field in record and _explicitly_false(record.get(field)):
                    candidate.status = REJECTED_INFEASIBLE_ROUTE
                    candidate.reason = f"existing feasibility field {field}={record.get(field)!r} not passed"
                    break
        prepared.append(candidate)
    if any(item.original_rank is None for item in prepared):
        from services.vhs_score_engine import _rank_varo_operational  # same key as the production order

        ranks = _rank_varo_operational(pd.DataFrame([item.record for item in prepared]))
        for item, rank in zip(prepared, ranks.tolist()):
            item.original_rank = float(rank)
    return prepared


def _applicable(candidate: _Candidate, caps: CapTable) -> dict[str, Cap]:
    present: dict[str, Cap] = {}
    for kind, slot in candidate.slots.items():
        cap = caps.get(slot)
        if cap is not None and cap.value is not None:
            present[kind] = cap
    return present


def _mode_gate(candidate: _Candidate, present: Mapping[str, Cap], mode: str) -> tuple[dict[str, Cap], list[str]]:
    """Caps applied in this mode, and the required dimensions left unchecked (BENCHMARK only). Sets STRICT failures."""
    unchecked = [label for label, kinds in (("SOURCE", SOURCE_KINDS), ("TARGET_NEED", (TARGET_NEED,)))
                 if not any(kind in present for kind in kinds)]
    if mode == BENCHMARK_PROXY:
        return dict(present), unchecked
    unvalidated = [f"{kind}={cap.provenance}" for kind, cap in present.items() if not cap.strict_accepted]
    missing = [label for label, kinds in (("SOURCE", SOURCE_KINDS), ("TARGET_NEED", (TARGET_NEED,)))
               if not any(kind in present and present[kind].strict_accepted for kind in kinds)]
    if unvalidated or missing:
        parts = []
        if missing:
            parts.append("no validated cap for " + ", ".join(missing))
        if unvalidated:
            parts.append("cap provenance not accepted in STRICT_ACTUAL: " + ", ".join(unvalidated))
        candidate.status, candidate.reason = INSUFFICIENT_CAP_DATA, "; ".join(parts)
    return dict(present), []


def _unit_gate(candidate: _Candidate, applied: Mapping[str, Cap]) -> None:
    declared = {kind: cap.unit for kind, cap in applied.items()}
    if not declared:
        candidate.unit_status = "DECLARED" if candidate.unit else "UNDECLARED"
        return
    if candidate.unit is None and not any(declared.values()):
        candidate.unit_status = "UNDECLARED_SAME_SOURCE"
        return
    if candidate.unit is None or not all(declared.values()):
        candidate.status, candidate.unit_status = UNKNOWN_UNIT, UNKNOWN_UNIT
        candidate.reason = (f"quantity unit declared on one side only (candidate={candidate.unit or 'UNDECLARED'}, caps="
                            + ", ".join(f"{kind}={unit or 'UNDECLARED'}" for kind, unit in declared.items()) + ")")
        return
    mismatched = {kind: unit for kind, unit in declared.items() if unit != candidate.unit}
    if mismatched:
        candidate.status, candidate.unit_status = UNIT_MISMATCH, UNIT_MISMATCH
        candidate.reason = (f"candidate unit {candidate.unit} vs cap unit "
                            + ", ".join(f"{kind}={unit}" for kind, unit in mismatched.items()) + "; no conversion basis")
        return
    candidate.unit_status = "MATCHED"


def _partial_quantity(candidate: _Candidate, executable: float, policy: str) -> tuple[float, str | None]:
    if policy == PARTIAL_NONE:
        return 0.0, "partial allocation disabled (policy NONE)"
    if not candidate.allow_partial:
        return 0.0, "candidate forbids partial allocation (allow_partial=False)"
    quantity = executable
    if candidate.lot:
        quantity = math.floor(quantity / candidate.lot + EPS) * candidate.lot
    elif abs(float(candidate.qty) - round(float(candidate.qty))) <= EPS:
        quantity = float(math.floor(quantity + EPS))
    if candidate.minimum and quantity < candidate.minimum - EPS:
        return 0.0, f"remaining {round(executable, 6)} below the declared minimum transfer qty {candidate.minimum}"
    if quantity <= EPS:
        return 0.0, f"remaining {round(executable, 6)} holds no whole {'lot of ' + str(candidate.lot) if candidate.lot else 'unit'}"
    return quantity, None


def _partial_cost(candidate: _Candidate, quantity: float, recompute: CostRecompute | None) -> tuple[float | None, str]:
    if candidate.cost is None:
        return None, "COST_MISSING"
    if candidate.cost_basis == FIXED_PER_TRIP:
        return candidate.cost, "FIXED_PER_TRIP_UNCHANGED"
    if candidate.cost_basis == PER_UNIT:
        return candidate.cost / float(candidate.qty) * quantity, "PER_UNIT_SCALED"
    if candidate.cost_basis == QUANTITY_SPECIFIC and recompute is not None:
        value = recompute(candidate.record, quantity)
        if value is not None and math.isfinite(float(value)) and float(value) >= 0:
            return float(value), "QUANTITY_SPECIFIC_RECOMPUTED"
    return None, COST_NOT_COMPARABLE


def _evaluate(candidate: _Candidate, applied: Mapping[str, Cap], used: Mapping[tuple, float], policy: str,
              recompute: CostRecompute | None) -> _Evaluation:
    requested = float(candidate.qty)
    limits = [(max(0.0, float(cap.value) - used.get((cap.kind, cap.key), 0.0)), cap) for cap in applied.values()]
    executable = min([requested, *(remaining for remaining, _ in limits)])
    binding = [cap for remaining, cap in limits if remaining < requested - EPS and abs(remaining - executable) <= EPS]
    if executable >= requested - EPS:
        return _Evaluation(requested, requested, candidate.cost, "ORIGINAL" if candidate.cost is not None else "COST_MISSING",
                           False, [])
    if executable <= EPS:
        return _Evaluation(0.0, 0.0, None, "", False, binding)
    quantity, blocked = _partial_quantity(candidate, executable, policy)
    if blocked:
        return _Evaluation(0.0, executable, None, "", False, binding, blocked)
    cost, cost_status = _partial_cost(candidate, quantity, recompute)
    if cost_status == COST_NOT_COMPARABLE and policy == PARTIAL_COST_UNKNOWN:
        # Quantity-feasibility view: the split is executable under the caps; its cost stays unknown (never scaled).
        return _Evaluation(quantity, quantity, None, COST_NOT_COMPARABLE, True, binding)
    if cost_status == COST_NOT_COMPARABLE:
        reason = (f"move cost for {round(quantity, 6)} of {round(requested, 6)} cannot be derived "
                  f"(cost_basis={candidate.cost_basis}{'' if candidate.cost_basis_declared else ' assumed: cost stated for recommended_qty'})")
        return _Evaluation(0.0, quantity, None, cost_status, False, binding, reason, COST_NOT_COMPARABLE)
    return _Evaluation(quantity, quantity, cost, cost_status, True, binding)


def _order_key(candidate: _Candidate, evaluation: _Evaluation) -> tuple:
    """Varo Final key with the executable quantity as key 1 (see ``vhs_score_engine._rank_varo_operational``)."""
    cost = evaluation.cost if evaluation.cost is not None else INF
    rank = candidate.original_rank if candidate.original_rank is not None else INF
    return (-evaluation.allocated, cost, candidate.vhs_rank, candidate.route_id, rank, candidate.candidate_id,
            candidate.tiebreak)


def _base_key(candidate: _Candidate) -> tuple:
    rank = candidate.original_rank if candidate.original_rank is not None else INF
    return rank, candidate.route_id, candidate.candidate_id, candidate.tiebreak


def _source_remaining(applied: Mapping[str, Cap], used: Mapping[tuple, float]) -> float | None:
    values = [float(cap.value) - used.get((cap.kind, cap.key), 0.0) for kind, cap in applied.items() if kind in SOURCE_KINDS]
    return round(min(values), 6) if values else None


def _target_remaining(applied: Mapping[str, Cap], used: Mapping[tuple, float]) -> float | None:
    cap = applied.get(TARGET_NEED)
    return round(float(cap.value) - used.get((cap.kind, cap.key), 0.0), 6) if cap else None


def select_shared_feasible(
    records: Sequence[Mapping[str, Any]],
    caps: CapTable,
    *,
    mode: str = STRICT_ACTUAL,
    max_routes: int | None = DEFAULT_MAX_ROUTES,
    partial_policy: str = PARTIAL_IF_SAFE,
    cost_recompute: CostRecompute | None = None,
    legacy_ids: Sequence[str] | None = None,
    default_unit: str | None = None,
) -> dict[str, Any]:
    """Select up to ``max_routes`` jointly executable moves in Varo Final order. Inputs are never mutated."""
    if mode not in MODES:
        raise ValueError(f"unknown feasibility mode {mode!r}")
    if partial_policy not in PARTIAL_POLICIES:
        raise ValueError(f"unknown partial policy {partial_policy!r}")
    prepared = _prepare(records, default_unit)
    applied_by_id: dict[int, dict[str, Cap]] = {}
    unchecked_by_id: dict[int, list[str]] = {}
    for candidate in prepared:
        present = _applicable(candidate, caps)
        applied, unchecked = _mode_gate(candidate, present, mode) if candidate.status is None else (present, [])
        applied_by_id[candidate.index], unchecked_by_id[candidate.index] = applied, unchecked
        if candidate.status is None:
            _unit_gate(candidate, applied)

    first_by_id: dict[str, _Candidate] = {}
    for candidate in sorted((item for item in prepared if item.status is None), key=_base_key):
        if candidate.candidate_id in first_by_id:
            candidate.status, candidate.duplicate_of = REJECTED_DUPLICATE, first_by_id[candidate.candidate_id].candidate_id
            candidate.reason = "same candidate id appears more than once"
        else:
            first_by_id[candidate.candidate_id] = candidate
    pool = [item for item in prepared if item.status is None]
    used: defaultdict[tuple, float] = defaultdict(float)
    consumers: defaultdict[tuple, list[str]] = defaultdict(list)
    lanes: dict[tuple[str, ...], str] = {}
    chosen: list[tuple[_Candidate, _Evaluation, float | None, float | None]] = []
    limit = None if max_routes is None else max(0, int(max_routes))
    while pool and (limit is None or len(chosen) < limit):
        best: tuple[tuple, _Candidate, _Evaluation] | None = None
        for candidate in pool:
            if candidate.lane in lanes:
                continue
            evaluation = _evaluate(candidate, applied_by_id[candidate.index], used, partial_policy, cost_recompute)
            if evaluation.allocated <= EPS:
                continue
            key = _order_key(candidate, evaluation)
            if best is None or key < best[0]:
                best = (key, candidate, evaluation)
        if best is None:
            break
        _, candidate, evaluation = best
        applied = applied_by_id[candidate.index]
        for cap in applied.values():
            used[(cap.kind, cap.key)] += evaluation.allocated
            consumers[(cap.kind, cap.key)].append(candidate.candidate_id)
        lanes[candidate.lane] = candidate.candidate_id
        chosen.append((candidate, evaluation, _source_remaining(applied, used), _target_remaining(applied, used)))
        pool.remove(candidate)

    legacy = set(legacy_ids) if legacy_ids is not None else None
    rows: list[dict[str, Any]] = []
    for order, (candidate, evaluation, source_left, target_left) in enumerate(chosen, start=1):
        rows.append(_row(candidate, mode, applied_by_id, unchecked_by_id, caps, legacy, limit,
                         PARTIALLY_SELECTED if evaluation.partial else SELECTED, "", evaluation, order,
                         source_left, target_left))
    chosen_ids = {id(candidate) for candidate, *_ in chosen}
    for candidate in sorted((item for item in prepared if id(item) not in chosen_ids), key=_base_key):
        applied = applied_by_id[candidate.index]
        evaluation = _Evaluation(0.0, 0.0, None, "", False, [])
        status, reason = candidate.status, candidate.reason
        duplicate_of = candidate.duplicate_of
        if status is None and candidate.lane in lanes:
            status, duplicate_of = REJECTED_DUPLICATE, lanes[candidate.lane]
            reason = f"lane {'/'.join(candidate.lane)} already allocated to {duplicate_of}"
        elif status is None:
            evaluation = _evaluate(candidate, applied, used, partial_policy, cost_recompute)
            if evaluation.allocated > EPS:
                status = REJECTED_SELECTION_LIMIT
                reason = f"executable {round(evaluation.allocated, 6)} of {round(float(candidate.qty), 6)} but the selection limit {limit} was reached"
            elif evaluation.blocked_status:
                status, reason = evaluation.blocked_status, evaluation.blocked or ""
            else:
                cap = _primary_binding(evaluation.binding)
                status = CAP_REJECTION[cap.kind]
                remaining = max(0.0, float(cap.value) - used.get((cap.kind, cap.key), 0.0))
                reason = (f"{cap.kind} {'/'.join(cap.key)} remaining {round(remaining, 6)} of {round(float(cap.value), 6)} "
                          f"({cap.provenance}) < requested {round(float(candidate.qty), 6)}")
                if evaluation.blocked:
                    reason += f"; partial not allocated: {evaluation.blocked}"
        candidate_row = _row(candidate, mode, applied_by_id, unchecked_by_id, caps, legacy, limit, status, reason,
                             evaluation, None, _source_remaining(applied, used), _target_remaining(applied, used))
        candidate_row["duplicate_of"] = duplicate_of
        candidate_row["blocking_selected_ids"] = "|".join(sorted({
            item for cap in evaluation.binding for item in consumers.get((cap.kind, cap.key), [])})) or None
        rows.append(candidate_row)
    plan = _plan_summary(rows, prepared, caps, mode, partial_policy, limit, chosen)
    plan["rows"] = rows
    return plan


DATA_STATUSES = frozenset({INSUFFICIENT_CAP_DATA, UNIT_MISMATCH, UNKNOWN_UNIT, REJECTED_INVALID_INPUT, REJECTED_INFEASIBLE_ROUTE})


def _individual(candidate: _Candidate, applied: Mapping[str, Cap]) -> tuple[float | None, list[str]]:
    """Cap room of the candidate if it were selected alone, and every cap it exceeds alone."""
    if candidate.qty is None or candidate.qty <= 0:
        return None, []
    room = min([float(candidate.qty), *(float(cap.value) for cap in applied.values())])
    over = [f"{kind}({round(float(cap.value), 6)}<{round(float(candidate.qty), 6)})"
            for kind, cap in sorted(applied.items()) if float(cap.value) < float(candidate.qty) - EPS]
    return round(room, 6), over


def _category(status: str, room: float | None, qty: float | None) -> str:
    """Separates candidates that are infeasible on their own from ones blocked by the shared selection."""
    if status in (SELECTED, PARTIALLY_SELECTED):
        return "SELECTED"
    if status in DATA_STATUSES:
        return "DATA_OR_INPUT"
    if status == REJECTED_DUPLICATE:
        return "DUPLICATE"
    if status == REJECTED_SELECTION_LIMIT:
        return "SELECTION_LIMIT"
    if room is None or room <= EPS:
        return "INDIVIDUALLY_OVER_CAP"
    if qty is not None and room < float(qty) - EPS:
        return "PARTIAL_ONLY"
    return "SHARED_CAP_CONSUMED"


def _primary_binding(binding: Sequence[Cap]) -> Cap:
    order = {kind: position for position, kind in enumerate((SOURCE_STOCK, SOURCE_SURPLUS, TARGET_NEED, ROUTE_CAPACITY, DC_CAPACITY))}
    return sorted(binding, key=lambda cap: order[cap.kind])[0]


def _cap_value(caps: CapTable, candidate: _Candidate, kind: str) -> float | None:
    cap = caps.get(candidate.slots[kind]) if kind in candidate.slots else None
    return None if cap is None or cap.value is None else round(float(cap.value), 6)


def _row(candidate: _Candidate, mode: str, applied_by_id: Mapping[int, Mapping[str, Cap]],
         unchecked_by_id: Mapping[int, Sequence[str]], caps: CapTable, legacy: set[str] | None, limit: int | None,
         status: str, reason: str, evaluation: _Evaluation, order: int | None, source_left: float | None,
         target_left: float | None) -> dict[str, Any]:
    applied = applied_by_id[candidate.index]
    provenance = []
    for kind in CAP_KINDS:
        if kind in candidate.slots:
            cap = caps.get(candidate.slots[kind])
            provenance.append(f"{kind}={cap.provenance if cap is not None and cap.value is not None else 'MISSING'}")
    selected = status in (SELECTED, PARTIALLY_SELECTED)
    in_legacy = (candidate.candidate_id in legacy) if legacy is not None else (
        candidate.original_rank is not None and limit is not None and candidate.original_rank <= limit)
    cost_status = evaluation.cost_status if selected else (
        COST_NOT_COMPARABLE if status == COST_NOT_COMPARABLE else "NOT_SELECTED")
    room, over = _individual(candidate, applied) if status not in DATA_STATUSES else (None, [])
    return {
        "candidate_id": candidate.candidate_id, "route_id": candidate.route_id or None,
        "product_id": candidate.product, "source_id": candidate.source, "target_id": candidate.target,
        "route_type": candidate.route_type, "dc_id": candidate.dc or None,
        "original_rank": None if candidate.original_rank is None else int(candidate.original_rank),
        "in_legacy_top_n": bool(in_legacy), "shared_feasibility_rank": order,
        "original_recommended_qty": candidate.qty,
        "allocated_qty": round(evaluation.allocated, 6) if selected else 0.0,
        "executable_qty_at_end": None if selected else round(evaluation.executable, 6),
        "source_remaining_qty": source_left, "target_remaining_need": target_left,
        "source_surplus_cap": _cap_value(caps, candidate, SOURCE_SURPLUS),
        "source_stock_cap": _cap_value(caps, candidate, SOURCE_STOCK),
        "target_need_cap": _cap_value(caps, candidate, TARGET_NEED),
        "route_capacity_cap": _cap_value(caps, candidate, ROUTE_CAPACITY),
        "dc_capacity_cap": _cap_value(caps, candidate, DC_CAPACITY),
        "cap_provenance": ";".join(provenance), "applied_caps": "|".join(sorted(applied)) if status != INSUFFICIENT_CAP_DATA else "",
        "unchecked_constraints": "|".join(unchecked_by_id.get(candidate.index, [])),
        "feasibility_mode": mode, "selection_status": status, "rejection_reason": reason or None,
        "binding_caps": "|".join(cap.kind for cap in evaluation.binding) or None,
        "individual_cap_room": room, "individual_over_caps": "|".join(over) or None,
        "rejection_category": _category(status, room, candidate.qty),
        "blocking_selected_ids": None, "duplicate_of": None,
        "partial_blocked_reason": evaluation.blocked if not selected else None,
        "quantity_unit": candidate.unit or "UNDECLARED", "unit_status": candidate.unit_status,
        "cost_basis": candidate.cost_basis, "cost_basis_declared": candidate.cost_basis_declared,
        "currency": candidate.currency, "original_move_cost": candidate.cost,
        "allocated_move_cost": None if not selected or evaluation.cost is None else round(float(evaluation.cost), 6),
        "cost_status": cost_status,
    }


def _plan_summary(rows: Sequence[Mapping[str, Any]], prepared: Sequence[_Candidate], caps: CapTable, mode: str,
                  partial_policy: str, limit: int | None, chosen: Sequence[tuple]) -> dict[str, Any]:
    selected = [row for row in rows if row["selection_status"] in (SELECTED, PARTIALLY_SELECTED)]
    statuses = Counter(row["selection_status"] for row in rows)
    eligible = [item for item in prepared if item.status not in (REJECTED_INVALID_INPUT, REJECTED_INFEASIBLE_ROUTE)]
    if not prepared:
        plan_status = "EMPTY_INPUT"
    elif selected:
        plan_status = "SELECTED"
    elif eligible and all(item.status == INSUFFICIENT_CAP_DATA for item in eligible):
        plan_status = INSUFFICIENT_CAP_DATA
    elif statuses.get(COST_NOT_COMPARABLE):
        plan_status = NO_COST_COMPARABLE_SELECTION
    else:
        plan_status = "NO_FEASIBLE_SELECTION"
    applied_provenance: dict[str, Counter] = defaultdict(Counter)
    proxy_used = False
    for candidate, *_ in chosen:
        for kind, slot in candidate.slots.items():
            cap = caps.get(slot)
            if cap is not None and cap.value is not None:
                applied_provenance[kind][cap.provenance] += 1
                proxy_used = proxy_used or not cap.strict_accepted
    categories = Counter(row["rejection_category"] for row in rows)
    cause = None
    if plan_status == NO_COST_COMPARABLE_SELECTION:
        cause = (f"{statuses[COST_NOT_COMPARABLE]} candidate(s) have a cap-feasible smaller quantity, but its move cost "
                 "cannot be derived from the declared cost basis; nothing is selected without a comparable cost "
                 "(see the quantity-feasibility view)")
    elif plan_status == "NO_FEASIBLE_SELECTION":
        blocked = {key: value for key, value in categories.items() if key != "SELECTED"}
        if set(blocked) == {"INDIVIDUALLY_OVER_CAP"}:
            cause = ("every candidate exceeds its own cap even when selected alone (cap room 0): the input quantities, "
                     "not the shared selection, are infeasible under these caps")
        else:
            cause = "no candidate is executable: " + ", ".join(f"{key}={value}" for key, value in sorted(blocked.items()))
    unchecked = sorted({part for row in selected for part in (row["unchecked_constraints"] or "").split("|") if part})
    costs = [row["allocated_move_cost"] for row in selected]
    currencies = sorted({row["currency"] for row in selected if row["currency"]})
    cost_complete = all(value is not None for value in costs) and len(currencies) <= 1
    if plan_status == INSUFFICIENT_CAP_DATA:
        claim = "NO_CLAIM_REQUIRED_CAPS_NOT_VALIDATED"
    elif not selected:
        claim = "NO_SELECTION"
    elif mode == STRICT_ACTUAL or not proxy_used:
        claim = "ACTUAL_CAPS_SATISFIED"
    else:
        claim = "PROXY_CAPS_SATISFIED_NOT_AN_OPERATIONAL_GUARANTEE"
    if selected and unchecked:
        claim += "; UNCHECKED:" + "|".join(unchecked)
    unknown_cost = sum(row["cost_status"] == COST_NOT_COMPARABLE for row in selected)
    if unknown_cost:
        claim += f"; QUANTITY_FEASIBILITY_ONLY: move cost not comparable for {unknown_cost} selected row(s)"
    return {
        "version": SELECTION_VERSION, "mode": mode, "partial_policy": partial_policy, "max_routes": limit,
        "plan_status": plan_status, "feasibility_claim": claim, "proxy_caps_used": proxy_used,
        "unchecked_constraints": unchecked,
        "candidate_count": len(prepared), "selected_count": len(selected),
        "partial_count": sum(row["selection_status"] == PARTIALLY_SELECTED for row in selected),
        "selected_ids": [row["candidate_id"] for row in selected],
        "total_allocated_qty": round(sum(float(row["allocated_qty"]) for row in selected), 6),
        "total_original_qty_of_selected": round(sum(float(row["original_recommended_qty"]) for row in selected), 6),
        "total_move_cost": round(sum(costs), 6) if selected and cost_complete else (0.0 if not selected else None),
        "cost_complete": cost_complete, "cost_currencies": currencies,
        "status_counts": dict(sorted(statuses.items())),
        "rejection_category_counts": dict(sorted(categories.items())), "no_selection_cause": cause,
        "applied_cap_provenance": {kind: dict(sorted(counter.items())) for kind, counter in sorted(applied_provenance.items())},
        "validation": validate_plan(selected, caps, mode=mode, max_routes=limit, quantity_field="allocated_qty"),
    }


# ------------------------------------------------------------------------------------------------ validation


def validate_plan(
    rows: Sequence[Mapping[str, Any]], caps: CapTable, *, mode: str, max_routes: int | None,
    quantity_field: str | None = None,
) -> dict[str, Any]:
    """Independent re-check of a set of moves against the caps of ``mode`` (used for legacy Top-N and new plans)."""
    usage: defaultdict[tuple, float] = defaultdict(float)
    ids: Counter = Counter()
    lanes: Counter = Counter()
    for index, row in enumerate(rows):
        quantity = _number(row.get(quantity_field)) if quantity_field else _first_number(row, QTY_FIELDS)[0]
        quantity = float(quantity or 0.0)
        ids[candidate_identity(row, index)] += 1
        product, source, target, route_type, dc = _ids(row)
        lanes[(product, source, target, route_type, dc)] += 1
        for slot in cap_slots(row, index).values():
            usage[slot] += quantity
    violations: list[dict[str, Any]] = []
    unverifiable: set[str] = set()
    excess: defaultdict[str, float] = defaultdict(float)
    for label, kinds in (("SOURCE", SOURCE_KINDS), ("TARGET_NEED", (TARGET_NEED,)), ("ROUTE_CAPACITY", (ROUTE_CAPACITY,)),
                         ("DC_CAPACITY", (DC_CAPACITY,))):
        groups = sorted({slot[1] for slot in usage if slot[0] in kinds})
        for key in groups:
            used = usage.get((kinds[0], key), 0.0)
            limits, present = [], False
            for kind in kinds:
                cap = caps.get((kind, key))
                if cap is None or cap.value is None:
                    continue
                present = True
                if mode == STRICT_ACTUAL and not cap.strict_accepted:
                    unverifiable.add(f"{kind}={cap.provenance}")
                    continue
                limits.append(cap)
            if not limits:
                if not present and label in ("SOURCE", "TARGET_NEED"):
                    unverifiable.add(f"{label}=MISSING")
                continue
            tightest = min(limits, key=lambda cap: float(cap.value))
            over = used - float(tightest.value)
            if over > 1e-8:
                excess[label] += over
                violations.append({"constraint": tightest.kind, "group": "/".join(key), "used": round(used, 6),
                                   "cap": round(float(tightest.value), 6), "excess": round(over, 6),
                                   "provenance": tightest.provenance})
    duplicate_ids = sum(count - 1 for count in ids.values() if count > 1)
    duplicate_lanes = sum(count - 1 for count in lanes.values() if count > 1)
    limit_exceeded = max_routes is not None and len(rows) > int(max_routes)
    count = len(violations) + duplicate_ids + duplicate_lanes + int(limit_exceeded)
    return {
        "selected_count": len(rows), "selection_limit": max_routes, "selection_limit_exceeded": bool(limit_exceeded),
        "duplicate_candidate_count": duplicate_ids, "duplicate_lane_count": duplicate_lanes,
        "source_excess_qty": round(excess["SOURCE"], 6), "target_excess_qty": round(excess["TARGET_NEED"], 6),
        "route_capacity_excess_qty": round(excess["ROUTE_CAPACITY"], 6), "dc_capacity_excess_qty": round(excess["DC_CAPACITY"], 6),
        "violation_count": count, "violations": violations, "unverifiable_constraints": sorted(unverifiable),
        "certified": count == 0 and not unverifiable,
    }


def compare_with_legacy(plan: Mapping[str, Any], records: Sequence[Mapping[str, Any]], caps: CapTable,
                        legacy_ids: Sequence[str], *, mode: str, max_routes: int | None) -> dict[str, Any]:
    """Legacy Top-N (rank slice, recommended quantities) next to the shared-feasibility plan, same caps and mode."""
    by_id = {candidate_identity(record, index): record for index, record in enumerate(records)}
    legacy_rows = [by_id[item] for item in legacy_ids if item in by_id]
    legacy_qty = sum(float(_first_number(row, QTY_FIELDS)[0] or 0.0) for row in legacy_rows)
    legacy_costs = [_first_number(row, COST_FIELDS)[0] for row in legacy_rows]
    legacy_cost = sum(legacy_costs) if all(value is not None for value in legacy_costs) else None
    new_ids = list(plan.get("selected_ids") or [])
    status = {row["candidate_id"]: row for row in plan.get("rows") or []}
    new_cost = plan.get("total_move_cost")
    return {
        "legacy_route_ids": list(legacy_ids), "new_route_ids": new_ids,
        "kept": [item for item in legacy_ids if item in new_ids],
        "dropped_from_legacy": [{"candidate_id": item, "selection_status": status.get(item, {}).get("selection_status"),
                                 "rejection_reason": status.get(item, {}).get("rejection_reason")}
                                for item in legacy_ids if item not in new_ids],
        "added_by_shared_feasibility": [item for item in new_ids if item not in legacy_ids],
        "legacy_selected_count": len(legacy_rows), "new_selected_count": len(new_ids),
        "legacy_total_qty": round(legacy_qty, 6), "new_total_qty": plan.get("total_allocated_qty"),
        "qty_delta": round(float(plan.get("total_allocated_qty") or 0.0) - legacy_qty, 6),
        "legacy_total_cost": None if legacy_cost is None else round(legacy_cost, 6), "new_total_cost": new_cost,
        "cost_delta": None if legacy_cost is None or new_cost is None else round(float(new_cost) - legacy_cost, 6),
        "legacy_validation": validate_plan(legacy_rows, caps, mode=mode, max_routes=max_routes),
        "new_validation": plan.get("validation"),
    }


# ------------------------------------------------------------------------------------------------ pipeline


def real_transport_cost_recompute(record: Mapping[str, Any], quantity: float) -> float | None:
    """Exact cost of a smaller quantity with the same official-tariff engine; None unless that engine priced the row."""
    flag = record.get("real_transport_applied")
    if not (flag is True or str(flag).strip().lower() == "true"):
        return None
    from services.real_transport_enrichment import enrich_real_transport

    frame = pd.DataFrame([{**{name: record.get(name) for name in ("route_id", "product_id", "source_id", "target_id")},
                           "recommended_qty": quantity}])
    priced = enrich_real_transport(frame)
    row = priced.iloc[0]
    if not bool(row.get("real_transport_applied")):
        return None
    return _number(row.get("move_cost"))


def build_shared_feasibility_analysis(
    uploaded_data: Mapping[str, Any] | None, candidates: Any, legacy_top: Sequence[Mapping[str, Any]],
    *, max_routes: int | None = DEFAULT_MAX_ROUTES, partial_policy: str = PARTIAL_IF_SAFE,
) -> dict[str, Any]:
    """Shared-feasibility plans next to the production Top-N. Never changes recommendations, ranks, Top-5 or KPIs.

    The result is deterministic for a given input (no timings), like every other PipelineResult field.
    """
    if isinstance(candidates, pd.DataFrame):
        records = candidates.where(pd.notna(candidates), None).to_dict("records")
    else:
        records = [dict(item) for item in candidates or []]
    caps = pipeline_caps(records, uploaded_data)
    default_unit = normalize_unit(_config((uploaded_data or {}).get("config")).get("quantity_unit"))
    legacy_ids = [candidate_identity(item, index) for index, item in enumerate(legacy_top or [])]
    modes: dict[str, Any] = {}
    for mode in MODES:
        plan = select_shared_feasible(records, caps, mode=mode, max_routes=max_routes, partial_policy=partial_policy,
                                      cost_recompute=real_transport_cost_recompute, legacy_ids=legacy_ids,
                                      default_unit=default_unit)
        plan["legacy_comparison"] = compare_with_legacy(plan, records, caps, legacy_ids, mode=mode, max_routes=max_routes)
        # Executability separated from cost comparability: same caps, splits allowed with their cost left NULL.
        plan["quantity_feasibility_view"] = select_shared_feasible(
            records, caps, mode=mode, max_routes=max_routes, partial_policy=PARTIAL_COST_UNKNOWN,
            cost_recompute=real_transport_cost_recompute, legacy_ids=legacy_ids, default_unit=default_unit)
        modes[mode] = plan
    return {
        "status": "parallel_only", "version": SELECTION_VERSION,
        "constraint_definition": f"{CONSTRAINT_VERSION} caps (optimality_gap_service.build_constraint_context) + on-hand stock",
        "selection_rule": "Varo Final key with executable quantity under remaining shared caps as key 1",
        "strict_accepted_provenance": sorted(STRICT_ACCEPTED_PROVENANCE),
        "max_routes": max_routes, "max_routes_provenance": "CONFIG", "partial_policy": partial_policy,
        "production_action_applied": False, "legacy_top_n_replaced": False,
        "legacy_top_n_ids": legacy_ids,
        "cap_provenance": [cap.as_row() for _, cap in sorted(caps.items(), key=lambda item: (item[0][0], item[0][1]))],
        "modes": modes,
    }
