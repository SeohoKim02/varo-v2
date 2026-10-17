"""Candidate-only adapters onto canonical + TargetNeed, never an input imputer.

Unknown state/lot/reservations stay unknown. Duplicate identities are quarantined,
not summed or silently dropped. Production adapters and formulas are untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter
from typing import Any, Mapping, Sequence
import hashlib
import json
import math

from services.canonical_schema import LINEAGE
from services.target_need_contract import TargetNeed, source_availability
from services import operational_evidence as oe

VERSION = "operating-snapshot-candidate-0.1"


def signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def optional_text(value: Any) -> str | None:
    if value is None or str(value).strip().lower() in {"", "nan", "none", "<na>"}:
        return None
    return str(value).strip()


def number(value: Any) -> float | None:
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class StockPosition:
    dataset: str
    network: str
    as_of: str
    location: str
    product: str
    state: str | None
    lot: str | None
    unit: str | None
    on_hand: float | None
    reserved: float | None
    approved_transfer: float | None
    inbound_confirmed: float | None
    outbound_confirmed: float | None
    expiry_date: str | None
    lineage: Mapping[str, Any]
    issues: tuple[str, ...] = ()

    @property
    def key(self) -> tuple:
        return (self.dataset, self.network, self.as_of, self.location,
                self.product, self.state, self.lot, self.unit)


@dataclass(frozen=True)
class OperationalSnapshot:
    dataset: str
    network: str
    as_of: str
    horizon_days: int
    positions: tuple[StockPosition, ...]
    target_needs: tuple[TargetNeed, ...]
    policy: str
    report: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def base_problem_signature(self) -> str:
        data = asdict(self)
        data["positions"] = sorted(data["positions"], key=lambda x: json.dumps(x, sort_keys=True))
        data["target_needs"] = sorted(data["target_needs"], key=lambda x: json.dumps(x, sort_keys=True))
        return signature(data)


def from_canonical(rows: Sequence[Mapping[str, Any]], *, dataset: str, network: str,
                   as_of: str, horizon_days: int, target_needs: Sequence[TargetNeed] = (),
                   context: Mapping[str, Any] | None = None) -> OperationalSnapshot:
    """Caller supplies already validated canonical rows and accepted operator facts.

    Canonical ``available_qty`` is NOT assumed approved for transfer. The explicit
    ``approved_transfer_qty`` extension must have accepted provenance, validity and
    the same unit. Reuse source_availability to keep reservation semantics intact.
    """
    if not dataset or not network or horizon_days < 1:
        raise ValueError("dataset, network and positive horizon required")
    positions = []
    for row in rows:
        issues: list[str] = []
        state, lot, unit = (optional_text(row.get(k)) for k in ("product_state", "lot_id", "unit"))
        location = optional_text(row.get("location_key"))
        product = optional_text(row.get("product_key"))
        if not location or not product:
            issues.append("NAMESPACED_ID_REQUIRED")
        if str(row.get("source_dataset")) != dataset:
            issues.append("DATASET_MISMATCH")
        if str(row.get("snapshot_date")) != as_of:
            issues.append("AS_OF_MISMATCH")
        if not state or state.upper() in {"UNKNOWN", "UNDECLARED", "UNCONFIRMED"}:
            issues.append("STATE_UNKNOWN")
        if not unit or unit.upper() in {"UNKNOWN", "UNDECLARED", "UNCONFIRMED"} or row.get("unit_status") not in {"DIRECT", "SOURCE_METADATA", "DERIVED"}:
            issues.append("UNIT_UNCONFIRMED")
        if row.get("lot_required") and not lot:
            issues.append("LOT_REQUIRED")
        if row.get("analysis_eligible") is False:
            issues.append("CANONICAL_ROW_NOT_ELIGIBLE")
        on_hand, reserved, cap = (number(row.get(k)) for k in
                                  ("inventory_qty", "reserved_qty", "approved_transfer_qty"))
        if on_hand is None or on_hand < 0:
            issues.append("ON_HAND_INVALID")
        if reserved is not None and reserved < 0:
            issues.append("RESERVED_INVALID")
        facts = {}
        if reserved is not None:
            facts["RESERVED"] = (reserved, unit, oe.DIRECT_REAL, row.get("source_file"))
        if cap is not None:
            valid = row.get("transfer_valid_until")
            if cap < 0 or not oe.strict_accepted(str(row.get("transfer_evidence", ""))) or not valid or str(valid) < as_of:
                issues.append("TRANSFER_APPROVAL_INVALID")
            else:
                facts["TRANSFER_CAP"] = (cap, unit, row["transfer_evidence"], row.get("source_file"))
        availability = source_availability(network_id=network, location_id=location or "",
            product_id=product or "", as_of=as_of, dataset_on_hand=on_hand,
            dataset_on_hand_evidence=oe.DIRECT_REAL, dataset_unit=unit, operator=facts)
        if availability["conflict_codes"]:
            issues.extend(availability["conflict_codes"].split("|"))
        approved = availability["transferable_qty"]
        # An approval cap never authorizes reservation overlap in this candidate adapter.
        if approved is not None and reserved is not None and on_hand is not None:
            approved = min(approved, max(0.0, on_hand - reserved))
        if approved is None:
            issues.append("TRANSFER_APPROVAL_REQUIRED")
        if not all(optional_text(row.get(k)) for k in LINEAGE):
            issues.append("LINEAGE_INCOMPLETE")
        positions.append(StockPosition(dataset, network, as_of, location or "", product or "",
            state, lot, unit, on_hand, reserved, approved,
            number(row.get("confirmed_inbound_qty")), number(row.get("confirmed_outbound_qty")),
            optional_text(row.get("expiry_date")), {k: optional_text(row.get(k)) for k in LINEAGE},
            tuple(sorted(set(issues)))))
    counts = Counter(p.key for p in positions)
    positions = [StockPosition(**{**asdict(p), "issues": tuple(sorted(set(p.issues) |
                 ({"DUPLICATE_IDENTITY_CONFLICT"} if counts[p.key] > 1 else set())))}) for p in positions]
    needs_issues = []
    needs_keys = Counter((n.location_id, n.product_id) for n in target_needs)
    for n in target_needs:
        if (n.dataset_id != dataset or n.network_id != network or n.as_of_date != as_of or
                n.planning_horizon_days != horizon_days or not n.accepted or
                not n.quantity_unit or not n.valid_until or n.valid_until < as_of or
                needs_keys[(n.location_id, n.product_id)] != 1):
            needs_issues.append(n.as_dict())
    totals: dict[str, float] = {}
    accepted: dict[str, float] = {}
    for p in positions:
        unit_key = p.unit or "UNDECLARED"
        if p.on_hand is not None:
            totals[unit_key] = totals.get(unit_key, 0) + p.on_hand
            if not p.issues:
                accepted[unit_key] = accepted.get(unit_key, 0) + p.on_hand
    report = {"rows_before": len(rows), "rows_after": len(positions), "keys": len(counts),
        "duplicate_rows": sum(v for v in counts.values() if v > 1),
        "unknown_state_rows": sum(p.state is None for p in positions),
        "unknown_lot_rows": sum(p.lot is None for p in positions),
        "unknown_reserved_rows": sum(p.reserved is None for p in positions),
        "observed_totals_by_unit": totals, "accepted_totals_by_unit": accepted,
        "quarantined_rows": sum(bool(p.issues) for p in positions), "invalid_needs": needs_issues,
        "production_approved": False, "note": "Candidate validation is not operational approval"}
    return OperationalSnapshot(dataset, network, as_of, horizon_days, tuple(positions),
                               tuple(target_needs), "STRICT_OPERATOR", report, context or {})
