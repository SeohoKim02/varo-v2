"""Operational evidence, demand semantics and candidate eligibility (P1-B).

Shadow and research only.  It answers, per transfer candidate, which of the inputs a move depends on are backed by
operational evidence, which rest on PROXY rules, and which are unknown.  Nothing here changes candidate_generator, the
production pool, Varo Final ranks, Top-5, T1 plans, T2 actions, the MILP, Seller Loss or KPIs.

* evidence classes      DIRECT_REAL / DERIVED_REAL / VERIFIED_OPERATOR_INPUT / USER_ESTIMATE / PROXY / CONFIG /
                        UNKNOWN / MISSING, a metadata layer over the existing provenance contract (``base_provenance``)
* target need formulas  the generator and benchmark formulas, their exact difference, an actual-sales formula and the
                        operator-declared need; ``TARGET_NEED_CONTRACT`` is the proposed single contract
* route evidence        A observed transfer / B operator allowed / C road-connectable / D distance only / E generated;
                        only A and B count as operational evidence, a road distance never does
* operator inputs       ``operational_route_evidence`` and ``operational_demand_evidence`` record contracts (scope,
                        validity window, network namespace, conflicts); SAMPLE rows are never usable
* eligibility           ``evaluate_lane``: per-dimension evidence, STRICT_ELIGIBLE only when every required dimension
                        is accepted, otherwise PROXY_FEASIBLE or the first unverified dimension, reason codes kept
* planning              the missing-input planner, fair cost comparison and the cross-dataset join guard
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date as _date
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from services import shared_feasibility_selection as sf

EVIDENCE_VERSION = "operational-evidence-v0.1"
NEED_HORIZON_DAYS = 7  # the 7-day horizon both existing need formulas use (generator and benchmark)

# ------------------------------------------------------------------------------------------------ evidence classes

DIRECT_REAL, DERIVED_REAL = "DIRECT_REAL", "DERIVED_REAL"
VERIFIED_OPERATOR_INPUT, USER_ESTIMATE = "VERIFIED_OPERATOR_INPUT", "USER_ESTIMATE"
PROXY, CONFIG, UNKNOWN, MISSING = "PROXY", "CONFIG", "UNKNOWN", "MISSING"
EVIDENCE_CLASSES = (DIRECT_REAL, DERIVED_REAL, VERIFIED_OPERATOR_INPUT, USER_ESTIMATE, PROXY, CONFIG, UNKNOWN, MISSING)
ACCEPTED_EVIDENCE = frozenset({DIRECT_REAL, DERIVED_REAL, VERIFIED_OPERATOR_INPUT})
ACTUAL_SPEND_EVIDENCE = frozenset({DIRECT_REAL, VERIFIED_OPERATOR_INPUT})  # an invoice / contract, not a computed tariff
# The existing provenance contract (seller_loss_engine.PROVENANCE_CLASSES, P1 cost UNKNOWN) is kept; the new classes
# are metadata on top of it.  An operator input counts as verified only when it says so: USER_INPUT without a
# verification flag is an estimate here, while T1 keeps accepting USER_INPUT caps unchanged.
BASE_PROVENANCE = {
    DIRECT_REAL: "DIRECT_REAL", DERIVED_REAL: "DERIVED_REAL", VERIFIED_OPERATOR_INPUT: "USER_INPUT",
    USER_ESTIMATE: "SCENARIO_INPUT", PROXY: "PROXY", CONFIG: "CONFIG", UNKNOWN: "UNKNOWN", MISSING: "MISSING",
}
VERIFIED_FLAGS = frozenset({"VERIFIED", "OPERATOR_VERIFIED", "VERIFIED_OPERATOR_INPUT", "CONFIRMED"})
ESTIMATE_FLAGS = frozenset({"ESTIMATE", "ESTIMATED", "USER_ESTIMATE", "UNVERIFIED", "DRAFT"})


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evidence_class(label: Any, verification: Any = None) -> str:
    """Evidence class of a provenance label (existing contract or new class) and an optional verification flag."""
    text = _text(label).upper()
    flag = _text(verification).upper()
    if not text:
        return MISSING
    if text in (DIRECT_REAL, "ACTUAL", "OBSERVED"):
        return DIRECT_REAL
    if text == DERIVED_REAL:
        return DERIVED_REAL
    if text in (VERIFIED_OPERATOR_INPUT, USER_ESTIMATE):
        if text == VERIFIED_OPERATOR_INPUT and flag in ESTIMATE_FLAGS:
            return USER_ESTIMATE
        return text
    if text in ("USER_INPUT", "DERIVED_FROM_USER_INPUT"):
        return VERIFIED_OPERATOR_INPUT if flag in VERIFIED_FLAGS else USER_ESTIMATE
    if text in ("SCENARIO_INPUT", "SCENARIO", "SAMPLE"):
        return USER_ESTIMATE
    if "PROXY" in text:
        return PROXY
    if text == CONFIG:
        return CONFIG
    if text == MISSING:
        return MISSING
    return UNKNOWN


def base_provenance(evidence: str) -> str:
    return BASE_PROVENANCE.get(evidence, "UNKNOWN")


def strict_accepted(evidence: str) -> bool:
    """Same acceptance as T1 STRICT_ACTUAL (sf.STRICT_ACCEPTED_PROVENANCE) through the base label."""
    return evidence in ACCEPTED_EVIDENCE and base_provenance(evidence) in sf.STRICT_ACCEPTED_PROVENANCE


@dataclass(frozen=True)
class Evidence:
    """One dimension of one candidate: the value used, its evidence class, the basis and the reasons it is not accepted."""

    dimension: str
    value: Any = None
    evidence: str = MISSING
    basis: str | None = None
    source: str | None = None
    unit: str | None = None
    status: str | None = None
    reasons: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()  # accepted, but an open operational condition remains

    @property
    def accepted(self) -> bool:
        return self.value is not None and self.value is not False and strict_accepted(self.evidence)

    def label(self) -> str:
        return f"{self.status or ('ACCEPTED' if self.accepted else 'NOT_ACCEPTED')}:{self.evidence}"

    def as_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "value": self.value, "evidence_class": self.evidence,
                "base_provenance": base_provenance(self.evidence), "accepted": self.accepted, "status": self.status,
                "basis": self.basis, "source": self.source, "unit": self.unit, "reasons": list(self.reasons),
                "conditions": list(self.conditions)}


# ------------------------------------------------------------------------------------------------ target need formulas

GENERATOR_FORMULA, BENCHMARK_FORMULA = "GENERATOR_NEED", "BENCHMARK_NEED"
ACTUAL_SALES_FORMULA, OPERATOR_DECLARED_FORMULA = "ACTUAL_SALES_NET_NEED", "OPERATOR_DECLARED_NEED"
NEED_FORMULAS = {
    GENERATOR_FORMULA: {
        "expression": "max(0, median - target_stock) + 7 x target_demand",
        "code": "candidate_generator.generate_candidates (targets loop); P1 research lane enumeration (enumerate_universe)",
        "inputs": "upload inventory rows (pairs under several state codes excluded); median over those rows; demand = "
                  "avg_daily_sales | sales_qty | demand_qty (Suhyup: outbound_qty); a target without a row gets "
                  "stock = median and demand 0",
        "meaning": "median deficit plus the full 7-day outflow; stock above the median is never netted against the "
                   "outflow (gross requirement)",
        "used_by": "the generator's target choice, its quantity rule and candidate_score (never a cap)"},
    BENCHMARK_FORMULA: {
        "expression": "max(0, median + 7 x target_demand - target_stock)",
        "code": "suhyup_algorithm_revalidation.enrich_inventory_constraints (target_need_7d)",
        "inputs": "actual inventory flow summed over state codes per (date, center, product); median over the six "
                  "network nodes; demand = outbound_qty",
        "meaning": "order-up-to shortfall: target position median + 7-day outflow minus on-hand stock (net requirement)",
        "used_by": "T1 / T3 / P1 TARGET_NEED cap (PROXY), Varo Final ordered selection, the lexicographic MILP"},
    ACTUAL_SALES_FORMULA: {
        "expression": "max(0, 7 x actual daily sales - target available stock)",
        "inputs": "actual retail sales of the same location, product, unit and observation window; on-hand stock",
        "meaning": "net 7-day coverage shortfall from observed sales (no cross-node median rule)",
        "used_by": "nothing yet: no dataset joinable to Suhyup holds actual sales"},
    OPERATOR_DECLARED_FORMULA: {
        "expression": "operator target_need for the location, product and decision date",
        "inputs": "operational_demand_evidence row quantity_kind DECLARED_TARGET_NEED",
        "meaning": "the business's own replenishment requirement",
        "used_by": "nothing yet: no operator input exists"},
}


def generator_need(median: float, target_stock: float, demand: float, horizon: float = NEED_HORIZON_DAYS) -> float:
    return max(0.0, median - target_stock) + demand * horizon


def benchmark_need(median: float, target_stock: float, demand: float, horizon: float = NEED_HORIZON_DAYS) -> float:
    return max(0.0, median + horizon * demand - target_stock)


def need_formula_gap(median: float, target_stock: float, demand: float, horizon: float = NEED_HORIZON_DAYS) -> float:
    """generator - benchmark for demand >= 0: min(max(0, stock - median), horizon x demand) (always >= 0)."""
    return min(max(0.0, target_stock - median), horizon * demand)


def need_case(median: float, target_stock: float, demand: float, horizon: float = NEED_HORIZON_DAYS) -> str:
    """Which branch of the two formulas applies (section 8)."""
    if demand < 0:
        return "NEGATIVE_DEMAND"
    level = "BELOW" if target_stock < median else "AT" if target_stock == median else "ABOVE"
    if demand == 0:
        return f"STOCK_{level}_MEDIAN_DEMAND_ZERO"
    if level != "ABOVE":
        return f"STOCK_{level}_MEDIAN_DEMAND_POSITIVE"
    return ("STOCK_ABOVE_MEDIAN_WITHIN_7D_OUTFLOW" if target_stock <= median + horizon * demand
            else "STOCK_ABOVE_MEDIAN_PLUS_7D_OUTFLOW")


NEED_CASES = {
    "STOCK_BELOW_MEDIAN_DEMAND_ZERO": "equal: m - s",
    "STOCK_AT_MEDIAN_DEMAND_ZERO": "equal: 0",
    "STOCK_ABOVE_MEDIAN_DEMAND_ZERO": "equal: 0",
    "STOCK_BELOW_MEDIAN_DEMAND_POSITIVE": "equal: m - s + 7d",
    "STOCK_AT_MEDIAN_DEMAND_POSITIVE": "equal: 7d",
    "STOCK_ABOVE_MEDIAN_WITHIN_7D_OUTFLOW": "generator 7d > benchmark m + 7d - s; gap s - m",
    "STOCK_ABOVE_MEDIAN_PLUS_7D_OUTFLOW": "generator 7d > benchmark 0; gap 7d (generator asks for stock the target "
                                          "already holds beyond its 7-day outflow)",
}

TARGET_NEED_CONTRACT = {
    "field": "target_need_qty",
    "required_metadata": ["formula_id", "horizon_days", "as_of_date", "observation_window", "quantity_unit",
                          "evidence_class", "inputs (target_stock, demand, target_level) each with evidence_class"],
    "formula_ids": list(NEED_FORMULAS),
    "rules": [
        "a value without formula_id is not a target need",
        "two target needs are comparable only with the same formula_id, horizon, unit and as_of_date; otherwise "
        "NOT_COMPARABLE (never mixed under the name target_need)",
        "STRICT_ACTUAL accepts OPERATOR_DECLARED_NEED (verified) or ACTUAL_SALES_NET_NEED (actual sales, same "
        "location / product / unit / window, window ending within the horizon before the decision date)",
        "GENERATOR_NEED is a generator ranking heuristic and never a cap; BENCHMARK_NEED is the research cap, PROXY",
        "an outbound, order, regional flow or forecast quantity is PROXY demand, never actual sales",
    ],
    "proposal": "operator-declared need first, actual-sales net need second, benchmark need only as a labelled PROXY "
                "research cap; production formulas are not changed in this step",
}

# ------------------------------------------------------------------------------------------------ dimensions & statuses

ROUTE, SUPPLY, DEMAND, HANDLING, UNIT, COST, CAPACITY = (
    "ROUTE", "SOURCE_SUPPLY", "TARGET_DEMAND", "TARGET_HANDLING", "QUANTITY_UNIT", "TRANSPORT_COST", "CAPACITY")
DIMENSIONS = (ROUTE, SUPPLY, DEMAND, HANDLING, UNIT, COST, CAPACITY)
REQUIRED_DIMENSIONS = (ROUTE, SUPPLY, DEMAND, HANDLING, UNIT, COST)   # STRICT_ELIGIBLE needs every one accepted
CONDITIONAL_DIMENSIONS = (CAPACITY,)                                   # VERIFIED_OPERATIONAL also needs these

VERIFIED_OPERATIONAL, VERIFIED_CONDITIONAL, PROXY_FEASIBLE = "VERIFIED_OPERATIONAL", "VERIFIED_CONDITIONAL", "PROXY_FEASIBLE"
INELIGIBLE = "INELIGIBLE"
UNVERIFIED_STATUS = {ROUTE: "ROUTE_UNVERIFIED", SUPPLY: "SUPPLY_UNVERIFIED", DEMAND: "DEMAND_UNVERIFIED",
                     HANDLING: "TARGET_HANDLING_UNVERIFIED", UNIT: "UNIT_UNVERIFIED", COST: "COST_UNVERIFIED",
                     CAPACITY: "CAPACITY_UNVERIFIED"}
ELIGIBILITY_STATUSES = (VERIFIED_OPERATIONAL, VERIFIED_CONDITIONAL, PROXY_FEASIBLE,
                        *(UNVERIFIED_STATUS[d] for d in DIMENSIONS), INELIGIBLE)
STRICT_ELIGIBLE, INSUFFICIENT_OPERATIONAL_EVIDENCE = "STRICT_ELIGIBLE", "INSUFFICIENT_OPERATIONAL_EVIDENCE"
SCOPE_STRICT, SCOPE_BENCHMARK, SCOPE_NONE = "STRICT_ACTUAL", "BENCHMARK_PROXY", "NOT_COMPARABLE"

# route classes (section 13): only A and B are operational evidence
ROUTE_A, ROUTE_B = "A_OBSERVED_TRANSFER", "B_OPERATOR_ALLOWED"
ROUTE_C, ROUTE_D, ROUTE_E = "C_ROAD_CONNECTABLE", "D_DISTANCE_ONLY", "E_GENERATED_NO_ROUTE_ROW"
ROUTE_PROHIBITED, ROUTE_CONFLICT = "ROUTE_PROHIBITED", "ROUTE_EVIDENCE_CONFLICT"
ROUTE_CLASSES = (ROUTE_A, ROUTE_B, ROUTE_C, ROUTE_D, ROUTE_E)
ROAD_ROUTE_TOKENS = ("osrm",)  # a distance computed on the road network between verified coordinates
OSRM_DOES_NOT_ESTABLISH = ("transport contract", "product handling", "cold-chain (frozen / chilled) carriage",
                           "delivery schedule", "vehicle availability", "distribution-center handling")

# quantity kinds of the demand / quantity evidence contract (section 11)
ACTUAL_SALES, DECLARED_TARGET_NEED, DECLARED_TRANSFERABLE = "ACTUAL_SALES", "DECLARED_TARGET_NEED", "DECLARED_TRANSFERABLE_STOCK"
DECLARED_RECEIVABLE = "DECLARED_RECEIVABLE"
OUTBOUND_SHIPMENT, ORDER, FORECAST = "OUTBOUND_SHIPMENT", "ORDER", "FORECAST"
REGIONAL_FLOW, LOGISTICS_MOVEMENT = "REGIONAL_FLOW", "LOGISTICS_MOVEMENT"
QUANTITY_KINDS = (ACTUAL_SALES, DECLARED_TARGET_NEED, DECLARED_TRANSFERABLE, DECLARED_RECEIVABLE, OUTBOUND_SHIPMENT,
                  ORDER, FORECAST, REGIONAL_FLOW, LOGISTICS_MOVEMENT)
DEMAND_SEMANTICS = {
    ACTUAL_SALES: "observed sales to customers at the location (actual demand)",
    DECLARED_TARGET_NEED: "operator's replenishment requirement for the location (actual requirement)",
    DECLARED_TRANSFERABLE: "operator's quantity that may leave the source (actual movable stock)",
    DECLARED_RECEIVABLE: "operator states the location may receive the product (handling, no quantity)",
    OUTBOUND_SHIPMENT: "quantity shipped out of the location: sales, auctions, transfers and returns mixed; PROXY",
    ORDER: "quantity ordered / requested; not receipts and not sales; PROXY",
    FORECAST: "model output; never an observation; PROXY",
    REGIONAL_FLOW: "regional distribution volume; not a location's demand; PROXY",
    LOGISTICS_MOVEMENT: "logistics movement volume; not demand; PROXY",
}

# ------------------------------------------------------------------------------------------------ dates and SAMPLE rows


def parse_date(value: Any) -> _date | None:
    text = _text(value)
    if not text:
        return None
    try:
        return pd.Timestamp(text).date()
    except (ValueError, TypeError):
        return None


def is_sample_row(row: Mapping[str, Any]) -> bool:
    """Template rows: a SAMPLE id or a SAMPLE note / source.  They show a format and are never evidence."""
    for key, value in row.items():
        text = _text(value).upper()
        if not text:
            continue
        if key in ("note", "evidence_source", "data_source") and text.startswith("SAMPLE"):
            return True
        if key.endswith("_id") and text.startswith("SAMPLE"):
            return True
    return False


def _truth(value: Any) -> bool | None:
    text = _text(value).lower()
    if text in {"true", "1", "yes", "y", "allowed", "허용", "가능"}:
        return True
    if text in {"false", "0", "no", "n", "prohibited", "forbidden", "금지", "불가"}:
        return False
    return None


# ------------------------------------------------------------------------------------------------ operator input contracts

ROUTE_EVIDENCE_REQUIRED = ("network_id", "source_id", "target_id", "allowed", "effective_from", "evidence_class",
                           "evidence_source")
ROUTE_EVIDENCE_OPTIONAL = ("product_id", "product_group", "effective_to", "verification_status", "transport_mode",
                           "temperature_requirement", "route_capacity_qty", "quantity_unit", "cost_basis", "cost_value",
                           "currency", "note")
ROUTE_EVIDENCE_NOT_ADOPTED = {
    "lead_time": "no selector, cap or MILP reads a lead time; transit time stays informational",
    "vehicle_type": "the tariff engine sizes vehicles by kg classes; a per-route vehicle type has no consumer yet",
    "time_window": "no delivery time-window constraint exists in T1 / T3",
}
OBSERVED_TRANSFER_REQUIRED = ("network_id", "source_id", "target_id", "transfer_date", "evidence_source")
DEMAND_EVIDENCE_REQUIRED = ("network_id", "location_id", "product_id", "quantity_kind", "observation_start",
                            "evidence_class", "data_source")
DEMAND_EVIDENCE_OPTIONAL = ("observation_end", "quantity", "quantity_unit", "verification_status", "note")
TEMPERATURES = ("FROZEN", "CHILLED", "AMBIENT")
COST_BASIS_VALUES = (sf.FIXED_PER_TRIP, sf.PER_UNIT)


def _issue(row_index: int, code: str, detail: str = "") -> dict[str, Any]:
    return {"row": row_index, "code": code, "detail": detail}


def _records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, pd.DataFrame):
        return frame.where(pd.notna(frame), None).to_dict("records")
    return [dict(row) for row in frame]


def normalize_route_evidence(frame: Any, *, network_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """operational_route_evidence rows -> usable rule records and issues.  A rejected row is never partly used."""
    records, issues = [], []
    for index, row in enumerate(_records(frame)):
        if is_sample_row(row):
            issues.append(_issue(index, "SAMPLE_ROW_NOT_USABLE"))
            continue
        missing = [name for name in ROUTE_EVIDENCE_REQUIRED if not _text(row.get(name))]
        if missing:
            issues.append(_issue(index, "MISSING_REQUIRED_FIELD", "|".join(missing)))
            continue
        if _text(row.get("network_id")) != network_id:
            issues.append(_issue(index, "CROSS_NETWORK_EVIDENCE_REJECTED", f"{row.get('network_id')} != {network_id}"))
            continue
        allowed = _truth(row.get("allowed"))
        start, end = parse_date(row.get("effective_from")), parse_date(row.get("effective_to"))
        if allowed is None:
            issues.append(_issue(index, "INVALID_ALLOWED_FLAG", _text(row.get("allowed"))))
            continue
        if start is None or (_text(row.get("effective_to")) and end is None) or (end is not None and end < start):
            issues.append(_issue(index, "INVALID_EFFECTIVE_WINDOW"))
            continue
        evidence = evidence_class(row.get("evidence_class"), row.get("verification_status"))
        temperature = _text(row.get("temperature_requirement")).upper() or None
        if temperature is not None and temperature not in TEMPERATURES:
            issues.append(_issue(index, "INVALID_TEMPERATURE", temperature))
            continue
        capacity, unit = _finite(row.get("route_capacity_qty")), sf.normalize_unit(row.get("quantity_unit"))
        notes = []
        if capacity is not None and unit is None:
            notes.append("ROUTE_CAPACITY_UNIT_MISSING")
            capacity = None
        cost_basis = _text(row.get("cost_basis")).upper() or None
        cost_value = _finite(row.get("cost_value"))
        if cost_value is not None and cost_basis not in COST_BASIS_VALUES:
            notes.append("ROUTE_COST_BASIS_UNDECLARED")
            cost_value = None
        records.append({
            "row": index, "network_id": network_id, "source_id": _text(row.get("source_id")),
            "target_id": _text(row.get("target_id")), "product_id": _text(row.get("product_id")) or None,
            "product_group": _text(row.get("product_group")) or None, "allowed": allowed,
            "effective_from": start, "effective_to": end, "evidence": evidence,
            "evidence_source": _text(row.get("evidence_source")), "transport_mode": _text(row.get("transport_mode")) or None,
            "temperature": temperature, "route_capacity_qty": capacity, "quantity_unit": unit,
            "cost_basis": cost_basis if cost_value is not None else None, "cost_value": cost_value,
            "currency": _text(row.get("currency")) or None, "notes": notes})
        if notes:
            issues.extend(_issue(index, note) for note in notes)
    return records, issues


def normalize_observed_transfers(frame: Any, *, network_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Observed inter-location transfer records (class A).  An observation is DIRECT_REAL by definition."""
    records, issues = [], []
    for index, row in enumerate(_records(frame)):
        if is_sample_row(row):
            issues.append(_issue(index, "SAMPLE_ROW_NOT_USABLE"))
            continue
        missing = [name for name in OBSERVED_TRANSFER_REQUIRED if not _text(row.get(name))]
        if missing:
            issues.append(_issue(index, "MISSING_REQUIRED_FIELD", "|".join(missing)))
            continue
        if _text(row.get("network_id")) != network_id:
            issues.append(_issue(index, "CROSS_NETWORK_EVIDENCE_REJECTED", f"{row.get('network_id')} != {network_id}"))
            continue
        when = parse_date(row.get("transfer_date"))
        if when is None:
            issues.append(_issue(index, "INVALID_TRANSFER_DATE"))
            continue
        records.append({"row": index, "network_id": network_id, "source_id": _text(row.get("source_id")),
                        "target_id": _text(row.get("target_id")), "product_id": _text(row.get("product_id")) or None,
                        "transfer_date": when, "evidence": DIRECT_REAL, "evidence_source": _text(row.get("evidence_source"))})
    return records, issues


def normalize_demand_evidence(frame: Any, *, network_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """operational_demand_evidence rows (actual sales, declared need / transferable / receivable, and PROXY kinds)."""
    records, issues = [], []
    for index, row in enumerate(_records(frame)):
        if is_sample_row(row):
            issues.append(_issue(index, "SAMPLE_ROW_NOT_USABLE"))
            continue
        missing = [name for name in DEMAND_EVIDENCE_REQUIRED if not _text(row.get(name))]
        if missing:
            issues.append(_issue(index, "MISSING_REQUIRED_FIELD", "|".join(missing)))
            continue
        if _text(row.get("network_id")) != network_id:
            issues.append(_issue(index, "CROSS_NETWORK_EVIDENCE_REJECTED", f"{row.get('network_id')} != {network_id}"))
            continue
        kind = _text(row.get("quantity_kind")).upper()
        if kind not in QUANTITY_KINDS:
            issues.append(_issue(index, "UNKNOWN_QUANTITY_KIND", kind))
            continue
        start, end = parse_date(row.get("observation_start")), parse_date(row.get("observation_end"))
        if start is None or (_text(row.get("observation_end")) and end is None) or (end is not None and end < start):
            issues.append(_issue(index, "INVALID_OBSERVATION_WINDOW"))
            continue
        quantity, unit = _finite(row.get("quantity")), sf.normalize_unit(row.get("quantity_unit"))
        if kind != DECLARED_RECEIVABLE and (quantity is None or quantity < 0):
            issues.append(_issue(index, "INVALID_QUANTITY", _text(row.get("quantity"))))
            continue
        if kind != DECLARED_RECEIVABLE and unit is None:
            issues.append(_issue(index, "QUANTITY_UNIT_MISSING"))
            continue
        evidence = evidence_class(row.get("evidence_class"), row.get("verification_status"))
        if kind in (OUTBOUND_SHIPMENT, ORDER, FORECAST, REGIONAL_FLOW, LOGISTICS_MOVEMENT) and evidence in ACCEPTED_EVIDENCE:
            issues.append(_issue(index, "PROXY_KIND_RECLASSIFIED", f"{kind} is not demand; kept as PROXY"))
            evidence = PROXY
        records.append({"row": index, "network_id": network_id, "location_id": _text(row.get("location_id")),
                        "product_id": _text(row.get("product_id")), "quantity_kind": kind, "start": start,
                        "end": end or start, "quantity": quantity, "unit": unit, "evidence": evidence,
                        "data_source": _text(row.get("data_source"))})
    return records, issues


def route_evidence_template_rows() -> list[dict[str, Any]]:
    """SAMPLE rows showing the operational_route_evidence format: placeholder ids, example values, never defaults."""
    base = {name: None for name in (*ROUTE_EVIDENCE_REQUIRED, *ROUTE_EVIDENCE_OPTIONAL)}
    return [
        {**base, "network_id": "SAMPLE_NETWORK", "source_id": "SAMPLE_CENTER_A", "target_id": "SAMPLE_CENTER_B",
         "allowed": "true", "effective_from": "2026-07-01", "effective_to": "2026-12-31",
         "evidence_class": VERIFIED_OPERATOR_INPUT, "verification_status": "VERIFIED", "evidence_source": "SAMPLE_CONTRACT",
         "temperature_requirement": "FROZEN", "note": "SAMPLE: A→B 모든 상품 운송 허용(냉동). 실제 계약·운영 규칙으로 바꾸세요"},
        {**base, "network_id": "SAMPLE_NETWORK", "source_id": "SAMPLE_CENTER_A", "target_id": "SAMPLE_CENTER_B",
         "product_id": "SAMPLE_PRODUCT_1", "allowed": "true", "effective_from": "2026-07-01",
         "evidence_class": VERIFIED_OPERATOR_INPUT, "verification_status": "VERIFIED", "evidence_source": "SAMPLE_CONTRACT",
         "route_capacity_qty": 400, "quantity_unit": "BOX", "cost_basis": sf.FIXED_PER_TRIP, "cost_value": 150000,
         "currency": "KRW", "note": "SAMPLE: 상품 단위 허용·1회 최대 400 BOX·회당 계약 운임 150,000원"},
        {**base, "network_id": "SAMPLE_NETWORK", "source_id": "SAMPLE_CENTER_B", "target_id": "SAMPLE_CENTER_C",
         "allowed": "false", "effective_from": "2026-07-15", "evidence_class": VERIFIED_OPERATOR_INPUT,
         "verification_status": "VERIFIED", "evidence_source": "SAMPLE_POLICY",
         "note": "SAMPLE: B→C 운송 금지(7/15부터). 금지 규칙은 거리 정보보다 우선"},
        {**base, "network_id": "SAMPLE_NETWORK", "source_id": "SAMPLE_CENTER_C", "target_id": "SAMPLE_CENTER_A",
         "allowed": "true", "effective_from": "2026-07-01", "evidence_class": USER_ESTIMATE,
         "verification_status": "ESTIMATE", "evidence_source": "SAMPLE_INTERVIEW",
         "note": "SAMPLE: 확인되지 않은 추정 허용 — STRICT 검증에는 쓰이지 않음"},
    ]


def demand_evidence_template_rows() -> list[dict[str, Any]]:
    """SAMPLE rows showing the operational_demand_evidence format: placeholder ids, example values, never defaults."""
    base = {name: None for name in (*DEMAND_EVIDENCE_REQUIRED, *DEMAND_EVIDENCE_OPTIONAL)}
    return [
        {**base, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_B", "product_id": "SAMPLE_PRODUCT_1",
         "quantity_kind": ACTUAL_SALES, "observation_start": "2026-07-09", "observation_end": "2026-07-15", "quantity": 140,
         "quantity_unit": "BOX", "evidence_class": DIRECT_REAL, "data_source": "SAMPLE_POS_EXPORT",
         "note": "SAMPLE: B점 7일 실제 판매 합계(BOX). 출고량이 아니라 판매량만 입력"},
        {**base, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_B", "product_id": "SAMPLE_PRODUCT_1",
         "quantity_kind": DECLARED_TARGET_NEED, "observation_start": "2026-07-16", "observation_end": "2026-07-16",
         "quantity": 60, "quantity_unit": "BOX", "evidence_class": VERIFIED_OPERATOR_INPUT,
         "verification_status": "VERIFIED", "data_source": "SAMPLE_REPLENISHMENT_PLAN",
         "note": "SAMPLE: 운영자가 확정한 7/16 B점 필요량"},
        {**base, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_A", "product_id": "SAMPLE_PRODUCT_1",
         "quantity_kind": DECLARED_TRANSFERABLE, "observation_start": "2026-07-16", "observation_end": "2026-07-16",
         "quantity": 120, "quantity_unit": "BOX", "evidence_class": VERIFIED_OPERATOR_INPUT,
         "verification_status": "VERIFIED", "data_source": "SAMPLE_WMS_ALLOCATION",
         "note": "SAMPLE: A점에서 내보낼 수 있는 수량(예약·이동중 제외)"},
        {**base, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_C", "product_id": "SAMPLE_PRODUCT_1",
         "quantity_kind": DECLARED_RECEIVABLE, "observation_start": "2026-07-01", "observation_end": "2026-12-31",
         "evidence_class": VERIFIED_OPERATOR_INPUT, "verification_status": "VERIFIED", "data_source": "SAMPLE_ASSORTMENT",
         "note": "SAMPLE: 현재 재고가 없는 C점이 상품을 입고할 수 있다는 운영 근거"},
    ]


# ------------------------------------------------------------------------------------------------ context

DEFAULT_LANE_EVIDENCE = {
    # lane column -> (evidence class when the row declares none, basis)
    "source_stock": (DIRECT_REAL, "actual daily center stock_qty, summed over state codes"),
    "source_surplus": (PROXY, "max(0, source stock - same-day cross-node product median): a rule, not transferable stock"),
    "target_stock": (DIRECT_REAL, "actual daily center stock_qty, summed over state codes"),
    "target_need_7d": (PROXY, "benchmark: max(0, median + 7 x outbound - target stock); outbound is not sales"),
}


@dataclass
class OperationalContext:
    """Evidence available for one network.  Real-data lane columns arrive on the lane; operator evidence here."""

    network_id: str
    dataset_id: str = ""
    route_records: Sequence[Mapping[str, Any]] = ()
    observed_transfers: Sequence[Mapping[str, Any]] = ()
    quantity_records: Sequence[Mapping[str, Any]] = ()
    product_units: Mapping[str, tuple[str, str]] = field(default_factory=dict)       # product -> (unit, evidence)
    product_groups: Mapping[str, str] = field(default_factory=dict)
    product_temperature: Mapping[str, str] = field(default_factory=dict)             # product -> FROZEN/CHILLED/AMBIENT
    dc_capacity: Mapping[str, tuple[float, str, str]] = field(default_factory=dict)  # dc -> (qty, unit, evidence)
    handling_history: Mapping[tuple[str, str], str] = field(default_factory=dict)    # (location, product) -> last seen
    unit_weight_status: Mapping[str, str] = field(default_factory=dict)              # product -> safe/conditional/...
    lane_evidence: Mapping[str, tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_LANE_EVIDENCE))
    observed_route_lookback_days: int | None = None   # None: no recency policy declared (a condition, not a default)
    demand_staleness_days: int = NEED_HORIZON_DAYS


def _lane_unit(lane: Mapping[str, Any], context: OperationalContext) -> str | None:
    """The quantity unit of the lane's stock values: the row's own unit, else a declared product unit."""
    unit = sf.normalize_unit(lane.get("quantity_unit"))
    declared = context.product_units.get(_text(lane.get("product_id")))
    return unit or (sf.normalize_unit(declared[0]) if declared else None)


def _lane_value(lane: Mapping[str, Any], column: str, context: OperationalContext) -> tuple[float | None, str, str]:
    value = _finite(lane.get(column))
    default, basis = context.lane_evidence.get(column, (UNKNOWN, column))
    declared = lane.get(f"{column}_provenance")
    evidence = evidence_class(declared, lane.get(f"{column}_verification")) if _text(declared) else default
    if value is None:
        evidence = MISSING
    return value, evidence, basis


# ------------------------------------------------------------------------------------------------ resolvers


def route_network_class(lane: Mapping[str, Any]) -> str:
    """C/D/E from the uploaded routes table only; never operational evidence."""
    status = _text(lane.get("route_status"))
    if status in ("", "ROUTE_UNVERIFIED") or lane.get("distance_km") is None or _finite(lane.get("distance_km")) is None:
        return ROUTE_E
    provenance = _text(lane.get("distance_provenance")).lower()
    legs = provenance.split("+") if provenance else [""]
    if all(any(token in leg for token in ROAD_ROUTE_TOKENS) for leg in legs):
        return ROUTE_C
    return ROUTE_D


def _effective(record: Mapping[str, Any], when: _date) -> bool:
    return record["effective_from"] <= when and (record["effective_to"] is None or when <= record["effective_to"])


def _specificity(record: Mapping[str, Any], product: str, group: str | None) -> int | None:
    if record.get("product_id"):
        return 2 if record["product_id"] == product else None
    if record.get("product_group"):
        return 1 if group is not None and record["product_group"] == group else None
    return 0


def _route_rules(source: str, target: str, product: str, when: _date, context: OperationalContext) -> dict[str, Any]:
    group = context.product_groups.get(product)
    matching = [r for r in context.route_records if r["source_id"] == source and r["target_id"] == target]
    unmapped_group = any(r.get("product_group") and not r.get("product_id") and group is None for r in matching)
    scoped = [(r, _specificity(r, product, group)) for r in matching]
    scoped = [(r, s) for r, s in scoped if s is not None]
    effective = [(r, s) for r, s in scoped if _effective(r, when)]
    expired = [r for r, s in scoped if not _effective(r, when)]
    authoritative = [(r, s) for r, s in effective if r["evidence"] in ACCEPTED_EVIDENCE]
    chosen = authoritative or [(r, s) for r, s in effective]
    decision, conflict, decisive = None, False, []
    if chosen:
        top = max(s for _, s in chosen)
        decisive = [r for r, s in chosen if s == top]
        flags = {r["allowed"] for r in decisive}
        conflict = len(flags) > 1
        decision = None if conflict else flags.pop()
    disagreeing_estimates = bool(authoritative) and any(
        r["allowed"] != decision for r, _ in effective if r["evidence"] not in ACCEPTED_EVIDENCE and decision is not None)
    return {"decision": decision, "conflict": conflict, "decisive": decisive, "authoritative": bool(authoritative),
            "expired": expired, "unmapped_group": unmapped_group, "disagreeing_estimates": disagreeing_estimates}


def _observed(source: str, target: str, product: str, when: _date, context: OperationalContext) -> list[Mapping[str, Any]]:
    seen = []
    for record in context.observed_transfers:
        if record["source_id"] != source or record["target_id"] != target or record["transfer_date"] > when:
            continue
        if context.observed_route_lookback_days is not None and (when - record["transfer_date"]).days > context.observed_route_lookback_days:
            continue
        seen.append(record)
    return seen


def resolve_leg(source: str, target: str, product: str, when: _date, context: OperationalContext,
                network_class: str) -> Evidence:
    """Operational route evidence for one directed leg (section 13 / 26)."""
    rules = _route_rules(source, target, product, when, context)
    observed = _observed(source, target, product, when, context)
    reasons, conditions = [], []
    record_ids = "|".join(str(r["row"]) for r in rules["decisive"]) or None
    if rules["unmapped_group"]:
        reasons.append("ROUTE_PRODUCT_GROUP_UNMAPPED")
    if rules["expired"] and not rules["decisive"]:
        reasons.append("ROUTE_EVIDENCE_EXPIRED")
    if rules["disagreeing_estimates"]:
        conditions.append("ROUTE_ESTIMATE_DISAGREES_WITH_VERIFIED_RULE")
    if rules["conflict"]:
        return Evidence(ROUTE, None, MISSING, "allowed and prohibited at the same scope and authority", record_ids,
                        status=ROUTE_CONFLICT, reasons=(*reasons, ROUTE_CONFLICT))
    if rules["decision"] is False and rules["authoritative"]:
        rule_start = min(r["effective_from"] for r in rules["decisive"])
        contradicting = [o for o in observed if o["transfer_date"] >= rule_start]
        if contradicting:  # moved while the prohibition was in force: the sources disagree
            return Evidence(ROUTE, None, MISSING, "observed transfer during an effective prohibition", record_ids,
                            status=ROUTE_CONFLICT, reasons=(*reasons, ROUTE_CONFLICT, "OBSERVED_TRANSFER_UNDER_PROHIBITION"))
        return Evidence(ROUTE, False, rules["decisive"][0]["evidence"], "effective operator prohibition", record_ids,
                        status=ROUTE_PROHIBITED, reasons=(*reasons, ROUTE_PROHIBITED))
    if rules["decision"] is True and rules["authoritative"]:
        record = rules["decisive"][0]
        temperature = context.product_temperature.get(product)
        declared = {r.get("temperature") for r in rules["decisive"] if r.get("temperature")}
        if temperature and declared and temperature not in declared:
            return Evidence(ROUTE, False, record["evidence"], f"route carries {sorted(declared)}, product needs {temperature}",
                            record_ids, status=ROUTE_PROHIBITED, reasons=(*reasons, "ROUTE_TEMPERATURE_MISMATCH"))
        if temperature in ("FROZEN", "CHILLED") and not declared:
            conditions.append("COLD_CHAIN_UNVERIFIED")
        if not any(r.get("product_id") for r in rules["decisive"]):
            conditions.append("ROUTE_ALLOWED_FOR_ALL_PRODUCTS")
        return Evidence(ROUTE, True, record["evidence"], "effective operator rule", record_ids, status=ROUTE_B,
                        reasons=tuple(reasons), conditions=tuple(conditions))
    if observed:
        if context.observed_route_lookback_days is None:
            conditions.append("ROUTE_RECENCY_POLICY_UNDECLARED")
        if not any(o.get("product_id") == product for o in observed):
            conditions.append("OBSERVED_FOR_OTHER_PRODUCTS_ONLY")
        if context.product_temperature.get(product) in ("FROZEN", "CHILLED"):
            conditions.append("COLD_CHAIN_UNVERIFIED")
        return Evidence(ROUTE, True, DIRECT_REAL, f"{len(observed)} observed transfer(s), last "
                        f"{max(o['transfer_date'] for o in observed).isoformat()}", "observed_transfers", status=ROUTE_A,
                        reasons=tuple(reasons), conditions=tuple(conditions))
    if rules["decision"] is True:
        reasons.append("ROUTE_ESTIMATE_ONLY")
    elif rules["decision"] is False:
        reasons.append("ROUTE_PROHIBITION_ESTIMATE_ONLY")
    reasons.append("ROUTE_NOT_OPERATIONALLY_EVIDENCED")
    if network_class == ROUTE_E:
        reasons.append("NO_ROUTE_ROW")
    return Evidence(ROUTE, None, MISSING if network_class == ROUTE_E else UNKNOWN,
                    f"{network_class}: a routes-table distance shows road connectivity at most, not an operated lane",
                    record_ids, status=network_class, reasons=tuple(reasons))


def resolve_route(lane: Mapping[str, Any], when: _date, context: OperationalContext) -> Evidence:
    source, target, product = (_text(lane.get(k)) for k in ("source_id", "target_id", "product_id"))
    network_class = route_network_class(lane)
    if _text(lane.get("route_type")).upper() != "VIA_DC":
        return resolve_leg(source, target, product, when, context, network_class)
    dc = _text(lane.get("dc_id"))
    legs = [resolve_leg(source, dc, product, when, context, network_class),
            resolve_leg(dc, target, product, when, context, network_class)]
    if any(leg.status in (ROUTE_PROHIBITED, ROUTE_CONFLICT) for leg in legs):
        worst = next(leg for leg in legs if leg.status in (ROUTE_PROHIBITED, ROUTE_CONFLICT))
        return Evidence(ROUTE, worst.value, worst.evidence, f"VIA_DC leg: {worst.basis}", worst.source, status=worst.status,
                        reasons=tuple(dict.fromkeys(r for leg in legs for r in leg.reasons)))
    if all(leg.accepted for leg in legs):
        weakest = legs[0] if legs[0].evidence != DIRECT_REAL else legs[1]
        return Evidence(ROUTE, True, weakest.evidence, "both VIA_DC legs evidenced", "|".join(filter(None, (l.source for l in legs))),
                        status=weakest.status, conditions=tuple(dict.fromkeys(c for leg in legs for c in leg.conditions)))
    return Evidence(ROUTE, None, UNKNOWN, "VIA_DC legs not both evidenced", None, status=network_class,
                    reasons=tuple(dict.fromkeys(["VIA_DC_LEG_UNVERIFIED", *(r for leg in legs for r in leg.reasons)])))


def _quantity_records(location: str, product: str, kinds: Iterable[str], context: OperationalContext) -> list[Mapping[str, Any]]:
    kinds = set(kinds)
    return [r for r in context.quantity_records
            if r["location_id"] == location and r["product_id"] == product and r["quantity_kind"] in kinds]


def _single(records: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, bool]:
    """(the record, conflict) among records at the same kind and window; differing quantities are a conflict."""
    accepted = [r for r in records if r["evidence"] in ACCEPTED_EVIDENCE] or list(records)
    if not accepted:
        return None, False
    values = {(r["quantity"], r["unit"]) for r in accepted}
    return (accepted[0], False) if len(values) == 1 else (None, True)


def _first(candidates: Sequence[Evidence]) -> Evidence | None:
    """Accepted evidence first (in the given precedence), then an estimate, else None (the lane value is used)."""
    return next((item for item in candidates if item.accepted), None) or next(iter(candidates), None)


def resolve_target_demand(lane: Mapping[str, Any], when: _date, context: OperationalContext) -> Evidence:
    """Target need: operator-declared, else actual-sales net need, else the lane's PROXY need (section 10-12)."""
    target, product = _text(lane.get("target_id")), _text(lane.get("product_id"))
    stock, stock_evidence, _ = _lane_value(lane, "target_stock", context)
    stock_unit = _lane_unit(lane, context)
    reasons: list[str] = []
    candidates: list[Evidence] = []
    declared = _quantity_records(target, product, (DECLARED_TARGET_NEED,), context)
    valid = [r for r in declared if r["start"] <= when <= r["end"]]
    if declared and not valid:
        reasons.append("DECLARED_NEED_OUTSIDE_VALIDITY" if all(r["end"] < when for r in declared) else "DECLARED_NEED_NOT_YET_VALID")
    record, conflict = _single(valid)
    if conflict:
        reasons.append("DECLARED_NEED_CONFLICT")
    elif record is not None:
        if stock_unit is not None and record["unit"] != stock_unit:
            reasons.append("DEMAND_UNIT_MISMATCH")
        else:
            candidates.append(Evidence(
                DEMAND, record["quantity"], record["evidence"], OPERATOR_DECLARED_FORMULA,
                f"operational_demand_evidence row {record['row']}", record["unit"], status=OPERATOR_DECLARED_FORMULA,
                reasons=() if record["evidence"] in ACCEPTED_EVIDENCE else ("DECLARED_NEED_ESTIMATE_ONLY",)))
    sales = _quantity_records(target, product, (ACTUAL_SALES,), context)
    window_ok = []
    for r in sales:
        if r["end"] > when:
            reasons.append("DEMAND_EVIDENCE_FROM_FUTURE")
        elif (when - r["end"]).days > context.demand_staleness_days:
            reasons.append("DEMAND_EVIDENCE_EXPIRED")
        else:
            window_ok.append(r)
    record, conflict = _single(window_ok)
    if conflict:
        reasons.append("ACTUAL_SALES_CONFLICT")
    elif record is not None:
        if stock is None or not strict_accepted(stock_evidence):
            reasons.append("TARGET_STOCK_NOT_ACCEPTED")
        elif stock_unit is not None and record["unit"] != stock_unit:
            reasons.append("DEMAND_UNIT_MISMATCH")
        else:
            days = (record["end"] - record["start"]).days + 1
            daily = record["quantity"] / days
            need = max(0.0, NEED_HORIZON_DAYS * daily - stock)
            evidence = DERIVED_REAL if record["evidence"] in (DIRECT_REAL, DERIVED_REAL) else record["evidence"]
            candidates.append(Evidence(
                DEMAND, need, evidence, f"{ACTUAL_SALES_FORMULA}: 7 x {daily:.4g}/day - stock {stock:g}",
                f"operational_demand_evidence row {record['row']}", record["unit"], status=ACTUAL_SALES_FORMULA,
                reasons=() if evidence in ACCEPTED_EVIDENCE else ("ACTUAL_SALES_ESTIMATE_ONLY",)))
    chosen = _first(candidates)
    if chosen is not None:
        return chosen if chosen.accepted else replace(chosen, reasons=tuple(dict.fromkeys([*chosen.reasons, *reasons])))
    proxy = _quantity_records(target, product, (OUTBOUND_SHIPMENT, ORDER, FORECAST, REGIONAL_FLOW, LOGISTICS_MOVEMENT), context)
    if proxy:
        reasons.append("OPERATOR_DEMAND_IS_PROXY_KIND")
    value, evidence, basis = _lane_value(lane, "target_need_7d", context)
    formula = _text(lane.get("target_need_formula")) or BENCHMARK_FORMULA
    if evidence == PROXY:
        reasons.append("TARGET_NEED_PROXY_OUTBOUND")
    elif evidence == MISSING:
        reasons.append("TARGET_NEED_MISSING")
    elif not strict_accepted(evidence):
        reasons.append(f"TARGET_NEED_{evidence}")
    return Evidence(DEMAND, value, evidence, f"{formula}: {basis}", "lane.target_need_7d", stock_unit, status=formula,
                    reasons=tuple(dict.fromkeys(reasons)))


def resolve_source_supply(lane: Mapping[str, Any], when: _date, context: OperationalContext) -> Evidence:
    """Transferable source quantity (section 6).  On-hand stock alone never verifies a surplus."""
    source, product = _text(lane.get("source_id")), _text(lane.get("product_id"))
    stock, stock_evidence, _ = _lane_value(lane, "source_stock", context)
    unit = _lane_unit(lane, context)
    reasons: list[str] = []
    candidates: list[Evidence] = []
    declared = [r for r in _quantity_records(source, product, (DECLARED_TRANSFERABLE,), context) if r["start"] <= when <= r["end"]]
    record, conflict = _single(declared)
    if conflict:
        reasons.append("DECLARED_TRANSFERABLE_CONFLICT")
    elif record is not None:
        if unit is not None and record["unit"] != unit:
            reasons.append("SUPPLY_UNIT_MISMATCH")
        elif stock is not None and strict_accepted(stock_evidence) and record["quantity"] > stock + 1e-9:
            reasons.append("DECLARED_TRANSFERABLE_EXCEEDS_ON_HAND")
        else:
            candidates.append(Evidence(
                SUPPLY, record["quantity"], record["evidence"], "OPERATOR_DECLARED_TRANSFERABLE",
                f"operational_demand_evidence row {record['row']}", record["unit"], status="OPERATOR_DECLARED_TRANSFERABLE",
                reasons=() if record["evidence"] in ACCEPTED_EVIDENCE else ("DECLARED_TRANSFERABLE_ESTIMATE_ONLY",)))
    sales = [r for r in _quantity_records(source, product, (ACTUAL_SALES,), context)
             if r["end"] <= when and (when - r["end"]).days <= context.demand_staleness_days]
    record, conflict = _single(sales)
    if record is not None and not conflict and stock is not None and strict_accepted(stock_evidence) \
            and record["evidence"] in ACCEPTED_EVIDENCE and (unit is None or record["unit"] == unit):
        daily = record["quantity"] / ((record["end"] - record["start"]).days + 1)
        value = max(0.0, stock - NEED_HORIZON_DAYS * daily)
        candidates.append(Evidence(
            SUPPLY, value, DERIVED_REAL, f"ON_HAND_MINUS_OWN_7D_SALES: {stock:g} - 7 x {daily:.4g}",
            f"operational_demand_evidence row {record['row']}", record["unit"], status="ON_HAND_MINUS_OWN_7D_SALES",
            conditions=("RESERVED_AND_IN_TRANSIT_UNKNOWN",)))
    chosen = _first(candidates)
    if chosen is not None:
        return chosen if chosen.accepted else replace(chosen, reasons=tuple(dict.fromkeys([*chosen.reasons, *reasons])))
    value, evidence, basis = _lane_value(lane, "source_surplus", context)
    if evidence == PROXY:
        reasons.append("SOURCE_SURPLUS_MEDIAN_RULE_PROXY")
    elif evidence == MISSING:
        reasons.append("SOURCE_SURPLUS_MISSING")
    elif not strict_accepted(evidence):
        reasons.append(f"SOURCE_SURPLUS_{evidence}")
    if stock is not None and strict_accepted(stock_evidence) and not strict_accepted(evidence):
        reasons.append("ONLY_ON_HAND_STOCK_OBSERVED")
    status = "LANE_DECLARED" if strict_accepted(evidence) else "MEDIAN_RULE_PROXY" if evidence == PROXY else evidence
    return Evidence(SUPPLY, value, evidence, basis, "lane.source_surplus", unit, status=status,
                    reasons=tuple(dict.fromkeys(reasons)),
                    conditions=("RESERVED_AND_IN_TRANSIT_UNKNOWN",) if strict_accepted(evidence) else ())


def resolve_target_handling(lane: Mapping[str, Any], when: _date, context: OperationalContext,
                            route: Evidence | None = None) -> Evidence:
    """Can the target receive this product (section 17)?  Holding it now is not the same as being allowed to receive."""
    target, product = _text(lane.get("target_id")), _text(lane.get("product_id"))
    declared = [r for r in _quantity_records(target, product, (DECLARED_RECEIVABLE, DECLARED_TARGET_NEED), context)
                if r["start"] <= when <= r["end"] and r["evidence"] in ACCEPTED_EVIDENCE]
    if declared:
        return Evidence(HANDLING, True, declared[0]["evidence"], f"operator {declared[0]['quantity_kind']}",
                        f"operational_demand_evidence row {declared[0]['row']}", status="OPERATOR_DECLARED_RECEIVABLE")
    if route is not None and route.accepted and route.status == ROUTE_B and route.source:
        product_rules = [r for r in context.route_records if str(r["row"]) in route.source.split("|") and r.get("product_id") == product]
        if product_rules:
            return Evidence(HANDLING, True, product_rules[0]["evidence"], "product-specific operator route rule",
                            route.source, status="OPERATOR_PRODUCT_ROUTE_ALLOWED")
    has_row = bool(lane.get("target_has_inventory_row")) and lane.get("inventory_target_matched") is not False
    stock = _finite(lane.get("target_stock"))
    if has_row:
        status = "CURRENT_STOCK_POSITIVE" if (stock or 0) > 0 else "CURRENT_ROW_ZERO_STOCK"
        return Evidence(HANDLING, True, DIRECT_REAL, "target reports the product on the decision date", "inventory row",
                        status=status, conditions=("STORAGE_CAPACITY_UNKNOWN",))
    last = context.handling_history.get((target, product))
    if last:
        return Evidence(HANDLING, None, PROXY, f"product seen at the target before (last {last}), not on the decision date",
                        "inventory history", status="HISTORICAL_ONLY", reasons=("TARGET_HANDLING_HISTORICAL_ONLY",))
    return Evidence(HANDLING, None, MISSING, "no record of the product at the target", None, status="NOT_HANDLED_OR_UNKNOWN",
                    reasons=("TARGET_HANDLING_NOT_EVIDENCED",))


def resolve_unit(lane: Mapping[str, Any], context: OperationalContext) -> Evidence:
    """Quantity unit (section 21): benchmark arithmetic inside one source is not an executable unit."""
    product = _text(lane.get("product_id"))
    status = _text(lane.get("unit_status")) or None
    declared = context.product_units.get(product)
    lane_unit = sf.normalize_unit(lane.get("quantity_unit"))
    if declared is not None:
        unit, evidence = sf.normalize_unit(declared[0]), declared[1]
        if lane_unit is not None and unit != lane_unit:
            return Evidence(UNIT, None, evidence, f"declared {unit} vs lane {lane_unit}", "product_units", unit,
                            status="UNIT_MISMATCH", reasons=("UNIT_MISMATCH",))
        return Evidence(UNIT, unit, evidence, "declared product quantity unit", "product_units", unit, status="DECLARED",
                        reasons=() if evidence in ACCEPTED_EVIDENCE else ("QUANTITY_UNIT_ESTIMATE_ONLY",))
    if status == "UNIT_MISMATCH":
        return Evidence(UNIT, None, DIRECT_REAL, "source and target declare different units", "lane", None,
                        status="UNIT_MISMATCH", reasons=("UNIT_MISMATCH",))
    if status == "MATCHED" and lane_unit is not None:
        evidence, _ = _lane_unit_evidence(lane)
        return Evidence(UNIT, lane_unit, evidence, "source and target rows declare the same unit", "lane", lane_unit,
                        status="MATCHED", reasons=() if evidence in ACCEPTED_EVIDENCE else ("QUANTITY_UNIT_ESTIMATE_ONLY",))
    reasons = ["QUANTITY_UNIT_UNKNOWN"]
    weight = context.unit_weight_status.get(product)
    if weight in ("safe", "conditional"):
        reasons.append("KG_CONVERSION_DERIVED_ONLY")
    return Evidence(UNIT, None, UNKNOWN if status == "UNDECLARED_SAME_SOURCE" else MISSING,
                    "no declared unit; quantities comparable only inside the same source" if status == "UNDECLARED_SAME_SOURCE"
                    else "unit declared on one side only or absent", "lane", None, status=status or "UNIT_UNKNOWN",
                    reasons=tuple(reasons))


def _lane_unit_evidence(lane: Mapping[str, Any]) -> tuple[str, str]:
    declared = lane.get("quantity_unit_provenance")
    return (evidence_class(declared, lane.get("quantity_unit_verification")) if _text(declared) else DIRECT_REAL), "lane"


def resolve_cost(lane: Mapping[str, Any], route: Evidence, context: OperationalContext) -> Evidence:
    """Transport cost (section 19): a computed tariff is DERIVED_REAL at most, never an actual expense."""
    qty = _finite(lane.get("recommended_qty"))
    if route.accepted and route.status == ROUTE_B and route.source:
        rules = [r for r in context.route_records if str(r["row"]) in route.source.split("|") and r.get("cost_value") is not None]
        if rules and qty is not None and rules[0]["evidence"] in ACCEPTED_EVIDENCE:
            rule = rules[0]
            value = rule["cost_value"] if rule["cost_basis"] == sf.FIXED_PER_TRIP else rule["cost_value"] * qty
            return Evidence(COST, float(value), rule["evidence"], f"operator contract {rule['cost_basis']}",
                            f"operational_route_evidence row {rule['row']}", status="CONTRACT_COST")
    value = _finite(lane.get("move_cost"))
    provenance = _text(lane.get("cost_provenance")).upper()
    evidence = evidence_class(provenance, lane.get("cost_verification")) if provenance else MISSING
    if value is None:
        return Evidence(COST, None, UNKNOWN, _text(lane.get("cost_source")) or "no computable cost", None,
                        status="COST_UNKNOWN", reasons=("COST_UNKNOWN",))
    source = _text(lane.get("cost_source")) or None
    if evidence == DERIVED_REAL:
        return Evidence(COST, value, DERIVED_REAL, "official tariff x road distance x derived unit weight", source,
                        status="CALCULATED_TARIFF", conditions=("COST_CALCULATED_NOT_ACTUAL_SPEND",))
    if evidence == PROXY:
        return Evidence(COST, value, PROXY, "official tariff with an upper-class proxy vehicle", source,
                        status="CALCULATED_TARIFF_PROXY_VEHICLE", reasons=("COST_PROXY_VEHICLE_CLASS",))
    if evidence == CONFIG:
        return Evidence(COST, value, CONFIG, "distance x 100 default", source, status="CONFIG_DEFAULT",
                        reasons=("COST_CONFIG_DEFAULT",))
    if evidence == USER_ESTIMATE:
        return Evidence(COST, value, USER_ESTIMATE, "uploaded estimate, basis undeclared", source, status="USER_ESTIMATE",
                        reasons=("COST_USER_ESTIMATE",))
    return Evidence(COST, value, evidence, "declared cost", source, status="DECLARED_COST",
                    reasons=() if evidence in ACCEPTED_EVIDENCE else (f"COST_{evidence}",))


def resolve_capacity(lane: Mapping[str, Any], route: Evidence, context: OperationalContext) -> Evidence:
    """Route / DC capacity in the lane's quantity unit (section 20).  A kg vehicle class is not a route capacity."""
    unit = _lane_unit(lane, context)
    limits: list[tuple[str, float, str]] = []
    reasons: list[str] = []
    rule_caps = []
    if route.accepted and route.status == ROUTE_B and route.source:
        rule_caps = [r for r in context.route_records if str(r["row"]) in route.source.split("|")
                     and r.get("route_capacity_qty") is not None]
    if rule_caps:
        cap = rule_caps[0]
        if unit is None or cap["quantity_unit"] != unit:
            reasons.append("ROUTE_CAPACITY_UNIT_NOT_COMPARABLE")
        else:
            limits.append(("ROUTE_CAPACITY", cap["route_capacity_qty"], cap["evidence"]))
    else:
        reasons.append("ROUTE_CAPACITY_MISSING")
    if _text(lane.get("route_type")).upper() == "VIA_DC":
        dc = context.dc_capacity.get(_text(lane.get("dc_id")))
        if dc is None:
            reasons.append("DC_CAPACITY_MISSING")
        elif unit is None or sf.normalize_unit(dc[1]) != unit:
            reasons.append("DC_CAPACITY_UNIT_NOT_COMPARABLE")
        else:
            limits.append(("DC_CAPACITY", float(dc[0]), dc[2]))
    if reasons:
        return Evidence(CAPACITY, min((l[1] for l in limits), default=None), MISSING if not limits else limits[0][2],
                        "; ".join(f"{k}={v:g}" for k, v, _ in limits) or "no capacity in the lane's quantity unit", None,
                        unit, status="CAPACITY_UNVERIFIED", reasons=tuple(reasons))
    weakest = min(limits, key=lambda item: item[1])
    evidence = next((l[2] for l in limits if l[2] not in ACCEPTED_EVIDENCE), weakest[2])
    return Evidence(CAPACITY, weakest[1], evidence, "; ".join(f"{k}={v:g}" for k, v, _ in limits), "operator evidence",
                    unit, status="CAPACITY_EVIDENCED", reasons=() if evidence in ACCEPTED_EVIDENCE else ("CAPACITY_ESTIMATE_ONLY",))


# ------------------------------------------------------------------------------------------------ eligibility


def hard_blockers(lane: Mapping[str, Any]) -> list[str]:
    """Data contradictions that make the lane impossible as stated (not missing evidence)."""
    reasons = []
    source, target, product = (_text(lane.get(k)) for k in ("source_id", "target_id", "product_id"))
    if not (source and target and product):
        reasons.append("INVALID_KEY")
    if source and source == target:
        reasons.append("SELF_MOVE")
    if bool(lane.get("is_duplicate")):
        reasons.append("DUPLICATE_LANE")
    qty = _finite(lane.get("recommended_qty"))
    if qty is None:
        reasons.append("QTY_MISSING")
    elif qty < 1 or abs(qty - round(qty)) > 1e-9:
        reasons.append("QTY_INVALID")
    stock = _finite(lane.get("source_stock"))
    gen_stock = _finite(lane.get("gen_source_stock"))
    if (stock is not None and stock <= 0) or (gen_stock is not None and gen_stock <= 0):
        reasons.append("PRODUCT_NOT_AT_SOURCE")
    if _text(lane.get("route_status")) == "ROUTE_FORBIDDEN":
        reasons.append("ROUTE_TABLE_FLAG_FORBIDDEN")
    if _text(lane.get("network_id")) and _text(lane.get("network_id")) != _text(lane.get("_context_network")):
        reasons.append("CROSS_NETWORK_LANE")
    return reasons


def _proxy_checks(lane: Mapping[str, Any], dims: Mapping[str, Evidence], qty: float | None) -> dict[str, str | None]:
    """Why a dimension fails even BENCHMARK_PROXY at the requested quantity (None = passes)."""
    out: dict[str, str | None] = {}
    route = dims[ROUTE]
    out[ROUTE] = None if (route.accepted or route.status in (ROUTE_C, ROUTE_D)) else (
        "ROUTE_NOT_IN_NETWORK_TABLE" if route.status == ROUTE_E else route.status)
    supply, demand = dims[SUPPLY], dims[DEMAND]
    out[SUPPLY] = ("SOURCE_CAP_MISSING" if _finite(supply.value) is None else
                   "PROXY_SOURCE_CAP_BELOW_REQUESTED_QTY" if qty is not None and _finite(supply.value) < qty - 1e-9 else None)
    out[DEMAND] = ("TARGET_CAP_MISSING" if _finite(demand.value) is None else
                   "PROXY_TARGET_NEED_BELOW_REQUESTED_QTY" if qty is not None and _finite(demand.value) < qty - 1e-9 else None)
    handling = dims[HANDLING]
    out[HANDLING] = None if handling.accepted else handling.status
    unit = dims[UNIT]
    out[UNIT] = None if (unit.accepted or unit.status == "UNDECLARED_SAME_SOURCE" or unit.status == "DECLARED") else unit.status
    cost = dims[COST]
    out[COST] = None if cost.value is not None and cost.evidence not in (UNKNOWN, MISSING) else "COST_NOT_COMPUTABLE"
    capacity = dims[CAPACITY]
    cap_value = _finite(capacity.value)
    out[CAPACITY] = "CAPACITY_BELOW_REQUESTED_QTY" if cap_value is not None and qty is not None and cap_value < qty - 1e-9 else None
    return out


EVALUATION_ORDER = (ROUTE, HANDLING, SUPPLY, DEMAND, UNIT, COST, CAPACITY)  # lane admissibility, then quantities


def evaluate_lane(lane: Mapping[str, Any], context: OperationalContext, *, when: Any = None) -> dict[str, Any]:
    """The section-24 candidate record: evidence per dimension, eligibility, reason codes.  Missing values stay NULL."""
    when = parse_date(when or lane.get("snapshot_date"))
    qty = _finite(lane.get("recommended_qty"))
    blockers = hard_blockers({**lane, "_context_network": context.network_id})
    if when is None:
        blockers.append("DECISION_DATE_MISSING")
        when = _date.min
    route = resolve_route(lane, when, context)
    dims = {ROUTE: route, SUPPLY: resolve_source_supply(lane, when, context),
            DEMAND: resolve_target_demand(lane, when, context),
            HANDLING: resolve_target_handling(lane, when, context, route), UNIT: resolve_unit(lane, context),
            COST: resolve_cost(lane, route, context), CAPACITY: resolve_capacity(lane, route, context)}
    if route.status == ROUTE_PROHIBITED:
        blockers.extend(r for r in route.reasons if r in (ROUTE_PROHIBITED, "ROUTE_TEMPERATURE_MISMATCH"))
    if dims[UNIT].status == "UNIT_MISMATCH":
        blockers.append("UNIT_MISMATCH")
    for dim in (SUPPLY, DEMAND, CAPACITY):
        evidence = dims[dim]
        if evidence.accepted and _finite(evidence.value) is not None and _finite(evidence.value) < 1 - 1e-9:
            blockers.append(f"VERIFIED_{dim}_BELOW_ONE_UNIT")
    required_ok = all(dims[d].accepted for d in REQUIRED_DIMENSIONS)
    conditions = sorted({c for d in DIMENSIONS for c in dims[d].conditions})
    conditional_open = [*(r for d in CONDITIONAL_DIMENSIONS if not dims[d].accepted for r in dims[d].reasons), *conditions]
    if dims[COST].accepted and dims[COST].evidence not in ACTUAL_SPEND_EVIDENCE and "COST_CALCULATED_NOT_ACTUAL_SPEND" not in conditional_open:
        conditional_open.append("COST_NOT_ACTUAL_SPEND")
    proxy = _proxy_checks(lane, dims, qty)
    proxy_caps = [v for v in (_finite(dims[SUPPLY].value), _finite(dims[DEMAND].value), _finite(dims[CAPACITY].value)) if v is not None]
    proxy_feasible = not blockers and all(value is None for value in proxy.values())
    accepted_caps = [_finite(dims[d].value) for d in (SUPPLY, DEMAND, CAPACITY)
                     if dims[d].accepted and _finite(dims[d].value) is not None]
    validated = None
    if required_ok and not blockers and qty is not None:
        validated = float(math.floor(min([qty, *accepted_caps]) + 1e-9))
    reasons = list(dict.fromkeys([*blockers, *(r for d in EVALUATION_ORDER for r in dims[d].reasons),
                                  *(v for v in proxy.values() if v)]))
    if blockers:
        status = INELIGIBLE
    elif required_ok:
        status = VERIFIED_CONDITIONAL if conditional_open else VERIFIED_OPERATIONAL
        if validated is not None and qty is not None and validated < qty:
            reasons.append("PARTIAL_QTY_ONLY")
    elif proxy_feasible:
        status = PROXY_FEASIBLE
    else:
        status = UNVERIFIED_STATUS[next(d for d in EVALUATION_ORDER if proxy[d] is not None)]
    strict = STRICT_ELIGIBLE if status in (VERIFIED_OPERATIONAL, VERIFIED_CONDITIONAL) else (
        INELIGIBLE if status == INELIGIBLE else INSUFFICIENT_OPERATIONAL_EVIDENCE)
    comparable = proxy[COST] is None and proxy[UNIT] is None and not blockers
    scope = SCOPE_STRICT if strict == STRICT_ELIGIBLE else SCOPE_BENCHMARK if comparable else SCOPE_NONE
    supply, demand = dims[SUPPLY], dims[DEMAND]
    return {
        "snapshot_date": when.isoformat() if when != _date.min else None,
        "candidate_id": _text(lane.get("candidate_id")) or _text(lane.get("lane_key")) or _text(lane.get("route_id")) or None,
        "route_id": _text(lane.get("route_id")) or None,
        "source_id": _text(lane.get("source_id")) or None, "target_id": _text(lane.get("target_id")) or None,
        "product_id": _text(lane.get("product_id")) or None, "route_type": _text(lane.get("route_type")) or None,
        "dc_id": _text(lane.get("dc_id")) or None,
        "requested_qty": qty, "validated_qty": validated,
        "proxy_feasible_qty": float(math.floor(min([qty, *proxy_caps]) + 1e-9)) if qty is not None and not blockers else None,
        "source_surplus_value": _finite(supply.value), "source_surplus_basis": supply.basis,
        "source_surplus_provenance": supply.evidence,
        "source_surplus_validation_status": "VERIFIED" if supply.accepted else supply.status,
        "target_need_value": _finite(demand.value), "target_need_basis": demand.basis, "target_need_formula": demand.status,
        "target_need_provenance": demand.evidence, "target_need_validation_status": "VERIFIED" if demand.accepted else "UNVERIFIED",
        "route_evidence": route.label(), "route_class": route.status, "route_basis": route.basis,
        "handling_evidence": dims[HANDLING].label(), "capacity_evidence": dims[CAPACITY].label(),
        "quantity_unit_evidence": dims[UNIT].label(), "cost_evidence": dims[COST].label(),
        "transport_cost": _finite(dims[COST].value),
        **{f"{d.lower()}_accepted": dims[d].accepted for d in DIMENSIONS},
        "operational_eligibility": status, "strict_eligibility": strict, "proxy_feasible": proxy_feasible,
        "blocking_reasons": "|".join(reasons) or None,
        "conditional_reasons": "|".join(dict.fromkeys(conditional_open)) or None,
        "provenance": "|".join(f"{d}={dims[d].evidence}" for d in DIMENSIONS),
        "comparison_scope": scope, "network_id": context.network_id, "evidence_version": EVIDENCE_VERSION,
    }


def evaluate_lanes(lanes: pd.DataFrame | Sequence[Mapping[str, Any]], context: OperationalContext) -> pd.DataFrame:
    rows = [evaluate_lane(lane, context) for lane in _records(lanes)]
    return pd.DataFrame(rows)


CAPS_ONLY_DIMENSIONS = (SUPPLY, DEMAND)


def caps_only_strict(lane: Mapping[str, Any], context: OperationalContext) -> bool:
    """The T1 STRICT_ACTUAL question restated in this contract (source and target caps accepted), for cross-checks."""
    when = parse_date(lane.get("snapshot_date")) or _date.min
    return resolve_source_supply(lane, when, context).accepted and resolve_target_demand(lane, when, context).accepted


# ------------------------------------------------------------------------------------------------ missing-input planner

EVIDENCE_FIELDS = {
    # field -> (dimension, key columns, role); roles: SAFETY = releases a quantity safety cap, EXECUTION = needed to
    # issue the move, COMPARISON = needed to rank / compare cost
    "verified_allowed_route": (ROUTE, ("source_id", "target_id"), "EXECUTION"),
    "actual_target_need": (DEMAND, ("snapshot_date", "target_id", "product_id"), "SAFETY"),
    "actual_source_transferable": (SUPPLY, ("snapshot_date", "source_id", "product_id"), "SAFETY"),
    "target_receivable": (HANDLING, ("target_id", "product_id"), "EXECUTION"),
    "quantity_unit": (UNIT, ("product_id",), "EXECUTION"),
    "transport_cost_basis": (COST, ("source_id", "target_id"), "COMPARISON"),
    "route_capacity": (CAPACITY, ("source_id", "target_id"), "SAFETY"),
}
FIELD_FOR_DIMENSION = {dimension: name for name, (dimension, _, _) in EVIDENCE_FIELDS.items()}
ROLE_ORDER = {"SAFETY": 0, "EXECUTION": 1, "COMPARISON": 2}


def missing_fields(record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(required fields, conditional fields) a candidate record still needs; INELIGIBLE records need none."""
    if record.get("operational_eligibility") == INELIGIBLE:
        return [], []
    required = [FIELD_FOR_DIMENSION[d] for d in REQUIRED_DIMENSIONS if not record.get(f"{d.lower()}_accepted")]
    conditional = [FIELD_FOR_DIMENSION[d] for d in CONDITIONAL_DIMENSIONS if not record.get(f"{d.lower()}_accepted")]
    return required, conditional


def plan_missing_inputs(records: pd.DataFrame) -> pd.DataFrame:
    """Greedy request order: each step adds the field that makes the most candidates STRICT_ELIGIBLE (assuming the
    supplied value verifies and does not cut the quantity), ties by candidates needing it, role, then distinct keys."""
    rows = [r for r in _records(records)]
    need = {i: set(missing_fields(r)[0]) for i, r in enumerate(rows) if r.get("operational_eligibility") != INELIGIBLE}
    conditional = {i: set(missing_fields(r)[1]) for i, r in enumerate(rows) if r.get("operational_eligibility") != INELIGIBLE}
    fields = [f for f in EVIDENCE_FIELDS if any(f in s for s in need.values())]
    wanted = {i: need[i] | conditional[i] for i in need}
    distinct = {f: len({tuple(rows[i].get(c) for c in EVIDENCE_FIELDS[f][1]) for i, s in wanted.items() if f in s})
                for f in EVIDENCE_FIELDS}
    supplied: set[str] = set()
    unlocked_before = sum(1 for s in need.values() if not s)
    out, step = [], 0
    while True:
        remaining = [f for f in fields if f not in supplied]
        if not remaining:
            break

        def gain(name: str) -> tuple:
            unlocked = sum(1 for s in need.values() if s and s <= supplied | {name})
            requiring = sum(1 for s in need.values() if name in s)
            return (unlocked, requiring, -ROLE_ORDER[EVIDENCE_FIELDS[name][2]], -distinct[name])

        best = max(remaining, key=gain)
        supplied.add(best)
        step += 1
        unlocked = sum(1 for s in need.values() if s <= supplied)
        out.append({"priority": step, "field": best, "dimension": EVIDENCE_FIELDS[best][0],
                    "key_granularity": "+".join(EVIDENCE_FIELDS[best][1]), "role": EVIDENCE_FIELDS[best][2],
                    "requirement": "REQUIRED_FOR_STRICT", "candidates_requiring": sum(1 for s in need.values() if best in s),
                    "distinct_keys_to_collect": distinct[best], "cumulative_strict_eligible_upper_bound": unlocked,
                    "newly_unlocked": unlocked - unlocked_before})
        unlocked_before = unlocked
    for name in EVIDENCE_FIELDS:
        if any(name in s for s in conditional.values()):
            step += 1
            out.append({"priority": step, "field": name, "dimension": EVIDENCE_FIELDS[name][0],
                        "key_granularity": "+".join(EVIDENCE_FIELDS[name][1]), "role": EVIDENCE_FIELDS[name][2],
                        "requirement": "CONDITIONAL_FOR_VERIFIED_OPERATIONAL",
                        "candidates_requiring": sum(1 for s in conditional.values() if name in s),
                        "distinct_keys_to_collect": distinct[name],
                        "cumulative_strict_eligible_upper_bound": None, "newly_unlocked": None})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ comparison & join guard

COMPARABLE, NOT_COMPARABLE = "COMPARABLE", "NOT_COMPARABLE"
COMPARISON_KEYS = ("eligibility_scope", "constraint_definition", "cost_basis", "quantity_basis")


def fair_cost_comparison(left: Mapping[str, Any], right: Mapping[str, Any], *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Direct cost comparison only at equal service, eligibility scope, constraints, cost basis and quantity basis."""
    reasons = []
    if abs(float(left.get("service") or 0.0) - float(right.get("service") or 0.0)) > tolerance:
        reasons.append("SERVICE_NOT_EQUAL")
    for key in COMPARISON_KEYS:
        if left.get(key) != right.get(key):
            reasons.append(f"{key.upper()}_DIFFERS")
    if not left.get("eligible_count") or not right.get("eligible_count"):
        reasons.append("NO_ELIGIBLE_CANDIDATES")
    if reasons:
        return {"status": NOT_COMPARABLE, "cost_delta": None, "claim": None, "reason_codes": reasons}
    delta = float(left["cost"]) - float(right["cost"])
    actual = left.get("eligibility_scope") == SCOPE_STRICT and left.get("cost_basis") == "ACTUAL_SPEND"
    return {"status": COMPARABLE, "cost_delta": round(delta, 6),
            "claim": "ACTUAL_COST_DIFFERENCE" if actual else "COMPUTED_BENCHMARK_DIFFERENCE_NOT_ACTUAL_SAVING",
            "reason_codes": []}


@dataclass(frozen=True)
class DataScope:
    dataset_id: str
    network_id: str
    product_namespace: str
    location_namespace: str
    unit: str | None = None
    date: str | None = None


VERIFIED_KEY_MAPPINGS: dict[tuple[str, str], str] = {
    # (namespace, namespace) -> rule.  Verified on the data by operational_evidence_validation (ids and names agree).
    ("suhyup_center_code", "suhyup_canonical_location_id"): "strip leading zeros (000156 -> 156)",
}


def join_guard(left: DataScope, right: DataScope) -> dict[str, Any]:
    """Two datasets may be joined only inside one operational network on verified keys, units and dates."""
    reasons = []
    if left.network_id != right.network_id:
        reasons.append("DIFFERENT_OPERATIONAL_NETWORK")
    if left.dataset_id != right.dataset_id and left.network_id == right.network_id:
        pass  # several files of one business may be joined on verified keys
    for kind in ("product_namespace", "location_namespace"):
        a, b = getattr(left, kind), getattr(right, kind)
        if a != b and (a, b) not in VERIFIED_KEY_MAPPINGS and (b, a) not in VERIFIED_KEY_MAPPINGS:
            reasons.append(f"UNVERIFIED_{kind.upper()}_MAPPING")
    if left.unit and right.unit and sf.normalize_unit(left.unit) != sf.normalize_unit(right.unit):
        reasons.append("UNIT_MISMATCH")
    if left.date and right.date and left.date != right.date:
        reasons.append("DATE_MISMATCH")
    return {"allowed": not reasons, "reason_codes": reasons}


def normalize_location_id(value: Any, namespace: str) -> str:
    text = _text(value)
    if namespace == "suhyup_canonical_location_id" and text.isdigit():
        return text.lstrip("0") or "0"
    return text


# ------------------------------------------------------------------------------------------------ generator re-run with a need


def regenerated_choices(universe: pd.DataFrame, need_column: str) -> pd.DataFrame:
    """The generator's one-target choice, quantity rule and candidate_score re-applied with another need column, before
    the MAX_CANDIDATES cut (research only).  With ``gen_target_need`` it reproduces the production order."""
    from services import candidate_generator as cg

    frame = universe[universe["route_status"].isin(("ROUTE_NETWORK_TABLE", "ROUTE_VIA_DC_DERIVED", "ROUTE_FORBIDDEN"))
                     & universe["recommended_qty"].notna()].copy()
    if frame.empty:
        return frame
    frame["_need"] = pd.to_numeric(frame[need_column], errors="coerce").fillna(0.0)
    kept = []
    for _, group in frame.groupby("inventory_row", sort=False):
        # candidate_generator: targets.sort(reverse=True) on (need, -distance | -cost, target id)
        ordered = sorted(group.to_dict("records"),
                         key=lambda r: (r["_need"], float(r["target_sort_value"]), str(r["target_id"])), reverse=True)
        kept.append(ordered[0])
    kept.sort(key=lambda r: int(r["inventory_row"]))
    seen, chosen = set(), []
    for r in kept:  # dedupe on the kept target, then quantity, then (legacy mode) positive saving
        key = (r["product_id"], r["source_id"], r["target_id"])
        if key in seen:
            continue
        seen.add(key)
        need = r["_need"]
        source_rule = float(r["gen_surplus_qty_rule"])
        need_rule = max(1.0, need if need > 0 else source_rule)
        r["recommended_qty"] = int(max(1, min(source_rule, need_rule, float(r["gen_source_stock"]), float(r["move_cap"]))))
        if bool(r.get("real_transport_mode")):
            r["gen_saving"] = 0.0
        else:
            cost = _finite(r.get("route_estimated_cost")) or (_finite(r.get("distance_km")) or 0.0) * 100.0
            r["gen_saving"] = r["recommended_qty"] * float(r["unit_price"]) * cg._DISPOSAL_FRACTION - cost
            if r["gen_saving"] <= 0:
                continue
        chosen.append(r)
    savings = [r["gen_saving"] for r in chosen] or [0.0]
    s_min, s_range = min(savings), max(1.0, max(savings) - min(savings))
    for r in chosen:
        median = float(r["gen_median_stock"])
        need_score = cg._clamp((r["_need"] / max(1.0, median)) * 100.0)
        saving_score = cg._clamp((r["gen_saving"] - s_min) / s_range * 100.0)
        r["need_score"], r["saving_score"] = need_score, round(saving_score, 1)
        r["candidate_score"] = round(cg._clamp(
            0.25 * r["expiry_score"] + 0.20 * r["surplus_score"] + 0.20 * need_score
            + 0.20 * saving_score + 0.10 * r["route_score"] + 0.05 * r["distance_score"]), 1)
    chosen.sort(key=lambda r: r["candidate_score"], reverse=True)  # stable, as production
    pool = pd.DataFrame(chosen).drop(columns="_need")
    pool["regenerated_rank"] = range(1, len(pool) + 1)
    pool["regenerated_need_column"] = need_column
    return pool.reset_index(drop=True)


def regenerate_with_need(universe: pd.DataFrame, need_column: str, limit: int | None = None) -> pd.DataFrame:
    """The generator re-run with ``need_column`` and its MAX_CANDIDATES cut (``limit`` overrides it)."""
    from services import candidate_generator as cg

    choices = regenerated_choices(universe, need_column)
    return choices.head(cg.MAX_CANDIDATES if limit is None else limit).reset_index(drop=True)


def summarize_statuses(records: pd.DataFrame, column: str = "operational_eligibility") -> dict[str, int]:
    counts = Counter(records[column]) if not records.empty else Counter()
    return {status: int(counts.get(status, 0)) for status in ELIGIBILITY_STATUSES}


def dimension_flag_counts(records: pd.DataFrame) -> dict[str, int]:
    """Non-exclusive: candidates whose dimension is not accepted (a candidate counts once per dimension)."""
    if records.empty:
        return {UNVERIFIED_STATUS[d]: 0 for d in DIMENSIONS}
    return {UNVERIFIED_STATUS[d]: int((~records[f"{d.lower()}_accepted"].astype(bool)).sum()) for d in DIMENSIONS}


def reason_counts(records: pd.DataFrame) -> dict[str, int]:
    counts: Counter = Counter()
    for text in records.get("blocking_reasons", pd.Series(dtype=object)).dropna():
        counts.update(str(text).split("|"))
    return dict(sorted(counts.items()))


def group_counts(values: Iterable[Any]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))
