"""Target need contract and operational evidence intake (P1-C).

Shadow and research only.  Every target need the code computes gets an explicit formula id, version, inputs, unit,
horizon and evidence class; on-hand stock is kept apart from transferable stock; and optional operator evidence
(workbook sheets or CSV: ``operational_route_evidence`` / ``operational_demand_evidence`` /
``operational_inventory_evidence``) passes key, date, unit, origin, verification and conflict checks before any value
reaches the P1-B evaluator (``services.operational_evidence``) or a T1 STRICT_ACTUAL cap.

Nothing here changes candidate_generator, the production pool, Varo Final, Top-5, T1 / T2 / T3 / P1 / P1-B, the MILP,
Seller Loss or Forecast.  SAMPLE and TEST rows never become operational evidence.  A program cannot authenticate the
person behind a declared verification: the strongest row status is OPERATOR_VERIFIED, meaning the verification is
declared completely (origin, level, verifier, date, source), not that anyone's authority was checked.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from services import operational_evidence as oe
from services import shared_feasibility_selection as sf
from services.optimality_gap_service import _first_number, _target_shortage
from services.seller_loss_engine import _config, declared_provenance, demand_provenance

CONTRACT_VERSION = "target-need-contract-v0.1"
HORIZON_DAYS = oe.NEED_HORIZON_DAYS

# ------------------------------------------------------------------------------------------------ formula registry

GENERATOR_PROXY, BENCHMARK_PROXY = "GENERATOR_PROXY", "BENCHMARK_PROXY"
PIPELINE_SHORTAGE_7D, UPLOADED_SHORTAGE_INPUT = "PIPELINE_SHORTAGE_7D", "UPLOADED_SHORTAGE_INPUT"
SELLER_LOSS_HORIZON_NEED = "SELLER_LOSS_HORIZON_NEED"
OPERATOR_CONFIRMED_NEED, DERIVED_FROM_OBSERVED_DEMAND = "OPERATOR_CONFIRMED_NEED", "DERIVED_FROM_OBSERVED_DEMAND"
NOT_OBSERVED_FORMULA = "NOT_OBSERVED"

# intended uses: what a value of the formula is allowed to drive
GENERATOR_RANKING, BENCHMARK_CAP, PIPELINE_CAP = "GENERATOR_RANKING", "BENCHMARK_CAP", "PIPELINE_CAP"
SELLER_LOSS_INTERNAL, STRICT_CAP = "SELLER_LOSS_INTERNAL", "STRICT_CAP"

FORMULAS: dict[str, dict[str, Any]] = {
    GENERATOR_PROXY: {
        "version": "v1", "p1b_alias": oe.GENERATOR_FORMULA,
        "expression": "max(0, median_stock - target_stock) + 7 x target_daily_demand",
        "code": "services/candidate_generator.py generate_candidates (need = ...; targets loop); "
                "P1 research lane enumeration (reached via candidate_generation_validation.cgr) gen_target_need",
        "inputs": "normalized upload inventory: stock_qty; demand = first of avg_daily_sales | sales_qty | demand_qty "
                  "(Suhyup: one day's outbound_qty; avg_daily_sales is an alias copy); median over the upload rows of "
                  "the product (Suhyup: (center, product) pairs reported under >1 state code are excluded)",
        "missing_target_row": "the generator substitutes stock = median and demand = 0 (need 0); the contract keeps "
                              "the need NULL and records the substitute",
        "unit": "the upload's stock unit (Suhyup count unit UNKNOWN)", "period": "one snapshot day",
        "demand_signal_type": "upload demand column (Suhyup OUTBOUND_SHIPMENT)", "demand_signal_period_days": 1,
        "horizon_days": HORIZON_DAYS, "target_stock_policy": "CROSS_NODE_MEDIAN_STOCK (a rule, not a target)",
        "intended_use": GENERATOR_RANKING, "default_evidence": oe.PROXY, "active": True,
        "purpose": "target choice (highest need), quantity rule max(1, need) and need_score of the production generator",
        "limitations": ["gross requirement: stock above the median is never netted against the 7-day outflow",
                        "the median is a balancing rule, not a stock target", "outbound is not retail demand",
                        "never a cap in any selector"]},
    BENCHMARK_PROXY: {
        "version": "v1", "p1b_alias": oe.BENCHMARK_FORMULA,
        "expression": "max(0, median_stock + 7 x target_daily_outbound - target_stock)",
        "code": "services/suhyup_algorithm_revalidation.py enrich_inventory_constraints (target_need_7d)",
        "inputs": "actual inventory flow summed over state codes per (date, center, product); median over the six "
                  "network nodes that report the product; demand = outbound_qty of the same day",
        "missing_target_row": "NULL (no cap)", "unit": "flow stock unit (Suhyup UNKNOWN)", "period": "one snapshot day",
        "demand_signal_type": oe.OUTBOUND_SHIPMENT, "demand_signal_period_days": 1, "horizon_days": HORIZON_DAYS,
        "target_stock_policy": "CROSS_NODE_MEDIAN_STOCK (a rule, not a target)",
        "intended_use": BENCHMARK_CAP, "default_evidence": oe.PROXY, "active": True,
        "purpose": "TARGET_NEED cap of T1 BENCHMARK_PROXY, the T3 / P1 MILP and the Varo Final ordered selection",
        "limitations": ["order-up-to rule over a cross-node median", "outbound is shipments, not sales",
                        "sums stock over state codes (frozen, dried, chilled rows of one product code)"]},
    PIPELINE_SHORTAGE_7D: {
        "version": "v1", "p1b_alias": None,
        "expression": "max(0, demand_7d - target_stock); demand_7d = demand_qty | demand_forecast_7d | sales_7d | "
                      "avg_daily_sales x 7 | sales_qty x 7 | sales_30d / 30 x 7",
        "code": "services/optimality_gap_service.py _target_shortage (build_constraint_context target_caps); "
                "services/inventory_transition_service.py _target_shortage_limit",
        "inputs": "uploaded inventory row of the target (no median)", "missing_target_row": "NULL (no cap)",
        "unit": "the workbook's quantity unit (config quantity_unit or undeclared)", "period": "the inventory snapshot",
        "demand_signal_type": "uploaded demand / sales column", "demand_signal_period_days": None,
        "horizon_days": HORIZON_DAYS, "target_stock_policy": "NONE (cover 7 days of demand)",
        "intended_use": PIPELINE_CAP, "default_evidence": "PROXY when the demand column is a proxy, else USER_ESTIMATE",
        "active": True,
        "purpose": "in-app optimality gap and T1 pipeline TARGET_NEED cap (shared-feasibility-v1); transition engine "
                   "target_shortage_limit, which Seller Loss reads as target_need_cap",
        "limitations": ["no target stock level", "the demand column decides the meaning (sales, forecast or proxy)"]},
    UPLOADED_SHORTAGE_INPUT: {
        "version": "v1", "p1b_alias": None,
        "expression": "uploaded shortage_qty | demand_shortage | target_shortage_qty | unmet_demand",
        "code": "services/optimality_gap_service.py _target_shortage (explicit field first)",
        "inputs": "a workbook column", "missing_target_row": "NULL", "unit": "the workbook's quantity unit",
        "period": "the inventory snapshot", "demand_signal_type": None, "demand_signal_period_days": None,
        "horizon_days": None, "target_stock_policy": "UNDECLARED", "intended_use": PIPELINE_CAP,
        "default_evidence": oe.USER_ESTIMATE, "active": True,
        "purpose": "an uploaded shortage used as the T1 pipeline TARGET_NEED cap (T1 labels it USER_INPUT)",
        "limitations": ["no formula, horizon or verification is declared with the value"]},
    SELLER_LOSS_HORIZON_NEED: {
        "version": "v1", "p1b_alias": None,
        "expression": "max(0, r_t x (H - transit_days) - S_t)",
        "code": "services/seller_loss_engine.py _transfer_strategy (engine_need)",
        "inputs": "Seller Loss inputs: target daily demand, target stock, decision horizon, transit time",
        "missing_target_row": "TRANSFER unavailable", "unit": "Seller Loss quantity unit", "period": "decision horizon H",
        "demand_signal_type": "target_daily_demand input", "demand_signal_period_days": 1, "horizon_days": "H - transit",
        "target_stock_policy": "NONE", "intended_use": SELLER_LOSS_INTERNAL, "default_evidence": "per Seller Loss input",
        "active": True,
        "purpose": "Seller Loss TRANSFER quantity cap (frozen; documented, never re-used here)",
        "limitations": ["horizon differs from the 7-day caps; never compared with them"]},
    OPERATOR_CONFIRMED_NEED: {
        "version": "v1", "p1b_alias": oe.OPERATOR_DECLARED_FORMULA,
        "expression": "operator REQUIRED_NEED quantity for (location, product, decision date)",
        "code": "this module; operational_evidence.resolve_target_demand (OPERATOR_DECLARED_NEED)",
        "inputs": "operational_demand_evidence row quantity_kind DECLARED_TARGET_NEED (quantity_meaning REQUIRED_NEED)",
        "missing_target_row": "not needed", "unit": "declared on the row", "period": "the row's validity window",
        "demand_signal_type": "OPERATOR_REQUIREMENT", "demand_signal_period_days": None, "horizon_days": "declared",
        "target_stock_policy": "OPERATOR", "intended_use": STRICT_CAP, "default_evidence": oe.VERIFIED_OPERATOR_INPUT,
        "active": True, "purpose": "the STRICT_ACTUAL target cap when declared and verified",
        "limitations": ["accepted only from ACTUAL_OPERATION rows with a complete verification declaration",
                        "a receivable capacity, order or preference is never a need"]},
    DERIVED_FROM_OBSERVED_DEMAND: {
        "version": "v1", "p1b_alias": oe.ACTUAL_SALES_FORMULA,
        "expression": "max(0, 7 x observed daily sales - target stock)",
        "code": "operational_evidence.resolve_target_demand (ACTUAL_SALES_NET_NEED)",
        "inputs": "ACTUAL_SALES rows of the same location, product and unit, window ending within 7 days of the decision",
        "missing_target_row": "NULL", "unit": "declared on the sales row", "period": "the observation window",
        "demand_signal_type": oe.ACTUAL_SALES, "demand_signal_period_days": "window length", "horizon_days": HORIZON_DAYS,
        "target_stock_policy": "NONE (cover 7 days of observed sales)", "intended_use": STRICT_CAP,
        "default_evidence": oe.DERIVED_REAL, "active": False,
        "activation_rule": "inactive until verified ACTUAL_OPERATION ACTUAL_SALES rows exist for the network; no "
                           "Suhyup source holds retail sales",
        "purpose": "a STRICT_ACTUAL target cap from observed sales", "limitations": ["needs actual retail sales"]},
}
P1B_ALIASES = {entry["p1b_alias"]: formula for formula, entry in FORMULAS.items() if entry.get("p1b_alias")}
NEED_STATUS_FORMULA = {  # P1-B DEMAND dimension status -> formula id
    oe.OPERATOR_DECLARED_FORMULA: OPERATOR_CONFIRMED_NEED, oe.ACTUAL_SALES_FORMULA: DERIVED_FROM_OBSERVED_DEMAND,
    oe.BENCHMARK_FORMULA: BENCHMARK_PROXY, oe.GENERATOR_FORMULA: GENERATOR_PROXY,
}

# ------------------------------------------------------------------------------------------------ meanings & statuses

REQUIRED_NEED, ORDER_QTY, RECEIVABLE_CAPACITY, PREFERENCE = "REQUIRED_NEED", "ORDER_QTY", "RECEIVABLE_CAPACITY", "PREFERENCE"
HANDLING_ALLOWED, TRANSFERABLE_QTY, OBSERVED_SALES, PROXY_SIGNAL = (
    "HANDLING_ALLOWED", "TRANSFERABLE_QTY", "OBSERVED_SALES", "PROXY_SIGNAL")
QUANTITY_MEANINGS = {
    REQUIRED_NEED: "필요량: quantity the location needs (the only meaning that is a target need)",
    ORDER_QTY: "주문량: quantity requested or ordered; not a need, not receipts (PROXY)",
    RECEIVABLE_CAPACITY: "수용 가능량: the most the location may additionally receive; an inflow limit, not a need",
    PREFERENCE: "선호량: a preferred quantity; informational only, never a cap",
    HANDLING_ALLOWED: "the location may receive the product (no quantity)",
    TRANSFERABLE_QTY: "quantity that may leave the source",
    OBSERVED_SALES: "observed retail sales",
    PROXY_SIGNAL: "outbound / forecast / regional flow: a demand proxy",
}
DECLARED_PREFERENCE = "DECLARED_PREFERENCE"
PROXY_KINDS = (oe.OUTBOUND_SHIPMENT, oe.FORECAST, oe.REGIONAL_FLOW, oe.LOGISTICS_MOVEMENT)

VERIFIED, PROXY_STATUS, ESTIMATE_STATUS = "VERIFIED", "PROXY", "USER_ESTIMATE"
NOT_OBSERVED, CONFLICT, EXPIRED, NOT_ACTIVATED = "NOT_OBSERVED", "CONFLICT", "EXPIRED", "NOT_ACTIVATED"
OBSERVED, OBSERVED_ZERO, SUBSTITUTED = "OBSERVED", "OBSERVED_ZERO", "SUBSTITUTED"

# record origins and verification (sections 22-24)
SAMPLE, TEST, SCENARIO, ACTUAL_OPERATION, UNDECLARED = "SAMPLE", "TEST", "SCENARIO", "ACTUAL_OPERATION", "UNDECLARED"
RECORD_ORIGINS = (SAMPLE, TEST, SCENARIO, ACTUAL_OPERATION)
OBSERVED_RECORD, INPUT_CONFIRMED, OPERATIONALLY_APPROVED = "OBSERVED_RECORD", "INPUT_CONFIRMED", "OPERATIONALLY_APPROVED"
VERIFICATION_LEVELS = {
    OBSERVED_RECORD: "the value is a system / observation record (e.g. a WMS or POS export)",
    INPUT_CONFIRMED: "a person re-checked the entered value against its source",
    OPERATIONALLY_APPROVED: "an operator approved the value for use in moves (route, cap, receivable quantity)",
}
OPERATOR_VERIFIED, OPERATOR_UNVERIFIED = "OPERATOR_VERIFIED", "OPERATOR_UNVERIFIED"
SAMPLE_NOT_USABLE, TEST_ONLY, SCENARIO_ONLY = "SAMPLE_NOT_USABLE", "TEST_ONLY", "SCENARIO_ONLY"
AUTHORITY_NOTE = ("the program checks that a verification is declared with origin, level, verifier, date and source; "
                  "it cannot authenticate the verifier or their authority")
# a verification level accepts a field only for these field kinds
ACCEPTED_LEVELS = {
    REQUIRED_NEED: {INPUT_CONFIRMED, OPERATIONALLY_APPROVED},
    RECEIVABLE_CAPACITY: {OPERATIONALLY_APPROVED},
    HANDLING_ALLOWED: {INPUT_CONFIRMED, OPERATIONALLY_APPROVED},
    TRANSFERABLE_QTY: {OBSERVED_RECORD, INPUT_CONFIRMED, OPERATIONALLY_APPROVED},
    OBSERVED_SALES: {OBSERVED_RECORD, INPUT_CONFIRMED},
    ORDER_QTY: set(), PREFERENCE: set(), PROXY_SIGNAL: set(),
    "ROUTE_ALLOWED": {OPERATIONALLY_APPROVED},
    "ROUTE_PROHIBITED": {OBSERVED_RECORD, INPUT_CONFIRMED, OPERATIONALLY_APPROVED},
    "ON_HAND": {OBSERVED_RECORD, INPUT_CONFIRMED},
    "RESERVED": {OBSERVED_RECORD, INPUT_CONFIRMED},
    "AVAILABLE_FOR_TRANSFER": {OBSERVED_RECORD, INPUT_CONFIRMED},
    "TRANSFER_CAP": {OPERATIONALLY_APPROVED},
    "DATASET_UNIT": {INPUT_CONFIRMED, OPERATIONALLY_APPROVED},
    "HANDLING": {INPUT_CONFIRMED, OPERATIONALLY_APPROVED},
}

# sheet statuses (section 21)
MISSING_EVIDENCE, VALIDATION_ERROR, VALIDATED_INPUT, NO_USABLE_EVIDENCE = (
    "MISSING_EVIDENCE", "VALIDATION_ERROR", "VALIDATED_INPUT", "NO_USABLE_EVIDENCE")
ROW_ACCEPTED, ROW_ACCEPTED_TEST, ROW_ESTIMATE, ROW_REJECTED = (
    "ACCEPTED_ACTUAL", "ACCEPTED_TEST_LINKAGE_ONLY", "USABLE_NOT_ACCEPTED", "REJECTED")
NO_EVIDENCE, TEST_LINKAGE = "NO_EVIDENCE", "TEST_LINKAGE_ONLY"

ROUTE_SHEET, DEMAND_SHEET, INVENTORY_SHEET = (
    "operational_route_evidence", "operational_demand_evidence", "operational_inventory_evidence")
LOAD_ERROR_COLUMN = "__load_error__"  # services.data_loader marker of an unreadable evidence sheet
EVIDENCE_SHEETS = (ROUTE_SHEET, DEMAND_SHEET, INVENTORY_SHEET)
TRUST_COLUMNS = ("record_origin", "verification_status", "verification_level", "verified_by", "verified_at", "dataset_id")
SHEET_REQUIRED = {  # network_id may be blank inside a workbook (the workbook's own network)
    ROUTE_SHEET: ("source_id", "target_id", "allowed", "effective_from", "evidence_class", "evidence_source"),
    DEMAND_SHEET: ("location_id", "product_id", "quantity_kind", "observation_start", "evidence_class", "data_source"),
    INVENTORY_SHEET: ("location_id", "product_id", "effective_date", "evidence_class", "evidence_source"),
}
INVENTORY_QUANTITIES = {  # column -> field kind
    "on_hand_qty": "ON_HAND", "reserved_qty": "RESERVED", "available_for_transfer_qty": "AVAILABLE_FOR_TRANSFER",
    "transfer_cap_qty": "TRANSFER_CAP",
}
SHEET_OPTIONAL = {
    ROUTE_SHEET: ("network_id", *oe.ROUTE_EVIDENCE_OPTIONAL, *TRUST_COLUMNS),
    DEMAND_SHEET: ("network_id", *oe.DEMAND_EVIDENCE_OPTIONAL, "quantity_meaning", "valid_until", *TRUST_COLUMNS),
    INVENTORY_SHEET: ("network_id", "valid_until", *INVENTORY_QUANTITIES, "quantity_unit", "dataset_unit_confirmed",
                      "lot_restriction", "source_expected_demand_qty", "source_expected_demand_days", "handling_allowed",
                      "handling_conditions", "note", *TRUST_COLUMNS),
}

# source availability kinds (section 13)
SOURCE_SURPLUS_PROXY = "SOURCE_SURPLUS_PROXY"
OBSERVED_AVAILABLE_FOR_TRANSFER = "OBSERVED_AVAILABLE_FOR_TRANSFER"
OPERATOR_APPROVED_TRANSFER_CAP = "OPERATOR_APPROVED_TRANSFER_CAP"
UNRESERVED_ON_HAND = "UNRESERVED_ON_HAND"
ON_HAND_ONLY, PROXY_ONLY, MISSING_STATUS = "ON_HAND_ONLY", "PROXY_ONLY", "MISSING"
SOURCE_AVAILABILITY_KINDS = {
    SOURCE_SURPLUS_PROXY: "max(0, on-hand - cross-node median): a balancing rule (PROXY), never transferable stock",
    OBSERVED_AVAILABLE_FOR_TRANSFER: "available-for-transfer quantity from a system record (reserved, held and "
                                     "in-transit stock already excluded by the source system)",
    OPERATOR_APPROVED_TRANSFER_CAP: "the most an operator approved to leave the source",
    UNRESERVED_ON_HAND: "on-hand minus reserved: an upper bound only, not a transfer allowance",
}

# missing-input report names (section 30) over the P1-B evidence fields
MISSING_FIELD_NAMES = {
    "actual_source_transferable": "missing_actual_surplus", "actual_target_need": "missing_actual_target_need",
    "verified_allowed_route": "missing_route_approval", "quantity_unit": "missing_quantity_unit",
    "route_capacity": "missing_capacity", "transport_cost_basis": "missing_cost_basis",
    "target_receivable": "missing_target_handling",
}
MISSING_FLAGS = (*MISSING_FIELD_NAMES.values(), "conflicting_evidence", "expired_evidence")
REQUEST_TIER = {  # section 31 criterion 1: without the input a move is unsafe / conditionally unsafe / only uncomparable
    "missing_route_approval": "SAFETY_REQUIRED", "missing_actual_surplus": "SAFETY_REQUIRED",
    "missing_actual_target_need": "SAFETY_REQUIRED", "missing_target_handling": "SAFETY_REQUIRED",
    "missing_quantity_unit": "SAFETY_REQUIRED", "missing_capacity": "CONDITIONAL_SAFETY",
    "missing_cost_basis": "COMPARISON_ONLY",
}
TIER_ORDER = {"SAFETY_REQUIRED": 0, "CONDITIONAL_SAFETY": 1, "COMPARISON_ONLY": 2}
FIELD_FOR_FLAG = {flag: name for name, flag in MISSING_FIELD_NAMES.items()}


def _text(value: Any) -> str:
    return oe._text(value)


def _finite(value: Any) -> float | None:
    return oe._finite(value)


def _iso(value: _date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


# ------------------------------------------------------------------------------------------------ target need record


@dataclass(frozen=True)
class TargetNeed:
    """One target need value with everything needed to know what it means (section 6)."""

    formula_id: str
    network_id: str
    location_id: str
    product_id: str
    as_of_date: str | None
    target_need_value: float | None
    evidence: str
    dataset_id: str = ""
    observation_start: str | None = None
    observation_end: str | None = None
    valid_until: str | None = None
    planning_horizon_days: int | None = HORIZON_DAYS
    current_stock: float | None = None
    current_stock_status: str = NOT_OBSERVED
    current_stock_evidence: str = oe.MISSING
    demand_signal: float | None = None
    demand_signal_type: str | None = None
    demand_signal_period_days: int | None = None
    target_level: float | None = None
    target_stock_policy: str | None = None
    quantity_unit: str | None = None
    quantity_meaning: str = REQUIRED_NEED
    substitute_value: float | None = None
    evidence_source: str | None = None
    validation_status: str = PROXY_STATUS
    reason_codes: tuple[str, ...] = ()

    @property
    def formula_version(self) -> str:
        return str(FORMULAS.get(self.formula_id, {}).get("version", "n/a"))

    @property
    def intended_use(self) -> str:
        return str(FORMULAS.get(self.formula_id, {}).get("intended_use", "NONE"))

    @property
    def accepted(self) -> bool:
        return self.target_need_value is not None and self.validation_status == VERIFIED and oe.strict_accepted(self.evidence)

    def signature(self) -> dict[str, Any]:
        return {"formula_id": self.formula_id, "formula_version": self.formula_version,
                "horizon_days": self.planning_horizon_days, "quantity_unit": self.quantity_unit,
                "as_of_date": self.as_of_date, "quantity_meaning": self.quantity_meaning}

    def as_dict(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "network_id": self.network_id, "location_id": self.location_id,
                "product_id": self.product_id, "as_of_date": self.as_of_date, "observation_start": self.observation_start,
                "observation_end": self.observation_end, "valid_until": self.valid_until,
                "planning_horizon_days": self.planning_horizon_days, "current_stock": _round(self.current_stock),
                "current_stock_status": self.current_stock_status, "current_stock_evidence": self.current_stock_evidence,
                "current_stock_unit": self.quantity_unit, "demand_signal": _round(self.demand_signal),
                "demand_signal_type": self.demand_signal_type, "demand_signal_unit": self.quantity_unit,
                "demand_signal_period_days": self.demand_signal_period_days, "target_level": _round(self.target_level),
                "target_stock_policy": self.target_stock_policy, "target_need_value": _round(self.target_need_value),
                "target_need_unit": self.quantity_unit, "quantity_meaning": self.quantity_meaning,
                "substitute_value": _round(self.substitute_value), "formula_id": self.formula_id,
                "formula_version": self.formula_version, "provenance": self.evidence,
                "base_provenance": oe.base_provenance(self.evidence), "evidence_source": self.evidence_source,
                "validation_status": self.validation_status, "intended_use": self.intended_use,
                "accepted_for_strict": self.accepted, "reason_codes": "|".join(self.reason_codes) or None}


def _stock_status(stock: float | None) -> str:
    if stock is None:
        return NOT_OBSERVED
    return OBSERVED_ZERO if abs(stock) <= 1e-12 else OBSERVED


def generator_proxy_need(*, network_id: str, location_id: str, product_id: str, as_of: str | None, median: float | None,
                         stock: float | None, demand: float | None, unit: str | None = None, dataset_id: str = "",
                         stock_evidence: str = oe.DIRECT_REAL, demand_type: str = oe.OUTBOUND_SHIPMENT) -> TargetNeed:
    """The production generator's need.  A target without a row is NULL here; the generator's substitute is kept apart."""
    common = dict(formula_id=GENERATOR_PROXY, network_id=network_id, location_id=location_id, product_id=product_id,
                  as_of_date=as_of, dataset_id=dataset_id, observation_start=as_of, observation_end=as_of,
                  target_level=median, target_stock_policy=FORMULAS[GENERATOR_PROXY]["target_stock_policy"],
                  quantity_unit=unit, demand_signal_type=demand_type, demand_signal_period_days=1)
    if median is None:
        return TargetNeed(target_need_value=None, evidence=oe.MISSING, validation_status=NOT_OBSERVED,
                          reason_codes=("PRODUCT_MEDIAN_UNAVAILABLE",), **common)
    if stock is None:
        substitute = oe.generator_need(median, median, 0.0)
        return TargetNeed(target_need_value=None, evidence=oe.MISSING, validation_status=NOT_OBSERVED,
                          substitute_value=substitute, current_stock_status=NOT_OBSERVED,
                          reason_codes=("TARGET_NOT_OBSERVED", "GENERATOR_SUBSTITUTES_MEDIAN_STOCK_AND_ZERO_DEMAND"),
                          **common)
    value = oe.generator_need(median, stock, demand or 0.0)
    reasons = ["GENERATOR_RANKING_ONLY_NEVER_A_CAP", "DEMAND_SIGNAL_IS_PROXY"]
    if demand is None:
        reasons.append("DEMAND_NOT_OBSERVED_TREATED_AS_ZERO_BY_GENERATOR")
    return TargetNeed(target_need_value=value, evidence=oe.PROXY, current_stock=stock, current_stock_status=_stock_status(stock),
                      current_stock_evidence=stock_evidence, demand_signal=demand, validation_status=PROXY_STATUS,
                      reason_codes=tuple(reasons), **common)


def benchmark_proxy_need(*, network_id: str, location_id: str, product_id: str, as_of: str | None, median: float | None,
                         stock: float | None, outbound: float | None, unit: str | None = None, dataset_id: str = "",
                         stock_evidence: str = oe.DIRECT_REAL) -> TargetNeed:
    common = dict(formula_id=BENCHMARK_PROXY, network_id=network_id, location_id=location_id, product_id=product_id,
                  as_of_date=as_of, dataset_id=dataset_id, observation_start=as_of, observation_end=as_of,
                  target_level=median, target_stock_policy=FORMULAS[BENCHMARK_PROXY]["target_stock_policy"],
                  quantity_unit=unit, demand_signal_type=oe.OUTBOUND_SHIPMENT, demand_signal_period_days=1)
    if stock is None or median is None:
        return TargetNeed(target_need_value=None, evidence=oe.MISSING, validation_status=NOT_OBSERVED,
                          reason_codes=("TARGET_NOT_OBSERVED",) if stock is None else ("PRODUCT_MEDIAN_UNAVAILABLE",), **common)
    value = oe.benchmark_need(median, stock, outbound or 0.0)
    return TargetNeed(target_need_value=value, evidence=oe.PROXY, current_stock=stock, current_stock_status=_stock_status(stock),
                      current_stock_evidence=stock_evidence, demand_signal=outbound, validation_status=PROXY_STATUS,
                      reason_codes=("TARGET_NEED_PROXY_OUTBOUND", "MEDIAN_IS_A_RULE"), **common)


def pipeline_need(row: Mapping[str, Any] | None, *, network_id: str, location_id: str, product_id: str,
                  as_of: str | None, unit: str | None = None, dataset_id: str = "") -> TargetNeed:
    """The in-app target cap (optimality_gap_service._target_shortage) named: explicit uploaded shortage or 7-day shortage."""
    common = dict(network_id=network_id, location_id=location_id, product_id=product_id, as_of_date=as_of,
                  dataset_id=dataset_id, quantity_unit=unit)
    if not row:
        return TargetNeed(formula_id=NOT_OBSERVED_FORMULA, target_need_value=None, evidence=oe.MISSING,
                          validation_status=NOT_OBSERVED, reason_codes=("TARGET_NOT_OBSERVED",), **common)
    value, basis = _target_shortage(row)
    stock, stock_field = _first_number(row, sf.STOCK_FIELDS)
    stock_evidence = oe.evidence_class(declared_provenance(row, str(stock_field), "USER_INPUT")[0]) if stock is not None else oe.MISSING
    if value is None:
        return TargetNeed(formula_id=NOT_OBSERVED_FORMULA, target_need_value=None, evidence=oe.MISSING,
                          validation_status=NOT_OBSERVED, current_stock=stock, current_stock_status=_stock_status(stock),
                          current_stock_evidence=stock_evidence, reason_codes=("STOCK_OR_DEMAND_NOT_OBSERVED",), **common)
    if not basis.startswith("max("):
        label, _ = declared_provenance(row, basis, "USER_INPUT")
        evidence = oe.evidence_class(label, row.get(f"{basis}_verification"))
        return TargetNeed(formula_id=UPLOADED_SHORTAGE_INPUT, target_need_value=value, evidence=evidence,
                          planning_horizon_days=None, current_stock=stock, current_stock_status=_stock_status(stock),
                          current_stock_evidence=stock_evidence, evidence_source=f"inventory.{basis}",
                          validation_status=VERIFIED if oe.strict_accepted(evidence) else ESTIMATE_STATUS,
                          reason_codes=() if oe.strict_accepted(evidence) else ("UPLOADED_SHORTAGE_UNVERIFIED",), **common)
    demand = demand_provenance(row)
    evidence = oe.PROXY if demand == "PROXY" else (
        oe.DERIVED_REAL if demand == "DERIVED_REAL" and stock_evidence == oe.DIRECT_REAL else oe.USER_ESTIMATE)
    reasons = {oe.PROXY: ("DEMAND_SIGNAL_IS_PROXY",), oe.USER_ESTIMATE: ("UPLOADED_DEMAND_UNVERIFIED",)}.get(evidence, ())
    return TargetNeed(formula_id=PIPELINE_SHORTAGE_7D, target_need_value=value, evidence=evidence, current_stock=stock,
                      current_stock_status=_stock_status(stock), current_stock_evidence=stock_evidence,
                      demand_signal_type=basis, evidence_source="inventory row",
                      target_stock_policy=FORMULAS[PIPELINE_SHORTAGE_7D]["target_stock_policy"],
                      validation_status={oe.PROXY: PROXY_STATUS, oe.DERIVED_REAL: VERIFIED}.get(evidence, ESTIMATE_STATUS),
                      reason_codes=reasons, **common)


def need_comparability(left: TargetNeed, right: TargetNeed) -> dict[str, Any]:
    """Two needs are compared as one quantity only with the same formula, version, horizon, unit, date and meaning."""
    reasons = []
    a, b = left.signature(), right.signature()
    for key, code in (("formula_id", "FORMULA_DIFFERS"), ("formula_version", "FORMULA_VERSION_DIFFERS"),
                      ("horizon_days", "HORIZON_DIFFERS"), ("as_of_date", "AS_OF_DATE_DIFFERS"),
                      ("quantity_meaning", "QUANTITY_MEANING_DIFFERS")):
        if a[key] != b[key]:
            reasons.append(code)
    if a["quantity_unit"] is None or b["quantity_unit"] is None:
        reasons.append("QUANTITY_UNIT_UNKNOWN")
    elif a["quantity_unit"] != b["quantity_unit"]:
        reasons.append("QUANTITY_UNIT_DIFFERS")
    if left.target_need_value is None or right.target_need_value is None:
        reasons.append("VALUE_NOT_OBSERVED")
    return {"status": oe.NOT_COMPARABLE if reasons else oe.COMPARABLE, "reason_codes": reasons}


def constraint_signature(formula_id: str, *, cap_provenance: Iterable[str], window: tuple[str | None, str | None],
                         horizon_days: Any = HORIZON_DAYS, quantity_unit: str | None = None,
                         source_cap: str = SOURCE_SURPLUS_PROXY) -> dict[str, Any]:
    """Section 38: what a benchmark result's target / source caps mean.  Different signatures are never compared as
    optimality claims."""
    body = {"contract_version": CONTRACT_VERSION, "formula_id": formula_id,
            "formula_version": FORMULAS.get(formula_id, {}).get("version"), "source_cap": source_cap,
            "cap_provenance": sorted(set(cap_provenance)), "window": list(window), "horizon_days": horizon_days,
            "quantity_unit": quantity_unit}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return {**body, "signature": digest}


def signatures_comparable(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    reasons = [f"{key.upper()}_DIFFERS" for key in ("formula_id", "formula_version", "source_cap", "cap_provenance",
                                                    "window", "horizon_days", "quantity_unit")
               if left.get(key) != right.get(key)]
    return {"status": oe.NOT_COMPARABLE if reasons else oe.COMPARABLE, "reason_codes": reasons,
            "optimality_comparison_allowed": not reasons}


# ------------------------------------------------------------------------------------------------ source availability


def source_availability(*, network_id: str, location_id: str, product_id: str, as_of: str | None,
                        dataset_on_hand: float | None = None, dataset_on_hand_evidence: str = oe.MISSING,
                        dataset_unit: str | None = None, surplus_proxy: float | None = None,
                        surplus_proxy_basis: str | None = None, operator: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Section 11-13: on-hand, reserved, available-for-transfer, operator cap and the PROXY surplus kept apart.

    ``operator`` holds the accepted operator facts valid on ``as_of``: field kind -> (value, unit, evidence, source).
    Only an accepted available-for-transfer or approved cap yields a transferable quantity (the smaller of them, both
    being upper bounds, further bounded by accepted on-hand stock); the PROXY surplus never joins them.
    """
    operator = dict(operator or {})
    reasons, conditions, conflicts = [], [], []
    facts = {kind: operator.get(kind) for kind in ("ON_HAND", "RESERVED", "AVAILABLE_FOR_TRANSFER", "TRANSFER_CAP")}
    units = {fact[1] for fact in facts.values() if fact is not None and fact[1]}
    if len(units) > 1:
        conflicts.append("OPERATOR_UNITS_DIFFER")
    operator_unit = next(iter(units)) if len(units) == 1 else None
    on_hand = facts["ON_HAND"][0] if facts["ON_HAND"] else None
    dataset_accepted = dataset_on_hand is not None and oe.strict_accepted(dataset_on_hand_evidence)
    if on_hand is not None and dataset_accepted:
        if dataset_unit is not None and operator_unit is not None and dataset_unit == operator_unit:
            if abs(on_hand - dataset_on_hand) > 1e-9:
                conflicts.append("ON_HAND_SOURCES_DISAGREE")
        else:
            conditions.append("ON_HAND_UNITS_NOT_COMPARABLE")
    bound_on_hand = on_hand if on_hand is not None else (
        dataset_on_hand if dataset_accepted and (operator_unit is None or dataset_unit == operator_unit) else None)
    reserved = facts["RESERVED"][0] if facts["RESERVED"] else None
    available = facts["AVAILABLE_FOR_TRANSFER"][0] if facts["AVAILABLE_FOR_TRANSFER"] else None
    cap = facts["TRANSFER_CAP"][0] if facts["TRANSFER_CAP"] else None
    if bound_on_hand is not None and reserved is not None and reserved > bound_on_hand + 1e-9:
        conflicts.append("RESERVED_EXCEEDS_ON_HAND")
    if bound_on_hand is not None and available is not None:
        if available > bound_on_hand + 1e-9:
            conflicts.append("AVAILABLE_EXCEEDS_ON_HAND")
        elif reserved is not None and available > bound_on_hand - reserved + 1e-9:
            conflicts.append("AVAILABLE_EXCEEDS_UNRESERVED")
    unreserved = (bound_on_hand - reserved) if bound_on_hand is not None and reserved is not None else None
    bounds = [(kind, value) for kind, value in ((OBSERVED_AVAILABLE_FOR_TRANSFER, available),
                                                (OPERATOR_APPROVED_TRANSFER_CAP, cap)) if value is not None]
    transferable, kind = None, None
    if conflicts:
        status = CONFLICT
    elif bounds:
        kind, transferable = min(bounds, key=lambda item: (item[1], item[0]))
        if bound_on_hand is not None and transferable > bound_on_hand + 1e-9:
            conditions.append("TRANSFER_CAP_ABOVE_ON_HAND_BOUNDED")
            transferable = bound_on_hand
        if cap is not None and available is None:
            conditions.append("AVAILABLE_FOR_TRANSFER_NOT_OBSERVED_CAP_ONLY")
        status = VERIFIED
    else:
        reasons += ["AVAILABLE_FOR_TRANSFER_NOT_OBSERVED", "OPERATOR_TRANSFER_CAP_MISSING"]
        if reserved is None:
            reasons.append("RESERVED_QTY_UNKNOWN")
        if surplus_proxy is not None:
            reasons.append("SOURCE_SURPLUS_MEDIAN_RULE_PROXY")
        if dataset_accepted or on_hand is not None:
            reasons.append("ONLY_ON_HAND_STOCK_OBSERVED")
            status = ON_HAND_ONLY
        else:
            status = PROXY_ONLY if surplus_proxy is not None else MISSING_STATUS
    if operator.get("LOT_RESTRICTION"):
        conditions.append("LOT_RESTRICTION_DECLARED")
    expected = operator.get("SOURCE_EXPECTED_DEMAND")
    return {
        "network_id": network_id, "location_id": location_id, "product_id": product_id, "as_of_date": as_of,
        "dataset_on_hand_qty": _round(dataset_on_hand), "dataset_on_hand_evidence": dataset_on_hand_evidence,
        "dataset_on_hand_status": _stock_status(dataset_on_hand), "dataset_quantity_unit": dataset_unit,
        "on_hand_qty": _round(on_hand), "reserved_qty": _round(reserved),
        "available_for_transfer_qty": _round(available), "operator_transfer_cap_qty": _round(cap),
        "unreserved_on_hand_qty": _round(unreserved), "lot_restriction": operator.get("LOT_RESTRICTION"),
        "source_expected_demand_qty": _round(expected[0]) if expected else None,
        "source_surplus_proxy_qty": _round(surplus_proxy), "source_surplus_proxy_basis": surplus_proxy_basis,
        "quantity_unit": operator_unit or dataset_unit,
        "transferable_qty": _round(transferable), "transferable_kind": kind,
        "transferable_evidence": (operator[{OBSERVED_AVAILABLE_FOR_TRANSFER: "AVAILABLE_FOR_TRANSFER",
                                            OPERATOR_APPROVED_TRANSFER_CAP: "TRANSFER_CAP"}[kind]][2] if kind else None),
        "validation_status": status, "conflict_codes": "|".join(conflicts) or None,
        "reason_codes": "|".join(dict.fromkeys(reasons)) or None, "conditions": "|".join(dict.fromkeys(conditions)) or None,
        "operator_sources": "|".join(sorted({str(f[3]) for f in operator.values() if isinstance(f, tuple) and len(f) > 3})) or None,
    }


# ------------------------------------------------------------------------------------------------ intake (sections 20-24)


@dataclass(frozen=True)
class KeyUniverse:
    """The identifiers a network knows.  ``None`` sets are not checkable (reported, never guessed)."""

    network_id: str
    dataset_id: str = ""
    locations: frozenset[str] | None = None
    products: frozenset[str] | None = None
    location_namespace: str = ""
    workbook: bool = False

    def location(self, value: Any) -> str:
        return oe.normalize_location_id(value, self.location_namespace)


def _origin(row: Mapping[str, Any]) -> str:
    if oe.is_sample_row(row):
        return SAMPLE
    text = _text(row.get("record_origin")).upper()
    return text if text in RECORD_ORIGINS else UNDECLARED


def _trust(row: Mapping[str, Any], field_kind: str, source_column: str) -> dict[str, Any]:
    """Row trust for one field kind: origin, declared verification completeness and the resulting acceptance."""
    origin = _origin(row)
    declared = oe.evidence_class(row.get("evidence_class"), row.get("verification_status"))
    level = _text(row.get("verification_level")).upper() or None
    flag = _text(row.get("verification_status")).upper()
    source = _text(row.get(source_column))
    reasons = []
    if origin == SAMPLE:
        return {"origin": origin, "status": SAMPLE_NOT_USABLE, "accepted": False, "level": level, "declared": declared,
                "reasons": ["SAMPLE_ROW_NOT_USABLE"]}
    if origin == ACTUAL_OPERATION and source.upper().startswith(("SAMPLE", "TEST")):
        return {"origin": origin, "status": "REJECTED", "accepted": False, "level": level, "declared": declared,
                "reasons": ["ORIGIN_SOURCE_MISMATCH"]}
    if origin == UNDECLARED:
        reasons.append("RECORD_ORIGIN_UNDECLARED")
    if declared not in oe.ACCEPTED_EVIDENCE:
        reasons.append(f"EVIDENCE_CLASS_{declared}")
    if declared == oe.VERIFIED_OPERATOR_INPUT and flag not in oe.VERIFIED_FLAGS:
        reasons.append("VERIFICATION_NOT_DECLARED")
    if level is None:
        reasons.append("VERIFICATION_LEVEL_MISSING")
    elif level not in VERIFICATION_LEVELS:
        reasons.append("VERIFICATION_LEVEL_UNKNOWN")
    elif level not in ACCEPTED_LEVELS.get(field_kind, set()):
        reasons.append(f"VERIFICATION_LEVEL_INSUFFICIENT_FOR_{field_kind}")
    if not _text(row.get("verified_by")):
        reasons.append("VERIFIER_MISSING")
    if oe.parse_date(row.get("verified_at")) is None:
        reasons.append("VERIFIED_AT_MISSING")
    if not source:
        reasons.append("EVIDENCE_SOURCE_MISSING")
    complete = not [r for r in reasons if r != "RECORD_ORIGIN_UNDECLARED"]
    if origin == ACTUAL_OPERATION and complete:
        status, accepted = OPERATOR_VERIFIED, True
    elif origin == TEST and complete:
        status, accepted = TEST_ONLY, False
    elif origin == SCENARIO:
        status, accepted = SCENARIO_ONLY, False
    else:
        status, accepted = OPERATOR_UNVERIFIED, False
    if origin == TEST:
        reasons.append("TEST_INPUT_NOT_OPERATIONAL_EVIDENCE")
    return {"origin": origin, "status": status, "accepted": accepted, "complete": complete, "level": level,
            "declared": declared, "reasons": reasons}


def _passed(row: Mapping[str, Any], trust: Mapping[str, Any], *, allow_test: bool, proxy_kind: bool = False) -> dict[str, Any]:
    """The row as the P1-B normalizers read it: accepted rows keep their class, everything else becomes an estimate."""
    out = dict(row)
    if trust["accepted"] or (allow_test and trust["status"] == TEST_ONLY):
        out["evidence_class"] = trust["declared"]
        out["verification_status"] = "VERIFIED"
    elif proxy_kind or trust["declared"] == oe.PROXY:
        out["evidence_class"], out["verification_status"] = oe.PROXY, "UNVERIFIED"
    else:
        out["evidence_class"], out["verification_status"] = oe.USER_ESTIMATE, "UNVERIFIED"
    return out


def _row_state(trust: Mapping[str, Any], *, allow_test: bool) -> str:
    if trust["accepted"]:
        return ROW_ACCEPTED
    if trust["status"] == TEST_ONLY and allow_test:
        return ROW_ACCEPTED_TEST
    return ROW_ESTIMATE


def _window(start: Any, end: Any) -> tuple[_date | None, _date | None, bool]:
    first, last = oe.parse_date(start), oe.parse_date(end)
    bad = first is None or (_text(end) and last is None) or (last is not None and first is not None and last < first)
    return first, last, bool(bad)


def _key_issues(universe: KeyUniverse, row: Mapping[str, Any], location_columns: Sequence[str],
                product: bool = True) -> tuple[dict[str, str], list[str]]:
    keys, issues = {}, []
    network = _text(row.get("network_id"))
    if not network and universe.workbook:
        keys["network_id"] = universe.network_id
    elif not network:
        issues.append("MISSING_REQUIRED_FIELD:network_id")
    elif network != universe.network_id:
        issues.append("CROSS_NETWORK_EVIDENCE_REJECTED")
    else:
        keys["network_id"] = network
    dataset = _text(row.get("dataset_id"))
    if dataset and universe.dataset_id and dataset != universe.dataset_id:
        issues.append("CROSS_DATASET_EVIDENCE_REJECTED")
    for column in location_columns:
        value = universe.location(row.get(column))
        keys[column] = value
        if value and universe.locations is not None and value not in universe.locations:
            issues.append(f"UNKNOWN_LOCATION_ID:{column}")
    if product:
        value = _text(row.get("product_id"))
        keys["product_id"] = value
        if value and universe.products is not None and value not in universe.products:
            issues.append("UNKNOWN_PRODUCT_ID")
    return keys, issues


def _missing_columns(frame: pd.DataFrame, sheet: str) -> list[str]:
    return [name for name in SHEET_REQUIRED[sheet] if name not in frame.columns]


def _frame(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(list(value))
    raise TypeError(f"unsupported evidence container {type(value).__name__}")


@dataclass
class Intake:
    """Validated operator evidence of one network: per-sheet status, per-row validation and normalized records."""

    universe: KeyUniverse
    allow_test: bool
    sheets: dict[str, dict[str, Any]] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    routes: list[dict[str, Any]] = field(default_factory=list)       # P1-B route rule records (oe format)
    quantities: list[dict[str, Any]] = field(default_factory=list)   # P1-B quantity records (oe format)
    receivable_caps: list[dict[str, Any]] = field(default_factory=list)
    preferences: list[dict[str, Any]] = field(default_factory=list)
    inventory: list[dict[str, Any]] = field(default_factory=list)    # per accepted inventory field
    units: list[dict[str, Any]] = field(default_factory=list)        # dataset unit confirmations
    handling: list[dict[str, Any]] = field(default_factory=list)     # handling_allowed declarations
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scope(self) -> str:
        states = {row["row_state"] for row in self.rows}
        if ROW_ACCEPTED_TEST in states:
            return TEST_LINKAGE
        return ACTUAL_OPERATION if ROW_ACCEPTED in states else NO_EVIDENCE

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(row["row_state"] for row in self.rows).items()))


def _record_row(intake: Intake, sheet: str, index: int, trust: Mapping[str, Any] | None, issues: Sequence[str],
                keys: Mapping[str, Any], fields: Mapping[str, Any], state: str) -> None:
    intake.rows.append({
        "sheet": sheet, "row": index, "row_state": state, "record_origin": (trust or {}).get("origin"),
        "verification": (trust or {}).get("status"), "verification_level": (trust or {}).get("level"),
        "declared_evidence_class": (trust or {}).get("declared"),
        "issues": "|".join(dict.fromkeys([*issues, *((trust or {}).get("reasons") or [])])) or None,
        "network_id": keys.get("network_id"), "location_id": keys.get("location_id"), "source_id": keys.get("source_id"),
        "target_id": keys.get("target_id"), "product_id": keys.get("product_id"),
        "fields": "|".join(f"{k}={v}" for k, v in fields.items() if v is not None) or None,
        "authority_check": "NOT_PERFORMED_BY_PROGRAM"})


def _intake_routes(intake: Intake, frame: pd.DataFrame) -> None:
    universe = intake.universe
    for index, row in enumerate(oe._records(frame)):
        keys, issues = _key_issues(universe, row, ("source_id", "target_id"), product=bool(_text(row.get("product_id"))))
        allowed = oe._truth(row.get("allowed"))
        kind = "ROUTE_PROHIBITED" if allowed is False else "ROUTE_ALLOWED"
        trust = _trust(row, kind, "evidence_source")
        if keys.get("source_id") and keys.get("source_id") == keys.get("target_id"):
            issues.append("SELF_ROUTE")
        if trust["status"] in (SAMPLE_NOT_USABLE, "REJECTED") or issues:
            _record_row(intake, ROUTE_SHEET, index, trust, issues, keys, {"allowed": allowed}, ROW_REJECTED)
            continue
        passed = _passed({**row, **keys}, trust, allow_test=intake.allow_test)
        records, normalizer_issues = oe.normalize_route_evidence([passed], network_id=universe.network_id)
        codes = [issue["code"] + (f":{issue['detail']}" if issue.get("detail") else "") for issue in normalizer_issues]
        if not records:
            _record_row(intake, ROUTE_SHEET, index, trust, codes, keys, {"allowed": allowed}, ROW_REJECTED)
            continue
        state = _row_state(trust, allow_test=intake.allow_test)
        for record in records:
            record.update(row=f"{ROUTE_SHEET}:{index}", origin=trust["origin"], row_state=state)
            intake.routes.append(record)
        _record_row(intake, ROUTE_SHEET, index, trust, codes, keys,
                    {"allowed": allowed, "effective_from": _iso(records[0]["effective_from"]),
                     "effective_to": _iso(records[0]["effective_to"]), "route_capacity_qty": records[0]["route_capacity_qty"],
                     "quantity_unit": records[0]["quantity_unit"], "cost_basis": records[0]["cost_basis"]}, state)


KIND_MEANING = {
    oe.DECLARED_TARGET_NEED: REQUIRED_NEED, oe.ORDER: ORDER_QTY, DECLARED_PREFERENCE: PREFERENCE,
    oe.DECLARED_TRANSFERABLE: TRANSFERABLE_QTY, oe.ACTUAL_SALES: OBSERVED_SALES,
    **{kind: PROXY_SIGNAL for kind in PROXY_KINDS},
}


def _meaning(kind: str, quantity: float | None) -> str | None:
    if kind == oe.DECLARED_RECEIVABLE:
        return RECEIVABLE_CAPACITY if quantity is not None else HANDLING_ALLOWED
    return KIND_MEANING.get(kind)


def _intake_demand(intake: Intake, frame: pd.DataFrame) -> None:
    universe = intake.universe
    for index, row in enumerate(oe._records(frame)):
        keys, issues = _key_issues(universe, row, ("location_id",))
        kind = _text(row.get("quantity_kind")).upper()
        quantity = _finite(row.get("quantity"))
        inferred = _meaning(kind, quantity)
        declared_meaning = _text(row.get("quantity_meaning")).upper() or None
        if inferred is None:
            issues.append(f"UNKNOWN_QUANTITY_KIND:{kind}")
        elif declared_meaning is not None and declared_meaning != inferred:
            issues.append(f"QUANTITY_MEANING_KIND_MISMATCH:{declared_meaning}!={inferred}")
        meaning = declared_meaning or inferred
        if quantity is not None and quantity < 0:
            issues.append("INVALID_QUANTITY")
        if quantity is not None and sf.normalize_unit(row.get("quantity_unit")) is None:
            issues.append("QUANTITY_UNIT_MISSING")
        start, end, bad = _window(row.get("observation_start"), row.get("observation_end"))
        until = oe.parse_date(row.get("valid_until"))
        if bad or (_text(row.get("valid_until")) and until is None) or (until is not None and start is not None and until < start):
            issues.append("INVALID_OBSERVATION_WINDOW")
        trust = _trust(row, meaning or "UNKNOWN", "data_source")
        if trust["status"] in (SAMPLE_NOT_USABLE, "REJECTED") or issues:
            _record_row(intake, DEMAND_SHEET, index, trust, issues, keys, {"quantity_kind": kind, "quantity": quantity},
                        ROW_REJECTED)
            continue
        state = _row_state(trust, allow_test=intake.allow_test)
        last = min(d for d in (end or start, until) if d is not None)
        fields = {"quantity_kind": kind, "quantity_meaning": meaning, "quantity": quantity,
                  "quantity_unit": sf.normalize_unit(row.get("quantity_unit")), "window": f"{_iso(start)}..{_iso(last)}"}
        if meaning == PREFERENCE:  # informational only, never handed to a selector
            intake.preferences.append({"row": f"{DEMAND_SHEET}:{index}", **keys, "quantity": quantity,
                                       "unit": fields["quantity_unit"], "start": start, "end": last})
            _record_row(intake, DEMAND_SHEET, index, trust, ["PREFERENCE_INFORMATIONAL_ONLY"], keys, fields, state)
            continue
        proxy_kind = meaning in (ORDER_QTY, PROXY_SIGNAL)
        if proxy_kind:  # an order or shipment quantity is a demand proxy whatever its label (section 9)
            issues.append("PROXY_KIND_NEVER_ACTUAL_DEMAND")
            if trust["declared"] in oe.ACCEPTED_EVIDENCE:
                issues.append("PROXY_LABELLED_AS_ACTUAL")
            trust, state = {**trust, "accepted": False, "status": OPERATOR_UNVERIFIED}, ROW_ESTIMATE
        passed = _passed({**row, **keys, "quantity_kind": kind, "observation_end": _iso(last)}, trust,
                         allow_test=intake.allow_test, proxy_kind=proxy_kind)
        records, normalizer_issues = oe.normalize_demand_evidence([passed], network_id=universe.network_id)
        codes = [*issues, *(issue["code"] for issue in normalizer_issues)]
        if not records:
            _record_row(intake, DEMAND_SHEET, index, trust, codes, keys, fields, ROW_REJECTED)
            continue
        for record in records:
            record.update(row=f"{DEMAND_SHEET}:{index}", origin=trust["origin"], row_state=state, meaning=meaning)
            if meaning == RECEIVABLE_CAPACITY:
                intake.receivable_caps.append(dict(record))
            intake.quantities.append(record)
        _record_row(intake, DEMAND_SHEET, index, trust, codes, keys, fields, state)


def _intake_inventory(intake: Intake, frame: pd.DataFrame) -> None:
    universe = intake.universe
    for index, row in enumerate(oe._records(frame)):
        keys, issues = _key_issues(universe, row, ("location_id",))
        issues += [f"MISSING_REQUIRED_FIELD:{name}" for name in SHEET_REQUIRED[INVENTORY_SHEET] if not _text(row.get(name))]
        start, until, bad = _window(row.get("effective_date"), row.get("valid_until"))
        if bad:
            issues.append("INVALID_EFFECTIVE_WINDOW")
        unit = sf.normalize_unit(row.get("quantity_unit"))
        values = {column: _finite(row.get(column)) for column in INVENTORY_QUANTITIES}
        for column, value in values.items():
            if _text(row.get(column)) and value is None:
                issues.append(f"INVALID_QUANTITY:{column}")
            elif value is not None and value < 0:
                issues.append(f"NEGATIVE_QUANTITY:{column}")
        handling = oe._truth(row.get("handling_allowed")) if _text(row.get("handling_allowed")) else None
        unit_flag = oe._truth(row.get("dataset_unit_confirmed")) if _text(row.get("dataset_unit_confirmed")) else None
        if any(v is not None for v in values.values()) and unit is None:
            issues.append("QUANTITY_UNIT_MISSING")
        if unit_flag and unit is None:
            issues.append("DATASET_UNIT_FLAG_WITHOUT_UNIT")
        if not any(v is not None for v in values.values()) and handling is None and not unit_flag:
            issues.append("NO_EVIDENCE_FIELD")
        base = _trust(row, "ON_HAND", "evidence_source")
        if base["status"] in (SAMPLE_NOT_USABLE, "REJECTED") or issues:
            _record_row(intake, INVENTORY_SHEET, index, base, issues, keys, values, ROW_REJECTED)
            continue
        window_end = until or start
        common = {"row": f"{INVENTORY_SHEET}:{index}", **keys, "unit": unit, "start": start, "end": window_end,
                  "source": _text(row.get("evidence_source"))}
        states = []
        for column, kind in INVENTORY_QUANTITIES.items():
            if values[column] is None:
                continue
            trust = _trust(row, kind, "evidence_source")
            state = _row_state(trust, allow_test=intake.allow_test)
            states.append(state)
            intake.inventory.append({**common, "field": kind, "value": values[column], "origin": trust["origin"],
                                     "row_state": state, "evidence": trust["declared"],
                                     "lot_restriction": _text(row.get("lot_restriction")) or None,
                                     "source_expected_demand_qty": _finite(row.get("source_expected_demand_qty")),
                                     "trust_reasons": trust["reasons"]})
        if handling is not None:
            trust = _trust(row, "HANDLING", "evidence_source")
            state = _row_state(trust, allow_test=intake.allow_test)
            states.append(state)
            intake.handling.append({**common, "allowed": handling, "origin": trust["origin"], "row_state": state,
                                    "evidence": trust["declared"], "conditions": _text(row.get("handling_conditions")) or None})
        if unit_flag:
            trust = _trust(row, "DATASET_UNIT", "evidence_source")
            state = _row_state(trust, allow_test=intake.allow_test)
            states.append(state)
            intake.units.append({**common, "origin": trust["origin"], "row_state": state, "evidence": trust["declared"]})
        overall = (ROW_ACCEPTED if ROW_ACCEPTED in states else ROW_ACCEPTED_TEST if ROW_ACCEPTED_TEST in states
                   else ROW_ESTIMATE)
        partial = [f"FIELD_NOT_ACCEPTED:{item['field']}" for item in intake.inventory
                   if item["row"] == common["row"] and item["row_state"] == ROW_ESTIMATE] if overall != ROW_ESTIMATE else []
        _record_row(intake, INVENTORY_SHEET, index, base, partial, keys,
                    {**values, "quantity_unit": unit, "handling_allowed": handling, "dataset_unit_confirmed": unit_flag,
                     "window": f"{_iso(start)}..{_iso(window_end)}"}, overall)


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return (a["end"] is None or b["start"] <= a["end"]) and (b["end"] is None or a["start"] <= b["end"])


def _usable(item: Mapping[str, Any]) -> bool:
    return item.get("row_state") in (ROW_ACCEPTED, ROW_ACCEPTED_TEST)


def _pairwise_conflicts(items: Sequence[Mapping[str, Any]], scope_of, value_of, code: str) -> list[dict[str, Any]]:
    """Same scope, overlapping windows, different values: CONFLICT (never first or latest wins)."""
    groups: defaultdict[tuple, list] = defaultdict(list)
    for item in items:
        groups[scope_of(item)].append(item)
    out = []
    for scope, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for a_index, a in enumerate(members):
            for b in members[a_index + 1:]:
                if value_of(a) != value_of(b) and _overlap(a, b):
                    out.append({"code": code, "scope": "/".join(str(s) for s in scope), "rows": f"{a['row']}|{b['row']}",
                                "values": f"{value_of(a)}|{value_of(b)}",
                                "window_start": _iso(max(a["start"], b["start"])),
                                "window_end": _iso(min(d for d in (a["end"], b["end"]) if d is not None))
                                if a["end"] is not None or b["end"] is not None else None})
    return out


def _route_window(record: Mapping[str, Any]) -> dict[str, Any]:
    return {**record, "start": record["effective_from"], "end": record["effective_to"]}


def detect_conflicts(intake: Intake) -> list[dict[str, Any]]:
    """Section 28 conflicts among usable evidence.  Fields that measure different things (on-hand vs available) are
    checked for consistency inside source_availability, never treated as rival values."""
    conflicts = []
    inventory = [i for i in intake.inventory if _usable(i)]
    for kind in INVENTORY_QUANTITIES.values():
        conflicts += _pairwise_conflicts([i for i in inventory if i["field"] == kind],
                                         lambda i: (i["location_id"], i["product_id"]), lambda i: (i["value"], i["unit"]),
                                         f"{kind}_CONFLICT")
    quantities = [{**q, "start": q["start"], "end": q["end"]} for q in intake.quantities if _usable(q)]
    for meaning, code in ((REQUIRED_NEED, "TARGET_NEED_CONFLICT"), (RECEIVABLE_CAPACITY, "RECEIVABLE_CAPACITY_CONFLICT"),
                          (TRANSFERABLE_QTY, "DECLARED_TRANSFERABLE_CONFLICT"), (OBSERVED_SALES, "ACTUAL_SALES_CONFLICT")):
        conflicts += _pairwise_conflicts([q for q in quantities if q.get("meaning") == meaning],
                                         lambda q: (q["location_id"], q["product_id"]), lambda q: (q["quantity"], q["unit"]), code)
    routes = [_route_window(r) for r in intake.routes if _usable(r)]
    conflicts += _pairwise_conflicts(routes, lambda r: (r["source_id"], r["target_id"], r.get("product_id") or r.get("product_group") or "*"),
                                     lambda r: r["allowed"], "ROUTE_ALLOWED_PROHIBITED_CONFLICT")
    conflicts += _pairwise_conflicts([r for r in routes if r["allowed"]],
                                     lambda r: (r["source_id"], r["target_id"], r.get("product_id") or r.get("product_group") or "*"),
                                     lambda r: (r.get("route_capacity_qty"), r.get("quantity_unit"), r.get("cost_basis"), r.get("cost_value")),
                                     "ROUTE_TERMS_CONFLICT")
    conflicts += _pairwise_conflicts([u for u in intake.units if _usable(u)], lambda u: (u["product_id"],),
                                     lambda u: u["unit"], "DATASET_UNIT_CONFLICT")
    conflicts += _pairwise_conflicts([h for h in intake.handling if _usable(h)], lambda h: (h["location_id"], h["product_id"]),
                                     lambda h: h["allowed"], "HANDLING_CONFLICT")
    return conflicts


def intake_evidence(sources: Mapping[str, Any], universe: KeyUniverse, *, allow_test: bool = False) -> Intake:
    """Validate the three optional evidence tables.  Never raises on bad input: a broken sheet is VALIDATION_ERROR."""
    intake = Intake(universe=universe, allow_test=allow_test)
    handlers = {ROUTE_SHEET: _intake_routes, DEMAND_SHEET: _intake_demand, INVENTORY_SHEET: _intake_inventory}
    for sheet, handler in handlers.items():
        try:
            frame = _frame(sources.get(sheet))
        except TypeError as exc:
            intake.sheets[sheet] = {"status": VALIDATION_ERROR, "rows": 0, "detail": str(exc)}
            continue
        if frame is None:
            intake.sheets[sheet] = {"status": MISSING_EVIDENCE, "rows": 0, "detail": "sheet / file not provided"}
            continue
        if LOAD_ERROR_COLUMN in frame.columns:  # data_loader could not read the sheet
            intake.sheets[sheet] = {"status": VALIDATION_ERROR, "rows": 0,
                                    "detail": f"sheet unreadable: {frame[LOAD_ERROR_COLUMN].iloc[0] if len(frame) else ''}"}
            continue
        missing = _missing_columns(frame, sheet) if not frame.empty else []
        if missing:
            intake.sheets[sheet] = {"status": VALIDATION_ERROR, "rows": int(len(frame)),
                                    "detail": "missing required columns: " + ", ".join(missing)}
            continue
        before = len(intake.rows)
        collected = ("routes", "quantities", "receivable_caps", "preferences", "inventory", "units", "handling")
        sizes = {name: len(getattr(intake, name)) for name in collected}
        try:
            handler(intake, frame)
        except Exception as exc:  # a malformed sheet never stops the analysis and contributes nothing
            del intake.rows[before:]
            for name, size in sizes.items():
                del getattr(intake, name)[size:]
            intake.sheets[sheet] = {"status": VALIDATION_ERROR, "rows": int(len(frame)),
                                    "detail": f"{type(exc).__name__}: {exc}"}
            continue
        rows = intake.rows[before:]
        states = Counter(row["row_state"] for row in rows)
        usable = sum(states[s] for s in (ROW_ACCEPTED, ROW_ACCEPTED_TEST, ROW_ESTIMATE))
        samples = sum(1 for row in rows if row["verification"] == SAMPLE_NOT_USABLE)
        if usable:
            status = VALIDATED_INPUT
        elif rows and samples == len(rows) or not rows:
            status = NO_USABLE_EVIDENCE
        else:
            status = VALIDATION_ERROR
        intake.sheets[sheet] = {
            "status": status, "rows": len(rows), "accepted_actual_rows": states[ROW_ACCEPTED],
            "accepted_test_rows": states[ROW_ACCEPTED_TEST], "usable_not_accepted_rows": states[ROW_ESTIMATE],
            "rejected_rows": states[ROW_REJECTED], "sample_rows": samples,
            "detail": "every row is SAMPLE (template)" if rows and samples == len(rows) else None}
    intake.conflicts = detect_conflicts(intake)
    return intake


def inventory_evidence_template_rows() -> list[dict[str, Any]]:
    """SAMPLE rows showing the operational_inventory_evidence format: placeholder ids, example values, never defaults."""
    base = {name: None for name in (*SHEET_REQUIRED[INVENTORY_SHEET], *SHEET_OPTIONAL[INVENTORY_SHEET])}
    trust = {"record_origin": SAMPLE, "verification_status": "VERIFIED", "verified_by": "SAMPLE_ROLE", "verified_at": "2026-07-16"}
    return [
        {**base, **trust, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_A", "product_id": "SAMPLE_PRODUCT_1",
         "effective_date": "2026-07-16", "on_hand_qty": 100, "reserved_qty": 30, "available_for_transfer_qty": 40,
         "quantity_unit": "BOX", "evidence_class": oe.DIRECT_REAL, "verification_level": OBSERVED_RECORD,
         "evidence_source": "SAMPLE_WMS_EXPORT",
         "note": "SAMPLE: 보유 100, 예약 30, 이동 가능 40 (BOX). 이동 가능량은 100도 70도 아닌 40"},
        {**base, **trust, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_A", "product_id": "SAMPLE_PRODUCT_1",
         "effective_date": "2026-07-16", "transfer_cap_qty": 35, "quantity_unit": "BOX",
         "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "verification_level": OPERATIONALLY_APPROVED,
         "evidence_source": "SAMPLE_TRANSFER_APPROVAL", "note": "SAMPLE: 운영자가 승인한 오늘 최대 반출량 35 BOX"},
        {**base, **trust, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_A", "product_id": "SAMPLE_PRODUCT_1",
         "effective_date": "2026-07-01", "valid_until": "2026-12-31", "quantity_unit": "BOX", "dataset_unit_confirmed": "true",
         "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "verification_level": INPUT_CONFIRMED,
         "evidence_source": "SAMPLE_PRODUCT_MASTER", "note": "SAMPLE: 데이터의 수량 단위가 BOX임을 확인"},
        {**base, **trust, "network_id": "SAMPLE_NETWORK", "location_id": "SAMPLE_CENTER_C", "product_id": "SAMPLE_PRODUCT_1",
         "effective_date": "2026-07-01", "valid_until": "2026-12-31", "handling_allowed": "false",
         "handling_conditions": "SAMPLE: 냉동 보관 불가", "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
         "verification_level": INPUT_CONFIRMED, "evidence_source": "SAMPLE_ASSORTMENT",
         "note": "SAMPLE: C점은 이 상품을 받을 수 없음 (재고 0과는 다른 정보)"},
    ]


# ------------------------------------------------------------------------------------------------ evidence for one decision date


def _valid_on(item: Mapping[str, Any], when: _date) -> bool:
    return item["start"] <= when and (item["end"] is None or when <= item["end"])


def _conflict_rows(intake: Intake, when: _date) -> set[str]:
    rows = set()
    for conflict in intake.conflicts:
        start = oe.parse_date(conflict["window_start"])
        end = oe.parse_date(conflict["window_end"])
        if start is not None and start <= when and (end is None or when <= end):
            rows.update(conflict["rows"].split("|"))
    return rows


DECISION_DATE_UNDECLARED = "DECISION_DATE_UNDECLARED"


@dataclass
class DayEvidence:
    """The intake narrowed to one decision date, in the P1-B context shape plus the contract-only facts.
    ``when`` None: no decision date is declared, so no dated evidence applies (``undated_evidence``)."""

    when: _date | None
    scope: str
    routes: list[dict[str, Any]]
    quantities: list[dict[str, Any]]
    product_units: dict[str, tuple[str, str]]
    operator_facts: dict[tuple[str, str], dict[str, Any]]
    receivable: dict[tuple[str, str], tuple[float, str, str]]
    handling_prohibited: set[tuple[str, str]]
    conflicts: list[dict[str, Any]]
    conflict_rows: set[str]
    expired: dict[tuple[str, ...], list[str]]


def day_evidence(intake: Intake, when: Any) -> DayEvidence:
    """Usable evidence valid on ``when``; rows in a CONFLICT window are withheld (no value chosen for them)."""
    day = oe.parse_date(when)
    if day is None:
        raise ValueError("decision date required")
    blocked = _conflict_rows(intake, day)
    usable = lambda item: _usable(item) and item["row"] not in blocked  # noqa: E731
    # estimates reach P1-B too (it reports them as *_ESTIMATE_ONLY and never accepts them); accepted-only facts below
    shown = lambda item: item.get("row_state") in (ROW_ACCEPTED, ROW_ACCEPTED_TEST, ROW_ESTIMATE) and item["row"] not in blocked  # noqa: E731
    expired: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)

    def note_expired(item: Mapping[str, Any], scope: tuple[str, ...]) -> None:
        if _usable(item) and not _valid_on(item, day):
            expired[scope].append("EVIDENCE_EXPIRED" if item["end"] is not None and item["end"] < day else "EVIDENCE_NOT_YET_VALID")

    routes = [r for r in intake.routes if shown(r)]  # P1-B applies each rule's own window
    for record in intake.routes:
        note_expired(_route_window(record), ("ROUTE", record["source_id"], record["target_id"]))
    quantities = [q for q in intake.quantities if shown(q) and q.get("meaning") not in (RECEIVABLE_CAPACITY,)]
    for record in intake.quantities:
        note_expired(record, ("QTY", record["location_id"], record["product_id"]))
    receivable: dict[tuple[str, str], tuple[float, str, str]] = {}
    for record in intake.receivable_caps:
        if usable(record) and _valid_on(record, day):
            receivable[(record["location_id"], record["product_id"])] = (float(record["quantity"]), record["unit"], record["row"])
        # P1-B reads the handling meaning of the same row (DECLARED_RECEIVABLE without its quantity)
        if shown(record):
            quantities.append({**record, "quantity": None})
    facts: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for item in intake.inventory:
        note_expired(item, ("INV", item["location_id"], item["product_id"]))
        if usable(item) and _valid_on(item, day):
            facts[(item["location_id"], item["product_id"])][item["field"]] = (item["value"], item["unit"], item["evidence"], item["row"])
            if item.get("lot_restriction"):
                facts[(item["location_id"], item["product_id"])]["LOT_RESTRICTION"] = item["lot_restriction"]
            if item.get("source_expected_demand_qty") is not None:
                facts[(item["location_id"], item["product_id"])]["SOURCE_EXPECTED_DEMAND"] = (item["source_expected_demand_qty"], item["unit"])
    units = {}
    for item in intake.units:
        note_expired(item, ("UNIT", item["product_id"]))
        if usable(item) and _valid_on(item, day):
            units[item["product_id"]] = (item["unit"], oe.VERIFIED_OPERATOR_INPUT)
    prohibited = set()
    for item in intake.handling:
        note_expired(item, ("QTY", item["location_id"], item["product_id"]))
        if usable(item) and _valid_on(item, day):
            if item["allowed"]:
                quantities.append({"row": item["row"], "network_id": item["network_id"], "location_id": item["location_id"],
                                   "product_id": item["product_id"], "quantity_kind": oe.DECLARED_RECEIVABLE,
                                   "start": item["start"], "end": item["end"], "quantity": None, "unit": None,
                                   "evidence": item["evidence"], "data_source": item["source"], "row_state": item["row_state"]})
            else:
                prohibited.add((item["location_id"], item["product_id"]))
    active = [c for c in intake.conflicts if any(r in blocked for r in c["rows"].split("|"))]
    return DayEvidence(when=day, scope=intake.scope, routes=routes, quantities=quantities, product_units=units,
                       operator_facts=dict(facts), receivable=receivable, handling_prohibited=prohibited,
                       conflicts=active, conflict_rows=blocked, expired=dict(expired))


def undated_evidence() -> DayEvidence:
    """No decision date: operator evidence cannot be shown valid, so none is applied (it is still validated by intake)."""
    return DayEvidence(when=None, scope=NO_EVIDENCE, routes=[], quantities=[], product_units={}, operator_facts={},
                       receivable={}, handling_prohibited=set(), conflicts=[], conflict_rows=set(), expired={})


def transferable_records(evidence: DayEvidence, network_id: str) -> list[dict[str, Any]]:
    """Operator-side transferable quantity per (source, product) as the P1-B DECLARED_TRANSFERABLE record."""
    out = []
    for (location, product), facts in sorted(evidence.operator_facts.items()):
        record = source_availability(network_id=network_id, location_id=location, product_id=product,
                                     as_of=evidence.when.isoformat(), operator=facts)
        if record["validation_status"] != VERIFIED:
            continue
        kind = record["transferable_kind"]
        fact = facts["AVAILABLE_FOR_TRANSFER" if kind == OBSERVED_AVAILABLE_FOR_TRANSFER else "TRANSFER_CAP"]
        out.append({"row": f"source_availability:{location}/{product}", "network_id": network_id, "location_id": location,
                    "product_id": product, "quantity_kind": oe.DECLARED_TRANSFERABLE, "start": evidence.when,
                    "end": evidence.when, "quantity": float(record["transferable_qty"]), "unit": record["quantity_unit"],
                    "evidence": fact[2], "data_source": f"{kind}:{fact[3]}"})
    return out


def oe_context(evidence: DayEvidence, base: Mapping[str, Any]) -> oe.OperationalContext:
    """The P1-B context for the day: ``base`` holds the network facts (temperature, history, ...)."""
    kwargs = dict(base)
    network = kwargs["network_id"]
    kwargs["route_records"] = [*kwargs.get("route_records", ()), *evidence.routes]
    kwargs["quantity_records"] = [*kwargs.get("quantity_records", ()), *evidence.quantities,
                                  *transferable_records(evidence, network)]
    kwargs["product_units"] = {**dict(kwargs.get("product_units") or {}), **evidence.product_units}
    return oe.OperationalContext(**kwargs)


# ------------------------------------------------------------------------------------------------ lane contract


def _lane_scopes(lane: Mapping[str, Any]) -> list[tuple[str, ...]]:
    source, target, product = (_text(lane.get(k)) for k in ("source_id", "target_id", "product_id"))
    scopes = [("ROUTE", source, target), ("QTY", target, product), ("QTY", source, product), ("INV", source, product),
              ("UNIT", product)]
    if _text(lane.get("route_type")).upper() == "VIA_DC" and _text(lane.get("dc_id")):
        dc = _text(lane.get("dc_id"))
        scopes += [("ROUTE", source, dc), ("ROUTE", dc, target)]
    return scopes


def _conflict_touches(conflict: Mapping[str, Any], lane: Mapping[str, Any]) -> bool:
    source, target, product = (_text(lane.get(k)) for k in ("source_id", "target_id", "product_id"))
    parts = conflict["scope"].split("/")
    if conflict["code"].startswith("ROUTE_"):
        return parts[:2] == [source, target] and parts[2] in ("*", product)
    if conflict["code"] == "DATASET_UNIT_CONFLICT":
        return parts == [product]
    return parts in ([source, product], [target, product])


def evaluate_lane_contract(lane: Mapping[str, Any], evidence: DayEvidence, context: oe.OperationalContext) -> dict[str, Any]:
    """The P1-B record plus the contract view: need formula id, source availability kind, conflicts, expiry,
    receivable capacity, handling prohibition, missing fields and the evidence scope of any strict result."""
    record = oe.evaluate_lane(lane, context, when=evidence.when)
    when = evidence.when or _date.min
    as_of = evidence.when.isoformat() if evidence.when is not None else None
    source, target, product = (_text(lane.get(k)) for k in ("source_id", "target_id", "product_id"))
    conflicts = sorted({c["code"] for c in evidence.conflicts if _conflict_touches(c, lane)})
    expired = sorted({code for scope in _lane_scopes(lane) for code in evidence.expired.get(scope, [])})
    blockers = []
    if (target, product) in evidence.handling_prohibited:
        blockers.append("TARGET_HANDLING_PROHIBITED")
    unit = sf.normalize_unit(context.product_units.get(product, (None,))[0])
    capacity = oe.resolve_capacity(lane, oe.resolve_route(lane, when, context), context)
    availability = source_availability(
        network_id=context.network_id, location_id=source, product_id=product, as_of=as_of,
        dataset_on_hand=_finite(lane.get("source_stock")),
        dataset_on_hand_evidence=oe._lane_value(lane, "source_stock", context)[1],
        dataset_unit=sf.normalize_unit(lane.get("quantity_unit")) or unit, surplus_proxy=_finite(lane.get("source_surplus")),
        surplus_proxy_basis="max(0, stock - cross-node median)" if _finite(lane.get("source_surplus")) is not None else None,
        operator=evidence.operator_facts.get((source, product)))
    if availability["conflict_codes"]:
        conflicts = sorted({*conflicts, *availability["conflict_codes"].split("|")})
    validated = record["validated_qty"]
    receivable = evidence.receivable.get((target, product))
    receivable_note = None
    if receivable is not None:
        if unit is not None and receivable[1] != unit:
            blockers.append("RECEIVABLE_CAPACITY_UNIT_MISMATCH")
        elif validated is not None and receivable[0] < validated - 1e-9:
            validated = float(math.floor(receivable[0] + 1e-9))
            receivable_note = "RECEIVABLE_CAPACITY_BINDING"
            if validated < 1 - 1e-9:
                blockers.append("RECEIVABLE_CAPACITY_BELOW_ONE_UNIT")
    # undated: P1-B marks DECISION_DATE_MISSING (INELIGIBLE); unless a real blocker exists the inputs are still listed
    date_only = evidence.when is None and not oe.hard_blockers({**lane, "_context_network": context.network_id})         and record["route_class"] != oe.ROUTE_PROHIBITED and not str(record["quantity_unit_evidence"]).startswith("UNIT_MISMATCH")
    status = (CONFLICT if conflicts else oe.INELIGIBLE if blockers else DECISION_DATE_UNDECLARED if date_only
              else record["operational_eligibility"])
    strict = record["strict_eligibility"] == oe.STRICT_ELIGIBLE and status not in (CONFLICT, oe.INELIGIBLE)
    required, conditional = oe.missing_fields({**record, "operational_eligibility": oe.PROXY_FEASIBLE} if date_only else record)
    flags = {name: False for name in MISSING_FLAGS}
    for name in [*required, *conditional]:
        flags[MISSING_FIELD_NAMES[name]] = True
    flags["conflicting_evidence"] = bool(conflicts)
    flags["expired_evidence"] = bool(expired)
    need_formula = NEED_STATUS_FORMULA.get(record["target_need_formula"], record["target_need_formula"])
    supply_kind = (availability["transferable_kind"] if record["source_supply_accepted"] and availability["transferable_kind"]
                   else "ON_HAND_MINUS_OWN_7D_SALES" if record["source_surplus_validation_status"] == "VERIFIED"
                   else SOURCE_SURPLUS_PROXY if record["source_surplus_provenance"] == oe.PROXY else record["source_surplus_provenance"])
    reasons = [*blockers, *(f"EVIDENCE_CONFLICT:{code}" for code in conflicts), *expired]
    if receivable_note:
        reasons.append(receivable_note)
    return {
        **record, "contract_version": CONTRACT_VERSION, "contract_eligibility": status,
        "contract_strict": ("STRICT_ELIGIBLE_TEST_ONLY" if evidence.scope == TEST_LINKAGE else oe.STRICT_ELIGIBLE)
        if strict else (oe.INELIGIBLE if status in (oe.INELIGIBLE, CONFLICT) else oe.INSUFFICIENT_OPERATIONAL_EVIDENCE),
        "evidence_scope": evidence.scope, "contract_validated_qty": validated if strict else None,
        "contract_quantity_unit": unit if record["quantity_unit_accepted"] else None,
        "capacity_value": _finite(capacity.value) if capacity.accepted else None,
        "capacity_provenance": capacity.evidence,
        "need_formula_id": need_formula, "need_formula_version": FORMULAS.get(need_formula, {}).get("version"),
        "source_availability_kind": supply_kind, "source_availability_status": availability["validation_status"],
        "source_on_hand_qty": availability["dataset_on_hand_qty"], "source_transferable_qty": availability["transferable_qty"],
        "receivable_capacity_qty": receivable[0] if receivable else None,
        "contract_reasons": "|".join(dict.fromkeys(reasons)) or None, **flags,
        "missing_required_count": len(required), "missing_fields": "|".join(MISSING_FIELD_NAMES[n] for n in [*required, *conditional]) or None,
    }


# ------------------------------------------------------------------------------------------------ missing inputs & priority


def missing_by_scope(records: pd.DataFrame) -> pd.DataFrame:
    """Section 30: candidates lacking each input, per product, per source / target location and per route."""
    if records.empty:
        return pd.DataFrame(columns=["scope", "key", "candidates", *MISSING_FLAGS])
    frame = records.copy()
    frame["route"] = frame["source_id"].astype(str) + "->" + frame["target_id"].astype(str)
    out = []
    for scope, column in (("product", "product_id"), ("source_location", "source_id"), ("target_location", "target_id"),
                          ("route", "route")):
        grouped = frame.groupby(column)
        table = grouped[list(MISSING_FLAGS)].sum().astype(int)
        table.insert(0, "candidates", grouped.size())
        table = table.reset_index().rename(columns={column: "key"})
        table.insert(0, "scope", scope)
        out.append(table)
    return pd.concat(out, ignore_index=True)


KEY_COLUMNS = {
    "missing_route_approval": ("source_id", "target_id"), "missing_actual_surplus": ("snapshot_date", "source_id", "product_id"),
    "missing_actual_target_need": ("snapshot_date", "target_id", "product_id"), "missing_target_handling": ("target_id", "product_id"),
    "missing_quantity_unit": ("product_id",), "missing_capacity": ("source_id", "target_id"),
    "missing_cost_basis": ("source_id", "target_id"),
}
INPUT_CYCLE = {
    "missing_route_approval": "per contract / route-rule change (validity window)",
    "missing_actual_surplus": "daily per source x product (or each WMS snapshot)",
    "missing_actual_target_need": "daily per target x product (each replenishment decision)",
    "missing_target_handling": "on assortment change (validity window)", "missing_quantity_unit": "once per product (master data)",
    "missing_capacity": "per route and contract period (vehicle / slot plan)",
    "missing_cost_basis": "per route and contract period (tariff / invoice basis)",
}
REUSE_SCOPE = {
    "missing_route_approval": "every product and day on the directed pair while the rule is valid",
    "missing_actual_surplus": "every target of that source and product on that day",
    "missing_actual_target_need": "every source supplying that target and product on that day",
    "missing_target_handling": "every source and day while valid", "missing_quantity_unit": "every lane, day and table of the product",
    "missing_capacity": "every product and day on the pair while valid", "missing_cost_basis": "every product and day on the pair",
}


def request_priority(records: pd.DataFrame) -> pd.DataFrame:
    """Section 31: which input to request first.  Lexicographic, no weights: (1) safety tier, (2) verification checks
    released (candidates blocked), (3) reuse (candidates per input row), (4) STRICT candidates unlocked when added
    greedily to the inputs ranked before it (upper bound: assumes each supplied value verifies)."""
    frame = records[records["contract_eligibility"] != oe.INELIGIBLE] if not records.empty else records
    flags = [f for f in MISSING_FIELD_NAMES.values() if not frame.empty and bool(frame[f].any())]
    rows = []
    for flag in flags:
        part = frame[frame[flag].astype(bool)]
        keys = part[list(KEY_COLUMNS[flag])].drop_duplicates()
        rows.append({"field": flag, "evidence_field": FIELD_FOR_FLAG[flag], "tier": REQUEST_TIER[flag],
                     "candidates_blocked": len(part), "rows_to_collect": len(keys),
                     "key_granularity": "+".join(KEY_COLUMNS[flag]),
                     "reuse_candidates_per_row": round(len(part) / max(1, len(keys)), 3),
                     "input_cycle": INPUT_CYCLE[flag], "reuse_scope": REUSE_SCOPE[flag]})
    rows.sort(key=lambda r: (TIER_ORDER[r["tier"]], -r["candidates_blocked"], -r["reuse_candidates_per_row"], r["field"]))
    required = [f for f in MISSING_FIELD_NAMES.values() if f != "missing_capacity"]
    needs = [frozenset(f for f in required if not frame.empty and bool(row[f])) for _, row in frame.iterrows()] if not frame.empty else []
    supplied: set[str] = set()
    before = sum(1 for need in needs if not need)
    for priority, row in enumerate(rows, start=1):
        supplied.add(row["field"])
        unlocked = sum(1 for need in needs if need <= supplied)
        row.update(priority=priority, strict_upper_bound_after=unlocked, strict_unlocked_by_this=unlocked - before)
        before = unlocked
    columns = ["priority", "field", "evidence_field", "tier", "candidates_blocked", "rows_to_collect",
               "reuse_candidates_per_row", "strict_unlocked_by_this", "strict_upper_bound_after", "key_granularity",
               "input_cycle", "reuse_scope"]
    return pd.DataFrame(rows, columns=columns)


# ------------------------------------------------------------------------------------------------ T1 STRICT bridge


CONTRACT_STRICT_STATUSES = (oe.STRICT_ELIGIBLE, "STRICT_ELIGIBLE_TEST_ONLY")


def t1_strict_caps(lanes: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> tuple[sf.CapTable, list[dict]]:
    """STRICT caps from the contract only (section 36): accepted transferable stock, accepted need (an accepted
    receivable capacity is a second, separately recorded upper bound on the same target cap) and accepted route
    capacity.  Nothing else becomes a STRICT cap: on-hand stock alone or a PROXY rule stays MISSING."""
    caps: sf.CapTable = {}
    prepared = []
    for index, (lane, result) in enumerate(zip(lanes, results)):
        unit = result.get("contract_quantity_unit")
        record = {**dict(lane), "quantity_unit": unit}
        slots = sf.cap_slots(record, index)
        strict_ok = result.get("contract_strict") in CONTRACT_STRICT_STATUSES
        transferable = _finite(result.get("source_surplus_value")) if result.get("source_supply_accepted") else None
        need = _finite(result.get("target_need_value")) if result.get("target_demand_accepted") else None
        receivable = _finite(result.get("receivable_capacity_qty"))
        if strict_ok and transferable is not None:
            sf.add_cap(caps, sf.Cap(sf.SOURCE_SURPLUS, slots[sf.SOURCE_SURPLUS][1], transferable,
                                    oe.base_provenance(result["source_surplus_provenance"]),
                                    f"contract {result['source_availability_kind']}", unit))
        else:
            sf.add_cap(caps, sf.Cap(sf.SOURCE_SURPLUS, slots[sf.SOURCE_SURPLUS][1], None, "MISSING",
                                    "no accepted transferable quantity"))
        if strict_ok and need is not None:
            value = min(need, receivable) if receivable is not None else need
            basis = f"contract {result['need_formula_id']}" + (
                f"; bounded by RECEIVABLE_CAPACITY {receivable:g} (kept as its own field)" if receivable is not None else "")
            sf.add_cap(caps, sf.Cap(sf.TARGET_NEED, slots[sf.TARGET_NEED][1], value,
                                    oe.base_provenance(result["target_need_provenance"]), basis, unit))
        else:
            sf.add_cap(caps, sf.Cap(sf.TARGET_NEED, slots[sf.TARGET_NEED][1], None, "MISSING", "no accepted target need"))
        capacity = _finite(result.get("capacity_value"))
        if strict_ok and result.get("capacity_accepted") and capacity is not None:
            sf.add_cap(caps, sf.Cap(sf.ROUTE_CAPACITY, slots[sf.ROUTE_CAPACITY][1], capacity,
                                    oe.base_provenance(result["capacity_provenance"]), "contract route capacity", unit))
        prepared.append(record)
    return caps, prepared


def t1_strict_bridge(lanes: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]], *, scope: str,
                     max_routes: int | None = None) -> dict[str, Any]:
    """T1 STRICT_ACTUAL on contract caps: an explicit, separate path; the production selector and its field are untouched.

    T1 checks caps only, so a candidate that is not contract-strict (e.g. no approved route) is excluded before T1 runs;
    it could otherwise draw on caps another, verified candidate supplied.  With no contract-strict candidate T1 runs on
    every candidate with MISSING caps and keeps its own INSUFFICIENT_CAP_DATA."""
    strict_idx = [i for i, result in enumerate(results) if result.get("contract_strict") in CONTRACT_STRICT_STATUSES]
    chosen = strict_idx or list(range(len(lanes)))
    caps, prepared = t1_strict_caps([lanes[i] for i in chosen], [results[i] for i in chosen])
    plan = sf.select_shared_feasible(prepared, caps, mode=sf.STRICT_ACTUAL, max_routes=max_routes,
                                     partial_policy=sf.PARTIAL_NONE)
    statuses = Counter(row["selection_status"] for row in plan["rows"])
    excluded = len(lanes) - len(chosen) if strict_idx else 0
    if excluded:
        statuses["EXCLUDED_NOT_CONTRACT_STRICT"] = excluded
    selected = int(statuses.get(sf.SELECTED, 0) + statuses.get(sf.PARTIALLY_SELECTED, 0))
    return {"mode": sf.STRICT_ACTUAL, "evidence_scope": scope, "plan_status": plan.get("plan_status"),
            "candidates": len(lanes), "t1_input_candidates": len(prepared), "contract_strict_candidates": len(strict_idx),
            "selected": selected, "selection_statuses": dict(sorted(statuses.items())),
            "selected_ids": list(plan.get("selected_ids") or []),
            "total_allocated_qty": plan.get("total_allocated_qty"), "total_move_cost": plan.get("total_move_cost"),
            "feasibility_claim": plan.get("feasibility_claim") if scope == ACTUAL_OPERATION
            else f"{scope}: T1 claim on non-operational caps ({plan.get('feasibility_claim')})",
            "violations": (plan.get("validation") or {}).get("violation_count"),
            "strict_actual_claim": scope == ACTUAL_OPERATION and selected > 0,
            "label": ("STRICT_ACTUAL on verified operator evidence" if scope == ACTUAL_OPERATION
                      else "TEST_LINKAGE_ONLY: not an operational result" if scope == TEST_LINKAGE
                      else "no accepted operator evidence"),
            "rows": plan["rows"]}


# ------------------------------------------------------------------------------------------------ pipeline (parallel field)

PIPELINE_LANE_EVIDENCE = {
    "source_stock": (oe.USER_ESTIMATE, "uploaded inventory stock (declare <column>_provenance=actual for observed stock)"),
    "source_surplus": (oe.USER_ESTIMATE, "uploaded movable-stock column"),
    "target_stock": (oe.USER_ESTIMATE, "uploaded inventory stock"),
    "target_need_7d": (oe.PROXY, "pipeline shortage"),
}


def _decision_date(uploaded: Mapping[str, Any]) -> tuple[str | None, str]:
    config = _config(uploaded.get("config"))
    for key in ("as_of_date", "snapshot_date", "analysis_date", "decision_date"):
        day = oe.parse_date(config.get(key))
        if day is not None:
            return day.isoformat(), f"config.{key}"
    inventory = uploaded.get("inventory")
    if isinstance(inventory, pd.DataFrame):
        for column in ("snapshot_date", "date", "as_of_date"):
            if column in inventory.columns:
                days = {oe.parse_date(v) for v in inventory[column].dropna()} - {None}
                if len(days) == 1:
                    return next(iter(days)).isoformat(), f"inventory.{column}"
    return None, "UNDECLARED"


def _pipeline_lane(record: Mapping[str, Any], inventory: Mapping[tuple[str, str], Mapping[str, Any]], when: str | None,
                   need: TargetNeed, default_unit: str | None) -> dict[str, Any]:
    product, source, target, route_type, dc = sf._ids(record)
    source_row, target_row = inventory.get((source, product), {}), inventory.get((target, product), {})
    stock, stock_field = _first_number(source_row, sf.STOCK_FIELDS)
    target_stock, target_field = _first_number(target_row, sf.STOCK_FIELDS)
    movable, movable_field = _first_number(record, sf.MOVABLE_FIELDS)
    if movable is None:
        movable, movable_field = _first_number(source_row, sf.MOVABLE_FIELDS)
    cost, _ = _first_number(record, sf.COST_FIELDS)
    real = record.get("real_transport_applied")
    distance = _finite(record.get("distance_km"))
    unit = sf.normalize_unit(sf._first_text(record, sf.UNIT_FIELDS)) or default_unit
    lane = {
        "snapshot_date": when, "candidate_id": sf.candidate_identity(record, 0), "route_id": record.get("route_id"),
        "source_id": source, "target_id": target, "product_id": product, "route_type": route_type, "dc_id": dc or None,
        "recommended_qty": _first_number(record, sf.QTY_FIELDS)[0], "source_stock": stock, "source_surplus": movable,
        "target_stock": target_stock, "target_need_7d": need.target_need_value, "target_need_formula": need.formula_id,
        "route_status": "ROUTE_NETWORK_TABLE" if distance is not None else "",
        "distance_km": distance, "distance_provenance": _text(record.get("distance_provenance")) or "uploaded_routes",
        "unit_status": "MATCHED" if unit else "UNDECLARED_SAME_SOURCE", "quantity_unit": unit, "move_cost": cost,
        "cost_provenance": "DERIVED_REAL" if real is True or str(real).strip().lower() == "true" else
        (_text(record.get("cost_provenance")) or "USER_INPUT"),
        "target_has_inventory_row": bool(target_row), "inventory_target_matched": bool(target_row), "is_duplicate": False,
    }
    for column, row, name in (("source_stock", source_row, stock_field), ("target_stock", target_row, target_field)):
        if name is not None:
            label = _text(row.get(f"{name}_provenance"))
            if label:
                lane[f"{column}_provenance"] = declared_provenance(row, str(name), "USER_INPUT")[0]
    if movable_field is not None:
        lane["source_surplus_provenance"] = declared_provenance(record if _finite(record.get(movable_field)) is not None
                                                                else source_row, str(movable_field), "USER_INPUT")[0]
    lane["target_need_7d_provenance"] = need.evidence if need.target_need_value is not None else None
    return lane


def build_operational_evidence_analysis(uploaded_data: Mapping[str, Any] | None, candidates: Any) -> dict[str, Any]:
    """PipelineResult.operational_evidence: the need contract and operator-evidence intake next to the production result.

    Never changes recommendations, ranks, Top-5, actions, KPIs or the T1 field.  Deterministic (no timings)."""
    data = dict(uploaded_data or {})
    config = _config(data.get("config"))
    network = _text(config.get("network_id")) or _text(config.get("dataset_id")) or "UPLOADED_WORKBOOK"
    when, when_basis = _decision_date(data)
    stores = data.get("stores")
    products = data.get("products")
    locations = frozenset(str(v) for v in stores["node_id"].dropna()) if isinstance(stores, pd.DataFrame) and "node_id" in stores.columns else None
    product_ids = frozenset(str(v) for v in products["product_id"].dropna()) if isinstance(products, pd.DataFrame) and "product_id" in products.columns else None
    universe = KeyUniverse(network_id=network, dataset_id=_text(config.get("dataset_id")), locations=locations,
                           products=product_ids, workbook=True)
    intake = intake_evidence({sheet: data.get(sheet) for sheet in EVIDENCE_SHEETS}, universe, allow_test=False)
    records = candidates.where(pd.notna(candidates), None).to_dict("records") if isinstance(candidates, pd.DataFrame) \
        else [dict(item) for item in candidates or []]
    inventory = {}
    for row in oe._records(data.get("inventory") if isinstance(data.get("inventory"), pd.DataFrame) else None):
        key = (_text(row.get("store_id") or row.get("node_id")), _text(row.get("product_id") or row.get("item_id")))
        if key[0] and key[1]:
            inventory.setdefault(key, row)
    default_unit = sf.normalize_unit(config.get("quantity_unit"))
    needs: dict[tuple[str, str], TargetNeed] = {}
    for record in records:
        product, _, target, _, _ = sf._ids(record)
        if (target, product) not in needs:
            needs[(target, product)] = pipeline_need(inventory.get((target, product)), network_id=network,
                                                     location_id=target, product_id=product, as_of=when, unit=default_unit,
                                                     dataset_id=universe.dataset_id)
    lanes = [_pipeline_lane(r, inventory, when, needs[(sf._ids(r)[2], sf._ids(r)[0])], default_unit) for r in records]
    candidate_rows, bridge = [], {"status": "NOT_RUN", "reason": "NO_ACCEPTED_OPERATOR_EVIDENCE"}
    keep = ("snapshot_date", "candidate_id", "route_id", "source_id", "target_id", "product_id", "requested_qty",
            "contract_validated_qty", "operational_eligibility", "contract_eligibility", "contract_strict",
            "need_formula_id", "source_availability_kind", "source_availability_status", "target_need_provenance",
            "source_surplus_provenance", "route_class", "blocking_reasons", "contract_reasons", "missing_fields",
            *MISSING_FLAGS)
    if when is not None:
        evidence = day_evidence(intake, when)
        context = oe_context(evidence, {"network_id": network, "lane_evidence": dict(PIPELINE_LANE_EVIDENCE)})
        results = [evaluate_lane_contract(lane, evidence, context) for lane in lanes]
        if evidence.scope == ACTUAL_OPERATION:
            bridge = {"status": "RUN", **t1_strict_bridge(records, results, scope=evidence.scope)}
            bridge["rows"] = [{k: row.get(k) for k in ("candidate_id", "selection_status", "allocated_qty", "rejection_reason")}
                              for row in bridge["rows"]]
        candidate_rows = [{k: r.get(k) for k in keep} for r in results]
    else:
        context = oe_context(undated_evidence(), {"network_id": network, "lane_evidence": dict(PIPELINE_LANE_EVIDENCE)})
        results = [evaluate_lane_contract(lane, undated_evidence(), context) for lane in lanes]
        candidate_rows = [{k: r.get(k) for k in keep} for r in results]
        bridge = {"status": "NOT_RUN", "reason": "DECISION_DATE_UNDECLARED: operator evidence cannot be dated"}
    frame = pd.DataFrame(candidate_rows)
    priority = request_priority(frame).to_dict("records") if not frame.empty else []
    need_rows = [needs[key].as_dict() for key in sorted(needs)]
    operator_facts = day_evidence(intake, when).operator_facts if when is not None else {}
    source_rows = []
    for key in sorted({(sf._ids(r)[1], sf._ids(r)[0]) for r in records}):
        row = inventory.get(key, {})
        stock, field_name = _first_number(row, sf.STOCK_FIELDS)
        movable, _ = _first_number(row, sf.MOVABLE_FIELDS)
        source_rows.append(source_availability(
            network_id=network, location_id=key[0], product_id=key[1], as_of=when, dataset_on_hand=stock,
            dataset_on_hand_evidence=oe.evidence_class(declared_provenance(row, str(field_name), "USER_INPUT")[0]) if stock is not None else oe.MISSING,
            dataset_unit=default_unit, surplus_proxy=movable, surplus_proxy_basis="uploaded movable-stock column" if movable is not None else None,
            operator=operator_facts.get(key)))
    strict_count = sum(1 for r in candidate_rows if r["contract_strict"] == oe.STRICT_ELIGIBLE)
    return {
        "status": "parallel_only", "version": CONTRACT_VERSION, "evidence_version": oe.EVIDENCE_VERSION,
        "production_action_applied": False, "legacy_top_n_replaced": False,
        "network_id": network, "decision_date": when, "decision_date_basis": when_basis,
        "evidence_scope": intake.scope, "sheets": intake.sheets, "input_rows": intake.rows, "conflicts": intake.conflicts,
        "formula_ids_in_use": sorted({n["formula_id"] for n in need_rows}),
        "target_need": need_rows, "source_availability": source_rows, "candidates": candidate_rows,
        "contract_eligibility_counts": dict(sorted(Counter(r["contract_eligibility"] for r in candidate_rows).items())),
        "strict_eligible_count": strict_count, "strict_actual_claim": bool(bridge.get("strict_actual_claim")),
        "request_priority": priority, "t1_strict_bridge": bridge, "authority_check": AUTHORITY_NOTE,
    }
