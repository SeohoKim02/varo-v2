"""Observed operator outcomes, not predictions or counterfactuals.

Rows are complete snapshots, not patches: a blank always stays NULL. Recorded
amounts require observed provenance and source. Realized loss is either an
explicit observed accounting total, or observed loss debits minus credits, with
an explicit accounting basis. Inventory quantities/revenue never imply loss.
Only OBSERVED_AVOIDABLE_LOSS can calibrate the Seller Loss monetary prediction;
this declaration must be supported by the operator's accounting records.
"""
from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any

import pandas as pd

from services.seller_loss_promotion_gate import ACTION_MAPPING, _nonproduction

SHEET_KEY = "seller_outcomes"
CONTRACT_VERSION = "seller-shadow-outcomes-0.1"
DATA_MODES = ("PRODUCTION", "TEST", "SAMPLE", "SCENARIO")
EXECUTION_STATUSES = ("NOT_EXECUTED", "PLANNED", "EXECUTED", "CANCELLED", "UNKNOWN")
OUTCOME_PROVENANCE = ("DIRECT_REAL", "OPERATOR_CONFIRMED", "IMPORTED_REAL", "MISSING")
OPERATOR_ACTIONS = tuple(dict.fromkeys((*ACTION_MAPPING, "긴급 할인", "긴급할인", "1+1", "폐기", "보류")))
LOSS_BASES = ("OBSERVED_AVOIDABLE_LOSS", "ACCOUNTING_LOSS", "OPERATING_COST")
ACTUAL_FIELDS = (
    "actual_sold_qty", "actual_unsold_qty", "actual_transfer_qty", "actual_transfer_cost",
    "actual_discount_rate", "actual_revenue", "actual_disposal_qty", "actual_disposal_cost",
    "actual_holding_cost", "actual_realized_loss", "actual_loss_debits", "actual_loss_credits",
)
MONEY_FIELDS = tuple(f for f in ACTUAL_FIELDS if f.endswith("cost") or f in (
    "actual_revenue", "actual_realized_loss", "actual_loss_debits", "actual_loss_credits"))
QUANTITY_FIELDS = tuple(f for f in ACTUAL_FIELDS if f.endswith("qty"))
META_FIELDS = ("decision_id", "decision_version", "operator_action", "execution_status", "data_mode",
    "currency", "quantity_unit", "recorded_at", "executed_at", "outcome_provenance", "outcome_source", "loss_basis")
MARKER_FIELDS = ("is_test", "is_sample", "synthetic_fixture", "data_kind", "source_file", "note")


def blank(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value)) or str(value).strip() == ""
    except (ValueError, TypeError):
        return False


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value):
    """Require an ISO date/time, retaining timezone or a declared local timestamp."""
    if blank(value):
        return None
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")).isoformat()
    except (ValueError, TypeError):
        return None


def _instant(value):
    from datetime import timezone
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def parse_seller_outcomes(source: Any):
    result = {"status": "ABSENT", "contract_version": CONTRACT_VERSION, "rows": [], "errors": [], "warnings": []}
    if source is None:
        return result
    try:
        frame = source.copy() if isinstance(source, pd.DataFrame) else pd.DataFrame(source) if isinstance(source, list) else None
    except (ValueError, TypeError):
        frame = None
    if frame is None:
        return {**result, "status": "INVALID", "errors": [{"code": "NOT_A_TABLE"}]}
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if frame.columns.duplicated().any():
        return {**result, "status": "INVALID", "errors": [{"code": "DUPLICATE_COLUMNS"}]}
    unknown = sorted(set(frame.columns) - set(META_FIELDS) - set(ACTUAL_FIELDS) - set(MARKER_FIELDS))
    if unknown:
        result["warnings"].append({"code": "UNSUPPORTED_COLUMNS_DROPPED", "columns": unknown})
    for index, raw in enumerate(frame.to_dict("records"), 1):
        if all(blank(v) for v in raw.values()):
            continue
        errors = []
        row = {k: None if blank(raw.get(k)) else str(raw[k]).strip() for k in META_FIELDS}
        row["data_mode"] = (row["data_mode"] or "PRODUCTION").upper()
        row["execution_status"] = (row["execution_status"] or "UNKNOWN").upper()
        row["outcome_provenance"] = (row["outcome_provenance"] or "MISSING").upper()
        row["currency"] = row["currency"].upper() if row["currency"] else None
        if not row["decision_id"]:
            errors.append("DECISION_ID_REQUIRED")
        if row["data_mode"] not in DATA_MODES:
            errors.append("INVALID_DATA_MODE")
        else:
            labels = " ".join(str(raw.get(k) or "") for k in ("outcome_source","note","data_kind","source_file"))
            flags = {k:str(raw.get(k, "")).lower() in ("true","1","1.0","yes")
                     for k in ("is_test","is_sample","synthetic_fixture")}
            if row["data_mode"]=="SCENARIO" or re.search(r"(?i)\bscenario\b", labels.replace("_", " ")):
                row["data_mode"] = "SCENARIO"
            elif row["data_mode"]=="SAMPLE" or flags["is_sample"] or re.search(r"(?i)\bsamples?\b", labels.replace("_", " ")):
                row["data_mode"] = "SAMPLE"
            elif row["data_mode"]=="TEST" or _nonproduction({}, {**flags,"label":labels}):
                row["data_mode"] = "TEST"
        if row["execution_status"] not in EXECUTION_STATUSES:
            errors.append("INVALID_EXECUTION_STATUS")
        if row["operator_action"] is not None and row["operator_action"] not in OPERATOR_ACTIONS:
            errors.append("INVALID_OPERATOR_ACTION")
        version = raw.get("decision_version")
        row["decision_version"] = None
        if not blank(version):
            try:
                number = float(version)
                if isinstance(version, bool) or not math.isfinite(number) or number < 1 or not number.is_integer():
                    raise ValueError
                row["decision_version"] = int(number)
            except (ValueError, TypeError, OverflowError):
                errors.append("INVALID_DECISION_VERSION")
        for name in ACTUAL_FIELDS:
            value = raw.get(name)
            row[name] = None
            if not blank(value):
                try:
                    number = float(value)
                    if isinstance(value, bool) or not math.isfinite(number) or (number < 0 and name != "actual_realized_loss"):
                        raise ValueError
                    if name == "actual_discount_rate" and number > 1:
                        raise ValueError
                    row[name] = number
                except (ValueError, TypeError, OverflowError):
                    errors.append("INVALID_ACTUAL_VALUE:" + name)
        row["recorded_at"] = timestamp(raw.get("recorded_at"))
        row["executed_at"] = timestamp(raw.get("executed_at"))
        if not row["recorded_at"]:
            errors.append("RECORDED_AT_REQUIRED")
        if not blank(raw.get("executed_at")) and not row["executed_at"]:
            errors.append("INVALID_EXECUTED_AT")
        if row["outcome_provenance"] not in OUTCOME_PROVENANCE:
            errors.append("INVALID_OUTCOME_PROVENANCE")
        actual_present = any(row[k] is not None for k in ACTUAL_FIELDS)
        if actual_present and row["execution_status"] != "EXECUTED":
            errors.append("ACTUAL_REQUIRES_EXECUTED")
        if row["execution_status"] == "EXECUTED":
            if not row["operator_action"]:
                errors.append("EXECUTED_ACTION_REQUIRED")
            if not row["executed_at"]:
                errors.append("EXECUTED_AT_REQUIRED")
        if row["operator_action"] or actual_present:
            if row["outcome_provenance"] not in OUTCOME_PROVENANCE[:3] or not row["outcome_source"]:
                errors.append("OBSERVED_PROVENANCE_AND_SOURCE_REQUIRED")
        if row["recorded_at"] and row["executed_at"] and _instant(row["recorded_at"]) < _instant(row["executed_at"]):
            errors.append("RECORDED_BEFORE_EXECUTION")
        if row["recorded_at"] and row["executed_at"] and (datetime.fromisoformat(row["recorded_at"]).tzinfo is None) != (datetime.fromisoformat(row["executed_at"]).tzinfo is None):
            errors.append("TIMEZONE_CONVENTION_MISMATCH")
        if any(row[k] is not None for k in MONEY_FIELDS) and (not row["currency"] or not re.fullmatch(r"[A-Z]{3}", row["currency"])):
            errors.append("ACTUAL_CURRENCY_REQUIRED")
        if any(row[k] is not None for k in QUANTITY_FIELDS) and not row["quantity_unit"]:
            errors.append("ACTUAL_QUANTITY_UNIT_REQUIRED")
        if row["loss_basis"] and row["loss_basis"] not in LOSS_BASES:
            errors.append("INVALID_LOSS_BASIS")
        if any(row[k] is not None for k in ("actual_realized_loss", "actual_loss_debits", "actual_loss_credits")) and not row["loss_basis"]:
            errors.append("REALIZED_LOSS_BASIS_REQUIRED")
        if errors:
            result["errors"].extend({"row": index, "code": e} for e in sorted(set(errors)))
        else:
            result["rows"].append(row)
    result["status"] = "PARTIAL" if result["errors"] and result["rows"] else "INVALID" if result["errors"] else "VALID"
    return result


def realized_loss(outcome):
    """No forecast inputs. Sum only explicitly observed accounting debits/credits."""
    available = (outcome.get("execution_status") == "EXECUTED"
        and outcome.get("outcome_provenance") in OUTCOME_PROVENANCE[:3]
        and outcome.get("outcome_source") and outcome.get("currency") and outcome.get("loss_basis") in LOSS_BASES)
    explicit = outcome.get("actual_realized_loss")
    debit, credit = outcome.get("actual_loss_debits"), outcome.get("actual_loss_credits")
    calculated = debit - credit if finite(debit) and finite(credit) else None
    if available and finite(explicit) and calculated is not None and not math.isclose(explicit, calculated, rel_tol=0, abs_tol=1e-9):
        return {"status": "REALIZED_LOSS_CONFLICT", "value": None, "basis": outcome.get("loss_basis")}
    value = explicit if finite(explicit) else calculated
    if not available or not finite(value):
        return {"status": "REALIZED_LOSS_UNAVAILABLE", "value": None, "basis": outcome.get("loss_basis")}
    return {"status": "REALIZED_LOSS_AVAILABLE", "value": value, "currency": outcome["currency"],
            "basis": outcome["loss_basis"], "method": "OBSERVED_TOTAL" if finite(explicit) else "OBSERVED_DEBITS_MINUS_CREDITS"}


def compare_outcome(decision, outcome=None):
    """Calibration for the one observed action. Never declare a strategy winner."""
    out = {"status": "OUTCOME_PENDING", "winner": None, "counterfactual_computed": False,
        "legacy_expected_vs_actual": None, "seller_expected_vs_actual": None,
        "reason_codes": [], "meaning": "Observed-action calibration only; no claim that either recommendation is better."}
    if not outcome or outcome.get("execution_status") != "EXECUTED" or not any(outcome.get(k) is not None for k in ACTUAL_FIELDS):
        return out
    observed = ACTION_MAPPING.get(outcome.get("operator_action"))
    legacy = ACTION_MAPPING.get(decision.get("legacy_action"))
    seller = ACTION_MAPPING.get(decision.get("seller_loss_action"))
    if not observed:
        return {**out, "status": "NOT_COMPARABLE", "reason_codes": ["action_mapping_unavailable"]}
    if (outcome.get("currency") and outcome.get("currency") != decision.get("currency")
        or outcome.get("quantity_unit") and str(outcome["quantity_unit"]).casefold() != str(decision.get("quantity_unit")).casefold()):
        return {**out, "status": "NOT_COMPARABLE", "reason_codes": ["OUTCOME_CURRENCY_OR_UNIT_MISMATCH"]}
    loss = realized_loss(outcome)
    expected_field = {"TRANSFER":"expected_loss_transfer", "NORMAL_SALE":"expected_loss_normal_sale",
        "DISCOUNT_SALE":"expected_loss_discount_sale"}[observed]
    expected = decision.get(expected_field)
    if loss["status"] == "REALIZED_LOSS_AVAILABLE" and outcome.get("loss_basis") == "OBSERVED_AVOIDABLE_LOSS" and finite(expected):
        out["seller_expected_vs_actual"] = {"observed_strategy": observed, "expected_loss": expected,
            "actual_realized_loss": loss["value"], "actual_minus_expected": loss["value"] - expected,
            "seller_recommendation_executed": observed == seller}
    if observed == legacy == "TRANSFER" and finite(decision.get("legacy_cost")) and finite(outcome.get("actual_transfer_cost")):
        out["legacy_expected_vs_actual"] = {"metric": "transfer_cost", "expected": decision["legacy_cost"],
            "actual": outcome["actual_transfer_cost"], "actual_minus_expected": outcome["actual_transfer_cost"] - decision["legacy_cost"]}
    out["status"] = "CONFIRMED" if legacy == seller == observed and out["seller_expected_vs_actual"] else "PARTIAL_EVIDENCE"
    out["reason_codes"] = ["ONE_ACTION_OBSERVED_NO_COUNTERFACTUAL"]
    if not legacy:
        out["reason_codes"].append("legacy_not_comparable")
    if loss["status"] != "REALIZED_LOSS_AVAILABLE":
        out["reason_codes"].append(loss["status"])
    elif outcome.get("loss_basis") != "OBSERVED_AVOIDABLE_LOSS":
        out["reason_codes"].append("LOSS_BASIS_NOT_EXPECTED_AVOIDABLE_LOSS")
    return out


def contract_document():
    return {"version": CONTRACT_VERSION, "sheet": SHEET_KEY, "fields": [*META_FIELDS, *ACTUAL_FIELDS],
        "actions": list(OPERATOR_ACTIONS), "execution_statuses": list(EXECUTION_STATUSES),
        "outcome_provenance": list(OUTCOME_PROVENANCE), "loss_bases": list(LOSS_BASES),
        "rules": ["Blank is NULL, zero is observed zero.", "Full snapshots, not field patches.",
            "Actual values only after execution, with source, provenance and executed_at.",
            "All timestamps use one timezone convention per pilot.", "No actual values inferred from expected inputs.",
            "No counterfactual or absolute strategy winner.", "Multiple decision versions require explicit decision_version."]}
