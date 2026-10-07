"""Optional, explicit reusable USER_INPUT. No storage service, defaults or engine formulas.

Profiles use the existing seller input scope contract. Prices and policies require
closed inclusive validity windows; current demand/uplift require PRODUCT_STORE.
Remaining shelf life is never a reusable profile field. Overlapping equal-scope
versions conflict, rather than choosing the latest row. Decision input overrides
profile input, but neither overrides production data in ACTUAL_OPERATION.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from services import seller_loss_inputs as sli

SHEET_KEY = "seller_business_profile"
VERSION = "seller-business-profile-0.1"
STABLE = {"currency", "quantity_unit"}
WINDOWED = set(sli.VALUE_COLUMNS) - {"remaining_shelf_life_days", "unit_cost"}
FREQUENCY = {
    "currency": "BUSINESS", "quantity_unit": "PRODUCT",
    "holding_cost_per_unit_day": "STORE_OR_PRODUCT_POLICY",
    "disposal_cost_per_unit": "STORE_OR_PRODUCT_POLICY",
    "normal_price": "PRODUCT_OR_PRODUCT_STORE_POLICY",
    "discount_rate": "STORE_OR_PRODUCT_POLICY",
    "salvage_value_per_unit": "PRODUCT_OR_PRODUCT_STORE_POLICY",
    "promotion_uplift": "PRODUCT_STORE_TIME_WINDOW",
    "daily_demand": "PRODUCT_STORE_TIME_WINDOW",
    "transfer_cost": "ROUTE_TIME_WINDOW", "transfer_cost_per_unit": "ROUTE_TIME_WINDOW",
    "transit_time_days": "ROUTE_TIME_WINDOW",
    "remaining_shelf_life_days": "DECISION_BATCH",
    "source_current_stock": "DECISION_BATCH", "target_current_stock": "DECISION_BATCH",
    "decision_qty": "DECISION_BATCH", "source_surplus_cap": "DECISION_BATCH",
    "target_need_cap": "DECISION_BATCH", "unit_cost": "NOT_REQUIRED_SUNK_COST",
}


def _date(value):
    if sli._blank(value):
        return None
    try:
        date = pd.Timestamp(value)
        return None if pd.isna(date) else date.strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        return None


def parse_profile(source: Any) -> sli.SellerInputTable:
    """Invalid profile cells become validation errors, never a pipeline exception."""
    validation = {"status": "ABSENT", "contract_version": VERSION, "errors": [], "accepted_cells": 0}
    if source is None:
        return sli.SellerInputTable([], validation)
    if isinstance(source, list):
        try:
            source = pd.DataFrame(source)
        except (ValueError, TypeError):
            source = None
    if not isinstance(source, pd.DataFrame):
        validation.update(status="INVALID_SHEET", errors=[{"code": "NOT_A_TABLE"}])
        return sli.SellerInputTable([], validation)
    frame = source.copy()
    frame.columns = [sli.COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in frame.columns]
    if frame.columns.duplicated().any():
        validation.update(status="INVALID_SHEET", errors=[{"code": "DUPLICATE_COLUMNS"}])
        return sli.SellerInputTable([], validation)
    rows = frame.to_dict("records")
    global_currency = {str(r.get("currency")).strip().upper() for r in rows
                       if str(r.get("scope", "")).upper() == "GLOBAL" and not sli._blank(r.get("currency"))
                       and sli._blank(r.get("effective_date")) and sli._blank(r.get("effective_until"))
                       and str(r.get("input_type", "ACTUAL")).upper() in ("ACTUAL", "NAN", "")
                       and not sli._scope_error("GLOBAL", {k: sli._text_id(r.get(k)) for k in sli.KEY_COLUMNS})}
    global_currency = {v for v in global_currency if sli._CURRENCY.fullmatch(v)}
    if len(global_currency) > 1:
        validation["errors"].append({"code": "GLOBAL_CURRENCY_CONFLICT"})
    cells = []
    for row_no, row in enumerate(rows, 1):
        scope = str(row.get("scope", "")).strip().upper()
        start, end = _date(row.get("effective_date")), _date(row.get("effective_until"))
        has_window = not sli._blank(row.get("effective_date")) or not sli._blank(row.get("effective_until"))
        valid_window = bool(start and end and start <= end)
        errors = []
        if has_window and not valid_window:
            errors.append({"row": row_no, "code": "INVALID_EFFECTIVE_WINDOW"})
            validation["errors"].extend(errors)
            continue
        work = {k: v for k, v in row.items() if k not in {"effective_date", "effective_until", "profile_source"}}
        currency = row.get("currency")
        if sli._blank(currency) and len(global_currency) == 1:
            work["currency"] = next(iter(global_currency))
        for column in sli.VALUE_COLUMNS:
            if sli._blank(row.get(column)):
                continue
            code = None
            if column == "remaining_shelf_life_days":
                code = "DECISION_BATCH_ONLY"
            elif column == "unit_cost":
                code = "NOT_USED_BY_LOSS_COMPARISON"
            elif column in WINDOWED and not valid_window:
                code = "EFFECTIVE_WINDOW_REQUIRED"
            elif column in {"daily_demand", "promotion_uplift"} and scope != "PRODUCT_STORE":
                code = "PRODUCT_STORE_WINDOW_REQUIRED"
            elif column in sli.ROUTE_FIELDS and scope != "ROUTE":
                code = "ROUTE_SCOPE_REQUIRED"
            elif column == "normal_price" and scope not in {"PRODUCT", "PRODUCT_STORE"}:
                code = "PRODUCT_PRICE_SCOPE_REQUIRED"
            elif column in sli.MONEY_COLUMNS and sli._blank(work.get("currency")) and sli._blank(work.get("price_unit")):
                code = "EXPLICIT_CURRENCY_REQUIRED"
            if code:
                work.pop(column, None)
                errors.append({"row": row_no, "column": column, "code": code})
        # Metadata-only GLOBAL currency is valid; keep an empty value column so
        # the existing parser still validates scope/currency/input_type.
        work.setdefault("normal_price", None)
        parsed = sli.parse_seller_loss_inputs([work], upload_currency="", upload_currency_basis="explicit profile only")
        for error in parsed.validation.get("errors", []):
            errors.append({**error, "row": row_no})
        for cell in parsed.cells:
            keys = dict(cell.keys)
            if start:
                keys["effective_date"] = start
            currency_basis = cell.currency_basis
            if cell.currency and sli._blank(row.get("currency")) and sli._blank(row.get("price_unit")) and len(global_currency) == 1:
                currency_basis = "seller_business_profile GLOBAL.currency"
                inherited_labels = tuple(label for r in rows
                    if str(r.get("scope", "")).upper() == "GLOBAL" and str(r.get("currency", "")).upper() in global_currency
                    for label in sli.promotion_data_labels(r))
                cell = replace(cell, data_labels=(*cell.data_labels, *inherited_labels))
            cells.append(replace(cell, row=row_no, keys=tuple(sorted(keys.items())), input_source=SHEET_KEY,
                                 effective_until=end, currency_basis=currency_basis, profile_source=str(row.get("profile_source"))
                                 if not sli._blank(row.get("profile_source")) else f"{SHEET_KEY}[row {row_no}]"))
        validation["errors"].extend(errors)
    validation.update(status="VALID" if not validation["errors"] else "PARTIAL" if cells else "INVALID_SHEET",
                      rows=len(rows), accepted_cells=len(cells), explicit_global_currencies=sorted(global_currency))
    return sli.SellerInputTable(cells, validation)


def combine_inputs(table: sli.SellerInputTable | None, profile: sli.SellerInputTable | None):
    if profile is None or profile.validation.get("status") == "ABSENT":
        return table
    return sli.SellerInputTable([*(table.cells if table else []), *profile.cells],
                               {**(table.validation if table else {}), SHEET_KEY: profile.validation})


def enrich_plan(plan, merged, record, table):
    """Only partition actual planner requests; never hide or remove a requirement."""
    if table is None or SHEET_KEY not in table.validation:
        return plan
    available = [name for name, info in record.get("fields", {}).items()
                 if info.get("origin") == "SELLER_BUSINESS_PROFILE"]
    setup, decision, windowed = [], [], []
    for item in plan["required_user_inputs"]:
        entry = item.get("seller_entry") or {}
        column = entry.get("column")
        if column in WINDOWED and column not in {"daily_demand", "promotion_uplift"} or column in STABLE:
            setup.append(item)
        elif column in {"daily_demand", "promotion_uplift"}:
            windowed.append(item)
        else:
            decision.append(item)
    plan.update(profile_available_fields=available,
                profile_missing_fields=[i["field"] for i in [*setup, *windowed]],
                decision_specific_required_inputs=decision,
                time_window_required_inputs=windowed,
                one_time_setup_inputs=setup,
                profile_validation=table.validation[SHEET_KEY],
                profile_count_note="Setup is reusable only within explicitly stated keys/validity windows; demand and uplift need renewed evidence.")
    return plan


def evaluate_profile_decision(base, profile=None, table=None, **kwargs):
    from services.seller_loss_input_requirements import plan_input_requirements
    return plan_input_requirements(base, combine_inputs(table, profile), **kwargs)


def contract_document():
    return {"version": VERSION, "sheet": SHEET_KEY, "scopes": list(sli.SCOPES), "field_frequency": FREQUENCY,
            "priority": ["production DIRECT_REAL/DERIVED_REAL", "production USER_INPUT",
                         "decision seller_loss_inputs", "seller_business_profile", "MISSING"],
            "within_profile": "ROUTE > PRODUCT_STORE > PRODUCT > STORE > GLOBAL; equal specificity conflicts",
            "effective_dates": "Closed inclusive effective_date/effective_until; no latest/nearest fallback; overlap conflicts",
            "provenance": "ACTUAL profile remains USER_INPUT; SCENARIO remains SCENARIO_INPUT",
            "no_default": True, "remaining_shelf_life": "decision/batch only",
            "demand_uplift": "PRODUCT_STORE with explicit finite validity window; never perpetual GLOBAL",
            "production_action_applied": False}
