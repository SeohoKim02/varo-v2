"""Seller Loss production input layer: real data + the seller's explicit business inputs -> SellerDecisionInput.

The Seller Loss Engine (services.seller_loss_engine) compares TRANSFER / NORMAL_SALE / DISCOUNT_SALE. Real datasets
rarely carry every business value it needs (price, holding/disposal cost, remaining shelf life, discount effect), so
this layer lets the seller state the missing values explicitly and keeps each value's origin visible:

    production row (inventory / route / real transport / forecast)        -> DIRECT_REAL, DERIVED_REAL, ...
  + explicit seller inputs: uploaded_data["seller_loss_inputs"] or the optional workbook sheet `seller_loss_inputs`
  -> merged SellerDecisionInput -> evaluate_seller_decision -> evidence, readiness and audit next to the result

Rules (no value is ever invented):
    * A blank cell stays MISSING. No price, cost, rate or uplift default exists anywhere in this module.
    * ACTUAL_OPERATION (default): a seller value fills a MISSING value or replaces a PROXY. It never replaces a value of the
      production row (real or uploaded); a different seller value is recorded as a conflict and the data value is kept.
      The workbook config promotion rows (legacy 20% / 80% placeholders) are not used.
    * SCENARIO: rows with input_type=SCENARIO are what-if values (SCENARIO_INPUT) applied on top of everything, and the
      config promotion rows are usable as CONFIG. Every result of a SCENARIO run says so and is never recommendable.
    * The most specific scope wins (ROUTE > PRODUCT_STORE > PRODUCT > STORE > GLOBAL). Two different values at the same
      specificity are a CONFLICT and that field is not used; there is no first/last-row rule and no nearest-key match.
    * Currencies are never converted; a declared price unit that cannot be checked against the inventory unit blocks.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from services.seller_loss_engine import (
    DISCOUNT_SALE, NORMAL_SALE, RECOMMENDED, STATUS_FULL, STATUS_PARTIAL, STATUS_UNAVAILABLE, STRATEGIES, TRANSFER,
    WORKBOOK_DATASET, InputField, SellerDecisionInput, _text_id, evaluate_seller_decision, known,
)

INPUT_CONTRACT_VERSION = "seller-loss-inputs-1.0.0"
SHEET_KEY = "seller_loss_inputs"
DECISION_MODE_KEY = "seller_loss_decision_mode"

ACTUAL_OPERATION, SCENARIO = "ACTUAL_OPERATION", "SCENARIO"
DECISION_MODES = (ACTUAL_OPERATION, SCENARIO)
# input_type column -> provenance of the value (a blank input_type is the seller's actual business value).
INPUT_TYPES = {"ACTUAL": "USER_INPUT", "SCENARIO": "SCENARIO_INPUT"}

REAL_PROVENANCE = frozenset({"DIRECT_REAL", "DERIVED_REAL"})
USER_PROVENANCE = frozenset({"USER_INPUT", "DERIVED_FROM_USER_INPUT"})
SCENARIO_CLASSES = frozenset({"SCENARIO_INPUT", "CONFIG"})

REAL_ONLY, REAL_PLUS_USER_INPUT, USER_INPUT_ONLY = "REAL_ONLY", "REAL_PLUS_USER_INPUT", "USER_INPUT_ONLY"
SCENARIO_EVIDENCE, INSUFFICIENT = "SCENARIO", "INSUFFICIENT"
EVIDENCE_LEVELS = {
    REAL_ONLY: "every input used by the comparison is DIRECT_REAL or DERIVED_REAL",
    REAL_PLUS_USER_INPUT: "real values plus values the seller entered (uploaded rows or seller_loss_inputs)",
    USER_INPUT_ONLY: "every input used is seller-provided (uploaded rows or seller_loss_inputs); none is verified real",
    SCENARIO_EVIDENCE: "a what-if value (SCENARIO_INPUT) or a workbook config setting (CONFIG) is used, or the run is SCENARIO",
    INSUFFICIENT: "fewer than two strategies are comparable: no fair comparison",
}
RECOMMENDABLE, NOT_ROBUST, SCENARIO_ONLY, UNAVAILABLE = "RECOMMENDABLE", "NOT_ROBUST", "SCENARIO_ONLY", "UNAVAILABLE"
READINESS = {
    RECOMMENDABLE: "ACTUAL_OPERATION, real/seller evidence, a winner that holds for every unknown value, no unresolved "
                   "seller-input conflict: the only class that may later be promoted to a production action",
    NOT_ROBUST: "comparable, but the cheapest strategy depends on unknown inputs or on conflicting seller inputs",
    SCENARIO_ONLY: "computed with what-if/config values or in a SCENARIO run: never a production recommendation",
    UNAVAILABLE: "no comparison (fewer than two strategies have complete inputs)",
}

SCOPES = ("ROUTE", "PRODUCT_STORE", "PRODUCT", "STORE", "GLOBAL")
SCOPE_RANK = {"ROUTE": 5, "PRODUCT_STORE": 4, "PRODUCT": 3, "STORE": 2, "GLOBAL": 1}
KEY_COLUMNS = ("product_id", "store_id", "source_store_id", "target_store_id", "route_id", "effective_date")
COLUMN_ALIASES = {"location_id": "store_id", "source_location_id": "source_store_id",
                  "target_location_id": "target_store_id"}
META_COLUMNS = ("scope", "input_type", "currency", "quantity_unit", "price_unit", "note")
PROMOTION_LABEL_COLUMNS = ("is_sample", "is_test", "synthetic_fixture", "data_kind", "source_file")


def promotion_data_labels(record):
    """Retain nonproduction declarations without changing any economic value."""
    labels = tuple(str(record[c]).strip() for c in ("note", "data_kind", "source_file", "profile_source")
                   if not _blank(record.get(c)))
    return labels + tuple(c for c in ("is_sample", "is_test", "synthetic_fixture")
                          if str(record.get(c, "")).lower() in ("true", "1", "yes"))

# Store attributes: the value of a product at one store. The same row serves that store as decision source or target.
STORE_FIELDS: dict[str, dict[str, str]] = {
    "normal_price": {"source": "source_normal_price", "target": "target_normal_price"},
    "holding_cost_per_unit_day": {"source": "source_holding_cost_per_unit_day", "target": "target_holding_cost_per_unit_day"},
    "disposal_cost_per_unit": {"source": "source_disposal_cost_per_unit", "target": "target_disposal_cost_per_unit"},
    "daily_demand": {"source": "source_daily_demand", "target": "target_daily_demand"},
    "remaining_shelf_life_days": {"source": "remaining_shelf_life_days"},
    "salvage_value_per_unit": {"source": "salvage_value_per_unit"},
    "unit_cost": {"source": "unit_cost"},
    "discount_rate": {"source": "discount_rate"},
    "promotion_uplift": {"source": "promotion_uplift"},
}
# Route attributes: moving the product from source_store_id to target_store_id. (engine field, transfer cost basis)
ROUTE_FIELDS: dict[str, tuple[str, str | None]] = {
    "transfer_cost": ("transfer_cost", "FIXED_PER_TRIP"),
    "transfer_cost_per_unit": ("transfer_cost", "PER_UNIT"),
    "transit_time_days": ("transit_time_days", None),
}
VALUE_COLUMNS = (*STORE_FIELDS, *ROUTE_FIELDS)
MONEY_COLUMNS = frozenset({"normal_price", "holding_cost_per_unit_day", "disposal_cost_per_unit",
                           "salvage_value_per_unit", "unit_cost", "transfer_cost", "transfer_cost_per_unit"})
# Values expressed per quantity unit (money per unit, units per day): their unit must match the inventory unit.
UNIT_BOUND_COLUMNS = frozenset({"normal_price", "holding_cost_per_unit_day", "disposal_cost_per_unit",
                                "salvage_value_per_unit", "unit_cost", "transfer_cost_per_unit", "daily_demand"})
VALUE_RULES = {name: "NON_NEGATIVE" for name in VALUE_COLUMNS}
VALUE_RULES["discount_rate"] = "FRACTION_0_TO_BELOW_1"
STORE_SCOPES = frozenset({"PRODUCT_STORE", "PRODUCT", "STORE", "GLOBAL"})
ROUTE_SCOPES = frozenset({"ROUTE", "PRODUCT", "GLOBAL"})
# Engine field names that belong to a role; entered as one store attribute instead (hint for misnamed columns).
ENGINE_NAME_HINTS = {engine: column for column, roles in STORE_FIELDS.items() for engine in roles.values() if engine != column}
# Engine fields this layer may set, with the sheet column(s) and role that feed them.
ENGINE_TARGETS: dict[str, tuple[tuple[str, ...], str | None]] = {
    **{engine: ((column,), role) for column, roles in STORE_FIELDS.items() for role, engine in roles.items()},
    "transfer_cost": (("transfer_cost", "transfer_cost_per_unit"), None),
    "transit_time_days": (("transit_time_days",), None),
}
# Inputs whose plus/minus variation a later sensitivity analysis may apply (all are plain mapping entries, no constants).
SENSITIVITY_FIELDS = ("discount_rate", "promotion_uplift", "source_disposal_cost_per_unit", "target_disposal_cost_per_unit",
                      "source_holding_cost_per_unit_day", "target_holding_cost_per_unit_day", "salvage_value_per_unit",
                      "source_normal_price", "target_normal_price", "source_daily_demand", "target_daily_demand",
                      "remaining_shelf_life_days", "transfer_cost", "transit_time_days")

# Inputs each strategy needs (decision_qty is the question itself, transfer_cost_qty describes transfer_cost).
CORE_REQUIRED = ("source_current_stock", "source_daily_demand", "remaining_shelf_life_days", "source_normal_price")
CORE_RATES = ("source_holding_cost_per_unit_day", "source_disposal_cost_per_unit", "salvage_value_per_unit")
STRATEGY_REQUIRED = {
    NORMAL_SALE: CORE_REQUIRED,
    DISCOUNT_SALE: (*CORE_REQUIRED, "discount_rate", "promotion_uplift"),
    TRANSFER: (*CORE_REQUIRED, "target_current_stock", "target_daily_demand", "target_normal_price", "transfer_cost",
               "transit_time_days"),
}
STRATEGY_OPTIONAL = {
    NORMAL_SALE: CORE_RATES,
    DISCOUNT_SALE: CORE_RATES,
    TRANSFER: (*CORE_RATES, "target_holding_cost_per_unit_day", "target_disposal_cost_per_unit", "source_surplus_cap",
               "target_need_cap", "route_capacity_qty"),
}
DECISION_SCOPE_FIELDS = ("decision_qty", "transfer_cost_qty")
READINESS_CODES = {
    "DECISION_QTY_MISSING": "MISSING_DECISION_QTY", "SOURCE_STOCK_MISSING": "MISSING_SOURCE_STOCK",
    "SOURCE_DEMAND_MISSING": "MISSING_SOURCE_DEMAND", "SHELF_LIFE_MISSING": "MISSING_REMAINING_SHELF_LIFE",
    "PRICE_MISSING": "MISSING_NORMAL_PRICE", "DISCOUNT_RATE_MISSING": "MISSING_DISCOUNT_RATE",
    "UPLIFT_MISSING": "MISSING_PROMOTION_UPLIFT", "TARGET_STORE_MISSING": "MISSING_TARGET_STORE",
    "TARGET_STOCK_MISSING": "MISSING_TARGET_STOCK", "TARGET_DEMAND_MISSING": "MISSING_TARGET_DEMAND",
    "TARGET_PRICE_MISSING": "MISSING_TARGET_PRICE", "TRANSFER_COST_MISSING": "MISSING_TRANSFER_COST",
    "TRANSIT_TIME_MISSING": "MISSING_TRANSIT_TIME",
}
_CURRENCY = re.compile(r"^[A-Z]{3}$")
TOL = 1e-9


# ---------------------------------------------------------------- parsing and validation


@dataclass(frozen=True)
class SellerInputCell:
    """One accepted value of the seller_loss_inputs sheet."""

    row: int                       # 1-based data row (header excluded)
    column: str                    # contract column (normal_price, transfer_cost, ..., quantity_unit)
    value: float | str
    scope: str
    keys: tuple[tuple[str, str], ...]
    input_type: str                # ACTUAL | SCENARIO
    currency: str | None = None    # money columns only
    currency_basis: str = ""
    unit: str | None = None        # declared quantity unit of a per-unit value (None = not declared on the row)
    basis: str | None = None       # transfer cost basis
    input_source: str = "seller_loss_inputs"
    effective_until: str | None = None
    profile_source: str | None = None
    data_labels: tuple[str, ...] = ()

    @property
    def key_map(self) -> dict[str, str]:
        return dict(self.keys)

    @property
    def provenance(self) -> str:
        return INPUT_TYPES[self.input_type]

    @property
    def specificity(self) -> tuple[int, int, int]:
        keys = self.key_map
        count = len([k for k in keys if k != "effective_date"]) if self.input_source == "seller_business_profile" else len(keys)
        return SCOPE_RANK[self.scope], int(bool(keys.get("route_id"))), count

    def label(self) -> str:
        keys = ", ".join(f"{k}={v}" for k, v in self.keys)
        return f"{self.input_source}[row {self.row}].{self.column} (scope {self.scope}{'; ' + keys if keys else ''})"


@dataclass
class SellerInputTable:
    cells: list[SellerInputCell] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return bool(self.cells)


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def parse_number(value: Any) -> tuple[float | None, str | None]:
    """(number, error code). Blank -> (None, None). Text that is not a plain number is an error, never 0."""
    if _blank(value):
        return None, None
    if isinstance(value, bool):
        return None, "NOT_A_NUMBER"
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None, "NOT_A_NUMBER"
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None, "NOT_A_NUMBER"
    if not math.isfinite(number):
        return None, "NOT_FINITE"
    return number, None


def _check_rule(column: str, number: float) -> str | None:
    if VALUE_RULES[column] == "FRACTION_0_TO_BELOW_1":
        return None if 0.0 <= number < 1.0 else "DISCOUNT_RATE_OUT_OF_RANGE"
    return None if number >= 0.0 else "NEGATIVE_VALUE"


def _date_text(value: Any) -> str | None:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _unit_text(value: Any) -> str | None:
    return None if _blank(value) else str(value).strip().upper()


ERROR_MESSAGES = {
    "NOT_A_TABLE": "seller_loss_inputs는 표(DataFrame 또는 행 목록)여야 합니다",
    "SCOPE_COLUMN_MISSING": "scope 열이 없습니다(각 행의 적용 범위를 명시해야 합니다)",
    "NO_VALUE_COLUMNS": "입력값 열이 하나도 없습니다",
    "INVALID_SCOPE": "scope는 ROUTE, PRODUCT_STORE, PRODUCT, STORE, GLOBAL 중 하나여야 합니다",
    "SCOPE_KEYS_MISMATCH": "scope와 식별 키 조합이 맞지 않습니다",
    "INVALID_INPUT_TYPE": "input_type은 ACTUAL 또는 SCENARIO여야 합니다",
    "INVALID_EFFECTIVE_DATE": "effective_date가 날짜가 아닙니다",
    "INVALID_CURRENCY": "currency는 KRW, USD 같은 3자리 통화 코드여야 합니다",
    "INVALID_PRICE_UNIT": "price_unit은 'KRW/EA'처럼 통화/수량단위 형식이어야 합니다",
    "PRICE_UNIT_CONFLICT": "price_unit이 같은 행의 currency 또는 quantity_unit과 다릅니다",
    "NOT_A_NUMBER": "숫자가 아닙니다(쉼표·단위·%를 넣지 말고 숫자만 입력)",
    "NOT_FINITE": "NaN 또는 무한대는 입력값으로 쓸 수 없습니다",
    "NEGATIVE_VALUE": "음수는 허용되지 않습니다",
    "DISCOUNT_RATE_OUT_OF_RANGE": "discount_rate는 0 이상 1 미만의 소수입니다(예: 20% 할인 = 0.2)",
    "FIELD_NOT_ALLOWED_IN_SCOPE": "이 입력값은 해당 scope에서 쓸 수 없습니다(점포 값은 store_id, 경로 값은 source/target)",
    "AMBIGUOUS_TRANSFER_COST_BASIS": "transfer_cost(1회 운송비)와 transfer_cost_per_unit(개당 운송비)를 한 행에 함께 쓸 수 없습니다",
}


def _scope_error(scope: str, keys: Mapping[str, str]) -> bool:
    has = {name for name in ("product_id", "store_id", "source_store_id", "target_store_id", "route_id") if keys.get(name)}
    if scope == "ROUTE":
        return "store_id" in has or not ("route_id" in has or {"source_store_id", "target_store_id"} <= has)
    if scope == "PRODUCT_STORE":
        return has != {"product_id", "store_id"}
    if scope == "PRODUCT":
        return has != {"product_id"}
    if scope == "STORE":
        return has != {"store_id"}
    return bool(has)  # GLOBAL


def parse_seller_loss_inputs(source: Any, *, upload_currency: str, upload_currency_basis: str) -> SellerInputTable:
    """Validate the seller_loss_inputs table. Never raises: problems are returned in ``validation``.

    ``upload_currency`` is the currency contract of the upload the sheet belongs to (config.currency, else the Varo
    workbook contract KRW); it applies to money cells whose row leaves currency and price_unit blank.
    """
    validation: dict[str, Any] = {"status": "ABSENT", "rows": 0, "accepted_cells": 0, "rejected_cells": 0,
                                  "actual_cells": 0, "scenario_cells": 0, "errors": [], "warnings": [],
                                  "ignored_columns": [], "upload_currency": upload_currency,
                                  "upload_currency_basis": upload_currency_basis}
    if source is None:
        return SellerInputTable([], validation)
    if isinstance(source, list):
        try:
            source = pd.DataFrame(source)
        except (TypeError, ValueError):
            source = None
    if not isinstance(source, pd.DataFrame):
        validation["status"] = "INVALID_SHEET"
        validation["errors"].append({"row": None, "column": None, "code": "NOT_A_TABLE", "message": ERROR_MESSAGES["NOT_A_TABLE"]})
        return SellerInputTable([], validation)
    frame = source.copy()
    frame.columns = [COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in frame.columns]
    frame = frame.loc[:, ~pd.Index(frame.columns).duplicated()]
    frame = frame[[not all(_blank(v) for v in row) for row in frame.itertuples(index=False)]] if len(frame) else frame
    validation["rows"] = int(len(frame))
    known_columns = {*KEY_COLUMNS, *META_COLUMNS, *VALUE_COLUMNS, *PROMOTION_LABEL_COLUMNS}
    for column in frame.columns:
        if column not in known_columns:
            hint = ENGINE_NAME_HINTS.get(column)
            validation["ignored_columns"].append(column)
            validation["warnings"].append({"row": None, "column": column, "code": "UNKNOWN_COLUMN",
                                           "message": f"알 수 없는 열이라 사용하지 않습니다"
                                                      + (f"(점포 값은 store_id와 {hint} 열로 입력)" if hint else "")})
    if frame.empty:
        validation["status"] = "EMPTY"
        return SellerInputTable([], validation)
    if "scope" not in frame.columns:
        validation["status"] = "INVALID_SHEET"
        validation["errors"].append({"row": None, "column": "scope", "code": "SCOPE_COLUMN_MISSING",
                                     "message": ERROR_MESSAGES["SCOPE_COLUMN_MISSING"]})
        return SellerInputTable([], validation)
    value_columns = [c for c in VALUE_COLUMNS if c in frame.columns]
    if not value_columns and "quantity_unit" not in frame.columns:
        validation["status"] = "INVALID_SHEET"
        validation["errors"].append({"row": None, "column": None, "code": "NO_VALUE_COLUMNS",
                                     "message": ERROR_MESSAGES["NO_VALUE_COLUMNS"]})
        return SellerInputTable([], validation)

    cells: list[SellerInputCell] = []
    errors: list[dict[str, Any]] = validation["errors"]

    def error(row: int, column: str | None, code: str, value: Any = None) -> None:
        errors.append({"row": row, "column": column, "code": code, "value": None if _blank(value) else repr(value),
                       "message": ERROR_MESSAGES.get(code, code)})

    for position, record in enumerate(frame.to_dict("records"), start=1):
        scope = str(record.get("scope") or "").strip().upper() if not _blank(record.get("scope")) else ""
        if scope not in SCOPES:
            error(position, "scope", "INVALID_SCOPE", record.get("scope"))
            validation["rejected_cells"] += sum(1 for c in value_columns if not _blank(record.get(c)))
            continue
        keys: dict[str, str] = {}
        for name in ("product_id", "store_id", "source_store_id", "target_store_id", "route_id"):
            text = _text_id(record.get(name))
            if text:
                keys[name] = text
        row_errors: list[tuple[str, str]] = []
        if not _blank(record.get("effective_date")):
            date = _date_text(record.get("effective_date"))
            if date is None:
                row_errors.append(("effective_date", "INVALID_EFFECTIVE_DATE"))
            else:
                keys["effective_date"] = date
        if _scope_error(scope, keys):
            row_errors.append(("scope", "SCOPE_KEYS_MISMATCH"))
        input_type = "ACTUAL" if _blank(record.get("input_type")) else str(record.get("input_type")).strip().upper()
        if input_type not in INPUT_TYPES:
            row_errors.append(("input_type", "INVALID_INPUT_TYPE"))
        currency = None if _blank(record.get("currency")) else str(record.get("currency")).strip().upper()
        if currency is not None and not _CURRENCY.match(currency):
            row_errors.append(("currency", "INVALID_CURRENCY"))
        quantity_unit = _unit_text(record.get("quantity_unit"))
        price_currency = price_unit = None
        if not _blank(record.get("price_unit")):
            parts = str(record.get("price_unit")).strip().upper().split("/")
            if len(parts) != 2 or not _CURRENCY.match(parts[0].strip()) or not parts[1].strip():
                row_errors.append(("price_unit", "INVALID_PRICE_UNIT"))
            else:
                price_currency, price_unit = parts[0].strip(), parts[1].strip()
                if (currency and currency != price_currency) or (quantity_unit and quantity_unit != price_unit):
                    row_errors.append(("price_unit", "PRICE_UNIT_CONFLICT"))
        if row_errors:
            for column, code in row_errors:
                error(position, column, code, record.get(column))
            validation["rejected_cells"] += sum(1 for c in value_columns if not _blank(record.get(c)))
            continue
        key_tuple = tuple(sorted(keys.items()))
        labels = promotion_data_labels(record)
        row_currency = currency or price_currency
        row_unit = quantity_unit or price_unit
        both_costs = not _blank(record.get("transfer_cost")) and not _blank(record.get("transfer_cost_per_unit"))
        row_cells = 0
        for column in value_columns:
            raw = record.get(column)
            if _blank(raw):
                continue
            number, code = parse_number(raw)
            code = code or _check_rule(column, number)
            if code is None and ((column in STORE_FIELDS and scope not in STORE_SCOPES)
                                 or (column in ROUTE_FIELDS and scope not in ROUTE_SCOPES)):
                code = "FIELD_NOT_ALLOWED_IN_SCOPE"
            if code is None and both_costs and column in ("transfer_cost", "transfer_cost_per_unit"):
                code = "AMBIGUOUS_TRANSFER_COST_BASIS"
            if code is not None:
                error(position, column, code, raw)
                validation["rejected_cells"] += 1
                continue
            money = column in MONEY_COLUMNS
            cells.append(SellerInputCell(
                row=position, column=column, value=number, scope=scope, keys=key_tuple, input_type=input_type,
                currency=(row_currency or upload_currency) if money else None,
                currency_basis=("ROW" if currency else "PRICE_UNIT" if price_currency else upload_currency_basis) if money else "",
                unit=row_unit if column in UNIT_BOUND_COLUMNS else None,
                basis=ROUTE_FIELDS[column][1] if column in ROUTE_FIELDS else None,
                data_labels=labels,
            ))
            row_cells += 1
        if quantity_unit and scope in STORE_SCOPES:
            cells.append(SellerInputCell(row=position, column="quantity_unit", value=quantity_unit, scope=scope,
                                         keys=key_tuple, input_type=input_type, data_labels=labels))
        elif not row_cells:
            validation["warnings"].append({"row": position, "column": None, "code": "EMPTY_ROW",
                                           "message": "적용할 입력값이 없는 행입니다"})
    validation["accepted_cells"] = sum(1 for c in cells if c.column != "quantity_unit")
    validation["actual_cells"] = sum(1 for c in cells if c.column != "quantity_unit" and c.input_type == "ACTUAL")
    validation["scenario_cells"] = sum(1 for c in cells if c.column != "quantity_unit" and c.input_type == "SCENARIO")
    validation["quantity_unit_declarations"] = sum(1 for c in cells if c.column == "quantity_unit")
    validation["status"] = "VALID_WITH_ERRORS" if errors else "VALID"
    return SellerInputTable(cells, validation)


def resolve_decision_mode(explicit: Any, uploaded_data: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Requested mode: argument > uploaded_data['seller_loss_decision_mode'] > config row > ACTUAL_OPERATION."""
    for basis, value in (("argument", explicit), (f"uploaded_data.{DECISION_MODE_KEY}", (uploaded_data or {}).get(DECISION_MODE_KEY)),
                         (f"config.{DECISION_MODE_KEY}", (config or {}).get(DECISION_MODE_KEY))):
        if _blank(value):
            continue
        mode = str(value).strip().upper()
        if mode in DECISION_MODES:
            return mode, {"decision_mode": mode, "basis": basis}
        return ACTUAL_OPERATION, {"decision_mode": ACTUAL_OPERATION, "basis": "default (invalid value rejected)",
                                  "error": {"code": "INVALID_DECISION_MODE", "source": basis, "value": repr(value),
                                            "message": "seller_loss_decision_mode는 ACTUAL_OPERATION 또는 SCENARIO여야 합니다"}}
    return ACTUAL_OPERATION, {"decision_mode": ACTUAL_OPERATION, "basis": "default"}


# ---------------------------------------------------------------- resolution and merge


@dataclass(frozen=True)
class DecisionKey:
    decision_id: str
    product_id: str
    source_store_id: str
    target_store_id: str | None
    decision_date: str | None


def _matches(cell: SellerInputCell, key: DecisionKey, role: str | None) -> bool:
    keys = cell.key_map
    if cell.input_source == "seller_business_profile" and keys.get("effective_date"):
        if not key.decision_date or not (keys["effective_date"] <= key.decision_date <= cell.effective_until):
            return False
    elif keys.get("effective_date") and keys["effective_date"] != key.decision_date:
        return False
    if keys.get("product_id") and keys["product_id"] != key.product_id:
        return False
    if role is not None:  # store attribute (or quantity_unit) for the store playing ``role``
        store = key.source_store_id if role == "source" else key.target_store_id
        return bool(store) and (not keys.get("store_id") or keys["store_id"] == store)
    if not key.target_store_id:
        return False
    return ((not keys.get("route_id") or keys["route_id"] == key.decision_id)
            and (not keys.get("source_store_id") or keys["source_store_id"] == key.source_store_id)
            and (not keys.get("target_store_id") or keys["target_store_id"] == key.target_store_id))


def _same(a: SellerInputCell, b: SellerInputCell) -> bool:
    if isinstance(a.value, str) or isinstance(b.value, str):
        return str(a.value) == str(b.value)
    return (abs(float(a.value) - float(b.value)) <= TOL * max(1.0, abs(float(a.value)))
            and a.basis == b.basis and a.currency == b.currency
            and (a.unit or "").casefold() == (b.unit or "").casefold())


@dataclass(frozen=True)
class Resolution:
    status: str                                   # RESOLVED | CONFLICT | NONE
    winners: tuple[SellerInputCell, ...] = ()     # top-specificity cells
    shadowed: tuple[SellerInputCell, ...] = ()    # matching cells of lower specificity

    @property
    def cell(self) -> SellerInputCell | None:
        return self.winners[0] if self.status == "RESOLVED" else None


def resolve_cells(cells: Iterable[SellerInputCell]) -> Resolution:
    """Most specific matching cells win; different values at the same specificity are a CONFLICT."""
    matching = sorted(cells, key=lambda c: (c.specificity, c.row), reverse=True)
    if not matching:
        return Resolution("NONE")
    overridden = []
    if any(c.input_source == "seller_loss_inputs" for c in matching):
        overridden = [c for c in matching if c.input_source == "seller_business_profile"]
        matching = [c for c in matching if c.input_source != "seller_business_profile"]
    top = matching[0].specificity
    winners = tuple(sorted((c for c in matching if c.specificity == top), key=lambda c: (c.row, c.column)))
    shadowed = tuple(sorted([c for c in matching if c.specificity != top] + overridden, key=lambda c: (c.row, c.column)))
    status = "RESOLVED" if all(_same(winners[0], other) for other in winners[1:]) else "CONFLICT"
    return Resolution(status, winners, shadowed)


def _engine_matches(table: SellerInputTable, key: DecisionKey, engine_field: str, input_type: str) -> list[SellerInputCell]:
    columns, role = ENGINE_TARGETS[engine_field]
    return [c for c in table.cells if c.column in columns and c.input_type == input_type and _matches(c, key, role)]


def _equal_values(item: InputField, cell: SellerInputCell, base: SellerDecisionInput, engine_field: str) -> bool:
    value = float(cell.value)
    if engine_field == "transfer_cost":
        basis, qty = base.transfer_cost_basis, base.transfer_cost_qty.value
        if cell.basis == "PER_UNIT" and basis == "QUANTITY_SPECIFIC" and qty is not None:
            value *= float(qty)
        elif cell.basis != basis and not {cell.basis, basis} <= {"FIXED_PER_TRIP", "QUANTITY_SPECIFIC"}:
            return False
    same_value = abs(float(item.value) - value) <= TOL * max(1.0, abs(value))
    return same_value and (cell.currency is None or item.currency is None or cell.currency == item.currency)


def _seller_item(cell: SellerInputCell, unit: str | None, dataset: str | None) -> InputField:
    scenario = cell.input_type == "SCENARIO"
    note = ("what-if scenario value entered by the seller, not observed" if scenario else
            "explicit seller business input")
    if cell.currency_basis and cell.currency_basis not in ("ROW", "PRICE_UNIT"):
        note += f"; currency from {cell.currency_basis}"
    if cell.data_labels:
        note += "; input labels: " + " | ".join(cell.data_labels)
    return known(cell.value, cell.provenance, cell.label(), unit=unit, currency=cell.currency,
                 dataset=None if scenario else dataset, note=note)


def merge_seller_inputs(
    base: SellerDecisionInput, table: SellerInputTable | None, *, decision_mode: str = ACTUAL_OPERATION,
    decision_date: str | None = None, dataset: str | None = WORKBOOK_DATASET, exclude_fields: Iterable[str] = (),
) -> tuple[SellerDecisionInput, dict[str, Any]]:
    """Merge explicit seller inputs into the production-row input; returns (merged input, merge record).

    ``dataset`` is the upload the seller inputs belong to (the same business as the production rows).
    ``exclude_fields`` leaves those engine fields (or "quantity_unit") untouched (leave-one-out audit).
    """
    key = DecisionKey(base.decision_id, base.product_id, base.source_store_id, base.target_store_id, decision_date)
    record: dict[str, Any] = {"decision_mode": decision_mode, "decision_date": decision_date, "fields": {},
                              "audit": [], "conflicts": [], "issues": [], "applied_fields": [], "ignored_scenario_cells": 0}
    if table is None or not table.usable:
        return base, record
    excluded = frozenset(exclude_fields)

    changes: dict[str, Any] = {}
    issues: list[str] = []
    # Decision quantity unit: a declared data unit wins; otherwise the seller may declare it (store scope, source store).
    data_unit = base.quantity_unit
    unit_resolution = resolve_cells(c for c in table.cells if c.column == "quantity_unit" and c.input_type == "ACTUAL"
                                    and "quantity_unit" not in excluded and _matches(c, key, "source"))
    effective_unit = data_unit
    if unit_resolution.status == "RESOLVED":
        declared = str(unit_resolution.cell.value)
        if data_unit is None:
            effective_unit = declared
            changes["quantity_unit"] = declared
            record["applied_fields"].append("quantity_unit")
            record["fields"]["quantity_unit"] = {"origin": "SELLER_LOSS_INPUTS", "outcome": "APPLIED",
                                                 "provenance": "USER_INPUT", "value": declared,
                                                 "source": unit_resolution.cell.label()}
            if unit_resolution.cell.input_source == "seller_business_profile":
                cell = unit_resolution.cell
                record["fields"]["quantity_unit"].update(origin="SELLER_BUSINESS_PROFILE", scope=cell.scope,
                    input_scope=cell.scope, profile_source=cell.profile_source or cell.label(), effective_value=declared)
                record["audit"].append({"field": "quantity_unit", "column": cell.column, "row": cell.row,
                    "scope": cell.scope, "input_scope": cell.scope, "source": cell.input_source,
                    "profile_source": cell.profile_source or cell.label(), "provenance": "USER_INPUT",
                    "input_value": declared, "effective_value": declared, "outcome": "APPLIED",
                    **({"data_labels": list(cell.data_labels)} if cell.data_labels else {})})
            elif unit_resolution.cell.data_labels:
                record["audit"].append({"field": "quantity_unit", "scope": unit_resolution.cell.scope,
                    "keys": dict(unit_resolution.cell.keys), "outcome": "APPLIED",
                    "data_labels": list(unit_resolution.cell.data_labels)})
        elif declared.casefold() != str(data_unit).casefold():
            record["conflicts"].append({"field": "quantity_unit", "kind": "DATA_VS_SELLER_INPUT", "data_value": data_unit,
                                        "seller_values": [declared], "rows": [unit_resolution.cell.row], "kept": data_unit})
    elif unit_resolution.status == "CONFLICT":
        record["conflicts"].append({"field": "quantity_unit", "kind": "SELLER_INPUT_SAME_SCOPE",
                                    "seller_values": sorted({str(c.value) for c in unit_resolution.winners}),
                                    "rows": [c.row for c in unit_resolution.winners], "kept": data_unit})

    for engine_field in ENGINE_TARGETS:
        if engine_field in excluded:
            continue
        base_item: InputField = getattr(base, engine_field)
        actual = resolve_cells(_engine_matches(table, key, engine_field, "ACTUAL"))
        scenario_cells = _engine_matches(table, key, engine_field, "SCENARIO")
        scenario = resolve_cells(scenario_cells) if decision_mode == SCENARIO else Resolution("NONE")
        if decision_mode != SCENARIO:
            record["ignored_scenario_cells"] += len(scenario_cells)
        if actual.status == "NONE" and scenario.status == "NONE":
            continue

        chosen: SellerInputCell | None = None
        outcome = ""
        if scenario.status == "RESOLVED":
            chosen, outcome = scenario.cell, "SCENARIO_OVERRIDE"
        elif base_item.present and base_item.provenance in (REAL_PROVENANCE | USER_PROVENANCE):
            outcome = "DATA_KEPT"
        elif actual.status == "RESOLVED":
            chosen = actual.cell
            outcome = {"PROXY": "APPLIED_OVER_PROXY", "CONFIG": "APPLIED_OVER_CONFIG"}.get(
                base_item.provenance if base_item.present else "", "APPLIED")
        else:
            outcome = "DATA_KEPT" if base_item.present else "NOT_RESOLVED"

        def audit(cell: SellerInputCell, cell_outcome: str) -> None:
            record["audit"].append({
                "field": engine_field, "column": cell.column, "row": cell.row, "scope": cell.scope,
                "keys": dict(cell.keys), "input_type": cell.input_type, "provenance": cell.provenance,
                "input_value": cell.value, "currency": cell.currency, "unit": cell.unit, "basis": cell.basis,
                "outcome": cell_outcome,
                **({"data_labels": list(cell.data_labels)} if cell.data_labels else {}),
                **({"input_scope": cell.scope, "profile_source": cell.profile_source or cell.label(),
                    "source": cell.input_source, "effective_value": changes.get(engine_field, base_item).value,
                    "effective_until": cell.effective_until, "currency_basis": cell.currency_basis}
                   if cell.input_source == "seller_business_profile" else {}),
            })

        if chosen is not None:
            unit = chosen.unit
            if chosen.column in UNIT_BOUND_COLUMNS:
                if chosen.unit is None:
                    unit = effective_unit   # per stock unit by the workbook contract
                elif effective_unit is None:
                    issues.append(f"SELLER_UNIT_UNVERIFIABLE:{engine_field}")
                elif chosen.unit.casefold() == str(effective_unit).casefold():
                    unit = effective_unit
                elif engine_field == "transfer_cost":
                    issues.append(f"UNIT_MISMATCH:{engine_field}")   # the engine does not size transfer_cost
            item = _seller_item(chosen, unit if engine_field != "transfer_cost" else None, dataset)
            changes[engine_field] = item
            if engine_field == "transfer_cost":
                changes["transfer_cost_basis"] = chosen.basis
                changes["transfer_cost_qty"] = InputField(source=f"not used for a {chosen.basis} transfer cost")
            record["applied_fields"].append(engine_field)
            winners = scenario.winners if outcome == "SCENARIO_OVERRIDE" else actual.winners
            for cell in winners:
                audit(cell, outcome)
            if outcome == "SCENARIO_OVERRIDE" and actual.status == "RESOLVED":
                for cell in actual.winners:
                    audit(cell, "OVERRIDDEN_BY_SCENARIO")
            record["fields"][engine_field] = {
                "origin": "SCENARIO" if outcome == "SCENARIO_OVERRIDE" else chosen.input_source.upper(), "outcome": outcome,
                "provenance": chosen.provenance, "value": chosen.value, "scope": chosen.scope,
                "rows": [c.row for c in winners], "source": chosen.label(),
                "replaced": base_item.as_dict() if base_item.present else None,
                **({"input_scope": chosen.scope, "profile_source": chosen.profile_source or chosen.label(),
                    "effective_value": chosen.value} if chosen.input_source == "seller_business_profile" else {}),
            }
        elif actual.status == "RESOLVED":   # a data value exists: kept, the seller value is compared with it
            cell = actual.cell
            same = _equal_values(base_item, cell, base, engine_field)
            kind = "REAL_VS_SELLER_INPUT" if base_item.provenance in REAL_PROVENANCE else "DATA_VS_SELLER_INPUT"
            cell_outcome = ("MATCHES_REAL" if kind == "REAL_VS_SELLER_INPUT" else "MATCHES_DATA") if same else (
                "REAL_KEPT_CONFLICT" if kind == "REAL_VS_SELLER_INPUT" else "DATA_KEPT_CONFLICT")
            for winner in actual.winners:
                audit(winner, cell_outcome)
            if not same:
                record["conflicts"].append({"field": engine_field, "kind": kind, "data_value": base_item.value,
                                            "data_provenance": base_item.provenance, "seller_values": [cell.value],
                                            "rows": [c.row for c in actual.winners], "kept": base_item.value})
            record["fields"][engine_field] = {"origin": "DATA", "outcome": cell_outcome, "provenance": base_item.provenance,
                                              "value": base_item.value, "source": base_item.source}
        if actual.status == "CONFLICT":
            for cell in actual.winners:
                audit(cell, "CONFLICT_NOT_USED")
            record["conflicts"].append({"field": engine_field, "kind": "SELLER_INPUT_SAME_SCOPE",
                                        "seller_values": sorted({c.value for c in actual.winners}, key=str),
                                        "rows": [c.row for c in actual.winners],
                                        "kept": base_item.value if base_item.present else None})
            record["fields"].setdefault(engine_field, {"origin": "DATA" if base_item.present else "NONE",
                                                       "outcome": "CONFLICT_NOT_USED", "provenance": base_item.provenance,
                                                       "value": base_item.value, "source": base_item.source})
        if scenario.status == "CONFLICT":
            for cell in scenario.winners:
                audit(cell, "CONFLICT_NOT_USED")
            record["conflicts"].append({"field": engine_field, "kind": "SCENARIO_INPUT_SAME_SCOPE",
                                        "seller_values": sorted({c.value for c in scenario.winners}, key=str),
                                        "rows": [c.row for c in scenario.winners], "kept": None})
        for cell in actual.shadowed + (scenario.shadowed if decision_mode == SCENARIO else ()):
            resolution = scenario if cell.input_type == "SCENARIO" else actual
            override = (cell.input_source == "seller_business_profile"
                        and any(c.input_source == "seller_loss_inputs" for c in resolution.winners))
            audit(cell, "OVERRIDDEN_BY_DECISION_INPUT" if override else "SHADOWED_BY_MORE_SPECIFIC_SCOPE")

    record["issues"] = issues
    if issues:
        changes["input_issues"] = tuple(dict.fromkeys((*base.input_issues, *issues)))
    return replace(base, **changes), record


# ---------------------------------------------------------------- evidence, readiness, audit


def _class(provenance: str) -> str:
    if provenance in REAL_PROVENANCE:
        return "REAL"
    if provenance in USER_PROVENANCE:
        return "USER"
    if provenance in SCENARIO_CLASSES:
        return "SCENARIO"
    return provenance


def _origin(name: str, item: InputField, fields_info: Mapping[str, Any]) -> str:
    if name in fields_info and fields_info[name]["origin"] in ("SELLER_LOSS_INPUTS", "SELLER_BUSINESS_PROFILE", "SCENARIO"):
        return fields_info[name]["origin"]
    if not item.present:
        return "NONE"
    return "CONFIG" if item.provenance == "CONFIG" else "DATA"


def used_fields(result: Mapping[str, Any], merged: SellerDecisionInput) -> list[str]:
    """Inputs that entered at least one comparable strategy (decision scope and sunk unit_cost excluded)."""
    items = merged.input_fields()
    names: list[str] = []
    for strategy in result.get("comparable_strategies") or []:
        for name in (*STRATEGY_REQUIRED[strategy], *STRATEGY_OPTIONAL[strategy]):
            if items[name].usable and name not in names:
                names.append(name)
    return names


def evidence_level(result: Mapping[str, Any], merged: SellerDecisionInput, decision_mode: str) -> str:
    if result["comparison_status"] == STATUS_UNAVAILABLE:
        return INSUFFICIENT
    items = merged.input_fields()
    classes = {_class(items[name].provenance) for name in used_fields(result, merged)}
    if decision_mode == SCENARIO or "SCENARIO" in classes:
        return SCENARIO_EVIDENCE
    if classes <= {"REAL"}:
        return REAL_ONLY
    return REAL_PLUS_USER_INPUT if "REAL" in classes else USER_INPUT_ONLY


def strategy_readiness(result: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for strategy in STRATEGIES:
        reasons = result["strategies"][strategy]["unavailable_reasons"]
        if result["strategies"][strategy]["available"]:
            out[strategy] = "READY"
            continue
        codes = []
        for code in reasons:
            head, _, name = code.partition(":")
            if head == "PROXY_REJECTED":
                codes.append(f"PROXY_ONLY_{name.upper()}")
            else:
                codes.append(READINESS_CODES.get(head, code))
        out[strategy] = "|".join(dict.fromkeys(codes)) or "UNAVAILABLE"
    return out


def missing_required(merged: SellerDecisionInput) -> tuple[dict[str, list[str]], list[str]]:
    items = merged.input_fields()
    by_strategy = {s: [name for name in STRATEGY_REQUIRED[s] if not items[name].usable] for s in STRATEGIES}
    if not merged.target_store_id:
        by_strategy[TRANSFER] = ["target_store_id", *by_strategy[TRANSFER]]
    optional = sorted({name for s in STRATEGIES for name in STRATEGY_OPTIONAL[s]
                       if name not in ("source_surplus_cap", "target_need_cap", "route_capacity_qty") and not items[name].usable})
    return by_strategy, optional


def readiness(result: Mapping[str, Any], level: str, decision_mode: str, conflicts: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if result["comparison_status"] == STATUS_UNAVAILABLE:
        return UNAVAILABLE, ["COMPARISON_UNAVAILABLE"]
    if decision_mode == SCENARIO:
        reasons.append("SCENARIO_RUN")
    if level == SCENARIO_EVIDENCE:
        reasons.extend(f"SCENARIO_VALUE:{name}" for name in result.get("scenario_inputs") or [])
        return SCENARIO_ONLY, reasons or ["SCENARIO_EVIDENCE"]
    same_scope = sorted({c["field"] for c in conflicts if c["kind"] in ("SELLER_INPUT_SAME_SCOPE", "SCENARIO_INPUT_SAME_SCOPE")})
    if result["recommendation_status"] != RECOMMENDED:
        reasons.append("RANKING_DEPENDS_ON_UNKNOWN")
    if same_scope:
        reasons.extend(f"INPUT_CONFLICT:{name}" for name in same_scope)
    if reasons:
        return NOT_ROBUST, reasons
    # Informational only: the production value was used, the contradicting seller value is in input_conflicts.
    reasons.extend(f"SELLER_INPUT_OVERRULED_BY_DATA:{c['field']}" for c in conflicts
                   if c["kind"] in ("REAL_VS_SELLER_INPUT", "DATA_VS_SELLER_INPUT"))
    if result["comparison_status"] == STATUS_PARTIAL:
        missing = next(s for s in STRATEGIES if s not in result["comparable_strategies"])
        reasons.append(f"PARTIAL_COMPARISON:{missing}_EXCLUDED")
    return RECOMMENDABLE, reasons


def discount_price_preview(merged: SellerDecisionInput) -> dict[str, Any] | None:
    """The markdown itself is computable from price and rate; how much more sells is never estimated here."""
    price, rate, uplift = merged.source_normal_price, merged.discount_rate, merged.promotion_uplift
    if not (price.usable and rate.usable):
        return None
    p, d = float(price.value), float(rate.value)
    return {"normal_price": p, "discount_rate": d, "discounted_unit_price": round(p * (1.0 - d), 4),
            "markdown_per_unit": round(p * d, 4), "currency": price.currency,
            "promotion_uplift": float(uplift.value) if uplift.usable else None,
            "promotion_uplift_status": "AVAILABLE" if uplift.usable else ("PROXY_REJECTED" if uplift.present else "MISSING"),
            "note": "할인 후 단가는 정상가와 할인율로 계산; 할인 후 판매량 변화는 promotion_uplift 근거가 있을 때만 사용"}


def _influence(base: SellerDecisionInput, record: Mapping[str, Any], result: Mapping[str, Any],
               remerge: Callable[[frozenset[str]], SellerDecisionInput] | None) -> dict[str, Any]:
    """Which seller inputs the recommendation depends on: without all of them, and leaving each field out (re-merged)."""
    applied = list(record.get("applied_fields") or [])
    if not applied:
        return {"seller_inputs_applied": False}
    winner = result.get("recommended_strategy")
    without = evaluate_seller_decision(base)
    per_field: dict[str, Any] = {}
    for name in sorted(applied) if remerge is not None else ():
        variant = evaluate_seller_decision(remerge(frozenset({name})))
        per_field[name] = {"recommended_strategy_without": variant["recommended_strategy"],
                           "comparison_status_without": variant["comparison_status"],
                           "changes_recommendation": variant["recommended_strategy"] != winner}
    return {"seller_inputs_applied": True,
            "recommended_strategy_without_seller_inputs": without["recommended_strategy"],
            "comparison_status_without_seller_inputs": without["comparison_status"],
            "seller_inputs_changed_recommendation": without["recommended_strategy"] != winner,
            "per_field": per_field,
            "fields_changing_recommendation": sorted(n for n, v in per_field.items() if v["changes_recommendation"])}


def decision_evidence(result: Mapping[str, Any], base: SellerDecisionInput, merged: SellerDecisionInput,
                      record: Mapping[str, Any], *, decision_mode: str,
                      remerge: Callable[[frozenset[str]], SellerDecisionInput] | None = None) -> dict[str, Any]:
    """Evidence, readiness, sources and audit for one evaluated decision (added next to the engine result)."""
    items = merged.input_fields()
    fields_info = record.get("fields") or {}
    used = used_fields(result, merged)
    level = evidence_level(result, merged, decision_mode)
    conflicts = list(record.get("conflicts") or [])
    ready, ready_reasons = readiness(result, level, decision_mode, conflicts)
    by_strategy, optional_unknown = missing_required(merged)
    comparable = result.get("comparable_strategies") or []
    sources = {}
    for name, item in items.items():
        info = fields_info.get(name, {})
        sources[name] = {"provenance": item.provenance, "origin": _origin(name, item, fields_info), "value": item.value,
                         "source": item.source, "scope": info.get("scope"), "outcome": info.get("outcome"),
                         "used_in_comparison": name in used}
        if info.get("profile_source"):
            sources[name].update({k: info[k] for k in ("input_scope", "profile_source", "effective_value")})
    listed = [name for name in items if name not in DECISION_SCOPE_FIELDS and items[name].usable]
    status = result["comparison_status"]
    return {
        "decision_mode": decision_mode,
        "evidence_level": level,
        "recommendation_readiness": ready,
        "readiness_reasons": ready_reasons,
        "production_promotion_candidate": ready == RECOMMENDABLE,
        "production_action_applied": False,
        "data_completeness_status": {STATUS_FULL: "FULL", STATUS_PARTIAL: "PARTIAL"}.get(status, INSUFFICIENT),
        "strategy_readiness": strategy_readiness(result),
        "input_sources": sources,
        "quantity_unit_source": (fields_info.get("quantity_unit") or {}).get("source")
        or ("data" if merged.quantity_unit else "undeclared"),
        "real_input_fields": [n for n in listed if _class(items[n].provenance) == "REAL"],
        "user_input_fields": [n for n in listed if _class(items[n].provenance) == "USER"],
        "seller_input_fields": [n for n in listed if sources[n]["origin"] == "SELLER_LOSS_INPUTS"],
        "scenario_input_fields": [n for n in listed if _class(items[n].provenance) == "SCENARIO"],
        "proxy_input_fields": [n for n in items if items[n].present and items[n].provenance == "PROXY"],
        "used_input_fields": used,
        "missing_required_fields": sorted({n for s in STRATEGIES for n in by_strategy[s]}),
        "missing_required_by_strategy": by_strategy,
        "missing_for_excluded_strategies": {s: by_strategy[s] for s in STRATEGIES if s not in comparable},
        "unknown_optional_fields": optional_unknown,
        "conflicting_fields": sorted({c["field"] for c in conflicts}),
        "input_conflicts": conflicts,
        "seller_input_issues": list(record.get("issues") or []),
        "seller_input_audit": list(record.get("audit") or []),
        "ignored_scenario_cells": record.get("ignored_scenario_cells", 0),
        "seller_input_influence": _influence(base, record, result, remerge),
        "discount_price_preview": discount_price_preview(merged),
    }


def evaluate_with_seller_inputs(base: SellerDecisionInput, table: SellerInputTable | None, *,
                                decision_mode: str = ACTUAL_OPERATION, decision_date: str | None = None,
                                dataset: str | None = WORKBOOK_DATASET) -> dict[str, Any]:
    """merge -> evaluate_seller_decision -> evidence; the engine result keys are unchanged, evidence keys are added."""
    options = dict(decision_mode=decision_mode, decision_date=decision_date, dataset=dataset)
    merged, record = merge_seller_inputs(base, table, **options)
    result = evaluate_seller_decision(merged)
    result.update(decision_evidence(
        result, base, merged, record, decision_mode=decision_mode,
        remerge=lambda exclude: merge_seller_inputs(base, table, exclude_fields=exclude, **options)[0]))
    if table and "seller_business_profile" in table.validation and result["comparison_status"] == STATUS_PARTIAL:
        result["production_promotion_candidate"] = False
    return result


def apply_scenario_overrides(inp: SellerDecisionInput, overrides: Mapping[str, Any], *, source: str = "sensitivity") -> SellerDecisionInput:
    """Sensitivity/what-if hook: replace engine fields by SCENARIO_INPUT values (units/currency of the replaced field)."""
    changes: dict[str, Any] = {}
    for name, value in overrides.items():
        current = getattr(inp, name)
        if not isinstance(current, InputField):
            raise KeyError(f"{name} is not an input field")
        changes[name] = known(value, "SCENARIO_INPUT", f"{source}:{name}", unit=current.unit, currency=current.currency,
                              note="scenario value (sensitivity / what-if), not observed")
    return replace(inp, **changes)


# ---------------------------------------------------------------- contract and template


def input_contract_document() -> dict[str, Any]:
    document = {
        "contract_version": INPUT_CONTRACT_VERSION,
        "sources": {"uploaded_data_key": SHEET_KEY, "workbook_sheet": SHEET_KEY,
                    "decision_mode": f"argument > uploaded_data['{DECISION_MODE_KEY}'] > config row '{DECISION_MODE_KEY}' > ACTUAL_OPERATION"},
        "decision_modes": {ACTUAL_OPERATION: "real data + the seller's actual business values; no assumption, no config promotion default",
                           SCENARIO: "adds input_type=SCENARIO rows (override every value, recorded) and config promotion rows; results are SCENARIO_ONLY"},
        "input_types": {k: v for k, v in INPUT_TYPES.items()},
        "key_columns": list(KEY_COLUMNS),
        "key_aliases": dict(COLUMN_ALIASES),
        "metadata_columns": list(META_COLUMNS),
        "scopes": {
            "ROUTE": "route_id, or source_store_id + target_store_id (product_id optional); route values only",
            "PRODUCT_STORE": "product_id + store_id; store values",
            "PRODUCT": "product_id; store values at every store, route values on every route of the product",
            "STORE": "store_id; store values of every product at the store",
            "GLOBAL": "no key; explicit value for everything",
        },
        "scope_precedence": list(SCOPES),
        "specificity": "(scope rank, route_id given, number of keys incl. effective_date); highest wins",
        "effective_date": "optional exact date key: the row applies only to decisions of that snapshot date (no nearest date)",
        "store_fields": {column: roles for column, roles in STORE_FIELDS.items()},
        "route_fields": {column: {"engine_field": engine, "transfer_cost_basis": basis} for column, (engine, basis) in ROUTE_FIELDS.items()},
        "validation_rules": {**VALUE_RULES, "currency": "ISO-like 3 letters", "price_unit": "CUR/UNIT, consistent with currency and quantity_unit",
                             "text_numbers": "plain numbers only; '1,000', '20%', NaN, inf and booleans are rejected, never set to 0"},
        "currency_rule": "Money cells use the row currency (or price_unit); a blank row currency follows the upload's currency contract "
                         "(config.currency, else the Varo workbook contract KRW). No exchange rate is ever applied: a different "
                         "currency makes the affected strategy or the comparison unavailable.",
        "unit_rule": "quantity_unit on a store-scope row declares the inventory unit when the data declares none. A value with a "
                     "declared unit must match the inventory unit; if the inventory unit is unknown the value blocks "
                     "(SELLER_UNIT_UNVERIFIABLE). Values without a declared unit are per inventory unit (workbook contract).",
        "merge_priority": ["SCENARIO_INPUT (SCENARIO mode only, recorded override)", "DIRECT_REAL", "DERIVED_REAL",
                           "uploaded production rows (USER_INPUT / DERIVED_FROM_USER_INPUT)",
                           "seller_loss_inputs USER_INPUT (fills MISSING, replaces PROXY or CONFIG)", "MISSING"],
        "conflict_rule": "Different values at the same specificity -> CONFLICT, field not used. A seller value that differs from a "
                         "production value -> production value kept, conflict recorded.",
        "evidence_levels": dict(EVIDENCE_LEVELS),
        "readiness": dict(READINESS),
        "strategy_required_fields": {s: list(v) for s, v in STRATEGY_REQUIRED.items()},
        "strategy_optional_rates": {s: list(v) for s, v in STRATEGY_OPTIONAL.items()},
        "discount_rule": "discount_rate alone gives the discounted unit price; DISCOUNT_SALE needs promotion_uplift "
                         "(DIRECT_REAL, USER_INPUT or SCENARIO_INPUT). Legacy promotion 20% / 80% placeholders are never used.",
        "sensitivity_fields": list(SENSITIVITY_FIELDS),
        "no_default_rule": "Blank stays MISSING. No price, cost, rate, uplift or 'industry average' is inserted.",
    }
    document["contract_signature"] = hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return document


TEMPLATE_COLUMNS = ("scope", "input_type", "product_id", "store_id", "source_store_id", "target_store_id", "route_id",
                    "effective_date", "normal_price", "unit_cost", "discount_rate", "promotion_uplift",
                    "holding_cost_per_unit_day", "disposal_cost_per_unit", "salvage_value_per_unit",
                    "remaining_shelf_life_days", "daily_demand", "transfer_cost", "transfer_cost_per_unit",
                    "transit_time_days", "currency", "quantity_unit", "price_unit", "note")


def template_frame() -> pd.DataFrame:
    """SAMPLE rows that show the format only: ids are placeholders and every number is an example, not a default."""
    rows = [
        {"scope": "GLOBAL", "disposal_cost_per_unit": 300, "currency": "KRW", "quantity_unit": "EA",
         "note": "SAMPLE: 모든 점포·상품의 폐기 처리비(원/EA). 실제 값으로 바꾸거나 행을 지우세요"},
        {"scope": "STORE", "store_id": "SAMPLE_STORE_A", "holding_cost_per_unit_day": 5, "currency": "KRW",
         "note": "SAMPLE: A점 보관비(원/EA/일)"},
        {"scope": "PRODUCT", "product_id": "SAMPLE_PRODUCT_1", "remaining_shelf_life_days": 4, "unit_cost": 2500,
         "currency": "KRW", "note": "SAMPLE: 상품 단위 값 (unit_cost는 매몰원가라 비교에는 쓰지 않음)"},
        {"scope": "PRODUCT_STORE", "product_id": "SAMPLE_PRODUCT_1", "store_id": "SAMPLE_STORE_A", "normal_price": 4900,
         "discount_rate": 0.2, "promotion_uplift": 0.5, "daily_demand": 12, "price_unit": "KRW/EA",
         "note": "SAMPLE: A점 정상가·할인율(0.2=20%)·할인 시 판매 증가율(0.5=+50%)·일 수요"},
        {"scope": "PRODUCT_STORE", "product_id": "SAMPLE_PRODUCT_1", "store_id": "SAMPLE_STORE_B", "normal_price": 5200,
         "daily_demand": 30, "price_unit": "KRW/EA", "note": "SAMPLE: B점(도착 후보) 정상가·일 수요"},
        {"scope": "ROUTE", "source_store_id": "SAMPLE_STORE_A", "target_store_id": "SAMPLE_STORE_B", "transfer_cost": 15000,
         "transit_time_days": 0.5, "currency": "KRW", "note": "SAMPLE: A→B 1회 운송비·이동시간(일). 실제 운송비 데이터가 있으면 그 값이 우선"},
        {"scope": "PRODUCT", "input_type": "SCENARIO", "product_id": "SAMPLE_PRODUCT_1", "discount_rate": 0.3,
         "note": "SAMPLE what-if: SCENARIO 모드에서만 사용, 결과는 SCENARIO_ONLY"},
    ]
    return pd.DataFrame(rows, columns=list(TEMPLATE_COLUMNS))
