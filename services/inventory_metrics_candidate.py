"""Explicit-basis inventory metrics, CANDIDATE only; no legacy defaults changed."""
from __future__ import annotations

import math
from datetime import date
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from services import inventory_metric_integrity as existing
from services.shared_feasibility_selection import normalize_unit

VERSION = "inventory-metrics-candidate-v0.1"
ABC_BASES = {"SALES_REVENUE": ("sales_qty", "unit_price"),
             "SALES_COST_VALUE": ("sales_qty", "unit_cost"),
             "SALES_QUANTITY": ("sales_qty", None),
             "OUTBOUND_QUANTITY": ("outbound_qty", None)}


def _number(value):
    try:
        result = float(value)
        return result if not isinstance(value, bool) and math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def _unavailable(*reasons, status="INPUT_REQUIRED"):
    return {"version": VERSION, "status": status, "value": None, "reason_codes": list(reasons),
            "production_approval": False}


def _period(start, end):
    try:
        return date.fromisoformat(str(start)) <= date.fromisoformat(str(end))
    except (ValueError, TypeError):
        return False


def _declared_text(value):
    return isinstance(value, str) and value.strip().upper() not in {"", "UNKNOWN", "UNDECLARED", "NONE", "NAN", "N/A"}


def abc_explicit_basis(rows: Sequence[Mapping[str, Any]], *, basis: str) -> dict[str, Any]:
    """No auto-switch from money to quantity and no missing cost multiplied by one.

    Each row is one product aggregate over an explicit identical period/location
    scope; duplicates must be resolved by a lineage-preserving upstream adapter.
    """
    if basis not in ABC_BASES:
        return _unavailable("UNKNOWN_ABC_BASIS", status="INVALID_INPUT")
    if not rows:
        return _unavailable("NO_ROWS")
    required = ("product_key", "network_id", "location_scope", "period_start", "period_end", "quantity_unit")
    if any(any(r.get(k) in (None, "") for k in required) for r in rows):
        return _unavailable("IDENTITY_PERIOD_UNIT_REQUIRED")
    scope = {(r["network_id"], r["location_scope"], r["period_start"], r["period_end"]) for r in rows}
    if len(scope) != 1:
        return _unavailable("INCOMPARABLE_SCOPE_PERIOD", status="NOT_COMPARABLE")
    if any(not _period(r["period_start"], r["period_end"]) for r in rows):
        return _unavailable("INVALID_PERIOD", status="INVALID_INPUT")
    if any(normalize_unit(r["quantity_unit"]) is None for r in rows):
        return _unavailable("DECLARED_QUANTITY_UNIT_REQUIRED")
    ids = [r["product_key"] for r in rows]
    if len(set(ids)) != len(ids):
        return _unavailable("DUPLICATE_PRODUCT_SCOPE", status="INVALID_INPUT")
    qfield, pricefield = ABC_BASES[basis]
    expected_signal = existing.OBSERVED_OUTBOUND_FLOW if basis == "OUTBOUND_QUANTITY" else existing.OBSERVED_RETAIL_SALES
    if any(r.get("signal_type") != expected_signal for r in rows):
        return _unavailable("SIGNAL_MEANING_MISMATCH", status="NOT_COMPARABLE")
    if pricefield:
        currencies = {r.get("currency") for r in rows}
        if len(currencies) != 1 or not all(_declared_text(v) for v in currencies):
            return _unavailable("COMMON_CURRENCY_REQUIRED")
        if any(normalize_unit(r.get("price_quantity_unit")) != normalize_unit(r["quantity_unit"]) for r in rows):
            return _unavailable("PRICE_QUANTITY_UNIT_MISMATCH", status="NOT_COMPARABLE")
    elif len({normalize_unit(r["quantity_unit"]) for r in rows}) != 1:
        return _unavailable("MIXED_QUANTITY_UNITS", status="NOT_COMPARABLE")
    values = {}
    for row in rows:
        quantity = _number(row.get(qfield))
        price = _number(row.get(pricefield)) if pricefield else 1.0
        if quantity is None or price is None:
            return _unavailable("BASIS_VALUE_MISSING_OR_INVALID")
        values[row["product_key"]] = quantity * price
    grades = existing.abc_grades(values)
    return {"version": VERSION, "status": "COMPUTED" if grades else "ZERO_TOTAL_UNCLASSIFIED", "basis": basis,
            "grades": grades, "total": sum(values.values()), "thresholds": existing.abc_thresholds(),
            "quantity_meaning": expected_signal, "production_approval": False}


def aligned_turnover(sales: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse T4 turnover after requiring explicit same identity/unit/period."""
    identity = ("dataset_id", "network_id", "location_id", "product_id", "product_state", "lot_id")
    if any(not sales.get(k) or not inventory.get(k) for k in identity[:4]):
        return _unavailable("IDENTITY_REQUIRED")
    if any(sales.get(k) != inventory.get(k) for k in identity):
        return _unavailable("IDENTITY_MISMATCH", status="NOT_COMPARABLE")
    if normalize_unit(sales.get("quantity_unit")) is None or normalize_unit(sales.get("quantity_unit")) != normalize_unit(inventory.get("quantity_unit")):
        return _unavailable("QUANTITY_UNIT_MISMATCH", status="NOT_COMPARABLE")
    sp = (sales.get("window_start"), sales.get("window_end"))
    ip = (inventory.get("period_start"), inventory.get("period_end"))
    if None in sp or None in ip or sp != ip:
        return _unavailable("PERIOD_MISMATCH", status="NOT_COMPARABLE")
    if not _period(*sp):
        return _unavailable("INVALID_PERIOD", status="INVALID_INPUT")
    return {"version": VERSION, **existing.physical_turnover(sales, inventory), "production_approval": False}


def safety_stock(*, daily_mean=None, daily_std=None, lead_time_days=None, cycle_service_level=None,
                 quantity_unit=None, demand_basis=None, stationary_independent=False, constant_lead_time=False):
    """Normal lead-time-demand approximation, explicitly not a fill-rate guarantee.

    SS = Phi^-1(CSL) sigma_daily sqrt(L); ROP = mean_daily L + SS.
    Assumptions must be declared.  Zero observed variability stays zero.
    """
    mean, std, lead = map(_number, (daily_mean, daily_std, lead_time_days))
    level = _number(cycle_service_level)
    if None in (mean, std, lead, level) or normalize_unit(quantity_unit) is None or not _declared_text(demand_basis):
        return _unavailable("MEAN_STD_LEAD_CSL_UNIT_BASIS_REQUIRED")
    if lead <= 0 or not 0.5 < level < 1:
        return _unavailable("INVALID_LEAD_OR_SERVICE_LEVEL", status="INVALID_INPUT")
    if not stationary_independent or not constant_lead_time:
        return _unavailable("NORMAL_IID_CONSTANT_LEAD_ASSUMPTIONS_NOT_MET", status="NOT_APPLICABLE")
    z = NormalDist().inv_cdf(level)
    value = z * std * math.sqrt(lead)
    return {"version": VERSION, "status": "COMPUTED_UNDER_ASSUMPTIONS", "value": value,
            "reorder_point": mean * lead + value, "cycle_service_level": level, "z": z,
            "quantity_unit": quantity_unit, "demand_basis": demand_basis, "fill_rate_guaranteed": False,
            "production_approval": False}


def eoq(*, annual_demand=None, order_cost=None, annual_holding_cost_per_unit=None, quantity_unit=None,
        currency=None, stationary_demand=False, perishability=None, capacity_qty=None, minimum_lot=None):
    """Classical unconstrained EOQ, never a feasible order recommendation by itself."""
    demand, setup, holding = map(_number, (annual_demand, order_cost, annual_holding_cost_per_unit))
    if None in (demand, setup, holding) or normalize_unit(quantity_unit) is None or not _declared_text(currency):
        return _unavailable("DEMAND_ORDER_HOLDING_UNIT_CURRENCY_REQUIRED")
    if holding <= 0 or setup <= 0:
        return _unavailable("POSITIVE_ORDER_AND_HOLDING_REQUIRED", status="INVALID_INPUT")
    if stationary_demand is not True or perishability is not False:
        return _unavailable("CLASSICAL_EOQ_ASSUMPTIONS_NOT_MET", status="NOT_APPLICABLE")
    quantity = math.sqrt(2 * demand * setup / holding)
    limitations = []
    if capacity_qty is not None:
        cap = _number(capacity_qty)
        if cap is None:
            return _unavailable("INVALID_CAPACITY", status="INVALID_INPUT")
        if quantity > cap:
            limitations.append("CAPACITY_EXCEEDED_UNCONSTRAINED_REFERENCE_ONLY")
    if minimum_lot is not None:
        lot = _number(minimum_lot)
        if lot is None or lot <= 0:
            return _unavailable("INVALID_LOT", status="INVALID_INPUT")
        if quantity > 0 and not math.isclose(quantity / lot, round(quantity / lot), abs_tol=1e-9):
            limitations.append("LOT_ROUNDING_NOT_OPTIMIZED")
    total = demand / quantity * setup + quantity / 2 * holding if quantity else 0.0
    return {"version": VERSION, "status": "REFERENCE_ONLY" if limitations else "COMPUTED_UNDER_ASSUMPTIONS",
            "value": quantity, "annual_order_holding_cost": total, "reason_codes": limitations,
            "currency": currency, "quantity_unit": quantity_unit, "production_approval": False}


def expiry_exposure(*, on_hand=None, days_to_expiry=None, daily_demand=None, quantity_unit=None,
                    disposal_history=None, expiry_dates_observed=False):
    """A deterministic no-replenishment scenario, not a calibrated waste probability."""
    stock, days, demand = map(_number, (on_hand, days_to_expiry, daily_demand))
    probability = existing.disposal_probability(disposal_history, expiry_dates_observed)
    if None in (stock, days, demand) or normalize_unit(quantity_unit) is None:
        return {**_unavailable("STOCK_REMAINING_LIFE_DEMAND_UNIT_REQUIRED"), "probability": probability}
    return {"version": VERSION, "status": "SCENARIO_ESTIMATE", "value": max(0.0, stock - demand * days),
            "meaning": "POTENTIAL_UNSOLD_AT_EXPIRY", "quantity_unit": quantity_unit,
            "is_observed_waste": False, "probability": probability, "production_approval": False}
