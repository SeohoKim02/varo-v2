"""Candidate-only inventory metrics: units, periods, explicit assumptions and no fake defaults."""
import copy
import math

import pytest

from services import inventory_metric_integrity as legacy
from services import inventory_metrics_candidate as candidate


def row(product="shop:p1", **kw):
    return {"product_key": product, "network_id": "shop", "location_scope": "shop:A",
            "period_start": "2026-07-01", "period_end": "2026-07-30", "quantity_unit": "EA",
            "signal_type": legacy.OBSERVED_RETAIL_SALES, "sales_qty": 10,
            "unit_price": 1000, "unit_cost": 400, "price_quantity_unit": "EA", "currency": "KRW", **kw}


@pytest.mark.parametrize("basis,total", [("SALES_REVENUE", 10000), ("SALES_COST_VALUE", 4000), ("SALES_QUANTITY", 10)])
def test_abc_explicit_bases(basis, total):
    result = candidate.abc_explicit_basis([row()], basis=basis)
    assert result["total"] == total and result["basis"] == basis
    assert result["production_approval"] is False


def test_outbound_is_not_retail_sales():
    out = row(signal_type=legacy.OBSERVED_OUTBOUND_FLOW, outbound_qty=20)
    assert candidate.abc_explicit_basis([out], basis="SALES_REVENUE")["status"] == "NOT_COMPARABLE"
    assert candidate.abc_explicit_basis([out], basis="OUTBOUND_QUANTITY")["total"] == 20


@pytest.mark.parametrize("changes,reason", [
    ({"unit_cost": None}, "BASIS_VALUE_MISSING_OR_INVALID"),
    ({"sales_qty": float("nan")}, "BASIS_VALUE_MISSING_OR_INVALID"),
    ({"sales_qty": -1}, "BASIS_VALUE_MISSING_OR_INVALID"),
    ({"sales_qty": True}, "BASIS_VALUE_MISSING_OR_INVALID"),
    ({"quantity_unit": "UNKNOWN"}, "DECLARED_QUANTITY_UNIT_REQUIRED"),
    ({"currency": None}, "COMMON_CURRENCY_REQUIRED"),
    ({"price_quantity_unit": "KG"}, "PRICE_QUANTITY_UNIT_MISMATCH"),
    ({"period_start": "2026-08-01"}, "INVALID_PERIOD"),
])
def test_abc_missing_never_gets_implicit_cost_one(changes, reason):
    result = candidate.abc_explicit_basis([row(**changes)], basis="SALES_COST_VALUE")
    assert result["value"] is None and reason in result["reason_codes"]


def test_abc_scope_duplicate_and_zero_total():
    assert candidate.abc_explicit_basis([row(), row()], basis="SALES_QUANTITY")["status"] == "INVALID_INPUT"
    mixed = [row(), row("shop:p2", location_scope="shop:B")]
    assert candidate.abc_explicit_basis(mixed, basis="SALES_QUANTITY")["status"] == "NOT_COMPARABLE"
    assert candidate.abc_explicit_basis([row(sales_qty=0)], basis="SALES_QUANTITY")["status"] == "ZERO_TOTAL_UNCLASSIFIED"


def turnover_pair(unit="EA", divisor=1):
    records = [{"date": f"2026-07-{day:02}", "quantity": 10 / divisor} for day in range(1, 31)]
    stock = [{**r, "quantity": 100 / divisor} for r in records]
    sales = legacy.aggregate_daily_window(records, window_end="2026-07-30", window_days=30,
        dataset_id="shop", signal_type=legacy.OBSERVED_RETAIL_SALES, quantity_unit=unit,
        zero_semantics=legacy.ZERO_RECORDED, semantics_evidence=legacy.SOURCE_DOCUMENTED)
    inventory = legacy.average_inventory(stock, period_start="2026-07-01", period_end="2026-07-30",
        dataset_id="shop", quantity_unit=unit)
    for item in (sales, inventory):
        item.update(network_id="shop", location_id="A", product_id="p1", product_state="FROZEN", lot_id="L1")
    return sales, inventory


def test_turnover_is_hand_calculated_and_unit_invariant():
    ea = candidate.aligned_turnover(*turnover_pair())
    box = candidate.aligned_turnover(*turnover_pair("BOX", 10))
    assert ea["turnover_value"] == box["turnover_value"] == 3.0
    assert ea["turnover_basis"] == legacy.REAL_SALES_TURNOVER


@pytest.mark.parametrize("field,value", [("dataset_id", "other"), ("network_id", "other"),
    ("location_id", "B"), ("product_id", "p2"), ("product_state", "CHILLED"), ("lot_id", "L2"),
    ("quantity_unit", "KG"), ("period_start", "2026-06-01")])
def test_turnover_blocks_incomparable_period_or_scope(field, value):
    sales, inv = turnover_pair()
    inv[field] = value
    assert candidate.aligned_turnover(sales, inv)["status"] == "NOT_COMPARABLE"


def test_outbound_turnover_remains_proxy():
    sales, inv = turnover_pair()
    sales["signal_type"] = legacy.OBSERVED_OUTBOUND_FLOW
    result = candidate.aligned_turnover(sales, inv)
    assert result["turnover_basis"] == legacy.FLOW_BASED_TURNOVER_PROXY
    assert result["gate_status"] == legacy.PROXY_ONLY


SS = dict(daily_mean=20, daily_std=4, lead_time_days=9, cycle_service_level=0.975,
          quantity_unit="EA", demand_basis="OBSERVED_RETAIL_SALES", stationary_independent=True,
          constant_lead_time=True)


def test_safety_stock_hand_calculation_and_zero_variability():
    result = candidate.safety_stock(**SS)
    assert result["value"] == pytest.approx(1.959963984540054 * 4 * 3)
    assert result["reorder_point"] == pytest.approx(180 + result["value"])
    assert result["fill_rate_guaranteed"] is False
    assert candidate.safety_stock(**{**SS, "daily_std": 0})["value"] == 0


@pytest.mark.parametrize("key,value,status", [("daily_std", None, "INPUT_REQUIRED"),
    ("lead_time_days", None, "INPUT_REQUIRED"), ("lead_time_days", 0, "INVALID_INPUT"),
    ("cycle_service_level", 1, "INVALID_INPUT"), ("quantity_unit", "UNDECLARED", "INPUT_REQUIRED"),
    ("stationary_independent", False, "NOT_APPLICABLE"), ("constant_lead_time", False, "NOT_APPLICABLE")])
def test_safety_stock_has_no_fallback_defaults(key, value, status):
    assert candidate.safety_stock(**{**SS, key: value})["status"] == status


def test_safety_stock_unit_scaling():
    ea = candidate.safety_stock(**SS)
    box = candidate.safety_stock(**{**SS, "daily_mean": 2, "daily_std": .4, "quantity_unit": "BOX"})
    assert box["value"] * 10 == pytest.approx(ea["value"])
    assert box["reorder_point"] * 10 == pytest.approx(ea["reorder_point"])


EOQ = dict(annual_demand=10000, order_cost=500, annual_holding_cost_per_unit=100,
           quantity_unit="EA", currency="KRW", stationary_demand=True, perishability=False)


def test_eoq_formula_and_zero_demand():
    result = candidate.eoq(**EOQ)
    assert result["value"] == pytest.approx(math.sqrt(100000))
    assert result["annual_order_holding_cost"] == pytest.approx(math.sqrt(100000) * 100)
    zero = candidate.eoq(**{**EOQ, "annual_demand": 0})
    assert zero["value"] == zero["annual_order_holding_cost"] == 0


@pytest.mark.parametrize("changes,status", [({"annual_holding_cost_per_unit": None}, "INPUT_REQUIRED"),
    ({"annual_holding_cost_per_unit": 0}, "INVALID_INPUT"), ({"order_cost": None}, "INPUT_REQUIRED"),
    ({"perishability": True}, "NOT_APPLICABLE"), ({"stationary_demand": False}, "NOT_APPLICABLE"),
    ({"perishability": None}, "NOT_APPLICABLE"), ({"currency": "UNKNOWN"}, "INPUT_REQUIRED"),
    ({"capacity_qty": 10}, "REFERENCE_ONLY"), ({"minimum_lot": 20}, "REFERENCE_ONLY")])
def test_eoq_missing_or_inapplicable_never_production_order(changes, status):
    result = candidate.eoq(**{**EOQ, **changes})
    assert result["status"] == status and result["production_approval"] is False


def test_eoq_units_and_total_cost_are_consistent():
    ea = candidate.eoq(**EOQ)
    box = candidate.eoq(**{**EOQ, "annual_demand": 1000, "annual_holding_cost_per_unit": 1000, "quantity_unit": "BOX"})
    assert box["value"] * 10 == pytest.approx(ea["value"])
    assert box["annual_order_holding_cost"] == pytest.approx(ea["annual_order_holding_cost"])


def test_expiry_exposure_is_not_observed_waste_or_probability():
    result = candidate.expiry_exposure(on_hand=100, days_to_expiry=3, daily_demand=20, quantity_unit="EA",
        disposal_history=[{"quantity": 4}], expiry_dates_observed=True)
    assert result["value"] == 40 and result["status"] == "SCENARIO_ESTIMATE"
    assert result["is_observed_waste"] is False and result["probability"]["value"] is None
    assert "NO_VALIDATED_DISPOSAL_MODEL" in result["probability"]["reason_codes"]


def test_candidate_functions_do_not_mutate_inputs():
    rows = [row()]
    original = copy.deepcopy(rows)
    candidate.abc_explicit_basis(rows, basis="SALES_REVENUE")
    sales, inv = turnover_pair()
    before = copy.deepcopy((sales, inv))
    candidate.aligned_turnover(sales, inv)
    assert rows == original and (sales, inv) == before
