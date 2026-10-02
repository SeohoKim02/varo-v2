"""Seller Loss production input layer: real data + explicit seller inputs (services.seller_loss_inputs)."""
import copy
import io
import math
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from services import analysis_pipeline
from services.analysis_pipeline import build_v2_state, run_analysis_pipeline
from services.data_loader import OPTIONAL_SHEETS, load_excel_data, normalize_loaded_data
from services.seller_decision_validation import controlled_scenarios
from services.seller_loss_engine import (
    DISCOUNT_SALE, NORMAL_SALE, STATUS_FULL, STATUS_PARTIAL, STATUS_UNAVAILABLE, TRANSFER, WORKBOOK_DATASET,
    InputField, SellerDecisionInput, build_seller_loss_analysis, evaluate_seller_decision, known, pipeline_decision_input,
)
from services import seller_loss_inputs as sli
from services.seller_loss_inputs import (
    ACTUAL_OPERATION, INSUFFICIENT, NOT_ROBUST, REAL_ONLY, REAL_PLUS_USER_INPUT, RECOMMENDABLE, SCENARIO,
    SCENARIO_EVIDENCE, SCENARIO_ONLY, UNAVAILABLE, USER_INPUT_ONLY, evaluate_with_seller_inputs, merge_seller_inputs,
    parse_seller_loss_inputs,
)
from tests.fixtures import sample_workbook, workbook_excel_bytes
from tests.test_real_transport_enrichment import _write_real_data

REPO = Path(__file__).resolve().parents[1]
NETWORK_SAMPLE = REPO / "data" / "Varo_V2_네트워크_샘플.xlsx"
TEMPLATE = REPO / "samples" / "seller_loss_inputs_TEMPLATE_SAMPLE.csv"
W = WORKBOOK_DATASET


def q(value, provenance="DIRECT_REAL", unit=None):
    return known(value, provenance, "production row", unit=unit, dataset=W)


def m(value, provenance="DERIVED_REAL", currency="KRW"):
    return known(value, provenance, "production row", currency=currency, dataset=W)


def base_input(**overrides):
    """A production row: real stock / demand / route, no business values (they come from the seller)."""
    fields = dict(
        decision_id="R1", product_id="P1", source_store_id="S1", target_store_id="S2", legacy_action="보류",
        decision_qty=q(20, "DERIVED_FROM_USER_INPUT"), source_current_stock=q(50), source_daily_demand=q(3, "DERIVED_REAL"),
        target_current_stock=q(5), target_daily_demand=q(4, "DERIVED_REAL"),
        transfer_cost=m(1800), transfer_cost_basis="FIXED_PER_TRIP", transit_time_days=known(0.1, "DERIVED_REAL", "osrm"),
    )
    fields.update(overrides)
    return SellerDecisionInput(**fields)


# Seller rows that complete controlled scenario A (S=50, Q=20, r=3, H=10, p=1000, h=5, c=200, s=0, d=0.3, u=0.5).
SCENARIO_A_ROWS = [
    {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "normal_price": 1000, "remaining_shelf_life_days": 10},
    {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S2", "normal_price": 1000},
    {"scope": "GLOBAL", "holding_cost_per_unit_day": 5, "disposal_cost_per_unit": 200, "salvage_value_per_unit": 0},
    {"scope": "PRODUCT", "product_id": "P1", "discount_rate": 0.3, "promotion_uplift": 0.5},
]


def table(rows, currency="KRW"):
    return parse_seller_loss_inputs(pd.DataFrame(rows), upload_currency=currency, upload_currency_basis="test upload contract")


def run(rows=SCENARIO_A_ROWS, base=None, **kwargs):
    return evaluate_with_seller_inputs(base or base_input(), table(rows), **kwargs)


def _load(path=NETWORK_SAMPLE):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_excel_data(path)


# ---------------------------------------------------------------- merge feeds the unchanged engine formula


def test_real_plus_user_input_reproduces_the_engine_formula_exactly():
    merged_result = run()
    scenario_a = evaluate_seller_decision(controlled_scenarios()[0]["input"])
    assert merged_result["recommended_strategy"] == scenario_a["recommended_strategy"] == TRANSFER
    for key in ("expected_loss_transfer", "expected_loss_normal_sale", "expected_loss_discount_sale",
                "loss_difference_vs_second_best", "breakdown", "expected_sold_qty", "expected_unsold_qty"):
        assert merged_result[key] == scenario_a[key]
    assert merged_result["expected_loss_transfer"] == 2175
    assert merged_result["evidence_level"] == REAL_PLUS_USER_INPUT
    assert merged_result["recommendation_readiness"] == RECOMMENDABLE
    assert merged_result["production_promotion_candidate"] is True and merged_result["production_action_applied"] is False


def test_controlled_scenarios_still_pass_unchanged():
    for scenario in controlled_scenarios():
        result = evaluate_seller_decision(scenario["input"])
        for key, value in scenario["expect"].items():
            if key == "reason":
                assert value in result["reason_codes"] or any(value in r for r in result["unavailable_strategies"].values())
            elif key == "unavailable":
                assert value in result["unavailable_strategies"]
            else:
                assert result[key] == value, (scenario["id"], key)


# ---------------------------------------------------------------- scopes, precedence, conflicts


def test_exact_product_store_key_applies_only_to_that_product_and_store():
    rows = [{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "normal_price": 1000}]
    result = run(rows)
    sources = result["input_sources"]
    assert sources["source_normal_price"]["value"] == 1000 and sources["source_normal_price"]["scope"] == "PRODUCT_STORE"
    assert sources["target_normal_price"]["value"] is None          # S2 is not S1
    other = run(rows, base=base_input(product_id="P9"))
    assert other["input_sources"]["source_normal_price"]["value"] is None   # no nearest/partial key match


def test_product_scope_applies_at_every_store_of_the_product():
    result = run([{"scope": "PRODUCT", "product_id": "P1", "normal_price": 900}])
    assert result["input_provenance"]["source_normal_price"]["value"] == 900
    assert result["input_provenance"]["target_normal_price"]["value"] == 900
    assert run([{"scope": "PRODUCT", "product_id": "P2", "normal_price": 900}])["input_provenance"]["source_normal_price"]["value"] is None


def test_store_scope_maps_one_store_value_to_the_role_the_store_plays():
    rows = [{"scope": "STORE", "store_id": "S2", "holding_cost_per_unit_day": 7}]
    result = run(rows)
    assert result["input_provenance"]["target_holding_cost_per_unit_day"]["value"] == 7
    assert result["input_provenance"]["source_holding_cost_per_unit_day"]["value"] is None
    reversed_route = run(rows, base=base_input(source_store_id="S2", target_store_id="S1"))
    assert reversed_route["input_provenance"]["source_holding_cost_per_unit_day"]["value"] == 7


def test_global_scope_applies_everywhere():
    result = run([{"scope": "GLOBAL", "disposal_cost_per_unit": 250}])
    assert result["input_provenance"]["source_disposal_cost_per_unit"]["value"] == 250
    assert result["input_provenance"]["target_disposal_cost_per_unit"]["value"] == 250


def test_more_specific_scope_wins_and_the_shadowed_row_is_audited():
    rows = [{"scope": "GLOBAL", "disposal_cost_per_unit": 300},
            {"scope": "PRODUCT", "product_id": "P1", "disposal_cost_per_unit": 500},
            {"scope": "STORE", "store_id": "S1", "disposal_cost_per_unit": 400}]
    result = run(rows)
    assert result["input_provenance"]["source_disposal_cost_per_unit"]["value"] == 500   # PRODUCT > STORE > GLOBAL
    assert result["input_provenance"]["target_disposal_cost_per_unit"]["value"] == 500
    product_store = run([*rows, {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "disposal_cost_per_unit": 650}])
    assert product_store["input_provenance"]["source_disposal_cost_per_unit"]["value"] == 650
    outcomes = {(a["field"], a["row"]): a["outcome"] for a in product_store["seller_input_audit"]}
    assert outcomes[("source_disposal_cost_per_unit", 4)] == "APPLIED"
    assert outcomes[("source_disposal_cost_per_unit", 1)] == "SHADOWED_BY_MORE_SPECIFIC_SCOPE"
    # A route_id row is more specific than the source/target pair row.
    route_rows = [{"scope": "ROUTE", "source_store_id": "S1", "target_store_id": "S2", "transit_time_days": 0.5},
                  {"scope": "ROUTE", "route_id": "R1", "transit_time_days": 0.25}]
    routed = run(route_rows, base=base_input(transit_time_days=InputField(source="absent")))
    assert routed["input_provenance"]["transit_time_days"]["value"] == 0.25


def test_same_scope_conflict_blocks_the_field_without_picking_a_row():
    rows = [*SCENARIO_A_ROWS,
            {"scope": "GLOBAL", "disposal_cost_per_unit": 300}]   # second GLOBAL disposal value != 200
    result = run(rows)
    assert "source_disposal_cost_per_unit" in result["conflicting_fields"]
    assert result["input_provenance"]["source_disposal_cost_per_unit"]["value"] is None
    assert "source_disposal_cost_per_unit" in result["unknown_inputs"] or result["comparison_status"] != STATUS_FULL
    conflict = next(c for c in result["input_conflicts"] if c["field"] == "source_disposal_cost_per_unit")
    assert conflict["kind"] == "SELLER_INPUT_SAME_SCOPE" and sorted(conflict["seller_values"]) == [200, 300]
    assert result["recommendation_readiness"] == NOT_ROBUST
    assert any(r.startswith("INPUT_CONFLICT:") for r in result["readiness_reasons"])
    # Identical duplicate values are not a conflict.
    duplicate = run([*SCENARIO_A_ROWS, {"scope": "GLOBAL", "disposal_cost_per_unit": 200}])
    assert duplicate["conflicting_fields"] == [] and duplicate["input_provenance"]["source_disposal_cost_per_unit"]["value"] == 200


def test_quantity_unit_conflict_at_the_same_scope_is_recorded():
    rows = [*SCENARIO_A_ROWS, {"scope": "GLOBAL", "quantity_unit": "EA"}, {"scope": "GLOBAL", "quantity_unit": "BOX"}]
    result = run(rows)
    assert "quantity_unit" in result["conflicting_fields"] and result["quantity_unit"] is None


# ---------------------------------------------------------------- provenance priority


def test_direct_real_value_beats_a_different_user_input_and_the_conflict_is_recorded():
    base = base_input(source_normal_price=m(1000, "DIRECT_REAL"))
    result = run([{**SCENARIO_A_ROWS[0], "normal_price": 1200}, *SCENARIO_A_ROWS[1:]], base=base)
    assert result["input_provenance"]["source_normal_price"]["value"] == 1000
    assert result["input_provenance"]["source_normal_price"]["provenance"] == "DIRECT_REAL"
    conflict = next(c for c in result["input_conflicts"] if c["field"] == "source_normal_price")
    assert conflict["kind"] == "REAL_VS_SELLER_INPUT" and conflict["kept"] == 1000
    assert "source_normal_price" in result["real_input_fields"]
    same = run(SCENARIO_A_ROWS, base=base)
    audit = [a for a in same["seller_input_audit"] if a["field"] == "source_normal_price"]
    assert audit and audit[0]["outcome"] == "MATCHES_REAL" and same["conflicting_fields"] == []


def test_user_input_replaces_a_proxy_and_records_what_it_replaced():
    base = base_input(source_daily_demand=q(3, "PROXY"))
    blocked = evaluate_seller_decision(base)
    assert "PROXY_REJECTED:source_daily_demand" in blocked["reason_codes"]
    result = run([*SCENARIO_A_ROWS, {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "daily_demand": 3}], base=base)
    assert result["input_provenance"]["source_daily_demand"]["provenance"] == "USER_INPUT"
    applied = next(a for a in result["seller_input_audit"] if a["field"] == "source_daily_demand")
    assert applied["outcome"] == "APPLIED_OVER_PROXY"
    assert result["input_sources"]["source_daily_demand"]["origin"] == "SELLER_LOSS_INPUTS"


def test_uploaded_production_value_is_kept_over_a_seller_input():
    base = base_input(source_normal_price=m(950, "USER_INPUT"))
    result = run(SCENARIO_A_ROWS, base=base)
    assert result["input_provenance"]["source_normal_price"]["value"] == 950
    assert next(c for c in result["input_conflicts"] if c["field"] == "source_normal_price")["kind"] == "DATA_VS_SELLER_INPUT"


def test_missing_value_stays_missing_and_is_never_zero():
    result = run([{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "normal_price": 1000,
                   "disposal_cost_per_unit": None, "remaining_shelf_life_days": ""}])
    provenance = result["input_provenance"]
    for name in ("source_disposal_cost_per_unit", "remaining_shelf_life_days", "source_holding_cost_per_unit_day",
                 "salvage_value_per_unit", "discount_rate", "promotion_uplift", "target_normal_price"):
        assert provenance[name]["value"] is None and provenance[name]["provenance"] == "MISSING"
    assert result["comparison_status"] == STATUS_UNAVAILABLE and "SHELF_LIFE_MISSING" in result["reason_codes"]
    assert "remaining_shelf_life_days" in result["missing_required_fields"]
    assert result["evidence_level"] == INSUFFICIENT and result["recommendation_readiness"] == UNAVAILABLE


def test_declared_row_provenance_is_honoured_and_proxy_caps_are_rejected():
    recommendation = {"route_id": "R1", "product_id": "P1", "source_id": "S1", "target_id": "S2", "recommended_qty": 10,
                      "move_cost": 5000, "travel_time_min": 60}
    inventory = {("S1", "P1"): {"store_id": "S1", "product_id": "P1", "stock_qty": 100, "stock_qty_provenance": "actual",
                                "sales_qty": 4, "sales_qty_semantics": "demand_proxy_not_retail_sales"},
                 ("S2", "P1"): {"store_id": "S2", "product_id": "P1", "stock_qty": 8, "stock_qty_provenance": "actual"}}
    forecast = {("S1", "P1"): {"demand_forecast_daily": 4, "sales_qty_semantics": "demand_proxy_not_retail_sales"},
                ("S2", "P1"): {"demand_forecast_daily": 2, "sales_qty_provenance": "actual"}}
    transition = {"metadata": {"movable_stock": 90, "movable_stock_basis": "max(stock_qty-재고상태 수요, 0)",
                               "target_shortage_limit": 6, "target_shortage_basis": "max(sales_qty×7-stock_qty, 0)"}}
    inp = pipeline_decision_input(recommendation, inventory_rows=inventory, forecast_rows=forecast, product_rows={},
                                  config={}, transition=transition)
    assert inp.source_current_stock.provenance == "DIRECT_REAL" and "declared provenance 'actual'" in inp.source_current_stock.note
    assert inp.target_current_stock.provenance == "DIRECT_REAL"
    assert inp.source_daily_demand.provenance == "PROXY"
    assert inp.target_daily_demand.provenance == "DERIVED_REAL"
    assert inp.source_surplus_cap.provenance == "PROXY"
    explicit = dict(transition, metadata={**transition["metadata"], "target_shortage_basis": "shortage_qty"})
    assert pipeline_decision_input(recommendation, inventory_rows=inventory, forecast_rows=forecast, product_rows={},
                                   config={}, transition=explicit).target_need_cap.provenance == "DERIVED_FROM_USER_INPUT"


# ---------------------------------------------------------------- modes and evidence


def test_scenario_rows_are_ignored_in_actual_operation_and_applied_only_in_scenario_mode():
    rows = [*SCENARIO_A_ROWS, {"scope": "ROUTE", "source_store_id": "S1", "target_store_id": "S2", "input_type": "SCENARIO",
                               "transfer_cost": 90000, "currency": "KRW"}]
    actual = run(rows)
    assert actual["decision_mode"] == ACTUAL_OPERATION and actual["ignored_scenario_cells"] == 1
    assert actual["input_provenance"]["transfer_cost"]["value"] == 1800
    assert actual["input_provenance"]["transfer_cost"]["provenance"] == "DERIVED_REAL"
    scenario = run(rows, decision_mode=SCENARIO)
    assert scenario["input_provenance"]["transfer_cost"]["provenance"] == "SCENARIO_INPUT"
    assert scenario["input_provenance"]["transfer_cost"]["value"] == 90000
    override = next(a for a in scenario["seller_input_audit"] if a["field"] == "transfer_cost")
    assert override["outcome"] == "SCENARIO_OVERRIDE" and scenario["input_sources"]["transfer_cost"]["origin"] == "SCENARIO"
    assert scenario["evidence_level"] == SCENARIO_EVIDENCE and scenario["recommendation_readiness"] == SCENARIO_ONLY
    assert scenario["recommended_strategy"] != TRANSFER and actual["recommended_strategy"] == TRANSFER
    assert "SCENARIO_INPUT:transfer_cost" in scenario["reason_codes"]


def test_evidence_levels():
    assert run()["evidence_level"] == REAL_PLUS_USER_INPUT
    all_real = base_input(
        source_normal_price=m(1000, "DIRECT_REAL"), target_normal_price=m(1000, "DIRECT_REAL"),
        remaining_shelf_life_days=known(10, "DERIVED_REAL", "expiry"), source_holding_cost_per_unit_day=m(5, "DIRECT_REAL"),
        target_holding_cost_per_unit_day=m(5, "DIRECT_REAL"), source_disposal_cost_per_unit=m(200, "DIRECT_REAL"),
        target_disposal_cost_per_unit=m(200, "DIRECT_REAL"), salvage_value_per_unit=m(0, "DIRECT_REAL"))
    real_only = evaluate_with_seller_inputs(all_real, None)
    assert real_only["evidence_level"] == REAL_ONLY and real_only["comparison_status"] == STATUS_PARTIAL
    with_config = evaluate_with_seller_inputs(controlled_scenarios()[1]["input"], None)
    assert with_config["evidence_level"] == SCENARIO_EVIDENCE   # its discount/uplift are CONFIG settings
    user_only = evaluate_with_seller_inputs(replace(controlled_scenarios()[4]["input"], discount_rate=InputField()), None)
    assert user_only["evidence_level"] == USER_INPUT_ONLY
    assert run([])["evidence_level"] == INSUFFICIENT
    assert run(decision_mode=SCENARIO)["evidence_level"] == SCENARIO_EVIDENCE


def test_recommendation_readiness_classes():
    assert run()["recommendation_readiness"] == RECOMMENDABLE
    unknown_holding = [r for r in SCENARIO_A_ROWS if r["scope"] != "GLOBAL"] + [
        {"scope": "GLOBAL", "disposal_cost_per_unit": 200, "salvage_value_per_unit": 0}]
    not_robust = run(unknown_holding)   # holding unknown at both stores: scenario H depends on it
    assert not_robust["recommendation_status"] == "NOT_ROBUST_TO_UNKNOWN_INPUTS"
    assert not_robust["recommendation_readiness"] == NOT_ROBUST and not not_robust["production_promotion_candidate"]
    assert run(decision_mode=SCENARIO)["recommendation_readiness"] == SCENARIO_ONLY
    assert run([])["recommendation_readiness"] == UNAVAILABLE


def test_strategy_completeness_and_missing_fields_per_strategy():
    no_uplift = [r if r["scope"] != "PRODUCT" else {"scope": "PRODUCT", "product_id": "P1", "discount_rate": 0.3}
                 for r in SCENARIO_A_ROWS]
    result = run(no_uplift)
    assert result["strategy_readiness"] == {TRANSFER: "READY", NORMAL_SALE: "READY", DISCOUNT_SALE: "MISSING_PROMOTION_UPLIFT"}
    assert result["data_completeness_status"] == "PARTIAL" and result["comparison_status"] == STATUS_PARTIAL
    assert result["missing_required_fields"] == ["promotion_uplift"]
    assert result["missing_for_excluded_strategies"] == {DISCOUNT_SALE: ["promotion_uplift"]}
    assert run()["data_completeness_status"] == "FULL"


# ---------------------------------------------------------------- validation of seller values


@pytest.mark.parametrize("raw, code", [("1,000", "NOT_A_NUMBER"), ("abc", "NOT_A_NUMBER"), ("20%", "NOT_A_NUMBER"),
                                       ("nan", "NOT_FINITE"), (math.inf, "NOT_FINITE"), (True, "NOT_A_NUMBER")])
def test_invalid_numeric_values_are_rejected_not_zeroed(raw, code):
    parsed = table([{"scope": "GLOBAL", "disposal_cost_per_unit": raw}])
    assert parsed.cells == [] and parsed.validation["status"] == "VALID_WITH_ERRORS"
    assert parsed.validation["errors"][0]["code"] == code and parsed.validation["rejected_cells"] == 1
    assert parse_seller_loss_inputs(pd.DataFrame([{"scope": "GLOBAL", "disposal_cost_per_unit": "250"}]),
                                    upload_currency="KRW", upload_currency_basis="t").cells[0].value == 250.0


@pytest.mark.parametrize("column", ["normal_price", "unit_cost", "transfer_cost", "holding_cost_per_unit_day",
                                    "disposal_cost_per_unit", "salvage_value_per_unit", "remaining_shelf_life_days",
                                    "transit_time_days", "promotion_uplift", "daily_demand"])
def test_negative_values_are_rejected(column):
    scope = "ROUTE" if column in ("transfer_cost", "transit_time_days") else "GLOBAL"
    row = {"scope": scope, column: -1}
    if scope == "ROUTE":
        row.update(source_store_id="S1", target_store_id="S2")
    parsed = table([row])
    assert parsed.cells == [] and parsed.validation["errors"][0]["code"] == "NEGATIVE_VALUE"


@pytest.mark.parametrize("rate, ok", [(0.0, True), (0.2, True), (0.999, True), (1.0, False), (1.5, False), (-0.1, False)])
def test_discount_rate_must_be_a_fraction_below_one(rate, ok):
    parsed = table([{"scope": "GLOBAL", "discount_rate": rate}])
    assert bool(parsed.cells) is ok
    if not ok:
        assert parsed.validation["errors"][0]["code"] == "DISCOUNT_RATE_OUT_OF_RANGE"
    if rate == 0.0:   # accepted as input, but a 0% markdown is not a discount strategy for the engine
        result = run([*SCENARIO_A_ROWS[:3], {"scope": "PRODUCT", "product_id": "P1", "discount_rate": 0.0, "promotion_uplift": 0.5}])
        assert "INVALID_DISCOUNT_RATE" in result["unavailable_strategies"][DISCOUNT_SALE]


def test_invalid_sheet_and_rows_return_validation_errors_instead_of_crashing():
    assert table([{"normal_price": 1000}]).validation["status"] == "INVALID_SHEET"            # no scope column
    assert parse_seller_loss_inputs("not a table", upload_currency="KRW", upload_currency_basis="t").validation["status"] == "INVALID_SHEET"
    parsed = table([{"scope": "PLANET", "normal_price": 1},
                    {"scope": "PRODUCT", "normal_price": 1},                                     # missing product_id
                    {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "transfer_cost": 5},
                    {"scope": "GLOBAL", "currency": "원", "normal_price": 1},
                    {"scope": "GLOBAL", "input_type": "MAYBE", "normal_price": 1},
                    {"scope": "GLOBAL", "price_unit": "KRW/EA", "quantity_unit": "KG", "normal_price": 1},
                    {"scope": "ROUTE", "source_store_id": "S1", "target_store_id": "S2", "transfer_cost": 1, "transfer_cost_per_unit": 2},
                    {"scope": "GLOBAL", "source_holding_cost_per_unit_day": 3}])
    codes = [e["code"] for e in parsed.validation["errors"]]
    assert codes == ["INVALID_SCOPE", "SCOPE_KEYS_MISMATCH", "FIELD_NOT_ALLOWED_IN_SCOPE", "INVALID_CURRENCY",
                     "INVALID_INPUT_TYPE", "PRICE_UNIT_CONFLICT", "AMBIGUOUS_TRANSFER_COST_BASIS", "AMBIGUOUS_TRANSFER_COST_BASIS"]
    assert parsed.cells == [] and "source_holding_cost_per_unit_day" in parsed.validation["ignored_columns"]
    data = sample_workbook()
    data["seller_loss_inputs"] = pd.DataFrame([{"normal_price": "x"}])
    state = build_v2_state(data, detail_level="core")
    analysis = state["pipeline_result"]["seller_loss_analysis"]
    assert analysis["status"] == "parallel_only" and analysis["seller_loss_inputs"]["status"] == "INVALID_SHEET"
    assert state["recommendations"]


def test_invalid_decision_mode_falls_back_to_actual_operation_with_an_error():
    mode, info = sli.resolve_decision_mode("WHAT_IF", {}, {})
    assert mode == ACTUAL_OPERATION and info["error"]["code"] == "INVALID_DECISION_MODE"
    assert sli.resolve_decision_mode(None, {}, {"seller_loss_decision_mode": "scenario"})[0] == SCENARIO


# ---------------------------------------------------------------- currency and units


def test_currency_mismatch_is_never_converted():
    usd_everything = [{**row, "currency": "USD"} for row in SCENARIO_A_ROWS]
    result = run(usd_everything)                    # real transfer cost is KRW: only TRANSFER is affected
    assert result["comparison_status"] == STATUS_PARTIAL and TRANSFER in result["unavailable_strategies"]
    assert "CURRENCY_MISMATCH" in result["unavailable_strategies"][TRANSFER]
    assert result["input_provenance"]["source_normal_price"]["currency"] == "USD"
    assert result["input_provenance"]["transfer_cost"]["currency"] == "KRW" and result["currency"] is None   # not converted
    mixed = run([*SCENARIO_A_ROWS[:2], {**SCENARIO_A_ROWS[2], "currency": "USD"}, SCENARIO_A_ROWS[3]])
    assert mixed["comparison_status"] == STATUS_UNAVAILABLE and "CURRENCY_MISMATCH" in mixed["reason_codes"]
    blank = table([{"scope": "GLOBAL", "disposal_cost_per_unit": 1}], currency="KRW").cells[0]
    assert blank.currency == "KRW" and blank.currency_basis == "test upload contract"


def test_unit_rules_for_seller_values():
    ea = base_input(quantity_unit="EA")
    declared = [{**SCENARIO_A_ROWS[0], "price_unit": "KRW/EA"}, *SCENARIO_A_ROWS[1:]]
    assert evaluate_with_seller_inputs(ea, table(declared))["comparison_status"] == STATUS_FULL
    kg = [{**SCENARIO_A_ROWS[0], "price_unit": "KRW/KG"}, *SCENARIO_A_ROWS[1:]]
    mismatch = evaluate_with_seller_inputs(ea, table(kg))
    assert mismatch["comparison_status"] == STATUS_UNAVAILABLE and "UNIT_MISMATCH" in mismatch["reason_codes"]
    unknown_unit = run(declared)                    # inventory unit undeclared, price declared per EA
    assert unknown_unit["comparison_status"] == STATUS_UNAVAILABLE
    assert "SELLER_UNIT_UNVERIFIABLE:source_normal_price" in unknown_unit["reason_codes"]
    explicit = run([{"scope": "GLOBAL", "quantity_unit": "EA"}, *declared])   # seller declares the inventory unit too
    assert explicit["comparison_status"] == STATUS_FULL and explicit["quantity_unit"] == "EA"
    assert explicit["quantity_unit_source"].startswith("seller_loss_inputs[row 1]")
    per_unit_kg = run([{"scope": "GLOBAL", "quantity_unit": "EA"}, *SCENARIO_A_ROWS,
                       {"scope": "ROUTE", "route_id": "R1", "transfer_cost_per_unit": 50, "price_unit": "KRW/KG"}],
                      base=base_input(transfer_cost=InputField(source="absent")))
    assert "UNIT_MISMATCH:transfer_cost" in per_unit_kg["unavailable_strategies"][TRANSFER]


# ---------------------------------------------------------------- strategy inputs


def test_target_demand_missing_and_filled_by_the_seller():
    base = base_input(target_daily_demand=InputField(source="no forecast for the target"))
    missing = run(base=base)
    assert missing["strategy_readiness"][TRANSFER] == "MISSING_TARGET_DEMAND"
    assert "target_daily_demand" in missing["missing_for_excluded_strategies"][TRANSFER]
    filled = run([*SCENARIO_A_ROWS, {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S2", "daily_demand": 4}], base=base)
    assert filled["input_provenance"]["target_daily_demand"]["provenance"] == "USER_INPUT"
    assert filled["expected_loss_transfer"] == run()["expected_loss_transfer"]


def test_source_demand_missing_blocks_everything_until_the_seller_states_it():
    base = base_input(source_daily_demand=InputField(source="no forecast"))
    missing = run(base=base)
    assert missing["comparison_status"] == STATUS_UNAVAILABLE and "SOURCE_DEMAND_MISSING" in missing["reason_codes"]
    filled = run([*SCENARIO_A_ROWS, {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "daily_demand": 3}], base=base)
    assert filled["comparison_status"] == STATUS_FULL and filled["input_provenance"]["source_daily_demand"]["provenance"] == "USER_INPUT"


def test_promotion_uplift_missing_excludes_discount_but_the_markdown_price_is_shown():
    rows = [*SCENARIO_A_ROWS[:3], {"scope": "PRODUCT", "product_id": "P1", "discount_rate": 0.3}]
    result = run(rows)
    assert DISCOUNT_SALE in result["unavailable_strategies"] and "UPLIFT_MISSING" in result["unavailable_strategies"][DISCOUNT_SALE]
    preview = result["discount_price_preview"]
    assert preview["discounted_unit_price"] == 700 and preview["markdown_per_unit"] == 300
    assert preview["promotion_uplift"] is None and preview["promotion_uplift_status"] == "MISSING"


def test_config_promotion_placeholders_are_scenario_only():
    data = _load()
    actual = build_v2_state(copy.deepcopy(data), detail_level="core")["pipeline_result"]["seller_loss_analysis"]
    for decision in actual["decisions"]:
        for name in ("discount_rate", "promotion_uplift"):
            assert decision["input_provenance"][name]["value"] is None
            assert "not used in ACTUAL_OPERATION" in decision["input_provenance"][name]["source"]
    scenario_data = copy.deepcopy(data)
    scenario_data["seller_loss_decision_mode"] = SCENARIO
    scenario = build_v2_state(scenario_data, detail_level="core")["pipeline_result"]["seller_loss_analysis"]
    assert scenario["decision_mode"] == SCENARIO
    for decision in scenario["decisions"]:
        assert decision["input_provenance"]["promotion_uplift"]["provenance"] == "CONFIG"
        assert decision["recommendation_readiness"] in (SCENARIO_ONLY, UNAVAILABLE)


def test_actual_transfer_cost_is_preserved_and_a_seller_cost_fills_only_a_gap():
    rows = [*SCENARIO_A_ROWS, {"scope": "ROUTE", "source_store_id": "S1", "target_store_id": "S2", "transfer_cost": 999}]
    kept = run(rows)
    assert kept["input_provenance"]["transfer_cost"]["value"] == 1800
    assert kept["input_provenance"]["transfer_cost"]["provenance"] == "DERIVED_REAL"
    assert next(c for c in kept["input_conflicts"] if c["field"] == "transfer_cost")["kind"] == "REAL_VS_SELLER_INPUT"
    gap = run(rows, base=base_input(transfer_cost=InputField(source="no route cost")))
    assert gap["input_provenance"]["transfer_cost"]["value"] == 999
    assert gap["strategies"][TRANSFER]["components"]["transfer_cost"] == 999      # FIXED_PER_TRIP
    per_unit = run([*SCENARIO_A_ROWS, {"scope": "ROUTE", "route_id": "R1", "transfer_cost_per_unit": 50}],
                   base=base_input(transfer_cost=InputField(source="no route cost")))
    moved = per_unit["strategies"][TRANSFER]["quantities"]["moved_qty"]
    assert per_unit["strategies"][TRANSFER]["components"]["transfer_cost"] == 50 * moved
    proxy = run(rows, base=base_input(transfer_cost=m(1800, "PROXY")))
    assert proxy["input_provenance"]["transfer_cost"]["provenance"] == "USER_INPUT"   # a proxy is not a real cost


# ---------------------------------------------------------------- audit, influence, determinism


def test_audit_trail_records_field_source_scope_and_effective_value():
    result = run()
    audit = {a["field"]: a for a in result["seller_input_audit"]}
    record = audit["source_normal_price"]
    assert record["column"] == "normal_price" and record["row"] == 1 and record["scope"] == "PRODUCT_STORE"
    assert record["keys"] == {"product_id": "P1", "store_id": "S1"} and record["input_value"] == 1000
    assert record["provenance"] == "USER_INPUT" and record["outcome"] == "APPLIED" and record["currency"] == "KRW"
    assert result["input_sources"]["source_normal_price"]["source"].startswith("seller_loss_inputs[row 1].normal_price")
    assert set(result["seller_input_fields"]) >= {"source_normal_price", "target_normal_price", "remaining_shelf_life_days",
                                                  "discount_rate", "promotion_uplift"}
    assert set(result["real_input_fields"]) >= {"source_current_stock", "transfer_cost", "transit_time_days"}
    influence = result["seller_input_influence"]
    assert influence["recommended_strategy_without_seller_inputs"] is None and influence["seller_inputs_changed_recommendation"]
    assert "source_normal_price" in influence["fields_changing_recommendation"]
    assert influence["per_field"]["promotion_uplift"]["changes_recommendation"] is False   # TRANSFER wins either way


def test_sensitivity_hook_varies_seller_inputs_without_constants():
    merged, _ = merge_seller_inputs(base_input(), table(SCENARIO_A_ROWS))
    assert set(sli.SENSITIVITY_FIELDS) <= set(merged.input_fields())
    for factor in (0.5, 1.5):
        varied = sli.apply_scenario_overrides(merged, {"source_disposal_cost_per_unit": 200 * factor})
        assert varied.source_disposal_cost_per_unit.provenance == "SCENARIO_INPUT"
        assert varied.source_disposal_cost_per_unit.currency == "KRW"
        assert evaluate_seller_decision(varied)["comparison_status"] == STATUS_FULL


def test_deterministic_result_independent_of_row_order():
    first = run(SCENARIO_A_ROWS)
    second = run(list(reversed(SCENARIO_A_ROWS)))
    strip = lambda r: {k: v for k, v in r.items() if k not in ("seller_input_audit", "input_sources", "input_provenance", "explanation_facts")}  # noqa: E731
    assert strip(first) == strip(second)
    assert run(SCENARIO_A_ROWS) == first


# ---------------------------------------------------------------- pipeline and workbook


def _seller_rows_for_sample():
    return pd.DataFrame([
        {"scope": "GLOBAL", "holding_cost_per_unit_day": 3, "disposal_cost_per_unit": 400, "salvage_value_per_unit": 0},
        {"scope": "PRODUCT", "product_id": "P001", "discount_rate": 0.2, "promotion_uplift": 0.6},
    ])


def test_pipeline_without_sheet_matches_the_engine_on_the_production_rows():
    data = sample_workbook()
    analysis = build_v2_state(copy.deepcopy(data), detail_level="core")["pipeline_result"]["seller_loss_analysis"]
    assert analysis["seller_loss_inputs"]["status"] == "ABSENT" and analysis["decision_mode"] == ACTUAL_OPERATION
    for decision in analysis["decisions"]:
        assert decision["seller_input_audit"] == [] and decision["seller_input_influence"] == {"seller_inputs_applied": False}
        assert all(source["origin"] in ("DATA", "NONE") for source in decision["input_sources"].values())


def test_workbook_sheet_is_optional_and_recognised_when_present():
    assert "seller_loss_inputs" in OPTIONAL_SHEETS
    plain = load_excel_data(workbook_excel_bytes(sample_workbook()))
    assert "seller_loss_inputs" not in plain
    with_sheet = load_excel_data(workbook_excel_bytes({**sample_workbook(), "seller_loss_inputs": _seller_rows_for_sample()}))
    assert "seller_loss_inputs" in with_sheet and len(with_sheet["seller_loss_inputs"]) == 2
    without = build_v2_state(copy.deepcopy(plain), detail_level="core")
    applied = build_v2_state(copy.deepcopy(with_sheet), detail_level="core")
    analysis = applied["pipeline_result"]["seller_loss_analysis"]
    assert analysis["seller_loss_inputs"]["status"] == "VALID" and analysis["seller_loss_inputs"]["accepted_cells"] == 5
    assert any(d["seller_input_fields"] for d in analysis["decisions"])
    # Legacy outputs are untouched by the seller inputs.
    assert applied["recommendations"] == without["recommendations"]
    for key in ("summary", "top5", "connected_algorithms"):
        assert applied["pipeline_result"][key] == without["pipeline_result"][key]
    for row, rec in zip(analysis["rows"], applied["recommendations"]):
        assert row["legacy_action"] == rec["varo_action"]
    assert all(d["legacy_action_changed"] is False for d in analysis["decisions"])


def test_existing_repository_workbooks_still_load_and_ignore_the_absent_sheet():
    for path in sorted((REPO / "samples").glob("*.xlsx")):
        data = _load(path)
        assert "seller_loss_inputs" not in data
        analysis = build_v2_state(data, detail_level="core")["pipeline_result"]["seller_loss_analysis"]
        assert analysis["seller_loss_inputs"]["status"] == "ABSENT" and analysis["legacy_action_replaced"] is False


def test_legacy_action_untouched_with_seller_inputs_in_both_modes():
    data = sample_workbook()
    baseline = build_v2_state(copy.deepcopy(data), detail_level="full")
    for mode in (ACTUAL_OPERATION, SCENARIO):
        enriched = {**copy.deepcopy(data), "seller_loss_inputs": _seller_rows_for_sample(), "seller_loss_decision_mode": mode}
        state = build_v2_state(enriched, detail_level="full")
        assert state["recommendations"] == baseline["recommendations"]
        assert state["pipeline_result"]["summary"] == baseline["pipeline_result"]["summary"]
        assert state["pipeline_result"]["seller_loss_analysis"]["legacy_action_replaced"] is False


def test_template_is_sample_only_and_valid():
    frame = pd.read_csv(TEMPLATE, encoding="utf-8-sig", dtype=str)
    assert list(frame.columns) == list(sli.TEMPLATE_COLUMNS)
    assert frame["note"].str.startswith("SAMPLE").all()
    assert all(str(value).startswith("SAMPLE_") for column in ("product_id", "store_id", "source_store_id", "target_store_id")
               for value in frame[column].dropna())
    parsed = parse_seller_loss_inputs(frame, upload_currency="KRW", upload_currency_basis="t")
    assert parsed.validation["status"] == "VALID" and parsed.validation["scenario_cells"] == 1
    assert parsed.validation["errors"] == [] and parsed.validation["warnings"] == []
    generated = pd.read_csv(io.StringIO(sli.template_frame().to_csv(index=False)))
    pd.testing.assert_frame_equal(pd.read_csv(TEMPLATE, encoding="utf-8-sig"), generated)


def test_input_contract_is_stable_and_lists_the_rules():
    first, second = sli.input_contract_document(), sli.input_contract_document()
    assert first == second and len(first["contract_signature"]) == 64
    assert first["scope_precedence"] == ["ROUTE", "PRODUCT_STORE", "PRODUCT", "STORE", "GLOBAL"]
    assert set(first["evidence_levels"]) == {REAL_ONLY, REAL_PLUS_USER_INPUT, USER_INPUT_ONLY, SCENARIO_EVIDENCE, INSUFFICIENT}
    assert set(first["readiness"]) == {RECOMMENDABLE, NOT_ROBUST, SCENARIO_ONLY, UNAVAILABLE}
    from services.seller_loss_engine import ACCEPTED_PROVENANCE, SCENARIO_PROVENANCE
    assert "SCENARIO_INPUT" in ACCEPTED_PROVENANCE and "SCENARIO_INPUT" in SCENARIO_PROVENANCE


# ---------------------------------------------------------------- real-data-shaped E2E through the production pipeline


def _suhyup_shaped_upload():
    """Same columns as C:/VARO_V2_REAL_DATA/16_VARO_E2E_20260731/processed (actual stock, outbound demand proxy)."""
    stores = pd.DataFrame([{"node_id": c, "node_name": f"{c} center", "node_type": "STORE", "network_mode": "DIRECT_NETWORK",
                            "data_provenance": "actual_suhyup_center"} for c in ("A", "B")])
    products = pd.DataFrame([{"product_id": "P_REAL", "product_name": "Test product"}])
    inventory = pd.DataFrame([
        {"store_id": "A", "product_id": "P_REAL", "stock_qty": 400, "sales_qty": 2, "snapshot_date": "2026-07-31",
         "stock_qty_provenance": "actual", "sales_qty_provenance": "derived_proxy_from_actual_outbound_qty",
         "sales_qty_semantics": "demand_proxy_not_retail_sales"},
        {"store_id": "B", "product_id": "P_REAL", "stock_qty": 30, "sales_qty": 9, "snapshot_date": "2026-07-31",
         "stock_qty_provenance": "actual", "sales_qty_provenance": "derived_proxy_from_actual_outbound_qty",
         "sales_qty_semantics": "demand_proxy_not_retail_sales"},
    ])
    routes = pd.DataFrame([{"source_id": "A", "target_id": "B", "distance_km": 153.415, "estimated_cost": 0.0, "travel_time_min": 124.02},
                           {"source_id": "B", "target_id": "A", "distance_km": 153.415, "estimated_cost": 0.0, "travel_time_min": 124.02}])
    recommendations = pd.DataFrame([{"route_id": "E2E-1", "product_id": "P_REAL", "source_id": "A", "target_id": "B",
                                     "dc_id": None, "route_type": "DIRECT", "recommended_qty": 100, "estimated_cost": 0.0,
                                     "distance_km": 153.415, "travel_time_min": 124.02}])
    return normalize_loaded_data({"stores": stores, "products": products, "inventory": inventory, "routes": routes,
                                  "recommendations": recommendations})


E2E_TEST_USER_INPUT = pd.DataFrame([
    {"scope": "GLOBAL", "quantity_unit": "BOX", "currency": "KRW", "holding_cost_per_unit_day": 20,
     "disposal_cost_per_unit": 500, "salvage_value_per_unit": 0, "note": "TEST USER INPUT"},
    {"scope": "PRODUCT", "product_id": "P_REAL", "normal_price": 30000, "price_unit": "KRW/BOX", "remaining_shelf_life_days": 21,
     "discount_rate": 0.3, "note": "TEST USER INPUT"},
    {"scope": "PRODUCT_STORE", "product_id": "P_REAL", "store_id": "A", "daily_demand": 5, "note": "TEST USER INPUT"},
    {"scope": "PRODUCT_STORE", "product_id": "P_REAL", "store_id": "B", "daily_demand": 60, "note": "TEST USER INPUT"},
    {"scope": "ROUTE", "source_store_id": "A", "target_store_id": "B", "transfer_cost": 1, "currency": "KRW",
     "note": "TEST USER INPUT that contradicts the real route cost"},
])


def test_real_data_plus_user_input_e2e_through_the_production_pipeline(tmp_path, monkeypatch):
    _write_real_data(tmp_path)
    monkeypatch.setenv("VARO_REAL_DATA_ROOT", str(tmp_path))
    baseline = run_analysis_pipeline(_suhyup_shaped_upload())
    plain = baseline.seller_loss_analysis["decisions"][0]
    assert plain["comparison_status"] == STATUS_UNAVAILABLE and plain["evidence_level"] == INSUFFICIENT
    assert {"PRICE_MISSING", "SHELF_LIFE_MISSING", "PROXY_REJECTED:source_daily_demand"} <= set(plain["reason_codes"])

    upload = {**_suhyup_shaped_upload(), "seller_loss_inputs": E2E_TEST_USER_INPUT.copy()}
    result = run_analysis_pipeline(upload)
    decision = result.seller_loss_analysis["decisions"][0]
    provenance = decision["input_provenance"]
    assert provenance["source_current_stock"]["value"] == 400 and provenance["source_current_stock"]["provenance"] == "DIRECT_REAL"
    assert provenance["target_current_stock"]["provenance"] == "DIRECT_REAL"
    assert provenance["transfer_cost"]["provenance"] == "DERIVED_REAL"                    # 1t official direct tariff
    assert provenance["transfer_cost"]["value"] == baseline.seller_loss_analysis["decisions"][0]["input_provenance"]["transfer_cost"]["value"]
    assert provenance["transfer_cost"]["value"] != 1                                     # the contradicting TEST value lost
    assert "transfer_cost" in decision["conflicting_fields"]
    assert "SELLER_INPUT_OVERRULED_BY_DATA:transfer_cost" in decision["readiness_reasons"]
    assert provenance["source_daily_demand"]["provenance"] == "USER_INPUT"               # replaced the outbound proxy
    assert provenance["source_surplus_cap"]["provenance"] == "PROXY" and provenance["target_need_cap"]["provenance"] == "PROXY"
    assert decision["quantity_unit"] == "BOX"
    assert decision["evidence_level"] == REAL_PLUS_USER_INPUT
    assert decision["comparison_status"] == STATUS_PARTIAL                              # no uplift evidence: discount excluded
    assert decision["strategy_readiness"][DISCOUNT_SALE] == "MISSING_PROMOTION_UPLIFT"
    assert decision["recommended_strategy"] in (TRANSFER, NORMAL_SALE)
    assert set(decision["real_input_fields"]) >= {"source_current_stock", "target_current_stock", "transfer_cost", "transit_time_days"}
    assert set(decision["seller_input_fields"]) >= {"source_normal_price", "remaining_shelf_life_days", "source_daily_demand",
                                                    "target_daily_demand", "source_disposal_cost_per_unit"}
    assert result.recommendations == baseline.recommendations                         # legacy outputs untouched
    assert decision["legacy_action_changed"] is False
