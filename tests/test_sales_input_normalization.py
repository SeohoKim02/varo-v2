"""T4-B: real evidence boundaries, not production algorithm replacements."""
import copy
import json

import pandas as pd
import pytest

from services import inventory_metric_integrity as imi
from services.data_loader import normalize_loaded_data
from services.legacy_adapters.data_adapter import prepare_legacy_data


def data(history=True, stock=True):
    row = dict(store_id="S", product_id="P", dataset_id="D", business_id="B", quantity_unit="item",
               snapshot_date="2026-07-30", stock_qty=100, unit_price=5, unit_cost=2)
    out = {"inventory": pd.DataFrame([row])}
    records = [dict(date=f"2026-07-{d:02}", location_id="S", product_id="P", dataset_id="D",
                    business_id="B", quantity_unit="item", signal_type=imi.OBSERVED_RETAIL_SALES,
                    quantity=2.0) for d in range(1, 31)]
    if history:
        out["daily_sales_history"] = pd.DataFrame(records)
    if stock:
        out["inventory_history"] = pd.DataFrame([{**r, "quantity": 100.0} for r in records])
    return out


def normalized(upload):
    return imi.normalize_sales_signals(upload, semantics_mode="VALIDATED")["rows"][0]


@pytest.mark.parametrize("scenario", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["AA", "AB", "AC", "AD", "AE", "AF"])
def test_controlled_normalization(scenario):
    upload = data()
    inv = upload["inventory"]
    if scenario in ("A", "B", "C", "R", "S", "T", "U", "V", "W", "X", "Y"):
        upload.pop("daily_sales_history")
    if scenario in ("A", "B"):
        inv["sales_qty"] = 3.0
        inv["sales_qty_semantics"] = "daily_retail_sales" if scenario == "A" else imi.DAILY_PROXY_SEMANTICS
        n = normalized(normalize_loaded_data(upload))
        assert n["sales_30d"] is None and n["sales_7d"] is None
        assert n["avg_daily_sales"] == (3.0 if scenario == "A" else None)
        assert n["signals"]["sales_qty"]["signal_type"] == (imi.OBSERVED_RETAIL_SALES if scenario == "A" else imi.OBSERVED_OUTBOUND_FLOW)
        assert n["signals"]["sales_30d"]["value"] is None
        assert "ALIAS_COPY_ACROSS_PERIODS" in n["signals"]["sales_30d"]["reason_codes"]
    elif scenario in ("C", "R", "S", "T", "U"):
        inv["sales_30d"] = 60.0
        inv["sales_30d_semantics"] = "monthly_retail_sales"
        inv["sales_30d_observation_start"] = "2026-07-01"
        inv["sales_30d_observation_end"] = "2026-07-30"
        inv["sales_30d_observed_days"] = 30
        if scenario in ("R", "T"):
            inv.drop(columns="unit_price", inplace=True)
        if scenario in ("S", "T"):
            inv.drop(columns="unit_cost", inplace=True)
        result = imi.build_sales_normalization_analysis(upload)
        n = result["validated_sales_signals"][0]
        assert n["sales_30d"] == 60.0 and n["avg_daily_sales"] == 2.0
        abc = result["validated_abc"]
        assert abc["basis"] == (imi.ABC_COST_VALUED if scenario == "R" else imi.ABC_QUANTITY_PROXY if scenario == "T" else imi.ABC_REVENUE)
        if scenario == "T":
            assert abc["display_basis"] == "QUANTITY_BASED_ABC"
        if scenario == "U":
            assert abc["grades"]["P"]["value"] == 300.0
    elif scenario in ("D", "Q", "N"):
        n = normalized(upload)
        turn = imi.build_sales_normalization_analysis(upload)["validated_turnover"][0]
        assert n["sales_30d"] == 60.0 and n["sales_7d"] == 14.0 and n["avg_daily_sales"] == 2.0
        assert n["retail_30d"]["observed_days"] == 30 and n["retail_30d"]["quantity_unit"] == "item"
        assert n["retail_30d"]["semantics_evidence"] == imi.DECLARED_SEMANTICS
        assert turn["turnover_value"] == 0.6 and turn["turnover_status"] == imi.TURNOVER_AVAILABLE
        assert turn["turnover_classification"] == "NORMAL"
        assert turn["source"] == "daily_sales_history" and turn["period"]["days"] == 30
    elif scenario == "E":
        upload["daily_sales_history"] = upload["daily_sales_history"].iloc[:10]
        n = normalized(upload)
        assert n["sales_30d"] is None and n["avg_daily_sales"] == 2.0
        assert n["windows"]["retail_30d"]["observed_days"] == 10
        assert "NO_RECORD_DAYS" in n["windows"]["retail_30d"]["reason_codes"]
    elif scenario == "F":
        upload["daily_sales_history"]["quantity"] = 0.0
        n = normalized(upload)
        assert n["sales_30d"] == 0 and n["avg_daily_sales"] == 0
        assert n["signals"]["monthly_retail_sales"]["day_state"] == imi.OBSERVED_ZERO
        assert imi.build_sales_normalization_analysis(upload)["validated_turnover"][0]["turnover_value"] == 0
        assert imi.build_sales_normalization_analysis(upload)["validated_turnover"][0]["days_of_supply"] == "INF"
        json.dumps(imi.build_sales_normalization_analysis(upload), allow_nan=False)
    elif scenario == "G":
        upload["daily_sales_history"] = upload["daily_sales_history"].iloc[:0]
        assert normalized(upload)["avg_daily_sales"] is None
    elif scenario in ("H", "I"):
        upload["daily_sales_history"].loc[0, "quantity"] = -3.0
        upload["daily_sales_history"]["negative_semantics"] = "DOCUMENTED_RETURN" if scenario == "H" else "UNKNOWN_NEGATIVE"
        n = normalized(upload)
        assert n["windows"]["retail_30d"]["negative_total"] == -3.0
        assert n["sales_30d"] == (55.0 if scenario == "H" else None)
        assert n["avg_daily_sales"] == (round(55 / 30, 6) if scenario == "H" else None)
    elif scenario in ("J", "K", "L", "M"):
        if scenario == "J":
            inv["stock_qty"] = 0
        elif scenario == "K":
            upload["inventory_history"]["quantity"] = 0
        else:
            upload["inventory_history"] = upload["inventory_history"].iloc[:1] if scenario == "L" else upload["inventory_history"].iloc[-1:]
        turn = imi.build_sales_normalization_analysis(upload)["validated_turnover"][0]
        assert turn["turnover_classification"] == "UNKNOWN"
        if scenario != "J":
            assert turn["turnover_value"] is None
        else:
            assert turn["days_of_supply_status"] == imi.DOS_ZERO_STOCK
    elif scenario == "O":
        upload["daily_sales_history"]["day_state"] = imi.NOT_CARRIED
        n = normalized(upload)
        assert n["sales_30d"] is None and n["avg_daily_sales"] is None
        assert n["windows"]["retail_30d"]["not_carried_days"] == 30
        assert "NOT_CARRIED_NOT_ZERO" in n["windows"]["retail_30d"]["reason_codes"]
    elif scenario == "P":
        upload["daily_sales_history"]["signal_type"] = imi.OBSERVED_OUTBOUND_FLOW
        n = normalized(upload)
        assert n["sales_30d"] is None and n["avg_daily_sales"] is None
        assert n["signals"]["monthly_outbound"]["value"] == 60.0
        assert n["signals"]["monthly_outbound"]["validation_status"] == imi.PROXY_ONLY
    elif scenario in ("V", "W"):
        result = imi.build_sales_normalization_analysis(upload, pd.DataFrame([{"disposal_risk_score": 72.0}]))
        risk = result["validated_risk_basis"][0]
        assert risk["risk_score"] == 72.0 and risk["risk_is_probability"] is False
        assert risk["actual_disposal_probability"] is None and risk["expected_disposal_qty"] is None
    elif scenario in ("X", "Y"):
        candidates = pd.DataFrame([dict(source_id="S", target_id="T", product_id="P", route_id="R", varo_action="보류", target_sales_30d=60)])
        result = imi.build_sales_normalization_analysis(upload, candidates=candidates)
        impact = result["validated_action_impact"][0]
        assert impact["matching_status"] == "DESTINATION_DEMAND_UNAVAILABLE"
        assert impact["destination_daily_demand"] is None and impact["validated_action"] is None
    elif scenario == "Z":
        upload["daily_sales_history"].loc[0, "date"] = "2026-06-30"
        n = normalized(upload)
        assert n["sales_30d"] is None and n["windows"]["retail_30d"]["observed_days"] == 29
    elif scenario == "AA":
        upload["daily_sales_history"]["quantity_unit"] = "kg"
        assert normalized(upload)["avg_daily_sales"] is None
    elif scenario == "AB":
        upload["daily_sales_history"].loc[0, "business_id"] = "OTHER"
        upload["daily_sales_history"].loc[1, "dataset_id"] = "OTHER"
        n = normalized(upload)
        assert n["sales_30d"] is None and n["windows"]["retail_30d"]["observed_days"] == 28
    elif scenario == "AC":
        upload["daily_sales_history"].loc[0, "quantity"] = -3
        upload["daily_sales_history"]["negative_semantics"] = "DOCUMENTED_RETURN"
        future = upload["daily_sales_history"].iloc[-1:].copy()
        future["date"], future["quantity"] = "2026-07-31", -999.0
        future["negative_semantics"] = "UNKNOWN_NEGATIVE"
        upload["daily_sales_history"] = pd.concat([upload["daily_sales_history"], future])
        n = normalized(upload)
        assert n["sales_30d"] == 55.0 and n["windows"]["retail_30d"]["future_records_excluded"] == 1
        assert n["windows"]["retail_30d"]["negative_semantics"] == imi.RETURN_DOCUMENTED
    elif scenario == "AD":
        assert imi.build_sales_normalization_analysis(upload) == imi.build_sales_normalization_analysis(upload)
    elif scenario == "AE":
        inv["sales_qty"] = 3.0
        upload = normalize_loaded_data(upload)
        expected = prepare_legacy_data(upload)["inventory"].iloc[0]
        n = imi.normalize_sales_signals(upload)["rows"][0]
        for key in ("sales_30d", "avg_daily_sales"):
            assert n[key] == expected[key]
    elif scenario == "AF":
        original = copy.deepcopy(upload)
        result = imi.build_sales_normalization_analysis(upload)
        for key in upload:
            pd.testing.assert_frame_equal(upload[key], original[key])
        assert result["semantics_mode"] == "LEGACY" and result["production_replacement_ready"] is False
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("state", ["MISSING_RECORD", "NOT_CARRIED", "OUTSIDE_PERIOD", "UNKNOWN", "NULL_VALUE"])
def test_explicit_missing_states_are_not_zero(state):
    upload = data(False, False)
    upload["inventory"]["daily_retail_sales"] = None
    upload["inventory"]["daily_retail_sales_day_state"] = state
    n = normalized(upload)
    assert n["sales_30d"] is None and n["avg_daily_sales"] is None
    assert n["signals"]["daily_retail_sales"]["day_state"] == state


def test_calendar_month_uses_calendar_days_not_thirty():
    upload = data(False, False)
    row = upload["inventory"]
    row["snapshot_date"] = "2026-02-28"
    row["monthly_retail_sales"] = 56.0
    row["monthly_retail_sales_aggregation"] = imi.CALENDAR_MONTH_SUM
    row["monthly_retail_sales_observation_start"] = "2026-02-01"
    row["monthly_retail_sales_observation_end"] = "2026-02-28"
    row["monthly_retail_sales_observed_days"] = 28
    n = normalized(upload)
    assert n["sales_30d"] is None and n["avg_daily_sales"] == 2.0
    assert n["signals"]["monthly_retail_sales"]["period"]["days"] == 28


def test_normalization_mode_is_explicit_and_strict():
    with pytest.raises(ValueError):
        imi.normalize_sales_signals({}, semantics_mode="AUTO")


def test_outbound_average_has_its_own_contract_not_retail_alias():
    upload = data()
    upload["daily_sales_history"]["signal_type"] = imi.OBSERVED_OUTBOUND_FLOW
    n = normalized(upload)
    mean = n["signals"]["average_daily_outbound"]
    assert mean["value"] == 2 and mean["signal_type"] == imi.OBSERVED_OUTBOUND_FLOW
    assert mean["observed_days"] == 30 and mean["period"]["days"] == 30
    assert n["avg_daily_sales"] is None and "average_daily_sales" not in n["signals"]


def test_stockout_zero_is_observed_but_not_uncensored_demand():
    upload = data()
    upload["daily_sales_history"].loc[0, "quantity"] = 0
    upload["daily_sales_history"]["stockout_hours"] = 0
    upload["daily_sales_history"].loc[0, "stockout_hours"] = 24
    n = normalized(upload)
    assert n["sales_30d"] == 58
    assert n["windows"]["retail_30d"]["stockout_zero_days"] == 1
    assert "ZERO_DURING_STOCKOUT" in n["windows"]["retail_30d"]["reason_codes"]


def test_business_identity_cannot_be_lost_from_history():
    upload = data()
    upload["daily_sales_history"].drop(columns="business_id", inplace=True)
    assert normalized(upload)["sales_30d"] is None


def test_copied_destination_field_never_replaces_actual_keyed_destination_history():
    upload = data()
    destination = upload["inventory"].copy()
    destination["store_id"] = "T"
    upload["inventory"] = pd.concat([upload["inventory"], destination], ignore_index=True)
    history = upload["daily_sales_history"].copy()
    history["location_id"], history["quantity"] = "T", 7
    upload["daily_sales_history"] = pd.concat([upload["daily_sales_history"], history])
    candidates = pd.DataFrame([dict(source_id="S", target_id="T", product_id="P", route_id="R", target_sales_30d=60)])
    result = imi.build_sales_normalization_analysis(upload, candidates=candidates)
    route = result["validated_route_matching_input"][0]
    assert route["destination_daily_demand"] == 7
    assert route["matching_status"] == "DESTINATION_DEMAND_AVAILABLE"
    assert result["validated_action_impact"][0]["validated_action"] is None


@pytest.mark.parametrize("state", ["MISSING_RECORD", "NULL_VALUE", "UNKNOWN", "OUTSIDE_PERIOD"])
def test_daily_history_missing_states_block_false_complete_window(state):
    upload = data()
    upload["daily_sales_history"].loc[0, "day_state"] = state
    n = normalized(upload)
    assert n["sales_30d"] is None
    assert n["windows"]["retail_30d"]["observed_days"] == 29
    assert n["windows"]["retail_30d"]["declared_day_states"][state] == 1


def test_weekly_partial_upload_is_not_a_complete_week():
    upload = data(False, False)
    row = upload["inventory"]
    row["sales_7d"] = 9.0
    row["sales_7d_semantics"] = "weekly_retail_sales"
    row["sales_7d_observation_start"] = "2026-07-24"
    row["sales_7d_observation_end"] = "2026-07-30"
    row["sales_7d_observed_days"] = 3
    n = normalized(upload)
    assert n["sales_7d"] is None
    assert "PARTIAL_OBSERVATION" in n["signals"]["sales_7d"]["reason_codes"]


def test_report_bundle_cannot_overwrite_previous_results(tmp_path):
    from services.sales_input_normalization_validation import run_validation
    folder = tmp_path / "_SALES_INPUT_NORMALIZATION_existing"
    folder.mkdir()
    with pytest.raises(FileExistsError):
        run_validation(tmp_path, folder)
    with pytest.raises(ValueError):
        run_validation(tmp_path, tmp_path / "_INVENTORY_METRIC_INTEGRITY")


def test_inventory_unit_is_never_inherited_from_sales():
    upload = data()
    upload["inventory_history"].drop(columns="quantity_unit", inplace=True)
    turn = imi.build_sales_normalization_analysis(upload)["validated_turnover"][0]
    assert turn["turnover_value"] is None and turn["turnover_classification"] == "UNKNOWN"
    assert "INVENTORY_UNIT_UNVERIFIABLE" in turn["reason_codes"]
