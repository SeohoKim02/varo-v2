"""Semantic tests use tiny explicit fixtures; no fabricated production data."""
import json

import pandas as pd
import pytest

from services.canonical_schema import VERSION, canonicalize, parse_period, schema_document
from services.real_data_adapters import adapt_aihub, adapt_jangbogo, adapt_logisall, adapt_nfqs, adapt_suhyup, base_frame, suhyup_to_existing, verified_identifier_crosswalk
from services.canonical_data_pipeline import Exporter, algorithm_coverage


def stock_frame(values):
    raw = pd.DataFrame({"qty": values})
    out = base_frame(raw, "test", "raw/test.csv")
    out["inventory_qty"] = raw.qty
    out["snapshot_date"] = "2026-01-01"
    out["product_id"], out["location_id"], out["unit"] = "P", "L", "kg"
    return out


def test_canonical_schema_defines_required_nullable_and_keys():
    tables = schema_document()["tables"]
    assert len(tables) == 7
    assert tables["inventory_snapshot"]["columns"]["inventory_qty"]["nullable"] is False
    assert tables["inventory_snapshot"]["columns"]["capacity"]["nullable"] is True
    assert "product_state" in tables["inventory_snapshot"]["candidate_key"]


def test_missing_is_not_zero():
    out = canonicalize(stock_frame([0, None, 2.25]), "inventory_snapshot")
    assert out.inventory_qty.iloc[0] == 0
    assert pd.isna(out.inventory_qty.iloc[1])
    assert out.inventory_qty.iloc[2] == 2.25
    assert "missing:inventory_qty" in out.validation_flags.iloc[1]


def test_units_remain_observed_not_guessed():
    data = stock_frame([1, 2])
    data["unit"] = ["ton", None]
    out = canonicalize(data, "inventory_snapshot")
    assert out.unit.iloc[0] == "ton"
    assert pd.isna(out.unit.iloc[1])
    assert "unit_unspecified" in out.validation_flags.iloc[1]


def test_provenance_is_preserved():
    out = canonicalize(stock_frame([1]), "inventory_snapshot")
    assert out.source_file.iloc[0] == "raw/test.csv"
    assert out.source_row_id.iloc[0] == "1"
    assert out.source_sheet_or_table.iloc[0] == "csv"
    assert out.transform_version.iloc[0] == VERSION


def test_invalid_dates_are_not_repaired():
    start, end = parse_period(pd.Series(["20230229", "20240229", "99991301"]), "daily")
    assert pd.isna(start.iloc[0]) and pd.isna(end.iloc[2])
    assert start.iloc[1] == "2024-02-29"


def test_monthly_grain_not_daily_interpolation():
    start, end = parse_period(pd.Series(["202402", "202413"]), "monthly")
    assert start.iloc[0] == "2024-02-01" and end.iloc[0] == "2024-02-29"
    assert pd.isna(start.iloc[1])


def test_negative_quantity_preserved_and_ineligible():
    out = canonicalize(stock_frame([-3.5]), "inventory_snapshot")
    assert out.inventory_qty.iloc[0] == -3.5
    assert "negative:inventory_qty" in out.validation_flags.iloc[0]
    assert out.analysis_eligible.iloc[0] == "false"


def test_infinite_and_invalid_numeric_are_flagged():
    out = canonicalize(stock_frame([float("inf"), "wrong"]), "inventory_snapshot")
    assert out.inventory_qty.isna().all()
    assert "nonfinite:inventory_qty" in out.validation_flags.iloc[0]
    assert "invalid_numeric:inventory_qty" in out.validation_flags.iloc[1]


def suhyup_raw():
    return pd.DataFrame({"물류센터-공판장 코드": ["001"], "물류센터-공판장명": ["센터"], "수산물품목코드": ["010"], "수산물품목명": ["생선"], "상태가공분류코드": ["30"], "기준일자": ["2026-07-01"], "재고량": [7.5], "재고량(킬로그램)": [40.0], "입고량": [2.5], "출고량": [0], "입고량(킬로그램)": [10], "출고량(킬로그램)": [0]})


def test_suhyup_adapter_preserves_state_count_and_weight():
    _, out, _ = adapt_suhyup(suhyup_raw(), "raw/stock.csv", "stock")
    assert out.location_id.iloc[0] == "001" and out.product_id.iloc[0] == "010"
    assert out.product_state.iloc[0] == "30"
    assert out.inventory_qty.iloc[0] == 7.5 and out.inventory_weight_kg.iloc[0] == 40
    assert pd.isna(out.unit.iloc[0])


def test_suhyup_bridge_exactly_preserves_existing_input():
    _, stock, _ = adapt_suhyup(suhyup_raw(), "raw/stock.csv", "stock")
    _, flow, _ = adapt_suhyup(suhyup_raw(), "raw/flow.csv", "flow")
    legacy = suhyup_to_existing(stock, flow)
    assert legacy.stock_qty.iloc[0] == 7.5
    assert legacy.outbound_qty.iloc[0] == 0
    assert legacy.center_code.iloc[0] == "001"


def test_suhyup_bridge_rejects_unmatched_stock_flow():
    _, stock, _ = adapt_suhyup(suhyup_raw(), "raw/stock.csv", "stock")
    _, flow, _ = adapt_suhyup(suhyup_raw(), "raw/flow.csv", "flow")
    flow.loc[0, "product_id"] = "different"
    with pytest.raises(ValueError, match="key mismatch"):
        suhyup_to_existing(stock, flow)


def test_logisall_distribution_not_recommendation():
    raw = pd.DataFrame({"BASE_YMD": ["20221101"], "AGFD_PDLT_NM": ["배"], "FRWAR_ZIP": ["00100"], "ARVL_ZIP": ["00200"], "DSBN_QY": [12]})
    table, out, _ = adapt_logisall(raw, "raw/file.csv", "network")
    assert table == "transfer_network" and out.shipment_qty.iloc[0] == 12
    assert pd.isna(out.recommended_qty.iloc[0]) and pd.isna(out.transport_cost.iloc[0])
    assert out.source_id.iloc[0] == "00100"


def test_logisall_monthly_national_stock_has_no_fake_location():
    raw = pd.DataFrame({"BASE_YR": ["2022"], "BASE_MM": ["2"], "AGFD_PDLT_NM": ["배"], "STRGE_QY": [12]})
    _, out, _ = adapt_logisall(raw, "raw/file.csv", "stock")
    assert out.date_grain.iloc[0] == "monthly"
    assert out.period_end.iloc[0] == "2022-02-28"
    assert pd.isna(out.location_id.iloc[0]) and out.scope.iloc[0] == "national"


def test_jangbogo_purchase_request_is_not_receipt_or_demand():
    raw = pd.DataFrame({"cfmtn_ym": ["202304"], "fdmt_pdlt_code": ["0001"], "fdmt_pdlt_nm": ["상품"], "prca_dmnd_qyt": [1.5]})
    _, out, _, _ = adapt_jangbogo(raw, "raw/file.csv", "orders")
    assert out.order_qty.iloc[0] == 1.5
    assert out.inbound_qty.isna().all() and out.demand_qty.isna().all()
    assert out.location_id.isna().all()


def test_jangbogo_bad_date_remains_traceable():
    raw = pd.DataFrame({"cfmtn_ym": ["202399"], "fdmt_pdlt_code": ["0001"], "fdmt_pdlt_nm": ["상품"], "prca_dmnd_qyt": [1]})
    _, out, _, _ = adapt_jangbogo(raw, "raw/file.csv", "orders")
    assert out.raw_date.iloc[0] == "202399" and pd.isna(out.date.iloc[0])
    assert "invalid_date" in out.validation_flags.iloc[0]


def test_warehouse_occurrence_does_not_invent_quantity():
    raw = pd.DataFrame({"cfmtn_ymd": ["20230401"], "wrhs_code": ["01"], "fdmt_pdlt_code": ["0001"], "fdmt_pdlt_nm": ["상품"]})
    table, out, measures, _ = adapt_jangbogo(raw, "raw/file.csv", "observation")
    assert table == "location_product_observation" and not measures
    assert out.purchase_count.isna().all()


def test_nfqs_keeps_quarter_grain_and_region_missing():
    out = adapt_nfqs([{"ICEGDFG": "011", "CODEKNM": "고등어", "SEOUL": 0, "INVENTOTAL": 2.25}], "raw/quarter.json", 2020, 1)
    assert len(out) == 15 and out.date_grain.eq("quarterly").all()
    assert out.snapshot_date.eq("2020-03-31").all()
    assert out.loc[out.location_id == "SEOUL", "inventory_qty"].iloc[0] == 0
    assert out.loc[out.location_id == "BUSAN", "inventory_qty"].isna().all()
    assert out.unit.eq("ton").all()


def test_nfqs_reported_total_not_additional_product():
    out = adapt_nfqs([{"ICEGDFG": "999", "CODEKNM": "합계", "INVENTOTAL": 10}], "raw/q.json", 2020, 1)
    assert out.product_grain.eq("all_products").all()
    assert out.validation_flags.str.contains("aggregate_product_total").all()


def test_aihub_numeric_subset_does_not_assign_site_or_capacity():
    raw = pd.DataFrame([[pd.Timestamp("2024-08-01"), 10, 2, 20, *([0]*24), 10.0, 0.5]])
    outputs = adapt_aihub(raw, "raw/flow.xlsx", "flow")
    snapshot = outputs[0][1]
    assert snapshot.inventory_qty.iloc[0] == 20 and snapshot.inventory_volume_m3.iloc[0] == 10
    assert snapshot.capacity.isna().all() and snapshot.location_id.isna().all() and snapshot.product_id.isna().all()
    assert snapshot.source_row_id.iloc[0] == "3"


def test_cross_chunk_duplicate_handling_preserves_rows(tmp_path):
    e = Exporter(tmp_path, "aihub")
    out = canonicalize(stock_frame([1]), "inventory_snapshot")
    e.write("inventory_snapshot", out, masters=False)
    e.write("inventory_snapshot", out, masters=False)
    report = e.finish()
    assert report["row_counts"]["inventory_snapshot"] == 2
    assert report["flags"]["inventory_snapshot"]["duplicate_key"] == 1
    assert len(pd.read_parquet(e.processed / "canonical_inventory_snapshot.parquet")) == 2


def test_quantity_conservation_retains_fraction_zero_and_missing(tmp_path):
    e = Exporter(tmp_path, "aihub")
    raw = pd.DataFrame({"qty": [0, 1.25, None, -0.5]})
    out = canonicalize(stock_frame(raw.qty), "inventory_snapshot")
    e.write("inventory_snapshot", out, raw, {"inventory_qty": "qty"}, masters=False)
    report = e.finish()
    assert report["all_quantity_checks_passed"]
    check = report["quantity_conservation"][0]
    assert check["source_sum"] == check["canonical_sum"] == 0.75
    assert check["canonical_nonnull"] == 3


def test_unsupported_algorithm_not_enabled_by_unrelated_columns():
    out = algorithm_coverage("nfqs", {"inventory_qty", "product_id", "location_id"}).set_index("algorithm")
    assert out.loc["DQN", "support_level"] == "UNSUPPORTED"
    assert out.loc["Varo Final", "supported"] == False
    assert "transport_cost" in out.loc["MILP", "missing_fields"]


def test_verified_crosswalk_preserves_raw_ids_and_maps_only_observed_counterparts():
    assert verified_identifier_crosswalk(["000156", "000157"], ["156", "157"]) == {"000156": "156", "000157": "157"}
    with pytest.raises(ValueError, match="counterpart"):
        verified_identifier_crosswalk(["000999"], ["156"])


def test_crosswalk_rejects_ambiguous_normalization():
    with pytest.raises(ValueError, match="Ambiguous"):
        verified_identifier_crosswalk(["0001"], ["1", "01"])


def test_missing_required_source_column_fails_loudly():
    with pytest.raises(ValueError, match="Required mapped source column absent"):
        adapt_suhyup(suhyup_raw().drop(columns="재고량"), "raw/stock.csv", "stock")
