"""External canonical adapters (schema 1.2.0): tiny explicit fixtures, no production data."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.canonical_data_pipeline import ALGORITHMS, Exporter, algorithm_coverage
from services.canonical_schema import EXTERNAL_VERSION, QUALITY_CODES, canonicalize, finalize_quality, is_blocking, quality_code, schema_document
from services.external_canonical_pipeline import run_favorita, run_freshretailnet, run_kamp, run_m5, write_validation_matrix
from services.external_data_adapters import (KAMP_GRADES, adapt_favorita_masters, adapt_favorita_train, adapt_frn_sales, adapt_kamp_shipments,
    adapt_m5_sales, collapse_identical, source_boolean)
from services.real_data_adapters import DATASETS, base_frame
from tests.test_canonical_integrity import read, write_csv

ROUTING = ["Transport Cost", "Greedy", "VHS", "Pareto", "MILP", "Optimality Gap", "Varo Final", "DQN"]
EVERY_FIELD = {f for fields in ALGORITHMS.values() for f in fields}


# ---------------------------------------------------------------- M5 fixtures

CALENDAR = pd.DataFrame({"date": ["2011-01-29", "2011-01-30", "2011-01-31", "2011-02-05"], "wm_yr_wk": ["11101", "11101", "11101", "11102"],
                         "d": ["d_1", "d_2", "d_3", "d_4"], "event_name_1": [None, "SuperBowl", None, None], "event_type_1": [None, "Sporting", None, None],
                         "event_name_2": [None] * 4, "event_type_2": [None] * 4,
                         "snap_CA": ["0", "1", "0", "1"], "snap_TX": ["0", "0", "1", "1"], "snap_WI": ["1", "0", "0", "0"]})


def m5_sales(days=("d_1", "d_2", "d_3"), suffix="evaluation"):
    values = {"d_1": ["0", "3", "1"], "d_2": ["0", "0", "1"], "d_3": ["0", "4", "0"]}
    series = pd.DataFrame({"item_id": ["FOODS_1_001", "FOODS_1_001", "HOBBIES_1_002"], "dept_id": ["FOODS_1", "FOODS_1", "HOBBIES_1"],
                           "cat_id": ["FOODS", "FOODS", "HOBBIES"], "store_id": ["CA_1", "TX_1", "CA_1"], "state_id": ["CA", "TX", "CA"]})
    series.insert(0, "id", series.item_id + "_" + series.store_id + "_" + suffix)
    return series.assign(**{d: values[d] for d in days})


# CA_1 FOODS never has a price (never sold); TX_1 week 11102 lies beyond the sales horizon.
PRICES = pd.DataFrame({"store_id": ["TX_1", "CA_1", "TX_1"], "item_id": ["FOODS_1_001", "HOBBIES_1_002", "FOODS_1_001"],
                       "wm_yr_wk": ["11101", "11101", "11102"], "sell_price": ["3.0", "9.5", "3.1"]})


def write_m5(root):
    folder = root / DATASETS["m5"] / "raw/extracted/m5-forecasting-accuracy"
    write_csv(folder / "calendar.csv", CALENDAR)
    write_csv(folder / "sales_train_evaluation.csv", m5_sales())
    write_csv(folder / "sales_train_validation.csv", m5_sales(("d_1", "d_2"), "validation"))
    write_csv(folder / "sell_prices.csv", PRICES)
    write_csv(folder / "sample_submission.csv", pd.DataFrame({"id": ["x_validation"], "F1": ["0"], "F2": ["0"]}))


def run(root, dataset, runner):
    e = Exporter(root, dataset)
    runner(e)
    return e, e.finish()


def test_m5_wide_to_long_conservation(tmp_path):
    write_m5(tmp_path)
    e, report = run(tmp_path, "m5", run_m5)
    sales = read(e, "demand_series")
    # 3 series x 3 days: every cell exactly once; zeros stay zeros.
    assert len(sales) == 9 and report["row_counts"]["demand_series"] == 9
    check = next(c for c in report["quantity_conservation"] if c["field"] == "sales_qty")
    assert check["source_cells"] == check["rows_canonical"] == 9 and check["rows_source"] == 3
    assert check["source_sum"] == check["canonical_sum"] == 9 and check["source_zero"] == check["canonical_zero"] == 5 and check["passed"]
    assert report["semantic_checks"]["key_uniqueness"]["demand_series"]["keys_repeated"] == 0
    assert report["semantic_checks"]["duplicate_release_investigation"]["strict_prefix"] is True
    # The validation release and the scoring template are never converted.
    assert set(sales.source_file) == {f"{DATASETS['m5']}/raw/extracted/m5-forecasting-accuracy/sales_train_evaluation.csv"}
    assert not (e.processed / "canonical_inventory_snapshot.parquet").exists() and not (e.processed / "canonical_transfer_network.parquet").exists()
    assert report["all_quantity_checks_passed"]


def test_m5_price_join(tmp_path):
    write_m5(tmp_path)
    e, report = run(tmp_path, "m5", run_m5)
    sales = read(e, "demand_series").set_index(["location_id", "product_id", "date"]).sort_index()
    # The weekly price repeats on each sales day of its week; currency only where a price exists.
    assert sales.loc[("TX_1", "FOODS_1_001"), "price"].tolist() == [3.0, 3.0, 3.0]
    assert sales.loc[("CA_1", "HOBBIES_1_002"), "currency"].eq("USD").all()
    price = next(c for c in report["quantity_conservation"] if c["field"] == "price")
    assert price["passed"] and price["priced_rows"] == price["expected_priced_rows"] == 6
    assert price["records_in_horizon"] == price["records_joined"] == 2 and price["records_beyond_horizon"] == 1
    assert price["canonical_price_sum"] == price["source_price_x_days_sum"] == 37.5
    # A price for a week without sales days is not attached to any row.
    assert 3.1 not in set(sales.price.dropna())


def test_m5_missing_is_not_zero():
    wide = m5_sales()
    wide.loc[1, "d_2"] = None
    table, out, _ = adapt_m5_sales(wide, "raw/sales.csv", CALENDAR, PRICES)
    out = out.set_index("source_row_id")
    # A NULL cell stays NULL and is flagged; a zero stays an observed zero.
    assert pd.isna(out.loc["2:d_2", "sales_qty"]) and "missing:sales_qty" in out.loc["2:d_2", "validation_flags"]
    assert out.loc["2:d_2", "analysis_eligible"] == "false"
    never_sold = out.loc[["1:d_1", "1:d_2", "1:d_3"]]
    assert never_sold.sales_qty.eq(0).all() and never_sold.price.isna().all() and never_sold.currency.isna().all()
    # Guide: a missing weekly price means not sold that week -> informational, never a 0 price.
    assert never_sold.validation_flags.eq("weekly_price_absent").all() and never_sold.analysis_eligible.eq("true").all()
    assert never_sold.quality_status.eq("MISSING_VALUE").all()
    # A sale in a week without a price contradicts the guide and blocks the row.
    sold = wide.copy()
    sold.loc[0, "d_1"] = "2"
    _, contradiction, _ = adapt_m5_sales(sold, "raw/sales.csv", CALENDAR, PRICES)
    row = contradiction.set_index("source_row_id").loc["1:d_1"]
    assert row.sales_qty == 2 and pd.isna(row.price) and "sales_without_weekly_price" in row.validation_flags and row.analysis_eligible == "false"


# ---------------------------------------------------------------- Favorita

def favorita_train():
    return pd.DataFrame({"id": ["0", "1", "2", "3"], "date": ["2013-01-01", "2013-01-01", "2013-01-02", "2013-01-02"],
                         "store_nbr": ["25", "25", "1", "1"], "item_nbr": ["103665", "105574", "103665", "105574"],
                         "unit_sales": ["7.0", "-2.0", "1.5", "3.0"], "onpromotion": [None, "True", "False", None]}, dtype="string")


FAMILIES = pd.Series({"103665": "BREAD/BAKERY", "105574": "GROCERY I"})


def test_favorita_signed_sales_semantics(tmp_path):
    table, out, measures = adapt_favorita_train(favorita_train(), "raw/train.csv", FAMILIES)
    assert table == "demand_series" and out.sales_qty.tolist() == [7.0, -2.0, 1.5, 3.0]
    ret = out.iloc[1]
    # Official description: negative = return. Kept signed, never abs()/0; no returns quantity is invented.
    assert "negative:sales_qty" in ret.validation_flags and "documented_return:sales_qty" in ret.validation_flags
    assert ret.quality_status == "NEGATIVE_QUANTITY" and ret.analysis_eligible == "false"
    assert quality_code("documented_return:sales_qty") == "NEGATIVE_QUANTITY" and not is_blocking("documented_return:sales_qty")
    assert "returns_qty" not in out.columns and out.demand_qty.isna().all()
    assert not out.iloc[[0, 2, 3]].validation_flags.str.contains("documented_return").any()
    # Item-dependent count/kg unit is never guessed.
    assert out.unit.isna().all() and out.unit_status.eq("UNKNOWN").all()
    e = Exporter(tmp_path, "favorita")
    e.write(table, out, favorita_train(), measures, masters=False)
    report = e.finish()
    check = report["quantity_conservation"][0]
    assert check["source_sum"] == check["canonical_sum"] == 9.5 and check["passed"]


def test_favorita_product_location_identity():
    stores = pd.DataFrame({"store_nbr": ["1", "25"], "city": ["Quito", "Salinas"], "state": ["Pichincha", "Santa Elena"], "type": ["D", "D"], "cluster": ["13", "1"]}, dtype="string")
    items = pd.DataFrame({"item_nbr": ["103665", "105574"], "family": ["BREAD/BAKERY", "GROCERY I"], "class": ["2712", "1045"], "perishable": ["1", "0"]}, dtype="string")
    locations, products = adapt_favorita_masters(stores, items, "raw/stores.csv", "raw/items.csv")
    assert locations.location_key.tolist() == ["Favorita:store_nbr:1", "Favorita:store_nbr:25"]
    assert locations[["city", "region", "location_subtype", "location_cluster"]].iloc[1].tolist() == ["Salinas", "Santa Elena", "D", "1"]
    assert products.product_key.tolist() == ["Favorita:item_nbr:103665", "Favorita:item_nbr:105574"]
    assert products.category_path.tolist() == ["family=BREAD/BAKERY|class=2712", "family=GROCERY I|class=1045"]
    assert products.perishable.tolist() == ["true", "false"]
    _, sales, _ = adapt_favorita_train(favorita_train(), "raw/train.csv", FAMILIES)
    # The raw dataset-local id is unchanged; the key carries the namespace.
    assert sales.product_id.iloc[0] == "103665" and sales.location_key.iloc[0] == "Favorita:store_nbr:25"
    assert sales.category.tolist() == ["BREAD/BAKERY", "GROCERY I", "BREAD/BAKERY", "GROCERY I"]


def test_favorita_promotion_preservation():
    _, out, _ = adapt_favorita_train(favorita_train(), "raw/train.csv", FAMILIES)
    # NULL is not False.
    assert pd.isna(out.promotion.iloc[0]) and out.promotion.iloc[1:3].tolist() == ["true", "false"] and pd.isna(out.promotion.iloc[3])
    assert out.validation_flags.str.contains("promotion_not_reported").tolist() == [True, False, False, True]
    assert quality_code("promotion_not_reported") == "MISSING_VALUE" and not is_blocking("promotion_not_reported")
    with pytest.raises(ValueError, match="Undocumented boolean"):
        source_boolean(pd.Series(["True", "yes"]), ("True",), ("False",))


def test_favorita_run_keeps_absent_rows_absent(tmp_path):
    folder = tmp_path / DATASETS["favorita"] / "raw/favorita_grocery_sales_forecasting/extracted/favorita-grocery-sales-forecasting"
    write_csv(folder / "train.csv", favorita_train())
    write_csv(folder / "test.csv", pd.DataFrame({"id": ["4"], "date": ["2013-01-03"], "store_nbr": ["1"], "item_nbr": ["103665"], "onpromotion": ["False"]}))
    write_csv(folder / "sample_submission.csv", pd.DataFrame({"id": ["4"], "unit_sales": ["0"]}))
    write_csv(folder / "stores.csv", pd.DataFrame({"store_nbr": ["1", "25"], "city": ["Quito", "Salinas"], "state": ["P", "S"], "type": ["D", "D"], "cluster": ["13", "1"]}))
    write_csv(folder / "items.csv", pd.DataFrame({"item_nbr": ["103665", "105574"], "family": ["BREAD/BAKERY", "GROCERY I"], "class": ["2712", "1045"], "perishable": ["1", "0"]}))
    write_csv(folder / "transactions.csv", pd.DataFrame({"date": ["2013-01-01"], "store_nbr": ["25"], "transactions": ["770"]}))
    write_csv(folder / "oil.csv", pd.DataFrame({"date": ["2013-01-01", "2013-01-02"], "dcoilwtico": [None, "93.14"]}))
    write_csv(folder / "holidays_events.csv", pd.DataFrame({"date": ["2012-10-09"], "type": ["Holiday"], "locale": ["Local"], "locale_name": ["Guayaquil"],
                                                             "description": ["Independencia de Guayaquil"], "transferred": ["True"]}))
    e, report = run(tmp_path, "favorita", run_favorita)
    sales = read(e, "demand_series")
    # 4 source records -> 4 rows; no zero rows are generated for absent (date, store, item) keys; test.csv is not converted.
    assert len(sales) == 4 and report["semantic_checks"]["zero_sales_absence"]["zero_rows"] == 0
    assert not sales.source_file.str.contains("test.csv").any()
    covariates = read(e, "covariate_series").set_index("covariate_name")
    oil = covariates.loc["dcoilwtico"]
    assert pd.isna(oil.covariate_value.iloc[0]) and "missing:covariate_value" in oil.validation_flags.iloc[0] and oil.covariate_value.iloc[1] == 93.14
    assert covariates.loc["transactions", "unit"] == "transaction" and covariates.loc["transactions", "location_key"] == "Favorita:store_nbr:25"
    event = read(e, "calendar_event").iloc[0]
    assert event.event_transferred == "true" and event.event_locale_name == "Guayaquil" and "location_id" not in read(e, "calendar_event")
    assert report["semantic_checks"]["product_master_mapping"]["unmatched_identifiers"] == []
    assert report["all_quantity_checks_passed"]


# ---------------------------------------------------------------- FreshRetailNet

def frn_table(stock_cnt=(3, 0), discount=(1.0, 0.0), weather=(8.8, 8.8)):
    status = [[1, 1, 1] + [0] * 18 + [1, 1, 1], [0] * 24]
    status[0][6], status[0][7] = 1, 1
    hours = [[0.0] * 24, [0.0] * 24]
    hours[0][10], hours[0][19] = 0.1, 0.30000000000000004
    hours[1][9] = 1.2
    return pa.table({"city_id": [0, 0], "store_id": [3, 3], "management_group_id": [0, 0], "first_category_id": [5, 5], "second_category_id": [6, 6],
                     "third_category_id": [65, 65], "product_id": [38, 39], "dt": ["2024-06-26", "2024-06-26"],
                     "sale_amount": [sum(hours[0]), sum(hours[1])], "hours_sale": hours, "stock_hour6_22_cnt": pa.array(stock_cnt, pa.int32()),
                     "hours_stock_status": status, "discount": list(discount), "holiday_flag": pa.array([0, 0], pa.int32()),
                     "activity_flag": pa.array([0, 1], pa.int32()), "precpt": list(weather), "avg_temperature": [27.4, 27.4],
                     "avg_humidity": [81.7, 81.7], "avg_wind_level": [1.55, 1.55]})


def test_freshretailnet_schema_integrity():
    frame, scalars, status, hourly = adapt_frn_sales(frn_table(), "raw/eval.parquet", 0)
    first = frame.iloc[0]
    assert first.hourly_stockout_status == "111000110000000000000111" and len(first.hourly_stockout_status) == 24
    # Out-of-stock hours 6, 7 and 21 fall in the 6:00-22:00 window.
    assert first.stockout_hours == 3 and first.stockout_window_hours == 16
    # Hourly sales text round-trips to the same doubles and sums to the daily amount.
    assert [float(v) for v in first.hourly_sales.split("|")] == frn_table().column("hours_sale").to_pylist()[0]
    assert first.unit == "normalized_sales_amount" and first.unit_status == "SOURCE_METADATA"
    assert frame.promotion.tolist() == ["false", "true"] and frame.discount_rate.tolist() == [1.0, 0.0]
    assert frame.product_key.tolist() == ["FreshRetailNet:product_id:38", "FreshRetailNet:product_id:39"]
    # Discount 0 is a flagged covariate anomaly; the sales observation stays usable. No stock quantity exists.
    assert frame.validation_flags.tolist() == ["", "discount_rate_zero"] and frame.analysis_eligible.eq("true").all()
    assert frame.columns.intersection(["inventory_qty", "available_qty"]).empty
    broken, _, _, _ = adapt_frn_sales(frn_table(stock_cnt=(5, 0)), "raw/eval.parquet", 0)
    assert "stockout_count_mismatch" in broken.validation_flags.iloc[0] and broken.analysis_eligible.iloc[0] == "false"
    short = frn_table().set_column(9, "hours_sale", pa.array([[0.0] * 23, [0.0] * 24]))
    with pytest.raises(ValueError, match="24-hour"):
        adapt_frn_sales(short, "raw/eval.parquet", 0)
    with pytest.raises(ValueError, match="Values differ"):
        collapse_identical(frn_table(weather=(8.8, 9.9)).to_pandas(), ("store_id", "dt"), ("precpt",))


def test_freshretailnet_run_covariate_lineage(tmp_path):
    raw = tmp_path / DATASETS["freshretailnet"] / "raw"
    raw.mkdir(parents=True)
    pq.write_table(frn_table(), raw / "train.parquet")
    pq.write_table(frn_table().set_column(7, "dt", pa.array(["2024-06-27", "2024-06-27"])), raw / "eval.parquet")
    e = Exporter(tmp_path, "freshretailnet")
    run_freshretailnet(e, batch_size=1)
    report = e.finish()
    covariates = read(e, "covariate_series")
    weather = covariates[covariates.covariate_name.eq("precpt")]
    # Identical store-day weather of two product rows collapses to one row that lists both records.
    assert weather.source_row_id.tolist() == ["1|2", "1|2"] and weather.source_record_count.tolist() == [2, 2]
    assert covariates[covariates.covariate_name.eq("holiday_flag")].scope.eq("national").all()
    assert len(read(e, "demand_series")) == 4 and not (e.processed / "canonical_inventory_snapshot.parquet").exists()
    assert report["semantic_checks"]["key_uniqueness"]["demand_series"]["keys_repeated"] == 0 and report["all_quantity_checks_passed"]


# ---------------------------------------------------------------- KAMP

def kamp_raw():
    rows = {"공사": ["SK에코**  A**구역"] * 2, "부위": ["104동 9층 벽체", "113동 기초 상부근"], "출하일자": ["2023/09/25", "2022/10/11"],
            "합계": [10.129, 16.456], "HD10": [8, 0], "HD13": [1.522, 0], "SHD10": [0, 0], "SHD13": [0.134, 0], "UHD16": [0.473, 0],
            "UHD19": [0, 16.456], "UHD22": [0, 0], "UHD22S": [0, 0], "UHD25": [0, 0]}
    return pd.DataFrame(rows)


def write_kamp(root):
    import openpyxl
    folder = root / DATASETS["kamp"]
    (folder / "raw").mkdir(parents=True)
    (folder / "processed").mkdir()
    (folder / "processed/guidebook_text.txt").write_text("공사 명목형 발주공사명\n부위 발주공사부위\n", encoding="utf-8")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Export"
    raw = kamp_raw()
    sheet.append(list(raw.columns))
    for row in raw.itertuples(index=False):
        sheet.append(list(row))
    book.save(folder / "raw/kamp.xlsx")


def test_kamp_wide_rebar_conversion():
    raw = kamp_raw()
    raw.loc[1, "UHD25"] = None
    out, long = adapt_kamp_shipments(raw, "raw/kamp.xlsx", "Export", [2, 3])
    # 2 rows x (9 grades + 합계) cells, in worksheet-row then column order.
    assert len(out) == 20 and out.source_row_id.iloc[:3].tolist() == ["2:HD10", "2:HD13", "2:SHD10"]
    grade = out[out.product_grain.eq("product")]
    assert set(grade.product_id) == set(KAMP_GRADES) and grade.product_key.str.startswith("KAMP:rebar_grade:").all()
    assert out.set_index("source_row_id").loc["2:HD10", "shipment_qty"] == 8 and out.set_index("source_row_id").loc["2:SHD10", "shipment_qty"] == 0
    total = out[out.product_grain.eq("all_products")]
    assert total.product_id.isna().all() and total.product_key.isna().all() and total.shipment_qty.tolist() == [10.129, 16.456]
    assert total.validation_flags.str.contains("aggregate_product_total").all() and total.quality_flags.str.contains("OUT_OF_SCOPE").all()
    null_cell = out.set_index("source_row_id").loc["3:UHD25"]
    assert pd.isna(null_cell.shipment_qty) and "missing:shipment_qty" in null_cell.validation_flags
    assert out.date.tolist()[0] == "2023-09-25" and out.raw_date.iloc[0] == "2023/09/25" and out.date_grain.eq("event").all()
    assert out.project_id.eq("SK에코**  A**구역").all() and out.project_part.iloc[0] == "104동 9층 벽체"


def test_kamp_shipment_conservation(tmp_path):
    write_kamp(tmp_path)
    e, report = run(tmp_path, "kamp", run_kamp)
    checks = {c["source_column"]: c for c in report["quantity_conservation"] if c["field"] == "shipment_qty"}
    assert set(checks) == {*KAMP_GRADES, "합계"} and all(c["passed"] for c in checks.values())
    assert checks["HD13"]["source_sum"] == checks["HD13"]["canonical_sum"] == 1.522 and checks["UHD19"]["canonical_zero"] == 1
    assert checks["합계"]["canonical_sum"] == pytest.approx(26.585)
    flow = read(e, "inventory_flow")
    # Grade rows sum to the reported row totals; totals are never added on top.
    assert flow[flow.product_grain.eq("product")].shipment_qty.sum() == pytest.approx(flow[flow.product_grain.eq("all_products")].shipment_qty.sum())
    assert report["semantic_checks"]["row_total_identity"]["rows_equal_within_1e-9"] == 2
    assert report["semantic_checks"]["guidebook_semantics"]["공사"]["found"] is True


def test_kamp_no_fake_inventory(tmp_path):
    write_kamp(tmp_path)
    e, report = run(tmp_path, "kamp", run_kamp)
    for table in ("inventory_snapshot", "location_master", "transfer_network", "demand_series"):
        assert not (e.processed / f"canonical_{table}.parquet").exists(), table
    flow = read(e, "inventory_flow")
    assert "inventory_qty" not in flow and flow[["inbound_qty", "outbound_qty", "sales_qty", "order_qty", "demand_qty"]].isna().all().all()
    # No shipping location is invented; the project is not promoted to a location.
    assert flow.location_id.isna().all() and flow.quality_flags.str.contains("UNKNOWN_LOCATION").all()
    assert flow.unit.isna().all() and flow.analysis_eligible.eq("false").all()
    coverage = pd.read_csv(e.results / "algorithm_coverage.csv").set_index("algorithm").support_level
    assert coverage[["Turnover", "Disposal Risk", "Safety Stock", "EOQ", *ROUTING]].eq("UNSUPPORTED").all()


# ---------------------------------------------------------------- identity and provenance

def test_cross_dataset_id_namespace_separation():
    raw = favorita_train().assign(item_nbr="1", store_nbr="1")
    _, favorita, _ = adapt_favorita_train(raw, "raw/train.csv", pd.Series({"1": "X"}))
    table = frn_table().set_column(6, "product_id", pa.array([1, 1])).set_column(1, "store_id", pa.array([1, 1]))
    frn, _, _, _ = adapt_frn_sales(table, "raw/eval.parquet", 0)
    union = pd.concat([favorita, frn], ignore_index=True)
    # The same raw spelling in two datasets stays two products and two locations.
    assert set(union.product_id) == {"1"} and union.product_key.nunique() == 2 and union.location_key.nunique() == 2
    assert set(union.product_key) == {"Favorita:item_nbr:1", "FreshRetailNet:product_id:1"}
    # A key is derived only from a namespace: an un-namespaced (1.1.0) row keeps a NULL key, never a guessed one.
    plain = base_frame(pd.DataFrame({"q": [1]}), "t", "raw/t.csv").assign(date="2026-01-01", location_id="1", product_id="1", unit="kg", unit_status="DIRECT")
    out = canonicalize(plain.assign(product_key="forged"), "demand_series")
    assert pd.isna(out.product_key.iloc[0]) and pd.isna(out.location_key.iloc[0])
    assert "identity_policy" in schema_document() and schema_document()["version"] == "1.2.0"


def test_provenance_for_wide_to_long():
    wide = m5_sales()
    wide.index = pd.RangeIndex(500, 503)
    _, m5, _ = adapt_m5_sales(wide, "raw/sales.csv", CALENDAR, PRICES)
    cells = {f"{r}:{d}" for r in (501, 502, 503) for d in ("d_1", "d_2", "d_3")}
    # Every (record, source column) appears exactly once and records keep their position across chunks.
    assert set(m5.source_row_id) == cells and m5.source_row_id.is_unique
    assert m5.set_index("source_row_id").loc["502:d_3", "sales_qty"] == 4
    assert m5.transform_version.eq(EXTERNAL_VERSION).all() and m5.source_file.eq("raw/sales.csv").all() and m5.raw_date.isin(["d_1", "d_2", "d_3"]).all()
    kamp, _ = adapt_kamp_shipments(kamp_raw(), "raw/kamp.xlsx", "Export", [7, 9])
    ids = kamp.source_row_id.str.split(":", expand=True)
    assert set(ids[0]) == {"7", "9"} and set(ids[1]) == {*KAMP_GRADES, "합계"} and kamp.source_row_id.is_unique
    assert kamp.source_sheet_or_table.eq("Export").all()


# ---------------------------------------------------------------- coverage and exporter

def test_algorithm_coverage_gate():
    m5 = algorithm_coverage("m5", EVERY_FIELD | {"consistent_unit"}).set_index("algorithm")
    assert m5.loc["Demand Forecast", "support_level"] == "FULL" and m5.loc["ABC", "support_level"] == "PARTIAL"
    # FULL needs the verified pseudo-field too, not only the recorded review.
    unverified = algorithm_coverage("m5", EVERY_FIELD - {"consistent_unit"}).set_index("algorithm")
    assert unverified.loc["Demand Forecast", "support_level"] == "PARTIAL" and "consistent_unit" in unverified.loc["Demand Forecast", "full_gate_blockers"]
    for dataset in ("m5", "favorita", "freshretailnet", "kamp"):
        coverage = algorithm_coverage(dataset, EVERY_FIELD | {"consistent_unit"}).set_index("algorithm")
        assert coverage.loc[ROUTING, "support_level"].eq("UNSUPPORTED").all(), dataset
        assert coverage.loc[["Turnover", "Disposal Risk", "Safety Stock", "EOQ", "Store/Product Matching"], "support_level"].eq("UNSUPPORTED").all(), dataset
        if dataset != "m5":
            assert not coverage.support_level.eq("FULL").any(), dataset
    frn = algorithm_coverage("freshretailnet", EVERY_FIELD | {"consistent_unit"}).set_index("algorithm")
    assert frn.loc["Demand Forecast", "support_level"] == "PARTIAL" and "normalised" in frn.loc["Demand Forecast", "full_gate_blockers"]


def test_bulk_key_mode_matches_counter_mode(tmp_path):
    frame = canonicalize(base_frame(pd.DataFrame({"q": [1, 2, 3]}), "t", "raw/t.csv").assign(
        date="2026-01-01", location_id="L", product_id=["P", "P", "Q"], sales_qty=[1, 2, 3], unit="item", unit_status="DIRECT"), "demand_series")
    reports = {}
    for mode in ("counter", "bulk"):
        e = Exporter(tmp_path / mode, "m5")
        if mode == "bulk":
            e.bulk_key_tables.add("demand_series")
        e.write("demand_series", frame, masters=False)
        e.write("demand_series", frame.iloc[[2]], masters=False)
        reports[mode] = (e.finish(), read(e, "demand_series"))
    (counter, a), (bulk, b) = reports["counter"], reports["bulk"]
    assert counter["flags"]["demand_series"] == bulk["flags"]["demand_series"]
    assert a.validation_flags.map(lambda s: sorted(s.split("|"))).tolist() == b.validation_flags.map(lambda s: sorted(s.split("|"))).tolist()
    assert bulk["semantic_checks"]["key_uniqueness"]["demand_series"]["keys_repeated"] == 2


def reference_finalize(frame):
    # Per-row formulation used before 1.2.0.
    flags = frame["validation_flags"].fillna("").astype(str)
    split = flags.map(lambda s: [f for f in s.split("|") if f])
    order = {code: i for i, code in enumerate(QUALITY_CODES)}
    codes = split.map(lambda fs: sorted({quality_code(f) for f in fs}, key=order.__getitem__))
    return pd.DataFrame({"validation_flags": flags.map(lambda s: "|".join(dict.fromkeys(f for f in s.split("|") if f))),
                         "quality_flags": codes.map("|".join), "quality_status": codes.map(lambda c: c[0] if c else "VALID"),
                         "analysis_eligible": split.map(lambda fs: not any(is_blocking(f) for f in fs)).map({True: "true", False: "false"})})


def test_finalize_quality_matches_per_row_reference():
    values = ["", None, "unit_unspecified", "negative:order_qty|unit_unspecified|negative:order_qty", "float_residue_negative:inventory_qty",
              "documented_return:sales_qty|negative:sales_qty", "|weekly_price_absent|", "product_identity_versioned|duplicate_key|missing:target_id"]
    frame = pd.DataFrame({"validation_flags": values * 3})
    expected = reference_finalize(frame.copy())
    actual = finalize_quality(frame.copy())
    for col in expected:
        assert actual[col].astype(str).tolist() == expected[col].astype(str).tolist(), col


def truth(value):
    return str(value) == "True"


def test_validation_matrix_is_derived_from_outputs(tmp_path):
    write_m5(tmp_path)
    write_kamp(tmp_path)
    for dataset, runner in (("m5", run_m5), ("kamp", run_kamp)):
        run(tmp_path, dataset, runner)
    matrix = pd.read_csv(write_validation_matrix(tmp_path), encoding="utf-8-sig").set_index("dataset")
    required = ["real_or_synthetic", "date_range", "rows", "products", "locations", "has_inventory", "has_sales", "has_demand", "has_price", "has_promotion",
                "has_flow", "has_source_target", "has_cost", "has_capacity", "forecast_ready", "inventory_ready", "network_ready", "dqn_ready", "notes"]
    assert set(required) <= set(matrix.columns) and set(matrix.index) == set(DATASETS)
    m5 = matrix.loc["m5"]
    assert m5.rows == 9 and m5.products == 2 and m5.locations == 4 and m5.date_range == "2011-01-29..2011-01-31"
    assert truth(m5.has_sales) and truth(m5.has_price) and not truth(m5.has_inventory) and not truth(m5.has_source_target)
    assert m5.forecast_ready == "FULL" and m5.dqn_ready == "UNSUPPORTED" and m5.network_ready == "UNSUPPORTED"
    kamp = matrix.loc["kamp"]
    assert truth(kamp.has_flow) and not truth(kamp.has_inventory) and not truth(kamp.has_sales) and kamp.locations == 0
    assert matrix.loc["suhyup", "notes"] == "not converted to canonical"
