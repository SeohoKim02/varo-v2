"""Integrity rules for canonical 1.1.0; tiny explicit fixtures, no production data."""
import hashlib
import json

import pandas as pd
import pytest

from services.canonical_data_pipeline import Exporter, adjacent_identical_positions, algorithm_coverage, run_jangbogo, stream_file
from services.canonical_schema import canonicalize, quality_code, schema_document
from services.real_data_adapters import (LOGISALL_COLLAPSE, NFQS_FLOAT_RESIDUE_TOLERANCE_TON, adapt_aihub, adapt_aihub_measurements,
    adapt_jangbogo, adapt_logisall, adapt_nfqs, adapt_suhyup, base_frame, classify_name_versions, collapse_source_records,
    nfqs_flow_identity_diff, nfqs_panel_coverage, pack_spec_tokens)
from tests.test_canonical_data import suhyup_raw


def read(e, table):
    return pd.read_parquet(e.processed / f"canonical_{table}.parquet")


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def sales_raw():
    # Two sub-zone records share (date, product, zone); one is fully identical to another.
    return pd.DataFrame({"BASE_YMD": ["20221101", "20221101", "20221101", "20221102"], "AGFD_PDLT_NM": ["배", "배", "배", "배"],
                         "AGFD_SLPL_ZIP": ["10000", "10000", "10000", "10000"], "SLE_QY": ["5", "5", "2.5", "7"]}, dtype="string")


def test_unknown_unit_is_not_guessed():
    _, logisall, _ = adapt_logisall(sales_raw(), "raw/s.csv", "sales")
    _, jangbogo, _, _ = adapt_jangbogo(pd.DataFrame({"cfmtn_ym": ["202304"], "fdmt_pdlt_code": ["0001"], "fdmt_pdlt_nm": ["딸기 1KG"], "prca_dmnd_qyt": ["3"]}), "raw/j.csv", "orders")
    _, suhyup, _ = adapt_suhyup(suhyup_raw(), "raw/stock.csv", "stock")
    for out in (logisall, jangbogo, suhyup):
        assert out.unit.isna().all() and out.unit_status.eq("UNKNOWN").all()
        assert out.validation_flags.str.contains("unit_unspecified").all()
        assert out.quality_flags.str.contains("UNKNOWN_UNIT").all()
    # A pack size printed in a product name is not the quantity unit.
    assert pd.isna(jangbogo.unit.iloc[0])
    # The parallel Suhyup kg column is a header-stated unit, independent of the count unit.
    assert suhyup.weight_unit.iloc[0] == "kg" and suhyup.weight_unit_status.iloc[0] == "DIRECT"


def test_unit_without_evidence_or_inconsistent_status_is_flagged():
    raw = pd.DataFrame({"q": [1, 2, 3]})
    frame = base_frame(raw, "test", "raw/t.csv")
    frame["inventory_qty"], frame["snapshot_date"], frame["location_id"], frame["product_id"] = raw.q, "2026-01-01", "L", "P"
    frame["unit"] = ["kg", "kg", None]
    frame["unit_status"] = [None, "UNKNOWN", "DIRECT"]
    out = canonicalize(frame, "inventory_snapshot")
    assert "unit_evidence_missing:unit" in out.validation_flags.iloc[0]
    assert "unit_status_inconsistent:unit" in out.validation_flags.iloc[1]
    assert "unit_status_inconsistent:unit" in out.validation_flags.iloc[2]
    assert out.unit.iloc[0] == "kg" and pd.isna(out.unit.iloc[2])
    assert out.analysis_eligible.eq("false").all()


def test_missing_is_not_zero_through_collapse_and_panel():
    raw = pd.DataFrame({"BASE_YMD": ["20221101"] * 3, "AGFD_PDLT_NM": ["배"] * 3, "AGFD_SLPL_ZIP": ["10000"] * 3, "SLE_QY": [None, None, "4"]}, dtype="string")
    collapsed = collapse_source_records(raw, *LOGISALL_COLLAPSE["sales"])
    _, out, _ = adapt_logisall(collapsed, "raw/s.csv", "sales")
    # NULL quantities are never merged into (or summed as) zero.
    assert len(out) == 3 and out.sales_qty.isna().sum() == 2 and out.sales_qty.dropna().tolist() == [4]
    assert out.source_record_count.tolist() == [1, 1, 1]
    panel = nfqs_panel_coverage({"2021Q4": {"011": "고등어", "161": "병어"}, "2022Q1": {"011": "고등어"}})
    absent = panel[(panel.product_id == "161") & (panel.period == "2022Q1")].iloc[0]
    assert absent.status == "NOT_REPORTED" and pd.isna(absent.product_name)
    assert "inventory_qty" not in panel


def test_logisall_duplicate_policy_sums_subgrain_records_with_lineage():
    collapsed = collapse_source_records(sales_raw(), *LOGISALL_COLLAPSE["sales"])
    table, out, _ = adapt_logisall(collapsed, "raw/s.csv", "sales")
    assert table == "demand_series" and len(out) == 2
    first = out.iloc[0]
    # Identical records are distinct sub-zone records: summed, never dropped.
    assert first.sales_qty == 12.5 and first.source_record_count == 3 and first.source_row_id == "1|2|3"
    assert out.iloc[1].source_row_id == "4" and out.iloc[1].sales_qty == 7
    assert not out.validation_flags.str.contains("duplicate").any()


def test_adjacent_identical_records_are_kept_and_marked(tmp_path):
    raw = sales_raw()
    # The first record has nothing above it; only record 2 repeats record 1.
    assert adjacent_identical_positions(raw) == [2] and adjacent_identical_positions(raw.iloc[:1]) == []
    collapsed = collapse_source_records(raw, *LOGISALL_COLLAPSE["sales"])
    _, out, measures = adapt_logisall(collapsed, "raw/s.csv", "sales")
    e = Exporter(tmp_path, "logisall")
    e.write("demand_series", out, raw, measures)
    e.post_flags["demand_series"].append(("raw/s.csv", [2], "adjacent_identical_record"))
    e.finish()
    rows = read(e, "demand_series")
    assert rows.sales_qty.tolist() == [12.5, 7] and rows.source_row_id.tolist() == ["1|2|3", "4"]
    assert "adjacent_identical_record" in rows.validation_flags.iloc[0].split("|") and "adjacent" not in rows.validation_flags.iloc[1]
    assert rows.quality_status.iloc[0] == "DUPLICATE_OBSERVATION"


def test_collapse_never_hides_differing_attributes_or_flagged_values():
    raw = pd.DataFrame({"k": ["a", "a"], "label": ["x", "y"], "q": ["1", "2"]}, dtype="string")
    with pytest.raises(ValueError, match="cover all non-measure"):
        collapse_source_records(raw, ["k"], ["q"])
    signed = pd.DataFrame({"k": ["a", "a", "a"], "q": ["5", "-3", "bad"]}, dtype="string")
    out = collapse_source_records(signed, ["k"], ["q"])
    assert len(out) == 3 and out._source_record_count.tolist() == [1, 1, 1]


def test_logisall_missing_destination_not_restored_or_merged(tmp_path):
    raw = pd.DataFrame({"BASE_YMD": ["20220801"] * 3, "AGFD_PDLT_NM": ["사과"] * 3, "FRWAR_ZIP": ["40000"] * 3,
                        "ARVL_ZIP": [None, None, "50000"], "DSBN_QY": ["3200", "100", "7"]}, dtype="string")
    collapsed = collapse_source_records(raw, *LOGISALL_COLLAPSE["network"])
    _, out, _ = adapt_logisall(collapsed, "raw/d.csv", "network")
    missing = out[out.target_id.isna()]
    # Two unknown destinations are not assumed to be the same place.
    assert len(missing) == 2 and missing.source_row_id.tolist() == ["1", "2"]
    assert missing.validation_flags.str.contains("missing:target_id").all()
    assert missing.quality_status.eq("MISSING_KEY").all() and missing.analysis_eligible.eq("false").all()
    e = Exporter(tmp_path, "logisall")
    e.write("transfer_network", out, collapsed, {"shipment_qty": "DSBN_QY"})
    e.finish()
    assert set(read(e, "location_master").location_id) == {"40000", "50000"}


def test_jangbogo_signed_quantity_preserved(tmp_path):
    raw = pd.DataFrame({"cfmtn_ym": ["202309", "202309"], "fdmt_pdlt_code": ["1121", "1134"], "fdmt_pdlt_nm": ["고등어구이/국산", "우럭구이/국산"],
                        "brcd_info": ["NULL", "NULL"], "prca_dmnd_qyt": ["-3987.00", "12.00"]}, dtype="string")
    table, out, measures, work = adapt_jangbogo(raw, "raw/buy.csv", "orders")
    assert out.order_qty.tolist() == [-3987.0, 12.0]
    assert "negative:order_qty" in out.validation_flags.iloc[0] and out.quality_flags.iloc[0].count("NEGATIVE_QUANTITY") == 1
    assert out.analysis_eligible.iloc[0] == "false"
    # No return/cancellation meaning is invented.
    assert out.returns_qty.isna().all() and out.inbound_qty.isna().all()
    e = Exporter(tmp_path, "jangbogo")
    e.write(table, out, work, measures)
    report = e.finish()
    assert report["quantity_conservation"][0]["source_sum"] == report["quantity_conservation"][0]["canonical_sum"] == -3975.0


def test_jangbogo_negative_count_matches_in_scope_source_rows(tmp_path):
    raw_dir = tmp_path / "23_KADX_JANGBOGO/raw/kadx_full_data"
    spot = {"CFMTN_YM": ["202309"], "WRHS_CODE": ["3"], "WRHS_NM": ["월배점"], "ZIP": ["41000"], "ZIP_ADDR": ["대구"],
            "FDMT_PDLT_LGLS_CODE": ["1001"], "FDMT_PDLT_LGLS_NM": ["가공상품"]}
    write_csv(raw_dir / "TB_SPOT_CL_MTH_BUY_YM_20231010.csv", pd.DataFrame({**spot, "PRCA_DMND_QYT": ["10"]}))
    write_csv(raw_dir / "TB_SPOT_CL_MTH_SALES_YM_20231010.csv", pd.DataFrame({**spot, "TOT_SLE_QYT": ["7"]}))
    for name in ["TB_LGTC_PDLT_SALES_YMD_20230301.csv", "TB_LGTC_PDLT_SALES_YMD_20231010.csv"]:
        write_csv(raw_dir / name, pd.DataFrame({"CFMTN_YMD": ["20230901"], "WRHS_CODE": ["3"], "WRHS_NM": ["월배점"], "FDMT_PDLT_CODE": ["0001"],
                                                "FDMT_PDLT_NM": ["무우"], "FDMT_PDLT_LGLS_CODE": ["1002"], "FDMT_PDLT_LGLS_NM": ["농산물"]}))

    def buy(codes, qty):
        return pd.DataFrame({"CFMTN_YM": ["202309"] * len(qty), "FDMT_PDLT_CODE": codes, "FDMT_PDLT_NM": [f"상품{c}" for c in codes],
                             "BRCD_INFO": ["NULL"] * len(qty), "PRCA_DMND_QYT": qty})

    write_csv(raw_dir / "TB_PDLT_BUY_YM_231010.csv", buy(["1121", "1134", "0001"], ["-3987.00", "-3993.00", "12.00"]))
    # An older purchase-request partition is read as evidence only, never converted.
    write_csv(raw_dir / "TB_PDLT_BUY_YM_20221031.csv", buy(["0916", "72511", "87803"], ["-996", "-1", "-1"]))
    e = Exporter(tmp_path, "jangbogo")
    run_jangbogo(e)
    report = e.finish()
    flow = read(e, "inventory_flow")
    negative = flow[flow.order_qty.lt(0)]
    evidence = report["semantic_checks"]["negative_quantity_investigation"]
    # One count per in-scope source row, whichever flag column is counted; no out-of-scope partition is added.
    in_scope = sorted((r["source_row_id"], r["order_qty"]) for r in evidence["in_scope_negative_rows"])
    assert sorted(zip(negative.source_row_id, negative.order_qty)) == in_scope == [("1", -3987.0), ("2", -3993.0)]
    quality = report["quality"]["inventory_flow"]
    assert report["flags"]["inventory_flow"]["negative:order_qty"] == quality["quality_flags"]["NEGATIVE_QUANTITY"] == quality["quality_status"]["NEGATIVE_QUANTITY"] == 2
    assert negative.source_file.str.endswith("TB_PDLT_BUY_YM_231010.csv").all() and negative.analysis_eligible.eq("false").all()
    assert sum(len(p["negative_rows"]) for p in evidence["all_partitions"]) == 5
    assert not flow.source_file.str.contains("TB_PDLT_BUY_YM_20221031").any()


def test_jangbogo_barcode_grain_and_missing_subkey(tmp_path):
    raw = pd.DataFrame({"cfmtn_ym": ["202304"] * 4, "fdmt_pdlt_code": ["0234"] * 4, "fdmt_pdlt_nm": ["전복 /국산"] * 4,
                        "brcd_info": ["880001", "880002", "NULL", None], "prca_dmnd_qyt": ["40000", "388", "5", "6"]}, dtype="string")
    table, out, measures, work = adapt_jangbogo(raw, "raw/buy.csv", "orders")
    assert out.product_variant_id.tolist()[:2] == ["880001", "880002"] and out.product_variant_id.iloc[2:].isna().all()
    e = Exporter(tmp_path, "jangbogo")
    e.write(table, out, work, measures)
    e.finish()
    flow = read(e, "inventory_flow")
    # Barcode variants are separate facts; the two missing-barcode records are both marked, none summed.
    assert len(flow) == 4 and flow.order_qty.tolist() == [40000, 388, 5, 6]
    assert not flow.validation_flags.iloc[:2].str.contains("repeated").any()
    assert flow.validation_flags.iloc[2:].str.contains("repeated_key_missing_subkey").all()


def test_jangbogo_product_identity_collision_is_versioned(tmp_path):
    raw = pd.DataFrame({"cfmtn_ymd": ["20200229", "20220208", "20230414"], "wrhs_code": ["007"] * 3, "wrhs_nm": ["센터"] * 3,
                        "fdmt_pdlt_code": ["22949"] * 3, "fdmt_pdlt_nm": ["딸기(3번) 1팩", "딸기(3번) 1KG/국산", "딸기(3번) 1KG/국산"]}, dtype="string")
    table, out, measures, work = adapt_jangbogo(raw, "raw/lgtc.csv", "observation")
    e = Exporter(tmp_path, "jangbogo")
    e.write(table, out, work, measures)
    report = e.finish()
    versions = read(e, "product_identity_version").sort_values("version_seq")
    assert versions.product_id.tolist() == ["22949", "22949"]
    assert versions.product_name.tolist() == ["딸기(3번) 1팩", "딸기(3번) 1KG/국산"]
    assert versions.valid_from.tolist() == ["2020-02-29", "2022-02-08"] and versions.valid_to.tolist() == ["2020-02-29", "2023-04-14"]
    assert versions.temporal_relation.eq("SEQUENTIAL").all() and versions.name_change_class.eq("PACK_SPEC_CHANGED").all()
    facts = read(e, "location_product_observation")
    # product_id is never renumbered; rows keep their own names and stay usable per version.
    assert facts.product_id.eq("22949").all() and facts.product_name.tolist() == raw.fdmt_pdlt_nm.tolist()
    assert facts.validation_flags.str.contains("product_identity_versioned").all()
    master = read(e, "product_master")
    assert master.quality_status.iloc[0] == "AMBIGUOUS_PRODUCT"
    assert report["semantic_checks"]["product_master_attribute_conflicts"]["conflicting_name_keys"] == 1


def test_name_version_classification_is_deterministic():
    assert pack_spec_tokens("청포도(샤인머스컷) 2KG/국산") == pack_spec_tokens("청포도(샤인머스켓) 2KG/국산") == ("2KG",)
    assert pack_spec_tokens("공업용랩(스트레치) 20T*50CM*350M") == pack_spec_tokens("공업용랩(스트레치) 20T*50CMx350M")
    assert pack_spec_tokens("배선물세트(6-10과) 1BOX") != pack_spec_tokens("배선물세트(8-10과) 1BOX")
    overlapping = [{"product_name": "a 1KG", "valid_from": "2020-01-01", "valid_to": "2020-12-31"},
                   {"product_name": "b 1KG", "valid_from": "2020-06-01", "valid_to": "2021-01-31"}]
    assert classify_name_versions(overlapping) == ("CONCURRENT", "LABEL_CHANGED_SAME_PACK_TOKENS")


def test_cross_release_conflict_marks_every_member(tmp_path):
    e = Exporter(tmp_path, "jangbogo")
    for name, qty in [("rel_a.csv", "5220"), ("rel_b.csv", "90307")]:
        raw = pd.DataFrame({"CFMTN_YM": ["201805"], "WRHS_CODE": ["3"], "WRHS_NM": ["월배점"], "FDMT_PDLT_LGLS_CODE": ["1001.0"],
                            "FDMT_PDLT_LGLS_NM": ["가공상품"], "TOT_SLE_QYT": [qty]}, dtype="string")
        table, out, measures, work = adapt_jangbogo(raw, f"raw/{name}", "sales")
        e.write(table, out, work, measures)
    report = e.finish()
    sales = read(e, "demand_series")
    assert sales.sales_qty.tolist() == [5220, 90307]
    assert sales.validation_flags.str.contains("cross_release_conflict").all() and sales.analysis_eligible.eq("false").all()
    assert report["flags"]["demand_series"]["duplicate_key"] == 1


def nfqs_item(**values):
    item = {"ICEGDFG": "020", "CODEKNM": "명태", "ICEALIST": 10.0, "ICEAIN": 5.0, "ICEAOUT": 3.0, "INVENTOTAL": 12.0, "YEOSU": 0.0, "BUSAN": 12.0}
    item.update(values)
    return item


def test_nfqs_tiny_negative_tolerance():
    residue = -3.885780586188048e-16
    assert abs(residue) < NFQS_FLOAT_RESIDUE_TOLERANCE_TON
    out = adapt_nfqs([nfqs_item(YEOSU=residue), nfqs_item(ICEGDFG="030", CODEKNM="조기", YEOSU=-0.5, BUSAN=12.5)], "raw/q.json", 2025, 1)
    tiny = out[(out.product_id == "020") & (out.location_id == "YEOSU")].iloc[0]
    real = out[(out.product_id == "030") & (out.location_id == "YEOSU")].iloc[0]
    # Preserved bit-for-bit; never clamped to zero.
    assert tiny.inventory_qty == residue and tiny.inventory_qty != 0
    assert tiny.validation_flags.split("|") == ["float_residue_negative:inventory_qty"]
    assert tiny.analysis_eligible == "true" and tiny.quality_status == "SOURCE_ANOMALY"
    assert real.inventory_qty == -0.5 and "negative:inventory_qty" in real.validation_flags.split("|") and real.analysis_eligible == "false"


def test_nfqs_quarterly_snapshot_semantics():
    # Opening stock (ICEALIST) differs from any previous close: not an error, no cross-quarter flow is assumed.
    consistent = nfqs_item(ICEALIST=999.0, ICEAIN=5.0, ICEAOUT=992.0)
    broken = nfqs_item(ICEGDFG="011", CODEKNM="고등어", ICEALIST=10.0, ICEAIN=5.0, ICEAOUT=3.0, INVENTOTAL=13.0, BUSAN=13.0)
    assert nfqs_flow_identity_diff(consistent) == 0 and nfqs_flow_identity_diff(broken) == -1
    out = adapt_nfqs([consistent, broken], "raw/q.json", 2022, 1)
    assert out.date_grain.eq("quarterly").all() and out.snapshot_date.eq("2022-03-31").all()
    assert set(out.columns).isdisjoint({"inbound_qty", "outbound_qty"})
    flagged = out[out.validation_flags.str.contains("source_flow_identity_mismatch")]
    assert flagged[["product_id", "location_id"]].values.tolist() == [["011", "INVENTOTAL"]]
    # The stock value itself is reported, consistent with its regions, and stays usable.
    assert flagged.inventory_qty.iloc[0] == 13.0 and flagged.analysis_eligible.iloc[0] == "true"
    assert not out[out.product_id == "020"].validation_flags.str.contains("mismatch").any()


def test_nfqs_unbalanced_product_panel():
    q4 = [nfqs_item(), nfqs_item(ICEGDFG="161", CODEKNM="병어")]
    q1 = [nfqs_item()]
    out = adapt_nfqs(q1, "raw/2022Q1.json", 2022, 1)
    assert "161" not in set(out.product_id)
    panel = nfqs_panel_coverage({"2021Q4": {r["ICEGDFG"]: r["CODEKNM"] for r in q4}, "2022Q1": {r["ICEGDFG"]: r["CODEKNM"] for r in q1}})
    status = panel.set_index(["product_id", "period"]).status
    assert status[("161", "2021Q4")] == "REPORTED" and status[("161", "2022Q1")] == "NOT_REPORTED"
    assert panel[panel.product_id == "161"].last_reported.iloc[0] == "2021Q4"


def measurement_raw():
    return pd.DataFrame({"flow_dir": ["01_입고물품", "02_출고물품"], "category_l1": ["01_가공식품"] * 2, "category_l2": ["06_통조림,병"] * 2,
                         "kan_code": ["01060101"] * 2, "barcode": ["038900002053"] * 2, "product_name": ["돌 파인애플 슬라이스 3062g", "Dole 파인애플 3062g"],
                         "length_cm": ["15.3", "20.0"], "width_cm": ["15.4", "20.0"], "height_cm": ["17.7", "20.0"], "weight_kg": ["3.434", "3.626"],
                         "source_member": ["물품측정데이터/01_입고물품/x.csv", "물품측정데이터/02_출고물품/x.csv"]}, dtype="string")


def test_aihub_repeated_measurement_is_not_a_duplicate_product(tmp_path):
    raw = measurement_raw()
    out, measures = adapt_aihub_measurements(raw, "raw/Other.zip")
    assert out.measurement_context.tolist() == ["inbound_item", "outbound_item"]
    assert out.weight.tolist() == [3.434, 3.626] and out.dimension_unit.eq("cm").all() and out.weight_unit_status.eq("SOURCE_METADATA").all()
    e = Exporter(tmp_path, "aihub")
    e.write("product_measurement", out, raw, measures)
    report = e.finish()
    measurements, master = read(e, "product_measurement"), read(e, "product_master")
    assert len(measurements) == 2 and measurements.analysis_eligible.eq("true").all()
    assert not measurements.validation_flags.str.contains("duplicate").any()
    # One product identity; no weight is picked from one handling context.
    assert len(master) == 1 and pd.isna(master.unit_weight.iloc[0])
    assert master.validation_flags.iloc[0] == "product_label_variants" and master.analysis_eligible.iloc[0] == "true"
    assert all(c["passed"] for c in report["quantity_conservation"]) and len(report["quantity_conservation"]) == 4


def test_aihub_unknown_location_is_not_an_entity(tmp_path):
    raw = pd.DataFrame([[pd.Timestamp("2024-08-01"), 10, 2, 20, *([0] * 24), 10.0, 0.5]])
    e = Exporter(tmp_path, "aihub")
    for table, frame, measures, work in adapt_aihub(raw, "raw/flow.xlsx", "flow"):
        assert frame.location_id.isna().all() and frame.quality_status.eq("UNKNOWN_LOCATION").all()
        e.write(table, frame, work, measures)
    e.finish()
    assert not (e.processed / "canonical_location_master.parquet").exists()
    assert not (e.processed / "canonical_transfer_network.parquet").exists()
    coverage = algorithm_coverage("aihub", {"inventory_qty", "outbound_qty", "inbound_qty", "date"}).set_index("algorithm")
    assert coverage.loc[["Greedy", "VHS", "Pareto", "DQN", "MILP", "Varo Final"], "support_level"].eq("UNSUPPORTED").all()


def test_provenance_retained_after_cleaning(tmp_path):
    path = write_csv(tmp_path / "22_KADX_LOGISALL/raw/kadx_full_data/TB_MRI_SLE_CW-x.csv", sales_raw())
    e = Exporter(tmp_path, "logisall")
    stream_file(e, path, adapt_logisall, "sales", collapse=LOGISALL_COLLAPSE["sales"])
    e.finish()
    out = read(e, "demand_series")
    assert out.source_file.eq("22_KADX_LOGISALL/raw/kadx_full_data/TB_MRI_SLE_CW-x.csv").all()
    assert out.source_row_id.tolist() == ["1|2|3", "4"] and out.transform_version.eq("1.1.0").all()
    ids = out.source_row_id.str.split("|").explode().astype(int)
    assert sorted(ids) == [1, 2, 3, 4]
    mapping = json.loads((e.results / "canonical_mapping.json").read_text(encoding="utf-8"))
    assert mapping["mappings"][0]["fields"]["source_record_count"]["coverage"] == "DERIVED"


def test_quantity_conservation_after_cleaning(tmp_path):
    path = write_csv(tmp_path / "22_KADX_LOGISALL/raw/kadx_full_data/TB_MRI_SLE_CW-x.csv", sales_raw())
    e = Exporter(tmp_path, "logisall")
    stream_file(e, path, adapt_logisall, "sales", collapse=LOGISALL_COLLAPSE["sales"])
    report = e.finish()
    quantity = next(c for c in report["quantity_conservation"] if c["field"] == "sales_qty")
    records = next(c for c in report["quantity_conservation"] if c["field"] == "source_record_count")
    assert quantity["source_sum"] == quantity["canonical_sum"] == 19.5
    assert quantity["rows_source"] == quantity["source_records_represented"] == 4 and quantity["rows_canonical"] == 2
    assert records["distinct_source_row_ids"] == 4 and report["all_quantity_checks_passed"]
    # A canonical frame that drops a record cannot pass.
    collapsed = collapse_source_records(sales_raw(), *LOGISALL_COLLAPSE["sales"])
    _, out, measures = adapt_logisall(collapsed.iloc[:1], "raw/s.csv", "sales")
    with pytest.raises(AssertionError, match="Record conservation"):
        Exporter(tmp_path / "other", "logisall").write("demand_series", out, sales_raw(), measures)


def test_raw_source_untouched(tmp_path):
    path = write_csv(tmp_path / "22_KADX_LOGISALL/raw/kadx_full_data/TB_MRI_SLE_CW-x.csv", sales_raw())
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns, path.stat().st_size)
    e = Exporter(tmp_path, "logisall")
    stream_file(e, path, adapt_logisall, "sales", collapse=LOGISALL_COLLAPSE["sales"])
    e.finish()
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns, path.stat().st_size) == before
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert all(p.startswith(("22_KADX_LOGISALL/processed/canonical_", "22_KADX_LOGISALL/results/")) or p == path.relative_to(tmp_path).as_posix() for p in written)


def test_quality_codes_are_standard_and_ordered():
    doc = schema_document()
    assert {"product_measurement", "product_identity_version"} == set(doc["auxiliary_tables"])
    assert quality_code("missing:target_id") == "MISSING_KEY" and quality_code("missing:inventory_qty") == "MISSING_VALUE"
    assert quality_code("negative:order_qty") == "NEGATIVE_QUANTITY" and quality_code("aggregate_product_total") == "OUT_OF_SCOPE"
    assert quality_code("cross_release_conflict") == "DUPLICATE_OBSERVATION" and quality_code("location_unresolved") == "UNKNOWN_LOCATION"
    assert set(quality_code(f) for f in ["unit_unspecified", "product_identity_versioned", "benchmark_proxy_constraints", "invalid_date"]) == {
        "UNKNOWN_UNIT", "AMBIGUOUS_PRODUCT", "BENCHMARK_PROXY", "SOURCE_ANOMALY"}
    assert doc["quality_codes"][0] == "MISSING_KEY" and "VALID" not in doc["quality_codes"]
    out = canonicalize(base_frame(pd.DataFrame({"q": [1]}), "t", "raw/t.csv").assign(date="2026-01-01", location_id="L", validation_flags="product_identity_versioned|negative:order_qty"), "inventory_flow")
    assert out.quality_flags.iloc[0] == "NEGATIVE_QUANTITY|UNKNOWN_UNIT|AMBIGUOUS_PRODUCT" and out.quality_status.iloc[0] == "NEGATIVE_QUANTITY"


def test_coverage_is_never_full_without_observed_constraints():
    every_field = {"inventory_qty", "unit_cost", "sales_qty", "date", "recommended_qty", "source_surplus", "target_need", "transport_cost",
                   "distance_km", "currency", "location_id", "product_id", "expiry_date", "vhs_score", "feasibility_score"}
    for dataset in ["suhyup", "logisall", "jangbogo", "nfqs", "aihub"]:
        coverage = algorithm_coverage(dataset, every_field).set_index("algorithm")
        assert not coverage.support_level.eq("FULL").any(), dataset
    suhyup = algorithm_coverage("suhyup", every_field).set_index("algorithm")
    assert suhyup.loc[["Greedy", "VHS", "Pareto", "DQN", "MILP", "Varo Final"], "support_level"].eq("BENCHMARK_ONLY").all()
    assert "no observed source surplus" in suhyup.loc["Varo Final", "full_gate_blockers"]
