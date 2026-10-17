from dataclasses import replace
import copy
import io
import pytest
from services.candidate_input import read_uploads, prepare_input, run_prepared
from services.candidate_input_examples import csv_test_files, mixed_test_metadata


def prepared(change=None):
    b = read_uploads(csv_test_files())
    if change: change(b)
    return prepare_input(b, mixed_test_metadata())


def change_column(table, column, value):
    def change(bundle):
        frame = bundle.tables[table].copy()
        frame[column] = value
        bundle.tables[table] = frame
    return change


def test_csv_to_joint_economics_all_products():
    p = prepared()
    assert p.ready, p.issues
    r = run_prepared(p)
    assert r["status"] == "OPTIMAL", r
    assert r["expected_net_value"] == pytest.approx(74)
    assert sum(m["quantity"] for m in r["transfers"]) == 4
    assert not r["production_approved"] and r["actual_savings"] is None
    assert r["input_data_class"] == "TEST"
    assert len(r["input_file_hashes"]) == 5


@pytest.mark.parametrize("table,column,value", [
    ("economic_stock_policy", "holding_cost_per_unit_day", ""),
    ("economic_stock_policy", "holding_cost_per_unit_day", -1),
    ("economic_stock_policy", "quantum", 0),
    ("economic_stock_policy", "quantity_unit", "BOX"),
    ("economic_route_policy", "route_approved", False),
    ("economic_route_policy", "transit_days", 0.5),
    ("economic_route_policy", "capacity_unit", "M3"),
    ("economic_route_policy", "valid_to", "2026-10-16"),
    ("economic_sales_policy", "demand_basis", "ACTUAL_SALES"),
    ("economic_sales_policy", "unit_price", float("inf")),
    ("economic_location_policy", "currency", "USD"),
    ("inventory_snapshot", "unit_status", "UNKNOWN"),
    ("inventory_snapshot", "analysis_eligible", "false"),
])
def test_invalid_input_never_solved(table,column,value):
    p = prepared(change_column(table, column, value))
    assert not p.ready
    assert run_prepared(p)["status"] == "INPUT_REQUIRED"


def test_unknown_reservations_not_zero():
    p = prepared(change_column("inventory_snapshot", "reserved_qty", ""))
    assert not p.ready or run_prepared(p)["status"] == "INPUT_REQUIRED"


def test_explicit_zero_is_accepted():
    assert prepared().ready


def test_duplicate_cost_rows_rejected():
    import pandas as pd
    p = prepared(lambda b: b.tables.__setitem__("economic_stock_policy", pd.concat([b.tables["economic_stock_policy"]]*2)))
    assert not p.ready


def test_real_mode_does_not_promote_test_demand():
    b = read_uploads(csv_test_files())
    p = prepare_input(b, {**mixed_test_metadata(), "data_class": "REAL"})
    assert not p.ready


def test_no_permission_or_unknown_mode():
    for extra in ({"owner_authorized": False}, {"data_class": "ACTUAL_PRODUCTION"}):
        assert not prepare_input(read_uploads(csv_test_files()), {**mixed_test_metadata(), **extra}).ready


def test_ids_keep_leading_zeros():
    p = prepared(lambda b: None)
    b = read_uploads({"inventory_snapshot.csv": b"location_id,product_id\n001,002\n"})
    assert b.tables["inventory_snapshot"].iloc[0]["location_id"] == "001"


def test_duplicate_headers_and_invalid_zip():
    assert read_uploads({"inventory_snapshot.csv": b"x,x\n1,2"}).issues
    assert read_uploads({"bad.xlsx": b"not a zip"}).issues


def test_previous_result_is_immutable_on_bad_input():
    p = prepared(); result = run_prepared(p); before = copy.deepcopy(result)
    bad = prepared(change_column("economic_stock_policy", "holding_cost_per_unit_day", ""))
    assert not bad.ready
    assert result == before


def test_uploaded_result_history_preserves_test_class(tmp_path, monkeypatch):
    import pages.candidate_review as page
    monkeypatch.setattr(page, "HISTORY_DIR", tmp_path)
    p = prepared(); r = run_prepared(p)
    import json
    path = page._save_review(p.problem, r, "TEST 업로드")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["mode"] == "TEST" and not saved["production_action_applied"]
    assert saved["products"][0]["계획 이동량"] == 4


def test_two_leg_dc_route_counts_goods_once():
    import pandas as pd
    from pages.candidate_review import _product_frame
    b = read_uploads(csv_test_files())
    route = b.tables["economic_route_policy"].iloc[0].to_dict()
    b.tables["economic_route_policy"] = pd.DataFrame([
        {**route, "route_id": "S_DC", "target_id": "DC", "departure_periods": "0"},
        {**route, "route_id": "DC_T", "source_id": "DC", "departure_periods": "1"}])
    loc = b.tables["economic_location_policy"].iloc[0].to_dict()
    b.tables["economic_location_policy"] = pd.concat([b.tables["economic_location_policy"], pd.DataFrame([{**loc,"location_id":"DC"}])], ignore_index=True)
    sales = b.tables["economic_sales_policy"].copy()
    sales.loc[sales["location_id"]=="T", "period"] = "2"
    b.tables["economic_sales_policy"] = sales
    p = prepare_input(b, {**mixed_test_metadata(), "horizon_days": 3})
    assert p.ready,p.issues
    r = run_prepared(p)
    assert r["status"]=="OPTIMAL",r
    assert sum(m["quantity"] for m in r["transfers"]) == 8
    assert _product_frame(p.problem,r).iloc[0]["계획 이동량"] == 4
    assert r["transport_cost"]==4


def test_scheduled_inbound_input_reaches_model():
    import pandas as pd
    b = read_uploads(csv_test_files())
    inv = b.tables["inventory_snapshot"].copy(); inv["confirmed_inbound_qty"]="2"; b.tables["inventory_snapshot"]=inv
    b.tables["economic_flow_policy"]=pd.DataFrame([{"event_id":"in1","lot_id":"lot_A","location_id":"S",
        "period":0,"quantity":2,"kind":"CONFIRMED_INBOUND","quantity_unit":"EA"}])
    p=prepare_input(b,mixed_test_metadata());assert p.ready,p.issues
    r=run_prepared(p);assert r["status"]=="OPTIMAL",r
    assert r["production_approved"] is False
