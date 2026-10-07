"""Reusable explicit inputs must reach the unchanged loss engine, never become actual observations."""
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from services import seller_business_profile as bp, seller_loss_inputs as sli, seller_loss_input_requirements as req
from services.seller_loss_engine import InputField, known
from tests.test_seller_loss_inputs import base_input, table, SCENARIO_A_ROWS, W

DATE = "2026-07-31"


def profile(rows):
    return bp.parse_profile(pd.DataFrame([{ "effective_date": "2026-07-01", "effective_until": "2026-07-31",
                                          "currency": "KRW", **r} for r in rows]))


def run(rows, base=None, inputs=None, date=DATE, **kwargs):
    return bp.evaluate_profile_decision(base or base_input(), profile(rows), table(inputs) if inputs else None,
                                       decision_date=date, **kwargs)


def merged(rows, **kwargs):
    return sli.merge_seller_inputs(kwargs.pop("base", base_input()), profile(rows), decision_date=kwargs.pop("date", DATE), **kwargs)


@pytest.mark.parametrize("rows", [None, SCENARIO_A_ROWS])
def test_no_profile_exact_backward_compatibility(rows):
    t = table(rows) if rows else None
    assert bp.evaluate_profile_decision(base_input(), None, t) == req.plan_input_requirements(base_input(), t)
    assert bp.evaluate_profile_decision(base_input(), bp.parse_profile(None), t) == req.plan_input_requirements(base_input(), t)


@pytest.mark.parametrize("scope,keys", [("GLOBAL", {}), ("STORE", {"store_id": "S1"}),
    ("PRODUCT", {"product_id": "P1"}), ("PRODUCT_STORE", {"product_id": "P1", "store_id": "S1"})])
@pytest.mark.parametrize("column,engine", [("holding_cost_per_unit_day", "source_holding_cost_per_unit_day"),
                                        ("disposal_cost_per_unit", "source_disposal_cost_per_unit")])
def test_cost_scopes(scope, keys, column, engine):
    m, _ = merged([{"scope": scope, **keys, column: 12}])
    assert getattr(m, engine).value == 12 and getattr(m, engine).provenance == "USER_INPUT"
    other, _ = merged([{"scope": scope, **keys, column: 12}], base=replace(base_input(), product_id="OTHER", source_store_id="OTHER"))
    assert getattr(other, engine).present == (scope == "GLOBAL")


def test_scope_precedence():
    rows = [{"scope": "GLOBAL", "holding_cost_per_unit_day": 1},
            {"scope": "STORE", "store_id": "S1", "holding_cost_per_unit_day": 2},
            {"scope": "PRODUCT", "product_id": "P1", "holding_cost_per_unit_day": 3},
            {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "holding_cost_per_unit_day": 4}]
    assert merged(rows)[0].source_holding_cost_per_unit_day.value == 4


def test_same_scope_conflict_no_first_last_choice():
    rows = [{"scope": "PRODUCT", "product_id": "P1", "normal_price": n} for n in (100, 200)]
    a, rec = merged(rows)
    b, _ = merged(list(reversed(rows)))
    assert not a.source_normal_price.present and not b.source_normal_price.present
    assert rec["conflicts"][0]["kind"] == "SELLER_INPUT_SAME_SCOPE"


@pytest.mark.parametrize("provenance", ["DIRECT_REAL", "DERIVED_REAL"])
def test_real_beats_profile(provenance):
    actual = known(300, provenance, "real price", currency="KRW", dataset=W)
    m, record = merged([{"scope": "PRODUCT", "product_id": "P1", "normal_price": 100}],
                       base=base_input(source_normal_price=actual))
    assert m.source_normal_price == actual
    assert any(c["kind"] == "REAL_VS_SELLER_INPUT" for c in record["conflicts"])


def test_decision_input_overrides_more_specific_profile():
    p = profile([{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "normal_price": 100}])
    t = table([{"scope": "PRODUCT", "product_id": "P1", "normal_price": 200}])
    m, rec = sli.merge_seller_inputs(base_input(), bp.combine_inputs(t, p), decision_date=DATE)
    assert m.source_normal_price.value == 200
    assert any(a.get("source") == bp.SHEET_KEY and a["effective_value"] == 200 for a in rec["audit"])


@pytest.mark.parametrize("date,expected", [("2026-06-30", None), ("2026-07-01", 100),
    ("2026-07-31", 100), ("2026-08-01", None), (None, None)])
def test_effective_date_boundaries(date, expected):
    assert merged([{"scope": "PRODUCT", "product_id": "P1", "normal_price": 100}], date=date)[0].source_normal_price.value == expected


def test_price_versions_do_not_rewrite_history_and_overlap_conflicts():
    rows = [{"scope": "PRODUCT", "product_id": "P1", "normal_price": 100},
            {"scope": "PRODUCT", "product_id": "P1", "normal_price": 200, "effective_date": "2026-08-01", "effective_until": "2026-08-31"}]
    assert merged(rows)[0].source_normal_price.value == 100
    assert merged(rows, date="2026-08-01")[0].source_normal_price.value == 200
    rows[1]["effective_date"] = "2026-07-30"
    assert not merged(rows)[0].source_normal_price.present


@pytest.mark.parametrize("field", ["daily_demand", "promotion_uplift"])
def test_dynamic_fields_never_frozen_global_or_undated(field):
    invalid = bp.parse_profile([{"scope": "GLOBAL", field: 3}])
    assert not any(c.column == field for c in invalid.cells)
    invalid = profile([{"scope": "GLOBAL", field: 3}])
    assert not any(c.column == field for c in invalid.cells)
    p = profile([{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", field: 3}])
    assert any(c.column == field and c.provenance == "USER_INPUT" for c in p.cells)


def test_shelf_life_and_sunk_cost_not_profile_requirements():
    p = profile([{"scope": "PRODUCT", "product_id": "P1", "remaining_shelf_life_days": 10, "unit_cost": 20}])
    assert not p.cells
    assert {e["code"] for e in p.validation["errors"]} == {"DECISION_BATCH_ONLY", "NOT_USED_BY_LOSS_COMPARISON"}


def test_route_profile_only_missing_actual_cost():
    rows = [{"scope": "ROUTE", "route_id": "R1", "transfer_cost": 30, "transit_time_days": 0.5}]
    assert merged(rows)[0].transfer_cost.value == 1800
    m, _ = merged(rows, base=base_input(transfer_cost=InputField(), transit_time_days=InputField()))
    assert m.transfer_cost.value == 30 and m.transit_time_days.value == 0.5
    assert m.transfer_cost.provenance == "USER_INPUT"


def test_discount_does_not_invent_uplift():
    m, _ = merged([{"scope": "PRODUCT", "product_id": "P1", "discount_rate": 0.1}])
    assert m.discount_rate.value == 0.1 and not m.promotion_uplift.present


def test_global_currency_is_explicit_and_conflict_not_defaulted():
    p = bp.parse_profile([{"scope": "GLOBAL", "currency": "USD"},
        {"scope": "PRODUCT", "product_id": "P1", "normal_price": 100, "effective_date": DATE, "effective_until": DATE}])
    assert p.cells[0].currency == "USD"
    assert p.cells[0].currency_basis == "seller_business_profile GLOBAL.currency"
    p = bp.parse_profile([{"scope": "GLOBAL", "currency": "USD"}, {"scope": "GLOBAL", "currency": "KRW"},
        {"scope": "PRODUCT", "product_id": "P1", "normal_price": 100, "effective_date": DATE, "effective_until": DATE}])
    assert not p.cells and any(e["code"] == "GLOBAL_CURRENCY_CONFLICT" for e in p.validation["errors"])


@pytest.mark.parametrize("bad", [17, "bad sheet", [{"scope": "PRODUCT", "product_id": "P1", "normal_price": 3,
    "effective_date": "bad", "effective_until": DATE}]])
def test_bad_profile_validation_not_crash(bad):
    assert bp.parse_profile(bad).validation["status"] == "INVALID_SHEET"


def complete_rows():
    return [{"scope": "PRODUCT", "product_id": "P1", "normal_price": 1000, "discount_rate": 0.3},
            {"scope": "GLOBAL", "holding_cost_per_unit_day": 5, "disposal_cost_per_unit": 200, "salvage_value_per_unit": 0},
            {"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "promotion_uplift": 0.5}]


def test_first_repeat_counts_engine_audit_and_determinism():
    _, first = req.plan_input_requirements(base_input())
    _, repeat = run(complete_rows())
    assert first["required_input_count"] == 5 and repeat["required_input_count"] == 1
    assert repeat["decision_specific_required_inputs"][0]["field"] == "remaining_shelf_life_days"
    decision_inputs = [{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "remaining_shelf_life_days": 10}]
    d, p = run(complete_rows(), inputs=decision_inputs)
    assert d["recommendation_readiness"] == "RECOMMENDABLE" and p["stop_asking"]
    assert d["expected_loss_transfer"] == 2175
    assert run(complete_rows(), inputs=decision_inputs) == (d, p)
    source = d["input_sources"]["source_normal_price"]
    assert source["origin"] == "SELLER_BUSINESS_PROFILE" and source["provenance"] == "USER_INPUT"
    assert source["effective_value"] == 1000 and source["input_scope"] == "PRODUCT" and source["used_in_comparison"]
    assert any(a.get("profile_source") and a["effective_value"] == 1000 for a in d["seller_input_audit"])


def test_partial_profile_cannot_promote():
    rows = complete_rows()[:-1]
    d, _ = run(rows, inputs=[{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "remaining_shelf_life_days": 10}])
    assert d["comparison_status"] == "PARTIAL"
    assert not d["production_promotion_candidate"] and not d["production_action_applied"]


def test_scenario_profile_stays_separate():
    rows = [{**r, "input_type": "SCENARIO"} for r in complete_rows()]
    d, _ = run(rows)
    assert d["input_provenance"]["source_normal_price"]["value"] is None
    d, _ = run(rows, decision_mode=sli.SCENARIO, inputs=[{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "remaining_shelf_life_days": 10}])
    assert d["recommendation_readiness"] == "SCENARIO_ONLY"
    assert not d["production_promotion_candidate"]


def test_template_never_auto_loaded():
    from services.data_loader import OPTIONAL_SHEETS
    assert bp.SHEET_KEY in OPTIONAL_SHEETS
    assert bp.parse_profile(None).cells == []
    p = Path(__file__).resolve().parents[1] / "samples/seller_business_profile_TEMPLATE_SAMPLE.csv"
    frame = pd.read_csv(p)
    assert frame.note.str.contains("SAMPLE").all()
    assert bp.parse_profile(frame).validation["status"] == "VALID"


def test_optional_workbook_sheet_roundtrip_and_bad_sheet_isolated():
    from services.data_loader import load_excel_data
    from services.analysis_pipeline import build_v2_state
    from tests.fixtures import sample_workbook, workbook_excel_bytes
    data = sample_workbook()
    data[bp.SHEET_KEY] = pd.DataFrame([{"scope": "INVALID", "normal_price": "wrong"}])
    loaded = load_excel_data(workbook_excel_bytes(data))
    assert bp.SHEET_KEY in loaded
    analysis = build_v2_state(loaded, detail_level="core")["pipeline_result"]["seller_loss_analysis"]
    assert analysis[bp.SHEET_KEY]["status"] == "INVALID_SHEET"
    assert analysis["status"] == "parallel_only"


def test_suhyup_shaped_staged_profile_e2e(tmp_path, monkeypatch):
    from services.analysis_pipeline import run_analysis_pipeline
    from tests.test_real_transport_enrichment import _write_real_data
    from tests.test_seller_loss_inputs import _suhyup_shaped_upload
    from services.seller_loss_requirement_validation import TestSeller, TEST_INPUT_LABEL
    _write_real_data(tmp_path)
    monkeypatch.setenv("VARO_REAL_DATA_ROOT", str(tmp_path))
    upload = _suhyup_shaped_upload()
    baseline = run_analysis_pipeline(upload)
    a = baseline.seller_loss_analysis["decisions"][0]
    product = a["product_id"]
    # Use the uploaded snapshot date, never today's date.
    date = str(pd.to_datetime(upload["inventory"].snapshot_date).max().date())
    rows = [{"scope": "PRODUCT", "product_id": product, "normal_price": 30000, "discount_rate": 0.3,
             "effective_date": date, "effective_until": date, "currency": "KRW", "note": TEST_INPUT_LABEL}]
    upload[bp.SHEET_KEY] = pd.DataFrame(rows)
    b = run_analysis_pipeline(upload)
    assert b.seller_loss_analysis["decisions"][0]["input_requirements"]["required_input_count"] < a["input_requirements"]["required_input_count"]
    seller = TestSeller(b.seller_loss_analysis["decisions"])
    c = b
    for _ in range(8):
        items = [i for d in c.seller_loss_analysis["decisions"] for i in d["input_requirements"]["required_user_inputs"]]
        if not items:
            break
        seller.answer(items)
        upload[sli.SHEET_KEY] = seller.frame()
        c = run_analysis_pipeline(upload)
    assert c.recommendations == baseline.recommendations
    assert all(d["recommendation_readiness"] == sli.RECOMMENDABLE for d in c.seller_loss_analysis["decisions"])
    assert all(d["input_requirements"]["stop_asking"] for d in c.seller_loss_analysis["decisions"])
    assert all(not d["production_action_applied"] for d in c.seller_loss_analysis["decisions"])
    assert any(s["origin"] == "SELLER_BUSINESS_PROFILE" and s["used_in_comparison"]
               for d in c.seller_loss_analysis["decisions"] for s in d["input_sources"].values())


def test_profile_unit_audit_and_real_unit_preserved():
    p = bp.parse_profile([{"scope": "PRODUCT", "product_id": "P1", "quantity_unit": "EA"}])
    m, record = sli.merge_seller_inputs(base_input(), p)
    assert m.quantity_unit == "EA"
    assert record["fields"]["quantity_unit"]["origin"] == "SELLER_BUSINESS_PROFILE"
    assert record["audit"][0]["effective_value"] == "EA"
    m, record = sli.merge_seller_inputs(base_input(quantity_unit="KG"), p)
    assert m.quantity_unit == "KG" and record["conflicts"]


def test_currency_and_unit_mismatch_not_silently_converted():
    rows = [{"scope": "PRODUCT", "product_id": "P1", "normal_price": 1000, "currency": "USD", "quantity_unit": "KG"}]
    d, _ = run(rows, base=base_input(quantity_unit="EA"),
               inputs=[{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "remaining_shelf_life_days": 10}])
    assert d["recommendation_readiness"] != "RECOMMENDABLE"
    assert not d["production_action_applied"]
