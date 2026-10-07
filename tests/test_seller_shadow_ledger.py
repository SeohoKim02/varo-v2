"""Artificial contract fixtures only. No fixture is real operational evidence."""
import copy
import io
import json
import sqlite3
from dataclasses import replace

import pandas as pd
import pytest

from services import seller_shadow_ledger as ledger, seller_shadow_outcomes as outcomes
from services import seller_loss_inputs as sli, seller_loss_promotion_gate as gate
from services.seller_loss_engine import InputField
from services.seller_business_profile import parse_profile
from services.simulation_history import initialize_history_storage, history_db_path, list_simulation_runs, record_failed_run
from services.analysis_pipeline import build_v2_state
from services.data_loader import load_excel_data
from tests.fixtures import sample_workbook
from tests.test_seller_loss_promotion_gate import eligible_input, decision

DATE = "2026-07-31T00:00:00+09:00"


def make_record(directory, mode="PRODUCTION", key="run1", d=None, legacy="재고 이동"):
    # Emulated PRODUCTION branches test the contract, never real business data.
    d = copy.deepcopy(d or decision())
    d["legacy_action"] = legacy
    run_id = ledger.save_run(data_signature="sig",execution_key=key,decision_at=DATE,directory=directory)
    rec = {"route_id":d["decision_id"], "varo_action":legacy,"varo_final_rank":1,
           "recommended_qty":10,"move_cost":100,"reason":"operational recommendation"}
    shadow = gate.shadow_decision(d,context={"decision_date":"2026-07-31"})
    return ledger.decision_record(run_id,DATE,d,rec,shadow=shadow,context={"data_mode":mode})


def event(record, **changes):
    return {"decision_id":record["decision_id"],"decision_version":1,"execution_status":"EXECUTED",
        "operator_action":"TRANSFER","recorded_at":"2026-08-02T12:00:00+09:00",
        "executed_at":"2026-08-01T12:00:00+09:00","outcome_provenance":"OPERATOR_CONFIRMED",
        "outcome_source":"audited sales ledger reference","currency":"KRW","quantity_unit":"ea",
        "data_mode":"PRODUCTION", **changes}


def test_create_ledger_and_schema_version(tmp_path):
    path = ledger.initialize_ledger(tmp_path)
    with sqlite3.connect(path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"simulation_runs","simulation_selected_routes","seller_shadow_runs","seller_shadow_decisions",
                "seller_shadow_outcomes","seller_shadow_schema"} == tables
        assert db.execute("SELECT version FROM seller_shadow_schema").fetchone()[0] == 1


def test_existing_db_migration_preserves_history(tmp_path):
    initialize_history_storage(tmp_path)
    old = record_failed_run(run_key="old",error_summary="recorded historical failure",directory=tmp_path)
    before = list_simulation_runs(directory=tmp_path)
    ledger.initialize_ledger(tmp_path)
    assert list_simulation_runs(directory=tmp_path) == before
    record = make_record(tmp_path)
    assert ledger.save_decision(record,tmp_path)["decision_version"] == 1
    assert old == before[0]["run_id"]


def test_future_schema_not_overwritten(tmp_path):
    path = ledger.initialize_ledger(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE seller_shadow_schema SET version=999")
    with pytest.raises(ledger.LedgerError,match="FUTURE"):
        ledger.initialize_ledger(tmp_path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM seller_shadow_schema").fetchone()[0] == 999


@pytest.mark.parametrize("mode", outcomes.DATA_MODES)
def test_save_modes_and_production_isolation(tmp_path,mode):
    r = make_record(tmp_path,mode=mode)
    ledger.save_decision(r,tmp_path)
    assert ledger.list_decisions(tmp_path)[0]["data_mode"] == mode
    assert ledger.summarize_ledger(tmp_path)["total_decisions"] == (1 if mode=="PRODUCTION" else 0)


def test_run_and_decision_ids_are_business_key_stable(tmp_path):
    first = make_record(tmp_path)
    assert first["decision_id"] == make_record(tmp_path)["decision_id"]
    assert first["decision_id"] != make_record(tmp_path,key="run2")["decision_id"]
    assert first["run_id"] != first["decision_id"]
    changed = dict(first,route_id="R999")
    assert first["decision_id"] != ledger.build_decision_id(first["run_id"],**{k:changed[k] for k in (
        "decision_at","product_id","source_store_id","target_store_id","route_id")})


def test_duplicate_decision_and_append_versions(tmp_path):
    r = make_record(tmp_path)
    assert ledger.save_decision(r,tmp_path)["inserted"]
    assert not ledger.save_decision(r,tmp_path)["inserted"]
    changed = {**r,"expected_loss_transfer":123}
    assert ledger.save_decision(changed,tmp_path)["decision_version"] == 2
    all_rows = ledger.list_decisions(tmp_path,include_versions=True)
    assert len(all_rows)==2 and all_rows[0]["expected_loss_transfer"]==r["expected_loss_transfer"]
    assert ledger.list_decisions(tmp_path)[0]["expected_loss_transfer"]==123


def test_data_mode_cannot_change_same_lineage(tmp_path):
    r = make_record(tmp_path,mode="TEST")
    ledger.save_decision(r,tmp_path)
    with pytest.raises(ledger.LedgerError,match="DATA_MODE_LINEAGE_CONFLICT"):
        ledger.save_decision({**r,"data_mode":"PRODUCTION"},tmp_path)


def test_outcome_update_append_and_idempotency(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    e = event(r,actual_sold_qty=0,actual_transfer_cost=100)
    assert ledger.import_outcomes([e],tmp_path)["saved"] == 1
    assert ledger.import_outcomes([e],tmp_path)["duplicates"] == 1
    assert ledger.import_outcomes([{**e,"actual_sold_qty":2}],tmp_path)["saved"] == 1
    rows = ledger.list_outcomes(tmp_path,include_versions=True)
    assert len(rows)==2 and rows[0]["actual_sold_qty"]==0 and rows[1]["actual_sold_qty"]==2


def test_unmatched_outcome_does_not_crash(tmp_path):
    r = make_record(tmp_path)
    report = ledger.import_outcomes([event(r)],tmp_path)
    assert report["status"]=="INVALID" and report["errors"][0]["code"]=="DECISION_NOT_FOUND"
    assert ledger.list_outcomes(tmp_path)==[]


def test_expected_actual_null_and_zero_separated(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    ledger.import_outcomes([event(r,actual_transfer_cost=0)],tmp_path)
    saved = ledger.list_outcomes(tmp_path)[0]
    assert saved["actual_transfer_cost"]==0 and saved["actual_sold_qty"] is None
    assert saved["actual_realized_loss"] is None
    assert ledger.list_decisions(tmp_path)[0]["expected_loss_transfer"]==r["expected_loss_transfer"]


@pytest.mark.parametrize("status", ["NOT_EXECUTED","PLANNED","CANCELLED","UNKNOWN"])
def test_actual_values_require_execution(status):
    r = {"decision_id":"D1"}
    parsed = outcomes.parse_seller_outcomes([event(r,execution_status=status,actual_sold_qty=1)])
    assert parsed["status"]=="INVALID" and any(e["code"]=="ACTUAL_REQUIRES_EXECUTED" for e in parsed["errors"])


@pytest.mark.parametrize("action,agreement", [("재고 이동","AGREE"),("할인","DISAGREE"),("보류","UNMAPPABLE")])
def test_agreement_mapping_reused(tmp_path,action,agreement):
    r = make_record(tmp_path,legacy=action)
    assert r["legacy_seller_agreement"]==agreement


def test_outcome_pending_does_not_assume_operator_execution(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    view = ledger.list_evaluations(tmp_path)[0]
    assert view["operator_action"] is None and view["execution_status"]=="NOT_EXECUTED"
    assert view["comparison"]["status"]=="OUTCOME_PENDING"
    assert ledger.summarize_ledger(tmp_path)["executed_decisions"]==0


def test_unknown_legacy_operator_action_not_comparable(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    ledger.import_outcomes([event(r,operator_action="폐기",actual_disposal_qty=1)],tmp_path)
    comparison = ledger.list_evaluations(tmp_path)[0]["comparison"]
    assert comparison["status"]=="NOT_COMPARABLE" and comparison["winner"] is None


def test_promotion_candidate_never_applies(tmp_path):
    r = make_record(tmp_path)
    assert r["promotion_status"]==gate.PROMOTION_ELIGIBLE
    assert r["production_action"]==r["legacy_action"] and not r["production_action_applied"]
    ledger.save_decision(r,tmp_path)
    for changed in ({**r,"production_action":"할인"},{**r,"production_action_applied":True}):
        with pytest.raises(ledger.LedgerError,match="PRODUCTION_ACTION"):
            ledger.save_decision(changed,tmp_path)


def test_profile_provenance_persisted_and_test_profile_isolated(tmp_path):
    table = parse_profile([{"scope":"PRODUCT","product_id":"P001","normal_price":1000,"currency":"KRW",
        "effective_date":"2026-07-01","effective_until":"2026-07-31","profile_source":"TEST USER INPUT"}])
    d = sli.evaluate_with_seller_inputs(replace(eligible_input(),source_normal_price=InputField()),table,decision_date="2026-07-31")
    r = make_record(tmp_path,d=d)
    ledger.save_decision(r,tmp_path)
    saved = ledger.list_decisions(tmp_path)[0]
    assert saved["input_sources"]["source_normal_price"]["profile_source"]=="TEST USER INPUT"
    assert saved["input_provenance"]["source_normal_price"]["provenance"]=="USER_INPUT"
    assert saved["data_mode"]=="TEST" and ledger.summarize_ledger(tmp_path)["total_decisions"]==0


def test_export_all_versions_but_production_summary_only(tmp_path):
    for mode in outcomes.DATA_MODES:
        r = make_record(tmp_path,mode=mode,key=mode)
        ledger.save_decision(r,tmp_path)
    folder = tmp_path/"exports"
    result = ledger.export_ledger(folder,tmp_path)
    assert result["total_decisions"]==1
    assert set(pd.read_csv(folder/"seller_shadow_decisions.csv").data_mode)==set(outcomes.DATA_MODES)
    assert pd.read_csv(folder/"seller_shadow_outcomes.csv").empty
    assert json.loads((folder/"seller_shadow_pilot_summary.json").read_text())["total_decisions"]==1


def test_missing_and_valid_workbook_outcome_sheet(tmp_path):
    data = sample_workbook()
    plain = build_v2_state(data)
    assert "seller_outcomes_validation" not in plain["pipeline_result"]["seller_loss_analysis"]
    buf = io.BytesIO()
    row = {"decision_id":"EXPORTED_ID_REQUIRED","execution_status":"NOT_EXECUTED","data_mode":"SAMPLE","recorded_at":DATE}
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        for key,frame in {**data,"seller_outcomes":pd.DataFrame([row])}.items():
            if isinstance(frame,pd.DataFrame):
                frame.to_excel(writer,sheet_name=key,index=False)
    loaded = load_excel_data(buf)
    valid = build_v2_state(loaded)
    assert valid["pipeline_result"]["seller_loss_analysis"]["seller_outcomes_validation"]["status"]=="VALID"


def test_invalid_outcome_sheet_keeps_analysis_identical():
    data = sample_workbook()
    plain = build_v2_state(data)
    bad = build_v2_state({**data,"seller_outcomes":pd.DataFrame([{"decision_id":"D1","actual_sold_qty":"bad"}])})
    assert bad["recommendations"]==plain["recommendations"]
    assert bad["pipeline_result"]["summary"]==plain["pipeline_result"]["summary"]
    assert bad["pipeline_result"]["seller_loss_analysis"]["seller_outcomes_validation"]["status"]=="INVALID"


def test_pipeline_opt_in_preserves_outputs_and_reports_ledger_errors(tmp_path):
    data = sample_workbook()
    plain = build_v2_state(data)
    stored = build_v2_state(data,shadow_ledger_context={"directory":tmp_path,"execution_key":"one",
        "data_signature":"sig","decision_at":DATE,"data_mode":"SAMPLE"})
    assert stored["seller_shadow_ledger"]["status"]=="RECORDED"
    assert stored["recommendations"]==plain["recommendations"] and stored["pipeline_result"]==plain["pipeline_result"]
    bad = build_v2_state(data,shadow_ledger_context={"directory":tmp_path,"execution_key":"two",
        "data_signature":"sig","decision_at":"bad"})
    assert bad["seller_shadow_ledger"]["status"]=="ERROR" and bad["recommendations"]==plain["recommendations"]


def test_realized_loss_only_observed_accounting_components():
    assert outcomes.realized_loss({"actual_revenue":100,"actual_sold_qty":1})["status"]=="REALIZED_LOSS_UNAVAILABLE"
    e = event({"decision_id":"D1"},actual_loss_debits=50,actual_loss_credits=20,loss_basis="ACCOUNTING_LOSS")
    assert outcomes.realized_loss(e)["value"]==30
    assert outcomes.realized_loss({**e,"actual_loss_credits":None})["value"] is None
    assert outcomes.realized_loss({**e,"actual_realized_loss":99})["status"]=="REALIZED_LOSS_CONFLICT"


def test_one_action_never_confirms_absolute_winner(tmp_path):
    r = make_record(tmp_path,legacy="할인")
    e = event(r,actual_realized_loss=100,loss_basis="OBSERVED_AVOIDABLE_LOSS")
    comparison = outcomes.compare_outcome(r,e)
    assert comparison["status"]=="PARTIAL_EVIDENCE"
    assert comparison["winner"] is None and not comparison["counterfactual_computed"]
    assert comparison["seller_expected_vs_actual"]["observed_strategy"]=="TRANSFER"


def test_confirmed_means_calibration_not_strategy_superiority(tmp_path):
    r = make_record(tmp_path)
    e = event(r,actual_realized_loss=0,loss_basis="OBSERVED_AVOIDABLE_LOSS")
    assert outcomes.compare_outcome(r,e)["status"]=="CONFIRMED"
    ledger.save_decision(r,tmp_path)
    ledger.import_outcomes([e],tmp_path)
    s = ledger.summarize_ledger(tmp_path)
    assert s["outcomes_available"]==1 and s["confirmed_better"]==s["confirmed_worse"]==0


def test_outcome_mismatched_units_and_currency_not_comparable(tmp_path):
    r = make_record(tmp_path)
    for changes in ({"currency":"USD"},{"quantity_unit":"KG"}):
        assert outcomes.compare_outcome(r,event(r,actual_transfer_cost=1,**changes))["status"]=="NOT_COMPARABLE"


def test_outcomes_link_exact_version(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    ledger.save_decision({**r,"expected_loss_transfer":123},tmp_path)
    report = ledger.import_outcomes([event(r,decision_version=None)],tmp_path)
    assert report["errors"][0]["code"]=="DECISION_VERSION_REQUIRED"
    assert ledger.import_outcomes([event(r,actual_transfer_cost=50)],tmp_path)["saved"]==1
    assert ledger.list_evaluations(tmp_path)[0]["outcome"] is None


def test_test_outcome_never_counts_or_masks_production_outcome(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    ledger.import_outcomes([event(r,actual_transfer_cost=1)],tmp_path)
    ledger.import_outcomes([event(r,actual_transfer_cost=999,data_mode="TEST",outcome_source="TEST USER INPUT")],tmp_path)
    assert ledger.list_evaluations(tmp_path)[0]["outcome"]["actual_transfer_cost"]==1
    assert ledger.summarize_ledger(tmp_path)["outcomes_available"]==1


def test_production_label_cannot_promote_test_outcome_source(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    ledger.import_outcomes([event(r,actual_transfer_cost=1,outcome_source="TEST USER INPUT")],tmp_path)
    assert ledger.list_outcomes(tmp_path)[0]["data_mode"]=="TEST"
    assert ledger.summarize_ledger(tmp_path)["outcomes_available"]==0


@pytest.mark.parametrize("bad", [{"operator_action":"made up"},{"outcome_provenance":"USER_INPUT"},
    {"actual_discount_rate":2},{"actual_sold_qty":float('inf')},{"actual_sold_qty":-1},
    {"actual_revenue":1,"currency":None},{"actual_sold_qty":1,"quantity_unit":None},
    {"actual_sold_qty":1,"outcome_source":None},{"recorded_at":"bad"}])
def test_invalid_contract_rejected(bad):
    assert outcomes.parse_seller_outcomes([event({"decision_id":"D1"},**bad)])["status"]=="INVALID"


def test_privacy_fields_not_saved(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision({**r,"operator_name":"private","phone":"private"},tmp_path)
    saved = ledger.list_decisions(tmp_path)[0]
    assert "phone" not in saved and "operator_name" not in saved
    report = ledger.import_outcomes([event(r,operator_name="private",phone="private")],tmp_path)
    assert report["saved"]==1
    assert "private" not in json.dumps(ledger.list_outcomes(tmp_path))


def test_pilot_readiness_without_real_outcomes(tmp_path):
    report = ledger.pilot_readiness(tmp_path)
    assert report["status"]=="READY_WITH_LIMITATIONS" and all(report["checks"].values())
    assert ledger.list_decisions(tmp_path)==[] and ledger.list_outcomes(tmp_path)==[]


@pytest.mark.parametrize("marker,mode", [({"is_test":True},"TEST"),({"is_sample":1.0},"SAMPLE"),
    ({"synthetic_fixture":True},"TEST"),({"source_file":"samples/outcomes.csv"},"SAMPLE"),
    ({"data_kind":"CONTROLLED_SCENARIO"},"SCENARIO")])
def test_outcome_markers_cannot_enter_production(tmp_path,marker,mode):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    report = ledger.import_outcomes([event(r,actual_transfer_cost=1,**marker)],tmp_path)
    assert report["saved"]==1
    assert ledger.list_outcomes(tmp_path)[0]["data_mode"]==mode
    assert ledger.summarize_ledger(tmp_path)["outcomes_available"]==0


def test_outcome_timestamp_lineage_and_stale_update(tmp_path):
    r = make_record(tmp_path)
    ledger.save_decision(r,tmp_path)
    ledger.import_outcomes([event(r,actual_transfer_cost=2)],tmp_path)
    stale = event(r,actual_transfer_cost=3,recorded_at="2026-08-01T13:00:00+09:00")
    report = ledger.import_outcomes([stale],tmp_path)
    assert report["errors"][0]["code"]=="STALE_OUTCOME_SNAPSHOT"
    naive = event(r,recorded_at="2026-08-02T12:00:00",executed_at="2026-08-01T12:00:00")
    assert ledger.import_outcomes([naive],tmp_path)["errors"][0]["code"]=="TIMEZONE_CONVENTION_MISMATCH"


def test_invalid_sheet_types_and_readiness_storage_failure(tmp_path):
    for source in ("not a sheet",123,{"actual_sold_qty":1}):
        assert outcomes.parse_seller_outcomes(source)["status"]=="INVALID"
    bad = tmp_path/"not_a_directory"
    bad.touch()
    assert ledger.pilot_readiness(bad)["status"]=="NOT_READY"
