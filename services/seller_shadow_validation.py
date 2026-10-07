"""UI-free shadow pilot wiring checks. Records no fabricated operator outcomes.

Default storage reuses the existing local simulation history SQLite file. Run
with --data-root and optional --history-dir/--output-dir. Validation runs have
fixed execution keys, so rerunning this command is idempotent. Reports are
local-only, and production summaries exclude every test/sample/scenario row.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import pandas as pd

from services import seller_shadow_ledger as ledger, seller_loss_promotion_gate as gate
from services import seller_loss_inputs as sli, seller_loss_input_requirements as req
from services.seller_loss_input_validation import load_suhyup_upload, _run, _sha256, PROCESSED, PROCESSED_FILES
from services.seller_loss_requirement_validation import staged_flow
from services.seller_decision_validation import controlled_scenarios, run_scenarios
from services.seller_loss_promotion_validation import gate_logic_cases
from services.simulation_history import initialize_history_storage, history_db_path


def _history_hash(directory):
    initialize_history_storage(directory)
    with sqlite3.connect(history_db_path(directory)) as db:
        return {t:ledger._hash(db.execute("SELECT * FROM "+t+" ORDER BY 1,2").fetchall())
                for t in ("simulation_runs","simulation_selected_routes")}


def run_validation(data_root: Path, history_dir=None, output_dir=None):
    root = Path(data_root)
    paths = [root/PROCESSED/name for name in PROCESSED_FILES]
    hashes = {str(p):_sha256(p) for p in paths}
    history_before = _history_hash(history_dir)
    upload = load_suhyup_upload(root)
    state = _run(upload,root)
    date = str(pd.to_datetime(upload["inventory"].snapshot_date).max().date())
    decision_at = date+"T00:00:00+09:00"
    signature = ledger._hash(hashes)
    recorded = ledger.record_pipeline_run(state,upload,execution_key="suhyup-actual-shadow-v0.1",data_signature=signature,
        decision_at=decision_at,directory=history_dir)
    actual_ids = {r["decision_id"] for r in recorded["decisions"]}
    checks = []
    for scope in (req.TRANSFER_VS_NORMAL,req.ALL_THREE):
        def runner(frame):
            return _run({**upload,req.COMPARISON_SCOPE_KEY:scope,sli.SHEET_KEY:frame},root)
        flow = staged_flow(runner,scope)
        assert not flow["problems"] and not flow["audit_problems"]
        data = {**upload,req.COMPARISON_SCOPE_KEY:scope,sli.SHEET_KEY:pd.DataFrame(flow["test_rows"]["STEP"])}
        test_state = _run(data,root)
        assert test_state["recommendations"]==state["recommendations"]
        assert test_state["pipeline_result"]["summary"]==state["pipeline_result"]["summary"]
        ledger.record_pipeline_run(test_state,data,execution_key="suhyup-test-shadow-v0.1-"+scope,data_signature=signature,
            decision_at=decision_at,data_mode="TEST",directory=history_dir)
    for case in controlled_scenarios():
        d = sli.evaluate_with_seller_inputs(case["input"],None)
        run = ledger.save_run(data_signature="controlled-scenario-contract",execution_key="shadow-v0.1-"+case["id"],
                             decision_at=decision_at,directory=history_dir)
        shadow = gate.shadow_decision(d,context={"is_test":True})
        rec = {"route_id":d["decision_id"],"varo_action":d["legacy_action"]}
        row = ledger.decision_record(run,decision_at,d,rec,shadow=shadow,context={"data_mode":"SCENARIO"})
        assert row["data_mode"]=="SCENARIO"
        ledger.save_decision(row,history_dir)
    for name,d,scope,expected in gate_logic_cases():
        run = ledger.save_run(data_signature="artificial-gate-contract",execution_key="shadow-v0.1-"+name,
            decision_at=decision_at,directory=history_dir)
        shadow = gate.shadow_decision(d,comparison_scope=scope,explicit_scope=scope!=req.ALL_THREE,
                                     context={"is_test":True,"decision_date":date})
        mode = "SAMPLE" if name=="UNKNOWN_OPTIONAL" else "TEST"
        row = ledger.decision_record(run,decision_at,d,{"route_id":d["decision_id"],"varo_action":d["legacy_action"]},
                                     shadow=shadow,context={"data_mode":mode})
        ledger.save_decision(row,history_dir)
    rows = ledger.list_decisions(history_dir)
    actual = [r for r in rows if r["decision_id"] in actual_ids]
    assert len(actual)==20 and all(r["data_mode"]=="PRODUCTION" and r["promotion_status"]==gate.INSUFFICIENT_DATA for r in actual)
    assert all(r["production_action"]==r["legacy_action"] and not r["production_action_applied"] for r in rows)
    summary = ledger.summarize_ledger(history_dir)
    assert all(r["data_mode"]=="PRODUCTION" for r in ledger.list_decisions(history_dir,data_mode="PRODUCTION"))
    assert summary["promotion_eligible"]==0
    repeated = ledger.record_pipeline_run(state,upload,execution_key="suhyup-actual-shadow-v0.1",data_signature=signature,
        decision_at=decision_at,directory=history_dir)
    assert not any(r["inserted"] for r in repeated["decisions"])
    assert hashes=={str(p):_sha256(p) for p in paths}
    assert history_before==_history_hash(history_dir)
    scenarios,_ = run_scenarios()
    assert scenarios["pass"].all()
    readiness = ledger.pilot_readiness(history_dir)
    assert readiness["status"]=="READY_WITH_LIMITATIONS" and all(readiness["checks"].values())
    validation = {"production_summary":summary,"current_actual_run_decisions":len(actual),
        "ledger_modes":dict(sorted(Counter(r["data_mode"] for r in rows).items())),
        "recommendations_and_kpis_unchanged":True,"existing_simulation_history_unchanged":True,
        "processed_inputs_hashes_unchanged":True,"idempotency_verified":True,
        "controlled_scenarios_passed":len(scenarios),"fabricated_operator_outcomes":0,
        "production_action_applied":False,"database":str(history_db_path(history_dir)),
        "limits":"No actual operator outcomes were supplied. TEST/SAMPLE/SCENARIO are excluded from production evaluation."}
    if output_dir is not None:
        output = Path(output_dir)
        ledger.export_ledger(output,history_dir)
        (output/"seller_pilot_readiness.json").write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding="utf-8")
        (output/"seller_shadow_ledger_validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    return validation,readiness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path("C:/VARO_V2_REAL_DATA"))
    parser.add_argument("--history-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    args = parser.parse_args()
    result,readiness = run_validation(args.data_root,args.history_dir,args.output_dir or args.data_root/"_SELLER_LOSS_VALIDATION")
    print(json.dumps({**result,"pilot_readiness":readiness["status"]},ensure_ascii=False,indent=2))
