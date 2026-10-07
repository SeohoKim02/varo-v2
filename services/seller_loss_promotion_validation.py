"""Offline promotion-gate proof: real Suhyup, labelled TEST flows and artificial gate branches.

Artificial branch simulations never count as production eligibility. Every
fixture is also checked with its TEST context, which must block promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from services import seller_loss_promotion_gate as gate, seller_loss_inputs as sli, seller_loss_input_requirements as req
from services.seller_loss_engine import InputField, WORKBOOK_DATASET
from services.seller_loss_input_validation import load_suhyup_upload, _run, _sha256, PROCESSED, PROCESSED_FILES
from services.seller_loss_requirement_validation import staged_flow
from services.seller_decision_validation import controlled_scenarios, scenario_input, run_scenarios


def gate_logic_cases():
    """Artificial declared-production contracts exercise branches only, never real evidence."""
    base = scenario_input()
    base = replace(base, **{n: replace(f, provenance="DIRECT_REAL" if f.present else "MISSING",
        source="declared seller record", dataset=WORKBOOK_DATASET, note="") for n,f in base.input_fields().items()},
        decision_id="R001", product_id="P001", legacy_action="재고 이동")
    date = "2026-07-31"
    parse = lambda rows: sli.parse_seller_loss_inputs(rows, upload_currency="KRW", upload_currency_basis="seller upload contract")
    user = sli.evaluate_with_seller_inputs(replace(base, source_normal_price=InputField()),
        parse([{"scope":"PRODUCT", "product_id":"P001", "normal_price":1000, "effective_date":date}]), decision_date=date)
    only_user = replace(base, **{n:replace(f, provenance="USER_INPUT") for n,f in base.input_fields().items() if f.present})
    conflict = sli.evaluate_with_seller_inputs(base, parse([{"scope":"PRODUCT", "product_id":"P001", "normal_price":999}]), decision_date=date)
    pairs = sli.evaluate_with_seller_inputs(replace(base, promotion_uplift=InputField()), None, decision_date=date)
    values = [
        ("REAL_ONLY", sli.evaluate_with_seller_inputs(base,None), req.ALL_THREE, gate.PROMOTION_ELIGIBLE),
        ("REAL_PLUS_USER_INPUT",user,req.ALL_THREE,gate.PROMOTION_ELIGIBLE),
        ("USER_INPUT_ONLY",sli.evaluate_with_seller_inputs(only_user,None),req.ALL_THREE,gate.SHADOW_ONLY),
        ("SCENARIO",sli.evaluate_with_seller_inputs(base,None,decision_mode=sli.SCENARIO),req.ALL_THREE,gate.SCENARIO_ONLY),
        ("ALL_THREE_PARTIAL",pairs,req.ALL_THREE,gate.INSUFFICIENT_DATA),
        ("EXPLICIT_COMPLETE_PAIR",pairs,req.TRANSFER_VS_NORMAL,gate.PROMOTION_ELIGIBLE),
        ("CONFLICT",conflict,req.ALL_THREE,gate.BLOCKED),
    ]
    real = sli.evaluate_with_seller_inputs(base,None)
    for code in ("UNIT_MISMATCH","CURRENCY_MISMATCH"):
        values.append((code,{**real,"seller_input_issues":[code]},req.ALL_THREE,gate.BLOCKED))
    values.append(("EXACT_TIE",{**real,"tie":True},req.ALL_THREE,gate.SHADOW_ONLY))
    unknown = sli.evaluate_with_seller_inputs(replace(base,source_holding_cost_per_unit_day=InputField()),None)
    values.append(("UNKNOWN_OPTIONAL",unknown,req.ALL_THREE,gate.SHADOW_ONLY))
    return values


def run_validation(data_root: Path, output_dir: Path | None = None):
    root = Path(data_root)
    upload = load_suhyup_upload(root)
    inputs = [root / PROCESSED / name for name in PROCESSED_FILES]
    hashes = {str(p):_sha256(p) for p in inputs}
    actual = _run(upload, root)
    actual_analysis = actual["pipeline_result"]["seller_loss_analysis"]
    assert actual_analysis["promotion_summary"]["promotion_eligible"] == 0
    records, checks = [], []
    for row in actual_analysis["shadow_decisions"]:
        records.append({"dataset_group":"SUHYUP_ACTUAL", "scope":req.ALL_THREE, "data_kind":"actual_only", **row})
    for scope in (req.TRANSFER_VS_NORMAL,req.ALL_THREE):
        def runner(frame):
            return _run({**upload,req.COMPARISON_SCOPE_KEY:scope,sli.SHEET_KEY:frame}, root)
        flow = staged_flow(runner, scope)
        assert not flow["problems"] and not flow["audit_problems"]
        final = flow["final_step"]
        assert all(d["recommendation_readiness"] == sli.RECOMMENDABLE for d in final["decisions"])
        # Re-run the exact final TEST rows through the integrated pipeline.
        state = runner(pd.DataFrame(flow["test_rows"]["STEP"]))
        analysis = state["pipeline_result"]["seller_loss_analysis"]
        assert state["recommendations"] == actual["recommendations"]
        assert state["pipeline_result"]["summary"] == actual["pipeline_result"]["summary"]
        assert analysis["promotion_summary"]["promotion_eligible"] == 0
        assert all(r["promotion_status"] == gate.BLOCKED and "TEST_OR_SAMPLE_INPUT" in r["promotion_blockers"] for r in analysis["shadow_decisions"])
        for row in analysis["shadow_decisions"]:
            records.append({"dataset_group":"SUHYUP_TEST_USER_INPUT", "scope":scope, "data_kind":"TEST USER INPUT", **row})
    for scenario in controlled_scenarios():
        d = sli.evaluate_with_seller_inputs(scenario["input"],None)
        row = gate.shadow_decision(d, context={"data_kind":"CONTROLLED_SCENARIO", "is_test":True})
        assert not row["promotion_candidate"]
        records.append({"dataset_group":"CONTROLLED_SCENARIO", "scope":req.ALL_THREE, "data_kind":"synthetic fixture", "case":scenario["id"], **row})
    for name,d,scope,expected in gate_logic_cases():
        options = {"comparison_scope":scope,"explicit_scope":scope != req.ALL_THREE}
        hypothetical = gate.evaluate_promotion_gate(d,**options,context={"decision_date":"2026-07-31"})
        assert hypothetical["promotion_status"] == expected,(name,hypothetical)
        marked = gate.shadow_decision(d,**options,context={"is_test":True,"decision_date":"2026-07-31"})
        assert marked["promotion_status"] == gate.BLOCKED and not marked["promotion_candidate"]
        checks.append({"case":name,"label":"ARTIFICIAL GATE LOGIC SIMULATION; NOT PRODUCTION EVIDENCE",
            "expected_logic_branch":expected,"hypothetical_branch_without_fixture_labels":hypothetical["promotion_status"],
            "fixture_promotion_status":marked["promotion_status"],"check":"PASS","production_action_applied":False})
        records.append({"dataset_group":"GATE_FIXTURE", "scope":scope,"data_kind":"synthetic fixture","case":name,**marked})
    scenarios,_ = run_scenarios()
    assert hashes == {str(p):_sha256(p) for p in inputs}
    groups = {g:gate.summarize_shadow([r for r in records if r["dataset_group"]==g]) for g in sorted({r["dataset_group"] for r in records})}
    summary = {"policy":gate.policy_document(),"groups":groups,"all_recorded_decisions":gate.summarize_shadow(records),
        "gate_logic_simulation_checks":checks,"controlled_scenario_passed":int(scenarios["pass"].sum()),
        "controlled_scenario_total":len(scenarios),"processed_inputs_hashes_unchanged":True,"input_hashes":hashes,
        "legacy_actions_unchanged":True,"production_action_applied":False,
        "limits":"Artificial eligibility branches validate code only; recorded fixture decisions are BLOCKED. No production-like fixture is real operational evidence."}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True,exist_ok=True)
        flat = [{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records]
        pd.DataFrame(flat).to_csv(output/"seller_loss_shadow_decisions.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(checks).to_csv(output/"seller_loss_promotion_gate_validation.csv",index=False,encoding="utf-8-sig")
        (output/"seller_loss_promotion_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        (output/"seller_loss_action_mapping.json").write_text(json.dumps({"mapping":gate.ACTION_MAPPING,"unmappable":list(gate.UNMAPPABLE),"basis":gate.policy_document()["mapping_basis"]},ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path("C:/VARO_V2_REAL_DATA"))
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    result=run_validation(args.data_root,args.output_dir or args.data_root/"_SELLER_LOSS_VALIDATION")
    print(json.dumps({"groups":result["groups"],"all_recorded_decisions":result["all_recorded_decisions"],"controlled_scenarios":result["controlled_scenario_passed"],"gate_logic_checks":len(result["gate_logic_simulation_checks"])},ensure_ascii=False,indent=2))
