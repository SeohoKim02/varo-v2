"""Third-batch observed replanning and fair, fixed-exogenous-path evidence.

Only new output directories are writable. Financial/demand numbers below are
explicit SCENARIO inputs, not actual operating profit or recovered demand.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from services.candidate_adaptive import Observation, plan_adaptive
from services.candidate_economics import _canonical
from services.candidate_economics_experiments import demo_problem
from services.candidate_economics_policies import evaluate_policies, scenario_paths
from services.candidate_economics_second_experiments import adaptive_problem, operating_cases


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _case_metrics(p, report):
    paths = {x.path_id:x for x in scenario_paths(p)}
    markets = {x.market_id:x for x in p.markets}
    rows = []
    for policy, aggregate in report["results"].items():
        for result in aggregate["paths"]:
            path = paths[result["path_id"]]
            components = result.get("accounting_components", result.get("result",{}).get("accounting_components",{}).get("oracle",{}))
            if policy == "PERFECT_INFORMATION":
                solved=result.get("result",{})
                unmet=sum(x["unmet"] for x in solved.get("market_outcomes",[]))
                waste=sum(x["disposal"] for x in solved.get("inventory",[]))
                moves={(x["lot_id"],x["route_id"],x["departure_period"]):x["quantity"]
                       for x in solved.get("scenario_transfers",[]) if x["scenario"]=="oracle"}
            else:
                unmet=waste=0.
                moves={}
                for step in result.get("trajectory",[]):
                    for price in step["prices"]:
                        sold=sum(x["quantity"] for x in step["sales"] if x["market_id"]==price["market_id"] and x["level"]==price["level"])
                        unmet+=max(0.,path.demand[price["market_id"]][price["level"]]-sold)
                    waste+=sum(x["disposal"] for x in step["inventory"])
                    moves.update({(x["lot_id"],x["route_id"],x["period"]):x["quantity"] for x in step["moves"]})
            # This metric is deviation from the initial FIXED plan, not a count
            # of irrevocable executions revised by a planner.
            reference = {(x["lot_id"],x["route_id"],x["departure_period"]):x["quantity"]
                         for x in report["plans"]["OPEN_LOOP"].get("transfers",[])}
            deviation=sum(abs(moves.get(k,0)-reference.get(k,0)) for k in moves.keys()|reference.keys())
            future_cost=sum(components.get(k,0) for k in ("transport_cost","handling_cost","holding_cost","disposal_cost","justified_penalty"))
            rows.append({"case":p.problem_id,"policy":policy,"path":path.path_id,"probability":path.probability,
                "status":result["status"],"external_sales_revenue":components.get("external_sales_revenue"),
                "future_cost":future_cost if components else None,"terminal_value":components.get("terminal_value"),
                "net_cash_value":result.get("net_cash_value"),"unmet":unmet,"disposal":waste,
                "move_change_vs_initial_fixed_plan_l1":deviation,
                "replan_count":max(0,p.horizon-1) if policy=="ROLLING_HORIZON" else 0,
                "planning_calls":p.horizon if policy=="ROLLING_HORIZON" else 1,
                "trajectory_accounting_solves":p.horizon if policy!="PERFECT_INFORMATION" else 0,
                "runtime_seconds":result.get("runtime_seconds"),
                "constraint_violations":result.get("constraint_violations",result.get("result",{}).get("constraint_violations")),
                "data_class":"SCENARIO","objective":"EXPECTED_NET_CASH_MAX",
                "problem_signature":_digest(_canonical(p)),"exogenous_path_signature":_digest(path.__dict__),
                "information":"full future oracle" if policy=="PERFECT_INFORMATION" else "known prefix only",
                "time_limit_per_solver_seconds":30.,"automatic_execution":False})
    return rows


def small_enumeration(stock, capacity=4, fixed=2):
    """Independent enumeration of a one-lot two-market two-stage example."""
    alternatives=[]
    for q in range(stock+1):
        vehicles=math.ceil(q/capacity)
        if vehicles>2:
            continue
        values={"low":4*(stock-q)-fixed*vehicles,"high":10*q-fixed*vehicles}
        alternatives.append({"q":q,"vehicles":vehicles,"low":values["low"],"high":values["high"],
                             "expected":.5*(values["low"]+values["high"])})
    fixed_value=max(row["expected"] for row in alternatives)
    adaptive_value=.5*max(row["low"] for row in alternatives)+.5*max(row["high"] for row in alternatives)
    return {"stock":stock,"alternatives":alternatives,"open_loop_value":fixed_value,"adaptive_value":adaptive_value}


def observation_examples():
    p=adaptive_problem()
    origin="2026-10-17T00:00:00+09:00"
    now="2026-10-18T00:00:00+09:00"
    signal=Observation("signal","SIGNAL","2026-10-17T23:59:00+09:00",now,source="TEST operator signal",signal="HIGH")
    sale=Observation("sale","SALE","2026-10-17T12:00:00+09:00","2026-10-17T12:05:00+09:00",
        "P","S","CHILLED","lot_A","EA",2,"TEST receipt")
    receipt=replace(sale,event_id="inbound",kind="RECEIPT",quantity=1,occurred_at="2026-10-17T11:00:00+09:00",known_at="2026-10-17T11:05:00+09:00")
    output={}
    for policy in ("OPEN_LOOP","ROLLING_HORIZON","MULTISTAGE_RECOURSE"):
        initial=plan_adaptive(p,policy=policy,origin_at=origin,decision_at=origin)
        updated=plan_adaptive(p,policy=policy,origin_at=origin,decision_at=now,observations=(receipt,sale,signal),previous=initial)
        output[policy]={"initial":initial,"updated":updated}
    future=replace(signal,known_at="2026-10-19T00:00:00+09:00")
    a=plan_adaptive(p,policy="MULTISTAGE_RECOURSE",origin_at=origin,decision_at=now,observations=(future,))
    b=plan_adaptive(p,policy="MULTISTAGE_RECOURSE",origin_at=origin,decision_at=now,observations=(replace(future,signal="LOW"),))
    output["nonanticipativity"]={"same_current_action":a["current_actions"]==b["current_actions"],
        "same_problem_signature":a["problem_signature"]==b["problem_signature"],
        "deferred_ids":a["deferred_observations"],"future_observation_modified_only":True}
    return output


def run(output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Existing evidence must not be overwritten")
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    cases=[adaptive_problem(),adaptive_problem(noisy=True),adaptive_problem(late=True),
           demo_problem("shared_vehicle"),demo_problem("expiry_block")]
    cases.extend(operating_cases()[:4])
    comparisons,rows,enumerations={ },[],{}
    for p in cases:
        result=evaluate_policies(p)
        comparisons[p.problem_id]={"problem":_canonical(p),"evaluation":result}
        rows.extend(_case_metrics(p,result))
    for stock in range(7):
        p=adaptive_problem()
        p=replace(p,problem_id=f"enumeration_stock_{stock}",lots=(replace(p.lots[0],stock=stock),),
            routes=(replace(p.routes[0],vehicle_capacity=4,max_vehicles=2),))
        result=evaluate_policies(p)
        exact=small_enumeration(stock)
        exact["computed"]={key:value["expected_net_value"] for key,value in result["results"].items()}
        # Rolling solves an open-loop continuation and can commit early when
        # departure times tie. It is NOT mathematically equal to multistage.
        hand_rolling=0.
        for path in result["results"]["ROLLING_HORIZON"]["paths"]:
            q_by_t={step["period"]:sum(x["quantity"] for x in step["moves"]) for step in path["trajectory"]}
            q=sum(q_by_t.values())
            freight=sum(2*math.ceil(value/4) for value in q_by_t.values())
            value=(10*q if path["path_id"]=="high" else 4*(stock-q))-freight
            hand_rolling+=.5*value
        exact["rolling_trajectory_hand_value"]=hand_rolling
        exact["rolling_equals_multistage"]=abs(hand_rolling-exact["adaptive_value"])<1e-7
        exact["matches"]=all(abs(value-(exact["open_loop_value"] if key=="OPEN_LOOP" else
            hand_rolling if key=="ROLLING_HORIZON" else exact["adaptive_value"]))<1e-7
            for key,value in exact["computed"].items())
        enumerations[str(stock)]=exact
        comparisons[p.problem_id]={"problem":_canonical(p),"evaluation":result}
        rows.extend(_case_metrics(p,result))
    aggregated=[]
    for case in comparisons:
        for policy in ("OPEN_LOOP","ROLLING_HORIZON","MULTISTAGE_RECOURSE","PERFECT_INFORMATION"):
            selected=[x for x in rows if x["case"]==case and x["policy"]==policy]
            good=all(x["net_cash_value"] is not None for x in selected)
            summary={"case":case,"policy":policy,"status":"COMPLETED" if good else "INCOMPLETE",
                "worst_net_value":min(x["net_cash_value"] for x in selected) if good else None,
                "runtime_seconds":sum(x["runtime_seconds"] for x in selected),
                "constraint_violations":sum(x["constraint_violations"] or 0 for x in selected),
                "replan_count_per_path":selected[0]["replan_count"]}
            for key in ("external_sales_revenue","future_cost","terminal_value","net_cash_value","unmet","disposal","move_change_vs_initial_fixed_plan_l1"):
                summary["expected_"+key]=sum(x[key]*x["probability"] for x in selected) if good else None
            aggregated.append(summary)
    report={"version":"third-adaptive-policy-v0.1","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "data_class":"SCENARIO","production_approved":False,"case_count":len(comparisons),
        "path_policy_runs":len(rows),"comparisons":comparisons,"aggregate_metrics":aggregated,
        "small_exhaustive_enumeration":enumerations,"observed_update_examples":observation_examples(),
        "runtime_seconds":time.perf_counter()-started,
        "comparison_contract":{"initial_stock":"Identical within each case","exogenous_paths":"Identical within each case",
            "financial_equations":"Unchanged candidate economic equations","price_options":"Same supplied finite set",
            "oracle":"Research-only full future information, not operationally achievable",
            "counts":"Simulation planning/replanning only; no actual operation performed",
            "timing":"Each policy playback includes its accounting feasibility solves; shared initial-plan time is separate in full evaluation plans"},
        "limitations":["Finite supplied latent-demand scenarios, not historical-sales counterfactual evidence",
            "No claim that adaptive decisions always beat a fixed plan",
            "One-unit rolling example commits before the signal under an equal-open-loop-value timing tie: rolling 3 versus multistage 6; not a solver arithmetic failure",
            "Replanning UI uses completed-period observations at declared period boundaries",
            "Movement deviation is relative to initial fixed plan, not cancellation of executed movement",
            "No field operator, real outcome or realized savings","No change to protected model/data/DB/Golden files"]}
    (output/"adaptive_policy_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    for filename,values in (("policy_path_metrics.csv",rows),("policy_aggregate_metrics.csv",aggregated)):
        with (output/filename).open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    return {"report":str(output/"adaptive_policy_report.json"),"cases":len(comparisons),"path_policy_runs":len(rows),
            "enumeration_matches":sum(x["matches"] for x in enumerations.values()),
            "violations":sum(x["constraint_violations"] or 0 for x in rows),"seconds":report["runtime_seconds"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=False))
