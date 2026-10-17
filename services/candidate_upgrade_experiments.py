"""Frozen real-data candidate replay and declared SCENARIO stress experiments.

The Suhyup experiment is explicitly PROXY, with undeclared physical units and
unconfirmed operational route/capacity. It is not a production transfer plan.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
import argparse
import json
import math
import time
import tracemalloc
import hashlib
import pandas as pd
import numpy as np
from services.candidate_selection import AllocationOption as O, Candidate, SelectionProblem, solve, evaluate
from services.operating_snapshot import signature
from services.real_transport_enrichment import _load_tables, _optimize_vehicle_mix

POLICIES = ("LEGACY_20", "TOP_30", "TOP_40", "GENERATOR_ALL", "MULTI_TARGET", "ALL_VALID")


class TariffOptions:
    """Reuse the exact legacy tariff kernel; recompute integer trucks for every q."""
    def __init__(self, root: Path):
        self.tables = _load_tables(str(root))
        if self.tables is None:
            raise ValueError("official tariff / weight tables unavailable")

    @lru_cache(maxsize=10000)
    def price(self, product: str, source: str, target: str, q: int) -> float | None:
        weight = self.tables["weights"].get(product)
        route = self.tables["routes"].get((source, target))
        if not weight or not route or weight["status"] not in {"safe", "conditional"} or not weight["weight_kg"]:
            return None
        mix = self._mix(product, q)
        return None if mix is None else float(round(route["distance_km"] * mix["total_rate_per_km"]))

    @lru_cache(maxsize=2000)
    def _mix(self, product: str, q: int):
        return _optimize_vehicle_mix(self.tables["weights"][product]["weight_kg"] * q, self.tables["vehicles"])


def frozen_day_problem(day: pd.DataFrame, policy: str, tariffs: TariffOptions,
                       *, partial: bool = True) -> SelectionProblem:
    if policy not in POLICIES:
        raise ValueError("ALL_LANES is not an approved comparison policy")
    if day.snapshot_date.nunique() != 1:
        raise ValueError("one decision date required")
    caps, needs = {}, {}
    for r in day.to_dict("records"):
        p,s,t = str(r["product_id"]),str(r["source_id"]),str(r["target_id"])
        for key, value in ((f"source:{s}:{p}", min(float(r["source_stock"]), float(r["source_surplus"]))),
                           (f"target:{t}:{p}", float(r["target_need"]))):
            if math.isfinite(value):
                if key in caps and abs(caps[key] - value) > 1e-6:
                    raise ValueError("conflicting frozen cap: " + key)
                caps[key] = max(0.0,value)
        if math.isfinite(float(r["target_need"])):
            needs[f"target:{t}:{p}"] = max(0.0,float(r["target_need"]))
    candidates=[]
    for r in day[day["pool_"+policy] == "IN_POOL"].to_dict("records"):
        if r["validity_status"] != "VALID":
            continue
        p,s,t = str(r["product_id"]),str(r["source_id"]),str(r["target_id"])
        sk,tk=f"source:{s}:{p}",f"target:{t}:{p}"
        q=int(r["recommended_qty"])
        options=[]
        for amount in range(1,q+1) if partial else (q,):
            cost=tariffs.price(p,s,t,amount)
            if cost is not None:
                options.append(O(amount,cost))
        # Exact original-quantity consistency is required; never substitute a cheaper new tariff.
        original=next((o.cost for o in options if o.quantity==q),None)
        if original is None or abs(original-float(r["transport_cost"])) > 1e-6:
            raise ValueError("legacy tariff mismatch: " + str(r["candidate_id"]))
        candidates.append(Candidate(str(r["candidate_id"]),p,s,t,None,None,None,
            {sk:1,tk:1},tk,tuple(options),float(r["candidate_score"]),
            cost_basis="RECOMPUTED_QUANTITY_SPECIFIC_TARIFF",lineage=str(r["lineage"])))
    return SelectionProblem("Suhyup:"+str(day.snapshot_date.iloc[0]),caps,needs,tuple(candidates),5,"PROXY",
        service_metric="LEGACY_RAW_QTY",policy={"horizon_days":7,"quantity_rule":"original request upper bound; integer raw quantity",
        "need_formula":"BENCHMARK_PROXY","cost":"official tariff * road distance * derived item weight; integer truck mix",
        "unconfirmed":["physical quantity unit","state","lot","operator approved source/need/routes","route/DC capacity"],
        "operational_approval":False})


def stress_problem(kind: str) -> SelectionProblem:
    # Hand-checkable six-stock, two-target example with per-route fixed + variable cost.
    caps={"source":6.,"targetA":4.,"targetB":4.}
    candidates=[]
    for cid,target,fixed,score in (("a","targetA",2,10),("b","targetB",1,20)):
        resources={"source":1.,target:1.}
        if kind=="bottleneck":
            resources["DC"]=2
        candidates.append(Candidate(cid,"P","S",target,"CHILLED","L1","EA",resources,target,
            tuple(O(q,fixed+q) for q in range(1,5)),score,10,"FIXED_PLUS_LINEAR","SCENARIO hand case"))
    if kind=="bottleneck": caps["DC"]=8
    if kind=="scarce": caps["source"]=3
    return SelectionProblem(kind,caps,{"targetA":4.,"targetB":4.},tuple(candidates),2,"SCENARIO")


def run(root: Path, output: Path, *, policies=POLICIES) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    source=root/"_CANDIDATE_GENERATION_VALIDATION/candidate_universe_rows.parquet"
    universe=pd.read_parquet(source)
    hypothesis={"version":"upgrade-0.1","frozen_source":str(source),
        "primary":"Under identical proxy caps and tariff, partial allocation and candidate expansion may reduce computed transport cost at equal planned quantity",
        "fairness":"same date/common caps/objective/max_routes; candidate-set differences are treatment, not solver gap",
        "statistical_unit":"31 dependent daily snapshots, descriptive paired summaries only; no independent candidate-row p-values",
        "selection":"no model selected for production; all prespecified policies reported",
        "seeds":[],"time_limit_each":5,"failure":"any common-cap violation or tariff-original mismatch",
        "not_claimed":["actual saving","physical fulfillment","production readiness"]}
    (output/"preregistration.json").write_text(json.dumps(hypothesis,ensure_ascii=False,indent=2),encoding="utf-8")
    code_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__), Path(__file__).with_name("candidate_selection.py"))}
    tariff=TariffOptions(root); results=[]; details=[]; start=time.perf_counter()
    cap_excess=universe[(universe.pool_ALL_VALID=="IN_POOL") &
        ((universe.recommended_qty>universe.target_need+1e-9)|(universe.recommended_qty>universe.source_surplus+1e-9))]
    cap_excess.to_csv(output/"request_exceeds_common_cap.csv",index=False,encoding="utf-8-sig")
    for date,day in universe.groupby("snapshot_date",sort=True):
        for policy in policies:
            problem=frozen_day_problem(day,policy,tariff)
            for algo in ("VARO","PARTIAL","LOCAL","MILP"):
                result=solve(problem,algo,time_limit=5)
                results.append({"date":date,"policy":policy,"algorithm":algo,"candidate_count":len(problem.candidates),
                    **{k:result[k] for k in ("service","cost","quantity","routes","seconds")},
                    "violations":len(result["violations"]),"optimality_proven":result["solver"]["optimality_proven"],
                    "base_problem_signature":result["base_problem_signature"],"candidate_set_signature":result["candidate_set_signature"]})
                details.append({"date":date,"policy":policy,**result})
        print("completed",date,flush=True)
    # Equal-pool ranking ablations and unsafe diagnostic remain separate from eligible ranking.
    stress=[]
    for kind in ("mixed","bottleneck","scarce"):
        p=stress_problem(kind)
        for a in ("VARO","PARTIAL","LOCAL","VHS","NO_VHS","COST_ONLY","ECONOMIC","MILP"):
            stress.append({"case":kind,**solve(p,a)})
        unsafe={c.candidate_id:max(o.quantity for o in c.options) for c in p.candidates}
        stress.append({"case":kind,"algorithm":"UNSAFE_DIAGNOSTIC","plan":unsafe,**evaluate(p,unsafe)})
    frame=pd.DataFrame(results); frame.to_csv(output/"daily_results.csv",index=False,encoding="utf-8-sig")
    summaries=frame.groupby(["policy","algorithm"]).agg(days=("date","count"),quantity=("quantity","sum"),
        cost=("cost","sum"),violations=("violations","sum"),seconds=("seconds","sum"),
        median_seconds=("seconds","median"),max_seconds=("seconds","max"),proven_days=("optimality_proven","sum")).reset_index()
    summaries.to_csv(output/"summary.csv",index=False,encoding="utf-8-sig")
    # First run profiled Python allocations; final timings omit tracemalloc overhead.
    # Peak memory is reported by the dedicated W6 process-level stress run.
    peak=None
    report={"status":"PASS" if frame.violations.sum()==0 else "FAIL","evidence_class":"REAL_DATA_PROXY",
        "dates":int(frame.date.nunique()),"original_all_valid_candidates":int((universe.pool_ALL_VALID=="IN_POOL").sum()),
        "request_exceeds_common_cap":len(cap_excess),"results":summaries.to_dict("records"),"stress":stress,
        "seconds":time.perf_counter()-start,"python_peak_bytes":peak,"preregistration":hypothesis,
        "code_hashes_at_start":code_hashes,
        "algorithm_label_note":"VARO means whole-request common quantity-cost heuristic with generator-score tie, NOT full production VHS. Full VHS ablation is W5.",
        "limitations":hypothesis["not_claimed"]+["raw quantity units unknown","VHS field here is candidate-generator score, not full VHS; W5 separate"],
        "production_approved":False}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"plans.json").write_text(json.dumps(details,ensure_ascii=False),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--data-root",type=Path,default=Path(r"C:\VARO_V2_REAL_DATA"))
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    report=run(args.data_root,args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in {"stress","preregistration"}},ensure_ascii=False,indent=2))
