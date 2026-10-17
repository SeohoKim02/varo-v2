"""Second-batch selection replay with prospective raw logs; read-only real data.

Previous 744 saved calculations are indexed as historical evidence, never relabeled
as a new run. New all-valid and prespecified synthetic comparisons use fresh IDs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import pandas as pd

from services.candidate_selection import (
    AllocationOption, Candidate, SelectionProblem, evaluate, exact_objective,
    improve_selection, incumbent_compatibility, protected_selection, solve,
)
from services.candidate_upgrade_experiments import TariffOptions, frozen_day_problem
from services.milp_benchmark_integrity_validation import process_peak_working_set_mb
from services.operating_snapshot import signature
from services.real_transport_enrichment import _optimize_vehicle_mix


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_rows(result, *, run_id, date, evidence, runtime_kind, baseline="VARO", policy="ALL_VALID", p=None):
    common = {"run_id": run_id, "case_id": str(date), "dataset_id": "Suhyup_2026_07_frozen" if p is None or p.evidence_class == "PROXY" else "DECLARED_SYNTHETIC",
              "decision_time": str(date), "base_problem_signature": result["base_problem_signature"],
              "candidate_set_signature": result["candidate_set_signature"], "algorithm_signature": result["algorithm_signature"],
              "information_set_id": signature({"date": str(date), "base": result["base_problem_signature"], "future_actual_used": False}),
              "objective_id": "LEX_RAW_QUANTITY_COST_GENERATOR_TIE" if p is None or p.service_metric == "LEGACY_RAW_QTY" else "LEX_MACRO_FULFILLMENT_COST_VHS_TIE",
              "allocation_policy": "WHOLE_REQUEST" if result["algorithm"] == "VARO" else "FINITE_PARTIAL_OPTIONS",
              "candidate_policy": policy, "algorithm": result["algorithm"], "data_class": result["evidence_class"],
              "solver_status": "OPTIMAL_FINITE_MODEL" if result["solver"].get("optimality_proven") else result["solver"].get("status", "FEASIBLE_NOT_PROVEN_OPTIMAL"),
              "feasibility_status": "VALID" if not result["violations"] else "INVALID",
              "baseline_id": baseline, "runtime_kind": runtime_kind, "runtime_seconds": result["seconds"], "evidence_path": str(evidence)}
    fulfillment = list(result["fulfillment"].values())
    metrics = [("primary_service", result["service"], "raw_quantity_UNCONFIRMED" if common["objective_id"].startswith("LEX_RAW") else "ratio"),
               ("raw_quantity", result["quantity"], "raw_quantity_UNCONFIRMED" if result["evidence_class"] == "PROXY" else "EA"),
               ("computed_transport_cost", result["cost"], "KRW_PROXY" if result["evidence_class"] == "PROXY" else "SCENARIO_CURRENCY"),
               ("macro_fulfillment", sum(fulfillment) / max(1, len(fulfillment)), "ratio"),
               ("minimum_need_fulfillment", min(fulfillment, default=0), "ratio"), ("active_routes", result["routes"], "count")]
    if p is not None:
        metrics.append(("raw_need_weighted_fulfillment", result["quantity"] / max(1e-30, sum(p.needs.values())), "ratio_PROXY" if p.evidence_class == "PROXY" else "ratio"))
    return [{**common, "metric_name": name, "metric_value": value, "metric_unit": unit} for name, value, unit in metrics]


def greedy_decisions(p, partial):
    """Transparent replay of the unchanged raw greedy rule, including rejections."""
    ordered = sorted(p.candidates, key=lambda c: (-max(o.quantity for o in c.options),
        max(c.options, key=lambda o: o.quantity).cost, -c.vhs, c.candidate_id))
    plan = {}; decisions = []
    for c in ordered:
        options = sorted(c.options, key=lambda o: (-o.quantity, o.cost))
        largest = options[0]; selected = None
        before = evaluate(p, plan)
        whole_trial = evaluate(p, {**plan, c.candidate_id: largest.quantity})
        for option in options if partial else options[:1]:
            if not evaluate(p, {**plan, c.candidate_id: option.quantity})["violations"]:
                selected = option; break
        if selected is not None:
            plan[c.candidate_id] = selected.quantity
        after = evaluate(p, plan)
        decisions.append({"candidate_id": c.candidate_id, "request": largest.quantity,
                          "selected_quantity": selected.quantity if selected else 0,
                          "selected_cost": selected.cost if selected else 0,
                          "whole_rejection": whole_trial["violations"],
                          "remaining_route_slots_before": p.max_routes - before["routes"],
                          "remaining_route_slots_after": p.max_routes - after["routes"],
                          "resource_consumed": {k: (selected.quantity if selected else 0) * v for k, v in c.resources.items()},
                          "relevant_remaining_before": {k: before["remaining"][k] for k in c.resources},
                          "relevant_remaining_after": {k: after["remaining"][k] for k in c.resources}})
    return plan, decisions


def daily_cause(p, results):
    whole, wt = greedy_decisions(p, False); partial, pt = greedy_decisions(p, True)
    assert whole == results["VARO"]["plan"] and partial == results["PARTIAL"]["plan"]
    first = next((i for i, (w, v) in enumerate(zip(wt, pt)) if w["selected_quantity"] != v["selected_quantity"]), None)
    blocked = []
    for cid, qty in whole.items():
        if cid not in partial:
            ev = evaluate(p, {**partial, cid: qty})
            blocked.append({"candidate_id": cid, "quantity": qty, "blocked_by_final_partial": ev["violations"]})
    first_partial = pt[first] if first is not None else None
    return {"date": p.problem_id.split(":")[-1], "whole_decisions": wt, "partial_decisions": pt,
            "first_divergence_index": first, "first_whole": wt[first] if first is not None else None,
            "first_partial": first_partial, "later_whole_choices_blocked": blocked,
            "quantity_delta_partial_minus_whole": results["PARTIAL"]["quantity"] - results["VARO"]["quantity"],
            "quantity_delta_local_minus_whole": results["LOCAL"]["quantity"] - results["VARO"]["quantity"],
            "observed_partial_consumes_route_slot": bool(first_partial and 0 < first_partial["selected_quantity"] < first_partial["request"]),
            "observed_whole_cap_rejection": bool(first_partial and any(v.startswith("CAP:") for v in first_partial["whole_rejection"])),
            "local_search_scope": "remove one selected candidate, exclude that identity, greedy refill; no baseline seed or simultaneous two-route quantity reoptimization",
            "objective": "same raw planned quantity, then cost, then generator-score tie; not macro fulfillment",
            "not_causes": ["different tariff function", "different quantity unit", "cost-first acceptance sacrificing service"],
            "unmodeled_resources": ["shared vehicle capacity", "DC capacity", "operator-approved stock/need"]}


def synthetic_problem(seed, size, profile="BASE"):
    rng = random.Random(seed * 1000 + size); width = max(4, math.ceil(math.sqrt(size)))
    needs = {f"T{i}": float(rng.randint(4, 16)) for i in range(width)}
    capacities = {**needs, **{f"S{i}": float(rng.randint(3, 16)) for i in range(width)}, "DC1": 22., "DC2": 22., "VEHICLE": 34.}
    if profile == "DC_BOTTLENECK": capacities.update(DC1=4., DC2=4., VEHICLE=6.)
    lanes = [(s, t) for s in range(width) for t in range(width)]; rng.shuffle(lanes)
    candidates = []
    for i, (s, t) in enumerate(lanes[:size]):
        fixed, variable = rng.randint(4, 30), rng.randint(1, 9)
        candidates.append(Candidate(f"C{i:03}", "P", f"S{s}", f"T{t}", "CHILLED", "L", "EA",
            {f"S{s}": 1., f"T{t}": 1., f"DC{1+t%2}": 1., "VEHICLE": 1.}, f"T{t}",
            tuple(AllocationOption(q, fixed+variable*q) for q in (2., 4., 6.)), rng.random(),
            cost_basis="FIXED_PLUS_VARIABLE", lineage="prespecified second-batch synthetic case"))
    return SelectionProblem(f"second-{seed}-{size}-{profile}", capacities, needs, tuple(candidates), 5,
                            "SCENARIO", policy={"decision_time": "t0", "information_set_id": "synthetic-known-t0", "profile": profile})


def run(root: Path, output: Path, prior: Path):
    output.mkdir(parents=True, exist_ok=False); (output / "days").mkdir()
    source = root / "_CANDIDATE_GENERATION_VALIDATION/candidate_universe_rows.parquet"
    protected_paths = [source, prior / "plans.json", prior / "report.json", prior / "daily_results.csv"]
    before = {str(p): digest(p) for p in protected_paths}
    prereg = {"date_count": 31, "daily_policy": "ALL_VALID", "new_raw_budget_seconds": 5., "milp_budget_seconds": 5.,
              "raw_algorithms": ["VARO", "PARTIAL", "LOCAL", "IMPROVED_RAW", "MILP"],
              "protected_portfolio": ["VARO baseline", "PARTIAL", "LOCAL", "IMPROVED_RAW"],
              "milp_in_protected_portfolio": False,
              "new_stress_cases": [[seed, size, "BASE"] for seed in (61, 73, 89) for size in (12, 64, 196)] + [[73, 64, "DC_BOTTLENECK"]],
              "stress_budgets_seconds": [.05, .25, 1.], "stress_neighborhoods": [1, 2],
              "primary": "strict lexicographic primary service, computed cost, tie; preserve verified equal-information incumbent",
              "promotion": "no operational promotion; SCENARIO/PROXY evidence only",
              "same_time_note": "MILP/IMPROVED_RAW given equal wall-budget requests; initialization/final-validation overhead measured separately by actual seconds; not hard real-time",
              "runtime_kind": "wall clock per call; portfolio total is all supplied challenger calls plus gate, not gate alone",
              "memory_kind": "cumulative process peak MB, not isolated solver memory"}
    dump(output / "preregistration.json", prereg)
    historical = json.loads((prior / "plans.json").read_text(encoding="utf-8"))
    historical_long = []
    for r in historical:
        historical_long += metric_rows(r, run_id="historical_upgrade_v01_744", date=r["date"], evidence=prior / "plans.json",
                                      runtime_kind="historical_saved_execution", policy=r["policy"])
    pd.DataFrame(historical_long).to_csv(output / "historical_744_longform.csv", index=False, encoding="utf-8-sig")
    historical_index = {(r["date"], r["policy"], r["algorithm"]): r for r in historical}
    universe = pd.read_parquet(source); tariff = TariffOptions(root)
    @lru_cache(maxsize=None)
    def independent_mix(product, q):
        return _optimize_vehicle_mix(tariff.tables["weights"][product]["weight_kg"] * q, tariff.tables["vehicles"])
    rows = []; causes = []; longform = []; costs = []; inclusion = []; replay = []; paired = []
    start = time.perf_counter()
    for date, day in universe.groupby("snapshot_date", sort=True):
        p = frozen_day_problem(day, "ALL_VALID", tariff)
        whole_p = frozen_day_problem(day, "ALL_VALID", tariff, partial=False)
        results = {name: solve(p, name, time_limit=5) for name in ("VARO", "PARTIAL", "LOCAL", "MILP")}
        for name, result in results.items():
            old = historical_index[(date, "ALL_VALID", name)]
            replay.append({"date": date, "algorithm": name, "quantity_equal": old["quantity"] == result["quantity"],
                           "cost_equal": old["cost"] == result["cost"], "plan_equal": old["plan"] == result["plan"],
                           "objective_equal": (old["service"], old["cost"], old["vhs"]) == (result["service"], result["cost"], result["vhs"]),
                           "candidate_set_equal": old["candidate_set_signature"] == result["candidate_set_signature"]})
        results["IMPROVED_RAW"] = improve_selection(p, time_limit=5)
        candidates = {name: results[name] for name in ("PARTIAL", "LOCAL", "IMPROVED_RAW")}
        results["PROTECTED"] = protected_selection(p, baseline_problem=whole_p, baseline_plan=results["VARO"]["plan"], challengers=candidates)
        results["PROTECTED"]["portfolio_seconds"] = sum(results[x]["seconds"] for x in ("VARO", "PARTIAL", "LOCAL", "IMPROVED_RAW")) + results["PROTECTED"]["seconds"]
        inclusion.append({"date": date, **incumbent_compatibility(p, whole_p, results["VARO"]["plan"])})
        cause = daily_cause(p, results); evidence = output / "days" / (str(date) + ".json")
        dump(evidence, {"problem": asdict(p), "results": results, "cause": cause})
        causes.append({k: v for k, v in cause.items() if k not in ("whole_decisions", "partial_decisions", "later_whole_choices_blocked")})
        for name, result in results.items():
            independent = evaluate(p, result["plan"])
            assert all(result[k] == value for k, value in independent.items())
            row = {"date": date, "algorithm": name, **{k: result[k] for k in ("service", "quantity", "cost", "routes", "seconds")},
                   "portfolio_seconds": result.get("portfolio_seconds"), "violations": len(result["violations"]),
                   "proven": result["solver"]["optimality_proven"], "selected_source": result.get("selected_source"),
                   "fallback_used": result.get("fallback_used"), "baseline_retained": result.get("baseline_retained"),
                   "peak_process_mb": process_peak_working_set_mb(), "evidence_path": str(evidence)}
            rows.append(row)
            longform += metric_rows(result, run_id="second_all_valid_replay", date=date, evidence=evidence, runtime_kind="fresh_wall_clock", p=p)
        baseline = exact_objective(p, results["VARO"]["plan"])
        raw = exact_objective(p, results["IMPROVED_RAW"]["plan"])
        final = exact_objective(p, results["PROTECTED"]["plan"])
        paired.append({"date": date, "raw_vs_baseline": (raw > baseline) - (raw < baseline),
                       "protected_vs_baseline": (final > baseline) - (final < baseline),
                       "raw_same_plan": results["IMPROVED_RAW"]["plan"] == results["VARO"]["plan"],
                       "protected_same_plan": results["PROTECTED"]["plan"] == results["VARO"]["plan"]})
        for c in p.candidates:
            for o in c.options:
                mix = independent_mix(c.product, o.quantity)
                calculated = float(round(tariff.tables["routes"][(c.source, c.target)]["distance_km"] * mix["total_rate_per_km"]))
                costs.append({"date": date, "candidate_id": c.candidate_id, "quantity": o.quantity,
                              "milp_cost_vector_coefficient": o.cost, "independent_tariff_recalculation": calculated,
                              "difference": calculated - o.cost, "cost_basis": c.cost_basis,
                              "scope": "integer q=1..original request; independent integer vehicle mix per route"})
        print(date, "VARO", results["VARO"]["quantity"], "raw", results["IMPROVED_RAW"]["quantity"],
              "protected", results["PROTECTED"]["quantity"], "cost", results["PROTECTED"]["cost"], flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output / "daily_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(costs).to_csv(output / "quantity_cost_parity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(replay).to_csv(output / "prior_replay_parity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired).to_csv(output / "paired_quality.csv", index=False, encoding="utf-8-sig")
    dump(output / "daily_causes.json", causes); dump(output / "feasible_set_inclusion.json", inclusion)
    aggregate = frame.groupby("algorithm").agg(days=("date", "count"), quantity=("quantity", "sum"), cost=("cost", "sum"),
        seconds=("seconds", "sum"), max_seconds=("seconds", "max"), violations=("violations", "sum"), proven=("proven", "sum")).reset_index()
    aggregate.to_csv(output / "aggregate.csv", index=False, encoding="utf-8-sig")
    stress_rows = []; stress_details = []
    for seed, size, profile in prereg["new_stress_cases"]:
        p = synthetic_problem(seed, size, profile); baseline = solve(p, "VARO")
        for budget in prereg["stress_budgets_seconds"]:
            runs = {"MILP": solve(p, "MILP", time_limit=budget)}
            for removed in prereg["stress_neighborhoods"]:
                r = improve_selection(p, time_limit=budget, max_removed=removed)
                label = "IMPROVED_" + str(removed); runs[label] = r
                runs["PROTECTED_" + str(removed)] = protected_selection(p, baseline_plan=baseline["plan"], challengers={label: r})
            for label, r in {"VARO": baseline, **runs}.items():
                ev = evaluate(p, r["plan"])
                stress_rows.append({"case": p.problem_id, "seed": seed, "size": len(p.candidates), "profile": profile,
                                    "budget": budget, "method": label, "service": ev["service"], "cost": ev["cost"],
                                    "violations": len(ev["violations"]), "seconds": r["seconds"],
                                    "optimality_proven": r["solver"]["optimality_proven"],
                                    "versus_baseline": (exact_objective(p, r["plan"]) > exact_objective(p, baseline["plan"])) -
                                                       (exact_objective(p, r["plan"]) < exact_objective(p, baseline["plan"])),
                                    "fallback": r.get("fallback_used"), "peak_process_mb": process_peak_working_set_mb()})
                longform += metric_rows({**r, "algorithm": label}, run_id="second_stress", date=p.problem_id,
                    evidence=output / "stress_plans.json", runtime_kind="fresh_wall_clock", policy="DECLARED_SYNTHETIC", p=p)
            stress_details.append({"problem": asdict(p), "budget": budget, "results": {"VARO": baseline, **runs}})
        print("stress", p.problem_id, flush=True)
    pd.DataFrame(stress_rows).to_csv(output / "stress_results.csv", index=False, encoding="utf-8-sig")
    dump(output / "stress_plans.json", stress_details)
    pd.DataFrame(longform).to_csv(output / "new_longform.csv", index=False, encoding="utf-8-sig")
    after = {str(p): digest(p) for p in protected_paths}
    report = {"status": "COMPLETED", "historical_executions_indexed": len(historical), "new_daily_executions": len(rows),
              "prior_all_valid_replay": {key: sum(not x[key] for x in replay) for key in ("quantity_equal", "cost_equal", "plan_equal", "objective_equal", "candidate_set_equal")},
              "daily_aggregate": aggregate.to_dict("records"), "cost_parity_options": len(costs),
              "cost_parity_mismatches": sum(x["difference"] != 0 for x in costs),
              "finite_set_inclusion_days": sum(x["compatible"] for x in inclusion),
              "partial_service_worse_days": sum(x["quantity_delta_partial_minus_whole"] < 0 for x in causes),
              "local_service_worse_days": sum(x["quantity_delta_local_minus_whole"] < 0 for x in causes),
              "raw_better_days": sum(x["raw_vs_baseline"] > 0 for x in paired), "raw_equal_days": sum(x["raw_vs_baseline"] == 0 for x in paired),
              "raw_worse_days": sum(x["raw_vs_baseline"] < 0 for x in paired),
              "protected_better_days": sum(x["protected_vs_baseline"] > 0 for x in paired),
              "protected_worse_days": sum(x["protected_vs_baseline"] < 0 for x in paired),
              "fallback_days": int(frame.fallback_used.fillna(False).sum()),
              "baseline_retained_days": int(frame.baseline_retained.fillna(False).sum()),
              "portfolio_total_seconds": float(frame.portfolio_seconds.sum()),
              "new_synthetic_cases": len(prereg["new_stress_cases"]), "stress_rows": len(stress_rows),
              "stress_violations": sum(x["violations"] for x in stress_rows),
              "stress_protected_worse": sum(x["versus_baseline"] < 0 for x in stress_rows if x["method"].startswith("PROTECTED")),
              "protected_hashes_before": before, "protected_hashes_after": after, "protected_unchanged": before == after,
              "environment": {"python": sys.version, "platform": platform.platform(), "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in ("candidate_selection.py", "candidate_upgrade_experiments.py", "selection_second_audit.py")}},
              "elapsed_seconds": time.perf_counter() - start,
              "limitations": ["31 dependent frozen daily proxy problems, not operational experiments", "raw unit/state/lot/operator capacities unconfirmed",
                  "route-level tariff parity does not prove shared multi-product vehicle cost equality; economic joint model is separate",
                  "old 744 results retained as historical, not rerun except ALL_VALID 124", "stress data are prespecified synthetic, not new field observations",
                  "neighborhood method does not claim global optimality; finite MILP only exact when all lex stages proven"],
              "readiness": {"MODEL_CORRECTNESS": "PASS_WITH_FINITE_SCOPE", "DECISION_QUALITY": "SEE_RAW_AND_PROTECTED_RESULTS", "OPERATIONAL_EVIDENCE": "NOT_ESTABLISHED"}}
    dump(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\VARO_V2_REAL_DATA"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/upgrade_20261016/w2_final"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run(args.data_root, args.output, args.prior)
