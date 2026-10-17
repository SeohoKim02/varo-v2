"""R2 reproducible raw evidence; never overwrite old experiment directories."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from services.candidate_economics import (ConfirmedFlow, CommittedTransfer, DemandMarket,
    PriceOption, Scenario, _canonical, solve_economics)
from services.candidate_economics_experiments import amount, demo_problem
from services.candidate_economics_policies import fair_baselines, evaluate_policies


def adaptive_problem(*, noisy=False, late=False):
    p = demo_problem("transfer")
    if noisy:
        scenarios = (Scenario("low_low", .4, {0: "LOW"}), Scenario("low_high", .1, {0: "LOW"}),
                     Scenario("high_low", .1, {0: "HIGH"}), Scenario("high_high", .4, {0: "HIGH"}))
    else:
        scenarios = (Scenario("low", .5, {1 if late else 0: "LOW"}), Scenario("high", .5, {1 if late else 0: "HIGH"}))
    high = {s.scenario_id: (10 if s.scenario_id.endswith("high") else 0) for s in scenarios}
    low = {sid: 10 - q for sid, q in high.items()}
    markets = (DemandMarket("target", "T", "P", "CHILLED", "EA", 2,
        (PriceOption("normal", amount(10), high),)),
        DemandMarket("source", "S", "P", "CHILLED", "EA", 2,
        (PriceOption("normal", amount(4), low),)))
    return replace(p, problem_id="adaptive_noisy" if noisy else "adaptive_late" if late else "adaptive_signal",
        horizon=3, routes=(replace(p.routes[0], departure_periods=(0, 1)),), markets=markets, scenarios=scenarios,
        lineage={**p.lineage, "signal_assumption": "Synthetic observed signal with declared distribution; not a measured predictive feature"})


def operating_cases():
    p = demo_problem("normal")
    incoming = replace(p, problem_id="scheduled_inbound", lots=(replace(p.lots[0], stock=0),),
        confirmed_flows=(ConfirmedFlow("receipt", "lot_A", "S", 1, 10, "CONFIRMED_INBOUND"),))
    reserved = replace(p, problem_id="reserved_outbound",
        confirmed_flows=(ConfirmedFlow("order", "lot_A", "S", 1, 4, "RESERVED_OUTBOUND"),))
    transit = demo_problem("multiperiod")
    transit = replace(transit, problem_id="already_in_transit", lots=(replace(transit.lots[0], stock=0),),
        committed_transfers=(CommittedTransfer("departed", "lot_A", "S_T", -1, 6, cost_already_paid=True),))
    committed = replace(demo_problem("transfer"), problem_id="committed_departure",
        committed_transfers=(CommittedTransfer("approved", "lot_A", "S_T", 0, 4),))
    capacity = replace(demo_problem("transfer"), problem_id="handling_processing_holding",
        handling_per_capacity={"S": amount(1, "KG"), "T": amount(2, "KG")},
        processing_capacity={("T", 1): 10}, holding_by_period={("lot_A", 0): amount(1)})
    penalty = replace(demo_problem("transfer"), problem_id="evidenced_shortfall_penalty",
        shortage_penalties={"target": replace(amount(3), reason="SCENARIO contractual per-unit shortfall fee")})
    infeasible = replace(p, problem_id="immutable_infeasible_order",
        confirmed_flows=(ConfirmedFlow("order", "lot_A", "S", 0, 11, "RESERVED_OUTBOUND"),))
    return (incoming, reserved, transit, committed, capacity, penalty, infeasible)


def independent_conservation(p, r):
    """Audit returned physical ledger without rebuilding solver constraints."""
    if r.get("expected_net_value") is None:
        return {"status": "NOT_EVALUABLE", "reason": r["status"]}
    routes = {x.route_id: x for x in p.routes}
    maximum = 0.0
    rows = []
    for s in p.scenarios:
        sid = s.scenario_id
        for lot in p.lots:
            for t in range(p.horizon):
                onhand = sum(x["quantity"] for x in r["inventory"] if x["scenario"] == sid and x["lot_id"] == lot.lot_id and x["period"] == t)
                transit = sum(x["quantity"] for x in r["in_transit"] if x.get("scenario", sid) == sid and x["lot_id"] == lot.lot_id and x["period"] == t)
                sold = sum(x["quantity"] for x in r["sales"] if x["scenario"] == sid and x["lot_id"] == lot.lot_id
                    and next(m.period for m in p.markets if m.market_id == x["market_id"]) <= t)
                waste = sum(x["disposal"] for x in r["inventory"] if x["scenario"] == sid and x["lot_id"] == lot.lot_id and x["period"] <= t)
                outgoing = sum(e.quantity for e in p.confirmed_flows if e.lot_id == lot.lot_id and e.period <= t and e.kind in {"RESERVED_OUTBOUND", "CONFIRMED_OUTBOUND"})
                available = lot.stock + sum(e.quantity for e in p.confirmed_flows if e.lot_id == lot.lot_id and e.period <= t and e.kind == "CONFIRMED_INBOUND")
                available += sum(e.quantity for e in p.committed_transfers if e.lot_id == lot.lot_id and e.departure_period < 0)
                difference = onhand + transit + sold + waste + outgoing - available
                maximum = max(maximum, abs(difference))
                rows.append({"scenario": sid, "lot_id": lot.lot_id, "period": t, "on_hand": onhand, "in_transit": transit,
                    "cumulative_sales": sold, "cumulative_disposal": waste, "cumulative_external_outbound": outgoing,
                    "initial_plus_external_supply": available, "difference": difference})
    return {"status": "PASS" if maximum < 1e-7 else "FAIL", "maximum_difference": maximum, "rows": rows}


def run(output: Path):
    if output.exists() and any(output.iterdir()): raise FileExistsError("Existing evidence is immutable")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    old_path = Path("outputs/upgrade_20261016/w3/final_v01/economic_experiments.json")
    old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest() if old_path.exists() else None
    names = ("normal", "discount", "transfer", "mixed", "no_transfer", "expiry_block", "shared_vehicle", "multiperiod", "uncertainty", "missing_cost")
    originals, comparisons, conservation = {}, {}, {}
    for name in names:
        p = demo_problem(name)
        originals[name] = solve_economics(p)
        comparisons[name] = fair_baselines(p)
        conservation[name] = independent_conservation(p, originals[name])
    operations = {}
    for p in operating_cases():
        r = solve_economics(p, allow_transfers=p.problem_id != "committed_departure")
        operations[p.problem_id] = {"problem": _canonical(p), "result": r, "independent_conservation": independent_conservation(p, r)}
    adaptive = {p.problem_id: evaluate_policies(p) for p in (adaptive_problem(), adaptive_problem(noisy=True), adaptive_problem(late=True))}
    uncertain = demo_problem("uncertainty")
    risk = {policy: solve_economics(uncertain, risk=policy) for policy in ("expected", "worst_case")}
    risk["interpretation"] = "Expected policy: expected 51 / worst 26; worst-case policy: expected 30 / worst 30. Objective trade-off, not unconditional +21 improvement."
    old_comparison = {}
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        old_comparison = {name: {"before": old["results"][name]["candidate"].get("expected_net_value"),
            "after": originals[name].get("expected_net_value"),
            "same": old["results"][name]["candidate"].get("expected_net_value") == originals[name].get("expected_net_value")} for name in names}
    code = {name: hashlib.sha256(Path("services", name).read_bytes()).hexdigest() for name in (
        "candidate_economics.py", "candidate_economics_extended.py", "candidate_economics_policies.py", "candidate_economics_second_experiments.py")}
    report = {"version": "R2-second-v0.1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SCENARIO_RESEARCH_ONLY", "production_approved": False, "old_results_sha256_before": old_hash,
        "old_results_sha256_after": hashlib.sha256(old_path.read_bytes()).hexdigest() if old_path.exists() else None,
        "old_ten_exact_value_comparison": old_comparison, "original_runs": originals, "fair_baselines": comparisons,
        "original_conservation": conservation, "operating_cases": operations, "adaptive_policy_cases": adaptive,
        "risk_tradeoff": risk, "algorithm_file_hashes": code, "runtime_seconds": time.perf_counter() - started,
        "limitations": ["All added financial inputs and demand responses are explicit SCENARIO assumptions",
            "No measured operator outcomes or causal elasticity", "No real actual-savings claim",
            "Signal distributions are supplied, not learned or independently verified",
            "No automatic promotion of candidate decisions or latest model changes",
            "OPEN_LOOP original planner retains original scenario-specific sales recourse; playback removes realised-future access",
            "Rolling unseen history returns INPUT_REQUIRED; no invented distribution or oracle fallback",
            "No general proof that the particular sequential/independent procedures are best within their algorithm classes"]}
    evidence = output / "economic_second_report.json"
    evidence.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["run_id", "case_id", "dataset_id", "decision_time", "base_problem_signature", "candidate_set_signature",
        "algorithm_signature", "information_set_id", "objective_id", "allocation_policy", "data_class", "solver_status",
        "feasibility_status", "metric_name", "metric_value", "metric_unit", "baseline_id", "runtime_kind", "runtime_seconds", "evidence_path"]
    rows = []
    algo_sig = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    for name, comparison in comparisons.items():
        for policy, result in comparison.get("results", {}).items():
            base = dict(zip(fields, [None] * len(fields)))
            base.update(run_id=f"R2-{name}-{policy}", case_id=name, dataset_id="R2_CONTROLLED_SCENARIOS",
                decision_time=0, base_problem_signature=originals[name]["problem_signature"], candidate_set_signature="FINITE_SUPPLIED_ROUTES_AND_PRICES",
                algorithm_signature=algo_sig, information_set_id="AS_OF_0_COMMON_PRIOR", objective_id="EXPECTED_NET_CASH_MAX",
                allocation_policy=policy, data_class="SCENARIO", solver_status=result["status"],
                feasibility_status="PASS" if result.get("constraint_violations") == 0 else "NOT_EVALUABLE",
                baseline_id="A", runtime_kind="SOLVER_WALL_SECONDS", runtime_seconds=result.get("runtime_seconds"),
                evidence_path=str(evidence.resolve()))
            for metric in ("expected_net_value", "worst_net_value", "transport_cost", "solver_gap", "bound_net_value"):
                if result.get(metric) is not None:
                    rows.append({**base, "metric_name": metric, "metric_value": result[metric],
                                 "metric_unit": "ratio" if metric == "solver_gap" else "KRW"})
            for k, row in enumerate(result.get("transfers", [])):
                rows.append({**base, "decision_time": row["departure_period"], "metric_name": f"transfer:{row['lot_id']}:{row['route_id']}",
                    "metric_value": row["quantity"], "metric_unit": "EA"})
            parts = result.get("accounting_components", result.get("scenario_cash_components", {}))
            for sid, values in parts.items():
                for key, value in values.items():
                    rows.append({**base, "metric_name": f"cash:{sid}:{key}", "metric_value": value, "metric_unit": "KRW"})
    with (output / "economic_long_form.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return {"report": str(evidence.resolve()), "long_form_rows": len(rows), "legacy_exact": sum(x["same"] for x in old_comparison.values()),
        "old_evidence_unchanged": report["old_results_sha256_before"] == report["old_results_sha256_after"],
        "fair_cases": len(comparisons), "operating_cases": len(operations), "adaptive_cases": len(adaptive), "seconds": report["runtime_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False))
