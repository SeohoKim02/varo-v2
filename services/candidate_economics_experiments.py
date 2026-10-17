"""Small controlled W3 experiments, explicitly SCENARIO rather than real outcomes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import sys
import time

import scipy

from services.candidate_economics import (
    DemandMarket, EconomicProblem, InventoryLot, Money, PriceOption, Route,
    Scenario, solve_economics,
)
from services.seller_loss_engine import known


def amount(value: float, unit: str | None = "EA") -> Money:
    return Money(known(value, "SCENARIO_INPUT", "controlled W3 scenario", unit=unit,
                       currency="KRW", dataset="W3_SCENARIO"), "KNOWN")


def demo_problem(name: str = "mixed") -> EconomicProblem:
    """Public example for candidate app integration; never represents Suhyup prices."""
    lot = InventoryLot("lot_A", "P", "S", "CHILLED", "EA", 10, 3, 1,
                       amount(0), amount(0), amount(0))
    route = Route("S_T", "S", "T", 1, 10, 2, amount(2, None), amount(0, "KG"), ("CHILLED",))
    source = DemandMarket("source", "S", "P", "CHILLED", "EA", 0, (
        PriceOption("normal", amount(10), {"base": 2}),
        PriceOption("discount_40pct", amount(6), {"base": 6})))
    target = DemandMarket("target", "T", "P", "CHILLED", "EA", 1,
                          (PriceOption("normal", amount(10), {"base": 4}),))
    p = EconomicProblem(name, 2, "KRW", (lot,), (route,), (source, target),
                        (Scenario("base", 1),), "KG", {"S": 20, "T": 20},
                        lineage={"source": "synthetic hand-calculable scenario", "capacity": "1 EA = 1 KG assumed"})
    if name == "normal":
        p = replace(p, markets=(replace(source, options=(PriceOption("normal", amount(10), {"base": 10}),)), target))
    elif name == "discount":
        p = replace(p, routes=(replace(route, fixed_per_vehicle=amount(30, None)),), markets=(
            replace(source, options=(source.options[0], PriceOption("discount_20pct", amount(8), {"base": 10}))), target))
    elif name == "transfer":
        p = replace(p, markets=(replace(source, options=(PriceOption("normal", amount(10), {"base": 0}),)),
                               replace(target, options=(PriceOption("normal", amount(12), {"base": 10}),))))
    elif name == "no_transfer":
        p = replace(p, routes=(replace(route, fixed_per_vehicle=amount(200, None)),))
    elif name == "expiry_block":
        p = replace(p, lots=(replace(lot, expiry_period=0),))
    elif name == "missing_cost":
        p = replace(p, routes=(replace(route, fixed_per_vehicle=Money(reason="운송 견적 미확인")),))
    elif name == "shared_vehicle":
        lots = tuple(replace(lot, lot_id=f"lot_{pid}", product_id=pid, stock=5) for pid in ("P", "Q"))
        markets = tuple(DemandMarket(f"target_{pid}", "T", pid, "CHILLED", "EA", 1,
                                    (PriceOption("normal", amount(10), {"base": 5}),)) for pid in ("P", "Q"))
        p = replace(p, lots=lots, markets=markets, routes=(replace(route, fixed_per_vehicle=amount(10, None)),))
    elif name == "multiperiod":
        p = replace(p, horizon=3, lots=(replace(lot, stock=6),),
                    routes=(replace(route, transit_periods=2),),
                    markets=(replace(target, period=2, options=(PriceOption("normal", amount(10), {"base": 6}),)),))
    elif name == "uncertainty":
        p = replace(p, scenarios=(Scenario("low", .5), Scenario("high", .5)),
                    routes=(replace(route, fixed_per_vehicle=amount(4, None)),), markets=(
                        replace(source, options=(PriceOption("normal", amount(6), {"low": 5, "high": 5}),)),
                        replace(target, options=(PriceOption("normal", amount(10), {"low": 0, "high": 10}),))))
    elif name != "mixed":
        raise ValueError(name)
    return p


def brute_force_two_period(p: EconomicProblem) -> dict:
    """Independent enumeration for one lot, one route and two markets only.

    This intentionally does not use MILP rows or its variable builder.
    All supplied fixtures have zero holding/disposal/terminal value and no expiry.
    """
    lot, route = p.lots[0], p.routes[0]
    source, target = sorted(p.markets, key=lambda m: m.period)
    best = -math.inf
    plans = []
    for k in range(round(lot.stock / lot.quantum) + 1):
        q = k * lot.quantum
        n = math.ceil(q * lot.capacity_per_unit / route.vehicle_capacity - 1e-10)
        if n > route.max_vehicles: continue
        cost = n * route.fixed_per_vehicle.field.value + q * lot.capacity_per_unit * route.variable_per_capacity.field.value
        for option in source.options:
            for other in target.options:
                value = sum(s.probability * (min(lot.stock - q, option.demand[s.scenario_id])
                    * option.unit_price.field.value + min(q, other.demand[s.scenario_id])
                    * other.unit_price.field.value - cost) for s in p.scenarios)
                if value > best + 1e-9:
                    best, plans = value, []
                if abs(value - best) < 1e-9:
                    plans.append({"move_quantity": q, "source_level": option.level})
    return {"net_value": best, "optimal_plans": plans}


def run_experiments(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Existing experiment evidence will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    names = ("normal", "discount", "transfer", "mixed", "no_transfer", "expiry_block",
             "shared_vehicle", "multiperiod", "uncertainty", "missing_cost")
    results = {}
    comparisons = []
    for name in names:
        p = demo_problem(name)
        candidate = solve_economics(p)
        baseline = solve_economics(p, allow_transfers=False)
        results[name] = {"candidate": candidate, "no_transfer_baseline": baseline}
        comparisons.append({"problem": name, "evidence": "SCENARIO", "baseline": "joint pricing, no transfer",
            "candidate": "joint finite-price and transfer MILP", "metric": "expected prospective net cash value (KRW)",
            "baseline_value": baseline["expected_net_value"], "candidate_value": candidate["expected_net_value"],
            "difference": None if candidate["expected_net_value"] is None else candidate["expected_net_value"] - baseline["expected_net_value"],
            "constraint_violations": candidate.get("constraint_violations"), "n_independent_problems": 1,
            "runtime_seconds": candidate.get("runtime_seconds"), "interpretation": "Synthetic scenario, not actual realised savings"})
        if name in {"mixed", "normal", "discount", "transfer", "no_transfer", "uncertainty"}:
            results[name]["independent_enumeration"] = brute_force_two_period(p)
    uncertain = demo_problem("uncertainty")
    results["uncertainty"]["worst_case"] = solve_economics(uncertain, risk="worst_case")
    perfect_info = []
    for s in uncertain.scenarios:
        p = replace(uncertain, scenarios=(replace(s, probability=1),), markets=tuple(
            replace(m, options=tuple(replace(o, demand={s.scenario_id: o.demand[s.scenario_id]})
                                     for o in m.options)) for m in uncertain.markets))
        perfect_info.append({"scenario": s.scenario_id, "probability": s.probability, "result": solve_economics(p)})
    results["uncertainty"]["perfect_information_upper_bound_not_deployable"] = perfect_info
    report = {"version": "W3-v0.1", "status": "SCENARIO_RESEARCH_ONLY", "production_approved": False,
        "hypotheses_predefined": ["Joint allocation can outperform no-transfer pricing under fixed costs",
            "Shared fixed cost should be paid once per vehicle, not per product",
            "Expiry can remove an otherwise profitable route",
            "Worst-case policy can trade expected value for downside protection"],
        "selection_criteria": "Max net cash; expected or worst-case risk fixed before solving",
        "data": "Ten synthetic hand-calculable cases; zero actual pilot observations",
        "versions": {"python": platform.python_version(), "scipy": scipy.__version__},
        "comparisons": comparisons, "results": results, "runtime_seconds": time.perf_counter() - start,
        "limitations": ["Not a measured real-world intervention", "Open-loop decisions, not adaptive multistage optimization",
            "Known zero costs are explicit scenario assumptions; real missing costs prevent optimization",
            "Finite supplied demand responses; no estimated price elasticity",
            "Existing Seller Loss A/B/C unchanged and independently validated elsewhere"]}
    (output / "economic_experiments.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/upgrade_20261016/w3"))
    args = parser.parse_args()
    report = run_experiments(args.output)
    print(json.dumps({"status": report["status"], "cases": len(report["comparisons"]),
                      "output": str(args.output.resolve()), "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False))
