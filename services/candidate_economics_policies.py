"""Fair baselines and observed-only policy playback for candidate economics.

Planning distributions are frozen separately from realised paths. Playback reads
only the prefix delivered at each event; it is NOT an historical sales replay.
Unseen signals fail visibly rather than obtaining a future oracle forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import Mapping

from services.candidate_economics import EconomicProblem, Scenario, solve_economics
from services.candidate_economics_extended import history_key


def _move_keys(p):
    return {(lot.lot_id, r.route_id, t): 0.0 for lot in p.lots for r in p.routes for t in r.departure_periods
            if t + r.transit_periods < p.horizon and t + r.transit_periods <= lot.expiry_period and lot.state in r.allowed_states}


def _move_map(result, p, scenario=None):
    answer = _move_keys(p)
    rows = result.get("scenario_transfers", result.get("transfers", []))
    for row in rows:
        if "scenario" in row and row["scenario"] != (scenario or p.scenarios[0].scenario_id):
            continue
        answer[row["lot_id"], row["route_id"], row["departure_period"]] = row["quantity"]
    return answer


def fair_baselines(p: EconomicProblem, *, time_limit=30.0) -> dict:
    """Five declared treatments, all seeing the same underlying input contract.

    C commits the normal-price economic move plan before optimising price; D
    proposes routes independently, then jointly repairs quantities under the
    original resource constraints. Their decision rules are disclosed, not
    represented as globally optimal sequential/independent algorithms.
    """
    missing = [m.market_id for m in p.markets if not any(o.level == "normal" for o in m.options)]
    if missing:
        return {"status": "INPUT_REQUIRED", "reason": "NORMAL_PRICE_LEVEL_REQUIRED", "markets": missing}
    normal = replace(p, markets=tuple(replace(m, options=tuple(o for o in m.options if o.level == "normal")) for m in p.markets))
    options = {"time_limit": time_limit}
    a = solve_economics(normal, allow_transfers=False, **options)
    b = solve_economics(p, allow_transfers=False, **options)
    stage_one = solve_economics(normal, **options)
    c = solve_economics(p, fixed_transfers=_move_map(stage_one, p), **options) if stage_one.get("expected_net_value") is not None else stage_one
    independent = []
    upper = _move_keys(p)
    available = {lot.lot_id: lot.stock
        + sum(e.quantity for e in p.confirmed_flows if e.lot_id == lot.lot_id and e.kind == "CONFIRMED_INBOUND")
        + sum(e.quantity for e in p.committed_transfers if e.lot_id == lot.lot_id and e.departure_period < 0) for lot in p.lots}
    for route in p.routes:
        # Other routes are prohibited, not removed: immutable commitments survive.
        permitted = {key: available[key[0]] for key in upper if key[1] == route.route_id}
        single = solve_economics(p, transfer_upper_bounds=permitted, **options)
        independent.append({"route_id": route.route_id, "result": single})
        if single.get("expected_net_value") is not None:
            for key, value in _move_map(single, p).items(): upper[key] = max(upper[key], value)
    d = solve_economics(p, transfer_upper_bounds=upper, **options)
    e = solve_economics(p, **options)
    return {"status": "COMPLETED", "evidence_class": p.evidence_class, "problem_id": p.problem_id,
        "baseline_definitions": {"A": "Normal price only, no new transfers", "B": "All supplied prices, no new transfers",
            "C": "Normal-price economic transfers fixed first, then all supplied prices optimised",
            "D": "Each route proposed independently using all prices, then common-constraint quantity/price repair",
            "E": "Joint supplied-price and transfer optimisation"},
        "same_input_scope": "Same stock, supplied demand responses, period, prices, costs and physical constraints; only decision procedure/allowed actions vary",
        "results": {"A": a, "B": b, "C": c, "D": d, "E": e},
        "sequential_stage_one": stage_one, "independent_proposals": independent,
        "interpretation": "A to B isolates discount permission; B to E includes transfer permission and cannot by itself prove superiority to C/D",
        "production_approved": False}


@dataclass(frozen=True)
class RealizedPath:
    path_id: str
    probability: float
    signals: Mapping[int, str]
    # market_id -> price level -> assumed potential demand (not observed sales)
    demand: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class ObservedPrefix:
    period: int
    signals: Mapping[int, str]
    demand: Mapping[str, Mapping[str, float]]
    market_periods: Mapping[str, int]

    def demand_at(self, market_id: str):
        if self.market_periods[market_id] > self.period:
            raise ValueError("FUTURE_OBSERVATION_NOT_AVAILABLE")
        return self.demand[market_id]


def observed_prefix(p: EconomicProblem, path: RealizedPath, period: int) -> ObservedPrefix:
    """Simulation environment exposes a prefix, never the realised future."""
    periods = {m.market_id: m.period for m in p.markets}
    return ObservedPrefix(period, {t: value for t, value in path.signals.items() if t <= period},
        {mid: dict(values) for mid, values in path.demand.items() if periods[mid] <= period}, periods)


def scenario_paths(p: EconomicProblem) -> tuple[RealizedPath, ...]:
    return tuple(RealizedPath(s.scenario_id, s.probability, dict(s.observations),
        {m.market_id: {o.level: o.demand[s.scenario_id] for o in m.options} for m in p.markets}) for s in p.scenarios)


def _condition(p: EconomicProblem, prefix: ObservedPrefix) -> EconomicProblem | None:
    remaining = [s for s in p.scenarios if all(s.observations.get(t) == v for t, v in prefix.signals.items())]
    if not remaining: return None
    total = sum(s.probability for s in remaining)
    ids = {s.scenario_id for s in remaining}
    # This assignment only handles observed periods. Future demand stays in the
    # frozen planning distribution, even if the path evaluator knows its future.
    markets = tuple(replace(m, options=tuple(replace(o, demand={sid:
        prefix.demand_at(m.market_id)[o.level] if m.period <= prefix.period else o.demand[sid]
        for sid in ids}) for o in m.options)) for m in p.markets)
    return replace(p, scenarios=tuple(replace(s, probability=s.probability / total) for s in remaining), markets=markets)


def evaluate_policy_on_path(p: EconomicProblem, policy: str, path: RealizedPath, *, frozen_plan=None, time_limit=30.0) -> dict:
    """Execute only current actions; replan using observations available so far.

    Sales may be withheld according to an economic plan (e.g. terminal value),
    consistently with the original model; they are not forced to historical
    recorded sales. Actual observed outcomes are never written by this function.
    """
    if policy not in {"OPEN_LOOP", "ROLLING_HORIZON", "MULTISTAGE_RECOURSE", "PERFECT_INFORMATION"}:
        raise ValueError("Unknown policy")
    started = time.perf_counter()
    if policy == "PERFECT_INFORMATION":
        sid = "oracle"
        oracle = replace(p, scenarios=(Scenario(sid, 1, dict(path.signals)),), markets=tuple(replace(m,
            options=tuple(replace(o, demand={sid: path.demand[m.market_id][o.level]}) for o in m.options)) for m in p.markets))
        result = solve_economics(oracle, policy="PERFECT_INFORMATION", time_limit=time_limit)
        return {"policy": policy, "path_id": path.path_id, "status": result["status"], "result": result,
                "net_cash_value": result.get("expected_net_value"), "operational_policy": False,
                "runtime_seconds": time.perf_counter() - started}
    plan = frozen_plan
    if plan is None and policy != "ROLLING_HORIZON":
        plan = solve_economics(p, policy="MULTISTAGE_RECOURSE" if policy == "MULTISTAGE_RECOURSE" else "OPEN_LOOP", time_limit=time_limit)
    fixed_moves, fixed_prices, fixed_sales = {}, {}, {}
    trajectory = []
    post = None
    for t in range(p.horizon):
        prior = observed_prefix(p, path, t - 1)
        conditional = _condition(p, prior)
        if conditional is None:
            return {"policy": policy, "path_id": path.path_id, "status": "INPUT_REQUIRED", "reason": "UNSEEN_OBSERVATION_HISTORY",
                    "trajectory": trajectory, "net_cash_value": None, "production_approved": False}
        if policy == "ROLLING_HORIZON":
            plan = solve_economics(conditional, policy="OPEN_LOOP", fixed_transfers=fixed_moves,
                fixed_prices=fixed_prices, fixed_sales=fixed_sales, time_limit=time_limit)
        if plan is None or plan.get("expected_net_value") is None:
            return {"policy": policy, "path_id": path.path_id, "status": "NO_SOLUTION", "trajectory": trajectory,
                    "net_cash_value": None, "details": plan}
        sid = conditional.scenarios[0].scenario_id
        if policy == "OPEN_LOOP": sid = p.scenarios[0].scenario_id
        current_moves = {k: v for k, v in _move_map(plan, p, sid).items() if k[2] == t}
        fixed_moves.update(current_moves)
        rows = plan.get("scenario_price_levels", plan.get("price_levels", []))
        current_prices = {}
        for row in rows:
            if "scenario" in row and row["scenario"] != sid: continue
            market = next(m for m in p.markets if m.market_id == row["market_id"])
            if market.period == t:
                current_prices[market.market_id, t] = row["level"]
        fixed_prices.update(current_prices)
        observed = observed_prefix(p, path, t)
        # Current demand is observed while selling. The separate end-of-period
        # signal is not available until NEXT period and cannot choose today's lot.
        sale_observed = replace(observed, signals={k: v for k, v in observed.signals.items() if k < t})
        conditional_after = _condition(p, sale_observed)
        if conditional_after is None:
            return {"policy": policy, "path_id": path.path_id, "status": "INPUT_REQUIRED", "reason": "UNSEEN_OBSERVATION_HISTORY",
                    "trajectory": trajectory + [{"period": t, "moves": list(current_moves.items()), "prices": list(current_prices.items())}],
                    "net_cash_value": None, "production_approved": False}
        # Freeze all open-loop future commitments, not just actions executed yet.
        post_moves, post_prices = dict(fixed_moves), dict(fixed_prices)
        if policy == "OPEN_LOOP":
            post_moves.update(_move_map(plan, p, sid))
            post_prices.update({(row["market_id"], next(m.period for m in p.markets if m.market_id == row["market_id"])): row["level"]
                for row in plan.get("price_levels", [])})
        post = solve_economics(conditional_after, policy="MULTISTAGE_RECOURSE", fixed_transfers=post_moves,
            fixed_prices=post_prices, fixed_sales=fixed_sales, time_limit=time_limit)
        if post.get("expected_net_value") is None:
            return {"policy": policy, "path_id": path.path_id, "status": "INFEASIBLE_POLICY", "trajectory": trajectory,
                    "net_cash_value": None, "details": post}
        representative = conditional_after.scenarios[0].scenario_id
        current_sales = {(l.lot_id, m.market_id, o.level): 0.0 for l in p.lots for m in p.markets if m.period == t for o in m.options}
        for row in post["sales"]:
            if row["scenario"] == representative and (row["lot_id"], row["market_id"], row["level"]) in current_sales:
                current_sales[row["lot_id"], row["market_id"], row["level"]] = row["quantity"]
        fixed_sales.update(current_sales)
        trajectory.append({"period": t, "information_set": json.dumps(sorted(prior.signals.items())),
            "available_demand_period_max": prior.period, "moves": [{"lot_id": lid, "route_id": rid, "period": period, "quantity": q}
                for (lid, rid, period), q in current_moves.items()],
            "prices": [{"market_id": mid, "period": period, "level": level} for (mid, period), level in current_prices.items()],
            "sales": [{"lot_id": lid, "market_id": mid, "level": level, "quantity": q} for (lid, mid, level), q in current_sales.items()],
            "inventory": [row for row in post["inventory"] if row["scenario"] == representative and row["period"] == t],
            "constraint_violations": post["constraint_violations"]})
    # No decision follows the final end-of-period signal. All realised sales and
    # actions are now fixed, so no final future-information conditioning is needed.
    representative = conditional_after.scenarios[0].scenario_id
    return {"policy": policy, "path_id": path.path_id, "status": "COMPLETED", "operational_policy": True,
        "trajectory": trajectory, "net_cash_value": post["scenario_net_values"][representative],
        "accounting_components": post["accounting_components"][representative],
        "constraint_violations": sum(step["constraint_violations"] for step in trajectory),
        "runtime_seconds": time.perf_counter() - started, "production_approved": False,
        "interpretation": "Supplied latent-demand scenario playback; not causal elasticity or actual operating profit"}


def evaluate_policies(p: EconomicProblem, *, realized_paths=None, time_limit=30.0) -> dict:
    paths = tuple(realized_paths or scenario_paths(p))
    if abs(sum(x.probability for x in paths) - 1) > 1e-9: raise ValueError("Path probabilities must sum to one")
    plans = {policy: solve_economics(p, policy=policy, time_limit=time_limit) for policy in ("OPEN_LOOP", "MULTISTAGE_RECOURSE")}
    results = {}
    for policy in ("OPEN_LOOP", "ROLLING_HORIZON", "MULTISTAGE_RECOURSE", "PERFECT_INFORMATION"):
        rows = [evaluate_policy_on_path(p, policy, path, frozen_plan=plans.get(policy), time_limit=time_limit) for path in paths]
        good = all(row.get("net_cash_value") is not None for row in rows)
        results[policy] = {"paths": rows, "expected_net_value": sum(path.probability * row["net_cash_value"] for path, row in zip(paths, rows)) if good else None,
            "worst_net_value": min(row["net_cash_value"] for row in rows) if good else None,
            "status": "COMPLETED" if good else "INCOMPLETE", "operational_policy": policy != "PERFECT_INFORMATION"}
    return {"problem_id": p.problem_id, "evidence_class": "SCENARIO", "frozen_planning_signature": plans["OPEN_LOOP"]["problem_signature"],
        "results": results, "plans": plans, "production_approved": False,
        "interpretation": "All policies evaluated on identical exogenous latent-demand paths, not fixed historical sales"}
