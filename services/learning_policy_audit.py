"""Candidate-only policy instrumentation. Does not train or write any model."""
from __future__ import annotations
from dataclasses import replace
import random
import time
import numpy as np
import torch
from services import candidate_dqn as dqn
from services.candidate_selection import evaluate, solve
from services.operating_snapshot import signature


def structural_signature(problem, *, include_values=True):
    """Relabelling-invariant local incidence multiset; not a graph-isomorphism proof.

    Topology deliberately ignores demand/cost values to expose repeated templates.
    Exact incidence overlap with numbers is reported separately from this proxy.
    """
    degrees = {k: sum(k in c.resources for c in problem.candidates) for k in problem.capacities}
    candidates = []
    for c in problem.candidates:
        resources = sorted((degrees[k], float(v), float(problem.capacities[k]) if include_values else None,
                            k in problem.needs) for k, v in c.resources.items())
        candidates.append({"resources": resources, "options": [(o.quantity, o.cost) for o in c.options] if include_values else len(c.options),
                           "need": problem.needs[c.need_key] if include_values else None})
    return signature({"candidates": sorted(candidates, key=signature), "routes": problem.max_routes,
                      "metric": problem.service_metric})


def inspect_policy(problem, *, policy, seed=17, model=None):
    """All policies observe the identical current finite allocation problem."""
    if policy == "TRAINED_DQN":
        return dqn.infer(model, problem)
    if policy == "UNTRAINED_DQN":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = dqn._network().eval()
        return dqn.infer(dqn.TrainedCandidate(network, {"seed": seed, "training_decision_fingerprints": []}), problem)
    if policy in {"VARO", "PARTIAL", "MILP"}:
        result = solve(problem, policy)
        return {"plan": result["plan"], "evaluation": evaluate(problem, result["plan"]),
                "trace": [], "seconds": result.get("seconds", 0.), "solver_result": result}
    if policy not in {"RANDOM_VALID", "GREEDY_IMMEDIATE"}:
        raise ValueError("unknown audit policy")
    env, rng = dqn.SelectionEnvironment(problem), random.Random(seed)
    trace, start = [], time.perf_counter()
    while not env.done:
        valid = env.valid_actions()
        if policy == "RANDOM_VALID":
            action = rng.choice(valid)
        else:
            def key(a):
                candidate, option = env.actions[a]
                if candidate is None:
                    return (0., 0., -a)
                from services.candidate_selection import _service
                return (_service(problem, candidate, option), -option.cost, -a)
            action = max(valid, key=key)
        before = dict(env.plan)
        reward, _ = env.step(action)
        trace.append({"action": action, "reward": reward, "valid_action_count": len(valid),
                      "state_plan": before, "valid_actions": valid, "invalid_action_attempts": 0})
    return {"plan": env.plan, "evaluation": evaluate(problem, env.plan), "trace": trace,
            "seconds": time.perf_counter()-start}


def rolling_allocation_bridge(model, problem, observed_demands, *, policy="TRAINED_DQN"):
    """Research bridge to persistent inventory, not retraining in a multi-period MDP.

    Initial need is the day-0 forecast. Subsequent forecasts use only prior
    observed sales. Realized observations are exposed after the allocation.
    Source stock and received target inventory persist; no future MILP oracle.
    """
    from services.candidate_selection import AllocationOption
    targets = set(problem.needs)
    sources = {c.source for c in problem.candidates}
    stock = {k: float(problem.capacities[k]) for k in sources}
    inventory = {k: 0. for k in targets}
    history, records = [], []
    for period, demand in enumerate(observed_demands):
        forecast = dict(problem.needs) if not history else {k: float(np.mean([x[k] for x in history])) for k in targets}
        needs = {k: max(0., float(np.floor(forecast[k]-inventory[k]))) for k in targets}
        candidates = tuple(c for c in problem.candidates if needs[c.need_key] > 0 and stock[c.source] > 0)
        capacities = {**problem.capacities, **stock, **needs}
        current = replace(problem, problem_id=f"{problem.problem_id}:period:{period}", capacities=capacities,
                          needs=needs, candidates=candidates)
        result = inspect_policy(current, policy=policy, model=model)
        for cid, quantity in result["plan"].items():
            c = next(c for c in candidates if c.candidate_id == cid)
            stock[c.source] -= quantity
            inventory[c.target] += quantity
        sold = {k: min(inventory[k], demand[k]) for k in targets}
        for k in targets:
            inventory[k] -= sold[k]
        records.append({"period": period, "forecast": forecast, "need": needs,
                        "plan": result["plan"], "cost": result["evaluation"]["cost"],
                        "sales_proxy": sold, "source_stock": dict(stock), "target_inventory": dict(inventory),
                        "information_cutoff": period-1, "violations": result["evaluation"]["violations"]})
        history.append(dict(demand))
    return {"records": records, "total_sales_proxy": sum(sum(r["sales_proxy"].values()) for r in records),
            "total_cost": sum(r["cost"] for r in records), "scope": "SCENARIO_ROLLING_INVENTORY_BRIDGE",
            "trained_for_multiperiod": False, "production_approved": False}
