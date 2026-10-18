"""Opt-in finite-scenario information value. Never changes a production decision.

Payoffs are comparable net values (larger is better), not observed profit.
With explicit probabilities this calculates EVSI for a perfectly revealed group.
Without probabilities it calculates a minimax-regret reduction, NOT an expectation.
The caller owns scenario completeness, payoff provenance and collection costs.
"""
from __future__ import annotations

from math import isfinite, fsum
from numbers import Real
from typing import Mapping, Sequence


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def prioritize_information(
    payoffs: Mapping[str, Mapping[str, float]],
    observations: Mapping[str, Sequence[str]],
    costs: Mapping[str, float],
    probabilities: Mapping[str, float] | None = None,
) -> dict:
    """Rank one-step information queries; no adaptive multi-query claim.

    Each query has one observation label per scenario in payoffs insertion order.
    Equal labels mean the scenarios remain indistinguishable after that query.
    Costs and payoff differences must have the same currency and decision horizon.
    The returned net value is a finite-scenario upper bound for that exact signal,
    not a field-effect estimate. Zero-probability states do not influence EVSI.
    """
    if not isinstance(payoffs, Mapping) or not 1 <= len(payoffs) <= 512:
        raise ValueError("Provide 1 to 512 scenarios")
    states = tuple(payoffs)
    if any(not isinstance(s, str) or not s.strip() for s in states):
        raise ValueError("Scenario labels must be nonempty text")
    first = payoffs[states[0]]
    if not isinstance(first, Mapping) or not 1 <= len(first) <= 64:
        raise ValueError("Provide 1 to 64 common actions")
    actions = tuple(sorted(first))
    if any(not isinstance(a, str) or not a.strip() for a in actions):
        raise ValueError("Action labels must be nonempty text")
    values = {}
    for s, row in payoffs.items():
        if not isinstance(row, Mapping) or set(row) != set(actions):
            raise ValueError("Every scenario must contain exactly the same actions")
        values[s] = {a: _number(row[a], "Payoff") for a in actions}
    if not isinstance(observations, Mapping) or len(observations) > 64:
        raise ValueError("Provide at most 64 queries")
    if set(costs) != set(observations):
        raise ValueError("Every query requires its own explicit collection cost")
    query_costs = {q: _number(c, "Collection cost") for q, c in costs.items()}
    if any(c < 0 for c in query_costs.values()):
        raise ValueError("Collection costs cannot be negative")
    groups = {}
    for q, labels in observations.items():
        if not isinstance(q, str) or not q.strip():
            raise ValueError("Query labels must be nonempty text")
        if isinstance(labels, (str, bytes)) or len(labels) != len(states):
            raise ValueError("Each query needs one observation per scenario")
        grouped = {}
        for s, label in zip(states, labels):
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Observation labels must be nonempty text")
            grouped.setdefault(label, []).append(s)
        groups[q] = grouped
    probs = None
    if probabilities is not None:
        if set(probabilities) != set(states):
            raise ValueError("Probabilities must match every scenario")
        probs = {s: _number(probabilities[s], "Probability") for s in states}
        if any(p < 0 for p in probs.values()) or abs(fsum(probs.values()) - 1) > 1e-9:
            raise ValueError("Probabilities must be nonnegative and sum to one")
    regret = {s: {a: max(values[s].values()) - values[s][a] for a in actions} for s in states}

    def choose(subset):
        if probs is not None:
            scores = {a: fsum(probs[s] * values[s][a] for s in subset) for a in actions}
            action = max(actions, key=lambda a: scores[a])
        else:
            scores = {a: max(regret[s][a] for s in subset) for a in actions}
            action = min(actions, key=lambda a: scores[a])
        return action, scores[action]

    baseline_action, baseline = choose(states)
    perfect = fsum(probs[s] * max(values[s].values()) for s in states) if probs is not None else 0.0
    upper_bound = perfect - baseline if probs is not None else baseline
    rows = []
    for q, grouped in groups.items():
        conditional = {label: choose(subset) for label, subset in sorted(grouped.items())
                       if probs is None or fsum(probs[s] for s in subset) > 0}
        score = fsum(v[1] for v in conditional.values()) if probs is not None else max(v[1] for v in conditional.values())
        gross = score - baseline if probs is not None else baseline - score
        # Clamp only roundoff; a material bound violation indicates a defect.
        if gross < -1e-7 or gross > upper_bound + 1e-7:
            raise ArithmeticError("Information bound violation")
        gross = max(0.0, min(gross, upper_bound))
        rows.append({"query": q, "gross_value_bound": gross, "collection_cost": query_costs[q],
                     "net_value_bound": gross - query_costs[q],
                     "conditional_actions": {k: v[0] for k, v in conditional.items()}})
    rows.sort(key=lambda row: (-row["net_value_bound"], row["query"]))
    return {"schema_version": 1, "evidence_class": "SCENARIO",
            "mode": "EXPECTED_INFORMATION" if probs is not None else "WORST_REGRET_REDUCTION",
            "baseline_action": baseline_action, "baseline_score": baseline,
            "perfect_information_bound": max(0.0, upper_bound), "queries": rows,
            "recommended_query": rows[0]["query"] if rows and rows[0]["net_value_bound"] > 1e-9 else None,
            "production_action_changed": False}


def seller_scenario_example():
    """81 explicit what-if states evaluated by the UNCHANGED Seller Loss engine.

    Actions are fixed: keep 20, discount 20 by 30%, or move exactly 4 and keep 16.
    An invariant checks that the engine does not grant hidden quantity recourse.
    Scenario probabilities are deliberately absent. Rates are hypotheses, not
    estimated causal price response or observed selling performance.
    """
    from itertools import product
    from services.seller_decision_validation import scenario_input, SCENARIO_DATASET
    from services.seller_loss_engine import known, evaluate_seller_decision, STRATEGIES, TRANSFER

    def field(value, money=False):
        return known(value, "SCENARIO_INPUT", "정보 우선순위 가상 범위", dataset=SCENARIO_DATASET,
                     currency="KRW" if money else None)

    queries = {name: [] for name in ("출발 점포 수요", "도착 점포 수요", "이동 비용", "할인 반응")}
    payoffs = {}
    records = []
    for i, (source, target, freight, uplift) in enumerate(product((1,3,5), (1,4,8), (0,3000,8000), (0,.5,1))):
        inp = scenario_input(source_daily_demand=field(source), target_daily_demand=field(target),
            transfer_cost=field(freight, True), promotion_uplift=field(uplift),
            source_surplus_cap=field(4), target_need_cap=field(4), route_capacity_qty=field(4))
        result = evaluate_seller_decision(inp)
        if set(result["comparable_strategies"]) != set(STRATEGIES):
            raise ArithmeticError("Example requires all three Seller Loss actions")
        if result["strategies"][TRANSFER]["quantities"]["moved_qty"] != 4:
            raise ArithmeticError("Information example must use one fixed transfer quantity")
        name = f"scenario_{i+1:02d}"
        losses = {s: result["strategies"][s]["expected_loss"] for s in STRATEGIES}
        payoffs[name] = {s: -loss for s, loss in losses.items()}
        records.append({"scenario": name, "source_daily_demand": source, "target_daily_demand": target,
                        "transfer_cost": freight, "promotion_uplift": uplift, "losses": losses, "move_quantity": 4})
        for key, value in zip(queries, (source, target, freight, uplift)):
            queries[key].append(str(value))
    return {"schema_version": 1, "evidence_class": "SCENARIO", "payoffs": payoffs,
            "observations": queries, "costs": dict(zip(queries, (100,100,100,100))),
            "probabilities": None, "records": records,
            "unit": "KRW", "assumptions": "20개 처리, 이동 대안은 4개 이동·16개 정상판매, 할인율 30%, 기존 Seller Loss 가상 기준 입력 사용. 확인비 100원은 예시 가정."}
