"""Read-only research diagnostics around unchanged finite-allocation solvers.

No operational recommendation, data writer, model loading or price estimation.
Candidate identities and quantity options stay fixed during perturbations.
"""
from dataclasses import replace
from fractions import Fraction
import math

from services.candidate_selection import validate


def perturb_problem(problem, factor, relative_change, *, candidate_id=None):
    """One-factor declared-input sensitivity, not a new observed data set."""
    if isinstance(relative_change, bool) or not math.isfinite(relative_change) or abs(relative_change) > .5:
        raise ValueError("relative change must be finite and within +/-50%")
    multiplier = 1 + relative_change
    capacities, needs = dict(problem.capacities), dict(problem.needs)
    candidates = problem.candidates
    if factor == "source_capacity":
        keys = [k for k in capacities if k.startswith("source:")]
        if not keys:
            raise ValueError("explicit source capacity keys required")
        for key in keys:
            capacities[key] *= multiplier
    elif factor == "target_need":
        for key in needs:
            needs[key] *= multiplier
            capacities[key] *= multiplier
    elif factor in {"selected_route_cost", "all_route_cost"}:
        if factor == "selected_route_cost" and candidate_id not in {c.candidate_id for c in candidates}:
            raise ValueError("selected route identity required")
        candidates = tuple(replace(c, options=tuple(replace(o, cost=o.cost * multiplier) for o in c.options))
                           if factor == "all_route_cost" or c.candidate_id == candidate_id else c for c in candidates)
    else:
        raise ValueError("unsupported sensitivity factor")
    result = replace(problem, capacities=capacities, needs=needs, candidates=candidates,
                     policy={**problem.policy, "research_perturbation": {"factor": factor, "relative_change": relative_change,
                                                                      "candidate_id": candidate_id, "fixed_candidate_pool": True}})
    validate(result)
    return result


def independently_account(problem, plan):
    """Recompute resource usage and all objective terms without solver evaluate()."""
    by_id = {c.candidate_id: c for c in problem.candidates}
    used = {key: Fraction(0) for key in problem.capacities}
    service = cost = tie = quantity = Fraction(0)
    violations = []
    positive_needs = sum(v > 0 for v in problem.needs.values())
    if len(plan) > problem.max_routes:
        violations.append("MAX_ROUTES")
    for key, value in plan.items():
        c = by_id.get(key)
        option = next((o for o in c.options if o.quantity == value), None) if c else None
        if option is None:
            violations.append("UNKNOWN_ALLOCATION:" + key)
            continue
        q = Fraction(str(value))
        quantity += q
        cost += Fraction(str(option.cost))
        service += q if problem.service_metric == "LEGACY_RAW_QTY" else q / Fraction(str(problem.needs[c.need_key])) / max(1, positive_needs)
        tie += Fraction(str(c.vhs)) * q / Fraction(str(problem.needs[c.need_key]))
        for resource, coefficient in c.resources.items():
            used[resource] += q * Fraction(str(coefficient))
    violations.extend("CAP:" + key for key, value in used.items()
                      if value - Fraction(str(problem.capacities[key])) > Fraction(1, 10_000_000))
    return {"quantity": float(quantity), "service": float(service), "cost": float(cost), "vhs": float(tie),
            "objective": (service, -cost, tie), "violations": violations,
            "resource_used": {key: float(value) for key, value in used.items()}}


def compare_plans(problem, previous_plan, current_plan):
    """Evaluate both plans under the *new* inputs; infeasibility is not regret."""
    previous = independently_account(problem, previous_plan)
    current = independently_account(problem, current_plan)
    if current["violations"]:
        raise ValueError("current sensitivity plan is infeasible")
    before, after = set(previous_plan), set(current_plan)
    union = before | after
    previous_feasible = not previous["violations"]
    same_service = previous_feasible and previous["objective"][0] == current["objective"][0]
    return {"exact_plan_same": dict(previous_plan) == dict(current_plan), "route_set_same": before == after,
            "route_jaccard": len(before & after) / len(union) if union else 1.,
            "quantity_reallocated": sum(abs(previous_plan.get(k, 0) - current_plan.get(k, 0)) for k in union),
            "previous_plan_feasible": previous_feasible, "previous_plan_violations": previous["violations"],
            "equivalent_objective": previous_feasible and previous["objective"] == current["objective"],
            "service_difference": current["service"] - previous["service"] if previous_feasible else None,
            "cost_difference_at_equal_service": previous["cost"] - current["cost"] if same_service else None,
            "current_quantity": current["quantity"], "current_cost": current["cost"],
            "current_constraint_violations": len(current["violations"])}


def finite_marginals(levels, values, *, increasing_resource=True):
    """Discrete finite differences, not LP duals or an investment ROI claim."""
    if len(levels) != len(values) or len(levels) < 2:
        raise ValueError("matching level/value sequences required")
    if any(isinstance(x, bool) or not math.isfinite(x) for x in [*levels, *values]):
        raise ValueError("finite numeric levels and values required")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("resource levels must strictly increase")
    rows = []
    for a, b, va, vb in zip(levels, levels[1:], values, values[1:]):
        gain = vb - va
        rows.append({"from": a, "to": b, "value_before": va, "value_after": vb,
                     "value_difference": gain, "difference_per_unit": gain / (b-a),
                     "zero_effect": abs(gain) <= 1e-7,
                     "monotonicity_violation": gain < -1e-7 if increasing_resource else gain > 1e-7})
    return rows
