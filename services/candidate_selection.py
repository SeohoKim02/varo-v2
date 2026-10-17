"""Finite-allocation candidate selectors with one shared, independently checked model.

Lexicographic: fulfilled service -> recomputed cost -> VHS tie break. MILP uses
three sequential solves, never a fragile weighted sum. No production imports it.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields, replace
from fractions import Fraction
from typing import Any, Mapping
import itertools
import math
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix
from services.operating_snapshot import signature

VERSION = "finite-allocation-0.1.1"


@dataclass(frozen=True)
class AllocationOption:
    quantity: float
    cost: float


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    product: str
    source: str
    target: str
    state: str | None
    lot: str | None
    unit: str | None
    resources: Mapping[str, float]
    need_key: str
    options: tuple[AllocationOption, ...]
    vhs: float = 0.0
    marginal_value: float | None = None
    cost_basis: str = "QUANTITY_SPECIFIC"
    lineage: str = ""


@dataclass(frozen=True)
class SelectionProblem:
    problem_id: str
    capacities: Mapping[str, float]
    needs: Mapping[str, float]
    candidates: tuple[Candidate, ...]
    max_routes: int
    evidence_class: str
    service_metric: str = "MACRO_FULFILLMENT"
    policy: Mapping[str, Any] = field(default_factory=dict)

    @property
    def base_problem_signature(self) -> str:
        # Do not recursively copy thousands of options only to discard them.
        # The encoded fields and hash are exactly the old asdict-minus-candidates.
        return signature({f.name: getattr(self, f.name) for f in fields(self) if f.name != "candidates"})

    @property
    def candidate_set_signature(self) -> str:
        return signature(sorted([asdict(c) for c in self.candidates], key=lambda x: x["candidate_id"]))


def validate(p: SelectionProblem) -> None:
    if p.evidence_class not in {"SCENARIO", "PROXY", "VERIFIED_INPUT"}:
        raise ValueError("explicit evidence class required")
    if p.max_routes < 0 or p.service_metric not in {"MACRO_FULFILLMENT", "LEGACY_RAW_QTY"}:
        raise ValueError("invalid selection policy")
    if len({c.candidate_id for c in p.candidates}) != len(p.candidates):
        raise ValueError("duplicate candidate identity")
    for v in list(p.capacities.values()) + list(p.needs.values()):
        if not math.isfinite(v) or v < 0:
            raise ValueError("invalid cap/need")
    for c in p.candidates:
        if not c.resources or c.need_key not in p.needs or p.needs[c.need_key] <= 0:
            raise ValueError("candidate resource/need missing")
        if c.need_key not in c.resources or c.resources[c.need_key] != 1:
            raise ValueError("target need must be a shared resource")
        if p.capacities[c.need_key] > p.needs[c.need_key] + 1e-9:
            raise ValueError("target capacity exceeds declared need")
        unknown = {"UNKNOWN", "UNDECLARED", "UNCONFIRMED", "NONE", "NAN"}
        if p.evidence_class != "PROXY" and (not c.state or not c.unit or c.state.upper() in unknown or c.unit.upper() in unknown):
            raise ValueError("state/unit required outside legacy proxy")
        for k, coeff in c.resources.items():
            if k not in p.capacities or not math.isfinite(coeff) or coeff <= 0:
                raise ValueError("resource unit/conversion missing")
        if len({o.quantity for o in c.options}) != len(c.options):
            raise ValueError("duplicate quantity option")
        for o in c.options:
            if not math.isfinite(o.quantity) or o.quantity <= 0 or not math.isfinite(o.cost) or o.cost < 0:
                raise ValueError("unknown/invalid quantity cost")


def _service(p: SelectionProblem, c: Candidate, o: AllocationOption) -> float:
    return o.quantity if p.service_metric == "LEGACY_RAW_QTY" else o.quantity / p.needs[c.need_key] / max(1, sum(v > 0 for v in p.needs.values()))


def _tie_value(p: SelectionProblem, c: Candidate, quantity: float) -> float:
    # A dimensionless score must not change when EA and BOX are converted.
    return c.vhs * quantity / p.needs[c.need_key]


def evaluate(p: SelectionProblem, plan: Mapping[str, float]) -> dict[str, Any]:
    by_id = {c.candidate_id: c for c in p.candidates}
    used = {k: 0.0 for k in p.capacities}
    service = cost = vhs = 0.0
    violations = []
    fulfilled = {k: 0.0 for k in p.needs}
    if len(plan) > p.max_routes:
        violations.append("MAX_ROUTES")
    for cid, q in plan.items():
        c = by_id.get(cid)
        # Plans select declared finite options. Approximate lookup can silently
        # price a different nearby option; resource tolerances are separate.
        opt = next((o for o in c.options if o.quantity == q), None) if c else None
        if opt is None:
            violations.append("UNKNOWN_ALLOCATION:" + cid)
            continue
        for k, coeff in c.resources.items():
            used[k] += q * coeff
        fulfilled[c.need_key] += q
        service += _service(p, c, opt)
        cost += opt.cost
        vhs += _tie_value(p, c, q)
    violations += ["CAP:" + k for k, v in used.items() if v > p.capacities[k] + 1e-7]
    return {"service": service, "cost": cost, "vhs": vhs, "quantity": sum(plan.values()),
            "routes": len(plan), "violations": violations, "resource_used": used,
            "remaining": {k: p.capacities[k] - v for k, v in used.items()},
            "fulfillment": {k: v / p.needs[k] for k, v in fulfilled.items() if p.needs[k] > 0}}


def _objective(ev: Mapping[str, Any]) -> tuple:
    return (round(ev["service"], 10), -round(ev["cost"], 6), round(ev["vhs"], 6))


def _greedy(p: SelectionProblem, *, partial: bool, order: str = "VARO", exclude=frozenset(), start=None):
    plan = dict(start or {})
    initial = evaluate(p, plan)
    if initial["violations"]:
        raise ValueError("greedy initial plan is infeasible")
    used = initial["resource_used"]
    trace = []
    candidates = []
    for c in p.candidates:
        if c.candidate_id in exclude or c.candidate_id in plan or not c.options:
            continue
        largest = max(c.options, key=lambda o: o.quantity)
        if order == "ECONOMIC":
            if c.marginal_value is None:
                raise ValueError("economic marginal value requires evidence")
            key = (-(c.marginal_value * largest.quantity - largest.cost), largest.cost, c.candidate_id)
        elif order == "VHS":
            key = (-c.vhs, largest.cost, c.candidate_id)
        elif order == "COST_ONLY":
            key = (largest.cost, -_service(p, c, largest), c.candidate_id)
        else:
            key = (-_service(p, c, largest), largest.cost, -c.vhs if order != "NO_VHS" else 0, c.candidate_id)
        candidates.append((key, c))
    for _, c in sorted(candidates, key=lambda x: x[0]):
        if len(plan) >= p.max_routes:
            break
        options = sorted(c.options, key=lambda o: (-o.quantity, o.cost))
        if not partial:
            options = options[:1]
        for o in options:
            # Same summation order/tolerance as the independent evaluator, but
            # test only the changed resources instead of re-evaluating the plan.
            if all(used[k] + o.quantity * coeff <= p.capacities[k] + 1e-7
                   for k, coeff in c.resources.items()):
                if order == "ECONOMIC" and c.marginal_value * o.quantity - o.cost <= 0:
                    continue
                plan[c.candidate_id] = o.quantity
                for k, coeff in c.resources.items():
                    used[k] += o.quantity * coeff
                trace.append({"candidate": c.candidate_id, "quantity": o.quantity,
                              "cost": o.cost, "reason": order,
                              "remaining": {k: p.capacities[k] - v for k, v in used.items()}})
                break
    return plan, trace


def solve(p: SelectionProblem, algorithm: str = "MILP", *, time_limit: float = 30.0) -> dict[str, Any]:
    if algorithm == "IMPROVED_RAW":
        return improve_selection(p, time_limit=time_limit)
    if algorithm == "PROTECTED":
        return protected_selection(p, time_limit=time_limit)
    validate(p)
    started = time.perf_counter()
    plan: dict[str, float] = {}
    trace = []
    solver_info: dict[str, Any] = {"optimality_proven": False}
    if algorithm in {"VARO", "PARTIAL", "LOCAL", "ECONOMIC", "VHS", "NO_VHS", "COST_ONLY"}:
        plan, trace = _greedy(p, partial=algorithm != "VARO", order=algorithm)
        if algorithm == "LOCAL":
            # Single removal + greedy reinsertion, repeated while strict improvement.
            for _ in range(max(1, len(p.candidates))):
                best = plan
                for removed in sorted(plan):
                    trial, _ = _greedy(p, partial=True, exclude={removed},
                                       start={k: v for k, v in plan.items() if k != removed})
                    if _objective(evaluate(p, trial)) > _objective(evaluate(p, best)):
                        best = trial
                if best == plan:
                    break
                trace.append({"reason": "REMOVE_INSERT", "before": plan, "after": best})
                plan = best
    elif algorithm == "MILP":
        items = [(c, o) for c in sorted(p.candidates, key=lambda c: c.candidate_id)
                 for o in sorted(c.options, key=lambda o: o.quantity)]
        if items:
            # Preserve the old row/column order exactly, without materializing
            # O(options * all_resources) zeros for this sparse incidence model.
            candidate_rows = {c.candidate_id: i for i, c in enumerate(p.candidates)}
            capacity_rows = {key: len(candidate_rows) + i for i, key in enumerate(sorted(p.capacities))}
            route_row = len(candidate_rows) + len(capacity_rows)
            row_indices, column_indices, values = [], [], []
            for j, (c, o) in enumerate(items):
                row_indices.extend((candidate_rows[c.candidate_id], route_row))
                column_indices.extend((j, j)); values.extend((1., 1.))
                for key, coefficient in c.resources.items():
                    row_indices.append(capacity_rows[key]); column_indices.append(j)
                    values.append(o.quantity * coefficient)
            bounds = [1.] * len(candidate_rows) + [p.capacities[k] for k in sorted(p.capacities)] + [p.max_routes]
            matrix = csr_matrix((values, (row_indices, column_indices)), shape=(route_row + 1, len(items)))
            constraints = [LinearConstraint(matrix, -np.inf, np.array(bounds))]
            service = np.array([_service(p, c, o) for c, o in items])
            cost = np.array([o.cost for _, o in items])
            vhs = np.array([_tie_value(p, c, o.quantity) for c, o in items])
            stages = []
            result = None
            for name, objective in (("service", -service), ("cost", cost), ("vhs", -vhs)):
                remaining = time_limit - (time.perf_counter() - started)
                if remaining <= 0:
                    break
                r = milp(objective, integrality=np.ones(len(items)), bounds=Bounds(0, 1),
                         constraints=constraints, options={"time_limit": remaining, "mip_rel_gap": 0.0})
                stages.append({"stage": name, "status": int(r.status), "message": str(r.message),
                    "incumbent": None if r.fun is None else float(r.fun),
                    "bound": None if getattr(r, "mip_dual_bound", None) is None else float(r.mip_dual_bound),
                    "gap": None if getattr(r, "mip_gap", None) is None else float(r.mip_gap)})
                if r.x is not None:
                    result = r
                if r.status != 0 or r.x is None:
                    break
                value = float(objective @ np.rint(r.x))
                constraints.append(LinearConstraint(objective, value - 1e-7, value + 1e-7))
            if result is not None:
                plan = {c.candidate_id: o.quantity for (c, o), x in zip(items, result.x) if x > .5}
            solver_info = {"stages": stages, "optimality_proven": len(stages) == 3 and all(s["status"] == 0 for s in stages),
                           "gap_scope": "same finite-allocation problem only"}
    else:
        raise ValueError("unknown candidate algorithm")
    ev = evaluate(p, plan)
    if ev["violations"]:
        raise AssertionError("candidate plan violates common constraints: " + str(ev["violations"]))
    return {"version": VERSION, "algorithm": algorithm, "plan": plan, **ev,
            "trace": trace, "solver": solver_info, "seconds": time.perf_counter() - started,
            "base_problem_signature": p.base_problem_signature, "candidate_set_signature": p.candidate_set_signature,
            "algorithm_signature": signature({"version": VERSION, "algorithm": algorithm, "time_limit": time_limit}),
            "evidence_class": p.evidence_class, "production_approved": False}


def exhaustive(p: SelectionProblem, *, max_combinations: int = 100000) -> dict[str, float]:
    validate(p)
    if math.prod(len(c.options) + 1 for c in p.candidates) > max_combinations:
        raise ValueError("enumeration budget exceeded")
    best = {}
    for options in itertools.product(*[(None, *c.options) for c in p.candidates]):
        plan = {c.candidate_id: o.quantity for c, o in zip(p.candidates, options) if o is not None}
        ev = evaluate(p, plan)
        if not ev["violations"] and _objective(ev) > _objective(evaluate(p, best)):
            best = plan
    return best


SAFE_VERSION = "verified-neighborhood-0.3"


def _option_exact_objective(p, c, option):
    """Separable exact contribution; used only for a conservative search bound."""
    f = lambda value: Fraction(str(value))
    quantity = f(option.quantity)
    service = quantity if p.service_metric == "LEGACY_RAW_QTY" else quantity / f(p.needs[c.need_key]) / max(1, sum(v > 0 for v in p.needs.values()))
    return service, -f(option.cost), f(c.vhs) * quantity / f(p.needs[c.need_key])


def _relaxed_upper_bound(p, candidates, slots, fixed_objective):
    """Lexicographic upper bound after dropping all residual shared constraints.

    Each candidate contributes its largest feasible quantity (positive primary
    service), then choose the lexicographically best ``slots`` contributions.
    Lexicographic ordering is translation-invariant, so this is an optimistic
    bound even with negative tie scores or non-monotone quantity costs. It never
    certifies full-problem optimality; it only avoids a non-improving neighbor.
    """
    contributions = sorted((_option_exact_objective(p, c, max(c.options, key=lambda o: o.quantity))
                            for c in candidates), reverse=True)[:slots]
    return tuple(fixed_objective[i] + sum((v[i] for v in contributions), Fraction(0)) for i in range(3))


def exact_objective(p: SelectionProblem, plan: Mapping[str, float]) -> tuple[Fraction, Fraction, Fraction]:
    """No rounding or service epsilon: compare the declared finite model exactly.

    Fractions use decimal representations of input values, not binary float noise.
    Solver tolerances are separate; a solver result still has to pass this gate.
    """
    ev = evaluate(p, plan)
    if ev["violations"]:
        raise ValueError("cannot compare an infeasible plan")
    f = lambda x: Fraction(str(x))
    by_id = {c.candidate_id: c for c in p.candidates}
    count = max(1, sum(v > 0 for v in p.needs.values()))
    service = cost = tie = Fraction(0)
    for cid, q in sorted(plan.items()):
        c = by_id[cid]
        option = next((o for o in c.options if o.quantity == q), None)
        if option is None:
            raise ValueError("allocation is not an exact declared finite option: " + cid)
        service += f(q) if p.service_metric == "LEGACY_RAW_QTY" else f(q) / f(p.needs[c.need_key]) / count
        cost += f(option.cost)
        tie += f(c.vhs) * f(q) / f(p.needs[c.need_key])
    return service, -cost, tie


def incumbent_compatibility(p: SelectionProblem, old: SelectionProblem,
                            plan: Mapping[str, float]) -> dict[str, Any]:
    """Accept equal or expanded finite options, never trust an unbound plan/hash.

    The base includes decision identity, information policy, caps, needs, objective,
    and route limit. Every old candidate and its costed options must be contained.
    Zero allocation is represented by absence, as in both the MILP and evaluator.
    """
    validate(p); validate(old)
    reasons = []
    if p.base_problem_signature != old.base_problem_signature:
        reasons.append("BASE_PROBLEM_OR_INFORMATION_CHANGED")
    current = {c.candidate_id: c for c in p.candidates}
    for c in old.candidates:
        new = current.get(c.candidate_id)
        a = {f.name: getattr(c, f.name) for f in fields(c) if f.name != "options"}
        b = {f.name: getattr(new, f.name) for f in fields(new) if f.name != "options"} if new else {}
        if a != b:
            reasons.append("CANDIDATE_SEMANTICS_CHANGED:" + c.candidate_id)
        elif not set(c.options).issubset(set(new.options)):
            reasons.append("OLD_QUANTITY_COST_NOT_CONTAINED:" + c.candidate_id)
    for label, problem in (("OLD", old), ("CURRENT", p)):
        reasons += [label + ":" + x for x in evaluate(problem, plan)["violations"]]
    return {"compatible": not reasons, "reasons": reasons,
            "zero_allocation": "implicit candidate absence", "all_old_options_contained": not any(
                x.startswith(("CANDIDATE_", "OLD_QUANTITY_")) for x in reasons)}


def _safe_result(p, algorithm, plan, trace, started, solver, **extra):
    ev = evaluate(p, plan)
    if ev["violations"]:
        raise AssertionError("independent final verification failed: " + str(ev["violations"]))
    return {"version": SAFE_VERSION, "algorithm": algorithm, "plan": dict(plan), **ev,
            "trace": trace, "solver": solver, "seconds": time.perf_counter() - started,
            "base_problem_signature": p.base_problem_signature,
            "candidate_set_signature": p.candidate_set_signature,
            "algorithm_signature": signature({"version": SAFE_VERSION, "algorithm": algorithm,
                                                "configuration": extra.get("configuration", {})}),
            "evidence_class": p.evidence_class, "production_approved": False, **extra}


def improve_selection(p: SelectionProblem, *, time_limit: float = 2.0,
                      max_removed: int = 2, neighborhood_limit: int = 40,
                      initial_plan: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Raw partial-seeded improvement, distinct from baseline protection.

    Hold all other selected quantities fixed, remove one/two routes, then solve
    all feasible unselected/removed options against residual shared capacities.
    This permits replacement and quantity rebalancing without claiming global
    optimality. SciPy MILP has no warm-start interface; none is claimed or used.
    """
    validate(p)
    if not math.isfinite(time_limit) or time_limit < 0 or max_removed not in (0, 1, 2):
        raise ValueError("invalid neighborhood budget")
    started = time.perf_counter(); deadline = started + time_limit
    plan = dict(initial_plan) if initial_plan is not None else _greedy(p, partial=True)[0]
    exact_objective(p, plan)
    trace = []; attempted = 0; stop = "NEIGHBORHOOD_STABLE"
    bound_pruned = 0; milp_calls = 0; enumeration_calls = 0
    current_objective = exact_objective(p, plan)
    by_id = {c.candidate_id: c for c in p.candidates}
    while max_removed and attempted < neighborhood_limit:
        before_round = dict(plan)
        ordered = sorted(plan, key=lambda cid: (plan[cid], -next(o.cost for o in by_id[cid].options if o.quantity == plan[cid]), cid))
        removals = [()] + [combo for n in range(1, min(max_removed, len(ordered)) + 1)
                           for combo in itertools.combinations(ordered, n)]
        for removed in removals:
            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0 or attempted >= neighborhood_limit:
                stop = "TIME_LIMIT" if remaining_time <= 0 else "NEIGHBORHOOD_LIMIT"
                break
            # A prior improvement restarts the neighborhood list around its incumbent.
            fixed = {cid: q for cid, q in plan.items() if cid not in removed}
            slots = p.max_routes - len(fixed)
            if slots <= 0:
                continue
            fixed_ev = evaluate(p, fixed)
            residual_caps = {k: max(0., v) for k, v in fixed_ev["remaining"].items()}
            residual_candidates = []
            for c in p.candidates:
                if c.candidate_id in fixed:
                    continue
                options = tuple(o for o in c.options if all(
                    o.quantity * coefficient <= residual_caps[key] + 1e-9
                    for key, coefficient in c.resources.items()))
                if options:
                    residual_candidates.append(replace(c, options=options))
            residual = replace(p, capacities=residual_caps, candidates=tuple(residual_candidates), max_routes=slots)
            attempted += 1
            upper = _relaxed_upper_bound(p, residual_candidates, slots, exact_objective(p, fixed))
            if upper <= current_objective:
                bound_pruned += 1
                trace.append({"reason": "RELAXED_BOUND_PRUNE", "removed": list(removed),
                              "fixed": fixed, "accepted": False,
                              "upper_bound": [str(x) for x in upper],
                              "incumbent_objective": [str(x) for x in current_objective],
                              "proof_scope": "this residual neighborhood only", "warm_start_used": False})
                continue
            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0:
                stop = "TIME_LIMIT"
                break
            if slots == 1:
                # For one slot each candidate's largest feasible quantity dominates
                # its smaller options on the primary service objective. Enumerate
                # these exact finite alternatives instead of spending a MILP budget.
                tick = time.perf_counter(); residual_plan = {}; best_objective = (Fraction(0),) * 3
                enumeration_calls += 1
                for c in residual_candidates:
                    option = max(c.options, key=lambda o: o.quantity)
                    choice = {c.candidate_id: option.quantity}
                    objective = _option_exact_objective(residual, c, option)
                    if objective > best_objective:
                        residual_plan = choice
                        best_objective = objective
                result = {"plan": residual_plan, "seconds": time.perf_counter() - tick,
                          "solver": {"optimality_proven": True, "method": "ONE_ROUTE_ENUMERATION",
                                     "scope": "this residual one-route finite problem only"}}
            else:
                milp_calls += 1
                result = solve(residual, "MILP", time_limit=min(.5, remaining_time))
            trial = {**fixed, **result["plan"]}
            evaluation = evaluate(p, trial)
            trial_objective = exact_objective(p, trial) if not evaluation["violations"] else None
            accepted = trial_objective is not None and trial_objective > current_objective
            trace.append({"reason": "RESIDUAL_ENUMERATION" if slots == 1 else "RESIDUAL_MILP", "removed": list(removed),
                          "fixed": fixed, "before": dict(plan), "trial": trial,
                          "accepted": accepted, "solver": result["solver"],
                          "seconds": result["seconds"], "warm_start_used": False})
            if accepted:
                plan = trial
                current_objective = trial_objective
                break
        if stop != "NEIGHBORHOOD_STABLE" or plan == before_round:
            break
    return _safe_result(p, "IMPROVED_RAW", plan, trace, started,
                        {"optimality_proven": False, "status": "FEASIBLE_NOT_PROVEN_OPTIMAL",
                         "termination": stop, "neighborhoods": attempted, "warm_start_used": False,
                         "bound_pruned": bound_pruned, "residual_milp_calls": milp_calls,
                         "enumeration_calls": enumeration_calls},
                        seed_source="PARTIAL" if initial_plan is None else "CALLER_VERIFIED_PLAN",
                        configuration={"time_limit": time_limit, "max_removed": max_removed,
                                       "neighborhood_limit": neighborhood_limit})


def protected_selection(p: SelectionProblem, *, baseline_problem: SelectionProblem | None = None,
                        baseline_plan: Mapping[str, float] | None = None,
                        challengers: Mapping[str, Mapping[str, Any]] | None = None,
                        time_limit: float = 2.0) -> dict[str, Any]:
    """Verified portfolio selection; no future outcomes or external labels used."""
    started = time.perf_counter()
    old = baseline_problem or p
    baseline = dict(baseline_plan) if baseline_plan is not None else solve(old, "VARO")["plan"]
    compatibility = incumbent_compatibility(p, old, baseline)
    if not compatibility["compatible"]:
        raise ValueError("incumbent cannot be reused: " + ";".join(compatibility["reasons"]))
    if challengers is None:
        challengers = {"IMPROVED_RAW": improve_selection(p, time_limit=max(0., time_limit - (time.perf_counter() - started)))}
    plan, source = baseline, "BASELINE_VARO"
    baseline_objective = exact_objective(p, baseline)
    best_objective = baseline_objective
    base_signature, candidate_signature = p.base_problem_signature, p.candidate_set_signature
    rows = []
    for label, result in challengers.items():
        if result.get("base_problem_signature") != base_signature or result.get("candidate_set_signature") != candidate_signature:
            raise ValueError("challenger problem or candidate signature mismatch: " + label)
        candidate_plan = result["plan"]
        ev = evaluate(p, candidate_plan)
        if ev["violations"]:
            raise ValueError("challenger failed independent validation: " + label)
        objective = exact_objective(p, candidate_plan)
        comparison = (objective > baseline_objective) - (objective < baseline_objective)
        rows.append({"source": label, "versus_baseline": comparison, "service": ev["service"],
                     "cost": ev["cost"], "quantity": ev["quantity"], "plan": dict(candidate_plan)})
        if objective > best_objective:
            plan, source = dict(candidate_plan), label
            best_objective = objective
    fallback = source == "BASELINE_VARO" and bool(rows) and all(r["versus_baseline"] < 0 for r in rows)
    return _safe_result(p, "PROTECTED", plan, rows, started,
                        {"optimality_proven": False, "status": "VERIFIED_PORTFOLIO_INCUMBENT",
                         "warm_start_used": False}, selected_source=source, fallback_used=fallback,
                        baseline_retained=source == "BASELINE_VARO", compatibility=compatibility,
                        configuration={"challenger_names": list(challengers), "time_limit": time_limit},
                        runtime_note="portfolio gate time only when precomputed challengers are supplied")
