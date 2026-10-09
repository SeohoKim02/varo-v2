"""MILP benchmark integrity: what a MILP comparison proved, over which candidates, caps and objective.

Validation-only and parallel.  Nothing here changes a MILP result, a Varo Final rank, the Top-5, a T1 plan, a T2 action
or a KPI.  For one benchmark run it reports:

* scope   which candidate pool the optimum is over (restricted by a cutoff, or the whole input pool) and whether the
          caps are PROXY.  A restricted or PROXY optimum is never reported as a network-wide (global) optimum, and
          ``global_optimum_claimable`` is always False: no dataset carries every real route, vehicle and DC constraint.
* status  the solver outcome normalised to OPTIMAL / FEASIBLE_NOT_PROVEN_OPTIMAL / TIME_LIMIT / INFEASIBLE / UNBOUNDED /
          ERROR / NOT_SOLVED / UNKNOWN, next to the raw code and message.  OPTIMAL means HiGHS proved optimality within
          its MIP tolerance (default mip_rel_gap 1e-4, mip_abs_gap 1e-6) for that model, nothing more.
* gaps    the solver MIP gap (incumbent vs bound of one model) and the decision-performance gap (a plan vs the
          benchmark under an identical comparison signature) are separate fields.  Either is NULL with reason codes
          when it cannot be computed; it is never 0 by default.
* proof   a solver-independent re-check of the selected moves (shared caps, duplicates, Top-N, quantity granularity)
          and, for small pools, exhaustive enumeration of the lexicographic optimum.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from services import shared_feasibility_selection as sf

BENCHMARK_INTEGRITY_VERSION = "milp-benchmark-integrity-v0.1"

# ---------------------------------------------------------------------------------------------- vocabularies

OPTIMAL, FEASIBLE_NOT_PROVEN_OPTIMAL, TIME_LIMIT = "OPTIMAL", "FEASIBLE_NOT_PROVEN_OPTIMAL", "TIME_LIMIT"
INFEASIBLE, UNBOUNDED, ERROR, NOT_SOLVED, UNKNOWN = "INFEASIBLE", "UNBOUNDED", "ERROR", "NOT_SOLVED", "UNKNOWN"
SOLVER_STATUSES = (OPTIMAL, FEASIBLE_NOT_PROVEN_OPTIMAL, TIME_LIMIT, INFEASIBLE, UNBOUNDED, ERROR, NOT_SOLVED, UNKNOWN)

RESTRICTED_CANDIDATE_OPTIMUM = "RESTRICTED_CANDIDATE_OPTIMUM"
FULL_CANDIDATE_MODEL_OPTIMUM = "FULL_CANDIDATE_MODEL_OPTIMUM"
FEASIBLE_SOLUTION_ONLY, PROXY_BENCHMARK = "FEASIBLE_SOLUTION_ONLY", "PROXY_BENCHMARK"
NOT_COMPARABLE, INSUFFICIENT_DATA = "NOT_COMPARABLE", "INSUFFICIENT_DATA"
SCOPES = (RESTRICTED_CANDIDATE_OPTIMUM, FULL_CANDIDATE_MODEL_OPTIMUM, FEASIBLE_SOLUTION_ONLY, PROXY_BENCHMARK,
          NOT_COMPARABLE, INSUFFICIENT_DATA)

DIRECTLY_COMPARABLE, COMPARABLE_REFERENCE_NOT_PROVEN = "DIRECTLY_COMPARABLE", "COMPARABLE_REFERENCE_NOT_PROVEN"
GAP_COMPUTED, GAP_NOT_COMPUTABLE = "GAP_COMPUTED", "GAP_NOT_COMPUTABLE"

SAME_RESTRICTED_OBJECTIVE, SAME_SERVICE_AND_COST = "SAME_RESTRICTED_OBJECTIVE", "SAME_SERVICE_AND_COST"
SAME_ROUTES, DIFFERENT_ROUTES_SAME_TOTALS = "SAME_ROUTES", "DIFFERENT_ROUTES_SAME_TOTALS"
SOLVER_PROVEN_OPTIMAL, ENUMERATION_PROVEN_OPTIMAL = "SOLVER_PROVEN_OPTIMAL", "ENUMERATION_PROVEN_OPTIMAL"
NOT_PROVEN_GLOBAL = "NOT_PROVEN_GLOBAL"

FIXED_BINARY, PARTIAL_ALLOWED = "FIXED_RECOMMENDED_QTY_BINARY", "PARTIAL_ALLOCATION_ALLOWED"
LEXICOGRAPHIC_OBJECTIVE = (
    "LEXICOGRAPHIC_SERVICE_THEN_COST: stage 1 max sum(recommended_qty*x); stage 2 min sum(move_cost*x) + "
    "1e-7*route_order*x subject to sum(recommended_qty*x) >= stage-1 optimum - 1e-7; x binary")
SAVING_OBJECTIVE = "MAX_EXPECTED_SAVING: max sum(expected_saving*x), x binary, single stage; service and cost are not in it"
HIGHS_DEFAULT_TOLERANCE = {"mip_rel_gap": 1e-4, "mip_abs_gap": 1e-6, "source": "HiGHS defaults (scipy.optimize.milp)"}
ENUMERATION_COMBINATION_LIMIT = 750_000  # same order as optimality_gap_service.AUTO_EXACT_COMBINATION_LIMIT
EPS = 1e-9

REASONS = {
    "CANDIDATE_POOL_TRUNCATED": "the solved pool is a cutoff of a larger candidate pool; optimum holds inside the cut only",
    "CANDIDATE_POOL_TOTAL_UNKNOWN": "the size of the pool before the cutoff is unknown; treated as restricted",
    "CUTOFF_BY_EVALUATED_STRATEGY_RANK": "the pool was cut by the rank of the strategy being evaluated (selection bias)",
    "PROXY_CAPS": "a source/target cap is a PROXY rule, not an observed limit",
    "MISSING_CAP": "a required source/target cap is missing for a candidate group",
    "CAP_PROVENANCE_NOT_TRACKED": "the run does not record where its caps come from",
    "NO_ROUTE_CAPACITY_DATA": "no unit-compatible route/vehicle capacity exists; not constrained",
    "NO_DC_CAPACITY_DATA": "no DC capacity exists; not constrained",
    "NETWORK_CONSTRAINTS_ABSENT": "real routes, vehicles, budgets and time windows are not all in the model",
    "SOLVER_TOLERANCE_DEFAULT": "OPTIMAL is proven within HiGHS default mip_rel_gap 1e-4 / mip_abs_gap 1e-6",
    "SOLVER_NOT_OPTIMAL": "the solver did not prove optimality",
    "SOLVER_TIME_LIMIT": "the solver stopped at its time limit",
    "LIMITED_SEARCH_MODE": "limited search mode never certifies optimality, even when exhausted",
    "NO_INCUMBENT": "no feasible solution value to measure a gap from",
    "MISSING_BOUND": "the solver returned no valid bound",
    "INCONSISTENT_BOUND": "the bound lies on the wrong side of the incumbent",
    "ZERO_OBJECTIVE": "the reference objective is 0, a relative gap is undefined",
    "NO_POSITIVE_OBJECTIVE": "no feasible combination has a positive objective value",
    "SERVICE_NOT_IN_OBJECTIVE": "service quantity is not part of this objective",
    "COST_NOT_IN_OBJECTIVE": "move cost is not part of this objective",
    "SERVICE_NOT_EQUAL": "cost gap is defined only at equal service under a service-first objective",
    "REFERENCE_NOT_PROVEN_OPTIMAL": "the benchmark solution is not proven optimal; the gap is against an incumbent",
    "PLAN_EXCEEDS_PROVEN_OPTIMUM": "a plan beats a proven optimum: the plan or the model is inconsistent",
    "DIFFERENT_DATASET_OR_DATE": "different dataset or date",
    "DIFFERENT_CANDIDATE_POOL": "different candidate pool (ids, quantities or costs)",
    "DIFFERENT_CONSTRAINTS": "different constraint set",
    "DIFFERENT_CAP_VALUES": "same constraints but different cap values or provenance",
    "DIFFERENT_COST_BASIS": "different cost basis",
    "DIFFERENT_QUANTITY_BASIS": "different quantity granularity (fixed 0/1 vs partial allocation)",
    "DIFFERENT_SELECTION_LIMIT": "different Top-N / max_routes",
    "DIFFERENT_OBJECTIVE": "different objective or priority",
    "INDEPENDENT_CHECK_FAILED": "the solver-independent re-check found a violation",
    "ALTERNATIVE_OPTIMA": "several selections reach the same optimum; route identity is not unique",
    "ENUMERATION_TOO_LARGE": "pool too large for exhaustive enumeration",
    "NO_FEASIBLE_CANDIDATES": "no candidate survived the model's exclusions; the optimum is vacuous",
    "INPUT_POOL_MAY_BE_PRE_CUT": "the input pool may itself be a cutoff (candidate_generator keeps 20 by candidate_score)",
}
SIGNATURE_REASONS = {
    "dataset": "DIFFERENT_DATASET_OR_DATE", "date": "DIFFERENT_DATASET_OR_DATE",
    "candidate_pool": "DIFFERENT_CANDIDATE_POOL", "constraints": "DIFFERENT_CONSTRAINTS", "caps": "DIFFERENT_CAP_VALUES",
    "cost_basis": "DIFFERENT_COST_BASIS", "quantity_basis": "DIFFERENT_QUANTITY_BASIS",
    "selection_limit": "DIFFERENT_SELECTION_LIMIT", "objective": "DIFFERENT_OBJECTIVE",
}
CONTRACT_FIELDS = (
    "benchmark_mode", "benchmark_scope", "scope_flags", "candidate_pool_size", "candidate_pool_total",
    "candidate_pool_truncated", "candidate_cutoff_basis", "candidate_limit", "max_routes", "time_limit_s",
    "quantity_granularity", "objective_definition", "constraint_signature", "cap_provenance", "cost_provenance",
    "solver_status", "termination_reason", "solver_status_code", "solver_tolerance", "incumbent_objective",
    "best_bound", "solver_mip_gap", "solver_mip_gap_abs", "solver_mip_gap_status", "decision_service_gap",
    "decision_cost_gap", "decision_gap_status", "comparison_status", "optimality_claim", "global_optimum_claimable",
    "limitations", "reason_codes",
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


# ---------------------------------------------------------------------------------------------- solver status


def normalize_solver_status(status_code: Any, message: Any = None, *, has_solution: bool | None = None,
                            error: Any = None) -> dict[str, Any]:
    """SciPy/HiGHS ``milp`` status -> normalised status. Status 1 (time or iteration limit) is never OPTIMAL."""
    text = str(message or "").lower()
    code = None if status_code is None else int(status_code)
    if error:
        status, reason = ERROR, "SOLVER_ERROR"
    elif code is None:
        status, reason = NOT_SOLVED, "NOT_RUN"
    elif code == 0:
        status, reason = OPTIMAL, "OPTIMALITY_PROVEN_WITHIN_TOLERANCE"
    elif code == 1:
        reason = "TIME_LIMIT" if "time limit" in text else "ITERATION_LIMIT" if "iteration limit" in text else "SOLVER_LIMIT"
        status = FEASIBLE_NOT_PROVEN_OPTIMAL if has_solution else TIME_LIMIT if reason == "TIME_LIMIT" else NOT_SOLVED
    elif code == 2:
        status, reason = INFEASIBLE, "INFEASIBLE"
    elif code == 3:
        status, reason = UNBOUNDED, "UNBOUNDED"
    elif code == 4:
        status, reason = UNKNOWN, "UNBOUNDED_OR_INFEASIBLE" if "unbounded or infeasible" in text else "SOLVER_OTHER"
    else:
        status, reason = UNKNOWN, "UNRECOGNIZED_STATUS"
    return {"solver_status": status, "termination_reason": reason, "solver_status_code": code,
            "solver_message": None if message is None else str(message), "has_incumbent": has_solution,
            "optimality_proven": status == OPTIMAL}


def normalize_search_status(search: Mapping[str, Any] | None) -> dict[str, Any]:
    """The in-app ``optimality_gap_service.search_best_combination`` search dict -> normalised status."""
    search = dict(search or {})
    if not search:
        return normalize_solver_status(None)
    if search.get("available") is False:
        return {**normalize_solver_status(None, search.get("error"), error=True), "termination_reason": "SOLVER_UNAVAILABLE"}
    if search.get("error"):
        return normalize_solver_status(None, search.get("error"), error=True)
    if "status_code" in search:  # SciPy MILP path
        result = normalize_solver_status(search["status_code"], search.get("message"),
                                         has_solution=search.get("has_incumbent"))
        if result["solver_status"] == OPTIMAL and not search.get("optimal"):
            result.update(solver_status=ERROR, termination_reason="INDEPENDENT_CHECK_FAILED", optimality_proven=False)
        return result
    selected = bool(search.get("selected_indices"))
    if search.get("optimal"):
        reason = "SEARCH_EXHAUSTED" if "BnB" in str(search.get("method")) else "EMPTY_PROBLEM"
        return {"solver_status": OPTIMAL, "termination_reason": reason, "solver_status_code": None, "solver_message": None,
                "has_incumbent": True, "optimality_proven": True}
    reason = "TIME_LIMIT" if search.get("timed_out") else "LIMITED_SEARCH_MODE"
    return {"solver_status": FEASIBLE_NOT_PROVEN_OPTIMAL, "termination_reason": reason, "solver_status_code": None,
            "solver_message": None, "has_incumbent": selected or None, "optimality_proven": False,
            "search_exhausted": bool(search.get("search_exhausted"))}


def solver_mip_gap(incumbent: Any, bound: Any, *, sense: str = "max") -> dict[str, Any]:
    """Solver MIP gap |bound - incumbent| / |incumbent| (HiGHS definition). NULL + reasons when not computable."""
    value, limit = _number(incumbent), _number(bound)
    reasons = [code for code, missing in (("NO_INCUMBENT", value is None), ("MISSING_BOUND", limit is None)) if missing]
    if reasons:
        return {"solver_mip_gap_status": GAP_NOT_COMPUTABLE, "solver_mip_gap": None, "solver_mip_gap_abs": None,
                "reason_codes": reasons}
    absolute = (limit - value) if sense == "max" else (value - limit)
    if absolute < -1e-6 * max(1.0, abs(value)):
        return {"solver_mip_gap_status": GAP_NOT_COMPUTABLE, "solver_mip_gap": None, "solver_mip_gap_abs": None,
                "reason_codes": ["INCONSISTENT_BOUND"]}
    absolute = max(0.0, absolute)
    if abs(value) <= EPS:
        return {"solver_mip_gap_status": GAP_NOT_COMPUTABLE, "solver_mip_gap": None,
                "solver_mip_gap_abs": round(absolute, 9), "reason_codes": ["ZERO_OBJECTIVE"]}
    return {"solver_mip_gap_status": GAP_COMPUTED, "solver_mip_gap": round(absolute / abs(value), 12),
            "solver_mip_gap_abs": round(absolute, 9), "reason_codes": []}


def lexicographic_solver_evidence(milp_result: Mapping[str, Any], quantities: Sequence[float] = ()) -> dict[str, Any]:
    """Two-stage evidence from ``suhyup_algorithm_revalidation.lexicographic_milp`` diagnostics.

    Stage 1 (service) is exact even under the default relative tolerance when every quantity is an integer and the
    stage-1 bound is less than one unit above the incumbent.  Stage 2 (cost) is proven within the tolerance only.
    """
    diagnostics = dict(milp_result.get("diagnostics") or {})
    stages: dict[str, dict[str, Any]] = {}
    for name, sense, sign in (("stage1", "max", -1.0), ("stage2", "min", 1.0)):
        raw = diagnostics.get(name)
        if not raw:
            stages[name] = {**normalize_solver_status(None), **solver_mip_gap(None, None), "objective": None, "bound": None}
            continue
        status = normalize_solver_status(raw.get("status"), raw.get("message"), has_solution=raw.get("has_solution"))
        objective = None if raw.get("objective") is None else sign * float(raw["objective"])
        bound = None if raw.get("dual_bound") is None else sign * float(raw["dual_bound"])
        stages[name] = {**status, **solver_mip_gap(objective, bound, sense=sense), "objective": objective, "bound": bound,
                        "solver_reported_mip_gap": raw.get("mip_gap"), "node_count": raw.get("node_count"),
                        "elapsed_ms": raw.get("elapsed_ms")}
    first, second = stages["stage1"], stages["stage2"]
    if first["solver_status"] != OPTIMAL:
        overall = first
    elif second["solver_status"] == NOT_SOLVED and second.get("termination_reason") == "NOT_RUN":
        overall = {**second, "solver_status": NOT_SOLVED}
    else:
        overall = second
    integral = bool(quantities) and all(abs(float(q) - round(float(q))) <= EPS for q in quantities)
    service_exact = bool(first["solver_status"] == OPTIMAL and integral and first["bound"] is not None
                         and first["objective"] is not None and first["bound"] - first["objective"] < 1.0 - EPS)
    options = dict(diagnostics.get("options") or {})
    tolerance = {key: options[key] for key in ("mip_rel_gap", "mip_abs_gap") if key in options} or HIGHS_DEFAULT_TOLERANCE
    return {
        "solver_status": overall["solver_status"], "termination_reason": overall["termination_reason"],
        "solver_status_code": overall.get("solver_status_code"),
        "optimality_proven": first["solver_status"] == OPTIMAL and second["solver_status"] == OPTIMAL,
        "stage1": first, "stage2": second, "stage1_service": diagnostics.get("stage1_service"),
        "stage2_tie_break": diagnostics.get("stage2_tie_break"),
        "service_optimum_exact_by_integrality": service_exact, "solver_tolerance": tolerance,
        "time_limit_s": options.get("time_limit"),
    }


# ---------------------------------------------------------------------------------------------- scope and claim


def benchmark_scope(*, solver_status: str, pool_size: int | None, pool_total: int | None,
                    cap_provenance: Iterable[str] = (), missing_caps: bool = False,
                    feasible_count: int | None = None) -> dict[str, Any]:
    """Scope of the optimum and the strongest claim it supports. Never a global-optimum claim.

    FULL_CANDIDATE_MODEL_OPTIMUM means optimal over every candidate the model received; that input pool can itself be
    the output of an upstream cutoff the model cannot see.
    """
    provenance = {str(item) for item in cap_provenance if item}
    proxy = bool(provenance - sf.STRICT_ACCEPTED_PROVENANCE - {"MISSING"})
    truncated = None if pool_total is None or pool_size is None else int(pool_total) > int(pool_size)
    reasons = ["NETWORK_CONSTRAINTS_ABSENT"]
    if feasible_count is not None and int(feasible_count) == 0:
        return {"benchmark_scope": INSUFFICIENT_DATA, "scope_flags": [INSUFFICIENT_DATA],
                "candidate_pool_truncated": truncated, "optimality_claim": "NO_OPTIMALITY_CLAIM",
                "global_optimum_claimable": False, "reason_codes": [*reasons, "NO_FEASIBLE_CANDIDATES"]}
    if solver_status == OPTIMAL:
        if truncated is False:
            scope, claim = FULL_CANDIDATE_MODEL_OPTIMUM, "OPTIMAL_FOR_FULL_INPUT_CANDIDATE_MODEL"
        else:
            scope, claim = RESTRICTED_CANDIDATE_OPTIMUM, "OPTIMAL_WITHIN_RESTRICTED_CANDIDATE_POOL"
            reasons.append("CANDIDATE_POOL_TRUNCATED" if truncated else "CANDIDATE_POOL_TOTAL_UNKNOWN")
        reasons.append("SOLVER_TOLERANCE_DEFAULT")
    elif solver_status == FEASIBLE_NOT_PROVEN_OPTIMAL:
        scope, claim = FEASIBLE_SOLUTION_ONLY, "FEASIBLE_NOT_PROVEN_OPTIMAL"
        reasons.append("SOLVER_NOT_OPTIMAL")
    else:
        scope, claim = INSUFFICIENT_DATA, "NO_OPTIMALITY_CLAIM"
        reasons.append("SOLVER_NOT_OPTIMAL")
    flags = [scope]
    if proxy:
        flags.append(PROXY_BENCHMARK)
        reasons.append("PROXY_CAPS")
        if claim.startswith("OPTIMAL"):
            claim += "_UNDER_PROXY_CAPS"
    if missing_caps:
        reasons.append("MISSING_CAP")
    return {"benchmark_scope": scope, "scope_flags": flags, "candidate_pool_truncated": truncated,
            "optimality_claim": claim, "global_optimum_claimable": False, "reason_codes": _unique(reasons)}


# ---------------------------------------------------------------------------------------------- comparison signature


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def comparison_signature(*, dataset: str, date: str | None, candidates: Sequence[Mapping[str, Any]],
                         constraints: Iterable[str], caps: Mapping[tuple, sf.Cap] | None, cost_basis: str,
                         quantity_basis: str, selection_limit: int | None, objective: str) -> dict[str, Any]:
    """Everything that must be identical before two plans are compared directly; one digest per component."""
    pool = sorted(
        (str(item.get("route_id") or item.get("candidate_id")),
         _number(sf._first_number(item, sf.QTY_FIELDS)[0]), _number(sf._first_number(item, sf.COST_FIELDS)[0]))
        for item in candidates)
    cap_rows = sorted(
        (kind, "/".join(key), None if cap.value is None else round(float(cap.value), 6), cap.provenance)
        for (kind, key), cap in (caps or {}).items())
    components = {
        "dataset": dataset, "date": date, "candidate_pool": pool, "constraints": sorted(set(constraints)),
        "caps": cap_rows, "cost_basis": cost_basis, "quantity_basis": quantity_basis,
        "selection_limit": selection_limit, "objective": objective,
    }
    digests = {name: _digest(value) for name, value in components.items()}
    return {"component_digests": digests, "signature": _digest(digests), "candidate_count": len(pool),
            "cap_count": len(cap_rows)}


def signature_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    a, b = left.get("component_digests") or {}, right.get("component_digests") or {}
    return [name for name in SIGNATURE_REASONS if a.get(name) != b.get(name)]


# ---------------------------------------------------------------------------------------------- decision gap


def decision_gap(plan: Mapping[str, Any], reference: Mapping[str, Any], *, plan_signature: Mapping[str, Any],
                 reference_signature: Mapping[str, Any], reference_status: str,
                 reference_enumeration_proven: bool = False, plan_feasible: bool | None = None) -> dict[str, Any]:
    """Decision-performance gap of a plan vs the benchmark (service first, cost at equal service).

    ``plan`` / ``reference``: {"service": float, "cost": float | None, "route_ids": [...]}. Computed only under an
    identical comparison signature and for a plan that passed the independent check; otherwise NULL with reasons.
    """
    differences = signature_differences(plan_signature, reference_signature)
    reasons = _unique(SIGNATURE_REASONS[name] for name in differences)
    if plan_feasible is False:
        reasons.append("INDEPENDENT_CHECK_FAILED")
    p_service, r_service = _number(plan.get("service")), _number(reference.get("service"))
    p_cost, r_cost = _number(plan.get("cost")), _number(reference.get("cost"))
    p_routes, r_routes = set(map(str, plan.get("route_ids") or [])), set(map(str, reference.get("route_ids") or []))
    totals_equal = (p_service is not None and r_service is not None and abs(p_service - r_service) <= EPS
                    and p_cost is not None and r_cost is not None and abs(p_cost - r_cost) <= 1e-6)
    flags = []
    if totals_equal:
        flags.append(SAME_SERVICE_AND_COST)
        flags.append(SAME_ROUTES if p_routes == r_routes else DIFFERENT_ROUTES_SAME_TOTALS)
    elif p_routes and p_routes == r_routes:
        flags.append(SAME_ROUTES)
    if reference_status == OPTIMAL:
        flags.append(SOLVER_PROVEN_OPTIMAL)
    if reference_enumeration_proven:
        flags.append(ENUMERATION_PROVEN_OPTIMAL)
    flags.append(NOT_PROVEN_GLOBAL)
    result = {"comparison_status": None, "decision_gap_status": GAP_NOT_COMPUTABLE, "decision_service_gap": None,
              "decision_service_gap_pct": None, "decision_cost_gap": None, "decision_cost_gap_pct": None,
              "signature_differences": differences, "zero_gap_flags": flags, "reason_codes": reasons}
    if p_service is None or r_service is None:
        result.update(comparison_status=INSUFFICIENT_DATA, reason_codes=_unique([*reasons, "NO_INCUMBENT"]))
        return result
    if differences or plan_feasible is False:
        result["comparison_status"] = NOT_COMPARABLE
        return result
    proven = reference_status == OPTIMAL or reference_enumeration_proven
    if not proven:
        reasons.append("REFERENCE_NOT_PROVEN_OPTIMAL")
    service_gap = r_service - p_service
    result.update(comparison_status=DIRECTLY_COMPARABLE if proven else COMPARABLE_REFERENCE_NOT_PROVEN,
                  decision_gap_status=GAP_COMPUTED, decision_service_gap=round(service_gap, 9))
    if proven and service_gap < -EPS:
        reasons.append("PLAN_EXCEEDS_PROVEN_OPTIMUM")
    if r_service > EPS:
        result["decision_service_gap_pct"] = round(100.0 * service_gap / r_service, 9)
    else:
        reasons.append("ZERO_OBJECTIVE")
    if abs(service_gap) > EPS:
        reasons.append("SERVICE_NOT_EQUAL")
    elif p_cost is None or r_cost is None:
        reasons.append("NO_INCUMBENT")
    else:
        cost_gap = p_cost - r_cost
        result["decision_cost_gap"] = round(cost_gap, 6)
        if r_cost > EPS:
            result["decision_cost_gap_pct"] = round(100.0 * cost_gap / r_cost, 9)
        if proven and cost_gap < -1e-6:
            reasons.append("PLAN_EXCEEDS_PROVEN_OPTIMUM")
        if abs(cost_gap) <= 1e-6:
            flags.insert(0, SAME_RESTRICTED_OBJECTIVE)
    result["reason_codes"] = _unique(reasons)
    return result


# ---------------------------------------------------------------------------------------------- independent proof


def independent_check(selected: Sequence[Mapping[str, Any]], caps: Mapping[tuple, sf.Cap], *, max_routes: int | None,
                      quantity_basis: str = FIXED_BINARY, allocated_field: str | None = None) -> dict[str, Any]:
    """Re-check selected moves without the solver: shared caps, duplicates and Top-N (T1 ``validate_plan``), plus
    quantity granularity and that the product is held at the source."""
    rows = [dict(item) for item in selected]
    benchmark = sf.validate_plan(rows, caps, mode=sf.BENCHMARK_PROXY, max_routes=max_routes, quantity_field=allocated_field)
    strict = sf.validate_plan(rows, caps, mode=sf.STRICT_ACTUAL, max_routes=max_routes, quantity_field=allocated_field)
    granularity, integer, unmatched = [], [], []
    for index, row in enumerate(rows):
        original = _number(sf._first_number(row, sf.QTY_FIELDS)[0])
        allocated = _number(row.get(allocated_field)) if allocated_field else original
        identity = sf.candidate_identity(row, index)
        if quantity_basis == FIXED_BINARY and (original is None or allocated is None or abs(allocated - original) > EPS):
            granularity.append(identity)
        if original is not None and allocated is not None and float(original).is_integer() and not float(allocated).is_integer():
            integer.append(identity)
        slot = sf.cap_slots(row, index).get(sf.SOURCE_STOCK)
        stock = caps.get(slot) if slot else None
        if stock is None or stock.value is None or float(stock.value) <= 0:
            unmatched.append(identity)
    violations = benchmark["violation_count"] + len(granularity) + len(integer)
    return {
        "violation_count": violations, "cap_violations": benchmark["violations"],
        "source_excess_qty": benchmark["source_excess_qty"], "target_excess_qty": benchmark["target_excess_qty"],
        "route_capacity_excess_qty": benchmark["route_capacity_excess_qty"],
        "dc_capacity_excess_qty": benchmark["dc_capacity_excess_qty"],
        "duplicate_count": benchmark["duplicate_candidate_count"] + benchmark["duplicate_lane_count"],
        "selection_limit_exceeded": benchmark["selection_limit_exceeded"],
        "granularity_violations": granularity, "non_integer_allocations": integer,
        "product_not_verified_at_source": unmatched,
        "strict_unverifiable_constraints": strict["unverifiable_constraints"],
        "passed": violations == 0,
    }


def lexicographic_groups(records: Sequence[Mapping[str, Any]], *, max_routes: int | None,
                         source_cap_field: str = "source_surplus", target_cap_field: str = "target_need_7d"
                         ) -> dict[str, Any]:
    """Constraint groups of the lexicographic MILP, rebuilt independently (tightest cap per group, conflicts flagged)."""
    caps: dict[tuple, list[float]] = defaultdict(list)
    members: dict[tuple, list[int]] = defaultdict(list)
    lanes: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        product, source, target, route_type, dc = sf._ids(row)
        for kind, key, field in (("source", (source, product), source_cap_field), ("target", (target, product), target_cap_field)):
            members[(kind, key)].append(index)
            value = _number(row.get(field))
            if value is not None:
                caps[(kind, key)].append(max(0.0, value))
        lanes[(product, source, target, route_type, dc)].append(index)
    groups, conflicts = [], []
    for slot, indices in members.items():
        values = caps.get(slot)
        if not values:
            continue
        if max(values) - min(values) > EPS:
            conflicts.append({"group": f"{slot[0]}:{'/'.join(slot[1])}", "values": sorted(set(values))})
        groups.append({"kind": slot[0], "indices": indices, "weight": "qty", "cap": min(values)})
    for indices in lanes.values():
        if len(indices) > 1:
            groups.append({"kind": "lane", "indices": indices, "weight": "count", "cap": 1.0})
    return {"groups": groups, "cap_conflicts": conflicts, "max_routes": max_routes}


def enumerate_lexicographic_optimum(records: Sequence[Mapping[str, Any]], *, max_routes: int | None,
                                    source_cap_field: str = "source_surplus", target_cap_field: str = "target_need_7d",
                                    combination_limit: int = ENUMERATION_COMBINATION_LIMIT) -> dict[str, Any]:
    """Exhaustive (exactly pruned) search for max service, then min cost; counts alternative optima.

    Pruning is exact: a branch is cut only when its service upper bound is below the best service, or when it cannot
    exceed the best service and its cost already exceeds the best cost (costs are non-negative).
    """
    n = len(records)
    slots = n if max_routes is None else min(n, int(max_routes))
    combinations = sum(math.comb(n, k) for k in range(slots + 1))
    if combinations > combination_limit:
        return {"status": "NOT_ENUMERATED", "combinations": combinations, "reason_codes": ["ENUMERATION_TOO_LARGE"]}
    qty = [float(sf._first_number(row, sf.QTY_FIELDS)[0] or 0.0) for row in records]
    cost = [float(sf._first_number(row, sf.COST_FIELDS)[0] or 0.0) for row in records]
    if any(value < 0 for value in qty + cost):
        return {"status": "NOT_ENUMERATED", "combinations": combinations, "reason_codes": ["INCONSISTENT_BOUND"]}
    built = lexicographic_groups(records, max_routes=max_routes, source_cap_field=source_cap_field,
                                 target_cap_field=target_cap_field)
    by_index: dict[int, list[tuple[int, float]]] = defaultdict(list)
    caps = []
    for group_index, group in enumerate(built["groups"]):
        caps.append(group["cap"])
        for index in group["indices"]:
            by_index[index].append((group_index, qty[index] if group["weight"] == "qty" else 1.0))
    order = {value: rank for rank, value in enumerate(sorted(str(row.get("route_id") or "") for row in records), start=1)}
    tie = [order[str(row.get("route_id") or "")] for row in records]
    suffix_best = [sorted(qty[i:], reverse=True) for i in range(n + 1)]
    usage = [0.0] * len(caps)
    best = {"service": -1.0, "cost": math.inf, "optima": [], "visited": 0}

    def visit(position: int, chosen: list[int], service: float, spent: float) -> None:
        best["visited"] += 1  # every visited node is a feasible selection
        if service > best["service"] + EPS:
            best.update(service=service, cost=spent, optima=[list(chosen)])
        elif abs(service - best["service"]) <= EPS:
            if spent < best["cost"] - 1e-6:
                best.update(cost=spent, optima=[list(chosen)])
            elif abs(spent - best["cost"]) <= 1e-6:
                best["optima"].append(list(chosen))
        room = slots - len(chosen)
        if room <= 0:
            return
        for index in range(position, n):
            upper = service + sum(suffix_best[index][:room])
            if upper < best["service"] - EPS:
                return
            if upper <= best["service"] + EPS and spent > best["cost"] + 1e-6:
                return
            if any(usage[g] + w > caps[g] + 1e-8 for g, w in by_index[index]):
                continue
            for g, w in by_index[index]:
                usage[g] += w
            chosen.append(index)
            visit(index + 1, chosen, service + qty[index], spent + cost[index])
            chosen.pop()
            for g, w in by_index[index]:
                usage[g] -= w

    visit(0, [], 0.0, 0.0)
    ranked = sorted(best["optima"], key=lambda combo: (sum(tie[i] for i in combo), combo))
    tie_values = [sum(tie[i] for i in combo) for combo in ranked]
    ids = [[str(records[i].get("route_id")) for i in combo] for combo in ranked]
    return {
        "status": "ENUMERATED", "combinations": combinations, "nodes_visited": best["visited"],
        "optimal_service": round(best["service"], 9), "optimal_cost": round(best["cost"], 6),
        "alternative_optima": len(ranked), "tie_break_route_ids": sorted(ids[0]) if ids else [],
        "tie_break_unique": len(tie_values) < 2 or tie_values[0] != tie_values[1],
        "optimal_selections": [sorted(item) for item in ids[:200]],
        "cap_conflicts": built["cap_conflicts"],
        "reason_codes": ["ALTERNATIVE_OPTIMA"] if len(ranked) > 1 else [],
    }


# ---------------------------------------------------------------------------------------------- contract


def benchmark_contract(**values: Any) -> dict[str, Any]:
    """Parallel benchmark contract: every field present, missing values NULL (never filled with 0)."""
    contract = {field: values.get(field) for field in CONTRACT_FIELDS}
    contract["reason_codes"] = _unique(values.get("reason_codes") or [])
    contract["global_optimum_claimable"] = False
    contract["contract_version"] = BENCHMARK_INTEGRITY_VERSION
    return contract


def app_gap_benchmark_integrity(result: Mapping[str, Any]) -> dict[str, Any]:
    """Contract for one ``optimality_gap_service.run_optimality_gap`` result (saving objective, in-app pool)."""
    settings = dict(result.get("settings") or {})
    summary = dict(result.get("summary") or {})
    search = dict(result.get("search") or {})
    gap = dict(result.get("gap") or {})
    combinations = dict(result.get("combinations") or {})
    best = dict(combinations.get("best") or {})
    status = normalize_search_status(search)
    pool_total, pool_size = summary.get("input_count"), summary.get("ranked_candidate_count")
    scope = benchmark_scope(solver_status=status["solver_status"], pool_size=pool_size, pool_total=pool_total,
                            feasible_count=summary.get("feasible_candidate_count"))
    reasons = [*scope["reason_codes"], "INPUT_POOL_MAY_BE_PRE_CUT", "CAP_PROVENANCE_NOT_TRACKED",
               "SERVICE_NOT_IN_OBJECTIVE", "COST_NOT_IN_OBJECTIVE"]
    if scope["candidate_pool_truncated"]:
        reasons.append("CUTOFF_BY_EVALUATED_STRATEGY_RANK")
    if status["termination_reason"] == "LIMITED_SEARCH_MODE":
        reasons.append("LIMITED_SEARCH_MODE")
    if status["termination_reason"] == "TIME_LIMIT":
        reasons.append("SOLVER_TIME_LIMIT")
    incumbent = _number(best.get("total_saving")) if status["solver_status"] in (OPTIMAL, FEASIBLE_NOT_PROVEN_OPTIMAL) else None
    bound = _number(search.get("upper_bound")) if search.get("bound_reliable") else None
    mip = solver_mip_gap(incumbent, bound, sense="max")
    reasons.extend(mip["reason_codes"])
    if gap.get("available"):
        decision_status = GAP_COMPUTED
        comparison = DIRECTLY_COMPARABLE if status["solver_status"] == OPTIMAL else COMPARABLE_REFERENCE_NOT_PROVEN
        if comparison != DIRECTLY_COMPARABLE:
            reasons.append("REFERENCE_NOT_PROVEN_OPTIMAL")
    else:
        decision_status, comparison = GAP_NOT_COMPUTABLE, INSUFFICIENT_DATA
        reasons.append("NO_POSITIVE_OBJECTIVE")
    rows = sorted((str(row.get("constraint")), str(row.get("status")), str(row.get("scope")))
                  for row in result.get("constraint_rows") or [])
    contract = benchmark_contract(
        benchmark_mode=f"IN_APP_SAVING_GAP/{settings.get('search_mode')}",
        benchmark_scope=scope["benchmark_scope"], scope_flags=scope["scope_flags"],
        candidate_pool_size=pool_size, candidate_pool_total=pool_total,
        candidate_pool_truncated=scope["candidate_pool_truncated"],
        candidate_cutoff_basis="varo_final_rank -> vhs_rank -> rank (the evaluated Varo order), applied before "
                               "feasibility/saving exclusion",
        candidate_limit=settings.get("candidate_limit"), max_routes=settings.get("max_routes"),
        time_limit_s=settings.get("time_limit"), quantity_granularity=FIXED_BINARY,
        objective_definition=SAVING_OBJECTIVE, constraint_signature=_digest(rows),
        cap_provenance={"status": "NOT_TRACKED", "constraint_version": (result.get("metadata") or {}).get("constraint_version")},
        cost_provenance={"status": "NOT_IN_OBJECTIVE"},
        solver_status=status["solver_status"], termination_reason=status["termination_reason"],
        solver_status_code=status.get("solver_status_code"),
        solver_tolerance=HIGHS_DEFAULT_TOLERANCE if "status_code" in search else None,
        incumbent_objective=incumbent, best_bound=bound, solver_mip_gap=mip["solver_mip_gap"],
        solver_mip_gap_abs=mip["solver_mip_gap_abs"], solver_mip_gap_status=mip["solver_mip_gap_status"],
        decision_service_gap=None, decision_cost_gap=None, decision_gap_status=decision_status,
        comparison_status=comparison, optimality_claim=scope["optimality_claim"],
        limitations=["objective is expected_saving; the Suhyup 'Varo = MILP' benchmark uses service-then-cost",
                     "caps are shared-feasibility-v1 (7-day demand - stock), not the offline median-based caps"],
        reason_codes=reasons,
    )
    contract.update({
        "feasible_candidate_count": summary.get("feasible_candidate_count"),
        "decision_objective": "expected_saving",
        "decision_objective_gap_pct": gap.get("gap_pct") if gap.get("available") else None,
        "decision_gap_label": gap.get("label"),
        "solver_reported_mip_gap": search.get("solver_gap"),
        "search_exhausted": search.get("search_exhausted"),
    })
    return contract
