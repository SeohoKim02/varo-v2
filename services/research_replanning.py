"""Bounded research simulator and past-only policy selection.

Separate from production planners. Prices, capacity, expiry and stock are
declared scenario assumptions. Observed sales are only a demand-pattern proxy.
"""
from __future__ import annotations

from functools import lru_cache
from math import floor, isfinite, sqrt
from numbers import Integral, Real
from statistics import fmean

POLICIES = ("ALWAYS", "PERIODIC_2", "THRESHOLD")
HORIZON = 6
INITIAL_STOCK = 12
CAPACITY = 4
PRICE = 8.0
TERMINAL = 2.0
TRIP_COST = 2.0
HOLDING = .2


def _sales(values):
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in values):
        raise ValueError("Sales pattern must contain nonnegative integers, without rounding or imputation")


def forecast(observed, remaining):
    _sales(observed)
    mean = fmean(observed[-2:]) if observed else 3.0
    return (max(0, min(6, int(floor(mean + .5)))),) * remaining


def plan_value(plan, prediction, stock):
    total = 0.0
    for q, d in zip(plan, prediction):
        if not 0 <= q <= min(CAPACITY, stock):
            raise ValueError("Infeasible research plan")
        stock -= q
        total += PRICE * min(q, d) - TRIP_COST * (q > 0) - HOLDING * stock
    return total + TERMINAL * stock


@lru_cache(maxsize=32768)
def optimal_plan(prediction, stock):
    @lru_cache(None)
    def dp(t, available):
        if t == len(prediction):
            return TERMINAL * available, ()
        best = (-float("inf"), ())
        for q in range(min(CAPACITY, available), -1, -1):
            val, tail = dp(t + 1, available - q)
            val += PRICE * min(q, prediction[t]) - TRIP_COST * (q > 0) - HOLDING * (available - q)
            if val > best[0] + 1e-9:
                best = val, (q,) + tail
        return best
    return dp(0, stock)


def run_path(path, policy, fixed, per_unit, initial_observed=()):
    if len(path) != HORIZON or policy not in POLICIES:
        raise ValueError("Provide six observations and a supported policy")
    _sales(path)
    _sales(initial_observed)
    if any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or v < 0 for v in (fixed, per_unit)):
        raise ValueError("Change costs must be finite and nonnegative")
    stock = INITIAL_STOCK
    observed = list(initial_observed)
    _, plan = optimal_plan(forecast(observed, HORIZON), stock)
    gross = fees = 0.0
    changes = changed_quantity = fulfilled = discarded = 0
    trace = []
    for t, realised in enumerate(path):
        prediction = forecast(observed, HORIZON - t)
        old = plan
        old_value = plan_value(old, prediction, stock)
        candidate_value, candidate = optimal_plan(prediction, stock)
        distance = sum(abs(a-b) for a, b in zip(old, candidate))
        fee = fixed + per_unit * distance if distance else 0.0
        eligible = t > 0 and (policy != "PERIODIC_2" or t % 2 == 0)
        accept = eligible and candidate_value > old_value + 1e-8 and (
            policy != "THRESHOLD" or candidate_value - old_value > fee + 1e-8)
        if accept:
            plan = candidate
            fees += fee
            changes += 1
            changed_quantity += distance
        q = plan[0]
        stock -= q
        sold = min(q, realised)
        gross += sold * PRICE - TRIP_COST * (q > 0) - HOLDING * stock
        fulfilled += sold
        discarded += q - sold
        trace.append({"period": t, "observed_prefix": tuple(observed), "forecast": prediction,
                      "previous_plan": old, "candidate_plan": candidate, "chosen_plan": plan,
                      "accepted": bool(accept), "forecast_gain": candidate_value - old_value,
                      "change_cost_if_accepted": fee, "dispatch": q, "observed_after_dispatch": realised,
                      "sold": sold, "discarded": q-sold, "source_remaining": stock})
        observed.append(realised)
        plan = plan[1:]
        if sum(r["dispatch"] for r in trace) + stock != INITIAL_STOCK or sum(plan) > stock:
            raise ArithmeticError("Inventory conservation violated")
    gross += TERMINAL * stock
    return {"policy": policy, "gross_value": gross, "coordination_cost": fees,
            "net_value": gross-fees, "changes": changes, "changed_quantity": changed_quantity,
            "fulfilled": fulfilled, "discarded": discarded, "terminal_stock": stock,
            "feasible": True, "trace": trace}


def observed_regime(history):
    """Features use only the already observed 14-period prefix."""
    _sales(history)
    tail = history[-14:]
    if len(tail) < 14:
        return "INSUFFICIENT_HISTORY"
    level = fmean(tail)
    trend = fmean(tail[-7:]) - fmean(tail[:7])
    cv = sqrt(fmean((v-level)**2 for v in tail)) / max(level, 1e-9)
    if cv > .75:
        return "VARIABLE"
    if trend > .75:
        return "RISING"
    if trend < -.75:
        return "FALLING"
    return "STABLE"


def fit_policy_selector(training_rows, minimum_group_size=20):
    """Freeze a lookup rule from labelled TRAIN rows; evaluation rows rejected."""
    if not training_rows or minimum_group_size < 1:
        raise ValueError("Training observations and a positive minimum group size are required")
    groups = {}
    global_groups = {}
    for row in training_rows:
        if row.get("phase") != "TRAIN":
            raise ValueError("The selector accepts training outcomes only")
        key = (row["dataset"], float(row["fixed"]), float(row["per_unit"]))
        if set(row["results"]) != set(POLICIES):
            raise ValueError("Each training row must compare all three policies")
        for p in POLICIES:
            if any(not isfinite(float(row["results"][p][k])) for k in ("net_value", "changes")):
                raise ValueError("Training outcomes must be finite")
        global_groups.setdefault(key, []).append(row)
        groups.setdefault((*key, row["regime"]), []).append(row)

    def best(rows):
        # Same net value: fewer revisions, then a stable declared policy order.
        return max(POLICIES, key=lambda p: (round(fmean(r["results"][p]["net_value"] for r in rows), 8),
                                           -fmean(r["results"][p]["changes"] for r in rows), -POLICIES.index(p)))
    rules = []
    for key, rows in sorted(global_groups.items()):
        rules.append({"dataset": key[0], "fixed": key[1], "per_unit": key[2], "regime": "FALLBACK",
                      "policy": best(rows), "training_cases": len(rows)})
    for key, rows in sorted(groups.items()):
        if len(rows) >= minimum_group_size:
            rules.append({"dataset": key[0], "fixed": key[1], "per_unit": key[2], "regime": key[3],
                          "policy": best(rows), "training_cases": len(rows)})
    return {"schema_version": 1, "minimum_group_size": minimum_group_size, "rules": rules,
            "training_case_count": len(training_rows), "production_default_changed": False}


def select_policy(selector, dataset, fixed, per_unit, observed_history):
    regime = observed_regime(observed_history)
    applicable = [r for r in selector["rules"] if (r["dataset"], r["fixed"], r["per_unit"]) == (dataset, fixed, per_unit)]
    for wanted in (regime, "FALLBACK"):
        for rule in applicable:
            if rule["regime"] == wanted and rule["policy"] in POLICIES:
                return rule["policy"]
    raise ValueError("No trained rule for this dataset and cost setting; do not extrapolate")
