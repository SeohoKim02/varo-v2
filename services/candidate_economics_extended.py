"""Candidate-only scheduled-flow and scenario-tree extension of the v0.1 model.

The original solver remains the default for old inputs. A period is explicitly
``arrivals -> immutable departures -> new departures -> sales -> expiry -> hold``.
Signals dated t are observable at the END of t; a decision at t cannot use them.
Demand response is a supplied scenario assumption, not recovered latent demand.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, milp

from services.candidate_economics import EconomicProblem, _Model, _canonical, _money, validate_problem

VERSION = "candidate-economics-scheduled-tree-v0.1"


def history_key(scenario, period: int) -> str:
    """Only signals whose observation period has already ended are readable."""
    return json.dumps([(int(t), str(v)) for t, v in sorted(scenario.observations.items())
                       if int(t) < period], ensure_ascii=False, separators=(",", ":"))


def _value(money) -> float:
    return 0.0 if money.treatment == "NOT_APPLICABLE" else float(money.field.value)


def validate_extension(p: EconomicProblem) -> list[dict[str, Any]]:
    issues = validate_problem(p)
    lots = {x.lot_id: x for x in p.lots}
    routes = {x.route_id: x for x in p.routes}
    def fail(key, reason):
        issues.append({"input": str(key), "classification": reason})
    if not math.isfinite(p.period_days) or p.period_days <= 0:
        fail("period_days", "INVALID_TIME_UNIT")
    ids = [x.event_id for x in p.confirmed_flows] + [x.commitment_id for x in p.committed_transfers]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        fail("commitments", "DUPLICATE_EVENT_ID")
    for event in p.confirmed_flows:
        lot = lots.get(event.lot_id)
        if (lot is None or event.location not in p.storage_capacity
                or not isinstance(event.period, int) or not 0 <= event.period < p.horizon
                or event.kind not in {"CONFIRMED_INBOUND", "CONFIRMED_OUTBOUND", "RESERVED_OUTBOUND"}
                or not math.isfinite(event.quantity) or event.quantity < 0):
            fail(event.event_id, "INVALID_CONFIRMED_FLOW")
        elif (abs(event.quantity / lot.quantum - round(event.quantity / lot.quantum)) > 1e-7
              or event.period > lot.expiry_period):
            fail(event.event_id, "FLOW_QUANTUM_OR_EXPIRY_MISMATCH")
    for event in p.committed_transfers:
        lot, route = lots.get(event.lot_id), routes.get(event.route_id)
        if (lot is None or route is None or not isinstance(event.departure_period, int)
                or not isinstance(event.vehicle_count, int) or event.vehicle_count < 1
                or not math.isfinite(event.quantity) or event.quantity <= 0):
            fail(event.commitment_id, "INVALID_COMMITTED_TRANSFER")
            continue
        arrival = event.departure_period + route.transit_periods
        if (not 0 <= arrival < p.horizon or arrival > lot.expiry_period
                or event.departure_period >= p.horizon
                or abs(event.quantity / lot.quantum - round(event.quantity / lot.quantum)) > 1e-7
                or lot.state not in route.allowed_states
                or event.quantity * lot.capacity_per_unit > event.vehicle_count * route.vehicle_capacity + 1e-7):
            fail(event.commitment_id, "COMMITMENT_TIME_CAPACITY_OR_STATE_MISMATCH")
        if event.departure_period >= 0 and event.departure_period not in route.departure_periods:
            fail(event.commitment_id, "COMMITTED_DEPARTURE_POLICY_MISMATCH")
    committed_vehicle_counts = {}
    for event in p.committed_transfers:
        if event.route_id in routes and event.departure_period >= 0:
            key = event.route_id, event.departure_period
            committed_vehicle_counts[key] = committed_vehicle_counts.get(key, 0) + event.vehicle_count
    for (rid, period), count in committed_vehicle_counts.items():
        if count > routes[rid].max_vehicles:
            fail((rid, period), "COMMITTED_VEHICLE_CAPACITY_EXCEEDED")
    for location, money in p.handling_per_capacity.items():
        if location not in p.storage_capacity:
            fail(location, "HANDLING_LOCATION_UNKNOWN")
        _money(money, f"{location}.handling", p.currency, p.capacity_unit, issues)
    for key, value in p.processing_capacity.items():
        if (len(key) != 2 or key[0] not in p.storage_capacity or not isinstance(key[1], int)
                or not 0 <= key[1] < p.horizon or not math.isfinite(value) or value < 0):
            fail(key, "PROCESSING_CAPACITY_UNIT_OR_PERIOD_INVALID")
    for (lid, period), money in p.holding_by_period.items():
        if lid not in lots or not isinstance(period, int) or not 0 <= period < p.horizon:
            fail((lid, period), "HOLDING_PERIOD_INVALID")
        else:
            _money(money, f"{lid}.{period}.holding", p.currency, lots[lid].unit, issues)
    markets = {m.market_id: m for m in p.markets}
    for mid, money in p.shortage_penalties.items():
        if mid not in markets:
            fail(mid, "PENALTY_MARKET_UNKNOWN")
        else:
            _money(money, f"{mid}.penalty", p.currency, markets[mid].unit, issues)
            if not money.reason:
                fail(mid, "PENALTY_JUSTIFICATION_REQUIRED")
    extra = list(p.handling_per_capacity.values()) + list(p.holding_by_period.values()) + list(p.shortage_penalties.values())
    if p.evidence_class != "SCENARIO" and any(m.field.provenance in {"SCENARIO_INPUT", "CONFIG"} for m in extra):
        fail("economic_evidence", "SCENARIO_NOT_REAL_EVIDENCE")
    all_money = extra + [getattr(l, key) for l in p.lots for key in (
        "holding_per_unit_period", "disposal_per_unit", "terminal_per_unit")]
    all_money += [v for r in p.routes for v in (r.fixed_per_vehicle, r.variable_per_capacity)]
    all_money += [o.unit_price for m in p.markets for o in m.options]
    if len({m.field.dataset for m in all_money if m.field.usable and m.field.dataset}) > 1:
        fail("economic_evidence", "CROSS_DATASET_ECONOMICS")
    for scenario in p.scenarios:
        if any(not isinstance(t, int) or not 0 <= t < p.horizon or not isinstance(v, str)
               for t, v in scenario.observations.items()):
            fail(scenario.scenario_id, "OBSERVATION_TIME_INVALID")
    return issues


def solve_extended(p: EconomicProblem, *, risk="expected", time_limit=30.0, allow_transfers=True,
                   policy="OPEN_LOOP", fixed_transfers=None, transfer_upper_bounds=None,
                   fixed_prices=None, fixed_sales=None, initial_inventory=None) -> dict[str, Any]:
    start = time.perf_counter()
    issues = validate_extension(p)
    if policy not in {"OPEN_LOOP", "MULTISTAGE_RECOURSE", "PERFECT_INFORMATION"}:
        issues.append({"input": "policy", "classification": "UNSUPPORTED_POLICY"})
    if risk not in {"expected", "worst_case"}:
        issues.append({"input": "risk", "classification": "UNSUPPORTED_RISK"})
    legal_moves = {(l.lot_id, r.route_id, t): l.quantum for l in p.lots for r in p.routes for t in r.departure_periods
        if t + r.transit_periods < p.horizon and t + r.transit_periods <= l.expiry_period and l.state in r.allowed_states}
    for label, supplied in (("fixed_transfers", fixed_transfers), ("transfer_upper_bounds", transfer_upper_bounds)):
        for key, value in (supplied or {}).items():
            try:
                valid = key in legal_moves and math.isfinite(value) and value >= 0
                if label == "fixed_transfers" and valid:
                    valid = abs(value / legal_moves[key] - round(value / legal_moves[key])) < 1e-7
            except (TypeError, ValueError): valid = False
            if not valid: issues.append({"input": str(key), "classification": "INVALID_" + label.upper()})
    legal_prices = {(m.market_id, m.period): {o.level for o in m.options} for m in p.markets}
    for key, value in (fixed_prices or {}).items():
        if key not in legal_prices or value not in legal_prices[key]:
            issues.append({"input": str(key), "classification": "INVALID_FIXED_PRICE"})
    lots_by_id = {lot.lot_id: lot for lot in p.lots}
    if initial_inventory is not None:
        expected_keys = {(lot.lot_id, loc) for lot in p.lots for loc in p.storage_capacity}
        if set(initial_inventory) != expected_keys:
            issues.append({"input": "initial_inventory", "classification": "INCOMPLETE_INITIAL_INVENTORY"})
        for key, value in initial_inventory.items():
            lot = lots_by_id.get(key[0]) if isinstance(key, tuple) and len(key) == 2 else None
            try:
                valid = (key in expected_keys and lot is not None and math.isfinite(value) and value >= 0
                         and abs(value / lot.quantum - round(value / lot.quantum)) < 1e-7)
            except (TypeError, ValueError):
                valid = False
            if not valid:
                issues.append({"input": str(key), "classification": "INVALID_INITIAL_INVENTORY"})
        for lot in p.lots:
            values = [initial_inventory.get((lot.lot_id, loc), 0) for loc in p.storage_capacity]
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) and abs(sum(values) - lot.stock) > 1e-7:
                issues.append({"input": lot.lot_id, "classification": "INITIAL_INVENTORY_TOTAL_MISMATCH"})
    signature_data = _canonical(p)
    if initial_inventory is not None:
        signature_data["observed_initial_inventory"] = sorted((str(k), v) for k, v in initial_inventory.items())
    signature = hashlib.sha256(json.dumps(signature_data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    out = {"version": VERSION, "status": "INPUT_REQUIRED", "mode": "NEW_CANDIDATE",
           "evidence_class": p.evidence_class, "production_approved": False,
           "problem_signature": signature, "risk": risk, "policy": policy,
           "input_issues": issues, "expected_net_value": None, "actual_savings": None,
           "absolute_loss": None, "accounting_scope": p.accounting_scope,
           "data_interpretation": "MODEL_BASED_LATENT_DEMAND_SCENARIO_NOT_OBSERVED_SALES_REPLAY",
           "cost_units": {"holding": f"{p.currency}/lot quantity unit/{p.period_days:g} days",
               "transport_fixed": f"{p.currency}/vehicle trip", "transport_variable": f"{p.currency}/{p.capacity_unit}",
               "handling": f"{p.currency}/{p.capacity_unit} handled", "processing": f"{p.capacity_unit}/period"},
           "limitations": ["Finite scenario distribution and finite supplied price responses",
               "No causal demand or realised savings claim", "Known commitments cannot be cancelled",
               "Perfect information is an oracle and cannot be used as an operational policy"]}
    if issues:
        return out
    lots = {x.lot_id: x for x in sorted(p.lots, key=lambda x: x.lot_id)}
    routes = {x.route_id: x for x in sorted(p.routes, key=lambda x: x.route_id)}
    markets = {x.market_id: x for x in sorted(p.markets, key=lambda x: x.market_id)}
    scenarios = sorted(p.scenarios, key=lambda x: x.scenario_id)
    model = _Model()
    shared = {}
    cash = {s.scenario_id: {} for s in scenarios}
    cash_constant = {s.scenario_id: 0.0 for s in scenarios}
    components = {s.scenario_id: {k: {} for k in ("external_sales_revenue", "transport_cost", "handling_cost",
        "holding_cost", "disposal_cost", "terminal_value", "justified_penalty")} for s in scenarios}
    constants = {s.scenario_id: {k: 0.0 for k in components[s.scenario_id]} for s in scenarios}
    moves, vehicles, prices, sales, inventory, disposal = {}, {}, {}, {}, {}, {}
    signs = {"external_sales_revenue": 1, "terminal_value": 1}
    def add(row, index, value):
        row[index] = row.get(index, 0.0) + value
    def cost(sid, kind, index, value):
        add(components[sid][kind], index, value)
        add(cash[sid], index, signs.get(kind, -1) * value)
    def constant(sid, kind, value):
        constants[sid][kind] += value
        cash_constant[sid] += signs.get(kind, -1) * value
    def variable(key, ub, integer=True):
        if key not in shared:
            shared[key] = model.variable(key, ub, integer)
        return shared[key]
    def node(s, t):
        return "OPEN" if policy == "OPEN_LOOP" else s.scenario_id if policy == "PERFECT_INFORMATION" else history_key(s, t)
    def handling(loc):
        return _value(p.handling_per_capacity[loc]) if loc in p.handling_per_capacity else 0.0
    def initial(lid, loc):
        if initial_inventory is not None:
            return initial_inventory[lid, loc]
        return lots[lid].stock if lots[lid].source == loc else 0.0
    # Fixed flows are true boundary events. In-transit stock is excluded from on-hand.
    arrivals, departures = {}, {}
    for e in p.confirmed_flows:
        target = arrivals if e.kind == "CONFIRMED_INBOUND" else departures
        key = (e.lot_id, e.location, e.period)
        target[key] = target.get(key, 0.0) + e.quantity
    for e in p.committed_transfers:
        r = routes[e.route_id]
        key = (e.lot_id, r.target, e.departure_period + r.transit_periods)
        arrivals[key] = arrivals.get(key, 0.0) + e.quantity
        if e.departure_period >= 0:
            key = (e.lot_id, r.source, e.departure_period)
            departures[key] = departures.get(key, 0.0) + e.quantity
    total = {lid: l.stock + sum(e.quantity for e in p.confirmed_flows if e.lot_id == lid and e.kind == "CONFIRMED_INBOUND")
             + sum(e.quantity for e in p.committed_transfers if e.lot_id == lid and e.departure_period < 0)
             for lid, l in lots.items()}
    fixed_vehicle = {}
    for e in p.committed_transfers:
        if e.departure_period >= 0:
            key = (e.route_id, e.departure_period)
            fixed_vehicle[key] = fixed_vehicle.get(key, 0) + e.vehicle_count
    for s in scenarios:
        sid = s.scenario_id
        for e in p.committed_transfers:
            r, lot = routes[e.route_id], lots[e.lot_id]
            if not e.cost_already_paid:
                constant(sid, "transport_cost", e.vehicle_count * _value(r.fixed_per_vehicle)
                         + e.quantity * lot.capacity_per_unit * _value(r.variable_per_capacity))
        for table in (arrivals, departures):
            for (lid, loc, _), q in table.items():
                constant(sid, "handling_cost", q * lots[lid].capacity_per_unit * handling(loc))
        for rid, r in routes.items():
            for t in sorted(r.departure_periods):
                committed_n = fixed_vehicle.get((rid, t), 0)
                if committed_n > r.max_vehicles:
                    model.row({}, lower=1, upper=1)  # immutable infeasibility, never silently repaired
                if t + r.transit_periods >= p.horizon:
                    continue
                n = variable(("vehicle", node(s, t), rid, t), max(0, r.max_vehicles - committed_n))
                vehicles[sid, rid, t] = n
                cost(sid, "transport_cost", n, _value(r.fixed_per_vehicle))
                capacity = {n: -r.vehicle_capacity}
                for lid, lot in lots.items():
                    if lot.state not in r.allowed_states or t + r.transit_periods > lot.expiry_period:
                        continue
                    upper = total[lid] if allow_transfers else 0.0
                    if transfer_upper_bounds is not None:
                        upper = min(upper, transfer_upper_bounds.get((lid, rid, t), 0.0))
                    q = variable(("move", node(s, t), lid, rid, t), upper / lot.quantum)
                    moves[sid, lid, rid, t] = q
                    add(capacity, q, lot.quantum * lot.capacity_per_unit)
                    cost(sid, "transport_cost", q, lot.quantum * lot.capacity_per_unit * _value(r.variable_per_capacity))
                    cost(sid, "handling_cost", q, lot.quantum * lot.capacity_per_unit * (handling(r.source) + handling(r.target)))
                    if fixed_transfers is not None and (lid, rid, t) in fixed_transfers:
                        fixed = fixed_transfers[lid, rid, t] / lot.quantum
                        model.row({q: 1}, fixed, fixed)
                model.row(capacity, upper=0)
        for lid, lot in lots.items():
            if lot.approved_departure_qty is not None:
                committed_q = sum(e.quantity for e in p.committed_transfers if e.lot_id == lid
                    and e.departure_period >= 0 and routes[e.route_id].source == lot.source)
                terms = {q: lot.quantum for (ss, ll, rr, _), q in moves.items()
                    if ss == sid and ll == lid and routes[rr].source == lot.source}
                model.row(terms, upper=lot.approved_departure_qty - committed_q)
        for (loc, product, state), limit in p.receipt_limits.items():
            terms = {q: lots[lid].quantum for (ss, lid, rid, _), q in moves.items()
                if ss == sid and routes[rid].target == loc and lots[lid].product_id == product and lots[lid].state == state}
            committed_q = sum(e.quantity for e in p.committed_transfers if routes[e.route_id].target == loc
                and lots[e.lot_id].product_id == product and lots[e.lot_id].state == state)
            model.row(terms, upper=limit - committed_q)
        for mid, market in markets.items():
            choice = {}
            penalty = _value(p.shortage_penalties[mid]) if mid in p.shortage_penalties else 0.0
            for option in sorted(market.options, key=lambda o: o.level):
                y = variable(("price", node(s, market.period), mid, option.level), 1)
                prices[sid, mid, option.level] = y
                choice[y] = 1
                if fixed_prices is not None and (mid, market.period) in fixed_prices:
                    target = int(fixed_prices[mid, market.period] == option.level)
                    model.row({y: 1}, target, target)
                demand = {y: -option.demand[sid]}
                cost(sid, "justified_penalty", y, option.demand[sid] * penalty)
                for lid, lot in lots.items():
                    if (lot.product_id, lot.state, lot.unit) != (market.product_id, market.state, market.unit) or lot.expiry_period < market.period:
                        continue
                    # Demand is observed before sale, not before departure/price.
                    # Only the response for THIS selected price is observable.
                    # Do not use counterfactual demand under unchosen prices as
                    # a signal that reveals the scenario/future.
                    # End-of-period signals have not arrived at the sale event.
                    history = (history_key(s, market.period), market.market_id, option.level, option.demand[sid])
                    sale_node = sid if policy in {"OPEN_LOOP", "PERFECT_INFORMATION"} else repr(history)
                    z = variable(("sale", sale_node, lid, mid, option.level), total[lid] / lot.quantum)
                    sales[sid, lid, mid, option.level] = z
                    if fixed_sales is not None and (lid, mid, option.level) in fixed_sales:
                        fixed = fixed_sales[lid, mid, option.level] / lot.quantum
                        model.row({z: 1}, fixed, fixed)
                    add(demand, z, lot.quantum)
                    cost(sid, "external_sales_revenue", z, lot.quantum * _value(option.unit_price))
                    cost(sid, "justified_penalty", z, -lot.quantum * penalty)
                    cost(sid, "handling_cost", z, lot.quantum * lot.capacity_per_unit * handling(market.location))
                model.row(demand, upper=0)
            model.row(choice, 1, 1)
        for lid, lot in lots.items():
            for loc in sorted(p.storage_capacity):
                for t in range(p.horizon):
                    key = sid, lid, loc, t
                    inventory[key] = model.variable(("inventory", *key), 0 if t >= lot.expiry_period else total[lid] / lot.quantum, True)
                    disposal[key] = model.variable(("disposal", *key), total[lid] / lot.quantum if t == lot.expiry_period else 0, True)
                    hold = p.holding_by_period.get((lid, t), lot.holding_per_unit_period)
                    cost(sid, "holding_cost", inventory[key], lot.quantum * _value(hold))
                    cost(sid, "disposal_cost", disposal[key], lot.quantum * _value(lot.disposal_per_unit))
                    if t == p.horizon - 1:
                        cost(sid, "terminal_value", inventory[key], lot.quantum * _value(lot.terminal_per_unit))
        for lid, lot in lots.items():
            for loc in sorted(p.storage_capacity):
                for t in range(p.horizon):
                    key = sid, lid, loc, t
                    balance = {inventory[key]: 1, disposal[key]: 1}
                    if t: add(balance, inventory[sid, lid, loc, t - 1], -1)
                    for rid, route in routes.items():
                        q = moves.get((sid, lid, rid, t))
                        if q is not None and route.source == loc: add(balance, q, 1)
                        q = moves.get((sid, lid, rid, t - route.transit_periods))
                        if q is not None and route.target == loc: add(balance, q, -1)
                    for mid, market in markets.items():
                        if market.location == loc and market.period == t:
                            for o in market.options:
                                z = sales.get((sid, lid, mid, o.level))
                                if z is not None: add(balance, z, 1)
                    rhs = initial(lid, loc) if t == 0 else 0.0
                    rhs += arrivals.get((lid, loc, t), 0.0) - departures.get((lid, loc, t), 0.0)
                    model.row(balance, rhs / lot.quantum, rhs / lot.quantum)
        for loc in sorted(p.storage_capacity):
            for t in range(p.horizon):
                peak, processing = {}, {}
                peak_fixed = throughput_fixed = 0.0
                for lid, lot in lots.items():
                    size = lot.quantum * lot.capacity_per_unit
                    peak_fixed += ((initial(lid, loc) if t == 0 else 0.0)
                                   + arrivals.get((lid, loc, t), 0.0)) * lot.capacity_per_unit
                    throughput_fixed += (arrivals.get((lid, loc, t), 0.0) + departures.get((lid, loc, t), 0.0)) * lot.capacity_per_unit
                    if t: add(peak, inventory[sid, lid, loc, t - 1], size)
                    for rid, r in routes.items():
                        q = moves.get((sid, lid, rid, t - r.transit_periods))
                        if q is not None and r.target == loc:
                            add(peak, q, size); add(processing, q, size)
                        q = moves.get((sid, lid, rid, t))
                        if q is not None and r.source == loc: add(processing, q, size)
                    for mid, m in markets.items():
                        if m.location == loc and m.period == t:
                            for o in m.options:
                                z = sales.get((sid, lid, mid, o.level))
                                if z is not None: add(processing, z, size)
                model.row(peak, upper=p.storage_capacity[loc] - peak_fixed)
                if (loc, t) in p.processing_capacity:
                    model.row(processing, upper=p.processing_capacity[loc, t] - throughput_fixed)
    worst = None
    if risk == "worst_case":
        pos, neg = model.variable(("worst_positive",)), model.variable(("worst_negative",))
        worst = pos, neg
        for sid, terms in cash.items():
            row = dict(terms); add(row, pos, -1); add(row, neg, 1)
            model.row(row, lower=-cash_constant[sid])
    c = np.zeros(len(model.names))
    objective_constant = 0.0
    if worst:
        c[worst[0]], c[worst[1]] = -1, 1
    else:
        for s in scenarios:
            for i, v in cash[s.scenario_id].items(): c[i] -= s.probability * v
            objective_constant += s.probability * cash_constant[s.scenario_id]
    result = milp(c, integrality=np.array(model.integer), bounds=Bounds(0, model.ub),
                  constraints=model.constraint(), options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    out.update(status="OPTIMAL" if result.status == 0 else "FEASIBLE_LIMIT" if result.x is not None else "NO_SOLUTION",
        solver_status=int(result.status), solver_message=str(result.message), solver_gap=getattr(result, "mip_gap", None),
        optimality_proven=result.status == 0, runtime_seconds=time.perf_counter() - start,
        variables=len(model.names), constraints=len(model.rows),
        objective="EXPECTED_NET_CASH_MAX" if risk == "expected" else "WORST_SCENARIO_NET_CASH_MAX",
        event_order="arrival -> irrevocable departure -> discretionary departure -> sale -> expiry disposal -> holding",
        nonanticipativity="Decision node uses only signals observed before its period; current demand is used only for sales",
        operational_policy=policy != "PERFECT_INFORMATION", committed_events=[asdict(x) for x in p.confirmed_flows + p.committed_transfers])
    if result.x is None: return out
    x = result.x
    profits = {sid: cash_constant[sid] + sum(x[i] * v for i, v in terms.items()) for sid, terms in cash.items()}
    out.update(expected_net_value=float(sum(s.probability * profits[s.scenario_id] for s in scenarios)),
        worst_net_value=float(min(profits.values())), scenario_net_values={k: float(v) for k, v in profits.items()},
        objective_value=float(-result.fun + objective_constant),
        bound_net_value=float(-result.mip_dual_bound + objective_constant) if getattr(result, "mip_dual_bound", None) is not None else None)
    out["scenario_transfers"] = [{"scenario": sid, "lot_id": lid, "route_id": rid, "departure_period": t,
        "quantity": float(x[i] * lots[lid].quantum), "information_set": node(next(s for s in scenarios if s.scenario_id == sid), t)}
        for (sid, lid, rid, t), i in moves.items() if x[i] > 1e-6]
    out["scenario_vehicles"] = [{"scenario": sid, "route_id": rid, "period": t, "count": int(round(x[i]))}
        for (sid, rid, t), i in vehicles.items() if x[i] > 1e-6]
    out["scenario_price_levels"] = [{"scenario": sid, "market_id": mid, "period": markets[mid].period, "level": level,
        "information_set": node(next(s for s in scenarios if s.scenario_id == sid), markets[mid].period)}
        for (sid, mid, level), i in prices.items() if x[i] > .5]
    # Backward-compatible common-plan fields are never used to flatten adaptive actions.
    first = scenarios[0].scenario_id
    out["transfers"] = [{k: v for k, v in row.items() if k not in {"scenario", "information_set"}}
        for row in out["scenario_transfers"] if row["scenario"] == first] if policy == "OPEN_LOOP" else []
    out["vehicles"] = [{k: v for k, v in row.items() if k != "scenario"}
        for row in out["scenario_vehicles"] if row["scenario"] == first] if policy == "OPEN_LOOP" else []
    out["price_levels"] = [{"market_id": row["market_id"], "level": row["level"]}
        for row in out["scenario_price_levels"] if row["scenario"] == first] if policy == "OPEN_LOOP" else []
    out["sales"] = [{"scenario": sid, "lot_id": lid, "market_id": mid, "level": level,
        "quantity": float(x[i] * lots[lid].quantum)} for (sid, lid, mid, level), i in sales.items() if x[i] > 1e-6]
    out["inventory"] = [{"scenario": sid, "lot_id": lid, "location": loc, "period": t,
        "quantity": float(x[i] * lots[lid].quantum), "disposal": float(x[disposal[sid, lid, loc, t]] * lots[lid].quantum)}
        for (sid, lid, loc, t), i in inventory.items()]
    out["in_transit"] = [{"scenario": sid, "lot_id": lid, "period": t,
        "quantity": sum(float(x[i] * lots[lid].quantum) for (ss, ll, rid, depart), i in moves.items()
            if ss == sid and ll == lid and depart <= t < depart + routes[rid].transit_periods)
          + sum(e.quantity for e in p.committed_transfers if e.lot_id == lid
                and e.departure_period <= t < e.departure_period + routes[e.route_id].transit_periods)}
        for sid in cash for lid in lots for t in range(p.horizon)]
    out["accounting_components"] = {}
    for sid, parts in components.items():
        values = {kind: float(constants[sid][kind] + sum(x[i] * v for i, v in terms.items())) for kind, terms in parts.items()}
        values["net_cash_value"] = float(profits[sid])
        out["accounting_components"][sid] = values
    out["scenario_cash_components"] = out["accounting_components"]
    out["transport_cost"] = float(sum(s.probability * out["accounting_components"][s.scenario_id]["transport_cost"] for s in scenarios))
    out["market_outcomes"] = []
    for s in scenarios:
        for mid, m in markets.items():
            o = next(o for o in m.options if x[prices[s.scenario_id, mid, o.level]] > .5)
            sold = sum(row["quantity"] for row in out["sales"] if row["scenario"] == s.scenario_id and row["market_id"] == mid)
            out["market_outcomes"].append({"scenario": s.scenario_id, "market_id": mid, "period": m.period,
                "level": o.level, "demand": o.demand[s.scenario_id], "sold": sold, "unmet": max(0., o.demand[s.scenario_id] - sold)})
    activity = model.constraint().A @ x
    lower, upper = np.array(model.lower), np.array(model.upper)
    out["constraint_max_violation"] = float(max(np.max(np.maximum(lower - activity, 0)), np.max(np.maximum(activity - upper, 0))))
    out["constraint_violations"] = int(np.count_nonzero((activity < lower - 1e-6) | (activity > upper + 1e-6)))
    out["integrality_max_violation"] = float(max((abs(x[i] - round(x[i])) for i, yes in enumerate(model.integer) if yes), default=0.0))
    return out
