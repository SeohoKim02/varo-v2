"""Candidate-only, finite-price, lot-preserving network cash-flow MILP.

No production decisions or Seller Loss formula are replaced.  Period events are
arrival -> departure -> sale -> expiry disposal -> holding.  Transfers and prices
are common, open-loop decisions for all demand scenarios; only sales, disposal and
remaining inventory are recourse.  This is deliberately not a multistage policy.
The economic scope is one business: internal movements never create revenue.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import time
from typing import Any, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from services.seller_loss_engine import InputField, MISSING

VERSION = "candidate-economics-v0.1"


@dataclass(frozen=True)
class Money:
    """Use the existing evidence contract; unknown is not silently zero.

    KNOWN requires an explicit finite value. NOT_APPLICABLE requires a reason;
    CANCELS_COMMON is only allowed for the problem's common sunk cost. BOUNDED
    and UNKNOWN are reported as unresolved, not optimistically substituted.
    """
    field: InputField = MISSING
    treatment: str = "UNKNOWN"
    reason: str = ""
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class InventoryLot:
    lot_id: str
    product_id: str
    source: str
    state: str
    unit: str
    stock: float
    expiry_period: int
    capacity_per_unit: float
    holding_per_unit_period: Money
    disposal_per_unit: Money
    terminal_per_unit: Money
    # Granularity is declared; e.g. EA=1, BOX=0.1 for a verified 10-EA box.
    quantum: float = 1.0
    approved_departure_qty: float | None = None


@dataclass(frozen=True)
class Route:
    route_id: str
    source: str
    target: str
    transit_periods: int
    vehicle_capacity: float
    max_vehicles: int
    fixed_per_vehicle: Money
    variable_per_capacity: Money
    allowed_states: tuple[str, ...]
    departure_periods: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class PriceOption:
    level: str
    unit_price: Money
    demand: Mapping[str, float]


@dataclass(frozen=True)
class DemandMarket:
    market_id: str
    location: str
    product_id: str
    state: str
    unit: str
    period: int
    options: tuple[PriceOption, ...]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    probability: float
    # Signals become observable AFTER this period. Never an unseen future label.
    observations: Mapping[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfirmedFlow:
    event_id: str
    lot_id: str
    location: str
    period: int
    quantity: float
    kind: str  # CONFIRMED_INBOUND, CONFIRMED_OUTBOUND or RESERVED_OUTBOUND; not a forecast.


@dataclass(frozen=True)
class CommittedTransfer:
    commitment_id: str
    lot_id: str
    route_id: str
    departure_period: int
    quantity: float
    vehicle_count: int = 1
    cost_already_paid: bool = False


@dataclass(frozen=True)
class EconomicProblem:
    problem_id: str
    horizon: int
    currency: str
    lots: tuple[InventoryLot, ...]
    routes: tuple[Route, ...]
    markets: tuple[DemandMarket, ...]
    scenarios: tuple[Scenario, ...]
    # Capacity units must be one declared physical dimension across products.
    capacity_unit: str
    # Peak arrival inventory capacity, shared by product/state/lot.
    storage_capacity: Mapping[str, float]
    evidence_class: str = "SCENARIO"
    accounting_scope: str = "SINGLE_BUSINESS_NETWORK"
    common_sunk_cost: Money = field(default_factory=lambda: Money(
        treatment="CANCELS_COMMON", reason="Previously purchased stock is common to all decisions"))
    lineage: Mapping[str, str] = field(default_factory=dict)
    receipt_limits: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    confirmed_flows: tuple[ConfirmedFlow, ...] = ()
    committed_transfers: tuple[CommittedTransfer, ...] = ()
    handling_per_capacity: Mapping[str, Money] = field(default_factory=dict)
    processing_capacity: Mapping[tuple[str, int], float] = field(default_factory=dict)
    holding_by_period: Mapping[tuple[str, int], Money] = field(default_factory=dict)
    shortage_penalties: Mapping[str, Money] = field(default_factory=dict)
    period_days: float = 1.0


def _money(value: Money, name: str, currency: str, unit: str | None,
           issues: list[dict[str, Any]]) -> float:
    if value.treatment == "NOT_APPLICABLE" and value.reason:
        return 0.0
    if value.treatment != "KNOWN" or not value.field.usable:
        issues.append({"input": name, "classification": value.treatment,
                       "reason": value.reason or "경제적 순위에 영향을 주는 금액 근거가 필요합니다.",
                       "bounds": [value.lower, value.upper]})
        return 0.0  # Model is NEVER solved when any issue exists.
    f = value.field
    if f.currency != currency or (unit is not None and f.unit != unit):
        issues.append({"input": name, "classification": "UNIT_OR_CURRENCY_MISMATCH"})
    if f.value is None or not math.isfinite(f.value) or f.value < 0:
        issues.append({"input": name, "classification": "INVALID_MONEY"})
        return 0.0
    return float(f.value)


def _canonical(problem: EconomicProblem) -> dict[str, Any]:
    data = asdict(problem)
    data["receipt_limits"] = sorted((list(key), value) for key, value in problem.receipt_limits.items())
    for key in ("processing_capacity", "holding_by_period"):
        data[key] = sorted((list(k), asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                           for k, v in getattr(problem, key).items())
    for key, identifier in (("lots", "lot_id"), ("routes", "route_id"),
                            ("markets", "market_id"), ("scenarios", "scenario_id")):
        data[key] = sorted(data[key], key=lambda x: x[identifier])
    for market in data["markets"]:
        market["options"] = sorted(market["options"], key=lambda x: x["level"])
    return data


def validate_problem(p: EconomicProblem) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    def require(ok: bool, name: str, reason: str = "INVALID_CONTRACT") -> None:
        if not ok:
            issues.append({"input": name, "classification": reason})
    require(p.horizon >= 1 and isinstance(p.horizon, int), "horizon")
    require(p.accounting_scope == "SINGLE_BUSINESS_NETWORK", "accounting_scope")
    require(bool(p.currency) and bool(p.capacity_unit), "currency/capacity_unit")
    require(p.evidence_class in {"SCENARIO", "USER_INPUT", "REAL_MODEL"}, "evidence_class")
    require(bool(p.lots) and bool(p.scenarios), "lots/scenarios")
    for label, values in (("lots", [x.lot_id for x in p.lots]),
                          ("routes", [x.route_id for x in p.routes]),
                          ("markets", [x.market_id for x in p.markets]),
                          ("scenarios", [x.scenario_id for x in p.scenarios])):
        require(len(values) == len(set(values)) and all(values), label, "DUPLICATE_OR_EMPTY_ID")
    require(all(math.isfinite(s.probability) and s.probability > 0 for s in p.scenarios)
            and abs(sum(s.probability for s in p.scenarios) - 1) < 1e-9, "probabilities")
    scenario_ids = {s.scenario_id for s in p.scenarios}
    product_units: dict[tuple[str, str], str] = {}
    for lot in p.lots:
        require(bool(lot.state) and lot.state.upper() != "UNKNOWN" and bool(lot.unit)
                and lot.unit.upper() != "UNKNOWN", lot.lot_id, "STATE_OR_UNIT_UNCONFIRMED")
        require(math.isfinite(lot.stock) and lot.stock >= 0 and math.isfinite(lot.quantum)
                and lot.quantum > 0 and math.isfinite(lot.capacity_per_unit)
                and lot.capacity_per_unit > 0, lot.lot_id, "INVALID_QUANTITY")
        if lot.quantum > 0:
            require(abs(lot.stock / lot.quantum - round(lot.stock / lot.quantum)) < 1e-7,
                    lot.lot_id, "STOCK_NOT_ON_QUANTUM")
        require(isinstance(lot.expiry_period, int) and lot.expiry_period >= 0, lot.lot_id, "EXPIRY_UNCONFIRMED")
        if lot.approved_departure_qty is not None:
            require(math.isfinite(lot.approved_departure_qty) and 0 <= lot.approved_departure_qty <= lot.stock,
                    lot.lot_id, "INVALID_TRANSFER_APPROVAL")
        elif p.evidence_class != "SCENARIO":
            require(False, lot.lot_id, "TRANSFER_APPROVAL_REQUIRED")
        key = (lot.product_id, lot.state)
        require(key not in product_units or product_units[key] == lot.unit, lot.lot_id, "MIXED_UNIT")
        product_units[key] = lot.unit
        for name in ("holding_per_unit_period", "disposal_per_unit", "terminal_per_unit"):
            _money(getattr(lot, name), f"{lot.lot_id}.{name}", p.currency, lot.unit, issues)
    for route in p.routes:
        require(route.source != route.target and route.transit_periods >= 1
                and isinstance(route.transit_periods, int), route.route_id, "INVALID_TRANSIT")
        require(route.vehicle_capacity > 0 and math.isfinite(route.vehicle_capacity)
                and route.max_vehicles >= 0 and isinstance(route.max_vehicles, int), route.route_id)
        require(bool(route.allowed_states) and all(isinstance(t, int) and 0 <= t < p.horizon
                for t in route.departure_periods) and len(set(route.departure_periods))
                == len(route.departure_periods), route.route_id, "INVALID_ROUTE_POLICY")
        _money(route.fixed_per_vehicle, f"{route.route_id}.fixed", p.currency, None, issues)
        _money(route.variable_per_capacity, f"{route.route_id}.variable", p.currency, p.capacity_unit, issues)
    market_keys: set[tuple[Any, ...]] = set()
    for market in p.markets:
        key = (market.location, market.product_id, market.state, market.period)
        require(key not in market_keys, market.market_id, "DUPLICATE_MARKET")
        market_keys.add(key)
        require(0 <= market.period < p.horizon and bool(market.options), market.market_id)
        require(product_units.get((market.product_id, market.state)) == market.unit,
                market.market_id, "MARKET_UNIT_OR_STATE_MISMATCH")
        levels = [o.level for o in market.options]
        require(len(levels) == len(set(levels)), market.market_id, "DUPLICATE_PRICE_LEVEL")
        for option in market.options:
            require(set(option.demand) == scenario_ids and all(math.isfinite(d) and d >= 0
                    for d in option.demand.values()), market.market_id, "DEMAND_SCENARIO_MISMATCH")
            _money(option.unit_price, f"{market.market_id}.{option.level}.price", p.currency, market.unit, issues)
    locations = {l.source for l in p.lots} | {r.source for r in p.routes} | {r.target for r in p.routes}
    locations |= {m.location for m in p.markets}
    for loc in locations:
        require(loc in p.storage_capacity and math.isfinite(p.storage_capacity.get(loc, float("nan")))
                and p.storage_capacity.get(loc, -1) >= 0, loc, "STORAGE_CAPACITY_UNCONFIRMED")
    require(p.common_sunk_cost.treatment == "CANCELS_COMMON" and bool(p.common_sunk_cost.reason),
            "common_sunk_cost", "ACCOUNTING_SCOPE_UNCONFIRMED")
    money_fields = [getattr(l, key) for l in p.lots for key in (
        "holding_per_unit_period", "disposal_per_unit", "terminal_per_unit")]
    money_fields += [v for r in p.routes for v in (r.fixed_per_vehicle, r.variable_per_capacity)]
    money_fields += [o.unit_price for m in p.markets for o in m.options]
    if p.evidence_class != "SCENARIO":
        require(not any(v.field.provenance in {"SCENARIO_INPUT", "CONFIG"} for v in money_fields),
                "economic_evidence", "SCENARIO_NOT_REAL_EVIDENCE")
    datasets = {v.field.dataset for v in money_fields if v.field.usable and v.field.dataset}
    require(len(datasets) <= 1, "economic_evidence", "CROSS_DATASET_ECONOMICS")
    for key, value in p.receipt_limits.items():
        require(isinstance(key, tuple) and len(key) == 3 and math.isfinite(value) and value >= 0,
                str(key), "INVALID_RECEIPT_LIMIT")
    return issues


class _Model:
    def __init__(self) -> None:
        self.names: list[tuple[Any, ...]] = []
        self.ub: list[float] = []
        self.integer: list[int] = []
        self.rows: list[dict[int, float]] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def variable(self, name: tuple[Any, ...], ub: float = np.inf, integer: bool = False) -> int:
        self.names.append(name); self.ub.append(ub); self.integer.append(int(integer))
        return len(self.names) - 1

    def row(self, terms: Mapping[int, float], lower: float = -np.inf, upper: float = np.inf) -> None:
        self.rows.append(dict(terms)); self.lower.append(lower); self.upper.append(upper)

    def constraint(self) -> LinearConstraint:
        rr, cc, vv = [], [], []
        for r, terms in enumerate(self.rows):
            for c, v in terms.items():
                rr.append(r); cc.append(c); vv.append(v)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(self.rows), len(self.names))).tocsc()
        return LinearConstraint(matrix, self.lower, self.upper)


def solve_economics(problem: EconomicProblem, *, risk: str = "expected", time_limit: float = 30.0,
                    allow_transfers: bool = True, policy: str = "OPEN_LOOP",
                    fixed_transfers: Mapping[tuple[str, str, int], float] | None = None,
                    transfer_upper_bounds: Mapping[tuple[str, str, int], float] | None = None,
                    fixed_prices: Mapping[tuple[str, int], str] | None = None,
                    fixed_sales: Mapping[tuple[str, str, str], float] | None = None) -> dict[str, Any]:
    """Solve a candidate plan. Missing evidence returns INPUT_REQUIRED, never a recommendation.

    Reported value is prospective net cash value, NOT observed savings or absolute
    seller loss. Common sunk acquisition cost is excluded. No shortage penalty is
    invented and no markdown loss is added after reduced revenue.
    """
    if (policy != "OPEN_LOOP" or fixed_transfers is not None or transfer_upper_bounds is not None
            or fixed_prices is not None or fixed_sales is not None or problem.confirmed_flows or problem.committed_transfers
            or problem.handling_per_capacity or problem.processing_capacity or problem.holding_by_period
            or problem.shortage_penalties):
        from services.candidate_economics_extended import solve_extended
        return solve_extended(problem, risk=risk, time_limit=time_limit, allow_transfers=allow_transfers,
                              policy=policy, fixed_transfers=fixed_transfers,
                              transfer_upper_bounds=transfer_upper_bounds, fixed_prices=fixed_prices,
                              fixed_sales=fixed_sales)
    start = time.perf_counter()
    issues = validate_problem(problem)
    if risk not in {"expected", "worst_case"}:
        issues.append({"input": "risk", "classification": "UNSUPPORTED_RISK"})
    base = {"version": VERSION, "status": "INPUT_REQUIRED", "mode": "NEW_CANDIDATE",
            "evidence_class": problem.evidence_class, "production_approved": False,
            "problem_signature": hashlib.sha256(json.dumps(_canonical(problem), sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(), "risk": risk,
            "input_issues": issues, "expected_net_value": None, "actual_savings": None,
            "absolute_loss": None, "accounting_scope": problem.accounting_scope,
            "limitations": ["Finite supplied price-response levels; no causal elasticity claim",
                "Open-loop transfer and price commitment; recourse only for sales and inventories",
                "No external revenue on internal transfers; sunk cost excluded",
                "No replenishment, credit, taxes or unobserved shortage penalty",
                "Economic value is modelled, not a realised outcome"],
            "cost_classification": {"sunk_purchase": "CANCELS_COMMON", "internal_transfer_revenue": "NOT_APPLICABLE",
                "extra_shortage_penalty": "NOT_APPLICABLE", "absolute_loss": "NOT_CALCULATED"}}
    if issues:
        return base
    p = problem
    lots = sorted(p.lots, key=lambda x: x.lot_id)
    routes = sorted(p.routes, key=lambda x: x.route_id)
    markets = sorted(p.markets, key=lambda x: x.market_id)
    scenarios = sorted(p.scenarios, key=lambda x: x.scenario_id)
    locations = sorted(p.storage_capacity)
    model = _Model()
    cash: dict[str, dict[int, float]] = {s.scenario_id: {} for s in scenarios}
    moves: dict[tuple[str, str, int], int] = {}
    vehicles: dict[tuple[str, int], int] = {}
    prices: dict[tuple[str, str], int] = {}
    inventory: dict[tuple[str, str, str, int], int] = {}
    disposal: dict[tuple[str, str, str, int], int] = {}
    sales: dict[tuple[str, str, str, str], int] = {}
    money = lambda x: 0.0 if x.treatment == "NOT_APPLICABLE" else float(x.field.value)
    def add(row: dict[int, float], index: int, value: float) -> None:
        row[index] = row.get(index, 0.0) + value
    def all_cash(index: int, value: float) -> None:
        for row in cash.values(): add(row, index, value)
    for route in routes:
        for t in sorted(route.departure_periods):
            if t + route.transit_periods >= p.horizon or not allow_transfers:
                continue
            n = model.variable(("vehicle", route.route_id, t), route.max_vehicles, True)
            vehicles[route.route_id, t] = n
            all_cash(n, -money(route.fixed_per_vehicle))
            cap = {n: -route.vehicle_capacity}
            for lot in lots:
                if lot.state not in route.allowed_states or t + route.transit_periods > lot.expiry_period:
                    continue
                q = model.variable(("move", lot.lot_id, route.route_id, t), lot.stock / lot.quantum, True)
                moves[lot.lot_id, route.route_id, t] = q
                add(cap, q, lot.quantum * lot.capacity_per_unit)
                all_cash(q, -lot.quantum * lot.capacity_per_unit * money(route.variable_per_capacity))
            model.row(cap, upper=0)
    for lot in lots:
        if lot.approved_departure_qty is not None:
            outgoing = {q: lot.quantum for (lid, rid, _), q in moves.items()
                        if lid == lot.lot_id and next(r for r in routes if r.route_id == rid).source == lot.source}
            model.row(outgoing, upper=lot.approved_departure_qty)
    for (loc, product, state), limit in sorted(p.receipt_limits.items()):
        incoming = {q: next(l for l in lots if l.lot_id == lid).quantum
                    for (lid, rid, _), q in moves.items()
                    if next(r for r in routes if r.route_id == rid).target == loc
                    and any(l.lot_id == lid and l.product_id == product and l.state == state for l in lots)}
        model.row(incoming, upper=limit)
    for market in markets:
        price_row = {}
        for option in sorted(market.options, key=lambda x: x.level):
            y = model.variable(("price", market.market_id, option.level), 1, True)
            prices[market.market_id, option.level] = y
            price_row[y] = 1
            for scenario in scenarios:
                demand_row = {y: -option.demand[scenario.scenario_id]}
                for lot in lots:
                    if (lot.product_id, lot.state, lot.unit) != (market.product_id, market.state, market.unit):
                        continue
                    if lot.expiry_period < market.period: continue
                    key = (scenario.scenario_id, lot.lot_id, market.market_id, option.level)
                    z = model.variable(("sale", *key), lot.stock / lot.quantum, True)
                    sales[key] = z
                    add(demand_row, z, lot.quantum)
                    add(cash[scenario.scenario_id], z, lot.quantum * money(option.unit_price))
                model.row(demand_row, upper=0)
        model.row(price_row, 1, 1)
    for scenario in scenarios:
        sid = scenario.scenario_id
        for lot in lots:
            for loc in locations:
                for t in range(p.horizon):
                    key = (sid, lot.lot_id, loc, t)
                    inv = model.variable(("inventory", *key), 0 if t >= lot.expiry_period else lot.stock / lot.quantum, True)
                    waste = model.variable(("disposal", *key), lot.stock / lot.quantum if t == lot.expiry_period else 0, True)
                    inventory[key], disposal[key] = inv, waste
                    add(cash[sid], inv, -lot.quantum * money(lot.holding_per_unit_period))
                    add(cash[sid], waste, -lot.quantum * money(lot.disposal_per_unit))
                    if t == p.horizon - 1:
                        add(cash[sid], inv, lot.quantum * money(lot.terminal_per_unit))
        for lot in lots:
            for loc in locations:
                for t in range(p.horizon):
                    key = (sid, lot.lot_id, loc, t)
                    balance = {inventory[key]: 1, disposal[key]: 1}
                    if t > 0: add(balance, inventory[sid, lot.lot_id, loc, t - 1], -1)
                    for route in routes:
                        q = moves.get((lot.lot_id, route.route_id, t))
                        if q is not None and route.source == loc: add(balance, q, 1)
                        incoming = moves.get((lot.lot_id, route.route_id, t - route.transit_periods))
                        if incoming is not None and route.target == loc: add(balance, incoming, -1)
                    for market in markets:
                        if market.location != loc or market.period != t: continue
                        for option in market.options:
                            z = sales.get((sid, lot.lot_id, market.market_id, option.level))
                            if z is not None: add(balance, z, 1)
                    initial = lot.stock / lot.quantum if t == 0 and lot.source == loc else 0
                    model.row(balance, initial, initial)
        # Beginning-of-period capacity includes arrivals before departure/sale.
        for loc in locations:
            for t in range(p.horizon):
                capacity_row: dict[int, float] = {}
                initial_capacity = 0.0
                for lot in lots:
                    size = lot.quantum * lot.capacity_per_unit
                    if t == 0 and lot.source == loc: initial_capacity += lot.stock * lot.capacity_per_unit
                    if t > 0: add(capacity_row, inventory[sid, lot.lot_id, loc, t - 1], size)
                    for route in routes:
                        q = moves.get((lot.lot_id, route.route_id, t - route.transit_periods))
                        if q is not None and route.target == loc: add(capacity_row, q, size)
                model.row(capacity_row, upper=p.storage_capacity[loc] - initial_capacity)
    z_worst = None
    if risk == "worst_case":
        # A finite lower bound permits negative prospective net value.
        z_pos = model.variable(("worst_positive",)); z_neg = model.variable(("worst_negative",))
        z_worst = (z_pos, z_neg)
        for terms in cash.values():
            row = dict(terms); add(row, z_pos, -1); add(row, z_neg, 1)
            model.row(row, lower=0)
    objective = np.zeros(len(model.names))
    if z_worst:
        objective[z_worst[0]], objective[z_worst[1]] = -1, 1
    else:
        for scenario in scenarios:
            for i, v in cash[scenario.scenario_id].items(): objective[i] -= scenario.probability * v
    result = milp(objective, integrality=np.array(model.integer), bounds=Bounds(0, model.ub),
                  constraints=model.constraint(), options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    base.update(status="OPTIMAL" if result.status == 0 else "FEASIBLE_LIMIT" if result.x is not None else "NO_SOLUTION",
                solver_status=int(result.status), solver_message=str(result.message),
                optimality_proven=result.status == 0, solver_gap=getattr(result, "mip_gap", None),
                runtime_seconds=time.perf_counter() - start, variables=len(model.names), constraints=len(model.rows),
                objective="EXPECTED_NET_CASH_MAX" if risk == "expected" else "WORST_SCENARIO_NET_CASH_MAX",
                event_order="arrival -> departure -> sale -> expiry disposal -> holding",
                nonanticipativity="All transfers and price levels are common across scenarios",
                price_policy="Exactly one price level per product/state/location/period; mixed transfer + remaining sale allowed")
    if result.x is None: return base
    x = result.x
    profits = {sid: float(sum(x[i] * v for i, v in row.items())) for sid, row in cash.items()}
    base.update(expected_net_value=sum(s.probability * profits[s.scenario_id] for s in scenarios),
                worst_net_value=min(profits.values()), scenario_net_values=profits,
                objective_value=float(-result.fun),
                bound_net_value=float(-result.mip_dual_bound) if getattr(result, "mip_dual_bound", None) is not None else None)
    lot_map = {l.lot_id: l for l in lots}
    base["transfers"] = [{"lot_id": lid, "route_id": rid, "departure_period": t,
                           "quantity": float(x[i] * lot_map[lid].quantum)}
                          for (lid, rid, t), i in moves.items() if x[i] > 1e-6]
    base["vehicles"] = [{"route_id": rid, "period": t, "count": int(round(x[i]))}
                        for (rid, t), i in vehicles.items() if x[i] > 1e-6]
    base["price_levels"] = [{"market_id": mid, "level": level} for (mid, level), i in prices.items() if x[i] > 0.5]
    base["sales"] = [{"scenario": sid, "lot_id": lid, "market_id": mid, "level": level,
                      "quantity": float(x[i] * lot_map[lid].quantum)}
                     for (sid, lid, mid, level), i in sales.items() if x[i] > 1e-6]
    base["inventory"] = [{"scenario": sid, "lot_id": lid, "location": loc, "period": t,
                          "quantity": float(x[i] * lot_map[lid].quantum),
                          "disposal": float(x[disposal[sid, lid, loc, t]] * lot_map[lid].quantum)}
                         for (sid, lid, loc, t), i in inventory.items()]
    matrix = model.constraint().A
    activity = matrix @ x
    violation = max(float(np.max(np.maximum(np.array(model.lower) - activity, 0))),
                    float(np.max(np.maximum(activity - np.array(model.upper), 0))))
    base["constraint_max_violation"] = violation
    base["constraint_violations"] = int(np.count_nonzero((activity < np.array(model.lower) - 1e-6)
                                                        | (activity > np.array(model.upper) + 1e-6)))
    base["transport_cost"] = sum(x[i] * money(next(r for r in routes if r.route_id == rid).fixed_per_vehicle)
                                 for (rid, _), i in vehicles.items()) + sum(
        x[i] * lot_map[lid].quantum * lot_map[lid].capacity_per_unit
        * money(next(r for r in routes if r.route_id == rid).variable_per_capacity)
        for (lid, rid, _), i in moves.items())
    route_map = {r.route_id: r for r in routes}
    base["in_transit"] = [{"lot_id": lot.lot_id, "period": t,
        "quantity": sum(float(x[i] * lot.quantum) for (lid, rid, departed), i in moves.items()
                        if lid == lot.lot_id and departed <= t < departed + route_map[rid].transit_periods)}
        for lot in lots for t in range(p.horizon)]
    base["market_outcomes"] = []
    for scenario in scenarios:
        for market in markets:
            option = next(o for o in market.options if x[prices[market.market_id, o.level]] > .5)
            sold = sum(v["quantity"] for v in base["sales"]
                       if v["scenario"] == scenario.scenario_id and v["market_id"] == market.market_id)
            demand = option.demand[scenario.scenario_id]
            base["market_outcomes"].append({"scenario": scenario.scenario_id,
                "market_id": market.market_id, "period": market.period, "level": option.level,
                "demand": demand, "sold": sold, "unmet": max(0.0, demand - sold)})
    base["scenario_cash_components"] = {}
    for scenario in scenarios:
        sid = scenario.scenario_id
        revenue = sum(x[i] * lot_map[lid].quantum * money(next(o for o in next(
            m for m in markets if m.market_id == mid).options if o.level == level).unit_price)
            for (s, lid, mid, level), i in sales.items() if s == sid)
        holding = sum(x[i] * lot_map[lid].quantum * money(lot_map[lid].holding_per_unit_period)
                      for (s, lid, _, _), i in inventory.items() if s == sid)
        waste = sum(x[i] * lot_map[lid].quantum * money(lot_map[lid].disposal_per_unit)
                    for (s, lid, _, _), i in disposal.items() if s == sid)
        terminal = sum(x[i] * lot_map[lid].quantum * money(lot_map[lid].terminal_per_unit)
                       for (s, lid, _, t), i in inventory.items() if s == sid and t == p.horizon - 1)
        base["scenario_cash_components"][sid] = {"external_sales_revenue": float(revenue),
            "transport_cost": float(base["transport_cost"]), "holding_cost": float(holding),
            "disposal_cost": float(waste), "terminal_value": float(terminal),
            "net_cash_value": float(revenue - base["transport_cost"] - holding - waste + terminal)}
    return base


def bind_snapshot_economics(snapshot: Any, problem: EconomicProblem) -> tuple[EconomicProblem | None, list[dict[str, Any]]]:
    """Strict adapter to the shared OperationalSnapshot, without changing it.

    Economic markets/costs must be supplied explicitly. The canonical schema has
    no price elasticity, so this adapter never fabricates demand responses. Pending
    totals must reconcile to dated immutable events. Unknown totals or a missing
    schedule still block the model; a date is never inferred from a total.
    """
    problems: list[dict[str, Any]] = []
    if snapshot.horizon_days != problem.horizon:
        problems.append({"input": "horizon", "classification": "SNAPSHOT_HORIZON_MISMATCH"})
    if snapshot.report.get("invalid_needs"):
        problems.append({"input": "needs", "classification": "SNAPSHOT_NEED_UNCONFIRMED"})
    approved_routes = snapshot.context.get("approved_route_ids")
    if approved_routes is None or any(r.route_id not in approved_routes for r in problem.routes):
        problems.append({"input": "routes", "classification": "ROUTE_APPROVAL_REQUIRED"})
    scheduled_lots = {e.lot_id for e in problem.confirmed_flows} | {e.lot_id for e in problem.committed_transfers}
    expected_ids = {(p.lot, p.product, p.location, p.state, p.unit) for p in snapshot.positions
                    if p.on_hand is None or p.on_hand > 0 or p.lot in scheduled_lots}
    supplied_ids = {(l.lot_id, l.product_id, l.source, l.state, l.unit) for l in problem.lots}
    if expected_ids != supplied_ids:
        problems.append({"input": "lots", "classification": "INITIAL_STOCK_SCOPE_MISMATCH"})
    lots = []
    for lot in problem.lots:
        matches = [p for p in snapshot.positions if (p.lot, p.product, p.location, p.state, p.unit)
                   == (lot.lot_id, lot.product_id, lot.source, lot.state, lot.unit)]
        if len(matches) != 1:
            problems.append({"input": lot.lot_id, "classification": "SNAPSHOT_IDENTITY_NOT_UNIQUE"})
            continue
        row = matches[0]
        if row.issues or row.on_hand != lot.stock or row.approved_transfer is None:
            problems.append({"input": lot.lot_id, "classification": "SNAPSHOT_STOCK_OR_APPROVAL_INVALID",
                             "details": list(row.issues)})
        if any(v is None for v in (row.reserved, row.inbound_confirmed, row.outbound_confirmed)):
            problems.append({"input": lot.lot_id, "classification": "PENDING_FLOW_TOTAL_UNCONFIRMED"})
        else:
            route_by_id = {r.route_id: r for r in problem.routes}
            events = [e for e in problem.confirmed_flows if e.lot_id == lot.lot_id and e.location == row.location]
            reserved = sum(e.quantity for e in events if e.kind == "RESERVED_OUTBOUND")
            inbound = sum(e.quantity for e in events if e.kind == "CONFIRMED_INBOUND")
            outbound = sum(e.quantity for e in events if e.kind == "CONFIRMED_OUTBOUND")
            for e in problem.committed_transfers:
                route = route_by_id.get(e.route_id)
                if e.lot_id != lot.lot_id or route is None: continue
                if e.departure_period >= 0 and route.source == row.location: outbound += e.quantity
                if route.target == row.location: inbound += e.quantity
            if any(abs(actual - scheduled) > 1e-7 for actual, scheduled in (
                    (row.reserved, reserved), (row.inbound_confirmed, inbound), (row.outbound_confirmed, outbound))):
                problems.append({"input": lot.lot_id, "classification": "PENDING_FLOW_SCHEDULE_MISMATCH",
                    "reported": {"reserved": row.reserved, "inbound": row.inbound_confirmed, "outbound": row.outbound_confirmed},
                    "scheduled": {"reserved": reserved, "inbound": inbound, "outbound": outbound}})
        if row.expiry_date is None:
            problems.append({"input": lot.lot_id, "classification": "SNAPSHOT_EXPIRY_UNCONFIRMED"})
        else:
            from datetime import date
            try:
                expiry = (date.fromisoformat(row.expiry_date) - date.fromisoformat(snapshot.as_of)).days
                if expiry != lot.expiry_period:
                    problems.append({"input": lot.lot_id, "classification": "SNAPSHOT_EXPIRY_MISMATCH"})
            except (TypeError, ValueError):
                problems.append({"input": lot.lot_id, "classification": "SNAPSHOT_DATE_INVALID"})
        lots.append(replace(lot, approved_departure_qty=row.approved_transfer))
    receipt = dict(problem.receipt_limits)
    for need in snapshot.target_needs:
        if (not need.accepted or need.as_of_date != snapshot.as_of or
                need.planning_horizon_days != problem.horizon or
                need.dataset_id != snapshot.dataset or need.network_id != snapshot.network or
                not need.valid_until or need.valid_until < snapshot.as_of):
            problems.append({"input": need.product_id, "classification": "NEED_EVIDENCE_OR_PERIOD_UNCONFIRMED"})
        states = {l.state for l in problem.lots if l.product_id == need.product_id}
        if len(states) != 1:
            problems.append({"input": need.product_id, "classification": "NEED_STATE_AMBIGUOUS"})
            continue
        state = next(iter(states))
        key = (need.location_id, need.product_id, state)
        quantity = need.target_need_value
        units = {l.unit for l in problem.lots if l.product_id == need.product_id}
        if quantity is None or need.quantity_unit not in units:
            problems.append({"input": need.product_id, "classification": "NEED_VALUE_OR_UNIT_UNCONFIRMED"})
        else:
            receipt[key] = min(receipt.get(key, float("inf")), float(quantity))
    if problems:
        return None, problems
    return replace(problem, lots=tuple(lots), receipt_limits=receipt,
        lineage={**problem.lineage, "snapshot": snapshot.base_problem_signature}), []


def solve_snapshot_economics(snapshot: Any, problem: EconomicProblem, **options: Any) -> dict[str, Any]:
    """Keep the original validation and solve contract, sharing only the pure bind step."""
    bound, problems = bind_snapshot_economics(snapshot, problem)
    if problems:
        return {"version": VERSION, "status": "INPUT_REQUIRED", "mode": "NEW_CANDIDATE",
                "input_issues": problems, "expected_net_value": None, "actual_savings": None,
                "production_approved": False, "snapshot_signature": snapshot.base_problem_signature}
    result = solve_economics(bound, **options)
    result["snapshot_signature"] = snapshot.base_problem_signature
    return result
