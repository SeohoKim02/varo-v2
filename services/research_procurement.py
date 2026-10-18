"""Bounded SCENARIO replenishment research using the unchanged economic solver.

One product, two stores, three periods. A single supplier order is placed at t=0
and delivered to T at a declared period. Existing stock can depart S at t=0,
arrive T at t=1, and sell at supplied price-response levels. All demand values
are known scenario inputs, not realized future observations or causal estimates.
No operational order, database, file, or production decision is written here.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from services.candidate_economics import (
    ConfirmedFlow, DemandMarket, EconomicProblem, InventoryLot, Money,
    PriceOption, Route, Scenario, solve_economics,
)
from services.seller_loss_engine import known


@dataclass(frozen=True)
class ProcurementScenario:
    source_stock: int = 8
    max_order: int = 6
    minimum_order: int = 1
    arrival_period: int = 1
    purchase_unit_cost: float = 3
    purchase_fixed_cost: float = 0
    purchase_budget: float = 24
    transfer_capacity: int = 4
    transfer_fixed_cost: float = 2
    target_storage: int = 8
    source_normal_demand: int = 2
    source_discount_demand: int = 6
    target_early_demand: int = 4
    target_late_demand: int = 2


def _validate(s):
    if not isinstance(s, ProcurementScenario):
        raise ValueError("A declared procurement scenario is required")
    for k in ("source_stock", "max_order", "minimum_order", "arrival_period",
              "transfer_capacity", "target_storage", "source_normal_demand",
              "source_discount_demand", "target_early_demand", "target_late_demand"):
        v = getattr(s, k)
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 12:
            raise ValueError(f"{k} requires an integer from 0 to 12")
    if not 1 <= s.minimum_order <= max(1, s.max_order) or s.arrival_period > 2:
        raise ValueError("Invalid minimum order or arrival period")
    if s.transfer_capacity < 1 or s.target_storage < 1 or s.max_order > 8:
        raise ValueError("Research capacity/order bound exceeded")
    for k in ("purchase_unit_cost", "purchase_fixed_cost", "purchase_budget", "transfer_fixed_cost"):
        v = getattr(s, k)
        if isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or v < 0:
            raise ValueError(f"{k} requires an explicit finite nonnegative value")


def purchase_cost(s, quantity):
    return 0.0 if quantity == 0 else s.purchase_unit_cost * quantity + s.purchase_fixed_cost


def order_quantities(s):
    _validate(s)
    return tuple(q for q in range(s.max_order + 1)
                 if (q == 0 or q >= s.minimum_order)
                 and purchase_cost(s, q) <= s.purchase_budget + 1e-9)


def _money(value, unit="EA"):
    return Money(known(value, "SCENARIO_INPUT", "bounded procurement experiment",
                       unit=unit, currency="KRW", dataset="PROCUREMENT_SCENARIO"), "KNOWN")


def build_problem(s, quantity):
    if quantity not in order_quantities(s):
        raise ValueError("Order violates quantity, minimum lot, or purchase budget")
    old = InventoryLot("existing", "P", "S", "CHILLED", "EA", s.source_stock, 2, 1,
                       _money(0), _money(0), _money(0))
    new = InventoryLot("purchased", "P", "T", "CHILLED", "EA", 0, 2, 1,
                       _money(0), _money(0), _money(0))
    route = Route("S_T", "S", "T", 1, s.transfer_capacity, 1,
                  _money(s.transfer_fixed_cost, None), _money(0, "KG"), ("CHILLED",))
    def market(name, location, period, normal, discount, discount_price):
        return DemandMarket(name, location, "P", "CHILLED", "EA", period, (
            PriceOption("normal", _money(10), {"base": normal}),
            PriceOption("discount", _money(discount_price), {"base": discount})))
    markets = (market("source", "S", 0, s.source_normal_demand, s.source_discount_demand, 6),
               market("early", "T", 1, s.target_early_demand, s.target_early_demand + 2, 7),
               market("late", "T", 2, s.target_late_demand, s.target_late_demand + 2, 7))
    # The fixed scenario receipt is conditional on this enumerated order. It is
    # NOT an actually confirmed supplier delivery and is never saved as one.
    return EconomicProblem("bounded_procurement", 3, "KRW", (old, new), (route,),
        markets, (Scenario("base", 1),), "KG", {"S": max(1, s.source_stock), "T": s.target_storage},
        evidence_class="SCENARIO", lineage={"receipt": "hypothetical order at t=0",
            "demand": "supplied deterministic price-response scenario", "capacity": "1 EA = 1 KG assumed"},
        confirmed_flows=(ConfirmedFlow("hypothetical_supplier_arrival", "purchased", "T",
                                      s.arrival_period, quantity, "CONFIRMED_INBOUND"),))


def compare_procurement(s):
    """Enumerate a bounded order set, reusing existing MILP for each order.

Sequential baseline fixes the best order with transfers excluded, then optimizes
transfer and prices with that order. Joint comparison can choose any same order.
Thus weak dominance follows from feasible-set inclusion, not solver novelty.
All feasible candidates must be proved optimal; limits/errors fail closed.
"""
    candidates = []
    for quantity in order_quantities(s):
        problem = build_problem(s, quantity)
        values = {}
        for label, allowed in (("local", False), ("joint", True)):
            r = solve_economics(problem, allow_transfers=allowed)
            if r["status"] == "NO_SOLUTION" and r.get("solver_status") == 2:
                values[label] = None
                continue
            if r["status"] != "OPTIMAL" or r.get("constraint_violations") != 0:
                raise RuntimeError(f"Procurement candidate not verified: {r['status']}")
            parts = r["accounting_components"]["base"]
            values[label] = {"order_quantity": quantity, "purchase_cost": purchase_cost(s, quantity),
                "net_value": r["expected_net_value"] - purchase_cost(s, quantity),
                "cash_before_purchase": r["expected_net_value"],
                "accounting": parts, "transfers": r["transfers"], "sales": r["sales"],
                "inventory": r["inventory"], "prices": r["price_levels"],
                "constraint_violations": r["constraint_violations"]}
        candidates.append({"order_quantity": quantity, **values})
    def best(label):
        valid = [row[label] for row in candidates if row[label] is not None]
        if not valid:
            raise RuntimeError("No feasible candidate in the declared research scenario")
        highest = max(row["net_value"] for row in valid)
        return min((row for row in valid if highest - row["net_value"] <= 1e-7), key=lambda row: row["order_quantity"])
    local, joint = best("local"), best("joint")
    sequential = next(row["joint"] for row in candidates if row["order_quantity"] == local["order_quantity"])
    if sequential is None or joint["net_value"] + 1e-6 < sequential["net_value"]:
        raise ArithmeticError("Joint feasible-set inclusion was violated")
    return {"evidence_class": "SCENARIO", "production_approved": False, "actual_savings": None,
            "joint": joint, "sequential": sequential, "local_order_basis": local,
            "candidates": candidates, "difference": joint["net_value"] - sequential["net_value"],
            "scope": "Single product, one t=0 order, 3 periods, deterministic supplied demands",
            "accounting": "Existing acquisition is sunk; new purchase is charged exactly once"}
