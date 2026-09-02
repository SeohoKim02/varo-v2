"""Whole-plan allocation tests: shared stock, shared need, routes, fallback, UI contract."""
from __future__ import annotations

import copy
import time
import unittest

import pandas as pd

from components.tables import build_recommendation_rows
from services.execution_plan import (
    PLAN_STATUS_EMPTY,
    PLAN_STATUS_UNAVAILABLE,
    build_execution_plan,
    compare_execution_plans,
    planned_recommendations,
    validate_execution_plan,
)
from services.home_state import READY, build_home_state


def _candidate(
    route_id: str,
    source: str,
    target: str,
    product: str = "P1",
    qty: int = 40,
    saving: float = 8_000,
    cost: float = 1_000,
    vhs: float = 80,
    route_type: str = "DIRECT",
    dc_id: str | None = None,
    vhs_rank: int = 1,
    greedy_rank: int | None = None,
) -> dict:
    return {
        "route_id": route_id,
        "source_id": source,
        "source_name": source,
        "target_id": target,
        "target_name": target,
        "product_id": product,
        "product_name": product,
        "route_type": route_type,
        "dc_id": dc_id,
        "dc_name": dc_id,
        "recommended_qty": qty,
        "expected_saving": saving,
        "estimated_cost": cost,
        "net_benefit": saving - cost,
        "vhs_score": vhs,
        "vhs_rank": vhs_rank,
        "varo_final_rank": vhs_rank,
        "greedy_rank": greedy_rank or vhs_rank,
        "robustness_status": "안정",
        "confidence_score": 80,
        "confidence_level": "높음",
        "pareto_status": "비지배",
        "feasibility_status": "추천 가능",
    }


def _workbook(
    inventory: list[dict],
    routes: list[dict],
    products: tuple[str, ...] = ("P1",),
) -> dict:
    node_ids = {
        str(row["store_id"]) for row in inventory
    } | {
        str(value)
        for row in routes
        for value in (row.get("source_id"), row.get("target_id"), row.get("dc_id"))
        if value
    }
    stores = pd.DataFrame([
        {"store_id": node_id, "node_id": node_id, "node_type": "DC" if node_id.startswith("DC") else "STORE"}
        for node_id in sorted(node_ids)
    ])
    return {
        "stores": stores,
        "products": pd.DataFrame([{"product_id": product} for product in products]),
        "inventory": pd.DataFrame(inventory),
        "routes": pd.DataFrame(routes),
    }


def _base_data(source_stock: int = 100, safety: int = 40, target_gap: int = 100) -> dict:
    return _workbook(
        [
            {"store_id": "S", "product_id": "P1", "stock_qty": source_stock, "safety_stock": safety, "demand_qty": 0},
            {"store_id": "T1", "product_id": "P1", "stock_qty": 0, "safety_stock": 0, "target_stock": target_gap},
            {"store_id": "T2", "product_id": "P1", "stock_qty": 0, "safety_stock": 0, "target_stock": target_gap},
            {"store_id": "T3", "product_id": "P1", "stock_qty": 0, "safety_stock": 0, "target_stock": target_gap},
        ],
        [
            {"source_id": "S", "target_id": "T1", "route_type": "DIRECT"},
            {"source_id": "S", "target_id": "T2", "route_type": "DIRECT"},
            {"source_id": "S", "target_id": "T3", "route_type": "DIRECT"},
        ],
    )


def _decision(plan: dict) -> tuple:
    return (
        plan.get("plan_id"),
        tuple((row.get("candidate_id"), row.get("planned_qty")) for row in plan.get("items") or []),
        tuple((row.get("route_id"), row.get("reason_code")) for row in plan.get("unselected_candidates") or []),
    )


class SharedSourceAndTargetTests(unittest.TestCase):
    def test_two_and_three_candidates_never_exceed_shared_source(self):
        data = _base_data(source_stock=100, safety=40)
        recs = [
            _candidate("R1", "S", "T1", qty=40, saving=10_000, vhs_rank=1),
            _candidate("R2", "S", "T2", qty=35, saving=7_000, vhs_rank=2),
            _candidate("R3", "S", "T3", qty=20, saving=3_000, vhs_rank=3),
        ]
        plan = build_execution_plan(recs, data, data_signature="shared-source")
        self.assertTrue(plan["validation"]["valid"])
        self.assertEqual(sum(row["planned_qty"] for row in plan["items"]), 60)
        self.assertTrue(any(row["quantity_adjusted"] for row in plan["items"]))

    def test_exact_limit_passes_and_one_extra_is_impossible(self):
        data = _base_data(source_stock=61, safety=1)
        recs = [_candidate("R1", "S", "T1", qty=40), _candidate("R2", "S", "T2", qty=21)]
        plan = build_execution_plan(recs, data, data_signature="exact-source")
        self.assertEqual(plan["total_transfer_qty"], 60)
        self.assertLessEqual(plan["total_transfer_qty"], 60)

    def test_shared_destination_never_exceeds_target_gap(self):
        inventory = [
            {"store_id": "S1", "product_id": "P1", "stock_qty": 50, "safety_stock": 0},
            {"store_id": "S2", "product_id": "P1", "stock_qty": 50, "safety_stock": 0},
            {"store_id": "T", "product_id": "P1", "stock_qty": 10, "target_stock": 40, "safety_stock": 0},
        ]
        routes = [{"source_id": source, "target_id": "T", "route_type": "DIRECT"} for source in ("S1", "S2")]
        data = _workbook(inventory, routes)
        recs = [_candidate("R1", "S1", "T", qty=20), _candidate("R2", "S2", "T", qty=20, vhs_rank=2)]
        plan = build_execution_plan(recs, data, data_signature="shared-target")
        self.assertEqual(plan["total_transfer_qty"], 30)
        self.assertEqual(plan["validation"]["destination_overfill_violations"], 0)

    def test_target_stock_exactly_filled_and_never_overfilled(self):
        data = _base_data(source_stock=200, safety=0, target_gap=30)
        plan = build_execution_plan([_candidate("R1", "S", "T1", qty=31)], data)
        self.assertEqual(plan["items"][0]["planned_qty"], 30)
        self.assertTrue(plan["validation"]["valid"])


class ProductRouteAndQuantityTests(unittest.TestCase):
    def test_products_use_independent_inventory_constraints(self):
        inventory = []
        for product, stock, floor, gap in (("P1", 40, 10, 30), ("P2", 70, 20, 50)):
            inventory.extend([
                {"store_id": "S", "product_id": product, "stock_qty": stock, "safety_stock": floor},
                {"store_id": "T", "product_id": product, "stock_qty": 0, "target_stock": gap, "safety_stock": 0},
            ])
        data = _workbook(inventory, [{"source_id": "S", "target_id": "T", "route_type": "DIRECT"}], ("P1", "P2"))
        recs = [_candidate("R1", "S", "T", "P1", qty=40), _candidate("R2", "S", "T", "P2", qty=60, vhs_rank=2)]
        plan = build_execution_plan(recs, data)
        by_product = {row["product_id"]: row["planned_qty"] for row in plan["items"]}
        self.assertEqual(by_product, {"P1": 30, "P2": 50})

    def _route_data(self) -> dict:
        return _workbook(
            [
                {"store_id": "S", "product_id": "P1", "stock_qty": 100, "safety_stock": 0},
                {"store_id": "T", "product_id": "P1", "stock_qty": 0, "target_stock": 50, "safety_stock": 0},
            ],
            [
                {"source_id": "S", "target_id": "T", "route_type": "DIRECT"},
                {"source_id": "S", "target_id": "T", "route_type": "VIA_DC", "dc_id": "DC01"},
                {"source_id": "S", "target_id": "T", "route_type": "VIA_DC", "dc_id": "DC02"},
            ],
        )

    def test_direct_and_via_dc_use_real_value_and_never_duplicate(self):
        data = self._route_data()
        direct = _candidate("RD", "S", "T", qty=40, saving=9_000, cost=1_000, route_type="DIRECT")
        via = _candidate("RV", "S", "T", qty=40, saving=8_000, cost=2_000, route_type="VIA_DC", dc_id="DC01", vhs_rank=2)
        first = build_execution_plan([direct, via], data)
        self.assertEqual([row["route_id"] for row in first["items"]], ["RD"])
        via["expected_saving"], via["estimated_cost"], via["net_benefit"] = 12_000, 1_000, 11_000
        second = build_execution_plan([direct, via], data)
        self.assertEqual([row["route_id"] for row in second["items"]], ["RV"])
        self.assertEqual(len(second["items"]), 1)

    def test_dc01_and_dc02_are_chosen_by_value_not_id_order(self):
        data = self._route_data()
        dc01 = _candidate("R1", "S", "T", route_type="VIA_DC", dc_id="DC01", saving=8_000, cost=2_000)
        dc02 = _candidate("R2", "S", "T", route_type="VIA_DC", dc_id="DC02", saving=10_000, cost=1_000, vhs_rank=2)
        plan = build_execution_plan([dc01, dc02], data)
        self.assertEqual(plan["items"][0]["dc_id"], "DC02")

    def test_full_recommended_quantity_and_zero_quantity_removal(self):
        data = _base_data()
        recs = [_candidate("R1", "S", "T1", qty=40), _candidate("R0", "S", "T2", qty=0, vhs_rank=2)]
        plan = build_execution_plan(recs, data)
        self.assertEqual(plan["items"][0]["planned_qty"], 40)
        self.assertEqual({row["reason_code"] for row in plan["unselected_candidates"]}, {"infeasible"})


class SafetyBenefitAndValidationTests(unittest.TestCase):
    def test_explicit_and_estimated_safety_floors_are_respected(self):
        explicit = _base_data(source_stock=100, safety=70)
        explicit_plan = build_execution_plan([_candidate("R1", "S", "T1", qty=50)], explicit)
        self.assertEqual(explicit_plan["total_transfer_qty"], 30)
        estimated = _base_data(source_stock=100, safety=0)
        estimated["inventory"] = estimated["inventory"].drop(columns=["safety_stock"])
        estimated["inventory"]["demand_std"] = 0.0
        estimated["inventory"].loc[estimated["inventory"]["store_id"] == "S", "demand_std"] = 20.0
        estimated_plan = build_execution_plan([_candidate("R1", "S", "T1", qty=80)], estimated)
        self.assertEqual(estimated_plan["total_transfer_qty"], 60)
        self.assertEqual(estimated_plan["validation"]["safety_stock_violations"], 0)

    def test_negative_is_not_selected_positive_total_is_exact(self):
        data = _base_data()
        positive = _candidate("RP", "S", "T1", qty=10, saving=1_000, cost=200)
        negative = _candidate("RN", "S", "T2", qty=10, saving=100, cost=200, vhs_rank=2)
        plan = build_execution_plan([positive, negative], data)
        self.assertEqual([row["route_id"] for row in plan["items"]], ["RP"])
        self.assertAlmostEqual(plan["total_net_benefit"], 800.0)
        self.assertIn("negative_net_benefit", {row["reason_code"] for row in plan["unselected_candidates"]})

    def test_validator_rejects_tampered_quantity_and_amounts(self):
        data = _base_data(source_stock=100, safety=40)
        recs = [_candidate("R1", "S", "T1", qty=40)]
        plan = build_execution_plan(recs, data, data_signature="tamper")
        bad = copy.deepcopy(plan)
        bad["items"][0]["planned_qty"] = 61
        bad["items"][0]["planned_net_benefit"] = float("nan")
        report = validate_execution_plan(bad, recs, data, data_signature="tamper")
        self.assertFalse(report["valid"])
        self.assertGreater(report["safety_stock_violations"], 0)


class DeterminismAndFallbackTests(unittest.TestCase):
    def test_same_input_reversed_order_and_tie_are_identical(self):
        data = _base_data(source_stock=50, safety=0)
        recs = [
            _candidate("R2", "S", "T2", qty=40, saving=8_000, vhs=80, vhs_rank=1),
            _candidate("R1", "S", "T1", qty=40, saving=8_000, vhs=80, vhs_rank=1),
        ]
        first = build_execution_plan(recs, data, data_signature="det")
        second = build_execution_plan(list(reversed(recs)), data, data_signature="det")
        third = build_execution_plan(recs, data, data_signature="det")
        self.assertEqual(_decision(first), _decision(second))
        self.assertEqual(_decision(first), _decision(third))

    def test_optimizer_failure_and_invalid_result_use_safe_fallback(self):
        data = _base_data(source_stock=60, safety=0)
        recs = [_candidate("R1", "S", "T1", qty=40), _candidate("R2", "S", "T2", qty=40, vhs_rank=2)]

        def fail(_candidates, _timeout):
            raise TimeoutError("forced")

        failed = build_execution_plan(recs, data, optimizer=fail)
        self.assertTrue(failed["fallback_used"])
        self.assertTrue(failed["validation"]["valid"])

        def invalid(candidates, _timeout):
            return [candidate.max_qty for candidate in candidates]

        infeasible = build_execution_plan(recs, data, optimizer=invalid)
        self.assertTrue(infeasible["fallback_used"])
        self.assertTrue(infeasible["validation"]["valid"])


class ComparisonAndUiContractTests(unittest.TestCase):
    def test_comparison_reports_independent_conflicts_and_safe_plans(self):
        data = _base_data(source_stock=60, safety=0)
        recs = [_candidate("R1", "S", "T1", qty=40), _candidate("R2", "S", "T2", qty=40, vhs_rank=2)]
        optimized = build_execution_plan(recs, data)
        greedy = build_execution_plan(recs, data, strategy="greedy")
        comparison = compare_execution_plans(optimized, greedy, recs, data)
        self.assertEqual(comparison["independent_candidates"]["safety_stock_violations"], 1)
        for key in ("constrained_greedy", "vhs_optimized_plan"):
            self.assertEqual(comparison[key]["safety_stock_violations"], 0)

    def test_home_and_action_page_share_first_plan_item_and_planned_quantity(self):
        data = _base_data(source_stock=60, safety=0)
        recs = [_candidate("R1", "S", "T1", qty=40), _candidate("R2", "S", "T2", qty=40, vhs_rank=2)]
        plan = build_execution_plan(recs, data, data_signature="ui")
        pipeline = {"status": "success", "execution_plan": plan, "confidence_status": {"status": "높음"}}
        state = {
            "varo_data": data,
            "varo_recommendations": recs,
            "varo_pipeline_result": pipeline,
            "analysis_result": pipeline,
            "selected_route_id": plan["items"][0]["route_id"],
        }
        home = build_home_state(state)
        actions = planned_recommendations(pipeline)
        self.assertEqual(home["state_code"], READY)
        self.assertEqual(home["top_recommendation"]["plan_id"], actions[0]["plan_id"])
        self.assertEqual(home["top_recommendation"]["planned_qty"], actions[0]["planned_qty"])
        rows = build_recommendation_rows(actions, include_route_id=False)
        self.assertEqual(rows[0]["수량"], f"{actions[0]['planned_qty']}개")
        for text in (plan["user_message"], str(rows)):
            for hidden in ("MILP", "solver", "objective function", "constraint matrix"):
                self.assertNotIn(hidden, text)


class StressTests(unittest.TestCase):
    def test_sixteen_required_scenarios_never_crash_violate_or_change(self):
        base = _base_data(source_stock=100, safety=40, target_gap=60)
        scenarios: list[tuple[str, list[dict], dict]] = []
        scenarios.append(("zero", [], base))
        scenarios.append(("one", [_candidate("R1", "S", "T1")], base))

        many_inventory = [{"store_id": "S", "product_id": "P1", "stock_qty": 4_000, "safety_stock": 1_000}]
        many_routes = []
        many_recs = []
        for index in range(300):
            target = f"T{index:03d}"
            many_inventory.append({"store_id": target, "product_id": "P1", "stock_qty": 0, "target_stock": 10, "safety_stock": 0})
            many_routes.append({"source_id": "S", "target_id": target, "route_type": "DIRECT"})
            many_recs.append(_candidate(f"R{index:03d}", "S", target, qty=10, vhs=80, vhs_rank=index + 1))
        scenarios.append(("three_hundred", many_recs, _workbook(many_inventory, many_routes)))
        scenarios.append(("shared_source", [_candidate("R1", "S", "T1", qty=50), _candidate("R2", "S", "T2", qty=50, vhs_rank=2)], base))

        target_data = _workbook(
            [
                {"store_id": "S1", "product_id": "P1", "stock_qty": 100, "safety_stock": 0},
                {"store_id": "S2", "product_id": "P1", "stock_qty": 100, "safety_stock": 0},
                {"store_id": "T", "product_id": "P1", "stock_qty": 0, "target_stock": 50, "safety_stock": 0},
            ],
            [{"source_id": "S1", "target_id": "T"}, {"source_id": "S2", "target_id": "T"}],
        )
        target_recs = [_candidate("R1", "S1", "T", qty=40), _candidate("R2", "S2", "T", qty=40, vhs_rank=2)]
        scenarios.append(("shared_target", target_recs, target_data))
        scenarios.append(("both_shared", target_recs, target_data))

        route_test = ProductRouteAndQuantityTests()
        route_data = route_test._route_data()
        scenarios.append(("direct_via", [
            _candidate("RD", "S", "T", route_type="DIRECT"),
            _candidate("RV", "S", "T", route_type="VIA_DC", dc_id="DC01", vhs_rank=2),
        ], route_data))
        scenarios.append(("multi_dc", [
            _candidate("R1", "S", "T", route_type="VIA_DC", dc_id="DC01"),
            _candidate("R2", "S", "T", route_type="VIA_DC", dc_id="DC02", vhs_rank=2),
        ], route_data))

        multi_inventory = [
            {"store_id": store, "product_id": product, "stock_qty": stock, "safety_stock": floor, "target_stock": target}
            for product in ("P1", "P2")
            for store, stock, floor, target in (("S", 100, 40, None), ("T", 0, 0, 60))
        ]
        multi_data = _workbook(multi_inventory, [{"source_id": "S", "target_id": "T"}], ("P1", "P2"))
        scenarios.append(("multi_product", [_candidate("R1", "S", "T", "P1"), _candidate("R2", "S", "T", "P2", vhs_rank=2)], multi_data))
        scenarios.append(("high_safety", [_candidate("R1", "S", "T1", qty=40)], _base_data(source_stock=100, safety=99)))
        scenarios.append(("zero_movable", [_candidate("R1", "S", "T1")], _base_data(source_stock=100, safety=100)))
        scenarios.append(("all_negative", [_candidate("R1", "S", "T1", saving=100, cost=200)], base))
        scenarios.append(("equal_vhs", [_candidate("R1", "S", "T1"), _candidate("R2", "S", "T2", vhs_rank=1)], base))
        scenarios.append(("supply_short", [_candidate("R1", "S", "T1", qty=50), _candidate("R2", "S", "T2", qty=50, vhs_rank=2)], base))
        scenarios.append(("supply_long", [_candidate("R1", "S", "T1", qty=100)], _base_data(source_stock=300, safety=0, target_gap=30)))
        bad = _candidate("BAD", "S", "MISSING")
        scenarios.append(("all_infeasible", [bad], base))

        self.assertEqual(len(scenarios), 16)
        for name, recs, data in scenarios:
            with self.subTest(name=name):
                first = build_execution_plan(recs, data, data_signature=name, strategy="greedy")
                second = build_execution_plan(list(reversed(recs)), data, data_signature=name, strategy="greedy")
                self.assertTrue(first["validation"]["valid"])
                self.assertEqual(_decision(first), _decision(second))

    def test_three_hundred_candidate_optimized_runtime_is_bounded(self):
        inventory = [{"store_id": "S", "product_id": "P1", "stock_qty": 4_000, "safety_stock": 1_000}]
        routes, recs = [], []
        for index in range(300):
            target = f"T{index:03d}"
            inventory.append({"store_id": target, "product_id": "P1", "stock_qty": 0, "target_stock": 10, "safety_stock": 0})
            routes.append({"source_id": "S", "target_id": target})
            recs.append(_candidate(f"R{index:03d}", "S", target, qty=10, vhs_rank=index + 1))
        started = time.perf_counter()
        plan = build_execution_plan(recs, _workbook(inventory, routes), timeout_seconds=10)
        elapsed = time.perf_counter() - started
        self.assertTrue(plan["validation"]["valid"])
        self.assertLess(elapsed, 15.0)


if __name__ == "__main__":
    unittest.main()
