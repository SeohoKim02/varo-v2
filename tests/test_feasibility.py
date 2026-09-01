"""Feasibility gate tests: infeasible moves are blocked BEFORE VHS ranking.

Covers each hard-block and soft-flag path with tiny in-code candidates plus an
inventory context, and confirms a valid workbook keeps every candidate.
"""
from __future__ import annotations

import unittest

import pandas as pd

from services.feasibility import (
    STATUS_BLOCKED,
    STATUS_CHECK,
    STATUS_OK,
    InventoryContext,
    annotate_feasibility,
    build_inventory_context,
    evaluate_feasibility,
)


def _ctx(stock=None, demand=None, safety=None, stores=None) -> InventoryContext:
    return InventoryContext(stock or {}, demand or {}, safety or {}, set(stores or []))


class HardBlockTests(unittest.TestCase):
    def test_same_source_and_target_blocked(self):
        rec = {"source_id": "S1", "target_id": "S1", "recommended_qty": 5, "route_type": "DIRECT", "product_id": "P1"}
        self.assertEqual(evaluate_feasibility(rec, _ctx()).status, STATUS_BLOCKED)

    def test_non_positive_or_invalid_quantity_blocked(self):
        for qty in (0, -3, float("nan"), float("inf"), None):
            rec = {"source_id": "S1", "target_id": "S2", "recommended_qty": qty, "route_type": "DIRECT", "product_id": "P1"}
            self.assertEqual(evaluate_feasibility(rec, _ctx()).status, STATUS_BLOCKED, msg=f"qty={qty}")

    def test_missing_or_unknown_route_type_blocked(self):
        for route in ("", "OTHER", None):
            rec = {"source_id": "S1", "target_id": "S2", "recommended_qty": 5, "route_type": route, "product_id": "P1"}
            self.assertEqual(evaluate_feasibility(rec, _ctx()).status, STATUS_BLOCKED, msg=f"route={route}")

    def test_via_dc_without_dc_blocked(self):
        rec = {"source_id": "S1", "target_id": "S2", "recommended_qty": 5, "route_type": "VIA_DC", "dc_id": None, "product_id": "P1"}
        self.assertEqual(evaluate_feasibility(rec, _ctx()).status, STATUS_BLOCKED)

    def test_quantity_exceeding_source_stock_blocked(self):
        ctx = _ctx(stock={("S1", "P1"): 10.0})
        rec = {"source_id": "S1", "target_id": "S2", "recommended_qty": 20, "route_type": "DIRECT",
               "product_id": "P1", "estimated_cost": 100, "expected_saving": 50}
        result = evaluate_feasibility(rec, ctx)
        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.reason_code, "quantity_exceeds_stock")


class SoftFlagTests(unittest.TestCase):
    def test_missing_cost_and_distance_is_check(self):
        rec = {"source_id": "S1", "target_id": "S2", "recommended_qty": 5, "route_type": "DIRECT",
               "product_id": "P1", "estimated_cost": None, "move_cost": None, "distance_km": None}
        self.assertEqual(evaluate_feasibility(rec, _ctx(stock={("S1", "P1"): 100})).status, STATUS_CHECK)

    def test_post_move_below_safety_is_hard_block(self):
        ctx = _ctx(stock={("S1", "P1"): 10.0}, safety={("S1", "P1"): 8.0})
        rec = {"source_id": "S1", "target_id": "S2", "recommended_qty": 5, "route_type": "DIRECT",
               "product_id": "P1", "estimated_cost": 100, "expected_saving": 5000}
        result = evaluate_feasibility(rec, ctx)
        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.reason_code, "inventory_floor_violation")

    def test_oversupply_is_check(self):
        ctx = _ctx(stock={("S1", "P1"): 100.0}, demand={("S2", "P1"): 2.0})
        rec = {"source_id": "S1", "target_id": "S2", "recommended_qty": 50, "route_type": "DIRECT",
               "product_id": "P1", "estimated_cost": 100, "expected_saving": 5000}
        result = evaluate_feasibility(rec, ctx)
        self.assertEqual(result.status, STATUS_CHECK)
        self.assertEqual(result.reason_code, "oversupply")


class FeasibleTests(unittest.TestCase):
    def test_clean_candidate_is_ok(self):
        ctx = _ctx(
            stock={("S1", "P1"): 100.0}, demand={("S2", "P1"): 40.0},
            safety={("S1", "P1"): 0.0},
        )
        rec = {"source_id": "S1", "target_id": "S2", "recommended_qty": 20, "route_type": "DIRECT",
               "product_id": "P1", "estimated_cost": 100, "expected_saving": 5000}
        self.assertEqual(evaluate_feasibility(rec, ctx).status, STATUS_OK)


class AnnotateTests(unittest.TestCase):
    def test_annotate_splits_feasible_and_blocked_and_flags_duplicates(self):
        recs = [
            {"route_id": "A", "source_id": "S1", "target_id": "S2", "product_id": "P1",
             "recommended_qty": 5, "route_type": "DIRECT", "estimated_cost": 10, "expected_saving": 100},
            # exact duplicate of A -> blocked as duplicate
            {"route_id": "B", "source_id": "S1", "target_id": "S2", "product_id": "P1",
             "recommended_qty": 5, "route_type": "DIRECT", "estimated_cost": 10, "expected_saving": 100},
            # same source/target -> blocked
            {"route_id": "C", "source_id": "S3", "target_id": "S3", "product_id": "P2",
             "recommended_qty": 5, "route_type": "DIRECT"},
        ]
        out = annotate_feasibility(recs, data=None)
        self.assertEqual(out["summary"]["blocked_count"], 2)
        self.assertEqual(len(out["feasible"]), 1)
        self.assertEqual(out["feasible"][0]["route_id"], "A")
        # every annotated rec carries the status + reason fields
        for rec in out["annotated"]:
            self.assertIn("feasibility_status", rec)
            self.assertIn("feasibility_reason", rec)

    def test_build_context_from_inventory_frame(self):
        data = {"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 30, "demand_qty": 10, "demand_std": 2},
            {"store_id": "S1", "product_id": "P1", "stock_qty": 20, "demand_qty": 5, "demand_std": 1},
        ])}
        ctx = build_inventory_context(data)
        self.assertEqual(ctx.source_stock("S1", "P1"), 50.0)  # aggregated
        self.assertEqual(ctx.target_demand("S1", "P1"), 15.0)
        self.assertIn("S1", ctx.known_stores)


if __name__ == "__main__":
    unittest.main()
