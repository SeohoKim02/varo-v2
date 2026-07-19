"""Inventory transition invariants and sequential execution contracts."""
from __future__ import annotations

import copy
import unittest

import pandas as pd

from services.inventory_transition_service import (
    INVENTORY_TRANSITION_VERSION,
    cached_inventory_scenario,
    calculate_inventory_transition,
    classify_inventory_state,
    clear_inventory_transition_cache,
    inventory_transition_cache_info,
    run_inventory_scenario,
)


def _data() -> dict:
    return {
        "stores": pd.DataFrame([
            {"node_id": "S1", "node_name": "출발점", "node_type": "STORE"},
            {"node_id": "S2", "node_name": "도착점", "node_type": "STORE"},
            {"node_id": "DC01", "node_name": "동부 물류센터", "node_type": "DC", "capacity": 500},
        ]),
        "products": pd.DataFrame([
            {"product_id": "P1", "product_name": "신선식품", "min_display_stock": 5},
        ]),
        "inventory": pd.DataFrame([
            {
                "store_id": "S1", "store_name": "출발점", "product_id": "P1", "product_name": "신선식품",
                "stock_qty": 30, "avg_daily_sales": 2, "days_to_expiry": 5, "sales_7d": 14,
                "disposal_risk_score": 80,
            },
            {
                "store_id": "S2", "store_name": "도착점", "product_id": "P1", "product_name": "신선식품",
                "stock_qty": 2, "avg_daily_sales": 4, "days_to_expiry": 5, "sales_7d": 28,
                "disposal_risk_score": 10,
            },
        ]),
    }


def _recommendation(route_type: str = "DIRECT", quantity: int = 15) -> dict:
    return {
        "recommendation_id": "REC1", "route_id": "R1", "product_id": "P1", "product_name": "신선식품",
        "source_id": "S1", "source_name": "출발점", "target_id": "S2", "target_name": "도착점",
        "route_type": route_type, "dc_id": "DC01" if route_type == "VIA_DC" else None,
        "dc_name": "동부 물류센터" if route_type == "VIA_DC" else None,
        "recommended_qty": quantity, "expected_saving": 15000,
    }


class InventoryTransitionTests(unittest.TestCase):
    def setUp(self):
        clear_inventory_transition_cache()

    def test_direct_transition_preserves_inventory_and_never_goes_negative(self):
        result = calculate_inventory_transition(_data(), _recommendation())
        self.assertEqual(result["version"], INVENTORY_TRANSITION_VERSION)
        self.assertEqual(result["applied_quantity"], 15)
        self.assertEqual(result["source"]["before_stock"], 30)
        self.assertEqual(result["source"]["after_stock"], 15)
        self.assertEqual(result["target"]["before_stock"], 2)
        self.assertEqual(result["target"]["after_stock"], 17)
        self.assertEqual(result["inventory_totals"], {"before": 32.0, "after": 32.0})
        self.assertTrue(result["invariants"]["all_passed"])
        self.assertTrue(result["invariants"]["direct_outbound_equals_inbound"])

    def test_via_dc_uses_two_leg_conservation_without_dc_stock_creation(self):
        result = calculate_inventory_transition(_data(), _recommendation("VIA_DC"))
        self.assertEqual(result["route_type"], "VIA_DC")
        self.assertEqual(result["source"]["outbound_qty"], result["target"]["inbound_qty"])
        self.assertTrue(result["invariants"]["via_dc_outbound_equals_inbound"])
        self.assertTrue(result["invariants"]["inventory_preserved"])

    def test_movable_and_target_limits_clip_requested_quantity(self):
        data = _data()
        data["inventory"].loc[data["inventory"]["store_id"] == "S2", "sales_7d"] = 8
        result = calculate_inventory_transition(data, _recommendation(quantity=50))
        self.assertEqual(result["metadata"]["movable_stock"], 15)
        self.assertEqual(result["metadata"]["target_shortage_limit"], 6)
        self.assertEqual(result["applied_quantity"], 6)
        self.assertEqual(result["status"], "제약 수량으로 실행")
        self.assertIsNone(result["expected_saving"])
        self.assertTrue(result["invariants"]["inbound_within_shortage"])

    def test_state_transition_uses_exact_actual_demand_boundary(self):
        result = calculate_inventory_transition(_data(), _recommendation())
        self.assertEqual(result["source"]["before_status"], "초과재고")
        self.assertEqual(result["source"]["after_status"], "적정재고")
        self.assertEqual(result["target"]["before_status"], "부족재고")
        self.assertTrue(result["source"]["status_improved"])
        self.assertTrue(result["target"]["status_improved"])
        self.assertEqual(classify_inventory_state(10, 10), "적정재고")

    def test_missing_required_fields_returns_data_shortage_without_fabrication(self):
        data = {"inventory": pd.DataFrame([{"store_id": "S1", "product_id": "P1"}])}
        result = calculate_inventory_transition(data, _recommendation())
        self.assertFalse(result["feasible"])
        self.assertEqual(result["applied_quantity"], 0)
        self.assertIn("확인할 수 없음", result["skipped_reason"])

    def test_original_frames_and_recommendation_are_unchanged(self):
        data = _data()
        inventory_before = data["inventory"].copy(deep=True)
        stores_before = data["stores"].copy(deep=True)
        recommendation = _recommendation()
        recommendation_before = copy.deepcopy(recommendation)
        calculate_inventory_transition(data, recommendation)
        pd.testing.assert_frame_equal(data["inventory"], inventory_before)
        pd.testing.assert_frame_equal(data["stores"], stores_before)
        self.assertEqual(recommendation, recommendation_before)

    def test_sequential_routes_cannot_reuse_the_same_source_excess(self):
        first = _recommendation(quantity=15)
        second = {**_recommendation(quantity=5), "route_id": "R2", "recommendation_id": "REC2"}
        scenario = run_inventory_scenario(_data(), [first, second])
        self.assertEqual(scenario["requested_route_count"], 2)
        self.assertEqual(scenario["executed_route_count"], 1)
        self.assertEqual(scenario["skipped_route_count"], 1)
        self.assertEqual(scenario["transitions"][1]["status"], "현재 상태에서 실행 불가")
        self.assertEqual(scenario["transitions"][1]["applied_quantity"], 0)

    def test_integer_inputs_keep_integer_transfer_quantity(self):
        result = calculate_inventory_transition(_data(), _recommendation(quantity=14))
        self.assertIsInstance(result["applied_quantity"], int)
        self.assertTrue(result["invariants"]["integer_rule_preserved"])

    def test_cached_scenario_is_reused_and_returned_as_a_copy(self):
        first = cached_inventory_scenario("sig", _data(), [_recommendation()])
        second = cached_inventory_scenario("sig", _data(), [_recommendation()])
        self.assertFalse(first["performance"]["cache_hit"])
        self.assertTrue(second["performance"]["cache_hit"])
        second["transitions"][0]["applied_quantity"] = -1
        third = cached_inventory_scenario("sig", _data(), [_recommendation()])
        self.assertNotEqual(third["transitions"][0]["applied_quantity"], -1)
        counters = inventory_transition_cache_info()
        self.assertEqual(counters["inventory_transition_calculations"], 1)
        self.assertEqual(counters["inventory_transition_cache_hits"], 2)


if __name__ == "__main__":
    unittest.main()

