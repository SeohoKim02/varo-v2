"""Operational inventory-floor contract: explicit policy first, safe fallback second."""
from __future__ import annotations

import unittest

import pandas as pd

from components.candidate_detail import render_quantity_basis
from services.analysis_pipeline import run_analysis_pipeline
from services.candidate_generator import generate_candidates
from services.column_aliases import INVENTORY_ALIASES, normalize_columns
from services.data_issues import collect_data_issues
from services.data_loader import normalize_loaded_data
from services.decision_metrics import quantity_plan
from services.feasibility import (
    STATUS_BLOCKED,
    STATUS_CHECK,
    STATUS_OK,
    build_inventory_context,
    evaluate_feasibility,
    inventory_floor_source_label,
)
from services.partial_data import build_usable_data
from tests.fixtures import sample_workbook


def _inventory(**overrides):
    row = {
        "store_id": "S1", "product_id": "P1", "stock_qty": 100,
        "demand_qty": 20, "demand_std": 4,
    }
    row.update(overrides)
    return {"inventory": pd.DataFrame([row])}


def _candidate(qty=10, **overrides):
    row = {
        "route_id": "R1", "source_id": "S1", "target_id": "S2",
        "product_id": "P1", "route_type": "DIRECT", "recommended_qty": qty,
        "estimated_cost": 100, "expected_saving": 5000,
    }
    row.update(overrides)
    return row


class ColumnRecognitionTests(unittest.TestCase):
    def test_supported_english_and_korean_aliases(self):
        cases = {
            "safety_stock": ("safety_stock", "안전재고"),
            "min_stock": ("min_stock", "minimum_stock", "minimum_inventory", "min_inventory",
                          "stock_floor", "safety_floor", "최소재고", "최소 보유량", "최소 보유재고"),
            "reorder_point": ("reorder_point", "재주문점"),
            "target_stock": ("target_stock", "목표재고"),
        }
        for standard, aliases in cases.items():
            for alias in aliases:
                with self.subTest(standard=standard, alias=alias):
                    normalized, _ = normalize_columns(pd.DataFrame({alias: [12]}), INVENTORY_ALIASES)
                    self.assertEqual(float(normalized[standard].iloc[0]), 12.0)

    def test_whitespace_and_case_are_normalized(self):
        normalized, _ = normalize_columns(
            pd.DataFrame({" Safety_Stock ": [15]}), INVENTORY_ALIASES,
        )
        self.assertEqual(normalized["safety_stock"].iloc[0], 15)

    def test_two_aliases_for_one_policy_are_a_file_conflict(self):
        raw = pd.DataFrame([{
            "store_id": "S1", "product_id": "P1", "stock_qty": 100,
            "안전재고": 10, "safety_inventory": 20,
        }])
        issues = collect_data_issues({"inventory": raw})["issues"]
        self.assertTrue(any(i["code"] == "alias_conflict" for i in issues))

    def test_unrelated_stock_column_is_not_a_safety_policy(self):
        normalized, _ = normalize_columns(pd.DataFrame({"stock": [30]}), INVENTORY_ALIASES)
        self.assertIn("stock_qty", normalized.columns)
        self.assertNotIn("safety_stock", normalized.columns)
        self.assertNotIn("min_stock", normalized.columns)


class FloorPriorityAndProvenanceTests(unittest.TestCase):
    def test_explicit_safety_stock_overrides_higher_or_lower_estimate(self):
        high = build_inventory_context(_inventory(safety_stock=15, demand_std=4))
        low = build_inventory_context(_inventory(safety_stock=3, demand_std=4))
        self.assertEqual(high.safety_floor("S1", "P1"), 15)
        self.assertEqual(low.safety_floor("S1", "P1"), 3)
        self.assertEqual(high.inventory_floor_source("S1", "P1"), "explicit_safety_stock")
        self.assertEqual(low.inventory_floor_source("S1", "P1"), "explicit_safety_stock")

    def test_explicit_zero_is_distinct_from_unavailable(self):
        explicit = build_inventory_context(_inventory(safety_stock=0, demand_std=None))
        unavailable = build_inventory_context(_inventory(demand_std=None))
        self.assertEqual(explicit.safety_floor("S1", "P1"), 0)
        self.assertEqual(explicit.inventory_floor_source("S1", "P1"), "explicit_safety_stock")
        self.assertIsNone(unavailable.safety_floor("S1", "P1"))
        self.assertEqual(unavailable.inventory_floor_source("S1", "P1"), "unavailable")

    def test_minimum_and_safety_remain_distinct_but_stricter_floor_wins(self):
        context = build_inventory_context(_inventory(min_stock=12, safety_stock=20))
        self.assertEqual(context.safety_floor("S1", "P1"), 20)
        self.assertEqual(context.inventory_floor_source("S1", "P1"), "explicit_combined")

    def test_estimate_is_used_only_when_explicit_value_is_missing(self):
        estimated = build_inventory_context(_inventory(demand_std=4))
        missing_std = build_inventory_context(_inventory(demand_std=None))
        self.assertEqual(estimated.safety_floor("S1", "P1"), 8)
        self.assertEqual(estimated.inventory_floor_source("S1", "P1"), "estimated")
        self.assertIsNone(missing_std.safety_floor("S1", "P1"))

    def test_invalid_nan_is_never_used_and_falls_back(self):
        context = build_inventory_context(_inventory(safety_stock=float("nan"), demand_std=5))
        self.assertEqual(context.safety_floor("S1", "P1"), 10)
        self.assertEqual(context.inventory_floor_source("S1", "P1"), "estimated")

    def test_reorder_and_target_are_not_departure_floors(self):
        context = build_inventory_context(_inventory(demand_std=None, reorder_point=70, target_stock=90))
        self.assertIsNone(context.safety_floor("S1", "P1"))
        self.assertEqual(context.reorder_point("S1", "P1"), 70)
        self.assertEqual(context.target_stock_level("S1", "P1"), 90)

    def test_store_product_values_do_not_cross_keys(self):
        frame = pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 100, "safety_stock": 10},
            {"store_id": "S1", "product_id": "P2", "stock_qty": 100, "safety_stock": 20},
            {"store_id": "S2", "product_id": "P1", "stock_qty": 100, "safety_stock": 30},
        ])
        context = build_inventory_context({"inventory": frame})
        self.assertEqual(context.safety_floor("S1", "P1"), 10)
        self.assertEqual(context.safety_floor("S1", "P2"), 20)
        self.assertEqual(context.safety_floor("S2", "P1"), 30)
        self.assertIsNone(context.safety_floor("S2", "P2"))

    def test_user_labels_never_expose_source_codes(self):
        for source in ("explicit_min_stock", "explicit_safety_stock", "explicit_combined", "estimated", "unavailable"):
            label = inventory_floor_source_label(source)
            self.assertNotIn(source, label)
            self.assertNotIn("_", label)


class ValidationAndPartialApplicationTests(unittest.TestCase):
    def test_normal_numeric_string_is_accepted(self):
        normalized = normalize_loaded_data({"inventory": pd.DataFrame([{
            "store_id": "S1", "product_id": "P1", "stock_qty": 100, "안전재고": "15",
        }])})
        self.assertEqual(normalized["inventory"]["safety_stock"].iloc[0], 15.0)

    def test_negative_infinite_and_text_policy_values_are_reported(self):
        frame = pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 100, "safety_stock": -10},
            {"store_id": "S2", "product_id": "P1", "stock_qty": 100, "safety_stock": float("inf")},
            {"store_id": "S3", "product_id": "P1", "stock_qty": 100, "safety_stock": "abc"},
        ])
        issues = [i for i in collect_data_issues({"inventory": frame})["issues"]
                  if i.get("canonical_column_name") == "safety_stock"]
        self.assertEqual({i["code"] for i in issues}, {"negative", "non_numeric"})
        self.assertTrue(all("0 이상의 숫자" in i["문제"] for i in issues))
        self.assertTrue(all(i["수정 방법"] == "0 이상의 숫자로 수정하세요." for i in issues))

    def test_conflicting_store_product_policy_is_excluded(self):
        frame = pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 100, "safety_stock": 10},
            {"store_id": "S1", "product_id": "P1", "stock_qty": 100, "safety_stock": 20},
        ])
        result = collect_data_issues({"inventory": frame})
        conflicts = [i for i in result["issues"] if i["code"] == "conflict_duplicate"]
        self.assertEqual(len(conflicts), 2)
        self.assertEqual({tuple(i["exclusion_rows"]) for i in conflicts}, {(2, 3)})

    def test_bad_optional_policy_row_uses_existing_row_exclusion(self):
        workbook = sample_workbook()
        workbook["inventory"] = workbook["inventory"].copy()
        workbook["inventory"]["safety_stock"] = 5.0
        workbook["inventory"].loc[0, "safety_stock"] = -1
        result = build_usable_data(workbook, workbook, {"source_type": "excel"})
        self.assertTrue(result["apply_allowed"])
        self.assertEqual(result["quality_summary"]["excluded_rows"], 1)
        self.assertFalse((result["usable_data"]["inventory"]["safety_stock"] < 0).any())


class QuantityAndHardConstraintTests(unittest.TestCase):
    def test_exact_floor_passes_and_one_unit_below_is_blocked(self):
        context = build_inventory_context(_inventory(stock_qty=30, safety_stock=20))
        exact = evaluate_feasibility(_candidate(10), context)
        below = evaluate_feasibility(_candidate(11), context)
        self.assertEqual(exact.status, STATUS_OK)
        self.assertEqual(below.status, STATUS_BLOCKED)
        self.assertEqual(below.reason_code, "inventory_floor_violation")

    def test_stock_equal_or_below_floor_has_zero_available(self):
        equal = build_inventory_context(_inventory(stock_qty=20, safety_stock=20))
        below = build_inventory_context(_inventory(stock_qty=19, safety_stock=20))
        self.assertEqual(equal.available_to_move("S1", "P1"), 0)
        self.assertEqual(below.available_to_move("S1", "P1"), 0)
        self.assertEqual(evaluate_feasibility(_candidate(1), equal).status, STATUS_BLOCKED)
        self.assertEqual(evaluate_feasibility(_candidate(1), below).status, STATUS_BLOCKED)

    def test_unavailable_floor_is_check_not_fabricated_zero(self):
        context = build_inventory_context(_inventory(demand_std=None))
        result = evaluate_feasibility(_candidate(10), context)
        self.assertEqual(result.status, STATUS_CHECK)
        self.assertEqual(result.reason_code, "inventory_floor_unavailable")
        self.assertIsNone(context.available_to_move("S1", "P1"))

    def test_target_goal_limits_shortfall_without_becoming_a_floor(self):
        context = build_inventory_context({"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 100, "safety_stock": 10},
            {"store_id": "S2", "product_id": "P1", "stock_qty": 25, "target_stock": 40, "safety_stock": 0},
        ])})
        plan = quantity_plan(_candidate(10), context)
        self.assertEqual(plan["target_shortfall"], 15)
        self.assertEqual(plan["target_stock_goal"], 40)
        self.assertEqual(plan["target_stock_basis"], "explicit_target_stock")
        self.assertEqual(context.safety_floor("S1", "P1"), 10)

    def test_generator_caps_quantity_at_available_to_move(self):
        data = {
            "stores": pd.DataFrame([
                {"node_id": "DC1", "node_name": "DC", "node_type": "DC"},
                {"node_id": "S1", "node_name": "A", "node_type": "STORE"},
                {"node_id": "S2", "node_name": "B", "node_type": "STORE"},
            ]),
            "products": pd.DataFrame([{"product_id": "P1", "product_name": "상품", "unit_price": 1000}]),
            "inventory": pd.DataFrame([
                {"store_id": "S1", "product_id": "P1", "stock_qty": 100, "safety_stock": 80,
                 "avg_daily_sales": 1, "days_to_expiry": 2},
                {"store_id": "S2", "product_id": "P1", "stock_qty": 0, "safety_stock": 0,
                 "avg_daily_sales": 20, "days_to_expiry": 20},
            ]),
            "routes": pd.DataFrame([{
                "source_id": "S1", "target_id": "S2", "distance_km": 1,
                "estimated_cost": 10, "travel_time_min": 5,
            }]),
        }
        generated, _ = generate_candidates(data)
        self.assertIsNotNone(generated)
        self.assertTrue((generated["recommended_qty"] <= 20).all())

    def test_pipeline_removes_floor_violations_before_final_recommendations(self):
        workbook = sample_workbook()
        inventory = workbook["inventory"].copy()
        inventory["safety_stock"] = 0.0
        mask = (inventory["store_id"] == "S001") & (inventory["product_id"] == "P001")
        inventory.loc[mask, "safety_stock"] = 80.0
        workbook["inventory"] = inventory
        result = run_analysis_pipeline(workbook)
        final_ids = {str(row["route_id"]) for row in result.recommendations}
        self.assertNotIn("R001", final_ids)
        blocked = [r for r in result.candidate_ledger if r.get("route_id") == "R001"]
        self.assertEqual(blocked[0]["status"], "이동 불가")
        self.assertEqual(blocked[0]["quantity_basis"]["inventory_floor_source"], "explicit_safety_stock")
        for row in result.recommendations:
            floor = row.get("inventory_floor_value")
            remaining = row.get("source_movable")
            self.assertFalse(floor is not None and remaining is not None and remaining < 0)


class UiContractTests(unittest.TestCase):
    class _FakeStreamlit:
        def __init__(self):
            self.captions = []

        def caption(self, value):
            self.captions.append(str(value))

    def test_quantity_detail_is_plain_korean_and_hides_internal_fields(self):
        st = self._FakeStreamlit()
        record = {"quantity_basis": {
            "source_stock": 100, "source_safety": 40, "source_movable": 60,
            "recommended_qty": 25,
            "basis_text": "도착 점포 부족량을 기준으로 25개 이동을 권장합니다.",
            "inventory_floor_source": "explicit_safety_stock",
            "inventory_floor_source_label": "등록된 안전재고 기준",
        }}
        render_quantity_basis(st, record)
        blob = " ".join(st.captions)
        for text in ("출발 현재 재고 100개", "남겨야 할 재고 40개", "이동 가능 60개",
                     "권장 이동 25개", "등록된 안전재고 기준"):
            self.assertIn(text, blob)
        for hidden in ("explicit_safety_stock", "inventory_floor", "demand_std", "× 2"):
            self.assertNotIn(hidden, blob)


if __name__ == "__main__":
    unittest.main()
