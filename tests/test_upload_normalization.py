"""Tests that messy real-world uploads normalize into the V2 standard shape."""
from __future__ import annotations

import unittest

from services.data_application import load_and_apply
from services.data_loader import load_excel_data
from tests.fixtures import (
    inventory_frame,
    korean_recommendations_frame,
    products_frame,
    routes_frame,
    stores_frame,
    workbook_excel_bytes,
    dqn_sample_excel_bytes,
)


def _korean_workbook_bytes():
    return workbook_excel_bytes({
        "stores": stores_frame(),
        "products": products_frame(),
        "inventory": inventory_frame(),
        "routes": routes_frame(),
        "recommendations": korean_recommendations_frame(),
    })


class UploadNormalizationTests(unittest.TestCase):
    def test_dqn_sample_01_merges_dc_and_applies_warning_only(self):
        state = {
            "selected_route_id": "OLD", "rec_filter_product": "OLD",
            "simulation_snapshot": {"old": True}, "home_sim_playing": True,
            "dqn_training_result": {"status": "정상"},
        }
        ok = load_and_apply(
            state, dqn_sample_excel_bytes(2, 1),
            "Varo_DQN_sample_01_2stores_1dc_fresh_meal.xlsx", "업로드된 추천 결과",
        )
        self.assertTrue(ok)
        stores = state["varo_data"]["stores"]
        self.assertEqual(int((stores["node_type"] == "DC").sum()), 1)
        route_ids = [row["route_id"] for row in state["varo_recommendations"]]
        self.assertEqual(len(route_ids), len(set(route_ids)))
        self.assertEqual(state["data_apply_message"], "데이터 적용 완료")
        self.assertNotEqual(state["selected_route_id"], "OLD")
        self.assertNotIn("rec_filter_product", state)
        self.assertIsNone(state["simulation_snapshot"])
        self.assertFalse(state["home_sim_playing"])
        self.assertIsNone(state["dqn_training_result"])

    def test_dqn_sample_10_resolves_name_endpoints_and_two_dcs(self):
        state: dict = {}
        ok = load_and_apply(
            state, dqn_sample_excel_bytes(10, 2, names_as_endpoints=True),
            "Varo_DQN_sample_10_10stores_2dc.xlsx", "업로드된 추천 결과",
        )
        self.assertTrue(ok)
        stores = state["varo_data"]["stores"]
        self.assertEqual(int((stores["node_type"] == "STORE").sum()), 10)
        self.assertEqual(int((stores["node_type"] == "DC").sum()), 2)
        self.assertTrue(all(str(row["source_id"]).startswith("S") for row in state["varo_recommendations"]))
        self.assertTrue(all(str(row["target_id"]).startswith("S") for row in state["varo_recommendations"]))
    def test_korean_columns_and_messy_numbers_are_normalized(self):
        data, report = load_excel_data(_korean_workbook_bytes(), return_report=True)
        recs = data["recommendations"]
        for standard in ("route_id", "product_id", "source_id", "target_id",
                         "route_type", "recommended_qty", "estimated_cost",
                         "expected_saving", "vhs_score"):
            self.assertIn(standard, recs.columns)
        # "12,000원" -> 12000 numeric
        self.assertEqual(float(recs.loc[recs["route_id"] == "R001", "expected_saving"].iloc[0]), 12000.0)
        self.assertEqual(float(recs.loc[recs["route_id"] == "R001", "estimated_cost"].iloc[0]), 3000.0)
        self.assertGreater(report["mapped_column_count"] if "mapped_column_count" in report else len(report["column_mappings"]), 0)

    def test_load_and_apply_handles_korean_upload(self):
        state: dict = {}
        ok = load_and_apply(state, _korean_workbook_bytes(), "korean.xlsx", "업로드된 추천 결과")
        self.assertTrue(ok)
        self.assertTrue(state["varo_recommendations"])
        report = state.get("upload_report") or {}
        self.assertGreater(report.get("mapped_column_count", 0), 0)
        self.assertEqual(report.get("recommendation_source"), "uploaded")
        self.assertTrue(report.get("analyzable"))
        # selected route persists and R002 is present
        ids = {r["route_id"] for r in state["varo_recommendations"]}
        self.assertIn("R002", ids)

    def test_date_column_is_converted_to_days_to_expiry(self):
        import pandas as pd
        inventory = inventory_frame().drop(columns=["days_to_expiry", "expiry_days"])
        inventory["만료일"] = "2026-12-31"
        data, report = load_excel_data(workbook_excel_bytes({
            "stores": stores_frame(), "products": products_frame(),
            "inventory": inventory, "routes": routes_frame(),
            "recommendations": korean_recommendations_frame(),
        }), return_report=True)
        self.assertIn("days_to_expiry", data["inventory"].columns)
        self.assertGreater(report["date_success"], 0)
        self.assertIn("만료일", report["date_columns"])

    def test_blank_rows_and_dqn_columns_do_not_leak(self):
        recs = korean_recommendations_frame()
        recs["reward"] = 999999  # injected DQN value column
        recs["loss"] = 1
        data = load_excel_data(workbook_excel_bytes({
            "stores": stores_frame(), "products": products_frame(),
            "inventory": inventory_frame(), "routes": routes_frame(),
            "recommendations": recs,
        }))
        from services.dqn_guard import strip_dqn_columns
        guarded = strip_dqn_columns(data["recommendations"])
        self.assertNotIn("reward", guarded.columns)
        self.assertNotIn("loss", guarded.columns)


if __name__ == "__main__":
    unittest.main()
