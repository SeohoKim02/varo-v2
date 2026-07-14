"""Tests for V2 candidate generation when an upload has no recommendation sheet."""
from __future__ import annotations

import unittest

import pandas as pd

from services import candidate_generator as cg
from services.analysis_pipeline import build_v2_state, run_analysis_pipeline
from services.recommendation_adapter import recommendations_from_dataframe
from tests.fixtures import sample_workbook


def _workbook_without_recommendations() -> dict:
    workbook = sample_workbook()
    workbook.pop("recommendations", None)
    return workbook


def _via_dc_workbook() -> dict:
    """A -> DC -> B reachable, no direct A->B; A surplus, B needy."""
    return {
        "stores": pd.DataFrame([
            {"node_id": "DC01", "node_name": "센터", "node_type": "DC"},
            {"node_id": "A", "node_name": "A점", "node_type": "STORE"},
            {"node_id": "B", "node_name": "B점", "node_type": "STORE"},
        ]),
        "products": pd.DataFrame([{"product_id": "P1", "product_name": "상품", "unit_price": 5000}]),
        "inventory": pd.DataFrame([
            {"store_id": "A", "product_id": "P1", "stock_qty": 100, "avg_daily_sales": 2, "days_to_expiry": 20},
            {"store_id": "B", "product_id": "P1", "stock_qty": 5, "avg_daily_sales": 1, "days_to_expiry": 20},
        ]),
        "routes": pd.DataFrame([
            {"source_id": "A", "target_id": "DC01", "distance_km": 3, "estimated_cost": 2000, "travel_time_min": 8},
            {"source_id": "DC01", "target_id": "B", "distance_km": 4, "estimated_cost": 2200, "travel_time_min": 9},
        ]),
    }


class CandidateGenerationTests(unittest.TestCase):
    def test_generates_valid_candidates(self):
        frame, info = cg.generate_candidates(_workbook_without_recommendations())
        self.assertTrue(info["generated"])
        self.assertGreater(info["count"], 0)
        self.assertIsNotNone(frame)
        recs = recommendations_from_dataframe(frame)  # must pass the adapter contract
        self.assertEqual(len(recs), info["count"])
        self.assertTrue(all(str(r["route_id"]).startswith("V2C") for r in recs))
        self.assertTrue(all(r["dqn_action"] == "미연결" for r in recs))

    def test_generated_candidates_flow_through_pipeline(self):
        state = build_v2_state(_workbook_without_recommendations())
        self.assertEqual(state["recommendation_source"], "generated")
        self.assertIn(state["pipeline_result"]["status"], ("success", "partial"))
        self.assertGreater(len(state["recommendations"]), 0)
        self.assertTrue(all(r["dqn_action"] == "미연결" for r in state["recommendations"]))

    def test_cannot_generate_without_routes(self):
        workbook = _workbook_without_recommendations()
        workbook["routes"] = workbook["routes"].iloc[0:0]  # empty routes
        frame, info = cg.generate_candidates(workbook)
        self.assertIsNone(frame)
        self.assertFalse(info["generated"])
        self.assertIn("reason", info)

    def test_missing_data_returns_reason_not_crash(self):
        frame, info = cg.generate_candidates({"stores": None})
        self.assertIsNone(frame)
        self.assertFalse(info["generated"])

    def test_uploaded_recommendations_take_priority(self):
        # when a recommendation sheet exists it is used as-is (not generated)
        state = build_v2_state(sample_workbook())
        self.assertEqual(state["recommendation_source"], "uploaded")

    def test_no_reward_or_loss_columns_in_generated(self):
        frame, _ = cg.generate_candidates(_workbook_without_recommendations())
        for column in frame.columns:
            lowered = str(column).lower()
            for token in ("reward", "loss", "q_table", "policy_table", "dqn_correction"):
                self.assertNotIn(token, lowered)

    def test_via_dc_candidate_when_no_direct_route(self):
        frame, info = cg.generate_candidates(_via_dc_workbook())
        self.assertTrue(info["generated"])
        self.assertEqual(info["via_dc_count"], info["count"])
        self.assertEqual(info["direct_count"], 0)
        self.assertEqual(frame["route_type"].iloc[0], "VIA_DC")
        self.assertEqual(frame["dc_id"].iloc[0], "DC01")
        recs = recommendations_from_dataframe(frame)
        self.assertEqual(recs[0]["route_type"], "VIA_DC")
        self.assertEqual(recs[0]["dc_id"], "DC01")
        self.assertTrue(info["candidates"][0]["via_dc_available"])
        self.assertFalse(info["candidates"][0]["direct_available"])

    def test_candidate_scores_in_range_and_qty_bounded(self):
        frame, info = cg.generate_candidates(_workbook_without_recommendations())
        for detail in info["candidates"]:
            self.assertGreaterEqual(detail["candidate_score"], 0)
            self.assertLessEqual(detail["candidate_score"], 100)
        # recommended_qty is at least 1 and never exceeds source stock
        inv = _workbook_without_recommendations()["inventory"]
        for _, row in frame.iterrows():
            src_stock = float(inv[(inv["store_id"] == row["source_id"]) & (inv["product_id"] == row["product_id"])]["stock_qty"].iloc[0])
            self.assertGreaterEqual(row["recommended_qty"], 1)
            self.assertLessEqual(row["recommended_qty"], src_stock)

    def test_self_send_excluded_and_unique_route_ids(self):
        frame, info = cg.generate_candidates(_workbook_without_recommendations())
        for detail in info["candidates"]:
            self.assertNotEqual(detail["source_name"], detail["target_name"])
        ids = [r["route_id"] for r in info["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(str(i).startswith("V2C") for i in ids))

    def test_generation_stats_present(self):
        _, info = cg.generate_candidates(_workbook_without_recommendations())
        for key in ("direct_count", "via_dc_count", "route_deferred",
                    "negative_saving_excluded", "duplicate_removed", "qty_excluded", "score_components"):
            self.assertIn(key, info)

    def test_dqn_status_is_disconnected_in_details(self):
        _, info = cg.generate_candidates(_via_dc_workbook())
        self.assertTrue(all(d["dqn_status"] == "미연결" for d in info["candidates"]))


if __name__ == "__main__":
    unittest.main()
