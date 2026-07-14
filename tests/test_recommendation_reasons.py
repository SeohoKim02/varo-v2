"""Tests for the V2 internal recommendation-reason summary."""
from __future__ import annotations

import unittest

from services import v2_summaries as v2
from services.analysis_pipeline import run_analysis_pipeline
from tests.fixtures import sample_workbook


class RecommendationReasonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_analysis_pipeline(sample_workbook())
        cls.recommendations = cls.result.recommendations
        cls.reasons = v2.recommendation_reasons(cls.recommendations)

    def test_reason_for_every_route(self):
        self.assertEqual(set(self.reasons), {str(r["route_id"]) for r in self.recommendations})

    def test_r002_reason_structure(self):
        detail = self.reasons["R002"]
        self.assertGreaterEqual(len(detail["sentences"]), 1)
        self.assertLessEqual(len(detail["sentences"]), 4)
        self.assertTrue(detail["caution"])
        self.assertIn("DQN", detail["dqn_note"])
        self.assertIn("반영하지 않", detail["dqn_note"])

    def test_reason_never_contains_dqn_value(self):
        for detail in self.reasons.values():
            text = " ".join(detail["sentences"]).lower()
            for token in ("reward", "loss", "q_table", "policy_table", "replay", "dqn_correction"):
                self.assertNotIn(token, text)

    def test_attached_to_pipeline_result(self):
        analysis = self.result.to_dict().get("reason_analysis") or {}
        self.assertIn("reasons", analysis)
        self.assertEqual(set(analysis["reasons"]), set(self.reasons))
        self.assertIn("보류", analysis.get("calculation_basis", ""))

    def test_reason_rows_for_export(self):
        rows = v2.recommendation_reason_rows(self.recommendations)
        self.assertEqual(len(rows), len(self.recommendations))
        for row in rows:
            for column in ("route_id", "product_name", "recommendation_reason", "caution", "dqn_note"):
                self.assertIn(column, row)

    def test_empty_input_returns_empty(self):
        self.assertEqual(v2.recommendation_reasons([]), {})
        self.assertEqual(v2.recommendation_reason_rows([]), [])


if __name__ == "__main__":
    unittest.main()
