"""Tests for the V2 internal sensitivity summary."""
from __future__ import annotations

import unittest

from services import v2_summaries as v2
from services.analysis_pipeline import run_analysis_pipeline
from tests.fixtures import sample_workbook

GRADES = {"낮음", "보통", "높음", "제한적"}


class V2SensitivitySummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_analysis_pipeline(sample_workbook())
        cls.recommendations = cls.result.recommendations
        cls.rows = v2.sensitivity_summary(cls.recommendations)

    def test_one_row_per_recommendation(self):
        self.assertEqual(len(self.rows), len(self.recommendations))
        ids = {str(row["route_id"]) for row in self.rows}
        self.assertEqual(ids, {str(r["route_id"]) for r in self.recommendations})

    def test_all_grades_are_in_domain(self):
        for row in self.rows:
            for key in (
                "sensitivity_cost", "sensitivity_distance", "sensitivity_quantity",
                "sensitivity_vhs", "overall_sensitivity",
            ):
                self.assertIn(row[key], GRADES)
            self.assertTrue(row["stability_note"])

    def test_r002_is_present(self):
        self.assertIn("R002", {str(row["route_id"]) for row in self.rows})

    def test_attached_to_pipeline_result(self):
        analysis = self.result.to_dict().get("sensitivity_analysis") or {}
        self.assertIn("rows", analysis)
        self.assertEqual(len(analysis["rows"]), len(self.recommendations))
        self.assertIn("보류", analysis.get("calculation_basis", ""))

    def test_no_dqn_column_in_rows(self):
        for row in self.rows:
            for key in row:
                self.assertNotIn("dqn", str(key).lower())
                self.assertNotIn("reward", str(key).lower())

    def test_degenerate_inputs_do_not_crash(self):
        self.assertEqual(v2.sensitivity_summary([]), [])
        single = v2.sensitivity_summary([self.recommendations[0]])
        self.assertEqual(len(single), 1)
        # with a single candidate, gaps cannot be computed -> 제한적
        self.assertEqual(single[0]["sensitivity_cost"], "제한적")


if __name__ == "__main__":
    unittest.main()
