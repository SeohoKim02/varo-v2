"""Tests for the V2 VHS neutral-components summary."""
from __future__ import annotations

import unittest

from services import v2_summaries as v2
from services.analysis_pipeline import run_analysis_pipeline
from tests.fixtures import sample_workbook

_DQN_VALUE_TOKENS = ("reward", "loss", "q_table", "policy_table", "replay", "dqn_correction")


class VhsNeutralComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_analysis_pipeline(sample_workbook())
        cls.pipeline = cls.result.to_dict()
        cls.recommendations = cls.result.recommendations

    def test_summary_counts_are_consistent(self):
        summary = v2.vhs_neutral_summary(self.pipeline)
        self.assertEqual(
            summary["total_components"],
            summary["calculated_components"] + summary["neutral_components"],
        )
        self.assertGreater(summary["total_components"], 0)
        self.assertGreaterEqual(summary["neutral_components"], 0)
        self.assertEqual(summary["excluded_components"], 1)  # DQN 보정 제외

    def test_summary_is_attached_to_pipeline_result(self):
        attached = self.pipeline.get("vhs_neutral_analysis") or {}
        self.assertEqual(attached.get("total_components"),
                         v2.vhs_neutral_summary(self.pipeline)["total_components"])

    def test_rows_have_uploaded_and_recalculated_vhs(self):
        rows = v2.vhs_neutral_rows(self.pipeline, self.recommendations)
        self.assertEqual(len(rows), len(self.recommendations))
        for row in rows:
            for column in (
                "route_id", "product_name", "uploaded_vhs", "recalculated_vhs",
                "calculated_components", "neutral_components", "excluded_components",
                "neutral_reason", "final_basis", "note",
            ):
                self.assertIn(column, row)

    def test_summary_contains_no_dqn_values(self):
        summary = v2.vhs_neutral_summary(self.pipeline)
        blob = " ".join(str(value) for value in summary.values()).lower()
        # explanatory mention of DQN is allowed, numeric DQN value tokens are not
        for token in ("reward", "loss", "q_table", "policy_table", "replay"):
            self.assertNotIn(token, blob)

    def test_empty_pipeline_does_not_crash(self):
        summary = v2.vhs_neutral_summary({})
        self.assertEqual(summary["total_components"], 0)
        self.assertEqual(v2.vhs_neutral_rows({}, []), [])


if __name__ == "__main__":
    unittest.main()
