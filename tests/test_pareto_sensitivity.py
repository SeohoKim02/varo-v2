"""Tests for Pareto analysis, VHS weight sensitivity, and DQN result fields.

These cover the research-extension comparison structures added on top of the
VHS/Greedy/DQN pipeline. All deterministic and self-contained; no PyTorch and no
real training run is required.
"""
from __future__ import annotations

import unittest

from services.analysis_pipeline import run_analysis_pipeline
from services.dqn_service import train_dqn
from services.pareto_analysis import PARETO_OBJECTIVES, compute_pareto, pareto_summary
from services.vhs_score_engine import apply_auto_vhs, weight_sensitivity
from tests.fixtures import recommendations_frame, sample_workbook


class ParetoTests(unittest.TestCase):
    def test_dominance_assigns_fronts(self):
        # A and C are mutually non-dominated (front 1); B is dominated by A (front 2).
        dominant = {name: 90.0 for name in PARETO_OBJECTIVES}
        dominated = {name: 50.0 for name in PARETO_OBJECTIVES}
        # Better on one current objective, worse on another -> neither dominates.
        trade_off = {**{name: 90.0 for name in PARETO_OBJECTIVES},
                     "net_benefit_score": 95.0, "disposal_risk_score": 10.0}
        rows = compute_pareto([dominant, dominated, trade_off])
        self.assertEqual(rows[0]["pareto_rank"], 1)
        self.assertEqual(rows[0]["pareto_status"], "비지배")
        self.assertEqual(rows[2]["pareto_rank"], 1)
        self.assertEqual(rows[1]["pareto_rank"], 2)
        self.assertEqual(rows[1]["pareto_status"], "지배됨")
        self.assertEqual(rows[1]["pareto_dominated_by"], 1)

    def test_summary_counts_front(self):
        rows = compute_pareto([{name: 90.0 for name in PARETO_OBJECTIVES},
                               {name: 50.0 for name in PARETO_OBJECTIVES}])
        summary = pareto_summary(rows)
        self.assertEqual(summary["front_size"], 1)
        self.assertEqual(summary["candidate_count"], 2)
        # Kept deliberately small (2-4 axes): more axes push nearly every
        # candidate onto the front and the check stops discriminating.
        self.assertEqual(len(summary["objectives"]), len(PARETO_OBJECTIVES))
        self.assertLessEqual(len(summary["objectives"]), 4)
        self.assertGreaterEqual(len(summary["objectives"]), 2)

    def test_empty_input(self):
        self.assertEqual(compute_pareto([]), [])
        self.assertEqual(pareto_summary([])["front_size"], 0)


class ParetoSensitivityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_analysis_pipeline(sample_workbook())
        cls.recommendations = cls.result.recommendations

    def test_recommendations_carry_pareto_fields(self):
        for rec in self.recommendations:
            self.assertIn(rec.get("pareto_status"), {"비지배", "지배됨"})
            self.assertGreaterEqual(int(rec.get("pareto_rank")), 1)
        # At least one non-dominated candidate must exist.
        self.assertTrue(any(rec.get("pareto_rank") == 1 for rec in self.recommendations))

    def test_pipeline_exposes_pareto_and_sensitivity(self):
        as_dict = self.result.to_dict()
        self.assertIn("pareto_analysis", as_dict)
        self.assertIn("weight_sensitivity_analysis", as_dict)
        self.assertGreaterEqual(as_dict["pareto_analysis"].get("candidate_count", 0), 1)
        self.assertGreater(as_dict["weight_sensitivity_analysis"].get("scenarios", 0), 0)

    def test_weight_sensitivity_shape(self):
        weights = self.result.vhs_analysis["weights"]
        sensitivity = weight_sensitivity(self.recommendations, weights)
        self.assertGreater(sensitivity["scenarios"], 0)
        self.assertGreaterEqual(sensitivity["top1_retention_rate"], 0.0)
        self.assertLessEqual(sensitivity["top1_retention_rate"], 1.0)
        self.assertGreaterEqual(sensitivity["rank_volatility"], 0.0)
        self.assertTrue(sensitivity["rows"])
        for row in sensitivity["rows"]:
            self.assertIn(row["Top1 유지"], {"유지", "부분", "변동"})

    def test_weight_sensitivity_handles_single_candidate(self):
        sensitivity = weight_sensitivity(self.recommendations[:1], self.result.vhs_analysis["weights"])
        self.assertEqual(sensitivity["scenarios"], 0)
        self.assertEqual(sensitivity["top1_retention_rate"], 1.0)


class DqnResultFieldTests(unittest.TestCase):
    def _small_recs(self):
        return recommendations_frame().head(2).to_dict("records")

    def test_result_carries_extended_fields_without_torch_training(self):
        # 2 candidates -> insufficient path, returns immediately (no torch, no files).
        result = train_dqn(
            self._small_recs(), data_signature="sig-1",
            sample_id="dqn_01", store_count=4, dc_count=1, sample_name="syn.xlsx",
        )
        payload = result.to_dict()
        for field_name in (
            "data_signature", "sample_id", "sample_name", "store_count", "dc_count",
            "episodes", "learning_rate", "created_at", "action_distribution",
            "reward_history", "loss_history", "final_status", "stability_status",
            "candidate_count", "dqn_action_by_route",
            "dqn_confidence_by_route", "dqn_reference_by_route",
        ):
            self.assertIn(field_name, payload)
        self.assertEqual(payload["sample_id"], "dqn_01")
        self.assertEqual(payload["sample_name"], "syn.xlsx")
        self.assertEqual(payload["stability_status"], payload["status"])
        self.assertTrue(payload["created_at"])
        self.assertEqual(payload["store_count"], 4)
        self.assertEqual(payload["dc_count"], 1)
        self.assertEqual(payload["final_status"], payload["status"])
        self.assertEqual(payload["data_signature"], "sig-1")
        self.assertIsInstance(payload["reward_history"], list)
        self.assertIsInstance(payload["loss_history"], list)


if __name__ == "__main__":
    unittest.main()
