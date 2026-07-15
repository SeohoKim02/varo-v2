"""Tests for the on-demand detailed sensitivity engine."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import pandas as pd

from services.analysis_pipeline import run_analysis_pipeline
from services.sensitivity_service import (
    DETAIL_EXPORT_HEADERS,
    SUMMARY_EXPORT_HEADERS,
    apply_sensitivity_perturbation,
    build_sensitivity_cache_key,
    build_sensitivity_settings,
    clear_sensitivity_cache,
    generate_sensitivity_steps,
    run_detailed_sensitivity,
    sensitivity_detail_frame,
    sensitivity_summary_frame,
)
from tests.fixtures import sample_workbook


class DetailedSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_workbook()
        cls.pipeline = run_analysis_pipeline(cls.data).to_dict()
        cls.recommendations = cls.pipeline["recommendations"]
        cls.weights = cls.pipeline["vhs_analysis"]["weights"]

    def setUp(self):
        clear_sensitivity_cache()
        self.settings = build_sensitivity_settings(
            self.recommendations, self.data, self.weights,
            variables=["transport_cost", "distance", "demand", "disposal_risk", "vhs_weight"],
        )

    def _run(self, signature: str = "fixture-signature"):
        return run_detailed_sensitivity(
            self.recommendations, self.data, self.weights, self.settings, signature,
        )

    def test_steps_always_include_zero(self):
        for low, high, count in ((-5, 5, 4), (1, 20, 3), (-20, -1, 9), (10, -10, 2)):
            with self.subTest(low=low, high=high, count=count):
                self.assertIn(0.0, generate_sensitivity_steps(low, high, count))

    def test_zero_scenarios_exactly_match_current_vhs_and_rank(self):
        result = self._run()
        zero_rows = [row for row in result["detail_rows"] if row["change_pct"] == 0]
        self.assertTrue(zero_rows)
        self.assertTrue(all(row["changed_vhs"] == row["base_vhs"] for row in zero_rows))
        self.assertTrue(all(row["changed_rank"] == row["base_rank"] for row in zero_rows))
        baseline = {str(item["route_id"]): item for item in self.recommendations}
        for row in zero_rows:
            self.assertEqual(row["changed_vhs"], float(baseline[row["route_id"]]["vhs_score"]))
            self.assertEqual(row["changed_rank"], int(baseline[row["route_id"]]["vhs_rank"]))

    def test_source_data_and_recommendations_are_not_modified(self):
        recommendations_before = copy.deepcopy(self.recommendations)
        frames_before = {key: value.copy(deep=True) for key, value in self.data.items()}
        self._run()
        self.assertEqual(self.recommendations, recommendations_before)
        for key, frame in self.data.items():
            pd.testing.assert_frame_equal(frame, frames_before[key])

    def test_weight_scenarios_sum_to_one_and_do_not_activate_dqn(self):
        result = self._run()
        weight_rows = result["weight_rows"]
        self.assertTrue(weight_rows)
        self.assertTrue(all(abs(float(row["weight_sum"]) - 1.0) < 1e-9 for row in weight_rows))
        self.assertNotIn("dqn_reference_score", {row["weight_component"] for row in weight_rows})

    def test_scores_and_stability_are_bounded(self):
        result = self._run()
        self.assertTrue(all(0 <= float(row["changed_vhs"]) <= 100 for row in result["detail_rows"]))
        self.assertGreaterEqual(float(result["summary"]["score"]), 0)
        self.assertLessEqual(float(result["summary"]["score"]), 100)

    def test_demand_and_quantity_changes_never_become_negative(self):
        from services.sensitivity_service import _inventory_context, _prepare_candidates

        frame = _prepare_candidates(self.recommendations, len(self.recommendations))
        contexts = _inventory_context(frame.to_dict("records"), self.data)
        _, _, demand_meta = apply_sensitivity_perturbation(
            frame, contexts, "demand", -100, self.weights,
        )
        _, _, quantity_meta = apply_sensitivity_perturbation(
            frame, contexts, "quantity", -100, self.weights,
        )
        self.assertGreaterEqual(demand_meta["minimum_changed_demand"], 0)
        self.assertGreaterEqual(demand_meta["minimum_changed_quantity"], 0)
        self.assertGreaterEqual(quantity_meta["minimum_changed_quantity"], 0)

    def test_cache_key_is_stable_and_changes_with_signature(self):
        first = build_sensitivity_cache_key("A", self.settings, self.weights, self.recommendations)
        second = build_sensitivity_cache_key("A", copy.deepcopy(self.settings), self.weights, self.recommendations)
        changed = build_sensitivity_cache_key("B", self.settings, self.weights, self.recommendations)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_identical_execution_uses_cache_and_returns_copy(self):
        first = self._run()
        second = self._run()
        self.assertFalse(first["metadata"]["cache_hit"])
        self.assertTrue(second["metadata"]["cache_hit"])
        second["detail_rows"][0]["changed_vhs"] = -999
        third = self._run()
        self.assertNotEqual(third["detail_rows"][0]["changed_vhs"], -999)

    def test_detailed_analysis_never_loads_excel_or_trains_dqn(self):
        with (
            patch("services.data_loader.load_excel_data") as loader,
            patch("services.dqn_service.train_dqn") as trainer,
        ):
            self._run()
        loader.assert_not_called()
        trainer.assert_not_called()

    def test_invalid_numeric_scenario_does_not_stop_whole_analysis(self):
        recommendations = copy.deepcopy(self.recommendations)
        recommendations[0]["estimated_cost"] = float("inf")
        settings = build_sensitivity_settings(
            recommendations, self.data, self.weights,
            variables=["transport_cost", "disposal_risk"],
        )
        result = run_detailed_sensitivity(
            recommendations, self.data, self.weights, settings, "invalid-value-signature",
        )
        self.assertTrue(result["scenario_rows"])
        self.assertGreater(result["summary"]["completed_scenario_count"], 0)
        self.assertGreaterEqual(result["summary"]["excluded_scenario_count"], 1)

    def test_download_frames_have_required_columns(self):
        result = self._run()
        detail = sensitivity_detail_frame(result)
        summary = sensitivity_summary_frame(result)
        self.assertTrue(set(DETAIL_EXPORT_HEADERS.values()).issubset(detail.columns))
        self.assertTrue(set(SUMMARY_EXPORT_HEADERS.values()).issubset(summary.columns))

    def test_empty_input_returns_render_safe_result(self):
        settings = build_sensitivity_settings([], {}, {}, variables=[])
        result = run_detailed_sensitivity([], {}, {}, settings, "empty")
        self.assertEqual(result["detail_rows"], [])
        self.assertEqual(result["scenario_rows"], [])
        self.assertEqual(result["summary"], {})


if __name__ == "__main__":
    unittest.main()
