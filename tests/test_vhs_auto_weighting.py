"""Tests for Varo V2 automatic VHS weighting and strategy comparison."""
from __future__ import annotations

import unittest

import pandas as pd

from services.analysis_pipeline import run_analysis_pipeline, sort_recommendations
from services.data_loader import load_excel_data
from services.sample_catalog import SAMPLE_WORKBOOKS, sample_path
from services.vhs_score_engine import COMPONENTS, WEIGHT_BOUNDS, apply_auto_vhs, build_strategy_comparison
from tests.fixtures import recommendations_frame, sample_workbook


class VhsAutoWeightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_analysis_pipeline(sample_workbook())
        cls.recommendations = cls.result.recommendations
        cls.weights = cls.result.vhs_analysis["weights"]

    def test_weight_sum_is_one(self):
        self.assertAlmostEqual(sum(self.weights.values()), 1.0, places=5)

    def test_weights_stay_inside_component_bounds(self):
        for component, weight in self.weights.items():
            low, high = WEIGHT_BOUNDS[component]
            self.assertGreaterEqual(weight, low - 1e-6, component)
            self.assertLessEqual(weight, high + 1e-6, component)

    def test_missing_columns_do_not_break_vhs_scoring(self):
        minimal = recommendations_frame()[[
            "route_id", "product_id", "product_name", "source_id", "source_name",
            "target_id", "target_name", "dc_id", "dc_name", "route_type",
            "recommended_qty", "estimated_cost", "expected_saving",
        ]].copy()
        result = apply_auto_vhs(minimal)
        self.assertEqual(len(result.frame), len(minimal))
        self.assertTrue(result.frame["vhs_score"].between(0, 100).all())
        self.assertAlmostEqual(sum(result.analysis["weights"].values()), 1.0, places=5)

    def test_candidate_scores_are_in_0_to_100_range(self):
        for rec in self.recommendations:
            self.assertGreaterEqual(float(rec["vhs_score"]), 0.0)
            self.assertLessEqual(float(rec["vhs_score"]), 100.0)
            for component in COMPONENTS:
                if component in rec and rec[component] is not None:
                    self.assertGreaterEqual(float(rec[component]), 0.0)
                    self.assertLessEqual(float(rec[component]), 100.0)

    def test_component_metadata_records_missing_and_imputation_policy(self):
        rows = self.result.vhs_analysis.get("weight_rows") or []
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("missing_rate", row)
            self.assertIn("imputation_strategy", row)
            self.assertGreaterEqual(float(row["missing_rate"]), 0.0)
            self.assertLessEqual(float(row["missing_rate"]), 1.0)

    def test_vhs_rank_one_is_varo_final_recommendation(self):
        top = sort_recommendations(self.recommendations)[0]
        self.assertEqual(int(top["vhs_rank"]), 1)
        self.assertEqual(int(top["varo_final_rank"]), 1)
        self.assertEqual(top["varo_final_decision"], "최종 추천")

    def test_greedy_comparison_columns_are_present(self):
        comparison = build_strategy_comparison(self.recommendations)
        self.assertTrue(comparison)
        required = {
            "상품명", "보내는 점포", "받는 점포", "추천 수량", "VHS 순위",
            "VHS 점수", "Greedy 순위", "Greedy 전략", "DQN 상태",
            "DQN 전략", "DQN 참고 점수", "Varo 최종 추천", "일치 여부", "판단 근거",
        }
        self.assertTrue(required.issubset(comparison[0]))

    def test_pareto_fields_are_attached_as_auxiliary_validation(self):
        self.assertTrue(all(rec.get("pareto_rank") is not None for rec in self.recommendations))
        self.assertTrue(all(rec.get("pareto_status") for rec in self.recommendations))
        self.assertTrue(all(rec.get("pareto_reason") for rec in self.recommendations))
        self.assertEqual(
            self.result.pareto_analysis["criteria"],
            ["절감액", "폐기 위험", "수요 적합도", "경로 비용", "실행 가능성"],
        )

    def test_dqn_unconnected_does_not_influence_vhs(self):
        self.assertTrue(all(float(rec.get("dqn_reference_score") or 0) == 0 for rec in self.recommendations))
        self.assertEqual(float(self.weights.get("dqn_reference_score", 0)), 0.0)
        self.assertTrue(all(rec.get("dqn_action") for rec in self.recommendations))

    def test_normal_dqn_result_can_receive_low_reference_weight(self):
        frame = recommendations_frame().copy()
        frame["dqn_status"] = "정상"
        frame["dqn_action"] = "재고 이동"
        frame["dqn_confidence"] = 85.0
        training = {"status": "정상"}
        result = apply_auto_vhs(frame, training)
        weight = result.analysis["weights"]["dqn_reference_score"]
        self.assertGreater(weight, 0.0)
        self.assertLessEqual(weight, WEIGHT_BOUNDS["dqn_reference_score"][1])
        self.assertTrue(result.frame["vhs_score"].between(0, 100).all())

    def test_pipeline_exposes_weight_and_comparison_sections(self):
        as_dict = self.result.to_dict()
        self.assertIn("vhs_weight_analysis", as_dict)
        self.assertIn("vhs_greedy_dqn_comparison", as_dict)
        self.assertEqual(
            as_dict["vhs_weight_analysis"]["weight_profile_id"],
            "auto_distribution_v1",
        )
        self.assertEqual(len(as_dict["vhs_greedy_dqn_comparison"]), len(self.recommendations))
        validation = as_dict["validation_report"]
        self.assertEqual(len(validation["vhs_component_quality"]), len(COMPONENTS))
        self.assertEqual(validation["pareto_validation"]["comparison_count"], len(self.recommendations))

    def test_all_simulation_samples_can_compute_auto_vhs(self):
        for sample in SAMPLE_WORKBOOKS:
            with self.subTest(sample=sample.key):
                data = load_excel_data(sample_path(sample))
                result = run_analysis_pipeline(data)
                self.assertTrue(result.recommendations)
                self.assertAlmostEqual(sum(result.vhs_analysis["weights"].values()), 1.0, places=5)
                self.assertTrue(all(0 <= float(rec["vhs_score"]) <= 100 for rec in result.recommendations))

    def test_dual_dc_sample_keeps_via_dc_rows_on_both_dcs(self):
        sample = next(item for item in SAMPLE_WORKBOOKS if item.key == "dual_dc_10stores_2dc")
        data = load_excel_data(sample_path(sample))
        result = run_analysis_pipeline(data)
        via_dcs = {
            rec.get("dc_id")
            for rec in result.recommendations
            if rec.get("route_type") == "VIA_DC"
        }
        self.assertIn("DC01", via_dcs)
        self.assertIn("DC02", via_dcs)


if __name__ == "__main__":
    unittest.main()
