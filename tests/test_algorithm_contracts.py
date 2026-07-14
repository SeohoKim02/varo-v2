"""Input-column contracts and graceful-degradation tests for _local_modules.

Documents the minimum input columns each connected algorithm needs and verifies
that the in-project fixture provides them (so the recompute connects), that
missing columns degrade gracefully (deferred/neutral, never a crash), and that
the pipeline works without any DQN column present.
"""
from __future__ import annotations

import unittest

from services.analysis_pipeline import run_analysis_pipeline
from services.data_validator import validate_workbook_data
from tests.fixtures import sample_workbook

# Minimum input columns each algorithm needs from its primary frame.
INVENTORY_CONTRACTS = {
    "abc_analyzer.analyze_abc": ("product_id", "stock_qty", "unit_price"),
    "turnover_analyzer.analyze_turnover": ("stock_qty", "sales_30d"),
    "disposal_risk_analyzer.analyze_disposal_risk": ("stock_qty", "days_to_expiry"),
    "demand_forecast_analyzer.analyze_demand_forecast": ("sales_7d", "sales_30d"),
    "safety_stock_analyzer.analyze_safety_stock": ("demand_std", "lead_time_days"),
    "eoq_analyzer.analyze_eoq": ("avg_daily_sales", "order_cost"),
    "store_product_matcher.analyze_store_product_matching": ("store_id", "product_id", "stock_qty"),
    "store_clustering.analyze_store_clustering": ("store_name", "avg_daily_sales"),
    "min_cost_network.analyze_min_cost_network": ("store_name", "stock_qty", "dead_stock_qty"),
}
RECOMMENDATION_CONTRACT = (
    "route_id", "product_id", "product_name", "source_id", "target_id",
    "route_type", "recommended_qty", "estimated_cost", "expected_saving",
    "vhs_score", "confidence_score",
)
CORE_CONNECTED = (
    "varo_hybrid_score.calculate_varo_hybrid_score",
    "heuristic_optimizer.add_heuristic_scores",
    "transfer_path_analyzer.analyze_direct_vs_dc_transfer",
    "promotion_analyzer.analyze_promotion_vs_transfer",
    "varo_optimality_gap.calculate_optimality_gap",
)
DQN_FORBIDDEN_TOKENS = ("reward", "loss", "q_table", "qtable", "policy_table", "replay", "model_path")


class AlgorithmContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_workbook()
        cls.result = run_analysis_pipeline(cls.data)

    def test_fixture_inventory_satisfies_contracts(self):
        columns = set(self.data["inventory"].columns)
        for algorithm, required in INVENTORY_CONTRACTS.items():
            missing = [c for c in required if c not in columns]
            self.assertEqual(missing, [], f"{algorithm} missing inventory columns: {missing}")

    def test_fixture_recommendations_satisfy_contract(self):
        columns = set(self.data["recommendations"].columns)
        missing = [c for c in RECOMMENDATION_CONTRACT if c not in columns]
        self.assertEqual(missing, [], f"recommendations missing columns: {missing}")

    def test_fixture_connects_core_algorithms(self):
        self.assertEqual(self.result.status, "success")
        self.assertEqual(self.result.diagnostics.get("algorithm_errors"), [])
        connected = set(self.result.connected_algorithms)
        for algorithm in CORE_CONNECTED:
            self.assertIn(algorithm, connected)

    def test_missing_optional_inventory_column_degrades_gracefully(self):
        # order_cost is used by EOQ but is NOT validation-required, so dropping it
        # must defer EOQ (not crash) and keep recommendations intact.
        workbook = {k: v.copy() for k, v in self.data.items()}
        workbook["inventory"] = workbook["inventory"].drop(columns=["order_cost"])
        self.assertFalse(validate_workbook_data(workbook).has_errors)
        result = run_analysis_pipeline(workbook)
        self.assertEqual(len(result.recommendations), 4)
        self.assertIn(result.status, ("partial", "success"))
        self.assertTrue(all(item["dqn_action"] == "미연결" for item in result.recommendations))

    def test_missing_required_column_returns_validation_error_without_crashing(self):
        # stock_qty (inventory) and vhs_score (recommendations) are required, so the
        # pipeline must return a graceful validation_error rather than raising.
        for frame_name, column in (("inventory", "stock_qty"), ("recommendations", "vhs_score")):
            workbook = {k: v.copy() for k, v in self.data.items()}
            workbook[frame_name] = workbook[frame_name].drop(columns=[column])
            self.assertTrue(validate_workbook_data(workbook).has_errors)
            result = run_analysis_pipeline(workbook)  # must not raise
            self.assertEqual(result.status, "validation_error")

    def test_pipeline_works_without_dqn_columns(self):
        for frame_name in ("recommendations", "inventory"):
            columns = [str(c).lower() for c in self.data[frame_name].columns]
            for token in DQN_FORBIDDEN_TOKENS:
                self.assertFalse(
                    any(token in c for c in columns),
                    f"fixture {frame_name} unexpectedly contains a {token} column",
                )
        self.assertEqual(self.result.status, "success")
        self.assertTrue(all(item["dqn_action"] == "미연결" for item in self.result.recommendations))
        self.assertTrue(all(item["dqn_correction"] == 0 for item in self.result.recommendations))

    def test_neutral_components_recorded_when_inputs_missing(self):
        # Some VHS component columns are not in the fixture, so the provenance must
        # record them as defaulted (neutral) rather than failing.
        vhs = self.result.vhs_analysis
        self.assertGreater(len(vhs.get("defaulted_component_columns") or []), 0)
        self.assertFalse(vhs.get("dqn_correction_used"))


if __name__ == "__main__":
    unittest.main()
