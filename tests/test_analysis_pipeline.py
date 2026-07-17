"""Tests for the self-contained analysis pipeline recompute and DQN exclusion.

The approved non-DQN algorithms now live in _local_modules, so the pipeline
recomputes VHS/Greedy/route from the in-project sample without any original-
folder or backup access. Optimality is button-only and DQN stays fully excluded.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from services import analysis_pipeline
from services.analysis_pipeline import find_recommendation, run_analysis_pipeline
from services.data_loader import get_default_sample_path, load_excel_data
from services.dqn_guard import is_blocked_dqn_path, strip_dqn_columns
from services.legacy_adapters.loader import (
    LegacyAlgorithmUnavailable,
    available_legacy_algorithms,
    get_legacy_root,
    load_legacy_module,
)

CORE_MODULES = (
    "varo_hybrid_score", "heuristic_optimizer", "vhs_confidence", "varo_optimality_gap",
    "varo_validation", "transfer_path_analyzer", "promotion_analyzer", "abc_analyzer",
    "turnover_analyzer", "disposal_risk_analyzer", "safety_stock_analyzer", "eoq_analyzer",
    "store_product_matcher", "store_clustering", "route_analyzer", "cutline_analyzer",
    "time_window_analyzer", "min_cost_network", "network_path_analyzer", "demand_forecast_analyzer",
)


class SelfContainedRecomputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_excel_data(get_default_sample_path())  # in-project sample, 8 recs
        cls.result = run_analysis_pipeline(cls.data)

    # --- self-containment / no external access -----------------------------
    def test_legacy_root_is_internal_local_modules(self):
        root = str(get_legacy_root()).replace("\\", "/").lower()
        self.assertNotIn("bad_inventory_simulator", root)
        self.assertIn("varo_v2", root)
        self.assertIn("_local_modules", root)

    def test_forbidden_legacy_override_is_ignored(self):
        with patch.dict(os.environ, {"VARO_LEGACY_PATH": r"C:\Users\82102\Desktop\bad_inventory_simulator_backup"}):
            root = str(get_legacy_root()).replace("\\", "/").lower()
        self.assertNotIn("bad_inventory_simulator", root)

    def test_ported_non_dqn_modules_are_available(self):
        avail = available_legacy_algorithms()
        for mod in CORE_MODULES:
            self.assertTrue(avail.get(mod), f"{mod} should be importable from _local_modules")

    def test_dqn_module_import_is_blocked(self):
        for blocked in ("dqn_agent", "train_rl_agent", "replay_buffer", "rl_policy_helper"):
            with self.assertRaises(LegacyAlgorithmUnavailable):
                load_legacy_module(blocked)

    def test_dqn_artifact_paths_are_blocked(self):
        for name in (
            "rl_training_log.csv", "rl_training_summary.json", "rl_q_table.csv",
            "rl_policy_table.csv", "model.pth", "reward_history.csv", "loss_history.csv",
        ):
            self.assertTrue(is_blocked_dqn_path(name))

    # --- real recompute ----------------------------------------------------
    def test_pipeline_returns_required_structure(self):
        result = self.result.to_dict()
        for key in (
            "status", "summary", "recommendations", "top5", "vhs_analysis",
            "greedy_analysis", "route_analysis", "promotion_analysis",
            "demand_analysis", "risk_analysis", "validation_report", "warnings",
            "connected_algorithms", "deferred_algorithms", "excluded_dqn_artifacts",
        ):
            self.assertIn(key, result)
        self.assertEqual(len(result["recommendations"]), 8)

    def test_status_success_with_only_design_defers(self):
        self.assertEqual(self.result.status, "success")
        self.assertEqual(self.result.result_basis, "실제 V2 내부 알고리즘 재계산 결과 기준")
        self.assertEqual(self.result.diagnostics.get("algorithm_errors"), [])
        deferred = {d["algorithm"].split(".")[0] for d in self.result.deferred_algorithms}
        self.assertTrue(deferred.issubset({"varo_sensitivity", "vhs_reason"}))

    def test_core_algorithms_are_connected(self):
        connected = set(self.result.connected_algorithms)
        self.assertIn("varo_hybrid_score.calculate_varo_hybrid_score", connected)
        self.assertIn("heuristic_optimizer.add_heuristic_scores", connected)
        self.assertIn("transfer_path_analyzer.analyze_direct_vs_dc_transfer", connected)
        self.assertIn("promotion_analyzer.analyze_promotion_vs_transfer", connected)

    def test_vhs_is_recomputed_not_uploaded(self):
        avg = self.result.summary["average_vhs_score"]
        self.assertIsNotNone(avg)
        # uploaded average is 83.425; the automatic recompute must differ.
        self.assertNotAlmostEqual(avg, 83.425, places=1)
        self.assertGreaterEqual(avg, 0)
        self.assertLessEqual(avg, 100)
        self.assertEqual(self.result.vhs_analysis["score_basis"], "VHS 자동 가중치 최적화")
        self.assertTrue(bool(self.result.vhs_analysis))

    def test_optimality_is_button_only_and_confidence_is_real(self):
        gap = self.result.validation_report["optimality_gap"]
        self.assertEqual(gap["status"], "지연 실행")
        self.assertIn("버튼", gap["message"])
        self.assertEqual(self.result.confidence_analysis["average"], 66.0)

    def test_selected_route_lookup_and_fallback(self):
        r002 = find_recommendation(self.result.recommendations, "R002")
        self.assertEqual(r002["route_type"], "VIA_DC")
        self.assertEqual(r002["dc_id"], "DC01")
        self.assertIsNotNone(find_recommendation(self.result.recommendations, "UNKNOWN"))

    # --- DQN exclusion -----------------------------------------------------
    def test_dqn_excluded_from_results(self):
        self.assertTrue(all(item["dqn_action"] == "미연결" for item in self.result.recommendations))
        self.assertTrue(all(item["dqn_correction"] == 0 for item in self.result.recommendations))
        self.assertFalse(self.result.excluded_dqn_artifacts["artifacts_read"])
        self.assertFalse(self.result.diagnostics.get("dqn_artifacts_read"))

    def test_injected_dqn_columns_are_stripped(self):
        workbook = dict(self.data)
        workbook["recommendations"] = self.data["recommendations"].assign(
            dqn_action="과거값", reward=999999, loss=1.0,
        )
        result = run_analysis_pipeline(workbook)
        removed = result.confidence_analysis["removed_dqn_columns"]
        for column in ("dqn_action", "reward", "loss"):
            self.assertIn(column, removed)
        self.assertTrue(all(item["dqn_action"] == "미연결" for item in result.recommendations))
        for rec in result.recommendations:
            for key in rec:
                self.assertFalse(
                    any(t in str(key).lower() for t in ("reward", "loss", "q_table", "policy_table", "replay"))
                )

    def test_reward_loss_columns_stripped_by_guard(self):
        guarded = strip_dqn_columns(self.data["recommendations"].assign(reward=999999, loss=1))
        self.assertNotIn("reward", guarded.columns)
        self.assertNotIn("loss", guarded.columns)

    def test_algorithm_failure_is_isolated(self):
        original = analysis_pipeline.load_legacy_module

        def selective_loader(name):
            if name == "abc_analyzer":
                raise RuntimeError("forced test failure")
            return original(name)

        with patch.object(analysis_pipeline, "load_legacy_module", side_effect=selective_loader):
            result = run_analysis_pipeline(self.data)
        self.assertEqual(len(result.recommendations), 8)
        self.assertTrue(any("abc_analyzer" in item["algorithm"] for item in result.deferred_algorithms))
        self.assertEqual(result.status, "partial")  # a real failure downgrades to partial

if __name__ == "__main__":
    unittest.main()
