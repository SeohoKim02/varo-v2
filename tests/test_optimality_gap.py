"""Contract tests for the explicit, in-memory optimality-gap analysis."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import pandas as pd
from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

from streamlit.testing.v1 import AppTest

from services.app_state import apply_state_payload
from services.optimality_gap_service import (
    build_constraint_context,
    build_optimality_settings,
    calculate_gap_metrics,
    clear_optimality_cache,
    optimality_constraints_frame,
    optimality_routes_frame,
    optimality_summary_frame,
    prepare_optimality_problem,
    run_optimality_gap,
    validate_selection,
)


def _candidate(
    route_id: str,
    target: str,
    qty: float,
    saving: float,
    vhs_rank: int,
    greedy_rank: int,
    **updates,
) -> dict:
    row = {
        "recommendation_id": f"REC-{route_id}", "route_id": route_id,
        "product_id": "P1", "product_name": "상품1", "source_id": "S1",
        "target_id": target, "route_type": "DIRECT", "dc_id": None,
        "recommended_qty": qty, "expected_saving": saving,
        "vhs_rank": vhs_rank, "greedy_rank": greedy_rank,
        "feasible": True,
    }
    row.update(updates)
    return row


def _known_problem() -> tuple[list[dict], dict]:
    # VHS takes A first and then cannot add B/C under stock 10. Greedy and the
    # optimum take B+C for 160, so the known Varo gap is 37.5%.
    candidates = [
        _candidate("A", "T1", 7, 100, 1, 3),
        _candidate("B", "T2", 5, 80, 2, 1),
        _candidate("C", "T3", 5, 80, 3, 2),
        _candidate("B-DUP", "T2", 5, 79, 4, 4),
    ]
    inventory = pd.DataFrame([
        {"store_id": "S1", "product_id": "P1", "stock_qty": 10, "sales_7d": 0},
        {"store_id": "T1", "product_id": "P1", "stock_qty": 0, "sales_7d": 10},
        {"store_id": "T2", "product_id": "P1", "stock_qty": 0, "sales_7d": 10},
        {"store_id": "T3", "product_id": "P1", "stock_qty": 0, "sales_7d": 10},
    ])
    routes = pd.DataFrame([
        {"route_id": route_id, "source_id": "S1", "target_id": target, "route_type": "DIRECT", "feasible": True}
        for route_id, target in (("A", "T1"), ("B", "T2"), ("C", "T3"), ("B-DUP", "T2"))
    ])
    return candidates, {"inventory": inventory, "routes": routes, "stores": pd.DataFrame()}


class OptimalityGapServiceTests(unittest.TestCase):
    def setUp(self):
        clear_optimality_cache()
        self.recommendations, self.data = _known_problem()
        self.settings = build_optimality_settings(candidate_limit=20, max_routes=2, search_mode="auto", time_limit=3)

    def _run(self, **settings):
        configured = build_optimality_settings(**({
            "candidate_limit": 20, "max_routes": 2, "search_mode": "auto", "time_limit": 3,
        } | settings))
        return run_optimality_gap(self.recommendations, self.data, configured, "known-signature")

    def test_known_exact_optimum_and_gap_formula(self):
        result = self._run()
        self.assertTrue(result["search"]["optimal"])
        self.assertEqual(result["combinations"]["varo"]["total_saving"], 100)
        self.assertEqual(result["combinations"]["greedy"]["total_saving"], 160)
        self.assertEqual(result["combinations"]["best"]["total_saving"], 160)
        self.assertAlmostEqual(result["gap"]["gap_pct"], 37.5)
        self.assertAlmostEqual(result["gap"]["target_pct"], 62.5)

    def test_stock_capacity_is_shared_by_all_strategies(self):
        result = self._run()
        for strategy in ("varo", "greedy", "best"):
            combination = result["combinations"][strategy]
            qty = sum(
                self.recommendations[index]["recommended_qty"]
                for index in combination["selected_indices"]
            )
            self.assertLessEqual(qty, 10)

    def test_target_demand_limit_blocks_aggregate_overflow(self):
        recommendations = [
            _candidate("X", "T1", 6, 100, 1, 1),
            _candidate("Y", "T1", 5, 90, 2, 2, product_id="P1", route_type="VIA_DC", dc_id="DC01"),
        ]
        data = {"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 30},
            {"store_id": "T1", "product_id": "P1", "stock_qty": 2, "sales_7d": 10},
        ])}
        result = run_optimality_gap(recommendations, data, self.settings, "demand")
        self.assertEqual(result["combinations"]["best"]["count"], 1)
        self.assertLessEqual(result["combinations"]["best"]["total_qty"], 8)

    def test_max_routes_constraint(self):
        result = self._run(max_routes=1)
        self.assertEqual(result["combinations"]["best"]["count"], 1)
        self.assertEqual(result["combinations"]["best"]["total_saving"], 100)

    def test_infeasible_same_store_and_nonpositive_candidates_are_excluded(self):
        recs = [
            _candidate("BAD-FLAG", "T1", 1, 10, 1, 1, feasible=False),
            _candidate("BAD-SAME", "S1", 1, 10, 2, 2),
            _candidate("BAD-QTY", "T2", 0, 10, 3, 3),
            _candidate("BAD-SAVE", "T3", 1, -1, 4, 4),
        ]
        result = run_optimality_gap(recs, {}, self.settings, "excluded")
        self.assertEqual(result["summary"]["feasible_candidate_count"], 0)
        self.assertEqual(len(result["excluded_rows"]), 4)
        reasons = " ".join(row["reason"] for row in result["excluded_rows"])
        self.assertIn("실행 가능성", reasons)
        self.assertIn("출발 점포", reasons)
        self.assertIn("수량", reasons)
        self.assertIn("절감액", reasons)

    def test_exact_duplicate_tuple_can_be_selected_once(self):
        recs = [
            _candidate("D1", "T1", 1, 20, 1, 1),
            _candidate("D2", "T1", 1, 19, 2, 2),
        ]
        result = run_optimality_gap(recs, {}, self.settings, "duplicate")
        self.assertEqual(result["combinations"]["best"]["count"], 1)
        problem = prepare_optimality_problem(recs, {}, self.settings)
        context = build_constraint_context(problem["candidates"], {}, self.settings)
        self.assertFalse(validate_selection([0, 1], problem["candidates"], context)[0])

    def test_zero_gap_and_negative_tolerance(self):
        exact = calculate_gap_metrics(100, 100, 100, exact=True)
        self.assertEqual(exact["gap_pct"], 0)
        tolerant = calculate_gap_metrics(100.00000000001, 90, 100, exact=True)
        self.assertEqual(tolerant["gap_pct"], 0)
        inconsistent = calculate_gap_metrics(101, 90, 100, exact=True)
        self.assertTrue(inconsistent["inconsistency"])
        self.assertLess(inconsistent["gap_pct"], 0)

    def test_greedy_uses_existing_rank_order(self):
        result = self._run()
        self.assertEqual(result["combinations"]["greedy"]["route_ids"], ["B", "C"])

    def test_forced_limited_mode_is_never_labeled_exact(self):
        result = self._run(search_mode="limited", time_limit=0.05)
        self.assertFalse(result["search"]["optimal"])
        self.assertEqual(result["search"]["status"], "제한 탐색")
        self.assertEqual(result["gap"]["label"], "참고 Gap")
        self.assertIn("certified_gap_range", result["gap"])

    def test_original_recommendations_and_data_are_immutable(self):
        before_recommendations = copy.deepcopy(self.recommendations)
        before_inventory = self.data["inventory"].copy(deep=True)
        self._run()
        self.assertEqual(self.recommendations, before_recommendations)
        pd.testing.assert_frame_equal(self.data["inventory"], before_inventory)

    def test_empty_problem_is_truthful(self):
        result = run_optimality_gap([], {}, self.settings, "empty")
        self.assertEqual(result["summary"]["feasible_candidate_count"], 0)
        self.assertFalse(result["gap"]["available"])
        self.assertEqual(result["combinations"]["best"]["total_saving"], 0)

    def test_cache_returns_deep_copy(self):
        first = self._run()
        first["summary"]["status"] = "변조"
        second = self._run()
        self.assertTrue(second["metadata"]["cache_hit"])
        self.assertNotEqual(second["summary"]["status"], "변조")
        changed = self._run(max_routes=1)
        self.assertNotEqual(second["metadata"]["cache_key"], changed["metadata"]["cache_key"])

    def test_gap_service_never_loads_excel_or_runs_dqn(self):
        with (
            patch("services.data_loader.load_excel_data") as excel_loader,
            patch("services.dqn_service.train_dqn") as dqn_trainer,
            patch("services.dqn_service.load_latest_dqn_result") as dqn_loader,
        ):
            self._run()
        excel_loader.assert_not_called()
        dqn_trainer.assert_not_called()
        dqn_loader.assert_not_called()

    def test_download_frames_are_in_memory_and_nonempty(self):
        result = self._run()
        for frame in (
            optimality_summary_frame(result),
            optimality_routes_frame(result),
            optimality_constraints_frame(result),
        ):
            self.assertFalse(frame.empty)
            self.assertTrue(frame.to_csv(index=False).encode("utf-8-sig"))

    def test_new_data_state_application_resets_gap_only_runtime_state(self):
        state = {
            "optimality_gap_settings": {"max_routes": 3},
            "optimality_gap_result": {"old": True},
            "optimality_gap_data_signature": "old",
            "optimality_gap_is_running": True,
            "optimality_gap_last_error": "old",
        }
        apply_state_payload(state, {})
        self.assertEqual(state["optimality_gap_settings"], {})
        self.assertIsNone(state["optimality_gap_result"])
        self.assertIsNone(state["optimality_gap_data_signature"])
        self.assertFalse(state["optimality_gap_is_running"])
        self.assertIsNone(state["optimality_gap_last_error"])


class OptimalityGapAppTests(unittest.TestCase):
    def test_button_only_render_and_result_panels(self):
        with patch("pages.validation.run_optimality_gap") as gap_runner:
            app = AppTest.from_file("app_v2.py", default_timeout=120)
            app.run()
            next(button for button in app.button if button.key == "quick_empty_sample").click().run()
            app.session_state["current_menu"] = "분석 및 검증"
            app.run()
            self.assertFalse(app.exception)
            self.assertIn("최적성 Gap", [tab.label for tab in app.tabs])
            self.assertIsNone(app.session_state["optimality_gap_result"])
            self.assertIsNone(app.session_state["dqn_training_result"])
            gap_runner.assert_not_called()
        execute = next(button for button in app.button if button.key == "run_optimality_gap")
        execute.click().run()
        self.assertFalse(app.exception)
        self.assertIsNotNone(app.session_state["optimality_gap_result"])
        labels = {metric.label for metric in app.metric}
        self.assertIn("Varo 조합 절감액", labels)
        self.assertIn("Greedy 조합 절감액", labels)
        self.assertIn("탐색 상태", labels)
        tab_labels = [tab.label for tab in app.tabs]
        for required in ("종합 비교", "선택 경로 비교", "제약조건", "탐색 정보"):
            self.assertIn(required, tab_labels)
        downloads = {button.label for button in app.get("download_button")}
        for required in ("Gap 요약 CSV", "선택 경로 CSV", "제약조건 CSV"):
            self.assertIn(required, downloads)
        self.assertIsNone(app.session_state["dqn_training_result"])

        app.session_state["current_menu"] = "홈"
        app.run()
        home_copy = " ".join(item.value for item in app.markdown)
        self.assertNotIn("최적성 Gap 계산", home_copy)


if __name__ == "__main__":
    unittest.main()
