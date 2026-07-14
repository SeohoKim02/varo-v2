"""Tests for canonical Varo V2 application state (self-contained, backup-free).

These cover the state machine and payload contract using the in-package fixture
and a synthetic pipeline result, so they stay deterministic regardless of which
algorithms are present in _local_modules.
"""
import unittest

from services.analysis_pipeline import calculate_overview_kpis, find_recommendation
from services.app_state import (
    TRANSIENT_VIEW_KEYS,
    apply_state_payload,
    build_applied_state_payload,
    current_data_status,
    default_selected_route_id,
    has_app_data,
    resolve_selected_route_id,
)
from services.data_validator import validate_workbook_data
from services.recommendation_adapter import recommendations_from_dataframe
from tests.fixtures import (
    DEFAULT_SELECTED_ROUTE_ID,
    EXPECTED_AVERAGE_VHS,
    EXPECTED_RECOMMENDATION_COUNT,
    EXPECTED_TOTAL_QTY,
    EXPECTED_TOTAL_SAVING,
    PERSISTED_ROUTE_ID,
    sample_workbook,
    synthetic_pipeline_result,
)

FIXTURE_FILENAME = "varo_v2_fixture.xlsx"
SHARED_KEYS = (
    "varo_data", "varo_validation", "varo_recommendations", "selected_route_id",
    "uploaded_filename", "data_source_type", "analysis_result", "pipeline_summary",
    "connected_algorithms", "deferred_algorithms", "dqn_excluded",
)


class AppStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_workbook()
        cls.validation = validate_workbook_data(cls.data)
        cls.recommendations = recommendations_from_dataframe(cls.data["recommendations"])
        cls.pipeline_result = synthetic_pipeline_result(cls.recommendations)

    def test_fixture_is_valid_and_complete(self):
        self.assertFalse(self.validation.has_errors)
        self.assertEqual(len(self.recommendations), EXPECTED_RECOMMENDATION_COUNT)

    def test_sample_builds_canonical_payload(self):
        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
            data_signature="fixture-signature",
        )
        self.assertTrue(payload["varo_data"])
        self.assertEqual(len(payload["varo_recommendations"]), EXPECTED_RECOMMENDATION_COUNT)
        self.assertEqual(payload["selected_route_id"], DEFAULT_SELECTED_ROUTE_ID)
        self.assertEqual(payload["data_source_type"], "샘플 추천 데이터")
        self.assertEqual(payload["data_signature"], "fixture-signature")

    def test_pipeline_payload_contains_required_shared_keys(self):
        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations,
            FIXTURE_FILENAME, "샘플 추천 데이터", self.pipeline_result,
        )
        for key in SHARED_KEYS:
            self.assertIn(key, payload)
        self.assertEqual(payload["pipeline_summary"]["active_route_count"], EXPECTED_RECOMMENDATION_COUNT)
        self.assertFalse(payload["dqn_excluded"]["artifacts_read"])

    def test_has_app_data_requires_stores_and_recommendations(self):
        self.assertTrue(has_app_data(self.data, self.recommendations))
        self.assertFalse(has_app_data(self.data, []))
        self.assertFalse(has_app_data({}, self.recommendations))

    def test_overview_model_uses_fixture_values(self):
        kpis = calculate_overview_kpis(self.recommendations, self.validation)
        self.assertEqual(kpis["total_recommended_qty"], EXPECTED_TOTAL_QTY)
        self.assertEqual(kpis["active_route_count"], EXPECTED_RECOMMENDATION_COUNT)
        self.assertEqual(kpis["total_expected_saving"], EXPECTED_TOTAL_SAVING)
        self.assertAlmostEqual(kpis["average_vhs_score"], EXPECTED_AVERAGE_VHS)

    def test_default_selection_uses_first_top_route(self):
        self.assertEqual(default_selected_route_id(self.recommendations), DEFAULT_SELECTED_ROUTE_ID)
        self.assertEqual(resolve_selected_route_id(self.recommendations, None), DEFAULT_SELECTED_ROUTE_ID)
        self.assertEqual(resolve_selected_route_id(self.recommendations, PERSISTED_ROUTE_ID), PERSISTED_ROUTE_ID)
        self.assertEqual(resolve_selected_route_id(self.recommendations, "OLD"), DEFAULT_SELECTED_ROUTE_ID)
        self.assertEqual(find_recommendation(self.recommendations, "OLD")["route_id"], DEFAULT_SELECTED_ROUTE_ID)

    def test_page_navigation_keeps_analysis_and_r002_selection(self):
        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations,
            FIXTURE_FILENAME, "샘플 추천 데이터", self.pipeline_result,
        )
        state = {}
        apply_state_payload(state, payload)
        state["selected_route_id"] = PERSISTED_ROUTE_ID
        for menu in ("데이터 관리", "운영 현황", "추천 실행", "경로 상세", "분석 및 검증"):
            state["current_menu"] = menu
            self.assertEqual(state["selected_route_id"], PERSISTED_ROUTE_ID)
            self.assertEqual(len(state["varo_recommendations"]), EXPECTED_RECOMMENDATION_COUNT)
            self.assertEqual(state["analysis_result"]["summary"]["total_recommended_qty"], EXPECTED_TOTAL_QTY)
            for key in SHARED_KEYS:
                self.assertIn(key, state)

    def test_applying_new_data_resets_previous_runtime_state(self):
        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
        )
        state = {
            "selected_route_id": "OLD",
            "simulation_snapshot": object(),
            "show_all_routes": True,
            "simulation_speed": "빠름",
            "home_sim_playing": True,
            "dqn_training_result": object(),
            "dqn_reflection_mode": "DQN 약하게 반영",
            "dqn_batch_result": object(),
            "dqn_comparison_result": object(),
            "dqn_original_batch_result": object(),
            "dqn_balanced_batch_result": object(),
            "dqn_batch_comparison_result": object(),
            "dqn_sample_diagnosis": object(),
            "dqn_balanced_files": object(),
            "dqn_baseline_recommendations": object(),
            "dqn_baseline_pipeline": object(),
        }
        for key in TRANSIENT_VIEW_KEYS:
            state[key] = "OLD"
        apply_state_payload(state, payload)
        self.assertEqual(state["selected_route_id"], DEFAULT_SELECTED_ROUTE_ID)
        self.assertIsNone(state["simulation_snapshot"])
        self.assertFalse(state["show_all_routes"])
        self.assertEqual(state["simulation_speed"], "보통")
        self.assertFalse(state["home_sim_playing"])
        self.assertIsNone(state["dqn_training_result"])
        self.assertEqual(state["dqn_reflection_mode"], "DQN 참고만")
        self.assertIsNone(state["dqn_batch_result"])
        self.assertIsNone(state["dqn_comparison_result"])
        self.assertIsNone(state["dqn_original_batch_result"])
        self.assertIsNone(state["dqn_balanced_batch_result"])
        self.assertIsNone(state["dqn_batch_comparison_result"])
        self.assertIsNone(state["dqn_sample_diagnosis"])
        self.assertIsNone(state["dqn_balanced_files"])
        self.assertIsNone(state["dqn_baseline_recommendations"])
        self.assertIsNone(state["dqn_baseline_pipeline"])
        self.assertEqual(state["dqn_sample_training_mode"], "original")
        for key in TRANSIENT_VIEW_KEYS:
            self.assertNotIn(key, state)

    def test_status_uses_canonical_applied_state(self):
        state = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
        )
        self.assertEqual(current_data_status(state), "샘플 적용됨")
        self.assertEqual(current_data_status({}), "데이터 없음")


if __name__ == "__main__":
    unittest.main()
