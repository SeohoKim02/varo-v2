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

    def _dirty_previous_session_state(self) -> dict:
        """A state dict as it would look after a fully-used previous dataset."""
        state = {
            "selected_route_id": "OLD",
            "simulation_snapshot": object(),
            "show_all_routes": True,
            "simulation_speed": "빠름",
            "home_sim_playing": True,
            "dqn_training_result": {"data_signature": "old-signature", "status": "정상"},
            "dqn_reflection_mode": "DQN 약하게 반영",
            "kakao_map_state": {"loaded": True, "cache": "old"},
        }
        # Every widget-backed transient key carries a value from the old data.
        for key in TRANSIENT_VIEW_KEYS:
            state[key] = "OLD"
        # Realistic shadow-control values (mirrors what the home widgets store).
        state["home_speed_select"] = "빠름"
        state["home_show_all"] = True
        return state

    def test_applying_new_data_removes_all_transient_view_keys(self):
        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
        )
        state = self._dirty_previous_session_state()
        # Guard: the shadow controls must be part of the transient contract, or
        # resetting simulation_speed / show_all_routes would be silently undone.
        self.assertIn("home_speed_select", TRANSIENT_VIEW_KEYS)
        self.assertIn("home_show_all", TRANSIENT_VIEW_KEYS)
        apply_state_payload(state, payload)
        for key in TRANSIENT_VIEW_KEYS:
            self.assertNotIn(key, state, f"transient key survived new data: {key}")

    def test_applying_new_data_resets_previous_runtime_state(self):
        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
        )
        state = self._dirty_previous_session_state()
        apply_state_payload(state, payload)
        # selected_route_id must fall back to the new top route, not the old one.
        self.assertEqual(state["selected_route_id"], DEFAULT_SELECTED_ROUTE_ID)
        self.assertIsNone(state["simulation_snapshot"])
        # The shadow keys are gone AND the canonical resets held (not overwritten).
        self.assertFalse(state["show_all_routes"])
        self.assertEqual(state["simulation_speed"], "보통")
        self.assertFalse(state["home_sim_playing"])
        self.assertNotIn("home_speed_select", state)
        self.assertNotIn("home_show_all", state)

    def test_applying_new_data_does_not_carry_dqn_or_kakao_state(self):
        """Past DQN/Kakao runtime state must never auto-carry into new data."""
        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
        )
        state = self._dirty_previous_session_state()
        apply_state_payload(state, payload)
        # A stale DQN result (with an old data_signature) is dropped, so it can
        # never be reflected in the new dataset's VHS comparison.
        self.assertIsNone(state["dqn_training_result"])
        self.assertEqual(state["dqn_reflection_mode"], "DQN 참고만")
        # Kakao cache is cleared; the map is only prepared later, on demand.
        self.assertIsNone(state["kakao_map_state"])

    def test_apply_state_tolerates_widget_bound_reflection_mode_key(self):
        """Regression: applying data from within the DQN tab must not crash.

        There the dqn_reflection_mode radio is already instantiated, and Streamlit
        raises StreamlitAPIException when a key bound to a live widget is written.
        apply_state_payload must skip only that write and still apply every other
        canonical reset. The data-management load path (a plain dict, covered by
        the other tests) keeps resetting the value normally.
        """
        class StreamlitAPIException(Exception):  # name matched by _assign's guard
            pass

        class _LiveWidgetState(dict):
            def __setitem__(self, key, value):
                if key == "dqn_reflection_mode":
                    raise StreamlitAPIException("bound to an instantiated widget")
                super().__setitem__(key, value)

        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
        )
        state = _LiveWidgetState(self._dirty_previous_session_state())
        apply_state_payload(state, payload)  # must not raise
        # The live widget keeps its user-set value (the write was skipped).
        self.assertEqual(state["dqn_reflection_mode"], "DQN 약하게 반영")
        # Every other canonical reset still applied.
        self.assertEqual(state["uploaded_filename"], FIXTURE_FILENAME)
        self.assertEqual(len(state["varo_recommendations"]), EXPECTED_RECOMMENDATION_COUNT)
        self.assertIsNone(state["dqn_training_result"])
        self.assertIsNone(state["kakao_map_state"])
        for key in TRANSIENT_VIEW_KEYS:
            self.assertNotIn(key, state)

    def test_apply_state_reraises_non_widget_errors(self):
        """_assign must not swallow errors other than the live-widget one."""
        class _Boom(Exception):
            pass

        class _BrokenState(dict):
            def __setitem__(self, key, value):
                if key == "dqn_reflection_mode":
                    raise _Boom("unexpected failure")
                super().__setitem__(key, value)

        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
        )
        state = _BrokenState(self._dirty_previous_session_state())
        with self.assertRaises(_Boom):
            apply_state_payload(state, payload)

    def test_applying_new_data_preserves_core_results(self):
        """Uploaded data, recommendations, validation, filename, signature stay."""
        payload = build_applied_state_payload(
            self.data, self.validation, self.recommendations,
            FIXTURE_FILENAME, "샘플 추천 데이터", self.pipeline_result,
            data_signature="new-signature",
        )
        state = self._dirty_previous_session_state()
        apply_state_payload(state, payload)
        self.assertEqual(len(state["varo_recommendations"]), EXPECTED_RECOMMENDATION_COUNT)
        self.assertIs(state["varo_validation"], self.validation)
        self.assertFalse(state["varo_validation"].has_errors)
        self.assertEqual(state["uploaded_filename"], FIXTURE_FILENAME)
        self.assertEqual(state["data_source_type"], "샘플 추천 데이터")
        self.assertEqual(state["data_signature"], "new-signature")
        self.assertTrue(has_app_data(state["varo_data"], state["varo_recommendations"]))
        self.assertEqual(current_data_status(state), "알고리즘 연결됨")

    def test_status_uses_canonical_applied_state(self):
        state = build_applied_state_payload(
            self.data, self.validation, self.recommendations, FIXTURE_FILENAME, "샘플 추천 데이터",
        )
        self.assertEqual(current_data_status(state), "샘플 적용됨")
        self.assertEqual(current_data_status({}), "데이터 없음")


if __name__ == "__main__":
    unittest.main()
