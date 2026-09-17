"""End-to-end wiring for the simulation history and the DQN comparison column.

The simulation screen must save one row per real execution without showing any
new control, the history page must read that row back, and the comparison page
must say 비교 불가 while no trained model exists.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

try:
    from streamlit.testing.v1 import AppTest

    _APPTEST_AVAILABLE = True
except Exception:  # pragma: no cover - older streamlit
    _APPTEST_AVAILABLE = False

from services.analysis_pipeline import run_analysis_pipeline
from services.app_state import CANONICAL_DATA_KEYS, build_applied_state_payload
from services.data_loader import SAMPLE_FILENAME, get_default_sample_path, load_excel_data
from services.data_validator import validate_workbook_data
from services.simulation_history import (
    get_run_routes,
    history_storage_info,
    list_simulation_runs,
)

APP_PATH = str(Path(__file__).resolve().parents[1] / "app_v2.py")


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class SimulationHistoryUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_excel_data(get_default_sample_path())
        validation = validate_workbook_data(data)
        pipeline = run_analysis_pipeline(data).to_dict()
        cls.payload = build_applied_state_payload(
            data, validation, pipeline["recommendations"], SAMPLE_FILENAME,
            "샘플 추천 데이터", pipeline, data_signature="history-fixture-signature",
        )

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.history_dir = self.root / "simulation_history"
        self._env = patch.dict(os.environ, {"VARO_OUTPUT_ROOT": str(self.root)})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._temp.cleanup)

    def _new_app(self):
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        return app

    def _run_simulation(self, app, nonce: int):
        app.session_state["current_menu"] = "시뮬레이션"
        app.session_state["home_sim_playing"] = True
        app.session_state["home_sim_run_nonce"] = nonce
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_one_execution_saves_exactly_one_run(self):
        app = self._new_app()
        self._run_simulation(app, 1)
        runs = list_simulation_runs(directory=self.history_dir)
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run["status"], "완료")
        self.assertEqual(run["data_signature"], "history-fixture-signature")
        self.assertEqual(run["data_name"], SAMPLE_FILENAME)
        self.assertGreater(run["candidate_count"], 0)
        self.assertGreaterEqual(run["selected_count"], 0)
        self.assertTrue(get_run_routes(run["run_id"], self.history_dir))

    def test_reruns_of_the_same_execution_are_not_duplicated(self):
        app = self._new_app()
        self._run_simulation(app, 1)
        app.run()
        app.run()
        self.assertEqual(len(list_simulation_runs(directory=self.history_dir)), 1)

    def test_a_new_execution_adds_a_new_run(self):
        app = self._new_app()
        self._run_simulation(app, 1)
        self._run_simulation(app, 2)
        self.assertEqual(len(list_simulation_runs(directory=self.history_dir)), 2)

    def test_simulation_screen_gains_no_history_control(self):
        app = self._new_app()
        self._run_simulation(app, 1)
        labels = {item.label for item in app.main.button}
        self.assertNotIn("이력 저장", labels)
        self.assertNotIn("결과 저장", labels)
        self.assertEqual(labels, {"시뮬레이션 실행", "다시 실행"})

    def test_saved_run_stays_small(self):
        app = self._new_app()
        self._run_simulation(app, 1)
        info = history_storage_info(self.history_dir)
        self.assertEqual(info["run_count"], 1)
        self.assertLess(info["size_bytes"], 400 * 1024)

    def test_history_page_shows_the_stored_simulation_run(self):
        app = self._new_app()
        self._run_simulation(app, 1)
        app.session_state["current_menu"] = "결과 이력"
        app.run()
        self.assertFalse(app.exception)
        blob = " ".join(element.value for element in app.markdown)
        self.assertIn("실행 요약", blob)
        self.assertIn("최종 선택 경로", blob)
        self.assertIn("시뮬레이션 실행 목록", blob)
        tab_labels = {item.label for item in app.tabs}
        self.assertTrue({"시뮬레이션 이력", "DQN 학습/저장 결과"}.issubset(tab_labels))
        columns = {
            str(column)
            for element in app.dataframe
            for column in getattr(element.value, "columns", [])
        }
        self.assertTrue({"최종 선택 건수", "총 이동수량", "예상 절감액", "상태"}.issubset(columns))

    def test_history_page_without_runs_says_so_instead_of_inventing_rows(self):
        app = self._new_app()
        app.session_state["current_menu"] = "결과 이력"
        app.run()
        self.assertFalse(app.exception)
        blob = " ".join(element.value for element in app.markdown)
        self.assertIn("저장된 시뮬레이션 이력이 없습니다", blob)

    def test_strategy_comparison_reports_dqn_unavailable_without_a_model(self):
        app = self._new_app()
        app.session_state["current_menu"] = "전략 비교"
        with patch("services.dqn_service.find_compatible_model", return_value=None):
            app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("DQN 비교 불가" in item.value for item in app.info))
        dqn_column = [
            list(element.value["DQN"])
            for element in app.dataframe
            if "DQN" in getattr(element.value, "columns", [])
        ]
        self.assertTrue(dqn_column)
        self.assertTrue(all(value == "비교 불가" for value in dqn_column[0]))


    def test_strategy_comparison_uses_a_real_trained_model_when_one_exists(self):
        from services import dqn_service
        from tests.test_dqn_training_contract import TORCH_AVAILABLE, _artifact_directory

        if not TORCH_AVAILABLE:
            self.skipTest("DQN runtime unavailable")
        app = self._new_app()
        app.session_state["current_menu"] = "전략 비교"
        recommendations = list(self.payload["varo_recommendations"])
        with _artifact_directory():
            trained = dqn_service.train_dqn(
                recommendations,
                data_signature="history-fixture-signature",
                episodes=20,
                learning_rate=0.002,
            )
            self.assertEqual(trained.status, dqn_service.NORMAL_STATUS)
            app.run()
        self.assertFalse(app.exception)
        blob = " ".join(element.value for element in app.markdown)
        self.assertIn("전략 비교", blob)
        captions = " ".join(item.value for item in app.caption)
        self.assertIn("저장된 모델을 실제로 불러와 추론한 결과", captions)
        frames = [
            element.value for element in app.dataframe
            if "DQN" in getattr(element.value, "columns", [])
        ]
        self.assertTrue(frames)
        summary = frames[0].set_index("지표")
        self.assertEqual(summary.loc["실행 상태", "DQN"], "비교 가능")
        self.assertNotEqual(summary.loc["선택/평가 건수", "DQN"], "비교 불가")
        # DQN must stay an independent strategy, not a copy of Greedy or VHS.
        judgements = [
            element.value for element in app.dataframe
            if "DQN 판단" in getattr(element.value, "columns", [])
        ]
        self.assertTrue(judgements)
        self.assertGreater(len(set(judgements[0]["DQN 판단"])), 1)


if __name__ == "__main__":
    unittest.main()
