from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.app_state import CANONICAL_DATA_KEYS
from services.data_application import load_and_apply
from services.execution_history import get_recorded_plan, list_recorded_plans
from tests.fixtures import sample_workbook, workbook_excel_bytes

try:
    from streamlit.testing.v1 import AppTest
except Exception:  # pragma: no cover
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit AppTest unavailable")
class ExecutionHistoryUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state: dict = {}
        assert load_and_apply(
            state, workbook_excel_bytes(sample_workbook()),
            "anonymous-history-ui.xlsx", "샘플 추천 데이터",
        )
        cls.payload = state
        cls.app_path = str(Path(__file__).resolve().parents[1] / "app_v2.py")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "history.sqlite3"
        self.env = mock.patch.dict(os.environ, {"VARO_HISTORY_DB_PATH": str(self.db)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def app(self):
        app = AppTest.from_file(self.app_path, default_timeout=120)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["current_menu"] = "추천 실행"
        app.run()
        self.assertFalse(app.exception)
        return app

    @staticmethod
    def markdown_blob(app) -> str:
        return " ".join(str(element.value) for element in app.markdown)

    def test_plan_is_saved_only_after_explicit_button(self):
        app = self.app()
        self.assertEqual(list_recorded_plans(self.db)["plans"], [])
        self.assertFalse(self.db.exists())
        button = next(item for item in app.button if item.key == "record_execution_plan")
        self.assertEqual(button.label, "이 계획 기록")
        self.assertFalse(button.disabled)
        button.click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(list_recorded_plans(self.db)["plans"]), 1)

    def test_operator_can_record_status_quantity_and_outcome_without_internal_ids(self):
        app = self.app()
        next(item for item in app.button if item.key == "record_execution_plan").click().run()
        app.run()
        self.assertFalse(app.exception)

        plan = self.payload["varo_pipeline_result"]["execution_plan"]
        stored = get_recorded_plan(plan["plan_id"], self.db)
        first = stored["items"][0]
        actual_qty = int(first["planned_qty"]) + 1

        next(item for item in app.selectbox if str(item.key).startswith("history_execution_status_")).select("실행")
        next(item for item in app.text_input if str(item.key).startswith("history_actual_qty_")).input(str(actual_qty))
        next(item for item in app.selectbox if str(item.key).startswith("history_reason_")).select("현장 판단")
        next(item for item in app.text_input if str(item.key).startswith("history_cost_")).input("2500")
        next(item for item in app.text_input if str(item.key).startswith("history_saving_")).input("9000")
        next(item for item in app.button if item.label == "실행 결과 저장").click().run()
        self.assertFalse(app.exception)

        updated = get_recorded_plan(plan["plan_id"], self.db)
        item = next(row for row in updated["items"] if row["candidate_id"] == first["candidate_id"])
        self.assertEqual(item["execution_status"], "executed")
        self.assertEqual(item["actual_qty"], actual_qty)
        self.assertEqual(item["actual_transport_cost"], 2500)
        self.assertEqual(item["actual_saving"], 9000)

        blob = self.markdown_blob(app)
        for hidden in (plan["plan_id"], plan["data_signature"], first["candidate_id"], "Traceback", "sqlite"):
            self.assertNotIn(str(hidden), blob)


if __name__ == "__main__":
    unittest.main()
