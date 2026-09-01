"""User-flow check on the real pages, driven through Streamlit's AppTest.

Scope (stated explicitly so the coverage is not overclaimed): this drives the
actual Streamlit script and clicks the real buttons — 문제 행을 제외하고 사용 and
추천 실행 — then reads the rendered element tree. The only step that is not a UI
interaction is the OS file picker: the file is handed to the same service the
``st.file_uploader`` branch calls (``prepare_pending_data``). No human browser
session is simulated here.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

try:
    from streamlit.testing.v1 import AppTest

    _APPTEST_AVAILABLE = True
except Exception:  # pragma: no cover - older streamlit
    _APPTEST_AVAILABLE = False

from services.data_application import prepare_pending_data
from tools.generate_anonymized_operational_workbook import MANIFEST_NAME, WORKBOOK_NAME, generate

APP_PATH = str(Path(__file__).resolve().parents[1] / "app_v2.py")
PAGES = ["운영 현황", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리"]
LEAKED_TOKENS = ("candidate_id", "usable_signature", "pending_usable_data", "Traceback")


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class OperationalUiFlowTests(unittest.TestCase):
    """One shared flow: 검사 → 제외 후 적용 → 추천 실행 → 모든 페이지 확인."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="varo_ui_flow_"))
        generate(cls.temp_dir)
        cls.workbook = cls.temp_dir / WORKBOOK_NAME
        cls.manifest = json.loads((cls.temp_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

        app = AppTest.from_file(APP_PATH, default_timeout=300)
        app.run()
        # Same entry point the file_uploader branch uses; only the OS picker is
        # skipped. AppTest's session_state proxy is not a full MutableMapping, so
        # the intake is produced in a plain dict and copied in verbatim.
        staged: dict[str, Any] = {}
        prepare_pending_data(staged, str(cls.workbook), WORKBOOK_NAME, "업로드된 추천 결과")
        for key, value in staged.items():
            app.session_state[key] = value
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        cls.inspect_snapshot = _snapshot(app)

        app.button(key="apply_pending").click().run()
        cls.applied_snapshot = _snapshot(app)

        app.session_state["current_menu"] = "추천 실행"
        app.run()
        cls.before_run_snapshot = _snapshot(app)
        app.button(key="run_applied_analysis").click().run()
        cls.after_run_snapshot = _snapshot(app)
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # --- 검사 화면 ---------------------------------------------------------- #
    def test_inspection_screen_shows_the_real_row_counts(self):
        expected = self.manifest["expected"]
        metrics = self.inspect_snapshot["metrics"]
        self.assertEqual(metrics.get("전체 행"), str(expected["total_rows"]))
        self.assertEqual(metrics.get("분석 사용 행"), str(expected["applied_rows"]))
        self.assertEqual(metrics.get("제외 행"), str(expected["excluded_rows"]))
        self.assertEqual(metrics.get("경고 행"), str(expected["warning_rows"]))

    def test_apply_button_says_what_it_will_do(self):
        self.assertIn("문제 행을 제외하고 사용", self.inspect_snapshot["button_labels"])
        self.assertNotIn("이 데이터 사용", self.inspect_snapshot["button_labels"])

    def test_inspection_screen_explains_exclusions_without_internal_terms(self):
        blob = self.inspect_snapshot["text"]
        self.assertIn("주요 제외 이유", blob)
        self.assertIn("제외", blob)
        for token in LEAKED_TOKENS:
            self.assertNotIn(token, blob)

    def test_nothing_is_applied_before_the_button_is_clicked(self):
        self.assertFalse(self.inspect_snapshot["has_applied_data"])
        self.assertEqual(self.inspect_snapshot["recommendation_count"], 0)

    # --- 적용 결과 ---------------------------------------------------------- #
    def test_clicking_apply_replaces_the_current_data_and_clears_the_intake(self):
        self.assertTrue(self.applied_snapshot["has_applied_data"])
        self.assertFalse(self.applied_snapshot["pending_keys"])
        self.assertEqual(self.applied_snapshot["recommendation_count"], 0)
        self.assertTrue(self.applied_snapshot["analysis_run_required"])
        self.assertIn("제외한 데이터가 적용", self.applied_snapshot["text"])

    def test_applied_screen_reports_the_same_rows_as_the_manifest(self):
        expected = self.manifest["expected"]
        blob = self.applied_snapshot["text"]
        self.assertIn(str(expected["applied_rows"]), blob)
        self.assertIn(str(expected["excluded_rows"]), blob)

    # --- 추천 실행 ---------------------------------------------------------- #
    def test_analysis_does_not_start_on_its_own(self):
        self.assertEqual(self.before_run_snapshot["recommendation_count"], 0)
        self.assertIn("추천 실행", self.before_run_snapshot["button_labels"])

    def test_clicking_run_produces_recommendations(self):
        self.assertGreater(self.after_run_snapshot["recommendation_count"], 0)
        self.assertFalse(self.after_run_snapshot["analysis_run_required"])
        self.assertFalse(self.after_run_snapshot["exceptions"])

    # --- 모든 페이지 -------------------------------------------------------- #
    def test_every_page_renders_and_keeps_internals_off_screen(self):
        signature = str(dict(self.app.session_state.filtered_state).get("data_signature") or "")
        for page in PAGES:
            self.app.session_state["current_menu"] = page
            self.app.run()
            self.assertFalse(self.app.exception, msg=f"{page}: {list(self.app.exception)}")
            blob = _snapshot(self.app)["text"]
            for token in LEAKED_TOKENS:
                self.assertNotIn(token, blob, msg=f"{page}에 {token}이 노출됐습니다")
            if signature:
                self.assertNotIn(signature[:16], blob, msg=f"{page}에 내부 signature가 노출됐습니다")

    def test_home_and_data_management_agree_on_screen(self):
        state = dict(self.app.session_state.filtered_state)
        self.app.session_state["current_menu"] = "운영 현황"
        self.app.run()
        home_blob = _snapshot(self.app)["text"]
        self.app.session_state["current_menu"] = "데이터 관리"
        self.app.run()
        data_blob = _snapshot(self.app)["text"]
        applied_rows = str(self.manifest["expected"]["applied_rows"])
        self.assertIn(applied_rows, data_blob)
        top = state["varo_recommendations"][0]
        self.assertTrue(
            str(top["product_name"]) in home_blob or str(top["target_name"]) in home_blob,
            msg="홈에 최우선 추천 정보가 보이지 않습니다",
        )


def _snapshot(app: Any) -> dict[str, Any]:
    texts = [element.value for element in app.markdown]
    texts += [element.value for element in app.caption] if hasattr(app, "caption") else []
    for attribute in ("info", "success", "warning", "error"):
        if hasattr(app, attribute):
            texts += [element.value for element in getattr(app, attribute)]
    # AppTest's session_state proxy is not a full mapping; filtered_state is the
    # plain dict of non-widget keys.
    state = dict(app.session_state.filtered_state)
    return {
        "text": " ".join(str(item) for item in texts),
        "metrics": {element.label: element.value for element in app.metric},
        "button_labels": [element.label for element in app.button],
        "exceptions": [str(item) for item in app.exception],
        "has_applied_data": (state.get("varo_data") or {}).get("stores") is not None,
        "recommendation_count": len(state.get("varo_recommendations") or []),
        "analysis_run_required": bool(state.get("analysis_run_required")),
        "pending_keys": [
            key for key, value in state.items()
            if key.startswith("pending_") and value not in (None, {}, [], "")
        ],
    }


if __name__ == "__main__":
    unittest.main()
