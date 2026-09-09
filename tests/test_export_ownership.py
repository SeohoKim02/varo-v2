"""내보내기의 주인은 화면마다 하나다 (4차 UI 정리).

    재고 운영      실행계획 CSV / Excel        현장에서 실행할 계획 (실행 수량 기준)
    데이터 관리     데이터 오류 목록 CSV        입력 데이터를 고치기 위한 목록
    분석 및 검증    상세 분석 결과 · 검증 결과 · 제외된 이동 검토 · 학습 결과 (연구/검증)
    실행 이력      실행 기록 CSV              계획 대비 실제 결과 (calibration)

이 파일이 지키는 규칙:

* 같은 목적의 파일이 두 화면에서 서로 다른 이름으로 반복 제공되지 않는다.
* 데이터 관리는 추천 결과·실행계획·분석 결과를 내보내지 않는다.
* 내보낼 것이 없으면 빈 파일을 만들지 않고 이유를 말한다.
* 파일 생성에 실패하면 traceback·경로·내부 이름이 아니라 한 줄 안내만 남는다.
* 다운로드 이름은 사용자 목적어(실행계획·데이터 오류 목록·분석 결과·검증 결과)로 쓴다.
* 위치를 옮겼다고 파일 내용이 바뀌지 않는다.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

try:
    from streamlit.testing.v1 import AppTest

    _APPTEST_AVAILABLE = True
except Exception:  # pragma: no cover - older streamlit
    _APPTEST_AVAILABLE = False

from components.exports import EMPTY_PLAN_MESSAGE, EXPORT_FAILED_MESSAGE, render_download
from services import export_service
from services.analysis_pipeline import run_analysis_pipeline
from services.app_state import CANONICAL_DATA_KEYS, build_applied_state_payload
from services.data_application import PENDING_KEYS, prepare_pending_data
from services.data_loader import SAMPLE_FILENAME, get_default_sample_path, load_excel_data
from services.data_validator import validate_workbook_data
from tests.fixtures import sample_workbook, workbook_excel_bytes

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "app_v2.py")

WORKSPACE = "재고 운영"
DATA = "데이터 관리"
VALIDATION = "분석 및 검증"
SIMULATION = "운영 현황"
LEGACY_LIST = "추천 실행"
PRIMARY_SCREENS = [WORKSPACE, DATA, VALIDATION, SIMULATION]

# 사용자에게 보이는 다운로드 전부와 그 단 하나의 주인.
EXPORT_OWNERS = {
    "CSV": WORKSPACE,                          # 실행계획
    "Excel": WORKSPACE,                        # 실행계획
    "실행 기록 CSV": WORKSPACE,                 # 실행 이력 탭 (calibration)
    "검사 중 데이터 오류 목록 CSV": DATA,
    "적용 데이터 오류 목록 CSV": DATA,
    "상세 분석 결과 Excel": VALIDATION,
    "검증 결과 Excel": VALIDATION,
    "제외된 이동 검토 CSV": VALIDATION,
    "학습 결과 다운로드": VALIDATION,           # DQN 학습 탭 (JSON)
    "상세 추천 목록 CSV": LEGACY_LIST,          # 보조 화면(호환 route)
    "상세 추천 목록 Excel": LEGACY_LIST,
}

# 데이터 관리에 두지 않기로 한 결과 다운로드.
RESULT_EXPORT_LABELS = (
    "추천 결과 CSV", "추천 결과 Excel", "분석 결과 전체 Excel", "검증 리포트 Excel",
    "검증 결과 다운로드",
)

UI_SOURCE_FILES = sorted((ROOT / "pages").glob("*.py")) + sorted((ROOT / "components").glob("*.py"))

_RENDER_DOWNLOAD = re.compile(r'render_download\(\s*[A-Za-z_][\w\.\[\]]*\s*,\s*"([^"]+)"')


def _labels_in_source() -> dict[str, str]:
    """{다운로드 이름: 정의된 파일} — 화면 코드 전체에서 실제로 만들어지는 버튼."""
    found: dict[str, str] = {}
    for path in UI_SOURCE_FILES:
        for label in _RENDER_DOWNLOAD.findall(path.read_text(encoding="utf-8")):
            found[label] = path.name
    return found


class ExportHelperTests(unittest.TestCase):
    """§18 · §17 — 실패와 빈 결과를 사용자 문장으로 처리한다."""

    class _Container:
        def __init__(self):
            self.captions: list[str] = []
            self.downloads: list[tuple] = []

        def caption(self, text):
            self.captions.append(str(text))

        def download_button(self, label, **kwargs):
            self.downloads.append((label, kwargs))

    def test_a_working_builder_produces_one_button(self):
        container = self._Container()
        ok = render_download(
            container, "실행계획 CSV", lambda: b"a,b\n", "x.csv", "text/csv", "k",
        )
        self.assertTrue(ok)
        self.assertEqual(len(container.downloads), 1)
        self.assertEqual(container.downloads[0][1]["data"], b"a,b\n")
        self.assertEqual(container.captions, [])

    def test_a_failing_builder_leaves_a_message_not_a_traceback(self):
        container = self._Container()

        def boom() -> bytes:
            raise RuntimeError("openpyxl said no: C:/internal/path.xlsx")

        ok = render_download(container, "실행계획 CSV", boom, "x.csv", "text/csv", "k")
        self.assertFalse(ok)
        self.assertEqual(container.downloads, [])
        self.assertEqual(container.captions, [EXPORT_FAILED_MESSAGE])
        for internal in ("Traceback", "RuntimeError", "openpyxl", "C:/", "path.xlsx"):
            self.assertNotIn(internal, EXPORT_FAILED_MESSAGE)

    def test_an_empty_plan_says_so_instead_of_downloading_an_empty_file(self):
        import pages.workspace as workspace_page

        stub = self._Container()
        with mock.patch.object(workspace_page, "st", stub):
            workspace_page._render_export([])
        self.assertEqual(stub.captions, [EMPTY_PLAN_MESSAGE])
        self.assertEqual(stub.downloads, [])


class ExportSurfaceTests(unittest.TestCase):
    """§8 · §9 · §21 — 진입점 하나, 사용자 용어, 기술 이름 금지."""

    def test_every_download_goes_through_the_shared_helper(self):
        for path in UI_SOURCE_FILES:
            if path.name == "exports.py":
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "download_button(", source,
                f"{path.name}: 다운로드는 components.exports.render_download로만 만든다",
            )

    def test_the_visible_downloads_are_exactly_the_owned_set(self):
        found = _labels_in_source()
        self.assertEqual(set(found), set(EXPORT_OWNERS))

    def test_each_download_is_defined_in_exactly_one_place(self):
        labels = [
            label
            for path in UI_SOURCE_FILES
            for label in _RENDER_DOWNLOAD.findall(path.read_text(encoding="utf-8"))
        ]
        duplicated = {label for label in labels if labels.count(label) > 1}
        self.assertEqual(duplicated, set(), "같은 다운로드가 두 곳에서 만들어진다")

    def test_download_names_read_as_user_purposes_not_as_code(self):
        banned = (
            "recommendations_csv", "candidate export", "full dataframe",
            "validation dump", "raw score", "dataframe", "dump", "raw",
            "export_service", "bytes", "json", "df",
        )
        for label in EXPORT_OWNERS:
            lowered = label.lower()
            for word in banned:
                self.assertNotIn(word, lowered, f"'{label}'은 기술 중심 이름이다")

    def test_only_the_workspace_export_uses_the_execution_plan_vocabulary(self):
        """실행계획이라는 말과 파일은 재고 운영 화면 것이다."""
        workspace = (ROOT / "pages" / "workspace.py").read_text(encoding="utf-8")
        self.assertIn("execution_plan_csv_bytes", workspace)
        self.assertIn("execution_plan_excel_bytes", workspace)
        for name in ("data_management.py", "validation.py", "recommendations.py"):
            source = (ROOT / "pages" / name).read_text(encoding="utf-8")
            self.assertNotIn("execution_plan_csv_bytes", source)
            self.assertNotIn("execution_plan_excel_bytes", source)
            self.assertNotIn("varo_v2_실행계획", source)

    def test_the_calibration_export_is_never_merged_into_the_plan_export(self):
        """§19 — 실행 이력은 계획 export와 목적이 다르므로 따로 남는다."""
        panel = (ROOT / "components" / "execution_history_panel.py").read_text(encoding="utf-8")
        self.assertIn("varo_v2_실행이력.csv", panel)
        self.assertIn("export_execution_history_csv", panel)
        workspace = (ROOT / "pages" / "workspace.py").read_text(encoding="utf-8")
        self.assertNotIn("export_execution_history_csv", workspace)
        self.assertNotIn("varo_v2_실행이력", workspace)

    def test_data_management_never_builds_a_result_file(self):
        source = (ROOT / "pages" / "data_management.py").read_text(encoding="utf-8")
        self.assertNotIn("export_service", source)
        for label in RESULT_EXPORT_LABELS:
            self.assertNotIn(label, source)
        self.assertIn("데이터 오류 목록 CSV", source)


class ExportContentRegressionTests(unittest.TestCase):
    """§30 — 위치를 옮겼다고 파일 내용이 바뀌지 않는다."""

    @classmethod
    def setUpClass(cls):
        data = load_excel_data(get_default_sample_path())
        cls.pipeline = run_analysis_pipeline(data).to_dict()
        cls.recommendations = cls.pipeline["recommendations"]

    def _sheets(self, payload: bytes) -> list[str]:
        return pd.ExcelFile(pd.io.common.BytesIO(payload)).sheet_names

    def test_the_research_workbook_keeps_its_sheets_after_the_move(self):
        payload = export_service.analysis_result_excel_bytes(self.pipeline, self.recommendations, {})
        sheets = self._sheets(payload)
        for required in ("추천결과", "VHS비교", "민감도요약", "추천사유", "검증요약"):
            self.assertIn(required, sheets)

    def test_the_validation_workbook_keeps_its_sheets_after_the_move(self):
        payload = export_service.validation_report_excel_bytes(None, self.pipeline, self.recommendations, {})
        sheets = self._sheets(payload)
        for required in ("검증개요", "알고리즘상태"):
            self.assertIn(required, sheets)

    def test_an_empty_research_export_still_produces_a_readable_file(self):
        for payload in (
            export_service.analysis_result_excel_bytes({}, []),
            export_service.validation_report_excel_bytes(None, {}, []),
        ):
            self.assertTrue(payload)
            self.assertTrue(self._sheets(payload), "빈 입력이어도 열 수 있는 파일이어야 한다")

    def test_the_execution_plan_export_still_carries_the_execution_quantity(self):
        items = [
            {
                "route_id": "R1", "source_name": "S1", "target_name": "S2",
                "product_name": "P", "planned_qty": 7, "recommended_qty": 40,
                "route_type": "DIRECT",
            }
        ]
        text = export_service.execution_plan_csv_bytes(items).decode("utf-8-sig")
        self.assertIn("실행 수량", text)
        self.assertIn("7", text.splitlines()[1])
        self.assertNotIn("40", text.splitlines()[1])


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class ScreenOwnershipTests(unittest.TestCase):
    """실제로 그려진 화면에서 각 다운로드가 자기 화면에만 있는지."""

    @classmethod
    def setUpClass(cls):
        data = load_excel_data(get_default_sample_path())
        validation = validate_workbook_data(data)
        pipeline = run_analysis_pipeline(data).to_dict()
        cls.payload = build_applied_state_payload(
            data, validation, pipeline["recommendations"],
            SAMPLE_FILENAME, "샘플 추천 데이터", pipeline,
        )

    def _screen(self, menu: str) -> "AppTest":
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["current_menu"] = menu
        app.run()
        self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")
        return app

    def _labels(self, app) -> set[str]:
        return {element.label for element in app.get("download_button")}

    def test_no_export_is_offered_by_two_screens(self):
        seen: dict[str, str] = {}
        for menu in PRIMARY_SCREENS:
            for label in self._labels(self._screen(menu)):
                self.assertNotIn(label, seen, f"'{label}'이 {seen.get(label)}와 {menu} 두 곳에 있다")
                seen[label] = menu
                self.assertEqual(
                    EXPORT_OWNERS.get(label), menu,
                    f"'{label}'의 주인은 {EXPORT_OWNERS.get(label)}인데 {menu}에 있다",
                )

    def test_the_workspace_offers_the_execution_plan_and_nothing_else(self):
        app = self._screen(WORKSPACE)
        self.assertEqual(self._labels(app), {"CSV", "Excel"})
        self.assertIn("내보내기", " ".join(item.label for item in app.expander))

    def test_the_analysis_screen_owns_the_research_files_in_a_folded_area(self):
        app = self._screen(VALIDATION)
        labels = self._labels(app)
        self.assertIn("상세 분석 결과 Excel", labels)
        self.assertIn("검증 결과 Excel", labels)
        # 실행용 파일은 여기서 나가지 않는다.
        self.assertNotIn("CSV", labels)
        self.assertNotIn("Excel", labels)
        self.assertIn("내보내기", {item.label for item in app.expander})

    def test_the_applied_data_screen_offers_no_result_file(self):
        app = self._screen(DATA)
        labels = self._labels(app)
        for banned in RESULT_EXPORT_LABELS:
            self.assertNotIn(banned, labels)
        captions = " ".join(str(item.value) for item in app.caption)
        self.assertIn("분석 및 검증 화면의 내보내기", captions)


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class DataManagementExportStateTests(unittest.TestCase):
    """§17 · §5 — 데이터 관리는 오류가 있을 때만, 오류 목록만 내보낸다."""

    def _app(self) -> "AppTest":
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        return app

    def _labels(self, app) -> set[str]:
        return {element.label for element in app.get("download_button")}

    def test_no_data_offers_no_download_at_all(self):
        app = self._app()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(self._labels(app), set())

    def test_clean_applied_data_does_not_offer_an_empty_problem_list(self):
        app = self._app()
        sample = next(button for button in app.button if button.key == "quick_empty_sample")
        sample.click().run()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        from services.data_issues import collect_data_issues

        issues = collect_data_issues(
            app.session_state["varo_data"],
            app.session_state["raw_data"],
            app.session_state["source_metadata"],
        )
        self.assertEqual(issues["summary"]["total_issues"], 0)
        self.assertEqual(self._labels(app), set(), "오류 0건이면 빈 오류 CSV를 만들지 않는다")

    def _pending_with_problem_rows(self) -> dict:
        import warnings

        workbook = sample_workbook()
        workbook["inventory"].loc[0, "stock_qty"] = -1
        state: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepare_pending_data(state, workbook_excel_bytes(workbook), "부분적용.xlsx", "업로드된 추천 결과")
        return state

    def test_problem_rows_offer_the_problem_list_and_only_that(self):
        pending = self._pending_with_problem_rows()
        app = self._app()
        for key in PENDING_KEYS:
            app.session_state[key] = pending.get(key)
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(self._labels(app), {"검사 중 데이터 오류 목록 CSV"})

        # 부분 적용 후에도 이 화면은 결과 파일을 내보내지 않는다.
        apply_button = next(button for button in app.button if button.key == "apply_pending")
        apply_button.click().run()
        self.assertFalse(app.exception)
        for banned in RESULT_EXPORT_LABELS:
            self.assertNotIn(banned, self._labels(app))


if __name__ == "__main__":
    unittest.main()
