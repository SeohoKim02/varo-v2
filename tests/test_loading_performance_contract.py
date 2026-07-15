from __future__ import annotations

import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import router
from services import data_application
from services.analysis_pipeline import run_analysis_pipeline
from services.data_loader import load_excel_data
from tests.fixtures import recommendations_frame, sample_workbook, workbook_excel_bytes


class LoadingPerformanceContractTests(unittest.TestCase):
    def setUp(self):
        data_application.clear_load_caches()

    def test_same_content_reuses_excel_and_core_pipeline(self):
        content = workbook_excel_bytes()
        states = [{}, {}]
        original_loader = data_application.load_excel_data
        original_pipeline = data_application.run_analysis_pipeline
        with (
            patch.object(data_application, "load_excel_data", wraps=original_loader) as loader,
            patch.object(data_application, "run_analysis_pipeline", wraps=original_pipeline) as pipeline,
        ):
            self.assertTrue(data_application.load_and_apply(states[0], content, "same.xlsx", "업로드"))
            self.assertTrue(data_application.load_and_apply(states[1], content, "same.xlsx", "업로드"))
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(pipeline.call_count, 1)
        self.assertIsNot(states[0]["varo_data"], states[1]["varo_data"])
        self.assertIsNot(states[0]["varo_recommendations"], states[1]["varo_recommendations"])

    def test_changed_content_invalidates_all_content_caches(self):
        first = workbook_excel_bytes()
        changed_recommendations = recommendations_frame().copy()
        changed_recommendations.loc[0, "expected_saving"] += 1
        changed_workbook = sample_workbook()
        changed_workbook["recommendations"] = changed_recommendations
        second = workbook_excel_bytes(changed_workbook)
        original_loader = data_application.load_excel_data
        original_pipeline = data_application.run_analysis_pipeline
        with (
            patch.object(data_application, "load_excel_data", wraps=original_loader) as loader,
            patch.object(data_application, "run_analysis_pipeline", wraps=original_pipeline) as pipeline,
        ):
            self.assertTrue(data_application.load_and_apply({}, first, "same-name.xlsx", "업로드"))
            self.assertTrue(data_application.load_and_apply({}, second, "same-name.xlsx", "업로드"))
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(pipeline.call_count, 2)

    def test_progress_labels_match_real_load_stages(self):
        labels: list[str] = []
        self.assertTrue(
            data_application.load_and_apply(
                {}, workbook_excel_bytes(), "stages.xlsx", "업로드",
                progress_callback=labels.append,
            )
        )
        self.assertEqual(labels, [
            "데이터 읽는 중", "데이터 확인 중", "추천 계산 중",
            "결과 적용 중", "데이터 적용 완료",
        ])

    def test_core_load_preserves_full_recommendation_order(self):
        data = load_excel_data(BytesIO(workbook_excel_bytes().getvalue()))
        core = run_analysis_pipeline(data, detail_level="core")
        full = run_analysis_pipeline(data, detail_level="full")
        core_order = [(item.get("route_id"), item.get("varo_final_rank")) for item in core.recommendations]
        full_order = [(item.get("route_id"), item.get("varo_final_rank")) for item in full.recommendations]
        self.assertEqual(core_order, full_order)
        self.assertEqual(core.recommendations, full.recommendations)
        self.assertEqual(core.diagnostics.get("detail_level"), "core")
        self.assertEqual(core.validation_report["optimality_gap"]["status"], "지연 실행")

    def test_router_imports_and_calls_only_selected_page(self):
        calls: list[str] = []
        module = SimpleNamespace(render_overview_page=lambda: calls.append("홈"))
        with (
            patch.object(router, "get_current_menu", return_value="홈"),
            patch.object(router, "import_module", return_value=module) as importer,
        ):
            router.render_current_page()
        importer.assert_called_once_with("pages.overview")
        self.assertEqual(calls, ["홈"])


if __name__ == "__main__":
    unittest.main()
