"""Tests for upload validation messaging and crash-proof failure handling."""
from __future__ import annotations

import unittest
from io import BytesIO

from services.data_application import load_and_apply
from services.upload_quality import build_upload_report, upload_quality_rows
from tests.fixtures import (
    inventory_frame,
    products_frame,
    recommendations_frame,
    routes_frame,
    stores_frame,
    workbook_excel_bytes,
)


class UploadValidationTests(unittest.TestCase):
    def test_missing_sheet_reports_the_exact_sheet_name(self):
        state: dict = {}
        ok = load_and_apply(state, workbook_excel_bytes({
            "stores": stores_frame(), "products": products_frame(),
            "inventory": inventory_frame(), "recommendations": recommendations_frame(),
        }), "missing_routes.xlsx", "업로드된 추천 결과")
        self.assertFalse(ok)
        self.assertEqual(state.get("pending_load_error"), "필수 시트 누락: routes")

    def test_missing_required_column_blocks_apply_with_report(self):
        inventory = inventory_frame().drop(columns=["stock_qty"])  # required
        state: dict = {}
        ok = load_and_apply(state, workbook_excel_bytes({
            "stores": stores_frame(), "products": products_frame(),
            "inventory": inventory, "routes": routes_frame(),
            "recommendations": recommendations_frame(),
        }), "missing.xlsx", "업로드된 추천 결과")
        self.assertFalse(ok)
        self.assertIsNone(state.get("pending_load_error"))  # file read fine
        self.assertEqual(state.get("pending_validation_error"), "필수 컬럼 누락: inventory.stock_qty")
        report = state.get("pending_upload_report") or {}
        self.assertFalse(report.get("analyzable"))
        self.assertGreaterEqual(report.get("missing_required_count", 0), 1)

    def test_corrupt_file_does_not_crash_and_preserves_previous_state(self):
        # apply a good workbook first
        state: dict = {}
        self.assertTrue(load_and_apply(state, workbook_excel_bytes(), "good.xlsx", "업로드된 추천 결과"))
        previous_recs = list(state["varo_recommendations"])
        # then a corrupt upload
        ok = load_and_apply(state, BytesIO(b"this is not an excel file"), "bad.xlsx", "업로드된 추천 결과")
        self.assertFalse(ok)
        self.assertTrue(state.get("pending_load_error"))
        self.assertIn("엑셀 파일을 읽을 수 없습니다", state["pending_load_error"])
        # previous good data is preserved (menu navigation still possible)
        self.assertEqual(state["varo_recommendations"], previous_recs)

    def test_build_upload_report_shape(self):
        class _Validation:
            status = "통과"
            has_errors = False
            messages = []

        report = build_upload_report(
            {"recognized_sheets": ["stores"], "column_mappings": [{"sheet": "stores", "original": "점포명", "standard": "node_name"}],
             "numeric_failed": {"inventory": 2}, "blank_removed": {"stores": 1}, "row_counts": {"stores": 4}},
            _Validation(), "generated", {"reason": "추천 시트 없음", "count": 3}, "f.xlsx",
        )
        self.assertEqual(report["recommendation_source_label"], "V2 생성 후보 사용")
        self.assertEqual(report["mapped_column_count"], 1)
        self.assertEqual(report["numeric_failed_total"], 2)
        self.assertEqual(report["blank_removed_total"], 1)
        self.assertTrue(report["analyzable"])
        rows = upload_quality_rows(report)
        keys = {row["항목"] for row in rows}
        self.assertIn("추천 생성 방식", keys)
        self.assertIn("DQN", keys)


if __name__ == "__main__":
    unittest.main()
