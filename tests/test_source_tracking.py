"""원본 데이터 기준 오류 추적 테스트.

정규화된 값이 아니라 사용자가 올린 원본 파일의 행·컬럼·값 기준으로 문제를 찾을 수
있는지 검증한다. 픽스처는 in-package 워크북(Excel 왕복 포함)으로 생성하며 외부 파일에
의존하지 않는다.
"""
from __future__ import annotations

import unittest

import pandas as pd

from services.data_application import load_and_apply
from services.data_issues import (
    HEAVY_EXCLUSION_RATIO,
    collect_data_issues,
    detail_rows,
    display_rows,
    issues_to_csv_bytes,
)
from services.data_loader import load_excel_data, normalize_loaded_data
from services.data_validator import validate_workbook_data
from tests.fixtures import (
    inventory_frame,
    products_frame,
    recommendations_frame,
    routes_frame,
    sample_workbook,
    stores_frame,
    workbook_excel_bytes,
)


def _load_with_raw(sheets: dict[str, pd.DataFrame]):
    """Real Excel round-trip: returns (normalized, raw_snapshots, metadata)."""
    data, report = load_excel_data(workbook_excel_bytes(sheets), return_report=True)
    meta = {"filename": "t.xlsx", "source_type": "excel", "sheet_names": report["raw_sheet_names"]}
    return data, report["raw_sheets"], meta


def _codes(out):
    return [i["code"] for i in out["issues"]]


def _by_code(out, code):
    return [i for i in out["issues"] if i["code"] == code]


# ── A. 원본 행 번호 ──────────────────────────────────────────────────────────
class OriginalRowTrackingTests(unittest.TestCase):
    def test_excel_first_data_row_is_user_row_2(self):
        inv = inventory_frame()
        inv["stock_qty"] = inv["stock_qty"].astype(object)
        inv.loc[0, "stock_qty"] = "십오"
        data, raw, meta = _load_with_raw({**sample_workbook(), "inventory": inv})
        issue = _by_code(collect_data_issues(data, raw, meta), "non_numeric")[0]
        self.assertEqual(issue["행"], 2)
        self.assertEqual(issue["source_row_number"], 2)

    def test_blank_middle_row_keeps_following_row_numbers(self):
        inv = pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 10},
            {"store_id": "", "product_id": "", "stock_qty": ""},   # fully blank -> skipped
            {"store_id": "S3", "product_id": "P1", "stock_qty": "십오"},  # stays row 4
        ])
        out = collect_data_issues({"inventory": inv})
        nonnum = _by_code(out, "non_numeric")[0]
        self.assertEqual(nonnum["행"], 4)
        # the fully-blank row is not reported as a missing-id error
        self.assertEqual([i for i in _codes(out) if i == "missing_id"], [])

    def test_multi_sheet_row_and_sheet_name_preserved(self):
        recs = recommendations_frame()
        recs.loc[0, "target_id"] = recs.loc[0, "source_id"]  # same_source_target
        data, raw, meta = _load_with_raw({**sample_workbook(), "recommendations": recs})
        issue = _by_code(collect_data_issues(data, raw, meta), "same_source_target")[0]
        self.assertEqual(issue["행"], 2)
        # sheet is written as v2_recommendations; the file sheet name is surfaced
        self.assertEqual(issue["source_sheet_name"], "v2_recommendations")

    def test_original_row_number_survives_normalization(self):
        inv = inventory_frame()
        inv["stock_qty"] = inv["stock_qty"].astype(object)
        inv.loc[3, "stock_qty"] = "십오"  # 4th data row -> file row 5
        data, raw, meta = _load_with_raw({**sample_workbook(), "inventory": inv})
        issue = _by_code(collect_data_issues(data, raw, meta), "non_numeric")[0]
        self.assertEqual(issue["행"], 5)


# ── B. 원본 값 보존 ──────────────────────────────────────────────────────────
class OriginalValuePreservationTests(unittest.TestCase):
    def test_non_numeric_string_is_preserved(self):
        inv = pd.DataFrame([{"store_id": "S1", "product_id": "P1", "stock_qty": "십오"}])
        issue = _by_code(collect_data_issues({"inventory": inv}), "non_numeric")[0]
        self.assertEqual(issue["original_value"], "십오")
        self.assertEqual(issue["normalized_value"], "변환 실패")
        self.assertEqual(issue["값"], "십오")

    def test_comma_number_is_accepted_without_issue(self):
        inv = pd.DataFrame([{"store_id": "S1", "product_id": "P1", "stock_qty": "1,500"}])
        self.assertEqual(collect_data_issues({"inventory": inv})["summary"]["total_issues"], 0)

    def test_unit_suffix_string_is_non_numeric_and_preserved(self):
        inv = pd.DataFrame([{"store_id": "S1", "product_id": "P1", "stock_qty": "20개"}])
        # "20개" -> 20 is recoverable by the normalizer, so it is not an error;
        # a purely non-numeric unit like "십오개" is.
        inv2 = pd.DataFrame([{"store_id": "S1", "product_id": "P1", "stock_qty": "열개"}])
        self.assertEqual(collect_data_issues({"inventory": inv})["summary"]["total_issues"], 0)
        issue = _by_code(collect_data_issues({"inventory": inv2}), "non_numeric")[0]
        self.assertEqual(issue["original_value"], "열개")

    def test_blank_value_shows_friendly_label_not_nan(self):
        inv = pd.DataFrame([{"store_id": None, "product_id": "P1", "stock_qty": 10}])
        issue = _by_code(collect_data_issues({"inventory": inv}), "missing_id")[0]
        self.assertEqual(issue["값"], "빈 값")
        self.assertNotIn(issue["값"], ("nan", "None", "NaN"))
        self.assertEqual(issue["original_value"], "")

    def test_leading_zero_identifier_is_not_stripped(self):
        inv = pd.DataFrame([{"store_id": "001", "product_id": "000123", "stock_qty": 10}])
        normalized = normalize_loaded_data({"inventory": inv})["inventory"]
        self.assertEqual(str(normalized.loc[0, "store_id"]), "001")
        self.assertEqual(str(normalized.loc[0, "product_id"]), "000123")

    def test_korean_value_preserved(self):
        recs = recommendations_frame()
        recs["recommended_qty"] = recs["recommended_qty"].astype(object)
        recs.loc[0, "recommended_qty"] = "다섯개들이"
        issue = _by_code(collect_data_issues({"recommendations": recs}), "non_numeric")[0]
        self.assertEqual(issue["original_value"], "다섯개들이")


# ── C. 컬럼 추적 ─────────────────────────────────────────────────────────────
class ColumnTrackingTests(unittest.TestCase):
    def test_korean_alias_column_name_is_shown(self):
        inv = pd.DataFrame([{"점포id": "S1", "상품코드": "P1", "현재고": "십오"}])
        issue = _by_code(collect_data_issues({"inventory": inv}), "non_numeric")[0]
        self.assertEqual(issue["컬럼"], "현재고")
        self.assertEqual(issue["source_column_name"], "현재고")
        self.assertEqual(issue["canonical_column_name"], "stock_qty")

    def test_whitespace_original_column_name_preserved(self):
        inv = pd.DataFrame([{"store_id": "S1", "product_id": "P1", "재고 수량 ": "십오"}])
        issue = _by_code(collect_data_issues({"inventory": inv}), "non_numeric")[0]
        self.assertIn("재고 수량", issue["source_column_name"])

    def test_two_columns_mapping_to_same_standard_conflict(self):
        inv = pd.DataFrame([{"store_id": "S1", "product_id": "P1", "현재고": 10, "재고": 12}])
        conflict = _by_code(collect_data_issues({"inventory": inv}), "alias_conflict")
        self.assertTrue(conflict)
        self.assertIn("현재고", conflict[0]["source_column_name"])
        self.assertIn("재고", conflict[0]["source_column_name"])
        self.assertFalse(conflict[0]["blocks_analysis"])  # 확인 필요, not 사용 불가

    def test_no_false_conflict_when_standard_column_present(self):
        # sales_qty and avg_daily_sales are cross-listed aliases but both are real
        # columns in the fixture; standard present -> no conflict.
        out = collect_data_issues({"inventory": inventory_frame()})
        self.assertEqual(_by_code(out, "alias_conflict"), [])

    def test_numeric_identifier_column_warns_about_leading_zero(self):
        inv = pd.DataFrame([{"store_id": 1, "product_id": 2, "stock_qty": 10}])
        self.assertIn("id_numeric", _codes(collect_data_issues({"inventory": inv})))


# ── D. 중복 추적 ─────────────────────────────────────────────────────────────
class DuplicateTrackingTests(unittest.TestCase):
    def test_exact_duplicate_reports_all_related_rows(self):
        inv = pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 10},
            {"store_id": "S1", "product_id": "P1", "stock_qty": 10},
        ])
        issue = _by_code(collect_data_issues({"inventory": inv}), "exact_duplicate")[0]
        self.assertEqual(issue["related_rows"], [2, 3])

    def test_value_conflict_duplicate_reports_related_rows(self):
        inv = pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 10},
            {"store_id": "S1", "product_id": "P1", "stock_qty": 20},
        ])
        out = collect_data_issues({"inventory": inv})
        issue = _by_code(out, "conflict_duplicate")[0]
        self.assertEqual(issue["related_rows"], [2, 3])
        self.assertFalse(out["summary"]["has_blocking"])  # 확인 필요, not merged

    def test_time_series_rows_not_flagged_as_duplicate(self):
        inv = pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 10, "snapshot_date": "2026-01-01"},
            {"store_id": "S1", "product_id": "P1", "stock_qty": 8, "snapshot_date": "2026-01-02"},
        ])
        out = collect_data_issues({"inventory": inv})
        self.assertNotIn("conflict_duplicate", _codes(out))
        self.assertNotIn("exact_duplicate", _codes(out))


# ── E. 심각도와 게이트 ───────────────────────────────────────────────────────
class SeverityGateTests(unittest.TestCase):
    def test_error_codes_block_and_warning_codes_do_not(self):
        inv = pd.DataFrame([
            {"store_id": "", "product_id": "P1", "stock_qty": "십오"},   # missing_id + non_numeric
            {"store_id": "S1", "product_id": "P1", "stock_qty": 5},
            {"store_id": "S1", "product_id": "P1", "stock_qty": 9},      # conflict warning
        ])
        out = collect_data_issues({"inventory": inv})
        for item in out["issues"]:
            if item["severity"] == "오류":
                self.assertTrue(item["blocks_analysis"])
            else:
                self.assertFalse(item["blocks_analysis"])

    def test_blocking_issue_implies_validation_error(self):
        def check(mutate):
            sheets = {k: v.copy() for k, v in sample_workbook().items()}
            mutate(sheets)
            raw = {k: v.copy() for k, v in sheets.items()}
            norm = normalize_loaded_data(sheets)
            has_blocking = collect_data_issues(norm, raw)["summary"]["has_blocking"]
            if has_blocking:
                self.assertTrue(validate_workbook_data(norm).has_errors)

        def nonnum(s):
            s["inventory"]["stock_qty"] = s["inventory"]["stock_qty"].astype(object)
            s["inventory"].loc[0, "stock_qty"] = "십오"

        def blank(s):
            s["inventory"]["store_id"] = s["inventory"]["store_id"].astype(object)
            s["inventory"].loc[0, "store_id"] = ""

        def zero(s):
            s["recommendations"].loc[0, "recommended_qty"] = 0

        def sst(s):
            s["recommendations"].loc[0, "target_id"] = s["recommendations"].loc[0, "source_id"]

        for mutate in (nonnum, blank, zero, sst):
            check(mutate)

    def test_excluded_rows_equal_error_rows_and_usable_math(self):
        inv = pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": "십오"},
            {"store_id": "S2", "product_id": "P1", "stock_qty": 10},
        ])
        summary = collect_data_issues({"inventory": inv})["summary"]
        self.assertEqual(summary["excluded_rows"], summary["error_rows"])
        self.assertEqual(summary["usable_rows"], summary["total_rows"] - summary["excluded_rows"])

    def test_mostly_excluded_file_is_blocking(self):
        rows = [{"store_id": "S{}".format(i), "product_id": "P1", "stock_qty": "십오"} for i in range(6)]
        rows += [{"store_id": "S{}".format(i), "product_id": "P1", "stock_qty": 10} for i in range(6, 10)]
        summary = collect_data_issues({"inventory": pd.DataFrame(rows)})["summary"]
        self.assertGreater(summary["excluded_rows"] / summary["total_rows"], HEAVY_EXCLUSION_RATIO)
        self.assertTrue(summary["mostly_excluded"])
        self.assertTrue(summary["has_blocking"])

    def test_direct_pipeline_gate_still_blocks_bad_data(self):
        sheets = {k: v.copy() for k, v in sample_workbook().items()}
        sheets["recommendations"].loc[0, "target_id"] = sheets["recommendations"].loc[0, "source_id"]
        report = validate_workbook_data(normalize_loaded_data(sheets))
        self.assertTrue(report.has_errors)


# ── F. 상태 분리 ─────────────────────────────────────────────────────────────
class StateSeparationTests(unittest.TestCase):
    def test_unusable_pending_keeps_raw_snapshot_and_is_not_applied(self):
        sheets = {k: v.copy() for k, v in sample_workbook().items()}
        sheets["recommendations"].loc[0, "target_id"] = sheets["recommendations"].loc[0, "source_id"]
        state: dict = {}
        ok = load_and_apply(state, workbook_excel_bytes(sheets), "bad.xlsx", "업로드된 추천 결과")
        self.assertFalse(ok)
        self.assertIsNone(state.get("varo_data"))
        self.assertTrue(state.get("pending_raw_data"))          # raw snapshot kept for 오류 추적
        self.assertEqual(state["pending_source_metadata"]["source_type"], "excel")

    def test_apply_stores_raw_snapshot_and_metadata(self):
        state: dict = {}
        ok = load_and_apply(state, workbook_excel_bytes(), "good.xlsx", "업로드된 추천 결과")
        self.assertTrue(ok)
        self.assertTrue(state.get("raw_data"))
        self.assertEqual(state["source_metadata"]["filename"], "good.xlsx")
        # pending is cleared once applied
        self.assertIsNone(state.get("pending_raw_data"))

    def test_new_file_resets_previous_pending_raw(self):
        sheets = {k: v.copy() for k, v in sample_workbook().items()}
        sheets["recommendations"].loc[0, "target_id"] = sheets["recommendations"].loc[0, "source_id"]
        state: dict = {}
        load_and_apply(state, workbook_excel_bytes(sheets), "bad.xlsx", "업로드된 추천 결과")
        self.assertTrue(state.get("pending_raw_data"))
        # a clean file applies and clears the failed pending snapshot
        load_and_apply(state, workbook_excel_bytes(), "good.xlsx", "업로드된 추천 결과")
        self.assertIsNone(state.get("pending_raw_data"))


# ── G. UI / CSV ──────────────────────────────────────────────────────────────
class DisplayAndExportTests(unittest.TestCase):
    def _issues(self):
        inv = pd.DataFrame([{"점포id": "S1", "상품코드": "P1", "현재고": "십오"}])
        return collect_data_issues(
            {"inventory": inv}, {"inventory": inv},
            {"filename": "t.xlsx", "source_type": "excel", "sheet_names": {"inventory": "inventory"}},
        )["issues"]

    def test_display_rows_have_only_user_facing_columns(self):
        rows = display_rows(self._issues())
        self.assertEqual(set(rows[0]), {"시트", "행", "컬럼", "값", "구분", "문제", "수정 방법"})
        for internal in ("code", "canonical_column_name", "source_type", "blocks_analysis"):
            self.assertNotIn(internal, rows[0])

    def test_detail_rows_add_standard_column_and_block_flag(self):
        rows = detail_rows(self._issues())
        self.assertIn("표준 컬럼", rows[0])
        self.assertIn("정규화 값", rows[0])
        self.assertIn("분석 차단", rows[0])

    def test_csv_is_utf8_bom_with_original_location(self):
        csv = issues_to_csv_bytes(self._issues())
        self.assertTrue(csv.startswith(b"\xef\xbb\xbf"))
        text = csv.decode("utf-8-sig")
        for header in ("파일명", "원본 행 번호", "원본 컬럼명", "입력값", "수정 방법", "관련 행"):
            self.assertIn(header, text)
        self.assertIn("현재고", text)   # original column
        self.assertIn("십오", text)     # original value (full)

    def test_csv_has_no_internal_leakage(self):
        text = issues_to_csv_bytes(self._issues()).decode("utf-8-sig")
        for leak in ("Traceback", "DataFrame", "astype", "canonical_column_name", "\\Projects\\"):
            self.assertNotIn(leak, text)

    def test_empty_issue_list_csv_is_header_only(self):
        csv = issues_to_csv_bytes([])
        self.assertTrue(csv.startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
