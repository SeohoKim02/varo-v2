"""Safety contract for row-exclusion partial application."""
from __future__ import annotations

import unittest

import pandas as pd

from services.app_state import has_applied_data
from services.data_application import commit_pending_data, prepare_pending_data
from services.data_issues import (
    FILE_BLOCKING, INFORMATIONAL, ROW_EXCLUDABLE, ROW_WARNING,
    collect_data_issues, issue_policy, issues_to_csv_bytes,
)
from services.data_management_view import build_data_management_view
from services.partial_data import build_usable_data, usable_data_signature
from tests.fixtures import sample_workbook, workbook_excel_bytes


def _copy_workbook() -> dict[str, pd.DataFrame]:
    return {key: frame.copy(deep=True) for key, frame in sample_workbook().items()}


def _build(workbook: dict[str, pd.DataFrame]):
    raw = {key: frame.copy(deep=True) for key, frame in workbook.items()}
    metadata = {"filename": "fixture.xlsx", "source_type": "excel", "sheet_names": {
        key: "v2_recommendations" if key == "recommendations" else key for key in raw
    }}
    return build_usable_data(workbook, raw, metadata)


class PolicyClassificationTests(unittest.TestCase):
    def test_four_treatments_and_unknown_fail_closed(self):
        self.assertEqual(issue_policy("alias_conflict").treatment, FILE_BLOCKING)
        self.assertEqual(issue_policy("negative").treatment, ROW_EXCLUDABLE)
        self.assertEqual(issue_policy("id_numeric").treatment, ROW_WARNING)
        self.assertEqual(issue_policy("blank_rows_removed").treatment, INFORMATIONAL)
        unknown = issue_policy("future_unclassified_problem")
        self.assertEqual(unknown.treatment, FILE_BLOCKING)
        self.assertTrue(unknown.blocks_analysis)

    def test_same_severity_can_have_different_treatment(self):
        self.assertEqual(issue_policy("alias_conflict").severity, issue_policy("negative").severity)
        self.assertNotEqual(issue_policy("alias_conflict").treatment, issue_policy("negative").treatment)


class RowExclusionTests(unittest.TestCase):
    def test_negative_inventory_row_is_removed_and_normal_rows_remain(self):
        workbook = _copy_workbook()
        workbook["inventory"].loc[0, "stock_qty"] = -1
        result = _build(workbook)
        self.assertTrue(result["apply_allowed"])
        self.assertEqual(result["quality_summary"]["excluded_rows"], 1)
        self.assertEqual(len(result["usable_data"]["inventory"]), len(workbook["inventory"]) - 1)
        self.assertFalse((result["usable_data"]["inventory"]["stock_qty"] < 0).any())

    def test_non_numeric_missing_id_and_multiple_issues_count_unique_rows(self):
        workbook = _copy_workbook()
        workbook["inventory"]["stock_qty"] = workbook["inventory"]["stock_qty"].astype(object)
        workbook["inventory"].loc[0, ["store_id", "stock_qty"]] = ["", "not-a-number"]
        result = _build(workbook)
        self.assertEqual(result["quality_summary"]["excluded_rows"], 1)
        self.assertGreaterEqual(result["quality_summary"]["error_items"], 2)

    def test_bad_route_and_bad_recommendation_do_not_remove_good_paths(self):
        workbook = _copy_workbook()
        workbook["routes"]["estimated_cost"] = workbook["routes"]["estimated_cost"].astype(object)
        workbook["routes"].loc[0, "estimated_cost"] = "bad"
        result = _build(workbook)
        self.assertTrue(result["apply_allowed"])
        self.assertFalse((result["usable_data"]["routes"]["estimated_cost"].astype(str) == "bad").any())
        remaining_ids = set(result["usable_data"]["recommendations"]["route_id"].astype(str))
        self.assertNotIn("R001", remaining_ids)
        self.assertIn("R002", remaining_ids)  # VIA_DC path is independent

    def test_missing_dc_and_invalid_route_type_rows_are_excluded(self):
        workbook = _copy_workbook()
        extras = pd.concat([
            workbook["recommendations"].iloc[[0]].assign(route_id="R005"),
            workbook["recommendations"].iloc[[2]].assign(route_id="R006"),
        ], ignore_index=True)
        workbook["recommendations"] = pd.concat([workbook["recommendations"], extras], ignore_index=True)
        workbook["recommendations"].loc[1, "dc_id"] = None
        workbook["recommendations"].loc[2, "route_type"] = "UNKNOWN"
        result = _build(workbook)
        self.assertEqual(result["quality_summary"]["excluded_by_table"]["recommendations"], 2)
        self.assertTrue(result["apply_allowed"])


class DuplicateTests(unittest.TestCase):
    def test_exact_duplicate_keeps_first_and_excludes_later(self):
        workbook = _copy_workbook()
        workbook["inventory"] = pd.concat(
            [workbook["inventory"], workbook["inventory"].iloc[[0]]], ignore_index=True,
        )
        result = _build(workbook)
        self.assertEqual(result["quality_summary"]["excluded_rows"], 1)
        keys = result["usable_data"]["inventory"][["store_id", "product_id"]]
        self.assertFalse(keys.duplicated().any())

    def test_conflicting_duplicate_excludes_every_related_row(self):
        workbook = _copy_workbook()
        conflict = workbook["inventory"].iloc[[0]].copy()
        conflict["stock_qty"] = 999
        workbook["inventory"] = pd.concat([workbook["inventory"], conflict], ignore_index=True)
        result = _build(workbook)
        self.assertEqual(result["quality_summary"]["excluded_by_table"]["inventory"], 2)
        usable = result["usable_data"]["inventory"]
        pair = (usable["store_id"].astype(str) == "S001") & (usable["product_id"].astype(str) == "P001")
        self.assertFalse(pair.any())

    def test_time_series_rows_are_not_treated_as_duplicates(self):
        workbook = _copy_workbook()
        workbook["inventory"]["snapshot_date"] = "2026-01-01"
        later = workbook["inventory"].iloc[[0]].copy()
        later["snapshot_date"] = "2026-01-02"
        workbook["inventory"] = pd.concat([workbook["inventory"], later], ignore_index=True)
        result = _build(workbook)
        duplicate_codes = {item["code"] for item in result["issues"] if "duplicate" in item["code"]}
        self.assertFalse(duplicate_codes)


class RelationshipAndThresholdTests(unittest.TestCase):
    def test_bad_master_row_cascades_and_leaves_no_orphans(self):
        workbook = _copy_workbook()
        workbook["stores"].loc[3, "node_name"] = ""  # S003
        result = _build(workbook)
        usable = result["usable_data"]
        self.assertNotIn("S003", set(usable["stores"]["node_id"].astype(str)))
        self.assertNotIn("S003", set(usable["inventory"]["store_id"].astype(str)))
        self.assertNotIn("S003", set(usable["routes"]["source_id"].astype(str)))
        self.assertNotIn("S003", set(usable["routes"]["target_id"].astype(str)))

    def test_dc01_failure_does_not_remove_dc02_path(self):
        workbook = _copy_workbook()
        workbook["stores"] = pd.concat([workbook["stores"], pd.DataFrame([{
            "node_id": "DC02", "node_name": "보조센터", "store_name": "보조센터", "node_type": "DC",
        }])], ignore_index=True)
        extra_routes = pd.DataFrame([
            {"source_id": "S001", "target_id": "DC02", "distance_km": 2, "estimated_cost": 100, "travel_time_min": 3},
            {"source_id": "DC02", "target_id": "S003", "distance_km": 2, "estimated_cost": 100, "travel_time_min": 3},
            {"source_id": "S002", "target_id": "DC02", "distance_km": 2, "estimated_cost": 100, "travel_time_min": 3},
            {"source_id": "DC02", "target_id": "S002", "distance_km": 2, "estimated_cost": 100, "travel_time_min": 3},
        ])
        workbook["routes"] = pd.concat([workbook["routes"], extra_routes], ignore_index=True)
        rec = workbook["recommendations"].iloc[[1]].copy()
        rec["route_id"], rec["dc_id"], rec["dc_name"] = "R-DC02", "DC02", "보조센터"
        workbook["recommendations"] = pd.concat([workbook["recommendations"], rec], ignore_index=True)
        workbook["stores"].loc[0, "node_name"] = ""  # DC01 only
        result = _build(workbook)
        remaining = set(result["usable_data"]["recommendations"]["route_id"].astype(str))
        self.assertIn("R-DC02", remaining)
        self.assertNotIn("R002", remaining)

    def test_all_inventory_removed_is_unusable(self):
        workbook = _copy_workbook()
        workbook["inventory"]["stock_qty"] = -1
        result = _build(workbook)
        self.assertFalse(result["apply_allowed"])
        self.assertTrue(result["validation"].has_errors)

    def test_below_half_allowed_and_half_blocked_per_critical_table(self):
        below = _copy_workbook()
        below["products"] = pd.concat([below["products"], pd.DataFrame([{
            "product_id": "P003", "product_name": "", "unit_price": 100,
        }])], ignore_index=True)
        self.assertTrue(_build(below)["apply_allowed"])  # 1/3

        boundary = _copy_workbook()
        boundary["products"].loc[1, "product_name"] = ""
        result = _build(boundary)
        self.assertFalse(result["apply_allowed"])  # 1/2 == 0.5
        self.assertEqual(result["quality_summary"]["table_exclusion_ratios"]["products"], 0.5)


class WarningCountsAndExportTests(unittest.TestCase):
    def test_numeric_identifier_warning_is_retained(self):
        workbook = _copy_workbook()
        workbook["products"]["product_id"] = [1, 2]
        workbook["inventory"]["product_id"] = workbook["inventory"]["product_id"].map({"P001": 1, "P002": 2})
        workbook["recommendations"]["product_id"] = workbook["recommendations"]["product_id"].map({"P001": 1, "P002": 2})
        result = _build(workbook)
        self.assertEqual(result["quality_summary"]["excluded_rows"], 0)
        self.assertGreater(result["quality_summary"]["warning_included_rows"], 0)
        self.assertEqual(len(result["usable_data"]["inventory"]), len(workbook["inventory"]))

    def test_counts_are_consistent_and_csv_is_user_facing(self):
        workbook = _copy_workbook()
        workbook["inventory"].loc[0, "stock_qty"] = -1
        result = _build(workbook)
        quality = result["quality_summary"]
        self.assertEqual(quality["total_rows"], quality["applied_rows"] + quality["excluded_rows"])
        csv_text = issues_to_csv_bytes(result["issues"]).decode("utf-8-sig")
        self.assertIn("처리 결과", csv_text)
        self.assertIn("적용 데이터 포함 여부", csv_text)
        self.assertNotIn("issue_code", csv_text)
        self.assertNotIn("canonical", csv_text)


class CommitAndViewTests(unittest.TestCase):
    def _prepare(self, workbook):
        state: dict = {}
        prepare_pending_data(state, workbook_excel_bytes(workbook), "partial.xlsx", "업로드 데이터")
        return state

    def test_commit_uses_only_usable_data_and_does_not_auto_run(self):
        workbook = _copy_workbook()
        workbook["inventory"].loc[0, "stock_qty"] = -1
        state = self._prepare(workbook)
        raw_rows = len(state["pending_raw_data"]["inventory"])
        self.assertTrue(commit_pending_data(state))
        self.assertTrue(has_applied_data(state["varo_data"]))
        self.assertEqual(len(state["varo_data"]["inventory"]), raw_rows - 1)
        self.assertEqual(len(state["raw_data"]["inventory"]), raw_rows)
        self.assertEqual(state["data_signature"], usable_data_signature(state["varo_data"]))
        self.assertEqual(state["varo_recommendations"], [])
        self.assertTrue(state["analysis_run_required"])

    def test_signature_mutation_and_missing_usable_data_preserve_current(self):
        current = self._prepare(_copy_workbook())
        self.assertTrue(commit_pending_data(current))
        applied_signature = current["data_signature"]
        prepare_pending_data(current, workbook_excel_bytes(_copy_workbook()), "next.xlsx", "업로드 데이터")
        current["pending_usable_data"]["inventory"].loc[0, "stock_qty"] = 99999
        self.assertFalse(commit_pending_data(current))
        self.assertEqual(current["data_signature"], applied_signature)
        self.assertTrue(current.get("pending_data_issues") is not None)
        current["pending_usable_data"] = None
        self.assertFalse(commit_pending_data(current))
        self.assertEqual(current["data_signature"], applied_signature)

    def test_button_labels_match_real_exclusion_state(self):
        clean = self._prepare(_copy_workbook())
        self.assertEqual(build_data_management_view(clean)["pending"]["apply_label"], "이 데이터 사용")

        partial_book = _copy_workbook()
        partial_book["inventory"].loc[0, "stock_qty"] = -1
        partial = self._prepare(partial_book)
        view = build_data_management_view(partial)["pending"]
        self.assertEqual(view["apply_label"], "문제 행을 제외하고 사용")
        self.assertEqual(view["excluded_rows"], 1)
        self.assertLessEqual(len(view["top_exclusion_reasons"]), 3)
        self.assertTrue(view["excluded_preview"])

        blocked_book = _copy_workbook()
        blocked_book["inventory"] = blocked_book["inventory"].drop(columns=["stock_qty"])
        blocked = self._prepare(blocked_book)
        self.assertIsNone(build_data_management_view(blocked)["pending"]["apply_label"])
