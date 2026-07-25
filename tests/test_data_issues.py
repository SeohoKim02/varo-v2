"""Row-level data issue collection tests."""
from __future__ import annotations

import unittest

import pandas as pd

from services.data_issues import collect_data_issues, issues_to_csv_bytes


def _codes(issues):
    return {item["code"] for item in issues}


class CollectDataIssuesTests(unittest.TestCase):
    def test_clean_data_has_no_issues(self):
        data = {
            "inventory": pd.DataFrame([
                {"store_id": "S1", "product_id": "P1", "stock_qty": 10},
                {"store_id": "S2", "product_id": "P1", "stock_qty": 20},
            ]),
            "recommendations": pd.DataFrame([
                {"route_id": "R1", "product_id": "P1", "source_id": "S1", "target_id": "S2", "recommended_qty": 5},
            ]),
        }
        out = collect_data_issues(data)
        self.assertEqual(out["summary"]["total_issues"], 0)
        self.assertFalse(out["summary"]["has_blocking"])

    def test_non_numeric_and_negative_and_missing_id(self):
        data = {"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": "십오"},   # non_numeric error
            {"store_id": "", "product_id": "P2", "stock_qty": -3},          # missing_id error + negative warning
        ])}
        out = collect_data_issues(data)
        codes = _codes(out["issues"])
        self.assertIn("non_numeric", codes)
        self.assertIn("missing_id", codes)
        self.assertIn("negative", codes)
        self.assertTrue(out["summary"]["has_blocking"])

    def test_zero_and_same_source_target_in_recommendations(self):
        data = {"recommendations": pd.DataFrame([
            {"route_id": "R1", "product_id": "P1", "source_id": "S1", "target_id": "S1", "recommended_qty": 0},
        ])}
        codes = _codes(collect_data_issues(data)["issues"])
        self.assertIn("zero_quantity", codes)
        self.assertIn("same_source_target", codes)

    def test_conflict_duplicate_same_key_different_value(self):
        data = {"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 10},
            {"store_id": "S1", "product_id": "P1", "stock_qty": 20},  # conflict: warning, not merged
        ])}
        out = collect_data_issues(data)
        self.assertIn("conflict_duplicate", _codes(out["issues"]))
        # a value conflict is a warning (확인 필요), never silently merged
        self.assertFalse(out["summary"]["has_blocking"])

    def test_row_numbers_are_one_based_with_header(self):
        data = {"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": "bad"},  # first data row -> sheet row 2
        ])}
        issue = collect_data_issues(data)["issues"][0]
        self.assertEqual(issue["행"], 2)

    def test_summary_counts_and_top_limit(self):
        rows = [{"store_id": "S1", "product_id": "P1", "stock_qty": "x"} for _ in range(10)]
        out = collect_data_issues({"inventory": pd.DataFrame(rows)})
        self.assertEqual(out["summary"]["error_rows"], 10)
        self.assertLessEqual(len(out["summary"]["top"]), 5)

    def test_csv_export_is_utf8_bom(self):
        out = collect_data_issues({"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": "x"}])})
        csv = issues_to_csv_bytes(out["issues"])
        self.assertTrue(csv.startswith(b"\xef\xbb\xbf"))  # BOM for Excel
        self.assertIn("문제".encode("utf-8"), csv)


if __name__ == "__main__":
    unittest.main()
