"""Tests for column alias mapping and numeric/blank normalization."""
from __future__ import annotations

import unittest

import pandas as pd

from services import column_aliases as ca


class CleanNumericTests(unittest.TestCase):
    def test_currency_and_units_are_stripped(self):
        self.assertEqual(ca.clean_numeric_value("10,000원"), 10000.0)
        self.assertEqual(ca.clean_numeric_value("₩1,234.5"), 1234.5)
        self.assertEqual(ca.clean_numeric_value("3.5km"), 3.5)
        self.assertEqual(ca.clean_numeric_value("15분"), 15.0)
        self.assertEqual(ca.clean_numeric_value("100개"), 100.0)

    def test_blank_and_invalid_become_none(self):
        for value in ("", "  ", "nan", "N/A", "-", "abc", None):
            self.assertIsNone(ca.clean_numeric_value(value))

    def test_numeric_passthrough(self):
        self.assertEqual(ca.clean_numeric_value(42), 42.0)
        self.assertEqual(ca.clean_numeric_value(-5.5), -5.5)
        self.assertIsNone(ca.clean_numeric_value(float("nan")))


class AliasMappingTests(unittest.TestCase):
    def test_korean_recommendation_aliases(self):
        df = pd.DataFrame([{
            "추천id": "R1", "상품코드": "P1", "상품명": "만두",
            "출발점포": "A", "도착점포": "B", "경로유형": "direct",
            "추천수량": "5", "절감액": "10,000원", "기존VHS": "80",
        }])
        norm, applied = ca.normalize_columns(df, ca.RECOMMENDATION_ALIASES)
        for standard in ("route_id", "product_id", "product_name", "source_name",
                         "target_name", "route_type", "recommended_qty",
                         "expected_saving", "vhs_score"):
            self.assertIn(standard, norm.columns)
        self.assertGreaterEqual(len(applied), 8)

    def test_english_aliases_and_no_op_on_standard(self):
        df = pd.DataFrame([{"from_id": "A", "to_id": "B", "cost": 100, "distance": 3}])
        norm, applied = ca.normalize_columns(df, ca.ROUTE_ALIASES)
        self.assertIn("source_id", norm.columns)
        self.assertIn("target_id", norm.columns)
        self.assertIn("estimated_cost", norm.columns)
        self.assertIn("distance_km", norm.columns)
        # already-standard frame yields no new mappings
        standard = pd.DataFrame([{"source_id": "A", "target_id": "B"}])
        _, applied2 = ca.normalize_columns(standard, ca.ROUTE_ALIASES)
        self.assertEqual(applied2, [])


class NumericColumnCoercionTests(unittest.TestCase):
    def test_coerce_counts_failures(self):
        df = pd.DataFrame({"stock_qty": ["10", "20,000", "bad", ""]})
        coerced, failed = ca.coerce_numeric_columns(df, ("stock_qty",))
        self.assertEqual(coerced["stock_qty"].tolist()[:2], [10.0, 20000.0])
        self.assertTrue(pd.isna(coerced["stock_qty"].iloc[2]))
        self.assertEqual(failed, 1)  # "bad" is the only non-empty failure


class BlankRowTests(unittest.TestCase):
    def test_blank_rows_removed(self):
        df = pd.DataFrame([{"a": 1, "b": 2}, {"a": None, "b": ""}, {"a": 3, "b": 4}])
        result, removed = ca.drop_blank_rows(df)
        self.assertEqual(removed, 1)
        self.assertEqual(len(result), 2)


class DateConversionTests(unittest.TestCase):
    def setUp(self):
        from datetime import date
        self.ref = date(2026, 6, 20)

    def test_date_formats_to_days(self):
        self.assertEqual(ca.days_from_date("2026-06-30", self.ref), 10)
        self.assertEqual(ca.days_from_date("2026.07.05", self.ref), 15)
        self.assertEqual(ca.days_from_date("2026/07/01", self.ref), 11)
        self.assertEqual(ca.days_from_date("2026-06-10", self.ref), -10)  # expired allowed

    def test_unparseable_date_returns_none_without_crash(self):
        self.assertIsNone(ca.days_from_date("not a date", self.ref))
        self.assertIsNone(ca.days_from_date("", self.ref))
        self.assertIsNone(ca.days_from_date(None, self.ref))

    def test_normalize_date_columns_computes_days(self):
        df = pd.DataFrame([
            {"store_id": "S1", "만료일": "2026-06-30"},
            {"store_id": "S2", "만료일": "broken"},
        ])
        out, ok, fail, cols = ca.normalize_date_columns(df, self.ref)
        self.assertEqual(cols, ["만료일"])
        self.assertEqual(ok, 1)
        self.assertEqual(fail, 1)
        self.assertEqual(out["days_to_expiry"].iloc[0], 10.0)

    def test_existing_days_to_expiry_takes_priority(self):
        df = pd.DataFrame([{"days_to_expiry": 7, "만료일": "2026-12-31"}])
        out, ok, fail, _ = ca.normalize_date_columns(df, self.ref)
        self.assertEqual(out["days_to_expiry"].iloc[0], 7)
        self.assertEqual(ok, 0)


if __name__ == "__main__":
    unittest.main()
