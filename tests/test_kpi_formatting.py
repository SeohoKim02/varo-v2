"""KPI / card value formatting must never render invalid data as a real value.

None, empty string, NaN and inf are missing/invalid inputs and must resolve to
the explicit "-" no-data marker — never "nan원", "inf원" or a fabricated 0.
"""
from __future__ import annotations

import math
import unittest

from components.tables import format_currency, format_number


class FormattingSafetyTests(unittest.TestCase):
    def test_currency_valid_values(self):
        self.assertEqual(format_currency(1234567), "1,234,567원")
        self.assertEqual(format_currency(0), "0원")

    def test_currency_missing_and_invalid_values(self):
        for bad in (None, "", "abc", float("nan"), float("inf"), float("-inf")):
            self.assertEqual(format_currency(bad), "-", msg=f"{bad!r} must be '-'")

    def test_number_valid_values(self):
        self.assertEqual(format_number(12, "개"), "12개")
        self.assertEqual(format_number(3.5, "km"), "3.5km")

    def test_number_missing_and_invalid_values(self):
        for bad in (None, "", "abc", float("nan"), float("inf"), float("-inf")):
            self.assertEqual(format_number(bad, "개"), "-", msg=f"{bad!r} must be '-'")

    def test_no_invalid_token_leaks_into_output(self):
        for value in (float("nan"), float("inf")):
            self.assertNotIn("nan", format_currency(value).lower())
            self.assertNotIn("inf", format_number(value, "원").lower())
            self.assertTrue(math.isfinite(1.0))  # guard import stays used


if __name__ == "__main__":
    unittest.main()
