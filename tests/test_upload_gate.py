"""End-to-end upload gate + state-separation tests (env-independent).

Uses a plain dict as session state and in-memory workbooks, so it exercises the
real load_and_apply / pipeline gate without Streamlit or external files.
"""
from __future__ import annotations

import io
import unittest
import warnings

import pandas as pd

from services.analysis_pipeline import run_analysis_pipeline
from services.data_application import load_and_apply
from tests.fixtures import sample_workbook, workbook_excel_bytes


def _canonical_applied(state) -> bool:
    return bool(state.get("varo_data")) and bool(state.get("varo_recommendations"))


class UnsupportedAndCorruptGateTests(unittest.TestCase):
    def test_unsupported_extension_is_blocked_and_not_applied(self):
        state: dict = {}
        ok = load_and_apply(state, io.BytesIO(b"whatever"), "data.pdf", "업로드된 추천 결과")
        self.assertFalse(ok)
        self.assertIn("지원하지 않는", state.get("pending_load_error", ""))
        self.assertFalse(_canonical_applied(state))

    def test_empty_file_is_blocked(self):
        state: dict = {}
        ok = load_and_apply(state, io.BytesIO(b""), "data.xlsx", "업로드된 추천 결과")
        self.assertFalse(ok)
        self.assertFalse(_canonical_applied(state))


class ValidationErrorGateTests(unittest.TestCase):
    def _errored_workbook_bytes(self):
        sheets = sample_workbook()
        stores = sheets["stores"].copy()
        stores["node_type"] = "마감임박형"  # not DC/STORE -> validation error
        sheets["stores"] = stores
        return workbook_excel_bytes(sheets)

    def test_validation_error_data_is_not_applied(self):
        state: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok = load_and_apply(state, self._errored_workbook_bytes(), "bad.xlsx", "업로드된 추천 결과")
        self.assertFalse(ok)
        self.assertFalse(_canonical_applied(state))
        # kept as a pending (failed) candidate with its validation, never applied
        self.assertIsNotNone(state.get("pending_varo_validation"))

    def test_pipeline_direct_call_also_gates_invalid_data(self):
        # Direct function call must not analyze invalid data (not just the UI button).
        data = sample_workbook()
        stores = data["stores"].copy()
        stores["node_type"] = "마감임박형"
        data["stores"] = stores
        result = run_analysis_pipeline(data)
        self.assertEqual(result.status, "validation_error")
        self.assertEqual(result.recommendations, [])


class StateSeparationTests(unittest.TestCase):
    def _apply(self, state, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return load_and_apply(state, workbook_excel_bytes(), filename, "업로드된 추천 결과")

    def test_reapply_and_signature_and_dqn_reset(self):
        state: dict = {"dqn_training_result": {"status": "정상", "data_signature": "old"}}
        self.assertTrue(self._apply(state, "first.xlsx"))
        self.assertTrue(_canonical_applied(state))
        first_sig = state.get("data_signature")
        self.assertTrue(first_sig)
        # applying data clears any prior DQN result so a stale result is not reused
        self.assertIsNone(state.get("dqn_training_result"))

        # A different workbook (extra saving) changes the signature.
        sheets = sample_workbook()
        recs = sheets["recommendations"].copy()
        recs.loc[0, "expected_saving"] = 999999
        sheets["recommendations"] = recs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_and_apply(state, workbook_excel_bytes(sheets), "second.xlsx", "업로드된 추천 결과")
        self.assertNotEqual(state.get("data_signature"), first_sig)


if __name__ == "__main__":
    unittest.main()
