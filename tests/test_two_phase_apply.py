"""Two-phase intake tests: inspect (prepare) vs explicit apply (commit).

Covers spec groups A(분리)·B(정상 적용)·C(경고 적용)·D(사용 불가)·E(적용 실패)·
F(signature)·H(상태 분리)·I(초기화). All fixtures are built in-package from
tests.fixtures — no external files.
"""
from __future__ import annotations

import io
import unittest
import warnings
from unittest import mock

import pandas as pd

from services.app_state import has_app_data
from services.data_application import (
    cancel_pending_data,
    commit_pending_data,
    load_and_apply,
    prepare_pending_data,
)
from services.home_state import READY, build_home_state
from tests.fixtures import sample_workbook, workbook_excel_bytes


def _valid_bytes(saving: int | None = None):
    workbook = sample_workbook()
    if saving is not None:
        recs = workbook["recommendations"].copy()
        recs.loc[0, "expected_saving"] = saving  # changes the content signature
        workbook["recommendations"] = recs
    return workbook_excel_bytes(workbook)


def _warning_bytes():
    workbook = sample_workbook()
    routes = workbook["routes"]
    # duplicate source/target combo → WARNING (분석 가능, 오류 아님)
    workbook["routes"] = pd.concat([routes, routes.iloc[[0]]], ignore_index=True)
    return workbook_excel_bytes(workbook)


def _unusable_bytes():
    workbook = sample_workbook()
    workbook["inventory"] = workbook["inventory"].drop(columns=["stock_qty"])  # required → ERROR
    return workbook_excel_bytes(workbook)


def _applied_state() -> dict:
    state: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_and_apply(state, _valid_bytes(), "current.xlsx", "샘플 추천 데이터")
    return state


def _prepare(state, source, name="new.xlsx"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare_pending_data(state, source, name, "업로드된 추천 결과")


# --------------------------------------------------------------------------- #
# A. 검사와 적용 분리
# --------------------------------------------------------------------------- #
class InspectDoesNotApplyTests(unittest.TestCase):
    def test_valid_upload_inspects_only(self):
        state: dict = {}
        status = _prepare(state, _valid_bytes())
        self.assertEqual(status, "사용 가능")
        self.assertIsNone(state.get("varo_data"))          # NOT applied
        self.assertFalse(state.get("varo_recommendations"))
        self.assertTrue(state.get("pending_varo_data"))     # pending only
        self.assertTrue(state.get("pending_data_signature"))
        self.assertTrue(state.get("pending_apply_allowed"))

    def test_warning_upload_inspects_only(self):
        state: dict = {}
        status = _prepare(state, _warning_bytes())
        self.assertEqual(status, "확인 필요")
        self.assertIsNone(state.get("varo_data"))
        self.assertTrue(state.get("pending_apply_allowed"))

    def test_unusable_upload_inspects_only(self):
        state: dict = {}
        status = _prepare(state, _unusable_bytes())
        self.assertEqual(status, "사용 불가")
        self.assertIsNone(state.get("varo_data"))
        self.assertFalse(state.get("pending_apply_allowed"))

    def test_inspect_preserves_existing_applied_data(self):
        state = _applied_state()
        recs_before = list(state["varo_recommendations"])
        sig_before = state["data_signature"]
        _prepare(state, _unusable_bytes())            # bad new file
        self.assertEqual(state["varo_recommendations"], recs_before)
        self.assertEqual(state["data_signature"], sig_before)
        _prepare(state, _valid_bytes(saving=555))     # good new file, still not applied
        self.assertEqual(state["varo_recommendations"], recs_before)
        self.assertEqual(state["data_signature"], sig_before)


# --------------------------------------------------------------------------- #
# B/C. 정상·경고 적용
# --------------------------------------------------------------------------- #
class CommitTests(unittest.TestCase):
    def test_commit_valid_applies_and_clears_pending(self):
        state: dict = {}
        _prepare(state, _valid_bytes())
        self.assertTrue(commit_pending_data(state))
        self.assertTrue(has_app_data(state.get("varo_data"), state.get("varo_recommendations")))
        self.assertIsNone(state.get("pending_varo_data"))     # pending cleared
        self.assertTrue(state.get("data_apply_message"))
        self.assertEqual(build_home_state(state)["state_code"], READY)

    def test_commit_warning_applies_with_usable_rows(self):
        state: dict = {}
        status = _prepare(state, _warning_bytes())
        self.assertEqual(status, "확인 필요")
        self.assertGreaterEqual(int(state.get("pending_usable_rows") or 0), 1)
        self.assertTrue(commit_pending_data(state))
        self.assertTrue(state.get("varo_recommendations"))


# --------------------------------------------------------------------------- #
# D. 사용 불가 데이터
# --------------------------------------------------------------------------- #
class UnusableBlockTests(unittest.TestCase):
    def test_commit_unusable_is_blocked_and_preserves_current(self):
        state = _applied_state()
        recs_before = list(state["varo_recommendations"])
        _prepare(state, _unusable_bytes())
        self.assertFalse(commit_pending_data(state))          # apply blocked
        self.assertEqual(state["varo_recommendations"], recs_before)  # current kept
        self.assertTrue(state.get("data_apply_error"))
        self.assertNotIn("Traceback", state["data_apply_error"])

    def test_commit_without_pending_is_expired(self):
        state: dict = {}
        self.assertFalse(commit_pending_data(state))
        self.assertIn("만료", state.get("data_apply_error", ""))


# --------------------------------------------------------------------------- #
# E. 적용 실패 시 기존 상태 보존 (exception path)
# --------------------------------------------------------------------------- #
class CommitFailurePreservationTests(unittest.TestCase):
    def test_pipeline_exception_preserves_current_and_pending(self):
        state = _applied_state()
        recs_before = list(state["varo_recommendations"])
        sig_before = state["data_signature"]
        _prepare(state, _valid_bytes(saving=777))     # a different valid file
        with mock.patch(
            "services.data_application.run_analysis_pipeline", side_effect=RuntimeError("boom")
        ):
            self.assertFalse(commit_pending_data(state))
        # current applied data + recs preserved
        self.assertEqual(state["varo_recommendations"], recs_before)
        self.assertEqual(state["data_signature"], sig_before)
        # pending kept so the user can retry
        self.assertTrue(state.get("pending_varo_data"))
        # plain message, no internal exception text
        message = state.get("data_apply_error", "")
        self.assertTrue(message)
        for internal in ("RuntimeError", "boom", "Traceback", "run_analysis_pipeline"):
            self.assertNotIn(internal, message)


# --------------------------------------------------------------------------- #
# F. signature 처리
# --------------------------------------------------------------------------- #
class SignatureTests(unittest.TestCase):
    def test_same_content_reupload_is_marked_same(self):
        # Same *bytes* (a real re-upload of the same file) → identical signature.
        raw = _valid_bytes().getvalue()
        state: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_and_apply(state, io.BytesIO(raw), "current.xlsx", "샘플 추천 데이터")
        _prepare(state, io.BytesIO(raw), name="current_again.xlsx")
        self.assertEqual(state.get("pending_status"), "현재 데이터와 동일")

    def test_commit_same_signature_does_not_reset_results(self):
        raw = _valid_bytes().getvalue()
        state: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_and_apply(state, io.BytesIO(raw), "current.xlsx", "샘플 추천 데이터")
        # tag a derived result, then re-apply identical content
        state["dqn_training_result"] = {"status": "정상", "data_signature": state["data_signature"]}
        _prepare(state, io.BytesIO(raw), name="current_again.xlsx")
        self.assertTrue(commit_pending_data(state))
        # identical content → no unnecessary wipe of the existing derived result
        self.assertIsNotNone(state.get("dqn_training_result"))
        self.assertIsNone(state.get("pending_varo_data"))

    def test_different_content_resets_prior_analysis(self):
        state = _applied_state()
        state["selected_route_id"] = state["varo_recommendations"][0]["route_id"]
        state["dqn_training_result"] = {"status": "정상", "data_signature": "old"}
        state["simulation_snapshot"] = {"frame": 3}
        _prepare(state, _valid_bytes(saving=987654))
        self.assertTrue(commit_pending_data(state))
        self.assertIsNone(state.get("dqn_training_result"))
        self.assertIsNone(state.get("simulation_snapshot"))
        # selected candidate re-resolved to the new result's top (never a stale id)
        valid_ids = {str(r["route_id"]) for r in state["varo_recommendations"]}
        self.assertIn(str(state.get("selected_route_id")), valid_ids)

    def test_same_filename_different_content_is_new_pending(self):
        state = _applied_state()
        sig_before = state["data_signature"]
        _prepare(state, _valid_bytes(saving=13579), name="current.xlsx")  # same name, new content
        self.assertNotEqual(state.get("pending_data_signature"), sig_before)
        self.assertNotEqual(state.get("pending_status"), "현재 데이터와 동일")


# --------------------------------------------------------------------------- #
# H. workspace 상태 vs pending 상태 분리
# --------------------------------------------------------------------------- #
class StateSeparationTests(unittest.TestCase):
    def test_applied_plus_pending_keeps_workspace_ready(self):
        for source in (_valid_bytes(saving=1), _warning_bytes(), _unusable_bytes()):
            state = _applied_state()
            _prepare(state, source)
            home = build_home_state(state)
            self.assertEqual(home["state_code"], READY)   # applied result preserved
            self.assertTrue(home["pending_notice"])        # a new intake exists

    def test_recommendations_use_applied_not_pending(self):
        state = _applied_state()
        applied_recs = list(state["varo_recommendations"])
        _prepare(state, _valid_bytes(saving=222))          # different pending
        # the canonical recommendations the 추천 실행 page reads are still the applied set
        self.assertEqual(state["varo_recommendations"], applied_recs)


# --------------------------------------------------------------------------- #
# cancel
# --------------------------------------------------------------------------- #
class CancelTests(unittest.TestCase):
    def test_cancel_drops_pending_keeps_applied(self):
        state = _applied_state()
        recs_before = list(state["varo_recommendations"])
        _prepare(state, _valid_bytes(saving=42))
        cancel_pending_data(state)
        self.assertIsNone(state.get("pending_varo_data"))
        self.assertIsNone(state.get("pending_load_error"))
        self.assertEqual(state["varo_recommendations"], recs_before)


if __name__ == "__main__":
    unittest.main()
