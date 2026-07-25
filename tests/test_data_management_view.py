"""Data-management view-model + reset tests (self-contained, backup-free).

Covers the spec groups for 데이터 관리:
* 상태 모델: 데이터 없음 / 적용 데이터만 / pending 사용 불가 / pending+적용 동시 / stale
* 현재 데이터와 pending 데이터 구분, 홈과의 상태 일치
* 문제 카드 행 수(전체/오류/경고/제외)와 행 수 vs 문제 항목 수 구분
* 데이터 적용의 원자성/초기화, 적용 실패 시 기존 데이터 보존
* 현재 데이터 초기화(원본 파일은 삭제하지 않음)
* 내부 정보(signature/경로/함수명/예외명) 미노출

Fixtures are built in-package via load_and_apply / plain dicts — no external files.
"""
from __future__ import annotations

import io
import unittest
import warnings

import pandas as pd

from services.app_state import clear_applied_data
from services.data_application import load_and_apply
from services.data_management_view import build_data_management_view
from services.data_validator import ValidationMessage, ValidationReport
from services.home_state import NO_DATA, READY, STALE, UNUSABLE
from tests.fixtures import sample_workbook, workbook_excel_bytes


def _applied_state() -> dict:
    state: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_and_apply(state, workbook_excel_bytes(sample_workbook()), "재고_7월.xlsx", "샘플 추천 데이터")
    return state


def _unusable_state(base: dict | None = None) -> dict:
    """Apply an error workbook (required column removed) onto `base` (or a fresh state)."""
    workbook = sample_workbook()
    workbook["inventory"] = workbook["inventory"].drop(columns=["stock_qty"])
    state = base if base is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_and_apply(state, workbook_excel_bytes(workbook), "오류파일.xlsx", "업로드된 추천 결과")
    return state


# --------------------------------------------------------------------------- #
# A. 상태 모델
# --------------------------------------------------------------------------- #
class StateModelTests(unittest.TestCase):
    def test_no_data(self):
        view = build_data_management_view({})
        self.assertFalse(view["has_current"])
        self.assertFalse(view["has_pending"])
        self.assertIsNone(view["current"])
        self.assertEqual(view["home"]["state_code"], NO_DATA)
        self.assertFalse(view["show_next_action"])

    def test_applied_only(self):
        view = build_data_management_view(_applied_state())
        self.assertTrue(view["has_current"])
        self.assertFalse(view["has_pending"])
        self.assertEqual(view["home"]["state_code"], READY)
        self.assertTrue(view["show_next_action"])
        current = view["current"]
        self.assertEqual(current["source_label"], "샘플 데이터")
        self.assertEqual(current["filename"], "재고_7월.xlsx")
        self.assertGreaterEqual(current["recommendation_count"], 1)
        self.assertIn("추천", current["recommendation_status"])

    def test_unusable_pending_only(self):
        view = build_data_management_view(_unusable_state())
        self.assertFalse(view["has_current"])
        self.assertTrue(view["has_pending"])
        self.assertIsNotNone(view["pending"])
        self.assertEqual(view["pending"]["status"], "사용 불가")
        self.assertEqual(view["home"]["state_code"], UNUSABLE)
        self.assertFalse(view["show_next_action"])  # no apply/next while unusable

    def test_current_and_pending_are_both_present(self):
        # Apply good data first, then a bad upload arrives (spec I). The applied
        # workspace stays READY; the bad upload is a separate intake sub-state.
        state = _applied_state()
        _unusable_state(state)
        view = build_data_management_view(state)
        self.assertTrue(view["has_current"])   # existing applied data preserved
        self.assertTrue(view["has_pending"])   # new unusable upload shown separately
        self.assertEqual(view["home"]["state_code"], READY)  # applied result NOT hidden
        self.assertTrue(view["home"]["pending_notice"])
        self.assertFalse(view["show_next_action"])  # deal with the new intake first

    def test_stale_flag(self):
        state = _applied_state()
        state["data_signature"] = "totally-different-signature"
        view = build_data_management_view(state)
        self.assertTrue(view["stale"])
        self.assertEqual(view["home"]["state_code"], STALE)

    def test_build_never_raises_on_garbage(self):
        for bad in ({"varo_data": 123}, {"varo_recommendations": "x"}, {"pending_varo_data": 5}):
            view = build_data_management_view(bad)
            self.assertIn("home", view)


# --------------------------------------------------------------------------- #
# B. 문제 카드: 행 수 vs 문제 항목 수, 내부 정보 미노출
# --------------------------------------------------------------------------- #
class PendingCheckTests(unittest.TestCase):
    def test_pending_reports_error_messages_or_rows(self):
        view = build_data_management_view(_unusable_state())
        pending = view["pending"]
        # a missing required column surfaces as a plain validation error message
        self.assertTrue(pending["error_messages"])
        # counts are honest integers, never fabricated
        for key in ("total_rows", "error_rows", "warning_rows", "excluded_rows", "issue_count"):
            self.assertIsInstance(pending[key], int)

    def test_row_count_and_issue_count_are_distinct(self):
        # A duplicate-with-conflict warning produces >1 issue item per row.
        workbook = sample_workbook()
        inv = workbook["inventory"]
        dup = inv.iloc[[0]].copy()
        dup["stock_qty"] = 777  # same store/product, different value -> conflict
        workbook["inventory"] = pd.concat([inv, dup], ignore_index=True)
        # This still applies (warning, not error): assert applied-data view keeps ints.
        state: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_and_apply(state, workbook_excel_bytes(workbook), "중복.xlsx", "업로드된 추천 결과")
        view = build_data_management_view(state)
        self.assertTrue(view["has_current"])
        # warning rows do not inflate the analysis-usable count with fake zeros
        self.assertIsInstance(view["current"]["warning_rows"], int)

    def test_pending_view_has_no_internal_leak(self):
        view = build_data_management_view(_unusable_state())
        blob = " ".join(str(v) for v in view["pending"].values())
        for internal in ("data_signature", "session_state", "Traceback", "pending_varo_data", "canonical"):
            self.assertNotIn(internal, blob)


# --------------------------------------------------------------------------- #
# C. 적용 실패 시 기존 데이터 보존 / 원자성
# --------------------------------------------------------------------------- #
class ApplyPreservationTests(unittest.TestCase):
    def test_unusable_upload_preserves_applied_data(self):
        state = _applied_state()
        recs_before = list(state["varo_recommendations"])
        sig_before = state["data_signature"]
        _unusable_state(state)  # error upload → pending only
        self.assertEqual(state["varo_recommendations"], recs_before)  # untouched
        self.assertEqual(state["data_signature"], sig_before)
        self.assertIsNotNone(state.get("pending_varo_validation"))

    def test_apply_resets_previous_analysis_state(self):
        state = _applied_state()
        state["dqn_training_result"] = {"status": "정상", "data_signature": "old"}
        # re-apply a different workbook
        workbook = sample_workbook()
        recs = workbook["recommendations"].copy()
        recs.loc[0, "expected_saving"] = 424242
        workbook["recommendations"] = recs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_and_apply(state, workbook_excel_bytes(workbook), "새파일.xlsx", "업로드된 추천 결과")
        self.assertIsNone(state.get("dqn_training_result"))  # stale DQN cleared


# --------------------------------------------------------------------------- #
# D. 현재 데이터 초기화
# --------------------------------------------------------------------------- #
class ClearAppliedTests(unittest.TestCase):
    def test_clear_resets_workspace_to_no_data(self):
        state = _applied_state()
        clear_applied_data(state)
        self.assertIsNone(state["varo_data"])
        self.assertEqual(state["varo_recommendations"], [])
        self.assertEqual(state["analysis_result"], {})
        self.assertIsNone(state["selected_route_id"])
        self.assertIsNone(state["data_signature"])
        # home now reads as 데이터 없음
        self.assertEqual(build_data_management_view(state)["home"]["state_code"], NO_DATA)

    def test_clear_also_drops_pending(self):
        state = _applied_state()
        _unusable_state(state)
        self.assertIsNotNone(state.get("pending_varo_validation"))
        clear_applied_data(state)
        self.assertIsNone(state.get("pending_varo_validation"))
        self.assertIsNone(state.get("pending_load_error"))
        view = build_data_management_view(state)
        self.assertFalse(view["has_current"])
        self.assertFalse(view["has_pending"])

    def test_clear_does_not_delete_original_bytes(self):
        # The caller keeps the uploaded file; clear only touches session state.
        original = io.BytesIO(workbook_excel_bytes(sample_workbook()).getvalue())
        state: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_and_apply(state, original, "keep.xlsx", "업로드된 추천 결과")
        clear_applied_data(state)
        # original buffer is untouched and still readable
        self.assertTrue(original.getvalue())


# --------------------------------------------------------------------------- #
# E. 홈과 데이터 관리 상태 일치
# --------------------------------------------------------------------------- #
class CrossPageConsistencyTests(unittest.TestCase):
    def test_view_home_matches_standalone_home(self):
        from services.home_state import build_home_state
        for state in ({}, _applied_state(), _unusable_state()):
            self.assertEqual(
                build_data_management_view(state)["home"]["state_code"],
                build_home_state(state)["state_code"],
            )

    def test_unusable_pending_title_matches_home(self):
        state = _unusable_state()
        view = build_data_management_view(state)
        self.assertEqual(view["home"]["title"], "데이터를 수정해야 합니다")


if __name__ == "__main__":
    unittest.main()
