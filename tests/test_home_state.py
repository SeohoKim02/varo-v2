"""Home workspace-status tests: one resolved state, correct priority, honest KPIs.

Covers spec groups A(상태 모델)·B(우선순위)·C(KPI)·D(행동 버튼 대상)·
E(후보 없음 원인)·F(상태 변경/ signature). Fixtures are built in-package via
load_and_apply or plain dicts (no external files).
"""
from __future__ import annotations

import unittest

from services.data_application import load_and_apply
from services.data_validator import ValidationMessage, ValidationReport
from services.home_state import (
    FAILED,
    NO_CANDIDATES,
    NO_DATA,
    PAGE_DATA,
    PAGE_RECOMMENDATIONS,
    PAGE_ROUTE_DETAIL,
    READY,
    STALE,
    UNUSABLE,
    build_home_state,
)
from tests.fixtures import sample_workbook, workbook_excel_bytes


def _applied_state() -> dict:
    state: dict = {}
    load_and_apply(state, workbook_excel_bytes(sample_workbook()), "sample.xlsx", "샘플 추천 데이터")
    return state


def _all_blocked_state() -> dict:
    workbook = sample_workbook()
    recs = workbook["recommendations"].copy()
    recs["recommended_qty"] = 999999  # every move exceeds source stock → all blocked
    workbook["recommendations"] = recs
    state: dict = {}
    load_and_apply(state, workbook_excel_bytes(workbook), "blocked.xlsx", "업로드된 추천 결과")
    return state


def _unusable_state() -> dict:
    workbook = sample_workbook()
    workbook["inventory"] = workbook["inventory"].drop(columns=["stock_qty"])  # required col
    state: dict = {}
    load_and_apply(state, workbook_excel_bytes(workbook), "unusable.xlsx", "업로드된 추천 결과")
    return state


# --------------------------------------------------------------------------- #
# A. State model
# --------------------------------------------------------------------------- #
class StateModelTests(unittest.TestCase):
    def test_no_data(self):
        home = build_home_state({})
        self.assertEqual(home["state_code"], NO_DATA)
        self.assertEqual(home["next_page"], PAGE_DATA)
        self.assertFalse(home["show_result_kpis"])

    def test_ready(self):
        home = build_home_state(_applied_state())
        self.assertEqual(home["state_code"], READY)
        self.assertTrue(home["show_result_kpis"])
        self.assertEqual(home["recommendation_count"], 4)
        self.assertIsNotNone(home["top_recommendation"])
        self.assertEqual(home["next_page"], PAGE_ROUTE_DETAIL)

    def test_no_candidates(self):
        home = build_home_state(_all_blocked_state())
        self.assertEqual(home["state_code"], NO_CANDIDATES)
        self.assertFalse(home["show_result_kpis"])
        self.assertEqual(home["recommendation_count"], 0)
        self.assertTrue(home["no_candidate_cause"])
        self.assertEqual(home["next_page"], PAGE_RECOMMENDATIONS)

    def test_unusable_pending(self):
        home = build_home_state(_unusable_state())
        self.assertEqual(home["state_code"], UNUSABLE)
        self.assertEqual(home["data_status"], "사용 불가")
        self.assertEqual(home["next_page"], PAGE_DATA)
        self.assertFalse(home["show_result_kpis"])

    def test_pending_load_error_is_unusable(self):
        home = build_home_state({"pending_load_error": "파일을 처리할 수 없습니다."})
        self.assertEqual(home["state_code"], UNUSABLE)

    def test_failed_analysis(self):
        state = {
            "varo_data": {"stores": _applied_state()["varo_data"]["stores"]},
            "varo_recommendations": [],
            "varo_pipeline_result": {
                "status": "adapter_error",
                "ledger_summary": {"generated": 0},
                "diagnostics": {"algorithm_errors": [{"algorithm": "x", "error_type": "ValueError"}]},
            },
        }
        home = build_home_state(state)
        self.assertEqual(home["state_code"], FAILED)
        self.assertEqual(home["analysis_status"], "실패")
        self.assertFalse(home["show_result_kpis"])

    def test_stale_signature_hides_result(self):
        state = _applied_state()
        state["data_signature"] = "totally-different-signature"
        home = build_home_state(state)
        self.assertEqual(home["state_code"], STALE)
        self.assertFalse(home["show_result_kpis"])

    def test_selected_candidate_validity(self):
        state = _applied_state()
        valid_id = state["varo_recommendations"][0]["route_id"]
        state["selected_route_id"] = valid_id
        self.assertTrue(build_home_state(state)["selected_candidate_valid"])
        state["selected_route_id"] = "does-not-exist"
        self.assertFalse(build_home_state(state)["selected_candidate_valid"])


# --------------------------------------------------------------------------- #
# B. Priority
# --------------------------------------------------------------------------- #
class PriorityTests(unittest.TestCase):
    def test_unusable_pending_does_not_hide_applied_result(self):
        # Two-phase intake: a bad *new* upload while good data is applied is an
        # intake sub-state (shown in 데이터 관리), not an app-wide 사용 불가. The
        # already-applied result must stay visible.
        state = _applied_state()  # has old good recs
        state["pending_load_error"] = "새 파일을 처리할 수 없습니다."
        home = build_home_state(state)
        self.assertEqual(home["state_code"], READY)
        self.assertTrue(home["show_result_kpis"])
        self.assertTrue(home["pending_notice"])

    def test_unusable_pending_validation_does_not_hide_result(self):
        state = _applied_state()
        state["pending_varo_validation"] = ValidationReport(
            "오류", [ValidationMessage("오류", "inventory", "필수 컬럼 `stock_qty`이 없습니다.")]
        )
        self.assertEqual(build_home_state(state)["state_code"], READY)

    def test_unusable_pending_without_applied_data_is_unusable(self):
        # With no good applied data to fall back on, an unusable upload defines the
        # whole workspace (기존 규칙 유지).
        home = build_home_state({"pending_load_error": "형식 오류"})
        self.assertEqual(home["state_code"], UNUSABLE)

    def test_failed_and_no_candidates_are_distinct(self):
        self.assertEqual(build_home_state(_all_blocked_state())["state_code"], NO_CANDIDATES)

    def test_sample_vs_upload_source_label(self):
        sample = build_home_state(_applied_state())
        self.assertIn("샘플", sample["data_source"])
        upload_state = {}
        load_and_apply(upload_state, workbook_excel_bytes(sample_workbook()), "up.xlsx", "업로드된 추천 결과")
        self.assertNotIn("샘플", build_home_state(upload_state)["data_source"])


# --------------------------------------------------------------------------- #
# C. KPI honesty
# --------------------------------------------------------------------------- #
class KpiHonestyTests(unittest.TestCase):
    def test_no_result_kpis_before_result(self):
        for state in ({}, _all_blocked_state(), _unusable_state()):
            self.assertFalse(build_home_state(state)["show_result_kpis"])

    def test_ready_uses_ledger_top_recommendation(self):
        state = _applied_state()
        home = build_home_state(state)
        top = home["top_recommendation"]
        # matches the shared ranking's rank-1 (no re-sort in home_state)
        from services.analysis_pipeline import top_recommendations
        expected = top_recommendations(state["varo_recommendations"], limit=1)[0]
        self.assertEqual(top["route_id"], expected["route_id"])

    def test_no_data_has_no_fake_counts(self):
        home = build_home_state({})
        self.assertEqual(home["recommendation_count"], 0)
        self.assertIsNone(home["top_recommendation"])
        self.assertIsNone(home["confidence_status"])


# --------------------------------------------------------------------------- #
# D. Action targets
# --------------------------------------------------------------------------- #
class ActionTargetTests(unittest.TestCase):
    def test_action_pages_are_valid_menu_items(self):
        from components.navigation import MENU_ITEMS
        for state in ({}, _applied_state(), _all_blocked_state(), _unusable_state()):
            home = build_home_state(state)
            self.assertIn(home["next_page"], MENU_ITEMS)
            self.assertTrue(home["next_action_label"])

    def test_no_candidates_routes_to_recommendations(self):
        self.assertEqual(build_home_state(_all_blocked_state())["next_page"], PAGE_RECOMMENDATIONS)

    def test_ready_detail_action_targets_route_detail(self):
        self.assertEqual(build_home_state(_applied_state())["next_page"], PAGE_ROUTE_DETAIL)


# --------------------------------------------------------------------------- #
# E. No-candidate cause
# --------------------------------------------------------------------------- #
class NoCandidateCauseTests(unittest.TestCase):
    def test_all_blocked_cause_mentions_count_and_reason(self):
        home = build_home_state(_all_blocked_state())
        cause = home["no_candidate_cause"]
        self.assertIn("후보", cause)
        # no internal reason codes leak into the cause text
        for internal in ("reason_code", "quantity_exceeds_stock", "feasibility"):
            self.assertNotIn(internal, cause)

    def test_cause_is_safe_when_summary_missing(self):
        state = {
            "varo_data": {"stores": _applied_state()["varo_data"]["stores"]},
            "varo_recommendations": [],
            "varo_pipeline_result": {"status": "success", "ledger_summary": {"generated": 0}},
        }
        home = build_home_state(state)
        self.assertEqual(home["state_code"], NO_CANDIDATES)
        self.assertTrue(home["no_candidate_cause"])


# --------------------------------------------------------------------------- #
# F. State change / signature
# --------------------------------------------------------------------------- #
class StateChangeTests(unittest.TestCase):
    def test_pending_intake_does_not_change_applied_workspace(self):
        state = _applied_state()
        self.assertEqual(build_home_state(state)["state_code"], READY)
        # a new upload is inspected (pending) but not applied → workspace unchanged
        state["pending_load_error"] = "형식 오류"
        home = build_home_state(state)
        self.assertEqual(home["state_code"], READY)
        self.assertTrue(home["pending_notice"])

    def test_matching_signature_keeps_ready(self):
        state = _applied_state()
        # data_signature equals the analysed signature → stays READY
        self.assertEqual(build_home_state(state)["state_code"], READY)

    def test_build_never_raises_on_garbage(self):
        for bad in ({"varo_data": 123}, {"varo_recommendations": "x"}, {"varo_pipeline_result": []}):
            home = build_home_state(bad)
            self.assertIn(home["state_code"], {NO_DATA, UNUSABLE, FAILED, NO_CANDIDATES, READY, STALE})


# --------------------------------------------------------------------------- #
# G. Cross-page consistency (home ↔ validation share one source)
# --------------------------------------------------------------------------- #
class CrossPageConsistencyTests(unittest.TestCase):
    def test_home_and_validation_use_same_ledger_counts(self):
        state = _all_blocked_state()
        home = build_home_state(state)
        summary = state["varo_pipeline_result"]["ledger_summary"]
        # home 후보 수 = 추천 가능 후보(0); 검증 상태 카드 = 같은 ledger_summary
        self.assertEqual(home["state_code"], NO_CANDIDATES)
        self.assertEqual(home["recommendation_count"], 0)
        self.assertEqual(summary["recommendable_total"], 0)
        # 상태 버킷 합 == 전체 생성 후보 (검증 카드가 쓰는 값과 동일)
        total = (summary["recommendable_total"] + summary["check_needed"]
                 + summary["blocked_move"] + summary["insufficient_data"] + summary["not_computable"])
        self.assertEqual(total, summary["generated"])

    def test_home_cause_uses_same_top_exclusion_reasons(self):
        state = _all_blocked_state()
        home = build_home_state(state)
        reasons = state["varo_pipeline_result"]["ledger_summary"]["top_exclusion_reasons"]
        self.assertTrue(reasons)
        # the home cause is built from the same top reason the validation card lists
        top_reason = str(reasons[0]["reason"]).rstrip(".")
        self.assertIn(top_reason, home["no_candidate_cause"])

    def test_home_and_validation_agree_on_ready(self):
        state = _applied_state()
        self.assertEqual(build_home_state(state)["state_code"], READY)
        # a READY workspace has a non-empty ledger the validation tabs render from
        self.assertTrue(state["varo_pipeline_result"]["candidate_ledger"])


if __name__ == "__main__":
    unittest.main()
