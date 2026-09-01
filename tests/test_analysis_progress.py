"""Contract for the recommendation-run progress feedback.

Two things must hold at once:

* the user gets real, stage-based feedback while the (blocking) analysis runs, and
* **nothing about the analysis result changes because of it** — with or without a
  progress callback the recommendations are byte-identical, and a callback that
  raises must not disturb the run.

``ResultRegressionTests`` pins the recommendation output of the current
``ALGORITHM_VERSION`` on the anonymized operational workbook, so any accidental
change to the ranking shows up here (docs/ALGORITHM_BENCHMARK.md records the one
deliberate change and why).
"""
from __future__ import annotations

import inspect
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

try:
    from streamlit.testing.v1 import AppTest

    _APPTEST_AVAILABLE = True
except Exception:  # pragma: no cover - older streamlit
    _APPTEST_AVAILABLE = False

from services import analysis_pipeline, data_application
from services.analysis_pipeline import run_analysis_pipeline, sort_recommendations
from services.analysis_progress import (
    ANALYSIS_STAGES, FAILED_STAGE, ProgressEvent, ProgressReporter, stage_order,
)
from services.app_state import ANALYSIS_RUN_KEYS, apply_state_payload
from services.data_application import (
    commit_pending_data, prepare_pending_data, run_applied_analysis,
)
from tests.fixtures import sample_workbook, workbook_excel_bytes
from tools.generate_anonymized_operational_workbook import WORKBOOK_NAME, generate

SOURCE_TYPE = "업로드 데이터"

# Internal vocabulary that must never reach a user-facing progress line.
FORBIDDEN_TERMS = (
    "candidate_generator", "feasibility", "annotate", "normalization", "ledger",
    "signature", "DataFrame", "callback", "pipeline", "phase", "VHS normalization",
    "candidate_id", "session_state", "Traceback",
)

_TEMP_DIR: Path | None = None


def setUpModule() -> None:  # noqa: N802 - unittest hook
    global _TEMP_DIR
    _TEMP_DIR = Path(tempfile.mkdtemp(prefix="varo_progress_"))
    generate(_TEMP_DIR)


def tearDownModule() -> None:  # noqa: N802 - unittest hook
    if _TEMP_DIR is not None:
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)


def operational_workbook() -> Path:
    assert _TEMP_DIR is not None
    return _TEMP_DIR / WORKBOOK_NAME


def fixture_state() -> dict[str, Any]:
    """A small applied workspace built through the real intake services."""
    state: dict[str, Any] = {}
    prepare_pending_data(state, workbook_excel_bytes(sample_workbook()), "progress.xlsx", SOURCE_TYPE)
    assert commit_pending_data(state), state.get("data_apply_error")
    return state


def operational_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    prepare_pending_data(state, str(operational_workbook()), WORKBOOK_NAME, SOURCE_TYPE)
    assert commit_pending_data(state), state.get("data_apply_error")
    return state


def _comparable(recommendations: list[dict]) -> str:
    """Stable text form of the final recommendation set for exact comparison."""
    return json.dumps(
        [
            {key: str(value) for key, value in sorted(item.items()) if key != "calculated_at"}
            for item in sort_recommendations(recommendations)
        ],
        ensure_ascii=False, sort_keys=True,
    )


# --------------------------------------------------------------------------- #
# A. pipeline callback
# --------------------------------------------------------------------------- #
class PipelineCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture_state()["varo_data"]

    def test_result_is_identical_with_and_without_a_callback(self):
        without = run_analysis_pipeline(self.data)
        events: list[ProgressEvent] = []
        with_callback = run_analysis_pipeline(self.data, progress_callback=events.append)
        self.assertTrue(events)
        self.assertEqual(with_callback.status, without.status)
        self.assertEqual(
            _comparable(with_callback.recommendations), _comparable(without.recommendations)
        )
        self.assertEqual(
            with_callback.ledger_summary["status_counts"],
            without.ledger_summary["status_counts"],
        )

    def test_events_follow_the_real_pipeline_order_and_end_with_complete(self):
        events: list[ProgressEvent] = []
        run_analysis_pipeline(self.data, progress_callback=events.append)
        self.assertEqual([event.stage for event in events], stage_order())
        self.assertEqual(events[-1].stage, "complete")
        self.assertEqual(events[-1].progress, 1.0)
        self.assertTrue(all(event.ok for event in events))

    def test_progress_never_moves_backwards_and_stays_in_range(self):
        events: list[ProgressEvent] = []
        run_analysis_pipeline(self.data, progress_callback=events.append)
        values = [event.progress for event in events]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_messages_are_plain_korean_without_internal_vocabulary(self):
        for _code, message, _progress in ANALYSIS_STAGES:
            self.assertTrue(message.endswith("."), msg=message)
            self.assertLessEqual(len(message), 40, msg=message)
            for term in FORBIDDEN_TERMS:
                self.assertNotIn(term.lower(), message.lower(), msg=message)

    def test_no_stage_claims_that_dqn_is_running(self):
        """DQN stays a manual, opt-in feature: the run must never advertise it."""
        for _code, message, _progress in ANALYSIS_STAGES:
            self.assertNotIn("DQN", message.upper(), msg=message)
            self.assertNotIn("학습", message, msg=message)

    def test_failure_paths_report_a_failed_event_instead_of_complete(self):
        empty_events: list[ProgressEvent] = []
        run_analysis_pipeline({}, progress_callback=empty_events.append)
        self.assertEqual([event.stage for event in empty_events], [FAILED_STAGE])
        self.assertFalse(empty_events[0].ok)

        broken = {key: frame.copy() for key, frame in self.data.items()}
        broken["stores"] = broken["stores"].drop(columns=["node_type"])
        events: list[ProgressEvent] = []
        result = run_analysis_pipeline(broken, progress_callback=events.append)
        self.assertEqual(result.status, "validation_error")
        self.assertEqual(events[-1].stage, FAILED_STAGE)
        self.assertNotIn("complete", [event.stage for event in events])

    def test_a_raising_callback_cannot_change_the_result(self):
        def exploding(_event: ProgressEvent) -> None:
            raise RuntimeError("UI 오류")

        baseline = run_analysis_pipeline(self.data)
        result = run_analysis_pipeline(self.data, progress_callback=exploding)
        self.assertEqual(result.status, baseline.status)
        self.assertEqual(_comparable(result.recommendations), _comparable(baseline.recommendations))

    def test_reporter_is_a_no_op_without_a_callback(self):
        reporter = ProgressReporter(None)
        self.assertFalse(reporter.enabled)
        reporter("validation")
        reporter.fail()  # must not raise


# --------------------------------------------------------------------------- #
# B. 진행 표시 위젯 (Streamlit 없이 계약만 검증)
# --------------------------------------------------------------------------- #
class _StubBar:
    def __init__(self) -> None:
        self.calls: list[tuple[float, str]] = []

    def progress(self, value: float, text: str = "") -> None:
        self.calls.append((value, text))


class _StubStatus:
    def __init__(self, label: str, expanded: bool) -> None:
        self.label = label
        self.expanded = expanded
        self.state = "running"
        self.bar = _StubBar()

    def progress(self, value: float, text: str = "") -> _StubBar:
        self.bar.progress(value, text)
        return self.bar

    def update(self, label: str = "", state: str = "", expanded: bool = True) -> None:
        self.label, self.state, self.expanded = label, state, expanded


class _StubContainer:
    def __init__(self) -> None:
        self.status_block: _StubStatus | None = None

    def status(self, label: str, expanded: bool = False) -> _StubStatus:
        self.status_block = _StubStatus(label, expanded)
        return self.status_block


class ProgressViewTests(unittest.TestCase):
    def _view(self):
        from components.analysis_progress import AnalysisProgressView

        container = _StubContainer()
        return container, AnalysisProgressView(container)

    def test_view_shows_the_current_step_for_every_stage(self):
        container, view = self._view()
        for index, (code, message, progress) in enumerate(ANALYSIS_STAGES, start=1):
            view.callback(ProgressEvent(code, message, progress, index, len(ANALYSIS_STAGES)))
        shown = [text for _value, text in container.status_block.bar.calls]
        for _code, message, _progress in ANALYSIS_STAGES:
            self.assertIn(message, shown)
        values = [value for value, _text in container.status_block.bar.calls]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[-1], 1.0)

    def test_view_collapses_into_a_short_outcome_on_success(self):
        container, view = self._view()
        view.finish(True, 4.23)
        block = container.status_block
        self.assertEqual(block.state, "complete")
        self.assertFalse(block.expanded)
        self.assertIn("4.2초", block.label)
        self.assertNotIn("4.23", block.label)

    def test_short_runs_do_not_advertise_a_duration(self):
        container, view = self._view()
        view.finish(True, 0.4)
        self.assertNotIn("초", container.status_block.label)

    def test_view_reports_failure_without_internal_detail(self):
        container, view = self._view()
        view.callback(ProgressEvent(FAILED_STAGE, "추천을 완료하지 못했습니다.", 0.4, 0, 8, False))
        view.finish(False, None)
        block = container.status_block
        self.assertEqual(block.state, "error")
        self.assertFalse(block.expanded)
        for term in FORBIDDEN_TERMS:
            self.assertNotIn(term.lower(), block.label.lower())

    def test_completion_note_only_mentions_a_measured_duration(self):
        from components.analysis_progress import completion_note

        self.assertEqual(completion_note(None), "추천 계산이 완료됐습니다.")
        self.assertEqual(completion_note(0.2), "추천 계산이 완료됐습니다.")
        self.assertIn("4.2초", completion_note(4.21))


# --------------------------------------------------------------------------- #
# C. 실행 중 상태
# --------------------------------------------------------------------------- #
class RunStateTests(unittest.TestCase):
    def test_running_is_set_during_the_run_and_cleared_on_success(self):
        state = fixture_state()
        seen: list[bool] = []
        run_applied_analysis(
            state, progress_callback=lambda _event: seen.append(bool(state.get("analysis_running")))
        )
        self.assertTrue(seen and all(seen), msg="실행 중 상태가 표시되지 않았습니다")
        self.assertNotIn("analysis_running", state)
        self.assertTrue(state["analysis_completed_notice"])
        self.assertGreater(state["analysis_elapsed_seconds"], 0)

    def test_running_is_cleared_after_a_failure_and_the_run_can_be_retried(self):
        state = fixture_state()
        with mock.patch.object(
            data_application, "run_analysis_pipeline", side_effect=RuntimeError("boom")
        ):
            self.assertFalse(run_applied_analysis(state))
        self.assertNotIn("analysis_running", state)
        self.assertNotIn("analysis_completed_notice", state)
        self.assertTrue(state["analysis_run_required"], msg="실패 후 다시 실행할 수 없습니다")
        self.assertIn("analysis_run_error", state)
        # Previous applied data survives and a retry works.
        self.assertTrue(run_applied_analysis(state))
        self.assertNotIn("analysis_running", state)
        self.assertNotIn("analysis_run_error", state)

    def test_missing_data_does_not_leave_a_running_flag(self):
        state: dict[str, Any] = {}
        self.assertFalse(run_applied_analysis(state))
        self.assertNotIn("analysis_running", state)

    def test_applying_new_data_drops_any_stale_run_state(self):
        state = fixture_state()
        self.assertTrue(run_applied_analysis(state))
        state["analysis_running"] = True  # simulate a value left behind
        prepare_pending_data(
            state, workbook_excel_bytes(sample_workbook()), "again.xlsx", "다른 업로드",
        )
        apply_state_payload(state, {"varo_data": state["varo_data"]})
        for key in ANALYSIS_RUN_KEYS:
            self.assertNotIn(key, state, msg=f"{key}가 데이터 적용 후에도 남아 있습니다")

    def test_clearing_applied_data_drops_run_state(self):
        from services.app_state import clear_applied_data

        state = fixture_state()
        self.assertTrue(run_applied_analysis(state))
        state["analysis_running"] = True
        clear_applied_data(state)
        for key in ANALYSIS_RUN_KEYS:
            self.assertNotIn(key, state)


# --------------------------------------------------------------------------- #
# D. 결과 회귀 (진행 UI 도입 전 측정값)
# --------------------------------------------------------------------------- #
class ResultRegressionTests(unittest.TestCase):
    """Recommendation output pinned for the current algorithm version.

    Measured on the anonymized operational workbook. These numbers move only when
    the scoring logic deliberately changes (and then ``ALGORITHM_VERSION`` moves
    with them); they must never move because of progress reporting, which is what
    ``test_running_with_progress_matches_running_without_it`` guards.
    """

    BASELINE_TOP = {
        "route_id": "R0005",
        "product_name": "가상상품 03",
        "source_name": "가상점포 09",
        "target_name": "가상점포 14",
        "route_type": "DIRECT",
        "dc_id": None,
        "recommended_qty": 60.0,
        "estimated_cost": 8500.0,
        "expected_saving": 203180.0,
        # vhs-2.2 removes inventory-floor violations before VHS normalization;
        # the winner and every operational value stay unchanged, while its score
        # is recomputed against the executable candidate population only.
        "vhs_score": 84.81,
        "confidence": 66.0,
    }
    BASELINE_COUNTS = {
        "generated": 62,
        "recommendable_total": 58,
        "excluded_total": 4,
        "status_counts": {
            "추천": 1, "추천 가능": 57, "확인 필요": 0,
            "이동 불가": 4, "데이터 부족": 0, "계산 불가": 0,
        },
    }

    @classmethod
    def setUpClass(cls):
        cls.state = operational_state()
        cls.events: list[ProgressEvent] = []
        assert run_applied_analysis(cls.state, progress_callback=cls.events.append)

    def test_top_recommendation_is_unchanged(self):
        top = sort_recommendations(self.state["varo_recommendations"])[0]
        for field, expected in self.BASELINE_TOP.items():
            self.assertEqual(top.get(field), expected, msg=field)
        self.assertEqual(str(self.state["selected_route_id"]), self.BASELINE_TOP["route_id"])

    def test_candidate_counts_are_unchanged(self):
        summary = self.state["varo_pipeline_result"]["ledger_summary"]
        for field, expected in self.BASELINE_COUNTS.items():
            self.assertEqual(summary[field], expected, msg=field)
        self.assertEqual(len(self.state["varo_recommendations"]), 58)

    def test_running_with_progress_matches_running_without_it(self):
        plain = run_analysis_pipeline(self.state["varo_data"])
        self.assertEqual(
            _comparable(plain.recommendations),
            _comparable(self.state["varo_recommendations"]),
        )


# --------------------------------------------------------------------------- #
# E. 성능
# --------------------------------------------------------------------------- #
class ProgressCostTests(unittest.TestCase):
    def test_event_count_is_fixed_and_does_not_scale_with_data(self):
        small: list[ProgressEvent] = []
        run_analysis_pipeline(fixture_state()["varo_data"], progress_callback=small.append)
        large_state = operational_state()
        large: list[ProgressEvent] = []
        run_analysis_pipeline(large_state["varo_data"], progress_callback=large.append)
        self.assertEqual(len(small), len(ANALYSIS_STAGES))
        self.assertEqual(len(large), len(ANALYSIS_STAGES))
        candidates = len(large_state["varo_data"]["recommendations"])
        inventory_rows = len(large_state["varo_data"]["inventory"])
        self.assertGreater(candidates, len(large))
        self.assertGreater(inventory_rows, len(large))

    def test_no_sleep_or_artificial_delay_was_introduced(self):
        import components.analysis_progress as progress_view
        import services.analysis_progress as progress_service

        for module in (progress_service, progress_view, analysis_pipeline, data_application):
            source = inspect.getsource(module)
            self.assertNotIn("time.sleep", source, msg=module.__name__)
            self.assertNotIn("sleep(", source, msg=module.__name__)

    def test_a_small_dataset_is_not_slowed_down_by_progress_reporting(self):
        data = fixture_state()["varo_data"]
        run_analysis_pipeline(data)  # warm the legacy module cache
        start = time.perf_counter()
        run_analysis_pipeline(data)
        plain = time.perf_counter() - start
        start = time.perf_counter()
        run_analysis_pipeline(data, progress_callback=lambda _event: None)
        reported = time.perf_counter() - start
        # Deliberately loose: this catches an added delay, not machine variation.
        self.assertLess(reported, plain * 2.0 + 1.0)

    def test_services_do_not_import_streamlit(self):
        for module in (analysis_pipeline, data_application):
            source = inspect.getsource(module)
            self.assertNotIn("import streamlit", source, msg=module.__name__)
            self.assertNotIn("st.progress", source, msg=module.__name__)
            self.assertNotIn("st.status", source, msg=module.__name__)
        import services.analysis_progress as progress_service

        self.assertNotIn("streamlit", inspect.getsource(progress_service))


# --------------------------------------------------------------------------- #
# B-2. 실제 추천 실행 페이지에 연결됐는지 (AppTest)
# --------------------------------------------------------------------------- #
class _RecordingView:
    """Stands in for the real progress view to prove the page wires it up."""

    instances: list["_RecordingView"] = []

    def __init__(self, _container: Any = None) -> None:
        self.events: list[ProgressEvent] = []
        self.finished: tuple[bool, Any] | None = None
        _RecordingView.instances.append(self)

    def callback(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def finish(self, succeeded: bool, elapsed_seconds: Any = None) -> None:
        self.finished = (succeeded, elapsed_seconds)


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class RecommendationPageProgressTests(unittest.TestCase):
    """Clicks the real 추천 실행 button and inspects the rendered page."""

    app_path = str(Path(__file__).resolve().parents[1] / "app_v2.py")

    def _app(self):
        app = AppTest.from_file(self.app_path, default_timeout=300)
        app.run()
        staged: dict[str, Any] = {}
        prepare_pending_data(
            staged, workbook_excel_bytes(sample_workbook()), "progress.xlsx", "업로드된 추천 결과",
        )
        for key, value in staged.items():
            app.session_state[key] = value
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        app.button(key="apply_pending").click().run()
        app.session_state["current_menu"] = "추천 실행"
        app.run()
        return app

    def _text(self, app) -> str:
        parts = [element.value for element in app.markdown]
        for attribute in ("caption", "info", "success", "warning", "error"):
            if hasattr(app, attribute):
                parts += [element.value for element in getattr(app, attribute)]
        return " ".join(str(part) for part in parts)

    def test_run_button_is_present_and_enabled_before_the_run(self):
        app = self._app()
        keys = {button.key for button in app.button}
        self.assertIn("run_applied_analysis", keys)
        self.assertFalse(dict(app.session_state.filtered_state).get("analysis_running"))
        self.assertEqual(len(dict(app.session_state.filtered_state)["varo_recommendations"]), 0)

    def test_clicking_run_drives_every_real_stage_through_the_progress_view(self):
        app = self._app()
        _RecordingView.instances = []
        import pages.recommendations as page

        with mock.patch.object(page, "AnalysisProgressView", _RecordingView):
            app.button(key="run_applied_analysis").click().run()
        self.assertEqual(len(_RecordingView.instances), 1, msg="진행 표시가 연결되지 않았습니다")
        view = _RecordingView.instances[0]
        self.assertEqual([event.stage for event in view.events], stage_order())
        self.assertEqual(view.finished[0], True)
        state = dict(app.session_state.filtered_state)
        self.assertGreater(len(state["varo_recommendations"]), 0)
        self.assertNotIn("analysis_running", state)

    def test_completion_note_appears_once_and_results_render(self):
        app = self._app()
        app.button(key="run_applied_analysis").click().run()
        first = self._text(app)
        self.assertIn("추천 계산이 완료됐습니다", first)
        self.assertFalse(app.exception)
        app.run()  # a plain rerun must not repeat the notice
        second = self._text(app)
        self.assertNotIn("추천 계산이 완료됐습니다", second)
        self.assertNotIn("추천 실행 전입니다", second)

    def test_finished_page_keeps_no_progress_widget_and_no_internal_terms(self):
        app = self._app()
        app.button(key="run_applied_analysis").click().run()
        blob = self._text(app)
        self.assertNotIn("분석을 시작하고 있습니다", blob)
        for _code, message, _progress in ANALYSIS_STAGES[:-1]:
            self.assertNotIn(message, blob, msg="완료 후에도 진행 문구가 남아 있습니다")
        for term in ("Traceback", "candidate_id", "session_state", "run_applied_analysis",
                     "analysis_running", "progress_callback"):
            self.assertNotIn(term, blob, msg=f"{term}이 화면에 노출됐습니다")


if __name__ == "__main__":
    unittest.main()
