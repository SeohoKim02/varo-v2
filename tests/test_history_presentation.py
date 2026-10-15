"""Display-only fixes must preserve history, input state and error visibility."""
from copy import deepcopy
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.elements.lib import policies

from pages import results_history as history
from components.status import user_status_label

APP = str(Path(__file__).resolve().parents[1] / "app_v2.py")
FILTER_APP = '''
import streamlit as st
from datetime import datetime
from components.navigation import _restore_inputs, _remember_inputs
from pages.results_history import _filter_history
_restore_inputs()
rows = [{"timestamp":datetime(2026,7,d), "strategy":"학습 결과", "status":"완료", "name":"상품", "id":str(d)} for d in (1,15,31)]
filtered = _filter_history(rows)
st.session_state["filtered_ids"] = [r["id"] for r in filtered]
_remember_inputs()
'''


def test_restored_period_has_no_duplicate_default_warning_and_keeps_filter():
    app = AppTest.from_string(FILTER_APP)
    app.session_state["history_period"] = (date(2026,7,1),date(2026,7,15))
    with patch.object(policies, "_shown_default_value_warning", False):
        app.run().run()
    assert not app.exception and not app.warning and not app.error
    assert app.session_state["filtered_ids"] == ["1", "15"]
    assert tuple(app.date_input[0].value) == (date(2026,7,1),date(2026,7,15))


@pytest.mark.parametrize("saved", [(), (date(2026,7,15),)])
def test_incomplete_date_range_requests_completion(saved):
    app = AppTest.from_string(FILTER_APP)
    app.session_state["history_period"] = saved
    app.run()
    assert not app.exception and not app.warning
    assert app.session_state["filtered_ids"] == []
    assert any("모두 선택" in item.value for item in app.info)


def test_stale_date_range_clamps_without_warning():
    app = AppTest.from_string(FILTER_APP)
    app.session_state["history_period"] = (date(2000,1,1),date(2000,1,2))
    app.run()
    assert not app.exception and not app.warning
    assert tuple(app.date_input[0].value) == (date(2026,7,1),date(2026,7,1))


def test_route_history_display_preserves_raw_values_and_distinguishes_reference(caplog):
    run = {"status":"FAILED", "final_strategy":"할인", "data_source_type":"V2 생성 후보",
           "error_summary":"ValueError: secret_internal_field\nTraceback", "run_id":"r", "total_expected_saving":0}
    routes = [{"action":"할인", "route_type":"DIRECT", "expected_saving":0, "executed":True}]
    before = deepcopy((run,routes))
    app = AppTest.from_string('from pages.results_history import _render_simulation_detail\nimport streamlit as st\n_render_simulation_detail(st.session_state["run"])')
    app.session_state["run"] = run
    with patch.object(history, "get_run_routes", return_value=routes):
        app.run()
    assert not app.exception
    text = " ".join(x.value for x in app.markdown)
    assert "기존 분석 판단(참고)" in text and "실패 안내" in text
    assert "ValueError" not in text and "secret_internal_field" not in text and "Traceback" not in text
    assert "secret_internal_field" in caplog.text
    frame = app.dataframe[0].value
    assert frame.iloc[0]["예상 절감액"] == "미산정 · 비교 근거 미기록"
    assert frame.iloc[0]["경로 유형"] == "직접 이동"
    assert frame.iloc[0]["기존 분석 판단(참고)"] == "할인"
    assert (run,routes) == before


@pytest.mark.parametrize("status, expected", [("BLOCKED","진행 불가 · 조건 확인 필요"), ("COMPLETED","완료"), ("PRIVATE_INTERNAL_CODE","상태 확인 필요")])
def test_internal_status_is_localized(status, expected):
    assert user_status_label(status) == expected


def test_page_failure_is_reported_not_disguised_and_logged(caplog):
    with patch.object(history, "list_simulation_runs", side_effect=RuntimeError("private_variable Traceback")):
        app = AppTest.from_file(APP).run()
        app.session_state["current_menu"] = "결과 이력"
        app.run()
    assert not app.exception
    assert any("화면을 불러오지 못했습니다" in x.value for x in app.error)
    assert all("private_variable" not in x.value for x in app.error)
    assert "private_variable" in caplog.text and "Traceback" in caplog.text


def test_corrupt_learning_history_is_visible_failure_and_logged(tmp_path,caplog):
    (tmp_path / "dqn_result_bad.json").write_text("broken", encoding="utf-8")
    history._saved_dqn_runs.clear()
    app=AppTest.from_string('from pages.results_history import _saved_dqn_runs\nimport streamlit as st\n_saved_dqn_runs(st.session_state["directory"])')
    app.session_state["directory"] = str(tmp_path)
    app.run()
    assert not app.exception
    assert any("읽지 못했습니다" in x.value for x in app.warning)
    assert "dqn_result_bad.json" in caplog.text


def test_failed_learning_result_is_not_a_success_badge(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("{}", encoding="utf-8")
    row = {"id":"one", "display_time":"2026-07-01", "strategy":"학습 결과", "name":"상품", "candidate_count":1,
           "episodes":1, "learning_rate":0.001, "confidence":None, "status":"오류", "path":str(result)}
    app=AppTest.from_string('from pages.results_history import _render_selected_history\nimport streamlit as st\n_render_selected_history(st.session_state["rows"])')
    app.session_state["rows"] = [row]
    app.run()
    text = " ".join(item.value for item in app.markdown)
    assert "v2-badge-warning" in text and "v2-badge-success" not in text
    assert "오류" in text and not app.exception


@pytest.mark.parametrize("menu", ["시뮬레이션","전략 비교","학습 관리","결과 이력","설정"])
def test_all_pages_do_not_hide_unexpected_failures(menu):
    app=AppTest.from_file(APP, default_timeout=60).run()
    app.session_state["current_menu"] = menu
    app.run()
    assert not app.exception and not app.error
    assert not app.get("code") and not app.get("json")
