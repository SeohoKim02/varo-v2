from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.delenv("VARO_CANDIDATE_REVIEW", raising=False)
    import pages.candidate_review as page
    monkeypatch.setattr(page, "HISTORY_DIR", tmp_path / "isolated_candidate_history")
    at = AppTest.from_file(str(ROOT / "app_candidate.py"), default_timeout=30).run()
    assert not at.exception
    at.button(key="candidate_opt_in").click().run()
    assert not at.exception
    at.button(key="settings_candidate_review").click().run()
    assert not at.exception
    return at


@pytest.mark.parametrize("label,value,move", [("정상판매 중심", 100, 0), ("할인판매 중심", 80, 0),
    ("재고 이동 중심", 118, 10), ("이동과 할인 혼합", 74, 4)])
def test_candidate_scenario_e2e(app, label, value, move):
    app.selectbox(key="candidate_scenario_choice").select(label).run()
    app.button(key="candidate_load_scenario").click().run()
    app.button(key="candidate_run").click().run()
    assert not app.exception
    result = app.session_state["_candidate_result"]
    assert result["expected_net_value"] == pytest.approx(value)
    assert sum(x["quantity"] for x in result["transfers"]) == pytest.approx(move)
    assert any(m.label == "실제 절감액" and m.value == "미산정" for m in app.metric)
    assert len(app.get("download_button")) == 1
    assert app.session_state["varo_pipeline_result"] == {}
    assert not result["production_approved"]


def test_candidate_navigation_retains_result_and_search(app):
    app.button(key="candidate_load_scenario").click().run()
    app.button(key="candidate_run").click().run()
    app.text_input(key="candidate_product_search").set_value("P").run()
    before = app.session_state["_candidate_result"]["problem_signature"]
    app.button(key="ui_home").click().run()
    assert not app.exception
    app.button(key="nav_설정").click().run()
    app.button(key="settings_candidate_review").click().run()
    assert not app.exception
    assert app.session_state["_candidate_result"]["problem_signature"] == before
    assert app.text_input(key="candidate_product_search").value == "P"


def test_candidate_storage_isolated_reuses_run_ledger(app, tmp_path):
    app.button(key="candidate_load_scenario").click().run()
    app.button(key="candidate_run").click().run()
    app.button(key="candidate_save").click().run()
    assert not app.exception
    folder = tmp_path / "isolated_candidate_history"
    assert len(list(folder.glob("candidate_*.json"))) == 1
    assert list(folder.glob("*.sqlite*")) or list(folder.glob("*.db"))
    assert any("기존 결과 이력과 운영 기록은 변경하지 않았습니다" in x.value for x in app.success)


def test_real_data_insufficient_never_scenario_fallback(app):
    app.radio(key="candidate_data_mode").set_value("실제 자료 검토").run()
    assert not app.exception
    app.button(key="candidate_validate_upload").click().run()
    assert not app.exception
    assert any("먼저 재고 CSV" in x.value for x in app.warning)
    assert not app.metric
    assert not any(x.key == "candidate_run" for x in app.button)


def test_baseline_has_only_five_menu_no_opt_in(monkeypatch):
    monkeypatch.delenv("VARO_CANDIDATE_REVIEW", raising=False)
    at = AppTest.from_file(str(ROOT / "app_v2.py"), default_timeout=30)
    at.session_state["current_menu"] = "설정"
    at.run()
    assert not at.exception
    assert not any(x.key == "settings_candidate_review" for x in at.button)
    assert len([x for x in at.button if x.key and x.key.startswith("nav_")]) == 5
    at.session_state["current_menu"] = "후보 결과 검토"
    at.run()
    assert at.session_state["current_menu"] == "시뮬레이션"


def test_candidate_error_is_korean_no_traceback(app, monkeypatch):
    import pages.candidate_review as page
    app.button(key="candidate_load_scenario").click().run()
    def fail(*args, **kwargs):
        raise RuntimeError("SECRET_INTERNAL_STATE")
    monkeypatch.setattr(page, "solve_snapshot_economics", fail)
    app.button(key="candidate_run").click().run()
    assert not app.exception
    assert any("計算" not in x.value and "계산을 완료하지 못했습니다" in x.value for x in app.error)
    visible = "\n".join(x.value for kind in (app.error, app.warning, app.info, app.markdown) for x in kind)
    assert "SECRET_INTERNAL_STATE" not in visible
    assert "Traceback" not in visible


def test_previous_button_memory_does_not_break_settings(app):
    app.session_state["_ui_input_memory"] = {"signature": None, "values": {"settings_source_sample": False,
        "settings_auto_weight": True, "settings_dqn_mode": "참고만"}}
    app.session_state["current_menu"] = "설정"
    app.run()
    assert not app.exception
    assert not app.error
    assert app.button(key="settings_candidate_review")


def test_failed_rerun_does_not_present_previous_result_as_success(app, monkeypatch):
    import pages.candidate_review as page
    app.button(key="candidate_load_scenario").click().run()
    app.button(key="candidate_run").click().run()
    assert app.metric
    def fail(*args, **kwargs):
        raise RuntimeError("failed current calculation")
    monkeypatch.setattr(page, "solve_snapshot_economics", fail)
    app.button(key="candidate_run").click().run()
    assert not app.metric
    assert app.error
    assert not app.exception
