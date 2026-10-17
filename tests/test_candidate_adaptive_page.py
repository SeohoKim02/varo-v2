"""UI contracts retain the original upload E2E tests and add adaptive paths."""
from datetime import date
from types import SimpleNamespace
import json

import pytest
from streamlit.testing.v1 import AppTest

from services.candidate_input_examples import csv_test_files


def app(monkeypatch, tmp_path):
    import streamlit as st
    import pages.candidate_review as page
    monkeypatch.setenv("VARO_CANDIDATE_REVIEW", "1")
    monkeypatch.setattr(page, "HISTORY_DIR", tmp_path)
    objects = [SimpleNamespace(name=k, getvalue=lambda v=v:v) for k,v in csv_test_files().items()]
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw:objects)
    at = AppTest.from_string("from pages.candidate_review import render_candidate_review_page\nrender_candidate_review_page()", default_timeout=30)
    for k,v in {"candidate_data_mode":"실제 자료 검토", "candidate_dataset":"EXPLICIT_TEST", "candidate_network":"test-network",
        "candidate_as_of":date(2026,10,16), "candidate_input_kind":"실제 자료 + 시험 가정", "candidate_consent":True}.items():
        at.session_state[k] = v
    at.run(); at.button(key="candidate_validate_upload").click().run()
    assert not at.exception
    return at


@pytest.mark.parametrize("policy", ["처음 정한 계획 유지", "새 관측으로 남은 계획 조정", "상황별 후속 계획 함께 검토"])
def test_each_policy_real_upload_results_and_history(monkeypatch,tmp_path,policy):
    at = app(monkeypatch,tmp_path)
    at.selectbox(key="candidate_policy").set_value(policy).run()
    at.button(key="candidate_real_run").click().run()
    assert not at.exception
    result = at.session_state["_candidate_uploaded_result"]
    assert result["expected_net_value"] == pytest.approx(74)
    assert result["production_approved"] is False and result["actual_savings"] is None
    assert "alternative_comparison" not in result
    assert any("지금 검토할 행동" in x.value for x in at.subheader)
    at.text_input(key="candidate_uploaded_search").set_value("P").run()
    assert not any("이전 입력·관측" in x.value for x in at.warning)
    at.button(key="candidate_uploaded_save").click().run()
    record = json.loads(next(tmp_path.glob("candidate_*.json")).read_text(encoding="utf-8"))
    assert record["products"][0]["계획 이동량"] == 4
    assert record["products"][0]["출발 점포 판매 계획"] == 6
    assert record["lifecycle"] == "ANALYSIS_PLAN" and record["approval_status"] == "NOT_APPROVED"
    at.checkbox(key="candidate_history_detail_open").check().run()
    assert not at.exception and len(at.get("download_button")) == 2


def test_policy_change_marks_stale_preserves_previous_and_search(monkeypatch,tmp_path):
    at = app(monkeypatch,tmp_path)
    at.button(key="candidate_real_run").click().run()
    before = at.session_state["_candidate_uploaded_result"]
    at.text_input(key="candidate_uploaded_search").set_value("P").run()
    at.selectbox(key="candidate_policy").set_value("새 관측으로 남은 계획 조정").run()
    assert at.session_state["_candidate_uploaded_result"] == before
    assert any("이전 입력·관측" in w.value for w in at.warning)
    at.button(key="candidate_real_run").click().run()
    assert not at.exception and at.session_state["candidate_uploaded_search"] == "P"
    assert not any("이전 입력·관측" in w.value for w in at.warning)


def test_future_observation_excluded_and_known_sale_replans(monkeypatch,tmp_path):
    at = app(monkeypatch,tmp_path)
    at.selectbox(key="candidate_policy").set_value("새 관측으로 남은 계획 조정").run()
    at.button(key="candidate_real_run").click().run()
    key = at.session_state["_candidate_ready_input_key"]
    p = at.session_state["_candidate_ready_input"].problem
    lot = p.lots[0]
    observation = {"event_id":"sale1", "kind":"SALE", "occurred_at":"2026-10-16T12:00:00+09:00",
        "known_at":"2026-10-16T18:00:00+09:00", "product_id":lot.product_id, "location":lot.source,
        "state":lot.state, "lot_id":lot.lot_id, "unit":lot.unit, "quantity":1, "source":"TEST receipt", "price_level":"discount"}
    at.session_state["_candidate_observation_sets"] = {key:[observation]}
    at.run(); at.button(key="candidate_real_run").click().run()
    assert not at.exception
    envelope = at.session_state["_candidate_adaptive_result"]
    assert len(envelope["deferred_observations"]) == 1
    assert not envelope["observations_used"]
    at.date_input(key="candidate_decision_date").set_value(date(2026,10,17)).run()
    at.button(key="candidate_real_run").click().run()
    assert not at.exception
    envelope = at.session_state["_candidate_adaptive_result"]
    assert envelope["status"] == "COMPLETED"
    assert len(envelope["observations_used"]) == 1
    assert sum(r["quantity"] for r in envelope["current_inventory"]) == 9
    assert at.session_state["_candidate_uploaded_result"]["input_data_class"] == "TEST"


def test_duplicate_upload_names_cannot_mark_old_plan_current(monkeypatch, tmp_path):
    import streamlit as st
    at = app(monkeypatch, tmp_path)
    at.button(key="candidate_real_run").click().run()
    old = at.session_state["_candidate_uploaded_result"]
    objects = st.file_uploader()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: [objects[0], *objects])
    at.run()
    at.button(key="candidate_validate_upload").click().run()
    assert not at.exception
    assert at.session_state["_candidate_uploaded_result"] == old
    assert any("이전 입력·관측" in w.value for w in at.warning)
    assert any("파일을 읽지 못했습니다" in e.value for e in at.error)


def test_full_app_navigation_preserves_candidate_inputs_without_replaying_actions(monkeypatch,tmp_path):
    import streamlit as st
    import pages.candidate_review as page
    monkeypatch.setenv("VARO_CANDIDATE_REVIEW", "1")
    monkeypatch.setattr(page, "HISTORY_DIR", tmp_path)
    objects = [SimpleNamespace(name=k,getvalue=lambda v=v:v) for k,v in csv_test_files().items()]
    current = {"uploads":objects}
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw:current["uploads"] if kw.get("key")=="candidate_inventory_upload" else None)
    at=AppTest.from_file("app_candidate.py",default_timeout=30)
    for k,v in {"current_menu":"후보 결과 검토", "candidate_data_mode":"실제 자료 검토", "candidate_dataset":"EXPLICIT_TEST",
        "candidate_network":"test-network", "candidate_as_of":date(2026,10,16), "candidate_input_kind":"실제 자료 + 시험 가정", "candidate_consent":True}.items():at.session_state[k]=v
    at.run();at.button(key="candidate_validate_upload").click().run();at.button(key="candidate_real_run").click().run()
    at.text_input(key="candidate_uploaded_search").set_value("P").run()
    before=at.session_state["_candidate_uploaded_result"]
    at.button(key="ui_home").click().run()
    current["uploads"]=[]  # Streamlit does not retain an unrendered file uploader.
    at.button(key="nav_설정").click().run()
    at.button(key="settings_candidate_review").click().run()
    assert not at.exception
    assert at.session_state["_candidate_uploaded_result"]==before
    assert at.session_state["candidate_uploaded_search"]=="P"
    assert at.text_input(key="candidate_uploaded_search").value=="P"
    assert not at.error
    assert not any("이전 입력·관측" in w.value for w in at.warning)
    assert not list(tmp_path.glob("candidate_*.json"))
    at.session_state["candidate_inventory_upload"] = []
    at.button(key="ui_back").click().run()
    at.button(key="settings_candidate_review").click().run()
    assert not at.exception and not at.error
    assert at.text_input(key="candidate_uploaded_search").value=="P"
    assert "candidate_inventory_upload" not in at.session_state["_ui_input_memory"]["values"]
    assert at.session_state["_candidate_uploaded_result"]==before
