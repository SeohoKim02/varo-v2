"""Malformed inputs remain actionable without leaking Python diagnostics."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from services.candidate_input import prepare_input, read_uploads
from services.candidate_input_examples import csv_test_files, mixed_test_metadata


@pytest.mark.parametrize("value", ["bad", "1.5", "0,", "None"])
def test_departure_input_error_is_korean_and_logged(value, caplog):
    bundle = read_uploads(csv_test_files())
    bundle.tables["economic_route_policy"]["departure_periods"] = value
    prepared = prepare_input(bundle, mixed_test_metadata())
    assert not prepared.ready
    messages = " ".join(row["message_ko"] for row in prepared.issues)
    assert "출발 가능한 날짜 순서" in messages
    assert "invalid literal" not in messages and "Traceback" not in messages
    assert any(record.exc_info for record in caplog.records)


def test_invalid_utf8_does_not_expose_decode_exception(caplog):
    result = read_uploads({"inventory_snapshot.csv": b"\xff\xff"})
    assert result.issues
    assert "다시 저장" in result.issues[0]["message_ko"]
    assert "codec" not in result.issues[0]["message_ko"]
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("corrupt", [{}, {"created_at":"bad", "label":"bad", "economic_result":{"expected_net_value":0}}])
def test_invalid_history_not_added_to_detail_choices(tmp_path, monkeypatch, corrupt):
    import pages.candidate_review as page
    monkeypatch.setattr(page, "HISTORY_DIR", tmp_path)
    good = {"created_at":"2026-10-17", "label":"정상 기록", "economic_result":{"expected_net_value":74},
        "products":[{"상품":"P", "계획 이동량":4}], "mode":"TEST"}
    (tmp_path/"candidate_good.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path/"candidate_bad.json").write_text(json.dumps(corrupt), encoding="utf-8")
    at = AppTest.from_string("from pages.candidate_review import _render_history\n_render_history()")
    at.run(); at.checkbox(key="candidate_history_detail_open").check().run()
    assert not at.exception
    assert len(at.warning)==1
    assert len(at.selectbox(key="candidate_history_selection").options)==1
    assert "정상 기록" in at.selectbox(key="candidate_history_selection").options[0]
    assert len(at.get("download_button"))==1


def test_visual_fixture_uses_the_same_dc_decoration_as_application():
    at = AppTest.from_file("tests/network_visual_app.py")
    at.run(); at.radio[0].set_value("경유 이동").run()
    assert not at.exception
    html = " ".join(item.value for item in at.markdown)
    assert 'DC01 · 이번 경로 경유' in html
    assert 'data-active="true"' in html
