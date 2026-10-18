import json
from io import BytesIO
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def render():
    return AppTest.from_string("from components.research_upgrade_view import render_research_upgrade\nrender_research_upgrade()", default_timeout=30)


def test_research_example_renders_three_seller_actions_and_download():
    at=render().run()
    assert not at.exception
    assert len(at.dataframe)==2
    assert len(at.dataframe[1].value)==81
    assert "20개 정상판매" in at.dataframe[1].value.columns
    assert len(at.get("download_button"))==1
    assert not at.code
    assert not at.error


def test_missing_uploaded_input_is_not_replaced_by_fake_results():
    at=render().run()
    at.radio(key="research_information_source").set_value("준비한 연구 입력").run()
    assert not at.exception
    assert not at.dataframe


def test_invalid_upload_has_actionable_message_without_traceback():
    def upload(*args, **kwargs):
        return BytesIO(b'{"evidence_class":"SCENARIO","payoffs":null}') if kwargs.get('key')=='research_information_upload' else None
    with patch('streamlit.file_uploader', upload):
        at=render().run()
        at.radio(key="research_information_source").set_value("준비한 연구 입력").run()
    assert not at.exception
    assert at.error
    assert not at.code
    assert "Traceback" not in at.error[0].value
    assert "확인해" in at.error[0].value
