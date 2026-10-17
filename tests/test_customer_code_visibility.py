"""Customer screens keep technical diagnostics in logs without hiding failures."""
from copy import deepcopy
from pathlib import Path
import tomllib

import pytest
from streamlit.testing.v1 import AppTest

from components.user_feedback import analysis_label, user_text
from components.status import user_status_label


@pytest.mark.parametrize("value", [
    "Traceback (most recent call last):", "ValueError: invalid input",
    "import pandas as pd", "from services import pipeline", "def solve(x):",
    "st.write(result)", "model.predict(data)", "private_internal_state",
    "```python\nprint(1)\n```", r"C:\private\model.pt", "pipeline.py:17",
    "print(1)", "result = 3", "{'status': 'FAILED'}",
])
def test_code_is_replaced_and_retained_in_log(value, caplog):
    assert user_text(value, "입력과 계산 조건을 확인해 주세요.") == "입력과 계산 조건을 확인해 주세요."
    assert value in caplog.text


@pytest.mark.parametrize("value", ["P01", "R005", "냉동만두500g", "재고 이동", "판매 수량 14개", "42.86%"])
def test_product_identifiers_and_business_values_are_preserved(value):
    assert user_text(value) == value


def test_user_toolbar_hides_developer_actions_and_tracebacks():
    config=tomllib.loads((Path(__file__).resolve().parents[1]/".streamlit/config.toml").read_text(encoding="utf-8"))
    assert config["client"]["toolbarMode"] == "viewer"
    assert config["client"]["showErrorDetails"] == "none"


def test_code_in_error_message_is_not_shown_as_success(caplog):
    app=AppTest.from_string('from components.user_feedback import report_error\nreport_error("save", RuntimeError("private_variable"), "Traceback: private_variable")').run()
    assert not app.exception and len(app.error)==1 and not app.success
    assert "문제가 발생" in app.error[0].value
    assert "private_variable" not in app.error[0].value
    assert "private_variable" in caplog.text


def test_algorithm_family_is_readable_without_renaming_stored_functions():
    assert analysis_label("services.vhs_score_engine.apply_auto_vhs") == "재고 처리 우선순위 분석"
    assert analysis_label("vhs_confidence.add_confidence_columns") == "추천 안정성 분석"


@pytest.mark.parametrize("value,label", [("success","완료"),("failed","실패"),("blocked","진행 불가 · 조건 확인 필요")])
def test_lowercase_internal_status_is_readable(value,label):
    assert user_status_label(value)==label


def test_advanced_report_does_not_expose_functions_or_internal_codes(caplog):
    pipeline={"connected_algorithms":["services.vhs_score_engine.apply_auto_vhs"],
        "deferred_algorithms":[{"algorithm":"demand_forecast.predict", "reason":"missing_input_columns"}],
        "warnings":["Traceback: private_variable"], "status":"BLOCKED", "result_basis":"private_result_basis",
        "validation_report":{"calculation_sources":{"vhs":"private_module.calculate"}},
        "demand_analysis":{"demand_forecast":{"status":"BLOCKED", "function":"private_module.predict",
            "output_columns":["forecast_qty"], "missing_input_columns":["private_column"]}}}
    before=deepcopy(pipeline)
    app=AppTest.from_string('import streamlit as st\nfrom pages.validation import _render_validation_report\n_render_validation_report(st.session_state["pipeline"])')
    app.session_state["pipeline"]=pipeline
    app.run()
    assert not app.exception
    screen=" ".join(str(x.value) for kind in (app.markdown,app.caption,app.warning,app.error) for x in kind)
    screen+=" ".join(x.value.to_string() for x in app.dataframe)
    for token in ("private_", "apply_auto_vhs", "missing_input_columns", "forecast_qty", "Traceback", "BLOCKED"):
        assert token not in screen
    assert "진행 불가" in screen and app.warning
    assert "private_variable" in caplog.text and "private_module" in caplog.text
    assert app.session_state["pipeline"] == before
    assert not app.get("code") and not app.get("json")


@pytest.mark.parametrize("use_rows", [True, False])
def test_weight_labels_are_readable_without_changing_calculations(use_rows):
    analysis={"weights":{"savings_score":0.6,"demand_fit_score":0.4},
        "contributions":{"savings_score":30,"demand_fit_score":20}}
    if use_rows:
        analysis["weight_rows"]=[{"component":"savings_score","used":True,
            "weight":0.6,"fallback_reason":"missing_input_columns"},
            {"component":"demand_fit_score","used":True,"weight":0.4,"fallback_reason":""}]
    pipeline={"vhs_analysis":analysis}
    before=deepcopy(pipeline)
    app=AppTest.from_string('import streamlit as st\nfrom pages.validation import _render_vhs_weights\n_render_vhs_weights(st.session_state["pipeline"])')
    app.session_state["pipeline"]=pipeline
    app.run()
    assert not app.exception
    table=app.dataframe[0].value
    assert "절감 효과" in table["구성 요소"].tolist()
    assert not any(token in table.to_string() for token in ("savings_score","demand_fit_score","missing_input_columns"))
    assert table.iloc[0]["최종 가중치" if use_rows else "평균 적용 가중치"]==0.6
    if use_rows:
        assert table.iloc[1]["대체 적용 사유"]=="-"
    assert app.session_state["pipeline"]==before
