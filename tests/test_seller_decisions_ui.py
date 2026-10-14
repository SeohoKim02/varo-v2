"""Seller UI preserves contract coverage, evidence and legacy production results."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from components import seller_decisions as ui
from services import final_decision_integration as fdi
from tests.test_final_decision_integration import bundle

APP = str(Path(__file__).resolve().parents[1] / "app_v2.py")


@pytest.fixture
def payload():
    data, pipeline = bundle()
    contract = fdi.build_final_decision_integration(data, pipeline)
    pipeline["final_decision_integration"] = contract
    return data, pipeline, contract


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "bad", True])
def test_unknown_loss_never_zero(value):
    assert "0원" not in ui.amount(value, "KRW")
    assert "계산 불가" in ui.amount(value, "KRW")


@pytest.mark.parametrize("mode", ["ACTUAL_OPERATION", "BENCHMARK_PROXY", "SCENARIO", "TEST"])
def test_export_keeps_mode_and_null(payload, mode):
    _, _, contract = payload
    rows = deepcopy(contract["products"])
    rows[0]["decision_mode"] = mode
    rows[0]["actual_saving"] = None
    exported = pd.read_csv(BytesIO(ui.export_rows(rows)))
    assert exported.loc[0, "decision_mode"] == mode
    assert pd.isna(exported.loc[0, "actual_saving"])
    assert len(exported) == len(rows)


@pytest.mark.parametrize("status", list(ui.STATUS_LABELS))
def test_all_statuses_localized(payload, status):
    data, _, contract = payload
    contract["products"][0]["decision_status"] = status
    frame = ui.product_frame(contract, data)
    assert frame.loc[0, "상태"] == ui.STATUS_LABELS[status]
    assert status not in frame.loc[0, "상태"]


def test_coverage_locations_dates_and_no_mutation(payload):
    data, pipeline, contract = payload
    before = fdi.digest(pipeline)
    frame = ui.product_frame(contract, data)
    assert len(frame) == len(contract["products"])
    assert frame["_key"].is_unique
    assert len(frame["점포"].unique()) == 2
    assert set(frame["기준일"]) == {"2026-07-31"}
    assert ui.filter_products(frame, search="P001").shape[0] == 2
    assert fdi.digest(pipeline) == before


@pytest.mark.parametrize("sort", ["상품·점포", "상태·상품", "기준일·상품"])
def test_filters_not_top5(payload, sort):
    data, _, contract = payload
    rows = contract["products"]
    contract["products"] = [dict(rows[0], decision_id=f"D{i}", product_id=f"P{i}", location_id=f"S{i}") for i in range(25)]
    frame = ui.product_frame(contract, data)
    assert len(ui.filter_products(frame, sort=sort)) == 25
    assert len(ui.filter_products(frame, search="S24", sort=sort)) == 1
    assert ui.filter_products(frame, search="NO SUCH PRODUCT").empty
    assert ui.filter_products(pd.DataFrame()).empty


@pytest.mark.parametrize("case,winner", [("A", "TRANSFER"), ("B", "NORMAL_SALE"), ("C", "DISCOUNT_SALE")])
def test_scenario_existing_engine_isolated(case, winner):
    result = ui.scenario_result(case)
    assert result["recommended_strategy"] == winner
    assert result["decision_mode"] == "SCENARIO"
    assert not result["production_promotion_candidate"]


def _app(payload, menu="시뮬레이션"):
    data, pipeline, _ = payload
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.session_state["varo_data"] = data
    app.session_state["varo_pipeline_result"] = pipeline
    app.session_state["varo_recommendations"] = []
    app.session_state["current_menu"] = menu
    return app


def test_products_without_routes_render_and_filter_without_pipeline(payload):
    app = _app(payload, "결과 이력")
    before = fdi.digest(payload[1])
    with patch("services.analysis_pipeline.run_analysis_pipeline", side_effect=AssertionError("display must not analyze")):
        app.run()
        assert not app.exception
        assert len(next(f.value for f in app.dataframe if "상품 코드" in f.value)) == 2
        app.text_input(key="seller_history_search").set_value("no-such-product").run()
        assert not app.exception
        assert any("맞는 상품이 없습니다" in v.value for v in app.info)
        app.text_input(key="seller_history_search").set_value("").run()
        app.selectbox(key="seller_history_detail").select(1).run()
        assert not app.exception
    assert fdi.digest(app.session_state["varo_pipeline_result"]) == before


def test_scenario_submission_does_not_change_pipeline(payload):
    app = _app(payload, "전략 비교").run()
    before = fdi.digest(app.session_state["varo_pipeline_result"])
    next(b for b in app.button if b.label == "예시 계산").click().run()
    assert not app.exception
    assert app.session_state["seller_scenario_result"]["result"]["decision_mode"] == "SCENARIO"
    assert fdi.digest(app.session_state["varo_pipeline_result"]) == before


def test_new_upload_contract_resets_filters_not_result(payload):
    app = _app(payload, "결과 이력").run()
    app.text_input(key="seller_history_search").set_value("old search").run()
    pipeline = deepcopy(payload[1])
    pipeline["final_decision_integration"]["run_id"] = "different-file"
    app.session_state["varo_pipeline_result"] = pipeline
    app.run()
    assert not app.exception
    assert app.text_input(key="seller_history_search").value == ""
    assert app.session_state["varo_pipeline_result"]["final_decision_integration"]["run_id"] == "different-file"


def test_history_shadow_absent_no_write(tmp_path):
    script = "from pages.results_history import _render_shadow_history\n_render_shadow_history()"
    with patch("pages.results_history.history_db_path", return_value=tmp_path / "absent.sqlite"):
        app = AppTest.from_string(script).run()
    assert not app.exception
    assert not (tmp_path / "absent.sqlite").exists()


def test_selected_routes_exact_quantity_cost(payload):
    _, _, contract = payload
    row = next(p for p in contract["products"] if p["selected_transfer_routes"])
    selected = [r for r in contract["routes"] if r["route_id"] in row["selected_transfer_routes"]]
    assert row["allocated_qty"] == sum(r["allocated_qty"] for r in selected)
    assert row["calculated_transfer_cost"] == sum(r["calculated_transfer_cost"] for r in selected)
    decoded = pd.read_csv(BytesIO(ui.export_rows(selected)))
    assert decoded.allocated_qty.sum() == row["allocated_qty"]
    assert decoded.calculated_transfer_cost.sum() == row["calculated_transfer_cost"]


@pytest.mark.parametrize("mode", ["ACTUAL_OPERATION", "BENCHMARK_PROXY", "SCENARIO"])
def test_sample_not_presented_as_real_operation(mode):
    assert "운영 실증 아님" in ui.mode_label(mode, "샘플 추천 데이터")


@pytest.mark.parametrize("menu", ["시뮬레이션", "전략 비교", "학습 관리", "결과 이력", "설정"])
def test_five_pages_with_no_candidate_contract(payload, menu, tmp_path):
    with patch.dict("os.environ", {"VARO_OUTPUT_ROOT": str(tmp_path)}):
        app = _app(payload, menu).run()
    assert not app.exception
    assert any(menu in x.value for x in app.markdown)
    assert not app.sidebar.button
    assert len([b for b in app.button if (b.key or "").startswith("nav_")]) == 5


def test_long_names_and_duplicate_keys_remain_visible(payload):
    data, _, contract = payload
    data["products"]["product_name"] = "긴 한국어 상품 이름 " * 30
    row = deepcopy(contract["products"][0])
    contract["products"].append(row)
    frame = ui.product_frame(contract, data)
    assert len(frame) == 3  # A contract conflict must not be hidden by display deduplication.
    assert frame.iloc[0]["상품"] == "긴 한국어 상품 이름 " * 30
    assert frame._position.is_unique


def test_scenario_comparable_cards(payload):
    app = _app(payload, "전략 비교").run()
    next(b for b in app.button if b.label == "예시 계산").click().run()
    assert not app.exception
    assert sum(x.value == "비교 가능 · 예시 조건" for x in app.caption) == 3


def test_home_only_simulation_summary_and_small_popup(payload):
    app = _app(payload).run()
    assert not app.exception
    assert not app.sidebar.button
    assert not app.dataframe
    assert "전체 기능" in {p.proto.popover.label for p in app.get("popover")}
    assert {"VARO", "전체 결과 보기", "분석 실행", "샘플로 시작"}.issubset({b.label for b in app.button})
    assert "예시 계산" not in {b.label for b in app.button}
    app.button(key="home_results").click().run()
    assert not app.exception
    assert len(next(f.value for f in app.dataframe if "상품 코드" in f.value)) == len(payload[2]["products"])


def test_home_return_preserves_filters_and_analysis(payload):
    app = _app(payload, "결과 이력").run()
    before = fdi.digest(app.session_state["varo_pipeline_result"])
    app.text_input(key="seller_history_search").set_value("P001").run()
    app.selectbox(key="seller_history_sort").set_value("기준일·상품").run()
    app.selectbox(key="seller_history_detail").select(1).run()
    detail = app.selectbox(key="seller_history_detail").value
    app.button(key="ui_home").click().run()
    assert not app.exception
    assert app.session_state["current_menu"] == "시뮬레이션"
    app.button(key="nav_결과 이력").click().run()
    assert not app.exception
    assert app.text_input(key="seller_history_search").value == "P001"
    assert app.selectbox(key="seller_history_sort").value == "기준일·상품"
    assert app.selectbox(key="seller_history_detail").value == detail
    assert fdi.digest(app.session_state["varo_pipeline_result"]) == before


def test_back_tracks_pages_and_preserves_training_inputs(payload):
    app = _app(payload).run()
    before = fdi.digest(app.session_state["varo_pipeline_result"])
    app.button(key="nav_학습 관리").click().run()
    app.number_input(key="dqn_single_episodes").set_value(42).run()
    app.number_input(key="dqn_learning_rate").set_value(0.002).run()
    app.button(key="nav_설정").click().run()
    app.button(key="ui_back").click().run()
    assert not app.exception
    assert app.session_state["current_menu"] == "학습 관리"
    assert app.number_input(key="dqn_single_episodes").value == 42
    assert app.number_input(key="dqn_learning_rate").value == 0.002
    app.button(key="ui_back").click().run()
    assert not app.exception
    assert app.session_state["current_menu"] == "시뮬레이션"
    assert fdi.digest(app.session_state["varo_pipeline_result"]) == before


def test_changed_data_signature_does_not_restore_old_filter(payload):
    app = _app(payload, "결과 이력").run()
    app.text_input(key="seller_history_search").set_value("old-filter").run()
    app.button(key="ui_home").click().run()
    pipeline = deepcopy(payload[1])
    pipeline["final_decision_integration"]["run_id"] = "new-upload"
    app.session_state["data_signature"] = "new-upload"
    app.session_state["varo_pipeline_result"] = pipeline
    app.button(key="nav_결과 이력").click().run()
    assert not app.exception
    assert app.text_input(key="seller_history_search").value == ""


def test_pending_upload_survives_navigation_without_reanalysis(payload):
    app = _app(payload).run()
    before = fdi.digest(app.session_state["varo_pipeline_result"])
    pending = {"identity": ("upload-id", "pending.xlsx", 4), "name": "pending.xlsx", "content": b"test",
               "signature": app.session_state["data_signature"]}
    app.session_state["_home_pending_upload"] = pending
    with patch("services.analysis_pipeline.run_analysis_pipeline", side_effect=AssertionError("navigation cannot analyze")):
        app.button(key="nav_설정").click().run()
        app.button(key="ui_home").click().run()
    assert not app.exception
    assert app.session_state["_home_pending_upload"] == pending
    assert not app.button(key="home_analyze").disabled
    assert fdi.digest(app.session_state["varo_pipeline_result"]) == before


def test_pending_upload_clears_when_different_data_applied(payload):
    app = _app(payload).run()
    app.session_state["_home_pending_upload"] = {"name": "old.xlsx", "content": b"test", "signature": "old-data"}
    app.session_state["data_signature"] = "new-data"
    app.run()
    assert not app.exception
    assert "_home_pending_upload" not in app.session_state
    assert app.button(key="home_analyze").disabled


@pytest.mark.parametrize("declared,route_type,expected", [(True, "DIRECT", True), (False, "DIRECT", False), (True, "VIA_DC", False)])
def test_direct_network_does_not_invent_a_dc(declared, route_type, expected):
    from simulation.dynamic_network import compute_dynamic_layout
    nodes = [{"node_id": key, "node_type": "STORE", "network_mode": "DIRECT_NETWORK" if declared else ""} for key in ("S1", "S2")]
    layout = compute_dynamic_layout(nodes, [{"source_id": "S1", "target_id": "S2", "route_type": route_type}])
    assert layout.is_valid == expected
    assert not layout.dcs
    if expected:
        assert len(layout.stores) == 2
