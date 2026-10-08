"""Canonical action contract: controlled scenarios A-T, exact label mapping, invariants and pipeline isolation."""
from __future__ import annotations

import copy
import warnings
from pathlib import Path
from unittest import mock

import pytest

from services import action_consistency as ac
from services import analysis_pipeline
from services import shared_feasibility_selection as sf
from services.action_consistency_validation import (CONTROLLED_SCENARIOS, OUTPUT_FILES, _event, _plain, _rec, _seller,
                                                    _seller_decision, _shared, run_scenario)
from services.analysis_pipeline import build_v2_state
from services.data_loader import load_excel_data

REPO = Path(__file__).resolve().parents[1]
NETWORK_SAMPLE = REPO / "data" / "Varo_V2_네트워크_샘플.xlsx"
DQN_SAMPLE = REPO / "Varo_DQN_training_samples_10pack" / "Varo_DQN_sample_02_4stores_1dc_frozen.xlsx"


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_excel_data(path)


def _records(result):
    return {record["route_id"]: record for record in result["records"]}


# ------------------------------------------------------------------------------------------- scenarios A-T


@pytest.mark.parametrize("scenario", CONTROLLED_SCENARIOS, ids=[item["id"] for item in CONTROLLED_SCENARIOS])
def test_controlled_scenario(scenario):
    outcome = run_scenario(scenario)
    assert outcome["status"] == "PASS", outcome["failures"]


def test_scenarios_cover_a_to_t():
    assert [item["id"] for item in CONTROLLED_SCENARIOS] == list("ABCDEFGHIJKLMNOPQRST")


def test_no_template_sentence_claims_a_guarantee_or_completion():
    for scenario in CONTROLLED_SCENARIOS:
        result = ac.build_action_consistency(**scenario["build"]())
        for record in result["records"]:
            text = " ".join(record["explanation_ko"])
            assert not any(term in text for term in ac.GUARANTEE_TERMS), (scenario["id"], text)


# ------------------------------------------------------------------------------------------- mapping


@pytest.mark.parametrize("label,code,status", [
    ("재고 이동", "TRANSFER", "CANONICAL"), ("재배치 이동", "TRANSFER", "ALIAS"), ("할인", "DISCOUNT_SALE", "ALIAS"),
    ("할인 판매", "DISCOUNT_SALE", "CANONICAL"), ("보류", "HOLD", "CANONICAL"), ("정상 판매 유지", "NORMAL_SALE", "CANONICAL"),
    ("긴급 할인", "URGENT_DISCOUNT", "CANONICAL"), ("긴급할인", "URGENT_DISCOUNT", "ALIAS"), ("1+1", "BUNDLE_PROMOTION", "CANONICAL"),
    ("폐기", "DISPOSE", "CANONICAL"), ("discount", "DISCOUNT_SALE", "ALIAS"), ("hold", "HOLD", "ALIAS"),
    ("프로모션 추천", "PROMOTION", "ALIAS"), ("재배치 추천", "TRANSFER", "ALIAS"),
    ("이동 비추천", None, "UNMAPPABLE"), ("threshold", None, "UNMAPPABLE"), ("비교 불가", None, "SENTINEL"),
    ("미연결", None, "SENTINEL"), ("유지", None, "SENTINEL"), (None, None, "EMPTY"), ("", None, "EMPTY"),
])
def test_label_mapping_is_exact(label, code, status):
    assert ac.map_action_label(label) == (code, status)


def test_hold_and_promotion_variants_stay_outside_the_seller_loss_scope():
    compared = {code for code, item in ac.ACTION_CATALOG.items() if item["seller_loss_compared"]}
    assert compared == {"TRANSFER", "NORMAL_SALE", "DISCOUNT_SALE"}
    assert ac.map_action_label("보류")[0] != ac.map_action_label("정상 판매 유지")[0]
    assert len({ac.map_action_label(label)[0] for label in ("할인", "긴급 할인", "1+1", "프로모션 추천")}) == 4


def test_contract_document_is_stable_and_every_conflict_has_a_sentence():
    assert ac.contract_document() == ac.contract_document()
    assert set(ac.CONFLICTS) == set(ac._CONFLICT_TEXT)
    assert all(item["label_ko"] for item in ac.ACTION_CATALOG.values())
    assert set(OUTPUT_FILES[:6]) == {
        "action_consistency_summary.csv", "action_consistency_by_day.csv", "action_consistency_conflicts.csv",
        "action_consistency_examples.csv", "action_consistency_validation.json", "action_mapping_contract.json"}


# ------------------------------------------------------------------------------------------- plan rules


def test_primary_plan_is_strict_unless_strict_has_no_validated_caps():
    assert ac.primary_mode(None)[0] is None
    assert ac.primary_mode({"status": "error", "modes": {}})[0] is None
    strict_none = {"modes": {sf.STRICT_ACTUAL: {"plan_status": sf.INSUFFICIENT_CAP_DATA}, sf.BENCHMARK_PROXY: {"plan_status": "SELECTED"}}}
    assert ac.primary_mode(strict_none)[0] == sf.BENCHMARK_PROXY
    strict_rejects = {"modes": {sf.STRICT_ACTUAL: {"plan_status": "NO_FEASIBLE_SELECTION"}, sf.BENCHMARK_PROXY: {"plan_status": "SELECTED"}}}
    assert ac.primary_mode(strict_rejects)[0] == sf.STRICT_ACTUAL


def test_unavailable_selection_never_shows_a_transfer():
    result = ac.build_action_consistency([_rec("R1", 50, "재고 이동", rank=1)], shared_feasibility={"status": "error", "modes": {}})
    record = result["records"][0]
    assert record["action_code"] is None and record["action_status"] == ac.COMPARISON_UNAVAILABLE
    assert record["consistency_status"] == ac.NOT_COMPARABLE and "SHARED_FEASIBILITY_UNAVAILABLE" in record["reason_codes"]
    assert result["invariant_failures"] == []


def test_selected_plan_row_without_recommendation_is_an_orphan():
    plan_records = [_rec("R1", 20, rank=1), _rec("R5", 20, target="T5", rank=2)]
    result = ac.build_action_consistency([_rec("R1", 20, rank=1)], shared_feasibility=_shared(plan_records))
    assert result["orphan_plan_rows"] == ["R5"] and "I4" in result["invariant_failures"]


def test_partial_plan_flags_the_seller_loss_quantity_basis():
    records = [_rec("R1", 20, "재고 이동", rank=1, need=10, transfer_cost_basis="PER_UNIT")]
    result = ac.build_action_consistency(**_plain(records, seller_loss=_seller([_seller_decision("R1")])))
    record = _records(result)["R1"]
    assert record["allocated_qty"] == 10.0 and record["seller_loss_decision_qty"] == 20.0
    assert {"QTY_PARTIAL_ALLOCATION", "SELLER_LOSS_QTY_BASIS_DIFFERS"} <= set(record["reason_codes"])
    assert "금액 비교는 추천 수량 20개 기준이며 배정 수량과 다릅니다." in " ".join(record["explanation_ko"])


def test_uploaded_label_is_not_called_a_legacy_rule():
    result = ac.build_action_consistency(**_plain([_rec("R1", 50, "할인", rank=1)]), legacy_rule_connected=False)
    record = result["records"][0]
    assert record["legacy_action_source"] == "UPLOADED_LABEL" and "LEGACY_LABEL_PASS_THROUGH" in record["reason_codes"]


def test_operator_record_for_another_product_is_never_execution():
    kwargs = _plain([_rec("R1", 50, "재고 이동", rank=1)])
    result = ac.build_action_consistency(**kwargs, operator_events=[_event("R1", product_id="P9")])
    record = result["records"][0]
    assert record["execution_state"] == ac.NOT_RECORDED and record["operator_action"] is None and not record["is_executed"]
    assert "OPERATOR_EVENT_KEY_MISMATCH:product_id" in record["reason_codes"]


def test_inputs_are_not_mutated():
    kwargs = _plain([_rec("R1", 30, "할인", rank=1, need=40), _rec("R2", 30, "보류", source="S2", rank=2, need=40)])
    before = copy.deepcopy(kwargs)
    ac.build_action_consistency(**kwargs)
    assert kwargs == before


# ------------------------------------------------------------------------------------------- pipeline


def test_pipeline_field_is_parallel_and_fault_isolated():
    data = _load(NETWORK_SAMPLE)
    with_field = build_v2_state(copy.deepcopy(data), detail_level="full")
    with mock.patch.object(analysis_pipeline, "build_action_consistency", side_effect=RuntimeError("boom")):
        broken = build_v2_state(copy.deepcopy(data), detail_level="full")
    for key in ("top5", "summary", "connected_algorithms", "status", "seller_loss_analysis", "shared_feasibility_selection"):
        assert with_field["pipeline_result"][key] == broken["pipeline_result"][key]
    assert with_field["recommendations"] == broken["recommendations"]
    assert broken["pipeline_result"]["action_consistency"]["status"] == "error"
    analysis = with_field["pipeline_result"]["action_consistency"]
    again = build_v2_state(copy.deepcopy(data), detail_level="full")["pipeline_result"]["action_consistency"]
    assert again == analysis  # deterministic: no timings in the pipeline field
    assert analysis["status"] == "parallel_only" and analysis["production_action_applied"] is False
    assert analysis["invariant_failures"] == [] and analysis["record_count"] == len(with_field["recommendations"])
    for record, recommendation in zip(analysis["records"], with_field["recommendations"]):
        assert record["production_action"] == recommendation["varo_action"]
        assert record["action_source"] != "SELLER_LOSS" and record["operator_action"] is None


def test_pipeline_plans_match_the_shared_feasibility_rows():
    state = build_v2_state(_load(DQN_SAMPLE), detail_level="core")
    shared = state["pipeline_result"]["shared_feasibility_selection"]
    analysis = state["pipeline_result"]["action_consistency"]
    assert analysis["primary_selection_mode"] == sf.STRICT_ACTUAL
    plan = shared["modes"][sf.STRICT_ACTUAL]
    records = {record["candidate_id"]: record for record in analysis["records"]}
    for row in plan["rows"]:
        record = records[row["candidate_id"]]
        if row["selection_status"] in (sf.SELECTED, sf.PARTIALLY_SELECTED):
            assert record["action_status"] == ac.FEASIBLE_PLAN and record["allocated_qty"] == row["allocated_qty"]
            assert (record["product_id"], record["source_id"], record["target_id"]) == (row["product_id"], row["source_id"], row["target_id"])
        else:
            assert record["action_code"] is None and record["allocated_qty"] == 0.0
    assert analysis["selected_plan_count"] == plan["selected_count"] > 0
    assert analysis["total_allocated_qty"] == pytest.approx(plan["total_allocated_qty"])
    # every selected plan carries the legacy label '보류' -> review, never silently overwritten
    assert all("B_TRANSFER_PLAN_VS_LEGACY_HOLD" in r["conflict_codes"] for r in analysis["records"] if r["action_code"] == "TRANSFER")
    assert all(r["legacy_action"] == r["production_action"] == "보류" for r in analysis["records"])
