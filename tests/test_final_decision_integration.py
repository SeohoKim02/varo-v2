"""Synthetic branch proofs; no production approval or real performance claims."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pandas as pd
import pytest

from services import final_decision_integration as fdi
from services import seller_loss_engine as loss, seller_loss_inputs as sli
from services import seller_loss_promotion_gate as gate
from services.seller_decision_validation import controlled_scenarios, scenario_input
from tests.fixtures import sample_workbook

DATE = "2026-07-31"


def real_decision(base=None):
    base = base or scenario_input()
    base = replace(base, decision_id="R001", product_id="P001", legacy_action=None,
                   **{k: replace(v, provenance="DIRECT_REAL" if v.present else "MISSING", source="declared seller record",
                                 dataset=loss.WORKBOOK_DATASET, note="") for k, v in base.input_fields().items()})
    return sli.evaluate_with_seller_inputs(base, None, decision_date=DATE)


def bundle(decision=None):
    decision = decision or real_decision()
    source, target = decision["source_store_id"], decision["target_store_id"]
    stock = decision["input_provenance"]["source_current_stock"]["value"]
    data = {"inventory": pd.DataFrame([{"store_id": source, "product_id": "P001", "stock_qty": stock,
             "stock_qty_provenance": "DIRECT_REAL", "snapshot_date": DATE, "quantity_unit": "ea"},
             {"store_id": target, "product_id": "P001", "stock_qty": 5, "stock_qty_provenance": "DIRECT_REAL", "snapshot_date": DATE, "quantity_unit": "ea"}]),
            "products": pd.DataFrame([{"product_id": "P001", "product_name": "fixture product"}]),
            "config": pd.DataFrame([{"key": "currency", "value": "KRW"}, {"key": "quantity_unit", "value": "ea"}, {"key": "as_of", "value": DATE}])}
    rec = {"route_id": "R001", "source_id": source, "target_id": target, "product_id": "P001", "varo_action": None, "recommended_qty": 20}
    selected = {"candidate_id": "R001", "selection_status": "SELECTED", "allocated_qty": 20, "allocated_move_cost": 1800,
                "quantity_unit": "ea", "cost_status": "EXACT_FULL_QTY", "cap_provenance": "source= DIRECT_REAL"}
    pipeline = {"status": "success", "recommendations": [rec], "shared_feasibility_selection": {"modes": {
        "STRICT_ACTUAL": {"plan_status": "SELECTED", "rows": [selected], "validation": {"violation_count": 0}},
        "BENCHMARK_PROXY": {"plan_status": "SELECTED", "rows": [deepcopy(selected)], "validation": {"violation_count": 0}}}},
        "action_consistency": {"records": [{**rec, "candidate_id": "R001", "action_code": "TRANSFER", "consistency_status": "CONSISTENT", "execution_state": "NOT_RECORDED"}]},
        "seller_loss_analysis": {"decisions": [decision]},
        "operational_evidence": {"candidates": [{**rec, "contract_eligibility": "VERIFIED_OPERATIONAL"}]},
        "inventory_metric_integrity": {"validated_sales_signals": [{"store_id": source, "product_id": "P001", "sales_30d": 90,
            "retail_30d": {"semantics_evidence": "SOURCE_DOCUMENTED", "validation_status": "VALID"}}]}}
    return data, pipeline


def source_result(data, pipeline):
    result = fdi.build_final_decision_integration(data, pipeline)
    row = next(r for r in result["products"] if r["location_id"] == "S-SRC")
    return result, row, result["routes"][0] if result["routes"] else None


@pytest.mark.parametrize("case", ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","AA","AB","AC","AD","AE","AF","AG","AH","AI","AJ"])
def test_controlled_integration_contract(case):
    data, pipeline = bundle()
    decision = pipeline["seller_loss_analysis"]["decisions"][0]
    if case == "B": pipeline["shared_feasibility_selection"]["modes"]["STRICT_ACTUAL"]["plan_status"] = "INSUFFICIENT_CAP_DATA"
    if case in ("C", "D"):
        for plan in pipeline["shared_feasibility_selection"]["modes"].values():
            plan["rows"][0].update(selection_status="REJECTED_SELECTION_LIMIT" if case == "C" else "REJECTED_SOURCE_CAP", allocated_qty=0)
    if case == "E": pipeline["operational_evidence"] = {}
    if case in ("G", "H", "I"):
        index = {"G": 0, "H": 1, "I": 2}[case]
        data, pipeline = bundle(real_decision(controlled_scenarios()[index]["input"]))
    if case in ("J", "M", "N", "O", "P", "Q", "R"):
        field = {"J":"source_normal_price", "M":"source_normal_price", "N":"promotion_uplift", "O":"source_holding_cost_per_unit_day",
                 "P":"source_disposal_cost_per_unit", "Q":"remaining_shelf_life_days", "R":"source_daily_demand"}[case]
        data, pipeline = bundle(real_decision(replace(scenario_input(), **{field: loss.InputField()})))
    if case == "K": data, pipeline = bundle(real_decision(replace(scenario_input(), promotion_uplift=loss.InputField())))
    if case == "L":
        decision["input_requirements"] = {"comparison_scope": "TRANSFER_VS_NORMAL"}
        data["seller_loss_comparison_scope"] = "TRANSFER_VS_NORMAL"
    if case == "S":
        pipeline["inventory_metric_integrity"] = {}
        decision["input_provenance"]["source_daily_demand"].update(provenance="PROXY")
        decision["reason_codes"].append("PROXY_REJECTED:source_daily_demand")
        decision["recommendation_readiness"] = "UNAVAILABLE"
    if case in ("T", "U"):
        pipeline["recommendations"][0]["varo_action"] = "할인" if case == "T" else "보류"
        pipeline["action_consistency"]["records"][0].update(consistency_status="REVIEW_REQUIRED", conflict_codes=["I_LEGACY_VS_CANONICAL_MISMATCH"])
    if case == "W": data["is_test"] = True
    if case == "X": decision.update(decision_mode=sli.SCENARIO, evidence_level=sli.SCENARIO_EVIDENCE)
    if case == "Y": decision["recommendation_readiness"] = "UNAVAILABLE"
    if case in ("AB", "AE"):
        data["products"] = pd.concat([data["products"], pd.DataFrame([{"product_id": "P002"}])], ignore_index=True)
    if case == "AC":
        rec = {**pipeline["recommendations"][0], "route_id": "R002", "target_id": "S-OTHER"}
        pipeline["recommendations"].append(rec)
        for plan in pipeline["shared_feasibility_selection"]["modes"].values(): plan["rows"].append({**plan["rows"][0], "candidate_id": "R002"})
        pipeline["action_consistency"]["records"].append({**pipeline["action_consistency"]["records"][0], **rec})
    if case == "AD": pipeline["recommendations"] = []
    if case in ("AF", "AG", "AH", "AI"):
        decision["reason_codes"].append({"AF":"UNIT_MISMATCH:source_normal_price", "AG":"CURRENCY_MISMATCH:source_normal_price", "AH":"SELLER_UNIT_UNVERIFIABLE:source_daily_demand", "AI":"INPUT_CONFLICT"}[case])
        if case == "AI": decision["conflicting_fields"] = ["source_normal_price"]
        if case == "AH": pipeline["operational_evidence"]["candidates"][0]["expired_evidence"] = True
    before = deepcopy((data, pipeline))
    result, product, route = source_result(data, pipeline)
    assert result["coverage"]["pass"] and all(not r["production_action_applied"] for r in result["products"])
    assert fdi.digest(pipeline) == fdi.digest(before[1])
    pd.testing.assert_frame_equal(data["inventory"], before[0]["inventory"])
    if case in ("A", "G", "Z"):
        assert product["recommended_strategy"] == "TRANSFER" and product["promotion_status"] == gate.PROMOTION_ELIGIBLE
        assert not product["production_action_applied"]
    if case == "B":
        assert route["decision_mode"] == "BENCHMARK_PROXY" and "PROXY_PLAN_AVAILABLE" in route["status_flags"]
        assert product["recommended_strategy"] is None and "추정 조건" in product["explanation_ko"]
    if case in ("C", "D"):
        assert not route["selected"] and product["selected_transfer_routes"] == []
        if case == "D": assert "NO_FEASIBLE_TRANSFER" in product["status_flags"]
    if case in ("E", "T", "U", "AF", "AG", "AH", "AI"):
        assert product["decision_status"] == "REVIEW_REQUIRED" and product["recommended_strategy"] is None
    if case == "F": assert len(product["available_strategies"]) == 3 and all(v is not None for v in product["expected_loss_by_strategy"].values())
    if case == "F":
        # A verified financial demand input does not prove the forecast's history semantics.
        assert product["demand_validation_status"] == "ACTUAL_VERIFIED"
        assert product["forecast"]["input_meaning"] == "UNVERIFIED_AGGREGATE_OR_PROXY_INPUT"
    if case == "H": assert product["recommended_strategy"] == "NORMAL_SALE"
    if case == "I": assert product["recommended_strategy"] == "DISCOUNT_SALE"
    if case in ("J", "M", "Q", "R"):
        assert product["recommended_strategy"] is None
        assert product["seller_loss_status"] == loss.STATUS_UNAVAILABLE
        assert all(v is None for v in product["expected_loss_by_strategy"].values())
    if case in ("K", "N"):
        assert product["seller_loss_status"] == loss.STATUS_PARTIAL
        assert product["expected_loss_by_strategy"]["DISCOUNT_SALE"] is None
        assert product["promotion_status"] != gate.PROMOTION_ELIGIBLE
    if case in ("O", "P"): assert product["recommended_strategy"] is None and route["seller_loss"]["unknown_inputs"]
    if case == "L": assert product["excluded_strategies"] == ["DISCOUNT_SALE"]
    if case == "S": assert product["recommended_strategy"] is None and product["demand_validation_status"] == "PROXY"
    if case == "V": assert product["canonical_action"] == "TRANSFER"
    if case in ("W", "X"):
        assert product["decision_mode"] == "SCENARIO" and product["recommended_strategy"] is None
        assert "실제 운영 추천이 아닙니다" in product["explanation_ko"]
        assert product["input_quality"]["effective_evidence"] == "SCENARIO"
    if case == "Y": assert product["promotion_status"] != gate.PROMOTION_ELIGIBLE
    if case == "AA": assert product["operator_action"] is None and route["operator_action"] is None
    if case in ("AB", "AE"): assert any(r["product_id"] == "P002" for r in result["products"])
    if case == "AC":
        assert product["allocated_qty"] == 40
        assert all(v is None for v in product["expected_loss_by_strategy"].values())
        assert product["recommended_strategy"] is None
    if case == "AD": assert result["routes"] == [] and product["transfer_candidate_status"] == "NO_CANDIDATE"
    if case == "AJ": assert fdi.digest(result) == fdi.digest(fdi.build_final_decision_integration(data, pipeline))


@pytest.mark.parametrize("scenario", controlled_scenarios(), ids=lambda s: s["id"])
def test_existing_eighteen_engine_results_preserved(scenario):
    expected = loss.evaluate_seller_decision(scenario["input"])
    decision = sli.evaluate_with_seller_inputs(scenario["input"], None, decision_mode=sli.SCENARIO, decision_date=DATE)
    data, pipeline = bundle(decision)
    rec = pipeline["recommendations"][0]
    rec.update(route_id=decision["decision_id"], product_id=decision["product_id"])
    data["inventory"]["product_id"] = decision["product_id"]
    data["products"]["product_id"] = decision["product_id"]
    for plan in pipeline["shared_feasibility_selection"]["modes"].values(): plan["rows"][0]["candidate_id"] = decision["decision_id"]
    pipeline["action_consistency"]["records"] = []
    data["is_test"] = True
    result = fdi.build_final_decision_integration(data, pipeline)
    sl = result["routes"][0]["seller_loss"]
    assert sl["selected_strategy"] == expected["recommended_strategy"]
    assert sl["comparison_status"] == expected["comparison_status"]
    for strategy in loss.STRATEGIES:
        assert sl["expected_loss_by_strategy"][strategy] == expected["strategies"][strategy]["expected_loss"]
    assert result["routes"][0]["recommended_strategy"] is None
    assert result["routes"][0]["promotion_status"] != gate.PROMOTION_ELIGIBLE


@pytest.mark.parametrize("missing", ["shared_feasibility_selection", "action_consistency", "inventory_metric_integrity", "seller_loss_analysis", "operational_evidence"])
def test_missing_optional_analysis_does_not_crash(missing):
    data, pipeline = bundle()
    pipeline.pop(missing)
    result = fdi.safe_build_final_decision_integration(data, pipeline)
    assert result["status"] == "parallel_only" and result["coverage"]["pass"]


def test_same_product_different_locations_not_duplicate():
    data, pipeline = bundle()
    result = fdi.build_final_decision_integration(data, pipeline)
    assert len(result["products"]) == 2 and result["summary"]["unique_products"] == 1
    assert result["coverage"]["duplicates"] == []
    assert sum(len(r["selected_transfer_routes"]) for r in result["products"]) == 1


def test_duplicate_input_not_summed_and_no_silent_first_winner():
    data, pipeline = bundle()
    data["inventory"] = pd.concat([data["inventory"], data["inventory"].iloc[[0]]], ignore_index=True)
    _, row, route = source_result(data, pipeline)
    assert row["current_stock"] is None and row["stock_validation_status"] == "CONFLICTED"
    assert row["recommended_strategy"] is None and route["recommended_strategy"] is None


def test_invalid_keys_and_zero_products_visible():
    data = {"inventory": pd.DataFrame([{"product_id": None}, {"product_id": None}])}
    result = fdi.safe_build_final_decision_integration(data, {})
    assert len(result["products"]) == 2 and result["coverage"]["pass"]
    assert fdi.safe_build_final_decision_integration({}, {})["products"] == []


def test_catalog_coverage_fault_detection():
    data, pipeline = bundle()
    catalog = fdi.product_catalog(data)
    rows = fdi.build_final_decision_integration(data, pipeline)["products"]
    assert "MISSING_DECISION_ROW" in fdi.coverage_check(catalog, rows[:-1])["reason_codes"]
    assert "DUPLICATE_DECISION_ROW" in fdi.coverage_check(catalog, rows + rows[:1])["reason_codes"]


def test_fault_isolation_keeps_all_products():
    data, pipeline = bundle()
    with patch.object(fdi, "business_context", side_effect=RuntimeError("bad optional input")):
        result = fdi.safe_build_final_decision_integration(data, pipeline)
    assert result["coverage"]["pass"] and all(r["recommended_strategy"] is None for r in result["products"])
    with patch.object(fdi, "build_final_decision_integration", side_effect=RuntimeError("fixture failure")):
        result = fdi.safe_build_final_decision_integration(data, pipeline)
    assert result["status"] == "error" and len(result["products"]) == 2


def test_pipeline_old_contract_byte_for_byte_unchanged():
    from services.analysis_pipeline import run_analysis_pipeline
    data = sample_workbook()
    with patch("services.analysis_pipeline.safe_build_final_decision_integration", return_value={}):
        before = run_analysis_pipeline(data, detail_level="core").to_dict()
    after = run_analysis_pipeline(data, detail_level="core").to_dict()
    integrated = after.pop("final_decision_integration")
    before.pop("final_decision_integration")
    assert fdi.digest(before) == fdi.digest(after)
    assert integrated["status"] == "parallel_only" and integrated["coverage"]["pass"]


def test_partial_allocation_not_same_loss_problem():
    data, pipeline = bundle()
    pipeline["shared_feasibility_selection"]["modes"]["STRICT_ACTUAL"]["rows"][0]["allocated_qty"] = 10
    _, product, route = source_result(data, pipeline)
    assert product["recommended_strategy"] is None and "SELLER_LOSS_PLAN_QUANTITY_BASIS_DIFFERS" in route["reason_codes"]


def test_duplicate_route_cannot_double_count_allocated_quantity():
    data, pipeline = bundle()
    pipeline["recommendations"].append(deepcopy(pipeline["recommendations"][0]))
    result, product, route = source_result(data, pipeline)
    assert product["allocated_qty"] == 20 and len(result["routes"]) == 1
    assert product["recommended_strategy"] is None and "DUPLICATE_ROUTE_INPUT" in route["reason_codes"]


def test_unknown_price_never_zero_and_no_actual_effects():
    data, pipeline = bundle(real_decision(replace(scenario_input(), source_normal_price=loss.InputField())))
    _, product, route = source_result(data, pipeline)
    assert route["expected_loss_values"]["NORMAL_SALE"]["value"] is None
    assert product["actual_saving"] is None and product["actual_spend"] is None and product["outcome"] is None


def test_hold_not_normal_sale():
    data, pipeline = bundle()
    pipeline["recommendations"][0]["varo_action"] = "보류"
    _, product, _ = source_result(data, pipeline)
    assert product["legacy_action"] == "보류"
    assert product["recommended_strategy"] != "NORMAL_SALE"


def test_namespace_ambiguity_cannot_link_loss_implicitly():
    data, pipeline = bundle()
    data["inventory"]["dataset_id"] = "A"
    second = data["inventory"].iloc[[0]].assign(dataset_id="B")
    data["inventory"] = pd.concat([data["inventory"], second], ignore_index=True)
    result = fdi.build_final_decision_integration(data, pipeline)
    assert result["routes"] == [] and result["orphan_routes"][0]["reason"] == "ROUTE_PRODUCT_NAMESPACE_AMBIGUOUS"
    assert result["coverage"]["pass"]


@pytest.mark.parametrize("sheet", ["seller_loss_inputs", "seller_business_profile"])
def test_malformed_optional_sheet_blocks_new_recommendation_only(sheet):
    data, pipeline = bundle()
    data[sheet] = "invalid table"
    result, product, _ = source_result(data, pipeline)
    assert "OPTIONAL_INPUT_INVALID" in result["context_errors"]
    assert product["recommended_strategy"] is None
    assert result["coverage"]["pass"]
    data, pipeline = bundle()
    data["seller_loss_decision_mode"] = "invalid"
    result, product, _ = source_result(data, pipeline)
    assert "INVALID_DECISION_MODE" in result["context_errors"] and product["recommended_strategy"] is None


def test_zero_stock_no_action_no_unnecessary_price_questions():
    data, pipeline = bundle()
    data["inventory"].loc[0, "stock_qty"] = 0
    pipeline["recommendations"] = []
    _, product, _ = source_result(data, pipeline)
    assert product["decision_status"] == "NO_ACTION_REQUIRED"
    assert product["missing_required_inputs"] == []
    assert product["recommended_next_step"] == "현재 조치 불필요"


def test_invalid_stock_not_filled_to_zero():
    data, pipeline = bundle()
    data["inventory"]["stock_qty"] = data["inventory"]["stock_qty"].astype(object)
    data["inventory"].loc[0, "stock_qty"] = "invalid"
    _, product, _ = source_result(data, pipeline)
    assert product["current_stock"] is None and product["decision_status"] == "INSUFFICIENT_DATA"
    assert product["recommended_strategy"] is None


def test_declared_plan_unit_mismatch_blocks_winner():
    data, pipeline = bundle()
    pipeline["shared_feasibility_selection"]["modes"]["STRICT_ACTUAL"]["rows"][0]["quantity_unit"] = "kg"
    _, product, route = source_result(data, pipeline)
    assert product["recommended_strategy"] is None and "PLAN_SELLER_UNIT_MISMATCH" in route["reason_codes"]


def test_pipeline_validation_error_retains_all_products():
    from services.analysis_pipeline import build_v2_state
    data = sample_workbook()
    data["inventory"] = data["inventory"].drop(columns="stock_qty")
    state = build_v2_state(data)
    result = state["pipeline_result"]["final_decision_integration"]
    assert state["pipeline_result"]["status"] == "validation_error"
    assert result["coverage"]["pass"] and result["products"]
    assert all(r["recommended_strategy"] is None for r in result["products"])
