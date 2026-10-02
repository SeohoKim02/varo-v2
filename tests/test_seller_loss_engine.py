import copy
import json
import math
import re
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from services import analysis_pipeline
from services.analysis_pipeline import build_v2_state
from services.data_loader import load_excel_data
from services.seller_decision_validation import SCENARIO_DATASET, controlled_scenarios, scenario_input
from services.seller_loss_engine import (
    COMPONENTS, DISCOUNT_SALE, NORMAL_SALE, STATUS_FULL, STATUS_PARTIAL, STATUS_UNAVAILABLE, TRANSFER,
    InputField, contract_document, evaluate_seller_decision, known, legacy_agreement, unit_days, units_sold,
)

REPO = Path(__file__).resolve().parents[1]
NETWORK_SAMPLE = REPO / "data" / "Varo_V2_네트워크_샘플.xlsx"


def q(value, source="test"):
    return known(value, "USER_INPUT", source, dataset=SCENARIO_DATASET)


def m(value, source="test", currency="KRW"):
    return known(value, "USER_INPUT", source, currency=currency, dataset=SCENARIO_DATASET)


def run(**overrides):
    return evaluate_seller_decision(scenario_input(**overrides))


# ---------------------------------------------------------------- formulas (hand-computed)


def test_fluid_helpers():
    assert units_sold(10, 3, 2) == 6 and units_sold(10, 3, 5) == 10 and units_sold(10, 0, 5) == 0
    assert unit_days(50, 3, 10) == pytest.approx(50 * 10 - 3 * 100 / 2)   # not sold out within the horizon
    assert unit_days(30, 3, 10) == pytest.approx(30 * 10 / 2)             # sells out exactly at the horizon
    assert unit_days(5, 0, 4) == 20 and unit_days(0, 3, 4) == 0 and unit_days(5, 3, 0) == 0


def test_normal_sale_formula():
    # S=50, Q=20 (B=30), r=3/day, H=10 -> store sells 30 = the base stock, the 20 decision units stay unsold.
    result = run()
    normal = result["strategies"][NORMAL_SALE]
    assert normal["quantities"]["expected_sold_qty"] == 0 and normal["quantities"]["expected_unsold_qty"] == 20
    assert normal["quantities"]["holding_unit_days"] == pytest.approx((50 * 10 - 150) - 150)
    assert normal["components"]["opportunity_loss"] == 20 * 1000
    assert normal["components"]["holding_loss"] == pytest.approx(200 * 5)
    assert normal["components"]["disposal_loss"] == 20 * 200
    assert normal["components"]["discount_loss"] == 0 and normal["components"]["transfer_cost"] == 0
    assert result["expected_loss_normal_sale"] == 25000


def test_discount_sale_formula_charges_markdown_and_cannibalisation():
    # Discounted lot sells first at 3 x 1.5 = 4.5/day: 20 units in 4.44 days; base stock then sells 3/day for 5.56 days.
    result = run()
    discount = result["strategies"][DISCOUNT_SALE]
    first = 20 / 4.5
    base_sold = 3 * (10 - first)
    assert discount["quantities"]["discounted_units_sold"] == 20
    assert discount["quantities"]["cannibalized_regular_units"] == pytest.approx(30 - base_sold, abs=1e-4)
    net_sold = 20 + base_sold - 30
    assert discount["quantities"]["expected_sold_qty"] == pytest.approx(net_sold, abs=1e-4)
    assert discount["components"]["discount_loss"] == pytest.approx(1000 * 0.3 * 20)
    assert discount["components"]["opportunity_loss"] == pytest.approx(1000 * (20 - net_sold), abs=0.01)
    holding_days = 20 * first / 2 + 30 * first + (30 * (10 - first) - 3 * (10 - first) ** 2 / 2) - 150
    assert discount["components"]["holding_loss"] == pytest.approx(5 * holding_days, abs=0.01)
    assert discount["discounted_price"] == 700


def test_transfer_formula_and_constraints():
    # Target: stock 5, 4/day over 10 - 0.1 days -> need 34.6, so all 20 move and all sell at the target.
    result = run()
    transfer = result["strategies"][TRANSFER]
    assert transfer["quantities"]["moved_qty"] == 20 and transfer["quantities"]["target_sold_qty"] == 20
    assert transfer["quantities"]["target_horizon_need"] == pytest.approx(4 * 9.9 - 5)
    target_days = (25 * 25 / 4 / 2) - (5 * 5 / 4 / 2)
    assert transfer["components"]["transfer_cost"] == 1800
    assert transfer["components"]["holding_loss"] == pytest.approx(5 * target_days)
    assert transfer["components"]["opportunity_loss"] == 0
    assert result["expected_loss_transfer"] == pytest.approx(1800 + 5 * target_days)
    assert result["recommended_strategy"] == TRANSFER and result["comparison_status"] == STATUS_FULL


def test_quantity_constraints_cap_transfer_and_sales():
    result = run(source_surplus_cap=q(12), target_need_cap=q(15), route_capacity_qty=q(30))
    transfer = result["strategies"][TRANSFER]
    assert transfer["quantities"]["moved_qty"] == 12 and transfer["binding_constraints"] == ["source_surplus_cap"]
    assert transfer["quantities"]["kept_qty"] == 8
    assert "PARTIAL_TRANSFER_REMAINDER_NORMAL_SALE" in result["reason_codes"]
    capped = run(route_capacity_qty=q(7))["strategies"][TRANSFER]
    assert capped["quantities"]["moved_qty"] == 7 and capped["binding_constraints"] == ["route_capacity_qty"]
    for scenario in controlled_scenarios():
        out = evaluate_seller_decision(scenario["input"])
        for strategy in out["comparable_strategies"]:
            quantities = out["strategies"][strategy]["quantities"]
            assert -1e-9 <= quantities["expected_sold_qty"] <= quantities["decision_qty"] + 1e-9
            assert quantities["expected_unsold_qty"] >= -1e-9
            if strategy == TRANSFER:
                assert quantities["moved_qty"] <= quantities["decision_qty"]


def test_decision_quantity_above_stock_is_refused():
    result = run(decision_qty=q(60))
    assert result["comparison_status"] == STATUS_UNAVAILABLE and "DECISION_QTY_EXCEEDS_STOCK" in result["reason_codes"]


# ---------------------------------------------------------------- accounting: no double counting


def test_components_sum_exactly_to_expected_loss():
    for scenario in controlled_scenarios():
        result = evaluate_seller_decision(scenario["input"])
        for strategy in result["comparable_strategies"]:
            payload = result["strategies"][strategy]
            total = sum(payload["component_known_part"][name] for name in COMPONENTS)
            assert total == pytest.approx(payload["expected_loss_excluding_unknown"], abs=0.05)
            if payload["expected_loss"] is not None:
                assert sum(payload["components"].values()) == pytest.approx(payload["expected_loss"], abs=0.05)


def test_unit_cost_is_sunk_and_never_changes_losses():
    base = run()
    for cost in (None, 0, 600, 999999):
        other = run(unit_cost=m(cost) if cost is not None else InputField())
        for strategy in (TRANSFER, NORMAL_SALE, DISCOUNT_SALE):
            assert other["strategies"][strategy]["components"] == base["strategies"][strategy]["components"]
        assert other["recommended_strategy"] == base["recommended_strategy"]
    assert "UNIT_COST_SUNK_NOT_IN_RANKING" in base["reason_codes"]


def test_transfer_bridge_reconciles_without_adding_source_opportunity_twice():
    # Source sells everything at 6/day, the target sells everything too: moving the units shifts the sale, it does not lose it.
    result = run(source_daily_demand=q(6))
    transfer, normal = result["strategies"][TRANSFER], result["strategies"][NORMAL_SALE]
    bridge = transfer["transfer_bridge"]
    assert bridge["source_opportunity_units"] == 20 and bridge["source_opportunity_loss"] == 20000
    assert bridge["target_recovered_value"] == 20000
    difference = result["expected_loss_transfer"] - result["expected_loss_normal_sale"]
    holding = transfer["components"]["holding_loss"] - normal["components"]["holding_loss"]
    disposal = transfer["components"]["disposal_loss"] - normal["components"]["disposal_loss"]
    assert difference == pytest.approx(1800 + bridge["source_opportunity_loss"] - bridge["target_recovered_value"] + holding + disposal, abs=0.05)
    assert transfer["components"]["opportunity_loss"] == 0   # not charged p x 20 on top of the shifted sale


def test_disposal_charges_only_handling_cost_not_the_goods_again():
    result = run()
    normal = result["strategies"][NORMAL_SALE]["components"]
    assert normal["disposal_loss"] == 20 * 200          # cash paid to dispose
    assert normal["opportunity_loss"] == 20 * 1000      # the unsold goods' value, counted once
    salvage = run(salvage_value_per_unit=m(100))["strategies"][NORMAL_SALE]["components"]
    assert salvage["salvage_recovery"] == -20 * 100


# ---------------------------------------------------------------- strict mode, partial, unavailable


@pytest.mark.parametrize("scenario", controlled_scenarios(), ids=lambda s: s["id"])
def test_controlled_scenarios(scenario):
    result = evaluate_seller_decision(scenario["input"])
    expect = scenario["expect"]
    for key in ("comparison_status", "recommended_strategy", "recommendation_status", "tie"):
        if key in expect:
            assert result[key] == expect[key], (scenario["id"], key, result[key])
    if "unavailable" in expect:
        assert list(result["unavailable_strategies"]) == [expect["unavailable"]]
    if "reason" in expect:
        assert expect["reason"] in result["reason_codes"] or any(
            expect["reason"] in reasons for reasons in result["unavailable_strategies"].values())


def test_missing_price_blocks_everything_without_defaults():
    result = run(source_normal_price=InputField(source="absent"))
    assert result["comparison_status"] == STATUS_UNAVAILABLE and result["recommended_strategy"] is None
    assert result["expected_loss_transfer"] is None and result["expected_loss_normal_sale"] is None
    assert result["expected_loss_discount_sale"] is None and "PRICE_MISSING" in result["reason_codes"]
    assert result["explanation"].startswith("금액 비교 불가")


def test_missing_transfer_cost_and_target_demand_make_transfer_unavailable():
    for overrides, code in ((dict(transfer_cost=InputField()), "TRANSFER_COST_MISSING"),
                            (dict(target_daily_demand=InputField()), "TARGET_DEMAND_MISSING"),
                            (dict(transit_time_days=InputField()), "TRANSIT_TIME_MISSING"),
                            (dict(target_store_id=None), "TARGET_STORE_MISSING")):
        result = run(**overrides)
        assert result["comparison_status"] == STATUS_PARTIAL
        assert result["comparable_strategies"] == [NORMAL_SALE, DISCOUNT_SALE]
        assert code in result["unavailable_strategies"][TRANSFER]
        assert result["expected_loss_transfer"] is None


def test_missing_uplift_never_falls_back_to_the_legacy_80_percent():
    result = run(promotion_uplift=InputField(source="config.promotion_sales_increase_rate absent"))
    assert result["unavailable_strategies"] == {DISCOUNT_SALE: ["UPLIFT_MISSING"]}
    assert result["strategies"][DISCOUNT_SALE]["quantities"] == {}
    rate = run(discount_rate=known(1.2, "CONFIG", "bad"))
    assert rate["unavailable_strategies"] == {DISCOUNT_SALE: ["INVALID_DISCOUNT_RATE"]}


def test_partial_comparison_is_never_presented_as_three_way():
    result = run(promotion_uplift=InputField())
    assert result["comparison_status"] == STATUS_PARTIAL
    assert "두 가지만 비교" in result["explanation"] and "할인 판매 제외" in result["explanation"]
    assert "세 가지" not in result["explanation"]


def test_transfer_only_currency_problem_keeps_the_other_two_comparable():
    result = run(target_normal_price=m(1, currency="USD"))
    assert result["comparison_status"] == STATUS_PARTIAL
    assert "CURRENCY_MISMATCH" in result["unavailable_strategies"][TRANSFER]
    core = run(source_holding_cost_per_unit_day=m(5, currency="USD"))
    assert core["comparison_status"] == STATUS_UNAVAILABLE and "CURRENCY_MISMATCH" in core["reason_codes"]
    undeclared = run(source_disposal_cost_per_unit=known(200, "USER_INPUT", "no currency", dataset=SCENARIO_DATASET))
    assert "CURRENCY_UNDECLARED" in undeclared["reason_codes"]


def test_unit_rules():
    assert "UNIT_MISMATCH" in run(target_current_stock=known(5, "USER_INPUT", "kg", unit="kg", dataset=SCENARIO_DATASET))["unavailable_strategies"][TRANSFER]
    assert "UNIT_MISMATCH" in run(source_daily_demand=known(3, "USER_INPUT", "ton", unit="ton", dataset=SCENARIO_DATASET))["reason_codes"]
    undeclared = run(quantity_unit=None)
    assert undeclared["comparison_status"] == STATUS_FULL and "UNIT_UNDECLARED_SINGLE_SOURCE" in undeclared["reason_codes"]
    mixed = run(quantity_unit=None, source_current_stock=known(50, "USER_INPUT", "other upload", dataset="OTHER"))
    assert mixed["comparison_status"] == STATUS_UNAVAILABLE
    assert {"UNIT_UNVERIFIABLE_ACROSS_SOURCES", "CROSS_DATASET_INPUT"} <= set(mixed["reason_codes"])


def test_zero_inventory_zero_demand_negative_sales():
    assert run(decision_qty=q(0), source_current_stock=q(0))["reason_codes"][:1] == ["ZERO_INVENTORY"]
    zero = run(source_daily_demand=q(0), target_store_id=None)
    assert zero["strategies"][NORMAL_SALE]["quantities"]["expected_sold_qty"] == 0
    assert zero["strategies"][DISCOUNT_SALE]["quantities"]["expected_sold_qty"] == 0   # uplift of zero demand is zero
    assert zero["tie"] is True and zero["recommended_strategy"] == NORMAL_SALE
    assert "TIE_BROKEN_BY_OPERATIONAL_SIMPLICITY" in zero["reason_codes"]
    negative = run(source_daily_demand=q(-2))
    assert negative["comparison_status"] == STATUS_UNAVAILABLE and "NEGATIVE_INPUT:source_daily_demand" in negative["reason_codes"]
    negative_target = run(target_daily_demand=q(-1))
    assert negative_target["comparison_status"] == STATUS_PARTIAL
    assert "NEGATIVE_INPUT:target_daily_demand" in negative_target["unavailable_strategies"][TRANSFER]


def test_proxy_values_are_rejected_not_relabelled():
    result = run(source_daily_demand=known(3, "PROXY", "outbound", dataset=SCENARIO_DATASET))
    assert result["reason_codes"] == ["PROXY_REJECTED:source_daily_demand"]
    holding = run(source_holding_cost_per_unit_day=known(5, "PROXY", "tariff", currency="KRW", dataset=SCENARIO_DATASET))
    assert "PROXY_REJECTED:source_holding_cost_per_unit_day" in holding["reason_codes"]
    assert "source_holding_cost_per_unit_day" in holding["unknown_inputs"]


def test_unknown_rates_stay_unknown_and_need_a_robust_ranking():
    robust = run(source_holding_cost_per_unit_day=InputField())
    assert robust["recommendation_status"] == "RECOMMENDED" and robust["recommended_strategy"] == TRANSFER
    assert robust["expected_loss_normal_sale"] is None              # never computed with an assumed 0
    assert robust["strategies"][NORMAL_SALE]["expected_loss_range"][1] is None
    assert robust["unknown_input_bounds"] == {"source_holding_cost_per_unit_day": [0.0, None]}
    assert robust["decision_confidence"] == "MEDIUM"
    fragile = run(source_holding_cost_per_unit_day=InputField(), target_holding_cost_per_unit_day=InputField())
    assert fragile["recommended_strategy"] is None and "RANKING_DEPENDS_ON_UNKNOWN" in fragile["reason_codes"]
    salvage = run(salvage_value_per_unit=InputField())
    assert salvage["unknown_input_bounds"]["salvage_value_per_unit"] == [0.0, 1000.0]


def test_robust_margin_is_the_worst_case_over_the_unknown_range():
    result = run(source_holding_cost_per_unit_day=InputField())
    # At holding 0 NORMAL loses 24,000 and DISCOUNT ~22,000; TRANSFER keeps 0 source unit-days -> margin at h_s = 0.
    discount_at_zero = result["strategies"][DISCOUNT_SALE]["expected_loss_excluding_unknown"]
    assert result["loss_difference_vs_second_best"] == pytest.approx(discount_at_zero - result["expected_loss_transfer"], abs=0.01)


# ---------------------------------------------------------------- dominance / sanity sweeps


def test_transfer_cost_monotonicity():
    previous, lost = None, False
    for cost in (0, 500, 1800, 5000, 20000, 23000, 50000):
        result = run(transfer_cost=known(cost, "USER_INPUT", "sweep", currency="KRW"))
        loss = result["expected_loss_transfer"]
        if previous is not None:
            assert loss >= previous - 1e-9
        if lost:
            assert result["recommended_strategy"] != TRANSFER
        lost = lost or result["recommended_strategy"] != TRANSFER
        previous = loss
    assert lost


def test_discount_rate_monotonicity():
    previous = None
    for rate in (0.05, 0.1, 0.2, 0.3, 0.5, 0.9):
        result = run(discount_rate=known(rate, "CONFIG", "sweep"))
        payload = result["strategies"][DISCOUNT_SALE]
        if previous is not None:
            assert payload["components"]["discount_loss"] >= previous[0] - 1e-9
            assert payload["expected_loss"] >= previous[1] - 1e-9
        previous = (payload["components"]["discount_loss"], payload["expected_loss"])


def test_higher_disposal_risk_never_favours_keeping():
    def gaps(result):
        normal = result["expected_loss_normal_sale"]
        return normal - result["expected_loss_transfer"], normal - result["expected_loss_discount_sale"]

    previous = None
    for cost in (0, 100, 200, 500, 2000):
        current = gaps(run(source_disposal_cost_per_unit=m(cost), target_disposal_cost_per_unit=m(cost)))
        if previous is not None:
            assert current[0] >= previous[0] - 1e-9 and current[1] >= previous[1] - 1e-9
        previous = current
    previous = None
    for shelf_life in (12, 10, 8, 5, 3, 1):
        result = run(remaining_shelf_life_days=q(shelf_life))
        unsold = result["strategies"][NORMAL_SALE]["quantities"]["expected_unsold_qty"]
        if previous is not None:
            assert unsold >= previous - 1e-9
        previous = unsold


def test_target_demand_monotonicity_at_fixed_quantity():
    previous = None
    for rate in (0.5, 1, 2, 3, 4, 8):
        transfer = run(target_daily_demand=q(rate), route_capacity_qty=q(1), target_current_stock=q(0))["strategies"][TRANSFER]
        if previous is not None:
            assert transfer["quantities"]["target_sold_qty"] >= previous - 1e-9
        previous = transfer["quantities"]["target_sold_qty"]
    previous = None
    for rate in (0.5, 1, 2, 3, 4, 8):
        transfer = run(target_daily_demand=q(rate))["strategies"][TRANSFER]
        if transfer["available"]:
            if previous is not None:
                assert transfer["quantities"]["target_sold_qty"] >= previous - 1e-9
            previous = transfer["quantities"]["target_sold_qty"]


def test_source_demand_never_lowers_source_opportunity_loss():
    previous = None
    for rate in (0, 1, 2, 3, 3.5, 4, 5, 6, 10):
        bridge = run(source_daily_demand=q(rate))["strategies"][TRANSFER]["transfer_bridge"]
        if previous is not None:
            assert bridge["source_opportunity_loss"] >= previous - 1e-9
        previous = bridge["source_opportunity_loss"]


# ---------------------------------------------------------------- determinism, provenance, explanation


def test_deterministic_result():
    first = json.dumps(run(), sort_keys=True, ensure_ascii=False, default=str)
    second = json.dumps(run(), sort_keys=True, ensure_ascii=False, default=str)
    assert first == second


def test_provenance_preserved_for_every_field():
    inp = scenario_input(target_daily_demand=known(4, "DERIVED_FROM_USER_INPUT", "demand_forecast_router[v2]", dataset=SCENARIO_DATASET))
    result = evaluate_seller_decision(inp)
    for name, item in inp.input_fields().items():
        assert result["input_provenance"][name] == item.as_dict()
    assert result["input_provenance"]["target_daily_demand"]["source"] == "demand_forecast_router[v2]"
    assert result["input_provenance"]["salvage_value_per_unit"]["provenance"] == "USER_INPUT"
    assert result["scenario_inputs"] == ["discount_rate", "promotion_uplift"]
    assert "SCENARIO_INPUT:promotion_uplift" in result["reason_codes"]
    assert known(None, "DIRECT_REAL", "x").provenance == "MISSING"


def _amounts(text):
    return [int(value.replace(",", "")) for value in re.findall(r"([\d,]+)원", text)]


def test_explanation_matches_numeric_result():
    for scenario in controlled_scenarios():
        result = evaluate_seller_decision(scenario["input"])
        text, facts = result["explanation"], result["explanation_facts"]
        winner = result["recommended_strategy"]
        if not winner:
            assert facts["status"] == result["comparison_status"]
            continue
        labels = {TRANSFER: "재고 이동", NORMAL_SALE: "정상 판매 유지", DISCOUNT_SALE: "할인 판매"}
        assert f"{labels[winner]}의 예상 손실이" in text
        amounts = _amounts(text)
        assert round(facts["winner_loss"]) in amounts
        assert round(result["loss_difference_vs_second_best"]) in amounts or result["tie"]
        if facts.get("winner_top_value"):
            assert round(facts["winner_top_value"]) in amounts
        if facts.get("runner_up_driver_value"):
            assert round(facts["runner_up_driver_value"]) in amounts
        for name in result["unknown_inputs"]:
            assert "데이터가 없어 금액에 넣지 않았으며" in text


def test_explanation_example_sentence():
    result = run()
    assert result["explanation"].startswith("같은 재고 20ea에 대해 세 가지 처리 방법을 비교했습니다.")
    assert "재고 이동의 예상 손실이 2,175원으로 가장 작습니다." in result["explanation"]
    assert "재고 이동은 이동 비용 1,800원이 발생하지만" in result["explanation"]


def test_contract_document_is_stable_and_complete():
    first, second = contract_document(), contract_document()
    assert first == second and len(first["contract_signature"]) == 64
    assert {"decision_qty", "source_daily_demand", "target_daily_demand", "promotion_uplift", "salvage_value_per_unit",
            "transfer_cost", "remaining_shelf_life_days"} <= set(first["input_fields"])
    assert first["criterion"] == "EXPECTED_AVOIDABLE_LOSS_MIN" and "PROXY" not in first["accepted_in_strict_mode"]


def test_legacy_agreement_mapping():
    assert legacy_agreement("재고 이동", TRANSFER) == "SAME"
    assert legacy_agreement("할인", TRANSFER) == "DIFFERENT"
    assert legacy_agreement("긴급 할인", DISCOUNT_SALE) == "SAME"
    assert legacy_agreement("보류", NORMAL_SALE) == "SAME"
    assert legacy_agreement("폐기", NORMAL_SALE) == "LEGACY_ACTION_OUTSIDE_SCOPE"
    assert legacy_agreement("1+1", DISCOUNT_SALE) == "LEGACY_ACTION_OUTSIDE_SCOPE"
    assert legacy_agreement("재고 이동", None) == "NO_SELLER_LOSS_RECOMMENDATION"


# ---------------------------------------------------------------- pipeline: parallel only, legacy untouched


def _load(path=NETWORK_SAMPLE):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_excel_data(path)


def test_pipeline_runs_seller_loss_in_parallel_without_touching_legacy_outputs():
    data = _load()
    with_engine = build_v2_state(copy.deepcopy(data), detail_level="full")
    with mock.patch.object(analysis_pipeline, "build_seller_loss_analysis", return_value={"status": "disabled"}):
        without_engine = build_v2_state(copy.deepcopy(data), detail_level="full")
    assert with_engine["recommendations"] == without_engine["recommendations"]
    assert with_engine["pipeline_result"]["summary"] == without_engine["pipeline_result"]["summary"]
    assert with_engine["pipeline_result"]["top5"] == without_engine["pipeline_result"]["top5"]
    assert with_engine["pipeline_result"]["connected_algorithms"] == without_engine["pipeline_result"]["connected_algorithms"]
    analysis = with_engine["pipeline_result"]["seller_loss_analysis"]
    assert analysis["status"] == "parallel_only" and analysis["legacy_action_replaced"] is False
    by_route = {item["route_id"]: item for item in with_engine["recommendations"]}
    assert len(analysis["rows"]) == len(by_route)
    for row, decision in zip(analysis["rows"], analysis["decisions"]):
        assert row["legacy_action"] == by_route[row["decision_id"]]["varo_action"]
        assert decision["legacy_action_changed"] is False


def test_pipeline_uses_forecast_for_the_target_store_and_no_promotion_placeholders():
    data = _load()
    state = build_v2_state(copy.deepcopy(data), detail_level="core")
    from services.analysis_pipeline import _Runner, _run_inventory_analysis, PipelineResult
    from services.legacy_adapters.data_adapter import prepare_legacy_data

    analyzed, _ = _run_inventory_analysis(_Runner(PipelineResult()), prepare_legacy_data(data)["inventory"])
    forecast = {(str(r["store_id"]), str(r["product_id"])): r["demand_forecast_daily"] for r in analyzed.to_dict("records")}
    decisions = state["pipeline_result"]["seller_loss_analysis"]["decisions"]
    assert decisions
    for decision in decisions:
        target = decision["input_provenance"]["target_daily_demand"]
        assert target["source"].startswith("demand_forecast_router[v1]")
        assert target["value"] == pytest.approx(forecast[(decision["target_store_id"], decision["product_id"])])
        assert decision["input_provenance"]["source_daily_demand"]["provenance"] == "DERIVED_FROM_USER_INPUT"
    # No config row -> the legacy 20% / 80% defaults are never used by the seller-loss engine.
    no_config = copy.deepcopy(data)
    no_config["config"] = no_config["config"][~no_config["config"]["key"].astype(str).str.startswith("promotion")]
    decisions = build_v2_state(no_config, detail_level="core")["pipeline_result"]["seller_loss_analysis"]["decisions"]
    for decision in decisions:
        assert decision["input_provenance"]["promotion_uplift"]["value"] is None
        reasons = decision["strategies"][DISCOUNT_SALE]["unavailable_reasons"]
        assert "UPLIFT_MISSING" in reasons or decision["comparison_status"] == STATUS_UNAVAILABLE


def test_pipeline_recommends_only_when_data_is_sufficient():
    data = _load()
    sparse = build_v2_state(copy.deepcopy(data), detail_level="core")["pipeline_result"]["seller_loss_analysis"]
    enriched = copy.deepcopy(data)
    enriched["inventory"]["salvage_value_per_unit"] = 0   # an explicit seller input, not an engine default
    full = build_v2_state(enriched, detail_level="core")["pipeline_result"]["seller_loss_analysis"]
    assert sparse["recommended_count"] < full["recommended_count"]
    for decision in sparse["decisions"]:
        assert "salvage_value_per_unit" in decision["unknown_inputs"] or decision["comparison_status"] == STATUS_UNAVAILABLE \
            or decision["recommended_strategy"] is not None
    for row in full["rows"]:
        if row["recommended_strategy"]:
            assert row["comparison_status"] in (STATUS_FULL, STATUS_PARTIAL)
            assert row["seller_loss_action"] in ("재고 이동", "정상 판매 유지", "할인")
            assert row["currency"] == "KRW"
    assert full["currency_basis"].startswith("Varo V2 workbook contract")


def test_pipeline_real_transport_cost_is_labelled_reference_not_actual():
    from services.seller_loss_engine import pipeline_decision_input

    recommendation = {"route_id": "R1", "product_id": "P1", "source_id": "S1", "target_id": "S2", "recommended_qty": 10,
                      "move_cost": 5000, "travel_time_min": 60, "varo_action": "재고 이동"}
    common = dict(inventory_rows={}, forecast_rows={}, product_rows={}, config={})
    direct = pipeline_decision_input(recommendation, candidate={"real_transport_applied": True, "proxy_vehicle_count": 0}, **common)
    proxy = pipeline_decision_input(recommendation, candidate={"real_transport_applied": True, "proxy_vehicle_count": 1}, **common)
    plain = pipeline_decision_input(recommendation, **common)
    assert direct.transfer_cost.provenance == "DERIVED_REAL" and "not an invoice" in direct.transfer_cost.note
    assert proxy.transfer_cost.provenance == "PROXY"
    assert plain.transfer_cost.provenance == "USER_INPUT" and plain.transfer_cost.currency == "KRW"
    assert plain.remaining_shelf_life_days.provenance == "MISSING"
    assert plain.source_holding_cost_per_unit_day.provenance == "MISSING"
