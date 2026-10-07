"""Seller Loss input requirement planner (services.seller_loss_input_requirements)."""
import copy
import inspect
import io
import random
from pathlib import Path

import pandas as pd
import pytest

from services import seller_loss_input_requirements as req
from services import seller_loss_inputs as sli
from services.analysis_pipeline import build_v2_state, run_analysis_pipeline
from services.seller_decision_validation import controlled_scenarios
from services.seller_loss_engine import (
    DISCOUNT_SALE, NORMAL_SALE, STATUS_FULL, STATUS_PARTIAL, STRATEGIES, TRANSFER, InputField, evaluate_seller_decision,
    known,
)
from services.seller_loss_requirement_validation import TEST_INPUT_LABEL, staged_flow
from tests.fixtures import sample_workbook
from tests.test_real_transport_enrichment import _write_real_data
from tests.test_seller_loss_inputs import (
    NETWORK_SAMPLE, SCENARIO_A_ROWS, TEMPLATE, W, _load, _suhyup_shaped_upload, base_input, table,
)

REPO = Path(__file__).resolve().parents[1]
MINIMAL_SAMPLE = REPO / "samples" / "seller_loss_inputs_MINIMAL_SAMPLE.csv"
NO_UPLIFT_ROWS = [r if r["scope"] != "PRODUCT" else {"scope": "PRODUCT", "product_id": "P1", "discount_rate": 0.3}
                  for r in SCENARIO_A_ROWS]


def plan(rows=None, base=None, **kwargs):
    """(engine decision with evidence, plan) through the same merge + evaluation as production."""
    return req.plan_input_requirements(base or base_input(), table(rows) if rows is not None else None, **kwargs)


def fields(items):
    return [item["field"] for item in items]


# ---------------------------------------------------------------- what is (and is not) requested


def test_no_inputs_requirement_planning():
    decision, result = plan()
    assert result["planning_status"] == "NEEDS_INPUT" and not result["stop_asking"]
    # real stock / demand / route are present: only the business values are requested, most strategies first
    assert fields(result["required_user_inputs"]) == ["remaining_shelf_life_days", "source_normal_price", "discount_rate",
                                                      "promotion_uplift", "target_normal_price"]
    assert [i["unblock_count"] for i in result["required_user_inputs"]] == [3, 3, 1, 1, 1]
    for item in result["required_user_inputs"]:
        assert item["tier"] == 1 and item["requirement_class"] == req.REQUIRED and item["request_kind"] == req.ENTER_MISSING
        assert item["seller_entry"]["sheet"] == "seller_loss_inputs" and item["seller_entry"]["suggested_scope"] == "PRODUCT_STORE"
        assert item["current"]["value"] is None and "value" not in item["seller_entry"]   # no number is ever suggested
    target_price = result["required_user_inputs"][-1]["seller_entry"]
    assert target_price["keys"] == {"product_id": "P1", "store_id": "S2"} and target_price["column"] == "normal_price"
    assert result["comparison_status_if_run_now"]["engine_comparison_status"] == "COMPARISON_UNAVAILABLE"
    assert "입력하면 세 전략" in result["message_ko"] and "source_normal_price" not in result["message_ko"]
    assert req.check_plan_consistency(result, decision) == []


def test_already_available_field_not_requested():
    decision, result = plan()
    requested = set(fields(result["required_user_inputs"]))
    available = {a["field"]: a for a in result["already_available_fields"]}
    for name in ("source_current_stock", "source_daily_demand", "target_current_stock", "target_daily_demand",
                 "transfer_cost", "transit_time_days"):
        assert name not in requested and available[name]["satisfied_by"] == "REAL_DATA"
    _, with_price = plan([SCENARIO_A_ROWS[0]])
    available = {a["field"]: a for a in with_price["already_available_fields"]}
    assert "source_normal_price" not in fields(with_price["required_user_inputs"])
    assert available["source_normal_price"]["satisfied_by"] == "SELLER_INPUT"
    assert available["source_normal_price"]["audit"][0]["outcome"] == "APPLIED"


def test_exact_real_field_not_requested_even_against_a_contradicting_seller_value():
    real_price = known(1000, "DIRECT_REAL", "pos price", currency="KRW", dataset=W)
    base = base_input(source_normal_price=real_price, target_normal_price=real_price)
    decision, result = plan([{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "normal_price": 1200}], base=base)
    assert not {"source_normal_price", "target_normal_price"} & set(fields(result["required_user_inputs"]))
    conflict = next(c for c in result["conflicting_fields"] if c["field"] == "source_normal_price")
    assert conflict["kind"] == "REAL_VS_SELLER_INPUT" and conflict["blocking"] is False and conflict["kept"] == 1000
    assert decision["input_provenance"]["source_normal_price"]["value"] == 1000


def test_proxy_replacement_suggested():
    base = base_input(source_daily_demand=known(3, "PROXY", "outbound proxy", dataset=W))
    decision, result = plan(SCENARIO_A_ROWS, base=base)
    assert fields(result["required_user_inputs"]) == ["source_daily_demand"]
    item = result["required_user_inputs"][0]
    assert item["request_kind"] == req.REPLACE_PROXY and item["current"]["provenance"] == "PROXY"
    assert item["unblocks_strategies"] == list(STRATEGIES) and "대리지표" in item["reason"]
    assert fields(result["proxy_replacement_inputs"]) == ["source_daily_demand"]
    auto = {a["field"]: a for a in result["auto_derived_fields"]}
    assert auto["source_daily_demand"]["status"] == "AUTO_PROXY" and not auto["source_daily_demand"]["usable_in_strict_comparison"]
    assert "대리지표" in result["message_ko"]


# ---------------------------------------------------------------- strategy-specific requirements and scopes


def test_transfer_only_requirements():
    base = base_input(target_daily_demand=InputField(source="no target forecast"), transfer_cost=InputField(source="no cost"))
    rows = [r for r in SCENARIO_A_ROWS if r.get("store_id") != "S2"]
    decision, result = plan(rows, base=base, comparison_scope=req.TRANSFER_VS_NORMAL)
    transfer = result["strategy_requirements"][TRANSFER]
    assert transfer["unmet_fields"] == ["target_daily_demand", "target_normal_price", "transfer_cost"]
    assert result["strategy_requirements"][NORMAL_SALE]["status"] == req.READY
    assert fields(result["required_user_inputs"]) == ["target_daily_demand", "target_normal_price", "transfer_cost"]
    assert all(i["unblocks_strategies"] == [TRANSFER] for i in result["required_user_inputs"])
    route = result["required_user_inputs"][-1]["seller_entry"]
    assert route["suggested_scope"] == "ROUTE" and route["keys"] == {"source_store_id": "S1", "target_store_id": "S2"}
    assert route["alternative_columns"] == ["transfer_cost_per_unit"] and route["expected_currency"] == "KRW"


def test_normal_only_requirements_and_scope_without_transfer():
    rows = [{**SCENARIO_A_ROWS[0], "remaining_shelf_life_days": None}, *SCENARIO_A_ROWS[1:]]
    base = base_input(target_daily_demand=InputField(source="no target forecast"))
    decision, result = plan(rows, base=base, comparison_scope=req.NORMAL_VS_DISCOUNT)
    assert result["strategy_requirements"][NORMAL_SALE]["unmet_fields"] == ["remaining_shelf_life_days"]
    assert fields(result["required_user_inputs"]) == ["remaining_shelf_life_days"]   # target demand is out of scope
    assert "target_daily_demand" in result["strategy_requirements"][TRANSFER]["unmet_fields"]
    assert result["out_of_scope_strategies"] == [TRANSFER]


def test_discount_only_requirements_ask_uplift_alone():
    decision, result = plan(NO_UPLIFT_ROWS, comparison_scope=req.NORMAL_VS_DISCOUNT)
    assert fields(result["required_user_inputs"]) == ["promotion_uplift"]          # discount_rate is already there
    item = result["required_user_inputs"][0]
    assert item["unblocks_strategies"] == [DISCOUNT_SALE] and "할인 후 판매량 변화" in item["reason"]
    assert result["planning_status"] == "RECOMMENDABLE_PARTIAL_SCOPE"
    assert decision["discount_price_preview"]["discounted_unit_price"] == 700         # markdown price needs no uplift


def test_all_three_requirements():
    rows = [r for r in NO_UPLIFT_ROWS if r.get("store_id") != "S2"]
    decision, result = plan(rows)
    assert fields(result["required_user_inputs"]) == ["promotion_uplift", "target_normal_price"]
    assert [i["unblocks_strategies"] for i in result["required_user_inputs"]] == [[DISCOUNT_SALE], [TRANSFER]]
    assert result["comparison_status_if_run_now"]["scope_status"] == req.SCOPE_UNAVAILABLE


def test_one_input_unblocks_multiple_strategies_and_ranks_first():
    _, result = plan()
    first = result["recommended_next_inputs"][0]
    assert first["unblocks_strategies"] == [TRANSFER, NORMAL_SALE, DISCOUNT_SALE] and first["priority"] == 1
    assert "전략 3개" in first["priority_basis"]
    # A strategy blocked by upload data (no target store) is never asked about; core fields then unblock two.
    decision, blocked = plan(base=base_input(target_store_id=None))
    assert blocked["strategy_requirements"][TRANSFER]["status"] == req.BLOCKED_BY_DATA
    assert "target_normal_price" not in fields(blocked["required_user_inputs"])
    assert blocked["recommended_next_inputs"][0]["unblocks_strategies"] == [NORMAL_SALE, DISCOUNT_SALE]
    assert any(b["field"] == "target_store_id" for b in blocked["blocked_fields"])


def test_recommended_next_inputs_order_is_deterministic_and_explained():
    rows = [r for r in NO_UPLIFT_ROWS if r.get("store_id") != "S2"]
    _, first = plan(rows)
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    _, second = plan(shuffled)
    assert first["recommended_next_inputs"] == second["recommended_next_inputs"]
    _, base_plan = plan()
    keys = [(i["tier"], -i["unblock_count"], req.ORDER_INDEX[i["field"]]) for i in base_plan["recommended_next_inputs"] if i["tier"] == 1]
    assert keys == sorted(keys)
    assert [i["priority"] for i in base_plan["recommended_next_inputs"]] == list(range(1, len(base_plan["recommended_next_inputs"]) + 1))
    assert all(i["priority_basis"] for i in base_plan["recommended_next_inputs"])


def test_optional_input_not_marked_required():
    # holding cost unknown at the source only: the ranking is robust, so it is optional (exact loss), never required
    rows = [r for r in SCENARIO_A_ROWS if r["scope"] != "GLOBAL"] + [
        {"scope": "GLOBAL", "disposal_cost_per_unit": 200, "salvage_value_per_unit": 0},
        {"scope": "STORE", "store_id": "S2", "holding_cost_per_unit_day": 5}]
    decision, result = plan(rows)
    assert decision["recommendation_readiness"] == sli.RECOMMENDABLE and "source_holding_cost_per_unit_day" in decision["unknown_inputs"]
    assert result["required_user_inputs"] == []
    assert fields(result["optional_user_inputs"]) == ["source_holding_cost_per_unit_day"]
    assert result["optional_user_inputs"][0]["request_kind"] == req.IMPROVE_PRECISION
    assert {"unit_cost", "target_disposal_cost_per_unit"} <= set(fields(result["not_needed_fields"]))
    # before the comparison exists the rates are only PENDING_EVALUATION (conditional), never required
    _, pending = plan()
    assert {i["request_kind"] for i in pending["optional_user_inputs"]} == {req.PENDING_EVALUATION}
    assert not {"source_holding_cost_per_unit_day", "unit_cost"} & set(fields(pending["required_user_inputs"]))


def test_stop_asking_when_recommendable():
    decision, result = plan(SCENARIO_A_ROWS)
    assert decision["recommendation_readiness"] == sli.RECOMMENDABLE and decision["comparison_status"] == STATUS_FULL
    assert result["stop_asking"] and result["planning_status"] == "READY_RECOMMENDABLE"
    assert result["required_user_inputs"] == [] and result["can_unblock_with_seller_inputs"] is False
    assert "추가 입력 없이 추천 가능" in result["message_ko"]
    # PARTIAL but RECOMMENDABLE in a scope that only asks for two strategies: stop asking too
    _, two = plan(NO_UPLIFT_ROWS, comparison_scope=req.TRANSFER_VS_NORMAL)
    assert two["stop_asking"] and two["required_user_inputs"] == [] and two["scope_complete"]


def test_robustness_inputs_are_the_decisive_unknowns_only():
    holding_unknown = [r for r in SCENARIO_A_ROWS if r["scope"] != "GLOBAL"] + [
        {"scope": "GLOBAL", "disposal_cost_per_unit": 200, "salvage_value_per_unit": 0}]
    decision, result = plan(holding_unknown)
    assert decision["recommendation_readiness"] == sli.NOT_ROBUST
    assert result["planning_status"] == "NEEDS_INPUT_FOR_ROBUST_RECOMMENDATION"
    assert fields(result["required_user_inputs"]) == ["source_holding_cost_per_unit_day", "target_holding_cost_per_unit_day"]
    for item in result["required_user_inputs"]:
        assert item["tier"] == 2 and item["request_kind"] == req.RESOLVE_UNKNOWN and item["unblocks_recommendation"]
        assert item["requirement_class"] == req.CONDITIONAL and item["ranking_pairs"]
    assert result["ranking_dependency"]["ambiguous_pairs"]
    # knowing every decisive unknown always yields a robust winner (here: the values of controlled scenario A)
    decision2, after = plan(holding_unknown + [{"scope": "GLOBAL", "holding_cost_per_unit_day": 5}])
    assert decision2["recommendation_readiness"] == sli.RECOMMENDABLE and after["stop_asking"]
    check = req.verify_unblock_step(result, decision, after, decision2, fields(result["required_user_inputs"]))
    assert check["ok"], check["problems"]


# ---------------------------------------------------------------- conflicts, currency, units


def test_conflict_field_still_blocks():
    decision, result = plan(SCENARIO_A_ROWS + [{"scope": "GLOBAL", "disposal_cost_per_unit": 300}])
    assert decision["recommendation_readiness"] == sli.NOT_ROBUST and not result["stop_asking"]
    item = next(i for i in result["required_user_inputs"] if i["field"] == "source_disposal_cost_per_unit")
    assert item["request_kind"] == req.RESOLVE_CONFLICT and item["tier"] == 2 and item["unblocks_recommendation"]
    assert next(c for c in result["conflicting_fields"] if c["field"] == "source_disposal_cost_per_unit")["blocking"] is True
    # a conflict on a structurally required field blocks every strategy and is a tier-1 request
    decision, price = plan(SCENARIO_A_ROWS + [{"scope": "PRODUCT_STORE", "product_id": "P1", "store_id": "S1", "normal_price": 1100}])
    first = price["required_user_inputs"][0]
    assert first["field"] == "source_normal_price" and first["request_kind"] == req.RESOLVE_CONFLICT
    assert first["unblock_count"] == 3 and "1000" in first["detail"] and "1100" in first["detail"]
    assert req.check_plan_consistency(price, decision) == []


def test_currency_mismatch_requirement():
    usd_rates = SCENARIO_A_ROWS[:2] + [{**SCENARIO_A_ROWS[2], "currency": "USD"}] + SCENARIO_A_ROWS[3:]
    decision, result = plan(usd_rates)
    assert "CURRENCY_MISMATCH" in decision["reason_codes"]
    assert {i["request_kind"] for i in result["required_user_inputs"]} == {req.CORRECT_CURRENCY}
    assert set(fields(result["required_user_inputs"])) == {"source_holding_cost_per_unit_day", "source_disposal_cost_per_unit",
                                                           "salvage_value_per_unit"}
    blocker = result["strategy_requirements"][NORMAL_SALE]["blockers"][0]
    assert blocker["code"] == "CURRENCY_MISMATCH" and blocker["expected"] == "KRW" and blocker["resolution"] == req.SELLER
    decision2, fixed = plan(SCENARIO_A_ROWS)                                     # re-entered in KRW
    assert decision2["comparison_status"] == STATUS_FULL
    assert req.verify_unblock_step(result, decision, fixed, decision2, fields(result["required_user_inputs"]))["ok"]
    # everything in USD against the real KRW route cost: only TRANSFER is blocked, and the fix unblocks only TRANSFER
    _, usd_all = plan([{**r, "currency": "USD"} for r in SCENARIO_A_ROWS])
    assert all(i["unblocks_strategies"] == [TRANSFER] for i in usd_all["required_user_inputs"])
    # data-vs-data currency mixes cannot be fixed by the seller: nothing is requested
    k = next(s for s in controlled_scenarios() if s["id"] == "K")
    decision_k, plan_k = req.plan_input_requirements(k["input"], None)
    assert plan_k["planning_status"] == "BLOCKED_BY_DATA" and plan_k["required_user_inputs"] == []


def test_unit_mismatch_requirement():
    kg = [{**SCENARIO_A_ROWS[0], "price_unit": "KRW/KG"}, *SCENARIO_A_ROWS[1:]]
    decision, result = plan(kg, base=base_input(quantity_unit="EA"))
    assert "UNIT_MISMATCH" in decision["reason_codes"]
    item = result["required_user_inputs"][0]
    assert item["field"] == "source_normal_price" and item["request_kind"] == req.CORRECT_UNIT and item["unblock_count"] == 3
    assert result["strategy_requirements"][NORMAL_SALE]["blockers"][0]["expected"] == "EA"
    # a declared per-unit value with an undeclared inventory unit: the planner asks for the unit declaration
    declared = [{**SCENARIO_A_ROWS[0], "price_unit": "KRW/EA"}, *SCENARIO_A_ROWS[1:]]
    decision, unit_plan = plan(declared)
    item = unit_plan["required_user_inputs"][0]
    assert item["field"] == "quantity_unit" and item["request_kind"] == req.DECLARE_UNIT and item["requirement_class"] == req.CONDITIONAL
    entry = item["seller_entry"]
    row = {"scope": entry["suggested_scope"], **entry["keys"], "quantity_unit": "EA"}
    decision2, after = plan(declared + [row])
    assert decision2["comparison_status"] == STATUS_FULL and decision2["quantity_unit"] == "EA"
    assert req.verify_unblock_step(unit_plan, decision, after, decision2, ["quantity_unit"])["ok"]


# ---------------------------------------------------------------- inputs resolve requests


def test_user_input_resolves_missing():
    decision, before = plan(SCENARIO_A_ROWS[2:])
    assert "source_normal_price" in fields(before["required_user_inputs"])
    decision2, after = plan(SCENARIO_A_ROWS)
    check = req.verify_unblock_step(before, decision, after, decision2, fields(before["required_user_inputs"]))
    assert check["ok"], check["problems"]
    assert "PRICE_MISSING" in check["removed_engine_reasons"] and after["stop_asking"]


def test_user_input_resolves_proxy_and_is_audited():
    base = base_input(source_daily_demand=known(3, "PROXY", "outbound proxy", dataset=W))
    decision, before = plan(SCENARIO_A_ROWS, base=base)
    entry = before["required_user_inputs"][0]["seller_entry"]
    row = {"scope": entry["suggested_scope"], **entry["keys"], entry["column"]: 3}
    decision2, after = plan(SCENARIO_A_ROWS + [row], base=base)
    check = req.verify_unblock_step(before, decision, after, decision2, ["source_daily_demand"])
    assert check["ok"] and "PROXY_REJECTED:source_daily_demand" in check["removed_engine_reasons"]
    audit = next(a for a in decision2["seller_input_audit"] if a["field"] == "source_daily_demand")
    assert audit["outcome"] == "APPLIED_OVER_PROXY" and "source_daily_demand" in decision2["used_input_fields"]
    available = next(a for a in after["already_available_fields"] if a["field"] == "source_daily_demand")
    assert available["satisfied_by"] == "SELLER_INPUT" and available["audit"][0]["outcome"] == "APPLIED_OVER_PROXY"


def test_irrelevant_user_input_changes_nothing():
    decision, baseline = plan()
    irrelevant = [{"scope": "PRODUCT_STORE", "product_id": "P9", "store_id": "S1", "normal_price": 1000},
                  {"scope": "STORE", "store_id": "S7", "remaining_shelf_life_days": 5},
                  {"scope": "PRODUCT", "product_id": "P1", "unit_cost": 600},
                  {"scope": "PRODUCT", "product_id": "P1", "input_type": "SCENARIO", "promotion_uplift": 0.5}]
    decision2, other = plan(irrelevant)
    assert other["required_user_inputs"] == baseline["required_user_inputs"]
    assert other["recommended_next_inputs"] == baseline["recommended_next_inputs"]
    assert {k: decision2[k] for k in ("comparison_status", "reason_codes", "unavailable_strategies")} == \
        {k: decision[k] for k in ("comparison_status", "reason_codes", "unavailable_strategies")}


# ---------------------------------------------------------------- consistency with the engine and evidence layer


@pytest.mark.parametrize("scope", sorted(req.COMPARISON_SCOPES))
def test_planner_and_engine_readiness_consistent(scope):
    variants = [(s["input"], None) for s in controlled_scenarios()]
    variants += [(base_input(), table(rows)) for rows in (
        SCENARIO_A_ROWS, NO_UPLIFT_ROWS, SCENARIO_A_ROWS[2:], [], SCENARIO_A_ROWS + [{"scope": "GLOBAL", "disposal_cost_per_unit": 300}],
        [{**r, "currency": "USD"} for r in SCENARIO_A_ROWS], [{**SCENARIO_A_ROWS[0], "price_unit": "KRW/EA"}, *SCENARIO_A_ROWS[1:]])]
    variants.append((base_input(source_daily_demand=known(3, "PROXY", "proxy", dataset=W), target_store_id=None), table(SCENARIO_A_ROWS)))
    for inp, tbl in variants:
        decision, result = req.plan_input_requirements(inp, tbl, comparison_scope=scope)
        assert req.check_plan_consistency(result, decision) == [], (inp.decision_id, scope)
        for strategy in STRATEGIES:   # the planner's per-strategy gaps are the evidence layer's, field for field
            assert sorted(set(result["strategy_requirements"][strategy]["unmet_fields"]) - {"decision_qty"}) == \
                sorted(decision["missing_required_by_strategy"][strategy])
        assert set(result["ready_strategies"]) == set(decision["comparable_strategies"]) & set(req.COMPARISON_SCOPES[scope])


def test_engine_trace_matches_the_planner_requirements():
    trace = req.trace_engine_requirements()
    for strategy in STRATEGIES:
        traced_required = {n for n, f in trace["fields"].items() if f["per_strategy"][strategy]["class"] == req.REQUIRED}
        assert traced_required == set(req.strategy_required_fields(strategy))
        conditional = {n for n, f in trace["fields"].items() if f["per_strategy"][strategy]["class"] == req.CONDITIONAL}
        assert conditional - {"transfer_cost_qty"} <= set(sli.STRATEGY_OPTIONAL[strategy])
    per = {n: f["per_strategy"] for n, f in trace["fields"].items()}
    assert {per["unit_cost"][s]["class"] for s in STRATEGIES} == {req.NOT_USED}          # sunk cost
    assert per["target_disposal_cost_per_unit"][TRANSFER]["class"] == req.NOT_USED      # q <= target horizon need
    assert per["transfer_cost_qty"][TRANSFER]["condition"] == "TRANSFER_COST_BASIS_QUANTITY_SPECIFIC"
    assert per["salvage_value_per_unit"][NORMAL_SALE]["class"] == req.CONDITIONAL
    assert all(per[c][TRANSFER]["class"] == req.OPTIONAL for c in ("source_surplus_cap", "target_need_cap", "route_capacity_qty"))
    assert trace["fields"]["decision_qty"]["field_class"] == req.DERIVED
    for name, info in per.items():   # a proxy is treated exactly like a missing value
        for strategy in STRATEGIES:
            if info[strategy]["class"] == req.REQUIRED and name != "target_store_id":
                assert info[strategy]["proxy_unavailable_in"] == info[strategy]["unavailable_in"]
    assert set(req.ENGINE_CHECK_ORDER) == {*req.INPUT_FIELD_NAMES, "target_store_id", "quantity_unit"}


def test_partial_remains_non_promotable():
    decision, result = plan(NO_UPLIFT_ROWS)
    snapshot = copy.deepcopy(decision)
    again = req.plan_for_decision(base_input(), table(NO_UPLIFT_ROWS), decision)
    assert decision == snapshot and again == result            # planning never mutates the evaluated decision
    assert decision["comparison_status"] == STATUS_PARTIAL
    assert result["planning_status"] == "RECOMMENDABLE_PARTIAL_SCOPE"
    assert not result["scope_complete"] and not result["scope_recommendable"] and not result["stop_asking"]
    assert result["production_action_applied"] is False and result["legacy_action_changed"] is False
    assert result["production_promotion_candidate"] == decision["production_promotion_candidate"]   # copied, never raised
    assert "현재 2개 전략 비교 가능" in result["comparison_status_if_run_now"]["message"]
    assert fields(result["required_user_inputs"]) == ["promotion_uplift"]


def test_scenario_values_are_separated():
    rows = SCENARIO_A_ROWS[:3] + [{"scope": "PRODUCT", "product_id": "P1", "discount_rate": 0.3},
                                  {"scope": "PRODUCT", "product_id": "P1", "input_type": "SCENARIO", "promotion_uplift": 0.5}]
    _, actual = plan(rows)
    assert fields(actual["required_user_inputs"]) == ["promotion_uplift"]               # a what-if value is not an answer
    assert actual["required_user_inputs"][0]["request_kind"] == req.ENTER_MISSING
    decision, scenario = plan(rows, decision_mode=sli.SCENARIO)
    assert scenario["planning_status"] == "SCENARIO_ONLY" and not scenario["stop_asking"]
    uplift = next(a for a in scenario["already_available_fields"] if a["field"] == "promotion_uplift")
    assert uplift["satisfied_by"] == "SCENARIO" and decision["recommendation_readiness"] == sli.SCENARIO_ONLY
    # config placeholders used in ACTUAL_OPERATION must be replaced by actual values before a recommendation
    a = controlled_scenarios()[0]["input"]
    decision, config_plan = req.plan_input_requirements(a, None)
    assert {i["request_kind"] for i in config_plan["required_user_inputs"]} == {req.REPLACE_SCENARIO_VALUE}


def test_sample_templates_are_never_used_as_defaults():
    source = inspect.getsource(req)
    assert "TEMPLATE_SAMPLE" not in source and "MINIMAL_SAMPLE.csv" not in source and "read_csv" not in source
    decision, without = plan()
    assert not any(str(a["source"]).startswith("seller_loss_inputs") for a in without["already_available_fields"])
    assert all(d["origin"] in ("DATA", "NONE") for d in decision["input_sources"].values())   # nothing filled without an upload
    # SAMPLE-keyed template rows match no real decision (its GLOBAL example rows are documented "replace or delete")
    template = pd.read_csv(TEMPLATE, encoding="utf-8-sig", dtype=str)
    keyed = template[template["scope"] != "GLOBAL"]
    parsed = sli.parse_seller_loss_inputs(keyed, upload_currency="KRW", upload_currency_basis="t")
    _, with_template = req.plan_input_requirements(base_input(), parsed)
    assert with_template["required_user_inputs"] == without["required_user_inputs"]
    # the minimal sample: SAMPLE ids and notes only (no GLOBAL row), valid, and it changes nothing for a real decision
    frame = pd.read_csv(MINIMAL_SAMPLE, encoding="utf-8-sig", dtype=str)
    assert list(frame.columns) == list(req.MINIMAL_SAMPLE_COLUMNS) and frame["note"].str.startswith("SAMPLE").all()
    assert "GLOBAL" not in set(frame["scope"]) and frame["product_id"].str.startswith("SAMPLE_").all()
    assert all(str(v).startswith("SAMPLE_") for v in frame["store_id"].dropna())
    minimal = sli.parse_seller_loss_inputs(frame, upload_currency="KRW", upload_currency_basis="t")
    assert minimal.validation["status"] == "VALID" and minimal.validation["warnings"] == []
    _, with_minimal = req.plan_input_requirements(base_input(), minimal)
    assert with_minimal["required_user_inputs"] == without["required_user_inputs"]
    generated = pd.read_csv(io.StringIO(req.minimal_sample_frame().to_csv(index=False)))
    pd.testing.assert_frame_equal(pd.read_csv(MINIMAL_SAMPLE, encoding="utf-8-sig"), generated)
    proxy = lambda v: known(v, "PROXY", "outbound proxy", dataset=W)  # noqa: E731
    shaped = base_input(product_id="SAMPLE_PRODUCT_1", source_store_id="SAMPLE_STORE_A", target_store_id="SAMPLE_STORE_B",
                        source_daily_demand=proxy(3), target_daily_demand=proxy(4))
    _, asked = req.plan_input_requirements(shaped, None)
    assert {i["seller_entry"]["column"] for i in asked["required_user_inputs"]} <= set(req.MINIMAL_SAMPLE_COLUMNS)
    decision, filled = req.plan_input_requirements(shaped, minimal)
    assert not [i for i in filled["required_user_inputs"] if i["tier"] == 1] and decision["comparison_status"] == STATUS_FULL


def test_labels_reasons_and_bundles_are_seller_facing():
    _, result = plan(base=base_input(source_daily_demand=known(3, "PROXY", "proxy", dataset=W)))
    for item in result["recommended_next_inputs"]:
        assert item["label"] == req.FIELD_LABELS[item["field"]] and item["reason"]
        assert item["field"] not in item["reason"] and "_" not in item["label"]
    assert "_" not in result["message_ko"].split(":", 1)[1]
    bundles = {b["bundle"]: b for b in result["input_bundles"]}
    assert set(bundles["SALES"]["required_fields"]) == {"source_daily_demand", "source_normal_price", "target_normal_price"}
    assert bundles["DISCOUNT"]["required_fields"] == ["discount_rate", "promotion_uplift"]
    assert all("value" not in b for b in result["input_bundles"])
    assert "anthropic" not in inspect.getsource(req) and "openai" not in inspect.getsource(req)
    assert req.FIELD_LABELS["remaining_shelf_life_days"] == "남은 판매 가능 기간(일)"
    assert req.FIELD_LABELS["promotion_uplift"] == "할인 시 예상 판매 증가율"


def test_comparison_scope_resolution_and_contract():
    assert req.resolve_comparison_scope(None, {}, {})[0] == req.ALL_THREE
    assert req.resolve_comparison_scope(None, {req.COMPARISON_SCOPE_KEY: "transfer_vs_normal"}, {})[0] == req.TRANSFER_VS_NORMAL
    assert req.resolve_comparison_scope(None, {}, {req.COMPARISON_SCOPE_KEY: "NORMAL_VS_DISCOUNT"})[0] == req.NORMAL_VS_DISCOUNT
    scope, info = req.resolve_comparison_scope("EVERYTHING", {}, {})
    assert scope == req.ALL_THREE and info["error"]["code"] == "INVALID_COMPARISON_SCOPE"
    first, second = req.requirement_contract_document(), req.requirement_contract_document()
    assert first == second and len(first["contract_signature"]) == 64
    assert first["default_comparison_scope"] == req.ALL_THREE


# ---------------------------------------------------------------- pipeline


def test_pipeline_attaches_requirements_without_a_seller_sheet():
    data = sample_workbook()
    state = build_v2_state(copy.deepcopy(data), detail_level="core")
    analysis = state["pipeline_result"]["seller_loss_analysis"]
    assert analysis["seller_loss_inputs"]["status"] == "ABSENT" and analysis["comparison_scope"] == req.ALL_THREE
    summary = analysis["input_requirements_summary"]
    assert summary["decision_count"] == len(analysis["decisions"]) and summary["production_action_applied"] is False
    for decision, row in zip(analysis["decisions"], analysis["rows"]):
        assert decision["input_requirements"]["planner_version"] == req.PLANNER_VERSION
        assert req.check_plan_consistency(decision["input_requirements"], decision) == []
        assert row["input_planning_status"] == decision["input_requirements"]["planning_status"]
    scoped = build_v2_state({**copy.deepcopy(data), req.COMPARISON_SCOPE_KEY: req.TRANSFER_VS_NORMAL}, detail_level="core")
    scoped_analysis = scoped["pipeline_result"]["seller_loss_analysis"]
    assert scoped_analysis["comparison_scope"] == req.TRANSFER_VS_NORMAL
    strip = lambda ds: [{k: v for k, v in d.items() if k != "input_requirements"} for d in ds]  # noqa: E731
    assert strip(scoped_analysis["decisions"]) == strip(analysis["decisions"])     # the scope never changes the engine
    assert scoped["recommendations"] == state["recommendations"]


def test_network_sample_plans_are_consistent():
    analysis = build_v2_state(_load(NETWORK_SAMPLE), detail_level="core")["pipeline_result"]["seller_loss_analysis"]
    for decision in analysis["decisions"]:
        assert req.check_plan_consistency(decision["input_requirements"], decision) == []


def test_summary_deduplicates_seller_entries():
    _, first = plan(base=base_input(decision_id="R1"))
    _, second = plan(base=base_input(decision_id="R2"))     # same product and stores: same seller entries
    summary = req.summarize_plans([first, second])
    assert summary["distinct_required_entries"] == len(first["required_user_inputs"])
    assert all(len(e["decisions"]) == 2 for e in summary["required_entries"])
    assert summary["decisions_needing_input"] == 2 and "판매자 입력이 필요" in summary["message_ko"]


def test_suhyup_staged_e2e(tmp_path, monkeypatch):
    """Suhyup-shaped real rows: A no input -> B first request -> C structural minimum -> D robustness -> stop."""
    _write_real_data(tmp_path)
    monkeypatch.setenv("VARO_REAL_DATA_ROOT", str(tmp_path))
    baseline = run_analysis_pipeline(_suhyup_shaped_upload())

    def runner(scope):
        def run(frame):
            upload = {**_suhyup_shaped_upload(), req.COMPARISON_SCOPE_KEY: scope}
            if frame is not None:
                upload[sli.SHEET_KEY] = frame
            result = run_analysis_pipeline(upload)
            assert result.recommendations == baseline.recommendations      # legacy outputs untouched
            return {"pipeline_result": {"seller_loss_analysis": result.seller_loss_analysis}}
        return run

    for scope in (req.TRANSFER_VS_NORMAL, req.ALL_THREE):
        flow = staged_flow(runner(scope), scope)
        assert flow["a"]["comparison_scope"] == scope
        assert flow["problems"] == [] and flow["audit_problems"] == []
        a_plan = flow["a"]["decisions"][0]["input_requirements"]
        expected = ["source_daily_demand", "remaining_shelf_life_days", "source_normal_price"]
        expected += ["discount_rate", "promotion_uplift"] if scope == req.ALL_THREE else []
        expected += ["target_daily_demand", "target_normal_price"]
        assert fields(a_plan["required_user_inputs"]) == expected
        assert {"source_current_stock", "target_current_stock", "transfer_cost", "transit_time_days"} <= {
            a["field"] for a in a_plan["already_available_fields"]}
        b_row = next(r for r in flow["rows"] if r["stage"] == "B_FIRST_RECOMMENDED_INPUT")
        assert b_row["added_inputs"] == "source_daily_demand" and b_row["removed_engine_reasons"] == "PROXY_REJECTED:source_daily_demand"
        c = flow["c"]["decisions"][0]
        assert all(s in c["comparable_strategies"] for s in req.COMPARISON_SCOPES[scope])
        final = flow["final_step"]["decisions"][0]
        assert final["input_requirements"]["stop_asking"] and final["recommendation_readiness"] == sli.RECOMMENDABLE
        assert final["legacy_action_changed"] is False and final["input_requirements"]["production_action_applied"] is False
        assert all(row["note"] == TEST_INPUT_LABEL for row in flow["test_rows"]["STEP"])
