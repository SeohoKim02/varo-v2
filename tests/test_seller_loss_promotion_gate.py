"""Gate qualification tests. All fixtures are artificial; no production evidence is claimed."""
import copy
from dataclasses import replace

import pandas as pd
import pytest

from services import seller_loss_promotion_gate as gate, seller_loss_inputs as sli, seller_loss_input_requirements as req
from services.seller_decision_validation import controlled_scenarios, scenario_input
from services.seller_loss_engine import InputField, known, TRANSFER, NORMAL_SALE, DISCOUNT_SALE


def eligible_input():
    """Emulate a complete direct-real contract to exercise the gate, not real data."""
    artificial = scenario_input()
    fields = {}
    for name, field in artificial.input_fields().items():
        fields[name] = replace(field, provenance="DIRECT_REAL" if field.present else "MISSING",
                               source="verified business record", dataset="uploaded_workbook", note="")
    return replace(artificial, **fields, decision_id="R001", product_id="P001", legacy_action="보류")


def decision(inp=None):
    return sli.evaluate_with_seller_inputs(inp or eligible_input(), None, decision_date="2026-07-31")


def evaluate(d=None, scope=req.ALL_THREE, **kwargs):
    return gate.evaluate_promotion_gate(d or decision(), comparison_scope=scope,
        explicit_scope=scope != req.ALL_THREE, context={"decision_date": "2026-07-31", **kwargs})


def user_table(rows):
    return sli.parse_seller_loss_inputs(rows, upload_currency="KRW", upload_currency_basis="seller upload contract")


def test_real_only_eligible_logic():
    assert evaluate()["promotion_status"] == gate.PROMOTION_ELIGIBLE
    assert not evaluate()["production_action_applied"]


def test_real_plus_user_input_eligible_logic():
    inp = replace(eligible_input(), source_normal_price=InputField())
    table = user_table([{"scope": "PRODUCT", "product_id": "P001", "normal_price": 1000, "effective_date": "2026-07-31"}])
    d = sli.evaluate_with_seller_inputs(inp, table, decision_date="2026-07-31")
    assert d["evidence_level"] == sli.REAL_PLUS_USER_INPUT
    assert evaluate(d)["promotion_status"] == gate.PROMOTION_ELIGIBLE


def test_profile_eligible_with_valid_window():
    from services.seller_business_profile import parse_profile
    inp = replace(eligible_input(), source_normal_price=InputField(), target_normal_price=InputField())
    table = parse_profile([{"scope": "PRODUCT", "product_id": "P001", "normal_price": 1000, "currency": "KRW",
        "effective_date": "2026-07-01", "effective_until": "2026-07-31", "profile_source": "seller approved price list"}])
    d = sli.evaluate_with_seller_inputs(inp, table, decision_date="2026-07-31")
    assert evaluate(d)["promotion_status"] == gate.PROMOTION_ELIGIBLE
    bad = gate.evaluate_promotion_gate(d, context={"decision_date": "2026-08-01"})
    assert bad["promotion_status"] == gate.BLOCKED
    assert any(c.startswith("USER_INPUT_EFFECTIVE_DATE_INVALID") for c in bad["promotion_blockers"])


def test_user_only_remains_shadow():
    inp = eligible_input()
    inp = replace(inp, **{n: replace(f, provenance="USER_INPUT") for n,f in inp.input_fields().items() if f.present})
    assert evaluate(decision(inp))["promotion_status"] == gate.SHADOW_ONLY


def test_scenario_never_promotes():
    d = sli.evaluate_with_seller_inputs(eligible_input(), None, decision_mode=sli.SCENARIO)
    assert evaluate(d)["promotion_status"] == gate.SCENARIO_ONLY


def test_insufficient_blocks():
    d = decision(replace(eligible_input(), source_normal_price=InputField()))
    assert evaluate(d)["promotion_status"] == gate.INSUFFICIENT_DATA


@pytest.mark.parametrize("context", [{"is_test": True}, {"is_sample": True}, {"source_file": "samples/prices.csv"},
    {"label": "TEST_USER_INPUT"}, {"data_kind": "synthetic fixture"}])
def test_execution_fixture_and_sample_labels_block(context):
    r = evaluate(**context)
    assert r["promotion_status"] == gate.BLOCKED and "TEST_OR_SAMPLE_INPUT" in r["promotion_blockers"]


@pytest.mark.parametrize("note", ["SAMPLE", "TEST USER INPUT", "TEST_USER_INPUT", "synthetic fixture"])
def test_cell_labels_survive_merge_and_block(note):
    inp = replace(eligible_input(), source_normal_price=InputField())
    table = user_table([{"scope": "PRODUCT", "product_id": "P001", "normal_price": 1000, "note": note}])
    d = sli.evaluate_with_seller_inputs(inp, table, decision_date="2026-07-31")
    assert d["recommendation_readiness"] == sli.RECOMMENDABLE
    assert evaluate(d)["promotion_status"] == gate.BLOCKED
    assert d["seller_input_audit"][0]["data_labels"] == [note]


@pytest.mark.parametrize("scope,missing_field", [(req.TRANSFER_VS_NORMAL, "promotion_uplift"),
    (req.NORMAL_VS_DISCOUNT, "target_daily_demand"), (req.TRANSFER_VS_DISCOUNT, "promotion_uplift")])
def test_scope_complete_pair_vs_partial_all_three(scope, missing_field):
    inp = replace(eligible_input(), **{missing_field: InputField()})
    if scope == req.TRANSFER_VS_DISCOUNT:
        # NORMAL always shares core fields, so emulate a result with exactly the requested pair.
        d = decision()
        d["comparable_strategies"] = [TRANSFER, DISCOUNT_SALE]
        d["comparison_status"] = "PARTIAL"
        d["strategies"][NORMAL_SALE]["available"] = False
    else:
        d = decision(inp)
        if scope == req.NORMAL_VS_DISCOUNT:
            inp = replace(inp, source_daily_demand=known(6,"DIRECT_REAL","daily sales",dataset="uploaded_workbook"))
            d = decision(inp)
    pair = evaluate(d, scope=scope)
    assert pair["promotion_status"] == gate.PROMOTION_ELIGIBLE
    assert pair["promotion_comparison_status"] == "FULL_REQUESTED_SCOPE"
    all_three = evaluate(d)
    assert all_three["promotion_status"] == gate.INSUFFICIENT_DATA
    assert "PARTIAL_REQUESTED_SCOPE" in all_three["promotion_blockers"]


def test_pair_requires_explicit_scope():
    r = gate.evaluate_promotion_gate(decision(), comparison_scope=req.TRANSFER_VS_NORMAL)
    assert "TWO_STRATEGY_SCOPE_NOT_EXPLICIT" in r["promotion_blockers"]


def test_winner_outside_scope_not_replaced():
    d = decision()
    r = evaluate(d, scope=req.NORMAL_VS_DISCOUNT)
    assert r["promotion_status"] == gate.BLOCKED and "WINNER_OUTSIDE_REQUESTED_SCOPE" in r["promotion_blockers"]
    assert d["recommended_strategy"] == TRANSFER


def test_conflict_even_real_value_kept_blocks():
    d = sli.evaluate_with_seller_inputs(eligible_input(), user_table([{"scope": "PRODUCT", "product_id": "P001", "normal_price": 999}]))
    assert d["recommendation_readiness"] == sli.RECOMMENDABLE
    assert evaluate(d)["promotion_status"] == gate.BLOCKED


@pytest.mark.parametrize("reason", ["CURRENCY_MISMATCH", "UNIT_MISMATCH", "SELLER_UNIT_UNVERIFIABLE:normal_price"])
def test_mismatch_blocks(reason):
    d = decision()
    d["seller_input_issues"] = [reason]
    assert evaluate(d)["promotion_status"] == gate.BLOCKED


def test_exact_tie_and_reported_numerical_zero_remain_shadow():
    d = decision()
    d["tie"] = True
    assert evaluate(d)["promotion_status"] == gate.SHADOW_ONLY
    d["tie"] = False
    d["loss_difference_vs_second_best"] = 0
    assert "TIE_OR_NUMERICAL_TOLERANCE" in evaluate(d)["promotion_blockers"]


def test_unknown_optional_robust_engine_is_shadow():
    d = decision(replace(eligible_input(), source_holding_cost_per_unit_day=InputField()))
    assert d["recommendation_readiness"] == sli.RECOMMENDABLE
    assert evaluate(d)["promotion_status"] == gate.SHADOW_ONLY
    assert any(c.startswith("UNKNOWN_MONETARY_COMPONENT") for c in evaluate(d)["promotion_blockers"])


@pytest.mark.parametrize("legacy,expected", [("재고 이동","SAME"), ("할인","DIFFERENT"),
    ("정상 판매 유지","DIFFERENT"), ("긴급할인","UNMAPPABLE"), ("1+1","UNMAPPABLE"),
    ("폐기","UNMAPPABLE"), ("보류","UNMAPPABLE"), ("유지","UNMAPPABLE")])
def test_action_mapping_and_agreement(legacy,expected):
    d = decision(replace(eligible_input(), legacy_action=legacy))
    shadow = gate.shadow_decision(d, context={"decision_date":"2026-07-31"})
    assert shadow["legacy_vs_seller_agreement"] == expected
    assert shadow["final_action_candidate"]["current_production_action"] == legacy
    if expected == "DIFFERENT": assert "different_objective" in shadow["disagreement_reason_codes"]
    if expected == "UNMAPPABLE": assert "action_mapping_unavailable" in shadow["disagreement_reason_codes"]


def test_deterministic_policy_gate_summary_and_no_mutation():
    d = decision()
    before = copy.deepcopy(d)
    rows = [gate.shadow_decision(d, context={"decision_date":"2026-07-31"}),
            gate.shadow_decision(d, context={"is_test":True})]
    assert gate.policy_document() == gate.policy_document()
    assert evaluate(d) == evaluate(d) and d == before
    summary = gate.summarize_shadow(rows)
    assert summary["total_decisions"] == 2 and summary["promotion_eligible"] == 1 and summary["blocked"] == 1
    assert not summary["production_action_applied"]


def test_controlled_scenarios_never_promote():
    for scenario in controlled_scenarios():
        d = sli.evaluate_with_seller_inputs(scenario["input"], None)
        assert not evaluate(d, is_test=True)["promotion_candidate"]


def test_pipeline_shadow_does_not_change_final_actions():
    from services.analysis_pipeline import build_v2_state
    from tests.fixtures import sample_workbook
    data = sample_workbook()
    first = build_v2_state(data, detail_level="core")
    marked = build_v2_state({**data, "is_test":True}, detail_level="core")
    assert first["recommendations"] == marked["recommendations"]
    assert first["pipeline_result"]["summary"] == marked["pipeline_result"]["summary"]
    analysis = marked["pipeline_result"]["seller_loss_analysis"]
    assert analysis["promotion_summary"]["promotion_eligible"] == 0
    assert all(r["promotion_status"] == gate.BLOCKED for r in analysis["shadow_decisions"])


def test_suhyup_proxy_input_never_falsely_promotes():
    inp = replace(eligible_input(), source_daily_demand=known(3,"PROXY","outbound",dataset="seller_upload"))
    assert not evaluate(decision(inp))["promotion_candidate"]


def test_empty_component_contract_fails_closed():
    d = decision()
    d["strategies"][TRANSFER]["components"] = {}
    assert evaluate(d)["promotion_status"] == gate.SHADOW_ONLY


def test_user_input_requires_scope_date_and_source_audit():
    table = user_table([{"scope":"PRODUCT", "product_id":"P001", "normal_price":1000, "effective_date":"2026-07-31"}])
    d = sli.evaluate_with_seller_inputs(replace(eligible_input(),source_normal_price=InputField()),table,decision_date="2026-07-31")
    bad = copy.deepcopy(d)
    bad["seller_input_audit"][0]["keys"]["product_id"] = "OTHER"
    assert "USER_INPUT_SCOPE_OR_DATE_MISMATCH:source_normal_price" in evaluate(bad)["promotion_blockers"]
    bad = copy.deepcopy(d)
    bad["seller_input_audit"] = []
    assert "USER_INPUT_AUDIT_MISSING:source_normal_price" in evaluate(bad)["promotion_blockers"]
    no_date = gate.evaluate_promotion_gate(d)
    assert no_date["promotion_status"] == gate.BLOCKED


@pytest.mark.parametrize("marker", [{"note":"SAMPLE"}, {"is_sample":True},
    {"is_test":True}, {"source_file":"samples/currency.csv"}, {"profile_source":"TEST_USER_INPUT"}])
def test_global_sample_currency_provenance_not_lost(marker):
    from services.seller_business_profile import parse_profile
    table = parse_profile([{"scope":"GLOBAL","currency":"KRW", **marker},
        {"scope":"PRODUCT","product_id":"P001","normal_price":1000,
         "effective_date":"2026-07-01","effective_until":"2026-07-31"}])
    d = sli.evaluate_with_seller_inputs(replace(eligible_input(),source_normal_price=InputField()),table,decision_date="2026-07-31")
    assert evaluate(d)["promotion_status"] == gate.BLOCKED


def test_boolean_sample_cell_marker_not_lost():
    table = user_table([{"scope":"PRODUCT","product_id":"P001","normal_price":1000,"is_sample":True}])
    d = sli.evaluate_with_seller_inputs(replace(eligible_input(),source_normal_price=InputField()),table)
    assert evaluate(d)["promotion_status"] == gate.BLOCKED


def test_nonfinite_loss_and_invalid_scope_cannot_qualify():
    d = decision()
    d["strategies"][TRANSFER]["expected_loss"] = float("inf")
    assert not evaluate(d)["promotion_candidate"]
    assert gate.evaluate_promotion_gate(decision(),comparison_scope="INVALID")["promotion_status"] == gate.BLOCKED


def test_profile_incomplete_audit_window_blocks_without_exception():
    from services.seller_business_profile import parse_profile
    table = parse_profile([{"scope":"PRODUCT", "product_id":"P001", "normal_price":1000,
        "currency":"KRW", "effective_date":"2026-07-01", "effective_until":"2026-07-31"}])
    d = sli.evaluate_with_seller_inputs(replace(eligible_input(),source_normal_price=InputField()),
        table, decision_date="2026-07-31")
    assert evaluate(d)["promotion_status"] == gate.PROMOTION_ELIGIBLE
    d["seller_input_audit"][0]["effective_until"] = None
    result = evaluate(d)
    assert result["promotion_status"] == gate.BLOCKED
    assert "USER_INPUT_EFFECTIVE_DATE_INVALID:source_normal_price" in result["promotion_blockers"]


def test_inconsistent_comparison_contract_blocks():
    for status in ("PARTIAL", "COMPARISON_UNAVAILABLE", "UNKNOWN"):
        d = decision()
        d["comparison_status"] = status
        assert "COMPARISON_CONTRACT_INCONSISTENT" in evaluate(d)["promotion_blockers"]
    d = decision()
    d["comparable_strategies"].append(TRANSFER)
    assert evaluate(d)["promotion_status"] == gate.BLOCKED
