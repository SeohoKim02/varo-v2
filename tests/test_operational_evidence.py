"""P1-B operational evidence and candidate eligibility: controlled scenarios A-Z plus contract, formula and planner checks.

Lanes are plain records in the P1 universe layout; operator evidence goes through the same normalizers the CLI uses.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services import candidate_generation_research as cgr
from services import candidate_generator as cg
from services import operational_evidence as oe
from services import operational_evidence_validation as oev
from services import shared_feasibility_selection as sf
from services import suhyup_algorithm_revalidation as rv

REPO = Path(__file__).resolve().parents[1]
DATE = "2026-07-16"
NET = "net_test"
CONTRACT_FIELDS = ("candidate_id", "source_id", "target_id", "product_id", "requested_qty", "validated_qty",
                   "source_surplus_basis", "target_need_basis", "route_evidence", "capacity_evidence", "quantity_unit_evidence",
                   "cost_evidence", "operational_eligibility", "blocking_reasons", "provenance", "comparison_scope")


@pytest.fixture(autouse=True)
def legacy_mode(monkeypatch):
    monkeypatch.delenv("VARO_REAL_DATA_ROOT", raising=False)


def _lane(**overrides):
    base = {"snapshot_date": DATE, "lane_key": "P1/S/T/DIRECT/", "route_id": "R1", "product_id": "P1", "source_id": "S",
            "target_id": "T", "route_type": "DIRECT", "dc_id": None, "recommended_qty": 20, "source_stock": 100.0,
            "source_surplus": 40.0, "target_stock": 10.0, "target_need_7d": 30.0, "gen_source_stock": 100.0,
            "route_status": "ROUTE_NETWORK_TABLE", "distance_km": 50.0,
            "distance_provenance": "osrm_from_public_verified_coordinates", "unit_status": "UNDECLARED_SAME_SOURCE",
            "quantity_unit": None, "move_cost": 50000.0, "cost_provenance": "DERIVED_REAL", "target_has_inventory_row": True,
            "inventory_target_matched": True, "is_duplicate": False}
    base.update(overrides)
    return base


def _rule(**overrides):
    base = {"network_id": NET, "source_id": "S", "target_id": "T", "allowed": "true", "effective_from": "2026-07-01",
            "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "verification_status": "VERIFIED", "evidence_source": "contract-7"}
    base.update(overrides)
    return base


def _qty(kind, location, quantity, **overrides):
    base = {"network_id": NET, "location_id": location, "product_id": "P1", "quantity_kind": kind,
            "observation_start": DATE, "observation_end": DATE, "quantity": quantity, "quantity_unit": "BOX",
            "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "verification_status": "VERIFIED", "data_source": "wms-3"}
    base.update(overrides)
    return base


def _ctx(routes=(), quantities=(), observed=(), **kwargs):
    route_records, route_issues = oe.normalize_route_evidence(list(routes), network_id=NET)
    quantity_records, quantity_issues = oe.normalize_demand_evidence(list(quantities), network_id=NET)
    observed_records, observed_issues = oe.normalize_observed_transfers(list(observed), network_id=NET)
    context = oe.OperationalContext(network_id=NET, route_records=route_records, quantity_records=quantity_records,
                                    observed_transfers=observed_records, **kwargs)
    context.issues = route_issues + quantity_issues + observed_issues
    return context


def _full(**rule_overrides):
    """Every dimension evidenced: verified product route with capacity, contract cost and temperature; declared
    transferable stock, declared target need, declared unit."""
    rule = _rule(**{"product_id": "P1", "temperature_requirement": "FROZEN", "route_capacity_qty": 100, "quantity_unit": "BOX",
                    "cost_basis": sf.FIXED_PER_TRIP, "cost_value": 90000, **rule_overrides})
    return _ctx([rule], [_qty(oe.DECLARED_TRANSFERABLE, "S", 60), _qty(oe.DECLARED_TARGET_NEED, "T", 40)],
                product_units={"P1": ("BOX", oe.VERIFIED_OPERATOR_INPUT)}, product_temperature={"P1": "FROZEN"})


def _eval(lane=None, context=None):
    return oe.evaluate_lane(lane or _lane(), context or _ctx())


def _reasons(record):
    return set((record["blocking_reasons"] or "").split("|")) - {""}


# --------------------------------------------------------------------------------------------- A-Z scenarios


def test_a_operator_allowed_route_is_operational_evidence():
    record = _eval(context=_ctx([_rule()]))
    assert record["route_class"] == oe.ROUTE_B and record["route_accepted"] is True
    assert "ROUTE_NOT_OPERATIONALLY_EVIDENCED" not in _reasons(record)
    observed = _eval(context=_ctx(observed=[{"network_id": NET, "source_id": "S", "target_id": "T", "product_id": "P1",
                                             "transfer_date": "2026-07-10", "evidence_source": "tms"}]))
    assert observed["route_class"] == oe.ROUTE_A and observed["route_accepted"] is True
    assert "ROUTE_RECENCY_POLICY_UNDECLARED" in observed["conditional_reasons"]


def test_b_road_distance_alone_never_verifies_a_route():
    record = _eval()
    assert record["route_class"] == oe.ROUTE_C and record["route_accepted"] is False
    assert "ROUTE_NOT_OPERATIONALLY_EVIDENCED" in _reasons(record)
    assert record["operational_eligibility"] == oe.PROXY_FEASIBLE and record["strict_eligibility"] == oe.INSUFFICIENT_OPERATIONAL_EVIDENCE
    undeclared = _eval(_lane(distance_provenance=None))
    assert undeclared["route_class"] == oe.ROUTE_D and undeclared["route_accepted"] is False
    no_row = _eval(_lane(route_status="ROUTE_UNVERIFIED", distance_km=None))
    assert no_row["route_class"] == oe.ROUTE_E and no_row["operational_eligibility"] == "ROUTE_UNVERIFIED"


def test_c_prohibited_route_is_ineligible_and_history_does_not_override_it():
    record = _eval(context=_ctx([_rule(allowed="false")]))
    assert record["operational_eligibility"] == oe.INELIGIBLE and oe.ROUTE_PROHIBITED in _reasons(record)
    before = {"network_id": NET, "source_id": "S", "target_id": "T", "transfer_date": "2026-06-20", "evidence_source": "tms"}
    later_rule = _ctx([_rule(allowed="false")], observed=[before])
    assert _eval(context=later_rule)["operational_eligibility"] == oe.INELIGIBLE
    during = {**before, "transfer_date": "2026-07-05"}
    conflict = _eval(context=_ctx([_rule(allowed="false")], observed=[during]))
    assert conflict["route_class"] == oe.ROUTE_CONFLICT and "OBSERVED_TRANSFER_UNDER_PROHIBITION" in _reasons(conflict)
    assert conflict["operational_eligibility"] != oe.INELIGIBLE and conflict["route_accepted"] is False
    table_flag = _eval(_lane(route_status="ROUTE_FORBIDDEN"))
    assert table_flag["operational_eligibility"] == oe.INELIGIBLE and "ROUTE_TABLE_FLAG_FORBIDDEN" in _reasons(table_flag)


def test_d_target_handling_from_current_stock_or_operator_registration():
    record = _eval()
    assert record["target_handling_accepted"] is True and record["handling_evidence"].startswith("CURRENT_STOCK_POSITIVE")
    zero = _eval(_lane(target_stock=0.0))
    assert zero["handling_evidence"].startswith("CURRENT_ROW_ZERO_STOCK") and zero["target_handling_accepted"] is True
    registered = _eval(_lane(target_has_inventory_row=False, target_stock=None, target_need_7d=None),
                       _ctx(quantities=[_qty(oe.DECLARED_RECEIVABLE, "T", None, quantity_unit=None)]))
    assert registered["target_handling_accepted"] is True and registered["handling_evidence"].startswith("OPERATOR_DECLARED_RECEIVABLE")


def test_e_target_without_the_product_is_not_receivable_without_evidence():
    lane = _lane(target_has_inventory_row=False, target_stock=None, target_need_7d=None)
    record = _eval(lane)
    assert record["operational_eligibility"] == "TARGET_HANDLING_UNVERIFIED"
    assert "TARGET_HANDLING_NOT_EVIDENCED" in _reasons(record)
    history = _eval(lane, _ctx(handling_history={("T", "P1"): "2026-07-02"}))
    assert history["handling_evidence"] == "HISTORICAL_ONLY:PROXY" and history["target_handling_accepted"] is False
    # a general (all-products) route allowance does not make the target receive this product
    general = _eval(lane, _ctx([_rule()]))
    assert general["target_handling_accepted"] is False


def test_f_actual_source_surplus():
    declared = _eval(context=_ctx(quantities=[_qty(oe.DECLARED_TRANSFERABLE, "S", 60)]))
    assert declared["source_supply_accepted"] is True and declared["source_surplus_validation_status"] == "VERIFIED"
    assert declared["source_surplus_basis"] == "OPERATOR_DECLARED_TRANSFERABLE" and declared["source_surplus_value"] == 60
    column = _eval(_lane(source_surplus_provenance="DIRECT_REAL"))
    assert column["source_supply_accepted"] is True and column["source_surplus_provenance"] == oe.DIRECT_REAL
    sales = _eval(_lane(quantity_unit="BOX"), _ctx(quantities=[
        _qty(oe.ACTUAL_SALES, "S", 70, observation_start="2026-07-09", observation_end="2026-07-15", evidence_class="DIRECT_REAL")]))
    assert sales["source_supply_accepted"] is True and sales["source_surplus_value"] == pytest.approx(100 - 70)
    too_much = _eval(context=_ctx(quantities=[_qty(oe.DECLARED_TRANSFERABLE, "S", 150)]))
    assert too_much["source_supply_accepted"] is False and "DECLARED_TRANSFERABLE_EXCEEDS_ON_HAND" in _reasons(too_much)


def test_g_proxy_source_surplus_and_on_hand_stock_are_not_transferable_evidence():
    record = _eval()
    assert record["source_supply_accepted"] is False and record["source_surplus_provenance"] == oe.PROXY
    assert {"SOURCE_SURPLUS_MEDIAN_RULE_PROXY", "ONLY_ON_HAND_STOCK_OBSERVED"} <= _reasons(record)
    no_surplus = _eval(_lane(source_surplus=None))
    assert no_surplus["source_supply_accepted"] is False and "SOURCE_SURPLUS_MISSING" in _reasons(no_surplus)


def test_h_actual_target_need_declared_or_from_actual_sales():
    declared = _eval(context=_ctx(quantities=[_qty(oe.DECLARED_TARGET_NEED, "T", 25)]))
    assert declared["target_demand_accepted"] is True and declared["target_need_formula"] == oe.OPERATOR_DECLARED_FORMULA
    sales = _eval(context=_ctx(quantities=[_qty(oe.ACTUAL_SALES, "T", 140, observation_start="2026-07-09",
                                                observation_end="2026-07-15", evidence_class="DIRECT_REAL")]))
    assert sales["target_demand_accepted"] is True and sales["target_need_formula"] == oe.ACTUAL_SALES_FORMULA
    assert sales["target_need_value"] == pytest.approx(7 * 20 - 10) and sales["target_need_provenance"] == oe.DERIVED_REAL


def test_i_proxy_target_need_and_outbound_never_become_actual_demand():
    record = _eval()
    assert record["target_demand_accepted"] is False and "TARGET_NEED_PROXY_OUTBOUND" in _reasons(record)
    context = _ctx(quantities=[_qty(oe.OUTBOUND_SHIPMENT, "T", 90, evidence_class="DIRECT_REAL")])
    assert any(issue["code"] == "PROXY_KIND_RECLASSIFIED" for issue in context.issues)
    outbound = _eval(context=context)
    assert outbound["target_demand_accepted"] is False and "OPERATOR_DEMAND_IS_PROXY_KIND" in _reasons(outbound)


def test_j_two_need_formulas_reproduce_the_0716_case_and_are_never_compared_as_one():
    assert oe.generator_need(45, 102, 9) == 63 and oe.benchmark_need(45, 102, 9) == 6
    assert oe.need_formula_gap(45, 102, 9) == 57 and oe.need_case(45, 102, 9) == "STOCK_ABOVE_MEDIAN_WITHIN_7D_OUTFLOW"
    lane = _lane(recommended_qty=50, source_surplus=71.0, gen_target_need=63.0, target_need_7d=6.0, target_stock=102.0)
    record = _eval(lane)
    assert record["operational_eligibility"] == "DEMAND_UNVERIFIED" and "PROXY_TARGET_NEED_BELOW_REQUESTED_QTY" in _reasons(record)
    assert record["proxy_feasible_qty"] == 6
    left = {"service": 250, "cost": 1.0, "eligibility_scope": oe.SCOPE_BENCHMARK, "constraint_definition": "TARGET=GENERATOR_NEED",
            "cost_basis": "COMPUTED_TARIFF", "quantity_basis": "FIXED_BINARY", "eligible_count": 20}
    right = {**left, "constraint_definition": "TARGET=BENCHMARK_NEED", "cost": 2.0}
    compared = oe.fair_cost_comparison(left, right)
    assert compared["status"] == oe.NOT_COMPARABLE and "CONSTRAINT_DEFINITION_DIFFERS" in compared["reason_codes"]


def test_k_matching_declared_units_are_accepted():
    record = _eval(_lane(quantity_unit="BOX", unit_status="MATCHED"))
    assert record["quantity_unit_accepted"] is True and record["quantity_unit_evidence"] == "MATCHED:DIRECT_REAL"
    declared = _eval(context=_ctx(product_units={"P1": ("box", oe.VERIFIED_OPERATOR_INPUT)}))
    assert declared["quantity_unit_accepted"] is True


def test_l_unknown_unit_is_benchmark_only_and_a_mismatch_is_ineligible():
    record = _eval(context=_ctx(unit_weight_status={"P1": "safe"}))
    assert record["quantity_unit_accepted"] is False and {"QUANTITY_UNIT_UNKNOWN", "KG_CONVERSION_DERIVED_ONLY"} <= _reasons(record)
    assert record["comparison_scope"] == oe.SCOPE_BENCHMARK
    mismatch = _eval(_lane(unit_status="UNIT_MISMATCH"))
    assert mismatch["operational_eligibility"] == oe.INELIGIBLE and "UNIT_MISMATCH" in _reasons(mismatch)
    one_side = _eval(_lane(unit_status="UNIT_UNKNOWN"))
    assert one_side["operational_eligibility"] == "UNIT_UNVERIFIED" and one_side["comparison_scope"] == oe.SCOPE_NONE


def test_m_actual_route_capacity_limits_the_validated_quantity():
    record = _eval(_lane(recommended_qty=50), _full(route_capacity_qty=35))
    assert record["capacity_accepted"] is True and record["strict_eligibility"] == oe.STRICT_ELIGIBLE
    assert record["validated_qty"] == 35 and "PARTIAL_QTY_ONLY" in _reasons(record)
    zero = _eval(_lane(), _full(route_capacity_qty=0))
    assert zero["operational_eligibility"] == oe.INELIGIBLE and "VERIFIED_CAPACITY_BELOW_ONE_UNIT" in _reasons(zero)


def test_n_unknown_route_capacity_leaves_a_verified_lane_conditional():
    record = _eval(context=_full(route_capacity_qty=None))
    assert record["operational_eligibility"] == oe.VERIFIED_CONDITIONAL and record["strict_eligibility"] == oe.STRICT_ELIGIBLE
    assert "ROUTE_CAPACITY_MISSING" in record["conditional_reasons"]
    via = _eval(_lane(route_type="VIA_DC", dc_id="DC"), _ctx([_rule(target_id="DC"), _rule(source_id="DC")]))
    assert via["route_accepted"] is True and "DC_CAPACITY_MISSING" in _reasons(via)


def test_o_contract_cost_is_actual_spend_evidence():
    record = _eval(context=_full())
    assert record["cost_evidence"] == "CONTRACT_COST:VERIFIED_OPERATOR_INPUT" and record["transport_cost"] == 90000
    assert "COST_NOT_ACTUAL_SPEND" not in (record["conditional_reasons"] or "")
    per_unit = _eval(_lane(recommended_qty=20), _full(cost_basis=sf.PER_UNIT, cost_value=1500))
    assert per_unit["transport_cost"] == 30000


def test_p_calculated_tariff_is_derived_and_a_proxy_vehicle_is_not_accepted():
    record = _eval()
    assert record["transport_cost_accepted"] is True and record["cost_evidence"] == "CALCULATED_TARIFF:DERIVED_REAL"
    assert "COST_CALCULATED_NOT_ACTUAL_SPEND" in record["conditional_reasons"]
    proxy = _eval(_lane(cost_provenance="PROXY"))
    assert proxy["transport_cost_accepted"] is False and "COST_PROXY_VEHICLE_CLASS" in _reasons(proxy)
    unknown = _eval(_lane(move_cost=None, cost_provenance="UNKNOWN"))
    assert unknown["operational_eligibility"] == "COST_UNVERIFIED" and unknown["comparison_scope"] == oe.SCOPE_NONE


def test_q_verified_user_input_is_accepted_like_t1_user_input():
    assert oe.evidence_class("USER_INPUT", "VERIFIED") == oe.VERIFIED_OPERATOR_INPUT
    assert oe.base_provenance(oe.VERIFIED_OPERATOR_INPUT) == "USER_INPUT" and oe.strict_accepted(oe.VERIFIED_OPERATOR_INPUT)
    record = _eval(context=_ctx(quantities=[_qty(oe.DECLARED_TARGET_NEED, "T", 25, evidence_class="USER_INPUT")]))
    assert record["target_demand_accepted"] is True and record["target_need_provenance"] == oe.VERIFIED_OPERATOR_INPUT


def test_r_user_estimates_never_verify():
    assert oe.evidence_class("USER_INPUT") == oe.USER_ESTIMATE
    assert oe.evidence_class(oe.VERIFIED_OPERATOR_INPUT, "ESTIMATE") == oe.USER_ESTIMATE
    assert not oe.strict_accepted(oe.USER_ESTIMATE) and oe.base_provenance(oe.USER_ESTIMATE) == "SCENARIO_INPUT"
    route = _eval(context=_ctx([_rule(evidence_class=oe.USER_ESTIMATE, verification_status="ESTIMATE")]))
    assert route["route_accepted"] is False and "ROUTE_ESTIMATE_ONLY" in _reasons(route)
    need = _eval(context=_ctx(quantities=[_qty(oe.DECLARED_TARGET_NEED, "T", 25, evidence_class="USER_INPUT", verification_status="")]))
    assert need["target_demand_accepted"] is False and "DECLARED_NEED_ESTIMATE_ONLY" in _reasons(need)


def test_s_expired_or_future_inputs_are_not_current_evidence():
    route = _eval(context=_ctx([_rule(effective_to="2026-07-10")]))
    assert route["route_accepted"] is False and "ROUTE_EVIDENCE_EXPIRED" in _reasons(route)
    stale = _eval(context=_ctx(quantities=[_qty(oe.ACTUAL_SALES, "T", 140, observation_start="2026-06-20",
                                                observation_end="2026-06-30", evidence_class="DIRECT_REAL")]))
    assert stale["target_demand_accepted"] is False and "DEMAND_EVIDENCE_EXPIRED" in _reasons(stale)
    future = _eval(context=_ctx(quantities=[_qty(oe.ACTUAL_SALES, "T", 140, observation_start="2026-07-17",
                                                 observation_end="2026-07-20", evidence_class="DIRECT_REAL")]))
    assert "DEMAND_EVIDENCE_FROM_FUTURE" in _reasons(future)
    old_need = _eval(context=_ctx(quantities=[_qty(oe.DECLARED_TARGET_NEED, "T", 25, observation_start="2026-07-01",
                                                   observation_end="2026-07-02")]))
    assert old_need["target_demand_accepted"] is False and "DECLARED_NEED_OUTSIDE_VALIDITY" in _reasons(old_need)


def test_t_same_scope_conflicts_are_explicit_and_specific_scope_wins():
    conflict = _eval(context=_ctx([_rule(), _rule(allowed="false", evidence_source="policy-2")]))
    assert conflict["route_class"] == oe.ROUTE_CONFLICT and conflict["route_accepted"] is False
    assert conflict["operational_eligibility"] != oe.INELIGIBLE
    specific = _eval(context=_ctx([_rule(allowed="false"), _rule(product_id="P1")]))
    assert specific["route_accepted"] is True
    specific_ban = _eval(context=_ctx([_rule(), _rule(product_id="P1", allowed="false")]))
    assert specific_ban["operational_eligibility"] == oe.INELIGIBLE
    estimate = _eval(context=_ctx([_rule(), _rule(allowed="false", evidence_class=oe.USER_ESTIMATE, verification_status="ESTIMATE")]))
    assert estimate["route_accepted"] is True and "ROUTE_ESTIMATE_DISAGREES_WITH_VERIFIED_RULE" in estimate["conditional_reasons"]
    needs = _eval(context=_ctx(quantities=[_qty(oe.DECLARED_TARGET_NEED, "T", 25), _qty(oe.DECLARED_TARGET_NEED, "T", 30)]))
    assert needs["target_demand_accepted"] is False and "DECLARED_NEED_CONFLICT" in _reasons(needs)


def test_u_evidence_from_another_business_is_rejected():
    context = _ctx([_rule(network_id="other_company")], [_qty(oe.DECLARED_TARGET_NEED, "T", 25, network_id="other_company")])
    assert {issue["code"] for issue in context.issues} == {"CROSS_NETWORK_EVIDENCE_REJECTED"}
    record = _eval(context=context)
    assert record["route_accepted"] is False and record["target_demand_accepted"] is False
    foreign = _eval(_lane(network_id="other_company"))
    assert foreign["operational_eligibility"] == oe.INELIGIBLE and "CROSS_NETWORK_LANE" in _reasons(foreign)


def test_v_price_and_demand_of_different_datasets_are_never_joined():
    price = oe.DataScope("kamis", "public_price_statistics", "kamis_item", "kamis_market")
    demand = oe.DataScope("suhyup_202607_31d", "suhyup_logistics_centers", "suhyup_product_code", "suhyup_center_code")
    guard = oe.join_guard(price, demand)
    assert guard["allowed"] is False and "DIFFERENT_OPERATIONAL_NETWORK" in guard["reason_codes"]
    canonical = oe.DataScope("suhyup_canonical", "suhyup_logistics_centers", "suhyup_product_code", "suhyup_canonical_location_id")
    assert oe.join_guard(demand, canonical)["allowed"] is True
    assert oe.normalize_location_id("000156", "suhyup_canonical_location_id") == "156"
    assert oe.join_guard(oe.DataScope("a", "n", "p", "l", unit="BOX"), oe.DataScope("b", "n", "p", "l", unit="KG"))["reason_codes"] == ["UNIT_MISMATCH"]


def test_w_no_executable_candidate_is_reported_not_hidden():
    records = oe.evaluate_lanes([_lane(), _lane(route_id="R2", lane_key="P1/S/U/DIRECT/", target_id="U")], _ctx())
    assert (records["strict_eligibility"] == oe.STRICT_ELIGIBLE).sum() == 0
    assert records["blocking_reasons"].notna().all()
    plan = oe.plan_missing_inputs(records)
    required = plan[plan["requirement"] == "REQUIRED_FOR_STRICT"]
    assert set(required["field"]) == {"verified_allowed_route", "actual_source_transferable", "actual_target_need", "quantity_unit"}
    assert required["cumulative_strict_eligible_upper_bound"].iloc[-1] == 2
    strict = {"service": 0, "cost": 0, "eligibility_scope": oe.SCOPE_STRICT, "constraint_definition": "STRICT",
              "cost_basis": "ACTUAL_SPEND", "quantity_basis": "FIXED_BINARY", "eligible_count": 0}
    assert oe.fair_cost_comparison(strict, strict)["reason_codes"] == ["NO_ELIGIBLE_CANDIDATES"]


def test_x_only_some_candidates_verified():
    context = _full()
    records = oe.evaluate_lanes([_lane(), _lane(route_id="R2", lane_key="P2/S/T/DIRECT/", product_id="P2")], context)
    assert list(records["strict_eligibility"]) == [oe.STRICT_ELIGIBLE, oe.INSUFFICIENT_OPERATIONAL_EVIDENCE]
    plan = oe.plan_missing_inputs(records)
    assert plan.loc[plan["requirement"] == "REQUIRED_FOR_STRICT", "candidates_requiring"].max() == 1


def test_y_proxy_only_lane_is_benchmark_comparable_but_never_validated():
    record = _eval()
    assert record["operational_eligibility"] == oe.PROXY_FEASIBLE and record["proxy_feasible"] is True
    assert record["validated_qty"] is None and record["proxy_feasible_qty"] == 20
    assert record["comparison_scope"] == oe.SCOPE_BENCHMARK


def test_z_every_requirement_met_is_strict_eligible():
    context = _full()
    record = _eval(context=context)
    assert record["operational_eligibility"] == oe.VERIFIED_OPERATIONAL and record["strict_eligibility"] == oe.STRICT_ELIGIBLE
    assert record["validated_qty"] == 20 and record["comparison_scope"] == oe.SCOPE_STRICT
    assert record["blocking_reasons"] is None and record["conditional_reasons"] is None
    assert oe.caps_only_strict(_lane(), context) is True
    left = {"service": 20, "cost": 90000, "eligibility_scope": oe.SCOPE_STRICT, "constraint_definition": "STRICT",
            "cost_basis": "ACTUAL_SPEND", "quantity_basis": "FIXED_BINARY", "eligible_count": 1}
    assert oe.fair_cost_comparison(left, {**left, "cost": 100000})["claim"] == "ACTUAL_COST_DIFFERENCE"


# --------------------------------------------------------------------------------------------- contract and formulas


def test_record_contract_fields_and_nulls():
    record = _eval(_lane(source_surplus=None, target_need_7d=None))
    assert all(field in record for field in CONTRACT_FIELDS)
    assert record["validated_qty"] is None and record["source_surplus_value"] is None and record["target_need_value"] is None
    assert record["operational_eligibility"] in oe.ELIGIBILITY_STATUSES


def test_evidence_classes_keep_the_existing_provenance_contract():
    for evidence in oe.EVIDENCE_CLASSES:
        assert oe.strict_accepted(evidence) == (oe.base_provenance(evidence) in sf.STRICT_ACCEPTED_PROVENANCE
                                                and evidence in oe.ACCEPTED_EVIDENCE)
    assert {oe.base_provenance(e) for e in oe.EVIDENCE_CLASSES} <= set(oe.BASE_PROVENANCE.values())
    assert oe.evidence_class("actual") == oe.DIRECT_REAL and oe.evidence_class("derived_proxy_from_actual_outbound_qty") == oe.PROXY
    assert oe.evidence_class(None) == oe.MISSING and oe.evidence_class("whatever") == oe.UNKNOWN


def test_need_formula_identity_and_cases():
    rng = np.random.default_rng(7)
    for median, stock, demand in rng.uniform(0, 200, size=(500, 3)):
        gap = oe.generator_need(median, stock, demand) - oe.benchmark_need(median, stock, demand)
        assert gap == pytest.approx(oe.need_formula_gap(median, stock, demand)) and gap >= -1e-9
    cases = {oe.need_case(m, s, d) for m, s, d in itertools.product((10.0,), (5.0, 10.0, 15.0, 30.0), (0.0, 1.0))}
    assert cases == set(oe.NEED_CASES)
    assert oe.generator_need(10, 5, 0) == oe.benchmark_need(10, 5, 0) == 5           # stock < median, no outflow
    assert oe.generator_need(10, 10, 2) == oe.benchmark_need(10, 10, 2) == 14        # stock = median
    assert oe.generator_need(10, 40, 2) == 14 and oe.benchmark_need(10, 40, 2) == 0  # stock beyond median + 7d


def test_templates_are_sample_only_and_match_the_contract():
    route = pd.read_csv(REPO / "samples" / "operational_route_evidence_TEMPLATE_SAMPLE.csv", dtype=str)
    demand = pd.read_csv(REPO / "samples" / "operational_demand_evidence_TEMPLATE_SAMPLE.csv", dtype=str)
    assert list(route.columns) == [*oe.ROUTE_EVIDENCE_REQUIRED, *oe.ROUTE_EVIDENCE_OPTIONAL]
    assert list(demand.columns) == [*oe.DEMAND_EVIDENCE_REQUIRED, *oe.DEMAND_EVIDENCE_OPTIONAL]
    assert len(route) == len(oe.route_evidence_template_rows()) and len(demand) == len(oe.demand_evidence_template_rows())
    for frame, normalize in ((route, oe.normalize_route_evidence), (demand, oe.normalize_demand_evidence)):
        records, issues = normalize(frame, network_id="SAMPLE_NETWORK")
        assert records == [] and {issue["code"] for issue in issues} == {"SAMPLE_ROW_NOT_USABLE"}
        assert frame["note"].str.startswith("SAMPLE").all()


def test_input_validation_rejects_malformed_rows():
    _, issues = oe.normalize_route_evidence([_rule(allowed="maybe"), _rule(effective_from="2026-07-10", effective_to="2026-07-01"),
                                             _rule(evidence_source=None), _rule(temperature_requirement="WARM")], network_id=NET)
    assert [issue["code"] for issue in issues] == ["INVALID_ALLOWED_FLAG", "INVALID_EFFECTIVE_WINDOW", "MISSING_REQUIRED_FIELD",
                                                   "INVALID_TEMPERATURE"]
    records, issues = oe.normalize_route_evidence([_rule(route_capacity_qty=10), _rule(cost_value=5)], network_id=NET)
    assert records[0]["route_capacity_qty"] is None and records[1]["cost_value"] is None
    assert {issue["code"] for issue in issues} == {"ROUTE_CAPACITY_UNIT_MISSING", "ROUTE_COST_BASIS_UNDECLARED"}
    _, issues = oe.normalize_demand_evidence([_qty("SALES", "T", 5), _qty(oe.ACTUAL_SALES, "T", -1), _qty(oe.ACTUAL_SALES, "T", 5, quantity_unit=None)],
                                             network_id=NET)
    assert [issue["code"] for issue in issues] == ["UNKNOWN_QUANTITY_KIND", "INVALID_QUANTITY", "QUANTITY_UNIT_MISSING"]


def test_temperature_mismatch_and_cold_chain_condition():
    mismatch = _eval(context=_ctx([_rule(temperature_requirement="AMBIENT")], product_temperature={"P1": "FROZEN"}))
    assert mismatch["operational_eligibility"] == oe.INELIGIBLE and "ROUTE_TEMPERATURE_MISMATCH" in _reasons(mismatch)
    undeclared = _eval(context=_ctx([_rule()], product_temperature={"P1": "FROZEN"}))
    assert undeclared["route_accepted"] is True and "COLD_CHAIN_UNVERIFIED" in undeclared["conditional_reasons"]


def test_planner_orders_by_unlocked_candidates():
    context = _ctx([_rule()], product_units={"P1": ("BOX", oe.VERIFIED_OPERATOR_INPUT)})
    lanes = [_lane(), _lane(route_id="R2", lane_key="P1/S/U/DIRECT/", target_id="U")]
    plan = oe.plan_missing_inputs(oe.evaluate_lanes(lanes, context))
    required = plan[plan["requirement"] == "REQUIRED_FOR_STRICT"]
    # both lanes need supply and demand; only S->U needs the route.  Supply needs 1 key, demand 2: supply first
    assert list(required["field"]) == ["actual_source_transferable", "actual_target_need", "verified_allowed_route"]
    assert list(required["candidates_requiring"]) == [2, 2, 1]
    assert list(required["cumulative_strict_eligible_upper_bound"]) == [0, 1, 2]
    assert set(plan.loc[plan["requirement"] != "REQUIRED_FOR_STRICT", "field"]) == {"route_capacity"}


# --------------------------------------------------------------------------------------------- P1 universe integration


def _upload(inventory):
    nodes = rv.NETWORK_NODES
    stores = pd.DataFrame([{"node_id": n, "node_name": f"n{n}", "node_type": "STORE"} for n in nodes])
    products = sorted({row[1] for row in inventory})
    routes = pd.DataFrame([{"source_id": s, "target_id": t, "distance_km": 30.0, "estimated_cost": 5000.0 + 10 * i,
                            "travel_time_min": 30.0} for i, (s, t) in enumerate((s, t) for s in nodes for t in nodes if s != t)])
    return {"stores": stores, "products": pd.DataFrame({"product_id": products, "product_name": products, "unit_price": 2000.0}),
            "inventory": pd.DataFrame([{"store_id": s, "product_id": p, "stock_qty": q, "sales_qty": d} for s, p, q, d in inventory]),
            "routes": routes}


def _universe(upload):
    lanes, _, _ = cgr.enumerate_universe(upload, transport_mode=False, snapshot_date=DATE)
    priced = cgr.price_universe(lanes, None)
    capped = cgr.attach_benchmark_caps(priced, cgr.upload_inventory_flow(upload["inventory"], DATE))
    store_ids, _ = cg._store_ids_by_type(upload["stores"])
    return cgr.validate_universe(capped, store_ids=store_ids, product_ids=upload["products"]["product_id"].astype(str))


def _many_products(n=24):
    a, b, c, d, e, f = rv.NETWORK_NODES
    inventory = []
    for index in range(n):
        product = f"P{index:02d}"
        inventory += [(a, product, 300 + index, 0), (b, product, 10, 1 + index % 4), (c, product, 120, 3), (d, product, 60, 0)]
    return _upload(inventory)


def test_regenerate_with_generator_need_reproduces_the_production_pool():
    universe = _universe(_many_products())
    legacy = cgr.policy_pool(universe, cgr.LEGACY_20)
    regenerated = oe.regenerate_with_need(universe, "gen_target_need")
    assert len(legacy) == cg.MAX_CANDIDATES and list(regenerated["lane_key"]) == list(legacy["lane_key"])
    assert list(regenerated["recommended_qty"].astype(int)) == list(legacy["recommended_qty"].astype(int))
    production, _ = cg.generate_candidates(_many_products())
    assert list(production["product_id"]) == list(regenerated["product_id"])


def test_benchmark_need_regeneration_changes_targets_above_the_median():
    a, b, c, d, *_ = rv.NETWORK_NODES
    # median 180.  b holds 260 (80 above the median) and ships 20/day: generator 0 + 140 = 140, benchmark 180 + 140 - 260
    # = 60.  c holds 90 with no outflow: both formulas 90.  The generator sends a's surplus to b, the benchmark to c.
    universe = _universe(_upload([(a, "P1", 500, 0), (b, "P1", 260, 20), (c, "P1", 90, 0), (d, "P1", 100, 0)]))
    universe["bench_need"] = [oe.benchmark_need(m, s, dd) for m, s, dd in
                              zip(universe["gen_median_stock"], universe["gen_target_stock"].fillna(0), universe["gen_target_demand"].fillna(0))]
    generator = oe.regenerate_with_need(universe, "gen_target_need")
    benchmark = oe.regenerate_with_need(universe, "bench_need")
    from_a = lambda frame: frame[frame["source_id"] == a].iloc[0]
    assert from_a(generator)["target_id"] == b and from_a(generator)["gen_target_need"] == 140
    assert from_a(benchmark)["target_id"] == c and from_a(benchmark)["bench_need"] == 90


def test_universe_lanes_without_operator_evidence_are_never_strict():
    universe = _universe(_many_products(6))
    routed = universe[universe["recommended_qty"].notna()]
    records = oe.evaluate_lanes(routed, oe.OperationalContext(network_id=NET))
    assert (records["strict_eligibility"] == oe.STRICT_ELIGIBLE).sum() == 0
    valid = records[(routed["validity_status"] == cgr.VALID).to_numpy()]
    assert set(valid["operational_eligibility"]) <= {oe.PROXY_FEASIBLE, "SUPPLY_UNVERIFIED", "DEMAND_UNVERIFIED", "COST_UNVERIFIED"}
    assert not records["route_accepted"].any()


def test_sensitivity_pool_helpers():
    universe = _universe(_many_products(6))
    pool = cgr.policy_pool(universe, cgr.ALL_VALID)
    switched = oev._with_need(pool, oe.GENERATOR_FORMULA)
    assert (switched["target_need_7d"] == pd.to_numeric(pool["gen_target_need"])).all()
    assert (switched["target_need_7d_benchmark"] == pd.to_numeric(pool["target_need_7d"])).all()
    generator_pool = oev.all_valid_generator_pool(universe)
    assert set(pool["lane_key"]) <= set(generator_pool["lane_key"])


def test_cli_writes_only_its_own_bundle():
    assert oev.OUTPUT_FOLDER == "_OPERATIONAL_EVIDENCE_VALIDATION"
    assert oev.OUTPUT_FOLDER not in oev.PROTECTED_BUNDLES
    for name in ("operational_candidate_evidence_summary.csv", "operational_candidate_evidence_by_day.csv",
                 "operational_candidate_eligibility.csv", "operational_route_evidence.csv", "operational_demand_semantics.csv",
                 "target_need_formula_comparison.csv", "operational_missing_inputs.csv", "operational_candidate_validation.json"):
        assert name in oev.OUTPUT_FILES
