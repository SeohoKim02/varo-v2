"""P1-C target need contract and operator evidence intake: controlled scenarios A-AD, the T1 STRICT bridge, the
optional workbook sheets and the parallel pipeline field.

Lanes are plain records in the P1 universe layout (as in the P1-B tests); operator evidence goes through the same
intake the CLI and the pipeline use.
"""
from __future__ import annotations

import copy
import io
import random
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from services import analysis_pipeline
from services import operational_evidence as oe
from services import shared_feasibility_selection as sf
from services import target_need_contract as tn
from services import target_need_contract_validation as tnv
from services.analysis_pipeline import build_v2_state
from services.data_loader import OPTIONAL_SHEETS, load_excel_data

REPO = Path(__file__).resolve().parents[1]
NETWORK_SAMPLE = REPO / "data" / "Varo_V2_네트워크_샘플.xlsx"
DATE = "2026-07-16"
NET = "net_test"
UNIVERSE = tn.KeyUniverse(network_id=NET, dataset_id="ds_test", locations=frozenset({"S", "T", "U", "V"}),
                          products=frozenset({"P1", "P2"}))


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


def _trust(level=tn.INPUT_CONFIRMED, origin=tn.ACTUAL_OPERATION, **overrides):
    base = {"record_origin": origin, "verification_status": "VERIFIED", "verification_level": level,
            "verified_by": "ops-lead", "verified_at": "2026-07-15"}
    base.update(overrides)
    return base


def _route(**overrides):
    base = {"network_id": NET, "source_id": "S", "target_id": "T", "product_id": "P1", "allowed": "true",
            "effective_from": "2026-07-01", "effective_to": "2026-07-31", "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
            "evidence_source": "contract-7", "route_capacity_qty": 100, "quantity_unit": "BOX",
            "cost_basis": sf.FIXED_PER_TRIP, "cost_value": 90000, **_trust(tn.OPERATIONALLY_APPROVED)}
    base.update(overrides)
    return base


def _need(quantity=40, location="T", **overrides):
    base = {"network_id": NET, "location_id": location, "product_id": "P1", "quantity_kind": oe.DECLARED_TARGET_NEED,
            "observation_start": DATE, "observation_end": DATE, "quantity": quantity, "quantity_unit": "BOX",
            "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "data_source": "plan-1", **_trust()}
    base.update(overrides)
    return base


def _inventory(location="S", **overrides):
    base = {"network_id": NET, "location_id": location, "product_id": "P1", "effective_date": DATE,
            "quantity_unit": "BOX", "evidence_class": oe.DIRECT_REAL, "evidence_source": "wms-1", **_trust(tn.OBSERVED_RECORD)}
    base.update(overrides)
    return base


def _unit(product="P1", unit="BOX", **overrides):
    base = {"network_id": NET, "location_id": "S", "product_id": product, "effective_date": "2026-07-01",
            "valid_until": "2026-12-31", "quantity_unit": unit, "dataset_unit_confirmed": "true",
            "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "evidence_source": "product-master", **_trust()}
    base.update(overrides)
    return base


def _full_inputs(**changes):
    """Every dimension evidenced for lane S->T / P1: approved route with capacity and contract cost, operator need,
    observed available-for-transfer stock and a confirmed dataset unit."""
    inputs = {tn.ROUTE_SHEET: [_route()], tn.DEMAND_SHEET: [_need()],
              tn.INVENTORY_SHEET: [_inventory(on_hand_qty=100, reserved_qty=30, available_for_transfer_qty=40), _unit()]}
    inputs.update(changes)
    return inputs


def _intake(inputs, allow_test=False, universe=UNIVERSE):
    return tn.intake_evidence(inputs, universe, allow_test=allow_test)


def _evaluate(lane=None, inputs=None, allow_test=False, when=DATE):
    intake = _intake(inputs or {}, allow_test)
    evidence = tn.day_evidence(intake, when)
    context = tn.oe_context(evidence, {"network_id": NET})
    return tn.evaluate_lane_contract(lane or _lane(), evidence, context), intake


def _reasons(record):
    return {part for key in ("blocking_reasons", "contract_reasons") for part in (record[key] or "").split("|") if part}


def _issues(intake):
    return {code.split(":")[0] for row in intake.rows for code in (row["issues"] or "").split("|") if code}


# --------------------------------------------------------------------------------------------- A-AD scenarios


def test_a_generator_need_formula_is_a_ranking_proxy():
    need = tn.generator_proxy_need(network_id=NET, location_id="T", product_id="P1", as_of=DATE, median=45, stock=102, demand=9)
    assert need.target_need_value == 63 and need.formula_id == tn.GENERATOR_PROXY and need.formula_version == "v1"
    assert need.intended_use == tn.GENERATOR_RANKING and need.evidence == oe.PROXY and not need.accepted
    assert "GENERATOR_RANKING_ONLY_NEVER_A_CAP" in need.reason_codes
    record = need.as_dict()
    assert record["target_stock_policy"].startswith("CROSS_NODE_MEDIAN_STOCK") and record["demand_signal_type"] == oe.OUTBOUND_SHIPMENT


def test_b_benchmark_need_formula_is_a_labelled_research_cap():
    need = tn.benchmark_proxy_need(network_id=NET, location_id="T", product_id="P1", as_of=DATE, median=45, stock=102, outbound=9)
    assert need.target_need_value == 6 and need.formula_id == tn.BENCHMARK_PROXY and need.intended_use == tn.BENCHMARK_CAP
    assert need.validation_status == tn.PROXY_STATUS and not need.accepted
    assert {"TARGET_NEED_PROXY_OUTBOUND", "MEDIAN_IS_A_RULE"} <= set(need.reason_codes)


def test_c_equal_values_are_still_two_formulas():
    common = dict(network_id=NET, location_id="T", product_id="P1", as_of=DATE, median=50, stock=20, unit="BOX")
    generator = tn.generator_proxy_need(demand=3, **common)
    benchmark = tn.benchmark_proxy_need(outbound=3, **common)
    assert generator.target_need_value == benchmark.target_need_value == 51
    verdict = tn.need_comparability(generator, benchmark)
    assert verdict["status"] == oe.NOT_COMPARABLE and verdict["reason_codes"] == ["FORMULA_DIFFERS"]


def test_d_the_0716_case_63_vs_6_is_the_formula_and_not_comparable():
    common = dict(network_id=NET, location_id="218020", product_id="614504", as_of=DATE, median=45, stock=102)
    generator, benchmark = tn.generator_proxy_need(demand=9, **common), tn.benchmark_proxy_need(outbound=9, **common)
    assert (generator.target_need_value, benchmark.target_need_value) == (63, 6)
    assert oe.need_formula_gap(45, 102, 9) == generator.target_need_value - benchmark.target_need_value
    verdict = tn.need_comparability(generator, benchmark)
    assert verdict["status"] == oe.NOT_COMPARABLE and {"FORMULA_DIFFERS", "QUANTITY_UNIT_UNKNOWN"} <= set(verdict["reason_codes"])


def test_e_verified_operator_need_is_the_strict_cap():
    record, intake = _evaluate(inputs=_full_inputs())
    assert intake.scope == tn.ACTUAL_OPERATION and all(r["row_state"] == tn.ROW_ACCEPTED for r in intake.rows)
    assert record["target_demand_accepted"] and record["need_formula_id"] == tn.OPERATOR_CONFIRMED_NEED
    assert record["target_need_value"] == 40 and record["contract_strict"] == oe.STRICT_ELIGIBLE
    assert record["need_formula_version"] == "v1"


def test_f_proxy_target_need_never_verifies():
    record, intake = _evaluate()
    assert intake.scope == tn.NO_EVIDENCE and all(v["status"] == tn.MISSING_EVIDENCE for v in intake.sheets.values())
    assert record["need_formula_id"] == tn.BENCHMARK_PROXY and not record["target_demand_accepted"]
    assert "TARGET_NEED_PROXY_OUTBOUND" in _reasons(record) and record["missing_actual_target_need"]
    assert record["contract_strict"] == oe.INSUFFICIENT_OPERATIONAL_EVIDENCE and record["contract_validated_qty"] is None


def test_g_actual_on_hand_alone_is_not_transferable():
    availability = tn.source_availability(network_id=NET, location_id="S", product_id="P1", as_of=DATE,
                                          dataset_on_hand=100.0, dataset_on_hand_evidence=oe.DIRECT_REAL,
                                          surplus_proxy=40.0, surplus_proxy_basis="median rule")
    assert availability["validation_status"] == tn.ON_HAND_ONLY and availability["transferable_qty"] is None
    assert {"ONLY_ON_HAND_STOCK_OBSERVED", "AVAILABLE_FOR_TRANSFER_NOT_OBSERVED", "RESERVED_QTY_UNKNOWN"} <= set(
        availability["reason_codes"].split("|"))
    inputs = _full_inputs(**{tn.INVENTORY_SHEET: [_inventory(on_hand_qty=100), _unit()]})
    record, _ = _evaluate(inputs=inputs)
    assert not record["source_supply_accepted"] and record["missing_actual_surplus"]
    assert record["source_availability_status"] == tn.ON_HAND_ONLY and "ONLY_ON_HAND_STOCK_OBSERVED" in _reasons(record)


def test_h_available_for_transfer_and_operator_cap_are_upper_bounds_never_summed():
    facts = {"ON_HAND": (100.0, "BOX", oe.DIRECT_REAL, "r1"), "RESERVED": (30.0, "BOX", oe.DIRECT_REAL, "r1"),
             "AVAILABLE_FOR_TRANSFER": (40.0, "BOX", oe.DIRECT_REAL, "r1")}
    record = tn.source_availability(network_id=NET, location_id="S", product_id="P1", as_of=DATE, operator=facts)
    assert record["transferable_qty"] == 40 and record["unreserved_on_hand_qty"] == 70
    assert record["transferable_kind"] == tn.OBSERVED_AVAILABLE_FOR_TRANSFER and record["validation_status"] == tn.VERIFIED
    capped = tn.source_availability(network_id=NET, location_id="S", product_id="P1", as_of=DATE,
                                    operator={**facts, "TRANSFER_CAP": (35.0, "BOX", oe.VERIFIED_OPERATOR_INPUT, "r2")})
    assert capped["transferable_qty"] == 35 and capped["transferable_kind"] == tn.OPERATOR_APPROVED_TRANSFER_CAP
    lane_record, _ = _evaluate(inputs=_full_inputs())
    assert lane_record["source_supply_accepted"] and lane_record["source_surplus_value"] == 40
    assert lane_record["source_availability_kind"] == tn.OBSERVED_AVAILABLE_FOR_TRANSFER


def test_i_source_surplus_proxy_stays_proxy():
    record = tn.source_availability(network_id=NET, location_id="S", product_id="P1", as_of=DATE, surplus_proxy=40.0,
                                    surplus_proxy_basis="median rule")
    assert record["validation_status"] == tn.PROXY_ONLY and record["transferable_qty"] is None
    lane_record, _ = _evaluate()
    assert lane_record["source_availability_kind"] == tn.SOURCE_SURPLUS_PROXY and not lane_record["source_supply_accepted"]


def test_j_operator_approved_route_is_route_evidence():
    record, _ = _evaluate(inputs={tn.ROUTE_SHEET: [_route()]})
    assert record["route_accepted"] and record["route_class"] == oe.ROUTE_B and not record["missing_route_approval"]


def test_k_distance_only_route_is_never_approval():
    record, _ = _evaluate()
    assert not record["route_accepted"] and record["route_class"] == oe.ROUTE_C and record["missing_route_approval"]
    estimate, intake = _evaluate(inputs={tn.ROUTE_SHEET: [_route(**_trust(tn.INPUT_CONFIRMED))]})
    assert not estimate["route_accepted"] and "VERIFICATION_LEVEL_INSUFFICIENT_FOR_ROUTE_ALLOWED" in _issues(intake)


def test_l_prohibited_route_is_ineligible():
    record, _ = _evaluate(inputs={tn.ROUTE_SHEET: [_route(allowed="false", **_trust(tn.INPUT_CONFIRMED))]})
    assert record["contract_eligibility"] == oe.INELIGIBLE and oe.ROUTE_PROHIBITED in _reasons(record)


def test_m_operator_handling_allowed_for_a_target_without_stock():
    lane = _lane(target_has_inventory_row=False, inventory_target_matched=False, target_stock=None, target_need_7d=None)
    handling = _inventory(location="T", handling_allowed="true", evidence_class=oe.VERIFIED_OPERATOR_INPUT,
                          **_trust(tn.INPUT_CONFIRMED))
    record, _ = _evaluate(lane, {tn.INVENTORY_SHEET: [handling]})
    assert record["target_handling_accepted"] and record["handling_evidence"].startswith("OPERATOR_DECLARED_RECEIVABLE")


def test_n_no_handling_evidence_and_prohibition():
    lane = _lane(target_has_inventory_row=False, inventory_target_matched=False, target_stock=None, target_need_7d=None)
    record, _ = _evaluate(lane)
    assert not record["target_handling_accepted"] and "TARGET_HANDLING_NOT_EVIDENCED" in _reasons(record)
    assert record["missing_target_handling"]
    # zero stock with a row is not "cannot receive"; a declared prohibition is
    zero, _ = _evaluate(_lane(target_stock=0.0))
    assert zero["target_handling_accepted"] and zero["handling_evidence"].startswith("CURRENT_ROW_ZERO_STOCK")
    banned = _inventory(location="T", handling_allowed="false", evidence_class=oe.VERIFIED_OPERATOR_INPUT,
                        **_trust(tn.INPUT_CONFIRMED))
    blocked, _ = _evaluate(inputs=_full_inputs(**{tn.INVENTORY_SHEET: [*_full_inputs()[tn.INVENTORY_SHEET], banned]}))
    assert blocked["contract_eligibility"] == oe.INELIGIBLE and "TARGET_HANDLING_PROHIBITED" in _reasons(blocked)
    assert blocked["contract_strict"] == oe.INELIGIBLE


def test_o_confirmed_dataset_unit_is_accepted():
    record, _ = _evaluate(inputs={tn.INVENTORY_SHEET: [_unit()]})
    assert record["quantity_unit_accepted"] and record["contract_quantity_unit"] == "BOX"


def test_p_unknown_unit_is_never_estimated():
    record, _ = _evaluate()
    assert not record["quantity_unit_accepted"] and "QUANTITY_UNIT_UNKNOWN" in _reasons(record)
    assert record["contract_quantity_unit"] is None and record["missing_quantity_unit"]
    _, intake = _evaluate(inputs={tn.DEMAND_SHEET: [_need(quantity_unit=None)]})
    assert "QUANTITY_UNIT_MISSING" in _issues(intake) and intake.rows[0]["row_state"] == tn.ROW_REJECTED
    _, intake = _evaluate(inputs={tn.INVENTORY_SHEET: [_inventory(on_hand_qty=5, quantity_unit="UNKNOWN")]})
    assert "QUANTITY_UNIT_MISSING" in _issues(intake)


def test_q_actual_route_capacity_bounds_the_validated_quantity():
    record, _ = _evaluate(inputs=_full_inputs(**{tn.ROUTE_SHEET: [_route(route_capacity_qty=15)]}))
    assert record["capacity_accepted"] and record["capacity_value"] == 15 and record["contract_validated_qty"] == 15
    assert "PARTIAL_QTY_ONLY" in _reasons(record)


def test_r_missing_route_capacity_leaves_the_lane_conditional():
    record, _ = _evaluate(inputs=_full_inputs(**{tn.ROUTE_SHEET: [_route(route_capacity_qty=None, quantity_unit=None)]}))
    assert record["operational_eligibility"] == oe.VERIFIED_CONDITIONAL and record["contract_strict"] == oe.STRICT_ELIGIBLE
    assert record["missing_capacity"] and record["missing_required_count"] == 0


def test_s_contract_cost_is_actual_spend_evidence():
    record, _ = _evaluate(inputs=_full_inputs())
    assert record["transport_cost_accepted"] and record["cost_evidence"].startswith("CONTRACT_COST") and record["transport_cost"] == 90000


def test_t_calculated_tariff_is_derived_and_proxy_vehicle_is_not_accepted():
    derived, _ = _evaluate(inputs=_full_inputs(**{tn.ROUTE_SHEET: [_route(cost_basis=None, cost_value=None)]}))
    assert derived["transport_cost_accepted"] and "COST_CALCULATED_NOT_ACTUAL_SPEND" in (derived["conditional_reasons"] or "")
    proxy, _ = _evaluate(_lane(cost_provenance="PROXY"), _full_inputs(**{tn.ROUTE_SHEET: [_route(cost_basis=None, cost_value=None)]}))
    assert not proxy["transport_cost_accepted"] and proxy["missing_cost_basis"] and proxy["contract_strict"] != oe.STRICT_ELIGIBLE


def test_u_operator_verified_input_is_accepted_with_declared_authority_only():
    _, intake = _evaluate(inputs=_full_inputs())
    assert {row["verification"] for row in intake.rows} == {tn.OPERATOR_VERIFIED}
    assert {row["authority_check"] for row in intake.rows} == {"NOT_PERFORMED_BY_PROGRAM"}
    assert intake.sheets[tn.DEMAND_SHEET]["status"] == tn.VALIDATED_INPUT and intake.sheets[tn.DEMAND_SHEET]["accepted_actual_rows"] == 1


@pytest.mark.parametrize("missing", ["verified_by", "verified_at", "verification_level", "record_origin", "verification_status"])
def test_v_operator_unverified_input_never_verifies(missing):
    record, intake = _evaluate(inputs=_full_inputs(**{tn.DEMAND_SHEET: [_need(**{missing: None})]}))
    row = next(r for r in intake.rows if r["sheet"] == tn.DEMAND_SHEET)
    assert row["row_state"] == tn.ROW_ESTIMATE and row["verification"] == tn.OPERATOR_UNVERIFIED
    assert not record["target_demand_accepted"] and record["contract_strict"] != oe.STRICT_ELIGIBLE
    assert "DECLARED_NEED_ESTIMATE_ONLY" in _reasons(record)


def test_w_sample_input_is_never_evidence():
    sample = {tn.ROUTE_SHEET: pd.read_csv(REPO / "samples" / "operational_route_evidence_TEMPLATE_SAMPLE.csv", dtype=str),
              tn.DEMAND_SHEET: pd.read_csv(REPO / "samples" / "operational_demand_evidence_TEMPLATE_SAMPLE.csv", dtype=str),
              tn.INVENTORY_SHEET: pd.read_csv(REPO / "samples" / "operational_inventory_evidence_TEMPLATE_SAMPLE.csv", dtype=str)}
    intake = _intake(sample)
    assert intake.rows and {row["verification"] for row in intake.rows} == {tn.SAMPLE_NOT_USABLE}
    assert {row["row_state"] for row in intake.rows} == {tn.ROW_REJECTED} and intake.scope == tn.NO_EVIDENCE
    # a SAMPLE id wins over a declared ACTUAL_OPERATION origin
    _, forged = _evaluate(inputs={tn.DEMAND_SHEET: [_need(location="SAMPLE_CENTER_B")]})
    assert forged.rows[0]["verification"] == tn.SAMPLE_NOT_USABLE
    _, mislabelled = _evaluate(inputs={tn.DEMAND_SHEET: [_need(data_source="TEST_PLAN")]})
    assert "ORIGIN_SOURCE_MISMATCH" in _issues(mislabelled) and mislabelled.rows[0]["row_state"] == tn.ROW_REJECTED


def test_x_expired_or_future_input_is_not_current_evidence():
    expired, _ = _evaluate(inputs=_full_inputs(**{tn.DEMAND_SHEET: [_need(observation_start="2026-07-10", observation_end="2026-07-10")]}))
    assert not expired["target_demand_accepted"] and expired["expired_evidence"] and "EVIDENCE_EXPIRED" in _reasons(expired)
    future, _ = _evaluate(inputs=_full_inputs(**{tn.INVENTORY_SHEET: [
        _inventory(effective_date="2026-07-20", available_for_transfer_qty=40, on_hand_qty=100), _unit()]}))
    assert not future["source_supply_accepted"] and "EVIDENCE_NOT_YET_VALID" in _reasons(future)
    until, _ = _evaluate(inputs=_full_inputs(**{tn.DEMAND_SHEET: [_need(observation_end="2026-07-31", valid_until="2026-07-15")]}))
    assert not until["target_demand_accepted"]


def test_y_conflicting_inputs_are_conflict_and_no_value_is_chosen():
    inputs = _full_inputs(**{tn.DEMAND_SHEET: [_need(40), _need(55, data_source="plan-2")]})
    record, intake = _evaluate(inputs=inputs)
    assert [c["code"] for c in intake.conflicts] == ["TARGET_NEED_CONFLICT"]
    assert record["contract_eligibility"] == tn.CONFLICT and record["conflicting_evidence"]
    assert record["contract_strict"] == oe.INELIGIBLE and not record["target_demand_accepted"]
    routes = _full_inputs(**{tn.ROUTE_SHEET: [_route(), _route(allowed="false")]})
    route_record, route_intake = _evaluate(inputs=routes)
    assert "ROUTE_ALLOWED_PROHIBITED_CONFLICT" in {c["code"] for c in route_intake.conflicts}
    assert route_record["contract_eligibility"] == tn.CONFLICT
    units = _full_inputs(**{tn.INVENTORY_SHEET: [_inventory(available_for_transfer_qty=40), _unit(), _unit(unit="KG")]})
    assert "DATASET_UNIT_CONFLICT" in {c["code"] for c in _intake(units).conflicts}
    over = _full_inputs(**{tn.INVENTORY_SHEET: [_inventory(on_hand_qty=30, available_for_transfer_qty=40), _unit()]})
    over_record, _ = _evaluate(inputs=over)
    assert over_record["contract_eligibility"] == tn.CONFLICT and "AVAILABLE_EXCEEDS_ON_HAND" in over_record["contract_reasons"]
    # on-hand and available are different fields: no conflict between them when consistent
    consistent, _ = _evaluate(inputs=_full_inputs())
    assert consistent["contract_eligibility"] != tn.CONFLICT


def test_z_another_dataset_or_unknown_key_is_rejected():
    rows = [_need(network_id="kamis_public_price"), _need(dataset_id="favorita"), _need(location="X9"),
            _need(product_id="P9"), {**_need(), "network_id": None}]
    intake = _intake({tn.DEMAND_SHEET: rows})
    assert {"CROSS_NETWORK_EVIDENCE_REJECTED", "CROSS_DATASET_EVIDENCE_REJECTED", "UNKNOWN_LOCATION_ID",
            "UNKNOWN_PRODUCT_ID", "MISSING_REQUIRED_FIELD"} <= _issues(intake)
    assert {row["row_state"] for row in intake.rows} == {tn.ROW_REJECTED}
    assert intake.sheets[tn.DEMAND_SHEET]["status"] == tn.VALIDATION_ERROR
    # a zero-padded Suhyup center code maps by the verified rule only; names are never matched
    suhyup = tn.KeyUniverse(network_id=NET, locations=frozenset({"156"}), products=frozenset({"P1"}),
                            location_namespace="suhyup_canonical_location_id")
    padded = tn.intake_evidence({tn.DEMAND_SHEET: [_need(location="000156")]}, suhyup)
    assert padded.rows[0]["location_id"] == "156" and padded.rows[0]["row_state"] == tn.ROW_ACCEPTED
    named = tn.intake_evidence({tn.DEMAND_SHEET: [_need(location="부산공판장")]}, suhyup)
    assert named.rows[0]["row_state"] == tn.ROW_REJECTED


def test_aa_observed_zero_is_a_value():
    need = tn.benchmark_proxy_need(network_id=NET, location_id="T", product_id="P1", as_of=DATE, median=10, stock=0, outbound=0)
    assert need.current_stock_status == tn.OBSERVED_ZERO and need.target_need_value == 10
    zero = tn.source_availability(network_id=NET, location_id="S", product_id="P1", as_of=DATE,
                                  operator={"AVAILABLE_FOR_TRANSFER": (0.0, "BOX", oe.DIRECT_REAL, "r")})
    assert zero["transferable_qty"] == 0 and zero["validation_status"] == tn.VERIFIED
    record, _ = _evaluate(inputs=_full_inputs(**{tn.INVENTORY_SHEET: [_inventory(available_for_transfer_qty=0), _unit()]}))
    assert record["contract_eligibility"] == oe.INELIGIBLE and "VERIFIED_SOURCE_SUPPLY_BELOW_ONE_UNIT" in _reasons(record)


def test_ab_missing_observation_is_null_never_zero():
    need = tn.generator_proxy_need(network_id=NET, location_id="T", product_id="P1", as_of=DATE, median=45, stock=None, demand=None)
    assert need.target_need_value is None and need.validation_status == tn.NOT_OBSERVED
    assert need.substitute_value == 0 and "GENERATOR_SUBSTITUTES_MEDIAN_STOCK_AND_ZERO_DEMAND" in need.reason_codes
    bench = tn.benchmark_proxy_need(network_id=NET, location_id="T", product_id="P1", as_of=DATE, median=45, stock=None, outbound=None)
    assert bench.target_need_value is None and bench.as_dict()["current_stock_status"] == tn.NOT_OBSERVED
    intake = _intake({tn.INVENTORY_SHEET: [_inventory(on_hand_qty="", available_for_transfer_qty="12")]})
    facts = tn.day_evidence(intake, DATE).operator_facts[("S", "P1")]
    assert "ON_HAND" not in facts and facts["AVAILABLE_FOR_TRANSFER"][0] == 12
    assert tn.pipeline_need(None, network_id=NET, location_id="T", product_id="P1", as_of=DATE).target_need_value is None


def test_ac_several_stores_and_products_keep_their_own_evidence():
    lanes = [_lane(), _lane(lane_key="P1/S/U/DIRECT/", route_id="R2", target_id="U", target_stock=5.0),
             _lane(lane_key="P2/V/T/DIRECT/", route_id="R3", product_id="P2", source_id="V")]
    inputs = {tn.ROUTE_SHEET: [_route(), _route(target_id="U")],
              tn.DEMAND_SHEET: [_need(20), _need(20, location="U")],
              tn.INVENTORY_SHEET: [_inventory(available_for_transfer_qty=25), _unit()]}
    intake = _intake(inputs)
    evidence = tn.day_evidence(intake, DATE)
    context = tn.oe_context(evidence, {"network_id": NET})
    results = [tn.evaluate_lane_contract(lane, evidence, context) for lane in lanes]
    assert [r["contract_strict"] for r in results] == [oe.STRICT_ELIGIBLE, oe.STRICT_ELIGIBLE, oe.INSUFFICIENT_OPERATIONAL_EVIDENCE]
    assert results[2]["missing_actual_surplus"] and results[2]["missing_quantity_unit"]
    bridge = tn.t1_strict_bridge(lanes, results, scope=intake.scope)
    # two strict lanes share source S / P1 (25 transferable): T1 takes the first (qty 20) and rejects the second
    assert bridge["contract_strict_candidates"] == 2 and bridge["selected"] == 1
    assert bridge["selection_statuses"].get(sf.REJECTED_SOURCE_CAP) == 1
    assert bridge["selection_statuses"]["EXCLUDED_NOT_CONTRACT_STRICT"] == 1 and bridge["strict_actual_claim"]


def test_ad_same_input_is_deterministic_and_order_independent():
    inputs = _full_inputs(**{tn.DEMAND_SHEET: [_need(40), _need(55, data_source="plan-2"), _need(12, location="U")]})
    first, intake_a = _evaluate(inputs=inputs)
    second, intake_b = _evaluate(inputs=copy.deepcopy(inputs))
    assert first == second and intake_a.rows == intake_b.rows and intake_a.conflicts == intake_b.conflicts
    shuffled = {key: list(reversed(rows)) for key, rows in inputs.items()}
    third, intake_c = _evaluate(inputs=shuffled)
    assert {c["code"] for c in intake_c.conflicts} == {c["code"] for c in intake_a.conflicts}
    assert (third["contract_eligibility"], third["contract_strict"]) == (first["contract_eligibility"], first["contract_strict"])


# --------------------------------------------------------------------------------------------- TEST origin, meanings, bridge


def test_test_origin_is_linkage_only():
    test_inputs = {key: [{**row, "record_origin": tn.TEST, "evidence_source": f"TEST_{i}", "data_source": f"TEST_{i}"}
                         for i, row in enumerate(rows)] for key, rows in _full_inputs().items()}
    record, intake = _evaluate(inputs=test_inputs)
    assert intake.scope == tn.NO_EVIDENCE and {r["verification"] for r in intake.rows} == {tn.TEST_ONLY}
    assert record["contract_strict"] == oe.INSUFFICIENT_OPERATIONAL_EVIDENCE
    linked, linked_intake = _evaluate(inputs=test_inputs, allow_test=True)
    assert linked_intake.scope == tn.TEST_LINKAGE and linked["contract_strict"] == "STRICT_ELIGIBLE_TEST_ONLY"
    bridge = tn.t1_strict_bridge([_lane()], [linked], scope=linked_intake.scope)
    assert bridge["selected"] == 1 and not bridge["strict_actual_claim"] and bridge["label"].startswith("TEST_LINKAGE_ONLY")
    # one TEST row taints a bundle that also holds actual rows
    mixed = {**_full_inputs(), tn.DEMAND_SHEET: test_inputs[tn.DEMAND_SHEET]}
    assert _intake(mixed, allow_test=True).scope == tn.TEST_LINKAGE


def test_receivable_capacity_is_its_own_field_and_binds():
    receivable = _need(12, quantity_kind=oe.DECLARED_RECEIVABLE, quantity_meaning=tn.RECEIVABLE_CAPACITY,
                       **_trust(tn.OPERATIONALLY_APPROVED))
    record, intake = _evaluate(inputs=_full_inputs(**{tn.DEMAND_SHEET: [_need(40), receivable]}))
    assert record["target_need_value"] == 40 and record["receivable_capacity_qty"] == 12
    assert record["contract_validated_qty"] == 12 and "RECEIVABLE_CAPACITY_BINDING" in _reasons(record)
    caps, _ = tn.t1_strict_caps([_lane()], [record])
    target = caps[(sf.TARGET_NEED, ("T", "P1"))]
    assert target.value == 12 and "RECEIVABLE_CAPACITY" in target.basis and target.provenance == "USER_INPUT"
    # a capacity filed as a need is rejected, never stored as target_need
    _, wrong = _evaluate(inputs={tn.DEMAND_SHEET: [_need(40, quantity_meaning=tn.RECEIVABLE_CAPACITY)]})
    assert "QUANTITY_MEANING_KIND_MISMATCH" in _issues(wrong) and wrong.rows[0]["row_state"] == tn.ROW_REJECTED


def test_order_preference_and_proxy_kinds_never_become_need():
    rows = [_need(30, quantity_kind=oe.ORDER, evidence_class=oe.DIRECT_REAL),
            _need(30, quantity_kind=tn.DECLARED_PREFERENCE, location="U"),
            _need(30, quantity_kind=oe.OUTBOUND_SHIPMENT, location="V")]
    record, intake = _evaluate(inputs={tn.DEMAND_SHEET: rows})
    assert {"PROXY_LABELLED_AS_ACTUAL", "PREFERENCE_INFORMATIONAL_ONLY", "PROXY_KIND_NEVER_ACTUAL_DEMAND"} <= _issues(intake)
    assert intake.preferences and all(q["evidence"] == oe.PROXY for q in intake.quantities)
    assert not record["target_demand_accepted"] and "OPERATOR_DEMAND_IS_PROXY_KIND" in _reasons(record)


def test_bridge_without_evidence_keeps_insufficient_cap_data():
    records = [_evaluate(_lane(lane_key=f"P1/S/T{i}/DIRECT/", route_id=f"R{i}", target_id="T"))[0] for i in range(2)]
    bridge = tn.t1_strict_bridge([_lane(route_id="R0"), _lane(route_id="R1", target_id="U")], records, scope=tn.NO_EVIDENCE)
    assert bridge["plan_status"] == sf.INSUFFICIENT_CAP_DATA and bridge["selected"] == 0 and not bridge["strict_actual_claim"]


def test_bridge_excludes_lanes_that_are_not_contract_strict():
    """A lane without route approval must not borrow the caps of a verified lane with the same source and target keys."""
    full, _ = _evaluate(inputs=_full_inputs())
    other = {**full, "contract_strict": oe.INSUFFICIENT_OPERATIONAL_EVIDENCE}
    lanes = [_lane(), _lane(route_id="R9", route_type="VIA_DC", dc_id="D1")]
    bridge = tn.t1_strict_bridge(lanes, [full, other], scope=tn.ACTUAL_OPERATION)
    assert bridge["t1_input_candidates"] == 1 and bridge["selection_statuses"]["EXCLUDED_NOT_CONTRACT_STRICT"] == 1


# --------------------------------------------------------------------------------------------- registry & planner


def test_formula_registry_documents_every_formula():
    for formula, entry in tn.FORMULAS.items():
        for key in ("version", "expression", "code", "inputs", "unit", "period", "intended_use", "purpose", "limitations"):
            assert entry.get(key) not in (None, ""), (formula, key)
    assert tn.FORMULAS[tn.DERIVED_FROM_OBSERVED_DEMAND]["active"] is False
    assert tn.P1B_ALIASES == {oe.GENERATOR_FORMULA: tn.GENERATOR_PROXY, oe.BENCHMARK_FORMULA: tn.BENCHMARK_PROXY,
                              oe.OPERATOR_DECLARED_FORMULA: tn.OPERATOR_CONFIRMED_NEED,
                              oe.ACTUAL_SALES_FORMULA: tn.DERIVED_FROM_OBSERVED_DEMAND}
    assert tn.FORMULAS[tn.GENERATOR_PROXY]["intended_use"] == tn.GENERATOR_RANKING


def test_constraint_signatures_block_cross_formula_optimality_claims():
    generator = tn.constraint_signature(tn.GENERATOR_PROXY, cap_provenance=["PROXY"], window=("2026-07-01", "2026-07-31"))
    benchmark = tn.constraint_signature(tn.BENCHMARK_PROXY, cap_provenance=["PROXY"], window=("2026-07-01", "2026-07-31"))
    again = tn.constraint_signature(tn.BENCHMARK_PROXY, cap_provenance=["PROXY"], window=("2026-07-01", "2026-07-31"))
    assert generator["signature"] != benchmark["signature"] == again["signature"]
    verdict = tn.signatures_comparable(generator, benchmark)
    assert verdict["status"] == oe.NOT_COMPARABLE and not verdict["optimality_comparison_allowed"]
    other_window = tn.constraint_signature(tn.BENCHMARK_PROXY, cap_provenance=["PROXY"], window=("2026-07-16", "2026-07-16"))
    assert "WINDOW_DIFFERS" in tn.signatures_comparable(benchmark, other_window)["reason_codes"]


def test_pipeline_need_names_the_in_app_formula():
    explicit = tn.pipeline_need({"stock_qty": 5, "shortage_qty": 12}, network_id=NET, location_id="T", product_id="P1", as_of=DATE)
    assert explicit.formula_id == tn.UPLOADED_SHORTAGE_INPUT and explicit.evidence == oe.USER_ESTIMATE and not explicit.accepted
    computed = tn.pipeline_need({"stock_qty": 5, "avg_daily_sales": 3}, network_id=NET, location_id="T", product_id="P1", as_of=DATE)
    assert computed.formula_id == tn.PIPELINE_SHORTAGE_7D and computed.target_need_value == 16
    proxy = tn.pipeline_need({"stock_qty": 5, "sales_qty": 3, "sales_qty_semantics": "demand_proxy_not_retail_sales"},
                             network_id=NET, location_id="T", product_id="P1", as_of=DATE)
    assert proxy.evidence == oe.PROXY and proxy.validation_status == tn.PROXY_STATUS


def test_missing_report_and_request_priority():
    records = pd.DataFrame([_evaluate(_lane(lane_key=f"k{i}", target_id=t, product_id=p))[0]
                            for i, (t, p) in enumerate([("T", "P1"), ("U", "P1"), ("T", "P2")])])
    scope = tn.missing_by_scope(records)
    product = scope[(scope["scope"] == "product") & (scope["key"] == "P1")].iloc[0]
    assert product["candidates"] == 2 and product["missing_route_approval"] == 2
    priority = tn.request_priority(records)
    assert list(priority["priority"]) == list(range(1, len(priority) + 1))
    tiers = [tn.TIER_ORDER[t] for t in priority["tier"]]
    assert tiers == sorted(tiers) and priority.iloc[-1]["field"] == "missing_cost_basis" or "missing_cost_basis" not in set(priority["field"])
    unit = priority.set_index("field").loc["missing_quantity_unit"]
    assert unit["rows_to_collect"] == 2 and unit["reuse_candidates_per_row"] == 1.5
    assert int(priority["strict_unlocked_by_this"].sum()) == int(priority["strict_upper_bound_after"].iloc[-1])


def test_sheet_statuses():
    intake = _intake({tn.ROUTE_SHEET: pd.DataFrame({"source_id": ["S"]}), tn.DEMAND_SHEET: pd.DataFrame(),
                      tn.INVENTORY_SHEET: pd.DataFrame({tn.LOAD_ERROR_COLUMN: ["BadZipFile: broken"]})})
    assert intake.sheets[tn.ROUTE_SHEET]["status"] == tn.VALIDATION_ERROR and "missing required columns" in intake.sheets[tn.ROUTE_SHEET]["detail"]
    assert intake.sheets[tn.DEMAND_SHEET]["status"] == tn.NO_USABLE_EVIDENCE
    assert intake.sheets[tn.INVENTORY_SHEET]["status"] == tn.VALIDATION_ERROR
    assert _intake({tn.DEMAND_SHEET: "not a table"}).sheets[tn.DEMAND_SHEET]["status"] == tn.VALIDATION_ERROR

    def half_then_fail(intake, frame):
        intake.quantities.append({"row": "partial"})
        intake.rows.append({"row_state": tn.ROW_ACCEPTED})
        raise RuntimeError("broken mid-sheet")

    with mock.patch.object(tn, "_intake_demand", side_effect=half_then_fail):
        broken = _intake({tn.DEMAND_SHEET: [_need()], tn.ROUTE_SHEET: [_route()]})
    assert broken.sheets[tn.DEMAND_SHEET]["status"] == tn.VALIDATION_ERROR and broken.quantities == []
    assert broken.sheets[tn.ROUTE_SHEET]["status"] == tn.VALIDATED_INPUT and broken.scope == tn.ACTUAL_OPERATION


# --------------------------------------------------------------------------------------------- workbook & pipeline


def _load(source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_excel_data(source)


def _workbook_with_sheets(sheets: dict[str, pd.DataFrame]) -> io.BytesIO:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original = pd.read_excel(NETWORK_SAMPLE, sheet_name=None)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in {**original, **sheets}.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    buffer.seek(0)
    return buffer


def test_optional_sheets_are_registered():
    for sheet in tn.EVIDENCE_SHEETS:
        assert sheet in OPTIONAL_SHEETS


def test_pipeline_field_is_parallel_fault_isolated_and_deterministic():
    data = _load(NETWORK_SAMPLE)
    with_field = build_v2_state(copy.deepcopy(data), detail_level="full")
    with mock.patch.object(analysis_pipeline, "build_operational_evidence_analysis", side_effect=RuntimeError("boom")):
        broken = build_v2_state(copy.deepcopy(data), detail_level="full")
    for key in ("top5", "summary", "connected_algorithms", "status", "seller_loss_analysis", "shared_feasibility_selection",
                "action_consistency"):
        assert with_field["pipeline_result"][key] == broken["pipeline_result"][key]
    assert with_field["recommendations"] == broken["recommendations"]
    assert broken["pipeline_result"]["operational_evidence"]["status"] == "error"
    analysis = with_field["pipeline_result"]["operational_evidence"]
    again = build_v2_state(copy.deepcopy(data), detail_level="full")["pipeline_result"]["operational_evidence"]
    assert again == analysis and analysis["status"] == "parallel_only" and analysis["production_action_applied"] is False
    assert {v["status"] for v in analysis["sheets"].values()} == {tn.MISSING_EVIDENCE}
    assert analysis["decision_date_basis"] == "UNDECLARED" and analysis["strict_eligible_count"] == 0
    assert {r["contract_eligibility"] for r in analysis["candidates"]} == {tn.DECISION_DATE_UNDECLARED}
    assert len(analysis["candidates"]) == len(with_field["recommendations"])
    assert analysis["formula_ids_in_use"] and analysis["t1_strict_bridge"]["status"] == "NOT_RUN"


def test_workbook_sheets_feed_the_contract_without_changing_production():
    data = _load(NETWORK_SAMPLE)
    base = build_v2_state(copy.deepcopy(data), detail_level="core")
    first = base["pipeline_result"]["operational_evidence"]["candidates"][0]
    source, target, product, qty = first["source_id"], first["target_id"], first["product_id"], first["requested_qty"]
    trust = {"record_origin": tn.ACTUAL_OPERATION, "verification_status": "VERIFIED", "verified_by": "ops-lead",
             "verified_at": "2026-07-15"}
    sheets = {
        "config": pd.concat([pd.read_excel(NETWORK_SAMPLE, sheet_name="config"),
                             pd.DataFrame([{"key": "as_of_date", "value": DATE}])], ignore_index=True),
        tn.ROUTE_SHEET: pd.DataFrame([{"source_id": source, "target_id": target, "product_id": product, "allowed": "true",
                                       "effective_from": "2026-07-01", "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
                                       "evidence_source": "contract-1", "route_capacity_qty": qty * 4, "quantity_unit": "EA",
                                       "cost_basis": sf.FIXED_PER_TRIP, "cost_value": 30000,
                                       "verification_level": tn.OPERATIONALLY_APPROVED, **trust}]),
        tn.DEMAND_SHEET: pd.DataFrame([{"location_id": target, "product_id": product, "quantity_kind": oe.DECLARED_TARGET_NEED,
                                        "observation_start": DATE, "quantity": qty * 2, "quantity_unit": "EA",
                                        "evidence_class": oe.VERIFIED_OPERATOR_INPUT, "data_source": "plan-1",
                                        "verification_level": tn.INPUT_CONFIRMED, **trust}]),
        tn.INVENTORY_SHEET: pd.DataFrame([
            {"location_id": source, "product_id": product, "effective_date": DATE, "available_for_transfer_qty": qty * 2,
             "quantity_unit": "EA", "evidence_class": oe.DIRECT_REAL, "evidence_source": "wms-1",
             "verification_level": tn.OBSERVED_RECORD, **trust},
            {"location_id": source, "product_id": product, "effective_date": "2026-07-01", "valid_until": "2026-12-31",
             "quantity_unit": "EA", "dataset_unit_confirmed": "true", "evidence_class": oe.VERIFIED_OPERATOR_INPUT,
             "evidence_source": "master-1",
             "verification_level": tn.INPUT_CONFIRMED, **trust}]),
    }
    loaded = _load(_workbook_with_sheets(sheets))
    state = build_v2_state(copy.deepcopy(loaded), detail_level="core")
    analysis = state["pipeline_result"]["operational_evidence"]
    assert analysis["decision_date"] == DATE and analysis["evidence_scope"] == tn.ACTUAL_OPERATION
    assert {v["status"] for v in analysis["sheets"].values()} == {tn.VALIDATED_INPUT}
    strict = [r for r in analysis["candidates"] if r["contract_strict"] == oe.STRICT_ELIGIBLE]
    assert [(r["source_id"], r["target_id"], r["product_id"]) for r in strict] == [(source, target, product)]
    assert analysis["t1_strict_bridge"]["status"] == "RUN" and analysis["t1_strict_bridge"]["selected"] == 1
    assert analysis["strict_actual_claim"] is True
    # production result is the same with and without the evidence sheets (config row aside)
    plain = build_v2_state(copy.deepcopy({**loaded, **{k: None for k in tn.EVIDENCE_SHEETS}}), detail_level="core")
    for key in ("top5", "summary", "shared_feasibility_selection", "action_consistency", "seller_loss_analysis"):
        assert state["pipeline_result"][key] == plain["pipeline_result"][key]
    assert state["recommendations"] == plain["recommendations"]


def test_bad_evidence_sheet_never_stops_the_workbook():
    loaded = _load(_workbook_with_sheets({tn.DEMAND_SHEET: pd.DataFrame({"wrong": [1, 2]}),
                                          tn.ROUTE_SHEET: pd.DataFrame({"note": ["SAMPLE: template"], "source_id": ["SAMPLE_A"],
                                                                        "target_id": ["SAMPLE_B"], "allowed": ["true"],
                                                                        "effective_from": ["2026-07-01"],
                                                                        "evidence_class": ["VERIFIED_OPERATOR_INPUT"],
                                                                        "evidence_source": ["SAMPLE_CONTRACT"]})}))
    state = build_v2_state(copy.deepcopy(loaded), detail_level="core")
    plain = build_v2_state(_load(NETWORK_SAMPLE), detail_level="core")
    analysis = state["pipeline_result"]["operational_evidence"]
    assert analysis["sheets"][tn.DEMAND_SHEET]["status"] == tn.VALIDATION_ERROR
    assert analysis["sheets"][tn.ROUTE_SHEET]["status"] == tn.NO_USABLE_EVIDENCE
    assert analysis["sheets"][tn.INVENTORY_SHEET]["status"] == tn.MISSING_EVIDENCE
    assert state["pipeline_result"]["top5"] == plain["pipeline_result"]["top5"]
    assert state["pipeline_result"]["summary"] == plain["pipeline_result"]["summary"]


# --------------------------------------------------------------------------------------------- validation helpers


def test_cli_writes_only_its_own_bundle():
    assert tnv.OUTPUT_FOLDER == "_TARGET_NEED_CONTRACT_VALIDATION" and tnv.OUTPUT_FOLDER not in tnv.PROTECTED_BUNDLES
    assert "_OPERATIONAL_EVIDENCE_VALIDATION" in tnv.PROTECTED_BUNDLES
    for name in ("target_need_contract.json", "target_need_formula_comparison.csv", "target_need_by_day.csv",
                 "operational_evidence_input_validation.csv", "operational_evidence_missing_fields.csv",
                 "operational_evidence_priority.csv", "operational_evidence_e2e.csv", "operational_need_contract_summary.json"):
        assert name in tnv.OUTPUT_FILES
    assert set(tnv.ALLOWED_MODIFIED) == {"services/data_loader.py", "services/analysis_pipeline.py"}


def test_test_operator_input_is_labelled_test():
    lanes = [{**_lane(lane_key=f"k{i}", source_id=f"S{i}", target_id=f"T{i}", product_id=f"P{i}"), "source_stock": 50.0,
              "recommended_qty": 10} for i in range(len(tnv.TEST_ROLES))]
    inputs = tnv.test_operator_input(lanes, {})
    rows = [row for frame in inputs.values() for row in frame]
    assert rows and all(row.get("record_origin") == tn.TEST or str(row.get("note", "")).startswith("SAMPLE") for row in rows)
    assert all(str(row.get("evidence_source") or row.get("data_source")).startswith(("TEST_", "SAMPLE_")) for row in rows)


def test_need_records_keep_null_and_formulas_apart():
    need = pd.DataFrame([{"date": DATE, "center_id": "T", "product_id": "P1", "generator_row_present": False,
                          "generator_median": 45.0, "generator_stock": 45.0, "generator_demand": 0.0,
                          "benchmark_median": 45.0, "benchmark_stock": 102.0, "benchmark_demand": 9.0,
                          "excluded_duplicate_state_pair": True, "lanes_in_legacy_20": 1, "lanes_in_all_valid": 2}])
    records = tnv.need_records(need)
    generator, benchmark = records
    assert generator["formula_id"] == tn.GENERATOR_PROXY and generator["target_need_value"] is None
    assert generator["substitute_value"] == 0 and benchmark["target_need_value"] == 6
    assert {r["comparable_with_other_formula"] for r in records} == {oe.NOT_COMPARABLE}
