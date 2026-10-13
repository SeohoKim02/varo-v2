"""P1 / T4 sales semantics, turnover and inventory risk metric integrity: controlled scenarios A-AF, the loader alias
replay, alias-copy detection, legacy diagnostics, ABC / disposal / safety stock / matching audits, the action
dependency probe, the parallel pipeline field and the streamed canonical aggregation.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from services import analysis_pipeline
from services import inventory_metric_integrity as imi
from services import inventory_metric_integrity_validation as imv
from services.analysis_pipeline import PipelineResult, _run_inventory_analysis, _Runner, build_v2_state
from services.data_loader import load_excel_data, normalize_loaded_data
from services.legacy_adapters.data_adapter import prepare_legacy_data
from services.legacy_adapters.loader import load_legacy_module

REPO = Path(__file__).resolve().parents[1]
NETWORK_SAMPLE = REPO / "data" / "Varo_V2_네트워크_샘플.xlsx"
D = "2026-07-01"


@pytest.fixture(autouse=True)
def legacy_mode(monkeypatch):
    monkeypatch.delenv("VARO_REAL_DATA_ROOT", raising=False)


def _records(n, qty, start="2026-07-01", **extra):
    first = pd.Timestamp(start)
    return [{"date": (first + pd.Timedelta(days=i)).date().isoformat(), "quantity": qty(i) if callable(qty) else qty, **extra}
            for i in range(n)]


def _window(records, **kw):
    params = {"window_end": "2026-07-30", "window_days": 30, "dataset_id": "T", "signal_type": imi.OBSERVED_RETAIL_SALES,
              "quantity_unit": "EA", "zero_semantics": imi.ZERO_RECORDED, "semantics_evidence": imi.SOURCE_DOCUMENTED}
    params.update(kw)
    return imi.aggregate_daily_window(records, **params)


def _inventory(snapshots, **kw):
    params = {"period_start": "2026-07-01", "period_end": "2026-07-30", "dataset_id": "T", "quantity_unit": "EA"}
    params.update(kw)
    return imi.average_inventory(snapshots, **params)


# --------------------------------------------------------------------------------------------- controlled scenarios


SCENARIOS = {row["scenario"]: row for row in imv.controlled_scenarios()}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_controlled_scenario(scenario):
    row = SCENARIOS[scenario]
    assert row["pass"], (row["expected"], row["actual"], row["reason_codes"])


def test_every_required_scenario_is_present():
    letters = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "AA", "AB", "AC", "AD", "AE", "AF"]
    assert set(letters) <= set(SCENARIOS)


def test_legacy_turnover_behaviour_is_documented_unchanged():
    legacy = load_legacy_module("turnover_analyzer").analyze_turnover(pd.DataFrame({
        "product_name": ["missing", "negative", "zero_stock"], "stock_qty": [200.0, 200.0, 0.0],
        "sales_30d": [float("nan"), -30.0, 300.0]}))
    assert legacy["turnover_grade"].tolist() == ["DEAD", "FAST", "FAST"]
    assert legacy["turnover_days"].tolist() == [999.0, -200.0, 0.0]


# --------------------------------------------------------------------------------------------- windows / NULL / zero


def test_missing_days_are_never_zero_filled():
    window = _window([r for i, r in enumerate(_records(30, 10.0)) if i % 2 == 0])
    assert window["observed_days"] == 15 and window["no_record_days"] == 15
    assert window["quantity"] == 150.0 and window["daily_rate"] == 10.0
    assert window["is_complete_observation"] is False and "PARTIAL_OBSERVATION" in window["reason_codes"]


def test_null_quantity_is_system_missing_not_zero():
    records = _records(30, lambda i: None if i < 5 else 10.0)
    window = _window(records)
    assert window["system_missing_days"] == 5 and window["zero_days"] == 0 and window["observed_days"] == 25
    assert window["quantity"] == 250.0


def test_observed_zero_is_kept_as_zero():
    window = _window(_records(30, 0.0))
    assert window["quantity"] == 0.0 and window["zero_days"] == 30 and window["gate_status"] == imi.VALID


def test_negatives_are_never_clipped():
    records = _records(30, lambda i: -5.0 if i == 0 else 10.0)
    returned = _window(records, negative_semantics=imi.RETURN_DOCUMENTED)
    unknown = _window(records)
    assert returned["quantity"] == unknown["quantity"] == 285.0
    assert returned["negative_total"] == -5.0 and returned["gate_status"] == imi.VALID_WITH_LIMITATION
    assert unknown["gate_status"] == imi.INVALID_VALUE and "NEGATIVE_UNRESOLVED" in unknown["reason_codes"]


def test_duplicate_day_is_neither_summed_nor_picked():
    window = _window(_records(30, 10.0) + [{"date": D, "quantity": 1.0}])
    assert window["quantity"] is None and window["gate_status"] == imi.INVALID_VALUE


def test_future_rows_are_excluded_and_window_after_as_of_is_invalid():
    window = _window(_records(40, 10.0), as_of="2026-07-30")
    assert window["future_records_excluded"] == 10 and window["quantity"] == 300.0
    late = _window(_records(30, 10.0), as_of="2026-07-20")
    assert late["gate_status"] == imi.INVALID_VALUE and "WINDOW_AFTER_AS_OF" in late["reason_codes"]


@pytest.mark.parametrize("field, value, code", [
    ("signal_type", imi.OBSERVED_OUTBOUND_FLOW, "SIGNAL_TYPE_MISMATCH"),
    ("dataset_id", "OTHER", "DATASET_MISMATCH"),
    ("quantity_unit", "KG", "UNIT_MISMATCH"),
])
def test_mixed_records_are_not_summed(field, value, code):
    records = _records(15, 10.0) + _records(15, 10.0, start="2026-07-16", **{field: value})
    window = _window(records)
    assert window["quantity"] is None and code in window["reason_codes"]


def test_declared_not_carried_and_not_sellable_days_are_excluded():
    records = _records(10, None, day_state=imi.NOT_SELLABLE) + _records(20, 10.0, start="2026-07-11")
    window = _window(records)
    assert window["not_sellable_days"] == 10 and window["observed_days"] == 20 and window["is_complete_observation"] is True
    carried = _window(_records(30, None, day_state=imi.NOT_CARRIED))
    assert carried["applicable"] is False and carried["quantity"] is None


def test_daily_rate_never_guesses_a_period():
    undeclared = imi.SalesSignal("T", "S", "P", 30.0, imi.OBSERVED_RETAIL_SALES, imi.PERIOD_UNDECLARED)
    assert imi.daily_rate(undeclared)["value"] is None and imi.daily_rate(undeclared)["gate"] == imi.PERIOD_MISMATCH
    month = imi.SalesSignal("T", "S", "P", 300.0, imi.OBSERVED_RETAIL_SALES, imi.WINDOW_SUM, period_days=30,
                            is_complete_observation=True, quantity_unit="EA", semantics_evidence=imi.SOURCE_DOCUMENTED)
    assert imi.daily_rate(month) == {"value": 10.0, "basis": "WINDOW_DAYS", "gate": imi.VALID, "reasons": [], "window_days": 30}
    partial = imi.SalesSignal(**{**month.__dict__, "observed_days": 20, "is_complete_observation": False, "quantity": 200.0})
    assert imi.daily_rate(partial)["value"] == 10.0 and "RATE_OVER_OBSERVED_DAYS" in imi.daily_rate(partial)["reasons"]


def test_sales_signal_rejects_unknown_enumerations():
    with pytest.raises(ValueError):
        imi.SalesSignal("T", "S", "P", 1.0, signal_type="SALES")
    with pytest.raises(ValueError):
        imi.SalesSignal("T", "S", "P", 1.0, aggregation="MONTH")


# --------------------------------------------------------------------------------------------- inventory / turnover


def test_average_inventory_uses_only_existing_snapshots():
    sparse = _inventory([{"date": "2026-07-01", "quantity": 100.0}, {"date": "2026-07-15", "quantity": 50.0},
                         {"date": "2026-07-30", "quantity": 30.0}])
    assert sparse["snapshot_count"] == 3 and sparse["value"] == 60.0 and sparse["endpoint_average"] == 65.0
    assert sparse["gate_status"] == imi.VALID_WITH_LIMITATION and "SPARSE_SNAPSHOTS" in sparse["reason_codes"]
    assert _inventory([{"date": "2026-07-30", "quantity": 30.0}])["reason_codes"] == ["END_SNAPSHOT_ONLY"]
    assert _inventory([{"date": "2026-07-01", "quantity": -1.0}, {"date": "2026-07-30", "quantity": 1.0}])["status"] == imi.INVALID_VALUE


@pytest.mark.parametrize("stock, rate, status, value", [
    (100.0, 10.0, imi.DOS_AVAILABLE, 10.0), (100.0, 0.0, imi.DOS_NO_MOVEMENT, math.inf),
    (0.0, 10.0, imi.DOS_ZERO_STOCK, 0.0), (0.0, 0.0, imi.DOS_NO_STOCK_NO_MOVEMENT, None),
    (None, 10.0, imi.DOS_STOCK_UNAVAILABLE, None), (100.0, None, imi.DOS_RATE_UNAVAILABLE, None),
    (100.0, -1.0, imi.DOS_NEGATIVE_RATE, None),
])
def test_days_of_supply_keeps_cases_apart(stock, rate, status, value):
    result = imi.days_of_supply(stock, rate)
    assert result["status"] == status and result["value"] == value


def test_classification_reuses_the_legacy_thresholds():
    module = load_legacy_module("turnover_analyzer")
    for days in (0.0, 30.0, 30.01, 60.0, 60.01, 90.0, 90.01, math.inf):
        assert imi.legacy_grade(days) == module._assign_grade(days)
    dos = imi.days_of_supply(450.0, 10.0)   # 45 days of supply
    real = imi.classify(dos, basis=imi.REAL_SALES_TURNOVER, evidence=imi.SOURCE_DOCUMENTED, window_days=30, complete=True)
    proxy = imi.classify(dos, basis=imi.FLOW_BASED_TURNOVER_PROXY, evidence=imi.SOURCE_DOCUMENTED, window_days=30, complete=True)
    claimed = imi.classify(dos, basis=imi.REAL_SALES_TURNOVER, evidence=imi.COLUMN_HEADER_CLAIM, window_days=30, complete=None)
    assert (real["classification"], real["confidence"], real["scope"]) == ("NORMAL", "HIGH", imi.SCOPE_REAL)
    assert (proxy["confidence"], proxy["scope"]) == ("LOW", imi.SCOPE_PROXY)
    assert (claimed["confidence"], claimed["scope"]) == ("MEDIUM", imi.SCOPE_CLAIMED)


def test_turnover_period_must_align():
    sales = _window(_records(30, 10.0))
    other = imi.average_inventory(_records(30, 100.0, start="2026-06-01"), period_start="2026-06-01",
                                  period_end="2026-06-30", dataset_id="T", quantity_unit="EA")
    assert imi.physical_turnover(sales, other)["turnover_status"] == imi.PERIOD_MISMATCH


def test_unknown_units_only_pass_within_one_dataset():
    sales = _window(_records(30, 10.0), quantity_unit=None)
    same = imi.physical_turnover(sales, _inventory(_records(30, 100.0), quantity_unit=None))
    assert same["turnover_status"] == imi.TURNOVER_AVAILABLE and "UNIT_UNKNOWN_SAME_DATASET" in same["reason_codes"]
    known = imi.physical_turnover(sales, _inventory(_records(30, 100.0), quantity_unit="EA"))
    assert known["turnover_status"] == imi.UNIT_MISMATCH and "UNIT_UNVERIFIABLE" in known["reason_codes"]


def test_order_quantities_are_not_an_outflow():
    orders = _window(_records(30, 10.0), signal_type=imi.OBSERVED_ORDER)
    result = imi.physical_turnover(orders, _inventory(_records(30, 100.0)))
    assert result["turnover_status"] == imi.NOT_COMPARABLE and "ORDER_IS_NOT_OUTFLOW" in result["reason_codes"]


def test_financial_turnover_needs_explicit_cogs_and_value():
    assert imi.financial_turnover(cogs=900.0, cogs_currency="KRW", average_inventory_value=300.0, value_currency="KRW",
                                  period_days=30)["value"] == 3.0
    assert imi.financial_turnover(cogs=900.0, cogs_currency="KRW", average_inventory_value=300.0,
                                  value_currency="USD")["reason_codes"] == ["CURRENCY_MISMATCH"]
    assert imi.financial_turnover(average_inventory_value=300.0, value_currency="KRW")["reason_codes"] == ["COGS_MISSING"]


def test_gate_combines_by_severity_and_reuses_canonical_codes():
    assert imi.combine_gate(imi.VALID, imi.PROXY_ONLY, imi.VALID_WITH_LIMITATION) == imi.PROXY_ONLY
    assert imi.combine_gate(imi.MISSING_DATA, imi.INVALID_VALUE) == imi.INVALID_VALUE
    assert set(imi.CANONICAL_CODE_GATE) >= {"MISSING_VALUE", "NEGATIVE_QUANTITY", "BENCHMARK_PROXY", "UNKNOWN_UNIT"}
    favorita = "documented_return:sales_qty|promotion_not_reported|negative:sales_qty|unit_unspecified"
    assert imi.canonical_flags_gate(favorita, "MISSING_VALUE|NEGATIVE_QUANTITY|UNKNOWN_UNIT") == imi.MISSING_DATA
    assert imi.canonical_flags_gate("negative:order_qty", "NEGATIVE_QUANTITY") == imi.INVALID_VALUE


# --------------------------------------------------------------------------------------------- loader aliases


def test_loader_alias_replay_matches_the_loader():
    paths = imv.loader_alias_paths()
    assert len(paths) > 100 and paths["replay_matches_loader"].all()


@pytest.mark.parametrize("raw, field, conflict", [
    (["avg_daily_sales"], "sales_30d", True), (["sales_30d"], "avg_daily_sales", True), (["sales_7d"], "sales_30d", True),
    (["sales_qty"], "sales_30d", False), (["avg_daily_sales", "sales_30d"], "sales_30d", False),
])
def test_loader_alias_period_conflicts(raw, field, conflict):
    assert imi.resolve_loader_aliases(raw)[field]["period_conflict"] is conflict


def test_upload_copy_across_periods_is_detected_without_guessing_direction():
    inventory = normalize_loaded_data({"inventory": pd.DataFrame({"store_id": ["S1", "S2"], "product_id": ["P1", "P1"],
                                                                  "stock_qty": [10.0, 20.0], "avg_daily_sales": [2.0, 3.0]})})["inventory"]
    fields = imi.upload_sales_fields(inventory)
    assert fields["sales_30d"]["period_conflict"] and fields["avg_daily_sales"]["period_conflict"]
    signal, used, skipped = imi.row_sales_signal(inventory.iloc[0].to_dict(), fields)
    assert signal is None and "ALIAS_COPY_ACROSS_PERIODS" in skipped


def test_declared_one_day_proxy_resolves_the_direction():
    inventory = normalize_loaded_data({"inventory": pd.DataFrame({
        "store_id": ["S1", "S2"], "product_id": ["P1", "P1"], "stock_qty": [100.0, 50.0], "sales_qty": [0.0, 4.0],
        "snapshot_date": [D, D], "sales_qty_provenance": ["derived_proxy_from_actual_outbound_qty"] * 2,
        "sales_qty_semantics": [imi.DAILY_PROXY_SEMANTICS] * 2})})["inventory"]
    fields = imi.upload_sales_fields(inventory)
    assert fields["sales_qty"]["resolved_aggregation"] == imi.DAILY_QUANTITY and not fields["sales_qty"]["period_conflict"]
    assert fields["sales_30d"]["copy_of_declared_daily"] == ["sales_qty"] and fields["sales_30d"]["period_conflict"]
    assert fields["avg_daily_sales"]["copy_of_declared_daily"] == ["sales_qty"] and not fields["avg_daily_sales"]["period_conflict"]
    signal, used, _ = imi.row_sales_signal(inventory.iloc[1].to_dict(), fields)
    assert used == "sales_qty" and signal.signal_type == imi.PROXY_DEMAND and signal.period_days == 1


def test_equal_fields_of_compatible_periods_are_not_flagged():
    frame = pd.DataFrame({"sales_qty": [1.0, 2.0], "avg_daily_sales": [1.0, 2.0], "sales_30d": [30.0, 60.0]})
    fields = imi.upload_sales_fields(frame)
    assert not any(item["period_conflict"] for item in fields.values())


# --------------------------------------------------------------------------------------------- legacy diagnostics


def test_legacy_turnover_inputs_name_what_the_analyzer_did():
    fields = {}
    cols = ["stock_qty", "sales_30d"]
    assert "LEGACY_SALES_NULL_AS_ZERO" in imi.legacy_turnover_inputs({"stock_qty": 5.0, "sales_30d": None}, cols, fields)["legacy_codes"]
    assert "LEGACY_NEGATIVE_SALES_NEGATIVE_DAYS" in imi.legacy_turnover_inputs({"stock_qty": 5.0, "sales_30d": -3.0}, cols, fields)["legacy_codes"]
    assert "LEGACY_ZERO_STOCK_GRADED_FAST" in imi.legacy_turnover_inputs({"stock_qty": 0.0, "sales_30d": 3.0}, cols, fields)["legacy_codes"]
    assert "LEGACY_SALES_COLUMN_ABSENT_AS_ZERO" in imi.legacy_turnover_inputs({"stock_qty": 5.0}, ["stock_qty"], fields)["legacy_codes"]
    assert "LEGACY_SINGLE_SNAPSHOT_AS_AVERAGE" in imi.legacy_turnover_inputs({"stock_qty": 5.0, "sales_30d": 3.0}, cols, fields)["legacy_codes"]


def test_grade_cause_attribution():
    assert imi.grade_cause("DEAD", {}, ["LEGACY_SALES_NULL_AS_ZERO"]) == "MISSING_SALES_GRADED_DEAD"
    assert imi.grade_cause("FAST", {}, ["LEGACY_NEGATIVE_SALES_NEGATIVE_DAYS"]) == "NEGATIVE_SALES_GRADED_FAST"
    assert imi.grade_cause("DEAD", {}, ["LEGACY_30D_PROJECTED_FROM_ONE_DAY"]) == "PERIOD_ALIAS_DISTORTION"
    assert imi.grade_cause("FAST", {"turnover_classification": "FAST", "classification_scope": imi.SCOPE_REAL}, []) == "SAME"


# --------------------------------------------------------------------------------------------- ABC / disposal / safety / matching


def test_legacy_abc_basis_is_named():
    assert imi.legacy_abc_basis(["sales_30d", "unit_cost"])["legacy_basis"] == imi.ABC_COST_VALUED
    no_cost = imi.legacy_abc_basis(["sales_30d"])
    assert no_cost["legacy_basis"] == imi.ABC_QUANTITY_PROXY and no_cost["legacy_cost_input"] == "1.0"


def test_validated_abc_never_values_a_missing_price_at_one():
    rows = [{"product_id": "A", "period_sales": 10.0, "unit_price": 5.0, "signal_type": imi.OBSERVED_RETAIL_SALES},
            {"product_id": "B", "period_sales": 100.0, "unit_price": None, "signal_type": imi.OBSERVED_RETAIL_SALES}]
    result = imi.validated_abc(rows)
    assert result["basis"] == imi.ABC_QUANTITY_PROXY and result["gate_status"] == imi.PROXY_ONLY
    mixed = imi.validated_abc([{**rows[0], "quantity_unit": "EA"}, {**rows[1], "quantity_unit": "KG"}])
    assert mixed["basis"] == imi.ABC_NOT_VERIFIABLE and mixed["gate_status"] == imi.UNIT_MISMATCH
    missing = imi.validated_abc([{**rows[0]}, {"product_id": "C", "period_sales": None, "unit_price": 3.0}])
    assert missing["excluded_rows"] == 1 and "SALES_MISSING" in missing["reason_codes"]


def test_abc_grades_use_the_legacy_cutoffs():
    module = load_legacy_module("abc_analyzer")
    grades = imi.abc_grades({"A": 80.0, "B": 15.0, "C": 5.0})
    assert {k: v["grade"] for k, v in grades.items()} == {"A": module._assign_grade(0.8), "B": module._assign_grade(0.95),
                                                          "C": module._assign_grade(1.0)}
    assert imi.abc_thresholds() == (module._A_THRESHOLD, module._B_THRESHOLD)


def test_disposal_audit_names_placeholders():
    audit = imi.disposal_input_audit({"stock_qty": 5.0, "sales_30d": None}, ["stock_qty", "sales_30d"], {},
                                     raw_columns=["shelf_life_days", "stock_qty"])
    assert audit["meaning"] == imi.DISPOSAL_MEANING and audit["is_probability"] is False
    assert {"DISPOSAL_EXPIRY_ABSENT_REWEIGHTED", "DISPOSAL_INBOUND_DAYS_MISSING_AS_ZERO", "DISPOSAL_SALES_MISSING_AS_ZERO",
            "DISPOSAL_SHELF_LIFE_ALIASED_AS_REMAINING"} <= set(audit["codes"])
    assert imi.disposal_probability()["value"] is None


def test_safety_stock_and_match_audits():
    safety = imi.safety_stock_input_audit({"stock_qty": 5.0, "avg_daily_sales": 2.0}, ["stock_qty", "avg_daily_sales"], {})
    assert set(safety["placeholder_inputs"]) == {"SS_STD_PLACEHOLDER_30PCT", "SS_LEAD_TIME_DEFAULT_3"}
    match = imi.store_product_match_audit({"avg_daily_sales": 0.0}, ["avg_daily_sales"])
    assert match["use"] == "REFERENCE_SCORE_ONLY"
    assert {"MATCH_TARGET_SALES_NOT_SUPPLIED_NEUTRAL", "MATCH_SOURCE_ZERO_ZEROES_DEMAND_FIT"} <= set(match["codes"])
    nulled = imi.store_product_match_audit({"avg_daily_sales": 1.0, "state_target_sales_30d": None},
                                           ["avg_daily_sales", "state_target_sales_30d"])
    assert "MATCH_TARGET_SALES_NULL_AS_ZERO" in nulled["codes"]


def test_action_dependency_probes_the_unchanged_legacy_rule():
    module = load_legacy_module("varo_hybrid_score")
    row = {"disposal_risk_grade": "LOW", "disposal_risk_score": 10.0, "turnover_grade": "DEAD", "abc_grade": "B",
           "match_grade": "POOR", "reorder_status": "SAFE", "demand_trend": "STABLE", "vhs_raw": 40.0, "demand_risk_score": 50.0}
    probe = imi.action_dependency(row, {"turnover_classification": imi.UNCLASSIFIED, "classification_confidence": "NONE",
                                        "classification_scope": imi.SCOPE_NONE})
    assert probe["legacy_rule_action"] == module._recommend_action(pd.Series(row)) == "할인 판매"
    assert probe["turnover_dependency"] == "DEPENDS_ON_UNSUPPORTED_TURNOVER"
    assert probe["action_by_turnover_grade"]["FAST"] == "보류"
    verified = imi.action_dependency(row, {"turnover_classification": "DEAD", "classification_confidence": "MEDIUM",
                                           "classification_scope": imi.SCOPE_CLAIMED})
    assert verified["turnover_dependency"] == "DEPENDS_ON_REPRODUCED_TURNOVER"
    hot = {**row, "disposal_risk_grade": "CRITICAL", "disposal_risk_score": 90.0}
    assert imi.action_dependency(hot, None)["turnover_dependency"] in ("TURNOVER_INDEPENDENT", "DEPENDS_ON_UNSUPPORTED_TURNOVER")


# --------------------------------------------------------------------------------------------- field inventory / reasons


def test_field_inventory_covers_the_requested_fields():
    listed = " ".join(item[0] for item in imi.FIELD_INVENTORY)
    for name in ("sales_qty", "sales_7d", "sales_30d", "avg_daily_sales", "demand_qty", "demand_forecast_daily",
                 "demand_forecast_7d", "outbound_qty", "inbound_qty", "inventory_qty", "current_stock", "stock_qty",
                 "turnover", "sales_velocity", "daily_demand"):
        assert name in listed
    assert all(len(item) == len(imi.FIELD_INVENTORY_COLUMNS) for item in imi.FIELD_INVENTORY)
    frame = imv.field_inventory(REPO)
    absent = frame[frame["field"].str.startswith("sales_velocity")]
    assert int(absent["referencing_module_count"].iloc[0]) == 0
    assert (frame.loc[~frame["field"].str.startswith("sales_velocity"), "referencing_module_count"] > 0).all()


def test_undocumented_reason_codes_are_rejected():
    with pytest.raises(KeyError):
        imi._reasons("NOT_A_CODE")


# --------------------------------------------------------------------------------------------- pipeline field


def _load(path):
    return load_excel_data(path)


def test_pipeline_field_is_parallel_fault_isolated_and_deterministic():
    data = _load(NETWORK_SAMPLE)
    with_field = build_v2_state(copy.deepcopy(data), detail_level="full")
    with mock.patch.object(analysis_pipeline, "build_inventory_metric_integrity", side_effect=RuntimeError("boom")):
        broken = build_v2_state(copy.deepcopy(data), detail_level="full")
    for key in ("top5", "summary", "connected_algorithms", "status", "seller_loss_analysis",
                "shared_feasibility_selection", "action_consistency", "operational_evidence"):
        assert with_field["pipeline_result"][key] == broken["pipeline_result"][key]
    # The legacy summary tables hold NaN for absent grades (NaN != NaN), so they are compared as serialized text.
    for key in ("risk_analysis", "demand_analysis"):
        assert json.dumps(with_field["pipeline_result"][key], sort_keys=True, default=str) == \
            json.dumps(broken["pipeline_result"][key], sort_keys=True, default=str)
    assert with_field["recommendations"] == broken["recommendations"]
    assert broken["pipeline_result"]["inventory_metric_integrity"]["status"] == "error"
    field = with_field["pipeline_result"]["inventory_metric_integrity"]
    again = build_v2_state(copy.deepcopy(data), detail_level="full")["pipeline_result"]["inventory_metric_integrity"]
    assert again == field and field["status"] == "parallel_only" and field["production_action_applied"] is False
    assert field["legacy_metrics_replaced"] is False and field["thresholds_changed"] is False
    assert field["counts"]["rows"] == len(data["inventory"]) == field["counts"]["legacy_rows_aligned"]
    assert field["abc"]["legacy_basis"] == imi.ABC_COST_VALUED and field["disposal_risk"]["is_probability"] is False
    assert len(field["action_dependency"]["rows"]) == len(with_field["recommendations"])
    json.dumps(field, allow_nan=False)


def test_pipeline_field_reproduces_the_legacy_rule_action():
    state = build_v2_state(copy.deepcopy(_load(NETWORK_SAMPLE)), detail_level="core")
    rows = state["pipeline_result"]["inventory_metric_integrity"]["action_dependency"]["rows"]
    from services.recommendation_adapter import normalize_action
    by_route = {r["route_id"]: r for r in state["recommendations"]}
    assert rows and all(normalize_action(r["legacy_rule_action"]) == by_route[r["route_id"]]["varo_action"] for r in rows)


def _suhyup_like_upload():
    stores = pd.DataFrame({"store_id": ["A", "B"], "store_name": ["a", "b"], "node_type": ["STORE", "STORE"],
                           "latitude": [37.5, 37.6], "longitude": [127.0, 127.1]})
    products = pd.DataFrame({"product_id": ["P1", "P2"], "product_name": ["p1", "p2"]})
    inventory = pd.DataFrame({"store_id": ["A", "A", "B", "B"], "product_id": ["P1", "P2", "P1", "P2"],
                              "stock_qty": [900.0, 0.0, 40.0, 70.0], "sales_qty": [0.0, 3.0, 12.0, 5.0],
                              "snapshot_date": [D] * 4, "stock_qty_provenance": ["actual"] * 4,
                              "sales_qty_provenance": ["derived_proxy_from_actual_outbound_qty"] * 4,
                              "sales_qty_semantics": [imi.DAILY_PROXY_SEMANTICS] * 4})
    routes = pd.DataFrame({"source_id": ["A"], "target_id": ["B"], "distance_km": [10.0], "estimated_cost": [1000.0]})
    return normalize_loaded_data({"stores": stores, "products": products, "inventory": inventory, "routes": routes})


def test_outbound_upload_is_never_real_sales_turnover():
    upload = _suhyup_like_upload()
    legacy = prepare_legacy_data(upload)["inventory"]
    analyzed, _ = _run_inventory_analysis(_Runner(PipelineResult()), legacy)
    field = imi.build_inventory_metric_integrity(upload, analyzed)
    rows = field["rows"]
    assert field["counts"]["real_sales_turnover_rows"] == 0
    assert {r["turnover_basis"] for r in rows} == {imi.FLOW_BASED_TURNOVER_PROXY}
    assert {r["sales_signal_type"] for r in rows} == {imi.PROXY_DEMAND}
    assert {r["classification_scope"] for r in rows} <= {imi.SCOPE_PROXY, imi.SCOPE_NONE}
    assert all("LEGACY_30D_PROJECTED_FROM_ONE_DAY" in r["legacy_codes"] for r in rows)
    zero_stock = next(r for r in rows if r["stock_qty"] == 0.0)
    assert zero_stock["legacy_turnover_grade"] == "FAST" and zero_stock["turnover_classification"] == imi.UNCLASSIFIED
    assert field["abc"]["legacy_basis"] == imi.ABC_QUANTITY_PROXY and field["abc"]["validated_basis"] == imi.ABC_NOT_VERIFIABLE
    # production analyzers keep their own values: the legacy columns are read, never rewritten
    assert analyzed["turnover_grade"].tolist() == [r["legacy_turnover_grade"] for r in rows]


def test_field_without_inventory_is_reported_not_raised():
    assert imi.build_inventory_metric_integrity({})["status"] == "no_inventory"


# --------------------------------------------------------------------------------------------- streamed canonical stats


def _write_demand(path: Path, rows):
    frame = pd.DataFrame(rows)
    for column in ("validation_flags", "promotion"):
        if column not in frame:
            frame[column] = None
    frame[["date", "location_id", "product_id", "validation_flags", "promotion"]] = \
        frame[["date", "location_id", "product_id", "validation_flags", "promotion"]].astype("string")
    frame["sales_qty"] = frame["sales_qty"].astype("float64")
    frame.to_parquet(path, index=False, row_group_size=7)


def test_stream_demand_counts_absent_days_and_keeps_negatives(tmp_path):
    rows = []
    for day in range(40):
        date = (pd.Timestamp("2026-06-01") + pd.Timedelta(days=day)).date().isoformat()
        rows.append({"date": date, "location_id": "S1", "product_id": "P1", "sales_qty": -2.0 if day == 35 else 3.0,
                     "validation_flags": "documented_return:sales_qty|negative:sales_qty" if day == 35 else ""})
        if day % 4 == 0:
            rows.append({"date": date, "location_id": "S1", "product_id": "P2", "sales_qty": 1.5})
    path = tmp_path / "demand.parquet"
    _write_demand(path, rows)
    stream = imv.stream_demand(path, window_end="2026-07-10", combine_every=2)
    summary = imv.series_summary(stream)
    series = stream["series"].set_index("product_id")
    assert stream["totals"]["negative"] == 1 and stream["totals"]["documented_return"] == 1
    assert stream["totals"]["negative_sum"] == -2.0
    assert series.loc["P1", "win_rows"] == 30 and series.loc["P2", "win_rows"] < 30
    assert summary["window_complete_series"] == 1 and summary["window_partial_series"] == 1
    assert summary["absent_days_within_span"] == 27   # P2 is recorded every 4th day over a 37-day span; never zero-filled
    assert summary["series_with_noninteger_quantity"] == 1
    fidelity = imv.sample_window_fidelity(path, stream, samples=2, signal_type=imi.OBSERVED_RETAIL_SALES,
                                          negative_semantics=imi.RETURN_DOCUMENTED)
    assert fidelity and all(item["match"] for item in fidelity)


def test_cli_writes_only_its_own_bundle():
    assert imv.OUTPUT_FOLDER == "_INVENTORY_METRIC_INTEGRITY" and imv.OUTPUT_FOLDER not in imv.PROTECTED_BUNDLES
    for name in ("sales_semantics_inventory.csv", "sales_signal_validation.csv", "turnover_formula_audit.csv",
                 "turnover_missing_zero_analysis.csv", "turnover_by_dataset.csv", "abc_semantics_audit.csv",
                 "disposal_risk_input_audit.csv", "inventory_metric_coverage_matrix.csv",
                 "inventory_metric_integrity_summary.json"):
        assert name in imv.OUTPUT_FILES
    assert {"_TARGET_NEED_CONTRACT_VALIDATION", "_SHARED_FEASIBILITY_VALIDATION"} <= set(imv.PROTECTED_BUNDLES)
    assert imv.ALLOWED_MODIFIED == ("services/analysis_pipeline.py",)
