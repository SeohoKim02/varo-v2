"""T5 fault injection uses synthetic copies, never real references."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from services import release_validation as rv
from services.release_regression import (compare_golden, compare_mapping, inspect_reference, readiness,
                                         semantic_digest, sha256, STATUSES)

DATES = ["2026-07-01", "2026-07-02"]
STRATEGIES = ["Varo Final", "MILP", "Greedy", "VHS", "Pareto", "DQN"]


def golden():
    return pd.DataFrame([dict(date=d, strategy=s, route_ids="R1|R2", service_qty=50.0,
                              total_cost=100.0, feasibility_violations=0, rank=1, action="TRANSFER", status="feasible")
                         for d in DATES for s in STRATEGIES])


def compare(actual):
    return compare_golden(golden(), actual, DATES, STRATEGIES)


def test_golden_identical_and_deterministic():
    a = compare(golden())
    b = compare(golden().sample(frac=1, random_state=17))
    assert a == b
    assert a["status"] == "PASS" and len(a["rows"]) == 12


def test_missing_row():
    result = compare(golden().iloc[:-1])
    assert result["status"] == "FAIL"
    assert result["failures"][0]["reason"] == "MISSING_ACTUAL_ROW"


@pytest.mark.parametrize("field,value", [("service_qty", 51.0), ("total_cost", 101.0),
    ("feasibility_violations", 1), ("route_ids", "R2|R1"), ("rank", 2), ("action", "DISCOUNT_SALE"), ("status", "GLOBAL_OPTIMAL")])
def test_each_golden_field_change(field, value):
    actual = golden()
    actual.loc[0, field] = value
    result = compare(actual)
    assert result["status"] == "FAIL" and result["failure_count"] == 1
    assert result["failures"][0]["field"] == field
    assert result["failures"][0]["date"] == DATES[0]


@pytest.mark.parametrize("field,value", [("date", DATES[0]), ("strategy", "VHS")])
def test_duplicate_with_same_total_row_count(field, value):
    actual = golden()
    actual.loc[6 if field == "date" else 0, field] = value
    result = compare(actual)
    assert len(actual) == 12
    assert result["status"] == "FAIL"
    assert any("DUPLICATE" in f["reason"] for f in result["failures"])


@pytest.mark.parametrize("column", ["route_ids", "action", "rank", "date", "strategy"])
def test_schema_missing_field(column):
    assert compare(golden().drop(columns=column))["difference"] == "SCHEMA_CHANGED"


def test_golden_missing_reference_key():
    result = compare_golden(golden().drop(columns="date"), golden(), DATES, STRATEGIES)
    assert result["status"] == "FAIL" and result["difference"] == "SCHEMA_CHANGED"


def test_null_is_not_zero():
    expected = golden()
    expected.loc[0, "service_qty"] = None
    assert compare_golden(expected, golden(), DATES, STRATEGIES)["status"] == "FAIL"
    assert compare_golden(expected, expected.copy(), DATES, STRATEGIES)["status"] == "PASS"


@pytest.mark.parametrize("expected,actual", [
    ({"plan": "PROXY_PLAN"}, {"plan": "ACTUAL"}),
    ({"promotion": "TEST_BLOCKED", "eligible": 0}, {"promotion": "PROMOTION_ELIGIBLE", "eligible": 1}),
    ({"config_signature": "fixed"}, {"config_signature": "changed"}),
    ({"GENERALIZATION_VERDICT": "FAIL"}, {"GENERALIZATION_VERDICT": "PASS"}),
    ({"operator_action": None}, {"operator_action": "TRANSFER"}),
    ({"global_optimal": False}, {"global_optimal": True}),
    ({"violations": 0}, {"violations": 1}),
])
def test_semantic_contract_regressions(expected, actual):
    result = compare_mapping(expected, actual)
    assert result["status"] == "FAIL" and result["difference"] == "VALUE_CHANGED"


def test_missing_reference(tmp_path):
    result = inspect_reference(tmp_path / "missing.csv", "expected")
    assert result["status"] == "BLOCKED" and result["difference"] == "MISSING_REFERENCE"


def test_access_error_distinct_from_code_failure(monkeypatch):
    def locked(path): raise PermissionError("OneDrive fixture locked")
    monkeypatch.setattr("services.release_regression.sha256", locked)
    assert inspect_reference(Path("locked"), "expected")["status"] == "ENVIRONMENT_ERROR"


def test_model_unavailable_no_training(monkeypatch):
    monkeypatch.setattr("services.action_consistency_validation._dqn_model", lambda: None)
    assert rv.model_check()["reason_codes"] == ["MODEL_UNAVAILABLE"]
    assert rv.model_check()["status"] == "BLOCKED"


def test_hash_detects_changed_file(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("old", encoding="utf-8")
    digest = sha256(path)
    path.write_text("new", encoding="utf-8")
    result = inspect_reference(path, digest)
    assert result["status"] == "FAIL" and result["difference"] == "VALUE_CHANGED"
    assert path.read_text() == "new"  # comparison never rewrites reference


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED", "ENVIRONMENT_ERROR", "NOT_RUN", "INSUFFICIENT_DATA"])
def test_nonpass_never_counted_as_pass(status):
    gate = readiness([{"status": status}], "full")
    assert gate["status_counts"]["PASS"] == 0
    assert gate["overall"] != "READY"
    assert not gate["production_approval"]


def test_frozen_not_actual_operational_approval():
    gate = readiness([{"status": "PASS", "execution_level": "REUSED_FROZEN_RESULT"}], "full")
    assert gate["code_regression"] == "READY_WITH_LIMITATIONS"
    assert gate["operational_evidence"] == "BLOCKED" and gate["seller_actual_application"] == "BLOCKED"


def test_critical_failure_blocks_release():
    gate = readiness([{"status": "FAIL", "validation_id": "GOLDEN"}, {"status": "PASS"}], "full")
    assert gate["overall"] == "BLOCKED" and gate["regression_verdict"] == "FAIL"


def test_volatile_metadata_excluded():
    a = {"qty": 10, "duration": 1, "generated_at": "a"}
    b = {"qty": 10, "duration": 2, "generated_at": "b"}
    assert compare_mapping(a, b)["status"] == "PASS"
    assert semantic_digest(a) == semantic_digest(b)


def test_historical_nonfinite_serializes_to_null_not_zero():
    from services.release_regression import json_safe
    value = {"actual": float("nan"), "runtime": 10, "nested": [float("inf")]}
    safe = json_safe(value)
    assert safe == {"actual": None, "runtime": 10, "nested": [None]}
    assert json.loads(json.dumps(safe, allow_nan=False))["actual"] is None
    assert semantic_digest(value) == semantic_digest({"actual": None, "runtime": 20, "nested": [None]})


def test_missing_contract_and_schema():
    assert compare_mapping(None, {})["difference"] == "MISSING_REFERENCE"
    assert compare_mapping({}, None)["difference"] == "MISSING_OUTPUT"
    assert compare_mapping({"a": 1}, {})["difference"] == "SCHEMA_CHANGED"


def test_failure_examples_bounded():
    actual = golden()
    actual["service_qty"] = 51
    result = compare(actual)
    assert result["failure_count"] == 12 and len(result["failures"]) == 12


def test_independent_check_continues_and_contract_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "git", lambda *args: "head" if args == ("rev-parse", "HEAD") else "")
    audit = rv.Audit(tmp_path, tmp_path, "quick")
    audit.check("A", "failure", lambda: {"status": "FAIL"})
    audit.check("B", "independent", lambda: {"status": "PASS"})
    assert [r["status"] for r in audit.results] == ["FAIL", "PASS"]
    audit.check("C", "invalid contract", lambda: {"status": "UNRECOGNIZED"})
    assert audit.results[-1]["status"] == "FAIL"
    assert "VALIDATION_RESULT_CONTRACT_INVALID" in audit.results[-1]["reason_codes"]
    required = {"validation_id", "validation_name", "validation_mode", "started_at", "finished_at", "duration",
                "status", "expected_result", "actual_result", "difference", "reason_codes", "data_source",
                "reference_version", "git_head", "git_dirty", "output_path"}
    assert required <= audit.results[0].keys()


def test_inventory_and_no_old_test_changes():
    items = rv.validator_inventory()
    assert len(items) >= 20
    assert {"canonical_reference_validation", "suhyup_algorithm_revalidation", "seller_business_profile_validation"} <= {r["validator"] for r in items}
    assert all(r["runtime_seconds"] is None for r in items)


def test_reject_overwriting_or_outside_output(tmp_path):
    existing = tmp_path / "_RELEASE_VALIDATION_existing"
    existing.mkdir()
    with pytest.raises(FileExistsError): rv.run_validation(tmp_path, output_dir=existing)
    with pytest.raises(ValueError): rv.run_validation(tmp_path, output_dir=tmp_path / "archive")


def test_manifest_has_complete_matrix_and_no_data():
    manifest = json.loads(rv.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["golden"]["rows"] == 186
    assert len(manifest["golden"]["dates"]) == 31 and len(manifest["golden"]["strategies"]) == 6
    assert "route_ids" in manifest["golden"]["columns"]
    assert len(manifest["files"]) >= 100
    assert all(len(digest) == 64 for digest in manifest["files"].values())


def test_invalid_mode_and_m5_option(tmp_path):
    with pytest.raises(ValueError): rv.run_validation(tmp_path, "unknown")
    with pytest.raises(ValueError): rv.run_validation(tmp_path, "quick", rerun_m5=True)
