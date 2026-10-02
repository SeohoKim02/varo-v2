import json
import os

import pandas as pd

from services.real_data_adapters import DATASETS
from services.seller_decision_validation import (
    CELL_STATUSES, COVERAGE_FIELDS, DATASET_COVERAGE, DATASET_ORDER, FULL_REQUIRED, LEGACY_ACTION_STRUCTURE,
    MONETARY_FIELD_SURVEY, _env, _parquet_nonnull, coverage_matrix, coverage_verdict, run_scenarios,
)

EXPECTED_VERDICTS = {
    "suhyup": "PARTIAL_MONETARY", "jangbogo": "NON_MONETARY_ONLY", "logisall": "NON_MONETARY_ONLY",
    "nfqs": "NON_MONETARY_ONLY", "aihub": "UNAVAILABLE", "kamp": "UNAVAILABLE", "m5": "PARTIAL_MONETARY",
    "favorita": "NON_MONETARY_ONLY", "freshretailnet": "NON_MONETARY_ONLY",
}


def _cells(status="DIRECT_REAL", **overrides):
    cells = {name: {"status": status, "evidence": "", "checks": []} for name in COVERAGE_FIELDS}
    for name, value in overrides.items():
        cells[name] = {"status": value, "evidence": "", "checks": []}
    return cells


def test_every_dataset_has_a_complete_classified_row():
    assert tuple(DATASET_COVERAGE) == DATASET_ORDER and len(DATASET_ORDER) == 9
    for spec in DATASET_COVERAGE.values():
        for name in COVERAGE_FIELDS:
            assert spec[name]["status"] in CELL_STATUSES and spec[name]["evidence"]


def test_dataset_verdicts_are_computed_per_dataset_and_none_is_full():
    verdicts = {name: coverage_verdict(DATASET_COVERAGE[name], DATASET_COVERAGE[name]["quantity_unit"])[0] for name in DATASET_ORDER}
    assert verdicts == EXPECTED_VERDICTS
    # A union across datasets (M5 price + Suhyup stock + FRN discount ...) is never formed: each verdict uses one dataset.
    union = {name: max((DATASET_COVERAGE[d][name] for d in DATASET_ORDER),
                       key=lambda cell: cell["status"] in ("DIRECT_REAL", "DERIVED_REAL")) for name in COVERAGE_FIELDS}
    assert sum(union[name]["status"] in ("DIRECT_REAL", "DERIVED_REAL") for name in FULL_REQUIRED) > \
        max(sum(DATASET_COVERAGE[d][name]["status"] in ("DIRECT_REAL", "DERIVED_REAL") for name in FULL_REQUIRED) for d in DATASET_ORDER)


def test_verdict_rules():
    assert coverage_verdict(_cells(), "ea")[0] == "FULL_MONETARY"
    assert coverage_verdict(_cells(), "UNKNOWN")[0] != "FULL_MONETARY"
    assert coverage_verdict(_cells(), "normalized_sales_amount (non-physical)")[0] != "FULL_MONETARY"
    assert coverage_verdict(_cells(shelf_life_expiry="MISSING"), "ea")[0] == "PARTIAL_MONETARY"
    no_money = _cells(**{name: "MISSING" for name in ("normal_selling_price", "unit_cost", "transfer_cost", "holding_cost",
                                                      "disposal_cost", "salvage_value")})
    assert coverage_verdict(no_money, "ea")[0] == "NON_MONETARY_ONLY"
    assert coverage_verdict(_cells("MISSING"), "ea")[0] == "UNAVAILABLE"
    assert coverage_verdict(_cells(transfer_cost="PROXY"), "ea")[0] == "PARTIAL_MONETARY"   # a proxy never makes FULL


def test_parquet_metadata_count(tmp_path):
    path = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, None, None]}).to_parquet(path)
    assert _parquet_nonnull(path, "a") == 2 and _parquet_nonnull(path, "b") == 0
    assert _parquet_nonnull(path, "absent") == 0 and _parquet_nonnull(tmp_path / "missing.parquet", "a") is None


def test_coverage_claims_are_checked_against_the_data(tmp_path):
    frame, verdicts = coverage_matrix(tmp_path)
    assert not (frame["check_result"] == "FAIL").any()                       # no files: undetermined, never "PASS"
    assert {verdicts[name]["verdict"] for name in verdicts} == set(EXPECTED_VERDICTS.values())
    folder = tmp_path / DATASETS["suhyup"] / "processed"
    folder.mkdir(parents=True)
    pd.DataFrame({"sell_price": [1200.0], "unit_cost": [None]}).to_parquet(folder / "canonical_product_master.parquet")
    frame, _ = coverage_matrix(tmp_path)
    row = frame[(frame["dataset"] == "suhyup") & (frame["field"] == "normal_selling_price")].iloc[0]
    assert row["check_result"] == "FAIL"                                      # a claimed MISSING price is contradicted
    assert json.loads(row["verified_checks"])[0]["nonnull_rows"] == 1


def test_controlled_scenarios_all_pass_and_are_labelled_synthetic():
    frame, results = run_scenarios()
    assert len(frame) == len(results) >= 18 and frame["pass"].all()
    assert frame["data_kind"].str.contains("synthetic test input, not real data").all()
    statuses = set(frame["comparison_status"])
    assert {"FULL", "PARTIAL", "COMPARISON_UNAVAILABLE"} <= statuses
    assert {"TRANSFER", "NORMAL_SALE", "DISCOUNT_SALE"} <= set(frame["recommended_strategy"].dropna())


def test_field_survey_and_legacy_structure_use_declared_classes():
    allowed = {"DIRECT_REAL", "DERIVED_REAL", "PROXY", "CONFIG", "MISSING", "USER_INPUT"}
    assert {item["classification"] for item in MONETARY_FIELD_SURVEY} <= allowed
    assert {item["kind"] for item in LEGACY_ACTION_STRUCTURE} <= {"RULE_BASED", "CALCULATED", "PLACEHOLDER", "ABSENT"}
    placeholders = [item for item in MONETARY_FIELD_SURVEY if "promotion_sales_increase_rate" in item["field"]]
    assert placeholders and placeholders[0]["classification"] == "CONFIG"


def test_env_context_restores_previous_value():
    name = "VARO_SELLER_LOSS_TEST_ENV"
    os.environ.pop(name, None)
    with _env(name, "x"):
        assert os.environ[name] == "x"
    assert name not in os.environ
    os.environ[name] = "keep"
    with _env(name, "y"):
        assert os.environ[name] == "y"
    assert os.environ.pop(name) == "keep"
