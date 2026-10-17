import hashlib

import pandas as pd
import pytest

from services.operator_intake_candidate import import_operator_csv_bundle, validate_operator_intake


def metadata():
    return {"dataset_id": "operator", "network_id": "retailer_a", "as_of_date": "2026-10-16",
            "owner_authorized": True, "allowed_use": "RESEARCH_SHADOW", "publication_scope": "AGGREGATES_ONLY",
            "source_hashes": {"inventory.csv": "a" * 64}}


def stock(**updates):
    row = {"source_dataset": "operator", "source_file": "inventory.csv", "source_sheet_or_table": "inventory",
           "source_row_id": "1", "transform_version": "1", "snapshot_date": "2026-10-15",
           "date_grain": "daily", "scope": "location_product", "location_id": "s1", "product_id": "p1",
           "inventory_qty": 10, "available_qty": None, "reserved_qty": None, "unit": "EA", "unit_status": "DIRECT",
           "location_id_namespace": "retailer_a:location_id", "location_key": "retailer_a:location_id:s1",
           "product_id_namespace": "retailer_a:product_id", "product_key": "retailer_a:product_id:p1",
           "product_state": "CHILLED", "lot_id": "lot1"}
    return {**row, **updates}


def report(rows=None, meta=None):
    return validate_operator_intake({"inventory_snapshot": pd.DataFrame(rows or [stock()])}, meta or metadata())


def codes(result):
    return {item["code"] for item in result.issues}


def test_valid_shadow_only_no_imputation_or_mutation():
    source = pd.DataFrame([stock()])
    before = source.copy(deep=True)
    result = validate_operator_intake({"inventory_snapshot": source}, metadata())
    pd.testing.assert_frame_equal(source, before)
    assert result.status == "VALID_FOR_SHADOW_REVIEW"
    assert result.records["inventory_snapshot"][0]["reserved_qty"] is None
    assert not result.production_approval and not result.effect_claim_allowed


@pytest.mark.parametrize("field,value,code", [
    ("source_file", "", "MISSING_REQUIRED"), ("source_dataset", "other", "DATASET_MISMATCH"),
    ("product_key", "other:p1", "IDENTITY_SCOPE_MISMATCH"), ("unit", None, "QUANTITY_UNIT_UNKNOWN"),
    ("unit_status", "UNKNOWN", "QUANTITY_UNIT_UNKNOWN"), ("snapshot_date", "2027-01-01", "INVALID_OBSERVATION_DATE"),
    ("snapshot_date", "bad", "INVALID_OBSERVATION_DATE"), ("inventory_qty", -1, "INVALID_NUMBER"),
    ("inventory_qty", "inf", "INVALID_NUMBER"), ("inventory_qty", True, "INVALID_NUMBER"),
    ("inventory_qty", "bad", "INVALID_NUMBER"), ("product_id", "SAMPLE_PRODUCT", "SAMPLE_NOT_OBSERVATION"),
    ("validation_flags", "negative:inventory_qty", "SOURCE_QUALITY_BLOCKED"),
])
def test_reject_invalid_stock(field, value, code):
    result = report([stock(**{field: value})])
    assert result.status == "REJECTED"
    assert code in codes(result)


@pytest.mark.parametrize("field,value,code", [
    ("owner_authorized", "true", "OWNER_APPROVAL_REQUIRED"),
    ("allowed_use", "PRODUCTION", "USE_SCOPE_INVALID"),
    ("publication_scope", "PUBLIC", "PUBLICATION_SCOPE_REQUIRED"),
    ("as_of_date", "NaT", "INVALID_AS_OF_DATE"),
    ("source_hashes", {}, "SOURCE_HASH_REQUIRED"),
])
def test_metadata_fail_closed(field, value, code):
    assert code in codes(report(meta={**metadata(), field: value}))


def test_duplicate_retained_not_merged():
    result = report([stock(), stock(source_row_id="2", inventory_qty=20)])
    assert "DUPLICATE_OR_CONFLICT" in codes(result)
    assert [r["inventory_qty"] for r in result.records["inventory_snapshot"]] == [10, 20]


def test_state_and_lot_are_separate():
    result = report([stock(), stock(source_row_id="2", product_state="FROZEN"), stock(source_row_id="3", lot_id="lot2")])
    assert "DUPLICATE_OR_CONFLICT" not in codes(result)
    assert len(result.records["inventory_snapshot"]) == 3


def test_empty_is_not_valid():
    assert "NO_OBSERVATIONS" in codes(validate_operator_intake({}, metadata()))


def test_reuses_existing_evidence_parser():
    row = {**stock(), "network_id": "foreign", "source_id": "s1", "target_id": "s2", "allowed": "true",
           "effective_from": "2026-10-01", "evidence_class": "VERIFIED_OPERATOR_INPUT", "evidence_source": "permission"}
    result = validate_operator_intake({"operational_route_evidence": pd.DataFrame([row])}, metadata())
    assert "CROSS_NETWORK_EVIDENCE_REJECTED" in codes(result)


def test_future_actual_sales_rejected():
    row = {**stock(), "network_id": "retailer_a", "quantity_kind": "ACTUAL_SALES", "quantity": 5,
           "quantity_unit": "EA", "observation_start": "2026-10-15", "observation_end": "2026-10-20",
           "evidence_class": "DIRECT_REAL", "data_source": "POS"}
    result = validate_operator_intake({"operational_demand_evidence": pd.DataFrame([row])}, metadata())
    assert "FUTURE_SALES" in codes(result)


def test_route_negative_capacity_and_duplicate_conflict_rejected():
    row = {**stock(), "network_id": "retailer_a", "source_id": "s1", "target_id": "s2", "allowed": "true",
           "effective_from": "2026-10-01", "evidence_class": "VERIFIED_OPERATOR_INPUT", "evidence_source": "permission",
           "route_capacity_qty": -5, "quantity_unit": "EA"}
    result = validate_operator_intake({"operational_route_evidence": pd.DataFrame([row, {**row, "allowed": "false"}])}, metadata())
    assert {"INVALID_NUMBER", "DUPLICATE_OR_CONFLICT"}.issubset(codes(result))


def test_invalid_table_returns_actionable_issue():
    result = validate_operator_intake({"inventory_snapshot": 1}, metadata())
    assert "INVALID_TABLE" in codes(result)


def outcome(**updates):
    row = {**stock(), "network_id": "retailer_a", "action_id": "a1", "action": "NORMAL_SALE",
           "action_date": "2026-10-01", "observation_start": "2026-10-01", "observation_end": "2026-10-15",
           "quantity_unit": "EA", "outcome_kind": "OBSERVED_ACTUAL", "operator_approved": True,
           "evidence_class": "DIRECT_REAL", "evidence_source": "POS", "sales_qty": 5, "currency": "KRW"}
    return {**row, **updates}


def test_missing_outcomes_not_zero_or_effect():
    result = validate_operator_intake({"action_outcome": pd.DataFrame([outcome()])}, metadata())
    assert result.status == "REVIEW_REQUIRED"
    assert "PARTIAL_OUTCOME" in codes(result)
    assert "actual_transport_spend" not in result.records["action_outcome"][0]
    assert not result.effect_claim_allowed


@pytest.mark.parametrize("updates,expected", [
    ({"outcome_kind": "SIMULATION"}, "NOT_OBSERVED_OUTCOME"),
    ({"evidence_class": "PROXY"}, "NOT_OBSERVED_OUTCOME"),
    ({"network_id": "other"}, "CROSS_NETWORK_OUTCOME"),
    ({"operator_approved": False}, "ACTION_APPROVAL_REQUIRED"),
    ({"observation_end": "2026-10-17"}, "INVALID_OUTCOME_WINDOW"),
    ({"observation_start": "2026-09-01"}, "INVALID_OUTCOME_WINDOW"),
    ({"sales_qty": None}, "OUTCOME_NOT_MEASURED"),
    ({"actual_revenue": 1, "currency": None}, "CURRENCY_REQUIRED"),
])
def test_outcome_guards(updates, expected):
    result = validate_operator_intake({"action_outcome": pd.DataFrame([outcome(**updates)])}, metadata())
    assert result.status == "REJECTED"
    assert expected in codes(result)


def test_import_hash_and_lineage(tmp_path):
    path = tmp_path / "inventory.csv"
    pd.DataFrame([stock(inventory_qty=0)]).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {**metadata(), "source_hashes": {"inventory.csv": digest}}
    result = import_operator_csv_bundle({"inventory_snapshot": "inventory.csv"}, meta, root=tmp_path)
    assert result.status == "VALID_FOR_SHADOW_REVIEW"
    assert result.file_hashes == {"inventory.csv": digest}
    assert result.records["inventory_snapshot"][0]["inventory_qty"] == "0"
    assert result.records["inventory_snapshot"][0]["reserved_qty"] == ""
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="원본 확인"):
        import_operator_csv_bundle({"inventory_snapshot": "inventory.csv"}, metadata(), root=tmp_path)


def test_import_cannot_escape_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="입력 폴더"):
        import_operator_csv_bundle({"inventory_snapshot": outside}, metadata(), root=root)


def test_duplicate_headers_rejected(tmp_path):
    path = tmp_path / "inventory.csv"
    path.write_text("a,a\n1,2\n", encoding="utf-8")
    meta = {**metadata(), "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
    with pytest.raises(ValueError, match="중복"):
        import_operator_csv_bundle({"inventory_snapshot": path.name}, meta, root=tmp_path)
