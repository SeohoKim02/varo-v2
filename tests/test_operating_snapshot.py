from dataclasses import replace
import pytest
from services.operating_snapshot import from_canonical
from services.target_need_contract import TargetNeed, OPERATOR_CONFIRMED_NEED, VERIFIED


def row(**kw):
    return {"source_dataset": "shop", "source_file": "inventory.csv", "source_sheet_or_table": "inventory",
        "source_row_id": "1", "transform_version": "1.2.0", "snapshot_date": "2026-10-16",
        "location_key": "shop:location:A", "product_key": "shop:product:P", "product_state": "CHILLED",
        "lot_id": "L1", "unit": "EA", "unit_status": "DIRECT", "inventory_qty": 10,
        "approved_transfer_qty": 8, "transfer_evidence": "VERIFIED_OPERATOR_INPUT",
        "transfer_valid_until": "2026-10-20", **kw}


def snap(rows, **kw):
    return from_canonical(rows, dataset="shop", network="N", as_of="2026-10-16", horizon_days=3, **kw)


def test_unknown_reservation_is_not_zero():
    p = snap([row()]).positions[0]
    assert p.reserved is None and p.approved_transfer == 8


def test_on_hand_is_not_permission():
    p = snap([row(approved_transfer_qty=None)]).positions[0]
    assert p.on_hand == 10 and p.approved_transfer is None
    assert "TRANSFER_APPROVAL_REQUIRED" in p.issues


def test_shared_source_reservation_cap():
    assert snap([row(reserved_qty=4)]).positions[0].approved_transfer == 6


def test_state_lot_never_merge():
    s = snap([row(), row(product_state="FROZEN"), row(lot_id="L2")])
    assert s.report["keys"] == 3 and s.report["rows_after"] == 3


def test_duplicate_rows_quarantined_without_drop():
    s = snap([row(), row(inventory_qty=20)])
    assert s.report["rows_after"] == 2 and s.report["duplicate_rows"] == 2
    assert s.report["observed_totals_by_unit"] == {"EA": 30}
    assert s.report["accepted_totals_by_unit"] == {}


@pytest.mark.parametrize("field,value,issue", [
    ("product_state", None, "STATE_UNKNOWN"), ("unit", None, "UNIT_UNCONFIRMED"),
    ("product_state", "UNKNOWN", "STATE_UNKNOWN"), ("unit", "UNKNOWN", "UNIT_UNCONFIRMED"),
    ("source_dataset", "else", "DATASET_MISMATCH"), ("snapshot_date", "2026-10-17", "AS_OF_MISMATCH"),
    ("transfer_valid_until", "2026-10-01", "TRANSFER_APPROVAL_INVALID"),
    ("source_file", None, "LINEAGE_INCOMPLETE"), ("inventory_qty", -1, "ON_HAND_INVALID"),
])
def test_explicit_invalid_input(field, value, issue):
    assert issue in snap([row(**{field: value})]).positions[0].issues


def test_order_invariant_signature():
    assert snap([row(), row(lot_id="L2")]).base_problem_signature == snap([row(lot_id="L2"), row()]).base_problem_signature


def test_need_contract_reused_and_period_enforced():
    n = TargetNeed(OPERATOR_CONFIRMED_NEED, "N", "shop:location:B", "shop:product:P", "2026-10-16",
        5, "VERIFIED_OPERATOR_INPUT", dataset_id="shop", valid_until="2026-10-20",
        planning_horizon_days=3, quantity_unit="EA", validation_status=VERIFIED)
    assert not snap([row()], target_needs=[n]).report["invalid_needs"]
    assert snap([row()], target_needs=[replace(n, planning_horizon_days=7)]).report["invalid_needs"]
