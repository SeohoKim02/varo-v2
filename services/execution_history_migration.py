"""Validated, one-way SQLite to external history-store migration."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.execution_history_store import (
    ExecutionHistoryStore,
    HistoryStoreError,
    SQLiteExecutionHistoryStore,
)


def validate_history_snapshot(snapshot: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    plans = [dict(row) for row in snapshot.get("plans", ())]
    items = [dict(row) for row in snapshot.get("items", ())]
    events = [dict(row) for row in snapshot.get("events", ())]
    issues: list[str] = []
    invalid_records: set[tuple[str, int]] = set()

    plan_ids = [str(row.get("plan_id") or "") for row in plans]
    plan_id_counts = Counter(plan_ids)
    duplicate_plan_ids = {value for value, count in plan_id_counts.items() if value and count > 1}
    for index, value in enumerate(plan_ids):
        if not value or value in duplicate_plan_ids:
            invalid_records.add(("plan", index))
    if any(not value for value in plan_ids) or duplicate_plan_ids:
        issues.append("계획 식별자가 비어 있거나 중복됩니다.")
    plan_id_set = set(plan_ids)

    item_keys = [
        (str(row.get("plan_id") or ""), str(row.get("candidate_id") or ""))
        for row in items
    ]
    item_key_counts = Counter(item_keys)
    duplicate_item_keys = {value for value, count in item_key_counts.items() if all(value) and count > 1}
    for index, (plan_id, candidate_id) in enumerate(item_keys):
        if not plan_id or not candidate_id or (plan_id, candidate_id) in duplicate_item_keys:
            invalid_records.add(("item", index))
    if any(not plan_id or not candidate_id for plan_id, candidate_id in item_keys):
        issues.append("실행 항목 식별자가 비어 있습니다.")
    if duplicate_item_keys:
        issues.append("실행 항목 식별자가 중복됩니다.")
    orphan_items = [index for index, (plan_id, _) in enumerate(item_keys) if plan_id not in plan_id_set]
    if orphan_items:
        invalid_records.update(("item", index) for index in orphan_items)
        issues.append("계획과 연결되지 않은 실행 항목이 있습니다.")

    item_key_set = set(item_keys)
    orphan_events = [
        index for index, row in enumerate(events)
        if (str(row.get("plan_id") or ""), str(row.get("candidate_id") or "")) not in item_key_set
    ]
    if orphan_events:
        invalid_records.update(("event", index) for index in orphan_events)
        issues.append("실행 항목과 연결되지 않은 감사 기록이 있습니다.")

    item_count_by_plan: dict[str, int] = {}
    qty_by_plan: dict[str, int] = {}
    for index, row in enumerate(items):
        plan_id = str(row.get("plan_id") or "")
        item_count_by_plan[plan_id] = item_count_by_plan.get(plan_id, 0) + 1
        try:
            planned_qty = int(row.get("planned_qty"))
        except (TypeError, ValueError):
            planned_qty = -1
        if planned_qty <= 0:
            invalid_records.add(("item", index))
            issues.append("계획 수량이 유효하지 않은 실행 항목이 있습니다.")
        qty_by_plan[plan_id] = qty_by_plan.get(plan_id, 0) + max(planned_qty, 0)

    for index, plan in enumerate(plans):
        plan_id = str(plan.get("plan_id") or "")
        try:
            expected_actions = int(plan.get("total_actions"))
            expected_qty = int(plan.get("total_planned_qty"))
        except (TypeError, ValueError):
            invalid_records.add(("plan", index))
            issues.append("계획 집계값 형식이 올바르지 않습니다.")
            continue
        if item_count_by_plan.get(plan_id, 0) != expected_actions:
            invalid_records.add(("plan", index))
            issues.append("계획 항목 수가 계획 집계와 일치하지 않습니다.")
        if qty_by_plan.get(plan_id, 0) != expected_qty:
            invalid_records.add(("plan", index))
            issues.append("계획 수량 합계가 계획 집계와 일치하지 않습니다.")

    total_records = len(plans) + len(items) + len(events)
    return {
        "valid": not issues,
        "issues": list(dict.fromkeys(issues)),
        "plan_count": len(plans),
        "item_count": len(items),
        "audit_count": len(events),
        "valid_record_count": total_records - len(invalid_records),
        "invalid_record_count": len(invalid_records),
    }


def migrate_sqlite_history(
    source_path: str | Path,
    destination: ExecutionHistoryStore,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Inspect or migrate a complete source snapshot without modifying SQLite."""
    resolved_source = Path(source_path)
    if not resolved_source.is_file():
        return {
            "ok": False,
            "code": "migration_error",
            "message": "SQLite 원본 실행 기록을 찾지 못했습니다.",
            "plan_count": 0,
            "item_count": 0,
            "audit_count": 0,
            "valid_record_count": 0,
            "invalid_record_count": 0,
            "duplicate_plan_count": 0,
        }
    source = SQLiteExecutionHistoryStore(resolved_source)
    try:
        snapshot = source.read_snapshot()
        validation = validate_history_snapshot(snapshot)
        if not validation["valid"]:
            return {
                "ok": False,
                "code": "invalid_source",
                "message": "SQLite 실행 기록 검증에 실패했습니다.",
                **validation,
                "duplicate_plan_count": 0,
            }

        existing_ids = destination.existing_plan_ids(initialize=not dry_run)
        source_ids = {str(row["plan_id"]) for row in snapshot["plans"]}
        duplicate_ids = source_ids & existing_ids
        new_ids = source_ids - duplicate_ids
        new_items = [row for row in snapshot["items"] if str(row["plan_id"]) in new_ids]
        new_events = [row for row in snapshot["events"] if str(row["plan_id"]) in new_ids]
        summary = {
            **validation,
            "duplicate_plan_count": len(duplicate_ids),
            "new_plan_count": len(new_ids),
            "new_item_count": len(new_items),
            "new_audit_count": len(new_events),
        }
        if dry_run:
            return {
                "ok": True,
                "code": "dry_run",
                "message": "이관 전 검증을 완료했습니다. 대상 DB는 변경하지 않았습니다.",
                **summary,
                "inserted_plan_count": 0,
                "inserted_item_count": 0,
                "inserted_audit_count": 0,
            }

        inserted = destination.import_snapshot(snapshot, skip_plan_ids=duplicate_ids)
        expected = {
            "plans": summary["new_plan_count"],
            "items": summary["new_item_count"],
            "events": summary["new_audit_count"],
        }
        if inserted != expected:
            return {
                "ok": False,
                "code": "count_mismatch",
                "message": "이관 건수 검증에 실패했습니다.",
                **summary,
            }
        return {
            "ok": True,
            "code": "migrated",
            "message": "실행 기록 이관을 완료했습니다.",
            **summary,
            "inserted_plan_count": inserted["plans"],
            "inserted_item_count": inserted["items"],
            "inserted_audit_count": inserted["events"],
        }
    except (HistoryStoreError, OSError):
        return {
            "ok": False,
            "code": "migration_error",
            "message": "실행 기록을 이관하지 못했습니다.",
            "plan_count": 0,
            "item_count": 0,
            "audit_count": 0,
            "valid_record_count": 0,
            "invalid_record_count": 0,
            "duplicate_plan_count": 0,
        }
