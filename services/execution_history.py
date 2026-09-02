"""Persistent execution-history service for shadow operations.

The UI never issues SQL directly.  This module snapshots a validated execution
plan, records what operators actually did, and exports the joined expected vs
actual facts for later offline calibration.  It does not feed any value back to
VHS, DQN, or the execution-plan optimizer.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DB = PROJECT_ROOT / "runtime_data" / "varo_execution_history.sqlite3"

STATUS_LABELS = {
    "unconfirmed": "미확인",
    "executed": "실행",
    "partial": "일부 실행",
    "not_executed": "미실행",
    "cancelled": "취소",
}
STATUS_CODES_BY_LABEL = {label: code for code, label in STATUS_LABELS.items()}

REASON_LABELS = {
    "inventory_shortage": "재고 부족",
    "field_decision": "현장 판단",
    "transport_unavailable": "운송 불가",
    "time_shortage": "시간 부족",
    "store_request": "점포 요청",
    "other": "기타",
}
REASON_CODES_BY_LABEL = {label: code for code, label in REASON_LABELS.items()}

_QUANTITY_OUTCOME_FIELDS = {
    "post_source_stock",
    "post_destination_stock",
    "actual_sales_qty",
    "actual_waste_qty",
    "actual_stockout_qty",
}
_MONEY_OUTCOME_FIELDS = {"actual_transport_cost", "actual_saving"}
_OUTCOME_FIELDS = _QUANTITY_OUTCOME_FIELDS | _MONEY_OUTCOME_FIELDS
_FEATURE_FIELDS = (
    "net_benefit_score", "disposal_risk_score", "demand_fit_score",
    "inventory_balance_score", "route_cost_score", "feasibility_score",
    "demand_risk_score", "post_move_risk_score", "pareto_rank",
    "pareto_status", "robustness_status", "confidence_score",
)
_MAX_ACTUAL_QTY = 1_000_000_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def history_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    configured = str(os.environ.get("VARO_HISTORY_DB_PATH") or "").strip()
    return Path(configured) if configured else DEFAULT_HISTORY_DB


def _result(ok: bool, code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": ok, "code": code, "message": message, **extra}


def _number(value: Any, *, integer: bool = False, allow_none: bool = True) -> int | float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_none:
            return None
        raise ValueError("값이 필요합니다.")
    if isinstance(value, bool):
        raise ValueError("숫자 형식이 아닙니다.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("숫자 형식이 아닙니다.") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("0 이상의 유한한 값만 입력할 수 있습니다.")
    if integer:
        if not number.is_integer():
            raise ValueError("수량은 정수로 입력해주세요.")
        return int(number)
    return number


def _text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned[:limit] if cleaned else None


def _optional_bool(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "있음"}:
        return 1
    if normalized in {"0", "false", "no", "n", "없음"}:
        return 0
    raise ValueError("품절 여부 값을 확인해주세요.")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=8.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise sqlite3.DatabaseError("unsupported execution-history schema")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS execution_plans (
            plan_id TEXT PRIMARY KEY,
            algorithm_version TEXT NOT NULL,
            candidate_algorithm_version TEXT,
            data_signature TEXT NOT NULL,
            created_at TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            plan_status TEXT NOT NULL,
            total_actions INTEGER NOT NULL CHECK(total_actions >= 0),
            total_planned_qty INTEGER NOT NULL CHECK(total_planned_qty >= 0),
            expected_total_cost REAL,
            expected_total_saving REAL,
            expected_total_net_benefit REAL
        );

        CREATE TABLE IF NOT EXISTS execution_items (
            plan_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_algorithm_version TEXT,
            source_store_id TEXT NOT NULL,
            source_store_name TEXT,
            destination_store_id TEXT NOT NULL,
            destination_store_name TEXT,
            product_id TEXT NOT NULL,
            product_name TEXT,
            route_type TEXT NOT NULL,
            dc_id TEXT,
            planned_qty INTEGER NOT NULL CHECK(planned_qty > 0),
            expected_cost REAL,
            expected_saving REAL,
            expected_net_benefit REAL,
            vhs_score REAL,
            stability TEXT,
            confidence REAL,
            feature_snapshot_json TEXT NOT NULL DEFAULT '{}',
            execution_status TEXT NOT NULL DEFAULT 'unconfirmed',
            actual_qty INTEGER,
            nonexecution_reason TEXT,
            operator_note TEXT,
            post_source_stock REAL,
            post_destination_stock REAL,
            actual_sales_qty REAL,
            actual_waste_qty REAL,
            actual_stockout_occurred INTEGER CHECK(actual_stockout_occurred IN (0, 1)),
            actual_stockout_qty REAL,
            actual_transport_cost REAL,
            actual_saving REAL,
            actual_net_benefit REAL,
            outcome_recorded_at TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(plan_id, candidate_id),
            FOREIGN KEY(plan_id) REFERENCES execution_plans(plan_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS execution_item_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            changed_at TEXT NOT NULL,
            previous_status TEXT,
            new_status TEXT NOT NULL,
            previous_actual_qty INTEGER,
            new_actual_qty INTEGER,
            reason_code TEXT,
            note_snapshot TEXT,
            FOREIGN KEY(plan_id, candidate_id)
                REFERENCES execution_items(plan_id, candidate_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_execution_plans_recorded
            ON execution_plans(recorded_at DESC);
        CREATE INDEX IF NOT EXISTS idx_execution_items_status
            ON execution_items(execution_status);
        """
    )
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()


def initialize_history_store(db_path: str | Path | None = None) -> dict[str, Any]:
    try:
        with closing(_connect(history_db_path(db_path))) as connection:
            _ensure_schema(connection)
        return _result(True, "ready", "실행 기록 저장소가 준비되었습니다.", schema_version=SCHEMA_VERSION)
    except (OSError, sqlite3.Error):
        return _result(False, "storage_error", "실행 기록 저장소를 준비하지 못했습니다.")


def _plan_snapshot(plan: Mapping[str, Any], recorded_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_id = _text(plan.get("plan_id"), 160)
    data_signature = _text(plan.get("data_signature"), 256)
    algorithm_version = _text(plan.get("algorithm_version"), 120)
    created_at = _text(plan.get("created_at"), 80)
    raw_items = list(plan.get("items") or [])
    if not plan_id or not data_signature or not algorithm_version or not created_at or not raw_items:
        raise ValueError("기록할 수 있는 실행계획이 없습니다.")
    if not bool((plan.get("validation") or {}).get("valid")):
        raise ValueError("검증을 통과한 실행계획만 기록할 수 있습니다.")

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ValueError("실행계획 항목 형식을 확인해주세요.")
        candidate_id = _text(raw.get("candidate_id"), 160)
        source_id = _text(raw.get("source_id"), 160)
        target_id = _text(raw.get("target_id"), 160)
        product_id = _text(raw.get("product_id"), 160)
        route_type = _text(raw.get("route_type"), 40)
        planned_qty = _number(raw.get("planned_qty"), integer=True, allow_none=False)
        if not all((candidate_id, source_id, target_id, product_id, route_type)) or planned_qty is None or planned_qty <= 0:
            raise ValueError("실행계획 항목의 필수 값이 없습니다.")
        if candidate_id in seen:
            raise ValueError("중복 실행계획 항목이 있습니다.")
        seen.add(candidate_id)
        features = {key: raw.get(key) for key in _FEATURE_FIELDS if raw.get(key) is not None}
        items.append({
            "plan_id": plan_id,
            "candidate_id": candidate_id,
            "candidate_algorithm_version": _text(raw.get("algorithm_version"), 120),
            "source_store_id": source_id,
            "source_store_name": _text(raw.get("source_name"), 200),
            "destination_store_id": target_id,
            "destination_store_name": _text(raw.get("target_name"), 200),
            "product_id": product_id,
            "product_name": _text(raw.get("product_name"), 200),
            "route_type": route_type,
            "dc_id": _text(raw.get("dc_id"), 160),
            "planned_qty": planned_qty,
            "expected_cost": _number(raw.get("planned_cost")),
            "expected_saving": _number(raw.get("planned_expected_saving")),
            "expected_net_benefit": _number(raw.get("planned_net_benefit")),
            "vhs_score": _number(raw.get("vhs_score")),
            "stability": _text(raw.get("robustness_status") or raw.get("stability"), 80),
            "confidence": _number(raw.get("confidence_score") or raw.get("confidence")),
            "feature_snapshot_json": json.dumps(
                features, ensure_ascii=False, sort_keys=True,
                default=lambda value: value.item() if hasattr(value, "item") else str(value),
            ),
            "updated_at": recorded_at,
        })

    candidate_version = next(
        (item["candidate_algorithm_version"] for item in items if item["candidate_algorithm_version"]),
        None,
    )
    snapshot = {
        "plan_id": plan_id,
        "algorithm_version": algorithm_version,
        "candidate_algorithm_version": candidate_version,
        "data_signature": data_signature,
        "created_at": created_at,
        "recorded_at": recorded_at,
        "updated_at": recorded_at,
        "plan_status": _text(plan.get("plan_status"), 80) or "계산 불가",
        "total_actions": len(items),
        "total_planned_qty": sum(int(item["planned_qty"]) for item in items),
        "expected_total_cost": _number(plan.get("total_cost")),
        "expected_total_saving": _number(plan.get("total_expected_saving")),
        "expected_total_net_benefit": _number(plan.get("total_net_benefit")),
    }
    return snapshot, items


def _insert_plan(connection: sqlite3.Connection, plan: Mapping[str, Any]) -> None:
    columns = tuple(plan)
    connection.execute(
        f"INSERT INTO execution_plans ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        tuple(plan[column] for column in columns),
    )


def _insert_plan_items(connection: sqlite3.Connection, items: Sequence[Mapping[str, Any]]) -> None:
    columns = tuple(items[0])
    connection.executemany(
        f"INSERT INTO execution_items ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        [tuple(item[column] for column in columns) for item in items],
    )


def record_execution_plan(
    plan: Mapping[str, Any], db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically store a plan and its items once."""
    recorded_at = _utc_now()
    try:
        snapshot, items = _plan_snapshot(plan, recorded_at)
    except (ValueError, TypeError, AttributeError) as exc:
        return _result(False, "invalid_plan", str(exc))

    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(history_db_path(db_path))
        _ensure_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        _insert_plan(connection, snapshot)
        _insert_plan_items(connection, items)
        connection.commit()
        return _result(
            True, "recorded", "실행계획을 기록했습니다.",
            plan_id=snapshot["plan_id"], created=True,
        )
    except sqlite3.IntegrityError:
        if connection is not None:
            connection.rollback()
            existing = connection.execute(
                "SELECT 1 FROM execution_plans WHERE plan_id = ?",
                (snapshot["plan_id"],),
            ).fetchone()
            if existing is not None:
                return _result(
                    True, "duplicate", "이미 기록된 실행계획입니다.",
                    plan_id=snapshot["plan_id"], created=False,
                )
        return _result(False, "storage_error", "실행 기록을 저장하지 못했습니다.")
    except (OSError, sqlite3.Error):
        if connection is not None:
            connection.rollback()
        return _result(False, "storage_error", "실행 기록을 저장하지 못했습니다.")
    finally:
        if connection is not None:
            connection.close()


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def list_recorded_plans(
    db_path: str | Path | None = None, *, limit: int = 50,
) -> dict[str, Any]:
    path = history_db_path(db_path)
    if not path.exists():
        return _result(True, "loaded", "기록된 실행계획이 없습니다.", plans=[])
    try:
        safe_limit = max(1, min(int(limit), 100_000))
        with closing(_connect(path)) as connection:
            _ensure_schema(connection)
            rows = connection.execute(
                "SELECT * FROM execution_plans ORDER BY recorded_at DESC, plan_id LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return _result(True, "loaded", "실행 기록을 불러왔습니다.", plans=[_row_dict(row) for row in rows])
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return _result(False, "storage_error", "실행 기록을 불러오지 못했습니다.", plans=[])


def get_recorded_plan(plan_id: str, db_path: str | Path | None = None) -> dict[str, Any]:
    path = history_db_path(db_path)
    if not path.exists():
        return _result(False, "not_found", "기록된 실행계획을 찾지 못했습니다.", plan=None, items=[])
    try:
        with closing(_connect(path)) as connection:
            _ensure_schema(connection)
            plan_row = connection.execute(
                "SELECT * FROM execution_plans WHERE plan_id = ?", (str(plan_id),),
            ).fetchone()
            if plan_row is None:
                return _result(False, "not_found", "기록된 실행계획을 찾지 못했습니다.", plan=None, items=[])
            item_rows = connection.execute(
                "SELECT * FROM execution_items WHERE plan_id = ? ORDER BY source_store_name, destination_store_name, product_name, candidate_id",
                (str(plan_id),),
            ).fetchall()
        items = [_item_with_comparison(_row_dict(row)) for row in item_rows]
        return _result(True, "loaded", "실행계획을 불러왔습니다.", plan=_row_dict(plan_row), items=items)
    except (OSError, sqlite3.Error):
        return _result(False, "storage_error", "실행 기록을 불러오지 못했습니다.", plan=None, items=[])


def _item_with_comparison(item: dict[str, Any]) -> dict[str, Any]:
    item["execution_status_label"] = STATUS_LABELS.get(item.get("execution_status"), "확인 필요")
    item["nonexecution_reason_label"] = REASON_LABELS.get(item.get("nonexecution_reason"), "-")
    item["quantity_difference"] = (
        None if item.get("actual_qty") is None else item["actual_qty"] - item["planned_qty"]
    )
    item["cost_difference"] = (
        None if item.get("actual_transport_cost") is None or item.get("expected_cost") is None
        else item["actual_transport_cost"] - item["expected_cost"]
    )
    item["saving_difference"] = (
        None if item.get("actual_saving") is None or item.get("expected_saving") is None
        else item["actual_saving"] - item["expected_saving"]
    )
    item["net_benefit_difference"] = (
        None if item.get("actual_net_benefit") is None or item.get("expected_net_benefit") is None
        else item["actual_net_benefit"] - item["expected_net_benefit"]
    )
    return item


def update_execution_item(
    plan_id: str,
    candidate_id: str,
    execution_status: str,
    actual_qty: Any = None,
    *,
    nonexecution_reason: str | None = None,
    operator_note: str | None = None,
    outcomes: Mapping[str, Any] | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Update one recorded action and append a minimal audit event."""
    status = STATUS_CODES_BY_LABEL.get(execution_status, execution_status)
    if status not in STATUS_LABELS:
        return _result(False, "invalid_status", "실행 상태를 확인해주세요.")
    reason = REASON_CODES_BY_LABEL.get(nonexecution_reason or "", nonexecution_reason)
    if reason and reason not in REASON_LABELS:
        return _result(False, "invalid_reason", "미실행 사유를 확인해주세요.")
    if reason not in REASON_LABELS:
        reason = None
    outcomes_provided = outcomes is not None
    if outcomes is not None and not isinstance(outcomes, Mapping):
        return _result(False, "invalid_value", "사후 결과 형식을 확인해주세요.")
    try:
        quantity = _number(actual_qty, integer=True)
        if quantity is not None and quantity > _MAX_ACTUAL_QTY:
            raise ValueError("실제 수량 값이 너무 큽니다.")
        if status in {"executed", "partial"} and (quantity is None or quantity <= 0):
            raise ValueError("실행한 수량을 1개 이상의 정수로 입력해주세요.")
        if status in {"not_executed", "cancelled"}:
            if quantity not in (None, 0):
                raise ValueError("미실행 또는 취소 상태의 실제 수량은 0이어야 합니다.")
            quantity = 0
        if status == "unconfirmed":
            quantity = None
        clean_outcomes = {
            key: _number((outcomes or {}).get(key))
            for key in _OUTCOME_FIELDS
        }
        stockout_occurred = _optional_bool((outcomes or {}).get("actual_stockout_occurred"))
    except ValueError as exc:
        return _result(False, "invalid_value", str(exc))

    now = _utc_now()
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect(history_db_path(db_path))
        _ensure_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        current = connection.execute(
            "SELECT * FROM execution_items WHERE plan_id = ? AND candidate_id = ?",
            (str(plan_id), str(candidate_id)),
        ).fetchone()
        if current is None:
            connection.rollback()
            return _result(False, "not_found", "기록된 이동을 찾지 못했습니다.")
        if not outcomes_provided:
            clean_outcomes = {key: current[key] for key in _OUTCOME_FIELDS}
            stockout_occurred = current["actual_stockout_occurred"]
        actual_net = None
        if clean_outcomes["actual_saving"] is not None and clean_outcomes["actual_transport_cost"] is not None:
            actual_net = clean_outcomes["actual_saving"] - clean_outcomes["actual_transport_cost"]
        outcome_recorded_at = (
            now if stockout_occurred is not None or any(value is not None for value in clean_outcomes.values())
            else current["outcome_recorded_at"]
        )
        warning = None
        if quantity is not None and quantity > int(current["planned_qty"]):
            warning = f"계획보다 {quantity - int(current['planned_qty'])}개 많이 실행되었습니다."
        connection.execute(
            """
            UPDATE execution_items SET
                execution_status = ?, actual_qty = ?, nonexecution_reason = ?, operator_note = ?,
                post_source_stock = ?, post_destination_stock = ?, actual_sales_qty = ?,
                actual_waste_qty = ?, actual_stockout_occurred = ?, actual_stockout_qty = ?, actual_transport_cost = ?,
                actual_saving = ?, actual_net_benefit = ?, outcome_recorded_at = ?, updated_at = ?
            WHERE plan_id = ? AND candidate_id = ?
            """,
            (
                status, quantity, reason, _text(operator_note, 300),
                clean_outcomes["post_source_stock"], clean_outcomes["post_destination_stock"],
                clean_outcomes["actual_sales_qty"], clean_outcomes["actual_waste_qty"],
                stockout_occurred, clean_outcomes["actual_stockout_qty"], clean_outcomes["actual_transport_cost"],
                clean_outcomes["actual_saving"], actual_net, outcome_recorded_at, now,
                str(plan_id), str(candidate_id),
            ),
        )
        connection.execute(
            """
            INSERT INTO execution_item_events (
                plan_id, candidate_id, changed_at, previous_status, new_status,
                previous_actual_qty, new_actual_qty, reason_code, note_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(plan_id), str(candidate_id), now, current["execution_status"], status,
                current["actual_qty"], quantity, reason, _text(operator_note, 300),
            ),
        )
        connection.execute(
            "UPDATE execution_plans SET updated_at = ? WHERE plan_id = ?", (now, str(plan_id)),
        )
        connection.commit()
        return _result(True, "updated", "실행 기록을 저장했습니다.", warning=warning)
    except (OSError, sqlite3.Error):
        if connection is not None:
            connection.rollback()
        return _result(False, "storage_error", "실행 기록을 저장하지 못했습니다.")
    finally:
        if connection is not None:
            connection.close()


def execution_history_metrics(db_path: str | Path | None = None) -> dict[str, Any]:
    path = history_db_path(db_path)
    if not path.exists():
        return _result(
            True, "calculated", "실행 기록 요약을 계산했습니다.",
            total_items=0, confirmed_items=0, execution_rate=None,
            partial_rate=None, not_executed_rate=None,
            quantity_adherence_rate=None, quantity_sample_count=0,
            cost_error={"sample_count": 0, "mean_error": None, "mean_absolute_error": None},
            saving_error={"sample_count": 0, "mean_error": None, "mean_absolute_error": None},
            net_benefit_error={"sample_count": 0, "mean_error": None, "mean_absolute_error": None},
        )
    try:
        with closing(_connect(path)) as connection:
            _ensure_schema(connection)
            rows = [_row_dict(row) for row in connection.execute("SELECT * FROM execution_items").fetchall()]
    except (OSError, sqlite3.Error):
        return _result(False, "storage_error", "실행 기록 요약을 계산하지 못했습니다.")

    confirmed = [row for row in rows if row["execution_status"] != "unconfirmed"]
    executed = [row for row in confirmed if row["execution_status"] in {"executed", "partial"}]
    quantity_rows = [row for row in confirmed if row["actual_qty"] is not None]

    def rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else round(numerator / denominator * 100.0, 2)

    def error_metrics(actual: str, expected: str) -> dict[str, Any]:
        values = [
            float(row[actual]) - float(row[expected])
            for row in rows if row[actual] is not None and row[expected] is not None
        ]
        return {
            "sample_count": len(values),
            "mean_error": None if not values else round(sum(values) / len(values), 6),
            "mean_absolute_error": None if not values else round(sum(abs(value) for value in values) / len(values), 6),
        }

    actual_total = sum(int(row["actual_qty"]) for row in quantity_rows)
    planned_total = sum(int(row["planned_qty"]) for row in quantity_rows)
    return _result(
        True, "calculated", "실행 기록 요약을 계산했습니다.",
        total_items=len(rows), confirmed_items=len(confirmed),
        execution_rate=rate(len(executed), len(confirmed)),
        partial_rate=rate(sum(row["execution_status"] == "partial" for row in confirmed), len(confirmed)),
        not_executed_rate=rate(sum(row["execution_status"] in {"not_executed", "cancelled"} for row in confirmed), len(confirmed)),
        quantity_adherence_rate=None if not planned_total else round(actual_total / planned_total * 100.0, 2),
        quantity_sample_count=len(quantity_rows),
        cost_error=error_metrics("actual_transport_cost", "expected_cost"),
        saving_error=error_metrics("actual_saving", "expected_saving"),
        net_benefit_error=error_metrics("actual_net_benefit", "expected_net_benefit"),
    )


def export_execution_history_csv(db_path: str | Path | None = None) -> dict[str, Any]:
    """Return a user-triggered UTF-8 BOM calibration export; never includes DB paths."""
    plans_result = list_recorded_plans(db_path, limit=100_000)
    if not plans_result["ok"]:
        return _result(False, "storage_error", "실행 기록을 내보내지 못했습니다.", data=b"")
    rows: list[dict[str, Any]] = []
    for plan in plans_result["plans"]:
        loaded = get_recorded_plan(plan["plan_id"], db_path)
        if not loaded["ok"]:
            continue
        for item in loaded["items"]:
            try:
                features = json.loads(item.get("feature_snapshot_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                features = {}
            rows.append({
                "plan_id": plan["plan_id"], "candidate_id": item["candidate_id"],
                "plan_algorithm_version": plan["algorithm_version"],
                "candidate_algorithm_version": item.get("candidate_algorithm_version") or plan.get("candidate_algorithm_version"),
                "data_signature": plan["data_signature"], "plan_created_at": plan["created_at"],
                "recorded_at": plan["recorded_at"], "source_store": item.get("source_store_name") or item["source_store_id"],
                "destination_store": item.get("destination_store_name") or item["destination_store_id"],
                "product_id": item["product_id"], "product": item.get("product_name"),
                "route_type": item["route_type"], "dc_id": item.get("dc_id"),
                "planned_qty": item["planned_qty"], "actual_qty": item.get("actual_qty"),
                "execution_status": item["execution_status_label"],
                "nonexecution_reason": item["nonexecution_reason_label"],
                "expected_cost": item.get("expected_cost"), "actual_transport_cost": item.get("actual_transport_cost"),
                "expected_saving": item.get("expected_saving"), "actual_saving": item.get("actual_saving"),
                "expected_net_benefit": item.get("expected_net_benefit"), "actual_net_benefit": item.get("actual_net_benefit"),
                "post_source_stock": item.get("post_source_stock"), "post_destination_stock": item.get("post_destination_stock"),
                "actual_sales_qty": item.get("actual_sales_qty"), "actual_waste_qty": item.get("actual_waste_qty"),
                "actual_stockout_occurred": (
                    "" if item.get("actual_stockout_occurred") is None
                    else "있음" if item.get("actual_stockout_occurred") else "없음"
                ),
                "actual_stockout_qty": item.get("actual_stockout_qty"), "vhs_score": item.get("vhs_score"),
                "stability": item.get("stability"), "confidence": item.get("confidence"),
                **{key: features.get(key) for key in _FEATURE_FIELDS},
            })
    if not rows:
        return _result(True, "empty", "내보낼 실행 기록이 없습니다.", data=b"", row_count=0)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return _result(
        True, "exported", "실행 기록을 내보냈습니다.",
        data=stream.getvalue().encode("utf-8-sig"), row_count=len(rows),
    )


def list_item_events(
    plan_id: str, candidate_id: str, db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Small audit hook used by validation/tests; not exposed in the basic UI."""
    path = history_db_path(db_path)
    if not path.exists():
        return []
    try:
        with closing(_connect(path)) as connection:
            _ensure_schema(connection)
            rows = connection.execute(
                "SELECT * FROM execution_item_events WHERE plan_id = ? AND candidate_id = ? ORDER BY event_id",
                (str(plan_id), str(candidate_id)),
            ).fetchall()
        return [_row_dict(row) for row in rows]
    except (OSError, sqlite3.Error):
        return []
