"""Backend-neutral persistence adapters for execution history."""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

from services.execution_history_config import (
    DEFAULT_CONNECT_RETRIES,
    DEFAULT_CONNECT_TIMEOUT,
    ExecutionHistoryConfig,
    load_execution_history_config,
)


SCHEMA_VERSION = 1

PLAN_COLUMNS = (
    "plan_id", "algorithm_version", "candidate_algorithm_version", "data_signature",
    "created_at", "recorded_at", "updated_at", "plan_status", "total_actions",
    "total_planned_qty", "expected_total_cost", "expected_total_saving",
    "expected_total_net_benefit",
)
ITEM_COLUMNS = (
    "plan_id", "candidate_id", "candidate_algorithm_version", "source_store_id",
    "source_store_name", "destination_store_id", "destination_store_name", "product_id",
    "product_name", "route_type", "dc_id", "planned_qty", "expected_cost",
    "expected_saving", "expected_net_benefit", "vhs_score", "stability", "confidence",
    "feature_snapshot_json", "execution_status", "actual_qty", "nonexecution_reason",
    "operator_note", "post_source_stock", "post_destination_stock", "actual_sales_qty",
    "actual_waste_qty", "actual_stockout_occurred", "actual_stockout_qty",
    "actual_transport_cost", "actual_saving", "actual_net_benefit", "outcome_recorded_at",
    "updated_at",
)
ITEM_SNAPSHOT_COLUMNS = (
    "plan_id", "candidate_id", "candidate_algorithm_version", "source_store_id",
    "source_store_name", "destination_store_id", "destination_store_name", "product_id",
    "product_name", "route_type", "dc_id", "planned_qty", "expected_cost",
    "expected_saving", "expected_net_benefit", "vhs_score", "stability", "confidence",
    "feature_snapshot_json", "updated_at",
)
EVENT_COLUMNS = (
    "plan_id", "candidate_id", "changed_at", "previous_status", "new_status",
    "previous_actual_qty", "new_actual_qty", "reason_code", "note_snapshot",
)
OUTCOME_FIELDS = (
    "post_source_stock", "post_destination_stock", "actual_sales_qty", "actual_waste_qty",
    "actual_stockout_qty", "actual_transport_cost", "actual_saving",
)


class HistoryStoreError(RuntimeError):
    """Safe persistence error.  The original DB exception is never user-facing."""


class DuplicatePlanError(HistoryStoreError):
    """Raised when a plan is already present in the selected backend."""


class HistoryItemNotFoundError(HistoryStoreError):
    """Raised when an update target no longer exists."""


class ExecutionHistoryStore(Protocol):
    backend: str

    def initialize(self) -> None: ...
    def save_plan(self, plan: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> None: ...
    def list_plans(self, *, limit: int, offset: int = 0) -> list[dict[str, Any]]: ...
    def get_plan(self, plan_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]: ...
    def update_execution_result(self, plan_id: str, candidate_id: str, **values: Any) -> int: ...
    def metrics_summary(self) -> dict[str, Any]: ...
    def export_rows(self) -> list[dict[str, Any]]: ...
    def list_events(self, plan_id: str, candidate_id: str) -> list[dict[str, Any]]: ...
    def read_snapshot(self) -> dict[str, list[dict[str, Any]]]: ...
    def existing_plan_ids(self, *, initialize: bool = True) -> set[str]: ...
    def import_snapshot(self, snapshot: Mapping[str, Sequence[Mapping[str, Any]]], *, skip_plan_ids: set[str]) -> dict[str, int]: ...
    def inspect_schema(self) -> dict[str, Any]: ...
    def health_check(self) -> dict[str, Any]: ...
    def delete_plans(self, plan_ids: Sequence[str]) -> dict[str, int]: ...


SQLITE_SCHEMA = """
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
CREATE INDEX IF NOT EXISTS idx_execution_item_events_item
    ON execution_item_events(plan_id, candidate_id);
"""

POSTGRES_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS execution_history_schema_meta (
        meta_key TEXT PRIMARY KEY,
        schema_version INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS execution_plans (
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
        expected_total_cost DOUBLE PRECISION,
        expected_total_saving DOUBLE PRECISION,
        expected_total_net_benefit DOUBLE PRECISION
    )""",
    """CREATE TABLE IF NOT EXISTS execution_items (
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
        expected_cost DOUBLE PRECISION,
        expected_saving DOUBLE PRECISION,
        expected_net_benefit DOUBLE PRECISION,
        vhs_score DOUBLE PRECISION,
        stability TEXT,
        confidence DOUBLE PRECISION,
        feature_snapshot_json TEXT NOT NULL DEFAULT '{}',
        execution_status TEXT NOT NULL DEFAULT 'unconfirmed',
        actual_qty INTEGER,
        nonexecution_reason TEXT,
        operator_note TEXT,
        post_source_stock DOUBLE PRECISION,
        post_destination_stock DOUBLE PRECISION,
        actual_sales_qty DOUBLE PRECISION,
        actual_waste_qty DOUBLE PRECISION,
        actual_stockout_occurred INTEGER CHECK(actual_stockout_occurred IN (0, 1)),
        actual_stockout_qty DOUBLE PRECISION,
        actual_transport_cost DOUBLE PRECISION,
        actual_saving DOUBLE PRECISION,
        actual_net_benefit DOUBLE PRECISION,
        outcome_recorded_at TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(plan_id, candidate_id),
        FOREIGN KEY(plan_id) REFERENCES execution_plans(plan_id) ON DELETE RESTRICT
    )""",
    """CREATE TABLE IF NOT EXISTS execution_item_events (
        event_id BIGSERIAL PRIMARY KEY,
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
    )""",
    "CREATE INDEX IF NOT EXISTS idx_execution_plans_recorded ON execution_plans(recorded_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_execution_items_status ON execution_items(execution_status)",
    "CREATE INDEX IF NOT EXISTS idx_execution_item_events_item "
    "ON execution_item_events(plan_id, candidate_id)",
)

# Structural expectations checked by ``inspect_schema`` and the operator CLI.
EXPECTED_SCHEMA = {
    "execution_plans": {
        "columns": frozenset(PLAN_COLUMNS),
        "primary_key": ("plan_id",),
        "references": frozenset(),
    },
    "execution_items": {
        "columns": frozenset(ITEM_COLUMNS),
        "primary_key": ("plan_id", "candidate_id"),
        "references": frozenset({"execution_plans"}),
    },
    "execution_item_events": {
        "columns": frozenset(EVENT_COLUMNS) | {"event_id"},
        "primary_key": ("event_id",),
        "references": frozenset({"execution_items"}),
    },
}
EXPECTED_INDEXES = (
    "idx_execution_plans_recorded",
    "idx_execution_items_status",
    "idx_execution_item_events_item",
)


def evaluate_schema(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Compare an observed structure against ``EXPECTED_SCHEMA``.

    Returns only structural facts.  No connection string, host, or credential
    ever reaches this layer, so the result is safe to print.
    """
    tables = dict(observed.get("tables") or {})
    indexes = {str(name) for name in observed.get("indexes") or ()}
    issues: list[str] = []
    table_report: dict[str, Any] = {}
    for name, expected in EXPECTED_SCHEMA.items():
        found = tables.get(name)
        if not found:
            issues.append(f"필수 테이블 누락: {name}")
            table_report[name] = {"present": False}
            continue
        columns = {str(column) for column in found.get("columns") or ()}
        primary_key = tuple(str(column) for column in found.get("primary_key") or ())
        references = {str(value) for value in found.get("references") or ()}
        missing_columns = sorted(set(expected["columns"]) - columns)
        missing_references = sorted(set(expected["references"]) - references)
        if missing_columns:
            issues.append(f"{name} 컬럼 누락: {', '.join(missing_columns)}")
        if primary_key != tuple(expected["primary_key"]):
            issues.append(f"{name} 기본키 불일치")
        if missing_references:
            issues.append(f"{name} 외래키 누락: {', '.join(missing_references)}")
        table_report[name] = {
            "present": True,
            "missing_columns": missing_columns,
            "primary_key": list(primary_key),
            "primary_key_ok": primary_key == tuple(expected["primary_key"]),
            "references": sorted(references),
            "missing_references": missing_references,
        }
    index_report = {name: name in indexes for name in EXPECTED_INDEXES}
    missing_indexes = sorted(name for name, present in index_report.items() if not present)
    if missing_indexes:
        issues.append(f"인덱스 누락: {', '.join(missing_indexes)}")

    version = observed.get("schema_version")
    if version is None:
        issues.append("스키마 버전을 확인하지 못했습니다.")
    elif int(version) != SCHEMA_VERSION:
        issues.append(f"스키마 버전 불일치: {int(version)} (기대 {SCHEMA_VERSION})")
    return {
        "ok": not issues,
        "schema_version": None if version is None else int(version),
        "expected_schema_version": SCHEMA_VERSION,
        "tables": table_report,
        "indexes": index_report,
        "issues": issues,
    }


def _row_dict(row: Any, cursor: Any | None = None) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    description = getattr(cursor, "description", None) or ()
    names = [column.name if hasattr(column, "name") else column[0] for column in description]
    return dict(zip(names, row))


class _DBAPIExecutionHistoryStore:
    backend = "unknown"
    placeholder = "?"

    def __init__(self) -> None:
        self._schema_ready = False

    def _open(self, *, read_only: bool = False) -> Any:
        raise NotImplementedError

    def _storage_missing(self) -> bool:
        return False

    def _adapt(self, sql: str) -> str:
        return sql if self.placeholder == "?" else sql.replace("?", self.placeholder)

    def _execute(self, connection: Any, sql: str, params: Sequence[Any] = ()) -> Any:
        try:
            return connection.execute(self._adapt(sql), tuple(params))
        except Exception as error:
            if self._is_missing_table(error):
                self._schema_ready = False
            raise

    def _executemany(self, connection: Any, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        cursor = connection.cursor()
        try:
            try:
                cursor.executemany(self._adapt(sql), rows)
            except Exception as error:
                if self._is_missing_table(error):
                    self._schema_ready = False
                raise
        finally:
            cursor.close()

    def _begin(self, connection: Any) -> None:
        self._execute(connection, "BEGIN")

    def _ensure_schema(self, connection: Any) -> None:
        raise NotImplementedError

    def _observe_schema(self, connection: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _read_schema_version(self, connection: Any) -> int | None:
        raise NotImplementedError

    def _is_unique_violation(self, error: Exception) -> bool:
        return False

    def _is_missing_table(self, error: Exception) -> bool:
        return False

    def _close(self, connection: Any | None) -> None:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def initialize(self) -> None:
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
        except HistoryStoreError:
            raise
        except Exception:
            raise HistoryStoreError("실행 기록 저장소를 준비하지 못했습니다.") from None
        finally:
            self._close(connection)

    def inspect_schema(self) -> dict[str, Any]:
        """Report structure only; the caller may print the result verbatim."""
        if self._storage_missing():
            return {"backend": self.backend, **evaluate_schema({})}
        connection = None
        try:
            connection = self._open()
            observed = self._observe_schema(connection)
        except HistoryStoreError:
            raise
        except Exception:
            raise HistoryStoreError("실행 기록 스키마를 확인하지 못했습니다.") from None
        finally:
            self._close(connection)
        return {"backend": self.backend, **evaluate_schema(observed)}

    def health_check(self) -> dict[str, Any]:
        """Never raise: a dead database must not disable the rest of the app."""
        if self._storage_missing():
            return {
                "backend": self.backend, "ok": False, "connection_ok": True,
                "schema_version": None, "expected_schema_version": SCHEMA_VERSION,
                "latency_ms": None, "message": "실행 기록 저장소가 아직 초기화되지 않았습니다.",
            }
        started = perf_counter()
        connection = None
        version: int | None = None
        try:
            connection = self._open()
            cursor = self._execute(connection, "SELECT 1 AS ping")
            cursor.fetchone()
            version = self._read_schema_version(connection)
        except Exception:
            return {
                "backend": self.backend, "ok": False, "connection_ok": False,
                "schema_version": None, "expected_schema_version": SCHEMA_VERSION,
                "latency_ms": None, "message": "실행 기록 저장소에 연결하지 못했습니다.",
            }
        finally:
            self._close(connection)
        matched = version is not None and int(version) == SCHEMA_VERSION
        return {
            "backend": self.backend, "ok": matched, "connection_ok": True,
            "schema_version": version, "expected_schema_version": SCHEMA_VERSION,
            "latency_ms": round((perf_counter() - started) * 1000.0, 2),
            "message": "실행 기록 저장소가 정상입니다." if matched else "실행 기록 스키마 버전을 확인해주세요.",
        }

    def delete_plans(self, plan_ids: Sequence[str]) -> dict[str, int]:
        """Maintenance-only removal of the named plans; never called by the app.

        Used by staging validation cleanup so that test records do not linger.
        Only the exact identifiers passed in are removed, in child-first order.
        """
        identifiers = sorted({str(value) for value in plan_ids if str(value or "").strip()})
        if not identifiers:
            return {"plans": 0, "items": 0, "events": 0}
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            self._begin(connection)
            placeholders = ",".join("?" for _ in identifiers)
            removed: dict[str, int] = {}
            for key, table in (
                ("events", "execution_item_events"),
                ("items", "execution_items"),
                ("plans", "execution_plans"),
            ):
                cursor = self._execute(
                    connection,
                    f"SELECT COUNT(*) AS row_count FROM {table} WHERE plan_id IN ({placeholders})",
                    identifiers,
                )
                removed[key] = int(_row_dict(cursor.fetchone(), cursor)["row_count"])
                self._execute(
                    connection, f"DELETE FROM {table} WHERE plan_id IN ({placeholders})", identifiers,
                )
            connection.commit()
            return removed
        except Exception:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise HistoryStoreError("실행 기록을 정리하지 못했습니다.") from None
        finally:
            self._close(connection)

    @staticmethod
    def _insert_sql(table: str, columns: Sequence[str]) -> str:
        return f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"

    def _insert_mapping(self, connection: Any, table: str, columns: Sequence[str], value: Mapping[str, Any]) -> None:
        self._execute(connection, self._insert_sql(table, columns), [value.get(column) for column in columns])

    def _insert_mappings(
        self, connection: Any, table: str, columns: Sequence[str], values: Sequence[Mapping[str, Any]],
    ) -> None:
        if not values:
            return
        self._executemany(
            connection,
            self._insert_sql(table, columns),
            [[value.get(column) for column in columns] for value in values],
        )

    def _plan_exists(self, connection: Any, plan_id: str) -> bool:
        cursor = self._execute(connection, "SELECT 1 AS present FROM execution_plans WHERE plan_id = ?", (plan_id,))
        return cursor.fetchone() is not None

    def save_plan(self, plan: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> None:
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            self._begin(connection)
            self._insert_mapping(connection, "execution_plans", PLAN_COLUMNS, plan)
            self._insert_mappings(connection, "execution_items", ITEM_SNAPSHOT_COLUMNS, items)
            connection.commit()
        except Exception as error:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            if self._is_unique_violation(error) and connection is not None:
                try:
                    if self._plan_exists(connection, str(plan.get("plan_id") or "")):
                        raise DuplicatePlanError("이미 기록된 실행계획입니다.") from None
                except DuplicatePlanError:
                    raise
                except Exception:
                    pass
            raise HistoryStoreError("실행 기록을 저장하지 못했습니다.") from None
        finally:
            self._close(connection)

    def list_plans(self, *, limit: int, offset: int = 0) -> list[dict[str, Any]]:
        if self._storage_missing():
            return []
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            cursor = self._execute(
                connection,
                "SELECT * FROM execution_plans ORDER BY recorded_at DESC, plan_id LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [_row_dict(row, cursor) for row in cursor.fetchall()]
        except Exception:
            raise HistoryStoreError("실행 기록을 불러오지 못했습니다.") from None
        finally:
            self._close(connection)

    def get_plan(self, plan_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if self._storage_missing():
            return None, []
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            plan_cursor = self._execute(connection, "SELECT * FROM execution_plans WHERE plan_id = ?", (plan_id,))
            plan_row = plan_cursor.fetchone()
            if plan_row is None:
                return None, []
            item_cursor = self._execute(
                connection,
                "SELECT * FROM execution_items WHERE plan_id = ? "
                "ORDER BY source_store_name, destination_store_name, product_name, candidate_id",
                (plan_id,),
            )
            return _row_dict(plan_row, plan_cursor), [_row_dict(row, item_cursor) for row in item_cursor.fetchall()]
        except Exception:
            raise HistoryStoreError("실행 기록을 불러오지 못했습니다.") from None
        finally:
            self._close(connection)

    def update_execution_result(self, plan_id: str, candidate_id: str, **values: Any) -> int:
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            self._begin(connection)
            cursor = self._execute(
                connection,
                "SELECT * FROM execution_items WHERE plan_id = ? AND candidate_id = ?" + self._row_lock_clause(),
                (plan_id, candidate_id),
            )
            current_row = cursor.fetchone()
            if current_row is None:
                connection.rollback()
                raise HistoryItemNotFoundError("기록된 이동을 찾지 못했습니다.")
            current = _row_dict(current_row, cursor)
            outcomes = dict(values["outcomes"])
            if not values["outcomes_provided"]:
                outcomes = {key: current.get(key) for key in OUTCOME_FIELDS}
                values["stockout_occurred"] = current.get("actual_stockout_occurred")
            actual_net = None
            if outcomes["actual_saving"] is not None and outcomes["actual_transport_cost"] is not None:
                actual_net = outcomes["actual_saving"] - outcomes["actual_transport_cost"]
            has_outcome = values["stockout_occurred"] is not None or any(
                value is not None for value in outcomes.values()
            )
            outcome_recorded_at = values["now"] if has_outcome else current.get("outcome_recorded_at")
            self._execute(
                connection,
                """UPDATE execution_items SET
                    execution_status = ?, actual_qty = ?, nonexecution_reason = ?, operator_note = ?,
                    post_source_stock = ?, post_destination_stock = ?, actual_sales_qty = ?,
                    actual_waste_qty = ?, actual_stockout_occurred = ?, actual_stockout_qty = ?,
                    actual_transport_cost = ?, actual_saving = ?, actual_net_benefit = ?,
                    outcome_recorded_at = ?, updated_at = ?
                WHERE plan_id = ? AND candidate_id = ?""",
                (
                    values["status"], values["quantity"], values["reason"], values["operator_note"],
                    outcomes["post_source_stock"], outcomes["post_destination_stock"],
                    outcomes["actual_sales_qty"], outcomes["actual_waste_qty"],
                    values["stockout_occurred"], outcomes["actual_stockout_qty"],
                    outcomes["actual_transport_cost"], outcomes["actual_saving"], actual_net,
                    outcome_recorded_at, values["now"], plan_id, candidate_id,
                ),
            )
            self._execute(
                connection,
                """INSERT INTO execution_item_events (
                    plan_id, candidate_id, changed_at, previous_status, new_status,
                    previous_actual_qty, new_actual_qty, reason_code, note_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    plan_id, candidate_id, values["now"], current.get("execution_status"),
                    values["status"], current.get("actual_qty"), values["quantity"],
                    values["reason"], values["operator_note"],
                ),
            )
            self._execute(
                connection, "UPDATE execution_plans SET updated_at = ? WHERE plan_id = ?",
                (values["now"], plan_id),
            )
            connection.commit()
            return int(current["planned_qty"])
        except HistoryItemNotFoundError:
            raise
        except Exception:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise HistoryStoreError("실행 기록을 저장하지 못했습니다.") from None
        finally:
            self._close(connection)

    def _row_lock_clause(self) -> str:
        return ""

    def metrics_summary(self) -> dict[str, Any]:
        empty = {
            "total_items": 0, "confirmed_items": 0, "executed_items": 0,
            "partial_items": 0, "not_executed_items": 0, "quantity_sample_count": 0,
            "actual_qty_total": 0, "planned_qty_sample_total": 0,
            "cost_error": {"sample_count": 0, "mean_error": None, "mean_absolute_error": None},
            "saving_error": {"sample_count": 0, "mean_error": None, "mean_absolute_error": None},
            "net_benefit_error": {"sample_count": 0, "mean_error": None, "mean_absolute_error": None},
        }
        if self._storage_missing():
            return empty
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            cursor = self._execute(
                connection,
                """SELECT
                    COUNT(*) AS total_items,
                    COALESCE(SUM(CASE WHEN execution_status <> 'unconfirmed' THEN 1 ELSE 0 END), 0) AS confirmed_items,
                    COALESCE(SUM(CASE WHEN execution_status IN ('executed','partial') THEN 1 ELSE 0 END), 0) AS executed_items,
                    COALESCE(SUM(CASE WHEN execution_status = 'partial' THEN 1 ELSE 0 END), 0) AS partial_items,
                    COALESCE(SUM(CASE WHEN execution_status IN ('not_executed','cancelled') THEN 1 ELSE 0 END), 0) AS not_executed_items,
                    COALESCE(SUM(CASE WHEN execution_status <> 'unconfirmed' AND actual_qty IS NOT NULL THEN 1 ELSE 0 END), 0) AS quantity_sample_count,
                    COALESCE(SUM(CASE WHEN execution_status <> 'unconfirmed' AND actual_qty IS NOT NULL THEN actual_qty ELSE 0 END), 0) AS actual_qty_total,
                    COALESCE(SUM(CASE WHEN execution_status <> 'unconfirmed' AND actual_qty IS NOT NULL THEN planned_qty ELSE 0 END), 0) AS planned_qty_sample_total
                FROM execution_items""",
            )
            summary = _row_dict(cursor.fetchone(), cursor)
            for name, actual, expected in (
                ("cost_error", "actual_transport_cost", "expected_cost"),
                ("saving_error", "actual_saving", "expected_saving"),
                ("net_benefit_error", "actual_net_benefit", "expected_net_benefit"),
            ):
                error_cursor = self._execute(
                    connection,
                    f"""SELECT COUNT(*) AS sample_count,
                        AVG({actual} - {expected}) AS mean_error,
                        AVG(ABS({actual} - {expected})) AS mean_absolute_error
                    FROM execution_items WHERE {actual} IS NOT NULL AND {expected} IS NOT NULL""",
                )
                summary[name] = _row_dict(error_cursor.fetchone(), error_cursor)
            return {**empty, **summary}
        except Exception:
            raise HistoryStoreError("실행 기록 요약을 계산하지 못했습니다.") from None
        finally:
            self._close(connection)

    def export_rows(self) -> list[dict[str, Any]]:
        if self._storage_missing():
            return []
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            item_columns = ", ".join(f"i.{column}" for column in ITEM_COLUMNS if column != "plan_id")
            cursor = self._execute(
                connection,
                f"""SELECT
                    p.plan_id, p.algorithm_version AS plan_algorithm_version,
                    p.candidate_algorithm_version AS plan_candidate_algorithm_version,
                    p.data_signature, p.created_at AS plan_created_at, p.recorded_at,
                    {item_columns}
                FROM execution_plans p
                JOIN execution_items i ON i.plan_id = p.plan_id
                ORDER BY p.recorded_at DESC, p.plan_id, i.source_store_name,
                    i.destination_store_name, i.product_name, i.candidate_id""",
            )
            return [_row_dict(row, cursor) for row in cursor.fetchall()]
        except Exception:
            raise HistoryStoreError("실행 기록을 내보내지 못했습니다.") from None
        finally:
            self._close(connection)

    def list_events(self, plan_id: str, candidate_id: str) -> list[dict[str, Any]]:
        if self._storage_missing():
            return []
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            cursor = self._execute(
                connection,
                "SELECT * FROM execution_item_events WHERE plan_id = ? AND candidate_id = ? ORDER BY event_id",
                (plan_id, candidate_id),
            )
            return [_row_dict(row, cursor) for row in cursor.fetchall()]
        except Exception:
            raise HistoryStoreError("실행 기록을 불러오지 못했습니다.") from None
        finally:
            self._close(connection)

    def read_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        if self._storage_missing():
            return {"plans": [], "items": [], "events": []}
        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            result: dict[str, list[dict[str, Any]]] = {}
            for key, table, order in (
                ("plans", "execution_plans", "plan_id"),
                ("items", "execution_items", "plan_id, candidate_id"),
                ("events", "execution_item_events", "event_id"),
            ):
                cursor = self._execute(connection, f"SELECT * FROM {table} ORDER BY {order}")
                result[key] = [_row_dict(row, cursor) for row in cursor.fetchall()]
            return result
        except Exception:
            raise HistoryStoreError("실행 기록을 불러오지 못했습니다.") from None
        finally:
            self._close(connection)

    def existing_plan_ids(self, *, initialize: bool = True) -> set[str]:
        if self._storage_missing():
            return set()
        connection = None
        try:
            connection = self._open()
            if initialize:
                self._ensure_schema(connection)
            cursor = self._execute(connection, "SELECT plan_id FROM execution_plans")
            return {str(_row_dict(row, cursor)["plan_id"]) for row in cursor.fetchall()}
        except Exception as error:
            if not initialize and self._is_missing_table(error):
                if connection is not None:
                    try:
                        connection.rollback()
                    except Exception:
                        pass
                return set()
            raise HistoryStoreError("실행 기록을 확인하지 못했습니다.") from None
        finally:
            self._close(connection)

    def import_snapshot(
        self,
        snapshot: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        skip_plan_ids: set[str],
    ) -> dict[str, int]:
        plans = [dict(row) for row in snapshot.get("plans", ()) if str(row.get("plan_id")) not in skip_plan_ids]
        new_ids = {str(row["plan_id"]) for row in plans}
        items = [dict(row) for row in snapshot.get("items", ()) if str(row.get("plan_id")) in new_ids]
        events = [dict(row) for row in snapshot.get("events", ()) if str(row.get("plan_id")) in new_ids]
        if not plans:
            return {"plans": 0, "items": 0, "events": 0}

        connection = None
        try:
            connection = self._open()
            self._ensure_schema(connection)
            self._begin(connection)
            self._insert_mappings(connection, "execution_plans", PLAN_COLUMNS, plans)
            self._insert_mappings(connection, "execution_items", ITEM_COLUMNS, items)
            self._insert_mappings(connection, "execution_item_events", EVENT_COLUMNS, events)
            placeholders = ",".join("?" for _ in new_ids)
            identifiers = sorted(new_ids)
            actual_counts: dict[str, int] = {}
            for key, table in (
                ("plans", "execution_plans"),
                ("items", "execution_items"),
                ("events", "execution_item_events"),
            ):
                cursor = self._execute(
                    connection,
                    f"SELECT COUNT(*) AS row_count FROM {table} WHERE plan_id IN ({placeholders})",
                    identifiers,
                )
                actual_counts[key] = int(_row_dict(cursor.fetchone(), cursor)["row_count"])
            expected_counts = {"plans": len(plans), "items": len(items), "events": len(events)}
            if actual_counts != expected_counts:
                raise HistoryStoreError("이관 건수 검증에 실패했습니다.")
            connection.commit()
            return actual_counts
        except HistoryStoreError:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise
        except Exception:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise HistoryStoreError("실행 기록을 이관하지 못했습니다.") from None
        finally:
            self._close(connection)


class SQLiteExecutionHistoryStore(_DBAPIExecutionHistoryStore):
    backend = "sqlite"
    placeholder = "?"

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)

    def __repr__(self) -> str:
        return "SQLiteExecutionHistoryStore(backend='sqlite')"

    def _storage_missing(self) -> bool:
        return not self.path.exists()

    def _open(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=8.0)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.path), timeout=8.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _begin(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            try:
                connection.execute("SELECT 1 FROM execution_plans LIMIT 1")
                connection.commit()
                return
            except sqlite3.OperationalError:
                self._schema_ready = False
                connection.rollback()
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise HistoryStoreError("지원하지 않는 실행 기록 스키마입니다.")
        connection.executescript(SQLITE_SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
        self._schema_ready = True

    def _read_schema_version(self, connection: sqlite3.Connection) -> int | None:
        row = connection.execute("PRAGMA user_version").fetchone()
        return None if row is None else int(row[0])

    def _observe_schema(self, connection: sqlite3.Connection) -> dict[str, Any]:
        tables: dict[str, Any] = {}
        indexes: set[str] = set()
        known = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        for table in EXPECTED_SCHEMA:
            if table not in known:
                continue
            columns = list(connection.execute(f"PRAGMA table_info({table})"))
            primary_key = [row[1] for row in sorted((row for row in columns if row[5]), key=lambda row: row[5])]
            tables[table] = {
                "columns": {row[1] for row in columns},
                "primary_key": primary_key,
                "references": {row[2] for row in connection.execute(f"PRAGMA foreign_key_list({table})")},
            }
            indexes.update(
                str(row[1]) for row in connection.execute(f"PRAGMA index_list({table})")
            )
        return {
            "tables": tables,
            "indexes": indexes,
            "schema_version": self._read_schema_version(connection),
        }

    def _is_unique_violation(self, error: Exception) -> bool:
        return isinstance(error, sqlite3.IntegrityError) and "UNIQUE" in str(error).upper()

    def _is_missing_table(self, error: Exception) -> bool:
        return isinstance(error, sqlite3.OperationalError) and "NO SUCH TABLE" in str(error).upper()

    def read_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        if self._storage_missing():
            return {"plans": [], "items": [], "events": []}
        connection = None
        try:
            connection = self._open(read_only=True)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != SCHEMA_VERSION:
                raise HistoryStoreError("SQLite 원본 스키마 버전을 확인해주세요.")
            result: dict[str, list[dict[str, Any]]] = {}
            for key, table, order in (
                ("plans", "execution_plans", "plan_id"),
                ("items", "execution_items", "plan_id, candidate_id"),
                ("events", "execution_item_events", "event_id"),
            ):
                cursor = connection.execute(f"SELECT * FROM {table} ORDER BY {order}")
                result[key] = [_row_dict(row, cursor) for row in cursor.fetchall()]
            return result
        except HistoryStoreError:
            raise
        except Exception:
            raise HistoryStoreError("SQLite 원본 실행 기록을 읽지 못했습니다.") from None
        finally:
            self._close(connection)


class PostgreSQLExecutionHistoryStore(_DBAPIExecutionHistoryStore):
    backend = "postgresql"
    placeholder = "%s"

    def __init__(
        self,
        database_url: str,
        *,
        connector: Callable[[str], Any] | None = None,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        connect_retries: int = DEFAULT_CONNECT_RETRIES,
    ):
        super().__init__()
        self._database_url = database_url
        self._connector = connector
        self.connect_timeout = int(connect_timeout)
        self.connect_retries = max(0, int(connect_retries))

    def __repr__(self) -> str:
        return "PostgreSQLExecutionHistoryStore(backend='postgresql')"

    def _connect_once(self) -> Any:
        if self._connector is not None:
            return self._connector(self._database_url)
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError:
            raise HistoryStoreError("PostgreSQL 드라이버를 사용할 수 없습니다.") from None
        # TLS options travel in the URL (``sslmode``/``sslrootcert``) so the
        # provider's recommended settings are passed to libpq untouched.  Nothing
        # here weakens certificate handling.
        return psycopg.connect(
            self._database_url,
            row_factory=dict_row,
            connect_timeout=self.connect_timeout,
            application_name="varo_execution_history",
        )

    def _open(self, *, read_only: bool = False) -> Any:
        """Retry establishing a connection only.

        Opening a connection has no side effect, so a bounded retry is safe.  No
        statement or transaction is ever replayed: a write that may already have
        committed is reported as a failure instead of being repeated.
        """
        del read_only
        for attempt in range(self.connect_retries + 1):
            try:
                return self._connect_once()
            except HistoryStoreError:
                raise
            except Exception:
                if attempt >= self.connect_retries:
                    raise HistoryStoreError("서버 실행 기록 저장소에 연결하지 못했습니다.") from None
                time.sleep(0.2 * (attempt + 1))
        raise HistoryStoreError("서버 실행 기록 저장소에 연결하지 못했습니다.")

    @staticmethod
    def _sqlstate(error: Exception) -> str | None:
        state = getattr(error, "sqlstate", None)
        if state:
            return str(state)
        return str(getattr(getattr(error, "diag", None), "sqlstate", "") or "") or None

    def _is_unique_violation(self, error: Exception) -> bool:
        return self._sqlstate(error) == "23505"

    def _is_missing_table(self, error: Exception) -> bool:
        return self._sqlstate(error) == "42P01"

    def _ensure_schema(self, connection: Any) -> None:
        if self._schema_ready:
            try:
                cursor = self._execute(
                    connection,
                    "SELECT schema_version FROM execution_history_schema_meta "
                    "WHERE meta_key = 'execution_history'",
                )
                row = _row_dict(cursor.fetchone(), cursor)
                version = int(row.get("schema_version", 0))
                if version > SCHEMA_VERSION:
                    raise HistoryStoreError("지원하지 않는 실행 기록 스키마입니다.")
                if version == SCHEMA_VERSION:
                    connection.commit()
                    return
                self._schema_ready = False
                connection.rollback()
            except HistoryStoreError:
                try:
                    connection.rollback()
                except Exception:
                    pass
                raise
            except Exception:
                self._schema_ready = False
                try:
                    connection.rollback()
                except Exception:
                    pass
        try:
            self._begin(connection)
            for statement in POSTGRES_SCHEMA_STATEMENTS:
                self._execute(connection, statement)
            self._execute(
                connection,
                """INSERT INTO execution_history_schema_meta (meta_key, schema_version, updated_at)
                VALUES ('execution_history', ?, ?)
                ON CONFLICT (meta_key) DO NOTHING""",
                (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            cursor = self._execute(
                connection,
                "SELECT schema_version FROM execution_history_schema_meta WHERE meta_key = 'execution_history'",
            )
            row = _row_dict(cursor.fetchone(), cursor)
            version = int(row.get("schema_version", 0))
            if version > SCHEMA_VERSION:
                raise HistoryStoreError("지원하지 않는 실행 기록 스키마입니다.")
            if version < SCHEMA_VERSION:
                self._execute(
                    connection,
                    "UPDATE execution_history_schema_meta SET schema_version = ?, updated_at = ? "
                    "WHERE meta_key = 'execution_history'",
                    (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat(timespec="seconds")),
                )
            connection.commit()
            self._schema_ready = True
        except HistoryStoreError:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise HistoryStoreError("서버 실행 기록 스키마를 준비하지 못했습니다.") from None

    def _row_lock_clause(self) -> str:
        return " FOR UPDATE"

    def _read_schema_version(self, connection: Any) -> int | None:
        try:
            cursor = self._execute(
                connection,
                "SELECT schema_version FROM execution_history_schema_meta "
                "WHERE meta_key = 'execution_history'",
            )
            row = _row_dict(cursor.fetchone(), cursor)
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            return None
        return None if not row else int(row["schema_version"])

    def _observe_schema(self, connection: Any) -> dict[str, Any]:
        """Read PostgreSQL catalogs for the tables this store owns."""
        tables: dict[str, Any] = {}
        for table in EXPECTED_SCHEMA:
            column_cursor = self._execute(
                connection,
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = ?",
                (table,),
            )
            columns = {
                str(_row_dict(row, column_cursor)["column_name"]) for row in column_cursor.fetchall()
            }
            if not columns:
                continue
            key_cursor = self._execute(
                connection,
                """SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON kcu.constraint_name = tc.constraint_name
                    AND kcu.constraint_schema = tc.constraint_schema
                WHERE tc.table_schema = current_schema() AND tc.table_name = ?
                    AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position""",
                (table,),
            )
            reference_cursor = self._execute(
                connection,
                """SELECT ccu.table_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.constraint_schema = tc.constraint_schema
                WHERE tc.table_schema = current_schema() AND tc.table_name = ?
                    AND tc.constraint_type = 'FOREIGN KEY'""",
                (table,),
            )
            tables[table] = {
                "columns": columns,
                "primary_key": [
                    str(_row_dict(row, key_cursor)["column_name"]) for row in key_cursor.fetchall()
                ],
                "references": {
                    str(_row_dict(row, reference_cursor)["table_name"])
                    for row in reference_cursor.fetchall()
                },
            }
        index_cursor = self._execute(
            connection, "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()",
        )
        return {
            "tables": tables,
            "indexes": {str(_row_dict(row, index_cursor)["indexname"]) for row in index_cursor.fetchall()},
            "schema_version": self._read_schema_version(connection),
        }


@lru_cache(maxsize=8)
def _cached_store(
    backend: str, locator: str, connect_timeout: int, connect_retries: int,
) -> ExecutionHistoryStore:
    if backend == "sqlite":
        return SQLiteExecutionHistoryStore(locator)
    return PostgreSQLExecutionHistoryStore(
        locator, connect_timeout=connect_timeout, connect_retries=connect_retries,
    )


def build_execution_history_store(
    db_path: str | Path | None = None,
    *,
    config: ExecutionHistoryConfig | None = None,
    environ: Mapping[str, str] | None = None,
    postgres_connector: Callable[[str], Any] | None = None,
) -> ExecutionHistoryStore:
    selected = config or load_execution_history_config(db_path, environ=environ)
    if postgres_connector is None:
        locator = str(selected.sqlite_path) if selected.backend == "sqlite" else str(selected.database_url or "")
        if locator:
            return _cached_store(
                selected.backend, locator, selected.connect_timeout, selected.connect_retries,
            )
    if selected.backend == "sqlite" and selected.sqlite_path is not None:
        return SQLiteExecutionHistoryStore(selected.sqlite_path)
    if selected.backend == "postgresql" and selected.database_url:
        return PostgreSQLExecutionHistoryStore(
            selected.database_url,
            connector=postgres_connector,
            connect_timeout=selected.connect_timeout,
            connect_retries=selected.connect_retries,
        )
    raise HistoryStoreError("실행 기록 저장 설정을 확인해주세요.")
