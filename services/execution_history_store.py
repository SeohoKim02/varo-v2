"""Backend-neutral persistence adapters for execution history."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from services.execution_history_config import (
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
)


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

    def __init__(self, database_url: str, *, connector: Callable[[str], Any] | None = None):
        super().__init__()
        self._database_url = database_url
        self._connector = connector

    def __repr__(self) -> str:
        return "PostgreSQLExecutionHistoryStore(backend='postgresql')"

    def _open(self, *, read_only: bool = False) -> Any:
        del read_only
        if self._connector is not None:
            return self._connector(self._database_url)
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError:
            raise HistoryStoreError("PostgreSQL 드라이버를 사용할 수 없습니다.") from None
        try:
            return psycopg.connect(
                self._database_url,
                row_factory=dict_row,
                connect_timeout=8,
                application_name="varo_execution_history",
            )
        except Exception:
            raise HistoryStoreError("서버 실행 기록 저장소에 연결하지 못했습니다.") from None

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


@lru_cache(maxsize=8)
def _cached_store(backend: str, locator: str) -> ExecutionHistoryStore:
    if backend == "sqlite":
        return SQLiteExecutionHistoryStore(locator)
    return PostgreSQLExecutionHistoryStore(locator)


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
            return _cached_store(selected.backend, locator)
    if selected.backend == "sqlite" and selected.sqlite_path is not None:
        return SQLiteExecutionHistoryStore(selected.sqlite_path)
    if selected.backend == "postgresql" and selected.database_url:
        return PostgreSQLExecutionHistoryStore(selected.database_url, connector=postgres_connector)
    raise HistoryStoreError("실행 기록 저장 설정을 확인해주세요.")
