from __future__ import annotations

import csv
import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from services.execution_history import (
    execution_history_metrics,
    export_execution_history_csv,
    get_recorded_plan,
    list_item_events,
    record_execution_plan,
    update_execution_item,
)
from services.execution_history_config import (
    DEFAULT_HISTORY_DB,
    HistoryConfigurationError,
    load_execution_history_config,
)
from services.execution_history_store import (
    HistoryStoreError,
    PostgreSQLExecutionHistoryStore,
    SQLiteExecutionHistoryStore,
    build_execution_history_store,
)


def plan_fixture(plan_id: str = "PLAN-PARITY") -> dict:
    return {
        "plan_id": plan_id,
        "algorithm_version": "execution-plan-1.0",
        "data_signature": "signature-anonymous",
        "created_at": "2026-01-02T03:04:05+00:00",
        "plan_status": "추천 가능",
        "total_cost": 12.5,
        "total_expected_saving": 50.75,
        "total_net_benefit": 38.25,
        "validation": {"valid": True},
        "items": [
            {
                "candidate_id": "C-001",
                "algorithm_version": "vhs-2.2",
                "source_id": "S-A",
                "source_name": "가상 출발점",
                "target_id": "S-B",
                "target_name": "가상 도착점",
                "product_id": "P-A",
                "product_name": "가상 상품",
                "route_type": "DIRECT",
                "dc_id": None,
                "planned_qty": 5,
                "planned_cost": 12.5,
                "planned_expected_saving": 50.75,
                "planned_net_benefit": 38.25,
                "vhs_score": 81.25,
                "robustness_status": "안정",
                "confidence_score": 72.5,
            }
        ],
    }


class FakePGError(Exception):
    def __init__(self, message: str, sqlstate: str):
        super().__init__(message)
        self.sqlstate = sqlstate


class _CompatCursor:
    def __init__(self, connection: "SQLitePostgresCompatConnection"):
        self.connection = connection
        self._cursor = connection.inner.cursor()

    def executemany(self, sql, rows):
        self.connection.sql.append(sql)
        if self.connection.state.get("fail_table") and self.connection.state["fail_table"] in sql:
            raise FakePGError("forced", "XX000")
        try:
            return self._cursor.executemany(self.connection.translate(sql), rows)
        except sqlite3.IntegrityError as exc:
            raise FakePGError("unique", "23505") from exc
        except sqlite3.OperationalError as exc:
            state = "42P01" if "no such table" in str(exc).lower() else "XX000"
            raise FakePGError("query", state) from exc

    def close(self):
        self._cursor.close()


class SQLitePostgresCompatConnection:
    """DB-API test double; no network and no real PostgreSQL credentials."""

    def __init__(self, path: Path, state: dict):
        self.state = state
        self.sql = state.setdefault("sql", [])
        self.inner = sqlite3.connect(path, timeout=8.0)
        self.inner.row_factory = sqlite3.Row
        self.inner.execute("PRAGMA foreign_keys = ON")
        state["opened"] = state.get("opened", 0) + 1

    @staticmethod
    def translate(sql: str) -> str:
        return (
            sql.replace("%s", "?")
            .replace("DOUBLE PRECISION", "REAL")
            .replace("BIGSERIAL", "INTEGER")
            .replace(" FOR UPDATE", "")
        )

    def execute(self, sql, params=()):
        self.sql.append(sql)
        if self.state.get("fail_query") and self.state["fail_query"] in sql:
            raise FakePGError("forced", "XX000")
        try:
            return self.inner.execute(self.translate(sql), tuple(params))
        except sqlite3.IntegrityError as exc:
            raise FakePGError("unique", "23505") from exc
        except sqlite3.OperationalError as exc:
            state = "42P01" if "no such table" in str(exc).lower() else "XX000"
            raise FakePGError("query", state) from exc

    def cursor(self):
        return _CompatCursor(self)

    def commit(self):
        self.inner.commit()

    def rollback(self):
        self.inner.rollback()

    def close(self):
        self.inner.close()
        self.state["closed"] = self.state.get("closed", 0) + 1


class CompatConnector:
    def __init__(self, path: Path):
        self.path = path
        self.state: dict = {}
        self.received_urls: list[str] = []

    def __call__(self, database_url: str):
        self.received_urls.append(database_url)
        return SQLitePostgresCompatConnection(self.path, self.state)


class BackendSelectionTests(unittest.TestCase):
    def test_no_url_selects_default_sqlite(self):
        config = load_execution_history_config(environ={})
        self.assertEqual(config.backend, "sqlite")
        self.assertEqual(config.sqlite_path, DEFAULT_HISTORY_DB)

    def test_sqlite_path_override_is_preserved(self):
        config = load_execution_history_config(environ={"VARO_HISTORY_DB_PATH": "local/history.sqlite3"})
        self.assertEqual(config.sqlite_path, Path("local/history.sqlite3"))

    def test_postgresql_url_has_environment_priority_and_safe_repr(self):
        secret = "postgresql://operator:very-secret@db.invalid:5432/varo"
        config = load_execution_history_config(
            environ={"VARO_HISTORY_DATABASE_URL": secret, "VARO_HISTORY_DB_PATH": "ignored.sqlite3"},
        )
        self.assertEqual(config.backend, "postgresql")
        self.assertEqual(config.user_label, "서버")
        self.assertNotIn("very-secret", repr(config))

    def test_explicit_path_forces_sqlite_even_if_server_url_exists(self):
        config = load_execution_history_config(
            Path("test.sqlite3"),
            environ={"VARO_HISTORY_DATABASE_URL": "postgresql://user:pass@db.invalid/varo"},
        )
        self.assertEqual(config.backend, "sqlite")

    def test_invalid_explicit_database_url_fails_closed(self):
        for value in ("sqlite:///wrong", "postgresql://missing-database", "not-a-url"):
            with self.subTest(value=value):
                with self.assertRaises(HistoryConfigurationError):
                    load_execution_history_config(environ={"VARO_HISTORY_DATABASE_URL": value})

    def test_factory_returns_common_contract_implementations(self):
        sqlite_store = build_execution_history_store(environ={})
        self.assertIsInstance(sqlite_store, SQLiteExecutionHistoryStore)
        config = load_execution_history_config(
            environ={"VARO_HISTORY_DATABASE_URL": "postgresql://user:pass@db.invalid/varo"},
        )
        pg_store = build_execution_history_store(config=config, postgres_connector=lambda _: None)
        self.assertIsInstance(pg_store, PostgreSQLExecutionHistoryStore)
        self.assertNotIn("pass", repr(pg_store))

    def test_secret_and_runtime_patterns_are_ignored_but_public_example_is_allowed(self):
        ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env.*", ignore)
        self.assertIn("!.env.example", ignore)
        self.assertIn(".streamlit/secrets.toml", ignore)
        self.assertIn("runtime_data/", ignore)
        self.assertIn("*.sqlite3", ignore)

    def test_postgresql_driver_is_pinned_for_deployment(self):
        requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("psycopg[binary]==3.3.5", requirements)


class PostgreSQLAdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pg_path = self.root / "postgres_compat.sqlite3"
        self.connector = CompatConnector(self.pg_path)
        self.url = "postgresql://test-user:test-password@db.invalid/test_db"
        self.store = PostgreSQLExecutionHistoryStore(self.url, connector=self.connector)

    def tearDown(self):
        self.temp.cleanup()

    def _service_patch(self):
        return mock.patch("services.execution_history.build_execution_history_store", return_value=self.store)

    def test_schema_primary_keys_foreign_keys_and_version(self):
        self.store.initialize()
        with closing(sqlite3.connect(self.pg_path)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            version = connection.execute(
                "SELECT schema_version FROM execution_history_schema_meta WHERE meta_key='execution_history'",
            ).fetchone()[0]
            item_fks = connection.execute("PRAGMA foreign_key_list(execution_items)").fetchall()
            event_fks = connection.execute("PRAGMA foreign_key_list(execution_item_events)").fetchall()
        self.assertTrue({"execution_plans", "execution_items", "execution_item_events"} <= tables)
        self.assertEqual(version, 1)
        self.assertTrue(item_fks)
        self.assertTrue(event_fks)
        self.assertTrue(all("IF NOT EXISTS" in sql for sql in self.connector.state["sql"] if sql.startswith("CREATE")))

    def test_cached_postgresql_store_recovers_after_schema_is_replaced(self):
        self.store.initialize()
        with closing(sqlite3.connect(self.pg_path)) as connection:
            connection.execute("DROP TABLE execution_item_events")
            connection.execute("DROP TABLE execution_items")
            connection.execute("DROP TABLE execution_plans")
            connection.execute("DROP TABLE execution_history_schema_meta")
            connection.commit()

        self.store.initialize()

        with closing(sqlite3.connect(self.pg_path)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            version = connection.execute(
                "SELECT schema_version FROM execution_history_schema_meta WHERE meta_key='execution_history'",
            ).fetchone()[0]
        self.assertTrue({"execution_plans", "execution_items", "execution_item_events"} <= tables)
        self.assertEqual(version, 1)

    def test_save_duplicate_update_null_audit_and_reconnect(self):
        with self._service_patch():
            first = record_execution_plan(plan_fixture())
            duplicate = record_execution_plan(plan_fixture())
            before = get_recorded_plan("PLAN-PARITY")
            updated = update_execution_item(
                "PLAN-PARITY", "C-001", "일부 실행", 7,
                nonexecution_reason="현장 판단",
                outcomes={"actual_transport_cost": 10.25, "actual_saving": 45.5},
            )
            after = get_recorded_plan("PLAN-PARITY")
            events = list_item_events("PLAN-PARITY", "C-001")
        self.assertEqual(first["code"], "recorded")
        self.assertEqual(duplicate["code"], "duplicate")
        self.assertEqual(before["plan"]["expected_total_cost"], 12.5)
        self.assertEqual(datetime.fromisoformat(before["plan"]["recorded_at"]).utcoffset(), timedelta(0))
        self.assertIsNone(before["items"][0]["actual_qty"])
        self.assertIsNone(before["items"][0]["post_source_stock"])
        self.assertEqual(updated["warning"], "계획보다 2개 많이 실행되었습니다.")
        self.assertEqual(after["items"][0]["execution_status"], "partial")
        self.assertEqual(after["items"][0]["actual_net_benefit"], 35.25)
        self.assertEqual(len(events), 1)
        self.assertGreaterEqual(self.connector.state["opened"], 6)
        self.assertEqual(self.connector.state["opened"], self.connector.state["closed"])

    def test_postgresql_parameters_are_bound_and_item_update_locks_row(self):
        plan = plan_fixture("PLAN-'BOUND")
        with self._service_patch():
            self.assertTrue(record_execution_plan(plan)["ok"])
            self.assertTrue(update_execution_item("PLAN-'BOUND", "C-001", "실행", 5)["ok"])
        statements = self.connector.state["sql"]
        insert = next(sql for sql in statements if sql.startswith("INSERT INTO execution_plans"))
        self.assertIn("%s", insert)
        self.assertNotIn("PLAN-'BOUND", insert)
        self.assertTrue(any("FOR UPDATE" in sql for sql in statements))

    def test_postgresql_and_sqlite_public_results_and_export_columns_match(self):
        sqlite_path = self.root / "local.sqlite3"
        local_plan = plan_fixture()
        self.assertTrue(record_execution_plan(local_plan, sqlite_path)["ok"])
        self.assertTrue(update_execution_item(
            "PLAN-PARITY", "C-001", "실행", 5,
            outcomes={"post_source_stock": 10, "actual_transport_cost": 12.5, "actual_saving": 50.75},
            db_path=sqlite_path,
        )["ok"])
        with self._service_patch():
            self.assertTrue(record_execution_plan(plan_fixture())["ok"])
            self.assertTrue(update_execution_item(
                "PLAN-PARITY", "C-001", "실행", 5,
                outcomes={"post_source_stock": 10, "actual_transport_cost": 12.5, "actual_saving": 50.75},
            )["ok"])
            pg_loaded = get_recorded_plan("PLAN-PARITY")
            pg_export = export_execution_history_csv()
        sqlite_loaded = get_recorded_plan("PLAN-PARITY", sqlite_path)
        sqlite_export = export_execution_history_csv(sqlite_path)
        for key in (
            "planned_qty", "actual_qty", "execution_status", "post_source_stock",
            "actual_transport_cost", "actual_saving", "actual_net_benefit",
        ):
            self.assertEqual(sqlite_loaded["items"][0][key], pg_loaded["items"][0][key])
        local_row = next(csv.DictReader(io.StringIO(sqlite_export["data"].decode("utf-8-sig"))))
        pg_row = next(csv.DictReader(io.StringIO(pg_export["data"].decode("utf-8-sig"))))
        self.assertEqual(list(local_row), list(pg_row))
        for key in ("planned_qty", "actual_qty", "execution_status", "actual_transport_cost", "actual_saving"):
            self.assertEqual(local_row[key], pg_row[key])

    def test_plan_and_items_rollback_on_postgresql_write_failure(self):
        self.store.initialize()
        self.connector.state["fail_table"] = "execution_items"
        with self._service_patch():
            result = record_execution_plan(plan_fixture())
        self.assertFalse(result["ok"])
        self.connector.state.pop("fail_table")
        self.assertEqual(self.store.list_plans(limit=50), [])

    def test_item_update_and_audit_rollback_together(self):
        with self._service_patch():
            self.assertTrue(record_execution_plan(plan_fixture())["ok"])
            self.connector.state["fail_query"] = "INSERT INTO execution_item_events"
            result = update_execution_item("PLAN-PARITY", "C-001", "실행", 5)
            self.connector.state.pop("fail_query")
            loaded = get_recorded_plan("PLAN-PARITY")
            events = list_item_events("PLAN-PARITY", "C-001")
        self.assertFalse(result["ok"])
        self.assertEqual(loaded["items"][0]["execution_status"], "unconfirmed")
        self.assertIsNone(loaded["items"][0]["actual_qty"])
        self.assertEqual(events, [])

    def test_metrics_use_aggregate_queries_and_plan_listing_is_paginated(self):
        with self._service_patch():
            self.assertTrue(record_execution_plan(plan_fixture("PLAN-1"))["ok"])
            self.assertTrue(record_execution_plan(plan_fixture("PLAN-2"))["ok"])
            self.assertTrue(update_execution_item("PLAN-1", "C-001", "실행", 5)["ok"])
            metrics = execution_history_metrics()
        first_page = self.store.list_plans(limit=1, offset=0)
        second_page = self.store.list_plans(limit=1, offset=1)
        self.assertEqual(metrics["confirmed_items"], 1)
        self.assertEqual(len(first_page), 1)
        self.assertEqual(len(second_page), 1)
        self.assertNotEqual(first_page[0]["plan_id"], second_page[0]["plan_id"])
        self.assertTrue(any("COUNT(*) AS total_items" in sql for sql in self.connector.state["sql"]))

    def test_query_failure_returns_safe_message(self):
        self.store.initialize()
        self.connector.state["fail_query"] = "SELECT * FROM execution_plans"
        with self._service_patch():
            loaded = get_recorded_plan("PLAN-PARITY")
        self.assertFalse(loaded["ok"])
        self.assertNotIn("test-password", loaded["message"])
        self.assertNotIn("SELECT", loaded["message"])

    def test_configured_postgresql_failure_never_creates_sqlite_fallback(self):
        local_path = self.root / "must-not-exist.sqlite3"
        unique_url = "postgresql://user:secret@unreachable.invalid/no_fallback"
        env = {
            "VARO_HISTORY_DATABASE_URL": unique_url,
            "VARO_HISTORY_DB_PATH": str(local_path),
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            PostgreSQLExecutionHistoryStore,
            "_open",
            side_effect=HistoryStoreError("connection failed"),
        ):
            result = record_execution_plan(plan_fixture("PLAN-NO-FALLBACK"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["message"], "실행 기록을 저장하지 못했습니다.")
        self.assertFalse(local_path.exists())
        self.assertNotIn("secret", result["message"])


class SQLiteAdapterRecoveryTests(unittest.TestCase):
    def test_cached_sqlite_store_recovers_after_database_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.sqlite3"
            store = SQLiteExecutionHistoryStore(path)
            store.initialize()
            path.unlink()

            store.initialize()

            with closing(sqlite3.connect(path)) as connection:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            self.assertTrue({"execution_plans", "execution_items", "execution_item_events"} <= tables)
            self.assertEqual(version, 1)


if __name__ == "__main__":
    unittest.main()
