from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from services.execution_history import record_execution_plan, update_execution_item
from services.execution_history_migration import migrate_sqlite_history, validate_history_snapshot
from services.execution_history_config import load_execution_history_config
from services.execution_history_store import PostgreSQLExecutionHistoryStore
from test_execution_history_backends import CompatConnector, plan_fixture
from tools import migrate_execution_history as migration_cli


class ExecutionHistoryMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.sqlite3"
        self.destination_path = self.root / "destination.sqlite3"
        self.connector = CompatConnector(self.destination_path)
        self.destination = PostgreSQLExecutionHistoryStore(
            "postgresql://test-user:test-password@db.invalid/test-db",
            connector=self.connector,
        )
        self.assertTrue(record_execution_plan(plan_fixture("PLAN-MIGRATE"), self.source)["ok"])
        self.assertTrue(update_execution_item(
            "PLAN-MIGRATE", "C-001", "일부 실행", 3,
            nonexecution_reason="현장 판단",
            outcomes={"actual_transport_cost": 11.0, "actual_saving": 44.0},
            db_path=self.source,
        )["ok"])

    def tearDown(self):
        self.temp.cleanup()

    def _source_digest(self) -> str:
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_dry_run_reports_counts_and_does_not_create_destination_schema(self):
        before = self._source_digest()
        result = migrate_sqlite_history(self.source, self.destination, dry_run=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["code"], "dry_run")
        self.assertEqual(
            (result["plan_count"], result["item_count"], result["audit_count"]),
            (1, 1, 1),
        )
        self.assertEqual(result["duplicate_plan_count"], 0)
        self.assertEqual(result["inserted_plan_count"], 0)
        self.assertEqual(self._source_digest(), before)
        with closing(sqlite3.connect(self.destination_path)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("execution_plans", tables)

    def test_apply_migrates_all_rows_atomically_and_preserves_source(self):
        before = self._source_digest()
        result = migrate_sqlite_history(self.source, self.destination, dry_run=False)
        self.assertTrue(result["ok"])
        self.assertEqual(result["code"], "migrated")
        self.assertEqual(
            (result["inserted_plan_count"], result["inserted_item_count"], result["inserted_audit_count"]),
            (1, 1, 1),
        )
        self.assertEqual(self._source_digest(), before)
        copied = self.destination.read_snapshot()
        self.assertEqual(len(copied["plans"]), 1)
        self.assertEqual(copied["items"][0]["execution_status"], "partial")
        self.assertEqual(copied["items"][0]["actual_qty"], 3)
        self.assertEqual(copied["items"][0]["actual_net_benefit"], 33.0)
        self.assertEqual(len(copied["events"]), 1)

    def test_existing_destination_plan_is_detected_and_skipped(self):
        first = migrate_sqlite_history(self.source, self.destination, dry_run=False)
        second = migrate_sqlite_history(self.source, self.destination, dry_run=False)
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(second["duplicate_plan_count"], 1)
        self.assertEqual(second["new_plan_count"], 0)
        self.assertEqual(second["inserted_plan_count"], 0)
        self.assertEqual(len(self.destination.read_snapshot()["plans"]), 1)

    def test_dry_run_detects_destination_duplicate_without_writing(self):
        self.assertTrue(migrate_sqlite_history(self.source, self.destination, dry_run=False)["ok"])
        before = self.destination_path.read_bytes()
        result = migrate_sqlite_history(self.source, self.destination, dry_run=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["duplicate_plan_count"], 1)
        self.assertEqual(result["new_plan_count"], 0)
        self.assertEqual(self.destination_path.read_bytes(), before)

    def test_destination_failure_rolls_back_plan_items_and_audit(self):
        self.destination.initialize()
        self.connector.state["fail_table"] = "execution_item_events"
        result = migrate_sqlite_history(self.source, self.destination, dry_run=False)
        self.assertFalse(result["ok"])
        self.connector.state.pop("fail_table")
        copied = self.destination.read_snapshot()
        self.assertEqual(copied, {"plans": [], "items": [], "events": []})

    def test_invalid_snapshot_is_rejected(self):
        snapshot = {
            "plans": [{"plan_id": "P", "total_actions": 2, "total_planned_qty": 5}],
            "items": [{"plan_id": "P", "candidate_id": "C", "planned_qty": 4}],
            "events": [{"plan_id": "P", "candidate_id": "missing"}],
        }
        result = validate_history_snapshot(snapshot)
        self.assertFalse(result["valid"])
        self.assertGreaterEqual(result["invalid_record_count"], 2)

    def test_unsupported_source_schema_fails_without_modification(self):
        bad_source = self.root / "future.sqlite3"
        with closing(sqlite3.connect(bad_source)) as connection:
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        before = bad_source.read_bytes()
        result = migrate_sqlite_history(bad_source, self.destination, dry_run=True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "migration_error")
        self.assertEqual(bad_source.read_bytes(), before)

    def test_missing_source_is_rejected_without_creating_it(self):
        missing = self.root / "missing.sqlite3"
        result = migrate_sqlite_history(missing, self.destination, dry_run=True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "migration_error")
        self.assertFalse(missing.exists())

    def test_cli_requires_explicit_server_configuration(self):
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"VARO_HISTORY_DATABASE_URL": ""}), redirect_stdout(output):
            exit_code = migration_cli.main(["--source", str(self.source), "--dry-run"])
        self.assertEqual(exit_code, 2)
        self.assertNotIn(str(self.source), output.getvalue())

    def test_cli_dry_run_prints_counts_without_identifiers_or_secrets(self):
        output = io.StringIO()
        config = load_execution_history_config(
            environ={"VARO_HISTORY_DATABASE_URL": "postgresql://user:secret@db.invalid/test-db"},
        )
        with mock.patch.object(migration_cli, "load_execution_history_config", return_value=config), mock.patch.object(
            migration_cli, "build_execution_history_store", return_value=self.destination,
        ), redirect_stdout(output):
            exit_code = migration_cli.main(["--source", str(self.source), "--dry-run"])
        text = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("계획 1건 / 항목 1건 / 감사 1건", text)
        for hidden in ("PLAN-MIGRATE", "secret", "db.invalid", str(self.source)):
            self.assertNotIn(hidden, text)


if __name__ == "__main__":
    unittest.main()
