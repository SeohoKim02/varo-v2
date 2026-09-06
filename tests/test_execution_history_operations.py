"""Deployment-readiness contracts for the execution-history storage layer.

These run without any external database.  Checks that genuinely need a live
PostgreSQL server live in ``tests/integration`` and are collected only when
``VARO_HISTORY_TEST_DATABASE_URL`` is set.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest import mock

from services.execution_history import (
    RECENT_PLAN_PAGE_SIZE,
    execution_history_health,
    inspect_execution_history_schema,
    list_recorded_plans,
    record_execution_plan,
    update_execution_item,
)
from services.execution_history_config import (
    DEFAULT_CONNECT_RETRIES,
    DEFAULT_CONNECT_TIMEOUT,
    HISTORY_DATABASE_URL_ENV,
    HISTORY_RESTORE_DATABASE_URL_ENV,
    HISTORY_TEST_DATABASE_URL_ENV,
    MAX_CONNECT_TIMEOUT,
    MIN_CONNECT_TIMEOUT,
    HistoryConfigurationError,
    database_name,
    database_url_transport,
    load_execution_history_config,
    load_restore_target_config,
    load_staging_test_config,
    redact_database_url,
)
from services.execution_history_store import (
    EXPECTED_INDEXES,
    SCHEMA_VERSION,
    HistoryStoreError,
    PostgreSQLExecutionHistoryStore,
    SQLiteExecutionHistoryStore,
    evaluate_schema,
)
from test_execution_history_backends import CompatConnector, plan_fixture
from tools import backup_execution_history as backup_cli
from tools import check_execution_history_db as check_cli
from tools import restore_execution_history as restore_cli
from tools import validate_postgresql_history as validate_cli

SECRET_URL = "postgresql://operator:top-secret@db.invalid:5432/varo_production"
STAGING_URL = "postgresql://tester:staging-secret@staging.invalid:5432/varo_staging"


class StagingConfigurationSafetyTests(unittest.TestCase):
    def test_absent_staging_url_disables_every_real_database_check(self):
        self.assertIsNone(load_staging_test_config(environ={}))
        self.assertIsNone(load_staging_test_config(environ={HISTORY_DATABASE_URL_ENV: SECRET_URL}))

    def test_staging_url_is_a_separate_variable_from_the_deployment_url(self):
        config = load_staging_test_config(
            environ={HISTORY_DATABASE_URL_ENV: SECRET_URL, HISTORY_TEST_DATABASE_URL_ENV: STAGING_URL},
        )
        self.assertIsNotNone(config)
        self.assertEqual(config.purpose, "staging_test")
        self.assertEqual(config.database_url, STAGING_URL)
        self.assertNotEqual(config.database_url, SECRET_URL)

    def test_staging_url_identical_to_production_is_refused(self):
        with self.assertRaises(HistoryConfigurationError):
            load_staging_test_config(
                environ={HISTORY_DATABASE_URL_ENV: SECRET_URL, HISTORY_TEST_DATABASE_URL_ENV: SECRET_URL},
            )

    def test_invalid_staging_url_fails_closed(self):
        for value in ("sqlite:///wrong", "postgresql://missing-database", "not-a-url"):
            with self.subTest(value=value):
                with self.assertRaises(HistoryConfigurationError):
                    load_staging_test_config(environ={HISTORY_TEST_DATABASE_URL_ENV: value})

    def test_default_test_run_is_isolated_from_any_configured_database(self):
        # tests/conftest.py rewrites both variables for every test in the suite.
        self.assertEqual(os.environ.get(HISTORY_DATABASE_URL_ENV), "")
        config = load_execution_history_config()
        self.assertEqual(config.backend, "sqlite")
        self.assertIn("isolated_execution_history", str(config.sqlite_path))

    def test_integration_directory_is_skipped_without_a_staging_url(self):
        integration = Path(__file__).resolve().parent / "integration" / "conftest.py"
        self.assertTrue(integration.is_file())
        self.assertIn("pytest_ignore_collect", integration.read_text(encoding="utf-8"))

    def test_redaction_removes_every_credential_component(self):
        redacted = redact_database_url(SECRET_URL)
        for hidden in ("operator", "top-secret", "db.invalid", "5432", "varo_production"):
            self.assertNotIn(hidden, redacted)
        self.assertEqual(redact_database_url(""), "")

    def test_database_name_is_extracted_only_for_explicit_confirmation(self):
        self.assertEqual(database_name(SECRET_URL), "varo_production")
        self.assertEqual(database_name(None), "")


class ConnectionPolicyTests(unittest.TestCase):
    def test_timeout_and_retry_defaults_apply_without_any_configuration(self):
        config = load_execution_history_config(environ={HISTORY_DATABASE_URL_ENV: SECRET_URL})
        self.assertEqual(config.connect_timeout, DEFAULT_CONNECT_TIMEOUT)
        self.assertEqual(config.connect_retries, DEFAULT_CONNECT_RETRIES)
        self.assertGreaterEqual(config.connect_timeout, MIN_CONNECT_TIMEOUT)

    def test_configured_timeout_is_honoured_and_bounded(self):
        for raw, expected in (("20", 20), ("0", MIN_CONNECT_TIMEOUT), ("9999", MAX_CONNECT_TIMEOUT), ("abc", DEFAULT_CONNECT_TIMEOUT)):
            with self.subTest(raw=raw):
                config = load_execution_history_config(environ={
                    HISTORY_DATABASE_URL_ENV: SECRET_URL,
                    "VARO_HISTORY_DB_CONNECT_TIMEOUT": raw,
                })
                self.assertEqual(config.connect_timeout, expected)

    def test_timeout_reaches_the_store_instance(self):
        config = load_execution_history_config(environ={
            HISTORY_DATABASE_URL_ENV: SECRET_URL, "VARO_HISTORY_DB_CONNECT_TIMEOUT": "15",
        })
        store = PostgreSQLExecutionHistoryStore(
            str(config.database_url),
            connect_timeout=config.connect_timeout,
            connect_retries=config.connect_retries,
        )
        self.assertEqual(store.connect_timeout, 15)

    def test_opening_a_connection_is_retried_but_statements_are_never_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            connector = CompatConnector(Path(temp) / "pg.sqlite3")
            attempts = {"count": 0}

            def flaky(database_url: str):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise OSError("network reset")
                return connector(database_url)

            store = PostgreSQLExecutionHistoryStore("postgresql://u:p@db.invalid/db", connector=flaky, connect_retries=1)
            store.initialize()
            self.assertEqual(attempts["count"], 2)

            connector.state["fail_query"] = "INSERT INTO execution_plans"
            with mock.patch("services.execution_history.build_execution_history_store", return_value=store):
                result = record_execution_plan(plan_fixture("PLAN-NO-REPLAY"))
            self.assertFalse(result["ok"])
            inserts = [sql for sql in connector.state["sql"] if sql.startswith("INSERT INTO execution_plans")]
            self.assertEqual(len(inserts), 1, "실패한 write가 재시도되면 중복 저장 위험이 있습니다.")

    def test_connection_failure_is_not_retried_beyond_the_configured_budget(self):
        attempts = {"count": 0}

        def always_failing(_: str):
            attempts["count"] += 1
            raise OSError("unreachable")

        store = PostgreSQLExecutionHistoryStore(SECRET_URL, connector=always_failing, connect_retries=0)
        with self.assertRaises(HistoryStoreError) as caught:
            store.initialize()
        self.assertEqual(attempts["count"], 1)
        for hidden in ("top-secret", "db.invalid", "unreachable"):
            self.assertNotIn(hidden, str(caught.exception))


class TransportSecurityTests(unittest.TestCase):
    def test_url_ssl_options_are_reported_without_revealing_the_host(self):
        transport = database_url_transport(
            "postgresql://u:p@db.invalid/varo?sslmode=verify-full&sslrootcert=/etc/ssl/root.crt",
        )
        self.assertEqual(transport["sslmode"], "verify-full")
        self.assertTrue(transport["encrypted"])
        self.assertTrue(transport["verified"])
        self.assertTrue(transport["sslrootcert_configured"])

    def test_weak_and_missing_ssl_modes_are_flagged_not_silently_accepted(self):
        self.assertFalse(database_url_transport("postgresql://u:p@h/db?sslmode=disable")["encrypted"])
        self.assertFalse(database_url_transport("postgresql://u:p@h/db?sslmode=require")["verified"])
        self.assertIsNone(database_url_transport("postgresql://u:p@h/db")["sslmode"])
        self.assertIn("경고", check_cli.transport_line("postgresql://u:p@h/db?sslmode=disable"))
        self.assertIn("미지정", check_cli.transport_line("postgresql://u:p@h/db"))
        line = check_cli.transport_line("postgresql://u:p@h.invalid/db?sslmode=verify-full")
        self.assertIn("인증서 검증", line)
        self.assertNotIn("h.invalid", line)

    def test_the_store_passes_the_url_through_untouched(self):
        url = "postgresql://u:p@db.invalid/varo?sslmode=verify-full&sslrootcert=/etc/ssl/root.crt"
        with tempfile.TemporaryDirectory() as temp:
            connector = CompatConnector(Path(temp) / "pg.sqlite3")
            PostgreSQLExecutionHistoryStore(url, connector=connector).initialize()
        self.assertEqual(connector.received_urls, [url])

    def test_driver_preserves_transport_options_or_reports_a_safe_error(self):
        url = "postgresql://u:p@db.invalid/varo?sslmode=verify-full&sslrootcert=/etc/ssl/root.crt"
        try:
            from psycopg.conninfo import conninfo_to_dict
        except ImportError:
            store = PostgreSQLExecutionHistoryStore(url, connect_retries=0)
            with self.assertRaises(HistoryStoreError) as caught:
                store._connect_once()
            self.assertNotIn("db.invalid", str(caught.exception))
            return
        parsed = conninfo_to_dict(url)
        self.assertEqual(parsed["sslmode"], "verify-full")
        self.assertEqual(parsed["sslrootcert"], "/etc/ssl/root.crt")


class SchemaInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "history.sqlite3"
        self.store = SQLiteExecutionHistoryStore(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_report_lists_tables_keys_foreign_keys_and_indexes(self):
        self.store.initialize()
        report = self.store.inspect_schema()
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["tables"]["execution_items"]["primary_key"], ["plan_id", "candidate_id"])
        self.assertEqual(report["tables"]["execution_plans"]["primary_key"], ["plan_id"])
        self.assertIn("execution_plans", report["tables"]["execution_items"]["references"])
        self.assertIn("execution_items", report["tables"]["execution_item_events"]["references"])
        self.assertTrue(all(report["indexes"].values()))
        self.assertIn("idx_execution_item_events_item", EXPECTED_INDEXES)

    def test_uninitialized_storage_is_reported_without_creating_a_file(self):
        report = self.store.inspect_schema()
        self.assertFalse(report["ok"])
        self.assertFalse(self.path.exists())

    def test_repeated_initialization_keeps_existing_rows_and_version(self):
        self.assertTrue(record_execution_plan(plan_fixture("PLAN-IDEMPOTENT"), self.path)["ok"])
        self.store.initialize()
        SQLiteExecutionHistoryStore(self.path).initialize()
        report = inspect_execution_history_schema(self.path)
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(len(list_recorded_plans(self.path)["plans"]), 1)

    def test_missing_table_and_index_are_detected(self):
        self.store.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP INDEX idx_execution_item_events_item")
            connection.commit()
        report = SQLiteExecutionHistoryStore(self.path).inspect_schema()
        self.assertFalse(report["ok"])
        self.assertFalse(report["indexes"]["idx_execution_item_events_item"])
        self.assertTrue(any("인덱스" in issue for issue in report["issues"]))

    def test_version_and_column_mismatches_are_reported(self):
        observed = {
            "tables": {
                "execution_plans": {"columns": {"plan_id"}, "primary_key": ["plan_id"], "references": set()},
            },
            "indexes": set(),
            "schema_version": 99,
        }
        report = evaluate_schema(observed)
        self.assertFalse(report["ok"])
        self.assertTrue(any("컬럼 누락" in issue for issue in report["issues"]))
        self.assertTrue(any("버전 불일치" in issue for issue in report["issues"]))
        self.assertTrue(any("execution_items" in issue for issue in report["issues"]))

    def test_schema_report_contains_no_connection_details(self):
        self.store.initialize()
        blob = json.dumps(self.store.inspect_schema(), ensure_ascii=False)
        for hidden in (str(self.path), "sqlite3", "password"):
            self.assertNotIn(hidden, blob)


class HealthCheckTests(unittest.TestCase):
    def test_healthy_store_reports_version_and_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.sqlite3"
            SQLiteExecutionHistoryStore(path).initialize()
            health = execution_history_health(path)
        self.assertTrue(health["ok"])
        self.assertTrue(health["connection_ok"])
        self.assertEqual(health["schema_version"], SCHEMA_VERSION)
        self.assertIsNotNone(health["latency_ms"])

    def test_unreachable_store_reports_a_safe_failure_without_raising(self):
        store = PostgreSQLExecutionHistoryStore(SECRET_URL, connector=lambda _: (_ for _ in ()).throw(OSError("down")))
        with mock.patch("services.execution_history.build_execution_history_store", return_value=store):
            health = execution_history_health()
        self.assertFalse(health["ok"])
        self.assertFalse(health["connection_ok"])
        for hidden in ("top-secret", "db.invalid", "down"):
            self.assertNotIn(hidden, json.dumps(health, ensure_ascii=False))


class PaginationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "history.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def _seed(self, count: int) -> None:
        for index in range(count):
            self.assertTrue(record_execution_plan(plan_fixture(f"PLAN-{index:04d}"), self.path)["ok"])

    def test_empty_history_reports_no_further_pages(self):
        page = list_recorded_plans(self.path, limit=RECENT_PLAN_PAGE_SIZE)
        self.assertTrue(page["ok"])
        self.assertEqual(page["plans"], [])
        self.assertFalse(page["has_more"])

    def test_recent_page_is_bounded_and_announces_older_records(self):
        self._seed(RECENT_PLAN_PAGE_SIZE + 5)
        page = list_recorded_plans(self.path, limit=RECENT_PLAN_PAGE_SIZE)
        self.assertEqual(len(page["plans"]), RECENT_PLAN_PAGE_SIZE)
        self.assertTrue(page["has_more"])

    def test_offset_pages_do_not_overlap_and_end_cleanly(self):
        self._seed(5)
        first = list_recorded_plans(self.path, limit=2, offset=0)
        second = list_recorded_plans(self.path, limit=2, offset=2)
        last = list_recorded_plans(self.path, limit=2, offset=4)
        identifiers = [row["plan_id"] for page in (first, second, last) for row in page["plans"]]
        self.assertEqual(len(identifiers), 5)
        self.assertEqual(len(set(identifiers)), 5)
        self.assertTrue(first["has_more"])
        self.assertFalse(last["has_more"])

    def test_storage_failure_returns_an_empty_safe_page(self):
        store = PostgreSQLExecutionHistoryStore(SECRET_URL, connector=lambda _: (_ for _ in ()).throw(OSError("down")))
        with mock.patch("services.execution_history.build_execution_history_store", return_value=store):
            page = list_recorded_plans(limit=10)
        self.assertFalse(page["ok"])
        self.assertEqual(page["plans"], [])
        self.assertFalse(page["has_more"])


class MaintenanceCleanupTests(unittest.TestCase):
    def test_delete_plans_removes_only_the_named_plans_with_their_children(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.sqlite3"
            for plan_id in ("KEEP-1", "VARO-STAGING-CHECK-1"):
                self.assertTrue(record_execution_plan(plan_fixture(plan_id), path)["ok"])
                self.assertTrue(update_execution_item(plan_id, "C-001", "실행", 5, db_path=path)["ok"])
            store = SQLiteExecutionHistoryStore(path)

            removed = store.delete_plans(["VARO-STAGING-CHECK-1"])

            self.assertEqual(removed, {"plans": 1, "items": 1, "events": 1})
            snapshot = store.read_snapshot()
            self.assertEqual([row["plan_id"] for row in snapshot["plans"]], ["KEEP-1"])
            self.assertEqual([row["plan_id"] for row in snapshot["items"]], ["KEEP-1"])
            self.assertEqual([row["plan_id"] for row in snapshot["events"]], ["KEEP-1"])
            self.assertEqual(store.delete_plans([]), {"plans": 0, "items": 0, "events": 0})


class SecretExposureAuditTests(unittest.TestCase):
    """Source-level audit: a connection URL must not be able to reach a log or screen."""

    MODULES = (
        "services/execution_history.py",
        "services/execution_history_config.py",
        "services/execution_history_store.py",
        "services/execution_history_migration.py",
        "components/execution_history_panel.py",
    )

    def _source(self, name: str) -> str:
        return (Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8")

    def test_history_modules_neither_log_nor_print(self):
        for name in self.MODULES:
            with self.subTest(module=name):
                source = self._source(name)
                self.assertNotIn("print(", source)
                self.assertNotIn("import logging", source)
                self.assertNotIn("logger.", source)

    def test_storage_errors_carry_only_fixed_messages(self):
        for name in self.MODULES:
            with self.subTest(module=name):
                source = self._source(name)
                for pattern in ('HistoryStoreError(f"', 'HistoryConfigurationError(f"', 'HistoryStoreError(str('):
                    self.assertNotIn(pattern, source)

    def test_driver_exceptions_are_not_chained_into_user_facing_errors(self):
        source = self._source("services/execution_history_store.py")
        self.assertIn("from None", source)
        for pattern in ("from error", "from exc", "from exception"):
            self.assertNotIn(f"raise HistoryStoreError({pattern}", source)

    def test_store_and_config_objects_never_expose_the_password(self):
        config = load_execution_history_config(environ={HISTORY_DATABASE_URL_ENV: SECRET_URL})
        store = PostgreSQLExecutionHistoryStore(SECRET_URL)
        for rendered in (repr(config), str(config), repr(store), str(store)):
            for hidden in ("top-secret", "operator", "db.invalid"):
                self.assertNotIn(hidden, rendered)

    def test_user_facing_failure_messages_contain_no_technical_detail(self):
        store = PostgreSQLExecutionHistoryStore(SECRET_URL, connector=lambda _: (_ for _ in ()).throw(OSError("down")))
        with mock.patch("services.execution_history.build_execution_history_store", return_value=store):
            messages = [
                record_execution_plan(plan_fixture("PLAN-SECRET"))["message"],
                list_recorded_plans()["message"],
                update_execution_item("PLAN-SECRET", "C-001", "실행", 1)["message"],
            ]
        for message in messages:
            for hidden in ("top-secret", "db.invalid", "5432", "SELECT", "INSERT", "Traceback", "psycopg", "OSError"):
                self.assertNotIn(hidden, message)


class OperatorToolSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "history.sqlite3"
        self.assertTrue(record_execution_plan(plan_fixture("PLAN-BACKUP"), self.source)["ok"])
        self.assertTrue(update_execution_item("PLAN-BACKUP", "C-001", "실행", 5, db_path=self.source)["ok"])

    def tearDown(self):
        self.temp.cleanup()

    def _backup(self, *extra: str) -> tuple[int, str, Path]:
        target = self.root / "backup.json"
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"VARO_HISTORY_DB_PATH": str(self.source)}), redirect_stdout(output):
            code = backup_cli.main(["--output", str(target), *extra])
        return code, output.getvalue(), target

    def test_check_tool_prints_state_but_never_connection_details(self):
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"VARO_HISTORY_DB_PATH": str(self.source)}), redirect_stdout(output):
            code = check_cli.main([])
        text = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Backend: SQLite", text)
        self.assertIn("Connection: OK", text)
        self.assertIn(f"Schema version: {SCHEMA_VERSION}", text)
        self.assertIn("Schema: OK", text)
        for hidden in (str(self.source), "sqlite3", "password", "PLAN-BACKUP"):
            self.assertNotIn(hidden, text)

    def test_check_tool_reports_missing_staging_configuration_as_unverified(self):
        output = io.StringIO()
        with mock.patch.dict(os.environ, {HISTORY_TEST_DATABASE_URL_ENV: ""}), redirect_stdout(output):
            code = check_cli.main(["--target", "staging"])
        self.assertEqual(code, 3)
        self.assertIn("PostgreSQL staging URL not configured.", output.getvalue())

    def test_validation_tool_reports_unverified_instead_of_claiming_success(self):
        output = io.StringIO()
        with mock.patch.dict(os.environ, {HISTORY_TEST_DATABASE_URL_ENV: ""}), redirect_stdout(output):
            code = validate_cli.main([])
        text = output.getvalue()
        self.assertEqual(code, 3)
        self.assertIn("PostgreSQL staging URL not configured.", text)
        self.assertIn("미검증", text)
        self.assertNotIn("PASS", text)

    def test_validation_tool_refuses_a_staging_url_equal_to_production(self):
        output = io.StringIO()
        environment = {HISTORY_DATABASE_URL_ENV: SECRET_URL, HISTORY_TEST_DATABASE_URL_ENV: SECRET_URL}
        with mock.patch.dict(os.environ, environment), redirect_stdout(output):
            code = validate_cli.main([])
        self.assertEqual(code, 2)
        self.assertNotIn("top-secret", output.getvalue())

    def test_validation_records_use_a_dedicated_namespace(self):
        plan = validate_cli.staging_plan(f"{validate_cli.CHECK_NAMESPACE}-abc")
        self.assertTrue(plan["plan_id"].startswith(validate_cli.CHECK_NAMESPACE))
        self.assertTrue(all(item["candidate_id"].startswith("STAGING-") for item in plan["items"]))

    def test_backup_writes_a_verified_snapshot_without_secrets(self):
        code, text, target = self._backup()
        document = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(document["format"], backup_cli.BACKUP_FORMAT)
        self.assertEqual(document["schema_version"], SCHEMA_VERSION)
        self.assertEqual(len(document["plans"]), 1)
        self.assertEqual(len(document["events"]), 1)
        self.assertIn("무결성 검증: OK", text)
        for hidden in (str(self.source), "password", "PLAN-BACKUP"):
            self.assertNotIn(hidden, text)

    def test_backup_refuses_to_overwrite_an_existing_file_by_default(self):
        self._backup()
        code, text, target = self._backup()
        self.assertEqual(code, 2)
        self.assertIn("--overwrite", text)
        self.assertEqual(self._backup("--overwrite")[0], 0)

    def test_restore_refuses_to_run_without_a_dedicated_destination(self):
        _, _, backup_path = self._backup()
        output = io.StringIO()
        with mock.patch.dict(os.environ, {HISTORY_RESTORE_DATABASE_URL_ENV: ""}), redirect_stdout(output):
            code = restore_cli.main(["--input", str(backup_path), "--confirm-database", "varo", "--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn(HISTORY_RESTORE_DATABASE_URL_ENV, output.getvalue())

    def test_restore_never_targets_the_deployment_database(self):
        _, _, backup_path = self._backup()
        output = io.StringIO()
        environment = {HISTORY_DATABASE_URL_ENV: SECRET_URL, HISTORY_RESTORE_DATABASE_URL_ENV: SECRET_URL}
        with mock.patch.dict(os.environ, environment), redirect_stdout(output):
            code = restore_cli.main([
                "--input", str(backup_path), "--confirm-database", "varo_production", "--apply",
            ])
        self.assertEqual(code, 2)
        self.assertNotIn("top-secret", output.getvalue())

    def test_restore_requires_the_destination_name_to_be_confirmed(self):
        _, _, backup_path = self._backup()
        output = io.StringIO()
        with mock.patch.dict(os.environ, {HISTORY_RESTORE_DATABASE_URL_ENV: STAGING_URL}), redirect_stdout(output):
            code = restore_cli.main([
                "--input", str(backup_path), "--confirm-database", "wrong_name", "--dry-run",
            ])
        self.assertEqual(code, 2)
        self.assertIn("--confirm-database", output.getvalue())

    def _restore(self, backup_path: Path, destination, *mode: str) -> tuple[int, str]:
        output = io.StringIO()
        with mock.patch.dict(os.environ, {HISTORY_RESTORE_DATABASE_URL_ENV: STAGING_URL}), mock.patch.object(
            restore_cli, "build_execution_history_store", return_value=destination,
        ), redirect_stdout(output):
            code = restore_cli.main([
                "--input", str(backup_path), "--confirm-database", "varo_staging", *mode,
            ])
        return code, output.getvalue()

    def test_restore_dry_run_writes_nothing_and_apply_never_overwrites(self):
        _, _, backup_path = self._backup()
        destination_path = self.root / "destination.sqlite3"
        destination = SQLiteExecutionHistoryStore(destination_path)

        code, text = self._restore(backup_path, destination, "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("변경하지 않았습니다", text)
        self.assertFalse(destination_path.exists())

        code, text = self._restore(backup_path, destination, "--apply")
        self.assertEqual(code, 0)
        self.assertIn("복원 완료: 계획 1건", text)
        self.assertTrue(update_execution_item(
            "PLAN-BACKUP", "C-001", "일부 실행", 2, db_path=destination_path,
        )["ok"])

        code, text = self._restore(backup_path, destination, "--apply")
        self.assertEqual(code, 0)
        self.assertIn("대상 중복 계획 1건", text)
        self.assertIn("복원 완료: 계획 0건", text)
        snapshot = destination.read_snapshot()
        self.assertEqual(len(snapshot["plans"]), 1)
        self.assertEqual(snapshot["items"][0]["actual_qty"], 2, "복원이 대상의 최신 값을 덮어썼습니다.")

    def test_restore_rejects_a_foreign_or_damaged_backup_file(self):
        broken = self.root / "broken.json"
        broken.write_text(json.dumps({"format": "other", "plans": []}), encoding="utf-8")
        destination = SQLiteExecutionHistoryStore(self.root / "unused.sqlite3")
        code, text = self._restore(broken, destination, "--apply")
        self.assertEqual(code, 2)
        self.assertIn("백업 파일 형식", text)


if __name__ == "__main__":
    unittest.main()
