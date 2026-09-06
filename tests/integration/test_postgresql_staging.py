"""End-to-end execution-history checks against a real staging PostgreSQL server.

Collected only when ``VARO_HISTORY_TEST_DATABASE_URL`` is set (see conftest).
Records are namespaced with ``VARO-STAGING-CHECK`` and removed afterwards; no
table is ever dropped or truncated.
"""
from __future__ import annotations

import uuid

import pytest

from services.execution_history import (
    execution_history_health,
    get_recorded_plan,
    inspect_execution_history_schema,
    list_item_events,
    record_execution_plan,
    update_execution_item,
)
from services.execution_history_store import build_execution_history_store
from tools.validate_postgresql_history import CHECK_NAMESPACE, _Report, _run_checks, staging_plan


@pytest.fixture(scope="function")
def staging_report():
    report = _Report()
    created: list[str] = []
    try:
        _run_checks(report, created, read_only=False)
    finally:
        if created:
            build_execution_history_store().delete_plans(created)
    return report


def test_every_staging_check_passes(staging_report):
    failures = [(name, detail) for name, ok, detail in staging_report.results if not ok]
    assert not failures, failures


def test_staging_suite_covers_the_deployment_critical_paths(staging_report):
    names = {name for name, _, _ in staging_report.results}
    assert {
        "connect",
        "schema initialize (idempotent)",
        "schema structure (PK/FK/index/version)",
        "write (plan + items)",
        "read back",
        "update + audit event",
        "duplicate plan is safe",
        "transaction rollback",
        "concurrent duplicate save",
        "concurrent item update (lost update)",
        "pagination",
        "reconnect after restart",
        "calibration export",
    } <= names


def test_server_schema_matches_the_expected_structure():
    report = inspect_execution_history_schema()
    assert report["ok"], report["issues"]
    assert report["schema_version"] == report["expected_schema_version"]
    assert all(report["indexes"].values())


def test_health_check_reports_a_live_server():
    health = execution_history_health()
    assert health["connection_ok"]
    assert health["ok"], health["message"]
    assert health["latency_ms"] is not None


def test_history_survives_a_simulated_restart():
    from services import execution_history_store

    plan_id = f"{CHECK_NAMESPACE}-{uuid.uuid4().hex[:12]}"
    try:
        assert record_execution_plan(staging_plan(plan_id))["ok"]
        assert update_execution_item(
            plan_id, "STAGING-C-001", "일부 실행", 2,
            outcomes={"actual_transport_cost": 9.0, "actual_saving": 30.0},
        )["ok"]

        # Discard every cached connection object, as a process restart would.
        execution_history_store._cached_store.cache_clear()

        reloaded = get_recorded_plan(plan_id)
        item = reloaded["items"][0]
        assert item["execution_status"] == "partial"
        assert item["actual_qty"] == 2
        assert item["actual_transport_cost"] == 9.0
        assert item["actual_net_benefit"] == 21.0
        assert len(list_item_events(plan_id, "STAGING-C-001")) == 1
    finally:
        build_execution_history_store().delete_plans([plan_id])
