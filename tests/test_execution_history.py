from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.app_state import clear_applied_data
from services.execution_history import (
    SCHEMA_VERSION,
    execution_history_metrics,
    export_execution_history_csv,
    get_recorded_plan,
    initialize_history_store,
    list_item_events,
    list_recorded_plans,
    record_execution_plan,
    update_execution_item,
)
from services.execution_history_store import SQLiteExecutionHistoryStore


def plan_fixture(plan_id: str = "PLAN-ANON-001") -> dict:
    return {
        "plan_id": plan_id,
        "algorithm_version": "execution-plan-1.0",
        "data_signature": "anonymous-data-signature",
        "created_at": "2026-09-02T00:00:00+00:00",
        "plan_status": "실행 가능",
        "selected_candidates": 2,
        "total_transfer_qty": 15,
        "total_cost": 3000,
        "total_expected_saving": 13000,
        "total_net_benefit": 10000,
        "validation": {"valid": True},
        "items": [
            {
                "candidate_id": "C-ANON-001", "algorithm_version": "vhs-2.2",
                "source_id": "S01", "source_name": "가상점포 01",
                "target_id": "S02", "target_name": "가상점포 02",
                "product_id": "P01", "product_name": "가상상품 01",
                "route_type": "DIRECT", "dc_id": None, "planned_qty": 10,
                "planned_cost": 2000, "planned_expected_saving": 9000,
                "planned_net_benefit": 7000, "vhs_score": 81,
                "robustness_status": "안정", "confidence_score": 77,
                "net_benefit_score": 85, "pareto_status": "비지배",
            },
            {
                "candidate_id": "C-ANON-002", "algorithm_version": "vhs-2.2",
                "source_id": "S03", "source_name": "가상점포 03",
                "target_id": "S04", "target_name": "가상점포 04",
                "product_id": "P02", "product_name": "가상상품 02",
                "route_type": "VIA_DC", "dc_id": "DC01", "planned_qty": 5,
                "planned_cost": 1000, "planned_expected_saving": 4000,
                "planned_net_benefit": 3000, "vhs_score": 72,
                "robustness_status": "검토 필요", "confidence_score": 68,
            },
        ],
    }


class ExecutionHistoryPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "history.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_first_use_creates_database_and_versioned_schema(self):
        result = initialize_history_store(self.db)
        self.assertTrue(result["ok"])
        self.assertTrue(self.db.is_file())
        connection = sqlite3.connect(self.db)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            connection.close()
        self.assertTrue({"execution_plans", "execution_items", "execution_item_events"}.issubset(tables))

    def test_plan_and_items_survive_new_connections(self):
        recorded = record_execution_plan(plan_fixture(), self.db)
        self.assertTrue(recorded["created"])
        loaded = get_recorded_plan("PLAN-ANON-001", self.db)
        self.assertTrue(loaded["ok"])
        self.assertEqual(len(loaded["items"]), 2)
        self.assertEqual(loaded["plan"]["total_planned_qty"], 15)

    def test_plan_and_candidate_lineage_are_preserved(self):
        record_execution_plan(plan_fixture(), self.db)
        loaded = get_recorded_plan("PLAN-ANON-001", self.db)
        self.assertEqual(loaded["plan"]["algorithm_version"], "execution-plan-1.0")
        self.assertEqual(loaded["plan"]["candidate_algorithm_version"], "vhs-2.2")
        self.assertEqual(loaded["plan"]["data_signature"], "anonymous-data-signature")
        self.assertEqual(loaded["items"][0]["plan_id"], "PLAN-ANON-001")
        self.assertEqual({item["candidate_id"] for item in loaded["items"]}, {"C-ANON-001", "C-ANON-002"})

    def test_same_plan_is_not_inserted_twice(self):
        first = record_execution_plan(plan_fixture(), self.db)
        second = record_execution_plan(plan_fixture(), self.db)
        self.assertEqual(first["code"], "recorded")
        self.assertEqual(second["code"], "duplicate")
        self.assertEqual(len(list_recorded_plans(self.db)["plans"]), 1)

    def test_unvalidated_plan_is_not_recorded(self):
        plan = plan_fixture()
        plan["validation"] = {"valid": False}
        result = record_execution_plan(plan, self.db)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "invalid_plan")
        self.assertFalse(self.db.exists())

    def test_plan_and_items_rollback_together(self):
        original = SQLiteExecutionHistoryStore._insert_mappings

        def fail_items(store, connection, table, columns, values):
            if table == "execution_items":
                raise sqlite3.OperationalError("test")
            return original(store, connection, table, columns, values)

        with mock.patch.object(SQLiteExecutionHistoryStore, "_insert_mappings", new=fail_items):
            result = record_execution_plan(plan_fixture(), self.db)
        self.assertFalse(result["ok"])
        self.assertEqual(list_recorded_plans(self.db)["plans"], [])

    def test_corrupt_database_returns_safe_message(self):
        self.db.write_bytes(b"not-a-sqlite-database")
        result = list_recorded_plans(self.db)
        self.assertFalse(result["ok"])
        self.assertNotIn(str(self.db), result["message"])


class ExecutionHistoryUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "history.sqlite3"
        record_execution_plan(plan_fixture(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def update(self, status, qty=None, **kwargs):
        return update_execution_item(
            "PLAN-ANON-001", "C-ANON-001", status, qty, db_path=self.db, **kwargs,
        )

    def item(self):
        return next(
            item for item in get_recorded_plan("PLAN-ANON-001", self.db)["items"]
            if item["candidate_id"] == "C-ANON-001"
        )

    def test_all_execution_statuses_can_be_recorded(self):
        cases = (("미확인", None, "unconfirmed"), ("실행", 10, "executed"),
                 ("일부 실행", 6, "partial"), ("미실행", 0, "not_executed"),
                 ("취소", 0, "cancelled"))
        for label, qty, code in cases:
            with self.subTest(label=label):
                self.assertTrue(self.update(label, qty)["ok"])
                self.assertEqual(self.item()["execution_status"], code)

    def test_actual_quantity_validation_and_over_plan_warning(self):
        for bad in (-1, 2.5, "not-number", float("inf")):
            with self.subTest(value=bad):
                self.assertFalse(self.update("실행", bad)["ok"])
        self.assertFalse(self.update("실행", 0)["ok"])
        result = self.update("실행", 13)
        self.assertTrue(result["ok"])
        self.assertIn("3개 많이", result["warning"])
        self.assertEqual(self.item()["actual_qty"], 13)

    def test_not_executed_reason_and_note_are_separate(self):
        result = self.update("미실행", 0, nonexecution_reason="운송 불가", operator_note="가상 메모")
        self.assertTrue(result["ok"])
        item = self.item()
        self.assertEqual(item["nonexecution_reason"], "transport_unavailable")
        self.assertEqual(item["operator_note"], "가상 메모")

    def test_optional_outcomes_remain_null_instead_of_zero(self):
        self.assertTrue(self.update("실행", 10)["ok"])
        item = self.item()
        for field in ("post_source_stock", "post_destination_stock", "actual_sales_qty",
                      "actual_waste_qty", "actual_stockout_occurred", "actual_stockout_qty", "actual_transport_cost",
                      "actual_saving", "actual_net_benefit"):
            self.assertIsNone(item[field])
        self.assertIsNone(item["cost_difference"])

    def test_actual_outcomes_and_expected_comparisons(self):
        result = self.update(
            "실행", 9,
            outcomes={
                "post_source_stock": 21, "post_destination_stock": 34,
                "actual_sales_qty": 8, "actual_waste_qty": 1,
                "actual_stockout_occurred": "없음", "actual_stockout_qty": 0, "actual_transport_cost": 2500,
                "actual_saving": 8500,
            },
        )
        self.assertTrue(result["ok"])
        item = self.item()
        self.assertEqual(item["actual_net_benefit"], 6000)
        self.assertEqual(item["actual_stockout_occurred"], 0)
        self.assertEqual(item["cost_difference"], 500)
        self.assertEqual(item["saving_difference"], -500)
        self.assertEqual(item["net_benefit_difference"], -1000)

    def test_status_only_update_preserves_existing_outcomes(self):
        self.update(
            "일부 실행", 5,
            outcomes={"actual_transport_cost": 2500, "actual_saving": 8500},
        )
        self.update("실행", 10)
        item = self.item()
        self.assertEqual(item["actual_transport_cost"], 2500)
        self.assertEqual(item["actual_saving"], 8500)
        self.assertEqual(item["actual_net_benefit"], 6000)

    def test_updates_append_audit_events_without_deleting_history(self):
        self.update("일부 실행", 4, nonexecution_reason="현장 판단")
        self.update("실행", 10)
        events = list_item_events("PLAN-ANON-001", "C-ANON-001", self.db)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["previous_status"], "unconfirmed")
        self.assertEqual(events[1]["new_status"], "executed")


class ExecutionHistoryMetricsAndExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "history.sqlite3"
        record_execution_plan(plan_fixture(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_metrics_are_unavailable_without_actual_samples(self):
        metrics = execution_history_metrics(self.db)
        self.assertIsNone(metrics["execution_rate"])
        self.assertIsNone(metrics["quantity_adherence_rate"])
        self.assertEqual(metrics["cost_error"]["sample_count"], 0)
        self.assertIsNone(metrics["cost_error"]["mean_error"])

    def test_metrics_report_sample_counts(self):
        update_execution_item(
            "PLAN-ANON-001", "C-ANON-001", "일부 실행", 5,
            outcomes={"actual_transport_cost": 2200, "actual_saving": 8000}, db_path=self.db,
        )
        update_execution_item(
            "PLAN-ANON-001", "C-ANON-002", "미실행", 0, db_path=self.db,
        )
        metrics = execution_history_metrics(self.db)
        self.assertEqual(metrics["confirmed_items"], 2)
        self.assertEqual(metrics["execution_rate"], 50.0)
        self.assertEqual(metrics["partial_rate"], 50.0)
        self.assertEqual(metrics["quantity_adherence_rate"], 33.33)
        self.assertEqual(metrics["net_benefit_error"]["sample_count"], 1)

    def test_csv_export_has_utf8_bom_and_no_database_path(self):
        update_execution_item(
            "PLAN-ANON-001", "C-ANON-001", "실행", 10,
            outcomes={"actual_transport_cost": 2100, "actual_saving": 8800}, db_path=self.db,
        )
        result = export_execution_history_csv(self.db)
        self.assertTrue(result["ok"])
        self.assertEqual(result["row_count"], 2)
        self.assertTrue(result["data"].startswith(b"\xef\xbb\xbf"))
        text = result["data"].decode("utf-8-sig")
        self.assertIn("plan_algorithm_version", text)
        self.assertIn("실행", text)
        self.assertNotIn(str(self.db), text)
        self.assertNotIn("sqlite", text.lower())

    def test_clearing_current_data_does_not_delete_history(self):
        state = {"varo_data": {"stores": [1]}, "varo_pipeline_result": {"execution_plan": plan_fixture()}}
        clear_applied_data(state)
        self.assertIsNone(state["varo_data"])
        self.assertTrue(get_recorded_plan("PLAN-ANON-001", self.db)["ok"])

    def test_runtime_databases_are_ignored_by_git(self):
        ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("runtime_data/", ignore)
        self.assertIn("*.sqlite3", ignore)


if __name__ == "__main__":
    unittest.main()
