"""Concurrency, failure isolation, and algorithm-independence of the history store."""
from __future__ import annotations

import concurrent.futures
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.app_state import CANONICAL_DATA_KEYS
from services.data_application import load_and_apply
from services.execution_history import (
    execution_history_metrics,
    export_execution_history_csv,
    get_recorded_plan,
    list_item_events,
    list_recorded_plans,
    record_execution_plan,
    update_execution_item,
)
from services.execution_history_store import PostgreSQLExecutionHistoryStore
from test_execution_history_backends import CompatConnector, plan_fixture
from tests.fixtures import sample_workbook, workbook_excel_bytes

SERVER_URL = "postgresql://operator:top-secret@db.invalid:5432/varo_production"


class ConcurrentWriteTests(unittest.TestCase):
    """Two operators acting at the same moment must not corrupt the history."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "history.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _parallel(action, arguments):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(arguments)) as pool:
            return list(pool.map(action, arguments))

    def test_simultaneous_saves_of_one_plan_create_exactly_one_record(self):
        plan = plan_fixture("PLAN-RACE")
        outcomes = self._parallel(lambda _: record_execution_plan(plan, self.path), range(2))
        self.assertEqual(sorted(result["code"] for result in outcomes), ["duplicate", "recorded"])
        self.assertTrue(all(result["ok"] for result in outcomes))
        self.assertEqual(len(list_recorded_plans(self.path)["plans"]), 1)
        self.assertEqual(len(get_recorded_plan("PLAN-RACE", self.path)["items"]), 1)

    def test_simultaneous_saves_on_the_server_backend_are_equally_safe(self):
        connector = CompatConnector(Path(self.temp.name) / "server.sqlite3")
        store = PostgreSQLExecutionHistoryStore(SERVER_URL, connector=connector)
        plan = plan_fixture("PLAN-SERVER-RACE")
        with mock.patch("services.execution_history.build_execution_history_store", return_value=store):
            outcomes = self._parallel(lambda _: record_execution_plan(plan), range(2))
            stored = get_recorded_plan("PLAN-SERVER-RACE")
        self.assertEqual(sorted(result["code"] for result in outcomes), ["duplicate", "recorded"])
        self.assertEqual(len(stored["items"]), 1)

    def test_simultaneous_item_updates_are_serialized_without_a_lost_update(self):
        self.assertTrue(record_execution_plan(plan_fixture("PLAN-LOST"), self.path)["ok"])

        outcomes = self._parallel(
            lambda quantity: update_execution_item(
                "PLAN-LOST", "C-001", "실행", quantity, db_path=self.path,
            ),
            (4, 6),
        )

        self.assertTrue(all(result["ok"] for result in outcomes))
        events = list_item_events("PLAN-LOST", "C-001", self.path)
        self.assertEqual(len(events), 2, "동시 수정 중 감사 기록이 사라졌습니다.")
        self.assertIsNone(events[0]["previous_actual_qty"])
        self.assertEqual(
            events[1]["previous_actual_qty"], events[0]["new_actual_qty"],
            "두 번째 수정이 첫 번째 결과를 보지 못했습니다 (lost update).",
        )
        stored = get_recorded_plan("PLAN-LOST", self.path)["items"][0]
        self.assertEqual(stored["actual_qty"], events[1]["new_actual_qty"])
        self.assertIn(stored["actual_qty"], (4, 6))

    def test_concurrent_reads_during_writes_never_return_a_partial_plan(self):
        self.assertTrue(record_execution_plan(plan_fixture("PLAN-READ"), self.path)["ok"])

        def read(_: int) -> int:
            loaded = get_recorded_plan("PLAN-READ", self.path)
            return len(loaded["items"]) if loaded["ok"] else -1

        def write(index: int) -> bool:
            return record_execution_plan(plan_fixture(f"PLAN-WRITE-{index}"), self.path)["ok"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            readers = [pool.submit(read, index) for index in range(4)]
            writers = [pool.submit(write, index) for index in range(4)]
        self.assertTrue(all(future.result() for future in writers))
        self.assertTrue(all(future.result() == 1 for future in readers))


class ConnectionInterruptionTests(unittest.TestCase):
    """A dead database degrades the history feature only, and recovers by itself."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.connector = CompatConnector(self.root / "server.sqlite3")
        self.failures = {"remaining": 0}
        self.store = PostgreSQLExecutionHistoryStore(
            SERVER_URL, connector=self._connect, connect_retries=0,
        )

    def tearDown(self):
        self.temp.cleanup()

    def _connect(self, database_url: str):
        if self.failures["remaining"] > 0:
            self.failures["remaining"] -= 1
            raise OSError("connection reset by peer")
        return self.connector(database_url)

    def _patch(self):
        return mock.patch("services.execution_history.build_execution_history_store", return_value=self.store)

    def test_connection_loss_is_reported_safely_and_the_next_request_reconnects(self):
        with self._patch():
            self.assertTrue(record_execution_plan(plan_fixture("PLAN-DROP"))["ok"])
            self.failures["remaining"] = 1
            during_outage = get_recorded_plan("PLAN-DROP")
            after_outage = get_recorded_plan("PLAN-DROP")
        self.assertFalse(during_outage["ok"])
        self.assertEqual(during_outage["message"], "실행 기록을 불러오지 못했습니다.")
        self.assertTrue(after_outage["ok"])
        self.assertEqual(after_outage["items"][0]["planned_qty"], 5)

    def test_a_write_that_could_not_start_is_never_reported_as_saved(self):
        with self._patch():
            self.failures["remaining"] = 1
            result = record_execution_plan(plan_fixture("PLAN-UNSAVED"))
            self.failures["remaining"] = 0
            stored = get_recorded_plan("PLAN-UNSAVED")
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "storage_error")
        self.assertEqual(stored["code"], "not_found")
        for hidden in ("top-secret", "db.invalid", "connection reset"):
            self.assertNotIn(hidden, result["message"])

    def test_failure_inside_a_transaction_leaves_no_partial_write(self):
        with self._patch():
            self.assertTrue(record_execution_plan(plan_fixture("PLAN-PARTIAL"))["ok"])
            self.connector.state["fail_query"] = "UPDATE execution_plans SET updated_at"
            result = update_execution_item("PLAN-PARTIAL", "C-001", "실행", 5)
            self.connector.state.pop("fail_query")
            stored = get_recorded_plan("PLAN-PARTIAL")
            events = list_item_events("PLAN-PARTIAL", "C-001")
        self.assertFalse(result["ok"])
        self.assertEqual(stored["items"][0]["execution_status"], "unconfirmed")
        self.assertIsNone(stored["items"][0]["actual_qty"])
        self.assertEqual(events, [])

    def test_repeated_reruns_close_every_connection_they_open(self):
        with self._patch():
            self.assertTrue(record_execution_plan(plan_fixture("PLAN-RERUN"))["ok"])
            for _ in range(3):
                # One Streamlit rerun of the history panel.
                list_recorded_plans()
                get_recorded_plan("PLAN-RERUN")
                execution_history_metrics()
                export_execution_history_csv()
        self.assertEqual(
            self.connector.state["opened"], self.connector.state["closed"],
            "재실행마다 연결이 누적되면 상용 환경에서 연결이 고갈됩니다.",
        )


class RecommendationIndependenceTests(unittest.TestCase):
    """Storage work must not move a single recommendation number."""

    @classmethod
    def setUpClass(cls):
        cls.workbook = workbook_excel_bytes(sample_workbook())

    @staticmethod
    def _signature(state: dict) -> dict:
        result = state.get("varo_pipeline_result") or {}
        plan = result.get("execution_plan") or {}
        return {
            "recommendation_count": len(result.get("recommendations") or []),
            "candidate_count": len(result.get("candidates") or []),
            "plan_status": plan.get("plan_status"),
            "total_actions": len(plan.get("items") or []),
            "total_planned_qty": sum(int(item["planned_qty"]) for item in plan.get("items") or []),
            "total_cost": plan.get("total_cost"),
            "total_expected_saving": plan.get("total_expected_saving"),
            "total_net_benefit": plan.get("total_net_benefit"),
            "data_signature": plan.get("data_signature"),
            "algorithm_version": plan.get("algorithm_version"),
            "vhs_scores": [item.get("vhs_score") for item in plan.get("items") or []],
        }

    def _run(self) -> dict:
        state: dict = {}
        self.assertTrue(load_and_apply(state, self.workbook, "anonymous.xlsx", "샘플 추천 데이터"))
        self.assertTrue(set(CANONICAL_DATA_KEYS) <= set(state))
        return self._signature(state)

    def test_results_are_identical_with_local_storage_and_with_a_dead_server(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.dict(os.environ, {
                "VARO_HISTORY_DB_PATH": str(Path(temp) / "history.sqlite3"),
                "VARO_HISTORY_DATABASE_URL": "",
            }):
                local = self._run()
            with mock.patch.dict(os.environ, {"VARO_HISTORY_DATABASE_URL": SERVER_URL}), mock.patch.object(
                PostgreSQLExecutionHistoryStore, "_open", side_effect=OSError("server down"),
            ):
                with_dead_server = self._run()
                # The history feature is unavailable...
                self.assertFalse(list_recorded_plans()["ok"])
        # ...but every recommendation number is unchanged.
        self.assertEqual(local, with_dead_server)
        self.assertGreater(local["recommendation_count"], 0)
        self.assertGreater(local["total_actions"], 0)

    def test_recording_a_plan_does_not_change_the_plan_it_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.sqlite3"
            state: dict = {}
            self.assertTrue(load_and_apply(state, self.workbook, "anonymous.xlsx", "샘플 추천 데이터"))
            before = self._signature(state)
            plan = state["varo_pipeline_result"]["execution_plan"]

            self.assertTrue(record_execution_plan(plan, path)["ok"])
            self.assertTrue(update_execution_item(
                plan["plan_id"], plan["items"][0]["candidate_id"], "실행",
                plan["items"][0]["planned_qty"], db_path=path,
            )["ok"])

            self.assertEqual(self._signature(state), before)
            stored = get_recorded_plan(plan["plan_id"], path)
            self.assertEqual(len(stored["items"]), before["total_actions"])
            self.assertEqual(
                sum(int(item["planned_qty"]) for item in stored["items"]),
                before["total_planned_qty"],
            )


if __name__ == "__main__":
    unittest.main()
