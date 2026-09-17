"""Local simulation history contract tests.

The history must keep exactly two record kinds, stay small, never duplicate a
rerun, and never invent a KPI the pipeline does not produce.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from services.inventory_transition_service import run_inventory_scenario
from services.simulation_history import (
    HISTORY_SCHEMA_VERSION,
    STATUS_COMPLETED,
    STATUS_FAILED,
    build_run_key,
    build_run_records,
    get_run_routes,
    get_simulation_run,
    history_db_path,
    history_storage_info,
    initialize_history_storage,
    list_simulation_runs,
    record_failed_run,
    record_simulation_run,
    resolve_snapshot_label,
)
from tests.fixtures import sample_workbook


def _routes() -> list[dict]:
    return [
        {
            "route_id": "R001", "recommendation_id": "REC-1", "product_id": "P001",
            "product_name": "우유", "source_id": "S001", "source_name": "1호점",
            "target_id": "S002", "target_name": "2호점", "dc_id": None,
            "route_type": "DIRECT", "recommended_qty": 12, "move_cost": 5200.0,
            "expected_saving": 31000.0, "varo_action": "재고 이동",
            "varo_final_rank": 1, "vhs_rank": 2, "greedy_rank": 1, "pareto_rank": 1,
        },
        {
            "route_id": "R002", "recommendation_id": "REC-2", "product_id": "P002",
            "product_name": "요구르트", "source_id": "S002", "source_name": "2호점",
            "target_id": "S001", "target_name": "1호점", "dc_id": "DC01",
            "route_type": "VIA_DC", "recommended_qty": 6, "move_cost": 8100.0,
            "expected_saving": 12000.0, "varo_action": "재고 이동",
            "varo_final_rank": 2, "vhs_rank": 1, "greedy_rank": 3, "pareto_rank": 1,
        },
    ]


class SimulationHistoryStorageTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def _records(self):
        data = sample_workbook()
        scenario = run_inventory_scenario(data, _routes())
        return build_run_records(
            data_signature="sig-1",
            data_source_type="샘플 추천 데이터",
            data_name="varo_sample.xlsx",
            snapshot_label=resolve_snapshot_label(data),
            filters={"센터/점포": "전체 센터/점포", "상품 범위": "전체 상품"},
            candidate_count=40,
            scoped_candidate_count=12,
            simulation_routes=_routes(),
            scenario=scenario,
            pipeline_result={"result_basis": "V2 재계산 기준"},
        )

    def test_storage_initializes_two_tables_only(self):
        initialize_history_storage(self.directory)
        import contextlib
        import sqlite3

        with contextlib.closing(sqlite3.connect(str(history_db_path(self.directory)))) as connection:
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        self.assertEqual(tables, {"simulation_runs", "simulation_selected_routes"})

    def test_summary_and_final_routes_are_stored(self):
        summary, routes = self._records()
        run_id = record_simulation_run(summary, routes, run_key="key-1", directory=self.directory)
        self.assertIsNotNone(run_id)
        stored = get_simulation_run(run_id, self.directory)
        self.assertEqual(stored["status"], STATUS_COMPLETED)
        self.assertEqual(stored["schema_version"], HISTORY_SCHEMA_VERSION)
        self.assertEqual(stored["data_signature"], "sig-1")
        self.assertEqual(stored["candidate_count"], 40)
        self.assertEqual(stored["scoped_candidate_count"], 12)
        self.assertEqual(stored["requested_route_count"], 2)
        self.assertIsNotNone(stored["total_moved_quantity"])
        stored_routes = get_run_routes(run_id, self.directory)
        self.assertEqual(len(stored_routes), 2)
        self.assertEqual(stored_routes[0]["route_id"], "R001")
        self.assertEqual(stored_routes[0]["recommendation_id"], "REC-1")
        self.assertEqual(stored_routes[0]["varo_final_rank"], 1.0)
        self.assertEqual(stored_routes[0]["vhs_rank"], 2.0)
        self.assertEqual(stored_routes[0]["greedy_rank"], 1.0)

    def test_same_run_key_is_never_stored_twice(self):
        summary, routes = self._records()
        first = record_simulation_run(summary, routes, run_key="same", directory=self.directory)
        second = record_simulation_run(summary, routes, run_key="same", directory=self.directory)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(list_simulation_runs(directory=self.directory)), 1)

    def test_run_key_changes_with_a_new_execution(self):
        routes = _routes()
        first = build_run_key("sig-1", 1, routes, "단일 경로")
        rerun = build_run_key("sig-1", 1, routes, "단일 경로")
        second = build_run_key("sig-1", 2, routes, "단일 경로")
        self.assertEqual(first, rerun)
        self.assertNotEqual(first, second)

    def test_failed_run_is_recorded_with_minimal_information(self):
        run_id = record_failed_run(
            run_key="failed-1",
            error_summary="ValueError: 재고 데이터 없음",
            data_signature="sig-1",
            data_name="varo_sample.xlsx",
            directory=self.directory,
        )
        stored = get_simulation_run(run_id, self.directory)
        self.assertEqual(stored["status"], STATUS_FAILED)
        self.assertIn("ValueError", stored["error_summary"])
        self.assertIsNone(stored["total_expected_saving"])
        self.assertEqual(get_run_routes(run_id, self.directory), [])

    def test_only_summary_and_selected_routes_are_persisted(self):
        summary, routes = self._records()
        self.assertNotIn("transitions", summary)
        self.assertNotIn("baseline_records", summary)
        self.assertNotIn("final_records", summary)
        for row in routes:
            self.assertNotIn("transitions", row)
            self.assertLessEqual(len(row), 20)

    def test_one_run_stays_small_on_disk(self):
        summary, routes = self._records()
        initialize_history_storage(self.directory)
        before = history_storage_info(self.directory)["size_bytes"]
        record_simulation_run(summary, routes, run_key="size-1", directory=self.directory)
        after = history_storage_info(self.directory)
        self.assertLess(after["size_bytes"] - before, 200 * 1024)
        self.assertEqual(after["run_count"], 1)
        self.assertEqual(after["route_row_count"], 2)

    def test_summary_keeps_real_kpis_and_scopes(self):
        summary, _ = self._records()
        filters = json.loads(summary["filters_json"])
        self.assertEqual(filters["상품 범위"], "전체 상품")
        self.assertIn("1호점", json.loads(summary["node_scope_json"]))
        self.assertIn("우유", json.loads(summary["product_scope_json"]))
        self.assertEqual(summary["strategy_basis"], "V2 재계산 기준")
        self.assertIsNotNone(summary["calculation_version"])

    def test_snapshot_label_is_none_without_a_real_date_column(self):
        self.assertIsNone(resolve_snapshot_label(sample_workbook()))

    def test_snapshot_label_uses_a_real_date_column_when_present(self):
        data = {"inventory": pd.DataFrame({"snapshot_date": ["2026-09-01", "2026-09-03"]})}
        self.assertEqual(resolve_snapshot_label(data), "2026-09-01 ~ 2026-09-03")


if __name__ == "__main__":
    unittest.main()
