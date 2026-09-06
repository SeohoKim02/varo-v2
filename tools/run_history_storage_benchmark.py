"""Measure execution-history storage performance only.

Recommendation quality and speed are measured by the algorithm benchmarks; this
tool deliberately reports database timings alone so the two are never mixed.

Default fixture is 1,000 plans x 30 items (30,000 items), which is well beyond
the expected first-year operating volume.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.execution_history import (  # noqa: E402
    export_execution_history_csv,
    list_recorded_plans,
    record_execution_plan,
    update_execution_item,
)
from services.execution_history_migration import migrate_sqlite_history  # noqa: E402
from services.execution_history_store import SQLiteExecutionHistoryStore  # noqa: E402


def _plan(plan_id: str, item_count: int) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "algorithm_version": "benchmark",
        "data_signature": "benchmark-signature",
        "created_at": "2026-01-01T00:00:00+00:00",
        "plan_status": "추천 가능",
        "total_cost": 10.0 * item_count,
        "total_expected_saving": 40.0 * item_count,
        "total_net_benefit": 30.0 * item_count,
        "validation": {"valid": True},
        "items": [{
            "candidate_id": f"{plan_id}-C{index:03d}",
            "algorithm_version": "benchmark",
            "source_id": f"S{index % 40:03d}", "source_name": f"출발 {index % 40:03d}",
            "target_id": f"T{index % 37:03d}", "target_name": f"도착 {index % 37:03d}",
            "product_id": f"P{index % 53:03d}", "product_name": f"상품 {index % 53:03d}",
            "route_type": "DIRECT" if index % 3 else "VIA_DC",
            "dc_id": None if index % 3 else "DC-001",
            "planned_qty": 1 + index % 9,
            "planned_cost": 10.0, "planned_expected_saving": 40.0, "planned_net_benefit": 30.0,
            "vhs_score": 50.0 + index % 40, "robustness_status": "안정", "confidence_score": 70.0,
        } for index in range(item_count)],
    }


def _timed(label: str, action: Any) -> tuple[str, float, Any]:
    started = perf_counter()
    value = action()
    return label, (perf_counter() - started) * 1000.0, value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="실행 이력 저장소(SQLite) 성능을 측정합니다.")
    parser.add_argument("--plans", type=int, default=1000, help="생성할 계획 수")
    parser.add_argument("--items", type=int, default=30, help="계획당 항목 수")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan_count = max(1, int(args.plans))
    item_count = max(1, int(args.items))
    rows = []

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "benchmark_history.sqlite3"

        started = perf_counter()
        for index in range(plan_count):
            result = record_execution_plan(_plan(f"BENCH-{index:06d}", item_count), source)
            if not result.get("ok"):
                print("저장 실패로 측정을 중단했습니다.")
                return 1
        save_ms = (perf_counter() - started) * 1000.0
        rows.append(("계획 저장 (합계)", save_ms))
        rows.append(("계획 저장 (계획 1건 평균)", save_ms / plan_count))

        rows.append(_timed(
            "최근 이력 20건 조회", lambda: list_recorded_plans(source, limit=20),
        )[:2])
        rows.append(_timed(
            "이력 20건 조회 (offset 500)",
            lambda: list_recorded_plans(source, limit=20, offset=min(500, plan_count - 1)),
        )[:2])
        rows.append(_timed(
            "항목 실행 결과 수정 1건",
            lambda: update_execution_item(
                "BENCH-000000", "BENCH-000000-C000", "실행", 3,
                outcomes={"actual_transport_cost": 10.0, "actual_saving": 40.0},
                db_path=source,
            ),
        )[:2])

        tracemalloc.start()
        label, export_ms, exported = _timed(
            "calibration export (전체)", lambda: export_execution_history_csv(source),
        )
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rows.append((label, export_ms))

        destination = SQLiteExecutionHistoryStore(root / "benchmark_destination.sqlite3")
        rows.append(_timed(
            "이관 dry-run", lambda: migrate_sqlite_history(source, destination, dry_run=True),
        )[:2])
        rows.append(_timed(
            "이관 apply", lambda: migrate_sqlite_history(source, destination, dry_run=False),
        )[:2])

        print("Backend: SQLite (임시 파일)")
        print(f"Fixture: 계획 {plan_count:,}건 / 항목 {plan_count * item_count:,}건")
        print(f"저장 파일 크기: {source.stat().st_size / 1024 / 1024:.1f} MB")
        for label, milliseconds in rows:
            print(f"  {label}: {milliseconds:,.1f} ms")
        print(f"내보내기 행 수: {int(exported.get('row_count') or 0):,}")
        print(f"내보내기 바이트: {len(exported.get('data') or b'') / 1024 / 1024:.1f} MB")
        print(f"내보내기 피크 메모리: {peak / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
