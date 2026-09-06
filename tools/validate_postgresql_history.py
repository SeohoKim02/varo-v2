"""Explicit staging-PostgreSQL integration validation for execution history.

Runs the real application code path (service -> store -> psycopg) against the
staging database named by ``VARO_HISTORY_TEST_DATABASE_URL``.  Without that
variable nothing is attempted and nothing is reported as verified.

Every record it writes carries the ``VARO-STAGING-CHECK`` plan-id namespace and
is deleted at the end.  It never targets ``VARO_HISTORY_DATABASE_URL``, never
drops or truncates anything, and never prints a URL, host, or password.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import execution_history_store  # noqa: E402
from services.execution_history import (  # noqa: E402
    execution_history_health,
    export_execution_history_csv,
    get_recorded_plan,
    inspect_execution_history_schema,
    list_item_events,
    list_recorded_plans,
    record_execution_plan,
    update_execution_item,
)
from services.execution_history_config import (  # noqa: E402
    HISTORY_DATABASE_URL_ENV,
    HISTORY_DB_PATH_ENV,
    HISTORY_TEST_DATABASE_URL_ENV,
    HistoryConfigurationError,
    load_staging_test_config,
)
from services.execution_history_store import (  # noqa: E402
    HistoryStoreError,
    build_execution_history_store,
)

CHECK_NAMESPACE = "VARO-STAGING-CHECK"


def staging_plan(plan_id: str, *, planned_qty: int = 5) -> dict[str, Any]:
    """A synthetic, anonymous plan; no real store, product, or operator data."""
    return {
        "plan_id": plan_id,
        "algorithm_version": "staging-check",
        "data_signature": "staging-check-signature",
        "created_at": "2026-01-01T00:00:00+00:00",
        "plan_status": "추천 가능",
        "total_cost": 12.5,
        "total_expected_saving": 50.75,
        "total_net_benefit": 38.25,
        "validation": {"valid": True},
        "items": [{
            "candidate_id": "STAGING-C-001",
            "algorithm_version": "staging-check",
            "source_id": "STAGING-S-A", "source_name": "검증용 출발점",
            "target_id": "STAGING-S-B", "target_name": "검증용 도착점",
            "product_id": "STAGING-P-A", "product_name": "검증용 상품",
            "route_type": "DIRECT", "dc_id": None,
            "planned_qty": planned_qty,
            "planned_cost": 12.5, "planned_expected_saving": 50.75, "planned_net_benefit": 38.25,
            "vhs_score": 81.25, "robustness_status": "안정", "confidence_score": 72.5,
        }],
    }


class _Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def run(self, name: str, check: Callable[[], str]) -> bool:
        try:
            detail = check()
            self.results.append((name, True, detail))
            return True
        except AssertionError as error:
            self.results.append((name, False, str(error)))
        except HistoryStoreError as error:
            self.results.append((name, False, str(error)))
        except Exception as error:  # noqa: BLE001 - surfaced as a failed check
            self.results.append((name, False, f"{type(error).__name__}"))
        return False

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.results if not ok)

    def print(self) -> None:
        for name, ok, detail in self.results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}{f' - {detail}' if detail else ''}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_checks(report: _Report, created: list[str], *, read_only: bool) -> None:
    store = build_execution_history_store()

    def check_connect() -> str:
        health = execution_history_health()
        _assert(bool(health.get("connection_ok")), "연결 실패")
        return f"latency {health.get('latency_ms')} ms"

    def check_schema_initialize() -> str:
        store.initialize()
        first = inspect_execution_history_schema()
        # Second initialization must be a no-op, not a reset.
        store.initialize()
        second = inspect_execution_history_schema()
        _assert(bool(first.get("ok")), "; ".join(first.get("issues") or ["스키마 불일치"]))
        _assert(first.get("schema_version") == second.get("schema_version"), "재초기화로 버전 변경")
        return f"schema version {first.get('schema_version')}, 재초기화 idempotent"

    def check_schema_structure() -> str:
        schema = inspect_execution_history_schema()
        _assert(bool(schema.get("ok")), "; ".join(schema.get("issues") or ["스키마 불일치"]))
        indexes = schema.get("indexes") or {}
        _assert(all(indexes.values()), f"인덱스 누락: {sorted(k for k, v in indexes.items() if not v)}")
        return f"테이블 {len(schema.get('tables') or {})}개, 인덱스 {len(indexes)}개"

    report.run("connect", check_connect)
    report.run("schema initialize (idempotent)", check_schema_initialize)
    report.run("schema structure (PK/FK/index/version)", check_schema_structure)
    if read_only:
        return

    plan_id = f"{CHECK_NAMESPACE}-{uuid.uuid4().hex[:12]}"
    second_id = f"{CHECK_NAMESPACE}-{uuid.uuid4().hex[:12]}"
    race_id = f"{CHECK_NAMESPACE}-{uuid.uuid4().hex[:12]}"
    created.extend([plan_id, second_id, race_id])

    def check_write() -> str:
        result = record_execution_plan(staging_plan(plan_id))
        _assert(bool(result.get("ok")) and result.get("created"), result.get("message", ""))
        record_execution_plan(staging_plan(second_id))
        return "계획 2건 저장"

    def check_read() -> str:
        loaded = get_recorded_plan(plan_id)
        _assert(bool(loaded.get("ok")), loaded.get("message", ""))
        _assert(loaded["items"][0]["planned_qty"] == 5, "planned_qty 불일치")
        _assert(loaded["items"][0]["actual_qty"] is None, "미입력 실제값이 NULL이 아님")
        _assert(loaded["plan"]["expected_total_cost"] == 12.5, "금액 왕복 불일치")
        return "저장값과 조회값 일치"

    def check_update_and_audit() -> str:
        updated = update_execution_item(
            plan_id, "STAGING-C-001", "일부 실행", 3,
            nonexecution_reason="현장 판단",
            outcomes={"actual_transport_cost": 11.0, "actual_saving": 44.0},
        )
        _assert(bool(updated.get("ok")), updated.get("message", ""))
        loaded = get_recorded_plan(plan_id)
        item = loaded["items"][0]
        _assert(item["execution_status"] == "partial", "실행 상태 불일치")
        _assert(item["actual_qty"] == 3, "실제 수량 불일치")
        _assert(item["actual_net_benefit"] == 33.0, "실제 순효과 불일치")
        events = list_item_events(plan_id, "STAGING-C-001")
        _assert(len(events) == 1, f"감사 기록 {len(events)}건")
        return "상태·수량·사후결과·감사 1건"

    def check_duplicate() -> str:
        duplicate = record_execution_plan(staging_plan(plan_id))
        _assert(duplicate.get("code") == "duplicate", f"중복 처리 코드 {duplicate.get('code')}")
        _assert(duplicate.get("created") is False, "중복 저장이 신규로 처리됨")
        loaded = get_recorded_plan(plan_id)
        _assert(loaded["items"][0]["actual_qty"] == 3, "중복 시도가 기존 값을 덮어씀")
        return "기존 값 보존"

    def check_transaction_rollback() -> str:
        rollback_id = f"{CHECK_NAMESPACE}-{uuid.uuid4().hex[:12]}"
        created.append(rollback_id)
        snapshot = {
            "plan_id": rollback_id, "algorithm_version": "staging-check",
            "candidate_algorithm_version": "staging-check",
            "data_signature": "staging-check-signature",
            "created_at": "2026-01-01T00:00:00+00:00",
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "plan_status": "추천 가능", "total_actions": 1, "total_planned_qty": 1,
            "expected_total_cost": 1.0, "expected_total_saving": 1.0,
            "expected_total_net_benefit": 0.0,
        }
        # planned_qty must violate the CHECK so the server aborts mid-transaction.
        bad_item = {
            "plan_id": rollback_id, "candidate_id": "STAGING-BAD",
            "source_store_id": "S", "destination_store_id": "D", "product_id": "P",
            "route_type": "DIRECT", "planned_qty": 0, "feature_snapshot_json": "{}",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        try:
            store.save_plan(snapshot, [bad_item])
        except HistoryStoreError:
            pass
        else:
            raise AssertionError("제약 위반이 저장 성공으로 처리됨")
        loaded = get_recorded_plan(rollback_id)
        _assert(loaded.get("code") == "not_found", "계획 일부가 남음")
        return "제약 위반 시 계획·항목 모두 rollback"

    def check_concurrent_duplicate() -> str:
        payload = staging_plan(race_id)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: record_execution_plan(payload), range(2)))
        codes = sorted(str(outcome.get("code")) for outcome in outcomes)
        _assert(codes == ["duplicate", "recorded"], f"동시 저장 결과 {codes}")
        loaded = get_recorded_plan(race_id)
        _assert(len(loaded.get("items") or []) == 1, "중복 항목 생성")
        return "동시 저장 1건만 생성"

    def check_concurrent_update() -> str:
        def update(quantity: int) -> dict[str, Any]:
            return update_execution_item(race_id, "STAGING-C-001", "실행", quantity)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(update, (4, 6)))
        _assert(all(outcome.get("ok") for outcome in outcomes), "동시 수정 실패")
        events = list_item_events(race_id, "STAGING-C-001")
        _assert(len(events) == 2, f"감사 기록 {len(events)}건 (기대 2건)")
        # A lost update would leave the second event unaware of the first value.
        _assert(
            events[1]["previous_actual_qty"] == events[0]["new_actual_qty"],
            "두 번째 수정이 첫 번째 결과를 보지 못함 (lost update)",
        )
        final = get_recorded_plan(race_id)["items"][0]["actual_qty"]
        _assert(final == events[1]["new_actual_qty"], "최종 값과 감사 기록 불일치")
        return f"직렬화된 수정, 최종 {final}"

    def check_pagination() -> str:
        first = list_recorded_plans(limit=1, offset=0)
        second = list_recorded_plans(limit=1, offset=1)
        _assert(bool(first.get("ok")) and bool(second.get("ok")), "목록 조회 실패")
        _assert(len(first["plans"]) == 1 and len(second["plans"]) == 1, "페이지 크기 불일치")
        _assert(bool(first.get("has_more")), "다음 페이지 판정 실패")
        _assert(first["plans"][0]["plan_id"] != second["plans"][0]["plan_id"], "페이지 중복")
        return "limit/offset 페이지 분리"

    def check_reconnect() -> str:
        # Drop every cached connection object: the next call reconnects exactly
        # as a restarted Streamlit process would.
        execution_history_store._cached_store.cache_clear()
        fresh = build_execution_history_store()
        _assert(fresh is not store, "저장소 인스턴스가 재생성되지 않음")
        plan, items = fresh.get_plan(plan_id)
        _assert(plan is not None, "재연결 후 계획 없음")
        _assert(items[0]["actual_qty"] == 3, "재연결 후 실제 수량 불일치")
        _assert(items[0]["execution_status"] == "partial", "재연결 후 상태 불일치")
        return "새 인스턴스에서 동일 이력 조회"

    def check_export() -> str:
        exported = export_execution_history_csv()
        _assert(bool(exported.get("ok")), exported.get("message", ""))
        text = exported["data"].decode("utf-8-sig")
        _assert(plan_id in text, "내보내기에 검증 기록 없음")
        return f"{exported.get('row_count')}행 내보내기"

    report.run("write (plan + items)", check_write)
    report.run("read back", check_read)
    report.run("update + audit event", check_update_and_audit)
    report.run("duplicate plan is safe", check_duplicate)
    report.run("transaction rollback", check_transaction_rollback)
    report.run("concurrent duplicate save", check_concurrent_duplicate)
    report.run("concurrent item update (lost update)", check_concurrent_update)
    report.run("pagination", check_pagination)
    report.run("reconnect after restart", check_reconnect)
    report.run("calibration export", check_export)


def _cleanup(created: list[str]) -> str:
    if not created:
        return "정리할 검증 기록 없음"
    unsafe = [plan_id for plan_id in created if not plan_id.startswith(CHECK_NAMESPACE)]
    if unsafe:
        return "검증 namespace 밖의 식별자가 있어 정리를 중단했습니다."
    try:
        removed = build_execution_history_store().delete_plans(created)
    except HistoryStoreError:
        return "검증 기록 정리에 실패했습니다. 남은 기록은 " + CHECK_NAMESPACE + " 접두사로 확인하세요."
    return (
        f"검증 기록 정리 완료: 계획 {removed['plans']}건 / 항목 {removed['items']}건 / "
        f"감사 {removed['events']}건"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "staging PostgreSQL에 대해 실행 이력 저장소를 통합 검증합니다. "
            f"{HISTORY_TEST_DATABASE_URL_ENV} 이 설정된 경우에만 동작합니다."
        ),
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="연결과 스키마만 확인하고 쓰기 검증은 건너뜁니다.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_staging_test_config()
    except HistoryConfigurationError as error:
        print(f"검증용 데이터베이스 설정 오류: {error}")
        return 2
    if config is None:
        print("PostgreSQL staging URL not configured.")
        print("실제 PostgreSQL 통합 검증은 수행하지 않았습니다. (미검증)")
        return 3

    # Route the whole service stack at staging for this process only, and make a
    # SQLite fallback impossible while the checks run.
    previous = {
        HISTORY_DATABASE_URL_ENV: os.environ.get(HISTORY_DATABASE_URL_ENV),
        HISTORY_DB_PATH_ENV: os.environ.get(HISTORY_DB_PATH_ENV),
    }
    os.environ[HISTORY_DATABASE_URL_ENV] = str(config.database_url)
    os.environ.pop(HISTORY_DB_PATH_ENV, None)
    execution_history_store._cached_store.cache_clear()

    report = _Report()
    created: list[str] = []
    try:
        _run_checks(report, created, read_only=bool(args.read_only))
    finally:
        cleanup_message = _cleanup(created)
        execution_history_store._cached_store.cache_clear()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    report.print()
    print(cleanup_message)
    passed = len(report.results) - report.failed
    print(f"검증 {passed}/{len(report.results)} 통과")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
