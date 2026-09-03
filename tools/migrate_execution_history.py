"""Explicit SQLite -> PostgreSQL execution-history migration command."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.execution_history import history_db_path  # noqa: E402
from services.execution_history_config import (  # noqa: E402
    HistoryConfigurationError,
    load_execution_history_config,
)
from services.execution_history_migration import migrate_sqlite_history  # noqa: E402
from services.execution_history_store import build_execution_history_store  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="로컬 SQLite 실행 이력을 설정된 PostgreSQL 저장소로 단방향 이관합니다.",
    )
    parser.add_argument("--source", type=Path, default=None, help="원본 SQLite 파일 경로")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="쓰기 없이 건수·중복·유효성을 검사")
    mode.add_argument("--apply", action="store_true", help="검증 후 신규 계획을 transaction으로 이관")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source or history_db_path()
    if not source.is_file():
        print("원본 SQLite 실행 기록 파일을 찾지 못했습니다.")
        return 2
    try:
        config = load_execution_history_config()
        if config.backend != "postgresql":
            print("서버 실행 기록 저장 설정이 필요합니다.")
            return 2
        destination = build_execution_history_store(config=config)
    except HistoryConfigurationError:
        print("서버 실행 기록 저장 설정을 확인해주세요.")
        return 2

    result = migrate_sqlite_history(source, destination, dry_run=bool(args.dry_run))
    print(result["message"])
    print(
        "계획 {plan_count}건 / 항목 {item_count}건 / 감사 {audit_count}건 / "
        "중복 계획 {duplicate_plan_count}건 / 유효 {valid_record_count}건 / "
        "무효 {invalid_record_count}건".format(**result)
    )
    if result.get("ok") and result.get("code") == "migrated":
        print(
            "이관 완료: 계획 {inserted_plan_count}건 / 항목 {inserted_item_count}건 / "
            "감사 {inserted_audit_count}건".format(**result)
        )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
