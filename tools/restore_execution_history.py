"""Manual, operator-only restore of an execution-history backup.

Safety rules this tool enforces:

* the destination comes only from ``VARO_HISTORY_RESTORE_DATABASE_URL``; the
  deployment variable ``VARO_HISTORY_DATABASE_URL`` is never used as a target,
* ``--confirm-database`` must match the destination database name,
* ``--dry-run`` or ``--apply`` must be stated explicitly,
* existing plans are never overwritten or deleted; only unseen plan ids are
  inserted, inside one transaction.

There is deliberately no restore action anywhere in the app UI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.execution_history_config import (  # noqa: E402
    HISTORY_RESTORE_DATABASE_URL_ENV,
    HistoryConfigurationError,
    database_name,
    load_restore_target_config,
)
from services.execution_history_migration import validate_history_snapshot  # noqa: E402
from services.execution_history_store import (  # noqa: E402
    SCHEMA_VERSION,
    HistoryStoreError,
    build_execution_history_store,
)

BACKUP_FORMAT = "varo-execution-history-backup"


def read_backup(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != BACKUP_FORMAT:
        raise ValueError("백업 파일 형식을 확인해주세요.")
    if int(document.get("schema_version") or 0) != SCHEMA_VERSION:
        raise ValueError("백업 파일의 스키마 버전이 현재 버전과 다릅니다.")
    return {
        "plans": list(document.get("plans") or ()),
        "items": list(document.get("items") or ()),
        "events": list(document.get("events") or ()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "백업 JSON을 복원 전용 대상에 복원합니다. 대상은 "
            f"{HISTORY_RESTORE_DATABASE_URL_ENV} 로만 지정합니다."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="복원할 백업 파일 경로")
    parser.add_argument(
        "--confirm-database", required=True,
        help="복원 대상 데이터베이스 이름. 대상 URL의 이름과 정확히 일치해야 합니다.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="쓰기 없이 건수와 중복만 확인")
    mode.add_argument("--apply", action="store_true", help="신규 계획만 transaction으로 삽입")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.input.is_file():
        print("백업 파일을 찾지 못했습니다.")
        return 2
    try:
        config = load_restore_target_config()
    except HistoryConfigurationError as error:
        print(str(error))
        return 2
    if config is None:
        print(f"복원 대상이 설정되지 않았습니다. {HISTORY_RESTORE_DATABASE_URL_ENV} 를 지정해주세요.")
        return 2
    if database_name(config.database_url) != str(args.confirm_database).strip():
        print("복원 대상 확인에 실패했습니다. --confirm-database 값이 대상과 일치하지 않습니다.")
        return 2

    try:
        snapshot = read_backup(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error) if isinstance(error, ValueError) else "백업 파일을 읽지 못했습니다.")
        return 2

    validation = validate_history_snapshot(snapshot)
    print(
        "백업 내용: 계획 {plan_count}건 / 항목 {item_count}건 / 감사 {audit_count}건 / "
        "무효 {invalid_record_count}건".format(**validation)
    )
    if not validation["valid"]:
        print("백업 무결성 검증에 실패해 복원하지 않았습니다.")
        for issue in validation["issues"]:
            print(f"  - {issue}")
        return 1

    try:
        destination = build_execution_history_store(config=config)
        existing = destination.existing_plan_ids(initialize=not args.dry_run)
        source_ids = {str(row["plan_id"]) for row in snapshot["plans"]}
        duplicates = source_ids & existing
        print(f"대상 중복 계획 {len(duplicates)}건 / 신규 계획 {len(source_ids - duplicates)}건")
        if args.dry_run:
            print("확인만 수행했습니다. 대상 데이터베이스는 변경하지 않았습니다.")
            return 0
        inserted = destination.import_snapshot(snapshot, skip_plan_ids=duplicates)
    except HistoryConfigurationError:
        print("복원 대상 설정을 확인해주세요.")
        return 2
    except (HistoryStoreError, OSError):
        print("복원에 실패했습니다. 대상 데이터베이스는 변경되지 않았습니다.")
        return 1

    print(
        f"복원 완료: 계획 {inserted['plans']}건 / 항목 {inserted['items']}건 / "
        f"감사 {inserted['events']}건 (기존 기록은 덮어쓰지 않았습니다)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
