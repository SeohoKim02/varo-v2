"""Read-only logical backup of the execution history.

This complements — it does not replace — the managed provider's PostgreSQL
backup (`pg_dump` / provider snapshots), which is the operational source of
truth for point-in-time recovery.  This tool produces a portable JSON snapshot
that `tools/restore_execution_history.py` can verify and re-import.

It only reads.  It never prints a URL, host, user, password, or database name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.execution_history_config import (  # noqa: E402
    HistoryConfigurationError,
    load_execution_history_config,
    load_staging_test_config,
)
from services.execution_history_migration import validate_history_snapshot  # noqa: E402
from services.execution_history_store import (  # noqa: E402
    SCHEMA_VERSION,
    HistoryStoreError,
    build_execution_history_store,
)

BACKUP_FORMAT = "varo-execution-history-backup"


def build_backup_document(snapshot: dict) -> dict:
    return {
        "format": BACKUP_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plans": list(snapshot.get("plans") or ()),
        "items": list(snapshot.get("items") or ()),
        "events": list(snapshot.get("events") or ()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="설정된 실행 이력 저장소를 읽기 전용 JSON 스냅샷으로 백업합니다.",
    )
    parser.add_argument("--output", type=Path, required=True, help="저장할 백업 파일 경로")
    parser.add_argument(
        "--source", choices=("configured", "staging"), default="configured",
        help="configured=배포 설정, staging=VARO_HISTORY_TEST_DATABASE_URL",
    )
    parser.add_argument("--overwrite", action="store_true", help="기존 백업 파일 덮어쓰기 허용")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        print("백업 파일이 이미 있습니다. --overwrite 를 명시해주세요.")
        return 2
    try:
        if args.source == "staging":
            config = load_staging_test_config()
            if config is None:
                print("PostgreSQL staging URL not configured.")
                return 3
        else:
            config = load_execution_history_config()
        snapshot = build_execution_history_store(config=config).read_snapshot()
    except HistoryConfigurationError:
        print("실행 기록 저장 설정을 확인해주세요.")
        return 2
    except (HistoryStoreError, OSError):
        print("실행 기록을 읽지 못했습니다.")
        return 2

    validation = validate_history_snapshot(snapshot)
    document = build_backup_document(snapshot)
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)

    print(f"Backend: {'PostgreSQL' if config.backend == 'postgresql' else 'SQLite'}")
    print(
        "백업 완료: 계획 {plan_count}건 / 항목 {item_count}건 / 감사 {audit_count}건".format(**validation)
    )
    print(f"무결성 검증: {'OK' if validation['valid'] else 'FAILED'}")
    for issue in validation["issues"]:
        print(f"  - {issue}")
    print(f"SHA-256: {hashlib.sha256(payload).hexdigest()}")
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
