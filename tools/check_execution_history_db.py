"""Operator check for the execution-history database.

Prints backend, connectivity, TLS posture, schema version and structure.  It
never prints a connection URL, host, user, password, or database name.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.execution_history_config import (  # noqa: E402
    HISTORY_TEST_DATABASE_URL_ENV,
    HistoryConfigurationError,
    database_url_transport,
    load_execution_history_config,
    load_staging_test_config,
)
from services.execution_history_store import build_execution_history_store  # noqa: E402

BACKEND_LABELS = {"postgresql": "PostgreSQL", "sqlite": "SQLite"}


def transport_line(database_url: str | None) -> str:
    """Describe TLS posture without revealing where the server is."""
    transport = database_url_transport(database_url)
    mode = transport["sslmode"]
    if not transport["explicit"]:
        return "Transport: sslmode 미지정 (libpq 기본값 사용, 운영에서는 명시 권장)"
    if not transport["encrypted"]:
        return f"Transport: 경고 - 암호화되지 않음 (sslmode={mode})"
    if not transport["verified"]:
        return f"Transport: 암호화됨, 인증서 미검증 (sslmode={mode})"
    return f"Transport: 암호화 및 인증서 검증 (sslmode={mode})"


def _health(store: Any) -> dict[str, Any]:
    try:
        return store.health_check()
    except Exception:
        return {
            "connection_ok": False, "latency_ms": None,
            "message": "실행 기록 저장소에 연결하지 못했습니다.",
        }


def _schema(store: Any) -> dict[str, Any]:
    try:
        return store.inspect_schema()
    except Exception:
        return {"ok": False, "schema_version": None, "issues": ["스키마를 읽지 못했습니다."]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="실행 이력 데이터베이스의 backend, 연결, 스키마 상태를 점검합니다.",
    )
    parser.add_argument(
        "--target", choices=("configured", "staging"), default="configured",
        help=f"configured=배포 설정, staging={HISTORY_TEST_DATABASE_URL_ENV}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.target == "staging":
            config = load_staging_test_config()
            if config is None:
                print("PostgreSQL staging URL not configured.")
                return 3
        else:
            config = load_execution_history_config()
    except HistoryConfigurationError as error:
        print("Backend: 설정 오류")
        print(f"Connection: FAILED ({error})")
        return 2

    print(f"Backend: {BACKEND_LABELS.get(config.backend, config.backend)}")
    if config.backend == "postgresql":
        print(transport_line(config.database_url))
        print(f"Connect timeout: {config.connect_timeout}s (연결 재시도 {config.connect_retries}회)")

    store = build_execution_history_store(config=config)
    health = _health(store)
    print(f"Connection: {'OK' if health.get('connection_ok') else 'FAILED'}")
    if health.get("latency_ms") is not None:
        print(f"Latency: {health['latency_ms']} ms")
    if not health.get("connection_ok"):
        print(health.get("message") or "실행 기록 저장소에 연결하지 못했습니다.")
        return 2

    schema = _schema(store)
    version = schema.get("schema_version")
    tables = schema.get("tables") or {}
    print(f"Schema version: {version if version is not None else '확인 불가'}")
    if tables and not any(table.get("present") for table in tables.values()):
        print("Schema: NOT INITIALIZED (앱을 한 번 실행하거나 이관 도구를 실행하면 생성됩니다)")
        return 1
    print(f"Schema: {'OK' if schema.get('ok') else 'MISMATCH'}")
    for issue in schema.get("issues") or ():
        print(f"  - {issue}")
    return 0 if schema.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
