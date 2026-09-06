"""Backend selection for persistent execution history.

This module deliberately has no Streamlit dependency.  Streamlit Cloud root
secrets are exposed as environment variables, while local/server deployments
can set the same variables directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qsl, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DB = PROJECT_ROOT / "runtime_data" / "varo_execution_history.sqlite3"
HISTORY_DATABASE_URL_ENV = "VARO_HISTORY_DATABASE_URL"
HISTORY_DB_PATH_ENV = "VARO_HISTORY_DB_PATH"
HISTORY_TEST_DATABASE_URL_ENV = "VARO_HISTORY_TEST_DATABASE_URL"
HISTORY_RESTORE_DATABASE_URL_ENV = "VARO_HISTORY_RESTORE_DATABASE_URL"
HISTORY_CONNECT_TIMEOUT_ENV = "VARO_HISTORY_DB_CONNECT_TIMEOUT"
HISTORY_CONNECT_RETRIES_ENV = "VARO_HISTORY_DB_CONNECT_RETRIES"

DEFAULT_CONNECT_TIMEOUT = 8
MIN_CONNECT_TIMEOUT = 2
MAX_CONNECT_TIMEOUT = 60
DEFAULT_CONNECT_RETRIES = 1
MAX_CONNECT_RETRIES = 3

# Every user-visible or logged reference to a server URL uses this constant.
REDACTED_DATABASE_URL = "postgresql://<가려짐>"
# libpq modes that leave the transport unencrypted or unauthenticated.
UNENCRYPTED_SSL_MODES = ("disable",)
UNVERIFIED_SSL_MODES = ("allow", "prefer", "require")


class HistoryConfigurationError(ValueError):
    """Raised for an explicitly configured but unsupported backend."""


@dataclass(frozen=True)
class ExecutionHistoryConfig:
    backend: str
    sqlite_path: Path | None = None
    database_url: str | None = field(default=None, repr=False)
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT
    connect_retries: int = DEFAULT_CONNECT_RETRIES
    purpose: str = "primary"

    @property
    def user_label(self) -> str:
        return "서버" if self.backend == "postgresql" else "로컬"


def redact_database_url(database_url: str | None) -> str:
    """Never return host, user, password, port, or database name."""
    return REDACTED_DATABASE_URL if str(database_url or "").strip() else ""


def _validated_postgres_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or not parsed.path.strip("/"):
        raise HistoryConfigurationError("운영 데이터베이스 설정을 확인해주세요.")
    return database_url


def _bounded_int(
    values: Mapping[str, str], key: str, default: int, minimum: int, maximum: int,
) -> int:
    """A typo must not break startup; fall back to the safe default."""
    raw = str(values.get(key) or "").strip()
    if not raw:
        return default
    try:
        number = int(raw)
    except ValueError:
        return default
    return max(minimum, min(number, maximum))


def database_url_transport(database_url: str | None) -> dict[str, object]:
    """Describe only the TLS posture of a URL, never its identity."""
    query = dict(parse_qsl(urlsplit(str(database_url or "")).query, keep_blank_values=True))
    sslmode = (query.get("sslmode") or "").strip().lower() or None
    return {
        "sslmode": sslmode,
        "explicit": sslmode is not None,
        "encrypted": sslmode not in UNENCRYPTED_SSL_MODES if sslmode else None,
        "verified": bool(sslmode) and sslmode not in UNENCRYPTED_SSL_MODES + UNVERIFIED_SSL_MODES,
        "sslrootcert_configured": bool((query.get("sslrootcert") or "").strip()),
    }


def load_execution_history_config(
    db_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ExecutionHistoryConfig:
    """Select exactly one backend without ever falling back silently.

    An explicit ``db_path`` is the compatibility/test override and always
    selects SQLite.  Otherwise the PostgreSQL URL wins over the optional local
    path.  Unsupported or incomplete URLs fail closed.
    """
    if db_path is not None:
        return ExecutionHistoryConfig(backend="sqlite", sqlite_path=Path(db_path))

    values = os.environ if environ is None else environ
    database_url = str(values.get(HISTORY_DATABASE_URL_ENV) or "").strip()
    if database_url:
        return ExecutionHistoryConfig(
            backend="postgresql",
            database_url=_validated_postgres_url(database_url),
            connect_timeout=_bounded_int(
                values, HISTORY_CONNECT_TIMEOUT_ENV,
                DEFAULT_CONNECT_TIMEOUT, MIN_CONNECT_TIMEOUT, MAX_CONNECT_TIMEOUT,
            ),
            connect_retries=_bounded_int(
                values, HISTORY_CONNECT_RETRIES_ENV, DEFAULT_CONNECT_RETRIES, 0, MAX_CONNECT_RETRIES,
            ),
        )

    configured_path = str(values.get(HISTORY_DB_PATH_ENV) or "").strip()
    return ExecutionHistoryConfig(
        backend="sqlite",
        sqlite_path=Path(configured_path) if configured_path else DEFAULT_HISTORY_DB,
    )


def _load_alternate_config(
    env_name: str,
    purpose: str,
    conflict_message: str,
    environ: Mapping[str, str] | None,
) -> ExecutionHistoryConfig | None:
    """Load a non-deployment PostgreSQL target that must differ from production."""
    values = os.environ if environ is None else environ
    database_url = str(values.get(env_name) or "").strip()
    if not database_url:
        return None
    production_url = str(values.get(HISTORY_DATABASE_URL_ENV) or "").strip()
    if production_url and production_url == database_url:
        raise HistoryConfigurationError(conflict_message)
    return ExecutionHistoryConfig(
        backend="postgresql",
        database_url=_validated_postgres_url(database_url),
        connect_timeout=_bounded_int(
            values, HISTORY_CONNECT_TIMEOUT_ENV,
            DEFAULT_CONNECT_TIMEOUT, MIN_CONNECT_TIMEOUT, MAX_CONNECT_TIMEOUT,
        ),
        connect_retries=_bounded_int(
            values, HISTORY_CONNECT_RETRIES_ENV, DEFAULT_CONNECT_RETRIES, 0, MAX_CONNECT_RETRIES,
        ),
        purpose=purpose,
    )


def load_staging_test_config(
    *, environ: Mapping[str, str] | None = None,
) -> ExecutionHistoryConfig | None:
    """Return the opt-in staging/test PostgreSQL target, or ``None``.

    Integration checks that write rows must never reach the deployment URL, so
    the staging target lives in its own variable and is rejected outright when
    it points at the same string as the configured production backend.
    """
    return _load_alternate_config(
        HISTORY_TEST_DATABASE_URL_ENV,
        "staging_test",
        "검증용 데이터베이스와 운영 데이터베이스가 동일합니다. 별도 staging DB를 사용해주세요.",
        environ,
    )


def load_restore_target_config(
    *, environ: Mapping[str, str] | None = None,
) -> ExecutionHistoryConfig | None:
    """Return the explicit restore destination, or ``None``.

    Restore never reads the deployment variable, so an operator cannot overwrite
    production by forgetting a flag.
    """
    return _load_alternate_config(
        HISTORY_RESTORE_DATABASE_URL_ENV,
        "restore_target",
        "복원 대상과 운영 데이터베이스가 동일합니다. 복원 전용 대상을 지정해주세요.",
        environ,
    )


def database_name(database_url: str | None) -> str:
    """The database component only; used for an explicit operator confirmation."""
    return urlsplit(str(database_url or "")).path.strip("/")
