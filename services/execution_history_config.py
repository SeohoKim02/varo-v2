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
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DB = PROJECT_ROOT / "runtime_data" / "varo_execution_history.sqlite3"
HISTORY_DATABASE_URL_ENV = "VARO_HISTORY_DATABASE_URL"
HISTORY_DB_PATH_ENV = "VARO_HISTORY_DB_PATH"


class HistoryConfigurationError(ValueError):
    """Raised for an explicitly configured but unsupported backend."""


@dataclass(frozen=True)
class ExecutionHistoryConfig:
    backend: str
    sqlite_path: Path | None = None
    database_url: str | None = field(default=None, repr=False)

    @property
    def user_label(self) -> str:
        return "서버" if self.backend == "postgresql" else "로컬"


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
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or not parsed.path.strip("/"):
            raise HistoryConfigurationError("운영 데이터베이스 설정을 확인해주세요.")
        return ExecutionHistoryConfig(backend="postgresql", database_url=database_url)

    configured_path = str(values.get(HISTORY_DB_PATH_ENV) or "").strip()
    return ExecutionHistoryConfig(
        backend="sqlite",
        sqlite_path=Path(configured_path) if configured_path else DEFAULT_HISTORY_DB,
    )

