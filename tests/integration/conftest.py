"""Real-PostgreSQL integration tests.

Nothing in this directory is collected unless ``VARO_HISTORY_TEST_DATABASE_URL``
names a staging database, so the default ``python -m pytest -q`` run needs no
server, adds no skips, and can never touch a deployment database.
"""
from __future__ import annotations

import os

import pytest

from services.execution_history_config import (
    HISTORY_DATABASE_URL_ENV,
    HISTORY_DB_PATH_ENV,
    HISTORY_TEST_DATABASE_URL_ENV,
    load_staging_test_config,
)


def staging_url() -> str:
    return str(os.environ.get(HISTORY_TEST_DATABASE_URL_ENV) or "").strip()


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001 - pytest hook
    return not staging_url()


@pytest.fixture(autouse=True)
def route_history_at_staging(monkeypatch):
    """Point the whole service stack at staging and forbid a SQLite fallback."""
    from services import execution_history_store

    config = load_staging_test_config()
    assert config is not None, "staging URL이 필요합니다."
    monkeypatch.setenv(HISTORY_DATABASE_URL_ENV, str(config.database_url))
    monkeypatch.delenv(HISTORY_DB_PATH_ENV, raising=False)
    execution_history_store._cached_store.cache_clear()
    yield
    execution_history_store._cached_store.cache_clear()
