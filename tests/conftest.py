"""Global safety boundary: tests never touch configured production history."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_execution_history_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("VARO_HISTORY_DATABASE_URL", "")
    monkeypatch.setenv("VARO_HISTORY_DB_PATH", str(tmp_path / "isolated_execution_history.sqlite3"))

