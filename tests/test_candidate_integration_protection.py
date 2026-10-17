"""Integration boundaries: opt-in code and isolated candidate output."""
import os
from pathlib import Path
import subprocess
import sys

from services.git_protection import is_protected


def test_research_entrypoint_is_protected_code():
    assert is_protected("app_candidate.py")
    assert not is_protected("app_candidate.py.notes")


def test_candidate_history_respects_explicit_output_root(tmp_path):
    isolated = tmp_path / "isolated-runtime"
    env = dict(os.environ, VARO_OUTPUT_ROOT=str(isolated), PYTHONDONTWRITEBYTECODE="1")
    code = "from pages.candidate_review import HISTORY_DIR; print(HISTORY_DIR)"
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=True)
    assert Path(result.stdout.strip()) == isolated / "third_upgrade_20261017" / "candidate_history"
    assert not isolated.exists()  # Import alone never creates a history/DB.
