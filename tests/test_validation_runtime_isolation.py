"""Output isolation and observable test-runner failures, never model behavior."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from services import release_validation as rv
from services.dqn_samples import save_balanced_recommendations
from services.local_paths import PROJECT_ROOT, balanced_samples_dir


def test_balanced_output_uses_default_when_not_configured(monkeypatch):
    monkeypatch.delenv("VARO_OUTPUT_ROOT", raising=False)
    assert balanced_samples_dir() == PROJECT_ROOT / "outputs" / "dqn_balanced_samples"


def test_balanced_writer_resolves_environment_after_import_and_preserves_rows(tmp_path, monkeypatch):
    rows = [{"route_id": "test-route", "recommended_qty": 3, "estimated_cost": 7, "target_action": "이동"}]
    before = [dict(row) for row in rows]
    for name in ("first", "second"):
        root = tmp_path / name
        monkeypatch.setenv("VARO_OUTPUT_ROOT", str(root))
        path = save_balanced_recommendations(rows, "isolation", 2, 1, derived_from="read-only.xlsx")
        assert path.parent == root / "dqn_balanced_samples"
        assert pd.read_excel(path, sheet_name="recommendations").to_dict("records") == before
        metadata = pd.read_excel(path, sheet_name="metadata").set_index("key")["value"].to_dict()
        assert metadata["derived_from"] == "read-only.xlsx"
        assert rows == before
    assert len(list(tmp_path.rglob("*.xlsx"))) == 2


@pytest.mark.parametrize("configured", [None, "", "  ", "outputs", "outputs/dqn", "outputs/dqn_balanced_samples", "outputs/nested/other"])
def test_pytest_redirects_all_production_descendants_and_restores_environment(tmp_path, configured):
    repo = Path(__file__).resolve().parents[1]
    original = str(repo / configured) if configured and configured.startswith("outputs") else configured
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if original is None:
        env.pop("VARO_OUTPUT_ROOT", None)
    else:
        env["VARO_OUTPUT_ROOT"] = original
    code = """
import importlib.util,json,os
from pathlib import Path
spec=importlib.util.spec_from_file_location('isolation_config',Path('tests/conftest.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
before=os.environ.get('VARO_OUTPUT_ROOT')
m.pytest_configure(None)
root=Path(os.environ['VARO_OUTPUT_ROOT'])
assert not root.resolve().is_relative_to((Path.cwd()/'outputs').resolve())
assert root.is_dir()
m.pytest_unconfigure(None)
assert os.environ.get('VARO_OUTPUT_ROOT') == before
assert not root.exists()
print(json.dumps({'isolated':True,'restored':True}))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(result.stdout) == {"isolated": True, "restored": True}


def test_pytest_reuses_explicit_external_scratch_without_removing_it(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    scratch = tmp_path / "managed"
    scratch.mkdir()
    sentinel = scratch / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    env = dict(os.environ, VARO_OUTPUT_ROOT=str(scratch), PYTHONDONTWRITEBYTECODE="1")
    code = """
import importlib.util,os
from pathlib import Path
spec=importlib.util.spec_from_file_location('isolation_config',Path('tests/conftest.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
before=os.environ['VARO_OUTPUT_ROOT'];m.pytest_configure(None);m.pytest_unconfigure(None)
assert os.environ['VARO_OUTPUT_ROOT']==before
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, check=True, timeout=30)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_release_timeout_is_failure_with_persistent_diagnostic_log(tmp_path, monkeypatch):
    audit = object.__new__(rv.Audit)
    audit.output, audit.mode = tmp_path, "full"
    def stall(args, **kwargs):
        assert "-vv" in args and "faulthandler_timeout=60" in args
        assert kwargs["timeout"] == 900
        kwargs["stdout"].write("tests/test_example.py::test_running\n")
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])
    monkeypatch.setattr(rv.subprocess, "run", stall)
    result = audit.tests(tmp_path / "scratch")
    assert result["status"] == "FAIL"
    assert result["reason_codes"] == ["PYTEST_TIMEOUT"]
    assert result["actual_result"]["timed_out"] is True
    assert result["actual_result"]["exit_code"] is None
    assert "test_running" in (tmp_path / "pytest.log").read_text()
    assert "PYTEST_TIMEOUT" in (tmp_path / "pytest.log").read_text()


def test_release_runner_keeps_normal_pass_contract(tmp_path, monkeypatch):
    audit = object.__new__(rv.Audit)
    audit.output, audit.mode = tmp_path, "full"
    def completed(args, **kwargs):
        kwargs["stdout"].write("2143 passed, 27 subtests passed in 1.00s\n")
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(rv.subprocess, "run", completed)
    result = audit.tests(tmp_path / "scratch")
    assert result["status"] == "PASS"
    assert result["actual_result"]["timed_out"] is False
    assert result["actual_result"]["passed"] == 2143
