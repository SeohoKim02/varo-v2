"""Real isolated Git fixtures. No commit or mutation in the Varo repository."""
from pathlib import Path
import subprocess

import pytest

from services import git_protection as gp, release_validation as rv
from services.release_regression import readiness


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def put(repo, name, text="changed\n"):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "independent repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Varo fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "core.autocrlf", "false")
    for name in ("pages/overview.py", "services/release_validation.py", "notes.txt"):
        put(root, name, "baseline\n")
    git(root, "add", "--all")
    git(root, "commit", "-m", "isolated test baseline")
    return root


# Expectations fixed before execution: all protected changes BLOCKED, never FAIL.
@pytest.mark.parametrize("case,expected", [
    ("clean", "PASS"), ("unstaged", "BLOCKED"), ("staged", "BLOCKED"),
    ("new", "BLOCKED"), ("delete", "BLOCKED"), ("rename_out", "BLOCKED"),
    ("rename_in", "BLOCKED"), ("nested", "BLOCKED"), ("unprotected", "PASS"),
    ("mixed", "BLOCKED"), ("spaces", "BLOCKED"), ("unicode", "BLOCKED"),
    ("several", "BLOCKED"), ("index_worktree_differ", "BLOCKED"), ("self", "BLOCKED"),
    ("simulation", "BLOCKED"), ("guard_new", "BLOCKED"),
])
def test_actual_git_states(repo, case, expected):
    if case in ("unstaged", "staged", "mixed", "several", "index_worktree_differ"):
        put(repo, "pages/overview.py")
    if case in ("staged", "index_worktree_differ"):
        git(repo, "add", "pages/overview.py")
    if case == "index_worktree_differ":
        put(repo, "pages/overview.py", "third version\n")
    if case in ("unprotected", "mixed", "several"):
        put(repo, "notes.txt")
    if case == "new": put(repo, "pages/new.py")
    if case == "delete": (repo / "pages/overview.py").unlink()
    if case == "rename_out": git(repo, "mv", "pages/overview.py", "renamed.txt")
    if case == "rename_in": git(repo, "mv", "notes.txt", "pages/renamed.py")
    if case == "nested": put(repo, "pages/a/b/c.py")
    if case == "spaces": put(repo, "pages/ leading name .py")
    if case == "unicode": put(repo, "pages/한글 경로.py")
    if case in ("self", "several"): put(repo, "services/release_validation.py")
    if case == "simulation": put(repo, "simulation/network_geometry.py")
    if case == "guard_new": put(repo, "services/git_protection.py")
    result = gp.inspect_git_protection(repo)
    assert result["status"] == expected
    evidence = result["actual_result"]
    assert evidence["git_dirty"] is (case != "clean")
    assert bool(evidence["protected_changes"]) is (expected == "BLOCKED")
    assert evidence["execution_version"]["git_protection.py"]
    if case == "index_worktree_differ":
        assert evidence["changes"][0]["index_status"] == evidence["changes"][0]["worktree_status"] == "M"
    if case == "rename_out":
        assert set(evidence["changed_paths"]) == {"pages/overview.py", "renamed.txt"}
        assert evidence["protected_changes"] == ["pages/overview.py"]
    if case == "unicode": assert evidence["protected_changes"] == ["pages/한글 경로.py"]
    if case == "spaces": assert evidence["protected_changes"] == ["pages/ leading name .py"]
    if case == "several":
        assert set(evidence["changed_paths"]) == {"notes.txt", "pages/overview.py", "services/release_validation.py"}
        assert set(evidence["protected_changes"]) == {"pages/overview.py", "services/release_validation.py"}


def test_reported_leading_space_bug_and_nested_cwd(repo):
    put(repo, "pages/overview.py")
    raw = git(repo, "status", "--porcelain=v1").decode()
    assert raw.startswith(" M pages/overview.py")
    assert raw.strip().splitlines()[0][3:] == "ages/overview.py"  # reproduce old defect
    a, b = gp.inspect_git_protection(repo), gp.inspect_git_protection(repo / "pages")
    assert a == b and a["status"] == "BLOCKED"
    assert a["actual_result"]["protected_changes"] == ["pages/overview.py"]


@pytest.mark.parametrize("stage", ["rev-parse", "status", "for-each-ref"])
def test_command_failure_never_passes(repo, monkeypatch, stage):
    original = gp.subprocess.run
    def fail(args, **kwargs):
        if stage in args:
            return subprocess.CompletedProcess(args, 128, stdout=b"misleading stdout", stderr=b"fixture error")
        return original(args, **kwargs)
    monkeypatch.setattr(gp.subprocess, "run", fail)
    result = gp.inspect_git_protection(repo)
    assert result["status"] == "ENVIRONMENT_ERROR"
    assert "fixture error" in result["reason_codes"][-1]
    assert "misleading stdout" not in result["reason_codes"][-1]


@pytest.mark.parametrize("error", [FileNotFoundError("git missing"), PermissionError("denied"), subprocess.TimeoutExpired("git", 30)])
def test_process_exception_never_passes(repo, monkeypatch, error):
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(gp.subprocess, "run", fail)
    assert gp.inspect_git_protection(repo)["status"] == "ENVIRONMENT_ERROR"


@pytest.mark.parametrize("raw", [b" M pages/a.py", b"bad\0", b"R  pages/new.py\0", b"ZZ x\0", b" M ../escape\0", b" M /absolute\0", b" M C:/absolute\0"])
def test_malformed_output_rejected(raw):
    with pytest.raises(ValueError): gp.parse_status_z(raw)


def test_copy_and_separator_and_directory_boundaries():
    rows = gp.parse_status_z(b"C  notes.txt\0pages/old.py\0?? pages\\new file.py\0")
    assert rows[0]["original_path"] == "pages/old.py"
    assert rows[1]["path"] == "pages/new file.py"
    assert gp.is_protected(rows[1]["path"])
    assert not gp.is_protected("pages_backup/a.py")
    assert not gp.is_protected("notes.txt")


def test_local_remote_difference_separate_from_dirty(repo):
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    git(repo, "remote", "add", "origin", str(repo.parent / "never-contacted"))
    git(repo, "config", "branch.main.remote", "origin")
    git(repo, "config", "branch.main.merge", "refs/heads/main")
    git(repo, "update-ref", "refs/remotes/origin/main", head)
    result = gp.inspect_git_protection(repo)
    assert result["status"] == "PASS"
    assert result["actual_result"]["upstream"]["status"] == "SYNCHRONIZED_LOCAL_REF"
    put(repo, "notes.txt")
    git(repo, "add", "notes.txt")
    git(repo, "commit", "-m", "fixture ahead only")
    result = gp.inspect_git_protection(repo)
    assert result["status"] == "PASS" and not result["actual_result"]["git_dirty"]
    assert result["actual_result"]["upstream"]["ahead"] == 1
    assert result["actual_result"]["remote_contacted"] is False


def test_t5_audit_uses_guard_and_records_errors(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "REPO", repo)
    clean = rv.Audit(tmp_path, tmp_path, "quick")
    assert clean.git_protection["status"] == "PASS"
    put(repo, "pages/overview.py")
    dirty = rv.Audit(tmp_path, tmp_path, "quick")
    dirty.check("GIT_PROTECTION", "fixture", lambda: dirty.git_protection, level="GIT_PROTECTION")
    assert dirty.results[0]["status"] == "BLOCKED"
    assert readiness(dirty.results, "quick")["overall"] == "BLOCKED"
    monkeypatch.setattr(rv, "REPO", tmp_path / "does not exist")
    unavailable = rv.Audit(tmp_path, tmp_path, "quick")
    assert unavailable.head is None and unavailable.dirty is None
    assert unavailable.git_protection["status"] == "ENVIRONMENT_ERROR"


@pytest.mark.parametrize("status", ["UNKNOWN", "ENVIRONMENT_ERROR", "BLOCKED", "FAIL", "NOT_RUN", "unexpected"])
def test_incomplete_or_unknown_never_overall_pass(status):
    gate = readiness([{"status":status}], "full")
    assert gate["regression_verdict"] != "PASS"


def test_calculation_pass_is_separate_from_dirty_block():
    rows = [{"status":"PASS", "validation_id":"GOLDEN", "execution_level":"ACTUAL_RERUN"},
            {"status":"BLOCKED", "validation_id":"GIT_PROTECTION", "execution_level":"GIT_PROTECTION"}]
    result = readiness(rows, "full")
    assert result["regression_verdict"] == "BLOCKED"
    assert result["calculation_regression"] == "PASS"


def test_empty_evidence_is_unknown_not_pass():
    assert readiness([], "full")["regression_verdict"] == "UNKNOWN"
