"""Independent repositories prove collection sharing does not flatten legacy policies."""
import importlib
import subprocess

import pytest

from services import git_protection as gp, release_validation as rv


POLICIES = (
    ("candidate_generation_validation", True, ()),
    ("operational_evidence_validation", True, ()),
    ("target_need_contract_validation", True, ("services/data_loader.py", "services/analysis_pipeline.py")),
    ("inventory_metric_integrity_validation", True, ("services/analysis_pipeline.py",)),
    ("milp_benchmark_integrity_validation", False, ()),
    ("external_forecast_validation", False, ()),
)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


def put(root, path, text="changed\n"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "policy fixture"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Varo isolated policy fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "core.autocrlf", "false")
    paths = ["services/git_protection.py", "services/data_loader.py", "services/analysis_pipeline.py",
             "pages/overview.py", "notes.txt", "services/demand_forecast_router.py"]
    paths += [f"services/{name}.py" for name, _, _ in POLICIES]
    for path in paths:
        put(root, path, "baseline\n")
    git(root, "add", "--all")
    git(root, "commit", "-m", "isolated fixture only")
    return root


@pytest.mark.parametrize("name,tracked,allowed", POLICIES)
def test_each_policy_clean_and_declared(repo, name, tracked, allowed):
    module = importlib.import_module("services." + name)
    result = module.validation_git_guard(repo)
    assert result["status"] == "PASS"
    policy = result["actual_result"]
    assert policy["protect_all_tracked"] is tracked
    assert tuple(policy["allowed_tracked"]) == allowed
    assert f"services/{name}.py" in policy["protected_roots"]
    assert "services/git_protection.py" in policy["protected_roots"]
    if name == "milp_benchmark_integrity_validation":
        assert set(module.PROTECTED_PATHS) < set(policy["protected_roots"])
    if name == "external_forecast_validation":
        assert set(module.FROZEN_SOURCES) < set(policy["protected_roots"])


@pytest.mark.parametrize("name,tracked,allowed", POLICIES)
@pytest.mark.parametrize("target", ["self", "policy"])
def test_validator_and_policy_cannot_change_unnoticed(repo, name, tracked, allowed, target):
    path = f"services/{name}.py" if target == "self" else "services/git_protection.py"
    put(repo, path)
    result = importlib.import_module("services." + name).validation_git_guard(repo)
    assert result["status"] == "BLOCKED"
    assert path in result["actual_result"]["protected_changes"]
    record = gp.git_validation_check(result, "fixture")
    assert record["status"] == "BLOCKED" and record["reason_codes"] == ["PROTECTED_CODE_UNCOMMITTED"]


@pytest.mark.parametrize("name,tracked,allowed", POLICIES)
def test_unrelated_tracked_policy_is_preserved(repo, name, tracked, allowed):
    put(repo, "notes.txt")
    result = importlib.import_module("services." + name).validation_git_guard(repo)
    assert result["status"] == ("BLOCKED" if tracked else "PASS")
    assert result["actual_result"]["git_dirty"] is True


@pytest.mark.parametrize("name,tracked,allowed", POLICIES)
def test_nested_untracked_is_collected_but_legacy_new_file_policy_preserved(repo, name, tracked, allowed):
    path = "research/new folder/한글 새 파일.py"
    put(repo, path)
    result = importlib.import_module("services." + name).validation_git_guard(repo / "services")
    assert result["status"] == "PASS"
    assert result["actual_result"]["changed_paths"] == [path]


@pytest.mark.parametrize("name,allowed_path", [
    ("target_need_contract_validation", "services/data_loader.py"),
    ("target_need_contract_validation", "services/analysis_pipeline.py"),
    ("inventory_metric_integrity_validation", "services/analysis_pipeline.py"),
])
def test_tracked_exceptions_exact_and_rename_cannot_escape(repo, name, allowed_path):
    module = importlib.import_module("services." + name)
    put(repo, allowed_path)
    git(repo, "add", allowed_path)
    assert module.validation_git_guard(repo)["status"] == "PASS"
    git(repo, "mv", allowed_path, "renamed_policy.py")
    result = module.validation_git_guard(repo)
    assert result["status"] == "BLOCKED"
    assert result["actual_result"]["protected_changes"] == ["renamed_policy.py"]
    assert set(result["actual_result"]["changed_paths"]) == {allowed_path, "renamed_policy.py"}


@pytest.mark.parametrize("direction", ["in", "out"])
def test_t3_rename_checks_both_endpoints(repo, direction):
    from services import milp_benchmark_integrity_validation as v
    source, target = (("pages/overview.py", "outside.txt") if direction == "out" else ("notes.txt", "pages/renamed.py"))
    git(repo, "mv", source, target)
    result = v.validation_git_guard(repo)
    assert result["status"] == "BLOCKED"
    assert set(result["actual_result"]["changed_paths"]) == {source, target}


def test_t3_nested_new_file_and_prefix_boundary(repo):
    from services import milp_benchmark_integrity_validation as v
    put(repo, "pages_backup/not_a_page.py")
    assert v.validation_git_guard(repo)["status"] == "PASS"
    put(repo, "pages/new folder/상품 화면.py")
    assert v.validation_git_guard(repo)["actual_result"]["protected_changes"] == ["pages/new folder/상품 화면.py"]


@pytest.mark.parametrize("name,tracked,allowed", POLICIES)
def test_git_error_never_disappears_into_empty_changes(repo, monkeypatch, name, tracked, allowed):
    def fail(*args, **kwargs):
        raise FileNotFoundError("git unavailable fixture")
    monkeypatch.setattr(gp, "_git", fail)
    result = importlib.import_module("services." + name).validation_git_guard(repo)
    assert result["status"] == "ENVIRONMENT_ERROR"
    assert result["actual_result"]["git_dirty"] is None
    assert gp.git_validation_check(result, "fixture")["status"] != "PASS"


def test_truncated_output_is_environment_error(repo, monkeypatch):
    original = gp._git
    def malformed(cwd, *args):
        return b"R  pages/new.py\0" if args[0] == "status" else original(cwd, *args)
    monkeypatch.setattr(gp, "_git", malformed)
    assert gp.inspect_validator_git(repo, "services/policy.py")["status"] == "ENVIRONMENT_ERROR"


def test_outside_repository_not_assumed_clean(tmp_path):
    result = gp.inspect_validator_git(tmp_path, "services/policy.py")
    assert result["status"] == "ENVIRONMENT_ERROR" and result["actual_result"]["git_dirty"] is None


def test_external_unknown_frozen_source_is_not_unchanged():
    from services import external_forecast_validation as ev
    record = ev.frozen_config_record()
    record["frozen_sources_match_git_head"] = None
    check = ev.frozen_config_check(record)
    assert check["FROZEN_CONFIG_UNCHANGED"] is False
    assert "frozen_sources_match_git_head" in check["differences"]
    assert ev._code_validation_status({"record": {}}) == "UNKNOWN"


def test_external_git_failure_is_explicit(monkeypatch):
    from services import external_forecast_validation as ev
    def fail(*args, **kwargs):
        raise PermissionError("fixture git denied")
    monkeypatch.setattr(gp, "_git", fail)
    check = ev.frozen_config_check()
    assert check["FROZEN_CONFIG_UNCHANGED"] is False
    assert ev._code_validation_status(check) == "ENVIRONMENT_ERROR"


@pytest.mark.parametrize("mode,reason", [
    ("quick", "QUICK_DEFAULT_OMITS_M5_RERUN"),
    ("full", "FULL_M5_RERUN_OPTION_NOT_REQUESTED"),
])
def test_m5_not_run_reason_and_execution_level(repo, tmp_path, monkeypatch, mode, reason):
    monkeypatch.setattr(rv, "REPO", repo)
    audit = rv.Audit(tmp_path, tmp_path, mode)
    audit.check("M5_RERUN", "optional M5", not_run_reason=rv.m5_not_run_reason(mode))
    row = audit.results[0]
    assert row["status"] == row["execution_level"] == "NOT_RUN"
    assert row["reason_codes"] == [reason]


def test_m5_unknown_mode_rejected():
    with pytest.raises(ValueError):
        rv.m5_not_run_reason("other")


@pytest.mark.parametrize("name", [name for name, _, _ in POLICIES if name != "external_forecast_validation"] + ["action_consistency_validation"])
@pytest.mark.parametrize("status", ["PASS", "BLOCKED", "ENVIRONMENT_ERROR"])
def test_standalone_cli_nonpass_is_not_success(monkeypatch, name, status):
    module = importlib.import_module("services." + name)
    fake = {"failed_checks": [] if status == "PASS" else ["G1"],
            "checks": [{"check_id": "G1", "status": status}], "verdicts": {},
            "cutoff_totals": {}, "readiness": {}, "runtime": {}}
    monkeypatch.setattr("sys.argv", [name])
    monkeypatch.setattr(module, "run_validation", lambda *args, **kwargs: fake)
    assert module.main() == (0 if status == "PASS" else 1)


@pytest.mark.parametrize("status", ["PASS", "BLOCKED", "ENVIRONMENT_ERROR", "UNKNOWN"])
def test_external_cli_exposes_code_protection_and_exit(monkeypatch, capsys, status):
    from services import external_forecast_validation as ev
    monkeypatch.setattr("sys.argv", ["external_forecast_validation", "--summary"])
    fake = {"GENERALIZATION_VERDICT": "FAIL", "FROZEN_CONFIG_UNCHANGED": True,
            "code_validation_status": status, "datasets": {}}
    monkeypatch.setattr(ev, "write_summary", lambda *args, **kwargs: fake)
    assert ev.main() == (0 if status == "PASS" else 1)
    assert '"code_validation_status": "' + status + '"' in capsys.readouterr().out
