"""Read-only Git collection; each validator declares its own protection policy."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


POLICY_VERSION = "varo-git-protection-v1-z-2"
# Directory boundaries are deliberate. No old protected path is removed.
PROTECTED_PATHS = (
    "services", "pages", "components", "simulation", "legacy_adapters", "tests",
    "app_v2.py", "app_candidate.py", "router.py", "styles.py", "requirements.txt", "requirements-dqn.txt",
    ".streamlit", "pyproject.toml", "pytest.ini", "conftest.py", "setup.cfg",
    "data", "samples", "outputs/dqn", "outputs/simulation_history",
)


def _path(value: str) -> str:
    value = value.replace("\\", "/")
    if not value or value.startswith("/") or ":" in value.split("/")[0] or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("invalid repository-relative Git path")
    return value


def is_protected(path: str, protected=PROTECTED_PATHS) -> bool:
    path = _path(path)
    return any(path == _path(p) or path.startswith(_path(p) + "/") for p in protected)


def parse_status_z(raw: bytes) -> list[dict]:
    """Porcelain v1 -z: XY destination NUL [rename/copy source NUL]."""
    if not raw:
        return []
    if not raw.endswith(b"\0"):
        raise ValueError("unterminated porcelain status")
    fields = raw[:-1].split(b"\0")
    result, i = [], 0
    while i < len(fields):
        item = fields[i]
        if len(item) < 4 or item[2:3] != b" ":
            raise ValueError("malformed porcelain status record")
        status = item[:2].decode("ascii")
        if any(c not in " MADRCUT?" for c in status) or status == "  ":
            raise ValueError("unexpected porcelain status code")
        if "?" in status and status != "??":
            raise ValueError("invalid untracked status")
        row = {"index_status": status[0], "worktree_status": status[1],
               "path": _path(item[3:].decode("utf-8")), "original_path": None}
        if "R" in status or "C" in status:
            i += 1
            if i >= len(fields):
                raise ValueError("missing rename/copy source")
            row["original_path"] = _path(fields[i].decode("utf-8"))
        result.append(row)
        i += 1
    return result


def _git(cwd: Path, *args: str) -> bytes:
    process = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, timeout=30)
    if process.returncode:
        raise OSError(f"Git {args[0]} exit {process.returncode}: " + process.stderr.decode("utf-8", errors="replace").rstrip())
    return process.stdout


def _text(cwd: Path, *args: str) -> str:
    return _git(cwd, *args).decode("utf-8").rstrip("\r\n")


def execution_version() -> dict:
    base = Path(__file__).parent
    names = ("git_protection.py", "release_validation.py", "release_regression.py", "action_consistency_validation.py",
             "candidate_generation_validation.py", "operational_evidence_validation.py",
             "target_need_contract_validation.py", "inventory_metric_integrity_validation.py",
             "milp_benchmark_integrity_validation.py", "external_forecast_validation.py")
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}


def inspect_git_protection(cwd: Path, protected=PROTECTED_PATHS, *,
                           protect_all_tracked=False, allowed_tracked=(), policy_name=None) -> dict:
    """Unknown/error is never clean. Remote comparison is local-ref evidence only."""
    actual = {"repo_root": None, "head": None, "git_dirty": None,
              "changes": [], "changed_paths": [], "protected_changes": [],
              "policy_version": POLICY_VERSION, "protected_roots": list(protected),
              "policy_name": policy_name, "protect_all_tracked": protect_all_tracked,
              "allowed_tracked": list(allowed_tracked),
              "porcelain_version": "v1 -z", "remote_contacted": False}
    try:
        root = Path(_text(Path(cwd), "rev-parse", "--show-toplevel")).resolve()
        actual.update(repo_root=str(root), head=_text(root, "rev-parse", "--verify", "HEAD"),
                      execution_version=execution_version())
        raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none")
        changes = parse_status_z(raw)
        paths = sorted({p for row in changes for p in (row["path"], row["original_path"]) if p is not None})
        # Legacy research policies allow new research files but block tracked edits.
        # Explicit roots (validator + policy implementation) always take precedence.
        allowed = {_path(p) for p in allowed_tracked}
        tracked_paths = {p for row in changes
                         if (row["index_status"], row["worktree_status"]) != ("?", "?")
                         for p in (row["path"], row["original_path"]) if p is not None}
        protected_changes = [p for p in paths if is_protected(p, protected)
                             or (protect_all_tracked and p in tracked_paths and p not in allowed)]
        actual.update(changes=changes, changed_paths=paths, protected_changes=protected_changes,
                      tracked_changed_paths=sorted(tracked_paths),
                      git_dirty=bool(changes), status_sha256=hashlib.sha256(raw).hexdigest(),
                      changed_content_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                              if (root / p).is_file() else None for p in protected_changes})
        # Lack of an upstream is not a calculation failure. Never fetch or push.
        branch = _text(root, "rev-parse", "--abbrev-ref", "HEAD")
        ref = _text(root, "for-each-ref", "--format=%(upstream:short)", "refs/heads/" + branch)
        if ref:
            ahead, behind = map(int, _text(root, "rev-list", "--left-right", "--count", "HEAD..." + ref).split())
            actual["upstream"] = {"ref": ref, "ahead": ahead, "behind": behind,
                                  "status": "SYNCHRONIZED_LOCAL_REF" if ahead == behind == 0 else "DIVERGED_LOCAL_REF"}
        else:
            actual["upstream"] = {"status": "NOT_CONFIGURED"}
        return {"status": "BLOCKED" if protected_changes else "PASS", "actual_result": actual,
                "reason_codes": ["PROTECTED_CODE_UNCOMMITTED"] if protected_changes else []}
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError) as error:
        return {"status": "ENVIRONMENT_ERROR", "actual_result": actual,
                "reason_codes": ["GIT_INSPECTION_FAILED", type(error).__name__, str(error)]}


def inspect_validator_git(cwd: Path, validator_path: str, *, protected=(),
                          protect_all_tracked=False, allowed_tracked=()) -> dict:
    """Reuse collection without replacing a validator's historical policy scope."""
    roots = tuple(dict.fromkeys((*protected, validator_path, "services/git_protection.py")))
    return inspect_git_protection(cwd, roots, protect_all_tracked=protect_all_tracked,
                                  allowed_tracked=allowed_tracked, policy_name=validator_path)


def git_validation_check(guard: dict, description: str, check_id="G1") -> dict:
    """Keep protection/environment status distinct from numerical calculation FAIL."""
    return {"check_id": check_id, "description": description, "status": guard["status"],
            "detail": guard["actual_result"], "reason_codes": guard["reason_codes"]}
