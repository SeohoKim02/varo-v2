"""T5 local-only release audit; never refreshes references or production actions."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import pandas as pd

from services.release_regression import (compare_golden, compare_mapping, inspect_reference,
                                         readiness, semantic_digest, sha256, json_safe, STATUSES)
from services.git_protection import inspect_git_protection

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("release_reference_manifest.json")
QUICK_TESTS = (
    "test_algorithm_contracts.py", "test_action_consistency.py", "test_shared_feasibility_selection.py",
    "test_seller_loss_engine.py", "test_seller_decision_validation.py", "test_seller_loss_inputs.py",
    "test_seller_loss_input_requirements.py", "test_seller_business_profile.py",
    "test_seller_loss_promotion_gate.py", "test_seller_shadow_ledger.py", "test_sales_input_normalization.py",
    "test_canonical_data.py", "test_canonical_integrity.py", "test_target_need_contract.py", "test_release_validation.py",
    "test_final_decision_integration.py",
    "test_git_protection.py",
    "test_validation_git_policies.py",
    "test_validation_runtime_isolation.py", "test_validation_apptest_lifecycle.py",
)
RESULT_FILES = ("release_validation_summary.json", "release_validation_matrix.csv", "release_validation_failures.csv",
                "golden_regression_by_day.csv", "golden_regression_by_strategy.csv", "validation_runtime.csv",
                "reference_integrity.json", "release_readiness.json")
TEST_TIMEOUT_SECONDS = 900


def now():
    return datetime.now(timezone.utc).isoformat()


def m5_not_run_reason(mode):
    if mode == "quick":
        return "QUICK_DEFAULT_OMITS_M5_RERUN"
    if mode == "full":
        return "FULL_M5_RERUN_OPTION_NOT_REQUESTED"
    raise ValueError("unknown release validation mode")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def git(*args):
    result = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise OSError(result.stderr.strip())
    return result.stdout.rstrip("\r\n")


def validator_inventory():
    """Static inventory: no imports/side effects from archived command entrypoints."""
    rows = []
    paths = sorted(set((REPO / "services").rglob("*validation.py")) |
                   set((REPO / "services").rglob("*revalidation.py")))
    for path in paths:
        if path.name == "release_validation.py":
            continue
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        strings = sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)})
        functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        args = [s for s in strings if s.startswith("--") and " " not in s]
        command = "python -m " + ".".join(path.relative_to(REPO).with_suffix("").parts)
        if path.stem == "m5_forecast_v2_validation": command += " --phase final"
        if path.stem == "m5_forecast_router_validation": command += " --phase production"
        rows.append({"validator": path.stem, "file": str(path.relative_to(REPO)), "command": command,
                     "functions": "|".join(functions), "arguments": "|".join(args),
                     "input_and_output_path_literals": "|".join(s for s in strings if s.endswith((".csv", ".parquet", ".json", ".xlsx", ".pt", ".sqlite3")) and "\n" not in s),
                     "writes_files": any(s in source for s in ("write_text", "to_csv", "to_parquet", "record_pipeline_run")),
                     "git_dependency": "subprocess" in source and "git" in source,
                     "golden_usage": "daily_strategy_summary" in source or "REFERENCE" in source or "golden_186" in source,
                     "model_requirement": "saved seed17/300 DQN" if "_dqn" in source or "infer_dqn" in source else "forecast inputs/config" if "forecast" in path.stem else None,
                     "runtime_seconds": None,
                     "safety_note": "library helper, no standalone validation CLI" if not any(f in functions for f in ("main", "validate_reference")) else
                                    "DO NOT RUN ORIGINAL CLI: includes DQN training" if path.stem == "suhyup_algorithm_revalidation" else
                                    "original CLI may overwrite archived reports; runner uses read-only helpers or fresh output"})
    return rows


def metadata_inventory(root, exclude):
    """Raw files not read by this audit: metadata protection (not content attestation)."""
    result = {}
    for folder in root.iterdir():
        if folder == exclude or folder.name.startswith("_RELEASE_VALIDATION_") or folder.name == "_SECRETS" or not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                stat = path.stat()
                result[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return result


def protected_hashes(root, manifest):
    from services import inventory_metric_integrity_validation as t4
    paths = {root / p for p in manifest["files"]}
    for name in set(t4.PROTECTED_BUNDLES) | {t4.OUTPUT_FOLDER}:
        paths.update(p for p in (root / name).rglob("*") if p.is_file())
    paths.update(p for p in (REPO / "outputs").rglob("*") if p.is_file() and p.suffix in (".pt", ".pth", ".sqlite3", ".db", ".wal", ".shm"))
    # SQLite sidecars can have suffixes .sqlite3-wal/.sqlite3-shm.
    paths.update(p for p in (REPO / "outputs").rglob("*") if p.is_file() and any(k in p.name for k in ("sqlite", "-wal", "-shm")))
    return {str(p): sha256(p) for p in sorted(paths)}


def model_check():
    from services.action_consistency_validation import _dqn_model
    model = _dqn_model()
    if model is None:
        return {"status": "BLOCKED", "difference": "MISSING_OUTPUT", "reason_codes": ["MODEL_UNAVAILABLE"], "actual_result": None}
    actual = {"sha256": sha256(model[1]), "signature": model[0], "seed": 17, "episodes": 300}
    result = compare_mapping(load(MANIFEST)["model"], actual)
    result["actual_result"] = {**actual, "path": str(model[1]), "training": False}
    return result


def golden_replay(root, spec):
    from services import shared_feasibility_validation as sfv, action_consistency_validation as acv
    from services import suhyup_algorithm_revalidation as rv
    from services.seller_decision_validation import _env
    from services.inventory_metric_integrity_validation import REAL_DATA_ROOT_ENV
    model = acv._dqn_model()
    if model is None:
        return {"status": "BLOCKED", "difference": "MISSING_OUTPUT", "reason_codes": ["MODEL_UNAVAILABLE"], "rows": []}
    reference = pd.read_csv(root / spec["path"], dtype={"date": str})
    if list(reference.columns) != spec["columns"]:
        return {"status": "FAIL", "difference": "SCHEMA_CHANGED", "reason_codes": ["GOLDEN_SCHEMA_CHANGED"], "rows": []}
    candidates = sfv.load_suhyup_candidates(root)
    rows, repeat = [], []
    with _env(REAL_DATA_ROOT_ENV, str(root)):
        for day in spec["dates"]:
            ranked = sfv.ranked_day(candidates, day)
            refs = sfv._reference_rows(ranked, day)
            refs["DQN"] = acv._dqn_row(ranked, day, model, refs["MILP"]["service_qty"])
            _, pareto = rv.pareto_operational_selection(ranked)
            refs["Pareto"].update(frontier_size=pareto.trace[0]["frontier_size"] if pareto.trace else 0,
                                  frontier_recalculation_steps=len(pareto.trace),
                                  active_objectives="|".join(pareto.active_objectives),
                                  inactive_objectives="|".join(pareto.inactive_objectives))
            rows.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in refs.values())
            # Representative repeat, actual code, not comparison of a copy with itself.
            if day == spec["dates"][0]:
                again = sfv._reference_rows(ranked, day)
                again["DQN"] = acv._dqn_row(ranked, day, model, again["MILP"]["service_qty"])
                repeat = [{"strategy": k, "same": all(r.get(f) == again[k].get(f) for f in
                           ("route_ids", "service_qty", "total_cost", "status", "feasibility_violations"))}
                          for k, r in refs.items()]
    actual = pd.DataFrame(rows)
    result = compare_golden(reference, actual, spec["dates"], spec["strategies"])
    result["expected_result"] = {"rows": spec["rows"], "dates": spec["dates"], "strategies": spec["strategies"],
                                 "columns": spec["columns"], "sha256": spec["sha256"]}
    result["actual_result"] = {"rows": len(actual), "equal_rows": sum(r["equal"] for r in result["rows"]),
                               "comparison_fields": list(reference), "repeat_first_day": repeat,
                               "model_sha256": sha256(model[1]), "production_action_applied": False}
    if not all(r["same"] for r in repeat):
        result.update(status="FAIL", difference="VALUE_CHANGED", reason_codes=["NONDETERMINISTIC"])
    return result


def t2_replay(root):
    from services import action_consistency_validation as v
    result = v.run_suhyup(root)
    days, records = result["by_day"], result["records"]
    actual = {"candidates": len(records), "selected": int(days.t1_selected_count.sum()),
              "executable_claims": int(days.proxy_executable_claims.sum()),
              "violations": int(result["t1_regression"].violations.sum()),
              "deterministic": all(result["determinism"]), "invariant_failures": result["invariant_failures"]}
    cmp = compare_mapping({"candidates": 620, "selected": 155, "executable_claims": 0,
                           "violations": 0, "deterministic": True, "invariant_failures": []}, actual)
    cmp["actual_result"] = actual
    return cmp


def t3_replay(root, output):
    from services import milp_benchmark_integrity_validation as v
    result = v.run_suhyup(root)
    actual = result["by_day"]
    reference = pd.read_csv(root / v.OUTPUT_FOLDER / "milp_benchmark_by_day.csv")
    # Every stable field, including route ids, signatures, solver status and both gap types.
    columns = [c for c in reference if not any(s in c for s in ("_ms", "seconds", "runtime", "memory"))]
    a, b = reference[columns].sort_values("date").reset_index(drop=True), actual[columns].sort_values("date").reset_index(drop=True)
    # A pipe-joined empty list is serialized as a blank CSV cell. pandas reads it
    # as NaN; this column is a list-of-groups string, not a nullable measurement.
    a["saved_vs_new_cap_groups_differing"] = a["saved_vs_new_cap_groups_differing"].fillna("")
    failures = []
    if len(a) != len(b) or a.date.nunique() != 31 or b.date.nunique() != 31:
        failures.append({"field": "date", "reason": "COMPLETENESS_CHANGED"})
    else:
        from services.release_regression import equal
        for i in range(len(a)):
            for c in columns:
                if not equal(a.loc[i, c], b.loc[i, c]):
                    failures.append({"date": a.loc[i, "date"], "field": c, "expected": a.loc[i, c], "actual": b.loc[i, c], "reason": "VALUE_CHANGED"})
    actual.to_csv(output / "t3_actual_by_day.csv", index=False)
    return {"status": "FAIL" if failures else "PASS", "difference": "VALUE_CHANGED" if failures else "UNCHANGED",
            "failure_count": len(failures), "failures": failures[:30],
            "actual_result": {"days": len(actual), "fields": columns, "global_optimal_claim": False}}


def sales_replay(root):
    from services import inventory_metric_integrity_validation as v, inventory_metric_integrity as imi
    from services import milp_benchmark_integrity_validation as mbv, shared_feasibility_validation as sfv
    rebuild, saved = mbv.GeneratorRebuild(root), sfv.load_suhyup_candidates(root)
    affected, unavailable, preserved = 0, 0, True
    day_results = []
    for day in sorted(saved.snapshot_date.unique()):
        declared, plain = v.declared_day_upload(rebuild, day), rebuild.upload(day)
        a, b = v._pipeline(declared, root), v._pipeline(plain, root)
        ar = {r["route_id"]: r["varo_action"] for r in a["recommendations"]}
        br = {r["route_id"]: r["varo_action"] for r in b["recommendations"]}
        old = saved[saved.snapshot_date == day]
        preserved &= ar == dict(zip(old.route_id, old.varo_action))
        shifts = sum(br.get(k) != action for k, action in ar.items())
        affected += shifts
        shadow = imi.build_sales_normalization_analysis(declared, candidates=pd.DataFrame(a["recommendations"]))
        no_sales = all(r["sales_30d"] is None for r in shadow["validated_sales_signals"])
        action_rows = shadow["validated_action_impact"]
        valid_unavailable = len(action_rows) == len(ar) and all(
            r["validated_action"] is None and r["validated_canonical_action"] is None
            and r["status"] == "VALIDATED_ACTION_UNAVAILABLE" and not r["production_action_applied"]
            for r in action_rows)
        if no_sales and valid_unavailable:
            unavailable += len(ar)
        day_results.append({"date": day, "alias_sensitive": shifts, "same_saved_actions": ar == dict(zip(old.route_id, old.varo_action)),
                            "validated_sales_unavailable": no_sales, "validated_action_unavailable": valid_unavailable})
    scenarios = v.controlled_scenarios()
    actual = {"alias_sensitive": affected, "validated_action_unavailable": unavailable, "saved_actions_preserved": preserved,
              "days": len(day_results), "controlled_pass": all(s["pass"] for s in scenarios)}
    result = compare_mapping({"alias_sensitive": 84, "validated_action_unavailable": 620,
                              "saved_actions_preserved": True, "days": 31, "controlled_pass": True}, actual)
    result["actual_result"], result["days"] = actual, day_results
    return result


class Audit:
    def __init__(self, root, output, mode):
        self.root, self.output, self.mode = root, output, mode
        self.results, self.failures, self.golden_rows = [], [], []
        self.git_protection = inspect_git_protection(REPO)
        context = self.git_protection["actual_result"]
        self.head, self.dirty = context["head"], context["git_dirty"]
        self.execution_version = context.get("execution_version", {})
        self.manifest = load(MANIFEST)

    def check(self, key, name, fn=None, *, level="ACTUAL_RERUN", source=None, not_run_reason=None):
        started, tick = now(), time.perf_counter()
        print(f"[{self.mode}] {key}: {name}", flush=True)
        try:
            result = fn() if fn else {"status": "NOT_RUN", "difference": "NOT_COMPARABLE",
                                     "reason_codes": [not_run_reason or ("QUICK_SCOPE_EXCLUDES_ACTUAL_BENCHMARK"
                                                      if self.mode == "quick" else "FULL_CHECK_NOT_EXECUTED")]}
            if not isinstance(result, dict) or result.get("status") not in STATUSES:
                raise ValueError("VALIDATION_RESULT_CONTRACT_INVALID")
        except FileNotFoundError as error:
            result = {"status": "BLOCKED", "difference": "MISSING_REFERENCE", "reason_codes": [str(error)]}
        except OSError as error:
            result = {"status": "ENVIRONMENT_ERROR", "difference": "NOT_COMPARABLE", "reason_codes": ["FILE_ACCESS_OR_ENVIRONMENT", str(error)]}
        except Exception as error:
            result = {"status": "FAIL", "difference": "NOT_COMPARABLE", "reason_codes": [type(error).__name__, str(error)]}
        row = {"validation_id": key, "validation_name": name, "validation_mode": self.mode,
               "started_at": started, "finished_at": now(), "duration": round(time.perf_counter() - tick, 4),
               "status": result["status"], "expected_result": result.get("expected_result"), "actual_result": result.get("actual_result"),
               "difference": result.get("difference", "UNCHANGED" if result["status"] == "PASS" else "NOT_COMPARABLE"),
               "reason_codes": result.get("reason_codes", []), "data_source": source,
               "reference_version": self.manifest["version"], "git_head": self.head, "git_dirty": self.dirty,
               "output_path": str(self.output), "execution_level": level if fn else "NOT_RUN",
               "failure_count": result.get("failure_count", len(result.get("failures", [])))}
        self.results.append(row)
        for failure in result.get("failures", []):
            self.failures.append({"validation_id": key, **failure})
        if row["status"] != "PASS" and not result.get("failures"):
            self.failures.append({"validation_id": key, "field": None, "reason": "|".join(row["reason_codes"]), "status": row["status"]})
        if key == "GOLDEN": self.golden_rows = result.get("rows", [])
        if result.get("days"):
            pd.DataFrame(result["days"]).to_csv(self.output / "sales_actual_by_day.csv", index=False)
        return result

    def tests(self, scratch):
        # Preserve the last running node and thread stacks if a native/UI test stalls.
        args = [sys.executable, "-m", "pytest", "-vv", "-o", "faulthandler_timeout=60", "-p", "no:cacheprovider"]
        if self.mode == "quick": args.extend(str(Path("tests") / n) for n in QUICK_TESTS)
        env = dict(os.environ, VARO_OUTPUT_ROOT=str(scratch / "model_outputs"), PYTHONDONTWRITEBYTECODE="1", TMP=str(scratch), TEMP=str(scratch))
        start = time.perf_counter()
        timed_out = False
        with (self.output / "pytest.log").open("w", encoding="utf-8") as handle:
            try:
                process = subprocess.run(args, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                         timeout=TEST_TIMEOUT_SECONDS)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out, exit_code = True, None
                handle.write(f"\nPYTEST_TIMEOUT after {TEST_TIMEOUT_SECONDS}s; result is not a pass.\n")
        log = (self.output / "pytest.log").read_text(encoding="utf-8")
        count = re.search(r"(\d+) passed", log)
        passed = int(count.group(1)) if count else None
        skip = bool(re.search(r"\b\d+ (?:skipped|xfailed|xpassed|failed|error|warnings?)\b", log))
        # Windows native faults can be logged even when pytest later exits zero.
        # Keep that evidence and fail closed instead of trusting only its summary.
        native_failures = [marker for marker in (
            "Windows fatal exception:", "Fatal Python error:", "Exception ignored in thread started by:"
        ) if marker in log]
        ok = not timed_out and not native_failures and exit_code == 0 and passed is not None and not skip and (self.mode != "full" or passed >= 1338)
        reasons = (["PYTEST_TIMEOUT"] if timed_out else []) + (["PYTEST_NATIVE_FAILURE"] if native_failures else [])
        if not ok and not reasons:
            reasons = ["PYTEST_OR_COLLECTION_REGRESSION"]
        return {"status": "PASS" if ok else "FAIL", "actual_result": {"exit_code": exit_code, "passed": passed,
                "timed_out": timed_out, "timeout_seconds": TEST_TIMEOUT_SECONDS,
                "native_failure_markers": native_failures,
                "pytest_summary": log.splitlines()[-1] if log.splitlines() else None, "duration": round(time.perf_counter() - start, 2),
                "production_model_training": False, "note": "existing unit tests may train tiny fixtures in managed scratch"},
                "reason_codes": reasons}

    def references(self):
        rows = [{"path": path, "expected": digest, **inspect_reference(self.root / path, digest)}
                for path, digest in self.manifest["files"].items()]
        (self.output / "reference_integrity.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        statuses = {r["status"] for r in rows}
        status = "ENVIRONMENT_ERROR" if "ENVIRONMENT_ERROR" in statuses else "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
        return {"status": status, "failure_count": sum(r["status"] != "PASS" for r in rows),
                "actual_result": {"references": len(rows), "unchanged": sum(r["status"] == "PASS" for r in rows)},
                "failures": [{"field": r["path"], "expected": r["expected"], "actual": r["actual"], "reason": r["difference"]} for r in rows if r["status"] != "PASS"][:30]}

    def frozen(self, name, path):
        # Hash is compared against the checked-in pin, NOT a hash taken at this run's start.
        digest = self.manifest["files"][path]
        result = inspect_reference(self.root / path, digest)
        result["expected_result"] = {"sha256": digest, "reference_version": self.manifest["version"]}
        value = load(self.root / path)
        result["actual_result"] = {"sha256": result.pop("actual"), "result": value,
                                    "benchmark_reexecuted": False, "label": "REUSED_FROZEN_RESULT"}
        return result

    def finish(self, seconds):
        gate = readiness(self.results, self.mode)
        semantic = [{k: r[k] for k in ("validation_id", "status", "difference", "actual_result", "reason_codes", "execution_level")}
                    for r in self.results]
        summary = {"mode": self.mode, "results": self.results, "readiness": gate, "runtime_seconds": seconds,
                   "execution_version": self.execution_version,
                   "peak_memory_mb": None, "semantic_digest": semantic_digest(semantic),
                   "production_action_applied": False, "limitations": ["frozen validation is not a benchmark rerun",
                    "raw unread files protected by size/mtime metadata; referenced canonical/archive/model/DB files by SHA256",
                    "no actual outcomes or operational approval", "no automatic baseline refresh"]}
        def write_json(name, value):
            (self.output / name).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
        write_json(RESULT_FILES[0], summary)
        write_json(RESULT_FILES[7], gate)
        pd.DataFrame([{**r, **{k: json.dumps(r[k], ensure_ascii=False, default=str) for k in ("expected_result", "actual_result", "reason_codes")}} for r in self.results]).to_csv(self.output / RESULT_FILES[1], index=False)
        pd.DataFrame(self.failures, columns=["validation_id", "date", "strategy", "product", "route", "field", "expected", "actual", "reason", "status"]).to_csv(self.output / RESULT_FILES[2], index=False)
        pd.DataFrame(self.results)[["validation_id", "duration", "status", "execution_level"]].to_csv(self.output / RESULT_FILES[5], index=False)
        golden = pd.DataFrame(self.golden_rows, columns=["date", "strategy", "equal", "changed_fields"])
        golden.to_csv(self.output / RESULT_FILES[3], index=False)
        aggregated = golden.groupby("strategy").agg(compared=("equal", "size"), equal=("equal", "sum")).reset_index() if not golden.empty else pd.DataFrame(columns=["strategy", "compared", "equal"])
        aggregated.to_csv(self.output / RESULT_FILES[4], index=False)
        return {"output_dir": str(self.output), "mode": self.mode, "readiness": gate, "runtime_seconds": seconds,
                "semantic_digest": summary["semantic_digest"]}


def run_validation(root, mode="quick", *, output_dir=None, rerun_m5=False):
    if mode not in ("quick", "full"): raise ValueError("quick/full only")
    if rerun_m5 and mode != "full": raise ValueError("M5 rerun requires full")
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / ("_RELEASE_VALIDATION_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    if output.parent != root or not output.name.startswith("_RELEASE_VALIDATION_"):
        raise ValueError("new direct-child _RELEASE_VALIDATION_* output required")
    output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    audit = Audit(root, output, mode)
    pd.DataFrame(validator_inventory()).to_csv(output / "validator_inventory.csv", index=False)
    before, raw_before = None, None
    def snapshot():
        nonlocal before, raw_before
        before = protected_hashes(root, audit.manifest)
        raw_before = metadata_inventory(root, output)
        return {"status": "PASS", "actual_result": {"content_hashed": len(before), "metadata_checked": len(raw_before)}}
    audit.check("ENV", "protected input snapshot", snapshot)
    audit.check("REFERENCES", "pinned reference hashes", audit.references, level="REUSED_FROZEN_RESULT", source=str(root))
    audit.check("MODEL", "DQN seed17 saved model", model_check, level="READ_ONLY_CONTRACT")
    audit.check("GIT_PROTECTION", "protected working tree (separate from calculation)",
                lambda: audit.git_protection, level="GIT_PROTECTION")
    audit.check("DIFF_CHECK", "Git whitespace check", lambda: {"status": "PASS" if subprocess.run(["git", "diff", "--check"], cwd=REPO, capture_output=True).returncode == 0 else "FAIL"})
    with tempfile.TemporaryDirectory(prefix="varo-release-") as temp:
        scratch = Path(temp)
        audit.check("TESTS", "unit tests" if mode == "quick" else "all existing and new tests", lambda: audit.tests(scratch))
        full = mode == "full"
        from services import canonical_reference_validation as crv, inventory_metric_integrity_validation as t4
        audit.check("CANONICAL", "canonical bridge and conservation", (lambda: {"status": "PASS", "actual_result": crv.validate_reference(root, output_path=output / "canonical_actual.json")}) if full else None)
        audit.check("GOLDEN", "31 days x 6 strategies", (lambda: golden_replay(root, audit.manifest["golden"])) if full else None)
        def t1():
            actual = t4.t1_rerun(root)
            expected = {"benchmark_service": 7750.0, "benchmark_cost": 13629146.0, "benchmark_violations": 0, "strict_selected": 0,
                        "strict_claim": "INSUFFICIENT_CAP_DATA on 31 days: target_need and source_surplus are PROXY"}
            result = compare_mapping(expected, {k: actual[k] for k in expected})
            result.update(expected_result=expected, actual_result=actual)
            return result
        audit.check("T1", "shared feasibility actual 31-day replay", t1 if full else None)
        audit.check("T2", "action and proxy consistency actual 620 candidates", (lambda: t2_replay(root)) if full else None)
        audit.check("T3", "limited-pool MILP actual 31-day replay", (lambda: t3_replay(root, output)) if full else None)
        frozen = [
            ("P1", "candidate coverage", "_CANDIDATE_GENERATION_VALIDATION/candidate_generation_validation.json"),
            ("P1B", "operational evidence", "_OPERATIONAL_EVIDENCE_VALIDATION/operational_candidate_validation.json"),
            ("P1C", "need contract", "_TARGET_NEED_CONTRACT_VALIDATION/operational_need_contract_summary.json"),
            ("T4_FROZEN", "inventory metrics", "_INVENTORY_METRIC_INTEGRITY/inventory_metric_integrity_summary.json"),
            ("SELLER_PROFILE", "input/planner/profile", "_SELLER_LOSS_VALIDATION/seller_loss_profile_summary.json"),
            ("PROMOTION", "promotion gate", "_SELLER_LOSS_VALIDATION/seller_loss_promotion_summary.json"),
            ("LEDGER", "shadow ledger", "_SELLER_LOSS_VALIDATION/seller_shadow_ledger_validation.json"),
            ("M5", "forecast gate/config/metrics (all frozen files pinned)", "20_WALMART_M5/results/m5_forecast_v2_promotion_gate.json"),
            ("EXTERNAL", "generalization FAIL remains FAIL", "09_External_Validation/results/forecast_external_generalization.json"),
        ]
        for key, name, path in frozen:
            audit.check(key, name, lambda k=key, p=path: audit.frozen(k, p), level="REUSED_FROZEN_RESULT", source=path)
        audit.check("T4B", "validated sales/legacy actions actual 31-day replay", (lambda: sales_replay(root)) if full else None)
        from services.seller_decision_validation import run_scenarios
        def seller():
            first, _ = run_scenarios()
            second, _ = run_scenarios()
            # DataFrame NaN != NaN is not a decision change. Compare typed unavailable values.
            same = semantic_digest(first.to_dict("records")) == semantic_digest(second.to_dict("records"))
            return {"status": "PASS" if len(first) == 18 and first["pass"].all() and same else "FAIL",
                    "actual_result": {"scenarios": len(first), "passed": int(first["pass"].sum()), "deterministic": same}}
        audit.check("SELLER", "18 controlled scenarios, repeated", seller)
        def seller_saved_output(module, name):
            destination = output / name
            destination.mkdir()
            result = importlib.import_module("services." + module).run_validation(root, output_dir=destination)
            failed = result.get("failed_checks", [])
            checks = result.get("checks", {})
            if isinstance(checks, dict):
                failed += [k for k, value in checks.items() if isinstance(value, bool) and not value]
            return {"status": "FAIL" if failed else "PASS", "actual_result": result, "reason_codes": failed}
        for key, module in (("SELLER_INPUT_RERUN", "seller_loss_input_validation"),
                            ("SELLER_PLANNER_RERUN", "seller_loss_requirement_validation"),
                            ("SELLER_PROFILE_RERUN", "seller_business_profile_validation"),
                            ("PROMOTION_RERUN", "seller_loss_promotion_validation")):
            audit.check(key, "real input plus explicitly labelled TEST wiring", (lambda m=module, k=key: seller_saved_output(m, k.lower())) if full else None)
        def m5():
            from services import m5_forecast_v2_validation as v
            destination = output / "m5_rerun"
            destination.mkdir()
            # Read-only frozen selection copied into a fresh report directory; never reselect after holdout.
            shutil.copy2(root / "20_WALMART_M5/results/m5_forecast_v2_model_selection.json", destination)
            report = v.run_final(root, destination, write_rows=False)
            expected = load(root / "20_WALMART_M5/results/m5_forecast_v2_promotion_gate.json")
            actual = load(destination / "m5_forecast_v2_promotion_gate.json")
            columns = ("config_signature", "decision", "passed", "failed", "final_metrics")
            result = compare_mapping({k: expected[k] for k in columns}, {k: actual[k] for k in columns})
            result["actual_result"] = {"rerun": True, "decision": report["decision"]}
            return result
        audit.check("M5_RERUN", "optional frozen-selected forecast actual replay", m5 if rerun_m5 else None,
                    not_run_reason=m5_not_run_reason(mode))
        def isolated_ledger():
            from services.seller_shadow_validation import run_validation as run
            destination = output / "shadow_rerun"
            destination.mkdir()
            validation, pilot = run(root, history_dir=scratch / "history", output_dir=destination)
            return {"status": "PASS" if all(pilot["checks"].values()) else "FAIL", "actual_result": validation}
        audit.check("LEDGER_RERUN", "separate SQLite shadow validation", isolated_ledger if full else None)
    def unchanged():
        if before is None or raw_before is None:
            return {"status": "BLOCKED", "reason_codes": ["NO_PRE_EXECUTION_SNAPSHOT"]}
        after, raw_after = protected_hashes(root, audit.manifest), metadata_inventory(root, output)
        content_changes = [p for p in set(before) | set(after) if before.get(p) != after.get(p)]
        metadata_changes = [p for p in set(raw_before) | set(raw_after) if raw_before.get(p) != raw_after.get(p)]
        return {"status": "FAIL" if content_changes or metadata_changes else "PASS",
                "actual_result": {"content_changes": content_changes, "metadata_changes": metadata_changes,
                                  "content_hash_count": len(before), "metadata_file_count": len(raw_before)},
                "reason_codes": ["PROTECTED_DATA_CHANGED"] if content_changes or metadata_changes else []}
    audit.check("PROTECTION_AFTER", "all protected files unchanged", unchanged, level="READ_ONLY_CONTRACT")
    audit.check("GIT_PROTECTION_AFTER", "protected working tree after validation",
                lambda: inspect_git_protection(REPO), level="GIT_PROTECTION")
    return audit.finish(round(time.perf_counter() - tick, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\VARO_V2_REAL_DATA"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rerun-m5", action="store_true")
    args = parser.parse_args()
    result = run_validation(args.data_root, args.mode, output_dir=args.output_dir, rerun_m5=args.rerun_m5)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["readiness"]["overall"] == "BLOCKED" else 0)


if __name__ == "__main__":
    main()
