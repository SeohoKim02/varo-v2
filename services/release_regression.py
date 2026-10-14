"""Read-only comparison primitives. No baseline creation or update API."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

STATUSES = {"PASS", "FAIL", "BLOCKED", "NOT_RUN", "INSUFFICIENT_DATA", "ENVIRONMENT_ERROR"}
VOLATILE = {"started_at", "finished_at", "duration", "runtime", "seconds", "generated_at", "output_path", "pytest_summary", "database"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in sorted(value.items()) if k not in VOLATILE}
    if isinstance(value, (tuple, list)):
        return [stable(v) for v in value]
    if hasattr(value, "item"):
        return stable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_safe(value):
    """Keep metadata; serialize historical NaN/Infinity as unavailable, never zero."""
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [json_safe(v) for v in value]
    if hasattr(value, "item"): return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def semantic_digest(value) -> str:
    return hashlib.sha256(json.dumps(stable(value), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, default=str).encode()).hexdigest()


def equal(a, b, tolerance=1e-6):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tolerance
    return a == b


def compare_mapping(expected, actual):
    if expected is None:
        return {"status": "BLOCKED", "difference": "MISSING_REFERENCE", "failures": []}
    if actual is None:
        return {"status": "FAIL", "difference": "MISSING_OUTPUT", "failures": []}
    expected, actual = stable(expected), stable(actual)
    failures = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    failures.append({"field": path + key, "expected": a.get(key), "actual": b.get(key),
                                     "reason": "SCHEMA_CHANGED"})
                else:
                    walk(a[key], b[key], path + key + ".")
        elif a != b:
            failures.append({"field": path.rstrip("."), "expected": a, "actual": b, "reason": "VALUE_CHANGED"})
    walk(expected, actual, "")
    difference = "SCHEMA_CHANGED" if any(f["reason"] == "SCHEMA_CHANGED" for f in failures) else "VALUE_CHANGED"
    return {"status": "FAIL" if failures else "PASS", "difference": difference if failures else "UNCHANGED",
            "failure_count": len(failures), "failures": failures[:30],
            "expected_result": expected, "actual_result": actual}


def compare_golden(expected: pd.DataFrame, actual: pd.DataFrame, dates, strategies):
    """Every date x strategy exactly once; compare every reference-owned field.

    Reference rank/action columns, when present, are required in actual. No invented
    default values. Nulls remain null. Route sequence is significant.
    """
    keys = ["date", "strategy"]
    failures, rows = [], []
    desired = {(str(d), str(s)) for d in dates for s in strategies}
    schema = [c for c in expected if c not in actual] + [c for c in keys if c not in expected]
    if schema:
        return {"status": "FAIL", "difference": "SCHEMA_CHANGED", "failure_count": len(schema),
                "failures": [{"field": c, "reason": "MISSING_COLUMN"} for c in schema[:30]], "rows": []}
    for label, frame in (("reference", expected), ("actual", actual)):
        seen = set(zip(frame.date.astype(str), frame.strategy.astype(str)))
        for d, s in sorted(desired - seen):
            failures.append({"date": d, "strategy": s, "field": "date/strategy", "reason": f"MISSING_{label.upper()}_ROW"})
        for d, s in sorted(seen - desired):
            failures.append({"date": d, "strategy": s, "field": "date/strategy", "reason": f"UNEXPECTED_{label.upper()}_ROW"})
        for item in frame[frame.duplicated(keys, keep=False)][keys].drop_duplicates().to_dict("records"):
            failures.append({**item, "field": "date/strategy", "reason": f"DUPLICATE_{label.upper()}_ROW"})
    if failures:
        return {"status": "FAIL", "difference": "VALUE_CHANGED", "failure_count": len(failures),
                "failures": failures[:30], "rows": []}
    a, b = expected.set_index(keys), actual.set_index(keys)
    for key in sorted(desired):
        fields = []
        for column in expected.columns.difference(keys, sort=False):
            if not equal(a.loc[key, column], b.loc[key, column]):
                fields.append(column)
                failures.append({"date": key[0], "strategy": key[1], "field": column,
                                 "expected": a.loc[key, column], "actual": b.loc[key, column],
                                 "reason": "VALUE_CHANGED"})
        rows.append({"date": key[0], "strategy": key[1], "equal": not fields, "changed_fields": "|".join(fields)})
    return {"status": "FAIL" if failures else "PASS", "difference": "VALUE_CHANGED" if failures else "UNCHANGED",
            "failure_count": len(failures), "failures": failures[:30], "rows": rows}


def inspect_reference(path, expected_hash):
    try:
        actual = sha256(Path(path))
    except FileNotFoundError:
        return {"status": "BLOCKED", "difference": "MISSING_REFERENCE", "actual": None,
                "reason_codes": ["REFERENCE_UNAVAILABLE"]}
    except OSError as error:
        return {"status": "ENVIRONMENT_ERROR", "difference": "NOT_COMPARABLE", "actual": None,
                "reason_codes": ["FILE_ACCESS_ERROR", str(error)]}
    return {"status": "PASS" if actual == expected_hash else "FAIL",
            "difference": "UNCHANGED" if actual == expected_hash else "VALUE_CHANGED",
            "actual": actual, "reason_codes": [] if actual == expected_hash else ["REFERENCE_HASH_CHANGED"]}


def readiness(results, mode):
    counts = {status: sum(r["status"] == status for r in results) for status in sorted(STATUSES)}
    blocked = any(counts[s] for s in ("FAIL", "BLOCKED", "ENVIRONMENT_ERROR"))
    incomplete = any(counts[s] for s in ("NOT_RUN", "INSUFFICIENT_DATA"))
    frozen = any(r.get("execution_level") == "REUSED_FROZEN_RESULT" for r in results)
    code = "BLOCKED" if blocked else "READY_WITH_LIMITATIONS" if incomplete or mode == "quick" or frozen else "READY"
    return {"code_regression": code, "regression_verdict": "FAIL" if counts["FAIL"] else "BLOCKED" if blocked else "PASS",
            "operational_evidence": "BLOCKED", "seller_actual_application": "BLOCKED",
            "demo": "BLOCKED" if blocked else "READY_WITH_LIMITATIONS", "overall": code,
            "production_action_applied": False, "production_approval": False,
            "reasons": ["STRICT_ACTUAL_CAPS_MISSING", "RETAIL_SALES_EVIDENCE_INSUFFICIENT", "ACTUAL_SELLER_OUTCOMES_ZERO"],
            "status_counts": counts}
