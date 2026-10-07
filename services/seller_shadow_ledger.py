"""Opt-in, local, append-only Seller Loss shadow evaluation ledger.

Reuses simulation_history's database path, without altering its two tables or
schema version. This component owns an independent additive migration. It is
not a training database. No production action is written back to the pipeline.
Call record_pipeline_run explicitly with a stable execution key; normal UI
reruns do not write by default. Outcome snapshots bind to a decision version.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from services import seller_loss_promotion_gate as gate, seller_shadow_outcomes as outcomes
from services.simulation_history import history_db_path, initialize_history_storage

SCHEMA_VERSION = 1
COMPONENT = "seller-shadow-ledger"
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS seller_shadow_schema (component TEXT PRIMARY KEY, version INTEGER NOT NULL)",
    """CREATE TABLE IF NOT EXISTS seller_shadow_runs (
        run_id TEXT PRIMARY KEY, execution_key TEXT NOT NULL, data_signature TEXT NOT NULL,
        decision_at TEXT NOT NULL, simulation_run_id TEXT, recorded_at TEXT NOT NULL,
        UNIQUE(execution_key, data_signature, decision_at))""",
    """CREATE TABLE IF NOT EXISTS seller_shadow_decisions (
        decision_id TEXT NOT NULL, decision_version INTEGER NOT NULL, run_id TEXT NOT NULL,
        route_id TEXT NOT NULL, product_id TEXT NOT NULL, source_store_id TEXT NOT NULL,
        target_store_id TEXT, decision_at TEXT NOT NULL, data_mode TEXT NOT NULL
        CHECK(data_mode IN ('PRODUCTION','TEST','SAMPLE','SCENARIO')),
        payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL, recorded_at TEXT NOT NULL,
        PRIMARY KEY(decision_id,decision_version), UNIQUE(decision_id,payload_hash),
        FOREIGN KEY(run_id) REFERENCES seller_shadow_runs(run_id))""",
    """CREATE TABLE IF NOT EXISTS seller_shadow_outcomes (
        outcome_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL, decision_version INTEGER NOT NULL,
        outcome_version INTEGER NOT NULL, data_mode TEXT NOT NULL
        CHECK(data_mode IN ('PRODUCTION','TEST','SAMPLE','SCENARIO')),
        payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL, recorded_at TEXT NOT NULL,
        UNIQUE(decision_id,decision_version,outcome_version),
        UNIQUE(decision_id,decision_version,payload_hash),
        FOREIGN KEY(decision_id,decision_version) REFERENCES seller_shadow_decisions(decision_id,decision_version))""",
    "CREATE INDEX IF NOT EXISTS idx_seller_shadow_modes ON seller_shadow_decisions(data_mode,run_id)",
)


class LedgerError(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@contextmanager
def _connect(directory=None):
    connection = sqlite3.connect(str(history_db_path(directory)), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_ledger(directory=None):
    path = initialize_history_storage(directory)
    with _connect(directory) as db:
        db.execute(_SCHEMA[0])
        row = db.execute("SELECT version FROM seller_shadow_schema WHERE component=?", (COMPONENT,)).fetchone()
        if row and row[0] > SCHEMA_VERSION:
            raise LedgerError("UNSUPPORTED_FUTURE_LEDGER_SCHEMA")
        for statement in _SCHEMA[1:]:
            db.execute(statement)
        db.execute("INSERT INTO seller_shadow_schema VALUES (?,?) ON CONFLICT(component) DO UPDATE SET version=excluded.version",
                   (COMPONENT, SCHEMA_VERSION))
    return path


def build_run_id(data_signature, execution_key, decision_at):
    date = outcomes.timestamp(decision_at)
    if not data_signature or not execution_key or not date:
        raise LedgerError("RUN_REQUIRES_SIGNATURE_EXECUTION_KEY_AND_DATE")
    return "SR-" + _hash({"data_signature": str(data_signature), "execution_key": str(execution_key), "decision_at": date})


def build_decision_id(run_id, *, decision_at, product_id, source_store_id, target_store_id, route_id):
    if not all((run_id, outcomes.timestamp(decision_at), product_id, source_store_id, route_id)):
        raise LedgerError("DECISION_BUSINESS_KEYS_MISSING")
    return "SD-" + _hash({"run_id": run_id, "decision_at": outcomes.timestamp(decision_at),
        "product_id": str(product_id), "source_store_id": str(source_store_id),
        "target_store_id": str(target_store_id) if target_store_id is not None else None, "route_id": str(route_id)})


def classify_data_mode(decision, context=None):
    context = dict(context or {})
    requested = str(context.get("data_mode") or "PRODUCTION").upper()
    if requested not in outcomes.DATA_MODES:
        raise LedgerError("INVALID_DATA_MODE")
    if requested == "SCENARIO" or decision.get("decision_mode") == "SCENARIO" or decision.get("scenario_inputs") or decision.get("evidence_level") == "SCENARIO":
        return "SCENARIO"
    labels = _json({"context":context, "provenance":decision.get("input_provenance", {}),
        "audit":decision.get("seller_input_audit", []), "identifiers":[decision.get(k) for k in (
            "decision_id", "product_id", "source_store_id", "target_store_id")]})
    if requested == "SAMPLE" or re.search(r"(?i)\bsamples?\b", labels.replace("_", " ")) or context.get("is_sample") is True:
        return "SAMPLE"
    if requested == "TEST" or gate._nonproduction(decision, context):
        return "TEST"
    return "PRODUCTION"


_SELLER_FIELDS = (
    "seller_loss_action", "comparison_status", "recommendation_readiness", "evidence_level",
    "recommended_strategy", "expected_loss_transfer", "expected_loss_normal_sale", "expected_loss_discount_sale",
    "loss_difference_vs_second_best", "currency", "quantity_unit", "real_input_fields", "user_input_fields", "seller_input_fields",
    "missing_required_fields", "missing_for_excluded_strategies", "unknown_optional_fields", "conflicting_fields",
    "input_conflicts", "reason_codes", "decision_mode",
)
_PROVENANCE_FIELDS = ("value", "provenance", "source", "dataset", "unit", "currency")
_SOURCE_FIELDS = ("origin", "provenance", "input_scope", "profile_source", "effective_date", "effective_until")
_GATE_FIELDS = ("promotion_status", "promotion_reason_codes", "promotion_blockers", "promotion_candidate",
    "policy_version", "policy_signature", "requested_strategies", "excluded_strategies", "comparison_scope",
    "promotion_comparison_status", "explicit_scope", "disagreement_reason_codes")
_RECORD_FIELDS = {*_SELLER_FIELDS, *_GATE_FIELDS, "run_id", "decision_id", "decision_at", "route_id",
    "engine_decision_id", "product_id", "source_store_id", "target_store_id", "data_mode", "legacy_action",
    "production_action", "production_action_applied", "varo_final_rank", "legacy_strategy", "legacy_service_qty",
    "legacy_cost", "legacy_reason", "legacy_result_status", "input_provenance", "input_sources", "legacy_seller_agreement"}


def decision_record(run_id, decision_at, decision, recommendation, *, shadow=None, context=None):
    """Whitelist business facts. Never persist a whole workbook or personal fields."""
    current = recommendation.get("varo_action") or decision.get("legacy_action")
    if current != decision.get("legacy_action"):
        raise LedgerError("PRODUCTION_LEGACY_ACTION_MISMATCH")
    mode = classify_data_mode(decision, context)
    if shadow is None:
        shadow = gate.shadow_decision(decision, context={**(context or {}), "decision_date": str(decision_at)[:10]})
    record = {k: decision.get(k) for k in _SELLER_FIELDS}
    record.update({k: shadow.get(k) for k in _GATE_FIELDS})
    record.update({"run_id": run_id, "decision_at": outcomes.timestamp(decision_at),
        "route_id": str(recommendation.get("route_id") or decision.get("decision_id") or ""),
        "engine_decision_id": decision.get("decision_id"), "product_id": decision.get("product_id"),
        "source_store_id": decision.get("source_store_id"), "target_store_id": decision.get("target_store_id"),
        "data_mode": mode, "legacy_action": current, "production_action": current, "production_action_applied": False,
        "varo_final_rank": recommendation.get("varo_final_rank"), "legacy_strategy": recommendation.get("varo_final_decision"),
        "legacy_service_qty": recommendation.get("recommended_qty"), "legacy_cost": recommendation.get("move_cost"),
        "legacy_reason": recommendation.get("final_reason") or recommendation.get("reason"),
        "legacy_result_status": recommendation.get("status"),
        "input_provenance": {n:{k:item.get(k) for k in _PROVENANCE_FIELDS} for n,item in decision.get("input_provenance", {}).items()},
        "input_sources": {n:{k:item.get(k) for k in _SOURCE_FIELDS if k in item} for n,item in decision.get("input_sources", {}).items()},
        "legacy_seller_agreement": {"SAME":"AGREE", "DIFFERENT":"DISAGREE"}.get(
            shadow.get("legacy_vs_seller_agreement"), shadow.get("legacy_vs_seller_agreement", "NO_RECOMMENDATION"))})
    record["decision_id"] = build_decision_id(run_id, **{k:record[k] for k in (
        "decision_at", "product_id", "source_store_id", "target_store_id", "route_id")})
    return _plain(record)


def save_run(*, data_signature, execution_key, decision_at, simulation_run_id=None, directory=None):
    initialize_ledger(directory)
    run_id = build_run_id(data_signature, execution_key, decision_at)
    with _connect(directory) as db:
        if simulation_run_id and not db.execute("SELECT 1 FROM simulation_runs WHERE run_id=?", (simulation_run_id,)).fetchone():
            raise LedgerError("SIMULATION_RUN_LINK_NOT_FOUND")
        existing = db.execute("SELECT simulation_run_id FROM seller_shadow_runs WHERE run_id=?", (run_id,)).fetchone()
        if existing and existing[0] != simulation_run_id:
            raise LedgerError("RUN_LINEAGE_CONFLICT")
        db.execute("INSERT OR IGNORE INTO seller_shadow_runs VALUES (?,?,?,?,?,?)", (
            run_id, str(execution_key), str(data_signature), outcomes.timestamp(decision_at), simulation_run_id, _now()))
    return run_id


def save_decision(record, directory=None):
    initialize_ledger(directory)
    if record.get("production_action") != record.get("legacy_action") or record.get("production_action_applied") is not False:
        raise LedgerError("PRODUCTION_ACTION_MUST_REMAIN_LEGACY")
    if record.get("data_mode") not in outcomes.DATA_MODES:
        raise LedgerError("INVALID_DATA_MODE")
    record = dict(record)
    record["data_mode"] = classify_data_mode(record, {"data_mode":record["data_mode"]})
    record = {k:v for k,v in record.items() if k in _RECORD_FIELDS}
    record["input_provenance"] = {n:{k:v for k,v in item.items() if k in _PROVENANCE_FIELDS}
        for n,item in record.get("input_provenance", {}).items()}
    record["input_sources"] = {n:{k:v for k,v in item.items() if k in _SOURCE_FIELDS}
        for n,item in record.get("input_sources", {}).items()}
    payload = _json(record)
    digest = _hash(record)
    decision_id = record["decision_id"]
    if decision_id != build_decision_id(record["run_id"], **{k:record[k] for k in (
        "decision_at", "product_id", "source_store_id", "target_store_id", "route_id")}):
        raise LedgerError("DECISION_ID_LINEAGE_MISMATCH")
    with _connect(directory) as db:
        original_mode = db.execute("SELECT data_mode FROM seller_shadow_decisions WHERE decision_id=? LIMIT 1", (decision_id,)).fetchone()
        if original_mode and original_mode[0] != record["data_mode"]:
            raise LedgerError("DATA_MODE_LINEAGE_CONFLICT")
        existing = db.execute("SELECT decision_version FROM seller_shadow_decisions WHERE decision_id=? AND payload_hash=?", (decision_id,digest)).fetchone()
        if existing:
            return {"decision_id":decision_id, "decision_version":existing[0], "inserted":False}
        version = db.execute("SELECT COALESCE(MAX(decision_version),0)+1 FROM seller_shadow_decisions WHERE decision_id=?", (decision_id,)).fetchone()[0]
        db.execute("INSERT INTO seller_shadow_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            decision_id, version, record["run_id"], record["route_id"], record["product_id"], record["source_store_id"],
            record["target_store_id"], record["decision_at"], record["data_mode"], digest, payload, _now()))
    return {"decision_id":decision_id, "decision_version":version, "inserted":True}


def list_decisions(directory=None, *, data_mode=None, include_versions=False):
    initialize_ledger(directory)
    with _connect(directory) as db:
        query = "SELECT d.* FROM seller_shadow_decisions d"
        conditions, params = [], []
        if not include_versions:
            conditions.append("d.decision_version=(SELECT MAX(v.decision_version) FROM seller_shadow_decisions v WHERE v.decision_id=d.decision_id)")
        if data_mode:
            conditions.append("d.data_mode=?")
            params.append(data_mode)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        rows = db.execute(query+" ORDER BY d.decision_at,d.decision_id,d.decision_version", params).fetchall()
    return [{**json.loads(r["payload_json"]), "decision_version":r["decision_version"], "recorded_at":r["recorded_at"]} for r in rows]


def list_outcomes(directory=None, *, include_versions=False, data_mode=None):
    initialize_ledger(directory)
    with _connect(directory) as db:
        conditions, params = [], []
        if data_mode:
            conditions.append("o.data_mode=?")
            params.append(data_mode)
        if not include_versions:
            # Filter mode inside the latest-version search as well: TEST outcome
            # records must neither count nor mask an earlier production snapshot.
            extra = " AND v.data_mode=o.data_mode" if data_mode else ""
            conditions.append("o.outcome_version=(SELECT MAX(v.outcome_version) FROM seller_shadow_outcomes v WHERE v.decision_id=o.decision_id AND v.decision_version=o.decision_version"+extra+")")
        rows = db.execute("SELECT o.* FROM seller_shadow_outcomes o"+(" WHERE "+" AND ".join(conditions) if conditions else "")+
                          " ORDER BY o.decision_id,o.decision_version,o.outcome_version", params).fetchall()
    return [{**json.loads(r["payload_json"]), "outcome_id":r["outcome_id"], "outcome_version":r["outcome_version"]} for r in rows]


def import_outcomes(source, directory=None):
    parsed = outcomes.parse_seller_outcomes(source)
    report = {"status":parsed["status"], "errors":list(parsed["errors"]), "warnings":parsed["warnings"], "saved":0, "duplicates":0}
    if not parsed["rows"]:
        return report
    initialize_ledger(directory)
    with _connect(directory) as db:
        for index,row in enumerate(parsed["rows"],1):
            candidates = db.execute("SELECT * FROM seller_shadow_decisions WHERE decision_id=? ORDER BY decision_version", (row["decision_id"],)).fetchall()
            code = None
            if not candidates:
                code = "DECISION_NOT_FOUND"
            elif row["decision_version"] is None and len(candidates)>1:
                code = "DECISION_VERSION_REQUIRED"
            selected = next((r for r in candidates if r["decision_version"] == row["decision_version"]), None) if row["decision_version"] is not None else candidates[0] if candidates else None
            if candidates and row["decision_version"] is not None and selected is None:
                code = "DECISION_VERSION_NOT_FOUND"
            if code:
                report["errors"].append({"row":index,"code":code})
                continue
            decision = json.loads(selected["payload_json"])
            if (datetime.fromisoformat(row["recorded_at"]).tzinfo is None) != (datetime.fromisoformat(decision["decision_at"]).tzinfo is None):
                report["errors"].append({"row":index,"code":"TIMEZONE_CONVENTION_MISMATCH"})
                continue
            if outcomes._instant(row["recorded_at"]) < outcomes._instant(decision["decision_at"]):
                report["errors"].append({"row":index,"code":"RECORDED_BEFORE_DECISION"})
                continue
            if row["executed_at"] and outcomes._instant(row["executed_at"]) < outcomes._instant(decision["decision_at"]):
                report["errors"].append({"row":index,"code":"EXECUTED_BEFORE_DECISION"})
                continue
            row = dict(row)
            row["decision_version"] = selected["decision_version"]
            mode = classify_data_mode({}, {"data_mode":row["data_mode"], "label":row["outcome_source"]})
            row["data_mode"] = selected["data_mode"] if selected["data_mode"] != "PRODUCTION" else mode
            row["realized_loss"] = outcomes.realized_loss(row)
            if row["realized_loss"]["status"] == "REALIZED_LOSS_CONFLICT":
                report["errors"].append({"row":index,"code":"REALIZED_LOSS_CONFLICT"})
                continue
            digest = _hash(row)
            existing = db.execute("SELECT 1 FROM seller_shadow_outcomes WHERE decision_id=? AND decision_version=? AND payload_hash=?",
                                  (row["decision_id"],row["decision_version"],digest)).fetchone()
            if existing:
                report["duplicates"] += 1
                continue
            latest = db.execute("SELECT recorded_at FROM seller_shadow_outcomes WHERE decision_id=? AND decision_version=? AND data_mode=? ORDER BY outcome_version DESC LIMIT 1",
                (row["decision_id"],row["decision_version"],row["data_mode"])).fetchone()
            if latest and outcomes._instant(row["recorded_at"]) < outcomes._instant(latest[0]):
                report["errors"].append({"row":index,"code":"STALE_OUTCOME_SNAPSHOT"})
                continue
            version = db.execute("SELECT COALESCE(MAX(outcome_version),0)+1 FROM seller_shadow_outcomes WHERE decision_id=? AND decision_version=?",
                                 (row["decision_id"],row["decision_version"])).fetchone()[0]
            outcome_id = "SO-"+_hash({"decision_id":row["decision_id"],"decision_version":row["decision_version"],"payload_hash":digest})
            db.execute("INSERT INTO seller_shadow_outcomes VALUES (?,?,?,?,?,?,?,?)", (
                outcome_id,row["decision_id"],row["decision_version"],version,row["data_mode"],digest,_json(row),row["recorded_at"]))
            report["saved"] += 1
    report["status"] = "PARTIAL" if report["errors"] and report["saved"] else "INVALID" if report["errors"] else "VALID"
    return report


def record_pipeline_run(state, uploaded_data, *, execution_key, data_signature, decision_at,
                        data_mode="PRODUCTION", simulation_run_id=None, directory=None):
    """Explicit opt-in writer; outcome import errors never alter pipeline outputs."""
    analysis = state.get("pipeline_result", {}).get("seller_loss_analysis", {})
    if analysis.get("status") != "parallel_only":
        raise LedgerError("SELLER_LOSS_ANALYSIS_UNAVAILABLE")
    run_id = save_run(data_signature=data_signature, execution_key=execution_key, decision_at=decision_at,
                      simulation_run_id=simulation_run_id, directory=directory)
    decisions = analysis.get("decisions", [])
    shadow = {r["decision_id"]:r for r in analysis.get("shadow_decisions", [])}
    recs = {(str(r.get("route_id")), str(r.get("product_id")), str(r.get("source_id")), str(r.get("target_id"))):r
            for r in state.get("recommendations", [])}
    saved = []
    context = {**{k:uploaded_data[k] for k in ("is_test","is_sample","synthetic_fixture","label","data_kind","source_file","source_path") if k in uploaded_data}, "data_mode":data_mode}
    for d in decisions:
        key = tuple(str(d.get(k)) for k in ("decision_id","product_id","source_store_id","target_store_id"))
        rec = recs.get(key)
        if rec is None:
            raise LedgerError("RECOMMENDATION_LINEAGE_NOT_FOUND")
        record = decision_record(run_id,decision_at,d,rec,shadow=shadow.get(d["decision_id"]),context=context)
        saved.append(save_decision(record,directory))
    imported = import_outcomes(uploaded_data.get(outcomes.SHEET_KEY),directory)
    return {"status":"RECORDED", "run_id":run_id, "decisions":saved, "outcome_import":imported,
            "production_action_applied":False, "database":str(history_db_path(directory))}


def summarize_ledger(directory=None, *, data_mode="PRODUCTION"):
    rows = list_decisions(directory,data_mode=data_mode)
    events = {(r["decision_id"],r["decision_version"]):r for r in list_outcomes(directory,data_mode=data_mode)}
    states, executed, available, pending = Counter(),0,0,0
    for row in rows:
        outcome = events.get((row["decision_id"],row["decision_version"]))
        executed += bool(outcome and outcome["execution_status"] == "EXECUTED")
        has_actual = bool(outcome and outcome["execution_status"] == "EXECUTED" and any(outcome[k] is not None for k in outcomes.ACTUAL_FIELDS))
        available += has_actual
        pending += not has_actual
        states[outcomes.compare_outcome(row,outcome)["status"]] += 1
    all_rows = list_decisions(directory)
    return {"data_mode":data_mode, "total_decisions":len(rows),
        "seller_recommendable":sum(r["recommendation_readiness"]=="RECOMMENDABLE" for r in rows),
        "promotion_eligible":sum(r["promotion_status"]==gate.PROMOTION_ELIGIBLE for r in rows),
        "legacy_seller_agreement":sum(r["legacy_seller_agreement"]=="AGREE" for r in rows),
        "disagreement":sum(r["legacy_seller_agreement"]=="DISAGREE" for r in rows),
        "unmappable":sum(r["legacy_seller_agreement"]=="UNMAPPABLE" for r in rows),
        "executed_decisions":executed,"outcomes_available":available,"outcomes_pending":pending,
        "confirmed_better":0,"confirmed_worse":0,"not_comparable":states["NOT_COMPARABLE"],
        "comparison_status_counts":dict(sorted(states.items())),
        "excluded_decisions":dict(sorted(Counter(r["data_mode"] for r in all_rows if r["data_mode"]!=data_mode).items())),
        "production_action_applied":False,"winner_policy":"No counterfactual outcomes. No absolute strategy winner."}


def list_evaluations(directory=None, *, data_mode="PRODUCTION"):
    """Join snapshots without modifying the original recommendation audit."""
    events = {(r["decision_id"],r["decision_version"]):r for r in list_outcomes(directory,data_mode=data_mode)}
    result = []
    for decision in list_decisions(directory,data_mode=data_mode):
        event = events.get((decision["decision_id"],decision["decision_version"]))
        result.append({"decision":decision,"outcome":event,
            "operator_action":event["operator_action"] if event else None,
            "execution_status":event["execution_status"] if event else "NOT_EXECUTED",
            "comparison":outcomes.compare_outcome(decision,event)})
    return result


def export_ledger(output_dir, directory=None):
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    # Audit exports include all modes and versions explicitly labelled. Production
    # evaluation summary, unlike audit exports, includes production rows only.
    for filename,rows in (("seller_shadow_decisions.csv",list_decisions(directory,include_versions=True)),
                          ("seller_shadow_outcomes.csv",list_outcomes(directory,include_versions=True))):
        columns = sorted({k for r in rows for k in r}) or (["decision_id","data_mode"] if "decisions" in filename else [*outcomes.META_FIELDS,*outcomes.ACTUAL_FIELDS])
        with (output/filename).open("w",encoding="utf-8-sig",newline="") as file:
            writer = csv.DictWriter(file,fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:_json(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    summary = summarize_ledger(directory)
    (output/"seller_shadow_pilot_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


def pilot_readiness(directory=None):
    """Technical readiness, not evidence that Seller Loss outperforms production."""
    from services.seller_loss_engine import ENGINE_VERSION
    from services.seller_business_profile import VERSION as PROFILE_VERSION
    from services.seller_loss_inputs import INPUT_CONTRACT_VERSION
    checks = {"engine_usable":bool(ENGINE_VERSION), "input_contract_usable":bool(INPUT_CONTRACT_VERSION),
        "profile_usable":bool(PROFILE_VERSION),"gate_usable":bool(gate.policy_document()["signature"]),
        "ledger_writable":False,"outcomes_importable":False,"production_action_unchanged":False,"test_sample_isolated":False}
    errors = []
    try:
        initialize_ledger(directory)
        # Verify write permission transactionally, without retaining a fake run.
        with _connect(directory) as db:
            db.execute("UPDATE seller_shadow_schema SET version=version WHERE component=?",(COMPONENT,))
        checks["ledger_writable"] = True
        checks["outcomes_importable"] = outcomes.parse_seller_outcomes([{"decision_id":"READINESS-CHECK",
            "execution_status":"NOT_EXECUTED", "data_mode":"TEST", "recorded_at":_now()}])["status"] == "VALID"
        rows = list_decisions(directory,include_versions=True)
        checks["production_action_unchanged"] = all(r["production_action"]==r["legacy_action"] and r["production_action_applied"] is False for r in rows)
        checks["test_sample_isolated"] = all(r["data_mode"] in outcomes.DATA_MODES for r in rows) and summarize_ledger(directory)["total_decisions"] == len(list_decisions(directory,data_mode="PRODUCTION"))
    except (OSError,sqlite3.Error,LedgerError) as exc:
        errors.append(type(exc).__name__)
    return {"status":"READY_WITH_LIMITATIONS" if all(checks.values()) else "NOT_READY", "checks":checks,"errors":errors,
        "limitations":["No authentic seller outcomes observed yet; validate sources before a real pilot.",
            "Single observed action cannot prove the unexecuted recommendation was better.",
            "Opt-in local backend ledger, no UI workflow or automatic production promotion.",
            "Local backup/retention is operator-managed; there is no automatic purge."],
        "schema_version":SCHEMA_VERSION,"outcome_contract":outcomes.contract_document(),"production_action_applied":False}
