"""Local, append-only history for ordinary Varo simulation runs.

Only two record kinds are stored, both derived from values the current pipeline
really produces:

1. one run summary row per executed simulation, and
2. the routes the final Varo ranking actually selected for that run.

Intermediate scoring frames, whole candidate sets, feature matrices, replay
buffers, plots, and raw workbook copies are never written here, so one run stays
in the kilobyte range.  Storage is a single local SQLite file under the local
output root; nothing is sent anywhere and the directory is git-ignored.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from services.local_paths import simulation_history_dir

HISTORY_SCHEMA_VERSION = "simulation-history-2026-09-18.1"
DATABASE_FILENAME = "varo_simulation_history.sqlite3"

STATUS_COMPLETED = "완료"
STATUS_FAILED = "실패"

_DATE_COLUMNS = (
    "snapshot_date", "base_date", "as_of_date", "as_of", "reference_date",
    "기준일자", "기준일", "스냅샷일자",
)

_RUN_TABLE = """
CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id TEXT PRIMARY KEY,
    run_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error_summary TEXT,
    data_source_type TEXT,
    data_name TEXT,
    data_signature TEXT,
    snapshot_label TEXT,
    filters_json TEXT,
    node_scope_json TEXT,
    product_scope_json TEXT,
    candidate_count INTEGER,
    scoped_candidate_count INTEGER,
    requested_route_count INTEGER,
    feasible_candidate_count INTEGER,
    selected_count INTEGER,
    skipped_count INTEGER,
    total_moved_quantity REAL,
    total_move_cost REAL,
    total_expected_saving REAL,
    excess_reduction REAL,
    shortage_reduction REAL,
    service_fill_rate REAL,
    final_strategy TEXT,
    strategy_basis TEXT,
    calculation_version TEXT,
    schema_version TEXT
)
"""

_ROUTE_TABLE = """
CREATE TABLE IF NOT EXISTS simulation_selected_routes (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    route_id TEXT,
    recommendation_id TEXT,
    product_id TEXT,
    product_name TEXT,
    source_id TEXT,
    target_id TEXT,
    dc_id TEXT,
    route_type TEXT,
    action TEXT,
    recommended_qty REAL,
    applied_qty REAL,
    move_cost REAL,
    expected_saving REAL,
    varo_final_rank REAL,
    vhs_rank REAL,
    greedy_rank REAL,
    pareto_rank REAL,
    executed INTEGER,
    skipped_reason TEXT,
    PRIMARY KEY (run_id, sequence)
)
"""

_ROUTE_FIELDS = (
    "route_id", "recommendation_id", "product_id", "product_name", "source_id",
    "target_id", "dc_id", "route_type", "action", "recommended_qty", "applied_qty",
    "move_cost", "expected_saving", "varo_final_rank", "vhs_rank", "greedy_rank",
    "pareto_rank", "executed", "skipped_reason",
)

_SUMMARY_FIELDS = (
    "run_key", "created_at", "status", "error_summary", "data_source_type", "data_name",
    "data_signature", "snapshot_label", "filters_json", "node_scope_json",
    "product_scope_json", "candidate_count", "scoped_candidate_count",
    "requested_route_count", "feasible_candidate_count", "selected_count",
    "skipped_count", "total_moved_quantity", "total_move_cost", "total_expected_saving",
    "excess_reduction", "shortage_reduction", "service_fill_rate", "final_strategy",
    "strategy_basis", "calculation_version", "schema_version",
)


def history_db_path(directory: str | Path | None = None) -> Path:
    base = Path(directory) if directory is not None else simulation_history_dir()
    return base / DATABASE_FILENAME


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@contextlib.contextmanager
def _connect(db_path: Path):
    """Open, commit, and always close so the file is never left locked."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_history_storage(directory: str | Path | None = None) -> Path:
    """Create the local SQLite file and both tables when missing."""
    db_path = history_db_path(directory)
    with _connect(db_path) as connection:
        connection.execute(_RUN_TABLE)
        connection.execute(_ROUTE_TABLE)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_simulation_runs_created_at "
            "ON simulation_runs (created_at DESC)"
        )
    return db_path


def build_run_key(
    data_signature: str | None,
    run_nonce: Any,
    routes: Sequence[Mapping[str, Any]],
    display_mode: str | None = None,
) -> str:
    """Identify one real user execution so Streamlit reruns never duplicate it."""
    payload = json.dumps(
        {
            "schema": HISTORY_SCHEMA_VERSION,
            "data_signature": str(data_signature or "unknown"),
            "run_nonce": str(run_nonce),
            "display_mode": str(display_mode or ""),
            "routes": [str(item.get("route_id") or "") for item in routes or []],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_snapshot_label(data: Mapping[str, Any] | None) -> str | None:
    """Return a real snapshot date or period, never a fabricated one."""
    for frame in (data or {}).values():
        columns = getattr(frame, "columns", None)
        if columns is None:
            continue
        for column in _DATE_COLUMNS:
            if column not in columns:
                continue
            values = sorted({
                str(item).strip()
                for item in frame[column].tolist()
                if str(item).strip() and str(item).strip().lower() != "nan"
            })
            if not values:
                continue
            return values[0] if len(values) == 1 else f"{values[0]} ~ {values[-1]}"
    return None


def _scope(items: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in items if value not in (None, "")})


def build_run_records(
    *,
    data_signature: str | None,
    data_source_type: str | None,
    data_name: str | None,
    snapshot_label: str | None,
    filters: Mapping[str, Any] | None,
    candidate_count: int,
    scoped_candidate_count: int,
    simulation_routes: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any] | None,
    pipeline_result: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one compact summary row plus the final selected route rows."""
    scenario = dict(scenario or {})
    kpis = dict(scenario.get("kpis") or {})
    transitions = list(scenario.get("transitions") or [])
    by_route = {str(item.get("route_id") or ""): item for item in transitions}

    routes: list[dict[str, Any]] = []
    executed_saving = 0.0
    executed_cost = 0.0
    before_shortage = 0.0
    for index, item in enumerate(simulation_routes or [], start=1):
        route_id = str(item.get("route_id") or "")
        transition = dict(by_route.get(route_id) or {})
        executed = bool(transition.get("feasible")) if transition else False
        applied = _number(transition.get("applied_quantity"))
        cost = _number(item.get("move_cost")) or _number(item.get("estimated_cost"))
        saving = _number(item.get("expected_saving"))
        if executed:
            executed_cost += cost or 0.0
            executed_saving += saving or 0.0
            before_shortage += _number((transition.get("target") or {}).get("before_shortage")) or 0.0
        routes.append({
            "route_id": route_id or None,
            "recommendation_id": _text(item.get("recommendation_id")),
            "product_id": _text(item.get("product_id")),
            "product_name": _text(item.get("product_name")),
            "source_id": _text(item.get("source_id")),
            "target_id": _text(item.get("target_id")),
            "dc_id": _text(item.get("dc_id")),
            "route_type": _text(item.get("route_type")),
            "action": _text(item.get("varo_action") or item.get("final_recommendation")),
            "recommended_qty": _number(item.get("recommended_qty")),
            "applied_qty": applied,
            "move_cost": cost,
            "expected_saving": saving,
            "varo_final_rank": _number(item.get("varo_final_rank") or item.get("rank")),
            "vhs_rank": _number(item.get("vhs_rank")),
            "greedy_rank": _number(item.get("greedy_rank")),
            "pareto_rank": _number(item.get("pareto_rank")),
            "executed": 1 if executed else 0,
            "skipped_reason": None if executed else _text(transition.get("skipped_reason")),
        })

    executed_rows = [row for row in routes if row["executed"]]
    actions = [row["action"] for row in executed_rows if row["action"]]
    dominant_action = max(set(actions), key=actions.count) if actions else None
    shortage_reduction = _number(kpis.get("shortage_reduction")) or 0.0
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": STATUS_COMPLETED,
        "error_summary": None,
        "data_source_type": _text(data_source_type),
        "data_name": _text(data_name),
        "data_signature": _text(data_signature),
        "snapshot_label": _text(snapshot_label),
        "filters_json": json.dumps(dict(filters or {}), ensure_ascii=False, sort_keys=True),
        "node_scope_json": json.dumps(
            _scope(
                value
                for item in simulation_routes or []
                for value in (
                    item.get("source_name") or item.get("source_id"),
                    item.get("target_name") or item.get("target_id"),
                    item.get("dc_name") or item.get("dc_id"),
                )
            ),
            ensure_ascii=False,
        ),
        "product_scope_json": json.dumps(
            _scope(
                item.get("product_name") or item.get("product_id")
                for item in simulation_routes or []
            ),
            ensure_ascii=False,
        ),
        "candidate_count": int(candidate_count),
        "scoped_candidate_count": int(scoped_candidate_count),
        "requested_route_count": int(scenario.get("requested_route_count") or len(routes)),
        "feasible_candidate_count": int(scenario.get("executed_route_count") or len(executed_rows)),
        "selected_count": len(executed_rows),
        "skipped_count": int(scenario.get("skipped_route_count") or 0),
        "total_moved_quantity": _number(kpis.get("moved_quantity")),
        "total_move_cost": round(executed_cost, 3),
        "total_expected_saving": (
            _number(kpis.get("expected_saving"))
            if kpis.get("expected_saving") is not None
            else round(executed_saving, 3)
        ),
        "excess_reduction": _number(kpis.get("excess_reduction")),
        "shortage_reduction": _number(kpis.get("shortage_reduction")),
        "service_fill_rate": (
            round(min(1.0, shortage_reduction / before_shortage) * 100.0, 2)
            if before_shortage > 0 else None
        ),
        "final_strategy": dominant_action,
        "strategy_basis": _text((pipeline_result or {}).get("result_basis")),
        "calculation_version": _text(scenario.get("version")),
        "schema_version": HISTORY_SCHEMA_VERSION,
    }
    return summary, routes


def record_simulation_run(
    summary: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]] = (),
    *,
    run_key: str,
    directory: str | Path | None = None,
) -> str | None:
    """Persist one finished run.  Returns None when the run was already stored."""
    db_path = initialize_history_storage(directory)
    run_id = str(uuid.uuid4())
    values = dict(summary)
    values["run_key"] = run_key
    values.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    values.setdefault("status", STATUS_COMPLETED)
    values["schema_version"] = HISTORY_SCHEMA_VERSION
    columns = ", ".join(("run_id", *_SUMMARY_FIELDS))
    placeholders = ", ".join("?" for _ in range(len(_SUMMARY_FIELDS) + 1))
    with _connect(db_path) as connection:
        cursor = connection.execute(
            f"INSERT OR IGNORE INTO simulation_runs ({columns}) VALUES ({placeholders})",
            (run_id, *(values.get(field) for field in _SUMMARY_FIELDS)),
        )
        if cursor.rowcount == 0:
            return None
        if routes:
            route_columns = ", ".join(("run_id", "sequence", *_ROUTE_FIELDS))
            route_placeholders = ", ".join("?" for _ in range(len(_ROUTE_FIELDS) + 2))
            connection.executemany(
                f"INSERT OR REPLACE INTO simulation_selected_routes ({route_columns}) "
                f"VALUES ({route_placeholders})",
                [
                    (run_id, index, *(dict(row).get(field) for field in _ROUTE_FIELDS))
                    for index, row in enumerate(routes, start=1)
                ],
            )
    return run_id


def record_failed_run(
    *,
    run_key: str,
    error_summary: str,
    data_signature: str | None = None,
    data_name: str | None = None,
    data_source_type: str | None = None,
    directory: str | Path | None = None,
) -> str | None:
    """Store the minimum needed to see that a run did not finish."""
    return record_simulation_run(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": STATUS_FAILED,
            "error_summary": str(error_summary)[:300],
            "data_signature": _text(data_signature),
            "data_name": _text(data_name),
            "data_source_type": _text(data_source_type),
            "schema_version": HISTORY_SCHEMA_VERSION,
        },
        (),
        run_key=run_key,
        directory=directory,
    )


def list_simulation_runs(
    limit: int = 200, directory: str | Path | None = None,
) -> list[dict[str, Any]]:
    db_path = history_db_path(directory)
    if not db_path.exists():
        return []
    with _connect(db_path) as connection:
        try:
            rows = connection.execute(
                "SELECT * FROM simulation_runs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        except sqlite3.DatabaseError:
            return []
    return [dict(row) for row in rows]


def get_simulation_run(run_id: str, directory: str | Path | None = None) -> dict[str, Any] | None:
    db_path = history_db_path(directory)
    if not db_path.exists():
        return None
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM simulation_runs WHERE run_id = ?", (str(run_id),)
        ).fetchone()
    return dict(row) if row else None


def get_run_routes(run_id: str, directory: str | Path | None = None) -> list[dict[str, Any]]:
    db_path = history_db_path(directory)
    if not db_path.exists():
        return []
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM simulation_selected_routes WHERE run_id = ? ORDER BY sequence",
            (str(run_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def history_storage_info(directory: str | Path | None = None) -> dict[str, Any]:
    """Expose real storage usage so the history stays provably small."""
    db_path = history_db_path(directory)
    size = db_path.stat().st_size if db_path.exists() else 0
    runs = route_rows = 0
    if db_path.exists():
        with _connect(db_path) as connection:
            try:
                runs = int(connection.execute("SELECT COUNT(*) FROM simulation_runs").fetchone()[0])
                route_rows = int(
                    connection.execute("SELECT COUNT(*) FROM simulation_selected_routes").fetchone()[0]
                )
            except sqlite3.DatabaseError:
                runs = route_rows = 0
    return {
        "path": str(db_path),
        "exists": db_path.exists(),
        "size_bytes": size,
        "size_kb": round(size / 1024.0, 2),
        "run_count": runs,
        "route_row_count": route_rows,
        "schema_version": HISTORY_SCHEMA_VERSION,
    }
