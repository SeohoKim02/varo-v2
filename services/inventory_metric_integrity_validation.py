"""Sales semantics, inventory turnover and inventory risk metric integrity on real and repository data (P1 / T4).

Validation only. It reads the canonical datasets, the Suhyup E2E upload and per-day flow, the 16 repository workbooks,
the saved Suhyup candidates and the earlier validation bundles, and never writes there. It runs the unchanged
production pipeline (``build_v2_state``) and reads the new parallel field ``inventory_metric_integrity``; legacy
analyzers, thresholds, recommendations and actions are untouched.

Sections:
* field inventory (every sales / demand / stock / turnover field) and the loader alias replay checked against the
  loader itself;
* Suhyup: the 2026-07-31 E2E upload, the 31 per-day uploads, and the 31-day flow-based turnover proxy built from the
  canonical inventory snapshot / flow (outbound is never called sales);
* external sales datasets (M5, Favorita, FreshRetailNet) streamed from the canonical demand_series: zero / missing /
  return / window completeness; no inventory exists, so no real turnover is produced;
* domestic datasets (LogisAll, Jangbogo, NFQS, AI Hub, KAMP) coverage;
* repository workbooks: legacy vs validated grades, ABC basis, disposal inputs, action dependency;
* controlled scenarios A-AF;
* regressions (Suhyup 186 reference rows, T1 totals re-run, earlier bundle figures) and protection (no data or bundle
  file changed, tracked changes limited to services/analysis_pipeline.py).

Outputs go to <data-root>/_INVENTORY_METRIC_INTEGRITY (local only, never into git); a re-run replaces only OUTPUT_FILES.
"""
from __future__ import annotations

from services.git_protection import inspect_validator_git, git_validation_check

import argparse
import copy
import json
import math
import re
import time
import warnings
from collections import Counter
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from services import inventory_metric_integrity as imi
from services import operational_evidence_validation as oev
from services import target_need_contract_validation as tnv
from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import _env

OUTPUT_FOLDER = "_INVENTORY_METRIC_INTEGRITY"
OUTPUT_FILES = (
    "sales_semantics_inventory.csv",
    "sales_signal_validation.csv",
    "turnover_formula_audit.csv",
    "turnover_missing_zero_analysis.csv",
    "turnover_by_dataset.csv",
    "abc_semantics_audit.csv",
    "disposal_risk_input_audit.csv",
    "inventory_metric_coverage_matrix.csv",
    "inventory_metric_integrity_summary.json",
    "loader_alias_paths.csv",
    "controlled_scenarios.csv",
    "legacy_vs_validated_comparison.csv",
    "action_dependency.csv",
    "downstream_impact.csv",
    "suhyup_flow_turnover_proxy.csv",
    "inventory_metric_rows.parquet",
    "suhyup_alias_action_shift.csv",
)
REAL_DATA_ROOT_ENV = "VARO_REAL_DATA_ROOT"
REPO_FOLDERS = ("data", "samples", "Varo_DQN_training_samples_10pack")
SUHYUP = "02_Korea_Suhyup_Logistics/processed"
CANONICAL = {
    "m5": "20_WALMART_M5/processed/canonical_demand_series.parquet",
    "favorita": "27_FAVORITA/processed/canonical_demand_series.parquet",
    "freshretailnet": "05_FreshRetailNet/processed/canonical_demand_series.parquet",
    "logisall_sales": "22_KADX_LOGISALL/processed/canonical_demand_series.parquet",
    "logisall_stock": "22_KADX_LOGISALL/processed/canonical_inventory_snapshot.parquet",
    "jangbogo_sales": "23_KADX_JANGBOGO/processed/canonical_demand_series.parquet",
    "jangbogo_flow": "23_KADX_JANGBOGO/processed/canonical_inventory_flow.parquet",
    "nfqs_stock": "29_DOMESTIC_ADDITIONAL/NFQS_FISH_INVENTORY/processed/canonical_inventory_snapshot.parquet",
    "aihub_stock": "26_AIHUB_INDUSTRIAL_LOGISTICS/processed/canonical_inventory_snapshot.parquet",
    "aihub_flow": "26_AIHUB_INDUSTRIAL_LOGISTICS/processed/canonical_inventory_flow.parquet",
    "kamp_flow": "30_KAMP_REBAR_SUPPLY_CHAIN/processed/canonical_inventory_flow.parquet",
}
PROTECTED_BUNDLES = (*tnv.PROTECTED_BUNDLES, tnv.OUTPUT_FOLDER, "_OPERATIONAL_EVIDENCE_VALIDATION",
                     "05_FreshRetailNet/results", "27_FAVORITA/results", "05_FreshRetailNet/processed",
                     "20_WALMART_M5/processed", "27_FAVORITA/processed", "22_KADX_LOGISALL/processed",
                     "23_KADX_JANGBOGO/processed", "29_DOMESTIC_ADDITIONAL/NFQS_FISH_INVENTORY/processed",
                     "26_AIHUB_INDUSTRIAL_LOGISTICS/processed", "30_KAMP_REBAR_SUPPLY_CHAIN/processed")
ALLOWED_MODIFIED = ("services/analysis_pipeline.py",)
SUHYUP_PERIOD = ("2026-07-01", "2026-07-31")
SALES_COLUMNS_FOR_REPLAY = ("avg_daily_sales", "sales_qty", "sales_30d", "sales_30", "sales_7d", "sales")
WINDOW_DAYS = 30


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return oev._check(check_id, description, ok, detail)


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "INF"
    return str(value)


def _codes(values: Sequence[str] | None) -> str:
    return "|".join(values or [])


# ------------------------------------------------------------------------------------------------ field inventory (§2)


def field_inventory(repo_root: Path) -> pd.DataFrame:
    """FIELD_INVENTORY with the modules that reference each field (re-read from the code on every run)."""
    sources = {str(p.relative_to(repo_root)).replace("\\", "/"): p.read_text(encoding="utf-8", errors="ignore")
               for p in sorted((repo_root / "services").rglob("*.py")) if "__pycache__" not in p.parts}
    rows = []
    for item in imi.FIELD_INVENTORY:
        record = dict(zip(imi.FIELD_INVENTORY_COLUMNS, item))
        names = [n.strip() for n in record["field"].replace(" / ", "/").split("/")]
        hits = sorted({path for path, text in sources.items() for n in names
                       if n and re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", text)
                       and path not in ("services/inventory_metric_integrity.py",
                                        "services/inventory_metric_integrity_validation.py")})
        record.update({"referencing_modules": "|".join(hits), "referencing_module_count": len(hits)})
        rows.append(record)
    return pd.DataFrame(rows)


def loader_alias_paths() -> pd.DataFrame:
    """Every subset of raw sales columns through normalize_loaded_data vs the contract's replay of the alias chain."""
    from services.data_loader import normalize_loaded_data

    rows = []
    for size in range(1, len(SALES_COLUMNS_FOR_REPLAY) + 1):
        for subset in combinations(SALES_COLUMNS_FOR_REPLAY, size):
            values = {column: [float(101 + i)] for i, column in enumerate(subset)}
            frame = pd.DataFrame({"store_id": ["S1"], "product_id": ["P1"], "stock_qty": [10.0], **values})
            loaded = normalize_loaded_data({"inventory": frame})["inventory"]
            replay = imi.resolve_loader_aliases(subset)
            for standard, item in replay.items():
                actual_value = float(pd.to_numeric(loaded[standard], errors="coerce").iloc[0]) if standard in loaded else None
                expected_value = values[item["raw_source_column"]][0] if item["raw_source_column"] in values else None
                rows.append({"raw_columns": "|".join(subset), "standard_field": standard,
                             "raw_source_column": item["raw_source_column"], "alias_stage": item["alias_stage"],
                             "field_period": f"{item['field_period'][0]}:{item['field_period'][1]}",
                             "source_period": f"{item['source_period'][0]}:{item['source_period'][1]}",
                             "period_conflict": item["period_conflict"],
                             "replay_matches_loader": actual_value == expected_value})
            missing = [c for c in loaded.columns if c in imi.SALES_FIELDS and c not in replay]
            for column in missing:
                rows.append({"raw_columns": "|".join(subset), "standard_field": column, "raw_source_column": None,
                             "alias_stage": "UNPREDICTED", "field_period": None, "source_period": None,
                             "period_conflict": None, "replay_matches_loader": False})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ pipeline runs


def _pipeline(data: Mapping[str, Any], data_root: Path | None) -> dict[str, Any]:
    from services.analysis_pipeline import build_v2_state

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if data_root is None:
            return build_v2_state(copy.deepcopy(dict(data)), detail_level="core")
        with _env(REAL_DATA_ROOT_ENV, str(data_root)):
            return build_v2_state(copy.deepcopy(dict(data)), detail_level="core")


def _rows_frame(field: Mapping[str, Any], scope: str, dataset: str, snapshot: str | None) -> pd.DataFrame:
    rows = []
    for row in field.get("rows", []):
        flat = {k: (_codes(v) if isinstance(v, list) else json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
                for k, v in row.items()}
        rows.append({"scope": scope, "dataset": dataset, "snapshot_date": snapshot, **flat})
    return pd.DataFrame(rows)


def run_workbooks(repo_root: Path) -> dict[str, Any]:
    from services.data_loader import load_excel_data

    frames, actions, abc_rows, summaries = [], [], [], []
    for folder in REPO_FOLDERS:
        for path in sorted((repo_root / folder).glob("*.xlsx")):
            name = f"{folder}/{path.name}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load_excel_data(path)
            state = _pipeline(data, None)
            field = state["pipeline_result"]["inventory_metric_integrity"]
            frames.append(_rows_frame(field, "WORKBOOK", name, None))
            for action in field.get("action_dependency", {}).get("rows", []):
                actions.append({"scope": "WORKBOOK", "dataset": name, "snapshot_date": None, **action})
            abc = field.get("abc", {})
            abc_rows.append({"scope": "WORKBOOK", "dataset": name, **{k: (json.dumps(v) if isinstance(v, (dict, list, tuple)) else v)
                                                                       for k, v in abc.items() if k != "validated_grades"},
                             "validated_products": len(abc.get("validated_grades", {}))})
            summaries.append({"dataset": name, "status": field.get("status"), **{f"count_{k}": v for k, v in field.get("counts", {}).items()
                                                                                if not isinstance(v, dict)}})
    return {"rows": pd.concat(frames, ignore_index=True), "actions": pd.DataFrame(actions), "abc": pd.DataFrame(abc_rows),
            "summaries": pd.DataFrame(summaries)}


# ------------------------------------------------------------------------------------------------ Suhyup (§12, 34)


def suhyup_flow_proxy(data_root: Path) -> dict[str, Any]:
    """31-day flow-based turnover proxy per (center, product, state) from the canonical snapshot and flow tables."""
    folder = data_root / SUHYUP
    snap = pd.read_parquet(folder / "canonical_inventory_snapshot.parquet",
                           columns=["snapshot_date", "location_id", "product_id", "product_state", "inventory_qty",
                                    "inventory_weight_kg", "unit", "unit_status", "validation_flags"])
    flow = pd.read_parquet(folder / "canonical_inventory_flow.parquet",
                           columns=["date", "location_id", "product_id", "product_state", "inbound_qty", "outbound_qty",
                                    "outbound_weight_kg", "unit", "unit_status", "validation_flags", "flow_kind"])
    csv = pd.read_csv(data_root / "02_Korea_Suhyup_Logistics/processed/suhyup_logistics_inventory_flow_actual.csv",
                      dtype={"center_code": str, "product_code": str, "state_code": str})
    conservation = {
        "stock_qty": (float(snap["inventory_qty"].sum()), float(csv["stock_qty"].sum())),
        "outbound_qty": (float(flow["outbound_qty"].sum()), float(csv["outbound_qty"].sum())),
        "rows": (len(snap), len(csv)),
    }
    start, end = SUHYUP_PERIOD
    keys = ["location_id", "product_id", "product_state"]
    flow_groups = {k: g for k, g in flow.groupby(keys, sort=True)}
    rows = []
    for key, s_group in snap.groupby(keys, sort=True):
        f_group = flow_groups.get(key, flow.iloc[0:0])
        out = {}
        for basis, qty_col, stock_col, unit in (("count", "outbound_qty", "inventory_qty", None),
                                                ("kg", "outbound_weight_kg", "inventory_weight_kg", "kg")):
            window = imi.aggregate_daily_window(
                ({"date": d, "quantity": q} for d, q in zip(f_group["date"], f_group[qty_col])),
                window_end=end, window_days=31, as_of=end, dataset_id="suhyup", location_id=key[0], product_id=key[1],
                signal_type=imi.OBSERVED_OUTBOUND_FLOW, quantity_unit=unit, zero_semantics=imi.ZERO_RECORDED,
                semantics_evidence=imi.SOURCE_DOCUMENTED,
                provenance=f"canonical inventory_flow.{qty_col} (출고량, logistics outbound)",
                source_lineage="02_Korea_Suhyup_Logistics raw 입출고현황 -> canonical_inventory_flow")
            inventory = imi.average_inventory(({"date": d, "quantity": q} for d, q in zip(s_group["snapshot_date"], s_group[stock_col])),
                                              period_start=start, period_end=end, as_of=end, dataset_id="suhyup",
                                              quantity_unit=unit)
            turnover = imi.physical_turnover(window, inventory)
            out[basis] = (window, inventory, turnover)
        window, inventory, turnover = out["count"]
        kg_turnover = out["kg"][2]
        dates = sorted(set(s_group["snapshot_date"]) | set(f_group["date"]))
        stock_by = dict(zip(s_group["snapshot_date"], s_group["inventory_qty"]))
        flow_by = {d: (i, o) for d, i, o in zip(f_group["date"], f_group["inbound_qty"], f_group["outbound_qty"])}
        checked = mismatched = 0
        for previous, current in zip(dates, dates[1:]):
            if previous in stock_by and current in stock_by and current in flow_by \
                    and (date.fromisoformat(current) - date.fromisoformat(previous)).days == 1:
                checked += 1
                inbound, outbound = flow_by[current]
                if abs((stock_by[current] - stock_by[previous]) - (inbound - outbound)) > 1e-9:
                    mismatched += 1
        rows.append({
            "location_id": key[0], "center_code": key[0].lstrip("0") or "0", "product_id": key[1], "product_state": key[2],
            "snapshot_days": inventory["snapshot_count"], "flow_days": window["observed_days"],
            "no_record_days": window["no_record_days"], "zero_outbound_days": window["zero_days"],
            "outbound_31d": window["quantity"], "inbound_31d": float(f_group["inbound_qty"].sum()) if len(f_group) else None,
            "average_inventory": inventory["value"], "start_stock": inventory["start_qty"], "end_stock": inventory["end_qty"],
            "inventory_status": inventory["status"], "flow_daily_rate": window["daily_rate"],
            "turnover_value_31d": turnover["turnover_value"], "turnover_annualized_365": turnover["turnover_annualized_365"],
            "turnover_status": turnover["turnover_status"], "turnover_basis": turnover["turnover_basis"],
            "days_of_supply": "INF" if isinstance(turnover["days_of_supply"], float) and math.isinf(turnover["days_of_supply"])
            else turnover["days_of_supply"], "days_of_supply_status": turnover["days_of_supply_status"],
            "proxy_classification": turnover["turnover_classification"],
            "classification_confidence": turnover["classification_confidence"],
            "classification_scope": turnover["classification_scope"], "gate_status": turnover["gate_status"],
            "reason_codes": _codes(turnover["reason_codes"]),
            "kg_turnover_value_31d": kg_turnover["turnover_value"], "kg_turnover_status": kg_turnover["turnover_status"],
            "kg_proxy_classification": kg_turnover["turnover_classification"], "kg_gate_status": kg_turnover["gate_status"],
            "kg_reason_codes": _codes(kg_turnover["reason_codes"]),
            "stock_flow_identity_checked_days": checked, "stock_flow_identity_mismatch_days": mismatched,
        })
    return {"rows": pd.DataFrame(rows), "conservation": conservation}


def suhyup_e2e(data_root: Path) -> dict[str, Any]:
    from services.seller_loss_input_validation import load_suhyup_upload

    upload = load_suhyup_upload(data_root)
    state = _pipeline(upload, data_root)
    field = state["pipeline_result"]["inventory_metric_integrity"]
    recs = pd.DataFrame(state["recommendations"])
    rows = _rows_frame(field, "SUHYUP_E2E_20260731", "suhyup", "2026-07-31")
    rows["source_state_code"] = upload["inventory"]["source_state_code"].astype(str).values if len(rows) == len(upload["inventory"]) else None
    return {"field": field, "rows": rows, "recommendations": recs, "upload": upload}


# The declaration columns of the processed Suhyup E2E inventory (16_VARO_E2E_20260731/processed/inventory.csv).
SUHYUP_DECLARATION = {"stock_qty_provenance": "actual", "sales_qty_provenance": "derived_proxy_from_actual_outbound_qty",
                      "sales_qty_semantics": imi.DAILY_PROXY_SEMANTICS}


def declared_day_upload(rebuild: Any, day: str) -> dict[str, pd.DataFrame]:
    """The per-day upload with the same declaration columns as the processed 07-31 E2E file (the construction that
    reproduces the saved 31-day candidates and their varo_action)."""
    from services.data_loader import normalize_loaded_data

    inventory, _ = rebuild.inventory(day)
    inventory = inventory.assign(**SUHYUP_DECLARATION)
    return normalize_loaded_data({"stores": rebuild.stores.copy(), "products": rebuild.products.copy(),
                                  "inventory": inventory, "routes": rebuild.routes.copy()})


def suhyup_days(data_root: Path) -> dict[str, Any]:
    """The 31 per-day uploads through the production pipeline, declared (as saved) and undeclared (no semantics column).

    The declared run is the saved 2026-07 construction; the undeclared run is what the loader does with the same
    numbers when nothing says sales_qty is one day's outbound (sales_30d then holds the one-day value).
    """
    from services import milp_benchmark_integrity_validation as mbv
    from services import shared_feasibility_validation as sfv

    rebuild = mbv.GeneratorRebuild(data_root)
    saved = sfv.load_suhyup_candidates(data_root)
    frames, actions, fidelity, label_shift = [], [], [], []
    for day in sorted(saved["snapshot_date"].astype(str).unique()):
        state = _pipeline(declared_day_upload(rebuild, day), data_root)
        plain = _pipeline(rebuild.upload(day), data_root)
        field = state["pipeline_result"]["inventory_metric_integrity"]
        frames.append(_rows_frame(field, "SUHYUP_DAY_UPLOAD", "suhyup", day))
        plain_field = plain["pipeline_result"]["inventory_metric_integrity"]
        frames.append(_rows_frame(plain_field, "SUHYUP_DAY_UPLOAD_UNDECLARED", "suhyup", day))
        for action in field.get("action_dependency", {}).get("rows", []):
            actions.append({"scope": "SUHYUP_DAY_UPLOAD", "dataset": "suhyup", "snapshot_date": day, **action})
        recs, plain_recs = pd.DataFrame(state["recommendations"]), pd.DataFrame(plain["recommendations"])
        saved_day = saved[saved["snapshot_date"].astype(str) == day]
        left = dict(zip(recs["route_id"].astype(str), recs["varo_action"].astype(str))) if not recs.empty else {}
        right = dict(zip(saved_day["route_id"].astype(str), saved_day["varo_action"].astype(str)))
        undeclared = dict(zip(plain_recs["route_id"].astype(str), plain_recs["varo_action"].astype(str))) if not plain_recs.empty else {}
        fidelity.append({"date": day, "routes_rebuilt": len(left), "routes_saved": len(right),
                         "same_route_ids": set(left) == set(right), "same_varo_action": left == right,
                         "undeclared_same_route_ids": set(undeclared) == set(right)})
        for route, action in right.items():
            label_shift.append({"date": day, "route_id": route, "saved_varo_action": action,
                                "declared_varo_action": left.get(route), "undeclared_varo_action": undeclared.get(route)})
    return {"rows": pd.concat(frames, ignore_index=True), "actions": pd.DataFrame(actions), "fidelity": pd.DataFrame(fidelity),
            "label_shift": pd.DataFrame(label_shift)}


# ------------------------------------------------------------------------------------------------ external sales (§9-11, 35)


def _date_window(end: str, days: int = WINDOW_DAYS) -> tuple[str, str]:
    stop = date.fromisoformat(end)
    return (stop - timedelta(days=days - 1)).isoformat(), end


def stream_demand(path: Path, *, window_end: str, price: bool = False, stockout: bool = False,
                  combine_every: int = 12) -> dict[str, Any]:
    """Per-series statistics of a canonical demand_series, streamed by row group (no zero filling, no clipping)."""
    pf = pq.ParquetFile(path)
    columns = ["date", "location_id", "product_id", "sales_qty", "validation_flags", "promotion"]
    if price:
        columns.append("price")
    if stockout:
        columns += ["stockout_hours", "stockout_window_hours", "discount_rate"]
    start, end = _date_window(window_end)
    totals: Counter[str] = Counter()
    partials: list[pa.Table] = []
    accumulated: pa.Table | None = None
    keys = ["location_id", "product_id"]
    sum_cols = ["rows", "qty_sum", "zero_rows", "neg_rows", "nonint_rows", "win_rows", "win_sum", "win_neg", "win_zero",
                "win_revenue", "win_price_rows"]

    for index in range(pf.num_row_groups):
        t = pf.read_row_group(index, columns=columns)
        q = t["sales_qty"]
        flags = pc.fill_null(t["validation_flags"], "")
        in_window = pc.and_(pc.greater_equal(t["date"], start), pc.less_equal(t["date"], end))
        valid = pc.is_valid(q)
        totals["rows"] += t.num_rows
        totals["null_qty"] += int(pc.sum(pc.cast(pc.invert(valid), pa.int64())).as_py() or 0)
        totals["zero"] += int(pc.sum(pc.cast(pc.equal(q, 0), pa.int64())).as_py() or 0)
        totals["negative"] += int(pc.sum(pc.cast(pc.less(q, 0), pa.int64())).as_py() or 0)
        totals["negative_sum"] += float(pc.sum(pc.if_else(pc.less(q, 0), q, 0.0)).as_py() or 0.0)
        totals["qty_sum"] += float(pc.sum(q).as_py() or 0.0)
        totals["documented_return"] += int(pc.sum(pc.cast(pc.match_substring(flags, "documented_return"), pa.int64())).as_py() or 0)
        totals["duplicate_flag"] += int(pc.sum(pc.cast(pc.or_(pc.match_substring(flags, "duplicate_key"),
                                                               pc.match_substring(flags, "adjacent_identical")), pa.int64())).as_py() or 0)
        totals["promotion_null"] += int(pc.sum(pc.cast(pc.is_null(t["promotion"]), pa.int64())).as_py() or 0)
        totals["promotion_not_reported_flag"] += int(pc.sum(pc.cast(pc.match_substring(flags, "promotion_not_reported"), pa.int64())).as_py() or 0)
        totals["nonint"] += int(pc.sum(pc.cast(pc.not_equal(q, pc.floor(q)), pa.int64())).as_py() or 0)
        if price:
            totals["price_null"] += int(pc.sum(pc.cast(pc.is_null(t["price"]), pa.int64())).as_py() or 0)
            totals["weekly_price_absent_flag"] += int(pc.sum(pc.cast(pc.match_substring(flags, "weekly_price_absent"), pa.int64())).as_py() or 0)
            totals["sales_without_weekly_price_flag"] += int(pc.sum(pc.cast(pc.match_substring(flags, "sales_without_weekly_price"), pa.int64())).as_py() or 0)
        if stockout:
            hours, span = t["stockout_hours"], t["stockout_window_hours"]
            zero = pc.equal(q, 0)
            totals["zero_full_day_stockout"] += int(pc.sum(pc.cast(pc.and_(zero, pc.equal(hours, span)), pa.int64())).as_py() or 0)
            totals["zero_partial_stockout"] += int(pc.sum(pc.cast(pc.and_(zero, pc.and_(pc.greater(hours, 0), pc.less(hours, span))), pa.int64())).as_py() or 0)
            totals["zero_no_stockout"] += int(pc.sum(pc.cast(pc.and_(zero, pc.equal(hours, 0)), pa.int64())).as_py() or 0)
            totals["zero_stockout_unknown"] += int(pc.sum(pc.cast(pc.and_(zero, pc.is_null(hours)), pa.int64())).as_py() or 0)
            totals["positive_with_stockout"] += int(pc.sum(pc.cast(pc.and_(pc.greater(q, 0), pc.greater(hours, 0)), pa.int64())).as_py() or 0)
            totals["discount_flagged"] += int(pc.sum(pc.cast(pc.or_(pc.match_substring(flags, "discount_rate_zero"),
                                                                     pc.match_substring(flags, "discount_rate_above_one")), pa.int64())).as_py() or 0)
        as_int = lambda mask: pc.cast(mask, pa.int64())  # noqa: E731
        win_q = pc.if_else(in_window, q, None)
        revenue = pc.multiply(win_q, t["price"]) if price else pa.nulls(t.num_rows, pa.float64())
        table = pa.table({
            "location_id": t["location_id"], "product_id": t["product_id"], "rows": as_int(valid), "qty_sum": q,
            "zero_rows": as_int(pc.equal(q, 0)), "neg_rows": as_int(pc.less(q, 0)),
            "nonint_rows": as_int(pc.not_equal(q, pc.floor(q))), "win_rows": as_int(pc.and_(in_window, valid)),
            "win_sum": win_q, "win_neg": as_int(pc.and_(in_window, pc.less(q, 0))),
            "win_zero": as_int(pc.and_(in_window, pc.equal(q, 0))), "win_revenue": revenue,
            "win_price_rows": as_int(pc.and_(in_window, pc.is_valid(t["price"]))) if price else pc.fill_null(pa.nulls(t.num_rows, pa.int64()), 0),
            "min_date": pc.if_else(valid, t["date"], None), "max_date": pc.if_else(valid, t["date"], None),
            "first_positive": pc.if_else(pc.greater(q, 0), t["date"], None)})
        partials.append(table)
        if len(partials) >= combine_every:
            accumulated = _fold(([accumulated] if accumulated is not None else []) + [_aggregate(p, keys, sum_cols) for p in partials], keys, sum_cols)
            partials = []
    if partials or accumulated is None:
        accumulated = _fold(([accumulated] if accumulated is not None else []) + [_aggregate(p, keys, sum_cols) for p in partials], keys, sum_cols)
    series = accumulated.to_pandas()
    return {"totals": dict(totals), "series": series, "window": (start, end)}


def _aggregate(table: pa.Table, keys: Sequence[str], sum_cols: Sequence[str]) -> pa.Table:
    agg = table.group_by(list(keys)).aggregate([(c, "sum") for c in sum_cols] + [("min_date", "min"), ("max_date", "max"),
                                                                              ("first_positive", "min")])
    names = {f"{c}_sum": c for c in sum_cols}
    names.update({"min_date_min": "min_date", "max_date_max": "max_date", "first_positive_min": "first_positive"})
    return agg.rename_columns([names.get(n, n) for n in agg.column_names])


def _fold(tables: Sequence[pa.Table], keys: Sequence[str], sum_cols: Sequence[str]) -> pa.Table:
    table = pa.concat_tables([t.select([*keys, *sum_cols, "min_date", "max_date", "first_positive"]) for t in tables])
    return _aggregate(table, keys, sum_cols)


def series_summary(stream: Mapping[str, Any]) -> dict[str, Any]:
    series = stream["series"].copy()
    series["span_days"] = (pd.to_datetime(series["max_date"]) - pd.to_datetime(series["min_date"])).dt.days + 1
    series["absent_days"] = series["span_days"] - series["rows"]
    series["leading_zero_days"] = (pd.to_datetime(series["first_positive"]) - pd.to_datetime(series["min_date"])).dt.days
    complete = series["win_rows"] == WINDOW_DAYS
    partial = (series["win_rows"] > 0) & ~complete
    return {
        "series": int(len(series)), "series_with_absent_days": int((series["absent_days"] > 0).sum()),
        "absent_days_within_span": int(series["absent_days"].sum()), "observed_series_days": int(series["rows"].sum()),
        "window": list(stream["window"]), "window_complete_series": int(complete.sum()),
        "window_partial_series": int(partial.sum()), "window_no_observation_series": int((series["win_rows"] == 0).sum()),
        "window_partial_volume_share": float(series.loc[partial, "win_sum"].sum() / series["win_sum"].sum())
        if series["win_sum"].sum() else None,
        "window_series_with_negative": int((series["win_neg"] > 0).sum()),
        "series_with_noninteger_quantity": int((series["nonint_rows"] > 0).sum()),
        "leading_zero_days_total": int(series["leading_zero_days"].fillna(0).sum()),
        "series_never_positive": int(series["first_positive"].isna().sum()),
    }


def sample_window_fidelity(path: Path, stream: Mapping[str, Any], *, samples: int = 4, signal_type: str,
                           negative_semantics: str | None = None) -> list[dict[str, Any]]:
    """Re-aggregate a few series' windows with the contract function; the streamed figures must match."""
    series = stream["series"]
    picks = series.sort_values(["win_rows", "location_id", "product_id"]).iloc[
        [0, len(series) // 3, (2 * len(series)) // 3, len(series) - 1]].head(samples)
    start, end = stream["window"]
    out = []
    for row in picks.itertuples(index=False):
        table = pq.read_table(path, columns=["date", "sales_qty"],
                              filters=[("location_id", "=", row.location_id), ("product_id", "=", row.product_id),
                                       ("date", ">=", start), ("date", "<=", end)])
        frame = table.to_pandas()
        window = imi.aggregate_daily_window([{"date": d, "quantity": q} for d, q in zip(frame["date"], frame["sales_qty"])],
                                            window_end=end, window_days=WINDOW_DAYS, signal_type=signal_type,
                                            negative_semantics=negative_semantics)
        streamed_sum = None if row.win_rows == 0 else float(row.win_sum)
        out.append({"location_id": row.location_id, "product_id": row.product_id, "streamed_rows": int(row.win_rows),
                    "contract_observed_days": window["observed_days"], "streamed_sum": streamed_sum,
                    "contract_sum": window["quantity"],
                    "match": int(row.win_rows) == window["observed_days"]
                    and (streamed_sum is None and window["quantity"] is None
                         or streamed_sum is not None and window["quantity"] is not None
                         and abs(streamed_sum - window["quantity"]) < 1e-6)})
    return out


def m5_abc(series: pd.DataFrame) -> dict[str, Any]:
    """M5 last-30-day ABC per item (all stores): revenue basis (sales x weekly price) vs quantity basis."""
    per_item = series.groupby("product_id").agg(qty=("win_sum", "sum"), revenue=("win_revenue", "sum"))
    revenue = imi.validated_abc([{"product_id": p, "period_sales": r.qty, "unit_price": (r.revenue / r.qty) if r.qty else None,
                                  "signal_type": imi.OBSERVED_RETAIL_SALES, "quantity_unit": "item"}
                                 for p, r in per_item.iterrows() if r.qty > 0])
    quantity = imi.abc_grades({p: float(r.qty) for p, r in per_item.iterrows()})
    rev_grades = {p: g["grade"] for p, g in revenue["grades"].items()}
    qty_grades = {p: g["grade"] for p, g in quantity.items()}
    common = set(rev_grades) & set(qty_grades)
    changed = Counter(f"{qty_grades[p]}->{rev_grades[p]}" for p in common if qty_grades[p] != rev_grades[p])
    return {"items": int(len(per_item)), "items_with_window_sales": len(rev_grades), "revenue_basis": revenue["basis"],
            "revenue_grade_counts": dict(sorted(Counter(rev_grades.values()).items())),
            "quantity_grade_counts": dict(sorted(Counter(qty_grades.values()).items())),
            "items_grade_changed_quantity_to_revenue": int(sum(changed.values())), "changes": dict(sorted(changed.items())),
            "items_without_window_sales": int((per_item["qty"] <= 0).sum())}


def external_sales(data_root: Path) -> dict[str, Any]:
    out = {}
    specs = {
        "m5": dict(window_end="2016-05-22", price=True, signal=imi.OBSERVED_RETAIL_SALES, negative=None,
                   zero=imi.ZERO_RECORDED_POSSIBLY_CENSORED, unit="item"),
        "favorita": dict(window_end="2017-08-15", price=False, signal=imi.OBSERVED_RETAIL_SALES,
                         negative=imi.RETURN_DOCUMENTED, zero=imi.ZERO_ROWS_ABSENT, unit=None),
        "freshretailnet": dict(window_end="2024-07-02", price=False, stockout=True, signal=imi.OBSERVED_RETAIL_SALES,
                               negative=None, zero=imi.ZERO_RECORDED_AVAILABILITY_KNOWN, unit="normalized_sales_amount"),
    }
    for name, spec in specs.items():
        path = data_root / CANONICAL[name]
        started = time.perf_counter()
        stream = stream_demand(path, window_end=spec["window_end"], price=spec["price"], stockout=spec.get("stockout", False))
        summary = series_summary(stream)
        fidelity = sample_window_fidelity(path, stream, signal_type=spec["signal"], negative_semantics=spec["negative"])
        item = {"totals": stream["totals"], "summary": summary, "fidelity": fidelity, "spec": {k: v for k, v in spec.items()},
                "seconds": round(time.perf_counter() - started, 1)}
        if name == "m5":
            item["abc"] = m5_abc(stream["series"])
        if name == "favorita":
            per_item = stream["series"].groupby("product_id")["nonint_rows"].sum()
            item["items_with_noninteger_sales"] = int((per_item > 0).sum())
            item["items"] = int(len(per_item))
        out[name] = item
        del stream
    return out


def domestic_sales(data_root: Path) -> dict[str, Any]:
    out = {}
    logisall = pd.read_parquet(data_root / CANONICAL["logisall_sales"], columns=["date", "location_id", "product_id", "sales_qty", "validation_flags"])
    span = logisall.groupby(["location_id", "product_id"])["date"].agg(["min", "max", "count"])
    span_days = (pd.to_datetime(span["max"]) - pd.to_datetime(span["min"])).dt.days + 1
    out["logisall_sales"] = {"rows": len(logisall), "series": len(span), "zero_rows": int((logisall["sales_qty"] == 0).sum()),
                             "negative_rows": int((logisall["sales_qty"] < 0).sum()),
                             "null_rows": int(logisall["sales_qty"].isna().sum()),
                             "absent_days_within_span": int((span_days - span["count"]).sum()),
                             "duplicate_flag_rows": int(logisall["validation_flags"].fillna("").str.contains("adjacent_identical|duplicate_key").sum()),
                             "date_range": [logisall["date"].min(), logisall["date"].max()]}
    stock = pd.read_parquet(data_root / CANONICAL["logisall_stock"], columns=["snapshot_date", "product_id", "inventory_qty", "scope"])
    out["logisall_stock"] = {"rows": len(stock), "scope": dict(Counter(stock["scope"])), "snapshots_per_product": float(stock.groupby("product_id").size().median()),
                             "products": int(stock["product_id"].nunique()),
                             "sales_products_matching_stock_products": int(len(set(logisall["product_id"]) & set(stock["product_id"])))}
    jangbogo = pd.read_parquet(data_root / CANONICAL["jangbogo_sales"], columns=["date", "location_id", "product_id", "sales_qty", "validation_flags", "product_grain"])
    out["jangbogo_sales"] = {"rows": len(jangbogo), "grain": dict(Counter(jangbogo["product_grain"])),
                             "zero_rows": int((jangbogo["sales_qty"] == 0).sum()), "negative_rows": int((jangbogo["sales_qty"] < 0).sum()),
                             "cross_release_conflict_rows": int(jangbogo["validation_flags"].fillna("").str.contains("cross_release_conflict").sum())}
    jflow = pd.read_parquet(data_root / CANONICAL["jangbogo_flow"], columns=["order_qty", "flow_kind", "validation_flags"])
    out["jangbogo_orders"] = {"rows": len(jflow), "flow_kind": dict(Counter(jflow["flow_kind"])),
                              "negative_rows": int((jflow["order_qty"] < 0).sum())}
    nfqs = pd.read_parquet(data_root / CANONICAL["nfqs_stock"], columns=["snapshot_date", "location_id", "product_id", "inventory_qty", "date_grain", "unit", "validation_flags"])
    out["nfqs_stock"] = {"rows": len(nfqs), "grain": dict(Counter(nfqs["date_grain"])), "unit": dict(Counter(nfqs["unit"])),
                         "snapshot_dates": int(nfqs["snapshot_date"].nunique()),
                         "negative_rows": int((nfqs["inventory_qty"] < 0).sum()),
                         "float_residue_flag_rows": int(nfqs["validation_flags"].fillna("").str.contains("float_residue").sum())}
    a_stock = pd.read_parquet(data_root / CANONICAL["aihub_stock"], columns=["snapshot_date", "inventory_qty", "unit"])
    a_flow = pd.read_parquet(data_root / CANONICAL["aihub_flow"], columns=["date", "inbound_qty", "outbound_qty", "unit", "flow_kind"])
    period = (a_stock["snapshot_date"].min(), a_stock["snapshot_date"].max())
    window = imi.aggregate_daily_window([{"date": d, "quantity": q} for d, q in zip(a_flow["date"], a_flow["outbound_qty"])],
                                        window_end=period[1], window_days=(date.fromisoformat(period[1]) - date.fromisoformat(period[0])).days + 1,
                                        dataset_id="aihub", signal_type=imi.OBSERVED_OUTBOUND_FLOW, quantity_unit="item",
                                        zero_semantics=imi.ZERO_RECORDED, semantics_evidence=imi.SOURCE_DOCUMENTED,
                                        provenance="canonical inventory_flow.outbound_qty (site aggregate outbound)")
    inventory = imi.average_inventory([{"date": d, "quantity": q} for d, q in zip(a_stock["snapshot_date"], a_stock["inventory_qty"])],
                                      period_start=period[0], period_end=period[1], dataset_id="aihub", quantity_unit="item")
    turnover = imi.physical_turnover(window, inventory)
    out["aihub"] = {"snapshot_days": len(a_stock), "period": list(period), "period_days": window["period_days"],
                    "flow_days": window["observed_days"], "no_record_days": window["no_record_days"],
                    "turnover_value": turnover["turnover_value"], "turnover_status": turnover["turnover_status"],
                    "turnover_basis": turnover["turnover_basis"], "gate_status": turnover["gate_status"],
                    "reason_codes": turnover["reason_codes"], "proxy_classification": turnover["turnover_classification"],
                    "classification_confidence": turnover["classification_confidence"]}
    kamp = pd.read_parquet(data_root / CANONICAL["kamp_flow"], columns=["shipment_qty", "flow_kind", "date_grain"])
    out["kamp"] = {"rows": len(kamp), "flow_kind": dict(Counter(kamp["flow_kind"])), "grain": dict(Counter(kamp["date_grain"])),
                   "zero_rows": int((kamp["shipment_qty"] == 0).sum())}
    return out


# ------------------------------------------------------------------------------------------------ coverage matrix (§28)


def coverage_matrix(external: Mapping[str, Any], domestic: Mapping[str, Any], suhyup_proxy: pd.DataFrame) -> pd.DataFrame:
    u = "UNSUPPORTED"
    proxy_rows = int((suhyup_proxy["turnover_status"].isin([imi.TURNOVER_AVAILABLE, imi.ZERO_TURNOVER])).sum())
    m5_abc_info = external.get("m5", {}).get("abc", {})
    rows = [
        {"dataset": "suhyup", "retail_sales": False, "logistics_outbound": True, "inventory": True,
         "inventory_grain": "daily snapshot per center x product x state", "snapshots_in_period": "31 days (2026-07)",
         "price_or_cost": False, "sales_inventory_same_unit": "count UNKNOWN on both (same source); kg DIRECT on both",
         "turnover": f"{imi.FLOW_BASED_TURNOVER_PROXY} ({proxy_rows} keys)", "real_sales_turnover": u,
         "financial_turnover": u, "abc": f"{imi.ABC_QUANTITY_PROXY} only (no price/cost; legacy multiplies by 1.0)",
         "disposal_risk": "HEURISTIC only (no disposal history, no expiry)",
         "evidence": "canonical inventory_snapshot + inventory_flow (출고량 = outbound, not retail sales)"},
        {"dataset": "jangbogo", "retail_sales": "sales (channel unspecified)", "logistics_outbound": False, "inventory": False,
         "inventory_grain": "none", "snapshots_in_period": "0", "price_or_cost": False,
         "sales_inventory_same_unit": "no inventory", "turnover": u, "real_sales_turnover": u, "financial_turnover": u,
         "abc": f"{u} (category grain, unit UNKNOWN; purchase requests are {imi.OBSERVED_ORDER})", "disposal_risk": u,
         "evidence": f"monthly warehouse x category sales {domestic['jangbogo_sales']['rows']} rows; "
                     f"{domestic['jangbogo_orders']['rows']} purchase-request rows (not receipts)"},
        {"dataset": "logisall", "retail_sales": "sales (channel unspecified)", "logistics_outbound": True, "inventory": True,
         "inventory_grain": "monthly national", "snapshots_in_period": f"median {domestic['logisall_stock']['snapshots_per_product']} months per product",
         "price_or_cost": False, "sales_inventory_same_unit": "UNKNOWN on both; zone-daily sales vs national-monthly stock (scope differs)",
         "turnover": f"{u} (NOT_COMPARABLE scope)", "real_sales_turnover": u, "financial_turnover": u,
         "abc": f"{u} (unit UNKNOWN, zero days absent)", "disposal_risk": u,
         "evidence": f"zone-daily sales {domestic['logisall_sales']['rows']} rows, zero rows {domestic['logisall_sales']['zero_rows']} "
                     f"(absent days {domestic['logisall_sales']['absent_days_within_span']} within series spans)"},
        {"dataset": "nfqs", "retail_sales": False, "logistics_outbound": False, "inventory": True,
         "inventory_grain": "quarterly regional", "snapshots_in_period": f"{domestic['nfqs_stock']['snapshot_dates']} quarters",
         "price_or_cost": False, "sales_inventory_same_unit": "no sales", "turnover": f"{u} (no outflow)",
         "real_sales_turnover": u, "financial_turnover": u, "abc": u, "disposal_risk": u,
         "evidence": "cold-storage stock in tons (SOURCE_METADATA); average inventory possible, numerator absent"},
        {"dataset": "aihub", "retail_sales": False, "logistics_outbound": True, "inventory": True,
         "inventory_grain": "daily site aggregate (all products)", "snapshots_in_period": f"{domestic['aihub']['snapshot_days']} of {domestic['aihub']['period_days']} days",
         "price_or_cost": False, "sales_inventory_same_unit": "item on both (SOURCE_METADATA)",
         "turnover": f"{imi.FLOW_BASED_TURNOVER_PROXY} (site aggregate, {domestic['aihub']['turnover_status']})",
         "real_sales_turnover": u, "financial_turnover": u, "abc": u, "disposal_risk": u,
         "evidence": "unidentified site, all products summed; no product turnover"},
        {"dataset": "kamp", "retail_sales": False, "logistics_outbound": True, "inventory": False,
         "inventory_grain": "none", "snapshots_in_period": "0", "price_or_cost": False, "sales_inventory_same_unit": "no inventory",
         "turnover": u, "real_sales_turnover": u, "financial_turnover": u, "abc": u, "disposal_risk": u,
         "evidence": "order-based rebar shipments to one project (event grain)"},
        {"dataset": "m5", "retail_sales": True, "logistics_outbound": False, "inventory": False, "inventory_grain": "none",
         "snapshots_in_period": "0", "price_or_cost": "weekly sell price (no cost)", "sales_inventory_same_unit": "no inventory",
         "turnover": f"{u} (no inventory)", "real_sales_turnover": u, "financial_turnover": u,
         "abc": f"{imi.ABC_REVENUE} verifiable ({m5_abc_info.get('items_with_window_sales')} items, last 30 days)",
         "disposal_risk": u, "evidence": "actual daily unit sales, zero rows recorded (stock-outs unobserved)"},
        {"dataset": "favorita", "retail_sales": True, "logistics_outbound": False, "inventory": False, "inventory_grain": "none",
         "snapshots_in_period": "0", "price_or_cost": False, "sales_inventory_same_unit": "no inventory",
         "turnover": f"{u} (no inventory)", "real_sales_turnover": u, "financial_turnover": u,
         "abc": f"{imi.ABC_NOT_VERIFIABLE} (no price; unit item-dependent: {external.get('favorita', {}).get('items_with_noninteger_sales')} items with fractional quantities)",
         "disposal_risk": u, "evidence": "actual daily sales, signed (returns), zero rows absent"},
        {"dataset": "freshretailnet", "retail_sales": True, "logistics_outbound": False, "inventory": False,
         "inventory_grain": "none (hourly stock-out status is availability, not quantity)", "snapshots_in_period": "0",
         "price_or_cost": False, "sales_inventory_same_unit": "no inventory", "turnover": f"{u} (no inventory)",
         "real_sales_turnover": u, "financial_turnover": u,
         "abc": f"{imi.ABC_NOT_VERIFIABLE} (normalised amount; whether quantity or money is undisclosed)", "disposal_risk": u,
         "evidence": "actual normalised sales with stock-out hours"},
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ controlled scenarios (§30)

D0 = date(2026, 7, 1)
TEST_DS = "TEST_DATASET"


def _days(n: int, qty: Any, *, start: date = D0, **extra: Any) -> list[dict[str, Any]]:
    return [{"date": (start + timedelta(days=i)).isoformat(), "quantity": qty if not callable(qty) else qty(i), **extra}
            for i in range(n)]


def _end(n: int = 30) -> str:
    return (D0 + timedelta(days=n - 1)).isoformat()


def _window(records: Sequence[Mapping[str, Any]], **kw: Any) -> dict[str, Any]:
    params = {"window_end": _end(), "window_days": 30, "dataset_id": TEST_DS, "signal_type": imi.OBSERVED_RETAIL_SALES,
              "quantity_unit": "EA", "zero_semantics": imi.ZERO_RECORDED, "semantics_evidence": imi.SOURCE_DOCUMENTED,
              "provenance": "TEST POS sales"}
    params.update(kw)
    return imi.aggregate_daily_window(records, **params)


def _inventory(snapshots: Sequence[Mapping[str, Any]], **kw: Any) -> dict[str, Any]:
    params = {"period_start": D0.isoformat(), "period_end": _end(), "dataset_id": TEST_DS, "quantity_unit": "EA"}
    params.update(kw)
    return imi.average_inventory(snapshots, **params)


def _legacy_turnover(frame: pd.DataFrame) -> pd.DataFrame:
    from services.legacy_adapters.loader import load_legacy_module

    return load_legacy_module("turnover_analyzer").analyze_turnover(frame)


def _scenario(sid: str, name: str, result: Mapping[str, Any], expected: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    actual = {k: result.get(k) for k in expected}
    ok = all((actual[k] == v) if not isinstance(v, set) else set(actual[k] or []) >= v for k, v in expected.items())
    return {"scenario": sid, "name": name, "value": result.get("turnover_value", result.get("value")),
            "unit": result.get("quantity_unit") or extra.get("unit"), "period": extra.get("period"),
            "provenance": extra.get("provenance") or result.get("turnover_basis"),
            "status": result.get("turnover_status") or result.get("status") or result.get("gate_status"),
            "classification": result.get("turnover_classification"),
            "confidence": result.get("classification_confidence"),
            "gate_status": result.get("gate_status"), "reason_codes": _codes(result.get("reason_codes")),
            "expected": json.dumps({k: sorted(v) if isinstance(v, set) else v for k, v in expected.items()}, ensure_ascii=False),
            "actual": json.dumps({k: sorted(v) if isinstance(v, (set, list)) else v for k, v in actual.items()},
                                 ensure_ascii=False, default=_json_default),
            "legacy": extra.get("legacy"), "pass": bool(ok)}


def controlled_scenarios() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stock = _days(30, 200.0)
    full_inv = _inventory(stock)
    sales = _window(_days(30, 10.0))

    a = imi.physical_turnover(sales, full_inv)
    rows.append(_scenario("A", "normal observed retail sales", a,
                          {"turnover_status": imi.TURNOVER_AVAILABLE, "turnover_basis": imi.REAL_SALES_TURNOVER,
                           "gate_status": imi.VALID, "turnover_classification": "FAST", "classification_confidence": "HIGH",
                           "turnover_value": 1.5}, unit="EA", period="2026-07-01..2026-07-30"))
    b = imi.physical_turnover(_window(_days(30, 10.0), signal_type=imi.OBSERVED_OUTBOUND_FLOW, provenance="TEST outbound"), full_inv)
    rows.append(_scenario("B", "normal observed outbound flow", b,
                          {"turnover_basis": imi.FLOW_BASED_TURNOVER_PROXY, "gate_status": imi.PROXY_ONLY,
                           "classification_scope": imi.SCOPE_PROXY, "classification_confidence": "LOW",
                           "turnover_status": imi.TURNOVER_AVAILABLE}, unit="EA", period="30 days"))
    c = imi.physical_turnover(_window([]), full_inv)
    legacy_c = _legacy_turnover(pd.DataFrame({"product_name": ["x"], "stock_qty": [200.0], "sales_30d": [float("nan")]}))
    rows.append(_scenario("C", "sales missing (no record): never 0", c,
                          {"turnover_status": imi.TURNOVER_UNAVAILABLE, "gate_status": imi.MISSING_DATA,
                           "turnover_classification": imi.UNCLASSIFIED, "reason_codes": {"SALES_MISSING", "NO_OBSERVATION"}},
                          legacy=f"legacy turnover_days {legacy_c['turnover_days'].iloc[0]} grade {legacy_c['turnover_grade'].iloc[0]}"))
    d = imi.physical_turnover(_window(_days(30, 0.0)), full_inv)
    rows.append(_scenario("D", "observed zero sales", d,
                          {"turnover_status": imi.ZERO_TURNOVER, "turnover_value": 0.0, "gate_status": imi.VALID,
                           "days_of_supply_status": imi.DOS_NO_MOVEMENT, "turnover_classification": "DEAD",
                           "classification_confidence": "HIGH"}))
    returns = _days(30, lambda i: -4.0 if i == 10 else 10.0)
    e = imi.physical_turnover(_window(returns, negative_semantics=imi.RETURN_DOCUMENTED), full_inv)
    rows.append(_scenario("E", "negative = documented return: netted, never clipped", e,
                          {"turnover_status": imi.TURNOVER_AVAILABLE, "sales_quantity": 286.0,
                           "gate_status": imi.VALID_WITH_LIMITATION, "reason_codes": {"RETURNS_NETTED_DOCUMENTED"}}))
    e2 = imi.physical_turnover(_window(returns), full_inv)
    legacy_e = _legacy_turnover(pd.DataFrame({"product_name": ["x"], "stock_qty": [200.0], "sales_30d": [-30.0]}))
    rows.append(_scenario("E2", "negative without semantics: invalid, unclassified", e2,
                          {"turnover_status": imi.INVALID_VALUE, "gate_status": imi.INVALID_VALUE,
                           "turnover_classification": imi.UNCLASSIFIED, "reason_codes": {"NEGATIVE_UNRESOLVED"}},
                          legacy=f"legacy sales_30d=-30: turnover_days {legacy_e['turnover_days'].iloc[0]} grade {legacy_e['turnover_grade'].iloc[0]}"))
    f = imi.physical_turnover(sales, _inventory(_days(30, 0.0)))
    rows.append(_scenario("F", "average inventory 0", f,
                          {"turnover_status": imi.DIVISION_BY_ZERO, "turnover_value": None,
                           "days_of_supply_status": imi.DOS_ZERO_STOCK, "turnover_classification": imi.UNCLASSIFIED}))
    g = imi.physical_turnover(sales, _inventory([]))
    rows.append(_scenario("G", "average inventory missing", g,
                          {"turnover_status": imi.AVG_INVENTORY_UNAVAILABLE, "turnover_value": None,
                           "turnover_classification": imi.UNCLASSIFIED, "reason_codes": {"NO_SNAPSHOT"}}))
    h_inv = _inventory([{"date": D0.isoformat(), "quantity": 200.0}])
    h = imi.physical_turnover(sales, h_inv)
    rows.append(_scenario("H", "opening snapshot only", h,
                          {"turnover_status": imi.AVG_INVENTORY_UNAVAILABLE, "turnover_classification": imi.UNCLASSIFIED,
                           "reason_codes": {"START_SNAPSHOT_ONLY"}}))
    i_inv = _inventory([{"date": _end(), "quantity": 200.0}])
    i = imi.physical_turnover(sales, i_inv)
    rows.append(_scenario("I", "closing snapshot only: days of supply yes, ratio no", i,
                          {"turnover_status": imi.AVG_INVENTORY_UNAVAILABLE, "turnover_value": None, "days_of_supply": 20.0,
                           "turnover_classification": "FAST", "reason_codes": {"END_SNAPSHOT_ONLY"}}))
    partial = [r for k, r in enumerate(_days(30, 10.0)) if k % 3 != 0]
    j = imi.physical_turnover(_window(partial), full_inv)
    rows.append(_scenario("J", "incomplete observation period", j,
                          {"turnover_status": imi.PARTIAL_PERIOD, "sales_quantity": 200.0, "daily_rate": 10.0,
                           "classification_confidence": "LOW", "reason_codes": {"PARTIAL_OBSERVATION", "RATE_OVER_OBSERVED_DAYS"}}))
    k_sig = imi.SalesSignal(dataset_id=TEST_DS, location_id="S", product_id="P", quantity=70.0,
                            signal_type=imi.OBSERVED_RETAIL_SALES, aggregation=imi.WINDOW_SUM, period_days=7,
                            quantity_unit="EA", semantics_evidence=imi.COLUMN_HEADER_CLAIM)
    l_sig = imi.SalesSignal(**{**k_sig.__dict__, "quantity": 300.0, "period_days": 30})
    m_sig = imi.SalesSignal(**{**k_sig.__dict__, "quantity": 10.0, "aggregation": imi.DAILY_MEAN, "period_days": None})
    n_sig = imi.SalesSignal(**{**k_sig.__dict__, "quantity": 310.0, "aggregation": imi.CALENDAR_MONTH_SUM, "period_days": 31})
    for sid, name, sig, conf, extra_reason in (("K", "7-day sum", k_sig, "LOW", "RATE_WINDOW_SHORTER_THAN_LEGACY_30D"),
                                               ("L", "30-day sum", l_sig, "MEDIUM", "WINDOW_COMPLETENESS_UNDECLARED"),
                                               ("M", "per-day mean (window undeclared)", m_sig, "MEDIUM", "AVERAGE_WINDOW_UNDECLARED"),
                                               ("N", "calendar month sum", n_sig, "MEDIUM", "WINDOW_COMPLETENESS_UNDECLARED")):
        window = imi.signal_window(sig)
        per_day = sig.aggregation in imi.PER_DAY_AGGREGATIONS
        res = imi.physical_turnover({**window, "quantity": sig.quantity} if per_day else window,
                                    imi.declared_average_inventory(180.0, dataset_id=TEST_DS, quantity_unit="EA"),
                                    end_stock=200.0, per_day_only=per_day)
        expected = {"daily_rate": 10.0, "days_of_supply": 20.0, "turnover_classification": "FAST",
                    "classification_confidence": conf, "reason_codes": {extra_reason}}
        if per_day:
            expected["turnover_status"] = imi.DAYS_OF_SUPPLY_ONLY
        rows.append(_scenario(sid, name, res, expected, period=f"{sig.aggregation}:{sig.period_days}"))
    comparison = imi.signals_comparable(k_sig, l_sig)
    rows.append(_scenario("KLM", "raw 7-day vs 30-day quantities are not compared", comparison,
                          {"comparable": False, "gate": imi.PERIOD_MISMATCH}))
    o_inv = imi.average_inventory([{"date": d, "quantity": q} for d, q in (("2025-03-31", 40.0), ("2025-06-30", 55.0),
                                                                          ("2025-09-30", 35.0), ("2025-12-31", 50.0))],
                                  period_start="2025-03-31", period_end="2025-12-31", dataset_id=TEST_DS, quantity_unit="ton")
    o = imi.physical_turnover(_window([], window_end="2025-12-31", window_days=276, quantity_unit="ton"), o_inv)
    rows.append(_scenario("O", "quarterly inventory, no sales", o,
                          {"turnover_status": imi.TURNOVER_UNAVAILABLE, "average_inventory": 45.0},
                          provenance=f"inventory {o_inv['status']} {_codes(o_inv['reason_codes'])}"))
    p = imi.physical_turnover(_window(_days(30, 10.0), quantity_unit="KG"), _inventory(stock, quantity_unit="BOX"))
    rows.append(_scenario("P", "sales KG vs inventory BOX", p,
                          {"turnover_status": imi.UNIT_MISMATCH, "gate_status": imi.UNIT_MISMATCH, "turnover_value": None}))
    abc_rows = [{"product_id": f"P{n}", "period_sales": float(s), "unit_cost": float(c), "signal_type": imi.OBSERVED_RETAIL_SALES,
                 "quantity_unit": "EA"} for n, (s, c) in enumerate(((500, 10), (300, 20), (100, 5), (50, 50), (10, 1)))]
    q_res = imi.validated_abc(abc_rows)
    rows.append(_scenario("Q", "price missing: cost-valued ABC, never revenue", {**q_res, "status": q_res["basis"]},
                          {"basis": imi.ABC_COST_VALUED, "gate_status": imi.VALID}))
    r_legacy = imi.legacy_abc_basis(["product_name", "sales_30d", "stock_qty"])
    r_fin = imi.financial_turnover(average_inventory_value=1000.0, value_currency="KRW", period_days=30)
    rows.append(_scenario("R", "cost missing: legacy x1.0 quantity ABC; financial turnover unavailable",
                          {**r_fin, "legacy_basis": r_legacy["legacy_basis"], "legacy_codes": r_legacy["legacy_codes"]},
                          {"status": imi.FINANCIAL_TURNOVER_UNAVAILABLE, "legacy_basis": imi.ABC_QUANTITY_PROXY,
                           "reason_codes": {"COGS_MISSING"}, "legacy_codes": {"LEGACY_ABC_COST_ABSENT_TIMES_ONE"}}))
    rows.append(_scenario("S", "real-sales turnover computable", a,
                          {"turnover_value": 1.5, "turnover_annualized_365": round(1.5 * 365 / 30, 6),
                           "turnover_basis": imi.REAL_SALES_TURNOVER, "classification_scope": imi.SCOPE_REAL}))
    rows.append(_scenario("T", "flow-based turnover proxy", b,
                          {"turnover_value": 1.5, "turnover_basis": imi.FLOW_BASED_TURNOVER_PROXY,
                           "classification_scope": imi.SCOPE_PROXY}))
    u_rows = [{**r, "unit_price": float(pr)} for r, pr in zip(abc_rows, (20, 30, 8, 60, 2))]
    u_res = imi.validated_abc(u_rows)
    rows.append(_scenario("U", "ABC on revenue", {**u_res, "status": u_res["basis"],
                                                   "top_grade": u_res["grades"]["P0"]["grade"]},
                          {"basis": imi.ABC_REVENUE, "gate_status": imi.VALID, "top_grade": "A"}))
    v_res = imi.validated_abc([{k: v for k, v in r.items() if k != "unit_cost"} for r in abc_rows])
    rows.append(_scenario("V", "ABC without price or cost: quantity proxy", {**v_res, "status": v_res["basis"]},
                          {"basis": imi.ABC_QUANTITY_PROXY, "gate_status": imi.PROXY_ONLY}))
    w_audit = imi.disposal_input_audit({"stock_qty": 100.0, "sales_30d": 30.0, "days_to_expiry": 5.0, "category": "신선"},
                                       ["stock_qty", "sales_30d", "days_to_expiry", "category"], {})
    rows.append(_scenario("W", "disposal risk is a heuristic score", {**w_audit, "status": w_audit["meaning"]},
                          {"meaning": imi.DISPOSAL_MEANING, "is_probability": False, "predicts_disposed_units": False}))
    x = imi.disposal_probability()
    rows.append(_scenario("X", "no disposal history: no probability", x,
                          {"status": imi.DISPOSAL_PROBABILITY_UNAVAILABLE, "value": None,
                           "reason_codes": {"NO_DISPOSAL_HISTORY", "NO_OBSERVED_EXPIRY"}}))
    y = imi.physical_turnover(_window(_days(30, None, day_state=imi.NOT_CARRIED)), full_inv)
    rows.append(_scenario("Y", "location does not carry the product: not zero demand", y,
                          {"turnover_status": imi.NOT_APPLICABLE, "turnover_classification": imi.UNCLASSIFIED,
                           "reason_codes": {"NOT_CARRIED_NOT_ZERO"}}))
    z = imi.physical_turnover(sales, full_inv, end_stock=0.0)
    legacy_z = _legacy_turnover(pd.DataFrame({"product_name": ["x"], "stock_qty": [0.0], "sales_30d": [300.0]}))
    rows.append(_scenario("Z", "actual zero stock", z,
                          {"days_of_supply_status": imi.DOS_ZERO_STOCK, "turnover_classification": imi.UNCLASSIFIED,
                           "reason_codes": {"ZERO_STOCK_NOT_CLASSIFIED"}},
                          legacy=f"legacy stock 0: turnover_days {legacy_z['turnover_days'].iloc[0]} grade {legacy_z['turnover_grade'].iloc[0]}"))
    aa = imi.physical_turnover(_window(_days(30, 10.0), signal_type=imi.FORECAST_DEMAND, provenance="TEST forecast"), full_inv)
    rows.append(_scenario("AA", "forecast demand only", aa,
                          {"turnover_basis": imi.FORECAST_DEMAND_PROXY, "gate_status": imi.PROXY_ONLY,
                           "classification_scope": imi.SCOPE_PROXY}))
    mixed = _days(15, 10.0, signal_type=imi.OBSERVED_RETAIL_SALES) + _days(15, 10.0, start=D0 + timedelta(days=15),
                                                                           signal_type=imi.OBSERVED_OUTBOUND_FLOW)
    ab_window = _window(mixed)
    ab = imi.physical_turnover(ab_window, full_inv)
    rows.append(_scenario("AB", "sales and outbound mixed", {**ab, "window_quantity": ab_window["quantity"]},
                          {"turnover_status": imi.NOT_COMPARABLE, "window_quantity": None,
                           "reason_codes": {"SIGNAL_TYPE_MISMATCH"}}))
    dup = _days(30, 10.0) + [{"date": D0.isoformat(), "quantity": 5.0}]
    ac_window = _window(dup)
    ac = imi.physical_turnover(ac_window, full_inv)
    rows.append(_scenario("AC", "duplicate date", {**ac, "window_quantity": ac_window["quantity"]},
                          {"turnover_status": imi.INVALID_VALUE, "window_quantity": None, "reason_codes": {"DUPLICATE_DATE"}}))
    future = _window(_days(35, 10.0), as_of=_end())
    rows.append(_scenario("AD", "future data blocked", {**future, "status": future["gate_status"]},
                          {"future_records_excluded": 5, "observed_days": 30, "quantity": 300.0,
                           "reason_codes": {"FUTURE_DATA_EXCLUDED"}}))
    again = imi.physical_turnover(_window(_days(30, 10.0)), _inventory(_days(30, 200.0)))
    frame = pd.DataFrame({"store_id": ["S1", "S2"], "product_id": ["P1", "P1"], "stock_qty": [100.0, 0.0],
                          "sales_30d": [30.0, float("nan")], "unit_cost": [10.0, 10.0]})
    det = imi.build_inventory_metric_integrity({"inventory": frame}) == imi.build_inventory_metric_integrity({"inventory": frame.copy()})
    rows.append(_scenario("AE", "same input, same result", {"identical": again == a and det, "status": "DETERMINISTIC"},
                          {"identical": True}))
    af = imi.physical_turnover(_window(_days(30, 10.0), dataset_id="D1"), _inventory(stock, dataset_id="D2"))
    af_window = _window([*_days(15, 10.0, dataset_id="D1"), *_days(15, 10.0, start=D0 + timedelta(days=15), dataset_id="D2")])
    rows.append(_scenario("AF", "datasets mixed", {**af, "window_gate": af_window["gate_status"]},
                          {"turnover_status": imi.NOT_COMPARABLE, "window_gate": imi.NOT_COMPARABLE,
                           "reason_codes": {"DATASET_MISMATCH"}}))
    return rows


# ------------------------------------------------------------------------------------------------ tables


TURNOVER_FORMULA_AUDIT = (
    # element, legacy (turnover_analyzer), validated contract, difference
    ("numerator", "sales_30d (first of sales_30d | state_source_sales_30d | source_sales_30d, else avg_daily_sales x 30); "
     "NULL -> 0", "units moved out in the period from daily records (observed retail sales = real; outbound / forecast = proxy)",
     "missing is never 0; signal type decides the basis"),
    ("denominator (ratio)", "avg_inventory if > 0 else current stock_qty (single snapshot)",
     "mean of the observed snapshots in the same period (opening + closing required); a declared average is accepted "
     "with WINDOW_UNDECLARED", "one snapshot is not an average"),
    ("period", "sales_30d treated as 30 days; the ratio is annualised x 12 (360 days)",
     "the sales window and the inventory period must be the same; annualised x 365 / period days",
     "periods are checked, not assumed"),
    ("days of supply (legacy turnover_days)", "stock_qty / (sales_30d / 30); sales 0 or NULL -> 999; negative -> negative",
     "end stock / validated daily rate; rate 0 -> NO_MOVEMENT (inf); stock 0 -> ZERO_STOCK; NULL -> unavailable",
     "0, NULL and negative are kept apart"),
    ("grade", "turnover_analyzer._assign_grade on turnover_days (30/60/90)", "the same function on validated days of supply, "
     "with scope (REAL / CLAIMED / PROXY) and confidence; UNCLASSIFIED when not computable", "thresholds unchanged"),
    ("missing sales", "fillna(0) -> 999 days -> DEAD", "TURNOVER_UNAVAILABLE / MISSING_DATA, UNCLASSIFIED", "no DEAD from missing data"),
    ("negative sales", "kept: negative days -> FAST", "documented returns netted (labelled); unexplained -> INVALID_VALUE",
     "no FAST from returns"),
    ("average inventory 0", "rate 0.0 (fillna)", "DIVISION_BY_ZERO", "not a zero turnover"),
    ("units", "not checked", "sales and inventory units must match (unknown on both: same dataset only, limitation)",
     "UNIT_MISMATCH blocks"),
    ("cost basis", "none (units only)", "financial turnover only from explicit COGS and inventory value in one currency",
     "no money turnover without COGS"),
)

DOWNSTREAM = (
    ("ABC", "abc_analyzer", "sales_30d x unit_cost (cost absent -> 1.0)", "cost-valued sales labelled revenue; quantity "
     "ABC hidden when cost is absent; sales_30d may be a daily copy", "basis named (REVENUE / COST_VALUED / QUANTITY_PROXY); "
     "missing values excluded, never 1 or 0", "VHS abc_score 0.04, legacy action 폐기 branch"),
    ("Turnover", "turnover_analyzer", "sales_30d, stock_qty, avg_inventory", "missing -> DEAD, negative -> FAST, stock 0 -> "
     "FAST, daily value in sales_30d, single-day projection", "UNCLASSIFIED / PROXY / ZERO_STOCK statuses",
     "VHS turnover_score 0.06, sit_DEAD_STOCK weights, legacy action 할인/폐기"),
    ("Disposal risk", "disposal_risk_analyzer", "expiry, sales_30d, stock, inbound_days, category",
     "heuristic points; missing inbound -> fresh, missing sales -> maximum points, total shelf life aliased as remaining",
     "named HEURISTIC_RISK_SCORE; no probability without disposal history", "VHS 0.09, auto VHS disposal component, legacy action 할인/폐기"),
    ("Safety stock", "safety_stock_analyzer", "avg_daily_sales | sales_30d/30, demand_std, lead_time_days",
     "outbound mean used as demand mean; std 0.30 x mean and lead time 3 placeholders", "placeholders and proxy flagged per row",
     "reorder_status -> matcher urgency, legacy action 재배치 branch"),
    ("Store-product matching", "store_product_matcher", "avg_daily_sales | sales_30d, state_target_sales_30d",
     "target sales never supplied: neutral copy of the source; source zero gives demand fit 0", "REFERENCE_SCORE_ONLY unless "
     "target sales are observed", "match_grade -> 재배치 branch (unreachable on Suhyup)"),
    ("Promotion", "promotion_analyzer", "suggested_qty, config uplift / discount, holding cost", "no sales signal; config "
     "placeholders (20% / 80%) as in the Seller Loss findings", "outside this contract (no sales input)", "promotion_recommended"),
    ("VHS / auto VHS", "varo_hybrid_score, vhs_score_engine", "turnover/abc/disposal scores; demand_forecast_7d + 0.12 x sales_30d",
     "mixes a 7-day forecast with a 30-day field; inherits the grades above", "grades carry scope/confidence; period mix "
     "documented", "varo_final_rank (Top-5)"),
    ("Seller Loss", "seller_loss_engine", "source/target daily demand with provenance", "already rejects PROXY demand "
     "(demand_proxy_not_retail_sales)", "consistent: the same semantics label is read here", "parallel field only"),
    ("Forecast V1 / router", "demand_forecast_analyzer, demand_forecast_router", "sales_7d, avg_daily_sales, sales_30d; "
     "daily_sales_history", "aliased fields enter V1; the router has no signal-type field (outbound would read as sales)",
     "FORECAST_V1_INPUT_* and FORECAST_ROUTER_NO_SIGNAL_TYPE per row", "demand_forecast_* -> VHS, risk"),
    ("Candidate generator / caps", "candidate_generator, optimality_gap_service, inventory_transition", "avg_daily_sales | "
     "sales_qty | demand_qty (x7); demand_qty read as 7-day demand", "demand_qty means a 30-day shortfall in the network "
     "sample, a 7-day forecast in the V2 samples", "demand_qty typed UNKNOWN (overloaded)", "recommended_qty, target caps"),
)


# ------------------------------------------------------------------------------------------------ regressions


def golden_186(data_root: Path) -> pd.DataFrame:
    from services import candidate_generation_validation as cgv
    from services import shared_feasibility_validation as sfv

    saved = sfv.load_suhyup_candidates(data_root)
    reference = pd.read_csv(data_root / sfv.REFERENCE, dtype={"date": str})
    rows = []
    with _env(REAL_DATA_ROOT_ENV, str(data_root)):
        for day in sorted(saved["snapshot_date"].astype(str).unique()):
            ranked = sfv.ranked_day(saved[saved["snapshot_date"] == day], day)
            rows.extend(cgv._golden({"date": day, "ranked_saved": ranked}, reference))
    return pd.DataFrame(rows)


def t1_rerun(data_root: Path) -> dict[str, Any]:
    from services import shared_feasibility_validation as sfv

    with _env(REAL_DATA_ROOT_ENV, str(data_root)):
        suhyup = sfv.run_suhyup(data_root)
    summary = sfv._summary(suhyup).set_index("plan")
    bench = summary.loc["SF_BENCHMARK_ALL_OR_NOTHING"]
    strict = summary.loc["SF_STRICT_ACTUAL"]
    return {"benchmark_service": float(bench["total_service_qty"]), "benchmark_cost": float(bench["total_cost"]),
            "benchmark_violations": int(bench["violations"]), "strict_selected": int(strict["selected_routes"]),
            "strict_claim": str(strict["claim"])}


def earlier_bundles(data_root: Path) -> dict[str, Any]:
    def load(path: str) -> dict[str, Any]:
        return json.loads((data_root / path).read_text(encoding="utf-8"))

    out: dict[str, Any] = {}
    t1 = pd.read_csv(data_root / "_SHARED_FEASIBILITY_VALIDATION/shared_feasibility_summary.csv").set_index("plan")
    out["T1"] = {"service": float(t1.loc["SF_BENCHMARK_ALL_OR_NOTHING", "total_service_qty"]),
                 "cost": float(t1.loc["SF_BENCHMARK_ALL_OR_NOTHING", "total_cost"]),
                 "violations": int(t1.loc["SF_BENCHMARK_ALL_OR_NOTHING", "violations"]),
                 "failed_checks": load("_SHARED_FEASIBILITY_VALIDATION/shared_feasibility_validation.json")["failed_checks"]}
    t2 = load("_ACTION_CONSISTENCY_VALIDATION/action_consistency_validation.json")
    t2_candidates = pd.read_csv(data_root / "_ACTION_CONSISTENCY_VALIDATION/action_consistency_candidates.csv", dtype=str)
    out["T2"] = {"candidates": int((t2_candidates["scope"] == "SUHYUP_31D").sum()), "failed_checks": t2["failed_checks"],
                 "legacy_action_preserved": not bool(t2.get("production_action_applied"))}
    t3 = load("_MILP_BENCHMARK_INTEGRITY/milp_benchmark_integrity.json")
    t3_days = pd.read_csv(data_root / "_MILP_BENCHMARK_INTEGRITY/milp_benchmark_by_day.csv", dtype=str)
    status_col = "solver_status" if "solver_status" in t3_days.columns else None
    out["T3"] = {"days": int(len(t3_days)), "failed_checks": t3["failed_checks"],
                 "status_column": status_col,
                 "optimal_days": int(t3_days[status_col].astype(str).str.upper().str.contains("OPTIMAL").sum()) if status_col else None}
    p1 = load("_CANDIDATE_GENERATION_VALIDATION/candidate_generation_validation.json")
    out["P1"] = {"failed_checks": p1["failed_checks"], "verdicts": {k: v.get("status") if isinstance(v, dict) else v
                                                                    for k, v in (p1.get("verdicts") or {}).items()}}
    p1b = load("_OPERATIONAL_EVIDENCE_VALIDATION/operational_candidate_validation.json")
    out["P1B"] = {"failed_checks": p1b["failed_checks"], "eligibility_totals": p1b.get("eligibility_totals")}
    p1c = load("_TARGET_NEED_CONTRACT_VALIDATION/operational_need_contract_summary.json")
    out["P1C"] = {"failed_checks": p1c["failed_checks"], "eligibility": p1c.get("eligibility"),
                  "t1_bridge_real": (p1c.get("t1_bridge") or {}).get("A_REAL_ONLY")}
    scenarios = pd.read_csv(data_root / "_SELLER_LOSS_VALIDATION/seller_loss_scenario_validation.csv")
    pass_col = next((c for c in scenarios.columns if c.lower() in ("pass", "status", "passed")), None)
    out["SELLER_LOSS"] = {"scenarios": int(len(scenarios)),
                          "passed": int(scenarios[pass_col].astype(str).str.upper().isin(["TRUE", "PASS", "1"]).sum()) if pass_col else None}
    ext = load("09_External_Validation/results/forecast_external_generalization.json")
    gate = load("20_WALMART_M5/results/m5_forecast_v2_promotion_gate.json")
    out["FORECAST"] = {"external_verdict": ext.get("GENERALIZATION_VERDICT"), "m5_gate_passed": gate.get("passed"),
                       "m5_config_signature": gate.get("config_signature")}
    return out


def validation_git_guard(repo_root: Path) -> dict:
    # Preserve T4: only this specific tracked integration file may change.
    return inspect_validator_git(repo_root, "services/inventory_metric_integrity_validation.py",
                                 protect_all_tracked=True, allowed_tracked=ALLOWED_MODIFIED)


# ------------------------------------------------------------------------------------------------ run


def run_validation(data_root: Path, output_dir: Path | None = None, repo_root: Path | None = None,
                   include_external: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(data_root)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    output_dir = Path(output_dir or data_root / OUTPUT_FOLDER)
    before = oev._hashes(data_root, PROTECTED_BUNDLES)
    timings: dict[str, float] = {}

    def timed(name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        value = fn(*args, **kwargs)
        timings[name] = round(time.perf_counter() - t0, 1)
        return value

    inventory_fields = timed("field_inventory", field_inventory, repo_root)
    alias_paths = timed("loader_alias_paths", loader_alias_paths)
    scenarios = pd.DataFrame(timed("controlled_scenarios", controlled_scenarios))
    workbooks = timed("workbooks", run_workbooks, repo_root)
    proxy = timed("suhyup_flow_proxy", suhyup_flow_proxy, data_root)
    e2e = timed("suhyup_e2e", suhyup_e2e, data_root)
    days = timed("suhyup_days", suhyup_days, data_root)
    domestic = timed("domestic", domestic_sales, data_root)
    external = timed("external", external_sales, data_root) if include_external else {}
    golden = timed("golden_186", golden_186, data_root)
    t1 = timed("t1_rerun", t1_rerun, data_root)
    bundles = earlier_bundles(data_root)
    git_guard = validation_git_guard(repo_root)

    proxy_rows = proxy["rows"]
    e2e_rows = e2e["rows"]
    day_rows = days["rows"][days["rows"]["scope"] == "SUHYUP_DAY_UPLOAD"]
    undeclared_rows = days["rows"][days["rows"]["scope"] == "SUHYUP_DAY_UPLOAD_UNDECLARED"]
    shift = days["label_shift"]
    shifted = shift[shift["undeclared_varo_action"] != shift["saved_varo_action"]]
    label_shift = {"routes": len(shift), "declared_equals_saved": int((shift["declared_varo_action"] == shift["saved_varo_action"]).sum()),
                   "undeclared_differs_from_saved": len(shifted),
                   "transitions": dict(Counter(f"{a}->{b}" for a, b in zip(shifted["saved_varo_action"], shifted["undeclared_varo_action"]))),
                   "days_affected": int(shifted["date"].nunique())}
    all_rows = pd.concat([workbooks["rows"], e2e_rows, days["rows"]], ignore_index=True)
    actions = pd.concat([workbooks["actions"],
                         pd.DataFrame([{"scope": "SUHYUP_E2E_20260731", "dataset": "suhyup", "snapshot_date": "2026-07-31", **a}
                                       for a in e2e["field"]["action_dependency"]["rows"]]),
                         days["actions"]], ignore_index=True)

    # Suhyup: legacy single-day grade vs the 31-day flow proxy on the same (center, product, state).
    keyed = proxy_rows.set_index(["center_code", "product_id", "product_state"])
    compare = e2e_rows[["store_id", "product_id", "source_state_code", "legacy_turnover_grade", "legacy_sales_used",
                        "turnover_classification", "classification_confidence", "grade_cause"]].copy()
    compare["state"] = compare["source_state_code"].astype(str).str.zfill(2)
    joined = []
    for row in compare.itertuples(index=False):
        key = (str(row.store_id).lstrip("0") or "0", str(row.product_id), row.state)
        match = keyed.loc[key] if key in keyed.index else None
        joined.append({**row._asdict(), "outbound_31d": None if match is None else match["outbound_31d"],
                       "proxy_31d_classification": None if match is None else match["proxy_classification"],
                       "proxy_31d_turnover": None if match is None else match["turnover_value_31d"]})
    joined = pd.DataFrame(joined)
    dead_single_day = joined[joined["legacy_turnover_grade"] == "DEAD"]
    single_day_zero_but_moving = int(((dead_single_day["legacy_sales_used"] == 0) & (dead_single_day["outbound_31d"] > 0)).sum())

    # Missing vs zero (§27) per scope.
    def mz(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
        legacy_codes = frame["legacy_codes"].fillna("")
        return {"scope": scope, "rows": len(frame),
                "dead_from_missing_sales": int((frame["grade_cause"] == "MISSING_SALES_GRADED_DEAD").sum()),
                "observed_zero_sales_rows": int((frame["days_of_supply_status"] == imi.DOS_NO_MOVEMENT).sum()),
                "legacy_zero_sales_dead_rows": int(legacy_codes.str.contains("LEGACY_ZERO_SALES_DEAD").sum()),
                "negative_sales_anomaly_rows": int((frame["grade_cause"] == "NEGATIVE_SALES_GRADED_FAST").sum()),
                "zero_stock_graded_rows": int((frame["grade_cause"] == "ZERO_STOCK_GRADED").sum()),
                "average_inventory_zero_rows": int((frame["turnover_status"] == imi.DIVISION_BY_ZERO).sum()),
                "partial_period_rows": int((frame["turnover_status"] == imi.PARTIAL_PERIOD).sum()),
                "unit_mismatch_rows": int((frame["turnover_status"] == imi.UNIT_MISMATCH).sum()),
                "period_alias_rows": int((frame["grade_cause"] == "PERIOD_ALIAS_DISTORTION").sum()),
                "proxy_only_rows": int((frame["classification_scope"] == imi.SCOPE_PROXY).sum()),
                "unclassified_rows": int((frame["turnover_classification"] == imi.UNCLASSIFIED).sum()),
                "legacy_dead_rows": int((frame["legacy_turnover_grade"] == "DEAD").sum()),
                "legacy_fast_rows": int((frame["legacy_turnover_grade"] == "FAST").sum())}

    missing_zero = pd.DataFrame([mz(workbooks["rows"], "REPOSITORY_WORKBOOKS (synthetic)"),
                                 mz(e2e_rows, "SUHYUP_E2E_20260731"), mz(day_rows, "SUHYUP_31_DAY_UPLOADS (declared, as saved)"),
                                 mz(undeclared_rows, "SUHYUP_31_DAY_UPLOADS (no semantics column)")]
                                + [{"scope": "CONTROLLED_SCENARIOS", "rows": len(scenarios),
                                    "dead_from_missing_sales": int(scenarios["legacy"].fillna("").str.contains("grade DEAD").sum()),
                                    "negative_sales_anomaly_rows": int(scenarios["legacy"].fillna("").str.contains("sales_30d=-30").sum())}])
    missing_zero.loc[missing_zero["scope"] == "SUHYUP_E2E_20260731", "single_day_zero_dead_with_31d_outbound"] = single_day_zero_but_moving
    for name, item in external.items():
        totals, summary = item["totals"], item["summary"]
        missing_zero = pd.concat([missing_zero, pd.DataFrame([{
            "scope": f"{name.upper()} (sales only, no inventory)", "rows": totals.get("rows"),
            "observed_zero_sales_rows": totals.get("zero"), "negative_sales_anomaly_rows": totals.get("negative"),
            "absent_days_within_span": summary["absent_days_within_span"],
            "window_partial_series": summary["window_partial_series"], "window_complete_series": summary["window_complete_series"]}])],
            ignore_index=True)

    comparison = all_rows[["scope", "dataset", "snapshot_date", "store_id", "product_id", "sales_field_used",
                           "sales_signal_type", "sales_aggregation", "stock_qty", "legacy_turnover_days",
                           "legacy_turnover_rate_annual", "legacy_turnover_grade", "legacy_codes", "days_of_supply",
                           "days_of_supply_status", "turnover_value", "turnover_status", "turnover_basis",
                           "turnover_classification", "classification_confidence", "classification_scope", "gate_status",
                           "grade_cause", "abc_legacy_grade", "disposal_legacy_grade", "disposal_legacy_score",
                           "reason_codes"]]

    # Sales signal validation (§4-11, 35).
    signals = []
    p = proxy_rows
    signals.append({"dataset": "suhyup", "table": "canonical_inventory_flow", "quantity_column": "outbound_qty",
                    "signal_type": imi.OBSERVED_OUTBOUND_FLOW, "evidence": imi.SOURCE_DOCUMENTED, "date_grain": "daily",
                    "unit": None, "unit_status": "UNKNOWN (kg DIRECT in outbound_weight_kg)", "rows": int(p["flow_days"].sum()),
                    "series": len(p), "zero_rows": int(p["zero_outbound_days"].sum()), "negative_rows": 0,
                    "absent_days_within_window": int(p["no_record_days"].sum()), "zero_semantics": imi.ZERO_RECORDED,
                    "negative_semantics": imi.NO_NEGATIVES, "window": "2026-07-01..2026-07-31",
                    "window_complete_series": int((p["flow_days"] == 31).sum()),
                    "window_partial_series": int(((p["flow_days"] > 0) & (p["flow_days"] < 31)).sum()),
                    "gate_status": imi.PROXY_ONLY, "note": "logistics outbound, never retail sales; used only as a flow proxy"})
    e2e_fields = e2e["field"]["sales_fields"]
    for name, item in e2e_fields.items():
        signals.append({"dataset": "suhyup_e2e_upload", "table": "16_VARO_E2E_20260731/processed/inventory.csv (normalized)",
                        "quantity_column": name, "signal_type": imi.field_signal_type(name, item)[0],
                        "evidence": imi.field_signal_type(name, item)[1], "date_grain": item.get("resolved_aggregation", item["aggregation"]),
                        "rows": item["rows"], "zero_rows": item["zero_rows"], "negative_rows": item["negative_rows"],
                        "equal_to": "|".join(item["equal_to"]), "period_conflict": item["period_conflict"],
                        "gate_status": imi.PERIOD_MISMATCH if item["period_conflict"] else imi.PROXY_ONLY,
                        "note": "alias copy of one day's outbound" if item.get("copy_of_declared_daily") else item.get("declared_semantics")})
    for name, item in external.items():
        totals, summary, spec = item["totals"], item["summary"], item["spec"]
        signals.append({"dataset": name, "table": "canonical_demand_series", "quantity_column": "sales_qty",
                        "signal_type": spec["signal"], "evidence": imi.SOURCE_DOCUMENTED, "date_grain": "daily",
                        "unit": spec["unit"], "rows": totals.get("rows"), "series": summary["series"],
                        "null_rows": totals.get("null_qty"), "zero_rows": totals.get("zero"),
                        "negative_rows": totals.get("negative"), "documented_returns": totals.get("documented_return"),
                        "duplicate_flag_rows": totals.get("duplicate_flag"), "promotion_null_rows": totals.get("promotion_null"),
                        "absent_days_within_window": None, "absent_days_within_span": summary["absent_days_within_span"],
                        "series_with_absent_days": summary["series_with_absent_days"], "zero_semantics": spec["zero"],
                        "negative_semantics": spec["negative"] or imi.NO_NEGATIVES, "window": "..".join(summary["window"]),
                        "window_complete_series": summary["window_complete_series"],
                        "window_partial_series": summary["window_partial_series"],
                        "window_no_observation_series": summary["window_no_observation_series"],
                        "window_partial_volume_share": summary["window_partial_volume_share"],
                        "leading_zero_days_total": summary["leading_zero_days_total"],
                        "gate_status": imi.VALID_WITH_LIMITATION,
                        "note": json.dumps({k: v for k, v in totals.items() if k not in ("rows", "null_qty", "zero", "negative")},
                                           default=_json_default),
                        "contract_fidelity": all(f["match"] for f in item["fidelity"])})
    for name, info in (("logisall", domestic["logisall_sales"]), ("jangbogo", domestic["jangbogo_sales"])):
        # KADX specifications name the column 판매량 (sales) without stating the channel (retail vs B2B): a header claim.
        signals.append({"dataset": name, "table": "canonical_demand_series", "quantity_column": "sales_qty",
                        "signal_type": imi.OBSERVED_RETAIL_SALES, "evidence": imi.COLUMN_HEADER_CLAIM,
                        "date_grain": "daily (zone)" if name == "logisall" else "monthly (category)", "unit": None,
                        "rows": info["rows"], "zero_rows": info["zero_rows"], "negative_rows": info["negative_rows"],
                        "absent_days_within_span": info.get("absent_days_within_span"),
                        "gate_status": imi.VALID_WITH_LIMITATION, "note": json.dumps(info, ensure_ascii=False, default=_json_default)})
    signals.append({"dataset": "jangbogo", "table": "canonical_inventory_flow", "quantity_column": "order_qty",
                    "signal_type": imi.OBSERVED_ORDER, "evidence": imi.SOURCE_DOCUMENTED, "date_grain": "monthly",
                    "rows": domestic["jangbogo_orders"]["rows"], "negative_rows": domestic["jangbogo_orders"]["negative_rows"],
                    "gate_status": imi.NOT_COMPARABLE, "note": "purchase requests: not receipts, not demand, not an outflow"})
    signals.append({"dataset": "kamp", "table": "canonical_inventory_flow", "quantity_column": "shipment_qty",
                    "signal_type": imi.OBSERVED_OUTBOUND_FLOW, "evidence": imi.SOURCE_DOCUMENTED, "date_grain": "event",
                    "rows": domestic["kamp"]["rows"], "zero_rows": domestic["kamp"]["zero_rows"], "gate_status": imi.PROXY_ONLY,
                    "note": "order-based outbound shipments to one project; no inventory"})
    signals_frame = pd.DataFrame(signals)

    turnover_by_dataset = pd.DataFrame([
        {"dataset": "suhyup", "basis": imi.FLOW_BASED_TURNOVER_PROXY, "unit": "count (UNKNOWN, same source)",
         "keys": len(p), "status_counts": json.dumps(dict(Counter(p["turnover_status"]))),
         "classification_counts": json.dumps(dict(Counter(p["proxy_classification"]))),
         "confidence": json.dumps(dict(Counter(p["classification_confidence"]))),
         "real_sales_turnover_rows": 0, "note": "31-day outbound / mean of daily snapshots; outbound is not retail sales"},
        {"dataset": "suhyup", "basis": imi.FLOW_BASED_TURNOVER_PROXY, "unit": "kg (DIRECT)", "keys": len(p),
         "status_counts": json.dumps(dict(Counter(p["kg_turnover_status"]))),
         "classification_counts": json.dumps(dict(Counter(p["kg_proxy_classification"]))),
         "real_sales_turnover_rows": 0, "note": "same, in kg (unit DIRECT on both sides)"},
        {"dataset": "aihub", "basis": domestic["aihub"]["turnover_basis"], "unit": "item (SOURCE_METADATA)", "keys": 1,
         "status_counts": json.dumps({domestic["aihub"]["turnover_status"]: 1}),
         "classification_counts": json.dumps({domestic["aihub"]["proxy_classification"]: 1}), "real_sales_turnover_rows": 0,
         "turnover_value": domestic["aihub"]["turnover_value"], "note": "site aggregate over all products; "
                                                                          + _codes(domestic["aihub"]["reason_codes"])},
        *[{"dataset": name, "basis": imi.NO_BASIS, "unit": None, "keys": 0, "status_counts": json.dumps({imi.TURNOVER_UNAVAILABLE: "all"}),
           "real_sales_turnover_rows": 0, "note": note}
          for name, note in (("m5", "retail sales, no inventory: real turnover not computable"),
                             ("favorita", "retail sales, no inventory"), ("freshretailnet", "stock-out status is not inventory"),
                             ("logisall", "zone-daily sales vs national-monthly stock: scope differs (NOT_COMPARABLE)"),
                             ("jangbogo", "no inventory; purchase requests are orders"), ("nfqs", "stock only, no outflow"),
                             ("kamp", "shipments only, no inventory"))],
    ])

    abc_audit = pd.concat([workbooks["abc"], pd.DataFrame([{
        "scope": "SUHYUP_E2E_20260731", "dataset": "suhyup", **{k: (json.dumps(v) if isinstance(v, (dict, list, tuple)) else v)
                                                              for k, v in e2e["field"]["abc"].items() if k != "validated_grades"}}])],
        ignore_index=True)
    if "m5" in external:
        abc_audit = pd.concat([abc_audit, pd.DataFrame([{"scope": "M5_LAST_30_DAYS", "dataset": "m5",
                                                        "validated_basis": external["m5"]["abc"]["revenue_basis"],
                                                        **{k: json.dumps(v) if isinstance(v, dict) else v for k, v in external["m5"]["abc"].items()}}])],
                              ignore_index=True)
    disposal = all_rows[["scope", "dataset", "snapshot_date", "store_id", "product_id", "disposal_legacy_grade",
                         "disposal_legacy_score", "disposal_components", "disposal_codes"]].copy()
    disposal["meaning"] = imi.DISPOSAL_MEANING
    disposal["is_probability"] = False

    coverage = coverage_matrix(external, domestic, proxy_rows)
    downstream = pd.DataFrame(DOWNSTREAM, columns=["consumer", "module", "inputs", "problem_found", "contract_effect",
                                                   "production_link"])
    downstream["production_changed"] = False

    scen_pass = int(scenarios["pass"].sum())
    e2e_counts = e2e["field"]["counts"]
    action_counts = actions.groupby("scope")["turnover_dependency"].value_counts().unstack(fill_value=0)
    golden_ok = bool(golden["equal"].all()) and len(golden) == 186
    t2 = bundles["T2"]
    checks = [
        _check("F1", "field inventory: every listed field is referenced in services (except the two absent names)",
               bool((inventory_fields.loc[~inventory_fields["field"].str.startswith("sales_velocity"), "referencing_module_count"] > 0).all())),
        _check("F2", "loader alias replay equals normalize_loaded_data for every subset of sales columns",
               bool(alias_paths["replay_matches_loader"].all()), int((~alias_paths["replay_matches_loader"]).sum())),
        _check("F3", "the daily -> sales_30d alias path exists in the loader (sales_30d holds avg_daily_sales when only it is uploaded)",
               bool(((alias_paths["raw_columns"] == "avg_daily_sales") & (alias_paths["standard_field"] == "sales_30d")
                     & alias_paths["period_conflict"]).any())),
        _check("S1", "Suhyup E2E: sales_30d detected as a copy of one day's outbound; real-sales turnover rows 0",
               bool(e2e_fields.get("sales_30d", {}).get("copy_of_declared_daily")) and e2e_counts["real_sales_turnover_rows"] == 0),
        _check("S2", "Suhyup 31-day proxy: every key labelled FLOW_BASED_TURNOVER_PROXY / PROXY_ONLY",
               bool((proxy_rows["turnover_basis"] == imi.FLOW_BASED_TURNOVER_PROXY).all())
               and not bool(proxy_rows["gate_status"].isin([imi.VALID, imi.VALID_WITH_LIMITATION]).any())),
        _check("S3", "Suhyup canonical conservation: snapshot / flow sums equal the processed CSV",
               all(abs(a - b) < 1e-6 for a, b in proxy["conservation"].values())),
        _check("S4", "Suhyup 31 per-day uploads (declared as saved) reproduce the saved route ids and varo_action",
               bool(days["fidelity"]["same_varo_action"].all()) and bool(days["fidelity"]["same_route_ids"].all()),
               f"{int(days['fidelity']['same_varo_action'].sum())}/{len(days['fidelity'])}"),
        _check("S5", "Suhyup undeclared per-day uploads: no row is validated as sales (every row alias-flagged or zero-stock)",
               bool(undeclared_rows["grade_cause"].isin(["PERIOD_ALIAS_DISTORTION", "ZERO_STOCK_GRADED"]).all())
               and bool((undeclared_rows["classification_scope"] != imi.SCOPE_REAL).all()),
               dict(Counter(undeclared_rows["grade_cause"]))),
        _check("C1", f"controlled scenarios {scen_pass}/{len(scenarios)}", scen_pass == len(scenarios),
               scenarios.loc[~scenarios["pass"], "scenario"].tolist()),
        _check("X1", "external streamed window figures equal the contract aggregation on sampled series",
               all(all(f["match"] for f in item["fidelity"]) for item in external.values()) if external else True),
        _check("X2", "Favorita negatives kept (not clipped): negatives present and all carry documented_return",
               (not external) or (external["favorita"]["totals"]["negative"] > 0
                                  and external["favorita"]["totals"]["negative"] == external["favorita"]["totals"]["documented_return"])),
        _check("X3", "no real turnover produced for M5 / Favorita / FreshRetailNet (no inventory)",
               bool((turnover_by_dataset.loc[turnover_by_dataset["dataset"].isin(["m5", "favorita", "freshretailnet"]),
                                             "real_sales_turnover_rows"] == 0).all())),
        _check("R1", "Suhyup 186 reference rows reproduced", golden_ok, f"{int(golden['equal'].sum())}/{len(golden)}"),
        _check("R2", "T1 re-run: BENCHMARK_PROXY 7,750 / 13,629,146, 0 violations; STRICT selects 0",
               t1["benchmark_service"] == 7750 and abs(t1["benchmark_cost"] - 13629146) < 1e-6 and t1["benchmark_violations"] == 0
               and t1["strict_selected"] == 0, t1),
        _check("R3", "earlier bundles: T1/T2/T3/P1/P1-B/P1-C failed_checks empty; T2 620 candidates; P1-C A_all_valid 1,728/7",
               all(not bundles[k]["failed_checks"] for k in ("T1", "T2", "T3", "P1", "P1B", "P1C")) and t2["candidates"] == 620
               and (bundles["P1C"]["eligibility"] or {}).get("A_all_valid") == {"PROXY_FEASIBLE": 1728, "DEMAND_UNVERIFIED": 7}),
        _check("R4", "Seller Loss scenarios 18/18 in the saved bundle", bundles["SELLER_LOSS"]["scenarios"] == 18
               and bundles["SELLER_LOSS"]["passed"] == 18, bundles["SELLER_LOSS"]),
        _check("R5", "Forecast: external GENERALIZATION FAIL kept; M5 V2 gate passed",
               bundles["FORECAST"]["external_verdict"] == "FAIL" and bool(bundles["FORECAST"]["m5_gate_passed"])),
        git_validation_check(git_guard, f"tracked changes limited to {', '.join(ALLOWED_MODIFIED)}; validator/policy protected"),
    ]
    after = oev._hashes(data_root, PROTECTED_BUNDLES)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    checks.append(_check("G2", "no data / canonical / bundle file changed during the run", not changed, changed))
    failed = [c["check_id"] for c in checks if c["status"] != "PASS"]
    summary = {
        "contract_version": imi.CONTRACT_VERSION, "git_head": git_guard["actual_result"]["head"], "git_protection": git_guard,
        "data_root": str(data_root), "checks": checks, "failed_checks": failed,
        "signal_types": imi.SIGNAL_TYPES, "gate_statuses": imi.GATE_STATUSES, "turnover_bases": imi.TURNOVER_BASES,
        "turnover_statuses": list(imi.TURNOVER_STATUSES), "day_states": imi.DAY_STATES,
        "legacy_grade_source": imi.LEGACY_GRADE_SOURCE, "abc_thresholds": imi.abc_thresholds(),
        "suhyup": {"e2e_counts": e2e_counts, "e2e_abc": {k: v for k, v in e2e["field"]["abc"].items() if k != "validated_grades"},
                   "e2e_disposal": e2e["field"]["disposal_risk"], "e2e_safety_stock": e2e["field"]["safety_stock"],
                   "e2e_match": e2e["field"]["store_product_match"], "e2e_forecast_inputs": e2e["field"]["forecast_inputs"],
                   "e2e_action_dependency": {"counts": e2e["field"]["action_dependency"]["counts"],
                                             "disposal": e2e["field"]["action_dependency"]["disposal_counts"],
                                             "varo_action": dict(Counter(a["varo_action"] for a in e2e["field"]["action_dependency"]["rows"]))},
                   "single_day_zero_dead_with_31d_outbound": single_day_zero_but_moving,
                   "legacy_dead_rows_e2e": int((e2e_rows["legacy_turnover_grade"] == "DEAD").sum()),
                   "day_uploads": {"rows": len(day_rows), "grade_causes": dict(Counter(day_rows["grade_cause"])),
                                   "legacy_grades": dict(Counter(day_rows["legacy_turnover_grade"])),
                                   "validated": dict(Counter(f"{a}->{b}" for a, b in zip(day_rows["legacy_turnover_grade"],
                                                                                         day_rows["turnover_classification"]))),
                                   "fidelity_days": int(days["fidelity"]["same_varo_action"].sum())},
                   "undeclared_day_uploads": {"rows": len(undeclared_rows),
                                              "grade_causes": dict(Counter(undeclared_rows["grade_cause"])),
                                              "legacy_grades": dict(Counter(undeclared_rows["legacy_turnover_grade"]))},
                   "alias_effect_on_varo_action": label_shift,
                   "flow_proxy": {"keys": len(proxy_rows), "status": dict(Counter(proxy_rows["turnover_status"])),
                                  "classification": dict(Counter(proxy_rows["proxy_classification"])),
                                  "identity_checked_days": int(proxy_rows["stock_flow_identity_checked_days"].sum()),
                                  "identity_mismatch_days": int(proxy_rows["stock_flow_identity_mismatch_days"].sum()),
                                  "conservation": proxy["conservation"]},
                   "proxy_vs_legacy": dict(Counter(f"{a}->{b}" for a, b in zip(joined["legacy_turnover_grade"],
                                                                               joined["proxy_31d_classification"])))},
        "workbooks": workbooks["summaries"].to_dict("records"),
        "action_dependency": {scope: {k: int(v) for k, v in row.items()} for scope, row in action_counts.iterrows()},
        "external": {name: {"totals": item["totals"], "summary": item["summary"], "fidelity": item["fidelity"],
                            **({"abc": item["abc"]} if "abc" in item else {}),
                            **({"items_with_noninteger_sales": item["items_with_noninteger_sales"], "items": item["items"]}
                               if "items" in item else {}), "seconds": item["seconds"]}
                     for name, item in external.items()},
        "domestic": domestic, "regressions": {"golden_186": f"{int(golden['equal'].sum())}/{len(golden)}", "t1_rerun": t1,
                                              "earlier_bundles": bundles},
        "readiness": {
            "sales_semantics_contract": "READY" if not set(failed) - {"G1"} else "NOT_READY",
            "turnover_validation": "READY_WITH_LIMITATIONS (no dataset has observed retail sales and inventory together)",
            "real_sales_turnover_on_real_data": "UNSUPPORTED (no dataset)",
            "flow_based_proxy": "AVAILABLE_PROXY_ONLY (Suhyup, AI Hub aggregate)",
            "abc_revenue_basis": "VERIFIED_ON_M5_ONLY", "disposal_probability": "UNSUPPORTED (no disposal history)",
            "production_replacement": "NOT_READY (parallel field only; replacing legacy grades needs approval and real data)"},
        "limitations": [
            "no dataset combines observed retail sales with observed inventory: real-sales turnover is verified on controlled "
            "scenarios only",
            "upload fields carry no record of the loader alias; copies are detected by exact equality (direction is only "
            "resolved by a declared semantics column)",
            "a sales column header is treated as a claim (SALES_CLAIMED_BY_HEADER), not as verified retail sales",
            "the legacy 30/60/90 thresholds are reused unchanged and are not operationally validated",
            "the Suhyup stock / flow identity does not hold on every day; outbound is not the only stock movement",
        ],
        "production_action_applied": False, "legacy_metrics_replaced": False,
        "runtime": {"wall_seconds": round(time.perf_counter() - started, 1), "sections": timings},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write = lambda frame, name: frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")  # noqa: E731
    write(inventory_fields, OUTPUT_FILES[0])
    write(signals_frame, OUTPUT_FILES[1])
    write(pd.DataFrame(TURNOVER_FORMULA_AUDIT, columns=["element", "legacy_turnover_analyzer", "validated_contract", "difference"]),
          OUTPUT_FILES[2])
    write(missing_zero, OUTPUT_FILES[3])
    write(turnover_by_dataset, OUTPUT_FILES[4])
    write(abc_audit, OUTPUT_FILES[5])
    write(disposal, OUTPUT_FILES[6])
    write(coverage, OUTPUT_FILES[7])
    (output_dir / OUTPUT_FILES[8]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
                                              encoding="utf-8")
    write(alias_paths, OUTPUT_FILES[9])
    write(scenarios, OUTPUT_FILES[10])
    write(comparison, OUTPUT_FILES[11])
    write(actions.assign(**{c: actions[c].astype(str) for c in actions.columns if actions[c].dtype == object}), OUTPUT_FILES[12])
    write(downstream, OUTPUT_FILES[13])
    write(proxy_rows.merge(joined.rename(columns={"store_id": "center_code_e2e"})[["center_code_e2e", "product_id", "state",
                                                                                   "legacy_turnover_grade", "grade_cause"]],
                           left_on=["center_code", "product_id", "product_state"],
                           right_on=["center_code_e2e", "product_id", "state"], how="left"), OUTPUT_FILES[14])
    all_rows.astype({c: str for c in all_rows.columns if all_rows[c].dtype == object}).to_parquet(output_dir / OUTPUT_FILES[15], index=False)
    write(shift, OUTPUT_FILES[16])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-external", action="store_true", help="skip the M5 / Favorita / FreshRetailNet streams")
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir, include_external=not args.skip_external)
    print(json.dumps({"failed_checks": result["failed_checks"], "checks": result["checks"],
                      "readiness": result["readiness"], "runtime": result["runtime"]},
                     ensure_ascii=False, indent=2, default=_json_default))
    return 0 if not result["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
