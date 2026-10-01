"""External generalisation check of the frozen production demand-forecast router on Favorita and FreshRetailNet.

Nothing is fitted, selected or tuned here. The V2 configuration, routing / cold-start / mostly_zero rules, demand-type
definition and the M5 gates are the frozen ones (verified against FROZEN_REFERENCE before and after every run). The
protocol, the quantity-semantics rule and the external gate below were written before any external series was scored;
``freeze_gate`` records their signature in ``forecast_external_gate_frozen.json`` before the first scoring and every
later run refuses a different one.

Per dataset and forecast origin the production path is exercised exactly as the analysis pipeline does it:

    canonical demand_series --services.demand_history_bridge (as_of = cutoff)--> daily_sales_history
    inventory rows (recorded sales of the last 7 / 30 days, snapshot_date = cutoff) --V1 production call path--> V1
    services.demand_forecast_router.route_demand_forecast(V1 output, daily_sales_history) --> routed forecast

next to the research V2 Core (every series), naive / seasonal naive / moving-average baselines and the observed sales
of the 7 target days. A target day counts only when observed (a row exists); a missing day is never 0.

Dataset-specific code is limited to reading files, mapping columns and the dated protocol (``load_*`` / ``*_protocol``);
the engine (``build_panel`` .. ``score``) and the router never see a dataset name, item family or store id.

    python -m services.external_forecast_validation --dataset freshretailnet
    python -m services.external_forecast_validation --dataset favorita
    python -m services.external_forecast_validation --summary
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import shutil
import subprocess
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from services import demand_forecast_router as router
from services import demand_forecast_v2 as core
from services.demand_history_bridge import canonical_to_daily_history, series_boundaries
from services.m5_forecast_validation import AGE_BANDS, EXACT_TOLERANCE, _clean, _git_commit, _write_csv, finalize_metrics, peak_memory_mb
from services.real_data_adapters import DATA_ROOT, DATASETS

EVALUATION_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON = core.OPERATIONAL_HORIZON
DAY = np.timedelta64(1, "D")
V1, ROUTED, V2 = "varo_v1_production", "varo_router_production", "varo_v2_core"
BASELINES = ("naive_last", "seasonal_naive_7", "moving_average_7", "moving_average_14", "moving_average_28")
METHODS = (V1, ROUTED, V2, *BASELINES)
METHOD_ROLES = {V1: "production_v1", ROUTED: "production_router", V2: "v2_core_direct", **{b: "baseline" for b in BASELINES}}
METHOD_DEFINITIONS = {
    V1: "FORECAST_V1_BASELINE through its production call path (prepare_legacy_data -> analyze_demand_forecast) on inventory rows "
        "whose sales_7d / sales_30d are the recorded sales of the last 7 / 30 days up to the cutoff, avg_daily_sales = sales_30d / 30, "
        "demand_std = std of the recorded days of the last 30 (ddof 1); daily vector = flat demand_forecast_daily",
    ROUTED: "services.demand_forecast_router.route_demand_forecast(V1 output, bridged daily history up to the cutoff): frozen V2 where "
            "eligible, V1 elsewhere; 7d = demand_forecast_7d, daily vector = demand_forecast_d1..d7",
    V2: "frozen V2 Core (router.v2_config) on every series' day matrix up to the cutoff, groups store x category; 7d = round(aggregate, 1)",
    "naive_last": "last observed day repeated (0 when none)",
    "seasonal_naive_7": "the same weekday of the last 7 days; a day without a row takes the observed mean of those 7 days",
    "moving_average_7": "mean of the observed days among the last 7 calendar days, flat (0 when none)",
    "moving_average_14": "mean of the observed days among the last 14 calendar days, flat (0 when none)",
    "moving_average_28": "mean of the observed days among the last 28 calendar days, flat (0 when none)",
}
CANONICAL_COLUMNS = ("source_dataset", "source_file", "source_row_id", "date_grain", "product_grain", "date", "location_id", "product_id",
                     "category", "sales_qty", "unit", "unit_status", "validation_flags", "location_id_namespace", "product_id_namespace",
                     "stockout_hours")
COMMON_RESULTS = Path("09_External_Validation") / "results"
GATE_FREEZE_FILE = "forecast_external_gate_frozen.json"
GENERALIZATION_FILE = "forecast_external_generalization.json"
SUPPORT_MATRIX_FILE = "forecast_daily_history_support_matrix.csv"
FILE_PREFIX = {"favorita": "favorita", "freshretailnet": "freshretail"}
RESULT_SUFFIXES = {"summary": "_forecast_external_summary.csv", "by_cutoff": "_forecast_by_cutoff.csv",
                   "by_demand_type": "_forecast_by_demand_type.csv", "coverage": "_forecast_router_coverage.csv",
                   "bias": "_forecast_bias.csv", "metrics": "_forecast_external_metrics.csv", "json": "_forecast_external.json",
                   "rows": "_forecast_external_rows.parquet"}


class FreezeViolation(RuntimeError):
    """The run was asked to use a gate, protocol or frozen V2 configuration other than the recorded one."""


# ---------------------------------------------------------------- frozen V2 / router configuration (read, never changed)

FROZEN_REFERENCE: dict[str, Any] = {
    "v2_config_signature": "8109536223f4ed0a0d2842813f89381ab9e5f9288e937c7d155909a7bb120b69",
    "v2_routes": {"high_frequency": {"level": "ewma_w0.4", "weekday": "pooled_8w"},
                  "medium_frequency": {"level": "ewma_w0.4", "weekday": "pooled_8w"},
                  "intermittent": {"level": "ewma_w0.4", "weekday": "shrunk_8w"},
                  "mostly_zero": {"level": "ma28", "weekday": "shrunk_8w"}},
    "router": {"version": "1.0.0", "min_sales_age_days": 30, "recent_observed_days": 30, "mostly_zero_policy": "v1",
               "reason_order": ["missing_daily_history", "invalid_history", "negative_sales", "no_sales_history", "cold_start",
                                "insufficient_history", "mostly_zero_safety", "v2_error_fallback", "sufficient_daily_history"]},
    "demand_type": {"types": ["high_frequency", "medium_frequency", "intermittent", "mostly_zero", "no_sales_history"],
                    "profile_window_days": 364, "high_share": 1 / 1.32, "medium_share": 0.5, "intermittent_share": 0.1},
    "m5_promotion_gate_signature": "c53a077c7af852a62c97a72af6d600b6f5e342225077256f71ce89f124493746",
    "m5_mostly_zero_rule_signature": "6c60fc6eb6ec5f1850916b0bdc4ccf5519970a55028a3ef6c84862286e503e75",
}
FROZEN_SOURCES = ("services/demand_forecast_router.py", "services/demand_forecast_v2.py",
                  "services/legacy_adapters/_local_modules/demand_forecast_analyzer.py")


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def frozen_config_record() -> dict[str, Any]:
    """The frozen configuration as the code currently defines it, plus source fingerprints as evidence."""
    from services.m5_forecast_router_validation import rule_signature
    from services.m5_forecast_v2_validation import gate_signature as m5_gate_signature

    config = core.make_config(router.V2_ROUTES)
    try:
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *FROZEN_SOURCES], cwd=PROJECT_ROOT, timeout=30).returncode
        sources_match_head = diff == 0
    except (OSError, subprocess.SubprocessError):
        sources_match_head = None
    return {
        "v2_config_signature": core.config_signature(config),
        "v2_config_signature_pinned_in_router": router.V2_CONFIG_SIGNATURE,
        "v2_routes": json.loads(json.dumps(router.V2_ROUTES)),
        "router": {"version": router.ROUTER_VERSION, "min_sales_age_days": router.MIN_SALES_AGE_DAYS,
                   "recent_observed_days": router.RECENT_OBSERVED_DAYS, "mostly_zero_policy": router.MOSTLY_ZERO_POLICY,
                   "reason_order": list(router.REASON_CODES)},
        "demand_type": {"types": list(core.DEMAND_TYPES), "profile_window_days": core.PROFILE_WINDOW, "high_share": core.HIGH_SHARE,
                        "medium_share": core.MEDIUM_SHARE, "intermittent_share": core.INTERMITTENT_SHARE},
        "m5_promotion_gate_signature": m5_gate_signature(),
        "m5_mostly_zero_rule_signature": rule_signature(),
        "source_sha256_lf": {name: _sha256_lf(PROJECT_ROOT / name) for name in FROZEN_SOURCES},
        "frozen_sources_match_git_head": sources_match_head,
        "v1_fingerprint_matches_baseline": core.v1_baseline_fingerprint()["source_sha256_lf"] == core.FORECAST_V1_BASELINE["source_sha256_lf"],
    }


def frozen_config_check(record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """FROZEN_CONFIG_UNCHANGED: every frozen value equals FROZEN_REFERENCE and the sources equal git HEAD."""
    record = frozen_config_record() if record is None else record
    differences = [key for key, value in FROZEN_REFERENCE.items() if json.loads(json.dumps(record[key])) != json.loads(json.dumps(value))]
    if record["v2_config_signature_pinned_in_router"] != FROZEN_REFERENCE["v2_config_signature"]:
        differences.append("v2_config_signature_pinned_in_router")
    if record["frozen_sources_match_git_head"] is False:
        differences.append("frozen_sources_match_git_head")
    if not record["v1_fingerprint_matches_baseline"]:
        differences.append("v1_fingerprint_matches_baseline")
    return {"FROZEN_CONFIG_UNCHANGED": not differences, "differences": differences, "record": record}


# ---------------------------------------------------------------- pre-registered protocol, semantics rule and gate

PROTOCOLS: dict[str, Any] = {
    "version": "1.0.0",
    "common": {
        "split": "chronological origins only; no random split; nothing fitted, selected or tuned (no parameter selection exists here)",
        "history": "every canonical row up to the cutoff (full history, no truncation), through the production bridge with as_of = cutoff",
        "universe": "every series with at least one history row up to the cutoff (one inventory row per series)",
        "scoring": ("a target day counts only when observed (a row exists); total_7d counts only when all 7 target days are observed; "
                    "a missing day is never 0"),
        "horizon_days": HORIZON,
        "processing": "series are processed in store batches; the router's pooled weekday groups are store x category, so batches never split a group",
    },
    "freshretailnet": {
        "final": {"cutoff": "last day of the train file", "targets": "the 7 days of the eval file", "role": "official final test (gated)"},
        "development": {"cutoffs": "final cutoff - 21, - 14 and - 7 days (inside train)", "targets": "the next 7 train days",
                        "role": "stability report only; selects nothing"},
        "train_file": "05_FreshRetailNet/raw/train.parquet", "eval_file": "05_FreshRetailNet/raw/eval.parquet",
        "primary_scope": "final"},
    "favorita": {
        "external_test": {"cutoffs": "last data day - 28, - 21, - 14 and - 7 days", "targets": "the 7 days after each cutoff (together the last 28 days)",
                          "role": "4 rolling origins, gated pooled"},
        "primary_scope": "external_test_pooled"},
}

SEMANTICS_CRITERIA: dict[str, str] = {
    "observed_sales": "the quantity is an observed per-day sale of a location x product, not a proxy (shipments, outbound, purchase requests)",
    "daily_location_product": "daily grain with location and product keys",
    "physical_unit": "a known physical unit (count or weight): not UNKNOWN and not a normalised amount",
    "zero_days_recorded": "observed zero-sale days are recorded as 0, so a day without a row is a missing observation",
    "unmodified_meaning": "no documented change of the sales meaning (returns netted into sales, stock-out censoring, normalisation)",
}
SEMANTICS_RULE: dict[str, str] = {
    "FULL": "every criterion holds",
    "PARTIAL": "observed_sales and daily_location_product hold; at least one other criterion fails",
    "UNAVAILABLE": "observed_sales or daily_location_product fails",
}

EXTERNAL_GATE: dict[str, Any] = {
    "version": "1.0.0",
    "baseline": V1,
    "candidate": ROUTED,
    "integrity": {
        "frozen_config": "FROZEN_CONFIG_UNCHANGED before and after the run (signature, routes, router policy, demand types, M5 gates, sources = git HEAD)",
        "no_leakage": ("every history row handed to the router is dated on or before its cutoff; the bridge excluded exactly the rows dated "
                       "after the cutoff; the router read no future row; given the full history (future rows included) the router drops "
                       "them itself and returns the same forecast"),
        "fallback_policy": ("V1-routed rows equal V1 exactly; V2-routed rows equal the research V2 Core on the same history (7-day total and "
                            "daily vector); every reason equals router.assess_matrix on that history; every series with a negative value up "
                            "to the cutoff is routed V1 (negative_sales, or invalid_history when invalid); no v2_error_fallback"),
        "train_eval_separation": "FreshRetailNet: the final origin's history comes only from the train file and its targets only from the eval file",
    },
    "safety": {
        "total_7d_wape": {"window": "total_7d", "metric": "wape", "max_relative_degradation": 0.01},
        "daily_wape": {"window": "daily_h1_7", "metric": "wape", "max_relative_degradation": 0.01},
        "bias": {"window": "total_7d", "rule": "|bias_pct router| <= max(|bias_pct V1|, floor)", "floor": 0.02},
        "demand_type": {"window": "total_7d", "metric": "wape", "segments": ["high_frequency", "medium_frequency"],
                        "max_relative_degradation": 0.03,
                        "applies_when": {"min_scored_series_weeks": 100, "min_actual_volume_share": 0.01}},
    },
    "improvement": {"window": "total_7d", "metric": "wape", "min_relative_improvement": 0.01,
                    "consistency": "router WAPE < V1 WAPE on a strict majority of the primary origins"},
    "coverage": {"metric": "V2-routed share of the scored total_7d actual volume in the primary scope", "min_share": 0.20},
    "dataset_verdict": {
        "FAIL": "any integrity or safety criterion fails",
        "PASS": "no failure; improved; V2 coverage >= min_share; quantity semantics FULL",
        "PASS_WITH_LIMITATION": "no failure, but not improved, or V2 coverage < min_share, or quantity semantics PARTIAL (each listed)"},
    "combined_verdict": {
        "FAIL": "any dataset FAIL, or no dataset improved",
        "GENERALIZATION_PASS": "no dataset FAIL and every dataset improved",
        "PARTIAL": "no dataset FAIL; at least one dataset improved and at least one did not"},
    "rationale": {
        "safety_1pct": ("The router may not make the production forecast measurably worse: 1% relative WAPE is below the M5 origin-to-origin "
                        "noise of a pooled 3-origin result (about 1.5%), so a larger loss is a regression, not noise."),
        "bias": "The frozen M5 promotion-gate bias rule, unchanged: orders follow the forecast, so the router may not add bias beyond a +-2% band.",
        "demand_type": ("High and medium frequency carry most of the volume; the M5 gate's 3% demand-type safety threshold applies. Segments "
                        "under 100 scored series-weeks or 1% of the volume are reported, not gated (too noisy)."),
        "improvement": ("Improved = at least 1% lower pooled 7-day WAPE than V1 and better on most origins. Half of the M5 promotion threshold, "
                        "because the router changes only the V2-eligible part of a dataset; smaller differences are neutral."),
        "coverage": "Below a fifth of the scored volume the dataset-level comparison mostly measures V1 against itself.",
        "semantics": "PARTIAL quantity semantics (unknown or normalised unit, absent zero days, netted returns, censoring) limit what a PASS means.",
    },
}

DIMENSIONS: dict[str, tuple[str, ...]] = {
    "overall": ("all",),
    "demand_type": tuple(core.DEMAND_TYPES),
    "router_reason": tuple(router.REASON_CODES),
    "router_version": (router.VERSION_V1, router.VERSION_V2),
    "age_band": tuple(AGE_BANDS),
    "negative_history": ("no_negative", "negative_in_history"),
    "first_row_age": ("first_row_ge_30d_before_cutoff", "first_row_lt_30d_before_cutoff"),
    "target_censoring": ("no_stockout_in_target_week", "stockout_in_target_week", "censoring_unknown"),
}
WINDOWS = ("total_7d", "daily_h1_7")
STATS = ("n", "abs_err", "sq_err", "err", "forecast", "actual", "over_n", "under_n", "over_units", "under_units", "missed_demand_n", "scored")


def gate_signature() -> str:
    """Protocol, semantics rule, gate constants AND the code applying them: an edit after the freeze is detected."""
    payload = {"protocols": PROTOCOLS, "semantics_criteria": SEMANTICS_CRITERIA, "semantics_rule": SEMANTICS_RULE, "gate": EXTERNAL_GATE,
               "dimensions": DIMENSIONS, "methods": METHOD_DEFINITIONS,
               "code": [inspect.getsource(f) for f in (classify_semantics, evaluate_dataset_gate, combined_verdict, window_stats)]}
    return core.config_signature(json.loads(json.dumps(payload)))


def freeze_gate(common_dir: Path) -> dict[str, Any]:
    """Write the gate record before any scoring; afterwards refuse a run whose gate/protocol/frozen config differ."""
    common_dir.mkdir(parents=True, exist_ok=True)
    path = common_dir / GATE_FREEZE_FILE
    check = frozen_config_check()
    if not check["FROZEN_CONFIG_UNCHANGED"]:
        raise FreezeViolation(f"frozen V2/router configuration changed: {check['differences']}")
    current = {"gate_signature": gate_signature(), "v2_config_signature": FROZEN_REFERENCE["v2_config_signature"]}
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if {k: frozen.get(k) for k in current} != current:
            raise FreezeViolation(f"{path.name} holds another gate or configuration: {frozen.get('gate_signature')} != {current['gate_signature']}")
        return frozen
    record = {**current, "frozen_at": _now(), "git_commit": _git_commit(), "evaluation_version": EVALUATION_VERSION,
              "protocols": PROTOCOLS, "semantics_criteria": SEMANTICS_CRITERIA, "semantics_rule": SEMANTICS_RULE, "gate": EXTERNAL_GATE,
              "frozen_config": FROZEN_REFERENCE,
              "note": "Written before any external series was scored; results never change these values."}
    path.write_text(json.dumps(_clean(record), ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- generic engine: panel, inputs, baselines


@dataclass
class Panel:
    """Series x calendar-day view of one batch's bridged history (all dates; the engine slices it per cutoff)."""
    keys: pd.DataFrame            # store_id, product_id, category (bridge series order)
    day0: np.datetime64
    quantity: np.ndarray          # (n, T) observed quantity; NaN = no row, an invalid quantity or a repeated day
    present: np.ndarray           # (n, T) a history row exists
    censored: np.ndarray | None   # (n, T) 1 stock-out hours > 0, 0 none, NaN unknown / no row
    source_code: np.ndarray       # (n, T) index into source_files, -1 = no row
    source_files: list[str]
    history: pd.DataFrame         # full bridged history (every date), production contract
    bridge_report: dict[str, Any]
    series_meta: pd.DataFrame
    late_quarantine_dates: np.ndarray   # dates of rows quarantined only for their quantity or unit (NULL, mixed unit)
    history_code: np.ndarray            # panel row of every full-history row


def build_panel(canonical: pd.DataFrame, first_day: np.datetime64, last_day: np.datetime64) -> Panel:
    full = canonical_to_daily_history(canonical, as_of=None, keep_row_lineage=True)
    history = full.history
    code, starts, repeated = series_boundaries(history)
    n, width = len(starts), int((last_day - first_day) / DAY) + 1
    column = ((history["date"].to_numpy().astype("datetime64[D]") - first_day) / DAY).astype(np.int64)
    if len(column) and (column.min() < 0 or column.max() >= width):
        raise ValueError("history dates fall outside the dataset day range")
    quantity = np.full((n, width), np.nan)
    present = np.zeros((n, width), dtype=bool)
    q = history["quantity"].to_numpy(dtype=np.float64)
    quantity[code, column] = q
    present[code, column] = True
    quantity[code[repeated], column[repeated]] = np.nan          # a repeated day is neither summed nor picked
    files, source_values = pd.factorize(history["source_file"]) if "source_file" in history else (np.zeros(len(history), int), ["?"])
    source_code = np.full((n, width), -1, dtype=np.int16)
    source_code[code, column] = files
    censored = None
    if "stockout_hours" in canonical.columns and canonical["stockout_hours"].notna().any():
        hours = pd.to_numeric(canonical["stockout_hours"], errors="coerce").to_numpy(dtype=np.float64)[history["input_position"].to_numpy()]
        censored = np.full((n, width), np.nan)
        censored[code, column] = np.where(np.isnan(hours), np.nan, (hours > 0).astype(np.float64))
    keys = pd.DataFrame({"store_id": history["store_id"].to_numpy(dtype=object)[starts],
                         "product_id": history["product_id"].to_numpy(dtype=object)[starts],
                         "category": full.series["category"].to_numpy(dtype=object)})
    late = full.quarantine[full.quarantine["reason"].isin(["missing_quantity", "mixed_unit_series"])]
    late_dates = pd.to_datetime(late["raw_date"].astype("string"), format="ISO8601", errors="coerce").to_numpy()
    return Panel(keys, first_day, quantity, present, censored, source_code, [str(s) for s in source_values], history, full.report,
                 full.series, late_dates, code)


def inventory_rows(keys: pd.DataFrame, history: np.ndarray, cutoff: np.datetime64) -> pd.DataFrame:
    """Inventory demand fields from the recorded daily sales (sums over recorded rows; a day without a row adds nothing)."""
    recent, month = history[:, -7:], history[:, -30:]
    observed = (~np.isnan(month)).sum(axis=1)
    sales_7d, sales_30d = np.nansum(recent, axis=1), np.nansum(month, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.where(observed >= 2, np.nanstd(month, axis=1, ddof=1), np.nan)
    return pd.DataFrame({"store_id": keys["store_id"].to_numpy(dtype=object), "product_id": keys["product_id"].to_numpy(dtype=object),
                         "category": keys["category"].to_numpy(dtype=object), "product_name": keys["product_id"].to_numpy(dtype=object),
                         "sales_7d": sales_7d, "avg_daily_sales": sales_30d / 30, "sales_30d": sales_30d, "demand_std": std,
                         "snapshot_date": str(cutoff)})


def observed_mean(history: np.ndarray, window: int) -> np.ndarray:
    tail = history[:, -window:]
    count = (~np.isnan(tail)).sum(axis=1)
    return np.divide(np.nansum(tail, axis=1), count, out=np.zeros(len(history)), where=count > 0)


def last_observed(history: np.ndarray) -> np.ndarray:
    observed = ~np.isnan(history)
    last = history.shape[1] - 1 - np.argmax(observed[:, ::-1], axis=1)
    return np.where(observed.any(axis=1), history[np.arange(len(history)), last], 0.0)


def baseline_forecasts(history: np.ndarray, horizon: int = HORIZON) -> dict[str, np.ndarray]:
    def flat(level):
        return np.repeat(level[:, None], horizon, axis=1)
    week = history[:, -7:]
    seasonal = np.where(np.isnan(week), observed_mean(history, 7)[:, None], week)[:, np.arange(horizon) % 7]
    return {"naive_last": flat(last_observed(history)), "seasonal_naive_7": seasonal, "moving_average_7": flat(observed_mean(history, 7)),
            "moving_average_14": flat(observed_mean(history, 14)), "moving_average_28": flat(observed_mean(history, 28))}


def window_stats(forecast: np.ndarray, actual: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Additive per-series error sums over the valid points; an invalid (missing) point contributes nothing."""
    err = np.where(valid, forecast - np.nan_to_num(actual), 0.0)
    return {"n": valid.sum(axis=1).astype(np.float64), "abs_err": np.abs(err).sum(axis=1), "sq_err": (err * err).sum(axis=1),
            "err": err.sum(axis=1), "forecast": np.where(valid, forecast, 0.0).sum(axis=1),
            "actual": np.where(valid, np.nan_to_num(actual), 0.0).sum(axis=1),
            "over_n": (err > EXACT_TOLERANCE).sum(axis=1).astype(np.float64), "under_n": (err < -EXACT_TOLERANCE).sum(axis=1).astype(np.float64),
            "over_units": np.clip(err, 0.0, None).sum(axis=1), "under_units": np.clip(-err, 0.0, None).sum(axis=1),
            "missed_demand_n": (valid & (forecast <= EXACT_TOLERANCE) & (np.nan_to_num(actual) > 0)).sum(axis=1).astype(np.float64),
            "scored": valid.any(axis=1).astype(np.float64)}


def _pool_codes(stores: Iterable[Any], categories: Iterable[Any]) -> np.ndarray:
    """The router's weekday pooling groups: store x category text as router.assess_history builds them."""
    labels = [f"{router._key(s)}|{router._key(c) or ''}" for s, c in zip(stores, categories)]
    return pd.factorize(pd.Series(labels, dtype=object), sort=True)[0].astype(np.int64)


# ---------------------------------------------------------------- generic engine: one origin of one batch


@dataclass
class OriginOutput:
    sums: dict[tuple[str, str, str], np.ndarray]     # (method, window, dimension) -> (segments x STATS)
    rows: pd.DataFrame
    checks: dict[str, Any]
    negative: dict[str, Any]
    seconds: dict[str, float]


def evaluate_origin(panel: Panel, canonical: pd.DataFrame, cutoff: np.datetime64, config: Mapping[str, Any], *,
                    self_exclusion_check: bool = False) -> OriginOutput:
    """One forecast origin of one store batch through the production path, the research V2 and the baselines."""
    seconds: dict[str, float] = {}
    ci = int((cutoff - panel.day0) / DAY)
    universe = np.flatnonzero(panel.present[:, :ci + 1].any(axis=1))
    keys = panel.keys.iloc[universe].reset_index(drop=True)
    history = panel.quantity[universe, :ci + 1]
    present = panel.present[universe, :ci + 1]

    started = time.perf_counter()
    bridged = canonical_to_daily_history(canonical, as_of=cutoff, keep_row_lineage=False)
    seconds["bridge"] = time.perf_counter() - started
    inventory = inventory_rows(keys, history, cutoff)
    started = time.perf_counter()
    v1_out = core.run_v1_baseline(inventory)
    seconds["v1"] = time.perf_counter() - started
    started = time.perf_counter()
    routed, diagnostics = router.route_demand_forecast(v1_out, bridged.history)
    seconds["router"] = time.perf_counter() - started

    reason = routed["demand_forecast_reason"].to_numpy(dtype=object)
    is_v2 = reason == router.REASON_V2
    v1_total = v1_out["demand_forecast_7d"].to_numpy(dtype=np.float64)
    v1_daily = np.repeat(v1_out["demand_forecast_daily"].to_numpy(dtype=np.float64)[:, None], HORIZON, axis=1)
    routed_total = routed["demand_forecast_7d"].to_numpy(dtype=np.float64)
    routed_daily = routed[list(router.DAILY_COLUMNS)].to_numpy(dtype=np.float64)

    # research V2 on exactly the router's matrix (usable rows, first history day .. cutoff, same pooling groups)
    started = time.perf_counter()
    usable = ~np.isin(reason, [router.REASON_MISSING, router.REASON_INVALID, router.REASON_NEGATIVE, router.REASON_ERROR])
    exact = {"v2_rows_7d_equal_research": True, "v2_rows_daily_equal_research": True, "reasons_equal_router_assessment": True}
    if usable.any():
        start = int(np.argmax(present[usable], axis=1).min())
        matrix = history[usable, start:]
        groups = _pool_codes(routed["store_id"].to_numpy(dtype=object)[usable], routed["category"].to_numpy(dtype=object)[usable]
                             if "category" in routed else [""] * int(usable.sum()))
        research = core.forecast_v2(matrix, config, HORIZON, groups)
        on_v2 = is_v2[usable]
        exact = {"v2_rows_7d_equal_research": bool(np.array_equal(routed_total[usable][on_v2], np.round(research.aggregate_7d, 1)[on_v2])),
                 "v2_rows_daily_equal_research": bool(np.array_equal(routed_daily[usable][on_v2], research.daily[on_v2, :HORIZON])),
                 "reasons_equal_router_assessment": bool((reason[usable] == router.assess_matrix(matrix, research.profile)).all())}
    all_groups = _pool_codes(keys["store_id"], keys["category"])
    direct = core.forecast_v2(history, config, HORIZON, all_groups)
    seconds["v2_core"] = time.perf_counter() - started
    forecasts = {V1: (v1_daily, v1_total), ROUTED: (routed_daily, routed_total),
                 V2: (direct.daily[:, :HORIZON], np.round(direct.aggregate_7d, 1))}
    for name, daily in baseline_forecasts(history).items():
        forecasts[name] = (daily, daily.sum(axis=1))

    # targets: observed days only
    end = min(ci + 1 + HORIZON, panel.quantity.shape[1])
    target = np.full((len(universe), HORIZON), np.nan)
    target[:, :end - ci - 1] = panel.quantity[universe, ci + 1:end]
    valid = ~np.isnan(target)
    week_valid = valid.all(axis=1)
    actual_7d = np.where(week_valid, np.nansum(target, axis=1), np.nan)

    # dimensions
    profile = direct.profile
    ever = profile.first_sale < history.shape[1]
    age_band = np.select([~ever, profile.age < 30, profile.age < 90, profile.age < 364], [0, 1, 2, 3], default=4)
    negative = (history < 0).any(axis=1)
    first_row = np.argmax(present, axis=1)
    censoring = np.full(len(universe), 2)
    if panel.censored is not None:
        window = np.full((len(universe), HORIZON), np.nan)
        window[:, :end - ci - 1] = panel.censored[universe, ci + 1:end]
        censoring = np.where((window == 1).any(axis=1), 1, np.where((window == 0).all(axis=1), 0, 2))
    reason_index = {r: i for i, r in enumerate(router.REASON_CODES)}
    codes = {"overall": np.zeros(len(universe), dtype=np.int64), "demand_type": profile.demand_type.astype(np.int64),
             "router_reason": np.array([reason_index[r] for r in reason], dtype=np.int64), "router_version": is_v2.astype(np.int64),
             "age_band": age_band.astype(np.int64), "negative_history": negative.astype(np.int64),
             "first_row_age": ((ci - first_row) < 30).astype(np.int64), "target_censoring": censoring.astype(np.int64)}
    sums = {}
    for method, (daily, total) in forecasts.items():
        per_window = {"daily_h1_7": window_stats(daily, target, valid),
                      "total_7d": window_stats(total[:, None], actual_7d[:, None], week_valid[:, None])}
        for window, stats in per_window.items():
            matrix_stats = np.stack([stats[k] for k in STATS], axis=1)
            for dimension, labels in DIMENSIONS.items():
                grouped = np.zeros((len(labels), len(STATS)))
                np.add.at(grouped, codes[dimension], matrix_stats)
                sums[(method, window, dimension)] = grouped

    # checks
    neg_cf = router.assess_matrix(history[reason == router.REASON_NEGATIVE]) if (reason == router.REASON_NEGATIVE).any() else np.array([], dtype=object)
    full_dates = panel.history["date"].to_numpy()
    # every row the full bridge kept or quarantined only for its quantity/unit is a future row once as_of = cutoff
    expected_future = int((full_dates > cutoff).sum()) + int((panel.late_quarantine_dates > cutoff).sum())
    keys_match = ([router._key(v) for v in routed["store_id"]] == [router._key(v) for v in keys["store_id"]]
                  and [router._key(v) for v in routed["product_id"]] == [router._key(v) for v in keys["product_id"]])
    cold = ever & (profile.age < router.MIN_SALES_AGE_DAYS)

    def files(block: np.ndarray) -> list[str]:
        return sorted(panel.source_files[c] for c in np.unique(block) if c >= 0)

    checks = {
        "cutoff": str(cutoff), "universe_series": int(len(universe)),
        "router_rows_aligned_with_inventory": bool(keys_match),
        "history_series_equal_universe": int(bridged.report["series"]) == len(universe) and diagnostics.get("rows_without_inventory_match") == 0,
        "history_max_date_le_cutoff": bool(len(bridged.history) == 0 or bridged.history["date"].max() <= cutoff),
        "bridge_future_rows_excluded": int(bridged.report["future_rows_excluded"]),
        "bridge_future_rows_expected": expected_future,
        "router_future_rows_excluded": int(diagnostics.get("future_rows_excluded", -1)),
        "router_cutoff_ok": diagnostics.get("cutoffs") == [str(cutoff)] or not usable.any(),
        "v1_rows_equal_v1": bool(np.array_equal(routed_total[~is_v2], v1_total[~is_v2])),
        **exact,
        "negative_series_routed_v1": bool(np.isin(reason[negative], [router.REASON_NEGATIVE, router.REASON_INVALID]).all()),
        "cold_start_series_routed_v1": bool((~is_v2[cold]).all()),
        "no_v2_errors": not diagnostics.get("errors") and router.REASON_ERROR not in set(reason),
        "duplicate_date_series": int(bridged.report["duplicate_date_series"]),
        "cold_start_series": int(cold.sum()),
        "history_source_files": files(panel.source_code[universe, :ci + 1]),
        "target_source_files": files(panel.source_code[universe, ci + 1:end]),
    }
    if self_exclusion_check:
        started = time.perf_counter()
        again, again_diag = router.route_demand_forecast(v1_out, panel.history)
        same = all(np.array_equal(again[c].to_numpy(dtype=np.float64), routed[c].to_numpy(dtype=np.float64))
                   for c in ("demand_forecast_7d", *router.DAILY_COLUMNS)) and bool((again["demand_forecast_reason"] == routed["demand_forecast_reason"]).all())
        checks["router_self_exclusion"] = {"future_rows_excluded": int(again_diag.get("future_rows_excluded", -1)),
                                           "future_rows_expected": int(((full_dates > cutoff) & np.isin(panel.history_code, universe)).sum()),
                                           "same_forecast": bool(same)}
        seconds["router_self_exclusion"] = time.perf_counter() - started

    hist28 = np.nansum(history[:, -28:], axis=1)
    rows = pd.DataFrame({"store_id": keys["store_id"].to_numpy(dtype=object), "product_id": keys["product_id"].to_numpy(dtype=object),
                         "category": keys["category"].to_numpy(dtype=object), "cutoff": str(cutoff), "reason": reason,
                         "version": np.where(is_v2, router.VERSION_V2, router.VERSION_V1),
                         "demand_type": np.array(core.DEMAND_TYPES, dtype=object)[profile.demand_type],
                         "days_since_first_sale": np.where(ever, profile.age, -1), "days_since_first_row": ci - first_row,
                         "negative_in_history": negative, "history_volume_28d": hist28, "target_days_observed": valid.sum(axis=1),
                         "week_scored": week_valid, "actual_7d": actual_7d, "target_censoring": np.array(DIMENSIONS["target_censoring"])[censoring],
                         **{f"forecast_7d_{m}": total for m, (_, total) in forecasts.items()}})
    negative_info = {"negative_series": int(negative.sum()), "routed_negative_sales": int((reason == router.REASON_NEGATIVE).sum()),
                     "otherwise_v2_eligible": int((neg_cf == router.REASON_V2).sum()),
                     "counterfactual_reasons": {str(k): int(v) for k, v in pd.Series(neg_cf, dtype=object).value_counts().items()}}
    return OriginOutput(sums, rows, checks, negative_info, seconds)


# ---------------------------------------------------------------- quantity semantics, external gate and verdicts (frozen)


def classify_semantics(evidence: Mapping[str, Mapping[str, Any]]) -> str:
    """SEMANTICS_RULE on {criterion: {"holds": bool, "evidence": text}} for every SEMANTICS_CRITERIA entry."""
    missing = [c for c in SEMANTICS_CRITERIA if c not in evidence]
    if missing:
        raise ValueError(f"semantics evidence missing for {missing}")
    holds = {c: bool(evidence[c]["holds"]) for c in SEMANTICS_CRITERIA}
    if not (holds["observed_sales"] and holds["daily_location_product"]):
        return "UNAVAILABLE"
    return "FULL" if all(holds.values()) else "PARTIAL"


def _relative(candidate: float, baseline: float) -> float:
    return float(candidate) / float(baseline) - 1.0 if baseline and np.isfinite(baseline) else float("nan")


def evaluate_dataset_gate(evidence: Mapping[str, Any], gate: Mapping[str, Any] = EXTERNAL_GATE) -> dict[str, Any]:
    """EXTERNAL_GATE on one dataset's primary-scope evidence: every criterion, the improvement flag and the verdict.

    evidence: integrity {name: bool, or None = not applicable}; total_7d / daily_h1_7 {"v1" | "router": {"wape",
    "bias_pct"}}; demand_type {segment: {"v1_wape", "router_wape", "scored", "actual_share"}}; origins [{"origin",
    "v1_wape", "router_wape"}]; coverage_share; semantics (classify_semantics). A NaN comparison never passes.
    """
    if evidence["semantics"] not in ("FULL", "PARTIAL"):
        raise ValueError(f"quantity semantics {evidence['semantics']}: not a daily sales history, the gate does not apply")
    criteria: list[dict[str, Any]] = []

    def add(name: str, group: str, passed: bool | None, value: Any = None, threshold: Any = None, note: str = "") -> None:
        criteria.append({"criterion": name, "group": group, "pass": passed, "value": value, "threshold": threshold, "note": note})

    for name, passed in evidence["integrity"].items():
        add(name, "integrity", None if passed is None else bool(passed), note="not applicable" if passed is None else "")
    safety = gate["safety"]
    for name in ("total_7d_wape", "daily_wape"):
        rule, block = safety[name], evidence[safety[name]["window"]]
        rel = _relative(block["router"]["wape"], block["v1"]["wape"])
        add(name, "safety", bool(rel <= rule["max_relative_degradation"]), rel, rule["max_relative_degradation"],
            f"router WAPE {block['router']['wape']:.4f} vs V1 {block['v1']['wape']:.4f}")
    rule = safety["bias"]
    v1_bias, router_bias = evidence[rule["window"]]["v1"]["bias_pct"], evidence[rule["window"]]["router"]["bias_pct"]
    bound = max(abs(v1_bias), rule["floor"])
    add("bias", "safety", bool(abs(router_bias) <= bound), router_bias, bound,
        f"|router bias| <= max(|V1 bias| = {abs(v1_bias):.4f}, floor {rule['floor']})")
    rule = safety["demand_type"]
    for segment in rule["segments"]:
        seg = evidence["demand_type"].get(segment) or {"v1_wape": float("nan"), "router_wape": float("nan"), "scored": 0, "actual_share": 0.0}
        rel = _relative(seg["router_wape"], seg["v1_wape"])
        size = rule["applies_when"]
        if seg["scored"] < size["min_scored_series_weeks"] or seg["actual_share"] < size["min_actual_volume_share"]:
            add(f"demand_type:{segment}", "safety", None, rel, rule["max_relative_degradation"],
                f"reported, not gated: {int(seg['scored'])} scored series-weeks, {seg['actual_share']:.4f} of the volume")
        else:
            add(f"demand_type:{segment}", "safety", bool(rel <= rule["max_relative_degradation"]), rel, rule["max_relative_degradation"])
    rule = gate["improvement"]
    total = evidence[rule["window"]]
    rel = _relative(total["router"]["wape"], total["v1"]["wape"])
    wins = sum(bool(o["router_wape"] < o["v1_wape"]) for o in evidence["origins"])
    improved = bool(rel <= -rule["min_relative_improvement"] and 2 * wins > len(evidence["origins"]))
    add("improvement", "improvement", improved, rel, -rule["min_relative_improvement"],
        f"router WAPE below V1 on {wins}/{len(evidence['origins'])} primary origins")
    covered = bool(evidence["coverage_share"] >= gate["coverage"]["min_share"])
    add("v2_coverage", "coverage", covered, evidence["coverage_share"], gate["coverage"]["min_share"])
    add("quantity_semantics", "semantics", evidence["semantics"] == "FULL", evidence["semantics"], "FULL")
    failed = [c["criterion"] for c in criteria if c["group"] in ("integrity", "safety") and c["pass"] is False]
    limitations = [label for label, ok in (("not improved over V1", improved),
                                           (f"V2 coverage below {gate['coverage']['min_share']:.0%} of the scored volume", covered),
                                           (f"quantity semantics {evidence['semantics']}", evidence["semantics"] == "FULL")) if not ok]
    verdict = "FAIL" if failed else "PASS" if not limitations else "PASS_WITH_LIMITATION"
    return {"verdict": verdict, "improved": improved, "failed": failed, "limitations": limitations, "criteria": criteria}


def combined_verdict(results: Mapping[str, Mapping[str, Any]]) -> str:
    """EXTERNAL_GATE combined verdict over {dataset: evaluate_dataset_gate result}."""
    if not results:
        raise ValueError("no dataset result")
    if any(r["verdict"] == "FAIL" for r in results.values()) or not any(r["improved"] for r in results.values()):
        return "FAIL"
    return "GENERALIZATION_PASS" if all(r["improved"] for r in results.values()) else "PARTIAL"


# ---------------------------------------------------------------- aggregation and result tables

PROXY_NOTE = ("No actual inventory or cost exists in these datasets: *_proxy columns are forecast-error quantities "
              "(over-forecast units ~ overstock risk, under-forecast units ~ stock-out risk), never actual costs.")
FALLBACK_CHECKS = ("router_rows_aligned_with_inventory", "history_series_equal_universe", "v1_rows_equal_v1", "v2_rows_7d_equal_research",
                   "v2_rows_daily_equal_research", "reasons_equal_router_assessment", "negative_series_routed_v1",
                   "cold_start_series_routed_v1", "no_v2_errors")
LEAKAGE_CHECKS = ("history_max_date_le_cutoff", "bridge_future_rows_excluded == bridge_future_rows_expected", "router_future_rows_excluded == 0",
                  "router_cutoff_ok", "router_self_exclusion: future_rows_excluded == future_rows_expected and same_forecast")


def merge_checks(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Batch checks of one origin -> one record: booleans all(), counts summed, file lists united."""
    out: dict[str, Any] = {}
    for key in parts[0]:
        values = [p[key] for p in parts]
        if isinstance(values[0], (bool, np.bool_)):
            out[key] = bool(all(values))
        elif isinstance(values[0], (int, np.integer)):
            out[key] = int(sum(values))
        elif isinstance(values[0], list):
            out[key] = sorted(set().union(*values))
        elif isinstance(values[0], Mapping):
            out[key] = merge_checks(values)
        else:
            out[key] = values[0] if all(v == values[0] for v in values) else list(values)
    return out


def integrity_checks(merged: Mapping[int, Mapping[str, Any]], split_check: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[int, Any]]:
    per_origin = {}
    for origin, c in merged.items():
        self_ex = c.get("router_self_exclusion")
        leakage = bool(c["history_max_date_le_cutoff"] and c["bridge_future_rows_excluded"] == c["bridge_future_rows_expected"]
                       and c["router_future_rows_excluded"] == 0 and c["router_cutoff_ok"]
                       and (self_ex is None or (self_ex["future_rows_excluded"] == self_ex["future_rows_expected"] and self_ex["same_forecast"])))
        per_origin[origin] = {"no_leakage": leakage, "fallback_policy": all(bool(c[k]) for k in FALLBACK_CHECKS)}
    split = None
    if split_check is not None:
        c = merged[split_check["origin"]]
        split = (c["history_source_files"] == sorted(split_check["history_files"])
                 and c["target_source_files"] == sorted(split_check["target_files"]))
    return {"no_leakage": all(p["no_leakage"] for p in per_origin.values()),
            "fallback_policy": all(p["fallback_policy"] for p in per_origin.values()), "train_eval_separation": split}, per_origin


def metrics_frame(sums: Mapping[tuple[int, str, str, str], np.ndarray], scopes: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    """Pooled metrics per scope x method x window x dimension x segment (statistics summed over the scope's origins)."""
    records = []
    for scope, origins in scopes.items():
        for method in METHODS:
            for window in WINDOWS:
                for dimension, labels in DIMENSIONS.items():
                    total = np.sum([sums[(o, method, window, dimension)] for o in origins], axis=0)
                    records += [{"scope": scope, "origins": len(origins), "method": method, "method_role": METHOD_ROLES[method],
                                 "window": window, "dimension": dimension, "segment": segment, **dict(zip(STATS, total[i]))}
                                for i, segment in enumerate(labels)]
    frame = finalize_metrics(pd.DataFrame(records))
    actual, n = frame["actual"].where(frame["actual"] > 0), frame["n"].where(frame["n"] > 0)
    frame["overforecast_ratio_proxy"] = frame["over_units"] / actual
    frame["underforecast_ratio_proxy"] = frame["under_units"] / actual
    frame["fill_rate_proxy"] = 1.0 - frame["under_units"] / actual
    frame["zero_forecast_missed_sale_rate_proxy"] = frame["missed_demand_n"] / n
    key = ["scope", "window", "dimension", "segment"]
    v1 = frame.loc[frame["method"] == V1, key + ["wape", "mae", "rmse", "bias_pct"]]
    frame = frame.merge(v1.rename(columns={c: f"v1_{c}" for c in ("wape", "mae", "rmse", "bias_pct")}), on=key, how="left")
    for metric in ("wape", "mae", "rmse"):
        frame[f"{metric}_vs_v1_relative"] = frame[metric] / frame[f"v1_{metric}"] - 1.0
    frame["rank_wape"] = frame.assign(_w=frame["wape"].round(12)).groupby(key)["_w"].rank(method="min")   # float-noise ties stay ties
    return frame


def coverage_table(rows: pd.DataFrame, scopes: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    """Router coverage per scope by router version and reason.

    all_series: every series with history up to the cutoff, volume = its recorded sales of the 28 days before the cutoff.
    scored_series: series whose 7 target days are all observed, volume = their actual 7-day sales (the gate basis).
    """
    records = []
    for scope, origins in scopes.items():
        part = rows[rows["origin"].isin(list(origins))]
        for basis, mask, volume_column in (("all_series", np.ones(len(part), dtype=bool), "history_volume_28d"),
                                           ("scored_series", part["week_scored"].to_numpy(dtype=bool), "actual_7d")):
            sel = part[mask]
            volume = sel[volume_column].fillna(0.0).to_numpy(dtype=np.float64)
            total_volume = float(volume.sum())
            for dimension, column, labels in (("router_version", "version", (router.VERSION_V1, router.VERSION_V2)),
                                              ("router_reason", "reason", tuple(router.REASON_CODES))):
                values = sel[column].to_numpy(dtype=object)
                for label in labels:
                    hit = values == label
                    records.append({"scope": scope, "origins": len(origins), "basis": basis, "volume_measure": volume_column,
                                    "dimension": dimension, "segment": label, "series_origins": int(hit.sum()),
                                    "series_origin_share": float(hit.sum() / len(sel)) if len(sel) else np.nan,
                                    "distinct_series": int(len(sel.loc[hit, ["store_id", "product_id"]].drop_duplicates())),
                                    "volume": float(volume[hit].sum()),
                                    "volume_share": float(volume[hit].sum() / total_volume) if total_volume > 0 else np.nan})
    return pd.DataFrame(records)


def demand_type_table(metrics: pd.DataFrame, rows: pd.DataFrame, scopes: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    """Per frozen demand type: size, volume share, V1 / router / V2 (and MA28) WAPE and bias, router V2 coverage."""
    records = []
    for scope, origins in scopes.items():
        part = rows[rows["origin"].isin(list(origins))]
        scored = part[part["week_scored"].to_numpy(dtype=bool)]
        total_actual = float(scored["actual_7d"].sum())
        pick = metrics[(metrics["scope"] == scope) & (metrics["dimension"] == "demand_type")]
        for segment in DIMENSIONS["demand_type"]:
            seg_rows, seg_scored = part[part["demand_type"] == segment], scored[scored["demand_type"] == segment]
            seg_actual = float(seg_scored["actual_7d"].sum())
            record = {"scope": scope, "origins": len(origins), "demand_type": segment, "series_origins": int(len(seg_rows)),
                      "scored_series_weeks": int(len(seg_scored)), "actual_7d_volume": seg_actual,
                      "actual_volume_share": seg_actual / total_actual if total_actual > 0 else np.nan,
                      "router_v2_series_origin_share": float((seg_rows["version"] == router.VERSION_V2).mean()) if len(seg_rows) else np.nan,
                      "router_v2_scored_volume_share": (float(seg_scored.loc[seg_scored["version"] == router.VERSION_V2, "actual_7d"].sum()) / seg_actual
                                                        if seg_actual > 0 else np.nan)}
            for window, tag in (("total_7d", ""), ("daily_h1_7", "daily_")):
                by_method = pick[(pick["window"] == window) & (pick["segment"] == segment)].set_index("method")
                for method, short in ((V1, "v1"), (ROUTED, "router"), (V2, "v2"), ("moving_average_28", "ma28")):
                    record[f"{tag}{short}_wape"] = by_method.at[method, "wape"]
                    if window == "total_7d":
                        record[f"{short}_bias_pct"] = by_method.at[method, "bias_pct"]
            records.append(record)
    return pd.DataFrame(records)


SUMMARY_COLUMNS = ["scope", "origins", "window", "method", "method_role", "n", "scored", "actual", "forecast", "wape", "mae", "rmse",
                   "bias_pct", "mean_error", "over_units", "under_units", "over_forecast_rate", "under_forecast_rate",
                   "overforecast_ratio_proxy", "underforecast_ratio_proxy", "fill_rate_proxy", "zero_forecast_missed_sale_rate_proxy",
                   "v1_wape", "wape_vs_v1_relative", "mae_vs_v1_relative", "rmse_vs_v1_relative", "rank_wape"]
BIAS_DIMENSIONS = ("overall", "demand_type", "router_version", "router_reason", "negative_history", "age_band", "first_row_age", "target_censoring")
BIAS_TABLE_COLUMNS = ["scope", "origins", "window", "dimension", "segment", "method", "method_role", "n", "actual", "forecast", "bias_units",
                      "bias_pct", "v1_bias_pct", "mean_error", "over_forecast_rate", "under_forecast_rate", "over_units", "under_units",
                      "overforecast_ratio_proxy", "underforecast_ratio_proxy", "fill_rate_proxy", "zero_forecast_missed_sale_rate_proxy"]


def _metric(metrics: pd.DataFrame, scope: str, window: str, method: str, dimension: str = "overall", segment: str = "all") -> pd.Series:
    hit = metrics[(metrics["scope"] == scope) & (metrics["window"] == window) & (metrics["method"] == method)
                  & (metrics["dimension"] == dimension) & (metrics["segment"] == segment)]
    if len(hit) != 1:
        raise KeyError((scope, window, method, dimension, segment))
    return hit.iloc[0]


def gate_evidence(metrics: pd.DataFrame, coverage: pd.DataFrame, scope: str, origins: Sequence[int], integrity: Mapping[str, Any],
                  semantics: str) -> dict[str, Any]:
    def pair(window: str, dimension: str = "overall", segment: str = "all") -> dict[str, Any]:
        return {name: {"wape": float(row["wape"]), "bias_pct": float(row["bias_pct"]), "n": float(row["n"]), "actual": float(row["actual"])}
                for name, row in (("v1", _metric(metrics, scope, window, V1, dimension, segment)),
                                  ("router", _metric(metrics, scope, window, ROUTED, dimension, segment)))}

    overall_actual = float(_metric(metrics, scope, "total_7d", V1)["actual"])
    demand = {}
    for segment in DIMENSIONS["demand_type"]:
        p = pair("total_7d", "demand_type", segment)
        demand[segment] = {"v1_wape": p["v1"]["wape"], "router_wape": p["router"]["wape"], "scored": p["router"]["n"],
                           "actual_share": p["router"]["actual"] / overall_actual if overall_actual > 0 else 0.0}
    per_origin = []
    for origin in origins:
        p = {m: float(_metric(metrics, f"origin_{origin:02d}", "total_7d", m)["wape"]) for m in (V1, ROUTED)}
        per_origin.append({"origin": origin, "v1_wape": p[V1], "router_wape": p[ROUTED]})
    share = coverage[(coverage["scope"] == scope) & (coverage["basis"] == "scored_series") & (coverage["dimension"] == "router_version")
                     & (coverage["segment"] == router.VERSION_V2)]["volume_share"]
    return {"integrity": dict(integrity), "total_7d": pair("total_7d"), "daily_h1_7": pair("daily_h1_7"), "demand_type": demand,
            "origins": per_origin, "coverage_share": float(share.iloc[0]) if len(share) and pd.notna(share.iloc[0]) else 0.0,
            "semantics": semantics}


# ---------------------------------------------------------------- dataset inputs (dataset-specific: files, columns, dated protocol)


@dataclass
class DatasetInput:
    dataset: str
    batches: Callable[[], Iterator[tuple[str, pd.DataFrame]]]   # canonical rows of whole locations, one frame per batch
    first_day: np.datetime64
    last_day: np.datetime64
    origins: list[dict[str, Any]]           # {"origin", "phase", "cutoff" (datetime64[D])}, chronological
    scopes: dict[str, list[int]]            # pooled scopes (run_dataset adds one scope per origin)
    primary_scope: str
    semantics: dict[str, dict[str, Any]]    # SEMANTICS_CRITERIA -> {"holds", "evidence"}
    profile: dict[str, Any]
    split_check: dict[str, Any] | None = None     # {"origin", "history_files", "target_files"}
    cleanup: Callable[[], None] | None = None


def _canonical_path(data_root: Path | str, dataset: str) -> Path:
    return Path(data_root) / DATASETS[dataset] / "processed" / "canonical_demand_series.parquet"


def _canonical_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    missing = [c for c in ("date", "date_grain", "location_id", "product_id", "sales_qty") if c not in names]
    if missing:
        raise ValueError(f"{path} is not a canonical demand_series: {missing} absent")
    return [c for c in CANONICAL_COLUMNS if c in names]


def _natural(value: Any) -> tuple:
    text = str(value)
    return (0, int(text), text) if text.isdigit() else (1, 0, text)


NULL_TEXT = "<NULL>"


def _new_facts() -> dict[str, Any]:
    return {"rows": 0, "null_quantity_rows": 0, "zero_rows": 0, "negative_rows": 0, "negative_sum_parts": [], "stockout_rows": 0,
            "stockout_known_rows": 0, "date_min": None, "date_max": None, "locations": set(), "products": set(),
            **{name: Counter() for name in ("date_grain", "product_grain", "unit", "unit_status", "source_file", "validation_flags",
                                             "location_id_namespace", "product_id_namespace")}}


def _accumulate(facts: dict[str, Any], table: Any) -> None:
    """Measured facts of canonical demand_series rows (an Arrow table chunk), for the profile and the semantics evidence."""
    import pyarrow.compute as pc
    q = table["sales_qty"]
    facts["rows"] += table.num_rows
    facts["null_quantity_rows"] += q.null_count
    facts["zero_rows"] += int(pc.sum(pc.equal(q, 0.0)).as_py() or 0)
    negative = pc.less(q, 0.0)
    facts["negative_rows"] += int(pc.sum(negative).as_py() or 0)
    facts["negative_sum_parts"].append(float(pc.sum(pc.filter(q, negative)).as_py() or 0.0))
    if "stockout_hours" in table.column_names:
        hours = table["stockout_hours"]
        facts["stockout_known_rows"] += table.num_rows - hours.null_count
        facts["stockout_rows"] += int(pc.sum(pc.greater(hours, 0.0)).as_py() or 0)
    dates = pc.min_max(table["date"]).as_py()
    if dates["min"] is not None:
        facts["date_min"] = min(filter(None, (facts["date_min"], dates["min"])))
        facts["date_max"] = max(filter(None, (facts["date_max"], dates["max"])))
    facts["locations"].update(pc.unique(table["location_id"]).to_pylist())
    facts["products"].update(pc.unique(table["product_id"]).to_pylist())
    for name in ("date_grain", "product_grain", "unit", "unit_status", "source_file", "validation_flags", "location_id_namespace",
                 "product_id_namespace"):
        if name in table.column_names:
            for item in pc.value_counts(table[name]).to_pylist():
                facts[name][NULL_TEXT if item["values"] is None else str(item["values"])] += int(item["counts"])



def _facts_record(facts: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in facts.items() if k not in ("negative_sum_parts", "locations", "products")}
    out.update(negative_quantity_sum=math.fsum(facts["negative_sum_parts"]), locations=len(facts["locations"] - {None}),
               products=len(facts["products"] - {None}), validation_flags=dict(facts["validation_flags"].most_common(12)))
    return {k: dict(v) if isinstance(v, Counter) else v for k, v in out.items()}


def measured_semantics(facts: Mapping[str, Any], observed_sales: tuple[bool, str], *, complete_grid: bool | None) -> dict[str, dict[str, Any]]:
    """SEMANTICS_CRITERIA from measured canonical facts; only ``observed_sales`` is documentary (source specification)."""
    rows = facts["rows"]
    daily = facts["date_grain"].get("daily", 0) == rows
    product = sum(v for k, v in facts["product_grain"].items() if k in ("product", NULL_TEXT)) == rows
    units, statuses = set(facts["unit"]), set(facts["unit_status"])
    physical = NULL_TEXT not in units and not any("normali" in u for u in units) and "UNKNOWN" not in statuses
    zero_days = facts["zero_rows"] > 0 and complete_grid is not False
    netted, censored = facts["negative_rows"] > 0, facts["stockout_rows"] > 0
    normalised = any("normali" in u for u in units)
    return {
        "observed_sales": {"holds": bool(observed_sales[0]), "evidence": observed_sales[1]},
        "daily_location_product": {"holds": bool(daily and product and facts["locations"] and facts["products"]),
                                   "evidence": f"date_grain {dict(facts['date_grain'])}; product_grain {dict(facts['product_grain'])}; "
                                               f"{len(facts['locations'])} locations x {len(facts['products'])} products"},
        "physical_unit": {"holds": bool(physical), "evidence": f"unit {dict(facts['unit'])}; unit_status {dict(facts['unit_status'])}"},
        "zero_days_recorded": {"holds": bool(zero_days), "evidence": f"{facts['zero_rows']:,} zero-sale rows of {rows:,}; complete day grid per "
                                                                     f"series: {complete_grid}"},
        "unmodified_meaning": {"holds": not (netted or censored or normalised),
                               "evidence": f"negative (netted return) rows {facts['negative_rows']:,}; rows with stock-out hours > 0 "
                                           f"{facts['stockout_rows']:,} of {facts['stockout_known_rows']:,} annotated; normalised unit: {normalised}"},
    }


def _quality(dataset: str, data_root: Path | str) -> dict[str, Any]:
    path = Path(data_root) / DATASETS[dataset] / "results" / "data_quality_report.json"
    return json.loads(path.read_text(encoding="utf-8")).get("semantic_checks", {}) if path.exists() else {}


def _location_batches(canonical: pd.DataFrame, locations_per_batch: int) -> Callable[[], Iterator[tuple[str, pd.DataFrame]]]:
    """Whole locations per batch (the router's weekday pooling groups are location x category, so a batch never splits one)."""
    codes, values = pd.factorize(canonical["location_id"], use_na_sentinel=True)
    order = sorted(range(len(values)), key=lambda i: _natural(values[i]))
    rank = np.empty(len(values) + 1, dtype=np.int64)
    rank[order] = np.arange(len(values))
    rank[-1] = len(values)                                 # rows without a location: last batch (the bridge quarantines them)
    batch = rank[codes] // max(1, locations_per_batch)
    position = np.argsort(batch, kind="stable")
    groups = np.split(position, np.flatnonzero(np.diff(batch[position])) + 1) if len(position) else []

    def iterate() -> Iterator[tuple[str, pd.DataFrame]]:
        for group in groups:
            part = canonical.iloc[group].reset_index(drop=True)
            names = sorted(part["location_id"].dropna().astype(str).unique(), key=_natural)
            yield (f"locations {names[0]}..{names[-1]} ({len(names)})" if names else "no location"), part
    return iterate


def load_freshretailnet(data_root: Path | str = DATA_ROOT, *, locations_per_batch: int = 100,
                        canonical: pd.DataFrame | None = None) -> DatasetInput:
    """FreshRetailNet: the canonical table fits in memory (4.85M rows); train/eval dates fix the protocol."""
    import pyarrow as pa
    protocol = PROTOCOLS["freshretailnet"]
    if canonical is None:
        import pyarrow.parquet as pq
        path = _canonical_path(data_root, "freshretailnet")
        table = pq.read_table(path, columns=_canonical_columns(path))
    else:
        table = pa.Table.from_pandas(canonical, preserve_index=False)
    facts = _new_facts()
    _accumulate(facts, table)
    frame = table.to_pandas()
    del table
    train_file, eval_file = protocol["train_file"], protocol["eval_file"]
    files = frame["source_file"].astype(str).to_numpy()
    if set(np.unique(files)) != {train_file, eval_file}:
        raise ValueError(f"FreshRetailNet source files {sorted(set(files))} are not {train_file} + {eval_file}")
    day = pd.to_datetime(frame["date"], format="ISO8601").to_numpy().astype("datetime64[D]")
    train_days, eval_days = np.unique(day[files == train_file]), np.unique(day[files == eval_file])
    final = train_days.max()
    if not np.array_equal(eval_days, final + np.arange(1, HORIZON + 1)):
        raise ValueError(f"eval dates {eval_days} are not the {HORIZON} days after the last train day {final}")
    pairs = pd.factorize(frame["location_id"].astype(str) + "\x1f" + frame["product_id"].astype(str))[0]
    per_series = np.bincount(pairs)
    days = int((day.max() - day.min()).astype(np.int64)) + 1
    complete = bool(per_series.min() == per_series.max() == days)
    origins = [{"origin": i + 1, "phase": "development", "cutoff": final - np.timedelta64(back, "D")} for i, back in enumerate((21, 14, 7))]
    origins.append({"origin": 4, "phase": "final", "cutoff": final})
    quality = _quality("freshretailnet", data_root)
    realness = str(quality.get("provenance_and_realness", {}).get("classification", ""))
    profile = {"canonical": _facts_record(facts), "series": int(len(per_series)), "rows_per_series": [int(per_series.min()), int(per_series.max())],
               "calendar_days": days, "train_dates": [str(train_days.min()), str(final)], "eval_dates": [str(eval_days.min()), str(eval_days.max())],
               "train_rows": int((files == train_file).sum()), "eval_rows": int((files == eval_file).sum()),
               "stockout_semantics": quality.get("stockout_semantics", {}).get("inventory"), "realness": realness}
    semantics = measured_semantics(facts, (realness.startswith("REAL_SALES"), f"data_quality_report provenance_and_realness: {realness}"),
                                   complete_grid=complete)
    return DatasetInput("freshretailnet", _location_batches(frame, locations_per_batch), day.min(), day.max(), origins,
                        {"final": [4], "development_pooled": [1, 2, 3]}, protocol["primary_scope"], semantics, profile,
                        {"origin": 4, "history_files": [train_file], "target_files": [eval_file]})


def partition_by_location(path: Path, workdir: Path, columns: Sequence[str]) -> tuple[list[Path], dict[str, Any]]:
    """Stream a canonical parquet once, row group by row group, into one parquet file per location (never all rows in
    memory), measuring the canonical facts on the way."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    source = pq.ParquetFile(path)
    writers: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    facts = _new_facts()
    try:
        for group in range(source.num_row_groups):
            table = source.read_row_group(group, columns=list(columns))
            _accumulate(facts, table)
            location = table["location_id"]
            for value in pc.unique(location).to_pylist():
                key = NULL_TEXT if value is None else str(value)
                if key not in writers:
                    paths[key] = workdir / f"location_{len(paths):04d}.parquet"
                    writers[key] = pq.ParquetWriter(paths[key], table.schema)
                writers[key].write_table(table.filter(pc.is_null(location) if value is None else pc.equal(location, value)))
    finally:
        for writer in writers.values():
            writer.close()
    return [paths[k] for k in sorted(paths, key=_natural)], facts


def load_favorita(data_root: Path | str = DATA_ROOT, *, workdir: Path | None = None) -> DatasetInput:
    """Favorita: 125M canonical rows are split once into per-store files in a temporary folder (removed afterwards);
    each store is then one batch with its full history."""
    import pyarrow.parquet as pq
    path = _canonical_path(data_root, "favorita")
    temporary = workdir is None
    folder = Path(tempfile.mkdtemp(prefix="varo_favorita_")) if temporary else Path(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        files, facts = partition_by_location(path, folder, _canonical_columns(path))
    except BaseException:
        if temporary:
            shutil.rmtree(folder, ignore_errors=True)
        raise
    seconds = time.perf_counter() - started
    first, last = np.datetime64(facts["date_min"], "D"), np.datetime64(facts["date_max"], "D")
    origins = [{"origin": i + 1, "phase": "external_test", "cutoff": last - np.timedelta64(back, "D")} for i, back in enumerate((28, 21, 14, 7))]
    quality = _quality("favorita", data_root)
    signed = quality.get("signed_sales_investigation", {})
    profile = {"canonical": _facts_record(facts), "partition_files": len(files), "partition_seconds": round(seconds, 3),
               "documented": {"signed_sales_investigation": {k: signed.get(k) for k in ("negative_rows", "negative_sum", "items_with_negative", "specification")},
                              "zero_sales_absence": {k: quality.get("zero_sales_absence", {}).get(k) for k in ("zero_rows", "specification", "policy")},
                              "unit_investigation": quality.get("unit_investigation", {}).get("conclusion")}}
    semantics = measured_semantics(facts, (bool(signed), "Kaggle Favorita train.csv unit_sales per (date, store_nbr, item_nbr); "
                                                         f"{signed.get('specification', '')}"), complete_grid=facts["zero_rows"] > 0)

    def iterate() -> Iterator[tuple[str, pd.DataFrame]]:
        for file in files:
            frame = pq.read_table(file).to_pandas()
            names = sorted(frame["location_id"].dropna().astype(str).unique(), key=_natural)
            yield (f"location {names[0]}" if names else "no location"), frame

    return DatasetInput("favorita", iterate, first, last, origins, {"external_test_pooled": [1, 2, 3, 4]},
                        PROTOCOLS["favorita"]["primary_scope"], semantics, profile, None,
                        (lambda: shutil.rmtree(folder, ignore_errors=True)) if temporary else None)


LOADERS: dict[str, Callable[..., DatasetInput]] = {"freshretailnet": load_freshretailnet, "favorita": load_favorita}


# ---------------------------------------------------------------- production bridge end-to-end (real canonical rows)

# case -> expected router reason. Cases are picked by their data (frozen thresholds), never by the router's output.
E2E_CASES: dict[str, str] = {
    "negative_history": router.REASON_NEGATIVE,
    "no_positive_sale": router.REASON_NO_SALES,
    "cold_start": router.REASON_COLD_START,
    "missing_day": router.REASON_INSUFFICIENT,
    "mostly_zero": router.REASON_MOSTLY_ZERO,
    "zero_day": router.REASON_V2,
    "v2_eligible": router.REASON_V2,
    "duplicate_date": router.REASON_INVALID,
    "history_withheld": router.REASON_MISSING,
}


def production_bridge_e2e(canonical: pd.DataFrame, cutoff: Any, *, per_case: int = 2) -> dict[str, Any]:
    """Real canonical rows -> demand_history_bridge.attach_daily_sales_history -> the analysis pipeline's inventory step
    (V1 -> forecast router), next to the same inventory without a history (must stay V1 exactly)."""
    from services.analysis_pipeline import PipelineResult, _run_inventory_analysis, _Runner
    from services.demand_history_bridge import attach_daily_sales_history
    from services.legacy_adapters.data_adapter import prepare_legacy_data

    cutoff = np.datetime64(pd.Timestamp(cutoff).date(), "D")
    full = canonical_to_daily_history(canonical, keep_row_lineage=True)
    dates = full.history["date"].to_numpy().astype("datetime64[D]")
    panel = build_panel(canonical, dates.min(), max(dates.max(), cutoff))
    ci = int((cutoff - panel.day0) / DAY)
    history, present = panel.quantity[:, :ci + 1], panel.present[:, :ci + 1]
    has = present.any(axis=1)
    invalid = (present & np.isnan(history)).any(axis=1)            # invalid quantity or a repeated day
    negative = (history < 0).any(axis=1)
    profile = core.demand_profile(history)
    ever = profile.first_sale < history.shape[1]
    recent = history[:, -router.RECENT_OBSERVED_DAYS:]
    gap = np.isnan(recent).any(axis=1) | (history.shape[1] < router.RECENT_OBSERVED_DAYS)
    mostly = profile.demand_type == core.DEMAND_TYPES.index("mostly_zero")
    clean = has & ~invalid & ~negative
    mature = clean & ever & (profile.age >= router.MIN_SALES_AGE_DAYS)
    eligible = mature & ~gap & ~mostly
    pools = {"negative_history": has & ~invalid & negative, "no_positive_sale": clean & ~ever,
             "cold_start": clean & ever & (profile.age < router.MIN_SALES_AGE_DAYS), "missing_day": mature & gap,
             "mostly_zero": mature & ~gap & mostly, "zero_day": eligible & (recent == 0).any(axis=1),
             "v2_eligible": eligible & ~(recent == 0).any(axis=1)}
    picked = {case: np.flatnonzero(mask)[:per_case] for case, mask in pools.items()}
    spare = np.setdiff1d(np.flatnonzero(eligible), np.concatenate(list(picked.values())))
    picked["duplicate_date"], picked["history_withheld"] = spare[:1], spare[1:2]
    order = np.concatenate([picked[c] for c in E2E_CASES]).astype(np.int64)
    case_of = np.concatenate([[c] * len(picked[c]) for c in E2E_CASES]).astype(object)

    # canonical input: every row (all dates, future included) of the picked series, minus the withheld one, plus one
    # repeated day for the duplicate case
    sent = np.setdiff1d(order, picked["history_withheld"])
    rows_of = np.flatnonzero(np.isin(panel.history_code, sent))
    subset = canonical.iloc[panel.history["input_position"].to_numpy()[rows_of]]
    extra = []
    for s in picked["duplicate_date"]:
        mine = np.flatnonzero((panel.history_code == s) & (panel.history["date"].to_numpy().astype("datetime64[D]") <= cutoff))
        extra.append(canonical.iloc[[int(panel.history["input_position"].to_numpy()[mine[-1]])]])
    subset = pd.concat([subset, *extra], ignore_index=True)
    inventory = inventory_rows(panel.keys.iloc[order].reset_index(drop=True), history[order], cutoff)
    uploaded = {"inventory": inventory}
    with_history, bridged = attach_daily_sales_history(uploaded, subset, as_of=str(cutoff))

    def analyse(data: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], PipelineResult]:
        result = PipelineResult()
        analyzed, summaries = _run_inventory_analysis(_Runner(result), prepare_legacy_data(data)["inventory"],
                                                      daily_sales_history=data.get(router.DAILY_SALES_HISTORY_KEY))
        position = {(router._key(s), router._key(p)): i for i, (s, p) in enumerate(zip(analyzed["store_id"], analyzed["product_id"]))}
        align = [position[(router._key(s), router._key(p))] for s, p in zip(inventory["store_id"], inventory["product_id"])]
        return analyzed.iloc[align].reset_index(drop=True), summaries, result

    analyzed, summaries, result = analyse(with_history)
    plain, plain_summaries, plain_result = analyse(uploaded)
    reason = analyzed["demand_forecast_reason"].to_numpy(dtype=object)
    expected = np.array([E2E_CASES[c] for c in case_of], dtype=object)
    is_v2 = reason == router.REASON_V2
    v1_total = core.run_v1_baseline(inventory)["demand_forecast_7d"].to_numpy(dtype=np.float64)
    total = analyzed["demand_forecast_7d"].to_numpy(dtype=np.float64)
    daily = analyzed[list(router.DAILY_COLUMNS)].to_numpy(dtype=np.float64)
    usable = ~np.isin(reason, [router.REASON_MISSING, router.REASON_INVALID, router.REASON_NEGATIVE, router.REASON_ERROR])
    research_equal = True
    if usable.any():
        rows = order[usable]
        start = int(np.argmax(present[rows], axis=1).min())
        research = core.forecast_v2(history[rows, start:], router.v2_config(), HORIZON,
                                    _pool_codes(inventory["store_id"].to_numpy(dtype=object)[usable], inventory["category"].to_numpy(dtype=object)[usable]))
        on_v2 = is_v2[usable]
        research_equal = bool(np.array_equal(total[usable][on_v2], np.round(research.aggregate_7d, 1)[on_v2])
                              and np.array_equal(daily[usable][on_v2], research.daily[on_v2, :HORIZON]))
    bridged_code = pd.factorize(bridged.history["store_id"].astype(str) + "\x1f" + bridged.history["product_id"].astype(str))[0]
    key_of = {f"{s}\x1f{p}": i for i, (s, p) in enumerate(zip(panel.keys["store_id"].astype(str), panel.keys["product_id"].astype(str)))}
    first_rows = np.flatnonzero(np.r_[True, bridged_code[1:] != bridged_code[:-1]])
    bridged_series = np.array([key_of[f"{s}\x1f{p}"] for s, p in zip(bridged.history["store_id"].astype(str).to_numpy()[first_rows],
                                                                    bridged.history["product_id"].astype(str).to_numpy()[first_rows])], dtype=np.int64)
    series_of_row = bridged_series[bridged_code]
    q = bridged.history["quantity"].to_numpy(dtype=np.float64)

    def rows_for(case: str) -> np.ndarray:
        return np.isin(series_of_row, picked[case])

    subset_dates = pd.to_datetime(subset["date"], format="ISO8601").to_numpy().astype("datetime64[D]")
    subset_qty = pd.to_numeric(subset["sales_qty"], errors="coerce").to_numpy(dtype=np.float64)
    zero_mask, negative_rows, missing_rows = rows_for("zero_day"), rows_for("negative_history"), rows_for("missing_day")
    negative_positions = bridged.history["input_position"].to_numpy()[negative_rows & (q < 0)]
    checks = {
        "reasons_equal_expected": bool((reason == expected).all()),
        "v1_rows_equal_v1": bool(np.array_equal(total[~is_v2], v1_total[~is_v2])),
        "v2_rows_equal_research_v2": research_equal,
        "zero_days_kept_as_zero": bool(int((q[zero_mask] == 0).sum()) == int((history[picked["zero_day"]] == 0).sum())),
        "negative_values_unchanged": bool(np.array_equal(q[negative_rows & (q < 0)], subset_qty[negative_positions])),
        "missing_days_have_no_row": bool(int(missing_rows.sum()) == int(present[picked["missing_day"]].sum())),
        "duplicate_day_reported": int(bridged.report["duplicate_date_series"]) == len(picked["duplicate_date"]),
        "bridge_excluded_future_rows": int(bridged.report["future_rows_excluded"]) == int((subset_dates > cutoff).sum()),
        "router_read_no_future_row": summaries["demand_forecast"]["forecast_router"].get("future_rows_excluded") == 0,
        "without_history_all_v1": bool((plain["demand_forecast_reason"] == router.REASON_MISSING).all()
                                       and np.array_equal(plain["demand_forecast_7d"].to_numpy(dtype=np.float64), v1_total)),
        "function_label_unchanged": summaries["demand_forecast"]["function"] == plain_summaries["demand_forecast"]["function"]
                                    == "demand_forecast_analyzer.analyze_demand_forecast",
        "router_connected_iff_v2_rows": ("services.demand_forecast_router.route_demand_forecast" in result.connected_algorithms) == bool(is_v2.any()),
        "no_pipeline_warnings": not result.warnings and not plain_result.warnings,
    }
    cases = [{"case": case, "expected_reason": E2E_CASES[case], "series": int(len(picked[case])),
              "observed_reasons": {str(k): int(v) for k, v in pd.Series(reason[case_of == case], dtype=object).value_counts().items()}}
             for case in E2E_CASES]
    return {"cutoff": str(cutoff), "path": "canonical demand_series -> attach_daily_sales_history -> analysis_pipeline._run_inventory_analysis "
                                           "(V1 step -> _route_forecast -> route_demand_forecast)",
            "series": int(len(order)), "canonical_rows_sent": int(len(subset)), "cases": cases, "checks": checks,
            "pass": all(checks.values()), "unavailable_cases": [c["case"] for c in cases if c["series"] == 0],
            "bridge_report": {k: bridged.report[k] for k in ("history_rows", "series", "future_rows_excluded", "zero_rows", "negative_rows",
                                                             "negative_series", "duplicate_date_series", "units")},
            "router_diagnostics": {k: v for k, v in summaries["demand_forecast"]["forecast_router"].items() if k != "errors"}}


# ---------------------------------------------------------------- one dataset: run, score, gate, write


def run_dataset(dataset: str, data_root: Path | str = DATA_ROOT, output_dir: Path | str | None = None,
                common_dir: Path | str | None = None, *, inputs: DatasetInput | None = None, self_exclusion_check: bool = True,
                e2e_per_case: int = 2) -> dict[str, Any]:
    """Freeze (or verify) the gate, then run every origin of every batch through the production path and write the results."""
    started = time.perf_counter()
    root = Path(data_root)
    common = Path(common_dir) if common_dir else root / COMMON_RESULTS
    before = frozen_config_check()
    freeze = freeze_gate(common)                       # written before the first scoring; refuses a changed gate afterwards
    inputs = inputs if inputs is not None else LOADERS[dataset](root)
    try:
        return _run_dataset(dataset, root, output_dir, inputs, before, freeze, started, self_exclusion_check, e2e_per_case)
    finally:
        if inputs.cleanup is not None:
            inputs.cleanup()


def _run_dataset(dataset: str, root: Path, output_dir: Path | str | None, inputs: DatasetInput, before: Mapping[str, Any],
                 freeze: Mapping[str, Any], started: float, self_exclusion_check: bool, e2e_per_case: int) -> dict[str, Any]:
    semantics = classify_semantics(inputs.semantics)
    if semantics == "UNAVAILABLE":
        raise ValueError(f"{dataset}: quantity semantics UNAVAILABLE; the external forecast validation does not apply")
    output = Path(output_dir) if output_dir else root / DATASETS[dataset] / "results"
    output.mkdir(parents=True, exist_ok=True)
    config = router.v2_config()
    origin_ids = [o["origin"] for o in inputs.origins]
    primary_origins = inputs.scopes[inputs.primary_scope]
    primary_cutoff = max(o["cutoff"] for o in inputs.origins if o["origin"] in primary_origins)
    sums: dict[tuple[int, str, str, str], np.ndarray] = {}
    rows: list[pd.DataFrame] = []
    checks: dict[int, list] = {o: [] for o in origin_ids}
    negatives: dict[int, list] = {o: [] for o in origin_ids}
    bridge_reports, batch_log = [], []
    seconds: Counter = Counter()
    processed: Counter = Counter()
    e2e = None
    for label, canonical in inputs.batches():
        batch_started = time.perf_counter()
        panel = build_panel(canonical, inputs.first_day, inputs.last_day)
        seconds["panel_bridge"] += time.perf_counter() - batch_started
        bridge_reports.append(panel.bridge_report)
        processed.update(batches=1, canonical_rows=len(canonical), series=len(panel.keys))
        for spec in inputs.origins:
            ci = int((spec["cutoff"] - panel.day0) / DAY)
            if not panel.present[:, :ci + 1].any():
                processed["batch_origins_without_history"] += 1
                continue
            out = evaluate_origin(panel, canonical, spec["cutoff"], config, self_exclusion_check=self_exclusion_check)
            for key, value in out.sums.items():
                sums[(spec["origin"], *key)] = sums[(spec["origin"], *key)] + value if (spec["origin"], *key) in sums else value
            rows.append(out.rows.assign(origin=spec["origin"], phase=spec["phase"]))
            checks[spec["origin"]].append(out.checks)
            negatives[spec["origin"]].append(out.negative)
            seconds.update(out.seconds)
            processed["series_origins"] += out.checks["universe_series"]
        if e2e is None and e2e_per_case:
            e2e_started = time.perf_counter()
            e2e = production_bridge_e2e(canonical, primary_cutoff, per_case=e2e_per_case)
            seconds["e2e"] += time.perf_counter() - e2e_started
        batch_log.append({"batch": label, "canonical_rows": int(len(canonical)), "series": int(len(panel.keys)),
                          "seconds": round(time.perf_counter() - batch_started, 3)})
        del panel, canonical
    if not rows:
        raise ValueError(f"{dataset}: no series has history before any cutoff")

    frame = pd.concat(rows, ignore_index=True)
    merged = {o: merge_checks(parts) for o, parts in checks.items() if parts}
    integrity, per_origin_integrity = integrity_checks(merged, inputs.split_check)
    scopes = {**inputs.scopes, **{f"origin_{o:02d}": [o] for o in origin_ids}}
    metrics = metrics_frame(sums, scopes)
    coverage = coverage_table(frame, scopes)
    by_type = demand_type_table(metrics, frame, scopes)
    after = frozen_config_check()
    integrity = {"frozen_config": bool(before["FROZEN_CONFIG_UNCHANGED"] and after["FROZEN_CONFIG_UNCHANGED"]), **integrity}
    evidence = gate_evidence(metrics, coverage, inputs.primary_scope, primary_origins, integrity, semantics)
    evaluation = evaluate_dataset_gate(evidence)

    paths = _write_dataset_tables(dataset, output, metrics, coverage, by_type, frame, inputs)
    report = _dataset_report(dataset, inputs, semantics, metrics, coverage, by_type, frame, merged, per_origin_integrity, integrity,
                             evidence, evaluation, before, after, freeze, bridge_reports, negatives, e2e, seconds, processed,
                             batch_log, started, paths)
    paths["json"].write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return _clean(report)


def _write_dataset_tables(dataset: str, output: Path, metrics: pd.DataFrame, coverage: pd.DataFrame, by_type: pd.DataFrame,
                          frame: pd.DataFrame, inputs: DatasetInput) -> dict[str, Path]:
    prefix = FILE_PREFIX.get(dataset, dataset)
    paths = {name: output / f"{prefix}{suffix}" for name, suffix in RESULT_SUFFIXES.items()}
    pooled = [s for s in metrics["scope"].unique() if not s.startswith("origin_")]
    overall = metrics[metrics["dimension"] == "overall"]
    _write_csv(overall[overall["scope"].isin(pooled)][SUMMARY_COLUMNS], paths["summary"])
    cutoffs = {f"origin_{o['origin']:02d}": (str(o["cutoff"]), o["phase"]) for o in inputs.origins}
    by_cutoff = overall[overall["scope"].isin(list(cutoffs))].copy()
    by_cutoff.insert(1, "cutoff", by_cutoff["scope"].map(lambda s: cutoffs[s][0]))
    by_cutoff.insert(2, "phase", by_cutoff["scope"].map(lambda s: cutoffs[s][1]))
    _write_csv(by_cutoff[["scope", "cutoff", "phase", *SUMMARY_COLUMNS[1:]]], paths["by_cutoff"])
    _write_csv(by_type, paths["by_demand_type"])
    _write_csv(coverage, paths["coverage"])
    bias = metrics[metrics["dimension"].isin(BIAS_DIMENSIONS) & (metrics["n"] > 0)]
    _write_csv(bias[BIAS_TABLE_COLUMNS], paths["bias"])
    _write_csv(metrics, paths["metrics"])
    frame.to_parquet(paths["rows"], index=False)
    return paths


def _dataset_report(dataset, inputs, semantics, metrics, coverage, by_type, frame, merged, per_origin_integrity, integrity, evidence,
                    evaluation, before, after, freeze, bridge_reports, negatives, e2e, seconds, processed, batch_log, started, paths):
    scope = inputs.primary_scope
    primary = inputs.scopes[scope]

    def methods_block(scope_name: str, window: str, dimension: str = "overall", segment: str = "all") -> dict[str, Any]:
        out = {}
        for method in METHODS:
            row = _metric(metrics, scope_name, window, method, dimension, segment)
            out[method] = {k: row[k] for k in ("n", "actual", "forecast", "wape", "mae", "rmse", "bias_pct", "over_units", "under_units",
                                                "overforecast_ratio_proxy", "underforecast_ratio_proxy", "fill_rate_proxy",
                                                "wape_vs_v1_relative", "rank_wape")}
        return out

    def segments(scope_name: str, dimension: str, methods: Sequence[str] = (V1, ROUTED, V2, "moving_average_28")) -> dict[str, Any]:
        return {segment: {m: {k: _metric(metrics, scope_name, "total_7d", m, dimension, segment)[k] for k in ("n", "actual", "wape", "bias_pct")}
                          for m in methods} for segment in DIMENSIONS[dimension]}

    def coverage_block(scope_name: str, basis: str, dimension: str) -> dict[str, Any]:
        part = coverage[(coverage["scope"] == scope_name) & (coverage["basis"] == basis) & (coverage["dimension"] == dimension)]
        return {r["segment"]: {k: r[k] for k in ("series_origins", "series_origin_share", "distinct_series", "volume", "volume_share")}
                for _, r in part.iterrows()}

    full_bridge = {"input_rows": sum(r["input_rows"] for r in bridge_reports), "history_rows": sum(r["history_rows"] for r in bridge_reports),
                   "series": sum(r["series"] for r in bridge_reports), "zero_rows": sum(r["zero_rows"] for r in bridge_reports),
                   "negative_rows": sum(r["negative_rows"] for r in bridge_reports), "negative_series": sum(r["negative_series"] for r in bridge_reports),
                   "negative_quantity_sum": math.fsum(r["negative_quantity_sum"] for r in bridge_reports),
                   "duplicate_date_series": sum(r["duplicate_date_series"] for r in bridge_reports),
                   "quarantined_rows": {k: sum(r["quarantined_rows"][k] for r in bridge_reports) for k in bridge_reports[0]["quarantined_rows"]},
                   "units": dict(sum((Counter(r["units"]) for r in bridge_reports), Counter())),
                   "bridge_version": bridge_reports[0]["bridge_version"], "rules": bridge_reports[0]["rules"]}
    canonical = inputs.profile["canonical"]
    returns_preserved = bool(full_bridge["negative_rows"] == canonical["negative_rows"]
                             and math.isclose(full_bridge["negative_quantity_sum"], canonical["negative_quantity_sum"], rel_tol=1e-12, abs_tol=1e-9))
    primary_rows = frame[frame["origin"].isin(primary)]
    scored = primary_rows[primary_rows["week_scored"].to_numpy(dtype=bool)]
    neg_scored = scored[scored["negative_in_history"].to_numpy(dtype=bool)]
    negative_report = {
        "full_history": {"canonical_negative_rows": canonical["negative_rows"], "canonical_negative_quantity_sum": canonical["negative_quantity_sum"],
                         "bridged_negative_rows": full_bridge["negative_rows"], "bridged_negative_quantity_sum": full_bridge["negative_quantity_sum"],
                         "series_with_negative": full_bridge["negative_series"], "values_preserved_unchanged": returns_preserved},
        "per_origin": {str(o): {k: (sum(n[k] for n in parts) if k != "counterfactual_reasons"
                                    else dict(sum((Counter(n[k]) for n in parts), Counter()))) for k in parts[0]}
                       for o, parts in negatives.items() if parts},
        "primary_scope": {"series_origins_with_negative": int(primary_rows["negative_in_history"].sum()),
                          "share_of_series_origins": float(primary_rows["negative_in_history"].mean()) if len(primary_rows) else None,
                          "scored_actual_volume_share": float(neg_scored["actual_7d"].sum() / scored["actual_7d"].sum()) if scored["actual_7d"].sum() > 0 else None,
                          "reasons": {str(k): int(v) for k, v in primary_rows.loc[primary_rows["negative_in_history"], "reason"].value_counts().items()},
                          "total_7d_by_negative_history": segments(scope, "negative_history")},
        "policy": "negative values are never modified; the router routes such series to V1 (negative_sales) under the frozen rule",
    }
    young = primary_rows[(primary_rows["days_since_first_sale"] >= 0) & (primary_rows["days_since_first_sale"] < router.MIN_SALES_AGE_DAYS)]
    cold_report = {
        "definition": f"first positive sale fewer than {router.MIN_SALES_AGE_DAYS} days before the cutoff (frozen MIN_SALES_AGE_DAYS); "
                      "also reported: first history row fewer than 30 days before the cutoff",
        "series_origins": int(len(young)), "routed_v1_share": float((young["version"] == router.VERSION_V1).mean()) if len(young) else None,
        "reasons": {str(k): int(v) for k, v in young["reason"].value_counts().items()},
        "total_7d_by_age_band": segments(scope, "age_band"), "total_7d_by_first_row_age": segments(scope, "first_row_age"),
        "fallback_as_intended": bool(len(young) == 0 or (young["version"] == router.VERSION_V1).all()),
    }
    per_origin = []
    for spec in inputs.origins:
        name = f"origin_{spec['origin']:02d}"
        wape = {m: float(_metric(metrics, name, "total_7d", m)["wape"]) for m in METHODS}
        rows = frame[frame["origin"] == spec["origin"]]
        per_origin.append({"origin": spec["origin"], "phase": spec["phase"], "cutoff": str(spec["cutoff"]),
                           "targets": [str(spec["cutoff"] + DAY), str(spec["cutoff"] + HORIZON * DAY)],
                           "series": int(len(rows)), "scored_series": int(rows["week_scored"].sum()),
                           "total_7d_wape": wape, "router_better_than_v1": wape[ROUTED] < wape[V1],
                           "v2_series_share": float((rows["version"] == router.VERSION_V2).mean()),
                           "integrity": per_origin_integrity.get(spec["origin"]), "checks": merged.get(spec["origin"])})
    return {
        "evaluation": {"name": f"Varo forecast router - external generalisation on {dataset}", "version": EVALUATION_VERSION,
                       "dataset": dataset, "generated_at": _now(), "git_commit": _git_commit(),
                       "command": f"python -m services.external_forecast_validation --dataset {dataset}"},
        "FROZEN_CONFIG_UNCHANGED": bool(before["FROZEN_CONFIG_UNCHANGED"] and after["FROZEN_CONFIG_UNCHANGED"]),
        "frozen_config": {"before": before, "after": after},
        "gate": {"gate_signature": freeze["gate_signature"], "frozen_at": freeze.get("frozen_at"), "freeze_file": GATE_FREEZE_FILE,
                 "evidence": evidence, "evaluation": evaluation},
        "verdict": evaluation["verdict"],
        "data": {"profile": inputs.profile, "semantics": inputs.semantics, "quantity_semantics": semantics,
                 "first_day": str(inputs.first_day), "last_day": str(inputs.last_day)},
        "protocol": {"definition": {**PROTOCOLS["common"], **PROTOCOLS.get(dataset, {})},
                     "origins": [{"origin": o["origin"], "phase": o["phase"], "cutoff": str(o["cutoff"]),
                                  "targets": [str(o["cutoff"] + DAY), str(o["cutoff"] + HORIZON * DAY)]} for o in inputs.origins],
                     "scopes": inputs.scopes, "primary_scope": scope},
        "bridge": full_bridge,
        "integrity": {"criteria": integrity, "leakage_checks": LEAKAGE_CHECKS, "fallback_checks": FALLBACK_CHECKS},
        "headline": {s: {w: methods_block(s, w) for w in WINDOWS} for s in inputs.scopes},
        "per_origin": per_origin,
        "demand_type": by_type[by_type["scope"] == scope].to_dict(orient="records"),
        "coverage": {"primary_scope": scope,
                     "scored_series": {"router_version": coverage_block(scope, "scored_series", "router_version"),
                                       "router_reason": coverage_block(scope, "scored_series", "router_reason")},
                     "all_series": {"router_version": coverage_block(scope, "all_series", "router_version"),
                                    "router_reason": coverage_block(scope, "all_series", "router_reason")}},
        "negative_returns": negative_report,
        "cold_start": cold_report,
        "target_censoring": segments(scope, "target_censoring"),
        "production_bridge_e2e": e2e,
        "proxy_note": PROXY_NOTE,
        "runtime": {"total_seconds": round(time.perf_counter() - started, 3), "peak_memory_mb": peak_memory_mb(),
                    "stage_seconds": {k: round(v, 3) for k, v in seconds.items()}, "processed": dict(processed), "batches": batch_log},
        "files": {name: str(path) for name, path in paths.items()},
    }


# ---------------------------------------------------------------- forecast -> Varo decision path (verified against the code)

DECISION_PATH: tuple[dict[str, str], ...] = (
    {"step": "inventory analysis", "file": "services/analysis_pipeline.py", "token": "output, router_diagnostics = _route_forecast(runner, output, daily_sales_history)",
     "effect": "V1 demand_forecast_analyzer output -> forecast router: demand_forecast_7d/_daily/_d1..d7, upper/lower, demand_stockout_days, "
               "demand_risk_score, demand_forecast_score (V2 rows recomputed with V1's formulas), demand_trend, version/reason"},
    {"step": "store/product matching", "file": "services/legacy_adapters/_local_modules/store_product_matcher.py",
     "token": 'score += _safe(df["demand_risk_score"], 0.0) / 100.0 * 5', "effect": "demand_risk_score adds up to 5 urgency points"},
    {"step": "candidate frame", "file": "services/legacy_adapters/data_adapter.py", "token": 'candidates.merge(source_metrics, on=["source_id", "product_id"]',
     "effect": "the analysed inventory row of the SOURCE store (forecast fields included) is joined to every transfer candidate; the target "
               "store's forecast is not joined"},
    {"step": "VHS (legacy hybrid score)", "file": "services/legacy_adapters/_local_modules/varo_hybrid_score.py", "token": '"demand_forecast_score": 0.08',
     "effect": "demand_forecast_score is VHS component #6 (weight 0.08); demand_trend sets sit_DEMAND_SURGE"},
    {"step": "recommended action", "file": "services/legacy_adapters/_local_modules/varo_hybrid_score.py",
     "token": 'demand_rs = float(row.get("demand_risk_score",  50))',
     "effect": "_recommend_action (disposal > transfer > discount > hold) reads demand_risk_score (disposal only when < 40) and demand_trend"},
    {"step": "auto VHS", "file": "services/vhs_score_engine.py", "token": '_series(df, ("demand_forecast_7d",), 0)',
     "effect": "demand_fit_score = normalised demand_forecast_7d + 0.12 x sales_30d + 0.08 x recovered_margin"},
    {"step": "not reached: inventory transition / optimality gap / sensitivity", "file": "services/inventory_transition_service.py",
     "token": '("demand_forecast_7d", "sales_7d")',
     "effect": "these read the uploaded workbook (session varo_data), not the analysed inventory: they fall back to sales_7d, so no computed "
               "forecast (V1 or V2) reaches them"},
)


def decision_path() -> list[dict[str, Any]]:
    """DECISION_PATH with a check that every cited line still exists in the cited file."""
    return [{**step, "verified_in_code": step["token"] in (PROJECT_ROOT / step["file"]).read_text(encoding="utf-8")} for step in DECISION_PATH]


# ---------------------------------------------------------------- daily-history support matrix and combined summary

SUPPORT_COLUMNS = ("dataset", "dataset_role", "operational_use", "daily_history_support", "quantity_semantics", "bridge_input",
                   "bridge_series", "router_v2_series_share", "router_v2_scored_volume_share", "forecast_validation", "notes")
SUPPORT_RULE = {
    "daily_history_support": {"FULL": "daily location x product sales whose quantity semantics are FULL",
                              "PARTIAL": "a daily location x product quantity series exists, with a limitation (proxy, unit, absent zero days, "
                                         "aggregate locations, netted returns, censoring)",
                              "UNAVAILABLE": "no daily location x product quantity series"},
    "operational_use": "production_operational = Varo's domestic operating data; external_benchmark = public validation data only",
}


# documentary evidence of the domestic sources (canonical_data_pipeline coverage review); everything else is measured
DOMESTIC_OBSERVED_SALES = {"logisall": (True, "canonical coverage review: actual zone-daily sales (KADX LogisAll)"),
                           "jangbogo": (True, "canonical coverage review: actual warehouse x category monthly sales (KADX Jangbogo)")}


def _domestic_router_share(data_root: Path, dataset: str) -> dict[str, Any]:
    """Bridge a domestic canonical demand_series and ask the frozen router which series it would route to V2 at the last date."""
    import pyarrow.parquet as pq
    path = _canonical_path(data_root, dataset)
    if not path.exists():
        return {"bridge_input": "no canonical demand_series", "bridge_series": 0, "quantity_semantics": None}
    table = pq.read_table(path, columns=_canonical_columns(path))
    facts = _new_facts()
    _accumulate(facts, table)
    bridged = canonical_to_daily_history(table.to_pandas(), keep_row_lineage=False)
    semantics = measured_semantics(facts, DOMESTIC_OBSERVED_SALES.get(dataset, (False, "no documented sales semantics")), complete_grid=None)
    out = {"bridge_input": "canonical demand_series", "bridge_series": int(bridged.report["series"]),
           "bridge_quarantine": {k: v for k, v in bridged.report["quarantined_rows"].items() if v},
           "facts": _facts_record(facts), "semantics": semantics, "quantity_semantics": classify_semantics(semantics)}
    if bridged.report["series"]:
        keys = bridged.series[["store_id", "product_id"]].reset_index(drop=True)
        assessment = router.assess_history(keys, bridged.history, as_of=bridged.report["date_range"][1])
        reasons = pd.Series(assessment.reason, dtype=object).value_counts()
        out.update(as_of=bridged.report["date_range"][1], router_reasons={str(k): int(v) for k, v in reasons.items()},
                   router_v2_series_share=float(reasons.get(router.REASON_V2, 0) / len(keys)))
    return out


def daily_history_support_matrix(data_root: Path | str, reports: Mapping[str, Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_root)
    measured = {d: _domestic_router_share(root, d) for d in ("suhyup", "logisall", "jangbogo")}
    suhyup_flow = root / DATASETS["suhyup"] / "processed" / "canonical_inventory_flow.parquet"
    flow_days = None
    if suhyup_flow.exists():
        import pyarrow.parquet as pq
        flow = pq.read_table(suhyup_flow, columns=["date", "outbound_qty", "sales_qty"]).to_pandas()
        flow_days = {"rows": int(len(flow)), "dates": int(flow["date"].nunique()), "date_range": [flow["date"].min(), flow["date"].max()],
                     "outbound_rows": int(flow["outbound_qty"].notna().sum()), "sales_rows": int(flow["sales_qty"].notna().sum())}
    m5_dir = root / DATASETS["m5"] / "results"
    gate = json.loads((m5_dir / "m5_forecast_v2_promotion_gate.json").read_text(encoding="utf-8")) if (m5_dir / "m5_forecast_v2_promotion_gate.json").exists() else {}
    path_report = json.loads((m5_dir / "m5_forecast_router_production_path.json").read_text(encoding="utf-8")) if (m5_dir / "m5_forecast_router_production_path.json").exists() else {}
    m5_final = path_report.get("total_7d", {}).get("final_rolling_pooled", {}).get("all", {})
    m5_last = (path_report.get("per_origin") or [{}])[-1]
    m5_counts = m5_last.get("counts_by_reason", {})

    def ext(dataset: str) -> dict[str, Any]:
        r = reports.get(dataset)
        if not r:
            return {"quantity_semantics": "not run", "router_v2_series_share": None, "router_v2_scored_volume_share": None, "forecast_validation": "not run"}
        cov = r["coverage"]["scored_series"]["router_version"].get(router.VERSION_V2, {})
        allv = r["coverage"]["all_series"]["router_version"].get(router.VERSION_V2, {})
        return {"quantity_semantics": r["data"]["quantity_semantics"], "bridge_series": r["bridge"]["series"],
                "router_v2_series_share": allv.get("series_origin_share"), "router_v2_scored_volume_share": cov.get("volume_share"),
                "forecast_validation": f"{r['verdict']} (external gate {r['gate']['gate_signature'][:12]}); limitations: "
                                       f"{'; '.join(r['gate']['evaluation']['limitations']) or 'none'}"}

    rows = [
        {"dataset": "suhyup", "dataset_role": "production_operational", "operational_use": "yes: Suhyup 31-day x 6-strategy benchmark",
         "daily_history_support": "PARTIAL", "quantity_semantics": "UNAVAILABLE (outbound shipments, not sales)",
         "bridge_input": measured["suhyup"]["bridge_input"], "bridge_series": 0, "router_v2_series_share": None,
         "router_v2_scored_volume_share": None, "forecast_validation": "not validated: no sales history",
         "notes": f"daily center x product outbound flow only ({flow_days}); outbound is a demand proxy, not units sold, so it is not bridged "
                  "into daily_sales_history (the bridge maps sales_qty / demand_qty only)"},
        {"dataset": "logisall", "dataset_role": "production_operational", "operational_use": "yes: domestic KADX logistics data (zone level)",
         "daily_history_support": "PARTIAL", "quantity_semantics": measured["logisall"]["quantity_semantics"],
         "bridge_input": measured["logisall"]["bridge_input"], "bridge_series": measured["logisall"]["bridge_series"],
         "router_v2_series_share": measured["logisall"].get("router_v2_series_share"), "router_v2_scored_volume_share": None,
         "forecast_validation": "not validated (no gate run); router assessment only",
         "notes": f"7 first-digit ZIP zones, not stores; zero-sale days absent; unit UNKNOWN; router reasons at {measured['logisall'].get('as_of')}: "
                  f"{measured['logisall'].get('router_reasons')}"},
        {"dataset": "jangbogo", "dataset_role": "production_operational", "operational_use": "yes: domestic KADX warehouse data (monthly)",
         "daily_history_support": "UNAVAILABLE", "quantity_semantics": measured["jangbogo"]["quantity_semantics"],
         "bridge_input": measured["jangbogo"]["bridge_input"], "bridge_series": measured["jangbogo"]["bridge_series"],
         "router_v2_series_share": None, "router_v2_scored_volume_share": None, "forecast_validation": "not applicable",
         "notes": f"every row quarantined by the bridge: {measured['jangbogo'].get('bridge_quarantine')}"},
        {"dataset": "m5", "dataset_role": "external_benchmark", "operational_use": "no", "daily_history_support": "FULL",
         "quantity_semantics": "FULL (canonical semantic review 2026-09-29)", "bridge_input": "canonical demand_series (validated through the router's "
                                                                                              "long-history path)",
         "bridge_series": None, "router_v2_series_share": (m5_counts.get(router.REASON_V2, 0) / sum(m5_counts.values())) if m5_counts else None,
         "router_v2_scored_volume_share": None,
         "forecast_validation": f"V2 promotion gate {gate.get('decision')}; router final rolling 7-day WAPE V1 {m5_final.get(V1, {}).get('wape')} -> "
                                f"router {m5_final.get('varo_router_production', {}).get('wape')}",
         "notes": "training/selection data of V2; V2 share is the official holdout origin of the production-path run"},
    ]
    for dataset in ("favorita", "freshretailnet"):
        e = ext(dataset)
        rows.append({"dataset": dataset, "dataset_role": "external_benchmark", "operational_use": "no", "daily_history_support": "PARTIAL",
                     "bridge_input": "canonical demand_series", **e,
                     "notes": ("absent rows are missing days (zero sales not recorded); signed sales (returns); unit UNKNOWN" if dataset == "favorita"
                               else "complete day grid with zeros; normalised sales amount; stock-out censoring")})
    for row in rows:
        row.setdefault("bridge_series", None)
    return pd.DataFrame(rows)[list(SUPPORT_COLUMNS)], {"measured": measured, "suhyup_flow": flow_days, "rule": SUPPORT_RULE}


def write_summary(data_root: Path | str = DATA_ROOT, common_dir: Path | str | None = None,
                  dataset_dirs: Mapping[str, Path | str] | None = None) -> dict[str, Any]:
    """Combined generalisation verdict and the daily-history support matrix from the two dataset reports."""
    root = Path(data_root)
    common = Path(common_dir) if common_dir else root / COMMON_RESULTS
    frozen = json.loads((common / GATE_FREEZE_FILE).read_text(encoding="utf-8"))
    reports = {}
    for dataset in ("freshretailnet", "favorita"):
        folder = Path(dataset_dirs[dataset]) if dataset_dirs and dataset in dataset_dirs else root / DATASETS[dataset] / "results"
        path = folder / f"{FILE_PREFIX[dataset]}{RESULT_SUFFIXES['json']}"
        if path.exists():
            reports[dataset] = json.loads(path.read_text(encoding="utf-8"))
    if set(reports) != {"freshretailnet", "favorita"}:
        raise FileNotFoundError(f"dataset reports missing: {sorted({'freshretailnet', 'favorita'} - set(reports))}")
    gate_same = all(r["gate"]["gate_signature"] == frozen["gate_signature"] == gate_signature() for r in reports.values())
    results = {d: r["gate"]["evaluation"] for d, r in reports.items()}
    verdict = combined_verdict(results)
    check = frozen_config_check()
    matrix, matrix_detail = daily_history_support_matrix(root, reports)
    _write_csv(matrix, common / SUPPORT_MATRIX_FILE)
    summary = {
        "evaluation": {"name": "Varo forecast router - external generalisation (Favorita + FreshRetailNet)", "version": EVALUATION_VERSION,
                       "generated_at": _now(), "git_commit": _git_commit(), "command": "python -m services.external_forecast_validation --summary"},
        "FROZEN_CONFIG_UNCHANGED": bool(check["FROZEN_CONFIG_UNCHANGED"] and all(r["FROZEN_CONFIG_UNCHANGED"] for r in reports.values())),
        "frozen_config": check["record"],
        "gate": {"gate_signature": frozen["gate_signature"], "frozen_at": frozen.get("frozen_at"), "same_gate_in_every_report": gate_same,
                 "rule": EXTERNAL_GATE},
        "datasets": {d: {"verdict": r["verdict"], "improved": r["gate"]["evaluation"]["improved"],
                         "limitations": r["gate"]["evaluation"]["limitations"], "failed": r["gate"]["evaluation"]["failed"],
                         "quantity_semantics": r["data"]["quantity_semantics"], "primary_scope": r["protocol"]["primary_scope"],
                         "total_7d": {m: r["headline"][r["protocol"]["primary_scope"]]["total_7d"][m]["wape"] for m in METHODS},
                         "v2_scored_volume_share": r["gate"]["evidence"]["coverage_share"],
                         "e2e_pass": (r.get("production_bridge_e2e") or {}).get("pass")} for d, r in reports.items()},
        "GENERALIZATION_VERDICT": verdict if gate_same else "FAIL",
        "verdict_rule": EXTERNAL_GATE["combined_verdict"],
        "daily_history_support_matrix": matrix.to_dict(orient="records"),
        "support_matrix_detail": matrix_detail,
        "forecast_to_decision_path": decision_path(),
        "files": {"generalization": str(common / GENERALIZATION_FILE), "support_matrix": str(common / SUPPORT_MATRIX_FILE)},
    }
    (common / GENERALIZATION_FILE).write_text(json.dumps(_clean(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return _clean(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(LOADERS))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--common-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.dataset and not args.summary:
        parser.error("give --dataset and/or --summary")
    if args.dataset:
        report = run_dataset(args.dataset, args.data_root, args.output_dir, args.common_dir)
        print(json.dumps({"dataset": args.dataset, "verdict": report["verdict"], "FROZEN_CONFIG_UNCHANGED": report["FROZEN_CONFIG_UNCHANGED"],
                          "gate": {c["criterion"]: c["pass"] for c in report["gate"]["evaluation"]["criteria"]},
                          "headline": {s: {m: v["wape"] for m, v in b["total_7d"].items()} for s, b in report["headline"].items()},
                          "runtime": {k: report["runtime"][k] for k in ("total_seconds", "peak_memory_mb")}}, ensure_ascii=False, indent=2))
    if args.summary:
        summary = write_summary(args.data_root, args.common_dir)
        print(json.dumps({k: summary[k] for k in ("GENERALIZATION_VERDICT", "FROZEN_CONFIG_UNCHANGED", "datasets")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
