"""External validation of the production Varo demand forecast on Walmart M5.

Scores the forecast the analysis pipeline actually runs
(``demand_forecast_analyzer.analyze_demand_forecast``, loaded through the legacy
allowlist and fed through ``prepare_legacy_data``) against fixed, pre-registered
baselines on observed M5 store x item daily unit sales. Only chronological
origins are used: every forecast, class and scale at a cutoff is computed from
the history up to that cutoff, never from later sales or prices.

Read-only on raw and processed data; writes only the ``m5_forecast_*`` files.
No inventory is generated: M5 has none, so the stock-dependent outputs of the
production function (risk score, stock-out days) are discarded unscored.

    python -m services.m5_forecast_validation --data-root C:/VARO_V2_REAL_DATA
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from services.dqn_guard import strip_dqn_columns
from services.legacy_adapters.data_adapter import prepare_legacy_data
from services.legacy_adapters.loader import legacy_module_path, load_legacy_module
from services.real_data_adapters import DATA_ROOT, DATASETS

EVALUATION_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODULE = "demand_forecast_analyzer"
RAW_DIR = Path("raw/extracted/m5-forecasting-accuracy")
RAW_INPUTS = ("calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv")
CANONICAL_INPUTS = ("canonical_demand_series.parquet", "canonical_product_master.parquet", "canonical_location_master.parquet",
                    "canonical_calendar_event.parquet", "canonical_covariate_series.parquet")

# Pre-registered before any score was computed; nothing below is selected on validation or holdout results.
HORIZON = 28                # M5 official horizon: the holdout d_1914..d_1941 is exactly 28 days
OPERATIONAL_HORIZON = 7     # Varo contract: demand_forecast_7d = "향후 7일 예측 판매량"; downstream needs are 7-day
ORIGIN_COUNT = 13           # 13 x 28 = 364 target days: one full year (every weekday, month and holiday once)
ORIGIN_STEP = 28            # = HORIZON, so target windows tile the final year without overlap
RECENT_WINDOW = 7           # production input sales_7d
HISTORY_WINDOW = 30         # production input sales_30d; avg_daily_sales = sales_30d / 30 (sample workbook convention)
MOVING_AVERAGE_WINDOW = 28  # 4 whole weeks: every weekday weighted equally
PROFILE_WINDOW = 364        # demand-type window: 52 whole weeks before the cutoff
PRICE_CHANGE = 0.05         # diagnostic grouping only (post-cutoff price is never a forecast input)
EXACT_TOLERANCE = 1e-9

METHODS: dict[str, dict[str, str]] = {
    "varo_production": {"role": "varo_production", "definition": (
        "analyze_demand_forecast with sales_7d, avg_daily_sales, sales_30d, demand_std (the sample workbook inventory schema): "
        "daily = 0.6*sales_7d/7 + 0.4*avg_daily_sales, 7d = round(7*daily, 1); per row NAIVE fallback daily = avg_daily_sales "
        "when sales_7d = 0; flat over the horizon")},
    "varo_branch_naive30": {"role": "varo_variant", "definition": (
        "Same production function when the upload has no sales_7d column: NAIVE branch, daily = avg_daily_sales (30-day mean)")},
    "varo_branch_sma7": {"role": "varo_variant", "definition": (
        "Same production function when the upload has only sales_7d: SMA branch, daily = sales_7d / 7 (7-day mean)")},
    "naive_last": {"role": "baseline", "definition": "y[T] (last observed day) repeated over the horizon"},
    "seasonal_naive_7": {"role": "baseline", "definition": "y[T+h-7*ceil(h/7)]: the same weekday of the last observed week"},
    "moving_average_28": {"role": "baseline", "definition": "mean(y[T-27..T]) flat over the horizon"},
    "zero_reference": {"role": "reference", "definition": (
        "0 every day. Not a candidate: shown because MAE is minimised by the median, which is 0 for most M5 series")},
}
VARO_INPUT_COLUMNS = {
    "varo_production": ("sales_7d", "avg_daily_sales", "sales_30d", "demand_std"),
    "varo_branch_naive30": ("avg_daily_sales", "sales_30d", "demand_std"),
    "varo_branch_sma7": ("sales_7d",),
}
METHOD_ORDER = list(METHODS)
CANDIDATES = [m for m, spec in METHODS.items() if spec["role"] != "reference"]
BASELINES = [m for m, spec in METHODS.items() if spec["role"] == "baseline"]

WINDOWS = {
    "daily_h1_7": "daily points, horizon days 1-7 (Varo operational horizon)",
    "daily_h1_28": "daily points, horizon days 1-28 (M5 official horizon; Varo's daily rate held flat)",
    "total_7d": "one 7-day total per series (Varo demand_forecast_7d vs actual days 1-7)",
}
WINDOW_ORDER = list(WINDOWS)

# Selling-day share p over the profile window, counted from the first sale. ADI = 1/p.
FREQUENCY_CLASSES = ("high_frequency", "medium_frequency", "intermittent", "mostly_zero", "no_sales_history")
FREQUENCY_RULES = {
    "high_frequency": "p >= 1/1.32 (ADI < 1.32, the Syntetos-Boylan non-intermittent boundary)",
    "medium_frequency": "0.5 <= p < 1/1.32 (ADI 1.32-2)",
    "intermittent": "0.1 <= p < 0.5 (ADI 2-10)",
    "mostly_zero": "p < 0.1 (ADI >= 10, including no sale in the window)",
    "no_sales_history": "never sold up to the cutoff (every method forecasts 0)",
}
VOLUME_BANDS = ("0 (no sale in window)", "(0,0.25)/day", "[0.25,1)/day", "[1,3)/day", "[3,10)/day", ">=10/day")
SBC_CLASSES = ("smooth", "erratic", "intermittent", "lumpy", "undefined")
AGE_BANDS = ("never_sold", "<30d", "30-89d", "90-363d", ">=364d")
VARO_METHOD_LABELS = ("WMA", "NAIVE", "SMA")
VARO_TREND_LABELS = ("INCREASING", "STABLE", "DECREASING")
PRICE_STATUS = ("price_absent(not_on_sale)", "price_cut>=5%", "price_up>=5%", "price_within_5%", "no_prior_price")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

SUM_FLOAT = ("abs_err", "sq_err", "err", "forecast", "actual", "over_units", "under_units", "mase_sum", "rmsse_sum")
SUM_COUNT = ("n", "over_n", "under_n", "scored", "mase_n", "in_interval")
SUM_COLUMNS = (*SUM_COUNT, *SUM_FLOAT)
BIAS_COLUMNS = ("n", "actual", "forecast", "mean_error", "bias_units", "bias_pct", "over_forecast_rate", "under_forecast_rate",
                "exact_rate", "over_units", "under_units")

OUTPUT_FILES = {
    "summary": "m5_forecast_validation_summary.csv",
    "by_cutoff": "m5_forecast_by_cutoff.csv",
    "by_store": "m5_forecast_by_store.csv",
    "by_category": "m5_forecast_by_category.csv",
    "by_demand_type": "m5_forecast_by_demand_type.csv",
    "bias": "m5_forecast_bias.csv",
    "by_horizon": "m5_forecast_by_horizon.csv",
    "by_calendar": "m5_forecast_by_calendar.csv",
    "json": "m5_forecast_validation.json",
    "series_profile": "m5_forecast_series_profile.parquet",
    "rows": "m5_forecast_rows.parquet",
}


# ---------------------------------------------------------------- data

@dataclass
class M5Panel:
    sales: np.ndarray        # (series, days) float64: NaN = missing cell, 0 = observed zero sale
    price: np.ndarray        # (series, days) float32: NaN = no weekly price (guide: not sold that week)
    dates: np.ndarray        # contiguous ISO dates
    series: pd.DataFrame     # store_id, item_id, state_id, category, department (matrix row order)
    event_label: np.ndarray  # per day: "none" or "+"-joined event types
    snap: np.ndarray         # (series, days) SNAP indicator of the series' state; NaN if unknown
    canonical_rows: int


def _sorted_codes(column: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted distinct values and each row's int32 rank, without materialising one string per row."""
    encoded = column.combine_chunks() if pa.types.is_dictionary(column.type) else pc.dictionary_encode(column).combine_chunks()
    values = np.array(encoded.dictionary.to_pylist(), dtype=object)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order), dtype=np.int32)
    return values[order], rank[encoded.indices.to_numpy()]


def load_panel(folder: Path) -> M5Panel:
    """Dense (series x day) view of canonical demand_series; every cell must appear exactly once."""
    processed = folder / "processed"
    keys = ["date", "location_id", "product_id"]
    table = pq.read_table(processed / CANONICAL_INPUTS[0], columns=[*keys, "sales_qty", "price"], read_dictionary=keys).unify_dictionaries()
    dates, day = _sorted_codes(table["date"])
    stores, store = _sorted_codes(table["location_id"])
    items, item = _sorted_codes(table["product_id"])
    pair = store.astype(np.int64) * len(items) + item
    del store, item
    pairs = np.flatnonzero(np.bincount(pair, minlength=len(stores) * len(items)))   # observed (store, item) pairs, sorted
    position = np.full(len(stores) * len(items), -1, dtype=np.int64)
    position[pairs] = np.arange(len(pairs))
    flat = position[pair] * len(dates) + day
    del pair, day
    if not (np.bincount(flat, minlength=len(pairs) * len(dates)) == 1).all():
        raise ValueError("canonical demand_series is not one row per (store, item, date)")
    parsed = pd.to_datetime(pd.Series(dates))
    if len(dates) > 1 and not parsed.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("demand_series dates are not contiguous days")
    sales = np.empty(len(pairs) * len(dates))
    sales[flat] = table["sales_qty"].to_numpy()
    price = np.empty(len(pairs) * len(dates), dtype=np.float32)
    price[flat] = table["price"].to_numpy()
    rows = table.num_rows
    del table, flat
    frame = pd.DataFrame({"store_id": stores[pairs // len(items)], "item_id": items[pairs % len(items)]})

    products = pd.read_parquet(processed / CANONICAL_INPUTS[1], columns=["product_id", "category", "category_path"])
    products["department"] = products["category_path"].str.extract(r"dept_id=([^|]+)", expand=False)
    locations = pd.read_parquet(processed / CANONICAL_INPUTS[2], columns=["location_id", "location_type", "region"])
    locations = locations[locations["location_type"] == "store"]
    frame = frame.merge(products[["product_id", "category", "department"]], left_on="item_id", right_on="product_id",
                        how="left", validate="many_to_one").drop(columns="product_id")
    frame = frame.merge(locations[["location_id", "region"]].rename(columns={"region": "state_id"}), left_on="store_id",
                        right_on="location_id", how="left", validate="many_to_one").drop(columns="location_id")
    if frame[["category", "department", "state_id"]].isna().any().any():
        raise ValueError("series without product/location master attributes")

    events = pd.read_parquet(processed / CANONICAL_INPUTS[3], columns=["date", "event_type"])
    labels = events.groupby("date")["event_type"].agg(lambda s: "+".join(sorted(set(s))))
    event_label = pd.Series(dates).map(labels).fillna("none").to_numpy(dtype=object)
    covariates = pd.read_parquet(processed / CANONICAL_INPUTS[4], columns=["date", "location_id", "covariate_name", "covariate_value"])
    snap_table = covariates[covariates["covariate_name"] == "snap"].pivot(index="location_id", columns="date", values="covariate_value")
    snap_table = snap_table.reindex(columns=dates)
    snap = snap_table.reindex(frame["state_id"]).to_numpy(dtype=np.float32)
    return M5Panel(sales.reshape(len(pairs), len(dates)), price.reshape(len(pairs), len(dates)), dates.astype(str), frame,
                   event_label, snap, rows)


def raw_crosscheck(folder: Path, panel: M5Panel) -> dict[str, Any]:
    """Read-only cell-for-cell comparison of the evaluated matrix with raw sales_train_evaluation.csv."""
    raw = folder / RAW_DIR
    calendar = pd.read_csv(raw / "calendar.csv", usecols=["d", "date"], dtype=str)
    sales = pd.read_csv(raw / "sales_train_evaluation.csv", dtype={c: str for c in ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")})
    days = [c for c in sales.columns if c.startswith("d_")]
    raw_dates = calendar.set_index("d").loc[days, "date"].to_numpy(dtype=str)
    lookup = {(s, i): k for k, (s, i) in enumerate(zip(panel.series["store_id"], panel.series["item_id"]))}
    position = np.array([lookup.get(key, -1) for key in zip(sales["store_id"], sales["item_id"])])
    matrix = sales[days].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    same_series = bool(len(position) == len(panel.series) and (position >= 0).all() and len(set(position)) == len(position))
    equal = bool(same_series and np.array_equal(raw_dates, panel.dates)
                 and np.array_equal(panel.sales[position], matrix, equal_nan=True))
    departments = sales.set_index(["store_id", "item_id"])["dept_id"]
    dept_equal = bool(same_series and np.array_equal(departments.to_numpy(dtype=str), panel.series["department"].to_numpy(dtype=str)[position]))
    return {"raw_file": str(RAW_DIR / "sales_train_evaluation.csv").replace("\\", "/"), "raw_series": len(sales), "raw_days": len(days),
            "same_series_set": same_series, "calendar_d_to_date_matches": bool(np.array_equal(raw_dates, panel.dates)),
            "department_matches": dept_equal, "cells_equal_including_missing": equal,
            "raw_missing_cells": int(np.isnan(matrix).sum()), "raw_sum": float(np.nansum(matrix))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprints(folder: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {name: {"bytes": (folder / name).stat().st_size, "mtime_ns": (folder / name).stat().st_mtime_ns, "sha256": _sha256(folder / name)}
            for name in names if (folder / name).exists()}


# ---------------------------------------------------------------- split

def rolling_origins(n_days: int, horizon: int = HORIZON, count: int = ORIGIN_COUNT, step: int = ORIGIN_STEP,
                    min_history: int = HISTORY_WINDOW) -> list[dict[str, Any]]:
    """Chronological origins ending at the last day: the last is the holdout, the one before it the validation."""
    if horizon < OPERATIONAL_HORIZON or step < horizon or count < 3:
        raise ValueError("need horizon >= 7, step >= horizon (non-overlapping targets) and at least 3 origins")
    last = n_days - 1 - horizon
    cutoffs = [last - step * k for k in reversed(range(count))]
    if cutoffs[0] + 1 < min_history:
        raise ValueError(f"first cutoff has {cutoffs[0] + 1} history days; {min_history} are required")
    roles = ["rolling_backtest"] * (count - 2) + ["validation", "holdout"]
    return [{"origin": k + 1, "role": role, "cutoff": cutoff, "target_start": cutoff + 1, "target_end": cutoff + horizon}
            for k, (cutoff, role) in enumerate(zip(cutoffs, roles))]


# ---------------------------------------------------------------- history-only features

def naive_scales(history: np.ndarray, first: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """M5 RMSSE/MASE scale: mean squared / absolute one-step naive difference from the first sale to the cutoff."""
    n, t = history.shape
    mae, mse = np.full(n, np.nan), np.full(n, np.nan)
    later = np.arange(1, t)[None, :]
    for lo in range(0, n, chunk):
        diff = np.diff(history[lo:lo + chunk], axis=1)
        keep = (later > first[lo:lo + chunk, None]) & ~np.isnan(diff)
        count = keep.sum(axis=1)
        d = np.where(keep, diff, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mae[lo:lo + chunk] = np.where(count > 0, np.abs(d).sum(axis=1) / count, np.nan)
            mse[lo:lo + chunk] = np.where(count > 0, (d * d).sum(axis=1) / count, np.nan)
    return mae, mse


def series_profile(history: np.ndarray) -> dict[str, np.ndarray]:
    """Demand type, volume, age and error scales from the history up to the cutoff only."""
    n, t = history.shape
    observed = ~np.isnan(history)
    sold = np.where(observed, history, 0.0) > 0
    ever = sold.any(axis=1)
    first = np.where(ever, sold.argmax(axis=1), t)
    age = np.where(ever, t - first, 0)
    width = min(PROFILE_WINDOW, t)
    window = np.where(observed[:, t - width:], history[:, t - width:], 0.0)
    active_mask = observed[:, t - width:] & (np.arange(t - width, t)[None, :] >= first[:, None])
    selling_mask = active_mask & (window > 0)
    active, selling = active_mask.sum(axis=1), selling_mask.sum(axis=1)
    share = np.divide(selling, active, out=np.zeros(n), where=active > 0)
    mean_daily = np.divide(np.where(active_mask, window, 0.0).sum(axis=1), active, out=np.zeros(n), where=active > 0)
    s1 = np.where(selling_mask, window, 0.0).sum(axis=1)
    s2 = np.where(selling_mask, window * window, 0.0).sum(axis=1)
    size_mean = np.divide(s1, selling, out=np.zeros(n), where=selling > 0)
    cv2 = np.divide(np.divide(s2, selling, out=np.zeros(n), where=selling > 0) - size_mean ** 2, size_mean ** 2,
                    out=np.full(n, np.nan), where=size_mean > 0)
    adi = np.divide(active, selling, out=np.full(n, np.inf), where=selling > 0)

    frequency = np.select([~ever, share >= 1 / 1.32, share >= 0.5, share >= 0.1], [4, 0, 1, 2], default=3)
    volume = np.select([mean_daily <= 0, mean_daily < 0.25, mean_daily < 1, mean_daily < 3, mean_daily < 10], [0, 1, 2, 3, 4], default=5)
    sbc = np.select([selling < 2, (adi < 1.32) & (cv2 < 0.49), adi < 1.32, cv2 < 0.49], [4, 0, 1, 2], default=3)
    age_band = np.select([~ever, age < 30, age < 90, age < 364], [0, 1, 2, 3], default=4)
    mae_scale, mse_scale = naive_scales(history, first)
    eligible = observed[:, -HISTORY_WINDOW:].all(axis=1) if t >= HISTORY_WINDOW else np.zeros(n, dtype=bool)
    return {"eligible": eligible, "days_since_first_sale": np.where(ever, age, -1), "selling_day_share": share, "adi": adi,
            "cv2_nonzero": cv2, "mean_daily_sales": mean_daily, "frequency_class": frequency, "volume_band": volume,
            "sbc_class": sbc, "age_band": age_band, "mae_scale": mae_scale, "mse_scale": mse_scale}


# ---------------------------------------------------------------- forecasts

def _flat(level: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(np.asarray(level, dtype=np.float64)[:, None], horizon, axis=1)


def naive_last(history: np.ndarray, horizon: int) -> np.ndarray:
    return _flat(history[:, -1], horizon)


def seasonal_naive_7(history: np.ndarray, horizon: int) -> np.ndarray:
    return history[:, -7:][:, np.arange(horizon) % 7].astype(np.float64)


def moving_average(history: np.ndarray, horizon: int, window: int = MOVING_AVERAGE_WINDOW) -> np.ndarray:
    return _flat(history[:, -window:].mean(axis=1), horizon)


def zero_reference(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.zeros((history.shape[0], horizon))


BASELINE_FUNCTIONS = {"naive_last": naive_last, "seasonal_naive_7": seasonal_naive_7,
                      "moving_average_28": moving_average, "zero_reference": zero_reference}


def varo_inputs(history: np.ndarray) -> pd.DataFrame:
    """The inventory-sheet demand fields as the sample workbooks define them, from the last 30 days only."""
    recent, month = history[:, -RECENT_WINDOW:], history[:, -HISTORY_WINDOW:]
    sales_30d = month.sum(axis=1)
    return pd.DataFrame({"sales_7d": recent.sum(axis=1), "avg_daily_sales": sales_30d / HISTORY_WINDOW,
                         "sales_30d": sales_30d, "demand_std": month.std(axis=1, ddof=1)})


def run_varo_forecast(inputs: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Production call path: prepare_legacy_data -> strip_dqn_columns -> allowlisted analyze_demand_forecast."""
    inventory = prepare_legacy_data({"inventory": inputs[list(columns)].copy()})["inventory"]
    output = load_legacy_module(PRODUCTION_MODULE).analyze_demand_forecast(strip_dqn_columns(inventory))
    return output[["demand_forecast_7d", "demand_forecast_daily", "demand_forecast_upper", "demand_forecast_lower",
                   "demand_trend", "demand_forecast_method"]].reset_index(drop=True)


def origin_forecasts(history: np.ndarray, horizon: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, float]:
    """Every method's daily forecasts and 7-day totals from ``history`` alone (the only data passed in)."""
    inputs = varo_inputs(history)
    started = time.perf_counter()
    varo = {name: run_varo_forecast(inputs, columns) for name, columns in VARO_INPUT_COLUMNS.items()}
    varo_seconds = time.perf_counter() - started
    daily = {name: _flat(out["demand_forecast_daily"].to_numpy(dtype=np.float64), horizon) for name, out in varo.items()}
    daily.update({name: function(history, horizon) for name, function in BASELINE_FUNCTIONS.items()})
    totals = {name: daily[name][:, :OPERATIONAL_HORIZON].sum(axis=1) for name in daily}
    totals.update({name: out["demand_forecast_7d"].to_numpy(dtype=np.float64) for name, out in varo.items()})
    return {m: daily[m] for m in METHOD_ORDER}, {m: totals[m] for m in METHOD_ORDER}, varo["varo_production"], inputs, varo_seconds


# ---------------------------------------------------------------- metrics

def elementwise_stats(forecast: np.ndarray, actual: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Additive error statistics; invalid (missing) points contribute nothing, never a zero actual."""
    err = np.where(valid, forecast - actual, 0.0)
    return {"n": valid.astype(np.int64), "abs_err": np.abs(err), "sq_err": err * err, "err": err,
            "forecast": np.where(valid, forecast, 0.0), "actual": np.where(valid, actual, 0.0),
            "over_n": (err > EXACT_TOLERANCE).astype(np.int64), "under_n": (err < -EXACT_TOLERANCE).astype(np.int64),
            "over_units": np.clip(err, 0.0, None), "under_units": np.clip(-err, 0.0, None)}


def finalize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE/WAPE/bias from summed statistics. WAPE and bias_pct are undefined when actual sales sum to 0."""
    out = frame.copy()
    n = out["n"].where(out["n"] > 0).astype(float)
    actual = out["actual"].where(out["actual"] > 0)
    out["mae"] = out["abs_err"] / n
    out["rmse"] = np.sqrt(out["sq_err"] / n)
    out["wape"] = out["abs_err"] / actual
    out["mean_error"] = out["err"] / n
    out["bias_units"] = out["forecast"] - out["actual"]
    out["bias_pct"] = out["bias_units"] / actual
    out["over_forecast_rate"] = out["over_n"] / n
    out["under_forecast_rate"] = out["under_n"] / n
    out["exact_rate"] = 1.0 - out["over_forecast_rate"] - out["under_forecast_rate"]
    if "mase_n" in out:
        scaled = out["mase_n"].where(out["mase_n"] > 0).astype(float)
        out["mase_mean"] = out["mase_sum"] / scaled
        out["rmsse_mean"] = out["rmsse_sum"] / scaled
    if "in_interval" in out and "method" in out:
        out["interval_coverage"] = (out["in_interval"] / n).where(out["method"].astype(str) == "varo_production")
    return out


def point_metrics(forecast: np.ndarray, actual: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """Pooled metrics for arrays of forecasts and actuals (NaN actuals are excluded, not zero-filled)."""
    forecast, actual = np.asarray(forecast, dtype=float), np.asarray(actual, dtype=float)
    valid = ~np.isnan(actual) if valid is None else valid & ~np.isnan(actual)
    stats = {k: float(v.sum()) for k, v in elementwise_stats(forecast, actual, valid).items()}
    row = finalize_metrics(pd.DataFrame([stats])).iloc[0]
    return {k: float(row[k]) for k in ("n", "mae", "rmse", "wape", "mean_error", "bias_units", "bias_pct", "over_forecast_rate",
                                         "under_forecast_rate", "exact_rate", "over_units", "under_units")}


def scaled_errors(abs_err: np.ndarray, sq_err: np.ndarray, n: np.ndarray, mae_scale: np.ndarray, mse_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-series MASE and RMSSE; NaN when the series has no scored point or a zero in-sample scale."""
    ok = (n > 0) & (mae_scale > 0) & (mse_scale > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mase = np.where(ok, abs_err / np.maximum(n, 1) / mae_scale, np.nan)
        rmsse = np.where(ok, np.sqrt(sq_err / np.maximum(n, 1) / mse_scale), np.nan)
    return mase, rmsse


# ---------------------------------------------------------------- one origin

def _series_rows(origin: int, method: str, window: str, stats: Mapping[str, np.ndarray], mase: np.ndarray, rmsse: np.ndarray,
                 in_interval: np.ndarray) -> dict[str, np.ndarray]:
    n = len(stats["n"])
    has_scale = ~np.isnan(mase)
    return {"origin": np.full(n, origin, dtype=np.int16), "method": np.full(n, METHOD_ORDER.index(method), dtype=np.int8),
            "window": np.full(n, WINDOW_ORDER.index(window), dtype=np.int8), "series": np.arange(n, dtype=np.int32),
            **{k: stats[k].astype(np.int32 if k in SUM_COUNT else np.float64) for k in stats},
            "scored": (stats["n"] > 0).astype(np.int32), "mase_n": has_scale.astype(np.int32),
            "mase_sum": np.where(has_scale, mase, 0.0), "rmsse_sum": np.where(has_scale, rmsse, 0.0),
            "in_interval": in_interval.astype(np.int32)}


def _price_status(price: np.ndarray, cutoff: int, horizon: int) -> np.ndarray:
    """Diagnostic only: target-day weekly price vs the last price known at the cutoff."""
    history = price[:, :cutoff + 1]
    has = ~np.isnan(history)
    last = history.shape[1] - 1 - has[:, ::-1].argmax(axis=1)
    reference = np.where(has.any(axis=1), history[np.arange(len(price)), last], np.nan)[:, None]
    target = price[:, cutoff + 1:cutoff + 1 + horizon]
    with np.errstate(invalid="ignore"):
        return np.select([np.isnan(target), np.isnan(reference), target <= reference * (1 - PRICE_CHANGE),
                          target >= reference * (1 + PRICE_CHANGE)], [0, 4, 1, 2], default=3)


def _group_points(stats: Mapping[str, np.ndarray], codes: np.ndarray, labels: Sequence[str], mask: np.ndarray) -> pd.DataFrame:
    selected = codes[mask]
    frame = pd.DataFrame({k: np.bincount(selected, weights=v[mask], minlength=len(labels)) for k, v in stats.items()})
    frame.insert(0, "segment", list(labels))
    return frame[frame["n"] > 0]


def evaluate_origin(panel: M5Panel, spec: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    cutoff = spec["cutoff"]
    history = panel.sales[:, :cutoff + 1]
    actual = panel.sales[:, cutoff + 1:cutoff + 1 + horizon]
    profile = series_profile(history)
    daily, totals, varo, inputs, varo_seconds = origin_forecasts(history, horizon)
    valid = profile["eligible"][:, None] & ~np.isnan(actual)
    week_valid = valid[:, :OPERATIONAL_HORIZON].all(axis=1)
    lower = varo["demand_forecast_lower"].to_numpy(dtype=np.float64)[:, None]
    upper = varo["demand_forecast_upper"].to_numpy(dtype=np.float64)[:, None]
    covered = valid & (actual >= lower - EXACT_TOLERANCE) & (actual <= upper + EXACT_TOLERANCE)

    n_series, origin = len(history), spec["origin"]
    series_parts, point_parts = [], []
    target_days = np.arange(spec["target_start"], spec["target_end"] + 1)
    weekday = pd.to_datetime(pd.Series(panel.dates[target_days])).dt.day_name().map(WEEKDAYS.index).to_numpy()
    events = panel.event_label[target_days]
    event_labels = sorted(set(panel.event_label))
    point_codes = {
        "horizon_day": (np.broadcast_to(np.arange(horizon), actual.shape), [str(h) for h in range(1, horizon + 1)]),
        "horizon_week": (np.broadcast_to(np.arange(horizon) // 7, actual.shape), [f"week_{w}" for w in range(1, horizon // 7 + 1)]),
        "weekday": (np.broadcast_to(weekday, actual.shape), list(WEEKDAYS)),
        "event_type": (np.broadcast_to(np.array([event_labels.index(e) for e in events]), actual.shape), event_labels),
        "snap": (np.nan_to_num(panel.snap[:, target_days], nan=2).astype(np.int64), ["snap_0", "snap_1", "snap_unknown"]),
        "price_status": (_price_status(panel.price, cutoff, horizon), list(PRICE_STATUS)),
    }
    first_week = np.zeros(actual.shape, dtype=bool)
    first_week[:, :OPERATIONAL_HORIZON] = True
    for method in METHOD_ORDER:
        forecast = daily[method]
        stats = elementwise_stats(forecast, actual, valid)
        interval = covered if method == "varo_production" else np.zeros(actual.shape, dtype=bool)
        for window, columns in (("daily_h1_7", slice(0, OPERATIONAL_HORIZON)), ("daily_h1_28", slice(0, horizon))):
            summed = {k: v[:, columns].sum(axis=1) for k, v in stats.items()}
            mase, rmsse = scaled_errors(summed["abs_err"], summed["sq_err"], summed["n"], profile["mae_scale"], profile["mse_scale"])
            series_parts.append(_series_rows(origin, method, window, summed, mase, rmsse, interval[:, columns].sum(axis=1)))
        week_actual = np.where(week_valid, np.nan_to_num(actual[:, :OPERATIONAL_HORIZON]).sum(axis=1), np.nan)
        week_stats = elementwise_stats(totals[method], week_actual, week_valid)
        nan = np.full(n_series, np.nan)
        series_parts.append(_series_rows(origin, method, "total_7d", week_stats, nan, nan, np.zeros(n_series)))
        point_stats = {k: v.astype(np.float64) for k, v in stats.items()}
        point_stats["in_interval"] = interval.astype(np.float64)
        for dimension, (codes, labels) in point_codes.items():
            windows = [("daily_h1_28", valid)] if dimension.startswith("horizon") else [
                ("daily_h1_7", valid & first_week), ("daily_h1_28", valid)]
            for window, mask in windows:
                part = _group_points(point_stats, codes, labels, mask)
                point_parts.append(part.assign(origin=origin, method=method, window=window, dimension=dimension))

    profile_frame = pd.DataFrame({"origin": origin, "series": np.arange(n_series), **{k: v for k, v in profile.items()},
                                  **{c: inputs[c].to_numpy() for c in inputs.columns},
                                  "demand_forecast_7d": varo["demand_forecast_7d"].to_numpy(dtype=np.float64),
                                  "demand_forecast_daily": varo["demand_forecast_daily"].to_numpy(dtype=np.float64),
                                  "demand_forecast_lower": lower[:, 0], "demand_forecast_upper": upper[:, 0],
                                  "varo_method": varo["demand_forecast_method"].map(VARO_METHOD_LABELS.index).to_numpy(),
                                  "varo_trend": varo["demand_trend"].map(VARO_TREND_LABELS.index).to_numpy(),
                                  "actual_7d": np.where(week_valid, np.nan_to_num(actual[:, :OPERATIONAL_HORIZON]).sum(axis=1), np.nan),
                                  "actual_horizon": np.where(valid.all(axis=1), np.nan_to_num(actual).sum(axis=1), np.nan)})
    series_stats = pd.DataFrame({k: np.concatenate([part[k] for part in series_parts]) for k in series_parts[0]})
    return {"series_stats": series_stats, "points": pd.concat(point_parts, ignore_index=True), "profile": profile_frame,
            "daily": daily, "actual": actual, "varo_seconds": varo_seconds,
            "eligible_series": int(profile["eligible"].sum()), "excluded_series": int((~profile["eligible"]).sum()),
            "missing_target_points": int(np.isnan(actual).sum())}


# ---------------------------------------------------------------- aggregation

ORIGIN_ATTRIBUTES = ("frequency_class", "volume_band", "sbc_class", "age_band", "varo_method", "varo_trend")


def attach_origin_attributes(stats: pd.DataFrame, profile: pd.DataFrame, n_series: int) -> None:
    """Copy each (origin, series) class onto its score rows; keys are int64 (int16 origin x 30,490 series overflows)."""
    key = (stats["origin"].to_numpy(dtype=np.int64) - 1) * n_series + stats["series"].to_numpy(dtype=np.int64)
    position = (profile["origin"].to_numpy(dtype=np.int64) - 1) * n_series + profile["series"].to_numpy(dtype=np.int64)
    lookup = np.full(int(position.max()) + 1, -1, dtype=np.int64)
    lookup[position] = np.arange(len(profile))
    rows = lookup[key]
    if (rows < 0).any():
        raise ValueError("score rows without a series profile at their origin")
    for column in ORIGIN_ATTRIBUTES:
        stats[column] = profile[column].to_numpy()[rows].astype(np.int8)


def scopes_for(origins: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    ids = [o["origin"] for o in origins]
    return {"holdout": ids[-1:], "validation": ids[-2:-1], "rolling_backtest": ids[:-2], "rolling_all": ids}


def _ordered(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    order = {"method": METHOD_ORDER, "window": WINDOW_ORDER}
    ranked = frame.assign(**{f"_{k}": frame[k].map({v: i for i, v in enumerate(order[k])}) for k in order if k in frame})
    sort = [f"_{k}" if k in order else k for k in keys]
    return ranked.sort_values(sort, kind="mergesort").drop(columns=[f"_{k}" for k in order if k in frame]).reset_index(drop=True)


def compare_to_varo(metrics: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Varo minus method for MAE/RMSE/WAPE (positive = Varo worse) and ranks among the six candidates."""
    keys = list(keys)
    varo = metrics[metrics["method"] == "varo_production"][[*keys, "mae", "rmse", "wape"]]
    out = metrics.merge(varo.rename(columns={m: f"varo_{m}" for m in ("mae", "rmse", "wape")}), on=keys, how="left")
    for metric in ("mae", "rmse", "wape"):
        out[f"varo_minus_method_{metric}"] = out[f"varo_{metric}"] - out[metric]
        candidate = out["method"].isin(CANDIDATES)
        out[f"rank_{metric}"] = out[metric].where(candidate).groupby([out[k] for k in keys]).rank(method="min")
    return out


def series_group_metrics(stats: pd.DataFrame, scopes: Mapping[str, Sequence[int]], dims: Sequence[str] = ()) -> pd.DataFrame:
    parts = []
    every = set(stats["origin"].unique())
    for scope, origins in scopes.items():
        subset = stats if set(origins) >= every else stats[stats["origin"].isin(origins)]
        grouped = subset.groupby(["window", "method", *dims], observed=True, sort=False)[list(SUM_COLUMNS)].sum().reset_index()
        parts.append(grouped[grouped["n"] > 0].assign(scope=scope, origins=len(origins)))
    frame = pd.concat(parts, ignore_index=True)
    frame["window"] = frame["window"].map(dict(enumerate(WINDOW_ORDER)))
    frame["method"] = frame["method"].map(dict(enumerate(METHOD_ORDER)))
    return finalize_metrics(frame)


def segment_table(stats: pd.DataFrame, scopes: Mapping[str, Sequence[int]], dimensions: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    parts = []
    for dimension, labels in dimensions.items():
        frame = series_group_metrics(stats, scopes, [dimension])
        frame["segment"] = frame[dimension].map(dict(enumerate(labels)))
        parts.append(frame.drop(columns=dimension).assign(dimension=dimension))
    table = compare_to_varo(pd.concat(parts, ignore_index=True), ["scope", "window", "dimension", "segment"])
    table["_segment"] = table.apply(lambda r: list(dimensions[r["dimension"]]).index(r["segment"]), axis=1)
    table["_dimension"] = table["dimension"].map(list(dimensions).index)
    table["_scope"] = table["scope"].map(list(scopes).index)
    table = _ordered(table, ["_scope", "window", "_dimension", "_segment", "method"]).drop(columns=["_segment", "_dimension", "_scope"])
    return table


def point_group_metrics(points: pd.DataFrame, scopes: Mapping[str, Sequence[int]], dimensions: Sequence[str]) -> pd.DataFrame:
    parts = []
    stats = [c for c in points.columns if c not in {"segment", "origin", "method", "window", "dimension"}]
    for scope, origins in scopes.items():
        subset = points[points["origin"].isin(origins) & points["dimension"].isin(dimensions)]
        grouped = subset.groupby(["window", "dimension", "segment", "method"], sort=False)[stats].sum().reset_index()
        parts.append(grouped.assign(scope=scope, origins=len(origins)))
    frame = finalize_metrics(pd.concat(parts, ignore_index=True))
    return compare_to_varo(frame, ["scope", "window", "dimension", "segment"])


# ---------------------------------------------------------------- outputs

COMMON_COLUMNS = ["method", "method_role", "n", "scored", "actual", "forecast", "mae", "rmse", "wape", "mean_error", "bias_units",
                  "bias_pct", "over_forecast_rate", "under_forecast_rate", "exact_rate", "over_units", "under_units"]
COMPARISON_COLUMNS = ["varo_minus_method_mae", "varo_minus_method_rmse", "varo_minus_method_wape", "rank_mae", "rank_rmse", "rank_wape"]


def _columns(frame: pd.DataFrame, leading: Sequence[str], extra: Sequence[str] = ()) -> pd.DataFrame:
    frame = frame.assign(method_role=frame["method"].map({m: s["role"] for m, s in METHODS.items()}))
    wanted = [c for c in [*leading, *COMMON_COLUMNS, *extra, *COMPARISON_COLUMNS] if c in frame.columns]
    return frame[wanted].rename(columns={"n": "points", "scored": "series_origins", "actual": "actual_units", "forecast": "forecast_units"})


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.round(6).to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    return value


def peak_memory_mb() -> float | None:
    """Peak working set (Windows) or max RSS (POSIX) of this process."""
    try:
        if sys.platform == "win32":
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / 2 ** 20
            return None
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # measurement must never break the evaluation
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def data_profile(panel: M5Panel) -> dict[str, Any]:
    sales, price = panel.sales, panel.price
    zero = sales == 0
    no_price = np.isnan(price)
    events = pd.Series(panel.event_label)
    snap_days = {state: int(np.nansum(panel.snap[panel.series["state_id"].to_numpy() == state][0]))
                 for state in sorted(panel.series["state_id"].unique())}
    return {
        "date_range": [panel.dates[0], panel.dates[-1]], "days": len(panel.dates), "series": len(panel.series),
        "product_count": int(panel.series["item_id"].nunique()), "store_count": int(panel.series["store_id"].nunique()),
        "state_count": int(panel.series["state_id"].nunique()), "category_count": int(panel.series["category"].nunique()),
        "department_count": int(panel.series["department"].nunique()), "canonical_rows": panel.canonical_rows,
        "sales_qty_sum": float(np.nansum(sales)), "sales_qty_max": float(np.nanmax(sales)),
        "sales_missing_cells": int(np.isnan(sales).sum()), "sales_zero_cells": int(zero.sum()), "sales_zero_share": float(zero.mean()),
        "zero_sales_without_weekly_price": int((zero & no_price).sum()), "zero_sales_with_weekly_price": int((zero & ~no_price).sum()),
        "positive_sales_without_weekly_price": int(((sales > 0) & no_price).sum()),
        "price_present_cells": int((~no_price).sum()), "price_missing_cells": int(no_price.sum()),
        "price_range": [float(np.nanmin(price)), float(np.nanmax(price))],
        "event_days": int((events != "none").sum()), "event_days_by_type": events[events != "none"].value_counts().sort_index().to_dict(),
        "snap_days_by_state": snap_days,
        "interpretation": {
            "zero": "An observed zero sale. Zeros on days without a weekly price are 'not on sale that week' (M5 guide); zeros with a price "
                    "are no sale while listed, which includes unobserved stock-outs (censored demand).",
            "missing": "A missing sales cell stays NaN: it is excluded from features and scores, never read as 0. The release has none.",
            "price": "A missing weekly price stays NaN and is never 0; price is used only for post-scoring diagnostics, never as a forecast input."},
    }


def _metric_block(frame: pd.DataFrame) -> dict[str, Any]:
    keep = ["points", "actual_units", "forecast_units", "mae", "rmse", "wape", "mean_error", "bias_units", "bias_pct",
            "over_forecast_rate", "under_forecast_rate", "mase_mean", "rmsse_mean", "rank_wape"]
    return {r["method"]: {k: r[k] for k in keep if k in frame.columns} for _, r in frame.iterrows()}


def stability(by_cutoff: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for window in WINDOW_ORDER:
        frame = by_cutoff[by_cutoff["window"] == window]
        varo = frame[frame["method"] == "varo_production"].set_index("origin")
        entry: dict[str, Any] = {}
        for method in CANDIDATES:
            if method == "varo_production":
                continue
            other = frame[frame["method"] == method].set_index("origin")
            delta = (varo["wape"] - other["wape"]).dropna()
            entry[method] = {"origins": int(len(delta)), "varo_better_wape": int((delta < 0).sum()), "varo_worse_wape": int((delta > 0).sum()),
                             "wape_delta_mean": float(delta.mean()), "wape_delta_min": float(delta.min()), "wape_delta_max": float(delta.max())}
        best = frame[frame["rank_wape"] == 1].groupby("origin")["method"].agg(lambda s: "|".join(s))
        entry["best_method_by_origin"] = best.to_dict()
        entry["varo_wape_by_origin"] = varo["wape"].to_dict()
        entry["varo_rank_wape_by_origin"] = varo["rank_wape"].to_dict()
        result[window] = entry
    return result


def series_level_comparison(stats: pd.DataFrame, origin: int) -> dict[str, Any]:
    """Share of series whose own MAE (days 1-7) is lower/equal/higher for Varo than for each baseline."""
    frame = stats[(stats["origin"] == origin) & (stats["window"] == WINDOW_ORDER.index("daily_h1_7")) & (stats["n"] > 0)]
    abs_err = frame.pivot(index="series", columns="method", values="abs_err")
    varo = abs_err[METHOD_ORDER.index("varo_production")]
    result = {}
    for method in [*BASELINES, "varo_branch_naive30", "varo_branch_sma7", "zero_reference"]:
        other = abs_err[METHOD_ORDER.index(method)]
        diff = varo - other
        result[method] = {"series": int(len(diff)), "varo_lower_mae": float((diff < -EXACT_TOLERANCE).mean()),
                          "tie": float((diff.abs() <= EXACT_TOLERANCE).mean()), "varo_higher_mae": float((diff > EXACT_TOLERANCE).mean())}
    return result


def _write_rows(writer: pq.ParquetWriter | None, path: Path, panel: M5Panel, spec: Mapping[str, Any], result: Mapping[str, Any]) -> pq.ParquetWriter:
    n, horizon = result["actual"].shape
    target = np.arange(spec["target_start"], spec["target_end"] + 1)
    columns = {
        "origin": pa.array(np.full(n * horizon, spec["origin"], dtype=np.int16)),
        "cutoff_date": pa.array(np.full(n * horizon, panel.dates[spec["cutoff"]], dtype=object), pa.string()).dictionary_encode(),
        "role": pa.array(np.full(n * horizon, spec["role"], dtype=object), pa.string()).dictionary_encode(),
        "store_id": pa.array(np.repeat(panel.series["store_id"].to_numpy(dtype=object), horizon), pa.string()).dictionary_encode(),
        "item_id": pa.array(np.repeat(panel.series["item_id"].to_numpy(dtype=object), horizon), pa.string()).dictionary_encode(),
        "h": pa.array(np.tile(np.arange(1, horizon + 1, dtype=np.int8), n)),
        "target_date": pa.array(np.tile(panel.dates[target].astype(object), n), pa.string()).dictionary_encode(),
        "actual": pa.array(result["actual"].ravel()),
        **{f"forecast_{m}": pa.array(result["daily"][m].ravel()) for m in METHOD_ORDER},
    }
    table = pa.table(columns)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def run_validation(data_root: Path | str = DATA_ROOT, output_dir: Path | str | None = None, *, horizon: int = HORIZON,
                   origin_count: int = ORIGIN_COUNT, origin_step: int = ORIGIN_STEP, write_rows: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    stages: dict[str, float] = {}

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        stages[name] = round(now - since, 3)
        return now

    folder = Path(data_root) / DATASETS["m5"]
    output = Path(output_dir) if output_dir else folder / "results"
    raw_before = file_fingerprints(folder / RAW_DIR, RAW_INPUTS)
    canonical = {name: _sha256(folder / "processed" / name) for name in CANONICAL_INPUTS}
    manifest_path = folder / "results/raw_manifest.json"
    manifest = {}
    if manifest_path.exists():
        for item in json.loads(manifest_path.read_text(encoding="utf-8")).get("files", []):
            manifest[Path(str(item.get("path", "")).replace("\\", "/")).name] = item.get("sha256")
    tick = lap("hash_inputs", started)
    panel = load_panel(folder)
    tick = lap("load_canonical_panel", tick)
    crosscheck = raw_crosscheck(folder, panel)
    if not crosscheck["cells_equal_including_missing"]:
        raise AssertionError(f"canonical sales differ from raw sales_train_evaluation.csv: {crosscheck}")
    tick = lap("raw_crosscheck", tick)

    origins = rolling_origins(len(panel.dates), horizon, origin_count, origin_step)
    scopes = scopes_for(origins)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / OUTPUT_FILES["rows"]
    writer = None
    series_stats, points, profiles, per_origin = [], [], [], []
    try:
        for spec in origins:
            origin_started = time.perf_counter()
            result = evaluate_origin(panel, spec, horizon)
            series_stats.append(result["series_stats"])
            points.append(result["points"])
            profiles.append(result["profile"])
            if write_rows:
                writer = _write_rows(writer, rows_path, panel, spec, result)
            per_origin.append({"origin": spec["origin"], "role": spec["role"], "eligible_series": result["eligible_series"],
                               "excluded_series_missing_history": result["excluded_series"],
                               "missing_target_points": result["missing_target_points"],
                               "production_forecast_seconds": round(result["varo_seconds"], 3),
                               "origin_seconds": round(time.perf_counter() - origin_started, 3)})
    finally:
        if writer is not None:
            writer.close()
    tick = lap("forecast_and_score", tick)

    stats = pd.concat(series_stats, ignore_index=True)
    profile = pd.concat(profiles, ignore_index=True)
    labels = {c: tuple(sorted(panel.series[c].unique())) for c in ("store_id", "state_id", "category", "department")}
    for column in labels:
        codes = pd.factorize(panel.series[column], sort=True)[0]
        stats[column] = codes[stats["series"].to_numpy()].astype(np.int16)
    attach_origin_attributes(stats, profile, len(panel.series))

    summary = compare_to_varo(series_group_metrics(stats, scopes), ["scope", "window"])
    medians = []
    for scope, ids in scopes.items():
        subset = stats[stats["origin"].isin(ids) & (stats["mase_n"] > 0)]
        med = subset.groupby(["window", "method"])[["mase_sum", "rmsse_sum"]].median().reset_index()
        medians.append(med.rename(columns={"mase_sum": "mase_median", "rmsse_sum": "rmsse_median"}).assign(scope=scope))
    median = pd.concat(medians, ignore_index=True)
    median["window"] = median["window"].map(dict(enumerate(WINDOW_ORDER)))
    median["method"] = median["method"].map(dict(enumerate(METHOD_ORDER)))
    summary = summary.merge(median, on=["scope", "window", "method"], how="left")
    summary["_scope"] = summary["scope"].map(list(scopes).index)
    summary = _ordered(summary, ["_scope", "window", "method"]).drop(columns="_scope")

    per_origin_scope = {f"origin_{o['origin']}": [o["origin"]] for o in origins}
    by_cutoff = compare_to_varo(series_group_metrics(stats, per_origin_scope), ["scope", "window"])
    meta = pd.DataFrame([{"scope": f"origin_{o['origin']}", "origin": o["origin"], "role": o["role"], "cutoff_day": f"d_{o['cutoff'] + 1}",
                          "cutoff_date": panel.dates[o["cutoff"]], "target_start": panel.dates[o["target_start"]],
                          "target_end": panel.dates[o["target_end"]]} for o in origins])
    by_cutoff = _ordered(by_cutoff.merge(meta, on="scope"), ["origin", "window", "method"])

    by_store = segment_table(stats, scopes, {"store_id": labels["store_id"], "state_id": labels["state_id"]})
    by_category = segment_table(stats, scopes, {"category": labels["category"], "department": labels["department"]})
    by_demand = segment_table(stats, scopes, {"frequency_class": FREQUENCY_CLASSES, "volume_band": VOLUME_BANDS, "sbc_class": SBC_CLASSES,
                                              "age_band": AGE_BANDS, "varo_method": VARO_METHOD_LABELS, "varo_trend": VARO_TREND_LABELS})
    overall_segment = summary.assign(dimension="overall", segment="all")
    bias_dims = segment_table(stats, scopes, {"frequency_class": FREQUENCY_CLASSES, "volume_band": VOLUME_BANDS,
                                              "category": labels["category"], "store_id": labels["store_id"], "varo_method": VARO_METHOD_LABELS})
    bias = pd.concat([overall_segment, bias_dims], ignore_index=True)
    bias["_scope"] = bias["scope"].map(list(scopes).index)
    bias["_dimension"] = bias["dimension"].map(["overall", "frequency_class", "volume_band", "category", "store_id", "varo_method"].index)
    bias = _ordered(bias, ["_scope", "window", "_dimension"]).drop(columns=["_scope", "_dimension"])
    point_frame = pd.concat(points, ignore_index=True)
    by_horizon = point_group_metrics(point_frame, scopes, ["horizon_day", "horizon_week"])
    by_horizon["_segment"] = by_horizon["segment"].map(lambda s: int(s) if s.isdigit() else 100 + int(s.split("_")[1]))
    by_horizon["_scope"] = by_horizon["scope"].map(list(scopes).index)
    by_horizon = _ordered(by_horizon, ["_scope", "dimension", "_segment", "method"]).drop(columns=["_segment", "_scope"])
    calendar_dims = ["weekday", "event_type", "snap", "price_status"]
    by_calendar = point_group_metrics(point_frame, scopes, calendar_dims)
    by_calendar["_scope"] = by_calendar["scope"].map(list(scopes).index)
    by_calendar["_dimension"] = by_calendar["dimension"].map(calendar_dims.index)
    by_calendar = _ordered(by_calendar, ["_scope", "window", "_dimension", "segment", "method"]).drop(columns=["_scope", "_dimension"])
    tick = lap("aggregate", tick)

    _write_csv(_columns(summary, ["scope", "origins", "window"], ["mase_mean", "mase_median", "rmsse_mean", "rmsse_median", "interval_coverage"]),
               output / OUTPUT_FILES["summary"])
    _write_csv(_columns(by_cutoff, ["origin", "role", "cutoff_day", "cutoff_date", "target_start", "target_end", "window"],
                        ["mase_mean", "rmsse_mean", "interval_coverage"]), output / OUTPUT_FILES["by_cutoff"])
    for name, frame in (("by_store", by_store), ("by_category", by_category), ("by_demand_type", by_demand)):
        _write_csv(_columns(frame, ["scope", "origins", "window", "dimension", "segment"], ["mase_mean", "rmsse_mean", "interval_coverage"]),
                   output / OUTPUT_FILES[name])
    bias_out = bias.assign(method_role=bias["method"].map({m: s["role"] for m, s in METHODS.items()}))
    bias_out = bias_out[["scope", "origins", "window", "dimension", "segment", "method", "method_role", *BIAS_COLUMNS]]
    _write_csv(bias_out.rename(columns={"n": "points", "actual": "actual_units", "forecast": "forecast_units"}), output / OUTPUT_FILES["bias"])
    _write_csv(_columns(by_horizon, ["scope", "origins", "window", "dimension", "segment"], ["interval_coverage"]), output / OUTPUT_FILES["by_horizon"])
    _write_csv(_columns(by_calendar, ["scope", "origins", "window", "dimension", "segment"], ["interval_coverage"]), output / OUTPUT_FILES["by_calendar"])
    profile_out = profile.copy()
    profile_out.insert(1, "store_id", panel.series["store_id"].to_numpy()[profile["series"]])
    profile_out.insert(2, "item_id", panel.series["item_id"].to_numpy()[profile["series"]])
    profile_out.insert(1, "cutoff_date", profile["origin"].map({o["origin"]: panel.dates[o["cutoff"]] for o in origins}))
    for column, names in (("frequency_class", FREQUENCY_CLASSES), ("volume_band", VOLUME_BANDS), ("sbc_class", SBC_CLASSES),
                          ("age_band", AGE_BANDS), ("varo_method", VARO_METHOD_LABELS), ("varo_trend", VARO_TREND_LABELS)):
        profile_out[column] = profile_out[column].map(dict(enumerate(names)))
    pq.write_table(pa.Table.from_pandas(profile_out.drop(columns="series"), preserve_index=False), output / OUTPUT_FILES["series_profile"],
                   compression="zstd")
    tick = lap("write_outputs", tick)

    raw_after = file_fingerprints(folder / RAW_DIR, RAW_INPUTS)
    if raw_after != raw_before:
        raise AssertionError("raw M5 files changed during the evaluation")
    module = load_legacy_module(PRODUCTION_MODULE)
    module_path = legacy_module_path(PRODUCTION_MODULE)
    holdout, validation = origins[-1], origins[-2]
    summary_out = _columns(summary, ["scope", "origins", "window"], ["mase_mean", "mase_median", "rmsse_mean", "rmsse_median", "interval_coverage"])
    by_cutoff_out = _columns(by_cutoff, ["origin", "role", "window"], ["mase_mean", "rmsse_mean"])
    headline = {scope: {window: _metric_block(summary_out[(summary_out["scope"] == scope) & (summary_out["window"] == window)])
                        for window in WINDOW_ORDER} for scope in scopes}
    evaluated_points = {w: int(summary_out[(summary_out["scope"] == "rolling_all") & (summary_out["window"] == w)
                                           & (summary_out["method"] == "varo_production")]["points"].iloc[0]) for w in WINDOW_ORDER}
    report = {
        "evaluation": {"name": "Walmart M5 external validation of the Varo production demand forecast", "version": EVALUATION_VERSION,
                       "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "command": "python -m services.m5_forecast_validation --data-root <root>"},
        "data_signature": {"raw_files": {name: {**info, "manifest_sha256_match": manifest.get(name) == info["sha256"] if manifest.get(name) else None}
                                         for name, info in raw_before.items()},
                           "canonical_files_sha256": canonical,
                           "sales_matrix_sha256": hashlib.sha256(np.ascontiguousarray(panel.sales).tobytes()).hexdigest(),
                           "sales_matrix_shape": list(panel.sales.shape), "raw_canonical_crosscheck": crosscheck,
                           "raw_unchanged_during_run": raw_after == raw_before},
        "data_profile": data_profile(panel),
        "date_range": [panel.dates[0], panel.dates[-1]],
        "train_range": [panel.dates[0], panel.dates[validation["cutoff"]]],
        "validation_range": [panel.dates[validation["target_start"]], panel.dates[validation["target_end"]]],
        "holdout_range": [panel.dates[holdout["target_start"]], panel.dates[holdout["target_end"]]],
        "holdout_history_range": [panel.dates[0], panel.dates[holdout["cutoff"]]],
        "split_policy": ("Chronological only. No parameter is estimated or tuned: every method is a fixed, pre-registered rule, so the validation "
                         "window is a pre-holdout check of the method ranking, not a tuning set. At the holdout cutoff the history includes the "
                         "validation days because they are observed before it (expanding window)."),
        "forecast_horizon": {"operational_days": OPERATIONAL_HORIZON, "extended_days": horizon, "rationale": [
            "Varo's production output is demand_forecast_7d ('향후 7일 예측 판매량'); inventory_transition, optimality_gap and sensitivity use it as the "
            "7-day need, so days 1-7 are the primary horizon.",
            "A 7-day window holds each weekday once, so the 7-day total is free of weekday-position bias.",
            "The M5 official horizon is 28 days and the holdout d_1914-d_1941 is exactly 28 days; days 8-28 hold Varo's daily rate flat, which "
            "measures how error grows when the 7-day rate is reused for longer windows."]},
        "cutoffs": [{**o, "cutoff_day": f"d_{o['cutoff'] + 1}", "cutoff_date": panel.dates[o["cutoff"]],
                     "target_range": [panel.dates[o["target_start"]], panel.dates[o["target_end"]]]} for o in origins],
        "per_origin_run": per_origin,
        "leakage_controls": [
            "Each forecast function receives only the history matrix sliced to columns <= cutoff; targets are sliced separately.",
            "Varo inputs (sales_7d, sales_30d, avg_daily_sales, demand_std), demand-type classes, volume bands, product age and MASE/RMSSE "
            "scales are computed from the same history slice.",
            "No normalisation or parameter is fitted; the only data-dependent quantities above are history-only.",
            "Price and calendar after the cutoff are never forecast inputs. Calendar events/SNAP (published in advance) and the target-day "
            "weekly price are used only to group errors after scoring (m5_forecast_by_calendar.csv).",
            "Verified by tests/test_m5_forecast_validation.py::test_no_future_leakage (overwriting every post-cutoff value leaves all "
            "forecasts, classes and scales unchanged)."],
        "algorithm_version": {"production_module": str(module_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if module_path.is_relative_to(PROJECT_ROOT) else str(module_path),
                              "production_module_sha256": _sha256(module_path), "git_commit": _git_commit(),
                              "production_constants": {name: getattr(module, name) for name in ("_W_RECENT", "_W_HISTORY", "_Z_95", "_TREND_UP_THR", "_TREND_DN_THR")},
                              "call_path": "prepare_legacy_data -> strip_dqn_columns -> load_legacy_module('demand_forecast_analyzer').analyze_demand_forecast",
                              "evaluation_version": EVALUATION_VERSION},
        "parameters": {"horizon": horizon, "operational_horizon": OPERATIONAL_HORIZON, "origin_count": origin_count, "origin_step": origin_step,
                       "recent_window": RECENT_WINDOW, "history_window": HISTORY_WINDOW, "moving_average_window": MOVING_AVERAGE_WINDOW,
                       "profile_window": PROFILE_WINDOW, "price_change_threshold": PRICE_CHANGE, "exact_tolerance": EXACT_TOLERANCE,
                       "demand_std": "sample standard deviation (ddof=1) of the last 30 daily sales",
                       "stock_inputs": "none: M5 has no inventory; stock_qty/lead_time are not supplied and the stock-dependent outputs are discarded"},
        "baseline_definitions": {m: s for m, s in METHODS.items()},
        "metric_definitions": {
            "mae": "sum|F-A| / points", "rmse": "sqrt(sum (F-A)^2 / points)", "wape": "sum|F-A| / sum A (undefined when sum A = 0)",
            "mean_error": "sum (F-A) / points", "bias_units": "sum F - sum A", "bias_pct": "(sum F - sum A) / sum A",
            "over/under_forecast_rate": "share of points with F-A > 1e-9 / < -1e-9",
            "over_units / under_units": "sum max(F-A,0) (excess-stock exposure) / sum max(A-F,0) (stock-out exposure)",
            "mase / rmsse": "per series: MAE / mean|y_t-y_(t-1)| and sqrt(MSE / mean (y_t-y_(t-1))^2) over the history from the first sale "
                            "(M5 RMSSE scale); averaged unweighted over series with a non-zero scale. Not the dollar-weighted WRMSSE.",
            "mape": f"not computed: {float((panel.sales == 0).mean()):.1%} of all cells are zero sales, where percentage error is undefined",
            "interval_coverage": "Varo only: share of daily actuals inside [demand_forecast_lower, demand_forecast_upper] (±1.65 demand_std)"},
        "demand_type_definitions": {"window": f"last {PROFILE_WINDOW} days up to the cutoff, from the first sale onward", "frequency_class": FREQUENCY_RULES,
                                    "volume_band": "mean daily sales over the same window", "sbc_class": "Syntetos-Boylan-Croston: ADI 1.32, CV^2 0.49 (undefined: < 2 selling days)",
                                    "age_band": "days from the first sale to the cutoff; <30d means Varo's 30-day average still contains pre-launch days"},
        "metrics": headline,
        "stability": stability(by_cutoff_out),
        "series_level_holdout_days_1_7": series_level_comparison(stats, holdout["origin"]),
        "row_count": {"canonical_rows": panel.canonical_rows, "series": len(panel.series), "origins": len(origins),
                      "series_origins_scored": int(summary_out[(summary_out["scope"] == "rolling_all") & (summary_out["window"] == "daily_h1_7")
                                                               & (summary_out["method"] == "varo_production")]["series_origins"].iloc[0]),
                      "points_scored_per_method": evaluated_points},
        "product_count": int(panel.series["item_id"].nunique()),
        "store_count": int(panel.series["store_id"].nunique()),
        "runtime": {"stages_seconds": stages, "total_seconds": round(time.perf_counter() - started, 3), "peak_memory_mb": peak_memory_mb()},
        "random_seed": None,
        "random_seed_note": "Not applicable: every step is deterministic (no sampling, no stochastic model, no random split).",
        "outputs": {k: v for k, v in OUTPUT_FILES.items() if k != "rows" or write_rows},
        "limitations": [
            "M5 sales are censored by unobserved stock-outs; this validates sales forecasting, not latent-demand recovery.",
            "M5 has no inventory, lead time or cost, so Varo's stock-dependent outputs (demand_risk_score, demand_stockout_days, "
            "demand_forecast_score) are not evaluated and no stock is generated.",
            "demand_std and the 30-day window follow the sample workbook schema; production uploads that define these fields differently "
            "would feed the same function different inputs.",
            "All 13 origins start on the same weekday (28-day spacing), so horizon day h always maps to the same weekday.",
            "No WRMSSE: MASE/RMSSE are unweighted series means, not the dollar-weighted hierarchical M5 score."],
    }
    (output / OUTPUT_FILES["json"]).write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return _clean(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-rows", action="store_true", help="skip the row-level parquet")
    args = parser.parse_args()
    report = run_validation(args.data_root, args.output_dir, write_rows=not args.no_rows)
    print(json.dumps({"metrics": report["metrics"]["holdout"], "runtime": report["runtime"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
