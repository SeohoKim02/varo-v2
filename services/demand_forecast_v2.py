"""Varo Demand Forecast v2 - a generic, explainable forecast candidate (not wired into production).

Production keeps ``demand_forecast_analyzer.analyze_demand_forecast`` (frozen below as
``FORECAST_V1_BASELINE``). This module only builds the v2 candidate that a separate,
pre-registered validation may recommend for promotion; nothing here is called by the
analysis pipeline.

Input of the Core is a day-aligned demand history matrix ``history`` (series x days,
last column = the forecast cutoff): observed sales per day, ``0`` = an observed zero
sale, ``NaN`` = a missing observation that is skipped, never read as 0. The Core uses
nothing else - no calendar, price, promotion or product identity - so every Varo
dataset with a daily sales history can run it. ``groups`` (e.g. location x category
codes) is optional and only pools weekday shapes.

Core forecast of series i on horizon day h (h = 1 .. H):

    F[i, h] = level[i] x trend[i] x weekday_index[i, h mod 7]

- level: one of LEVEL_METHODS (moving averages over active days, weekly EWMA,
  Croston / SBA / TSB for intermittent demand), chosen per demand type.
- trend: 1, or the damped recent/long ratio clip(MA7 / MA28, 1/2, 2) ** 0.5.
- weekday_index: 1 (flat), or a weekday profile normalised to mean 1, so the
  7-day total is always 7 x level x trend and never depends on the weekday shape.

The demand type (share of selling days over the last 364 days since the first sale,
Syntetos-Boylan ADI 1.32 boundary) selects the route; each route is one level method
and one weekday method. Optional exogenous adjustments (calendar events, binary day
flags such as SNAP or promotion, planned price) are a separate research variant and
never part of the Core.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

CORE_VERSION = "2.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- V1 baseline (frozen)

V1_MODULE = "services/legacy_adapters/_local_modules/demand_forecast_analyzer.py"
FORECAST_V1_BASELINE: dict[str, Any] = {
    "name": "FORECAST_V1_BASELINE",
    "role": "production demand forecast, preserved unchanged; every v2 candidate is compared against it",
    "call_path": ["services.analysis_pipeline.run_analysis_pipeline", "services.legacy_adapters.data_adapter.prepare_legacy_data",
                  "services.analysis_pipeline._run_inventory_analysis",
                  "services.legacy_adapters.loader.load_legacy_module('demand_forecast_analyzer').analyze_demand_forecast"],
    "module": V1_MODULE,
    "function": "analyze_demand_forecast",
    "source_sha256_lf": "62a795c89f745ef28bf4bca5a4201ebfb3364561ffab528ffeaf8fdbb0a7344b",
    "constants": {"_W_RECENT": 0.6, "_W_HISTORY": 0.4, "_Z_95": 1.65, "_TREND_UP_THR": 1.1, "_TREND_DN_THR": 0.9},
    "inputs": {"hist_daily": "avg_daily_sales, else sales_30d / 30, else state_source_sales_30d / 30, else 0",
               "recent_daily": "sales_7d / 7 clipped at 0 (NaN when the column is absent)"},
    "branches": {
        "WMA": "sales_7d column present and some hist_daily > 0: daily = 0.6 * recent_daily + 0.4 * hist_daily",
        "SMA": "sales_7d column present, no positive hist_daily: daily = recent_daily",
        "NAIVE": "no sales_7d column: daily = hist_daily"},
    "fallbacks": ["WMA branch, per row with recent_daily <= 0: NAIVE (daily = hist_daily, 7d = round(7 * hist_daily, 1))",
                  "demand_std missing or <= 0: 0.25 * hist_daily (interval only)",
                  "no stock_qty: state_source_stock, else 0; no lead_time_days: 3 (risk score only)"],
    "outputs": {"demand_forecast_7d": "round(7 * daily, 1)", "demand_forecast_daily": "demand_forecast_7d / 7 rounded to 2 decimals "
                "(NAIVE rows: hist_daily)", "shape": "flat - the same daily rate on every horizon day"},
    "downstream_consumers": {"demand_forecast_7d": ["services.inventory_transition_service", "services.optimality_gap_service",
                                                    "services.sensitivity_service", "services.vhs_score_engine (demand_fit_score)"]},
}


def _source_sha256_lf(path: Path) -> str:
    """SHA-256 with CRLF normalised to LF, so the fingerprint does not depend on the git checkout's line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def v1_baseline_fingerprint() -> dict[str, Any]:
    """Current fingerprint of the production forecast, comparable with FORECAST_V1_BASELINE."""
    from services.legacy_adapters.loader import legacy_module_path, load_legacy_module

    module = load_legacy_module("demand_forecast_analyzer")
    return {"source_sha256_lf": _source_sha256_lf(legacy_module_path("demand_forecast_analyzer")),
            "constants": {name: getattr(module, name) for name in FORECAST_V1_BASELINE["constants"]}}


def run_v1_baseline(inventory: pd.DataFrame) -> pd.DataFrame:
    """FORECAST_V1_BASELINE through its production call path (prepare_legacy_data -> strip_dqn_columns -> allowlisted module)."""
    from services.dqn_guard import strip_dqn_columns
    from services.legacy_adapters.data_adapter import prepare_legacy_data
    from services.legacy_adapters.loader import load_legacy_module

    prepared = prepare_legacy_data({"inventory": inventory.copy()})["inventory"]
    return load_legacy_module("demand_forecast_analyzer").analyze_demand_forecast(strip_dqn_columns(prepared))


# ---------------------------------------------------------------- declared constants

WEEK = 7
OPERATIONAL_HORIZON = 7        # demand_forecast_7d contract
PROFILE_WINDOW = 364           # 52 whole weeks: demand type, intermittent estimation window
HIGH_SHARE = 1 / 1.32          # ADI < 1.32: Syntetos-Boylan non-intermittent boundary
MEDIUM_SHARE = 0.5             # ADI < 2
INTERMITTENT_SHARE = 0.1       # ADI < 10
DEMAND_TYPES = ("high_frequency", "medium_frequency", "intermittent", "mostly_zero", "no_sales_history")
ROUTED_TYPES = DEMAND_TYPES[:4]  # no_sales_history always forecasts 0
TREND = {"short_window": 7, "long_window": 28, "damping_exponent": 0.5, "ratio_cap": 2.0}
FALLBACK_LEVEL = "ma28"        # Croston-family route on a series with negative values (returns): not applicable
WEEKDAY_SHRINK_UNITS = 175     # Buhlmann credibility K = 7 / tau^2 with a prior weekday-index spread tau = 0.2

CORE_CONSTANTS: dict[str, Any] = {
    "profile_window_days": PROFILE_WINDOW,
    "demand_type_rules": {"high_frequency": "selling-day share p >= 1/1.32 (ADI < 1.32)", "medium_frequency": "0.5 <= p < 1/1.32",
                          "intermittent": "0.1 <= p < 0.5", "mostly_zero": "p < 0.1 (incl. no sale in the window)",
                          "no_sales_history": "never sold up to the cutoff -> forecast 0",
                          "window": "last 364 days, from the first sale onward; observed days only"},
    "trend": TREND,
    "fallback_level": FALLBACK_LEVEL,
    "weekday_shrink_units": WEEKDAY_SHRINK_UNITS,
    "active_days": "every level uses only observed days on or after the first sale (cold start: mean since the first sale)",
    "zero_vs_missing": "0 is an observed zero demand; NaN is skipped (no mean contribution, no Croston/TSB update)",
    "aggregate_7d": "7 x level x trend (= sum of the daily vector); compatible demand_forecast_7d rounds it to 0.1 like V1",
}

# Declared order = tie-break preference (simpler first). Nothing outside these two tables is ever searched.
LEVEL_METHODS: dict[str, dict[str, Any]] = {
    "ma28": {"kind": "window_mean", "window": 28, "trend": False},
    "ma14": {"kind": "window_mean", "window": 14, "trend": False},
    "ma7": {"kind": "window_mean", "window": 7, "trend": False},
    "ma91": {"kind": "window_mean", "window": 91, "trend": False},
    "ewma_w0.2": {"kind": "weekly_ewma", "alpha": 0.2, "weeks": 26, "trend": False},
    "ewma_w0.4": {"kind": "weekly_ewma", "alpha": 0.4, "weeks": 26, "trend": False},
    "sba_0.1": {"kind": "croston", "variant": "sba", "alpha": 0.1, "window": PROFILE_WINDOW, "trend": False},
    "sba_0.2": {"kind": "croston", "variant": "sba", "alpha": 0.2, "window": PROFILE_WINDOW, "trend": False},
    "croston_0.1": {"kind": "croston", "variant": "croston", "alpha": 0.1, "window": PROFILE_WINDOW, "trend": False},
    "tsb_0.1_0.05": {"kind": "tsb", "alpha": 0.1, "beta": 0.05, "window": PROFILE_WINDOW, "trend": False},
    "tsb_0.2_0.1": {"kind": "tsb", "alpha": 0.2, "beta": 0.1, "window": PROFILE_WINDOW, "trend": False},
    "ma28_trend": {"kind": "window_mean", "window": 28, "trend": True},
    "ma14_trend": {"kind": "window_mean", "window": 14, "trend": True},
    "ma7_trend": {"kind": "window_mean", "window": 7, "trend": True},
    "ma91_trend": {"kind": "window_mean", "window": 91, "trend": True},
    "ewma_w0.2_trend": {"kind": "weekly_ewma", "alpha": 0.2, "weeks": 26, "trend": True},
    "ewma_w0.4_trend": {"kind": "weekly_ewma", "alpha": 0.4, "weeks": 26, "trend": True},
}
INTERMITTENT_KINDS = ("croston", "tsb")
WEEKDAY_METHODS: dict[str, dict[str, Any]] = {
    "flat": {"source": "none", "weeks": 0, "min_age_days": 0},
    "series_4w": {"source": "series", "weeks": 4, "min_age_days": 28},
    "series_8w": {"source": "series", "weeks": 8, "min_age_days": 56},
    "shrunk_8w": {"source": "credibility", "weeks": 8, "min_age_days": 56, "shrink_units": WEEKDAY_SHRINK_UNITS},
    "pooled_8w": {"source": "pooled", "weeks": 8, "min_age_days": 28},
}

# ---------------------------------------------------------------- demand profile


@dataclass(frozen=True)
class DemandProfile:
    demand_type: np.ndarray      # int8 code into DEMAND_TYPES
    first_sale: np.ndarray       # column of the first positive sale (history.shape[1] when never sold)
    age: np.ndarray              # days from the first sale to the cutoff, inclusive (0 = never sold)
    nonzero_ratio: np.ndarray    # selling-day share over the profile window
    adi: np.ndarray              # average inter-demand interval = active days / selling days (inf without a sale)
    cv: np.ndarray               # coefficient of variation of daily demand over active days
    cv2_nonzero: np.ndarray      # squared CV of the non-zero demand sizes (Syntetos-Boylan)
    recent_mean: np.ndarray      # mean of the last 7 active days
    long_mean: np.ndarray        # mean of the last 28 active days
    trend_ratio: np.ndarray      # recent_mean / long_mean (NaN when long_mean = 0)
    weekday_strength: np.ndarray  # std of the series' own 8-week weekday index (NaN when not computable)
    history_length: np.ndarray   # observed days from the first sale to the cutoff
    has_negative: np.ndarray     # any negative value (returns) in the profile window


def _as_history(history: Any) -> np.ndarray:
    array = np.asarray(history, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("history must be a (series x days) matrix with at least one day")
    return array


def recent_block(history: np.ndarray, first_sale: np.ndarray, width: int = PROFILE_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Last ``width`` columns (a view) and each row's first active column inside it (<= 0: active throughout)."""
    t = history.shape[1]
    w = min(width, t)
    return history[:, t - w:], np.asarray(first_sale) - (t - w)


def _active(block: np.ndarray, start: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Tail of ``width`` columns and its mask of observed days on or after the first sale."""
    w = min(width, block.shape[1])
    tail = block[:, block.shape[1] - w:]
    cols = np.arange(block.shape[1] - w, block.shape[1])
    return tail, (cols[None, :] >= np.asarray(start)[:, None]) & ~np.isnan(tail)


def window_mean(block: np.ndarray, start: np.ndarray, window: int) -> np.ndarray:
    """Mean over the last ``window`` observed active days; younger series use every day since their first sale."""
    tail, use = _active(block, start, window)
    total, count = np.where(use, tail, 0.0).sum(axis=1), use.sum(axis=1)
    return np.divide(total, count, out=np.zeros(len(block)), where=count > 0)


def weekly_ewma(block: np.ndarray, start: np.ndarray, alpha: float, weeks: int) -> np.ndarray:
    """Exponentially weighted mean of whole-week totals ending at the cutoff (weights alpha(1-alpha)^age), per day.

    Only fully observed, fully active weeks count, so each weight covers every weekday once. A series younger than
    one full week falls back to its mean since the first sale.
    """
    n, width = block.shape
    k = min(weeks, width // WEEK)
    fallback = window_mean(block, start, WEEK)
    if k == 0:
        return fallback
    tail = block[:, width - WEEK * k:].reshape(n, k, WEEK)
    week_start = width - WEEK * k + WEEK * np.arange(k)
    full = (week_start[None, :] >= np.asarray(start)[:, None]) & ~np.isnan(tail).any(axis=2)
    totals = np.where(full, np.nan_to_num(tail).sum(axis=2), 0.0)
    weights = np.where(full, alpha * (1.0 - alpha) ** ((k - 1) - np.arange(k))[None, :], 0.0)
    wsum = weights.sum(axis=1)
    level = np.divide((weights * totals).sum(axis=1), WEEK * wsum, out=np.zeros(n), where=wsum > 0)
    return np.where(wsum > 0, level, fallback)


def trend_multiplier(block: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Damped recent/long ratio: clip(MA7 / MA28, 1/cap, cap) ** damping; 1 when MA28 = 0."""
    short = window_mean(block, start, TREND["short_window"])
    long = window_mean(block, start, TREND["long_window"])
    ratio = np.divide(short, long, out=np.ones(len(block)), where=long > 0)
    return np.clip(ratio, 1.0 / TREND["ratio_cap"], TREND["ratio_cap"]) ** TREND["damping_exponent"]


def _intermittent_start(block: np.ndarray, start: np.ndarray, window: int):
    tail, active = _active(block, start, window)
    demand = active & (tail > 0)
    n_demand, n_active = demand.sum(axis=1), active.sum(axis=1)
    size0 = np.divide(np.where(demand, tail, 0.0).sum(axis=1), n_demand, out=np.zeros(len(block)), where=n_demand > 0)
    return tail, active, demand, n_demand, n_active, size0


def croston(block: np.ndarray, start: np.ndarray, alpha: float, variant: str = "croston", window: int = PROFILE_WINDOW) -> np.ndarray:
    """Croston (1972) / SBA (Syntetos-Boylan 2005) demand rate per day over the estimation window.

    Initialisation (mean method): size z0 = mean positive demand, interval p0 = active days / demand days in the
    window. Then, per observed active day: q += 1; on a positive demand z += alpha (y - z), p += alpha (q - p), q = 0.
    The first demand inside the window updates only z (its interval began before the window or before launch).
    Rate = z / p; SBA multiplies by (1 - alpha / 2). Zero days are demand-free periods; NaN days are skipped.
    """
    if variant not in ("croston", "sba"):
        raise ValueError(f"unknown Croston variant: {variant}")
    tail, active, demand, n_demand, n_active, size0 = _intermittent_start(block, start, window)
    z = size0.copy()
    p = np.divide(n_active, n_demand, out=np.ones(len(block)), where=n_demand > 0).astype(np.float64)
    q = np.zeros(len(block))
    seen = np.zeros(len(block), dtype=bool)
    for j in range(tail.shape[1]):
        d = demand[:, j]
        q += active[:, j]
        z = np.where(d, z + alpha * (tail[:, j] - z), z)
        p = np.where(d & seen, p + alpha * (q - p), p)
        q = np.where(d, 0.0, q)
        seen |= d
    rate = np.divide(z, p, out=np.zeros(len(block)), where=(n_demand > 0) & (p > 0))
    return rate * (1.0 - alpha / 2.0) if variant == "sba" else rate


def tsb(block: np.ndarray, start: np.ndarray, alpha: float, beta: float, window: int = PROFILE_WINDOW) -> np.ndarray:
    """TSB (Teunter-Syntetos-Babai 2011): rate = demand probability x demand size.

    Initialisation: probability = demand days / active days, size = mean positive demand in the window. Per observed
    active day the probability moves toward 1 (demand) or 0 (zero demand) by beta; the size moves toward y by alpha on
    demand days only. Unlike Croston it decays after a long run of zeros (obsolescence).
    """
    tail, active, demand, n_demand, n_active, size0 = _intermittent_start(block, start, window)
    z = size0.copy()
    prob = np.divide(n_demand, n_active, out=np.zeros(len(block)), where=n_active > 0).astype(np.float64)
    for j in range(tail.shape[1]):
        a, d = active[:, j], demand[:, j]
        prob = np.where(a, prob + beta * (d - prob), prob)
        z = np.where(d, z + alpha * (tail[:, j] - z), z)
    return np.where(n_demand > 0, prob * z, 0.0)


def level_forecast(block: np.ndarray, start: np.ndarray, method: str, has_negative: np.ndarray | None = None) -> np.ndarray:
    """Daily level x trend of one LEVEL_METHODS entry; Croston-family rows with negative values use FALLBACK_LEVEL."""
    spec = LEVEL_METHODS[method]
    kind = spec["kind"]
    if kind == "window_mean":
        level = window_mean(block, start, spec["window"])
    elif kind == "weekly_ewma":
        level = weekly_ewma(block, start, spec["alpha"], spec["weeks"])
    elif kind == "croston":
        level = croston(block, start, spec["alpha"], spec["variant"], spec["window"])
    elif kind == "tsb":
        level = tsb(block, start, spec["alpha"], spec["beta"], spec["window"])
    else:
        raise ValueError(f"unknown level kind: {kind}")
    if kind in INTERMITTENT_KINDS and has_negative is not None and np.any(has_negative):
        level = np.where(has_negative, level_forecast(block, start, FALLBACK_LEVEL), level)
    if spec["trend"]:
        level = level * trend_multiplier(block, start)
    return level


def weekday_sums(block: np.ndarray, start: np.ndarray, weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row: units and observed active days at each weekday position over the last ``weeks`` whole weeks.

    Position p = (day - cutoff) mod 7, so the cutoff is position 0 and horizon day h has position h mod 7; no
    calendar is needed because the matrix is day-aligned.
    """
    days = WEEK * weeks
    tail, use = _active(block, start, days)
    position = (np.arange(tail.shape[1]) - (tail.shape[1] - 1)) % WEEK
    values = np.where(use, tail, 0.0)
    units = np.stack([values[:, position == p].sum(axis=1) for p in range(WEEK)], axis=1)
    count = np.stack([use[:, position == p].sum(axis=1) for p in range(WEEK)], axis=1).astype(np.float64)
    return units, count


def _index_from_sums(units: np.ndarray, count: np.ndarray) -> np.ndarray:
    mean = np.divide(units, count, out=np.full(units.shape, np.nan), where=count > 0)
    overall = np.divide(units.sum(axis=-1), count.sum(axis=-1), out=np.zeros(units.shape[:-1]), where=count.sum(axis=-1) > 0)
    index = np.divide(mean, overall[..., None], out=np.ones(units.shape), where=(overall[..., None] > 0) & (count > 0))
    return index


def _normalise(index: np.ndarray) -> np.ndarray:
    index = np.clip(index, 0.0, None)
    mean = index.mean(axis=1, keepdims=True)
    return np.divide(index, mean, out=np.ones(index.shape), where=mean > 0)


def weekday_index(block: np.ndarray, start: np.ndarray, method: str, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(series x 7) multiplicative weekday index with mean 1, and whether it was applied.

    series: the series' own profile; pooled: the profile of its group (all series' units summed);
    credibility: Z x own + (1 - Z) x pooled with Z = own units / (own units + K). Applied only when the series has
    sold for at least ``min_age_days`` (cold start stays flat) and, for own profiles, sold in the window.
    """
    spec = WEEKDAY_METHODS[method]
    n, width = block.shape
    flat = np.ones((n, WEEK))
    if spec["source"] == "none" or width < WEEK * spec["weeks"]:
        return flat, np.zeros(n, dtype=bool)
    units, count = weekday_sums(block, start, spec["weeks"])
    own_units = units.sum(axis=1)
    age = width - np.asarray(start)
    old_enough = age >= spec["min_age_days"]
    if spec["source"] == "series":
        index, applied = _normalise(_index_from_sums(units, count)), old_enough & (own_units > 0)
    else:
        codes = np.zeros(n, dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
        group_units = np.zeros((codes.max() + 1, WEEK))
        group_count = np.zeros((codes.max() + 1, WEEK))
        np.add.at(group_units, codes, units)
        np.add.at(group_count, codes, count)
        pooled = _normalise(_index_from_sums(group_units, group_count))[codes]
        pooled_ok = (group_units.sum(axis=1) > 0)[codes]
        if spec["source"] == "pooled":
            index, applied = pooled, old_enough & pooled_ok
        else:   # both profiles have mean 1, so the credibility mix has mean 1 too
            weight = (own_units / (own_units + spec["shrink_units"]))[:, None]
            index = weight * _normalise(_index_from_sums(units, count)) + (1.0 - weight) * pooled
            applied = old_enough & pooled_ok
    index = np.where(applied[:, None], _normalise(index), 1.0)
    return index, applied


def demand_profile(history: Any) -> DemandProfile:
    """History-only series features and the demand type used for routing."""
    history = _as_history(history)
    n, t = history.shape
    sold = history > 0                                   # NaN compares False: missing is never a sale
    ever = sold.any(axis=1)
    first = np.where(ever, sold.argmax(axis=1), t)
    block, start = recent_block(history, first)
    tail, active = _active(block, start, PROFILE_WINDOW)
    selling = active & (tail > 0)
    n_active, n_selling = active.sum(axis=1), selling.sum(axis=1)
    share = np.divide(n_selling, n_active, out=np.zeros(n), where=n_active > 0)
    values = np.where(active, tail, 0.0)
    mean = np.divide(values.sum(axis=1), n_active, out=np.zeros(n), where=n_active > 0)
    var = np.divide(np.where(active, (tail - mean[:, None]) ** 2, 0.0).sum(axis=1), n_active, out=np.zeros(n), where=n_active > 0)
    cv = np.divide(np.sqrt(var), mean, out=np.full(n, np.nan), where=mean > 0)
    s1 = np.where(selling, tail, 0.0).sum(axis=1)
    s2 = np.where(selling, tail * tail, 0.0).sum(axis=1)
    size = np.divide(s1, n_selling, out=np.zeros(n), where=n_selling > 0)
    cv2 = np.divide(np.divide(s2, n_selling, out=np.zeros(n), where=n_selling > 0) - size ** 2, size ** 2,
                    out=np.full(n, np.nan), where=size > 0)
    recent, long = window_mean(block, start, 7), window_mean(block, start, 28)
    own, applied = weekday_index(block, start, "series_8w")
    strength = np.where(applied, own.std(axis=1), np.nan)
    observed_since_first = np.where(ever, (~np.isnan(history) & (np.arange(t)[None, :] >= first[:, None])).sum(axis=1), 0)
    demand_type = np.select([~ever, share >= HIGH_SHARE, share >= MEDIUM_SHARE, share >= INTERMITTENT_SHARE], [4, 0, 1, 2],
                            default=3).astype(np.int8)
    return DemandProfile(
        demand_type=demand_type, first_sale=first, age=np.where(ever, t - first, 0), nonzero_ratio=share,
        adi=np.divide(n_active, n_selling, out=np.full(n, np.inf), where=n_selling > 0), cv=cv, cv2_nonzero=cv2,
        recent_mean=recent, long_mean=long, trend_ratio=np.divide(recent, long, out=np.full(n, np.nan), where=long > 0),
        weekday_strength=strength, history_length=observed_since_first, has_negative=(block < 0).any(axis=1))


# ---------------------------------------------------------------- configuration


def make_config(routes: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """A complete, self-describing v2 configuration: routes plus the full spec of every method they use."""
    missing = [t for t in ROUTED_TYPES if t not in routes]
    if missing:
        raise ValueError(f"routes missing for demand types: {missing}")
    chosen = {t: {"level": str(routes[t]["level"]), "weekday": str(routes[t]["weekday"])} for t in ROUTED_TYPES}
    for route in chosen.values():
        if route["level"] not in LEVEL_METHODS or route["weekday"] not in WEEKDAY_METHODS:
            raise ValueError(f"unknown method in route {route}")
    return {"core_version": CORE_VERSION, "routes": chosen,
            "level_methods": {m: LEVEL_METHODS[m] for m in sorted({r["level"] for r in chosen.values()} | {FALLBACK_LEVEL})},
            "weekday_methods": {m: WEEKDAY_METHODS[m] for m in sorted({r["weekday"] for r in chosen.values()})},
            "constants": CORE_CONSTANTS}


def validate_config(config: Mapping[str, Any]) -> None:
    """Refuse a configuration whose method specs or constants differ from this module (a tampered or stale config)."""
    rebuilt = make_config(config["routes"])
    if json.loads(json.dumps(rebuilt)) != json.loads(json.dumps(config)):
        raise ValueError("configuration does not match the declared methods/constants of demand_forecast_v2")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_signature(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- forecast


@dataclass(frozen=True)
class ForecastV2Result:
    daily: np.ndarray            # (series x horizon) daily forecast vector
    aggregate_7d: np.ndarray     # 7 x level x trend = sum of daily[:, :7]
    level: np.ndarray            # daily level x trend before the weekday index
    weekday_index: np.ndarray    # (series x 7) indexed by position h mod 7
    weekday_applied: np.ndarray
    demand_type: np.ndarray
    level_method: np.ndarray     # method actually used per series (object)
    weekday_method: np.ndarray
    negative_fallback: np.ndarray
    profile: DemandProfile


def forecast_v2(history: Any, config: Mapping[str, Any], horizon: int = OPERATIONAL_HORIZON,
                groups: np.ndarray | None = None) -> ForecastV2Result:
    """Varo v2 Core: route every series by its demand type and build its daily forecast vector."""
    if horizon < OPERATIONAL_HORIZON:
        raise ValueError("horizon must cover the 7-day operational horizon")
    validate_config(config)
    history = _as_history(history)
    n = history.shape[0]
    profile = demand_profile(history)
    block, start = recent_block(history, profile.first_sale)
    level = np.zeros(n)
    index = np.ones((n, WEEK))
    applied = np.zeros(n, dtype=bool)
    level_method = np.full(n, "zero_no_sales_history", dtype=object)
    weekday_method = np.full(n, "flat", dtype=object)
    negative = np.zeros(n, dtype=bool)
    weekday_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, demand_type in enumerate(ROUTED_TYPES):
        rows = np.flatnonzero(profile.demand_type == code)
        if rows.size == 0:
            continue
        route = config["routes"][demand_type]
        sub_block, sub_start, sub_negative = block[rows], start[rows], profile.has_negative[rows]
        level[rows] = level_forecast(sub_block, sub_start, route["level"], sub_negative)
        uses_fallback = sub_negative & (LEVEL_METHODS[route["level"]]["kind"] in INTERMITTENT_KINDS)
        negative[rows] = uses_fallback
        level_method[rows] = np.where(uses_fallback, FALLBACK_LEVEL, route["level"])
        if route["weekday"] not in weekday_cache:
            weekday_cache[route["weekday"]] = weekday_index(block, start, route["weekday"], groups)
        wd_index, wd_applied = weekday_cache[route["weekday"]]
        index[rows], applied[rows] = wd_index[rows], wd_applied[rows]
        weekday_method[rows] = route["weekday"]
    position = np.arange(1, horizon + 1) % WEEK
    daily = level[:, None] * index[:, position]
    return ForecastV2Result(daily=daily, aggregate_7d=OPERATIONAL_HORIZON * level, level=level, weekday_index=index,
                            weekday_applied=applied, demand_type=profile.demand_type, level_method=level_method,
                            weekday_method=weekday_method, negative_fallback=negative, profile=profile)


def compatible_output(result: ForecastV2Result, index: Any = None, version: str = "varo_v2_core") -> pd.DataFrame:
    """V1-compatible columns plus the daily vector.

    demand_forecast_7d    : round(7-day aggregate, 1), the scalar every V1 consumer reads
    demand_forecast_daily : demand_forecast_7d / 7 rounded to 2 decimals (flat-equivalent daily rate)
    demand_forecast_d1..7 : the daily vector (unrounded); their sum is the 7-day aggregate before rounding
    """
    if result.daily.shape[1] < OPERATIONAL_HORIZON:
        raise ValueError("the daily vector must cover 7 days")
    out = pd.DataFrame({"demand_forecast_7d": np.round(result.aggregate_7d, 1)}, index=index)
    out["demand_forecast_daily"] = (out["demand_forecast_7d"] / OPERATIONAL_HORIZON).round(2)
    for h in range(OPERATIONAL_HORIZON):
        out[f"demand_forecast_d{h + 1}"] = result.daily[:, h]
    out["demand_type"] = np.array(DEMAND_TYPES, dtype=object)[result.demand_type]
    out["demand_forecast_route"] = [f"{lvl}+{wd}" for lvl, wd in zip(result.level_method, result.weekday_method)]
    out["demand_forecast_version"] = version
    return out


# ---------------------------------------------------------------- optional exogenous variant (research only)

EXOGENOUS_CONSTANTS: dict[str, Any] = {
    "event": {"baseline": "mean of the same weekday in the 4 weeks before the event day (group totals)",
              "statistic": "median ratio over past occurrences", "min_occurrences": 2, "ratio_clip": [0.0, 3.0]},
    "flag": {"window_days": PROFILE_WINDOW, "statistic": "group mean on flagged days / on unflagged days",
             "ratio_clip": [0.5, 2.0], "normalisation": "divided by the mean flag multiplier over the last 28 days (the level window)"},
    "price": {"reference": "mean price over the last 28 days", "elasticity": "pooled within-series OLS of log(1 + weekly units) on "
              "log(weekly price), last 52 whole weeks, per group", "elasticity_clip": [-3.0, 0.0], "factor_clip": [0.5, 2.0],
              "missing_target_price": "0 when the dataset declares a missing price = not listed, else 1"},
    "groups": "location x category codes supplied by the dataset adapter",
}


@dataclass(frozen=True)
class ExogenousInputs:
    events: np.ndarray | None = None          # (days x k) int event ids per day, -1 = none; days = history days + horizon
    flags: Mapping[str, np.ndarray] | None = None  # name -> (series x days) 0/1 day flags known in advance (SNAP, promotion)
    price: np.ndarray | None = None           # (series x days) planned/list price, NaN = no price
    missing_price_means_not_listed: bool = False


def _group_totals(history: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    totals = np.zeros((n_groups, history.shape[1]))
    for g in range(n_groups):
        rows = np.flatnonzero(codes == g)
        if rows.size:
            totals[g] = np.nansum(history[rows], axis=0)
    return totals


def exogenous_factors(history: Any, horizon: int, inputs: ExogenousInputs, groups: np.ndarray | None = None,
                      first_sale: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Multiplicative (series x horizon) event, flag and price factors estimated from history <= cutoff only.

    Target-day covariates (event calendar, flags, planned price) are read for the horizon because they are known in
    advance; target-day sales never are.
    """
    history = _as_history(history)
    n, t = history.shape
    days = [len(inputs.events)] if inputs.events is not None else []
    days += [np.asarray(m).shape[1] for m in [inputs.price, *(inputs.flags or {}).values()] if m is not None]
    if any(d < t + horizon for d in days):
        raise ValueError("exogenous inputs must cover every history day and the whole horizon")
    codes = np.zeros(n, dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
    n_groups = int(codes.max()) + 1
    totals = _group_totals(history, codes, n_groups)
    factor = {"event": np.ones((n, horizon)), "flag": np.ones((n, horizon)), "price": np.ones((n, horizon))}
    spec = EXOGENOUS_CONSTANTS

    if inputs.events is not None:
        events = np.asarray(inputs.events)
        future = events[t:t + horizon]
        ratios: dict[int, list[np.ndarray]] = {int(e): [] for e in np.unique(future) if e >= 0}
        for day in np.flatnonzero((events[:t] >= 0).any(axis=1)):
            if day < 4 * WEEK:
                continue
            base = totals[:, [day - WEEK * k for k in range(1, 5)]].mean(axis=1)
            ratio = np.divide(totals[:, day], base, out=np.full(n_groups, np.nan), where=base > 0)
            for e in events[day]:
                if int(e) in ratios:
                    ratios[int(e)].append(ratio)
        lo, hi = spec["event"]["ratio_clip"]
        effect = {}
        for e, found in ratios.items():
            effect[e] = np.ones(n_groups)
            for g in range(n_groups):
                values = np.array([r[g] for r in found if not np.isnan(r[g])])
                if len(values) >= spec["event"]["min_occurrences"]:
                    effect[e][g] = float(np.clip(np.median(values), lo, hi))
        for h in range(horizon):
            for e in future[h]:
                if int(e) in effect:
                    factor["event"][:, h] *= effect[int(e)][codes]

    if inputs.flags:
        lo, hi = spec["flag"]["ratio_clip"]
        window = min(spec["flag"]["window_days"], t)
        for flag in inputs.flags.values():
            flag = np.asarray(flag, dtype=np.float64)
            ratio = np.ones(n_groups)
            for g in range(n_groups):
                rows = np.flatnonzero(codes == g)
                if rows.size == 0:
                    continue
                marks = np.fmax.reduce(flag[rows, t - window:t], axis=0)   # the group's day flag; NaN only if unknown for all
                on, off = marks == 1, marks == 0
                if on.any() and off.any():
                    off_mean = totals[g, t - window:t][off].mean()
                    if off_mean > 0:
                        ratio[g] = np.clip(totals[g, t - window:t][on].mean() / off_mean, lo, hi)
            series_ratio = ratio[codes][:, None]
            multiplier = np.where(flag == 1, series_ratio, 1.0)
            reference = multiplier[:, max(0, t - 28):t].mean(axis=1, keepdims=True)
            factor["flag"] *= np.divide(multiplier[:, t:t + horizon], reference, out=np.ones((n, horizon)), where=reference > 0)

    if inputs.price is not None:
        price = np.asarray(inputs.price, dtype=np.float64)
        first = demand_profile(history).first_sale if first_sale is None else np.asarray(first_sale)
        recent = price[:, max(0, t - 28):t]
        have = ~np.isnan(recent)
        reference = np.divide(np.where(have, recent, 0.0).sum(axis=1), have.sum(axis=1), out=np.full(n, np.nan),
                              where=have.sum(axis=1) > 0)
        weeks = min(52, t // WEEK)
        elasticity = np.zeros(n_groups)
        if weeks >= 2:
            span = slice(t - WEEK * weeks, t)
            units = history[:, span].reshape(n, weeks, WEEK)
            week_price = price[:, span].reshape(n, weeks, WEEK)
            week_start = t - WEEK * weeks + WEEK * np.arange(weeks)
            ok = (~np.isnan(week_price).any(axis=2) & ~np.isnan(units).any(axis=2)
                  & (week_start[None, :] >= first[:, None]) & (np.nan_to_num(week_price).min(axis=2) > 0))
            x = np.where(ok, np.log(np.where(ok, np.nan_to_num(week_price).mean(axis=2), 1.0)), 0.0)
            y = np.where(ok, np.log1p(np.clip(np.nan_to_num(units).sum(axis=2), 0, None)), 0.0)
            count = ok.sum(axis=1)
            xc = np.where(ok, x - np.divide(x.sum(axis=1), count, out=np.zeros(n), where=count > 0)[:, None], 0.0)
            yc = np.where(ok, y - np.divide(y.sum(axis=1), count, out=np.zeros(n), where=count > 0)[:, None], 0.0)
            sxy, sxx = np.zeros(n_groups), np.zeros(n_groups)
            np.add.at(sxy, codes, (xc * yc).sum(axis=1))
            np.add.at(sxx, codes, (xc * xc).sum(axis=1))
            lo, hi = spec["price"]["elasticity_clip"]
            elasticity = np.where(sxx > 1e-8, np.clip(np.divide(sxy, sxx, out=np.zeros(n_groups), where=sxx > 1e-8), lo, hi), 0.0)
        target = price[:, t:t + horizon]
        ratio = np.divide(target, reference[:, None], out=np.full((n, horizon), np.nan),
                          where=~np.isnan(target) & (np.nan_to_num(reference)[:, None] > 0))
        lo, hi = spec["price"]["factor_clip"]
        with np.errstate(invalid="ignore", divide="ignore"):
            adjusted = np.clip(ratio ** elasticity[codes][:, None], lo, hi)
        price_factor = np.where(np.isnan(adjusted), 1.0, adjusted)
        if inputs.missing_price_means_not_listed:
            price_factor = np.where(np.isnan(target) & ~np.isnan(reference)[:, None], 0.0, price_factor)
        factor["price"] = price_factor
    factor["total"] = factor["event"] * factor["flag"] * factor["price"]
    return factor


def forecast_v2_exogenous(history: Any, config: Mapping[str, Any], inputs: ExogenousInputs, horizon: int = OPERATIONAL_HORIZON,
                          groups: np.ndarray | None = None, core: ForecastV2Result | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Research variant: the Core daily vector times the exogenous factors. Never a promotion candidate on its own."""
    core = forecast_v2(history, config, horizon, groups) if core is None else core
    factors = exogenous_factors(history, horizon, inputs, groups, core.profile.first_sale)
    return core.daily * factors["total"], factors
