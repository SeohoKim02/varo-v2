"""Walmart M5 development / final validation of the Varo Demand Forecast v2 candidate.

Two phases on the 13 chronological cutoffs of ``services.m5_forecast_validation`` (unchanged):

    development  origins 1-10: every v2 structure/parameter is compared and ONE configuration is chosen by the
                 pre-declared SELECTION_RULE. The sales matrix is truncated after the last development target day
                 before any selection code runs. The chosen config, its signature and the PROMOTION_GATE are then
                 written (frozen) to m5_forecast_v2_model_selection.json.
    final        origins 11-13 (origin 13 = the official M5 holdout d_1914..d_1941): the frozen config is scored
                 once against FORECAST_V1_BASELINE and the baselines, and the frozen gate decides PASS / FAIL.
                 The phase refuses to run when the config, gate, protocol or development data differ from the
                 frozen ones, or when another config was already evaluated on the final origins.

Nothing here changes production: a PASS only marks v2 as a replacement candidate for a separate task.
Read-only on raw and processed data; writes only the ``m5_forecast_v2_*`` files.

    python -m services.m5_forecast_v2_validation --phase development
    python -m services.m5_forecast_v2_validation --phase final
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from services import demand_forecast_v2 as core
from services.m5_forecast_validation import (
    AGE_BANDS, CANONICAL_INPUTS, EXACT_TOLERANCE, FREQUENCY_CLASSES, HORIZON, OPERATIONAL_HORIZON, ORIGIN_COUNT, ORIGIN_STEP,
    OUTPUT_FILES as V1_OUTPUT_FILES, RAW_DIR, RAW_INPUTS, SBC_CLASSES, VARO_INPUT_COLUMNS, VOLUME_BANDS, WEEKDAYS, M5Panel, _clean,
    _git_commit, _sha256, _write_csv, elementwise_stats, file_fingerprints, finalize_metrics, load_panel, moving_average, naive_last,
    peak_memory_mb, raw_crosscheck, rolling_origins, run_varo_forecast, scaled_errors, seasonal_naive_7, series_profile, varo_inputs)
from services.real_data_adapters import DATA_ROOT, DATASETS

EVALUATION_VERSION = "2.0.0"
DEV_COUNT = 10
V1, V2, EXO = "varo_v1_production", "varo_v2_core", "varo_v2_exogenous"

# ---------------------------------------------------------------- pre-registered protocol, search space, rule, gate
# Fixed in code before any v2 score was computed. Changing any of them changes a signature frozen by the development
# phase, and the final phase then refuses to run.

PROTOCOL: dict[str, Any] = {
    "version": "1.0.0",
    "origins": "services.m5_forecast_validation.rolling_origins: 13 cutoffs, 28-day spacing, 28-day targets (unchanged from V1 validation)",
    "development_origins": "first 10 (origins 1-10)",
    "final_rolling_origins": "last 3 (origins 11-13)",
    "official_holdout": ("origin 13 = M5 evaluation days d_1914..d_1941. It is the last of the 13 cutoffs, hence also the third final "
                         "rolling origin; it is reported and gated on its own in addition to the pooled final result."),
    "rules": [
        "Structures, parameters and routes are chosen on the development origins only.",
        "Selection code receives the sales/price/calendar matrices truncated after the last development target day.",
        "The chosen configuration, its signature and the promotion gate are frozen (written) before any final origin is scored.",
        "The final phase verifies the frozen signatures and refuses a configuration different from one already evaluated on final data.",
        "Final results never change a parameter or model; FAIL keeps V1 in production and v2 as a research candidate.",
        "Chronological expanding-window origins only; no random split; no stochastic step."],
}

SEARCH_SPACE: dict[str, Any] = {
    "routing": "per demand type (high_frequency, medium_frequency, intermittent, mostly_zero); no_sales_history is fixed at 0",
    "level_methods": core.LEVEL_METHODS,
    "weekday_methods": core.WEEKDAY_METHODS,
    "fixed_core_constants": core.CORE_CONSTANTS,
    "size": "17 level options x 4 routed types (stage 1), then 5 weekday options x 4 types (stage 2) = 88 evaluated route choices",
    "exogenous_research_constants": core.EXOGENOUS_CONSTANTS,
    "not_searched": "no per-item, per-store, per-category or per-M5-ID rule; no parameter outside the two tables",
}

SELECTION_RULE: dict[str, Any] = {
    "stage_1_level": {
        "objective": "minimise development-pooled total_7d absolute error (equivalently WAPE) within each demand type",
        "admissible": "|bias_pct| <= max(|V1 bias_pct of the same demand type|, bias_floor) on the development pool",
        "bias_floor": 0.03,
        "no_admissible_option": core.FALLBACK_LEVEL,
        "tie_tolerance": 0.001,
        "tie_break": "within (1 + tie_tolerance) x best, the first option in the declared order (simpler first)"},
    "stage_2_weekday": {
        "objective": "minimise development-pooled daily_h1_7 absolute error within each demand type, given the stage-1 level",
        "tie_tolerance": 0.001,
        "tie_break": "declared order (flat first)",
        "note": "the index has mean 1, so stage 2 cannot move the 7-day total (the primary metric)"},
    "no_development_series_of_a_type": {"level": core.FALLBACK_LEVEL, "weekday": "flat"},
}

PROMOTION_GATE: dict[str, Any] = {
    "version": "1.0.0",
    "baseline": V1,
    "candidate": V2,
    "primary": {"scope": "final_rolling_pooled", "window": "total_7d", "metric": "wape", "min_relative_improvement": 0.02},
    "consistency": {"window": "total_7d", "metric": "wape", "rule": "v2 < v1 on a strict majority of the final rolling origins (2 of 3)"},
    "official_holdout": {"windows": ["total_7d", "daily_h1_7"], "metric": "wape", "max_relative_degradation": 0.01},
    "bias": {"scope": "final_rolling_pooled", "window": "total_7d", "rule": "|bias_pct v2| <= max(|bias_pct v1|, floor)", "floor": 0.02},
    "secondary": [{"window": "daily_h1_7", "metric": "wape", "max_relative_degradation": 0.0},
                  {"window": "total_7d", "metric": "mae", "max_relative_degradation": 0.0},
                  {"window": "total_7d", "metric": "rmse", "max_relative_degradation": 0.02},
                  {"window": "daily_h1_7", "metric": "rmse", "max_relative_degradation": 0.02}],
    "demand_type_safety": {"scope": "final_rolling_pooled", "window": "total_7d", "metric": "wape",
                           "segments": ["high_frequency", "medium_frequency", "intermittent"], "max_relative_degradation": 0.03},
    "decision": "PASS only if every criterion passes; PASS = production replacement candidate (applied in a separate task); "
                "FAIL = V1 stays in production, v2 is kept as a research candidate",
    "rationale": {
        "primary": ("Across the 13 V1-validation origins the per-origin relative 7-day WAPE difference between V1 and MA28 (two methods of "
                    "near-equal accuracy) has a standard deviation of 2.5%; pooling 3 origins leaves about 1.5%. A 2% pooled improvement "
                    "(about 0.0075 WAPE points at V1's level) is therefore above origin-to-origin noise."),
        "consistency": "A pooled gain carried by a single origin is not enough: v2 must beat V1 on most final origins.",
        "official_holdout": "The official M5 holdout must not get worse than V1 by more than 1% on the 7-day total or the daily error.",
        "bias": ("Replenishment orders follow the forecast, so a biased forecast shifts every order: over-forecast turns into excess stock "
                 "and disposal, under-forecast into stock-outs. v2 may not be more biased than V1, except inside a +-2% band that "
                 "safety stock absorbs."),
        "secondary": "Daily WAPE and 7-day MAE may not get worse; RMSE (large misses drive stock-outs) may not grow by more than 2%.",
        "demand_type_safety": "No major demand type (high, medium, intermittent) may regress by more than 3% relative WAPE.",
        "mostly_zero": "Reported, not gated: 1.3% of units and a WAPE dominated by a few sales make it too noisy for a pass/fail rule."},
}

# ---------------------------------------------------------------- methods and outputs

COMPARATORS = (V1, "naive_last", "seasonal_naive_7", "moving_average_7", "moving_average_14", "moving_average_28",
               "sba_standalone", "tsb_standalone", "zero_reference")
REPORT_METHODS = (*COMPARATORS[:-1], V2, EXO, "zero_reference")
METHOD_ROLES = {V1: "production_baseline (FORECAST_V1_BASELINE)", "naive_last": "baseline", "seasonal_naive_7": "baseline",
                "moving_average_7": "baseline", "moving_average_14": "baseline", "moving_average_28": "baseline",
                "sba_standalone": "intermittent_standalone", "tsb_standalone": "intermittent_standalone",
                V2: "v2_core_candidate", EXO: "v2_exogenous_research", "zero_reference": "reference"}
METHOD_DEFINITIONS = {
    V1: "analyze_demand_forecast through its production call path; sales_7d, avg_daily_sales = sales_30d/30, sales_30d, demand_std "
        "from the last 30 days; flat daily rate; 7d = round(7*daily, 1)",
    "naive_last": "last observed day repeated", "seasonal_naive_7": "same weekday of the last observed week",
    "moving_average_7": "mean of the last 7 days, flat", "moving_average_14": "mean of the last 14 days, flat",
    "moving_average_28": "mean of the last 28 days, flat",
    "sba_standalone": "SBA alpha=0.1 on every series (core.croston variant sba), flat, no routing",
    "tsb_standalone": "TSB alpha=0.1 beta=0.05 on every series, flat, no routing",
    V2: "frozen v2 Core configuration (history only); 7d = round(7-day aggregate, 1) as in V1",
    EXO: "v2 Core x event x SNAP x planned-price factors (research only, never gated)",
    "zero_reference": "0 every day (reference, not a candidate)"}
UNROUNDED_NOTE = "V1/v2 7-day totals are rounded to 0.1 exactly as the production scalar; baselines stay unrounded as in V1 validation"
LEVEL_OPTION, WEEKDAY_OPTION = "option:", "weekday:"
SOURCE_OPTIONS = {"sba_standalone": "sba_0.1", "tsb_standalone": "tsb_0.1_0.05"}

SERIES_DIMS = {"overall": ("all",), "frequency_class": FREQUENCY_CLASSES, "age_band": AGE_BANDS, "volume_band": VOLUME_BANDS,
               "sbc_class": SBC_CLASSES}
OPTION_DIMS = ("overall", "frequency_class")
POINT_DIMS = ("horizon_day", "horizon_week", "weekday", "event_type", "snap")
WINDOWS = ("daily_h1_7", "daily_h1_28", "total_7d")
SUMS = ("n", "abs_err", "sq_err", "err", "forecast", "actual", "over_n", "under_n", "over_units", "under_units",
        "missed_demand_n", "idle_forecast_units", "scored", "mase_n", "mase_sum", "rmsse_sum")

RESULT_FILES = {
    "development": "m5_forecast_v2_development.csv",
    "final": "m5_forecast_v2_final.csv",
    "by_demand_type": "m5_forecast_v2_by_demand_type.csv",
    "by_horizon": "m5_forecast_v2_by_horizon.csv",
    "bias": "m5_forecast_v2_bias.csv",
    "by_calendar": "m5_forecast_v2_by_calendar.csv",
    "model_selection": "m5_forecast_v2_model_selection.json",
    "promotion_gate": "m5_forecast_v2_promotion_gate.json",
    "development_sums": "m5_forecast_v2_development_sums.parquet",
    "final_rows": "m5_forecast_v2_final_rows.parquet",
    "final_series": "m5_forecast_v2_final_series.parquet",
}


class FreezeViolation(RuntimeError):
    """The final phase was asked to run on something other than the frozen development selection."""


# ---------------------------------------------------------------- signatures


def signature(value: Any) -> str:
    return core.config_signature(json.loads(json.dumps(value)))


def gate_signature(gate: Mapping[str, Any] | None = None) -> str:
    """The gate constants AND the code that applies them: editing either after the freeze is detected."""
    return signature({"gate": PROMOTION_GATE if gate is None else gate, "evaluate_gate": inspect.getsource(evaluate_gate)})


# ---------------------------------------------------------------- protocol


def split_protocol(n_days: int, horizon: int = HORIZON, origin_count: int = ORIGIN_COUNT, origin_step: int = ORIGIN_STEP,
                   dev_count: int = DEV_COUNT) -> dict[str, Any]:
    origins = rolling_origins(n_days, horizon, origin_count, origin_step)
    if dev_count < 1 or origin_count - dev_count != 3:
        raise ValueError("need at least one development origin and exactly the last 3 origins as the final test")
    development, final = origins[:dev_count], origins[dev_count:]
    last_dev_day = development[-1]["target_end"]
    if not last_dev_day < final[0]["target_start"]:
        raise ValueError("development targets overlap the final targets")
    return {"development": development, "final": final, "holdout": origins[-1], "development_last_day": last_dev_day,
            "horizon": horizon, "count": origin_count, "step": origin_step, "dev_count": dev_count}


def protocol_record(protocol: Mapping[str, Any], dates: Sequence[str]) -> dict[str, Any]:
    def describe(o):
        return {"origin": o["origin"], "cutoff_day": f"d_{o['cutoff'] + 1}", "cutoff_date": str(dates[o["cutoff"]]),
                "target_days": [f"d_{o['target_start'] + 1}", f"d_{o['target_end'] + 1}"],
                "target_dates": [str(dates[o["target_start"]]), str(dates[o["target_end"]])]}
    return {**PROTOCOL, "horizon": protocol["horizon"], "operational_horizon": OPERATIONAL_HORIZON, "origin_count": protocol["count"],
            "origin_step": protocol["step"], "development": [describe(o) for o in protocol["development"]],
            "final": [describe(o) for o in protocol["final"]], "official_holdout_origin": describe(protocol["holdout"]),
            "development_last_readable_day": f"d_{protocol['development_last_day'] + 1}",
            "development_last_readable_date": str(dates[protocol["development_last_day"]])}


def truncate_panel(panel: M5Panel, last_day: int) -> M5Panel:
    """Views of every day-indexed array up to ``last_day`` inclusive: later sales, prices and flags do not exist here."""
    keep = slice(0, last_day + 1)
    return M5Panel(panel.sales[:, keep], panel.price[:, keep], panel.dates[keep], panel.series, panel.event_label[keep],
                   panel.snap[:, keep], panel.canonical_rows)


# ---------------------------------------------------------------- inputs


@dataclass
class Inputs:
    panel: M5Panel
    event_names: list[tuple[str, ...]]
    folder: Path | None
    data_signature: dict[str, Any]
    prior_v1: pd.DataFrame | None


def _event_names_from_labels(labels: Sequence[str]) -> list[tuple[str, ...]]:
    return [() if label == "none" else tuple(str(label).split("+")) for label in labels]


def load_inputs(data_root: Path | str | None, panel: M5Panel | None = None) -> Inputs:
    if panel is not None:
        return Inputs(panel, _event_names_from_labels(panel.event_label), None,
                      {"source": "in-memory panel", "sales_matrix_sha256": _matrix_sha(panel.sales)}, None)
    folder = Path(data_root) / DATASETS["m5"]
    canonical = {name: _sha256(folder / "processed" / name) for name in CANONICAL_INPUTS}
    raw = file_fingerprints(folder / RAW_DIR, RAW_INPUTS)
    panel = load_panel(folder)
    prior_path = folder / "results" / V1_OUTPUT_FILES["json"]
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    prior_signature = prior.get("data_signature", {})
    reused = (prior_signature.get("canonical_files_sha256") == canonical
              and prior_signature.get("raw_canonical_crosscheck", {}).get("cells_equal_including_missing") is True)
    crosscheck = {"reused_from": str(prior_path.name), "reason": "canonical sha256 identical to the V1 validation run whose raw "
                  "cell-for-cell crosscheck passed"} if reused else raw_crosscheck(folder, panel)
    if not reused and not crosscheck["cells_equal_including_missing"]:
        raise AssertionError(f"canonical sales differ from raw: {crosscheck}")
    events = pd.read_parquet(folder / "processed" / CANONICAL_INPUTS[3], columns=["date", "event_name"])
    names = events.groupby("date")["event_name"].agg(lambda s: tuple(sorted(set(map(str, s)))))
    prior_profile = folder / "results" / V1_OUTPUT_FILES["series_profile"]
    prior_v1 = pd.read_parquet(prior_profile, columns=["origin", "store_id", "item_id", "demand_forecast_7d", "frequency_class"]) \
        if reused and prior_profile.exists() else None
    return Inputs(panel, [names.get(d, ()) for d in panel.dates], folder,
                  {"canonical_files_sha256": canonical, "raw_files": raw, "raw_canonical_crosscheck": crosscheck,
                   "sales_matrix_shape": list(panel.sales.shape)}, prior_v1)


def _matrix_sha(matrix: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()


def pool_groups(series: pd.DataFrame) -> np.ndarray:
    """Location x category codes (generic canonical fields) used to pool weekday shapes and exogenous effects."""
    return pd.factorize(series["store_id"].astype(str) + "|" + series["category"].astype(str), sort=True)[0].astype(np.int64)


def event_matrix(names: Sequence[tuple[str, ...]]) -> tuple[np.ndarray, list[str]]:
    vocabulary = sorted({name for day in names for name in day})
    width = max([len(day) for day in names] + [1])
    matrix = np.full((len(names), width), -1, dtype=np.int64)
    for d, day in enumerate(names):
        for k, name in enumerate(day):
            matrix[d, k] = vocabulary.index(name)
    return matrix, vocabulary


# ---------------------------------------------------------------- per-origin data and scoring


@dataclass
class OriginData:
    spec: Mapping[str, Any]
    history: np.ndarray
    actual: np.ndarray
    valid: np.ndarray
    week_actual: np.ndarray
    week_valid: np.ndarray
    profile: dict[str, np.ndarray]
    core_profile: core.DemandProfile
    series_dims: dict[str, tuple[np.ndarray, Sequence[str]]]
    point_dims: dict[str, tuple[np.ndarray, Sequence[str]]]


def build_origin(panel: M5Panel, spec: Mapping[str, Any], horizon: int) -> OriginData:
    cutoff = spec["cutoff"]
    history = panel.sales[:, :cutoff + 1]
    actual = panel.sales[:, cutoff + 1:cutoff + 1 + horizon]
    if actual.shape[1] != horizon:
        raise ValueError("the panel does not contain the whole target window of this origin")
    profile = series_profile(history)
    core_profile = core.demand_profile(history)
    if not np.array_equal(core_profile.demand_type, profile["frequency_class"]):
        raise AssertionError("v2 demand types differ from the V1-validation frequency classes")
    valid = profile["eligible"][:, None] & ~np.isnan(actual)
    week_valid = valid[:, :OPERATIONAL_HORIZON].all(axis=1)
    week_actual = np.where(week_valid, np.nan_to_num(actual[:, :OPERATIONAL_HORIZON]).sum(axis=1), np.nan)
    n = len(history)
    series_dims = {"overall": (np.zeros(n, dtype=np.int64), SERIES_DIMS["overall"]),
                   **{dim: (profile[dim].astype(np.int64), labels) for dim, labels in SERIES_DIMS.items() if dim != "overall"}}
    target = np.arange(cutoff + 1, cutoff + 1 + horizon)
    weekday = pd.to_datetime(pd.Series(panel.dates[target])).dt.day_name().map(WEEKDAYS.index).to_numpy()
    event_labels = sorted(set(panel.event_label[target]))
    shape = actual.shape
    point_dims = {
        "horizon_day": (np.broadcast_to(np.arange(horizon), shape), [str(h) for h in range(1, horizon + 1)]),
        "horizon_week": (np.broadcast_to(np.arange(horizon) // 7, shape), [f"week_{w}" for w in range(1, (horizon - 1) // 7 + 2)]),
        "weekday": (np.broadcast_to(weekday, shape), list(WEEKDAYS)),
        "event_type": (np.broadcast_to(np.array([event_labels.index(e) for e in panel.event_label[target]]), shape), event_labels),
        "snap": (np.nan_to_num(panel.snap[:, target], nan=2).astype(np.int64), ["snap_0", "snap_1", "snap_unknown"]),
    }
    return OriginData(spec, history, actual, valid, week_actual, week_valid, profile, core_profile, series_dims, point_dims)


def point_stats(forecast: np.ndarray, actual: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """elementwise_stats plus two inventory proxies: demand met by a zero forecast, and forecast units where nothing sold."""
    stats = elementwise_stats(forecast, actual, valid)
    stats["missed_demand_n"] = (valid & (forecast <= EXACT_TOLERANCE) & (np.nan_to_num(actual) > 0)).astype(np.int64)
    stats["idle_forecast_units"] = np.where(valid & (np.nan_to_num(actual, nan=-1.0) == 0), forecast, 0.0)
    return stats


def _grouped(values: Mapping[str, np.ndarray], codes: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame({k: np.bincount(codes, weights=np.asarray(v, dtype=np.float64), minlength=len(labels)) for k, v in values.items()})
    frame.insert(0, "segment", [str(label) for label in labels])
    return frame[frame["n"] > 0]


def score(od: OriginData, method: str, daily: np.ndarray, total_7d: np.ndarray, *, dims: Sequence[str] = tuple(SERIES_DIMS),
          points: bool = True) -> list[pd.DataFrame]:
    """Additive error sums of one method at one origin, grouped by series dimensions and (optionally) point dimensions."""
    width = daily.shape[1]
    stats = point_stats(daily, od.actual[:, :width], od.valid[:, :width])
    zeros = np.zeros(len(daily))
    frames = []
    windows = [("daily_h1_7", slice(0, OPERATIONAL_HORIZON))] + ([("daily_h1_28", slice(0, width))] if width > OPERATIONAL_HORIZON
                                                                   or width == od.actual.shape[1] else [])
    for window, columns in windows:
        per = {k: v[:, columns].sum(axis=1) for k, v in stats.items()}
        mase, rmsse = scaled_errors(per["abs_err"], per["sq_err"], per["n"], od.profile["mae_scale"], od.profile["mse_scale"])
        per.update(scored=(per["n"] > 0).astype(float), mase_n=(~np.isnan(mase)).astype(float), mase_sum=np.nan_to_num(mase),
                   rmsse_sum=np.nan_to_num(rmsse))
        frames += [_grouped(per, *od.series_dims[d]).assign(window=window, dimension=d) for d in dims]
    week = point_stats(total_7d, od.week_actual, od.week_valid)
    week.update(scored=od.week_valid.astype(float), mase_n=zeros, mase_sum=zeros, rmsse_sum=zeros)
    frames += [_grouped(week, *od.series_dims[d]).assign(window="total_7d", dimension=d) for d in dims]
    if points:
        first_week = np.zeros(od.actual.shape, dtype=bool)
        first_week[:, :OPERATIONAL_HORIZON] = True
        for d in POINT_DIMS:
            codes, labels = od.point_dims[d]
            masks = [("daily_h1_28", od.valid)] if d.startswith("horizon") else [("daily_h1_7", od.valid & first_week),
                                                                                  ("daily_h1_28", od.valid)]
            for window, mask in masks:
                values = {k: v[mask] for k, v in stats.items()}
                values.update(scored=np.zeros(int(mask.sum())), mase_n=np.zeros(int(mask.sum())), mase_sum=np.zeros(int(mask.sum())),
                              rmsse_sum=np.zeros(int(mask.sum())))
                frames.append(_grouped(values, codes[mask], labels).assign(window=window, dimension=d))
    return [f.assign(origin=od.spec["origin"], method=method) for f in frames]


def flat(level: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(np.asarray(level, dtype=np.float64)[:, None], horizon, axis=1)


def comparator_forecasts(od: OriginData, horizon: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    history = od.history
    v1 = run_varo_forecast(varo_inputs(history), VARO_INPUT_COLUMNS["varo_production"])
    out = {V1: (flat(v1["demand_forecast_daily"].to_numpy(dtype=np.float64), horizon), v1["demand_forecast_7d"].to_numpy(dtype=np.float64))}
    for name, daily in (("naive_last", naive_last(history, horizon)), ("seasonal_naive_7", seasonal_naive_7(history, horizon)),
                        ("moving_average_7", moving_average(history, horizon, 7)), ("moving_average_14", moving_average(history, horizon, 14)),
                        ("moving_average_28", moving_average(history, horizon, 28))):
        out[name] = (daily, daily[:, :OPERATIONAL_HORIZON].sum(axis=1))
    block, start = core.recent_block(history, od.core_profile.first_sale)
    for name, option in SOURCE_OPTIONS.items():
        daily = flat(core.level_forecast(block, start, option, od.core_profile.has_negative), horizon)
        out[name] = (daily, daily[:, :OPERATIONAL_HORIZON].sum(axis=1))
    out["zero_reference"] = (np.zeros((len(history), horizon)), np.zeros(len(history)))
    return out


def exogenous_inputs(panel: M5Panel, event_ids: np.ndarray, cutoff: int, horizon: int) -> core.ExogenousInputs:
    """Covariates known in advance, up to the end of the horizon. Sales after the cutoff are never part of them."""
    end = cutoff + 1 + horizon
    return core.ExogenousInputs(events=event_ids[:end], flags={"snap": panel.snap[:, :end]}, price=panel.price[:, :end],
                                missing_price_means_not_listed=True)


def v2_forecasts(od: OriginData, panel: M5Panel, config: Mapping[str, Any], horizon: int, groups: np.ndarray,
                 event_ids: np.ndarray) -> tuple[core.ForecastV2Result, np.ndarray, float]:
    started = time.perf_counter()
    result = core.forecast_v2(od.history, config, horizon, groups)
    seconds = time.perf_counter() - started
    exo, _ = core.forecast_v2_exogenous(od.history, config, exogenous_inputs(panel, event_ids, od.spec["cutoff"], horizon), horizon,
                                        groups, core=result)
    return result, exo, seconds


# ---------------------------------------------------------------- aggregation


def pooled(sums: pd.DataFrame, scopes: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    parts = []
    for scope, origins in scopes.items():
        subset = sums[sums["origin"].isin(list(origins))]
        grouped = subset.groupby(["method", "window", "dimension", "segment"], sort=False)[list(SUMS)].sum().reset_index()
        parts.append(grouped.assign(scope=scope, origins=len(origins)))
    out = finalize_metrics(pd.concat(parts, ignore_index=True))
    actual = out["actual"].where(out["actual"] > 0)
    n = out["n"].where(out["n"] > 0)
    out["fill_rate_proxy"] = 1.0 - out["under_units"] / actual
    out["excess_units_ratio"] = out["over_units"] / actual
    out["missed_demand_rate"] = out["missed_demand_n"] / n
    out["idle_forecast_ratio"] = out["idle_forecast_units"] / out["forecast"].where(out["forecast"] > 0)
    out["abs_bias_pct"] = out["bias_pct"].abs()
    return out


def with_v1_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["scope", "window", "dimension", "segment"]
    base = frame[frame["method"] == V1][[*keys, "wape", "mae", "rmse", "bias_pct"]]
    out = frame.merge(base.rename(columns={m: f"v1_{m}" for m in ("wape", "mae", "rmse", "bias_pct")}), on=keys, how="left")
    for metric in ("wape", "mae", "rmse"):
        out[f"{metric}_vs_v1_relative"] = out[metric] / out[f"v1_{metric}"] - 1.0
    candidates = out["method"].isin([m for m in REPORT_METHODS if m != "zero_reference"])
    out["rank_wape"] = out["wape"].where(candidates).groupby([out[k] for k in keys]).rank(method="min")
    return out.drop(columns=[f"v1_{m}" for m in ("mae", "rmse")])


METRIC_COLUMNS = ["points", "series_origins", "actual_units", "forecast_units", "mae", "rmse", "wape", "mean_error", "bias_units",
                  "bias_pct", "abs_bias_pct", "over_forecast_rate", "under_forecast_rate", "over_units", "under_units", "fill_rate_proxy",
                  "excess_units_ratio", "missed_demand_rate", "idle_forecast_ratio", "mase_mean", "rmsse_mean"]
COMPARE_COLUMNS = ["v1_wape", "wape_vs_v1_relative", "mae_vs_v1_relative", "rmse_vs_v1_relative", "v1_bias_pct", "rank_wape"]
BIAS_COLUMNS = ["points", "actual_units", "forecast_units", "mean_error", "bias_units", "bias_pct", "abs_bias_pct", "over_forecast_rate",
                "under_forecast_rate", "exact_rate", "over_units", "under_units", "fill_rate_proxy", "excess_units_ratio",
                "missed_demand_rate", "idle_forecast_ratio", "v1_bias_pct"]


def _method_order(method: str) -> tuple[int, int, str]:
    if method in REPORT_METHODS:
        return (0, REPORT_METHODS.index(method), method)
    if method.startswith(LEVEL_OPTION):
        return (1, list(core.LEVEL_METHODS).index(method[len(LEVEL_OPTION):]), method)
    return (2, list(core.WEEKDAY_METHODS).index(method[len(WEEKDAY_OPTION):]), method)


def _role(method: str) -> str:
    if method.startswith(LEVEL_OPTION):
        return "development_level_option"
    if method.startswith(WEEKDAY_OPTION):
        return "development_weekday_option"
    return METHOD_ROLES[method]


def tidy(frame: pd.DataFrame, leading: Sequence[str], metrics: Sequence[str], scope_order: Sequence[str]) -> pd.DataFrame:
    out = frame.rename(columns={"n": "points", "scored": "series_origins", "actual": "actual_units", "forecast": "forecast_units"})
    out = out.assign(method_role=out["method"].map(_role))
    segment_rank = {d: {s: i for i, s in enumerate(labels)} for d, labels in {**SERIES_DIMS, "weekday": WEEKDAYS}.items()}
    order = pd.DataFrame({
        "_scope": out["scope"].map({s: i for i, s in enumerate(scope_order)}),
        "_window": out["window"].map({w: i for i, w in enumerate(WINDOWS)}),
        "_dimension": out["dimension"].map({d: i for i, d in enumerate([*SERIES_DIMS, *POINT_DIMS])}),
        "_segment": [segment_rank.get(d, {}).get(s, int(s) if str(s).isdigit() else 1000) for d, s in zip(out["dimension"], out["segment"])],
        "_segment_text": out["segment"].astype(str),
        "_method": [_method_order(m) for m in out["method"]]}, index=out.index)
    out = out.join(order).sort_values(["_scope", "_window", "_dimension", "_segment", "_segment_text", "_method"], kind="mergesort")
    columns = [c for c in [*leading, "method", "method_role", *metrics] if c in out.columns]
    return out[columns].reset_index(drop=True)


# ---------------------------------------------------------------- selection


def select_levels(stage1: pd.DataFrame, rule: Mapping[str, Any] = SELECTION_RULE) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Stage 1: per demand type, the admissible level option with the lowest development 7-day WAPE (declared tie-break)."""
    spec = rule["stage_1_level"]
    chosen, decisions = {}, []
    for demand_type in core.ROUTED_TYPES:
        rows = stage1[(stage1["segment"] == demand_type)]
        v1 = rows[rows["method"] == V1]
        v1_bias = float(v1["bias_pct"].iloc[0]) if len(v1) and pd.notna(v1["bias_pct"].iloc[0]) else np.nan
        bound = max(abs(v1_bias), spec["bias_floor"]) if np.isfinite(v1_bias) else spec["bias_floor"]
        options = []
        for name in core.LEVEL_METHODS:
            row = rows[rows["method"] == LEVEL_OPTION + name]
            if len(row):
                r = row.iloc[0]
                admissible = bool(pd.notna(r["wape"]) and pd.notna(r["bias_pct"]) and abs(r["bias_pct"]) <= bound + 1e-12)
                options.append({"option": name, "wape": float(r["wape"]), "bias_pct": float(r["bias_pct"]), "abs_err": float(r["abs_err"]),
                                "actual": float(r["actual"]), "admissible": admissible})
        admissible = [o for o in options if o["admissible"]]
        if not options:
            pick, reason = rule["no_development_series_of_a_type"]["level"], "no development series of this type"
        elif not admissible:
            pick, reason = spec["no_admissible_option"], "no option inside the bias bound"
        else:
            best = min(o["wape"] for o in admissible)
            pick = next(o["option"] for o in admissible if o["wape"] <= best * (1 + spec["tie_tolerance"]))
            winner = min(admissible, key=lambda o: o["wape"])["option"]
            reason = "lowest development 7-day WAPE" if pick == winner else f"tie-break: within {spec['tie_tolerance']:.1%} of {winner}"
        chosen[demand_type] = pick
        decisions.append({"demand_type": demand_type, "v1_bias_pct": v1_bias, "bias_bound": bound, "chosen": pick, "reason": reason,
                          "options": options})
    return chosen, decisions


def select_weekdays(stage2: pd.DataFrame, rule: Mapping[str, Any] = SELECTION_RULE) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Stage 2: per demand type, the weekday option with the lowest development daily (days 1-7) absolute error."""
    spec = rule["stage_2_weekday"]
    chosen, decisions = {}, []
    for demand_type in core.ROUTED_TYPES:
        rows = stage2[stage2["segment"] == demand_type]
        options = []
        for name in core.WEEKDAY_METHODS:
            row = rows[rows["method"] == WEEKDAY_OPTION + name]
            if len(row):
                r = row.iloc[0]
                options.append({"option": name, "abs_err": float(r["abs_err"]), "wape": float(r["wape"]), "rmse": float(r["rmse"])})
        if not options:
            pick, reason = rule["no_development_series_of_a_type"]["weekday"], "no development series of this type"
        else:
            best = min(o["abs_err"] for o in options)
            pick = next(o["option"] for o in options if o["abs_err"] <= best * (1 + spec["tie_tolerance"]))
            winner = min(options, key=lambda o: o["abs_err"])["option"]
            reason = "lowest development daily absolute error" if pick == winner else f"tie-break: within {spec['tie_tolerance']:.1%} of {winner}"
        chosen[demand_type] = pick
        decisions.append({"demand_type": demand_type, "chosen": pick, "reason": reason, "options": options})
    return chosen, decisions


def routed_levels(levels: Mapping[str, np.ndarray], demand_type: np.ndarray, routes: Mapping[str, str]) -> np.ndarray:
    out = np.zeros(len(demand_type))
    for code, name in enumerate(core.ROUTED_TYPES):
        rows = demand_type == code
        out[rows] = levels[routes[name]][rows]
    return out


# ---------------------------------------------------------------- promotion gate


def evaluate_gate(metrics: pd.DataFrame, pooled_scope: str, origin_scopes: Sequence[str], holdout_scope: str | None,
                  gate: Mapping[str, Any] = PROMOTION_GATE) -> dict[str, Any]:
    """Apply the frozen gate to a metrics table (scope, window, dimension, segment, method, wape/mae/rmse/bias_pct)."""
    base, cand = gate["baseline"], gate["candidate"]

    def value(method, scope, window, metric, dimension="overall", segment="all"):
        row = metrics[(metrics["method"] == method) & (metrics["scope"] == scope) & (metrics["window"] == window)
                      & (metrics["dimension"] == dimension) & (metrics["segment"] == segment)]
        return float(row[metric].iloc[0]) if len(row) else float("nan")

    criteria = []

    def add(cid, description, v1, v2, threshold, passed, **extra):
        criteria.append({"id": cid, "description": description, "v1": v1, "v2": v2,
                         "relative_change": (v2 / v1 - 1.0) if v1 and np.isfinite(v1) and np.isfinite(v2) else None,
                         "threshold": threshold, "pass": bool(passed), **extra})

    primary = gate["primary"]
    v1, v2 = value(base, pooled_scope, primary["window"], "wape"), value(cand, pooled_scope, primary["window"], "wape")
    limit = v1 * (1 - primary["min_relative_improvement"])
    add("G1_primary_7d_wape", f"{pooled_scope} total_7d WAPE improves by >= {primary['min_relative_improvement']:.0%}", v1, v2,
        limit, np.isfinite(v2) and v2 <= limit)
    wins = []
    for scope in origin_scopes:
        a, b = value(base, scope, gate["consistency"]["window"], "wape"), value(cand, scope, gate["consistency"]["window"], "wape")
        wins.append({"scope": scope, "v1": a, "v2": b, "v2_better": bool(b < a)})
    won = sum(w["v2_better"] for w in wins)
    add("G2_consistency_origin_wins", "v2 total_7d WAPE < V1 on a strict majority of origins", None, None,
        f"> {len(wins) / 2:g} of {len(wins)}", won > len(wins) / 2, wins=won, origins=len(wins), per_origin=wins)
    if holdout_scope is not None:
        tolerance = gate["official_holdout"]["max_relative_degradation"]
        for window in gate["official_holdout"]["windows"]:
            a, b = value(base, holdout_scope, window, "wape"), value(cand, holdout_scope, window, "wape")
            add(f"G3_official_holdout_{window}_wape", f"official holdout {window} WAPE no more than {tolerance:.0%} worse", a, b,
                a * (1 + tolerance), np.isfinite(b) and b <= a * (1 + tolerance))
    bias = gate["bias"]
    a, b = value(base, pooled_scope, bias["window"], "bias_pct"), value(cand, pooled_scope, bias["window"], "bias_pct")
    bound = max(abs(a), bias["floor"])
    add("G4_abs_bias", f"|bias| <= max(|V1 bias|, {bias['floor']:.0%}) on {pooled_scope} total_7d", a, b, bound,
        np.isfinite(b) and abs(b) <= bound + 1e-12)
    for item in gate["secondary"]:
        a, b = value(base, pooled_scope, item["window"], item["metric"]), value(cand, pooled_scope, item["window"], item["metric"])
        limit = a * (1 + item["max_relative_degradation"])
        add(f"G5_secondary_{item['window']}_{item['metric']}", f"{item['window']} {item['metric'].upper()} no more than "
            f"{item['max_relative_degradation']:.0%} worse", a, b, limit, np.isfinite(b) and b <= limit + 1e-12)
    safety = gate["demand_type_safety"]
    for segment in safety["segments"]:
        a = value(base, pooled_scope, safety["window"], "wape", "frequency_class", segment)
        b = value(cand, pooled_scope, safety["window"], "wape", "frequency_class", segment)
        limit = a * (1 + safety["max_relative_degradation"])
        passed = (not np.isfinite(a)) or (np.isfinite(b) and b <= limit)
        add(f"G6_demand_type_{segment}", f"{segment} total_7d WAPE no more than {safety['max_relative_degradation']:.0%} worse", a, b,
            limit, passed)
    decision = "PASS" if all(c["pass"] for c in criteria) else "FAIL"
    return {"decision": decision, "passed": sum(c["pass"] for c in criteria), "criteria_count": len(criteria),
            "failed": [c["id"] for c in criteria if not c["pass"]], "criteria": criteria}


# ---------------------------------------------------------------- phases


def _timer():
    started = time.perf_counter()
    stages: dict[str, float] = {}

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        stages[name] = round(now - since, 3)
        return now
    return started, stages, lap


def _write_tables(output: Path, metrics: pd.DataFrame, scope_order: Sequence[str], holdout_scope: str) -> None:
    """by_demand_type / by_horizon / bias / by_calendar for the pooled phase scopes present in ``metrics``."""
    report = metrics[metrics["method"].isin(REPORT_METHODS) & metrics["scope"].isin(
        ["development_pooled", "final_rolling_pooled", holdout_scope])].copy()
    report["phase"] = np.where(report["scope"] == "development_pooled", "development", "final")
    lead = ["phase", "scope", "origins", "window", "dimension", "segment"]
    by_type = report[report["dimension"].isin(["frequency_class", "age_band", "volume_band", "sbc_class"])]
    _write_csv(tidy(by_type, lead, [*METRIC_COLUMNS, *COMPARE_COLUMNS], scope_order), output / RESULT_FILES["by_demand_type"])
    horizon = report[report["dimension"].isin(["horizon_day", "horizon_week"])]
    _write_csv(tidy(horizon, lead, [*METRIC_COLUMNS, *COMPARE_COLUMNS], scope_order), output / RESULT_FILES["by_horizon"])
    bias = report[report["dimension"].isin(["overall", "frequency_class", "age_band", "volume_band"])]
    _write_csv(tidy(bias, lead, BIAS_COLUMNS, scope_order), output / RESULT_FILES["bias"])
    calendar = report[report["dimension"].isin(["weekday", "event_type", "snap"])]
    _write_csv(tidy(calendar, lead, [*METRIC_COLUMNS, *COMPARE_COLUMNS], scope_order), output / RESULT_FILES["by_calendar"])


def _headline(metrics: pd.DataFrame, scope: str, methods: Sequence[str] = REPORT_METHODS) -> dict[str, Any]:
    rows = metrics[(metrics["scope"] == scope) & (metrics["dimension"] == "overall") & metrics["method"].isin(methods)]
    keep = ["wape", "mae", "rmse", "bias_pct", "mase_mean", "fill_rate_proxy", "excess_units_ratio", "rank_wape"]
    return {w: {r["method"]: {k: r[k] for k in keep if k in rows.columns and pd.notna(r[k])} for _, r in rows[rows["window"] == w].iterrows()}
            for w in WINDOWS}


def run_development(data_root: Path | str | None = DATA_ROOT, output_dir: Path | str | None = None, *, horizon: int = HORIZON,
                    origin_count: int = ORIGIN_COUNT, origin_step: int = ORIGIN_STEP, dev_count: int = DEV_COUNT,
                    panel: M5Panel | None = None) -> dict[str, Any]:
    """Choose and freeze ONE v2 configuration from the development origins."""
    started, stages, lap = _timer()
    inputs = load_inputs(data_root, panel)
    output = Path(output_dir) if output_dir else inputs.folder / "results"
    output.mkdir(parents=True, exist_ok=True)
    tick = lap("load_inputs", started)
    protocol = split_protocol(len(inputs.panel.dates), horizon, origin_count, origin_step, dev_count)
    # Everything below sees only days <= the last development target day.
    last = protocol["development_last_day"]
    dev_panel = truncate_panel(inputs.panel, last)
    events, _ = event_matrix(inputs.event_names[:last + 1])
    groups = pool_groups(dev_panel.series)
    dev_ids = [o["origin"] for o in protocol["development"]]
    position = np.arange(1, OPERATIONAL_HORIZON + 1) % core.WEEK

    sums, caches, v1_check = [], [], []
    for spec in protocol["development"]:
        od = build_origin(dev_panel, spec, horizon)
        for name, (daily, total) in comparator_forecasts(od, horizon).items():
            sums += score(od, name, daily, total)
            if name == V1 and inputs.prior_v1 is not None:
                prior = inputs.prior_v1[inputs.prior_v1["origin"] == spec["origin"]]
                v1_check.append(bool(np.array_equal(prior["demand_forecast_7d"].to_numpy(), total)
                                     and np.array_equal(prior["store_id"].astype(str).to_numpy(), dev_panel.series["store_id"].astype(str).to_numpy())
                                     and np.array_equal(prior["item_id"].astype(str).to_numpy(), dev_panel.series["item_id"].astype(str).to_numpy())))
        block, start = core.recent_block(od.history, od.core_profile.first_sale)
        levels = {m: core.level_forecast(block, start, m, od.core_profile.has_negative) for m in core.LEVEL_METHODS}
        weekday = {w: core.weekday_index(block, start, w, groups) for w in core.WEEKDAY_METHODS}
        for name, level in levels.items():
            sums += score(od, LEVEL_OPTION + name, flat(level, OPERATIONAL_HORIZON), np.round(OPERATIONAL_HORIZON * level, 1),
                          dims=OPTION_DIMS, points=False)
        caches.append((od, levels, weekday))
    tick = lap("development_pass1_comparators_and_level_options", tick)

    scopes = {"development_pooled": dev_ids, **{f"development_origin_{k:02d}": [k] for k in dev_ids}}
    pass1 = pd.concat(sums, ignore_index=True)
    stage1 = pooled(pass1[(pass1["method"] == V1) | pass1["method"].str.startswith(LEVEL_OPTION)], {"development_pooled": dev_ids})
    stage1 = stage1[(stage1["window"] == "total_7d") & (stage1["dimension"] == "frequency_class")]
    level_routes, level_decisions = select_levels(stage1)
    for od, levels, weekday in caches:
        chosen = routed_levels(levels, od.core_profile.demand_type, level_routes)
        for name, (index, _) in weekday.items():
            sums += score(od, WEEKDAY_OPTION + name, chosen[:, None] * index[:, position], np.round(OPERATIONAL_HORIZON * chosen, 1),
                          dims=OPTION_DIMS, points=False)
    pass2 = pd.concat(sums, ignore_index=True)
    stage2 = pooled(pass2[pass2["method"].str.startswith(WEEKDAY_OPTION)], {"development_pooled": dev_ids})
    stage2 = stage2[(stage2["window"] == "daily_h1_7") & (stage2["dimension"] == "frequency_class")]
    weekday_routes, weekday_decisions = select_weekdays(stage2)
    config = core.make_config({t: {"level": level_routes[t], "weekday": weekday_routes[t]} for t in core.ROUTED_TYPES})
    config_sig = core.config_signature(config)
    tick = lap("development_selection", tick)

    v2_seconds, consistency = [], []
    for od, levels, weekday in caches:
        result, exo, seconds = v2_forecasts(od, dev_panel, config, horizon, groups, events)
        v2_seconds.append(seconds)
        expected = routed_levels(levels, od.core_profile.demand_type, level_routes)
        index = np.ones((len(expected), core.WEEK))
        for code, name in enumerate(core.ROUTED_TYPES):
            rows = od.core_profile.demand_type == code
            index[rows] = weekday[weekday_routes[name]][0][rows]
        consistency.append(bool(np.array_equal(result.level, expected) and np.allclose(result.weekday_index, index)))
        sums += score(od, V2, result.daily, np.round(result.aggregate_7d, 1))
        sums += score(od, EXO, exo, np.round(exo[:, :OPERATIONAL_HORIZON].sum(axis=1), 1))
    if not all(consistency):
        raise AssertionError("forecast_v2 differs from the development option cache it was selected from")
    tick = lap("development_pass3_v2_and_exogenous", tick)

    dev_sums = pd.concat(sums, ignore_index=True)
    metrics = with_v1_comparison(pooled(dev_sums, scopes))
    scope_order = list(scopes)
    development = metrics[metrics["dimension"].isin(OPTION_DIMS)]
    _write_csv(tidy(development, ["scope", "origins", "window", "dimension", "segment"], [*METRIC_COLUMNS, *COMPARE_COLUMNS], scope_order),
               output / RESULT_FILES["development"])
    pq.write_table(pa.Table.from_pandas(dev_sums, preserve_index=False), output / RESULT_FILES["development_sums"], compression="zstd")
    holdout_scope = f"final_origin_{protocol['holdout']['origin']}_official_holdout"
    _write_tables(output, metrics, [*scope_order, "final_rolling_pooled", holdout_scope], holdout_scope)
    tick = lap("development_outputs", tick)

    precheck = evaluate_gate(metrics, "development_pooled", [f"development_origin_{k:02d}" for k in dev_ids], None)
    record = protocol_record(protocol, inputs.panel.dates)
    selection = {
        "evaluation": {"name": "Varo Demand Forecast v2 - development selection on Walmart M5", "version": EVALUATION_VERSION,
                       "phase": "development", "generated_at": _now(), "git_commit": _git_commit(),
                       "command": "python -m services.m5_forecast_v2_validation --phase development"},
        "v1_baseline": {**core.FORECAST_V1_BASELINE, "current_fingerprint": core.v1_baseline_fingerprint(),
                        "reproduces_v1_validation_forecasts": (all(v1_check) if v1_check else None)},
        "protocol": record, "protocol_signature": signature(record),
        "search_space": SEARCH_SPACE, "search_space_signature": signature(SEARCH_SPACE),
        "selection_rule": SELECTION_RULE, "selection_rule_signature": signature(SELECTION_RULE),
        "stage_1_level_selection": level_decisions, "stage_2_weekday_selection": weekday_decisions,
        "selected": {"routes": config["routes"], "config": config, "config_signature": config_sig,
                     "consistency_check": "forecast_v2 reproduces the cached option levels and weekday indexes it was selected from"},
        "development_metrics": _headline(metrics, "development_pooled"),
        "development_gate_precheck": {"note": "informational only: the gate is decided on the final origins", **precheck},
        "promotion_gate": PROMOTION_GATE, "promotion_gate_signature": gate_signature(), "frozen_at": _now(),
        "data_signature": {**inputs.data_signature, "development_sales_sha256": _matrix_sha(dev_panel.sales)},
        "leakage_controls": [
            f"Selection runs on matrices truncated after {record['development_last_readable_day']} (the last development target day).",
            "Each forecast receives only the history columns <= its cutoff; targets are sliced separately.",
            "Demand types, weekday profiles, Croston/TSB states and MASE scales are history-only.",
            "The exogenous research variant reads event calendar, SNAP flags and planned prices for target days (known in advance), "
            "never target-day sales."],
        "runtime": {"stages_seconds": stages, "total_seconds": round(time.perf_counter() - started, 3), "peak_memory_mb": peak_memory_mb(),
                    "v2_core_seconds_per_origin": [round(s, 4) for s in v2_seconds],
                    "v2_core_microseconds_per_series": round(1e6 * float(np.mean(v2_seconds)) / len(inputs.panel.series), 3)},
        "random_seed": None, "random_seed_note": "deterministic: no sampling, no stochastic model, no random split",
        "rounding_note": UNROUNDED_NOTE,
        "method_definitions": METHOD_DEFINITIONS,
    }
    (output / RESULT_FILES["model_selection"]).write_text(json.dumps(_clean(selection), ensure_ascii=False, indent=2), encoding="utf-8")
    return _clean(selection)


def verify_freeze(selection: Mapping[str, Any], protocol_now: Mapping[str, Any], development_sha: str) -> dict[str, bool]:
    checks = {
        "config_signature_matches_config": core.config_signature(selection["selected"]["config"]) == selection["selected"]["config_signature"],
        "config_matches_declared_methods": True,
        "gate_unchanged_since_freeze": gate_signature() == selection["promotion_gate_signature"],
        "gate_constants_equal_frozen_copy": json.loads(json.dumps(PROMOTION_GATE)) == selection["promotion_gate"],
        "protocol_unchanged": signature(protocol_now) == selection["protocol_signature"],
        "search_space_unchanged": signature(SEARCH_SPACE) == selection["search_space_signature"],
        "selection_rule_unchanged": signature(SELECTION_RULE) == selection["selection_rule_signature"],
        "development_data_unchanged": development_sha == selection["data_signature"]["development_sales_sha256"],
        "frozen_before_final": bool(selection.get("frozen_at")),
    }
    try:
        core.validate_config(selection["selected"]["config"])
    except (ValueError, KeyError):
        checks["config_matches_declared_methods"] = False
    return checks


def run_final(data_root: Path | str | None = DATA_ROOT, output_dir: Path | str | None = None, *, horizon: int = HORIZON,
              origin_count: int = ORIGIN_COUNT, origin_step: int = ORIGIN_STEP, dev_count: int = DEV_COUNT,
              panel: M5Panel | None = None, write_rows: bool = True) -> dict[str, Any]:
    """Score the frozen configuration once on the final origins and apply the frozen promotion gate."""
    started, stages, lap = _timer()
    inputs = load_inputs(data_root, panel)
    output = Path(output_dir) if output_dir else inputs.folder / "results"
    selection_path = output / RESULT_FILES["model_selection"]
    if not selection_path.exists():
        raise FreezeViolation("no frozen development selection: run the development phase first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    protocol = split_protocol(len(inputs.panel.dates), horizon, origin_count, origin_step, dev_count)
    record = protocol_record(protocol, inputs.panel.dates)
    checks = verify_freeze(selection, record, _matrix_sha(inputs.panel.sales[:, :protocol["development_last_day"] + 1]))
    if not all(checks.values()):
        raise FreezeViolation(f"frozen selection verification failed: {[k for k, v in checks.items() if not v]}")
    config, config_sig = selection["selected"]["config"], selection["selected"]["config_signature"]
    gate_path = output / RESULT_FILES["promotion_gate"]
    if gate_path.exists():
        previous = json.loads(gate_path.read_text(encoding="utf-8"))
        if previous.get("config_signature") != config_sig:
            raise FreezeViolation("a different configuration was already evaluated on the final origins; re-selection after seeing "
                                  "final results is not allowed")
    evaluated_at = _now()
    tick = lap("load_and_verify", started)

    events, _ = event_matrix(inputs.event_names)
    groups = pool_groups(inputs.panel.series)
    final_ids = [o["origin"] for o in protocol["final"]]
    holdout_id = protocol["holdout"]["origin"]
    sums, v2_seconds, series_parts = [], [], []
    writer = None
    rows_path = output / RESULT_FILES["final_rows"]
    try:
        for spec in protocol["final"]:
            od = build_origin(inputs.panel, spec, horizon)
            comparators = comparator_forecasts(od, horizon)
            for name, (daily, total) in comparators.items():
                sums += score(od, name, daily, total)
            result, exo, seconds = v2_forecasts(od, inputs.panel, config, horizon, groups, events)
            v2_seconds.append(seconds)
            v2_total, exo_total = np.round(result.aggregate_7d, 1), np.round(exo[:, :OPERATIONAL_HORIZON].sum(axis=1), 1)
            sums += score(od, V2, result.daily, v2_total)
            sums += score(od, EXO, exo, exo_total)
            series_parts.append(_series_frame(inputs.panel, od, result, comparators, v2_total, exo_total))
            if write_rows:
                writer = _write_rows(writer, rows_path, inputs.panel, od, {V1: comparators[V1][0], V2: result.daily, EXO: exo,
                                                                            "moving_average_28": comparators["moving_average_28"][0]}, result)
    finally:
        if writer is not None:
            writer.close()
    tick = lap("final_forecast_and_score", tick)

    final_sums = pd.concat(sums, ignore_index=True)
    holdout_scope = f"final_origin_{holdout_id}_official_holdout"
    origin_scopes = {f"final_origin_{k}" + ("_official_holdout" if k == holdout_id else ""): [k] for k in final_ids}
    scopes = {"final_rolling_pooled": final_ids, **origin_scopes}
    metrics = with_v1_comparison(pooled(final_sums, scopes))
    gate = evaluate_gate(metrics, "final_rolling_pooled", list(origin_scopes), holdout_scope)
    dev_sums_path = output / RESULT_FILES["development_sums"]
    dev_metrics = with_v1_comparison(pooled(pd.read_parquet(dev_sums_path), {"development_pooled": list(range(1, dev_count + 1))})) \
        if dev_sums_path.exists() else metrics.iloc[0:0]
    scope_order = ["development_pooled", *scopes]
    overall = metrics[(metrics["dimension"] == "overall") & metrics["method"].isin(REPORT_METHODS)]
    _write_csv(tidy(overall, ["scope", "origins", "window", "dimension", "segment"], [*METRIC_COLUMNS, *COMPARE_COLUMNS], scope_order),
               output / RESULT_FILES["final"])
    _write_tables(output, pd.concat([dev_metrics, metrics], ignore_index=True), scope_order, holdout_scope)
    series = pd.concat(series_parts, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(series, preserve_index=False), output / RESULT_FILES["final_series"], compression="zstd")
    tick = lap("final_outputs", tick)

    raw_after = file_fingerprints(inputs.folder / RAW_DIR, RAW_INPUTS) if inputs.folder else None
    raw_unchanged = None if inputs.folder is None else raw_after == inputs.data_signature["raw_files"]
    if raw_unchanged is False:
        raise AssertionError("raw M5 files changed during the evaluation")
    report = {
        "evaluation": {"name": "Varo Demand Forecast v2 - final test and promotion gate on Walmart M5", "version": EVALUATION_VERSION,
                       "phase": "final", "generated_at": evaluated_at, "git_commit": _git_commit(),
                       "command": "python -m services.m5_forecast_v2_validation --phase final"},
        "decision": gate["decision"],
        "decision_meaning": ("PASS: v2 Core is a production replacement candidate (not applied in this task)" if gate["decision"] == "PASS"
                             else "FAIL: V1 stays in production; v2 Core is kept as a research candidate"),
        "production_changed": False,
        "config_signature": config_sig, "routes": config["routes"], "protocol": record,
        "promotion_gate_signature": selection["promotion_gate_signature"], "frozen_at": selection["frozen_at"],
        "final_evaluated_at": evaluated_at, "freeze_verification": checks,
        "gate": PROMOTION_GATE, **{k: gate[k] for k in ("passed", "criteria_count", "failed", "criteria")},
        "final_metrics": {scope: _headline(metrics, scope) for scope in scopes},
        "development_metrics": selection.get("development_metrics"),
        "exogenous_note": "varo_v2_exogenous is research only: reported next to the Core, never gated.",
        "inventory_interpretation": inventory_interpretation(metrics, "final_rolling_pooled"),
        "data_signature": {**inputs.data_signature, "raw_unchanged_during_run": raw_unchanged},
        "runtime": {"stages_seconds": stages, "total_seconds": round(time.perf_counter() - started, 3), "peak_memory_mb": peak_memory_mb(),
                    "v2_core_seconds_per_origin": [round(s, 4) for s in v2_seconds],
                    "v2_core_microseconds_per_series": round(1e6 * float(np.mean(v2_seconds)) / len(inputs.panel.series), 3)},
        "rounding_note": UNROUNDED_NOTE,
        "limitations": [
            "M5 sales are censored by unobserved stock-outs: this validates sales forecasting, not latent demand.",
            "Production uploads carry sales_7d/sales_30d aggregates, not a daily history; v2 Core needs the daily history "
            "(canonical demand_series) and falls back to V1 where only aggregates exist - to be wired in the production task.",
            "All 13 origins start on the same weekday (28-day spacing), so horizon day h always maps to the same weekday.",
            "No WRMSSE: MASE/RMSSE are unweighted series means."],
    }
    gate_path.write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return _clean(report)


def inventory_interpretation(metrics: pd.DataFrame, scope: str) -> dict[str, Any]:
    rows = metrics[(metrics["scope"] == scope) & (metrics["window"] == "total_7d") & (metrics["dimension"] == "overall")]
    table = {r["method"]: {"bias_pct": r["bias_pct"], "fill_rate_proxy": r["fill_rate_proxy"], "excess_units_ratio": r["excess_units_ratio"],
                           "under_forecast_rate": r["under_forecast_rate"], "over_forecast_rate": r["over_forecast_rate"],
                           "missed_demand_rate": r["missed_demand_rate"]} for _, r in rows.iterrows() if r["method"] in REPORT_METHODS}
    return {"definitions": {
        "fill_rate_proxy": "1 - sum max(A-F,0) / sum A: share of 7-day demand covered if the week's stock equalled the forecast",
        "excess_units_ratio": "sum max(F-A,0) / sum A: stock left over per unit sold under the same policy (holding / disposal exposure)",
        "under_forecast_rate": "share of series-weeks with A > F (stock-out risk)",
        "over_forecast_rate": "share of series-weeks with F > A (excess-stock risk)",
        "missed_demand_rate": "share of series-weeks with F = 0 but A > 0 (certain stock-out when stock follows the forecast)"},
        "why_bias_matters": ("An order-up-to or min-max policy sets stock = forecast over the lead time + safety stock. Safety stock covers "
                             "random error, not bias: a +x% bias raises every order by x% (holding cost, markdown and disposal for "
                             "perishables), a -x% bias lowers every order and converts directly into stock-outs. WAPE can be reduced by "
                             "under-forecasting intermittent items (the median of their weekly demand is often 0), so bias is gated "
                             "separately from WAPE."),
        "by_method": table}


def _series_frame(panel: M5Panel, od: OriginData, result: core.ForecastV2Result, comparators: Mapping[str, Any], v2_total: np.ndarray,
                  exo_total: np.ndarray) -> pd.DataFrame:
    p = result.profile
    return pd.DataFrame({
        "origin": od.spec["origin"], "cutoff_date": panel.dates[od.spec["cutoff"]], "store_id": panel.series["store_id"].to_numpy(),
        "item_id": panel.series["item_id"].to_numpy(), "demand_type": np.array(core.DEMAND_TYPES, dtype=object)[p.demand_type],
        "age_band": np.array(AGE_BANDS, dtype=object)[od.profile["age_band"]], "level_method": result.level_method,
        "weekday_method": result.weekday_method, "weekday_applied": result.weekday_applied, "nonzero_ratio": p.nonzero_ratio,
        "recent_mean": p.recent_mean, "long_mean": p.long_mean, "cv": p.cv, "adi": np.where(np.isfinite(p.adi), p.adi, np.nan),
        "trend_ratio": p.trend_ratio, "weekday_strength": p.weekday_strength, "history_length": p.history_length,
        "forecast_7d_v1": comparators[V1][1], "forecast_7d_v2_core": v2_total, "forecast_7d_v2_exogenous": exo_total,
        "forecast_7d_ma28": comparators["moving_average_28"][1], "actual_7d": od.week_actual})


def _write_rows(writer, path: Path, panel: M5Panel, od: OriginData, forecasts: Mapping[str, np.ndarray],
                result: core.ForecastV2Result) -> pq.ParquetWriter:
    n, horizon = od.actual.shape
    target = np.arange(od.spec["cutoff"] + 1, od.spec["cutoff"] + 1 + horizon)
    encode = lambda values: pa.array(values, pa.string()).dictionary_encode()
    table = pa.table({
        "origin": pa.array(np.full(n * horizon, od.spec["origin"], dtype=np.int16)),
        "store_id": encode(np.repeat(panel.series["store_id"].to_numpy(dtype=object), horizon)),
        "item_id": encode(np.repeat(panel.series["item_id"].to_numpy(dtype=object), horizon)),
        "h": pa.array(np.tile(np.arange(1, horizon + 1, dtype=np.int8), n)),
        "target_date": encode(np.tile(panel.dates[target].astype(object), n)),
        "demand_type": encode(np.repeat(np.array(core.DEMAND_TYPES, dtype=object)[result.demand_type], horizon)),
        "actual": pa.array(od.actual.ravel()),
        **{f"forecast_{name}": pa.array(np.asarray(values, dtype=np.float64).ravel()) for name, values in forecasts.items()}})
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=["development", "final"], required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-rows", action="store_true", help="final phase: skip the row-level parquet")
    args = parser.parse_args()
    if args.phase == "development":
        report = run_development(args.data_root, args.output_dir)
        print(json.dumps({"selected": report["selected"]["routes"], "config_signature": report["selected"]["config_signature"],
                          "development_total_7d": report["development_metrics"]["total_7d"], "runtime": report["runtime"]},
                         ensure_ascii=False, indent=2))
    else:
        report = run_final(args.data_root, args.output_dir, write_rows=not args.no_rows)
        print(json.dumps({"decision": report["decision"], "failed": report["failed"], "final_total_7d":
                          report["final_metrics"]["final_rolling_pooled"]["total_7d"], "runtime": report["runtime"]},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
