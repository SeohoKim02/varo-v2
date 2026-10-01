"""Walmart M5 checks of the production demand-forecast router (``services.demand_forecast_router``).

Two phases on the 13 cutoffs of ``services.m5_forecast_validation`` (split exactly as in the v2 validation):

    mostly-zero  development origins 1-10 only; the panel is truncated after the last development target day before
                 anything is scored. The pre-declared MOSTLY_ZERO_CANDIDATES are scored on the series that reach the
                 router's mostly_zero check, and the pre-declared MOSTLY_ZERO_RULE picks the production policy.
    production   the production router itself on real inventory/history inputs: V1 through its production call path,
                 the long daily history (including 7 days after the cutoff, which the router must exclude) through
                 ``route_demand_forecast``. Checks routing, exact equality of production V2 and research V2, and
                 reports the routed 7-day metrics next to V1 and V2. Nothing here selects or changes a parameter.

Read-only on raw and processed data; writes only the ``m5_forecast_router_*`` files.

    python -m services.m5_forecast_router_validation --phase mostly-zero
    python -m services.m5_forecast_router_validation --phase production
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from services import demand_forecast_router as router
from services import demand_forecast_v2 as core
from services import m5_forecast_v2_validation as v2val
from services.analysis_pipeline import PipelineResult, _run_inventory_analysis, _Runner
from services.legacy_adapters.data_adapter import prepare_legacy_data
from services.m5_forecast_validation import (
    HORIZON, OPERATIONAL_HORIZON, ORIGIN_COUNT, ORIGIN_STEP, VARO_INPUT_COLUMNS, M5Panel, _clean, _git_commit, peak_memory_mb,
    run_varo_forecast, varo_inputs)
from services.real_data_adapters import DATA_ROOT

EVALUATION_VERSION = "1.0.0"
V1, V2, ROUTED = v2val.V1, v2val.V2, "varo_router_production"
MOSTLY_ZERO = core.DEMAND_TYPES.index("mostly_zero")
FUTURE_DAYS = OPERATIONAL_HORIZON   # history rows after the cutoff handed to the router (it must never read them)

RESULT_FILES = {
    "mostly_zero_selection": "m5_forecast_router_mostly_zero_selection.json",
    "mostly_zero_metrics": "m5_forecast_router_mostly_zero_development.csv",
    "production_path": "m5_forecast_router_production_path.json",
    "production_metrics": "m5_forecast_router_production_path.csv",
}

# ---------------------------------------------------------------- pre-declared mostly_zero candidates and rule
# Declared before any candidate was scored. Declared order = preference on ties (simplest / most conservative first).

MOSTLY_ZERO_CANDIDATES: dict[str, str] = {
    "v1": "V1 production forecast unchanged (FORECAST_V1_BASELINE)",
    "v2_floor_v1": "safety floor: per series max(V2, V1) - the V2 forecast, never below V1",
    "blend_v1_v2": "conservative blend: 0.5 x V1 + 0.5 x V2 (equal weights, nothing fitted)",
    "sparse_mean_364": "simple sparse fallback: mean daily sales over the 364-day profile window since the first sale, flat",
    "v2": "V2 Core as frozen (mostly_zero route ma28 + shrunk_8w)",
}
MOSTLY_ZERO_RULE: dict[str, Any] = {
    "population": ("development series of demand type mostly_zero that pass every earlier router check (sold at least "
                   f"{router.MIN_SALES_AGE_DAYS} days before the cutoff, last {router.RECENT_OBSERVED_DAYS} days observed, "
                   "no negative value), scored with the M5 evaluation eligibility"),
    "metric": "development-pooled total_7d WAPE and bias_pct (origins 1-10)",
    "admissible": "|bias_pct| <= max(|V1 bias_pct|, bias_floor): the frozen stage-1 bias rule of the v2 selection",
    "bias_floor": v2val.SELECTION_RULE["stage_1_level"]["bias_floor"],
    "default": "v1",
    "switch": ("an admissible candidate replaces the default only if its WAPE is at least min_relative_improvement below "
               "V1's (the frozen promotion gate's primary threshold); among those the lowest WAPE wins"),
    "min_relative_improvement": v2val.PROMOTION_GATE["primary"]["min_relative_improvement"],
    "tie_tolerance": v2val.SELECTION_RULE["stage_1_level"]["tie_tolerance"],
    "tie_break": "within (1 + tie_tolerance) x best WAPE, the first candidate in declared order",
    "final_origins": "never read by this phase",
}


def rule_signature() -> str:
    return v2val.signature({"candidates": MOSTLY_ZERO_CANDIDATES, "rule": MOSTLY_ZERO_RULE})


def router_population(history: np.ndarray, profile: core.DemandProfile) -> np.ndarray:
    """Series that reach the router's mostly_zero check (every earlier check passed) and are mostly_zero."""
    reasons = router.assess_matrix(history, profile)
    earlier = np.isin(reasons, [router.REASON_NO_SALES, router.REASON_COLD_START, router.REASON_INSUFFICIENT])
    return (profile.demand_type == MOSTLY_ZERO) & ~earlier & ~(history < 0).any(axis=1)


def mostly_zero_candidates(od: v2val.OriginData, v1_daily: np.ndarray, v1_total: np.ndarray,
                           result: core.ForecastV2Result, horizon: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    v2_total = np.round(result.aggregate_7d, 1)
    floor = v2_total >= v1_total
    block, start = core.recent_block(od.history, result.profile.first_sale)
    sparse = core.window_mean(block, start, core.PROFILE_WINDOW)
    return {
        "v1": (v1_daily, v1_total),
        "v2_floor_v1": (np.where(floor[:, None], result.daily, v1_daily), np.where(floor, v2_total, v1_total)),
        "blend_v1_v2": (0.5 * (v1_daily + result.daily), np.round(0.5 * (v1_total + v2_total), 1)),
        "sparse_mean_364": (v2val.flat(sparse, horizon), np.round(OPERATIONAL_HORIZON * sparse, 1)),
        "v2": (result.daily, v2_total),
    }


def select_mostly_zero(metrics: pd.DataFrame, rule: Mapping[str, Any] = MOSTLY_ZERO_RULE) -> dict[str, Any]:
    rows = metrics.set_index("method")
    v1_wape, v1_bias = float(rows.loc["v1", "wape"]), float(rows.loc["v1", "bias_pct"])
    bound = max(abs(v1_bias), rule["bias_floor"])
    options = [{"candidate": name, "wape": float(rows.loc[name, "wape"]), "bias_pct": float(rows.loc[name, "bias_pct"]),
                "wape_vs_v1_relative": float(rows.loc[name, "wape"]) / v1_wape - 1.0,
                "admissible": bool(abs(float(rows.loc[name, "bias_pct"])) <= bound + 1e-12)}
               for name in MOSTLY_ZERO_CANDIDATES if name in rows.index]
    qualifying = [o for o in options if o["admissible"] and o["candidate"] != rule["default"]
                  and o["wape"] <= v1_wape * (1 - rule["min_relative_improvement"])]
    if qualifying:
        best = min(o["wape"] for o in qualifying)
        chosen = next(o["candidate"] for o in qualifying if o["wape"] <= best * (1 + rule["tie_tolerance"]))
        reason = "admissible and at least min_relative_improvement better than V1"
    else:
        chosen, reason = rule["default"], "no admissible candidate improves V1 by min_relative_improvement"
    return {"chosen": chosen, "reason": reason, "bias_bound": bound, "v1_wape": v1_wape, "v1_bias_pct": v1_bias, "options": options}


def v1_forecast(history: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    out = run_varo_forecast(varo_inputs(history), VARO_INPUT_COLUMNS["varo_production"])
    return v2val.flat(out["demand_forecast_daily"].to_numpy(dtype=np.float64), horizon), out["demand_forecast_7d"].to_numpy(dtype=np.float64)


def _segment(od: v2val.OriginData, name: str, codes: np.ndarray, labels: Sequence[str]) -> None:
    od.series_dims[name] = (np.asarray(codes, dtype=np.int64), tuple(labels))


def run_mostly_zero_selection(data_root: Path | str | None = DATA_ROOT, output_dir: Path | str | None = None, *,
                              horizon: int = HORIZON, origin_count: int = ORIGIN_COUNT, origin_step: int = ORIGIN_STEP,
                              panel: M5Panel | None = None) -> dict[str, Any]:
    """Score the declared mostly_zero candidates on development origins only and apply the declared rule."""
    started = time.perf_counter()
    inputs = v2val.load_inputs(data_root, panel)
    output = Path(output_dir) if output_dir else inputs.folder / "results"
    output.mkdir(parents=True, exist_ok=True)
    protocol = v2val.split_protocol(len(inputs.panel.dates), horizon, origin_count, origin_step, v2val.DEV_COUNT)
    dev_panel = v2val.truncate_panel(inputs.panel, protocol["development_last_day"])   # final days do not exist here
    groups = v2val.pool_groups(dev_panel.series)
    config = router.v2_config()
    sums, population_sizes = [], {}
    for spec in protocol["development"]:
        od = v2val.build_origin(dev_panel, spec, horizon)
        v1_daily, v1_total = v1_forecast(od.history, horizon)
        result = core.forecast_v2(od.history, config, horizon, groups)
        population = router_population(od.history, result.profile)
        _segment(od, "router_mostly_zero", np.where(population, 0, 1), ("mostly_zero_policy_population", "other"))
        population_sizes[spec["origin"]] = int((population & od.week_valid).sum())
        for name, (daily, total) in mostly_zero_candidates(od, v1_daily, v1_total, result, horizon).items():
            sums += v2val.score(od, name, daily, total, dims=("router_mostly_zero",), points=False)
    dev_ids = [o["origin"] for o in protocol["development"]]
    frame = pd.concat(sums, ignore_index=True)
    metrics = v2val.pooled(frame, {"development_pooled": dev_ids, **{f"development_origin_{k:02d}": [k] for k in dev_ids}})
    metrics = metrics[metrics["segment"] == "mostly_zero_policy_population"].reset_index(drop=True)
    pooled_7d = metrics[(metrics["scope"] == "development_pooled") & (metrics["window"] == "total_7d")]
    decision = select_mostly_zero(pooled_7d)
    columns = ["scope", "origins", "window", "segment", "method", "n", "scored", "actual", "forecast", "wape", "mae", "rmse",
               "bias_pct", "over_units", "under_units", "fill_rate_proxy", "excess_units_ratio"]
    metrics[[c for c in columns if c in metrics.columns]].to_csv(output / RESULT_FILES["mostly_zero_metrics"], index=False,
                                                                 encoding="utf-8-sig")
    daily_7 = metrics[(metrics["scope"] == "development_pooled") & (metrics["window"] == "daily_h1_7")]
    report = {
        "evaluation": {"name": "Varo forecast router - mostly_zero policy selection on M5 development origins",
                       "version": EVALUATION_VERSION, "phase": "mostly_zero_selection", "generated_at": v2val._now(),
                       "git_commit": _git_commit(), "command": "python -m services.m5_forecast_router_validation --phase mostly-zero"},
        "candidates": MOSTLY_ZERO_CANDIDATES, "rule": MOSTLY_ZERO_RULE, "rule_signature": rule_signature(),
        "v2_config_signature": router.V2_CONFIG_SIGNATURE,
        "development_last_readable_day": f"d_{protocol['development_last_day'] + 1}",
        "population_series_weeks_per_origin": population_sizes,
        "decision": decision,
        "development_total_7d": {r["method"]: {k: r[k] for k in ("wape", "mae", "bias_pct", "over_units", "under_units",
                                                                  "fill_rate_proxy", "excess_units_ratio", "actual", "forecast")}
                                 for _, r in pooled_7d.iterrows()},
        "development_daily_h1_7": {r["method"]: {k: r[k] for k in ("wape", "bias_pct")} for _, r in daily_7.iterrows()},
        "router_constant_matches": decision["chosen"] == router.MOSTLY_ZERO_POLICY,
        "runtime": {"total_seconds": round(time.perf_counter() - started, 3), "peak_memory_mb": peak_memory_mb()},
    }
    (output / RESULT_FILES["mostly_zero_selection"]).write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2),
                                                                encoding="utf-8")
    return _clean(report)


# ---------------------------------------------------------------- production path


def production_inputs(panel: M5Panel, cutoff: int, future_days: int = FUTURE_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inventory rows (production aggregate fields from the last 30 days) and the long daily history.

    The history holds every day from d_1 to ``future_days`` past the cutoff; the inventory snapshot_date is the cutoff
    date, so the router has to drop the future rows itself.
    """
    history = panel.sales[:, :cutoff + 1]
    inventory = varo_inputs(history)
    inventory.insert(0, "store_id", panel.series["store_id"].to_numpy())
    inventory.insert(1, "product_id", panel.series["item_id"].to_numpy())
    inventory.insert(2, "category", panel.series["category"].to_numpy())
    inventory["product_name"] = inventory["product_id"]
    inventory["snapshot_date"] = str(panel.dates[cutoff])
    end = min(cutoff + 1 + future_days, panel.sales.shape[1])
    n, days = len(panel.series), end
    store = pd.Categorical(panel.series["store_id"].astype(str))
    item = pd.Categorical(panel.series["item_id"].astype(str))
    long = pd.DataFrame({
        "store_id": pd.Categorical.from_codes(np.repeat(store.codes, days), store.categories),
        "product_id": pd.Categorical.from_codes(np.repeat(item.codes, days), item.categories),
        "date": np.tile(pd.to_datetime(pd.Series(panel.dates[:end])).to_numpy(), n),
        "quantity": panel.sales[:, :end].ravel(),
    })
    keep = ~np.isnan(long["quantity"].to_numpy())     # a missing cell has no row (the contract's missing day)
    return inventory, long[keep].reset_index(drop=True) if not keep.all() else long


def production_v1(inventory: pd.DataFrame) -> pd.DataFrame:
    """V1 through its production call path (prepare_legacy_data -> allowlisted analyze_demand_forecast)."""
    from services.dqn_guard import strip_dqn_columns
    from services.legacy_adapters.loader import load_legacy_module
    prepared = prepare_legacy_data({"inventory": inventory.copy()})["inventory"]
    return load_legacy_module("demand_forecast_analyzer").analyze_demand_forecast(strip_dqn_columns(prepared))


def production_origin(panel: M5Panel, od: v2val.OriginData, groups: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    """Route one origin through the production router and compare it with V1 and the research V2 on identical input."""
    cutoff = od.spec["cutoff"]
    inventory, long = production_inputs(panel, cutoff)
    started = time.perf_counter()
    v1_out = production_v1(inventory)
    routed, diagnostics = router.route_demand_forecast(v1_out, long)
    seconds = time.perf_counter() - started
    del long
    research = core.forecast_v2(od.history, config, OPERATIONAL_HORIZON, groups)
    reason = routed["demand_forecast_reason"].to_numpy(dtype=object)
    is_v2 = reason == router.REASON_V2
    v1_total = v1_out["demand_forecast_7d"].to_numpy(dtype=np.float64)
    total = routed["demand_forecast_7d"].to_numpy(dtype=np.float64)
    daily = routed[list(router.DAILY_COLUMNS)].to_numpy(dtype=np.float64)
    research_total = np.round(research.aggregate_7d, 1)
    labels = np.array([f"V2:{a}+{b}" for a, b in zip(research.level_method, research.weekday_method)], dtype=object)
    expected_reason = router.assess_matrix(od.history, research.profile)
    research_v1 = run_varo_forecast(varo_inputs(od.history), VARO_INPUT_COLUMNS["varo_production"])["demand_forecast_7d"]
    checks = {
        "production_v1_equal_research_v1": bool(np.array_equal(v1_total, research_v1.to_numpy(dtype=np.float64))),
        "v2_rows_7d_equal_research": bool(np.array_equal(total[is_v2], research_total[is_v2])),
        "v2_rows_daily_vector_equal_research": bool(np.array_equal(daily[is_v2], research.daily[is_v2, :OPERATIONAL_HORIZON])),
        "v2_rows_method_equal_research_route": bool((routed["demand_forecast_method"].to_numpy(dtype=object)[is_v2] == labels[is_v2]).all()),
        "v1_rows_7d_equal_v1": bool(np.array_equal(total[~is_v2], v1_total[~is_v2])),
        "reasons_equal_router_assessment_of_research_history": bool((reason == expected_reason).all()),
        "future_rows_excluded": diagnostics.get("future_rows_excluded"),
        "future_rows_expected": int(len(od.history) * min(FUTURE_DAYS, panel.sales.shape[1] - cutoff - 1)),
        "cutoff": diagnostics.get("cutoffs"),
        "cutoff_expected": [str(panel.dates[cutoff])],
        "no_errors": not diagnostics.get("errors"),
    }
    checks["future_rows_all_excluded"] = checks["future_rows_excluded"] == checks["future_rows_expected"]
    codes = np.array([list(router.REASON_CODES).index(r) for r in reason])
    v1_daily = v2val.flat(v1_out["demand_forecast_daily"].to_numpy(dtype=np.float64), OPERATIONAL_HORIZON)
    return {"reason": reason, "codes": codes, "diagnostics": diagnostics, "checks": checks, "seconds": seconds,
            "forecasts": {V1: (v1_daily, v1_total), V2: (research.daily[:, :OPERATIONAL_HORIZON], research_total), ROUTED: (daily, total)},
            "demand_type": research.profile.demand_type}


def run_production_path(data_root: Path | str | None = DATA_ROOT, output_dir: Path | str | None = None, *,
                        horizon: int = HORIZON, origin_count: int = ORIGIN_COUNT, origin_step: int = ORIGIN_STEP,
                        panel: M5Panel | None = None, origins: str = "all", wiring_series: int = 300) -> dict[str, Any]:
    """Production router on every origin (dev + final by default); per-origin checks and pooled routed metrics."""
    started = time.perf_counter()
    inputs = v2val.load_inputs(data_root, panel)
    output = Path(output_dir) if output_dir else inputs.folder / "results"
    output.mkdir(parents=True, exist_ok=True)
    protocol = v2val.split_protocol(len(inputs.panel.dates), horizon, origin_count, origin_step, v2val.DEV_COUNT)
    specs = {"all": protocol["development"] + protocol["final"], "final": protocol["final"],
             "development": protocol["development"]}[origins]
    groups = v2val.pool_groups(inputs.panel.series)
    config = router.v2_config()
    reason_labels = list(router.REASON_CODES)
    sums, per_origin = [], []
    for spec in specs:
        od = v2val.build_origin(inputs.panel, spec, horizon)
        result = production_origin(inputs.panel, od, groups, config)
        _segment(od, "router_reason", result["codes"], reason_labels)
        for method, (daily, total) in result["forecasts"].items():
            sums += v2val.score(od, method, daily, total, dims=("overall", "frequency_class", "router_reason"), points=False)
        scored = od.week_valid
        per_origin.append({"origin": spec["origin"], "cutoff_date": str(inputs.panel.dates[spec["cutoff"]]),
                           "phase": "final" if spec in protocol["final"] else "development",
                           "router_seconds": round(result["seconds"], 3), "checks": result["checks"],
                           "counts_by_reason": {r: int(c) for r, c in pd.Series(result["reason"]).value_counts().items()},
                           "scored_counts_by_reason": {r: int(c) for r, c in pd.Series(result["reason"][scored]).value_counts().items()},
                           "v2_rows_by_demand_type": {core.DEMAND_TYPES[t]: int(c) for t, c in zip(*np.unique(
                               result["demand_type"][result["reason"] == router.REASON_V2], return_counts=True))}})
        del result
    frame = pd.concat(sums, ignore_index=True)
    ids = {p["origin"]: p["phase"] for p in per_origin}
    scopes = {**({"development_pooled": [k for k, v in ids.items() if v == "development"]} if "development" in ids.values() else {}),
              **({"final_rolling_pooled": [k for k, v in ids.items() if v == "final"]} if "final" in ids.values() else {}),
              **{f"origin_{k:02d}": [k] for k in ids}}
    metrics = v2val.pooled(frame, scopes)
    columns = ["scope", "origins", "window", "dimension", "segment", "method", "n", "scored", "actual", "forecast", "wape", "mae",
               "rmse", "bias_pct", "over_units", "under_units", "fill_rate_proxy", "excess_units_ratio"]
    metrics[[c for c in columns if c in metrics.columns]].to_csv(output / RESULT_FILES["production_metrics"], index=False,
                                                                 encoding="utf-8-sig")

    def table(scope: str, dimension: str = "overall") -> dict[str, Any]:
        rows = metrics[(metrics["scope"] == scope) & (metrics["window"] == "total_7d") & (metrics["dimension"] == dimension)]
        return {seg: {r["method"]: {k: r[k] for k in ("wape", "mae", "bias_pct", "over_units", "under_units", "scored")}
                      for _, r in part.iterrows()} for seg, part in rows.groupby("segment", sort=False)}

    wins = []
    for origin in ids:
        rows = metrics[(metrics["scope"] == f"origin_{origin:02d}") & (metrics["window"] == "total_7d") & (metrics["dimension"] == "overall")]
        wape = rows.set_index("method")["wape"]
        wins.append({"origin": origin, "phase": ids[origin], "v1": float(wape[V1]), "v2": float(wape[V2]),
                     "router": float(wape[ROUTED]), "router_better_than_v1": bool(wape[ROUTED] < wape[V1])})
    all_checks = [p["checks"] for p in per_origin]
    wiring = pipeline_wiring_check(inputs.panel, protocol["holdout"], config, min(wiring_series, len(inputs.panel.series)))
    report = {
        "evaluation": {"name": "Varo forecast router - production-path simulation on M5", "version": EVALUATION_VERSION,
                       "phase": "production_path", "generated_at": v2val._now(), "git_commit": _git_commit(),
                       "command": "python -m services.m5_forecast_router_validation --phase production"},
        "router_policy": router.ROUTER_POLICY,
        "inputs": {"inventory": "one row per M5 series: store_id, product_id, category, sales_7d, avg_daily_sales, sales_30d, "
                                "demand_std (last 30 days), snapshot_date = cutoff date",
                   "daily_sales_history": f"long rows d_1 .. cutoff + {FUTURE_DAYS} days (future rows must be excluded by the router)",
                   "v1": "prepare_legacy_data -> analyze_demand_forecast (production call path)",
                   "router": "services.demand_forecast_router.route_demand_forecast",
                   "research_v2": "services.demand_forecast_v2.forecast_v2 on the (series x day) matrix <= cutoff, "
                                  "groups = store x category (m5_forecast_v2_validation.pool_groups)"},
        "all_checks_pass": wiring["pass"] and all(all(v for v in c.values() if isinstance(v, bool)) and c["future_rows_all_excluded"]
                                                  and c["cutoff"] == c["cutoff_expected"] for c in all_checks),
        "pipeline_wiring_check": wiring,
        "per_origin": per_origin,
        "per_origin_total_7d_wape": wins,
        "total_7d": {scope: table(scope) for scope in scopes if not scope.startswith("origin_")},
        "total_7d_by_router_reason": {scope: table(scope, "router_reason") for scope in scopes if not scope.startswith("origin_")},
        "total_7d_by_demand_type": {scope: table(scope, "frequency_class") for scope in scopes if not scope.startswith("origin_")},
        "note": ("Reporting only: the router policy was fixed before this run (mostly_zero policy from development origins). "
                 "V1/V2 7-day totals are rounded to 0.1 as the production scalar; the routed daily vector is the production "
                 "demand_forecast_d1..d7 (V1 rows: flat demand_forecast_7d / 7)."),
        "runtime": {"total_seconds": round(time.perf_counter() - started, 3), "peak_memory_mb": peak_memory_mb()},
    }
    (output / RESULT_FILES["production_path"]).write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return _clean(report)


INJECTIONS = (("missing_daily_history", 20), ("invalid_history:repeated_day", 5), ("invalid_history:nan_quantity", 5),
              ("negative_sales", 5), ("insufficient_history", 5))


def pipeline_wiring_check(panel: M5Panel, spec: Mapping[str, Any], config: Mapping[str, Any], size: int = 300) -> dict[str, Any]:
    """The first ``size`` M5 series through the analysis pipeline's own inventory step, with injected anomalies.

    Anomalies are injected into series that are V2-eligible when clean, so each expected reason is exact. Untouched
    V2 rows must keep the research 7-day total (the level does not depend on the pooled weekday shape).
    """
    cutoff = spec["cutoff"]
    rows = list(range(size))
    sub = M5Panel(panel.sales[rows], panel.price[rows], panel.dates, panel.series.iloc[rows].reset_index(drop=True),
                  panel.event_label, panel.snap[rows], panel.canonical_rows)
    history = sub.sales[:, :cutoff + 1]
    research = core.forecast_v2(history, config, OPERATIONAL_HORIZON)
    clean_reason = router.assess_matrix(history, research.profile)
    inventory, long = production_inputs(sub, cutoff)
    eligible = [i for i in range(size) if clean_reason[i] == router.REASON_V2]
    expected = clean_reason.copy()
    product = long["product_id"].astype(str).to_numpy()
    day = long["date"].to_numpy().astype("datetime64[D]")
    cut = np.datetime64(str(sub.dates[cutoff]), "D")
    drop = np.zeros(len(long), dtype=bool)
    extra, cursor = [], 0
    for label, count in INJECTIONS:
        for i in eligible[cursor:cursor + count]:
            mine = product == sub.series["item_id"].iloc[i]
            expected[i] = label.split(":")[0]
            if label == "missing_daily_history":
                drop |= mine
            elif label == "invalid_history:repeated_day":
                extra.append(long[mine & (day == cut - 10)])
            elif label == "invalid_history:nan_quantity":
                long.loc[mine & (day == cut - 5), "quantity"] = np.nan
            elif label == "negative_sales":
                long.loc[mine & (day == cut - 5), "quantity"] = -1.0
            else:
                drop |= mine & (day > cut - 3) & (day <= cut)
        cursor += count
    injected = pd.concat([long[~drop], *extra], ignore_index=True)
    result = PipelineResult()
    analyzed, summaries = _run_inventory_analysis(_Runner(result), prepare_legacy_data({"inventory": inventory})["inventory"],
                                                  daily_sales_history=injected)
    analyzed = analyzed.set_index("product_id").loc[inventory["product_id"]].reset_index()
    reason = analyzed["demand_forecast_reason"].to_numpy(dtype=object)
    untouched_v2 = (expected == router.REASON_V2) & (reason == router.REASON_V2)
    v1_total = production_v1(inventory)["demand_forecast_7d"].to_numpy(dtype=np.float64)
    total = analyzed["demand_forecast_7d"].to_numpy(dtype=np.float64)
    checks = {
        "reasons_equal_expected": bool((reason == expected).all()),
        "untouched_v2_rows_7d_equal_research": bool(np.array_equal(total[untouched_v2], np.round(research.aggregate_7d, 1)[untouched_v2])),
        "fallback_rows_7d_equal_v1": bool(np.array_equal(total[reason != router.REASON_V2], v1_total[reason != router.REASON_V2])),
        "function_label_unchanged": summaries["demand_forecast"]["function"] == "demand_forecast_analyzer.analyze_demand_forecast",
        "router_listed_as_connected_iff_v2_rows": ("services.demand_forecast_router.route_demand_forecast" in result.connected_algorithms)
                                                  == bool((reason == router.REASON_V2).any()),
        "no_technical_errors": not result.warnings,
    }
    return {"series": size, "origin": spec["origin"], "injections": dict(INJECTIONS), "checks": checks, "pass": all(checks.values()),
            "counts_by_reason": {r: int(c) for r, c in pd.Series(reason).value_counts().items()},
            "router_diagnostics": summaries["demand_forecast"].get("forecast_router")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=["mostly-zero", "production"], required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--origins", choices=["all", "final", "development"], default="all")
    args = parser.parse_args()
    if args.phase == "mostly-zero":
        report = run_mostly_zero_selection(args.data_root, args.output_dir)
        print(json.dumps({"decision": report["decision"], "development_total_7d": report["development_total_7d"],
                          "runtime": report["runtime"]}, ensure_ascii=False, indent=2))
    else:
        report = run_production_path(args.data_root, args.output_dir, origins=args.origins)
        print(json.dumps({"all_checks_pass": report["all_checks_pass"], "total_7d": report["total_7d"],
                          "per_origin_total_7d_wape": report["per_origin_total_7d_wape"], "runtime": report["runtime"]},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
