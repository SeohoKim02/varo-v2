"""Candidate-only, nested rolling forecast evaluation; no production promotion.

Observed sales during stock-outs are censored observations, not true demand.
Masking them in training is a candidate assumption, not a demand reconstruction.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

VERSION = "rolling-censored-candidate-0.1"
METHODS = ("MA7", "MA28", "EWMA20", "EWMA50", "SEASONAL7")
SECOND_METHODS = ("NAIVE", "SEASONAL7", "MA7", "MA28", "EWMA20", "INTERMITTENT", "CALIBRATED_BLEND")
THIRD_METHODS = ("EWMA20", "MA28", "SEASONAL7", "INTERMITTENT")


def select_third_train_only(history, stockout, horizon=7) -> dict:
    """Conservative inner-validation selector; no outer labels are accepted.

    EWMA20 is the declared simple reference, not an oracle selected on the new
    evaluation. A challenger must reduce pooled inner MAE by 5% and win at least
    two of three inner windows. Missing/stock-out validation labels are excluded
    identically for every method. This is a research rule, not production gating.
    """
    h, censored = np.asarray(history, float), np.asarray(stockout, bool)
    if h.ndim != 1 or h.shape != censored.shape or len(h) < 28 + 3 * horizon or horizon < 1:
        raise ValueError("three aligned inner windows after 28 training days required")
    if np.any(np.isfinite(h) & (h < 0)):
        raise ValueError("signed sales require a declared model")
    origins = list(range(len(h) - 3 * horizon, len(h), horizon))
    records = []
    for method in THIRD_METHODS:
        errors, counts = [], []
        for origin in origins:
            valid = np.isfinite(h[origin:origin+horizon]) & ~censored[origin:origin+horizon]
            prediction = forecast(h[:origin], method, horizon)
            errors.append(float(np.abs(prediction[valid]-h[origin:origin+horizon][valid]).sum()))
            counts.append(int(valid.sum()))
        records.append({"method": method, "errors_by_window": errors, "counts_by_window": counts,
                        "inner_mae": sum(errors)/sum(counts) if sum(counts) else None})
    reference = records[0]
    if not all(reference["counts_by_window"]):
        return {"status": "INSUFFICIENT_UNCENSORED_VALIDATION", "candidates": records,
                "inner_origins": origins, "production_approved": False}
    eligible = [r for r in records[1:] if reference["inner_mae"]-r["inner_mae"] > max(1e-12, .05*reference["inner_mae"])
                and sum(a < b for a, b in zip(r["errors_by_window"], reference["errors_by_window"])) >= 2]
    selected = min(eligible, key=lambda r: (r["inner_mae"], THIRD_METHODS.index(r["method"]))) if eligible else reference
    return {"status": "CANDIDATE", "selected": {**selected, "mask_censored_training": False},
            "candidates": records, "inner_origins": origins, "training_days": len(h),
            "max_inner_label_index": len(h)-1, "reference_method": "EWMA20",
            "reference_retained": selected is reference, "production_approved": False}


def forecast(history, method: str, horizon=7) -> np.ndarray:
    h = np.asarray(history, dtype=float)
    if np.any(np.isfinite(h) & (h < 0)):
        raise ValueError("returns require a separately declared signed-sales model")
    observed = h[np.isfinite(h)]
    if not len(observed):
        raise ValueError("no observed history; missing is not zero")
    if method == "NAIVE":
        return np.full(horizon, observed[-1])
    if method == "INTERMITTENT":
        # Occurrence probability and positive size are separate EWMAs. Missing
        # availability skips the update, whereas an observed zero updates it.
        positive = observed[observed > 0]
        if not len(positive):
            return np.zeros(horizon)
        probability, size = float(observed[0] > 0), float(positive[0])
        for value in observed[1:]:
            probability = .2 * float(value > 0) + .8 * probability
            if value > 0:
                size = .2 * value + .8 * size
        return np.full(horizon, probability * size)
    if method == "CALIBRATED_BLEND":
        # Calibration uses only one-step predictions made strictly before each
        # observed training label. No outer test values enter this estimator.
        actual, predicted = [], []
        for index in range(max(28, len(h) - 21), len(h)):
            if np.isfinite(h[index]) and np.isfinite(h[:index]).any():
                actual.append(h[index])
                predicted.append(.5 * forecast(h[:index], "MA28", 1)[0]
                                 + .5 * forecast(h[:index], "SEASONAL7", 1)[0])
        factor = np.clip(sum(actual) / sum(predicted), .5, 1.5) if sum(predicted) > 0 else 1.
        return factor * (.5 * forecast(h, "MA28", horizon) + .5 * forecast(h, "SEASONAL7", horizon))
    if method in ("MA7", "MA28"):
        window = h[-int(method[2:]):]
        value = np.nanmean(window) if np.isfinite(window).any() else np.mean(observed)
        return np.full(horizon, value)
    if method.startswith("EWMA"):
        alpha = int(method[4:]) / 100
        value = observed[0]
        for x in observed[1:]:
            value = alpha * x + (1 - alpha) * value
        return np.full(horizon, value)
    if method == "SEASONAL7":
        tail = h[-7:]
        fallback = np.mean(observed[-28:])
        tail = np.where(np.isfinite(tail), tail, fallback)
        return np.resize(tail, horizon)
    if method == "OLD_V1":
        from services.demand_forecast_v2 import run_v1_baseline
        row = {"product_id": "evaluation", "stock_qty": 0,
               "avg_daily_sales": float(np.nanmean(h[-30:])),
               "sales_7d": float(np.nansum(h[-7:])) if np.isfinite(h[-7:]).all() else np.nan}
        result = run_v1_baseline(pd.DataFrame([row]))
        return np.full(horizon, float(result.iloc[0]["demand_forecast_7d"]) / 7)
    if method == "OLD_V2":
        from services.demand_forecast_v2 import forecast_v2, make_config
        from services.demand_forecast_router import V2_ROUTES
        return forecast_v2(h.reshape(1, -1), make_config(V2_ROUTES), horizon=horizon).daily[0]
    raise ValueError("unknown forecast method")


def select_train_only(history, stockout, horizon=7) -> dict:
    """All inner labels precede the outer origin. No future array is accepted."""
    h, censored = np.asarray(history, float), np.asarray(stockout, bool)
    if h.shape != censored.shape or len(h) < 42:
        raise ValueError("at least 42 aligned training days required")
    records = []
    for mask in (False, True):
        for method in METHODS:
            error = 0.0
            count = 0
            origins = list(range(max(28, len(h) - 3 * horizon), len(h) - horizon + 1, horizon))
            for origin in origins:
                train = h[:origin].copy()
                if mask:
                    train[censored[:origin]] = np.nan
                if not np.isfinite(train).any():
                    continue
                prediction = forecast(train, method, horizon)
                truth = h[origin:origin + horizon]
                valid = np.isfinite(truth) & ~censored[origin:origin + horizon]
                error += np.abs(prediction[valid] - truth[valid]).sum()
                count += int(valid.sum())
            records.append({"method": method, "mask_censored_training": mask,
                            "inner_mae": float(error / count) if count else None,
                            "inner_scored_days": count, "inner_origins": origins})
    eligible = [r for r in records if r["inner_mae"] is not None]
    if not eligible:
        return {"status": "INSUFFICIENT_UNCENSORED_VALIDATION", "candidates": records}
    selected = min(eligible, key=lambda r: (r["inner_mae"], r["mask_censored_training"], METHODS.index(r["method"])))
    return {"status": "CANDIDATE", "selected": selected, "candidates": records,
            "training_days": len(h), "max_inner_label_index": len(h) - 1}


def select_second_train_only(history, stockout, horizon=7) -> dict:
    """Predeclared finite selector; all scoring labels precede the decision."""
    h, censored = np.asarray(history, float), np.asarray(stockout, bool)
    if h.shape != censored.shape or len(h) < 42:
        raise ValueError("at least 42 aligned training days required")
    origins = list(range(max(28, len(h) - 3 * horizon), len(h) - horizon + 1, horizon))
    records = []
    for mask in (False, True):
        for method in SECOND_METHODS:
            errors = []
            for origin in origins:
                training = h[:origin].copy()
                if mask:
                    training[censored[:origin]] = np.nan
                if not np.isfinite(training).any():
                    continue
                valid = np.isfinite(h[origin:origin+horizon]) & ~censored[origin:origin+horizon]
                prediction = forecast(training, method, horizon)
                errors.extend(np.abs(prediction[valid] - h[origin:origin+horizon][valid]))
            records.append({"method": method, "mask_censored_training": mask,
                            "inner_mae": float(np.mean(errors)) if errors else None,
                            "inner_scored_days": len(errors), "inner_origins": origins})
    eligible = [r for r in records if r["inner_mae"] is not None]
    if not eligible:
        return {"status": "INSUFFICIENT_UNCENSORED_VALIDATION", "candidates": records}
    selected = min(eligible, key=lambda r: (r["inner_mae"], r["mask_censored_training"], SECOND_METHODS.index(r["method"])))
    return {"status": "CANDIDATE", "selected": selected, "candidates": records,
            "training_days": len(h), "max_inner_label_index": len(h)-1}


def evaluate_second_rolling(frame: pd.DataFrame, origins: Sequence[int], horizon=7):
    """Shared origin/row mask for all baselines and both candidate versions."""
    if frame.duplicated(["series", "date"]).any():
        raise ValueError("duplicate series/day; no silent aggregation")
    rows, decisions = [], []
    for key, group in frame.groupby("series", sort=True):
        g = group.copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.set_index("date").sort_index().reindex(pd.date_range(g["date"].min(), g["date"].max()))
        values = pd.to_numeric(g["sales_qty"], errors="coerce").to_numpy(float)
        stockout = pd.to_numeric(g["stockout_hours"], errors="coerce").to_numpy(float)
        censored = ~np.isfinite(stockout) | (stockout > 0)
        for origin in origins:
            if origin < 42 or origin+horizon > len(values):
                raise ValueError("origin lacks declared history or full evaluation window")
            train = values[:origin]
            old = select_train_only(train, censored[:origin], horizon)
            new = select_second_train_only(train, censored[:origin], horizon)
            common = old["status"] == new["status"] == "CANDIDATE"
            decisions.append({"series": str(key), "origin": origin, "common_eligible": common,
                              "exclusion_reason": None if common else "NO_COMMON_UNCENSORED_INNER_LABELS",
                              "cutoff": str(g.index[origin-1].date()), "prior": old, "second": new})
            if not common:
                continue
            predictions = {m: forecast(train, m, horizon) for m in ("OLD_V1", "OLD_V2", *SECOND_METHODS)}
            for label, choice in (("PRIOR_CANDIDATE", old), ("SECOND_CANDIDATE", new)):
                history = train.copy()
                if choice["selected"]["mask_censored_training"]:
                    history[censored[:origin]] = np.nan
                predictions[label] = forecast(history, choice["selected"]["method"], horizon)
            for method, prediction in predictions.items():
                for step, estimate in enumerate(prediction):
                    idx = origin+step
                    rows.append({"series": str(key), "origin": origin, "date": str(g.index[idx].date()),
                                 "method": method, "observed_sales": float(values[idx]) if np.isfinite(values[idx]) else None,
                                 "forecast": float(estimate), "target_observed": bool(np.isfinite(values[idx])),
                                 "target_uncensored": bool(stockout[idx] == 0), "true_demand_observed": False,
                                 "stockout_hours": float(stockout[idx]) if np.isfinite(stockout[idx]) else None,
                                 "evidence_class": "OBSERVED_SALES_EVALUATION"})
    return pd.DataFrame(rows), decisions


def evaluate_rolling(frame: pd.DataFrame, origins: Sequence[int], horizon=7) -> tuple[pd.DataFrame, list[dict]]:
    required = {"series", "date", "sales_qty", "stockout_hours"}
    if not required <= set(frame):
        raise ValueError("series/date/sales_qty/stockout_hours required")
    if frame.duplicated(["series", "date"]).any():
        raise ValueError("duplicate series/day; no silent aggregation")
    rows, decisions = [], []
    for key, group in frame.groupby("series", sort=True):
        g = group.copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.set_index("date").sort_index().reindex(pd.date_range(g["date"].min(), g["date"].max()))
        values = pd.to_numeric(g["sales_qty"], errors="coerce").to_numpy(float)
        stockout = pd.to_numeric(g["stockout_hours"], errors="coerce").to_numpy(float)
        # Unknown availability is not proof that sales were uncensored.
        censored = ~np.isfinite(stockout) | (stockout > 0)
        for origin in origins:
            if origin < 42 or origin + horizon > len(values):
                raise ValueError("origin lacks declared history or full evaluation window")
            train = values[:origin].copy()
            selection = select_train_only(train, censored[:origin], horizon)
            decisions.append({"series": key, "origin": origin, "cutoff": str(g.index[origin - 1].date()), **selection})
            if selection["status"] != "CANDIDATE":
                continue
            chosen = selection["selected"]
            candidate_history = train.copy()
            if chosen["mask_censored_training"]:
                candidate_history[censored[:origin]] = np.nan
            predictions = {m: forecast(train, m, horizon) for m in ("MA28", "SEASONAL7", "OLD_V1", "OLD_V2")}
            predictions["NEW_CANDIDATE"] = forecast(candidate_history, chosen["method"], horizon)
            for method, prediction in predictions.items():
                for step, estimate in enumerate(prediction):
                    idx = origin + step
                    truth = values[idx]
                    rows.append({"series": key, "origin": origin, "date": str(g.index[idx].date()), "method": method,
                                 "observed_sales": float(truth) if np.isfinite(truth) else None,
                                 "forecast": float(estimate), "stockout_hours": float(stockout[idx]) if np.isfinite(stockout[idx]) else None,
                                 "target_observed": bool(np.isfinite(truth)), "target_uncensored": bool(stockout[idx] == 0),
                                 "true_demand_observed": False, "evidence_class": "OBSERVED_SALES_EVALUATION"})
    return pd.DataFrame(rows), decisions


def summarize(rows: pd.DataFrame) -> list[dict]:
    summaries = []
    for scope in ("ALL_OBSERVED_SALES", "CONFIRMED_NO_STOCKOUT_SALES"):
        part = rows[rows["target_observed"]]
        if scope == "CONFIRMED_NO_STOCKOUT_SALES":
            part = part[part["target_uncensored"]]
        for method, g in part.groupby("method", sort=True):
            error = g["forecast"] - g["observed_sales"]
            denom = g["observed_sales"].abs().sum()
            summaries.append({"scope": scope, "method": method, "rows": len(g), "series": g["series"].nunique(),
                              "origins": g["origin"].nunique(), "wape": float(error.abs().sum() / denom) if denom else None,
                              "mae": float(error.abs().mean()), "bias": float(error.sum() / denom) if denom else None,
                              "independent_experiment_count": None, "actual_business_effect": "NOT_OBSERVED"})
    return summaries


def load_small_canonical(path: Path, series_count=16) -> tuple[pd.DataFrame, dict]:
    """Deterministic first physical series, then complete their history by streaming.

    This convenience sample is not a population-wide generalization experiment.
    """
    import pyarrow.parquet as pq
    file = pq.ParquetFile(path)
    selected = None
    parts = []
    columns = ["date", "location_id", "product_id", "sales_qty", "stockout_hours", "unit", "unit_status"]
    for batch in file.iter_batches(batch_size=65536, columns=columns):
        f = batch.to_pandas()
        f["series"] = f["location_id"].astype(str) + ":" + f["product_id"].astype(str)
        if selected is None:
            selected = list(f["series"].drop_duplicates().head(series_count))
        parts.append(f[f["series"].isin(selected)].copy())
    data = pd.concat(parts, ignore_index=True)
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return data, {"source": str(path), "source_sha256": hasher.hexdigest(), "series": selected,
                  "rows": len(data), "sample_rule": "first physical series, no performance-based selection",
                  "record_type": "REAL_SALES_WITH_STOCKOUT_FLAGS", "true_demand": "UNOBSERVED",
                  "source_units": sorted(data["unit"].dropna().unique().tolist()),
                  "source_unit_status": sorted(data["unit_status"].dropna().unique().tolist()),
                  "external_generalization_verdict": "EXISTING_FAIL_UNCHANGED"}
