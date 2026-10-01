"""Production demand-forecast router: the frozen V2 Core where a daily sales history supports it, V1 everywhere else.

``services.analysis_pipeline._run_inventory_analysis`` runs the unchanged V1 step
(``demand_forecast_analyzer.analyze_demand_forecast``) and hands its output frame to :func:`route_demand_forecast`.
The router never recomputes V1: V1's output is the fallback of every row. The router adds the provenance fields and
replaces the forecast fields of the rows whose daily history passes every eligibility check with the V2 Core forecast
(``services.demand_forecast_v2`` with the configuration that passed the frozen M5 promotion gate).

Daily sales history contract (optional; ``uploaded_data["daily_sales_history"]``, long format, one row per day):
    store_id    inventory store key (``location_id`` accepted, as in canonical demand_series)
    product_id  inventory product key
    date        sales day
    quantity    units sold that day (``sales_qty`` accepted); 0 is an observed zero sale
- A day without a row is a missing observation, never a zero sale.
- One row per (store_id, product_id, date). A repeated day is neither summed nor picked: the series is invalid.
- Quantities must be finite numbers. A negative value (a return) is kept as data but routes the series to V1.
- Forecast cutoff: ``as_of`` when given, else the inventory row's ``snapshot_date``, else the latest history date.
  History rows dated after the cutoff are future data and are excluded before anything is computed.
- Identifiers are matched as the workbook loader writes them (stripped text, integral numbers without ".0").
  An inventory key that occurs on several inventory rows cannot receive one series' demand: those rows stay V1.

Output (every row, V1 and V2 alike; the V1 keys keep their meaning and unit):
    demand_forecast_7d        forecast units over the next 7 days (a 7-day total), rounded to 0.1
    demand_forecast_daily     demand_forecast_7d / 7 rounded to 0.01 (flat daily rate)
    demand_forecast_upper/lower, demand_stockout_days, demand_risk_score, demand_forecast_score
                              V1's own formulas on that daily rate (V2 rows are recomputed with V1's helpers)
    demand_trend              V1's sales_7d vs avg_daily_sales label (an input description, not the forecast)
    demand_forecast_method    V1: WMA / SMA / NAIVE; V2: "V2:<level method>+<weekday method>"
    demand_forecast_version   "v1" | "v2"
    demand_forecast_reason    why that version was used (REASON_CODES)
    demand_forecast_fallback_reason  the same code for V1 rows, missing (NA) for V2 rows
    demand_forecast_d1..d7    daily forecast vector; V2: the weekday-shaped Core vector (sums to the unrounded
                              7-day total), V1: flat demand_forecast_7d / 7
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from services import demand_forecast_v2 as core

logger = logging.getLogger(__name__)

ROUTER_VERSION = "1.0.0"
DAILY_SALES_HISTORY_KEY = "daily_sales_history"
HORIZON = core.OPERATIONAL_HORIZON
DAILY_COLUMNS = tuple(f"demand_forecast_d{h}" for h in range(1, HORIZON + 1))
ROUTER_COLUMNS = ("demand_forecast_version", "demand_forecast_reason", "demand_forecast_fallback_reason", *DAILY_COLUMNS)
VERSION_V1, VERSION_V2 = "v1", "v2"

# The only V2 configuration allowed in production: the routes frozen by the M5 development phase that passed the
# pre-registered promotion gate (m5_forecast_v2_promotion_gate.json, PASS 12/12). History-only and dataset-agnostic.
V2_ROUTES: dict[str, dict[str, str]] = {
    "high_frequency": {"level": "ewma_w0.4", "weekday": "pooled_8w"},
    "medium_frequency": {"level": "ewma_w0.4", "weekday": "pooled_8w"},
    "intermittent": {"level": "ewma_w0.4", "weekday": "shrunk_8w"},
    "mostly_zero": {"level": "ma28", "weekday": "shrunk_8w"},
}
V2_CONFIG_SIGNATURE = "8109536223f4ed0a0d2842813f89381ab9e5f9288e937c7d155909a7bb120b69"

# Eligibility thresholds are definitions of the pre-registered M5 protocol (services.m5_forecast_validation), not
# values fitted for the router.
MIN_SALES_AGE_DAYS = 30       # AGE_BANDS "<30d" boundary: V2 was worse than V1 on development cold starts
RECENT_OBSERVED_DAYS = 30     # HISTORY_WINDOW: M5 scored only series whose last 30 days are all observed
MOSTLY_ZERO_POLICY = "v1"     # chosen on development origins only (services.m5_forecast_router_validation)

REASON_V2 = "sufficient_daily_history"
REASON_MISSING = "missing_daily_history"
REASON_INVALID = "invalid_history"
REASON_NEGATIVE = "negative_sales"
REASON_NO_SALES = "no_sales_history"
REASON_COLD_START = "cold_start"
REASON_INSUFFICIENT = "insufficient_history"
REASON_MOSTLY_ZERO = "mostly_zero_safety"
REASON_ERROR = "v2_error_fallback"
# Checks run in this order; the first failing check is the row's reason.
REASON_CODES: dict[str, str] = {
    REASON_MISSING: "V1: no daily sales history row for this store/product up to the cutoff",
    REASON_INVALID: "V1: history unusable - unparseable date, non-finite quantity, repeated day, or an inventory key "
                    "shared by several inventory rows",
    REASON_NEGATIVE: "V1: negative daily quantity (returns); V2 was not validated on signed demand",
    REASON_NO_SALES: "V1: no positive sale up to the cutoff",
    REASON_COLD_START: f"V1: first sale less than {MIN_SALES_AGE_DAYS} days before the cutoff (cold start)",
    REASON_INSUFFICIENT: f"V1: one of the last {RECENT_OBSERVED_DAYS} days is not observed",
    REASON_MOSTLY_ZERO: "V1: sold on under 10% of days; V2 under-forecasts this demand type",
    REASON_ERROR: "V1: the V2 computation raised an error (logged, see forecast_router.errors)",
    REASON_V2: "V2: daily history passes every eligibility check",
}

ROUTER_POLICY: dict[str, Any] = {
    "version": ROUTER_VERSION,
    "v2_routes": V2_ROUTES,
    "v2_config_signature": V2_CONFIG_SIGNATURE,
    "min_sales_age_days": MIN_SALES_AGE_DAYS,
    "recent_observed_days": RECENT_OBSERVED_DAYS,
    "mostly_zero_policy": MOSTLY_ZERO_POLICY,
    "reason_codes": REASON_CODES,
    "evidence": {
        "v2": "M5 frozen gate PASS 12/12 (final rolling 7-day WAPE 0.3668 -> 0.3519, holdout 0.3644 -> 0.3546)",
        "cold_start": "M5 development age band <30d: 7-day WAPE V1 0.485 vs V2 0.542, bias V1 -9.5% vs V2 +11.0%",
        "recent_observed_days": "M5 evaluation eligibility: last 30 days observed; V2 is unvalidated on gappy recent history",
        "mostly_zero": "M5 development mostly_zero bias V1 -6.0% vs V2 -24.4%; policy chosen by "
                       "m5_forecast_router_mostly_zero_selection.json (development origins only)",
        "negative_sales": "M5 has no negative sales, so V2 was never validated on returns",
    },
}


def v2_config() -> dict[str, Any]:
    """The frozen production V2 configuration; refuses to run if its signature drifted from the gated one."""
    config = core.make_config(V2_ROUTES)
    signature = core.config_signature(config)
    if signature != V2_CONFIG_SIGNATURE:
        raise RuntimeError(f"V2 configuration signature {signature} differs from the gated {V2_CONFIG_SIGNATURE}")
    return config


# ---------------------------------------------------------------- history assessment


def _key(value: Any) -> str | None:
    """Identifier text as services.data_loader writes it: stripped, integral numbers without '.0', blank -> None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _codes(values: pd.Series, vocabulary: Mapping[str, int]) -> np.ndarray:
    """Each value's index in ``vocabulary`` (-1: blank or unknown), normalising each distinct value once."""
    codes, uniques = pd.factorize(values)
    lookup = np.array([vocabulary.get(_key(u), -1) for u in uniques] + [-1], dtype=np.int64)
    return lookup[codes]


def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _to_days(values: Any) -> np.ndarray:
    parsed = pd.to_datetime(pd.Series(values), errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.to_numpy().astype("datetime64[D]")


@dataclass
class SeriesGroup:
    """Inventory rows with a usable history that share one cutoff: a day-aligned (series x days) matrix."""
    cutoff: np.datetime64
    positions: np.ndarray        # inventory row positions, one per matrix row
    matrix: np.ndarray           # observed quantity per day; NaN = missing day; last column = cutoff
    pool_codes: np.ndarray       # store x category codes for the pooled weekday profile


@dataclass
class HistoryAssessment:
    reason: np.ndarray           # per inventory row: REASON_* (REASON_V2 = eligible)
    groups: list[SeriesGroup]
    diagnostics: dict[str, Any]


def assess_matrix(matrix: np.ndarray, profile: core.DemandProfile | None = None) -> np.ndarray:
    """Checks on a valid day-aligned history (no invalid / negative rows): REASON_V2 or the first failing reason."""
    profile = core.demand_profile(matrix) if profile is None else profile
    t = matrix.shape[1]
    never = profile.demand_type == core.DEMAND_TYPES.index("no_sales_history")
    recent_gap = np.isnan(matrix[:, max(0, t - RECENT_OBSERVED_DAYS):]).any(axis=1) | (t < RECENT_OBSERVED_DAYS)
    mostly_zero = (profile.demand_type == core.DEMAND_TYPES.index("mostly_zero")) & (MOSTLY_ZERO_POLICY == VERSION_V1)
    return np.select([never, profile.age < MIN_SALES_AGE_DAYS, recent_gap, mostly_zero],
                     [REASON_NO_SALES, REASON_COLD_START, REASON_INSUFFICIENT, REASON_MOSTLY_ZERO],
                     default=REASON_V2).astype(object)


def assess_history(inventory: pd.DataFrame, history: pd.DataFrame, as_of: Any = None) -> HistoryAssessment:
    """Match the long daily history to inventory rows, apply the input contract and group usable series by cutoff."""
    n = len(inventory)
    reason = np.full(n, REASON_MISSING, dtype=object)
    diagnostics: dict[str, Any] = {"history_rows": int(len(history))}
    store_col = _column(history, ("store_id", "location_id"))
    product_col, date_col = _column(history, ("product_id",)), _column(history, ("date",))
    quantity_col = _column(history, ("quantity", "sales_qty"))
    missing = [label for label, col in (("store_id|location_id", store_col), ("product_id", product_col), ("date", date_col),
                                        ("quantity|sales_qty", quantity_col)) if col is None]
    if missing or not {"store_id", "product_id"} <= set(inventory.columns):
        diagnostics["contract_error"] = f"missing history columns: {missing}" if missing else "inventory has no store_id/product_id"
        reason[:] = REASON_INVALID if missing else REASON_MISSING
        return HistoryAssessment(reason, [], diagnostics)

    # Keys: every inventory row -> a (store, product) pair id; every history row -> the same id space.
    inv_store = [_key(v) for v in inventory["store_id"]]
    inv_product = [_key(v) for v in inventory["product_id"]]
    stores = {k: i for i, k in enumerate(dict.fromkeys(k for k in inv_store if k is not None))}
    products = {k: i for i, k in enumerate(dict.fromkeys(k for k in inv_product if k is not None))}
    width = max(len(products), 1)
    inv_pair = np.array([stores[s] * width + products[p] if s is not None and p is not None else -1
                         for s, p in zip(inv_store, inv_product)], dtype=np.int64)
    h_store, h_product = _codes(history[store_col], stores), _codes(history[product_col], products)
    h_pair = np.where((h_store >= 0) & (h_product >= 0), h_store * width + h_product, -1)
    # A pair on exactly one inventory row owns its history rows; a pair on several rows is ambiguous.
    keyed = np.flatnonzero(inv_pair >= 0)
    pairs, first, counts = np.unique(inv_pair[keyed], return_index=True, return_counts=True)
    owner = np.where(counts == 1, keyed[first], -1)                     # inventory position per pair, -1 if shared
    slot = np.searchsorted(pairs, h_pair).clip(0, max(len(pairs) - 1, 0))
    hit = (h_pair >= 0) & (pairs[slot] == h_pair) if len(pairs) else np.zeros(len(h_pair), dtype=bool)
    diagnostics["rows_without_inventory_match"] = int((~hit).sum())
    shared_with_history = np.zeros(len(pairs), dtype=bool)
    shared_with_history[slot[hit]] = counts[slot[hit]] > 1
    row_slot = np.searchsorted(pairs, inv_pair).clip(0, max(len(pairs) - 1, 0))
    ambiguous = np.flatnonzero((inv_pair >= 0) & shared_with_history[row_slot]) if len(pairs) else np.zeros(0, dtype=np.int64)
    reason[ambiguous] = REASON_INVALID
    diagnostics["ambiguous_inventory_rows"] = int(len(ambiguous))

    all_days = _to_days(history[date_col])
    all_quantity = pd.to_numeric(history[quantity_col], errors="coerce").to_numpy(dtype=np.float64)
    parsed = all_days[~np.isnat(all_days)]
    latest = parsed.max() if len(parsed) else np.datetime64("NaT", "D")
    rows = np.flatnonzero(hit)
    rows = rows[owner[slot[rows]] >= 0]
    position, days, quantity = owner[slot[rows]], all_days[rows], all_quantity[rows]

    # Cutoff per inventory row; history after it is future data and never read.
    if as_of is not None:
        cutoff = np.full(n, _to_days([as_of])[0])
        if np.isnat(cutoff).any():
            raise ValueError(f"as_of is not a date: {as_of!r}")
    else:
        cutoff = np.full(n, latest)
        if "snapshot_date" in inventory.columns:
            snapshot = _to_days(inventory["snapshot_date"])
            cutoff = np.where(np.isnat(snapshot), cutoff, snapshot)
    row_cutoff = cutoff[position]
    bad_date = np.isnat(days)
    future = ~bad_date & (days > row_cutoff)
    used = ~bad_date & ~future
    bad_quantity = ~np.isfinite(quantity) & ~future
    diagnostics.update(future_rows_excluded=int(future.sum()), unparseable_date_rows=int(bad_date.sum()),
                       non_finite_quantity_rows=int(bad_quantity.sum()))

    def flag(mask: np.ndarray) -> np.ndarray:
        return np.bincount(position[mask], minlength=n) > 0

    has_rows = flag(used)
    invalid = flag(bad_date) | flag(bad_quantity)
    if used.any():   # a repeated (series, day) is neither summed nor picked
        origin = days[used].min()
        span = int((days[used].max() - origin).astype(np.int64)) + 1
        cell = np.sort(position[used] * span + (days[used] - origin).astype(np.int64))
        repeated = np.unique(cell[1:][cell[1:] == cell[:-1]] // span)
        invalid[repeated] = True
        diagnostics["duplicate_date_series"] = int(len(repeated))
    negative = flag(used & np.isfinite(quantity) & (quantity < 0))
    open_rows = reason == REASON_MISSING
    reason[open_rows & invalid] = REASON_INVALID
    reason[open_rows & ~invalid & has_rows & negative] = REASON_NEGATIVE
    usable = open_rows & ~invalid & has_rows & ~negative

    groups: list[SeriesGroup] = []
    category = [_key(v) or "" for v in inventory["category"]] if "category" in inventory.columns else [""] * n
    for day in np.unique(cutoff[usable]):
        members = np.flatnonzero(usable & (cutoff == day))
        local = np.full(n, -1, dtype=np.int64)
        local[members] = np.arange(len(members))
        take = used & (local[position] >= 0)
        start = days[take].min()
        span = int((day - start).astype(np.int64)) + 1
        matrix = np.full((len(members), span), np.nan)
        matrix[local[position[take]], (days[take] - start).astype(np.int64)] = quantity[take]
        pool = pd.factorize(pd.Series([f"{inv_store[p]}|{category[p]}" for p in members]), sort=True)[0].astype(np.int64)
        groups.append(SeriesGroup(day, members, matrix, pool))
        reason[members] = assess_matrix(matrix)
    diagnostics["cutoffs"] = [str(g.cutoff) for g in groups]
    diagnostics["history_rows_used"] = int(sum(np.isfinite(g.matrix).sum() for g in groups))
    return HistoryAssessment(reason, groups, diagnostics)


# ---------------------------------------------------------------- V1-formula fields for V2 rows


def v1_dependent_fields(frame: pd.DataFrame, forecast_daily: np.ndarray) -> dict[str, np.ndarray]:
    """Interval, stock-out days and risk score exactly as analyze_demand_forecast derives them from a daily rate."""
    from services.legacy_adapters.loader import load_legacy_module

    v1 = load_legacy_module("demand_forecast_analyzer")
    hist_daily = v1._get_daily_sales(frame)
    if "demand_std" in frame.columns:
        d_std = v1._safe_num(frame["demand_std"], 0.0)
        d_std = d_std.where(d_std > 0, hist_daily * 0.25)
    else:
        d_std = hist_daily * 0.25
    stock = v1._safe_num(frame["stock_qty"], 0.0) if "stock_qty" in frame.columns \
        else v1._safe_num(frame.get("state_source_stock", pd.Series([0.0] * len(frame), index=frame.index)), 0.0)
    if "lead_time_days" in frame.columns:
        lead = v1._safe_num(frame["lead_time_days"], 3.0).clip(lower=1)
    else:
        lead = pd.Series([3.0] * len(frame), index=frame.index)
    daily = pd.Series(np.asarray(forecast_daily, dtype=np.float64), index=frame.index)
    upper, lower = v1._calc_intervals(daily, d_std)
    risk, stockout = v1._calc_risk_score(stock, daily, lead)
    return {"demand_forecast_upper": upper.to_numpy(), "demand_forecast_lower": lower.to_numpy(),
            "demand_stockout_days": stockout.to_numpy(), "demand_risk_score": risk.to_numpy(),
            "demand_forecast_score": risk.to_numpy()}


# ---------------------------------------------------------------- router


def _record_error(diagnostics: dict[str, Any], stage: str, exc: Exception) -> None:
    logger.warning("demand forecast V2 %s failed (%s: %s); the affected rows keep V1", stage, type(exc).__name__, exc,
                   exc_info=True)
    diagnostics["errors"].append({"stage": stage, "error_type": type(exc).__name__, "message": str(exc)[:300]})


def route_demand_forecast(v1_output: pd.DataFrame, daily_sales_history: pd.DataFrame | None = None, *,
                          as_of: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """V1 output plus provenance fields; rows whose daily history passes every check get the frozen V2 forecast.

    V2 values are written only after every step for a cutoff group succeeded, so an exception can never leave a
    partially replaced row: the affected rows keep V1 with REASON_ERROR and the exception is logged and reported.
    """
    out = v1_output.copy()
    n = len(out)
    supplied = isinstance(daily_sales_history, pd.DataFrame) and not daily_sales_history.empty
    reason = np.full(n, REASON_MISSING, dtype=object)
    diagnostics: dict[str, Any] = {"router_version": ROUTER_VERSION, "v2_config_signature": V2_CONFIG_SIGNATURE,
                                   "history_supplied": bool(supplied), "errors": []}
    v2_values: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    if supplied and n:
        try:
            assessment = assess_history(out, daily_sales_history, as_of)
            reason = assessment.reason
            diagnostics.update(assessment.diagnostics)
        except Exception as exc:   # a history the contract did not foresee must not stop the inventory analysis
            _record_error(diagnostics, "history_assessment", exc)
            assessment, reason = None, np.full(n, REASON_ERROR, dtype=object)
        if assessment is not None and (reason == REASON_V2).any():
            config = None
            for group in assessment.groups:
                eligible = reason[group.positions] == REASON_V2
                if not eligible.any():
                    continue
                rows = group.positions[eligible]
                try:
                    config = v2_config() if config is None else config
                    result = core.forecast_v2(group.matrix, config, HORIZON, group.pool_codes)
                    total_7d = np.round(result.aggregate_7d[eligible], 1)
                    fields = v1_dependent_fields(out.iloc[rows], total_7d / HORIZON)
                    fields["demand_forecast_7d"] = total_7d
                    fields["demand_forecast_daily"] = np.round(total_7d / HORIZON, 2)
                    fields["demand_forecast_method"] = np.array(
                        [f"V2:{lvl}+{wd}" for lvl, wd in zip(result.level_method[eligible], result.weekday_method[eligible])],
                        dtype=object)
                    for h, column in enumerate(DAILY_COLUMNS):
                        fields[column] = result.daily[eligible, h]
                    v2_values.append((rows, fields))
                except Exception as exc:   # V2 failure: V1 stays for these rows, the error is logged and reported
                    _record_error(diagnostics, f"v2_forecast cutoff={group.cutoff}", exc)
                    reason[rows] = REASON_ERROR

    daily = out["demand_forecast_7d"].to_numpy(dtype=np.float64) / HORIZON
    for column in DAILY_COLUMNS:
        out[column] = daily
    for rows, fields in v2_values:
        for column, values in fields.items():
            out.iloc[rows, out.columns.get_loc(column)] = values
    is_v2 = reason == REASON_V2
    out["demand_forecast_version"] = np.where(is_v2, VERSION_V2, VERSION_V1).astype(object)
    out["demand_forecast_reason"] = reason.astype(object)
    out["demand_forecast_fallback_reason"] = np.where(is_v2, None, reason).astype(object)
    out = out[[c for c in out.columns if c not in ROUTER_COLUMNS] + list(ROUTER_COLUMNS)]
    diagnostics["rows"] = n
    diagnostics["counts_by_version"] = {v: int(c) for v, c in out["demand_forecast_version"].value_counts().items()}
    diagnostics["counts_by_reason"] = {r: int(c) for r, c in out["demand_forecast_reason"].value_counts().items()}
    return out, diagnostics
