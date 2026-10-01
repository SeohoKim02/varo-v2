"""Canonical demand_series -> production daily sales history (the ``services.demand_forecast_router`` input contract).

    canonical demand_series (schema 1.1.0 / 1.2.0, any dataset)
        -> dataset-neutral daily history: store_id, product_id, date, quantity (+ row lineage)
        -> uploaded_data["daily_sales_history"] (:func:`attach_daily_sales_history`)
        -> services.analysis_pipeline -> services.demand_forecast_router

The bridge maps; it never repairs a value and never decides forecast eligibility (the router does):
- grain: only ``date_grain == "daily"`` rows at product grain (``product_grain`` "product" or NULL) are history rows.
- keys: store_id <- location_id, product_id <- product_id, stripped text. Raw ids are dataset-local, so one call may
  hold one id namespace only (``identity="key"`` uses the namespaced location_key / product_key instead). A row without
  a location, a product or a parseable calendar date is quarantined.
- quantity <- ``quantity_field`` (default sales_qty) unchanged: 0 stays 0 (an observed zero sale); a negative value (a
  return) stays negative - the router then routes the series to V1 with ``negative_sales``. Never abs(), clipped or 0.
- a source NULL quantity is a missing observation: no history row, because a day without a row is a missing day,
  never a zero sale. A quantity the canonical layer flagged unparseable / non-finite is passed on as NaN, so the router
  marks the series ``invalid_history``.
- a repeated (store_id, product_id, date) is neither summed nor picked: every row is passed on, the router marks the
  series ``invalid_history`` and the bridge reports it.
- unit: a series carries one quantity unit (a NULL unit is one value, "unknown"). A series mixing units is quarantined
  whole; units are never converted and never compared across series.
- ``as_of``: rows dated after it are future data and quarantined (the router also drops rows after its cutoff); unit
  consistency is judged on the rows up to ``as_of`` only.
- output rows are sorted by store_id, product_id, date (ascending; ties keep the input order). Lineage: source_dataset /
  source_file / source_row_id and the input row position on every history row (``keep_row_lineage``), and one
  metadata row per series.
A series must be complete in one call: repeated days and unit consistency are judged per series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from services.demand_forecast_router import DAILY_SALES_HISTORY_KEY

BRIDGE_VERSION = "1.0.0"
HISTORY_COLUMNS = ("store_id", "product_id", "date", "quantity")
LINEAGE_COLUMNS = ("source_dataset", "source_file", "source_row_id")
QUANTITY_FIELDS = ("sales_qty", "demand_qty")
REQUIRED_COLUMNS = ("date", "date_grain")
IDENTITY_COLUMNS = {"raw": ("location_id", "product_id"), "key": ("location_key", "product_key")}
NAMESPACE_COLUMNS = ("location_id_namespace", "product_id_namespace")
DAILY_GRAIN = "daily"
PRODUCT_GRAIN = "product"
NULL_UNIT = "<NULL>"

# Row checks in this order; the first failing check is the row's reason. mixed_unit_series is judged per series on
# the rows that pass every row check.
QUARANTINE_REASONS: dict[str, str] = {
    "non_daily_grain": "date_grain is not 'daily' (monthly/event rows are not a daily history)",
    "non_product_grain": "product_grain is neither 'product' nor NULL (category / all-product totals are not a product series)",
    "missing_location_id": "no location identifier",
    "missing_product_id": "no product identifier",
    "invalid_date": "date is not a calendar date",
    "after_as_of": "dated after as_of: future data relative to the forecast cutoff",
    "missing_quantity": "source quantity NULL: a missing observation, so no history row (never a zero sale)",
    "mixed_unit_series": "the series carries more than one quantity unit; units are never converted",
}
SERIES_COLUMNS = ("store_id", "product_id", "category", "unit", "unit_status", "source_datasets", "source_files", "location_namespace",
                  "product_namespace", "rows", "first_date", "last_date", "zero_rows", "negative_rows", "invalid_quantity_rows",
                  "duplicate_date_rows", "source_row_id_first", "source_row_id_last")
# series metadata field -> (canonical column, aggregation): "first" = first non-NULL value in date order,
# "joined" = sorted distinct values joined by "|".
SERIES_LINEAGE = {"category": ("category", "first"), "unit_status": ("unit_status", "joined"),
                  "source_datasets": ("source_dataset", "joined"), "source_files": ("source_file", "joined"),
                  "location_namespace": ("location_id_namespace", "joined"), "product_namespace": ("product_id_namespace", "joined")}


@dataclass
class BridgeResult:
    history: pd.DataFrame       # HISTORY_COLUMNS (+ LINEAGE_COLUMNS, input_position), sorted by store_id, product_id, date
    series: pd.DataFrame        # SERIES_COLUMNS, one row per (store_id, product_id), in history order
    quarantine: pd.DataFrame    # excluded input rows: input_position, reason, keys, raw date (+ source_row_id)
    report: dict[str, Any]


def _text(values: pd.Series) -> pd.Series:
    """Stripped identifier text; blank -> NA. Integral floats lose '.0' like the workbook loader's keys."""
    if pd.api.types.is_float_dtype(values):
        values = values.map(lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer() else v)
    text = values.astype("string").str.strip()
    return text.mask(text == "")


def _dates(values: pd.Series) -> pd.Series:
    """Calendar dates (datetime64, midnight). Unparseable or time-of-day values -> NaT."""
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.Series(values)
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
    else:
        parsed = pd.to_datetime(values.astype("string").str.strip(), format="ISO8601", errors="coerce")
    parsed = parsed.astype("datetime64[ns]")
    return parsed.where(parsed == parsed.dt.normalize())


def _flagged(flags: pd.Series, labels: tuple[str, ...]) -> np.ndarray:
    """Rows whose validation_flags contain any of ``labels`` (evaluated once per distinct flag string)."""
    codes, uniques = pd.factorize(flags)
    hit = np.array([any(label in str(u).split("|") for label in labels) for u in uniques] + [False])
    return hit[codes]


def _check_identity(frame: pd.DataFrame, identity: str) -> None:
    if identity not in IDENTITY_COLUMNS:
        raise ValueError(f"identity must be one of {sorted(IDENTITY_COLUMNS)}")
    missing = [c for c in IDENTITY_COLUMNS[identity] if c not in frame.columns]
    if missing:
        raise ValueError(f"canonical demand_series columns absent: {missing}")
    if identity == "key":
        return
    for column in NAMESPACE_COLUMNS:
        if column in frame.columns and frame[column].dropna().nunique() > 1:
            raise ValueError(f"{column} holds several namespaces; raw ids may collide - use identity='key'")
    namespaced = any(c in frame.columns and frame[c].notna().any() for c in NAMESPACE_COLUMNS)
    if not namespaced and "source_dataset" in frame.columns and frame["source_dataset"].dropna().nunique() > 1:
        raise ValueError("rows from several source datasets without namespaces; raw ids may collide")


def _take(values: pd.Series, rows: np.ndarray) -> np.ndarray:
    """Selected rows as an object array (selection first: no conversion of the whole column)."""
    return values.iloc[rows].to_numpy(dtype=object, na_value=None)


def canonical_to_daily_history(canonical: pd.DataFrame, *, as_of: Any = None, quantity_field: str = "sales_qty",
                               identity: str = "raw", keep_row_lineage: bool = True) -> BridgeResult:
    """Map canonical demand_series rows to the router's daily sales history (see the module docstring for the rules)."""
    if quantity_field not in QUANTITY_FIELDS:
        raise ValueError(f"quantity_field must be one of {QUANTITY_FIELDS}")
    missing = [c for c in (*REQUIRED_COLUMNS, quantity_field) if c not in canonical.columns]
    if missing:
        raise ValueError(f"not a canonical demand_series: columns absent {missing}")
    _check_identity(canonical, identity)
    cutoff = None
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if pd.isna(cutoff) or cutoff != cutoff.normalize():
            raise ValueError(f"as_of is not a calendar date: {as_of!r}")
    frame = canonical.reset_index(drop=True)
    n = len(frame)
    location_col, product_col = IDENTITY_COLUMNS[identity]
    store, product = _text(frame[location_col]), _text(frame[product_col])
    date = _dates(frame["date"])
    day = date.to_numpy()
    raw_quantity = frame[quantity_field]
    quantity = pd.to_numeric(raw_quantity, errors="coerce").astype("float64").to_numpy()
    flags = frame["validation_flags"] if "validation_flags" in frame.columns else pd.Series("", index=frame.index)
    invalid_quantity = (_flagged(flags, (f"invalid_numeric:{quantity_field}", f"nonfinite:{quantity_field}"))
                        | (raw_quantity.notna().to_numpy() & np.isnan(quantity)) | np.isinf(quantity))
    grain = frame["date_grain"].astype("string")
    product_grain = (frame["product_grain"].astype("string") if "product_grain" in frame.columns
                     else pd.Series(pd.NA, index=frame.index, dtype="string"))

    checks = [
        ("non_daily_grain", (grain != DAILY_GRAIN).fillna(True).to_numpy(dtype=bool)),
        ("non_product_grain", (product_grain.notna() & (product_grain != PRODUCT_GRAIN)).fillna(False).to_numpy(dtype=bool)),
        ("missing_location_id", store.isna().to_numpy()),
        ("missing_product_id", product.isna().to_numpy()),
        ("invalid_date", np.isnat(day)),
        ("after_as_of", (day > cutoff.to_datetime64()) if cutoff is not None else np.zeros(n, dtype=bool)),
        ("missing_quantity", np.isnan(quantity) & ~invalid_quantity),
    ]
    reason = np.full(n, "", dtype=object)
    for label, mask in checks:
        reason[(reason == "") & mask] = label
    kept = reason == ""

    unit = frame["unit"] if "unit" in frame.columns else pd.Series(pd.NA, index=frame.index, dtype="string")
    unit_code, unit_values = pd.factorize(unit, use_na_sentinel=False)       # NULL is one unit value: "unknown"
    store_code, store_values = pd.factorize(store, sort=True)                # sorted: integer order = text order
    product_code, product_values = pd.factorize(product, sort=True)
    series_code = store_code.astype(np.int64) * (len(product_values) + 1) + product_code
    if kept.any():
        combined = np.unique(series_code[kept] * (len(unit_values) + 1) + unit_code[kept])
        owners, units_per_series = np.unique(combined // (len(unit_values) + 1), return_counts=True)
        mixed = kept & np.isin(series_code, owners[units_per_series > 1])
        reason[mixed] = "mixed_unit_series"
        kept &= ~mixed

    position = np.flatnonzero(kept)
    position = position[np.lexsort((position, day[position], product_code[position], store_code[position]))]
    clean_quantity = np.where(invalid_quantity, np.nan, quantity)
    out = pd.DataFrame({"store_id": store.iloc[position].reset_index(drop=True),
                        "product_id": product.iloc[position].reset_index(drop=True),
                        "date": day[position], "quantity": clean_quantity[position]})
    if keep_row_lineage:
        for column in LINEAGE_COLUMNS:
            out[column] = frame[column].iloc[position].reset_index(drop=True) if column in frame.columns else None
        out["input_position"] = position          # row of the input frame (any other canonical field can be joined back)

    series = _series_metadata(frame, position, series_code[position], day[position], clean_quantity[position],
                              unit, unit_code, store, product)
    excluded = np.flatnonzero(~kept)
    quarantine = pd.DataFrame({"input_position": excluded, "reason": reason[excluded], "store_id": _take(store, excluded),
                               "product_id": _take(product, excluded), "raw_date": _take(frame["date"], excluded)})
    if "source_row_id" in frame.columns:
        quarantine["source_row_id"] = _take(frame["source_row_id"], excluded)
    report = _report(out, series, quarantine, quantity_field, identity, cutoff, n)
    return BridgeResult(out, series, quarantine, report)


def _per_series(values: pd.Series, position: np.ndarray, code: np.ndarray, n_series: int, how: str) -> np.ndarray:
    """Per series (``code`` = 0..n_series-1 per output row, rows in date order): the first non-NULL value ('first') or
    the sorted distinct values joined by '|' ('joined'). Works on factor codes; only multi-valued series touch text."""
    out = np.full(n_series, None, dtype=object)
    vcode, uniques = pd.factorize(values)
    v = vcode[position]
    have = v >= 0
    if not have.any():
        return out
    texts = np.array([str(u) for u in uniques], dtype=object)
    if how == "first":
        rows = np.flatnonzero(have)
        series, first = np.unique(code[rows], return_index=True)
        out[series] = texts[v[rows[first]]]
        return out
    pairs = np.unique(code[have].astype(np.int64) * (len(uniques) + 1) + v[have])
    owner, value = pairs // (len(uniques) + 1), pairs % (len(uniques) + 1)
    counts = np.bincount(owner, minlength=n_series)
    single = counts[owner] == 1
    out[owner[single]] = texts[value[single]]                                   # the common case: one distinct value
    for s in np.flatnonzero(counts > 1):
        out[s] = "|".join(sorted(texts[value[owner == s]]))
    return out


def series_boundaries(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a history sorted by store_id, product_id, date: the series code of every row, each series' first row, and
    the rows that share their (series, day) with another row (a repeated day)."""
    s, _ = pd.factorize(history["store_id"])
    p, _ = pd.factorize(history["product_id"])
    new = np.r_[True, (s[1:] != s[:-1]) | (p[1:] != p[:-1])] if len(history) else np.zeros(0, dtype=bool)
    code = np.cumsum(new) - 1
    day = history["date"].to_numpy()
    same = (code[1:] == code[:-1]) & (day[1:] == day[:-1])
    return code, np.flatnonzero(new), np.r_[same, False] | np.r_[False, same]


def _series_metadata(frame: pd.DataFrame, position: np.ndarray, pair_code: np.ndarray, day: np.ndarray, q: np.ndarray,
                     unit: pd.Series, unit_code: np.ndarray, store: pd.Series, product: pd.Series) -> pd.DataFrame:
    """One row per series: counts over its history rows and the lineage it was built from."""
    if len(position) == 0:
        return pd.DataFrame(columns=list(SERIES_COLUMNS))
    new = np.r_[True, pair_code[1:] != pair_code[:-1]]
    code = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], len(position)] - 1
    n_series = len(starts)
    same = (code[1:] == code[:-1]) & (day[1:] == day[:-1])
    duplicate = np.r_[same, False] | np.r_[False, same]

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(code, weights=mask.astype(np.float64), minlength=n_series).astype(np.int64)

    meta = {"store_id": _take(store, position[starts]), "product_id": _take(product, position[starts])}
    for name, (column, how) in SERIES_LINEAGE.items():
        meta[name] = (_per_series(frame[column], position, code, n_series, how) if column in frame.columns
                      else np.full(n_series, None, dtype=object))
    meta["unit"] = _take(unit, position[starts])
    row_ids = frame["source_row_id"] if "source_row_id" in frame.columns else None
    meta.update(rows=np.diff(np.r_[starts, len(position)]), first_date=day[starts], last_date=day[ends],
                zero_rows=count(q == 0), negative_rows=count(q < 0), invalid_quantity_rows=count(np.isnan(q)),
                duplicate_date_rows=count(duplicate),
                source_row_id_first=_take(row_ids, position[starts]) if row_ids is not None else None,
                source_row_id_last=_take(row_ids, position[ends]) if row_ids is not None else None)
    return pd.DataFrame(meta)[list(SERIES_COLUMNS)]


def _report(out: pd.DataFrame, series: pd.DataFrame, quarantine: pd.DataFrame, quantity_field: str, identity: str,
            cutoff: pd.Timestamp | None, n: int) -> dict[str, Any]:
    q = out["quantity"].to_numpy()
    reasons = quarantine["reason"].value_counts()
    has = len(series) > 0
    mixed = quarantine.loc[quarantine["reason"] == "mixed_unit_series", ["store_id", "product_id"]].drop_duplicates()
    return {
        "bridge_version": BRIDGE_VERSION, "quantity_field": quantity_field, "identity": identity,
        "as_of": None if cutoff is None else str(cutoff.date()),
        "input_rows": int(n), "history_rows": int(len(out)), "series": int(len(series)),
        "quarantined_rows": {r: int(reasons.get(r, 0)) for r in QUARANTINE_REASONS},
        "future_rows_excluded": int(reasons.get("after_as_of", 0)),
        "missing_quantity_rows_not_emitted": int(reasons.get("missing_quantity", 0)),
        "invalid_quantity_rows_emitted_as_nan": int(np.isnan(q).sum()),
        "zero_rows": int((q == 0).sum()),
        "negative_rows": int((q < 0).sum()),
        "negative_series": int((series["negative_rows"] > 0).sum()) if has else 0,
        "negative_quantity_sum": float(q[q < 0].sum()),
        "duplicate_date_rows": int(series["duplicate_date_rows"].sum()) if has else 0,
        "duplicate_date_series": int((series["duplicate_date_rows"] > 0).sum()) if has else 0,
        "mixed_unit_series_quarantined": int(len(mixed)),
        "units": {str(k): int(v) for k, v in series["unit"].fillna(NULL_UNIT).value_counts().items()} if has else {},
        "unit_status": {str(k): int(v) for k, v in series["unit_status"].fillna(NULL_UNIT).value_counts().items()} if has else {},
        "date_range": [str(pd.Timestamp(out["date"].min()).date()), str(pd.Timestamp(out["date"].max()).date())] if len(out) else None,
        "rules": {"zero": "kept as an observed zero", "negative": "kept unchanged (router: negative_sales -> V1)",
                  "missing_day": "no row (never 0)", "null_quantity": "missing observation (no row)",
                  "invalid_quantity": "NaN row (router: invalid_history)", "duplicate_date": "every row kept (router: invalid_history)",
                  "unit": "one unit per series, else the series is quarantined", "order": "store_id, product_id, date ascending"},
    }


def attach_daily_sales_history(uploaded_data: Mapping[str, Any], canonical: pd.DataFrame, *, as_of: Any = None,
                               quantity_field: str = "sales_qty", identity: str = "raw",
                               replace: bool = False) -> tuple[dict[str, Any], BridgeResult]:
    """A copy of ``uploaded_data`` whose ``daily_sales_history`` is the bridged canonical history.

    The analysis pipeline then hands it to the forecast router like a workbook ``daily_sales_history`` sheet; canonical
    location_id / product_id must be the inventory's store_id / product_id. An existing history is never overwritten
    silently.
    """
    if uploaded_data.get(DAILY_SALES_HISTORY_KEY) is not None and not replace:
        raise ValueError("uploaded_data already holds a daily_sales_history; pass replace=True to substitute it")
    result = canonical_to_daily_history(canonical, as_of=as_of, quantity_field=quantity_field, identity=identity)
    return {**uploaded_data, DAILY_SALES_HISTORY_KEY: result.history}, result
