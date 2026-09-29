"""Explicit source mappings for the external generalisation datasets (schema 1.2.0).

M5, Favorita, FreshRetailNet and KAMP. No inventory, cost, location or unit is
inferred. Wide sources are unpivoted with '<record>:<source column>' lineage; a
source NULL stays NULL (flagged), a source zero stays zero, and no row is
generated for an absent record. Every product/location carries its source
namespace so equal raw ids from different datasets never merge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.canonical_schema import EXTERNAL_VERSION, canonicalize, parse_period
from services.real_data_adapters import base_frame

NAMESPACES = {
    "m5": {"product": "M5:item_id", "store": "M5:store_id", "state": "M5:state_id"},
    "favorita": {"product": "Favorita:item_nbr", "store": "Favorita:store_nbr"},
    "freshretailnet": {"product": "FreshRetailNet:product_id", "store": "FreshRetailNet:store_id"},
    "kamp": {"product": "KAMP:rebar_grade"},
}

# Unit evidence: only what an official guide, data card or guidebook states.
EXTERNAL_UNIT_EVIDENCE = {
    "m5": {"quantity": {"unit": "item", "status": "SOURCE_METADATA", "evidence": "M5-Competitors-Guide.pdf p.5: 'd_1, d_2, ..., d_1941: The number of units sold at day i, starting from 2011-01-29.'"},
           "price": {"currency": "USD", "status": "SOURCE_METADATA", "evidence": "Guide p.3: products 'sold in the USA'; weights use 'cumulative actual dollar sales'. p.5: sell_price is 'The price of the product for the given week/store ... (average across seven days). If not available, this means that the product was not sold during the examined week.'"},
           "snap": {"unit": "binary_indicator", "status": "SOURCE_METADATA", "evidence": "Guide p.5: 'snap_CA, snap_TX, and snap_WI: A binary variable (0 or 1) indicating whether the stores of CA, TX or WI allow SNAP purchases on the examined date.'"}},
    "favorita": {"quantity": {"unit": None, "status": "UNKNOWN", "evidence": "Kaggle data description: 'The target unit_sales can be integer (e.g., a bag of chips) or float (e.g., 1.5 kg of cheese).' The unit is item-dependent (count or kg) and no field labels it, so the quantity unit stays NULL."},
                 "transactions": {"unit": "transaction", "status": "SOURCE_METADATA", "evidence": "Kaggle data description: transactions.csv 'The count of sales transactions for each date, store_nbr combination.'"},
                 "oil": {"unit": None, "status": "UNKNOWN", "evidence": "Kaggle data description: oil.csv 'Daily oil price.' No currency or volume unit is stated (the column name resembles a FRED series id, which is not evidence)."}},
    "freshretailnet": {"quantity": {"unit": "normalized_sales_amount", "status": "SOURCE_METADATA", "evidence": "Hugging Face data card: sale_amount 'The daily sales amount after global normalization (Multiplied by a specific coefficient)'; the coefficient is not disclosed, so no physical unit exists."},
                       "stockout": {"unit": "hour", "status": "SOURCE_METADATA", "evidence": "Data card: stock_hour6_22_cnt 'The number of out-of-stock hours between 6:00 and 22:00'; hours_stock_status 'The hourly out-of-stock status' (1 = out of stock verified: sum of hours 6..21 equals stock_hour6_22_cnt in every row)."},
                       "holiday": {"unit": "binary_indicator", "status": "SOURCE_METADATA", "evidence": "Data card: holiday_flag 'Holiday indicator'; technical report 3.1: 'systematically labeled Chinese statutory holidays'."},
                       "weather": {"unit": None, "status": "UNKNOWN", "evidence": "Data card: 'The total precipitation', 'The average temperature', 'The average humidity', 'The average wind force' without units."}},
    "kamp": {"quantity": {"unit": None, "status": "UNKNOWN", "evidence": "Guidebook (processed/guidebook_text.txt) variable table: HD10..UHD25 '해당강종출하량' with an empty 단위 column; no unit is printed in the workbook header."}},
}

M5_META = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
M5_STATES = ("CA", "TX", "WI")
KAMP_GRADES = ("HD10", "HD13", "SHD10", "SHD13", "UHD16", "UHD19", "UHD22", "UHD22S", "UHD25")
KAMP_TOTAL = "합계"
KAMP_KEYS = ("공사", "부위", "출하일자")
FRN_HOURS = 24
# Data card: 'out-of-stock hours between 6:00 and 22:00' = hourly slots 6..21.
FRN_STOCKOUT_WINDOW = (6, 22)
FRN_WEATHER = ("precpt", "avg_temperature", "avg_humidity", "avg_wind_level")
FRN_SCALARS = ("city_id", "store_id", "management_group_id", "first_category_id", "second_category_id", "third_category_id",
               "product_id", "dt", "sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag", "activity_flag", *FRN_WEATHER)
FRN_CATEGORY_LEVELS = ("management_group_id", "first_category_id", "second_category_id", "third_category_id")


def external_frame(raw, dataset, source, grain, scope, sheet="csv"):
    return base_frame(raw, dataset, source, grain, scope, sheet, version=EXTERNAL_VERSION)


def source_boolean(values, true_tokens, false_tokens):
    """Map a documented boolean spelling to 'true'/'false'; NULL stays NULL; an undocumented token fails loudly."""
    text = values.astype("string")
    unexpected = text.notna() & ~text.isin([*true_tokens, *false_tokens])
    if unexpected.any():
        raise ValueError(f"Undocumented boolean tokens: {sorted(text[unexpected].unique().tolist())[:10]}")
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    out[text.isin(true_tokens).fillna(False)] = "true"
    out[text.isin(false_tokens).fillna(False)] = "false"
    return out


def flag_where(frame, mask, label):
    mask = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
    current = frame["validation_flags"].fillna("").astype(str) if "validation_flags" in frame else pd.Series("", index=frame.index)
    frame["validation_flags"] = current.where(~mask, current.where(current.eq(""), current + "|") + label)
    return frame


def category_path(frame, levels):
    """Lossless, self-describing hierarchy: 'level=value|level=value'."""
    parts = [(level + "=") + frame[level].astype("string") for level in levels]
    path = parts[0]
    for part in parts[1:]:
        path = path + "|" + part
    return path


# ---------------------------------------------------------------- M5

def adapt_m5_sales(wide, source, calendar, prices):
    """Unpivot one chunk of sales_train_evaluation.csv to demand_series.

    wide: string frame with id/item/dept/cat/store/state and d_* columns; its index is the
    0-based data-record position. calendar: d, date, wm_yr_wk. prices: sell_prices rows
    (store_id, item_id, wm_yr_wk, sell_price), at least those of this chunk's series.
    """
    missing = set(M5_META) - set(wide.columns)
    if missing:
        raise ValueError(f"Required M5 columns absent: {sorted(missing)}")
    days = [c for c in wide.columns if c.startswith("d_")]
    unknown = set(days) - set(calendar["d"])
    if unknown:
        raise ValueError(f"d_* columns without a calendar date: {sorted(unknown)[:5]}")
    work = wide[[*M5_META, *days]].copy()
    work["_record"] = (wide.index + 1).astype(str)
    long = work.melt(id_vars=["_record", *M5_META], value_vars=days, var_name="d", value_name="_value")
    long = long.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left", validate="many_to_one")
    long = long.merge(prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]], on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    ns = NAMESPACES["m5"]
    out = external_frame(long, "m5", source, "daily", "store_product")
    out["source_row_id"] = long["_record"] + ":" + long["d"]
    out["raw_date"] = long["d"]
    out["date"] = out["period_start"] = out["period_end"] = long["date"]
    out["location_id"], out["location_id_namespace"] = long["store_id"], ns["store"]
    out["product_id"], out["product_id_namespace"] = long["item_id"], ns["product"]
    out["category"] = long["cat_id"]
    out["sales_qty"] = long["_value"]
    unit = EXTERNAL_UNIT_EVIDENCE["m5"]["quantity"]
    out["unit"], out["unit_status"] = unit["unit"], unit["status"]
    priced = long["sell_price"].notna()
    out["price"] = long["sell_price"]
    out["currency"] = pd.Series("USD", index=out.index).where(priced)
    out["currency_status"] = pd.Series("SOURCE_METADATA", index=out.index).where(priced)
    sales = pd.to_numeric(long["_value"], errors="coerce")
    out["validation_flags"] = ""
    # Guide: a missing weekly price means the product was not sold that week.
    flag_where(out, ~priced & sales.eq(0), "weekly_price_absent")
    flag_where(out, ~priced & sales.gt(0), "sales_without_weekly_price")
    flag_where(out, long["_value"].isna(), "missing:sales_qty")
    return "demand_series", canonicalize(out, "demand_series"), long


def adapt_m5_calendar(calendar, source):
    """calendar.csv -> calendar_event (named events) and covariate_series (state SNAP indicators)."""
    records = pd.Series(calendar.index + 1, index=calendar.index).astype(str)
    events = []
    for slot in (1, 2):
        name, kind = f"event_name_{slot}", f"event_type_{slot}"
        sub = calendar[calendar[name].notna()]
        frame = external_frame(sub, "m5", source, "daily", "dataset_calendar")
        frame["source_row_id"] = records[sub.index] + ":" + name
        frame["raw_date"] = sub["date"]
        frame["date"] = frame["period_start"] = frame["period_end"] = sub["date"]
        frame["event_name"], frame["event_type"] = sub[name], sub[kind]
        events.append(frame)
    event_frame = canonicalize(pd.concat(events).sort_values("source_row_id", key=lambda s: s.str.split(":").str[0].astype(int), kind="stable"), "calendar_event")
    snap = EXTERNAL_UNIT_EVIDENCE["m5"]["snap"]
    covariates = []
    for state in M5_STATES:
        column = f"snap_{state}"
        frame = external_frame(calendar, "m5", source, "daily", "state")
        frame["source_row_id"] = records + ":" + column
        frame["raw_date"] = calendar["date"]
        frame["date"] = frame["period_start"] = frame["period_end"] = calendar["date"]
        frame["location_id"], frame["location_id_namespace"] = state, NAMESPACES["m5"]["state"]
        frame["covariate_name"], frame["covariate_value"] = "snap", calendar[column]
        frame["unit"], frame["unit_status"] = snap["unit"], snap["status"]
        frame["validation_flags"] = ""
        flag_where(frame, calendar[column].isna(), "missing:covariate_value")
        covariates.append(frame)
    return event_frame, canonicalize(pd.concat(covariates, ignore_index=True), "covariate_series")


def adapt_m5_masters(first_rows, source):
    """first_rows: first observed sales record per series (record, item/dept/cat/store/state)."""
    items = first_rows.drop_duplicates("item_id")
    product = external_frame(items, "m5", source, "static", "dataset_product")
    product["source_row_id"] = items["_record"]
    product["product_id"], product["product_id_namespace"] = items["item_id"], NAMESPACES["m5"]["product"]
    product["category"] = items["cat_id"]
    product["category_path"] = category_path(items, ("cat_id", "dept_id"))
    stores = first_rows.drop_duplicates("store_id")
    store = external_frame(stores, "m5", source, "static", "dataset_location")
    store["source_row_id"] = stores["_record"]
    store["location_id"], store["location_id_namespace"] = stores["store_id"], NAMESPACES["m5"]["store"]
    store["location_type"], store["region"] = "store", stores["state_id"]
    states = first_rows.drop_duplicates("state_id")
    state = external_frame(states, "m5", source, "static", "dataset_location")
    state["source_row_id"] = states["_record"]
    state["location_id"], state["location_id_namespace"] = states["state_id"], NAMESPACES["m5"]["state"]
    state["location_type"], state["region"] = "state", states["state_id"]
    return canonicalize(product, "product_master"), canonicalize(pd.concat([store, state], ignore_index=True), "location_master")


# ---------------------------------------------------------------- Favorita

def adapt_favorita_train(raw, source, families):
    """train.csv chunk (strings; index = 0-based record position) -> demand_series.

    unit_sales is signed: the Kaggle description states negative values are returns.
    They stay negative (NEGATIVE_QUANTITY, documented_return); no abs(), no zero, no returns_qty invented.
    """
    missing = {"date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"} - set(raw.columns)
    if missing:
        raise ValueError(f"Required Favorita columns absent: {sorted(missing)}")
    ns = NAMESPACES["favorita"]
    out = external_frame(raw, "favorita", source, "daily", "store_product")
    out["raw_date"] = raw["date"]
    start, end = parse_period(raw["date"], "daily")
    out["date"], out["period_start"], out["period_end"] = start, start, end
    out["location_id"], out["location_id_namespace"] = raw["store_nbr"], ns["store"]
    out["product_id"], out["product_id_namespace"] = raw["item_nbr"], ns["product"]
    out["category"] = raw["item_nbr"].map(families)
    out["sales_qty"] = raw["unit_sales"]
    out["promotion"] = source_boolean(raw["onpromotion"], ("True",), ("False",))
    sales = pd.to_numeric(raw["unit_sales"], errors="coerce")
    out["validation_flags"] = ""
    flag_where(out, sales.lt(0), "documented_return:sales_qty")
    flag_where(out, raw["onpromotion"].isna(), "promotion_not_reported")
    return "demand_series", canonicalize(out, "demand_series"), {"sales_qty": "unit_sales"}


def adapt_favorita_masters(stores, items, store_source, item_source):
    ns = NAMESPACES["favorita"]
    location = external_frame(stores, "favorita", store_source, "static", "dataset_location")
    location["location_id"], location["location_id_namespace"] = stores["store_nbr"], ns["store"]
    location["location_type"] = "store"
    location["city"], location["region"] = stores["city"], stores["state"]
    location["location_subtype"], location["location_cluster"] = stores["type"], stores["cluster"]
    product = external_frame(items, "favorita", item_source, "static", "dataset_product")
    product["product_id"], product["product_id_namespace"] = items["item_nbr"], ns["product"]
    product["category"] = items["family"]
    product["category_path"] = category_path(items, ("family", "class"))
    product["perishable"] = source_boolean(items["perishable"], ("1",), ("0",))
    return canonicalize(location, "location_master"), canonicalize(product, "product_master")


def adapt_favorita_transactions(raw, source):
    evidence = EXTERNAL_UNIT_EVIDENCE["favorita"]["transactions"]
    out = external_frame(raw, "favorita", source, "daily", "store")
    out["raw_date"] = raw["date"]
    start, end = parse_period(raw["date"], "daily")
    out["date"], out["period_start"], out["period_end"] = start, start, end
    out["location_id"], out["location_id_namespace"] = raw["store_nbr"], NAMESPACES["favorita"]["store"]
    out["covariate_name"], out["covariate_value"] = "transactions", raw["transactions"]
    out["unit"], out["unit_status"] = evidence["unit"], evidence["status"]
    out["validation_flags"] = ""
    flag_where(out, raw["transactions"].isna(), "missing:covariate_value")
    return canonicalize(out, "covariate_series")


def adapt_favorita_oil(raw, source):
    # Dataset-wide daily covariate; a NULL price stays NULL (MISSING_VALUE), never zero or carried forward.
    out = external_frame(raw, "favorita", source, "daily", "national")
    out["raw_date"] = raw["date"]
    start, end = parse_period(raw["date"], "daily")
    out["date"], out["period_start"], out["period_end"] = start, start, end
    out["covariate_name"], out["covariate_value"] = "dcoilwtico", raw["dcoilwtico"]
    out["validation_flags"] = ""
    flag_where(out, raw["dcoilwtico"].isna(), "missing:covariate_value")
    return canonicalize(out, "covariate_series")


def adapt_favorita_holidays(raw, source):
    # locale_name is a city/state/country name, not a store id: kept as text, never forced to a location.
    out = external_frame(raw, "favorita", source, "daily", "dataset_calendar")
    out["raw_date"] = raw["date"]
    start, end = parse_period(raw["date"], "daily")
    out["date"], out["period_start"], out["period_end"] = start, start, end
    out["event_name"], out["event_type"] = raw["description"], raw["type"]
    out["event_locale"], out["event_locale_name"] = raw["locale"], raw["locale_name"]
    out["event_transferred"] = source_boolean(raw["transferred"], ("True",), ("False",))
    return canonicalize(out, "calendar_event")


# ---------------------------------------------------------------- FreshRetailNet

def frn_hourly_text(table):
    """Lossless 24-slot text of hours_stock_status ('0'/'1' chars) and hours_sale ('|'-joined round-trip floats)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    lengths = {name: pc.list_value_length(table.column(name)) for name in ("hours_sale", "hours_stock_status")}
    for name, length in lengths.items():
        if pc.any(pc.not_equal(length, FRN_HOURS)).as_py() or table.column(name).null_count:
            raise ValueError(f"{name} is not a complete 24-hour profile in every row")
    status = np.asarray(pc.list_flatten(table.column("hours_stock_status")).to_numpy(zero_copy_only=False)).reshape(-1, FRN_HOURS)
    if not np.isin(status, (0, 1)).all():
        raise ValueError("hours_stock_status holds values other than 0/1")
    status_text = np.ascontiguousarray((status + ord("0")).astype(np.uint8)).view(f"S{FRN_HOURS}").ravel().astype(str)
    sales = pc.list_flatten(table.column("hours_sale")).combine_chunks()
    offsets = pa.array(np.arange(0, (table.num_rows + 1) * FRN_HOURS, FRN_HOURS, dtype=np.int32))
    sales_list = pa.ListArray.from_arrays(offsets, pc.cast(sales, pa.string()))
    sales_text = pc.binary_join(sales_list, "|").to_numpy(zero_copy_only=False)
    # The text must round-trip to the same doubles.
    back = pc.cast(pc.list_flatten(pc.split_pattern(pa.array(sales_text), "|")), pa.float64())
    if not pc.all(pc.equal(back, sales)).as_py():
        raise ValueError("hours_sale text does not round-trip")
    return status, np.asarray(sales.to_numpy(zero_copy_only=False)).reshape(-1, FRN_HOURS), status_text, sales_text


def adapt_frn_sales(table, source, first_record):
    """One FreshRetailNet parquet batch (pyarrow Table) -> demand_series and its scalar frame.

    first_record is the 0-based position of the batch's first row in the file.
    """
    missing = {*FRN_SCALARS, "hours_sale", "hours_stock_status"} - set(table.column_names)
    if missing:
        raise ValueError(f"Required FreshRetailNet columns absent: {sorted(missing)}")
    scalars = table.select(list(FRN_SCALARS)).to_pandas()
    scalars.index = pd.RangeIndex(first_record, first_record + len(scalars))
    status, hourly_sales, status_text, sales_text = frn_hourly_text(table)
    ns = NAMESPACES["freshretailnet"]
    out = external_frame(scalars, "freshretailnet", source, "daily", "store_product")
    out["raw_date"] = scalars["dt"]
    start, end = parse_period(scalars["dt"], "daily")
    out["date"], out["period_start"], out["period_end"] = start, start, end
    out["location_id"], out["location_id_namespace"] = scalars["store_id"].astype("string"), ns["store"]
    out["product_id"], out["product_id_namespace"] = scalars["product_id"].astype("string"), ns["product"]
    out["category"] = scalars["management_group_id"].astype("string")
    unit = EXTERNAL_UNIT_EVIDENCE["freshretailnet"]["quantity"]
    out["sales_qty"], out["unit"], out["unit_status"] = scalars["sale_amount"], unit["unit"], unit["status"]
    out["discount_rate"] = scalars["discount"]
    out["promotion"] = source_boolean(scalars["activity_flag"].astype("string"), ("1",), ("0",))
    out["stockout_hours"] = scalars["stock_hour6_22_cnt"]
    out["stockout_window_hours"] = FRN_STOCKOUT_WINDOW[1] - FRN_STOCKOUT_WINDOW[0]
    out["hourly_stockout_status"], out["hourly_sales"] = status_text, sales_text
    lo, hi = FRN_STOCKOUT_WINDOW
    window = pd.Series(status[:, lo:hi].sum(axis=1), index=scalars.index)
    out["validation_flags"] = ""
    flag_where(out, window.ne(scalars["stock_hour6_22_cnt"]), "stockout_count_mismatch")
    flag_where(out, (pd.Series(hourly_sales.sum(axis=1), index=scalars.index) - scalars["sale_amount"]).abs().gt(1e-9), "hourly_sales_sum_mismatch")
    # Card: 1.0 means no discount, 0.9 means 10% off. Values outside (0, 1] are preserved and flagged.
    flag_where(out, scalars["discount"].eq(0), "discount_rate_zero")
    flag_where(out, scalars["discount"].gt(1), "discount_rate_above_one")
    return canonicalize(out, "demand_series"), scalars, status, hourly_sales


def collapse_identical(scalars, keys, values):
    """One row per key whose members all carry identical values; every member record id is kept.

    Fails loudly when any member differs: a differing covariate is never picked or averaged.
    """
    work = scalars[[*keys, *values]].copy()
    work["_id"] = pd.Series(scalars.index + 1, index=scalars.index).astype(str)
    groups = work.groupby(list(keys), sort=True)
    varying = groups[list(values)].nunique(dropna=False).gt(1).any(axis=1)
    if varying.any():
        raise ValueError(f"Values differ within {keys}: {varying[varying].index[:5].tolist()}")
    return groups.agg(**{v: (v, "first") for v in values}, _source_row_ids=("_id", "|".join), _source_record_count=("_id", "size")).reset_index()


def adapt_frn_covariates(scalars, source):
    """Store-day weather and the date-level holiday indicator, collapsed from their identical member rows."""
    frames = []
    weather = collapse_identical(scalars, ("store_id", "dt"), FRN_WEATHER)
    evidence = EXTERNAL_UNIT_EVIDENCE["freshretailnet"]
    for name in FRN_WEATHER:
        frame = external_frame(weather, "freshretailnet", source, "daily", "store")
        frame["location_id"], frame["location_id_namespace"] = weather["store_id"].astype("string"), NAMESPACES["freshretailnet"]["store"]
        frame["covariate_name"], frame["covariate_value"] = name, weather[name]
        frames.append((weather, frame))
    holiday = collapse_identical(scalars, ("dt",), ("holiday_flag",))
    frame = external_frame(holiday, "freshretailnet", source, "daily", "national")
    frame["covariate_name"], frame["covariate_value"] = "holiday_flag", holiday["holiday_flag"]
    frame["unit"], frame["unit_status"] = evidence["holiday"]["unit"], evidence["holiday"]["status"]
    frames.append((holiday, frame))
    out = []
    for grouped, frame in frames:
        frame["raw_date"] = grouped["dt"]
        start, end = parse_period(grouped["dt"], "daily")
        frame["date"], frame["period_start"], frame["period_end"] = start, start, end
        frame["validation_flags"] = ""
        flag_where(frame, frame["covariate_value"].isna(), "missing:covariate_value")
        out.append(frame)
    return canonicalize(pd.concat(out, ignore_index=True), "covariate_series")


def adapt_frn_masters(first_rows):
    """first_rows: first observed row per product/store (with '_file' and '_record')."""
    ns = NAMESPACES["freshretailnet"]
    products = first_rows.drop_duplicates("product_id")
    product = external_frame(products, "freshretailnet", "", "static", "dataset_product")
    product["source_file"], product["source_row_id"] = products["_file"], products["_record"]
    product["product_id"], product["product_id_namespace"] = products["product_id"].astype("string"), ns["product"]
    product["category"] = products["management_group_id"].astype("string")
    product["category_path"] = category_path(products, FRN_CATEGORY_LEVELS)
    stores = first_rows.drop_duplicates("store_id")
    store = external_frame(stores, "freshretailnet", "", "static", "dataset_location")
    store["source_file"], store["source_row_id"] = stores["_file"], stores["_record"]
    store["location_id"], store["location_id_namespace"] = stores["store_id"].astype("string"), ns["store"]
    store["location_type"], store["city"] = "store", stores["city_id"].astype("string")
    return canonicalize(product, "product_master"), canonicalize(store, "location_master")


# ---------------------------------------------------------------- KAMP

def adapt_kamp_shipments(raw, source, sheet, sheet_rows):
    """Wide rebar-grade shipment rows -> long inventory_flow.

    One row per (worksheet row, grade column); 합계 becomes a separately scoped all-products
    row (aggregate_product_total), never summed with the grades. 공사/부위 are the ordering
    project and building part (guidebook: 발주공사명, 발주공사부위), not locations. No stock exists.
    """
    missing = {*KAMP_KEYS, KAMP_TOTAL, *KAMP_GRADES} - set(raw.columns)
    if missing:
        raise ValueError(f"Required KAMP columns absent: {sorted(missing)}")
    columns = [*KAMP_GRADES, KAMP_TOTAL]
    work = raw[[*KAMP_KEYS, *columns]].copy()
    work["_sheet_row"] = pd.Series(sheet_rows, index=raw.index).astype(int)
    long = work.melt(id_vars=["_sheet_row", *KAMP_KEYS], value_vars=columns, var_name="_column", value_name="_value")
    long["_order"] = long["_column"].map({c: i for i, c in enumerate(columns)})
    long = long.sort_values(["_sheet_row", "_order"], kind="stable").reset_index(drop=True)
    total = long["_column"].eq(KAMP_TOTAL)
    out = external_frame(long, "kamp", source, "event", "shipper_outbound_by_order_project", sheet)
    out["source_row_id"] = long["_sheet_row"].astype(str) + ":" + long["_column"]
    raw_date = long["출하일자"].astype("string")
    out["raw_date"] = raw_date
    date = pd.to_datetime(raw_date, format="%Y/%m/%d", errors="coerce").dt.strftime("%Y-%m-%d")
    out["date"] = out["period_start"] = out["period_end"] = date
    out["project_id"], out["project_part"] = long["공사"], long["부위"]
    grade = long["_column"].where(~total)
    out["product_id"], out["product_name"] = grade, grade
    out["product_id_namespace"] = pd.Series(NAMESPACES["kamp"]["product"], index=long.index).where(~total)
    out["product_grain"] = np.where(total, "all_products", "product")
    out["shipment_qty"] = long["_value"]
    out["flow_kind"] = "outbound_shipment_to_order_project"
    out["validation_flags"] = ""
    flag_where(out, total, "aggregate_product_total")
    flag_where(out, long["_value"].isna(), "missing:shipment_qty")
    return canonicalize(out, "inventory_flow"), long
