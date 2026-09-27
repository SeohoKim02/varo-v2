"""Lossless, nullable interchange contract; never an algorithm input imputer."""
from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "1.0.0"
LINEAGE = ("source_dataset", "source_file", "source_sheet_or_table", "source_row_id", "transform_version")
COMMON = (*LINEAGE, "date_grain", "period_start", "period_end", "unit", "scope", "product_grain", "raw_date", "validation_flags", "analysis_eligible")
TABLE_FIELDS = {
    "inventory_snapshot": ("snapshot_date", "location_id", "location_name", "product_id", "product_name", "category", "inventory_qty", "available_qty", "reserved_qty", "damaged_qty", "expiry_date", "capacity", "capacity_unit", "product_state", "inventory_weight_kg", "inventory_volume_m3", "utilization_ratio", "reported_total_qty"),
    "inventory_flow": ("date", "location_id", "location_name", "product_id", "product_name", "category", "inbound_qty", "outbound_qty", "sales_qty", "shipment_qty", "order_qty", "demand_qty", "returns_qty", "product_state", "inbound_weight_kg", "outbound_weight_kg", "flow_kind", "purchase_count"),
    "transfer_network": ("date", "product_id", "product_name", "source_id", "target_id", "source_surplus", "target_need", "recommended_qty", "shipment_qty", "distance_km", "travel_time_min", "transport_cost", "currency", "capacity", "capacity_unit", "route_id", "route_type", "feasible", "record_kind", "expected_saving"),
    "product_master": ("product_id", "product_name", "category", "unit_weight", "weight_unit", "unit_volume", "volume_unit", "unit_cost", "sell_price", "shelf_life", "perishable", "holding_cost", "ordering_cost", "disposal_cost", "product_state"),
    "location_master": ("location_id", "location_name", "location_type", "region", "address", "latitude", "longitude", "capacity", "capacity_unit"),
    "demand_series": ("date", "location_id", "product_id", "product_name", "category", "demand_qty", "sales_qty", "price", "promotion"),
    # A warehouse/SKU occurrence is neither a movement quantity nor demand.
    "location_product_observation": ("date", "location_id", "location_name", "product_id", "product_name", "category", "purchase_count"),
}
NUMERIC = set("inventory_qty available_qty reserved_qty damaged_qty capacity inventory_weight_kg inventory_volume_m3 utilization_ratio reported_total_qty inbound_qty outbound_qty sales_qty shipment_qty order_qty demand_qty returns_qty inbound_weight_kg outbound_weight_kg purchase_count source_surplus target_need recommended_qty distance_km travel_time_min transport_cost expected_saving unit_weight unit_volume unit_cost sell_price shelf_life holding_cost ordering_cost disposal_cost latitude longitude price".split())
NONNEGATIVE = NUMERIC - {"latitude", "longitude", "expected_saving"}
REQUIRED = {
    "inventory_snapshot": ("snapshot_date", "inventory_qty"),
    "inventory_flow": ("date",),
    "transfer_network": ("date", "product_id", "source_id", "target_id"),
    "product_master": ("product_id",),
    "location_master": ("location_id",),
    "demand_series": ("date",),
    "location_product_observation": ("date", "location_id", "product_id"),
}
KEYS = {
    "inventory_snapshot": ("snapshot_date", "location_id", "product_id", "product_state", "scope", "unit"),
    "inventory_flow": ("date", "location_id", "product_id", "product_state", "flow_kind", "scope", "unit"),
    "transfer_network": ("date", "product_id", "source_id", "target_id", "route_id", "record_kind"),
    "product_master": ("product_id", "product_state"),
    "location_master": ("location_id",),
    "demand_series": ("date", "location_id", "product_id", "scope", "unit"),
    "location_product_observation": ("date", "location_id", "product_id"),
}


def schema_document():
    return {"version": VERSION, "missing_policy": "NULL is unknown; zero only from source", "identifier_policy": "dataset-local strings; no cross-dataset join without explicit crosswalk", "unit_policy": "NULL when unspecified; never assume kg/items or compare incompatible units", "aggregate_policy": "location_id/product_id may be NULL for explicit aggregate/unknown scope; never invent entities", "tables": {
        name: {"required_nonnull": list((*LINEAGE, "date_grain", "scope", *REQUIRED[name])), "columns": {c: {"dtype": "float64 nullable" if c in NUMERIC else "string nullable", "nullable": c not in (*LINEAGE, "date_grain", "scope", *REQUIRED[name])} for c in (*COMMON, *fields)}, "candidate_key": list(KEYS[name])} for name, fields in TABLE_FIELDS.items()}}


def canonicalize(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Flag bad values without discarding rows, clipping negatives or imputing."""
    out = frame.copy().reset_index(drop=True)
    flags = out.get("validation_flags", pd.Series("", index=out.index)).fillna("").astype(str)

    def flag(mask, label):
        nonlocal flags
        flags.loc[mask] = flags.loc[mask].map(lambda s: s + ("|" if s else "") + label)

    for col in (*COMMON, *TABLE_FIELDS[table]):
        if col not in out:
            out[col] = pd.NA
        if col in NUMERIC:
            original = out[col]
            values = pd.to_numeric(original, errors="coerce").astype("Float64")
            flag(original.notna() & values.isna(), "invalid_numeric:" + col)
            finite = pd.Series(np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)), index=out.index)
            flag(values.notna() & ~finite, "nonfinite:" + col)
            values.loc[~finite] = pd.NA
            out[col] = values
            if col in NONNEGATIVE:
                flag(values.lt(0).fillna(False), "negative:" + col)
        else:
            out[col] = out[col].astype("string").replace("", pd.NA)
    for col in (*LINEAGE, "date_grain", "scope", *REQUIRED[table]):
        flag(out[col].isna(), "missing:" + col)
    date_col = "snapshot_date" if table == "inventory_snapshot" else "date"
    if date_col in out:
        date = pd.to_datetime(out[date_col], format="%Y-%m-%d", errors="coerce")
        flag(date.isna(), "invalid_date")
        out[date_col] = date.dt.strftime("%Y-%m-%d").astype("string")
    if table not in {"product_master", "location_master"}:
        flag(out["unit"].isna(), "unit_unspecified")
        if "location_id" in out:
            flag(out["location_id"].isna() & out["scope"].ne("national"), "location_unresolved")
    flags = flags.map(lambda value: "|".join(dict.fromkeys(value.split("|"))) if value else "")
    out["validation_flags"] = flags.astype("string")
    out["analysis_eligible"] = flags.eq("").map({True: "true", False: "false"}).astype("string")
    return out[list(dict.fromkeys((*COMMON, *TABLE_FIELDS[table])))].copy()


def parse_period(values: pd.Series, grain: str):
    text = values.astype("string").str.strip()
    fmt = "%Y%m" if grain == "monthly" else "%Y%m%d"
    compact = text.str.replace("-", "", regex=False)
    valid_shape = compact.str.fullmatch(r"\d{6}" if grain == "monthly" else r"\d{8}").fillna(False)
    parsed = pd.to_datetime(compact.where(valid_shape), format=fmt, errors="coerce")
    end = parsed + pd.offsets.MonthEnd(0) if grain == "monthly" else parsed
    return parsed.dt.strftime("%Y-%m-%d"), end.dt.strftime("%Y-%m-%d")
