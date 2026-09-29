"""Lossless, nullable interchange contract; never an algorithm input imputer."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Transform version of the domestic adapters (Suhyup, LogisAll, Jangbogo, NFQS, AI Hub); their transforms are unchanged.
VERSION = "1.1.0"
# Schema 1.2.0 adds namespaced identity, context tables and the external adapters (M5, Favorita, FreshRetailNet, KAMP).
SCHEMA_VERSION = "1.2.0"
EXTERNAL_VERSION = "1.2.0"
LINEAGE = ("source_dataset", "source_file", "source_sheet_or_table", "source_row_id", "transform_version")
# `unit` is the quantity unit of the row's count/volume quantities; unit_status is its evidence.
COMMON = (*LINEAGE, "date_grain", "period_start", "period_end", "unit", "unit_status", "scope", "product_grain", "raw_date",
          "validation_flags", "quality_flags", "quality_status", "analysis_eligible")
# <entity>_id_namespace names the source dataset and field (e.g. "M5:item_id"); <entity>_key = namespace + ":" + raw id
# is the cross-dataset identity. The raw id stays dataset-local; equal raw ids in different namespaces are never merged.
PRODUCT_IDENTITY = ("product_id_namespace", "product_key")
LOCATION_IDENTITY = ("location_id_namespace", "location_key")
IDENTITY_KEYS = (("product_id", "product_id_namespace", "product_key"), ("location_id", "location_id_namespace", "location_key"))
TABLE_FIELDS = {
    "inventory_snapshot": ("snapshot_date", "location_id", "location_name", "product_id", "product_name", "category", "inventory_qty", "available_qty", "reserved_qty", "damaged_qty", "expiry_date", "capacity", "capacity_unit", "capacity_unit_status", "product_state", "inventory_weight_kg", "weight_unit", "weight_unit_status", "inventory_volume_m3", "volume_unit", "volume_unit_status", "utilization_ratio", "reported_total_qty", "source_record_count", *PRODUCT_IDENTITY, *LOCATION_IDENTITY),
    "inventory_flow": ("date", "location_id", "location_name", "product_id", "product_variant_id", "product_name", "category", "inbound_qty", "outbound_qty", "sales_qty", "shipment_qty", "order_qty", "demand_qty", "returns_qty", "product_state", "inbound_weight_kg", "outbound_weight_kg", "weight_unit", "weight_unit_status", "flow_kind", "purchase_count", "source_record_count",
                       # Ordering project and building part of an order-based shipment (KAMP 공사/부위); not a location.
                       "project_id", "project_part", *PRODUCT_IDENTITY, *LOCATION_IDENTITY),
    "transfer_network": ("date", "product_id", "product_name", "source_id", "target_id", "source_surplus", "target_need", "recommended_qty", "shipment_qty", "distance_km", "distance_unit", "distance_unit_status", "travel_time_min", "time_unit", "time_unit_status", "transport_cost", "currency", "currency_status", "capacity", "capacity_unit", "capacity_unit_status", "route_id", "route_type", "feasible", "record_kind", "expected_saving", "source_record_count"),
    "product_master": ("product_id", "product_name", "category", "unit_weight", "weight_unit", "weight_unit_status", "unit_volume", "volume_unit", "volume_unit_status", "unit_cost", "sell_price", "currency", "currency_status", "shelf_life", "time_unit", "time_unit_status", "perishable", "holding_cost", "ordering_cost", "disposal_cost", "product_state", "category_path", *PRODUCT_IDENTITY),
    "location_master": ("location_id", "location_name", "location_type", "region", "address", "latitude", "longitude", "capacity", "capacity_unit", "capacity_unit_status", "city", "location_subtype", "location_cluster", *LOCATION_IDENTITY),
    "demand_series": ("date", "location_id", "product_id", "product_variant_id", "product_name", "category", "demand_qty", "sales_qty", "price", "currency", "currency_status", "promotion", "source_record_count",
                      # Availability (censoring) and discount evidence of a sales observation; hourly profiles are lossless text.
                      "discount_rate", "stockout_hours", "stockout_window_hours", "hourly_sales", "hourly_stockout_status", *PRODUCT_IDENTITY, *LOCATION_IDENTITY),
    # A warehouse/SKU occurrence is neither a movement quantity nor demand.
    "location_product_observation": ("date", "location_id", "location_name", "product_id", "product_name", "category", "purchase_count", "source_record_count", *PRODUCT_IDENTITY, *LOCATION_IDENTITY),
    # One physical measurement of an item; repeated measurements are not product duplicates.
    "product_measurement": ("product_id", "product_name", "category", "category_code", "measurement_context", "length", "width", "height", "dimension_unit", "dimension_unit_status", "weight", "weight_unit", "weight_unit_status", "source_record_count"),
    # Dataset-local product identity over time; product_id is never renumbered.
    "product_identity_version": ("product_id", "product_name", "version_seq", "valid_from", "valid_to", "observation_count", "source_tables", "source_files", "pack_spec_tokens", "temporal_relation", "name_change_class"),
    # Context tables (1.2.0): dated calendar events and covariates; never quantities of the fact tables.
    "calendar_event": ("date", "event_name", "event_type", "event_locale", "event_locale_name", "event_transferred"),
    "covariate_series": ("date", "location_id", "covariate_name", "covariate_value", "source_record_count", *LOCATION_IDENTITY),
}
NUMERIC = set("inventory_qty available_qty reserved_qty damaged_qty capacity inventory_weight_kg inventory_volume_m3 utilization_ratio reported_total_qty inbound_qty outbound_qty sales_qty shipment_qty order_qty demand_qty returns_qty inbound_weight_kg outbound_weight_kg purchase_count source_surplus target_need recommended_qty distance_km travel_time_min transport_cost expected_saving unit_weight unit_volume unit_cost sell_price shelf_life holding_cost ordering_cost disposal_cost latitude longitude price source_record_count length width height weight version_seq observation_count discount_rate stockout_hours stockout_window_hours covariate_value".split())
# A covariate (temperature, oil price) may legitimately be negative; it is not a quantity.
NONNEGATIVE = NUMERIC - {"latitude", "longitude", "expected_saving", "covariate_value"}
REQUIRED = {
    "inventory_snapshot": ("snapshot_date", "inventory_qty"),
    "inventory_flow": ("date",),
    "transfer_network": ("date", "product_id", "source_id", "target_id"),
    "product_master": ("product_id",),
    "location_master": ("location_id",),
    "demand_series": ("date",),
    "location_product_observation": ("date", "location_id", "product_id"),
    "product_measurement": ("product_id", "measurement_context"),
    "product_identity_version": ("product_id", "version_seq", "valid_from"),
    "calendar_event": ("date", "event_name"),
    "covariate_series": ("date", "covariate_name"),
}
KEYS = {
    "inventory_snapshot": ("snapshot_date", "location_id", "product_id", "product_state", "scope", "unit"),
    "inventory_flow": ("date", "location_id", "product_id", "product_variant_id", "product_state", "flow_kind", "scope", "unit", "project_id", "project_part"),
    "transfer_network": ("date", "product_id", "source_id", "target_id", "route_id", "record_kind"),
    "product_master": ("product_id", "product_state"),
    "location_master": ("location_id",),
    "demand_series": ("date", "location_id", "product_id", "product_variant_id", "scope", "unit"),
    "location_product_observation": ("date", "location_id", "product_id"),
    "product_measurement": ("product_id", "measurement_context", "source_sheet_or_table"),
    "product_identity_version": ("source_dataset", "product_id", "product_name"),
    "calendar_event": ("date", "event_locale", "event_locale_name", "event_name", "event_type"),
    "covariate_series": ("date", "location_id", "covariate_name", "scope"),
}
# Tables whose rows carry no count quantity, so a NULL `unit` is not a defect.
NO_QUANTITY_UNIT = {"product_master", "location_master", "product_measurement", "product_identity_version", "calendar_event"}
KEY_IDENTIFIERS = {"date", "snapshot_date", "location_id", "product_id", "product_variant_id", "source_id", "target_id", "measurement_context", "valid_from", "version_seq", *LINEAGE, "date_grain", "scope",
                   "covariate_name", "event_name"}

UNIT_STATUSES = ("DIRECT", "SOURCE_METADATA", "DERIVED", "UNKNOWN")
UNIT_STATUS_COLUMNS = {"unit": "unit_status", "weight_unit": "weight_unit_status", "volume_unit": "volume_unit_status",
                       "dimension_unit": "dimension_unit_status", "currency": "currency_status", "distance_unit": "distance_unit_status",
                       "time_unit": "time_unit_status", "capacity_unit": "capacity_unit_status"}

# Standard quality codes, most severe first. quality_status is the first present code or VALID.
QUALITY_CODES = ("MISSING_KEY", "MISSING_VALUE", "SOURCE_ANOMALY", "NEGATIVE_QUANTITY", "DUPLICATE_OBSERVATION",
                 "UNKNOWN_UNIT", "UNKNOWN_LOCATION", "AMBIGUOUS_PRODUCT", "BENCHMARK_PROXY", "OUT_OF_SCOPE")
_EXACT_CODES = {
    "invalid_date": "SOURCE_ANOMALY", "unit_unspecified": "UNKNOWN_UNIT", "location_unresolved": "UNKNOWN_LOCATION",
    "duplicate_key": "DUPLICATE_OBSERVATION", "cross_release_conflict": "DUPLICATE_OBSERVATION",
    "cross_release_repeat": "DUPLICATE_OBSERVATION", "repeated_key_unexplained": "DUPLICATE_OBSERVATION",
    "adjacent_identical_record": "DUPLICATE_OBSERVATION",
    "repeated_key_missing_subkey": "MISSING_KEY",
    "product_identity_versioned": "AMBIGUOUS_PRODUCT", "product_label_variants": "AMBIGUOUS_PRODUCT",
    "benchmark_proxy_constraints": "BENCHMARK_PROXY", "aggregate_product_total": "OUT_OF_SCOPE",
    "source_flow_identity_mismatch": "SOURCE_ANOMALY",
    # 1.2.0 external datasets.
    "weekly_price_absent": "MISSING_VALUE", "sales_without_weekly_price": "SOURCE_ANOMALY",
    "promotion_not_reported": "MISSING_VALUE", "discount_rate_zero": "SOURCE_ANOMALY", "discount_rate_above_one": "SOURCE_ANOMALY",
    "stockout_count_mismatch": "SOURCE_ANOMALY", "hourly_sales_sum_mismatch": "SOURCE_ANOMALY",
}
_PREFIX_CODES = {"negative": "NEGATIVE_QUANTITY", "invalid_numeric": "SOURCE_ANOMALY", "nonfinite": "SOURCE_ANOMALY",
                 "float_residue_negative": "SOURCE_ANOMALY", "invalid_unit_status": "UNKNOWN_UNIT",
                 "unit_evidence_missing": "UNKNOWN_UNIT", "unit_status_inconsistent": "UNKNOWN_UNIT",
                 "documented_return": "NEGATIVE_QUANTITY"}
# Flags that document scope/semantics but do not make a row unusable on its own.
INFORMATIONAL_FLAGS = {"float_residue_negative", "product_identity_versioned", "product_label_variants", "source_flow_identity_mismatch",
                       "adjacent_identical_record",
                       # 1.2.0: documented source semantics; a documented return still carries the blocking negative:<column>.
                       "documented_return", "weekly_price_absent", "promotion_not_reported", "discount_rate_zero", "discount_rate_above_one"}


def quality_code(flag: str) -> str:
    """Map one detailed validation flag to its standard quality code."""
    if flag in _EXACT_CODES:
        return _EXACT_CODES[flag]
    prefix, _, column = flag.partition(":")
    if prefix == "missing":
        return "MISSING_KEY" if column in KEY_IDENTIFIERS else "MISSING_VALUE"
    return _PREFIX_CODES.get(prefix, "SOURCE_ANOMALY")


def is_blocking(flag: str) -> bool:
    return flag.partition(":")[0] not in INFORMATIONAL_FLAGS


def finalize_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive quality_flags/quality_status/analysis_eligible from validation_flags."""
    flags = frame["validation_flags"].fillna("").astype(str)
    order = {code: i for i, code in enumerate(QUALITY_CODES)}
    # Evaluated once per distinct flag string: identical to per-row evaluation, bounded cost on large tables.
    cleaned, joined, status, eligible = {}, {}, {}, {}
    for value in flags.unique():
        parts = [f for f in str(value).split("|") if f]
        codes = sorted({quality_code(f) for f in parts}, key=order.__getitem__)
        cleaned[value] = "|".join(dict.fromkeys(parts))
        joined[value] = "|".join(codes)
        status[value] = codes[0] if codes else "VALID"
        eligible[value] = "false" if any(is_blocking(f) for f in parts) else "true"
    frame["validation_flags"] = flags.map(cleaned).astype("string")
    frame["quality_flags"] = flags.map(joined).astype("string")
    frame["quality_status"] = flags.map(status).astype("string")
    frame["analysis_eligible"] = flags.map(eligible).astype("string")
    return frame


def _append_label(current: pd.Series, label: str) -> pd.Series:
    """s + ("|" if s else "") + label, vectorised."""
    return current.where(current.eq(""), current + "|") + label


def add_flag(frame: pd.DataFrame, mask, label: str) -> pd.DataFrame:
    mask = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
    if mask.any():
        current = frame.loc[mask, "validation_flags"].fillna("").astype(str)
        frame.loc[mask, "validation_flags"] = _append_label(current, label)
    return frame


CORE_TABLES = ("inventory_snapshot", "inventory_flow", "transfer_network", "product_master", "location_master", "demand_series", "location_product_observation")
# Evidence tables added in 1.1.0: repeated physical measurements and product identity over time.
AUXILIARY_TABLES = ("product_measurement", "product_identity_version")
# Context tables added in 1.2.0: calendar events and dated covariates (store traffic, oil price, weather, SNAP/holiday indicators).
CONTEXT_TABLES = ("calendar_event", "covariate_series")


def _table_document(name):
    fields = TABLE_FIELDS[name]
    return {"required_nonnull": list((*LINEAGE, "date_grain", "scope", *REQUIRED[name])), "columns": {c: {"dtype": "float64 nullable" if c in NUMERIC else "string nullable", "nullable": c not in (*LINEAGE, "date_grain", "scope", *REQUIRED[name])} for c in (*COMMON, *fields)}, "candidate_key": list(KEYS[name])}


def schema_document():
    return {"version": SCHEMA_VERSION, "domestic_transform_version": VERSION, "external_transform_version": EXTERNAL_VERSION,
            "missing_policy": "NULL is unknown; zero only from source", "identifier_policy": "dataset-local strings; no cross-dataset join without explicit crosswalk",
            "identity_policy": "product_id/location_id keep the raw dataset-local spelling. External (1.2.0) rows also carry <entity>_id_namespace ('<Dataset>:<source field>') and <entity>_key = namespace + ':' + raw id, derived centrally; cross-dataset unions must key on product_key/location_key, never on the raw id. 1.1.0 domestic outputs carry no namespace (key NULL; (source_dataset, id) identifies them).",
            "wide_to_long_policy": "An unpivoted cell keeps source_row_id '<record>:<source column>'; a source NULL cell stays NULL (flagged), a source zero stays zero; no cell is generated for an absent record.", "unit_policy": "NULL when unspecified; never assume kg/items or compare incompatible units", "unit_status_values": list(UNIT_STATUSES),
            "unit_status_policy": "DIRECT=unit printed in the source header/value; SOURCE_METADATA=official specification/page/label states it; DERIVED=produced by a documented transformation or reference engine; UNKNOWN=no evidence (unit NULL). A unit without status is flagged.",
            "quality_codes": list(QUALITY_CODES), "informational_flags": sorted(INFORMATIONAL_FLAGS),
            "quality_policy": "Flags state analysis scope; they never delete, clip or repair rows. analysis_eligible=false when any blocking flag exists.",
            "aggregate_policy": "location_id/product_id may be NULL for explicit aggregate/unknown scope; never invent entities",
            "source_record_policy": "source_record_count>1 only when source records sharing the canonical key were collapsed by a documented dataset policy; source_row_id then lists every 1-based record id.",
            "tables": {name: _table_document(name) for name in CORE_TABLES},
            "auxiliary_tables": {name: _table_document(name) for name in AUXILIARY_TABLES},
            "context_tables": {name: _table_document(name) for name in CONTEXT_TABLES}}


def canonicalize(frame: pd.DataFrame, table: str, residue_tolerance: dict | None = None) -> pd.DataFrame:
    """Flag bad values without discarding rows, clipping negatives or imputing.

    residue_tolerance maps a column to an absolute tolerance below which a
    negative value is a documented floating-point residue; it stays unchanged.
    """
    out = frame.copy().reset_index(drop=True)
    flags = out.get("validation_flags", pd.Series("", index=out.index)).fillna("").astype(str)
    residue_tolerance = residue_tolerance or {}

    def flag(mask, label):
        nonlocal flags
        if mask.any():
            flags.loc[mask] = _append_label(flags.loc[mask], label)

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
                negative = values.lt(0).fillna(False)
                if col in residue_tolerance:
                    residue = negative & values.gt(-residue_tolerance[col]).fillna(False)
                    flag(residue, "float_residue_negative:" + col)
                    negative = negative & ~residue
                flag(negative, "negative:" + col)
        else:
            out[col] = out[col].astype("string").replace("", pd.NA)
    for idcol, namespace_col, key_col in IDENTITY_KEYS:
        if key_col in out:
            # Derived centrally so a key never disagrees with its namespace and raw id; a NULL part gives a NULL key.
            out[key_col] = (out[namespace_col] + ":" + out[idcol]).astype("string")
    for unit_col, status_col in UNIT_STATUS_COLUMNS.items():
        if unit_col not in out or status_col not in out:
            continue
        unit, status = out[unit_col], out[status_col]
        flag(status.notna() & ~status.isin(UNIT_STATUSES), "invalid_unit_status:" + unit_col)
        flag(unit.notna() & status.isna(), "unit_evidence_missing:" + unit_col)
        flag((unit.notna() & status.eq("UNKNOWN")) | (unit.isna() & status.isin(UNIT_STATUSES[:3])), "unit_status_inconsistent:" + unit_col)
        # A NULL unit without a claim is, by definition, UNKNOWN; no unit is invented.
        out[status_col] = status.mask(unit.isna() & status.isna(), "UNKNOWN").astype("string")
    for col in (*LINEAGE, "date_grain", "scope", *REQUIRED[table]):
        flag(out[col].isna(), "missing:" + col)
    date_col = "snapshot_date" if table == "inventory_snapshot" else "date"
    if date_col in out:
        date = pd.to_datetime(out[date_col], format="%Y-%m-%d", errors="coerce")
        flag(date.isna(), "invalid_date")
        out[date_col] = date.dt.strftime("%Y-%m-%d").astype("string")
    if table not in NO_QUANTITY_UNIT:
        flag(out["unit"].isna(), "unit_unspecified")
        if "location_id" in out:
            flag(out["location_id"].isna() & out["scope"].ne("national"), "location_unresolved")
    out["validation_flags"] = flags
    out = finalize_quality(out)
    return out[list(dict.fromkeys((*COMMON, *TABLE_FIELDS[table])))].copy()


def parse_period(values: pd.Series, grain: str):
    text = values.astype("string").str.strip()
    fmt = "%Y%m" if grain == "monthly" else "%Y%m%d"
    compact = text.str.replace("-", "", regex=False)
    valid_shape = compact.str.fullmatch(r"\d{6}" if grain == "monthly" else r"\d{8}").fillna(False)
    parsed = pd.to_datetime(compact.where(valid_shape), format=fmt, errors="coerce")
    end = parsed + pd.offsets.MonthEnd(0) if grain == "monthly" else parsed
    return parsed.dt.strftime("%Y-%m-%d"), end.dt.strftime("%Y-%m-%d")
