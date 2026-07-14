"""Data-shape adapters between V2 workbooks and legacy algorithm inputs."""
from __future__ import annotations

from datetime import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from services.dqn_guard import strip_dqn_columns


def _alias(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> pd.DataFrame:
    if target in df.columns:
        return df
    for candidate in candidates:
        if candidate in df.columns:
            df[target] = df[candidate]
            break
    return df


def prepare_legacy_data(data: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Create legacy-compatible copies without mutating uploaded DataFrames."""
    stores = strip_dqn_columns(data.get("stores"))
    products = strip_dqn_columns(data.get("products"))
    inventory = strip_dqn_columns(data.get("inventory"))
    routes = strip_dqn_columns(data.get("routes"))
    config = strip_dqn_columns(data.get("config"))

    for target, candidates in {
        "store_id": ("node_id", "id"),
        "store_name": ("node_name", "name"),
        "type": ("node_type", "store_type"),
    }.items():
        stores = _alias(stores, target, candidates)
    if "type" in stores:
        stores["type"] = stores["type"].astype(str).str.upper()

    products = _alias(products, "product_id", ("item_id", "id"))
    products = _alias(products, "product_name", ("item_name", "name"))
    products = _alias(products, "disposal_cost_per_unit", ("disposal_cost",))

    inventory_aliases = {
        "store_id": ("node_id",),
        "product_id": ("item_id",),
        "stock_qty": ("current_stock", "quantity", "inventory_qty", "stock"),
        "current_stock": ("stock_qty", "quantity", "inventory_qty"),
        "sales_30d": ("sales_30", "sales_qty", "avg_daily_sales"),
        "sales_30": ("sales_30d", "sales_qty"),
        "expiry_days": ("days_to_expiry", "shelf_life_days"),
        "days_to_expiry": ("expiry_days", "shelf_life_days"),
        "category": ("inventory_category",),
        "inventory_category": ("category",),
        "disposal_cost_per_unit": ("disposal_cost",),
        "disposal_cost": ("disposal_cost_per_unit",),
    }
    for target, candidates in inventory_aliases.items():
        inventory = _alias(inventory, target, candidates)

    # Some DQN workbooks intentionally keep only operational stock/sales
    # columns. Legacy clustering and min-cost helpers need explicit supply and
    # demand quantities, so derive them on this copied frame only.
    if "stock_qty" in inventory.columns:
        stock = pd.to_numeric(inventory["stock_qty"], errors="coerce").fillna(0).clip(lower=0)
        if "sales_30d" in inventory.columns:
            demand_30d = pd.to_numeric(inventory["sales_30d"], errors="coerce").fillna(0).clip(lower=0)
        elif "avg_daily_sales" in inventory.columns:
            demand_30d = (
                pd.to_numeric(inventory["avg_daily_sales"], errors="coerce").fillna(0).clip(lower=0) * 30
            )
        else:
            demand_30d = pd.Series(0.0, index=inventory.index, dtype="float64")
        if "dead_stock_qty" not in inventory.columns:
            inventory["dead_stock_qty"] = (stock - demand_30d).clip(lower=0)
        if "demand_qty" not in inventory.columns:
            inventory["demand_qty"] = demand_30d

    route_aliases = {
        "from_id": ("source_id", "from_store_id"),
        "to_id": ("target_id", "to_store_id"),
        "source_id": ("from_id", "from_store_id"),
        "target_id": ("to_id", "to_store_id"),
        "distance_km": ("route_distance_km", "direct_distance_km", "distance"),
        "travel_time_min": ("route_time_min", "time_min", "duration_min"),
        "transport_cost": ("estimated_cost", "direct_cost", "cost"),
        "estimated_cost": ("transport_cost", "direct_cost", "cost"),
    }
    for target, candidates in route_aliases.items():
        routes = _alias(routes, target, candidates)

    return {
        "stores": stores,
        "products": products,
        "inventory": inventory,
        "routes": routes,
        "config": config,
    }


def config_value(config: pd.DataFrame, key: str, default: Any = None) -> Any:
    if config is None or config.empty:
        return default
    key_col = next((column for column in ("key", "config_key", "setting", "name") if column in config.columns), None)
    value_col = next((column for column in ("value", "config_value", "setting_value") if column in config.columns), None)
    if not key_col or not value_col:
        return default
    rows = config[config[key_col].astype(str).str.strip() == key]
    if rows.empty:
        return default
    value = rows.iloc[0][value_col]
    return default if pd.isna(value) else value


def config_time(config: pd.DataFrame, key: str = "departure_time", default: time = time(9, 0)) -> time:
    value = config_value(config, key, default)
    if isinstance(value, time):
        return value
    text = str(value).strip()
    try:
        hour, minute = text.split(":")[:2]
        return time(int(float(hour)), int(float(minute)))
    except (TypeError, ValueError):
        return default


def build_candidate_frame(recommendations: Sequence[Mapping[str, object]], inventory: pd.DataFrame) -> pd.DataFrame:
    candidates = strip_dqn_columns(pd.DataFrame([dict(item) for item in recommendations]))
    if candidates.empty:
        return candidates
    candidates["source_store_id"] = candidates.get("source_id")
    candidates["target_store_id"] = candidates.get("target_id")
    candidates["source_store"] = candidates.get("source_name")
    candidates["target_store"] = candidates.get("target_name")
    candidates["suggested_qty"] = candidates.get("recommended_qty")
    candidates["uploaded_vhs_score"] = pd.to_numeric(
        candidates.get("vhs_score"), errors="coerce"
    )
    primary_action = candidates.get("varo_action", pd.Series(index=candidates.index, dtype=object))
    fallback_action = candidates.get("greedy_action", pd.Series("재고 이동", index=candidates.index))
    candidates["final_recommendation"] = primary_action.where(
        ~primary_action.isin([None, "", "비교 불가"]), fallback_action
    ).fillna("재고 이동")
    candidates["recommended_distance_km"] = candidates.get("distance_km")
    candidates["recommended_time_min"] = candidates.get("expected_time_min", candidates.get("travel_time_min"))
    candidates["estimated_cost"] = candidates.get("move_cost", candidates.get("estimated_cost"))

    if inventory is None or inventory.empty or not {"store_id", "product_id"}.issubset(inventory.columns):
        return candidates
    source_metrics = strip_dqn_columns(inventory).copy()
    source_metrics["source_id"] = source_metrics["store_id"].astype(str)
    source_metrics["product_id"] = source_metrics["product_id"].astype(str)
    candidates["source_id"] = candidates["source_id"].astype(str)
    candidates["product_id"] = candidates["product_id"].astype(str)
    drop_columns = [
        column for column in ("store_id", "store_name", "product_name", "source_name", "target_name")
        if column in source_metrics.columns
    ]
    source_metrics = source_metrics.drop(columns=drop_columns)
    return candidates.merge(source_metrics, on=["source_id", "product_id"], how="left", suffixes=("", "_inventory"))


def add_cluster_context(candidates: pd.DataFrame, cluster_map: Mapping[object, object]) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return candidates
    result = candidates.copy()
    normalized = {str(key): value for key, value in cluster_map.items()}
    result["source_cluster"] = result["source_id"].astype(str).map(normalized)
    result["target_cluster"] = result["target_id"].astype(str).map(normalized)
    result["is_same_cluster"] = (
        result["source_cluster"].notna()
        & result["target_cluster"].notna()
        & (result["source_cluster"] == result["target_cluster"])
    )
    return result


def safe_transfer_route_lookup(routes: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Replacement for legacy underscore-column itertuples lookup."""
    lookup: dict[tuple[str, str], tuple[float, float, float]] = {}
    if routes is None or routes.empty:
        return lookup
    sorted_routes = routes.sort_values(["_transport_cost", "_distance_km"], na_position="last")
    for row in sorted_routes.to_dict("records"):
        source = str(row.get("from_id", "")).strip()
        target = str(row.get("to_id", "")).strip()
        if not source or not target:
            continue
        value = (
            float(row.get("_distance_km", np.nan)),
            float(row.get("_transport_cost", np.nan)),
            float(row.get("_time_min", np.nan)),
        )
        lookup.setdefault((source, target), value)
        lookup.setdefault((target, source), value)
    return lookup
