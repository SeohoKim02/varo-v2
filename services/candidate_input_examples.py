"""Explicit TEST fixtures for adapter/browser acceptance, not operating evidence."""
from __future__ import annotations
import pandas as pd


def mixed_test_tables():
    period = {"valid_from": "2026-10-16", "valid_to": "2026-10-20", "currency": "KRW"}
    inventory = [{"snapshot_date": "2026-10-16", "location_id": "S", "product_id": "P",
        "product_state": "CHILLED", "lot_id": "lot_A", "unit": "EA", "unit_status": "DIRECT",
        "inventory_qty": 10, "reserved_qty": 0, "approved_transfer_qty": 10,
        "transfer_evidence": "VERIFIED_OPERATOR_INPUT", "transfer_valid_until": "2026-10-20",
        "confirmed_inbound_qty": 0, "confirmed_outbound_qty": 0, "expiry_date": "2026-10-19",
        "date_grain": "DAY", "scope": "LOCATION_PRODUCT_STATE_LOT", "analysis_eligible": True}]
    stock = [{**period, "location_id": "S", "product_id": "P", "product_state": "CHILLED", "lot_id": "lot_A",
        "quantity_unit": "EA", "capacity_per_unit": 1, "quantum": 1, "holding_cost_per_unit_day": 0,
        "disposal_cost_per_unit": 0, "salvage_value_per_unit": 0}]
    routes = [{**period, "route_id": "S_T", "source_id": "S", "target_id": "T", "transit_days": 1,
        "vehicle_capacity": 10, "max_vehicles": 2, "fixed_cost_per_vehicle": 2, "transfer_cost_per_capacity": 0,
        "allowed_states": "CHILLED", "departure_periods": "0", "route_approved": True, "capacity_unit": "KG"}]
    sales = [{**period, "location_id": loc, "product_id": "P", "product_state": "CHILLED", "quantity_unit": "EA",
        "period": t, "price_level": level, "unit_price": price, "demand_qty": qty, "scenario_id": "base",
        "probability": 1, "demand_basis": "SCENARIO"}
        for loc,t,level,price,qty in [("S",0,"normal",10,2),("S",0,"discount_40pct",6,6),("T",1,"normal",10,4)]]
    locations = [{**period, "location_id": loc, "capacity_unit": "KG", "storage_capacity": 20,
        "handling_cost_per_capacity": 0, "processing_capacity": 20} for loc in ("S", "T")]
    return {k: pd.DataFrame(rows) for k,rows in {"inventory_snapshot": inventory, "economic_stock_policy": stock,
        "economic_route_policy": routes, "economic_sales_policy": sales, "economic_location_policy": locations}.items()}


def mixed_test_metadata():
    return {"dataset_id": "EXPLICIT_TEST", "network_id": "test-network", "as_of_date": "2026-10-16",
        "horizon_days": 2, "data_class": "TEST", "currency": "KRW", "capacity_unit": "KG", "owner_authorized": True}


def csv_test_files():
    return {f"{name}.csv": frame.to_csv(index=False).encode("utf-8-sig") for name,frame in mixed_test_tables().items()}
