import pandas as pd


def analyze_dc_retailer_routes(stores, routes):
    """DC-점포 거리 및 운송비 계산 속도 개선판.

    기존 결과 컬럼은 유지하면서 iterrows 반복을 줄이고,
    stores 시트의 type / store_type 컬럼을 모두 지원한다.
    """
    stores_data = stores.copy()
    routes_data = routes.copy()

    if stores_data.empty or routes_data.empty:
        empty = pd.DataFrame()
        return empty, empty

    type_col = "type" if "type" in stores_data.columns else "store_type" if "store_type" in stores_data.columns else None

    if type_col is None:
        stores_data["type"] = "STORE"
        type_col = "type"

    stores_data["_type_upper"] = stores_data[type_col].astype(str).str.upper()
    stores_data["_type_lower"] = stores_data[type_col].astype(str).str.lower()

    dc_ids = set(stores_data[stores_data["_type_upper"] == "DC"]["store_id"].astype(str))
    retailer_ids = set(
        stores_data[
            stores_data["_type_lower"].isin(["retailer", "store", "점포"])
        ]["store_id"].astype(str)
    )

    store_name_map = dict(zip(stores_data["store_id"].astype(str), stores_data["store_name"]))

    if "from_id" not in routes_data.columns or "to_id" not in routes_data.columns:
        empty = pd.DataFrame()
        return empty, empty

    routes_data["from_id"] = routes_data["from_id"].astype(str)
    routes_data["to_id"] = routes_data["to_id"].astype(str)

    forward = routes_data["from_id"].isin(dc_ids) & routes_data["to_id"].isin(retailer_ids)
    reverse = routes_data["from_id"].isin(retailer_ids) & routes_data["to_id"].isin(dc_ids)

    selected = routes_data[forward | reverse].copy()

    if selected.empty:
        empty = pd.DataFrame()
        return empty, empty

    selected["dc_id"] = selected["from_id"].where(forward[forward | reverse], selected["to_id"])
    selected["retailer_id"] = selected["to_id"].where(forward[forward | reverse], selected["from_id"])

    selected["dc_name"] = selected["dc_id"].map(store_name_map).fillna(selected["dc_id"])
    selected["retailer_name"] = selected["retailer_id"].map(store_name_map).fillna(selected["retailer_id"])

    if "travel_time_min" not in selected.columns:
        if "time_min" in selected.columns:
            selected["travel_time_min"] = selected["time_min"]
        else:
            selected["travel_time_min"] = pd.to_numeric(selected.get("distance_km", 0), errors="coerce").fillna(0) / 30 * 60

    if "cost_per_km" not in selected.columns:
        selected["cost_per_km"] = 900

    selected["distance_km"] = pd.to_numeric(selected["distance_km"], errors="coerce").fillna(0)
    selected["travel_time_min"] = pd.to_numeric(selected["travel_time_min"], errors="coerce").fillna(0)
    selected["cost_per_km"] = pd.to_numeric(selected["cost_per_km"], errors="coerce").fillna(900)
    selected["transport_cost"] = selected["distance_km"] * selected["cost_per_km"]

    result = selected[
        [
            "dc_id",
            "dc_name",
            "retailer_id",
            "retailer_name",
            "distance_km",
            "travel_time_min",
            "cost_per_km",
            "transport_cost",
        ]
    ].copy()

    if result.empty:
        return result, result

    best_dc_by_retailer = (
        result.sort_values("transport_cost")
        .groupby("retailer_id")
        .first()
        .reset_index()
    )

    return result, best_dc_by_retailer
