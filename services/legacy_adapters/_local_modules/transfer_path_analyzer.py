import math
import pandas as pd
import numpy as np


def _num(value, default=0.0):
    try:
        if isinstance(value, pd.Series):
            return pd.to_numeric(value, errors="coerce").fillna(default)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _first_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _ensure_id_text(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except Exception:
        return np.nan

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _prepare_stores(stores):
    if stores is None or stores.empty:
        return pd.DataFrame(columns=["store_id", "store_name", "type", "latitude", "longitude"])

    df = stores.copy()
    if "store_id" not in df.columns:
        df["store_id"] = df.index.astype(str)
    if "store_name" not in df.columns:
        df["store_name"] = df["store_id"].astype(str)
    if "type" not in df.columns:
        df["type"] = "store"
    if "latitude" not in df.columns:
        df["latitude"] = np.nan
    if "longitude" not in df.columns:
        df["longitude"] = np.nan

    return _ensure_id_text(df, ["store_id"])


def _prepare_products(products):
    if products is None or products.empty:
        return pd.DataFrame(columns=["product_id"])

    df = products.copy()
    if "product_id" not in df.columns:
        df["product_id"] = df.index.astype(str)
    if "product_name" not in df.columns:
        df["product_name"] = df["product_id"]
    if "category" not in df.columns:
        df["category"] = "기타"
    if "sales_30d" not in df.columns and "sales_30" in df.columns:
        df["sales_30d"] = df["sales_30"]
    if "sales_30" not in df.columns and "sales_30d" in df.columns:
        df["sales_30"] = df["sales_30d"]
    if "sales_30d" not in df.columns:
        df["sales_30d"] = 0
    if "sales_30" not in df.columns:
        df["sales_30"] = df["sales_30d"]
    if "unit_cost" not in df.columns and "unit_price" in df.columns:
        df["unit_cost"] = df["unit_price"]
    if "unit_cost" not in df.columns:
        df["unit_cost"] = 1000
    if "unit_price" not in df.columns:
        df["unit_price"] = df["unit_cost"]

    keep = [
        "product_id", "product_name", "category", "sales_30", "sales_30d",
        "unit_cost", "unit_price", "distance_cutline_km", "cold_required",
        "disposal_cost_per_unit"
    ]
    df = df[[c for c in keep if c in df.columns]].drop_duplicates("product_id")
    return _ensure_id_text(df, ["product_id"])


def _prepare_inventory(inventory, stores, products):
    if inventory is None or inventory.empty:
        return pd.DataFrame()

    inv = inventory.copy()
    if "store_id" not in inv.columns:
        inv["store_id"] = "UNKNOWN_STORE"
    if "product_id" not in inv.columns:
        inv["product_id"] = "UNKNOWN_PRODUCT"

    inv = _ensure_id_text(inv, ["store_id", "product_id"])
    stores = _prepare_stores(stores)
    products = _prepare_products(products)

    # inventory에 이미 이름이 있어도, stores/products 기준으로 보강
    if "store_name" not in inv.columns and not stores.empty:
        inv = inv.merge(stores[["store_id", "store_name"]].drop_duplicates("store_id"), on="store_id", how="left")
    elif "store_name" in inv.columns and not stores.empty:
        name_map = stores.set_index("store_id")["store_name"].to_dict()
        inv["store_name"] = inv["store_id"].map(name_map).fillna(inv["store_name"])

    if "store_name" not in inv.columns:
        inv["store_name"] = inv["store_id"]

    if not products.empty:
        prod_cols = [c for c in products.columns if c != "product_id"]
        # 중복 컬럼은 inventory 값을 유지하고, 없는 것만 products에서 가져옴
        missing_cols = [c for c in prod_cols if c not in inv.columns]
        if missing_cols:
            inv = inv.merge(products[["product_id"] + missing_cols], on="product_id", how="left")

    if "product_name" not in inv.columns:
        if "inventory_product_name" in inv.columns:
            inv["product_name"] = inv["inventory_product_name"]
        else:
            inv["product_name"] = inv["product_id"]

    if "category" not in inv.columns:
        if "inventory_category" in inv.columns:
            inv["category"] = inv["inventory_category"]
        else:
            inv["category"] = "기타"

    qty_col = _first_col(inv, ["stock_qty", "current_stock", "quantity", "inventory_qty", "qty"], None)
    inv["stock_qty"] = _num(inv[qty_col], 0) if qty_col else 0

    if "current_stock" not in inv.columns:
        inv["current_stock"] = inv["stock_qty"]

    if "sales_30d" not in inv.columns and "sales_30" in inv.columns:
        inv["sales_30d"] = inv["sales_30"]
    if "sales_30" not in inv.columns and "sales_30d" in inv.columns:
        inv["sales_30"] = inv["sales_30d"]

    if "sales_30d" not in inv.columns:
        if "demand_qty" in inv.columns:
            inv["sales_30d"] = _num(inv["demand_qty"], 0) * 6
        elif "avg_daily_sales" in inv.columns:
            inv["sales_30d"] = _num(inv["avg_daily_sales"], 0) * 30
        else:
            inv["sales_30d"] = 0
    if "sales_30" not in inv.columns:
        inv["sales_30"] = inv["sales_30d"]

    inv["sales_30d"] = _num(inv["sales_30d"], 0)
    inv["sales_30"] = _num(inv["sales_30"], 0)

    if "dead_stock_qty" not in inv.columns:
        inv["dead_stock_qty"] = np.maximum(inv["stock_qty"] - inv["sales_30d"], 0)
    else:
        inv["dead_stock_qty"] = _num(inv["dead_stock_qty"], 0)

    if "demand_qty" not in inv.columns:
        inv["demand_qty"] = np.maximum(inv["sales_30d"] - inv["stock_qty"], 0)
    else:
        inv["demand_qty"] = _num(inv["demand_qty"], 0)

    if "unit_cost" not in inv.columns:
        inv["unit_cost"] = 1000
    if "unit_price" not in inv.columns:
        inv["unit_price"] = inv["unit_cost"]
    if "disposal_cost_per_unit" not in inv.columns:
        inv["disposal_cost_per_unit"] = 300

    return inv


def _prepare_routes(routes):
    if routes is None or routes.empty:
        return pd.DataFrame(columns=["from_id", "to_id", "_distance_km", "_time_min", "_transport_cost"])

    df = routes.copy()
    if "from_id" not in df.columns and "source_id" in df.columns:
        df["from_id"] = df["source_id"]
    if "to_id" not in df.columns and "target_id" in df.columns:
        df["to_id"] = df["target_id"]

    if "from_id" not in df.columns or "to_id" not in df.columns:
        return pd.DataFrame(columns=["from_id", "to_id", "_distance_km", "_time_min", "_transport_cost"])

    distance_col = _first_col(df, ["distance_km", "route_distance_km", "direct_distance_km"], None)
    time_col = _first_col(df, ["travel_time_min", "route_time_min", "time_min", "duration_min"], None)
    cost_col = _first_col(df, ["transport_cost", "estimated_cost", "direct_cost", "transfer_cost"], None)

    df = _ensure_id_text(df, ["from_id", "to_id"])
    df["_distance_km"] = _num(df[distance_col], np.nan) if distance_col else np.nan
    df["_time_min"] = _num(df[time_col], np.nan) if time_col else df["_distance_km"] / 30 * 60

    if cost_col:
        df["_transport_cost"] = _num(df[cost_col], np.nan)
    elif "cost_per_km" in df.columns:
        fixed = _num(df["fixed_cost"], 0) if "fixed_cost" in df.columns else 0
        df["_transport_cost"] = fixed + df["_distance_km"] * _num(df["cost_per_km"], 900)
    else:
        df["_transport_cost"] = 1500 + df["_distance_km"] * 900

    return df


def _build_route_lookup(routes):
    lookup = {}
    if routes is None or routes.empty:
        return lookup

    # 같은 from/to가 여러 개면 비용이 낮은 경로를 사용
    sorted_routes = routes.sort_values(["_transport_cost", "_distance_km"], na_position="last")

    for row in sorted_routes.itertuples(index=False):
        r = row._asdict()
        a = str(r.get("from_id", "")).strip()
        b = str(r.get("to_id", "")).strip()
        if not a or not b:
            continue

        value = (
            float(r.get("_distance_km", np.nan)),
            float(r.get("_transport_cost", np.nan)),
            float(r.get("_time_min", np.nan)),
        )

        lookup.setdefault((a, b), value)
        lookup.setdefault((b, a), value)

    return lookup


def _build_store_info(stores):
    info = {}
    for row in stores.itertuples(index=False):
        r = row._asdict()
        store_id = str(r.get("store_id", "")).strip()
        info[store_id] = {
            "name": r.get("store_name", store_id),
            "lat": r.get("latitude", np.nan),
            "lng": r.get("longitude", np.nan),
            "type": r.get("type", ""),
        }
    return info


def _route_cost_distance(route_lookup, store_info, a, b):
    a = str(a).strip()
    b = str(b).strip()

    if (a, b) in route_lookup:
        return route_lookup[(a, b)]

    sa = store_info.get(a)
    sb = store_info.get(b)
    if not sa or not sb:
        return np.nan, np.nan, np.nan

    distance = _haversine_km(sa.get("lat"), sa.get("lng"), sb.get("lat"), sb.get("lng"))
    if pd.isna(distance):
        return np.nan, np.nan, np.nan

    distance = round(distance * 1.35, 2)
    cost = round(1500 + distance * 900, 0)
    time_min = round(distance / 30 * 60, 0)
    return distance, cost, time_min


def _best_dc_via_cost(route_lookup, store_info, dc_ids, source_id, target_id):
    best = None

    for dc_id in dc_ids:
        d1, c1, t1 = _route_cost_distance(route_lookup, store_info, source_id, dc_id)
        d2, c2, t2 = _route_cost_distance(route_lookup, store_info, dc_id, target_id)

        if pd.isna(d1) or pd.isna(d2):
            continue

        cost = (0 if pd.isna(c1) else c1) + (0 if pd.isna(c2) else c2)
        distance = d1 + d2
        time_min = (0 if pd.isna(t1) else t1) + (0 if pd.isna(t2) else t2)

        if best is None or cost < best["cost"]:
            best = {
                "dc_id": dc_id,
                "dc_name": store_info.get(dc_id, {}).get("name", dc_id),
                "distance": distance,
                "cost": cost,
                "time": time_min,
            }

    if best is None:
        return np.nan, np.nan, np.nan, "-"

    return best["distance"], best["cost"], best["time"], best["dc_name"]


def _select_top(group, source_limit=3, target_limit=3):
    source = group[
        (group["dead_stock_qty"] > 0)
        | (group["stock_qty"] > group["sales_30d"])
    ].copy()

    target = group[
        (group["demand_qty"] > 0)
        | (group["stock_qty"] < group["sales_30d"])
    ].copy()

    if source.empty:
        source = group.sort_values("stock_qty", ascending=False).head(source_limit)

    if target.empty:
        target = group.sort_values("stock_qty", ascending=True).head(target_limit)

    source = source.sort_values(
        ["dead_stock_qty", "stock_qty"],
        ascending=[False, False],
        na_position="last",
    ).head(source_limit)

    target = target.sort_values(
        ["demand_qty", "stock_qty"],
        ascending=[False, True],
        na_position="last",
    ).head(target_limit)

    return source, target


def analyze_direct_vs_dc_transfer(*args, **kwargs):
    """
    속도 개선판.
    핵심 변경:
    - routes를 매번 DataFrame 필터링하지 않고 dict lookup으로 조회
    - 상품별 source/target 후보 수를 3개씩으로 제한
    - 최대 결과 수를 500개로 제한해서 뒤쪽 promotion/network 속도까지 같이 개선
    """

    if len(args) >= 4:
        stores, products, inventory, routes = args[:4]
    else:
        stores = kwargs.get("stores")
        products = kwargs.get("products")
        inventory = kwargs.get("inventory") or kwargs.get("candidate_df")
        routes = kwargs.get("routes")

    if inventory is None or not isinstance(inventory, pd.DataFrame) or inventory.empty:
        return pd.DataFrame()

    stores = _prepare_stores(stores)
    products = _prepare_products(products)
    candidates = _prepare_inventory(inventory, stores, products)
    routes = _prepare_routes(routes)

    route_lookup = _build_route_lookup(routes)
    store_info = _build_store_info(stores)

    if not stores.empty:
        dc_ids = stores[
            stores["type"].astype(str).str.upper().str.contains("DC", na=False)
        ]["store_id"].astype(str).tolist()
    else:
        dc_ids = []

    results = []
    max_result_rows = int(kwargs.get("max_result_rows", 500))

    # 점수가 높은 상품/재고부터 처리해서 불필요한 후보 생성을 줄임
    candidates["_speed_priority"] = (
        _num(candidates["dead_stock_qty"], 0) * 2
        + _num(candidates["demand_qty"], 0)
        + _num(candidates["stock_qty"], 0) * 0.05
    )
    candidates = candidates.sort_values("_speed_priority", ascending=False)

    for product_id, group in candidates.groupby("product_id", sort=False):
        if len(results) >= max_result_rows:
            break

        group = group.copy()
        product_name = group["product_name"].iloc[0] if "product_name" in group.columns else str(product_id)
        category = group["category"].iloc[0] if "category" in group.columns else "기타"

        source_candidates, target_candidates = _select_top(group, source_limit=3, target_limit=3)

        for source in source_candidates.itertuples(index=False):
            if len(results) >= max_result_rows:
                break

            s = source._asdict()
            for target in target_candidates.itertuples(index=False):
                t = target._asdict()
                if str(s["store_id"]) == str(t["store_id"]):
                    continue

                source_surplus = max(
                    _num(s.get("dead_stock_qty"), 0),
                    _num(s.get("stock_qty"), 0) - _num(s.get("sales_30d"), 0),
                    0,
                )

                target_shortage = max(
                    _num(t.get("demand_qty"), 0),
                    _num(t.get("sales_30d"), 0) - _num(t.get("stock_qty"), 0),
                    0,
                )

                suggested_qty = int(max(0, min(source_surplus, target_shortage)))
                if suggested_qty <= 0:
                    suggested_qty = int(max(1, min(_num(s.get("stock_qty"), 0), 10)))

                source_id = str(s.get("store_id"))
                target_id = str(t.get("store_id"))
                source_store = s.get("store_name", store_info.get(source_id, {}).get("name", source_id))
                target_store = t.get("store_name", store_info.get(target_id, {}).get("name", target_id))

                direct_distance, direct_cost, direct_time = _route_cost_distance(
                    route_lookup, store_info, source_id, target_id
                )

                via_distance, via_cost, via_time, via_dc_name = _best_dc_via_cost(
                    route_lookup, store_info, dc_ids, source_id, target_id
                )

                if pd.isna(direct_cost) and pd.isna(via_cost):
                    continue

                if pd.isna(via_cost):
                    recommended_path = "직접 이동 추천"
                    estimated_cost = direct_cost
                    recommended_distance = direct_distance
                    recommended_time = direct_time
                    path_reason = "DC 경유 경로가 없어 점포 간 직접 이동을 추천합니다."
                elif pd.isna(direct_cost):
                    recommended_path = "DC 경유 추천"
                    estimated_cost = via_cost
                    recommended_distance = via_distance
                    recommended_time = via_time
                    path_reason = "직접 이동 경로가 없어 DC 경유 이동을 추천합니다."
                elif direct_cost <= via_cost:
                    recommended_path = "직접 이동 추천"
                    estimated_cost = direct_cost
                    recommended_distance = direct_distance
                    recommended_time = direct_time
                    path_reason = "직접 이동 비용이 DC 경유보다 낮아 직접 이동을 추천합니다."
                else:
                    recommended_path = "DC 경유 추천"
                    estimated_cost = via_cost
                    recommended_distance = via_distance
                    recommended_time = via_time
                    path_reason = "DC 경유 비용이 직접 이동보다 낮아 DC 경유를 추천합니다."

                results.append(
                    {
                        "product_id": product_id,
                        "product_name": product_name,
                        "category": category,
                        "source_store_id": source_id,
                        "target_store_id": target_id,
                        "source_store": source_store,
                        "target_store": target_store,
                        "source_stock_qty": _num(s.get("stock_qty"), 0),
                        "target_stock_qty": _num(t.get("stock_qty"), 0),
                        "sales_30d": _num(s.get("sales_30d"), 0),
                        "sales_30": _num(s.get("sales_30"), 0),
                        "source_dead_stock_qty": source_surplus,
                        "target_shortage_qty": target_shortage,
                        "suggested_qty": suggested_qty,
                        "direct_distance_km": direct_distance,
                        "direct_cost": direct_cost,
                        "direct_time_min": direct_time,
                        "via_dc_name": via_dc_name,
                        "via_distance_km": via_distance,
                        "via_cost": via_cost,
                        "via_time_min": via_time,
                        "recommended_path": recommended_path,
                        "estimated_cost": estimated_cost,
                        "recommended_distance_km": recommended_distance,
                        "recommended_time_min": recommended_time,
                        "transfer_reason": path_reason,
                        "reason": path_reason,
                    }
                )

    result = pd.DataFrame(results)

    if result.empty:
        return result

    result["estimated_cost"] = pd.to_numeric(result["estimated_cost"], errors="coerce")
    result["suggested_qty"] = pd.to_numeric(result["suggested_qty"], errors="coerce").fillna(0)
    result["cost_per_unit"] = result["estimated_cost"] / result["suggested_qty"].replace(0, np.nan)
    result["cost_per_unit"] = result["cost_per_unit"].fillna(result["estimated_cost"])

    result = result.sort_values(
        ["estimated_cost", "suggested_qty"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    return result
