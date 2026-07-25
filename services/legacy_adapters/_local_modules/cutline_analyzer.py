
import pandas as pd
import numpy as np


def _safe_numeric(series, default=0):
    try:
        return pd.to_numeric(series, errors="coerce").fillna(default)
    except Exception:
        return pd.Series([default] * len(series))


def _first_existing_col(df, candidates, default=None):
    for col in candidates:
        if col in df.columns:
            return col
    return default


def _coalesce_columns(df, target_col, candidates, default=None):
    """
    merge 후 product_name_x, product_name_y처럼 컬럼명이 바뀌어도
    하나의 target_col로 안전하게 통합한다.
    """
    existing = [c for c in candidates if c in df.columns]

    if target_col in df.columns:
        existing = [target_col] + [c for c in existing if c != target_col]

    if not existing:
        df[target_col] = default
        return df

    base = df[existing[0]]

    for col in existing[1:]:
        base = base.where(base.notna(), df[col])

    df[target_col] = base.fillna(default)
    return df


def _get_single_series(df, col, default=None):
    """
    df[col]이 Series가 아니라 DataFrame으로 나오는 경우가 있다.
    원인은 merge 과정에서 같은 이름의 컬럼이 중복으로 생긴 경우다.
    이 함수는 중복 컬럼을 하나의 Series로 합쳐서 반환한다.
    """
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)

    value = df[col]

    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            return pd.Series([default] * len(df), index=df.index)

        merged = value.bfill(axis=1).iloc[:, 0]
        return merged.fillna(default)

    return value.fillna(default)


def _dedupe_duplicate_columns(df):
    """
    DataFrame 안에 같은 이름의 컬럼이 여러 개 있을 때 하나로 합친다.
    예: transport_cost 컬럼이 2개 이상이면 첫 번째 non-null 값을 사용한다.
    """
    if not df.columns.duplicated().any():
        return df

    new_df = pd.DataFrame(index=df.index)

    for col in dict.fromkeys(df.columns):
        value = df.loc[:, df.columns == col]

        if isinstance(value, pd.DataFrame) and value.shape[1] > 1:
            new_df[col] = value.bfill(axis=1).iloc[:, 0]
        elif isinstance(value, pd.DataFrame):
            new_df[col] = value.iloc[:, 0]
        else:
            new_df[col] = value

    return new_df


def _identify_input_frames(args, kwargs):
    """
    기존 app.py에서 인자 순서가 조금 달라도 동작하도록 DataFrame을 자동 식별한다.
    """
    inventory = kwargs.get("inventory")
    products = kwargs.get("products")
    routes = (
        kwargs.get("dc_routes")
        if kwargs.get("dc_routes") is not None
        else kwargs.get("routes")
    )

    for df in args:
        if not isinstance(df, pd.DataFrame):
            continue

        cols = set(df.columns)

        if inventory is None and (
            "inventory_id" in cols
            or ("store_id" in cols and "product_id" in cols and ("quantity" in cols or "current_stock" in cols or "stock_qty" in cols))
        ):
            inventory = df
            continue

        if products is None and (
            "product_id" in cols
            and (
                "distance_cutline_km" in cols
                or "product_name" in cols
                or "sales_30" in cols
            )
            and "store_id" not in cols
        ):
            products = df
            continue

        if routes is None and (
            "distance_km" in cols
            or "route_distance_km" in cols
            or "min_distance_km" in cols
            or "transport_cost" in cols
            or "from_id" in cols
            or "to_id" in cols
            or "retailer_id" in cols
        ):
            routes = df
            continue

    return inventory, products, routes


def _prepare_products(products):
    products = products.copy()

    if "product_id" not in products.columns:
        products["product_id"] = products.index.astype(str)

    products = _coalesce_columns(
        products,
        "product_name",
        ["product_name", "product_name_x", "product_name_y", "inventory_product_name"],
        "상품명 없음",
    )

    products = _coalesce_columns(
        products,
        "category",
        ["category", "category_x", "category_y", "inventory_category"],
        "기타",
    )

    # sales_30 / sales_30d 호환 처리
    if "sales_30" not in products.columns and "sales_30d" in products.columns:
        products["sales_30"] = products["sales_30d"]

    if "sales_30d" not in products.columns and "sales_30" in products.columns:
        products["sales_30d"] = products["sales_30"]

    if "sales_30" not in products.columns:
        # 수요 정보가 없으면 0으로 채운다.
        products["sales_30"] = 0

    if "sales_30d" not in products.columns:
        products["sales_30d"] = products["sales_30"]

    if "distance_cutline_km" not in products.columns:
        # 상품별 이동 제한 거리가 없으면 기본 10km로 둔다.
        products["distance_cutline_km"] = 10

    if "unit_price" not in products.columns:
        products["unit_price"] = 0

    if "disposal_cost_per_unit" not in products.columns:
        products["disposal_cost_per_unit"] = 300

    keep_cols = [
        "product_id",
        "product_name",
        "category",
        "sales_30",
        "sales_30d",
        "distance_cutline_km",
        "unit_price",
        "disposal_cost_per_unit",
    ]

    return products[[c for c in keep_cols if c in products.columns]].drop_duplicates("product_id")


def _prepare_inventory(inventory):
    inventory = inventory.copy()

    if "store_id" not in inventory.columns:
        inventory["store_id"] = "UNKNOWN_STORE"

    if "product_id" not in inventory.columns:
        inventory["product_id"] = "UNKNOWN_PRODUCT"

    if "store_name" not in inventory.columns:
        inventory["store_name"] = inventory["store_id"].astype(str)

    quantity_col = _first_existing_col(
        inventory,
        ["quantity", "current_stock", "stock_qty", "inventory_qty", "qty"],
        None,
    )

    if quantity_col is None:
        inventory["quantity"] = 0
    elif quantity_col != "quantity":
        inventory["quantity"] = inventory[quantity_col]

    if "current_stock" not in inventory.columns:
        inventory["current_stock"] = inventory["quantity"]

    if "dead_stock_qty" not in inventory.columns:
        inventory["dead_stock_qty"] = 0

    if "demand_qty" not in inventory.columns:
        inventory["demand_qty"] = 0

    if "days_to_expiry" not in inventory.columns:
        inventory["days_to_expiry"] = np.nan

    keep_cols = [
        "inventory_id",
        "store_id",
        "store_name",
        "product_id",
        "quantity",
        "current_stock",
        "dead_stock_qty",
        "demand_qty",
        "days_to_expiry",
    ]

    existing = [c for c in keep_cols if c in inventory.columns]
    return inventory[existing].copy()


def _prepare_store_routes(routes):
    """
    큰 routes 데이터도 빠르게 처리하기 위해 store별 최단 경로 1개만 사용한다.
    이러면 inventory × routes 전체 조합을 만들지 않아 속도가 훨씬 안정적이다.
    """
    if routes is None or routes.empty:
        return pd.DataFrame(columns=[
            "store_id",
            "route_distance_km",
            "transport_cost",
            "route_time_min",
            "route_source",
            "route_target",
        ])

    routes = routes.copy()

    # 거리 컬럼 탐색
    distance_col = _first_existing_col(
        routes,
        [
            "route_distance_km",
            "min_distance_km",
            "distance_km",
            "direct_distance_km",
            "network_distance_km",
            "total_distance_km",
        ],
        None,
    )

    if distance_col is None:
        routes["route_distance_km"] = 999999
        distance_col = "route_distance_km"

    # 비용 컬럼 탐색
    cost_col = _first_existing_col(
        routes,
        [
            "transport_cost",
            "estimated_cost",
            "min_transport_cost",
            "direct_cost",
            "transfer_cost",
            "network_cost",
        ],
        None,
    )

    if cost_col is None:
        routes["transport_cost"] = _safe_numeric(routes[distance_col], 0) * 900
        cost_col = "transport_cost"

    # 시간 컬럼 탐색
    time_col = _first_existing_col(
        routes,
        ["travel_time_min", "route_time_min", "time_min", "network_time_min"],
        None,
    )

    if time_col is None:
        routes["route_time_min"] = _safe_numeric(routes[distance_col], 0) / 30 * 60
        time_col = "route_time_min"

    # store id 식별
    if "retailer_id" in routes.columns:
        store_id_col = "retailer_id"
        route_target_col = _first_existing_col(routes, ["retailer_name", "target_store", "to_name"], "retailer_id")
        route_source_col = _first_existing_col(routes, ["dc_name", "source_store", "from_name"], None)
    elif "to_id" in routes.columns:
        # DC_TO_STORE가 있으면 그것만 우선 사용
        if "route_type" in routes.columns:
            dc_to_store = routes[routes["route_type"].astype(str).str.contains("DC_TO_STORE", case=False, na=False)]

            if not dc_to_store.empty:
                routes = dc_to_store.copy()

        store_id_col = "to_id"
        route_target_col = _first_existing_col(routes, ["to_name", "target_store"], "to_id")
        route_source_col = _first_existing_col(routes, ["from_name", "source_store"], None)
    elif "target_store_id" in routes.columns:
        store_id_col = "target_store_id"
        route_target_col = _first_existing_col(routes, ["target_store", "target_store_name"], "target_store_id")
        route_source_col = _first_existing_col(routes, ["source_store", "source_store_name"], None)
    else:
        # store_id를 못 찾으면 전체 평균 거리 하나로 처리
        return pd.DataFrame(
            {
                "store_id": [],
                "route_distance_km": [],
                "transport_cost": [],
                "route_time_min": [],
                "route_source": [],
                "route_target": [],
            }
        )

    routes["_distance"] = _safe_numeric(routes[distance_col], 999999)
    routes["_cost"] = _safe_numeric(routes[cost_col], 0)
    routes["_time"] = _safe_numeric(routes[time_col], 0)
    routes["_store_id"] = routes[store_id_col].astype(str)

    if route_source_col and route_source_col in routes.columns:
        routes["_route_source"] = routes[route_source_col].astype(str)
    else:
        routes["_route_source"] = "-"

    if route_target_col and route_target_col in routes.columns:
        routes["_route_target"] = routes[route_target_col].astype(str)
    else:
        routes["_route_target"] = routes["_store_id"]

    # store별 최단 경로 선택
    idx = routes.groupby("_store_id")["_distance"].idxmin()
    best = routes.loc[idx].copy()

    return best.rename(
        columns={
            "_store_id": "store_id",
            "_distance": "route_distance_km",
            "_cost": "transport_cost",
            "_time": "route_time_min",
            "_route_source": "route_source",
            "_route_target": "route_target",
        }
    )[[
        "store_id",
        "route_distance_km",
        "transport_cost",
        "route_time_min",
        "route_source",
        "route_target",
    ]]


def analyze_product_distance_cutline(*args, **kwargs):
    """
    상품별 거리 컷라인 판별 함수.

    역할:
    - products의 distance_cutline_km를 기준으로 상품별 허용 이동 거리를 확인한다.
    - routes/dc_routes에서 점포별 최단 이동거리를 구한다.
    - inventory와 products를 product_id 기준으로 연결한다.
    - 실제 이동거리 <= 상품 컷라인이면 이동 가능으로 판단한다.

    반환:
    1. cutline_result: 전체 판별 결과
    2. best_valid_routes: 이동 가능한 후보
    3. no_valid_items: 이동 불가능/경로 없음 후보
    """
    inventory, products, routes = _identify_input_frames(args, kwargs)

    if inventory is None or products is None:
        empty = pd.DataFrame()
        return empty, empty, empty

    inventory = _prepare_inventory(inventory)
    products = _prepare_products(products)
    store_routes = _prepare_store_routes(routes)

    # inventory와 products를 병합
    result = inventory.merge(
        products,
        on="product_id",
        how="left",
        suffixes=("", "_product"),
    )

    # merge 과정에서 같은 이름의 컬럼이 중복되면 하나로 정리
    result = _dedupe_duplicate_columns(result)

    # merge 후에도 안전하게 컬럼 복구
    result = _coalesce_columns(
        result,
        "product_name",
        ["product_name", "product_name_x", "product_name_y", "inventory_product_name"],
        "상품명 없음",
    )

    result = _coalesce_columns(
        result,
        "category",
        ["category", "category_x", "category_y", "inventory_category"],
        "기타",
    )

    if "sales_30" not in result.columns and "sales_30d" in result.columns:
        result["sales_30"] = result["sales_30d"]

    if "sales_30d" not in result.columns and "sales_30" in result.columns:
        result["sales_30d"] = result["sales_30"]

    if "sales_30" not in result.columns:
        result["sales_30"] = 0

    if "sales_30d" not in result.columns:
        result["sales_30d"] = result["sales_30"]

    if "distance_cutline_km" not in result.columns:
        result["distance_cutline_km"] = 10

    # store별 최단 경로 병합
    if not store_routes.empty:
        result["store_id"] = result["store_id"].astype(str)
        store_routes["store_id"] = store_routes["store_id"].astype(str)

        result = result.merge(
            store_routes,
            on="store_id",
            how="left",
        )

        # routes 병합 후 transport_cost 같은 컬럼이 중복될 수 있으므로 정리
        result = _dedupe_duplicate_columns(result)
    else:
        result["route_distance_km"] = np.nan
        result["transport_cost"] = np.nan
        result["route_time_min"] = np.nan
        result["route_source"] = "-"
        result["route_target"] = result["store_name"]

    result["distance_cutline_km"] = _safe_numeric(
        _get_single_series(result, "distance_cutline_km", 10),
        10,
    )
    result["route_distance_km"] = _safe_numeric(
        _get_single_series(result, "route_distance_km", np.nan),
        np.nan,
    )
    result["transport_cost"] = _safe_numeric(
        _get_single_series(result, "transport_cost", np.nan),
        np.nan,
    )
    result["route_time_min"] = _safe_numeric(
        _get_single_series(result, "route_time_min", np.nan),
        np.nan,
    )

    result["is_within_cutline"] = (
        result["route_distance_km"].notna()
        & (result["route_distance_km"] <= result["distance_cutline_km"])
    )

    result["cutline_status"] = np.select(
        [
            result["route_distance_km"].isna(),
            result["is_within_cutline"],
        ],
        [
            "경로 없음",
            "이동 가능",
        ],
        default="거리 초과",
    )

    result["cutline_reason"] = np.where(
        result["cutline_status"] == "이동 가능",
        "상품별 거리 컷라인 이내이므로 이동 가능",
        np.where(
            result["cutline_status"] == "거리 초과",
            "상품별 허용 이동거리보다 실제 이동거리가 길어 이동 제한",
            "해당 점포와 연결된 경로 정보가 없어 이동 판단 불가",
        ),
    )

    # 기존 코드 호환용 컬럼명도 함께 제공
    result["distance_km"] = _get_single_series(result, "route_distance_km", np.nan)
    result["estimated_cost"] = _get_single_series(result, "transport_cost", np.nan)
    result["source_store"] = _get_single_series(result, "route_source", "-")
    result["target_store"] = _get_single_series(result, "store_name", "-")
    result["suggested_qty"] = _get_single_series(result, "dead_stock_qty", 0)

    output_cols = [
        "inventory_id",
        "store_id",
        "store_name",
        "product_id",
        "product_name",
        "category",
        "quantity",
        "current_stock",
        "dead_stock_qty",
        "demand_qty",
        "sales_30",
        "sales_30d",
        "distance_cutline_km",
        "route_distance_km",
        "distance_km",
        "transport_cost",
        "estimated_cost",
        "route_time_min",
        "source_store",
        "target_store",
        "suggested_qty",
        "is_within_cutline",
        "cutline_status",
        "cutline_reason",
    ]

    result = _dedupe_duplicate_columns(result)
    result = result[[c for c in output_cols if c in result.columns]].copy()
    result = _dedupe_duplicate_columns(result)

    # 이동 가능한 후보
    best_valid_routes = result[result["cutline_status"] == "이동 가능"].copy()

    if not best_valid_routes.empty:
        if "dead_stock_qty" in best_valid_routes.columns:
            best_valid_routes = best_valid_routes.sort_values(
                ["dead_stock_qty", "transport_cost"],
                ascending=[False, True],
                na_position="last",
            )
        else:
            best_valid_routes = best_valid_routes.sort_values(
                ["transport_cost"],
                ascending=[True],
                na_position="last",
            )

    no_valid_items = result[result["cutline_status"] != "이동 가능"].copy()

    return (
        result.reset_index(drop=True),
        best_valid_routes.reset_index(drop=True),
        no_valid_items.reset_index(drop=True),
    )
