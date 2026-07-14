import math
import pandas as pd


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def get_first_existing_value(row, columns, default=None):
    for col in columns:
        if col in row.index:
            value = row.get(col)
            if not pd.isna(value):
                return value
    return default


def get_transfer_cost(row):
    recommended_path = str(row.get("recommended_path", ""))

    if "직접" in recommended_path:
        return safe_float(row.get("direct_cost", row.get("transfer_cost", row.get("estimated_cost", 0))))

    if "DC" in recommended_path or "경유" in recommended_path:
        return safe_float(row.get("via_cost", row.get("transfer_cost", row.get("estimated_cost", 0))))

    return safe_float(
        get_first_existing_value(
            row,
            ["transfer_cost", "network_cost", "estimated_cost", "direct_cost", "via_cost"],
            default=0,
        )
    )


def _build_inventory_lookup(inventory, stores):
    if inventory is None or inventory.empty:
        return {}

    inv = inventory.copy()

    if "store_name" not in inv.columns and stores is not None and not stores.empty:
        if "store_id" in inv.columns and "store_id" in stores.columns and "store_name" in stores.columns:
            inv = inv.merge(
                stores[["store_id", "store_name"]].drop_duplicates("store_id"),
                on="store_id",
                how="left",
            )

    if "product_name" not in inv.columns and "inventory_product_name" in inv.columns:
        inv["product_name"] = inv["inventory_product_name"]

    lookup = {}

    for row in inv.itertuples(index=False):
        r = row._asdict()
        store_name = str(r.get("store_name", "")).strip()
        product_name = str(r.get("product_name", "")).strip()

        if not store_name or not product_name:
            continue

        lookup.setdefault((store_name, product_name), r)

    return lookup


def _source_inventory_from_lookup(lookup, source_store, product_name):
    return lookup.get((str(source_store).strip(), str(product_name).strip()))


def estimate_unit_cost(source_inv, transfer_row):
    columns = ["unit_cost", "cost", "product_cost", "item_cost", "price", "unit_price"]

    if source_inv is not None:
        for col in columns:
            if col in source_inv and not pd.isna(source_inv.get(col)):
                return safe_float(source_inv.get(col), 1000)

    for col in columns:
        if col in transfer_row.index and not pd.isna(transfer_row.get(col)):
            return safe_float(transfer_row.get(col), 1000)

    return 1000.0


def estimate_daily_holding_cost(source_inv, transfer_row):
    columns = ["daily_holding_cost", "holding_cost", "storage_cost", "daily_storage_cost"]

    if source_inv is not None:
        for col in columns:
            if col in source_inv and not pd.isna(source_inv.get(col)):
                return safe_float(source_inv.get(col), 20)

    for col in columns:
        if col in transfer_row.index and not pd.isna(transfer_row.get(col)):
            return safe_float(transfer_row.get(col), 20)

    return 20.0


def calculate_promotion_cost(
    promotion_type,
    suggested_qty,
    unit_cost,
    daily_holding_cost,
    promotion_discount_rate,
    promotion_sales_increase_rate,
    promotion_fixed_cost,
):
    suggested_qty = safe_int(suggested_qty, 0)
    unit_cost = safe_float(unit_cost, 1000)
    daily_holding_cost = safe_float(daily_holding_cost, 20)

    discount_rate = safe_float(promotion_discount_rate, 0) / 100
    sales_increase_rate = safe_float(promotion_sales_increase_rate, 0) / 100
    fixed_cost = safe_float(promotion_fixed_cost, 0)

    expected_extra_sales = suggested_qty * sales_increase_rate
    holding_saving = expected_extra_sales * daily_holding_cost

    if promotion_type == "1+1 프로모션":
        promotion_loss = math.ceil(suggested_qty / 2) * unit_cost
        promotion_net_cost = promotion_loss + fixed_cost - holding_saving
        formula = (
            f"1+1 프로모션 순비용 = "
            f"증정 원가({math.ceil(suggested_qty / 2)}개 × {unit_cost:,.0f}원) "
            f"+ 고정비({fixed_cost:,.0f}원) "
            f"- 보관비 절감({holding_saving:,.0f}원) "
            f"= {promotion_net_cost:,.0f}원"
        )
    else:
        discount_loss = suggested_qty * unit_cost * discount_rate
        promotion_net_cost = discount_loss + fixed_cost - holding_saving
        formula = (
            f"할인 프로모션 순비용 = "
            f"할인 손실({suggested_qty}개 × {unit_cost:,.0f}원 × {discount_rate * 100:.1f}%) "
            f"+ 고정비({fixed_cost:,.0f}원) "
            f"- 보관비 절감({holding_saving:,.0f}원) "
            f"= {promotion_net_cost:,.0f}원"
        )

    return max(promotion_net_cost, 0), formula


def analyze_promotion_vs_transfer(
    stores,
    inventory,
    transfer_path_result,
    promotion_type,
    promotion_discount_rate,
    promotion_sales_increase_rate,
    promotion_fixed_cost,
):
    """
    속도 개선판.
    기존에는 transfer 후보마다 inventory를 다시 검색해서 느렸음.
    이제는 inventory lookup을 한 번만 만들어서 바로 찾음.
    """

    if transfer_path_result is None or transfer_path_result.empty:
        return pd.DataFrame()

    transfer_df = transfer_path_result.copy().head(500)
    inventory_lookup = _build_inventory_lookup(inventory, stores)

    rows = []

    for row in transfer_df.itertuples(index=False):
        r = pd.Series(row._asdict())

        recommended_path = str(r.get("recommended_path", ""))
        if recommended_path == "이동 비추천":
            continue

        product_name = r.get("product_name", "-")
        source_store = r.get("source_store", "-")
        target_store = r.get("target_store", "-")

        suggested_qty = safe_int(
            get_first_existing_value(
                r,
                ["suggested_transfer_qty", "suggested_qty", "transfer_qty", "qty"],
                default=0,
            ),
            default=0,
        )

        transfer_cost = get_transfer_cost(r)
        source_inv = _source_inventory_from_lookup(inventory_lookup, source_store, product_name)

        unit_cost = estimate_unit_cost(source_inv, r)
        daily_holding_cost = estimate_daily_holding_cost(source_inv, r)

        promotion_net_cost, promotion_formula = calculate_promotion_cost(
            promotion_type=promotion_type,
            suggested_qty=suggested_qty,
            unit_cost=unit_cost,
            daily_holding_cost=daily_holding_cost,
            promotion_discount_rate=promotion_discount_rate,
            promotion_sales_increase_rate=promotion_sales_increase_rate,
            promotion_fixed_cost=promotion_fixed_cost,
        )

        if transfer_cost <= promotion_net_cost:
            final_decision = "재배치 추천"
            decision_reason = "프로모션 순비용이 재배치 비용보다 높습니다."
        else:
            final_decision = "프로모션 추천"
            decision_reason = "프로모션 순비용이 재배치 비용보다 낮습니다."

        rows.append(
            {
                "product_name": product_name,
                "source_store": source_store,
                "target_store": target_store,
                "suggested_qty": suggested_qty,
                "recommended_transfer_path": recommended_path,
                "transfer_cost": round(transfer_cost, 0),
                "promotion_type": promotion_type,
                "promotion_net_cost": round(promotion_net_cost, 0),
                "final_decision": final_decision,
                "decision_reason": decision_reason,
                "promotion_formula": promotion_formula,
                "unit_cost_used": round(unit_cost, 0),
                "daily_holding_cost_used": round(daily_holding_cost, 0),
            }
        )

    return pd.DataFrame(rows)
