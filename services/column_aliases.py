"""Column alias mapping and numeric normalization for real Excel uploads.

Maps varied (English/Korean) column names onto Varo V2's standard columns and
coerces messy numeric strings ("10,000원", "3.5km", "15분") into numbers. None of
this reads or produces DQN values.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Iterable

import pandas as pd

# Each map: standard column -> accepted aliases (matched case-insensitively after
# trimming and removing spaces/underscores). The first existing alias wins and is
# copied to the standard name without dropping the original column.
STORE_ALIASES = {
    "node_id": ["node_id", "store_id", "id", "점포id", "점포코드", "노드id", "매장id"],
    "node_name": ["node_name", "store_name", "name", "점포명", "점포이름", "점포", "매장명"],
    # ``store_type`` in the DQN workbooks means trade-area type (주거형,
    # 역세권, 오피스형), not whether the row is a STORE or DC.  Only aliases
    # that explicitly describe a network node are accepted here.  The loader
    # still accepts store_type/type when *all* values are recognised node
    # labels, so older workbooks remain compatible without this collision.
    "node_type": ["node_type", "노드유형"],
    "store_name": ["store_name", "node_name", "점포명", "매장명"],
}
DC_ALIASES = {
    "node_id": ["node_id", "dc_id", "distribution_center_id", "center_id", "id", "물류센터id", "센터id"],
    "node_name": ["node_name", "dc_name", "distribution_center_name", "center_name", "name", "물류센터명", "센터명"],
    "node_type": ["node_type", "dc_type", "type", "유형", "구분"],
    "store_name": ["store_name", "node_name", "dc_name", "name", "물류센터명", "센터명"],
}
PRODUCT_ALIASES = {
    "product_id": ["product_id", "item_id", "id", "상품코드", "상품id", "제품코드"],
    "product_name": ["product_name", "item_name", "name", "상품명", "상품이름", "상품", "제품명"],
    "unit_price": ["unit_price", "price", "단가", "판매가", "가격"],
    "disposal_cost_per_unit": ["disposal_cost_per_unit", "disposal_cost", "폐기비용", "단위폐기비용"],
}
INVENTORY_ALIASES = {
    "store_id": ["store_id", "node_id", "점포id", "점포코드", "매장id"],
    "store_name": ["store_name", "node_name", "점포명", "매장명"],
    "product_id": ["product_id", "item_id", "상품코드", "상품id"],
    "product_name": ["product_name", "item_name", "상품명", "상품", "제품명"],
    "stock_qty": ["stock_qty", "current_stock", "quantity", "inventory_qty", "stock", "재고수량", "재고", "현재고"],
    "dead_stock_qty": ["dead_stock_qty", "악성재고", "불용재고"],
    "sales_qty": ["sales_qty", "avg_daily_sales", "sales", "일판매량", "판매량"],
    "avg_daily_sales": ["avg_daily_sales", "sales_qty", "일평균판매량", "일판매량"],
    "demand_qty": ["demand_qty", "수요량", "수요"],
    "sales_30d": ["sales_30d", "sales_30", "월판매량", "30일판매량"],
    "sales_7d": ["sales_7d", "주판매량", "7일판매량"],
    "expiry_days": ["expiry_days", "shelf_life_days", "유통기한", "유통기한일수"],
    "days_to_expiry": ["days_to_expiry", "남은일수", "유통기한남은일수", "잔여유통기한"],
    "unit_price": ["unit_price", "price", "단가", "가격"],
    "order_cost": ["order_cost", "주문비용", "발주비용"],
    "demand_std": ["demand_std", "수요표준편차"],
    "lead_time_days": ["lead_time_days", "리드타임", "조달기간"],
    "disposal_cost_per_unit": ["disposal_cost_per_unit", "disposal_cost", "폐기비용"],
}
ROUTE_ALIASES = {
    "source_id": ["source_id", "source", "from_id", "from_store_id", "from_store", "출발점포id", "출발id", "보내는점포id"],
    "target_id": ["target_id", "target", "to_id", "to_store_id", "to_store", "도착점포id", "도착id", "받는점포id"],
    "distance_km": ["distance_km", "route_distance_km", "direct_distance_km", "distance", "이동거리", "거리"],
    "estimated_cost": ["estimated_cost", "transport_cost", "direct_cost", "cost", "운송비", "이동비용"],
    "travel_time_min": ["travel_time_min", "route_time_min", "time_min", "time", "이동시간", "소요시간"],
}
RECOMMENDATION_ALIASES = {
    "route_id": ["route_id", "recommendation_id", "추천id", "경로id"],
    "product_id": ["product_id", "item_id", "상품코드", "상품id"],
    "product_name": ["product_name", "item_name", "상품명", "상품", "제품명"],
    "source_id": ["source_id", "source", "source_store_id", "from_id", "from_store_id", "출발점포id", "출발id", "보내는점포id"],
    "source_name": ["source_name", "source_store_name", "source_store", "from_store_name", "from_store", "출발점포", "보내는점포"],
    "target_id": ["target_id", "target", "target_store_id", "to_id", "to_store_id", "도착점포id", "도착id", "받는점포id"],
    "target_name": ["target_name", "target_store_name", "target_store", "to_store_name", "to_store", "도착점포", "받는점포"],
    "dc_id": ["dc_id", "물류센터id", "dcid", "센터id"],
    "dc_name": ["dc_name", "via_dc_name", "물류센터", "dc", "센터명"],
    "route_type": ["route_type", "경로유형", "경로구분", "유형"],
    "recommended_qty": ["recommended_qty", "suggested_qty", "transfer_qty", "move_qty", "quantity", "qty", "추천수량", "이동수량"],
    "transport_type": ["transport_type", "transport_mode", "이동수단", "운송수단"],
    "estimated_cost": ["estimated_cost", "transport_cost", "cost", "이동비용", "운송비"],
    "expected_saving": ["expected_saving", "expected_savings", "expected_savings_amount", "saving_amount", "savings_amount", "saving", "savings", "예상절감액", "절감액"],
    "vhs_score": ["vhs_score", "vhs", "uploaded_vhs", "기존vhs", "업로드vhs", "vhs점수"],
    "recommendation_grade": ["recommendation_grade", "grade", "vhs_grade", "추천등급", "등급"],
    "confidence_score": ["confidence_score", "confidence", "신뢰도", "신뢰도점수"],
    "confidence_grade": ["confidence_grade", "신뢰도등급"],
    "reason": ["reason", "vhs_reason", "transfer_reason", "추천사유", "사유"],
    "distance_km": ["distance_km", "이동거리", "거리"],
    "travel_time_min": ["travel_time_min", "이동시간", "소요시간"],
}

SHEET_ALIASES = {
    "stores": STORE_ALIASES,
    "dcs": DC_ALIASES,
    "products": PRODUCT_ALIASES,
    "inventory": INVENTORY_ALIASES,
    "routes": ROUTE_ALIASES,
    "recommendations": RECOMMENDATION_ALIASES,
}

NUMERIC_COLUMNS = {
    "products": ("unit_price", "disposal_cost_per_unit"),
    "inventory": (
        "stock_qty", "dead_stock_qty", "sales_qty", "avg_daily_sales", "demand_qty",
        "sales_30d", "sales_7d", "expiry_days", "days_to_expiry", "unit_price",
        "order_cost", "demand_std", "lead_time_days", "disposal_cost_per_unit",
    ),
    "routes": ("distance_km", "estimated_cost", "travel_time_min"),
    "recommendations": (
        "recommended_qty", "estimated_cost", "expected_saving", "vhs_score",
        "confidence_score", "distance_km", "travel_time_min",
    ),
}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _canonical(name: object) -> str:
    return re.sub(r"[\s_]+", "", str(name).strip().lower())


def clean_numeric_value(value: object) -> float | None:
    """Coerce a messy numeric string into a float.

    "10,000원" -> 10000.0, "3.5km" -> 3.5, "15분" -> 15.0, "" -> None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return None if value != value else float(value)  # drop NaN
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "n/a", "na"}:
        return None
    text = text.replace(",", "")
    match = _NUMBER_RE.search(text)
    return float(match.group()) if match else None


def coerce_numeric_columns(
    df: pd.DataFrame, columns: Iterable[str]
) -> tuple[pd.DataFrame, int]:
    """Coerce the given columns to numeric; return (df, failed_value_count)."""
    if df is None or df.empty:
        return df, 0
    result = df.copy()
    failed = 0
    for column in columns:
        if column not in result.columns:
            continue
        original = result[column]
        coerced = original.map(clean_numeric_value)
        non_empty = original.map(
            lambda v: bool(str(v).strip()) and str(v).strip().lower() not in {"nan", "none"}
            if not isinstance(v, (int, float)) else (v == v)
        )
        failed += int(((coerced.isna()) & non_empty).sum())
        result[column] = pd.to_numeric(coerced, errors="coerce")
    return result, failed


def normalize_columns(
    df: pd.DataFrame, alias_map: dict[str, list[str]]
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Copy recognised aliases onto standard column names.

    Returns the frame plus a log of {"original", "standard"} mappings applied.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(), []
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    canonical_lookup = {_canonical(column): column for column in result.columns}
    applied: list[dict[str, str]] = []
    for standard, aliases in alias_map.items():
        if standard in result.columns:
            continue
        for alias in aliases:
            source = canonical_lookup.get(_canonical(alias))
            if source is not None and source in result.columns:
                result[standard] = result[source]
                if _canonical(source) != _canonical(standard):
                    applied.append({"original": source, "standard": standard})
                break
    return result, applied


# Date-named columns are parsed into days_to_expiry. "유통기한" alone stays numeric
# (days); only explicit date-named columns are treated as calendar dates.
DATE_COLUMN_ALIASES = (
    "expiry_date", "expiration_date", "expire_date", "shelf_life_date", "exp_date",
    "만료일", "소비기한", "유통기한일자", "유통기한날짜", "유통기한일",
)


def days_from_date(value: object, reference: date | None = None) -> int | None:
    """Convert a date string / Excel serial into days remaining from ``reference``."""
    reference = reference or date.today()
    if value is None:
        return None
    # Excel serial date (numbers roughly between 2000-01-01 and ~2050).
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value != value:  # NaN
            return None
        if 36500 < float(value) < 60000:
            try:
                parsed = pd.to_datetime(float(value), origin="1899-12-30", unit="D")
                return int((parsed.date() - reference).days)
            except (ValueError, OverflowError):
                return None
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return int((pd.Timestamp(value).date() - reference).days)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "n/a", "na"}:
        return None
    normalized = text.replace(".", "-").replace("/", "-").replace(" ", "")
    for dayfirst in (False, True):
        try:
            parsed = pd.to_datetime(normalized, errors="raise", dayfirst=dayfirst)
            return int((parsed.date() - reference).days)
        except (ValueError, TypeError):
            continue
    return None


def normalize_date_columns(
    df: pd.DataFrame, reference: date | None = None
) -> tuple[pd.DataFrame, int, int, list[str]]:
    """Derive days_to_expiry from date-named columns.

    Existing days_to_expiry/expiry_days values take priority. Returns
    (df, success_count, failure_count, source_columns).
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(), 0, 0, []
    result = df.copy()
    has_days = "days_to_expiry" in result.columns and pd.to_numeric(
        result["days_to_expiry"], errors="coerce"
    ).notna().any()
    canonical_lookup = {_canonical(column): column for column in result.columns}
    date_columns = [
        canonical_lookup[_canonical(alias)]
        for alias in DATE_COLUMN_ALIASES
        if _canonical(alias) in canonical_lookup
    ]
    if not date_columns or has_days:
        return result, 0, 0, date_columns
    source = date_columns[0]
    computed = result[source].map(lambda v: days_from_date(v, reference))
    non_empty = result[source].map(
        lambda v: bool(str(v).strip()) and str(v).strip().lower() not in {"nan", "none"}
        if not isinstance(v, (int, float)) else (v == v)
    )
    success = int(computed.notna().sum())
    failure = int((computed.isna() & non_empty).sum())
    result["days_to_expiry"] = pd.to_numeric(computed, errors="coerce")
    if "expiry_days" not in result.columns:
        result["expiry_days"] = result["days_to_expiry"]
    return result, success, failure, date_columns


def drop_blank_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop fully-empty rows; return (df, removed_count)."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(), 0
    before = len(df)
    blank = df.apply(
        lambda row: all(
            (value is None) or (isinstance(value, float) and value != value)
            or (str(value).strip() == "")
            for value in row
        ),
        axis=1,
    )
    result = df[~blank].copy()
    return result, before - len(result)
