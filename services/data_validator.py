"""Data validation for Varo V2 sample workbooks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable
import math

import pandas as pd

PASS = "통과"
WARNING = "주의"
ERROR = "오류"

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "stores": ("node_id", "node_name", "node_type"),
    "products": ("product_id", "product_name"),
    "inventory": ("store_id", "product_id", "stock_qty"),
    "routes": ("source_id", "target_id", "distance_km", "estimated_cost", "travel_time_min"),
    "recommendations": (
        "route_id", "product_id", "product_name", "source_id", "source_name", "target_id", "target_name",
        "route_type", "recommended_qty", "transport_type", "estimated_cost", "expected_saving", "vhs_score",
        "recommendation_grade", "confidence_score", "reason",
    ),
}


@dataclass(frozen=True)
class ValidationMessage:
    level: str
    sheet: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    status: str
    messages: list[ValidationMessage] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(message.level == ERROR for message in self.messages)

    @property
    def has_warnings(self) -> bool:
        return any(message.level == WARNING for message in self.messages)

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "messages": [message.to_dict() for message in self.messages],
            "summary": dict(self.summary),
        }


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _add_missing(messages: list[ValidationMessage], sheet: str, df: pd.DataFrame, required: Iterable[str]) -> None:
    for column in _missing_columns(df, required):
        messages.append(ValidationMessage(ERROR, sheet, f"필수 컬럼 `{column}`이 없습니다."))


def _error_blank_id_values(messages: list[ValidationMessage], sheet: str, df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Flag rows whose required identifier is empty (blank/NaN).

    Kept in step with ``data_issues`` (missing_id) so a blank핵심 식별자 both blocks
    the analysis gate and shows up as a row-level fix. Missing columns are handled
    by ``_add_missing``; here we only check *values* in columns that exist.
    """
    for column in columns:
        if column in df.columns and _blank_mask(df[column]).any():
            messages.append(ValidationMessage(ERROR, sheet, f"`{column}`이 비어 있는 행이 있습니다."))


def _validate_numeric(messages: list[ValidationMessage], sheet: str, df: pd.DataFrame, column: str, *, positive: bool = False, allow_negative: bool = False) -> None:
    if column not in df.columns:
        return
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = values.isna() | ~values.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    if invalid.any():
        messages.append(ValidationMessage(ERROR, sheet, f"`{column}` 컬럼에 숫자가 아닌 값이 있습니다."))
    if positive and (values <= 0).any():
        messages.append(ValidationMessage(ERROR, sheet, f"`{column}` 값은 0보다 커야 합니다."))
    if not allow_negative and (values < 0).any():
        messages.append(ValidationMessage(ERROR, sheet, f"`{column}` 값은 음수일 수 없습니다."))


def _validate_stores(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> Dict[str, int]:
    stores = data.get("stores", pd.DataFrame())
    _add_missing(messages, "stores", stores, REQUIRED_COLUMNS["stores"])
    if stores.empty:
        messages.append(ValidationMessage(ERROR, "stores", "유효한 점포 데이터가 없습니다."))
    _error_blank_id_values(messages, "stores", stores, ("node_id", "node_name", "node_type"))
    if any(column not in stores.columns for column in ("node_id", "node_type")):
        return {"dc_count": 0, "store_count": 0}
    if stores["node_id"].duplicated().any():
        messages.append(ValidationMessage(ERROR, "stores", "node_id 중복 값이 있습니다."))
    node_type = stores["node_type"].astype(str).str.strip().str.upper()
    invalid = sorted(set(node_type) - {"DC", "STORE"})
    if invalid:
        messages.append(ValidationMessage(ERROR, "stores", f"node_type은 DC 또는 STORE만 허용합니다: {invalid}"))
    dc_count = int((node_type == "DC").sum())
    store_count = int((node_type == "STORE").sum())
    if dc_count < 1:
        messages.append(ValidationMessage(ERROR, "stores", "DC가 1개 이상 필요합니다."))
    if store_count < 1:
        messages.append(ValidationMessage(ERROR, "stores", "STORE가 1개 이상 필요합니다."))
    return {"dc_count": dc_count, "store_count": store_count}


def _validate_products(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> int:
    products = data.get("products", pd.DataFrame())
    _add_missing(messages, "products", products, REQUIRED_COLUMNS["products"])
    if products.empty:
        messages.append(ValidationMessage(ERROR, "products", "유효한 상품 데이터가 없습니다."))
    _error_blank_id_values(messages, "products", products, ("product_id", "product_name"))
    if "product_id" in products.columns and products["product_id"].duplicated().any():
        messages.append(ValidationMessage(ERROR, "products", "product_id 중복 값이 있습니다."))
    return int(len(products))


def _validate_inventory(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> int:
    inventory = data.get("inventory", pd.DataFrame())
    _add_missing(messages, "inventory", inventory, REQUIRED_COLUMNS["inventory"])
    if inventory.empty:
        messages.append(ValidationMessage(ERROR, "inventory", "유효한 재고 데이터가 없습니다."))
    _error_blank_id_values(messages, "inventory", inventory, ("store_id", "product_id"))
    _validate_numeric(messages, "inventory", inventory, "stock_qty")
    _validate_numeric(messages, "inventory", inventory, "sales_qty")
    store_ids = _node_ids(data)
    product_ids = _product_ids(data)
    _validate_reference(messages, "inventory", inventory, "store_id", store_ids, "stores.node_id")
    _validate_reference(messages, "inventory", inventory, "product_id", product_ids, "products.product_id")
    if "sales_qty" not in inventory.columns:
        messages.append(ValidationMessage(WARNING, "inventory", "판매량 컬럼을 찾지 못했습니다."))
    if not any(column in inventory.columns for column in ("expiry_days", "expiry_date", "expiration_date", "shelf_life_days")):
        messages.append(ValidationMessage(WARNING, "inventory", "유통기한 관련 컬럼을 찾지 못했습니다."))
    return int(len(inventory))


def _node_ids(data: dict[str, pd.DataFrame]) -> set[str]:
    stores = data.get("stores", pd.DataFrame())
    if "node_id" not in stores.columns:
        return set()
    return set(stores["node_id"].dropna().astype(str))


def _product_ids(data: dict[str, pd.DataFrame]) -> set[str]:
    products = data.get("products", pd.DataFrame())
    if "product_id" not in products.columns:
        return set()
    return set(products["product_id"].dropna().astype(str))


def _validate_reference(
    messages: list[ValidationMessage], sheet: str, frame: pd.DataFrame,
    column: str, valid: set[str], target: str,
) -> None:
    if column not in frame.columns or not valid:
        return
    invalid = sorted(set(frame.loc[~_blank_mask(frame[column]), column].astype(str)) - valid)
    if invalid:
        messages.append(ValidationMessage(ERROR, sheet, f"`{column}`에 {target}에 없는 값이 있습니다: {invalid}"))


def _validate_routes(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> int:
    routes = data.get("routes", pd.DataFrame())
    _add_missing(messages, "routes", routes, REQUIRED_COLUMNS["routes"])
    if routes.empty:
        messages.append(ValidationMessage(ERROR, "routes", "유효한 경로 데이터가 없습니다."))
    _error_blank_id_values(messages, "routes", routes, ("source_id", "target_id"))
    ids = _node_ids(data)
    for column in ("source_id", "target_id"):
        if column in routes.columns and ids:
            invalid = sorted(set(routes[column].dropna().astype(str)) - ids)
            if invalid:
                messages.append(ValidationMessage(ERROR, "routes", f"`{column}`에 stores.node_id에 없는 값이 있습니다: {invalid}"))
    if {"source_id", "target_id"}.issubset(routes.columns) and routes.duplicated(["source_id", "target_id"]).any():
        messages.append(ValidationMessage(WARNING, "routes", "동일 source_id/target_id 조합이 중복되어 있습니다."))
    if {"source_id", "target_id"}.issubset(routes.columns):
        same = (~_blank_mask(routes["source_id"])) & (
            routes["source_id"].astype(str).str.strip() == routes["target_id"].astype(str).str.strip()
        )
        if bool(same.any()):
            messages.append(ValidationMessage(ERROR, "routes", "출발지와 도착지가 같은 경로가 있습니다."))
    for column in ("distance_km", "estimated_cost", "travel_time_min"):
        _validate_numeric(messages, "routes", routes, column)
    return int(len(routes))


def _validate_recommendations(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> Dict[str, int]:
    recs = data.get("recommendations", pd.DataFrame())
    required = REQUIRED_COLUMNS["recommendations"]
    _add_missing(messages, "v2_recommendations", recs, required)
    if "route_id" in recs.columns:
        if _blank_mask(recs["route_id"]).any():
            messages.append(ValidationMessage(ERROR, "v2_recommendations", "route_id가 비어 있는 행이 있습니다."))
        if recs["route_id"].duplicated().any():
            messages.append(ValidationMessage(ERROR, "v2_recommendations", "중복 route_id가 있습니다."))
    if "route_type" in recs.columns:
        route_type = recs["route_type"].astype(str).str.strip().str.upper()
        invalid = sorted(set(route_type) - {"DIRECT", "VIA_DC"})
        if invalid:
            messages.append(ValidationMessage(ERROR, "v2_recommendations", f"지원하지 않는 route_type입니다: {invalid}"))
        via_dc = route_type == "VIA_DC"
        for column in ("dc_id", "dc_name"):
            if column not in recs.columns:
                if via_dc.any():
                    messages.append(ValidationMessage(ERROR, "v2_recommendations", f"VIA_DC 추천에는 `{column}` 컬럼이 필요합니다."))
            elif via_dc.any() and _blank_mask(recs.loc[via_dc, column]).any():
                messages.append(ValidationMessage(ERROR, "v2_recommendations", f"VIA_DC 추천에는 `{column}` 값이 필요합니다."))
    _error_blank_id_values(messages, "v2_recommendations", recs, ("product_id", "source_id", "target_id"))
    if {"source_id", "target_id"}.issubset(recs.columns):
        same = (~_blank_mask(recs["source_id"])) & (
            recs["source_id"].astype(str).str.strip() == recs["target_id"].astype(str).str.strip()
        )
        if bool(same.any()):
            messages.append(ValidationMessage(ERROR, "v2_recommendations", "출발지와 도착지가 같은 추천이 있습니다."))
    ids = _node_ids(data)
    product_ids = _product_ids(data)
    for column in ("source_id", "target_id"):
        if column in recs.columns and ids:
            invalid = sorted(set(recs[column].dropna().astype(str)) - ids)
            if invalid:
                messages.append(ValidationMessage(ERROR, "v2_recommendations", f"`{column}`에 stores.node_id에 없는 값이 있습니다: {invalid}"))
    _validate_reference(messages, "v2_recommendations", recs, "product_id", product_ids, "products.product_id")
    if "dc_id" in recs.columns and ids:
        dc_ids = set(
            data.get("stores", pd.DataFrame()).loc[
                data.get("stores", pd.DataFrame()).get("node_type", pd.Series(dtype=str)).astype(str).str.upper() == "DC",
                "node_id",
            ].astype(str)
        ) if {"node_id", "node_type"}.issubset(data.get("stores", pd.DataFrame()).columns) else set()
        via = recs.get("route_type", pd.Series(index=recs.index, dtype=str)).astype(str).str.upper() == "VIA_DC"
        invalid_dc = set(recs.loc[via & ~_blank_mask(recs.get("dc_id", pd.Series(index=recs.index, dtype=object))), "dc_id"].astype(str)) - dc_ids
        if invalid_dc:
            messages.append(ValidationMessage(ERROR, "v2_recommendations", f"`dc_id`에 유효한 DC가 아닌 값이 있습니다: {sorted(invalid_dc)}"))
    for column in ("recommended_qty", "estimated_cost", "expected_saving", "vhs_score", "confidence_score", "distance_km", "travel_time_min"):
        _validate_numeric(messages, "v2_recommendations", recs, column, positive=(column == "recommended_qty"))
    route_type = recs["route_type"].astype(str).str.strip().str.upper() if "route_type" in recs.columns else pd.Series(dtype=str)
    return {
        "recommendation_count": int(len(recs)),
        "direct_count": int((route_type == "DIRECT").sum()),
        "via_dc_count": int((route_type == "VIA_DC").sum()),
    }


def validate_workbook_data(data: dict[str, pd.DataFrame]) -> ValidationReport:
    messages: list[ValidationMessage] = []
    for key in ("stores", "products", "inventory", "routes", "recommendations"):
        if key not in data:
            messages.append(ValidationMessage(ERROR, key, f"필수 데이터 `{key}`가 로드되지 않았습니다."))

    store_summary = _validate_stores(data, messages)
    product_count = _validate_products(data, messages)
    inventory_count = _validate_inventory(data, messages)
    route_count = _validate_routes(data, messages)
    rec_summary = _validate_recommendations(data, messages)

    if any(message.level == ERROR for message in messages):
        status = ERROR
    elif any(message.level == WARNING for message in messages):
        status = WARNING
    else:
        status = PASS
        messages.append(ValidationMessage(PASS, "workbook", "필수 검증을 통과했습니다."))

    summary = {
        "store_count": store_summary.get("store_count", 0),
        "dc_count": store_summary.get("dc_count", 0),
        "product_count": product_count,
        "inventory_count": inventory_count,
        "route_count": route_count,
        **rec_summary,
    }
    return ValidationReport(status=status, messages=messages, summary=summary)
