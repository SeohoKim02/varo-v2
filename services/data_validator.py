"""Data validation for Varo V2 sample workbooks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable

import pandas as pd

PASS = "통과"
WARNING = "주의"
ERROR = "오류"


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


def _add_blank_required(
    messages: list[ValidationMessage], sheet: str, df: pd.DataFrame, required: Iterable[str]
) -> None:
    for column in required:
        if column in df.columns and _blank_mask(df[column]).any():
            messages.append(ValidationMessage(ERROR, sheet, f"`{column}` 값이 비어 있습니다."))


def _validate_numeric(messages: list[ValidationMessage], sheet: str, df: pd.DataFrame, column: str, *, positive: bool = False, allow_negative: bool = False) -> None:
    if column not in df.columns:
        return
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        messages.append(ValidationMessage(ERROR, sheet, f"`{column}` 컬럼에 숫자가 아닌 값이 있습니다."))
    if positive and (values <= 0).any():
        messages.append(ValidationMessage(ERROR, sheet, f"`{column}` 값은 0보다 커야 합니다."))
    if not allow_negative and (values < 0).any():
        messages.append(ValidationMessage(ERROR, sheet, f"`{column}` 값은 음수일 수 없습니다."))


def _validate_stores(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> Dict[str, int]:
    stores = data.get("stores", pd.DataFrame())
    _add_missing(messages, "stores", stores, ("node_id", "node_name", "node_type"))
    _add_blank_required(messages, "stores", stores, ("node_id", "node_name", "node_type"))
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
    _add_missing(messages, "products", products, ("product_id", "product_name"))
    _add_blank_required(messages, "products", products, ("product_id", "product_name"))
    if "product_id" in products.columns and products["product_id"].duplicated().any():
        messages.append(ValidationMessage(ERROR, "products", "product_id 중복 값이 있습니다."))
    return int(len(products))


def _validate_inventory(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> int:
    inventory = data.get("inventory", pd.DataFrame())
    _add_missing(messages, "inventory", inventory, ("store_id", "product_id", "stock_qty"))
    _add_blank_required(messages, "inventory", inventory, ("store_id", "product_id", "stock_qty"))
    _validate_numeric(messages, "inventory", inventory, "stock_qty")
    _validate_numeric(messages, "inventory", inventory, "sales_qty")
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


def _validate_routes(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> int:
    routes = data.get("routes", pd.DataFrame())
    _add_missing(messages, "routes", routes, ("source_id", "target_id", "distance_km", "estimated_cost", "travel_time_min"))
    _add_blank_required(messages, "routes", routes, ("source_id", "target_id", "distance_km", "estimated_cost", "travel_time_min"))
    ids = _node_ids(data)
    for column in ("source_id", "target_id"):
        if column in routes.columns and ids:
            invalid = sorted(set(routes[column].dropna().astype(str)) - ids)
            if invalid:
                messages.append(ValidationMessage(ERROR, "routes", f"`{column}`에 stores.node_id에 없는 값이 있습니다: {invalid}"))
    if {"source_id", "target_id"}.issubset(routes.columns) and routes.duplicated(["source_id", "target_id"]).any():
        messages.append(ValidationMessage(WARNING, "routes", "동일 source_id/target_id 조합이 중복되어 있습니다."))
    for column in ("distance_km", "estimated_cost", "travel_time_min"):
        _validate_numeric(messages, "routes", routes, column)
    return int(len(routes))


def _validate_recommendations(data: dict[str, pd.DataFrame], messages: list[ValidationMessage]) -> Dict[str, int]:
    recs = data.get("recommendations", pd.DataFrame())
    required = (
        "route_id", "product_id", "product_name", "source_id", "source_name", "target_id", "target_name",
        "route_type", "recommended_qty", "transport_type", "estimated_cost", "expected_saving", "vhs_score",
        "recommendation_grade", "confidence_score", "reason",
    )
    _add_missing(messages, "v2_recommendations", recs, required)
    _add_blank_required(
        messages,
        "v2_recommendations",
        recs,
        ("route_id", "product_id", "product_name", "source_id", "source_name", "target_id", "target_name", "route_type", "recommended_qty"),
    )
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
    ids = _node_ids(data)
    for column in ("source_id", "target_id"):
        if column in recs.columns and ids:
            invalid = sorted(set(recs[column].dropna().astype(str)) - ids)
            if invalid:
                messages.append(ValidationMessage(ERROR, "v2_recommendations", f"`{column}`에 stores.node_id에 없는 값이 있습니다: {invalid}"))
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
