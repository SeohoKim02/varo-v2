"""Excel data loading for Varo V2."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Dict
import re
import warnings

import pandas as pd

from services.column_aliases import (
    NUMERIC_COLUMNS,
    SHEET_ALIASES,
    coerce_numeric_columns,
    drop_blank_rows,
    normalize_columns,
    normalize_date_columns,
)

# Canonical data keys. Sheet names are resolved case-insensitively through the
# aliases below so DQN workbooks can keep their original naming convention.
REQUIRED_SHEETS = ("stores", "products", "inventory", "routes")
OPTIONAL_SHEETS = ("dcs", "recommendations", "transport_modes", "config", "quality_check", "readme")
SHEET_NAME_ALIASES = {
    "stores": ("stores", "store", "store_master", "점포", "점포목록"),
    "dcs": ("dcs", "dc", "distribution_centers", "distribution_center", "물류센터", "센터"),
    "products": ("products", "product", "items", "item", "상품", "상품목록"),
    "inventory": ("inventory", "inventories", "stock", "재고", "재고현황"),
    "routes": ("routes", "route", "network", "경로", "이동경로"),
    "recommendations": (
        "v2_recommendations", "recommendations", "recommendation",
        "dqn_recommendations", "training_recommendations", "추천", "추천결과",
    ),
    "transport_modes": ("transport_modes", "transport_mode", "운송수단"),
    "config": ("config", "configuration", "설정"),
    "quality_check": ("quality_check", "qualitycheck", "품질점검"),
    "readme": ("readme", "guide", "안내"),
}
SAMPLE_FILENAME = "Varo_V2_네트워크_샘플.xlsx"


class DataLoadError(Exception):
    """User-facing load error."""


_OPENPYXL_STYLE_WARNING_PATTERNS = (
    r".*Unknown extension is not supported and will be removed.*",
    r".*Conditional Formatting extension is not supported.*",
    r".*Data Validation extension.*",
)


@contextmanager
def _suppress_excel_style_warnings():
    """Suppress only known openpyxl style-extension warnings while reading Excel."""
    with warnings.catch_warnings():
        for pattern in _OPENPYXL_STYLE_WARNING_PATTERNS:
            warnings.filterwarnings("ignore", message=pattern, category=UserWarning, module=r"openpyxl\..*")
        yield


def get_default_sample_path(base_dir: Path | None = None) -> Path:
    root = base_dir or Path(__file__).resolve().parents[1]
    return root / "data" / SAMPLE_FILENAME


def _read_excel_source(source: str | Path | BinaryIO) -> pd.ExcelFile:
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        with _suppress_excel_style_warnings():
            return pd.ExcelFile(source)
    except Exception as exc:
        raise DataLoadError("엑셀 파일을 읽을 수 없습니다. 파일 형식과 손상 여부를 확인해주세요.") from exc


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def _canonical_sheet_name(value: object) -> str:
    return re.sub(r"[\s_\-]+", "", str(value).strip().lower())


def _resolve_sheet_names(sheet_names: list[str]) -> dict[str, str]:
    lookup = {_canonical_sheet_name(name): name for name in sheet_names}
    resolved: dict[str, str] = {}
    for key, aliases in SHEET_NAME_ALIASES.items():
        for alias in aliases:
            actual = lookup.get(_canonical_sheet_name(alias))
            if actual is not None:
                resolved[key] = actual
                break
    return resolved


def _ensure_alias(df: pd.DataFrame, target: str, candidates: tuple[str, ...]) -> pd.DataFrame:
    if target in df.columns:
        return df
    for candidate in candidates:
        if candidate in df.columns:
            df[target] = df[candidate]
            break
    return df


def _normalize_stores(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_alias(df, "node_id", ("store_id", "id"))
    df = _ensure_alias(df, "node_name", ("store_name", "name"))
    if "node_type" not in df.columns:
        # Generic ``store_type`` commonly contains a business/trade-area
        # classification rather than STORE/DC.  Reuse it only when every
        # populated value is an unambiguous network-node label.
        aliases = {
            "DC": "DC", "DISTRIBUTION_CENTER": "DC", "WAREHOUSE": "DC",
            "HUB": "DC", "물류센터": "DC", "센터": "DC",
            "STORE": "STORE", "SHOP": "STORE", "RETAIL": "STORE",
            "점포": "STORE", "매장": "STORE",
        }
        adopted = False
        for candidate in ("store_type", "type", "유형", "구분", "점포유형"):
            if candidate not in df.columns:
                continue
            values = df[candidate].dropna().astype(str).str.strip()
            normalized_values = values.str.upper().str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False)
            if not normalized_values.empty and normalized_values.isin(aliases).all():
                df["node_type"] = normalized_values.map(aliases)
                adopted = True
                break
        if not adopted:
            df["node_type"] = "STORE"
    else:
        blank = df["node_type"].isna() | (df["node_type"].astype(str).str.strip() == "")
        df.loc[blank, "node_type"] = "STORE"
        df["node_type"] = df["node_type"].astype(str).str.strip().str.upper()
    return df


def _normalize_dcs(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_alias(df, "node_id", ("dc_id", "distribution_center_id", "center_id", "id"))
    df = _ensure_alias(df, "node_name", ("dc_name", "distribution_center_name", "center_name", "name"))
    df["node_type"] = "DC"
    if "store_name" not in df.columns and "node_name" in df.columns:
        df["store_name"] = df["node_name"]
    return df


def _normalize_routes(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_alias(df, "source_id", ("from_id", "from_store_id"))
    df = _ensure_alias(df, "target_id", ("to_id", "to_store_id"))
    df = _ensure_alias(df, "distance_km", ("route_distance_km", "direct_distance_km", "distance"))
    df = _ensure_alias(df, "estimated_cost", ("transport_cost", "direct_cost", "cost"))
    df = _ensure_alias(df, "travel_time_min", ("route_time_min", "time_min", "time"))
    return df


def _normalize_products(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_alias(df, "product_id", ("item_id", "id"))
    df = _ensure_alias(df, "product_name", ("item_name", "name"))
    df = _ensure_alias(df, "disposal_cost_per_unit", ("disposal_cost",))
    return df


def _normalize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_alias(df, "store_id", ("node_id",))
    df = _ensure_alias(df, "product_id", ("item_id",))
    df = _ensure_alias(df, "stock_qty", ("current_stock", "quantity", "inventory_qty", "stock"))
    df = _ensure_alias(df, "sales_qty", ("avg_daily_sales", "sales_30d", "sales_30", "sales_7d", "sales"))
    df = _ensure_alias(df, "sales_30d", ("sales_30", "sales_qty", "avg_daily_sales"))
    df = _ensure_alias(df, "expiry_days", ("days_to_expiry", "shelf_life_days"))
    df = _ensure_alias(df, "days_to_expiry", ("expiry_days", "shelf_life_days"))
    df = _ensure_alias(df, "category", ("inventory_category",))
    df = _ensure_alias(df, "inventory_category", ("category",))
    df = _ensure_alias(df, "disposal_cost_per_unit", ("disposal_cost",))
    return df


def _normalize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    if "route_type" in df.columns:
        df["route_type"] = df["route_type"].astype(str).str.strip().str.upper()
    return df


_LEGACY_NORMALIZERS = {
    "stores": _normalize_stores,
    "dcs": _normalize_dcs,
    "products": _normalize_products,
    "routes": _normalize_routes,
    "inventory": _normalize_inventory,
    "recommendations": _normalize_recommendations,
}


def _blank(value: object) -> bool:
    try:
        return bool(pd.isna(value)) or str(value).strip() == ""
    except (TypeError, ValueError):
        return value is None


def _id_text(value: object) -> str | None:
    if _blank(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


_IDENTIFIER_COLUMNS = {
    "stores": ("node_id",),
    "dcs": ("node_id",),
    "products": ("product_id",),
    "inventory": ("store_id", "product_id"),
    "routes": ("source_id", "target_id"),
    "recommendations": ("route_id", "product_id", "source_id", "target_id", "dc_id"),
}


def _normalize_identifiers(key: str, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in _IDENTIFIER_COLUMNS.get(key, ()):
        if column in result.columns:
            result[column] = result[column].map(_id_text)
    return result


def _merge_dc_nodes(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    dcs = data.pop("dcs", pd.DataFrame())
    if not isinstance(dcs, pd.DataFrame) or dcs.empty:
        return data
    stores = data.get("stores", pd.DataFrame()).copy()
    if "node_id" in dcs.columns and "node_id" in stores.columns:
        dc_ids = set(dcs["node_id"].dropna().astype(str))
        stores = stores[~stores["node_id"].astype(str).isin(dc_ids)].copy()
    data["stores"] = pd.concat([stores, dcs], ignore_index=True, sort=False)
    return data


def _lookup_maps(frame: pd.DataFrame, id_column: str, name_column: str) -> tuple[dict[str, str], dict[str, str]]:
    id_to_name: dict[str, str] = {}
    name_to_id: dict[str, str] = {}
    if frame.empty or id_column not in frame.columns:
        return id_to_name, name_to_id
    for _, row in frame.iterrows():
        identifier = _id_text(row.get(id_column))
        name = None if _blank(row.get(name_column)) else str(row.get(name_column)).strip()
        if identifier:
            id_to_name[identifier] = name or identifier
            name_to_id[identifier.casefold()] = identifier
        if identifier and name:
            name_to_id[name.casefold()] = identifier
    return id_to_name, name_to_id


def _resolve_identifier(value: object, name_to_id: dict[str, str]) -> str | None:
    text = _id_text(value)
    if text is None:
        return None
    return name_to_id.get(text.casefold(), text)


def _fill_blank(frame: pd.DataFrame, column: str, value: object) -> None:
    if column not in frame.columns:
        frame[column] = value
        return
    mask = frame[column].isna() | (frame[column].astype(str).str.strip().isin({"", "nan", "None"}))
    frame.loc[mask, column] = value


def _normalize_route_type(value: object, dc_id: object = None) -> str:
    raw = "" if _blank(value) else str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if raw in {"DIRECT", "DIRECT_TRANSFER", "STORE_TO_STORE", "점포간", "직접", "직접_이동"}:
        return "DIRECT"
    if raw in {"VIA_DC", "DC", "DC_TRANSFER", "HUB", "DC_경유", "경유", "물류센터_경유"}:
        return "VIA_DC"
    if not raw:
        return "VIA_DC" if not _blank(dc_id) else "DIRECT"
    return raw


def _resolve_node_columns(frame: pd.DataFrame, stores: pd.DataFrame, *, include_names: bool) -> pd.DataFrame:
    result = frame.copy()
    id_to_name, name_to_id = _lookup_maps(stores, "node_id", "node_name")
    for prefix in ("source", "target"):
        id_column = f"{prefix}_id"
        name_column = f"{prefix}_name"
        if id_column not in result.columns and include_names and name_column in result.columns:
            result[id_column] = result[name_column]
        if id_column in result.columns:
            result[id_column] = result[id_column].map(lambda value: _resolve_identifier(value, name_to_id))
        if include_names:
            if name_column not in result.columns:
                result[name_column] = result.get(id_column, pd.Series(index=result.index, dtype=object)).map(id_to_name)
            else:
                missing = result[name_column].isna() | (result[name_column].astype(str).str.strip() == "")
                result.loc[missing, name_column] = result.loc[missing, id_column].map(id_to_name)
    return result


def _route_metrics(routes: pd.DataFrame, source: object, target: object, dc_id: object, route_type: str) -> dict[str, float]:
    if routes.empty or not {"source_id", "target_id"}.issubset(routes.columns):
        return {}
    pairs: list[tuple[str | None, str | None]]
    source_id, target_id, center_id = _id_text(source), _id_text(target), _id_text(dc_id)
    if route_type == "VIA_DC" and center_id:
        pairs = [(source_id, center_id), (center_id, target_id)]
    else:
        pairs = [(source_id, target_id)]
    totals = {"distance_km": 0.0, "estimated_cost": 0.0, "travel_time_min": 0.0}
    for start, end in pairs:
        match = routes[
            (routes["source_id"].astype(str) == str(start))
            & (routes["target_id"].astype(str) == str(end))
        ]
        if match.empty:
            return {}
        row = match.iloc[0]
        for column in totals:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.isna(value):
                return {}
            totals[column] += float(value)
    return totals


def _enrich_recommendations(
    frame: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    routes: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    result = _resolve_node_columns(frame, stores, include_names=True)
    node_id_to_name, node_name_to_id = _lookup_maps(stores, "node_id", "node_name")
    product_id_to_name, product_name_to_id = _lookup_maps(products, "product_id", "product_name")

    if "product_id" not in result.columns and "product_name" in result.columns:
        result["product_id"] = result["product_name"]
    if "product_id" in result.columns:
        result["product_id"] = result["product_id"].map(lambda value: _resolve_identifier(value, product_name_to_id))
    if "product_name" not in result.columns:
        result["product_name"] = result.get("product_id", pd.Series(index=result.index, dtype=object)).map(product_id_to_name)
    else:
        missing = result["product_name"].isna() | (result["product_name"].astype(str).str.strip() == "")
        result.loc[missing, "product_name"] = result.loc[missing, "product_id"].map(product_id_to_name)

    if "dc_id" not in result.columns and "dc_name" in result.columns:
        result["dc_id"] = result["dc_name"]
    if "dc_id" in result.columns:
        result["dc_id"] = result["dc_id"].map(lambda value: _resolve_identifier(value, node_name_to_id))
    _fill_blank(result, "dc_name", None)
    dc_rows = stores[stores.get("node_type", pd.Series(dtype=str)).astype(str).str.upper() == "DC"]
    dc_ids = sorted(dc_rows.get("node_id", pd.Series(dtype=object)).dropna().astype(str).tolist())

    _fill_blank(result, "route_type", None)
    result["route_type"] = [
        _normalize_route_type(route_type, dc_id)
        for route_type, dc_id in zip(result["route_type"], result.get("dc_id", pd.Series(index=result.index, dtype=object)))
    ]
    via_mask = result["route_type"] == "VIA_DC"
    if dc_ids:
        missing_dc = via_mask & (result["dc_id"].isna() | (result["dc_id"].astype(str).str.strip() == ""))
        result.loc[missing_dc, "dc_id"] = dc_ids[0]
    result.loc[via_mask, "dc_name"] = result.loc[via_mask, "dc_name"].where(
        result.loc[via_mask, "dc_name"].notna() & (result.loc[via_mask, "dc_name"].astype(str).str.strip() != ""),
        result.loc[via_mask, "dc_id"].map(node_id_to_name),
    )

    generated = 0
    used: set[str] = set()
    route_ids: list[str] = []
    source_ids = result.get("route_id", pd.Series(index=result.index, dtype=object))
    for position, value in enumerate(source_ids, start=1):
        candidate = _id_text(value)
        if not candidate or candidate in used:
            generated += 1
            candidate = f"DQN-R{position:03d}"
            suffix = 1
            while candidate in used:
                suffix += 1
                candidate = f"DQN-R{position:03d}-{suffix}"
        used.add(candidate)
        route_ids.append(candidate)
    result["route_id"] = route_ids

    defaults = {
        "transport_type": "일반 트럭",
        "expected_saving": 0.0,
        "vhs_score": 50.0,
        "recommendation_grade": "보통",
        "confidence_score": 50.0,
        "reason": "업로드 추천",
    }
    for column, value in defaults.items():
        _fill_blank(result, column, value)

    for row_index, row in result.iterrows():
        metrics = _route_metrics(
            routes, row.get("source_id"), row.get("target_id"), row.get("dc_id"), str(row.get("route_type"))
        )
        for column in ("distance_km", "estimated_cost", "travel_time_min"):
            if column not in result.columns:
                result[column] = pd.NA
            if _blank(result.at[row_index, column]) and column in metrics:
                result.at[row_index, column] = metrics[column]
    return result, generated


def normalize_loaded_data(
    data: Dict[str, pd.DataFrame], *, collect_report: bool = False
):
    """Apply legacy + alias normalization, numeric coercion, and blank-row drop.

    When ``collect_report`` is set, also return an upload-quality report.
    """
    normalized: Dict[str, pd.DataFrame] = {
        key: value.copy() if isinstance(value, pd.DataFrame) else value
        for key, value in data.items()
    }
    report: Dict[str, Any] = {
        "recognized_sheets": [k for k, v in normalized.items() if isinstance(v, pd.DataFrame)],
        "column_mappings": [],
        "numeric_failed": {},
        "blank_removed": {},
        "row_counts": {},
        "date_success": 0,
        "date_failed": 0,
        "date_columns": [],
        "route_ids_generated": 0,
    }
    for key in list(normalized.keys()):
        frame = normalized[key]
        if not isinstance(frame, pd.DataFrame):
            continue
        frame, removed = drop_blank_rows(frame)
        if key in _LEGACY_NORMALIZERS:
            frame = _LEGACY_NORMALIZERS[key](frame)
        alias_map = SHEET_ALIASES.get(key)
        applied: list[dict[str, str]] = []
        if alias_map:
            frame, applied = normalize_columns(frame, alias_map)
        if key == "inventory":
            frame, date_ok, date_fail, date_cols = normalize_date_columns(frame)
            report["date_success"] += date_ok
            report["date_failed"] += date_fail
            report["date_columns"].extend(date_cols)
        frame, failed = coerce_numeric_columns(frame, NUMERIC_COLUMNS.get(key, ()))
        frame = _normalize_identifiers(key, frame)
        if key == "stores" and "node_type" in frame.columns:
            frame["node_type"] = frame["node_type"].astype(str).str.strip().str.upper()
        if key == "recommendations" and "route_type" in frame.columns:
            frame["route_type"] = frame["route_type"].astype(str).str.strip().str.upper()
        normalized[key] = frame
        report["blank_removed"][key] = removed
        report["numeric_failed"][key] = failed
        report["row_counts"][key] = int(len(frame))
        report["column_mappings"].extend({"sheet": key, **mapping} for mapping in applied)

    normalized = _merge_dc_nodes(normalized)
    stores = normalized.get("stores", pd.DataFrame())
    products = normalized.get("products", pd.DataFrame())
    routes = normalized.get("routes", pd.DataFrame())
    if isinstance(routes, pd.DataFrame) and not routes.empty:
        routes = _resolve_node_columns(routes, stores, include_names=False)
        normalized["routes"] = routes
    inventory = normalized.get("inventory", pd.DataFrame())
    if isinstance(inventory, pd.DataFrame) and not inventory.empty:
        _, node_name_to_id = _lookup_maps(stores, "node_id", "node_name")
        _, product_name_to_id = _lookup_maps(products, "product_id", "product_name")
        if "store_id" in inventory.columns:
            inventory["store_id"] = inventory["store_id"].map(lambda value: _resolve_identifier(value, node_name_to_id))
        if "product_id" in inventory.columns:
            inventory["product_id"] = inventory["product_id"].map(lambda value: _resolve_identifier(value, product_name_to_id))
        normalized["inventory"] = inventory
    recommendations = normalized.get("recommendations")
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        recommendations, generated = _enrich_recommendations(recommendations, stores, products, routes)
        recommendations, failed = coerce_numeric_columns(recommendations, NUMERIC_COLUMNS["recommendations"])
        normalized["recommendations"] = recommendations
        report["route_ids_generated"] = generated
        report["numeric_failed"]["recommendations"] = (
            report["numeric_failed"].get("recommendations", 0) + failed
        )
        report["row_counts"]["recommendations"] = int(len(recommendations))
    report["row_counts"]["stores"] = int(len(stores))
    return (normalized, report) if collect_report else normalized


def load_excel_data(
    source: str | Path | BinaryIO, *, return_report: bool = False
):
    excel = _read_excel_source(source)
    try:
        sheet_names = list(excel.sheet_names)
        resolved = _resolve_sheet_names(sheet_names)
        missing = [sheet for sheet in REQUIRED_SHEETS if sheet not in resolved]
        if missing:
            raise DataLoadError(f"필수 시트 누락: {', '.join(missing)}")

        loaded: Dict[str, pd.DataFrame] = {}
        for key in (*REQUIRED_SHEETS, *OPTIONAL_SHEETS):
            sheet = resolved.get(key)
            if sheet is not None:
                try:
                    with _suppress_excel_style_warnings():
                        loaded[key] = _clean_columns(pd.read_excel(excel, sheet_name=sheet))
                except Exception as exc:
                    raise DataLoadError("엑셀 데이터를 읽는 중 오류가 발생했습니다.") from exc
        result = normalize_loaded_data(loaded, collect_report=return_report)
        if return_report:
            normalized, report = result
            report["loaded_sheets"] = list(loaded.keys())
            report["all_excel_sheets"] = sorted(sheet_names)
            report["resolved_sheets"] = dict(resolved)
            return normalized, report
        return result
    finally:
        excel.close()


def get_sheet_row_counts(data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    return {key: int(len(value)) for key, value in data.items() if isinstance(value, pd.DataFrame)}


def has_uploaded_data(uploaded_data: Dict[str, Any] | None) -> bool:
    return bool(uploaded_data)
