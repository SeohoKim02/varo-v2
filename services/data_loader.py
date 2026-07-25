"""Excel data loading for Varo V2."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Dict
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

# Data sheets that must be present to load at all. v2_recommendations is optional:
# when it is missing the app falls back to generated V2 candidates.
REQUIRED_SHEETS = ("stores", "products", "inventory", "routes")
# `recommendations` and `dcs` support the DQN training workbooks, which keep the
# recommendation sheet under a plain name and store DCs on a separate sheet.
OPTIONAL_SHEETS = (
    "v2_recommendations", "recommendations", "dcs", "transport_modes", "config",
    "dqn_cases", "dqn_config", "summary", "Quality_Check", "README",
)
SHEET_KEY_MAP = {
    "stores": "stores",
    "products": "products",
    "inventory": "inventory",
    "routes": "routes",
    "v2_recommendations": "recommendations",
    "recommendations": "recommendations",
    "dcs": "dcs",
    "transport_modes": "transport_modes",
    "config": "config",
    "dqn_cases": "dqn_cases",
    "dqn_config": "dqn_config",
    "summary": "summary",
    "Quality_Check": "quality_check",
    "README": "readme",
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
    df = _ensure_alias(df, "node_type", ("store_type", "type"))
    if "node_type" in df.columns:
        df["node_type"] = df["node_type"].astype(str).str.strip().str.upper()
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


def _derive_inventory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive dead_stock_qty and demand_qty when a sample omits them.

    The DQN inventory ships stock/sales but not the 악성재고(dead_stock_qty) or
    수요(demand_qty) columns some legacy analyzers (store_clustering,
    min_cost_network) require. Derived here so those steps connect; never written
    back to the source file.
    """
    if df is None or df.empty:
        return df
    stock = pd.to_numeric(df["stock_qty"], errors="coerce") if "stock_qty" in df.columns else None
    sales30 = pd.to_numeric(df["sales_30d"], errors="coerce") if "sales_30d" in df.columns else None
    daily = pd.to_numeric(df["avg_daily_sales"], errors="coerce") if "avg_daily_sales" in df.columns else None
    baseline = sales30 if sales30 is not None else (daily * 30.0 if daily is not None else None)
    if "demand_qty" not in df.columns and baseline is not None:
        df["demand_qty"] = baseline
    if "dead_stock_qty" not in df.columns and stock is not None:
        df["dead_stock_qty"] = (stock - baseline).clip(lower=0) if baseline is not None else 0
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
    return _derive_inventory_metrics(df)


def _ensure_recommendation_presentation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add presentation columns some pre-computed samples omit.

    The DQN training samples ship scores (vhs/greedy/dqn) but no transport_type,
    recommendation_grade, or reason column. These are derived on load so the
    validator and adapter see a complete frame; the source files are never touched.
    """
    if df is None or df.empty:
        return df
    if "transport_type" not in df.columns:
        df["transport_type"] = "일반 탑차"
    if "recommendation_grade" not in df.columns:
        if "vhs_score" in df.columns:
            vhs = pd.to_numeric(df["vhs_score"], errors="coerce")
            df["recommendation_grade"] = vhs.map(
                lambda value: "높음" if pd.notna(value) and value >= 75
                else "보통" if pd.notna(value) and value >= 55 else "낮음"
            )
        else:
            df["recommendation_grade"] = "보통"
    if "reason" not in df.columns:
        df["reason"] = ""
    return df


def _normalize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    # DQN-style sheets keep a unique recommendation_id plus a reusable route_id
    # (one route can carry several product recommendations). V2 needs a unique
    # per-row key, so promote recommendation_id to route_id when route_id repeats.
    if (
        "recommendation_id" in df.columns and "route_id" in df.columns
        and df["route_id"].duplicated().any() and not df["recommendation_id"].duplicated().any()
    ):
        df = df.copy()
        df["route_id"] = df["recommendation_id"].astype(str)
    df = _ensure_alias(df, "source_id", ("from_store_id", "from_id"))
    df = _ensure_alias(df, "source_name", ("from_store_name", "from_store"))
    df = _ensure_alias(df, "target_id", ("to_store_id", "to_id"))
    df = _ensure_alias(df, "target_name", ("to_store_name", "to_store"))
    df = _ensure_alias(df, "estimated_cost", ("transport_cost",))
    df = _ensure_alias(df, "confidence_score", ("confidence",))
    if "route_type" in df.columns:
        df["route_type"] = df["route_type"].astype(str).str.strip().str.upper()
    return _ensure_recommendation_presentation_columns(df)


_LEGACY_NORMALIZERS = {
    "stores": _normalize_stores,
    "products": _normalize_products,
    "routes": _normalize_routes,
    "inventory": _normalize_inventory,
    "recommendations": _normalize_recommendations,
}


def _merge_separate_dc_sheet(loaded: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Fold a separate ``dcs`` sheet into ``stores`` as node_type=DC rows.

    DQN-style workbooks keep DCs on their own ``dcs`` sheet and use ``store_type``
    on the ``stores`` sheet as a business tag (e.g. 마감임박형) rather than
    STORE/DC. In that layout every ``stores`` row is a STORE and every ``dcs`` row
    is a DC. Runs on the in-memory frames only; the original files are untouched.
    Workbooks without a ``dcs`` sheet (existing V2 samples) are returned unchanged.
    """
    dcs = loaded.get("dcs")
    stores = loaded.get("stores")
    if (
        not isinstance(dcs, pd.DataFrame) or dcs.empty
        or not isinstance(stores, pd.DataFrame) or stores.empty
    ):
        loaded.pop("dcs", None)
        return loaded
    stores = stores.copy()
    if "store_id" in stores.columns:
        stores["node_id"] = stores["store_id"]
    if "store_name" in stores.columns:
        stores["node_name"] = stores["store_name"]
    stores["node_type"] = "STORE"

    dcs = dcs.reset_index(drop=True)
    dc_rows = pd.DataFrame()
    dc_rows["node_id"] = dcs.get("dc_id")
    dc_rows["node_name"] = dcs.get("dc_name")
    dc_rows["node_type"] = "DC"
    dc_rows["store_id"] = dcs.get("dc_id")
    dc_rows["store_name"] = dcs.get("dc_name")
    for column in ("region", "latitude", "longitude", "capacity"):
        if column in dcs.columns:
            dc_rows[column] = dcs[column]

    result = dict(loaded)
    result["stores"] = pd.concat([stores, dc_rows], ignore_index=True)
    result.pop("dcs", None)
    return result


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
        if key == "stores" and "node_type" in frame.columns:
            frame["node_type"] = frame["node_type"].astype(str).str.strip().str.upper()
        if key == "recommendations" and "route_type" in frame.columns:
            frame["route_type"] = frame["route_type"].astype(str).str.strip().str.upper()
        normalized[key] = frame
        report["blank_removed"][key] = removed
        report["numeric_failed"][key] = failed
        report["row_counts"][key] = int(len(frame))
        report["column_mappings"].extend({"sheet": key, **mapping} for mapping in applied)
    return (normalized, report) if collect_report else normalized


def load_excel_data(
    source: str | Path | BinaryIO, *, return_report: bool = False
):
    excel = _read_excel_source(source)
    try:
        sheet_names = set(excel.sheet_names)
        missing = [sheet for sheet in REQUIRED_SHEETS if sheet not in sheet_names]
        if missing:
            raise DataLoadError("필수 시트 `" + "`, `".join(missing) + "`가 없습니다.")

        loaded: Dict[str, pd.DataFrame] = {}
        # Raw snapshots preserve the file exactly as read (original column names
        # incl. surrounding spaces, original cell values, original data-row order:
        # positional index i == 스프레드시트 데이터 i번째 행 == 파일 행 i+2, header=1행).
        # Only the *normalized* copy (`loaded`) is used for analysis; the raw copy
        # is the basis for원본 위치·값 기반 오류 추적. The source file is never touched.
        raw_snapshots: Dict[str, pd.DataFrame] = {}
        raw_sheet_names: Dict[str, str] = {}
        for sheet in (*REQUIRED_SHEETS, *OPTIONAL_SHEETS):
            if sheet in sheet_names:
                key = SHEET_KEY_MAP[sheet]
                try:
                    with _suppress_excel_style_warnings():
                        frame = pd.read_excel(excel, sheet_name=sheet)
                except Exception as exc:
                    raise DataLoadError(f"시트 `{sheet}`를 읽는 중 오류가 발생했습니다.") from exc
                raw_snapshots[key] = frame.reset_index(drop=True)
                raw_sheet_names[key] = sheet
                loaded[key] = _clean_columns(frame)
        loaded = _merge_separate_dc_sheet(loaded)
        result = normalize_loaded_data(loaded, collect_report=return_report)
        if return_report:
            normalized, report = result
            report["loaded_sheets"] = list(loaded.keys())
            report["all_excel_sheets"] = sorted(sheet_names)
            report["raw_sheets"] = raw_snapshots
            report["raw_sheet_names"] = raw_sheet_names
            return normalized, report
        return result
    finally:
        excel.close()


def get_sheet_row_counts(data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    return {key: int(len(value)) for key, value in data.items() if isinstance(value, pd.DataFrame)}


def has_uploaded_data(uploaded_data: Dict[str, Any] | None) -> bool:
    return bool(uploaded_data)
