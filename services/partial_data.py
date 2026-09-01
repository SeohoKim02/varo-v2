"""Build a safe, lineage-preserving usable dataset from an inspected upload.

Exactly one exclusion pass is performed.  Row-scoped issues are removed using
their original sheet/row references, dependent rows are removed in the same
pass, and the resulting dataset is then validated once.  A failed final
validation never triggers further automatic deletion.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

import pandas as pd

from services.analysis_pipeline import ensure_recommendations
from services.data_issues import (
    ERROR,
    FILE_BLOCKING,
    HEAVY_EXCLUSION_RATIO,
    WARNING,
    annotate_issue_treatments,
    collect_data_issues,
    exclusion_row_refs,
    issue_policy,
)
from services.data_validator import REQUIRED_COLUMNS, ValidationReport, validate_workbook_data

ANALYSIS_SHEETS = ("stores", "dcs", "products", "inventory", "routes", "recommendations")
CORE_TABLES = ("stores", "products", "inventory", "routes")


def usable_data_signature(data: Mapping[str, Any]) -> str:
    """Deterministic signature of only the frames that can reach analysis."""
    digest = hashlib.sha256()
    for key in sorted(data):
        frame = data[key]
        if not isinstance(frame, pd.DataFrame):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(json.dumps([str(c) for c in frame.columns], ensure_ascii=False).encode("utf-8"))
        stable = frame.copy()
        for column in stable.columns:
            if stable[column].dtype == "object":
                stable[column] = stable[column].map(
                    lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                    if isinstance(value, (dict, list, tuple, set)) else str(value)
                )
        digest.update(pd.util.hash_pandas_object(stable, index=True).values.tobytes())
    return digest.hexdigest()


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip().lower() in {"", "nan", "none"}


def _nonblank_indices(frame: pd.DataFrame) -> list[Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return [idx for idx, row in frame.iterrows() if not all(_is_blank(value) for value in row)]


def _source_counts(raw_data: Mapping[str, Any], normalized: Mapping[str, Any]) -> dict[str, int]:
    basis = raw_data or normalized
    return {
        sheet: len(_nonblank_indices(frame))
        for sheet, frame in basis.items()
        if sheet in ANALYSIS_SHEETS and isinstance(frame, pd.DataFrame)
    }


def _lineage_maps(
    normalized: Mapping[str, Any], raw_data: Mapping[str, Any],
) -> dict[str, dict[Any, tuple[str, int]]]:
    """Map normalized index labels to original sheet rows without adding columns."""
    maps: dict[str, dict[Any, tuple[str, int]]] = {}
    separate_dcs = isinstance(raw_data.get("dcs"), pd.DataFrame) and not raw_data.get("dcs").empty
    for sheet, frame in normalized.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        mapping: dict[Any, tuple[str, int]] = {}
        if sheet == "stores" and separate_dcs:
            store_rows = _nonblank_indices(raw_data.get("stores", pd.DataFrame()))
            dc_rows = _nonblank_indices(raw_data.get("dcs", pd.DataFrame()))
            refs = [("stores", int(idx) + 2) for idx in store_rows] + [("dcs", int(idx) + 2) for idx in dc_rows]
            for idx, ref in zip(frame.index, refs):
                mapping[idx] = ref
        else:
            raw = raw_data.get(sheet)
            raw_rows = _nonblank_indices(raw) if isinstance(raw, pd.DataFrame) else []
            by_label = {idx: (sheet, int(idx) + 2) for idx in raw_rows if isinstance(idx, int)}
            for pos, idx in enumerate(frame.index):
                if idx in by_label:
                    mapping[idx] = by_label[idx]
                elif pos < len(raw_rows):
                    mapping[idx] = (sheet, int(raw_rows[pos]) + 2)
                else:
                    try:
                        mapping[idx] = (sheet, int(idx) + 2)
                    except (TypeError, ValueError):
                        mapping[idx] = (sheet, pos + 2)
        maps[sheet] = mapping
    return maps


def _display(value: Any) -> str:
    if _is_blank(value):
        return "빈 값"
    text = str(value)
    return text if len(text) <= 40 else text[:40] + "…"


def _cascade_issue(
    sheet: str, row: int, column: str, value: Any, message: str, fix: str,
    filename: str, sheet_name: str,
) -> dict[str, Any]:
    policy = issue_policy("orphan_reference")
    return {
        "시트": sheet_name or sheet, "행": row, "컬럼": column, "값": _display(value),
        "구분": policy.severity, "문제": message, "수정 방법": fix,
        "code": "orphan_reference", "issue_code": "orphan_reference",
        "severity": policy.severity, "source_type": "", "source_file": filename,
        "source_sheet": sheet, "source_sheet_name": sheet_name or sheet,
        "source_row_number": row, "source_column_name": column,
        "canonical_column_name": column, "original_value": "" if _is_blank(value) else str(value),
        "normalized_value": "" if _is_blank(value) else str(value),
        "blocks_analysis": policy.blocks_analysis, "scope": policy.scope,
        "row_excludable": True, "retain_after_warning": False,
        "treatment": policy.treatment, "related_rows": [], "exclusion_rows": [row],
        "issue_message": message, "fix_message": fix,
    }


def _drop_refs(
    data: Mapping[str, Any], lineage: Mapping[str, Mapping[Any, tuple[str, int]]],
    refs: set[tuple[str, int]],
) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for table, value in data.items():
        if not isinstance(value, pd.DataFrame):
            cleaned[table] = value
            continue
        mapping = lineage.get(table, {})
        drop_index = [idx for idx in value.index if mapping.get(idx) in refs]
        cleaned[table] = value.drop(index=drop_index).copy()
    return cleaned


def _text_set(frame: pd.DataFrame, column: str, mask: pd.Series | None = None) -> set[str]:
    if column not in frame.columns:
        return set()
    series = frame.loc[mask, column] if mask is not None else frame[column]
    return {str(value).strip() for value in series.dropna() if str(value).strip()}


def _dependent_exclusions(
    data: Mapping[str, Any], lineage: Mapping[str, Mapping[Any, tuple[str, int]]],
    source_metadata: Mapping[str, Any],
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    stores = data.get("stores", pd.DataFrame())
    products = data.get("products", pd.DataFrame())
    inventory = data.get("inventory", pd.DataFrame())
    routes = data.get("routes", pd.DataFrame())
    recs = data.get("recommendations", pd.DataFrame())
    node_ids = _text_set(stores, "node_id")
    node_types = stores.get("node_type", pd.Series(index=stores.index, dtype=str)).astype(str).str.strip().str.upper()
    dc_ids = _text_set(stores, "node_id", node_types == "DC") if not stores.empty else set()
    product_ids = _text_set(products, "product_id")
    filename = str(source_metadata.get("filename") or "")
    sheet_names = dict(source_metadata.get("sheet_names") or {})
    refs: set[tuple[str, int]] = set()
    issues: list[dict[str, Any]] = []

    def reject(table: str, idx: Any, column: str, value: Any, message: str) -> None:
        ref = lineage.get(table, {}).get(idx)
        if not ref or ref in refs:
            return
        refs.add(ref)
        issues.append(_cascade_issue(
            ref[0], ref[1], column, value, message,
            "기준정보와 참조 식별자를 확인하세요.", filename, sheet_names.get(ref[0], ref[0]),
        ))

    if isinstance(inventory, pd.DataFrame):
        for idx, row in inventory.iterrows():
            if str(row.get("store_id", "")).strip() not in node_ids:
                reject("inventory", idx, "store_id", row.get("store_id"), "참조하는 점포가 없어 이 재고 행을 제외합니다.")
            elif str(row.get("product_id", "")).strip() not in product_ids:
                reject("inventory", idx, "product_id", row.get("product_id"), "참조하는 상품이 없어 이 재고 행을 제외합니다.")

    if isinstance(routes, pd.DataFrame):
        for idx, row in routes.iterrows():
            if str(row.get("source_id", "")).strip() not in node_ids:
                reject("routes", idx, "source_id", row.get("source_id"), "출발 기준정보가 없어 이 경로 행을 제외합니다.")
            elif str(row.get("target_id", "")).strip() not in node_ids:
                reject("routes", idx, "target_id", row.get("target_id"), "도착 기준정보가 없어 이 경로 행을 제외합니다.")

    valid_route_indices = [idx for idx in getattr(routes, "index", []) if lineage.get("routes", {}).get(idx) not in refs]
    route_edges = {
        (str(routes.at[idx, "source_id"]).strip(), str(routes.at[idx, "target_id"]).strip())
        for idx in valid_route_indices
        if {"source_id", "target_id"}.issubset(routes.columns)
    }
    direct_paths: set[tuple[str, str]] = set()
    via_paths: set[tuple[str, str, str]] = set()
    for idx in valid_route_indices:
        row = routes.loc[idx]
        source = str(row.get("source_id", "")).strip()
        target = str(row.get("target_id", "")).strip()
        kind = str(row.get("route_type", "")).strip().upper()
        if kind in {"", "DIRECT", "STORE_TO_STORE"}:
            direct_paths.add((source, target))
        if kind == "VIA_DC":
            via_paths.add((source, str(row.get("dc_id", "")).strip(), target))
    if isinstance(recs, pd.DataFrame):
        for idx, row in recs.iterrows():
            source, target = str(row.get("source_id", "")).strip(), str(row.get("target_id", "")).strip()
            product, route_type = str(row.get("product_id", "")).strip(), str(row.get("route_type", "")).strip().upper()
            dc = str(row.get("dc_id", "")).strip()
            if source not in node_ids:
                reject("recommendations", idx, "source_id", row.get("source_id"), "출발 점포가 없어 이 추천 행을 제외합니다.")
            elif target not in node_ids:
                reject("recommendations", idx, "target_id", row.get("target_id"), "도착 점포가 없어 이 추천 행을 제외합니다.")
            elif product not in product_ids:
                reject("recommendations", idx, "product_id", row.get("product_id"), "상품이 없어 이 추천 행을 제외합니다.")
            elif route_type == "VIA_DC" and dc not in dc_ids:
                reject("recommendations", idx, "dc_id", row.get("dc_id"), "유효한 DC가 없어 이 추천 행을 제외합니다.")
            elif route_type == "DIRECT" and (source, target) not in direct_paths:
                reject("recommendations", idx, "route_type", row.get("route_type"), "DIRECT 이동 경로가 없어 이 추천 행을 제외합니다.")
            elif route_type == "VIA_DC" and (
                (source, dc, target) not in via_paths
                and ((source, dc) not in route_edges or (dc, target) not in route_edges)
            ):
                reject("recommendations", idx, "route_type", row.get("route_type"), "DC 경유 이동 경로가 완성되지 않아 이 추천 행을 제외합니다.")
    return refs, issues


def _empty_recommendations() -> pd.DataFrame:
    return pd.DataFrame(columns=list(REQUIRED_COLUMNS["recommendations"]) + ["dc_id", "dc_name"])


def _structural_messages(data: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    for table in CORE_TABLES:
        frame = data.get(table)
        if not isinstance(frame, pd.DataFrame):
            messages.append(f"필수 데이터 `{table}`가 없습니다.")
            continue
        missing = [column for column in REQUIRED_COLUMNS[table] if column not in frame.columns]
        if missing:
            messages.append(f"{table} 필수 컬럼이 없습니다: {', '.join(missing)}")
        if frame.empty:
            messages.append(f"{table}에 데이터 행이 없습니다.")
    recs = data.get("recommendations")
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        missing = [column for column in REQUIRED_COLUMNS["recommendations"] if column not in recs.columns]
        if missing:
            messages.append(f"추천 입력의 필수 컬럼이 없습니다: {', '.join(missing)}")
    return messages


def build_usable_data(
    normalized_data: Mapping[str, Any], raw_data: Mapping[str, Any] | None,
    source_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return full intake details and the only dataset eligible for commit."""
    full = {key: value.copy() if isinstance(value, pd.DataFrame) else value for key, value in normalized_data.items()}
    raw = dict(raw_data or {})
    metadata = dict(source_metadata or {})
    source_counts = _source_counts(raw, full)
    issue_result = collect_data_issues(full, raw, metadata)
    issues = list(issue_result["issues"])
    structural = _structural_messages(full)
    file_blocking = structural or [
        str(item.get("문제") or "데이터 구조를 확인할 수 없습니다.")
        for item in issues if item.get("treatment") == FILE_BLOCKING
    ]
    initial_refs = exclusion_row_refs(issues)
    lineage = _lineage_maps(full, raw)
    first_clean = _drop_refs(full, lineage, initial_refs)
    cascade_refs, cascade_issues = _dependent_exclusions(first_clean, lineage, metadata)
    excluded_refs = initial_refs | cascade_refs
    issues.extend(cascade_issues)
    usable = _drop_refs(full, lineage, excluded_refs)

    recs = usable.get("recommendations")
    recommendation_source = "uploaded" if isinstance(recs, pd.DataFrame) and not recs.empty else "none"
    candidate_info: dict[str, Any] = {}
    if recommendation_source == "none":
        usable, recommendation_source, candidate_info = ensure_recommendations(usable)
        if "recommendations" not in usable or not isinstance(usable.get("recommendations"), pd.DataFrame):
            usable = dict(usable)
            usable["recommendations"] = _empty_recommendations()
    final_validation: ValidationReport = validate_workbook_data(usable)

    table_excluded = Counter(sheet for sheet, _ in excluded_refs)
    table_ratios = {
        table: (table_excluded.get(table, 0) / count if count else 0.0)
        for table, count in source_counts.items()
    }
    total_rows = sum(source_counts.values())
    excluded_rows = len(excluded_refs)
    overall_ratio = excluded_rows / total_rows if total_rows else 0.0
    ratio_blockers = [
        f"{table} 데이터의 절반 이상이 제외됩니다."
        for table, ratio in table_ratios.items() if ratio >= HEAVY_EXCLUSION_RATIO and source_counts.get(table, 0)
    ]
    if overall_ratio >= HEAVY_EXCLUSION_RATIO and total_rows:
        ratio_blockers.append("전체 분석 데이터의 절반 이상이 제외됩니다.")

    all_issues = annotate_issue_treatments(issues, excluded_refs)
    error_refs = {
        (str(item.get("source_sheet") or ""), int(item.get("source_row_number") or 0))
        for item in all_issues if item.get("severity") == ERROR and int(item.get("source_row_number") or 0) >= 2
    }
    warning_refs = {
        (str(item.get("source_sheet") or ""), int(item.get("source_row_number") or 0))
        for item in all_issues if item.get("severity") == WARNING and int(item.get("source_row_number") or 0) >= 2
    }
    warning_included = warning_refs - excluded_refs
    reason_distribution = Counter(
        str(item.get("문제") or "데이터 문제")
        for item in all_issues
        if (str(item.get("source_sheet") or ""), int(item.get("source_row_number") or 0)) in excluded_refs
    )
    blockers = list(dict.fromkeys([*file_blocking, *ratio_blockers]))
    if final_validation.has_errors:
        blockers.extend(message.message for message in final_validation.messages if message.level == ERROR)
    blockers = list(dict.fromkeys(blockers))
    apply_allowed = not blockers
    signature = usable_data_signature(usable)
    quality = {
        "total_rows": total_rows,
        "applied_rows": max(0, total_rows - excluded_rows),
        "usable_rows": max(0, total_rows - excluded_rows),
        "excluded_rows": excluded_rows,
        "warning_included_rows": len(warning_included),
        "error_items": sum(item.get("severity") == ERROR for item in all_issues),
        "warning_items": sum(item.get("severity") == WARNING for item in all_issues),
        "error_rows": len(error_refs),
        "warning_rows": len(warning_refs - error_refs),
        "duplicate_rows": len({
            (str(item.get("source_sheet") or ""), int(row))
            for item in all_issues if item.get("code") in {"exact_duplicate", "conflict_duplicate"}
            for row in (item.get("related_rows") or [])
        }),
        "source_table_rows": source_counts,
        "excluded_by_table": dict(table_excluded),
        "table_exclusion_ratios": table_ratios,
        "overall_exclusion_ratio": overall_ratio,
        "exclusion_reasons": dict(reason_distribution),
        "application_mode": "문제 행 제외 적용" if excluded_rows else "전체 적용",
        "filename": metadata.get("filename") or "-",
        "sheet_names": dict(metadata.get("sheet_names") or {}),
        "blockers": blockers,
    }
    return {
        "normalized_data": full,
        "usable_data": usable,
        "issues": all_issues,
        "excluded_row_refs": [
            {"source_sheet": sheet, "source_row_number": row}
            for sheet, row in sorted(excluded_refs)
        ],
        "quality_summary": quality,
        "validation": final_validation,
        "apply_allowed": apply_allowed,
        "usable_signature": signature,
        "recommendation_source": recommendation_source,
        "candidate_info": candidate_info,
    }
