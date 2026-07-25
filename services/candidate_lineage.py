"""Original-file row lineage for each recommendation candidate.

Given a candidate (product / source / target / route / DC) this module locates
the *original* file rows that fed the candidate's decision — the source-store
stock row, the destination demand row, the product row, the route row(s), and
(for VIA_DC) the DC row — so a user can open their Excel/CSV and find the exact
cell behind a recommendation or an exclusion.

It reuses the *same* row/column resolution the data-management 문제 목록 uses
(``data_issues._resolve_column`` / ``_row_number``), so a value flagged on Excel
행 12 in 데이터 관리 is the same 행 12 cited here (docs/VALIDATION.md 계보 일치).

Nothing is fabricated: when a frame or column is missing, or no row matches, the
reference is returned with ``traceable=False`` and a plain note instead of an
invented row. Lineage metadata never enters VHS or any numeric feature.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from services.data_issues import _is_blank, _resolve_column, _row_number

# User-facing roles (짧은 한글, 내부 코드 아님).
ROLE_SOURCE_STOCK = "출발 재고"
ROLE_TARGET_DEMAND = "도착 수요"
ROLE_PRODUCT = "상품 정보"
ROLE_ROUTE = "경로 정보"
ROLE_DC = "DC 정보"

_ROLE_IMPACT = {
    ROLE_SOURCE_STOCK: "안전재고를 제외한 이동 가능 수량 계산에 사용했습니다.",
    ROLE_TARGET_DEMAND: "도착 점포 부족 수량 계산에 사용했습니다.",
    ROLE_PRODUCT: "상품 단가와 정보를 확인하는 데 사용했습니다.",
    ROLE_ROUTE: "이동 거리·비용·시간 계산에 사용했습니다.",
    ROLE_DC: "경유 DC 정보를 확인하는 데 사용했습니다.",
}

_UNTRACEABLE_NOTE = "원본에서 해당 행을 찾을 수 없어 위치를 표시할 수 없습니다."
_NO_SHEET_NOTE = "원본 파일에 해당 시트가 없어 위치를 추적할 수 없습니다."
_NO_COLUMN_NOTE = "원본 시트에 해당 컬럼이 없어 위치를 추적할 수 없습니다."


def _frame(source: Mapping[str, Any], sheet: str) -> pd.DataFrame | None:
    frame = (source or {}).get(sheet)
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        return frame
    return None


def _match_rows(
    frame: pd.DataFrame, sheet: str, matchers: list[tuple[str, Any]]
) -> tuple[list[int], str | None] | None:
    """Rows (1-based file row numbers) where every (standard_col == value) holds.

    Returns (rows, column_used_for_display) or None when a needed column is
    absent (→ untraceable). ``column_used`` is the original column name of the
    first matcher so the UI can name the cell the user should open.
    """
    resolved: dict[str, str] = {}
    for standard, _ in matchers:
        column = _resolve_column(frame, sheet, standard)
        if column is None:
            return None
        resolved[standard] = column
    rows: list[int] = []
    for position, index in enumerate(frame.index):
        ok = True
        for standard, value in matchers:
            cell = frame.at[index, resolved[standard]]
            if _is_blank(cell) or str(cell).strip() != str(value).strip():
                ok = False
                break
        if ok:
            rows.append(_row_number(index, position))
    display_column = resolved[matchers[0][0]] if matchers else None
    return sorted(set(rows)), display_column


def _value_at(
    frame: pd.DataFrame, sheet: str, row_number: int, standard: str
) -> tuple[Any, str | None]:
    column = _resolve_column(frame, sheet, standard)
    if column is None:
        return None, None
    for position, index in enumerate(frame.index):
        if _row_number(index, position) == row_number:
            value = frame.at[index, column]
            return (None if _is_blank(value) else value), column
    return None, column


def _reference(
    role: str,
    source: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sheet: str,
    matchers: list[tuple[str, Any]],
    value_column: str,
) -> dict[str, Any]:
    filename = str((metadata or {}).get("filename") or "-")
    sheet_names = (metadata or {}).get("sheet_names") or {}
    sheet_name = str(sheet_names.get(sheet, sheet))
    base = {
        "role": role,
        "file": filename,
        "sheet": sheet,
        "sheet_name": sheet_name,
        "impact": _ROLE_IMPACT.get(role, ""),
        "rows": [],
        "column": None,
        "canonical_column": value_column,
        "value": None,
        "traceable": False,
        "note": _UNTRACEABLE_NOTE,
    }
    frame = _frame(source, sheet)
    if frame is None:
        base["note"] = _NO_SHEET_NOTE
        return base
    matched = _match_rows(frame, sheet, matchers)
    if matched is None:
        base["note"] = _NO_COLUMN_NOTE
        return base
    rows, key_column = matched
    if not rows:
        base["note"] = _UNTRACEABLE_NOTE
        return base
    value, value_col = _value_at(frame, sheet, rows[0], value_column)
    base.update({
        "rows": rows,
        "column": value_col or key_column,
        "value": None if value is None else str(value),
        "traceable": True,
        "note": "",
    })
    return base


def build_source_references(
    candidate: Mapping[str, Any],
    raw_data: Mapping[str, Any] | None,
    source_metadata: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return every locatable original-file reference for one candidate.

    ``raw_data`` is the pre-normalization snapshot (original Korean/aliased
    headers, original row order). When it is unavailable, callers may pass the
    normalized frames; resolution still works but shows normalized values.
    """
    source = dict(raw_data or {})
    metadata = dict(source_metadata or {})
    product = candidate.get("product_id")
    src = candidate.get("source_id")
    tgt = candidate.get("target_id")
    dc = candidate.get("dc_id")
    route_type = str(candidate.get("route_type") or "").upper()

    references: list[dict[str, Any]] = []
    if src is not None and product is not None:
        references.append(_reference(
            ROLE_SOURCE_STOCK, source, metadata, "inventory",
            [("store_id", src), ("product_id", product)], "stock_qty",
        ))
    if tgt is not None and product is not None:
        references.append(_reference(
            ROLE_TARGET_DEMAND, source, metadata, "inventory",
            [("store_id", tgt), ("product_id", product)], "demand_qty",
        ))
    if product is not None:
        references.append(_reference(
            ROLE_PRODUCT, source, metadata, "products",
            [("product_id", product)], "product_name",
        ))
    if src is not None and tgt is not None:
        if route_type == "VIA_DC" and dc:
            references.append(_reference(
                ROLE_ROUTE, source, metadata, "routes",
                [("source_id", src), ("target_id", dc)], "estimated_cost",
            ))
            references.append(_reference(
                ROLE_ROUTE, source, metadata, "routes",
                [("source_id", dc), ("target_id", tgt)], "estimated_cost",
            ))
        else:
            references.append(_reference(
                ROLE_ROUTE, source, metadata, "routes",
                [("source_id", src), ("target_id", tgt)], "estimated_cost",
            ))
    if route_type == "VIA_DC" and dc:
        references.append(_reference(
            ROLE_DC, source, metadata, "stores",
            [("node_id", dc)], "node_name",
        ))
    return references


def source_reference_rows(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten references for a compact folded detail table (원본 위치 상세)."""
    rows: list[dict[str, Any]] = []
    for ref in references:
        rows.append({
            "역할": ref.get("role", "-"),
            "파일": ref.get("file", "-"),
            "시트": ref.get("sheet_name", "-"),
            "원본 행": ", ".join(str(r) for r in ref.get("rows") or []) if ref.get("traceable") else "추적 불가",
            "컬럼": ref.get("column") or "-",
            "입력값": ref.get("value") if ref.get("traceable") else "-",
            "영향": ref.get("impact", "-") if ref.get("traceable") else ref.get("note", "-"),
        })
    return rows


def traceable_row_count(references: list[dict[str, Any]]) -> int:
    return sum(len(ref.get("rows") or []) for ref in references if ref.get("traceable"))
