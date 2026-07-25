"""Row-level data issue collection anchored to the *original* uploaded file.

The sheet-level gate lives in ``data_validator``; this module adds *row-level*
detail so a user can open their file and fix the exact cell. It never mutates
data and never changes the validation verdict — it is an explanatory layer that
powers a compact on-screen summary (counts + top issues) and a fixable CSV.

Original-vs-normalized separation
---------------------------------
When a raw snapshot is available (``raw_data`` captured at load time, before any
alias/coercion), issues are detected against the *original* values so the user
sees what they actually typed ("십오", "1,500개", 빈 값) at the *original* file
row/column — not the post-normalization NaN or the standard column name.
Analysis still runs only on the separate normalized frames.

Row numbers are 1-based spreadsheet rows (header = row 1, so the first data row
is row 2). Blank rows are skipped but never renumber the rows below them.

Severity ↔ gate consistency
---------------------------
Each issue carries ``severity`` (오류/경고) and ``blocks_analysis`` from one
central policy (``ISSUE_POLICY``). Every *blocking* code has a matching
``data_validator`` ERROR, so ``summary["has_blocking"]`` and the real apply gate
(``validation.has_errors``) always agree.
"""
from __future__ import annotations

import io
import math
from typing import Any, Mapping

import pandas as pd

from services.column_aliases import SHEET_ALIASES, _canonical, clean_numeric_value

ERROR = "오류"
WARNING = "경고"

# Single source of truth: issue_code -> (severity, blocks_analysis).
# error  <=> blocks_analysis=True  (행 사용 불가, 반드시 수정)
# warning <=> blocks_analysis=False (확인 필요, 정책상 해당 행 제외 후 분석 가능)
ISSUE_POLICY: dict[str, tuple[str, bool]] = {
    "missing_id": (ERROR, True),
    "non_numeric": (ERROR, True),
    "negative": (ERROR, True),
    "zero_quantity": (ERROR, True),
    "same_source_target": (ERROR, True),
    "conflict_duplicate": (WARNING, False),
    "exact_duplicate": (WARNING, False),
    "alias_conflict": (WARNING, False),
    "id_numeric": (WARNING, False),
}

# When more than this share of the original rows would be excluded, the file is
# treated as 사용 불가(강한 확인 필요) rather than a soft warning. Documented in
# docs/VALIDATION.md.
HEAVY_EXCLUSION_RATIO = 0.5

_VALUE_DISPLAY_LIMIT = 40

# Numeric fields checked per sheet: (standard_column, must_be_positive).
_NUMERIC_CHECKS = {
    "inventory": (("stock_qty", False),),
    "routes": (("distance_km", False), ("estimated_cost", False), ("travel_time_min", False)),
    "recommendations": (("recommended_qty", True), ("estimated_cost", False), ("expected_saving", False)),
}
# Identifier fields that must not be blank: sheet -> standard columns.
_REQUIRED_IDS = {
    "inventory": ("store_id", "product_id"),
    "recommendations": ("route_id", "product_id", "source_id", "target_id"),
    "routes": ("source_id", "target_id"),
}
# Identifier columns whose original type is checked for leading-zero loss.
_ID_COLUMNS = {
    "stores": ("node_id",),
    "inventory": ("store_id", "product_id"),
    "routes": ("source_id", "target_id"),
    "recommendations": ("route_id", "product_id", "source_id", "target_id"),
}
# Duplicate keys: sheet -> (key columns, value column that matters for conflicts).
_DUPLICATE_KEYS = {
    "inventory": (("store_id", "product_id"), "stock_qty"),
}
# Canonical fragments that mark a column as a time/snapshot dimension. Rows that
# share a key but differ on such a column are legitimate multi-row (시계열), not
# duplicates.
_TIME_MARKERS = ("date", "일자", "기준일", "snapshot", "시점", "월", "week", "주차")


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def _original_text(value: Any) -> str:
    """Full original value as text; blank -> empty string (never 'nan'/None)."""
    return "" if _is_blank(value) else str(value)


def _display_value(value: Any) -> str:
    """User-facing input value: blank -> '빈 값'; long values truncated."""
    if _is_blank(value):
        return "빈 값"
    text = str(value)
    if len(text) > _VALUE_DISPLAY_LIMIT:
        return text[:_VALUE_DISPLAY_LIMIT] + "…"
    return text


def _normalized_numeric(value: Any) -> str:
    """How the numeric normalizer would render the value, for display."""
    if _is_blank(value):
        return "빈 값"
    number = clean_numeric_value(value)
    if number is None:
        return "변환 실패"
    return str(int(number)) if float(number).is_integer() else str(number)


def _num(value: Any) -> float | None:
    number = clean_numeric_value(value)
    return number if number is not None and math.isfinite(number) else None


def _canonical_lookup(columns: Any) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in columns:
        lookup.setdefault(_canonical(column), str(column))
    return lookup


def _resolve_column(frame: pd.DataFrame, sheet: str, standard: str) -> str | None:
    """Find the original column in ``frame`` that fills the standard concept.

    Works for both a raw snapshot (Korean/aliased headers) and an already
    normalized frame (standard headers). Returns the original column name so the
    user can locate it in their file.
    """
    if standard in frame.columns:
        return standard
    lookup = _canonical_lookup(frame.columns)
    for alias in SHEET_ALIASES.get(sheet, {}).get(standard, (standard,)):
        column = lookup.get(_canonical(alias))
        if column is not None:
            return column
    return lookup.get(_canonical(standard))


def _blank_row_positions(frame: pd.DataFrame) -> set:
    """Index labels of fully-blank rows (dropped in normalization, skipped here)."""
    if frame.empty:
        return set()
    mask = frame.apply(lambda row: all(_is_blank(value) for value in row), axis=1)
    return set(frame.index[mask])


def _row_number(index: Any, position: int) -> int:
    """1-based file row (header = 1). Raw snapshots carry a RangeIndex so the
    positional order maps directly; a gapped index (already-normalized frame)
    still yields the original file row."""
    try:
        return int(index) + 2
    except (TypeError, ValueError):
        return position + 2


class _IssueBuilder:
    def __init__(self, sheet: str, sheet_name: str, source_type: str, source_file: str) -> None:
        self.sheet = sheet
        self.sheet_name = sheet_name or sheet
        self.source_type = source_type
        self.source_file = source_file

    def make(
        self, *, row: int, source_column: str | None, canonical_column: str,
        original_value: Any, normalized_value: str, message: str, fix: str, code: str,
        related_rows: list[int] | None = None,
    ) -> dict[str, Any]:
        severity, blocks = ISSUE_POLICY.get(code, (WARNING, False))
        source_column = source_column or canonical_column
        return {
            # Compact on-screen table (사용자에게 익숙한 원본 기준).
            "시트": self.sheet_name,
            "행": row,
            "컬럼": source_column,
            "값": _display_value(original_value),
            "구분": severity,
            "문제": message,
            "수정 방법": fix,
            # Folded detail / CSV / logs.
            "code": code,
            "issue_code": code,
            "severity": severity,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "source_sheet": self.sheet,
            "source_sheet_name": self.sheet_name,
            "source_row_number": row,
            "source_column_name": source_column,
            "canonical_column_name": canonical_column,
            "original_value": _original_text(original_value),
            "normalized_value": normalized_value,
            "blocks_analysis": blocks,
            "related_rows": related_rows or [],
            "fix_message": fix,
        }


def _sheet_source_type(default: str, sheet: str) -> str:
    return default or "excel"


def collect_data_issues(
    data: Mapping[str, Any] | None,
    raw_data: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return structured row issues plus a compact summary for the UI.

    ``data`` is the normalized frame dict (analysis basis). ``raw_data`` is the
    original snapshot; when given, detection/labels use the original file values,
    columns and row numbers. When omitted, ``data`` is used as its own source so
    callers/tests that pass raw-ish frames keep working unchanged.
    """
    normalized = dict(data or {})
    source_frames = dict(raw_data) if raw_data else normalized
    meta = dict(source_metadata or {})
    default_source_type = str(meta.get("source_type") or "")
    sheet_names = dict(meta.get("sheet_names") or {})
    source_file = str(meta.get("filename") or "")

    issues: list[dict[str, Any]] = []
    error_rows: set[tuple[str, int]] = set()
    warning_rows: set[tuple[str, int]] = set()

    for sheet, frame in source_frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        builder = _IssueBuilder(
            sheet, sheet_names.get(sheet, sheet),
            _sheet_source_type(default_source_type, sheet), source_file,
        )
        blank_positions = _blank_row_positions(frame)

        def record(issue: dict[str, Any]) -> None:
            issues.append(issue)
            key = (sheet, issue["행"])
            (error_rows if issue["severity"] == ERROR else warning_rows).add(key)

        _check_required_ids(frame, sheet, builder, blank_positions, record)
        _check_numeric(frame, sheet, builder, blank_positions, record)
        _check_same_source_target(frame, sheet, builder, blank_positions, record)
        _check_duplicates(frame, sheet, builder, blank_positions, record)
        _check_alias_conflicts(frame, sheet, builder, record)
        _check_identifier_types(frame, sheet, builder, blank_positions, record)

    total_rows = sum(
        int((~frame.apply(lambda r: all(_is_blank(v) for v in r), axis=1)).sum())
        for frame in source_frames.values()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    )
    error_count = len(error_rows)
    warning_count = len(warning_rows - error_rows)
    excluded_rows = error_count
    usable_rows = max(0, total_rows - excluded_rows)
    mostly_excluded = total_rows > 0 and (excluded_rows / total_rows) > HEAVY_EXCLUSION_RATIO
    summary = {
        "total_rows": total_rows,
        "total_issues": len(issues),
        "error_rows": error_count,
        "warning_rows": warning_count,
        "usable_rows": usable_rows,
        "warning_included_rows": usable_rows,
        "excluded_rows": excluded_rows,
        "duplicate_rows": sum(1 for i in issues if i["code"] in ("conflict_duplicate", "exact_duplicate")),
        "top": _rank_issues(issues)[:5],
        "has_blocking": error_count > 0 or mostly_excluded,
        "mostly_excluded": mostly_excluded,
    }
    return {"issues": issues, "summary": summary}


def _check_required_ids(frame, sheet, builder, blank_positions, record) -> None:
    for standard in _REQUIRED_IDS.get(sheet, ()):
        column = _resolve_column(frame, sheet, standard)
        if column is None:
            continue
        for position, (index, value) in enumerate(frame[column].items()):
            if index in blank_positions:
                continue
            if _is_blank(value):
                record(builder.make(
                    row=_row_number(index, position), source_column=column,
                    canonical_column=standard, original_value=value, normalized_value="빈 값",
                    message="값이 비어 있어 항목을 구분할 수 없습니다.",
                    fix=f"{column} 값을 입력한 뒤 다시 업로드하세요.", code="missing_id",
                ))


def _check_numeric(frame, sheet, builder, blank_positions, record) -> None:
    for standard, positive in _NUMERIC_CHECKS.get(sheet, ()):
        column = _resolve_column(frame, sheet, standard)
        if column is None:
            continue
        for position, (index, value) in enumerate(frame[column].items()):
            if index in blank_positions or _is_blank(value):
                continue
            row = _row_number(index, position)
            number = _num(value)
            if number is None:
                record(builder.make(
                    row=row, source_column=column, canonical_column=standard,
                    original_value=value, normalized_value="변환 실패",
                    message="숫자로 읽을 수 없습니다.", fix="해당 값을 숫자로 수정하세요.", code="non_numeric",
                ))
            elif number < 0:
                record(builder.make(
                    row=row, source_column=column, canonical_column=standard,
                    original_value=value, normalized_value=_normalized_numeric(value),
                    message="음수 값입니다.", fix="0 이상의 값으로 수정하세요.", code="negative",
                ))
            elif positive and number == 0:
                record(builder.make(
                    row=row, source_column=column, canonical_column=standard,
                    original_value=value, normalized_value="0",
                    message="이동 수량이 0입니다.", fix="1 이상으로 수정하세요.", code="zero_quantity",
                ))


def _check_same_source_target(frame, sheet, builder, blank_positions, record) -> None:
    if sheet != "recommendations":
        return
    src_col = _resolve_column(frame, sheet, "source_id")
    tgt_col = _resolve_column(frame, sheet, "target_id")
    if not src_col or not tgt_col:
        return
    for position, index in enumerate(frame.index):
        if index in blank_positions:
            continue
        src = frame.at[index, src_col]
        tgt = frame.at[index, tgt_col]
        if not _is_blank(src) and str(src).strip() == str(tgt).strip():
            record(builder.make(
                row=_row_number(index, position), source_column=tgt_col,
                canonical_column="target_id", original_value=tgt, normalized_value=_original_text(tgt),
                message="출발지와 도착지가 같습니다.", fix="도착지를 다른 점포로 수정하세요.", code="same_source_target",
            ))


def _check_duplicates(frame, sheet, builder, blank_positions, record) -> None:
    spec = _DUPLICATE_KEYS.get(sheet)
    if not spec:
        return
    key_standards, value_standard = spec
    key_cols = [_resolve_column(frame, sheet, s) for s in key_standards]
    value_col = _resolve_column(frame, sheet, value_standard)
    if any(c is None for c in key_cols) or value_col is None:
        return
    time_cols = [str(c) for c in frame.columns if any(m in _canonical(c) for m in _TIME_MARKERS)]
    live_index = [i for i in frame.index if i not in blank_positions]
    if not live_index:
        return
    subset = frame.loc[live_index]
    positions = {index: pos for pos, index in enumerate(frame.index)}
    grouped = subset.groupby([subset[c].astype(str).str.strip() for c in key_cols], sort=False)
    for _, group in grouped:
        if len(group) < 2:
            continue
        # Legitimate multi-row (시계열): distinguished by a time/snapshot column.
        if any(group[tc].astype(str).nunique() > 1 for tc in time_cols):
            continue
        rows = sorted(_row_number(idx, positions[idx]) for idx in group.index)
        distinct_values = {str(v).strip() for v in group[value_col] if not _is_blank(v)}
        row_phrase = "·".join(f"{r}행" for r in rows)
        if len(distinct_values) > 1:
            message = f"{row_phrase}에 같은 점포·상품 데이터가 있으며 값이 다릅니다."
            fix = "여러 행 중 올바른 데이터를 확인해 하나로 정리하세요."
            code = "conflict_duplicate"
        else:
            message = f"{row_phrase}에 완전히 동일한 데이터가 중복되어 있습니다."
            fix = "중복 행 중 하나만 남기세요."
            code = "exact_duplicate"
        for idx in group.index:
            record(builder.make(
                row=_row_number(idx, positions[idx]), source_column=value_col,
                canonical_column=value_standard, original_value=frame.at[idx, value_col],
                normalized_value=_normalized_numeric(frame.at[idx, value_col]),
                message=message, fix=fix, code=code, related_rows=rows,
            ))


def _check_alias_conflicts(frame, sheet, builder, record) -> None:
    """Two distinct original columns collapsing onto one absent standard column."""
    alias_map = SHEET_ALIASES.get(sheet)
    if not alias_map:
        return
    present_canon = {_canonical(c) for c in frame.columns}
    lookup = _canonical_lookup(frame.columns)
    for standard, aliases in alias_map.items():
        if _canonical(standard) in present_canon:
            continue  # the standard column itself exists — no ambiguity, it wins
        matched: list[str] = []
        for alias in aliases:
            column = lookup.get(_canonical(alias))
            if column is not None and column not in matched:
                matched.append(column)
        if len(matched) >= 2:
            names = "·".join(f"'{c}'" for c in matched)
            record(builder.make(
                row=1, source_column=", ".join(matched), canonical_column=standard,
                original_value="", normalized_value=standard,
                message=f"{names} 컬럼이 같은 항목으로 인식됐습니다.",
                fix="한 컬럼만 남기거나 서로 다른 의미라면 컬럼명을 구분하세요.", code="alias_conflict",
            ))


def _check_identifier_types(frame, sheet, builder, blank_positions, record) -> None:
    """Warn when an identifier column was stored as a number (leading zeros may
    have been dropped by Excel before Varo ever read the file)."""
    for standard in _ID_COLUMNS.get(sheet, ()):
        column = _resolve_column(frame, sheet, standard)
        if column is None or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        rows = [
            _row_number(index, position)
            for position, (index, value) in enumerate(frame[column].items())
            if index not in blank_positions and not _is_blank(value)
        ]
        if not rows:
            continue
        record(builder.make(
            row=rows[0], source_column=column, canonical_column=standard,
            original_value=frame[column].iloc[0],
            normalized_value=_original_text(frame[column].iloc[0]),
            message="식별자가 숫자로 저장되어 앞자리 0이 사라졌을 수 있습니다.",
            fix="원본에서 해당 컬럼을 텍스트 서식으로 바꾼 뒤 다시 저장하세요.",
            code="id_numeric", related_rows=rows,
        ))


def _rank_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(issues, key=lambda item: 0 if item["구분"] == ERROR else 1)


# On-screen compact table columns (원본 기준, 내부 코드명 미노출).
_DISPLAY_COLUMNS = ["시트", "행", "컬럼", "값", "구분", "문제", "수정 방법"]
# Folded detail table columns (adds 표준 컬럼/정규화 값/차단/관련 행).
_DETAIL_COLUMNS = [
    "시트", "행", "컬럼", "값", "구분", "문제", "수정 방법",
    "표준 컬럼", "정규화 값", "분석 차단", "관련 행",
]
# Fixable CSV columns (원본 위치 + 수정 방법 + 전체 원본 값).
_CSV_COLUMNS = [
    "파일명", "시트명", "원본 행 번호", "원본 컬럼명", "입력값", "정규화 값",
    "구분", "문제", "수정 방법", "오류 코드", "분석 차단", "관련 행",
]


def _related_text(related: Any) -> str:
    return ", ".join(str(r) for r in related) if related else ""


def display_rows(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: item.get(key, "") for key in _DISPLAY_COLUMNS} for item in issues]


def detail_rows(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in issues:
        row = {key: item.get(key, "") for key in _DISPLAY_COLUMNS}
        row["표준 컬럼"] = item.get("canonical_column_name", "")
        row["정규화 값"] = item.get("normalized_value", "")
        row["분석 차단"] = "예" if item.get("blocks_analysis") else "아니오"
        row["관련 행"] = _related_text(item.get("related_rows"))
        rows.append({key: row.get(key, "") for key in _DETAIL_COLUMNS})
    return rows


def issues_to_csv_bytes(issues: list[dict[str, Any]]) -> bytes:
    """Fixable CSV with original file location + full original value (UTF-8 BOM)."""
    records = [
        {
            "파일명": item.get("source_file", ""),
            "시트명": item.get("source_sheet_name", item.get("시트", "")),
            "원본 행 번호": item.get("source_row_number", item.get("행", "")),
            "원본 컬럼명": item.get("source_column_name", item.get("컬럼", "")),
            "입력값": item.get("original_value", ""),  # full value, not truncated
            "정규화 값": item.get("normalized_value", ""),
            "구분": item.get("구분", ""),
            "문제": item.get("문제", ""),
            "수정 방법": item.get("수정 방법", ""),
            "오류 코드": item.get("issue_code", item.get("code", "")),
            "분석 차단": "예" if item.get("blocks_analysis") else "아니오",
            "관련 행": _related_text(item.get("related_rows")),
        }
        for item in issues
    ]
    frame = pd.DataFrame(records, columns=_CSV_COLUMNS) if records else pd.DataFrame(columns=_CSV_COLUMNS)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")
