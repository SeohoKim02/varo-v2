"""Row-level data issue collection anchored to the *original* uploaded file.

The sheet-level gate lives in ``data_validator``; this module adds *row-level*
detail so a user can open their file and fix the exact cell. It never mutates
data. The partial-data builder consumes its explicit policy fields and original
row references to create a separate usable dataset.

Original-vs-normalized separation
---------------------------------
When a raw snapshot is available (``raw_data`` captured at load time, before any
alias/coercion), issues are detected against the *original* values so the user
sees what they actually typed ("십오", "1,500개", 빈 값) at the *original* file
row/column — not the post-normalization NaN or the standard column name.
Analysis still runs only on the separate normalized frames.

Row numbers are 1-based spreadsheet rows (header = row 1, so the first data row
is row 2). Blank rows are skipped but never renumber the rows below them.

Severity and treatment are independent. Each issue carries both user-facing
severity and a centralized treatment contract (file block, row exclusion,
retained warning, or information).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from services.column_aliases import SHEET_ALIASES, _canonical, clean_numeric_value

ERROR = "오류"
WARNING = "경고"
INFORMATION = "정보"

FILE_BLOCKING = "file_blocking"
ROW_EXCLUDABLE = "row_excludable"
ROW_WARNING = "row_warning"
INFORMATIONAL = "informational"


@dataclass(frozen=True)
class IssuePolicy:
    """One shared treatment contract for validators, intake, UI and exports."""

    severity: str
    blocks_analysis: bool
    scope: str
    row_excludable: bool
    retain_after_warning: bool
    treatment: str
    issue_message: str
    fix_message: str


_UNKNOWN_POLICY = IssuePolicy(
    ERROR, True, "file", False, False, FILE_BLOCKING,
    "데이터 구조를 안전하게 확인할 수 없습니다.",
    "문제를 수정한 뒤 파일을 다시 업로드하세요.",
)

# Single source of truth. Severity, analysis gate and exclusion scope are
# deliberately independent: an ERROR may be safely removable at row scope,
# while a sheet-level alias conflict blocks the whole file.
ISSUE_POLICY: dict[str, IssuePolicy] = {
    "missing_id": IssuePolicy(ERROR, True, "cell", True, False, ROW_EXCLUDABLE, "필수 식별자가 없습니다.", "식별자를 입력하세요."),
    "missing_required_value": IssuePolicy(ERROR, True, "cell", True, False, ROW_EXCLUDABLE, "필수 값이 없습니다.", "필수 값을 입력하세요."),
    "non_numeric": IssuePolicy(ERROR, True, "cell", True, False, ROW_EXCLUDABLE, "숫자로 읽을 수 없습니다.", "숫자 값으로 수정하세요."),
    "negative": IssuePolicy(ERROR, True, "cell", True, False, ROW_EXCLUDABLE, "음수 값입니다.", "0 이상의 값으로 수정하세요."),
    "zero_quantity": IssuePolicy(ERROR, True, "cell", True, False, ROW_EXCLUDABLE, "이동 수량이 0입니다.", "1 이상의 값으로 수정하세요."),
    "same_source_target": IssuePolicy(ERROR, True, "row", True, False, ROW_EXCLUDABLE, "출발지와 도착지가 같습니다.", "서로 다른 점포를 입력하세요."),
    "invalid_node_type": IssuePolicy(ERROR, True, "cell", True, False, ROW_EXCLUDABLE, "점포 유형을 확인할 수 없습니다.", "DC 또는 STORE로 수정하세요."),
    "invalid_route_type": IssuePolicy(ERROR, True, "cell", True, False, ROW_EXCLUDABLE, "경로 유형을 확인할 수 없습니다.", "DIRECT 또는 VIA_DC로 수정하세요."),
    "missing_dc": IssuePolicy(ERROR, True, "row", True, False, ROW_EXCLUDABLE, "물류센터 경유 정보가 올바르지 않습니다.", "유효한 DC 정보를 입력하세요."),
    "orphan_reference": IssuePolicy(ERROR, True, "row", True, False, ROW_EXCLUDABLE, "참조하는 기준정보가 없습니다.", "점포·상품·DC 식별자를 확인하세요."),
    "missing_route_path": IssuePolicy(ERROR, True, "row", True, False, ROW_EXCLUDABLE, "추천에 필요한 경로 정보가 없습니다.", "경로 시트의 연결 정보를 확인하세요."),
    "conflict_duplicate": IssuePolicy(WARNING, False, "row", True, False, ROW_EXCLUDABLE, "같은 키의 값이 서로 다릅니다.", "관련 행을 확인해 하나의 올바른 값으로 정리하세요."),
    "exact_duplicate": IssuePolicy(WARNING, False, "row", True, False, ROW_EXCLUDABLE, "동일한 행이 중복되어 있습니다.", "중복 행을 하나만 남기세요."),
    "alias_conflict": IssuePolicy(ERROR, True, "sheet", False, False, FILE_BLOCKING, "여러 원본 컬럼이 같은 필수 항목과 충돌합니다.", "충돌하는 컬럼을 정리한 뒤 다시 업로드하세요."),
    "id_numeric": IssuePolicy(WARNING, False, "cell", False, True, ROW_WARNING, "식별자가 숫자로 저장되어 있습니다.", "앞자리 0이 필요하면 텍스트 형식으로 저장하세요."),
    "blank_rows_removed": IssuePolicy(INFORMATION, False, "sheet", False, True, INFORMATIONAL, "빈 행을 건너뛰었습니다.", "수정하지 않아도 됩니다."),
}


def issue_policy(code: object) -> IssuePolicy:
    """Return a conservative policy; unknown codes can never bypass the gate."""
    return ISSUE_POLICY.get(str(code or ""), _UNKNOWN_POLICY)

# When more than this share of the original rows would be excluded, the file is
# treated as 사용 불가(강한 확인 필요) rather than a soft warning. Documented in
# docs/VALIDATION.md.
HEAVY_EXCLUSION_RATIO = 0.5

_VALUE_DISPLAY_LIMIT = 40

# Numeric fields checked per sheet: (standard_column, must_be_positive).
_NUMERIC_CHECKS = {
    "inventory": (("stock_qty", False),),
    "routes": (("distance_km", False), ("estimated_cost", False), ("travel_time_min", False)),
    "recommendations": (
        ("recommended_qty", True), ("estimated_cost", False), ("expected_saving", False),
        ("distance_km", False), ("travel_time_min", False),
    ),
}
# Identifier fields that must not be blank: sheet -> standard columns.
_REQUIRED_IDS = {
    "stores": ("node_id",),
    "dcs": ("dc_id",),
    "products": ("product_id",),
    "inventory": ("store_id", "product_id"),
    "recommendations": ("route_id", "product_id", "source_id", "target_id"),
    "routes": ("source_id", "target_id"),
}
_REQUIRED_VALUES = {
    "stores": ("node_name", "node_type"),
    "dcs": ("dc_name",),
    "products": ("product_name",),
}
# Identifier columns whose original type is checked for leading-zero loss.
_ID_COLUMNS = {
    "stores": ("node_id",),
    "dcs": ("dc_id",),
    "products": ("product_id",),
    "inventory": ("store_id", "product_id"),
    "routes": ("source_id", "target_id"),
    "recommendations": ("route_id", "product_id", "source_id", "target_id"),
}
# Duplicate keys: sheet -> (key columns, value columns used to distinguish an
# exact duplicate from a conflict). Exact duplicates keep the first original row;
# conflicting groups exclude every related row because no value is preferred.
_DUPLICATE_KEYS = {
    "stores": (("node_id",), ("node_name", "node_type")),
    "dcs": (("dc_id",), ("dc_name",)),
    "products": (("product_id",), ("product_name",)),
    "inventory": (("store_id", "product_id"), ("stock_qty",)),
    "routes": (("source_id", "target_id"), ("distance_km", "estimated_cost", "travel_time_min")),
    "recommendations": (
        ("route_id",),
        ("product_id", "source_id", "target_id", "route_type", "dc_id", "recommended_qty"),
    ),
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
        exclusion_rows: list[int] | None = None,
    ) -> dict[str, Any]:
        policy = issue_policy(code)
        source_column = source_column or canonical_column
        return {
            # Compact on-screen table (사용자에게 익숙한 원본 기준).
            "시트": self.sheet_name,
            "행": row,
            "컬럼": source_column,
            "값": _display_value(original_value),
            "구분": policy.severity,
            "문제": message or policy.issue_message,
            "수정 방법": fix or policy.fix_message,
            # Folded detail / CSV / logs.
            "code": code,
            "issue_code": code,
            "severity": policy.severity,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "source_sheet": self.sheet,
            "source_sheet_name": self.sheet_name,
            "source_row_number": row,
            "source_column_name": source_column,
            "canonical_column_name": canonical_column,
            "original_value": _original_text(original_value),
            "normalized_value": normalized_value,
            "blocks_analysis": policy.blocks_analysis,
            "scope": policy.scope,
            "row_excludable": policy.row_excludable,
            "retain_after_warning": policy.retain_after_warning,
            "treatment": policy.treatment,
            "related_rows": related_rows or [],
            "exclusion_rows": exclusion_rows if exclusion_rows is not None else (
                [row] if policy.row_excludable and row >= 2 else []
            ),
            "issue_message": message or policy.issue_message,
            "fix_message": fix,
        }


def exclusion_row_refs(issues: list[Mapping[str, Any]]) -> set[tuple[str, int]]:
    """Unique original rows selected by the shared row-exclusion policy."""
    refs: set[tuple[str, int]] = set()
    for item in issues:
        policy = issue_policy(item.get("issue_code") or item.get("code"))
        if not policy.row_excludable:
            continue
        sheet = str(item.get("source_sheet") or "")
        for value in item.get("exclusion_rows") or [item.get("source_row_number")]:
            try:
                row = int(value)
            except (TypeError, ValueError):
                continue
            if sheet and row >= 2:
                refs.add((sheet, row))
    return refs


def annotate_issue_treatments(
    issues: list[dict[str, Any]], excluded_refs: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Attach user-facing outcomes without exposing internal policy names."""
    excluded = excluded_refs if excluded_refs is not None else exclusion_row_refs(issues)
    annotated: list[dict[str, Any]] = []
    for raw in issues:
        item = dict(raw)
        policy = issue_policy(item.get("issue_code") or item.get("code"))
        try:
            source_row = int(item.get("source_row_number") or 0)
        except (TypeError, ValueError):
            source_row = 0
        ref = (str(item.get("source_sheet") or ""), source_row)
        if policy.treatment == FILE_BLOCKING:
            result, included = "파일 사용 차단", "아니오"
        elif ref in excluded:
            result, included = "제외", "아니오"
        else:
            result, included = "경고만 표시", "예"
        item["처리 결과"] = result
        item["적용 데이터 포함 여부"] = included
        annotated.append(item)
    return annotated


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
            if issue["severity"] == ERROR:
                error_rows.add(key)
            elif issue["severity"] == WARNING:
                warning_rows.add(key)

        _check_required_ids(frame, sheet, builder, blank_positions, record)
        _check_required_values(frame, sheet, builder, blank_positions, record)
        _check_numeric(frame, sheet, builder, blank_positions, record)
        _check_node_type(
            frame, sheet, builder, blank_positions, record,
            separate_dcs=isinstance(source_frames.get("dcs"), pd.DataFrame),
        )
        _check_route_type_and_dc(frame, sheet, builder, blank_positions, record)
        _check_same_source_target(frame, sheet, builder, blank_positions, record)
        _check_duplicates(frame, sheet, builder, blank_positions, record)
        _check_alias_conflicts(frame, sheet, builder, record)
        _check_identifier_types(frame, sheet, builder, blank_positions, record)

    analysis_sheets = {"stores", "dcs", "products", "inventory", "routes", "recommendations"}
    total_rows = sum(
        int((~frame.apply(lambda r: all(_is_blank(v) for v in r), axis=1)).sum())
        for sheet, frame in source_frames.items()
        if sheet in analysis_sheets
        if isinstance(frame, pd.DataFrame) and not frame.empty
    )
    excluded_refs = exclusion_row_refs(issues)
    issues = annotate_issue_treatments(issues, excluded_refs)
    error_count = len({row for row in error_rows if row[1] >= 2})
    warning_only = {row for row in warning_rows - error_rows if row[1] >= 2}
    warning_included = warning_only - excluded_refs
    warning_count = len(warning_only)
    excluded_rows = len(excluded_refs)
    usable_rows = max(0, total_rows - excluded_rows)
    mostly_excluded = total_rows > 0 and (excluded_rows / total_rows) >= HEAVY_EXCLUSION_RATIO
    file_blocking = [item for item in issues if item.get("treatment") == FILE_BLOCKING]
    duplicate_refs = {
        (str(item.get("source_sheet") or ""), int(row))
        for item in issues if item.get("code") in ("conflict_duplicate", "exact_duplicate")
        for row in (item.get("related_rows") or [item.get("source_row_number")])
        if row
    }
    summary = {
        "total_rows": total_rows,
        "total_issues": len(issues),
        "error_items": sum(item.get("severity") == ERROR for item in issues),
        "warning_items": sum(item.get("severity") == WARNING for item in issues),
        "error_rows": error_count,
        "warning_rows": warning_count,
        "usable_rows": usable_rows,
        "applied_rows": usable_rows,
        "warning_included_rows": len(warning_included),
        "excluded_rows": excluded_rows,
        "duplicate_rows": len(duplicate_refs),
        "excluded_row_refs": [
            {"source_sheet": sheet, "source_row_number": row}
            for sheet, row in sorted(excluded_refs)
        ],
        "top": _rank_issues(issues)[:5],
        # This is the issue-level gate only. The partial-application builder
        # evaluates the 50% exclusion threshold per table and overall.
        "has_blocking": any(item.get("blocks_analysis") for item in issues) or (
            mostly_excluded and any(item.get("severity") == ERROR for item in issues)
        ),
        "has_file_blocking": bool(file_blocking),
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


def _check_required_values(frame, sheet, builder, blank_positions, record) -> None:
    for standard in _REQUIRED_VALUES.get(sheet, ()):
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
                    message="필수 값이 비어 있습니다.",
                    fix=f"{column} 값을 입력한 뒤 다시 업로드하세요.",
                    code="missing_required_value",
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


def _check_node_type(frame, sheet, builder, blank_positions, record, *, separate_dcs: bool = False) -> None:
    if sheet != "stores":
        return
    if separate_dcs:
        # DQN workbooks use store_type as a business segment and keep real DC
        # rows in a separate dcs sheet; the loader safely assigns STORE/DC.
        return
    column = _resolve_column(frame, sheet, "node_type")
    if column is None:
        return
    for position, (index, value) in enumerate(frame[column].items()):
        if index in blank_positions or _is_blank(value):
            continue
        normalized = str(value).strip().upper()
        if normalized not in {"DC", "STORE"}:
            record(builder.make(
                row=_row_number(index, position), source_column=column,
                canonical_column="node_type", original_value=value, normalized_value=normalized,
                message="점포 유형은 DC 또는 STORE여야 합니다.",
                fix="점포 유형을 DC 또는 STORE로 수정하세요.", code="invalid_node_type",
            ))


def _check_route_type_and_dc(frame, sheet, builder, blank_positions, record) -> None:
    # routes.route_type describes an edge (STORE_TO_STORE/STORE_TO_DC/DC_TO_STORE),
    # while recommendations.route_type is the user decision mode (DIRECT/VIA_DC).
    if sheet != "recommendations":
        return
    type_column = _resolve_column(frame, sheet, "route_type")
    if type_column is None:
        return
    dc_column = _resolve_column(frame, sheet, "dc_id")
    for position, index in enumerate(frame.index):
        if index in blank_positions:
            continue
        value = frame.at[index, type_column]
        if _is_blank(value):
            continue
        normalized = str(value).strip().upper()
        row = _row_number(index, position)
        if normalized not in {"DIRECT", "VIA_DC"}:
            record(builder.make(
                row=row, source_column=type_column, canonical_column="route_type",
                original_value=value, normalized_value=normalized,
                message="지원하지 않는 경로 유형입니다.",
                fix="DIRECT 또는 VIA_DC로 수정하세요.", code="invalid_route_type",
            ))
            continue
        if normalized == "VIA_DC" and (dc_column is None or _is_blank(frame.at[index, dc_column])):
            record(builder.make(
                row=row, source_column=dc_column or "dc_id", canonical_column="dc_id",
                original_value="" if dc_column is None else frame.at[index, dc_column],
                normalized_value="빈 값",
                message="물류센터 경유 경로에 DC 정보가 없습니다.",
                fix="경유할 DC 식별자를 입력하세요.", code="missing_dc",
            ))


def _check_same_source_target(frame, sheet, builder, blank_positions, record) -> None:
    if sheet not in {"routes", "recommendations"}:
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
    key_standards, value_standards = spec
    key_cols = [_resolve_column(frame, sheet, s) for s in key_standards]
    if (
        sheet == "recommendations" and len(key_cols) == 1
        and "recommendation_id" in frame.columns
        and not frame["recommendation_id"].duplicated().any()
    ):
        key_cols = ["recommendation_id"]
    value_pairs = [(standard, _resolve_column(frame, sheet, standard)) for standard in value_standards]
    value_pairs = [(standard, column) for standard, column in value_pairs if column is not None]
    if any(c is None for c in key_cols) or not value_pairs:
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
        distinct_values = {
            tuple(_original_text(group.at[idx, column]).strip() for _, column in value_pairs)
            for idx in group.index
        }
        row_phrase = "·".join(f"{r}행" for r in rows)
        if len(distinct_values) > 1:
            message = f"{row_phrase}에 같은 점포·상품 데이터가 있으며 값이 다릅니다."
            fix = "여러 행 중 올바른 데이터를 확인해 하나로 정리하세요."
            code = "conflict_duplicate"
        else:
            message = f"{row_phrase}에 완전히 동일한 데이터가 중복되어 있습니다."
            fix = "중복 행 중 하나만 남기세요."
            code = "exact_duplicate"
        exclusion_rows = rows if code == "conflict_duplicate" else rows[1:]
        first_standard, first_column = value_pairs[0]
        for idx in group.index:
            record(builder.make(
                row=_row_number(idx, positions[idx]), source_column=first_column,
                canonical_column=first_standard, original_value=frame.at[idx, first_column],
                normalized_value=_original_text(frame.at[idx, first_column]),
                message=message, fix=fix, code=code, related_rows=rows,
                exclusion_rows=exclusion_rows,
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
    order = {ERROR: 0, WARNING: 1, INFORMATION: 2}
    return sorted(
        issues,
        key=lambda item: (
            0 if item.get("treatment") == FILE_BLOCKING else 1,
            order.get(item.get("구분"), 3),
            str(item.get("source_sheet") or ""),
            int(item.get("source_row_number") or 0),
        ),
    )


# On-screen compact table columns (원본 기준, 내부 코드명 미노출).
_DISPLAY_COLUMNS = ["시트", "행", "컬럼", "값", "구분", "문제", "수정 방법"]
# Folded detail table columns (adds 표준 컬럼/정규화 값/차단/관련 행).
_DETAIL_COLUMNS = [
    "시트", "행", "컬럼", "값", "구분", "문제", "수정 방법",
    "처리 결과", "적용 데이터 포함 여부", "표준 컬럼", "정규화 값", "분석 차단", "관련 행",
]
# Fixable CSV columns (원본 위치 + 수정 방법 + 전체 원본 값).
_CSV_COLUMNS = [
    "파일명", "시트명", "원본 행 번호", "원본 컬럼명", "입력값",
    "구분", "문제", "수정 방법", "처리 결과", "적용 데이터 포함 여부", "관련 행",
]


def _related_text(related: Any) -> str:
    return ", ".join(str(r) for r in related) if related else ""


def display_rows(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: item.get(key, "") for key in _DISPLAY_COLUMNS} for item in issues]


def detail_rows(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in issues:
        row = {key: item.get(key, "") for key in _DISPLAY_COLUMNS}
        row["처리 결과"] = item.get("처리 결과", "")
        row["적용 데이터 포함 여부"] = item.get("적용 데이터 포함 여부", "")
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
            "구분": item.get("구분", ""),
            "문제": item.get("문제", ""),
            "수정 방법": item.get("수정 방법", ""),
            "처리 결과": item.get("처리 결과", ""),
            "적용 데이터 포함 여부": item.get("적용 데이터 포함 여부", ""),
            "관련 행": _related_text(item.get("related_rows")),
        }
        for item in issues
    ]
    frame = pd.DataFrame(records, columns=_CSV_COLUMNS) if records else pd.DataFrame(columns=_CSV_COLUMNS)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")
