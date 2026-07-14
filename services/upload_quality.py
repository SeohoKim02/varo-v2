"""Assemble an operator-friendly upload-quality report.

Combines the loader's normalization report with validation results and the
recommendation source. DQN values are never included.
"""
from __future__ import annotations

from typing import Any, Mapping

_REC_SOURCE_LABELS = {
    "uploaded": "업로드 추천 사용",
    "generated": "V2 생성 후보 사용",
    "none": "추천 생성 보류",
}


def build_upload_report(
    load_report: Mapping[str, Any] | None,
    validation: Any,
    recommendation_source: str,
    candidate_info: Mapping[str, Any] | None,
    filename: str | None,
) -> dict[str, Any]:
    load_report = dict(load_report or {})
    messages = list(getattr(validation, "messages", None) or [])
    missing_required = [
        m.message for m in messages
        if getattr(m, "level", "") == "오류" and "필수 컬럼" in getattr(m, "message", "")
    ]
    warnings = [m.message for m in messages if getattr(m, "level", "") == "주의"]
    column_mappings = load_report.get("column_mappings", []) or []
    numeric_failed = load_report.get("numeric_failed", {}) or {}
    blank_removed = load_report.get("blank_removed", {}) or {}
    has_errors = bool(getattr(validation, "has_errors", False))
    return {
        "filename": filename or "-",
        "recognized_sheets": load_report.get("recognized_sheets", []),
        "all_excel_sheets": load_report.get("all_excel_sheets", []),
        "row_counts": load_report.get("row_counts", {}),
        "blank_removed_total": int(sum(blank_removed.values())) if blank_removed else 0,
        "blank_removed": blank_removed,
        "numeric_failed_total": int(sum(numeric_failed.values())) if numeric_failed else 0,
        "numeric_failed": numeric_failed,
        "date_success": int(load_report.get("date_success", 0) or 0),
        "date_failed": int(load_report.get("date_failed", 0) or 0),
        "date_columns": load_report.get("date_columns", []) or [],
        "column_mappings": column_mappings,
        "mapped_column_count": len(column_mappings),
        "missing_required_columns": missing_required,
        "missing_required_count": len(missing_required),
        "warnings": warnings,
        "validation_status": getattr(validation, "status", "-"),
        "recommendation_source": recommendation_source,
        "recommendation_source_label": _REC_SOURCE_LABELS.get(recommendation_source, recommendation_source),
        "candidate_info": dict(candidate_info or {}),
        "analyzable": not has_errors,
        "dqn_excluded": True,
    }


def upload_quality_rows(report: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Flat key/value rows for display and Excel export."""
    report = dict(report or {})
    candidate = report.get("candidate_info") or {}
    rows = [
        ("원본 파일", report.get("filename", "-")),
        ("인식된 시트", ", ".join(report.get("recognized_sheets") or []) or "-"),
        ("자동 매핑 컬럼 수", report.get("mapped_column_count", 0)),
        ("누락 필수 컬럼 수", report.get("missing_required_count", 0)),
        ("숫자 변환 실패 값", report.get("numeric_failed_total", 0)),
        ("빈 행 제거 수", report.get("blank_removed_total", 0)),
        ("날짜 환산 성공", report.get("date_success", 0)),
        ("날짜 환산 실패", report.get("date_failed", 0)),
        ("검증 상태", report.get("validation_status", "-")),
        ("추천 생성 방식", report.get("recommendation_source_label", "-")),
        ("추천 후보 안내", candidate.get("reason", "") if candidate else ""),
        ("분석 가능", "예" if report.get("analyzable") else "아니오"),
        ("DQN", "제외(미연결)"),
    ]
    return [{"항목": key, "값": "-" if value in (None, "") else str(value)} for key, value in rows]


def column_mapping_rows(report: Mapping[str, Any] | None) -> list[dict[str, str]]:
    mappings = (report or {}).get("column_mappings", []) or []
    return [
        {
            "시트": mapping.get("sheet", "-"),
            "원본 컬럼명": mapping.get("original", "-"),
            "표준 컬럼명": mapping.get("standard", "-"),
            "매핑 방식": "alias 자동 표준화",
        }
        for mapping in mappings
    ]


def validation_message_rows(validation: Any) -> list[dict[str, str]]:
    messages = list(getattr(validation, "messages", None) or [])
    return [
        {"등급": getattr(m, "level", "-"), "시트": getattr(m, "sheet", "-"), "내용": getattr(m, "message", "-")}
        for m in messages
    ]


def candidate_detail_rows(report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Per-candidate detail (V2 생성 후보) for display/export."""
    return list(((report or {}).get("candidate_info") or {}).get("candidates") or [])


def candidate_by_route(report: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {str(row.get("route_id")): row for row in candidate_detail_rows(report)}


def generation_summary_rows(report: Mapping[str, Any] | None) -> list[dict[str, str]]:
    info = (report or {}).get("candidate_info") or {}
    if not info.get("generated"):
        return []
    pairs = [
        ("생성 후보 수", info.get("count", 0)),
        ("DIRECT 후보 수", info.get("direct_count", 0)),
        ("VIA_DC 후보 수", info.get("via_dc_count", 0)),
        ("경로 부족 보류 수", info.get("route_deferred", 0)),
        ("예상 절감액 음수 제외 수", info.get("negative_saving_excluded", 0)),
        ("중복 제외 수", info.get("duplicate_removed", 0)),
        ("추천 수량 0 이하 제외 수", info.get("qty_excluded", 0)),
        ("후보 점수 구성요소", " · ".join(info.get("score_components") or [])),
        ("후보 생성 방식", info.get("method", "-")),
        ("DQN", "제외(점수 미반영)"),
    ]
    return [{"항목": key, "값": str(value)} for key, value in pairs]


def date_normalization_rows(report: Mapping[str, Any] | None) -> list[dict[str, str]]:
    report = report or {}
    pairs = [
        ("날짜 컬럼", ", ".join(report.get("date_columns") or []) or "없음"),
        ("날짜 환산 성공", report.get("date_success", 0)),
        ("날짜 환산 실패", report.get("date_failed", 0)),
        ("기준", "업로드 시점 오늘 날짜 기준 days_to_expiry 환산 (기존 값 우선)"),
    ]
    return [{"항목": key, "값": str(value)} for key, value in pairs]
