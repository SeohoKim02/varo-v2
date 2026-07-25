"""View-model for the 데이터 관리 page.

One place that reads the *real* session state and returns everything the page
needs, so the page never re-guesses status from scattered booleans. It reuses
``services.home_state.build_home_state`` for the shared title/short message/
next action/status grade, and adds the data-management-only detail:

* 현재 적용 데이터(applied) vs 검사 중 데이터(pending) 를 명확히 분리.
* pending 은 이 코드베이스에서 오직 *사용 불가* 업로드만 존재한다
  (``data_application.load_and_apply`` 는 유효/경고 데이터는 즉시 적용하고,
  검증 오류 데이터만 ``pending_*`` 로 보관). 그래서 pending = 사용 불가.
* 검사 결과는 행 수 중심(전체/분석 사용/오류/경고/제외)으로 요약.

Nothing internal (signatures, filesystem paths, session keys, exception names,
canonical field names) is placed in the returned view — only user-facing text
and integer counts. It never raises: any lookup failure degrades to a safe empty
summary instead of crashing the page.
"""
from __future__ import annotations

from typing import Any, Mapping

from services.data_issues import collect_data_issues, display_rows
from services.home_state import NO_CANDIDATES, READY, STALE, build_home_state


def _has_rows(value: Any) -> bool:
    if value is None:
        return False
    empty = getattr(value, "empty", None)
    if empty is not None:
        return not bool(empty)
    try:
        return len(value) > 0
    except TypeError:
        return False


def _stores_applied(data: Any) -> bool:
    """True when a workbook with store rows is currently applied (recs may be 0)."""
    if not isinstance(data, Mapping):
        return False
    return _has_rows(data.get("stores"))


_EMPTY_SUMMARY = {
    "total_rows": 0, "usable_rows": 0, "excluded_rows": 0,
    "error_rows": 0, "warning_rows": 0, "issue_count": 0,
    "top_rows": [], "issues": [],
}


def _issue_summary(data: Any, raw: Any, meta: Any) -> dict[str, Any]:
    """Safe row-level check summary; never raises to the page."""
    try:
        result = collect_data_issues(data, raw, meta)
        summary = result["summary"]
        issues = result["issues"]
        return {
            "total_rows": int(summary.get("total_rows") or 0),
            "usable_rows": int(summary.get("usable_rows") or 0),
            "excluded_rows": int(summary.get("excluded_rows") or 0),
            "error_rows": int(summary.get("error_rows") or 0),
            "warning_rows": int(summary.get("warning_rows") or 0),
            "issue_count": len(issues),
            "top_rows": display_rows(summary.get("top") or []),
            "issues": issues,
        }
    except Exception:  # pragma: no cover - defensive: page must not crash
        return dict(_EMPTY_SUMMARY)


def _source_label(source_type: Any) -> str:
    text = str(source_type or "")
    if "샘플" in text:
        return "샘플 데이터"
    return "업로드 데이터"


def _sheet_label(meta: Any) -> str:
    names = (meta or {}).get("sheet_names") if isinstance(meta, Mapping) else None
    values = [str(value) for value in (names or {}).values() if value]
    if not values:
        return "-"
    if len(values) <= 3:
        return ", ".join(values)
    return ", ".join(values[:3]) + f" 외 {len(values) - 3}개"


def _error_messages(validation: Any) -> list[str]:
    messages = getattr(validation, "messages", None)
    if messages is None and isinstance(validation, Mapping):
        messages = validation.get("messages")
    out: list[str] = []
    for message in messages or []:
        level = getattr(message, "level", None) or (
            message.get("level") if isinstance(message, Mapping) else None
        )
        if level == "오류":
            text = getattr(message, "message", None) or (
                message.get("message") if isinstance(message, Mapping) else None
            )
            if text:
                out.append(str(text))
    return out[:5]


def build_data_management_view(state: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the data-management view-model from real session state."""
    try:
        return _build(state)
    except Exception:  # pragma: no cover - defensive: page must not crash
        return {
            "home": build_home_state(state),
            "has_current": False,
            "current": None,
            "has_pending": False,
            "pending": None,
            "load_error": None,
            "stale": False,
            "show_next_action": False,
        }


def _build(state: Mapping[str, Any]) -> dict[str, Any]:
    home = build_home_state(state)
    has_current = _stores_applied(state.get("varo_data"))

    load_error = state.get("pending_load_error")
    pending_data = state.get("pending_varo_data")
    pending_validation = state.get("pending_varo_validation")
    has_pending = bool(pending_data) or bool(load_error) or bool(
        getattr(pending_validation, "has_errors", False)
    )

    current = None
    if has_current:
        summary = _issue_summary(
            state.get("varo_data"), state.get("raw_data"), state.get("source_metadata")
        )
        recommendations = state.get("varo_recommendations") or []
        rec_count = len(recommendations)
        if rec_count:
            rec_status = f"추천 {rec_count}건"
        elif home.get("state_code") == NO_CANDIDATES:
            rec_status = "추천할 이동 없음"
        else:
            rec_status = "추천 실행 필요"
        current = {
            "source_label": _source_label(state.get("data_source_type")),
            "filename": state.get("uploaded_filename") or "-",
            "sheet_label": _sheet_label(state.get("source_metadata")),
            "data_status": home.get("data_status") or "사용 가능",
            "total_rows": summary["total_rows"],
            "usable_rows": summary["usable_rows"],
            "excluded_rows": summary["excluded_rows"],
            "warning_rows": summary["warning_rows"],
            "recommendation_count": rec_count,
            "recommendation_status": rec_status,
        }

    pending = None
    if pending_data:
        summary = _issue_summary(
            pending_data,
            state.get("pending_raw_data"),
            state.get("pending_source_metadata"),
        )
        status = str(state.get("pending_status") or "사용 불가")
        apply_allowed = bool(state.get("pending_apply_allowed"))
        same_as_current = status == "현재 데이터와 동일"
        if status == "사용 가능":
            apply_label = "이 데이터 사용"
        elif status == "확인 필요":
            apply_label = "문제 행을 제외하고 사용"
        else:
            apply_label = None
        pending = {
            "filename": state.get("pending_uploaded_filename") or "-",
            "sheet_label": _sheet_label(state.get("pending_source_metadata")),
            "status": status,
            "apply_allowed": apply_allowed,
            "apply_label": apply_label,
            "same_as_current": same_as_current,
            "total_rows": summary["total_rows"],
            "usable_rows": int(state.get("pending_usable_rows") or summary["usable_rows"]),
            "excluded_rows": int(state.get("pending_excluded_rows") or summary["excluded_rows"]),
            "error_rows": summary["error_rows"],
            "warning_rows": summary["warning_rows"],
            "issue_count": summary["issue_count"],
            "top_rows": summary["top_rows"],
            "issues": summary["issues"],
            "error_messages": _error_messages(pending_validation),
        }

    return {
        "home": home,
        "has_current": has_current,
        "current": current,
        "has_pending": has_pending,
        "pending": pending,
        "load_error": load_error,
        "stale": home.get("state_code") == STALE,
        # A forward action ("추천 실행") only makes sense when there is applied data
        # and no unusable upload demanding a fix first.
        "show_next_action": has_current and not has_pending
        and home.get("state_code") in (READY, NO_CANDIDATES),
    }
