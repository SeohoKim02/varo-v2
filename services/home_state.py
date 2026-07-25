"""Single source of truth for the home-screen workspace status.

The home screen must answer, in three seconds: which data is loaded, is it
usable, was the recommendation produced, is there a result, and what should the
user do next. Rather than let each UI branch guess from scattered flags, this
module reads the *real* session state once and returns one resolved status with
a title, a short message, and exactly one primary next action.

Design facts it relies on (verified in code, not invented):
* Uploads are a two-phase intake: ``data_application.prepare_pending_data`` only
  *inspects* a file into ``pending_*`` (any status), and ``commit_pending_data``
  applies + runs the analysis on an explicit button click. So a ``pending_*`` set
  is an *intake* sub-state, not the workspace: it must not hide an already-applied
  result. Only when there is **no applied data** does an unusable pending define
  the whole workspace as 사용 불가.
* Applying (``commit_pending_data`` or the sample path ``load_and_apply``) runs the
  analysis in the same step, so "적용됐지만 실행 전" does not occur for applied data.
* ``varo_recommendations`` is the final feasible set and the single ranking
  source — the home never re-sorts or re-computes candidates.

Nothing here raises to the UI: every lookup is defensive and, if inputs are
missing, it degrades to a safe "데이터 없음 / 확인 필요" state instead of a fake
healthy one. Internal codes, signatures, and ids are never part of the message.
"""
from __future__ import annotations

from typing import Any, Mapping

from services.analysis_pipeline import top_recommendations

# Internal state codes (never shown to the user).
NO_DATA = "no_data"
UNUSABLE = "unusable"
STALE = "stale"
FAILED = "failed"
NO_CANDIDATES = "no_candidates"
READY = "ready"

# Page names must match components.navigation.MENU_ITEMS / router._PAGE_RENDERERS.
PAGE_DATA = "데이터 관리"
PAGE_RECOMMENDATIONS = "추천 실행"
PAGE_ROUTE_DETAIL = "경로 상세"
PAGE_VALIDATION = "분석 및 검증"


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


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


def _data_applied(data: Any) -> bool:
    """True when a workbook with store rows has been applied (recs may be 0)."""
    if not isinstance(data, Mapping):
        return False
    return _has_rows(data.get("stores"))


def _validation_status(validation: Any) -> str | None:
    status = getattr(validation, "status", None)
    if status is None and isinstance(validation, Mapping):
        status = validation.get("status")
    return str(status) if status else None


def _first_error_message(validation: Any) -> str | None:
    messages = getattr(validation, "messages", None)
    if messages is None and isinstance(validation, Mapping):
        messages = validation.get("messages")
    for message in messages or []:
        level = getattr(message, "level", None) or (message.get("level") if isinstance(message, Mapping) else None)
        if level == "오류":
            text = getattr(message, "message", None) or (message.get("message") if isinstance(message, Mapping) else None)
            if text:
                return str(text)
    return None


def _error_count(validation: Any) -> int:
    messages = getattr(validation, "messages", None)
    if messages is None and isinstance(validation, Mapping):
        messages = validation.get("messages")
    count = 0
    for message in messages or []:
        level = getattr(message, "level", None) or (message.get("level") if isinstance(message, Mapping) else None)
        if level == "오류":
            count += 1
    return count


def _pipeline(state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = state.get("analysis_result") or state.get("varo_pipeline_result")
    return value if isinstance(value, Mapping) else {}


def _data_source_label(state: Mapping[str, Any]) -> str:
    if str(state.get("recommendation_source")) == "generated":
        return "자동 생성 후보"
    source = str(state.get("data_source_type") or "")
    if "샘플" in source:
        return "기본 샘플 데이터"
    if not source:
        return "적용 데이터"
    return "업로드 데이터"


def _result_signature(pipeline: Mapping[str, Any]) -> str | None:
    ledger = pipeline.get("candidate_ledger") or []
    for record in ledger:
        signature = record.get("data_signature")
        if signature:
            return str(signature)
    return None


def _no_candidate_cause(pipeline: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    """One or two plain-language reasons why there is no recommendable move."""
    summary = pipeline.get("ledger_summary") or {}
    generated = int(summary.get("generated") or 0)
    if generated == 0:
        info = state.get("upload_report") or {}
        candidate_info = (info.get("candidate_info") if isinstance(info, Mapping) else None) or {}
        reason = candidate_info.get("reason") or (pipeline.get("candidate_info") or {}).get("reason")
        if reason:
            return str(reason)
        return "현재 조건에서 이동할 재고 후보를 만들지 못했습니다."
    reasons = summary.get("top_exclusion_reasons") or []
    parts = [
        f"{item.get('count')}건 — {str(item.get('reason')).rstrip('.')}"
        for item in reasons[:2] if item.get("reason")
    ]
    if parts:
        return f"후보 {generated}건 중 추천 가능한 이동이 없습니다. " + " · ".join(parts) + "."
    return f"생성된 후보 {generated}건이 모두 실행 조건을 통과하지 못했습니다."


def _top_recommendation(recommendations: list[dict]) -> dict | None:
    top = top_recommendations(list(recommendations or []), limit=1)
    return dict(top[0]) if top else None


def build_home_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the one home status from real session state (never raises)."""
    try:
        return _build(state)
    except Exception:  # pragma: no cover - defensive: home must never crash
        return {
            "state_code": NO_DATA,
            "title": "결과를 확인할 수 없습니다",
            "short_message": "데이터를 다시 불러와주세요.",
            "next_action_label": "데이터 불러오기",
            "next_page": PAGE_DATA,
            "data_source": "없음",
            "data_status": "확인 필요",
            "analysis_status": "-",
            "recommendation_status": "-",
            "recommendation_count": 0,
            "blocked_count": 0,
            "warning_count": 0,
            "show_result_kpis": False,
            "top_recommendation": None,
            "confidence_status": None,
            "no_candidate_cause": None,
            "selected_candidate_valid": False,
        }


def _build(state: Mapping[str, Any]) -> dict[str, Any]:
    recommendations = list(state.get("varo_recommendations") or [])
    data = state.get("varo_data")
    validation = state.get("varo_validation")
    pipeline = _pipeline(state)
    ledger_summary = pipeline.get("ledger_summary") or {}
    data_signature = state.get("data_signature")

    data_applied = _data_applied(data)
    pending_error = state.get("pending_load_error")
    pending_validation = state.get("pending_varo_validation")
    pending_data = state.get("pending_varo_data")
    pending_unusable = bool(pending_error) or bool(getattr(pending_validation, "has_errors", False))
    pending_present = bool(pending_data) or bool(pending_error)

    base = {
        "data_source": _data_source_label(state),
        "recommendation_count": 0,
        "blocked_count": int(ledger_summary.get("excluded_total") or 0),
        "warning_count": 0,
        "show_result_kpis": False,
        "top_recommendation": None,
        "confidence_status": None,
        "no_candidate_cause": None,
        "selected_candidate_valid": False,
        "analysis_status": "-",
        "recommendation_status": "-",
        # A new upload is being inspected while good data is already applied — a
        # side notice, not a workspace state change.
        "pending_notice": pending_present and data_applied,
    }

    # 1) Unusable upload (pending) defines the whole workspace ONLY when there is no
    # good applied data to fall back on. When applied data exists, a bad new upload
    # is an intake sub-state shown in 데이터 관리 — it must not hide the current result.
    if pending_unusable and not data_applied:
        message = pending_error or _first_error_message(pending_validation) or "업로드한 데이터에 오류가 있어 사용할 수 없습니다."
        errors = _error_count(pending_validation)
        detail = f" (오류 {errors}건)" if errors else ""
        return {
            **base,
            "state_code": UNUSABLE,
            "title": "데이터를 수정해야 합니다",
            "short_message": f"{message}{detail}",
            "next_action_label": "문제 확인",
            "next_page": PAGE_DATA,
            "data_source": "업로드 데이터",
            "data_status": "사용 불가",
            "pending_notice": False,
        }

    # 2) No applied data. If a usable file has already been inspected, point the user
    # at applying it; otherwise prompt to load data.
    if not data_applied:
        if pending_present:
            return {
                **base,
                "state_code": NO_DATA,
                "title": "검사한 데이터를 적용하세요",
                "short_message": "데이터 관리에서 검사 결과를 확인하고 ‘이 데이터 사용’을 누르면 적용됩니다.",
                "next_action_label": "데이터 적용",
                "next_page": PAGE_DATA,
                "data_source": "검사 중 데이터",
                "data_status": "검사 완료",
            }
        return {
            **base,
            "state_code": NO_DATA,
            "title": "데이터를 준비하세요",
            "short_message": "재고 데이터를 불러오면 이동 추천을 시작할 수 있습니다.",
            "next_action_label": "데이터 불러오기",
            "next_page": PAGE_DATA,
            "data_source": "없음",
            "data_status": "없음",
        }

    data_status_label = {
        "통과": "사용 가능", "주의": "확인 필요", "오류": "사용 불가",
    }.get(_validation_status(validation) or "", "사용 가능")
    warning_count = 1 if _validation_status(validation) == "주의" else 0

    # 3) Stale result: applied data signature differs from the analysed one.
    result_signature = _result_signature(pipeline)
    if data_signature and result_signature and str(data_signature) != str(result_signature):
        return {
            **base,
            "state_code": STALE,
            "title": "데이터가 변경되었습니다",
            "short_message": "새 데이터로 결과를 다시 확인해야 합니다.",
            "next_action_label": "데이터 다시 적용",
            "next_page": PAGE_DATA,
            "data_status": data_status_label,
            "warning_count": warning_count,
        }

    # 4) Recommendations present → ready result (single ranking source).
    if recommendations:
        top = _top_recommendation(recommendations)
        confidence = (pipeline.get("confidence_status") or {}).get("status")
        selected = str(state.get("selected_route_id") or "")
        valid_ids = {str(rec.get("route_id")) for rec in recommendations}
        return {
            **base,
            "state_code": READY,
            "title": "추천 결과를 확인하세요",
            "short_message": "가장 우선순위가 높은 이동부터 검토할 수 있습니다.",
            "next_action_label": "추천 상세 보기",
            "next_page": PAGE_ROUTE_DETAIL,
            "data_status": data_status_label,
            "analysis_status": "완료",
            "recommendation_status": "추천 있음",
            "recommendation_count": len(recommendations),
            "warning_count": warning_count,
            "show_result_kpis": True,
            "top_recommendation": top,
            "confidence_status": confidence,
            "selected_candidate_valid": selected in valid_ids if selected else True,
        }

    # 5) Applied but no final recommendation: distinguish failure from 0-candidate.
    diagnostics = pipeline.get("diagnostics") or {}
    technical_errors = diagnostics.get("algorithm_errors") or []
    status = str(pipeline.get("status") or "")
    generated = int(ledger_summary.get("generated") or 0)
    if generated == 0 and (technical_errors or status in {"adapter_error", "validation_error"}):
        return {
            **base,
            "state_code": FAILED,
            "title": "추천을 완료하지 못했습니다",
            "short_message": "데이터를 확인한 뒤 다시 적용해주세요.",
            "next_action_label": "데이터 확인",
            "next_page": PAGE_DATA,
            "data_status": data_status_label,
            "analysis_status": "실패",
            "recommendation_status": "추천 없음",
            "warning_count": warning_count,
        }

    return {
        **base,
        "state_code": NO_CANDIDATES,
        "title": "추천할 이동이 없습니다",
        "short_message": _no_candidate_cause(pipeline, state),
        "next_action_label": "제외 이유 확인",
        "next_page": PAGE_RECOMMENDATIONS,
        "data_status": data_status_label,
        "analysis_status": "완료",
        "recommendation_status": "추천 없음",
        "no_candidate_cause": _no_candidate_cause(pipeline, state),
        "warning_count": warning_count,
    }
