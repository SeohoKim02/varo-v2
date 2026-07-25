"""Shared Excel load-and-apply workflow for Varo V2 UI surfaces."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from services.analysis_pipeline import build_v2_state, ensure_recommendations, run_analysis_pipeline
from services.app_state import apply_state_payload, build_applied_state_payload
from services.data_issues import collect_data_issues
from services.data_loader import DataLoadError
from services.data_validator import ValidationReport, validate_workbook_data
from services.file_reader import file_extension, read_uploaded_data
from services.upload_quality import build_upload_report

logger = logging.getLogger(__name__)

PENDING_KEYS = (
    "pending_varo_data", "pending_varo_validation", "pending_varo_recommendations",
    "pending_uploaded_filename", "pending_data_source_type", "pending_upload_report",
    "pending_raw_data", "pending_source_metadata",
    # Two-phase intake (inspect → apply) fields.
    "pending_data_signature", "pending_recommendation_source", "pending_apply_allowed",
    "pending_usable_rows", "pending_excluded_rows", "pending_status", "pending_created_at",
)

# Pending intake status codes (also used as user-facing labels).
PENDING_USABLE = "사용 가능"
PENDING_CHECK = "확인 필요"
PENDING_UNUSABLE = "사용 불가"
PENDING_SAME = "현재 데이터와 동일"


def _build_source_metadata(
    load_report: dict[str, Any], filename: str, source_type: str,
) -> dict[str, Any]:
    """Original-file provenance used for row/column/value 오류 추적 (not analysis)."""
    if "샘플" in (source_type or ""):
        file_type = "sample"
    else:
        file_type = "csv" if file_extension(filename) == ".csv" else "excel"
    return {
        "filename": filename or "-",
        "source_type": file_type,
        "sheet_names": dict(load_report.get("raw_sheet_names") or {}),
    }


def clear_pending(state: MutableMapping[str, Any]) -> None:
    for key in PENDING_KEYS:
        state.pop(key, None)


def application_message(source_type: str, pipeline_result: dict[str, Any] | None = None) -> str:
    basis = (pipeline_result or {}).get("result_basis", "업로드된 사전 계산 추천 결과 기준")
    if "샘플" in (source_type or ""):
        return f"샘플 데이터가 앱에 적용되었습니다. 현재 결과는 {basis}입니다."
    return f"업로드 데이터가 앱에 적용되었습니다. 현재 결과는 {basis}입니다."


def uploaded_signature(uploaded_file: Any) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def source_signature(source: Any, filename: str = "") -> str:
    """Return a content-based signature without keeping file handles in state."""
    try:
        if hasattr(source, "getvalue"):
            return hashlib.sha256(source.getvalue()).hexdigest()
        if isinstance(source, (str, Path)):
            return hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if hasattr(source, "read"):
            position = source.tell() if hasattr(source, "tell") else None
            if hasattr(source, "seek"):
                source.seek(0)
            content = source.read()
            if position is not None and hasattr(source, "seek"):
                source.seek(position)
            if isinstance(content, str):
                content = content.encode("utf-8")
            return hashlib.sha256(content).hexdigest()
    except Exception:
        pass
    return hashlib.sha256(str(filename).encode("utf-8")).hexdigest()


def _store_failed_candidate(
    state: MutableMapping[str, Any], data: dict[str, pd.DataFrame], validation: ValidationReport,
    filename: str, source_type: str, upload_report: dict[str, Any] | None = None,
    raw_data: dict[str, pd.DataFrame] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> None:
    state["pending_varo_data"] = data
    state["pending_varo_validation"] = validation
    state["pending_varo_recommendations"] = []
    state["pending_uploaded_filename"] = filename
    state["pending_data_source_type"] = source_type
    state["pending_upload_report"] = upload_report or {}
    state["pending_raw_data"] = raw_data or {}
    state["pending_source_metadata"] = source_metadata or {}
    state["data_apply_message"] = None


def load_and_apply(
    state: MutableMapping[str, Any], source: Any, filename: str, source_type: str,
) -> bool:
    """Load, validate, run approved algorithms, and apply canonical state.

    Hardened so a malformed upload never crashes the app: any unexpected error is
    converted into a user-facing load error and previous state is preserved.
    """
    signature = source_signature(source, filename)
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        data, load_report = read_uploaded_data(source, filename, return_report=True)
        raw_data = dict(load_report.get("raw_sheets") or {})
        source_metadata = _build_source_metadata(load_report, filename, source_type)
        pipeline_state = build_v2_state(
            data, raw_data=raw_data, source_metadata=source_metadata, data_signature=signature,
        )
    except DataLoadError as exc:
        clear_pending(state)
        state["pending_load_error"] = str(exc)
        state["data_apply_message"] = None
        return False
    except Exception:  # pragma: no cover - defensive: never crash the upload
        # Keep the technical cause in the log; show the user a plain, non-technical
        # message only (no exception class / traceback / internal path on screen).
        logger.exception("load_and_apply failed for %s", filename)
        clear_pending(state)
        state["pending_load_error"] = "파일을 처리할 수 없습니다. 파일 형식과 필수 컬럼을 확인한 뒤 다시 업로드해주세요."
        state["data_apply_message"] = None
        return False

    state.pop("pending_load_error", None)
    validation = pipeline_state["validation"]
    recommendations = pipeline_state["recommendations"]
    pipeline_result = pipeline_state.get("pipeline_result", {})
    recommendation_source = pipeline_state.get("recommendation_source", "uploaded")
    candidate_info = pipeline_state.get("candidate_info", {})
    upload_report = build_upload_report(load_report, validation, recommendation_source, candidate_info, filename)

    if validation.has_errors:
        _store_failed_candidate(
            state, data, validation, filename, source_type, upload_report,
            raw_data=raw_data, source_metadata=source_metadata,
        )
        return False

    effective_source = "V2 생성 후보" if recommendation_source == "generated" else source_type
    payload = build_applied_state_payload(
        data=data,
        validation=validation,
        recommendations=recommendations,
        filename=filename,
        source_type=effective_source,
        pipeline_result=pipeline_result,
        data_signature=signature,
        raw_data=raw_data,
        source_metadata=source_metadata,
    )
    payload["upload_report"] = upload_report
    payload["recommendation_source"] = recommendation_source
    apply_state_payload(state, payload)
    clear_pending(state)
    state["data_apply_message"] = application_message(effective_source, pipeline_result)
    return True


# --------------------------------------------------------------------------- #
# Two-phase intake: prepare_pending_data (inspect) → commit_pending_data (apply)
#
# The UI upload path uses these so a file is *inspected* (read/validate/normalize/
# count) into pending_* state without touching the applied workspace. Only an
# explicit user action (commit_pending_data) replaces the current data and resets
# derived results. load_and_apply above is kept for the sample/quick-start path
# and programmatic callers that intentionally apply in one step.
# --------------------------------------------------------------------------- #
def _has_stores(data: Any) -> bool:
    if not isinstance(data, Mapping):
        return False
    stores = data.get("stores")
    empty = getattr(stores, "empty", None)
    if empty is not None:
        return not bool(empty)
    try:
        return len(stores) > 0
    except TypeError:
        return False


def _pending_status(
    state: Mapping[str, Any], signature: str, validation: ValidationReport, apply_allowed: bool,
) -> str:
    if state.get("data_signature") == signature and _has_stores(state.get("varo_data")):
        return PENDING_SAME
    if validation.has_errors or not apply_allowed:
        return PENDING_UNUSABLE
    if validation.status == "주의":
        return PENDING_CHECK
    return PENDING_USABLE


def prepare_pending_data(
    state: MutableMapping[str, Any], source: Any, filename: str, source_type: str,
) -> str:
    """Inspect an upload into pending_* state. Never touches the applied workspace.

    Reads the file, preserves the original snapshot, normalizes, validates and
    counts usable/excluded rows, then stores everything under pending_*. Returns a
    status code (사용 가능/확인 필요/사용 불가/현재 데이터와 동일/오류). The current
    varo_data, recommendations, signature and results are left untouched so the
    home/추천 실행 pages keep working on the already-applied data.
    """
    signature = source_signature(source, filename)
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        data, load_report = read_uploaded_data(source, filename, return_report=True)
        raw_data = dict(load_report.get("raw_sheets") or {})
        source_metadata = _build_source_metadata(load_report, filename, source_type)
        data, recommendation_source, candidate_info = ensure_recommendations(data)
        validation = validate_workbook_data(data)
    except DataLoadError as exc:
        clear_pending(state)
        state["pending_load_error"] = str(exc)
        return "오류"
    except Exception:  # pragma: no cover - defensive: never crash the upload
        logger.exception("prepare_pending_data failed for %s", filename)
        clear_pending(state)
        state["pending_load_error"] = "파일을 처리할 수 없습니다. 파일 형식과 필수 컬럼을 확인한 뒤 다시 업로드해주세요."
        return "오류"

    try:
        issue_summary = collect_data_issues(data, raw_data, source_metadata)["summary"]
    except Exception:  # pragma: no cover - defensive
        issue_summary = {}
    usable_rows = int(issue_summary.get("usable_rows") or 0)
    excluded_rows = int(issue_summary.get("excluded_rows") or 0)
    mostly_excluded = bool(issue_summary.get("mostly_excluded"))
    apply_allowed = (not validation.has_errors) and usable_rows >= 1 and not mostly_excluded
    upload_report = build_upload_report(
        load_report, validation, recommendation_source, candidate_info, filename
    )

    clear_pending(state)
    state.pop("pending_load_error", None)
    state["pending_varo_data"] = data
    state["pending_raw_data"] = raw_data
    state["pending_source_metadata"] = source_metadata
    state["pending_varo_validation"] = validation
    state["pending_varo_recommendations"] = []
    state["pending_uploaded_filename"] = filename
    state["pending_data_source_type"] = source_type
    state["pending_upload_report"] = upload_report
    state["pending_recommendation_source"] = recommendation_source
    state["pending_data_signature"] = signature
    state["pending_apply_allowed"] = apply_allowed
    state["pending_usable_rows"] = usable_rows
    state["pending_excluded_rows"] = excluded_rows
    state["pending_created_at"] = datetime.now().isoformat(timespec="seconds")
    status = _pending_status(state, signature, validation, apply_allowed)
    state["pending_status"] = status
    return status


def cancel_pending_data(state: MutableMapping[str, Any]) -> None:
    """Discard the inspected pending data only. Applied data/results are kept."""
    clear_pending(state)
    state.pop("pending_load_error", None)


def commit_pending_data(state: MutableMapping[str, Any]) -> bool:
    """Apply the already-inspected pending data (no file re-read). Atomic.

    Re-validates the stored pending result before applying. On any failure the
    current applied data, recommendations and pending intake are all preserved and
    a plain user-facing message is set in ``data_apply_error`` (never a traceback).
    Only a different signature triggers a real apply + reset of prior results.
    """
    data = state.get("pending_varo_data")
    validation = state.get("pending_varo_validation")
    signature = state.get("pending_data_signature")

    # Final re-check (spec: 적용 전 최종 재검증). Missing/mutated pending → 만료.
    if not data or validation is None or not signature:
        state["data_apply_error"] = "검사 결과가 만료됐습니다. 파일을 다시 확인하세요."
        return False
    if getattr(validation, "has_errors", False) or not state.get("pending_apply_allowed"):
        state["data_apply_error"] = "이 데이터는 지금 적용할 수 없습니다. 문제를 수정한 뒤 다시 업로드하세요."
        return False
    if int(state.get("pending_usable_rows") or 0) < 1 or not _has_stores(data):
        state["data_apply_error"] = "검사 결과가 만료됐습니다. 파일을 다시 확인하세요."
        return False

    # Same content as the current data → no unnecessary reset of existing results.
    if state.get("data_signature") == signature and _has_stores(state.get("varo_data")):
        clear_pending(state)
        state["data_apply_message"] = "현재 사용 중인 데이터와 같습니다."
        state.pop("data_apply_error", None)
        return True

    try:
        raw_data = dict(state.get("pending_raw_data") or {})
        source_metadata = dict(state.get("pending_source_metadata") or {})
        source_type = state.get("pending_data_source_type") or "업로드된 추천 결과"
        recommendation_source = state.get("pending_recommendation_source") or "uploaded"
        result = run_analysis_pipeline(
            data, raw_data=raw_data, source_metadata=source_metadata, data_signature=signature,
        )
        pipeline_result = result.to_dict()
        if result.status == "validation_error":
            state["data_apply_error"] = "검사 결과가 만료됐습니다. 파일을 다시 확인하세요."
            return False
        effective_source = "V2 생성 후보" if recommendation_source == "generated" else source_type
        payload = build_applied_state_payload(
            data=data,
            validation=validation,
            recommendations=result.recommendations,
            filename=state.get("pending_uploaded_filename") or "-",
            source_type=effective_source,
            pipeline_result=pipeline_result,
            data_signature=signature,
            raw_data=raw_data,
            source_metadata=source_metadata,
        )
        payload["upload_report"] = dict(state.get("pending_upload_report") or {})
        payload["recommendation_source"] = recommendation_source
        apply_state_payload(state, payload)
        clear_pending(state)
        state["data_apply_message"] = application_message(effective_source, pipeline_result)
        state.pop("data_apply_error", None)
        return True
    except Exception:  # pragma: no cover - defensive: never crash + preserve state
        logger.exception("commit_pending_data failed for %s", state.get("pending_uploaded_filename"))
        state["data_apply_error"] = (
            "새 데이터를 적용하지 못했습니다. 현재 사용 중인 데이터는 그대로 유지했습니다. 검사 결과를 다시 확인하세요."
        )
        return False
