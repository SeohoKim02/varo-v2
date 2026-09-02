"""Shared Excel load-and-apply workflow for Varo V2 UI surfaces."""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from services.analysis_pipeline import build_v2_state, ensure_recommendations, run_analysis_pipeline
from services.analysis_progress import ProgressCallback
from services.app_state import (
    apply_state_payload, build_applied_state_payload, default_selected_route_id,
)
from services.execution_plan import planned_recommendations
from services.data_loader import DataLoadError
from services.partial_data import build_usable_data, usable_data_signature
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
    "pending_usable_data", "pending_usable_signature", "pending_source_signature",
    "pending_data_issues", "pending_excluded_row_refs", "pending_quality_summary",
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
    current_source = state.get("source_signature") or state.get("data_signature")
    if current_source == signature and _has_stores(state.get("varo_data")):
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
        partial = build_usable_data(data, raw_data, source_metadata)
        validation = partial["validation"]
        recommendation_source = partial["recommendation_source"]
        candidate_info = partial["candidate_info"]
    except DataLoadError as exc:
        clear_pending(state)
        state["pending_load_error"] = str(exc)
        return "오류"
    except Exception:  # pragma: no cover - defensive: never crash the upload
        logger.exception("prepare_pending_data failed for %s", filename)
        clear_pending(state)
        state["pending_load_error"] = "파일을 처리할 수 없습니다. 파일 형식과 필수 컬럼을 확인한 뒤 다시 업로드해주세요."
        return "오류"

    quality = dict(partial["quality_summary"])
    usable_rows = int(quality.get("usable_rows") or 0)
    excluded_rows = int(quality.get("excluded_rows") or 0)
    apply_allowed = bool(partial["apply_allowed"])
    upload_report = build_upload_report(
        load_report, validation, recommendation_source, candidate_info, filename
    )

    clear_pending(state)
    state.pop("pending_load_error", None)
    state["pending_varo_data"] = data
    state["pending_usable_data"] = partial["usable_data"]
    state["pending_raw_data"] = raw_data
    state["pending_source_metadata"] = source_metadata
    state["pending_varo_validation"] = validation
    state["pending_varo_recommendations"] = []
    state["pending_uploaded_filename"] = filename
    state["pending_data_source_type"] = source_type
    state["pending_upload_report"] = upload_report
    state["pending_recommendation_source"] = recommendation_source
    state["pending_data_signature"] = signature
    state["pending_source_signature"] = signature
    state["pending_usable_signature"] = partial["usable_signature"]
    state["pending_data_issues"] = partial["issues"]
    state["pending_excluded_row_refs"] = partial["excluded_row_refs"]
    state["pending_quality_summary"] = quality
    state["pending_apply_allowed"] = apply_allowed
    state["pending_usable_rows"] = usable_rows
    state["pending_excluded_rows"] = excluded_rows
    state["pending_created_at"] = datetime.now().isoformat(timespec="seconds")
    status = _pending_status(state, signature, validation, apply_allowed)
    if status == PENDING_USABLE and (excluded_rows or quality.get("warning_rows")):
        status = PENDING_CHECK
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
    data = state.get("pending_usable_data")
    validation = state.get("pending_varo_validation")
    source_sig = state.get("pending_source_signature") or state.get("pending_data_signature")
    signature = state.get("pending_usable_signature")

    # Final re-check (spec: 적용 전 최종 재검증). Missing/mutated pending → 만료.
    if not data or validation is None or not signature or not source_sig:
        state["data_apply_error"] = "검사 결과가 만료됐습니다. 파일을 다시 확인하세요."
        return False
    if getattr(validation, "has_errors", False) or not state.get("pending_apply_allowed"):
        state["data_apply_error"] = "이 데이터는 지금 적용할 수 없습니다. 문제를 수정한 뒤 다시 업로드하세요."
        return False
    if int(state.get("pending_usable_rows") or 0) < 1 or not _has_stores(data):
        state["data_apply_error"] = "검사 결과가 만료됐습니다. 파일을 다시 확인하세요."
        return False

    try:
        if usable_data_signature(data) != signature:
            state["data_apply_error"] = "검사 결과가 변경됐습니다. 파일을 다시 확인하세요."
            return False
    except Exception:
        state["data_apply_error"] = "검사 결과가 만료됐습니다. 파일을 다시 확인하세요."
        return False
    quality = dict(state.get("pending_quality_summary") or {})
    excluded_refs = list(state.get("pending_excluded_row_refs") or [])
    if int(quality.get("excluded_rows") or 0) != len({
        (str(item.get("source_sheet") or ""), int(item.get("source_row_number") or 0))
        for item in excluded_refs
    }):
        state["data_apply_error"] = "검사 결과가 변경됐습니다. 파일을 다시 확인하세요."
        return False
    final_validation = validate_workbook_data(data)
    if final_validation.has_errors:
        state["data_apply_error"] = "제외 후에도 필수 데이터 문제가 남아 있어 적용할 수 없습니다."
        return False

    # Same original content as the current data → no unnecessary reset.
    if (state.get("source_signature") or state.get("data_signature")) == source_sig and _has_stores(state.get("varo_data")):
        clear_pending(state)
        state["data_apply_message"] = "현재 사용 중인 데이터와 같습니다."
        state.pop("data_apply_error", None)
        return True

    try:
        raw_data = dict(state.get("pending_raw_data") or {})
        source_metadata = dict(state.get("pending_source_metadata") or {})
        source_type = state.get("pending_data_source_type") or "업로드된 추천 결과"
        recommendation_source = state.get("pending_recommendation_source") or "uploaded"
        effective_source = "V2 생성 후보" if recommendation_source == "generated" else source_type
        payload = build_applied_state_payload(
            data=data,
            validation=final_validation,
            recommendations=[],
            filename=state.get("pending_uploaded_filename") or "-",
            source_type=effective_source,
            pipeline_result={},
            data_signature=signature,
            raw_data=raw_data,
            source_metadata=source_metadata,
            source_signature=source_sig,
            data_quality_summary=quality,
            data_issues=state.get("pending_data_issues") or [],
            excluded_row_refs=excluded_refs,
            analysis_run_required=True,
        )
        payload["upload_report"] = dict(state.get("pending_upload_report") or {})
        payload["recommendation_source"] = recommendation_source
        apply_state_payload(state, payload)
        clear_pending(state)
        if int(quality.get("excluded_rows") or 0):
            state["data_apply_message"] = f"문제 행 {quality['excluded_rows']}개를 제외한 데이터가 적용되었습니다. 추천 실행을 시작하세요."
        else:
            state["data_apply_message"] = "데이터가 적용되었습니다. 추천 실행을 시작하세요."
        state.pop("data_apply_error", None)
        return True
    except Exception:  # pragma: no cover - defensive: never crash + preserve state
        logger.exception("commit_pending_data failed for %s", state.get("pending_uploaded_filename"))
        state["data_apply_error"] = (
            "새 데이터를 적용하지 못했습니다. 현재 사용 중인 데이터는 그대로 유지했습니다. 검사 결과를 다시 확인하세요."
        )
        return False


def run_applied_analysis(
    state: MutableMapping[str, Any], progress_callback: ProgressCallback | None = None,
) -> bool:
    """Run algorithms only after the user explicitly requests recommendation execution.

    ``progress_callback`` is optional and observational only: it receives one event
    per real pipeline stage so a page can show progress. With no callback the
    behaviour and the result are identical to before.

    ``analysis_running`` marks the run for the UI and is always cleared, whether the
    run succeeds, fails, or raises, so the button can never stay blocked.
    """
    data = state.get("varo_data")
    if not _has_stores(data):
        state["analysis_run_error"] = "먼저 데이터 관리에서 사용할 데이터를 적용하세요."
        return False
    state["analysis_running"] = True
    state.pop("analysis_completed_notice", None)
    started = time.perf_counter()
    try:
        result = run_analysis_pipeline(
            data,
            raw_data=state.get("raw_data") or {},
            source_metadata=state.get("source_metadata") or {},
            data_signature=state.get("data_signature"),
            progress_callback=progress_callback,
        )
        if result.status in {"validation_error", "adapter_error"}:
            state["analysis_run_error"] = "추천을 계산할 수 없습니다. 적용 데이터 상태를 확인하세요."
            return False
        pipeline = result.to_dict()
        recommendations = [dict(item) for item in result.recommendations]
        state["varo_recommendations"] = recommendations
        state["varo_pipeline_result"] = pipeline
        state["analysis_result"] = pipeline
        state["pipeline_summary"] = pipeline.get("summary", {})
        state["connected_algorithms"] = pipeline.get("connected_algorithms", [])
        state["deferred_algorithms"] = pipeline.get("deferred_algorithms", [])
        state["dqn_excluded"] = pipeline.get("excluded_dqn_artifacts", {})
        actions = planned_recommendations(pipeline) if "execution_plan" in pipeline else recommendations
        state["selected_route_id"] = default_selected_route_id(actions)
        state["analysis_run_required"] = False
        state.pop("analysis_run_error", None)
        state["analysis_elapsed_seconds"] = round(time.perf_counter() - started, 2)
        state["analysis_completed_notice"] = True
        return True
    except Exception:  # pragma: no cover - defensive and user-safe
        logger.exception("run_applied_analysis failed")
        state["analysis_run_error"] = "추천 실행 중 문제가 발생했습니다. 적용 데이터는 그대로 유지했습니다."
        return False
    finally:
        state.pop("analysis_running", None)
