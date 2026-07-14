"""Shared Excel load-and-apply workflow for Varo V2 UI surfaces."""
from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from io import BytesIO
from pathlib import Path
import re
from typing import Any

import pandas as pd

from services.analysis_pipeline import build_v2_state
from services.app_state import apply_state_payload, build_applied_state_payload
from services.data_loader import DataLoadError, load_excel_data
from services.data_validator import ValidationReport
from services.upload_quality import build_upload_report

PENDING_KEYS = (
    "pending_varo_data", "pending_varo_validation", "pending_varo_recommendations",
    "pending_uploaded_filename", "pending_data_source_type", "pending_upload_report",
    "pending_validation_error",
)


def clear_pending(state: MutableMapping[str, Any]) -> None:
    for key in PENDING_KEYS:
        state.pop(key, None)


def application_message(has_warnings: bool = False) -> str:
    """Return the intentionally short upload status used throughout V2."""
    return "데이터 적용 완료 · 일부 항목 확인 필요" if has_warnings else "데이터 적용 완료"


def _source_bytes(source: Any) -> bytes:
    """Freeze every workbook input into immutable bytes.

    Streamlit's UploadedFile is a seekable in-memory stream.  Freezing it once
    prevents ExcelFile/read_excel from depending on a pointer left by hashing,
    validation, or a previous rerun.  Paths used by both sample catalogs go
    through this same path, so all three UI entry points share one parser.
    """
    if isinstance(source, bytes):
        return source
    if isinstance(source, (bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        value = source.getvalue()
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value)
    if hasattr(source, "read"):
        position = source.tell() if hasattr(source, "tell") else None
        if hasattr(source, "seek"):
            source.seek(0)
        value = source.read()
        if position is not None and hasattr(source, "seek"):
            source.seek(position)
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value)
    raise DataLoadError("엑셀 파일을 읽을 수 없습니다.")


def uploaded_signature(uploaded_file: Any) -> str:
    return hashlib.sha256(_source_bytes(uploaded_file)).hexdigest()


def source_signature(source: Any, filename: str = "") -> str:
    """Return a content-based signature without keeping file handles in state."""
    try:
        return hashlib.sha256(_source_bytes(source)).hexdigest()
    except Exception:
        pass
    return hashlib.sha256(str(filename).encode("utf-8")).hexdigest()


def validation_error_message(
    validation: ValidationReport, recommendation_source: str | None = None,
) -> str:
    """Collapse internal validation details into one actionable UI line."""
    errors = [message for message in validation.messages if message.level == "오류"]
    if recommendation_source == "none" or any(
        message.sheet in {"recommendations", "v2_recommendations"}
        and "로드되지 않았습니다" in message.message
        for message in errors
    ):
        return "추천 후보를 만들 수 없습니다: recommendations 시트 없음"

    missing: list[str] = []
    for message in errors:
        if "필수 컬럼" not in message.message:
            continue
        match = re.search(r"`([^`]+)`", message.message)
        if match:
            missing.append(f"{message.sheet}.{match.group(1)}")
    if missing:
        return "필수 컬럼 누락: " + ", ".join(dict.fromkeys(missing))
    if errors:
        first = errors[0]
        detail = first.message.replace("`", "")
        return f"{first.sheet}: {detail}" if first.sheet else detail
    return "데이터를 적용할 수 없습니다."


def _has_user_warnings(validation: ValidationReport) -> bool:
    """Hide known benign full-network duplicates from the success banner."""
    warnings = [message for message in validation.messages if message.level == "주의"]
    return any("동일 source_id/target_id 조합" not in message.message for message in warnings)


def _store_failed_candidate(
    state: MutableMapping[str, Any], data: dict[str, pd.DataFrame], validation: ValidationReport,
    filename: str, source_type: str, upload_report: dict[str, Any] | None = None,
) -> None:
    state["pending_varo_data"] = data
    state["pending_varo_validation"] = validation
    state["pending_varo_recommendations"] = []
    state["pending_uploaded_filename"] = filename
    state["pending_data_source_type"] = source_type
    state["pending_upload_report"] = upload_report or {}
    state["data_apply_message"] = None


def load_and_apply(
    state: MutableMapping[str, Any], source: Any, filename: str, source_type: str,
) -> bool:
    """Load, validate, run approved algorithms, and apply canonical state.

    Hardened so a malformed upload never crashes the app: any unexpected error is
    converted into a user-facing load error and previous state is preserved.
    """
    try:
        content = _source_bytes(source)
        signature = hashlib.sha256(content).hexdigest()
        data, load_report = load_excel_data(BytesIO(content), return_report=True)
        pipeline_state = build_v2_state(data)
    except DataLoadError as exc:
        clear_pending(state)
        state["pending_load_error"] = str(exc)
        state["data_apply_message"] = None
        return False
    except Exception:  # pragma: no cover - defensive: never crash the upload
        clear_pending(state)
        state["pending_load_error"] = "엑셀 파일을 처리할 수 없습니다."
        state["data_apply_message"] = None
        return False

    state.pop("pending_load_error", None)
    state.pop("pending_validation_error", None)
    validation = pipeline_state["validation"]
    recommendations = pipeline_state["recommendations"]
    pipeline_result = pipeline_state.get("pipeline_result", {})
    recommendation_source = pipeline_state.get("recommendation_source", "uploaded")
    candidate_info = pipeline_state.get("candidate_info", {})
    upload_report = build_upload_report(load_report, validation, recommendation_source, candidate_info, filename)

    if validation.has_errors:
        _store_failed_candidate(state, data, validation, filename, source_type, upload_report)
        state["pending_validation_error"] = validation_error_message(validation, recommendation_source)
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
    )
    payload["upload_report"] = upload_report
    payload["recommendation_source"] = recommendation_source
    apply_state_payload(state, payload)
    clear_pending(state)
    state["data_apply_message"] = application_message(_has_user_warnings(validation))
    return True
