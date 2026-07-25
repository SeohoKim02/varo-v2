"""Download/export builders for Varo V2.

All exports are derived from already DQN-stripped pipeline structures and the
standard recommendation contract. Historical DQN reward/loss/model/q-table
artifacts are never read or written here; ``dqn_action`` stays "미연결" and
``dqn_correction`` stays 0.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

import pandas as pd

from services.dqn_guard import is_dqn_column
from services.upload_quality import (
    candidate_detail_rows,
    column_mapping_rows,
    date_normalization_rows,
    generation_summary_rows,
    upload_quality_rows,
)
from services.v2_summaries import (
    recommendation_reason_rows,
    sensitivity_summary,
    vhs_neutral_rows,
)

DQN_STATUS_LABEL = "미연결"

_VHS_NEUTRAL_HEADERS = {
    "route_id": "route_id", "product_name": "상품",
    "uploaded_vhs": "업로드 VHS", "recalculated_vhs": "재계산 VHS",
    "calculated_components": "계산 구성요소", "neutral_components": "중립값 구성요소",
    "excluded_components": "제외 구성요소", "neutral_reason": "중립 사유",
    "final_basis": "기준", "note": "비고",
}
_SENSITIVITY_HEADERS = {
    "route_id": "route_id", "product_name": "상품",
    "sensitivity_cost": "비용 민감도", "sensitivity_distance": "거리 민감도",
    "sensitivity_quantity": "수량 민감도", "sensitivity_vhs": "VHS 민감도",
    "overall_sensitivity": "종합 민감도", "stability_note": "안정성 비고",
}
_REASON_HEADERS = {
    "route_id": "route_id", "product_name": "상품",
    "recommendation_reason": "추천 사유", "caution": "주의사항", "dqn_note": "DQN 안내",
}
_CANDIDATE_HEADERS = {
    "route_id": "route_id", "product_name": "상품",
    "source_name": "출발 점포", "target_name": "도착 점포", "route_type": "경로 유형",
    "transfer_qty": "추천 수량", "expected_saving": "예상 절감액",
    "candidate_score": "후보 점수", "score_reason": "점수 근거",
    "direct_available": "직접 경로", "via_dc_available": "DC 경유 경로",
    "selected_route_basis": "경로 선택 근거", "days_to_expiry_source": "출발지 잔여일수",
    "recommendation_source": "추천 출처", "dqn_status": "DQN 상태",
}
_CANDIDATE_SCORE_COLUMNS = ("route_id", "product_name", "candidate_score", "score_reason",
                           "direct_available", "via_dc_available", "selected_route_basis")


def _upload_quality_sheets(upload_report: Mapping[str, Any] | None) -> dict[str, pd.DataFrame]:
    if not upload_report:
        return {}
    sheets: dict[str, pd.DataFrame] = {
        "업로드품질": _frame(upload_quality_rows(upload_report)),
        "컬럼매핑": _frame(column_mapping_rows(upload_report)),
        "날짜환산": _frame(date_normalization_rows(upload_report)),
    }
    candidates = candidate_detail_rows(upload_report)
    if candidates:
        sheets["생성후보"] = _frame(candidates).rename(columns=_CANDIDATE_HEADERS)
        sheets["후보점수"] = _frame(candidates, list(_CANDIDATE_SCORE_COLUMNS)).rename(columns=_CANDIDATE_HEADERS)
        sheets["후보생성요약"] = _frame(generation_summary_rows(upload_report))
    return sheets

# Friendly, readable export columns. Built explicitly from safe fields so no
# reward/loss/q-table value can leak into a download.
_RECOMMENDATION_EXPORT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("순위", "rank"),
    ("route_id", "route_id"),
    ("상품", "product_name"),
    ("출발 점포", "source_name"),
    ("도착 점포", "target_name"),
    ("경로 유형", "route_type_label"),
    ("추천 수량", "recommended_qty"),
    ("이동수단", "transport_type"),
    ("거리(km)", "distance_km"),
    ("예상 시간(분)", "expected_time_min"),
    ("이동비용", "estimated_cost"),
    ("예상 절감액", "expected_saving"),
    ("VHS(재계산)", "vhs_score"),
    ("업로드 VHS", "uploaded_vhs_score"),
    ("신뢰도", "confidence_score"),
    ("추천 등급", "recommendation_grade"),
    ("Greedy 전략", "greedy_action"),
    ("Varo 추천", "varo_action"),
    ("DQN 상태", "dqn_action"),
    ("추천 이유", "reason"),
)


def _route_type_label(rec: Mapping[str, Any]) -> str:
    label = rec.get("route_type_label")
    if label:
        return str(label)
    return {"DIRECT": "직접 이동", "VIA_DC": "DC 경유"}.get(str(rec.get("route_type")), "-")


def _cell(rec: Mapping[str, Any], field: str) -> Any:
    if field == "route_type_label":
        return _route_type_label(rec)
    if field == "source_name":
        return rec.get("source_name") or rec.get("source_id") or "-"
    if field == "target_name":
        return rec.get("target_name") or rec.get("target_id") or "-"
    if field == "dqn_action":
        # Sentinel only — never a historical action value.
        return DQN_STATUS_LABEL
    value = rec.get(field)
    return "" if value is None else value


def recommendations_export_frame(
    recommendations: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Readable recommendation table with raw values (numbers kept numeric)."""
    from services.analysis_pipeline import sort_recommendations

    ordered = sort_recommendations(list(recommendations or []))
    rows: list[dict[str, Any]] = []
    for index, rec in enumerate(ordered, start=1):
        row: dict[str, Any] = {}
        for header, field in _RECOMMENDATION_EXPORT_COLUMNS:
            row[header] = index if field == "rank" else _cell(rec, field)
        rows.append(row)
    columns = [header for header, _ in _RECOMMENDATION_EXPORT_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def _frame(rows: Any, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    elif isinstance(rows, list):
        frame = pd.DataFrame(rows)
    else:
        frame = pd.DataFrame()
    if columns is not None and not frame.empty:
        keep = [column for column in columns if column in frame.columns]
        if keep:
            frame = frame[keep]
    return frame


# Benign Korean label columns that mention DQN only as a static sentinel/note,
# never a numeric DQN value. These are intentionally kept in exports.
_DQN_LABEL_ALLOWLIST = frozenset({"DQN 상태", "DQN 안내"})


def _assert_dqn_free(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop any reward/loss/q-table style column that slipped through.

    Friendly label columns ("DQN 상태"=미연결, "DQN 안내"=정적 안내문)은 값이 아닌
    표기이므로 유지하고, 그 외 DQN/reward/loss 계열 컬럼만 제거한다.
    """
    blocked = [
        column
        for column in frame.columns
        if is_dqn_column(column) and str(column) not in _DQN_LABEL_ALLOWLIST
    ]
    return frame.drop(columns=blocked, errors="ignore") if blocked else frame


def _kv_frame(pairs: Sequence[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"항목": key, "값": "" if value is None else value} for key, value in pairs]
    )


def vhs_comparison_frame(
    pipeline: Mapping[str, Any] | None,
    recommendations: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Uploaded vs recalculated VHS per route, with basis and neutral-component note."""
    pipeline = pipeline if isinstance(pipeline, Mapping) else {}
    vhs = pipeline.get("vhs_analysis") or {}
    defaulted = vhs.get("defaulted_component_columns") or []
    neutral_label = (
        f"{len(defaulted)}개 구성요소 중립값 적용" if defaulted else "중립값 없음"
    )
    columns = [
        "route_id", "product_name", "source_name", "target_name",
        "uploaded_vhs", "recalculated_vhs", "difference",
        "basis", "neutral_components", "note",
    ]
    rows: list[dict[str, Any]] = []
    for rec in recommendations or []:
        uploaded = rec.get("uploaded_vhs_score")
        recalculated = rec.get("vhs_score")
        difference = None
        try:
            if uploaded is not None and recalculated is not None:
                difference = round(float(recalculated) - float(uploaded), 1)
        except (TypeError, ValueError):
            difference = None
        if difference is None:
            note = "비교할 업로드 점수가 없습니다"
        elif difference < 0:
            note = "재계산값이 업로드값보다 낮습니다"
        elif difference > 0:
            note = "재계산값이 업로드값보다 높습니다"
        else:
            note = "두 값이 동일합니다"
        rows.append({
            "route_id": rec.get("route_id"),
            "product_name": rec.get("product_name"),
            "source_name": rec.get("source_name") or rec.get("source_id"),
            "target_name": rec.get("target_name") or rec.get("target_id"),
            "uploaded_vhs": uploaded,
            "recalculated_vhs": recalculated,
            "difference": difference,
            "basis": "재계산 VHS · Varo V2 내부 알고리즘",
            "neutral_components": neutral_label,
            "note": note,
        })
    return pd.DataFrame(rows, columns=columns)


def _confidence_frame(recommendations: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "route_id": rec.get("route_id"),
            "상품": rec.get("product_name"),
            "신뢰도": rec.get("confidence_score"),
            "추천 등급": rec.get("recommendation_grade"),
            "DQN 상태": DQN_STATUS_LABEL,
        }
        for rec in recommendations or []
    ]
    return pd.DataFrame(rows, columns=["route_id", "상품", "신뢰도", "추천 등급", "DQN 상태"])


def _optimality_frame(pipeline: Mapping[str, Any]) -> pd.DataFrame:
    gap = (pipeline.get("validation_report") or {}).get("optimality_gap") or {}
    if not gap:
        return pd.DataFrame()
    return _kv_frame([
        ("상태", gap.get("status", "-")),
        ("Optimality Gap", gap.get("gap_str", "-")),
        ("후보 일치율(%)", gap.get("match_rate", "-")),
        ("Varo 비용", gap.get("varo_total", "-")),
        ("최적 비용", gap.get("opt_total", "-")),
        ("비교 가능 후보", gap.get("comparable_candidate_count", gap.get("candidates_used", 0))),
        ("검증 방식", gap.get("opt_method", "-")),
        ("계산 함수", gap.get("calculation_function", "-")),
        ("공식", gap.get("formula", "-")),
    ])


def _algorithm_status_frame(pipeline: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in pipeline.get("connected_algorithms") or []:
        rows.append({"상태": "연결", "함수": name, "사유": ""})
    for item in pipeline.get("deferred_algorithms") or []:
        rows.append({
            "상태": "보류",
            "함수": item.get("algorithm", "-"),
            "사유": item.get("reason", ""),
        })
    return pd.DataFrame(rows, columns=["상태", "함수", "사유"])


def _dqn_exclusion_frame(pipeline: Mapping[str, Any]) -> pd.DataFrame:
    exclusion = pipeline.get("excluded_dqn_artifacts") or {}
    pairs: list[tuple[str, Any]] = [
        ("DQN 상태", exclusion.get("status", DQN_STATUS_LABEL)),
        ("제외 사유", exclusion.get("reason", "")),
        ("과거 아티팩트 사용", "아니오" if not exclusion.get("artifacts_read") else "예"),
        ("학습 실행", "아니오" if not exclusion.get("training_executed") else "예"),
        ("추론 실행", "아니오" if not exclusion.get("inference_executed") else "예"),
        ("점수 반영", "아니오" if not exclusion.get("score_influence") else "예"),
    ]
    for pattern in exclusion.get("blocked_patterns") or []:
        pairs.append(("차단 패턴", pattern))
    return _kv_frame(pairs)


def _write_excel(sheets: Mapping[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        wrote_any = False
        for name, frame in sheets.items():
            clean = _assert_dqn_free(frame if isinstance(frame, pd.DataFrame) else pd.DataFrame())
            if clean.empty:
                continue
            clean.to_excel(writer, sheet_name=name[:31], index=False)
            wrote_any = True
        if not wrote_any:
            pd.DataFrame([{"안내": "내보낼 데이터가 없습니다."}]).to_excel(
                writer, sheet_name="안내", index=False
            )
    return buffer.getvalue()


def recommendations_csv_bytes(recommendations: Sequence[Mapping[str, Any]]) -> bytes:
    """UTF-8 BOM CSV so Korean headers open correctly in Excel."""
    frame = _assert_dqn_free(recommendations_export_frame(recommendations))
    return frame.to_csv(index=False).encode("utf-8-sig")


def recommendations_excel_bytes(recommendations: Sequence[Mapping[str, Any]]) -> bytes:
    return _write_excel({"추천결과": recommendations_export_frame(recommendations)})


def analysis_result_excel_bytes(
    pipeline: Mapping[str, Any] | None,
    recommendations: Sequence[Mapping[str, Any]],
    upload_report: Mapping[str, Any] | None = None,
) -> bytes:
    """Multi-sheet analysis workbook from the connected pipeline result."""
    pipeline = pipeline if isinstance(pipeline, Mapping) else {}
    vhs = pipeline.get("vhs_analysis") or {}
    greedy = pipeline.get("greedy_analysis") or {}
    summary_pairs: list[tuple[str, Any]] = [
        ("최종 상태", pipeline.get("status", "-")),
        ("결과 기준", pipeline.get("result_basis", "-")),
        ("연결 함수 수", len(pipeline.get("connected_algorithms") or [])),
        ("보류 항목 수", len(pipeline.get("deferred_algorithms") or [])),
        ("추천 결과 수", len(recommendations or [])),
        ("DQN 상태", DQN_STATUS_LABEL),
        ("DQN 제외 사유", (pipeline.get("excluded_dqn_artifacts") or {}).get("reason", "")),
    ]
    sheets = {
        "추천결과": recommendations_export_frame(recommendations),
        "VHS분석": _frame(
            vhs.get("comparison_rows") or vhs.get("score_rows"),
            ["route_id", "product_name", "uploaded_vhs_score",
             "recalculated_vhs_score", "score_difference"],
        ),
        "VHS비교": vhs_comparison_frame(pipeline, recommendations),
        "VHS중립값": _frame(vhs_neutral_rows(pipeline, recommendations)).rename(columns=_VHS_NEUTRAL_HEADERS),
        "민감도요약": _frame(sensitivity_summary(recommendations)).rename(columns=_SENSITIVITY_HEADERS),
        "추천사유": _frame(recommendation_reason_rows(recommendations)).rename(columns=_REASON_HEADERS),
        "Greedy분석": _frame(greedy.get("rows")),
        "신뢰도": _confidence_frame(recommendations),
        "최적성검증": _optimality_frame(pipeline),
        "알고리즘상태": _algorithm_status_frame(pipeline),
        "검증요약": _kv_frame(summary_pairs),
        "DQN제외": _dqn_exclusion_frame(pipeline),
    }
    sheets.update(_upload_quality_sheets(upload_report))
    return _write_excel(sheets)


def validation_report_excel_bytes(
    validation: Any,
    pipeline: Mapping[str, Any] | None,
    recommendations: Sequence[Mapping[str, Any]] | None = None,
    upload_report: Mapping[str, Any] | None = None,
) -> bytes:
    """Validation report workbook: messages, sheet summary, algorithm status."""
    pipeline = pipeline if isinstance(pipeline, Mapping) else {}
    messages_frame = pd.DataFrame()
    summary_frame = pd.DataFrame()
    status = "데이터 없음"
    if validation is not None:
        status = getattr(validation, "status", "데이터 없음")
        messages = getattr(validation, "messages", None) or []
        messages_frame = pd.DataFrame(
            [message.to_dict() for message in messages]
        ).rename(columns={"level": "등급", "sheet": "시트", "message": "내용"})
        summary = getattr(validation, "summary", None) or {}
        if summary:
            summary_frame = _kv_frame(list(summary.items()))

    header_pairs: list[tuple[str, Any]] = [
        ("검증 상태", status),
        ("최종 분석 상태", pipeline.get("status", "-")),
        ("결과 기준", pipeline.get("result_basis", "-")),
        ("추천 결과 수", len(recommendations or [])),
        ("DQN 상태", DQN_STATUS_LABEL),
    ]
    sheets = {
        "검증개요": _kv_frame(header_pairs),
        "검증메시지": messages_frame,
        "시트요약": summary_frame,
        "알고리즘상태": _algorithm_status_frame(pipeline),
        "최적성검증": _optimality_frame(pipeline),
        "DQN제외": _dqn_exclusion_frame(pipeline),
    }
    sheets.update(_upload_quality_sheets(upload_report))
    return _write_excel(sheets)
