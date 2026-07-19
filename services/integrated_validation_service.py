"""Button-only, read-only integrated validation for the ten DQN workbooks.

The service composes the existing V2 pipeline, DQN diagnostics, detailed
sensitivity, and optimality-gap services.  It owns no Streamlit state and does
not persist report files.  Each sample is loaded and processed in a local
context so one failure cannot alter the active application data or stop later
samples.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from services.analysis_pipeline import run_analysis_pipeline
from services.data_application import LOAD_CACHE_VERSION
from services.data_loader import load_excel_data
from services.dqn_samples import (
    BALANCE_POLICY,
    DQN_SAMPLES,
    DqnSample,
    balanced_recommendations,
    diagnose_dqn_training_sets,
    dqn_sample_path,
)
from services.dqn_service import (
    can_apply_dqn_to_current_data,
    data_signature_from_recommendations,
    load_latest_dqn_result,
    train_dqn,
)
from services.optimality_gap_service import (
    CONSTRAINT_VERSION,
    OPTIMALITY_CALCULATION_VERSION,
    build_optimality_settings,
    run_optimality_gap,
)
from services.sensitivity_service import (
    SENSITIVITY_CALCULATION_VERSION,
    VHS_WEIGHT_VERSION,
    build_sensitivity_settings,
    run_detailed_sensitivity,
)


INTEGRATED_VALIDATION_VERSION = "integrated-samples-v1"
GAP_REVIEW_THRESHOLD_PCT = 10.0
DQN_TRAINING_EPISODES = 80
_CACHE_LIMIT = 32
_SAMPLE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()

SCOPE_LABELS = {
    "quick": "빠른 검증",
    "standard": "표준 검증",
    "full": "전체 검증",
}
ProgressCallback = Callable[[dict[str, Any]], None]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _average(values: Sequence[Any]) -> float | None:
    clean = [value for value in (_number(item) for item in values) if value is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _distribution_stats(value: Mapping[str, Any] | None) -> tuple[int, float | None]:
    distribution = {str(key): int(count or 0) for key, count in (value or {}).items()}
    total = sum(distribution.values())
    return (
        len([count for count in distribution.values() if count > 0]),
        round(max(distribution.values()) / total, 4) if total and distribution else None,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if value is pd.NA:
        return None
    return value


def algorithm_version() -> str:
    return "|".join((
        LOAD_CACHE_VERSION,
        SENSITIVITY_CALCULATION_VERSION,
        VHS_WEIGHT_VERSION,
        OPTIMALITY_CALCULATION_VERSION,
        CONSTRAINT_VERSION,
    ))


def sample_catalog() -> list[dict[str, Any]]:
    """Return the immutable 01~10 selection catalog without reading workbooks."""
    rows = []
    for sample in DQN_SAMPLES:
        path = dqn_sample_path(sample)
        rows.append({
            "sample_id": f"sample_{sample.number:02d}",
            "number": sample.number,
            "label": f"샘플 {sample.number:02d}",
            "filename": path.name,
            "path": str(path),
            "store_count_hint": sample.workbook.store_count,
            "dc_count_hint": sample.workbook.dc_count,
        })
    return rows


def build_integrated_settings(
    scope: str = "standard",
    sample_ids: Sequence[str] | None = None,
    train_original: bool = False,
    train_balanced: bool = False,
) -> dict[str, Any]:
    normalized_scope = str(scope or "standard").lower()
    if normalized_scope not in SCOPE_LABELS:
        normalized_scope = "standard"
    available = [row["sample_id"] for row in sample_catalog()]
    requested = available if sample_ids is None else [str(item) for item in sample_ids]
    selected = [sample_id for sample_id in available if sample_id in requested]
    training_allowed = normalized_scope == "full"
    if normalized_scope == "full":
        sensitivity = {
            "variables": [
                "transport_cost", "distance", "demand", "disposal_risk",
                "vhs_weight", "quantity", "shortage", "promotion",
            ],
            "minimum_pct": -20.0,
            "maximum_pct": 20.0,
            "step_count": 9,
            "candidate_limit": 20,
        }
        optimality = {
            "candidate_limit": None,
            "max_routes": 10,
            "search_mode": "auto",
            "time_limit": 5.0,
        }
    else:
        sensitivity = {
            "variables": ["transport_cost", "distance", "demand", "disposal_risk", "vhs_weight"],
            "minimum_pct": -10.0,
            "maximum_pct": 10.0,
            "step_count": 5,
            "candidate_limit": 10,
        }
        optimality = {
            "candidate_limit": 20,
            "max_routes": 5,
            "search_mode": "auto",
            "time_limit": 3.0,
        }
    return {
        "scope": normalized_scope,
        "scope_label": SCOPE_LABELS[normalized_scope],
        "sample_ids": selected,
        "train_original": bool(train_original and training_allowed),
        "train_balanced": bool(train_balanced and training_allowed),
        "sensitivity": sensitivity,
        "optimality": optimality,
        "integrated_version": INTEGRATED_VALIDATION_VERSION,
        "algorithm_version": algorithm_version(),
    }


def build_sample_cache_key(sample: DqnSample, file_hash: str, settings: Mapping[str, Any]) -> str:
    payload = {
        "sample_id": f"sample_{sample.number:02d}",
        "file_hash": file_hash,
        "scope": settings.get("scope"),
        "train_original": bool(settings.get("train_original")),
        "train_balanced": bool(settings.get("train_balanced")),
        "sensitivity": settings.get("sensitivity"),
        "optimality": settings.get("optimality"),
        "algorithm_version": settings.get("algorithm_version", algorithm_version()),
        "integrated_version": settings.get("integrated_version", INTEGRATED_VALIDATION_VERSION),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def clear_integrated_validation_cache() -> None:
    _SAMPLE_CACHE.clear()


def failed_sample_ids(result: Mapping[str, Any] | None) -> list[str]:
    return [
        str(item.get("sample_id"))
        for item in (result or {}).get("samples", [])
        if item.get("overall_status") == "실패" and item.get("sample_id")
    ]


def _notify(callback: ProgressCallback | None, **event: Any) -> None:
    if callback is not None:
        callback(dict(event))


def _compact_training_result(
    raw_result: Any,
    signature: str,
    *,
    stored_result_used: bool,
) -> dict[str, Any]:
    if raw_result is None:
        return {
            "training_status": "미학습",
            "prediction_type_count": 0,
            "prediction_dominant_ratio": None,
            "loss_start": None,
            "loss_end": None,
            "stability_status": "미학습",
            "vhs_applicable": False,
            "stored_result_used": False,
        }
    value = raw_result.to_dict() if hasattr(raw_result, "to_dict") else dict(raw_result)
    prediction = value.get("prediction_distribution") or value.get("action_distribution") or {}
    prediction_types, prediction_dominance = _distribution_stats(prediction)
    losses = [item for item in (value.get("loss_history") or []) if _number(item) is not None]
    status = str(
        value.get("final_status") or value.get("stability_status") or value.get("status") or "미학습"
    )
    return {
        "training_status": status,
        "prediction_type_count": prediction_types,
        "prediction_dominant_ratio": prediction_dominance,
        "prediction_distribution": dict(prediction),
        "loss_start": _number(losses[0]) if losses else None,
        "loss_end": _number(losses[-1]) if losses else None,
        "stability_status": str(value.get("stability_status") or status),
        "vhs_applicable": can_apply_dqn_to_current_data(value, signature),
        "stored_result_used": stored_result_used,
        "episodes": int(value.get("episodes") or 0),
        "message": str(value.get("message") or ""),
    }


def _dqn_variant(
    recommendations: list[dict[str, Any]],
    signature: str,
    sample: DqnSample,
    mode: str,
    train_requested: bool,
    store_count: int,
    dc_count: int,
) -> dict[str, Any]:
    training_set = {
        "sample_id": f"sample_{sample.number:02d}",
        "sample_name": sample.workbook.filename,
        "mode": mode,
        "recommendations": recommendations,
    }
    quality = diagnose_dqn_training_sets([training_set])[0]
    raw_result = None
    stored = False
    if train_requested:
        raw_result = train_dqn(
            recommendations,
            data_signature=signature,
            episodes=DQN_TRAINING_EPISODES,
            sample_id=f"sample_{sample.number:02d}",
            training_mode=mode,
            store_count=store_count,
            dc_count=dc_count,
            seed=17 + sample.number,
        )
    else:
        raw_result = load_latest_dqn_result(signature, training_mode=mode)
        stored = raw_result is not None
    return {
        "target_type_count": int(quality.get("target_type_count") or 0),
        "target_dominant_ratio": _number(quality.get("dominant_ratio")),
        "quality_status": quality.get("status") or "검토 필요",
        "quality_reason": quality.get("reason") or "",
        "target_distribution": dict(quality.get("target_distribution") or {}),
        **_compact_training_result(raw_result, signature, stored_result_used=stored),
    }


def _quick_sensitivity(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    rows = list((pipeline.get("sensitivity_analysis") or {}).get("rows") or [])
    labels = [str(row.get("overall_sensitivity")) for row in rows if row.get("overall_sensitivity")]
    common = Counter(labels).most_common(1)
    return {
        "status": "빠른 요약 완료" if rows else "계산 제외",
        "scenario_count": 0,
        "top1_retention_rate": None,
        "top3_retention_rate": None,
        "max_rank_change": None,
        "most_sensitive_variable": None,
        "most_stable_variable": None,
        "score": None,
        "rating": common[0][0] if common else "상세 계산 전",
        "excluded_scenario_count": 0,
        "calculation_ms": 0.0,
        "mode": "pipeline_fast_summary",
    }


def _detailed_sensitivity(
    recommendations: list[dict[str, Any]],
    data: Mapping[str, Any],
    weights: Mapping[str, float],
    signature: str,
    configured: Mapping[str, Any],
) -> dict[str, Any]:
    settings = build_sensitivity_settings(
        recommendations,
        data,
        weights,
        variables=configured.get("variables"),
        minimum_pct=float(configured.get("minimum_pct") or -10.0),
        maximum_pct=float(configured.get("maximum_pct") or 10.0),
        step_count=int(configured.get("step_count") or 5),
        candidate_limit=configured.get("candidate_limit"),
    )
    result = run_detailed_sensitivity(recommendations, data, weights, settings, signature)
    summary = result.get("summary") or {}
    metadata = result.get("metadata") or {}
    return {
        "status": "계산 완료" if summary else "계산 제외",
        "scenario_count": int(summary.get("scenario_count") or 0),
        "completed_scenario_count": int(summary.get("completed_scenario_count") or 0),
        "top1_retention_rate": _number(summary.get("top1_retention_rate")),
        "top3_retention_rate": _number(summary.get("top3_retention_rate")),
        "max_rank_change": int(summary.get("max_rank_change") or 0),
        "most_sensitive_variable": summary.get("most_sensitive_variable"),
        "most_stable_variable": summary.get("most_stable_variable"),
        "score": _number(summary.get("score")),
        "rating": summary.get("rating") or "-",
        "excluded_scenario_count": int(summary.get("excluded_scenario_count") or 0),
        "calculation_ms": _number(metadata.get("calculation_ms"), 0.0),
        "cache_hit": bool(metadata.get("cache_hit")),
        "mode": "detailed_oat",
        "settings": _json_safe(result.get("settings") or settings),
    }


def _compact_optimality(result: Mapping[str, Any]) -> dict[str, Any]:
    combinations = result.get("combinations") or {}
    gap = result.get("gap") or {}
    match = (result.get("matches") or {}).get("varo_vs_best") or {}
    summary = result.get("summary") or {}
    search = result.get("search") or {}
    metadata = result.get("metadata") or {}
    return {
        "status": summary.get("status") or search.get("status") or "계산 제외",
        "search_status": search.get("status") or "계산 제외",
        "search_method": search.get("method") or "-",
        "exact": bool(search.get("optimal")),
        "limited": bool(search.get("available") and not search.get("optimal")),
        "varo_total_saving": _number((combinations.get("varo") or {}).get("total_saving"), 0.0),
        "greedy_total_saving": _number((combinations.get("greedy") or {}).get("total_saving"), 0.0),
        "best_total_saving": _number((combinations.get("best") or {}).get("total_saving"), 0.0),
        "varo_gap_pct": _number(gap.get("gap_pct")),
        "greedy_gap_pct": _number(gap.get("greedy_gap_pct")),
        "target_pct": _number(gap.get("target_pct")),
        "route_match_pct": _number(match.get("jaccard_pct")),
        "applied_constraint_count": int(summary.get("applied_constraint_count") or 0),
        "unapplied_constraint_count": int(summary.get("unapplied_constraint_count") or 0),
        "constraint_violation_count": int(summary.get("constraint_violation_count") or 0),
        "feasible_candidate_count": int(summary.get("feasible_candidate_count") or 0),
        "excluded_candidate_count": int(summary.get("excluded_candidate_count") or 0),
        "calculation_ms": _number(metadata.get("calculation_ms"), 0.0),
        "cache_hit": bool(metadata.get("cache_hit")),
        "message": search.get("message") or "",
    }


def _collect_core(
    sample: DqnSample,
    path: Path,
    data: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    file_hash: str,
) -> dict[str, Any]:
    recommendations = [dict(item) for item in pipeline.get("recommendations") or []]
    ordered_vhs = sorted(
        recommendations,
        key=lambda item: (_number(item.get("vhs_rank"), float("inf")), str(item.get("route_id") or "")),
    )
    ordered_greedy = sorted(
        recommendations,
        key=lambda item: (_number(item.get("greedy_rank"), float("inf")), str(item.get("route_id") or "")),
    )
    top_vhs = ordered_vhs[0] if ordered_vhs else {}
    top_greedy = ordered_greedy[0] if ordered_greedy else {}
    vhs_analysis = pipeline.get("vhs_analysis") or {}
    greedy_analysis = pipeline.get("greedy_analysis") or {}
    pareto_analysis = pipeline.get("pareto_analysis") or {}
    weights = vhs_analysis.get("weights") or {}
    defaulted = list(vhs_analysis.get("defaulted_component_columns") or [])
    validation_summary = ((pipeline.get("validation_report") or {}).get("data_validation") or {}).get("summary") or {}
    stores = data.get("stores") if isinstance(data.get("stores"), pd.DataFrame) else pd.DataFrame()
    store_types = stores.get("node_type", pd.Series(index=stores.index, dtype=str)).astype(str).str.upper()
    store_count = int(validation_summary.get("store_count") or (store_types == "STORE").sum())
    dc_count = int(validation_summary.get("dc_count") or (store_types == "DC").sum())
    product_count = int(validation_summary.get("product_count") or len(data.get("products", [])))
    raw_row_count = sum(
        len(frame) for key, frame in data.items()
        if key in {"stores", "products", "inventory", "routes", "recommendations"}
        and isinstance(frame, pd.DataFrame)
    )
    pareto_rows = list(pareto_analysis.get("rows") or [])
    top_pareto = next(
        (row for row in pareto_rows if str(row.get("route_id")) == str(top_vhs.get("route_id"))),
        {},
    )
    top3_vhs = {str(row.get("route_id")) for row in ordered_vhs[:3]}
    top3_greedy = {str(row.get("route_id")) for row in ordered_greedy[:3]}
    positive_savings = [_number(row.get("expected_saving"), 0.0) or 0.0 for row in recommendations]
    vhs_values = [_number(row.get("vhs_score")) for row in recommendations]
    vhs_values = [value for value in vhs_values if value is not None]
    return {
        "sample_id": f"sample_{sample.number:02d}",
        "sample_label": f"샘플 {sample.number:02d}",
        "filename": path.name,
        "metadata": {
            "sample_id": f"sample_{sample.number:02d}",
            "filename": path.name,
            "file_path": str(path),
            "file_size": path.stat().st_size,
            "file_mtime_ns": path.stat().st_mtime_ns,
            "file_sha256_before": file_hash,
            "file_sha256_after": None,
            "source_unchanged": None,
            "store_count": store_count,
            "dc_count": dc_count,
            "product_count": product_count,
            "raw_row_count": int(raw_row_count),
            "candidate_count": len(recommendations),
            "executable_candidate_count": len(recommendations),
            "data_signature": data_signature_from_recommendations(recommendations),
            "load_status": "성공",
            "pipeline_status": pipeline.get("status") or "unknown",
        },
        "vhs": {
            "candidate_count": len(recommendations),
            "average_vhs": round(sum(vhs_values) / len(vhs_values), 4) if vhs_values else None,
            "highest_vhs": max(vhs_values) if vhs_values else None,
            "lowest_vhs": min(vhs_values) if vhs_values else None,
            "top1_route_id": top_vhs.get("route_id"),
            "top1_product": top_vhs.get("product_name"),
            "top1_source": top_vhs.get("source_name"),
            "top1_target": top_vhs.get("target_name"),
            "top1_route_type": top_vhs.get("route_type"),
            "top1_quantity": _number(top_vhs.get("recommended_qty")),
            "top1_expected_saving": _number(top_vhs.get("expected_saving")),
            "total_expected_saving": round(sum(positive_savings), 4),
            "weight_sum": round(sum(float(value or 0.0) for value in weights.values()), 6),
            "defaulted_component_count": len(defaulted),
            "neutral_value_use_count": len(defaulted) * len(recommendations),
            "status": "정상" if recommendations and vhs_values else "실패",
        },
        "greedy": {
            "candidate_count": int(greedy_analysis.get("comparison_count") or len(ordered_greedy)),
            "top1_route_id": top_greedy.get("route_id"),
            "vhs_top1_match": bool(top_vhs and top_greedy and top_vhs.get("route_id") == top_greedy.get("route_id")),
            "top3_match_count": len(top3_vhs & top3_greedy),
            "selection_match_pct": None,
            "varo_total_saving": round(sum(positive_savings), 4),
            "greedy_total_saving": None,
            "saving_difference": None,
            "status": "정상" if ordered_greedy else "계산 제외",
        },
        "pareto": {
            "candidate_count": int(pareto_analysis.get("comparison_count") or len(pareto_rows)),
            "front_candidate_count": int(pareto_analysis.get("non_dominated_count") or 0),
            "top1_status": top_pareto.get("pareto_status"),
            "top1_rank": _number(top_pareto.get("pareto_rank")),
            "varo_top1_is_front": _number(top_pareto.get("pareto_rank")) == 1.0,
            "dominated_candidate_count": max(0, len(pareto_rows) - int(pareto_analysis.get("non_dominated_count") or 0)),
            "status": pareto_analysis.get("status") or "계산 제외",
            "limited": False,
        },
        "pipeline_warnings": list(pipeline.get("warnings") or []),
    }


def _overall_status(result: Mapping[str, Any]) -> tuple[str, list[str]]:
    metadata = result.get("metadata") or {}
    vhs = result.get("vhs") or {}
    if metadata.get("load_status") != "성공":
        return "실패", ["데이터 로드 실패"]
    if metadata.get("pipeline_status") not in {"success", "partial"}:
        return "실패", ["추천 pipeline 실패"]
    if not metadata.get("candidate_count") or vhs.get("status") != "정상":
        return "실패", ["VHS 추천 후보 생성 실패"]
    reasons: list[str] = []
    if metadata.get("pipeline_status") == "partial":
        reasons.append("일부 pipeline 계산 보류")
    for label, section in (("원본", (result.get("dqn") or {}).get("original") or {}), ("균형형", (result.get("dqn") or {}).get("balanced") or {})):
        if section.get("quality_status") == "검토 필요":
            reasons.append(f"{label} DQN 데이터 품질 편향")
        if section.get("training_status") not in {None, "", "미학습", "정상", "연결"}:
            reasons.append(f"{label} DQN 상태 확인 필요")
    sensitivity = result.get("sensitivity") or {}
    if sensitivity.get("rating") in {"민감", "조건에 따라 변동"}:
        reasons.append("민감도 안정성 확인 필요")
    if int(sensitivity.get("excluded_scenario_count") or 0) > 0:
        reasons.append("일부 민감도 시나리오 제외")
    optimality = result.get("optimality") or {}
    gap = _number(optimality.get("varo_gap_pct"))
    if int(optimality.get("constraint_violation_count") or 0) > 0:
        reasons.append("최적성 비교 제약 위반")
    if gap is not None and gap > GAP_REVIEW_THRESHOLD_PCT:
        reasons.append(f"Varo Gap {GAP_REVIEW_THRESHOLD_PCT:.0f}% 초과")
    if optimality.get("limited"):
        reasons.append("제한 탐색 결과")
    if not (result.get("pareto") or {}).get("varo_top1_is_front", True):
        reasons.append("Varo Top1이 Pareto front 외부")
    if result.get("excluded_calculations"):
        reasons.append("일부 계산 제외")
    return ("확인 필요", list(dict.fromkeys(reasons))) if reasons else ("정상", [])


def _fatal_sample(sample: DqnSample, path: Path, stage: str, exc: Exception, started: float) -> dict[str, Any]:
    return {
        "sample_id": f"sample_{sample.number:02d}",
        "sample_label": f"샘플 {sample.number:02d}",
        "filename": path.name,
        "metadata": {
            "sample_id": f"sample_{sample.number:02d}",
            "filename": path.name,
            "file_path": str(path),
            "load_status": "실패" if stage == "데이터 읽는 중" else "성공",
            "pipeline_status": "실패",
            "candidate_count": 0,
            "executable_candidate_count": 0,
            "source_unchanged": None,
        },
        "vhs": {"status": "실패"},
        "greedy": {"status": "계산 제외"},
        "pareto": {"status": "계산 제외"},
        "dqn": {},
        "sensitivity": {"status": "계산 제외"},
        "optimality": {"status": "계산 제외"},
        "overall_status": "실패",
        "status_reasons": [f"{stage} 실패"],
        "error_stage": stage,
        "error_message": f"{type(exc).__name__}: {str(exc)[:240]}",
        "excluded_calculations": ["후속 검증 전체"],
        "processing_seconds": round(time.perf_counter() - started, 4),
        "cache_hit": False,
    }


def run_sample_validation(
    sample: DqnSample,
    settings: Mapping[str, Any],
    progress_callback: ProgressCallback | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Validate one source workbook without mutating it or external state."""
    started = time.perf_counter()
    path = dqn_sample_path(sample)
    try:
        file_hash = _file_sha256(path)
    except Exception as exc:
        return _fatal_sample(sample, path, "데이터 읽는 중", exc, started)
    cache_key = build_sample_cache_key(sample, file_hash, settings)
    if not force and cache_key in _SAMPLE_CACHE:
        cached = copy.deepcopy(_SAMPLE_CACHE[cache_key])
        cached["cache_hit"] = True
        cached["processing_seconds"] = round(time.perf_counter() - started, 4)
        _SAMPLE_CACHE.move_to_end(cache_key)
        return cached

    _notify(progress_callback, stage="데이터 읽는 중")
    try:
        data = load_excel_data(path)
    except Exception as exc:
        return _fatal_sample(sample, path, "데이터 읽는 중", exc, started)

    _notify(progress_callback, stage="추천 pipeline 실행 중")
    try:
        pipeline = run_analysis_pipeline(copy.deepcopy(data), detail_level="full").to_dict()
        if not pipeline.get("recommendations"):
            raise ValueError("추천 후보가 생성되지 않았습니다.")
    except Exception as exc:
        return _fatal_sample(sample, path, "추천 pipeline 실행 중", exc, started)

    _notify(progress_callback, stage="VHS·Greedy·Pareto 수집 중")
    result = _collect_core(sample, path, data, pipeline, file_hash)
    recommendations = [dict(item) for item in pipeline.get("recommendations") or []]
    metadata = result["metadata"]
    signature = str(metadata.get("data_signature") or "")
    exclusions: list[str] = []

    _notify(progress_callback, stage="DQN 품질 확인 중")
    original_rows = [dict(item) for item in recommendations]
    balanced_rows = balanced_recommendations(recommendations, offset=sample.number - 1)
    original_core = json.dumps(original_rows, ensure_ascii=False, sort_keys=True, default=str)
    balanced_core = json.dumps(
        [{key: value for key, value in row.items() if key != "target_action"} for row in balanced_rows],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    try:
        original_dqn = _dqn_variant(
            original_rows, signature, sample, "original", bool(settings.get("train_original")),
            int(metadata.get("store_count") or 0), int(metadata.get("dc_count") or 0),
        )
    except Exception as exc:
        original_dqn = {"quality_status": "검토 필요", "training_status": "실패", "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
        exclusions.append("원본 DQN 결과 일부")
    try:
        balanced_dqn = _dqn_variant(
            balanced_rows, signature, sample, "balanced", bool(settings.get("train_balanced")),
            int(metadata.get("store_count") or 0), int(metadata.get("dc_count") or 0),
        )
    except Exception as exc:
        balanced_dqn = {"quality_status": "검토 필요", "training_status": "실패", "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
        exclusions.append("균형형 DQN 결과 일부")
    original_ratio = _number(original_dqn.get("target_dominant_ratio"))
    balanced_ratio = _number(balanced_dqn.get("target_dominant_ratio"))
    result["dqn"] = {
        "original": original_dqn,
        "balanced": balanced_dqn,
        "numeric_features_preserved": original_core == balanced_core,
        "improved_vs_original": bool(
            balanced_dqn.get("quality_status") != "검토 필요"
            and (
                original_dqn.get("quality_status") == "검토 필요"
                or (balanced_ratio is not None and original_ratio is not None and balanced_ratio < original_ratio)
            )
        ),
        "balance_metadata": {
            "derived_from": path.name,
            "original_sample_id": result["sample_id"],
            "balance_policy": BALANCE_POLICY,
            "storage": "memory_only",
        },
    }

    _notify(progress_callback, stage="민감도 계산 중")
    try:
        if settings.get("scope") == "quick":
            result["sensitivity"] = _quick_sensitivity(pipeline)
        else:
            result["sensitivity"] = _detailed_sensitivity(
                recommendations,
                data,
                (pipeline.get("vhs_analysis") or {}).get("weights") or {},
                signature,
                settings.get("sensitivity") or {},
            )
    except Exception as exc:
        result["sensitivity"] = {"status": "계산 제외", "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
        exclusions.append("민감도 계산")

    _notify(progress_callback, stage="최적성 Gap 계산 중")
    try:
        optimality_settings = build_optimality_settings(**dict(settings.get("optimality") or {}))
        optimality_raw = run_optimality_gap(recommendations, data, optimality_settings, signature)
        result["optimality"] = _compact_optimality(optimality_raw)
        result["metadata"]["executable_candidate_count"] = result["optimality"]["feasible_candidate_count"]
        result["greedy"]["selection_match_pct"] = _number(
            ((optimality_raw.get("matches") or {}).get("varo_vs_greedy") or {}).get("jaccard_pct")
        )
        result["greedy"]["varo_total_saving"] = result["optimality"]["varo_total_saving"]
        result["greedy"]["greedy_total_saving"] = result["optimality"]["greedy_total_saving"]
        result["greedy"]["saving_difference"] = round(
            float(result["optimality"]["varo_total_saving"] or 0.0)
            - float(result["optimality"]["greedy_total_saving"] or 0.0),
            4,
        )
    except Exception as exc:
        result["optimality"] = {"status": "계산 제외", "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
        exclusions.append("최적성 Gap 계산")

    _notify(progress_callback, stage="결과 정리 중")
    try:
        after_hash = _file_sha256(path)
    except Exception:
        after_hash = None
    metadata["file_sha256_after"] = after_hash
    metadata["source_unchanged"] = after_hash == file_hash if after_hash else None
    if metadata["source_unchanged"] is False:
        exclusions.append("원본 파일 무결성 확인")
    result["excluded_calculations"] = exclusions
    result["error_stage"] = None
    result["error_message"] = None
    result["processing_seconds"] = round(time.perf_counter() - started, 4)
    result["cache_hit"] = False
    result["cache_key"] = cache_key
    status, reasons = _overall_status(result)
    if metadata["source_unchanged"] is False:
        status = "실패"
        reasons = ["원본 샘플 파일 해시 변경"] + reasons
    result["overall_status"] = status
    result["status_reasons"] = reasons
    safe_result = _json_safe(result)
    if status != "실패":
        _SAMPLE_CACHE[cache_key] = copy.deepcopy(safe_result)
        _SAMPLE_CACHE.move_to_end(cache_key)
        while len(_SAMPLE_CACHE) > _CACHE_LIMIT:
            _SAMPLE_CACHE.popitem(last=False)
    return safe_result


def _integrated_summary(samples: Sequence[Mapping[str, Any]], elapsed: float) -> dict[str, Any]:
    statuses = Counter(str(item.get("overall_status") or "실패") for item in samples)
    candidates = sum(int((item.get("metadata") or {}).get("candidate_count") or 0) for item in samples)
    savings = sum(float((item.get("vhs") or {}).get("total_expected_saving") or 0.0) for item in samples)
    return {
        "sample_count": len(samples),
        "normal_count": statuses.get("정상", 0),
        "review_count": statuses.get("확인 필요", 0),
        "failure_count": statuses.get("실패", 0),
        "total_candidate_count": candidates,
        "total_expected_saving": round(savings, 4),
        "average_vhs": _average([(item.get("vhs") or {}).get("average_vhs") for item in samples]),
        "average_sensitivity_score": _average([(item.get("sensitivity") or {}).get("score") for item in samples]),
        "average_target_pct": _average([(item.get("optimality") or {}).get("target_pct") for item in samples]),
        "average_varo_gap_pct": _average([(item.get("optimality") or {}).get("varo_gap_pct") for item in samples]),
        "total_processing_seconds": round(elapsed, 4),
        "average_sample_seconds": round(elapsed / len(samples), 4) if samples else 0.0,
        "cache_hit_count": sum(bool(item.get("cache_hit")) for item in samples),
    }


def run_integrated_validation(
    settings: Mapping[str, Any],
    progress_callback: ProgressCallback | None = None,
    *,
    force_sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run selected samples sequentially, continuing after every sample failure."""
    normalized = build_integrated_settings(
        scope=str(settings.get("scope") or "standard"),
        sample_ids=settings.get("sample_ids"),
        train_original=bool(settings.get("train_original")),
        train_balanced=bool(settings.get("train_balanced")),
    )
    selected = {str(item) for item in normalized["sample_ids"]}
    force_set = {str(item) for item in (force_sample_ids or [])}
    samples = [sample for sample in DQN_SAMPLES if f"sample_{sample.number:02d}" in selected]
    started_at = _now()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    _notify(progress_callback, stage="샘플 목록 준비 중", current=0, total=len(samples), completed=0)
    for index, sample in enumerate(samples, start=1):
        sample_id = f"sample_{sample.number:02d}"

        def sample_progress(event: dict[str, Any]) -> None:
            counts = Counter(str(item.get("overall_status") or "실패") for item in results)
            _notify(
                progress_callback,
                stage=event.get("stage"),
                sample_id=sample_id,
                sample_label=f"샘플 {sample.number:02d}",
                current=index,
                total=len(samples),
                completed=len(results),
                normal=counts.get("정상", 0),
                review=counts.get("확인 필요", 0),
                failure=counts.get("실패", 0),
                elapsed_seconds=round(time.perf_counter() - started, 2),
            )

        result = run_sample_validation(
            sample,
            normalized,
            sample_progress,
            force=sample_id in force_set,
        )
        results.append(result)
        sample_progress({"stage": "결과 정리 중"})
    elapsed = time.perf_counter() - started
    return _json_safe({
        "version": INTEGRATED_VALIDATION_VERSION,
        "algorithm_version": algorithm_version(),
        "settings": normalized,
        "started_at": started_at,
        "completed_at": _now(),
        "samples": results,
        "summary": _integrated_summary(results, elapsed),
    })


SUMMARY_COLUMNS = [
    "sample_id", "sample_label", "store_dc", "candidate_count", "total_expected_saving",
    "average_vhs", "vhs_greedy_top1_match", "dqn_original_status", "dqn_balanced_status",
    "sensitivity_score", "varo_gap_pct", "target_pct", "pareto_status", "overall_status",
    "processing_seconds",
]


def integrated_summary_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result.get("samples") or []:
        metadata = item.get("metadata") or {}
        rows.append({
            "sample_id": item.get("sample_id"),
            "sample_label": item.get("sample_label"),
            "store_dc": f"{metadata.get('store_count', 0)} / {metadata.get('dc_count', 0)}",
            "candidate_count": metadata.get("candidate_count", 0),
            "total_expected_saving": (item.get("vhs") or {}).get("total_expected_saving"),
            "average_vhs": (item.get("vhs") or {}).get("average_vhs"),
            "vhs_greedy_top1_match": (item.get("greedy") or {}).get("vhs_top1_match"),
            "dqn_original_status": ((item.get("dqn") or {}).get("original") or {}).get("training_status"),
            "dqn_balanced_status": ((item.get("dqn") or {}).get("balanced") or {}).get("training_status"),
            "sensitivity_score": (item.get("sensitivity") or {}).get("score"),
            "varo_gap_pct": (item.get("optimality") or {}).get("varo_gap_pct"),
            "target_pct": (item.get("optimality") or {}).get("target_pct"),
            "pareto_status": (item.get("pareto") or {}).get("status"),
            "overall_status": item.get("overall_status"),
            "processing_seconds": item.get("processing_seconds"),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _section_frame(result: Mapping[str, Any], sections: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in result.get("samples") or []:
        row: dict[str, Any] = {"sample_id": item.get("sample_id"), "sample_label": item.get("sample_label")}
        for section_name in sections:
            section = item.get(section_name) or {}
            if section_name == "dqn":
                for variant in ("original", "balanced"):
                    for key, value in (section.get(variant) or {}).items():
                        row[f"{variant}_{key}"] = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value
                row["numeric_features_preserved"] = section.get("numeric_features_preserved")
                row["improved_vs_original"] = section.get("improved_vs_original")
            else:
                for key, value in section.items():
                    row[f"{section_name}_{key}"] = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value
        rows.append(row)
    return pd.DataFrame(rows)


def integrated_detail_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return _section_frame(result, ("metadata", "vhs", "greedy", "pareto", "dqn", "sensitivity", "optimality"))


def dqn_comparison_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return _section_frame(result, ("dqn",))


def sensitivity_result_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return _section_frame(result, ("sensitivity",))


def optimality_result_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return _section_frame(result, ("optimality",))


def vhs_greedy_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return _section_frame(result, ("vhs", "greedy"))


def pareto_sensitivity_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return _section_frame(result, ("pareto", "sensitivity"))


def error_result_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result.get("samples") or []:
        optimality = item.get("optimality") or {}
        if item.get("overall_status") == "실패" or item.get("excluded_calculations") or optimality.get("unapplied_constraint_count"):
            rows.append({
                "sample_id": item.get("sample_id"),
                "sample_label": item.get("sample_label"),
                "overall_status": item.get("overall_status"),
                "error_stage": item.get("error_stage"),
                "error_message": item.get("error_message"),
                "excluded_calculations": ", ".join(item.get("excluded_calculations") or []),
                "unapplied_constraint_count": optimality.get("unapplied_constraint_count", 0),
                "status_reasons": ", ".join(item.get("status_reasons") or []),
            })
    return pd.DataFrame(rows, columns=[
        "sample_id", "sample_label", "overall_status", "error_stage", "error_message",
        "excluded_calculations", "unapplied_constraint_count", "status_reasons",
    ])


def metadata_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    rows = [
        {"key": "integrated_version", "value": result.get("version")},
        {"key": "algorithm_version", "value": result.get("algorithm_version")},
        {"key": "started_at", "value": result.get("started_at")},
        {"key": "completed_at", "value": result.get("completed_at")},
    ]
    for key, value in (result.get("settings") or {}).items():
        rows.append({"key": f"settings.{key}", "value": json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value})
    for key, value in (result.get("summary") or {}).items():
        rows.append({"key": f"summary.{key}", "value": value})
    return pd.DataFrame(rows)


def integrated_excel_bytes(result: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    sheets = {
        "종합 요약": integrated_summary_frame(result),
        "VHS_Greedy": vhs_greedy_frame(result),
        "DQN_Comparison": dqn_comparison_frame(result),
        "Pareto_Sensitivity": pareto_sensitivity_frame(result),
        "Optimality_Gap": optimality_result_frame(result),
        "Errors": error_result_frame(result),
        "Metadata": metadata_frame(result),
    }
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()


def integrated_json_bytes(result: Mapping[str, Any]) -> bytes:
    return json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
