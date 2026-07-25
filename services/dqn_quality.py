"""DQN training-DATA quality diagnosis (not model performance).

Reports whether a sample's target/action labels are balanced enough for DQN
training to reach a 정상 state, or whether they are skewed (검토 필요 / 불안정) or
too small (학습 부족). Read-only: never trains, never writes files, never touches
the source workbooks. This is a *data* diagnosis; the model's own stability is
判定 separately during training by ``dqn_service.evaluate_dqn_stability``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.dqn_service import normalize_action

NORMAL = "정상"
REVIEW = "검토 필요"
UNSTABLE = "불안정"
INSUFFICIENT = "학습 부족"

# Softer on-screen labels for the raw diagnosis statuses above (display only).
QUALITY_DISPLAY_LABELS = {
    NORMAL: "균형 양호",
    REVIEW: "데이터 확인 필요",
    UNSTABLE: "데이터 편향 큼",
    INSUFFICIENT: "후보 수 부족",
}


def quality_display_status(status: Any) -> str:
    text = str(status or "").strip()
    return QUALITY_DISPLAY_LABELS.get(text, text or "-")

MIN_CANDIDATES = 3
REVIEW_RATIO = 0.90
UNSTABLE_RATIO = 0.97


def target_action(row: Mapping[str, Any]) -> str:
    """The DQN training target label for a candidate (matches dqn_service)."""
    source = (
        row.get("varo_action")
        or row.get("greedy_strategy")
        or row.get("greedy_action")
        or row.get("target_action")
    )
    return normalize_action(source, route_type=row.get("route_type"))


def diagnose_actions(
    recommendations: Sequence[Mapping[str, Any]],
    sample_id: str | None = None,
    sample_name: str | None = None,
    store_count: int | None = None,
    dc_count: int | None = None,
) -> dict[str, Any]:
    """Pure diagnosis of a candidate set's target-action distribution."""
    recs = list(recommendations or [])
    count = len(recs)
    actions = [target_action(row) for row in recs]
    distribution = dict(Counter(actions))
    kinds = len(distribution)
    max_ratio = (max(distribution.values()) / count) if count else 0.0

    if count < MIN_CANDIDATES:
        status, reason = INSUFFICIENT, f"후보 수가 {count}개로 너무 적어 학습이 어렵습니다."
    elif kinds <= 1:
        status, reason = UNSTABLE, "target action이 한 종류뿐이라 학습이 불안정합니다."
    elif max_ratio >= UNSTABLE_RATIO:
        status, reason = UNSTABLE, f"한 action이 {max_ratio * 100:.0f}%로 극단적으로 쏠려 있습니다."
    elif max_ratio >= REVIEW_RATIO:
        status, reason = REVIEW, f"한 action이 {max_ratio * 100:.0f}%로 쏠려 있어 검토가 필요합니다."
    else:
        status, reason = NORMAL, "action 분포가 비교적 고릅니다."

    return {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "store_count": store_count,
        "dc_count": dc_count,
        "candidate_count": count,
        "action_kinds": kinds,
        "max_action_ratio": round(max_ratio, 4),
        "action_distribution": distribution,
        "status": status,
        "reason": reason,
        "basis": "학습 데이터 품질 진단 (DQN 모델 성능 아님)",
    }


# path -> ((mtime, size), diagnosis). Persists across Streamlit reruns in one
# process; a sample whose mtime or size changes is re-diagnosed. Never writes files.
_DIAG_CACHE: dict[str, tuple[tuple[float, int], dict[str, Any]]] = {}


def diagnosis_cache_key(info: Any) -> tuple[str, float, int]:
    """Stable cache key from sample id + file path + mtime + size."""
    try:
        stat = Path(info.file_path).stat()
        return (str(info.sample_id), stat.st_mtime, int(stat.st_size))
    except OSError:
        return (str(info.sample_id), 0.0, 0)


def diagnosis_progress_label(index: int, total: int, sample_name: str, status: Any = None) -> str:
    """Short progress line for the sequential-diagnosis loop (no Streamlit needed)."""
    label = f"{index}/{total} · {sample_name}"
    if status:
        label += f" · {quality_display_status(status)}"
    return label


def diagnose_sample(info: Any) -> dict[str, Any]:
    """Load one catalog sample read-only and diagnose its target labels.

    Cached by file mtime/size so re-runs of the sequential diagnosis are fast; the
    cache invalidates automatically when the underlying file changes.
    """
    from services.analysis_pipeline import build_v2_state
    from services.data_loader import DataLoadError, load_excel_data

    try:
        stat = Path(info.file_path).stat()
        signature = (stat.st_mtime, int(stat.st_size))
    except OSError:
        signature = None
    if signature is not None:
        cached = _DIAG_CACHE.get(str(info.file_path))
        if cached is not None and cached[0] == signature:
            return cached[1]

    base = {
        "sample_id": info.sample_id,
        "sample_name": info.file_name,
        "store_count": info.store_count,
        "dc_count": info.dc_count,
        "candidate_count": 0,
        "action_kinds": 0,
        "max_action_ratio": 0.0,
        "action_distribution": {},
        "basis": "학습 데이터 품질 진단 (DQN 모델 성능 아님)",
    }
    path = Path(info.file_path)
    if not path.exists():
        result = {**base, "status": "파일 없음", "reason": "샘플 파일을 찾을 수 없습니다."}
    else:
        try:
            state = build_v2_state(load_excel_data(path))
            recs = state.get("recommendations") or []
            result = diagnose_actions(recs, info.sample_id, info.file_name, info.store_count, info.dc_count)
        except DataLoadError as exc:
            result = {**base, "status": "로드 실패", "reason": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            result = {**base, "status": "로드 실패", "reason": f"{type(exc).__name__}"}
    if signature is not None:
        _DIAG_CACHE[str(info.file_path)] = (signature, result)
    return result


def diagnose_all(samples: Sequence[Any]) -> list[dict[str, Any]]:
    return [diagnose_sample(sample) for sample in samples]


def run_sequential_diagnosis(samples: Sequence[Any], on_progress=None) -> list[dict[str, Any]]:
    """Diagnose samples one by one, reporting progress via an optional callback.

    ``on_progress(index, total, sample_name, status)`` is called before each
    sample (status=None) and after (status=raw diagnosis status). Pure logic — the
    caller supplies the Streamlit progress/status widgets — so it is unit-testable
    and cannot raise on a missing UI name.
    """
    total = len(samples)
    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if on_progress is not None:
            on_progress(index, total, getattr(sample, "file_name", ""), None)
        result = diagnose_sample(sample)
        results.append(result)
        if on_progress is not None:
            on_progress(index, total, getattr(sample, "file_name", ""), result.get("status"))
    return results


def diagnosis_rows(diagnoses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten diagnoses into display rows (Korean columns) for a Streamlit table."""
    rows: list[dict[str, Any]] = []
    for item in diagnoses:
        distribution = item.get("action_distribution") or {}
        top = max(distribution.items(), key=lambda kv: kv[1])[0] if distribution else "-"
        rows.append({
            "번호": item.get("sample_id"),
            "점포/DC": f"{item.get('store_count')}/{item.get('dc_count')}",
            "후보 수": item.get("candidate_count"),
            "action 종류": item.get("action_kinds"),
            "최다 action": top,
            "최다 비율": f"{float(item.get('max_action_ratio') or 0) * 100:.0f}%",
            "품질 상태": quality_display_status(item.get("status")),
            "사유": item.get("reason"),
        })
    return rows
