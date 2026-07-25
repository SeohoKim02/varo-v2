"""Batch DQN training over the user's DQN sample workbooks.

Runs only on an explicit user action from the 분석 및 검증 → DQN 학습 tab. Each
sample is loaded read-only, pushed through the normal V2 pipeline to build the
same recommendation candidates the app would show, and trained with the same
``train_dqn`` used for the current data. Signatures use the file-content hash so
a batch result matches the signature the app assigns when that sample is loaded
via 데이터 관리. Without PyTorch this returns 실행 환경 필요 rows immediately —
no fake results and no files are written.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from services import dqn_service
from services.dqn_service import (
    ENV_REQUIRED_STATUS,
    can_apply_dqn_to_current_data,
    dqn_display_status,
    get_torch_status,
    train_dqn,
)
from services.sample_catalog import DqnSampleInfo


def _row(info: DqnSampleInfo, status: str, message: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "샘플": f"DQN 샘플 {info.sample_id}",
        "파일명": info.file_name,
        "점포/DC": f"{info.store_count}/{info.dc_count}",
        "상태": status,
        "메시지": message,
        "후보 수": info.recommendation_count,
        "결과 파일": "",
    }
    if extra:
        row.update(extra)
    return row


def train_dqn_on_sample(
    info: DqnSampleInfo,
    episodes: int = 300,
    learning_rate: float = 0.001,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Train on one catalog sample. Returns (summary_row, result_dict|None)."""
    from services.analysis_pipeline import build_v2_state
    from services.data_application import source_signature
    from services.data_loader import DataLoadError, load_excel_data

    path = Path(info.file_path)
    if not path.exists():
        return _row(info, "파일 없음", "샘플 파일을 찾을 수 없습니다."), None
    try:
        data = load_excel_data(path)
        state = build_v2_state(data)
    except DataLoadError as exc:
        return _row(info, "로드 실패", str(exc)), None
    validation = state.get("validation")
    if getattr(validation, "has_errors", False):
        return _row(info, "검증 오류", "검증 오류가 있어 학습에서 제외했습니다."), None
    recommendations = state.get("recommendations") or []
    signature = source_signature(path, info.file_name)
    result = train_dqn(
        recommendations,
        data_signature=signature,
        episodes=episodes,
        learning_rate=learning_rate,
        sample_id=info.sample_id,
        sample_name=info.file_name,
        store_count=info.store_count,
        dc_count=info.dc_count,
    )
    payload = result.to_dict()
    result_file = Path(payload.get("result_path") or "").name if payload.get("result_path") else ""
    row = _row(info, payload.get("status", "-"), payload.get("message", ""), {
        "후보 수": payload.get("candidate_count", len(recommendations)),
        "결과 파일": result_file,
    })
    return row, payload


def train_dqn_sample_batch(
    samples: Sequence[DqnSampleInfo],
    episodes: int = 300,
    learning_rate: float = 0.001,
) -> list[dict[str, Any]]:
    """Sequentially train every original sample; short-circuits when PyTorch is absent."""
    torch_ok, torch_message = get_torch_status()
    if not torch_ok:
        return [_row(info, ENV_REQUIRED_STATUS, torch_message) for info in samples]
    rows: list[dict[str, Any]] = []
    for info in samples:
        row, _ = train_dqn_on_sample(info, episodes=episodes, learning_rate=learning_rate)
        rows.append(row)
    return rows


def train_dqn_on_balanced_sample(
    info: DqnSampleInfo,
    episodes: int = 300,
    learning_rate: float = 0.001,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Generate a balanced derivative of one sample, then train on it (variant=balanced)."""
    from services.dqn_balanced import generate_balanced_sample, load_balanced_payload
    from services.dqn_service import data_signature_from_recommendations

    generated = generate_balanced_sample(info)
    if not generated.get("ok"):
        return _row(info, generated.get("status", "-"), generated.get("message", "")), None
    payload = load_balanced_payload(generated["path"]) or {}
    recommendations = payload.get("recommendations") or []
    if len(recommendations) < 3:
        return _row(info, "학습 부족", "균형형 후보 수가 너무 적습니다."), None
    signature = data_signature_from_recommendations(recommendations)
    result = train_dqn(
        recommendations,
        data_signature=signature,
        episodes=episodes,
        learning_rate=learning_rate,
        sample_id=info.sample_id,
        sample_name=payload.get("sample_name") or info.file_name,
        store_count=info.store_count,
        dc_count=info.dc_count,
        variant="balanced",
    )
    result_payload = result.to_dict()
    row = _row(info, result_payload.get("status", "-"), result_payload.get("message", ""), {
        "후보 수": result_payload.get("candidate_count", len(recommendations)),
        "결과 파일": Path(result_payload.get("result_path") or "").name if result_payload.get("result_path") else "",
    })
    return row, result_payload


def train_dqn_balanced_batch(
    samples: Sequence[DqnSampleInfo],
    episodes: int = 300,
    learning_rate: float = 0.001,
) -> list[dict[str, Any]]:
    """Generate + train balanced derivatives for every sample; guards PyTorch."""
    torch_ok, torch_message = get_torch_status()
    if not torch_ok:
        return [_row(info, ENV_REQUIRED_STATUS, torch_message) for info in samples]
    rows: list[dict[str, Any]] = []
    for info in samples:
        row, _ = train_dqn_on_balanced_sample(info, episodes=episodes, learning_rate=learning_rate)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Original vs balanced verification report
# --------------------------------------------------------------------------- #
COMPARISON_EPISODES = 80  # modest, verification-only run (real training, short)


def _report_entry(info: DqnSampleInfo, variant: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Structured per-sample training record (real loss/reward evidence)."""
    loss_history = payload.get("loss_history") or []
    diagnostics = payload.get("diagnostics") or {}
    status = payload.get("status", "-")
    return {
        "sample_id": info.sample_id,
        "sample_name": payload.get("sample_name") or info.file_name,
        "variant": variant,
        "store_count": info.store_count,
        "dc_count": info.dc_count,
        "candidate_count": payload.get("candidate_count"),
        "action_distribution": diagnostics.get("target_action_distribution") or {},
        "prediction_distribution": payload.get("action_distribution") or {},
        "initial_loss": round(float(loss_history[0]), 6) if loss_history else None,
        "final_loss": round(float(loss_history[-1]), 6) if loss_history else None,
        "reward_summary": payload.get("reward_summary") or {},
        "stability_status": payload.get("stability_status") or status,
        "dqn_status": status,
        "dqn_reflection_available": bool(can_apply_dqn_to_current_data(payload, payload.get("data_signature"))),
        "data_signature_match": bool(payload.get("data_signature")),
        "latest_result_path": payload.get("result_path"),
        "model_path": payload.get("model_path"),
    }


def compare_samples(
    samples: Sequence[DqnSampleInfo],
    episodes: int = COMPARISON_EPISODES,
    learning_rate: float = 0.001,
    on_progress: Callable[[int, int, str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Train original + balanced for each sample and return structured records.

    PyTorch-gated: without torch every entry is a 실행 환경 필요 stub and no files
    are written. Original samples typically land on 검토 필요 (target-label skew);
    their balanced derivatives typically reach 정상 — that contrast is the report.
    """
    torch_ok, torch_message = get_torch_status()
    entries: list[dict[str, Any]] = []
    total = len(samples) * 2
    step = 0
    variants = (("original", train_dqn_on_sample), ("balanced", train_dqn_on_balanced_sample))
    for info in samples:
        for variant, trainer in variants:
            step += 1
            if on_progress is not None:
                on_progress(step, total, info.sample_id, variant)
            if not torch_ok:
                entries.append({
                    "sample_id": info.sample_id, "sample_name": info.file_name, "variant": variant,
                    "store_count": info.store_count, "dc_count": info.dc_count,
                    "candidate_count": info.recommendation_count, "dqn_status": ENV_REQUIRED_STATUS,
                    "stability_status": ENV_REQUIRED_STATUS, "dqn_reflection_available": False,
                })
                continue
            _, payload = trainer(info, episodes=episodes, learning_rate=learning_rate)
            if payload:
                entries.append(_report_entry(info, variant, payload))
            else:
                entries.append({
                    "sample_id": info.sample_id, "sample_name": info.file_name, "variant": variant,
                    "dqn_status": "학습 부족", "stability_status": "학습 부족",
                    "dqn_reflection_available": False,
                })
    return entries


def comparison_display_rows(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact, softened rows for the on-screen 원본 vs 균형형 table."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        target = entry.get("action_distribution") or {}
        pred = entry.get("prediction_distribution") or {}
        rows.append({
            "샘플": entry.get("sample_id"),
            "구분": "원본" if entry.get("variant") == "original" else "균형형",
            "후보 수": entry.get("candidate_count"),
            "target 종류": len(target),
            "예측 종류": len(pred),
            "loss 시작→끝": (
                f"{entry.get('initial_loss')}→{entry.get('final_loss')}"
                if entry.get("initial_loss") is not None else "-"
            ),
            "상태": dqn_display_status(entry.get("dqn_status")),
            "VHS 반영 가능": "가능" if entry.get("dqn_reflection_available") else "참고만",
        })
    return rows


def save_comparison_report(entries: Sequence[Mapping[str, Any]]) -> str:
    """Persist the comparison report under outputs/dqn/ (never touches samples)."""
    output_dir = dqn_service.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"comparison_report_{timestamp}.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": COMPARISON_EPISODES,
        "sample_count": len({entry.get("sample_id") for entry in entries}),
        "entries": list(entries),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path)
