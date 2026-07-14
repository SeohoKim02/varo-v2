"""Read-only V2 sample catalog for DQN training flows."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from services.analysis_pipeline import run_analysis_pipeline
from services.data_loader import load_excel_data
from services.dqn_service import ACTION_LABELS, data_signature_from_recommendations
from services.sample_catalog import SAMPLE_WORKBOOKS, SampleWorkbook, sample_path


@dataclass(frozen=True)
class DqnSample:
    number: int
    workbook: SampleWorkbook
    mode: str
    source_path: Path | None = None

    @property
    def label(self) -> str:
        return f"DQN 샘플 {self.number:02d}"

    @property
    def mode_label(self) -> str:
        return "균형형" if self.mode == "balanced" else "원본"

    @property
    def source_label(self) -> str:
        return "DQN 원본 샘플" if self.source_path else "V2 내부 검수 샘플"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SAMPLES_DIR = PROJECT_ROOT.parent / "Varo_DQN_training_samples_10pack"
SAMPLE_DIR_CANDIDATES = (
    EXTERNAL_SAMPLES_DIR,
    PROJECT_ROOT / "dqn_samples",
    PROJECT_ROOT / "data" / "dqn_samples",
    PROJECT_ROOT / "Varo_DQN_training_samples_10pack",
)
BALANCED_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dqn_balanced_samples"
BALANCE_POLICY = "feature_rank_round_robin_v1"


def _sample_number(path: Path) -> int:
    match = re.search(r"sample[_ -]?(\d{1,2})", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 999


def _counts_from_name(path: Path) -> tuple[int, int]:
    match = re.search(r"(\d+)stores?[_ -]?(\d+)dcs?", path.stem, flags=re.IGNORECASE)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def discover_dqn_samples_dir() -> Path | None:
    """Return the first complete read-only 01~10 workbook directory."""
    for candidate in SAMPLE_DIR_CANDIDATES:
        if not candidate.is_dir():
            continue
        numbers = {_sample_number(path) for path in candidate.glob("*.xlsx")}
        if set(range(1, 11)).issubset(numbers):
            return candidate.resolve()
    return None


def _original_samples() -> tuple[DqnSample, ...]:
    """Discover approved workbook folders only; archives are never inspected."""
    root = discover_dqn_samples_dir()
    if root is None:
        return ()
    by_number: dict[int, Path] = {}
    for path in sorted(root.glob("*.xlsx"), key=lambda item: (_sample_number(item), item.name.lower())):
        number = _sample_number(path)
        if 1 <= number <= 10:
            by_number.setdefault(number, path)
    if set(by_number) != set(range(1, 11)):
        return ()
    samples: list[DqnSample] = []
    for index in range(1, 11):
        path = by_number[index]
        store_count, dc_count = _counts_from_name(path)
        workbook = SampleWorkbook(
            f"dqn_sample_{index:02d}",
            f"{store_count or '-'}점포 {dc_count or '-'}DC",
            path.name,
            store_count,
            dc_count,
        )
        samples.append(DqnSample(index, workbook, "original", path.resolve()))
    return tuple(samples)


def _fallback_samples() -> tuple[DqnSample, ...]:
    # Put the dual-DC workbook last so sample 10 always exercises two DCs.
    ordered = tuple(item for item in SAMPLE_WORKBOOKS if item.dc_count == 1) + tuple(
        item for item in SAMPLE_WORKBOOKS if item.dc_count >= 2
    )
    pairs = [(workbook, mode) for workbook in ordered for mode in ("original", "balanced")]
    return tuple(DqnSample(index, workbook, mode) for index, (workbook, mode) in enumerate(pairs, start=1))


DQN_SAMPLES = _original_samples() or _fallback_samples()


def dqn_sample_options() -> dict[str, DqnSample]:
    return {sample.label: sample for sample in DQN_SAMPLES}


def dqn_sample_path(sample: DqnSample, base_dir: Path | None = None) -> Path:
    if sample.source_path is not None:
        path = sample.source_path.resolve()
        allowed_roots = {candidate.resolve() for candidate in SAMPLE_DIR_CANDIDATES if candidate.is_dir()}
        if path.parent not in allowed_roots or path.suffix.lower() != ".xlsx":
            raise ValueError("DQN 샘플 경로가 허용된 원본 폴더를 벗어났습니다.")
        return path
    return sample_path(sample.workbook, base_dir)


def balanced_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return a feature-ranked, label-only balanced copy.

    Numeric and route fields remain byte-for-byte equivalent as Python values.
    Only ``target_action`` is added for DQN supervision; the operational
    ``varo_action`` remains untouched.
    """
    rows = [dict(recommendation) for recommendation in recommendations or []]

    def number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    ordered_indices = sorted(
        range(len(rows)),
        key=lambda index: (
            -number(rows[index].get("expected_saving")),
            -number(rows[index].get("disposal_risk_score")),
            -number(rows[index].get("demand_fit_score")),
            -number(rows[index].get("feasibility_score")),
            number(rows[index].get("estimated_cost") or rows[index].get("move_cost")),
            str(rows[index].get("route_id") or index),
        ),
    )
    for rank, index in enumerate(ordered_indices):
        rows[index]["target_action"] = ACTION_LABELS[(rank + offset) % len(ACTION_LABELS)]
    return rows


def prepare_dqn_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    mode: str,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if mode == "balanced":
        return balanced_recommendations(recommendations, offset=offset)
    return [dict(item) for item in recommendations or []]


def load_sample_recommendations(sample: DqnSample, base_dir: Path | None = None) -> list[dict[str, Any]]:
    data = load_excel_data(dqn_sample_path(sample, base_dir))
    result = run_analysis_pipeline(data)
    return [dict(item) for item in result.recommendations]


def build_dqn_training_sets(
    base_dir: Path | None = None,
    mode: str = "original",
) -> list[dict[str, Any]]:
    """Load the ten read-only catalog entries as explicit DQN training sets."""
    loaded: dict[str, list[dict[str, Any]]] = {}
    training_sets: list[dict[str, Any]] = []
    for sample in DQN_SAMPLES:
        filename = sample.workbook.filename
        if filename not in loaded:
            loaded[filename] = load_sample_recommendations(sample, base_dir)
        original = loaded[filename]
        selected_mode = "balanced" if mode == "balanced" else "original"
        recommendations = prepare_dqn_recommendations(original, selected_mode, offset=sample.number - 1)
        training_sets.append({
            "label": sample.label,
            "mode": selected_mode,
            "mode_label": "균형형" if selected_mode == "balanced" else "원본",
            "filename": filename,
            "sample_name": sample.workbook.filename,
            "sample_id": f"sample_{sample.number:02d}",
            "store_count": sample.workbook.store_count,
            "dc_count": sample.workbook.dc_count,
            "recommendations": recommendations,
            "data_signature": data_signature_from_recommendations(original),
        })
    return training_sets


def diagnose_dqn_training_sets(training_sets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return label-skew diagnostics without starting PyTorch training."""
    rows: list[dict[str, Any]] = []
    for item in training_sets:
        actions = [
            str(row.get("target_action") or row.get("varo_action") or "보류")
            for row in item.get("recommendations") or []
        ]
        distribution = Counter(actions)
        total = sum(distribution.values())
        dominance = max(distribution.values()) / total if total and distribution else 0.0
        status = "검토 필요" if len(distribution) < 2 or dominance >= 0.90 else "비교 가능"
        rows.append({
            "sample_id": item.get("sample_id"),
            "sample_name": item.get("sample_name") or item.get("filename"),
            "variant": item.get("mode") or "original",
            "candidate_count": total,
            "target_type_count": len(distribution),
            "dominant_ratio": round(dominance, 4),
            "status": status,
            "reason": "데이터 라벨 편향" if status == "검토 필요" else "라벨 분포 비교 가능",
            "target_distribution": dict(distribution),
        })
    return rows


def save_balanced_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    sample_id: str,
    store_count: int,
    dc_count: int,
    derived_from: str | None = None,
    balance_policy: str = BALANCE_POLICY,
) -> Path:
    """Persist a generated balanced copy without changing any source workbook."""
    BALANCED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_sample = re.sub(r"[^0-9A-Za-z_-]+", "_", str(sample_id or "current")).strip("_") or "current"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = BALANCED_OUTPUT_DIR / (
        f"dqn_balanced_{safe_sample}_{int(store_count)}stores_{int(dc_count)}dc_{stamp}.xlsx"
    )
    generated_at = datetime.now().isoformat(timespec="seconds")
    frame = pd.DataFrame([dict(row) for row in recommendations or []])
    metadata = balanced_sample_metadata(
        sample_id=sample_id,
        derived_from=derived_from,
        balance_policy=balance_policy,
        generated_at=generated_at,
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="recommendations", index=False)
        pd.DataFrame(
            [{"key": key, "value": value} for key, value in metadata.items()]
        ).to_excel(writer, sheet_name="metadata", index=False)
    return path


def balanced_sample_metadata(
    sample_id: str,
    derived_from: str | None = None,
    balance_policy: str = BALANCE_POLICY,
    generated_at: str | None = None,
) -> dict[str, str]:
    return {
        "derived_from": str(derived_from or "current_recommendations"),
        "original_sample_id": str(sample_id or "current"),
        "balance_policy": str(balance_policy),
        "generated_at": str(generated_at or datetime.now().isoformat(timespec="seconds")),
    }


def dqn_sample_table_rows() -> list[dict[str, Any]]:
    return [
        {
            "샘플": sample.label,
            "학습 세트": sample.mode_label,
            "데이터": sample.workbook.label,
            "파일": sample.workbook.filename,
            "점포": sample.workbook.store_count or "-",
            "DC": sample.workbook.dc_count or "-",
            "출처": sample.source_label,
        }
        for sample in DQN_SAMPLES
    ]
