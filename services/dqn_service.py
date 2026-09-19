"""V2-scoped DQN training, inference, and comparison helpers.

This module only uses the current Varo V2 recommendation candidates. It never
reads historical DQN artifacts from the original project.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import warnings
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

from services.local_paths import dqn_output_dir

ACTION_LABELS = (
    "재고 이동",
    "DC 경유 이동",
    "직접 이동",
    "할인",
    "긴급 할인",
    "1+1",
    "폐기",
    "보류",
)

ACTION_ALIASES = {
    "multi_store_transfer": "재고 이동",
    "transfer": "재고 이동",
    "store_transfer": "재고 이동",
    "relocation": "재고 이동",
    "재고 이동": "재고 이동",
    "이동": "재고 이동",
    "dc_transfer": "재고 이동",
    "via_dc": "DC 경유 이동",
    "dc 경유": "DC 경유 이동",
    "DC 경유 이동": "DC 경유 이동",
    "direct_transfer": "재고 이동",
    "direct": "직접 이동",
    "직접 이동": "직접 이동",
    "discount": "할인",
    "discount_sale": "할인",
    "할인": "할인",
    "urgent_discount": "긴급 할인",
    "emergency_discount": "긴급 할인",
    "긴급 할인": "긴급 할인",
    "one_plus_one": "1+1",
    "plus_one": "1+1",
    "1+1": "1+1",
    "dispose": "폐기",
    "discard": "폐기",
    "waste": "폐기",
    "폐기": "폐기",
    "keep_inventory": "보류",
    "hold": "보류",
    "no_action": "보류",
    "maintain": "보류",
    "보류": "보류",
}

FEATURE_COLUMNS = (
    "expected_saving",
    "savings_score",
    "disposal_risk_score",
    "days_to_expiry",
    "expiry_days",
    "demand_fit_score",
    "inventory_balance_score",
    "distance_km",
    "move_cost",
    "estimated_cost",
    "expected_time_min",
    "travel_time_min",
    "route_cost_score",
    "feasibility_score",
    "promotion_score",
    "vhs_score",
    "greedy_rank",
    "confidence_score",
)

# Other strategies' own outputs are deliberately kept out of the DQN state so the
# agent stays an independent strategy instead of a VHS/Greedy imitator.
STRATEGY_OUTPUT_COLUMNS = ("vhs_score", "greedy_rank", "confidence_score")
DQN_STATE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column not in STRATEGY_OUTPUT_COLUMNS
)

# Columns that carry a real observation time.  When one exists the train/holdout
# split follows it so a later period never trains an earlier decision.
TIME_ORDER_COLUMNS = (
    "snapshot_date", "base_date", "as_of_date", "as_of", "reference_date",
    "created_at", "order_date", "기준일자", "기준일",
)

DISCOUNT_FACTOR = 0.9
REPLAY_CAPACITY = 4096
REPLAY_BATCH_SIZE = 32
TARGET_SYNC_EPISODES = 5
EPSILON_START = 1.0
EPSILON_END = 0.05
MAX_STEPS_PER_EPISODE = 48
MAX_OPTIMIZER_STEPS = 12000
MIN_HOLDOUT_CANDIDATES = 8
HOLDOUT_RATIO = 0.2
ACTION_CONCENTRATION_LIMIT = 0.90

OUTPUT_DIR = dqn_output_dir()
TRAINING_COMPARISON_CSV = OUTPUT_DIR / "dqn_training_comparison.csv"
LATEST_JSON = OUTPUT_DIR / "latest_dqn_result.json"
LATEST_MODEL = OUTPUT_DIR / "latest_dqn_model.pt"
LATEST_BATCH_JSON = OUTPUT_DIR / "latest_dqn_batch.json"
LATEST_COMPARISON_JSON = OUTPUT_DIR / "latest_dqn_comparison.json"
LATEST_RESULT_BY_VARIANT = {
    "original": OUTPUT_DIR / "latest_dqn_result_original.json",
    "balanced": OUTPUT_DIR / "latest_dqn_result_balanced.json",
}
LATEST_MODEL_BY_VARIANT = {
    "original": OUTPUT_DIR / "latest_dqn_model_original.pt",
    "balanced": OUTPUT_DIR / "latest_dqn_model_balanced.pt",
}
VALID_VARIANTS = frozenset(LATEST_RESULT_BY_VARIANT)

NORMAL_STATUS = "정상"
NEEDS_TRAINING_STATUS = "학습 필요"
NEEDS_REVIEW_STATUS = "검토 필요"
INSUFFICIENT_STATUS = "학습 부족"
INACTIVE_STATUS = "비활성"
PAST_RESULT_STATUS = "과거 결과"
ENV_REQUIRED_STATUS = "실행 환경 필요"

APPLICABLE_STATUSES = {NORMAL_STATUS, "연결", "connected", "ok", "ready"}


@dataclass(frozen=True)
class DqnStatus:
    connected: bool = False
    training_enabled: bool = True
    inference_enabled: bool = False
    historical_artifacts_used: bool = False
    message: str = "DQN 학습 필요"
    status: str = NEEDS_TRAINING_STATUS
    reflection_mode: str = "DQN 참고만"


@dataclass
class DqnTrainingResult:
    status: str = NEEDS_TRAINING_STATUS
    message: str = "DQN 학습 필요"
    data_signature: str | None = None
    timestamp: str | None = None
    episodes: int = 0
    learning_rate: float = 0.001
    sample_id: str = "current"
    training_mode: str = "original"
    variant: str = "original"
    seed: int = 17
    final_status: str = NEEDS_TRAINING_STATUS
    stability_status: str = NEEDS_TRAINING_STATUS
    store_count: int = 0
    dc_count: int = 0
    candidate_count: int = 0
    action_distribution: dict[str, int] = field(default_factory=dict)
    prediction_distribution: dict[str, int] = field(default_factory=dict)
    target_distribution: dict[str, int] = field(default_factory=dict)
    reward_history: list[float] = field(default_factory=list)
    loss_history: list[float | None] = field(default_factory=list)
    reward_summary: dict[str, float] = field(default_factory=dict)
    loss_summary: dict[str, float | None] = field(default_factory=dict)
    average_confidence: float | None = None
    reflection_mode: str = "DQN 참고만"
    model_status: str = "not_trained"
    model_path: str | None = None
    result_path: str | None = None
    feature_columns: list[str] = field(default_factory=lambda: list(DQN_STATE_COLUMNS))
    action_labels: list[str] = field(default_factory=lambda: list(ACTION_LABELS))
    dqn_action_by_route: dict[str, str] = field(default_factory=dict)
    dqn_confidence_by_route: dict[str, float] = field(default_factory=dict)
    dqn_reference_by_route: dict[str, float] = field(default_factory=dict)
    q_value_summary_by_route: dict[str, dict[str, float]] = field(default_factory=dict)
    dqn_status_by_route: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    historical_artifacts_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def get_torch_status() -> tuple[bool, str]:
    runtime = get_torch_runtime_info()
    return bool(runtime["available"]), str(runtime["status"])


def get_torch_training_device(torch_module=None) -> str:
    """Select CUDA only when this wheel explicitly supports the installed GPU."""
    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if not bool(torch.cuda.is_available()):
                return "cpu"
            supported = set(torch.cuda.get_arch_list())
            major, minor = torch.cuda.get_device_capability()
        return "cuda" if f"sm_{major}{minor}" in supported else "cpu"
    except Exception:
        return "cpu"


def get_torch_runtime_info() -> dict[str, Any]:
    """Return user-facing runtime details without making PyTorch mandatory."""
    if not is_torch_available():
        return {
            "available": False,
            "status": "DQN 학습 실행 환경 필요",
            "device": "-",
            "version": "-",
            "message": "배포 환경에 PyTorch가 설치되지 않아 현재 학습을 실행할 수 없습니다.",
        }
    try:
        import torch

        device = "GPU" if get_torch_training_device(torch) == "cuda" else "CPU"
        return {
            "available": True,
            "status": "DQN 학습 실행 가능",
            "device": device,
            "version": str(torch.__version__),
            "message": f"{device}에서 버튼을 누른 경우에만 DQN 학습을 실행합니다.",
        }
    except Exception:
        return {
            "available": False,
            "status": "DQN 학습 실행 환경 필요",
            "device": "-",
            "version": "-",
            "message": "배포 환경의 PyTorch를 확인할 수 없어 현재 학습을 실행할 수 없습니다.",
        }


def build_action_mapping() -> dict[str, int]:
    return {label: index for index, label in enumerate(ACTION_LABELS)}


def normalize_action(value: Any, default: str = "재고 이동", route_type: Any = None) -> str:
    if value is None or str(value).strip() == "":
        route = str(route_type or "").upper()
        if route == "VIA_DC":
            return "DC 경유 이동"
        if route == "DIRECT":
            return "직접 이동"
        return default
    text = str(value).strip()
    lowered = text.lower()
    if text in ACTION_ALIASES:
        return ACTION_ALIASES[text]
    if lowered in ACTION_ALIASES:
        return ACTION_ALIASES[lowered]
    for key, label in ACTION_ALIASES.items():
        if key.lower() in lowered:
            if label == "재고 이동":
                return normalize_action(None, default=label, route_type=route_type)
            return label
    if text in ACTION_LABELS:
        return text
    return normalize_action(None, default=default, route_type=route_type)


def data_signature_from_recommendations(recommendations: Sequence[Mapping[str, Any]]) -> str:
    serializable = []
    for row in recommendations or []:
        serializable.append({
            "snapshot_date": row.get("snapshot_date"),
            "route_id": row.get("route_id"),
            "product_id": row.get("product_id"),
            "source_id": row.get("source_id"),
            "target_id": row.get("target_id"),
            "dc_id": row.get("dc_id"),
            "route_type": row.get("route_type"),
            "recommended_qty": row.get("recommended_qty"),
            "expected_saving": row.get("expected_saving"),
        })
    blob = json.dumps(serializable, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _feature_stats(recommendations: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = [_num(row.get(column)) for row in recommendations]
        clean = [value for value in values if value is not None]
        stats[column] = (min(clean), max(clean)) if clean else (0.0, 0.0)
    return stats


def _coerce_feature_stats(
    stats: Mapping[str, Sequence[float]] | None,
    columns: Sequence[str],
) -> dict[str, tuple[float, float]] | None:
    """Validate persisted train-only normalization statistics."""
    if not stats:
        return None
    normalized: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = stats.get(column)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
            return None
        low, high = _num(values[0]), _num(values[1])
        if low is None or high is None or high < low:
            return None
        normalized[column] = (low, high)
    return normalized


def _serializable_feature_stats(
    stats: Mapping[str, Sequence[float]],
    columns: Sequence[str],
) -> dict[str, list[float]]:
    validated = _coerce_feature_stats(stats, columns)
    if validated is None:
        raise ValueError("invalid DQN feature normalization statistics")
    return {column: [float(validated[column][0]), float(validated[column][1])] for column in columns}


def build_state_vectors(
    recommendations: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    feature_stats: Mapping[str, Sequence[float]] | None = None,
) -> list[list[float]]:
    """Map recommendation candidates to normalized DQN state vectors.

    Missing values become 0.5, a neutral midpoint. Route type and cold-chain
    hints are encoded in stable extra dimensions.
    """
    recs = [dict(row) for row in recommendations or []]
    stats = _coerce_feature_stats(feature_stats, feature_columns) or _feature_stats(recs, feature_columns)
    vectors: list[list[float]] = []
    for row in recs:
        vector: list[float] = []
        for column in feature_columns:
            value = _num(row.get(column))
            low, high = stats[column]
            if value is None or high == low:
                vector.append(0.5)
            else:
                vector.append(max(0.0, min(1.0, (value - low) / (high - low))))
        route_type = str(row.get("route_type") or "").upper()
        vector.append(1.0 if route_type == "VIA_DC" else 0.0)
        transport = str(row.get("transport_type") or row.get("transport_label") or "")
        vector.append(1.0 if any(token in transport for token in ("냉장", "냉동", "cold", "freeze")) else 0.0)
        vectors.append(vector)
    return vectors


def _route_ids(recommendations: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = list(recommendations or [])
    base_ids = [str(row.get("route_id") or f"R{index + 1:03d}") for index, row in enumerate(rows)]
    counts = Counter(base_ids)
    dated_ids: list[str] = []
    for index, (row, base_id) in enumerate(zip(rows, base_ids)):
        if counts[base_id] == 1:
            dated_ids.append(base_id)
            continue
        time_value = next(
            (str(row.get(column)).strip() for column in TIME_ORDER_COLUMNS if str(row.get(column) or "").strip()),
            "",
        )
        dated_ids.append(f"{time_value}::{base_id}" if time_value else f"{base_id}::{index + 1}")
    duplicate_counts = Counter(dated_ids)
    occurrences: Counter[str] = Counter()
    unique_ids: list[str] = []
    for value in dated_ids:
        occurrences[value] += 1
        unique_ids.append(
            value if duplicate_counts[value] == 1 else f"{value}::{occurrences[value]}"
        )
    return unique_ids


def _target_actions(recommendations: Sequence[Mapping[str, Any]]) -> list[str]:
    actions: list[str] = []
    for row in recommendations or []:
        source_action = row.get("target_action") or row.get("varo_action") or row.get("greedy_strategy") or row.get("greedy_action")
        actions.append(normalize_action(source_action, route_type=row.get("route_type")))
    return actions


def calculate_rewards(recommendations: Sequence[Mapping[str, Any]]) -> list[float]:
    """Build an independent, bounded reward signal for current V2 candidates."""
    recs = [dict(row) for row in recommendations or []]
    if not recs:
        return []

    def norm_high(name: str, neutral: float = 0.5) -> list[float]:
        values = [_num(row.get(name)) for row in recs]
        clean = [value for value in values if value is not None]
        if not clean or max(clean) == min(clean):
            return [neutral for _ in values]
        low, high = min(clean), max(clean)
        return [neutral if value is None else max(0.0, min(1.0, (value - low) / (high - low))) for value in values]

    def norm_low(name: str, neutral: float = 0.5) -> list[float]:
        return [1.0 - value for value in norm_high(name, 1.0 - neutral)]

    saving = norm_high("expected_saving")
    disposal = norm_high("disposal_risk_score")
    demand = norm_high("demand_fit_score")
    balance = norm_high("inventory_balance_score")
    feasibility = norm_high("feasibility_score", neutral=0.75)
    cost = norm_low("estimated_cost")
    distance = norm_low("distance_km")
    time = norm_low("expected_time_min")
    promotion = norm_high("promotion_score", neutral=0.55)

    rewards = []
    for index, row in enumerate(recs):
        value = (
            saving[index] * 0.24
            + disposal[index] * 0.12
            + demand[index] * 0.13
            + balance[index] * 0.12
            + feasibility[index] * 0.16
            + cost[index] * 0.09
            + distance[index] * 0.05
            + time[index] * 0.04
            + promotion[index] * 0.05
        )
        if feasibility[index] < 0.35:
            value -= 0.18
        route_type = str(row.get("route_type") or "").upper()
        if route_type not in {"DIRECT", "VIA_DC"}:
            value -= 0.08
        rewards.append(round(max(0.0, min(1.0, value)), 6))
    return rewards


def _norm_high(recs: Sequence[Mapping[str, Any]], names: Sequence[str], neutral: float = 0.5) -> list[float]:
    """Scale the first present column to 0..1; missing data stays neutral."""
    values: list[float | None] = []
    for row in recs:
        found: float | None = None
        for name in names:
            found = _num(row.get(name))
            if found is not None:
                break
        values.append(found)
    clean = [value for value in values if value is not None]
    if not clean or max(clean) == min(clean):
        return [neutral for _ in values]
    low, high = min(clean), max(clean)
    return [
        neutral if value is None else max(0.0, min(1.0, (value - low) / (high - low)))
        for value in values
    ]


def _norm_low(recs: Sequence[Mapping[str, Any]], names: Sequence[str], neutral: float = 0.5) -> list[float]:
    return [1.0 - value for value in _norm_high(recs, names, 1.0 - neutral)]


def build_reward_signals(recommendations: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """Derive the operational signals the reward function is allowed to use.

    Every signal comes from a value the current Varo pipeline produces at
    decision time.  No VHS rank, Greedy selection, MILP result, or realized
    future outcome takes part.
    """
    recs = [dict(row) for row in recommendations or []]
    expiry = _norm_high(recs, ("days_to_expiry", "expiry_days"))
    return {
        "saving": _norm_high(recs, ("expected_saving", "savings_score")),
        "cost": _norm_low(recs, ("move_cost", "estimated_cost")),
        "feasibility": _norm_high(recs, ("feasibility_score",), neutral=0.75),
        "demand": _norm_high(recs, ("demand_fit_score",)),
        "balance": _norm_high(recs, ("inventory_balance_score",)),
        "risk": _norm_high(recs, ("disposal_risk_score",)),
        "promotion": _norm_high(recs, ("promotion_score",), neutral=0.55),
        "quantity": _norm_high(recs, ("recommended_qty",)),
        "distance": _norm_low(recs, ("distance_km",)),
        "expiry_pressure": [1.0 - value for value in expiry],
        "route_ok": [
            1.0 if str(row.get("route_type") or "").upper() in {"DIRECT", "VIA_DC"} else 0.0
            for row in recs
        ],
        "is_direct": [1.0 if str(row.get("route_type") or "").upper() == "DIRECT" else 0.0 for row in recs],
        "is_via_dc": [1.0 if str(row.get("route_type") or "").upper() == "VIA_DC" else 0.0 for row in recs],
    }


def action_reward_matrix(recommendations: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    """Reward every (candidate, action) pair against the Varo operating goals.

    The order of priorities mirrors operations: secure executable service
    quantity first, then keep cost down, then capture savings, and always
    punish feasibility violations.  Nothing here reads another strategy's
    decision, so the agent must find its own policy.
    """
    recs = [dict(row) for row in recommendations or []]
    if not recs:
        return []
    signal = build_reward_signals(recs)
    matrix: list[list[float]] = []
    for index in range(len(recs)):
        saving = signal["saving"][index]
        cost = signal["cost"][index]
        feasibility = signal["feasibility"][index]
        demand = signal["demand"][index]
        balance = signal["balance"][index]
        risk = signal["risk"][index]
        promotion = signal["promotion"][index]
        quantity = signal["quantity"][index]
        distance = signal["distance"][index]
        pressure = signal["expiry_pressure"][index]
        service = 0.55 * demand + 0.45 * quantity
        move_expensive = 1.0 - cost

        transfer = (
            0.30 * service + 0.24 * saving + 0.18 * cost
            + 0.16 * feasibility + 0.08 * balance + 0.04 * distance
        )
        if signal["route_ok"][index] < 1.0:
            transfer -= 0.20
        if feasibility < 0.35:
            transfer -= 0.20

        direct = transfer + (0.06 if signal["is_direct"][index] else -0.10)
        via_dc = transfer + (0.06 if signal["is_via_dc"][index] else -0.10)

        discount = (
            0.30 * risk + 0.24 * pressure + 0.20 * (1.0 - feasibility)
            + 0.14 * (1.0 - demand) + 0.12 * move_expensive
        )
        urgent = (
            0.40 * pressure + 0.30 * risk + 0.18 * (1.0 - feasibility) + 0.12 * move_expensive
        )
        if pressure < 0.6:
            urgent -= 0.15
        bundle = (
            0.45 * promotion + 0.22 * risk + 0.18 * quantity + 0.15 * (1.0 - demand)
        )
        dispose = 0.10 + 0.32 * risk * pressure
        if feasibility >= 0.5:
            dispose -= 0.25
        hold = 0.30 + 0.18 * (1.0 - risk) + 0.12 * (1.0 - pressure)
        if feasibility < 0.35:
            hold += 0.15

        row = {
            "재고 이동": transfer,
            "DC 경유 이동": via_dc,
            "직접 이동": direct,
            "할인": discount,
            "긴급 할인": urgent,
            "1+1": bundle,
            "폐기": dispose,
            "보류": hold,
        }
        matrix.append([round(max(0.0, min(1.0, row[label])), 6) for label in ACTION_LABELS])
    return matrix


def reward_optimal_actions(recommendations: Sequence[Mapping[str, Any]]) -> list[str]:
    """Best-response action per candidate under the DQN reward only.

    This is the distribution a learned policy is measured against.  It comes
    from the reward function, never from the Greedy or VHS decision, so a
    uniform heuristic label set cannot make the DQN look unusable.
    """
    matrix = action_reward_matrix(recommendations)
    return [
        ACTION_LABELS[max(range(len(row)), key=lambda index: row[index])]
        for row in matrix
    ]


def _time_order_key(recommendations: Sequence[Mapping[str, Any]]) -> str | None:
    for column in TIME_ORDER_COLUMNS:
        if any(str(row.get(column) or "").strip() for row in recommendations or []):
            return column
    return None


def chronological_split(
    recommendations: Sequence[Mapping[str, Any]],
    holdout_ratio: float = HOLDOUT_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split train/holdout in time order, never by shuffling.

    When the data carries a real observation time the rows are sorted by it
    first; otherwise the supplied candidate order is treated as the arrival
    order.  The holdout is always the later slice, so no future row can reach
    the training window.
    """
    recs = [dict(row) for row in recommendations or []]
    if not recs:
        return [], []
    column = _time_order_key(recs)
    if column:
        recs = sorted(recs, key=lambda row: (str(row.get(column) or ""),))
        ordered_periods = list(dict.fromkeys(str(row.get(column) or "") for row in recs))
        if len(ordered_periods) >= 2 and holdout_ratio > 0.0:
            holdout_period_count = max(1, int(round(len(ordered_periods) * float(holdout_ratio))))
            holdout_period_count = min(holdout_period_count, len(ordered_periods) - 1)
            holdout_periods = set(ordered_periods[-holdout_period_count:])
            train = [row for row in recs if str(row.get(column) or "") not in holdout_periods]
            holdout = [row for row in recs if str(row.get(column) or "") in holdout_periods]
            return train, holdout
    if len(recs) < MIN_HOLDOUT_CANDIDATES or holdout_ratio <= 0.0:
        return recs, []
    holdout_size = max(1, int(round(len(recs) * float(holdout_ratio))))
    cut = len(recs) - holdout_size
    return recs[:cut], recs[cut:]


def action_concentration(actions: Sequence[str]) -> dict[str, Any]:
    """Report how strongly one action dominates real inference output."""
    distribution = Counter(str(action) for action in actions if action)
    total = sum(distribution.values())
    if not total:
        return {
            "total": 0, "dominant_action": None, "dominant_ratio": None,
            "distinct_actions": 0, "concentrated": False,
        }
    dominant_action, dominant_count = distribution.most_common(1)[0]
    ratio = dominant_count / total
    return {
        "total": total,
        "dominant_action": dominant_action,
        "dominant_ratio": round(ratio, 6),
        "distinct_actions": len([value for value in distribution.values() if value > 0]),
        "concentrated": bool(ratio >= ACTION_CONCENTRATION_LIMIT or len(distribution) < 2),
    }


def evaluate_dqn_stability(
    losses: Sequence[float],
    actions: Sequence[str],
    rewards: Sequence[float],
    candidate_count: int | None = None,
    data_signature: str | None = None,
    current_signature: str | None = None,
    target_actions: Sequence[str] | None = None,
) -> tuple[str, str]:
    count = int(candidate_count if candidate_count is not None else len(actions))
    if current_signature and data_signature and current_signature != data_signature:
        return PAST_RESULT_STATUS, "현재 데이터와 다른 DQN 결과입니다."
    if count < 3:
        return INSUFFICIENT_STATUS, "후보 수가 너무 적어 DQN은 참고 상태로만 유지합니다."
    if not actions:
        return NEEDS_TRAINING_STATUS, "DQN 학습 결과가 없습니다."
    if any(not math.isfinite(float(loss)) for loss in losses):
        return NEEDS_REVIEW_STATUS, "loss 값이 안정적이지 않습니다."
    distribution = Counter(actions)
    if len(distribution) < 2:
        return NEEDS_REVIEW_STATUS, "예측 action 종류가 부족합니다."
    if distribution and max(distribution.values()) / max(1, len(actions)) >= 0.90:
        return NEEDS_REVIEW_STATUS, "action 분포가 한쪽으로 치우쳤습니다."
    target_distribution = Counter(target_actions or [])
    if target_distribution and (
        len(target_distribution) < 2
        or max(target_distribution.values()) / max(1, sum(target_distribution.values())) >= 0.90
    ):
        return NEEDS_REVIEW_STATUS, "보상 기준 최적 action 분포가 한쪽으로 치우쳤습니다."
    if any(not math.isfinite(float(reward)) for reward in rewards):
        return NEEDS_REVIEW_STATUS, "reward 값이 안정적이지 않습니다."
    if rewards and max(rewards) == min(rewards):
        return NEEDS_REVIEW_STATUS, "reward 분포가 모두 동일합니다."
    return NORMAL_STATUS, "DQN 정상"


def validate_training_stability(losses: Sequence[float], actions: Sequence[str], rewards: Sequence[float]) -> tuple[str, str]:
    """Backward-compatible stability wrapper used by existing tests."""
    return evaluate_dqn_stability(losses, actions, rewards, candidate_count=len(actions))


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"first": None, "min": None, "max": None, "avg": None, "last": None}
    return {
        "first": round(clean[0], 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
        "avg": round(sum(clean) / len(clean), 6),
        "last": round(clean[-1], 6),
    }


def _now_parts() -> tuple[str, str]:
    now = datetime.now()
    return now.isoformat(timespec="microseconds"), now.strftime("%Y%m%d_%H%M%S_%f")


def _timestamp_slug(value: Any) -> str:
    text = str(value or datetime.now().isoformat(timespec="microseconds"))
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def _artifact_slug(value: Any, fallback: str) -> str:
    slug = "".join(character if character.isalnum() or character in "_-" else "_" for character in str(value or ""))
    return slug.strip("_") or fallback


def _artifact_context(
    sample_id: str, training_mode: str, store_count: int | None, dc_count: int | None,
    episodes: int, learning_rate: float,
) -> str:
    lr_slug = format(float(learning_rate), ".8g").replace(".", "p").replace("-", "m").replace("+", "")
    return (
        f"{_artifact_slug(sample_id, 'current')}_{_artifact_slug(training_mode, 'original')}_"
        f"{int(store_count or 0)}stores_{int(dc_count or 0)}dc_ep{int(episodes)}_lr{lr_slug}"
    )


def _empty_result(
    status: str,
    message: str,
    recommendations: Sequence[Mapping[str, Any]],
    data_signature: str | None = None,
    episodes: int = 0,
    learning_rate: float = 0.001,
    reflection_mode: str = "DQN 참고만",
    sample_id: str = "current",
    training_mode: str = "original",
    store_count: int = 0,
    dc_count: int = 0,
    seed: int = 17,
) -> DqnTrainingResult:
    labels = _target_actions(recommendations)
    rewards = calculate_rewards(recommendations)
    return DqnTrainingResult(
        status=status,
        final_status=status,
        stability_status=status,
        message=message,
        data_signature=data_signature,
        timestamp=datetime.now().isoformat(timespec="microseconds"),
        episodes=episodes,
        learning_rate=learning_rate,
        sample_id=sample_id,
        training_mode=training_mode,
        variant=training_mode if training_mode in VALID_VARIANTS else "original",
        seed=int(seed),
        store_count=store_count,
        dc_count=dc_count,
        candidate_count=len(recommendations or []),
        action_distribution=dict(Counter(labels)),
        prediction_distribution={},
        target_distribution=dict(Counter(labels)),
        reward_history=rewards,
        loss_history=[],
        reward_summary=_summary(rewards),
        loss_summary={},
        reflection_mode=reflection_mode,
        model_status="not_trained",
        diagnostics={"historical_artifacts_used": False},
        historical_artifacts_used=False,
    )


def _model(input_size: int, output_size: int, seed: int = 17):
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    return nn.Sequential(
        nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Linear(32, 24),
        nn.ReLU(),
        nn.Linear(24, output_size),
    )


def _cost_shares(recommendations: Sequence[Mapping[str, Any]]) -> list[float]:
    """Share of the run's transport cost each candidate would consume."""
    costs = [
        (_num(row.get("move_cost")) or _num(row.get("estimated_cost")) or 0.0)
        for row in recommendations
    ]
    total = sum(costs)
    if total <= 0:
        return [1.0 / max(1, len(costs)) for _ in costs]
    return [value / total for value in costs]


def build_training_states(
    recommendations: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str] = DQN_STATE_COLUMNS,
    feature_stats: Mapping[str, Sequence[float]] | None = None,
) -> list[list[float]]:
    """Candidate features for DQN, without the two per-episode dimensions."""
    return build_state_vectors(
        recommendations,
        feature_columns=feature_columns,
        feature_stats=feature_stats,
    )


def _episode_state(base: Sequence[float], progress: float, budget: float) -> list[float]:
    return [*base, round(max(0.0, min(1.0, progress)), 6), round(max(0.0, min(1.0, budget)), 6)]


TRANSFER_ACTIONS = frozenset({"재고 이동", "DC 경유 이동", "직접 이동"})
_TRANSFER_INDEXES = tuple(
    index for index, label in enumerate(ACTION_LABELS) if label in TRANSFER_ACTIONS
)


def valid_action_indexes(candidate: Mapping[str, Any]) -> tuple[int, ...]:
    """Return actions that are executable for one decision-time candidate."""
    valid = set(range(len(ACTION_LABELS)))
    route_type = str(candidate.get("route_type") or "").upper()
    if route_type == "DIRECT":
        valid.discard(ACTION_LABELS.index("DC 경유 이동"))
    elif route_type == "VIA_DC":
        valid.discard(ACTION_LABELS.index("직접 이동"))
    else:
        valid.difference_update(_TRANSFER_INDEXES)

    quantity = _num(candidate.get("recommended_qty"))
    feasibility = _num(candidate.get("feasibility_score"))
    feasibility_ratio = None if feasibility is None else feasibility / 100.0 if feasibility > 1.0 else feasibility
    if (quantity is not None and quantity <= 0.0) or (
        feasibility_ratio is not None and feasibility_ratio < 0.35
    ):
        valid.difference_update(_TRANSFER_INDEXES)
    if not valid:
        return (ACTION_LABELS.index("보류"),)
    return tuple(sorted(valid))


def _action_masks(recommendations: Sequence[Mapping[str, Any]]) -> list[tuple[int, ...]]:
    return [valid_action_indexes(row) for row in recommendations]


def _masked_argmax(values: Sequence[float], valid_indexes: Sequence[int]) -> int:
    allowed = tuple(int(index) for index in valid_indexes) or (ACTION_LABELS.index("보류"),)
    return max(allowed, key=lambda index: float(values[index]))


def state_schema_fingerprint(feature_columns: Sequence[str] = DQN_STATE_COLUMNS) -> str:
    """Identify the exact state layout a saved model expects."""
    blob = json.dumps(
        {
            "feature_columns": list(feature_columns),
            "extra_dimensions": ["route_via_dc", "cold_chain", "progress", "remaining_budget"],
            "action_labels": list(ACTION_LABELS),
            "normalization": "train_minmax_v1",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _greedy_rollout(
    model,
    base_states: Sequence[Sequence[float]],
    cost_shares: Sequence[float],
    action_masks: Sequence[Sequence[int]] | None = None,
):
    """Run the learned policy over the candidate sequence with no exploration."""
    import torch

    total = max(1, len(base_states))
    budget = 1.0
    chosen: list[int] = []
    q_rows: list[list[float]] = []
    with torch.no_grad():
        for index, base in enumerate(base_states):
            vector = _episode_state(base, index / total, budget)
            values = model(torch.tensor([vector], dtype=torch.float32))[0]
            q_values = [float(value) for value in values.tolist()]
            valid = action_masks[index] if action_masks is not None else range(len(q_values))
            action = int(_masked_argmax(q_values, valid))
            chosen.append(action)
            q_rows.append(q_values)
            if action in _TRANSFER_INDEXES:
                budget = max(0.0, budget - float(cost_shares[index]))
    return chosen, q_rows


def _confidence_from_q(q_values: Sequence[float]) -> float:
    """Softmax confidence of the selected action, computed without torch."""
    clean = [value for value in q_values if math.isfinite(value)]
    if not clean:
        return 0.0
    highest = max(clean)
    exponents = [math.exp(min(60.0, value - highest)) for value in clean]
    total = sum(exponents)
    return float(max(exponents) / total) if total > 0 else 0.0


def _policy_evaluation(
    recommendations: Sequence[Mapping[str, Any]],
    actions: Sequence[int],
    reward_matrix: Sequence[Sequence[float]],
    masks: Sequence[Sequence[int]],
    seed: int,
) -> dict[str, Any]:
    """Evaluate a policy on an untouched chronological holdout."""
    recs = [dict(row) for row in recommendations]
    selected_rewards = [float(reward_matrix[index][action]) for index, action in enumerate(actions)]
    random_rng = random.Random(int(seed) + 100_003)
    random_actions = [random_rng.choice(tuple(mask)) for mask in masks]
    random_rewards = [float(reward_matrix[index][action]) for index, action in enumerate(random_actions)]
    optimal_rewards = [max(float(row[index]) for index in masks[position]) for position, row in enumerate(reward_matrix)]
    transfer_rows = [
        recs[index] for index, action in enumerate(actions)
        if ACTION_LABELS[int(action)] in TRANSFER_ACTIONS
    ]
    demand_values = [_num(row.get("demand_fit_score")) for row in transfer_rows]
    clean_demand = [value for value in demand_values if value is not None]
    return {
        "mean_reward": round(sum(selected_rewards) / len(selected_rewards), 6) if selected_rewards else None,
        "median_reward": round(float(median(selected_rewards)), 6) if selected_rewards else None,
        "random_mean_reward": round(sum(random_rewards) / len(random_rewards), 6) if random_rewards else None,
        "reward_optimal_mean": round(sum(optimal_rewards) / len(optimal_rewards), 6) if optimal_rewards else None,
        "action_distribution": dict(Counter(ACTION_LABELS[int(action)] for action in actions)),
        "feasibility_violations": sum(
            1 for index, action in enumerate(actions) if int(action) not in set(masks[index])
        ),
        "selected_transfer_count": len(transfer_rows),
        "throughput": round(sum(max(0.0, _num(row.get("recommended_qty")) or 0.0) for row in transfer_rows), 6),
        "expected_saving": round(sum(_num(row.get("expected_saving")) or 0.0 for row in transfer_rows), 6),
        "transport_cost": round(sum(
            _num(row.get("move_cost")) or _num(row.get("estimated_cost")) or 0.0
            for row in transfer_rows
        ), 6),
        "mean_service_score": round(sum(clean_demand) / len(clean_demand), 6) if clean_demand else None,
    }


def run_dqn_training(
    recommendations: Sequence[Mapping[str, Any]],
    episodes: int,
    learning_rate: float,
    seed: int = 17,
) -> dict[str, Any]:
    """Train a Q network with replay, epsilon-greedy control, and a target net.

    One episode walks the candidate sequence in chronological order while the
    remaining transport budget changes with the actions taken, so each update is
    a real temporal-difference step instead of a fit to another strategy label.
    """
    import torch
    from torch import nn

    train_recs, holdout_recs = chronological_split(recommendations)
    feature_stats = _feature_stats(train_recs, DQN_STATE_COLUMNS)
    base_states = build_training_states(train_recs, feature_stats=feature_stats)
    reward_matrix = action_reward_matrix(train_recs)
    cost_shares = _cost_shares(train_recs)
    action_masks = _action_masks(train_recs)
    action_count = len(ACTION_LABELS)
    input_size = len(base_states[0]) + 2
    device = torch.device(get_torch_training_device(torch))

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    policy = _model(input_size, action_count, seed=seed).to(device)
    target = _model(input_size, action_count, seed=seed).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    initial_parameters = [parameter.detach().clone() for parameter in policy.parameters()]

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    criterion = nn.SmoothL1Loss()
    rng = random.Random(int(seed))
    buffer: deque = deque(maxlen=REPLAY_CAPACITY)

    total_candidates = max(1, len(base_states))
    window = min(total_candidates, MAX_STEPS_PER_EPISODE)
    episode_losses: list[float | None] = []
    episode_rewards: list[float] = []
    exploration_actions: list[int] = []
    optimizer_steps = 0
    diverged = False

    for episode in range(int(episodes)):
        span = max(1, int(episodes) - 1)
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * (episode / span)
        start = (episode * window) % total_candidates if total_candidates > window else 0
        indexes = list(range(start, min(start + window, total_candidates))) or [0]
        budget = 1.0
        reward_sum = 0.0
        step_losses: list[float] = []
        for order, index in enumerate(indexes):
            state = _episode_state(base_states[index], index / total_candidates, budget)
            valid_actions = action_masks[index]
            if rng.random() < epsilon:
                action = rng.choice(valid_actions)
            else:
                with torch.no_grad():
                    values = policy(torch.tensor([state], dtype=torch.float32, device=device))[0]
                action = _masked_argmax(values.tolist(), valid_actions)
            exploration_actions.append(action)
            reward = float(reward_matrix[index][action])
            reward_sum += reward
            next_budget = (
                max(0.0, budget - float(cost_shares[index]))
                if action in _TRANSFER_INDEXES else budget
            )
            done = order == len(indexes) - 1
            next_index = index + 1 if index + 1 < total_candidates else index
            next_state = _episode_state(
                base_states[next_index], next_index / total_candidates, next_budget
            )
            buffer.append((state, action, reward, next_state, done, action_masks[next_index]))
            budget = next_budget

            learn_ready = len(buffer) >= min(REPLAY_BATCH_SIZE, total_candidates)
            if learn_ready and optimizer_steps < MAX_OPTIMIZER_STEPS:
                batch = rng.sample(list(buffer), min(REPLAY_BATCH_SIZE, len(buffer)))
                states_tensor = torch.tensor([item[0] for item in batch], dtype=torch.float32, device=device)
                actions_tensor = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
                rewards_tensor = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
                next_tensor = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=device)
                done_tensor = torch.tensor(
                    [1.0 if item[4] else 0.0 for item in batch], dtype=torch.float32, device=device
                )
                predicted = policy(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_values = target(next_tensor)
                    valid_tensor = torch.zeros_like(next_values, dtype=torch.bool)
                    for row_index, item in enumerate(batch):
                        valid_tensor[row_index, list(item[5])] = True
                    bootstrap = next_values.masked_fill(~valid_tensor, float("-inf")).max(dim=1).values
                expected = rewards_tensor + DISCOUNT_FACTOR * bootstrap * (1.0 - done_tensor)
                loss = criterion(predicted, expected)
                if not torch.isfinite(loss):
                    diverged = True
                    step_losses.append(float("inf"))
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                step_losses.append(float(loss.detach().item()))
        episode_rewards.append(round(reward_sum / max(1, len(indexes)), 6))
        episode_losses.append(
            round(sum(step_losses) / len(step_losses), 8) if step_losses else None
        )
        if diverged:
            break
        if episode % TARGET_SYNC_EPISODES == 0:
            target.load_state_dict(policy.state_dict())

    target.load_state_dict(policy.state_dict())
    policy.eval()
    parameters_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial_parameters, policy.parameters())
    )

    policy_cpu = policy.to("cpu")
    all_base = build_training_states(recommendations, feature_stats=feature_stats)
    chosen, q_rows = _greedy_rollout(
        policy_cpu, all_base, _cost_shares(recommendations), _action_masks(recommendations)
    )
    holdout_reward = None
    holdout_metrics: dict[str, Any] = {}
    if holdout_recs:
        holdout_base = build_training_states(holdout_recs, feature_stats=feature_stats)
        holdout_matrix = action_reward_matrix(holdout_recs)
        holdout_masks = _action_masks(holdout_recs)
        holdout_actions, _ = _greedy_rollout(
            policy_cpu, holdout_base, _cost_shares(holdout_recs), holdout_masks
        )
        holdout_reward = round(
            sum(holdout_matrix[index][action] for index, action in enumerate(holdout_actions))
            / max(1, len(holdout_actions)),
            6,
        )
        holdout_metrics = _policy_evaluation(
            holdout_recs, holdout_actions, holdout_matrix, holdout_masks, seed,
        )
    return {
        "model": policy_cpu,
        "input_size": input_size,
        "episode_losses": episode_losses,
        "episode_rewards": episode_rewards,
        "selected_actions": chosen,
        "q_values": q_rows,
        "parameters_changed": bool(parameters_changed),
        "optimizer_steps": optimizer_steps,
        "diverged": diverged,
        "exploration_distribution": dict(Counter(ACTION_LABELS[index] for index in exploration_actions)),
        "train_candidate_count": len(train_recs),
        "holdout_candidate_count": len(holdout_recs),
        "holdout_mean_reward": holdout_reward,
        "holdout_metrics": holdout_metrics,
        "feature_stats": _serializable_feature_stats(feature_stats, DQN_STATE_COLUMNS),
        "time_order_column": _time_order_key(recommendations),
        "device": str(device),
    }


def train_dqn(
    recommendations: Sequence[Mapping[str, Any]],
    data_signature: str | None = None,
    episodes: int = 300,
    learning_rate: float = 0.001,
    candidate_count: int | None = None,
    reflection_mode: str = "DQN 참고만",
    sample_id: str = "current",
    training_mode: str = "original",
    store_count: int | None = None,
    dc_count: int | None = None,
    seed: int = 17,
    progress_callback: Callable[[str], None] | None = None,
) -> DqnTrainingResult:
    """Train a real Q network on the current V2 candidates after a user action."""
    recs = [dict(row) for row in recommendations or []]
    if candidate_count is not None:
        recs = recs[: max(0, int(candidate_count))]
    signature = data_signature or data_signature_from_recommendations(recs)

    torch_ok, torch_message = get_torch_status()
    if not torch_ok:
        return _empty_result(
            ENV_REQUIRED_STATUS, torch_message, recs, signature, episodes, learning_rate, reflection_mode,
            sample_id, training_mode, int(store_count or 0), int(dc_count or 0), seed,
        )
    if len(recs) < 3:
        return _empty_result(
            INSUFFICIENT_STATUS, "후보 수가 너무 적습니다.", recs, signature, episodes, learning_rate,
            reflection_mode, sample_id, training_mode, int(store_count or 0), int(dc_count or 0),
            seed,
        )

    if progress_callback is not None:
        progress_callback("데이터 구성 중")

    import torch

    max_episodes = max(1, min(int(episodes), 1200))
    if progress_callback is not None:
        progress_callback("DQN 학습 중")
    training = run_dqn_training(
        recs, episodes=max_episodes, learning_rate=float(learning_rate), seed=int(seed),
    )

    model = training["model"]
    losses = [value for value in training["episode_losses"] if value is not None]
    if training["diverged"]:
        losses = [*losses, float("inf")]
    q_rows = training["q_values"]
    q_values_are_finite = bool(q_rows) and all(
        math.isfinite(value) for row in q_rows for value in row
    )
    predicted_actions = [ACTION_LABELS[index] for index in training["selected_actions"]]
    confidences = [round(_confidence_from_q(row) * 100.0, 2) for row in q_rows]
    reward_matrix = action_reward_matrix(recs)
    action_rewards = [
        float(reward_matrix[index][action])
        for index, action in enumerate(training["selected_actions"])
    ]
    references = [
        round(max(0.0, min(100.0, (reward * 72.0) + (confidence / 100.0) * 28.0)), 2)
        for reward, confidence in zip(action_rewards, confidences)
    ]
    q_summaries = []
    for values in q_rows:
        clean = [float(value) for value in values if math.isfinite(float(value))]
        q_summaries.append({
            "max": round(max(clean), 6) if clean else None,
            "min": round(min(clean), 6) if clean else None,
            "avg": round(sum(clean) / len(clean), 6) if clean else None,
        })

    route_ids = _route_ids(recs)
    heuristic_actions = _target_actions(recs)
    reference_actions = reward_optimal_actions(recs)
    episode_rewards = training["episode_rewards"]
    concentration = action_concentration(predicted_actions)

    if progress_callback is not None:
        progress_callback("안정성 검사 중")
    status, message = evaluate_dqn_stability(
        losses, predicted_actions, episode_rewards, candidate_count=len(recs),
        data_signature=signature, current_signature=signature, target_actions=reference_actions,
    )
    if not q_values_are_finite:
        status, message = NEEDS_REVIEW_STATUS, "DQN 출력 값이 안정적이지 않습니다."
    elif not training["parameters_changed"]:
        status, message = NEEDS_REVIEW_STATUS, "학습 중 모델 파라미터가 변하지 않았습니다."

    trained_at, artifact_stamp = _now_parts()
    context_slug = _artifact_context(
        sample_id, training_mode, store_count, dc_count, max_episodes, float(learning_rate)
    )
    model_path = OUTPUT_DIR / f"dqn_model_{context_slug}_{artifact_stamp}.pt"
    result_path = OUTPUT_DIR / f"dqn_result_{context_slug}_{artifact_stamp}.json"
    variant = training_mode if training_mode in VALID_VARIANTS else "original"
    model_payload = {
        "state_dict": model.state_dict(),
        "input_size": training["input_size"],
        "feature_columns": list(DQN_STATE_COLUMNS),
        "feature_stats": training["feature_stats"],
        "state_schema": state_schema_fingerprint(),
        "action_labels": list(ACTION_LABELS),
        "data_signature": signature,
        "sample_id": sample_id,
        "training_mode": variant,
        "variant": variant,
        "seed": int(seed),
        "episodes": max_episodes,
        "learning_rate": float(learning_rate),
        "store_count": int(store_count or 0),
        "dc_count": int(dc_count or 0),
        "trained_at": trained_at,
    }
    saved_model_path: str | None = None
    artifact_error_type: str | None = None
    if status == NORMAL_STATUS and q_values_are_finite:
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model_payload, model_path)
            torch.save(model_payload, LATEST_MODEL_BY_VARIANT[variant])
            torch.save(model_payload, LATEST_MODEL)
            saved_model_path = str(model_path)
        except Exception as exc:
            artifact_error_type = type(exc).__name__
    result = DqnTrainingResult(
        status=status,
        final_status=status,
        stability_status=status,
        message=message,
        data_signature=signature,
        timestamp=trained_at,
        episodes=max_episodes,
        learning_rate=float(learning_rate),
        sample_id=sample_id,
        training_mode=training_mode,
        variant=variant,
        seed=int(seed),
        store_count=int(store_count or 0),
        dc_count=int(dc_count or 0),
        candidate_count=len(recs),
        action_distribution=dict(Counter(predicted_actions)),
        prediction_distribution=dict(Counter(predicted_actions)),
        target_distribution=dict(Counter(reference_actions)),
        reward_history=[round(float(value), 8) for value in episode_rewards],
        loss_history=[
            round(float(value), 8) if value is not None and math.isfinite(float(value)) else None
            for value in training["episode_losses"]
        ],
        reward_summary=_summary(episode_rewards),
        loss_summary=_summary(losses),
        average_confidence=round(sum(confidences) / len(confidences), 2) if confidences else None,
        reflection_mode=reflection_mode,
        model_status="trained" if status == NORMAL_STATUS else "not_applied",
        model_path=saved_model_path,
        result_path=str(result_path),
        feature_columns=list(DQN_STATE_COLUMNS),
        dqn_action_by_route=dict(zip(route_ids, predicted_actions)),
        dqn_confidence_by_route=dict(zip(route_ids, confidences)),
        dqn_reference_by_route=dict(zip(route_ids, references)),
        q_value_summary_by_route=dict(zip(route_ids, q_summaries)),
        dqn_status_by_route={route_id: status for route_id in route_ids},
        diagnostics={
            "historical_artifacts_used": False,
            "heuristic_action_distribution": dict(Counter(heuristic_actions)),
            "reward_optimal_distribution": dict(Counter(reference_actions)),
            "target_action_distribution": dict(Counter(reference_actions)),
            "prediction_distribution": dict(Counter(predicted_actions)),
            "exploration_distribution": training["exploration_distribution"],
            "action_concentration": concentration,
            "parameters_changed": training["parameters_changed"],
            "optimizer_steps": training["optimizer_steps"],
            "replay_capacity": REPLAY_CAPACITY,
            "replay_batch_size": REPLAY_BATCH_SIZE,
            "discount_factor": DISCOUNT_FACTOR,
            "target_sync_episodes": TARGET_SYNC_EPISODES,
            "epsilon_range": [EPSILON_START, EPSILON_END],
            "train_candidate_count": training["train_candidate_count"],
            "holdout_candidate_count": training["holdout_candidate_count"],
            "holdout_mean_reward": training["holdout_mean_reward"],
            "holdout_metrics": training["holdout_metrics"],
            "time_order_column": training["time_order_column"],
            "state_schema": state_schema_fingerprint(),
            "normalization": "train_minmax_v1",
            "normalization_fit_count": training["train_candidate_count"],
            "invalid_action_masking": True,
            "mean_selected_action_reward": (
                round(sum(action_rewards) / len(action_rewards), 6) if action_rewards else None
            ),
            "latest_model_path": str(LATEST_MODEL_BY_VARIANT[variant]) if saved_model_path else None,
            "seed": int(seed),
            "device": training["device"],
            "storage_status": "session_only" if artifact_error_type else "pending",
            "storage_error_type": artifact_error_type,
        },
        historical_artifacts_used=False,
    )
    try:
        saved = save_dqn_result(result)
        result.result_path = str(saved.get("result_path") or result_path)
        result.diagnostics["storage_status"] = "result_only" if artifact_error_type else "saved"
    except Exception as exc:
        result.result_path = None
        result.diagnostics["storage_status"] = "session_only"
        result.diagnostics["storage_error_type"] = type(exc).__name__
    try:
        append_training_comparison_row(result)
    except Exception as exc:  # pragma: no cover - the comparison log never blocks training
        result.diagnostics["comparison_log_error_type"] = type(exc).__name__
    return result


def train_dqn_on_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    reflection_mode: str = "DQN 참고만",
    epochs: int = 80,
) -> DqnTrainingResult:
    """Backward-compatible wrapper."""
    return train_dqn(recommendations, episodes=epochs, reflection_mode=reflection_mode)


def save_dqn_result(result: DqnTrainingResult | Mapping[str, Any]) -> dict[str, Any]:
    data = result.to_dict() if isinstance(result, DqnTrainingResult) else dict(result)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not data.get("timestamp"):
        data["timestamp"] = datetime.now().isoformat(timespec="microseconds")
    proposed = Path(str(data.get("result_path") or "")) if data.get("result_path") else None
    if proposed is not None and proposed.resolve().parent == OUTPUT_DIR.resolve():
        result_path = proposed
    else:
        result_path = OUTPUT_DIR / f"dqn_result_{_timestamp_slug(data['timestamp'])}.json"
    data["result_path"] = str(result_path)
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    status = str(data.get("final_status") or data.get("stability_status") or data.get("status") or "")
    promote_latest = bool(
        status == NORMAL_STATUS
        and data.get("model_path")
        and data.get("model_status") in {"trained", "loaded"}
    )
    if promote_latest:
        LATEST_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    variant = str(data.get("variant") or data.get("training_mode") or "original")
    if promote_latest and variant in LATEST_RESULT_BY_VARIANT:
        LATEST_RESULT_BY_VARIANT[variant].write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return data


COMPARISON_COLUMNS = (
    "timestamp", "sample_id", "variant", "episodes", "learning_rate", "candidate_count",
    "seed", "status", "message", "reward_first", "reward_last", "reward_avg",
    "loss_first", "loss_last", "average_confidence", "distinct_actions",
    "dominant_action", "dominant_ratio", "optimizer_steps", "parameters_changed",
    "holdout_mean_reward", "train_candidate_count", "holdout_candidate_count",
    "state_schema", "model_path",
)


def append_training_comparison_row(
    result: DqnTrainingResult | Mapping[str, Any], path: Path | None = None,
) -> Path:
    """Append one row to the cumulative local training comparison file.

    This file is written by V2 only and is never read back as pipeline input, so
    the historical-artifact guard in ``services.dqn_guard`` stays intact.
    """
    data = result.to_dict() if isinstance(result, DqnTrainingResult) else dict(result)
    diagnostics = dict(data.get("diagnostics") or {})
    concentration = dict(diagnostics.get("action_concentration") or {})
    reward = dict(data.get("reward_summary") or {})
    loss = dict(data.get("loss_summary") or {})
    row = {
        "timestamp": data.get("timestamp"),
        "sample_id": data.get("sample_id"),
        "variant": data.get("variant") or data.get("training_mode"),
        "episodes": data.get("episodes"),
        "learning_rate": data.get("learning_rate"),
        "candidate_count": data.get("candidate_count"),
        "seed": data.get("seed"),
        "status": data.get("final_status") or data.get("status"),
        "message": data.get("message"),
        "reward_first": reward.get("first"),
        "reward_last": reward.get("last"),
        "reward_avg": reward.get("avg"),
        "loss_first": loss.get("first"),
        "loss_last": loss.get("last"),
        "average_confidence": data.get("average_confidence"),
        "distinct_actions": concentration.get("distinct_actions"),
        "dominant_action": concentration.get("dominant_action"),
        "dominant_ratio": concentration.get("dominant_ratio"),
        "optimizer_steps": diagnostics.get("optimizer_steps"),
        "parameters_changed": diagnostics.get("parameters_changed"),
        "holdout_mean_reward": diagnostics.get("holdout_mean_reward"),
        "train_candidate_count": diagnostics.get("train_candidate_count"),
        "holdout_candidate_count": diagnostics.get("holdout_candidate_count"),
        "state_schema": diagnostics.get("state_schema"),
        "model_path": data.get("model_path"),
    }
    target = Path(path) if path is not None else TRAINING_COMPARISON_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return target


def read_training_comparison_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Read back the V2 comparison log for display only."""
    target = Path(path) if path is not None else TRAINING_COMPARISON_CSV
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _save_json_payload_best_effort(payload: dict[str, Any], path: Path) -> None:
    """Persist a report when possible while keeping the in-memory result usable."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["storage_status"] = "saved"
    except Exception as exc:
        payload["result_path"] = None
        payload["storage_status"] = "session_only"
        payload["storage_error_type"] = type(exc).__name__


def load_latest_dqn_result(
    current_signature: str | None = None,
    training_mode: str | None = None,
) -> dict[str, Any] | None:
    path = LATEST_RESULT_BY_VARIANT.get(str(training_mode), LATEST_JSON)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if training_mode and str(result.get("variant") or result.get("training_mode")) != str(training_mode):
        return None
    if current_signature and result.get("data_signature") != current_signature:
        return None
    return result


def can_apply_dqn_to_current_data(training_result: Mapping[str, Any] | None, data_signature: str | None) -> bool:
    if not training_result:
        return False
    status = str(
        training_result.get("final_status")
        or training_result.get("stability_status")
        or training_result.get("status")
        or ""
    )
    if status not in APPLICABLE_STATUSES:
        return False
    stored_signature = training_result.get("data_signature")
    if not data_signature or not stored_signature or stored_signature != data_signature:
        return False
    variant_value = training_result.get("variant") or training_result.get("training_mode")
    if not variant_value:
        return False
    variant = str(variant_value)
    if variant not in VALID_VARIANTS:
        return False
    action_by_route = training_result.get("dqn_action_by_route") or {}
    reference_by_route = training_result.get("dqn_reference_by_route") or {}
    if not action_by_route or not reference_by_route:
        return False
    candidate_count = int(training_result.get("candidate_count") or 0)
    if candidate_count <= 0 or len(action_by_route) != candidate_count or len(reference_by_route) != candidate_count:
        return False
    if any(route_id not in reference_by_route for route_id in action_by_route):
        return False
    for distribution_name in ("prediction_distribution", "target_distribution"):
        distribution = training_result.get(distribution_name) or {}
        if not distribution:
            return False
        total = sum(int(value or 0) for value in distribution.values())
        if total and (
            len([value for value in distribution.values() if int(value or 0) > 0]) < 2
            or max(int(value or 0) for value in distribution.values()) / total >= 0.90
        ):
            return False
    return True


def model_payload_is_compatible(
    payload: Mapping[str, Any] | None,
    data_signature: str | None,
    variant: str | None = None,
) -> tuple[bool, str]:
    """Check a saved model against the current state schema and data."""
    if not payload:
        return False, "저장된 DQN 모델이 없습니다."
    if list(payload.get("action_labels") or []) != list(ACTION_LABELS):
        return False, "DQN action 정의가 달라 비교할 수 없습니다."
    schema = str(payload.get("state_schema") or "")
    if schema != state_schema_fingerprint():
        return False, "DQN feature schema가 달라 비교할 수 없습니다."
    if list(payload.get("feature_columns") or []) != list(DQN_STATE_COLUMNS):
        return False, "DQN feature schema가 달라 비교할 수 없습니다."
    if _coerce_feature_stats(payload.get("feature_stats"), DQN_STATE_COLUMNS) is None:
        return False, "DQN 정규화 schema가 없어 비교할 수 없습니다."
    if variant is not None:
        payload_variant = str(payload.get("variant") or payload.get("training_mode") or "original")
        if payload_variant != variant:
            return False, "요청한 학습 유형과 다른 DQN 모델입니다."
    if data_signature and payload.get("data_signature") != data_signature:
        return False, "현재 데이터와 다른 DQN 모델입니다."
    return True, "DQN 모델 사용 가능"


def _load_model_payload(path: Path) -> Mapping[str, Any] | None:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def find_compatible_model(
    data_signature: str | None,
    variant: str = "original",
    limit: int = 30,
) -> tuple[Path, Mapping[str, Any]] | None:
    """Return the newest saved model that matches the current schema and data.

    Only models this V2 build wrote can match, because the state schema
    fingerprint is stored inside every payload.
    """
    if not OUTPUT_DIR.exists():
        return None
    candidates = [LATEST_MODEL_BY_VARIANT.get(variant), LATEST_MODEL]
    candidates.extend(
        sorted(OUTPUT_DIR.glob("dqn_model_*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]
    )
    seen: set[str] = set()
    for path in candidates:
        if path is None or not path.exists() or str(path) in seen:
            continue
        seen.add(str(path))
        payload = _load_model_payload(path)
        compatible, _ = model_payload_is_compatible(payload, data_signature, variant)
        if compatible and payload is not None:
            return path, payload
    return None


def infer_dqn_actions(
    recommendations: Sequence[Mapping[str, Any]],
    data_signature: str | None = None,
    model_path: str | None = None,
    training_mode: str = "original",
) -> DqnTrainingResult:
    """Run a real forward pass from a saved V2 model, or say training is needed.

    Priority is the explicitly requested model, then the newest compatible one.
    A missing model, a mismatched feature schema, or a failed load always ends
    in a ``학습 필요`` result; no other strategy's values are substituted.
    """
    recs = [dict(row) for row in recommendations or []]
    signature = data_signature or data_signature_from_recommendations(recs)
    torch_ok, torch_message = get_torch_status()
    if not torch_ok:
        return _empty_result(
            ENV_REQUIRED_STATUS, torch_message, recs, signature, training_mode=training_mode
        )
    if not recs:
        return _empty_result(
            NEEDS_TRAINING_STATUS, "후보가 없습니다.", recs, signature, training_mode=training_mode,
        )
    variant = training_mode if training_mode in VALID_VARIANTS else "original"

    payload: Mapping[str, Any] | None = None
    path: Path | None = None
    if model_path:
        path = Path(model_path)
        if not path.exists():
            return _empty_result(
                NEEDS_TRAINING_STATUS, "저장된 DQN 모델이 없습니다.", recs, signature,
                training_mode=training_mode,
            )
        payload = _load_model_payload(path)
        compatible, message = model_payload_is_compatible(payload, signature, variant)
        if not compatible:
            status = PAST_RESULT_STATUS if payload else NEEDS_TRAINING_STATUS
            return _empty_result(status, message, recs, signature, training_mode=training_mode)
    else:
        found = find_compatible_model(signature, variant)
        if found is None:
            return _empty_result(
                NEEDS_TRAINING_STATUS, "현재 데이터와 맞는 DQN 모델이 없습니다.", recs, signature,
                training_mode=training_mode,
            )
        path, payload = found

    import torch

    assert payload is not None and path is not None
    feature_stats = _coerce_feature_stats(payload.get("feature_stats"), DQN_STATE_COLUMNS)
    if feature_stats is None:
        return _empty_result(
            NEEDS_TRAINING_STATUS, "DQN 정규화 schema가 없어 비교할 수 없습니다.", recs, signature,
            training_mode=training_mode,
        )
    base_states = build_training_states(recs, feature_stats=feature_stats)
    expected_size = len(base_states[0]) + 2
    if int(payload.get("input_size") or 0) != expected_size:
        return _empty_result(
            NEEDS_TRAINING_STATUS, "DQN feature schema가 달라 비교할 수 없습니다.", recs, signature,
            training_mode=training_mode,
        )
    seed = int(payload.get("seed") or 17)
    model = _model(expected_size, len(ACTION_LABELS), seed=seed)
    try:
        model.load_state_dict(payload["state_dict"])
    except Exception:
        return _empty_result(
            NEEDS_TRAINING_STATUS, "DQN 모델을 불러오지 못했습니다.", recs, signature,
            training_mode=training_mode,
        )
    model.eval()
    with torch.no_grad():
        selected, q_rows = _greedy_rollout(
            model, base_states, _cost_shares(recs), _action_masks(recs)
        )

    route_ids = _route_ids(recs)
    actions = [ACTION_LABELS[int(index)] for index in selected]
    confidences = [round(_confidence_from_q(row) * 100.0, 2) for row in q_rows]
    reward_matrix = action_reward_matrix(recs)
    action_rewards = [float(reward_matrix[index][action]) for index, action in enumerate(selected)]
    references = [
        round(max(0.0, min(100.0, (reward * 72.0) + (confidence / 100.0) * 28.0)), 2)
        for reward, confidence in zip(action_rewards, confidences)
    ]
    q_summaries = []
    for values in q_rows:
        clean = [float(value) for value in values if math.isfinite(float(value))]
        q_summaries.append({
            "max": round(max(clean), 6) if clean else None,
            "min": round(min(clean), 6) if clean else None,
            "avg": round(sum(clean) / len(clean), 6) if clean else None,
        })
    reference_actions = reward_optimal_actions(recs)
    concentration = action_concentration(actions)
    status, message = evaluate_dqn_stability(
        [], actions, action_rewards, candidate_count=len(recs), data_signature=signature,
        current_signature=signature, target_actions=reference_actions,
    )
    return DqnTrainingResult(
        status=status,
        final_status=status,
        stability_status=status,
        message=message,
        data_signature=signature,
        timestamp=datetime.now().isoformat(timespec="microseconds"),
        episodes=int(payload.get("episodes") or 0),
        learning_rate=float(payload.get("learning_rate") or 0.001),
        sample_id=str(payload.get("sample_id") or "current"),
        candidate_count=len(recs),
        training_mode=variant,
        variant=variant,
        seed=seed,
        action_distribution=dict(Counter(actions)),
        prediction_distribution=dict(Counter(actions)),
        target_distribution=dict(Counter(reference_actions)),
        reward_history=[round(float(value), 8) for value in action_rewards],
        reward_summary=_summary(action_rewards),
        average_confidence=round(sum(confidences) / len(confidences), 2) if confidences else None,
        reflection_mode="DQN 참고만",
        model_status="loaded",
        model_path=str(path),
        feature_columns=list(DQN_STATE_COLUMNS),
        dqn_action_by_route=dict(zip(route_ids, actions)),
        dqn_confidence_by_route=dict(zip(route_ids, confidences)),
        dqn_reference_by_route=dict(zip(route_ids, references)),
        q_value_summary_by_route=dict(zip(route_ids, q_summaries)),
        dqn_status_by_route={route_id: status for route_id in route_ids},
        diagnostics={
            "historical_artifacts_used": False,
            "action_concentration": concentration,
            "state_schema": str(payload.get("state_schema") or ""),
            "inference_source": "saved_model_forward",
            "model_path": str(path),
            "normalization": "train_minmax_v1",
            "invalid_action_masking": True,
            "mean_selected_action_reward": (
                round(sum(action_rewards) / len(action_rewards), 6) if action_rewards else None
            ),
        },
        historical_artifacts_used=False,
    )


def dqn_inference_view(
    recommendations: Sequence[Mapping[str, Any]],
    data_signature: str | None = None,
    training_mode: str = "original",
    training_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach real DQN output to candidates for comparison surfaces.

    An in-session training result for the same data is reused; otherwise a saved
    model is loaded and a forward pass is run.  When neither is usable, every
    row keeps ``비교 불가`` instead of borrowing a Greedy or VHS value.
    """
    recs = [dict(row) for row in recommendations or []]
    signature = data_signature or data_signature_from_recommendations(recs)
    source = "training_result"
    result: Mapping[str, Any] | None = None
    if training_result and can_apply_dqn_to_current_data(training_result, signature):
        result = dict(training_result)
    else:
        source = "saved_model"
        result = infer_dqn_actions(recs, signature, training_mode=training_mode).to_dict()
    applicable = can_apply_dqn_to_current_data(result, signature)
    updated = apply_dqn_reference_to_recommendations(recs, result, signature)
    status = str(result.get("final_status") or result.get("status") or NEEDS_TRAINING_STATUS)
    concentration = dict((result.get("diagnostics") or {}).get("action_concentration") or {})
    if not concentration:
        concentration = action_concentration(
            [row.get("dqn_action") for row in updated if applicable]
        )
    return {
        "recommendations": updated,
        "available": bool(applicable),
        "status": status,
        "message": str(result.get("message") or status),
        "source": source,
        "model_path": result.get("model_path"),
        "model_status": result.get("model_status", "not_trained"),
        "action_concentration": concentration,
        "average_confidence": result.get("average_confidence"),
        "candidate_count": len(updated),
    }


def apply_dqn_reference_to_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    training_result: Mapping[str, Any] | None,
    data_signature: str | None = None,
) -> list[dict[str, Any]]:
    result = dict(training_result or {})
    status = str(
        result.get("final_status")
        or result.get("stability_status")
        or result.get("status")
        or NEEDS_TRAINING_STATUS
    )
    applicable = can_apply_dqn_to_current_data(result, data_signature)
    if result.get("data_signature") and data_signature and result.get("data_signature") != data_signature:
        status = PAST_RESULT_STATUS
        applicable = False
    action_by_route = result.get("dqn_action_by_route") or {}
    confidence_by_route = result.get("dqn_confidence_by_route") or {}
    reference_by_route = result.get("dqn_reference_by_route") or {}
    updated: list[dict[str, Any]] = []
    rows = list(recommendations or [])
    route_keys = _route_ids(rows)
    for row, route_key in zip(rows, route_keys):
        item = dict(row)
        if applicable and route_key in action_by_route:
            item["dqn_action"] = action_by_route.get(route_key)
            item["dqn_confidence"] = confidence_by_route.get(route_key)
            item["dqn_reference_score"] = reference_by_route.get(route_key, 0.0)
            item["dqn_status"] = status
        else:
            item["dqn_action"] = "비교 불가"
            item["dqn_confidence"] = None
            item["dqn_reference_score"] = 0.0
            item["dqn_status"] = status
        item["dqn_correction"] = 0.0
        updated.append(item)
    return updated


def apply_dqn_result_to_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    training_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Attach DQN comparison fields without changing VHS or Varo ranking.

    The old ``DQN 약하게 반영`` option is intentionally reference-only.
    DQN remains an independent policy comparison and never auto-corrects the
    production score through this compatibility entry point.
    """
    result = dict(training_result or {})
    return apply_dqn_reference_to_recommendations(
        recommendations,
        result,
        result.get("data_signature"),
    )


def get_dqn_status(training_result: Mapping[str, Any] | None = None) -> DqnStatus:
    if not training_result:
        return DqnStatus()
    status = str(
        training_result.get("final_status")
        or training_result.get("stability_status")
        or training_result.get("status")
        or NEEDS_TRAINING_STATUS
    )
    return DqnStatus(
        connected=status in APPLICABLE_STATUSES,
        training_enabled=True,
        inference_enabled=status in APPLICABLE_STATUSES,
        historical_artifacts_used=False,
        message=str(training_result.get("message") or status),
        status=status,
        reflection_mode=str(training_result.get("reflection_mode") or "DQN 참고만"),
    )


def dqn_result_summary(training_result: Mapping[str, Any] | None, recommendations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = dict(training_result or {})
    status = str(
        result.get("final_status")
        or result.get("stability_status")
        or result.get("status")
        or NEEDS_TRAINING_STATUS
    )
    return {
        "status": status,
        "message": result.get("message") or status,
        "data_signature": result.get("data_signature"),
        "candidate_count": result.get("candidate_count", len(recommendations or [])),
        "episodes": result.get("episodes", 0),
        "learning_rate": result.get("learning_rate"),
        "variant": result.get("variant") or result.get("training_mode") or "original",
        "seed": result.get("seed", 17),
        "action_distribution": result.get("action_distribution") or {},
        "prediction_distribution": result.get("prediction_distribution") or {},
        "target_distribution": result.get("target_distribution") or {},
        "reward_summary": result.get("reward_summary") or {},
        "loss_summary": result.get("loss_summary") or {},
        "average_confidence": result.get("average_confidence"),
        "reflection_mode": result.get("reflection_mode") or "DQN 참고만",
        "model_status": result.get("model_status", "not_trained"),
        "historical_artifacts_used": False,
    }


def _result_row(label: str, result: Mapping[str, Any], data_signature: str | None = None) -> dict[str, Any]:
    loss = result.get("loss_summary") or {}
    target_distribution = result.get("target_distribution") or {}
    prediction_distribution = result.get("prediction_distribution") or result.get("action_distribution") or {}
    applicable = can_apply_dqn_to_current_data(result, data_signature or result.get("data_signature"))
    status = (
        result.get("final_status")
        or result.get("stability_status")
        or result.get("status")
        or NEEDS_TRAINING_STATUS
    )
    return {
        "sample_id": result.get("sample_id") or label,
        "sample_name": label,
        "variant": result.get("variant") or result.get("training_mode") or "original",
        "후보 수": result.get("candidate_count", 0),
        "target 종류 수": len([value for value in target_distribution.values() if int(value or 0) > 0]),
        "예측 종류 수": len([value for value in prediction_distribution.values() if int(value or 0) > 0]),
        "loss 시작": loss.get("first") if not result.get("loss_history") else result["loss_history"][0],
        "loss 끝": loss.get("last"),
        "final_status": status,
        "stability_status": result.get("stability_status") or status,
        "VHS 반영 여부": "참고 반영" if applicable else "반영 안 함",
        "판단 근거": result.get("message") or status,
    }


def compare_dqn_training_sets(
    original_recommendations: Sequence[Mapping[str, Any]],
    balanced_recommendations: Sequence[Mapping[str, Any]],
    data_signature: str,
    episodes: int = 180,
    learning_rate: float = 0.001,
    sample_id: str = "current",
    store_count: int = 0,
    dc_count: int = 0,
) -> dict[str, Any]:
    """Train and persist an original/balanced comparison for one data set."""
    original = train_dqn(
        original_recommendations,
        data_signature=data_signature,
        episodes=episodes,
        learning_rate=learning_rate,
        reflection_mode="DQN 참고만",
        sample_id=sample_id,
        training_mode="original",
        store_count=store_count,
        dc_count=dc_count,
    ).to_dict()
    balanced = train_dqn(
        balanced_recommendations,
        data_signature=data_signature,
        episodes=episodes,
        learning_rate=learning_rate,
        reflection_mode="DQN 참고만",
        sample_id=sample_id,
        training_mode="balanced",
        store_count=store_count,
        dc_count=dc_count,
    ).to_dict()
    if can_apply_dqn_to_current_data(balanced, data_signature):
        preferred = "균형형"
    elif can_apply_dqn_to_current_data(original, data_signature):
        preferred = "원본"
    else:
        preferred = "없음"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="microseconds"),
        "data_signature": data_signature,
        "preferred": preferred,
        "rows": [
            _result_row("원본", original, data_signature),
            _result_row("균형형", balanced, data_signature),
        ],
        "original_result": original,
        "balanced_result": balanced,
        "result_path": str(LATEST_COMPARISON_JSON),
    }
    _save_json_payload_best_effort(payload, LATEST_COMPARISON_JSON)
    return payload


def build_dqn_batch_comparison_report(
    original_batch: Mapping[str, Any] | None,
    balanced_batch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Persist a comparison of already-run batches without starting training."""
    rows: list[dict[str, Any]] = []
    for batch, variant_label in ((original_batch, "원본"), (balanced_batch, "균형형")):
        for item in (batch or {}).get("results") or []:
            result = dict(item.get("result") or {})
            label = str(item.get("label") or result.get("sample_id") or variant_label)
            row = _result_row(label, result, result.get("data_signature"))
            row["학습 구분"] = variant_label
            rows.append(row)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="microseconds"),
        "rows": rows,
        "original_count": len((original_batch or {}).get("results") or []),
        "balanced_count": len((balanced_batch or {}).get("results") or []),
        "result_path": str(LATEST_COMPARISON_JSON),
    }
    _save_json_payload_best_effort(payload, LATEST_COMPARISON_JSON)
    return payload


def train_dqn_batch(
    training_sets: Sequence[Mapping[str, Any]],
    episodes: int = 90,
    learning_rate: float = 0.001,
    progress_callback: Callable[[int, int, str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train supplied V2 sample sets sequentially and persist a compact report."""
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, sample in enumerate(training_sets, start=1):
        recommendations = sample.get("recommendations") or []
        signature = str(sample.get("data_signature") or data_signature_from_recommendations(recommendations))
        label = str(sample.get("label") or f"DQN 샘플 {index:02d}")
        try:
            result = train_dqn(
                recommendations,
                data_signature=signature,
                episodes=episodes,
                learning_rate=learning_rate,
                reflection_mode="DQN 참고만",
                sample_id=str(sample.get("sample_id") or label),
                training_mode=str(sample.get("mode") or "original"),
                store_count=int(sample.get("store_count") or 0),
                dc_count=int(sample.get("dc_count") or 0),
                seed=17 + index,
            ).to_dict()
        except Exception as exc:
            result = _empty_result(
                NEEDS_REVIEW_STATUS,
                f"{label} 학습 오류: {type(exc).__name__}",
                recommendations,
                signature,
                episodes,
                learning_rate,
                sample_id=str(sample.get("sample_id") or label),
                training_mode=str(sample.get("mode") or "original"),
                store_count=int(sample.get("store_count") or 0),
                dc_count=int(sample.get("dc_count") or 0),
                seed=17 + index,
            ).to_dict()
            result["diagnostics"]["error_type"] = type(exc).__name__
        rows.append(_result_row(label, result, signature))
        results.append({"label": label, "result": result})
        if progress_callback is not None:
            progress_callback(index, len(training_sets), label, result)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="microseconds"),
        "count": len(results),
        "rows": rows,
        "results": results,
        "result_path": str(LATEST_BATCH_JSON),
    }
    _save_json_payload_best_effort(payload, LATEST_BATCH_JSON)
    return payload
