"""V2-scoped DQN training, inference, and comparison helpers.

This module only uses the current Varo V2 recommendation candidates. It never
reads historical DQN artifacts from the original project.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

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

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "dqn"
LATEST_JSON = OUTPUT_DIR / "latest_dqn_result.json"
LATEST_MODEL = OUTPUT_DIR / "latest_dqn_model.pt"

NORMAL_STATUS = "정상"
NEEDS_TRAINING_STATUS = "학습 필요"
NEEDS_REVIEW_STATUS = "검토 필요"
INSUFFICIENT_STATUS = "학습 부족"
INACTIVE_STATUS = "비활성"
PAST_RESULT_STATUS = "과거 결과"
ENV_REQUIRED_STATUS = "실행 환경 필요"

APPLICABLE_STATUSES = {NORMAL_STATUS, "연결", "connected", "ok", "ready"}

# UI-facing labels only. Internal status values (stored results, gating logic,
# saved files) keep the raw strings above; these soften how they read on screen.
DISPLAY_STATUS_LABELS = {
    NORMAL_STATUS: "비교 가능",
    "연결": "비교 가능",
    NEEDS_REVIEW_STATUS: "데이터 확인 필요",
    "불안정": "데이터 편향 큼",
    INSUFFICIENT_STATUS: "후보 수 부족",
    NEEDS_TRAINING_STATUS: "학습 전",
    "미연결": "학습 전",
    PAST_RESULT_STATUS: "이전 데이터 결과",
    ENV_REQUIRED_STATUS: "실행 환경 필요",
    INACTIVE_STATUS: "비활성",
}


def dqn_display_status(status: Any) -> str:
    """Softened display label for a raw DQN status (display only, logic unchanged)."""
    text = str(status or "").strip()
    return DISPLAY_STATUS_LABELS.get(text, text or "-")


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
    candidate_count: int = 0
    sample_id: str | None = None
    sample_name: str | None = None
    store_count: int | None = None
    dc_count: int | None = None
    created_at: str | None = None
    final_status: str = ""
    stability_status: str = ""
    action_distribution: dict[str, int] = field(default_factory=dict)
    reward_summary: dict[str, float] = field(default_factory=dict)
    loss_summary: dict[str, float | None] = field(default_factory=dict)
    reward_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    average_confidence: float | None = None
    reflection_mode: str = "DQN 참고만"
    model_status: str = "not_trained"
    model_path: str | None = None
    result_path: str | None = None
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
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
    if is_torch_available():
        return True, "DQN 실행 가능"
    return False, "DQN 실행 환경이 필요합니다."


def get_torch_runtime_status() -> dict[str, Any]:
    """Pure PyTorch runtime probe for the DQN tab badge.

    Optional import only; a missing torch is a normal 'needs environment' state,
    never an error/traceback. Diagnosis, sample loading, and VHS/Greedy/Pareto
    comparison all work regardless of this result — only real training needs torch.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return {
            "available": False,
            "version": None,
            "device": "cpu",
            "cuda_available": False,
            "can_train": False,
            "message": "PyTorch 설치 필요",
        }
    version = None
    cuda_available = False
    try:
        version = str(getattr(torch, "__version__", "") or "") or None
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive: never fail on probe
        pass
    device = "gpu" if cuda_available else "cpu"
    return {
        "available": True,
        "version": version,
        "device": device,
        "cuda_available": cuda_available,
        "can_train": True,
        "message": "GPU 학습 가능" if cuda_available else "CPU 학습 가능",
    }


# Explicit sentinel for an action value that cannot be mapped to a known action.
# Used only when the caller opts in via allow_unknown=True so unmappable values
# are surfaced honestly instead of being silently coerced to a normal action.
UNKNOWN_ACTION = "확인 필요"


def build_action_mapping() -> dict[str, int]:
    """The single label -> index mapping shared across training, inference, and UI."""
    return {label: index for index, label in enumerate(ACTION_LABELS)}


def action_index_from_label(label: Any) -> int | None:
    """Index of a label within the shared ACTION_LABELS vocabulary, else None."""
    try:
        return ACTION_LABELS.index(str(label))
    except ValueError:
        return None


def _default_for_route(route_type: Any, default: str) -> str:
    route = str(route_type or "").upper()
    if route == "VIA_DC":
        return "DC 경유 이동"
    if route == "DIRECT":
        return "직접 이동"
    return default


def _as_action_index(value: Any) -> int | None:
    """Parse a numeric action code (int/float/numeric string) into an index.

    Booleans and non-integral / non-numeric values return None so they fall
    through to the string-alias path. This keeps numeric action indices aligned
    with ACTION_LABELS everywhere instead of collapsing onto index 0.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def normalize_action(
    value: Any,
    default: str = "재고 이동",
    route_type: Any = None,
    allow_unknown: bool = False,
) -> str:
    """Normalize any action representation to one shared Korean action label.

    Handles numeric action indices, Korean labels, English/alias strings, and
    missing values. Missing/empty values are inferred from route type (an
    unspecified action, not an unknown one). A genuinely unrecognized non-empty
    value is coerced to ``default`` for backward compatibility unless
    ``allow_unknown`` is set, in which case it returns ``UNKNOWN_ACTION`` so the
    caller can display it as 확인 필요 rather than a fabricated normal action.
    """
    # Missing / empty → inferred from route type (未지정, not '알 수 없음').
    if value is None or (isinstance(value, float) and not math.isfinite(value)) or str(value).strip() == "":
        return _default_for_route(route_type, default)

    # Numeric action code → label by shared index.
    index = _as_action_index(value)
    if index is not None:
        if 0 <= index < len(ACTION_LABELS):
            return ACTION_LABELS[index]
        return UNKNOWN_ACTION if allow_unknown else default

    text = str(value).strip()
    lowered = text.lower()
    if text in ACTION_ALIASES:
        return ACTION_ALIASES[text]
    if lowered in ACTION_ALIASES:
        return ACTION_ALIASES[lowered]
    if text in ACTION_LABELS:
        return text
    for key, label in ACTION_ALIASES.items():
        if key.lower() in lowered:
            if label == "재고 이동":
                return _default_for_route(route_type, label)
            return label
    return UNKNOWN_ACTION if allow_unknown else _default_for_route(route_type, default)


def data_signature_from_recommendations(recommendations: Sequence[Mapping[str, Any]]) -> str:
    serializable = []
    for row in recommendations or []:
        serializable.append({
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


def build_state_vectors(
    recommendations: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> list[list[float]]:
    """Map recommendation candidates to normalized DQN state vectors.

    Missing values become 0.5, a neutral midpoint. Route type and cold-chain
    hints are encoded in stable extra dimensions.
    """
    recs = [dict(row) for row in recommendations or []]
    stats = _feature_stats(recs, feature_columns)
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
    return [str(row.get("route_id") or f"R{index + 1:03d}") for index, row in enumerate(recommendations or [])]


def _target_actions(recommendations: Sequence[Mapping[str, Any]]) -> list[str]:
    actions: list[str] = []
    for row in recommendations or []:
        source_action = row.get("varo_action") or row.get("greedy_strategy") or row.get("greedy_action")
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


def evaluate_dqn_stability(
    losses: Sequence[float],
    actions: Sequence[str],
    rewards: Sequence[float],
    candidate_count: int | None = None,
    data_signature: str | None = None,
    current_signature: str | None = None,
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
    if distribution and max(distribution.values()) / max(1, len(actions)) >= 0.90:
        return NEEDS_REVIEW_STATUS, "action 분포가 한쪽으로 치우쳤습니다."
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
        return {"min": None, "max": None, "avg": None, "last": None}
    return {
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
        "avg": round(sum(clean) / len(clean), 6),
        "last": round(clean[-1], 6),
    }


def _empty_result(
    status: str,
    message: str,
    recommendations: Sequence[Mapping[str, Any]],
    data_signature: str | None = None,
    episodes: int = 0,
    learning_rate: float = 0.001,
    reflection_mode: str = "DQN 참고만",
    sample_id: str | None = None,
    store_count: int | None = None,
    dc_count: int | None = None,
    sample_name: str | None = None,
) -> DqnTrainingResult:
    labels = _target_actions(recommendations)
    rewards = calculate_rewards(recommendations)
    now = datetime.now().isoformat(timespec="seconds")
    return DqnTrainingResult(
        status=status,
        message=message,
        data_signature=data_signature,
        timestamp=now,
        created_at=now,
        episodes=episodes,
        learning_rate=learning_rate,
        candidate_count=len(recommendations or []),
        sample_id=sample_id,
        sample_name=sample_name,
        store_count=store_count,
        dc_count=dc_count,
        final_status=status,
        stability_status=status,
        action_distribution=dict(Counter(labels)),
        reward_summary=_summary(rewards),
        loss_summary={},
        reward_history=[round(float(value), 6) for value in rewards],
        loss_history=[],
        reflection_mode=reflection_mode,
        model_status="not_trained",
        diagnostics={"historical_artifacts_used": False},
        historical_artifacts_used=False,
    )


def _model(input_size: int, output_size: int):
    import torch
    from torch import nn

    torch.manual_seed(17)
    return nn.Sequential(
        nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Linear(32, 24),
        nn.ReLU(),
        nn.Linear(24, output_size),
    )


def _training_stem(
    sample_id: str | None, store_count: int | None, dc_count: int | None,
    episodes: int, learning_rate: float, timestamp: str, variant: str = "original",
) -> str:
    """File stem carrying sample_id, store/dc counts, variant, episodes, lr, timestamp."""
    safe_id = "".join(ch for ch in str(sample_id or "sample") if ch.isalnum() or ch in "_-") or "sample"
    safe_variant = "balanced" if str(variant).lower().startswith("bal") else "original"
    lr_token = str(learning_rate).replace(".", "p")
    return (
        f"dqn_{safe_id}_s{int(store_count or 0)}dc{int(dc_count or 0)}_{safe_variant}"
        f"_ep{int(episodes)}_lr{lr_token}_{timestamp}"
    )


def train_dqn(
    recommendations: Sequence[Mapping[str, Any]],
    data_signature: str | None = None,
    episodes: int = 300,
    learning_rate: float = 0.001,
    candidate_count: int | None = None,
    reflection_mode: str = "DQN 참고만",
    sample_id: str | None = None,
    store_count: int | None = None,
    dc_count: int | None = None,
    sample_name: str | None = None,
    variant: str = "original",
) -> DqnTrainingResult:
    """Train a small V2-only Q network after an explicit user action."""
    recs = [dict(row) for row in recommendations or []]
    if candidate_count is not None:
        recs = recs[: max(0, int(candidate_count))]
    signature = data_signature or data_signature_from_recommendations(recs)

    torch_ok, torch_message = get_torch_status()
    if not torch_ok:
        return _empty_result(ENV_REQUIRED_STATUS, torch_message, recs, signature, episodes, learning_rate, reflection_mode, sample_id, store_count, dc_count, sample_name)
    if len(recs) < 3:
        return _empty_result(INSUFFICIENT_STATUS, "후보 수가 너무 적습니다.", recs, signature, episodes, learning_rate, reflection_mode, sample_id, store_count, dc_count, sample_name)

    import torch
    from torch import nn

    vectors = build_state_vectors(recs)
    rewards = calculate_rewards(recs)
    target_actions = _target_actions(recs)
    action_index = build_action_mapping()
    x = torch.tensor(vectors, dtype=torch.float32)
    y = torch.zeros((len(recs), len(ACTION_LABELS)), dtype=torch.float32)
    for row_index, (action, reward) in enumerate(zip(target_actions, rewards)):
        y[row_index, action_index.get(action, 0)] = float(reward)

    model = _model(len(vectors[0]), len(ACTION_LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()
    losses: list[float] = []
    max_episodes = max(1, min(int(episodes), 1200))
    for _ in range(max_episodes):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        if not torch.isfinite(loss):
            losses.append(float("inf"))
            break
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence_values, predicted_indices = torch.max(probabilities, dim=1)

    route_ids = _route_ids(recs)
    predicted_actions = [ACTION_LABELS[int(index)] for index in predicted_indices.tolist()]
    confidences = [round(float(value) * 100.0, 2) for value in confidence_values.tolist()]
    references = [
        round(max(0.0, min(100.0, (reward * 72.0) + (confidence / 100.0) * 28.0)), 2)
        for reward, confidence in zip(rewards, confidences)
    ]
    q_summaries = []
    for values in logits.tolist():
        clean = [float(value) for value in values]
        q_summaries.append({
            "max": round(max(clean), 6),
            "min": round(min(clean), 6),
            "avg": round(sum(clean) / len(clean), 6),
        })

    status, message = evaluate_dqn_stability(
        losses, predicted_actions, rewards, candidate_count=len(recs), data_signature=signature, current_signature=signature
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _training_stem(sample_id, store_count, dc_count, max_episodes, learning_rate, timestamp, variant)
    model_path = OUTPUT_DIR / f"{stem}.pt"
    result_path = OUTPUT_DIR / f"{stem}.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": len(vectors[0]),
            "feature_columns": list(FEATURE_COLUMNS),
            "action_labels": list(ACTION_LABELS),
            "data_signature": signature,
        },
        model_path,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": len(vectors[0]),
            "feature_columns": list(FEATURE_COLUMNS),
            "action_labels": list(ACTION_LABELS),
            "data_signature": signature,
        },
        LATEST_MODEL,
    )
    created_at = datetime.now().isoformat(timespec="seconds")
    result = DqnTrainingResult(
        status=status,
        message=message,
        data_signature=signature,
        timestamp=created_at,
        created_at=created_at,
        episodes=max_episodes,
        learning_rate=float(learning_rate),
        candidate_count=len(recs),
        sample_id=sample_id,
        sample_name=sample_name,
        store_count=store_count,
        dc_count=dc_count,
        final_status=status,
        stability_status=status,
        action_distribution=dict(Counter(predicted_actions)),
        reward_summary=_summary(rewards),
        loss_summary=_summary(losses),
        reward_history=[round(float(value), 6) for value in rewards],
        loss_history=[round(float(value), 6) for value in losses],
        average_confidence=round(sum(confidences) / len(confidences), 2) if confidences else None,
        reflection_mode=reflection_mode,
        model_status="trained" if status in {NORMAL_STATUS, NEEDS_REVIEW_STATUS} else "not_applied",
        model_path=str(model_path),
        result_path=str(result_path),
        dqn_action_by_route=dict(zip(route_ids, predicted_actions)),
        dqn_confidence_by_route=dict(zip(route_ids, confidences)),
        dqn_reference_by_route=dict(zip(route_ids, references)),
        q_value_summary_by_route=dict(zip(route_ids, q_summaries)),
        dqn_status_by_route={route_id: status for route_id in route_ids},
        diagnostics={
            "historical_artifacts_used": False,
            "target_action_distribution": dict(Counter(target_actions)),
            "latest_model_path": str(LATEST_MODEL),
        },
        historical_artifacts_used=False,
    )
    save_dqn_result(result)
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
        data["timestamp"] = datetime.now().isoformat(timespec="seconds")
    # Keep the traceable per-run filename that train_dqn already assigned; only
    # synthesize a timestamp-based name when the caller did not provide one.
    if data.get("result_path"):
        result_path = Path(data["result_path"])
    else:
        timestamp = str(data["timestamp"]).replace(":", "").replace("-", "").replace("T", "_")
        result_path = OUTPUT_DIR / f"dqn_result_{timestamp}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    data["result_path"] = str(result_path)
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def load_latest_dqn_result() -> dict[str, Any] | None:
    if not LATEST_JSON.exists():
        return None
    try:
        return json.loads(LATEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def can_apply_dqn_to_current_data(training_result: Mapping[str, Any] | None, data_signature: str | None) -> bool:
    if not training_result:
        return False
    status = str(training_result.get("status") or "")
    if status not in APPLICABLE_STATUSES:
        return False
    if data_signature and training_result.get("data_signature") != data_signature:
        return False
    return bool(training_result.get("dqn_action_by_route"))


def infer_dqn_actions(
    recommendations: Sequence[Mapping[str, Any]],
    data_signature: str | None = None,
    model_path: str | None = None,
) -> DqnTrainingResult:
    """Run inference from a saved V2 model only when explicitly requested."""
    recs = [dict(row) for row in recommendations or []]
    signature = data_signature or data_signature_from_recommendations(recs)
    torch_ok, torch_message = get_torch_status()
    if not torch_ok:
        return _empty_result(ENV_REQUIRED_STATUS, torch_message, recs, signature)
    path = Path(model_path) if model_path else LATEST_MODEL
    if not path.exists():
        return _empty_result(NEEDS_TRAINING_STATUS, "저장된 DQN 모델이 없습니다.", recs, signature)

    import torch

    payload = torch.load(path, map_location="cpu")
    model_signature = payload.get("data_signature")
    if model_signature != signature:
        return _empty_result(PAST_RESULT_STATUS, "현재 데이터와 다른 DQN 모델입니다.", recs, model_signature)

    vectors = build_state_vectors(recs)
    if not vectors:
        return _empty_result(NEEDS_TRAINING_STATUS, "후보가 없습니다.", recs, signature)
    model = _model(int(payload.get("input_size") or len(vectors[0])), len(ACTION_LABELS))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x = torch.tensor(vectors, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence_values, predicted_indices = torch.max(probabilities, dim=1)
    route_ids = _route_ids(recs)
    actions = [ACTION_LABELS[int(index)] for index in predicted_indices.tolist()]
    confidences = [round(float(value) * 100.0, 2) for value in confidence_values.tolist()]
    rewards = calculate_rewards(recs)
    references = [
        round(max(0.0, min(100.0, (reward * 72.0) + (confidence / 100.0) * 28.0)), 2)
        for reward, confidence in zip(rewards, confidences)
    ]
    status, message = evaluate_dqn_stability([], actions, rewards, candidate_count=len(recs), data_signature=signature, current_signature=signature)
    return DqnTrainingResult(
        status=status,
        message=message,
        data_signature=signature,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        candidate_count=len(recs),
        final_status=status,
        stability_status=status,
        created_at=datetime.now().isoformat(timespec="seconds"),
        action_distribution=dict(Counter(actions)),
        reward_summary=_summary(rewards),
        reward_history=[round(float(value), 6) for value in rewards],
        average_confidence=round(sum(confidences) / len(confidences), 2) if confidences else None,
        reflection_mode="DQN 참고만",
        model_status="loaded",
        model_path=str(path),
        dqn_action_by_route=dict(zip(route_ids, actions)),
        dqn_confidence_by_route=dict(zip(route_ids, confidences)),
        dqn_reference_by_route=dict(zip(route_ids, references)),
        dqn_status_by_route={route_id: status for route_id in route_ids},
        historical_artifacts_used=False,
    )


def apply_dqn_reference_to_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    training_result: Mapping[str, Any] | None,
    data_signature: str | None = None,
) -> list[dict[str, Any]]:
    result = dict(training_result or {})
    status = str(result.get("status") or NEEDS_TRAINING_STATUS)
    applicable = can_apply_dqn_to_current_data(result, data_signature)
    if result.get("data_signature") and data_signature and result.get("data_signature") != data_signature:
        status = PAST_RESULT_STATUS
        applicable = False
    action_by_route = result.get("dqn_action_by_route") or {}
    confidence_by_route = result.get("dqn_confidence_by_route") or {}
    reference_by_route = result.get("dqn_reference_by_route") or {}
    updated: list[dict[str, Any]] = []
    for row in recommendations or []:
        item = dict(row)
        route_id = str(item.get("route_id") or "")
        if applicable and route_id in action_by_route:
            item["dqn_action"] = action_by_route.get(route_id)
            item["dqn_confidence"] = confidence_by_route.get(route_id)
            item["dqn_reference_score"] = reference_by_route.get(route_id, 0.0)
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
    """Backward-compatible wrapper with the old weak-reflection behavior."""
    result = dict(training_result or {})
    updated = apply_dqn_reference_to_recommendations(recommendations, result, result.get("data_signature"))
    status = str(result.get("status") or NEEDS_TRAINING_STATUS)
    mode = str(result.get("reflection_mode") or "DQN 참고만")
    if status not in APPLICABLE_STATUSES or mode != "DQN 약하게 반영":
        return updated
    for item in updated:
        confidence = (_num(item.get("dqn_confidence")) or 0.0) / 100.0
        dqn_action = normalize_action(item.get("dqn_action"), default="보류", route_type=item.get("route_type"))
        baseline = normalize_action(item.get("varo_action") or item.get("greedy_action"), default="보류", route_type=item.get("route_type"))
        correction = round((2.0 if dqn_action == baseline else -1.0) * max(0.0, min(1.0, confidence)), 2)
        item["dqn_correction"] = correction
        vhs = _num(item.get("vhs_score"))
        if vhs is not None:
            item["vhs_score"] = round(max(0.0, min(100.0, vhs + correction)), 2)
    return updated


def get_dqn_status(training_result: Mapping[str, Any] | None = None) -> DqnStatus:
    if not training_result:
        return DqnStatus()
    status = str(training_result.get("status") or NEEDS_TRAINING_STATUS)
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
    status = str(result.get("status") or NEEDS_TRAINING_STATUS)
    return {
        "status": status,
        "message": result.get("message") or status,
        "data_signature": result.get("data_signature"),
        "candidate_count": result.get("candidate_count", len(recommendations or [])),
        "episodes": result.get("episodes", 0),
        "learning_rate": result.get("learning_rate"),
        "action_distribution": result.get("action_distribution") or {},
        "reward_summary": result.get("reward_summary") or {},
        "loss_summary": result.get("loss_summary") or {},
        "average_confidence": result.get("average_confidence"),
        "reflection_mode": result.get("reflection_mode") or "DQN 참고만",
        "model_status": result.get("model_status", "not_trained"),
        "stability_status": result.get("stability_status") or status,
        "sample_name": result.get("sample_name"),
        "historical_artifacts_used": False,
    }
