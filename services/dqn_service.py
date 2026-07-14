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
from typing import Any, Callable, Mapping, Sequence

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
    return False, "DQN 학습은 실행 환경 설정 후 사용할 수 있습니다."


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
        return NEEDS_REVIEW_STATUS, "원본 라벨 분포가 한쪽으로 치우쳤습니다."
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
) -> DqnTrainingResult:
    """Train a small V2-only Q network after an explicit user action."""
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

    torch.manual_seed(int(seed))
    model = _model(len(vectors[0]), len(ACTION_LABELS), seed=seed)
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
        logits_are_finite = bool(torch.isfinite(logits).all().item())
        if logits_are_finite:
            probabilities = torch.softmax(logits, dim=1)
            confidence_values, predicted_indices = torch.max(probabilities, dim=1)
        else:
            confidence_values = torch.zeros(len(recs), dtype=torch.float32)
            predicted_indices = torch.zeros(len(recs), dtype=torch.long)

    route_ids = _route_ids(recs)
    predicted_actions = [ACTION_LABELS[int(index)] for index in predicted_indices.tolist()]
    confidences = [round(float(value) * 100.0, 2) for value in confidence_values.tolist()]
    references = [
        round(max(0.0, min(100.0, (reward * 72.0) + (confidence / 100.0) * 28.0)), 2)
        for reward, confidence in zip(rewards, confidences)
    ]
    q_summaries = []
    for values in logits.tolist():
        clean = [float(value) for value in values if math.isfinite(float(value))]
        q_summaries.append({
            "max": round(max(clean), 6) if clean else None,
            "min": round(min(clean), 6) if clean else None,
            "avg": round(sum(clean) / len(clean), 6) if clean else None,
        })

    status, message = evaluate_dqn_stability(
        losses, predicted_actions, rewards, candidate_count=len(recs), data_signature=signature,
        current_signature=signature, target_actions=target_actions,
    )
    if not logits_are_finite:
        status, message = NEEDS_REVIEW_STATUS, "DQN 출력 값이 안정적이지 않습니다."
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trained_at, artifact_stamp = _now_parts()
    context_slug = _artifact_context(
        sample_id, training_mode, store_count, dc_count, max_episodes, float(learning_rate)
    )
    model_path = OUTPUT_DIR / f"dqn_model_{context_slug}_{artifact_stamp}.pt"
    result_path = OUTPUT_DIR / f"dqn_result_{context_slug}_{artifact_stamp}.json"
    variant = training_mode if training_mode in VALID_VARIANTS else "original"
    model_payload = {
        "state_dict": model.state_dict(),
        "input_size": len(vectors[0]),
        "feature_columns": list(FEATURE_COLUMNS),
        "action_labels": list(ACTION_LABELS),
        "data_signature": signature,
        "sample_id": sample_id,
        "training_mode": variant,
        "variant": variant,
        "seed": int(seed),
        "store_count": int(store_count or 0),
        "dc_count": int(dc_count or 0),
    }
    saved_model_path: str | None = None
    if status == NORMAL_STATUS and logits_are_finite:
        torch.save(model_payload, model_path)
        torch.save(model_payload, LATEST_MODEL_BY_VARIANT[variant])
        torch.save(model_payload, LATEST_MODEL)
        saved_model_path = str(model_path)
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
        target_distribution=dict(Counter(target_actions)),
        reward_history=[round(float(value), 8) for value in rewards],
        loss_history=[round(float(value), 8) if math.isfinite(float(value)) else None for value in losses],
        reward_summary=_summary(rewards),
        loss_summary=_summary(losses),
        average_confidence=round(sum(confidences) / len(confidences), 2) if confidences else None,
        reflection_mode=reflection_mode,
        model_status="trained" if status == NORMAL_STATUS else "not_applied",
        model_path=saved_model_path,
        result_path=str(result_path),
        dqn_action_by_route=dict(zip(route_ids, predicted_actions)),
        dqn_confidence_by_route=dict(zip(route_ids, confidences)),
        dqn_reference_by_route=dict(zip(route_ids, references)),
        q_value_summary_by_route=dict(zip(route_ids, q_summaries)),
        dqn_status_by_route={route_id: status for route_id in route_ids},
        diagnostics={
            "historical_artifacts_used": False,
            "target_action_distribution": dict(Counter(target_actions)),
            "prediction_distribution": dict(Counter(predicted_actions)),
            "latest_model_path": str(LATEST_MODEL_BY_VARIANT[variant]) if saved_model_path else None,
            "seed": int(seed),
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
        data["timestamp"] = datetime.now().isoformat(timespec="microseconds")
    proposed = Path(str(data.get("result_path") or "")) if data.get("result_path") else None
    if proposed is not None and proposed.resolve().parent == OUTPUT_DIR.resolve():
        result_path = proposed
    else:
        result_path = OUTPUT_DIR / f"dqn_result_{_timestamp_slug(data['timestamp'])}.json"
    data["result_path"] = str(result_path)
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    variant = str(data.get("variant") or data.get("training_mode") or "original")
    if variant in LATEST_RESULT_BY_VARIANT:
        LATEST_RESULT_BY_VARIANT[variant].write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return data


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


def infer_dqn_actions(
    recommendations: Sequence[Mapping[str, Any]],
    data_signature: str | None = None,
    model_path: str | None = None,
    training_mode: str = "original",
) -> DqnTrainingResult:
    """Run inference from a saved V2 model only when explicitly requested."""
    recs = [dict(row) for row in recommendations or []]
    signature = data_signature or data_signature_from_recommendations(recs)
    torch_ok, torch_message = get_torch_status()
    if not torch_ok:
        return _empty_result(
            ENV_REQUIRED_STATUS, torch_message, recs, signature, training_mode=training_mode
        )
    variant = training_mode if training_mode in VALID_VARIANTS else "original"
    path = Path(model_path) if model_path else LATEST_MODEL_BY_VARIANT[variant]
    if not path.exists():
        return _empty_result(
            NEEDS_TRAINING_STATUS, "저장된 DQN 모델이 없습니다.", recs, signature,
            training_mode=training_mode,
        )

    import torch

    payload = torch.load(path, map_location="cpu")
    model_signature = payload.get("data_signature")
    if model_signature != signature:
        return _empty_result(
            PAST_RESULT_STATUS, "현재 데이터와 다른 DQN 모델입니다.", recs, model_signature,
            training_mode=training_mode,
        )
    payload_variant = str(payload.get("variant") or payload.get("training_mode") or "original")
    if payload_variant != variant:
        return _empty_result(
            PAST_RESULT_STATUS, "요청한 학습 유형과 다른 DQN 모델입니다.", recs,
            model_signature, training_mode=training_mode,
        )

    vectors = build_state_vectors(recs)
    if not vectors:
        return _empty_result(
            NEEDS_TRAINING_STATUS, "후보가 없습니다.", recs, signature,
            training_mode=training_mode,
        )
    seed = int(payload.get("seed") or 17)
    model = _model(int(payload.get("input_size") or len(vectors[0])), len(ACTION_LABELS), seed=seed)
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
    target_actions = _target_actions(recs)
    status, message = evaluate_dqn_stability(
        [], actions, rewards, candidate_count=len(recs), data_signature=signature,
        current_signature=signature, target_actions=target_actions,
    )
    return DqnTrainingResult(
        status=status,
        final_status=status,
        stability_status=status,
        message=message,
        data_signature=signature,
        timestamp=datetime.now().isoformat(timespec="microseconds"),
        candidate_count=len(recs),
        training_mode=variant,
        variant=variant,
        seed=seed,
        action_distribution=dict(Counter(actions)),
        prediction_distribution=dict(Counter(actions)),
        target_distribution=dict(Counter(target_actions)),
        reward_history=[round(float(value), 8) for value in rewards],
        reward_summary=_summary(rewards),
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
    status = str(
        result.get("final_status")
        or result.get("stability_status")
        or result.get("status")
        or NEEDS_TRAINING_STATUS
    )
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_COMPARISON_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_COMPARISON_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_BATCH_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
