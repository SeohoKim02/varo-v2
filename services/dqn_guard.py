"""Hard guardrails that keep historical DQN outputs out of Varo V2."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

BLOCKED_DQN_MODULES = frozenset({
    "dqn_agent",
    "torch_dqn_agent",
    "dqn_recommender",
    "dqn_stability",
    "train_rl_agent",
    "replay_buffer",
    "rl_data_logger",
    "rl_policy_helper",
})

BLOCKED_DQN_PATTERNS = (
    "dqn_artifacts",
    "rl_training_log.csv",
    "rl_training_log_master.csv",
    "rl_training_summary.json",
    "rl_q_table.csv",
    "rl_policy_table.csv",
    "dqn_training_comparison.csv",
    "reward_history.csv",
    "loss_history.csv",
    "*.pt",
    "*.pth",
    "*.pkl",
    "*.joblib",
)

_DQN_COLUMN_TOKENS = (
    "dqn",
    "reward",
    "loss",
    "q_table",
    "qtable",
    "policy_table",
    "policy",
    "agreement_status",
    "model_agreement",
    "replay_buffer",
    "training_log",
    "training_summary",
    "model_path",
)

DQN_EXCLUSION_REASON = (
    "기존 DQN 학습 결과는 reward 이상치 가능성 때문에 V2 분석과 추천에 사용하지 않습니다."
)


def is_blocked_dqn_path(path: str | Path) -> bool:
    """Classify a name without opening or reading the file."""
    name = Path(path).name.lower()
    full = str(path).replace("\\", "/").lower()
    if "dqn_artifacts" in full:
        return True
    if name.endswith((".pt", ".pth", ".pkl", ".joblib")):
        return True
    blocked_names = {
        "rl_training_log.csv",
        "rl_training_log_master.csv",
        "rl_training_summary.json",
        "rl_q_table.csv",
        "rl_policy_table.csv",
        "dqn_training_comparison.csv",
        "reward_history.csv",
        "loss_history.csv",
    }
    return name in blocked_names


def is_dqn_column(column: object) -> bool:
    lowered = str(column).strip().lower()
    return any(token in lowered for token in _DQN_COLUMN_TOKENS)


def strip_dqn_columns(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a copy without DQN, reward, loss, or policy-derived columns."""
    if df is None:
        return pd.DataFrame()
    blocked = [column for column in df.columns if is_dqn_column(column)]
    return df.drop(columns=blocked, errors="ignore").copy()


def dqn_exclusion_report(extra_patterns: Iterable[str] = ()) -> dict[str, object]:
    patterns = list(dict.fromkeys([*BLOCKED_DQN_PATTERNS, *extra_patterns]))
    return {
        "status": "미연결",
        "reason": DQN_EXCLUSION_REASON,
        "blocked_patterns": patterns,
        "artifacts_read": False,
        "training_executed": False,
        "inference_executed": False,
        "score_influence": False,
    }
