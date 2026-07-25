"""Original-preserving balanced training data for DQN stability checks.

The source workbooks are never opened for writing. Each sample is loaded
read-only through the normal V2 pipeline; every core numeric feature (quantity,
distance, cost, expiry, demand, disposal) is kept exactly, and only each
candidate's ``target_action``/``varo_action`` label is re-derived from its own
feature affinity with a quota pass, so the DQN target distribution spans >= 4
realistic actions instead of collapsing onto 보류. Derived candidates are written
to ``outputs/dqn_balanced_samples/`` with provenance.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

BALANCED_DIR = Path(__file__).resolve().parents[1] / "outputs" / "dqn_balanced_samples"
BALANCE_POLICY = "affinity_quota_v1"

# Ordered action vocabulary used for balancing (all valid V2 DQN actions).
_ACTIONS = ("재고 이동", "DC 경유 이동", "할인", "긴급 할인", "폐기", "보류")
_DISPOSE_RISK_FLOOR = 55.0  # 폐기 is only realistic above this disposal-risk score


def _num(value: Any, default: float = 50.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _affinity(row: Mapping[str, Any]) -> dict[str, float]:
    save = _num(row.get("savings_score"))
    feas = _num(row.get("feasibility_score"))
    dem = _num(row.get("demand_fit_score"))
    promo = _num(row.get("promotion_score"))
    cost = _num(row.get("route_cost_score"))
    risk = _num(row.get("disposal_risk_score"))
    return {
        "재고 이동": 0.45 * save + 0.30 * feas + 0.15 * dem + 0.10 * cost,
        "DC 경유 이동": 0.40 * save + 0.25 * feas + 0.20 * cost + 10.0,
        "할인": 0.55 * promo + 0.30 * (100 - dem) + 0.15 * (100 - save),
        "긴급 할인": 0.50 * promo + 0.30 * risk + 0.20 * (100 - dem),
        "폐기": 0.65 * risk + 0.35 * (100 - save),
        "보류": 0.45 * (100 - save) + 0.30 * (100 - feas) + 0.25 * (100 - promo),
    }


def _eligible_actions(row: Mapping[str, Any]) -> list[str]:
    """Realism gate: never suggest an impossible/absurd action for a candidate."""
    via = str(row.get("route_type") or "").upper() == "VIA_DC"
    risk = _num(row.get("disposal_risk_score"))
    eligible = ["재고 이동", "할인", "긴급 할인", "보류"]  # always plausible (>= 4)
    if via:
        eligible.append("DC 경유 이동")
    if risk >= _DISPOSE_RISK_FLOOR:
        eligible.append("폐기")
    return eligible


def _assign_quota(affinities: Sequence[Mapping[str, float]], eligible: Sequence[Sequence[str]]) -> list[str]:
    """Balanced assignment: each action takes its best-fitting eligible candidates
    up to a per-action quota; leftovers fall to their top eligible action."""
    n = len(affinities)
    assigned: list[str | None] = [None] * n
    present = [action for action in _ACTIONS if any(action in elig for elig in eligible)]
    if not present:
        return ["보류"] * n
    quota = max(1, -(-n // len(present)))  # ceil(n / #actions)
    counts = {action: 0 for action in present}
    for _ in range(quota):
        for action in present:
            if counts[action] >= quota:
                continue
            cands = [i for i in range(n) if assigned[i] is None and action in eligible[i]]
            if not cands:
                continue
            best = max(cands, key=lambda i: affinities[i][action])
            assigned[best] = action
            counts[action] += 1
    for i in range(n):
        if assigned[i] is None:
            choices = list(eligible[i]) or list(_ACTIONS)
            assigned[i] = max(choices, key=lambda action: affinities[i][action])
    return [action or "보류" for action in assigned]


def balance_actions(recommendations: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (rebalanced_recs, meta). Core numeric fields are untouched; only the
    DQN target label (varo_action/target_action) is re-derived for balance."""
    recs = [dict(row) for row in recommendations or []]
    if not recs:
        return [], {"balance_policy": BALANCE_POLICY, "action_distribution": {}, "candidate_count": 0}
    affinities = [_affinity(row) for row in recs]
    eligible = [_eligible_actions(row) for row in recs]
    assigned = _assign_quota(affinities, eligible)
    for row, action in zip(recs, assigned):
        row["target_action"] = action
        row["varo_action"] = action  # DQN reads varo_action first as its target
    distribution = dict(Counter(assigned))
    return recs, {
        "balance_policy": BALANCE_POLICY,
        "action_distribution": distribution,
        "action_kinds": len(distribution),
        "candidate_count": len(recs),
    }


def generate_balanced_sample(info: Any) -> dict[str, Any]:
    """Load one catalog sample read-only, rebalance labels, and save a derived JSON."""
    from services.analysis_pipeline import build_v2_state
    from services.data_loader import DataLoadError, load_excel_data

    path = Path(info.file_path)
    if not path.exists():
        return {"ok": False, "status": "파일 없음", "message": "샘플 파일을 찾을 수 없습니다.", "sample_id": info.sample_id}
    try:
        state = build_v2_state(load_excel_data(path))
    except DataLoadError as exc:
        return {"ok": False, "status": "로드 실패", "message": str(exc), "sample_id": info.sample_id}
    recs = state.get("recommendations") or []
    balanced, meta = balance_actions(recs)
    generated_at = datetime.now().isoformat(timespec="seconds")
    payload = {
        "original_sample_id": info.sample_id,
        "sample_name": f"balanced_{info.sample_id}_{info.file_name}",
        "derived_from": info.file_name,
        "original_path": str(path),
        "generated_at": generated_at,
        "balance_policy": BALANCE_POLICY,
        "store_count": info.store_count,
        "dc_count": info.dc_count,
        "candidate_count": meta["candidate_count"],
        "action_distribution": meta["action_distribution"],
        "recommendations": balanced,
    }
    BALANCED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = BALANCED_DIR / f"balanced_{info.sample_id}_{stamp}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "ok": True,
        "status": "생성 완료",
        "sample_id": info.sample_id,
        "path": str(out_path),
        "file_name": out_path.name,
        "action_distribution": meta["action_distribution"],
        "action_kinds": meta["action_kinds"],
        "candidate_count": meta["candidate_count"],
        "generated_at": generated_at,
    }


def load_balanced_payload(path: str | Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def list_balanced_samples() -> list[dict[str, Any]]:
    """Metadata for previously generated balanced samples, newest first."""
    if not BALANCED_DIR.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(BALANCED_DIR.glob("balanced_*.json"), reverse=True):
        payload = load_balanced_payload(path)
        if not payload:
            continue
        items.append({
            "original_sample_id": payload.get("original_sample_id"),
            "derived_from": payload.get("derived_from"),
            "generated_at": payload.get("generated_at"),
            "balance_policy": payload.get("balance_policy"),
            "action_distribution": payload.get("action_distribution") or {},
            "candidate_count": payload.get("candidate_count", 0),
            "file_name": path.name,
            "path": str(path),
        })
    return items
