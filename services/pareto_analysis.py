"""Pareto (non-dominated) candidate analysis for Varo V2.

A self-contained, deterministic multi-objective comparison used as a
research-grade reference alongside VHS / Greedy / DQN. It never installs external
packages and never reads historical artifacts. All objectives are expressed on
the shared 0..100 "higher is better" scale that ``vhs_score_engine`` produces, so
a candidate that is at least as good on every objective and strictly better on
one dominates the other (standard Pareto dominance).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

# Objectives, all "higher is better" after V2 normalization.
#
# Deliberately four, not every VHS component: each extra axis makes dominance
# harder to achieve, so a long list pushes almost every candidate onto the front
# and the check stops discriminating. These four are the objectives an operator
# actually trades off — money, urgency, transport burden, and destination need.
PARETO_OBJECTIVES: tuple[str, ...] = (
    "net_benefit_score",     # 순효과 최대
    "disposal_risk_score",   # 폐기 위험 감소 최대
    "route_cost_score",      # 이동 부담 최소 (정규화: 높을수록 낮은 부담)
    "demand_fit_score",      # 수요 적합도 최대
)
OBJECTIVE_LABELS: dict[str, str] = {
    "net_benefit_score": "예상 순효과",
    "disposal_risk_score": "폐기 위험 감소",
    "route_cost_score": "이동 부담",
    "demand_fit_score": "수요 적합도",
}

_EPS = 1e-9


def _vector(row: Mapping[str, Any], objectives: Sequence[str]) -> list[float]:
    vector: list[float] = []
    for objective in objectives:
        try:
            value = float(row.get(objective))
        except (TypeError, ValueError):
            value = 50.0
        vector.append(value if math.isfinite(value) else 50.0)
    return vector


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True when a is >= b on every objective and strictly greater on one."""
    ge_all = all(x >= y - _EPS for x, y in zip(a, b))
    gt_any = any(x > y + _EPS for x, y in zip(a, b))
    return ge_all and gt_any


def compute_pareto(
    recommendations: Sequence[Mapping[str, Any]],
    objectives: Sequence[str] = PARETO_OBJECTIVES,
) -> list[dict[str, Any]]:
    """Return per-candidate Pareto info aligned with the input order.

    Uses fast non-dominated sorting to assign a Pareto front number
    (``pareto_rank``; 1 = non-dominated front). ``pareto_status`` is 비지배 for the
    first front and 지배됨 otherwise.
    """
    recs = [dict(row) for row in recommendations or []]
    count = len(recs)
    if count == 0:
        return []
    vectors = [_vector(row, objectives) for row in recs]

    dominates_set: list[set[int]] = [set() for _ in range(count)]
    dominated_count = [0] * count
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            if _dominates(vectors[i], vectors[j]):
                dominates_set[i].add(j)
            elif _dominates(vectors[j], vectors[i]):
                dominated_count[i] += 1

    rank = [0] * count
    working = dominated_count[:]
    current = [i for i in range(count) if working[i] == 0]
    front_number = 0
    assigned: set[int] = set()
    while current:
        front_number += 1
        nxt: list[int] = []
        for i in current:
            rank[i] = front_number
            assigned.add(i)
        for i in current:
            for j in dominates_set[i]:
                if j in assigned:
                    continue
                working[j] -= 1
                if working[j] <= 0 and j not in assigned and j not in nxt:
                    nxt.append(j)
        current = nxt
    fallback_front = front_number + 1
    results: list[dict[str, Any]] = []
    for i in range(count):
        front = rank[i] or fallback_front
        results.append({
            "pareto_rank": int(front),
            "pareto_status": "비지배" if front == 1 else "지배됨",
            "pareto_dominated_by": int(dominated_count[i]),
            "pareto_dominates": int(len(dominates_set[i])),
        })
    return results


def pareto_summary(pareto_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(pareto_rows)
    front_size = sum(1 for row in pareto_rows if int(row.get("pareto_rank") or 0) == 1)
    return {
        "front_size": front_size,
        "candidate_count": total,
        "front_ratio": round(front_size / total, 4) if total else 0.0,
        "objectives": list(PARETO_OBJECTIVES),
        "objective_labels": [OBJECTIVE_LABELS[name] for name in PARETO_OBJECTIVES],
        "basis": f"{len(PARETO_OBJECTIVES)}개 목적함수 비지배 정렬(Pareto front) · 보조 검증 기준",
    }
