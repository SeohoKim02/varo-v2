"""An explicit, single-objective Greedy baseline for comparison against VHS.

Why this exists
---------------
The in-app ``greedy_rank`` comes from the legacy ``heuristic_optimizer``, which
blends cost, quantity, strategy and a reason bonus. That is useful as a second
opinion, but it is a *composite* — comparing VHS against it does not answer the
question a reviewer actually asks:

    "단순한 규칙 대비 VHS가 어떤 상황에서 더 나은 결정을 하는가?"

So this module defines the classic myopic rule instead, with one objective and
nothing else in it:

    **가장 큰 예상 절감액부터 실행한다.** 동점이면 이동 비용이 낮은 쪽,
    그다음은 route_id 순서.

That is deliberately naive: it ignores the transport cost relative to the
benefit, whether the destination actually needs that much, what the source looks
like afterwards, and how uncertain the demand is. Those are exactly the things
VHS adds, so the difference between the two rankings is interpretable.

Nothing here changes what the app recommends. It is a measurement tool used by
the benchmark and reported in the technical validation output.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

BASELINE_NAME = "절감액 우선 Greedy"
BASELINE_OBJECTIVE = "예상 절감액 최대화 (단일 목적, 이동 비용·수요·이동 후 상태 미고려)"


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sort_key(rec: Mapping[str, Any]) -> tuple[float, float, str]:
    saving = _num(rec.get("expected_saving"))
    cost = _num(rec.get("estimated_cost"))
    if cost is None:
        cost = _num(rec.get("move_cost"))
    return (
        -(saving if saving is not None else -math.inf),
        cost if cost is not None else math.inf,
        str(rec.get("route_id") or ""),
    )


def greedy_ranking(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Candidates ordered by the baseline rule, each stamped with its rank."""
    ordered = sorted((dict(row) for row in recommendations or []), key=_sort_key)
    for position, row in enumerate(ordered, start=1):
        row["baseline_rank"] = position
    return ordered


def _vhs_order(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def key(rec: Mapping[str, Any]) -> tuple[float, str]:
        rank = _num(rec.get("varo_final_rank")) or _num(rec.get("vhs_rank")) or _num(rec.get("rank"))
        return (rank if rank is not None else math.inf, str(rec.get("route_id") or ""))
    return sorted((dict(row) for row in recommendations or []), key=key)


def _total(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [_num(row.get(field)) for row in rows]
    usable = [value for value in values if value is not None]
    return round(sum(usable), 2) if usable else None


def _coverage(rows: Sequence[Mapping[str, Any]], need_field: str) -> float | None:
    """How much real need the selection actually covers (never more than the need)."""
    total = 0.0
    seen = False
    for row in rows:
        quantity = _num(row.get("recommended_qty"))
        need = _num(row.get(need_field))
        if quantity is None or need is None:
            continue
        seen = True
        total += min(quantity, max(0.0, need))
    return round(total, 2) if seen else None


def compare_to_vhs(
    recommendations: Sequence[Mapping[str, Any]], top_k: int = 5,
) -> dict[str, Any]:
    """Head-to-head metrics for the two rankings over the same candidate set.

    Both rankings see exactly the same feasible candidates, so any difference is
    the ordering rule alone. ``top_k`` is the shortlist an operator would work
    through in one round.
    """
    recs = [dict(row) for row in recommendations or []]
    if not recs:
        return {
            "baseline": BASELINE_NAME, "objective": BASELINE_OBJECTIVE,
            "candidate_count": 0, "top_k": top_k, "comparable": False,
        }
    vhs_order = _vhs_order(recs)
    greedy_order = greedy_ranking(recs)
    vhs_top = vhs_order[:top_k]
    greedy_top = greedy_order[:top_k]
    vhs_ids = [str(row.get("route_id")) for row in vhs_top]
    greedy_ids = [str(row.get("route_id")) for row in greedy_top]

    top1_vhs = vhs_order[0]
    top1_greedy = greedy_order[0]
    return {
        "baseline": BASELINE_NAME,
        "objective": BASELINE_OBJECTIVE,
        "candidate_count": len(recs),
        "top_k": top_k,
        "comparable": True,
        "top1_match": vhs_ids[0] == greedy_ids[0],
        "topk_overlap": len(set(vhs_ids) & set(greedy_ids)),
        "topk_overlap_ratio": round(len(set(vhs_ids) & set(greedy_ids)) / max(1, len(vhs_ids)), 4),
        "vhs": {
            "top1_route_id": vhs_ids[0],
            "net_benefit_total": _total(vhs_top, "net_benefit"),
            "expected_saving_total": _total(vhs_top, "expected_saving"),
            "move_cost_total": _total(vhs_top, "estimated_cost"),
            "shortage_covered": _coverage(vhs_top, "target_shortfall"),
            "surplus_relieved": _coverage(vhs_top, "source_movable"),
            "top1_pareto_status": top1_vhs.get("pareto_status"),
            "top1_robustness": top1_vhs.get("robustness_status"),
            "top1_demand_scenario": top1_vhs.get("demand_scenario_status"),
        },
        "greedy": {
            "top1_route_id": greedy_ids[0],
            "net_benefit_total": _total(greedy_top, "net_benefit"),
            "expected_saving_total": _total(greedy_top, "expected_saving"),
            "move_cost_total": _total(greedy_top, "estimated_cost"),
            "shortage_covered": _coverage(greedy_top, "target_shortfall"),
            "surplus_relieved": _coverage(greedy_top, "source_movable"),
            "top1_pareto_status": top1_greedy.get("pareto_status"),
            "top1_robustness": top1_greedy.get("robustness_status"),
            "top1_demand_scenario": top1_greedy.get("demand_scenario_status"),
        },
    }
