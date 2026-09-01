"""Unified candidate judgment log for Varo V2 (후보 판단 기록).

One structured record per candidate — recommended *and* excluded — so every
screen answers the same five operator questions from the same data:

* 왜 이 추천이 1순위인가?          → status 추천 + recommendation_reasons
* 왜 이 후보는 추천되지 않았는가?   → status 이동 불가/데이터 부족/계산 불가 + exclusion_reasons
* 왜 이동 수량이 예상보다 적은가?   → quantity_basis (제한 근거)
* 어떤 원본 데이터 확인이 필요한가? → source_references (원본 행 계보)
* 원본에서 어느 셀을 고쳐야 하나?   → source_references[*].rows / column

The record is the single source of truth the UI reads, so a candidate can never
show 확인 필요 on one page and 정상 on another. Status is derived from the
existing feasibility gate (never a new parallel verdict); the ledger only splits
feasibility's single 이동 불가 into 이동 불가 / 데이터 부족 / 계산 불가 by the
existing ``reason_code`` and marks the top feasible candidate 추천.

Records store *references* (row numbers, a few values), never copies of whole
original frames, so session state stays small (docs/VALIDATION.md 판단 기록 구조).
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from services.candidate_lineage import build_source_references, traceable_row_count
from services.feasibility import (
    STATUS_BLOCKED,
    STATUS_CHECK,
    STATUS_OK,
    InventoryContext,
    build_inventory_context,
    inventory_floor_source_label,
)
from services.decision_metrics import quantity_plan
from services.v2_summaries import recommendation_reason

# --------------------------------------------------------------------------- #
# Status vocabulary (사용자 화면 문구). Derived from feasibility, never parallel.
# --------------------------------------------------------------------------- #
STATUS_RECOMMENDED = "추천"
STATUS_RECOMMENDABLE = "추천 가능"
STATUS_CHECK_NEEDED = "확인 필요"
STATUS_BLOCKED_MOVE = "이동 불가"
STATUS_INSUFFICIENT = "데이터 부족"
STATUS_NOT_COMPUTABLE = "계산 불가"

# Stable machine codes (logs/CSV only, never shown as-is on the basic screen).
_STATUS_CODE = {
    STATUS_RECOMMENDED: "recommended",
    STATUS_RECOMMENDABLE: "recommendable",
    STATUS_CHECK_NEEDED: "check_needed",
    STATUS_BLOCKED_MOVE: "blocked_move",
    STATUS_INSUFFICIENT: "insufficient_data",
    STATUS_NOT_COMPUTABLE: "not_computable",
}

# feasibility reason_code → which excluded status it maps to. 데이터/경로 부재는
# '데이터 부족', 수량 자체를 계산 못 하면 '계산 불가', 물리적으로 불가능하면
# '이동 불가'. (확인 필요 코드는 STATUS_CHECK 경로에서 처리되므로 여기 없음.)
_BLOCK_STATUS_BY_CODE = {
    "same_source_target": STATUS_BLOCKED_MOVE,
    "quantity_exceeds_stock": STATUS_BLOCKED_MOVE,
    "inventory_floor_violation": STATUS_BLOCKED_MOVE,
    "duplicate": STATUS_BLOCKED_MOVE,
    "no_route": STATUS_INSUFFICIENT,
    "via_dc_missing_dc": STATUS_INSUFFICIENT,
    # invalid_quantity: 수량 자체가 없으면(None/NaN/inf) 계산 불가, 0/음수는 이동 불가.
    # 이 세분은 _status_for에서 실제 수량 값을 보고 결정한다.
    "invalid_quantity": STATUS_BLOCKED_MOVE,
}

_EXCLUDED_STATUSES = (STATUS_BLOCKED_MOVE, STATUS_INSUFFICIENT, STATUS_NOT_COMPUTABLE)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def make_candidate_id(candidate: Mapping[str, Any], data_signature: str | None) -> str:
    """Stable, sort-independent id from the candidate's identity + data signature.

    Distinguishes DIRECT vs VIA_DC (route_type) and DC01 vs DC02 (dc_id), never
    reuses a DataFrame index, and cannot collide across data signatures. Internal
    only — the basic user screen never shows it.
    """
    signature = (str(data_signature or "")[:8]) or "nosig"
    parts = [
        candidate.get("product_id"),
        candidate.get("source_id"),
        candidate.get("target_id"),
        str(candidate.get("route_type") or "").upper() or "-",
        candidate.get("dc_id") or "-",
    ]
    base = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"C-{signature}-{digest}"


# --------------------------------------------------------------------------- #
# Status derivation
# --------------------------------------------------------------------------- #
def _status_for(candidate: Mapping[str, Any], is_top: bool) -> tuple[str, str]:
    feas = str(candidate.get("feasibility_status") or STATUS_OK)
    code = str(candidate.get("feasibility_reason_code") or "")
    if feas == STATUS_BLOCKED:
        # invalid_quantity: distinguish 계산 불가(수량 자체가 없음) vs 이동 불가(0/음수).
        if code == "invalid_quantity" and _num(candidate.get("recommended_qty")) is None:
            status = STATUS_NOT_COMPUTABLE
        else:
            status = _BLOCK_STATUS_BY_CODE.get(code, STATUS_BLOCKED_MOVE)
    elif feas == STATUS_CHECK:
        status = STATUS_CHECK_NEEDED
    else:  # STATUS_OK
        status = STATUS_RECOMMENDED if is_top else STATUS_RECOMMENDABLE
    return status, _STATUS_CODE[status]


# --------------------------------------------------------------------------- #
# Quantity basis (이동 수량 결정 근거)
# --------------------------------------------------------------------------- #
def quantity_basis(candidate: Mapping[str, Any], context: InventoryContext) -> dict[str, Any]:
    source = candidate.get("source_id")
    target = candidate.get("target_id")
    product = candidate.get("product_id")
    qty = _num(candidate.get("recommended_qty"))
    plan = quantity_plan(candidate, context)
    stock = plan.get("source_stock")
    safety = plan.get("inventory_floor_value")
    demand = plan.get("target_shortfall")
    movable = plan.get("available_to_move")
    floor_source = str(plan.get("inventory_floor_source") or "unavailable")

    basis: dict[str, Any] = {
        "recommended_qty": qty,
        "source_stock": stock,
        "source_safety": safety if stock is not None else None,
        "source_movable": movable,
        "target_demand": demand,
        "inventory_floor_source": floor_source,
        "inventory_floor_source_label": inventory_floor_source_label(floor_source),
        "quantity_limit_reason": plan.get("quantity_limit_reason"),
        "limiting_factor": None,
        "basis_text": None,
        "computable": False,
    }
    if qty is None:
        basis["basis_text"] = "이동 수량을 계산할 수 없어 확인이 필요합니다."
        return basis
    if movable is None and demand is None:
        basis["basis_text"] = f"권장 이동 수량은 {qty:,.0f}개입니다. 제한 근거는 원본 재고·수요 데이터를 확인해야 합니다."
        return basis

    limits: list[tuple[str, float]] = []
    if movable is not None:
        limits.append(("출발 점포 이동 가능량", movable))
    if demand is not None and demand > 0:
        target_label = (
            "도착 점포 목표 재고 부족량"
            if plan.get("target_stock_goal") is not None else "도착 점포 부족량"
        )
        limits.append((target_label, demand))
    if not limits:
        basis["basis_text"] = f"권장 이동 수량은 {qty:,.0f}개입니다."
        return basis
    limits.sort(key=lambda item: item[1])
    factor, limit_value = limits[0]
    if len(limits) == 2 and abs(limits[0][1] - limits[1][1]) < 1e-9:
        factor = "출발 가능량과 도착 부족량이 같음"
    basis.update({
        "limiting_factor": factor,
        "computable": True,
        "basis_text": f"{factor}을 기준으로 {qty:,.0f}개 이동을 권장합니다.",
    })
    return basis


# --------------------------------------------------------------------------- #
# Reasons
# --------------------------------------------------------------------------- #
def _recommendation_reasons(
    candidate: Mapping[str, Any],
    recommendations: Sequence[Mapping[str, Any]],
    reasons_by_route: Mapping[str, Any] | None,
) -> list[str]:
    route_id = str(candidate.get("route_id"))
    detail = (reasons_by_route or {}).get(route_id)
    if not detail:
        detail = recommendation_reason(candidate, recommendations)
    sentences = list(detail.get("sentences") or [])
    return sentences[:3]


def _exclusion_reasons(candidate: Mapping[str, Any], basis: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    primary = str(candidate.get("feasibility_reason") or "").strip()
    if primary:
        reasons.append(primary)
    # A second, most-actionable hint (원본 어디를 먼저 볼지) without internal codes.
    code = str(candidate.get("feasibility_reason_code") or "")
    hint = {
        "quantity_exceeds_stock": "출발 점포 재고 행을 확인해 수량을 조정하세요.",
        "inventory_floor_violation": "남겨야 할 재고보다 적어지지 않도록 이동 수량을 줄이세요.",
        "inventory_floor_unavailable": "재고현황 시트의 안전재고 또는 수요 변동 값을 확인하세요.",
        "no_route": "이동경로 시트에 해당 출발·도착 경로가 있는지 확인하세요.",
        "via_dc_missing_dc": "경유 DC 정보를 이동경로·점포 시트에서 확인하세요.",
        "cost_uncomputable": "이동경로 시트의 비용 또는 거리 값을 확인하세요.",
        "saving_uncomputable": "단가·비용 값을 확인해 절감액을 계산할 수 있게 하세요.",
        "source_stock_missing": "재고현황 시트에 출발 점포 재고 행이 있는지 확인하세요.",
    }.get(code)
    if hint:
        reasons.append(hint)
    return reasons[:2]


# --------------------------------------------------------------------------- #
# Ledger build
# --------------------------------------------------------------------------- #
def build_candidate_ledger(
    annotated: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None = None,
    raw_data: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    data_signature: str | None = None,
    reasons_by_route: Mapping[str, Any] | None = None,
    stability_status: str | None = None,
) -> list[dict[str, Any]]:
    """One judgment record per candidate (feasibility-annotated recs in, records out).

    ``annotated`` must be the full feasibility output (feasible + blocked) so the
    excluded candidates never silently disappear.
    """
    context = build_inventory_context(data)
    lineage_source = raw_data if raw_data else (data or {})
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Rank the OK (추천 가능) candidates to mark the single 추천 (top) one.
    feasible = [rec for rec in annotated if str(rec.get("feasibility_status")) == STATUS_OK]
    top_route_id = None
    if feasible:
        def _rank_key(rec: Mapping[str, Any]) -> float:
            rank = _num(rec.get("varo_final_rank")) or _num(rec.get("vhs_rank")) or _num(rec.get("rank"))
            if rank is not None:
                return rank
            return 1_000_000 - (_num(rec.get("vhs_score")) or 0.0)
        top_route_id = str(sorted(feasible, key=_rank_key)[0].get("route_id"))

    records: list[dict[str, Any]] = []
    for candidate in annotated:
        route_id = str(candidate.get("route_id"))
        is_top = route_id == top_route_id
        status, status_code = _status_for(candidate, is_top)
        blocks = status in _EXCLUDED_STATUSES
        basis = quantity_basis(candidate, context)
        references = build_source_references(candidate, lineage_source, source_metadata)

        if blocks or status == STATUS_CHECK_NEEDED:
            exclusion = _exclusion_reasons(candidate, basis)
            recommendation = []
            short_reason = exclusion[0] if exclusion else "확인이 필요한 후보입니다."
        else:
            recommendation = _recommendation_reasons(candidate, annotated, reasons_by_route)
            exclusion = []
            short_reason = recommendation[0] if recommendation else "연결된 지표를 기준으로 추천한 후보입니다."

        records.append({
            "candidate_id": make_candidate_id(candidate, data_signature),
            "route_id": route_id,
            "data_signature": data_signature,
            "status": status,
            "status_code": status_code,
            "blocks_recommendation": blocks,
            "is_top": is_top,
            "short_reason": short_reason,
            "detailed_reasons": recommendation + exclusion,
            "recommendation_reasons": recommendation,
            "exclusion_reasons": exclusion,
            "quantity_basis": basis,
            "source_references": references,
            "traceable_row_count": traceable_row_count(references),
            "score_components": {
                "vhs_score": _num(candidate.get("vhs_score")),
                "expected_saving": _num(candidate.get("expected_saving")),
                "estimated_cost": _num(candidate.get("estimated_cost")),
            },
            "confidence": _num(candidate.get("confidence_score")),
            "stability": stability_status,
            "calculated_at": created_at,
            # Display convenience (짧은 표에 재사용, 내부 ID는 미포함).
            "product_name": candidate.get("product_name"),
            "source_name": candidate.get("source_name") or candidate.get("source_id"),
            "target_name": candidate.get("target_name") or candidate.get("target_id"),
            "route_type": candidate.get("route_type"),
            "recommended_qty": _num(candidate.get("recommended_qty")),
        })
    return records


def ledger_by_candidate(ledger: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("candidate_id")): dict(record) for record in ledger}


def ledger_by_route(ledger: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("route_id")): dict(record) for record in ledger}


def excluded_candidates(ledger: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Blocked + check-needed candidates (추천에서 빠졌지만 사라지지 않는 후보)."""
    return [
        dict(record) for record in ledger
        if record.get("blocks_recommendation") or record.get("status") == STATUS_CHECK_NEEDED
    ]


def ledger_summary(ledger: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Status counts + top exclusion reasons for the 분석·검증 status cards.

    The buckets are additive: 추천 후보(추천+추천 가능) + 확인 필요 +
    이동 불가 + 데이터 부족 + 계산 불가 == 전체 생성 후보 수.
    """
    counts = {
        STATUS_RECOMMENDED: 0, STATUS_RECOMMENDABLE: 0, STATUS_CHECK_NEEDED: 0,
        STATUS_BLOCKED_MOVE: 0, STATUS_INSUFFICIENT: 0, STATUS_NOT_COMPUTABLE: 0,
    }
    reason_tally: dict[str, int] = {}
    for record in ledger:
        counts[record.get("status")] = counts.get(record.get("status"), 0) + 1
        if record.get("blocks_recommendation"):
            reason = record.get("short_reason") or "확인 필요"
            reason_tally[reason] = reason_tally.get(reason, 0) + 1
    top_reasons = sorted(reason_tally.items(), key=lambda item: item[1], reverse=True)[:3]
    recommend_total = counts[STATUS_RECOMMENDED] + counts[STATUS_RECOMMENDABLE]
    return {
        "generated": len(ledger),
        "recommended": counts[STATUS_RECOMMENDED],
        "recommendable_total": recommend_total,
        "check_needed": counts[STATUS_CHECK_NEEDED],
        "blocked_move": counts[STATUS_BLOCKED_MOVE],
        "insufficient_data": counts[STATUS_INSUFFICIENT],
        "not_computable": counts[STATUS_NOT_COMPUTABLE],
        "excluded_total": counts[STATUS_BLOCKED_MOVE] + counts[STATUS_INSUFFICIENT] + counts[STATUS_NOT_COMPUTABLE],
        "status_counts": counts,
        "top_exclusion_reasons": [{"reason": reason, "count": count} for reason, count in top_reasons],
    }


# --------------------------------------------------------------------------- #
# Review CSV (제외·확인 필요 후보 검토용, 제출 보고서 아님)
# --------------------------------------------------------------------------- #
_REVIEW_COLUMNS = [
    "상태", "출발 점포", "도착 점포", "상품", "권장 수량", "경로 유형",
    "핵심 이유", "파일명", "시트명", "관련 원본 행", "수정 또는 확인 방법",
]

_ROUTE_TYPE_LABELS = {"DIRECT": "직접 이동", "VIA_DC": "DC 경유"}


def _qty_text(value: Any) -> str:
    number = _num(value)
    return "확인 필요" if number is None else f"{number:,.0f}개"


def _primary_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    references = record.get("source_references") or []
    for ref in references:
        if ref.get("traceable"):
            return ref
    return references[0] if references else {}


def review_candidates_csv_bytes(ledger: Sequence[Mapping[str, Any]]) -> bytes:
    """UTF-8 BOM CSV of excluded/check candidates for real data review.

    No internal ids, paths, tracebacks, session keys, or model paths — only what
    a user needs to open the file and fix the cell.
    """
    rows: list[dict[str, str]] = []
    for record in excluded_candidates(ledger):
        reference = _primary_reference(record)
        reasons = record.get("exclusion_reasons") or [record.get("short_reason") or "-"]
        rows.append({
            "상태": record.get("status", "-"),
            "출발 점포": str(record.get("source_name") or "-"),
            "도착 점포": str(record.get("target_name") or "-"),
            "상품": str(record.get("product_name") or "-"),
            "권장 수량": _qty_text(record.get("recommended_qty")),
            "경로 유형": _ROUTE_TYPE_LABELS.get(str(record.get("route_type")), str(record.get("route_type") or "-")),
            "핵심 이유": " / ".join(reasons),
            "파일명": str(reference.get("file") or "-"),
            "시트명": str(reference.get("sheet_name") or "-"),
            "관련 원본 행": ", ".join(str(r) for r in reference.get("rows") or []) if reference.get("traceable") else "추적 불가",
            "수정 또는 확인 방법": reasons[-1] if reasons else "-",
        })
    frame = pd.DataFrame(rows, columns=_REVIEW_COLUMNS) if rows else pd.DataFrame(columns=_REVIEW_COLUMNS)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")
