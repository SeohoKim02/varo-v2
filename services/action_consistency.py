"""Canonical action contract computed next to the production action (it never replaces it).

The production row carries one action label, ``varo_action`` (shown as 'Varo 추천'), but a transfer candidate has four
different kinds of action information, and that one label mixes them:

  recommendation  suggestions to consider. The legacy VHS rule label (varo_hybrid_score._recommend_action, judged on the
                  source inventory row; its transfer branch is unreachable on real data), the legacy promotion-vs-transfer
                  rule (promotion_analyzer, placeholder discount/uplift) and the Seller Loss min-loss comparison
                  (parallel; promoted by nobody).
  rank            the Varo Final order (varo_final_rank, the production Top-5 slice): a priority, not a decision.
  decision        the shared-feasibility selection (PipelineResult.shared_feasibility_selection). A move is a plan only
                  when it is selected with allocated_qty > 0. STRICT_ACTUAL = validated caps, BENCHMARK_PROXY = PROXY.
  execution       what the operator actually did. Only an explicit, observed EXECUTED record (seller outcome contract).

Rules:
* every layer keeps its own source, basis and status; none overwrites another, and the production fields are not touched.
* the canonical action is decision-backed only: TRANSFER with the allocated quantity of the selected plan, or the
  operator's recorded action once executed. A candidate that is not selected never becomes a discount or a hold.
* one plan backs the canonical actions: STRICT_ACTUAL when it had validated caps to decide with, else BENCHMARK_PROXY.
  Mixing the two plans could exceed the shared caps, so the other plan is recorded per row but never displayed as a plan.
* a PROXY plan is never executable; nothing is EXECUTED without an operator record; Seller Loss is never applied.
* labels map by exact text only (``normalize_action`` substring matching turns '이동 비추천' into '재고 이동').
* disagreements get deterministic conflict codes: REVIEW_REQUIRED when independent sources differ, ACTION_CONFLICT when a
  display contradicts the computed result. No winner is picked.
* Korean sentences are templates over computed fields (no LLM, no invented value or unit).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from services import shared_feasibility_selection as sf
from services.optimality_gap_service import _identifier, _number
from services.recommendation_adapter import _ACTION_ALIASES
from services.seller_loss_engine import (DISCOUNT_SALE, NORMAL_SALE, STATUS_FULL, STATUS_UNAVAILABLE, STRATEGIES,
                                         STRATEGY_LABELS, TRANSFER, _josa, _qty)
from services.seller_loss_inputs import NOT_ROBUST, RECOMMENDABLE, SCENARIO_ONLY
from services.seller_loss_promotion_gate import PROMOTION_ELIGIBLE, _nonproduction
from services.seller_shadow_outcomes import OUTCOME_PROVENANCE, parse_seller_outcomes

CONTRACT_VERSION = "action-consistency-0.1"
EPS = 1e-9

# ------------------------------------------------------------------------------------------------ action codes

URGENT_DISCOUNT, BUNDLE_PROMOTION, PROMOTION, DISPOSE, HOLD = (
    "URGENT_DISCOUNT", "BUNDLE_PROMOTION", "PROMOTION", "DISPOSE", "HOLD")
ACTION_CATALOG: dict[str, dict[str, Any]] = {
    TRANSFER: {"label_ko": "재고 이동", "seller_loss_compared": True,
               "producers": "shared_feasibility_selection selected rows (plan); seller_loss TRANSFER; legacy rule '재배치 이동' "
                            "(unreachable: match_score max 57.5 < 60); promotion_analyzer '재배치 추천'; upload aliases; "
                            "DQN labels (button only)"},
    DISCOUNT_SALE: {"label_ko": "할인 판매", "seller_loss_compared": True,
                    "producers": "legacy rule '할인 판매' (normalized '할인'); seller_loss DISCOUNT_SALE (label '할인')"},
    NORMAL_SALE: {"label_ko": "정상 판매 유지", "seller_loss_compared": True,
                  "producers": "seller_loss NORMAL_SALE only; operator action. No legacy rule emits it"},
    HOLD: {"label_ko": "보류", "seller_loss_compared": False,
           "producers": "legacy rule fall-through (no disposal/transfer/discount condition met: monitoring); upload aliases "
                        "hold/keep_inventory/no_action/maintain/모니터링. Not NORMAL_SALE: it does not say keep selling"},
    DISPOSE: {"label_ko": "폐기", "seller_loss_compared": False,
              "producers": "legacy rule (disposal CRITICAL, turnover DEAD, ABC C, demand_risk < 40); upload aliases; DQN labels"},
    URGENT_DISCOUNT: {"label_ko": "긴급 할인", "seller_loss_compared": False,
                      "producers": "upload aliases and DQN labels only; the production pipeline never emits it"},
    BUNDLE_PROMOTION: {"label_ko": "1+1", "seller_loss_compared": False,
                       "producers": "upload aliases and DQN labels only; the production pipeline never emits it"},
    PROMOTION: {"label_ko": "프로모션", "seller_loss_compared": False,
                "producers": "promotion_analyzer '프로모션 추천' (type from config promotion_type; placeholder discount/uplift)"},
}
_LABEL_CODES = {
    "재고 이동": TRANSFER, "재배치 이동": TRANSFER, "재배치 추천": TRANSFER, "직접 이동": TRANSFER, "DC 경유 이동": TRANSFER,
    "할인": DISCOUNT_SALE, "할인 판매": DISCOUNT_SALE, "정상 판매 유지": NORMAL_SALE, "정상 판매": NORMAL_SALE,
    "보류": HOLD, "폐기": DISPOSE, "긴급 할인": URGENT_DISCOUNT, "긴급할인": URGENT_DISCOUNT, "1+1": BUNDLE_PROMOTION,
    "프로모션 추천": PROMOTION,
}
# Exact keys of the existing alias table, never its substring matching.
LABEL_TO_CODE = {**{code: code for code in ACTION_CATALOG}, **_LABEL_CODES,
                 **{alias: _LABEL_CODES[label] for alias, label in _ACTION_ALIASES.items() if label in _LABEL_CODES}}
SENTINEL_LABELS = {
    "비교 불가": "normalize_action default for a missing label; not an action",
    "미연결": "DQN not connected (pipeline sentinel)", "학습 필요": "DQN status sentinel",
    "유지": "ambiguous (keep selling or wait); unmappable in seller_loss_promotion_gate",
}

# ------------------------------------------------------------------------------------------------ statuses

EXECUTED, FEASIBLE_PLAN, PROXY_PLAN = "EXECUTED", "FEASIBLE_PLAN", "PROXY_PLAN"
SELECTED_PLAN, NOT_SELECTED, INSUFFICIENT_DATA, COMPARISON_UNAVAILABLE = (
    "SELECTED_PLAN", "NOT_SELECTED", "INSUFFICIENT_DATA", "COMPARISON_UNAVAILABLE")
RECOMMENDED, NOT_RECORDED = "RECOMMENDED", "NOT_RECORDED"
ACTION_STATUSES = (EXECUTED, FEASIBLE_PLAN, PROXY_PLAN, NOT_SELECTED, INSUFFICIENT_DATA, COMPARISON_UNAVAILABLE)
CONSISTENT, REVIEW_REQUIRED, ACTION_CONFLICT, NOT_COMPARABLE = "CONSISTENT", "REVIEW_REQUIRED", "ACTION_CONFLICT", "NOT_COMPARABLE"
EXACT_MATCH, SEMANTIC_MATCH, MISMATCH = "EXACT_MATCH", "SEMANTIC_MATCH", "MISMATCH"
UNMAPPABLE, INSUFFICIENT_EVIDENCE, DIFFERENT_PURPOSE = "UNMAPPABLE", "INSUFFICIENT_EVIDENCE", "DIFFERENT_PURPOSE"
ALIGNMENTS = (EXACT_MATCH, SEMANTIC_MATCH, MISMATCH, UNMAPPABLE, INSUFFICIENT_EVIDENCE, DIFFERENT_PURPOSE)
STATUS_DEFINITIONS = {
    EXECUTED: "the operator recorded this action as executed (observed provenance and source)",
    FEASIBLE_PLAN: "selected by STRICT_ACTUAL: allocated quantity fits validated source/target caps",
    PROXY_PLAN: "selected by BENCHMARK_PROXY only: caps are PROXY or missing; a calculated plan, not an operational guarantee",
    NOT_SELECTED: "evaluated by the shared-feasibility selection and not selected (cap, duplicate, limit or infeasible flag)",
    INSUFFICIENT_DATA: "not decidable: required caps not validated, unit or cost basis unknown, or invalid input",
    COMPARISON_UNAVAILABLE: "no shared-feasibility result linked to this candidate",
    RECOMMENDED: "a recommendation source suggests the action (to consider; not a decision)",
}

PROVENANCE_SOURCES = {
    "VARO_FINAL": "vhs_score_engine._rank_varo_operational order and the production Top-5 slice (priority, not a decision)",
    "SHARED_FEASIBILITY": "shared_feasibility_selection plan rows (decision)",
    "LEGACY_RULE": "varo_hybrid_score._recommend_action via normalize_action (source inventory-state rule)",
    "UPLOADED_LABEL": "uploaded or default label passed through because the legacy rule did not run",
    "LEGACY_PROMOTION_RULE": "promotion_analyzer.analyze_promotion_vs_transfer final_decision (placeholder assumptions)",
    "PARETO": "pareto_service.select_pareto_routes (parallel strategy, recorded only)",
    "SELLER_LOSS": "seller_loss_engine comparison of TRANSFER / NORMAL_SALE / DISCOUNT_SALE (parallel, never applied)",
    "OPERATOR_CONFIRMED": "explicit observed operator outcome record (seller outcome contract)",
    "USER_POLICY": "reserved: no current code path produces an action policy",
    "UNKNOWN": "label without a traceable producer",
}

# code -> (audit type, level, description)
CONFLICTS: dict[str, tuple[str, str, str]] = {
    "A_TRANSFER_PLAN_VS_LEGACY_DISCOUNT": ("A", REVIEW_REQUIRED, "transfer plan selected; the legacy rule label says discount"),
    "A_TRANSFER_PLAN_VS_LEGACY_OTHER": ("A", REVIEW_REQUIRED, "transfer plan selected; the legacy rule label says dispose / urgent discount / 1+1 / promotion"),
    "B_TRANSFER_PLAN_VS_LEGACY_HOLD": ("B", REVIEW_REQUIRED, "transfer plan selected; the legacy rule label says hold"),
    "C_LEGACY_TRANSFER_LABEL_NOT_SELECTED": ("C", ACTION_CONFLICT, "'Varo 추천' shows 재고 이동 but the move is not in the plan"),
    "C_FINAL_RECOMMENDATION_NOT_SELECTED": ("C", ACTION_CONFLICT, "varo_final_decision '최종 추천' but the move is not in the plan"),
    "C_LEGACY_TOP5_NOT_SELECTED": ("C", ACTION_CONFLICT, "in the production Top-5 slice but not in the plan"),
    "C_PROMOTION_RULE_TRANSFER_NOT_SELECTED": ("C", REVIEW_REQUIRED, "promotion rule says '재배치 추천' but the move is not in the plan"),
    "D_TRANSFER_PLAN_WITHOUT_QUANTITY": ("D", ACTION_CONFLICT, "a selection without a positive allocated quantity"),
    "D_ALLOCATION_EXCEEDS_CAP": ("D", ACTION_CONFLICT, "allocated quantity above an applied cap"),
    "E_KEY_MISMATCH": ("E", ACTION_CONFLICT, "route/product/source/target differ between linked layers"),
    "F_PROXY_PLAN_CLAIMED_EXECUTABLE": ("F", ACTION_CONFLICT, "a PROXY plan is marked or worded as executable"),
    "G_MIN_LOSS_CLAIM_WITHOUT_COMPARISON": ("G", ACTION_CONFLICT, "a min-loss claim without a recommendable Seller Loss comparison"),
    "H_EXECUTED_WITHOUT_OPERATOR_RECORD": ("H", ACTION_CONFLICT, "executed status without an operator record"),
    "I_LEGACY_VS_CANONICAL_MISMATCH": ("I", REVIEW_REQUIRED, "the production label and the canonical action differ"),
    "J_PRODUCT_SUMMARY_DOUBLE_COUNT": ("J", ACTION_CONFLICT, "product summary total or route list does not equal its route plans"),
    "PLAN_VS_SELLER_LOSS_DIFFERENT": ("SL", REVIEW_REQUIRED, "the plan and a recommendable Seller Loss comparison differ"),
}
REASON_CODES = {
    "PROXY_CAPS_APPLIED": "an applied cap is PROXY (not validated)",
    "REQUIRED_CAP_NOT_VALIDATED": "source or target cap missing in the backing plan",
    "ROUTE_CAPACITY_NOT_CHECKED": "no route/vehicle capacity in quantity units",
    "STRICT_INSUFFICIENT_CAP_DATA": "STRICT_ACTUAL could not decide this candidate (caps not validated)",
    "SELECTED_ONLY_UNDER_PROXY_CAPS": "BENCHMARK_PROXY selects it, the validated STRICT_ACTUAL plan does not",
    "QTY_PARTIAL_ALLOCATION": "allocated quantity is below the recommended quantity",
    "SELLER_LOSS_QTY_BASIS_DIFFERS": "Seller Loss compared the recommended quantity, not the allocated one",
    "LEGACY_LABEL_PASS_THROUGH": "the legacy rule did not run; varo_action is an uploaded/default label",
    "NONPRODUCTION_DATA": "upload marked as TEST/SAMPLE/synthetic: no executable claim",
    "PLAN_LINK_MISSING": "no shared-feasibility row for this candidate",
    "SHARED_FEASIBILITY_UNAVAILABLE": "the shared-feasibility field is missing or failed",
    "NOT_EXECUTED_NO_OPERATOR_RECORD": "no operator execution record",
    "EXECUTED_ACTION_DIFFERS_FROM_PLAN": "the operator executed an action other than the plan",
    "PARETO_SELECTED": "Pareto strategy selects this route (recorded only)",
    "SELLER_LOSS_PROMOTION_ELIGIBLE_NOT_APPLIED": "Promotion Gate eligible; production action unchanged by design",
}
GUARANTEE_TERMS = ("보장", "확정", "실행 완료")
_SELECTED = (sf.SELECTED, sf.PARTIALLY_SELECTED)
_DATA_STATUSES = (sf.INSUFFICIENT_CAP_DATA, sf.UNIT_MISMATCH, sf.UNKNOWN_UNIT, sf.REJECTED_INVALID_INPUT, sf.COST_NOT_COMPARABLE)
_STATUS_KO = {
    sf.REJECTED_SOURCE_CAP: "출발 점포 이동 가능 재고 소진", sf.REJECTED_TARGET_CAP: "도착 점포 필요량 소진",
    sf.REJECTED_ROUTE_CAP: "경로 적재 한도 초과", sf.REJECTED_DC_CAP: "DC 처리 한도 초과",
    sf.REJECTED_DUPLICATE: "같은 경로가 이미 선택됨", sf.REJECTED_SELECTION_LIMIT: "선택 한도 도달",
    sf.REJECTED_INFEASIBLE_ROUTE: "경로 실행 불가 표시", sf.COST_NOT_COMPARABLE: "부분 수량 운송비 산정 불가",
    sf.INSUFFICIENT_CAP_DATA: "검증된 재고·수요 기준 부족", sf.UNIT_MISMATCH: "수량 단위 불일치",
    sf.UNKNOWN_UNIT: "수량 단위 미확인", sf.REJECTED_INVALID_INPUT: "입력값 오류",
}
_UNIT_KO = {"EA": "개", "BOX": "박스", "KG": "kg", "TON": "톤"}
CONTEXT_KEYS = ("is_sample", "is_test", "synthetic_fixture", "data_kind", "source_file", "source_path", "label")


def map_action_label(value: Any) -> tuple[str | None, str]:
    """Exact label -> canonical code. Mapping status: CANONICAL | ALIAS | SENTINEL | UNMAPPABLE | EMPTY."""
    text = _identifier(value)
    if not text:
        return None, "EMPTY"
    if text in SENTINEL_LABELS:
        return None, "SENTINEL"
    code = LABEL_TO_CODE.get(text) or LABEL_TO_CODE.get(text.lower())
    if code is None:
        return None, UNMAPPABLE
    return code, "CANONICAL" if text in (code, ACTION_CATALOG[code]["label_ko"]) else "ALIAS"


def _label(code: str | None) -> str | None:
    return ACTION_CATALOG[code]["label_ko"] if code in ACTION_CATALOG else None


def _qty_text(value: float | None, unit: str | None) -> str:
    unit = (unit or "").upper()
    if unit in _UNIT_KO:
        return f"{_qty(value)}{_UNIT_KO[unit]}"
    return f"{_qty(value)} {unit}" if unit not in ("", "UNDECLARED") else f"{_qty(value)}(단위 미확인)"


def _provenance_pairs(text: Any) -> dict[str, str]:
    pairs = (part.split("=", 1) for part in str(text or "").split(";") if "=" in part)
    return {kind: provenance for kind, provenance in pairs}


# ------------------------------------------------------------------------------------------------ layers


def primary_mode(shared: Mapping[str, Any] | None) -> tuple[str | None, str]:
    """The one plan that backs canonical actions."""
    modes = (shared or {}).get("modes") or {}
    strict, bench = modes.get(sf.STRICT_ACTUAL), modes.get(sf.BENCHMARK_PROXY)
    if strict and strict.get("plan_status") not in (sf.INSUFFICIENT_CAP_DATA, "EMPTY_INPUT"):
        return sf.STRICT_ACTUAL, f"STRICT_ACTUAL plan_status={strict.get('plan_status')}: validated caps decide"
    if bench:
        status = strict.get("plan_status") if strict else "absent"
        return sf.BENCHMARK_PROXY, f"STRICT_ACTUAL plan_status={status}: BENCHMARK_PROXY plan backs actions, labelled PROXY"
    return None, f"shared-feasibility selection unavailable (status={(shared or {}).get('status')})"


def operator_events_by_route(events: Sequence[Mapping[str, Any]] | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Validate operator records with the seller outcome contract (one snapshot per call), keyed by route_id."""
    linked: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for position, event in enumerate(events or []):
        route = _identifier(event.get("route_id"))
        parsed = parse_seller_outcomes([{**{k: v for k, v in event.items() if k not in ("route_id", "product_id", "source_id",
                                                                                       "target_id")},
                                         "decision_id": route or None}])
        if parsed["errors"] or not parsed["rows"]:
            rejected.append({"position": position, "route_id": route or None,
                             "codes": sorted({item["code"] for item in parsed["errors"]}) or ["EMPTY_EVENT"]})
            continue
        row = parsed["rows"][0]
        row.update({key: _identifier(event.get(key)) for key in ("route_id", "product_id", "source_id", "target_id")})
        linked[route].append(row)
    latest = {route: sorted(rows, key=lambda item: (item["recorded_at"], item["executed_at"] or ""))[-1]
              for route, rows in linked.items()}
    for route, rows in linked.items():
        latest[route]["snapshot_count"] = len(rows)
    return latest, rejected


def _execution(record: Mapping[str, Any], event: Mapping[str, Any] | None) -> dict[str, Any]:
    if event is None:
        return {"execution_state": NOT_RECORDED, "operator_action": None, "operator_action_code": None, "executed_at": None,
                "operator_evidence": None, "execution_reasons": ["NOT_EXECUTED_NO_OPERATOR_RECORD"]}
    mismatch = [key for key in ("product_id", "source_id", "target_id")
                if event.get(key) and event[key] != record.get(key)]
    if mismatch:
        return {"execution_state": NOT_RECORDED, "operator_action": None, "operator_action_code": None, "executed_at": None,
                "operator_evidence": None, "execution_reasons": [f"OPERATOR_EVENT_KEY_MISMATCH:{'|'.join(mismatch)}"]}
    state, mode = event["execution_status"], event["data_mode"]
    evidence = {"provenance": event["outcome_provenance"], "source": event["outcome_source"],
                "recorded_at": event["recorded_at"], "data_mode": mode, "snapshots": event.get("snapshot_count", 1)}
    if state != EXECUTED or mode != "PRODUCTION" or event["outcome_provenance"] not in OUTCOME_PROVENANCE[:3]:
        reason = f"OPERATOR_RECORD_{state}" if mode == "PRODUCTION" else f"OPERATOR_RECORD_NONPRODUCTION:{mode}"
        return {"execution_state": state if mode == "PRODUCTION" else NOT_RECORDED, "operator_action": None,
                "operator_action_code": None, "executed_at": None, "operator_evidence": evidence, "execution_reasons": [reason]}
    code, _ = map_action_label(event["operator_action"])
    return {"execution_state": EXECUTED, "operator_action": event["operator_action"], "operator_action_code": code,
            "executed_at": event["executed_at"], "operator_evidence": evidence, "execution_reasons": []}


def _seller_layer(decision: Mapping[str, Any] | None, shadow: Mapping[str, Any] | None) -> dict[str, Any]:
    if decision is None:
        return {"seller_loss_action": None, "seller_loss_strategy": None, "seller_loss_comparison_status": None,
                "seller_loss_readiness": None, "seller_loss_evidence_level": None, "seller_loss_promotion_status": None,
                "seller_loss_decision_qty": None, "seller_loss_claim": "NOT_EVALUATED", "seller_loss_min_loss_claim": False,
                "seller_loss_nonproduction_evidence": [], "seller_loss_production_action_applied": False,
                "seller_loss_excluded_actions": None}
    shadow = shadow or {}
    readiness, status = decision.get("recommendation_readiness"), decision.get("comparison_status")
    nonproduction = list(shadow.get("nonproduction_evidence") or [])
    action = decision.get("seller_loss_action")
    if readiness == RECOMMENDABLE and action:
        claim = "MIN_LOSS_NONPRODUCTION_INPUT" if nonproduction else (
            "MIN_LOSS_FULL" if status == STATUS_FULL else "MIN_LOSS_PARTIAL_SCOPE")
    else:
        claim = {SCENARIO_ONLY: "SCENARIO_ONLY", NOT_ROBUST: "NOT_ROBUST"}.get(readiness, "NO_COMPARISON")
    return {"seller_loss_action": action, "seller_loss_strategy": decision.get("recommended_strategy"),
            "seller_loss_comparison_status": status, "seller_loss_readiness": readiness,
            "seller_loss_evidence_level": decision.get("evidence_level"),
            "seller_loss_promotion_status": shadow.get("promotion_status"),
            "seller_loss_decision_qty": _number(decision.get("decision_qty")),
            "seller_loss_claim": claim, "seller_loss_min_loss_claim": claim in ("MIN_LOSS_FULL", "MIN_LOSS_PARTIAL_SCOPE"),
            "seller_loss_nonproduction_evidence": nonproduction,
            "seller_loss_production_action_applied": bool(shadow.get("production_action_applied", False)),
            # Seller Loss compares only these three; anything else is outside its scope (never merged into DISCOUNT_SALE).
            "seller_loss_excluded_actions": [code for code, item in ACTION_CATALOG.items() if not item["seller_loss_compared"]]}


def _link(record: Mapping[str, Any], row: Mapping[str, Any] | None, fields: Mapping[str, str]) -> str:
    if row is None:
        return "MISSING"
    bad = [mine for mine, theirs in fields.items() if _identifier(row.get(theirs)) != (record.get(mine) or "")]
    return "OK" if not bad else "MISMATCH:" + "|".join(bad)


_SF_FIELDS = {"route_id": "route_id", "product_id": "product_id", "source_id": "source_id", "target_id": "target_id"}
_SL_FIELDS = {"route_id": "decision_id", "product_id": "product_id", "source_id": "source_store_id", "target_id": "target_store_id"}


def _route_record(index: int, rec: Mapping[str, Any], *, primary: str | None, rows: Mapping[str, Mapping[str, dict]],
                  plans: Mapping[str, Mapping[str, Any]], seller: Mapping[str, Mapping[str, Any]],
                  shadow: Mapping[str, Mapping[str, Any]], legacy_top: set[str], pareto: set[str],
                  legacy_source: str, nonproduction_data: list[str], event: Mapping[str, Any] | None) -> dict[str, Any]:
    cid = sf.candidate_identity(rec, index)
    record: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION, "decision_scope": "ROUTE", "candidate_id": cid,
        **{key: _identifier(rec.get(key)) or None for key in ("route_id", "product_id", "source_id", "target_id", "dc_id")},
        "route_type": _identifier(rec.get("route_type")).upper() or None,
        **{key: _identifier(rec.get(key)) or None for key in ("product_name", "source_name", "target_name")},
        "recommended_qty": _number(rec.get("recommended_qty")),
    }
    reasons: list[str] = []
    strict_row, bench_row = rows.get(sf.STRICT_ACTUAL, {}).get(cid), rows.get(sf.BENCHMARK_PROXY, {}).get(cid)
    row = rows.get(primary, {}).get(cid) if primary else None
    decision = seller.get(cid)
    record["links"] = {"shared_feasibility": _link(record, row, _SF_FIELDS) if primary else "UNAVAILABLE",
                       "seller_loss": _link(record, decision, _SL_FIELDS) if seller else "UNAVAILABLE"}
    status = row.get("selection_status") if row else None
    allocated = float(row.get("allocated_qty") or 0.0) if row else 0.0
    if primary is None:
        state = COMPARISON_UNAVAILABLE
        reasons.append("SHARED_FEASIBILITY_UNAVAILABLE")
    elif row is None or record["links"]["shared_feasibility"] != "OK":
        state = COMPARISON_UNAVAILABLE
        reasons.append("PLAN_LINK_MISSING" if row is None else "PLAN_LINK_KEY_MISMATCH")
    elif status in _SELECTED and allocated > EPS:
        state = SELECTED_PLAN
    elif status in _DATA_STATUSES:
        state = INSUFFICIENT_DATA
    else:
        state = NOT_SELECTED
    provenance = _provenance_pairs(row.get("cap_provenance") if row else None)
    caps = {kind: row.get(column) if row else None for kind, column in (
        ("source_stock", "source_stock_cap"), ("source_surplus", "source_surplus_cap"), ("target_need", "target_need_cap"),
        ("route_capacity", "route_capacity_cap"), ("dc_capacity", "dc_capacity_cap"))}
    if state == SELECTED_PLAN:
        missing = {kind for kind in sf.CAP_KINDS if provenance.get(kind, "MISSING") == "MISSING"}
        if any(value not in sf.STRICT_ACCEPTED_PROVENANCE | {"MISSING"} for value in provenance.values()):
            reasons.append("PROXY_CAPS_APPLIED")
        if sf.TARGET_NEED in missing or set(sf.SOURCE_KINDS) <= missing:
            reasons.append("REQUIRED_CAP_NOT_VALIDATED")
        if sf.ROUTE_CAPACITY in missing:
            reasons.append("ROUTE_CAPACITY_NOT_CHECKED")
        if record["recommended_qty"] is not None and allocated < record["recommended_qty"] - EPS:
            reasons.append("QTY_PARTIAL_ALLOCATION")
    if primary == sf.BENCHMARK_PROXY and strict_row and strict_row.get("selection_status") == sf.INSUFFICIENT_CAP_DATA:
        reasons.append("STRICT_INSUFFICIENT_CAP_DATA")
    if primary == sf.STRICT_ACTUAL and state != SELECTED_PLAN and bench_row and bench_row.get("selection_status") in _SELECTED:
        reasons.append("SELECTED_ONLY_UNDER_PROXY_CAPS")
    feasibility = (FEASIBLE_PLAN if primary == sf.STRICT_ACTUAL else PROXY_PLAN) if state == SELECTED_PLAN else None
    if nonproduction_data:
        reasons.append("NONPRODUCTION_DATA")
    if legacy_source == "UPLOADED_LABEL":
        reasons.append("LEGACY_LABEL_PASS_THROUGH")
    if cid in pareto:
        reasons.append("PARETO_SELECTED")
    plan = plans.get(primary) or {}
    record.update({
        "decision_state": state, "feasibility_status": feasibility,
        "selection_mode": primary if state == SELECTED_PLAN else None,
        "selected_route_id": record["route_id"] if state == SELECTED_PLAN else None,
        "allocated_qty": round(allocated, 6) if state == SELECTED_PLAN else 0.0,
        "quantity_unit": (row.get("quantity_unit") if row else None) or "UNDECLARED",
        "shared_feasibility_rank": row.get("shared_feasibility_rank") if row and state == SELECTED_PLAN else None,
        "selection_status": status, "rejection_reason": row.get("rejection_reason") if row else None,
        "executable_qty_at_end": row.get("executable_qty_at_end") if row and state != SELECTED_PLAN else None,
        "rejection_category": row.get("rejection_category") if row else None,
        "strict_selection_status": strict_row.get("selection_status") if strict_row else None,
        "strict_allocated_qty": strict_row.get("allocated_qty") if strict_row else None,
        "benchmark_selection_status": bench_row.get("selection_status") if bench_row else None,
        "benchmark_allocated_qty": bench_row.get("allocated_qty") if bench_row else None,
        "feasibility_claim": plan.get("feasibility_claim"), "plan_status": plan.get("plan_status"),
        "cap_provenance": row.get("cap_provenance") if row else None, "caps": caps,
        "allocated_move_cost": row.get("allocated_move_cost") if row and state == SELECTED_PLAN else None,
        "cost_status": row.get("cost_status") if row else None,
    })
    legacy_raw = _identifier(rec.get("varo_action")) or None
    legacy_code, legacy_mapping = map_action_label(legacy_raw)
    promotion_raw = _identifier(rec.get("promotion_recommended")) or None
    promotion_code, _ = map_action_label(promotion_raw)
    record.update({
        "legacy_action": legacy_raw, "legacy_action_code": legacy_code, "legacy_action_mapping": legacy_mapping,
        "legacy_action_source": legacy_source, "production_action": legacy_raw, "production_action_applied": False,
        "legacy_promotion_decision": promotion_raw, "legacy_promotion_code": promotion_code,
        "varo_final_rank": _number(rec.get("varo_final_rank")), "varo_final_decision": _identifier(rec.get("varo_final_decision")) or None,
        "in_legacy_top5": cid in legacy_top,
        "varo_final_status": "TOP5_RANKED" if cid in legacy_top else "RANKED" if _number(rec.get("varo_final_rank")) else "UNRANKED",
        "pareto_selected": cid in pareto,
        **_seller_layer(decision if record["links"]["seller_loss"] == "OK" else None, shadow.get(cid)),
    })
    if record["seller_loss_promotion_status"] == PROMOTION_ELIGIBLE:
        reasons.append("SELLER_LOSS_PROMOTION_ELIGIBLE_NOT_APPLIED")
    if (state == SELECTED_PLAN and record["seller_loss_decision_qty"] is not None
            and abs(record["seller_loss_decision_qty"] - allocated) > EPS):
        reasons.append("SELLER_LOSS_QTY_BASIS_DIFFERS")
    execution = _execution(record, event)
    reasons.extend(execution.pop("execution_reasons"))
    record.update(execution)
    record["reason_codes"] = sorted(set(reasons))
    if record["execution_state"] == EXECUTED:
        code, label = record["operator_action_code"], record["operator_action"]
        source, basis = "OPERATOR_CONFIRMED", f"operator record {record['operator_evidence']['provenance']}: {record['operator_evidence']['source']}"
        status_out = EXECUTED
        if state == SELECTED_PLAN and code != TRANSFER:
            record["reason_codes"] = sorted(set(record["reason_codes"]) | {"EXECUTED_ACTION_DIFFERS_FROM_PLAN"})
    elif state == SELECTED_PLAN:
        code, label, source = TRANSFER, _label(TRANSFER), "SHARED_FEASIBILITY"
        basis, status_out = f"{primary} selection, allocated_qty {round(allocated, 6)}", feasibility
    else:
        code, source, basis, status_out = None, None, f"no selected plan ({state})", state
        label = {NOT_SELECTED: "이동 계획 없음(미선택)", INSUFFICIENT_DATA: "이동 판단 불가(근거 부족)",
                 COMPARISON_UNAVAILABLE: "이동 판단 불가(선택 결과 없음)"}[state]
    if status_out == EXECUTED:
        evidence = "OPERATOR_CONFIRMED"
    elif feasibility == FEASIBLE_PLAN:
        evidence = "VALIDATED_CAPS"
    elif feasibility == PROXY_PLAN:
        evidence = "PROXY_CAPS" if "PROXY_CAPS_APPLIED" in record["reason_codes"] else "UNVALIDATED_CAPS"
    else:
        evidence = "NO_PLAN"
    record.update({
        "action_code": code, "action_label_ko": label, "action_source": source, "action_basis": basis, "action_status": status_out,
        "evidence_level": evidence,
        "is_executable": feasibility == FEASIBLE_PLAN and not nonproduction_data,
        "is_executed": status_out == EXECUTED,
        "recommendation_state": RECOMMENDED if (legacy_code or promotion_code or record["seller_loss_min_loss_claim"]) else "NONE",
    })
    record["action_provenance"] = _provenance(record)
    return record


def _provenance(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every layer that says something about this candidate, each with its own role, code and status."""
    items = [{"source": "VARO_FINAL", "role": "RANK", "code": None, "status": record["varo_final_status"],
              "detail": f"rank {record['varo_final_rank']}"}]
    items.append({"source": "SHARED_FEASIBILITY", "role": "DECISION",
                  "code": TRANSFER if record["decision_state"] == SELECTED_PLAN else None,
                  "status": record["feasibility_status"] or record["decision_state"],
                  "detail": f"primary={record['selection_mode'] or '-'}; strict={record['strict_selection_status']}; "
                            f"benchmark={record['benchmark_selection_status']}"})
    if record["legacy_action"]:
        items.append({"source": record["legacy_action_source"], "role": "RECOMMENDATION", "code": record["legacy_action_code"],
                      "status": RECOMMENDED if record["legacy_action_code"] else UNMAPPABLE, "detail": record["legacy_action"]})
    if record["legacy_promotion_decision"]:
        items.append({"source": "LEGACY_PROMOTION_RULE", "role": "RECOMMENDATION", "code": record["legacy_promotion_code"],
                      "status": RECOMMENDED, "detail": f"{record['legacy_promotion_decision']} (placeholder assumptions)"})
    if record["pareto_selected"]:
        items.append({"source": "PARETO", "role": "ALTERNATIVE_SELECTION", "code": TRANSFER, "status": "SELECTED_BY_STRATEGY",
                      "detail": "parallel strategy; not the canonical plan"})
    if record["seller_loss_claim"] != "NOT_EVALUATED":
        items.append({"source": "SELLER_LOSS", "role": "COMPARISON", "code": record["seller_loss_strategy"],
                      "status": RECOMMENDED if record["seller_loss_min_loss_claim"] else record["seller_loss_claim"],
                      "detail": f"{record['seller_loss_comparison_status']}/{record['seller_loss_readiness']}; "
                                f"promotion={record['seller_loss_promotion_status']}; applied=False"})
    if record["execution_state"] != NOT_RECORDED or record["operator_evidence"]:
        items.append({"source": "OPERATOR_CONFIRMED", "role": "EXECUTION", "code": record["operator_action_code"],
                      "status": record["execution_state"], "detail": record["operator_action"]})
    return items


# ------------------------------------------------------------------------------------------------ conflicts


def legacy_alignment(record: Mapping[str, Any]) -> str:
    """Production label vs canonical action, by meaning (not string equality)."""
    legacy, canonical = record.get("legacy_action_code"), record.get("action_code")
    if legacy is None:
        return UNMAPPABLE
    if canonical is None:
        if record.get("decision_state") in (INSUFFICIENT_DATA, COMPARISON_UNAVAILABLE):
            return INSUFFICIENT_EVIDENCE
        # The legacy rule judges the source inventory; with no plan there is no route action to compare it with.
        return MISMATCH if legacy == TRANSFER else DIFFERENT_PURPOSE
    if legacy == canonical:
        return EXACT_MATCH if record.get("legacy_action_mapping") == "CANONICAL" else SEMANTIC_MATCH
    return MISMATCH


def detect_record_conflicts(record: Mapping[str, Any]) -> list[str]:
    """Deterministic conflict codes from the record fields alone (also usable on records from other producers)."""
    codes: list[str] = []
    planned = record.get("decision_state") == SELECTED_PLAN
    no_plan = record.get("decision_state") in (NOT_SELECTED, INSUFFICIENT_DATA)
    legacy = record.get("legacy_action_code")
    if planned:
        if legacy == DISCOUNT_SALE:
            codes.append("A_TRANSFER_PLAN_VS_LEGACY_DISCOUNT")
        elif legacy in (DISPOSE, URGENT_DISCOUNT, BUNDLE_PROMOTION, PROMOTION):
            codes.append("A_TRANSFER_PLAN_VS_LEGACY_OTHER")
        elif legacy == HOLD:
            codes.append("B_TRANSFER_PLAN_VS_LEGACY_HOLD")
    if no_plan:
        if legacy == TRANSFER:
            codes.append("C_LEGACY_TRANSFER_LABEL_NOT_SELECTED")
        if record.get("varo_final_decision") == "최종 추천":
            codes.append("C_FINAL_RECOMMENDATION_NOT_SELECTED")
        if record.get("in_legacy_top5"):
            codes.append("C_LEGACY_TOP5_NOT_SELECTED")
        if record.get("legacy_promotion_code") == TRANSFER:
            codes.append("C_PROMOTION_RULE_TRANSFER_NOT_SELECTED")
    allocated = _number(record.get("allocated_qty")) or 0.0
    if record.get("action_code") == TRANSFER and record.get("action_source") == "SHARED_FEASIBILITY" and allocated <= EPS:
        codes.append("D_TRANSFER_PLAN_WITHOUT_QUANTITY")
    if planned and any(_number(cap) is not None and allocated > float(cap) + 1e-6
                       for cap in (record.get("caps") or {}).values()):
        codes.append("D_ALLOCATION_EXCEEDS_CAP")
    if any(str(value).startswith("MISMATCH") for value in (record.get("links") or {}).values()):
        codes.append("E_KEY_MISMATCH")
    text = " ".join([str(record.get("action_label_ko") or ""), *(record.get("explanation_ko") or [])])
    if record.get("feasibility_status") == PROXY_PLAN and (
            record.get("is_executable") or any(term in text for term in GUARANTEE_TERMS)):
        codes.append("F_PROXY_PLAN_CLAIMED_EXECUTABLE")
    if record.get("seller_loss_min_loss_claim") and (
            record.get("seller_loss_readiness") != RECOMMENDABLE or record.get("seller_loss_comparison_status") in (None, STATUS_UNAVAILABLE)
            or not record.get("seller_loss_action")):
        codes.append("G_MIN_LOSS_CLAIM_WITHOUT_COMPARISON")
    if (record.get("action_status") == EXECUTED or record.get("is_executed")) and not record.get("operator_action"):
        codes.append("H_EXECUTED_WITHOUT_OPERATOR_RECORD")
    if legacy_alignment(record) == MISMATCH:
        codes.append("I_LEGACY_VS_CANONICAL_MISMATCH")
    strategy = record.get("seller_loss_strategy") if record.get("seller_loss_min_loss_claim") else None
    if strategy and ((planned and strategy != TRANSFER) or (no_plan and strategy == TRANSFER)):
        codes.append("PLAN_VS_SELLER_LOSS_DIFFERENT")
    return codes


def consistency_status(record: Mapping[str, Any], codes: Sequence[str]) -> str:
    levels = {CONFLICTS[code][1] for code in codes}
    if ACTION_CONFLICT in levels:
        return ACTION_CONFLICT
    if REVIEW_REQUIRED in levels:
        return REVIEW_REQUIRED
    if record.get("decision_state") == COMPARISON_UNAVAILABLE and record.get("action_status") != EXECUTED:
        return NOT_COMPARABLE
    return CONSISTENT


# ------------------------------------------------------------------------------------------------ Korean text

_CONFLICT_TEXT = {
    "A_TRANSFER_PLAN_VS_LEGACY_DISCOUNT": lambda r: "재고 이동 계획과 기존 할인 권고가 달라 추가 검토가 필요합니다.",
    "A_TRANSFER_PLAN_VS_LEGACY_OTHER": lambda r: f"재고 이동 계획과 기존 {r['legacy_action']} 권고가 달라 추가 검토가 필요합니다.",
    "B_TRANSFER_PLAN_VS_LEGACY_HOLD": lambda r: "재고 이동 계획과 기존 보류 권고가 달라 추가 검토가 필요합니다.",
    "C_LEGACY_TRANSFER_LABEL_NOT_SELECTED": lambda r: "기존 규칙은 재고 이동을 권고했지만 공동 제약 선택에서는 이 경로가 선택되지 않았습니다.",
    "C_FINAL_RECOMMENDATION_NOT_SELECTED": lambda r: "기존 화면의 '최종 추천' 경로이지만 공동 제약 선택에서는 제외됐습니다.",
    "C_LEGACY_TOP5_NOT_SELECTED": lambda r: "기존 Top-5에 포함됐지만 공동 제약 선택에서는 제외된 후보입니다.",
    "C_PROMOTION_RULE_TRANSFER_NOT_SELECTED": lambda r: "기존 프로모션 비교 규칙(고정 가정값)은 재배치를 권고했지만 이 경로는 선택되지 않았습니다.",
    "D_TRANSFER_PLAN_WITHOUT_QUANTITY": lambda r: "배정 수량이 없는 이동은 계획으로 표시하지 않습니다.",
    "D_ALLOCATION_EXCEEDS_CAP": lambda r: "배정 수량이 적용된 재고·수요 한도를 넘습니다.",
    "E_KEY_MISMATCH": lambda r: "추천·선택·금액 비교 결과의 상품·점포·경로가 서로 맞지 않아 연결하지 않았습니다.",
    "F_PROXY_PLAN_CLAIMED_EXECUTABLE": lambda r: "추정 기준 계획을 실행 가능하다고 표시할 수 없습니다.",
    "G_MIN_LOSS_CLAIM_WITHOUT_COMPARISON": lambda r: "금액 비교가 완료되지 않아 최소손실 전략이라고 표시할 수 없습니다.",
    "H_EXECUTED_WITHOUT_OPERATOR_RECORD": lambda r: "판매자 실행 기록이 없어 실행된 것으로 표시할 수 없습니다.",
    "I_LEGACY_VS_CANONICAL_MISMATCH": lambda r: f"현재 화면의 'Varo 추천'({r['legacy_action']})은 계산된 이동 결정과 다릅니다.",
    "J_PRODUCT_SUMMARY_DOUBLE_COUNT": lambda r: "상품별 합계가 경로별 이동 계획 합과 다릅니다.",
    "PLAN_VS_SELLER_LOSS_DIFFERENT": lambda r: (f"이동 계획과 금액 기반 비교 결과({STRATEGY_LABELS.get(r['seller_loss_strategy'], '-')})가 "
                                                "달라 추가 검토가 필요합니다."),
}


def _names(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (record.get("source_name") or record.get("source_id") or "출발 점포",
            record.get("target_name") or record.get("target_id") or "도착 점포",
            record.get("product_name") or record.get("product_id") or "상품")


def explain_record(record: Mapping[str, Any]) -> list[str]:
    source, target, product = _names(record)
    unit, reasons, state = record.get("quantity_unit"), set(record.get("reason_codes") or []), record.get("decision_state")
    sentences: list[str] = []
    if record.get("execution_state") == EXECUTED:
        action = record.get("operator_action") or "-"
        sentences.append(f"판매자가 {str(record.get('executed_at'))[:10]}에 {_josa(action, '을/를')} 실행했다고 기록했습니다.")
    elif record.get("execution_state") not in (None, NOT_RECORDED):
        sentences.append(f"판매자 기록 상태는 {record.get('execution_state')}이며 실행으로 보지 않습니다.")
    if state == SELECTED_PLAN:
        quantity = _qty_text(record.get("allocated_qty"), unit)
        sentences.append(f"{source}의 {product} {_josa(quantity, '을/를')} {_josa(target, '으로/로')} 이동하는 계획입니다.")
        if record.get("feasibility_status") == PROXY_PLAN:
            sentences.append("이 계획은 추정 수요·재고(PROXY) 기준으로 계산됐으므로 실제 점포 수요와 이동 가능 재고 확인이 필요합니다."
                             if "PROXY_CAPS_APPLIED" in reasons else
                             "이 계획은 검증된 재고·수요 기준 없이 계산됐으므로 실제 실행 가능 여부 확인이 필요합니다.")
        else:
            sentences.append("검증된 재고·수요 기준을 충족한 계획입니다.")
        if "ROUTE_CAPACITY_NOT_CHECKED" in reasons:
            sentences.append("차량·경로 적재 한도는 데이터가 없어 확인하지 않았습니다.")
        if "QTY_PARTIAL_ALLOCATION" in reasons:
            sentences.append(f"기존 추천 수량 {_qty_text(record.get('recommended_qty'), unit)} 중 공동 제약을 반영해 "
                             f"{_qty_text(record.get('allocated_qty'), unit)}만 배정했습니다.")
        if "NONPRODUCTION_DATA" in reasons:
            sentences.append("검증용(TEST/SAMPLE) 데이터로 계산된 계획이라 실제 운영 계획으로 쓰지 않습니다.")
        if record.get("execution_state") == NOT_RECORDED:
            sentences.append("판매자 실행 기록은 아직 없습니다.")
    elif state == NOT_SELECTED:
        why = _STATUS_KO.get(record.get("selection_status"), "선택되지 않음")
        left = _number(record.get("executable_qty_at_end"))
        if record.get("selection_status") == sf.REJECTED_SELECTION_LIMIT and left:
            why += f", 남은 실행 가능 수량 {_qty_text(left, unit)}"
        sentences.append(f"이 이동 후보는 공동 제약 선택에서 제외됐습니다({why}).")
    elif state == INSUFFICIENT_DATA:
        sentences.append(f"이동 여부를 판단할 근거가 부족합니다({_STATUS_KO.get(record.get('selection_status'), '근거 부족')}).")
    else:
        sentences.append("공동 제약 선택 결과가 연결되지 않아 이동 계획 여부를 판단할 수 없습니다.")
    if "SELECTED_ONLY_UNDER_PROXY_CAPS" in reasons:
        sentences.append("추정(PROXY) 기준 계획에서는 선택됐지만 검증된 기준의 계획에는 들어가지 않았습니다.")
    codes = list(record.get("conflict_codes") or [])
    if any(code.startswith(("A_", "B_", "C_LEGACY_TRANSFER")) for code in codes):
        codes = [code for code in codes if code != "I_LEGACY_VS_CANONICAL_MISMATCH"]  # already said by A/B/C
    for code in codes:
        sentences.append(_CONFLICT_TEXT[code](record))
    claim, label = record.get("seller_loss_claim"), STRATEGY_LABELS.get(record.get("seller_loss_strategy"), "-")
    seller_text = {
        "MIN_LOSS_FULL": f"금액 비교(이동·정상 판매·할인)에서는 {_josa(label, '이/가')} 예상 손실이 가장 작았습니다(참고용이며 현재 행동은 바꾸지 않습니다).",
        "MIN_LOSS_PARTIAL_SCOPE": f"비교 가능한 전략 중에서는 {_josa(label, '이/가')} 예상 손실이 가장 작았습니다(일부 전략은 비교 불가, 참고용).",
        "MIN_LOSS_NONPRODUCTION_INPUT": f"검증용(TEST/SAMPLE) 입력 기준으로는 {_josa(label, '이/가')} 예상 손실이 가장 작지만 운영 근거로 쓸 수 없습니다.",
        "SCENARIO_ONLY": "가정 시나리오 기준 금액 비교이므로 운영 추천이 아닙니다.",
        "NOT_ROBUST": "금액 비교는 했지만 확인되지 않은 입력값에 따라 결과가 달라져 최소손실 전략을 정하지 않았습니다.",
        "NO_COMPARISON": "입력 부족으로 금액 기반 손실 비교를 하지 못했습니다(최소손실 판단 없음).",
    }.get(claim)
    if seller_text:
        sentences.append(seller_text)
    if "SELLER_LOSS_QTY_BASIS_DIFFERS" in reasons and record.get("seller_loss_min_loss_claim"):
        sentences.append(f"금액 비교는 추천 수량 {_qty_text(record.get('seller_loss_decision_qty'), unit)} 기준이며 배정 수량과 다릅니다.")
    if "SELLER_LOSS_PROMOTION_ELIGIBLE_NOT_APPLIED" in reasons:
        sentences.append("금액 비교 결과가 승격 조건을 충족했지만 현재 행동은 바꾸지 않습니다.")
    return sentences


# ------------------------------------------------------------------------------------------------ product level


def product_summaries(records: Sequence[Mapping[str, Any]], primary: str | None) -> list[dict[str, Any]]:
    """Product-level view that keeps every route plan (A->B 20, A->C 10) instead of one merged move."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("product_id") or ""].append(record)
    summaries = []
    for product, items in sorted(grouped.items()):
        plans = sorted((item for item in items if item["decision_state"] == SELECTED_PLAN),
                       key=lambda item: (item.get("shared_feasibility_rank") or 0, item.get("route_id") or ""))
        unit = plans[0]["quantity_unit"] if plans else None
        total = round(sum(float(item["allocated_qty"]) for item in plans), 6)
        name = items[0].get("product_name") or product
        legs = [{"route_id": item["route_id"], "source_id": item["source_id"], "target_id": item["target_id"],
                 "allocated_qty": item["allocated_qty"]} for item in plans]
        if plans:
            body = ", ".join(f"{_names(item)[0]}→{_names(item)[1]} {_qty_text(item['allocated_qty'], unit)}" for item in plans)
            label = f"{name}: 이동 계획 {len(plans)}건 ({body}), 합계 {_qty_text(total, unit)}"
        else:
            label = f"{name}: 선택된 이동 계획 없음"
        summaries.append({
            "decision_scope": "PRODUCT", "product_id": product or None, "product_name": name, "selection_mode": primary,
            "candidate_count": len(items), "plan_count": len(plans), "multi_route": len(plans) > 1, "plans": legs,
            "route_ids": [leg["route_id"] for leg in legs], "total_allocated_qty": total,
            "action_code": TRANSFER if plans else None,
            "action_status": plans[0]["feasibility_status"] if plans else "NO_PLAN",
            "legacy_actions_by_source": dict(sorted({f"{item.get('source_id')}": item.get("legacy_action_code") or "UNMAPPED"
                                                     for item in items}.items())),
            "label_ko": label,
        })
    return summaries


def detect_product_conflicts(summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[str]:
    plans = [item for item in records if item.get("product_id") == summary.get("product_id")
             and item.get("decision_state") == SELECTED_PLAN]
    route_ids = list(summary.get("route_ids") or [])
    total = sum(float(item.get("allocated_qty") or 0.0) for item in plans)
    legs = sum(float(leg.get("allocated_qty") or 0.0) for leg in summary.get("plans") or [])
    ok = (len(route_ids) == len(set(route_ids)) == summary.get("plan_count") == len(plans)
          and sorted(route_ids) == sorted(item.get("route_id") for item in plans)
          and abs(float(summary.get("total_allocated_qty") or 0.0) - total) <= 1e-6 and abs(legs - total) <= 1e-6)
    return [] if ok else ["J_PRODUCT_SUMMARY_DOUBLE_COUNT"]


# ------------------------------------------------------------------------------------------------ build


def check_invariants(result: Mapping[str, Any], recommendations: Sequence[Mapping[str, Any]],
                     plans: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = result["records"]
    primary = result.get("primary_selection_mode")
    plan = plans.get(primary) or {}
    by_id = {record["candidate_id"]: record for record in records}
    selected_rows = [row for row in plan.get("rows") or [] if row.get("selection_status") in _SELECTED
                     and float(row.get("allocated_qty") or 0.0) > EPS]
    production = {sf.candidate_identity(rec, index): _identifier(rec.get("varo_action")) or None
                  for index, rec in enumerate(recommendations)}

    def check(check_id: str, description: str, failures: list[Any]) -> dict[str, Any]:
        return {"check_id": check_id, "description": description, "status": "PASS" if not failures else "FAIL",
                "failures": failures[:20], "failure_count": len(failures)}

    return [
        check("I1", "selected transfer qty > 0 -> a transfer plan status exists",
              [row["candidate_id"] for row in selected_rows if row["candidate_id"] in by_id and (
                  by_id[row["candidate_id"]]["decision_state"] != SELECTED_PLAN
                  or by_id[row["candidate_id"]]["feasibility_status"] not in (FEASIBLE_PLAN, PROXY_PLAN))]),
        check("I2", "no transfer plan is shown with allocated_qty <= 0",
              [r["candidate_id"] for r in records if r["action_source"] == "SHARED_FEASIBILITY" and r["allocated_qty"] <= EPS]),
        check("I3", "allocated_qty <= every applied cap, and the backing plan has 0 shared-cap violations",
              [r["candidate_id"] for r in records if "D_ALLOCATION_EXCEEDS_CAP" in r["conflict_codes"]]
              + ([f"plan_violations={plan['validation']['violation_count']}"]
                 if (plan.get("validation") or {}).get("violation_count") else [])),
        check("I4", "route/product/store keys agree across linked layers; no plan row without a recommendation",
              [r["candidate_id"] for r in records if "E_KEY_MISMATCH" in r["conflict_codes"]]
              + [f"orphan:{item}" for item in result.get("orphan_plan_rows") or []]),
        check("I5", "production_action_applied=False and production_action equals the existing varo_action",
              [r["candidate_id"] for r in records if r["production_action_applied"] is not False
               or r["production_action"] != production.get(r["candidate_id"])]
              + [r["candidate_id"] for r in records if r["seller_loss_production_action_applied"] is not False]),
        check("I6", "operator_action NULL -> never EXECUTED",
              [r["candidate_id"] for r in records if not r["operator_action"] and (r["action_status"] == EXECUTED or r["is_executed"])]),
        check("I7", "PROXY plan -> never executable, never worded as a guarantee",
              [r["candidate_id"] for r in records if "F_PROXY_PLAN_CLAIMED_EXECUTABLE" in r["conflict_codes"]]),
        check("I8", "product summaries equal their route plans (no double count)",
              [s["product_id"] for s in result["product_summaries"] if s.get("conflict_codes")]),
        check("I9", "Seller Loss never backs the canonical action",
              [r["candidate_id"] for r in records if r["action_source"] == "SELLER_LOSS"]),
    ]


def build_action_consistency(
    recommendations: Sequence[Mapping[str, Any]], *, shared_feasibility: Mapping[str, Any] | None,
    seller_loss: Mapping[str, Any] | None = None, legacy_top_ids: Sequence[Any] | None = None,
    pareto_selected_ids: Sequence[Any] | None = None, legacy_rule_connected: bool = True,
    operator_events: Sequence[Mapping[str, Any]] | None = None, data_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical action records next to the production fields. Inputs are never mutated; the result has no timings."""
    primary, basis = primary_mode(shared_feasibility)
    plans = dict((shared_feasibility or {}).get("modes") or {})
    rows = {mode: {row["candidate_id"]: row for row in plan.get("rows") or []} for mode, plan in plans.items()}
    seller = {_identifier(item.get("decision_id")): item for item in (seller_loss or {}).get("decisions") or []}
    shadow = {_identifier(item.get("decision_id")): item for item in (seller_loss or {}).get("shadow_decisions") or []}
    events, rejected_events = operator_events_by_route(operator_events)
    nonproduction = _nonproduction({}, dict(data_context or {}))
    legacy_source = "LEGACY_RULE" if legacy_rule_connected else "UPLOADED_LABEL"
    legacy_top = {_identifier(item) for item in legacy_top_ids or []}
    pareto = {_identifier(item) for item in pareto_selected_ids or []}
    records = []
    for index, rec in enumerate(recommendations or []):
        record = _route_record(index, rec, primary=primary, rows=rows, plans=plans, seller=seller, shadow=shadow,
                               legacy_top=legacy_top, pareto=pareto, legacy_source=legacy_source,
                               nonproduction_data=nonproduction,
                               event=events.get(sf.candidate_identity(rec, index)))
        record["legacy_alignment"] = legacy_alignment(record)
        record["conflict_codes"] = detect_record_conflicts(record)
        record["explanation_ko"] = explain_record(record)
        # The detector also reads the wording, so check once more after the sentences exist.
        again = detect_record_conflicts(record)
        if again != record["conflict_codes"]:
            record["conflict_codes"] = again
            record["explanation_ko"] = explain_record(record)
        record["consistency_status"] = consistency_status(record, record["conflict_codes"])
        records.append(record)
    ids = {record["candidate_id"] for record in records}
    orphans = sorted(cid for cid, row in rows.get(primary, {}).items() if cid not in ids and row.get("selection_status") in _SELECTED)
    summaries = product_summaries(records, primary)
    for summary in summaries:
        summary["conflict_codes"] = detect_product_conflicts(summary, records)
    conflicts = Counter(code for record in records for code in record["conflict_codes"])
    conflicts.update(code for summary in summaries for code in summary["conflict_codes"])
    result = {
        "status": "parallel_only", "contract_version": CONTRACT_VERSION, "contract_signature": contract_document()["signature"],
        "production_action_applied": False, "legacy_action_replaced": False,
        "primary_selection_mode": primary, "primary_selection_basis": basis,
        "shared_feasibility_status": (shared_feasibility or {}).get("status"),
        "seller_loss_status": (seller_loss or {}).get("status"), "legacy_action_source": legacy_source,
        "nonproduction_data": nonproduction, "record_count": len(records),
        "action_status_counts": dict(sorted(Counter(r["action_status"] for r in records).items())),
        "decision_state_counts": dict(sorted(Counter(r["decision_state"] for r in records).items())),
        "consistency_counts": dict(sorted(Counter(r["consistency_status"] for r in records).items())),
        "alignment_counts": dict(sorted(Counter(r["legacy_alignment"] for r in records).items())),
        "conflict_counts": dict(sorted(conflicts.items())),
        "conflict_type_counts": dict(sorted(Counter(CONFLICTS[code][0] for code in conflicts.elements()).items())),
        "selected_plan_count": sum(r["decision_state"] == SELECTED_PLAN for r in records),
        "total_allocated_qty": round(sum(float(r["allocated_qty"]) for r in records if r["decision_state"] == SELECTED_PLAN), 6),
        "multi_route_products": [s["product_id"] for s in summaries if s["multi_route"]],
        "orphan_plan_rows": orphans, "rejected_operator_events": rejected_events,
        "records": records, "product_summaries": summaries,
    }
    result["invariants"] = check_invariants(result, list(recommendations or []), plans)
    result["invariant_failures"] = [item["check_id"] for item in result["invariants"] if item["status"] == "FAIL"]
    return result


def contract_document() -> dict[str, Any]:
    doc = {
        "version": CONTRACT_VERSION,
        "action_codes": {code: dict(item) for code, item in ACTION_CATALOG.items()},
        "label_to_code": dict(sorted(LABEL_TO_CODE.items())), "sentinel_labels": SENTINEL_LABELS,
        "mapping_rule": "exact label match only (normalize_action substring matching is not used)",
        "seller_loss_compared_strategies": list(STRATEGIES),
        "action_statuses": STATUS_DEFINITIONS,
        "decision_states": [SELECTED_PLAN, NOT_SELECTED, INSUFFICIENT_DATA, COMPARISON_UNAVAILABLE],
        "execution_states": [EXECUTED, "PLANNED", "CANCELLED", "NOT_EXECUTED", "UNKNOWN", NOT_RECORDED],
        "consistency_statuses": [CONSISTENT, REVIEW_REQUIRED, ACTION_CONFLICT, NOT_COMPARABLE],
        "legacy_alignment": list(ALIGNMENTS),
        "provenance_sources": PROVENANCE_SOURCES,
        "conflicts": {code: {"type": kind, "level": level, "description": text} for code, (kind, level, text) in CONFLICTS.items()},
        "reason_codes": REASON_CODES,
        "primary_plan_rule": "STRICT_ACTUAL backs actions unless its plan_status is INSUFFICIENT_CAP_DATA/EMPTY_INPUT; "
                             "then BENCHMARK_PROXY backs them as PROXY_PLAN",
        "quantity_rule": "a plan shows allocated_qty (never recommended_qty); allocated_qty must be > 0",
        "execution_rule": "EXECUTED only from a valid PRODUCTION operator record with execution_status EXECUTED, "
                          "operator_action, executed_at and observed provenance + source",
        "production_action_applied": False,
    }
    doc["signature"] = hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return doc
