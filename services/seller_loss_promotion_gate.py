"""Frozen v0.1 shadow gate. Qualifies evidence; never writes a production action.

FULL means all requested strategies are complete. A globally PARTIAL engine
result can qualify only for an explicitly requested, complete two-strategy scope.
Unknown monetary components or ties remain shadow decisions, even when the
engine's symbolic ranking is robust. No business percentage threshold is used.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from services import seller_loss_input_requirements as req, seller_loss_inputs as sli
from services.seller_loss_engine import (COMPONENTS, SELLER_LOSS_ACTIONS, STRATEGIES, TOL,
    STATUS_FULL, STATUS_PARTIAL, STATUS_UNAVAILABLE)

PROMOTION_ELIGIBLE, SHADOW_ONLY, BLOCKED, SCENARIO_ONLY, INSUFFICIENT_DATA = (
    "PROMOTION_ELIGIBLE", "SHADOW_ONLY", "BLOCKED", "SCENARIO_ONLY", "INSUFFICIENT_DATA")


@dataclass(frozen=True)
class PromotionPolicy:
    version: str = "seller-loss-promotion-gate-0.1"
    eligible_evidence: tuple[str, ...] = (sli.REAL_ONLY, sli.REAL_PLUS_USER_INPUT)
    numerical_tolerance: float = TOL
    require_complete_requested_scope: bool = True
    require_finite_monetary_components: bool = True
    require_explicit_two_strategy_scope: bool = True
    ties_remain_shadow: bool = True


POLICY = PromotionPolicy()
# Retain the engine's original legacy_agreement key. This narrower promotion
# mapping does not equate urgency, BOGO, disposal or waiting with normal sales.
ACTION_MAPPING = {
    "재고 이동": "TRANSFER", "TRANSFER": "TRANSFER",
    "할인": "DISCOUNT_SALE", "DISCOUNT_SALE": "DISCOUNT_SALE",
    "정상 판매": "NORMAL_SALE", "정상 판매 유지": "NORMAL_SALE", "NORMAL_SALE": "NORMAL_SALE",
}
UNMAPPABLE = ("긴급할인", "1+1", "폐기", "보류", "비교 불가", "유지")
_NONPRODUCTION = re.compile(r"(?i)(?:\bsamples?\b|\btest(?:[ _-]*user[ _-]*input)?\b|\bfixture\b|\bsynthetic\b|\bcontrolled[ _-]*scenario\b)")
MESSAGES = {
    PROMOTION_ELIGIBLE: "요청한 전략 비교와 입력 근거가 승격 조건을 충족했습니다. 현재 실행은 계속 기존 결정을 사용합니다.",
    SHADOW_ONLY: "참고용 비교로 유지합니다. 손실 계산 또는 입력 근거를 추가 확인해야 합니다.",
    BLOCKED: "검증용 입력 또는 입력 충돌 때문에 승격할 수 없습니다.",
    SCENARIO_ONLY: "가정한 조건의 비교이므로 운영 결정으로 승격할 수 없습니다.",
    INSUFFICIENT_DATA: "요청한 비교를 완성할 실제 입력이 부족합니다.",
}


def policy_document():
    doc = {**asdict(POLICY), "action_mapping": ACTION_MAPPING,
           "mapping_basis": "Explicit same-action semantics only; urgent discount, BOGO, disposal, waiting and ambiguous '유지' have no equivalent three-strategy contract.",
           "unmappable_actions": list(UNMAPPABLE),
           "test_sample_rule": "Hard block. Inspect execution labels, identifiers and every present input's source/dataset/note plus propagated seller audit labels.",
           "user_input_rule": "Declared source, valid scope and decision-date match; profile uses closed validity window. Production-row user values require dated production input context.",
           "unknown_rule": "Any unknown component in requested strategies remains shadow; robust symbolic recommendation is preserved.",
           "partial_rule": "Incomplete requested scope never qualifies; globally PARTIAL is allowed only for an explicitly requested complete pair.",
           "production_action_applied": False}
    doc["signature"] = hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return doc


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _date(value):
    from datetime import date
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _nonproduction(decision, context):
    tags = []
    for key in ("is_sample", "is_test", "synthetic_fixture"):
        if str(context.get(key, "")).lower() in {"true", "1", "yes"}:
            tags.append(key)
    for key in ("data_kind", "label", "source_file", "source_path", "execution_kind"):
        if _NONPRODUCTION.search(str(context.get(key, "")).replace("_", " ")):
            tags.append(key)
    for key in ("decision_id", "product_id", "source_store_id", "target_store_id"):
        if _NONPRODUCTION.search(str(decision.get(key, "")).replace("_", " ")):
            tags.append(key)
    for name, item in decision.get("input_provenance", {}).items():
        if item.get("value") is not None:
            if any(_NONPRODUCTION.search(str(item.get(k, "")).replace("_", " ")) for k in ("source", "dataset", "note")):
                tags.append(name)
    for row in decision.get("seller_input_audit", []):
        if any(_NONPRODUCTION.search(str(row.get(k, "")).replace("_", " "))
               for k in ("data_labels", "profile_source", "source")):
            tags.append(str(row.get("field")))
    return sorted(set(tags))


def _user_audit_blockers(decision, scope_strategies, context):
    blockers = []
    provenance = decision.get("input_provenance", {})
    needed = {"decision_qty", *[f for s in scope_strategies for f in sli.STRATEGY_REQUIRED[s]]}
    needed.update(f for s in scope_strategies for f in sli.STRATEGY_OPTIONAL[s] if provenance.get(f, {}).get("value") is not None)
    audit = decision.get("seller_input_audit", [])
    date = _date(context.get("decision_date"))
    for name in sorted(needed):
        item = provenance.get(name, {})
        if item.get("provenance") not in sli.USER_PROVENANCE or item.get("value") is None:
            continue
        if not item.get("source"):
            blockers.append("USER_INPUT_SOURCE_MISSING:" + name)
        origin = decision.get("input_sources", {}).get(name, {}).get("origin")
        if origin in {"SELLER_LOSS_INPUTS", "SELLER_BUSINESS_PROFILE"}:
            accepted = [r for r in audit if r.get("field") == name and str(r.get("outcome", "")).startswith("APPLIED")]
            if not accepted:
                blockers.append("USER_INPUT_AUDIT_MISSING:" + name)
            for row in accepted:
                scope, keys = row.get("scope"), row.get("keys", {})
                start = _date(keys.get("effective_date"))
                end = _date(row.get("effective_until"))
                valid_window = origin != "SELLER_BUSINESS_PROFILE" or (start and end and start <= end)
                if scope not in sli.SCOPES or sli._scope_error(scope, keys):
                    blockers.append("USER_INPUT_SCOPE_INVALID:" + name)
                elif valid_window:
                    role = sli.ENGINE_TARGETS.get(name, ((), None))[1]
                    column = sli.ENGINE_TARGETS.get(name, ((name,), None))[0][0]
                    cell = sli.SellerInputCell(1, column, row.get("input_value", 0), scope,
                        tuple(keys.items()), "ACTUAL", input_source="seller_business_profile" if origin == "SELLER_BUSINESS_PROFILE" else "seller_loss_inputs",
                        effective_until=row.get("effective_until"))
                    key = sli.DecisionKey(str(decision.get("decision_id")), str(decision.get("product_id")),
                        str(decision.get("source_store_id")), decision.get("target_store_id"), str(date) if date else None)
                    if not sli._matches(cell,key,role):
                        blockers.append("USER_INPUT_SCOPE_OR_DATE_MISMATCH:" + name)
                if origin == "SELLER_BUSINESS_PROFILE":
                    # Unit metadata may be timeless; economic values may not.
                    if name != "quantity_unit" and (not date or not start or not end or not start <= date <= end):
                        blockers.append("USER_INPUT_EFFECTIVE_DATE_INVALID:" + name)
                elif start and (not date or start != date):
                    blockers.append("USER_INPUT_EFFECTIVE_DATE_INVALID:" + name)
                elif not start and not date:
                    blockers.append("USER_INPUT_DECISION_DATE_MISSING:" + name)
        elif not date:
            blockers.append("USER_INPUT_DECISION_DATE_MISSING:" + name)
    return sorted(set(blockers))


def evaluate_promotion_gate(decision: Mapping[str, Any], *, comparison_scope: str | None = None,
                            explicit_scope: bool = False, context: Mapping[str, Any] | None = None):
    """Pure, fail-closed qualification of an already computed engine decision."""
    context = dict(context or {})
    scope = comparison_scope or (decision.get("input_requirements") or {}).get("comparison_scope") or req.ALL_THREE
    requested = req.COMPARISON_SCOPES.get(scope, ())
    comparable = list(decision.get("comparable_strategies") or [])
    winner = decision.get("recommended_strategy")
    action = decision.get("seller_loss_action")
    block, shadow, missing = [], [], []
    nonproduction = _nonproduction(decision, context)
    if nonproduction:
        block.append("TEST_OR_SAMPLE_INPUT")
    scenario = decision.get("decision_mode") == sli.SCENARIO or decision.get("evidence_level") == sli.SCENARIO_EVIDENCE or bool(decision.get("scenario_inputs"))
    if scenario:
        block.append("SCENARIO_INPUT")
    if scope not in req.COMPARISON_SCOPES:
        block.append("INVALID_COMPARISON_SCOPE")
    elif scope != req.ALL_THREE and not explicit_scope:
        block.append("TWO_STRATEGY_SCOPE_NOT_EXPLICIT")
    complete = bool(requested) and all(s in comparable and decision.get("strategies", {}).get(s, {}).get("available") for s in requested)
    if not complete:
        missing.append("PARTIAL_REQUESTED_SCOPE")
    if len(comparable) < 2:
        missing.append("FEWER_THAN_TWO_COMPARABLE_STRATEGIES")
    available = [s for s in STRATEGIES if decision.get("strategies", {}).get(s, {}).get("available")]
    expected_status = STATUS_FULL if len(available) == 3 else STATUS_PARTIAL if len(available) == 2 else STATUS_UNAVAILABLE
    if set(comparable) != set(available) or len(comparable) != len(set(comparable)) or decision.get("comparison_status") != expected_status:
        block.append("COMPARISON_CONTRACT_INCONSISTENT")
    if decision.get("recommendation_readiness") != sli.RECOMMENDABLE:
        missing.append("NOT_RECOMMENDABLE")
    if decision.get("decision_mode") not in (sli.ACTUAL_OPERATION, sli.SCENARIO):
        block.append("DECISION_MODE_UNVERIFIED")
    if not winner or not action:
        missing.append("SELLER_LOSS_ACTION_MISSING")
    elif winner not in requested:
        block.append("WINNER_OUTSIDE_REQUESTED_SCOPE")
    elif action != SELLER_LOSS_ACTIONS.get(winner):
        block.append("ACTION_STRATEGY_MISMATCH")
    if decision.get("conflicting_fields") or decision.get("input_conflicts"):
        block.append("INPUT_CONFLICT")
    codes = [*decision.get("reason_codes", []), *decision.get("seller_input_issues", [])]
    for strategy in decision.get("strategies", {}).values():
        codes.extend(strategy.get("unavailable_reasons", []))
    for prefix in ("CURRENCY_MISMATCH", "UNIT_MISMATCH", "SELLER_UNIT_UNVERIFIABLE", "CROSS_DATASET_INPUT"):
        if any(prefix in str(c) for c in codes):
            block.append(prefix)
    level = decision.get("evidence_level")
    if level == sli.USER_INPUT_ONLY:
        shadow.append("USER_INPUT_ONLY")
    elif level not in POLICY.eligible_evidence and not scenario:
        missing.append("INSUFFICIENT_EVIDENCE")
    if level in POLICY.eligible_evidence:
        block.extend(_user_audit_blockers(decision, requested, context))
        provenance = decision.get("input_provenance", {})
        for name in sorted({"decision_qty", *[f for s in requested for f in sli.STRATEGY_REQUIRED[s]]}):
            item = provenance.get(name, {})
            if not _finite(item.get("value")) or item.get("provenance") not in sli.REAL_PROVENANCE | sli.USER_PROVENANCE:
                missing.append("INPUT_PROVENANCE_INCOMPLETE:" + name)
        if not decision.get("currency") or not decision.get("quantity_unit"):
            missing.append("CURRENCY_OR_QUANTITY_UNIT_UNDECLARED")
    selected = decision.get("strategies", {}).get(winner, {})
    for s in requested:
        value = decision.get("strategies", {}).get(s, {})
        if value.get("available") and (not _finite(value.get("expected_loss")) or value.get("unknown_inputs")
                                      or set(value.get("components", {})) != set(COMPONENTS)
                                      or any(not _finite(v) for v in value.get("components", {}).values())):
            shadow.append("UNKNOWN_MONETARY_COMPONENT:" + s)
    if winner and not _finite(selected.get("expected_loss")):
        shadow.append("SELECTED_LOSS_INCOMPLETE")
    margin = decision.get("loss_difference_vs_second_best")
    if winner and (decision.get("tie") or _finite(margin) and margin <= POLICY.numerical_tolerance):
        shadow.append("TIE_OR_NUMERICAL_TOLERANCE")
    elif winner and not _finite(margin):
        shadow.append("ROBUST_MARGIN_UNAVAILABLE")
    if decision.get("recommendation_status") != "RECOMMENDED":
        shadow.append("RANKING_NOT_ROBUST")
    if complete and _finite(selected.get("expected_loss")):
        other = [decision["strategies"][s].get("expected_loss") for s in requested if s != winner]
        if any(_finite(v) and v < selected["expected_loss"] - POLICY.numerical_tolerance for v in other):
            block.append("LOSS_RANKING_INCONSISTENT")
        if any(_finite(v) and abs(v - selected["expected_loss"]) <= POLICY.numerical_tolerance for v in other):
            shadow.append("TIE_OR_NUMERICAL_TOLERANCE")
    status = BLOCKED if nonproduction else SCENARIO_ONLY if scenario else BLOCKED if block else INSUFFICIENT_DATA if missing else SHADOW_ONLY if shadow else PROMOTION_ELIGIBLE
    blockers = sorted(set([*block, *missing, *shadow]))
    reasons = blockers or ["COMPLETE_REQUESTED_SCOPE", "ACCEPTED_EVIDENCE", "FINITE_LOSS", "ROBUST_NON_TIED_WINNER"]
    return {"policy_version": POLICY.version, "policy_signature": policy_document()["signature"],
            "promotion_status": status, "promotion_reason_codes": reasons, "promotion_blockers": blockers,
            "promotion_candidate": status == PROMOTION_ELIGIBLE, "production_action_applied": False,
            "comparison_scope": scope, "explicit_scope": explicit_scope,
            "promotion_comparison_status": "FULL_REQUESTED_SCOPE" if complete else "PARTIAL_REQUESTED_SCOPE",
            "requested_strategies": list(requested), "excluded_strategies": [s for s in STRATEGIES if s not in requested],
            "nonproduction_evidence": nonproduction, "message_ko": MESSAGES[status]}


def shadow_decision(decision, **options):
    gate = evaluate_promotion_gate(decision, **options)
    legacy, winner = decision.get("legacy_action"), decision.get("recommended_strategy")
    mapped = ACTION_MAPPING.get(str(legacy).strip()) if legacy else None
    agreement = "NO_RECOMMENDATION" if not winner else "UNMAPPABLE" if not mapped else "SAME" if mapped == winner else "DIFFERENT"
    reasons = []
    if agreement == "UNMAPPABLE":
        reasons += ["action_mapping_unavailable", "legacy_not_comparable"]
    elif agreement == "DIFFERENT":
        reasons.append("different_objective")
    reasons.append("seller_loss_has_monetary_evidence" if _finite(decision.get("strategies", {}).get(winner, {}).get("expected_loss")) else "insufficient_monetary_evidence")
    if "WINNER_OUTSIDE_REQUESTED_SCOPE" in gate["promotion_blockers"]:
        reasons.append("scope_difference")
    return {"decision_id": decision.get("decision_id"), "product_id": decision.get("product_id"),
            "legacy_action": legacy, "seller_loss_action": decision.get("seller_loss_action"),
            "seller_loss_readiness": decision.get("recommendation_readiness"),
            "seller_loss_evidence_level": decision.get("evidence_level"),
            "seller_loss_comparison_status": decision.get("comparison_status"),
            "seller_loss_expected_loss": decision.get("strategies", {}).get(winner, {}).get("expected_loss"),
            "legacy_mapped_strategy": mapped, "legacy_vs_seller_agreement": agreement,
            "disagreement_reason_codes": sorted(set(reasons)), **gate,
            "final_action_candidate": {"current_production_action": legacy,
                "seller_loss_candidate_action": decision.get("seller_loss_action"),
                "promotion_status": gate["promotion_status"], "production_action_applied": False}}


def summarize_shadow(rows: Sequence[Mapping[str, Any]]):
    states = Counter(r["promotion_status"] for r in rows)
    agreement = Counter(r["legacy_vs_seller_agreement"] for r in rows)
    return {"total_decisions": len(rows), "seller_loss_recommendable": sum(r["seller_loss_readiness"] == sli.RECOMMENDABLE for r in rows),
            "promotion_eligible": states[PROMOTION_ELIGIBLE], "shadow_only": states[SHADOW_ONLY],
            "blocked": states[BLOCKED], "scenario_only": states[SCENARIO_ONLY], "insufficient_data": states[INSUFFICIENT_DATA],
            "status_counts": dict(sorted(states.items())), "legacy_agreement": agreement["SAME"],
            "legacy_disagreement": agreement["DIFFERENT"], "unmappable": agreement["UNMAPPABLE"],
            "no_recommendation": agreement["NO_RECOMMENDATION"], "production_action_applied": False}
