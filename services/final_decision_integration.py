"""Parallel v0.1 decision contract. No scoring, optimization, production or DB writes.

T1 plans, T2 labels, validated sales, the existing loss engine/planner/profile and
promotion policy remain separate authorities. A route is not a monetary winner.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Mapping

import pandas as pd

from services import action_consistency as ac
from services import shared_feasibility_selection as sf
from services import seller_loss_engine as loss
from services import seller_loss_inputs as inputs
from services import seller_business_profile as profiles
from services import seller_loss_input_requirements as requirements
from services import seller_loss_promotion_gate as gate

VERSION = "final-decision-integration-0.1"
STATUSES = ("RECOMMENDATION_AVAILABLE", "CONDITIONAL_RECOMMENDATION", "PROXY_PLAN_AVAILABLE",
            "REVIEW_REQUIRED", "ADDITIONAL_INPUT_REQUIRED", "COMPARISON_UNAVAILABLE", "NO_FEASIBLE_TRANSFER",
            "INSUFFICIENT_DATA", "NO_ACTION_REQUIRED")
LABELS = {"TRANSFER": "재고 이동", "NORMAL_SALE": "정상판매 유지", "DISCOUNT_SALE": "할인판매"}
TRUST = {"DIRECT_REAL": "ACTUAL_VERIFIED", "DERIVED_REAL": "DERIVED_FROM_REAL",
         "USER_INPUT": "OPERATOR_CONFIRMED", "DERIVED_FROM_USER_INPUT": "OPERATOR_CONFIRMED",
         "PROXY": "PROXY", "SCENARIO_INPUT": "SCENARIO", "CONFIG": "SCENARIO"}
CONTEXT_KEYS = ("is_sample", "is_test", "synthetic_fixture", "data_kind", "source_file", "source_path", "label")


def text(value):
    if value is None or not isinstance(value, (str, int, float)) or pd.isna(value): return None
    result = str(value).strip()
    return result or None


def number(value):
    if isinstance(value, bool): return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError): return None


def records(frame):
    return frame.to_dict("records") if isinstance(frame, pd.DataFrame) else []


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def codes(value):
    if isinstance(value, str): return [v for v in value.split("|") if v]
    return [str(v) for v in value or []]


def key(row):
    return (text(row.get("dataset_id")), text(row.get("business_id")),
            text(row.get("store_id", row.get("location_id", row.get("node_id")))), text(row.get("product_id")))


def product_catalog(data):
    """Inventory keys plus unrepresented master products; no invented store cross join.

    Duplicate snapshots are kept as conflicting evidence, never summed or silently
    chosen. Missing keys receive distinct internal identities and stay visible.
    """
    grouped = defaultdict(list)
    for position, row in enumerate(records(data.get("inventory"))):
        identity = key(row)
        if not identity[2] or not identity[3]: identity = (*identity, "invalid_inventory", position)
        grouped[identity].append({"sheet": "inventory", "row_position": position, "row": row})
    for position, row in enumerate(records(data.get("products"))):
        ds, business, _, product = key(row)
        represented = any(k[3] == product and (ds is None or k[0] == ds) and (business is None or k[1] == business)
                          for k in grouped if product)
        if not represented:
            identity = (ds, business, None, product)
            if product is None: identity = (*identity, "invalid_master", position)
            grouped[identity].append({"sheet": "products", "row_position": position, "row": row})
    return [(k, grouped[k]) for k in sorted(grouped, key=lambda k: tuple(str(v or "") for v in k))]


def coverage_check(catalog, rows):
    expected = [digest(k) for k, _ in catalog]
    actual = [r["product_key"] for r in rows]
    missing = sorted(set(expected) - set(actual))
    duplicate = sorted(k for k, n in Counter(actual).items() if n > 1)
    unexpected = sorted(set(actual) - set(expected))
    return {"missing": missing, "duplicates": duplicate, "unexpected": unexpected,
            "reason_codes": (["MISSING_DECISION_ROW"] if missing else []) +
                            (["DUPLICATE_DECISION_ROW"] if duplicate else []) +
                            (["UNEXPECTED_DECISION_ROW"] if unexpected else []),
            "pass": not (missing or duplicate or unexpected)}


def index_rows(rows, fields):
    grouped = defaultdict(list)
    for row in rows or []:
        if isinstance(row, Mapping): grouped[tuple(text(row.get(f)) for f in fields)].append(row)
    return grouped


def unique_match(index, row, fields):
    values = index.get(tuple(text(row.get(f)) for f in fields), [])
    return values[0] if len(values) == 1 else None


def money(amount, currency, source, state, *, kind):
    return {"value": number(amount), "currency": currency, "unit": "currency" if currency else None,
            "source": source, "calculation_status": state if number(amount) is not None else "UNKNOWN",
            "kind": kind, "actual": False}


def loss_contract(decision, context, scope, explicit):
    if not decision:
        return {"decision_id": None, "comparison_scope": scope, "comparison_status": "COMPARISON_UNAVAILABLE",
                "recommendation_readiness": "UNAVAILABLE", "selected_strategy": None,
                "available_strategies": [], "excluded_strategies": [], "expected_loss_by_strategy":
                    {s: None for s in loss.STRATEGIES}, "evidence_level": "INSUFFICIENT", "unknown_inputs": [],
                "reason_codes": ["SELLER_COMPARISON_CONTEXT_UNAVAILABLE"], "missing_required_inputs": [],
                "required_context_inputs": [],
                "promotion_status": gate.INSUFFICIENT_DATA, "promotion_blockers": ["SELLER_COMPARISON_CONTEXT_UNAVAILABLE"],
                "production_action_applied": False, "input_provenance": {}, "scope_complete": False,
                "scenario": False, "nonproduction": [], "quantity": None, "quantity_unit": None, "currency": None}
    plan = decision.get("input_requirements") or {}
    effective_scope = plan.get("comparison_scope") or scope
    qualified = gate.evaluate_promotion_gate(decision, comparison_scope=effective_scope, explicit_scope=explicit, context=context)
    requested = requirements.COMPARISON_SCOPES.get(effective_scope, ())
    available = [s for s in requested if decision.get("strategies", {}).get(s, {}).get("available")]
    fields = {s: number(decision.get("strategies", {}).get(s, {}).get("expected_loss"))
              if decision.get("strategies", {}).get(s, {}).get("available") else None for s in loss.STRATEGIES}
    scenario = decision.get("decision_mode") == inputs.SCENARIO or decision.get("evidence_level") == inputs.SCENARIO_EVIDENCE
    return {"decision_id": decision.get("decision_id"), "comparison_scope": effective_scope,
            "comparison_status": decision.get("comparison_status"), "recommendation_readiness": decision.get("recommendation_readiness"),
            "selected_strategy": decision.get("recommended_strategy"), "available_strategies": available,
            "excluded_strategies": [s for s in loss.STRATEGIES if s not in requested],
            "expected_loss_by_strategy": fields, "evidence_level": decision.get("evidence_level"),
            "unknown_inputs": deepcopy(decision.get("unknown_inputs") or []),
            "reason_codes": sorted(set(codes(decision.get("reason_codes")) + codes(decision.get("seller_input_issues")))),
            "missing_required_inputs": deepcopy(plan.get("required_user_inputs") or []),
            "required_context_inputs": deepcopy(plan.get("blocked_fields") or []),
            "promotion_status": qualified["promotion_status"], "promotion_blockers": qualified["promotion_blockers"],
            "production_action_applied": False, "input_provenance": deepcopy(decision.get("input_provenance") or {}),
            "scope_complete": qualified["promotion_comparison_status"] == "FULL_REQUESTED_SCOPE",
            "scenario": scenario, "nonproduction": qualified["nonproduction_evidence"],
            "quantity": number(decision.get("decision_qty")), "quantity_unit": decision.get("quantity_unit"),
            "currency": decision.get("currency"), "loss_components": deepcopy(decision.get("breakdown") or {}),
            "cost_difference": None, "cost_difference_status": "NOT_COMPARABLE_NO_SAME_PROBLEM_SIGNATURE",
            "engine_margin_reference": number(decision.get("loss_difference_vs_second_best"))}


def accepted_demand(sales, seller):
    if number((sales or {}).get("sales_30d")) is not None and (sales or {}).get("retail_30d") and sales["retail_30d"].get("validation_status") in ("VALID", "VALID_WITH_LIMITATION"):
        return "ACTUAL_VERIFIED" if sales["retail_30d"].get("semantics_evidence") == "SOURCE_DOCUMENTED" else "OPERATOR_CONFIRMED"
    field = (seller or {}).get("input_provenance", {}).get("source_daily_demand", {})
    return TRUST.get(field.get("provenance"), "INSUFFICIENT") if number(field.get("value")) is not None else "INSUFFICIENT"


def classify(*, seller, selected, mode, operational, demand, stock_status, conflicts, reasons):
    flags = set()
    if selected: flags.add("PROXY_PLAN_AVAILABLE" if mode == sf.BENCHMARK_PROXY else "CONDITIONAL_RECOMMENDATION")
    if seller["missing_required_inputs"] or seller["required_context_inputs"]: flags.add("ADDITIONAL_INPUT_REQUIRED")
    if seller["comparison_status"] == loss.STATUS_UNAVAILABLE: flags.add("COMPARISON_UNAVAILABLE")
    if not seller["scope_complete"]: reasons.append("COMPARISON_SCOPE_INCOMPLETE")
    if demand in ("PROXY", "INSUFFICIENT"): reasons.append("ACTUAL_RETAIL_DEMAND_UNAVAILABLE")
    if stock_status in ("MISSING", "INVALID", "CONFLICTED"): reasons.append("STOCK_EVIDENCE_UNAVAILABLE")
    if conflicts: flags.add("REVIEW_REQUIRED")
    if seller["promotion_status"] == gate.BLOCKED and not seller["nonproduction"] and not seller["scenario"]:
        flags.add("REVIEW_REQUIRED")
    winner = seller["selected_strategy"]
    recommended = None
    readiness = "UNAVAILABLE"
    safe = seller["promotion_status"] == gate.PROMOTION_ELIGIBLE and seller["scope_complete"] and not conflicts
    if safe and demand in ("ACTUAL_VERIFIED", "OPERATOR_CONFIRMED", "DERIVED_FROM_REAL") and stock_status not in ("MISSING", "INVALID", "CONFLICTED"):
        if winner != "TRANSFER" or selected and mode == sf.STRICT_ACTUAL and operational:
            recommended = winner
            readiness = "QUALIFIED_NOT_APPLIED"
            flags.add("RECOMMENDATION_AVAILABLE" if seller["evidence_level"] == inputs.REAL_ONLY else "CONDITIONAL_RECOMMENDATION")
        else:
            flags.add("REVIEW_REQUIRED")
            reasons.append("TRANSFER_NOT_OPERATIONALLY_VERIFIED")
    elif winner and not seller["scenario"]:
        reasons.append("MONETARY_WINNER_NOT_QUALIFIED")
    if seller["scenario"] or seller["nonproduction"]:
        recommended = None
        readiness = "SCENARIO_ONLY" if seller["scenario"] else "TEST_INPUT_ONLY"
        reasons.append("NOT_ACTUAL_OPERATION")
        if seller["scope_complete"] and winner: flags.add("CONDITIONAL_RECOMMENDATION")
    if selected and mode == sf.STRICT_ACTUAL and not operational:
        flags.add("REVIEW_REQUIRED")
        reasons.append("ROUTE_APPROVAL_UNVERIFIED")
    order = ("REVIEW_REQUIRED", "RECOMMENDATION_AVAILABLE", "CONDITIONAL_RECOMMENDATION", "PROXY_PLAN_AVAILABLE",
             "ADDITIONAL_INPUT_REQUIRED", "COMPARISON_UNAVAILABLE", "NO_FEASIBLE_TRANSFER", "INSUFFICIENT_DATA")
    return next((s for s in order if s in flags), "INSUFFICIENT_DATA"), sorted(flags), recommended, readiness


def explain(row):
    sentences = []
    if row.get("selected_transfer_routes"):
        proxy = row["decision_mode"] == sf.BENCHMARK_PROXY
        sentences.append("추정 조건에서 계산된 이동 계획이 있습니다." if proxy else "현재 제약 조건에서 계산된 이동 계획이 있습니다.")
        sentences.append("이 계획은 실제 실행 완료나 최소손실 전략을 뜻하지 않습니다.")
    elif row.get("transfer_candidate_status") == "CANDIDATE_ONLY":
        sentences.append("이동 후보는 있지만 선택된 이동 계획은 없습니다.")
    if row.get("recommended_strategy"):
        sentences.append(f"요청한 비교 범위에서 {LABELS.get(row['recommended_strategy'], '해당 전략')}의 예상 손실이 가장 작습니다. 실제 성과는 아닙니다.")
    elif row.get("scenario_recommended_strategy"):
        sentences.append(f"가정한 조건의 비교에서는 {LABELS.get(row['scenario_recommended_strategy'], '해당 전략')}의 예상 손실이 가장 작습니다. 검증·시나리오 입력이며 실제 운영 추천이 아닙니다.")
    else:
        sentences.append("현재 근거만으로 이동·정상판매·할인 중 최소손실 전략을 확정할 수 없습니다.")
    if row.get("legacy_action"):
        sentences.append("기존 행동 표시는 규칙 기반 참고 권고이며 금액 비교 결과와 별개입니다.")
    if row.get("demand_validation_status") in ("INSUFFICIENT", "PROXY"):
        sentences.append("실제 소매 판매 근거가 부족하며 물류 출고량은 소매 판매량으로 보지 않습니다.")
    if row["decision_status"] == "REVIEW_REQUIRED": sentences.append("계획·기존 권고 또는 입력 근거가 달라 추가 검토가 필요합니다.")
    if row["decision_status"] == "NO_ACTION_REQUIRED": sentences = ["확인된 입력 재고가 0이며 선택된 이동 계획이 없어 현재 재고에 대한 조치는 필요하지 않습니다."]
    labels = list(dict.fromkeys(i.get("label") or i.get("field") for i in row.get("missing_required_inputs", []) if i.get("label") or i.get("field")))
    if labels: sentences.append("추가 비교에 필요한 정보: " + ", ".join(labels[:4]) + (" 등" if len(labels) > 4 else "") + ".")
    return " ".join(sentences)


def business_context(data, analyzed):
    config = loss._config(data.get("config"))
    mode, mode_info = inputs.resolve_decision_mode(None, data, config)
    scope, scope_info = requirements.resolve_comparison_scope(None, data, config)
    table = inputs.parse_seller_loss_inputs(data.get(inputs.SHEET_KEY), upload_currency=config.get("currency") or loss.WORKBOOK_CURRENCY,
                                          upload_currency_basis="config.currency" if config.get("currency") else loss.WORKBOOK_CURRENCY_NOTE)
    profile = profiles.parse_profile(data.get(profiles.SHEET_KEY))
    validation_errors = (["seller_loss_inputs:" + str(e.get("code")) for e in table.validation.get("errors", [])] +
                         ["seller_business_profile:" + str(e.get("code")) for e in profile.validation.get("errors", [])])
    if mode_info.get("error"): validation_errors.append("INVALID_DECISION_MODE")
    if scope_info.get("error"): validation_errors.append("INVALID_COMPARISON_SCOPE")
    table = profiles.combine_inputs(table, profile)
    return {"config": config, "mode": mode, "scope": scope, "explicit": scope_info.get("basis") != "default" and not scope_info.get("error"),
            "table": table, "inventory": loss._unique_rows(data.get("inventory"), ("store_id", "location_id", "node_id")),
            "forecast": loss._unique_rows(analyzed, ("store_id", "location_id", "node_id")),
            "products": {text(r.get("product_id")): r for r in records(data.get("products"))}, "validation_errors": validation_errors}


def product_query(identity, row, context, run_id):
    """No-route loss query. Never invents a target, quantity, route or sales rate."""
    rec = {"product_id": identity[3], "source_id": identity[2], "target_id": None,
           "recommended_qty": number(row.get("decision_qty")), "route_id": "query-" + digest(identity)[:20]}
    base = loss.pipeline_decision_input(rec, inventory_rows=context["inventory"], forecast_rows=context["forecast"],
            product_rows=context["products"], config=context["config"], decision_mode=context["mode"])
    day = loss.decision_date(row, context["config"])
    decision = inputs.evaluate_with_seller_inputs(base, context["table"], decision_mode=context["mode"],
                                                  decision_date=day, dataset=loss.WORKBOOK_DATASET)
    decision["input_requirements"] = requirements.safe_plan(base, context["table"], decision, comparison_scope=context["scope"],
            decision_date=day, dataset=loss.WORKBOOK_DATASET, upload_currency=context["config"].get("currency") or loss.WORKBOOK_CURRENCY)
    return decision


def build_final_decision_integration(data: Mapping[str, Any] | None, pipeline: Mapping[str, Any] | None,
                                     analyzed_inventory=None):
    data, pipeline = data or {}, pipeline or {}
    catalog = product_catalog(data)
    run_id = text(data.get("run_id")) or "fdi-" + digest({"inventory": records(data.get("inventory")),
        "products": records(data.get("products")), "config": records(data.get("config")),
        "seller_inputs": records(data.get(inputs.SHEET_KEY)), "profile": records(data.get(profiles.SHEET_KEY)),
        "recommendations": pipeline.get("recommendations", []),
        "decision_layers": {name: pipeline.get(name) for name in ("seller_loss_analysis", "shared_feasibility_selection",
            "action_consistency", "inventory_metric_integrity", "operational_evidence")}})[:24]
    context_errors = []
    try:
        business = business_context(data, analyzed_inventory)
        context_errors = ["OPTIONAL_INPUT_INVALID", *business["validation_errors"]] if business["validation_errors"] else []
    except Exception as error:
        business = {"config": loss._config(data.get("config")), "mode": inputs.ACTUAL_OPERATION, "scope": requirements.ALL_THREE, "explicit": False}
        context_errors = ["OPTIONAL_INPUT_INVALID", type(error).__name__]
    base_context = {k: data[k] for k in CONTEXT_KEYS if k in data}
    shared = pipeline.get("shared_feasibility_selection") or {}
    primary, _ = ac.primary_mode(shared)
    plans = {mode: index_rows(plan.get("rows"), ("candidate_id",)) for mode, plan in shared.get("modes", {}).items()}
    t2_index = index_rows((pipeline.get("action_consistency") or {}).get("records"), ("route_id", "product_id", "source_id", "target_id"))
    seller_index = index_rows((pipeline.get("seller_loss_analysis") or {}).get("decisions"), ("decision_id", "product_id", "source_store_id", "target_store_id"))
    op_index = index_rows((pipeline.get("operational_evidence") or {}).get("candidates"), ("route_id", "product_id", "source_id", "target_id"))
    metric = pipeline.get("inventory_metric_integrity") or {}
    sales_index = index_rows(metric.get("validated_sales_signals"), ("store_id", "product_id"))
    turnover_index = index_rows([{**t, "store_id": s.get("store_id"), "product_id": s.get("product_id")}
        for s, t in zip(metric.get("validated_sales_signals") or [], metric.get("validated_turnover") or [])], ("store_id", "product_id"))
    forecast_index = index_rows([{**f, "store_id": s.get("store_id"), "product_id": s.get("product_id")}
        for s, f in zip(metric.get("validated_sales_signals") or [], metric.get("validated_forecast_input") or [])], ("store_id", "product_id"))
    inventory_index = index_rows(records(analyzed_inventory), ("store_id", "product_id"))
    by_source, incoming, route_rows, orphans = defaultdict(list), defaultdict(list), [], []
    seen_routes = {}
    valid_keys = [k for k, entries in catalog if len(k) == 4 and k[2] and k[3]]
    for rec in pipeline.get("recommendations") or []:
        route_identity = text(rec.get("route_id"))
        if route_identity and route_identity in seen_routes:
            original = seen_routes[route_identity]
            original["reason_codes"] = sorted(set(original["reason_codes"] + ["DUPLICATE_ROUTE_INPUT"]))
            original["decision_status"] = "REVIEW_REQUIRED"
            original["recommended_strategy"] = None
            original["recommendation_readiness"] = "UNAVAILABLE"
            original["duplicate_input_keys"].append({k: rec.get(k) for k in ("route_id", "product_id", "source_id", "target_id")})
            original["explanation_ko"] = explain(original)
            continue
        source_keys = [k for k in valid_keys if k[2] == text(rec.get("source_id")) and k[3] == text(rec.get("product_id"))
                       and (not text(rec.get("dataset_id")) or k[0] == text(rec.get("dataset_id")))]
        if len(source_keys) != 1:
            orphans.append({"route_id": rec.get("route_id"), "reason": "ROUTE_PRODUCT_NAMESPACE_AMBIGUOUS" if source_keys else "ROUTE_PRODUCT_KEY_MISSING"})
            continue
        identity = source_keys[0]
        t2 = unique_match(t2_index, rec, ("route_id", "product_id", "source_id", "target_id")) or {}
        op = unique_match(op_index, rec, ("route_id", "product_id", "source_id", "target_id")) or {}
        slkey = {"decision_id": rec.get("route_id"), "product_id": rec.get("product_id"), "source_store_id": rec.get("source_id"), "target_store_id": rec.get("target_id")}
        decision = unique_match(seller_index, slkey, tuple(slkey))
        source = next(entries[0]["row"] for k, entries in catalog if k == identity)
        ctx = {**base_context, "decision_date": loss.decision_date(source, business["config"])}
        sl = loss_contract(decision, ctx, business["scope"], business["explicit"])
        plan = unique_match(plans.get(primary, {}), {"candidate_id": rec.get("route_id")}, ("candidate_id",)) or {}
        all_modes = {mode: deepcopy(unique_match(index, {"candidate_id": rec.get("route_id")}, ("candidate_id",))) for mode, index in plans.items()}
        qty = number(plan.get("allocated_qty"))
        selected = plan.get("selection_status") in (sf.SELECTED, sf.PARTIALLY_SELECTED) and qty is not None and qty > 0
        reasons = codes(t2.get("reason_codes")) + sl["reason_codes"] + sl["promotion_blockers"] + context_errors
        if not t2: reasons.append("T2_RESULT_UNAVAILABLE")
        if not plan: reasons.append("T1_RESULT_UNAVAILABLE")
        operational = op.get("contract_strict") == "STRICT_ACTUAL_ELIGIBLE" or op.get("contract_eligibility") == "VERIFIED_OPERATIONAL"
        source_entries = next(entries for k, entries in catalog if k == identity)
        namespace_ambiguous = sum(k[2:4] == identity[2:4] for k in valid_keys) > 1
        conflict = bool(t2.get("consistency_status") in ("ACTION_CONFLICT", "REVIEW_REQUIRED") or context_errors or op.get("conflicting_evidence") or op.get("expired_evidence") or len(source_entries) > 1 or namespace_ambiguous)
        if namespace_ambiguous: reasons.append("NAMESPACE_REFERENCE_AMBIGUOUS")
        if len(source_entries) > 1: reasons.append("DUPLICATE_INPUT_KEY")
        if conflict: reasons.extend(codes(t2.get("conflict_codes")))
        sales = unique_match(sales_index, {"store_id": identity[2], "product_id": identity[3]}, ("store_id", "product_id"))
        demand = accepted_demand(sales, sl)
        current_stock = number(source.get("stock_qty", source.get("current_stock")))
        if selected and sl["quantity"] is not None and abs(sl["quantity"] - qty) > 1e-6:
            conflict = True
            reasons.append("SELLER_LOSS_PLAN_QUANTITY_BASIS_DIFFERS")
        plan_unit = text(plan.get("quantity_unit"))
        if selected and plan_unit not in (None, "UNDECLARED") and sl["quantity_unit"] and sf.normalize_unit(plan_unit) != sf.normalize_unit(sl["quantity_unit"]):
            conflict = True
            reasons.append("PLAN_SELLER_UNIT_MISMATCH")
        if selected and primary == sf.STRICT_ACTUAL and plan_unit in (None, "UNDECLARED"):
            conflict = True
            reasons.append("PLAN_QUANTITY_UNIT_UNVERIFIED")
        if selected and not t2:
            conflict = True
            reasons.append("CANONICAL_PLAN_LINK_UNAVAILABLE")
        compared_stock = number(sl["input_provenance"].get("source_current_stock", {}).get("value"))
        if not sl["scenario"] and compared_stock is not None and current_stock is not None and abs(compared_stock - current_stock) > 1e-6:
            conflict = True
            reasons.append("SELLER_STOCK_BASIS_DIFFERS")
        plan_validation = (shared.get("modes", {}).get(primary) or {}).get("validation") or {}
        if selected and (plan_validation.get("violation_count", 0) or current_stock is not None and qty > current_stock + 1e-6):
            conflict = True
            reasons.append("PLAN_CONSTRAINT_VIOLATION")
        if pipeline.get("status") in ("validation_error", "adapter_error"):
            conflict = True
            reasons.append("PRODUCTION_INPUT_VALIDATION_FAILED")
        if sl["promotion_status"] == gate.PROMOTION_ELIGIBLE and not ctx.get("decision_date") and not sl["scenario"]:
            conflict = True
            reasons.append("DECISION_DATE_UNAVAILABLE")
        stock_status = "CONFLICTED" if len(source_entries) > 1 else "AVAILABLE_UNVERIFIED" if current_stock is not None and current_stock >= 0 else "MISSING" if current_stock is None else "INVALID"
        status, flags, winner, ready = classify(seller=sl, selected=selected, mode=primary, operational=operational,
            demand=demand, stock_status=stock_status, conflicts=conflict, reasons=reasons)
        decision_mode = inputs.SCENARIO if sl["scenario"] or sl["nonproduction"] or gate._nonproduction({}, ctx) else sf.BENCHMARK_PROXY if selected and primary == sf.BENCHMARK_PROXY else inputs.ACTUAL_OPERATION
        execution_status = t2.get("execution_state") or "NOT_RECORDED"
        route = {"run_id": run_id, "decision_id": "fdi-route-" + digest((identity, rec.get("route_id")))[:24], "product_key": digest(identity),
            "route_id": rec.get("route_id"), "product_id": identity[3], "location_id": identity[2], "source_id": identity[2], "target_id": rec.get("target_id"),
            "decision_mode": decision_mode, "decision_status": status, "status_flags": flags, "recommended_strategy": winner,
            "recommendation_readiness": ready, "comparison_scope": sl["comparison_scope"], "available_strategies": sl["available_strategies"],
            "excluded_strategies": sl["excluded_strategies"], "seller_loss_status": sl["comparison_status"], "seller_loss_evidence_level": sl["evidence_level"],
            "seller_loss": sl, "expected_loss_by_strategy": sl["expected_loss_by_strategy"], "promotion_status": sl["promotion_status"],
            "expected_loss_values": {s: money(v, sl["currency"], "SellerLossEngine", "EXPECTED_NOT_ACTUAL", kind="EXPECTED_AVOIDABLE_LOSS")
                                     for s, v in sl["expected_loss_by_strategy"].items()},
            "selected": selected, "plan_mode": primary if selected else None, "allocated_qty": qty if selected else None,
            "quantity_unit": text(plan.get("quantity_unit")) if plan.get("quantity_unit") != "UNDECLARED" else None,
            "calculated_transfer_cost": number(plan.get("allocated_move_cost")) if selected else None,
            "transfer_cost": money(plan.get("allocated_move_cost") if selected else None, business["config"].get("currency"),
                                   "T1.allocated_move_cost", plan.get("cost_status"), kind="CALCULATED_TRANSPORT_COST"),
            "selected_transfer_routes": [rec.get("route_id")] if selected else [], "transfer_candidate_status": "SELECTED_PLAN" if selected else "CANDIDATE_ONLY",
            "all_plan_modes": all_modes, "plan_validation": deepcopy((shared.get("modes", {}).get(primary) or {}).get("validation")),
            "operational_evidence": deepcopy(op), "stock_validation_status": stock_status, "demand_validation_status": demand,
            "legacy_action": rec.get("varo_action"), "legacy_action_basis": "LEGACY_RULE_REFERENCE_ONLY",
            "canonical_action": t2.get("action_code"), "operator_action": t2.get("operator_action") if execution_status == "EXECUTED" else None,
            "execution_status": execution_status, "outcome": None, "production_action_applied": False,
            "scenario_recommended_strategy": sl["selected_strategy"] if decision_mode == inputs.SCENARIO and sl["scope_complete"] else None,
            "missing_required_inputs": sl["missing_required_inputs"] + sl["required_context_inputs"], "reason_codes": sorted(set(reasons)),
            "source_lineage": {"T1_candidate_id": plan.get("candidate_id"), "T2_candidate_id": t2.get("candidate_id"),
                               "seller_decision_id": sl["decision_id"], "seller_input_provenance": sl["input_provenance"]},
            "duplicate_input_keys": []}
        route["explanation_ko"] = explain(route)
        route_rows.append(route)
        if route_identity: seen_routes[route_identity] = route
        by_source[identity].append(route)
        for k in valid_keys:
            if k[2] == text(rec.get("target_id")) and k[3] == identity[3] and k[:2] == identity[:2] and selected: incoming[k].append(rec.get("route_id"))
    product_rows = []
    for identity, entries in catalog:
        raw = entries[0]["row"]
        routes = by_source.get(identity, [])
        selected = [r for r in routes if r["selected"]]
        reasons = list(context_errors)
        invalid_key = len(identity) != 4 or not identity[3] or not identity[2]
        duplicate = len(entries) > 1
        stock = number(raw.get("stock_qty", raw.get("current_stock"))) if entries[0]["sheet"] == "inventory" and not duplicate else None
        stock_status = "CONFLICTED" if duplicate else "MISSING" if stock is None else "INVALID" if stock < 0 else TRUST.get(loss.declared_provenance(raw, "stock_qty", "UNKNOWN")[0], "AVAILABLE_UNVERIFIED")
        if invalid_key: reasons.append("INVENTORY_KEY_OR_SNAPSHOT_MISSING")
        if duplicate: reasons.append("DUPLICATE_INPUT_KEY")
        if stock_status in ("MISSING", "INVALID", "CONFLICTED"): reasons.append("STOCK_EVIDENCE_UNAVAILABLE")
        match = {"store_id": identity[2], "product_id": identity[3]}
        sales = unique_match(sales_index, match, tuple(match)) if not duplicate else None
        analyzed = unique_match(inventory_index, match, tuple(match)) or {}
        forecast_contract = unique_match(forecast_index, match, tuple(match)) or {}
        comparisons = [r["seller_loss"] for r in routes]
        if not comparisons and not invalid_key and not duplicate and not context_errors:
            try:
                decision = product_query(identity, raw, business, run_id)
                comparisons = [loss_contract(decision, {**base_context, "decision_date": loss.decision_date(raw, business["config"])}, business["scope"], business["explicit"])]
            except Exception as error: reasons += ["PRODUCT_COMPARISON_QUERY_ERROR", type(error).__name__]
        sl = comparisons[0] if len(comparisons) == 1 else loss_contract(None, {}, business["scope"], business["explicit"])
        demand = accepted_demand(sales, sl)
        mode = inputs.SCENARIO if sl["scenario"] or sl["nonproduction"] or gate._nonproduction({}, base_context) else sf.BENCHMARK_PROXY if selected and primary == sf.BENCHMARK_PROXY else inputs.ACTUAL_OPERATION
        conflicts = duplicate or bool(context_errors) or any(r["decision_status"] == "REVIEW_REQUIRED" for r in routes)
        if len(comparisons) > 1:
            conflicts = True
            reasons.append("MULTI_ROUTE_LOSS_NOT_AGGREGATED")
        status, flags, winner, ready = classify(seller=sl, selected=bool(selected), mode=primary,
            operational=bool(selected) and all(r["operational_evidence"].get("contract_eligibility") == "VERIFIED_OPERATIONAL" for r in selected),
            demand=demand, stock_status=stock_status, conflicts=conflicts, reasons=reasons)
        if invalid_key or stock_status in ("MISSING", "INVALID"):
            status, winner, ready = "INSUFFICIENT_DATA", None, "UNAVAILABLE"
        if not routes: reasons.append("NO_TRANSFER_CANDIDATE")
        if routes and not selected: reasons.append("TRANSFER_CANDIDATE_NOT_SELECTED")
        if not selected and routes and all(any(s in str((r["all_plan_modes"].get(primary) or {}).get("selection_status", "")) for s in
                   ("REJECTED_SOURCE_CAP", "REJECTED_TARGET_CAP", "REJECTED_INFEASIBLE_ROUTE", "REJECTED_ROUTE_CAP")) for r in routes):
            flags = sorted(set(flags) | {"NO_FEASIBLE_TRANSFER"})
        if stock == 0 and stock_status in ("ACTUAL_VERIFIED", "OPERATOR_CONFIRMED", "DERIVED_FROM_REAL") and not routes and not incoming.get(identity) and not conflicts:
            status, winner, ready = "NO_ACTION_REQUIRED", None, "NO_ACTION_REQUIRED"
            sl["missing_required_inputs"] = []
        units = {r["quantity_unit"] for r in selected}
        allocated = sum(r["allocated_qty"] for r in selected) if selected and len(units) == 1 else None
        costs = [r["calculated_transfer_cost"] for r in selected]
        total_cost = sum(costs) if costs and all(c is not None for c in costs) else None
        legacy = list(dict.fromkeys(r["legacy_action"] for r in routes if r["legacy_action"]))
        all_missing = list({digest(i): i
                           for s in comparisons for i in s["missing_required_inputs"] + s["required_context_inputs"]}.values())
        item = {"run_id": run_id, "decision_id": "fdi-product-" + digest(identity)[:24], "product_key": digest(identity),
            "dataset_id": identity[0], "business_id": identity[1], "product_id": identity[3], "location_id": identity[2],
            "product_name": text(raw.get("product_name")), "current_stock": stock, "stock_validation_status": stock_status,
            "demand_validation_status": demand, "transfer_candidate_status": "SELECTED_PLAN" if selected else "CANDIDATE_ONLY" if routes else "NO_CANDIDATE",
            "seller_loss_status": sl["comparison_status"] if len(comparisons) <= 1 else "MULTIPLE_ROUTE_COMPARISONS",
            "decision_mode": mode, "decision_status": status, "status_flags": sorted(set(flags + [status])), "recommended_strategy": winner,
            "recommendation_readiness": ready, "comparison_scope": sl["comparison_scope"], "available_strategies": sl["available_strategies"],
            "excluded_strategies": sl["excluded_strategies"], "seller_loss_evidence_level": sl["evidence_level"], "promotion_status": sl["promotion_status"],
            "expected_loss_by_strategy": sl["expected_loss_by_strategy"], "seller_comparisons": comparisons,
            "expected_loss_values": {s: money(v, sl["currency"], "SellerLossEngine", "EXPECTED_NOT_ACTUAL", kind="EXPECTED_AVOIDABLE_LOSS")
                                     for s, v in sl["expected_loss_by_strategy"].items()},
            "selected_transfer_routes": [r["route_id"] for r in selected], "incoming_transfer_routes": list(dict.fromkeys(incoming.get(identity, []))),
            "allocated_qty": allocated, "quantity_unit": next(iter(units)) if len(units) == 1 else None,
            "allocated_qty_status": "UNIT_UNDECLARED" if selected and None in units else "MIXED_UNITS_NOT_SUMMED" if len(units) > 1 else "PLANNED_NOT_EXECUTED" if selected else "NO_PLAN",
            "calculated_transfer_cost": total_cost, "transport_cost_currency": business["config"].get("currency"),
            "actual_spend": None, "actual_revenue": None, "actual_loss": None, "actual_saving": None,
            "legacy_action": legacy[0] if len(legacy) == 1 else None, "legacy_actions": legacy, "legacy_action_basis": "LEGACY_RULE_REFERENCE_ONLY",
            "canonical_action": routes[0]["canonical_action"] if len(routes) == 1 else None,
            "operator_action": None, "execution_status": "ROUTE_DETAILS_ONLY" if any(r["operator_action"] for r in routes) else "NOT_RECORDED", "outcome": None,
            "scenario_recommended_strategy": sl["selected_strategy"] if mode == inputs.SCENARIO and sl["scope_complete"] else None,
            "production_action_applied": False, "input_quality": {"stock": stock_status, "demand": demand,
                "effective_evidence": "SCENARIO" if mode == inputs.SCENARIO else "PROXY" if mode == sf.BENCHMARK_PROXY or demand == "PROXY" else "INSUFFICIENT" if demand == "INSUFFICIENT" else demand,
                "quantity_unit": text(raw.get("quantity_unit")) or business["config"].get("quantity_unit"), "duplicate_input": duplicate,
                "sales_30d": number((sales or {}).get("sales_30d")), "sales_reason_codes": codes((sales or {}).get("reason_codes"))},
            "missing_required_inputs": [] if status == "NO_ACTION_REQUIRED" else all_missing, "reason_codes": sorted(set(reasons + sl["reason_codes"])),
            "source_lineage": [{"sheet": e["sheet"], "row_position": e["row_position"], "source_row_id": text(e["row"].get("source_row_id")),
                                "source_file": text(e["row"].get("source_file")), "source_dataset": text(e["row"].get("source_dataset"))} for e in entries],
            "route_decision_ids": [r["decision_id"] for r in routes],
            "reference_metrics": {"validated_turnover": deepcopy(unique_match(turnover_index, match, tuple(match))),
                "abc_basis": (metric.get("validated_abc") or {}).get("basis"), "abc_status": (metric.get("validated_abc") or {}).get("gate_status"),
                "legacy_disposal_score": number(analyzed.get("disposal_risk_score")), "disposal_meaning": "RULE_BASED_SCORE_NOT_PROBABILITY",
                "actual_disposal_probability": None},
            "forecast": {"version": text(analyzed.get("demand_forecast_version")), "fallback_reason": text(analyzed.get("demand_forecast_fallback_reason")),
                "reason": text(analyzed.get("demand_forecast_reason")), "history_coverage": (sales or {}).get("windows"),
                "input_meaning": "VALIDATED_RETAIL_HISTORY" if forecast_contract.get("retail_sales_history_available") else "UNVERIFIED_AGGREGATE_OR_PROXY_INPUT",
                "input_contract": deepcopy(forecast_contract),
                "limitation": "dataset-specific results; external generalization not established; no model changed"}}
        item["recommended_next_step"] = "현재 조치 불필요" if status == "NO_ACTION_REQUIRED" else "검토" if status == "REVIEW_REQUIRED" else "추가 입력" if all_missing else "계획 확인" if selected else "입력 근거 확인"
        item["display_order_key"] = [STATUSES.index(status), str(identity[3] or ""), str(identity[2] or "")]
        item["explanation_ko"] = explain(item)
        item["display"] = {"product": item["product_name"] or item["product_id"], "situation": item["decision_status"],
            "recommendation_or_plan": LABELS.get(winner) if winner else "추정 이동 계획" if selected and mode == sf.BENCHMARK_PROXY else "이동 계획" if selected else item["recommended_next_step"],
            "expected_effect": item["expected_loss_by_strategy"], "confirmation_needed": [i.get("label") or i.get("field") for i in item["missing_required_inputs"]],
            "evidence_status": {"mode": mode, "effective_evidence": item["input_quality"]["effective_evidence"],
                                "stock_declared_basis": stock_status, "demand_declared_basis": demand, "promotion": sl["promotion_status"]}}
        product_rows.append(item)
    coverage = coverage_check(catalog, product_rows)
    primary_selected = [r for r in route_rows if r["selected"]]
    return {"status": "parallel_only", "version": VERSION, "run_id": run_id, "production_action_applied": False,
            "legacy_action_replaced": False, "products": product_rows, "routes": route_rows, "coverage": coverage,
            "orphan_routes": orphans, "context_errors": context_errors,
            "technical_metadata": {"benchmark_integrity": deepcopy((pipeline.get("validation_report") or {}).get("optimality_gap")),
                "ranking_policy": "unchanged service -> cost -> VHS tie-break", "forecast_model_changed": False,
                "minimum_loss_scope": "SellerLoss requested scope, not a global optimization claim",
                "ledger_compatibility": "read-only identifiers; no DB write; existing ledger opt-in unchanged"},
            "summary": {"product_location_count": len(product_rows), "unique_products": len({r["product_id"] for r in product_rows if r["product_id"]}),
                "evaluated_candidates": len(pipeline.get("recommendations") or []), "linked_candidates": len(route_rows),
                "eligible_candidates": sum(r["operational_evidence"].get("contract_eligibility") == "VERIFIED_OPERATIONAL" for r in route_rows),
                "selected_transfer_routes": len(primary_selected), "proxy_plans": sum(r["plan_mode"] == sf.BENCHMARK_PROXY for r in primary_selected),
                "strict_actual_plans": sum(r["plan_mode"] == sf.STRICT_ACTUAL for r in primary_selected),
                "minimum_loss_recommendations": sum(bool(r["recommended_strategy"]) for r in product_rows),
                "additional_input_products": sum(bool(r["missing_required_inputs"]) for r in product_rows),
                "status_counts": dict(Counter(r["decision_status"] for r in product_rows)), "missing_products": len(coverage["missing"]),
                "duplicate_products": len(coverage["duplicates"]), "production_changes": 0},
            "limits": ["planning is not a monetary winner or executed action", "multiple-route losses are never summed",
                       "product queries without routes never invent quantities/targets", "no operational approval or outcome is created"]}


def safe_build_final_decision_integration(data, pipeline, analyzed_inventory=None):
    try: return build_final_decision_integration(data, pipeline, analyzed_inventory)
    except Exception as error:
        # Even a new optional integration bug cannot hide uploaded products or break production.
        catalog = product_catalog(data or {})
        rows = [{"product_key": digest(k), "product_id": k[3], "location_id": k[2], "decision_status": "INSUFFICIENT_DATA",
                 "recommended_strategy": None, "operator_action": None, "outcome": None, "production_action_applied": False,
                 "reason_codes": ["INTEGRATION_ERROR"], "explanation_ko": "통합 판단을 완료하지 못했습니다. 기존 분석 결과는 유지됩니다."} for k, _ in catalog]
        return {"status": "error", "version": VERSION, "error_type": type(error).__name__, "error_message": str(error),
                "products": rows, "routes": [], "coverage": coverage_check(catalog, rows), "production_action_applied": False}
