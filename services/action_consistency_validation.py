"""Action consistency validation: real Suhyup 31 days (620 candidates), the repository workbooks, the 2026-07-31
production E2E and the controlled scenarios A-T.

Validation only. It reads the saved Suhyup candidates and inventory, the 186-row reference and the T1 bundle, and never
writes there. Outputs go to <data-root>/_ACTION_CONSISTENCY_VALIDATION (local only, never into git); a re-run replaces
only OUTPUT_FILES.

Per Suhyup day the inputs are the T1 bundle's (shared_feasibility_validation), unchanged:
  * ranked_day: the saved 20 candidates, Varo Final rank recomputed as the offline revalidation does
  * legacy action: the saved production varo_action (the legacy rule label recorded when the snapshot was generated).
    The current pipeline reproduces it 20/20 on 2026-07-31 (the only day with a production upload, checked in E2E).
  * plans: SF_BENCHMARK_PARTIAL and SF_STRICT_ACTUAL (same caps, policy and tariff recompute as the T1 bundle)
  * Seller Loss: the strict engine on Suhyup real fields (seller_decision_validation.suhyup_seller_decisions)
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from unittest import mock

import pandas as pd

from services import action_consistency as ac
from services import shared_feasibility_selection as sf
from services import shared_feasibility_validation as sfv
from services import seller_loss_promotion_gate as gate
from services import suhyup_algorithm_revalidation as rv
from services.real_data_adapters import DATA_ROOT
from services.seller_decision_validation import _env
from services.seller_loss_engine import STRATEGIES

OUTPUT_FOLDER = "_ACTION_CONSISTENCY_VALIDATION"
OUTPUT_FILES = (
    "action_consistency_summary.csv",
    "action_consistency_by_day.csv",
    "action_consistency_conflicts.csv",
    "action_consistency_examples.csv",
    "action_consistency_validation.json",
    "action_mapping_contract.json",
    "action_consistency_candidates.csv",
)
REAL_DATA_ROOT_ENV = "VARO_REAL_DATA_ROOT"

# ------------------------------------------------------------------------------------------------ code inventory

# Where every action-like field comes from (read from the code, not the README). Classes:
# A computed move plan, B rule-based recommendation, C inventory-state advice, D Seller Loss comparison,
# E operator execution, F uncertain / not comparable, RANK priority only, NOT_ACTION same word but no action.
FIELD_INVENTORY: tuple[dict[str, str], ...] = (
    {"field": "vhs_action", "producer": "legacy_adapters/_local_modules/varo_hybrid_score.py _recommend_action (via calculate_varo_hybrid_score)",
     "rule": "priority 폐기 > 재배치 이동 > 할인 판매 > 보류 on source-row grades (disposal, turnover, ABC, match, reorder, trend) "
             "plus vhs_raw >= 70 inside the transfer branch",
     "downstream": "normalize_action -> varo_action in analysis_pipeline._finalize_candidate_columns",
     "exposed": "vhs_analysis.score_rows (validation page)", "plan_link": "none: judged on the source inventory row",
     "meaning": "B+C", "finding": "transfer branch unreachable: match_score max 57.5 < GOOD 60 because matching runs on source rows"},
    {"field": "varo_action", "producer": "analysis_pipeline._finalize_candidate_columns then recommendation_adapter.normalize_standard_recommendation",
     "rule": "normalize_action(vhs_action or final_recommendation)", "downstream": "reason text, seller loss legacy_action, shadow ledger production_action",
     "exposed": "'Varo 추천' on cards (components/cards.py) and tables (components/tables.py), export 'Varo 추천', "
                "simulation_history routes.action / final_strategy",
     "plan_link": "none", "meaning": "B+C (F when '비교 불가')", "finding": "Suhyup 620: 할인 535 / 보류 85 / 재고 이동 0 on transfer routes"},
    {"field": "reason", "producer": "analysis_pipeline._finalize_candidate_columns",
     "rule": "'{situation} 상황과 ... {varo_action} 전략을 권장합니다.'", "downstream": "-", "exposed": "'추천 이유' on cards",
     "plan_link": "none", "meaning": "B (text of varo_action)", "finding": "repeats the legacy label as a recommendation"},
    {"field": "greedy_action / greedy_strategy / final_recommendation", "producer": "legacy_adapters/data_adapter.build_candidate_frame, "
     "_finalize_candidate_columns, vhs_score_engine.apply_auto_vhs",
     "rule": "pass-through of the uploaded varo_action/greedy_action, else '재고 이동'/'비교 불가'; heuristic_optimizer ranks, never chooses an action",
     "downstream": "heuristic strategy_score keyword bonus", "exposed": "'Greedy 전략' column", "plan_link": "none",
     "meaning": "F (pass-through label)", "finding": "Suhyup 620/620 '비교 불가'"},
    {"field": "dqn_action", "producer": "_finalize_candidate_columns sentinel '미연결'; dqn_service ACTION_LABELS (8) only for button training",
     "rule": "-", "downstream": "strategy comparison", "exposed": "'DQN 상태'", "plan_link": "none", "meaning": "F", "finding": "never a production action"},
    {"field": "varo_final_rank / rank / varo_final_decision", "producer": "vhs_score_engine._rank_varo_operational / apply_auto_vhs",
     "rule": "recommended_qty desc, move cost asc, vhs_rank, route_id; rank 1 -> '최종 추천'", "downstream": "sort_recommendations, Top-5",
     "exposed": "order of every list, 'Varo 최종 추천' in strategy comparison", "plan_link": "none: no shared caps",
     "meaning": "RANK", "finding": "'최종 추천' can sit on a move the shared selection rejects"},
    {"field": "top5", "producer": "analysis_pipeline.top_recommendations(limit=5)", "rule": "first five of the Varo Final order",
     "downstream": "home / recommendations Top-5 table", "exposed": "Top-5 rows with recommended_qty", "plan_link": "none",
     "meaning": "RANK", "finding": "violates shared caps on 5/31 Suhyup days (T1)"},
    {"field": "pareto_status / pareto_selected", "producer": "vhs_score_engine.apply_auto_vhs -> pareto_service.select_pareto_routes",
     "rule": "frontier + ideal-point distance, up to 5, pareto _default_feasible", "downstream": "pareto_analysis.selected_route_ids",
     "exposed": "'운영 선택' status in strategy comparison", "plan_link": "parallel strategy", "meaning": "A (alternative strategy)",
     "finding": "pareto_selected is dropped by the adapter; the '운영 선택' text survives in pareto_status"},
    {"field": "shared_feasibility_selection rows", "producer": "services/shared_feasibility_selection.build_shared_feasibility_analysis",
     "rule": "Varo Final key with executable qty first, shared caps, STRICT_ACTUAL / BENCHMARK_PROXY", "downstream": "-",
     "exposed": "none (parallel field)", "plan_link": "this is the plan", "meaning": "A", "finding": "only computed plan in the pipeline"},
    {"field": "promotion_recommended", "producer": "promotion_analyzer.analyze_promotion_vs_transfer final_decision via _enrich_promotion (name join)",
     "rule": "transfer_cost <= promotion net cost -> '재배치 추천' else '프로모션 추천'; discount 20%, uplift 80%, unit cost 1,000, holding 20/day placeholders",
     "downstream": "VHS promotion_score, v2_summaries reason sentence", "exposed": "validation page '프로모션 권장 여부', reason text",
     "plan_link": "none", "meaning": "B", "finding": "Suhyup: '재배치 추천' 407 / blank 213 while varo_action says 할인/보류"},
    {"field": "seller_loss_action / recommended_strategy", "producer": "seller_loss_engine._finish (build_seller_loss_analysis)",
     "rule": "min expected avoidable loss over TRANSFER / NORMAL_SALE / DISCOUNT_SALE, robust over unknown ranges",
     "downstream": "promotion gate, shadow ledger", "exposed": "none (parallel field)", "plan_link": "same route, recommended_qty basis",
     "meaning": "D", "finding": "Suhyup: COMPARISON_UNAVAILABLE 620/620"},
    {"field": "promotion_status / final_action_candidate", "producer": "seller_loss_promotion_gate.shadow_decision",
     "rule": "qualification only", "downstream": "shadow ledger", "exposed": "none", "plan_link": "-", "meaning": "D",
     "finding": "production_action_applied=False always"},
    {"field": "ledger legacy_action / production_action", "producer": "seller_shadow_ledger.decision_record",
     "rule": "copy of varo_action; save_decision refuses production_action != legacy_action", "downstream": "-",
     "exposed": "ledger export", "plan_link": "none", "meaning": "B+C (copied)", "finding": "schema kept; no canonical field written"},
    {"field": "operator_action / execution_status", "producer": "seller_shadow_outcomes.parse_seller_outcomes (ledger import)",
     "rule": "EXECUTED needs operator_action, executed_at, observed provenance and source", "downstream": "compare_outcome",
     "exposed": "ledger evaluations", "plan_link": "ledger decision_id", "meaning": "E", "finding": "0 real outcomes"},
    {"field": "simulation_history routes.executed / final_strategy", "producer": "simulation_history (inventory transition)",
     "rule": "executed = the simulated transition applied the move; final_strategy = most common varo_action",
     "downstream": "history page", "exposed": "history", "plan_link": "simulation only", "meaning": "NOT_ACTION (simulation, not operator execution)",
     "finding": "'executed' here is a simulation result, never E"},
    {"field": "recommendation_grade / grade '보류'", "producer": "_finalize_candidate_columns / vhs_score_engine._grade",
     "rule": "auto VHS score band (< 50 -> '보류')", "downstream": "-", "exposed": "'추천 등급'", "plan_link": "none",
     "meaning": "NOT_ACTION", "finding": "same word as the legacy '보류' action (Suhyup 274/620)"},
)
EXISTING_MAPPINGS = {
    "seller_loss_engine.LEGACY_ACTION_TO_STRATEGY": {"재고 이동": "TRANSFER", "할인": "DISCOUNT_SALE", "긴급 할인": "DISCOUNT_SALE", "보류": "NORMAL_SALE"},
    "seller_loss_promotion_gate.ACTION_MAPPING": "재고 이동/할인/정상 판매(유지) only; 긴급할인, 1+1, 폐기, 보류, 비교 불가, 유지 unmappable",
    "canonical (this contract)": "보류 -> HOLD (not NORMAL_SALE); 긴급 할인 -> URGENT_DISCOUNT (not DISCOUNT_SALE); exact text only",
    "finding": "the engine's agreement_with_legacy maps 보류 to NORMAL_SALE and 긴급 할인 to DISCOUNT_SALE, while the gate and the "
               "canonical contract do not. The engine field is kept as is (Seller Loss is frozen); the gate/canonical mapping is used here.",
    "normalize_action substring pitfall": "'이동 비추천' -> '재고 이동', 'threshold' -> '보류' (substring tokens); the canonical mapping is exact",
}
UI_ADAPTER_CONTRACT = {
    "status": "DESIGN_ONLY_NOT_CONNECTED (UI unchanged; replacement needs separate approval)",
    "current_consumers": {
        "components/cards.py render_recommendation_summary": "'Varo 추천' = varo_action, '추천 수량' = recommended_qty, '추천 이유' = reason",
        "components/tables.py build_recommendation_rows / build_top5_rows": "Top-5 = sort_recommendations slice; '수량' = recommended_qty; 'Varo 추천' = varo_action",
        "pages/recommendations.py / vhs_score_engine.build_strategy_comparison": "'Varo 최종 추천' = varo_final_decision",
        "pages/validation.py": "'프로모션 권장 여부' = promotion_recommended",
        "services/v2_summaries.recommendation_reason": "'재배치가 유리합니다' sentence from promotion_recommended",
        "services/export_service.py": "'Varo 추천' = varo_action",
        "services/simulation_history.py": "routes.action = varo_action, final_strategy = most common varo_action",
        "analysis_pipeline.calculate_overview_kpis": "total_recommended_qty = sum of recommended_qty over all candidates",
    },
    "adapter": {
        "lookup": "pipeline_result.action_consistency.records keyed by candidate_id (= route_id)",
        "'Varo 추천'": "action_label_ko + action_status badge (FEASIBLE_PLAN 검증 계획 / PROXY_PLAN 추정 기준 계획 / NOT_SELECTED 이동 계획 없음 / "
                     "INSUFFICIENT_DATA / COMPARISON_UNAVAILABLE / EXECUTED); the legacy label moves to '기존 규칙 권고' (legacy_action, legacy_action_source)",
        "'추천 수량'": "plan rows show allocated_qty as '계획 수량'; recommended_qty stays '후보 수량'",
        "'추천 이유'": "explanation_ko",
        "Top-5": "records with decision_state SELECTED_PLAN ordered by shared_feasibility_rank (plan), product_summaries for product cards",
        "KPI": "action_consistency.total_allocated_qty next to the existing total_recommended_qty, never replacing it silently",
        "review badge": "consistency_status + conflict_codes",
        "Seller Loss": "seller_loss_claim sentence only; never the action",
        "execution": "operator records from the shadow ledger only (operator_action stays NULL otherwise)",
    },
    "compatibility": "existing fields keep their meaning and golden digests; the canonical fields are additive",
}

# ------------------------------------------------------------------------------------------------ controlled scenarios

_SPEC_COLUMNS = {sf.SOURCE_STOCK: ("source_stock",), sf.SOURCE_SURPLUS: ("source_surplus",), sf.TARGET_NEED: ("target_need",)}


def _rec(route: str, qty: float = 50, action: str | None = "재고 이동", *, product: str = "P1", source: str = "S1",
         target: str | None = "T1", rank: float | None = None, need: float = 100, surplus: float = 100, **extra) -> dict[str, Any]:
    row = {"route_id": route, "product_id": product, "product_name": f"상품{product}", "source_id": source,
           "source_name": f"점포{source}", "target_id": target, "target_name": f"점포{target}" if target else None,
           "route_type": "DIRECT", "recommended_qty": qty, "move_cost": 1000.0, "varo_action": action,
           "varo_final_rank": rank, "varo_final_decision": "최종 추천" if rank == 1 else "후보",
           "source_stock": 500, "source_surplus": surplus, "target_need": need, "quantity_unit": "EA",
           "source_stock_unit": "EA", "source_surplus_unit": "EA", "target_need_unit": "EA"}
    row.update(extra)
    return row


def _shared(records: Sequence[Mapping[str, Any]], provenance: str = "USER_INPUT") -> dict[str, Any]:
    spec = {kind: (columns, "DIRECT_REAL" if kind == sf.SOURCE_STOCK else provenance, "controlled scenario cap")
            for kind, columns in _SPEC_COLUMNS.items()}
    caps = sf.caps_from_columns(records, spec)
    return {"status": "parallel_only", "modes": {mode: sf.select_shared_feasible(
        records, caps, mode=mode, max_routes=5, partial_policy=sf.PARTIAL_IF_SAFE) for mode in sf.MODES}}


def _seller_decision(route: str, *, product: str = "P1", source: str = "S1", target: str = "T1", note: str = "",
                     missing_price: bool = False, legacy: str = "재고 이동") -> dict[str, Any]:
    """Real engine output on an artificial complete DIRECT_REAL contract (the promotion-gate test pattern)."""
    from services import seller_loss_inputs as sli
    from services.seller_decision_validation import scenario_input
    from services.seller_loss_engine import InputField

    base = scenario_input()
    fields = {name: replace(item, provenance="DIRECT_REAL" if item.present else "MISSING", source="verified business record",
                            dataset="uploaded_workbook", note=note) for name, item in base.input_fields().items()}
    if missing_price:
        fields["source_normal_price"] = InputField(source="no price")
    inp = replace(base, **fields, decision_id=route, product_id=product, source_store_id=source, target_store_id=target,
                  legacy_action=legacy)
    return sli.evaluate_with_seller_inputs(inp, None, decision_date="2026-07-31")


def _seller(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "parallel_only", "decisions": list(decisions),
            "shadow_decisions": [gate.shadow_decision(d, context={"decision_date": "2026-07-31"}) for d in decisions]}


def _event(route: str, status: str = "EXECUTED", **extra) -> dict[str, Any]:
    row = {"route_id": route, "product_id": "P1", "source_id": "S1", "target_id": "T1", "operator_action": "재고 이동",
           "execution_status": status, "executed_at": "2026-07-31T10:00:00", "recorded_at": "2026-07-31T12:00:00",
           "outcome_provenance": "OPERATOR_CONFIRMED", "outcome_source": "store transfer slip 0731-01", "data_mode": "PRODUCTION"}
    row.update(extra)
    return row


def _scenario(scenario_id: str, title: str, build: Callable[[], dict[str, Any]], expect: Mapping[str, Mapping[str, Any]],
              forged: Sequence[tuple[str, Mapping[str, Any], Sequence[str]]] = (), result_expect: Mapping[str, Any] | None = None,
              explanation_contains: Mapping[str, Sequence[str]] | None = None,
              explanation_excludes: Mapping[str, Sequence[str]] | None = None,
              invariant_failures: Sequence[str] = ()) -> dict[str, Any]:
    return {"id": scenario_id, "title": title, "build": build, "expect": expect, "forged": forged,
            "result_expect": result_expect or {}, "explanation_contains": explanation_contains or {},
            "explanation_excludes": explanation_excludes or {}, "invariant_failures": list(invariant_failures)}


def _plain(records: Sequence[dict[str, Any]], **kwargs) -> dict[str, Any]:
    return {"recommendations": records, "shared_feasibility": _shared(records, kwargs.pop("provenance", "USER_INPUT")), **kwargs}


def _scenario_d() -> dict[str, Any]:
    records = [_rec("R1", 50, "재고 이동", rank=1, need=50), _rec("R2", 50, "재고 이동", source="S2", rank=2, need=50, allow_partial=False)]
    return _plain(records, legacy_top_ids=["R1", "R2"])


def _scenario_g() -> dict[str, Any]:
    plan_records = [_rec("R1", 50, rank=1)]
    return {"recommendations": [*plan_records, _rec("R9", 30, "재고 이동", target="T9", rank=2)], "shared_feasibility": _shared(plan_records)}


def _scenario_i() -> dict[str, Any]:
    plan_records = [_rec("R1", 50, rank=1, product="P2")]
    return {"recommendations": [_rec("R1", 50, rank=1)], "shared_feasibility": _shared(plan_records),
            "seller_loss": _seller([_seller_decision("R1", product="P2")])}


def _scenario_m() -> dict[str, Any]:
    records = [_rec("R1", 20, "할인", rank=1, need=20), _rec("R2", 20, "할인", source="S2", rank=2, need=20, allow_partial=False)]
    return _plain(records, seller_loss=_seller([_seller_decision("R1", legacy="할인"),
                                                _seller_decision("R2", source="S2", legacy="할인")]))


CONTROLLED_SCENARIOS: tuple[dict[str, Any], ...] = (
    _scenario("A", "transfer selected + legacy transfer", lambda: _plain([_rec("R1", 50, "재고 이동", rank=1)]),
              {"R1": {"action_code": "TRANSFER", "action_label_ko": "재고 이동", "action_status": ac.FEASIBLE_PLAN,
                      "allocated_qty": 50.0, "conflict_codes": [], "consistency_status": ac.CONSISTENT,
                      "legacy_alignment": ac.EXACT_MATCH, "selected_route_id": "R1", "is_executable": True}},
              explanation_contains={"R1": ["점포S1의 상품P1 50개를 점포T1로 이동하는 계획입니다."]}),
    _scenario("B", "transfer selected + legacy discount", lambda: _plain([_rec("R1", 50, "할인", rank=1)]),
              {"R1": {"action_code": "TRANSFER", "action_status": ac.FEASIBLE_PLAN, "allocated_qty": 50.0,
                      "legacy_action_code": "DISCOUNT_SALE", "production_action": "할인",
                      "conflict_codes": ["A_TRANSFER_PLAN_VS_LEGACY_DISCOUNT", "I_LEGACY_VS_CANONICAL_MISMATCH"],
                      "consistency_status": ac.REVIEW_REQUIRED, "legacy_alignment": ac.MISMATCH}},
              explanation_contains={"R1": ["재고 이동 계획과 기존 할인 권고가 달라 추가 검토가 필요합니다."]}),
    _scenario("C", "transfer selected + legacy hold", lambda: _plain([_rec("R1", 50, "보류", rank=1)]),
              {"R1": {"action_code": "TRANSFER", "legacy_action_code": "HOLD",
                      "conflict_codes": ["B_TRANSFER_PLAN_VS_LEGACY_HOLD", "I_LEGACY_VS_CANONICAL_MISMATCH"],
                      "consistency_status": ac.REVIEW_REQUIRED}},
              explanation_contains={"R1": ["재고 이동 계획과 기존 보류 권고가 달라 추가 검토가 필요합니다."]}),
    _scenario("D", "transfer not selected + legacy transfer (target need consumed by R1)", _scenario_d,
              {"R1": {"action_code": "TRANSFER", "action_status": ac.FEASIBLE_PLAN, "conflict_codes": []},
               "R2": {"action_code": None, "action_label_ko": "이동 계획 없음(미선택)", "action_status": ac.NOT_SELECTED,
                      "allocated_qty": 0.0, "selection_status": sf.REJECTED_TARGET_CAP, "selected_route_id": None,
                      "conflict_codes": ["C_LEGACY_TRANSFER_LABEL_NOT_SELECTED", "C_LEGACY_TOP5_NOT_SELECTED",
                                         "I_LEGACY_VS_CANONICAL_MISMATCH"],
                      "consistency_status": ac.ACTION_CONFLICT, "legacy_alignment": ac.MISMATCH}},
              explanation_excludes={"R2": ["이동하는 계획입니다"]}),
    _scenario("E", "partial allocation 20 of 50",
              lambda: _plain([_rec("R1", 50, "재고 이동", rank=1, need=20, transfer_cost_basis="PER_UNIT")]),
              {"R1": {"action_code": "TRANSFER", "allocated_qty": 20.0, "recommended_qty": 50.0,
                      "selection_status": sf.PARTIALLY_SELECTED, "action_status": ac.FEASIBLE_PLAN}},
              explanation_contains={"R1": ["점포S1의 상품P1 20개를 점포T1로 이동하는 계획입니다.",
                                           "기존 추천 수량 50개 중 공동 제약을 반영해 20개만 배정했습니다."]},
              explanation_excludes={"R1": ["50개를 점포T1로"]}),
    _scenario("F", "zero quantity: target need 0", lambda: _plain([_rec("R1", 50, "재고 이동", rank=1, need=0)]),
              {"R1": {"action_code": None, "action_status": ac.NOT_SELECTED, "allocated_qty": 0.0,
                      "conflict_codes": ["C_LEGACY_TRANSFER_LABEL_NOT_SELECTED", "C_FINAL_RECOMMENDATION_NOT_SELECTED",
                                         "I_LEGACY_VS_CANONICAL_MISMATCH"]}},
              forged=[("R1", {"action_code": "TRANSFER", "action_source": "SHARED_FEASIBILITY", "allocated_qty": 0.0},
                       ["D_TRANSFER_PLAN_WITHOUT_QUANTITY"])],
              explanation_excludes={"R1": ["이동하는 계획입니다"]}),
    _scenario("G", "route missing from the plan", _scenario_g,
              {"R1": {"action_status": ac.FEASIBLE_PLAN},
               "R9": {"action_code": None, "action_status": ac.COMPARISON_UNAVAILABLE, "decision_state": ac.COMPARISON_UNAVAILABLE,
                      "consistency_status": ac.NOT_COMPARABLE, "legacy_alignment": ac.INSUFFICIENT_EVIDENCE}},
              explanation_contains={"R9": ["공동 제약 선택 결과가 연결되지 않아"]}),
    _scenario("H", "target missing", lambda: _plain([_rec("R1", 50, "재고 이동", rank=1, target=None)]),
              {"R1": {"action_code": None, "action_status": ac.INSUFFICIENT_DATA, "selection_status": sf.REJECTED_INVALID_INPUT,
                      "action_label_ko": "이동 판단 불가(근거 부족)", "legacy_alignment": ac.INSUFFICIENT_EVIDENCE}},
              explanation_contains={"R1": ["입력값 오류"]}),
    _scenario("I", "product mismatch between layers", _scenario_i,
              {"R1": {"action_code": None, "action_status": ac.COMPARISON_UNAVAILABLE,
                      "conflict_codes": ["E_KEY_MISMATCH"], "consistency_status": ac.ACTION_CONFLICT,
                      "seller_loss_claim": "NOT_EVALUATED"}},
              invariant_failures=["I1", "I4"]),
    _scenario("J", "two routes of one product",
              lambda: _plain([_rec("R1", 20, "재고 이동", rank=1, target="TB"), _rec("R2", 10, "재고 이동", rank=2, target="TC")]),
              {"R1": {"allocated_qty": 20.0, "action_status": ac.FEASIBLE_PLAN},
               "R2": {"allocated_qty": 10.0, "action_status": ac.FEASIBLE_PLAN}},
              result_expect={"product_summaries.P1": {"plan_count": 2, "total_allocated_qty": 30.0, "multi_route": True,
                                                      "route_ids": ["R1", "R2"], "conflict_codes": [],
                                                      "label_ko": "상품P1: 이동 계획 2건 (점포S1→점포TB 20개, 점포S1→점포TC 10개), 합계 30개"},
                             "multi_route_products": ["P1"]}),
    _scenario("K", "PROXY feasibility (BENCHMARK_PROXY backs the plan)",
              lambda: _plain([_rec("R1", 50, "재고 이동", rank=1)], provenance="PROXY"),
              {"R1": {"action_code": "TRANSFER", "action_status": ac.PROXY_PLAN, "is_executable": False,
                      "evidence_level": "PROXY_CAPS", "selection_mode": sf.BENCHMARK_PROXY,
                      "strict_selection_status": sf.INSUFFICIENT_CAP_DATA}},
              forged=[("R1", {"is_executable": True}, ["F_PROXY_PLAN_CLAIMED_EXECUTABLE"]),
                      ("R1", {"explanation_ko": ["실행이 보장된 계획입니다."]}, ["F_PROXY_PLAN_CLAIMED_EXECUTABLE"])],
              result_expect={"primary_selection_mode": sf.BENCHMARK_PROXY},
              explanation_contains={"R1": ["추정 수요·재고(PROXY) 기준으로 계산됐으므로 실제 점포 수요와 이동 가능 재고 확인이 필요합니다."]}),
    _scenario("L", "STRICT_ACTUAL feasibility", lambda: _plain([_rec("R1", 50, "재고 이동", rank=1)], provenance="DIRECT_REAL"),
              {"R1": {"action_status": ac.FEASIBLE_PLAN, "is_executable": True, "evidence_level": "VALIDATED_CAPS",
                      "selection_mode": sf.STRICT_ACTUAL}},
              result_expect={"primary_selection_mode": sf.STRICT_ACTUAL},
              explanation_contains={"R1": ["검증된 재고·수요 기준을 충족한 계획입니다."]}),
    _scenario("M", "Seller Loss RECOMMENDABLE (TRANSFER) next to the plan", _scenario_m,
              {"R1": {"action_code": "TRANSFER", "action_source": "SHARED_FEASIBILITY", "seller_loss_claim": "MIN_LOSS_FULL",
                      "seller_loss_min_loss_claim": True, "seller_loss_strategy": "TRANSFER",
                      "conflict_codes": ["A_TRANSFER_PLAN_VS_LEGACY_DISCOUNT", "I_LEGACY_VS_CANONICAL_MISMATCH"]},
               "R2": {"action_code": None, "action_status": ac.NOT_SELECTED,
                      "conflict_codes": ["PLAN_VS_SELLER_LOSS_DIFFERENT"], "consistency_status": ac.REVIEW_REQUIRED}},
              explanation_contains={"R1": ["금액 비교(이동·정상 판매·할인)에서는 재고 이동이 예상 손실이 가장 작았습니다"]}),
    _scenario("N", "Seller Loss COMPARISON_UNAVAILABLE",
              lambda: _plain([_rec("R1", 20, "할인", rank=1)], seller_loss=_seller([_seller_decision("R1", missing_price=True, legacy="할인")])),
              {"R1": {"seller_loss_comparison_status": "COMPARISON_UNAVAILABLE", "seller_loss_claim": "NO_COMPARISON",
                      "seller_loss_min_loss_claim": False, "seller_loss_action": None}},
              forged=[("R1", {"seller_loss_min_loss_claim": True}, ["G_MIN_LOSS_CLAIM_WITHOUT_COMPARISON"])],
              explanation_contains={"R1": ["입력 부족으로 금액 기반 손실 비교를 하지 못했습니다(최소손실 판단 없음)."]},
              explanation_excludes={"R1": ["예상 손실이 가장 작"]}),
    _scenario("O", "Promotion Gate eligible but not applied",
              lambda: _plain([_rec("R1", 20, "보류", rank=1)], seller_loss=_seller([_seller_decision("R1", legacy="보류")])),
              {"R1": {"seller_loss_promotion_status": "PROMOTION_ELIGIBLE", "production_action": "보류",
                      "production_action_applied": False, "seller_loss_production_action_applied": False,
                      "action_source": "SHARED_FEASIBILITY"}},
              explanation_contains={"R1": ["금액 비교 결과가 승격 조건을 충족했지만 현재 행동은 바꾸지 않습니다."]}),
    _scenario("P", "TEST/SAMPLE input",
              lambda: _plain([_rec("R1", 20, "재고 이동", rank=1)], data_context={"is_sample": True},
                             seller_loss=_seller([_seller_decision("R1", note="TEST USER INPUT")])),
              {"R1": {"action_status": ac.FEASIBLE_PLAN, "is_executable": False, "seller_loss_claim": "MIN_LOSS_NONPRODUCTION_INPUT",
                      "seller_loss_min_loss_claim": False, "seller_loss_promotion_status": "BLOCKED"}},
              result_expect={"nonproduction_data": ["is_sample"]},
              explanation_contains={"R1": ["검증용(TEST/SAMPLE) 데이터로 계산된 계획이라 실제 운영 계획으로 쓰지 않습니다."]}),
    _scenario("Q", "operator_action NULL", lambda: _plain([_rec("R1", 50, "재고 이동", rank=1)]),
              {"R1": {"operator_action": None, "execution_state": ac.NOT_RECORDED, "is_executed": False,
                      "action_status": ac.FEASIBLE_PLAN}},
              forged=[("R1", {"is_executed": True}, ["H_EXECUTED_WITHOUT_OPERATOR_RECORD"]),
                      ("R1", {"action_status": ac.EXECUTED}, ["H_EXECUTED_WITHOUT_OPERATOR_RECORD"])],
              explanation_contains={"R1": ["판매자 실행 기록은 아직 없습니다."]}),
    _scenario("R", "operator EXECUTED (and rejected / non-production records)",
              lambda: {**_plain([_rec("R1", 50, "재고 이동", rank=1), _rec("R2", 50, "할인", source="S2", rank=2, target="T2"),
                                 _rec("R3", 50, "재고 이동", source="S3", rank=3, target="T3")]),
                       "operator_events": [_event("R1"), _event("R2", source_id="S2", target_id="T2", data_mode="TEST"),
                                           _event("R3", source_id="S3", target_id="T3", executed_at=None)]},
              {"R1": {"action_status": ac.EXECUTED, "action_source": "OPERATOR_CONFIRMED", "operator_action": "재고 이동",
                      "operator_action_code": "TRANSFER", "is_executed": True, "execution_state": ac.EXECUTED,
                      "decision_state": ac.SELECTED_PLAN, "conflict_codes": []},
               "R2": {"is_executed": False, "operator_action": None, "execution_state": ac.NOT_RECORDED},
               "R3": {"is_executed": False, "operator_action": None, "execution_state": ac.NOT_RECORDED}},
              result_expect={"rejected_operator_events.0.codes": ["EXECUTED_AT_REQUIRED"]},
              explanation_contains={"R1": ["판매자가 2026-07-31에 재고 이동을 실행했다고 기록했습니다."]}),
    _scenario("S", "긴급할인 / 1+1 / 폐기 / 비교 불가 / '이동 비추천' are not merged",
              lambda: _plain([_rec("R1", 10, "긴급 할인", rank=1, target="TA"), _rec("R2", 10, "1+1", rank=2, target="TB"),
                              _rec("R3", 10, "폐기", rank=3, target="TC"), _rec("R4", 10, "비교 불가", rank=4, target="TD"),
                              _rec("R5", 10, "이동 비추천", rank=5, target="TE")]),
              {"R1": {"legacy_action_code": "URGENT_DISCOUNT", "conflict_codes": ["A_TRANSFER_PLAN_VS_LEGACY_OTHER", "I_LEGACY_VS_CANONICAL_MISMATCH"]},
               "R2": {"legacy_action_code": "BUNDLE_PROMOTION", "conflict_codes": ["A_TRANSFER_PLAN_VS_LEGACY_OTHER", "I_LEGACY_VS_CANONICAL_MISMATCH"]},
               "R3": {"legacy_action_code": "DISPOSE", "conflict_codes": ["A_TRANSFER_PLAN_VS_LEGACY_OTHER", "I_LEGACY_VS_CANONICAL_MISMATCH"]},
               "R4": {"legacy_action_code": None, "legacy_action_mapping": "SENTINEL", "legacy_alignment": ac.UNMAPPABLE, "conflict_codes": []},
               "R5": {"legacy_action_code": None, "legacy_action_mapping": ac.UNMAPPABLE, "legacy_alignment": ac.UNMAPPABLE}}),
    _scenario("T", "same input twice and shuffled -> identical result", lambda: _plain(
              [_rec("R1", 30, "할인", rank=1, need=40), _rec("R2", 30, "보류", source="S2", rank=2, need=40, transfer_cost_basis="PER_UNIT")],
              legacy_top_ids=["R1", "R2"]),
              {"R1": {"allocated_qty": 30.0}, "R2": {"allocated_qty": 10.0, "selection_status": sf.PARTIALLY_SELECTED}}),
)


def _lookup(result: Mapping[str, Any], path: str) -> Any:
    head, _, rest = path.partition(".")
    if head == "product_summaries":
        return next((item for item in result["product_summaries"] if item["product_id"] == rest), None)
    value: Any = result[head]
    for part in rest.split(".") if rest else []:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def run_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = scenario["build"]()
    before = copy.deepcopy(kwargs["recommendations"])
    result = ac.build_action_consistency(**kwargs)
    failures: list[str] = []
    by_route = {record["route_id"]: record for record in result["records"]}
    for route, fields in scenario["expect"].items():
        record = by_route.get(route)
        if record is None:
            failures.append(f"{route}: missing")
            continue
        for key, value in fields.items():
            if record.get(key) != value:
                failures.append(f"{route}.{key}={record.get(key)!r} expected {value!r}")
    for path, expected in scenario["result_expect"].items():
        actual = _lookup(result, path)
        if isinstance(expected, Mapping):
            for key, value in expected.items():
                if (actual or {}).get(key) != value:
                    failures.append(f"{path}.{key}={(actual or {}).get(key)!r} expected {value!r}")
        elif actual != expected:
            failures.append(f"{path}={actual!r} expected {expected!r}")
    for route, phrases in scenario["explanation_contains"].items():
        text = " ".join(by_route[route]["explanation_ko"])
        failures.extend(f"{route}: missing text {phrase!r}" for phrase in phrases if phrase not in text)
    for route, phrases in scenario["explanation_excludes"].items():
        text = " ".join(by_route[route]["explanation_ko"])
        failures.extend(f"{route}: unexpected text {phrase!r}" for phrase in phrases if phrase in text)
    for route, patch, codes in scenario["forged"]:
        forged = {**copy.deepcopy(by_route[route]), **patch}
        found = ac.detect_record_conflicts(forged)
        failures.extend(f"{route}: forged {sorted(patch)} missed {code}" for code in codes if code not in found)
    if scenario["id"] == "J":
        summary = copy.deepcopy(_lookup(result, "product_summaries.P1"))
        summary["total_allocated_qty"] = 50.0
        if ac.detect_product_conflicts(summary, result["records"]) != ["J_PRODUCT_SUMMARY_DOUBLE_COUNT"]:
            failures.append("J: forged product total not detected")
    if scenario["id"] == "T":
        again = ac.build_action_consistency(**scenario["build"]())
        shuffled = scenario["build"]()
        random.Random(7).shuffle(shuffled["recommendations"])
        third = ac.build_action_consistency(**shuffled)
        key = lambda res: sorted(json.dumps(r, sort_keys=True, ensure_ascii=False, default=str) for r in res["records"])  # noqa: E731
        if json.dumps(again, sort_keys=True, default=str) != json.dumps(result, sort_keys=True, default=str):
            failures.append("T: same input gave a different result")
        if key(third) != key(result):
            failures.append("T: shuffled input gave different records")
    if kwargs["recommendations"] != before:
        failures.append("input recommendations were mutated")
    if result["invariant_failures"] != scenario["invariant_failures"]:
        failures.append(f"invariant failures {result['invariant_failures']} expected {scenario['invariant_failures']}")
    rows = [{"scenario": scenario["id"], "title": scenario["title"], "route_id": record["route_id"],
             "action_code": record["action_code"], "action_label_ko": record["action_label_ko"],
             "action_status": record["action_status"], "allocated_qty": record["allocated_qty"],
             "recommended_qty": record["recommended_qty"], "conflict_codes": "|".join(record["conflict_codes"]),
             "reason_codes": "|".join(record["reason_codes"]), "consistency_status": record["consistency_status"],
             "explanation_ko": " ".join(record["explanation_ko"])} for record in result["records"]]
    return {"scenario": scenario["id"], "title": scenario["title"], "status": "PASS" if not failures else "FAIL",
            "failures": failures, "rows": rows}


# ------------------------------------------------------------------------------------------------ Suhyup 31 days


def _dqn_model() -> tuple[str, Path] | None:
    from services import dqn_service

    found = None
    for path in sorted(dqn_service.OUTPUT_DIR.glob("dqn_result_suhyup_202607_31d_original_*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("seed") == 17 and data.get("episodes") == 300:
            found = data
    if not found:
        return None
    model = dqn_service.OUTPUT_DIR / Path(found["model_path"]).name
    return (found["data_signature"], model) if model.exists() else None


def _dqn_row(day: pd.DataFrame, date: str, model: tuple[str, Path], milp_service: float) -> dict[str, Any]:
    """Saved seed-17 model forward pass, selected exactly as suhyup_algorithm_revalidation does (no training)."""
    from services import dqn_service

    records = rv._records(day)
    inferred = dqn_service.infer_dqn_actions(records, model[0], model_path=str(model[1]))
    keyed = dqn_service._route_ids(records)
    working = day.copy()
    working["dqn_action_eval"] = [inferred.dqn_action_by_route.get(key) for key in keyed]
    working["dqn_confidence_eval"] = [inferred.dqn_confidence_by_route.get(key) for key in keyed]
    selected = rv.ordered_feasible_selection(working[working["dqn_action_eval"].isin(dqn_service.TRANSFER_ACTIONS)],
                                             ("dqn_confidence_eval", "recommended_qty", "move_cost", "route_id"),
                                             (False, False, True, True))
    return rv._strategy_row(date, "DQN", day, selected, milp_service, "saved_model_forward")


def _suhyup_seller(data_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    from services.seller_decision_validation import suhyup_seller_decisions

    by_day: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in suhyup_seller_decisions(data_root)["results"]:
        date, _, route = str(result["decision_id"]).partition(":")
        by_day[date][route] = {**result, "decision_id": route}
    return by_day


def _record_row(scope: str, date: str | None, record: Mapping[str, Any]) -> dict[str, Any]:
    return {"scope": scope, "date": date, **{key: record.get(key) for key in (
        "candidate_id", "route_id", "product_id", "product_name", "source_id", "source_name", "target_id", "target_name",
        "recommended_qty", "allocated_qty", "quantity_unit", "legacy_action", "legacy_action_code", "legacy_promotion_decision",
        "varo_final_rank", "varo_final_decision", "in_legacy_top5", "pareto_selected", "decision_state", "selection_status",
        "rejection_reason", "strict_selection_status", "benchmark_selection_status", "action_code", "action_label_ko",
        "action_status", "action_source", "selection_mode", "feasibility_status", "evidence_level", "is_executable",
        "is_executed", "operator_action", "execution_state", "cap_provenance", "seller_loss_comparison_status",
        "seller_loss_readiness", "seller_loss_action", "seller_loss_claim", "seller_loss_promotion_status",
        "consistency_status", "legacy_alignment")},
        "links": json.dumps(record.get("links"), ensure_ascii=False), "conflict_codes": "|".join(record["conflict_codes"]),
        "reason_codes": "|".join(record["reason_codes"]), "explanation_ko": " ".join(record["explanation_ko"])}


def _conflict_rows(scope: str, date: str | None, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in result["records"]:
        for code in record["conflict_codes"]:
            kind, level, description = ac.CONFLICTS[code]
            rows.append({"scope": scope, "date": date, "route_id": record["route_id"], "product_id": record["product_id"],
                         "product_name": record["product_name"], "source_id": record["source_id"], "target_id": record["target_id"],
                         "legacy_action": record["legacy_action"], "action_code": record["action_code"],
                         "action_status": record["action_status"], "recommended_qty": record["recommended_qty"],
                         "allocated_qty": record["allocated_qty"], "conflict_code": code, "conflict_type": kind,
                         "level": level, "description": description, "sentence_ko": ac._CONFLICT_TEXT[code](record),
                         "explanation_ko": " ".join(record["explanation_ko"])})
    for summary in result["product_summaries"]:
        for code in summary.get("conflict_codes") or []:
            rows.append({"scope": scope, "date": date, "product_id": summary["product_id"], "conflict_code": code,
                         "conflict_type": ac.CONFLICTS[code][0], "level": ac.CONFLICTS[code][1], "description": ac.CONFLICTS[code][2]})
    return rows


def run_suhyup(data_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    candidates = sfv.load_suhyup_candidates(data_root)
    reference = pd.read_csv(data_root / sfv.REFERENCE, dtype={"date": str})
    saved_t1 = pd.read_csv(data_root / sfv.OUTPUT_FOLDER / "shared_feasibility_by_day.csv", dtype={"date": str})
    model = _dqn_model()
    with _env(REAL_DATA_ROOT_ENV, str(data_root)):
        seller_by_day = _suhyup_seller(data_root)
        recompute = sfv.TariffRecompute(candidates)
        by_day, records_out, conflicts, regression, t1_regression, products = [], [], [], [], [], []
        determinism, invariant_failures = [], []
        for date in sorted(candidates["snapshot_date"].astype(str).unique()):
            day = sfv.ranked_day(candidates, date)
            records = rv._records(day)
            caps = sf.caps_from_columns(records, sfv.SUHYUP_CAP_SPEC)
            legacy = day.sort_values(["varo_final_rank", "route_id"], kind="mergesort").head(rv.MAX_DAILY_ROUTES)
            legacy_ids = legacy["route_id"].astype(str).tolist()

            def plans(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
                table = sf.caps_from_columns(rows, sfv.SUHYUP_CAP_SPEC)
                return {"status": "parallel_only", "modes": {mode: sf.select_shared_feasible(
                    rows, table, mode=mode, max_routes=rv.MAX_DAILY_ROUTES, partial_policy=sf.PARTIAL_IF_SAFE,
                    cost_recompute=recompute, legacy_ids=legacy_ids) for mode in (sf.STRICT_ACTUAL, sf.BENCHMARK_PROXY)}}

            shared = plans(records)
            decisions = [seller_by_day[date][str(record["route_id"])] for record in records if str(record["route_id"]) in seller_by_day[date]]
            seller = {"status": "parallel_only", "decisions": decisions,
                      "shadow_decisions": [gate.shadow_decision(d, context={"decision_date": date}) for d in decisions]}
            pareto_ids = day.loc[day["pareto_selected"].fillna(False).astype(bool), "route_id"].astype(str).tolist()
            before = copy.deepcopy(records)
            kwargs = {"shared_feasibility": shared, "seller_loss": seller, "legacy_top_ids": legacy_ids,
                      "pareto_selected_ids": pareto_ids, "legacy_rule_connected": True}
            result = ac.build_action_consistency(records, **kwargs)
            shuffled = list(copy.deepcopy(records))
            random.Random(date).shuffle(shuffled)
            again = ac.build_action_consistency(shuffled, **{**kwargs, "shared_feasibility": plans(shuffled)})
            same = sorted(json.dumps(r, sort_keys=True, default=str) for r in again["records"]) == sorted(
                json.dumps(r, sort_keys=True, default=str) for r in result["records"])
            determinism.append(same and records == before)
            invariant_failures.extend(f"{date}:{item}" for item in result["invariant_failures"])
            bench, strict = shared["modes"][sf.BENCHMARK_PROXY], shared["modes"][sf.STRICT_ACTUAL]
            saved = saved_t1[saved_t1["date"] == date].iloc[0]
            legacy_check = sf.validate_plan(rv._records(legacy), caps, mode=sf.BENCHMARK_PROXY, max_routes=rv.MAX_DAILY_ROUTES)
            t1_regression.append({
                "date": date, "route_ids_equal": "|".join(bench["selected_ids"]) == str(saved["sf_partial_route_ids"]),
                "qty_equal": abs(float(bench["total_allocated_qty"]) - float(saved["sf_partial_service_qty"])) <= 1e-6,
                "cost_equal": abs(float(bench["total_move_cost"] or 0) - float(saved["sf_partial_cost"])) <= 1e-6,
                "strict_status_equal": strict["plan_status"] == saved["sf_strict_plan_status"],
                "violations": int(bench["validation"]["violation_count"]),
                "legacy_violations_equal": int(legacy_check["violation_count"]) == int(saved["legacy_violation_count"]),
                "qty": float(bench["total_allocated_qty"]), "cost": float(bench["total_move_cost"] or 0),
                "legacy_violation_count": int(legacy_check["violation_count"])})
            refs = sfv._reference_rows(day, date)
            if model is not None:
                refs["DQN"] = _dqn_row(day, date, model, float(refs["MILP"]["service_qty"]))
            for strategy, row in refs.items():
                saved_ref = reference[(reference["date"] == date) & (reference["strategy"] == strategy)].iloc[0]
                regression.append({
                    "date": date, "strategy": strategy,
                    "equal": abs(float(row["service_qty"]) - float(saved_ref["service_qty"])) <= 1e-6
                    and abs(float(row["total_cost"]) - float(saved_ref["total_cost"])) <= 1e-6
                    and str(row["route_ids"]) == str(saved_ref["route_ids"])
                    and int(row["feasibility_violations"]) == int(saved_ref["feasibility_violations"])})
            recs = result["records"]
            selected = [r for r in recs if r["decision_state"] == ac.SELECTED_PLAN]
            legacy_codes = Counter(r["legacy_action_code"] or "UNMAPPED" for r in recs)
            conflict_types = Counter(ac.CONFLICTS[c][0] for r in recs for c in r["conflict_codes"])
            selected_ids = [r["route_id"] for r in selected]
            by_day.append({
                "date": date, "candidate_count": len(recs),
                "legacy_discount": legacy_codes.get("DISCOUNT_SALE", 0), "legacy_hold": legacy_codes.get("HOLD", 0),
                "legacy_transfer": legacy_codes.get("TRANSFER", 0),
                "legacy_other": sum(v for k, v in legacy_codes.items() if k not in ("DISCOUNT_SALE", "HOLD", "TRANSFER")),
                "varo_final_top5_ids": "|".join(legacy_ids), "varo_final_top5_count": len(legacy_ids),
                "varo_final_top5_qty": float(legacy["recommended_qty"].sum()),
                "t1_mode": result["primary_selection_mode"], "t1_selected_ids": "|".join(selected_ids),
                "t1_selected_count": len(selected), "t1_allocated_qty": result["total_allocated_qty"],
                "t1_cost": bench["total_move_cost"], "t1_strict_plan_status": strict["plan_status"],
                "kept_from_top5": "|".join(i for i in legacy_ids if i in selected_ids),
                "dropped_from_top5": "|".join(i for i in legacy_ids if i not in selected_ids),
                "added_by_t1": "|".join(i for i in selected_ids if i not in legacy_ids),
                "route_mismatch_count": sum(i not in selected_ids for i in legacy_ids),
                "quantity_mismatch_count": sum(abs(float(r["allocated_qty"]) - float(r["recommended_qty"] or 0)) > 1e-9 for r in selected),
                "action_mismatch_count": sum(r["legacy_alignment"] == ac.MISMATCH for r in recs),
                **{f"conflict_{kind}": conflict_types.get(kind, 0) for kind in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "SL")},
                "review_required": sum(r["consistency_status"] == ac.REVIEW_REQUIRED for r in recs),
                "action_conflict": sum(r["consistency_status"] == ac.ACTION_CONFLICT for r in recs),
                "consistent": sum(r["consistency_status"] == ac.CONSISTENT for r in recs),
                "proxy_executable_claims": sum("F_PROXY_PLAN_CLAIMED_EXECUTABLE" in r["conflict_codes"] for r in recs),
                "unmappable_legacy_actions": sum(r["legacy_action_code"] is None for r in recs),
                "legacy_outside_seller_loss_scope": sum(r["legacy_action_code"] not in STRATEGIES for r in recs),
                "seller_loss_unavailable": sum(r["seller_loss_comparison_status"] == "COMPARISON_UNAVAILABLE" for r in recs),
                "multi_route_products": "|".join(result["multi_route_products"]),
                "invariant_failures": "|".join(result["invariant_failures"]), "deterministic": determinism[-1],
                "legacy_top5_shared_cap_violations": int(legacy_check["violation_count"]),
            })
            records_out.extend(_record_row("SUHYUP_31D", date, r) for r in recs)
            conflicts.extend(_conflict_rows("SUHYUP_31D", date, result))
            products.extend({"date": date, **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                                              for k, v in s.items()}} for s in result["product_summaries"])
    return {"by_day": pd.DataFrame(by_day), "records": pd.DataFrame(records_out), "conflicts": pd.DataFrame(conflicts),
            "products": pd.DataFrame(products), "regression": pd.DataFrame(regression),
            "t1_regression": pd.DataFrame(t1_regression), "determinism": determinism,
            "invariant_failures": invariant_failures, "dqn_model_available": model is not None,
            "wall_seconds": round(time.perf_counter() - started, 3)}


# ------------------------------------------------------------------------------------------------ workbooks and E2E


def run_workbooks(repo_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    from services.analysis_pipeline import build_v2_state
    from services.data_loader import load_excel_data

    rows, conflicts, records = [], [], []
    for folder in ("data", "samples", "Varo_DQN_training_samples_10pack"):
        for path in sorted((repo_root / folder).glob("*.xlsx")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state = build_v2_state(load_excel_data(path), detail_level="core")
            name = str(path.relative_to(repo_root)).replace("\\", "/")
            result = state["pipeline_result"].get("action_consistency") or {}
            recs = result.get("records") or []
            labels: dict[tuple[str, str], set[str]] = defaultdict(set)
            for record in recs:
                labels[(record["source_id"], record["product_id"])].add(record["legacy_action"] or "")
            production_same = all(r["production_action"] == rec.get("varo_action")
                                  for r, rec in zip(recs, state["recommendations"]))
            rows.append({
                "workbook": name, "status": result.get("status"), "primary_mode": result.get("primary_selection_mode"),
                "records": len(recs), "selected_plans": result.get("selected_plan_count"),
                "allocated_qty": result.get("total_allocated_qty"),
                "legacy_action_counts": json.dumps(dict(Counter(r["legacy_action"] for r in recs)), ensure_ascii=False),
                "action_status_counts": json.dumps(result.get("action_status_counts"), ensure_ascii=False),
                "consistency_counts": json.dumps(result.get("consistency_counts"), ensure_ascii=False),
                "alignment_counts": json.dumps(result.get("alignment_counts"), ensure_ascii=False),
                "conflict_counts": json.dumps(result.get("conflict_counts"), ensure_ascii=False),
                "legacy_transfer_labels": sum(r["legacy_action_code"] == "TRANSFER" for r in recs),
                "source_product_groups": len(labels), "groups_with_multiple_legacy_labels": sum(len(v) > 1 for v in labels.values()),
                "invariant_failures": "|".join(result.get("invariant_failures") or []),
                "production_action_unchanged": production_same and result.get("production_action_applied") is False,
            })
            conflicts.extend(_conflict_rows(f"WORKBOOK:{name}", None, result))
            records.extend(_record_row(f"WORKBOOK:{name}", None, r) for r in recs)
    return pd.DataFrame(rows), conflicts, records


def run_production_e2e(data_root: Path) -> dict[str, Any]:
    """The Suhyup 2026-07-31 production rows through build_v2_state, with and without the new field."""
    from services import analysis_pipeline
    from services.seller_loss_input_validation import _run, load_suhyup_upload

    upload = load_suhyup_upload(data_root)
    state = _run(copy.deepcopy(upload), data_root)
    with mock.patch.object(analysis_pipeline, "build_action_consistency", side_effect=RuntimeError("isolation probe")):
        broken = _run(copy.deepcopy(upload), data_root)
    unchanged = {key: state["pipeline_result"][key] == broken["pipeline_result"][key]
                 for key in ("top5", "summary", "status", "connected_algorithms", "shared_feasibility_selection", "seller_loss_analysis")}
    unchanged["recommendations"] = state["recommendations"] == broken["recommendations"]
    result = state["pipeline_result"]["action_consistency"]
    saved = pd.read_csv(data_root / sfv.CANDIDATES, dtype=str)
    saved = saved[saved["snapshot_date"] == "2026-07-31"].set_index("route_id")["varo_action"].to_dict()
    current = {item["route_id"]: item["varo_action"] for item in state["recommendations"]}
    return {
        "records": len(result["records"]), "status": result["status"], "primary_selection_mode": result["primary_selection_mode"],
        "primary_selection_basis": result["primary_selection_basis"],
        "selected_ids": [r["route_id"] for r in result["records"] if r["decision_state"] == ac.SELECTED_PLAN],
        "total_allocated_qty": result["total_allocated_qty"], "top5_ids": [i["route_id"] for i in state["pipeline_result"]["top5"]],
        "action_status_counts": result["action_status_counts"], "consistency_counts": result["consistency_counts"],
        "conflict_counts": result["conflict_counts"], "invariant_failures": result["invariant_failures"],
        "isolated_failure_status": broken["pipeline_result"]["action_consistency"].get("status"),
        "production_fields_unchanged_without_field": unchanged,
        "saved_varo_action_equals_current_pipeline": sum(saved.get(k) == v for k, v in current.items()),
        "saved_varo_action_compared": len(current),
        "conflicts": _conflict_rows("E2E_20260731", "2026-07-31", result),
        "records_rows": [_record_row("E2E_20260731", "2026-07-31", r) for r in result["records"]],
    }


# ------------------------------------------------------------------------------------------------ report


def _examples(records: pd.DataFrame, minimum: int = 12) -> pd.DataFrame:
    """Real Suhyup conflicts: every ACTION_CONFLICT case, then round-robin over the other conflict codes until >= minimum."""
    with_conflict = records[records["conflict_codes"] != ""].copy()
    if with_conflict.empty:
        return with_conflict
    with_conflict["primary_code"] = with_conflict["conflict_codes"].str.split("|").str[0]
    hard = with_conflict[with_conflict["consistency_status"] == ac.ACTION_CONFLICT].sort_values(["date", "route_id"])
    rest = with_conflict.drop(hard.index)
    groups = {code: frame.sort_values(["date", "route_id"]) for code, frame in rest.groupby("primary_code")}
    picked: list[pd.Series] = [row for _, row in hard.iterrows()]
    position = 0
    while len(picked) < max(minimum, len(hard) + len(groups)) and any(position < len(frame) for frame in groups.values()):
        for code in sorted(groups):
            if position < len(groups[code]):
                picked.append(groups[code].iloc[position])
        position += 1
    out = pd.DataFrame(picked)
    columns = {"date": "날짜", "product_name": "상품", "source_name": "source", "target_name": "target", "route_id": "route",
               "recommended_qty": "추천_수량", "allocated_qty": "T1_배정_수량", "legacy_action": "legacy_action",
               "action_label_ko": "T1_action", "action_status": "T1_status", "seller_loss_claim": "seller_loss",
               "conflict_codes": "충돌_원인_코드", "explanation_ko": "한글_설명"}
    out = out[[*columns, "product_id", "source_id", "target_id", "consistency_status", "rejection_reason"]].rename(columns=columns)
    out.insert(12, "충돌_원인", out["충돌_원인_코드"].map(lambda codes: "; ".join(ac.CONFLICTS[c][2] for c in codes.split("|"))))
    return out.reset_index(drop=True)


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        # rstrip only: porcelain lines start with a status column that may be a space
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout.rstrip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check(check_id: str, description: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"check_id": check_id, "description": description, "status": "PASS" if ok else "FAIL", "detail": detail}


PROTECTED_PATHS = ("services/shared_feasibility_selection.py", "services/seller_loss_engine.py", "services/seller_loss_inputs.py",
                   "services/seller_loss_promotion_gate.py", "services/seller_shadow_ledger.py", "services/seller_shadow_outcomes.py",
                   "services/vhs_score_engine.py", "services/demand_forecast_router.py", "services/demand_forecast_v2.py",
                   "services/recommendation_adapter.py", "services/legacy_adapters", "services/pareto_service.py",
                   "services/dqn_service.py", "services/suhyup_algorithm_revalidation.py", "pages", "components", "app_v2.py")


def run_validation(data_root: Path, output_dir: Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    data_root = Path(data_root)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    output_dir = Path(output_dir or data_root / OUTPUT_FOLDER)
    scenarios = [run_scenario(item) for item in CONTROLLED_SCENARIOS]
    suhyup = run_suhyup(data_root)
    workbooks, wb_conflicts, wb_records = run_workbooks(repo_root)
    e2e = run_production_e2e(data_root)
    by_day, records, regression, t1 = suhyup["by_day"], suhyup["records"], suhyup["regression"], suhyup["t1_regression"]
    conflicts = pd.DataFrame([*suhyup["conflicts"].to_dict("records"), *wb_conflicts, *e2e.pop("conflicts")])
    candidates = pd.DataFrame([*records.to_dict("records"), *e2e.pop("records_rows"), *wb_records])
    examples = _examples(records)
    selected = records[records["decision_state"] == ac.SELECTED_PLAN]
    alignment = records["legacy_alignment"].value_counts().to_dict()
    links_ok = records["links"].map(lambda text: all(value == "OK" for value in json.loads(text).values()))
    changed = (_git(repo_root, "status", "--porcelain") or "").splitlines()
    protected_changed = [line for line in changed if any(line[3:].startswith(path) for path in PROTECTED_PATHS)]
    t1_ok = t1[["route_ids_equal", "qty_equal", "cost_equal", "strict_status_equal", "legacy_violations_equal"]].all(axis=1)
    expected_regression_rows = 186 if suhyup["dqn_model_available"] else 155
    checks = [
        _check("C1", "controlled scenarios A-T all pass", all(item["status"] == "PASS" for item in scenarios),
               {item["scenario"]: item["status"] for item in scenarios}),
        _check("S1", "Suhyup 31 days, 620 candidates, every candidate linked to its T1 row and Seller Loss decision",
               len(by_day) == 31 and len(records) == 620 and bool(links_ok.all()), f"{int(links_ok.sum())}/{len(records)}"),
        _check("S2", "T1 BENCHMARK_PROXY plan reproduced from the saved T1 bundle on 31/31 days (ids, qty, cost, strict status, legacy violations)",
               bool(t1_ok.all()) and len(t1) == 31, int(t1_ok.sum())),
        _check("S3", "T1 BENCHMARK_PROXY totals 7,750 / 13,629,146 with 0 violations; STRICT_ACTUAL INSUFFICIENT_CAP_DATA on 31 days",
               abs(t1["qty"].sum() - 7750) <= 1e-6 and abs(t1["cost"].sum() - 13629146) <= 1e-6 and int(t1["violations"].sum()) == 0
               and bool((by_day["t1_strict_plan_status"] == sf.INSUFFICIENT_CAP_DATA).all()),
               {"qty": float(t1["qty"].sum()), "cost": float(t1["cost"].sum()), "violations": int(t1["violations"].sum())}),
        _check("S4", "legacy Top-5 shared-cap violations still on the 5 audited days",
               sorted(by_day.loc[by_day["legacy_top5_shared_cap_violations"] > 0, "date"].tolist())
               == ["2026-07-05", "2026-07-15", "2026-07-16", "2026-07-17", "2026-07-19"],
               by_day.loc[by_day["legacy_top5_shared_cap_violations"] > 0, "date"].tolist()),
        _check("S5", f"Suhyup reference rows recomputed with current code are identical ({expected_regression_rows} rows)",
               bool(regression["equal"].all()) and len(regression) == expected_regression_rows,
               f"{int(regression['equal'].sum())}/{len(regression)}; DQN model available={suhyup['dqn_model_available']}"),
        _check("S6", "canonical invariants hold on every Suhyup day", not suhyup["invariant_failures"], suhyup["invariant_failures"][:10]),
        _check("S7", "every selected plan shows allocated_qty > 0 and the plan route; no plan on unselected candidates",
               bool((selected["allocated_qty"] > 0).all()) and bool((selected["action_code"] == "TRANSFER").all())
               and bool((records.loc[records["decision_state"] != ac.SELECTED_PLAN, "action_code"].isna()).all())),
        _check("S8", "PROXY plans are never executable; no EXECUTED without operator record",
               int(by_day["proxy_executable_claims"].sum()) == 0 and not bool(records["is_executable"].fillna(False).astype(bool).any())
               and not bool(records["is_executed"].fillna(False).astype(bool).any())),
        _check("S9", "deterministic and order-invariant on 31/31 days; inputs not mutated", all(suhyup["determinism"])),
        _check("S10", "Seller Loss never backs an action and makes no min-loss claim on Suhyup (all COMPARISON_UNAVAILABLE)",
               int(by_day["seller_loss_unavailable"].sum()) == 620 and not (records["seller_loss_claim"] != "NO_COMPARISON").any()),
        _check("W1", "all 16 repository workbooks carry the field with 0 invariant failures and production actions unchanged",
               len(workbooks) == 16 and bool((workbooks["status"] == "parallel_only").all())
               and bool((workbooks["invariant_failures"] == "").all()) and bool(workbooks["production_action_unchanged"].all())),
        _check("W2", "legacy label is constant per (source, product) in every workbook (inventory-state advice, not a route decision)",
               int(workbooks["groups_with_multiple_legacy_labels"].sum()) == 0,
               int(workbooks["groups_with_multiple_legacy_labels"].sum())),
        _check("E1", "2026-07-31 production E2E: field present, invariants hold, production fields identical when the field fails",
               e2e["status"] == "parallel_only" and not e2e["invariant_failures"] and all(e2e["production_fields_unchanged_without_field"].values())
               and e2e["isolated_failure_status"] == "error", e2e["production_fields_unchanged_without_field"]),
        _check("E2", "saved Suhyup varo_action equals the current production pipeline on 2026-07-31",
               e2e["saved_varo_action_equals_current_pipeline"] == e2e["saved_varo_action_compared"] == 20,
               f"{e2e['saved_varo_action_equals_current_pipeline']}/{e2e['saved_varo_action_compared']}"),
        _check("G1", "no protected production module changed (selector, Seller Loss, ranking, forecast, adapters, UI)",
               not protected_changed, protected_changed),
    ]
    failed = [item["check_id"] for item in checks if item["status"] == "FAIL"]
    observed = set(records["legacy_action"].dropna()) | {label for text in workbooks["legacy_action_counts"]
                                                         for label in json.loads(text) if label}
    unmapped = sorted(label for label in observed if ac.map_action_label(label)[1] == ac.UNMAPPABLE)
    checks.append(_check("A1", "every action label observed in Suhyup and the workbooks maps to a canonical code or a sentinel",
                         not unmapped, {"observed": sorted(observed), "unmapped": unmapped}))
    failed = [item["check_id"] for item in checks if item["status"] == "FAIL"]
    readiness = {
        "A_contract_complete": "A1" not in failed, "B_linked_to_real_candidates": "S1" not in failed,
        "C_conflict_detection": "C1" not in failed, "D_korean_labels": bool(records["explanation_ko"].str.len().gt(0).all()),
        "E_parallel_validation": not {"S2", "S3", "S5", "W1", "E1"} & set(failed),
        "F_production_action_unchanged": not {"W1", "E1", "G1"} & set(failed),
    }
    verdict = "NOT_READY" if not all(readiness.values()) else "READY_WITH_LIMITATIONS"
    summary_rows = [
        ("SUHYUP_31D", "candidates", len(records)), ("SUHYUP_31D", "days", len(by_day)),
        *[("SUHYUP_31D", f"legacy_action[{k}]", int(v)) for k, v in sorted(records["legacy_action"].value_counts().items())],
        ("SUHYUP_31D", "varo_final_top5_slice_rows", int(records["in_legacy_top5"].sum())),
        ("SUHYUP_31D", "varo_final_rank1_final_recommendation", int((records["varo_final_decision"] == "최종 추천").sum())),
        ("SUHYUP_31D", "t1_selected_plans", int(len(selected))),
        ("SUHYUP_31D", "t1_allocated_qty", float(selected["allocated_qty"].sum())),
        ("SUHYUP_31D", "t1_partial_allocations", int((selected["selection_status"] == sf.PARTIALLY_SELECTED).sum())),
        ("SUHYUP_31D", "route_mismatch_top5_vs_t1", int(by_day["route_mismatch_count"].sum())),
        ("SUHYUP_31D", "quantity_mismatch_selected", int(by_day["quantity_mismatch_count"].sum())),
        ("SUHYUP_31D", "action_mismatch (legacy vs canonical, MISMATCH)", int(by_day["action_mismatch_count"].sum())),
        ("SUHYUP_31D", "proxy_executable_claims", int(by_day["proxy_executable_claims"].sum())),
        ("SUHYUP_31D", "unmappable_legacy_actions (canonical code)", int(by_day["unmappable_legacy_actions"].sum())),
        ("SUHYUP_31D", "legacy_actions_outside_seller_loss_scope", int(by_day["legacy_outside_seller_loss_scope"].sum())),
        *[("SUHYUP_31D", f"alignment[{k}]", int(v)) for k, v in sorted(alignment.items())],
        *[("SUHYUP_31D", f"consistency[{k}]", int(v)) for k, v in sorted(records["consistency_status"].value_counts().items())],
        *[("SUHYUP_31D", f"conflict[{k}]", int(v)) for k, v in sorted(suhyup["conflicts"]["conflict_code"].value_counts().items())],
        ("SUHYUP_31D", "candidates_with_any_conflict", int((records["conflict_codes"] != "").sum())),
        ("SUHYUP_31D", "seller_loss_comparison_unavailable", int(by_day["seller_loss_unavailable"].sum())),
        ("WORKBOOKS_16", "records", int(workbooks["records"].sum())),
        ("WORKBOOKS_16", "selected_plans", int(workbooks["selected_plans"].sum())),
        ("WORKBOOKS_16", "legacy_transfer_labels", int(workbooks["legacy_transfer_labels"].sum())),
        ("E2E_20260731", "selected_plans", len(e2e["selected_ids"])), ("E2E_20260731", "allocated_qty", e2e["total_allocated_qty"]),
    ]
    validation = {
        "contract_version": ac.CONTRACT_VERSION, "contract_signature": ac.contract_document()["signature"],
        "git_head": _git(repo_root, "rev-parse", "HEAD"), "git_status_porcelain": changed,
        "data_root": str(data_root), "checks": checks, "failed_checks": failed,
        "readiness": {"criteria": readiness, "verdict": verdict,
                      "meaning": "contract, linking, detection, labels and parallel validation are complete; production action "
                                 "replacement is a separate approved step"},
        "controlled_scenarios": [{k: v for k, v in item.items() if k != "rows"} for item in scenarios],
        "controlled_scenario_rows": [row for item in scenarios for row in item["rows"]],
        "suhyup": {
            "inputs": {"candidates": sfv.CANDIDATES, "inventory": sfv.INVENTORY, "reference": sfv.REFERENCE,
                       "t1_bundle": f"{sfv.OUTPUT_FOLDER}/shared_feasibility_by_day.csv"},
            "legacy_action_basis": "saved production varo_action (legacy rule); equal to the current pipeline on 2026-07-31 (E2)",
            "plan_basis": "SF_BENCHMARK_PARTIAL backs canonical actions (STRICT_ACTUAL INSUFFICIENT_CAP_DATA on 31 days)",
            "alignment_counts": {k: int(v) for k, v in alignment.items()},
            "t1_regression_days_equal": int(t1_ok.sum()), "reference_rows_equal": f"{int(regression['equal'].sum())}/{len(regression)}",
            "reference_rows_by_strategy": regression.groupby("strategy")["equal"].sum().astype(int).to_dict(),
            "wall_seconds": suhyup["wall_seconds"],
        },
        "workbooks": workbooks.to_dict("records"),
        "production_e2e_20260731": e2e,
        "field_inventory": list(FIELD_INVENTORY),
        "limitations": [
            "STRICT_ACTUAL has no validated caps on Suhyup (31/31 INSUFFICIENT_CAP_DATA): every Suhyup plan is a PROXY_PLAN",
            "Suhyup legacy labels for 07-01..07-30 are the saved production labels; only 07-31 has a production upload to recompute them",
            "no real operator outcome exists: execution_state is NOT_RECORDED everywhere outside the controlled scenarios",
            "Seller Loss is COMPARISON_UNAVAILABLE on all 620 Suhyup candidates (no price / shelf life; demand is a proxy)",
            "Suhyup quantity unit is UNKNOWN: labels say '(단위 미확인)' instead of inventing '개'",
            "the UI still reads varo_action / recommended_qty / Top-5 slice; the adapter contract is design only",
            "the shadow ledger schema is unchanged; canonical records are not persisted (proposal: a separate table keyed by decision_id/version)",
        ],
        "production_action_applied": False,
    }
    contract = {**ac.contract_document(), "field_inventory": list(FIELD_INVENTORY), "existing_mappings": EXISTING_MAPPINGS,
                "ui_adapter_contract": UI_ADAPTER_CONTRACT}
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows, columns=["scope", "metric", "value"]).to_csv(output_dir / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")
    by_day.to_csv(output_dir / OUTPUT_FILES[1], index=False, encoding="utf-8-sig")
    conflicts.to_csv(output_dir / OUTPUT_FILES[2], index=False, encoding="utf-8-sig")
    examples.to_csv(output_dir / OUTPUT_FILES[3], index=False, encoding="utf-8-sig")
    (output_dir / OUTPUT_FILES[4]).write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_dir / OUTPUT_FILES[5]).write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    candidates.to_csv(output_dir / OUTPUT_FILES[6], index=False, encoding="utf-8-sig")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.output_dir)
    print(json.dumps({"failed_checks": result["failed_checks"], "checks": result["checks"],
                      "readiness": result["readiness"]}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
