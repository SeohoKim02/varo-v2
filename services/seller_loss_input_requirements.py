"""Seller Loss input requirement planner: ask the seller only for the inputs that are still needed.

    production row (inventory, real transport, forecast)  + explicit seller_loss_inputs
    -> merge (services.seller_loss_inputs) -> evaluate (services.seller_loss_engine) -> THIS PLAN
    -> the seller enters the listed minimum -> the same merge and evaluation run again

The planner asks for a field only in two cases:
    * entering it removes a blocking reason of a strategy in the requested comparison scope, or
    * the strategies are already comparable and the robust ranking depends on it.
Rules for every request:
    * A value that is already available is never requested, and a real value is never asked again.
    * A proxy is listed as a proxy to replace.
    * A strategy blocked by upload data is not asked about, because seller inputs cannot unblock it.
The plan never changes the engine result, the merged inputs or the production action. It never suggests or defaults a
number: a request carries a field, a key, a label and a reason.

Every rule is deterministic and explainable (no model call):
    * Strategy requirements are services.seller_loss_inputs.STRATEGY_REQUIRED, the lists the evidence layer uses.
      `trace_engine_requirements` checks them against the engine execution path by removing one field at a time on
      synthetic structural probes.
    * Robustness inputs are unknown rates whose coefficient differs inside a strategy pair that neither side wins over
      the whole admissible range (engine `_margin`). Once all of them are known, a robust winner always exists.
    * Order: tier first (unblock strategies > make the recommendation robust > optional), then the number of strategies
      whose blocking reason the input removes, then the engine's own check order.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import fields as dataclass_fields, replace
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from services import seller_loss_inputs as sli
from services.seller_loss_engine import (
    DISCOUNT_SALE, MONEY_FIELDS, NORMAL_SALE, NOT_ROBUST as ENGINE_NOT_ROBUST, PER_UNIT_MONEY_FIELDS, QUANTITY_FIELDS,
    REASON_MESSAGES, STATUS_UNAVAILABLE, STRATEGIES, STRATEGY_LABELS, TOL, TRANSFER, TRANSFER_ONLY_FIELDS,
    WORKBOOK_DATASET, InputField, SellerDecisionInput, _consistency_blockers, _josa, _margin, _reason_text,
    evaluate_seller_decision, known,
)

PLANNER_VERSION = "seller-loss-input-requirements-1.0.0"
COMPARISON_SCOPE_KEY = "seller_loss_comparison_scope"

ALL_THREE, TRANSFER_VS_NORMAL, NORMAL_VS_DISCOUNT, TRANSFER_VS_DISCOUNT = (
    "ALL_THREE", "TRANSFER_VS_NORMAL", "NORMAL_VS_DISCOUNT", "TRANSFER_VS_DISCOUNT")
COMPARISON_SCOPES: dict[str, tuple[str, ...]] = {
    ALL_THREE: (TRANSFER, NORMAL_SALE, DISCOUNT_SALE),
    TRANSFER_VS_NORMAL: (TRANSFER, NORMAL_SALE),
    NORMAL_VS_DISCOUNT: (NORMAL_SALE, DISCOUNT_SALE),
    TRANSFER_VS_DISCOUNT: (TRANSFER, DISCOUNT_SALE),
}
# The engine always compares every strategy with complete inputs; the scope only decides what the seller is asked for.
DEFAULT_COMPARISON_SCOPE = ALL_THREE

REQUIRED, CONDITIONAL, OPTIONAL, DERIVED, NOT_USED = (
    "REQUIRED", "CONDITIONALLY_REQUIRED", "OPTIONAL", "DERIVED", "NOT_USED")
AVAILABLE, MISSING, PROXY, CONFLICT = "AVAILABLE", "MISSING", "PROXY", "CONFLICT"

# Request kinds (what the seller has to do) and tiers (why).
ENTER_MISSING, REPLACE_PROXY, RESOLVE_CONFLICT = "ENTER_MISSING", "REPLACE_PROXY", "RESOLVE_CONFLICT"
CORRECT_CURRENCY, CORRECT_UNIT, DECLARE_UNIT, CORRECT_VALUE = "CORRECT_CURRENCY", "CORRECT_UNIT", "DECLARE_UNIT", "CORRECT_VALUE"
RESOLVE_UNKNOWN, REPLACE_SCENARIO_VALUE = "RESOLVE_UNKNOWN_FOR_ROBUST_RANKING", "REPLACE_SCENARIO_VALUE"
IMPROVE_PRECISION, PENDING_EVALUATION = "IMPROVE_PRECISION", "PENDING_EVALUATION"
TIERS = {1: "UNBLOCK_STRATEGIES", 2: "MAKE_RECOMMENDATION_ROBUST", 3: "OPTIONAL"}
SELLER, UPLOAD_DATA = "SELLER_CORRECTION", "UPLOAD_DATA"

# Strategy readiness as seen by the planner.
READY, NEEDS_SELLER_INPUT, BLOCKED_BY_DATA, BLOCKED_BY_VALUES = "READY", "NEEDS_SELLER_INPUT", "BLOCKED_BY_DATA", "BLOCKED_BY_VALUES"
SCOPE_COMPLETE, SCOPE_PARTIAL, SCOPE_UNAVAILABLE = "SCOPE_COMPLETE", "SCOPE_PARTIAL", "SCOPE_UNAVAILABLE"
PLANNING_STATUSES = {
    "READY_RECOMMENDABLE": "every strategy of the scope is comparable and the result is RECOMMENDABLE: stop asking",
    "READY_RECOMMENDABLE_SCOPE_LIMITED": "RECOMMENDABLE; the rest of the scope is blocked by upload data or values, which seller inputs cannot fix: stop asking",
    "RECOMMENDABLE_PARTIAL_SCOPE": "RECOMMENDABLE on fewer strategies than the scope; the listed inputs complete the requested scope",
    "NEEDS_INPUT": "strategies of the scope are blocked by missing/proxy/conflicting values the seller can enter",
    "NEEDS_INPUT_FOR_ROBUST_RECOMMENDATION": "the scope is comparable but the ranking depends on unknown values the seller can enter",
    "NOT_ROBUST_DATA_DEPENDENT": "the ranking depends on an unknown that only upload data can resolve",
    "BLOCKED_BY_DATA": "fewer than two strategies of the scope can become comparable through seller inputs (upload data missing)",
    "BLOCKED_BY_VALUES": "inputs are complete but the current values make the scope incomparable (e.g. nothing can be moved)",
    "SCENARIO_ONLY": "SCENARIO run: what-if results are never recommendations",
}

INPUT_FIELD_NAMES = tuple(f.name for f in dataclass_fields(SellerDecisionInput) if f.type in ("InputField", InputField))
# The order in which evaluate_seller_decision examines its inputs. It is the explainable tie-break of the input ranking:
# unit/currency consistency first, then quantity/stock/demand/shelf life/price, the rates, DISCOUNT, then TRANSFER.
ENGINE_CHECK_ORDER = (
    "quantity_unit", "decision_qty", "source_current_stock", "source_daily_demand", "remaining_shelf_life_days",
    "source_normal_price", "source_holding_cost_per_unit_day", "source_disposal_cost_per_unit", "salvage_value_per_unit",
    "target_holding_cost_per_unit_day", "target_disposal_cost_per_unit", "discount_rate", "promotion_uplift",
    "target_store_id", "target_current_stock", "target_daily_demand", "target_normal_price", "transfer_cost",
    "transit_time_days", "transfer_cost_qty", "source_surplus_cap", "target_need_cap", "route_capacity_qty", "unit_cost",
)
ORDER_INDEX = {name: index for index, name in enumerate(ENGINE_CHECK_ORDER)}
# Fields the seller can state in seller_loss_inputs (everything else comes from the upload or is derived by production).
SELLER_ENTERABLE = frozenset(sli.ENGINE_TARGETS) | {"quantity_unit"}
# Computed by production from the recommendation / transition engine; never asked from the seller.
DERIVED_FIELDS = frozenset({"decision_qty", "transfer_cost_qty", "target_store_id", "source_surplus_cap",
                            "target_need_cap", "route_capacity_qty"})
PRODUCTION_SOURCES = {
    "decision_qty": "Varo Final 추천 실행 수량(recommendation.recommended_qty)",
    "source_current_stock": "업로드 inventory.stock_qty", "target_current_stock": "업로드 inventory.stock_qty",
    "source_daily_demand": "수요 예측 라우터 demand_forecast_daily(판매량 근거가 대리지표면 PROXY)",
    "target_daily_demand": "수요 예측 라우터 demand_forecast_daily(판매량 근거가 대리지표면 PROXY)",
    "remaining_shelf_life_days": "업로드 inventory.days_to_expiry",
    "source_normal_price": "업로드 inventory/products.unit_price", "target_normal_price": "업로드 inventory/products.unit_price",
    "unit_cost": "업로드 inventory/products.unit_cost",
    "source_holding_cost_per_unit_day": "업로드 inventory/products.daily_holding_cost",
    "target_holding_cost_per_unit_day": "업로드 inventory/products.daily_holding_cost",
    "source_disposal_cost_per_unit": "업로드 inventory/products.disposal_cost_per_unit",
    "target_disposal_cost_per_unit": "업로드 inventory/products.disposal_cost_per_unit",
    "salvage_value_per_unit": "업로드 inventory/products.salvage_value_per_unit",
    "discount_rate": "관측 데이터 없음(ACTUAL_OPERATION은 config 프로모션 설정을 쓰지 않음)",
    "promotion_uplift": "업로드 inventory.promotion_uplift(관측값이 있을 때만)",
    "transfer_cost": "추천 이동비 move_cost(실제 운송 모드: 공식 요율 기반 추정 DERIVED_REAL, 상위 차종 대체 시 PROXY)",
    "transfer_cost_qty": "추천 실행 수량(운송비가 산정된 수량)",
    "transit_time_days": "경로 이동 시간 travel_time_min / 1440",
    "source_surplus_cap": "재고 전이 엔진 movable_stock", "target_need_cap": "재고 전이 엔진 target_shortage_limit",
    "route_capacity_qty": "현재 production에서 제공하지 않음",
    "target_store_id": "Varo Final 추천의 도착 점포",
}

# ---------------------------------------------------------------- seller-facing labels (no internal names shown)

FIELD_LABELS = {
    "decision_qty": "결정 수량",
    "source_current_stock": "현재 재고(출발 점포)",
    "target_current_stock": "현재 재고(도착 점포)",
    "source_daily_demand": "정상가 기준 하루 예상 판매량(출발 점포)",
    "target_daily_demand": "정상가 기준 하루 예상 판매량(도착 점포)",
    "remaining_shelf_life_days": "남은 판매 가능 기간(일)",
    "source_normal_price": "정상 판매가격(출발 점포)",
    "target_normal_price": "정상 판매가격(도착 점포)",
    "unit_cost": "개당 매입원가",
    "source_holding_cost_per_unit_day": "개당 하루 보관비(출발 점포)",
    "target_holding_cost_per_unit_day": "개당 하루 보관비(도착 점포)",
    "source_disposal_cost_per_unit": "개당 폐기 처리비(출발 점포)",
    "target_disposal_cost_per_unit": "개당 폐기 처리비(도착 점포)",
    "salvage_value_per_unit": "개당 잔존가치(폐기 시 회수액)",
    "discount_rate": "할인율",
    "promotion_uplift": "할인 시 예상 판매 증가율",
    "target_store_id": "도착 점포",
    "transfer_cost": "이동 운송비",
    "transfer_cost_qty": "운송비 산정 수량",
    "transit_time_days": "이동 소요 기간(일)",
    "source_surplus_cap": "이동 가능 재고(출발 점포)",
    "target_need_cap": "부족 수량(도착 점포)",
    "route_capacity_qty": "경로 운송 가능 수량",
    "quantity_unit": "재고 수량 단위",
}
COLUMN_LABELS = {
    "normal_price": "정상 판매가격", "daily_demand": "정상가 기준 하루 예상 판매량",
    "remaining_shelf_life_days": "남은 판매 가능 기간(일)", "discount_rate": "할인율",
    "promotion_uplift": "할인 시 예상 판매 증가율", "holding_cost_per_unit_day": "개당 하루 보관비",
    "disposal_cost_per_unit": "개당 폐기 처리비", "salvage_value_per_unit": "개당 잔존가치(폐기 시 회수액)",
    "unit_cost": "개당 매입원가", "transfer_cost": "1회 운송비", "transfer_cost_per_unit": "개당 운송비",
    "transit_time_days": "이동 소요 기간(일)", "quantity_unit": "재고 수량 단위",
}
INPUT_FORMATS = {
    "discount_rate": "0 초과 1 미만 소수(20% 할인 = 0.2)",
    "promotion_uplift": "0 이상 소수(할인 중 판매량 50% 증가 = 0.5, 증가 없음 = 0)",
    "remaining_shelf_life_days": "일 수(숫자만)", "transit_time_days": "일 수(12시간 = 0.5)",
    "daily_demand": "재고 수량 단위 기준 하루 판매량(숫자만)",
    "quantity_unit": "재고 수량 단위 문자(예: EA, BOX, KG)",
}
MONEY_FORMAT = "재고 수량 단위 1개당 금액(쉼표·단위 없이 숫자만)"
ROLE_LABELS = {"source": "출발 점포", "target": "도착 점포", None: "경로"}
# Why the engine needs each field; reasons are composed from these purposes, the request kind and the strategies.
FIELD_PURPOSE = {
    "decision_qty": "세 전략을 같은 수량으로 비교하기 위해",
    "source_current_stock": "결정 수량 외 출발 점포 재고가 함께 팔리는 것을 반영하기 위해",
    "target_current_stock": "도착 점포에 이미 있는 재고가 먼저 팔리는 것을 반영하기 위해",
    "source_daily_demand": "출발 점포에서 남은 기간 동안 정상가로 팔릴 수량을 계산하기 위해",
    "target_daily_demand": "이동한 재고가 도착 점포에서 판매될 수 있는지 판단하기 위해",
    "remaining_shelf_life_days": "남은 기간 안에 팔리는 수량과 팔리지 않는 수량을 나누기 위해",
    "source_normal_price": "팔리지 않는 재고의 손실과 할인·이동 시 금액 차이를 정상가 기준으로 계산하기 위해",
    "target_normal_price": "이동한 재고가 도착 점포에서 팔릴 때 받는 금액을 계산하기 위해",
    "discount_rate": "할인 판매 시 개당 할인 금액을 계산하기 위해",
    "promotion_uplift": "할인 후 판매량 변화를 계산하기 위해(관측값이나 판매자 근거 없이 할인 판매량을 추정하지 않음)",
    "transfer_cost": "이동에 드는 운송비를 손실에 넣기 위해",
    "transit_time_days": "이동 기간을 빼고 도착 점포의 판매 가능 기간을 계산하기 위해",
    "target_store_id": "재고를 옮길 점포를 정하기 위해",
    "transfer_cost_qty": "운송비가 몇 개 기준으로 산정됐는지 확인하기 위해",
    "source_holding_cost_per_unit_day": "출발 점포에 재고가 머무는 기간의 보관비를 손실에 넣기 위해",
    "target_holding_cost_per_unit_day": "도착 점포에 재고가 머무는 기간의 보관비를 손실에 넣기 위해",
    "source_disposal_cost_per_unit": "팔리지 않아 폐기하는 재고의 처리 비용을 손실에 넣기 위해",
    "target_disposal_cost_per_unit": "도착 점포에서 폐기하는 재고의 처리 비용을 손실에 넣기 위해",
    "salvage_value_per_unit": "폐기 재고에서 회수하는 금액을 반영하기 위해",
    "unit_cost": "매입원가를 기록하기 위해",
    "quantity_unit": "입력한 단위(예: 원/BOX)를 재고 수량 단위와 대조하기 위해",
    "source_surplus_cap": "출발 점포에서 옮길 수 있는 수량 한도를 지키기 위해",
    "target_need_cap": "도착 점포 부족 수량 한도를 지키기 위해",
    "route_capacity_qty": "경로 운송 한도를 지키기 위해",
}
# UX grouping only: a bundle never creates or fills a value.
INPUT_BUNDLES = (
    ("SALES", "판매 정보", ("normal_price", "daily_demand")),
    ("STOCK_LOSS", "재고 손실 정보", ("remaining_shelf_life_days", "disposal_cost_per_unit", "salvage_value_per_unit")),
    ("HOLDING", "보관 정보", ("holding_cost_per_unit_day",)),
    ("DISCOUNT", "할인 정보", ("discount_rate", "promotion_uplift")),
    ("TRANSPORT", "이동 정보", ("transfer_cost", "transfer_cost_per_unit", "transit_time_days")),
    ("UNIT", "단위 정보", ("quantity_unit",)),
)
BUNDLE_OF_COLUMN = {column: key for key, _, columns in INPUT_BUNDLES for column in columns}

# Engine reason codes, by what resolves them.
MISSING_CODE_FIELDS = {
    "DECISION_QTY_MISSING": "decision_qty", "SOURCE_STOCK_MISSING": "source_current_stock",
    "SOURCE_DEMAND_MISSING": "source_daily_demand", "SHELF_LIFE_MISSING": "remaining_shelf_life_days",
    "PRICE_MISSING": "source_normal_price", "DISCOUNT_RATE_MISSING": "discount_rate", "UPLIFT_MISSING": "promotion_uplift",
    "TARGET_STORE_MISSING": "target_store_id", "TARGET_STOCK_MISSING": "target_current_stock",
    "TARGET_DEMAND_MISSING": "target_daily_demand", "TARGET_PRICE_MISSING": "target_normal_price",
    "TRANSFER_COST_MISSING": "transfer_cost", "TRANSIT_TIME_MISSING": "transit_time_days",
}
FIELD_MISSING_CODE = {name: code for code, name in MISSING_CODE_FIELDS.items()}
CONSISTENCY_HEADS = frozenset({"NON_PHYSICAL_UNIT", "UNIT_MISMATCH", "UNIT_UNVERIFIABLE_ACROSS_SOURCES", "CURRENCY_MISMATCH",
                               "CURRENCY_UNDECLARED", "CROSS_DATASET_INPUT", "NEGATIVE_INPUT", "SELLER_UNIT_UNVERIFIABLE"})
# Value checks the engine runs once a stage's inputs are complete (field the value belongs to, None = several).
VALUE_CODE_FIELDS = {
    "ZERO_INVENTORY": "source_current_stock", "DECISION_QTY_NOT_POSITIVE": "decision_qty",
    "DECISION_QTY_EXCEEDS_STOCK": "decision_qty", "INVALID_DISCOUNT_RATE": "discount_rate",
    "INVALID_UPLIFT": "promotion_uplift", "INVALID_TRANSFER_COST_BASIS": "transfer_cost",
    "TRANSFER_COST_QTY_MISMATCH": "transfer_cost_qty", "NO_EXECUTABLE_TRANSFER_QTY": None, "TRANSFER_QTY_CAPPED": None,
}
SAME_SCOPE_CONFLICTS = ("SELLER_INPUT_SAME_SCOPE", "SCENARIO_INPUT_SAME_SCOPE")
SEED_ORIGINS = ("SELLER_LOSS_INPUTS", "SELLER_BUSINESS_PROFILE", "SCENARIO")
APPLIED_OUTCOMES = ("APPLIED", "APPLIED_OVER_PROXY", "APPLIED_OVER_CONFIG", "SCENARIO_OVERRIDE")
REAL = frozenset({"DIRECT_REAL", "DERIVED_REAL"})
AUTO_STATUS = {"DIRECT_REAL": ("AUTO_REAL", "자동 확보(실제값)"),
               "DERIVED_REAL": ("AUTO_DERIVED_REAL", "자동 산출(실제값 기반)"),
               "USER_INPUT": ("AUTO_UPLOAD", "업로드 값"),
               "DERIVED_FROM_USER_INPUT": ("AUTO_DERIVED_UPLOAD", "자동 산출(업로드 값 기반)"),
               "PROXY": ("AUTO_PROXY", "자동 산출이지만 대리지표(PROXY) — 금액 비교에 쓰지 않음"),
               "CONFIG": ("AUTO_CONFIG", "설정값(SCENARIO 전용)"),
               "SCENARIO_INPUT": ("AUTO_SCENARIO", "가정값(SCENARIO 전용)")}
ORDERING_RULE = ("1) tier: 1=전략 계산 차단 해제(입력·대리지표 교체·충돌/단위/통화 정정) > 2=추천 확정(순위가 달라지는 미상 값, 충돌, "
                 "가정값 교체) > 3=선택 입력; 2) 같은 tier 안에서 그 입력이 차단 사유를 없애는 전략 수(tier 2는 순위가 갈리는 "
                 "전략 쌍 수)가 많은 것 먼저; 3) 엔진이 값을 검사하는 순서(ENGINE_CHECK_ORDER); 4) 필드 이름")
STOP_ASKING_RULE = ("recommendation_readiness가 RECOMMENDABLE이고 판매자가 풀 수 있는 tier 1 입력이 없으면 더 묻지 않는다"
                    "(required_user_inputs = []). 남은 값은 optional_user_inputs에만 둔다.")
MINIMAL_INPUT_RULE = ("요청하는 값은 1) 요청 범위(comparison scope) 안 전략의 차단 사유를 실제로 없애거나 2) 비교 가능한 상태에서 추천 "
                      "순위가 그 값에 따라 갈리는 경우뿐이다. 판매자 입력으로 풀 수 없는 전략(업로드 데이터 부족)은 그 전략을 위한 "
                      "값을 묻지 않고, 범위 안에서 비교 가능해질 수 있는 전략이 2개 미만이면 아무것도 묻지 않는다.")


# ---------------------------------------------------------------- comparison scope


def resolve_comparison_scope(explicit: Any, uploaded_data: Mapping[str, Any] | None,
                             config: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Requested scope: argument > uploaded_data[COMPARISON_SCOPE_KEY] > config row > ALL_THREE (existing behaviour)."""
    candidates = (("argument", explicit),
                  (f"uploaded_data.{COMPARISON_SCOPE_KEY}", (uploaded_data or {}).get(COMPARISON_SCOPE_KEY)),
                  (f"config.{COMPARISON_SCOPE_KEY}", (config or {}).get(COMPARISON_SCOPE_KEY)))
    for basis, value in candidates:
        if sli._blank(value):
            continue
        scope = str(value).strip().upper()
        if scope in COMPARISON_SCOPES:
            return scope, {"comparison_scope": scope, "basis": basis}
        return DEFAULT_COMPARISON_SCOPE, {
            "comparison_scope": DEFAULT_COMPARISON_SCOPE, "basis": "default (invalid value rejected)",
            "error": {"code": "INVALID_COMPARISON_SCOPE", "source": basis, "value": repr(value),
                      "message": "seller_loss_comparison_scope는 " + ", ".join(COMPARISON_SCOPES) + " 중 하나여야 합니다"}}
    return DEFAULT_COMPARISON_SCOPE, {"comparison_scope": DEFAULT_COMPARISON_SCOPE, "basis": "default"}


def strategy_required_fields(strategy: str) -> tuple[str, ...]:
    """Inputs the strategy cannot be computed without: the evidence layer's lists + decision_qty (+ target store)."""
    names = ["decision_qty", *sli.STRATEGY_REQUIRED[strategy]]
    if strategy == TRANSFER:
        names.insert(names.index("target_current_stock"), "target_store_id")
    return tuple(names)


def _scope_text(strategies: Sequence[str]) -> str:
    labels = "·".join(STRATEGY_LABELS[s] for s in strategies)
    return f"세 전략({labels})" if len(strategies) == 3 else labels


# ---------------------------------------------------------------- execution-path trace (structural probe)

PROBE_DATASET, PROBE_CURRENCY = "STRUCTURAL_PROBE", "XXX"   # ISO 4217 "XXX" = no currency


def _probe_references() -> dict[str, SellerDecisionInput]:
    """STRUCTURAL PROBES ONLY: complete synthetic decisions on which every strategy is computable.

    They exist to observe which fields each strategy's code path reads; their numbers never reach a plan, a result or a
    seller. The shapes cover the branches: full transfer, a transfer capped below Q by each shared constraint (units
    kept at the source) and a quantity-specific transfer cost.
    """
    def q(value: float) -> InputField:
        return known(value, "USER_INPUT", "structural probe", dataset=PROBE_DATASET)

    def m(value: float) -> InputField:
        return known(value, "USER_INPUT", "structural probe", currency=PROBE_CURRENCY, dataset=PROBE_DATASET)

    full = SellerDecisionInput(
        decision_id="PROBE", product_id="PROBE", source_store_id="PROBE_SOURCE", target_store_id="PROBE_TARGET",
        decision_qty=q(20), source_current_stock=q(50), source_daily_demand=q(3), remaining_shelf_life_days=q(10),
        source_normal_price=m(1000), unit_cost=m(600), source_holding_cost_per_unit_day=m(5),
        source_disposal_cost_per_unit=m(200), salvage_value_per_unit=m(10), discount_rate=q(0.3), promotion_uplift=q(0.5),
        target_current_stock=q(5), target_daily_demand=q(4), target_normal_price=m(1000),
        target_holding_cost_per_unit_day=m(5), target_disposal_cost_per_unit=m(200), transfer_cost=m(1800),
        transfer_cost_basis="FIXED_PER_TRIP", transit_time_days=q(0.1), source_surplus_cap=q(100), target_need_cap=q(100),
        route_capacity_qty=q(100),
    )
    return {"FIXED_COST_FULL_TRANSFER": full,
            "FIXED_COST_ROUTE_CAP_LIMITED": replace(full, route_capacity_qty=q(10)),
            "FIXED_COST_SOURCE_CAP_LIMITED": replace(full, source_surplus_cap=q(10)),
            "FIXED_COST_TARGET_CAP_LIMITED": replace(full, target_need_cap=q(10)),
            "QUANTITY_SPECIFIC_COST": replace(full, transfer_cost_basis="QUANTITY_SPECIFIC", transfer_cost_qty=q(20))}


def _probe_variant(reference: SellerDecisionInput, name: str, variant: str) -> SellerDecisionInput | None:
    if name == "target_store_id":
        return replace(reference, target_store_id=None) if variant == MISSING else None
    item = getattr(reference, name)
    if not item.present:
        return None
    if variant == MISSING:
        return replace(reference, **{name: InputField(source="structural probe: removed")})
    return replace(reference, **{name: replace(item, provenance="PROXY")})


@lru_cache(maxsize=1)
def _trace() -> dict[str, Any]:
    references = _probe_references()
    baseline = {name: evaluate_seller_decision(inp) for name, inp in references.items()}
    for name, result in baseline.items():
        if result["comparable_strategies"] != list(STRATEGIES):
            raise RuntimeError(f"structural probe {name} is not complete: {result['unavailable_strategies']}")
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for name in (*INPUT_FIELD_NAMES, "target_store_id"):
        per: dict[str, dict[str, Any]] = {s: {"unavailable_in": [], "unknown_in": [], "changed_in": [], "reason_codes": [],
                                              "proxy_unavailable_in": [], "proxy_unknown_in": []} for s in STRATEGIES}
        for ref_name, reference in references.items():
            for variant in (MISSING, PROXY):
                probe = _probe_variant(reference, name, variant)
                if probe is None:
                    continue
                result = evaluate_seller_decision(probe)
                for strategy in STRATEGIES:
                    new, old, entry = result["strategies"][strategy], baseline[ref_name]["strategies"][strategy], per[strategy]
                    prefix = "" if variant == MISSING else "proxy_"
                    if not new["available"]:
                        entry[prefix + "unavailable_in"].append(ref_name)
                        entry["reason_codes"].extend(c for c in new["unavailable_reasons"] if c not in entry["reason_codes"])
                    elif name in new["unknown_inputs"] and name not in old["unknown_inputs"]:
                        entry[prefix + "unknown_in"].append(ref_name)
                    elif variant == MISSING and (new["expected_loss_excluding_unknown"] != old["expected_loss_excluding_unknown"]
                                                 or new["quantities"] != old["quantities"]):
                        entry["changed_in"].append(ref_name)
        for strategy, entry in per.items():
            if len(entry["unavailable_in"]) == len(references):
                entry["class"], entry["condition"] = REQUIRED, None
            elif entry["unavailable_in"]:
                entry["class"] = CONDITIONAL
                entry["condition"] = ("TRANSFER_COST_BASIS_QUANTITY_SPECIFIC" if entry["unavailable_in"] == ["QUANTITY_SPECIFIC_COST"]
                                      else "PROBES:" + "|".join(entry["unavailable_in"]))
            elif entry["unknown_in"]:
                entry["class"], entry["condition"] = CONDITIONAL, "RANKING_DEPENDS_ON_THIS_UNKNOWN_RATE"
            elif entry["changed_in"]:
                entry["class"], entry["condition"] = OPTIONAL, "CHANGES_QUANTITY_OR_LOSS_ONLY"
            else:
                entry["class"], entry["condition"] = NOT_USED, None
        matrix[name] = per
    return {"references": sorted(references), "matrix": matrix}


def trace_engine_requirements() -> dict[str, Any]:
    """Field x strategy classes observed by removing (or proxying) one field at a time on the structural probes.

    REQUIRED: the strategy is unavailable without it on every probe. CONDITIONALLY_REQUIRED: unavailable on some probes
    (a condition such as the cost basis) or entering the loss as a bounded unknown (robustness). OPTIONAL: changes only
    quantities/loss. NOT_USED: no effect at all. The per-field class adds DERIVED for values production computes.
    """
    trace = _trace()
    fields_out: dict[str, Any] = {}
    for name, per in trace["matrix"].items():
        classes = {per[s]["class"] for s in STRATEGIES}
        if name in DERIVED_FIELDS:
            summary = DERIVED
        else:
            summary = next((c for c in (REQUIRED, CONDITIONAL, OPTIONAL) if c in classes), OPTIONAL)
        fields_out[name] = {
            "field_class": summary, "label": FIELD_LABELS.get(name, name),
            "seller_enterable": name in SELLER_ENTERABLE, "production_source": PRODUCTION_SOURCES.get(name),
            "used_by_strategies": [s for s in STRATEGIES if per[s]["class"] != NOT_USED],
            "per_strategy": {s: {k: v for k, v in per[s].items()} for s in STRATEGIES},
        }
    return {"references": trace["references"], "fields": fields_out,
            "note": "Structural probes are synthetic and only reveal code paths; no probe value is ever used as an input."}


def traced_inputs(strategy: str, cls: str) -> tuple[str, ...]:
    return tuple(name for name in ENGINE_CHECK_ORDER if name in _trace()["matrix"]
                 and _trace()["matrix"][name][strategy]["class"] == cls)


# ---------------------------------------------------------------- field states


def _state(name: str, merged: SellerDecisionInput, same_scope: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if name == "target_store_id":
        present = bool(merged.target_store_id)
        return {"state": AVAILABLE if present else MISSING, "value": merged.target_store_id,
                "provenance": "RECOMMENDATION" if present else "MISSING", "source": "recommendation.target_id"}
    item: InputField = getattr(merged, name)
    if item.usable:
        state = AVAILABLE
    elif item.present:
        state = PROXY
    elif name in same_scope:
        state = CONFLICT
    else:
        state = MISSING
    return {"state": state, "value": item.value, "provenance": item.provenance, "source": item.source,
            "currency": item.currency, "unit": item.unit}


def _origin(name: str, merged: SellerDecisionInput, fields_info: Mapping[str, Any]) -> str:
    if name == "target_store_id":
        return "DATA"
    return sli._origin(name, getattr(merged, name), fields_info)


def _seller_origin(name: str, merged: SellerDecisionInput, fields_info: Mapping[str, Any]) -> bool:
    return _origin(name, merged, fields_info) in SEED_ORIGINS


def _decision_key(merged: SellerDecisionInput) -> dict[str, Any]:
    return {"decision_id": merged.decision_id, "product_id": merged.product_id,
            "source_store_id": merged.source_store_id, "target_store_id": merged.target_store_id}


def _expected_currency(merged: SellerDecisionInput, fields_info: Mapping[str, Any], upload_currency: str | None) -> str | None:
    money = {name: item.currency for name, item in merged.input_fields().items()
             if name in MONEY_FIELDS and item.present and item.currency}
    data = {c for name, c in money.items() if not _seller_origin(name, merged, fields_info)}
    if data:
        return next(iter(data)) if len(data) == 1 else None
    if upload_currency:
        return upload_currency
    entered = set(money.values())   # only seller values so far: keep the currency they already share
    return next(iter(entered)) if len(entered) == 1 else None


def _seller_entry(name: str, merged: SellerDecisionInput, currency: str | None) -> dict[str, Any]:
    if name == "quantity_unit":
        columns, role = ("quantity_unit",), "source"
    else:
        columns, role = sli.ENGINE_TARGETS[name]
    column = columns[0]
    if role is None:
        scope, keys = "ROUTE", {"source_store_id": merged.source_store_id, "target_store_id": merged.target_store_id}
    else:
        store = merged.source_store_id if role == "source" else merged.target_store_id
        scope, keys = "PRODUCT_STORE", {"product_id": merged.product_id, "store_id": store}
    entry: dict[str, Any] = {
        "sheet": sli.SHEET_KEY, "column": column, "column_label": COLUMN_LABELS[column],
        "alternative_columns": list(columns[1:]), "suggested_scope": scope, "keys": keys, "store_role": role,
        "store_role_label": ROLE_LABELS[role],
        "format": MONEY_FORMAT if column in sli.MONEY_COLUMNS else INPUT_FORMATS.get(column, "숫자만"),
        "scope_note": "같은 값이면 PRODUCT·STORE·GLOBAL 범위 한 행으로 여러 결정에 함께 입력할 수 있습니다",
    }
    if column in sli.MONEY_COLUMNS:
        entry["expected_currency"] = currency
    if column in sli.UNIT_BOUND_COLUMNS:
        entry["expected_unit"] = merged.quantity_unit or "재고 수량 단위(미선언: 단위 없이 숫자만 입력하면 재고 단위 기준)"
    entry["entry_key"] = f"{column}|{scope}|" + "|".join(f"{k}={v}" for k, v in sorted(keys.items()))
    return entry


# ---------------------------------------------------------------- blockers (consistency, input issues, values)


def _fix(field: str, kind: str, detail: str, expected: Any = None) -> dict[str, Any]:
    return {"field": field, "kind": kind, "detail": detail, "expected": expected}


def _consistency_blocker(code: str, merged: SellerDecisionInput, fields_info: Mapping[str, Any], partition: Iterable[str],
                         currency: str | None) -> dict[str, Any]:
    """One engine consistency/input-issue code -> involved fields + who can fix it (seller correction or upload data)."""
    head, _, target = code.partition(":")
    items = {name: item for name, item in merged.input_fields().items() if name in set(partition) and item.present}
    seller = lambda name: _seller_origin(name, merged, fields_info)  # noqa: E731
    involved: list[str] = []
    fixes: list[dict[str, Any]] = []
    expected: Any = None
    if head == "CURRENCY_MISMATCH":
        money = {n: i.currency for n, i in items.items() if n in MONEY_FIELDS and i.currency}
        involved = sorted(money)
        data = {c for n, c in money.items() if not seller(n)}
        expected = next(iter(data)) if len(data) == 1 else (currency if not data else None)
        if expected is not None:
            fixes = [_fix(n, CORRECT_CURRENCY, f"{c} -> {expected}", expected) for n, c in sorted(money.items())
                     if seller(n) and c != expected]
    elif head == "CURRENCY_UNDECLARED":
        involved = sorted(n for n, i in items.items() if n in MONEY_FIELDS and not i.currency)
        fixes = [_fix(n, CORRECT_CURRENCY, "currency missing", currency) for n in involved if seller(n)]
        expected = currency
    elif head == "UNIT_MISMATCH" and target:
        involved, expected = [target], merged.quantity_unit
        fixes = [_fix(target, CORRECT_UNIT, f"per-unit value not in {merged.quantity_unit}", merged.quantity_unit)] if seller(target) else []
    elif head == "UNIT_MISMATCH":
        sized = {n: i.unit for n, i in items.items() if (n in QUANTITY_FIELDS or n in PER_UNIT_MONEY_FIELDS) and i.unit}
        involved = sorted(sized)
        data = {str(u).casefold() for n, u in sized.items() if not seller(n)}
        # The declared inventory unit (data or seller-declared quantity_unit) is the reference for per-unit values.
        expected = merged.quantity_unit or (next(iter(sized[n] for n in sized if not seller(n))) if len(data) == 1 else None)
        if expected is not None:
            fixes = [_fix(n, CORRECT_UNIT, f"{u} -> {expected}", expected) for n, u in sorted(sized.items())
                     if seller(n) and str(u).casefold() != str(expected).casefold()]
    elif head in ("SELLER_UNIT_UNVERIFIABLE", "UNIT_UNVERIFIABLE_ACROSS_SOURCES"):
        involved = [target] if target else sorted(n for n, i in items.items()
                                                  if (n in QUANTITY_FIELDS or n in PER_UNIT_MONEY_FIELDS) and not i.unit)
        if not merged.quantity_unit:
            fixes = [_fix("quantity_unit", DECLARE_UNIT, "declare the inventory quantity unit")]
    elif head == "NEGATIVE_INPUT":
        involved = [target]
        fixes = [_fix(target, CORRECT_VALUE, "negative value")] if seller(target) else []
    else:   # NON_PHYSICAL_UNIT, CROSS_DATASET_INPUT: only the upload can fix them
        involved = sorted(n for n in items if n in QUANTITY_FIELDS or n in PER_UNIT_MONEY_FIELDS) if head == "NON_PHYSICAL_UNIT" else sorted(items)
    return {"code": code, "message": _reason_text(code), "fields": involved, "expected": expected,
            "resolution": SELLER if fixes else UPLOAD_DATA, "fixes": fixes}


def _value_blocker(code: str, merged: SellerDecisionInput, fields_info: Mapping[str, Any]) -> dict[str, Any]:
    head = code.partition(":")[0]
    name = VALUE_CODE_FIELDS.get(head)
    seller = name is not None and name in SELLER_ENTERABLE and _seller_origin(name, merged, fields_info)
    fixes = [_fix(name, CORRECT_VALUE, head)] if seller and head in ("INVALID_DISCOUNT_RATE", "INVALID_UPLIFT") else []
    if fixes:
        resolution = SELLER
    elif head in ("NO_EXECUTABLE_TRANSFER_QTY", "TRANSFER_QTY_CAPPED"):
        resolution = "CURRENT_VALUES"   # an outcome of the stated values, not a missing input
    else:
        resolution = UPLOAD_DATA
    return {"code": code, "message": _reason_text(code), "field": name, "resolution": resolution, "fixes": fixes}


# ---------------------------------------------------------------- ranking dependency (robustness)


def ranking_dependency(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Unknown inputs the non-robust ranking depends on.

    A pair of comparable strategies is ambiguous when neither wins over the whole admissible range of the unknowns
    (engine `_margin` < 0 both ways). The decisive inputs are the unknowns whose loss coefficient differs inside an
    ambiguous pair. Once all of them are known, every pair is decided at every remaining point, so a robust winner
    always exists. Unknowns that only appear in decided pairs never change the recommendation.
    """
    empty = {"ambiguous_pairs": [], "decisive_inputs": [], "non_decisive_unknowns": list(decision.get("unknown_inputs") or [])}
    if decision.get("comparison_status") == STATUS_UNAVAILABLE or decision.get("recommendation_status") != ENGINE_NOT_ROBUST:
        return empty
    comparable = list(decision["comparable_strategies"])
    bounds = {name: (float(lo), math.inf if hi is None else float(hi))
              for name, (lo, hi) in (decision.get("unknown_input_bounds") or {}).items()}
    lin = {s: (float(decision["strategies"][s]["loss_linear"]["constant"]),
               {k: float(v) for k, v in decision["strategies"][s]["loss_linear"]["unknown_coefficients"].items()})
           for s in comparable}
    pairs: list[dict[str, Any]] = []
    for index, first in enumerate(comparable):
        for second in comparable[index + 1:]:
            names = sorted(set(lin[first][1]) | set(lin[second][1]))
            diff = {n: round(lin[second][1].get(n, 0.0) - lin[first][1].get(n, 0.0), 6) for n in names}
            diff = {n: v for n, v in diff.items() if abs(v) > 1e-9}
            if not diff:
                continue
            if _margin(lin[first], lin[second], bounds) < -TOL and _margin(lin[second], lin[first], bounds) < -TOL:
                pairs.append({"strategies": [first, second], "inputs": sorted(diff), "coefficient_difference": diff})
    if not pairs:   # rounding residue: fall back to every pair with a differing unknown (never under-asks)
        for index, first in enumerate(comparable):
            for second in comparable[index + 1:]:
                diff = {n: round(lin[second][1].get(n, 0.0) - lin[first][1].get(n, 0.0), 6)
                        for n in sorted(set(lin[first][1]) | set(lin[second][1]))}
                diff = {n: v for n, v in diff.items() if abs(v) > 1e-9}
                if diff:
                    pairs.append({"strategies": [first, second], "inputs": sorted(diff), "coefficient_difference": diff})
    decisive = sorted({n for pair in pairs for n in pair["inputs"]}, key=lambda n: (ORDER_INDEX.get(n, 99), n))
    return {"ambiguous_pairs": pairs, "decisive_inputs": decisive,
            "non_decisive_unknowns": [n for n in decision.get("unknown_inputs") or [] if n not in decisive]}


def _pair_text(name: str, pairs: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for pair in pairs:
        first, second = pair["strategies"]
        coef = pair["coefficient_difference"].get(name)
        if coef is None:
            continue
        heavier = second if coef > 0 else first     # coef = coefficient(second) - coefficient(first)
        lighter = first if heavier == second else second
        parts.append(f"이 값이 클수록 {STRATEGY_LABELS[heavier]}의 손실이 {STRATEGY_LABELS[lighter]}보다 더 커집니다")
    return "; ".join(dict.fromkeys(parts))


# ---------------------------------------------------------------- the plan


def _reason(kind: str, name: str, strategies: Sequence[str], state: Mapping[str, Any], extra: str = "") -> str:
    purpose = FIELD_PURPOSE.get(name, "")
    blocked = f"{_scope_text(strategies)} 계산이 막혀 있습니다" if strategies else ""
    if kind == ENTER_MISSING:
        return f"{purpose} 필요합니다. 값이 없어 {blocked}."
    if kind == REPLACE_PROXY:
        value = state.get("value")
        shown = f"{value:g}" if isinstance(value, (int, float)) else str(value)
        return (f"{purpose} 필요합니다. 현재 값({shown}, {state.get('source')})은 대리지표(PROXY)라 금액 비교에 쓸 수 없어 "
                f"{blocked}. 실제 값을 입력하면 대리지표 대신 쓰입니다.")
    if kind == RESOLVE_CONFLICT:
        return (f"같은 범위에 서로 다른 값({extra})이 입력되어 어느 값도 쓰지 않았습니다"
                + (f". {blocked}" if strategies else "; 충돌이 남아 있으면 추천 가능(RECOMMENDABLE)이 되지 않습니다")
                + ". 한 값으로 정리하거나 더 구체적인 범위에 한 값을 입력하세요.")
    if kind == CORRECT_CURRENCY:
        return f"입력한 금액의 통화가 다른 금액과 달라({extra}) {blocked}. 통화는 환산하지 않으므로 같은 통화로 다시 입력하세요."
    if kind == CORRECT_UNIT:
        return f"입력한 값의 단위가 재고 수량 단위와 달라({extra}) {blocked}. 재고 수량 단위 기준으로 다시 입력하세요."
    if kind == DECLARE_UNIT:
        return (f"{purpose} 필요합니다. 단위가 적힌 값을 재고 수량 단위와 대조할 수 없어 {blocked}. 재고 수량 단위를 "
                f"입력하거나 입력값의 단위 표기를 지우면(재고 단위 기준으로 간주) 풀립니다.")
    if kind == CORRECT_VALUE:
        return f"입력한 값이 허용 범위 밖이라({extra}) {blocked}. 값을 고쳐 입력하세요."
    if kind == RESOLVE_UNKNOWN:
        return (f"{purpose} 필요합니다. 값이 없어 미상(0 이상 어떤 값이든 가능)으로 두었는데, {extra}. 그래서 어느 방법이 "
                f"손실이 작은지 확정할 수 없습니다. 입력하면 추천을 확정할 수 있습니다.")
    if kind == REPLACE_SCENARIO_VALUE:
        return (f"현재 값은 가정·설정값(관측치 아님)이라 실제 운영 추천에 쓸 수 없습니다. 실제 값을 입력하면 설정값 대신 "
                f"쓰입니다.")
    if kind == IMPROVE_PRECISION:
        return ("입력하지 않아도 추천은 바뀌지 않습니다(허용 범위의 모든 값에서 같은 결과). 입력하면 예상 손실이 범위가 아닌 "
                "정확한 금액이 됩니다.")
    if kind == PENDING_EVALUATION:
        return (f"지금은 필요하지 않습니다. 필수 입력 후 비교했을 때 이 값에 따라 추천 순위가 달라지는 경우에만 요청합니다"
                f"({purpose}).")
    return purpose


def _item(name: str, kind: str, tier: int, cls: str, state: Mapping[str, Any], merged: SellerDecisionInput,
          currency: str | None) -> dict[str, Any]:
    entry = _seller_entry(name, merged, currency)
    return {"field": name, "label": FIELD_LABELS.get(name, name), "requirement_class": cls, "request_kind": kind,
            "tier": tier, "tier_name": TIERS[tier],
            "current": {k: state.get(k) for k in ("state", "value", "provenance", "source")},
            "unblocks_strategies": [], "unblock_count": 0, "unblocks_recommendation": False, "ranking_pairs": [],
            "seller_entry": entry, "bundle": BUNDLE_OF_COLUMN.get(entry["column"]), "reason": "", "reason_code": kind,
            "detail": ""}


def _rank_key(item: Mapping[str, Any]) -> tuple:
    weight = item["unblock_count"] if item["tier"] == 1 else len(item["ranking_pairs"]) or 1
    return (item["tier"], -weight, ORDER_INDEX.get(item["field"], 99), item["field"])


def _priority_basis(item: Mapping[str, Any]) -> str:
    if item["tier"] == 1:
        return (f"tier 1(전략 차단 해제): 차단 사유를 없애는 전략 {item['unblock_count']}개"
                f"({', '.join(STRATEGY_LABELS[s] for s in item['unblocks_strategies'])}); 엔진 검사 순서 {ORDER_INDEX.get(item['field'])}")
    if item["tier"] == 2:
        return (f"tier 2(추천 확정): 순위가 갈리는 전략 쌍 {len(item['ranking_pairs'])}개; "
                f"엔진 검사 순서 {ORDER_INDEX.get(item['field'])}")
    return f"tier 3(선택): 엔진 검사 순서 {ORDER_INDEX.get(item['field'])}"


def build_plan(base: SellerDecisionInput, merged: SellerDecisionInput, record: Mapping[str, Any],
               decision: Mapping[str, Any], *, comparison_scope: str | None = None,
               upload_currency: str | None = None) -> dict[str, Any]:
    """Requirement plan for one evaluated decision (pure; never changes ``decision``)."""
    scope = comparison_scope if comparison_scope in COMPARISON_SCOPES else DEFAULT_COMPARISON_SCOPE
    scope_strategies = COMPARISON_SCOPES[scope]
    mode = decision.get("decision_mode", sli.ACTUAL_OPERATION)
    fields_info = record.get("fields") or {}
    conflicts = list(record.get("conflicts") or [])
    same_scope = {c["field"]: c for c in conflicts if c["kind"] in SAME_SCOPE_CONFLICTS}
    tracked = [n for n in ENGINE_CHECK_ORDER if n != "quantity_unit"]
    states = {name: _state(name, merged, same_scope) for name in tracked}
    currency = _expected_currency(merged, fields_info, upload_currency)

    core_codes, transfer_codes, _ = _consistency_blockers(merged)
    for issue in merged.input_issues:
        (transfer_codes if issue.partition(":")[2] in TRANSFER_ONLY_FIELDS else core_codes).append(issue)
    core_fields = [n for n in INPUT_FIELD_NAMES if n not in TRANSFER_ONLY_FIELDS]
    core_blockers = [_consistency_blocker(c, merged, fields_info, core_fields, currency) for c in dict.fromkeys(core_codes)]
    transfer_blockers = [_consistency_blocker(c, merged, fields_info, INPUT_FIELD_NAMES, currency)
                         for c in dict.fromkeys(transfer_codes)]

    strategy_requirements: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        unmet = [n for n in strategy_required_fields(strategy) if states[n]["state"] != AVAILABLE]
        blockers = core_blockers + (transfer_blockers if strategy == TRANSFER else [])
        value_blockers = [_value_blocker(code, merged, fields_info)
                          for code in decision["strategies"][strategy]["unavailable_reasons"]
                          if code.partition(":")[0] in VALUE_CODE_FIELDS]
        available = bool(decision["strategies"][strategy]["available"])
        data_fields = [n for n in unmet if n not in SELLER_ENTERABLE]
        data_blockers = [b["code"] for b in blockers if b["resolution"] != SELLER]
        stuck_values = [v["code"] for v in value_blockers if v["resolution"] != SELLER]
        if available:
            status = READY
        elif data_fields or data_blockers:
            status = BLOCKED_BY_DATA
        elif stuck_values:
            status = BLOCKED_BY_VALUES
        else:
            status = NEEDS_SELLER_INPUT
        strategy_requirements[strategy] = {
            "in_scope": strategy in scope_strategies, "status": status, "engine_available": available,
            "unmet_fields": unmet, "unmet_states": {n: states[n]["state"] for n in unmet},
            "blockers": [{k: v for k, v in b.items() if k != "fixes"} for b in blockers],
            "value_blockers": [{k: v for k, v in v.items() if k != "fixes"} for v in value_blockers],
            "upload_data_fields": data_fields, "upload_data_blockers": data_blockers, "value_outcomes": stuck_values,
            "_fixes": [f for b in blockers for f in b["fixes"]] + [f for v in value_blockers for f in v["fixes"]],
        }

    reachable = [s for s in scope_strategies if strategy_requirements[s]["status"] in (READY, NEEDS_SELLER_INPUT)]
    scope_reachable = len(reachable) >= 2
    requests: dict[str, dict[str, Any]] = {}
    not_requested: list[dict[str, Any]] = []

    def add(name: str, kind: str, tier: int, cls: str, strategy: str | None, detail: str = "") -> dict[str, Any]:
        item = requests.get(name)
        if item is None:
            item = requests[name] = _item(name, kind, tier, cls, states.get(name, {"state": MISSING}), merged, currency)
            item["detail"] = detail
        if strategy and strategy not in item["unblocks_strategies"]:
            item["unblocks_strategies"].append(strategy)
        return item

    for strategy in scope_strategies:
        req = strategy_requirements[strategy]
        if req["status"] != NEEDS_SELLER_INPUT:
            continue
        if not scope_reachable:
            not_requested.extend({"field": n, "label": FIELD_LABELS.get(n, n), "strategy": strategy,
                                  "reason": "범위 안에서 비교 가능해질 수 있는 전략이 2개 미만이라 요청하지 않음"} for n in req["unmet_fields"])
            continue
        for name in req["unmet_fields"]:
            state = states[name]["state"]
            kind = {PROXY: REPLACE_PROXY, CONFLICT: RESOLVE_CONFLICT}.get(state, ENTER_MISSING)
            detail = (", ".join(str(v) for v in same_scope[name]["seller_values"]) + f" / 행 {same_scope[name]['rows']}"
                      if state == CONFLICT else "")
            add(name, kind, 1, REQUIRED, strategy, detail)
        for fix in req["_fixes"]:
            add(fix["field"], fix["kind"], 1, CONDITIONAL if fix["kind"] == DECLARE_UNIT else REQUIRED, strategy, fix["detail"])
    for strategy in STRATEGIES:
        if not strategy_requirements[strategy]["in_scope"]:
            continue
        if strategy_requirements[strategy]["status"] in (BLOCKED_BY_DATA, BLOCKED_BY_VALUES):
            not_requested.extend({"field": n, "label": FIELD_LABELS.get(n, n), "strategy": strategy,
                                  "reason": "이 전략은 업로드 데이터나 현재 값 때문에 막혀 있어 판매자 입력으로 풀 수 없음"}
                                 for n in strategy_requirements[strategy]["unmet_fields"] if n in SELLER_ENTERABLE)

    # tier 2: what keeps a comparable result from being RECOMMENDABLE (and the seller can fix)
    comparable = list(decision.get("comparable_strategies") or [])
    dependency = ranking_dependency(decision)
    blocked_fields: list[dict[str, Any]] = []
    if decision.get("comparison_status") != STATUS_UNAVAILABLE:
        for name in dependency["decisive_inputs"]:
            pairs = [p for p in dependency["ambiguous_pairs"] if name in p["inputs"]]
            item_state = states.get(name, {"state": MISSING})
            if name not in SELLER_ENTERABLE or item_state["state"] == AVAILABLE:
                blocked_fields.append({"field": name, "label": FIELD_LABELS.get(name, name), "kind": "RANKING_DEPENDS_ON_DATA",
                                       "reason": "추천 순위가 이 값에 따라 달라지지만 실제/업로드 데이터 값이라 판매자 입력으로 바꿀 수 없습니다"
                                                 + (" (실행 수량의 정확한 운송비를 모름: 운송비가 더 큰 수량 기준)" if name == "transfer_cost" else "")})
                continue
            kind = REPLACE_PROXY if item_state["state"] == PROXY else (RESOLVE_CONFLICT if item_state["state"] == CONFLICT else RESOLVE_UNKNOWN)
            item = add(name, kind, 2, CONDITIONAL, None, _pair_text(name, pairs))
            item["unblocks_recommendation"] = True
            item["ranking_pairs"] = [p["strategies"] for p in pairs]
            item["involves_out_of_scope_strategy"] = any(s not in scope_strategies for p in pairs for s in p["strategies"])
        if mode == sli.ACTUAL_OPERATION:
            for name in decision.get("scenario_inputs") or []:
                if name in SELLER_ENTERABLE and name not in requests:
                    add(name, REPLACE_SCENARIO_VALUE, 2, CONDITIONAL, None)["unblocks_recommendation"] = True
    for name, conflict in sorted(same_scope.items()):
        if name in requests or name not in SELLER_ENTERABLE:
            continue
        item = add(name, RESOLVE_CONFLICT, 2, CONDITIONAL, None,
                   ", ".join(str(v) for v in conflict["seller_values"]) + f" / 행 {conflict['rows']}")
        item["unblocks_recommendation"] = True

    # tier 3: optional (never required)
    optional: dict[str, dict[str, Any]] = {}
    unknown = list(decision.get("unknown_inputs") or [])
    if comparable and decision.get("comparison_status") != STATUS_UNAVAILABLE:
        for name in [n for n in unknown if n not in dependency["decisive_inputs"]]:
            if name in SELLER_ENTERABLE and name not in requests and states.get(name, {}).get("state") != AVAILABLE:
                optional[name] = _item(name, IMPROVE_PRECISION, 3, OPTIONAL, states[name], merged, currency)
    scope_complete = all(s in comparable for s in scope_strategies)
    if not scope_complete:
        for strategy in reachable:
            for name in traced_inputs(strategy, CONDITIONAL):
                if (name in SELLER_ENTERABLE and name not in requests and name not in optional
                        and states[name]["state"] != AVAILABLE):
                    optional[name] = _item(name, PENDING_EVALUATION, 3, CONDITIONAL, states[name], merged, currency)

    for item in requests.values():
        item["unblock_count"] = len(item["unblocks_strategies"])
        item["unblocks_strategies"] = [s for s in STRATEGIES if s in item["unblocks_strategies"]]
        item["reason"] = _reason(item["request_kind"], item["field"], item["unblocks_strategies"],
                                 states.get(item["field"], {}), item["detail"])
        item["priority_basis"] = _priority_basis(item)
    for item in optional.values():
        item["reason"] = _reason(item["request_kind"], item["field"], [], states.get(item["field"], {}))
        item["priority_basis"] = _priority_basis(item)

    readiness = decision.get("recommendation_readiness")
    tier1 = [i for i in requests.values() if i["tier"] == 1]
    tier2 = [i for i in requests.values() if i["tier"] == 2]
    stop_asking = readiness == sli.RECOMMENDABLE and not tier1
    required = [] if stop_asking else sorted(requests.values(), key=_rank_key)
    optional_items = sorted(optional.values(), key=_rank_key)
    recommended = [dict(i) for i in (*required, *optional_items)]
    for position, item in enumerate(recommended, start=1):
        item["priority"] = position
    for collection in (required, optional_items):
        for item in collection:
            item["priority"] = next(r["priority"] for r in recommended if r["field"] == item["field"])

    if mode == sli.SCENARIO and not tier1:
        status = "SCENARIO_ONLY"
    elif stop_asking:
        status = "READY_RECOMMENDABLE" if scope_complete else "READY_RECOMMENDABLE_SCOPE_LIMITED"
    elif tier1:
        status = "RECOMMENDABLE_PARTIAL_SCOPE" if readiness == sli.RECOMMENDABLE else "NEEDS_INPUT"
    elif tier2:
        status = "NEEDS_INPUT_FOR_ROBUST_RECOMMENDATION"
    elif comparable and readiness == sli.NOT_ROBUST:
        status = "NOT_ROBUST_DATA_DEPENDENT"
    elif any(strategy_requirements[s]["status"] == BLOCKED_BY_DATA for s in scope_strategies):
        status = "BLOCKED_BY_DATA"
    else:
        status = "BLOCKED_BY_VALUES"

    for strategy, req in strategy_requirements.items():
        for name in req["upload_data_fields"]:
            if req["in_scope"] and not any(b["field"] == name for b in blocked_fields):
                blocked_fields.append({"field": name, "label": FIELD_LABELS.get(name, name), "kind": "UPLOAD_DATA_MISSING",
                                       "strategies": [s for s in scope_strategies if name in strategy_requirements[s]["upload_data_fields"]],
                                       "reason": "판매자 입력(seller_loss_inputs)으로 넣을 수 없는 값이라 업로드 데이터에 있어야 합니다"})
    value_outcomes = [{"strategy": s, **v} for s in scope_strategies for v in strategy_requirements[s]["value_blockers"]]
    for req in strategy_requirements.values():
        req.pop("_fixes")

    applied_audit = [a for a in record.get("audit") or [] if a["outcome"] in APPLIED_OUTCOMES]
    relevant = list(dict.fromkeys(n for s in scope_strategies for n in (*strategy_required_fields(s), *traced_inputs(s, CONDITIONAL))))
    used = set(decision.get("used_input_fields") or [])
    available_fields = []
    for name in relevant:
        if states[name]["state"] != AVAILABLE:
            continue
        origin = _origin(name, merged, fields_info)
        provenance = states[name]["provenance"]
        available_fields.append({
            "field": name, "label": FIELD_LABELS.get(name, name), "value": states[name]["value"], "provenance": provenance,
            "origin": origin,
            "satisfied_by": {"SELLER_LOSS_INPUTS": "SELLER_INPUT", "SCENARIO": "SCENARIO"}.get(
                origin, "REAL_DATA" if provenance in REAL else "CONFIG" if provenance == "CONFIG" else "UPLOAD_OR_DERIVED"),
            "source": states[name]["source"], "used_in_comparison": name in used,
            "audit": [{k: a[k] for k in ("row", "scope", "column", "outcome", "input_value")} for a in applied_audit if a["field"] == name],
        })
    auto = []
    for name in (n for n in ENGINE_CHECK_ORDER if n != "quantity_unit"):
        if name == "target_store_id":
            if base.target_store_id:
                auto.append({"field": name, "label": FIELD_LABELS[name], "value": base.target_store_id, "provenance": "RECOMMENDATION",
                             "status": "AUTO_DERIVED_UPLOAD", "status_label": "추천에서 결정", "derived_from": PRODUCTION_SOURCES[name]})
            continue
        item = getattr(base, name)
        if not item.present:
            continue
        code, label = AUTO_STATUS.get(item.provenance, (item.provenance, item.provenance))
        auto.append({"field": name, "label": FIELD_LABELS.get(name, name), "value": item.value, "provenance": item.provenance,
                     "status": code, "status_label": label, "derived_from": PRODUCTION_SOURCES.get(name),
                     "replaced_by_seller_input": _seller_origin(name, merged, fields_info) and fields_info.get(name, {}).get("outcome") in APPLIED_OUTCOMES,
                     "usable_in_strict_comparison": item.usable})

    not_needed = [{"field": "unit_cost", "label": FIELD_LABELS["unit_cost"],
                   "reason": "매입원가는 세 전략에서 같아(매몰원가) 손실 비교 순위에 쓰이지 않습니다. 입력하지 않아도 됩니다."}]
    if TRANSFER in scope_strategies and _trace()["matrix"]["target_disposal_cost_per_unit"][TRANSFER]["class"] == NOT_USED:
        not_needed.append({"field": "target_disposal_cost_per_unit", "label": FIELD_LABELS["target_disposal_cost_per_unit"],
                           "reason": "이동 수량은 도착 점포가 남은 기간 안에 팔 수 있는 수량까지만 계산되어 도착 점포 폐기 처리비는 "
                                     "손실에 들어가지 않습니다."})
    for name in ("source_surplus_cap", "target_need_cap", "route_capacity_qty"):
        if TRANSFER in scope_strategies:
            not_needed.append({"field": name, "label": FIELD_LABELS[name],
                               "reason": "production이 계산하는 이동 한도(선택 제약)라 판매자에게 묻지 않습니다. 대리지표면 적용하지 않습니다."})

    conflict_rows = [{"field": c["field"], "label": FIELD_LABELS.get(c["field"], c["field"]), "kind": c["kind"],
                      "blocking": c["kind"] in SAME_SCOPE_CONFLICTS, "seller_values": c.get("seller_values"),
                      "rows": c.get("rows"), "kept": c.get("kept"),
                      "effect": ("같은 범위 충돌: 값을 쓰지 않으며 추천 가능(RECOMMENDABLE) 불가" if c["kind"] in SAME_SCOPE_CONFLICTS
                                 else "데이터 값을 유지하고 판매자 값은 기록만 함(차단 아님)")} for c in conflicts]
    entries = {i["seller_entry"]["entry_key"] for i in required}
    columns = {i["seller_entry"]["column"] for i in required}
    structural = [i for i in required if i["tier"] == 1]
    comparable_in_scope = [s for s in scope_strategies if s in comparable]
    scope_status = SCOPE_COMPLETE if scope_complete else SCOPE_PARTIAL if len(comparable_in_scope) >= 2 else SCOPE_UNAVAILABLE
    winner = decision.get("recommended_strategy")
    plan = {
        "planner_version": PLANNER_VERSION,
        "comparison_scope": scope,
        "scope_strategies": list(scope_strategies),
        "out_of_scope_strategies": [s for s in STRATEGIES if s not in scope_strategies],
        "decision_mode": mode,
        "decision": {**_decision_key(merged), "decision_qty": decision.get("decision_qty"),
                     "quantity_unit": merged.quantity_unit, "currency": decision.get("currency")},
        "planning_status": status,
        "stop_asking": stop_asking,
        "scope_complete": scope_complete,
        "scope_recommendable": scope_complete and readiness == sli.RECOMMENDABLE,
        "can_unblock_with_seller_inputs": bool(required),
        "comparison_status_if_run_now": {
            "engine_comparison_status": decision.get("comparison_status"),
            "comparable_strategies": comparable, "comparable_in_scope": comparable_in_scope, "scope_status": scope_status,
            "recommendation_status": decision.get("recommendation_status"), "recommendation_readiness": readiness,
            "readiness_reasons": list(decision.get("readiness_reasons") or []), "evidence_level": decision.get("evidence_level"),
            "recommended_strategy": winner,
            "recommended_strategy_in_scope": (winner in scope_strategies) if winner else None,
            "message": _status_message(comparable, scope_strategies),
        },
        "ready_strategies": [s for s in scope_strategies if s in comparable],
        "unavailable_strategies": {s: list(decision["strategies"][s]["unavailable_reasons"]) for s in scope_strategies if s not in comparable},
        "strategy_requirements": strategy_requirements,
        "already_available_fields": available_fields,
        "auto_derived_fields": auto,
        "required_user_inputs": required,
        "proxy_replacement_inputs": [i for i in required if i["request_kind"] == REPLACE_PROXY],
        "optional_user_inputs": optional_items,
        "not_needed_fields": not_needed,
        "not_requested_fields": not_requested,
        "conflicting_fields": conflict_rows,
        "blocked_fields": blocked_fields,
        "value_outcomes": value_outcomes,
        "ranking_dependency": dependency,
        "recommended_next_inputs": recommended,
        "ordering_rule": ORDERING_RULE,
        "stop_asking_rule": STOP_ASKING_RULE,
        "required_input_count": len(required),
        "structural_input_count": len(structural),
        "robustness_input_count": len(required) - len(structural),
        "required_entry_count": len(entries),
        "required_column_count": len(columns),
        "pending_conditional_count": sum(1 for i in optional_items if i["request_kind"] == PENDING_EVALUATION),
        "input_count_bucket": input_count_bucket(len(entries)),
        "input_bundles": _bundles(required, optional_items),
        "production_promotion_candidate": decision.get("production_promotion_candidate"),
        "production_action_applied": False,
        "legacy_action_changed": False,
    }
    plan["message_ko"] = _message(plan)
    return plan


def _status_message(comparable: Sequence[str], scope_strategies: Sequence[str]) -> str:
    if len(comparable) < 2:
        return "현재 금액 비교 불가(비교 가능한 전략 2개 미만)"
    in_scope = [s for s in scope_strategies if s in comparable]
    text = f"현재 {len(comparable)}개 전략 비교 가능({' · '.join(STRATEGY_LABELS[s] for s in comparable)})"
    if len(in_scope) < len(scope_strategies):
        text += f"; 요청 범위 중 {len(in_scope)}개만 비교 가능"
    return text


def input_count_bucket(count: int) -> str:
    return "0" if count == 0 else "1-3" if count <= 3 else "4-5" if count <= 5 else "6+"


def _bundles(required: Sequence[Mapping[str, Any]], optional: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key, label, columns in INPUT_BUNDLES:
        req = [i["field"] for i in required if i["seller_entry"]["column"] in columns]
        opt = [i["field"] for i in optional if i["seller_entry"]["column"] in columns]
        if req or opt:
            out.append({"bundle": key, "label": label, "columns": list(columns), "required_fields": req, "optional_fields": opt,
                        "note": "묶음은 입력 화면 구성용이며 값을 만들거나 채우지 않습니다"})
    return out


def _grouped_labels(items: Sequence[Mapping[str, Any]]) -> list[str]:
    groups: dict[str, list[str]] = {}
    for item in items:
        entry = item["seller_entry"]
        label = entry["column_label"]
        roles = groups.setdefault(label, [])
        if entry["store_role"] is not None and entry["store_role_label"] not in roles:
            roles.append(entry["store_role_label"])
    out = []
    for label, roles in groups.items():
        if len(roles) == 2:
            out.append(f"{label}(출발·도착 점포)")
        elif roles and label not in ("남은 판매 가능 기간(일)", "할인율", "할인 시 예상 판매 증가율", "개당 잔존가치(폐기 시 회수액)", "재고 수량 단위"):
            out.append(f"{label}({roles[0]})")
        else:
            out.append(label)
    return out


def _message(plan: Mapping[str, Any]) -> str:
    """One deterministic Korean sentence for the seller (template text only)."""
    decision = plan["decision"]
    who = f"상품 {decision['product_id']}({decision['source_store_id']}" + (f"→{decision['target_store_id']})" if decision["target_store_id"] else ")")
    scope_text = _scope_text(plan["scope_strategies"])
    required = plan["required_user_inputs"]
    status = plan["planning_status"]
    labels = ", ".join(_grouped_labels(required))
    proxy = [i for i in required if i["request_kind"] == REPLACE_PROXY and i["tier"] == 1]
    proxy_note = (f" 이 중 {_josa(', '.join(_grouped_labels(proxy)), '은/는')} 현재 대리지표라 실제 값으로 바꿔야 합니다."
                  if proxy else "")
    optional = [i for i in plan["optional_user_inputs"] if i["request_kind"] == IMPROVE_PRECISION]
    if status in ("READY_RECOMMENDABLE", "READY_RECOMMENDABLE_SCOPE_LIMITED"):
        text = f"{who}: 추가 입력 없이 추천 가능합니다."
        if status == "READY_RECOMMENDABLE_SCOPE_LIMITED":
            text += f" ({plan['comparison_status_if_run_now']['message']}; 나머지 전략은 업로드 데이터·현재 값 때문에 비교할 수 없습니다.)"
        if optional:
            text += f" 선택 입력: {', '.join(_grouped_labels(optional))}(입력해도 추천은 바뀌지 않음)."
        return text
    if status == "SCENARIO_ONLY":
        return f"{who}: SCENARIO 실행 결과(가정값 포함)는 추천 대상이 아닙니다. 실제 추천은 ACTUAL_OPERATION에서 평가합니다."
    if status == "RECOMMENDABLE_PARTIAL_SCOPE":
        missing = [STRATEGY_LABELS[s] for s in plan["scope_strategies"] if s not in plan["ready_strategies"]]
        return (f"{who}: {plan['comparison_status_if_run_now']['message']}으로 추천 가능합니다. {', '.join(missing)}까지 "
                f"비교하려면 {_josa(labels, '을/를')} 입력하세요.{proxy_note} 지금 비교 범위로 충분하면 comparison scope를 "
                f"좁히면 추가 입력이 필요 없습니다.")
    if status == "NEEDS_INPUT":
        return f"{who}: {_josa(labels, '을/를')} 입력하면 {scope_text} 비교가 가능합니다.{proxy_note} 이미 있는 값은 다시 묻지 않습니다."
    if status == "NEEDS_INPUT_FOR_ROBUST_RECOMMENDATION":
        parts = []
        unknown = [i for i in required if i["request_kind"] in (RESOLVE_UNKNOWN, REPLACE_PROXY)]
        conflict = [i for i in required if i["request_kind"] == RESOLVE_CONFLICT]
        scenario = [i for i in required if i["request_kind"] == REPLACE_SCENARIO_VALUE]
        if unknown:
            parts.append(f"{', '.join(_grouped_labels(unknown))} 값에 따라 손실이 가장 작은 방법이 달라집니다.")
        if conflict:
            parts.append(f"{', '.join(_grouped_labels(conflict))}의 같은 범위 충돌을 정리해야 합니다.")
        if scenario:
            parts.append(f"{_josa(', '.join(_grouped_labels(scenario)), '은/는')} 가정·설정값이라 실제 값으로 바꿔야 합니다.")
        return f"{who}: {scope_text} 비교는 가능하지만 " + " ".join(parts) + " 입력하면 추천을 확정할 수 있습니다."
    if status == "NOT_ROBUST_DATA_DEPENDENT":
        return f"{who}: 추천이 실제 데이터의 미상 값에 따라 달라지며 판매자 입력으로는 확정할 수 없습니다."
    if status == "BLOCKED_BY_DATA":
        blocked = ", ".join(dict.fromkeys(b["label"] for b in plan["blocked_fields"] if b["kind"] == "UPLOAD_DATA_MISSING")) or "업로드 데이터 정합성"
        return f"{who}: 업로드 데이터의 {blocked} 문제로 판매자 입력만으로는 {scope_text} 비교를 할 수 없습니다."
    reasons = "; ".join(dict.fromkeys(v["message"] for v in plan["value_outcomes"])) or "현재 값"
    return f"{who}: 필요한 값은 있지만 현재 값으로는 {scope_text} 비교를 할 수 없습니다({reasons})."


# ---------------------------------------------------------------- entry points


def plan_for_decision(base: SellerDecisionInput, table: sli.SellerInputTable | None, decision: Mapping[str, Any], *,
                      comparison_scope: str | None = None, decision_date: str | None = None,
                      dataset: str | None = WORKBOOK_DATASET, upload_currency: str | None = None) -> dict[str, Any]:
    """Plan for a decision that evaluate_with_seller_inputs already produced (the merge is repeated, deterministic)."""
    merged, record = sli.merge_seller_inputs(base, table, decision_mode=decision.get("decision_mode", sli.ACTUAL_OPERATION),
                                             decision_date=decision_date, dataset=dataset)
    plan = build_plan(base, merged, record, decision, comparison_scope=comparison_scope, upload_currency=upload_currency)
    from services.seller_business_profile import enrich_plan
    return enrich_plan(plan, merged, record, table)


def plan_input_requirements(base: SellerDecisionInput, table: sli.SellerInputTable | None = None, *,
                            comparison_scope: str | None = None, decision_mode: str = sli.ACTUAL_OPERATION,
                            decision_date: str | None = None, dataset: str | None = WORKBOOK_DATASET,
                            upload_currency: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(engine decision with evidence, plan): merge + evaluate exactly like production, then plan."""
    decision = sli.evaluate_with_seller_inputs(base, table, decision_mode=decision_mode, decision_date=decision_date,
                                               dataset=dataset)
    return decision, plan_for_decision(base, table, decision, comparison_scope=comparison_scope, decision_date=decision_date,
                                       dataset=dataset, upload_currency=upload_currency)


# ---------------------------------------------------------------- consistency and unblock verification


def _explained(code: str, requirement: Mapping[str, Any]) -> bool:
    head, _, name = code.partition(":")
    if head in MISSING_CODE_FIELDS:
        return MISSING_CODE_FIELDS[head] in requirement["unmet_fields"]
    if head == "PROXY_REJECTED":
        return requirement["unmet_states"].get(name) == PROXY
    if head in CONSISTENCY_HEADS:
        return code in {b["code"] for b in requirement["blockers"]}
    if head in VALUE_CODE_FIELDS:
        return code in {v["code"] for v in requirement["value_blockers"]}
    return False


def check_plan_consistency(plan: Mapping[str, Any], decision: Mapping[str, Any]) -> list[str]:
    """Contradictions between the plan and the engine/evidence result (an empty list is the contract)."""
    problems: list[str] = []
    for strategy in STRATEGIES:
        req = plan["strategy_requirements"][strategy]
        unmet = sorted(set(req["unmet_fields"]) - {"decision_qty"})
        evidence = sorted(decision["missing_required_by_strategy"][strategy])
        if unmet != evidence:
            problems.append(f"{strategy}:UNMET_FIELDS_DIFFER:{unmet}!={evidence}")
        available = bool(decision["strategies"][strategy]["available"])
        if available != (decision["strategy_readiness"][strategy] == "READY"):
            problems.append(f"{strategy}:ENGINE_READINESS_INCONSISTENT")
        clear = not req["unmet_fields"] and not req["blockers"] and not req["value_blockers"]
        if available and not clear:
            problems.append(f"{strategy}:ENGINE_READY_BUT_PLANNER_BLOCKED")
        if not available and clear:
            problems.append(f"{strategy}:ENGINE_BLOCKED_BUT_PLANNER_CLEAR")
        for code in decision["strategies"][strategy]["unavailable_reasons"]:
            if not _explained(code, req):
                problems.append(f"{strategy}:UNEXPLAINED_ENGINE_REASON:{code}")
    excluded = sorted(decision["missing_for_excluded_strategies"])
    if excluded != sorted(s for s in STRATEGIES if s not in decision["comparable_strategies"]):
        problems.append("MISSING_FOR_EXCLUDED_STRATEGIES_DIFFER")
    available_names = {a["field"] for a in plan["already_available_fields"]}
    scope = set(plan["scope_strategies"])
    for item in plan["required_user_inputs"]:
        kind = item["request_kind"]
        if kind in (ENTER_MISSING, REPLACE_PROXY) and item["field"] in available_names:
            problems.append(f"AVAILABLE_FIELD_REQUESTED:{item['field']}")
        if item["tier"] == 1 and not set(item["unblocks_strategies"]) & scope:
            problems.append(f"REQUEST_WITHOUT_EFFECT:{item['field']}")
        if item["tier"] == 2 and not item["unblocks_recommendation"]:
            problems.append(f"REQUEST_WITHOUT_EFFECT:{item['field']}")
    if plan["stop_asking"] and (plan["required_user_inputs"] or decision["recommendation_readiness"] != sli.RECOMMENDABLE):
        problems.append("STOP_ASKING_INCONSISTENT")
    if not plan["stop_asking"] and decision["recommendation_readiness"] == sli.RECOMMENDABLE and not any(
            i["tier"] == 1 for i in plan["required_user_inputs"]):
        problems.append("SHOULD_STOP_ASKING")
    if plan["comparison_status_if_run_now"]["engine_comparison_status"] != decision["comparison_status"]:
        problems.append("COMPARISON_STATUS_DIFFERS")
    if plan["production_action_applied"] or decision.get("production_action_applied") or decision.get("legacy_action_changed"):
        problems.append("PRODUCTION_ACTION_CHANGED")
    return problems


def _engine_codes(decision: Mapping[str, Any]) -> set[str]:
    codes = set(decision.get("reason_codes") or [])
    for reasons in (decision.get("unavailable_strategies") or {}).values():
        codes.update(reasons)
    return codes


def field_reason_codes(name: str) -> set[str]:
    codes = {f"PROXY_REJECTED:{name}", f"UNKNOWN_RATE:{name}"}
    if name in FIELD_MISSING_CODE:
        codes.add(FIELD_MISSING_CODE[name])
    return codes


def verify_unblock_step(before_plan: Mapping[str, Any], before_decision: Mapping[str, Any], after_plan: Mapping[str, Any],
                        after_decision: Mapping[str, Any], added_fields: Iterable[str]) -> dict[str, Any]:
    """After the seller entered ``added_fields`` (as requested): each must stop blocking; a request with no effect is a bug."""
    added = list(dict.fromkeys(added_fields))
    problems: list[str] = []
    before = {i["field"]: i for i in before_plan["required_user_inputs"]}
    after_required = {i["field"] for i in after_plan["required_user_inputs"]}
    after_available = {a["field"] for a in after_plan["already_available_fields"]}
    after_codes = _engine_codes(after_decision)
    for name in added:
        if name not in before:
            problems.append(f"NOT_REQUESTED_BEFORE:{name}")
            continue
        if name in after_required:
            problems.append(f"STILL_REQUESTED:{name}")
        if name != "quantity_unit" and name not in after_available and before[name]["request_kind"] not in (
                CORRECT_CURRENCY, CORRECT_UNIT):
            problems.append(f"NOT_AVAILABLE_AFTER_INPUT:{name}")
        lingering = sorted(field_reason_codes(name) & after_codes)
        if lingering:
            problems.append(f"ENGINE_STILL_REPORTS:{name}:{'|'.join(lingering)}")
        for strategy in before[name]["unblocks_strategies"]:
            if name in after_plan["strategy_requirements"][strategy]["unmet_fields"]:
                problems.append(f"STILL_UNMET:{strategy}:{name}")
    before_struct = {i["field"] for i in before_plan["required_user_inputs"] if i["tier"] == 1}
    after_struct = {i["field"] for i in after_plan["required_user_inputs"] if i["tier"] == 1}
    new_struct = sorted(after_struct - before_struct)
    if new_struct:
        problems.append(f"NEW_STRUCTURAL_REQUIREMENT:{'|'.join(new_struct)}")
    if added and _fingerprint(before_plan, before_decision) == _fingerprint(after_plan, after_decision):
        problems.append("NO_EFFECT")
    removed = sorted(_engine_codes(before_decision) - after_codes)
    return {"added_fields": added, "removed_engine_reasons": removed,
            "revealed_engine_reasons": sorted(after_codes - _engine_codes(before_decision)),
            "required_before": sorted(before), "required_after": sorted(after_required),
            "status_before": before_plan["planning_status"], "status_after": after_plan["planning_status"],
            "problems": problems, "ok": not problems}


def _fingerprint(plan: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    payload = {"required": [(i["field"], i["request_kind"]) for i in plan["required_user_inputs"]],
               "available": sorted(a["field"] for a in plan["already_available_fields"]),
               "codes": sorted(_engine_codes(decision)), "unknown": decision.get("unknown_inputs"),
               "losses": [decision.get(k) for k in ("expected_loss_transfer", "expected_loss_normal_sale", "expected_loss_discount_sale")]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- analysis-level summary


def summarize_plans(plans: Sequence[Mapping[str, Any]], comparison_scope: str | None = None) -> dict[str, Any]:
    """What the seller has to enter across all decisions of one analysis (deduplicated seller entries)."""
    valid = [p for p in plans if p and p.get("planner_version")]
    entries: dict[str, dict[str, Any]] = {}
    for plan in valid:
        for item in plan["required_user_inputs"]:
            entry = item["seller_entry"]
            row = entries.setdefault(entry["entry_key"], {
                "entry_key": entry["entry_key"], "column": entry["column"], "column_label": entry["column_label"],
                "suggested_scope": entry["suggested_scope"], "keys": entry["keys"], "request_kinds": [], "fields": [],
                "decisions": [], "best_priority": item["priority"], "tier": item["tier"]})
            for key, value in (("request_kinds", item["request_kind"]), ("fields", item["field"]),
                               ("decisions", plan["decision"]["decision_id"])):
                if value not in row[key]:
                    row[key].append(value)
            row["best_priority"] = min(row["best_priority"], item["priority"])
            row["tier"] = min(row["tier"], item["tier"])
    ordered = sorted(entries.values(), key=lambda r: (r["tier"], -len(r["decisions"]), r["best_priority"],
                                                      ORDER_INDEX.get(r["fields"][0], 99), r["entry_key"]))
    field_counts = Counter(i["field"] for p in valid for i in p["required_user_inputs"])
    status_counts = Counter(p["planning_status"] for p in valid)
    count_dist = Counter(p["required_entry_count"] for p in valid)
    columns = Counter(i["seller_entry"]["column"] for p in valid for i in p["required_user_inputs"])
    need = [p for p in valid if p["required_user_inputs"]]
    if not valid:
        message = "Seller Loss 결정이 없습니다."
    elif not need:
        message = f"{len(valid)}개 결정 모두 추가 판매자 입력이 필요하지 않습니다."
    else:
        top = ", ".join(COLUMN_LABELS[c] for c, _ in columns.most_common(5))
        message = (f"{len(valid)}개 결정 중 {len(need)}개는 판매자 입력이 필요합니다(고유 입력 {len(entries)}칸). "
                   f"가장 많이 필요한 입력: {top}.")
    return {
        "planner_version": PLANNER_VERSION, "comparison_scope": comparison_scope, "decision_count": len(valid),
        "planning_status_counts": dict(sorted(status_counts.items())),
        "stop_asking_count": sum(1 for p in valid if p["stop_asking"]),
        "decisions_needing_input": len(need),
        "required_field_counts": dict(sorted(field_counts.items(), key=lambda kv: (-kv[1], ORDER_INDEX.get(kv[0], 99)))),
        "required_entries_per_decision": {str(k): v for k, v in sorted(count_dist.items())},
        "input_count_bucket_counts": dict(sorted(Counter(p["input_count_bucket"] for p in valid).items())),
        "distinct_required_entries": len(entries),
        "required_entries": ordered,
        "message_ko": message,
        "production_action_applied": False,
    }


def safe_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """plan_for_decision, fault-isolated: a planner problem never stops the Seller Loss analysis."""
    try:
        return plan_for_decision(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive isolation
        return {"status": "error", "error_type": type(exc).__name__, "message": str(exc), "production_action_applied": False}


# ---------------------------------------------------------------- contract and minimal sample


def requirement_contract_document() -> dict[str, Any]:
    document = {
        "planner_version": PLANNER_VERSION,
        "comparison_scopes": {k: list(v) for k, v in COMPARISON_SCOPES.items()},
        "default_comparison_scope": DEFAULT_COMPARISON_SCOPE,
        "scope_resolution": f"argument > uploaded_data['{COMPARISON_SCOPE_KEY}'] > config row '{COMPARISON_SCOPE_KEY}' > {DEFAULT_COMPARISON_SCOPE}",
        "scope_effect": "The scope decides only what the seller is asked for; the engine comparison and ranking are unchanged.",
        "strategy_required_fields": {s: list(strategy_required_fields(s)) for s in STRATEGIES},
        "requirement_classes": [REQUIRED, CONDITIONAL, OPTIONAL, DERIVED],
        "request_kinds": [ENTER_MISSING, REPLACE_PROXY, RESOLVE_CONFLICT, CORRECT_CURRENCY, CORRECT_UNIT, DECLARE_UNIT,
                          CORRECT_VALUE, RESOLVE_UNKNOWN, REPLACE_SCENARIO_VALUE, IMPROVE_PRECISION, PENDING_EVALUATION],
        "tiers": {str(k): v for k, v in TIERS.items()},
        "planning_statuses": dict(PLANNING_STATUSES),
        "ordering_rule": ORDERING_RULE,
        "engine_check_order": list(ENGINE_CHECK_ORDER),
        "stop_asking_rule": STOP_ASKING_RULE,
        "minimal_input_rule": MINIMAL_INPUT_RULE,
        "robustness_rule": "Decisive inputs = unknown rates whose coefficient differs inside a pair of comparable strategies "
                           "that neither wins over the admissible range; knowing all of them always yields a robust winner.",
        "partial_policy": "The planner never promotes a result: production_action_applied is always False and the "
                          "readiness/promotion fields are copied from the evidence layer unchanged.",
        "no_value_rule": "A request carries a field, key, label and reason, never a number; nothing is defaulted.",
        "field_labels": dict(FIELD_LABELS), "column_labels": dict(COLUMN_LABELS),
        "bundles": {key: {"label": label, "columns": list(columns)} for key, label, columns in INPUT_BUNDLES},
    }
    document["contract_signature"] = hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return document


MINIMAL_SAMPLE_COLUMNS = ("scope", "product_id", "store_id", "normal_price", "remaining_shelf_life_days", "daily_demand",
                          "discount_rate", "promotion_uplift", "note")


def minimal_sample_frame() -> pd.DataFrame:
    """SAMPLE rows: the smallest seller sheet for a row that has real stock/route cost but no price, shelf life or real
    retail demand. Ids are placeholders and every number is an example, never a default."""
    rows = [
        {"scope": "PRODUCT", "product_id": "SAMPLE_PRODUCT_1", "normal_price": 4900, "remaining_shelf_life_days": 4,
         "note": "SAMPLE: 상품 정상 판매가격·남은 판매 가능 기간(점포 공통이면 PRODUCT 한 행). 예시 값이며 기본값이 아님"},
        {"scope": "PRODUCT_STORE", "product_id": "SAMPLE_PRODUCT_1", "store_id": "SAMPLE_STORE_A", "daily_demand": 12,
         "note": "SAMPLE: 출발 점포 정상가 기준 하루 판매량(출고량 대리지표를 실제 값으로 대체)"},
        {"scope": "PRODUCT_STORE", "product_id": "SAMPLE_PRODUCT_1", "store_id": "SAMPLE_STORE_B", "daily_demand": 30,
         "note": "SAMPLE: 도착 점포 정상가 기준 하루 판매량"},
        {"scope": "PRODUCT", "product_id": "SAMPLE_PRODUCT_1", "discount_rate": 0.2, "promotion_uplift": 0.5,
         "note": "SAMPLE: 할인 판매까지 비교할 때만(0.2=20% 할인, 0.5=할인 중 판매량 +50%)"},
    ]
    return pd.DataFrame(rows, columns=list(MINIMAL_SAMPLE_COLUMNS))
