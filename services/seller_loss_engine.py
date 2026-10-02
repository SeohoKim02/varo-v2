"""Seller Loss Decision Engine: which way of handling a store's remaining stock costs the seller the least money.

Exactly three strategies are compared, always for the SAME decision quantity Q of one product at one source store:

    TRANSFER       move q <= Q units to a target store (shared quantity constraints), keep Q - q at the normal price
    NORMAL_SALE    keep all Q units at the source and keep selling them at the normal price
    DISCOUNT_SALE  keep all Q units at the source and sell them at a markdown

Criterion (the only one used): expected avoidable seller loss, minimised.

    L(s) = Q x p_src - V(s)

    V(s) is the cash the seller is expected to recover because of the Q decision units under strategy s: sales revenue
    minus the strategy's own cash costs (transport, holding, disposal handling) plus salvage, measured as the change of
    the store-level outcome the Q units cause (the store's other units are a common reference for all strategies).
    Q x p_src (every decision unit sold at the source's normal price, at no extra cost) is the same constant for the
    three strategies, so minimising L is identical to maximising V. The engine reports L because "예상 손실" is what a
    seller reads; it never mixes L with a separate net-value objective.

Accounting rules (no double counting):
    * The purchase cost of the Q units (unit_cost) is sunk: it is identical in all three strategies, so it never enters
      L. It is kept as provenance only.
    * An unsold unit costs its forgone normal-price revenue once (opportunity_loss). Disposal adds only the extra cash
      paid to dispose (disposal_cost_per_unit); the goods' value is not counted again as a "disposal loss".
    * The source opportunity loss of a transfer (source sales that disappear because units left) is part of
      V(TRANSFER) - V(NORMAL_SALE); it is reported in a bridge, never added on top of the components.
    * The markdown is charged on every discounted unit sold, including regular-price sales it displaces
      (cannibalised base stock), and the displaced units' fate is valued once in opportunity_loss.

Demand model (deterministic fluid model on the point forecast; demand variability is not modelled because no
distribution is supplied, so "expected" means "on the expected demand rate"):
    * All units of the product at a store share the store row's remaining shelf life H (the data grain). Units unsold
      after H are expired: they lose their value and incur disposal handling, and may return salvage.
    * NORMAL_SALE: the store sells min(stock, r x H) at rate r.
    * DISCOUNT_SALE: the Q discounted units are taken first at rate r x (1 + uplift); the store's other units sell at
      rate r afterwards.
    * TRANSFER: the target sells the moved lot after its own stock at rate r_t within H - transit; the source keeps
      selling its remaining stock at rate r.

Strict monetary mode: a required value that is missing, a proxy, in another dataset, unit or currency never gets a
default. A strategy whose required inputs are incomplete is unavailable; with fewer than two comparable strategies
there is no comparison. Holding, disposal handling and salvage rates that are missing are UNKNOWN (never 0): the loss
is reported without them and a strategy is recommended only if it stays the cheapest for every admissible value
(holding >= 0, disposal >= 0, 0 <= salvage <= normal price).

The engine runs next to the existing Varo Final pipeline and never changes its actions or ranks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

ENGINE_VERSION = "seller-loss-1.0.0"
CRITERION = "EXPECTED_AVOIDABLE_LOSS_MIN"
LOSS_REFERENCE = "decision_qty x source normal price (full normal-price sell-through, no extra cost)"

TRANSFER, NORMAL_SALE, DISCOUNT_SALE = "TRANSFER", "NORMAL_SALE", "DISCOUNT_SALE"
STRATEGIES = (TRANSFER, NORMAL_SALE, DISCOUNT_SALE)
# Equal expected loss -> the option with the least operational action wins (keep < relabel < truck).
TIE_BREAK_ORDER = (NORMAL_SALE, DISCOUNT_SALE, TRANSFER)
STRATEGY_LABELS = {TRANSFER: "재고 이동", NORMAL_SALE: "정상 판매 유지", DISCOUNT_SALE: "할인 판매"}
SELLER_LOSS_ACTIONS = {TRANSFER: "재고 이동", NORMAL_SALE: "정상 판매 유지", DISCOUNT_SALE: "할인"}

STATUS_FULL, STATUS_PARTIAL, STATUS_UNAVAILABLE = "FULL", "PARTIAL", "COMPARISON_UNAVAILABLE"
RECOMMENDED, NOT_ROBUST, NO_COMPARISON = "RECOMMENDED", "NOT_ROBUST_TO_UNKNOWN_INPUTS", "NO_COMPARISON"

PROVENANCE_CLASSES = ("DIRECT_REAL", "DERIVED_REAL", "USER_INPUT", "DERIVED_FROM_USER_INPUT", "CONFIG", "PROXY", "MISSING")
ACCEPTED_PROVENANCE = frozenset({"DIRECT_REAL", "DERIVED_REAL", "USER_INPUT", "DERIVED_FROM_USER_INPUT", "CONFIG"})
# Explicit scenario values: allowed, never presented as observations.
SCENARIO_PROVENANCE = frozenset({"CONFIG"})
TRANSFER_COST_BASES = ("QUANTITY_SPECIFIC", "FIXED_PER_TRIP", "PER_UNIT")
NON_PHYSICAL_UNITS = frozenset({"normalized_sales_amount"})

COMPONENTS = ("transfer_cost", "discount_loss", "price_difference_loss", "opportunity_loss",
              "holding_loss", "disposal_loss", "salvage_recovery")
COMPONENT_LABELS = {
    "transfer_cost": "이동 비용", "discount_loss": "할인 손실", "price_difference_loss": "도착 점포 가격 차이 손실",
    "opportunity_loss": "미판매 재고 손실(정상가 기준)", "holding_loss": "보관 비용", "disposal_loss": "폐기 처리 비용",
    "salvage_recovery": "잔존가치 회수",
}
# Unknown non-negative rates and their admissible range; None = the source normal price.
UNKNOWN_RATE_PARAMS = {
    "source_holding_cost_per_unit_day": (0.0, math.inf),
    "target_holding_cost_per_unit_day": (0.0, math.inf),
    "source_disposal_cost_per_unit": (0.0, math.inf),
    "target_disposal_cost_per_unit": (0.0, math.inf),
    "salvage_value_per_unit": (0.0, None),
}
PARAM_LABELS = {
    "source_holding_cost_per_unit_day": "출발 점포 보관비", "target_holding_cost_per_unit_day": "도착 점포 보관비",
    "source_disposal_cost_per_unit": "출발 점포 폐기 처리비", "target_disposal_cost_per_unit": "도착 점포 폐기 처리비",
    "salvage_value_per_unit": "잔존가치", "transfer_cost": "실행 수량 기준 이동 비용",
}
TOL = 1e-6

# Fields whose values describe the product's economics at a store; they must come from one dataset.
PRODUCT_ECONOMIC_FIELDS = (
    "decision_qty", "source_current_stock", "source_daily_demand", "remaining_shelf_life_days", "source_normal_price",
    "unit_cost", "source_holding_cost_per_unit_day", "source_disposal_cost_per_unit", "salvage_value_per_unit",
    "target_current_stock", "target_daily_demand", "target_normal_price", "target_holding_cost_per_unit_day",
    "target_disposal_cost_per_unit", "promotion_uplift",
)
QUANTITY_FIELDS = ("decision_qty", "source_current_stock", "source_daily_demand", "target_current_stock",
                   "target_daily_demand", "source_surplus_cap", "target_need_cap", "route_capacity_qty")
PER_UNIT_MONEY_FIELDS = ("source_normal_price", "target_normal_price", "unit_cost", "source_holding_cost_per_unit_day",
                         "target_holding_cost_per_unit_day", "source_disposal_cost_per_unit",
                         "target_disposal_cost_per_unit", "salvage_value_per_unit")
MONEY_FIELDS = (*PER_UNIT_MONEY_FIELDS, "transfer_cost")
TRANSFER_ONLY_FIELDS = ("target_current_stock", "target_daily_demand", "target_normal_price",
                        "target_holding_cost_per_unit_day", "target_disposal_cost_per_unit", "transfer_cost",
                        "transfer_cost_qty", "transit_time_days", "source_surplus_cap", "target_need_cap",
                        "route_capacity_qty")
# Validated by their own strategy rules (0 < d < 1, uplift >= 0) instead of the generic sign check.
SIGN_FREE_FIELDS = ("discount_rate", "promotion_uplift")
COMPLETENESS_FIELDS = (
    "decision_qty", "source_current_stock", "source_daily_demand", "remaining_shelf_life_days", "source_normal_price",
    "source_holding_cost_per_unit_day", "source_disposal_cost_per_unit", "salvage_value_per_unit", "discount_rate",
    "promotion_uplift", "target_current_stock", "target_daily_demand", "target_normal_price",
    "target_holding_cost_per_unit_day", "target_disposal_cost_per_unit", "transfer_cost", "transit_time_days",
)

REASON_MESSAGES = {
    "DECISION_QTY_MISSING": "결정 수량이 없습니다",
    "DECISION_QTY_NOT_POSITIVE": "결정 수량이 0 이하입니다",
    "ZERO_INVENTORY": "현재 재고가 0이라 처리할 재고가 없습니다",
    "DECISION_QTY_EXCEEDS_STOCK": "결정 수량이 현재 재고보다 많습니다",
    "SOURCE_STOCK_MISSING": "출발 점포 현재 재고가 없습니다",
    "SOURCE_DEMAND_MISSING": "출발 점포 예상 수요가 없습니다",
    "SHELF_LIFE_MISSING": "잔여 유통기한이 없어 미판매 재고의 손실 시점을 알 수 없습니다",
    "PRICE_MISSING": "정상 판매가격이 없습니다",
    "UNIT_MISMATCH": "수량 단위가 서로 다릅니다",
    "NON_PHYSICAL_UNIT": "수량이 정규화 값이라 금액으로 환산할 수 없습니다",
    "UNIT_UNVERIFIABLE_ACROSS_SOURCES": "단위가 명시되지 않은 수량이 서로 다른 출처에서 왔습니다",
    "CURRENCY_MISMATCH": "통화가 서로 다릅니다",
    "CURRENCY_UNDECLARED": "금액의 통화가 명시되지 않았습니다",
    "CROSS_DATASET_INPUT": "서로 다른 데이터셋의 가격·재고·수요를 결합할 수 없습니다",
    "DISCOUNT_RATE_MISSING": "할인율이 없습니다",
    "INVALID_DISCOUNT_RATE": "할인율이 0 초과 1 미만이 아닙니다",
    "UPLIFT_MISSING": "할인 후 판매량 증가율의 근거(관측·사용자 입력·명시 설정)가 없어 할인 판매량을 알 수 없습니다",
    "INVALID_UPLIFT": "할인 후 판매량 증가율이 음수입니다",
    "TARGET_STORE_MISSING": "도착 점포가 지정되지 않았습니다",
    "TARGET_STOCK_MISSING": "도착 점포 현재 재고가 없습니다",
    "TARGET_DEMAND_MISSING": "도착 점포 예상 수요가 없습니다",
    "TARGET_PRICE_MISSING": "도착 점포 판매가격이 없습니다",
    "TRANSFER_COST_MISSING": "이동 비용이 없습니다",
    "INVALID_TRANSFER_COST_BASIS": "이동 비용 기준을 알 수 없습니다",
    "TRANSFER_COST_QTY_MISMATCH": "실행 가능한 이동 수량에 대한 이동 비용을 알 수 없습니다",
    "TRANSIT_TIME_MISSING": "이동 소요시간이 없어 도착 점포 판매 기간을 알 수 없습니다",
    "NO_EXECUTABLE_TRANSFER_QTY": "재고·수요·용량 제약 안에서 이동 가능한 수량이 없습니다",
}


# ---------------------------------------------------------------- contract


@dataclass(frozen=True)
class InputField:
    """One decision input with its evidence. ``value`` None means MISSING; a value is never invented."""

    value: float | None = None
    provenance: str = "MISSING"
    source: str = ""
    unit: str | None = None       # quantity unit (quantity fields) or the per-unit denominator (money per unit)
    currency: str | None = None   # money fields only
    dataset: str | None = None    # dataset/upload the value comes from; None for explicit scenario values
    note: str = ""

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_CLASSES:
            raise ValueError(f"unknown provenance class: {self.provenance}")
        if self.value is None and self.provenance != "MISSING":
            object.__setattr__(self, "provenance", "MISSING")

    @property
    def present(self) -> bool:
        return _finite(self.value) is not None

    @property
    def usable(self) -> bool:
        return self.present and self.provenance in ACCEPTED_PROVENANCE

    def as_dict(self) -> dict[str, Any]:
        return {"value": _finite(self.value), "provenance": self.provenance, "source": self.source, "unit": self.unit,
                "currency": self.currency, "dataset": self.dataset, "note": self.note}


MISSING = InputField()


def known(value: Any, provenance: str, source: str = "", *, unit: str | None = None, currency: str | None = None,
          dataset: str | None = None, note: str = "") -> InputField:
    """Build an InputField; a non-finite or empty value stays MISSING with its attempted source recorded."""
    number = _finite(value)
    if number is None:
        return InputField(None, "MISSING", source, unit, currency, dataset, note)
    return InputField(number, provenance, source, unit, currency, dataset, note)


@dataclass(frozen=True)
class SellerDecisionInput:
    """Everything the engine may use for one decision; each economic value carries value + provenance + status."""

    decision_id: str = ""
    product_id: str = ""
    source_store_id: str = ""
    target_store_id: str | None = None
    quantity_unit: str | None = None          # declared unit of every quantity (None = undeclared)
    legacy_action: str | None = None          # existing Varo Final action, recorded and never changed

    decision_qty: InputField = MISSING
    source_current_stock: InputField = MISSING
    source_daily_demand: InputField = MISSING       # expected units/day at the normal price
    remaining_shelf_life_days: InputField = MISSING
    source_normal_price: InputField = MISSING
    unit_cost: InputField = MISSING                 # sunk purchase cost: provenance only, never in the ranking
    source_holding_cost_per_unit_day: InputField = MISSING
    source_disposal_cost_per_unit: InputField = MISSING
    salvage_value_per_unit: InputField = MISSING
    discount_rate: InputField = MISSING             # fraction off the normal price, 0 < d < 1
    promotion_uplift: InputField = MISSING          # relative demand increase while discounted, >= 0

    target_current_stock: InputField = MISSING
    target_daily_demand: InputField = MISSING
    target_normal_price: InputField = MISSING
    target_holding_cost_per_unit_day: InputField = MISSING
    target_disposal_cost_per_unit: InputField = MISSING
    transfer_cost: InputField = MISSING
    transfer_cost_basis: str = "QUANTITY_SPECIFIC"
    transfer_cost_qty: InputField = MISSING         # the quantity a QUANTITY_SPECIFIC cost was priced for
    transit_time_days: InputField = MISSING

    source_surplus_cap: InputField = MISSING        # existing shared constraint (movable source stock)
    target_need_cap: InputField = MISSING           # existing shared constraint (target shortage limit)
    route_capacity_qty: InputField = MISSING
    source_surplus_basis: str = ""
    target_need_basis: str = ""

    def input_fields(self) -> dict[str, InputField]:
        return {f.name: getattr(self, f.name) for f in fields(self) if isinstance(getattr(self, f.name), InputField)}


def contract_document() -> dict[str, Any]:
    """Machine-readable contract (single source of truth for docs and the validation bundle)."""
    input_fields = [f.name for f in fields(SellerDecisionInput) if f.type in ("InputField", InputField)]
    document = {
        "engine_version": ENGINE_VERSION,
        "criterion": CRITERION,
        "loss_definition": "L(s) = decision_qty x source_normal_price - V(s); V(s) = expected cash recovered from the "
                           "decision units under s = revenue - strategy cash costs + salvage (store-level change "
                           "caused by the decision units). Minimising L == maximising V.",
        "loss_reference": LOSS_REFERENCE,
        "strategies": {s: STRATEGY_LABELS[s] for s in STRATEGIES},
        "same_quantity_rule": "All strategies are evaluated for the same decision_qty Q; TRANSFER moves q <= Q and keeps Q - q under NORMAL_SALE.",
        "input_fields": input_fields,
        "field_shape": ["value", "provenance", "source", "unit", "currency", "dataset", "note"],
        "provenance_classes": list(PROVENANCE_CLASSES),
        "accepted_in_strict_mode": sorted(ACCEPTED_PROVENANCE),
        "scenario_provenance": sorted(SCENARIO_PROVENANCE),
        "components": {c: COMPONENT_LABELS[c] for c in COMPONENTS},
        "formulas": {
            "common": "B = S - Q; ref_sold = min(B, r H); ref_unit_days = UD(B, r, H); UD(x, r, H) = integral_0^H max(0, x - r t) dt",
            NORMAL_SALE: "sold = min(S, rH) - ref_sold; unsold = Q - sold; L = p unsold + h_s UD_N + (c_d - s) unsold",
            DISCOUNT_SALE: "discounted lot sells first at r(1+u): sD = min(Q, r(1+u)H), T1 = Q / (r(1+u)); base sells at r "
                           "for H - T1: sB = min(B, r max(0, H - T1)); unsold = Q - (sD + sB - ref_sold); "
                           "L = p d sD + p unsold + h_s UD_D + (c_d - s) unsold",
            TRANSFER: "q = min(Q, S, source_surplus_cap, target_need_cap, capacity, max(0, r_t H_t - S_t)); H_t = H - transit; "
                      "sold_src = min(S - q, rH) - ref_sold; sold_t = min(S_t + q, r_t H_t) - min(S_t, r_t H_t); "
                      "L = C + (p - p_t) sold_t + p (unsold_src + unsold_t) + h_s UD_src + h_t UD_t + c_d unsold_src "
                      "+ c_dt unsold_t - s (unsold_src + unsold_t)",
            "transfer_bridge": "L_T - L_N = C + p (sold_N - sold_src) - p_t sold_t + d_holding + d_disposal - d_salvage "
                               "(source_opportunity_loss = p (sold_N - sold_src); reported, never added twice)",
        },
        "unknown_rates": {k: [lo, "source_normal_price" if hi is None else ("inf" if math.isinf(hi) else hi)]
                          for k, (lo, hi) in UNKNOWN_RATE_PARAMS.items()},
        "robustness_rule": "With unknown rates, s is recommended only if min over the admissible box of L(j) - L(s) >= 0 for every other comparable j.",
        "statuses": {STATUS_FULL: "all three strategies comparable", STATUS_PARTIAL: "exactly two comparable",
                     STATUS_UNAVAILABLE: "fewer than two comparable or a blocking input problem"},
        "recommendation_statuses": [RECOMMENDED, NOT_ROBUST, NO_COMPARISON],
        "tie_break_order": list(TIE_BREAK_ORDER),
        "reason_codes": dict(sorted(REASON_MESSAGES.items())),
        "unit_rule": "Declared units must all be equal; non-physical units block; undeclared units are accepted only when every quantity and per-unit value comes from one dataset/upload.",
        "currency_rule": "Every money field must carry the same declared currency; no exchange rate is applied.",
        "dataset_rule": "Product economic fields must come from one dataset; route transport cost is matched on the same store ids and checked for currency.",
        "sunk_cost_rule": "unit_cost is identical across strategies and excluded from L.",
        "legacy_relation": "Runs next to Varo Final; legacy_action is recorded and never replaced.",
    }
    document["contract_signature"] = hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return document


# ---------------------------------------------------------------- arithmetic


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean(value: float) -> float:
    """Remove floating residue around zero without clipping real negatives."""
    return 0.0 if abs(value) <= 1e-9 else value


def units_sold(stock: float, rate: float, horizon: float) -> float:
    """Units sold from ``stock`` at a constant demand ``rate`` within ``horizon`` days."""
    if stock <= 0 or rate <= 0 or horizon <= 0:
        return 0.0
    return min(stock, rate * horizon)


def unit_days(stock: float, rate: float, horizon: float) -> float:
    """Inventory unit-days of ``stock`` depleting at ``rate`` over ``horizon`` days (holding-cost basis)."""
    if stock <= 0 or horizon <= 0:
        return 0.0
    if rate <= 0:
        return stock * horizon
    sellout = stock / rate
    if sellout <= horizon:
        return stock * sellout / 2.0
    return stock * horizon - rate * horizon * horizon / 2.0


def _lin(rate: float | None, basis: float, param: str, sign: float = 1.0) -> tuple[float, dict[str, float]]:
    """Linear loss piece rate x basis; an unknown rate becomes a coefficient on ``param``."""
    if abs(basis) <= 1e-12:
        return 0.0, {}
    if rate is not None:
        return sign * rate * basis, {}
    return 0.0, {param: sign * basis}


def _sum(pieces: Iterable[tuple[float, dict[str, float]]]) -> tuple[float, dict[str, float]]:
    const, coef = 0.0, {}
    for piece_const, piece_coef in pieces:
        const += piece_const
        for name, value in piece_coef.items():
            coef[name] = coef.get(name, 0.0) + value
    return const, {name: value for name, value in sorted(coef.items()) if abs(value) > 1e-12}


def _range(lin: tuple[float, dict[str, float]], bounds: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
    low = high = lin[0]
    for name, coef in lin[1].items():
        lo, hi = bounds[name]
        if coef > 0:
            low += coef * lo
            high = math.inf if math.isinf(hi) else high + coef * hi
        else:
            low = -math.inf if math.isinf(hi) else low + coef * hi
            high += coef * lo
    return low, high


def _margin(best: tuple[float, dict[str, float]], other: tuple[float, dict[str, float]],
            bounds: Mapping[str, tuple[float, float]]) -> float:
    """Guaranteed advantage: min over the admissible box of L(other) - L(best)."""
    total = other[0] - best[0]
    for name in sorted(set(best[1]) | set(other[1])):
        coef = other[1].get(name, 0.0) - best[1].get(name, 0.0)
        if abs(coef) <= 1e-12:
            continue
        lo, hi = bounds[name]
        if coef > 0:
            total += coef * lo
        elif math.isinf(hi):
            return -math.inf
        else:
            total += coef * hi
    return total


# ---------------------------------------------------------------- validation of inputs


def _accepted(item: InputField, name: str, reasons: list[str]) -> float | None:
    if not item.present:
        return None
    if item.provenance not in ACCEPTED_PROVENANCE:
        reasons.append(f"PROXY_REJECTED:{name}")
        return None
    return float(item.value)


def _consistency(items: Mapping[str, InputField], declared_unit: str | None) -> tuple[list[str], list[str]]:
    """Unit, currency, dataset and sign checks over the present fields of ``items``.

    ``declared_unit`` is the caller's statement of the decision's quantity unit; it covers fields without their own unit.
    """
    blockers: list[str] = []
    notes: list[str] = []
    present = {name: item for name, item in items.items() if item.present}
    sized = {name: item for name, item in present.items() if name in QUANTITY_FIELDS or name in PER_UNIT_MONEY_FIELDS}

    declared = {item.unit for item in sized.values() if item.unit}
    if declared_unit:
        declared.add(declared_unit)
    if declared & NON_PHYSICAL_UNITS:
        blockers.append("NON_PHYSICAL_UNIT")
    if len(declared) > 1:
        blockers.append("UNIT_MISMATCH")
    # Explicit scenario values are defined per decision unit by contract and carry no source unit.
    undeclared = [name for name, item in sized.items() if not item.unit and item.provenance not in SCENARIO_PROVENANCE]
    if undeclared and not declared_unit:
        # An undeclared unit is accepted only when every sized value comes from one dataset/upload.
        sources = {item.dataset for item in sized.values() if item.provenance not in SCENARIO_PROVENANCE}
        if len(sources) != 1 or None in sources:
            blockers.append("UNIT_UNVERIFIABLE_ACROSS_SOURCES")
        else:
            notes.append("UNIT_UNDECLARED_SINGLE_SOURCE")

    money = {name: item for name, item in present.items() if name in MONEY_FIELDS}
    if len({item.currency for item in money.values() if item.currency}) > 1:
        blockers.append("CURRENCY_MISMATCH")
    if any(not item.currency for item in money.values()):
        blockers.append("CURRENCY_UNDECLARED")

    if len({item.dataset for name, item in present.items() if name in PRODUCT_ECONOMIC_FIELDS and item.dataset}) > 1:
        blockers.append("CROSS_DATASET_INPUT")
    for name, item in present.items():
        if name not in SIGN_FREE_FIELDS and float(item.value) < 0:
            blockers.append(f"NEGATIVE_INPUT:{name}")
    return list(dict.fromkeys(blockers)), list(dict.fromkeys(notes))


def _consistency_blockers(inp: SellerDecisionInput) -> tuple[list[str], list[str], list[str]]:
    """(global blockers, transfer-only blockers, notes): a transfer-only problem never blocks the other strategies."""
    items = inp.input_fields()
    core = {name: item for name, item in items.items() if name not in TRANSFER_ONLY_FIELDS}
    core_blockers, core_notes = _consistency(core, inp.quantity_unit)
    all_blockers, all_notes = _consistency(items, inp.quantity_unit)
    transfer_only = [code for code in all_blockers if code not in core_blockers]
    return core_blockers, transfer_only, list(dict.fromkeys(core_notes + all_notes))


def _integral(*values: float) -> bool:
    return all(abs(value - round(value)) <= 1e-9 for value in values)


# ---------------------------------------------------------------- strategies


def _result_lin(components: Mapping[str, tuple[float, dict[str, float]]]) -> tuple[float, dict[str, float]]:
    return _sum(components[name] for name in COMPONENTS)


def _strategy_payload(strategy: str, components: Mapping[str, tuple[float, dict[str, float]]],
                      quantities: Mapping[str, float], bounds: Mapping[str, tuple[float, float]],
                      extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    full = {name: components.get(name, (0.0, {})) for name in COMPONENTS}
    lin = _result_lin(full)
    low, high = _range(lin, bounds)
    payload = {
        "strategy": strategy,
        "label": STRATEGY_LABELS[strategy],
        "available": True,
        "unavailable_reasons": [],
        "quantities": {key: round(_clean(value), 4) for key, value in quantities.items()},
        "components": {name: (round(_clean(full[name][0]), 2) if not full[name][1] else None) for name in COMPONENTS},
        "component_known_part": {name: round(_clean(full[name][0]), 2) for name in COMPONENTS},
        "component_unknown_inputs": {name: sorted(full[name][1]) for name in COMPONENTS if full[name][1]},
        "expected_loss": round(_clean(lin[0]), 2) if not lin[1] else None,
        "expected_loss_excluding_unknown": round(_clean(lin[0]), 2),
        "expected_loss_range": [None if math.isinf(low) else round(_clean(low), 2),
                                None if math.isinf(high) else round(_clean(high), 2)],
        "unknown_inputs": sorted(lin[1]),
        "loss_linear": {"constant": round(_clean(lin[0]), 6), "unknown_coefficients": {k: round(v, 6) for k, v in lin[1].items()}},
    }
    if extra:
        payload.update(extra)
    payload["_lin"] = lin
    return payload


def _unavailable(strategy: str, reasons: Sequence[str]) -> dict[str, Any]:
    return {"strategy": strategy, "label": STRATEGY_LABELS[strategy], "available": False,
            "unavailable_reasons": list(dict.fromkeys(reasons)), "quantities": {}, "components": {name: None for name in COMPONENTS},
            "component_known_part": {}, "component_unknown_inputs": {}, "expected_loss": None,
            "expected_loss_excluding_unknown": None, "expected_loss_range": [None, None], "unknown_inputs": [],
            "loss_linear": None}


def evaluate_seller_decision(inp: SellerDecisionInput) -> dict[str, Any]:
    """Compare TRANSFER / NORMAL_SALE / DISCOUNT_SALE for ``inp.decision_qty`` (pure, deterministic)."""
    notes: list[str] = []
    blockers, transfer_blockers, unit_notes = _consistency_blockers(inp)
    notes.extend(unit_notes)
    rejected: list[str] = []

    Q = _accepted(inp.decision_qty, "decision_qty", rejected)
    S = _accepted(inp.source_current_stock, "source_current_stock", rejected)
    r = _accepted(inp.source_daily_demand, "source_daily_demand", rejected)
    H = _accepted(inp.remaining_shelf_life_days, "remaining_shelf_life_days", rejected)
    p = _accepted(inp.source_normal_price, "source_normal_price", rejected)
    def absent(name: str) -> bool:  # missing, as opposed to present but rejected (that reason is already recorded)
        return f"PROXY_REJECTED:{name}" not in rejected

    if Q is None and absent("decision_qty"):
        blockers.append("DECISION_QTY_MISSING")
    if S is None and absent("source_current_stock"):
        blockers.append("SOURCE_STOCK_MISSING")
    if S is not None and S <= 0 and (Q is None or Q <= 0):
        blockers.append("ZERO_INVENTORY")
    elif Q is not None and Q <= 0:
        blockers.append("DECISION_QTY_NOT_POSITIVE")
    if Q is not None and S is not None and Q > S + 1e-9:
        blockers.append("DECISION_QTY_EXCEEDS_STOCK")
    if r is None and absent("source_daily_demand"):
        blockers.append("SOURCE_DEMAND_MISSING")
    if H is None and absent("remaining_shelf_life_days"):
        blockers.append("SHELF_LIFE_MISSING")
    if p is None and absent("source_normal_price"):
        blockers.append("PRICE_MISSING")
    blockers.extend(rejected)
    blockers = list(dict.fromkeys(blockers))

    base = {
        "engine_version": ENGINE_VERSION,
        "criterion": CRITERION,
        "loss_reference": LOSS_REFERENCE,
        "decision_id": inp.decision_id,
        "product_id": inp.product_id,
        "source_store_id": inp.source_store_id,
        "target_store_id": inp.target_store_id,
        "decision_qty": Q,
        "quantity_unit": inp.quantity_unit,
        "currency": _currency(inp),
        "legacy_action": inp.legacy_action,
        "legacy_action_changed": False,
        "input_provenance": {name: item.as_dict() for name, item in inp.input_fields().items()},
        "data_completeness": _completeness(inp),
    }
    if blockers:
        strategies = {s: _unavailable(s, blockers) for s in STRATEGIES}
        return _finish(base, strategies, {}, notes, blockers)

    B = S - Q
    salvage_bound = {**UNKNOWN_RATE_PARAMS, "salvage_value_per_unit": (0.0, p)}
    bounds: dict[str, tuple[float, float]] = dict(salvage_bound)
    rate_notes: list[str] = []
    h_s = _accepted(inp.source_holding_cost_per_unit_day, "source_holding_cost_per_unit_day", rate_notes)
    c_s = _accepted(inp.source_disposal_cost_per_unit, "source_disposal_cost_per_unit", rate_notes)
    salv = _accepted(inp.salvage_value_per_unit, "salvage_value_per_unit", rate_notes)
    h_t = _accepted(inp.target_holding_cost_per_unit_day, "target_holding_cost_per_unit_day", rate_notes)
    c_t = _accepted(inp.target_disposal_cost_per_unit, "target_disposal_cost_per_unit", rate_notes)
    notes.extend(rate_notes)
    if salv is not None and salv > p + 1e-9:
        notes.append("SALVAGE_ABOVE_NORMAL_PRICE")
    if r <= 0:
        notes.append("ZERO_SOURCE_DEMAND")
    if H <= 0:
        notes.append("EXPIRED_STOCK")

    ref_sold = units_sold(B, r, H)
    ref_ud = unit_days(B, r, H)
    sold_n = units_sold(S, r, H) - ref_sold
    unsold_n = Q - sold_n
    ud_n = unit_days(S, r, H) - ref_ud
    strategies: dict[str, dict[str, Any]] = {}

    def keep_components(unsold: float, ud: float) -> dict[str, tuple[float, dict[str, float]]]:
        return {
            "opportunity_loss": (p * unsold, {}),
            "holding_loss": _lin(h_s, ud, "source_holding_cost_per_unit_day"),
            "disposal_loss": _lin(c_s, unsold, "source_disposal_cost_per_unit"),
            "salvage_recovery": _lin(salv, unsold, "salvage_value_per_unit", -1.0),
        }

    strategies[NORMAL_SALE] = _strategy_payload(
        NORMAL_SALE, keep_components(unsold_n, ud_n),
        {"decision_qty": Q, "expected_sold_qty": sold_n, "expected_unsold_qty": unsold_n, "holding_unit_days": ud_n},
        bounds,
    )

    # DISCOUNT_SALE
    discount_reasons: list[str] = []
    d = _accepted(inp.discount_rate, "discount_rate", discount_reasons)
    u = _accepted(inp.promotion_uplift, "promotion_uplift", discount_reasons)
    if d is None and not any(x.endswith(":discount_rate") for x in discount_reasons):
        discount_reasons.append("DISCOUNT_RATE_MISSING")
    elif d is not None and not (0.0 < d < 1.0):
        discount_reasons.append("INVALID_DISCOUNT_RATE")
    if u is None and not any(x.endswith(":promotion_uplift") for x in discount_reasons):
        discount_reasons.append("UPLIFT_MISSING")
    elif u is not None and u < 0:
        discount_reasons.append("INVALID_UPLIFT")
    if discount_reasons:
        strategies[DISCOUNT_SALE] = _unavailable(DISCOUNT_SALE, discount_reasons)
    else:
        rate_d = r * (1.0 + u)
        sold_d_lot = units_sold(Q, rate_d, H)
        first_phase = Q / rate_d if rate_d > 0 else math.inf
        second_phase = max(0.0, H - first_phase) if math.isfinite(first_phase) else 0.0
        sold_d_base = units_sold(B, r, second_phase)
        ud_d = unit_days(Q, rate_d, H) + B * min(first_phase, max(H, 0.0)) + unit_days(B, r, second_phase) - ref_ud
        net_sold = sold_d_lot + sold_d_base - ref_sold
        unsold_d = Q - net_sold
        cannibalised = ref_sold - sold_d_base
        components = keep_components(unsold_d, ud_d)
        components["discount_loss"] = (p * d * sold_d_lot, {})
        if cannibalised > 1e-9:
            notes.append("CANNIBALIZATION_PRESENT")
        if r <= 0:
            notes.append("ZERO_BASE_DEMAND_UPLIFT_HAS_NO_EFFECT")
        strategies[DISCOUNT_SALE] = _strategy_payload(
            DISCOUNT_SALE, components,
            {"decision_qty": Q, "expected_sold_qty": net_sold, "expected_unsold_qty": unsold_d,
             "discounted_units_sold": sold_d_lot, "cannibalized_regular_units": cannibalised, "holding_unit_days": ud_d},
            bounds, {"discount_rate": d, "promotion_uplift": u, "discounted_price": round(p * (1.0 - d), 4)},
        )

    # TRANSFER
    strategies[TRANSFER] = (
        _unavailable(TRANSFER, transfer_blockers) if transfer_blockers else
        _transfer(inp, Q, S, B, r, H, p, h_s, h_t, c_s, c_t, salv, ref_sold, ref_ud, sold_n, bounds, notes)
    )
    return _finish(base, strategies, bounds, notes, [])


def _transfer(inp: SellerDecisionInput, Q: float, S: float, B: float, r: float, H: float, p: float,
              h_s: float | None, h_t: float | None, c_s: float | None, c_t: float | None, salv: float | None,
              ref_sold: float, ref_ud: float, sold_n: float,
              bounds: dict[str, tuple[float, float]], notes: list[str]) -> dict[str, Any]:
    reasons: list[str] = []
    if not inp.target_store_id:
        reasons.append("TARGET_STORE_MISSING")
    S_t = _accepted(inp.target_current_stock, "target_current_stock", reasons)
    r_t = _accepted(inp.target_daily_demand, "target_daily_demand", reasons)
    p_t = _accepted(inp.target_normal_price, "target_normal_price", reasons)
    cost = _accepted(inp.transfer_cost, "transfer_cost", reasons)
    transit = _accepted(inp.transit_time_days, "transit_time_days", reasons)
    if S_t is None and not any(x.endswith(":target_current_stock") for x in reasons):
        reasons.append("TARGET_STOCK_MISSING")
    if r_t is None and not any(x.endswith(":target_daily_demand") for x in reasons):
        reasons.append("TARGET_DEMAND_MISSING")
    if p_t is None and not any(x.endswith(":target_normal_price") for x in reasons):
        reasons.append("TARGET_PRICE_MISSING")
    if cost is None and not any(x.endswith(":transfer_cost") for x in reasons):
        reasons.append("TRANSFER_COST_MISSING")
    if transit is None and not any(x.endswith(":transit_time_days") for x in reasons):
        reasons.append("TRANSIT_TIME_MISSING")
    if inp.transfer_cost_basis not in TRANSFER_COST_BASES:
        reasons.append("INVALID_TRANSFER_COST_BASIS")
    if reasons:
        return _unavailable(TRANSFER, reasons)

    H_t = H - transit
    engine_need = max(0.0, r_t * H_t - S_t)
    caps: list[tuple[str, float]] = [("decision_qty", Q), ("source_current_stock", S), ("target_horizon_need", engine_need)]
    for name, item in (("source_surplus_cap", inp.source_surplus_cap), ("target_need_cap", inp.target_need_cap),
                       ("route_capacity_qty", inp.route_capacity_qty)):
        value = _accepted(item, name, notes)
        if value is not None:
            caps.append((name, max(0.0, value)))
    q = min(value for _, value in caps)
    if _integral(Q, S, S_t):
        q = math.floor(q + 1e-9)
    q = max(0.0, float(q))
    binding = sorted(name for name, value in caps if abs(value - min(v for _, v in caps)) <= 1e-9)
    if q <= 0:
        return _unavailable(TRANSFER, ["NO_EXECUTABLE_TRANSFER_QTY", *(f"TRANSFER_QTY_CAPPED:{name}" for name in binding)])

    cost_piece: tuple[float, dict[str, float]]
    cost_qty = _finite(inp.transfer_cost_qty.value) if inp.transfer_cost_qty.usable else None
    if inp.transfer_cost_basis == "PER_UNIT":
        cost_piece = (cost * q, {})
    elif inp.transfer_cost_basis == "FIXED_PER_TRIP":
        cost_piece = (cost, {})
    elif cost_qty is not None and abs(q - cost_qty) <= 1e-9:
        cost_piece = (cost, {})
    elif cost_qty is not None and q < cost_qty:
        # A smaller shipment never costs more than the priced one; the exact cost is unknown in [0, priced cost].
        cost_piece = (0.0, {"transfer_cost": 1.0})
        bounds["transfer_cost"] = (0.0, cost)
        notes.append("TRANSFER_COST_UPPER_BOUND_ONLY")
    else:
        return _unavailable(TRANSFER, ["TRANSFER_COST_QTY_MISMATCH"])

    if q < Q - 1e-9:
        notes.append("PARTIAL_TRANSFER_REMAINDER_NORMAL_SALE")
        notes.extend(f"TRANSFER_QTY_CAPPED:{name}" for name in binding)
    source_after = S - q
    sold_src = units_sold(source_after, r, H) - ref_sold
    unsold_src = (Q - q) - sold_src
    ud_src = unit_days(source_after, r, H) - ref_ud
    sold_t = units_sold(S_t + q, r_t, H_t) - units_sold(S_t, r_t, H_t)
    unsold_t = q - sold_t
    ud_t = unit_days(S_t + q, r_t, H_t) - unit_days(S_t, r_t, H_t)
    unsold_total = unsold_src + unsold_t
    components = {
        "transfer_cost": cost_piece,
        "price_difference_loss": ((p - p_t) * sold_t, {}),
        "opportunity_loss": (p * unsold_total, {}),
        "holding_loss": _sum([_lin(h_s, ud_src, "source_holding_cost_per_unit_day"),
                              _lin(h_t, ud_t, "target_holding_cost_per_unit_day")]),
        "disposal_loss": _sum([_lin(c_s, unsold_src, "source_disposal_cost_per_unit"),
                               _lin(c_t, unsold_t, "target_disposal_cost_per_unit")]),
        "salvage_recovery": _lin(salv, unsold_total, "salvage_value_per_unit", -1.0),
    }
    opportunity_units = sold_n - sold_src
    if opportunity_units > 1e-9:
        notes.append("SOURCE_OPPORTUNITY_LOSS_PRESENT")
    bridge = {
        "source_opportunity_units": round(_clean(opportunity_units), 4),
        "source_opportunity_loss": round(_clean(p * opportunity_units), 2),
        "target_recovered_value": round(_clean(p_t * sold_t), 2),
        "note": "L(TRANSFER) - L(NORMAL_SALE) = transfer_cost + source_opportunity_loss - target_recovered_value "
                "+ holding/disposal/salvage differences; reported for explanation, not added to the components.",
    }
    return _strategy_payload(
        TRANSFER, components,
        {"decision_qty": Q, "moved_qty": q, "kept_qty": Q - q, "expected_sold_qty": sold_src + sold_t,
         "expected_unsold_qty": unsold_total, "target_sold_qty": sold_t, "target_unsold_qty": unsold_t,
         "source_kept_sold_qty": sold_src, "source_kept_unsold_qty": unsold_src, "holding_unit_days": ud_src + ud_t,
         "target_horizon_days": H_t, "target_horizon_need": engine_need},
        bounds,
        {"transfer_bridge": bridge, "binding_constraints": binding,
         "source_surplus_basis": inp.source_surplus_basis, "target_need_basis": inp.target_need_basis},
    )


# ---------------------------------------------------------------- ranking, status, explanation


def _currency(inp: SellerDecisionInput) -> str | None:
    currencies = {item.currency for name, item in inp.input_fields().items()
                  if name in MONEY_FIELDS and item.present and item.currency}
    return next(iter(currencies)) if len(currencies) == 1 else None


def _completeness(inp: SellerDecisionInput) -> float:
    items = inp.input_fields()
    return round(sum(1 for name in COMPLETENESS_FIELDS if items[name].usable) / len(COMPLETENESS_FIELDS), 3)


def rank_strategies(strategies: Mapping[str, Mapping[str, Any]], bounds: Mapping[str, tuple[float, float]]) -> dict[str, Any]:
    """Robust minimum-loss ranking over the admissible range of every unknown rate."""
    available = [s for s in TIE_BREAK_ORDER if strategies[s].get("available")]
    if len(available) < 2:
        return {"winner": None, "robust": False, "tie": False, "second": None, "margin": None}
    lin = {s: strategies[s]["_lin"] for s in available}
    dominators = [k for k in available if all(_margin(lin[k], lin[j], bounds) >= -TOL for j in available if j != k)]
    if not dominators:
        return {"winner": None, "robust": False, "tie": False, "second": None, "margin": None}
    winner = dominators[0]
    margins = sorted(((max(0.0, _margin(lin[winner], lin[j], bounds)), TIE_BREAK_ORDER.index(j), j)
                      for j in available if j != winner))
    margin, _, second = margins[0]
    margin = 0.0 if margin <= TOL else margin
    return {"winner": winner, "robust": True, "tie": margin == 0.0, "second": second, "margin": round(margin, 2)}


def _decision_confidence(status: str, ranking: Mapping[str, Any], unknown: Sequence[str], scenario: Sequence[str]) -> str | None:
    if not ranking.get("winner"):
        return None
    if status == STATUS_FULL and not unknown and not scenario:
        return "HIGH"
    return "MEDIUM" if status == STATUS_FULL else "LOW"


def _finish(base: dict[str, Any], strategies: dict[str, dict[str, Any]], bounds: Mapping[str, tuple[float, float]],
            notes: Sequence[str], blockers: Sequence[str]) -> dict[str, Any]:
    available = [s for s in STRATEGIES if strategies[s]["available"]]
    status = STATUS_FULL if len(available) == 3 else STATUS_PARTIAL if len(available) == 2 else STATUS_UNAVAILABLE
    ranking = rank_strategies(strategies, bounds) if status != STATUS_UNAVAILABLE else {
        "winner": None, "robust": False, "tie": False, "second": None, "margin": None}
    unknown = sorted({name for s in available for name in strategies[s]["unknown_inputs"]})
    scenario = sorted(name for name, item in base["input_provenance"].items()
                      if item["provenance"] in SCENARIO_PROVENANCE and item["value"] is not None
                      and (name not in ("discount_rate", "promotion_uplift") or DISCOUNT_SALE in available))
    reasons = list(blockers) + list(notes)
    for s in STRATEGIES:
        if not strategies[s]["available"] and not blockers:
            reasons.extend(f"{s}:{code}" for code in strategies[s]["unavailable_reasons"])
    for name in unknown:
        reasons.append(f"UNKNOWN_RATE:{name}")
    reasons.extend(f"SCENARIO_INPUT:{name}" for name in scenario)
    if status == STATUS_UNAVAILABLE:
        recommendation_status = NO_COMPARISON
    elif ranking["winner"]:
        recommendation_status = RECOMMENDED
        if unknown:
            reasons.append("RANKING_ROBUST_TO_UNKNOWN")
        if ranking["tie"]:
            reasons.append("TIE_BROKEN_BY_OPERATIONAL_SIMPLICITY")
    else:
        recommendation_status = NOT_ROBUST
        reasons.append("RANKING_DEPENDS_ON_UNKNOWN")
    if base["input_provenance"]["unit_cost"]["value"] is not None and status != STATUS_UNAVAILABLE:
        reasons.append("UNIT_COST_SUNK_NOT_IN_RANKING")
    winner = ranking["winner"]
    public = {s: {k: v for k, v in strategies[s].items() if k != "_lin"} for s in STRATEGIES}
    result = dict(base)
    result.update({
        "comparison_status": status,
        "comparable_strategies": available,
        "unavailable_strategies": {s: strategies[s]["unavailable_reasons"] for s in STRATEGIES if s not in available},
        "recommendation_status": recommendation_status,
        "recommended_strategy": winner,
        "seller_loss_action": SELLER_LOSS_ACTIONS.get(winner) if winner else None,
        "runner_up_strategy": ranking["second"],
        "loss_difference_vs_second_best": ranking["margin"],
        "tie": ranking["tie"],
        "expected_loss_transfer": public[TRANSFER]["expected_loss"],
        "expected_loss_normal_sale": public[NORMAL_SALE]["expected_loss"],
        "expected_loss_discount_sale": public[DISCOUNT_SALE]["expected_loss"],
        "expected_sold_qty": {s: public[s]["quantities"].get("expected_sold_qty") for s in STRATEGIES},
        "expected_unsold_qty": {s: public[s]["quantities"].get("expected_unsold_qty") for s in STRATEGIES},
        "breakdown": {s: public[s]["components"] for s in STRATEGIES},
        "strategies": public,
        "unknown_inputs": unknown,
        "unknown_input_bounds": {name: [bounds[name][0], None if math.isinf(bounds[name][1]) else bounds[name][1]]
                                 for name in unknown},
        "scenario_inputs": scenario,
        "decision_confidence": _decision_confidence(status, ranking, unknown, scenario),
        "reason_codes": list(dict.fromkeys(reasons)),
    })
    result["explanation"], result["explanation_facts"] = explain(result)
    result["agreement_with_legacy"] = legacy_agreement(base.get("legacy_action"), winner)
    return result


def _money(value: float | None, currency: str | None) -> str:
    if value is None:
        return "미상"
    if currency == "KRW":
        return f"{round(value):,}원"
    return f"{value:,.2f} {currency or ''}".strip()


def _qty(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{int(round(value)):,}" if abs(value - round(value)) <= 1e-9 else f"{value:,.2f}"


def _josa(word: str, pair: str) -> str:
    """Attach the Korean particle of ``pair`` ("은/는", "이/가", "과/와", "을/를", "으로/로") that fits the last syllable."""
    with_final, without_final = pair.split("/")
    syllable = next((ch for ch in reversed(word) if "가" <= ch <= "힣"), None)
    final = (ord(syllable) - 0xAC00) % 28 if syllable else 0
    if pair == "으로/로":
        return word + (without_final if final in (0, 8) else with_final)
    return word + (with_final if final else without_final)


def _reason_text(code: str) -> str:
    head, _, field_name = code.partition(":")
    if head == "PROXY_REJECTED":
        return f"{field_name} 값이 대리지표(proxy)라 금액 비교에 쓸 수 없습니다"
    if head == "NEGATIVE_INPUT":
        return f"{field_name}에 음수 값이 있습니다"
    if head == "TRANSFER_QTY_CAPPED":
        return f"{field_name} 제약으로 이동 수량이 제한됩니다"
    return REASON_MESSAGES.get(head, code)


def explain(result: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Korean explanation built only from the computed numbers (template text, no free-form inference)."""
    currency = result.get("currency")
    strategies = result["strategies"]
    status = result["comparison_status"]
    facts: dict[str, Any] = {"status": status, "currency": currency}
    if status == STATUS_UNAVAILABLE:
        codes: list[str] = []
        for s in STRATEGIES:
            codes.extend(strategies[s]["unavailable_reasons"])
        codes = list(dict.fromkeys(codes))
        facts["reasons"] = codes
        return "금액 비교 불가: " + "; ".join(_reason_text(code) for code in codes) + ".", facts

    available = result["comparable_strategies"]
    unit = result.get("quantity_unit") or "개"
    quantity = f"{_qty(result.get('decision_qty'))}{unit}"
    if status == STATUS_FULL:
        scope = f"같은 재고 {quantity}에 대해 세 가지 처리 방법을 비교했습니다."
    else:
        missing = next(s for s in STRATEGIES if s not in available)
        why = "; ".join(_reason_text(code) for code in strategies[missing]["unavailable_reasons"])
        scope = (f"같은 재고 {quantity}에 대해 비교 가능한 {' · '.join(STRATEGY_LABELS[s] for s in available)} "
                 f"두 가지만 비교했습니다({STRATEGY_LABELS[missing]} 제외: {why}).")
    winner = result.get("recommended_strategy")
    if not winner:
        names = list(result.get("unknown_inputs", []))
        labels = ", ".join(PARAM_LABELS.get(name, name) for name in names)
        facts["unknown_inputs"] = names
        return scope + f" {labels} 데이터가 없고 그 값에 따라 손실이 가장 작은 방법이 달라져 추천하지 않습니다.", facts

    second = result["runner_up_strategy"]
    margin = result["loss_difference_vs_second_best"]
    win, run = strategies[winner], strategies[second]
    exact = win["expected_loss"] is not None and run["expected_loss"] is not None
    win_loss = win["expected_loss"] if win["expected_loss"] is not None else win["expected_loss_excluding_unknown"]
    run_loss = run["expected_loss"] if run["expected_loss"] is not None else run["expected_loss_excluding_unknown"]
    facts.update({"winner": winner, "runner_up": second, "winner_loss": win_loss, "runner_up_loss": run_loss,
                  "margin": margin, "losses_exact": exact})
    win_label, run_label = STRATEGY_LABELS[winner], STRATEGY_LABELS[second]
    known_note = "" if exact else "(데이터 없는 항목 제외 금액)"
    sentences = [scope, f"{win_label}의 예상 손실이 {_josa(_money(win_loss, currency), '으로/로')} 가장 작습니다{known_note}."]
    if result.get("tie"):
        sentences.append(f"{_josa(run_label, '과/와')} 예상 손실이 같아 추가 작업이 적은 {_josa(win_label, '을/를')} 우선했습니다.")
    else:
        win_parts = {k: v for k, v in win["component_known_part"].items() if v > 0}
        gaps = {k: run["component_known_part"].get(k, 0.0) - win["component_known_part"].get(k, 0.0) for k in COMPONENTS}
        driver = max((k for k in COMPONENTS if gaps[k] > 0), key=lambda k: (gaps[k], -COMPONENTS.index(k)), default=None)
        top = max(win_parts, key=lambda k: (win_parts[k], -COMPONENTS.index(k)), default=None)
        facts.update({"winner_top_component": top, "winner_top_value": win_parts.get(top) if top else None,
                      "runner_up_driver": driver,
                      "runner_up_driver_value": run["component_known_part"][driver] if driver else None})
        lead = (f"{_josa(win_label, '은/는')} {COMPONENT_LABELS[top]} {_josa(_money(win_parts[top], currency), '이/가')} 발생하지만"
                if top else f"{_josa(win_label, '은/는')} 추가 손실 항목이 없고")
        if driver:
            lead += (f", {_josa(run_label, '은/는')} {COMPONENT_LABELS[driver]} "
                     f"{_josa(_money(run['component_known_part'][driver], currency), '이/가')} 발생해")
        bound_word = "" if exact else "최소 "
        sentences.append(f"{lead} 전체 예상 손실은 {_josa(win_label, '이/가')} {run_label}보다 {bound_word}"
                         f"{_money(margin, currency)} 적습니다.")
    transfer = strategies[TRANSFER]
    if winner == TRANSFER and transfer["quantities"].get("kept_qty", 0) > 0:
        sentences.append(f"이동 가능 수량은 {_qty(transfer['quantities']['moved_qty'])}{unit}이며 나머지 "
                         f"{_qty(transfer['quantities']['kept_qty'])}{unit}는 현재 점포 정상 판매로 계산했습니다.")
    if result.get("unknown_inputs"):
        labels = ", ".join(PARAM_LABELS.get(name, name) for name in result["unknown_inputs"])
        sentences.append(f"{labels} 데이터가 없어 금액에 넣지 않았으며, 허용 범위의 어떤 값이어도 추천은 바뀌지 않습니다.")
    if result.get("scenario_inputs"):
        values = ", ".join(f"{name}={result['input_provenance'][name]['value']:g}" for name in result["scenario_inputs"])
        sentences.append(f"설정값(관측치 아님)을 사용했습니다: {values}.")
    return " ".join(sentences), facts


LEGACY_ACTION_TO_STRATEGY = {"재고 이동": TRANSFER, "할인": DISCOUNT_SALE, "긴급 할인": DISCOUNT_SALE, "보류": NORMAL_SALE}


def legacy_agreement(legacy_action: str | None, winner: str | None) -> str:
    """Compare the existing Varo Final action with the seller-loss recommendation (recorded, never applied)."""
    if not winner:
        return "NO_SELLER_LOSS_RECOMMENDATION"
    if not legacy_action:
        return "NO_LEGACY_ACTION"
    strategy = LEGACY_ACTION_TO_STRATEGY.get(str(legacy_action).strip())
    if strategy is None:
        return "LEGACY_ACTION_OUTSIDE_SCOPE"   # 폐기, 1+1, 비교 불가: not one of the three compared strategies
    return "SAME" if strategy == winner else "DIFFERENT"


# ---------------------------------------------------------------- pipeline adapter (parallel, read-only)

WORKBOOK_DATASET = "uploaded_workbook"
WORKBOOK_CURRENCY = "KRW"
WORKBOOK_CURRENCY_NOTE = "Varo V2 workbook contract: amounts are entered and rendered in 원 (column aliases strip '원')."


def _text_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _unique_rows(frame: Any, store_columns: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "product_id" not in frame.columns:
        return {}
    store_col = next((c for c in store_columns if c in frame.columns), None)
    if store_col is None:
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates: set[tuple[str, str]] = set()
    for record in frame.to_dict("records"):
        key = (_text_id(record.get(store_col)), _text_id(record.get("product_id")))
        if key in rows:
            duplicates.add(key)
        rows[key] = record
    for key in duplicates:
        rows.pop(key, None)  # an ambiguous store-product row cannot supply one value
    return rows


def _config(config: Any) -> dict[str, Any]:
    if not isinstance(config, pd.DataFrame) or config.empty:
        return {}
    key_col = next((c for c in ("key", "config_key", "setting", "name") if c in config.columns), None)
    value_col = next((c for c in ("value", "config_value", "setting_value") if c in config.columns), None)
    if not key_col or not value_col:
        return {}
    values: dict[str, Any] = {}
    for record in config.to_dict("records"):
        key = str(record.get(key_col) or "").strip()
        value = record.get(value_col)
        if key and key not in values and not (value is None or (isinstance(value, float) and math.isnan(value))):
            values[key] = value
    return values


def _workbook_field(row: Mapping[str, Any] | None, columns: Sequence[str], label: str, *, money: bool,
                    currency: str, unit: str | None, provenance: str = "USER_INPUT") -> InputField:
    for column in columns:
        if row is not None and column in row:
            value = _finite(row.get(column))
            if value is not None:
                return known(value, provenance, f"{label}.{column}", unit=unit, currency=currency if money else None,
                             dataset=WORKBOOK_DATASET)
    return InputField(source=f"{label}.{'|'.join(columns)} absent")


def pipeline_decision_input(
    recommendation: Mapping[str, Any], *, inventory_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    forecast_rows: Mapping[tuple[str, str], Mapping[str, Any]], product_rows: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any], transition: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
) -> SellerDecisionInput:
    """Map one existing recommendation + uploaded workbook rows onto the contract without inventing values."""
    product = _text_id(recommendation.get("product_id"))
    source = _text_id(recommendation.get("source_id"))
    target = _text_id(recommendation.get("target_id")) or None
    currency = str(config.get("currency") or WORKBOOK_CURRENCY).strip()
    unit = str(config.get("quantity_unit")).strip() if config.get("quantity_unit") else None
    src_row = inventory_rows.get((source, product))
    tgt_row = inventory_rows.get((target or "", product))
    prod_row = product_rows.get(product)

    def price(row: Mapping[str, Any] | None, store_label: str) -> InputField:
        item = _workbook_field(row, ("unit_price",), f"inventory[{store_label}]", money=True, currency=currency, unit=unit)
        if item.present:
            return item
        return _workbook_field(prod_row, ("unit_price",), "products(product-level)", money=True, currency=currency, unit=unit)

    def money_field(row: Mapping[str, Any] | None, store_label: str, columns: Sequence[str]) -> InputField:
        item = _workbook_field(row, columns, f"inventory[{store_label}]", money=True, currency=currency, unit=unit)
        if item.present:
            return item
        return _workbook_field(prod_row, columns, "products(product-level)", money=True, currency=currency, unit=unit)

    def demand(store: str | None) -> InputField:
        row = forecast_rows.get((store or "", product))
        if row is None:
            return InputField(source=f"demand_forecast_router row ({store},{product}) absent")
        semantics = str(row.get("sales_qty_semantics") or "").strip()
        provenance = "PROXY" if semantics == "demand_proxy_not_retail_sales" else "DERIVED_FROM_USER_INPUT"
        version = str(row.get("demand_forecast_version") or "v1")
        return known(row.get("demand_forecast_daily"), provenance,
                     f"demand_forecast_router[{version}].demand_forecast_daily", unit=unit, dataset=WORKBOOK_DATASET,
                     note="7-day forecast rate applied over the remaining shelf life")

    qty = _finite(recommendation.get("recommended_qty"))
    cost_provenance, cost_note = "USER_INPUT", "recommendation move_cost for recommended_qty"
    if candidate and bool(candidate.get("real_transport_applied")):
        proxies = _finite(candidate.get("proxy_vehicle_count")) or 0.0
        cost_provenance = "PROXY" if proxies > 0 else "DERIVED_REAL"
        cost_note = "official-tariff reference estimate (not an invoice)" + (" with upper-class proxy vehicles" if proxies > 0 else "")
    travel = _finite(recommendation.get("travel_time_min", recommendation.get("expected_time_min")))
    movable = (transition or {}).get("metadata", {}).get("movable_stock") if transition else None
    shortage = (transition or {}).get("metadata", {}).get("target_shortage_limit") if transition else None

    discount = InputField(source="config.promotion_discount_rate absent")
    uplift = InputField(source="config.promotion_sales_increase_rate absent")
    promotion_type = str(config.get("promotion_type") or "").strip()
    # The legacy analyzer's in-code defaults (20%, 80%) are placeholders and are never used; only explicit config rows.
    if not promotion_type or "할인" in promotion_type:
        for key in ("promotion_discount_rate", "promotion_sales_increase_rate"):
            percent = _finite(config.get(key))
            if percent is None:
                continue
            item = known(percent / 100.0, "CONFIG", f"config.{key} (%)", note="explicit workbook scenario, not observed")
            if key == "promotion_discount_rate":
                discount = item
            else:
                uplift = item
    row_uplift = _workbook_field(src_row, ("promotion_uplift",), f"inventory[{source}]", money=False, currency=currency, unit=None)
    if row_uplift.present:
        uplift = row_uplift

    return SellerDecisionInput(
        decision_id=str(recommendation.get("route_id") or recommendation.get("recommendation_id") or ""),
        product_id=product, source_store_id=source, target_store_id=target, quantity_unit=unit,
        legacy_action=recommendation.get("varo_action"),
        decision_qty=known(qty, "DERIVED_FROM_USER_INPUT", "recommendation.recommended_qty", unit=unit, dataset=WORKBOOK_DATASET),
        source_current_stock=_workbook_field(src_row, ("stock_qty", "current_stock"), f"inventory[{source}]", money=False, currency=currency, unit=unit),
        source_daily_demand=demand(source),
        remaining_shelf_life_days=_workbook_field(src_row, ("days_to_expiry",), f"inventory[{source}]", money=False, currency=currency, unit=None),
        source_normal_price=price(src_row, source),
        unit_cost=money_field(src_row, source, ("unit_cost",)),
        source_holding_cost_per_unit_day=money_field(src_row, source, ("daily_holding_cost",)),
        source_disposal_cost_per_unit=money_field(src_row, source, ("disposal_cost_per_unit",)),
        salvage_value_per_unit=money_field(src_row, source, ("salvage_value_per_unit", "salvage_value")),
        discount_rate=discount, promotion_uplift=uplift,
        target_current_stock=_workbook_field(tgt_row, ("stock_qty", "current_stock"), f"inventory[{target}]", money=False, currency=currency, unit=unit),
        target_daily_demand=demand(target),
        target_normal_price=price(tgt_row, target or ""),
        target_holding_cost_per_unit_day=money_field(tgt_row, target or "", ("daily_holding_cost",)),
        target_disposal_cost_per_unit=money_field(tgt_row, target or "", ("disposal_cost_per_unit",)),
        transfer_cost=known(recommendation.get("move_cost", recommendation.get("estimated_cost")), cost_provenance,
                            "recommendation.move_cost", currency=currency, dataset=WORKBOOK_DATASET, note=cost_note),
        transfer_cost_basis="QUANTITY_SPECIFIC",
        transfer_cost_qty=known(qty, "DERIVED_FROM_USER_INPUT", "recommendation.recommended_qty", unit=unit, dataset=WORKBOOK_DATASET),
        transit_time_days=known(travel / 1440.0 if travel is not None else None,
                                "DERIVED_REAL" if candidate and bool(candidate.get("real_transport_applied")) else "USER_INPUT",
                                "recommendation.travel_time_min / 1440"),
        source_surplus_cap=known(movable, "DERIVED_FROM_USER_INPUT", "inventory_transition_service movable_stock", unit=unit, dataset=WORKBOOK_DATASET),
        target_need_cap=known(shortage, "DERIVED_FROM_USER_INPUT", "inventory_transition_service target_shortage_limit", unit=unit, dataset=WORKBOOK_DATASET),
        source_surplus_basis=str((transition or {}).get("metadata", {}).get("movable_stock_basis") or ""),
        target_need_basis=str((transition or {}).get("metadata", {}).get("target_shortage_basis") or ""),
    )


def build_seller_loss_analysis(
    uploaded_data: Mapping[str, Any], analyzed_inventory: Any, recommendations: Sequence[Mapping[str, Any]],
    candidates: Any = None,
) -> dict[str, Any]:
    """Seller-loss decisions next to the existing Varo Final results (never modifies them)."""
    from services.inventory_transition_service import build_inventory_baseline, calculate_inventory_transition

    data = {key: value for key, value in (uploaded_data or {}).items()}
    inventory_rows = _unique_rows(data.get("inventory"), ("store_id", "node_id"))
    forecast_source = analyzed_inventory if isinstance(analyzed_inventory, pd.DataFrame) else None
    forecast_rows = _unique_rows(forecast_source, ("store_id", "node_id"))
    products = data.get("products")
    product_rows: dict[str, dict[str, Any]] = {}
    if isinstance(products, pd.DataFrame) and "product_id" in products.columns:
        for record in products.to_dict("records"):
            product_rows.setdefault(_text_id(record.get("product_id")), record)
    config = _config(data.get("config"))
    candidate_rows: dict[str, dict[str, Any]] = {}
    if isinstance(candidates, pd.DataFrame) and "route_id" in candidates.columns:
        for record in candidates.to_dict("records"):
            candidate_rows.setdefault(str(record.get("route_id")), record)
    baseline = build_inventory_baseline(data)
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for recommendation in recommendations or []:
        rec = copy.deepcopy(dict(recommendation))
        transition = calculate_inventory_transition(data, rec, baseline=baseline)
        decision_input = pipeline_decision_input(
            rec, inventory_rows=inventory_rows, forecast_rows=forecast_rows, product_rows=product_rows,
            config=config, transition=transition, candidate=candidate_rows.get(str(rec.get("route_id"))),
        )
        decision = evaluate_seller_decision(decision_input)
        decisions.append(decision)
        rows.append(summary_row(decision))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["comparison_status"]] = counts.get(row["comparison_status"], 0) + 1
    return {
        "status": "parallel_only",
        "engine_version": ENGINE_VERSION,
        "criterion": CRITERION,
        "legacy_action_replaced": False,
        "note": "Seller-loss recommendations are computed next to Varo Final; production actions and ranks are unchanged.",
        "currency": str(config.get("currency") or WORKBOOK_CURRENCY),
        "currency_basis": "config.currency" if config.get("currency") else WORKBOOK_CURRENCY_NOTE,
        "status_counts": dict(sorted(counts.items())),
        "recommended_count": sum(1 for row in rows if row["recommended_strategy"]),
        "rows": rows,
        "decisions": decisions,
    }


def summary_row(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Flat row for tables/CSV: legacy_action and seller_loss_action side by side."""
    return {
        "decision_id": decision.get("decision_id"),
        "product_id": decision.get("product_id"),
        "source_store_id": decision.get("source_store_id"),
        "target_store_id": decision.get("target_store_id"),
        "decision_qty": decision.get("decision_qty"),
        "legacy_action": decision.get("legacy_action"),
        "seller_loss_action": decision.get("seller_loss_action"),
        "agreement_with_legacy": decision.get("agreement_with_legacy"),
        "comparison_status": decision.get("comparison_status"),
        "recommendation_status": decision.get("recommendation_status"),
        "recommended_strategy": decision.get("recommended_strategy"),
        "expected_loss_transfer": decision.get("expected_loss_transfer"),
        "expected_loss_normal_sale": decision.get("expected_loss_normal_sale"),
        "expected_loss_discount_sale": decision.get("expected_loss_discount_sale"),
        "loss_difference_vs_second_best": decision.get("loss_difference_vs_second_best"),
        "currency": decision.get("currency"),
        "data_completeness": decision.get("data_completeness"),
        "decision_confidence": decision.get("decision_confidence"),
        "reason_codes": "|".join(decision.get("reason_codes") or []),
        "explanation": decision.get("explanation"),
    }


def with_fields(inp: SellerDecisionInput, **changes: Any) -> SellerDecisionInput:
    """Return a copy with some fields replaced (scenario and sensitivity helpers)."""
    return replace(inp, **changes)
