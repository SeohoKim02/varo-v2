"""Pre-VHS feasibility gate for Varo V2 recommendations.

A recommendation is only worth ranking if it can actually be executed. This
module evaluates each candidate against concrete, data-driven conditions BEFORE
VHS scoring decides priority, so that infeasible moves are removed (not just
given a low score) and the reason is recorded for diagnostics.

Pure and deterministic: it reads the candidate row plus a small inventory
context and returns a 3-state status the UI can show directly:

* ``추천 가능``      — passes every hard check
* ``데이터 확인 필요`` — kept, but some input is missing/uncertain
* ``이동 불가``      — a hard violation; excluded from the final recommendation set

The detailed ``reason_code`` (for logs/tech docs) is separate from the short
Korean ``reason`` shown on screen. No score is fabricated and no candidate is
kept alive with an artificially low score to hide an impossible move.

Hard constraint vs soft preference
----------------------------------
A **hard constraint** is a fact that makes the move wrong to execute, so it is
enforced here by removing the candidate — never by lowering its VHS score:

    출발지=도착지 · 수량 ≤ 0 · 경로 유형 불명 · VIA_DC인데 DC 없음 ·
    이동 수량 > 출발 재고 · 이동 후 운영 재고 하한 침범 · 음수 이동 비용 ·
    예상 효과 ≤ 이동 비용 · 중복 후보

A **soft preference** is something an operator should weigh, so it stays here as
데이터 확인 필요 and/or feeds a VHS component — never both as a block and a
penalty for the same fact:

    비용/절감액을 계산할 수 없음 · 재고 하한을 계산할 수 없음 ·
    도착 필요량 대비 과다 이동 · 수요 불확실성 · 상대적으로 높은 비용·시간

``services/decision_metrics.py`` computes the numbers both sides read, so the
gate and the score can never disagree about the same quantity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

STATUS_OK = "추천 가능"
STATUS_CHECK = "데이터 확인 필요"
STATUS_BLOCKED = "이동 불가"

VALID_ROUTE_TYPES = {"DIRECT", "VIA_DC"}

INVENTORY_FLOOR_SOURCE_LABELS = {
    "explicit_min_stock": "등록된 최소 보유재고 기준",
    "explicit_safety_stock": "등록된 안전재고 기준",
    "explicit_combined": "등록된 재고 하한 기준",
    "estimated": "수요 변동을 기준으로 추정",
    "calculated": "운영 정책을 기준으로 계산",
    "unavailable": "안전재고 정보 없음",
}

# How far above the destination's measured need a move may go before it is
# flagged as over-supply (kept, but 데이터 확인 필요). Not a hard block.
_OVERSUPPLY_MULTIPLE = 3.0


@dataclass(frozen=True)
class FeasibilityResult:
    status: str
    reason: str          # short, user-facing
    reason_code: str     # stable, for logs / tech docs
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        return self.status == STATUS_BLOCKED

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasibility_status": self.status,
            "feasibility_reason": self.reason,
            "feasibility_reason_code": self.reason_code,
        }


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _key(store: Any, product: Any) -> tuple[str, str]:
    return (str(store), str(product))


def inventory_floor_source_label(source: Any) -> str:
    """Short user-facing provenance; internal codes never reach the screen."""
    return INVENTORY_FLOOR_SOURCE_LABELS.get(str(source or ""), "안전재고 정보 없음")


@dataclass
class InventoryContext:
    """Store-product lookups built once from the uploaded workbook."""
    stock: dict[tuple[str, str], float]
    demand: dict[tuple[str, str], float]
    safety: dict[tuple[str, str], float]
    known_stores: set[str]
    # Measured demand dispersion, only when the file actually carries it. Kept
    # separate from ``safety`` because the safety floor is an estimate derived
    # from it, while this is the raw value the scenario analysis needs.
    dispersion: dict[tuple[str, str], float] = field(default_factory=dict)
    floor_sources: dict[tuple[str, str], str] = field(default_factory=dict)
    target_levels: dict[tuple[str, str], float] = field(default_factory=dict)
    reorder_levels: dict[tuple[str, str], float] = field(default_factory=dict)
    # The candidate generator's established no-target-stock rule: the gap to
    # the product median plus seven days of observed demand.  Kept separate
    # from ``demand`` so existing candidate scoring semantics do not change.
    planning_shortfalls: dict[tuple[str, str], float] = field(default_factory=dict)

    def source_stock(self, store: Any, product: Any) -> float | None:
        return self.stock.get(_key(store, product))

    def target_demand(self, store: Any, product: Any) -> float | None:
        return self.demand.get(_key(store, product))

    def safety_floor(self, store: Any, product: Any) -> float | None:
        """Applied departure floor, preserving unknown separately from explicit 0."""
        return self.safety.get(_key(store, product))

    def inventory_floor_source(self, store: Any, product: Any) -> str:
        return self.floor_sources.get(_key(store, product), "unavailable")

    def available_to_move(self, store: Any, product: Any) -> float | None:
        stock = self.source_stock(store, product)
        floor = self.safety_floor(store, product)
        if stock is None or floor is None:
            return None
        return max(0.0, stock - floor)

    def target_stock_level(self, store: Any, product: Any) -> float | None:
        return self.target_levels.get(_key(store, product))

    def reorder_point(self, store: Any, product: Any) -> float | None:
        """Operational reorder trigger retained for diagnostics, never a move floor."""
        return self.reorder_levels.get(_key(store, product))

    def planning_shortfall(self, store: Any, product: Any) -> float | None:
        """Destination cap used by the existing candidate-generation policy."""
        key = _key(store, product)
        goal = self.target_levels.get(key)
        stock = self.stock.get(key)
        if goal is not None and stock is not None:
            return max(0.0, goal - stock)
        return self.planning_shortfalls.get(key)

    def demand_std(self, store: Any, product: Any) -> float | None:
        """Measured demand standard deviation, or ``None`` when the file has none."""
        return self.dispersion.get(_key(store, product))


def build_inventory_context(data: Mapping[str, Any] | None) -> InventoryContext:
    """Aggregate stock / demand / safety per (store, product) from inventory.

    Explicit min/safety values take priority.  Only rows without an explicit
    lower-bound policy fall back to the existing ``demand_std × 2`` estimate.
    Reorder points and target levels are retained separately and never substituted
    for the departure floor.
    """
    stock: dict[tuple[str, str], float] = {}
    demand: dict[tuple[str, str], float] = {}
    safety: dict[tuple[str, str], float] = {}
    dispersion: dict[tuple[str, str], float] = {}
    known: set[str] = set()
    inventory = (data or {}).get("inventory") if isinstance(data, Mapping) else None
    if not isinstance(inventory, pd.DataFrame) or inventory.empty:
        return InventoryContext(stock, demand, safety, known, dispersion)

    store_col = next((c for c in ("store_id", "node_id") if c in inventory.columns), None)
    product_col = next((c for c in ("product_id", "item_id") if c in inventory.columns), None)
    if store_col is None or product_col is None:
        return InventoryContext(stock, demand, safety, known, dispersion)
    stock_col = next((c for c in ("stock_qty", "current_stock", "quantity") if c in inventory.columns), None)
    demand_col = next((c for c in ("demand_qty", "avg_daily_sales", "sales_qty") if c in inventory.columns), None)
    demand_std_col = "demand_std" if "demand_std" in inventory.columns else None
    explicit_min: dict[tuple[str, str], list[float]] = {}
    explicit_safety: dict[tuple[str, str], list[float]] = {}
    estimated: dict[tuple[str, str], float] = {}
    target_levels: dict[tuple[str, str], list[float]] = {}
    reorder_levels: dict[tuple[str, str], list[float]] = {}

    for _, row in inventory.iterrows():
        store = str(row.get(store_col))
        known.add(store)
        key = _key(store, row.get(product_col))
        if stock_col is not None:
            value = _num(row.get(stock_col))
            if value is not None:
                stock[key] = stock.get(key, 0.0) + value
        if demand_col is not None:
            value = _num(row.get(demand_col))
            if value is not None:
                # avg_daily_sales is a daily rate; scale to a weekly need proxy.
                weekly = value * 7 if demand_col == "avg_daily_sales" else value
                demand[key] = demand.get(key, 0.0) + weekly
        if demand_std_col is not None:
            std = _num(row.get(demand_std_col))
            if std is not None and std >= 0:
                # ~1 week of demand-std as a light safety cushion (estimate, not policy).
                estimated[key] = estimated.get(key, 0.0) + std * 2.0
                dispersion[key] = dispersion.get(key, 0.0) + std
        minimum = _num(row.get("min_stock")) if "min_stock" in inventory.columns else None
        registered = _num(row.get("safety_stock")) if "safety_stock" in inventory.columns else None
        target_level = _num(row.get("target_stock")) if "target_stock" in inventory.columns else None
        reorder_level = _num(row.get("reorder_point")) if "reorder_point" in inventory.columns else None
        if minimum is not None and minimum >= 0:
            explicit_min.setdefault(key, []).append(minimum)
        if registered is not None and registered >= 0:
            explicit_safety.setdefault(key, []).append(registered)
        if target_level is not None and target_level >= 0:
            target_levels.setdefault(key, []).append(target_level)
        if reorder_level is not None and reorder_level >= 0:
            reorder_levels.setdefault(key, []).append(reorder_level)

    floor_sources: dict[tuple[str, str], str] = {}
    for key in set(estimated) | set(explicit_min) | set(explicit_safety):
        minimums = explicit_min.get(key, [])
        registered = explicit_safety.get(key, [])
        if minimums or registered:
            # Both are lower-bound policies.  Respecting the stricter registered
            # value satisfies each without treating reorder/target levels as floors.
            safety[key] = max([*minimums, *registered])
            if minimums and registered:
                floor_sources[key] = "explicit_combined"
            elif minimums:
                floor_sources[key] = "explicit_min_stock"
            else:
                floor_sources[key] = "explicit_safety_stock"
        else:
            safety[key] = estimated[key]
            floor_sources[key] = "estimated"

    target = {key: max(values) for key, values in target_levels.items() if values}
    reorder = {key: max(values) for key, values in reorder_levels.items() if values}
    # Match ``candidate_generator.generate_candidate_recommendations`` exactly
    # when no explicit target_stock exists.  That service uses product median
    # stock plus seven days of the first available daily-demand column.
    planning_shortfalls: dict[tuple[str, str], float] = {}
    planning_demand_col = next(
        (column for column in ("avg_daily_sales", "sales_qty", "demand_qty") if column in inventory.columns),
        None,
    )
    planning_rows: list[tuple[str, str, float, float]] = []
    for _, row in inventory.iterrows():
        store = str(row.get(store_col))
        product = str(row.get(product_col))
        stock_value = _num(row.get(stock_col)) if stock_col is not None else None
        demand_value = _num(row.get(planning_demand_col)) if planning_demand_col is not None else 0.0
        if stock_value is not None:
            planning_rows.append((store, product, stock_value, demand_value or 0.0))
    if planning_rows:
        planning_frame = pd.DataFrame(planning_rows, columns=["store", "product", "stock", "demand"])
        medians = planning_frame.groupby("product")["stock"].median().to_dict()
        for row in planning_frame.itertuples(index=False):
            key = _key(row.store, row.product)
            planning_shortfalls[key] = max(0.0, float(medians[row.product]) - float(row.stock)) + max(0.0, float(row.demand)) * 7.0
    return InventoryContext(
        stock, demand, safety, known, dispersion, floor_sources, target, reorder, planning_shortfalls,
    )


def evaluate_feasibility(rec: Mapping[str, Any], context: InventoryContext | None = None) -> FeasibilityResult:
    """Classify one candidate as 추천 가능 / 데이터 확인 필요 / 이동 불가.

    Hard violations (이동 불가) are unambiguous and executable-blocking. Missing or
    uncertain inputs downgrade to 데이터 확인 필요 rather than dropping a move on a
    guess. Everything else is 추천 가능.
    """
    source = rec.get("source_id")
    target = rec.get("target_id")
    product = rec.get("product_id")
    route_type = str(rec.get("route_type") or "").upper()
    qty = _num(rec.get("recommended_qty"))

    # --- Hard blocks -------------------------------------------------------- #
    if source is not None and target is not None and str(source) == str(target):
        return FeasibilityResult(STATUS_BLOCKED, "출발지와 도착지가 같습니다.", "same_source_target")
    if qty is None or qty <= 0:
        return FeasibilityResult(STATUS_BLOCKED, "이동 수량이 유효하지 않습니다.", "invalid_quantity")
    if route_type not in VALID_ROUTE_TYPES:
        return FeasibilityResult(STATUS_BLOCKED, "이동 경로 정보가 없습니다.", "no_route")
    if route_type == "VIA_DC" and not str(rec.get("dc_id") or "").strip():
        return FeasibilityResult(STATUS_BLOCKED, "경유 DC 정보가 없습니다.", "via_dc_missing_dc")

    context = context or InventoryContext({}, {}, {}, set())
    source_stock = context.source_stock(source, product)
    if source_stock is not None and qty > source_stock:
        return FeasibilityResult(
            STATUS_BLOCKED,
            "출발 점포 재고보다 이동 수량이 많습니다.",
            "quantity_exceeds_stock",
            {"source_stock": source_stock, "recommended_qty": qty},
        )

    saving = _num(rec.get("expected_saving"))
    cost = _num(rec.get("estimated_cost")) if rec.get("estimated_cost") is not None else _num(rec.get("move_cost"))
    if cost is not None and cost < 0:
        return FeasibilityResult(STATUS_BLOCKED, "이동 비용이 음수입니다.", "negative_cost")
    # Both sides known and the move costs more than it saves: an operator should
    # never see this ranked. Uncomputable inputs are a *soft* check below, because
    # "we cannot tell" is not the same as "we know it is not worth it".
    if saving is not None and cost is not None and (saving - cost) <= 0:
        return FeasibilityResult(
            STATUS_BLOCKED,
            "이동 비용이 예상 효과보다 크거나 같습니다.",
            "non_positive_net_benefit",
            {"expected_saving": saving, "estimated_cost": cost, "net_benefit": saving - cost},
        )

    # --- Soft checks (kept, but flagged) ------------------------------------ #
    if cost is None and _num(rec.get("distance_km")) is None:
        return FeasibilityResult(STATUS_CHECK, "이동 비용을 계산할 수 없습니다.", "cost_uncomputable")
    if saving is None:
        return FeasibilityResult(STATUS_CHECK, "예상 절감액을 계산할 수 없습니다.", "saving_uncomputable")

    if source_stock is None and context.known_stores and str(source) not in context.known_stores:
        return FeasibilityResult(STATUS_CHECK, "출발 점포 재고 데이터를 확인해야 합니다.", "source_stock_missing")

    safety_floor = context.safety_floor(source, product)
    if source_stock is not None and safety_floor is not None and (source_stock - qty) < safety_floor:
        return FeasibilityResult(
            STATUS_BLOCKED,
            "이동 후 재고가 남겨야 할 재고보다 적습니다.",
            "inventory_floor_violation",
            {
                "remaining": source_stock - qty,
                "inventory_floor": safety_floor,
                "inventory_floor_source": context.inventory_floor_source(source, product),
            },
        )
    need = context.target_demand(target, product)
    if need is not None and need > 0 and qty > need * _OVERSUPPLY_MULTIPLE:
        return FeasibilityResult(
            STATUS_CHECK,
            "도착 점포 필요량보다 이동 수량이 많습니다.",
            "oversupply",
            {"target_need": need, "recommended_qty": qty},
        )

    if source_stock is not None and safety_floor is None:
        return FeasibilityResult(
            STATUS_CHECK,
            "남겨야 할 재고 기준을 확인할 수 없습니다.",
            "inventory_floor_unavailable",
        )

    return FeasibilityResult(STATUS_OK, "이동 조건을 충족합니다.", "ok")


def annotate_feasibility(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None = None,
    context: InventoryContext | None = None,
) -> dict[str, Any]:
    """Attach feasibility status/reason to every candidate and split feasible/blocked.

    Exact duplicate route alternatives keep the first occurrence; DIRECT,
    VIA_DC, and different DC choices for the same movement remain separate so
    the execution-plan stage can choose one using their real values. Returns a
    dict with:

    * ``annotated``  — every rec with feasibility_* fields (order preserved)
    * ``feasible``   — recs whose status is not 이동 불가 (used as the final set)
    * ``blocked``    — recs removed from the final set (with reasons)
    * ``summary``    — counts + status distribution for diagnostics/UI
    """
    context = context or build_inventory_context(data)
    annotated: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    counts = {STATUS_OK: 0, STATUS_CHECK: 0, STATUS_BLOCKED: 0}

    for rec in recommendations or []:
        item = dict(rec)
        dup_key = (
            str(rec.get("product_id")),
            str(rec.get("source_id")),
            str(rec.get("target_id")),
            str(rec.get("route_type") or "").upper(),
            str(rec.get("dc_id") or "-"),
        )
        if dup_key in seen:
            result = FeasibilityResult(STATUS_BLOCKED, "동일 추천이 중복되었습니다.", "duplicate")
        else:
            seen.add(dup_key)
            result = evaluate_feasibility(rec, context)
        item.update(result.to_dict())
        counts[result.status] = counts.get(result.status, 0) + 1
        annotated.append(item)
        if result.is_blocked:
            blocked.append(item)
        else:
            feasible.append(item)

    total = len(annotated)
    summary = {
        "total": total,
        "feasible_count": len(feasible),
        "blocked_count": len(blocked),
        "check_count": counts[STATUS_CHECK],
        "ok_count": counts[STATUS_OK],
        "status_distribution": counts,
        "blocked_reasons": sorted({item.get("feasibility_reason") for item in blocked}),
        "all_feasible": len(blocked) == 0,
    }
    return {"annotated": annotated, "feasible": feasible, "blocked": blocked, "summary": summary}
