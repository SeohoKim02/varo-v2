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


@dataclass
class InventoryContext:
    """Store-product lookups built once from the uploaded workbook."""
    stock: dict[tuple[str, str], float]
    demand: dict[tuple[str, str], float]
    safety: dict[tuple[str, str], float]
    known_stores: set[str]

    def source_stock(self, store: Any, product: Any) -> float | None:
        return self.stock.get(_key(store, product))

    def target_demand(self, store: Any, product: Any) -> float | None:
        return self.demand.get(_key(store, product))

    def safety_floor(self, store: Any, product: Any) -> float:
        return self.safety.get(_key(store, product), 0.0)


def build_inventory_context(data: Mapping[str, Any] | None) -> InventoryContext:
    """Aggregate stock / demand / safety per (store, product) from inventory.

    Missing columns simply yield empty lookups; the evaluator then downgrades to
    데이터 확인 필요 instead of guessing. Safety floor is a conservative estimate
    from demand (never fabricated as a hard policy number).
    """
    stock: dict[tuple[str, str], float] = {}
    demand: dict[tuple[str, str], float] = {}
    safety: dict[tuple[str, str], float] = {}
    known: set[str] = set()
    inventory = (data or {}).get("inventory") if isinstance(data, Mapping) else None
    if not isinstance(inventory, pd.DataFrame) or inventory.empty:
        return InventoryContext(stock, demand, safety, known)

    store_col = next((c for c in ("store_id", "node_id") if c in inventory.columns), None)
    product_col = next((c for c in ("product_id", "item_id") if c in inventory.columns), None)
    if store_col is None or product_col is None:
        return InventoryContext(stock, demand, safety, known)
    stock_col = next((c for c in ("stock_qty", "current_stock", "quantity") if c in inventory.columns), None)
    demand_col = next((c for c in ("demand_qty", "avg_daily_sales", "sales_qty") if c in inventory.columns), None)
    demand_std_col = "demand_std" if "demand_std" in inventory.columns else None

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
            if std is not None:
                # ~1 week of demand-std as a light safety cushion (estimate, not policy).
                safety[key] = safety.get(key, 0.0) + std * 2.0
    return InventoryContext(stock, demand, safety, known)


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

    # --- Soft checks (kept, but flagged) ------------------------------------ #
    saving = _num(rec.get("expected_saving"))
    cost = _num(rec.get("estimated_cost")) if rec.get("estimated_cost") is not None else _num(rec.get("move_cost"))
    if cost is None and _num(rec.get("distance_km")) is None:
        return FeasibilityResult(STATUS_CHECK, "이동 비용을 계산할 수 없습니다.", "cost_uncomputable")
    if saving is None:
        return FeasibilityResult(STATUS_CHECK, "예상 절감액을 계산할 수 없습니다.", "saving_uncomputable")

    if source_stock is None and context.known_stores and str(source) not in context.known_stores:
        return FeasibilityResult(STATUS_CHECK, "출발 점포 재고 데이터를 확인해야 합니다.", "source_stock_missing")

    safety_floor = context.safety_floor(source, product)
    if source_stock is not None and (source_stock - qty) < safety_floor:
        return FeasibilityResult(
            STATUS_CHECK,
            "이동 후 출발 점포 재고가 부족할 수 있습니다.",
            "post_move_below_safety",
            {"remaining": source_stock - qty, "safety_floor": safety_floor},
        )

    need = context.target_demand(target, product)
    if need is not None and need > 0 and qty > need * _OVERSUPPLY_MULTIPLE:
        return FeasibilityResult(
            STATUS_CHECK,
            "도착 점포 필요량보다 이동 수량이 많습니다.",
            "oversupply",
            {"target_need": need, "recommended_qty": qty},
        )

    return FeasibilityResult(STATUS_OK, "이동 조건을 충족합니다.", "ok")


def annotate_feasibility(
    recommendations: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach feasibility status/reason to every candidate and split feasible/blocked.

    Duplicate (product, source, target) candidates keep the first occurrence; the
    later duplicates are blocked with a clear reason. Returns a dict with:

    * ``annotated``  — every rec with feasibility_* fields (order preserved)
    * ``feasible``   — recs whose status is not 이동 불가 (used as the final set)
    * ``blocked``    — recs removed from the final set (with reasons)
    * ``summary``    — counts + status distribution for diagnostics/UI
    """
    context = build_inventory_context(data)
    annotated: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    counts = {STATUS_OK: 0, STATUS_CHECK: 0, STATUS_BLOCKED: 0}

    for rec in recommendations or []:
        item = dict(rec)
        dup_key = (str(rec.get("product_id")), str(rec.get("source_id")), str(rec.get("target_id")))
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
