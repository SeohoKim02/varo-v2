"""View-model for the 재고 운영 Workspace — one screen, result first.

The Workspace shows the *decision* (which move to execute today) next to the
network it happens on. This module is the single place that reads real session
state and turns it into what that one screen needs, so the page never re-derives
a ranking, a quantity, or a status of its own.

Rules this module keeps (they are the reason it exists):

* The action list is always ``execution_plan.items`` — the same list the home,
  recommendation, and route-detail pages use. ``planned_qty`` is the action
  quantity; ``recommended_qty`` stays visible only as candidate context.
* Nothing is invented. Distance, travel time, vehicle capacity and real transport
  cost are not part of the collected data yet, so they are reported as 미확보 and
  never rendered as 0. Any value that cannot be computed stays ``None`` and the
  page prints 데이터 없음 / 계산 불가.
* No algorithm names, solver details, ids, or signatures leave this module.

Pure Python: no Streamlit import, so every rule here is unit-testable.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from services.execution_plan import planned_recommendations
from services.home_state import (
    ANALYSIS_PENDING,
    NO_CANDIDATES,
    READY,
    STALE,
    build_home_state,
)

# User-facing wording for values the current data cannot supply.
NO_DATA = "데이터 없음"
NOT_COMPUTABLE = "계산 불가"
CHECK_NEEDED = "확인 필요"

# Internal provenance grades → the four words a user actually reads.
PROVENANCE_LABELS = {
    "actual": "실제 관측",
    "estimated": "추정",
    "reference": "기준값",
    "assumed": "가정",
    "not_available": "미확보",
}

ROUTE_LABELS = {"DIRECT": "직접 이동", "VIA_DC": "DC 경유"}

# Widget keys owned by the Workspace. Registered in app_state.TRANSIENT_VIEW_KEYS
# so a new dataset never leaves a stale filter or selection behind.
WORKSPACE_VIEW_KEYS = (
    "ws_filter_product",
    "ws_filter_source",
    "ws_filter_target",
    "ws_filter_route_type",
    "ws_only_actionable",
    "ws_only_attention",
    "ws_network_scope",
    "ws_plan_pick",
)

ALL = "전체"

# The single wording for a move that needs a second look before it is executed.
ATTENTION = "주의"


# --------------------------------------------------------------------------- #
# Small value helpers (never turn missing data into a number)
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _has_rows(value: Any) -> bool:
    if value is None:
        return False
    empty = getattr(value, "empty", None)
    if empty is not None:
        return not bool(empty)
    try:
        return len(value) > 0
    except TypeError:
        return False


def qty_text(value: Any, suffix: str = "개") -> str:
    number = _num(value)
    if number is None:
        return NO_DATA
    return f"{int(round(number)):,}{suffix}"


def money_text(value: Any) -> str:
    number = _num(value)
    if number is None:
        return NO_DATA
    return f"{number:,.0f}원"


def route_label(item: Mapping[str, Any] | None) -> str:
    if not item:
        return "-"
    label = ROUTE_LABELS.get(_text(item.get("route_type")).upper(), CHECK_NEEDED)
    if _text(item.get("route_type")).upper() == "VIA_DC":
        dc = _text(item.get("dc_name")) or _text(item.get("dc_id"))
        if dc:
            return f"{label} · {dc}"
    return label


def move_title(item: Mapping[str, Any] | None) -> str:
    if not item:
        return "-"
    source = _text(item.get("source_name")) or _text(item.get("source_id")) or "-"
    target = _text(item.get("target_name")) or _text(item.get("target_id")) or "-"
    return f"{source} → {target}"


def action_qty(item: Mapping[str, Any] | None) -> float | None:
    """The quantity a user acts on: planned_qty, falling back to the candidate."""
    if not item:
        return None
    planned = _num(item.get("planned_qty"))
    return planned if planned is not None else _num(item.get("recommended_qty"))


def needs_attention(item: Mapping[str, Any] | None) -> bool:
    """True when this move should not be executed without a second look.

    Same three conditions the right panel and the plan list already describe in
    words — collected here so 주의 means exactly one thing on every surface.
    """
    if not item:
        return False
    if item.get("quantity_adjusted"):
        return True
    status = _text(item.get("feasibility_status"))
    if status and status != "추천 가능":
        return True
    stability = _text(item.get("robustness_status"))
    return bool(stability) and stability not in ("안정", "높음")


def plan_list_rows(
    items: Sequence[Mapping[str, Any]], selected_route_id: Any = None,
) -> list[dict[str, Any]]:
    """The compact 실행 계획 list: one short line per move, plan order preserved.

    Deliberately narrow — 출발 → 도착 · 상품 as the line a user scans, and one
    caption with 실행 수량 · 경로 (+ 주의). The product is part of the *label*, not
    the caption, because a leg on its own is not a move: the same 출발 → 도착 can
    carry two different products and the two rows must not read alike. Candidate
    scores, ids and cost breakdowns stay out of the list; they belong to the right
    panel and the 세부 정보 tab.
    """
    selected = _text(selected_route_id)
    rows: list[dict[str, Any]] = []
    for item in items or []:
        route_id = _text(item.get("route_id"))
        if not route_id:
            continue
        attention = needs_attention(item)
        product = _text(item.get("product_name")) or _text(item.get("product_id")) or "-"
        caption = f"{qty_text(action_qty(item))} · {route_label(item)}"
        rows.append({
            "route_id": route_id,
            "label": f"{move_title(item)} · {product}",
            "caption": caption + (f" · {ATTENTION}" if attention else ""),
            "product": product,
            "qty_text": qty_text(action_qty(item)),
            "route": route_label(item),
            "attention": attention,
            "selected": route_id == selected,
        })
    return rows


# --------------------------------------------------------------------------- #
# Selected move — the values the (now folded-away) 경로 상세 screen used to own
# --------------------------------------------------------------------------- #
def _row(label: str, value: str) -> dict[str, str]:
    return {"항목": label, "값": value}


def move_detail_rows(
    item: Mapping[str, Any] | None,
    quantity_basis: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Every decision number for the selected move, or 데이터 없음.

    The stock figures come from what the analysis already computed — the
    candidate's own decision metrics, and, where the plan item does not carry
    them, the candidate ledger's ``quantity_basis`` (the very dict the 이동 수량
    근거 line below this table prints). Reading both means the table and that
    line cannot state a different 출발 현재 재고. Nothing is recomputed here.

    The one derived value is 이동 후 재고, taken from the *action* quantity
    (planned_qty) rather than the candidate's recommended quantity, so it can
    never contradict the 실행 수량 shown right above it.
    """
    if not item:
        return []
    basis = quantity_basis or {}

    def value(*keys: str) -> float | None:
        for key in keys:
            found = _num(item.get(key))
            if found is None:
                found = _num(basis.get(key))
            if found is not None:
                return found
        return None

    quantity = action_qty(item)
    stock = value("source_stock")
    safety = value("source_safety_floor", "source_safety", "inventory_floor_value")
    movable = value("source_movable", "available_to_move")
    shortfall = value("target_shortfall", "target_demand")
    remaining = None if stock is None or quantity is None else stock - quantity
    target_stock = _num(item.get("target_stock"))
    post_target = None if target_stock is None or quantity is None else target_stock + quantity

    dc = _text(item.get("dc_name")) or _text(item.get("dc_id"))
    net = item.get("planned_net_benefit")
    if net is None:
        net = item.get("net_benefit")
    return [
        _row("출발 점포", _text(item.get("source_name")) or _text(item.get("source_id")) or NO_DATA),
        _row("도착 점포", _text(item.get("target_name")) or _text(item.get("target_id")) or NO_DATA),
        _row("경유 DC", dc or "경유 없음"),
        _row("경로 유형", route_label(item)),
        _row("실행 수량", qty_text(quantity)),
        _row("출발 현재재고", qty_text(stock)),
        _row("유지해야 할 재고", qty_text(safety)),
        _row("이동 가능 수량", qty_text(movable)),
        _row("이동 후 출발 재고", qty_text(remaining)),
        _row("도착 점포 필요량", qty_text(shortfall)),
        _row("이동 후 도착 재고", qty_text(post_target)),
        _row("예상 비용", money_text(item.get("planned_cost") if item.get("planned_cost") is not None else item.get("estimated_cost"))),
        _row("예상 절감", money_text(
            item.get("planned_expected_saving")
            if item.get("planned_expected_saving") is not None else item.get("expected_saving")
        )),
        _row("예상 순효과", money_text(net)),
        _row("안정성", _text(item.get("robustness_status")) or CHECK_NEEDED),
        _row("실행 상태", _text(item.get("feasibility_status")) or "추천 가능"),
    ]


# The logistics facts still being collected, and the exact optional fields a
# future real-data connection will fill in.
#
#   (표시 이름, 실제값 필드, 참고로만 쓰는 입력값 필드, 단위)
#
# ``actual`` fields are measured facts — a real road distance, a driven time, an
# invoiced transport cost. ``reference`` fields are numbers that came in with the
# workbook: the model reads them, but they are *not* the measured fact and must
# never be shown under the 실제 label. Until an actual field carries a value the
# row reads 미확보; it is never rendered as 0, and never borrows the workbook
# number to look complete.
LOGISTICS_FIELDS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...] = (
    ("실제 도로 거리", ("actual_distance_km", "road_distance_km"), ("distance_km",), "km"),
    ("실제 이동 시간", ("actual_travel_time_min",), ("travel_time_min", "expected_time_min"), "분"),
    ("차량 용량", ("actual_vehicle_capacity",), ("vehicle_capacity", "capacity", "max_load"), "개"),
    ("실제 운송비", ("actual_transport_cost",), ("transport_cost", "estimated_cost", "move_cost"), "원"),
)


def _first_number(item: Mapping[str, Any] | None, keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _num((item or {}).get(key))
        if value is not None:
            return value
    return None


def _measure_text(value: float, suffix: str) -> str:
    if suffix == "원":
        return money_text(value)
    if suffix == "개":
        return qty_text(value)
    text = f"{value:,.1f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def logistics_rows(item: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """실제 물류 값 — 확보된 것만 숫자로, 나머지는 미확보.

    See :data:`LOGISTICS_FIELDS`. A workbook value never fills an 실제 row: when
    one exists it is named in 설명 as an input the model used, so a reader can
    tell "we have a 4.7km number in the file" from "we measured 4.7km of road".
    """
    missing = PROVENANCE_LABELS["not_available"]
    rows: list[dict[str, str]] = []
    for label, actual_keys, reference_keys, suffix in LOGISTICS_FIELDS:
        actual = _first_number(item, actual_keys)
        if actual is not None:
            rows.append({
                "항목": label, "값": _measure_text(actual, suffix), "설명": "실제 측정값",
            })
            continue
        reference = _first_number(item, reference_keys)
        note = "수집 중"
        if reference is not None:
            note = f"수집 중 (입력값 {_measure_text(reference, suffix)} 사용)"
        rows.append({"항목": label, "값": missing, "설명": note})
    return rows


# --------------------------------------------------------------------------- #
# Store status used by both the Workspace network and the legacy home network
# --------------------------------------------------------------------------- #
def store_inventory_states(
    data: Mapping[str, Any] | None,
    recommendations: Sequence[Mapping[str, Any]] | None = (),
    highlight_limit: int = 3,
) -> dict[str, str]:
    """Per-store 과잉 / 부족 / 정상, with the top moves' sources marked 이동 대상."""
    import pandas as pd

    states: dict[str, str] = {}
    inventory = (data or {}).get("inventory")
    if isinstance(inventory, pd.DataFrame) and not inventory.empty and "store_id" in inventory.columns:
        stock = pd.to_numeric(inventory.get("stock_qty"), errors="coerce")
        demand = pd.to_numeric(inventory.get("demand_qty"), errors="coerce")
        if demand is None or demand.isna().all():
            demand = pd.to_numeric(inventory.get("sales_30d"), errors="coerce")
        dead = pd.to_numeric(inventory.get("dead_stock_qty"), errors="coerce")
        frame = pd.DataFrame({
            "store_id": inventory["store_id"].astype(str),
            "stock": stock, "demand": demand, "dead": dead,
        })
        grouped = frame.groupby("store_id").sum(min_count=1)
        for store_id, row in grouped.iterrows():
            total_stock = float(row.get("stock") or 0.0)
            total_demand = float(row.get("demand") or 0.0)
            total_dead = float(row.get("dead") or 0.0)
            ratio = total_stock / total_demand if total_demand > 0 else (2.0 if total_stock > 0 else 1.0)
            if (total_stock > 0 and total_dead / total_stock >= 0.30) or ratio >= 1.5:
                states[str(store_id)] = "과잉"
            elif ratio <= 0.7:
                states[str(store_id)] = "부족"
            else:
                states[str(store_id)] = "정상"
    for route in list(recommendations or [])[:highlight_limit]:
        source_id = _text(route.get("source_id"))
        if source_id:
            states[source_id] = "이동 대상"
    return states


# --------------------------------------------------------------------------- #
# Data readiness / provenance (§ real data collection is still in progress)
# --------------------------------------------------------------------------- #
def _column_present(frame: Any, *names: str) -> bool:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return False
    for name in names:
        if name in frame.columns and pd.to_numeric(frame[name], errors="coerce").notna().any():
            return True
    return False


def data_readiness(state: Mapping[str, Any], history_confirmed: int = 0) -> list[dict[str, str]]:
    """What the applied data really contains, per logistics fact.

    Values that came in with the workbook are 기준값 — they are inputs the model
    reads, not measured road distances or invoiced transport costs. Facts that are
    still being collected are 미확보 and must never be shown as 0.
    """
    data = state.get("varo_data") if isinstance(state.get("varo_data"), Mapping) else {}
    inventory = data.get("inventory")
    routes = data.get("routes")

    def row(label: str, grade: str, note: str) -> dict[str, str]:
        return {"항목": label, "상태": PROVENANCE_LABELS[grade], "설명": note, "grade": grade}

    rows: list[dict[str, str]] = []
    rows.append(
        row("판매·재고", "actual", "적용된 재고·판매 데이터를 그대로 사용합니다.")
        if _column_present(inventory, "stock_qty")
        else row("판매·재고", "not_available", "재고 수량 데이터가 없습니다.")
    )
    rows.append(
        row("안전재고 기준", "actual", "데이터에 있는 안전재고 값을 사용합니다.")
        if _column_present(inventory, "safety_stock", "min_stock", "reorder_point", "target_stock")
        else row("안전재고 기준", "estimated", "수요 변동으로 남겨야 할 재고를 추정합니다.")
    )
    rows.append(
        row("점포 간 이동경로", "reference", "데이터에 등록된 이동 가능 경로입니다.")
        if _has_rows(routes)
        else row("점포 간 이동경로", "not_available", "이동 가능 경로 정보가 없습니다.")
    )
    rows.append(
        row("이동 거리", "reference", "데이터에 입력된 거리 값이며 실제 도로 거리는 수집 중입니다.")
        if _column_present(routes, "distance_km")
        else row("이동 거리", "not_available", "실제 도로 거리는 아직 수집 중입니다.")
    )
    rows.append(
        row("이동 시간", "reference", "데이터에 입력된 소요 시간 값입니다.")
        if _column_present(routes, "travel_time_min", "expected_time_min")
        else row("이동 시간", "not_available", "실제 이동 시간은 아직 수집 중입니다.")
    )
    rows.append(
        row("운송비", "reference", "데이터에 입력된 이동비용 값이며 실제 청구 운송비는 수집 중입니다.")
        if _column_present(routes, "estimated_cost", "transport_cost", "move_cost")
        else row("운송비", "not_available", "실제 운송비는 아직 수집 중입니다.")
    )
    rows.append(
        row("차량 용량", "reference", "데이터에 입력된 차량 용량 값입니다.")
        if _column_present(routes, "vehicle_capacity", "capacity", "max_load")
        else row("차량 용량", "not_available", "차량 용량 정보는 아직 수집 중입니다.")
    )
    rows.append(
        row("실제 거점간 이동이력", "actual", "실행 이력에 기록된 실제 결과가 있습니다.")
        if int(_num(history_confirmed) or 0) > 0
        else row("실제 거점간 이동이력", "not_available", "실제 이동 결과는 실행 이력에 기록되면 반영됩니다.")
    )
    return rows


def readiness_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """One short line for the left panel: 준비 정도 + 미확보 항목 수."""
    missing = [str(row.get("항목")) for row in rows if row.get("grade") == "not_available"]
    actual = [str(row.get("항목")) for row in rows if row.get("grade") == "actual"]
    return {
        "missing": missing,
        "missing_count": len(missing),
        "actual": actual,
        "headline": (
            "데이터 준비됨" if not missing
            else f"데이터 준비됨 · 미확보 {len(missing)}개"
        ),
    }


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def filter_options(items: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    def values(*keys: str) -> list[str]:
        found = set()
        for item in items:
            for key in keys:
                value = _text(item.get(key))
                if value:
                    found.add(value)
                    break
        return [ALL] + sorted(found)

    return {
        "product": values("product_name", "product_id"),
        "source": values("source_name", "source_id"),
        "target": values("target_name", "target_id"),
        "route_type": [ALL, ROUTE_LABELS["DIRECT"], ROUTE_LABELS["VIA_DC"]],
    }


def apply_filters(
    items: Sequence[Mapping[str, Any]], filters: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Filter the plan list without ever re-ordering it (plan_rank is the order)."""
    filters = filters or {}
    result = [dict(item) for item in items]

    def keep(item: Mapping[str, Any], keys: tuple[str, ...], wanted: Any) -> bool:
        if not wanted or wanted == ALL:
            return True
        for key in keys:
            value = _text(item.get(key))
            if value:
                return value == str(wanted)
        return False

    result = [i for i in result if keep(i, ("product_name", "product_id"), filters.get("product"))]
    result = [i for i in result if keep(i, ("source_name", "source_id"), filters.get("source"))]
    result = [i for i in result if keep(i, ("target_name", "target_id"), filters.get("target"))]
    wanted_route = filters.get("route_type")
    if wanted_route and wanted_route != ALL:
        reverse = {label: code for code, label in ROUTE_LABELS.items()}
        code = reverse.get(str(wanted_route))
        result = [i for i in result if _text(i.get("route_type")).upper() == code]
    if filters.get("only_actionable"):
        result = [i for i in result if (_num(i.get("planned_net_benefit")) or 0) > 0]
    if filters.get("only_attention"):
        result = [i for i in result if needs_attention(i)]
    return result


# --------------------------------------------------------------------------- #
# KPI row (four cards, real values only)
# --------------------------------------------------------------------------- #
def plan_kpis(plan: Mapping[str, Any] | None, items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    plan = plan if isinstance(plan, Mapping) else {}
    attention = int(_num(plan.get("adjusted_candidates")) or 0) + len(plan.get("unselected_candidates") or [])
    total_qty = _num(plan.get("total_transfer_qty"))
    net = _num(plan.get("total_net_benefit"))
    return [
        {
            "title": "오늘 실행 이동",
            "value": f"{len(items):,}건",
            "caption": "지금 실행할 수 있는 이동 수",
        },
        {
            "title": "총 이동 수량",
            "value": qty_text(total_qty) if total_qty is not None else NO_DATA,
            "caption": "공유 재고를 함께 반영한 실행 수량",
        },
        {
            "title": "예상 순효과",
            "value": money_text(net) if net is not None else NOT_COMPUTABLE,
            "caption": "이동 비용을 뺀 기대 효과",
        },
        {
            "title": "주의 필요",
            "value": f"{attention:,}건",
            "caption": "수량이 조정되었거나 계획에서 빠진 이동",
        },
    ]


# --------------------------------------------------------------------------- #
# Selected-move detail: reasons, risks, alternatives
# --------------------------------------------------------------------------- #
def move_reasons(
    item: Mapping[str, Any] | None, ledger_record: Mapping[str, Any] | None = None,
) -> list[str]:
    """Two to four short reasons, each backed by a value that really exists."""
    if not item:
        return []
    reasons: list[str] = []
    for sentence in (ledger_record or {}).get("recommendation_reasons") or []:
        text = _text(sentence)
        if text and text not in reasons:
            reasons.append(text)

    quantity = action_qty(item)
    shortfall = _num(item.get("target_shortfall"))
    if len(reasons) < 4 and quantity is not None and shortfall is not None and shortfall > 0:
        line = f"도착 점포의 부족 재고를 {int(round(min(quantity, shortfall))):,}개 줄입니다."
        if line not in reasons:
            reasons.append(line)

    movable = _num(item.get("source_movable"))
    if len(reasons) < 4 and quantity is not None and movable is not None and movable >= quantity:
        line = "출발 점포의 안전재고를 유지한 채 이동할 수 있습니다."
        if line not in reasons:
            reasons.append(line)

    net = _num(item.get("planned_net_benefit"))
    if net is None:
        net = _num(item.get("net_benefit"))
    if len(reasons) < 4 and net is not None and net > 0:
        line = f"이동 비용을 뺀 예상 순효과가 {net:,.0f}원입니다."
        if line not in reasons:
            reasons.append(line)

    stability = _text(item.get("robustness_status"))
    if len(reasons) < 4 and stability in ("안정", "높음"):
        line = "조건이 조금 달라져도 추천이 유지되는 이동입니다."
        if line not in reasons:
            reasons.append(line)

    if not reasons and _text(item.get("reason")):
        reasons.append(_text(item.get("reason")))
    return reasons[:4]


def move_risks(
    item: Mapping[str, Any] | None,
    readiness: Sequence[Mapping[str, str]] = (),
) -> list[str]:
    """Only risks a user can act on; internal warning codes stay hidden."""
    if not item:
        return []
    risks: list[str] = []
    scenario = _text(item.get("demand_scenario_status"))
    if scenario in ("변동 가능성 큼", "확인 필요"):
        risks.append("도착 점포의 수요 변동이 큽니다.")

    quantity = action_qty(item)
    movable = _num(item.get("source_movable"))
    if quantity is not None and movable is not None and movable > 0 and quantity >= movable * 0.9:
        risks.append("출발 점포의 안전재고 여유가 적습니다.")

    if item.get("quantity_adjusted"):
        risks.append("다른 이동과 재고를 함께 고려해 수량이 조정되었습니다.")

    status = _text(item.get("feasibility_status"))
    if status and status != "추천 가능":
        risks.append("실행 조건을 한 번 더 확인해야 합니다.")

    stability = _text(item.get("robustness_status"))
    if stability and stability not in ("안정", "높음"):
        risks.append("조건이 달라지면 추천이 바뀔 수 있습니다.")

    missing = [str(row.get("항목")) for row in readiness or [] if row.get("grade") == "not_available"]
    logistics = [name for name in missing if name in ("이동 거리", "이동 시간", "운송비", "차량 용량")]
    if logistics:
        risks.append(f"실제 운송 정보({', '.join(logistics)})가 아직 확보되지 않았습니다.")
    return risks[:5]


def _exclusion_reasons(plan: Mapping[str, Any] | None) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for row in (plan or {}).get("unselected_candidates") or []:
        route_id = _text(row.get("route_id"))
        if route_id:
            reasons[route_id] = _text(row.get("reason"))
    return reasons


def alternatives_for(
    item: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    """Real alternatives for the selected move, on the same product.

    Three comparisons a user actually faces, in the order they matter:

    * 같은 구간 · 다른 경로 — the same 출발 → 도착 by 직접 이동 vs DC 경유, or via a
      different DC. This is the choice the plan itself had to make.
    * 같은 도착 점포 — a different source that could fill the same shortage.
    * 같은 출발 재고 — a different destination the same stock could go to.

    Only candidates that really exist in the current data appear; nothing is
    synthesised, and product substitution is deliberately not attempted. Values
    the data cannot supply read 데이터 없음.
    """
    if not item:
        return []
    product = _text(item.get("product_id")) or _text(item.get("product_name"))
    target = _text(item.get("target_id")) or _text(item.get("target_name"))
    source = _text(item.get("source_id")) or _text(item.get("source_name"))
    planned_ids = {_text(row.get("route_id")) for row in (plan or {}).get("items") or []}
    excluded = _exclusion_reasons(plan)
    selected_id = _text(item.get("route_id"))

    rows: list[dict[str, str]] = []
    for candidate in candidates or []:
        candidate_product = _text(candidate.get("product_id")) or _text(candidate.get("product_name"))
        candidate_target = _text(candidate.get("target_id")) or _text(candidate.get("target_name"))
        candidate_source = _text(candidate.get("source_id")) or _text(candidate.get("source_name"))
        if candidate_product != product:
            continue
        if candidate_source == source and candidate_target == target:
            kind = "같은 구간 · 다른 경로"
        elif candidate_target == target:
            kind = "같은 도착 점포"
        elif candidate_source == source:
            kind = "같은 출발 재고"
        else:
            continue
        route_id = _text(candidate.get("route_id"))
        in_plan = route_id in planned_ids
        if in_plan:
            source_row = next(
                (row for row in (plan or {}).get("items") or [] if _text(row.get("route_id")) == route_id),
                candidate,
            )
        else:
            source_row = candidate
        if route_id == selected_id:
            status = "현재 선택"
        elif in_plan:
            status = "계획에 포함"
        else:
            status = excluded.get(route_id) or "계획에서 제외"
        rows.append({
            "route_id": route_id,
            "선택": "●" if route_id == selected_id else "",
            "구분": "현재 이동" if route_id == selected_id else kind,
            "출발 점포": _text(candidate.get("source_name")) or _text(candidate.get("source_id")) or "-",
            "도착 점포": _text(candidate.get("target_name")) or _text(candidate.get("target_id")) or "-",
            "경로": route_label(candidate),
            "수량": qty_text(action_qty(source_row)),
            "예상 비용": money_text(source_row.get("planned_cost") if in_plan else candidate.get("estimated_cost")),
            "예상 효과": money_text(
                source_row.get("planned_expected_saving") if in_plan else candidate.get("expected_saving")
            ),
            "예상 순효과": money_text(
                source_row.get("planned_net_benefit") if in_plan else candidate.get("net_benefit")
            ),
            "안정성": _text(candidate.get("robustness_status")) or NO_DATA,
            "계획 반영": status,
        })
    order = {"현재 이동": 0, "같은 구간 · 다른 경로": 1, "같은 도착 점포": 2, "같은 출발 재고": 3}
    rows.sort(key=lambda row: (
        order.get(row["구분"], 4),
        row["계획 반영"] != "계획에 포함",
        row["출발 점포"],
        row["도착 점포"],
        row["경로"],
    ))
    return rows


# --------------------------------------------------------------------------- #
# What-if / 검증 (existing results only — no new optimisation is run here)
# --------------------------------------------------------------------------- #
def whatif_rows(pipeline: Mapping[str, Any] | None, item: Mapping[str, Any] | None) -> list[dict[str, str]]:
    pipeline = pipeline if isinstance(pipeline, Mapping) else {}
    stability = pipeline.get("stability_analysis_status") or {}
    sensitivity = pipeline.get("weight_sensitivity_analysis") or {}
    retention = _num(sensitivity.get("top1_retention_rate"))

    if retention is None:
        weight_answer = NOT_COMPUTABLE
    elif retention >= 0.999:
        weight_answer = "1순위 추천 유지"
    else:
        weight_answer = f"1순위 유지 {retention * 100:.0f}%"

    scenario = _text((item or {}).get("demand_scenario_status")) or NOT_COMPUTABLE
    stability_status = _text(stability.get("status")) or NOT_COMPUTABLE
    plan = pipeline.get("execution_plan") or {}
    adjusted = int(_num(plan.get("adjusted_candidates")) or 0)
    return [
        {
            "조건": "수요가 달라지면",
            "결과": scenario,
            "설명": "선택한 이동의 도착 점포 수요 변동 폭 기준입니다.",
        },
        {
            "조건": "판단 기준이 달라지면",
            "결과": weight_answer,
            "설명": "여러 판단 기준 조합에서 1순위 추천이 유지되는지 확인합니다.",
        },
        {
            "조건": "추천 전체 안정성",
            "결과": stability_status,
            "설명": "추천 순위가 조건 변화에 얼마나 견디는지입니다.",
        },
        {
            "조건": "안전재고를 지키면",
            "결과": f"{adjusted:,}건 수량 조정" if adjusted else "조정 없음",
            "설명": "안전재고와 도착 필요 수량을 함께 지키며 조정한 이동 수입니다.",
        },
    ]


def validation_rows(pipeline: Mapping[str, Any] | None) -> list[dict[str, str]]:
    pipeline = pipeline if isinstance(pipeline, Mapping) else {}
    plan = pipeline.get("execution_plan") or {}
    validation = plan.get("validation") or {}
    stability = pipeline.get("stability_analysis_status") or {}
    confidence = pipeline.get("confidence_status") or {}
    pareto = pipeline.get("pareto_analysis") or {}
    comparison = pipeline.get("plan_comparison") or {}
    greedy = comparison.get("constrained_greedy") or {}
    optimized = comparison.get("vhs_optimized_plan") or {}

    def ok(flag: bool) -> str:
        return "이상 없음" if flag else CHECK_NEEDED

    rows = [
        {"검증 항목": "계획 제약", "결과": ok(bool(validation.get("valid"))),
         "설명": "실행 수량·경로·중복 조건을 다시 계산해 확인했습니다."},
        {"검증 항목": "안전재고", "결과": ok(int(_num(validation.get("safety_stock_violations")) or 0) == 0),
         "설명": "출발 점포에 남겨야 할 재고를 침범하지 않았는지 확인합니다."},
        {"검증 항목": "도착 필요 수량", "결과": ok(int(_num(validation.get("destination_overfill_violations")) or 0) == 0),
         "설명": "도착 점포가 필요한 양보다 많이 받지 않는지 확인합니다."},
        {"검증 항목": "안정성", "결과": _text(stability.get("status")) or NOT_COMPUTABLE,
         "설명": "판단 기준이 달라져도 추천이 유지되는 정도입니다."},
        {"검증 항목": "추천 신뢰도", "결과": _text(confidence.get("status")) or NOT_COMPUTABLE,
         "설명": "입력 완성도와 실행 가능성을 함께 본 결과입니다."},
    ]
    net_optimized = _num(optimized.get("total_net_benefit"))
    net_greedy = _num(greedy.get("total_net_benefit"))
    if net_optimized is not None and net_greedy is not None:
        if net_optimized > net_greedy:
            verdict = "단순 이익 순 선택보다 순효과가 큽니다."
        elif net_optimized == net_greedy:
            verdict = "단순 이익 순 선택과 순효과가 같습니다."
        else:
            verdict = "단순 이익 순 선택과 비교가 필요합니다."
        rows.append({"검증 항목": "다른 선택 방식과 비교", "결과": verdict,
                     "설명": f"현재 계획 {net_optimized:,.0f}원 · 단순 이익 순 {net_greedy:,.0f}원"})
    else:
        rows.append({"검증 항목": "다른 선택 방식과 비교", "결과": NOT_COMPUTABLE,
                     "설명": "비교할 수 있는 계획 결과가 없습니다."})
    front = _num(pareto.get("front_size"))
    total = _num(pareto.get("candidate_count"))
    rows.append({
        "검증 항목": "목표 균형",
        "결과": (
            f"균형 후보 {int(front):,} / {int(total):,}건"
            if front is not None and total else NOT_COMPUTABLE
        ),
        "설명": "비용·효과·위험을 동시에 만족하는 후보 수입니다.",
    })
    return rows


# --------------------------------------------------------------------------- #
# The one workspace view
# --------------------------------------------------------------------------- #
def _pipeline(state: Mapping[str, Any]) -> dict[str, Any]:
    value = state.get("analysis_result") or state.get("varo_pipeline_result")
    return dict(value) if isinstance(value, Mapping) else {}


def resolve_selection(items: Sequence[Mapping[str, Any]], selected_route_id: Any) -> str | None:
    """The one selected plan item id, always inside the current plan."""
    ids = [_text(item.get("route_id")) for item in items if _text(item.get("route_id"))]
    if not ids:
        return None
    current = _text(selected_route_id)
    return current if current in ids else ids[0]


def build_workspace_view(state: Mapping[str, Any], history_confirmed: int = 0) -> dict[str, Any]:
    """Everything the Workspace screen needs, resolved once (never raises)."""
    try:
        return _build_workspace(state, history_confirmed)
    except Exception:  # pragma: no cover - defensive: the workspace must not crash
        home = build_home_state(state)
        return {
            "home": home,
            "state_code": home.get("state_code"),
            "ready": False,
            "stale": False,
            "analysis_pending": False,
            "data_status": CHECK_NEEDED,
            "analysis_status": CHECK_NEEDED,
            "pipeline": {},
            "plan": {},
            "plan_items": [],
            "selected_route_id": None,
            "selected": None,
            "kpis": [],
            "readiness": [],
            "readiness_summary": {"missing": [], "missing_count": 0, "actual": [], "headline": CHECK_NEEDED},
            "plan_message": "",
        }


def _build_workspace(state: Mapping[str, Any], history_confirmed: int = 0) -> dict[str, Any]:
    home = build_home_state(state)
    pipeline = _pipeline(state)
    plan = pipeline.get("execution_plan") if isinstance(pipeline.get("execution_plan"), Mapping) else {}
    items = planned_recommendations(pipeline)
    state_code = home.get("state_code")

    readiness = data_readiness(state, history_confirmed)
    selected_id = resolve_selection(items, state.get("selected_route_id"))
    selected = next(
        (dict(item) for item in items if _text(item.get("route_id")) == selected_id), None
    )

    if state_code == ANALYSIS_PENDING:
        analysis_status = "분석 필요"
    elif state_code == STALE:
        analysis_status = "다시 분석 필요"
    elif state_code == READY:
        analysis_status = "분석 완료"
    elif state_code == NO_CANDIDATES:
        analysis_status = "분석 완료 · 실행 이동 없음"
    else:
        analysis_status = _text(home.get("analysis_status")) or "미실행"

    return {
        "home": home,
        "state_code": state_code,
        "ready": state_code == READY and bool(items),
        "stale": state_code == STALE,
        "analysis_pending": state_code == ANALYSIS_PENDING,
        "data_status": _text(home.get("data_status")) or CHECK_NEEDED,
        "analysis_status": analysis_status,
        "pipeline": pipeline,
        "plan": dict(plan),
        "plan_items": items,
        "selected_route_id": selected_id,
        "selected": selected,
        "kpis": plan_kpis(plan, items),
        "readiness": readiness,
        "readiness_summary": readiness_summary(readiness),
        "plan_message": _text(plan.get("user_message")),
    }


__all__ = [
    "ALL",
    "ATTENTION",
    "CHECK_NEEDED",
    "LOGISTICS_FIELDS",
    "NOT_COMPUTABLE",
    "NO_DATA",
    "PROVENANCE_LABELS",
    "ROUTE_LABELS",
    "WORKSPACE_VIEW_KEYS",
    "action_qty",
    "alternatives_for",
    "apply_filters",
    "build_workspace_view",
    "data_readiness",
    "filter_options",
    "logistics_rows",
    "money_text",
    "move_detail_rows",
    "move_reasons",
    "move_risks",
    "move_title",
    "needs_attention",
    "plan_kpis",
    "plan_list_rows",
    "qty_text",
    "readiness_summary",
    "resolve_selection",
    "route_label",
    "store_inventory_states",
    "validation_rows",
    "whatif_rows",
]
