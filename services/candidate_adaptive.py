"""Observed-only, candidate-workspace planning; never an operating action API.

The existing finite economic solver is reused. Plans do not alter inventory:
only timestamped observations do. Replanning is supported at declared period
boundaries; partial-period forecasts are not silently treated as remaining
demand. Known commitments and in-transit lots survive horizon rebasing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
import time
from typing import Any, Mapping, Sequence

from services.candidate_economics import CommittedTransfer, EconomicProblem, _canonical
from services.candidate_economics_extended import solve_extended, validate_extension

LOGGER = logging.getLogger(__name__)
VERSION = "candidate-observed-replanning-v0.1"
POLICIES = ("OPEN_LOOP", "ROLLING_HORIZON", "MULTISTAGE_RECOURSE")
KINDS = {"SALE", "RECEIPT", "OUTBOUND", "STOCK_COUNT", "DISPOSAL",
         "TRANSFER_DEPARTED", "TRANSFER_ARRIVED", "SIGNAL"}


@dataclass(frozen=True)
class Observation:
    event_id: str
    kind: str
    occurred_at: str
    known_at: str
    product_id: str = ""
    location: str = ""
    state: str = ""
    lot_id: str = ""
    unit: str = ""
    quantity: float = 0.0
    source: str = ""
    correction_of: str = ""
    route_id: str = ""
    price_level: str = ""
    signal: str = ""
    reference_id: str = ""
    vehicle_count: int = 1
    cost_already_paid: bool = False


class ObservationError(ValueError):
    def __init__(self, code: str, message: str, event_id: str = ""):
        super().__init__(message)
        self.code, self.message, self.event_id = code, message, event_id


def _date(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ObservationError("TIMEZONE_REQUIRED", "관측 시각과 결정 시각에 시간대를 함께 입력해 주세요.")
    return result.astimezone(timezone.utc)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def _error(code, message, event_id=""):
    raise ObservationError(code, message, event_id)


def _observed(p, observations, origin, now):
    """Return an auditable active ledger, excluding information not yet known."""
    lots = {x.lot_id: x for x in p.lots}
    by_id, visible, deferred = {}, [], []
    for raw in observations:
        event = raw if isinstance(raw, Observation) else Observation(**raw)
        if not event.event_id or event.event_id in by_id:
            _error("DUPLICATE_OBSERVATION", "관측 기록의 번호가 비어 있거나 중복되어 있습니다.", event.event_id)
        by_id[event.event_id] = event
        occurred, known = _date(event.occurred_at), _date(event.known_at)
        if known > now:
            deferred.append(event.event_id)
            continue
        if known < occurred:
            _error("KNOWN_BEFORE_OCCURRED", "기록을 확인한 시각은 실제 발생 시각보다 빠를 수 없습니다.", event.event_id)
        if occurred > now:
            deferred.append(event.event_id)
            continue
        if occurred < origin:
            _error("BEFORE_SNAPSHOT", "초기 재고 시점 이전 기록입니다. 초기 자료에 반영한 뒤 다시 입력해 주세요.", event.event_id)
        if event.kind not in KINDS or not isinstance(event.source,str) or not event.source.strip():
            _error("OBSERVATION_SOURCE_REQUIRED", "관측 종류와 확인 출처를 입력해 주세요.", event.event_id)
        if event.kind == "SIGNAL":
            if not event.signal:
                _error("EMPTY_SIGNAL", "확인한 상황 정보를 입력해 주세요.", event.event_id)
        else:
            lot = lots.get(event.lot_id)
            if lot is None or (event.product_id, event.state, event.unit) != (lot.product_id, lot.state, lot.unit):
                _error("OBSERVATION_IDENTITY_MISMATCH", "상품·상태·재고 묶음·단위가 초기 자료와 일치하는지 확인해 주세요.", event.event_id)
            if event.location not in p.storage_capacity:
                _error("UNKNOWN_LOCATION", "관측한 점포 또는 창고가 입력 자료에 없습니다.", event.event_id)
            if isinstance(event.quantity,bool) or not math.isfinite(event.quantity) or event.quantity < 0 or abs(event.quantity / lot.quantum - round(event.quantity / lot.quantum)) > 1e-7:
                _error("INVALID_OBSERVED_QUANTITY", "관측 수량은 정해진 단위에 맞는 0 이상의 값이어야 합니다.", event.event_id)
        visible.append(event)
    visible.sort(key=lambda e: (_date(e.known_at), e.event_id))
    active, corrected = {}, []
    for event in visible:
        if event.correction_of:
            previous = active.get(event.correction_of)
            if previous is None:
                _error("CORRECTION_TARGET_UNAVAILABLE", "정정하려는 원래 기록을 먼저 확인해 주세요.", event.event_id)
            if previous.kind in {"TRANSFER_DEPARTED", "TRANSFER_ARRIVED"}:
                _error("TRANSFER_CORRECTION_REQUIRES_RECONCILIATION", "이미 출발·도착한 이동은 자동 정정할 수 없습니다. 운송 기록을 먼저 대조해 주세요.", event.event_id)
            identity = lambda x: (x.kind, x.product_id, x.location, x.state, x.lot_id, x.unit, x.reference_id)
            if identity(previous) != identity(event) or _date(previous.occurred_at) != _date(event.occurred_at):
                _error("CORRECTION_IDENTITY_MISMATCH", "정정 기록의 상품·위치·발생 시각은 원래 기록과 같아야 합니다.", event.event_id)
            del active[event.correction_of]
            corrected.append(event.correction_of)
        active[event.event_id] = event
    # Same-time event order must not depend on upload order. Counts are last.
    order = {"RECEIPT": 0, "TRANSFER_ARRIVED": 1, "TRANSFER_DEPARTED": 2,
             "SALE": 3, "OUTBOUND": 3, "DISPOSAL": 3, "SIGNAL": 4, "STOCK_COUNT": 5}
    events = sorted(active.values(), key=lambda e: (_date(e.occurred_at), order[e.kind], e.event_id))
    count_keys = [(_date(e.occurred_at), e.lot_id, e.location) for e in events if e.kind == "STOCK_COUNT"]
    if len(count_keys) != len(set(count_keys)):
        _error("DUPLICATE_STOCK_COUNT", "같은 시각의 재고 실사 값이 중복됩니다. 정정 기록으로 연결해 주세요.")
    return events, visible, deferred, corrected


def _remaining(p, period, origin, now, events):
    lots, routes = {x.lot_id: x for x in p.lots}, {x.route_id: x for x in p.routes}
    stock = {(l.lot_id, loc): l.stock if l.source == loc else 0.0
             for l in p.lots for loc in p.storage_capacity}
    departures, arrivals, referenced, signals = {}, {}, {}, {}
    used_approval = {l.lot_id: 0.0 for l in p.lots}
    received = {}
    fixed = {e.commitment_id: e for e in p.committed_transfers}
    flows = {e.event_id: e for e in p.confirmed_flows}
    boundary = origin + timedelta(days=period * p.period_days)
    for event in events:
        t = int((_date(event.occurred_at) - origin).total_seconds() // (p.period_days * 86400))
        if event.kind == "SIGNAL":
            # Signals are END-of-period information. A signal stamped at the
            # next boundary belongs to the preceding completed period.
            elapsed = (_date(event.occurred_at) - origin).total_seconds() / (p.period_days * 86400)
            signal_period = max(0, math.ceil(elapsed) - 1)
            if signal_period >= period:
                _error("SIGNAL_NOT_YET_DECISION_AVAILABLE", "해당 상황 정보는 기간 종료 후 다음 계획부터 사용할 수 있습니다.", event.event_id)
            if signal_period in signals and signals[signal_period] != event.signal:
                _error("CONFLICTING_SIGNAL", "같은 기간의 상황 정보가 서로 다릅니다. 정정 여부를 확인해 주세요.", event.event_id)
            signals[signal_period] = event.signal
            continue
        if _date(event.occurred_at) == boundary and event.kind in {"SALE", "OUTBOUND", "TRANSFER_DEPARTED", "DISPOSAL"}:
            _error("PARTIAL_PERIOD_OBSERVATION", "판매·출고 관측은 완료된 기간까지만 반영합니다. 다음 기간 시작 시점에 재계획해 주세요.", event.event_id)
        key = event.lot_id, event.location
        if event.reference_id and event.kind not in {"TRANSFER_ARRIVED"}:
            if event.reference_id in referenced:
                _error("DUPLICATE_REFERENCE", "하나의 확정 입출고에 여러 관측이 연결되어 있습니다. 합계와 정정을 확인해 주세요.", event.event_id)
            referenced[event.reference_id] = event
        if event.kind == "STOCK_COUNT":
            stock[key] = event.quantity
        elif event.kind in {"SALE", "OUTBOUND", "DISPOSAL"}:
            stock[key] -= event.quantity
        elif event.kind == "RECEIPT":
            stock[key] += event.quantity
        elif event.kind == "TRANSFER_DEPARTED":
            route = routes.get(event.route_id)
            if (route is None or route.source != event.location or isinstance(event.vehicle_count,bool)
                    or not isinstance(event.vehicle_count, int) or event.vehicle_count < 1 or not isinstance(event.cost_already_paid,bool)
                    or lots[event.lot_id].state not in route.allowed_states):
                _error("DEPARTURE_ROUTE_INVALID", "출발 위치·이동 경로·차량 수를 확인해 주세요.", event.event_id)
            if event.quantity <= 0 or event.quantity * lots[event.lot_id].capacity_per_unit > event.vehicle_count * route.vehicle_capacity + 1e-7:
                _error("DEPARTURE_CAPACITY_INVALID", "출발 수량이 차량 용량과 맞지 않습니다.", event.event_id)
            stock[key] -= event.quantity
            departures[event.event_id] = event
            if event.location == lots[event.lot_id].source:
                used_approval[event.lot_id] += event.quantity
        elif event.kind == "TRANSFER_ARRIVED":
            original = departures.get(event.reference_id)
            if original is None and event.reference_id in referenced:
                linked = referenced[event.reference_id]
                original = linked if linked.kind == "TRANSFER_DEPARTED" else None
            commitment = fixed.get(event.reference_id)
            if original is None and (commitment is None or commitment.departure_period >= 0):
                _error("ARRIVAL_WITHOUT_DEPARTURE", "도착 기록과 연결되는 실제 출발 기록을 먼저 입력해 주세요.", event.event_id)
            route_id = original.route_id if original else commitment.route_id
            route = routes[route_id]
            quantity = original.quantity if original else commitment.quantity
            lot_id = original.lot_id if original else commitment.lot_id
            arrival_reference = original.event_id if original else event.reference_id
            if (arrival_reference in arrivals or event.location != route.target or event.lot_id != lot_id
                    or abs(event.quantity - quantity) > 1e-7 or (event.route_id and event.route_id != route_id)):
                _error("ARRIVAL_MISMATCH", "도착 기록의 수량·상품·도착지를 출발 기록과 대조해 주세요.", event.event_id)
            departure_period = (int((_date(original.occurred_at) - origin).total_seconds() // (p.period_days * 86400))
                                if original else commitment.departure_period)
            if t < departure_period + route.transit_periods:
                _error("ARRIVAL_TOO_EARLY", "도착 시각이 입력한 운송시간보다 빠릅니다. 운송 기록을 확인해 주세요.", event.event_id)
            arrivals[arrival_reference] = event
            stock[key] += event.quantity
            receipt_key = event.location, lots[lot_id].product_id, lots[lot_id].state
            received[receipt_key] = received.get(receipt_key, 0.0) + event.quantity
        if stock[key] < -1e-7:
            _error("OBSERVED_NEGATIVE_STOCK", "관측한 판매·출고 수량이 확인된 재고보다 많습니다. 누락 입고나 단위를 확인해 주세요.", event.event_id)
    remaining_flows = []
    for flow in p.confirmed_flows:
        event = referenced.get(flow.event_id)
        if event:
            expected_kind = "RECEIPT" if flow.kind == "CONFIRMED_INBOUND" else "OUTBOUND"
            if (event.kind != expected_kind or (event.lot_id, event.location) != (flow.lot_id, flow.location)
                    or abs(event.quantity - flow.quantity) > 1e-7):
                _error("CONFIRMED_FLOW_MISMATCH", "확정 입출고와 실제 관측 수량·위치가 다릅니다. 먼저 대조해 주세요.", event.event_id)
        elif flow.period < period:
            _error("PAST_COMMITMENT_UNCONFIRMED", "예정 시각이 지난 확정 입출고의 실제 처리 여부를 입력해 주세요.", flow.event_id)
        else:
            remaining_flows.append(replace(flow, period=flow.period - period))
    commitments = []
    mapped_departures = set()
    for item in p.committed_transfers:
        event = referenced.get(item.commitment_id)
        if event:
            if (event.kind != "TRANSFER_DEPARTED" or event.route_id != item.route_id or event.lot_id != item.lot_id
                    or abs(event.quantity - item.quantity) > 1e-7 or event.vehicle_count != item.vehicle_count):
                _error("COMMITTED_TRANSFER_MISMATCH", "승인된 이동과 실제 출발 기록이 다릅니다. 운영 기록을 대조해 주세요.", event.event_id)
            mapped_departures.add(event.event_id)
            if event.event_id in arrivals:
                continue
            depart = int((_date(event.occurred_at) - origin).total_seconds() // (p.period_days * 86400))
            item = replace(item, departure_period=depart, cost_already_paid=event.cost_already_paid)
        elif item.departure_period >= 0 and item.departure_period < period:
            _error("PAST_TRANSFER_UNCONFIRMED", "출발 예정 시각이 지난 이동의 실제 출발 여부를 입력해 주세요.", item.commitment_id)
        elif item.commitment_id in arrivals:
            continue
        if item.departure_period + routes[item.route_id].transit_periods < period:
            _error("OVERDUE_ARRIVAL_UNCONFIRMED", "운송 중 재고의 도착 예정일이 지났습니다. 도착 여부를 확인해 주세요.", item.commitment_id)
        commitments.append(replace(item, departure_period=item.departure_period - period))
    for identifier, event in departures.items():
        if identifier in mapped_departures or identifier in arrivals:
            continue
        depart = int((_date(event.occurred_at) - origin).total_seconds() // (p.period_days * 86400))
        if depart + routes[event.route_id].transit_periods < period:
            _error("OVERDUE_ARRIVAL_UNCONFIRMED", "운송 중 재고의 도착 예정일이 지났습니다. 도착 여부를 확인해 주세요.", event.event_id)
        commitments.append(CommittedTransfer(event.event_id, event.lot_id, event.route_id,
                           depart - period, event.quantity, event.vehicle_count, event.cost_already_paid))
    # References must be real input contracts, not labels that suppress checks.
    for ref, event in referenced.items():
        if ref not in flows and ref not in fixed:
            _error("UNKNOWN_COMMITMENT_REFERENCE", "연결한 확정 입출고 번호를 입력 자료에서 찾을 수 없습니다.", event.event_id)
    scenarios = [s for s in p.scenarios if all(s.observations.get(t) == value for t, value in signals.items())]
    if not scenarios:
        _error("UNSEEN_INFORMATION", "새 상황이 기존 수요 가정에 없습니다. 수요 자료를 보완한 뒤 다시 계산해 주세요.")
    probability = sum(s.probability for s in scenarios)
    ids = {s.scenario_id for s in scenarios}
    if any(received.get(key,0.) > value+1e-7 for key,value in p.receipt_limits.items()):
        _error("OBSERVED_RECEIPT_LIMIT_EXCEEDED", "관측된 도착량이 승인된 수령 한도보다 많습니다. 승인과 도착 기록을 확인해 주세요.")
    new_lots = []
    for lot in p.lots:
        total = sum(q for (lid, _), q in stock.items() if lid == lot.lot_id)
        if lot.expiry_period < period and total > 1e-7:
            _error("EXPIRED_STOCK_UNRECONCILED", "유통기한이 지난 재고의 폐기·실사 결과를 먼저 입력해 주세요.", lot.lot_id)
        approval = None if lot.approved_departure_qty is None else max(0., min(
            lot.approved_departure_qty - used_approval[lot.lot_id], total))
        if lot.approved_departure_qty is not None and used_approval[lot.lot_id] > lot.approved_departure_qty + 1e-7:
            _error("OBSERVED_APPROVAL_EXCEEDED", "실제 출발량이 승인된 이동 가능량보다 많습니다. 승인 기록을 확인해 주세요.", lot.lot_id)
        new_lots.append(replace(lot, stock=total, expiry_period=max(0, lot.expiry_period-period), approved_departure_qty=approval))
    markets = tuple(replace(m, period=m.period-period, options=tuple(replace(o, demand={sid: o.demand[sid] for sid in ids})
                        for o in m.options)) for m in p.markets if m.period >= period)
    new_p = replace(p, problem_id=f"{p.problem_id}:remaining:{period}", horizon=p.horizon-period, lots=tuple(new_lots),
        routes=tuple(replace(r, departure_periods=tuple(t-period for t in r.departure_periods if t >= period)) for r in p.routes),
        markets=markets, scenarios=tuple(replace(s, probability=s.probability/probability,
            observations={t-period: v for t, v in s.observations.items() if t >= period}) for s in scenarios),
        confirmed_flows=tuple(remaining_flows), committed_transfers=tuple(commitments),
        receipt_limits={key: max(0., value-received.get(key, 0.)) for key,value in p.receipt_limits.items()},
        processing_capacity={(loc,t-period):v for (loc,t),v in p.processing_capacity.items() if t >= period},
        holding_by_period={(lid,t-period):v for (lid,t),v in p.holding_by_period.items() if t >= period},
        shortage_penalties={mid:v for mid,v in p.shortage_penalties.items() if mid in {m.market_id for m in markets}})
    return new_p, stock, signals


def _actions(result, period):
    rows = []
    for item in result.get("scenario_transfers", []):
        rows.append({"kind": "TRANSFER", "period": item["departure_period"] + period,
                     "lot_id": item["lot_id"], "route_id": item["route_id"], "quantity": item["quantity"],
                     "scenario": item["scenario"], "information_set": item.get("information_set", "")})
    for item in result.get("scenario_price_levels", []):
        rows.append({"kind": "PRICE", "period": item["period"]+period, "market_id": item["market_id"],
                     "level": item["level"], "scenario": item["scenario"], "information_set": item.get("information_set", "")})
    current = {}
    for row in rows:
        if row["period"] != period:
            continue
        item = {k:v for k,v in row.items() if k not in {"scenario", "information_set"}}
        current[json.dumps(item, sort_keys=True)] = item
    return list(current.values()), [row for row in rows if row["period"] > period], rows


def _changes(previous, actions, period):
    if not previous:
        return []
    def index(rows):
        result = {}
        for row in rows:
            if row["period"] < period:
                continue
            key = (row["kind"], row["period"], row.get("lot_id", ""), row.get("route_id", ""), row.get("market_id", ""), row.get("scenario", ""))
            result[key] = row.get("quantity", row.get("level"))
        return result
    before, after = index(previous.get("planned_actions", [])), index(actions)
    return [{"kind":key[0], "period":key[1], "lot_id":key[2], "route_id":key[3], "market_id":key[4],
             "scenario":key[5], "before":before.get(key), "after":after.get(key)}
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]


def plan_adaptive(problem: EconomicProblem, *, policy: str, decision_at: str, origin_at: str,
                  observations: Sequence[Observation | Mapping] = (), previous: Mapping | None = None,
                  risk: str = "expected", time_limit: float = 30.0) -> dict:
    """Initial plan or observed-only replan. Returns serializable, isolated data.

    ``result`` retains the economic solver shape and uses remaining-period
    indices. ``current_actions``/``future_conditional_actions`` use ORIGINAL
    indices. ``current_inventory`` is confirmed on-hand, not projected balance.
    A fixed plan that no longer fits is reported infeasible, never auto-repaired.
    """
    started = time.perf_counter()
    output = {"version": VERSION, "status": "INPUT_REQUIRED", "policy": policy,
              "decision_at": decision_at, "origin_at": origin_at, "risk": risk,
              "production_approved": False, "actual_savings": None,
              "evidence_class": problem.evidence_class, "current_actions": [],
              "future_conditional_actions": [], "planned_actions": [], "current_inventory": [],
              "observations_used": [], "deferred_observations": [], "plan_changes": [],
              "reasons_ko": [], "result": None, "diagnostics": []}
    try:
        if policy not in POLICIES:
            _error("OPERATIONAL_POLICY_REQUIRED", "현재 정보로 계획하는 방식을 선택해 주세요. 미래 완전정보 비교는 연구 전용입니다.")
        if validate_extension(problem):
            _error("BASE_INPUT_INVALID", "기초 운영 자료에 확인이 필요한 항목이 있습니다. 입력 검증 결과를 먼저 확인해 주세요.")
        origin, now = _date(origin_at), _date(decision_at)
        elapsed = (now-origin).total_seconds() / (problem.period_days*86400)
        if elapsed < 0 or abs(elapsed-round(elapsed)) > 1e-8:
            _error("PERIOD_BOUNDARY_REQUIRED", "현재 자료는 기간 시작 시점의 재계획을 지원합니다. 초기 시점과 동일한 시각을 선택해 주세요.")
        period = int(round(elapsed))
        output["current_period"] = period
        if period >= problem.horizon:
            _error("HORIZON_COMPLETE", "입력한 계획 기간이 끝났습니다. 새 기간의 운영 자료를 입력해 주세요.")
        events, visible, deferred, corrected = _observed(problem, observations, origin, now)
        remaining, inventory, signals = _remaining(problem, period, origin, now, events)
        input_signature = _hash(_canonical(problem))
        history = [asdict(event) for event in events]
        signature = _hash({"input": input_signature, "decision_at": now.isoformat(), "policy": policy,
                           "risk": risk, "observations": history, "version": VERSION,
                           "frozen_open_loop": previous.get("frozen_open_loop") if previous and policy == "OPEN_LOOP" else None})
        output.update(input_signature=input_signature, problem_signature=signature, observations_used=history,
            observation_audit=[asdict(e) for e in visible], corrected_observations=corrected,
            deferred_observations=deferred, information_used={"signals": signals, "as_of": now.isoformat()},
            initial_inventory=[{"lot_id":l.lot_id,"location":loc,"quantity":l.stock if l.source==loc else 0.,"unit":l.unit}
                               for l in problem.lots for loc in sorted(problem.storage_capacity)],
            current_inventory=[{"lot_id":lid,"location":loc,"quantity":q,"unit":next(l.unit for l in problem.lots if l.lot_id==lid)}
                               for (lid,loc),q in sorted(inventory.items())],
            remaining_horizon=remaining.horizon, remaining_problem=_canonical(remaining),
            immutable_events=[asdict(x) for x in remaining.confirmed_flows+remaining.committed_transfers])
        fixed_moves = fixed_prices = None
        if previous and previous.get("policy") == "OPEN_LOOP" and policy == "OPEN_LOOP":
            if previous.get("input_signature") != input_signature:
                _error("FIXED_PLAN_INPUT_CHANGED", "초기 조건이 바뀌어 기존 고정 계획을 그대로 사용할 수 없습니다. 새 계획을 생성해 주세요.")
            rows = previous.get("frozen_open_loop", previous.get("planned_actions", []))
            fixed_moves = {(l.lot_id,r.route_id,t):0.0 for l in remaining.lots for r in remaining.routes
                           for t in r.departure_periods if t+r.transit_periods < remaining.horizon
                           and t+r.transit_periods <= l.expiry_period and l.state in r.allowed_states}
            fixed_prices = {}
            for row in rows:
                if row["period"] < period:
                    continue
                if row["kind"] == "TRANSFER":
                    key = row["lot_id"],row["route_id"],row["period"]-period
                    if key not in fixed_moves:
                        _error("FIXED_PLAN_NO_LONGER_VALID", "기존 고정 계획의 경로·유통기한이 현재 조건에 맞지 않습니다. 새 계획이 필요합니다.")
                    fixed_moves[key] = row["quantity"]
                else:
                    fixed_prices[row["market_id"],row["period"]-period] = row["level"]
            output["frozen_open_loop"] = rows
        engine_policy = "MULTISTAGE_RECOURSE" if policy == "MULTISTAGE_RECOURSE" else "OPEN_LOOP"
        result = solve_extended(remaining, policy=engine_policy, risk=risk, time_limit=time_limit,
                                fixed_transfers=fixed_moves, fixed_prices=fixed_prices, initial_inventory=inventory)
        output["result"] = result
        if (result.get("expected_net_value") is None or result.get("constraint_violations", 0)
                or result.get("integrality_max_violation",0)>1e-6):
            output["status"] = result["status"] if result.get("expected_net_value") is None else "VALIDATION_FAILED"
            output["reasons_ko"] = ["현재 재고와 확정 입출고를 모두 지킬 수 있는 계획을 찾지 못했습니다. 수량·기한·용량을 확인해 주세요."]
            output["diagnostics"] = result.get("input_issues", [])
            LOGGER.warning("Adaptive solve not usable: %s", {"status":result["status"], "issues":result.get("input_issues")})
            return output
        current, future, actions = _actions(result, period)
        output.update(status="COMPLETED", current_actions=current, future_conditional_actions=future,
                      planned_actions=actions, plan_changes=_changes(previous, actions, period),
                      replan_count=(int(previous.get("replan_count",0))+1 if previous else 0) if policy!="OPEN_LOOP" else 0,
                      observation_update_count=int(previous.get("observation_update_count",0))+1 if previous else 0)
        if policy == "OPEN_LOOP" and "frozen_open_loop" not in output:
            output["frozen_open_loop"] = actions
        output["reasons_ko"] = ["현재까지 확인된 재고와 관측으로 남은 기간을 계산했습니다." if previous else "입력한 운영 조건으로 최초 계획을 생성했습니다.",
            "현재 검토할 행동과 아직 확정되지 않은 미래 계획을 구분했습니다. 분석만으로 실제 이동·판매가 실행되지는 않습니다."]
        if signals:
            output["reasons_ko"].append("새로 확인한 상황에 맞는 수요 가정만 사용했습니다. 조건부 확률은 기존 분포에서 계산했습니다.")
        if output["plan_changes"]:
            output["reasons_ko"].append("새 관측에 따라 남은 이동 수량 또는 판매 조건이 달라졌습니다. 변경 내역을 확인해 주세요.")
        if deferred:
            output["reasons_ko"].append("결정 시점 이후에 알려진 관측은 이번 계산에서 제외했습니다.")
    except (ObservationError, TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, ObservationError):
            diagnostic, message = {"code":exc.code,"event_id":exc.event_id}, exc.message
        else:
            diagnostic, message = {"code":"INVALID_OBSERVATION_FORMAT","detail":str(exc)}, "관측값의 형식·수량·시각을 확인해 주세요. 자세한 내용은 개발자 로그에 기록했습니다."
        output["diagnostics"] = [diagnostic]
        output["reasons_ko"] = [message]
        LOGGER.warning("Adaptive planning rejected: %s", diagnostic, exc_info=not isinstance(exc, ObservationError))
    finally:
        output["runtime_seconds"] = time.perf_counter()-started
    return output
