"""Small candidate-page controls; all calculations remain in services."""
from datetime import date, datetime, time, timezone, timedelta
import uuid

import pandas as pd
import streamlit as st

POLICY_LABELS = {"처음 정한 계획 유지": "OPEN_LOOP", "새 관측으로 남은 계획 조정": "ROLLING_HORIZON",
                 "상황별 후속 계획 함께 검토": "MULTISTAGE_RECOURSE"}
KIND_LABELS = {"판매 확인": "SALE", "입고 확인": "RECEIPT", "출고 확인": "OUTBOUND", "재고 실사": "STOCK_COUNT",
               "폐기 확인": "DISPOSAL", "이동 출발 확인": "TRANSFER_DEPARTED", "이동 도착 확인": "TRANSFER_ARRIVED",
               "수요 상황 확인": "SIGNAL"}
KST = timezone(timedelta(hours=9))


def _date_input(label, key, initial):
    if key not in st.session_state:
        st.session_state[key] = initial
    return st.date_input(label, key=key)


def render_controls(prepared, input_key):
    problem = prepared.problem
    origin = date.fromisoformat(str(prepared.snapshot.as_of)[:10])
    c1, c2 = st.columns([2, 1])
    policy_label = c1.selectbox("계획 방식", list(POLICY_LABELS), key="candidate_policy")
    with c2:
        decision = _date_input("이번 판단 기준일", "candidate_decision_date", origin)
    risk_label = st.selectbox("수요가 달라질 때의 기준", ["발생 가능성을 반영한 평균 결과", "가장 불리한 상황의 결과"], key="candidate_risk")
    st.caption("기준일 오전 0시까지 확인한 관측만 사용합니다. 분석 결과는 승인·출고·판매 기록이 아닙니다.")
    bucket = st.session_state.setdefault("_candidate_observation_sets", {})
    observations = bucket.setdefault(input_key, [])
    with st.expander("관측 추가 · 실제로 확인한 재고와 판매", expanded=False):
        st.caption("아직 실행하지 않은 계획을 관측으로 입력하지 마세요. 가상·시험 입력의 관측도 같은 자료 구분으로 유지됩니다.")
        with st.form("candidate_observation_form", clear_on_submit=False):
            kind_label = st.selectbox("확인한 내용", list(KIND_LABELS), key="candidate_obs_kind")
            lot_id = st.selectbox("상품 배치", [lot.lot_id for lot in problem.lots], key="candidate_obs_lot")
            location = st.selectbox("확인 점포", list(problem.storage_capacity), key="candidate_obs_location")
            c1, c2 = st.columns(2)
            with c1:
                occurred = _date_input("실제로 발생한 날짜", "candidate_obs_occurred", origin)
                occurred_time = st.time_input("발생 시각", value=None if "candidate_obs_occurred_time" in st.session_state else time(12), key="candidate_obs_occurred_time")
            with c2:
                known = _date_input("자료로 확인한 날짜", "candidate_obs_known", origin)
                known_time = st.time_input("확인 시각", value=None if "candidate_obs_known_time" in st.session_state else time(18), key="candidate_obs_known_time")
            quantity = st.number_input("확인 수량 (선택한 배치의 단위)", min_value=0.0, key="candidate_obs_quantity")
            source = st.text_input("확인 근거 (전표·실사 기록·시험 사례 등)", key="candidate_obs_source")
            route_id = st.selectbox("관련 이동 경로 (이동 확인 시)", ["", *[r.route_id for r in problem.routes]], key="candidate_obs_route")
            reference_id = st.text_input("기존 확정 입출고·이동 기록 번호 (해당 시)", key="candidate_obs_reference")
            price = st.selectbox("판매 조건 (판매 확인 시)", ["normal", *sorted({o.level for m in problem.markets for o in m.options if o.level != "normal"})],
                format_func=lambda v: "정상판매" if v == "normal" else "할인판매: " + v, key="candidate_obs_price")
            signal = st.text_input("확인된 수요 상황 (입력 양식의 상황 이름)", key="candidate_obs_signal")
            correction = st.selectbox("정정할 이전 관측 (정정 시에만 선택)", ["", *[x["event_id"] for x in observations]], key="candidate_obs_correction")
            vehicles = st.number_input("확정 차량 수 (이동 확인 시)", min_value=1,
                value=None if "candidate_obs_vehicles" in st.session_state else 1, key="candidate_obs_vehicles")
            paid = st.checkbox("위 이동의 운송비를 실제로 지출했습니다", key="candidate_obs_paid")
            add = st.form_submit_button("관측 기록 추가")
        if add:
            if not all((occurred, occurred_time, known, known_time, vehicles)):
                st.warning("발생·확인 시각과 차량 수를 확인해 주세요. 기록은 아직 추가하지 않았습니다.")
            elif not source.strip():
                st.warning("관측의 확인 근거를 입력해 주세요. 기록은 아직 추가하지 않았습니다.")
            else:
                lot = next(x for x in problem.lots if x.lot_id == lot_id)
                observations.append({"event_id": uuid.uuid4().hex, "kind": KIND_LABELS[kind_label],
                    "occurred_at": datetime.combine(occurred, occurred_time, KST).isoformat(),
                    "known_at": datetime.combine(known, known_time, KST).isoformat(),
                    "product_id": lot.product_id, "location": location, "state": lot.state,
                    "lot_id": lot_id, "unit": lot.unit, "quantity": quantity, "source": source.strip(),
                    "correction_of": correction, "route_id": route_id, "price_level": price,
                    "signal": signal.strip(), "reference_id": reference_id,
                    "vehicle_count": int(vehicles), "cost_already_paid": paid})
                st.session_state["_candidate_observation_sets"] = bucket
                st.success("관측을 추가했습니다. 계획 실행 버튼을 눌러 시간·재고·확정 기록을 함께 검증하세요.")
        if observations:
            names = {v:k for k,v in KIND_LABELS.items()}
            st.dataframe(pd.DataFrame([{"기록 번호": x["event_id"], "내용": names[x["kind"]],
                "발생 시각": x["occurred_at"], "확인 시각": x["known_at"], "배치": x["lot_id"],
                "수량": x["quantity"], "단위": x["unit"], "확인 근거": x["source"],
                "정정 대상": x.get("correction_of", "")} for x in observations]), hide_index=True, width="stretch")
    return {"policy": POLICY_LABELS[policy_label], "decision_at": datetime.combine(decision, time(), KST).isoformat(),
        "origin_at": datetime.combine(origin, time(), KST).isoformat(), "observations": list(observations),
        "risk": "expected" if risk_label == "발생 가능성을 반영한 평균 결과" else "worst_case"}


def action_frame(problem, actions, *, conditional=False):
    routes = {x.route_id: x for x in problem.routes}
    markets = {x.market_id: x for x in problem.markets}
    rows = []
    for action in actions:
        if action["kind"] == "TRANSFER":
            route = routes[action["route_id"]]
            row = {"검토 항목": "재고 이동", "배치·상품": action["lot_id"], "점포·경로": f"{route.source} → {route.target}",
                   "계획 수량": action["quantity"]}
        else:
            market = markets[action["market_id"]]
            row = {"검토 항목": "정상판매" if action["level"] == "normal" else "할인판매", "배치·상품": market.product_id,
                   "점포·경로": market.location, "계획 수량": None}
        row["계획 날짜 순서"] = action["period"] + 1
        if conditional:
            row["수요 상황"] = action.get("scenario", "기준")
            row["확정 여부"] = "조건부 미래 계획 · 미확정"
        else:
            row["확정 여부"] = "현재 검토할 계획 · 실행 전"
        rows.append(row)
    return pd.DataFrame(rows)


def render_adaptive_result(problem, envelope):
    if not envelope:
        return
    names = {v:k for k,v in POLICY_LABELS.items()}
    st.caption(f"계획 방식: {names.get(envelope['policy'], '확인 필요')} · 판단 기준: {envelope['decision_at'][:10]}")
    for reason in envelope.get("reasons_ko", []):
        st.caption(reason)
    st.subheader("지금 검토할 행동")
    frame = action_frame(problem, envelope.get("current_actions", []))
    if frame.empty: st.info("이번 시점에 새로 검토할 이동·판매 조건이 없습니다.")
    else: st.dataframe(frame, hide_index=True, width="stretch")
    with st.expander("남은 재고 · 사용한 관측 · 변경 이유", expanded=False):
        stock = pd.DataFrame(envelope.get("current_inventory", [])).rename(columns={"lot_id":"배치", "location":"점포", "quantity":"확인 재고", "unit":"단위"})
        if not stock.empty: st.dataframe(stock, hide_index=True, width="stretch")
        st.write(f"사용한 관측 {len(envelope.get('observations_used', []))}건 · 판단 시점 이후여서 제외한 관측 {len(envelope.get('deferred_observations', []))}건")
        changed = envelope.get("plan_changes", [])
        if changed:
            def display_change(value, kind):
                if value is None: return "없음"
                if kind == "PRICE": return "정상판매" if value == "normal" else "할인판매: " + str(value)
                return str(value)
            st.dataframe(pd.DataFrame([{"항목": "이동" if r["kind"] == "TRANSFER" else "판매 조건",
                "날짜 순서": r["period"]+1, "배치·경로": r.get("lot_id") or r.get("market_id"),
                "이전": display_change(r.get("before"), r["kind"]), "이번": display_change(r.get("after"), r["kind"])} for r in changed]), hide_index=True, width="stretch")
        else: st.caption("직전 계획과 비교할 변경이 없거나 최초 계획입니다.")
    with st.expander("미래의 조건부 계획 · 아직 실행할 계획 아님", expanded=False):
        future = action_frame(problem, envelope.get("future_conditional_actions", []), conditional=True)
        if future.empty: st.caption("남은 기간의 조건부 계획이 없습니다.")
        else: st.dataframe(future, hide_index=True, width="stretch")
