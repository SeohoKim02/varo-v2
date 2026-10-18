"""Opt-in candidate review, preserving the baseline session and production history."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from components.cards import render_page_header
from components.navigation import candidate_enabled
from components.user_feedback import report_error
from components.candidate_adaptive_view import render_controls, render_adaptive_result, POLICY_LABELS
from services.candidate_economics import solve_snapshot_economics, _canonical
from services.candidate_economics_experiments import demo_problem
from services.operating_snapshot import OperationalSnapshot, StockPosition, from_canonical
from services.operator_intake_candidate import validate_operator_intake
from services.candidate_input import read_uploads, prepare_input, run_prepared, TABLE_LABELS, FIELD_HELP
from services.candidate_session import cached, input_signature, request_key, begin_request, commit_result, signature, model_fingerprint
from services.local_paths import output_root

logger = logging.getLogger("varo.candidate")
_OUTPUTS = output_root()
HISTORY_DIR = _OUTPUTS / "third_upgrade_20261017" / "candidate_history"
_DEFAULT_HISTORY_DIR = HISTORY_DIR
SCENARIOS = {"정상판매 중심": "normal", "할인판매 중심": "discount", "재고 이동 중심": "transfer", "이동과 할인 혼합": "mixed"}


def _scenario_snapshot(problem):
    # Explicit scenario facts, not operator evidence or canonical observations.
    as_of = date(2026, 10, 16)
    positions = tuple(StockPosition("W3_SCENARIO", "scenario-network", as_of.isoformat(),
        lot.source, lot.product_id, lot.state, lot.lot_id, lot.unit, lot.stock, 0, lot.stock, 0, 0,
        (as_of + timedelta(days=lot.expiry_period)).isoformat(), {"source": "SCENARIO: supplied demonstration assumptions"})
        for lot in problem.lots)
    return OperationalSnapshot("W3_SCENARIO", "scenario-network", as_of.isoformat(), problem.horizon,
        positions, (), "SCENARIO", {"production_approved": False},
        {"approved_route_ids": [r.route_id for r in problem.routes], "evidence_class": "SCENARIO"})


def _product_frame(problem, result):
    rows = []
    envelope = result.get("adaptive_context") or {}
    probabilities = {s["scenario_id"]: s["probability"] for s in envelope.get("remaining_problem", {}).get("scenarios", [])}
    if not probabilities:
        probabilities = {s.scenario_id: s.probability for s in problem.scenarios}
    scenario_moves = result.get("scenario_transfers", [])
    routes = {r.route_id: r for r in problem.routes}
    markets = {m.market_id: m for m in problem.markets}
    for lot in problem.lots:
        # Count goods leaving the original stock once; a DC onward leg is not new goods.
        moves = [x for x in (scenario_moves or result.get("transfers", [])) if x["lot_id"] == lot.lot_id]
        moved = sum(x["quantity"] * (probabilities.get(x.get("scenario"), 0) if scenario_moves else 1)
                    for x in moves if routes[x["route_id"]].source == lot.source)
        sales = [x for x in result.get("sales", []) if x["lot_id"] == lot.lot_id]
        local = [x for x in sales if markets[x["market_id"]].location == lot.source]
        local_qty = sum(x["quantity"] * probabilities.get(x.get("scenario"), 1) for x in local)
        discount = any(x["level"] != "normal" for x in local)
        action = "일부 이동 + 잔여 할인" if moved and discount else "재고 이동 + 정상판매" if moved and local_qty else \
                 "재고 이동" if moved else "할인판매" if discount else "정상판매" if local_qty else "보관·추가 확인"
        row = {"상품": lot.product_id, "배치": lot.lot_id, "출발 점포": lot.source,
            "초기 재고": lot.stock, "계획 이동량": moved, "출발 점포 판매 계획": local_qty,
            "단위": lot.unit, "처리 계획": action}
        if envelope:
            # Replanning can start with observed stock at a destination or DC.
            # Keep the original-source metric stable, and label onward legs as
            # a leg total: aggregated lots do not identify distinct physical units.
            onward = sum(x["quantity"] * (probabilities.get(x.get("scenario"), 0) if scenario_moves else 1)
                         for x in moves if routes[x["route_id"]].source != lot.source)
            all_sales = sum(x["quantity"] * probabilities.get(x.get("scenario"), 1) for x in sales)
            has_discount = any(x["level"] != "normal" and x["quantity"] > 0 for x in sales)
            has_normal = any(x["level"] == "normal" and x["quantity"] > 0 for x in sales)
            sale_action = "정상·할인판매" if has_discount and has_normal else "할인판매" if has_discount else "정상판매" if all_sales else ""
            row["처리 계획"] = ("재고 이동" + (" + " + sale_action if sale_action else "")) if moved or onward else sale_action or "보관·추가 확인"
            row["후속 경로 이동량 (구간 합)"] = onward
            row["전체 점포 판매 계획"] = all_sales
            row["이번 판단 시 확인 재고"] = sum(x["quantity"] for x in envelope.get("current_inventory", []) if x["lot_id"] == lot.lot_id)
            row["계획 기준"] = "남은 기간의 시나리오 평균 · 미래 행동 미확정"
        rows.append(row)
    return pd.DataFrame(rows)


def _plan_day(result, relative_period):
    """Solver rows are relative to the remaining horizon, display original days."""
    return int(relative_period) + int((result.get("adaptive_context") or {}).get("current_period", 0)) + 1


def _save_review(problem, result, label):
    """Use the existing ledger's run lineage only; never map joint actions to legacy actions."""
    from services.seller_shadow_ledger import save_run
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    execution_key = uuid.uuid4().hex
    run_id = save_run(data_signature=result["problem_signature"], execution_key=execution_key,
                      decision_at=stamp, directory=HISTORY_DIR)
    payload = {"created_at": stamp, "label": label, "run_id": run_id, "mode": result.get("input_data_class", "SCENARIO"),
        "production_action_applied": False, "lifecycle": "ANALYSIS_PLAN", "approval_status": "NOT_APPROVED",
        "execution_status": "NOT_EXECUTED", "products": _product_frame(problem, result).to_dict("records"),
        "input_version": result.get("input_version"), "policy": result.get("adaptive_context", {}).get("policy", "OPEN_LOOP"),
        "snapshot_signature": result.get("snapshot_signature"), "request_signature": result.get("request_signature"),
        "economic_result": result}
    path = HISTORY_DIR / f"candidate_{execution_key}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return path


def _render_scenarios():
    st.info("가상 시나리오입니다. 입력값과 계산 결과는 실제 사업장의 매출·절감 실적이 아닙니다.")
    label = st.selectbox("검토할 사례", list(SCENARIOS), key="candidate_scenario_choice")
    if st.button("사례 불러오기", key="candidate_load_scenario"):
        problem = demo_problem(SCENARIOS[label])
        st.session_state["_candidate_problem"] = problem
        st.session_state["_candidate_snapshot"] = _scenario_snapshot(problem)
        st.session_state["_candidate_label"] = label
        st.session_state.pop("_candidate_result", None)
        st.session_state.pop("_candidate_saved", None)
    problem = st.session_state.get("_candidate_problem")
    if problem is None:
        st.caption("사례를 불러온 뒤 계획을 비교하세요.")
        return
    st.caption(f"불러온 사례: {st.session_state['_candidate_label']} · 계획기간 {problem.horizon}일")
    with st.expander("입력값과 가정 확인", expanded=False):
        inputs = []
        for market in problem.markets:
            for option in market.options:
                inputs.append({"점포": market.location, "시점": market.period + 1,
                    "판매 방식": "정상판매" if option.level == "normal" else "할인판매",
                    "단위 판매가": option.unit_price.field.value,
                    "가정 수요": next(iter(option.demand.values())), "단위": market.unit})
        st.dataframe(pd.DataFrame(inputs), hide_index=True, width="stretch")
        st.caption("동일 사업자 범위 · 차량 고정비는 공동 부담 · 보관비·폐기비·기말가치는 명시적으로 0원을 가정했습니다.")
    if st.button("계획 비교 실행", type="primary", key="candidate_run"):
        st.session_state.pop("_candidate_result", None)
        st.session_state.pop("_candidate_saved", None)
        try:
            result = solve_snapshot_economics(st.session_state["_candidate_snapshot"], problem)
            logger.info("Candidate review result: %r", result)
            st.session_state["_candidate_result"] = result
            st.session_state.pop("_candidate_saved", None)
        except Exception as exc:
            report_error("candidate solve", exc, "계산을 완료하지 못했습니다. 입력값을 다시 확인한 뒤 재시도하세요. 문제가 계속되면 관리자에게 문의하세요.")
    result = st.session_state.get("_candidate_result")
    if result is None: return
    if result["status"] not in {"OPTIMAL", "FEASIBLE_LIMIT"}:
        st.warning("이 자료로는 계획을 확정할 수 없습니다. 수량 단위·이동 승인·가격·운송비·유통기한을 확인해 주세요.")
        return
    if result["status"] == "FEASIBLE_LIMIT":
        st.warning("제한 시간 안에 찾은 계획입니다. 가장 좋은 계획임을 아직 확인하지 못했습니다.")
    st.subheader("상품별 처리 계획")
    metrics = st.columns(3)
    metrics[0].metric("예상 순현금가치", f"{result['expected_net_value']:,.0f}원")
    metrics[1].metric("계산 운송비", f"{result['transport_cost']:,.0f}원")
    metrics[2].metric("실제 절감액", "미산정")
    st.caption("외부 판매수입에서 앞으로 발생할 비용을 뺀 모형 예상값입니다. 매입원가와 실제 지출·절감 실적은 별개입니다.")
    frame = _product_frame(problem, result)
    search = st.text_input("전체 상품 검색", key="candidate_product_search")
    visible = frame[frame["상품"].str.contains(search, case=False, regex=False)] if search else frame
    st.dataframe(visible, hide_index=True, width="stretch")
    with st.expander("이동 경로와 수량", expanded=bool(result.get("transfers"))):
        rows = []
        for move in result.get("transfers", []):
            route = next(r for r in problem.routes if r.route_id == move["route_id"])
            rows.append({"출발": route.source, "도착": route.target, "계획 이동량": move["quantity"],
                "출발 시점": move["departure_period"] + 1, "도착 시점": move["departure_period"] + route.transit_periods + 1})
        if rows: st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else: st.write("이 사례에서는 이동하지 않는 계획입니다.")
        st.caption("운송 중 재고는 출발·도착 점포에서 판매할 수 없습니다. 실제 운행 지시는 아닙니다.")
    st.download_button("상품별 계획 CSV 다운로드", frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name="Varo_candidate_scenario.csv", mime="text/csv", key="candidate_download")
    if st.button("후보 검토 이력에 저장", key="candidate_save"):
        st.session_state.pop("_candidate_saved", None)
        try:
            path = _save_review(problem, result, st.session_state["_candidate_label"])
            st.session_state["_candidate_saved"] = path.name
        except Exception as exc:
            report_error("candidate history", exc, "검토 이력을 저장하지 못했습니다. 저장 공간과 쓰기 권한을 확인해 주세요. 계산 결과는 화면에 유지됩니다.")
    if st.session_state.get("_candidate_saved"):
        st.success("후보 검토용 이력에 저장했습니다. 기존 결과 이력과 운영 기록은 변경하지 않았습니다.")


def _capture_candidate_uploads():
    # An explicit removal clears the saved files; page widget cleanup does not.
    st.session_state["_candidate_upload_files"] = [(u.name, u.getvalue())
        for u in (st.session_state.get("candidate_inventory_upload") or [])]


def _render_real_intake():
    st.info("실제 자료와 시험 입력을 구분합니다. 누락된 가격·비용·단위는 추가로 확인한 뒤 계산합니다.")
    uploads = st.file_uploader("운영 자료 XLSX 또는 표별 CSV", type=["xlsx", "csv"],
        accept_multiple_files=True, key="candidate_inventory_upload", on_change=_capture_candidate_uploads)
    if uploads:
        st.session_state["_candidate_upload_files"] = [(u.name, u.getvalue()) for u in uploads]
    upload_entries = st.session_state.get("_candidate_upload_files", [])
    if not uploads and upload_entries:
        st.caption("화면 이동 전에 업로드한 자료를 이 세션에서 유지하고 있습니다. 새 파일을 선택하면 교체합니다.")
    dataset = st.text_input("자료 묶음 이름", key="candidate_dataset")
    network = st.text_input("사업장 식별 이름", key="candidate_network")
    as_of = st.date_input("자료 기준일", key="candidate_as_of")
    left, right = st.columns(2)
    horizon = left.number_input("계획기간 (일)", min_value=1, max_value=31,
        value=None if "candidate_horizon" in st.session_state else 2, key="candidate_horizon")
    label = right.selectbox("입력 자료 구분", ["실제 운영자료", "실제 자료 + 시험 가정", "가상 시나리오"], key="candidate_input_kind")
    kind = {"실제 운영자료": "REAL", "실제 자료 + 시험 가정": "TEST", "가상 시나리오": "SCENARIO"}[label]
    left, right = st.columns(2)
    currency = left.text_input("통화", value=None if "candidate_currency" in st.session_state else "KRW", key="candidate_currency")
    capacity_unit = right.text_input("공통 용량 단위", value=None if "candidate_capacity_unit" in st.session_state else "KG", key="candidate_capacity_unit")
    consent = st.checkbox("자료 제공자가 연구 검토에 이용하도록 허락했습니다", key="candidate_consent")
    metadata = {"dataset_id": dataset, "network_id": network, "as_of_date": as_of.isoformat(),
        "owner_authorized": consent, "horizon_days": int(horizon) if horizon is not None else None, "data_class": kind,
        "currency": currency, "capacity_unit": capacity_unit}
    file_bytes = dict(upload_entries)
    current_input_key = signature({"input": input_signature(file_bytes, metadata),
        "upload_sequence": [(name, hashlib.sha256(value).hexdigest()) for name, value in upload_entries]})
    if st.button("자료 검증", key="candidate_validate_upload"):
        if not upload_entries:
            st.warning("먼저 재고 CSV 또는 운영 자료 XLSX 파일을 선택해 주세요.")
        else:
            try:
                if len({name for name, _ in upload_entries}) != len(upload_entries):
                    raise ValueError("duplicate upload names")
                prepared, timing = cached(st.session_state, "input", current_input_key,
                    lambda: prepare_input(read_uploads(file_bytes), metadata))
                st.session_state["_candidate_input_timing"] = timing
                logger.info("Candidate input issues: %r", prepared.issues)
                st.session_state["_candidate_input_attempt"] = prepared
                # Failed new input never destroys the last valid problem or its result.
                if prepared.ready:
                    st.session_state["_candidate_ready_input"] = prepared
                    st.session_state["_candidate_ready_input_key"] = current_input_key
            except Exception as exc:
                report_error("candidate intake", exc, "파일을 읽지 못했습니다. 같은 이름의 파일·표 형식·날짜를 확인해 주세요. 이전 분석 결과는 유지합니다.")
    attempted = st.session_state.get("_candidate_input_attempt")
    if attempted:
        if attempted.issues:
            st.warning("새 자료에 확인할 항목이 있어 계산에 적용하지 않았습니다. 이전 분석 결과는 그대로 유지합니다.")
            st.dataframe(pd.DataFrame([{"표": TABLE_LABELS.get(i["table"], "입력 자료"),
                "행": i["row"], "확인 항목": i.get("column") or "행 전체", "조치": i["message_ko"]}
                for i in attempted.issues]), hide_index=True, width="stretch")
        else: st.success("입력값과 근거를 확인했습니다. 계획 비교를 실행할 수 있습니다.")
    ready = st.session_state.get("_candidate_ready_input")
    active_key = None
    if ready is not None:
        if attempted is not None and attempted is not ready:
            st.caption("아래 실행 버튼은 마지막으로 검증을 통과한 자료를 사용합니다.")
        st.caption(f"계산할 자료: {ready.snapshot.dataset} · 기준일 {ready.snapshot.as_of} · 계획기간 {ready.problem.horizon}일")
        ready_key = st.session_state.get("_candidate_ready_input_key", current_input_key)
        controls = render_controls(ready, ready_key)
        previous = st.session_state.get("_candidate_adaptive_result")
        if st.session_state.get("_candidate_result_input_key") != ready_key:
            previous = None
        active_key = request_key(ready, **{k:controls[k] for k in ("policy", "decision_at", "origin_at", "observations", "risk")})
        if st.session_state.get("_candidate_active_request") != active_key:
            begin_request(st.session_state, active_key)
        if current_input_key != ready_key:
            st.warning("입력 내용이 바뀌어 아직 검증되지 않았습니다. 실행 버튼은 마지막으로 검증한 자료에 적용됩니다.")
        if st.button("입력 자료로 계획 비교", type="primary", key="candidate_real_run"):
            try:
                from services.candidate_adaptive import plan_adaptive
                from services.candidate_input import VERSION as input_version
                token = begin_request(st.session_state, active_key)
                cache_key = request_key(ready, previous=previous,
                    **{k:controls[k] for k in ("policy", "decision_at", "origin_at", "observations", "risk")})
                envelope, timing = cached(st.session_state, "plan", cache_key,
                    lambda: plan_adaptive(ready.problem, previous=previous, **controls))
                calculated = envelope.get("result") or {"status": "INPUT_REQUIRED"}
                logger.info("Uploaded economic result: %r", calculated)
                st.session_state["_candidate_uploaded_attempt_result"] = calculated
                if envelope["status"] == "COMPLETED":
                    context = {k:v for k,v in envelope.items() if k not in {"result", "diagnostics"}}
                    calculated.update(input_data_class=ready.data_class, input_file_hashes=ready.file_hashes,
                        actual_savings=None, production_approved=False, adaptive_context=context,
                        snapshot_signature=ready.snapshot.base_problem_signature, request_signature=active_key, input_version=input_version)
                    if commit_result(st.session_state, token, calculated, ready.problem, input_key=ready_key, envelope=envelope):
                        st.session_state["_candidate_plan_timing"] = timing
                    else: st.warning("더 최근 입력이 있어 이전 계산을 적용하지 않았습니다. 현재 입력으로 다시 실행해 주세요.")
                else:
                    for reason in envelope.get("reasons_ko", []): st.warning(reason)
                    st.warning("이번 계산은 완료되지 않았습니다. 아래에는 이전 결과가 유지됩니다.")
            except Exception as exc:
                report_error("uploaded candidate solve", exc, "계산을 완료하지 못했습니다. 입력 근거를 확인해 주세요. 이전 결과는 유지합니다.")
    with st.expander("입력 항목 안내"):
        st.write("기존 9개 시트는 유지합니다. 경제적 계획 계산에는 상품별 비용·이동 경로·판매가격과 예상 수요·점포 용량 정책 표를 함께 입력하세요.")
        st.write("빈칸은 미확인입니다. 비용 없음·예약 없음은 근거를 확인한 뒤 숫자 0으로 입력하세요. 수식은 엑셀에서 계산한 후 값만 붙여넣어 저장해 주세요.")
        st.dataframe(pd.DataFrame([{"양식 열": k, "설명": v} for k,v in FIELD_HELP.items()]), hide_index=True, width="stretch")
    result = st.session_state.get("_candidate_uploaded_result")
    problem = st.session_state.get("_candidate_uploaded_problem")
    if result and problem:
        labels = {"REAL": "실제 자료에 기반한 예상 계획", "TEST": "시험 입력을 포함한 계산 · 실제 성과 아님", "SCENARIO": "가상 시나리오 · 실제 성과 아님"}
        st.subheader("입력 자료의 전체 상품 결과")
        stale = (current_input_key != st.session_state.get("_candidate_result_input_key") or
            (active_key is not None and active_key != st.session_state.get("_candidate_result_request")))
        if stale:
            st.warning("이전 입력·관측·계획 방식의 결과입니다. 현재 조건의 최신 결과가 아닙니다. 자료 검증과 계획 실행 후 새 결과를 확인하세요.")
        st.info(labels[result["input_data_class"]])
        cols = st.columns(3)
        cols[0].metric("예상 순현금가치", f"{result['expected_net_value']:,.0f} {problem.currency}")
        cols[1].metric("계산 운송비", f"{result['transport_cost']:,.0f} {problem.currency}")
        cols[2].metric("실제 절감액", "미산정")
        render_adaptive_result(problem, result.get("adaptive_context"))
        frame = _product_frame(problem, result)
        query = st.text_input("전체 상품 검색", key="candidate_uploaded_search")
        st.dataframe(frame[frame["상품"].str.contains(query, regex=False, case=False)] if query else frame,
            hide_index=True, width="stretch")
        chosen = st.selectbox("상품 상세", list(frame["배치"]), key="candidate_uploaded_detail")
        with st.expander("선택한 상품의 계획과 근거", expanded=False):
            st.dataframe(frame[frame["배치"] == chosen], hide_index=True, width="stretch")
            st.caption("계획 이동량은 초기 출발 점포 기준입니다. 후속 경로 이동량은 다른 점포·센터에서 이어지는 구간의 합이므로 고유 상품 수량과 다를 수 있습니다. 전체 점포 판매 계획에는 이동 도착 후 판매도 포함합니다.")
        with st.expander("이동 경로·재고 변화·계산 근거", expanded=True):
            route_map = {r.route_id: r for r in problem.routes}
            moves = [{"상품 배치": m["lot_id"], "출발": route_map[m["route_id"]].source,
                "도착": route_map[m["route_id"]].target, "계획 이동량": m["quantity"],
                "출발일 순서": _plan_day(result, m["departure_period"]),
                "도착일 순서": _plan_day(result, m["departure_period"]+route_map[m["route_id"]].transit_periods)} for m in result.get("transfers", [])]
            if moves: st.dataframe(pd.DataFrame(moves), hide_index=True, width="stretch")
            elif result.get("scenario_transfers"):
                st.write("상황에 따라 이동 계획이 달라집니다. 위의 현재 행동과 미래 조건부 계획을 확인하세요.")
            else: st.write("이 계획에서는 재고를 이동하지 않습니다.")
            st.caption("표의 이동은 계산한 계획입니다. 실제 출고·판매·지출을 기록한 것이 아닙니다.")
            remaining = [{"수요 시나리오": row.get("scenario", "기준"), "배치": row["lot_id"],
                "점포": row["location"], "날짜 순서": _plan_day(result, row["period"]), "시뮬레이션 잔여 재고": row["quantity"]}
                for row in result.get("inventory", [])]
            if remaining: st.dataframe(pd.DataFrame(remaining), hide_index=True, width="stretch")
        if st.button("최초 입력의 판매·이동 대안 비교", key="candidate_compare_alternatives"):
            try:
                from services.candidate_economics_policies import fair_baselines
                compared, _ = cached(st.session_state, "comparison", signature({"problem": _canonical(problem), "source": model_fingerprint()}), lambda: fair_baselines(problem))
                result["alternative_comparison"] = compared
            except Exception as exc:
                report_error("candidate alternatives", exc, "대안 비교를 완료하지 못했습니다. 현재 계획은 유지합니다. 입력 조건을 확인한 뒤 다시 실행하세요.")
        comparison = result.get("alternative_comparison", {})
        if comparison.get("status") == "COMPLETED":
            names = {"A": "정상판매만", "B": "이동 없이 판매·할인", "C": "먼저 이동 후 판매·할인", "D": "경로별 판단 후 공동 용량 보정", "E": "이동·판매·할인 공동 결정"}
            st.subheader("같은 입력에서의 대안 비교")
            st.dataframe(pd.DataFrame([{"대안": names[k], "예상 순현금가치": v.get("expected_net_value"),
                "계산 운송비": v.get("transport_cost"), "확인 상태": "계산 완료" if v.get("status")=="OPTIMAL" else "추가 검토 필요"}
                for k,v in comparison["results"].items()]), hide_index=True, width="stretch")
            st.caption("예상 수요는 입력한 가정입니다. 이동·할인 허용 범위와 결정 순서가 다르며, 이 결과는 실제 이익이나 절감 실적이 아닙니다.")
        st.caption("대안 비교는 최초 입력을 기준으로 별도 실행합니다. 관측 후 계획과 직접 같은 시점의 결과로 비교하지 마세요.")
        export = frame.assign(자료구분=labels[result["input_data_class"]], 결과상태="이전 조건의 결과" if stale else "현재 조건의 계획",
            판단기준=result.get("adaptive_context", {}).get("decision_at", ""), 실행상태="미실행 · 검토용",
            계획방식={v:k for k,v in POLICY_LABELS.items()}.get(result.get("adaptive_context", {}).get("policy"), "처음 정한 계획"))
        st.download_button("전체 상품 계획 CSV 다운로드", export.to_csv(index=False).encode("utf-8-sig"),
            file_name="Varo_candidate_plan.csv", mime="text/csv", key="candidate_uploaded_download")
        if st.button("입력 자료 검토 이력 저장", key="candidate_uploaded_save"):
            try:
                _save_review(problem, result, "업로드 자료 검토" + (" · 이전 조건의 결과" if stale else ""))
                st.success("격리된 후보 이력에 저장했습니다. 기존 운영 이력은 변경하지 않았습니다.")
            except Exception as exc:
                report_error("uploaded candidate history", exc, "저장하지 못했습니다. 저장 공간과 권한을 확인해 주세요. 화면 결과는 유지합니다.")


def _render_history():
    with st.expander("후보 검토 이력", expanded=False):
        paths = list(HISTORY_DIR.glob("candidate_*.json")) if HISTORY_DIR.exists() else []
        if HISTORY_DIR == _DEFAULT_HISTORY_DIR:
            paths += list((_OUTPUTS / "upgrade_20261016" / "candidate_history").glob("candidate_*.json"))
        paths = sorted(paths, key=lambda p:p.stat().st_mtime, reverse=True)
        records = []
        payloads = []
        for path in paths:
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                record = {"저장시각": row["created_at"], "검토 사례": row["label"],
                    "예상 순현금가치": row["economic_result"]["expected_net_value"], "단계": "분석 계획 · 미승인·미실행", "자료 구분":
                    {"SCENARIO": "가상 시나리오", "TEST": "시험 입력", "REAL": "실제 자료 기반 예상"}.get(row.get("mode"), "자료 구분 확인 필요")}
                if not isinstance(row["products"], list) or not all(isinstance(item, dict) for item in row["products"]):
                    raise ValueError("Candidate history products must be a list of records")
                pd.DataFrame(row["products"])  # Validate detail shape before exposing a selectable record.
                records.append(record)
                payloads.append(row)
            except Exception as exc:
                logger.exception("Candidate history unreadable: %s", path)
                st.warning("일부 후보 검토 기록을 읽지 못했습니다. 관리자에게 확인을 요청해 주세요.")
        if records:
            st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
            if st.checkbox("저장한 상품 상세 보기", key="candidate_history_detail_open"):
                index = st.selectbox("저장 결과 선택", range(len(payloads)), format_func=lambda i: f"{payloads[i]['created_at']} · {payloads[i]['label']}", key="candidate_history_selection")
                saved = payloads[index]
                st.dataframe(pd.DataFrame(saved["products"]), hide_index=True, width="stretch")
                st.caption("저장 당시의 입력·계획입니다. 현재 관측이나 실행 기록으로 간주하지 않습니다.")
                st.download_button("저장 당시 상품 계획 CSV", pd.DataFrame(saved["products"]).assign(결과상태="저장 당시 계획 · 미실행").to_csv(index=False).encode("utf-8-sig"),
                    file_name="Varo_saved_candidate_plan.csv", mime="text/csv", key="candidate_history_download")
        else: st.caption("저장된 후보 검토 이력이 없습니다.")


def render_candidate_review_page():
    if not candidate_enabled():
        st.info("별도 후보 검토 화면에서 먼저 검토 시작에 동의해 주세요.")
        return
    render_page_header(st, "후보 의사결정 검토", "기존 분석을 바꾸지 않고 새로운 처리 계획을 비교합니다.")
    mode = st.radio("자료 선택", ["가상 사례", "실제 자료 검토", "추가 연구"], horizontal=True, key="candidate_data_mode")
    if mode == "추가 연구":
        from components.research_upgrade_view import render_research_upgrade
        render_research_upgrade()
        return
    if mode == "가상 사례": _render_scenarios()
    else: _render_real_intake()
    _render_history()
