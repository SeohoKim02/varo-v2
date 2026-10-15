"""Read-only seller view of the parallel decision contract; no production writes."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from components.cards import render_section_header
from components.status import user_status_label
from components.user_feedback import diagnostic_details, explanation_text, FIELD_LABELS, user_text

STATUS_LABELS = {
    "RECOMMENDATION_AVAILABLE": "검증 조건을 충족한 추천",
    "CONDITIONAL_RECOMMENDATION": "조건 확인 후 검토",
    "PROXY_PLAN_AVAILABLE": "추정 조건 기반 이동 계획",
    "REVIEW_REQUIRED": "확인 필요",
    "ADDITIONAL_INPUT_REQUIRED": "추가 정보 필요",
    "COMPARISON_UNAVAILABLE": "전략 비교 자료 부족",
    "NO_FEASIBLE_TRANSFER": "이동 가능한 계획 없음",
    "INSUFFICIENT_DATA": "분석 자료 부족",
    "NO_ACTION_REQUIRED": "현재 추가 조치 없음",
}
STRATEGIES = {"NORMAL_SALE": "정상판매", "DISCOUNT_SALE": "할인판매", "TRANSFER": "재고 이동"}
MODE_LABELS = {"ACTUAL_OPERATION": "운영 자료 기반", "BENCHMARK_PROXY": "추정 조건 기반",
               "SCENARIO": "예시 조건", "TEST": "테스트 자료"}
EVIDENCE_LABELS = {"ACTUAL_VERIFIED": "실제 자료 확인", "DERIVED_FROM_REAL": "실제 자료에서 계산",
                   "OPERATOR_CONFIRMED": "운영자 입력", "PROXY": "추정 자료", "SCENARIO": "예시 조건",
                   "INSUFFICIENT": "근거 부족"}


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def amount(value: Any, currency: str | None = None) -> str:
    value = number(value)
    if value is None:
        return "계산 불가 · 자료 부족"
    return f"{value:,.0f}" + ("원" if currency == "KRW" else f" {currency}" if currency else " (통화 미확인)")


def quantity(value: Any, unit: str | None = None) -> str:
    value = number(value)
    if value is None:
        return "수량 미확인"
    return f"{value:,.3f}".rstrip("0").rstrip(".") + (f" {unit}" if unit else " (단위 미확인)")


def mode_label(mode: str | None, source_type: str | None = None) -> str:
    if "샘플" in str(source_type or ""):
        return "예시 샘플 · 운영 실증 아님"
    return MODE_LABELS.get(mode, "자료 구분 확인 필요")


def current_contract() -> dict:
    pipeline = st.session_state.get("varo_pipeline_result") or {}
    contract = pipeline.get("final_decision_integration")
    if not isinstance(contract, dict):
        return {}
    run_id = contract.get("run_id")
    if st.session_state.get("seller_view_run_id") != run_id:
        for key in list(st.session_state):
            if key.startswith("seller_") and key.endswith(("_search", "_status", "_evidence", "_need", "_plan", "_sort", "_detail")):
                st.session_state.pop(key, None)
        st.session_state["seller_view_run_id"] = run_id
    return contract


def _name_map(data: Mapping[str, Any]) -> dict[str, str]:
    master = data.get("products")
    if not isinstance(master, pd.DataFrame) or not {"product_id", "product_name"} <= set(master):
        return {}
    # Conflicting names are not silently selected from a different business.
    result = {}
    for code, rows in master.groupby("product_id", dropna=False):
        names = rows.product_name.dropna().astype(str).unique()
        if len(names) == 1:
            result[str(code)] = names[0]
    return result


def _store_names(data: Mapping[str, Any]) -> dict[str, str]:
    stores = data.get("stores")
    if not isinstance(stores, pd.DataFrame):
        return {}
    id_field = "store_id" if "store_id" in stores else "node_id"
    name_field = "store_name" if "store_name" in stores else "node_name"
    if not {id_field, name_field} <= set(stores):
        return {}
    result = {}
    for code, rows in stores.groupby(id_field, dropna=False):
        names = rows[name_field].dropna().astype(str).unique()
        if len(names) == 1:
            result[str(code)] = f"{names[0]} · {code}"
    return result


def product_frame(contract: Mapping[str, Any], data: Mapping[str, Any] | None = None, source_type: str | None = None) -> pd.DataFrame:
    """One display row per contract row, with stable identity and no Top-N truncation."""
    names = _name_map(data or {})
    stores = _store_names(data or {})
    rows = []
    for position, item in enumerate(contract.get("products") or []):
        product = str(item.get("product_id") or "코드 미확인")
        mode = str(item.get("decision_mode") or "")
        evidence = str((item.get("input_quality") or {}).get("effective_evidence") or "INSUFFICIENT")
        lineage = item.get("source_lineage") or []
        dates = {str(v.get("snapshot_date") or v.get("date")) for v in lineage if v.get("snapshot_date") or v.get("date")}
        for source in lineage:
            original = (data or {}).get(source.get("sheet"))
            pos = source.get("row_position")
            if isinstance(original, pd.DataFrame) and isinstance(pos, int) and 0 <= pos < len(original):
                raw = original.iloc[pos]
                for field in ("snapshot_date", "date", "as_of"):
                    if field in raw and pd.notna(raw[field]):
                        dates.add(str(raw[field]))
        rows.append({
            "_position": position, "_key": str(item.get("decision_id") or f"row-{position}"),
            "상품": item.get("product_name") or names.get(product) or product, "상품 코드": product,
            "점포": stores.get(str(item.get("location_id")), str(item.get("location_id") or "점포 미확인")),
            "기준일": str(item.get("date") or ", ".join(sorted(dates)) or "자료 기준"),
            "현재 재고": quantity(item.get("current_stock"), item.get("quantity_unit")),
            "상태": STATUS_LABELS.get(item.get("decision_status"), "확인 필요"),
            "추천·계획": (item.get("display") or {}).get("recommendation_or_plan") or "상세 확인",
            "추가 입력": bool(item.get("missing_required_inputs")),
            "이동 계획": bool(item.get("selected_transfer_routes")),
            "자료 구분": mode_label(mode, source_type),
            "근거": EVIDENCE_LABELS.get(evidence, "근거 확인 필요"),
        })
    return pd.DataFrame(rows)


def filter_products(frame: pd.DataFrame, search="", status="전체 상태", evidence="전체 근거", need=False, plan=False, sort="상품·점포") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index)
    needle = str(search or "").strip()
    if needle:
        haystack = frame["상품"].astype(str)
        for field in ("상품 코드", "점포", "기준일"):
            haystack = haystack.str.cat(frame[field].astype(str), sep=" ")
        mask &= haystack.str.contains(needle, case=False, regex=False)
    if status != "전체 상태":
        mask &= frame["상태"].eq(status)
    if evidence != "전체 근거":
        mask &= frame["근거"].eq(evidence)
    if need:
        mask &= frame["추가 입력"]
    if plan:
        mask &= frame["이동 계획"]
    fields = {"상품·점포": ["상품", "점포"], "상태·상품": ["상태", "상품"], "기준일·상품": ["기준일", "상품"]}[sort]
    return frame.loc[mask].sort_values(fields, kind="stable")


def export_rows(rows: list[dict]) -> bytes:
    """Keep all contract fields, including TEST/SCENARIO and NULL, in CSV exports."""
    flat = [{k: json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list, tuple)) else v
             for k, v in row.items()} for row in rows]
    return pd.DataFrame(flat).to_csv(index=False, na_rep="", lineterminator="\n").encode("utf-8-sig")


def render_summary(contract: Mapping[str, Any]) -> None:
    summary = contract.get("summary") or {}
    with st.container(key="seller_summary"):
        columns = st.columns(4, gap="medium")
        for col, label, field in zip(columns,
            ("분석 상품", "이동 계획 검토", "검증 조건 충족 추천", "추가 입력 필요"),
            ("unique_products", "selected_transfer_routes", "minimum_loss_recommendations", "additional_input_products")):
            col.metric(label, f"{int(summary[field]):,}" if number(summary.get(field)) is not None else "미확인")
    st.caption("읽기 전용 분석 · 점포별 결과 포함 · 계획은 실행 기록이 아닙니다. 기존 생산용 판단은 변경하지 않습니다.")
    st.caption("이동 계획은 선택 경로 수, 추가 입력 필요는 상품·점포 결과 수입니다.")
    if "샘플" in str(st.session_state.get("data_source_type") or ""):
        st.warning("현재 파일은 예시 샘플입니다. 표시된 값은 실제 운영 성과가 아닙니다.")
    if contract.get("context_errors"):
        st.warning("선택 입력 시트 중 적용할 수 없는 항목이 있습니다. 나머지 결과와 구분해 확인하세요.")
        with st.expander("입력 시트 오류 상세"):
            diagnostic_details("seller input errors", contract["context_errors"])
            st.caption("입력 양식의 필수 항목·날짜·중복된 상품과 점포를 확인해 주세요. 적용하지 못한 항목은 분석 완료로 처리하지 않습니다.")
    if contract.get("coverage") and not contract["coverage"].get("pass", False):
        st.error("상품 결과의 누락·중복 또는 연결 문제를 발견했습니다. 운영 판단 전에 전체 결과를 확인하세요.")


def render_strategy_cards(item: Mapping[str, Any], prefix="seller") -> None:
    losses = item.get("expected_loss_by_strategy") or {}
    values = item.get("expected_loss_values") or {}
    available = item.get("available_strategies") or []
    with st.container(key=f"{prefix}_strategy_cards"):
        for col, (strategy, label) in zip(st.columns(3, gap="medium"), STRATEGIES.items()):
            with col, st.container(border=True):
                st.markdown(f"**{label}**")
                value = losses.get(strategy)
                currency = (values.get(strategy) or {}).get("currency")
                st.metric("예상 손실", amount(value, currency))
                st.caption(("비교 가능 · 예시 조건" if prefix == "seller_scenario" else "비교 가능")
                           if strategy in available and number(value) is not None else "비교 불가 · 추천으로 사용하지 않습니다")
                if values.get(strategy):
                    st.caption("금액 출처: " + EVIDENCE_LABELS.get(values[strategy].get('source'), "계산 근거 확인 필요"))
    st.caption("세 전략은 같은 결정 수량·비교 범위에서만 비교합니다. 알 수 없는 손실은 0원으로 바꾸지 않습니다.")


def render_product_detail(item: Mapping[str, Any], contract: Mapping[str, Any], prefix="seller") -> None:
    st.info(explanation_text(item.get("explanation_ko")))
    display = item.get("display") or {}
    st.markdown(f"**검토할 내용:** {display.get('recommendation_or_plan') or '추가 자료 확인'}")
    st.caption(f"자료 구분: {mode_label(item.get('decision_mode'), st.session_state.get('data_source_type'))} · "
               f"실제 실행: {'기록 있음 · 경로별 확인' if item.get('execution_status') == 'ROUTE_DETAILS_ONLY' else '미확인'}")
    selected_ids = set(item.get("selected_transfer_routes") or [])
    routes = [r for r in contract.get("routes") or [] if r.get("decision_id") in (item.get("route_decision_ids") or [])
              or r.get("route_id") in selected_ids]
    selected = [r for r in routes if r.get("selected")]
    stores = _store_names(st.session_state.get("varo_data") or {})
    with st.expander("이동 계획 · 경로와 수량", expanded=bool(selected)):
        if not selected:
            st.caption("계산으로 선택된 출발 이동 계획이 없습니다. 도착 계획은 상세 근거에서 별도로 확인하세요.")
        for route in selected:
            with st.container(border=True):
                source = stores.get(str(route.get('source_id')), route.get('source_id') or '-')
                target = stores.get(str(route.get('target_id')), route.get('target_id') or '-')
                st.markdown(f"**{source} → {target}**")
                cols = st.columns(2)
                cols[0].metric("계획 수량", quantity(route.get("allocated_qty"), route.get("quantity_unit")))
                cost = route.get("transfer_cost") or {}
                cols[1].metric("계산 운송비", amount(route.get("calculated_transfer_cost"), cost.get("currency")))
                st.caption("추정 제약 기반 계획 · 실제 실행을 보장하지 않습니다" if route.get("plan_mode") == "BENCHMARK_PROXY"
                           else "운영 근거 기반 계획 · 실행 전 승인과 조건 확인이 필요합니다")
                validation = route.get("plan_validation") or {}
                execution = {"NOT_RECORDED": "미확인", "EXECUTED": "실행 기록 있음"}.get(route.get("execution_status"), "상세 확인 필요")
                st.caption(f"제약 위반: {validation.get('violation_count', '미확인')} · 실행 기록: {execution}")
                st.caption(explanation_text(route.get("explanation_ko")))
    render_strategy_cards(item, prefix)
    missing = item.get("missing_required_inputs") or []
    with st.expander("필요한 추가 입력", expanded=bool(missing)):
        if missing:
            st.dataframe(pd.DataFrame([{"필요한 값": user_text(v.get("label") or FIELD_LABELS.get(v.get("field")), "추가 자료 확인"),
                                        "안내": {
                                            "ENTER_MISSING": "누락된 값 입력", "CONFIRM_OR_REPLACE": "값과 근거 확인",
                                            "RESOLVE_UNKNOWN_FOR_ROBUST_RANKING": "조건에 따라 판단이 달라져 확인 필요",
                                            "RANKING_DEPENDS_ON_DATA": "전략 비교를 위한 근거 확인",
                                        }.get(v.get("kind") or v.get("request_kind"), "자료와 적용 조건 확인")}
                                       for v in missing]), hide_index=True, width="stretch")
            st.caption("설정의 데이터 교체에서 해당 상품·점포의 판매자 손실 입력 또는 사업장 기본정보 양식을 포함하세요. 이미 유효한 값은 다시 요청하지 않습니다.")
        else:
            st.caption("현재 계약에서 요청하는 추가 입력이 없습니다. 검증 근거 부족과 입력 부족은 서로 다릅니다.")
    with st.expander("근거 · 참고 지표 · 기존 판단"):
        refs = item.get("reference_metrics") or {}
        forecast = item.get("forecast") or {}
        st.markdown("**참고 지표**")
        st.caption(f"ABC 기준: {user_text(refs.get('abc_basis'), '근거 확인 필요')} · 회전율은 검증된 판매와 물류 출고를 구분합니다.")
        st.caption(f"폐기 위험: 규칙 기반 점수 {refs.get('legacy_disposal_score') if refs.get('legacy_disposal_score') is not None else '미확인'} · 실제 확률이 아닙니다.")
        st.caption("수요예측: 기본 방식으로 대체됨 · 자료 조건 확인 필요" if forecast.get('fallback_reason') else "수요예측: 현재 자료 기준 계산 · 외부 자료에 대한 적용성은 확인되지 않았습니다.")
        st.caption(f"기존 판단(참고): {user_status_label(item.get('legacy_action'), '단일 판단 없음')} · 운영자 실제 행동: {user_status_label(item.get('operator_action'), '미확인')}")
        st.caption("실제 지출·매출·손실·절감액은 실행 및 outcome 기록 없이는 확인할 수 없습니다.")
        diagnostic_details("seller decision evidence", {k: item.get(k) for k in ("input_quality", "seller_loss_status", "promotion_status", "comparison_scope",
            "available_strategies", "excluded_strategies", "seller_comparisons", "reference_metrics", "forecast",
            "source_lineage", "reason_codes", "incoming_transfer_routes", "actual_spend", "actual_saving", "outcome")})


def render_downloads(contract: Mapping[str, Any], prefix: str) -> None:
    with st.expander("전체 분석 결과 저장 · 다운로드"):
        st.caption("필터와 관계없이 전체 결과를 저장합니다. CSV의 자료 구분과 입력 근거를 유지합니다.")
        cache = st.session_state.get("seller_export_cache") or {}
        source_type = str(st.session_state.get("data_source_type") or "")
        signature = (contract.get("run_id"), st.session_state.get("data_signature"), source_type)
        source_context = {"ui_data_source_type": source_type, "ui_sample_notice": "예시 샘플 · 운영 실증 아님" if "샘플" in source_type else None}
        if cache.get("signature") != signature:
            cache = {"signature": signature, "payloads": {}}
            st.session_state["seller_export_cache"] = cache
        for suffix, label, rows in (
            ("products", "전체 상품 결과 CSV", contract.get("products") or []),
            ("routes", "전체 이동 경로 CSV", contract.get("routes") or []),
            ("inputs", "필요한 추가 입력 CSV", [{"decision_id": p.get("decision_id"), "product_id": p.get("product_id"),
                "location_id": p.get("location_id"), "decision_mode": p.get("decision_mode"), **v}
                for p in contract.get("products") or [] for v in p.get("missing_required_inputs") or []]),
        ):
            if suffix not in cache["payloads"]:
                cache["payloads"][suffix] = export_rows([{**row, **source_context} for row in rows])
            st.download_button(label, cache["payloads"][suffix], file_name=f"varo_seller_{suffix}.csv", mime="text/csv", key=f"{prefix}_download_{suffix}")
        if "json" not in cache["payloads"]:
            cache["payloads"]["json"] = json.dumps({**contract, "ui_source_context": source_context}, ensure_ascii=False, default=str).encode("utf-8")
        st.download_button("근거·제한사항 포함 전체 JSON", cache["payloads"]["json"],
                           file_name="varo_seller_decisions.json", mime="application/json", key=f"{prefix}_download_contract")


def render_product_results(contract: Mapping[str, Any], prefix="seller") -> None:
    render_section_header(st, "전체 상품 확인", "상품을 찾아 계획·비교 가능 여부·필요한 입력을 확인하세요.")
    frame = product_frame(contract, st.session_state.get("varo_data") or {}, st.session_state.get("data_source_type"))
    if frame.empty:
        st.info("상품별 분석 결과가 없습니다. 데이터를 불러와 다시 확인하세요.")
        return
    with st.container(key=f"{prefix}_filters", border=True):
        cols = st.columns([2, 1, 1])
        search = cols[0].text_input("상품·코드·점포 검색", key=f"{prefix}_search")
        status_options = ["전체 상태", *sorted(frame["상태"].unique())]
        evidence_options = ["전체 근거", *sorted(frame["근거"].unique())]
        for name, options in (("status", status_options), ("evidence", evidence_options)):
            if st.session_state.get(f"{prefix}_{name}") not in options:
                st.session_state[f"{prefix}_{name}"] = options[0]
        status = cols[1].selectbox("결과 상태", status_options, key=f"{prefix}_status")
        evidence = cols[2].selectbox("검증 근거", evidence_options, key=f"{prefix}_evidence")
        cols = st.columns(3)
        need = cols[0].checkbox("추가 입력 필요한 상품만", key=f"{prefix}_need")
        plan = cols[1].checkbox("이동 계획 있는 상품만", key=f"{prefix}_plan")
        sort = cols[2].selectbox("정렬", ["상품·점포", "상태·상품", "기준일·상품"], key=f"{prefix}_sort")
    filtered = filter_products(frame, search, status, evidence, need, plan, sort)
    st.caption(f"전체 {len(frame):,}개 상품·점포 결과 중 {len(filtered):,}개 표시 · 화면 필터는 분석을 다시 실행하지 않습니다.")
    if filtered.empty:
        st.info("검색 조건에 맞는 상품이 없습니다. 검색어나 필터를 바꿔주세요.")
    else:
        st.dataframe(filtered[["상품", "상품 코드", "점포", "기준일", "현재 재고", "상태", "추천·계획", "자료 구분", "근거"]],
                     hide_index=True, width="stretch", height=330,
                     column_config={"상품": st.column_config.TextColumn(width="medium"), "상태": st.column_config.TextColumn(width="medium")})
        # Position, not product code, distinguishes every location/date/duplicate conflict.
        options = filtered["_position"].tolist()
        key = f"{prefix}_detail"
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[0]
        labels = {int(r["_position"]): f"{r['상품']} · {r['점포']} · {r['기준일']}" for _, r in filtered.iterrows()}
        position = st.selectbox("상품 상세 보기", options, format_func=labels.get, key=key)
        render_product_detail(contract["products"][position], contract, prefix)
    render_downloads(contract, prefix)


def scenario_result(case: str, overrides: Mapping[str, float] | None = None) -> dict:
    from services.seller_decision_validation import controlled_scenarios
    from services.seller_loss_inputs import evaluate_with_seller_inputs, SCENARIO
    from services.seller_loss_engine import known
    base = next(s["input"] for s in controlled_scenarios() if s["id"] == case)
    fields = {key: known(value, "SCENARIO_INPUT", "UI explicit scenario", unit=base.input_fields()[key].unit,
                        currency=base.input_fields()[key].currency, dataset="CONTROLLED_SCENARIO")
              for key, value in (overrides or {}).items()}
    return evaluate_with_seller_inputs(replace(base, **fields), None, decision_mode=SCENARIO)


def render_scenario_demo() -> None:
    with st.expander("예시 조건으로 판매 전략 비교 · 실제 운영 결과와 별개"):
        st.warning("SCENARIO · 예시 조건으로 계산한 결과입니다. 실제 운영 추천으로 승격하거나 실행하지 않습니다.")
        choices = {"A": "A · 이동이 유리한 예시", "B": "B · 정상판매가 유리한 예시", "C": "C · 할인판매가 유리한 예시"}
        case = st.selectbox("시연 조건", list(choices), format_func=choices.get, key="seller_scenario_case")
        from services.seller_decision_validation import controlled_scenarios
        base = next(s["input"] for s in controlled_scenarios() if s["id"] == case)
        with st.form(f"seller_scenario_form_{case}"):
            values = {}
            cols = st.columns(3)
            for i, (field, label) in enumerate((("source_normal_price", "정상 판매가격 (원)"),
                ("source_current_stock", "출발 재고 (ea)"), ("remaining_shelf_life_days", "남은 유통기한 (일)"),
                ("discount_rate", "할인율 (0~1 미만)"), ("promotion_uplift", "할인 판매 증가율 (예시)"), ("transfer_cost", "이동비 (원)"))):
                values[field] = cols[i % 3].number_input(label, min_value=0.0,
                    max_value=0.99 if field == "discount_rate" else None, value=float(base.input_fields()[field].value),
                    key=f"seller_scenario_{case}_{field}")
            if st.form_submit_button("예시 계산", type="primary"):
                st.session_state["seller_scenario_result"] = {"case": case, "result": scenario_result(case, values)}
        saved = st.session_state.get("seller_scenario_result") or {}
        if saved.get("case") != case:
            st.caption("예시 계산을 누르면 기존 Seller Loss 엔진으로 세 전략을 비교합니다.")
            return
        result = saved["result"]
        item = {"expected_loss_by_strategy": {s: result.get(f"expected_loss_{s.lower()}") for s in STRATEGIES},
                "expected_loss_values": {s: {"currency": "KRW", "source": "SCENARIO"} for s in STRATEGIES},
                "available_strategies": result.get("comparable_strategies") or []}
        render_strategy_cards(item, "seller_scenario")
        winner = result.get("recommended_strategy")
        st.info(f"예시 조건의 최소손실 전략: {STRATEGIES.get(winner, '비교 불가')}")
        st.caption(explanation_text(result.get("explanation")))
        with st.expander("예시 입력·계산 근거"):
            st.dataframe(pd.DataFrame([{"입력 조건": FIELD_LABELS.get(key, "입력 조건"), "값": value} for key, value in values.items()]), hide_index=True)
            st.caption("입력 출처: 사용자가 지정한 예시 조건 · 실제 관측치가 아닙니다.")
            diagnostic_details("seller scenario result", result)
