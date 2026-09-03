"""Compact Streamlit panel for recording plan execution and outcomes."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Mapping

import streamlit as st

from components.cards import render_section_header
from components.tables import format_currency, format_number
from services.execution_history import (
    REASON_LABELS,
    STATUS_LABELS,
    execution_history_backend_info,
    execution_history_metrics,
    export_execution_history_csv,
    get_recorded_plan,
    list_recorded_plans,
    record_execution_plan,
    update_execution_item,
)


def _display_date(value: Any) -> str:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return text[:16] or "날짜 확인 필요"


def _plan_label(plan: Mapping[str, Any]) -> str:
    return f"{_display_date(plan.get('recorded_at'))} · 이동 {int(plan.get('total_actions') or 0)}건 · 계획 {int(plan.get('total_planned_qty') or 0):,}개"


def _item_label(item: Mapping[str, Any]) -> str:
    source = item.get("source_store_name") or item.get("source_store_id") or "-"
    target = item.get("destination_store_name") or item.get("destination_store_id") or "-"
    product = item.get("product_name") or item.get("product_id") or "상품"
    return f"{source} → {target} · {product} · 계획 {int(item.get('planned_qty') or 0):,}개"


def _optional_text(value: Any) -> str:
    return "" if value is None else str(value)


def _widget_suffix(*values: Any) -> str:
    payload = "|".join(str(value or "") for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _actual_text(value: Any, suffix: str = "") -> str:
    if value is None:
        return "계산 불가"
    if isinstance(value, (int, float)):
        return f"{value:,.0f}{suffix}"
    return str(value)


def _render_comparison(item: Mapping[str, Any]) -> None:
    st.caption("실제값이 입력된 항목만 비교합니다.")
    cols = st.columns(4, gap="small")
    cols[0].metric(
        "계획 / 실제 수량",
        f"{int(item.get('planned_qty') or 0):,} / {_actual_text(item.get('actual_qty'))}",
    )
    cols[1].metric(
        "예상 / 실제 비용",
        f"{format_currency(item.get('expected_cost'))} / {_actual_text(item.get('actual_transport_cost'), '원')}",
    )
    cols[2].metric(
        "예상 / 실제 절감",
        f"{format_currency(item.get('expected_saving'))} / {_actual_text(item.get('actual_saving'), '원')}",
    )
    cols[3].metric(
        "예상 / 실제 순효과",
        f"{format_currency(item.get('expected_net_benefit'))} / {_actual_text(item.get('actual_net_benefit'), '원')}",
    )
    difference = item.get("quantity_difference")
    if difference is not None and difference > 0:
        st.warning(f"계획보다 {int(difference):,}개 많이 실행되었습니다.")


def _render_metrics() -> None:
    metrics = execution_history_metrics()
    if not metrics.get("ok") or int(metrics.get("confirmed_items") or 0) == 0:
        return
    cols = st.columns(4, gap="small")
    cols[0].metric("실행 확인", format_number(metrics.get("confirmed_items"), "건"))
    cols[1].metric("계획 항목 실행률", _actual_text(metrics.get("execution_rate"), "%"))
    cols[2].metric("계획 수량 준수율", _actual_text(metrics.get("quantity_adherence_rate"), "%"))
    samples = int((metrics.get("net_benefit_error") or {}).get("sample_count") or 0)
    cols[3].metric("실제 결과 확인", format_number(samples, "건") if samples else "계산 불가")
    st.caption("실행률은 현장 실행 여부이며 알고리즘 정확도를 뜻하지 않습니다.")


def _render_item_editor(loaded: Mapping[str, Any]) -> None:
    items = list(loaded.get("items") or [])
    if not items:
        st.info("기록된 이동 항목이 없습니다.")
        return
    by_id = {str(item["candidate_id"]): item for item in items}
    plan_id = str(loaded["plan"]["plan_id"])
    plan_suffix = _widget_suffix(plan_id)
    candidate_id = st.selectbox(
        "결과를 기록할 이동",
        list(by_id),
        format_func=lambda value: _item_label(by_id[value]),
        key=f"history_item_select_{plan_suffix}",
    )
    item = by_id[candidate_id]
    item_suffix = _widget_suffix(plan_id, candidate_id)
    status_codes = list(STATUS_LABELS)
    current_status = str(item.get("execution_status") or "unconfirmed")
    status_index = status_codes.index(current_status) if current_status in status_codes else 0
    reason_codes = [""] + list(REASON_LABELS)
    current_reason = str(item.get("nonexecution_reason") or "")
    reason_index = reason_codes.index(current_reason) if current_reason in reason_codes else 0

    with st.form(f"execution_history_update_form_{item_suffix}", clear_on_submit=False):
        cols = st.columns(3, gap="small")
        status = cols[0].selectbox(
            "실제 실행 상태", status_codes, index=status_index,
            format_func=lambda value: STATUS_LABELS.get(value, str(value)), key=f"history_execution_status_{item_suffix}",
        )
        actual_qty = cols[1].text_input(
            "실제 이동 수량", value=_optional_text(item.get("actual_qty")),
            placeholder="실행한 정수 수량", key=f"history_actual_qty_{item_suffix}",
        )
        reason = cols[2].selectbox(
            "미실행·일부 실행 사유", reason_codes, index=reason_index,
            format_func=lambda value: "선택 안 함" if not value else REASON_LABELS.get(value, str(value)),
            key=f"history_reason_{item_suffix}",
        )
        note = st.text_input(
            "짧은 메모 (선택)", value=_optional_text(item.get("operator_note")),
            max_chars=120, key=f"history_note_{item_suffix}",
        )
        with st.expander("사후 결과 선택 입력", expanded=False):
            st.caption("보유한 실제값만 입력하세요. 빈 값은 계산 불가로 유지됩니다.")
            row1 = st.columns(3, gap="small")
            post_source = row1[0].text_input("실행 후 출발 재고", value=_optional_text(item.get("post_source_stock")), key=f"history_post_source_{item_suffix}")
            post_target = row1[1].text_input("실행 후 도착 재고", value=_optional_text(item.get("post_destination_stock")), key=f"history_post_target_{item_suffix}")
            actual_sales = row1[2].text_input("실제 판매량", value=_optional_text(item.get("actual_sales_qty")), key=f"history_sales_{item_suffix}")
            row2 = st.columns(3, gap="small")
            actual_waste = row2[0].text_input("실제 폐기량", value=_optional_text(item.get("actual_waste_qty")), key=f"history_waste_{item_suffix}")
            stockout_values = ["", "no", "yes"]
            current_stockout = "" if item.get("actual_stockout_occurred") is None else "yes" if item.get("actual_stockout_occurred") else "no"
            stockout_occurred = row2[1].selectbox(
                "실제 품절 여부", stockout_values, index=stockout_values.index(current_stockout),
                format_func=lambda value: {"": "기록 안 함", "no": "없음", "yes": "있음"}.get(value, str(value)),
                key=f"history_stockout_occurred_{item_suffix}",
            )
            actual_stockout = row2[2].text_input("실제 품절량", value=_optional_text(item.get("actual_stockout_qty")), key=f"history_stockout_{item_suffix}")
            row3 = st.columns(2, gap="small")
            actual_cost = row3[0].text_input("실제 운송비", value=_optional_text(item.get("actual_transport_cost")), key=f"history_cost_{item_suffix}")
            actual_saving = row3[1].text_input("실제 절감액", value=_optional_text(item.get("actual_saving")), key=f"history_saving_{item_suffix}")
        submitted = st.form_submit_button("실행 결과 저장", type="primary", width="stretch")

    if submitted:
        result = update_execution_item(
            str(loaded["plan"]["plan_id"]), candidate_id, status, actual_qty,
            nonexecution_reason=reason or None,
            operator_note=note,
            outcomes={
                "post_source_stock": post_source,
                "post_destination_stock": post_target,
                "actual_sales_qty": actual_sales,
                "actual_waste_qty": actual_waste,
                "actual_stockout_occurred": stockout_occurred,
                "actual_stockout_qty": actual_stockout,
                "actual_transport_cost": actual_cost,
                "actual_saving": actual_saving,
            },
        )
        if result.get("ok"):
            st.success(result["message"])
            if result.get("warning"):
                st.warning(result["warning"])
        else:
            st.error(result["message"])

    _render_comparison(item)


def render_execution_history_panel(current_plan: Mapping[str, Any] | None) -> None:
    """Render the record action and compact history editor on the recommendation page."""
    plan = dict(current_plan or {})
    has_current_plan = bool(plan.get("plan_id") and plan.get("items") and (plan.get("validation") or {}).get("valid"))
    history = list_recorded_plans()
    stored_plans = list(history.get("plans") or []) if history.get("ok") else []
    if not has_current_plan and not stored_plans and history.get("ok"):
        return

    render_section_header(st, "실행 기록", "추천과 실제 실행 결과를 분리해 남깁니다.")
    backend = execution_history_backend_info()
    st.caption(f"실행 이력 저장: {backend.get('label') or '설정 확인 필요'}")
    if not history.get("ok"):
        st.warning(history.get("message") or "실행 기록을 불러오지 못했습니다.")

    if has_current_plan:
        recorded = get_recorded_plan(str(plan["plan_id"]))
        already_recorded = bool(recorded.get("ok"))
        cols = st.columns([1.2, 3], gap="small")
        if cols[0].button(
            "이 계획 기록", key="record_execution_plan", type="primary",
            width="stretch", disabled=already_recorded,
        ):
            result = record_execution_plan(plan)
            if result.get("ok"):
                st.success(result["message"])
                history = list_recorded_plans()
                stored_plans = list(history.get("plans") or []) if history.get("ok") else []
            else:
                st.error(result["message"])
        cols[1].caption(
            "이미 기록된 실행계획입니다. 아래에서 실제 실행 결과를 수정할 수 있습니다."
            if already_recorded else "버튼을 누를 때만 현재 계획이 기록됩니다. 화면 재실행만으로 저장되지 않습니다."
        )

    if not stored_plans:
        return
    _render_metrics()
    with st.expander("기록된 이동의 실행 결과", expanded=False):
        by_plan = {str(item["plan_id"]): item for item in stored_plans}
        current_id = str(plan.get("plan_id") or "")
        options = list(by_plan)
        index = options.index(current_id) if current_id in options else 0
        selected_plan_id = st.selectbox(
            "기록된 계획", options, index=index,
            format_func=lambda value: _plan_label(by_plan[value]), key="history_plan_select",
        )
        loaded = get_recorded_plan(selected_plan_id)
        if loaded.get("ok"):
            _render_item_editor(loaded)
        else:
            st.warning(loaded.get("message") or "실행 기록을 불러오지 못했습니다.")

        exported = export_execution_history_csv()
        if exported.get("ok") and int(exported.get("row_count") or 0) > 0:
            st.download_button(
                "실행 기록 CSV", data=exported["data"],
                file_name="varo_v2_실행이력.csv", mime="text/csv",
                key="download_execution_history", width="stretch",
            )
            st.caption("사용자가 내려받을 때만 파일이 생성됩니다. 실제값이 없는 칸은 비어 있습니다.")
