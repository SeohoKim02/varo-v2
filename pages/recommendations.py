"""Recommendation execution page for Varo V2."""
from __future__ import annotations

import html

import streamlit as st

from components.cards import render_empty_state, render_page_header, render_recommendation_summary, render_section_header
from components.status import badge_html, route_type_badge, user_status_label
from components.tables import build_recommendation_rows, format_currency, format_number, render_recommendation_table
from services import export_service, upload_quality, v2_summaries
from services.analysis_pipeline import find_recommendation, sort_recommendations
from services.app_state import current_data_status, current_result_basis, has_app_data, resolve_selected_route_id
from services.dqn_service import dqn_result_summary
from services.vhs_score_engine import build_strategy_comparison


def _candidate_detail(route_id: object) -> dict:
    report = st.session_state.get("upload_report") or {}
    return upload_quality.candidate_by_route(report).get(str(route_id), {})


def _reason_detail(recommendation: dict | None) -> dict:
    if not recommendation:
        return {}
    reasons = (st.session_state.get("analysis_result") or {}).get("reason_analysis") or {}
    detail = (reasons.get("reasons") or {}).get(str(recommendation.get("route_id")))
    if detail:
        return detail
    return v2_summaries.recommendation_reason(
        recommendation, st.session_state.get("varo_recommendations") or []
    )

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _all_recommendations() -> list[dict]:
    return sort_recommendations(st.session_state.get("varo_recommendations") or [])


def _options(recommendations: list[dict], key: str, fallback_key: str | None = None) -> list[str]:
    values = []
    for rec in recommendations:
        value = rec.get(key) or (rec.get(fallback_key) if fallback_key else None)
        if value not in (None, ""):
            values.append(str(value))
    return ["전체"] + sorted(set(values))


def _apply_filters(recommendations: list[dict], filters: dict[str, str]) -> list[dict]:
    results = recommendations
    if filters["product"] != "전체":
        results = [rec for rec in results if str(rec.get("product_name")) == filters["product"]]
    if filters["source"] != "전체":
        results = [rec for rec in results if str(rec.get("source_name") or rec.get("source_id")) == filters["source"]]
    if filters["target"] != "전체":
        results = [rec for rec in results if str(rec.get("target_name") or rec.get("target_id")) == filters["target"]]
    if filters["route_type"] != "전체":
        reverse = {"직접 이동": "DIRECT", "DC 경유": "VIA_DC"}
        results = [rec for rec in results if rec.get("route_type") == reverse[filters["route_type"]]]
    if filters["grade"] != "전체":
        results = [rec for rec in results if str(rec.get("recommendation_grade")) == filters["grade"]]
    if filters["transport"] != "전체":
        results = [rec for rec in results if str(rec.get("transport_type")) == filters["transport"]]
    return sort_recommendations(results)


def _render_filters(recommendations: list[dict]) -> dict[str, str]:
    render_section_header(st, "필터", "")
    cols = st.columns(3, gap="small")
    product = cols[0].selectbox("상품", _options(recommendations, "product_name"), key="rec_filter_product")
    source = cols[1].selectbox(
        "출발 점포", _options(recommendations, "source_name", "source_id"), key="rec_filter_source",
    )
    target = cols[2].selectbox(
        "도착 점포", _options(recommendations, "target_name", "target_id"), key="rec_filter_target",
    )
    cols2 = st.columns(3, gap="small")
    route_type = cols2[0].selectbox(
        "경로 유형", ["전체", "직접 이동", "DC 경유"], key="rec_filter_route_type",
    )
    grade = cols2[1].selectbox(
        "추천 등급", _options(recommendations, "recommendation_grade"), key="rec_filter_grade",
    )
    transport = cols2[2].selectbox(
        "이동수단", _options(recommendations, "transport_type"), key="rec_filter_transport",
    )
    return {"product": product, "source": source, "target": target, "route_type": route_type, "grade": grade, "transport": transport}


def _render_best_recommendation(recommendation: dict) -> None:
    render_section_header(st, "1순위 추천", "")
    st.markdown(
        f"""
        <div class="v2-wrap v2-card">
          <div class="v2-card-head"><div>
            <div class="v2-card-title">{html.escape(str(recommendation.get('product_name') or '-'))}</div>
            <div class="v2-card-caption">{html.escape(str(recommendation.get('source_name') or recommendation.get('source_id')))} → {html.escape(str(recommendation.get('target_name') or recommendation.get('target_id')))}</div>
          </div><div>{route_type_badge(str(recommendation.get('route_type')))}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4, gap="small")
    cols[0].metric("추천 수량", format_number(recommendation.get("recommended_qty"), "개"))
    cols[1].metric("예상 절감액", format_currency(recommendation.get("expected_saving")))
    cols[2].metric("이동 거리", format_number(recommendation.get("distance_km"), "km"))
    cols[3].metric("추천 등급", recommendation.get("recommendation_grade") or "-")
    info_items = (
        ("이동수단", recommendation.get("transport_type") or "-"),
        ("경로 유형", recommendation.get("route_type_label") or recommendation.get("route_type") or "-"),
        ("추천 등급", recommendation.get("recommendation_grade") or "-"),
        ("예상 시간", format_number(recommendation.get("expected_time_min") or recommendation.get("travel_time_min"), "분")),
    )
    info_html = "".join(
        f'<div class="v2-info-item"><span class="v2-card-caption">{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in info_items
    )
    st.markdown(f'<div class="v2-wrap v2-card v2-recommendation-info">{info_html}</div>', unsafe_allow_html=True)


def _detailed_comparison_rows(recommendations: list[dict]) -> list[dict]:
    rows = []
    for item in build_strategy_comparison(recommendations):
        rows.append({
            "추천 ID": item.get("route_id"),
            "VHS 점수": item.get("VHS 점수"),
            "VHS 순위": item.get("VHS 순위"),
            "Greedy 전략": item.get("Greedy 전략"),
            "Greedy 순위": item.get("Greedy 순위"),
            "DQN 상태": user_status_label(item.get("DQN 상태")),
            "DQN action": item.get("DQN 전략"),
            "DQN confidence": item.get("DQN confidence"),
            "DQN 참고 점수": item.get("DQN 참고 점수"),
            "DQN 반영": user_status_label(item.get("DQN 반영 여부")),
            "Pareto 순위": item.get("Pareto rank"),
            "Pareto 상태": item.get("Pareto 상태"),
            "Varo 최종 추천": item.get("Varo 최종 추천"),
            "판단 근거": item.get("판단 근거"),
        })
    return rows


def _apply_route_selection() -> None:
    st.session_state["selected_route_id"] = st.session_state.get("recommendation_route_select")
    st.session_state["simulation_snapshot"] = None


def _render_selection(filtered: list[dict]) -> dict | None:
    if not filtered:
        return None
    options = [str(rec["route_id"]) for rec in filtered]
    current = st.session_state.get("selected_route_id")
    index = options.index(current) if current in options else 0
    if (
        "recommendation_route_select" in st.session_state
        and st.session_state.get("recommendation_route_select") != options[index]
    ):
        st.session_state.pop("recommendation_route_select", None)
    selected = st.selectbox(
        "상세 확인할 추천", options, index=index,
        format_func=lambda route_id: f"{route_id} · {find_recommendation(filtered, route_id).get('product_name') if find_recommendation(filtered, route_id) else route_id}",
        key="recommendation_route_select",
        on_change=_apply_route_selection,
    )
    return find_recommendation(filtered, selected)


def _render_downloads(filtered: list[dict]) -> None:
    cols = st.columns([1, 1, 2.2], gap="small")
    cols[0].download_button(
        "현재 추천 CSV",
        data=export_service.recommendations_csv_bytes(filtered),
        file_name="varo_v2_추천결과.csv",
        mime="text/csv",
        width="stretch",
        key="dl_rec_page_csv",
    )
    cols[1].download_button(
        "현재 추천 Excel",
        data=export_service.recommendations_excel_bytes(filtered),
        file_name="varo_v2_추천결과.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_rec_page_xlsx",
    )
    cols[2].caption("현재 필터가 적용된 추천 결과를 내려받습니다.")


def _render_dqn_detail(selected: dict | None) -> None:
    summary = dqn_result_summary(st.session_state.get("dqn_training_result"), st.session_state.get("varo_recommendations") or [])
    status = selected.get("dqn_status") if selected else None
    action = selected.get("dqn_action") if selected else None
    confidence = selected.get("dqn_confidence") if selected else None
    reference = selected.get("dqn_reference_score") if selected else None
    st.markdown(
        f"""
        <div class="v2-wrap v2-card">
          <div class="v2-detail-row"><span class="v2-card-caption">DQN 상태</span><strong>{html.escape(str(status or summary['status']))}</strong></div>
          <div class="v2-detail-row"><span class="v2-card-caption">DQN action</span><strong>{html.escape(str(action or '미연결'))}</strong></div>
          <div class="v2-detail-row"><span class="v2-card-caption">DQN confidence</span><strong>{html.escape(str(confidence if confidence is not None else '-'))}</strong></div>
          <div class="v2-detail-row"><span class="v2-card-caption">DQN 참고 점수</span><strong>{html.escape(str(reference if reference is not None else '-'))}</strong></div>
          <div class="v2-detail-row"><span class="v2-card-caption">반영 방식</span><strong>{html.escape(str(summary['reflection_mode']))}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations_page() -> None:
    data = st.session_state.get("varo_data")
    recommendations = _all_recommendations()
    data_available = has_app_data(data, recommendations)
    render_page_header(
        st, "추천 실행", "",
        badge=badge_html(current_data_status(st.session_state), "accent" if data_available else "neutral"),
    )
    if not data_available:
        render_empty_state(st, "분석 결과가 없습니다. 데이터 관리에서 엑셀 파일을 업로드하고 분석을 실행해주세요.")
        return
    selected_route_id = resolve_selected_route_id(recommendations, st.session_state.get("selected_route_id"))
    if selected_route_id != st.session_state.get("selected_route_id"):
        st.session_state["selected_route_id"] = selected_route_id

    filtered = recommendations
    render_section_header(st, "추천 결과", "")
    top = filtered[0] if filtered else {}
    reason_detail = _reason_detail(top)
    top_reason = (
        reason_detail.get("summary")
        or top.get("final_reason")
        or top.get("reason")
        or next(iter(reason_detail.get("sentences") or []), None)
    )
    if top_reason:
        st.caption(f"1순위 선정 이유: {top_reason}")
    render_recommendation_table(
        build_recommendation_rows(filtered, include_route_id=False, include_status=False),
        key="recommendations_table",
    )
    with st.expander("상세 비교", expanded=False):
        st.dataframe(_detailed_comparison_rows(filtered), hide_index=True, width="stretch")
