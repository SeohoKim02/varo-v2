"""Recommendation execution page for Varo V2."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from components import cards
from components.analysis_progress import AnalysisProgressView, completion_note
from components.cards import render_empty_state, render_page_header, render_recommendation_summary, render_section_header
from components.candidate_detail import (
    ledger_record,
    render_excluded_candidates,
    render_quantity_basis,
    render_source_locations,
)
from components.status import route_type_badge
from components.tables import build_recommendation_rows, format_currency, format_number, render_capped_table
from services import export_service, upload_quality, v2_summaries
from services.analysis_pipeline import find_recommendation, sort_recommendations
from services.app_state import has_app_data, has_applied_data, resolve_selected_route_id
from services.data_application import run_applied_analysis
from services.execution_plan import planned_recommendations
from services.vhs_score_engine import build_strategy_detail


def _safe_txt(value: object) -> str:
    return "-" if value in (None, "") else str(value)


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
    pipeline = _pipeline_result()
    if "execution_plan" in pipeline:
        return planned_recommendations(pipeline)
    return sort_recommendations(st.session_state.get("varo_recommendations") or [])


def _route_option_label(recommendations: list[dict], route_id: str) -> str:
    """User-facing option label hides the internal route_id: '상품 | 출발 → 도착'."""
    route = find_recommendation(recommendations, route_id)
    if not route:
        return str(route_id)
    source = route.get("source_name") or route.get("source_id") or "-"
    target = route.get("target_name") or route.get("target_id") or "-"
    return f"{route.get('product_name') or '상품'} | {source} → {target}"


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


def _pipeline_result() -> dict:
    value = st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result")
    return value if isinstance(value, dict) else {}


def _render_decision_summary() -> None:
    """One action-focused line; formulas and solver details stay internal."""
    pipeline = _pipeline_result()
    plan = pipeline.get("execution_plan") or {}
    cols = st.columns(4, gap="small")
    cols[0].metric("실행할 이동", format_number(plan.get("selected_candidates"), "건"))
    cols[1].metric("총 이동 수량", format_number(plan.get("total_transfer_qty"), "개"))
    cols[2].metric("예상 순효과", format_currency(plan.get("total_net_benefit")))
    attention = int(plan.get("adjusted_candidates") or 0) + len(plan.get("unselected_candidates") or [])
    cols[3].metric("주의 필요", format_number(attention, "건"))
    if plan.get("user_message"):
        st.caption(str(plan.get("user_message")))


def _render_best_recommendation(recommendation: dict) -> None:
    render_section_header(st, "최우선 이동", "")
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
    planned_qty = recommendation.get("planned_qty")
    cols[0].metric("실행 수량", format_number(planned_qty if planned_qty is not None else recommendation.get("recommended_qty"), "개"))
    cols[1].metric("예상 순효과", format_currency(recommendation.get("net_benefit")))
    cols[2].metric("추천 안정성", str(recommendation.get("robustness_status") or "-"))
    cols[3].metric("추천 신뢰도", str(recommendation.get("confidence_level") or "-"))
    route_label = "DC 경유" if recommendation.get("route_type") == "VIA_DC" else "직접 이동"
    info_items = (
        ("이동 경로 방식", route_label),
        ("Greedy 전략", recommendation.get("greedy_strategy") or recommendation.get("greedy_action") or "-"),
        ("최종 처리 전략", recommendation.get("varo_action") or "-"),
        ("추천 등급", recommendation.get("recommendation_grade") or "-"),
    )
    st.caption(f"실행 상태: {recommendation.get('feasibility_status') or '추천 가능'}")
    if recommendation.get("quantity_adjusted"):
        st.caption(str(recommendation.get("selection_reason") or "다른 추천과 재고를 함께 배분해 실행 수량을 조정했습니다."))
    info_html = "".join(
        f'<div class="v2-info-item"><span class="v2-card-caption">{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in info_items
    )
    st.markdown(f'<div class="v2-wrap v2-recommendation-info">{info_html}</div>', unsafe_allow_html=True)
    hint = cards._final_action_hint(recommendation)
    if hint:
        st.caption(hint)
    elif recommendation.get("vhs_vs_greedy_match") is False:
        st.caption("절감액 단일 기준 대신 폐기 위험·수요·비용을 함께 본 종합 점수로 1순위를 정했습니다.")


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
        format_func=lambda route_id: _route_option_label(filtered, route_id),
        key="recommendation_route_select",
        on_change=_apply_route_selection,
    )
    return find_recommendation(filtered, selected)


def _render_downloads(filtered: list[dict]) -> None:
    export_rows = [
        {**row, "recommended_qty": row.get("planned_qty") if row.get("planned_qty") is not None else row.get("recommended_qty")}
        for row in filtered
    ]
    cols = st.columns([1, 1, 2.2], gap="small")
    csv_label = "현재 추천 CSV"
    excel_label = "현재 추천 Excel"
    cols[0].download_button(
        csv_label,
        data=export_service.recommendations_csv_bytes(export_rows),
        file_name="varo_v2_실행계획.csv",
        mime="text/csv",
        width="stretch",
        key="dl_rec_page_csv",
    )
    cols[1].download_button(
        excel_label,
        data=export_service.recommendations_excel_bytes(export_rows),
        file_name="varo_v2_실행계획.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_rec_page_xlsx",
    )
    cols[2].caption("현재 필터가 적용된 실제 실행 수량을 내려받습니다.")


def _render_plan_exclusions(pipeline: dict) -> None:
    plan = pipeline.get("execution_plan") or {}
    rows = []
    candidates = {
        str(item.get("route_id")): item
        for item in (st.session_state.get("varo_recommendations") or [])
    }
    for entry in plan.get("unselected_candidates") or []:
        candidate = candidates.get(str(entry.get("route_id"))) or {}
        rows.append({
            "상품": candidate.get("product_name") or "-",
            "출발": candidate.get("source_name") or candidate.get("source_id") or "-",
            "도착": candidate.get("target_name") or candidate.get("target_id") or "-",
            "경로": "DC 경유" if candidate.get("route_type") == "VIA_DC" else "직접 이동",
            "제외 이유": entry.get("reason") or "전체 이동계획에서 다른 이동을 우선했습니다.",
        })
    if rows:
        with st.expander(f"실행계획에 포함되지 않은 후보 ({len(rows)}건)", expanded=False):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _has_applied_stores(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    stores = data.get("stores")
    empty = getattr(stores, "empty", None)
    return not bool(empty) if empty is not None else False


def render_recommendations_page() -> None:
    data = st.session_state.get("varo_data")
    recommendations = _all_recommendations()
    data_available = has_app_data(data, recommendations)
    render_page_header(st, "오늘 권장 이동", "서로 충돌하지 않는 실제 실행 수량과 순서를 확인합니다.")
    # Shown once, on the first render after a successful run, then dropped so it
    # cannot pile up across reruns.
    if st.session_state.pop("analysis_completed_notice", None):
        st.success(completion_note(st.session_state.get("analysis_elapsed_seconds")))
    if has_applied_data(data) and st.session_state.get("analysis_run_required"):
        quality = st.session_state.get("data_quality_summary") or {}
        excluded = int(quality.get("excluded_rows") or 0)
        if excluded:
            st.caption(f"문제 행 {excluded}개를 제외한 적용 데이터로 추천을 계산합니다.")
        running = bool(st.session_state.get("analysis_running"))
        if st.button(
            "추천 실행", key="run_applied_analysis", type="primary",
            width="stretch", disabled=running,
        ):
            # The progress view streams each real stage while the (blocking) run
            # is in flight, so the screen never looks frozen.
            view = AnalysisProgressView(st)
            succeeded = run_applied_analysis(st.session_state, progress_callback=view.callback)
            view.finish(succeeded, st.session_state.get("analysis_elapsed_seconds"))
            st.rerun()
        error = st.session_state.get("analysis_run_error")
        if error:
            st.error(error)
        st.info("추천 실행 전입니다. 적용된 데이터는 준비되어 있습니다.")
        return
    if not data_available:
        # Data applied but no feasible move (모두 제외): show the excluded candidates
        # here so the home "제외 이유 확인" action lands on real content, not "데이터 없음".
        pipeline = _pipeline_result()
        if _has_applied_stores(data) and (pipeline.get("candidate_ledger") or []):
            summary = pipeline.get("ledger_summary") or {}
            generated = int(summary.get("generated") or 0)
            plan = pipeline.get("execution_plan") or {}
            st.warning(str(plan.get("user_message") or f"현재 조건에서 추천 가능한 이동이 없습니다. 생성된 후보 {generated}건의 제외 이유를 확인하세요."))
            _render_plan_exclusions(pipeline)
            render_excluded_candidates(st, pipeline)
        else:
            render_empty_state(st, "분석 결과가 없습니다. 데이터 관리에서 엑셀 파일을 업로드하고 분석을 실행해주세요.")
        return
    if st.session_state.get("recommendation_source") == "generated":
        st.info("추천 결과 시트가 없어 자동 생성한 후보입니다.")

    selected_route_id = resolve_selected_route_id(recommendations, st.session_state.get("selected_route_id"))
    if selected_route_id != st.session_state.get("selected_route_id"):
        st.session_state["selected_route_id"] = selected_route_id

    filtered = _apply_filters(recommendations, _render_filters(recommendations))
    if not filtered:
        render_empty_state(st, "필터 조건에 맞는 추천 결과가 없습니다", compact=True)
        return

    _render_decision_summary()
    _render_best_recommendation(filtered[0])
    render_section_header(st, "오늘 실행할 이동 · 추천 후보", "")
    render_capped_table(
        build_recommendation_rows(
            filtered, include_route_id=False, include_status=False,
            include_vhs=False, include_grade=True, include_feasibility=True,
        ),
        limit=10,
    )
    with st.expander("상세 비교 보기 (VHS · Greedy · DQN · Pareto)", expanded=False):
        comparison = build_strategy_detail(filtered)
        if comparison:
            st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
        st.caption("넓은 표는 가로로 스크롤하세요.")
    _render_downloads(filtered)

    with st.expander("전체 추천 후보 분석", expanded=False):
        candidate_rows = build_recommendation_rows(
            st.session_state.get("varo_recommendations") or [],
            include_route_id=False, include_status=False, include_vhs=True,
            include_grade=True, include_feasibility=True,
        )
        if candidate_rows:
            st.dataframe(pd.DataFrame(candidate_rows), hide_index=True, width="stretch")

    _render_plan_exclusions(_pipeline_result())

    render_excluded_candidates(st, _pipeline_result())

    render_section_header(st, "선택 후보 상세", "")
    selected = _render_selection(filtered)
    render_recommendation_summary(st, selected)

    record = ledger_record(_pipeline_result(), (selected or {}).get("route_id"))
    render_section_header(st, "추천 판단 근거", "")
    detail = _reason_detail(selected)
    sentences = (record.get("recommendation_reasons") if record else None) or detail.get("sentences") or [
        str((selected or {}).get("reason") or "추천 사유가 없습니다.")
    ]
    for line in sentences[:3]:
        st.markdown(f"- {line}")
    render_quantity_basis(st, record)
    if (selected or {}).get("quantity_adjusted"):
        st.caption(
            f"후보 권장 {(selected or {}).get('recommended_qty'):,.0f}개 중 실제 실행 수량은 "
            f"{(selected or {}).get('planned_qty'):,.0f}개입니다."
        )
    render_source_locations(st, record)

    with st.expander("기술 정보", expanded=False):
        candidate = _candidate_detail(selected.get("route_id")) if selected else {}
        st.markdown(
            f"- 추천 ID: {_safe_txt((selected or {}).get('route_id'))}\n"
            f"- VHS 순위: {_safe_txt((selected or {}).get('vhs_rank'))} · Greedy 순위: {_safe_txt((selected or {}).get('greedy_rank'))}\n"
            f"- Pareto: {_safe_txt((selected or {}).get('pareto_status'))}\n"
            f"- DQN: {_safe_txt((selected or {}).get('dqn_status'))} · {_safe_txt((selected or {}).get('dqn_action'))}"
        )
        if candidate:
            st.caption("경로 선택 근거: " + str(candidate.get("selected_route_basis", "-")))
