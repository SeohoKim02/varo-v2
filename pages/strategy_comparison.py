"""Product-oriented comparison of the existing VHS, Greedy, DQN and Pareto outputs."""
from __future__ import annotations

import html
import math
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_page_header, render_section_header
from components.status import badge_html, user_status_label
from components.tables import format_currency
from services.analysis_pipeline import sort_recommendations
from services.app_state import current_data_status, has_app_data
from services.dqn_service import ACTION_CONCENTRATION_LIMIT, dqn_inference_view


_STRATEGY_META = {
    "VHS": ("수요·재고 균형 우선순위", "#1976f3"),
    "Greedy": ("현재 비용·수량 기준 선택", "#16a56f"),
    "DQN": ("학습 결과 기반 참고 판단", "#7c4dff"),
    "Pareto": ("여러 기준의 비지배 후보", "#f59e0b"),
}


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cost(item: Mapping[str, object]) -> float:
    return _number(item.get("move_cost") or item.get("estimated_cost")) or 0.0


def _saving(item: Mapping[str, object]) -> float:
    return _number(item.get("expected_saving")) or 0.0


def _strategy_sets(recommendations: Sequence[Mapping[str, object]]) -> dict[str, list[dict]]:
    rows = [dict(item) for item in recommendations]
    dqn_unavailable = {"", "-", "미연결", "학습 필요", "비교 불가", "not connected"}
    return {
        "VHS": [item for item in rows if _number(item.get("vhs_rank")) is not None],
        "Greedy": [item for item in rows if bool(item.get("greedy_selected"))],
        "DQN": [
            item for item in rows
            if str(item.get("dqn_action") or "").strip().lower() not in dqn_unavailable
            and str(item.get("dqn_status") or "") not in {"학습 필요", "미연결"}
        ],
        "Pareto": [item for item in rows if _number(item.get("pareto_rank")) == 1.0],
    }


def _filter_recommendations(recommendations: list[dict]) -> list[dict]:
    products = sorted({
        str(item.get("product_name") or item.get("product_id"))
        for item in recommendations if item.get("product_name") or item.get("product_id")
    })
    nodes = sorted({
        str(value)
        for item in recommendations
        for value in (
            item.get("source_name") or item.get("source_id"),
            item.get("target_name") or item.get("target_id"),
            item.get("dc_name") or item.get("dc_id"),
        )
        if value
    })
    product_options = ["전체 상품", *products]
    node_options = ["전체 센터/점포", *nodes]
    if st.session_state.get("strategy_product_filter") not in product_options:
        st.session_state["strategy_product_filter"] = product_options[0]
    if st.session_state.get("strategy_node_filter") not in node_options:
        st.session_state["strategy_node_filter"] = node_options[0]
    controls = st.columns([1.25, 1.25, 0.72], gap="medium")
    product = controls[0].selectbox("상품 범위", product_options, key="strategy_product_filter")
    node = controls[1].selectbox("센터/점포", node_options, key="strategy_node_filter")
    controls[2].button("비교 실행", key="strategy_compare_run", type="primary", width="stretch")
    filtered = []
    for item in recommendations:
        product_name = str(item.get("product_name") or item.get("product_id") or "")
        item_nodes = {
            str(item.get("source_name") or item.get("source_id") or ""),
            str(item.get("target_name") or item.get("target_id") or ""),
            str(item.get("dc_name") or item.get("dc_id") or ""),
        }
        if product != "전체 상품" and product != product_name:
            continue
        if node != "전체 센터/점포" and node not in item_nodes:
            continue
        filtered.append(item)
    return filtered


def _dqn_view(recommendations: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Run DQN inference once per data set and reuse it across reruns.

    VHS, Greedy and Pareto keep their own pipeline values; the DQN column is
    filled only by a real forward pass from a trained model.
    """
    signature = st.session_state.get("data_signature")
    training_result = st.session_state.get("dqn_training_result")
    variant = str(
        (training_result or {}).get("variant")
        or st.session_state.get("dqn_sample_training_mode")
        or "original"
    )
    key = [
        str(signature or ""), len(recommendations), variant,
        str((training_result or {}).get("timestamp") or ""),
    ]
    cached = st.session_state.get("strategy_dqn_view")
    if isinstance(cached, dict) and cached.get("key") == key:
        return dict(cached["view"])
    try:
        view = dqn_inference_view(
            recommendations, signature, training_mode=variant, training_result=training_result,
        )
    except Exception as exc:  # pragma: no cover - comparison never crashes the page
        view = {
            "recommendations": [dict(item) for item in recommendations],
            "available": False,
            "status": "비교 불가",
            "message": f"DQN 추론을 실행하지 못했습니다 ({type(exc).__name__}).",
            "source": "error",
            "model_path": None,
            "model_status": "not_trained",
            "action_concentration": {},
            "average_confidence": None,
        }
        for item in view["recommendations"]:
            item["dqn_action"] = "비교 불가"
            item["dqn_status"] = "비교 불가"
    st.session_state["strategy_dqn_view"] = {"key": key, "view": view}
    return dict(view)


def _render_dqn_status(view: Mapping[str, Any]) -> None:
    concentration = dict(view.get("action_concentration") or {})
    ratio = concentration.get("dominant_ratio")
    if not view.get("available"):
        st.info(f"DQN 비교 불가 · {view.get('message') or '학습된 모델이 없습니다.'}")
        return
    caption = ["DQN은 저장된 모델을 실제로 불러와 추론한 결과입니다."]
    if view.get("model_path"):
        caption.append(f"모델: {str(view['model_path']).rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}")
    if view.get("average_confidence") is not None:
        caption.append(f"평균 신뢰도 {float(view['average_confidence']):.1f}%")
    st.caption(" · ".join(caption))
    if ratio is not None and float(ratio) >= ACTION_CONCENTRATION_LIMIT:
        st.warning(
            f"DQN action이 '{concentration.get('dominant_action')}' 하나로 "
            f"{float(ratio) * 100:.0f}% 쏠려 있습니다. 검토 필요 상태로 비교에서 제외하고 판단하세요."
        )


def _strategy_card(name: str, rows: Sequence[Mapping[str, object]]) -> str:
    subtitle, color = _STRATEGY_META[name]
    status = "비교 가능" if rows else "비교 불가"
    variant = "success" if rows else "warning"
    count = f"{len(rows):,}건" if rows else "-"
    cost = format_currency(sum(_cost(item) for item in rows)) if rows else "-"
    saving = format_currency(sum(_saving(item) for item in rows)) if rows else "-"
    return f"""
      <div class="v2-wrap v2-card v3-strategy-card" style="--strategy-color:{color};">
        <div class="v3-strategy-name">{html.escape(name)}</div>
        <div class="v3-strategy-subtitle">{html.escape(subtitle)}</div>
        <div class="v3-strategy-metric"><span>선택/평가 건수</span><strong>{html.escape(count)}</strong></div>
        <div class="v3-strategy-metric"><span>총 이동 비용</span><strong>{html.escape(cost)}</strong></div>
        <div class="v3-strategy-metric"><span>예상 절감액</span><strong>{html.escape(saving)}</strong></div>
        <div style="margin-top:.7rem;">{badge_html(status, variant)}</div>
      </div>
    """


def _comparison_frame(strategy_sets: Mapping[str, Sequence[Mapping[str, object]]]) -> pd.DataFrame:
    metrics = []
    for label, getter in (
        ("선택/평가 건수", lambda rows: f"{len(rows):,}건" if rows else "비교 불가"),
        ("총 이동 비용", lambda rows: format_currency(sum(_cost(item) for item in rows)) if rows else "비교 불가"),
        ("예상 절감액", lambda rows: format_currency(sum(_saving(item) for item in rows)) if rows else "비교 불가"),
        ("실행 상태", lambda rows: "비교 가능" if rows else "비교 불가"),
    ):
        metrics.append({"지표": label, **{name: getter(rows) for name, rows in strategy_sets.items()}})
    return pd.DataFrame(metrics)


def _candidate_frame(recommendations: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for item in recommendations:
        rows.append({
            "상품": item.get("product_name") or item.get("product_id") or "-",
            "경로": f"{item.get('source_name') or item.get('source_id') or '-'} → {item.get('target_name') or item.get('target_id') or '-'}",
            "VHS 순위": item.get("vhs_rank"),
            "Greedy 순위": item.get("greedy_rank"),
            "Greedy 판단": item.get("greedy_action") or "비교 불가",
            "DQN 판단": user_status_label(item.get("dqn_action") or item.get("dqn_status") or "비교 불가"),
            "Pareto 순위": item.get("pareto_rank"),
            "VARO 최종 순위": item.get("varo_final_rank") or item.get("rank"),
        })
    return pd.DataFrame(rows)


def _render_advanced_tools(pipeline: dict, recommendations: list[dict]) -> None:
    from pages import validation as validation_page

    with st.expander("고급 분석 도구", expanded=False):
        tool = st.selectbox(
            "확인할 분석",
            ["VHS 가중치·신뢰도", "최적성 Gap", "민감도 분석", "전체 샘플 검증", "현재 데이터 진단"],
            key="strategy_advanced_tool",
        )
        if tool == "VHS 가중치·신뢰도":
            validation_page._render_vhs_weights(pipeline)
            validation_page._render_confidence(pipeline)
        elif tool == "최적성 Gap":
            validation_page._render_optimality_gap()
        elif tool == "민감도 분석":
            validation_page._render_detailed_sensitivity(pipeline, recommendations)
        elif tool == "전체 샘플 검증":
            validation_page._render_integrated_validation()
        else:
            validation_page._render_validation_report(pipeline)


def render_strategy_comparison_page() -> None:
    recommendations = sort_recommendations(st.session_state.get("varo_recommendations") or [])
    data_available = has_app_data(st.session_state.get("varo_data"), recommendations)
    render_page_header(
        st,
        "전략 비교",
        "같은 후보를 VHS, Greedy, DQN, Pareto 관점에서 비교하세요.",
        badge=badge_html(current_data_status(st.session_state), "accent" if data_available else "neutral"),
    )
    if not data_available:
        render_empty_state(st, "비교할 데이터가 없습니다", "설정에서 데이터를 불러온 뒤 다시 확인해주세요.")
        return

    dqn_view = _dqn_view(recommendations)
    with st.container(border=True, key="strategy_filters"):
        filtered = _filter_recommendations(dqn_view["recommendations"])
    if not filtered:
        render_empty_state(st, "선택 조건에 맞는 후보가 없습니다", compact=True)
        return
    strategy_sets = _strategy_sets(filtered)
    card_area = st.container(key="strategy_cards")
    cards = card_area.columns(4, gap="medium")
    for column, name in zip(cards, _STRATEGY_META):
        with column:
            st.markdown(_strategy_card(name, strategy_sets[name]), unsafe_allow_html=True)

    render_section_header(st, "전략별 핵심 비교", "현재 계산 결과에 공통으로 존재하는 값만 표시합니다.")
    st.dataframe(_comparison_frame(strategy_sets), hide_index=True, width="stretch")
    _render_dqn_status(dqn_view)
    with st.expander("후보별 상세 판단", expanded=False):
        render_section_header(st, "후보별 판단 차이", "최종 VARO 순위는 서비스 우선·비용 우선·VHS tie-break 원칙을 유지합니다.")
        st.dataframe(_candidate_frame(filtered), hide_index=True, width="stretch", height=310)
    _render_advanced_tools(st.session_state.get("varo_pipeline_result") or {}, filtered)
