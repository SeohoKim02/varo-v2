"""Analysis and validation page for Varo V2."""
from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_page_header, render_section_header
from components.candidate_detail import render_excluded_candidates
from components.state_banner import render_state_action_card
from components.tables import render_capped_table, render_html_table
from components.status import badge_html
from services.home_state import NO_CANDIDATES, READY, build_home_state
from services import export_service, v2_summaries
from services.dqn_service import (
    apply_dqn_result_to_recommendations,
    apply_dqn_reference_to_recommendations,
    can_apply_dqn_to_current_data,
    dqn_display_status,
    dqn_result_summary,
    infer_dqn_actions,
    load_latest_dqn_result,
    save_dqn_result,
    train_dqn,
    get_torch_runtime_status,
    get_torch_status,
    train_dqn_on_recommendations,
)
from services.data_application import load_and_apply
from services.dqn_balanced import generate_balanced_sample
from services.dqn_batch import (
    compare_samples,
    comparison_display_rows,
    save_comparison_report,
    train_dqn_on_balanced_sample,
)
from services.dqn_quality import (
    diagnose_sample,
    diagnosis_progress_label,
    diagnosis_rows,
    quality_display_status,
    run_sequential_diagnosis,
)
from services.kakao_service import kakao_status_label
from services.sample_catalog import discover_dqn_samples, sample_id_from_filename
from services.vhs_score_engine import (
    STRATEGY_CORE_COLUMNS,
    apply_auto_vhs,
    build_strategy_core,
    build_strategy_detail,
)

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

TABS = ["추천 점수", "점수 구성", "비교 분석", "민감도", "DQN 학습", "검증 결과"]


def _pipeline() -> dict:
    value = st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result")
    return value if isinstance(value, dict) else {}


def _validation_status() -> str:
    report = st.session_state.get("varo_validation")
    return getattr(report, "status", "데이터 없음") if report else "데이터 없음"


def _badge_variant(status: str) -> str:
    return {"통과": "success", "주의": "warning", "오류": "error"}.get(status, "neutral")


def _frame(rows) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Number formatting — one decimal for scores, integers for counts, 원 for money.
# --------------------------------------------------------------------------- #
def _score(value) -> str:
    n = _as_number(value)
    return f"{n:.1f}" if n is not None else "-"


def _diff(value) -> str:
    n = _as_number(value)
    return f"{n:+.1f}" if n is not None else "-"


def _pct(value) -> str:
    n = _as_number(value)
    return f"{n:.1f}%" if n is not None else "-"


def _ratio_pct(value) -> str:
    n = _as_number(value)
    return f"{n * 100:.1f}%" if n is not None else "-"


def _int_str(value) -> str:
    n = _as_number(value)
    return f"{int(round(n)):,}" if n is not None else "-"


def _qty(value) -> str:
    n = _as_number(value)
    return f"{int(round(n)):,}개" if n is not None else "-"


def _won(value) -> str:
    n = _as_number(value)
    return f"{n:,.0f}원" if n is not None else "-"


_COMPONENT_NAMES = {
    "savings_score": "절감 효과",
    "disposal_risk_score": "폐기 위험",
    "demand_fit_score": "수요 적합도",
    "inventory_balance_score": "재고 균형",
    "route_cost_score": "이동 비용",
    "feasibility_score": "실행 가능성",
    "promotion_score": "프로모션 효과",
    "greedy_score": "Greedy 기준",
    "confidence_score": "추천 신뢰도",
    "dqn_reference_score": "DQN 참고",
    "expiry_risk_score": "유통기한 위험",
    "sales_score": "판매 가능성",
    "route_efficiency_score": "경로 효율",
}


def _component_name(key) -> str:
    if key in _COMPONENT_NAMES:
        return _COMPONENT_NAMES[key]
    text = str(key or "").replace("_score", "").replace("_", " ").strip()
    return text or str(key)


def _conclusion_card(text: str) -> None:
    """A single light result card (not a repeated blue info box)."""
    st.markdown(
        f'<div class="v2-wrap v2-card"><div class="v2-card-caption">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_hbar(rows, label_key: str, value_key: str) -> None:
    """A light HTML horizontal-bar summary (no Vega, so no chart warnings)."""
    values = [max(0.0, _as_number(row.get(value_key)) or 0.0) for row in rows]
    top = max(values) if values else 1.0
    top = top or 1.0
    cells = []
    for row, value in zip(rows, values):
        width = max(3.0, value / top * 100.0)
        cells.append(
            '<div class="v2-hbar-row">'
            f'<span class="v2-hbar-label">{html.escape(str(row.get(label_key)))}</span>'
            f'<span class="v2-hbar-track"><span class="v2-hbar-fill" style="width:{width:.0f}%"></span></span>'
            f'<span class="v2-hbar-value">{html.escape(str(row.get(value_key)))}</span>'
            '</div>'
        )
    st.markdown('<div class="v2-wrap v2-hbar">' + "".join(cells) + '</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Safe chart / table helpers (explicit Altair y-domain, no st.line_chart).
# --------------------------------------------------------------------------- #
_CHART_AXIS = "#334155"
_CHART_LINE = "#2d6fa8"


def _finite_numbers(values) -> list[float]:
    if values is None:
        return []
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    series = series.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return [float(item) for item in series.tolist()]


def _render_action_distribution(distribution) -> None:
    rows = []
    total = 0
    for key, value in (distribution or {}).items():
        count = pd.to_numeric(value, errors="coerce")
        if pd.isna(count) or float(count) <= 0:
            continue
        rows.append([str(key), int(float(count))])
        total += int(float(count))
    if not rows:
        st.caption("표시할 행동 분포가 없습니다.")
        return
    rows.sort(key=lambda item: item[1], reverse=True)
    table_rows = [
        {"행동": action, "건수": count, "비중": f"{(count / total * 100):.0f}%" if total else "-"}
        for action, count in rows
    ]
    render_html_table(table_rows, ["행동", "건수", "비중"])


def _render_metric_line(values, container=st, x_title: str = "step", value_title: str = "값") -> None:
    clean = _finite_numbers(values)
    if len(clean) < 2:
        if clean:
            container.caption(f"{value_title} 기록 {len(clean)}건 · 현재 값 {clean[-1]:.4f}")
        else:
            container.caption(f"표시할 {value_title} 기록이 없습니다.")
        return
    frame = pd.DataFrame({"step": range(1, len(clean) + 1), "value": clean}).reset_index(drop=True)
    low, high = min(clean), max(clean)
    if low == high:
        pad = abs(low) * 0.05 or 0.5
        domain = [low - pad, high + pad]
    else:
        span = high - low
        domain = [low - span * 0.05, high + span * 0.05]
    chart = (
        alt.Chart(frame)
        .mark_line(color=_CHART_LINE, strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title=x_title,
                    scale=alt.Scale(domain=[1, len(clean)], nice=False),
                    axis=alt.Axis(labelColor=_CHART_AXIS, tickMinStep=1, format="d")),
            y=alt.Y("value:Q", title=None,
                    scale=alt.Scale(domain=domain, nice=False, zero=False),
                    axis=alt.Axis(labelColor=_CHART_AXIS)),
            tooltip=[alt.Tooltip("step:Q", title=x_title, format="d"),
                     alt.Tooltip("value:Q", title=value_title, format=".4f")],
        )
        .properties(width="container", height=170)
    )
    container.altair_chart(chart)


_VHS_DETAIL_HEADERS = {
    "route_id": "추천 ID", "product_name": "상품",
    "source_name": "출발 점포", "target_name": "도착 점포",
    "uploaded_vhs": "업로드 VHS", "recalculated_vhs": "현재 VHS",
    "difference": "차이", "basis": "기준",
    "neutral_components": "기본값 사용", "note": "비고",
}

_SENSITIVITY_DETAIL_HEADERS = {
    "route_id": "추천 ID", "product_name": "상품",
    "sensitivity_cost": "비용 민감도", "sensitivity_distance": "거리 민감도",
    "sensitivity_quantity": "수량 민감도", "sensitivity_vhs": "점수 민감도",
    "overall_sensitivity": "순위 변동 위험", "stability_note": "안정성 비고",
}


# --------------------------------------------------------------------------- #
# Tab 1 · 추천 점수
# --------------------------------------------------------------------------- #
def _render_scores(pipeline: dict) -> None:
    analysis = pipeline.get("vhs_analysis") or {}
    summary = analysis.get("summary") or {}
    recommendations = st.session_state.get("varo_recommendations") or []
    if not analysis and not recommendations:
        render_empty_state(st, "추천 점수 결과가 없습니다", compact=True)
        return
    uploaded = _as_number(analysis.get("uploaded_average"))
    current = _as_number(analysis.get("recalculated_average", summary.get("avg_vhs")))
    difference = round(current - uploaded, 1) if (uploaded is not None and current is not None) else None
    neutral = pipeline.get("vhs_neutral_analysis") or v2_summaries.vhs_neutral_summary(pipeline)
    default_c = int(_as_number(neutral.get("neutral_components")) or 0)

    cols = st.columns(4, gap="small")
    cols[0].metric("현재 VHS 평균", _score(current))
    cols[1].metric("업로드 VHS 평균", _score(uploaded))
    cols[2].metric("평균 차이", _diff(difference))
    cols[3].metric("실제 계산 후보 수", _int_str(len(recommendations)))

    if difference is not None and abs(difference) >= 20:
        message = "업로드 점수와 현재 계산 결과에 차이가 있어 운영 화면은 재계산 점수를 사용합니다."
    elif default_c == 0:
        message = "필요한 데이터가 모두 확인되어 입력값 기준으로 점수를 계산했습니다."
    else:
        message = f"일부 입력값이 없어 {default_c}개 항목에 기본값을 사용했습니다."
    _conclusion_card(message)

    comparison = export_service.vhs_comparison_frame(pipeline, recommendations)
    if not comparison.empty:
        sort_choice = st.radio(
            "정렬", ["현재 VHS 높은 순", "차이 큰 순"], horizontal=True, key="score_sort_order",
        )
        frame = comparison.copy()
        frame["_cur"] = pd.to_numeric(frame.get("recalculated_vhs"), errors="coerce")
        frame["_diff_abs"] = pd.to_numeric(frame.get("difference"), errors="coerce").abs()
        by = "_diff_abs" if sort_choice == "차이 큰 순" else "_cur"
        frame = frame.sort_values(by, ascending=False, na_position="last")
        basic_rows = [
            {
                "순위": rank,
                "상품": row.get("product_name") or "-",
                "출발 점포": row.get("source_name") or "-",
                "도착 점포": row.get("target_name") or "-",
                "업로드 VHS": _score(row.get("uploaded_vhs")),
                "현재 VHS": _score(row.get("recalculated_vhs")),
                "차이": _diff(row.get("difference")),
            }
            for rank, (_, row) in enumerate(frame.iterrows(), start=1)
        ]
        render_html_table(basic_rows[:10], ["순위", "상품", "출발 점포", "도착 점포", "업로드 VHS", "현재 VHS", "차이"])
        with st.expander("전체 결과 보기", expanded=False):
            st.dataframe(
                comparison.rename(columns=_VHS_DETAIL_HEADERS), hide_index=True, width="stretch",
            )


# --------------------------------------------------------------------------- #
# Tab 2 · 점수 구성
# --------------------------------------------------------------------------- #
_COMPONENT_GROUPS = [
    ("절감·프로모션", ("savings_score", "promotion_score")),
    ("위험·폐기", ("disposal_risk_score", "expiry_risk_score")),
    ("수요·판매", ("demand_fit_score", "sales_score")),
    ("비용·경로", ("route_cost_score", "route_efficiency_score")),
    ("실행·균형", ("feasibility_score", "inventory_balance_score")),
    ("참고 지표", ("greedy_score", "confidence_score", "dqn_reference_score")),
]


def _render_score_components(pipeline: dict) -> None:
    st.caption("추천 점수가 어떤 요소로 구성되었는지 확인합니다.")
    analysis = pipeline.get("vhs_analysis") or {}
    weight_rows = analysis.get("weight_rows") or []
    weights = analysis.get("weights") or {}
    contributions = analysis.get("contributions") or {}
    if not weight_rows and not weights:
        render_empty_state(st, "점수 구성 결과가 없습니다", compact=True)
        return
    comp: dict[str, tuple] = {}
    if weight_rows:
        for item in weight_rows:
            comp[item.get("component")] = (_as_number(item.get("average_score")), _as_number(item.get("weight")))
    else:
        for key, value in weights.items():
            comp[key] = (_as_number(contributions.get(key)), _as_number(value))

    group_rows: list[dict] = []
    grouped: set = set()
    for label, members in _COMPONENT_GROUPS:
        present = [m for m in members if m in comp]
        if not present:
            continue
        grouped.update(present)
        scores = [comp[m][0] for m in present if comp[m][0] is not None]
        wsum = sum(comp[m][1] for m in present if comp[m][1] is not None)
        avg = sum(scores) / len(scores) if scores else None
        group_rows.append({"구성 그룹": label, "평균 점수": _score(avg), "반영 비중": _ratio_pct(wsum)})
    other = [key for key in comp if key not in grouped]
    if other:
        scores = [comp[key][0] for key in other if comp[key][0] is not None]
        wsum = sum(comp[key][1] for key in other if comp[key][1] is not None)
        avg = sum(scores) / len(scores) if scores else None
        group_rows.append({"구성 그룹": "기타", "평균 점수": _score(avg), "반영 비중": _ratio_pct(wsum)})
    render_html_table(group_rows, ["구성 그룹", "평균 점수", "반영 비중"])

    with st.expander("후보별 상세 점수 보기", expanded=False):
        recommendations = st.session_state.get("varo_recommendations") or []
        detail_rows = []
        for rec in recommendations:
            row = {"상품": rec.get("product_name") or "-"}
            for key in comp:
                if rec.get(key) is not None:
                    row[_component_name(key)] = _score(rec.get(key))
            detail_rows.append(row)
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")
        st.caption("구성요소(영문 필드): " + ", ".join(f"{_component_name(key)}({key})" for key in comp))
        neutral = pipeline.get("vhs_neutral_analysis") or v2_summaries.vhs_neutral_summary(pipeline)
        st.caption(f"기본값 사용 사유: {neutral.get('neutral_reason', '-')}")


# --------------------------------------------------------------------------- #
# Tab 3 · 비교 분석
# --------------------------------------------------------------------------- #
def _render_comparison(pipeline: dict) -> None:
    greedy = pipeline.get("greedy_analysis") or {}
    pareto = pipeline.get("pareto_analysis") or {}
    recommendations = st.session_state.get("varo_recommendations") or []
    dqn_summary = dqn_result_summary(st.session_state.get("dqn_training_result"), recommendations)
    dqn_ok = dqn_summary["status"] == "정상"
    dqn_cell = "비교 가능" if dqn_ok else "학습 후 비교 가능"

    cards = st.columns(4, gap="small")
    cards[0].metric("Varo 최종 추천", "종합 판단")
    cards[1].metric("Greedy 비교", "즉시 이익")
    cards[2].metric("DQN 상태", dqn_cell)
    cards[3].metric("Pareto 검증", _ratio_pct(pareto.get("front_ratio")) if pareto.get("candidate_count") else "-")
    st.caption(
        "Varo: 비용·위험을 종합한 최종 추천 · Greedy: 즉시 이익이 큰 선택 · "
        "DQN: 학습 결과를 참고하는 비교 · Pareto: 목표 균형 보조 검증"
    )

    dqn_row = "반영" if dqn_ok else "미반영"
    basic_rows = []
    for rank, item in enumerate(recommendations, start=1):
        basic_rows.append({
            "순위": rank,
            "상품": item.get("product_name") or "-",
            "출발": item.get("source_name") or item.get("source_id") or "-",
            "도착": item.get("target_name") or item.get("target_id") or "-",
            "Varo 추천": item.get("varo_action") or "-",
            "Greedy 전략": item.get("greedy_strategy") or item.get("greedy_action") or "-",
            "DQN 상태": dqn_row,
            "Pareto 상태": item.get("pareto_status") or "-",
        })
    render_capped_table(
        basic_rows,
        ["순위", "상품", "출발", "도착", "Varo 추천", "Greedy 전략", "DQN 상태", "Pareto 상태"],
        limit=10,
    )

    with st.expander("상세 비교 보기", expanded=False):
        detail_rows = build_strategy_detail(recommendations)
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")
        if pareto.get("candidate_count"):
            st.caption(
                f"균형 검증: 비지배 후보 {pareto.get('front_size', 0)} / {pareto.get('candidate_count', 0)} · "
                + " · ".join(pareto.get("objective_labels") or [])
            )
        if dqn_ok:
            st.caption("DQN이 현재 데이터와 일치해 낮은 참고 비중으로 점수 비교에 반영됩니다.")


# --------------------------------------------------------------------------- #
# Tab 4 · 민감도
# --------------------------------------------------------------------------- #
def _render_sensitivity_view(pipeline: dict) -> None:
    st.caption("점수 비중이 달라져도 추천 순위가 유지되는지 확인합니다.")
    weight_sensitivity = pipeline.get("weight_sensitivity_analysis") or {}
    recommendations = st.session_state.get("varo_recommendations") or []
    top1 = _as_number(weight_sensitivity.get("top1_retention_rate"))
    volatility = weight_sensitivity.get("rank_volatility")
    fragile = weight_sensitivity.get("fragile_components") or []
    fragile_name = _component_name(fragile[0]) if fragile else "없음"
    stability = pipeline.get("stability_analysis_status") or {}
    stability_status = stability.get("status") or "계산 불가"

    cols = st.columns(4, gap="small")
    cols[0].metric("추천 안정성", stability_status)
    if top1 is None:
        cols[1].metric("상위 추천 유지율", "-")
    elif top1 >= 0.999:
        cols[1].metric("상위 추천 유지율", "유지")
    else:
        cols[1].metric("상위 추천 유지율", f"{top1 * 100:.0f}% 유지")
    cols[2].metric("순위 변동 후보", len(fragile))
    cols[3].metric("가장 민감한 요소", fragile_name)
    for reason in (stability.get("reasons") or [])[:2]:
        st.caption(reason)

    if top1 is not None and top1 >= 0.999:
        _conclusion_card("점수 비중을 조정해도 1순위 추천이 유지되었습니다.")
    elif fragile:
        _conclusion_card(f"{fragile_name} 비중이 높아질 때 1순위 추천이 변경될 수 있습니다.")

    analysis = pipeline.get("sensitivity_analysis") or {}
    rows = analysis.get("rows") or v2_summaries.sensitivity_summary(recommendations)
    if not rows:
        render_empty_state(st, "민감도 요약을 생성할 데이터가 없습니다", compact=True)
        return
    counts = {"높음": 0, "보통": 0, "낮음": 0, "제한적": 0}
    for row in rows:
        key = row.get("overall_sensitivity", "제한적")
        counts[key] = counts.get(key, 0) + 1
    bar_rows = [{"순위 변동 위험": key, "후보 수": value} for key, value in counts.items() if value]
    _render_hbar(bar_rows, "순위 변동 위험", "후보 수")

    with st.expander("상세 결과 보기", expanded=False):
        st.caption("점수 비중을 비중 낮춤 · 기존 비중 · 비중 높임으로 조정해 1순위 유지 여부를 확인합니다.")
        st.dataframe(
            pd.DataFrame(rows).rename(columns=_SENSITIVITY_DETAIL_HEADERS),
            hide_index=True, width="stretch",
        )


# --------------------------------------------------------------------------- #
# DQN training (logic preserved; layout reordered)
# --------------------------------------------------------------------------- #
def _refresh_recommendations_with_dqn(training_result: dict) -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    data_signature = st.session_state.get("data_signature")
    updated = apply_dqn_reference_to_recommendations(recommendations, training_result, data_signature)
    auto_vhs = apply_auto_vhs(pd.DataFrame(updated), training_result)
    if auto_vhs.frame.empty:
        st.session_state["varo_recommendations"] = updated
        return

    clean = auto_vhs.frame.where(pd.notna(auto_vhs.frame), None).to_dict("records")
    st.session_state["varo_recommendations"] = clean
    pipeline = dict(st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result") or {})
    vhs_analysis = dict(pipeline.get("vhs_analysis") or {})
    vhs_analysis.update(auto_vhs.analysis)
    pipeline["vhs_analysis"] = vhs_analysis
    pipeline["vhs_weight_analysis"] = auto_vhs.analysis
    pipeline["vhs_greedy_dqn_comparison"] = auto_vhs.comparison_rows
    greedy = dict(pipeline.get("greedy_analysis") or {})
    greedy["comparison_rows"] = auto_vhs.comparison_rows
    greedy["dqn_status"] = training_result.get("status")
    pipeline["greedy_analysis"] = greedy
    summary = dict(pipeline.get("summary") or {})
    summary["average_vhs_score"] = auto_vhs.analysis.get("vhs_average")
    summary["recommendation_count"] = len(clean)
    pipeline["summary"] = summary
    pipeline["top5"] = sorted(clean, key=lambda row: float(row.get("vhs_rank") or row.get("rank") or 999999))[:5]
    st.session_state["analysis_result"] = pipeline
    st.session_state["varo_pipeline_result"] = pipeline
    st.session_state["pipeline_summary"] = summary


def _dqn_state_label(summary: dict, latest_state: str) -> str:
    status = summary.get("status")
    if status == "정상":
        return "학습 완료 · 현재 데이터와 일치"
    if status in ("검토 필요", "불안정"):
        return "점검 필요"
    if latest_state == "과거 결과(미반영)":
        return "과거 학습 결과"
    return "학습 전"


def _render_torch_line() -> None:
    """PyTorch availability, shown once."""
    runtime = get_torch_runtime_status()
    if runtime["available"]:
        detail = f" · torch {runtime['version']}" if runtime.get("version") else ""
        st.markdown(badge_html(f"GPU 또는 CPU로 DQN 학습을 실행할 수 있습니다{detail}", "success"), unsafe_allow_html=True)
    else:
        st.markdown(badge_html("DQN 학습을 실행하려면 PyTorch가 필요합니다.", "accent"), unsafe_allow_html=True)


def _render_dqn() -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    data_signature = st.session_state.get("data_signature")
    torch_ok, _ = get_torch_status()
    summary = dqn_result_summary(st.session_state.get("dqn_training_result"), recommendations)
    latest = load_latest_dqn_result()
    if not latest:
        latest_state = "없음"
    elif data_signature and latest.get("data_signature") == data_signature:
        latest_state = "현재 데이터와 일치"
    else:
        latest_state = "과거 결과(미반영)"

    # 1) 현재 DQN 상태 (한 카드) + PyTorch 안내 한 번
    render_section_header(st, "현재 DQN 상태", "")
    _render_torch_line()
    status_cols = st.columns(4, gap="small")
    status_cols[0].metric("학습 상태", _dqn_state_label(summary, latest_state))
    status_cols[1].metric("학습 후보 수", summary["candidate_count"])
    status_cols[2].metric("평균 신뢰도", _score(summary["average_confidence"]) if summary["average_confidence"] is not None else "-")
    status_cols[3].metric("반영 방식", summary["reflection_mode"])

    # 2) 현재 샘플 정보
    validation = st.session_state.get("varo_validation")
    sample_summary = getattr(validation, "summary", {}) or {}
    info_cols = st.columns(4, gap="small")
    info_cols[0].metric("점포 / DC", f"{sample_summary.get('store_count', 0)} / {sample_summary.get('dc_count', 0)}")
    info_cols[1].metric("재고 행", sample_summary.get("inventory_count", 0))
    info_cols[2].metric("추천 후보", len(recommendations))
    info_cols[3].metric("최신 학습 결과", latest_state)
    st.caption(f"현재 샘플: {st.session_state.get('uploaded_filename') or '-'}")

    # 3) 학습 실행
    render_section_header(st, "학습 실행", "버튼을 눌렀을 때만 학습합니다.")
    mode = st.radio(
        "DQN 반영 방식",
        ["DQN 참고만", "DQN 약하게 반영"],
        index=0 if st.session_state.get("dqn_reflection_mode", "DQN 참고만") == "DQN 참고만" else 1,
        horizontal=True,
        key="dqn_reflection_mode",
    )
    ctrl = st.columns(3, gap="small")
    episodes = ctrl[0].number_input("반복 학습 횟수", min_value=20, max_value=1200, value=300, step=20)
    learning_rate = ctrl[1].number_input("학습률", min_value=0.0001, max_value=0.05, value=0.001, step=0.0005, format="%.4f")
    candidate_count = ctrl[2].number_input("학습 후보 수", min_value=1, max_value=max(1, len(recommendations)), value=max(1, len(recommendations)), step=1)

    actions = st.columns(3, gap="small")
    if actions[0].button("DQN 학습 실행", type="primary", disabled=not recommendations or not torch_ok, width="stretch"):
        counts = getattr(st.session_state.get("varo_validation"), "summary", {}) or {}
        result = train_dqn(
            recommendations,
            data_signature=data_signature,
            episodes=int(episodes),
            learning_rate=float(learning_rate),
            candidate_count=int(candidate_count),
            reflection_mode=mode,
            sample_id=sample_id_from_filename(st.session_state.get("uploaded_filename") or "sample"),
            store_count=counts.get("store_count"),
            dc_count=counts.get("dc_count"),
        )
        st.session_state["dqn_training_result"] = result.to_dict()
        _refresh_recommendations_with_dqn(st.session_state["dqn_training_result"])
        st.rerun()

    if actions[1].button("저장 결과 불러오기", disabled=not recommendations, width="stretch"):
        loaded = load_latest_dqn_result()
        if loaded and can_apply_dqn_to_current_data(loaded, data_signature):
            st.session_state["dqn_training_result"] = loaded
            _refresh_recommendations_with_dqn(loaded)
            st.rerun()
        elif loaded:
            loaded["status"] = "과거 결과"
            st.session_state["dqn_training_result"] = loaded
            st.warning("현재 데이터와 다른 학습 결과입니다.")
        else:
            st.info("저장된 DQN 결과가 없습니다.")

    if actions[2].button("저장 모델 추론", disabled=not recommendations or not torch_ok, width="stretch"):
        result = infer_dqn_actions(recommendations, data_signature=data_signature)
        saved = save_dqn_result(result)
        st.session_state["dqn_training_result"] = saved
        _refresh_recommendations_with_dqn(saved)
        st.rerun()

    # 4) 학습 결과 (핵심 5개 카드 + 상세 expander)
    result_summary = dqn_result_summary(st.session_state.get("dqn_training_result"), st.session_state.get("varo_recommendations") or [])
    distribution = result_summary.get("action_distribution") or {}
    raw_result = st.session_state.get("dqn_training_result") or {}
    reward_history = raw_result.get("reward_history") or []
    loss_history = _finite_numbers(raw_result.get("loss_history") or [])
    if distribution or loss_history:
        render_section_header(st, "학습 결과", "")
        counts = [int(_as_number(value) or 0) for value in distribution.values()]
        total = sum(counts) or 1
        rcols = st.columns(5, gap="small")
        rcols[0].metric("시작 loss", f"{loss_history[0]:.4f}" if loss_history else "-")
        rcols[1].metric("최종 loss", f"{loss_history[-1]:.4f}" if loss_history else "-")
        rcols[2].metric("action 종류 수", len([value for value in counts if value > 0]))
        rcols[3].metric("최대 action 비율", f"{max(counts) / total * 100:.0f}%" if counts else "-")
        rcols[4].metric("안정성 상태", dqn_display_status(raw_result.get("stability_status") or result_summary.get("status")))
        with st.expander("학습 결과 상세", expanded=False):
            if distribution:
                _render_action_distribution(distribution)
            chart_cols = st.columns(2, gap="small")
            chart_cols[0].caption("reward 흐름")
            _render_metric_line(reward_history, container=chart_cols[0], x_title="후보", value_title="reward")
            chart_cols[1].caption("loss 흐름")
            _render_metric_line(raw_result.get("loss_history") or [], container=chart_cols[1], x_title="episode", value_title="loss")
        st.download_button(
            "학습 결과 다운로드",
            data=json.dumps(raw_result, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            file_name="varo_v2_dqn_학습결과.json",
            mime="application/json",
            width="stretch",
            key="dl_dqn_training_result",
        )

    # 5+6) 진단 · 원본/균형형 비교 (한 expander)
    _render_dqn_diagnosis_expander(torch_ok, episodes, learning_rate, mode)

    with st.expander("DQN 반영 기준", expanded=False):
        st.markdown(
            "- 현재 데이터로 학습하고 안정성 검사를 통과한 결과만 낮은 비중으로 참고 반영합니다.\n"
            "- 데이터가 다르거나 편향이 크면 참고만 하고 최종 추천에는 반영하지 않습니다.\n"
            "- 원본 라벨 쏠림으로 점검 필요가 나오면 균형형 학습 결과와 비교하세요."
        )


def _render_dqn_diagnosis_expander(torch_ok: bool, episodes, learning_rate, mode: str) -> None:
    with st.expander("DQN 학습 데이터 진단 및 비교", expanded=False):
        samples = discover_dqn_samples()
        if not samples:
            st.info("DQN 샘플 폴더를 찾을 수 없습니다.")
            return
        options = {sample.label: sample for sample in samples}
        picked_label = st.selectbox("샘플 선택", list(options), key="dqn_train_sample_select")
        picked = options[picked_label]

        diagnosis = diagnose_sample(picked)
        diag_cols = st.columns(4, gap="small")
        diag_cols[0].metric("데이터 품질", quality_display_status(diagnosis["status"]))
        diag_cols[1].metric("행동 종류", diagnosis["action_kinds"])
        diag_cols[2].metric("최다 비율", f"{float(diagnosis['max_action_ratio']) * 100:.0f}%")
        diag_cols[3].metric("학습 후보 수", diagnosis["candidate_count"])
        picked_dist = diagnosis.get("action_distribution") or {}
        if picked_dist:
            _render_action_distribution(picked_dist)

        train_cols = st.columns(3, gap="small")
        if train_cols[0].button("원본 샘플로 학습", disabled=not torch_ok, width="stretch"):
            path = Path(picked.file_path)
            with st.spinner(f"{picked.label} 원본 학습 중…"):
                if load_and_apply(st.session_state, path, picked.file_name, "DQN 학습 샘플"):
                    applied_recs = st.session_state.get("varo_recommendations") or []
                    counts = getattr(st.session_state.get("varo_validation"), "summary", {}) or {}
                    result = train_dqn(
                        applied_recs,
                        data_signature=st.session_state.get("data_signature"),
                        episodes=int(episodes),
                        learning_rate=float(learning_rate),
                        reflection_mode=mode,
                        sample_id=picked.sample_id,
                        sample_name=picked.file_name,
                        store_count=counts.get("store_count"),
                        dc_count=counts.get("dc_count"),
                        variant="original",
                    )
                    st.session_state["dqn_training_result"] = result.to_dict()
                    _refresh_recommendations_with_dqn(st.session_state["dqn_training_result"])
                    st.rerun()
                else:
                    st.warning("샘플을 적용하지 못해 학습을 건너뛰었습니다.")
        if train_cols[1].button("균형형 샘플 생성", width="stretch"):
            generated = generate_balanced_sample(picked)
            if generated.get("ok"):
                st.success(f"균형형 샘플 생성: {generated['file_name']} · 행동 {generated['action_kinds']}종")
            else:
                st.warning(generated.get("message", "균형형 샘플을 생성하지 못했습니다."))
        if train_cols[2].button("균형형 샘플로 학습", disabled=not torch_ok, width="stretch"):
            with st.spinner(f"{picked.label} 균형형 학습 중…"):
                row, payload = train_dqn_on_balanced_sample(
                    picked, episodes=int(episodes), learning_rate=float(learning_rate),
                )
            if payload:
                st.info(f"균형형 학습 상태: {dqn_display_status(payload['status'])} · 결과 {row['결과 파일']}")
                _render_action_distribution(payload.get("action_distribution") or {})
            else:
                st.warning(row.get("메시지", "균형형 학습을 실행하지 못했습니다."))

        batch_cols = st.columns(2, gap="small")
        if batch_cols[0].button("10개 원본 샘플 순차 진단", width="stretch"):
            total = len(samples)
            progress = st.progress(0.0)
            status_line = st.empty()

            def _on_progress(index, total_count, name, status):
                status_line.caption(diagnosis_progress_label(index, total_count, name, status))
                progress.progress(index / total_count if total_count else 1.0)

            diagnoses = run_sequential_diagnosis(samples, _on_progress)
            status_line.caption(f"진단 완료 · {total}개")
            st.dataframe(pd.DataFrame(diagnosis_rows(diagnoses)), hide_index=True, width="stretch")
        if batch_cols[1].button("원본 vs 균형형 학습 비교", disabled=not torch_ok, width="stretch"):
            total = len(samples) * 2
            progress = st.progress(0.0)
            status_line = st.empty()

            def _on_compare(step, total_count, sample_id, variant):
                label = "원본" if variant == "original" else "균형형"
                status_line.caption(f"{step}/{total_count} · 샘플 {sample_id} {label} 학습 중")
                progress.progress(step / total_count if total_count else 1.0)

            entries = compare_samples(samples, on_progress=_on_compare)
            status_line.caption(f"비교 완료 · 원본/균형형 {len(samples)}쌍")
            st.dataframe(pd.DataFrame(comparison_display_rows(entries)), hide_index=True, width="stretch")
            report_path = save_comparison_report(entries)
            st.caption(f"결과를 {report_path.split(chr(92))[-1]}에 저장했습니다.")


# --------------------------------------------------------------------------- #
# Tab 6 · 검증 결과
# --------------------------------------------------------------------------- #
def _verification_status(pipeline: dict, confidence_avg, match_rate, optimality: dict) -> str:
    if pipeline.get("status") == "partial":
        return "일부 확인 필요"
    if confidence_avg is None:
        return "데이터 부족"
    if optimality and optimality.get("status") in ("비교 불가", "입력 컬럼 부족"):
        return "일부 확인 필요"
    return "확인 완료"


def _render_candidate_status(pipeline: dict) -> None:
    """Status buckets from the candidate ledger; card counts match the real lists."""
    summary = pipeline.get("ledger_summary") or {}
    if not summary.get("generated"):
        return
    render_section_header(st, "후보 판단 요약", "생성된 후보가 어떻게 분류됐는지 확인합니다.")
    top = st.columns(3, gap="small")
    top[0].metric("전체 생성 후보", _int_str(summary.get("generated")))
    top[1].metric("추천 후보", _int_str(summary.get("recommendable_total")))
    top[2].metric("확인 필요", _int_str(summary.get("check_needed")))
    bottom = st.columns(3, gap="small")
    bottom[0].metric("이동 불가", _int_str(summary.get("blocked_move")))
    bottom[1].metric("데이터 부족", _int_str(summary.get("insufficient_data")))
    bottom[2].metric("계산 불가", _int_str(summary.get("not_computable")))
    reasons = summary.get("top_exclusion_reasons") or []
    if reasons:
        st.caption("주요 제외 이유: " + " · ".join(f"{item['reason']} {item['count']}건" for item in reasons))
    render_excluded_candidates(st, pipeline)


def _render_verification(pipeline: dict) -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    _render_candidate_status(pipeline)
    confidence = pipeline.get("confidence_analysis") or {}
    greedy = pipeline.get("greedy_analysis") or {}
    weight_sensitivity = pipeline.get("weight_sensitivity_analysis") or {}
    optimality = (pipeline.get("validation_report") or {}).get("optimality_gap") or {}
    report = st.session_state.get("varo_validation")

    conf_values = [_as_number(item.get("confidence_score")) for item in recommendations]
    conf_values = [value for value in conf_values if value is not None]
    confidence_avg = _as_number(confidence.get("average"))
    if confidence_avg is None and conf_values:
        confidence_avg = sum(conf_values) / len(conf_values)
    top1 = _as_number(weight_sensitivity.get("top1_retention_rate"))
    match_rate = _as_number(greedy.get("strategy_match_rate"))

    status_label = _verification_status(pipeline, confidence_avg, match_rate, optimality)
    cols = st.columns(4, gap="small")
    cols[0].metric("추천 신뢰도", _score(confidence_avg))
    cols[1].metric("순위 안정성", "안정" if (top1 is not None and top1 >= 0.999) else (_ratio_pct(top1) if top1 is not None else "-"))
    cols[2].metric("비교 일치도", _pct(match_rate))
    cols[3].metric("검증 상태", status_label)

    if optimality and optimality.get("status") not in ("비교 불가", "입력 컬럼 부족"):
        _conclusion_card(
            f"최적해 차이 {optimality.get('gap_str', '-')} · "
            f"Varo 비용 {_won(optimality.get('varo_total'))} · 최적 비용 {_won(optimality.get('opt_total'))}"
        )
    else:
        _conclusion_card("추천 신뢰도와 순위 안정성 기준으로 현재 추천 결과를 확인했습니다.")

    with st.expander("검증 기준 자세히 보기", expanded=False):
        render_section_header(st, "분석 요약", "현재 상태를 한 번에 확인합니다.")
        _render_report_summary(pipeline)
        if report:
            st.markdown(badge_html(report.status, _badge_variant(report.status)), unsafe_allow_html=True)
            if report.messages:
                st.dataframe(pd.DataFrame([message.to_dict() for message in report.messages]), hide_index=True, width="stretch")
        if conf_values:
            st.caption("추천 신뢰도 등급: 높음 80 이상 · 보통 60 이상 · 낮음 60 미만")
        deferred = pipeline.get("deferred_algorithms") or []
        if deferred:
            st.caption("보류 항목은 추가 입력이 필요한 보조 기능이며 핵심 추천 결과에는 영향을 주지 않습니다.")

    render_section_header(st, "검증 결과 다운로드", "검증·분석 결과를 파일로 내려받습니다.")
    upload_report = st.session_state.get("upload_report") or {}
    dcols = st.columns([1, 1, 2], gap="small")
    dcols[0].download_button(
        "검증 결과 다운로드",
        data=export_service.validation_report_excel_bytes(report, pipeline, recommendations, upload_report),
        file_name="varo_v2_검증결과.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_validation_report_tab_xlsx",
    )
    dcols[1].download_button(
        "분석 결과 전체 Excel",
        data=export_service.analysis_result_excel_bytes(pipeline, recommendations, upload_report),
        file_name="varo_v2_분석결과.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_analysis_tab_xlsx",
    )
    dcols[2].caption("검증 메시지·알고리즘 연결·최적해 차이·업로드 품질이 포함됩니다.")


def _render_report_summary(pipeline: dict) -> None:
    summary = pipeline.get("summary") or {}
    vhs = pipeline.get("vhs_analysis") or {}
    greedy = pipeline.get("greedy_analysis") or {}
    optimality = (pipeline.get("validation_report") or {}).get("optimality_gap") or {}
    recommendations = st.session_state.get("varo_recommendations") or []
    uploaded = _as_number(vhs.get("uploaded_average"))
    current = _as_number(vhs.get("recalculated_average", summary.get("average_vhs_score")))
    difference = _diff(round(current - uploaded, 1)) if (uploaded is not None and current is not None) else "-"
    dqn_summary = dqn_result_summary(st.session_state.get("dqn_training_result"), recommendations)
    rows = [
        ("적용 파일", st.session_state.get("uploaded_filename") or "-"),
        ("추천 결과 수", _int_str(len(recommendations))),
        ("처리 대상 재고", _int_str(summary.get("total_recommended_qty"))),
        ("예상 절감액", _won(summary.get("total_expected_saving"))),
        ("업로드 점수", _score(uploaded)),
        ("현재 추천 점수", _score(current)),
        ("점수 차이", difference),
        ("비교 일치도", _pct(greedy.get("strategy_match_rate"))),
        ("최적해 차이", optimality.get("gap_str", "-")),
        ("DQN", dqn_display_status(dqn_summary["status"])),
        ("지도", kakao_status_label(st.secrets)),
    ]
    st.dataframe(
        pd.DataFrame([{"항목": key, "값": "-" if value is None else str(value)} for key, value in rows]),
        hide_index=True, width="stretch",
    )


def render_validation_page() -> None:
    pipeline = _pipeline()
    render_page_header(st, "분석 및 검증", "추천 결과의 점수, 비교 결과, 학습 상태를 확인합니다.")

    # Same workspace status as the home screen: only a real result set shows the
    # six algorithm tabs. Every non-result state shows one status card (+ the
    # candidate judgment summary when candidates exist but none are recommendable),
    # never empty tabs/charts or 0-filled validation KPIs.
    home = build_home_state(st.session_state)
    if home.get("state_code") != READY:
        render_state_action_card(home, key="validation_primary_action")
        if home.get("state_code") == NO_CANDIDATES:
            _render_candidate_status(pipeline)
        return

    tabs = st.tabs(TABS)
    renderers = (
        lambda: _render_scores(pipeline),
        lambda: _render_score_components(pipeline),
        lambda: _render_comparison(pipeline),
        lambda: _render_sensitivity_view(pipeline),
        _render_dqn,
        lambda: _render_verification(pipeline),
    )
    for tab, title, renderer in zip(tabs, TABS, renderers):
        with tab:
            renderer()
