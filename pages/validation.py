"""Analysis and validation page for Varo V2."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import re
import copy
from datetime import datetime

from components.cards import render_empty_state, render_page_header, render_section_header
from components.status import badge_html, user_status_label
from services import export_service, v2_summaries
from services.analysis_pipeline import run_analysis_pipeline
from services.app_state import current_result_basis
from services.dqn_service import (
    apply_dqn_reference_to_recommendations,
    build_dqn_batch_comparison_report,
    can_apply_dqn_to_current_data,
    compare_dqn_training_sets,
    dqn_result_summary,
    get_torch_runtime_info,
    load_latest_dqn_result,
    train_dqn,
    train_dqn_batch,
)
from services.dqn_samples import (
    balanced_recommendations,
    build_dqn_training_sets,
    diagnose_dqn_training_sets,
    prepare_dqn_recommendations,
    save_balanced_recommendations,
)
from services.vhs_score_engine import apply_auto_vhs, build_strategy_comparison
from services.sensitivity_service import (
    DEFAULT_VARIABLES,
    MAX_ANALYSIS_CANDIDATES,
    VARIABLE_LABELS,
    build_sensitivity_settings,
    run_detailed_sensitivity,
    sensitivity_detail_frame,
    sensitivity_summary_frame,
)

try:
    import altair as alt
except ImportError:  # pragma: no cover - Altair is normally bundled with Streamlit
    alt = None

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

TABS = ["VHS 분석", "Greedy 비교", "DQN 학습·비교", "Pareto 검증", "민감도/신뢰도"]

_WEIGHT_LABELS = {
    "savings_score": "절감 효과",
    "disposal_risk_score": "폐기 위험",
    "demand_fit_score": "수요 적합",
    "inventory_balance_score": "재고 균형",
    "route_cost_score": "경로 비용",
    "feasibility_score": "실행 가능성",
    "promotion_score": "프로모션 비교",
    "greedy_score": "Greedy 비교",
    "confidence_score": "추천 신뢰도",
    "dqn_reference_score": "DQN 참고",
}

DETAILED_ANALYSIS_VERSION = "v2-validation-2026-07-15.1"


@st.cache_data(show_spinner=False, max_entries=8)
def _detailed_pipeline_cached(
    data_signature: str, data: dict, analysis_version: str,
) -> dict:
    """Build report-only analysis after the validation page is requested."""
    _ = data_signature, analysis_version
    copied = {
        key: value.copy(deep=True) if isinstance(value, pd.DataFrame) else copy.deepcopy(value)
        for key, value in data.items()
    }
    return run_analysis_pipeline(copied, detail_level="full").to_dict()


def _ensure_detailed_pipeline() -> dict:
    pipeline = _pipeline()
    diagnostics = pipeline.get("diagnostics") or {}
    if diagnostics.get("detail_level") == "full":
        return pipeline
    data = st.session_state.get("varo_data") or {}
    if not data:
        return pipeline
    status = st.status("상세 검증 계산 중", expanded=False)
    detailed = _detailed_pipeline_cached(
        str(st.session_state.get("data_signature") or ""),
        data,
        DETAILED_ANALYSIS_VERSION,
    )
    st.session_state["analysis_result"] = detailed
    st.session_state["varo_pipeline_result"] = detailed
    st.session_state["pipeline_summary"] = dict(detailed.get("summary") or {})
    st.session_state["connected_algorithms"] = list(detailed.get("connected_algorithms") or [])
    st.session_state["deferred_algorithms"] = list(detailed.get("deferred_algorithms") or [])
    status.update(label="상세 검증 계산 완료", state="complete", expanded=False)
    return detailed


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


def _display_frame(rows) -> pd.DataFrame:
    """Format DQN status cells for UI while retaining raw session values."""
    frame = _frame(rows).copy()
    for column in (
        "상태", "status", "final_status", "stability_status",
        "DQN 상태", "DQN 반영", "DQN 반영 여부", "VHS 반영 여부",
    ):
        if column in frame.columns:
            frame[column] = frame[column].map(user_status_label)
    return frame


def _as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_VHS_COMPARISON_HEADERS = {
    "route_id": "route_id", "product_name": "상품",
    "source_name": "출발 점포", "target_name": "도착 점포",
    "uploaded_vhs": "업로드 VHS", "recalculated_vhs": "재계산 VHS",
    "difference": "차이", "basis": "기준",
    "neutral_components": "중립값 적용", "note": "비고",
}


def _render_vhs(pipeline: dict) -> None:
    analysis = pipeline.get("vhs_analysis") or {}
    summary = analysis.get("summary") or {}
    recommendations = st.session_state.get("varo_recommendations") or []
    if not summary and not recommendations:
        render_empty_state(st, "VHS 계산 결과가 없습니다", compact=True)
        return
    st.caption("VHS는 최종 우선순위의 기준이며 Greedy, DQN, Pareto 결과는 비교·보조 검증에 사용합니다.")
    uploaded_avg = _as_number(analysis.get("uploaded_average"))
    recalculated_avg = _as_number(analysis.get("recalculated_average", summary.get("avg_vhs")))
    difference = (
        round(recalculated_avg - uploaded_avg, 2)
        if uploaded_avg is not None and recalculated_avg is not None else None
    )
    cols = st.columns(4, gap="small")
    cols[0].metric("업로드 VHS 평균", uploaded_avg if uploaded_avg is not None else "-")
    cols[1].metric("재계산 VHS 평균", recalculated_avg if recalculated_avg is not None else "-")
    cols[2].metric("차이 (재계산 − 업로드)", difference if difference is not None else "-")
    cols[3].metric("중립값 적용 구성요소", len(analysis.get("defaulted_component_columns") or []))

    comparison = export_service.vhs_comparison_frame(pipeline, recommendations)
    if not comparison.empty:
        st.dataframe(
            comparison.rename(columns=_VHS_COMPARISON_HEADERS),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "업로드 VHS는 파일에 포함된 점수이고, 재계산 VHS는 현재 입력 데이터로 다시 계산한 점수입니다."
    )
    st.info(
        "두 값은 계산 기준과 사용 가능한 입력 컬럼 차이 때문에 다를 수 있습니다. "
        "현재 운영 기준은 재계산 VHS이며, 일부 구성요소는 입력 컬럼 부족 시 중립값(50)이 적용됩니다."
    )
    with st.expander("계산 기준", expanded=False):
        st.caption(f"재계산 함수: {analysis.get('calculation_function') or analysis.get('connected_function', '-')}")
        st.caption("현재 운영 기준은 재계산 VHS이며 추가 모델 보정 없이 계산합니다.")

    neutral = pipeline.get("vhs_neutral_analysis") or v2_summaries.vhs_neutral_summary(pipeline)
    render_section_header(st, "VHS 중립값 적용 현황", "입력 컬럼이 부족한 구성요소의 처리 현황입니다.")
    ncols = st.columns(4, gap="small")
    ncols[0].metric("전체 구성요소", neutral.get("total_components", 0))
    ncols[1].metric("실제 계산", neutral.get("calculated_components", 0))
    ncols[2].metric("중립값 적용", neutral.get("neutral_components", 0))
    ncols[3].metric("제외 항목", neutral.get("excluded_components", 0))
    st.info(neutral.get("interpretation", "중립값은 추천이 과도하게 흔들리지 않도록 처리한 기준값입니다."))
    st.caption(f"중립값 적용 사유: {neutral.get('neutral_reason', '-')}")


def _render_vhs_weights(pipeline: dict) -> None:
    analysis = pipeline.get("vhs_analysis") or {}
    weights = analysis.get("weights") or {}
    contributions = analysis.get("contributions") or {}
    if not weights:
        render_empty_state(st, "적용 가중치 결과가 없습니다", compact=True)
        return
    weight_rows = analysis.get("weight_rows") or []
    if weight_rows:
        table = pd.DataFrame(weight_rows)
        rename = {
            "component": "구성 요소",
            "used": "사용 여부",
            "coverage": "사용 가능 비율",
            "missing_rate": "결측률",
            "variation": "분산 신호",
            "weight": "최종 가중치",
            "min_weight": "최소",
            "max_weight": "최대",
            "average_score": "평균 점수",
            "fallback_reason": "fallback",
        }
        columns = [column for column in rename if column in table.columns]
        st.dataframe(table[columns].rename(columns=rename), hide_index=True, width="stretch")
        st.caption("자동 가중치는 현재 데이터 분포, 결측률, 분산 신호를 반영하고 min/max 제한 후 합계 1.0으로 정규화합니다.")
        return
    rows = [
        {"구성 요소": key, "평균 적용 가중치": value, "평균 기여 점수": contributions.get(key, "-")}
        for key, value in sorted(weights.items(), key=lambda item: item[1], reverse=True)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "기존 가중치 정의와 상황별 조정값을 유지했습니다. "
        f"실제 구성요소 {len(analysis.get('available_component_columns') or [])}개 · "
        f"중립값 적용 {len(analysis.get('defaulted_component_columns') or [])}개"
    )


_SENSITIVITY_HEADERS = {
    "route_id": "route_id", "product_name": "상품",
    "sensitivity_cost": "비용 민감도", "sensitivity_distance": "거리 민감도",
    "sensitivity_quantity": "수량 민감도", "sensitivity_vhs": "VHS 민감도",
    "overall_sensitivity": "종합 민감도", "stability_note": "안정성 비고",
}


def _render_sensitivity(pipeline: dict) -> None:
    render_section_header(st, "빠른 민감도 요약", "현재 후보 간 지표 차이로 순위 변동 위험을 빠르게 확인합니다.")
    recommendations = st.session_state.get("varo_recommendations") or []
    analysis = pipeline.get("sensitivity_analysis") or {}
    rows = analysis.get("rows") or v2_summaries.sensitivity_summary(recommendations)
    if not rows:
        render_empty_state(st, "민감도 요약을 생성할 데이터가 없습니다", compact=True)
    else:
        counts = {"높음": 0, "보통": 0, "낮음": 0, "제한적": 0}
        for row in rows:
            level = row.get("overall_sensitivity", "제한적")
            counts[level] = counts.get(level, 0) + 1
        cols = st.columns(4, gap="small")
        cols[0].metric("종합 높음", counts["높음"])
        cols[1].metric("종합 보통", counts["보통"])
        cols[2].metric("종합 낮음", counts["낮음"])
        cols[3].metric("제한적", counts["제한적"])
        st.dataframe(
            pd.DataFrame(rows).rename(columns=_SENSITIVITY_HEADERS),
            hide_index=True,
            width="stretch",
        )
        st.caption("민감도는 다른 후보와의 지표 차이를 기준으로 한 순위 변동 위험입니다.")
    _render_detailed_sensitivity(pipeline, recommendations)


def _clean_chart_frame(frame: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    required = [column for column in numeric_columns if column in cleaned.columns]
    cleaned = cleaned.dropna(subset=required)
    for column in required:
        cleaned = cleaned[cleaned[column].map(lambda value: float("-inf") < float(value) < float("inf"))]
    return cleaned


def _render_sensitivity_line_chart(
    frame: pd.DataFrame,
    *,
    y_column: str,
    y_title: str,
    reverse_y: bool = False,
    currency: bool = False,
) -> None:
    chart_frame = _clean_chart_frame(frame, ["change_pct", y_column])
    if chart_frame.empty:
        st.info("표시할 차트 데이터가 없습니다.")
        return
    if chart_frame["change_pct"].nunique() <= 1 or alt is None:
        st.dataframe(chart_frame[["change_pct", "route_id", y_column]], hide_index=True, width="stretch")
        return
    chart_data = chart_frame[["change_pct", y_column, "route_id", "product_name", "variable"]].copy()
    chart_data["change_pct"] = chart_data["change_pct"].astype(float)
    chart_data[y_column] = chart_data[y_column].astype(float)
    for column in ("route_id", "product_name", "variable"):
        chart_data[column] = chart_data[column].fillna("-").astype(str)
    records = chart_data.to_dict("records")
    y_format = ",.0f" if currency else ".2f"
    y_scale = alt.Scale(reverse=True, zero=False) if reverse_y else alt.Scale(zero=False)
    lines = alt.Chart(alt.Data(values=records)).mark_line(point=True).encode(
        x=alt.X("change_pct:Q", title="변화율 (%)"),
        y=alt.Y(f"{y_column}:Q", title=y_title, scale=y_scale, axis=alt.Axis(format=y_format)),
        color=alt.Color("route_id:N", title="후보"),
        tooltip=[
            alt.Tooltip("variable:N", title="변수"),
            alt.Tooltip("route_id:N", title="route_id"),
            alt.Tooltip("product_name:N", title="상품"),
            alt.Tooltip("change_pct:Q", title="변화율", format="+.1f"),
            alt.Tooltip(f"{y_column}:Q", title=y_title, format=y_format),
        ],
    )
    baseline = alt.Chart(alt.Data(values=[{"change_pct": 0.0}])).mark_rule(
        color="#64748b", strokeDash=[5, 4], opacity=0.8,
    ).encode(x=alt.X("change_pct:Q"))
    st.altair_chart((lines + baseline).properties(height=320), width="stretch")


def _render_detailed_sensitivity_results(result: dict) -> None:
    summary = result.get("summary") or {}
    first = st.columns(3, gap="small")
    first[0].metric("분석 변수 수", summary.get("variable_count", 0))
    first[1].metric("총 시나리오 수", summary.get("scenario_count", 0))
    first[2].metric("Top1 유지율", f"{float(summary.get('top1_retention_rate') or 0):.1f}%")
    second = st.columns(3, gap="small")
    second[0].metric("Top3 유지율", f"{float(summary.get('top3_retention_rate') or 0):.1f}%")
    second[1].metric("최대 순위 변동", summary.get("max_rank_change", 0))
    second[2].metric(
        "상세 민감도 안정성 점수",
        f"{float(summary.get('score') or 0):.1f} / 100",
        summary.get("rating") or "-",
    )
    st.caption(
        "기존 추천 신뢰도는 데이터 품질과 알고리즘 일치도를 기준으로 계산하고, "
        "상세 민감도 안정성은 입력 조건 변화에 대한 순위 유지 정도를 기준으로 계산합니다."
    )

    detail = pd.DataFrame(result.get("detail_rows") or [])
    variable_summary = pd.DataFrame(result.get("variable_summary") or [])
    tab_names = ["종합 요약", "순위 변화", "VHS 점수 변화", "절감액 변화"]
    if result.get("weight_rows"):
        tab_names.append("가중치 민감도")
    tab_names.append("전체 시나리오")
    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))

    with tab_map["종합 요약"]:
        overview = pd.DataFrame([
            {"항목": "가장 민감한 변수", "결과": summary.get("most_sensitive_variable", "-")},
            {"항목": "가장 안정적인 변수", "결과": summary.get("most_stable_variable", "-")},
            {"항목": "Top1 추천이 처음 변경된 조건", "결과": summary.get("first_top1_change", "변경 없음")},
            {"항목": "전체 예상 절감액 최소", "결과": f"{float(summary.get('total_saving_min') or 0):,.0f}원"},
            {"항목": "전체 예상 절감액 최대", "결과": f"{float(summary.get('total_saving_max') or 0):,.0f}원"},
            {"항목": "계산 제외 시나리오", "결과": str(summary.get("excluded_scenario_count", 0))},
        ])
        st.dataframe(overview, hide_index=True, width="stretch")
        if not variable_summary.empty:
            rename = {
                "variable": "변수", "scenario_count": "시나리오 수",
                "top1_retention_rate": "Top1 유지율(%)", "top3_retention_rate": "Top3 유지율(%)",
                "average_abs_rank_change": "평균 절대 순위 변화", "max_rank_change": "최대 순위 변화",
                "max_abs_vhs_change": "최대 VHS 변화", "saving_min": "절감액 최소", "saving_max": "절감액 최대",
            }
            st.dataframe(variable_summary.rename(columns=rename), hide_index=True, width="stretch")
        with st.expander("계산 기준", expanded=False):
            st.markdown(
                "Top1 유지율 40% + Top3 유지율 20% + 평균 순위 안정성 20% + "
                "VHS 점수 변동 안정성 10% + 예상 절감액 변동 안정성 10%로 계산합니다."
            )
            st.caption("0% 기준 시나리오는 기준 일치 확인에 사용하고 안정성 평균에서는 제외합니다.")
            st.caption("85 이상 매우 안정 · 70 이상 안정 · 50 이상 조건에 따라 변동 · 50 미만 민감")

    chart_variables = list(dict.fromkeys(detail.get("variable", pd.Series(dtype=str)).dropna().astype(str)))
    with tab_map["순위 변화"]:
        columns = [
            "variable", "change_pct", "product_name", "base_rank", "changed_rank",
            "rank_delta", "top3_retained", "grade_changed",
        ]
        if not detail.empty:
            table = detail[[column for column in columns if column in detail.columns]].rename(columns={
                "variable": "변수", "change_pct": "변화율(%)", "product_name": "상품",
                "base_rank": "기준 순위", "changed_rank": "변화 후 순위", "rank_delta": "순위 변화",
                "top3_retained": "Top3 유지", "grade_changed": "추천 등급 변화",
            })
            st.dataframe(table, hide_index=True, width="stretch")
        if chart_variables:
            selected = st.selectbox("순위 차트 분석 변수", chart_variables, key="sensitivity_rank_chart_variable")
            top_routes = list(
                detail.loc[detail["variable"] == selected]
                .sort_values("base_rank", kind="mergesort")["route_id"].drop_duplicates().head(5)
            )
            _render_sensitivity_line_chart(
                detail[(detail["variable"] == selected) & detail["route_id"].isin(top_routes)],
                y_column="changed_rank", y_title="변화 후 순위", reverse_y=True,
            )

    with tab_map["VHS 점수 변화"]:
        if chart_variables:
            selected = st.selectbox("VHS 차트 분석 변수", chart_variables, key="sensitivity_vhs_chart_variable")
            filtered = detail[detail["variable"] == selected]
            top_routes = list(filtered.sort_values("base_rank", kind="mergesort")["route_id"].drop_duplicates().head(5))
            _render_sensitivity_line_chart(
                filtered[filtered["route_id"].isin(top_routes)],
                y_column="changed_vhs", y_title="VHS 점수",
            )
            ranges = filtered.groupby(["route_id", "product_name"], as_index=False).agg(
                VHS_최소=("changed_vhs", "min"), VHS_기준=("base_vhs", "first"), VHS_최대=("changed_vhs", "max"),
            )
            st.dataframe(ranges, hide_index=True, width="stretch")

    with tab_map["절감액 변화"]:
        if chart_variables:
            selected = st.selectbox("절감액 차트 분석 변수", chart_variables, key="sensitivity_saving_chart_variable")
            filtered = detail[detail["variable"] == selected]
            top_routes = list(filtered.sort_values("base_rank", kind="mergesort")["route_id"].drop_duplicates().head(5))
            _render_sensitivity_line_chart(
                filtered[filtered["route_id"].isin(top_routes)],
                y_column="changed_saving", y_title="예상 절감액 (원)", currency=True,
            )
            ranges = filtered.groupby(["route_id", "product_name"], as_index=False).agg(
                절감액_최소=("changed_saving", "min"), 절감액_기준=("base_saving", "first"), 절감액_최대=("changed_saving", "max"),
            )
            st.dataframe(ranges, hide_index=True, width="stretch")

    if "가중치 민감도" in tab_map:
        with tab_map["가중치 민감도"]:
            weight_frame = pd.DataFrame(result.get("weight_rows") or [])
            columns = [
                "variable", "change_pct", "base_weight", "requested_weight", "normalized_weight",
                "top1_retained", "max_rank_change", "weight_sum",
            ]
            st.dataframe(
                weight_frame[[column for column in columns if column in weight_frame.columns]].rename(columns={
                    "variable": "요소명", "change_pct": "변화율(%)", "base_weight": "기준 가중치",
                    "requested_weight": "변화된 가중치", "normalized_weight": "정규화 후 가중치",
                    "top1_retained": "Top1 유지", "max_rank_change": "최대 순위 변화", "weight_sum": "가중치 합",
                }),
                hide_index=True, width="stretch",
            )

    with tab_map["전체 시나리오"]:
        st.dataframe(sensitivity_detail_frame(result), hide_index=True, width="stretch")
        with st.expander("내부 메타데이터", expanded=False):
            st.json(result.get("metadata") or {})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloads = st.columns(2, gap="small")
    detail_bytes = sensitivity_detail_frame(result).to_csv(index=False).encode("utf-8-sig")
    summary_bytes = sensitivity_summary_frame(result).to_csv(index=False).encode("utf-8-sig")
    downloads[0].download_button(
        "상세 민감도 결과 CSV", detail_bytes,
        file_name=f"varo_v2_sensitivity_detail_{timestamp}.csv", mime="text/csv", width="stretch",
    )
    downloads[1].download_button(
        "변수별 요약 CSV", summary_bytes,
        file_name=f"varo_v2_sensitivity_summary_{timestamp}.csv", mime="text/csv", width="stretch",
    )


def _render_detailed_sensitivity(pipeline: dict, recommendations: list[dict]) -> None:
    render_section_header(
        st, "상세 민감도 계산",
        "주요 입력값과 VHS 가중치를 변화시켜 추천 순위와 절감액의 안정성을 확인합니다. 계산 결과는 현재 추천을 변경하지 않습니다.",
    )
    st.info("변수와 변화 범위를 선택한 뒤 상세 민감도 계산을 실행할 수 있습니다.")
    data = st.session_state.get("varo_data") or {}
    weights = (pipeline.get("vhs_analysis") or {}).get("weights") or {}
    available_probe = build_sensitivity_settings(recommendations, data, weights, variables=[])
    available = available_probe.get("available_variables") or []
    available_labels = [VARIABLE_LABELS[key] for key in available]
    default_labels = [VARIABLE_LABELS[key] for key in DEFAULT_VARIABLES if key in available]

    st.selectbox("분석 방식", ["개별 변수 분석"], disabled=True, key="sensitivity_method_ui")
    st.caption("한 번에 한 변수씩 분석합니다. 복합 시나리오는 향후 확장 항목입니다.")
    selected_labels = st.multiselect(
        "분석 변수", available_labels, default=default_labels, key="sensitivity_variables_ui",
    )
    unavailable = [VARIABLE_LABELS[key] for key in VARIABLE_LABELS if key not in available]
    if unavailable:
        st.caption("현재 데이터에서 사용할 수 없음: " + ", ".join(unavailable))
    range_name = st.selectbox(
        "변화 범위", ["±5%", "±10%", "±20%", "직접 설정"], index=1, key="sensitivity_range_ui",
    )
    if range_name == "직접 설정":
        custom = st.columns(3, gap="small")
        minimum = custom[0].number_input("최소 변화율 (%)", min_value=-100.0, max_value=100.0, value=-20.0, step=1.0)
        maximum = custom[1].number_input("최대 변화율 (%)", min_value=-100.0, max_value=200.0, value=20.0, step=1.0)
        step_count = custom[2].number_input("단계 수", min_value=2, max_value=41, value=9, step=1)
    else:
        bound = {"±5%": 5.0, "±10%": 10.0, "±20%": 20.0}[range_name]
        minimum, maximum, step_count = -bound, bound, 5
    candidate_name = st.radio(
        "분석 후보", ["상위 5개", "상위 10개", "전체 후보"], index=1, horizontal=True,
        key="sensitivity_candidates_ui",
    )
    candidate_limit = {"상위 5개": 5, "상위 10개": 10, "전체 후보": None}[candidate_name]
    if candidate_limit is None and len(recommendations) > MAX_ANALYSIS_CANDIDATES:
        st.caption(
            f"Streamlit Cloud 응답성을 위해 전체 {len(recommendations)}개 중 상위 {MAX_ANALYSIS_CANDIDATES}개까지 분석합니다."
        )
    selected_keys = [key for key, label in VARIABLE_LABELS.items() if label in selected_labels]
    settings = build_sensitivity_settings(
        recommendations, data, weights, variables=selected_keys,
        minimum_pct=minimum, maximum_pct=maximum, step_count=int(step_count), candidate_limit=candidate_limit,
    )

    current_signature = str(st.session_state.get("data_signature") or "")
    if (
        st.session_state.get("sensitivity_result") is not None
        and str(st.session_state.get("sensitivity_data_signature") or "") != current_signature
    ):
        st.session_state["sensitivity_result"] = None
        st.session_state["sensitivity_summary"] = None
        st.session_state["sensitivity_last_error"] = None

    actions = st.columns([2, 1], gap="small")
    execute = actions[0].button(
        "상세 민감도 계산 실행", type="primary", width="stretch",
        disabled=not selected_keys or bool(st.session_state.get("sensitivity_is_running")),
    )
    clear = actions[1].button("최근 계산 결과 지우기", width="stretch")
    if clear:
        st.session_state["sensitivity_settings"] = {}
        st.session_state["sensitivity_result"] = None
        st.session_state["sensitivity_summary"] = None
        st.session_state["sensitivity_data_signature"] = None
        st.session_state["sensitivity_last_error"] = None

    if execute:
        status = st.status("기준 결과 준비 중", expanded=True)
        progress = st.progress(0.0, text="0 / 0")
        st.session_state["sensitivity_is_running"] = True
        st.session_state["sensitivity_last_error"] = None
        baseline_recommendations = copy.deepcopy(recommendations)

        def update_progress(stage: str, completed: int, total: int, variable: str) -> None:
            ratio = completed / max(1, total)
            progress.progress(ratio, text=f"{variable} · {completed} / {total}")
            status.update(label=stage, state="running", expanded=True)

        try:
            result = run_detailed_sensitivity(
                baseline_recommendations, data, weights, settings, current_signature,
                progress_callback=update_progress,
            )
            st.session_state["sensitivity_settings"] = copy.deepcopy(settings)
            st.session_state["sensitivity_result"] = result
            st.session_state["sensitivity_summary"] = copy.deepcopy(result.get("summary") or {})
            st.session_state["sensitivity_data_signature"] = current_signature
            progress.progress(1.0, text="계산 완료")
            cache_text = " · 동일 조건 캐시 사용" if (result.get("metadata") or {}).get("cache_hit") else ""
            status.update(label=f"상세 민감도 계산 완료{cache_text}", state="complete", expanded=False)
        except Exception as exc:
            st.session_state["sensitivity_last_error"] = str(exc)
            status.update(label="상세 민감도 계산 실패", state="error", expanded=False)
        finally:
            st.session_state["sensitivity_is_running"] = False

    error = st.session_state.get("sensitivity_last_error")
    if error:
        st.error(f"상세 민감도 계산을 완료하지 못했습니다: {error}")
    result = st.session_state.get("sensitivity_result")
    if result and str(st.session_state.get("sensitivity_data_signature") or "") == current_signature:
        _render_detailed_sensitivity_results(result)
    elif not execute:
        st.caption("분석 조건을 선택한 뒤 상세 민감도 계산을 실행하세요.")


def _render_greedy(pipeline: dict) -> None:
    greedy = pipeline.get("greedy_analysis") or {}
    rows = _frame(greedy.get("rows"))
    cols = st.columns(4, gap="small")
    cols[0].metric("비교 후보", greedy.get("comparison_count", 0))
    cols[1].metric("Greedy 1순위", greedy.get("selected_route_id") or "-")
    cols[2].metric("Varo 전략 일치율", f"{greedy.get('strategy_match_rate', 0):.1f}%")
    cols[3].metric("DQN", dqn_result_summary(st.session_state.get("dqn_training_result"), st.session_state.get("varo_recommendations") or [])["status"])
    comparison_rows = build_strategy_comparison(st.session_state.get("varo_recommendations") or [])
    if comparison_rows:
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
    if not rows.empty:
        st.dataframe(rows, hide_index=True, width="stretch")
    with st.expander("계산 기준", expanded=False):
        st.caption(
            f"계산 함수: {greedy.get('calculation_function', '-')} · 정렬: "
            + " → ".join(greedy.get("sort_order") or [])
        )
    st.caption("DQN은 미연결 상태이며 현재 비교값에는 포함되지 않습니다.")


def _refresh_recommendations_with_dqn(training_result: dict) -> None:
    current_recommendations = st.session_state.get("varo_recommendations") or []
    if not st.session_state.get("dqn_baseline_recommendations"):
        st.session_state["dqn_baseline_recommendations"] = copy.deepcopy(current_recommendations)
    if not st.session_state.get("dqn_baseline_pipeline"):
        st.session_state["dqn_baseline_pipeline"] = copy.deepcopy(
            st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result") or {}
        )
    recommendations = copy.deepcopy(st.session_state.get("dqn_baseline_recommendations") or current_recommendations)
    data_signature = st.session_state.get("data_signature")
    updated = apply_dqn_reference_to_recommendations(recommendations, training_result, data_signature)
    applicable_result = training_result if can_apply_dqn_to_current_data(training_result, data_signature) else None
    pipeline = copy.deepcopy(st.session_state.get("dqn_baseline_pipeline") or {})
    validation_report = dict(pipeline.get("validation_report") or {})
    validation_report["dqn_validation"] = {
        "status": training_result.get("final_status") or training_result.get("stability_status") or training_result.get("status"),
        "variant": training_result.get("variant") or training_result.get("training_mode"),
        "signature_match": bool(data_signature and training_result.get("data_signature") == data_signature),
        "vhs_applied": applicable_result is not None,
        "reason": training_result.get("message") or "DQN 학습 후 비교 가능",
    }
    pipeline["validation_report"] = validation_report
    if applicable_result is None:
        # A review/past/insufficient result is comparison-only: never recalculate VHS.
        st.session_state["varo_recommendations"] = updated
        comparison_rows = build_strategy_comparison(updated)
        pipeline["vhs_greedy_dqn_comparison"] = comparison_rows
        greedy = dict(pipeline.get("greedy_analysis") or {})
        greedy["comparison_rows"] = comparison_rows
        greedy["dqn_status"] = training_result.get("status")
        pipeline["greedy_analysis"] = greedy
        st.session_state["analysis_result"] = pipeline
        st.session_state["varo_pipeline_result"] = pipeline
        st.session_state["pipeline_summary"] = dict(pipeline.get("summary") or {})
        st.session_state["connected_algorithms"] = list(pipeline.get("connected_algorithms") or [])
        return
    auto_vhs = apply_auto_vhs(pd.DataFrame(updated), applicable_result)
    if auto_vhs.frame.empty:
        st.session_state["varo_recommendations"] = updated
        return

    clean = auto_vhs.frame.where(pd.notna(auto_vhs.frame), None).to_dict("records")
    st.session_state["varo_recommendations"] = clean
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
    connected = list(pipeline.get("connected_algorithms") or [])
    if "services.dqn_service.apply_dqn_reference_to_recommendations" not in connected:
        connected.append("services.dqn_service.apply_dqn_reference_to_recommendations")
    pipeline["connected_algorithms"] = connected
    st.session_state["analysis_result"] = pipeline
    st.session_state["varo_pipeline_result"] = pipeline
    st.session_state["pipeline_summary"] = summary
    st.session_state["connected_algorithms"] = connected


def _dqn_context() -> tuple[str, int, int]:
    label = str(
        st.session_state.get("dqn_selected_sample")
        or st.session_state.get("uploaded_filename")
        or "current"
    )
    match = re.search(r"(?:sample|샘플)[_ -]?(\d{1,2})", label, flags=re.IGNORECASE)
    sample_id = f"sample_{int(match.group(1)):02d}" if match else "current"
    stores = (st.session_state.get("varo_data") or {}).get("stores")
    if stores is None or stores.empty or "node_type" not in stores.columns:
        return sample_id, 0, 0
    node_types = stores["node_type"].astype(str).str.upper()
    return sample_id, int((node_types == "STORE").sum()), int((node_types == "DC").sum())


def _dqn_mode(selection: str) -> str:
    return "balanced" if selection == "균형형 데이터" else "original"


def _restore_dqn_page_state(snapshot: dict) -> None:
    for key, value in snapshot.items():
        st.session_state[key] = value


def _run_single_dqn(
    recommendations: list[dict],
    data_signature: str | None,
    mode: str,
    episodes: int,
    sample_id: str,
    store_count: int,
    dc_count: int,
) -> bool:
    """Connect the explicit UI action to the existing guarded trainer."""
    protected_keys = (
        "varo_recommendations", "analysis_result", "varo_pipeline_result", "pipeline_summary",
        "connected_algorithms", "dqn_training_result", "dqn_baseline_recommendations",
        "dqn_baseline_pipeline",
    )
    snapshot = {key: copy.deepcopy(st.session_state.get(key)) for key in protected_keys}
    status = st.status("학습 준비 중", expanded=True)
    storage_warning = False
    try:
        status.update(label="데이터 구성 중", state="running")
        prepared = prepare_dqn_recommendations(recommendations, mode)
        if mode == "balanced":
            try:
                save_balanced_recommendations(
                    prepared,
                    sample_id,
                    store_count,
                    dc_count,
                    derived_from=st.session_state.get("uploaded_filename"),
                )
            except Exception:
                storage_warning = True

        def update_stage(label: str) -> None:
            status.update(label=label, state="running")

        result = train_dqn(
            prepared,
            data_signature=data_signature,
            episodes=episodes,
            learning_rate=0.001,
            reflection_mode="DQN 참고만",
            sample_id=sample_id,
            training_mode=mode,
            store_count=store_count,
            dc_count=dc_count,
            seed=17,
            progress_callback=update_stage,
        ).to_dict()
        status.update(label="결과 적용 중", state="running")
        st.session_state["dqn_training_result"] = result
        _refresh_recommendations_with_dqn(result)
        if (result.get("diagnostics") or {}).get("storage_status") in {"session_only", "result_only"}:
            storage_warning = True
        st.session_state["dqn_notice"] = "학습 완료"
        st.session_state["dqn_storage_notice"] = storage_warning
        status.update(label="DQN 학습 완료", state="complete", expanded=False)
        return True
    except Exception:
        _restore_dqn_page_state(snapshot)
        st.session_state["dqn_notice"] = "학습 실패"
        status.update(label="DQN 학습 실패", state="error", expanded=False)
        return False


def _run_dqn_comparison(
    recommendations: list[dict],
    data_signature: str | None,
    episodes: int,
    sample_id: str,
    store_count: int,
    dc_count: int,
) -> bool:
    protected_keys = (
        "varo_recommendations", "analysis_result", "varo_pipeline_result", "pipeline_summary",
        "connected_algorithms", "dqn_training_result", "dqn_comparison_result",
        "dqn_baseline_recommendations", "dqn_baseline_pipeline",
    )
    snapshot = {key: copy.deepcopy(st.session_state.get(key)) for key in protected_keys}
    status = st.status("원본·균형형 비교 준비 중", expanded=True)
    try:
        balanced = balanced_recommendations(recommendations)
        try:
            save_balanced_recommendations(balanced, sample_id, store_count, dc_count)
        except Exception:
            st.session_state["dqn_storage_notice"] = True
        status.update(label="DQN 원본·균형형 학습 중", state="running")
        comparison = compare_dqn_training_sets(
            recommendations,
            balanced,
            str(data_signature or ""),
            episodes=episodes,
            sample_id=sample_id,
            store_count=store_count,
            dc_count=dc_count,
        )
        selected = (
            comparison.get("balanced_result")
            if comparison.get("preferred") == "균형형"
            else comparison.get("original_result")
        ) or comparison.get("balanced_result") or comparison.get("original_result")
        st.session_state["dqn_comparison_result"] = comparison
        if selected:
            st.session_state["dqn_training_result"] = selected
            _refresh_recommendations_with_dqn(selected)
        if comparison.get("storage_status") == "session_only":
            st.session_state["dqn_storage_notice"] = True
        st.session_state["dqn_notice"] = "비교 완료"
        status.update(label="DQN 비교 완료", state="complete", expanded=False)
        return True
    except Exception:
        _restore_dqn_page_state(snapshot)
        st.session_state["dqn_notice"] = "비교 실패"
        status.update(label="DQN 비교 실패", state="error", expanded=False)
        return False


def _format_loss(value) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "-"


def _render_dqn() -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    data_signature = st.session_state.get("data_signature")
    runtime = get_torch_runtime_info()
    torch_ok = bool(runtime["available"])
    sample_id, store_count, dc_count = _dqn_context()

    render_section_header(
        st,
        "DQN 학습 실행",
        "현재 적용된 데이터로 DQN을 학습합니다. 원본 또는 균형형 데이터를 선택할 수 있습니다.",
    )
    setup_cols = st.columns([1.25, 0.8, 1.0], gap="small")
    with setup_cols[0]:
        training_data = st.radio(
            "학습 데이터 선택",
            ("원본 데이터", "균형형 데이터"),
            horizontal=True,
            key="dqn_single_training_data",
        )
    with setup_cols[1]:
        episodes = int(st.number_input(
            "학습 에피소드",
            min_value=20,
            max_value=500,
            value=80,
            step=10,
            key="dqn_single_episodes",
        ))
    with setup_cols[2]:
        st.markdown("**학습 상태**")
        st.markdown(
            badge_html(str(runtime["status"]), "success" if torch_ok else "warning"),
            unsafe_allow_html=True,
        )
        if torch_ok:
            st.caption(f"실행 장치: {runtime['device']} · PyTorch {runtime['version']}")
        else:
            st.caption(str(runtime["message"]))

    selected_mode = _dqn_mode(training_data)
    actions = st.columns([1.4, 1.0, 1.0], gap="small")
    if actions[0].button(
        "DQN 학습 실행",
        key="dqn_primary_train",
        type="primary",
        disabled=not recommendations or not torch_ok,
        width="stretch",
    ):
        _run_single_dqn(
            recommendations, data_signature, selected_mode, episodes,
            sample_id, store_count, dc_count,
        )
        st.rerun()
    if actions[1].button(
        "DQN 원본 vs 균형형 비교",
        key="dqn_primary_compare",
        disabled=not recommendations or not torch_ok,
        width="stretch",
    ):
        _run_dqn_comparison(
            recommendations, data_signature, episodes, sample_id, store_count, dc_count,
        )
        st.rerun()
    if actions[2].button(
        "최근 학습 결과 불러오기",
        key="dqn_load_latest",
        disabled=not recommendations,
        width="stretch",
    ):
        latest = load_latest_dqn_result(data_signature, selected_mode)
        if latest:
            st.session_state["dqn_training_result"] = latest
            _refresh_recommendations_with_dqn(latest)
            st.session_state["dqn_notice"] = "최근 결과 불러오기 완료"
        else:
            st.session_state["dqn_notice"] = "최근 결과 없음"
        st.rerun()

    with st.expander("고급 학습 옵션", expanded=False):
        advanced = st.columns(2, gap="small")
        if advanced[0].button(
            "DQN 원본 학습 실행",
            key="dqn_advanced_original",
            disabled=not recommendations or not torch_ok,
            width="stretch",
        ):
            _run_single_dqn(
                recommendations, data_signature, "original", episodes,
                sample_id, store_count, dc_count,
            )
            st.rerun()
        if advanced[1].button(
            "DQN 균형형 학습 실행",
            key="dqn_advanced_balanced",
            disabled=not recommendations or not torch_ok,
            width="stretch",
        ):
            _run_single_dqn(
                recommendations, data_signature, "balanced", episodes,
                sample_id, store_count, dc_count,
            )
            st.rerun()

    current_diagnosis = diagnose_dqn_training_sets([{
        "sample_id": sample_id,
        "sample_name": st.session_state.get("uploaded_filename") or "현재 데이터",
        "mode": selected_mode,
        "recommendations": prepare_dqn_recommendations(recommendations, selected_mode),
    }])[0] if recommendations else {}
    render_section_header(st, "현재 샘플 진단", "학습을 시작하기 전에 후보 수와 라벨 편향을 확인합니다.")
    if st.button(
        "현재 샘플 진단",
        key="dqn_current_sample_diagnosis",
        disabled=not recommendations,
        width="stretch",
    ):
        st.success("현재 샘플 진단을 갱신했습니다.")
    diagnostic_cols = st.columns(3, gap="small")
    diagnostic_cols[0].metric("후보 수", current_diagnosis.get("candidate_count", 0))
    diagnostic_cols[1].metric("학습 라벨 종류", current_diagnosis.get("target_type_count", 0))
    diagnosis_status = user_status_label(current_diagnosis.get("status", "학습 필요"))
    compact_status = "확인 필요" if diagnosis_status == "비교 전 데이터 확인 필요" else diagnosis_status
    diagnostic_cols[2].metric("데이터 상태", compact_status)
    if compact_status != diagnosis_status:
        diagnostic_cols[2].caption(diagnosis_status)

    if selected_mode == "original" and current_diagnosis.get("status") == "검토 필요":
        st.warning("원본 데이터의 학습 라벨이 한쪽으로 치우쳐 있습니다.")
    elif selected_mode == "balanced" and current_diagnosis.get("status") == "비교 가능":
        st.success("균형형 데이터로 안정적인 비교가 가능합니다.")

    render_section_header(
        st,
        "DQN 샘플 10개 일괄 검증",
        "원본 10개와 균형형 10개를 별도로 진단·학습합니다. 순차 학습은 시간이 걸릴 수 있습니다.",
    )
    batch_actions = st.columns(3, gap="small")
    if batch_actions[0].button("원본 10개 데이터 진단", width="stretch"):
        st.session_state["dqn_sample_diagnosis"] = diagnose_dqn_training_sets(
            build_dqn_training_sets(mode="original")
        )
        st.session_state["dqn_notice"] = "완료"
        st.rerun()
    batch_actions[0].caption("원본 샘플 10개의 라벨 분포를 진단합니다.")

    if batch_actions[1].button("균형형 10개 데이터 생성", width="stretch"):
        generated = []
        failed = 0
        try:
            balanced_sets = build_dqn_training_sets(mode="balanced")
            for item in balanced_sets:
                try:
                    generated.append(str(save_balanced_recommendations(
                        item.get("recommendations") or [],
                        str(item.get("sample_id") or "sample"),
                        int(item.get("store_count") or 0),
                        int(item.get("dc_count") or 0),
                        derived_from=str(item.get("filename") or "DQN sample"),
                    )))
                except Exception:
                    failed += 1
            st.session_state["dqn_balanced_files"] = generated
            st.session_state["dqn_notice"] = "생성 완료" if not failed else "일부 저장 실패"
        except Exception:
            st.session_state["dqn_notice"] = "생성 실패"
        st.rerun()
    batch_actions[1].caption("원본은 유지하고 균형형 파생 데이터를 만듭니다.")

    if batch_actions[2].button("DQN 원본 10개 순차 학습", disabled=not torch_ok, width="stretch"):
        progress = st.progress(0.0, text="원본 학습 준비")
        callback = lambda index, total, label, result: progress.progress(
            index / max(1, total), text=f"{label} · {result.get('status', '-')}"
        )
        try:
            batch = train_dqn_batch(
                build_dqn_training_sets(mode="original"), episodes=90, progress_callback=callback
            )
            st.session_state["dqn_original_batch_result"] = batch
            st.session_state["dqn_batch_result"] = batch
            st.session_state["dqn_notice"] = "일괄 학습 완료"
        except Exception:
            st.session_state["dqn_notice"] = "일괄 학습 실패"
        st.rerun()
    batch_actions[2].caption("원본 샘플 10개를 차례로 학습합니다.")

    batch_actions_2 = st.columns(2, gap="small")
    if batch_actions_2[0].button("DQN 균형형 10개 순차 학습", disabled=not torch_ok, width="stretch"):
        progress = st.progress(0.0, text="균형형 학습 준비")
        callback = lambda index, total, label, result: progress.progress(
            index / max(1, total), text=f"{label} · {result.get('status', '-')}"
        )
        try:
            training_sets = build_dqn_training_sets(mode="balanced")
            batch = train_dqn_batch(training_sets, episodes=90, progress_callback=callback)
            st.session_state["dqn_balanced_batch_result"] = batch
            st.session_state["dqn_batch_result"] = batch
            st.session_state["dqn_notice"] = "일괄 학습 완료"
        except Exception:
            st.session_state["dqn_notice"] = "일괄 학습 실패"
        st.rerun()
    batch_actions_2[0].caption("균형형 샘플 10개를 차례로 학습합니다.")

    if batch_actions_2[1].button("원본 vs 균형형 비교 리포트", width="stretch"):
        try:
            report = build_dqn_batch_comparison_report(
                st.session_state.get("dqn_original_batch_result"),
                st.session_state.get("dqn_balanced_batch_result"),
            )
            st.session_state["dqn_batch_comparison_result"] = report
            st.session_state["dqn_notice"] = "비교 완료" if report.get("rows") else "비교 결과 없음"
        except Exception:
            st.session_state["dqn_notice"] = "비교 실패"
        st.rerun()
    batch_actions_2[1].caption("두 배치 결과를 하나의 비교표로 정리합니다.")

    notice = st.session_state.pop("dqn_notice", None)
    if notice and ("완료" in notice):
        st.success(notice)
    elif notice == "최근 결과 없음":
        st.info("현재 데이터와 선택 유형에 맞는 최근 학습 결과가 없습니다.")
    elif notice:
        st.error("요청을 완료하지 못했습니다. 기존 추천 결과는 유지됩니다.")
    if st.session_state.pop("dqn_storage_notice", False):
        st.warning("파일 저장은 완료되지 않았지만 학습 결과는 현재 세션에서 확인할 수 있습니다.")

    active = st.session_state.get("dqn_training_result")
    display = active
    if display:
        display = dict(display)
        if data_signature and display.get("data_signature") != data_signature:
            display["status"] = "과거 결과"
    summary = dqn_result_summary(display, recommendations)
    render_section_header(st, "학습 결과", "")
    result_mode = "균형형" if summary["variant"] == "balanced" else "원본"
    losses = (display or {}).get("loss_history") or []
    loss_summary = summary.get("loss_summary") or {}
    predictions = summary.get("prediction_distribution") or summary.get("action_distribution") or {}
    prediction_total = sum(int(value or 0) for value in predictions.values())
    dominance = max((int(value or 0) for value in predictions.values()), default=0) / max(1, prediction_total)
    applicable = bool(display and can_apply_dqn_to_current_data(display, data_signature))
    result_cols = st.columns(3, gap="small")
    result_cols[0].metric("학습 상태", user_status_label(summary["status"]))
    result_cols[1].metric("사용 데이터", result_mode if display else "-")
    result_cols[2].metric("에피소드", summary["episodes"] if display else "-")
    result_cols_2 = st.columns(3, gap="small")
    result_cols_2[0].metric("loss 시작값", _format_loss(losses[0] if losses else loss_summary.get("first")))
    result_cols_2[1].metric("loss 종료값", _format_loss(losses[-1] if losses else loss_summary.get("last")))
    result_cols_2[2].metric("예측 action 종류 수", len([value for value in predictions.values() if int(value or 0) > 0]))
    result_cols_3 = st.columns(3, gap="small")
    result_cols_3[0].metric("action 최대 쏠림 비율", f"{dominance * 100:.1f}%" if display else "-")
    result_cols_3[1].metric("안정성 상태", user_status_label(summary["status"]) if display else "-")
    result_cols_3[2].metric("VHS 참고 반영 여부", "반영" if applicable else "반영 안 함")
    if applicable:
        st.success("DQN 참고 점수가 최종 VHS에 낮은 비중으로 반영되었습니다.")
    elif display:
        st.info("현재 결과는 비교표에만 표시되며 최종 추천에는 반영하지 않습니다.")
    else:
        st.info("DQN은 학습 후 비교할 수 있습니다.")

    comparison = st.session_state.get("dqn_comparison_result") or {}
    if comparison.get("rows"):
        st.dataframe(_display_frame(comparison["rows"]), hide_index=True, width="stretch")
    batch = st.session_state.get("dqn_batch_result") or {}
    if batch.get("rows"):
        st.dataframe(_display_frame(batch["rows"]), hide_index=True, width="stretch")
    diagnosis = st.session_state.get("dqn_sample_diagnosis") or []
    if diagnosis:
        st.dataframe(_display_frame(diagnosis), hide_index=True, width="stretch")
    batch_comparison = st.session_state.get("dqn_batch_comparison_result") or {}
    if batch_comparison.get("rows"):
        st.dataframe(_display_frame(batch_comparison["rows"]), hide_index=True, width="stretch")
    st.caption("학습 결과는 현재 세션에서 바로 확인할 수 있으며 로컬 저장 실패가 앱 실행을 중단하지 않습니다.")

def _render_confidence(pipeline: dict) -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    analysis = pipeline.get("confidence_analysis") or {}
    rows = [
        {
            "route_id": item.get("route_id"), "상품": item.get("product_name"),
            "신뢰도": item.get("confidence_score"), "추천 등급": item.get("recommendation_grade"),
            "DQN": "미연결",
        }
        for item in recommendations
    ]
    if not rows:
        render_empty_state(st, "신뢰도 결과가 없습니다", compact=True)
        return
    values = [float(row["신뢰도"]) for row in rows if row["신뢰도"] is not None]
    cols = st.columns(3, gap="small")
    cols[0].metric("평균 추천 신뢰도", analysis.get("average", f"{sum(values) / len(values):.1f}" if values else "-"))
    score_range = analysis.get("score_range") or [None, None]
    cols[1].metric("점수 범위", f"{score_range[0]}~{score_range[1]}" if score_range[0] is not None else "-")
    cols[2].metric("추가 가점", "0점")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("등급: 높음 80 이상 / 보통 60 이상 / 낮음 60 미만")
    with st.expander("계산 기준", expanded=False):
        st.caption(f"계산 함수: {analysis.get('calculation_function', 'vhs_confidence.add_confidence_columns')}")
        st.caption("외부 모델 가점 없이 입력 데이터 기준으로 계산합니다.")


def _render_optimality(pipeline: dict) -> None:
    report = pipeline.get("validation_report") or {}
    gap = report.get("optimality_gap") or {}
    if not gap or gap.get("status") in ("비교 불가", "입력 컬럼 부족"):
        render_empty_state(st, "Optimality Gap 결과가 없습니다", compact=True)
        if gap:
            st.caption(gap.get("status"))
        return
    cols = st.columns(4, gap="small")
    cols[0].metric("Optimality Gap", gap.get("gap_str", "-"))
    cols[1].metric("후보 일치율", f"{float(gap.get('match_rate') or 0):.1f}%")
    cols[2].metric("Varo 비용", f"{float(gap.get('varo_total', 0)):,.0f}원")
    cols[3].metric("최적 비용", f"{float(gap.get('opt_total', 0)):,.0f}원")
    st.caption(
        f"상태: {gap.get('status', '-')} · 검증 방식: {gap.get('opt_method', '-')} · "
        f"비교 가능 후보: {gap.get('comparable_candidate_count', gap.get('candidates_used', 0))} · "
        f"공식: {gap.get('formula', '-')}"
    )


def _render_report_summary(pipeline: dict) -> None:
    summary = pipeline.get("summary") or {}
    vhs = pipeline.get("vhs_analysis") or {}
    greedy = pipeline.get("greedy_analysis") or {}
    optimality = (pipeline.get("validation_report") or {}).get("optimality_gap") or {}
    confidence = pipeline.get("confidence_analysis") or {}
    recommendations = st.session_state.get("varo_recommendations") or []
    uploaded_avg = _as_number(vhs.get("uploaded_average"))
    recalc_avg = _as_number(vhs.get("recalculated_average", summary.get("average_vhs_score")))
    difference = (
        round(recalc_avg - uploaded_avg, 2)
        if uploaded_avg is not None and recalc_avg is not None else "-"
    )
    status_label = {"success": "성공 (핵심 알고리즘 재계산)", "partial": "일부 보류"}.get(
        pipeline.get("status"), pipeline.get("status", "-")
    )
    dqn_summary = dqn_result_summary(st.session_state.get("dqn_training_result"), recommendations)
    rows = [
        ("적용 파일", st.session_state.get("uploaded_filename") or "-"),
        ("데이터 기준", pipeline.get("result_basis", "-")),
        ("분석 상태", status_label),
        ("추천 결과 수", len(recommendations)),
        ("처리 대상 재고", f"{float(summary.get('total_recommended_qty') or 0):,.0f}"),
        ("예상 절감액", f"{float(summary.get('total_expected_saving') or 0):,.0f}원"),
        ("업로드 VHS 평균", uploaded_avg if uploaded_avg is not None else "-"),
        ("재계산 VHS 평균", recalc_avg if recalc_avg is not None else "-"),
        ("VHS 차이 (재계산 − 업로드)", difference),
        ("Greedy 1순위", greedy.get("selected_route_id") or "-"),
        ("Greedy 전략 일치율", f"{float(greedy.get('strategy_match_rate') or 0):.1f}%"),
        ("Optimality Gap", optimality.get("gap_str", "-")),
        ("신뢰도 평균", confidence.get("average", "-")),
        ("연결 알고리즘 수", len(pipeline.get("connected_algorithms") or [])),
        ("보류 알고리즘 수 (원본)", len(pipeline.get("deferred_algorithms") or [])),
        ("V2 요약 기능 수 (연결)", len(pipeline.get("v2_summary_functions") or [])),
        ("VHS 중립값 적용 구성요소", (pipeline.get("vhs_neutral_analysis") or {}).get("neutral_components", "-")),
        ("DQN", f"{dqn_summary['status']} · {dqn_summary['reflection_mode']}"),
        ("다운로드", "추천 CSV · 추천 Excel · 분석결과 Excel · 검증리포트 Excel (4종)"),
        ("자동 테스트", "회귀·계약 테스트로 검증 (py -m unittest discover -s tests)"),
    ]
    st.dataframe(
        pd.DataFrame([{"항목": key, "값": "-" if value is None else str(value)} for key, value in rows]),
        hide_index=True,
        width="stretch",
    )


def _render_validation_report(pipeline: dict) -> None:
    render_section_header(st, "분석 요약", "현재 Varo V2 상태를 한 번에 확인합니다.")
    _render_report_summary(pipeline)

    report = st.session_state.get("varo_validation")
    if report:
        render_section_header(st, "데이터 검증 메시지", "V2 자체 데이터 검증 결과입니다.")
        st.markdown(badge_html(report.status, _badge_variant(report.status)), unsafe_allow_html=True)
        if report.messages:
            st.dataframe(pd.DataFrame([message.to_dict() for message in report.messages]), hide_index=True, width="stretch")

    connected = pipeline.get("connected_algorithms") or []
    deferred = pipeline.get("deferred_algorithms") or []
    warnings = pipeline.get("warnings") or []
    optimality = (pipeline.get("validation_report") or {}).get("optimality_gap") or {}
    render_section_header(st, "알고리즘 연결 상태", "현재 사용할 수 있는 분석과 보류 항목입니다.")
    cols = st.columns(4, gap="small")
    cols[0].metric("연결 함수", len(connected))
    cols[1].metric("보류 항목", len(deferred))
    cols[2].metric("추천 결과", len(st.session_state.get("varo_recommendations") or []))
    cols[3].metric("비교 가능 후보", optimality.get("comparable_candidate_count", 0))
    if connected:
        with st.expander("연결된 알고리즘", expanded=False):
            st.dataframe(pd.DataFrame({"함수": connected}), hide_index=True, width="stretch")
    if deferred:
        st.caption("보류 항목은 추가 입력 기준이 필요한 보조 기능이며 핵심 추천 결과에는 영향을 주지 않습니다.")
        with st.expander("보류된 알고리즘", expanded=False):
            st.dataframe(pd.DataFrame(deferred), hide_index=True, width="stretch")
    if warnings:
        st.warning("\n".join(f"- {warning}" for warning in warnings))

    v2_functions = pipeline.get("v2_summary_functions") or []
    render_section_header(st, "보류 분석과 V2 요약", "현재 제공 범위와 추가 입력이 필요한 항목입니다.")
    st.markdown(
        "- 상세 민감도 분석: **보류** → 현재 데이터 기준의 **V2 민감도 요약** 제공\n"
        "- 추천 사유 확장 분석: **보류** → 현재 결과 기준의 **V2 추천 사유 요약** 제공"
    )
    if v2_functions:
        with st.expander("내부 연결 정보", expanded=False):
            st.caption("연결된 V2 요약 기능: " + " · ".join(v2_functions))
    reasons = (pipeline.get("reason_analysis") or {}).get("reasons") or {}
    if reasons:
        reason_rows = [
            {
                "route_id": route_id,
                "추천 사유": " ".join(detail.get("sentences") or []),
                "주의사항": detail.get("caution", ""),
            }
            for route_id, detail in reasons.items()
        ]
        with st.expander("V2 추천 사유 요약", expanded=False):
            st.dataframe(pd.DataFrame(reason_rows), hide_index=True, width="stretch")
    exclusion = pipeline.get("excluded_dqn_artifacts") or {}
    st.info(exclusion.get("reason", "기존 DQN 학습 결과는 V2에 반영하지 않습니다."))
    with st.expander("DQN 제외 패턴", expanded=False):
        st.code("\n".join(exclusion.get("blocked_patterns") or []), language=None)

    sources = (pipeline.get("validation_report") or {}).get("calculation_sources") or {}
    if sources:
        with st.expander("계산 출처", expanded=False):
            st.dataframe(
                pd.DataFrame([{"항목": key, "계산 출처": value} for key, value in sources.items()]),
                hide_index=True,
                width="stretch",
            )

    recommendations = st.session_state.get("varo_recommendations") or []
    promotion_rows = [
        {
            "route_id": item.get("route_id"),
            "상품": item.get("product_name"),
            "재배치 예상 절감액": item.get("expected_saving"),
            "프로모션 예상 효과": item.get("promotion_effect"),
            "프로모션 권장 여부": item.get("promotion_recommended") or "프로모션 비교 보류",
            "최종 선택 이유": item.get("promotion_reason") or "입력 컬럼 부족",
        }
        for item in recommendations
    ]
    with st.expander("프로모션 비교", expanded=False):
        if promotion_rows:
            st.dataframe(pd.DataFrame(promotion_rows), hide_index=True, width="stretch")
        else:
            render_empty_state(st, "프로모션 비교 보류", "입력 컬럼 부족", compact=True)

    inventory_rows = []
    for group_name, group in (("수요·안전재고", pipeline.get("demand_analysis") or {}), ("폐기·재고", pipeline.get("risk_analysis") or {})):
        for key, value in group.items():
            if not isinstance(value, dict) or key == "store_clustering":
                continue
            inventory_rows.append({
                "영역": group_name,
                "분석": key,
                "상태": value.get("status", "연결" if value else "보류"),
                "함수": value.get("function", "-"),
                "계산 지표": ", ".join(value.get("output_columns") or []) or "-",
                "입력 부족": ", ".join(value.get("missing_input_columns") or []) or "없음",
            })
    with st.expander("수요·폐기·재고 분석 연결 상태", expanded=False):
        if inventory_rows:
            st.dataframe(pd.DataFrame(inventory_rows), hide_index=True, width="stretch")
        else:
            render_empty_state(st, "연결 상태를 확인할 수 없습니다", compact=True)

    upload_report = st.session_state.get("upload_report") or {}
    st.caption(f"최종 상태: {pipeline.get('status', '미연결')} · 결과 기준: {pipeline.get('result_basis', '-')}")

    render_section_header(st, "리포트 다운로드", "검증 결과와 분석 결과를 파일로 내려받습니다.")
    cols = st.columns([1, 1, 2], gap="small")
    cols[0].download_button(
        "검증 리포트 Excel",
        data=export_service.validation_report_excel_bytes(report, pipeline, recommendations, upload_report),
        file_name="varo_v2_검증리포트.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_validation_report_tab_xlsx",
    )
    cols[1].download_button(
        "분석 결과 전체 Excel",
        data=export_service.analysis_result_excel_bytes(pipeline, recommendations, upload_report),
        file_name="varo_v2_분석결과.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_analysis_tab_xlsx",
    )
    cols[2].caption("검증 메시지·알고리즘 연결·Optimality Gap·DQN 제외·업로드 품질·컬럼 매핑이 포함됩니다.")


def _render_recommendation_validation(pipeline: dict) -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    summary = pipeline.get("summary") or {}
    report = st.session_state.get("varo_validation")
    cols = st.columns(4, gap="small")
    cols[0].metric("검증 결과", getattr(report, "status", "데이터 없음") if report else "데이터 없음")
    cols[1].metric("추천 결과", len(recommendations))
    cols[2].metric("예상 절감액", f"{float(summary.get('total_expected_saving') or 0):,.0f}원")
    average_vhs = summary.get("average_vhs_score")
    average_vhs_display = f"{float(average_vhs):.1f}" if average_vhs is not None else "-"
    cols[3].metric("평균 VHS", average_vhs_display)


def _render_core_analysis(pipeline: dict) -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    weight_analysis = pipeline.get("vhs_weight_analysis") or pipeline.get("vhs_analysis") or {}
    weights = weight_analysis.get("weights") or {}
    confidence = pipeline.get("confidence_analysis") or {}
    sensitivity_rows = (pipeline.get("sensitivity_analysis") or {}).get("rows") or []

    st.caption("VHS는 최종 우선순위의 기준이며 Greedy, DQN, Pareto 결과는 비교·보조 검증에 사용합니다.")
    render_section_header(st, "VHS 자동 가중치", "현재 데이터 분포를 기준으로 계산합니다.")
    active_weights = [(key, float(value or 0)) for key, value in weights.items() if float(value or 0) > 0]
    cols = st.columns(3, gap="small")
    cols[0].metric("평균 VHS", f"{float(weight_analysis.get('vhs_average') or 0):.1f}")
    cols[1].metric("적용 평가 요소", len(active_weights))
    cols[2].metric("DQN 참고 가중치", f"{float(weights.get('dqn_reference_score') or 0) * 100:.1f}%")
    if active_weights:
        rows = [
            {"평가 요소": _WEIGHT_LABELS.get(key, key), "자동 가중치": f"{value * 100:.1f}%"}
            for key, value in sorted(active_weights, key=lambda item: item[1], reverse=True)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=245)

    render_section_header(st, "민감도 · 추천 신뢰도", "순위 변동 가능성과 추천 안정성을 함께 확인합니다.")
    counts = {"높음": 0, "보통": 0, "낮음": 0, "제한적": 0}
    for row in sensitivity_rows:
        level = str(row.get("overall_sensitivity") or "제한적")
        counts[level] = counts.get(level, 0) + 1
    score_range = confidence.get("score_range") or [None, None]
    cols = st.columns(4, gap="small")
    cols[0].metric("민감도 높음", counts.get("높음", 0))
    cols[1].metric("민감도 보통", counts.get("보통", 0))
    cols[2].metric("민감도 낮음", counts.get("낮음", 0))
    cols[3].metric("평균 신뢰도", confidence.get("average", "-"))
    if sensitivity_rows:
        compact_rows = [
            {
                "상품": row.get("product_name") or "-",
                "종합 민감도": row.get("overall_sensitivity") or "제한적",
                "추천 신뢰도": next(
                    (item.get("confidence_score") for item in recommendations if item.get("product_name") == row.get("product_name")),
                    "-",
                ),
            }
            for row in sensitivity_rows
        ]
        st.dataframe(pd.DataFrame(compact_rows), hide_index=True, width="stretch", height=245)


def _render_comparison_results(pipeline: dict) -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    render_section_header(
        st,
        "Pareto 보조 검증",
        "VHS 최종 순위와 Greedy·DQN 참고값을 함께 보며 후보 간 Pareto 우위를 확인합니다.",
    )
    rows = []
    for item in build_strategy_comparison(recommendations):
        rows.append({
            "상품": item.get("상품명"),
            "VHS 점수": item.get("VHS 점수"),
            "VHS 순위": item.get("VHS 순위"),
            "Greedy 순위": item.get("Greedy 순위"),
            "DQN 상태": user_status_label(item.get("DQN 상태")),
            "DQN 반영": user_status_label(item.get("DQN 반영 여부")),
            "Pareto 순위": item.get("Pareto rank"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        render_empty_state(st, "비교 결과가 없습니다", compact=True)

    render_section_header(st, "제한 탐색 기반 검증", "추천 비용과 제한 조건 탐색 결과를 비교합니다.")
    optimality = (pipeline.get("validation_report") or {}).get("optimality_gap") or {}
    if not optimality:
        render_empty_state(st, "비교 가능한 최적성 결과가 없습니다", compact=True)
        return
    cols = st.columns(4, gap="small")
    cols[0].metric("최적성 Gap", optimality.get("gap_str", "-"))
    cols[1].metric("후보 일치율", f"{float(optimality.get('match_rate') or 0):.1f}%")
    cols[2].metric("추천 비용", f"{float(optimality.get('varo_total') or 0):,.0f}원")
    cols[3].metric("비교 비용", f"{float(optimality.get('opt_total') or 0):,.0f}원")


def _render_sensitivity_confidence(pipeline: dict) -> None:
    render_section_header(st, "민감도 분석", "비용·거리·수량·VHS 변화에 따른 순위 안정성을 확인합니다.")
    _render_sensitivity(pipeline)
    render_section_header(st, "추천 신뢰도", "현재 데이터 품질을 기준으로 추천 신뢰도를 확인합니다.")
    _render_confidence(pipeline)


def render_validation_page() -> None:
    status = _validation_status()
    has_data = bool(st.session_state.get("varo_data")) and bool(st.session_state.get("varo_recommendations"))
    render_page_header(
        st, "분석 및 검증",
        "데이터 품질 진단 및 학습 안정성 비교를 위해 VHS, Greedy, DQN 학습·비교, Pareto, 민감도·신뢰도를 확인합니다.",
        badge=badge_html(status, _badge_variant(status)),
    )
    if not has_data:
        render_empty_state(st, "데이터가 없습니다", compact=True)
    pipeline = _ensure_detailed_pipeline() if has_data else _pipeline()

    tabs = st.tabs(TABS)
    renderers = (
        lambda: _render_core_analysis(pipeline),
        lambda: _render_greedy(pipeline),
        _render_dqn,
        lambda: _render_comparison_results(pipeline),
        lambda: _render_sensitivity_confidence(pipeline),
    )
    for tab, title, renderer in zip(tabs, TABS, renderers):
        with tab:
            if has_data or title == "DQN 학습·비교":
                renderer()
            else:
                render_empty_state(st, "데이터가 없습니다", compact=True)
