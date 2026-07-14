"""Analysis and validation page for Varo V2."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import re
import copy

from components.cards import render_empty_state, render_page_header, render_section_header
from components.status import badge_html, user_status_label
from services import export_service, v2_summaries
from services.app_state import current_result_basis
from services.dqn_service import (
    apply_dqn_reference_to_recommendations,
    build_dqn_batch_comparison_report,
    can_apply_dqn_to_current_data,
    compare_dqn_training_sets,
    dqn_result_summary,
    get_torch_status,
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
    st.info(
        "상세 민감도 계산은 추가 입력 기준이 필요해 보류되었습니다. "
        "현재는 비용·거리·수량·VHS·절감액 기준의 V2 민감도 요약을 제공합니다."
    )
    recommendations = st.session_state.get("varo_recommendations") or []
    analysis = pipeline.get("sensitivity_analysis") or {}
    rows = analysis.get("rows") or v2_summaries.sensitivity_summary(recommendations)
    if not rows:
        render_empty_state(st, "민감도 요약을 생성할 데이터가 없습니다", compact=True)
        return
    counts = {"높음": 0, "보통": 0, "낮음": 0, "제한적": 0}
    for row in rows:
        counts[row.get("overall_sensitivity", "제한적")] = counts.get(row.get("overall_sensitivity", "제한적"), 0) + 1
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


def _render_dqn() -> None:
    recommendations = st.session_state.get("varo_recommendations") or []
    data_signature = st.session_state.get("data_signature")
    torch_ok, torch_message = get_torch_status()
    torch_label = torch_message if torch_ok else "DQN 학습 실행 환경 필요"
    st.markdown(badge_html(torch_label, "success" if torch_ok else "warning"), unsafe_allow_html=True)
    st.caption("원본 라벨 편향을 진단하고 균형형 학습 데이터로 비교 검증합니다.")
    sample_id, store_count, dc_count = _dqn_context()
    current_diagnosis = diagnose_dqn_training_sets([{
        "sample_id": sample_id,
        "sample_name": st.session_state.get("uploaded_filename") or "현재 데이터",
        "mode": "original",
        "recommendations": recommendations,
    }])[0] if recommendations else {}
    render_section_header(st, "현재 샘플 진단", "학습을 시작하기 전에 후보 수와 라벨 편향을 확인합니다.")
    if st.button(
        "현재 샘플 진단",
        key="dqn_current_sample_diagnosis",
        type="primary",
        disabled=not recommendations,
        width="stretch",
    ):
        st.success("현재 샘플 진단을 갱신했습니다.")
    diagnostic_cols = st.columns(3, gap="small")
    diagnostic_cols[0].metric("후보 수", current_diagnosis.get("candidate_count", 0))
    diagnostic_cols[1].metric("라벨 종류", current_diagnosis.get("target_type_count", 0))
    diagnostic_cols[2].metric("상태", user_status_label(current_diagnosis.get("status", "학습 필요")))

    render_section_header(st, "선택 샘플 학습", "버튼을 누른 경우에만 선택된 데이터의 학습 또는 비교를 실행합니다.")
    actions = st.columns(3, gap="small")
    if actions[0].button("선택 샘플 원본 학습", type="primary", disabled=not recommendations or not torch_ok, width="stretch"):
        result = train_dqn(
            prepare_dqn_recommendations(recommendations, "original"),
            data_signature=data_signature,
            episodes=180,
            learning_rate=0.001,
            reflection_mode="DQN 참고만",
            sample_id=sample_id,
            training_mode="original",
            store_count=store_count,
            dc_count=dc_count,
            seed=17,
        )
        st.session_state["dqn_training_result"] = result.to_dict()
        _refresh_recommendations_with_dqn(st.session_state["dqn_training_result"])
        st.session_state["dqn_notice"] = "완료"
        st.rerun()
    actions[0].caption("현재 추천 라벨을 그대로 사용해 학습합니다.")

    if actions[1].button("선택 샘플 균형형 학습", disabled=not recommendations or not torch_ok, width="stretch"):
        balanced = balanced_recommendations(recommendations)
        save_balanced_recommendations(
            balanced, sample_id, store_count, dc_count,
            derived_from=st.session_state.get("uploaded_filename"),
        )
        result = train_dqn(
            balanced,
            data_signature=data_signature,
            episodes=180,
            learning_rate=0.001,
            reflection_mode="DQN 참고만",
            sample_id=sample_id,
            training_mode="balanced",
            store_count=store_count,
            dc_count=dc_count,
            seed=17,
        )
        st.session_state["dqn_training_result"] = result.to_dict()
        _refresh_recommendations_with_dqn(st.session_state["dqn_training_result"])
        st.session_state["dqn_notice"] = "완료"
        st.rerun()
    actions[1].caption("라벨 균형을 보정한 파생 데이터로 학습합니다.")

    if actions[2].button("선택 샘플 원본 vs 균형형 비교", disabled=not recommendations or not torch_ok, width="stretch"):
        balanced = balanced_recommendations(recommendations)
        save_balanced_recommendations(balanced, sample_id, store_count, dc_count)
        comparison = compare_dqn_training_sets(
            recommendations, balanced, str(data_signature), episodes=180,
            sample_id=sample_id, store_count=store_count, dc_count=dc_count,
        )
        st.session_state["dqn_comparison_result"] = comparison
        preferred = comparison.get("preferred")
        selected = comparison.get("balanced_result") if preferred == "균형형" else comparison.get("original_result")
        if not selected:
            selected = comparison.get("balanced_result")
        st.session_state["dqn_training_result"] = selected
        _refresh_recommendations_with_dqn(selected)
        st.session_state["dqn_notice"] = "완료"
        st.rerun()
    actions[2].caption("두 학습 결과의 안정성을 같은 기준으로 비교합니다.")

    render_section_header(st, "DQN 샘플 10개", "원본 10개와 균형형 10개를 순차 학습하고 결과를 비교합니다.")
    batch_actions = st.columns(3, gap="small")
    if batch_actions[0].button("10개 원본 진단", width="stretch"):
        st.session_state["dqn_sample_diagnosis"] = diagnose_dqn_training_sets(
            build_dqn_training_sets(mode="original")
        )
        st.session_state["dqn_notice"] = "완료"
        st.rerun()
    batch_actions[0].caption("원본 샘플 10개의 라벨 분포를 진단합니다.")

    if batch_actions[1].button("10개 균형형 데이터 생성", width="stretch"):
        balanced_sets = build_dqn_training_sets(mode="balanced")
        generated = []
        for item in balanced_sets:
            generated.append(str(save_balanced_recommendations(
                item.get("recommendations") or [],
                str(item.get("sample_id") or "sample"),
                int(item.get("store_count") or 0),
                int(item.get("dc_count") or 0),
                derived_from=str(item.get("filename") or "DQN sample"),
            )))
        st.session_state["dqn_balanced_files"] = generated
        st.session_state["dqn_notice"] = "완료"
        st.rerun()
    batch_actions[1].caption("원본은 유지하고 균형형 파생 데이터를 만듭니다.")

    if batch_actions[2].button("10개 원본 순차 학습", disabled=not torch_ok, width="stretch"):
        progress = st.progress(0.0, text="원본 학습 준비")
        callback = lambda index, total, label, result: progress.progress(
            index / max(1, total), text=f"{label} · {result.get('status', '-')}"
        )
        batch = train_dqn_batch(
            build_dqn_training_sets(mode="original"), episodes=90, progress_callback=callback
        )
        st.session_state["dqn_original_batch_result"] = batch
        st.session_state["dqn_batch_result"] = batch
        st.session_state["dqn_notice"] = "완료"
        st.rerun()
    batch_actions[2].caption("원본 샘플 10개를 차례로 학습합니다.")

    batch_actions_2 = st.columns(2, gap="small")
    if batch_actions_2[0].button("10개 균형형 순차 학습", disabled=not torch_ok, width="stretch"):
        training_sets = build_dqn_training_sets(mode="balanced")
        progress = st.progress(0.0, text="균형형 학습 준비")
        callback = lambda index, total, label, result: progress.progress(
            index / max(1, total), text=f"{label} · {result.get('status', '-')}"
        )
        batch = train_dqn_batch(training_sets, episodes=90, progress_callback=callback)
        st.session_state["dqn_balanced_batch_result"] = batch
        st.session_state["dqn_batch_result"] = batch
        st.session_state["dqn_notice"] = "완료"
        st.rerun()
    batch_actions_2[0].caption("균형형 샘플 10개를 차례로 학습합니다.")

    if batch_actions_2[1].button("원본 vs 균형형 비교 리포트", width="stretch"):
        report = build_dqn_batch_comparison_report(
            st.session_state.get("dqn_original_batch_result"),
            st.session_state.get("dqn_balanced_batch_result"),
        )
        st.session_state["dqn_batch_comparison_result"] = report
        st.session_state["dqn_notice"] = "완료" if report.get("rows") else "비교 결과 없음"
        st.rerun()
    batch_actions_2[1].caption("두 배치 결과를 하나의 비교표로 정리합니다.")

    notice = st.session_state.pop("dqn_notice", None)
    if notice == "완료":
        st.success("완료")
    elif notice:
        st.error("실패")

    active = st.session_state.get("dqn_training_result")
    active_mode = str((active or {}).get("variant") or (active or {}).get("training_mode") or "original")
    latest = load_latest_dqn_result(data_signature, active_mode)
    display = active or latest
    if display:
        display = dict(display)
        if data_signature and display.get("data_signature") != data_signature:
            display["status"] = "과거 결과"
    summary = dqn_result_summary(display, recommendations)
    render_section_header(st, "학습 결과", "")
    cols = st.columns(3, gap="small")
    cols[0].metric("상태", user_status_label(summary["status"]))
    cols[1].metric("후보 수", summary["candidate_count"])
    cols[2].metric("평균 confidence", summary["average_confidence"] if summary["average_confidence"] is not None else "-")
    if display and can_apply_dqn_to_current_data(display, data_signature):
        st.success("DQN 참고 점수가 낮은 비중으로 반영되었습니다.")
    elif display:
        st.info("DQN은 비교표에만 표시되며 최종 추천에는 반영하지 않습니다.")
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
    st.caption("학습 결과는 outputs/dqn, 균형형 파생 샘플은 outputs/dqn_balanced_samples에 저장됩니다.")

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
    pipeline = _pipeline()
    has_data = bool(st.session_state.get("varo_data")) and bool(st.session_state.get("varo_recommendations"))
    render_page_header(
        st, "분석 및 검증",
        "데이터 품질 진단 및 학습 안정성 비교를 위해 VHS, Greedy, DQN 학습·비교, Pareto, 민감도·신뢰도를 확인합니다.",
        badge=badge_html(status, _badge_variant(status)),
    )
    if not has_data:
        render_empty_state(st, "데이터가 없습니다", compact=True)

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
