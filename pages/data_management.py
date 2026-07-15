"""Data management page for Varo V2."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_error_card, render_page_header, render_section_header
from components.status import badge_html
from services import export_service, upload_quality
from services.analysis_pipeline import summarize_loaded_data
from services.app_state import current_result_basis, has_app_data
from services.data_application import load_and_apply
from services.data_loader import SAMPLE_FILENAME, get_default_sample_path
from services.data_validator import ValidationReport
from services.dqn_samples import dqn_sample_options, dqn_sample_path, dqn_sample_table_rows
from services.dqn_service import dqn_result_summary
from services.sample_catalog import sample_options, sample_path

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("string").fillna("")


def _validation_variant(status: str) -> str:
    return {"통과": "success", "주의": "warning"}.get(status, "error")


def _render_validation(report: ValidationReport | None) -> None:
    render_section_header(st, "데이터 검증", "V2 자체 검증 결과입니다.")
    if not report:
        render_empty_state(st, "검증할 데이터가 없습니다", compact=True)
        return
    st.markdown(badge_html(report.status, _validation_variant(report.status)), unsafe_allow_html=True)
    if report.messages:
        st.dataframe(_display_df(pd.DataFrame([message.to_dict() for message in report.messages])), hide_index=True, width="stretch")


def _render_sheet_summary(data: dict[str, pd.DataFrame] | None, report: ValidationReport | None) -> None:
    render_section_header(st, "현재 로드 상태", "")
    if not data:
        render_empty_state(st, "요약할 데이터가 없습니다", compact=True)
        return
    summary = summarize_loaded_data(data, report)
    cols = st.columns(4, gap="small")
    for idx, (label, value) in enumerate(summary.items()):
        cols[idx % 4].metric(label, value)


def _render_pipeline_status() -> None:
    pipeline = st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result") or {}
    if not pipeline:
        return
    render_section_header(st, "분석 적용 상태", current_result_basis(st.session_state))
    cols = st.columns(4, gap="small")
    cols[0].metric("연결 분석", len(pipeline.get("connected_algorithms") or []))
    cols[1].metric("보류 항목", len(pipeline.get("deferred_algorithms") or []))
    cols[2].metric("추천 결과", len(pipeline.get("recommendations") or []))
    cols[3].metric("분석 상태", "일부 연결" if pipeline.get("status") == "partial" else "적용 완료")
    deferred_count = len(pipeline.get("deferred_algorithms") or [])
    if deferred_count:
        st.caption(f"보류 {deferred_count}건은 추가 입력 기준이 필요한 보조 분석입니다.")


def _render_quality_check(data: dict[str, pd.DataFrame] | None) -> None:
    quality = data.get("quality_check") if data else None
    if quality is not None and not quality.empty:
        with st.expander("Quality_Check 시트 보기", expanded=False):
            st.caption("참고용 점검 시트이며 V2 자체 검증 결과와 별도로 표시합니다.")
            st.dataframe(_display_df(quality), hide_index=True, width="stretch")


def _render_raw_data_view(data: dict[str, pd.DataFrame] | None) -> None:
    if not data:
        return
    with st.expander("원본 데이터 보기", expanded=False):
        sheet_keys = [key for key, value in data.items() if isinstance(value, pd.DataFrame)]
        selected = st.selectbox("시트", sheet_keys, key="raw_sheet_select")
        st.dataframe(_display_df(data[selected]), hide_index=True, width="stretch")


def _render_downloads() -> None:
    render_section_header(st, "분석 결과 다운로드", "추천 결과와 연결된 분석 결과를 파일로 내려받습니다.")
    recommendations = st.session_state.get("varo_recommendations") or []
    if not recommendations:
        render_empty_state(
            st, "다운로드할 분석 결과가 없습니다",
            "데이터를 적용하면 추천·분석 결과 다운로드가 활성화됩니다.", compact=True,
        )
        return
    pipeline = st.session_state.get("analysis_result") or st.session_state.get("varo_pipeline_result") or {}
    validation = st.session_state.get("varo_validation")
    upload_report = st.session_state.get("upload_report") or {}
    cols = st.columns(4, gap="small")
    cols[0].download_button(
        "추천 결과 CSV",
        data=export_service.recommendations_csv_bytes(recommendations),
        file_name="varo_v2_추천결과.csv",
        mime="text/csv",
        width="stretch",
        key="dl_rec_csv",
    )
    cols[1].download_button(
        "추천 결과 Excel",
        data=export_service.recommendations_excel_bytes(recommendations),
        file_name="varo_v2_추천결과.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_rec_xlsx",
    )
    cols[2].download_button(
        "분석 결과 전체 Excel",
        data=export_service.analysis_result_excel_bytes(pipeline, recommendations, upload_report),
        file_name="varo_v2_분석결과.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_analysis_xlsx",
    )
    cols[3].download_button(
        "검증 리포트 Excel",
        data=export_service.validation_report_excel_bytes(validation, pipeline, recommendations, upload_report),
        file_name="varo_v2_검증리포트.xlsx",
        mime=XLSX_MIME,
        width="stretch",
        key="dl_validation_report_xlsx",
    )
    st.caption("CSV는 UTF-8(BOM) 인코딩이며 현재 적용된 추천·분석 결과를 내려받습니다.")


def _render_upload_quality() -> None:
    report = st.session_state.get("pending_upload_report") or st.session_state.get("upload_report") or {}
    if not report:
        return
    render_section_header(st, "업로드 품질 점검", "컬럼 자동 매핑·숫자 변환·추천 생성 방식을 확인합니다.")
    cols = st.columns(4, gap="small")
    cols[0].metric("자동 매핑 컬럼", report.get("mapped_column_count", 0))
    cols[1].metric("누락 필수 컬럼", report.get("missing_required_count", 0))
    cols[2].metric("숫자 변환 실패", report.get("numeric_failed_total", 0))
    cols[3].metric("빈 행 제거", report.get("blank_removed_total", 0))

    if report.get("date_columns") or report.get("date_success"):
        st.caption(
            f"유통기한 날짜 컬럼({', '.join(report.get('date_columns') or []) or '-'})을 남은 일수로 자동 환산했습니다. "
            f"성공 {report.get('date_success', 0)}건 / 해석 불가 {report.get('date_failed', 0)}건."
        )

    source = report.get("recommendation_source")
    label = report.get("recommendation_source_label", "-")
    if source == "generated":
        info = report.get("candidate_info") or {}
        st.info(
            f"추천 결과 시트가 없어 **{label}**로 동작합니다. "
            f"생성 후보 {info.get('count', 0)}건(DIRECT {info.get('direct_count', 0)}·VIA_DC {info.get('via_dc_count', 0)}). "
            + info.get("reason", "")
        )
    elif source == "none":
        st.warning(
            "파일은 읽혔지만 추천 결과를 만들기 위한 최소 정보가 부족합니다. "
            + ((report.get("candidate_info") or {}).get("reason", ""))
        )
    else:
        st.caption(f"추천 생성 방식: {label}")

    if report.get("mapped_column_count"):
        st.caption("컬럼명이 자동으로 표준화되었습니다. 상세 매핑은 검증 리포트·다운로드에서 확인할 수 있습니다.")
    if report.get("missing_required_count"):
        st.warning("필수 컬럼이 부족해 일부 분석이 제한될 수 있습니다. 아래 검증 메시지의 항목을 추가하면 정확도가 올라갑니다.")
    elif report.get("numeric_failed_total"):
        st.caption("일부 값이 숫자로 변환되지 않아 제외되었습니다(쉼표·단위는 자동 정리됩니다).")
    st.caption("선택 컬럼이 없으면 해당 분석 항목은 제한되거나 기준값으로 처리됩니다.")


def _render_sample_selector() -> None:
    render_section_header(st, "시뮬레이션 검수 샘플", "점포와 DC 구성이 다른 데이터로 홈 네트워크를 확인합니다.")
    options = sample_options()
    selected_label = st.selectbox(
        "샘플 선택",
        list(options),
        key="simulation_sample_select",
    )
    selected = options[selected_label]
    st.caption(f"점포 {selected.store_count}개 · DC {selected.dc_count}개")
    if st.button("선택 샘플 불러오기", key="load_simulation_sample", type="primary", width="stretch"):
        path = sample_path(selected)
        if not path.exists():
            st.session_state["pending_load_error"] = f"샘플 파일이 없습니다: {selected.filename}"
            st.rerun()
        if _load_with_progress(path, selected.filename, "샘플 추천 데이터"):
            st.session_state["current_menu"] = "홈"
        st.rerun()


def _load_with_progress(source, filename: str, source_type: str) -> bool:
    status = st.status("데이터 읽는 중", expanded=False)

    def update(label: str) -> None:
        status.update(
            label=label,
            state="complete" if label == "데이터 적용 완료" else "running",
            expanded=False,
        )

    applied = load_and_apply(
        st.session_state, source, filename, source_type, progress_callback=update,
    )
    if not applied:
        failed_stage = st.session_state.get("pending_load_stage") or "데이터 적용"
        status.update(label=f"{failed_stage} 실패", state="error", expanded=False)
    return applied

def render_data_management_page() -> None:
    render_page_header(st, "데이터 관리", "")
    render_section_header(st, "데이터 불러오기", "")
    base_col, dqn_col = st.columns([1, 2], gap="small")
    with base_col:
        st.caption("일반 샘플")
        if st.button("기본 샘플 불러오기", key="data_default_sample", type="primary", width="stretch"):
            path = get_default_sample_path()
            if _load_with_progress(path, SAMPLE_FILENAME, "샘플 추천 데이터"):
                st.session_state["dqn_sample_training_mode"] = "original"
                st.session_state["dqn_selected_sample"] = "기본 샘플"
            st.rerun()
    with dqn_col:
        st.caption("DQN 학습 샘플")
        options = dqn_sample_options()
        selected_label = st.selectbox("DQN 샘플 선택", list(options), key="dqn_sample_select")
        selected = options[selected_label]
        if st.button("선택한 DQN 샘플 불러오기", key="load_dqn_sample", width="stretch"):
            path = dqn_sample_path(selected)
            if _load_with_progress(path, selected.workbook.filename, "DQN 샘플"):
                st.session_state["dqn_sample_training_mode"] = selected.mode
                st.session_state["dqn_selected_sample"] = selected.label
            st.rerun()

    st.caption(
        "원본은 수정하지 않고, 학습·균형형 파생 산출물은 outputs 하위 로컬 폴더에만 저장되며 Git에는 포함되지 않습니다."
    )
    with st.expander("DQN 샘플 10개 목록", expanded=False):
        st.dataframe(pd.DataFrame(dqn_sample_table_rows()), hide_index=True, width="stretch")

    load_error = st.session_state.get("pending_load_error") or st.session_state.get("pending_validation_error")
    if load_error:
        render_error_card(st, "파일을 불러올 수 없습니다", load_error)

    pending_data = st.session_state.get("pending_varo_data")
    candidate_data = pending_data or st.session_state.get("varo_data")
    candidate_report = st.session_state.get("pending_varo_validation") or st.session_state.get("varo_validation")
    if not candidate_data:
        render_empty_state(st, "데이터가 없습니다", compact=True)
        return

    source_name = st.session_state.get("pending_uploaded_filename") or st.session_state.get("uploaded_filename") or "-"
    st.markdown(badge_html(source_name, "accent"), unsafe_allow_html=True)
    if st.session_state.get("data_apply_message"):
        st.success(st.session_state["data_apply_message"])
    elif candidate_report and candidate_report.has_errors:
        st.error(st.session_state.get("pending_validation_error") or "데이터를 적용할 수 없습니다.")
    _render_sheet_summary(candidate_data, candidate_report)
