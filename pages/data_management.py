"""Data management page for Varo V2.

Linear flow: 현재 상태 → 현재 사용 중 데이터 → 데이터 불러오기 → 검사 중인 데이터
(검사 결과·미리보기·적용/취소) → 적용 데이터 상세(품질·점검·미리보기·다운로드).

The upload is a two-phase intake: a file is *inspected* into pending state
(``prepare_pending_data``) and only replaces the applied data when the user clicks
``이 데이터 사용``/``문제 행을 제외하고 사용`` (``commit_pending_data``). Status
wording/next action come from the shared ``home_state`` via ``data_management_view``
so this page never re-guesses status; it only adds the data-management-only detail.
Nothing internal (signatures, paths, session keys, exception names) reaches the screen.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from components.cards import render_empty_state, render_error_card, render_page_header, render_section_header
from components.state_banner import render_state_summary_card
from components.status import badge_html
from services import export_service
from services.app_state import clear_applied_data
from services.data_application import cancel_pending_data, commit_pending_data, load_and_apply
from services.data_issues import collect_data_issues, detail_rows, display_rows, issues_to_csv_bytes
from services.data_management_view import build_data_management_view
from services.sample_catalog import discover_dqn_samples, sample_options, sample_path

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

PAGE_RECOMMENDATIONS = "추천 실행"
_PREVIEW_ROW_LIMIT = 200


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("string").fillna("")


def _safe(value) -> str:
    return html.escape(str(value)) if value is not None else "-"


# --------------------------------------------------------------------------- #
# 1) 현재 상태 (shared wording) + 다음 행동
# --------------------------------------------------------------------------- #
def _render_status_header(view: dict) -> None:
    home = view["home"]
    if view["show_next_action"]:
        render_state_summary_card(
            home.get("title"), home.get("short_message"),
            action_label="추천 실행", action_page=PAGE_RECOMMENDATIONS, key="data_next_action",
        )
    else:
        render_state_summary_card(home.get("title"), home.get("short_message"))


# --------------------------------------------------------------------------- #
# 검사 중 데이터 (pending intake) — 검사 결과 + 미리보기 + 명시적 적용/취소
# --------------------------------------------------------------------------- #
_PENDING_BADGE = {
    "사용 가능": "success", "확인 필요": "warning",
    "사용 불가": "error", "현재 데이터와 동일": "accent",
}


def _cancel_pending() -> None:
    cancel_pending_data(st.session_state)
    st.session_state["current_menu"] = "데이터 관리"


def _render_pending_preview() -> None:
    data = st.session_state.get("pending_varo_data")
    if not data:
        return
    with st.expander("검사 중 데이터 미리보기", expanded=False):
        sheet_keys = [key for key, value in data.items() if isinstance(value, pd.DataFrame)]
        if not sheet_keys:
            render_empty_state(st, "표시할 시트가 없습니다", compact=True)
            return
        selected = st.selectbox("시트", sheet_keys, key="pending_sheet_select")
        frame = data[selected]
        if len(frame) > _PREVIEW_ROW_LIMIT:
            st.caption(f"처음 {_PREVIEW_ROW_LIMIT}행만 미리보기로 표시합니다 (전체 {len(frame)}행).")
            frame = frame.head(_PREVIEW_ROW_LIMIT)
        st.dataframe(_display_df(frame), hide_index=True, width="stretch")


def _render_pending_check(view: dict) -> None:
    pending = view["pending"]
    render_section_header(st, "검사 중인 데이터", "")

    apply_error = st.session_state.pop("data_apply_error", None)
    if apply_error:
        st.error(apply_error)

    status = pending["status"]
    st.markdown(
        badge_html(f"{status} · {pending['filename']}", _PENDING_BADGE.get(status, "warning")),
        unsafe_allow_html=True,
    )

    # Honest counts: analysis-usable/excluded only when the file is actually usable;
    # otherwise show error/warning rows and let the messages explain why.
    if pending["apply_allowed"]:
        cols = st.columns(4, gap="small")
        cols[0].metric("전체 행", pending["total_rows"])
        cols[1].metric("분석 사용 행", pending["usable_rows"])
        cols[2].metric("제외 행", pending["excluded_rows"])
        cols[3].metric("경고 행", pending["warning_rows"])
    else:
        cols = st.columns(3, gap="small")
        cols[0].metric("전체 행", pending["total_rows"])
        cols[1].metric("오류 행", pending["error_rows"])
        cols[2].metric("경고 행", pending["warning_rows"])

    if pending["top_rows"]:
        st.caption("가장 먼저 확인할 문제")
        st.dataframe(pd.DataFrame(pending["top_rows"]), hide_index=True, width="stretch")
        if pending["issue_count"] > len(pending["top_rows"]):
            with st.expander(f"전체 문제 보기 ({pending['issue_count']}건)", expanded=False):
                st.dataframe(pd.DataFrame(detail_rows(pending["issues"])), hide_index=True, width="stretch")
        st.download_button(
            "문제 목록 CSV 내려받기",
            data=issues_to_csv_bytes(pending["issues"]),
            file_name="varo_v2_데이터점검.csv",
            mime="text/csv",
            key="dl_pending_issues_csv",
        )
    elif not pending["apply_allowed"] and pending["error_messages"]:
        st.caption("확인이 필요한 항목")
        for message in pending["error_messages"]:
            st.markdown(f"- {message}")

    _render_pending_preview()

    # Explicit apply / cancel — nothing is applied without a button click.
    apply_col, cancel_col = st.columns([2, 1], gap="small")
    if pending["same_as_current"]:
        apply_col.info("현재 사용 중인 데이터입니다.")
    elif pending["apply_label"]:
        if pending["excluded_rows"]:
            apply_col.caption(f"{pending['excluded_rows']}행을 제외하고 {pending['usable_rows']}행을 사용합니다.")
        if apply_col.button(pending["apply_label"], key="apply_pending", type="primary", width="stretch"):
            commit_pending_data(st.session_state)
            st.rerun()
    else:
        apply_col.caption("문제를 수정한 뒤 다시 업로드하세요.")
    cancel_col.button("검사 중인 데이터 취소", key="cancel_pending", on_click=_cancel_pending, width="stretch")

    if view["has_current"]:
        st.caption("적용하기 전까지 현재 사용 중인 데이터와 추천 결과는 그대로 유지됩니다.")


# --------------------------------------------------------------------------- #
# 3) 현재 사용 중 데이터 카드
# --------------------------------------------------------------------------- #
def _clear_current_data() -> None:
    clear_applied_data(st.session_state)
    st.session_state["current_menu"] = "데이터 관리"


def _render_current_data_card(view: dict) -> None:
    current = view["current"]
    render_section_header(st, "현재 사용 중 데이터", "")
    rows = [
        ("데이터 출처", current["source_label"]),
        ("파일 또는 샘플", current["filename"]),
        ("시트", current["sheet_label"]),
        ("분석 사용 행", f"{current['usable_rows']}행"),
        ("데이터 상태", current["data_status"]),
        ("추천 실행 상태", current["recommendation_status"]),
    ]
    grid = "".join(
        f'<div class="v2-info-item"><span class="v2-card-caption">{_safe(label)}</span>'
        f'<strong>{_safe(value)}</strong></div>'
        for label, value in rows
    )
    st.markdown(
        f'<div class="v2-wrap v2-card"><div class="v2-recommendation-info">{grid}</div></div>',
        unsafe_allow_html=True,
    )
    if current["excluded_rows"]:
        st.caption(f"데이터 문제로 {current['excluded_rows']}행을 분석에서 제외했습니다.")
    st.button(
        "현재 데이터 초기화",
        key="clear_applied_data",
        on_click=_clear_current_data,
        help="현재 앱에 적용된 데이터와 분석 결과만 초기화합니다. 원본 파일은 삭제하지 않습니다.",
    )


def _render_apply_message() -> None:
    """One-time success note after an apply (never repeated across reruns)."""
    message = st.session_state.pop("data_apply_message", None)
    if message:
        st.success(message)


# --------------------------------------------------------------------------- #
# 4) 데이터 불러오기 (샘플)
# --------------------------------------------------------------------------- #
def _render_sample_selector() -> None:
    render_section_header(st, "① 기본 시뮬레이션 샘플", "")
    options = sample_options()
    selected_label = st.selectbox("샘플 선택", list(options), key="simulation_sample_select")
    selected = options[selected_label]
    st.caption(f"점포 {selected.store_count}개 · DC {selected.dc_count}개")
    if st.button("선택한 샘플 적용", key="load_simulation_sample", type="primary", width="stretch"):
        path = sample_path(selected)
        if not path.exists():
            st.session_state["pending_load_error"] = f"샘플 파일이 없습니다: {selected.filename}"
            st.rerun()
        if load_and_apply(st.session_state, path, selected.filename, "샘플 추천 데이터"):
            st.session_state["current_menu"] = "운영 현황"
        st.rerun()


def _render_dqn_sample_selector() -> None:
    render_section_header(st, "② DQN 학습 샘플", "")
    samples = discover_dqn_samples()
    if not samples:
        render_empty_state(
            st, "DQN 학습 샘플을 찾지 못했습니다",
            "Varo_DQN_training_samples_10pack 폴더가 있는지 확인해주세요.", compact=True,
        )
        return
    options = {sample.label: sample for sample in samples}
    selected_label = st.selectbox("DQN 샘플 선택", list(options), key="dqn_sample_select")
    selected = options[selected_label]
    st.caption(
        f"점포 {selected.store_count} · DC {selected.dc_count} · 재고 {selected.inventory_count}행 · "
        f"추천 후보 {selected.recommendation_count} · 검증 {selected.validation_status}"
    )
    with st.expander("샘플 10개 목록 보기", expanded=False):
        st.dataframe(
            pd.DataFrame([
                {
                    "번호": sample.sample_id,
                    "파일명": sample.file_name,
                    "점포": sample.store_count,
                    "DC": sample.dc_count,
                    "카테고리": sample.category,
                    "검증": sample.validation_status,
                    "수정일": sample.modified_at,
                    "크기(KB)": round(sample.file_size / 1024, 1),
                }
                for sample in samples
            ]),
            hide_index=True, width="stretch",
        )
    if st.button("선택한 DQN 샘플 적용", key="load_dqn_sample", type="primary", width="stretch"):
        path = Path(selected.file_path)
        if not path.exists():
            st.session_state["pending_load_error"] = f"샘플 파일이 없습니다: {selected.file_name}"
            st.rerun()
        if load_and_apply(st.session_state, path, selected.file_name, "DQN 학습 샘플"):
            st.session_state["current_menu"] = "운영 현황"
        st.rerun()


# --------------------------------------------------------------------------- #
# 5) 적용 데이터 상세 (품질·점검·미리보기·다운로드)
# --------------------------------------------------------------------------- #
def _render_upload_quality() -> None:
    report = st.session_state.get("upload_report") or {}
    if not report:
        return
    render_section_header(st, "업로드 품질 점검", "")
    cols = st.columns(4, gap="small")
    cols[0].metric("자동 매핑 컬럼", report.get("mapped_column_count", 0))
    cols[1].metric("누락 필수 컬럼", report.get("missing_required_count", 0))
    cols[2].metric("숫자 변환 실패", report.get("numeric_failed_total", 0))
    cols[3].metric("빈 행 제거", report.get("blank_removed_total", 0))

    source = report.get("recommendation_source")
    if source == "generated":
        st.info("추천 결과 시트가 없어 후보를 자동 생성했습니다.")
    elif source == "none":
        st.warning("추천 결과를 만들기 위한 최소 정보가 부족합니다. 파일을 확인해 주세요.")
    if report.get("missing_required_count"):
        st.warning("필수 컬럼이 부족해 일부 분석이 제한됩니다.")


def _render_data_issues() -> None:
    """Compact row-level check on the *applied* data (counts + top 5 + fixable CSV)."""
    data = st.session_state.get("varo_data")
    if not data:
        return
    result = collect_data_issues(
        data, st.session_state.get("raw_data"), st.session_state.get("source_metadata")
    )
    summary = result["summary"]
    if summary["total_issues"] == 0:
        return
    render_section_header(st, "데이터 점검", "")
    total = summary.get("total_rows", 0)
    if total:
        st.caption(
            f"전체 {total}행 중 {summary['usable_rows']}행을 분석에 사용합니다. "
            f"{summary['excluded_rows']}행은 데이터 문제로 제외했습니다."
        )
    cols = st.columns(4, gap="small")
    cols[0].metric("오류 행", summary["error_rows"])
    cols[1].metric("경고 행", summary["warning_rows"])
    cols[2].metric("사용 가능한 행", summary["usable_rows"])
    cols[3].metric("제외될 행", summary["excluded_rows"])
    top_rows = display_rows(summary["top"])
    if top_rows:
        st.caption("가장 먼저 확인할 문제")
        st.dataframe(pd.DataFrame(top_rows), hide_index=True, width="stretch")
    issues = result["issues"]
    if len(issues) > len(top_rows):
        with st.expander(f"전체 문제 보기 ({len(issues)}건)", expanded=False):
            st.dataframe(pd.DataFrame(detail_rows(issues)), hide_index=True, width="stretch")
    st.download_button(
        "문제 목록 CSV 내려받기",
        data=issues_to_csv_bytes(issues),
        file_name="varo_v2_데이터점검.csv",
        mime="text/csv",
        key="dl_data_issues_csv",
    )


def _render_preview() -> None:
    data = st.session_state.get("varo_data")
    if not data:
        return
    with st.expander("원본 데이터 보기", expanded=False):
        sheet_keys = [key for key, value in data.items() if isinstance(value, pd.DataFrame)]
        if not sheet_keys:
            render_empty_state(st, "표시할 시트가 없습니다", compact=True)
            return
        selected = st.selectbox("시트", sheet_keys, key="raw_sheet_select")
        frame = data[selected]
        if len(frame) > _PREVIEW_ROW_LIMIT:
            st.caption(f"처음 {_PREVIEW_ROW_LIMIT}행만 미리보기로 표시합니다 (전체 {len(frame)}행).")
            frame = frame.head(_PREVIEW_ROW_LIMIT)
        st.dataframe(_display_df(frame), hide_index=True, width="stretch")

    quality = data.get("quality_check")
    if isinstance(quality, pd.DataFrame) and not quality.empty:
        with st.expander("Quality_Check 시트 보기", expanded=False):
            st.caption("참고용 점검 시트이며 V2 자체 검증 결과와 별도로 표시합니다.")
            st.dataframe(_display_df(quality.head(_PREVIEW_ROW_LIMIT)), hide_index=True, width="stretch")


def _render_downloads() -> None:
    render_section_header(st, "분석 결과 다운로드", "")
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


def render_data_management_page() -> None:
    render_page_header(
        st, "데이터 관리",
        "데이터를 불러오고 검사한 뒤 적용합니다. 파일 업로드·교체는 상단 데이터 바에서 합니다.",
    )
    view = build_data_management_view(st.session_state)

    # 1) 현재 상태 + 다음 행동
    _render_status_header(view)

    # 파일을 아예 읽지 못한 경우(행 단위 문제 없이 로드 실패)
    if view["load_error"] and not view["pending"]:
        render_error_card(st, "파일을 읽을 수 없습니다", view["load_error"])

    # 2) 현재 사용 중 데이터 (적용 성공 안내 포함)
    if view["has_current"]:
        _render_current_data_card(view)
        _render_apply_message()

    # 3) 데이터 불러오기 (샘플)
    render_section_header(st, "데이터 불러오기", "")
    if not view["has_current"] and not view["pending"]:
        st.caption("Excel 또는 CSV 파일은 상단 데이터 바에서 올리고, 아래에서 기본 샘플을 사용할 수 있습니다.")
    _render_sample_selector()
    _render_dqn_sample_selector()

    # 4) 검사 중인 데이터 (검사 결과·미리보기·명시적 적용/취소)
    if view["pending"]:
        _render_pending_check(view)

    # 5) 적용 데이터 상세
    if view["has_current"]:
        _render_upload_quality()
        _render_data_issues()
        _render_preview()
        _render_downloads()
