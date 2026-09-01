"""Compact global workbook controls for Varo V2."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.app_state import has_applied_data
from services.data_application import clear_pending, load_and_apply, prepare_pending_data, uploaded_signature
from services.data_loader import SAMPLE_FILENAME, get_default_sample_path


def _go_to_data_management() -> None:
    st.session_state["current_menu"] = "데이터 관리"


def _toggle_replace_controls() -> None:
    st.session_state["quick_replace_open"] = not bool(st.session_state.get("quick_replace_open", False))


def _load_sample() -> bool:
    sample_path = get_default_sample_path(Path(__file__).resolve().parents[1])
    if not sample_path.exists():
        clear_pending(st.session_state)
        st.session_state["pending_load_error"] = "V2 data 폴더에 기본 샘플 파일이 없습니다."
        return False
    return load_and_apply(st.session_state, sample_path, SAMPLE_FILENAME, "샘플 추천 데이터")


def _render_load_controls(key_prefix: str) -> None:
    upload_col, sample_col = st.columns([3.4, 1], gap="small")
    with upload_col:
        uploaded_file = st.file_uploader(
            "데이터 파일 (.xlsx · .xls · .csv)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=False,
            key=f"{key_prefix}_uploader",
        )
        if uploaded_file is None:
            st.session_state.pop(f"_{key_prefix}_signature", None)
        else:
            signature = uploaded_signature(uploaded_file)
            if st.session_state.get(f"_{key_prefix}_signature") != signature:
                # Two-phase: inspect the file into pending state only. It is NOT
                # applied until the user clicks the apply button on 데이터 관리.
                prepare_pending_data(
                    st.session_state,
                    uploaded_file,
                    uploaded_file.name,
                    "업로드된 추천 결과",
                )
                st.session_state[f"_{key_prefix}_signature"] = signature
                st.session_state["quick_replace_open"] = False
                st.session_state["current_menu"] = "데이터 관리"
                st.rerun()
    with sample_col:
        st.caption("빠른 시작")
        if st.button("기본 샘플 불러오기", key=f"{key_prefix}_sample", width="stretch"):
            if _load_sample():
                st.session_state["quick_replace_open"] = False
                st.rerun()


def _render_feedback() -> None:
    load_error = st.session_state.get("pending_load_error")
    if load_error:
        st.error(load_error)
        return
    # The detailed 검사 결과 + 적용 버튼 live on 데이터 관리; only a short intake
    # notice is shown on other pages so the current result stays in focus.
    if st.session_state.get("current_menu") == "데이터 관리":
        return
    status = st.session_state.get("pending_status")
    if not status:
        return
    if status == "사용 불가":
        st.warning("검사한 새 데이터에 오류가 있습니다. 데이터 관리에서 확인하세요.")
    elif status == "현재 데이터와 동일":
        st.info("검사한 데이터가 현재 사용 중인 데이터와 같습니다.")
    else:
        st.info("새 데이터 검사가 완료됐습니다. 데이터 관리에서 적용하세요.")


def render_quick_data_bar() -> None:
    """Render onboarding controls when empty and a compact replacement bar otherwise."""
    has_data = has_applied_data(st.session_state.get("varo_data"))
    if not has_data:
        st.markdown(
            """
            <div class="v2-wrap v2-data-onboarding">
              <div class="v2-data-title">엑셀을 먼저 업로드해주세요</div>
              <div class="v2-card-caption">검사 후 데이터를 적용하고 추천 실행을 시작할 수 있습니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_load_controls("quick_empty")
        _render_feedback()
        return

    _, replace_col = st.columns([6.2, 1], gap="small")
    replace_col.button(
        "데이터 교체",
        key="quick_replace_toggle",
        on_click=_toggle_replace_controls,
        width="stretch",
    )
    if st.session_state.get("quick_replace_open", False):
        _render_load_controls("quick_loaded")
    _render_feedback()
