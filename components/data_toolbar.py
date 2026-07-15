"""Compact global workbook controls for Varo V2."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.app_state import has_app_data

def _go_to_data_management() -> None:
    st.session_state["current_menu"] = "데이터 관리"


def _toggle_replace_controls() -> None:
    st.session_state["quick_replace_open"] = not bool(st.session_state.get("quick_replace_open", False))


def _load_sample(progress_callback=None) -> bool:
    from services.data_application import clear_pending, load_and_apply
    from services.data_loader import SAMPLE_FILENAME, get_default_sample_path

    sample_path = get_default_sample_path(Path(__file__).resolve().parents[1])
    if not sample_path.exists():
        clear_pending(st.session_state)
        st.session_state["pending_load_error"] = "V2 data 폴더에 기본 샘플 파일이 없습니다."
        return False
    return load_and_apply(
        st.session_state, sample_path, SAMPLE_FILENAME, "샘플 추천 데이터",
        progress_callback=progress_callback,
    )


def _run_with_load_status(loader) -> bool:
    status = st.status("데이터 읽는 중", expanded=False)

    def update(label: str) -> None:
        status.update(
            label=label,
            state="complete" if label == "데이터 적용 완료" else "running",
            expanded=False,
        )

    applied = bool(loader(update))
    if not applied:
        failed_stage = st.session_state.get("pending_load_stage") or "데이터 적용"
        status.update(label=f"{failed_stage} 실패", state="error", expanded=False)
    return applied


def _render_load_controls(key_prefix: str) -> None:
    from services.data_application import load_and_apply, uploaded_signature

    upload_col, sample_col = st.columns([3.4, 1], gap="small")
    with upload_col:
        uploaded_file = st.file_uploader(
            "V2 엑셀 파일",
            type=["xlsx"],
            accept_multiple_files=False,
            key=f"{key_prefix}_uploader",
        )
        if uploaded_file is None:
            st.session_state.pop(f"_{key_prefix}_signature", None)
        else:
            signature = uploaded_signature(uploaded_file)
            if st.session_state.get(f"_{key_prefix}_signature") != signature:
                applied = _run_with_load_status(
                    lambda callback: load_and_apply(
                        st.session_state,
                        uploaded_file,
                        uploaded_file.name,
                        "업로드된 추천 결과",
                        progress_callback=callback,
                    )
                )
                st.session_state[f"_{key_prefix}_signature"] = signature
                if applied:
                    st.session_state["quick_replace_open"] = False
                    st.rerun()
    with sample_col:
        if st.button("기본 샘플 불러오기", key=f"{key_prefix}_sample", width="stretch"):
            if _run_with_load_status(_load_sample):
                st.session_state["quick_replace_open"] = False
                st.rerun()


def _render_feedback() -> None:
    load_error = st.session_state.get("pending_load_error")
    validation_error = st.session_state.get("pending_validation_error")
    pending_validation = st.session_state.get("pending_varo_validation")
    if load_error or validation_error:
        st.error(load_error or validation_error)
    elif pending_validation and getattr(pending_validation, "has_errors", False):
        st.error("데이터를 적용할 수 없습니다.")
    elif st.session_state.get("data_apply_message"):
        st.success(st.session_state["data_apply_message"])


def render_quick_data_bar() -> None:
    """Render onboarding controls when empty and a compact replacement bar otherwise."""
    has_data = has_app_data(
        st.session_state.get("varo_data"),
        st.session_state.get("varo_recommendations"),
    )
    if not has_data:
        st.markdown(
            """
            <div class="v2-wrap v2-data-onboarding">
              <div class="v2-data-title">엑셀을 먼저 업로드해주세요</div>
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
