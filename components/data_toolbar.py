"""Compact global workbook controls for Varo V2."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.app_state import has_app_data

def _go_to_data_management() -> None:
    st.session_state["current_menu"] = "설정"


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
        if st.session_state.get("varo_data"):
            st.warning("새 파일을 적용하지 못했습니다. 화면에는 이전에 정상 적용한 데이터가 유지됩니다.")
        st.caption("파일·시트·필수 열을 확인하고 파일을 다시 선택하세요. 설정에서 상세 검증 결과를 확인할 수 있습니다.")
    elif pending_validation and getattr(pending_validation, "has_errors", False):
        st.error("데이터를 적용할 수 없습니다.")
    elif st.session_state.get("data_apply_message"):
        st.success(st.session_state["data_apply_message"])
        contract = (st.session_state.get("varo_pipeline_result") or {}).get("final_decision_integration") or {}
        summary = contract.get("summary") or {}
        if summary:
            st.caption(f"{st.session_state.get('uploaded_filename') or '적용 파일'} · 상품 {summary.get('unique_products', 0):,}개 · "
                       f"상품·점포 결과 {summary.get('product_location_count', 0):,}개 · 전체 상품 분석 완료")


def render_data_source_controls(key_prefix: str = "data_source") -> None:
    """Render the existing upload/sample flow inside a product page."""
    _render_load_controls(key_prefix)
    _render_feedback()


def render_home_analysis_inputs() -> None:
    """Upload then explicitly analyze, using the existing load/apply contract."""
    from services.data_application import load_and_apply
    active = bool(st.session_state.get("varo_data"))
    if active:
        st.caption(f"현재 결과: {st.session_state.get('uploaded_filename') or '불러온 데이터'}")
    with st.expander("데이터 업로드 · 분석 실행", expanded=not active):
        uploaded = st.file_uploader("분석할 엑셀 파일", type=["xlsx"], key="home_source_uploader")
        pending = st.session_state.get("_home_pending_upload")
        if uploaded is not None:
            identity = (getattr(uploaded, "file_id", None), uploaded.name, getattr(uploaded, "size", None))
            if not pending or pending.get("identity") != identity:
                pending = {"identity": identity, "name": uploaded.name, "content": uploaded.getvalue(),
                           "signature": st.session_state.get("data_signature")}
                st.session_state["_home_pending_upload"] = pending
        if pending and uploaded is None:
            st.caption(f"선택한 파일: {pending['name']}")
        controls = st.columns(2, gap="small")
        if controls[0].button("분석 실행", type="primary", key="home_analyze", disabled=not pending, width="stretch"):
            if _run_with_load_status(lambda callback: load_and_apply(
                st.session_state, pending["content"], pending["name"], "업로드된 추천 결과", progress_callback=callback,
            )):
                st.session_state.pop("_home_pending_upload", None)
                st.rerun()
        if controls[1].button("샘플로 시작", key="home_sample", width="stretch"):
            if _run_with_load_status(_load_sample):
                st.rerun()
        if active and pending:
            st.caption("선택한 파일은 ‘분석 실행’을 누르면 반영됩니다. 그전까지 기존 결과를 유지합니다.")
        _render_feedback()


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
