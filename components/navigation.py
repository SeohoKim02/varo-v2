"""App shell and navigation components for Varo V2."""
from __future__ import annotations

import streamlit as st

from components.data_toolbar import render_quick_data_bar
from components.status import user_status_label
from services.app_state import current_data_status, has_app_data

MENU_ITEMS = [
    "시뮬레이션",
    "전략 비교",
    "학습 관리",
    "결과 이력",
    "설정",
]

LEGACY_MENU_ITEMS = ["홈", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리"]

MENU_ICONS = {
    "시뮬레이션": "⌂",
    "전략 비교": "▥",
    "학습 관리": "◇",
    "결과 이력": "▤",
    "설정": "⚙",
}

LEGACY_ACTIVE_MENU = {
    "홈": "시뮬레이션",
    "추천 실행": "결과 이력",
    "경로 상세": "결과 이력",
    "분석 및 검증": "전략 비교",
    "데이터 관리": "설정",
}


def get_current_menu() -> str:
    current = st.session_state.get("current_menu")
    if current not in MENU_ITEMS and current not in LEGACY_MENU_ITEMS:
        st.session_state["current_menu"] = MENU_ITEMS[0]
    return st.session_state["current_menu"]


def _analysis_status() -> str:
    pending_validation = st.session_state.get("pending_varo_validation")
    if pending_validation and getattr(pending_validation, "has_errors", False):
        return "검증 오류"
    return current_data_status(st.session_state)


def _file_label() -> str:
    if not has_app_data(
        st.session_state.get("varo_data"),
        st.session_state.get("varo_recommendations"),
    ):
        return "데이터 없음"
    return st.session_state.get("uploaded_filename") or "파일명 없음"


def _navigate_to(menu: str) -> None:
    st.session_state["current_menu"] = menu


def render_sidebar_nav(current_menu: str) -> None:
    """Render the five product-oriented destinations in the sidebar."""
    active_menu = LEGACY_ACTIVE_MENU.get(current_menu, current_menu)
    with st.sidebar:
        st.markdown(
            """
            <div class="v3-sidebar-brand">
              <div class="v3-sidebar-logo">VARO</div>
              <div class="v3-sidebar-tagline">Smarter Inventory<br>A Better Tomorrow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="v2-sidenav-title">WORKSPACE</div>', unsafe_allow_html=True)
        for item in MENU_ITEMS:
            st.button(
                f"{MENU_ICONS[item]}　{item}",
                key=f"nav_{item}",
                width="stretch",
                type="primary" if item == active_menu else "secondary",
                on_click=_navigate_to,
                args=(item,),
            )
        training_result = st.session_state.get("dqn_training_result") or {}
        dqn_status = user_status_label(
            training_result.get("final_status")
            or training_result.get("stability_status")
            or training_result.get("status")
            or "학습 필요"
        )
        st.markdown(
            f'<div class="v3-sidebar-footer"><span>학습 상태</span><strong>{dqn_status}</strong></div>',
            unsafe_allow_html=True,
        )


def render_app_shell() -> None:
    current_menu = get_current_menu()
    render_sidebar_nav(current_menu)
    st.markdown(
        f"""
        <div class="v3-contextbar">
            <span class="v3-context-file">{_file_label()}</span>
            <span class="v2-pill">{_analysis_status()}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not has_app_data(
        st.session_state.get("varo_data"),
        st.session_state.get("varo_recommendations"),
    ):
        render_quick_data_bar()
