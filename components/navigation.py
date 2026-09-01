"""App shell and navigation components for Varo V2."""
from __future__ import annotations

import streamlit as st

from components.data_toolbar import render_quick_data_bar
from components.status import app_status_badges_html
from services.app_state import has_applied_data

MENU_ITEMS = [
    "운영 현황",
    "추천 실행",
    "경로 상세",
    "분석 및 검증",
    "데이터 관리",
]


def get_current_menu() -> str:
    current = st.session_state.get("current_menu")
    if current not in MENU_ITEMS:
        st.session_state["current_menu"] = MENU_ITEMS[0]
    return st.session_state["current_menu"]


def _file_label() -> str:
    if not has_applied_data(st.session_state.get("varo_data")):
        return "데이터 없음"
    return st.session_state.get("uploaded_filename") or "파일명 없음"


def _navigate_to(menu: str) -> None:
    st.session_state["current_menu"] = menu


def render_sidebar_nav(current_menu: str) -> None:
    """Page navigation lives in the collapsed sidebar (no horizontal menu)."""
    with st.sidebar:
        st.markdown('<div class="v2-sidenav-title">메뉴</div>', unsafe_allow_html=True)
        for item in MENU_ITEMS:
            st.button(
                item,
                key=f"nav_{item}",
                width="stretch",
                type="primary" if item == current_menu else "secondary",
                on_click=_navigate_to,
                args=(item,),
            )


def render_app_shell() -> None:
    current_menu = get_current_menu()
    st.markdown(
        f"""
        <div class="v2-topbar">
            <div class="v2-brand">VARO V2</div>
            <div class="v2-topbar-meta">
                {app_status_badges_html(st.session_state)}
                <span class="v2-file-label">{_file_label()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar_nav(current_menu)
    render_quick_data_bar()
