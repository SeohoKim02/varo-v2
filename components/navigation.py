"""App shell and navigation components for Varo V2.

Navigation is intentionally shallow: the everyday flow happens on one screen
(재고 운영 Workspace) and 데이터 관리 sits next to it. The remaining routes are
detail screens the workspace links into, so they are grouped separately in the
sidebar instead of competing with the main flow.
"""
from __future__ import annotations

import streamlit as st

from components.data_toolbar import render_quick_data_bar
from components.status import app_status_badges_html
from services.app_state import has_applied_data

WORKSPACE_MENU = "재고 운영"

# Primary work vs. detail screens. Order matters: MENU_ITEMS[0] is the landing page.
PRIMARY_MENU_ITEMS = [WORKSPACE_MENU, "데이터 관리"]
DETAIL_MENU_ITEMS = ["분석 및 검증", "추천 실행", "경로 상세", "운영 현황"]
MENU_ITEMS = [*PRIMARY_MENU_ITEMS, *DETAIL_MENU_ITEMS]

MENU_HINTS = {
    "운영 현황": "이동 시뮬레이션",
}


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


def _render_group(title: str, items: list[str], current_menu: str) -> None:
    st.markdown(f'<div class="v2-sidenav-title">{title}</div>', unsafe_allow_html=True)
    for item in items:
        st.button(
            item,
            key=f"nav_{item}",
            width="stretch",
            type="primary" if item == current_menu else "secondary",
            help=MENU_HINTS.get(item),
            on_click=_navigate_to,
            args=(item,),
        )


def render_sidebar_nav(current_menu: str) -> None:
    """Page navigation lives in the collapsed sidebar (no horizontal menu)."""
    with st.sidebar:
        _render_group("작업", PRIMARY_MENU_ITEMS, current_menu)
        _render_group("상세 보기", DETAIL_MENU_ITEMS, current_menu)


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
