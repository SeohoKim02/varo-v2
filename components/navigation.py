"""App shell and navigation components for Varo V2.

The menu a user reads is four entries, named after the work rather than the code:

    재고 운영          the Workspace — where a normal day starts and ends
    데이터 관리         upload / inspect / apply (heavy, genuinely separate work)
    분석 및 검증         research-grade validation detail
    운영 시뮬레이션      the moving-truck picture, a different job to the Workspace

``추천 실행`` and ``경로 상세`` are now fully covered by the Workspace, so they are
demoted: the routes still exist and still render (internal links, bookmarks and
tests keep working), but they sit behind a folded 예전 화면 group instead of
competing with the four screens above. Route keys stay in Korean and unchanged —
only the *label* differs from the key, via :data:`MENU_LABELS`.
"""
from __future__ import annotations

import streamlit as st

from components.data_toolbar import render_quick_data_bar
from components.status import app_status_badges_html
from services.app_state import has_applied_data

WORKSPACE_MENU = "재고 운영"
SIMULATION_MENU = "운영 현황"

# The four screens a user chooses between. MENU_ITEMS[0] is the landing page.
PRIMARY_MENU_ITEMS = [WORKSPACE_MENU, "데이터 관리", "분석 및 검증", SIMULATION_MENU]
# Kept reachable for compatibility, hidden from the everyday menu.
LEGACY_MENU_ITEMS = ["추천 실행", "경로 상세"]
MENU_ITEMS = [*PRIMARY_MENU_ITEMS, *LEGACY_MENU_ITEMS]

# Route key → what the user reads. Only entries that differ are listed.
MENU_LABELS = {
    SIMULATION_MENU: "운영 시뮬레이션",
}

MENU_HINTS = {
    WORKSPACE_MENU: "오늘 권장 이동을 확인하고 실행계획을 기록합니다.",
    "데이터 관리": "파일 업로드 · 검사 · 적용",
    "분석 및 검증": "연구용 상세 검증 지표",
    SIMULATION_MENU: "추천 경로의 이동 흐름을 애니메이션으로 확인합니다.",
    "추천 실행": "재고 운영 화면으로 대체된 예전 목록 화면입니다.",
    "경로 상세": "재고 운영 세부 정보로 대체된 예전 상세 화면입니다.",
}

LEGACY_GROUP_LABEL = "예전 화면"
LEGACY_GROUP_NOTE = "재고 운영 화면이 대신하는 화면입니다. 기존 링크 호환을 위해 남겨둡니다."


def menu_label(menu: str) -> str:
    """The user-facing name for a route key."""
    return MENU_LABELS.get(menu, menu)


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


def _nav_button(item: str, current_menu: str) -> None:
    st.button(
        menu_label(item),
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
        for item in PRIMARY_MENU_ITEMS:
            _nav_button(item, current_menu)
        # Folded, and opened automatically when a legacy screen is the one on
        # screen so the user can still see where they are.
        with st.expander(LEGACY_GROUP_LABEL, expanded=current_menu in LEGACY_MENU_ITEMS):
            st.caption(LEGACY_GROUP_NOTE)
            for item in LEGACY_MENU_ITEMS:
                _nav_button(item, current_menu)


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
