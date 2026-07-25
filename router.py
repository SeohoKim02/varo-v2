"""Page router for Varo V2."""
from __future__ import annotations

import streamlit as st

from components.navigation import MENU_ITEMS, get_current_menu
from pages.data_management import render_data_management_page
from pages.overview import render_overview_page
from pages.recommendations import render_recommendations_page
from pages.route_detail import render_route_detail_page
from pages.validation import render_validation_page

_PAGE_RENDERERS = {
    "운영 현황": render_overview_page,
    "추천 실행": render_recommendations_page,
    "경로 상세": render_route_detail_page,
    "분석 및 검증": render_validation_page,
    "데이터 관리": render_data_management_page,
}


def render_current_page() -> None:
    """Render the selected V2 page."""
    selected_page = get_current_menu()
    if selected_page not in MENU_ITEMS:
        st.session_state["current_menu"] = MENU_ITEMS[0]
        selected_page = MENU_ITEMS[0]
    _PAGE_RENDERERS[selected_page]()