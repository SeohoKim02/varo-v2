"""Page router for Varo V2."""
from __future__ import annotations

from importlib import import_module

import streamlit as st

from components.navigation import MENU_ITEMS, get_current_menu

_PAGE_RENDERERS = {
    "시뮬레이션": ("pages.overview", "render_overview_page"),
    "전략 비교": ("pages.strategy_comparison", "render_strategy_comparison_page"),
    "학습 관리": ("pages.training_management", "render_training_management_page"),
    "결과 이력": ("pages.results_history", "render_results_history_page"),
    "설정": ("pages.settings", "render_settings_page"),
    # 이전 세션과 기존 회귀 검증이 가진 화면 키는 계속 렌더링한다.
    "홈": ("pages.overview", "render_overview_page"),
    "추천 실행": ("pages.recommendations", "render_recommendations_page"),
    "경로 상세": ("pages.route_detail", "render_route_detail_page"),
    "분석 및 검증": ("pages.validation", "render_validation_page"),
    "데이터 관리": ("pages.data_management", "render_data_management_page"),
}


def render_current_page() -> None:
    """Render the selected V2 page."""
    selected_page = get_current_menu()
    if selected_page not in _PAGE_RENDERERS:
        st.session_state["current_menu"] = MENU_ITEMS[0]
        selected_page = MENU_ITEMS[0]
    module_name, function_name = _PAGE_RENDERERS[selected_page]
    renderer = getattr(import_module(module_name), function_name)
    renderer()
