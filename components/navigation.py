"""App shell and navigation components for Varo V2."""
from __future__ import annotations

import streamlit as st
import html
import os
from copy import deepcopy
from datetime import date, time
from services.app_state import current_data_status, has_app_data

MENU_ITEMS = [
    "시뮬레이션",
    "전략 비교",
    "학습 관리",
    "결과 이력",
    "설정",
]

LEGACY_MENU_ITEMS = ["홈", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리"]

_NON_PERSISTENT_INPUTS = {
    "settings_save", "settings_seller_input_template", "strategy_compare_run",
    "history_search_button", "history_current_recommendations", "settings_source_sample",
    "settings_candidate_review", "candidate_opt_in", "candidate_load_scenario", "candidate_run",
    "candidate_save", "candidate_validate_upload", "settings_auto_weight", "settings_dqn_mode",
    "candidate_real_run", "candidate_uploaded_save", "candidate_compare_alternatives",
    "candidate_inventory_upload",  # File widgets cannot be assigned through session_state, even an empty list.
}

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


def candidate_enabled() -> bool:
    """Research entry point only; never enable by merely selecting a menu key."""
    return os.environ.get("VARO_CANDIDATE_REVIEW") == "1" or st.session_state.get("_candidate_review_opt_in") is True


def get_current_menu() -> str:
    current = st.session_state.get("current_menu")
    if current == "후보 결과 검토" and candidate_enabled():
        return current
    if current not in MENU_ITEMS and current not in LEGACY_MENU_ITEMS:
        st.session_state["current_menu"] = MENU_ITEMS[0]
    return st.session_state["current_menu"]


def _analysis_status() -> str:
    pending_validation = st.session_state.get("pending_varo_validation")
    if pending_validation and getattr(pending_validation, "has_errors", False):
        return "검증 오류"
    if _has_current_data() and not st.session_state.get("varo_recommendations"):
        return "상품 분석 완료 · 이동 후보 없음"
    return current_data_status(st.session_state)


def _has_current_data() -> bool:
    contract = (st.session_state.get("varo_pipeline_result") or {}).get("final_decision_integration") or {}
    return has_app_data(st.session_state.get("varo_data"), st.session_state.get("varo_recommendations")) or bool(contract.get("products"))


def _file_label() -> str:
    if not _has_current_data():
        return "데이터 없음"
    return st.session_state.get("uploaded_filename") or "파일명 없음"


def _navigate_to(menu: str) -> None:
    _remember_inputs()
    current = get_current_menu()
    if current != menu:
        history = list(st.session_state.get("navigation_history") or [])
        st.session_state["navigation_history"] = [*history, current][-25:]
    st.session_state["current_menu"] = menu


def _go_back() -> None:
    _remember_inputs()
    history = list(st.session_state.get("navigation_history") or [])
    if history:
        st.session_state["current_menu"] = history.pop()
        st.session_state["navigation_history"] = history


def _remember_inputs() -> None:
    """Preserve view inputs separately from Streamlit's page-scoped widgets."""
    signature = st.session_state.get("data_signature")
    memory = st.session_state.get("_ui_input_memory") or {}
    values = dict(memory.get("values") or {}) if memory.get("signature") == signature else {}
    prefixes = ("seller_", "history_", "simulation_", "home_sim_", "home_speed_", "home_inventory_", "home_show_", "settings_", "strategy_", "rec_filter_", "route_detail_", "sensitivity_", "optimality_gap_", "integrated_validation_", "candidate_")
    dqn_inputs = {"dqn_single_training_data", "dqn_model_type", "dqn_single_episodes", "dqn_learning_rate", "dqn_candidate_count"}
    actions = _NON_PERSISTENT_INPUTS
    for key in actions:
        values.pop(key, None)
    for key in st.session_state:
        value = st.session_state[key]
        if key in actions or "download" in key:
            continue
        if (key.startswith(prefixes) or key in dqn_inputs) and isinstance(value, (str, int, float, bool, date, time, list, tuple)):
            if not isinstance(value, (list, tuple)) or all(isinstance(v, (str, int, float, bool, date, time)) for v in value):
                values[key] = deepcopy(value)
    st.session_state["_ui_input_memory"] = {"signature": signature, "values": values}


def _restore_inputs() -> None:
    pending = st.session_state.get("_home_pending_upload")
    if pending and pending.get("signature") != st.session_state.get("data_signature"):
        st.session_state.pop("_home_pending_upload", None)
    memory = st.session_state.get("_ui_input_memory") or {}
    if memory.get("signature") != st.session_state.get("data_signature"):
        st.session_state.pop("_ui_input_memory", None)
        return
    for key, value in (memory.get("values") or {}).items():
        if key in _NON_PERSISTENT_INPUTS:
            continue
        # Reassign even existing values to detach them from widget cleanup.
        # Preserve fresh browser input rather than overwriting it with memory.
        st.session_state[key] = deepcopy(st.session_state[key] if key in st.session_state else value)


def render_site_header(current_menu: str) -> None:
    """A website header with an on-demand menu, never a persistent sidebar."""
    active_menu = LEGACY_ACTIVE_MENU.get(current_menu, current_menu)
    is_home = active_menu == "시뮬레이션"
    with st.container(key="site_header"):
        columns = st.columns([8, 1], gap="small", vertical_alignment="center")
        columns[0].button("VARO", key="brand_home", on_click=_navigate_to, args=("시뮬레이션",))
        with columns[-1]:
            with st.popover("전체 기능", icon=":material/menu:", width="stretch"):
                st.caption("이동할 화면을 선택하세요")
                for item in MENU_ITEMS:
                    st.button(
                        f"{MENU_ICONS[item]}　{item}", key=f"nav_{item}", width="stretch",
                        type="primary" if item == active_menu else "secondary",
                        on_click=_navigate_to, args=(item,),
                    )
    if not is_home:
        with st.container(key="return_links"):
            links = st.columns([1, 1, 6], gap="small")
            links[0].button("홈으로", key="ui_home", on_click=_navigate_to, args=("시뮬레이션",), width="stretch")
            links[1].button("뒤로 가기", key="ui_back", on_click=_go_back,
                            disabled=not bool(st.session_state.get("navigation_history")), width="stretch")


def render_app_shell() -> None:
    _restore_inputs()
    current_menu = get_current_menu()
    render_site_header(current_menu)
    if LEGACY_ACTIVE_MENU.get(current_menu, current_menu) != "시뮬레이션":
        st.markdown(
        f"""
        <div class="v3-contextbar">
            <span class="v3-context-file">{html.escape(_file_label())}</span>
            <span class="v2-pill">{html.escape(_analysis_status())}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
