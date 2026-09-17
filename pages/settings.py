"""User-facing settings and data controls backed by existing Varo state."""
from __future__ import annotations

import streamlit as st

from components.cards import render_page_header, render_section_header
from components.data_toolbar import render_data_source_controls
from components.status import badge_html
from pages.data_management import (
    _render_downloads,
    _render_quality_check,
    _render_sheet_summary,
    _render_upload_quality,
    render_data_sample_controls,
)
from services.dqn_service import get_torch_runtime_info


def _save_settings(display_mode: str, speed: str, inventory_view: str, episodes: int, rate: float, candidates: int) -> None:
    st.session_state["home_sim_display_mode"] = display_mode
    st.session_state["simulation_speed"] = speed
    st.session_state["home_sim_inventory_view"] = inventory_view
    st.session_state["settings_default_episodes"] = episodes
    st.session_state["settings_default_learning_rate"] = rate
    st.session_state["settings_default_candidate_count"] = candidates
    st.session_state["dqn_single_episodes"] = episodes
    st.session_state["dqn_learning_rate"] = rate
    st.session_state["settings_saved"] = True


def render_settings_page() -> None:
    render_page_header(
        st,
        "설정",
        "시뮬레이션 기본값과 학습 조건, 사용할 데이터를 관리하세요.",
    )
    runtime = get_torch_runtime_info()
    sections = st.columns(2, gap="medium")
    with sections[0]:
        with st.container(border=True):
            st.markdown('<div class="v3-panel-title">기본 설정</div>', unsafe_allow_html=True)
            display_options = ["단일 경로", "상위 3개"]
            speed_options = ["느림", "보통", "빠름"]
            inventory_options = ["전후 비교", "이동 전", "이동 후"]
            display_mode = st.selectbox(
                "기본 표시 방식",
                display_options,
                index=display_options.index(st.session_state.get("home_sim_display_mode", "단일 경로")),
                key="settings_display_mode",
            )
            speed = st.selectbox(
                "기본 시뮬레이션 속도",
                speed_options,
                index=speed_options.index(st.session_state.get("simulation_speed", "보통")),
                key="settings_speed",
            )
            inventory_view = st.selectbox(
                "기본 재고 표시",
                inventory_options,
                index=inventory_options.index(st.session_state.get("home_sim_inventory_view", "전후 비교")),
                key="settings_inventory_view",
            )
    with sections[1]:
        with st.container(border=True):
            st.markdown('<div class="v3-panel-title">모델·의사결정 설정</div>', unsafe_allow_html=True)
            st.markdown(
                badge_html(
                    str(runtime.get("status") or "학습 환경 확인 필요"),
                    "success" if runtime.get("available") else "warning",
                ),
                unsafe_allow_html=True,
            )
            st.selectbox("DQN 반영 방식", ["참고만"], disabled=True, key="settings_dqn_mode")
            st.toggle("VHS 자동 가중치", value=True, disabled=True, key="settings_auto_weight")
            model_columns = st.columns(3, gap="small")
            episodes = int(model_columns[0].number_input(
                "기본 Episodes", min_value=20, max_value=500,
                value=int(st.session_state.get("settings_default_episodes", 80)), step=10,
                key="settings_episodes",
            ))
            rate = float(model_columns[1].number_input(
                "Learning Rate", min_value=0.0001, max_value=0.0100,
                value=float(st.session_state.get("settings_default_learning_rate", 0.001)),
                step=0.0001, format="%.4f", key="settings_learning_rate",
            ))
            candidates = int(model_columns[2].number_input(
                "기본 후보 수", min_value=3, max_value=500,
                value=int(st.session_state.get("settings_default_candidate_count", 20)), step=1,
                key="settings_candidate_count",
            ))

    _, save_column = st.columns([4.8, 1], gap="small")
    if save_column.button("설정 저장", type="primary", width="stretch", key="settings_save"):
        _save_settings(display_mode, speed, inventory_view, episodes, rate, candidates)
    if st.session_state.get("settings_saved"):
        st.success("현재 세션의 기본 설정을 저장했습니다.")

    render_section_header(st, "데이터·결과 설정", "실제 분석 데이터와 저장 결과를 관리합니다.")
    with st.container(border=True):
        st.markdown('<div class="v3-panel-title">엑셀 데이터 교체</div>', unsafe_allow_html=True)
        render_data_source_controls("settings_source")
        render_data_sample_controls()

    data = st.session_state.get("varo_data")
    validation = st.session_state.get("varo_validation")
    if data:
        with st.expander("현재 데이터 상태", expanded=False):
            _render_sheet_summary(data, validation)
            _render_upload_quality()
            _render_quality_check(data)
        with st.expander("결과 다운로드", expanded=False):
            _render_downloads()
