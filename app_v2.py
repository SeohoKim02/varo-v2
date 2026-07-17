"""Varo V2 Streamlit entry point."""
from __future__ import annotations

import streamlit as st

from components.navigation import render_app_shell
from router import render_current_page
from styles import apply_global_styles


def initialize_session_state() -> None:
    """Create the shared V2 session keys used across pages."""
    defaults = {
        "varo_data": None,
        "varo_validation": None,
        "varo_recommendations": [],
        "varo_pipeline_result": {},
        "analysis_result": {},
        "pipeline_summary": {},
        "connected_algorithms": [],
        "deferred_algorithms": [],
        "dqn_excluded": {},
        "dqn_training_result": None,
        "dqn_reflection_mode": "DQN 참고만",
        "dqn_batch_result": None,
        "dqn_comparison_result": None,
        "dqn_original_batch_result": None,
        "dqn_balanced_batch_result": None,
        "dqn_batch_comparison_result": None,
        "dqn_sample_diagnosis": None,
        "dqn_balanced_files": None,
        "dqn_baseline_recommendations": None,
        "dqn_baseline_pipeline": None,
        "dqn_sample_training_mode": "original",
        "dqn_selected_sample": None,
        "sensitivity_settings": {},
        "sensitivity_result": None,
        "sensitivity_summary": None,
        "sensitivity_data_signature": None,
        "sensitivity_is_running": False,
        "sensitivity_last_error": None,
        "optimality_gap_settings": {},
        "optimality_gap_result": None,
        "optimality_gap_data_signature": None,
        "optimality_gap_is_running": False,
        "optimality_gap_last_error": None,
        "selected_route_id": None,
        "uploaded_filename": None,
        "data_source_type": None,
        "data_signature": None,
        "upload_report": {},
        "recommendation_source": "uploaded",
        "simulation_snapshot": None,
        "simulation_speed": "보통",
        "show_all_routes": False,
        "home_sim_playing": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    for legacy_key in (
        "v2_uploaded_data", "v2_uploaded_file_name", "v2_standard_recommendations",
        "v2_selected_route_id", "v2_simulation_snapshot", "v2_simulation_speed",
        "v2_show_all_routes",
    ):
        st.session_state.pop(legacy_key, None)


def main() -> None:
    st.set_page_config(
        page_title="VARO V2", page_icon="V2", layout="wide", initial_sidebar_state="expanded",
    )
    initialize_session_state()
    apply_global_styles()
    render_app_shell()
    render_current_page()


if __name__ == "__main__":
    main()
