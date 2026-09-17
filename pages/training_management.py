"""DQN training workspace using the existing guarded training pipeline."""
from __future__ import annotations

import streamlit as st

from components.cards import render_page_header
from components.status import badge_html
from pages.validation import _render_dqn
from services.dqn_service import get_torch_runtime_info


def render_training_management_page() -> None:
    runtime = get_torch_runtime_info()
    render_page_header(
        st,
        "학습 관리",
        "데이터와 학습 조건을 선택하고 DQN 학습 결과를 관리하세요.",
        badge=badge_html(
            str(runtime.get("status") or "학습 환경 확인 필요"),
            "success" if runtime.get("available") else "warning",
        ),
    )
    _render_dqn()
