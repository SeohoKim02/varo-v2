"""Shared workspace-state card: one title, one short message, one action.

Both the home dashboard and the 분석·검증 page render non-result states through
this single card so their wording and navigation stay identical (no duplicated
state logic). The card reads a resolved ``home_state`` dict (see
``services.home_state``); it never re-derives status. The primary button
navigates with an ``on_click`` callback so the rerun lands on the target page in
one step.
"""
from __future__ import annotations

import html
from typing import Any, Mapping

import streamlit as st


def _safe(value: Any) -> str:
    return html.escape(str(value)) if value is not None else "-"


def _navigate(page: str) -> None:
    st.session_state["current_menu"] = page


def render_state_action_card(home: Mapping[str, Any], key: str) -> None:
    """Render title + short message + a single primary action for a non-result state."""
    st.markdown(
        f"""
        <div class="v2-wrap v2-card v2-home-state-card">
          <div class="v2-card-title">{_safe(home.get("title"))}</div>
          <div class="v2-card-caption" style="margin-top:0.4rem;">{_safe(home.get("short_message"))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1.4, 4], gap="small")
    cols[0].button(
        home.get("next_action_label") or "데이터 관리로 이동",
        key=key,
        type="primary",
        width="stretch",
        on_click=_navigate,
        args=(home.get("next_page") or "데이터 관리",),
    )


def render_state_summary_card(
    title: Any,
    message: Any,
    *,
    action_label: str | None = None,
    action_page: str | None = None,
    key: str | None = None,
) -> None:
    """Title + short message from the shared state, with an *optional* action.

    Used by the 데이터 관리 page so its top status line reuses the same wording as
    the home/검증 cards, but without forcing a navigation button when the page
    body itself is the next step (파일/샘플 선택, 문제 수정). When ``action_label``/
    ``action_page``/``key`` are all given, one primary button is shown.
    """
    st.markdown(
        f"""
        <div class="v2-wrap v2-card v2-home-state-card">
          <div class="v2-card-title">{_safe(title)}</div>
          <div class="v2-card-caption" style="margin-top:0.4rem;">{_safe(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action_label and action_page and key:
        cols = st.columns([1.4, 4], gap="small")
        cols[0].button(
            action_label,
            key=key,
            type="primary",
            width="stretch",
            on_click=_navigate,
            args=(action_page,),
        )
