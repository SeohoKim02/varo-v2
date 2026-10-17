"""Explicit research opt-in entry point; the ordinary app remains unchanged."""
from __future__ import annotations

import os
import streamlit as st


def main() -> None:
    if os.environ.get("VARO_CANDIDATE_REVIEW") == "1":
        st.session_state["_candidate_review_opt_in"] = True
    if not st.session_state.get("_candidate_review_opt_in"):
        st.set_page_config(page_title="Varo 후보 기능 검토", layout="wide")
        st.title("Varo 후보 기능 검토")
        st.write("기존 분석과 별도로 새 의사결정 방식을 검토합니다. 운영 반영이나 실제 이동·가격 변경은 하지 않습니다.")
        if st.button("후보 검토 시작", key="candidate_opt_in", type="primary"):
            st.session_state["_candidate_review_opt_in"] = True
            st.session_state["current_menu"] = "설정"
            st.rerun()
        return
    from app_v2 import main as existing_main
    existing_main()


if __name__ == "__main__":
    main()
