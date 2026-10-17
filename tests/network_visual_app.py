"""Read-only network visual regression harness; no history/model writes.

Run from the repository: python -m streamlit run tests/network_visual_app.py
"""
import streamlit as st
from pages.overview import _network_markup_cached, _decorate_nodes
from styles import apply_global_styles

st.set_page_config(layout="wide")
apply_global_styles()
st.title("이동 계획 연결 검증")
case = st.radio("검증 사례", ["직접 이동", "경유 이동", "여러 경로", "경유 정보 누락", "이동량 0", "수협 실제 자료"], horizontal=True)
playing = st.checkbox("차량 이동", value=False)
nodes = [dict(node_id=n, node_name=("긴 이름의 점포 " + n), node_type="STORE", network_mode="DIRECT_NETWORK")
         for n in ("A", "B", "C", "D")]
nodes.append(dict(node_id="DC01", node_name="중앙 물류센터", node_type="DC"))
routes = [dict(route_id="R1", source_id="A", target_id="B", route_type="DIRECT", recommended_qty=46, product_name="검증 상품")]
if case == "경유 이동":
    routes[0].update(route_type="VIA_DC", dc_id="DC01")
elif case == "경유 정보 누락":
    routes[0].update(route_type="VIA_DC")
elif case == "여러 경로":
    routes += [dict(route_id="R2", source_id="B", target_id="A", route_type="DIRECT", recommended_qty=20),
               dict(route_id="R3", source_id="A", target_id="D", route_type="VIA_DC", dc_id="DC01", recommended_qty=30)]
elif case == "이동량 0":
    routes[0]["recommended_qty"] = 0
scenario = {"transitions": [dict(route_id=r["route_id"], route_type=r["route_type"], dc_id=r.get("dc_id"),
    applied_quantity=r["recommended_qty"], feasible=r["recommended_qty"] > 0) for r in routes]}
if case == "수협 실제 자료":
    from pathlib import Path
    from services.data_loader import load_excel_data
    from services.analysis_pipeline import build_v2_state, sort_recommendations
    from services.inventory_transition_service import run_inventory_scenario
    from simulation.dynamic_network import build_network_nodes
    source = Path(__file__).resolve().parents[1] / "outputs/ui_validation/suhyup_20260701_ui_validation.xlsx"
    if not source.exists():
        st.info("기존 수협 검증 입력 파일을 확인해 주세요.")
        st.stop()
    data = load_excel_data(source)
    candidates = sort_recommendations(build_v2_state(data, detail_level="core")["recommendations"])
    routes = candidates[:1]
    scenario = run_inventory_scenario(data, routes)
    nodes = build_network_nodes(data, candidates)
    st.caption("수협 2026-07-01 기존 검증 입력 · 실제 실행 아님 · 원본/이력 저장 없음")
nodes = _decorate_nodes(nodes, scenario, routes, "전후 비교")
result = _network_markup_cached("visual-fixture", nodes, routes, [], playing, 14., False, "R1", scenario, display_mode="상위 3개")
if result["ok"]:
    st.markdown(result["html"], unsafe_allow_html=True)
else:
    st.error("경로 정보 확인 필요")
