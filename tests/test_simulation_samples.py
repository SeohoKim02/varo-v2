"""Regression coverage for the dynamic simulation review workbooks."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import pandas as pd

from tests.streamlit_log_silencer import quiet_streamlit_test_logs
from services.app_state import CANONICAL_DATA_KEYS
from services.data_application import load_and_apply
from services.data_loader import load_excel_data
from services.data_validator import validate_workbook_data
from services.sample_catalog import SAMPLE_WORKBOOKS, sample_path
from simulation.dynamic_network import (
    build_network_nodes,
    build_route_segments,
    compute_dynamic_layout,
)

quiet_streamlit_test_logs()

REQUIRED_SHEETS = {"stores", "products", "inventory", "routes", "v2_recommendations"}
EXPECTED = {
    "small_4stores_1dc": (4, 1, 5),
    "normal_6stores_1dc": (6, 1, 6),
    "standard_8stores_1dc": (8, 1, 8),
    "dual_dc_10stores_2dc": (10, 2, 10),
    "edge_3stores_1dc": (3, 1, 3),
}


class SimulationSampleWorkbookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = {
            sample.key: load_excel_data(sample_path(sample))
            for sample in SAMPLE_WORKBOOKS
        }

    def test_catalog_contains_only_requested_3_to_10_store_samples(self):
        self.assertEqual(set(EXPECTED), {sample.key for sample in SAMPLE_WORKBOOKS})
        self.assertTrue(all(3 <= sample.store_count <= 10 for sample in SAMPLE_WORKBOOKS))
        self.assertTrue(all(sample.dc_count in {1, 2} for sample in SAMPLE_WORKBOOKS))

    def test_sample_files_and_required_sheets_exist(self):
        for sample in SAMPLE_WORKBOOKS:
            path = sample_path(sample)
            self.assertTrue(path.is_file(), path)
            with pd.ExcelFile(path) as excel:
                self.assertTrue(REQUIRED_SHEETS.issubset(excel.sheet_names), sample.key)

    def test_all_samples_load_and_pass_v2_validation(self):
        for sample in SAMPLE_WORKBOOKS:
            report = validate_workbook_data(self.loaded[sample.key])
            self.assertFalse(report.has_errors, [message.to_dict() for message in report.messages])
            expected_stores, expected_dcs, expected_recs = EXPECTED[sample.key]
            self.assertEqual(report.summary["store_count"], expected_stores)
            self.assertEqual(report.summary["dc_count"], expected_dcs)
            self.assertEqual(report.summary["recommendation_count"], expected_recs)

    def test_dynamic_layout_reflects_each_sample_node_count(self):
        for sample in SAMPLE_WORKBOOKS:
            data = self.loaded[sample.key]
            recommendations = data["recommendations"].to_dict("records")
            layout = compute_dynamic_layout(build_network_nodes(data, recommendations), recommendations)
            self.assertTrue(layout.is_valid, layout.errors)
            self.assertEqual(len(layout.stores), sample.store_count)
            self.assertEqual(len(layout.dcs), sample.dc_count)
            self.assertEqual(layout.canvas["store_count"], sample.store_count)
            self.assertEqual(layout.canvas["dc_count"], sample.dc_count)

    def test_dual_dc_recommendations_and_segments_use_both_centers(self):
        data = self.loaded["dual_dc_10stores_2dc"]
        recommendations = data["recommendations"].to_dict("records")
        nodes = build_network_nodes(data, recommendations)
        via_routes = [route for route in recommendations if route["route_type"] == "VIA_DC"]
        self.assertEqual({route["dc_id"] for route in via_routes}, {"DC01", "DC02"})
        used = set()
        for route in via_routes:
            segments = build_route_segments(route, nodes)
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0]["to_node_id"], route["dc_id"])
            self.assertEqual(segments[1]["from_node_id"], route["dc_id"])
            used.add(segments[0]["to_node_id"])
        self.assertEqual(used, {"DC01", "DC02"})

    def test_direct_and_via_route_segment_counts(self):
        for sample in SAMPLE_WORKBOOKS:
            data = self.loaded[sample.key]
            recommendations = data["recommendations"].to_dict("records")
            nodes = build_network_nodes(data, recommendations)
            for route in recommendations:
                expected = 1 if route["route_type"] == "DIRECT" else 2
                self.assertEqual(len(build_route_segments(route, nodes)), expected)

    def test_samples_keep_dqn_excluded(self):
        forbidden = ("reward", "loss", "q_table", "policy")
        for data in self.loaded.values():
            recommendations = data["recommendations"]
            self.assertTrue((recommendations["dqn_action"] == "미연결").all())
            lowered = {str(column).lower() for column in recommendations.columns}
            self.assertFalse(any(token in column for column in lowered for token in forbidden))

    def test_narrow_geographic_coordinates_fall_back_to_ring_layout(self):
        nodes = [
            {"node_id": "DC01", "node_name": "센터", "node_type": "DC", "latitude": 37.5, "longitude": 126.9},
            {"node_id": "S01", "node_name": "점포1", "node_type": "STORE", "latitude": 37.50001, "longitude": 126.90001},
            {"node_id": "S02", "node_name": "점포2", "node_type": "STORE", "latitude": 37.50002, "longitude": 126.90002},
        ]
        layout = compute_dynamic_layout(nodes)
        self.assertEqual(layout.canvas["layout_mode"], "deterministic")

    def test_switching_samples_clears_stale_selection_filters_and_simulation(self):
        dual = next(sample for sample in SAMPLE_WORKBOOKS if sample.key == "dual_dc_10stores_2dc")
        edge = next(sample for sample in SAMPLE_WORKBOOKS if sample.key == "edge_3stores_1dc")
        state: dict = {}
        self.assertTrue(load_and_apply(state, sample_path(dual), dual.filename, "샘플 추천 데이터"))
        state["selected_route_id"] = "R010"
        state["simulation_snapshot"] = "old"
        state["show_all_routes"] = True
        state["rec_filter_product"] = "old"
        self.assertTrue(load_and_apply(state, sample_path(edge), edge.filename, "샘플 추천 데이터"))
        new_route_ids = {str(route["route_id"]) for route in state["varo_recommendations"]}
        self.assertIn(state["selected_route_id"], new_route_ids)
        self.assertNotEqual(state["selected_route_id"], "R010")
        self.assertIsNone(state["simulation_snapshot"])
        self.assertFalse(state["show_all_routes"])
        self.assertNotIn("rec_filter_product", state)
        store_ids = set(state["varo_data"]["stores"]["node_id"].astype(str))
        self.assertNotIn("S10", store_ids)
        self.assertEqual(len([node for node in store_ids if node.startswith("S")]), 3)


try:
    from streamlit.testing.v1 import AppTest
except Exception:  # pragma: no cover
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit AppTest unavailable")
class SimulationSamplePageTests(unittest.TestCase):
    def test_dual_dc_home_renders_two_centers_ten_stores_and_one_default_vehicle(self):
        dual = next(sample for sample in SAMPLE_WORKBOOKS if sample.key == "dual_dc_10stores_2dc")
        state: dict = {}
        self.assertTrue(load_and_apply(state, sample_path(dual), dual.filename, "샘플 추천 데이터"))
        app_path = str(Path(__file__).resolve().parents[1] / "app_v2.py")
        app = AppTest.from_file(app_path, default_timeout=120)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = state.get(key)
        app.session_state["current_menu"] = "홈"
        app.run()
        self.assertFalse(app.exception)
        blob = " ".join(element.value for element in app.markdown)
        self.assertEqual(blob.count('class="network-node dc-node"'), 2)
        self.assertEqual(blob.count('class="network-node store-node'), 10)
        self.assertEqual(blob.count('class="v2-vehicle'), 1)
        self.assertFalse(app.session_state["show_all_routes"])

    def test_vehicle_marker_is_a_truck_icon_not_a_dot(self):
        from pages.overview import _truck_icon
        markup = _truck_icon("#1f766d", "#2d6fa8", "#e7f1fb")
        # a truck has a cab + cargo body (rects) and wheels (circles), never a single dot
        self.assertGreaterEqual(markup.count("<rect"), 2)
        self.assertGreaterEqual(markup.count("<circle"), 2)

    def _render_dual_dc_home(self, **session_overrides):
        dual = next(sample for sample in SAMPLE_WORKBOOKS if sample.key == "dual_dc_10stores_2dc")
        state: dict = {}
        self.assertTrue(load_and_apply(state, sample_path(dual), dual.filename, "샘플 추천 데이터"))
        app_path = str(Path(__file__).resolve().parents[1] / "app_v2.py")
        app = AppTest.from_file(app_path, default_timeout=120)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = state.get(key)
        app.session_state["current_menu"] = "홈"
        for key, value in session_overrides.items():
            app.session_state[key] = value
        app.run()
        self.assertFalse(app.exception)
        return app, " ".join(element.value for element in app.markdown)

    def test_dual_dc_home_marks_via_dc_dashed_and_direct_solid(self):
        # one DIRECT + one VIA_DC route must render one solid path and one dashed path
        curated = [
            {"route_id": "RT_DIRECT", "product_name": "직접상품", "source_id": "S01",
             "source_name": "S01", "target_id": "S02", "target_name": "S02",
             "route_type": "DIRECT", "recommended_qty": 12, "expected_saving": 1000,
             "dqn_action": "미연결"},
            {"route_id": "RT_VIA", "product_name": "경유상품", "source_id": "S01",
             "source_name": "S01", "target_id": "S03", "target_name": "S03",
             "route_type": "VIA_DC", "dc_id": "DC01", "dc_name": "DC01",
             "recommended_qty": 9, "expected_saving": 2000, "dqn_action": "미연결"},
        ]
        _, blob = self._render_dual_dc_home(varo_recommendations=curated, home_sim_display_mode="상위 3개")
        self.assertEqual(blob.count('<path id="rp'), 2)
        self.assertEqual(blob.count('stroke-dasharray="11 8"'), 1)  # only the VIA_DC route is dashed

    def test_full_route_toggle_is_off_by_default_and_caps_background_when_on(self):
        from pages.overview import _MAX_BACKGROUND_ROUTES
        _, off_blob = self._render_dual_dc_home()
        self.assertEqual(off_blob.count('stroke-opacity="0.10"'), 0)
        _, on_blob = self._render_dual_dc_home(show_all_routes=True)
        background = on_blob.count('stroke-opacity="0.10"')
        self.assertGreater(background, 0)
        # 130-route workbook stays capped instead of drawing every pair
        self.assertLessEqual(background, 2 * _MAX_BACKGROUND_ROUTES)

    def test_top3_paths_use_curves_wide_lanes_and_separate_vehicle_phases(self):
        _, parked_blob = self._render_dual_dc_home(home_sim_display_mode="상위 3개")
        for lane in ("-54", "0", "54"):
            self.assertIn(f'data-lane="{lane}"', parked_blob)
        self.assertGreaterEqual(parked_blob.count(" Q "), 4)
        parked = re.findall(r'class="v2-vehicle [^"]+" transform="translate\(([^)]+)\)', parked_blob)
        self.assertEqual(len(parked), 3)
        self.assertEqual(len(set(parked)), 3)
        self.assertGreater(parked_blob.find('class="v2-vehicle '), parked_blob.find('class="network-node dc-node"'))

        _, playing_blob = self._render_dual_dc_home(home_sim_playing=True, home_sim_display_mode="상위 3개")
        phases = re.findall(r'<animateMotion[^>]+begin="([^"]+)"', playing_blob)
        self.assertLessEqual(len(phases), 3)
        self.assertEqual(len(set(phases)), len(phases))
        self.assertEqual(playing_blob.count('class="v2-vehicle'), 3)

    def test_data_management_dqn_selector_loads_selected_sample(self):
        from services.dqn_samples import dqn_sample_options

        app_path = str(Path(__file__).resolve().parents[1] / "app_v2.py")
        app = AppTest.from_file(app_path, default_timeout=120)
        app.run()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        selector = next(item for item in app.selectbox if item.key == "dqn_sample_select")
        selected = dqn_sample_options()["DQN 샘플 10"]
        selector.select("DQN 샘플 10").run()
        button = next(item for item in app.button if item.key == "load_dqn_sample")
        button.click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["uploaded_filename"], selected.workbook.filename)
        self.assertEqual(app.session_state["current_menu"], "데이터 관리")
        self.assertEqual(app.session_state["dqn_sample_training_mode"], selected.mode)
        self.assertEqual(app.session_state["varo_validation"].summary["dc_count"], 2)


if __name__ == "__main__":
    unittest.main()
