"""Home simulation speed and performance guardrails."""
from __future__ import annotations

import unittest
from pathlib import Path

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

from pages.overview import (
    _MAX_BACKGROUND_ROUTES,
    _ROUTE_LANES,
    _route_path_d,
    _route_path_points,
    animation_duration_seconds,
)


class OverviewSpeedContractTests(unittest.TestCase):
    def test_speed_options_map_to_smil_durations(self):
        self.assertEqual(animation_duration_seconds("느림"), 24.0)
        self.assertEqual(animation_duration_seconds("보통"), 14.0)
        self.assertEqual(animation_duration_seconds("빠름"), 8.0)
        self.assertEqual(animation_duration_seconds("unknown"), 14.0)

    def test_paths_start_at_node_edges_and_use_distinct_lanes(self):
        segments = [{"from_node_id": "A", "to_node_id": "B", "phase": "DIRECT"}]
        positions = {"A": (0.0, 0.0), "B": (300.0, 0.0)}
        dimensions = {"A": (100.0, 60.0), "B": (100.0, 60.0)}
        center_lane = _route_path_points(segments, positions, dimensions, 0.0)
        side_lane = _route_path_points(segments, positions, dimensions, 18.0)
        self.assertNotEqual(center_lane[0], positions["A"])
        self.assertNotEqual(center_lane[-1], positions["B"])
        self.assertEqual(center_lane[0][1], 0.0)
        self.assertEqual(side_lane[0][1], 18.0)

    def test_top5_lanes_are_wide_and_paths_are_curved(self):
        self.assertEqual(_ROUTE_LANES, (-54.0, 0.0, 54.0, -27.0, 27.0))
        ordered_lanes = sorted(_ROUTE_LANES)
        self.assertGreaterEqual(min(b - a for a, b in zip(ordered_lanes, ordered_lanes[1:])), 24.0)
        self.assertEqual(_ROUTE_LANES[:3], (-54.0, 0.0, 54.0))
        path = _route_path_d([(100.0, 100.0), (600.0, 300.0), (1000.0, 500.0)], 20.0)
        self.assertEqual(path.count(" Q "), 2)

    def test_home_uses_smil_duration_without_python_loop(self):
        source = (Path(__file__).resolve().parents[1] / "pages" / "overview.py").read_text(encoding="utf-8")
        self.assertIn('animateMotion dur="{speed_seconds:.1f}s"', source)
        self.assertIn('begin="-{phase:.1f}s"', source)
        self.assertNotIn("time.sleep", source)
        self.assertNotIn("while True", source)

    def test_network_svg_markup_is_cached_by_data_signature(self):
        source = (Path(__file__).resolve().parents[1] / "pages" / "overview.py").read_text(encoding="utf-8")
        self.assertIn("def _network_markup_cached", source)
        self.assertIn("@st.cache_data(show_spinner=False, max_entries=24)", source)
        self.assertIn("_data_signature()", source)

    def test_full_route_background_is_capped_for_home_speed(self):
        self.assertEqual(_MAX_BACKGROUND_ROUTES, 10)

    def test_vehicle_icon_exposes_transport_mode_not_a_plain_dot(self):
        from pages.overview import _truck_icon

        markup = _truck_icon("#1f766d", "#2d6fa8", "#e7f1fb", "냉장", "TOP1")
        self.assertGreaterEqual(markup.count("<rect"), 4)
        self.assertGreaterEqual(markup.count("<circle"), 2)
        self.assertIn("냉장", markup)
        self.assertIn("TOP1", markup)


if __name__ == "__main__":
    unittest.main()
