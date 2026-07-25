"""Home simulation speed and performance guardrails."""
from __future__ import annotations

import unittest
from pathlib import Path

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

from pages.overview import _MAX_BACKGROUND_ROUTES, animation_duration_seconds


class OverviewSpeedContractTests(unittest.TestCase):
    def test_speed_options_map_to_smil_durations(self):
        # Slower, calmer motion than before; 느림 clearly slow, 빠름 not too fast.
        self.assertEqual(animation_duration_seconds("느림"), 24.0)
        self.assertEqual(animation_duration_seconds("보통"), 15.0)
        self.assertEqual(animation_duration_seconds("빠름"), 9.5)
        self.assertEqual(animation_duration_seconds("unknown"), 15.0)
        # 느림 must stay clearly slower than 빠름.
        self.assertGreater(animation_duration_seconds("느림"), animation_duration_seconds("빠름"))

    def test_home_uses_smil_duration_without_python_loop(self):
        source = (Path(__file__).resolve().parents[1] / "pages" / "overview.py").read_text(encoding="utf-8")
        self.assertIn('animateMotion dur="{speed_seconds:.1f}s"', source)
        self.assertNotIn("time.sleep", source)
        self.assertNotIn("while True", source)

    def test_network_svg_markup_is_cached_by_data_signature(self):
        source = (Path(__file__).resolve().parents[1] / "pages" / "overview.py").read_text(encoding="utf-8")
        self.assertIn("def _network_markup_cached", source)
        self.assertIn("@st.cache_data(show_spinner=False, max_entries=24)", source)
        self.assertIn("_data_signature()", source)

    def test_full_route_background_is_capped_for_home_speed(self):
        self.assertEqual(_MAX_BACKGROUND_ROUTES, 12)

    def test_vehicle_icon_exposes_transport_mode_not_a_plain_dot(self):
        from pages.overview import _truck_icon

        markup = _truck_icon("#1f766d", "#2d6fa8", "#e7f1fb", "냉장", "TOP1")
        self.assertGreaterEqual(markup.count("<rect"), 4)
        self.assertGreaterEqual(markup.count("<circle"), 2)
        self.assertIn("냉장", markup)
        self.assertIn("TOP1", markup)


if __name__ == "__main__":
    unittest.main()
