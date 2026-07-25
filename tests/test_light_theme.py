"""Guards for the light-theme UI: the app must never fall back to dark surfaces."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class LightThemeTests(unittest.TestCase):
    def test_config_forces_light_base(self):
        config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        self.assertIn('base = "light"', config)
        self.assertRegex(config, r'backgroundColor\s*=\s*"#[Ff]')
        self.assertIn("secondaryBackgroundColor", config)
        self.assertIn("textColor", config)

    def test_design_tokens_backgrounds_are_light_text_is_dark(self):
        from styles import DESIGN_TOKENS

        for key in ("app_bg", "card_bg", "panel_soft"):
            self.assertGreater(_luminance(DESIGN_TOKENS[key]), 200, f"{key} must be a light surface")
        for key in ("text", "strong_text", "muted_text"):
            self.assertLess(_luminance(DESIGN_TOKENS[key]), 130, f"{key} must be a dark, readable text color")
        # Selection/highlight is a very light blue.
        self.assertGreater(_luminance(DESIGN_TOKENS["accent_soft"]), 220)

    def test_styles_have_no_black_or_near_black_surfaces(self):
        styles = (ROOT / "styles.py").read_text(encoding="utf-8")
        for banned in ("background: #000", "background:#000", "background-color: #000", "background-color:#000"):
            self.assertNotIn(banned, styles)
        # The old strong-coral primary button color must be gone.
        self.assertNotIn("#ec9a91", styles)

    def test_primary_button_uses_soft_accent(self):
        styles = (ROOT / "styles.py").read_text(encoding="utf-8")
        # Primary (active menu / main action) styled from the accent-soft token.
        self.assertIn("--varo-accent-soft", styles)
        self.assertIn('button[data-testid="stBaseButton-primary"]', styles)


if __name__ == "__main__":
    unittest.main()
