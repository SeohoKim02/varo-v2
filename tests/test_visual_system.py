"""Guards for the shared visual system: one type scale, one radius set, one
currency format, and numeric table columns that line up.

These are structural checks, never pixel comparisons: they assert that the app
keeps expressing sizes through the design tokens instead of drifting back to
per-screen values. The measurements that motivated each rule were taken in a
real headless Chrome at 1920/1600/1366.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from components.tables import format_currency, format_number, render_html_table
from services.workspace_view import money_text, qty_text
from styles import DESIGN_TOKENS, SPACING, TYPE_SCALE

ROOT = Path(__file__).resolve().parents[1]
STYLES = (ROOT / "styles.py").read_text(encoding="utf-8")
# Only the text inside the emitted <style> block reaches the browser.
CSS = STYLES[STYLES.index("<style>"):]


class TypeScaleTests(unittest.TestCase):
    def test_every_scale_role_is_exposed_as_a_css_variable(self):
        for role in ("page", "section", "card", "metric"):
            self.assertIn(f"--varo-fs-{role}:", CSS)
            self.assertIn(f"--varo-fw-{role}:", CSS)
        for role in ("body", "secondary", "caption"):
            self.assertIn(f"--varo-fs-{role}:", CSS)
        for name in ("section-gap", "block-gap", "card-padding"):
            self.assertIn(f"--varo-{name}:", CSS)

    def test_scale_steps_descend_and_never_drop_body_text_below_13px(self):
        def px(value: str) -> float:
            return float(value.removesuffix("rem")) * 16

        page, section, card = (px(TYPE_SCALE[k]) for k in ("page_size", "section_size", "card_size"))
        body, secondary, caption = (px(TYPE_SCALE[k]) for k in ("body_size", "secondary_size", "caption_size"))
        self.assertGreater(page, section)
        self.assertGreater(section, card)
        self.assertGreater(card, secondary)
        self.assertGreater(secondary, caption)
        # A desktop product must stay readable: body copy never shrinks to fit.
        self.assertGreaterEqual(body, 14)
        self.assertGreaterEqual(secondary, 13)
        self.assertGreaterEqual(caption, 12)

    def test_the_two_page_titles_share_one_level(self):
        """Both page headings resolve to the page size, not two different ones.

        The class-only rule lost to Streamlit's own markdown h1, which painted
        three screens at 44px while the fourth sat at 18px; the element-qualified
        selector is what makes the shared token actually win.
        """
        self.assertIn("h1.v2-page-title", CSS)
        for block in (".v2-page-title,", ".ws-header-title {"):
            self.assertIn(block, CSS)
        page_rules = re.findall(r"\.(?:v2-page-title|ws-header-title)[^{]*\{[^}]*\}", CSS)
        self.assertTrue(page_rules)
        for rule in page_rules:
            if "font-size" in rule:
                self.assertIn("var(--varo-fs-page)", rule)

    def test_metric_widgets_are_pulled_onto_the_product_kpi_scale(self):
        """st.metric defaults to 36px/400, a second KPI language beside .ws-kpi."""
        metric_rule = re.search(r'\[data-testid="stMetricValue"\]\s*\{[^}]*\}', CSS)
        self.assertIsNotNone(metric_rule)
        self.assertIn("var(--varo-fs-metric)", metric_rule.group())
        self.assertIn("var(--varo-fw-metric)", metric_rule.group())


class SurfaceTests(unittest.TestCase):
    def test_corner_radii_collapse_to_two_sizes_plus_the_badge_pill(self):
        declared = set(re.findall(r"border-radius:\s*([^;!]+)", CSS))
        allowed_tokens = {"var(--varo-radius-card)", "var(--varo-radius-button)", "999px", "50%", "inherit"}
        for value in declared:
            value = value.strip()
            if value in allowed_tokens:
                continue
            # Compound values (e.g. top corners only) must still be built from tokens.
            parts = set(value.split())
            self.assertTrue(
                parts <= allowed_tokens | {"0"},
                f"unexpected radius literal: {value!r}",
            )

    def test_only_badges_use_the_pill_radius(self):
        pill_rules = [
            rule for rule in re.findall(r"[^{}]+\{[^}]*\}", CSS)
            if re.search(r"border-radius:\s*999px", rule)
        ]
        for rule in pill_rules:
            selector = rule.split("{", 1)[0]
            self.assertTrue(
                any(hint in selector for hint in ("badge", "pill", "file-label", "hbar", "legend-dot")),
                f"999px radius outside a badge-like element: {selector.strip()!r}",
            )

    def test_cards_carry_no_drop_shadow(self):
        """Hierarchy comes from border + spacing; shadowing every card read as a template."""
        card_rule = re.search(r"\.v2-card\s*\{[^}]*\}", CSS)
        self.assertIsNotNone(card_rule)
        self.assertIn("box-shadow: none", card_rule.group())
        # The one remaining shadow token is a hairline, not an elevation.
        self.assertRegex(DESIGN_TOKENS["shadow"], r"^0 1px 2px ")

    def test_spacing_tokens_are_used_rather_than_re_typed(self):
        self.assertIn("var(--varo-section-gap)", CSS)
        self.assertIn("var(--varo-card-padding)", CSS)
        self.assertTrue(SPACING["card_padding"])


class NumberFormattingTests(unittest.TestCase):
    def test_currency_is_one_format_everywhere(self):
        for value in (0, 19119, 256440.570443, -4821.6):
            self.assertEqual(money_text(value), format_currency(value))
            self.assertRegex(money_text(value), r"^-?[\d,]+원$")

    def test_a_long_float_never_reaches_the_screen_as_itself(self):
        self.assertEqual(money_text(256440.570443), "256,441원")
        self.assertEqual(qty_text(57.6), "58개")
        self.assertNotIn(".", money_text(256440.570443))

    def test_missing_values_read_as_words_not_zero(self):
        for missing in (None, "", float("nan"), float("inf")):
            self.assertNotIn("0원", money_text(missing))
            self.assertEqual(format_currency(missing), "-")

    def test_quantities_and_counts_keep_their_units(self):
        self.assertEqual(qty_text(431), "431개")
        self.assertEqual(qty_text(8, "건"), "8건")
        self.assertEqual(format_number(8, "건"), "8건")


class NumericColumnTests(unittest.TestCase):
    """Numbers right-align so a column can be compared on its last digit."""

    def setUp(self):
        self.captured: list[str] = []

    def _render(self, rows, columns=None):
        import components.tables as tables

        original = tables.st.markdown
        tables.st.markdown = lambda body, **kwargs: self.captured.append(body)
        try:
            render_html_table(rows, columns)
        finally:
            tables.st.markdown = original
        return self.captured[-1]

    def test_numeric_columns_are_tagged_and_text_columns_are_not(self):
        html = self._render([
            {"상품": "냉동만두500g", "수량": "58개", "예상 순효과": "57,699원"},
            {"상품": "서울우유200ml", "수량": "57개", "예상 순효과": "41,498원"},
        ])
        self.assertIn('<th class="v2-num">수량</th>', html)
        self.assertIn('<th class="v2-num">예상 순효과</th>', html)
        self.assertIn("<th>상품</th>", html)
        self.assertIn('<td class="v2-num">58개</td>', html)
        self.assertIn("<td>냉동만두500g</td>", html)

    def test_a_column_holding_a_missing_word_stays_left_aligned(self):
        html = self._render([
            {"항목": "차량 용량", "값": "미확보"},
            {"항목": "이동 거리", "값": "12.4km"},
        ])
        # 값 mixes a number and 미확보, so it is not a numeric column.
        self.assertIn("<th>값</th>", html)
        self.assertNotIn('class="v2-num"', html)

    def test_a_dash_only_gap_does_not_break_a_numeric_column(self):
        html = self._render([
            {"순위": "1", "현재 VHS": "89.3"},
            {"순위": "2", "현재 VHS": "-"},
        ])
        self.assertIn('<th class="v2-num">현재 VHS</th>', html)
        self.assertIn("<td class=\"v2-num\">-</td>", html)

    def test_the_alignment_class_is_styled(self):
        self.assertIn(".v2-html-table td.v2-num", CSS)
        self.assertIn("text-align: right", CSS)

    def test_the_home_top_grid_aligns_its_numbers_the_same_way(self):
        """The one table that is a grid, not an in-DOM table, still lines up.

        ``st.dataframe`` draws on a canvas and left-aligns text, so 수량 and
        예상 순효과 are right-aligned through the grid's own column option — the
        printed value is untouched.
        """
        from components.tables import (
            HOME_TOP_NUMERIC_COLUMNS,
            build_home_top_rows,
            home_top_column_config,
        )

        rows = build_home_top_rows([{
            "route_id": "R1", "product_name": "냉동만두500g",
            "source_name": "연신내점", "target_name": "홍제점",
            "route_type": "DIRECT", "planned_qty": 58, "net_benefit": 57699.4,
        }])
        # The cell text is still exactly what every other table in the app prints.
        self.assertEqual(rows[0]["수량"], "58개")
        self.assertEqual(rows[0]["예상 순효과"], "57,699원")
        config = home_top_column_config()
        self.assertEqual(set(config), set(HOME_TOP_NUMERIC_COLUMNS))
        for column in HOME_TOP_NUMERIC_COLUMNS:
            self.assertEqual(config[column]["alignment"], "right")
        # Text columns are never aligned right by accident.
        self.assertNotIn("상품", config)
        self.assertNotIn("출발", config)

    def test_a_missing_number_still_reads_as_a_dash_not_a_blank(self):
        from components.tables import build_home_top_rows

        rows = build_home_top_rows([{
            "route_id": "R1", "product_name": "상품", "route_type": "DIRECT",
            "planned_qty": None, "recommended_qty": None, "net_benefit": None,
        }])
        self.assertEqual(rows[0]["수량"], "-")
        self.assertEqual(rows[0]["예상 순효과"], "-")


class StatusColourTests(unittest.TestCase):
    """Status colour follows meaning: warning is for a problem, not for a to-do.

    On a first run both header chips are pending at once. Painting both amber
    made an app with nothing wrong look like an app with two faults, so the two
    'not done yet' states are neutral/accent and warning stays reserved.
    """

    def _badges(self, **state):
        from components.status import app_status_badges

        base = {
            "varo_data": None, "varo_recommendations": [],
            "analysis_result": {}, "varo_pipeline_result": {},
        }
        return app_status_badges({**base, **state})

    def test_an_empty_app_shows_no_warning_at_all(self):
        variants = [variant for _label, variant in self._badges()]
        self.assertNotIn("warning", variants)
        self.assertNotIn("error", variants)
        self.assertEqual(variants, ["neutral", "neutral"])

    def test_applied_data_makes_the_run_the_accent_notice_not_a_warning(self):
        badges = self._badges(varo_data={"stores": [{"node_id": "S1"}]})
        labels = dict(badges)
        self.assertEqual(labels.get("데이터 적용 완료"), "success")
        self.assertEqual(labels.get("분석 실행 필요"), "accent")
        self.assertNotIn("warning", [variant for _label, variant in badges])

    def test_the_pending_data_chip_is_still_shown(self):
        """Recoloured, never hidden — the user still learns the step is open."""
        labels = [label for label, _variant in self._badges()]
        self.assertEqual(len(labels), 2)
        self.assertIn("분석 실행 필요", labels)
        self.assertTrue(any("데이터" in label for label in labels))

    def test_warning_and_error_pills_still_exist_for_real_problems(self):
        self.assertIn(".v2-badge-warning", CSS)
        self.assertIn(".v2-badge-error", CSS)


class StyleHygieneTests(unittest.TestCase):
    def test_emitted_css_comments_carry_no_user_facing_wording(self):
        """Comments ship inside the <style> block that AppTest reads as markdown.

        Product phrases written there show up in text assertions as if the screen
        had rendered them, so the wording stays out of the stylesheet.
        """
        banned = ("예상 순효과", "오늘 권장 이동", "실행 수량", "총 이동 수량",
                  "오늘 실행 이동", "데이터 없음", "현재 사용 중 데이터")
        for comment in re.findall(r"/\*.*?\*/", CSS, re.S):
            for phrase in banned:
                self.assertNotIn(phrase, comment, f"{phrase!r} must not appear in a CSS comment")

    def test_the_narrow_desktop_sidebar_rule_stays_scoped_to_the_open_state(self):
        """An unscoped width would make the collapsed sidebar occupy space too."""
        for rule in re.findall(r"section\[data-testid=\"stSidebar\"\][^{]*\{[^}]*\}", CSS):
            if "width" in rule:
                self.assertIn('[aria-expanded="true"]', rule)


if __name__ == "__main__":
    unittest.main()
