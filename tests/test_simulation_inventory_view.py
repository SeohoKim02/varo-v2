"""Home inventory-card, toolbar, cache, and product-language contracts."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

from streamlit.testing.v1 import AppTest

from pages import overview
from services.analysis_pipeline import run_analysis_pipeline
from services.app_state import CANONICAL_DATA_KEYS, build_applied_state_payload
from services.data_loader import SAMPLE_FILENAME, get_default_sample_path, load_excel_data
from services.data_validator import validate_workbook_data
from services.dqn_samples import DQN_SAMPLES, dqn_sample_path
from services.inventory_transition_service import run_inventory_scenario
from services.recommendation_adapter import (
    ALGORITHM_RESULT_FIELDS,
    algorithm_comparison_rows,
)
from simulation.dynamic_network import build_network_nodes, compute_dynamic_layout


APP_PATH = str(Path(__file__).resolve().parents[1] / "app_v2.py")


class SimulationInventoryViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_excel_data(get_default_sample_path())
        validation = validate_workbook_data(data)
        pipeline = run_analysis_pipeline(data).to_dict()
        cls.payload = build_applied_state_payload(
            data, validation, pipeline["recommendations"], SAMPLE_FILENAME,
            "샘플 추천 데이터", pipeline, data_signature="inventory-view-fixture",
        )

    def _app(self) -> AppTest:
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["current_menu"] = "홈"
        app.run()
        self.assertFalse(app.exception)
        return app

    @staticmethod
    def _blob(app: AppTest) -> str:
        return " ".join(item.value for item in app.markdown)

    def test_home_inventory_cards_show_badges_stock_bars_steps_and_kpis(self):
        app = self._app()
        blob = self._blob(app)
        self.assertEqual(
            next(item for item in app.selectbox if item.key == "home_inventory_view_select").value,
            "전후 비교",
        )
        self.assertIn('class="inventory-status-badge"', blob)
        self.assertIn('class="inventory-stock-bar"', blob)
        self.assertIn('class="v2-sim-steps"', blob)
        self.assertIn("출고 준비", blob)
        self.assertIn("재고 반영 완료", blob)
        self.assertIn("이동 수량", {metric.label for metric in app.metric})
        self.assertNotIn("TOP" + "1", blob)

    def test_inventory_view_and_top3_controls_render_without_reanalysis(self):
        app = self._app()
        with (
            patch("services.analysis_pipeline.run_analysis_pipeline") as pipeline,
            patch("services.data_application.load_excel_data") as loader,
        ):
            next(item for item in app.selectbox if item.key == "home_inventory_view_select").set_value("이동 후").run()
            next(item for item in app.selectbox if item.key == "home_sim_display_select").set_value("상위 3개").run()
            next(item for item in app.button if item.key == "sim_restart").click().run()
        pipeline.assert_not_called()
        loader.assert_not_called()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["home_sim_inventory_view"], "이동 후")
        self.assertEqual(app.session_state["home_sim_display_mode"], "상위 3개")
        self.assertEqual(self._blob(app).count('class="v2-vehicle'), 3)

    def test_inventory_view_change_reuses_static_layout(self):
        recommendations = list(self.payload["varo_recommendations"][:1])
        data = self.payload["varo_data"]
        scenario = run_inventory_scenario(data, recommendations)
        base_nodes = overview.build_network_nodes(data, recommendations)
        before_nodes = overview._decorate_nodes(base_nodes, scenario, recommendations, "이동 전")
        after_nodes = overview._decorate_nodes(base_nodes, scenario, recommendations, "이동 후")
        overview._layout_cached.clear()
        overview._network_markup_cached.clear()
        original = overview.compute_dynamic_layout
        selected = str(recommendations[0]["route_id"])
        with patch.object(overview, "compute_dynamic_layout", wraps=original) as layout:
            overview._network_markup_cached("sig", before_nodes, recommendations, [], False, 14.0, False, selected, scenario, "이동 전", "단일 경로")
            overview._network_markup_cached("sig", after_nodes, recommendations, [], False, 14.0, False, selected, scenario, "이동 후", "단일 경로")
        self.assertEqual(layout.call_count, 1)

    def test_algorithm_results_share_complete_projection_without_reordering(self):
        recommendations = list(self.payload["varo_recommendations"][:2])
        route_order = [item["route_id"] for item in recommendations]
        rows = algorithm_comparison_rows(recommendations, "sig")
        self.assertTrue(rows)
        self.assertTrue(all(tuple(row) == ALGORITHM_RESULT_FIELDS for row in rows))
        vhs_routes = [row["route_id"] for row in rows if row["algorithm_name"] == "VHS"]
        self.assertEqual(vhs_routes, route_order)

    def test_sample_10_network_cards_do_not_overlap(self):
        data = load_excel_data(dqn_sample_path(DQN_SAMPLES[-1]))
        recommendations = run_analysis_pipeline(data).to_dict()["recommendations"]
        layout = compute_dynamic_layout(build_network_nodes(data, recommendations), recommendations)
        cards = [*layout.dcs, *layout.stores]
        overlaps = []
        for index, left in enumerate(cards):
            for right in cards[index + 1:]:
                horizontal = abs(float(left["x"]) - float(right["x"]))
                vertical = abs(float(left["y"]) - float(right["y"]))
                if (
                    horizontal < (float(left["width"]) + float(right["width"])) / 2
                    and vertical < (float(left["height"]) + float(right["height"])) / 2
                ):
                    overlaps.append((left["node_id"], right["node_id"]))
        self.assertFalse(overlaps, overlaps)

    def test_runtime_ui_and_product_docs_have_no_deprecated_product_positioning(self):
        root = Path(__file__).resolve().parents[1]
        sources = [
            root / "README_V2.md", root / "APP_SUMMARY.md", root / "DEPLOY_CHECKLIST.md",
            root / "app_v2.py", root / "router.py", *sorted((root / "pages").glob("*.py")),
            *sorted((root / "components").glob("*.py")), *sorted((root / "services").glob("*.py")),
        ]
        text = "\n".join(path.read_text(encoding="utf-8") for path in sources).casefold()
        forbidden = (
            "제" + "출용", "공모" + "전", "심사" + "용", "심사" + "위원",
            "교수" + "님", "발표" + "용", "대회" + "용", "논문" + "급",
            "state" + " of the art", "완전한 " + "최적화",
        )
        for phrase in forbidden:
            self.assertNotIn(phrase.casefold(), text)

    def test_home_keeps_minimum_product_structure(self):
        app = self._app()
        blob = self._blob(app)
        for removed in ("추천 Top 5", "현재 실행 경로 Top 3", "상세 비교표", "분석 결과 다운로드"):
            self.assertNotIn(removed, blob)
        self.assertFalse(app.dataframe)


if __name__ == "__main__":
    unittest.main()
