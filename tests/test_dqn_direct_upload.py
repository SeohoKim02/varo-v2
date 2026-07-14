"""Read-only regression coverage for the ten real DQN upload workbooks."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest

from services.analysis_pipeline import top_recommendations
from services.app_state import CANONICAL_DATA_KEYS
from services.data_application import load_and_apply
from services.dqn_samples import discover_dqn_samples_dir
from simulation.dynamic_network import build_network_nodes, compute_dynamic_layout
from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()


class DqnDirectUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = discover_dqn_samples_dir()
        if cls.root is None:
            raise AssertionError("DQN 샘플 01~10 폴더를 찾지 못했습니다.")
        cls.paths = sorted(cls.root.glob("*.xlsx"))
        cls.paths = [path for path in cls.paths if "Varo_DQN_sample_" in path.name]
        if len(cls.paths) != 10:
            raise AssertionError(f"DQN 원본 샘플 수가 10개가 아닙니다: {len(cls.paths)}")
        cls.metadata = {
            path: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in cls.paths
        }

    @staticmethod
    def _upload(path: Path, state: dict | None = None) -> dict:
        state = state if state is not None else {}
        stream = BytesIO(path.read_bytes())
        stream.seek(0, 2)  # emulate an UploadedFile whose pointer was consumed
        ok = load_and_apply(state, stream, path.name, "업로드된 추천 결과")
        if not ok:
            validation = state.get("pending_varo_validation")
            detail = [message.to_dict() for message in getattr(validation, "messages", [])]
            raise AssertionError((path.name, state.get("pending_load_error"), state.get("pending_validation_error"), detail))
        return state

    def test_all_ten_workbooks_pass_the_direct_upload_path(self):
        for path in self.paths:
            with self.subTest(path=path.name):
                state = self._upload(path)
                self.assertEqual(state["uploaded_filename"], path.name)
                self.assertEqual(state["data_source_type"], "업로드된 추천 결과")
                self.assertEqual(state["data_apply_message"], "데이터 적용 완료")
                recommendations = state["varo_recommendations"]
                self.assertTrue(recommendations)
                self.assertEqual(state["varo_pipeline_result"]["status"], "success")
                self.assertEqual(state["varo_pipeline_result"]["diagnostics"]["algorithm_errors"], [])
                self.assertTrue(all(row.get("pareto_rank") is not None for row in recommendations))
                self.assertEqual(
                    len(top_recommendations(recommendations, limit=5)),
                    min(5, len(recommendations)),
                )
                self.assertIn(state["selected_route_id"], {
                    str(row["route_id"]) for row in state["varo_recommendations"]
                })
                self.assertIsNone(state["dqn_training_result"])
                self.assertEqual(
                    self.metadata[path],
                    (path.stat().st_size, path.stat().st_mtime_ns),
                )

    def test_samples_01_02_and_10_have_expected_store_and_dc_counts(self):
        expected = {"01": (2, 1), "02": (4, 1), "10": (10, 2)}
        for number, counts in expected.items():
            path = next(path for path in self.paths if f"sample_{number}_" in path.name)
            state = self._upload(path)
            summary = state["varo_validation"].summary
            self.assertEqual((summary["store_count"], summary["dc_count"]), counts)
            recommendations = state["varo_recommendations"]
            self.assertEqual(
                len(top_recommendations(recommendations, limit=5)),
                min(5, len(recommendations)),
            )

    def test_replacement_resets_route_filters_simulation_and_dqn_but_keeps_menu(self):
        first = next(path for path in self.paths if "sample_01_" in path.name)
        second = next(path for path in self.paths if "sample_02_" in path.name)
        state = self._upload(first)
        state.update({
            "current_menu": "데이터 관리",
            "selected_route_id": "OLD",
            "rec_filter_product": "OLD",
            "simulation_snapshot": {"old": True},
            "show_all_routes": True,
            "home_sim_playing": True,
            "dqn_training_result": {"status": "정상"},
            "dqn_batch_result": {"old": True},
            "route_detail_select": "OLD",
        })
        self._upload(second, state)
        self.assertEqual(state["current_menu"], "데이터 관리")
        self.assertNotEqual(state["selected_route_id"], "OLD")
        self.assertNotIn("rec_filter_product", state)
        self.assertIsNone(state["simulation_snapshot"])
        self.assertFalse(state["show_all_routes"])
        self.assertFalse(state["home_sim_playing"])
        self.assertIsNone(state["dqn_training_result"])
        self.assertIsNone(state["dqn_batch_result"])
        self.assertNotIn("route_detail_select", state)

    def test_small_and_dual_dc_layouts_are_balanced_and_inside_canvas(self):
        layouts = {}
        for number in ("01", "10"):
            path = next(path for path in self.paths if f"sample_{number}_" in path.name)
            state = self._upload(path)
            recommendations = state["varo_recommendations"]
            nodes = build_network_nodes(state["varo_data"], recommendations)
            layouts[number] = compute_dynamic_layout(nodes, recommendations)

        small = layouts["01"]
        self.assertEqual(len(small.stores), 2)
        left, right = sorted(small.stores, key=lambda node: node["x"])
        self.assertGreater(right["x"] - left["x"], 600)
        self.assertLess(abs(right["y"] - left["y"]), 5)
        self.assertLess(small.dcs[0]["y"], left["y"])

        dual = layouts["10"]
        self.assertEqual(len(dual.dcs), 2)
        first, second = dual.dcs
        self.assertLess(abs(first["y"] - second["y"]), 5)
        self.assertGreater(abs(first["x"] - second["x"]), (first["width"] + second["width"]) / 2)
        width, height = dual.canvas["width"], dual.canvas["height"]
        for node in dual.dcs + dual.stores:
            self.assertGreaterEqual(node["x"] - node["width"] / 2, 0)
            self.assertLessEqual(node["x"] + node["width"] / 2, width)
            self.assertGreaterEqual(node["y"] - node["height"] / 2, 0)
            self.assertLessEqual(node["y"] + node["height"] / 2, height)


try:
    from streamlit.testing.v1 import AppTest
except Exception:  # pragma: no cover
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit AppTest unavailable")
class DqnDirectUploadPageTests(unittest.TestCase):
    HOME_TOP_COLUMNS = ["순위", "상품", "출발", "도착", "경로", "수량", "예상 절감액"]
    MENUS = ["운영 현황", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리"]

    @staticmethod
    def _state_for_sample(number: str) -> tuple[dict, Path]:
        root = discover_dqn_samples_dir()
        path = next(root.glob(f"Varo_DQN_sample_{number}_*.xlsx"))
        state: dict = {}
        if not load_and_apply(state, BytesIO(path.read_bytes()), path.name, "업로드된 추천 결과"):
            raise AssertionError((path.name, state.get("pending_load_error"), state.get("pending_validation_error")))
        return state, path

    @staticmethod
    def _app_with_state(state: dict):
        app_path = str(Path(__file__).resolve().parents[1] / "app_v2.py")
        app = AppTest.from_file(app_path, default_timeout=120)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = state.get(key)
        return app

    def test_sample_02_state_renders_home_kpis_top_routes_and_simulation(self):
        state, _ = self._state_for_sample("02")
        app = self._app_with_state(state)
        app.session_state["current_menu"] = "운영 현황"
        app.run()
        self.assertFalse(app.exception)
        blob = " ".join(element.value for element in app.markdown)
        self.assertIn("추천 후보 수", blob)
        self.assertIn("추천 Top 5", blob)
        self.assertEqual(blob.count('class="network-node dc-node"'), 1)
        self.assertEqual(blob.count('class="network-node store-node'), 4)

    def test_samples_01_02_10_render_data_driven_top5_and_keep_upload_across_pages(self):
        expected_dcs = {"01": 1, "02": 1, "10": 2}
        for number in ("01", "02", "10"):
            with self.subTest(sample=number):
                state, path = self._state_for_sample(number)
                recommendations = state["varo_recommendations"]
                expected_top_count = min(5, len(recommendations))
                data_signature = state["data_signature"]
                route_ids = {str(row["route_id"]) for row in recommendations}

                app = self._app_with_state(state)
                app.session_state["current_menu"] = "운영 현황"
                app.run()
                self.assertFalse(app.exception)
                top_frame = next(
                    item.value for item in app.dataframe
                    if list(item.value.columns) == self.HOME_TOP_COLUMNS
                )
                self.assertEqual(len(top_frame), expected_top_count)
                home_blob = " ".join(element.value for element in app.markdown)
                self.assertEqual(home_blob.count('class="network-node dc-node"'), expected_dcs[number])
                self.assertEqual(home_blob.count('class="v2-vehicle"'), expected_top_count)
                self.assertNotIn("파일 구조를 확인해주세요", home_blob)

                for menu in self.MENUS:
                    button = next(item for item in app.sidebar.button if item.key == f"nav_{menu}")
                    button.click().run()
                    self.assertFalse(app.exception, msg=f"{path.name} / {menu}: {list(app.exception)}")
                    self.assertEqual(app.session_state["uploaded_filename"], path.name)
                    self.assertEqual(app.session_state["data_signature"], data_signature)
                    self.assertIn(app.session_state["selected_route_id"], route_ids)
                    page_blob = " ".join(element.value for element in app.markdown)
                    self.assertNotIn("Traceback", page_blob)
                    if menu == "경로 상세":
                        self.assertIn("이동 단계", page_blob)
                        self.assertTrue("DIRECT" in page_blob or "VIA_DC" in page_blob)

    def test_sample_10_route_detail_keeps_the_selected_dc02(self):
        state, _ = self._state_for_sample("10")
        route = next(
            item for item in state["varo_recommendations"]
            if item.get("route_type") == "VIA_DC" and str(item.get("dc_id")) == "DC02"
        )
        app = self._app_with_state(state)
        app.session_state["selected_route_id"] = str(route["route_id"])
        app.session_state["current_menu"] = "경로 상세"
        app.run()
        self.assertFalse(app.exception)
        blob = " ".join(element.value for element in app.markdown)
        self.assertIn("이동 단계", blob)
        self.assertIn(str(route.get("dc_name") or "DC02"), blob)


if __name__ == "__main__":
    unittest.main()
