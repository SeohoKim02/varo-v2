"""Headless render smoke tests for all five V2 pages via Streamlit AppTest.

Self-contained within varo_v2: the session payload is built from the pure
in-package adapter with no pipeline result, so no legacy/backup module is
imported. The tests assert each page renders without raising (which also proves
the download buttons build their bytes), that selected_route_id is shared across
pages, and that the data-management download stub was replaced.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

try:
    from streamlit.testing.v1 import AppTest

    _APPTEST_AVAILABLE = True
except Exception:  # pragma: no cover - older streamlit
    _APPTEST_AVAILABLE = False

from services.analysis_pipeline import run_analysis_pipeline
from services.app_state import CANONICAL_DATA_KEYS, build_applied_state_payload
from services.data_loader import SAMPLE_FILENAME, get_default_sample_path, load_excel_data
from services.data_validator import validate_workbook_data

APP_PATH = str(Path(__file__).resolve().parents[1] / "app_v2.py")
MENUS = ["홈", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리"]


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class PageRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_excel_data(get_default_sample_path())
        validation = validate_workbook_data(data)
        pipeline = run_analysis_pipeline(data).to_dict()  # self-contained recompute
        cls.payload = build_applied_state_payload(
            data,
            validation,
            pipeline["recommendations"],
            SAMPLE_FILENAME,
            "샘플 추천 데이터",
            pipeline,
            data_signature="render-fixture-signature",
        )

    def _new_app(self):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        return app

    def _markdown_blob(self, app) -> str:
        return " ".join(element.value for element in app.markdown)

    def test_all_pages_render_without_exception_and_keep_r002(self):
        app = self._new_app()
        app.session_state["selected_route_id"] = "R002"
        for menu in MENUS:
            app.session_state["current_menu"] = menu
            app.run()
            self.assertFalse(app.exception, msg=f"{menu} raised: {list(app.exception)}")
            self.assertEqual(app.session_state["selected_route_id"], "R002")

    def test_empty_app_renders_onboarding_without_exception(self):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.run()
        self.assertFalse(app.exception)

    def test_data_management_shows_dqn_samples_and_compact_summary(self):
        app = self._new_app()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("현재 로드 상태", blob)
        self.assertIn("DQN 샘플 10개 목록", {item.label for item in app.expander})
        self.assertTrue(any("원본은 수정하지 않고" in item.value for item in app.caption))
        self.assertNotIn("원본 데이터 보기", {item.label for item in app.expander})

    def test_route_detail_has_self_contained_route_steps(self):
        app = self._new_app()
        app.session_state["current_menu"] = "경로 상세"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        for required in ("상품", "출발 점포", "도착 점포", "경로 유형", "추천 수량", "예상 절감액", "이동 비용", "이동 거리", "예상 시간", "이동 방식"):
            self.assertIn(required, blob)
        self.assertIn("이동 단계", blob)
        self.assertTrue("DIRECT" in blob or "VIA_DC" in blob)
        for banned in ("route_id", "fallback", "calculation_function"):
            self.assertNotIn(banned, blob)

    def test_home_is_result_dashboard(self):
        app = self._new_app()
        app.session_state["current_menu"] = "홈"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        # required result-dashboard elements
        for required in (
            "Varo 운영 결과",
            "현재 데이터", "점포 / DC", "추천 후보 수", "예상 절감액", "평균 VHS",
            "업로드된 재고 데이터를 바탕으로 악성재고 후보, 이동 추천, 검증 결과를 요약합니다.",
            "다음에 볼 화면", "재고 이동 네트워크 미리보기",
            "파란 실선 · 직접 이동", "노란 점선 · DC 경유",
        ):
            self.assertIn(required, blob, f"home must contain: {required}")
        # The sidebar and home body both expose clear page navigation.
        sidebar_nav = {b.label: b.key for b in app.sidebar.button}
        self.assertEqual(set(sidebar_nav), set(MENUS))
        for menu in MENUS:
            self.assertEqual(sidebar_nav[menu], f"nav_{menu}")
        button_labels = {b.label for b in app.button}
        for quick_button in ("추천 실행 보기", "경로 상세 보기", "분석 및 검증 보기", "데이터 관리 보기"):
            self.assertIn(quick_button, button_labels, f"home should have button: {quick_button}")
        # the top toolbar keeps only the data-replace toggle (no duplicate 데이터 관리 button)
        self.assertIn("데이터 교체", button_labels)
        self.assertIn("시뮬레이션 실행", button_labels)
        self.assertIn("다시 실행", button_labels)
        # forbidden elements / developer copy
        for banned in (
            "파일 ·", "상태 ·",
            "실제 V2 내부 알고리즘 재계산 결과 기준",
            "DQN 과거 학습 결과는 제외",
            "varo_hybrid_score",
            "KPI 기준",
            "중립값",
            "분석 결과 다운로드",
            "원본 데이터 보기",
            "Quality_Check",
            "DQN 학습 이력",
            "운영 로그",
            "선택 후보 요약",
        ):
            self.assertNotIn(banned, blob, f"home should not contain: {banned}")
        # no download buttons on home
        self.assertNotIn("검증 리포트 Excel", button_labels)
        self.assertNotIn("추천 결과 CSV", button_labels)
        for removed_section in ("현재 실행 경로 Top 3", "추천 Top 5", "선택 경로 요약"):
            self.assertNotIn(removed_section, blob)
        self.assertFalse(app.dataframe, "home must not contain recommendation or validation tables")
        self.assertIn('class="network-node dc-node"', blob)
        self.assertIn('class="network-node store-node', blob)
        self.assertEqual(blob.count('class="v2-vehicle'), 3)
        self.assertNotIn('v2-running-route', blob)
        # 전체 경로 보기 defaults OFF (representative Top 3 only)
        self.assertFalse(app.session_state["show_all_routes"])
        self.assertEqual(app.session_state["simulation_speed"], "보통")

    def test_sidebar_nav_navigates_to_every_page(self):
        for menu in MENUS:
            app = self._new_app()
            app.session_state["current_menu"] = "홈"
            app.run()
            button = next(item for item in app.sidebar.button if item.key == f"nav_{menu}")
            button.click().run()
            self.assertEqual(app.session_state["current_menu"], menu)
            self.assertFalse(app.exception)

    def test_home_quick_navigation_only_changes_page(self):
        targets = {
            "추천 실행 보기": "추천 실행",
            "경로 상세 보기": "경로 상세",
            "분석 및 검증 보기": "분석 및 검증",
            "데이터 관리 보기": "데이터 관리",
        }
        for label, menu in targets.items():
            app = self._new_app()
            app.session_state["current_menu"] = "홈"
            app.run()
            next(item for item in app.button if item.label == label).click().run()
            self.assertFalse(app.exception, msg=f"{label}: {list(app.exception)}")
            self.assertEqual(app.session_state["current_menu"], menu)
            self.assertIsNone(app.session_state["dqn_training_result"])
            self.assertIsNone(app.session_state["dqn_batch_result"])
            self.assertIsNone(app.session_state["dqn_comparison_result"])

    def test_sidebar_nav_persists_selected_route_across_pages(self):
        app = self._new_app()
        app.session_state["selected_route_id"] = "R002"
        app.session_state["current_menu"] = "홈"
        app.run()
        for menu in ("추천 실행", "경로 상세", "분석 및 검증", "데이터 관리", "홈"):
            button = next(item for item in app.sidebar.button if item.key == f"nav_{menu}")
            button.click().run()
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")
            self.assertEqual(app.session_state["current_menu"], menu)
            self.assertEqual(app.session_state["selected_route_id"], "R002")

    def _dataframe_columns(self, app) -> set:
        columns: set = set()
        for element in app.dataframe:
            try:
                columns |= set(element.value.columns)
            except Exception:
                pass
        return columns

    def test_validation_page_shows_compact_validation_and_comparison(self):
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        columns = self._dataframe_columns(app)
        for required in ("VHS 순위", "Greedy 순위", "DQN 상태", "DQN 반영", "Pareto 순위"):
            self.assertIn(required, columns)
        blob = self._markdown_blob(app)
        for required in ("VHS 자동 가중치", "민감도 · 추천 신뢰도", "제한 탐색 기반 검증"):
            self.assertIn(required, blob)

    def test_validation_page_has_required_dqn_actions(self):
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        labels = {button.label for button in app.button}
        self.assertTrue({
            "현재 샘플 진단", "선택 샘플 원본 학습", "선택 샘플 균형형 학습",
            "10개 원본 순차 학습", "10개 균형형 순차 학습", "원본 vs 균형형 비교 리포트",
        }.issubset(labels))
        self.assertIn("학습 결과", self._markdown_blob(app))

    def test_validation_does_not_start_dqn_before_button_click(self):
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state["dqn_training_result"])
        self.assertIsNone(app.session_state["dqn_batch_result"])
        self.assertIsNone(app.session_state["dqn_comparison_result"])

    def test_dqn_buttons_run_training_batch_and_comparison(self):
        from services.dqn_service import get_torch_status

        if not get_torch_status()[0]:
            self.skipTest("DQN runtime unavailable")
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()

        next(button for button in app.button if button.label == "선택 샘플 원본 학습").click().run(timeout=180)
        self.assertFalse(app.exception)
        result = app.session_state["dqn_training_result"]
        self.assertTrue(Path(result["result_path"]).exists())

        next(button for button in app.button if button.label == "10개 원본 순차 학습").click().run(timeout=240)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["dqn_batch_result"]["count"], 10)

        next(button for button in app.button if button.label == "선택 샘플 원본 vs 균형형 비교").click().run(timeout=180)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["dqn_comparison_result"]["rows"]), 2)

        next(button for button in app.button if button.label == "원본 vs 균형형 비교 리포트").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["dqn_batch_comparison_result"]["rows"]), 10)

    def test_validation_has_five_clear_tabs(self):
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        labels = [tab.label for tab in app.tabs]
        self.assertEqual(labels, ["VHS 분석", "Greedy 비교", "DQN 학습·비교", "Pareto 검증", "민감도/신뢰도"])

    def test_recommendation_page_has_compact_table_and_detail_expander(self):
        app = self._new_app()
        app.session_state["current_menu"] = "추천 실행"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("추천 후보 Top 5", blob)
        self.assertIn("선택한 추천 요약", blob)
        basic = next(item for item in app.dataframe if "순위" in item.value.columns and "추천 등급" in item.value.columns)
        self.assertEqual(list(basic.value.columns), ["순위", "상품", "출발 점포", "도착 점포", "경로 유형", "수량", "예상 절감액", "추천 등급"])
        self.assertIn("상세 비교", {item.label for item in app.expander})
        self.assertIn("추천 후보 선택", {item.label for item in app.selectbox})
        detail_columns = self._dataframe_columns(app)
        for required in (
            "추천 ID", "VHS 점수", "Greedy 전략", "DQN action", "DQN confidence",
            "DQN 참고 점수", "Pareto 상태", "Varo 최종 추천", "판단 근거",
        ):
            self.assertIn(required, detail_columns)
        self.assertNotIn("필터", blob)
        self.assertNotIn("1순위 추천", blob)
        button_labels = {button.label for button in app.button}
        self.assertNotIn("현재 추천 CSV", button_labels)
        self.assertNotIn("현재 추천 Excel", button_labels)

    def test_recommendation_selector_updates_shared_route(self):
        from services.analysis_pipeline import sort_recommendations

        app = self._new_app()
        app.session_state["current_menu"] = "추천 실행"
        app.run()
        selector = next(item for item in app.selectbox if item.key == "recommendation_route_select")
        self.assertGreaterEqual(len(selector.options), 2)
        selected_route = sort_recommendations(self.payload["varo_recommendations"])[1]["route_id"]
        selector.set_value(selected_route).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_route_id"], selected_route)
        self.assertEqual(next(
            item for item in app.selectbox if item.key == "recommendation_route_select"
        ).value, selected_route)

    def test_route_detail_keeps_only_operator_summary(self):
        app = self._new_app()
        app.session_state["selected_route_id"] = "R002"
        app.session_state["current_menu"] = "경로 상세"
        app.run()
        blob = self._markdown_blob(app)
        for required in ("출발 점포", "도착 점포", "DC 경유 여부", "경로 유형", "추천 수량", "예상 절감액", "이동 거리", "예상 시간", "이동 방식", "경로 설명", "이동 단계"):
            self.assertIn(required, blob)
        for removed in ("직접 이동과 DC 경유 비교", "VHS 구성", "route_id"):
            self.assertNotIn(removed, blob)
        self.assertEqual(app.session_state["selected_route_id"], "R002")

    def test_recommendation_and_route_pages_hide_long_explanations(self):
        for menu in ("추천 실행", "경로 상세"):
            app = self._new_app()
            app.session_state["selected_route_id"] = "R002"
            app.session_state["current_menu"] = menu
            app.run()
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")
            blob = self._markdown_blob(app)
            self.assertNotIn("V2 추천 사유 요약", blob)
            self.assertNotIn("계산 함수", blob)

    def _generated_state(self):
        from services.data_application import load_and_apply
        from tests.fixtures import sample_workbook, workbook_excel_bytes
        workbook = sample_workbook()
        workbook.pop("recommendations", None)
        state: dict = {}
        load_and_apply(state, workbook_excel_bytes(workbook), "no_rec.xlsx", "업로드된 추천 결과")
        return state

    def _inject(self, app, state):
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = state.get(key)

    def test_generated_candidate_upload_renders(self):
        generated = self._generated_state()
        self.assertEqual(generated.get("recommendation_source"), "generated")
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        self._inject(app, generated)
        # recommendations page renders without exception
        app.session_state["current_menu"] = "추천 실행"
        app.run()
        self.assertFalse(app.exception)
        # data management remains compact for generated candidates
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("현재 로드 상태", self._markdown_blob(app))
        self.assertIn("DQN 샘플 10개 목록", {item.label for item in app.expander})

    def test_data_management_owns_samples_and_compact_summary(self):
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        sample_button = next(button for button in app.button if button.key == "quick_empty_sample")
        sample_button.click().run()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("DQN 샘플 10개 목록", {item.label for item in app.expander})
        self.assertIn("현재 로드 상태", blob)
        expander_labels = {item.label for item in app.expander}
        self.assertNotIn("원본 데이터 보기", expander_labels)
        button_labels = {button.label for button in app.button}
        self.assertIn("기본 샘플 불러오기", button_labels)
        self.assertIn("선택한 DQN 샘플 불러오기", button_labels)
        self.assertNotIn("분석 결과 전체 Excel", button_labels)

    def test_runtime_pages_are_backup_free_and_route_detail_is_self_contained(self):
        root = Path(__file__).resolve().parents[1]
        runtime_files = [root / "app_v2.py", root / "router.py", root / "components" / "navigation.py", *sorted((root / "pages").glob("*.py"))]
        source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files).lower()
        self.assertNotIn("bad_inventory_simulator_backup", source)
        self.assertNotIn("zipfile", source)
        route_source = (root / "pages" / "route_detail.py").read_text(encoding="utf-8").lower()
        other_source = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in runtime_files if path.name != "route_detail.py"
        )
        self.assertIn("_render_route_steps", route_source)
        self.assertNotIn("streamlit.components.v1", route_source)
        self.assertNotIn("components.html", route_source)
        self.assertNotIn("zipfile", other_source)

    def test_deployment_files_split_lightweight_and_dqn_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        basic = (root / "requirements.txt").read_text(encoding="utf-8")
        dqn = (root / "requirements-dqn.txt").read_text(encoding="utf-8")
        readme = (root / "README_V2.md").read_text(encoding="utf-8")
        for required in ("streamlit", "pandas", "numpy", "openpyxl", "scipy", "scikit-learn"):
            self.assertIn(required, basic)
        self.assertNotIn("torch", basic.lower())
        self.assertIn("-r requirements.txt", dqn)
        self.assertIn("torch", dqn.lower())
        for required in ("requirements.txt", "requirements-dqn.txt", "PyTorch"):
            self.assertIn(required, readme)
        for exaggerated in ("논문급", "State of the Art", "완전한 최적화"):
            self.assertNotIn(exaggerated, readme)

    def test_manual_deployment_checklist_covers_required_browser_review(self):
        root = Path(__file__).resolve().parents[1]
        checklist = (root / "DEPLOY_CHECKLIST.md").read_text(encoding="utf-8")
        for required in (
            "http://localhost:8501", "홈", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리",
            "샘플 01", "샘플 10", "10점포·2DC", "Console", "검은", "DQN 원본 10개",
            "DQN 균형형 10개", "VHS/Greedy/DQN/Pareto", "DIRECT", "VIA_DC", "DC01/DC02", "원본 보호",
        ):
            self.assertIn(required, checklist)

    def test_submission_documents_cover_required_sections(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README_V2.md").read_text(encoding="utf-8")
        summary = (root / "APP_SUMMARY.md").read_text(encoding="utf-8")
        for required in (
            "프로젝트 개요", "기존 Varo와 Varo V2의 차이", "교수님 피드백 반영 내용",
            "주요 기능", "DQN 처리 방식", "경로 상세 처리 방식", "실행 방법", "배포 방법",
            "DEPLOY_CHECKLIST.md", "남은 확인 사항",
        ):
            self.assertIn(required, readme)
        for required in (
            "한 줄 설명", "해결하려는 문제", "핵심 기능 5개", "알고리즘 구조",
            "DQN 원본 vs 균형형 비교 결과 요약", "교수님 피드백 반영 요약",
            "현재 완성 상태", "남은 확인 사항", "제출 시 사용할 설명 문장",
        ):
            self.assertIn(required, summary)
        for exaggerated in ("논문급", "SOTA", "완전 자동 최적화 완성", "최고 성능 보장", "무조건 최적"):
            self.assertNotIn(exaggerated, readme + summary)

    def test_submission_copy_uses_user_facing_dqn_labels(self):
        from components.status import user_status_label

        expected = {
            "불안정": "데이터 편향 큼",
            "검토 필요": "비교 전 데이터 확인 필요",
            "학습 필요": "학습 후 비교 가능",
            "PyTorch 미설치": "DQN 학습 실행 환경 필요",
            "DQN 반영 안 함": "최종 추천에는 참고 제외",
        }
        for raw, display in expected.items():
            self.assertEqual(user_status_label(raw), display)

        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        visible_copy = " ".join(
            [self._markdown_blob(app)]
            + [item.value for item in app.caption]
        )
        self.assertIn("데이터 품질 진단 및 학습 안정성 비교", visible_copy)
        self.assertIn("VHS는 최종 우선순위", visible_copy)

    def test_sample_load_shows_current_data_summary(self):
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        buttons = [b for b in app.button if b.key == "quick_empty_sample"]
        self.assertTrue(buttons)
        buttons[0].click().run()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("현재 로드 상태", blob)

    def test_sample_button_loads_and_navigates_backup_free(self):
        """The live '기본 샘플 불러오기' flow runs the self-contained pipeline."""
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        buttons = [b for b in app.button if b.key == "quick_empty_sample"]
        self.assertTrue(buttons, "기본 샘플 불러오기 버튼을 찾지 못했습니다")
        buttons[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("varo_recommendations", app.session_state)
        self.assertTrue(app.session_state["varo_recommendations"])
        app.session_state["selected_route_id"] = "R002"
        for menu in MENUS:
            app.session_state["current_menu"] = menu
            app.run()
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")
            self.assertEqual(app.session_state["selected_route_id"], "R002")


if __name__ == "__main__":
    unittest.main()
