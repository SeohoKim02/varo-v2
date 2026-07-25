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
MENUS = ["운영 현황", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리"]


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

    def test_data_management_download_section_replaces_stub(self):
        app = self._new_app()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("분석 결과 다운로드", blob)
        self.assertNotIn("다운로드 미연결", blob)

    def test_route_detail_renders_kakao_key_fallback(self):
        app = self._new_app()
        app.session_state["current_menu"] = "경로 상세"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("지도", blob)
        self.assertIn("지도 키가 설정되면 실제 지도에서 경로를 확인할 수 있습니다", blob)
        self.assertNotIn("지도 SDK는 이번 단계에서 연결하지 않았습니다", blob)

    def test_home_is_result_dashboard(self):
        app = self._new_app()
        app.session_state["current_menu"] = "운영 현황"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        # required result-dashboard elements
        for required in (
            "Varo 운영 결과",
            "재고 이동 추천과 예상 절감 효과를 확인합니다.",
            # result-only KPIs (no 평균 VHS on home): 추천 후보·권장 이동 수량·예상 순효과·신뢰도·데이터 상태
            "추천 후보", "권장 이동 수량", "예상 순효과", "추천 신뢰도", "데이터 상태",
            "엑셀 업로드", "재고 분석", "이동 추천", "결과 확인",
            "추천 경로 이동 현황", "추천 Top 5",
            "최우선 추천",
        ):
            self.assertIn(required, blob, f"home must contain: {required}")
        # page navigation lives in the collapsed sidebar, not a horizontal/bottom menu
        sidebar_nav = {b.label: b.key for b in app.sidebar.button}
        self.assertEqual(set(sidebar_nav), set(MENUS))
        for menu in MENUS:
            self.assertEqual(sidebar_nav[menu], f"nav_{menu}")
        button_labels = {b.label for b in app.button}
        # the old bottom page-nav buttons are gone from the home body
        for removed_btn in ("추천 실행 보기", "경로 상세 보기"):
            self.assertNotIn(removed_btn, button_labels, f"home should not have button: {removed_btn}")
        # the top toolbar keeps only the data-replace toggle (no duplicate 데이터 관리 button)
        self.assertIn("데이터 교체", button_labels)
        # forbidden elements / developer copy
        for banned in (
            "실제 V2 내부 알고리즘 재계산 결과 기준",
            "DQN 과거 학습 결과는 제외",
            "varo_hybrid_score",
            "카카오 SDK",
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
        # home Top table is result-only: exactly the 7 operator columns
        columns = self._dataframe_columns(app)
        for required_col in ("순위", "상품", "출발", "도착", "경로", "수량", "예상 절감액"):
            self.assertIn(required_col, columns, f"home Top5 must have column: {required_col}")
        for hidden in ("VHS", "VHS(재계산)", "DQN 상태", "Greedy", "신뢰도", "route_id", "상태"):
            self.assertNotIn(hidden, columns)
        self.assertIn('class="network-node dc-node"', blob)
        self.assertIn('class="network-node store-node', blob)
        self.assertEqual(blob.count('class="v2-vehicle"'), 3)
        # 전체 경로 보기 defaults OFF (only Top 3 routes shown)
        self.assertFalse(app.session_state["show_all_routes"])

    def _all_blocked_state(self):
        from services.data_application import load_and_apply
        from tests.fixtures import sample_workbook, workbook_excel_bytes
        workbook = sample_workbook()
        recs = workbook["recommendations"].copy()
        recs["recommended_qty"] = 999999
        workbook["recommendations"] = recs
        state: dict = {}
        load_and_apply(state, workbook_excel_bytes(workbook), "blocked.xlsx", "업로드된 추천 결과")
        return state

    def test_home_no_data_state_shows_single_action(self):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.run()
        app.session_state["current_menu"] = "운영 현황"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("데이터를 준비하세요", blob)
        # no result KPI labels / 0-values in an empty state
        for hidden in ("예상 순효과", "권장 이동 수량", "추천 Top 5"):
            self.assertNotIn(hidden, blob)
        button = next(b for b in app.button if b.key == "home_primary_action")
        self.assertEqual(button.label, "데이터 불러오기")
        button.click().run()
        self.assertEqual(app.session_state["current_menu"], "데이터 관리")

    def test_home_pending_intake_keeps_applied_result(self):
        # Two-phase: a bad *new* upload (pending) must not hide the applied result.
        app = self._new_app()  # applied good data present
        app.session_state["pending_load_error"] = "파일 형식을 확인해주세요."
        app.session_state["current_menu"] = "운영 현황"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("최우선 추천", blob)          # applied result stays
        self.assertIn("예상 순효과", blob)
        self.assertNotIn("데이터를 수정해야 합니다", blob)  # no app-wide 사용 불가 takeover
        infos = " ".join(el.value for el in app.info)
        self.assertIn("검사 완료된 새 데이터가 있습니다", infos)  # short intake notice

    def test_home_no_candidates_state_shows_cause_and_action(self):
        state = self._all_blocked_state()
        self.assertEqual(state["varo_recommendations"], [])
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        self._inject(app, state)
        app.session_state["current_menu"] = "운영 현황"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("추천할 이동이 없습니다", blob)
        self.assertIn("후보", blob)                 # a plain cause sentence
        self.assertNotIn("예상 순효과", blob)
        for internal in ("candidate_id", "data_signature", "reason_code", "Traceback"):
            self.assertNotIn(internal, blob)
        button = next(b for b in app.button if b.key == "home_primary_action")
        self.assertEqual(button.label, "제외 이유 확인")
        button.click().run()
        self.assertEqual(app.session_state["current_menu"], "추천 실행")
        # the action lands on real content: the excluded-candidate list, not "데이터 없음"
        self.assertFalse(app.exception)
        rec_blob = self._markdown_blob(app)
        self.assertNotIn("분석 결과가 없습니다", rec_blob)
        labels = {item.label for item in app.expander}
        self.assertTrue(any("추천에서 제외된 후보" in label for label in labels))

    def test_home_ready_detail_button_navigates_to_route_detail(self):
        app = self._new_app()
        app.session_state["current_menu"] = "운영 현황"
        app.run()
        self.assertFalse(app.exception)
        button = next(b for b in app.button if b.key == "home_detail_action")
        button.click().run()
        self.assertEqual(app.session_state["current_menu"], "경로 상세")
        # the selected candidate is the shared top recommendation (valid id)
        valid_ids = {str(r["route_id"]) for r in app.session_state["varo_recommendations"]}
        self.assertIn(str(app.session_state["selected_route_id"]), valid_ids)

    def test_validation_no_data_shows_state_card_not_empty_tabs(self):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.run()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 0)  # no six empty algorithm tabs
        blob = self._markdown_blob(app)
        self.assertIn("데이터를 준비하세요", blob)
        self.assertNotIn("데이터가 업로드되지 않았습니다", blob)
        button = next(b for b in app.button if b.key == "validation_primary_action")
        self.assertEqual(button.label, "데이터 불러오기")
        button.click().run()
        self.assertEqual(app.session_state["current_menu"], "데이터 관리")

    def test_validation_no_candidates_shows_status_summary_not_data_missing(self):
        state = self._all_blocked_state()
        self.assertEqual(state["varo_recommendations"], [])
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        self._inject(app, state)
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 0)  # no empty VHS/Greedy/Pareto/DQN tabs
        blob = self._markdown_blob(app)
        self.assertIn("추천할 이동이 없습니다", blob)
        self.assertIn("후보 판단 요약", blob)
        self.assertNotIn("데이터가 업로드되지 않았습니다", blob)
        metric_labels = {m.label for m in app.metric}
        self.assertIn("전체 생성 후보", metric_labels)
        self.assertIn("이동 불가", metric_labels)
        for internal in ("candidate_id", "data_signature", "reason_code", "Traceback", "status_code"):
            self.assertNotIn(internal, blob)
        expander_labels = {e.label for e in app.expander}
        self.assertTrue(any("추천에서 제외된 후보" in label for label in expander_labels))
        button = next(b for b in app.button if b.key == "validation_primary_action")
        self.assertEqual(button.label, "제외 이유 확인")

    def test_validation_pending_intake_keeps_applied_tabs(self):
        # A bad *new* upload (pending) does not hide the applied analysis: the six
        # algorithm tabs still render from the current applied result.
        app = self._new_app()  # applied good data present
        app.session_state["pending_load_error"] = "파일 형식을 확인해주세요."
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 6)

    def test_validation_ready_keeps_six_algorithm_tabs(self):
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 6)

    def test_sidebar_nav_navigates_to_every_page(self):
        for menu in MENUS:
            app = self._new_app()
            app.session_state["current_menu"] = "운영 현황"
            app.run()
            button = next(item for item in app.sidebar.button if item.key == f"nav_{menu}")
            button.click().run()
            self.assertEqual(app.session_state["current_menu"], menu)
            self.assertFalse(app.exception)

    def test_sidebar_nav_persists_selected_route_across_pages(self):
        app = self._new_app()
        app.session_state["selected_route_id"] = "R002"
        app.session_state["current_menu"] = "운영 현황"
        app.run()
        for menu in ("추천 실행", "경로 상세", "분석 및 검증", "데이터 관리", "운영 현황"):
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

    def test_validation_page_shows_score_summary_in_plain_words(self):
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        # Plain-language table headers, not developer terms.
        self.assertIn("업로드 VHS", blob)
        self.assertIn("현재 VHS", blob)
        self.assertNotIn("재계산 VHS", blob)
        # Detail comparison table (full results expander) uses friendly headers.
        columns = self._dataframe_columns(app)
        self.assertIn("현재 VHS", columns)
        self.assertIn("업로드 VHS", columns)

    def test_validation_page_hides_internal_field_names(self):
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        # 점수 구성 tab groups components in plain Korean, not English score fields.
        self.assertIn("구성 그룹", blob)
        self.assertIn("반영 비중", blob)
        for internal in ("disposal_risk_score", "demand_fit_score", "route_cost_score", "feasibility_score"):
            self.assertNotIn(internal, blob)

    def test_validation_tabs_are_simplified_to_six(self):
        app = self._new_app()
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        labels = [tab.label for tab in app.tabs]
        self.assertEqual(labels, [
            "추천 점수", "점수 구성", "비교 분석", "민감도", "DQN 학습", "검증 결과",
        ])

    def test_recommendation_page_keeps_downloads_and_compact_table(self):
        app = self._new_app()
        app.session_state["current_menu"] = "추천 실행"
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("추천 후보", self._markdown_blob(app))
        source = (Path(APP_PATH).parent / "pages" / "recommendations.py").read_text(encoding="utf-8")
        self.assertIn('"현재 추천 CSV"', source)
        self.assertIn('"현재 추천 Excel"', source)
        self.assertIn("download_button", source)
        # The compact operational table is an in-DOM HTML table, so every basic
        # header is present in the markdown (no dataframe virtualization).
        blob = self._markdown_blob(app)
        self.assertIn('class="v2-html-table"', blob)
        for required in ("순위", "상품", "출발 점포", "도착 점포", "경로 유형", "수량", "예상 절감액", "추천 등급"):
            self.assertIn(f"<th>{required}</th>", blob)
        for hidden in ("route_id", "VHS 점수", "Greedy 순위"):
            self.assertNotIn(f"<th>{hidden}</th>", blob)
        # VHS/Greedy/DQN/Pareto live only in the detailed comparison (expander),
        # with friendly column names (route_id → 추천 ID, Pareto → Pareto 상태).
        column_sets = []
        for element in app.dataframe:
            try:
                column_sets.append(set(element.value.columns))
            except Exception:
                pass
        detailed = next((cols for cols in column_sets if "Pareto 상태" in cols), None)
        self.assertIsNotNone(detailed, "detailed VHS/Greedy/DQN/Pareto comparison not found")
        self.assertIn("VHS 순위", detailed)
        self.assertIn("Greedy 순위", detailed)
        self.assertIn("추천 ID", detailed)
        self.assertIn("추천 ID", detailed)
        self.assertNotIn("route_id", detailed)

    def test_route_detail_keeps_steps_and_route_comparison(self):
        app = self._new_app()
        app.session_state["selected_route_id"] = "R002"
        app.session_state["current_menu"] = "경로 상세"
        app.run()
        blob = self._markdown_blob(app)
        self.assertIn("이동 단계", blob)
        self.assertIn("이동 방식 비교", blob)
        self.assertIn("출발 점포에서 DC로 이동", blob)
        self.assertEqual(app.session_state["selected_route_id"], "R002")

    def _excluded_ledger_record(self) -> dict:
        return {
            "candidate_id": "C-secret99-deadbeef", "route_id": "X999",
            "status": "이동 불가", "status_code": "blocked_move",
            "blocks_recommendation": True, "is_top": False,
            "short_reason": "출발 점포 재고보다 이동 수량이 많습니다.",
            "recommendation_reasons": [], "exclusion_reasons": ["출발 점포 재고보다 이동 수량이 많습니다."],
            "quantity_basis": {"basis_text": None}, "traceable_row_count": 1,
            "source_references": [{
                "role": "출발 재고", "file": "inv.xlsx", "sheet": "inventory",
                "sheet_name": "재고현황", "rows": [12], "column": "stock_qty",
                "value": "15", "traceable": True, "impact": "이동 가능 수량 계산에 사용했습니다.",
            }],
            "product_name": "우유", "source_name": "가게1", "target_name": "가게2",
            "route_type": "DIRECT", "recommended_qty": 99999.0,
        }

    def _state_with_excluded(self, app) -> None:
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        pipeline = dict(self.payload.get("varo_pipeline_result") or {})
        pipeline["candidate_ledger"] = list(pipeline.get("candidate_ledger") or []) + [self._excluded_ledger_record()]
        pipeline["ledger_summary"] = {
            "generated": 5, "recommendable_total": 4, "check_needed": 0,
            "blocked_move": 1, "insufficient_data": 0, "not_computable": 0,
            "excluded_total": 1, "top_exclusion_reasons": [{"reason": "출발 점포 재고보다 이동 수량이 많습니다.", "count": 1}],
        }
        app.session_state["varo_pipeline_result"] = pipeline
        app.session_state["analysis_result"] = pipeline

    def test_recommendations_shows_excluded_folded_without_internal_id(self):
        app = self._new_app()
        self._state_with_excluded(app)
        app.session_state["current_menu"] = "추천 실행"
        app.run()
        self.assertFalse(app.exception)
        labels = {item.label for item in app.expander}
        self.assertIn("추천에서 제외된 후보 1건", labels)
        blob = self._markdown_blob(app)
        self.assertNotIn("C-secret99-deadbeef", blob)  # internal id never on screen
        self.assertNotIn("blocked_move", blob)          # internal status code hidden
        columns = self._dataframe_columns(app)
        self.assertIn("가장 중요한 이유", columns)      # excluded list rendered
        self.assertIn("원본 위치", columns)
        self.assertNotIn("candidate_id", columns)

    def test_validation_shows_candidate_status_counts(self):
        app = self._new_app()
        self._state_with_excluded(app)
        app.session_state["current_menu"] = "분석 및 검증"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("후보 판단 요약", blob)
        metric_labels = {m.label for m in app.metric}
        self.assertIn("이동 불가", metric_labels)
        self.assertIn("전체 생성 후보", metric_labels)

    def test_route_detail_offers_source_location_detail(self):
        app = self._new_app()
        app.session_state["selected_route_id"] = "R002"
        app.session_state["current_menu"] = "경로 상세"
        app.run()
        self.assertFalse(app.exception)
        labels = {item.label for item in app.expander}
        self.assertTrue(any("원본 데이터 확인" in label for label in labels))

    def test_recommendation_and_route_pages_show_v2_reason(self):
        for menu in ("추천 실행", "경로 상세"):
            app = self._new_app()
            app.session_state["selected_route_id"] = "R002"
            app.session_state["current_menu"] = menu
            app.run()
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")
            blob = self._markdown_blob(app)
            self.assertIn("추천 판단 근거", blob)

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

    def _infeasible_state(self):
        from services.data_application import load_and_apply
        from tests.fixtures import sample_workbook, workbook_excel_bytes
        workbook = sample_workbook()
        recs = workbook["recommendations"].copy()
        recs.loc[recs["route_id"] == "R001", "recommended_qty"] = 99999  # exceeds S001 stock
        workbook["recommendations"] = recs
        state: dict = {}
        load_and_apply(state, workbook_excel_bytes(workbook), "inv.xlsx", "업로드된 추천 결과")
        return state

    def test_real_excluded_candidate_flow_end_to_end(self):
        state = self._infeasible_state()
        pipeline = state["varo_pipeline_result"]
        self.assertEqual(len(pipeline["excluded_candidates"]), 1)
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        self._inject(app, state)
        app.session_state["current_menu"] = "추천 실행"
        app.run()
        self.assertFalse(app.exception)
        labels = {item.label for item in app.expander}
        self.assertTrue(any("추천에서 제외된 후보" in label for label in labels))
        # the excluded candidate's source stock row is traceable to the original file
        excluded = pipeline["excluded_candidates"][0]
        stock_ref = next(r for r in excluded["source_references"] if r["role"] == "출발 재고")
        self.assertTrue(stock_ref["traceable"])
        self.assertEqual(stock_ref["rows"], [2])

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
        # data management owns upload quality and generated-candidate diagnostics
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("업로드 품질 점검", self._markdown_blob(app))
        alerts = " ".join(el.value for el in list(app.info) + list(app.warning) + list(app.success))
        self.assertIn("후보를 자동 생성", self._markdown_blob(app) + " " + alerts)

    def test_data_management_owns_quality_raw_data_and_downloads(self):
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        sample_button = next(button for button in app.button if button.key == "quick_empty_sample")
        sample_button.click().run()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("업로드 품질 점검", blob)
        self.assertIn("분석 결과 다운로드", blob)
        # DQN history/settings duplicates were consolidated into 분석 및 검증.
        self.assertNotIn("DQN 학습 이력", blob)
        self.assertNotIn("지도는 이후 연결 예정", blob)
        expander_labels = {item.label for item in app.expander}
        self.assertIn("원본 데이터 보기", expander_labels)
        # The two sample-load buttons must be clearly distinct, not confusable.
        button_labels = [b.label for b in app.button]
        self.assertIn("선택한 DQN 샘플 적용", button_labels)
        self.assertEqual(button_labels.count("선택한 샘플 적용"), 1)
        self.assertEqual(button_labels.count("선택한 DQN 샘플 적용"), 1)
        source = (Path(APP_PATH).parent / "pages" / "data_management.py").read_text(encoding="utf-8")
        self.assertIn('"추천 결과 CSV"', source)
        self.assertIn('"분석 결과 전체 Excel"', source)
        self.assertIn("download_button", source)

    def test_validation_page_imports_interactive_helpers(self):
        # Guards the missing-import class of NameError that only fires on button
        # clicks (diagnosis loop, torch badge) rather than on plain render.
        import pages.validation as validation_page
        for name in (
            "diagnosis_progress_label", "run_sequential_diagnosis", "diagnose_sample",
            "diagnosis_rows", "get_torch_runtime_status", "build_strategy_core",
            "build_strategy_detail", "generate_balanced_sample",
            "train_dqn_on_balanced_sample", "compare_samples",
            "comparison_display_rows", "save_comparison_report",
        ):
            self.assertTrue(hasattr(validation_page, name), f"validation.py must import {name}")

    def test_runtime_pages_do_not_connect_external_backup_and_home_does_not_load_kakao(self):
        root = Path(__file__).resolve().parents[1]
        runtime_files = [root / "app_v2.py", root / "router.py", *sorted((root / "pages").glob("*.py"))]
        source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files).lower()
        self.assertNotIn("bad_inventory_simulator_backup", source)
        self.assertNotIn("zipfile", source)
        home_source = (root / "pages" / "overview.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("dapi.kakao.com", home_source)
        self.assertNotIn("components.html", home_source)

    def test_sample_load_shows_upload_quality_section(self):
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        buttons = [b for b in app.button if b.key == "quick_empty_sample"]
        self.assertTrue(buttons)
        buttons[0].click().run()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("업로드 품질 점검", blob)

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

    # ----------------------------------------------------------------------- #
    # Data management page states (element tree)
    # ----------------------------------------------------------------------- #
    def test_data_management_no_data_shows_loaders_not_result_cards(self):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.run()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        # shared wording with the home 데이터 없음 state
        self.assertIn("데이터를 준비하세요", blob)
        self.assertIn("데이터 불러오기", blob)
        keys = {b.key for b in app.button}
        self.assertIn("load_simulation_sample", keys)
        # no current-data card / next action / clear button / download section on empty
        self.assertNotIn("현재 사용 중 데이터", blob)
        self.assertNotIn("data_next_action", keys)
        self.assertNotIn("clear_applied_data", keys)
        self.assertNotIn("분석 결과 다운로드", blob)
        for internal in ("data_signature", "session_state", "Traceback", "load_and_apply"):
            self.assertNotIn(internal, blob)

    def test_data_management_applied_shows_current_card_and_next_action(self):
        app = self._new_app()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("현재 사용 중 데이터", blob)
        keys = {b.key for b in app.button}
        self.assertIn("clear_applied_data", keys)
        self.assertIn("data_next_action", keys)
        button = next(b for b in app.button if b.key == "data_next_action")
        self.assertEqual(button.label, "추천 실행")
        button.click().run()
        self.assertEqual(app.session_state["current_menu"], "추천 실행")

    def test_data_management_clear_button_resets_workspace(self):
        app = self._new_app()
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        button = next(b for b in app.button if b.key == "clear_applied_data")
        button.click().run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state["varo_data"])
        self.assertEqual(app.session_state["varo_recommendations"], [])
        self.assertIn("데이터를 준비하세요", self._markdown_blob(app))

    def _applied_plus_unusable_pending(self) -> dict:
        from services.data_application import load_and_apply
        from tests.fixtures import sample_workbook, workbook_excel_bytes
        state: dict = {}
        load_and_apply(state, workbook_excel_bytes(sample_workbook()), "good.xlsx", "샘플 추천 데이터")
        workbook = sample_workbook()
        workbook["inventory"] = workbook["inventory"].drop(columns=["stock_qty"])
        load_and_apply(state, workbook_excel_bytes(workbook), "오류파일.xlsx", "업로드된 추천 결과")
        return state

    def test_data_management_unusable_pending_keeps_applied_and_blocks_apply(self):
        state = self._applied_plus_unusable_pending()
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        self._inject(app, state)
        for key in (
            "pending_varo_data", "pending_varo_validation", "pending_uploaded_filename",
            "pending_data_source_type", "pending_upload_report", "pending_raw_data",
            "pending_source_metadata",
        ):
            app.session_state[key] = state.get(key)
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("검사 중인 데이터", blob)         # pending shown separately
        self.assertIn("사용 불가", blob)                # pending intake status badge
        self.assertIn("현재 사용 중 데이터", blob)        # applied data preserved
        keys = {b.key for b in app.button}
        self.assertNotIn("apply_pending", keys)          # no apply button for 사용 불가
        self.assertIn("cancel_pending", keys)            # intake can be cancelled
        self.assertNotIn("data_next_action", keys)       # no forward action while a pending is open
        # applied recommendations are still in session (not deleted by the bad upload)
        self.assertTrue(app.session_state["varo_recommendations"])
        for internal in ("data_signature", "Traceback", "canonical_column_name"):
            self.assertNotIn(internal, blob)

    # ----------------------------------------------------------------------- #
    # Two-phase intake UI: inspect → explicit apply / cancel (element tree)
    # ----------------------------------------------------------------------- #
    def _prepared_pending(self, source_bytes, name="new.xlsx") -> dict:
        import warnings
        from services.data_application import prepare_pending_data
        state: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepare_pending_data(state, source_bytes, name, "업로드된 추천 결과")
        return state

    def _inject_pending(self, app, state) -> None:
        from services.data_application import PENDING_KEYS
        for key in PENDING_KEYS:
            app.session_state[key] = state.get(key)

    def test_data_management_valid_pending_apply_commits_on_click(self):
        from tests.fixtures import sample_workbook, workbook_excel_bytes
        pend = self._prepared_pending(workbook_excel_bytes(sample_workbook()), "정상.xlsx")
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        self._inject_pending(app, pend)
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        blob = self._markdown_blob(app)
        self.assertIn("검사 중인 데이터", blob)
        self.assertIn("사용 가능", blob)
        # a rerun alone must NOT apply — nothing changes without a button click
        self.assertIsNone(app.session_state["varo_data"])
        button = next(b for b in app.button if b.key == "apply_pending")
        self.assertEqual(button.label, "이 데이터 사용")
        button.click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["varo_recommendations"])       # applied now
        self.assertNotIn("pending_varo_data", app.session_state)         # pending cleared
        self.assertIn("현재 사용 중 데이터", self._markdown_blob(app))

    def test_data_management_warning_pending_shows_exclude_button(self):
        import pandas as pd
        from tests.fixtures import sample_workbook, workbook_excel_bytes
        workbook = sample_workbook()
        routes = workbook["routes"]
        workbook["routes"] = pd.concat([routes, routes.iloc[[0]]], ignore_index=True)  # WARNING
        pend = self._prepared_pending(workbook_excel_bytes(workbook), "경고.xlsx")
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        self._inject_pending(app, pend)
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("확인 필요", self._markdown_blob(app))
        button = next(b for b in app.button if b.key == "apply_pending")
        self.assertEqual(button.label, "문제 행을 제외하고 사용")

    def test_data_management_cancel_pending_keeps_applied(self):
        import warnings
        from services.app_state import CANONICAL_DATA_KEYS
        from services.data_application import load_and_apply, prepare_pending_data
        from tests.fixtures import sample_workbook, workbook_excel_bytes
        state: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_and_apply(state, workbook_excel_bytes(sample_workbook()), "good.xlsx", "샘플 추천 데이터")
            workbook = sample_workbook()
            recs = workbook["recommendations"].copy()
            recs.loc[0, "expected_saving"] = 314159
            workbook["recommendations"] = recs
            prepare_pending_data(state, workbook_excel_bytes(workbook), "새파일.xlsx", "업로드된 추천 결과")
        app = AppTest.from_file(APP_PATH, default_timeout=120)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = state.get(key)
        self._inject_pending(app, state)
        app.session_state["current_menu"] = "데이터 관리"
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("검사 중인 데이터", self._markdown_blob(app))
        recs_before = list(app.session_state["varo_recommendations"])
        button = next(b for b in app.button if b.key == "cancel_pending")
        button.click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("pending_varo_data", app.session_state)            # intake dropped
        self.assertEqual(app.session_state["varo_recommendations"], recs_before)  # applied kept
        self.assertNotIn("검사 중인 데이터", self._markdown_blob(app))


if __name__ == "__main__":
    unittest.main()
