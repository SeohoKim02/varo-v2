"""Rendered behaviour of the 재고 운영 Workspace, driven through Streamlit AppTest.

Covers what a user actually sees on the one screen: the state it opens in, the
network, the execution panel, the selection wiring between them, and the fact
that no internal vocabulary or invented number reaches the page.
"""
from __future__ import annotations

import html
import re
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
from pages.workspace import FILTER_EMPTY_MESSAGE
from services.workspace_view import action_qty, money_text, qty_text

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "app_v2.py")
WORKSPACE = "재고 운영"

BANNED_TOKENS = (
    "candidate_id", "data_signature", "scipy", "milp", "replay buffer", "epsilon",
    "Traceback", "session_state", "reason_code", "status_code", "PostgreSQL",
    "SQLite", "schema version", "varo_hybrid_score", "execution-plan-1.0",
)


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class WorkspaceRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_excel_data(get_default_sample_path())
        validation = validate_workbook_data(data)
        pipeline = run_analysis_pipeline(data).to_dict()
        cls.payload = build_applied_state_payload(
            data, validation, pipeline["recommendations"],
            SAMPLE_FILENAME, "샘플 추천 데이터", pipeline,
        )
        cls.plan_items = list(pipeline["execution_plan"]["items"])

    # ---------------------------------------------------------------- helpers
    def _ready_app(self):
        app = AppTest.from_file(APP_PATH, default_timeout=180)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["current_menu"] = WORKSPACE
        app.run()
        return app

    def _blob(self, app) -> str:
        texts = [element.value for element in app.markdown]
        texts += [element.value for element in app.caption]
        return " ".join(str(item) for item in texts)

    def _alerts(self, app) -> str:
        parts = []
        for attribute in ("info", "success", "warning", "error"):
            parts += [element.value for element in getattr(app, attribute)]
        return " ".join(str(item) for item in parts)

    # ------------------------------------------------------------ empty state
    def test_workspace_is_the_landing_page_and_survives_an_empty_workspace(self):
        app = AppTest.from_file(APP_PATH, default_timeout=180)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_menu"], WORKSPACE)
        blob = self._blob(app)
        self.assertIn("재고 운영 Workspace", blob)
        self.assertIn("현재 적용된 데이터가 없습니다", blob)
        # No result numbers are fabricated for an empty workspace.
        for hidden in ("오늘 실행 이동", "예상 순효과", "오늘 권장 이동"):
            self.assertNotIn(hidden, blob)
        button = next(b for b in app.button if b.key == "workspace_primary_action")
        self.assertEqual(button.label, "데이터 불러오기")
        button.click().run()
        self.assertEqual(app.session_state["current_menu"], "데이터 관리")

    def test_analysis_pending_state_offers_the_run_on_this_screen(self):
        app = AppTest.from_file(APP_PATH, default_timeout=180)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["varo_recommendations"] = []
        app.session_state["analysis_result"] = {}
        app.session_state["varo_pipeline_result"] = {}
        app.session_state["analysis_run_required"] = True
        app.session_state["current_menu"] = WORKSPACE
        app.run()
        self.assertFalse(app.exception)
        keys = {b.key for b in app.button}
        self.assertIn("ws_run_analysis_main", keys)
        self.assertIn("ws_go_data_waiting", keys)
        blob = self._blob(app)
        self.assertIn("분석을 실행하면 권장 이동과 네트워크가 여기에 표시됩니다", blob)

    def test_analysis_run_button_produces_a_plan_on_this_screen(self):
        app = AppTest.from_file(APP_PATH, default_timeout=300)
        app.run()
        from services.data_application import load_and_apply
        from tests.fixtures import sample_workbook, workbook_excel_bytes

        state: dict = {}
        load_and_apply(state, workbook_excel_bytes(sample_workbook()), "run.xlsx", "업로드된 추천 결과")
        state["varo_recommendations"] = []
        state["analysis_result"] = {}
        state["varo_pipeline_result"] = {}
        state["analysis_run_required"] = True
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = state.get(key)
        app.session_state["current_menu"] = WORKSPACE
        app.run()
        next(b for b in app.button if b.key == "ws_run_analysis_main").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state["analysis_run_required"])
        self.assertTrue(app.session_state["varo_recommendations"])
        self.assertIn("오늘 권장 이동", self._blob(app))

    def test_stale_result_tells_the_user_to_analyse_again(self):
        app = self._ready_app()
        pipeline = dict(self.payload["varo_pipeline_result"])
        pipeline["candidate_ledger"] = [
            {**record, "data_signature": "analysed-signature"}
            for record in (pipeline.get("candidate_ledger") or [{}])
        ]
        app.session_state["varo_pipeline_result"] = pipeline
        app.session_state["analysis_result"] = pipeline
        app.session_state["data_signature"] = "changed-after-the-run"
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("현재 결과가 최신 데이터 기준이 아닙니다", self._alerts(app))
        self.assertIn("ws_run_analysis_main", {b.key for b in app.button})

    def test_no_executable_move_says_so_and_points_at_the_exclusions(self):
        from services.data_application import load_and_apply
        from tests.fixtures import sample_workbook, workbook_excel_bytes

        workbook = sample_workbook()
        recs = workbook["recommendations"].copy()
        recs["recommended_qty"] = 999999
        workbook["recommendations"] = recs
        state: dict = {}
        load_and_apply(state, workbook_excel_bytes(workbook), "blocked.xlsx", "업로드된 추천 결과")
        app = AppTest.from_file(APP_PATH, default_timeout=300)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = state.get(key)
        app.session_state["current_menu"] = WORKSPACE
        app.run()
        self.assertFalse(app.exception)
        blob = self._blob(app)
        self.assertIn("추천할 이동이 없습니다", blob)
        self.assertNotIn("오늘 권장 이동", blob)
        labels = {item.label for item in app.expander}
        self.assertTrue(any("추천에서 제외된 후보" in label for label in labels))

    # ------------------------------------------------------------ ready state
    def test_ready_workspace_shows_kpis_network_and_execution_panel(self):
        app = self._ready_app()
        self.assertFalse(app.exception)
        blob = self._blob(app)
        for required in (
            "재고 운영 Workspace",
            "오늘 실행 이동", "총 이동 수량", "예상 순효과", "주의 필요",
            "재고 이동 네트워크",
            "오늘 권장 이동",
            "이 이동을 권장하는 이유",
            "위험 · 주의",
        ):
            self.assertIn(required, blob, f"workspace must contain: {required}")
        self.assertIn('class="ws-network-svg"', blob)
        self.assertIn('class="ws-action-qty"', blob)
        # Stores, DCs and at least one arrow-terminated edge are drawn.
        self.assertIn("ws-node-store", blob)
        self.assertIn("ws-arrow-selected", blob)
        # 실행 판단 → 세부 → 검증 → 기록.
        self.assertEqual([tab.label for tab in app.tabs], ["대안 비교", "세부 정보", "검증", "실행 이력"])

    def test_left_panel_holds_status_filters_and_the_plan_list(self):
        app = self._ready_app()
        blob = self._blob(app)
        for required in ("상태", "실행 계획"):
            self.assertIn(required, blob)
        labels = [item.label for item in app.selectbox]
        for required in ("상품", "출발 점포", "도착 점포", "경로 유형"):
            self.assertIn(required, labels)
        checkboxes = [item.label for item in app.checkbox]
        self.assertIn("순효과가 있는 이동만", checkboxes)
        self.assertIn("주의가 필요한 이동만", checkboxes)
        # No algorithm chooser is offered to a normal user.
        for banned in ("VHS", "Greedy", "MILP", "DQN"):
            self.assertNotIn(banned, labels)
        keys = {b.key for b in app.button}
        self.assertIn("ws_go_data", keys)
        # Re-running is available but never the primary call to action on a result.
        rerun = next(b for b in app.button if b.key == "ws_run_analysis_side")
        self.assertEqual(rerun.label, "다시 분석")
        self.assertNotIn("ws_run_analysis_main", keys)

    def test_planned_quantity_is_the_same_in_panel_edge_and_network(self):
        app = self._ready_app()
        blob = self._blob(app)
        selected = next(
            item for item in self.plan_items
            if str(item["route_id"]) == str(app.session_state["selected_route_id"])
        )
        quantity = qty_text(action_qty(selected))
        self.assertIn(f'class="ws-action-qty">{quantity}<', blob)   # right panel
        self.assertIn(f">{quantity}</text>", blob)                   # network edge chip
        net = selected.get("planned_net_benefit")
        self.assertIn(money_text(net), blob)

    def _plan_list(self, app):
        return next(item for item in app.radio if item.key == "ws_plan_pick")

    def test_the_plan_list_is_one_scrolling_control_over_every_move(self):
        app = self._ready_app()
        widget = self._plan_list(app)
        # Every move in the plan is selectable, and only one control does it —
        # no column of per-move buttons that would grow with the plan.
        self.assertEqual(len(widget.options), len(self.plan_items))
        self.assertEqual(
            [b for b in app.button if b.key.startswith(("ws_pick_", "ws_move_"))], [],
        )
        # 출발 → 도착 · 상품: a leg alone is not a move, so the product is part of
        # the line a user scans, not only of the caption below it.
        first = self.plan_items[0]
        self.assertIn(
            f"{first.get('source_name') or first.get('source_id')} → "
            f"{first.get('target_name') or first.get('target_id')} · "
            f"{first.get('product_name')}",
            widget.options,
        )
        self.assertEqual(len(set(widget.options)), len(widget.options), "목록 항목이 중복돼 보입니다")

    def test_selecting_a_move_from_the_list_moves_network_panel_and_tabs_together(self):
        app = self._ready_app()
        first = str(app.session_state["selected_route_id"])
        target = next(item for item in self.plan_items if str(item["route_id"]) != first)
        self._plan_list(app).set_value(str(target["route_id"])).run()
        self.assertFalse(app.exception)

        # One selection id, read by every region of the screen.
        self.assertEqual(app.session_state["selected_route_id"], str(target["route_id"]))
        blob = self._blob(app)
        label = (
            f"{target.get('source_name') or target.get('source_id')} → "
            f"{target.get('target_name') or target.get('target_id')}"
        )
        self.assertIn(f'class="ws-action-route">{label}<', blob)               # right panel
        self.assertIn(f'class="ws-action-qty">{qty_text(action_qty(target))}<', blob)
        self.assertIn("ws-arrow-selected", blob)                                # network
        self.assertIn(f">{qty_text(action_qty(target))}</text>", blob)          # network chip
        # 세부 정보 tab follows the same move.
        rows = self._detail_rows(app)
        self.assertEqual(rows["실행 수량"], qty_text(action_qty(target)))
        self.assertEqual(
            rows["출발 점포"], str(target.get("source_name") or target.get("source_id")),
        )

    def _detail_rows(self, app) -> dict[str, str]:
        """이동 세부 표는 in-DOM HTML 표라서 값이 그대로 읽힌다(가상화 그리드 아님)."""
        blob = self._blob(app)
        rows: dict[str, str] = {}
        for chunk in blob.split("<tr>"):
            cells = re.findall(r"<td>(.*?)</td>", chunk)
            if len(cells) >= 2:
                rows.setdefault(html.unescape(cells[0]), html.unescape(cells[1]))
        return rows

    def test_detail_tab_carries_the_route_detail_values(self):
        app = self._ready_app()
        rows = set(self._detail_rows(app))
        self.assertTrue(rows, "세부 정보 탭에 이동 세부 표가 없습니다")
        for required in (
            "출발 점포", "도착 점포", "경유 DC", "경로 유형", "실행 수량",
            "출발 현재재고", "유지해야 할 재고", "이동 가능 수량", "이동 후 출발 재고",
            "도착 점포 필요량", "예상 비용", "예상 절감", "예상 순효과", "안정성",
            # 미확보 물류 값도 같은 탭에서 0이 아니라 미확보로 읽힌다.
            "실제 도로 거리", "실제 이동 시간", "차량 용량", "실제 운송비",
        ):
            self.assertIn(required, rows, f"세부 정보에 없음: {required}")
        detail = self._detail_rows(app)
        for missing in ("실제 도로 거리", "차량 용량", "실제 운송비"):
            self.assertEqual(detail[missing], "미확보")

    def test_network_scope_can_be_narrowed_and_widened_without_breaking(self):
        app = self._ready_app()
        scope = next(item for item in app.selectbox if item.key == "ws_network_scope")
        for option in ("선택한 이동", "전체 네트워크", "현재 실행계획"):
            scope = next(item for item in app.selectbox if item.key == "ws_network_scope")
            scope.select(option).run()
            self.assertFalse(app.exception, msg=f"{option}: {list(app.exception)}")
            self.assertIn('class="ws-network-svg"', self._blob(app))

    def test_filters_narrow_the_plan_and_keep_the_screen_consistent(self):
        app = self._ready_app()
        product = str(self.plan_items[0]["product_name"])
        widget = next(item for item in app.selectbox if item.key == "ws_filter_product")
        widget.select(product).run()
        self.assertFalse(app.exception)
        blob = self._blob(app)
        self.assertIn(product, blob)
        self.assertIn('class="ws-network-svg"', blob)

    # ------------------------------------------------------------ bottom tabs
    def test_bottom_tabs_carry_alternatives_validation_history_and_detail(self):
        app = self._ready_app()
        blob = self._blob(app)
        self.assertIn("조건이 달라지면", blob)          # what-if, under 대안 비교
        self.assertIn("데이터 상태", blob)              # 세부정보
        self.assertIn("실행 기록", blob)                # 실행 이력 (reused panel)
        self.assertIn("record_execution_plan", {b.key for b in app.button})
        columns = set()
        for element in app.dataframe:
            try:
                columns |= set(element.value.columns)
            except Exception:
                pass
        self.assertIn("검증 항목", columns)
        self.assertIn("구분", columns)                  # alternatives comparison
        self.assertIn("항목", columns)                  # data readiness

    def test_data_status_marks_uncollected_logistics_as_missing_not_zero(self):
        app = self._ready_app()
        readiness = None
        for element in app.dataframe:
            try:
                if "항목" in element.value.columns and "상태" in element.value.columns:
                    readiness = element.value
            except Exception:
                continue
        self.assertIsNotNone(readiness)
        rows = {str(row["항목"]): str(row["상태"]) for _, row in readiness.iterrows()}
        self.assertEqual(rows["차량 용량"], "미확보")
        self.assertEqual(rows["실제 거점간 이동이력"], "미확보")
        self.assertEqual(rows["이동 거리"], "기준값")
        self.assertNotIn("0km", self._blob(app))

    # ------------------------------------------------------------- navigation
    def test_workspace_links_out_to_the_detail_screens(self):
        for key, expected in (
            ("ws_go_data", "데이터 관리"),
            ("ws_open_route_detail", "경로 상세"),
            ("ws_go_validation", "분석 및 검증"),
        ):
            app = self._ready_app()
            next(b for b in app.button if b.key == key).click().run()
            self.assertFalse(app.exception, msg=f"{key}: {list(app.exception)}")
            self.assertEqual(app.session_state["current_menu"], expected)

    def test_selection_made_in_the_workspace_survives_into_route_detail(self):
        app = self._ready_app()
        first = str(app.session_state["selected_route_id"])
        other = next(item for item in self.plan_items if str(item["route_id"]) != first)
        self._plan_list(app).set_value(str(other["route_id"])).run()
        chosen = str(app.session_state["selected_route_id"])
        self.assertNotEqual(chosen, first)
        next(b for b in app.button if b.key == "ws_open_route_detail").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_menu"], "경로 상세")
        self.assertEqual(app.session_state["selected_route_id"], chosen)

    # -------------------------------------------------------------- hygiene
    def test_workspace_never_shows_internal_vocabulary(self):
        app = self._ready_app()
        text = self._blob(app) + " " + self._alerts(app)
        for banned in BANNED_TOKENS:
            self.assertNotIn(banned, text, f"workspace must not expose: {banned}")
        for route_id in (str(item["route_id"]) for item in self.plan_items):
            self.assertNotIn(f">{route_id}<", text)

    def test_top_header_stays_minimal(self):
        app = self._ready_app()
        blob = self._blob(app)
        self.assertIn("VARO V2", blob)
        for banned in ("DQN 학습 전", "지도 미연결", "지도 연결됨", "execution-plan"):
            self.assertNotIn(banned, blob)

    def test_data_and_analysis_status_are_stated_once_above_the_result(self):
        """The top bar carries 데이터 · 분석 상태; the workspace strip must not repeat
        them. The duplicate chip also read '분석 분석 완료' on screen."""
        app = self._ready_app()
        blob = self._blob(app)
        self.assertIn('class="ws-header-title">재고 운영 Workspace<', blob)
        self.assertNotIn('class="ws-header-meta"', blob)
        self.assertNotIn("분석 분석", blob)
        self.assertEqual(blob.count("데이터 적용 완료"), 1)
        # The chip styles that produced the duplicate row are gone, not just unused.
        styles = (ROOT / "styles.py").read_text(encoding="utf-8")
        self.assertNotIn(".ws-chip", styles)

    # ------------------------------------------------- filtered down to nothing
    def _impossible_filter(self, app):
        """Same store as 출발 and 도착 — never a real move, so 0 rows survive."""
        sources = next(item for item in app.selectbox if item.key == "ws_filter_source")
        targets = next(item for item in app.selectbox if item.key == "ws_filter_target")
        shared = next(
            value for value in sources.options[1:] if value in targets.options[1:]
        )
        sources.select(shared).run()
        next(item for item in app.selectbox if item.key == "ws_filter_target").select(shared).run()
        return app

    def test_filtering_everything_out_explains_itself_instead_of_drawing_an_empty_network(self):
        app = self._impossible_filter(self._ready_app())
        self.assertFalse(app.exception)
        blob = self._blob(app)
        self.assertIn(FILTER_EMPTY_MESSAGE, blob)
        # No node-only picture, and no move that contradicts the filter on the right.
        self.assertNotIn('class="ws-network-svg"', blob)
        self.assertNotIn('class="ws-action-qty"', blob)
        self.assertIn("ws_reset_filters", {button.key for button in app.button})

    def test_resetting_the_filters_brings_every_move_back(self):
        app = self._impossible_filter(self._ready_app())
        next(button for button in app.button if button.key == "ws_reset_filters").click().run()
        self.assertFalse(app.exception)
        blob = self._blob(app)
        self.assertIn('class="ws-network-svg"', blob)
        self.assertIn('class="ws-action-qty"', blob)
        for key in ("ws_filter_source", "ws_filter_target", "ws_filter_product"):
            self.assertEqual(app.session_state[key], "전체")

    def test_validation_leads_with_a_verdict_and_keeps_the_numbers_folded_away(self):
        app = self._ready_app()
        blob = self._blob(app)
        for headline in ("계획 제약", "안전재고", "도착 필요 수량", "안정성"):
            self.assertIn(f'class="ws-kpi-title">{headline}<', blob)
        self.assertIn('class="ws-check-value">', blob)
        self.assertIn("검증 상세 보기", [item.label for item in app.expander])

    def test_alternatives_table_hides_internal_columns(self):
        app = self._ready_app()
        columns: set[str] = set()
        for element in app.dataframe:
            try:
                columns |= set(element.value.columns)
            except Exception:
                continue
        self.assertIn("구분", columns)
        for shown in ("경로", "수량", "예상 비용", "예상 순효과", "안정성"):
            self.assertIn(shown, columns)
        for hidden in ("route_id", "선택", "예상 효과"):
            self.assertNotIn(hidden, columns)

    # ------------------------------------------------------------ layout guard
    def test_workspace_cards_are_content_driven_not_fixed_height(self):
        styles = (ROOT / "styles.py").read_text(encoding="utf-8")
        self.assertIn(".ws-action-card", styles)
        self.assertIn(".ws-kpi", styles)
        # Wrapping, not clipping: no fixed height and no nowrap on workspace text.
        for banned in (
            ".ws-kpi {{ height:", ".ws-action-card {{ height:",
            ".ws-kpi-value {{ white-space: nowrap",
        ):
            self.assertNotIn(banned, styles)
        self.assertIn("aspect-ratio: 940 / 620", styles)
        self.assertIn("@media (max-width: 1400px)", styles)
        self.assertIn("@media (max-width: 1150px)", styles)


if __name__ == "__main__":
    unittest.main()
