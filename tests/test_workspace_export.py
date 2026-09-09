"""재고 운영 화면에서 끝나는 실행 계획 내보내기, 그리고 그 결과 남는 것들.

이 파일이 지키는 규칙:

* 사용자용 export는 **실행 계획**이다. 수량은 ``planned_qty``(실행 수량)이고,
  순서는 Varo가 정한 실행 우선순위 그대로다.
* 내부 식별자·점수 구성요소·solver 값은 사용자 export에 들어가지 않는다.
* 아직 수집 중인 물류 사실(거리·시간·차량용량·실제 운송비)은 0이 아니라 빈칸 또는
  미확보로 남는다.
* 일상 작업(분석 → 선택 → 비교 → 검증 → 기록 → 내보내기)은 재고 운영 한 화면에서
  끝난다. 호환용 route는 사라지지 않지만 방문할 필요가 없어야 한다.
"""
from __future__ import annotations

import io
import re
import time
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

try:
    from streamlit.testing.v1 import AppTest

    _APPTEST_AVAILABLE = True
except Exception:  # pragma: no cover - older streamlit
    _APPTEST_AVAILABLE = False

from components.navigation import LEGACY_MENU_ITEMS, MENU_ITEMS, PRIMARY_MENU_ITEMS
from services import export_service
from services.analysis_pipeline import run_analysis_pipeline
from services.app_state import CANONICAL_DATA_KEYS, build_applied_state_payload
from services.data_loader import SAMPLE_FILENAME, get_default_sample_path, load_excel_data
from services.data_validator import validate_workbook_data
from services.execution_plan import planned_recommendations
from services.workspace_view import action_qty

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "app_v2.py")
WORKSPACE = "재고 운영"

# 사용자 실행 계획 export에 절대로 들어가면 안 되는 내부 필드/값.
INTERNAL_FIELDS = (
    "route_id", "candidate_id", "data_signature", "vhs_score", "greedy_rank",
    "dqn", "pareto", "confidence_score", "plan_id", "session", "score",
)


def _sample_pipeline() -> dict:
    data = load_excel_data(get_default_sample_path())
    return run_analysis_pipeline(data).to_dict()


class ExecutionPlanExportFrameTests(unittest.TestCase):
    """§4·§5 — 사용자 export의 내용."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = _sample_pipeline()
        cls.items = planned_recommendations(cls.pipeline)

    def test_quantity_is_the_execution_quantity_not_the_candidate_quantity(self):
        item = {
            "route_id": "R1", "product_name": "고등어", "source_name": "A", "target_name": "B",
            "route_type": "DIRECT", "recommended_qty": 40, "planned_qty": 25,
        }
        frame = export_service.execution_plan_export_frame([item])
        self.assertEqual(int(frame.loc[0, "실행 수량"]), 25)
        self.assertNotIn(40, list(frame.loc[0]))

    def test_real_plan_rows_match_the_planned_quantity_one_for_one(self):
        frame = export_service.execution_plan_export_frame(self.items)
        self.assertEqual(len(frame), len(self.items))
        for index, item in enumerate(self.items):
            self.assertEqual(
                float(frame.loc[index, "실행 수량"]), float(action_qty(item)),
                f"{index}행 실행 수량이 실행 계획과 다릅니다",
            )

    def test_export_keeps_the_plan_order_and_never_re_sorts(self):
        frame = export_service.execution_plan_export_frame(self.items)
        self.assertEqual(list(frame["순위"]), list(range(1, len(self.items) + 1)))
        reversed_items = list(reversed(self.items))
        reversed_frame = export_service.execution_plan_export_frame(reversed_items)
        self.assertEqual(
            list(reversed_frame["출발 점포"]),
            [str(item.get("source_name") or item.get("source_id")) for item in reversed_items],
            "export가 스스로 정렬하면 안 된다 (표시 순서를 그대로 따른다)",
        )

    def test_columns_are_the_operator_set_only(self):
        frame = export_service.execution_plan_export_frame(self.items)
        self.assertEqual(list(frame.columns), [
            "순위", "출발 점포", "도착 점포", "상품", "실행 수량", "경로", "경유 DC",
            "예상 비용", "예상 절감", "예상 순효과", "안정성", "주요 주의", "데이터 기준",
        ])

    def test_no_internal_identifier_or_score_reaches_the_user_export(self):
        frame = export_service.execution_plan_export_frame(self.items)
        blob = frame.to_csv(index=False).lower()
        for field in INTERNAL_FIELDS:
            self.assertNotIn(field.lower(), blob, f"사용자 export에 {field}가 있으면 안 된다")
        for item in self.items:
            self.assertNotIn(str(item["route_id"]).lower(), blob)

    def test_uncollected_logistics_never_become_zero(self):
        """§6 — 거리·시간·차량용량·실제 운송비는 0이 아니라 미확보로 남는다."""
        frame = export_service.execution_plan_export_frame(self.items)
        for banned in ("거리", "이동 시간", "차량", "운송비"):
            self.assertNotIn(banned, list(frame.columns))
        basis = set(frame["데이터 기준"])
        self.assertEqual(len(basis), 1)
        self.assertIn(export_service.NOT_COLLECTED, basis.pop())

    def test_data_basis_flips_when_a_real_measurement_arrives(self):
        measured = {
            "route_id": "R1", "product_name": "고등어", "source_name": "A", "target_name": "B",
            "route_type": "DIRECT", "planned_qty": 10, "actual_distance_km": 4.7,
        }
        frame = export_service.execution_plan_export_frame([measured])
        self.assertNotIn(export_service.NOT_COLLECTED, str(frame.loc[0, "데이터 기준"]))

    def test_missing_money_stays_blank_rather_than_zero(self):
        item = {"route_id": "R1", "product_name": "P", "source_name": "A",
                "target_name": "B", "route_type": "DIRECT", "planned_qty": 3}
        frame = export_service.execution_plan_export_frame([item])
        for column in ("예상 비용", "예상 절감", "예상 순효과"):
            self.assertEqual(frame.loc[0, column], "", f"{column}은 빈칸이어야 한다")


class ExecutionPlanExportFileTests(unittest.TestCase):
    """§19·§20 — 파일명과 인코딩."""

    @classmethod
    def setUpClass(cls):
        cls.items = planned_recommendations(_sample_pipeline())

    def test_csv_is_utf8_bom_so_korean_opens_in_excel(self):
        payload = export_service.execution_plan_csv_bytes(self.items)
        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        text = payload.decode("utf-8-sig")
        self.assertTrue(text.startswith("순위,출발 점포,도착 점포,상품,실행 수량"))

    def test_excel_has_one_readable_execution_plan_sheet(self):
        payload = export_service.execution_plan_excel_bytes(self.items)
        book = pd.read_excel(io.BytesIO(payload), sheet_name=None)
        self.assertEqual(list(book), [export_service.EXECUTION_PLAN_SHEET])
        sheet = book[export_service.EXECUTION_PLAN_SHEET]
        self.assertEqual(len(sheet), len(self.items))
        self.assertIn("실행 수량", sheet.columns)

    def test_filenames_are_user_friendly_and_dated(self):
        csv_name = export_service.execution_plan_filename("csv", date(2026, 9, 9))
        xlsx_name = export_service.execution_plan_filename("xlsx", date(2026, 9, 9))
        self.assertEqual(csv_name, "varo_execution_plan_20260909.csv")
        self.assertEqual(xlsx_name, "varo_execution_plan_20260909.xlsx")
        for name in (csv_name, xlsx_name):
            for banned in ("candidate", "dump", "temp", "result"):
                self.assertNotIn(banned, name)
        # 인자 없이 부르면 오늘 날짜.
        self.assertTrue(
            export_service.execution_plan_filename("csv").endswith(
                f"{date.today():%Y%m%d}.csv"
            )
        )

    def test_empty_plan_exports_without_crashing(self):
        csv_payload = export_service.execution_plan_csv_bytes([])
        self.assertIn("순위", csv_payload.decode("utf-8-sig"))
        self.assertTrue(export_service.execution_plan_excel_bytes([]))

    def test_one_move_and_fifty_moves_both_export(self):
        one = export_service.execution_plan_export_frame(self.items[:1])
        self.assertEqual(len(one), 1)
        fifty = []
        for index in range(50):
            item = dict(self.items[index % len(self.items)])
            item["route_id"] = f"UI-{index}"
            fifty.append(item)
        started = time.perf_counter()
        csv_payload = export_service.execution_plan_csv_bytes(fifty)
        xlsx_payload = export_service.execution_plan_excel_bytes(fifty)
        elapsed = time.perf_counter() - started
        self.assertEqual(len(csv_payload.decode("utf-8-sig").strip().splitlines()), 51)
        self.assertEqual(len(pd.read_excel(io.BytesIO(xlsx_payload))), 50)
        # 화면이 멈춰 보이지 않을 정도로 빨라야 한다(매 rerun마다 만들어진다).
        self.assertLess(elapsed, 3.0, "50건 export가 UI를 멈추게 할 만큼 느립니다")

    def test_the_research_export_is_untouched(self):
        """§5 — 연구용 export는 그대로 보존한다."""
        frame = export_service.recommendations_export_frame(self.items)
        for column in ("route_id", "VHS(재계산)", "추천 등급", "거리(km)"):
            self.assertIn(column, frame.columns)


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class WorkspaceExportRenderTests(unittest.TestCase):
    """§3·§18 — 화면에서 실제로 눌리는 자리."""

    @classmethod
    def setUpClass(cls):
        data = load_excel_data(get_default_sample_path())
        pipeline = run_analysis_pipeline(data).to_dict()
        cls.pipeline = pipeline
        cls.payload = build_applied_state_payload(
            data, validate_workbook_data(data), pipeline["recommendations"],
            SAMPLE_FILENAME, "샘플 추천 데이터", pipeline,
        )
        cls.plan_items = list(pipeline["execution_plan"]["items"])

    def _app(self, payload=None):
        app = AppTest.from_file(APP_PATH, default_timeout=240)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = (payload or self.payload).get(key)
        app.session_state["current_menu"] = WORKSPACE
        app.run()
        return app

    def _sized_payload(self, size: int):
        """UI fixture: 같은 실행 계획을 size건으로 늘린다(알고리즘 결과는 그대로)."""
        pipeline = {key: value for key, value in self.pipeline.items()}
        plan = dict(pipeline["execution_plan"])
        items = []
        for index in range(size):
            item = dict(self.plan_items[index % len(self.plan_items)])
            item["route_id"] = f"UI-{index}"
            item["plan_rank"] = index + 1
            items.append(item)
        plan["items"] = items
        pipeline["execution_plan"] = plan
        payload = dict(self.payload)
        payload["analysis_result"] = pipeline
        payload["varo_pipeline_result"] = pipeline
        payload["selected_route_id"] = "UI-0"
        return payload

    def test_both_downloads_live_on_the_workspace(self):
        app = self._app()
        self.assertFalse(app.exception)
        labels = [element.label for element in app.get("download_button")]
        self.assertEqual(labels, ["CSV", "Excel"], "재고 운영에는 이 두 개만 있으면 된다")
        self.assertIn("내보내기", " ".join(item.label for item in app.expander))
        source = (ROOT / "pages" / "workspace.py").read_text(encoding="utf-8")
        self.assertIn("execution_plan_csv_bytes", source)
        self.assertIn("execution_plan_excel_bytes", source)
        self.assertIn("execution_plan_filename", source)

    def test_the_export_follows_the_filtered_list(self):
        app = self._app()
        products = next(item for item in app.selectbox if item.key == "ws_filter_product")
        chosen = products.options[1]
        products.select(chosen).run()
        self.assertFalse(app.exception)
        visible = [item for item in self.plan_items
                   if str(item.get("product_name")) == str(chosen)]
        label = next(
            item.label for item in app.expander if item.label.startswith("내보내기")
        )
        self.assertEqual(label, f"내보내기 ({len(visible)}건)")

    def test_fifty_moves_still_render_list_network_panel_and_export(self):
        app = self._app(self._sized_payload(50))
        self.assertFalse(app.exception)
        radio = next(item for item in app.radio if item.key == "ws_plan_pick")
        self.assertEqual(len(radio.options), 50)
        self.assertEqual(
            [element.label for element in app.get("download_button")], ["CSV", "Excel"],
        )
        blob = " ".join(str(item.value) for item in app.markdown)
        self.assertIn('class="ws-network-svg"', blob)
        self.assertIn('class="ws-action-qty"', blob)

    def test_recording_a_plan_starts_from_the_decision_panel(self):
        """§14 — 오른쪽 패널에는 기록 버튼 하나, 상세 form은 실행 이력 탭."""
        app = self._app()
        keys = {b.key for b in app.button}
        self.assertIn("ws_record_plan", keys)
        self.assertNotIn("record_execution_plan", keys, "한 화면에 기록 버튼이 둘이면 안 된다")
        button = next(b for b in app.button if b.key == "ws_record_plan")
        self.assertEqual(button.label, "이 계획 기록")


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class WorkspaceHasNoLegacyDependencyTests(unittest.TestCase):
    """§10·§24·§25 — 일상 작업에 호환용 화면이 필요 없다."""

    @classmethod
    def setUpClass(cls):
        data = load_excel_data(get_default_sample_path())
        pipeline = run_analysis_pipeline(data).to_dict()
        cls.payload = build_applied_state_payload(
            data, validate_workbook_data(data), pipeline["recommendations"],
            SAMPLE_FILENAME, "샘플 추천 데이터", pipeline,
        )

    def _app(self):
        app = AppTest.from_file(APP_PATH, default_timeout=240)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["current_menu"] = WORKSPACE
        app.run()
        return app

    def test_the_workspace_offers_no_jump_to_an_absorbed_screen(self):
        app = self._app()
        for button in app.button:
            for banned in ("추천 실행", "경로 상세", "예전", "legacy"):
                self.assertNotIn(banned, button.label, f"{button.key}: 옛 이동 버튼")

    def test_the_whole_daily_flow_is_reachable_without_leaving_the_workspace(self):
        """분석 → 선택 → 이해 → 비교 → 검증 → 기록 → 내보내기."""
        app = self._app()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_menu"], WORKSPACE)
        # 실행 결정
        self.assertTrue([r for r in app.radio if r.key == "ws_plan_pick"])
        blob = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("오늘 권장 이동", blob)
        self.assertIn('class="ws-network-svg"', blob)
        # 비교 · 세부 · 검증 · 기록
        self.assertEqual(
            [tab.label for tab in app.tabs][:4],
            ["대안 비교", "세부 정보", "검증", "실행 이력"],
        )
        # 기록 · 내보내기
        self.assertIn("ws_record_plan", {b.key for b in app.button})
        self.assertEqual(
            [element.label for element in app.get("download_button")], ["CSV", "Excel"],
        )
        # 그리고 이 과정에서 호환용 화면은 한 번도 열리지 않는다.
        self.assertEqual(app.session_state["current_menu"], WORKSPACE)

    def test_compatibility_routes_are_preserved_not_deleted(self):
        import router

        for menu in LEGACY_MENU_ITEMS:
            self.assertIn(menu, MENU_ITEMS)
            self.assertNotIn(menu, PRIMARY_MENU_ITEMS)
            self.assertIn(menu, router._PAGE_RENDERERS)
        app = self._app()
        for menu in LEGACY_MENU_ITEMS:
            app.session_state["current_menu"] = menu
            app.run()
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")


class VisibleTerminologyTests(unittest.TestCase):
    """§22·§23 — 사용자 화면 문자열에 옛 표현이 남지 않는다."""

    USER_SURFACES = (
        "pages/workspace.py",
        "components/workspace_network.py",
        "components/execution_history_panel.py",
        "components/candidate_detail.py",
        "components/state_banner.py",
    )

    def _visible_strings(self, path: Path) -> list[str]:
        """따옴표 안 문자열 중 한글이 들어간 것만(주석·docstring은 제외)."""
        source = path.read_text(encoding="utf-8")
        source = re.sub(r'""".*?"""', "", source, flags=re.S)
        source = re.sub(r"(?m)^\s*#.*$", "", source)
        source = re.sub(r"(?m)#\s.*$", "", source)
        return [
            text for text in re.findall(r'"([^"\n]*)"|\'([^\'\n]*)\'', source)
            for text in text if re.search(r"[가-힣]", text)
        ]

    def test_no_legacy_wording_on_the_screens_a_user_reads(self):
        banned = ("추천 후보", "candidate", "경로 상세", "옛 화면", "예전 화면", "Workspace", "legacy")
        for name in self.USER_SURFACES:
            for text in self._visible_strings(ROOT / name):
                for word in banned:
                    self.assertNotIn(word, text, f"{name}: '{text}'에 '{word}'가 남아 있습니다")

    def test_the_everyday_menu_reads_in_the_product_vocabulary(self):
        from components import navigation

        self.assertEqual(
            [navigation.menu_label(item) for item in PRIMARY_MENU_ITEMS],
            ["재고 운영", "데이터 관리", "분석 및 검증", "운영 시뮬레이션"],
        )
        # Route keys stay Korean and unchanged; only what a user reads is checked.
        visible = [navigation.menu_label(item) for item in MENU_ITEMS]
        visible += list(navigation.MENU_HINTS.values())
        visible += [navigation.LEGACY_GROUP_LABEL, navigation.LEGACY_GROUP_NOTE]
        for text in visible:
            for word in ("추천 후보", "candidate", "경로 상세", "옛 화면", "예전 화면",
                         "Workspace", "legacy"):
                self.assertNotIn(word, text, f"메뉴 문구 '{text}'에 '{word}'가 남아 있습니다")


class SidebarWidthTests(unittest.TestCase):
    """§15 — 사이드바를 펼쳐도 중앙 네트워크가 좁아지지 않는다."""

    def _narrow_block(self) -> tuple[int, str]:
        """The narrow-desktop media block, found by its own rule, and its bound.

        The breakpoint is looked up rather than written in, because it is a
        measured value: it has to cover every window in which an expanded sidebar
        would squeeze the centre column below what 1366 already gets.
        """
        styles = (ROOT / "styles.py").read_text(encoding="utf-8")
        # Anchor on the declaration, not the bare number (a comment mentions it
        # too), and never let the span run across an intervening @media — that
        # would credit a narrower block with a rule living in a wider one.
        match = re.search(
            r"@media \(max-width: (\d+)px\) \{\{(?:(?!@media)[\s\S])*?196px !important",
            styles,
        )
        self.assertIsNotNone(match, "펼친 사이드바를 좁히는 미디어 블록을 찾지 못했습니다")
        bound = int(match.group(1))
        body = styles.split(f"@media (max-width: {bound}px)", 1)[1]
        return bound, body.split("@media", 1)[0]

    def test_narrow_desktops_get_a_narrower_sidebar_not_smaller_text(self):
        _bound, block = self._narrow_block()
        # 접힌 사이드바는 건드리지 않는다: 펼친 상태만 좁힌다.
        self.assertIn('section[data-testid="stSidebar"][aria-expanded="true"]', block)
        self.assertIn("196px", block)
        # 글자를 줄이는 방식이 아니다.
        for banned in ("font-size: 9px", "font-size: 8px", "font-size: 10px"):
            self.assertNotIn(banned, block)
        styles = (ROOT / "styles.py").read_text(encoding="utf-8")
        self.assertNotIn("overflow-x: scroll", styles)

    def test_the_rule_reaches_every_window_an_open_sidebar_would_squeeze(self):
        """1450은 잘못된 경계였다: 1600에서 사이드바를 펼치면 중앙 열이 569px까지
        좁아져 1366(646px)보다 더 나빠지고 모든 라벨이 10px 아래로 떨어졌다."""
        bound, _block = self._narrow_block()
        self.assertGreaterEqual(bound, 1600)

    def test_the_three_columns_share_what_is_left_after_the_gaps(self):
        """퍼센트를 그대로 쓰면 열 간격만큼 넘쳐 우측 실행 패널이 화면 밖으로 나간다."""
        _bound, block = self._narrow_block()
        for ratio in ("flex: 20 1 0%", "flex: 60 1 0%"):
            self.assertIn(ratio, block)
        self.assertNotIn("flex: 0 0 21%", block)

    def test_the_css_hook_and_the_page_container_use_one_constant(self):
        from pages.workspace import MAIN_ROW_KEY
        from styles import WORKSPACE_MAIN_ROW_KEY

        self.assertEqual(MAIN_ROW_KEY, WORKSPACE_MAIN_ROW_KEY)
        styles = (ROOT / "styles.py").read_text(encoding="utf-8")
        self.assertIn(f".st-key-{{main_row}}", styles)
        source = (ROOT / "pages" / "workspace.py").read_text(encoding="utf-8")
        self.assertIn("st.container(key=MAIN_ROW_KEY)", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
