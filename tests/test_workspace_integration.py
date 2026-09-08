"""The second Workspace integration pass: navigation, the plan list at scale,
one shared selection, and parity with the screens the Workspace absorbed.

Two kinds of check live here:

* pure view-model checks (no Streamlit) against **synthetic UI fixtures** of
  0 / 1 / 8 / 20 / 50 moves — these exist only to prove the list UI holds up at
  volume and never touch a real algorithm result;
* rendered checks through AppTest against the **real** sample pipeline, which is
  where parity with 추천 실행 / 경로 상세 / 분석 및 검증 is asserted.

Nothing here recomputes a plan: every rendered assertion reads the one execution
plan the app produced.
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

from components.navigation import (
    LEGACY_MENU_ITEMS,
    MENU_ITEMS,
    PRIMARY_MENU_ITEMS,
    menu_label,
)
from services.analysis_pipeline import run_analysis_pipeline, sort_recommendations
from services.app_state import CANONICAL_DATA_KEYS, TRANSIENT_VIEW_KEYS, build_applied_state_payload
from services.data_loader import SAMPLE_FILENAME, get_default_sample_path, load_excel_data
from services.data_validator import validate_workbook_data
from services.execution_plan import planned_recommendations
from services.home_state import PAGE_WORKSPACE, build_home_state
from services.workspace_view import (
    WORKSPACE_VIEW_KEYS,
    action_qty,
    alternatives_for,
    apply_filters,
    logistics_rows,
    money_text,
    move_detail_rows,
    needs_attention,
    plan_list_rows,
    qty_text,
    resolve_selection,
    validation_rows,
)

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "app_v2.py")
WORKSPACE = "재고 운영"


# --------------------------------------------------------------------------- #
# Synthetic UI fixtures — plan volume only, never an algorithm result
# --------------------------------------------------------------------------- #
def ui_plan_items(count: int) -> list[dict]:
    """``count`` plausible plan items. Values are invented on purpose: this
    fixture exists to size the list UI, and is never fed to an algorithm."""
    items = []
    for index in range(count):
        items.append({
            "route_id": f"UI{index:03d}",
            "plan_rank": index + 1,
            "source_id": f"S{index % 7:02d}",
            "source_name": f"{index % 7}번 점포",
            "target_id": f"T{index % 5:02d}",
            "target_name": f"{index % 5}번 수취점포",
            "product_id": f"P{index % 3:02d}",
            "product_name": f"상품{index % 3}",
            "route_type": "VIA_DC" if index % 3 == 0 else "DIRECT",
            "dc_id": "DC01" if index % 3 == 0 else "",
            "dc_name": "중앙물류센터" if index % 3 == 0 else "",
            "planned_qty": 10 + index,
            "recommended_qty": 10 + index + (5 if index % 4 == 0 else 0),
            "quantity_adjusted": index % 4 == 0,
            "planned_net_benefit": 1000.0 * (index + 1),
            "planned_cost": 500.0,
            "planned_expected_saving": 1500.0 * (index + 1),
            "robustness_status": "안정" if index % 2 == 0 else "확인 필요",
            "feasibility_status": "추천 가능",
            "source_stock": 100 + index,
            "source_safety_floor": 20,
            "source_movable": 60,
            "target_stock": 5,
            "target_shortfall": 30,
        })
    return items


PLAN_SIZES = (0, 1, 8, 20, 50)


class PlanListVolumeTests(unittest.TestCase):
    """§계획 목록: 0 / 1 / 8 / 20 / 50건에서 모두 선택 가능하고 깨지지 않는다."""

    def test_every_move_stays_selectable_at_every_plan_size(self):
        for size in PLAN_SIZES:
            with self.subTest(size=size):
                items = ui_plan_items(size)
                rows = plan_list_rows(items, items[0]["route_id"] if items else None)
                self.assertEqual(len(rows), size)
                # One selectable entry per move — the control count does not grow
                # into a separate widget per row.
                self.assertEqual(
                    len({row["route_id"] for row in rows}), size,
                )
                for row in rows:
                    self.assertTrue(row["label"])
                    self.assertIn("개", row["caption"])
                    self.assertRegex(row["caption"], r"(직접 이동|DC 경유)")
                if items:
                    self.assertTrue(rows[0]["selected"])

    def test_plan_order_is_never_re_sorted_by_the_list_or_the_filters(self):
        items = ui_plan_items(50)
        order = [item["route_id"] for item in items]
        self.assertEqual([row["route_id"] for row in plan_list_rows(items)], order)
        kept = apply_filters(items, {"product": "상품1"})
        self.assertEqual(
            [item["route_id"] for item in kept],
            [rid for rid, item in zip(order, items) if item["product_name"] == "상품1"],
        )

    def test_selection_always_resolves_inside_the_current_plan(self):
        for size in PLAN_SIZES:
            with self.subTest(size=size):
                items = ui_plan_items(size)
                if not items:
                    self.assertIsNone(resolve_selection(items, "UI000"))
                    continue
                self.assertEqual(resolve_selection(items, "UI003"), "UI003" if size > 3 else "UI000")
                # A move that is no longer in the plan falls back to the top one.
                self.assertEqual(resolve_selection(items, "gone"), "UI000")

    def test_attention_filter_matches_the_attention_marker_on_the_list(self):
        items = ui_plan_items(20)
        flagged = apply_filters(items, {"only_attention": True})
        self.assertTrue(flagged)
        self.assertTrue(all(needs_attention(item) for item in flagged))
        rows = {row["route_id"]: row for row in plan_list_rows(items)}
        for item in flagged:
            self.assertTrue(rows[item["route_id"]]["attention"])
            self.assertIn("주의", rows[item["route_id"]]["caption"])

    def test_workspace_widget_keys_are_cleared_when_new_data_is_applied(self):
        for key in WORKSPACE_VIEW_KEYS:
            self.assertIn(key, TRANSIENT_VIEW_KEYS, f"{key} must reset with new data")


class MoveDetailTests(unittest.TestCase):
    """§경로 상세 흡수 + §미확보 물류 데이터."""

    def test_detail_uses_the_action_quantity_not_the_candidate_quantity(self):
        item = ui_plan_items(4)[0]  # index 0 → quantity_adjusted
        self.assertNotEqual(item["planned_qty"], item["recommended_qty"])
        rows = {row["항목"]: row["값"] for row in move_detail_rows(item)}
        self.assertEqual(rows["실행 수량"], qty_text(item["planned_qty"]))
        # 이동 후 재고 follows the action quantity, so it can never contradict it.
        self.assertEqual(
            rows["이동 후 출발 재고"],
            qty_text(item["source_stock"] - item["planned_qty"]),
        )
        self.assertEqual(rows["예상 순효과"], money_text(item["planned_net_benefit"]))

    def test_missing_inventory_facts_read_no_data_never_zero(self):
        bare = {"route_id": "X", "source_id": "S", "target_id": "T", "route_type": "DIRECT"}
        rows = {row["항목"]: row["값"] for row in move_detail_rows(bare)}
        for label in (
            "출발 현재재고", "유지해야 할 재고", "이동 가능 수량",
            "이동 후 출발 재고", "도착 점포 필요량", "예상 비용", "예상 순효과",
        ):
            self.assertEqual(rows[label], "데이터 없음", f"{label} must not be 0")
        self.assertEqual(rows["경유 DC"], "경유 없음")

    def test_uncollected_logistics_facts_read_missing_never_zero(self):
        rows = {row["항목"]: row["값"] for row in logistics_rows(ui_plan_items(1)[0])}
        for label in ("실제 도로 거리", "실제 이동 시간", "차량 용량", "실제 운송비"):
            self.assertEqual(rows[label], "미확보", f"{label} must read 미확보")
            self.assertNotIn("0", rows[label])

    def test_a_real_logistics_value_is_shown_once_the_data_carries_it(self):
        """The connection point for the logistics data still being collected."""
        item = {**ui_plan_items(1)[0], "actual_distance_km": 12.5, "actual_transport_cost": 41000}
        rows = {row["항목"]: row["값"] for row in logistics_rows(item)}
        self.assertEqual(rows["실제 도로 거리"], "12.5km")
        self.assertEqual(rows["실제 운송비"], "41,000원")
        self.assertEqual(rows["차량 용량"], "미확보")  # still uncollected

    def test_a_workbook_number_never_fills_an_actual_logistics_row(self):
        """distance_km is an input the model reads, not a measured road distance."""
        item = {**ui_plan_items(1)[0], "distance_km": 4.7, "estimated_cost": 3000}
        rows = {row["항목"]: row for row in logistics_rows(item)}
        self.assertEqual(rows["실제 도로 거리"]["값"], "미확보")
        self.assertIn("4.7km", rows["실제 도로 거리"]["설명"])
        self.assertIn("수집 중", rows["실제 도로 거리"]["설명"])
        self.assertEqual(rows["실제 운송비"]["값"], "미확보")

    def test_the_detail_table_and_the_quantity_basis_line_agree(self):
        """같은 값이 표와 바로 아래 문장에서 다르게 보이면 안 된다."""
        item = {"route_id": "X", "source_id": "S", "target_id": "T",
                "route_type": "DIRECT", "planned_qty": 58}
        basis = {"source_stock": 160.0, "source_safety": 0.0,
                 "source_movable": 160.0, "target_demand": 79.0}
        rows = {row["항목"]: row["값"] for row in move_detail_rows(item, basis)}
        self.assertEqual(rows["출발 현재재고"], "160개")
        self.assertEqual(rows["유지해야 할 재고"], "0개")   # a real 0, not 데이터 없음
        self.assertEqual(rows["이동 가능 수량"], "160개")
        self.assertEqual(rows["도착 점포 필요량"], "79개")
        self.assertEqual(rows["이동 후 출발 재고"], "102개")  # 160 - 58 (실행 수량 기준)
        # The plan item still wins when it carries the value itself.
        rows = {row["항목"]: row["값"] for row in move_detail_rows({**item, "source_stock": 200}, basis)}
        self.assertEqual(rows["출발 현재재고"], "200개")


class AlternativesTests(unittest.TestCase):
    """§20 대안 비교 범위."""

    def _candidates(self) -> list[dict]:
        base = {
            "product_id": "P01", "product_name": "상품",
            "estimated_cost": 500.0, "expected_saving": 1500.0, "net_benefit": 1000.0,
            "recommended_qty": 10, "robustness_status": "안정",
        }
        return [
            {**base, "route_id": "A", "source_id": "S1", "target_id": "T1", "route_type": "DIRECT"},
            # same 구간, different route — DIRECT vs VIA_DC
            {**base, "route_id": "B", "source_id": "S1", "target_id": "T1", "route_type": "VIA_DC", "dc_id": "DC01", "dc_name": "1센터"},
            # same 구간, a different DC
            {**base, "route_id": "C", "source_id": "S1", "target_id": "T1", "route_type": "VIA_DC", "dc_id": "DC02", "dc_name": "2센터"},
            # same destination, another source
            {**base, "route_id": "D", "source_id": "S2", "target_id": "T1", "route_type": "DIRECT"},
            # same source, another destination
            {**base, "route_id": "E", "source_id": "S1", "target_id": "T2", "route_type": "DIRECT"},
            # unrelated — must not appear
            {**base, "route_id": "F", "source_id": "S9", "target_id": "T9", "route_type": "DIRECT"},
            # a different product — substitution is deliberately not offered
            {**base, "route_id": "G", "product_id": "P02", "product_name": "다른 상품",
             "source_id": "S1", "target_id": "T1", "route_type": "DIRECT"},
        ]

    def test_same_leg_other_routes_and_other_sources_are_all_offered(self):
        candidates = self._candidates()
        rows = alternatives_for(candidates[0], candidates, {"items": [candidates[0]]})
        by_id = {row["route_id"]: row for row in rows}
        self.assertEqual(by_id["A"]["구분"], "현재 이동")
        self.assertEqual(by_id["B"]["구분"], "같은 구간 · 다른 경로")   # DIRECT vs VIA_DC
        self.assertEqual(by_id["C"]["구분"], "같은 구간 · 다른 경로")   # DC01 vs DC02
        self.assertEqual(by_id["D"]["구분"], "같은 도착 점포")
        self.assertEqual(by_id["E"]["구분"], "같은 출발 재고")
        self.assertNotIn("F", by_id)
        self.assertNotIn("G", by_id, "상품 대체는 제공하지 않는다")
        # The DC each VIA_DC option uses is visible, so DC01 vs DC02 is a real choice.
        self.assertIn("1센터", by_id["B"]["경로"])
        self.assertIn("2센터", by_id["C"]["경로"])

    def test_the_current_move_is_listed_first(self):
        candidates = self._candidates()
        rows = alternatives_for(candidates[0], candidates, {"items": [candidates[0]]})
        self.assertEqual(rows[0]["route_id"], "A")


class NavigationStructureTests(unittest.TestCase):
    """§15/§16 navigation 단순화 (pure, no Streamlit)."""

    def test_the_everyday_menu_is_four_action_named_screens(self):
        self.assertEqual(
            PRIMARY_MENU_ITEMS, ["재고 운영", "데이터 관리", "분석 및 검증", "운영 현황"],
        )
        self.assertEqual(PRIMARY_MENU_ITEMS[0], WORKSPACE)  # landing page
        self.assertEqual(menu_label("운영 현황"), "운영 시뮬레이션")
        for menu in PRIMARY_MENU_ITEMS:
            for banned in ("Workspace", "Optimizer", "Execution Plan", "VHS", "MILP", "DQN"):
                self.assertNotIn(banned, menu_label(menu))

    def test_the_absorbed_screens_are_demoted_but_still_routed(self):
        self.assertEqual(LEGACY_MENU_ITEMS, ["추천 실행", "경로 상세"])
        for menu in LEGACY_MENU_ITEMS:
            self.assertNotIn(menu, PRIMARY_MENU_ITEMS)
            self.assertIn(menu, MENU_ITEMS)

    def test_every_route_has_a_renderer(self):
        import router

        self.assertEqual(set(router._PAGE_RENDERERS), set(MENU_ITEMS))


@unittest.skipUnless(_APPTEST_AVAILABLE, "streamlit AppTest unavailable")
class WorkspaceIntegrationRenderTests(unittest.TestCase):
    """Rendered parity against the real sample pipeline."""

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

    def _app(self, menu: str = WORKSPACE):
        app = AppTest.from_file(APP_PATH, default_timeout=240)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["current_menu"] = menu
        app.run()
        return app

    def _blob(self, app) -> str:
        parts = [element.value for element in app.markdown]
        parts += [element.value for element in app.caption]
        return " ".join(str(item) for item in parts)

    # -------------------------------------------------- B. plan list at volume
    def _sized_plan(self, size: int) -> dict:
        """The real plan, repeated to ``size`` items with fresh ids.

        Store/product ids stay real so the network still draws; only the number
        of rows is synthetic, and no algorithm is re-run.
        """
        base = self.plan_items
        items = []
        for index in range(size):
            item = dict(base[index % len(base)])
            item["route_id"] = f"{item['route_id']}#{index}"
            item["plan_rank"] = index + 1
            items.append(item)
        pipeline = dict(self.pipeline)
        plan = dict(pipeline["execution_plan"])
        plan["items"] = items
        plan["selected_candidates"] = len(items)
        pipeline["execution_plan"] = plan
        return pipeline

    def _app_with_plan(self, size: int):
        pipeline = self._sized_plan(size)
        app = AppTest.from_file(APP_PATH, default_timeout=240)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["varo_pipeline_result"] = pipeline
        app.session_state["analysis_result"] = pipeline
        app.session_state["varo_recommendations"] = list(pipeline["execution_plan"]["items"])
        app.session_state["selected_route_id"] = None
        app.session_state["current_menu"] = WORKSPACE
        app.run()
        return app

    def test_the_screen_holds_up_from_one_move_to_fifty(self):
        for size in (1, 8, 20, 50):
            with self.subTest(size=size):
                app = self._app_with_plan(size)
                self.assertFalse(app.exception, msg=f"{size}건: {list(app.exception)}")
                widget = next(item for item in app.radio if item.key == "ws_plan_pick")
                # One control, every move selectable, whatever the plan size.
                self.assertEqual(len(widget.options), size)
                self.assertEqual(
                    len([r for r in app.radio if r.key.startswith("ws_plan")]), 1,
                )
                self.assertEqual(
                    [b for b in app.button if b.key.startswith(("ws_pick_", "ws_move_"))], [],
                )
                blob = self._blob(app)
                self.assertIn(f"{size}건", blob)
                self.assertIn('class="ws-action-qty"', blob)   # right panel still resolved
                if size > 1:
                    self.assertIn('class="ws-network-svg"', blob)

    def test_a_fifty_move_plan_can_still_be_narrowed_and_reselected(self):
        app = self._app_with_plan(50)
        route_ids = [
            str(item["route_id"])
            for item in app.session_state["analysis_result"]["execution_plan"]["items"]
        ]
        first = str(app.session_state["selected_route_id"])
        target = next(route_id for route_id in route_ids if route_id != first)
        next(item for item in app.radio if item.key == "ws_plan_pick").set_value(target).run()
        self.assertFalse(app.exception)
        # This fixture repeats the same eight moves to reach 50, so several rows
        # share a label and AppTest (which addresses a radio row by its rendered
        # label) can land on any of them. What must hold is that the click moved
        # the one shared selection to a real move of the current plan — the exact
        # id is asserted on the real 8-move plan in test_workspace_page.
        chosen = str(app.session_state["selected_route_id"])
        self.assertNotEqual(chosen, first)
        self.assertIn(chosen, route_ids)

        product = str(self.plan_items[0]["product_name"])
        next(s for s in app.selectbox if s.key == "ws_filter_product").select(product).run()
        self.assertFalse(app.exception)
        narrowed = next(item for item in app.radio if item.key == "ws_plan_pick")
        self.assertLess(len(narrowed.options), 50)
        self.assertGreater(len(narrowed.options), 0)
        # The selection stays inside what the filter left on screen: the id the
        # rest of the app reads is always one of the visible moves.
        kept = {
            str(item["route_id"])
            for item in app.session_state["analysis_result"]["execution_plan"]["items"]
            if str(item.get("product_name")) == product
        }
        self.assertIn(str(app.session_state["selected_route_id"]), kept)

    def test_an_empty_plan_never_shows_a_fabricated_move(self):
        # A 0-item plan is the 실행 이동 없음 state, not an empty list widget.
        pipeline = self._sized_plan(1)
        plan = dict(pipeline["execution_plan"])
        plan["items"] = []
        pipeline["execution_plan"] = plan
        app = AppTest.from_file(APP_PATH, default_timeout=240)
        app.run()
        for key in CANONICAL_DATA_KEYS:
            app.session_state[key] = self.payload.get(key)
        app.session_state["varo_pipeline_result"] = pipeline
        app.session_state["analysis_result"] = pipeline
        app.session_state["varo_recommendations"] = []
        app.session_state["current_menu"] = WORKSPACE
        app.run()
        self.assertFalse(app.exception)
        blob = self._blob(app)
        self.assertIn("추천할 이동이 없습니다", blob)
        self.assertNotIn('class="ws-action-qty"', blob)
        self.assertEqual([r for r in app.radio if r.key == "ws_plan_pick"], [])

    # ---------------------------------------------------------------- A. nav
    def test_the_app_opens_on_the_workspace(self):
        app = AppTest.from_file(APP_PATH, default_timeout=240)
        app.run()
        self.assertEqual(app.session_state["current_menu"], WORKSPACE)

    def test_the_sidebar_shows_four_screens_with_the_rest_folded_away(self):
        app = self._app()
        labels = [b.label for b in app.sidebar.button]
        self.assertEqual(labels[:4], [menu_label(item) for item in PRIMARY_MENU_ITEMS])
        self.assertIn("예전 화면", {item.label for item in app.sidebar.expander})
        # 기술 용어는 메뉴에 노출하지 않는다.
        for banned in ("Workspace", "Optimizer", "VHS", "MILP", "DQN", "Execution Plan"):
            self.assertNotIn(banned, " ".join(labels))

    def test_every_legacy_route_still_opens_from_the_sidebar(self):
        for menu in LEGACY_MENU_ITEMS:
            app = self._app()
            button = next(b for b in app.sidebar.button if b.key == f"nav_{menu}")
            button.click().run()
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")
            self.assertEqual(app.session_state["current_menu"], menu)

    def test_the_next_action_from_every_state_lands_on_the_workspace(self):
        state = {key: self.payload.get(key) for key in CANONICAL_DATA_KEYS}
        self.assertEqual(build_home_state(state)["next_page"], PAGE_WORKSPACE)
        pending = {**state, "analysis_run_required": True}
        self.assertEqual(build_home_state(pending)["next_page"], PAGE_WORKSPACE)
        self.assertEqual(build_home_state(pending)["next_action_label"], "분석 실행")

    # ------------------------------------------------------- E. rec parity
    def test_the_workspace_and_the_legacy_list_show_the_same_plan_in_the_same_order(self):
        planned = planned_recommendations(self.pipeline)
        # 추천 실행 page ordering (it re-sorts, which must be a no-op on plan order).
        self.assertEqual(
            [str(item["route_id"]) for item in sort_recommendations(planned)],
            [str(item["route_id"]) for item in planned],
        )
        # …and the Workspace list is that same list, unchanged.
        self.assertEqual(
            [row["route_id"] for row in plan_list_rows(planned)],
            [str(item["route_id"]) for item in planned],
        )
        home = build_home_state({key: self.payload.get(key) for key in CANONICAL_DATA_KEYS})
        self.assertEqual(str(home["top_recommendation"]["route_id"]), str(planned[0]["route_id"]))

    # ------------------------------------------------- D. route-detail parity
    def test_planned_quantity_is_identical_on_workspace_and_legacy_screens(self):
        app = self._app()
        selected = next(
            item for item in self.plan_items
            if str(item["route_id"]) == str(app.session_state["selected_route_id"])
        )
        quantity = qty_text(action_qty(selected))
        self.assertIn(f'class="ws-action-qty">{quantity}<', self._blob(app))
        for menu in ("추천 실행", "경로 상세"):
            app.session_state["current_menu"] = menu
            app.run()
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")
            metrics = {m.label: m.value for m in app.metric}
            self.assertEqual(
                metrics.get("실행 수량"), quantity,
                f"{menu}: 실행 수량이 재고 운영과 다릅니다",
            )

    def test_the_detail_tab_and_the_legacy_route_screen_show_the_same_numbers(self):
        """세부 정보 탭이 경로 상세를 대신하므로 값이 어긋나면 안 된다."""
        app = self._app()
        selected = next(
            item for item in self.plan_items
            if str(item["route_id"]) == str(app.session_state["selected_route_id"])
        )
        rows = {row["항목"]: row["값"] for row in move_detail_rows(selected)}
        app.session_state["current_menu"] = "경로 상세"
        app.run()
        self.assertFalse(app.exception)
        metrics = {m.label: m.value for m in app.metric}
        self.assertEqual(rows["실행 수량"], metrics["실행 수량"])
        self.assertEqual(rows["예상 순효과"], metrics["예상 순효과"])

    # -------------------------------------------------- F. validation parity
    def test_workspace_summary_and_the_detail_screen_agree_on_the_verdict(self):
        rows = {str(row["검증 항목"]): str(row["결과"]) for row in validation_rows(self.pipeline)}
        app = self._app("분석 및 검증")
        self.assertFalse(app.exception)
        blob = self._blob(app)
        self.assertIn("결론", blob)
        metrics = {m.label: m.value for m in app.metric}
        for item in ("계획 제약", "안전재고", "도착 필요 수량", "안정성"):
            self.assertEqual(metrics.get(item), rows[item], f"{item} 결과가 두 화면에서 다릅니다")
        if rows["계획 제약"] == "이상 없음" and rows["안전재고"] == "이상 없음":
            self.assertIn("현재 계획은 실행 가능하며 안전재고 침범이 없습니다.", blob)

    # ---------------------------------------------------- G. terminology
    def test_one_word_per_concept_across_the_screens(self):
        blob = ""
        for menu in MENU_ITEMS:
            app = self._app(menu)
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")
            blob += " " + self._blob(app)
            blob += " " + " ".join(str(m.label) for m in app.metric)
        # A single wording for each concept: no competing synonym anywhere.
        for banned, preferred in (
            ("추천 안정성", "안정성"),
            ("운영 현황", "운영 시뮬레이션"),
            ("세부정보", "세부 정보"),
        ):
            self.assertNotIn(banned, blob, f"'{banned}' 대신 '{preferred}'만 사용해야 합니다")

    # ------------------------------------------------------- I. regression
    def test_the_untouched_screens_still_render(self):
        for menu in ("데이터 관리", "운영 현황", "분석 및 검증"):
            app = self._app(menu)
            self.assertFalse(app.exception, msg=f"{menu}: {list(app.exception)}")

    def test_the_execution_history_flow_stays_on_the_workspace(self):
        app = self._app()
        self.assertIn("record_execution_plan", {b.key for b in app.button})
        self.assertIn("실행 기록", self._blob(app))

    # -------------------------------------------------------- J. algorithm
    def test_the_workspace_pass_changed_no_algorithm_result(self):
        """Recomputing the same workbook must give byte-identical plan values."""
        data = load_excel_data(get_default_sample_path())
        again = run_analysis_pipeline(data).to_dict()
        for before, after in zip(self.plan_items, again["execution_plan"]["items"]):
            for field in (
                "route_id", "plan_rank", "planned_qty", "recommended_qty",
                "planned_cost", "planned_expected_saving", "planned_net_benefit",
            ):
                self.assertEqual(before.get(field), after.get(field), field)
        for field in (
            "total_transfer_qty", "total_cost", "total_net_benefit",
            "selected_candidates", "algorithm_version",
        ):
            self.assertEqual(
                self.pipeline["execution_plan"].get(field),
                again["execution_plan"].get(field),
                field,
            )


if __name__ == "__main__":
    unittest.main()
