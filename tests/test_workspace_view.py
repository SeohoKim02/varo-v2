"""Rules of the 재고 운영 Workspace view-model.

Pure-Python checks (no Streamlit) for the guarantees the single-screen workspace
makes: one plan is the single source of truth, ``planned_qty`` is the action
quantity everywhere, missing logistics facts stay 미확보 instead of becoming 0,
and no internal term reaches the view.
"""
from __future__ import annotations

import unittest

import pandas as pd

from components import workspace_network as wsn
from components.workspace_network import (
    SCOPE_ALL,
    SCOPE_PLAN,
    SCOPE_SELECTED,
    build_workspace_network,
    plan_edge_options,
    visible_nodes,
)
from services.app_state import TRANSIENT_VIEW_KEYS, build_applied_state_payload
from services.analysis_pipeline import run_analysis_pipeline
from services.data_application import load_and_apply
from services.data_validator import validate_workbook_data
from services.workspace_view import (
    ALL,
    NOT_COMPUTABLE,
    NO_DATA,
    PROVENANCE_LABELS,
    WORKSPACE_VIEW_KEYS,
    action_qty,
    alternatives_for,
    apply_filters,
    build_workspace_view,
    data_readiness,
    filter_options,
    money_text,
    move_reasons,
    move_risks,
    move_title,
    plan_kpis,
    qty_text,
    readiness_summary,
    resolve_selection,
    route_label,
    store_inventory_states,
    validation_rows,
    whatif_rows,
)
from tests.fixtures import (
    dual_dc_workbook_sheets,
    sample_workbook,
    workbook_excel_bytes,
    write_dqn_style_workbook,
)

# Internal vocabulary that must never appear in anything the workspace renders.
BANNED_TOKENS = (
    "candidate_id", "data_signature", "scipy", "milp", "replay", "epsilon",
    "Traceback", "session_state", "reason_code", "status_code", "sqlite",
    "postgres", "schema_version", "winsor", "normalization",
)


def _applied_state(sheets=None, filename="workspace.xlsx") -> dict:
    """Run the real intake + pipeline once, exactly like the app does."""
    state: dict = {}
    load_and_apply(
        state, workbook_excel_bytes(sheets or sample_workbook()), filename, "업로드된 추천 결과",
    )
    return state


class WorkspaceStateTests(unittest.TestCase):
    """A · workspace states: 데이터 없음 / 분석 전 / 분석 완료 / plan 0·1·다수."""

    @classmethod
    def setUpClass(cls):
        cls.state = _applied_state()

    def test_no_data_state_offers_one_next_action_and_no_result(self):
        view = build_workspace_view({})
        self.assertFalse(view["ready"])
        self.assertEqual(view["state_code"], "no_data")
        self.assertEqual(view["plan_items"], [])
        self.assertIsNone(view["selected"])
        self.assertTrue(view["home"]["next_action_label"])

    def test_analysis_pending_state_asks_for_the_run_not_a_fake_result(self):
        state = dict(self.state)
        state["analysis_run_required"] = True
        state["varo_recommendations"] = []
        state["analysis_result"] = {}
        state["varo_pipeline_result"] = {}
        view = build_workspace_view(state)
        self.assertTrue(view["analysis_pending"])
        self.assertFalse(view["ready"])
        self.assertEqual(view["analysis_status"], "분석 필요")
        self.assertEqual(view["plan_items"], [])

    def test_ready_state_exposes_the_execution_plan_as_the_action_list(self):
        view = build_workspace_view(self.state)
        self.assertTrue(view["ready"])
        self.assertEqual(view["analysis_status"], "분석 완료")
        plan_items = view["plan_items"]
        self.assertGreater(len(plan_items), 0)
        expected = self.state["varo_pipeline_result"]["execution_plan"]["items"]
        self.assertEqual(
            [str(item["route_id"]) for item in plan_items],
            [str(item["route_id"]) for item in expected],
        )

    def test_plan_with_no_executable_move_is_reported_as_such(self):
        workbook = sample_workbook()
        recs = workbook["recommendations"].copy()
        recs["recommended_qty"] = 999999  # every move exceeds the source stock
        workbook["recommendations"] = recs
        view = build_workspace_view(_applied_state(workbook, "blocked.xlsx"))
        self.assertFalse(view["ready"])
        self.assertEqual(view["state_code"], "no_candidates")
        self.assertEqual(view["plan_items"], [])
        self.assertEqual(view["kpis"][0]["value"], "0건")

    def test_single_move_plan_still_selects_and_renders(self):
        view = build_workspace_view(self.state)
        one = view["plan_items"][:1]
        self.assertEqual(resolve_selection(one, None), str(one[0]["route_id"]))
        kpis = plan_kpis({"total_transfer_qty": one[0].get("planned_qty")}, one)
        self.assertEqual(kpis[0]["value"], "1건")

    def test_stale_result_is_flagged_when_the_data_changed(self):
        state = dict(self.state)
        state["data_signature"] = "changed-signature"
        view = build_workspace_view(state)
        self.assertTrue(view["stale"])
        self.assertEqual(view["analysis_status"], "다시 분석 필요")


class WorkspaceSelectionTests(unittest.TestCase):
    """B · selection: 하나의 선택이 네트워크·상세·하단에서 같은 항목을 가리킨다."""

    @classmethod
    def setUpClass(cls):
        cls.state = _applied_state()
        cls.view = build_workspace_view(cls.state)

    def test_default_selection_is_the_first_plan_item(self):
        self.assertEqual(
            self.view["selected_route_id"], str(self.view["plan_items"][0]["route_id"]),
        )

    def test_selection_change_moves_the_detail_to_the_same_item(self):
        items = self.view["plan_items"]
        self.assertGreater(len(items), 1)
        second = str(items[1]["route_id"])
        state = dict(self.state)
        state["selected_route_id"] = second
        view = build_workspace_view(state)
        self.assertEqual(view["selected_route_id"], second)
        self.assertEqual(str(view["selected"]["route_id"]), second)
        self.assertEqual(move_title(view["selected"]), move_title(items[1]))

    def test_selection_outside_the_plan_falls_back_instead_of_breaking(self):
        state = dict(self.state)
        state["selected_route_id"] = "NOT-A-ROUTE"
        view = build_workspace_view(state)
        self.assertEqual(view["selected_route_id"], str(self.view["plan_items"][0]["route_id"]))

    def test_network_picker_lists_exactly_the_plan_moves(self):
        options = plan_edge_options(self.view["plan_items"])
        self.assertEqual(
            [option["route_id"] for option in options],
            [str(item["route_id"]) for item in self.view["plan_items"]],
        )


class WorkspaceQuantityTests(unittest.TestCase):
    """C · planned_qty is the one action quantity in every place it appears."""

    @classmethod
    def setUpClass(cls):
        cls.view = build_workspace_view(_applied_state())

    def test_action_quantity_is_planned_qty_not_the_candidate_quantity(self):
        item = dict(self.view["selected"])
        item["planned_qty"] = 12
        item["recommended_qty"] = 40
        self.assertEqual(action_qty(item), 12)
        self.assertEqual(qty_text(action_qty(item)), "12개")

    def test_right_panel_edge_and_alternative_row_show_the_same_quantity(self):
        selected = self.view["selected"]
        expected = qty_text(action_qty(selected))
        edge = next(
            option for option in plan_edge_options(self.view["plan_items"])
            if option["route_id"] == str(selected["route_id"])
        )
        row = next(
            row for row in alternatives_for(
                selected, self.view["plan_items"], self.view["plan"],
            ) if row["route_id"] == str(selected["route_id"])
        )
        self.assertEqual(edge["qty"], expected)
        self.assertEqual(row["수량"], expected)

    def test_plan_kpi_quantity_matches_the_plan_total(self):
        plan = self.view["plan"]
        self.assertEqual(self.view["kpis"][1]["value"], qty_text(plan["total_transfer_qty"]))
        self.assertEqual(self.view["kpis"][2]["value"], money_text(plan["total_net_benefit"]))


class WorkspaceRouteTests(unittest.TestCase):
    """D · DIRECT / VIA_DC and more than one DC stay distinguishable."""

    def test_direct_and_via_dc_get_distinct_labels(self):
        self.assertEqual(route_label({"route_type": "DIRECT"}), "직접 이동")
        self.assertEqual(
            route_label({"route_type": "VIA_DC", "dc_name": "중앙DC"}), "DC 경유 · 중앙DC",
        )

    def test_two_dcs_are_named_separately_and_drawn_separately(self):
        sheets = dual_dc_workbook_sheets()
        state: dict = {}
        with_path = write_dqn_style_workbook("_ws_dual_dc.xlsx", sheets)
        try:
            load_and_apply(state, str(with_path), "dual_dc.xlsx", "업로드된 추천 결과")
        finally:
            with_path.unlink(missing_ok=True)
        view = build_workspace_view(state)
        candidates = state["varo_recommendations"]
        dcs = {str(item.get("dc_id")) for item in candidates if item.get("dc_id")}
        self.assertIn("DC01", dcs)
        self.assertIn("DC02", dcs)
        self.assertEqual(
            route_label({"route_type": "VIA_DC", "dc_id": "DC02", "dc_name": "남부DC"}),
            "DC 경유 · 남부DC",
        )
        network = build_workspace_network(
            state["varo_data"], candidates, str(candidates[0].get("route_id")), scope=SCOPE_ALL,
        )
        self.assertTrue(network["ok"])
        self.assertIn("중앙DC", network["html"])
        self.assertIn("남부DC", network["html"])
        self.assertTrue(view["readiness"])

    def test_scope_selected_draws_only_the_selected_move(self):
        state = _applied_state()
        view = build_workspace_view(state)
        items = view["plan_items"]
        selected = items[0]
        nodes_all = visible_nodes(
            [{"node_id": "S001"}, {"node_id": "S002"}, {"node_id": "S003"}], items, SCOPE_ALL,
        )
        self.assertEqual(len(nodes_all), 3)
        one = visible_nodes(
            [{"node_id": str(selected["source_id"])},
             {"node_id": str(selected["target_id"])},
             {"node_id": "UNRELATED"}],
            [selected], SCOPE_SELECTED,
        )
        self.assertNotIn("UNRELATED", {node["node_id"] for node in one})

    def test_edge_label_uses_quantity_only_never_a_distance_or_cost(self):
        state = _applied_state()
        view = build_workspace_view(state)
        network = build_workspace_network(
            state["varo_data"], view["plan_items"], view["selected_route_id"], scope=SCOPE_PLAN,
            store_states=store_inventory_states(state["varo_data"], view["plan_items"]),
        )
        self.assertTrue(network["ok"])
        self.assertIn("개<", network["html"])
        for banned in ("km", "원<", "분<"):
            self.assertNotIn(banned, network["html"])


class WorkspaceMissingLogisticsTests(unittest.TestCase):
    """E · 미확보 물류 정보는 0이 아니라 미확보로 남는다."""

    def test_missing_distance_time_cost_and_vehicle_are_reported_as_missing(self):
        workbook = sample_workbook()
        workbook["routes"] = workbook["routes"].drop(
            columns=["distance_km", "estimated_cost", "travel_time_min"]
        )
        rows = data_readiness({"varo_data": workbook})
        by_name = {row["항목"]: row for row in rows}
        for label in ("이동 거리", "이동 시간", "운송비", "차량 용량", "실제 거점간 이동이력"):
            self.assertEqual(by_name[label]["상태"], PROVENANCE_LABELS["not_available"], label)
        summary = readiness_summary(rows)
        self.assertIn("이동 거리", summary["missing"])
        self.assertIn("미확보", summary["headline"])

    def test_present_logistics_values_are_reference_not_actual_observation(self):
        rows = data_readiness({"varo_data": sample_workbook()})
        by_name = {row["항목"]: row for row in rows}
        self.assertEqual(by_name["이동 거리"]["상태"], PROVENANCE_LABELS["reference"])
        self.assertEqual(by_name["운송비"]["상태"], PROVENANCE_LABELS["reference"])
        self.assertEqual(by_name["판매·재고"]["상태"], PROVENANCE_LABELS["actual"])

    def test_recorded_execution_history_turns_the_movement_log_into_actual(self):
        rows = data_readiness({"varo_data": sample_workbook()}, history_confirmed=3)
        by_name = {row["항목"]: row for row in rows}
        self.assertEqual(by_name["실제 거점간 이동이력"]["상태"], PROVENANCE_LABELS["actual"])

    def test_no_missing_value_is_ever_formatted_as_zero(self):
        self.assertEqual(qty_text(None), NO_DATA)
        self.assertEqual(money_text(None), NO_DATA)
        self.assertEqual(qty_text(float("nan")), NO_DATA)
        self.assertEqual(money_text(float("inf")), NO_DATA)
        empty = plan_kpis({}, [])
        self.assertEqual(empty[1]["value"], NO_DATA)
        self.assertEqual(empty[2]["value"], NOT_COMPUTABLE)

    def test_missing_logistics_data_is_surfaced_as_a_risk_line(self):
        rows = data_readiness({"varo_data": {}})
        risks = move_risks({"route_id": "R1", "planned_qty": 3}, rows)
        self.assertTrue(any("실제 운송 정보" in line for line in risks))


class WorkspaceFilterTests(unittest.TestCase):
    """Filters narrow the plan without ever re-ordering or re-ranking it."""

    @classmethod
    def setUpClass(cls):
        cls.view = build_workspace_view(_applied_state())

    def test_default_filters_keep_the_plan_order(self):
        items = self.view["plan_items"]
        kept = apply_filters(items, {"product": ALL, "source": ALL, "target": ALL, "route_type": ALL})
        self.assertEqual(
            [item["route_id"] for item in kept], [item["route_id"] for item in items],
        )

    def test_product_filter_keeps_only_that_product(self):
        items = self.view["plan_items"]
        product = str(items[0]["product_name"])
        kept = apply_filters(items, {"product": product})
        self.assertTrue(kept)
        self.assertTrue(all(str(item["product_name"]) == product for item in kept))

    def test_options_are_offered_with_an_all_entry_first(self):
        options = filter_options(self.view["plan_items"])
        for key in ("product", "source", "target", "route_type"):
            self.assertEqual(options[key][0], ALL)

    def test_only_actionable_filter_keeps_positive_net_effect_moves(self):
        items = [
            {"route_id": "A", "planned_net_benefit": 100.0},
            {"route_id": "B", "planned_net_benefit": 0.0},
        ]
        kept = apply_filters(items, {"only_actionable": True})
        self.assertEqual([item["route_id"] for item in kept], ["A"])


class WorkspaceExplanationTests(unittest.TestCase):
    """Reasons, risks, alternatives and what-if only state what the data supports."""

    @classmethod
    def setUpClass(cls):
        cls.state = _applied_state()
        cls.view = build_workspace_view(cls.state)

    def test_reasons_are_between_two_and_four_short_lines(self):
        reasons = move_reasons(self.view["selected"], {})
        self.assertGreaterEqual(len(reasons), 2)
        self.assertLessEqual(len(reasons), 4)
        for line in reasons:
            self.assertLessEqual(len(line), 70)

    def test_no_reasons_are_invented_for_an_empty_candidate(self):
        self.assertEqual(move_reasons(None, {}), [])
        self.assertEqual(move_reasons({}, {}), [])

    def test_adjusted_quantity_is_explained_without_the_optimisation_formula(self):
        risks = move_risks({"quantity_adjusted": True}, [])
        self.assertTrue(any("수량이 조정" in line for line in risks))
        for line in risks:
            for banned in BANNED_TOKENS:
                self.assertNotIn(banned, line)

    def test_alternatives_compare_only_real_candidates(self):
        rows = alternatives_for(
            self.view["selected"], self.state["varo_recommendations"], self.view["plan"],
        )
        self.assertTrue(rows)
        self.assertEqual(rows[0]["선택"], "●")
        for row in rows:
            self.assertIn(row["구분"], ("현재 이동", "같은 도착 점포", "같은 출발 재고"))

    def test_whatif_reports_calculated_status_or_says_it_cannot_be_computed(self):
        rows = whatif_rows(self.view["pipeline"], self.view["selected"])
        self.assertEqual(len(rows), 4)
        empty = whatif_rows({}, None)
        self.assertEqual(empty[0]["결과"], NOT_COMPUTABLE)
        self.assertEqual(empty[1]["결과"], NOT_COMPUTABLE)

    def test_validation_rows_are_plain_operator_language(self):
        rows = validation_rows(self.view["pipeline"])
        text = " ".join(f"{row['검증 항목']} {row['결과']} {row['설명']}" for row in rows)
        self.assertIn("안전재고", text)
        for banned in BANNED_TOKENS:
            self.assertNotIn(banned, text)


class WorkspaceTextHygieneTests(unittest.TestCase):
    """G · no internal term reaches the workspace surface."""

    @classmethod
    def setUpClass(cls):
        cls.state = _applied_state()
        cls.view = build_workspace_view(cls.state)

    def _surface_text(self) -> str:
        view = self.view
        parts: list[str] = [
            view["data_status"], view["analysis_status"], view["plan_message"],
            move_title(view["selected"]), route_label(view["selected"]),
        ]
        parts += [f"{c['title']} {c['value']} {c['caption']}" for c in view["kpis"]]
        parts += [f"{r['항목']} {r['상태']} {r['설명']}" for r in view["readiness"]]
        parts += move_reasons(view["selected"], {})
        parts += move_risks(view["selected"], view["readiness"])
        parts += [
            " ".join(str(value) for key, value in row.items() if key != "route_id")
            for row in alternatives_for(
                view["selected"], self.state["varo_recommendations"], view["plan"],
            )
        ]
        parts += [f"{r['조건']} {r['결과']} {r['설명']}" for r in whatif_rows(view["pipeline"], view["selected"])]
        parts += [f"{r['검증 항목']} {r['결과']} {r['설명']}" for r in validation_rows(view["pipeline"])]
        return " ".join(parts)

    def test_workspace_text_has_no_internal_vocabulary(self):
        text = self._surface_text()
        for banned in BANNED_TOKENS:
            self.assertNotIn(banned, text, f"workspace must not expose: {banned}")

    def test_workspace_text_has_no_route_or_candidate_identifier(self):
        text = self._surface_text()
        for item in self.view["plan_items"]:
            self.assertNotIn(str(item["route_id"]), text)


class WorkspaceStateKeyTests(unittest.TestCase):
    """F · the workspace adds no session key that survives a data change."""

    def test_workspace_widget_keys_are_reset_with_new_data(self):
        for key in WORKSPACE_VIEW_KEYS:
            self.assertIn(key, TRANSIENT_VIEW_KEYS, f"{key} must be cleared on new data")

    def test_workspace_reuses_the_shared_selection_key(self):
        state = _applied_state()
        view = build_workspace_view(state)
        payload = build_applied_state_payload(
            state["varo_data"], state["varo_validation"], state["varo_recommendations"],
            "again.xlsx", "업로드된 추천 결과", state["varo_pipeline_result"],
        )
        self.assertEqual(payload["selected_route_id"], view["selected_route_id"])


class WorkspaceNetworkGeometryTests(unittest.TestCase):
    """H · the network never draws a node off-canvas or on top of another one."""

    def _positions(self, store_count: int, dc_count: int):
        from simulation.dynamic_network import compute_network_layout

        store_size = wsn._dimensions(store_count)
        capacity = wsn.ring_capacity(store_size, dc_count)
        uses_rings = store_count > wsn._SINGLE_RING_LIMIT or dc_count >= wsn._CENTRAL_ROW_DC_LIMIT
        shown = min(store_count, capacity) if uses_rings else store_count
        stores = [{"node_id": f"S{index:03d}"} for index in range(shown)]
        dcs = [{"node_id": f"DC{index:02d}"} for index in range(dc_count)]
        if uses_rings:
            positions = wsn._ring_positions(stores, dcs, store_size)
        else:
            positions = {
                key: (value[0], value[1])
                for key, value in compute_network_layout(
                    stores, dcs, wsn.CANVAS_WIDTH, wsn.CANVAS_HEIGHT, wsn.CANVAS_MARGIN,
                    set(), store_size,
                ).items()
            }
        return positions, store_size, shown

    def _box(self, node_id: str, store_size: tuple[float, float]) -> tuple[float, float]:
        if node_id.startswith("DC"):
            return wsn._dc_size(store_size)
        return store_size

    def test_every_node_stays_inside_the_canvas(self):
        for store_count in (1, 2, 4, 8, 12, 16, 17, 24, 30, 40, 60):
            for dc_count in (0, 1, 2, 3, 4):
                with self.subTest(stores=store_count, dcs=dc_count):
                    positions, store_size, _shown = self._positions(store_count, dc_count)
                    for node_id, (x, y) in positions.items():
                        width, height = self._box(node_id, store_size)
                        self.assertGreaterEqual(x - width / 2, -1)
                        self.assertLessEqual(x + width / 2, wsn.CANVAS_WIDTH + 1)
                        # The role label sits above the box; keep room for it too.
                        self.assertGreaterEqual(y - height / 2 - 18, -1)
                        self.assertLessEqual(y + height / 2, wsn.CANVAS_HEIGHT + 1)

    def test_no_two_nodes_are_drawn_on_top_of_each_other(self):
        for store_count in (2, 6, 12, 16, 20, 28, 40, 60):
            for dc_count in (0, 1, 2, 3, 4):
                with self.subTest(stores=store_count, dcs=dc_count):
                    positions, store_size, _shown = self._positions(store_count, dc_count)
                    ids = list(positions)
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            xi, yi = positions[ids[i]]
                            xj, yj = positions[ids[j]]
                            wi, hi = self._box(ids[i], store_size)
                            wj, hj = self._box(ids[j], store_size)
                            overlapping = (
                                abs(xi - xj) < (wi + wj) / 2 * 0.94
                                and abs(yi - yj) < (hi + hj) / 2 * 0.94
                            )
                            self.assertFalse(overlapping, f"{ids[i]} overlaps {ids[j]}")

    def test_a_network_larger_than_the_canvas_says_what_it_left_out(self):
        stores = pd.DataFrame(
            [{"node_id": f"S{index:03d}", "node_name": f"점포{index}", "node_type": "STORE"}
             for index in range(60)]
            + [{"node_id": "DC01", "node_name": "중앙DC", "node_type": "DC"}]
        )
        items = [{
            "route_id": "R1", "source_id": "S000", "target_id": "S001",
            "product_id": "P1", "route_type": "DIRECT", "planned_qty": 5,
        }]
        network = build_workspace_network(
            {"stores": stores}, items, "R1", scope=SCOPE_ALL,
        )
        self.assertTrue(network["ok"])
        self.assertIn("표시하지 않은 점포", network["message"])
        # The move's own stores are never the ones dropped.
        self.assertIn("점포0<", network["html"])
        self.assertIn("점포1<", network["html"])


class WorkspaceStoreStateTests(unittest.TestCase):
    """Store status is one rule, shared by the workspace and the simulation view."""

    def test_store_states_are_limited_to_the_four_known_labels(self):
        workbook = sample_workbook()
        states = store_inventory_states(workbook, [{"source_id": "S001"}])
        self.assertEqual(states.get("S001"), "이동 대상")
        self.assertTrue(set(states.values()) <= {"과잉", "부족", "정상", "이동 대상"})

    def test_missing_inventory_produces_no_invented_state(self):
        self.assertEqual(store_inventory_states({"inventory": pd.DataFrame()}, []), {})

    def test_overview_page_uses_the_same_rule(self):
        from pages import overview

        workbook = sample_workbook()
        self.assertEqual(
            overview._store_inventory_states(workbook, []),
            store_inventory_states(workbook, []),
        )


class WorkspaceNetworkLegibilityTests(unittest.TestCase):
    """I · what a person actually reads on screen stays readable.

    The SVG is scaled to the width of the centre column, so a user-unit font size
    is multiplied by (column width / CANVAS_WIDTH) before anyone sees it. Measured
    in a real browser, the narrowest supported desktop (1366px) gives the centre
    column ~625px, i.e. the scale asserted here. These checks fail if a change
    would push node names, edge numbers, or state pills back under ~10px, or draw
    a name outside its own box.
    """

    #: Centre-column width at 1366×768, measured in Chrome against the live app.
    NARROWEST_COLUMN = 625.0
    MIN_SCREEN_PX = 9.9

    def _scale(self) -> float:
        return self.NARROWEST_COLUMN / wsn.CANVAS_WIDTH

    def test_node_name_floor_still_reads_on_the_narrowest_desktop(self):
        for font in (wsn.NAME_FONT_MIN, wsn.DC_NAME_FONT_MIN, wsn.EDGE_FONT):
            with self.subTest(font=font):
                self.assertGreaterEqual(font * self._scale(), self.MIN_SCREEN_PX)

    def test_a_long_name_is_shortened_only_after_the_font_floor_is_reached(self):
        label, font = wsn._fit_label("연신내점", 120.0, wsn.NAME_FONT, wsn.NAME_FONT_MIN)
        self.assertEqual(label, "연신내점")
        self.assertEqual(font, wsn.NAME_FONT)
        long_label, long_font = wsn._fit_label(
            "아주아주긴이름을가진점포입니다", 120.0, wsn.NAME_FONT, wsn.NAME_FONT_MIN,
        )
        self.assertEqual(long_font, wsn.NAME_FONT_MIN)
        self.assertTrue(long_label.endswith("…"))
        self.assertLessEqual(wsn._glyph_width(long_label, long_font), 120.0 - 12.0)

    def test_every_fitted_name_stays_inside_its_box(self):
        names = ("연신내점", "서울 서북권 물류센터", "A", "가나다라마바사아자차카타파하")
        for count in (4, 10, 16, 26, 40):
            width, _height = wsn._dimensions(count)
            dc_width, _dc_height = wsn._dc_size(wsn._dimensions(count))
            for name in names:
                with self.subTest(count=count, name=name):
                    label, font = wsn._fit_label(name, width, wsn.NAME_FONT, wsn.NAME_FONT_MIN)
                    self.assertLessEqual(wsn._glyph_width(label, font), width - 12.0 + 0.01)
                    lines, dc_font = wsn._fit_dc_label(name, dc_width)
                    self.assertLessEqual(len(lines), 2)
                    for line in lines:
                        self.assertLessEqual(
                            wsn._glyph_width(line, dc_font), dc_width - 12.0 + 0.01,
                        )

    def test_a_long_dc_name_wraps_instead_of_being_cut(self):
        dc_width, _ = wsn._dc_size(wsn._dimensions(10))
        lines, font = wsn._fit_dc_label("서울 서북권 물류센터", dc_width)
        self.assertEqual(lines, ["서울 서북권", "물류센터"])
        self.assertGreaterEqual(font * self._scale(), self.MIN_SCREEN_PX)

    def test_edge_quantity_chips_never_overlap_a_node_or_each_other(self):
        state = _applied_state()
        view = build_workspace_view(state)
        items = list(view["plan_items"])
        self.assertGreater(len(items), 1)
        network = build_workspace_network(
            state["varo_data"], items, view["selected_route_id"], scope=SCOPE_PLAN,
        )
        self.assertTrue(network["ok"])
        boxes = _svg_boxes(network["html"])
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                (ax, ay, aw, ah), (bx, by, bw, bh) = boxes[first], boxes[second]
                with self.subTest(pair=(first, second)):
                    self.assertFalse(
                        ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah,
                        "an edge quantity chip is drawn on top of a node box",
                    )

    def test_the_selected_move_always_keeps_its_quantity_chip(self):
        state = _applied_state()
        view = build_workspace_view(state)
        items = list(view["plan_items"])
        selected = next(
            item for item in items
            if str(item["route_id"]) == str(view["selected_route_id"])
        )
        network = build_workspace_network(
            state["varo_data"], items, view["selected_route_id"], scope=SCOPE_PLAN,
        )
        self.assertIn(f">{qty_text(action_qty(selected))}</text>", network["html"])


def _svg_boxes(html: str) -> list[tuple[float, float, float, float]]:
    """Edge chips plus the node rects they must clear, as (x, y, w, h)."""
    import re

    boxes: list[tuple[float, float, float, float]] = []
    for match in re.finditer(
        r'<rect class="ws-edge-chip" x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"',
        html,
    ):
        boxes.append(tuple(float(value) for value in match.groups()))  # type: ignore[arg-type]
    for match in re.finditer(
        r'<g class="ws-node[^"]*" transform="translate\(([-\d.]+) ([-\d.]+)\)">(.*?)</g>', html,
    ):
        x, y, body = float(match.group(1)), float(match.group(2)), match.group(3)
        rect = re.search(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"', body)
        if rect:
            left, top, width, height = (float(value) for value in rect.groups())
            boxes.append((x + left, y + top, width, height))
    return boxes


if __name__ == "__main__":
    unittest.main()
