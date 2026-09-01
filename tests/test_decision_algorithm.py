"""Contract for the strengthened decision algorithm.

Covers the properties a reviewer needs to be able to check without reading the
code: what each VHS component means and how it behaves at the extremes, where the
hard/soft boundary is, how demand uncertainty and robustness are derived, how the
quantity and the net benefit are computed, that route and DC choice come from the
numbers rather than from row order, and that ties are broken deterministically.

Everything is built from small explicit candidate sets so a failure points at one
property, plus a few end-to-end checks on the anonymized operational workbook.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from services.analysis_pipeline import calculate_overview_kpis, run_analysis_pipeline, sort_recommendations
from services.decision_metrics import (
    ALGORITHM_VERSION, DEMAND_SCENARIO_Z, OVERSUPPLY_MULTIPLE, SCENARIO_UNKNOWN,
    annotate_decision_metrics, build_decision_context, demand_scenarios, net_benefit,
    quantity_plan, scenario_status,
)
from services.data_application import commit_pending_data, prepare_pending_data, run_applied_analysis
from services.feasibility import (
    STATUS_BLOCKED, STATUS_CHECK, STATUS_OK, InventoryContext, annotate_feasibility,
    evaluate_feasibility,
)
from services.greedy_baseline import BASELINE_NAME, compare_to_vhs, greedy_ranking
from services.pareto_analysis import PARETO_OBJECTIVES, compute_pareto
from services.vhs_score_engine import (
    COMPONENTS, COMPONENT_LABELS, ROBUST, ROBUST_FRAGILE, WEIGHT_BOUNDS,
    apply_auto_vhs, candidate_robustness,
)
from tests.fixtures import sample_workbook, workbook_excel_bytes
from tools.generate_anonymized_operational_workbook import WORKBOOK_NAME, generate

_TEMP_DIR: Path | None = None
_CACHE: dict[str, Any] = {}


def setUpModule() -> None:  # noqa: N802 - unittest hook
    global _TEMP_DIR
    _TEMP_DIR = Path(tempfile.mkdtemp(prefix="varo_algo_"))
    generate(_TEMP_DIR)


def tearDownModule() -> None:  # noqa: N802 - unittest hook
    if _TEMP_DIR is not None:
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)


def operational_result():
    """Analysed operational workbook, computed once and shared read-only."""
    if "result" not in _CACHE:
        assert _TEMP_DIR is not None
        state: dict[str, Any] = {}
        prepare_pending_data(state, str(_TEMP_DIR / WORKBOOK_NAME), WORKBOOK_NAME, "업로드 데이터")
        assert commit_pending_data(state)
        assert run_applied_analysis(state)
        _CACHE["result"] = state
    return _CACHE["result"]


def candidate(route_id: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "route_id": route_id, "product_id": "P1", "product_name": "가상상품 01",
        "source_id": "S1", "source_name": "가상점포 01",
        "target_id": "S2", "target_name": "가상점포 02",
        "route_type": "DIRECT", "dc_id": None, "dc_name": None,
        "recommended_qty": 10, "transport_type": "일반 탑차",
        "estimated_cost": 1000, "expected_saving": 20000,
        "distance_km": 5.0, "travel_time_min": 15.0,
        "vhs_score": 60.0, "recommendation_grade": "보통",
        "confidence_score": 60.0, "reason": "-",
    }
    base.update(overrides)
    return base


def context(**kwargs: Any) -> InventoryContext:
    return InventoryContext(
        stock=kwargs.get("stock", {}), demand=kwargs.get("demand", {}),
        safety=kwargs.get("safety", {}), known_stores=kwargs.get("known", set()),
        dispersion=kwargs.get("dispersion", {}),
    )


# --------------------------------------------------------------------------- #
# A. VHS 구성요소 · 정규화 · 극단값 · tie-break
# --------------------------------------------------------------------------- #
class VhsComponentTests(unittest.TestCase):
    def test_every_component_has_a_label_bounds_and_a_base_weight(self):
        from services.vhs_score_engine import BASE_WEIGHTS

        for component in COMPONENTS:
            self.assertIn(component, COMPONENT_LABELS, component)
            self.assertIn(component, WEIGHT_BOUNDS, component)
            self.assertIn(component, BASE_WEIGHTS, component)
            low, high = WEIGHT_BOUNDS[component]
            self.assertLessEqual(low, high, component)

    def test_the_baseline_and_confidence_are_not_scored_inside_vhs(self):
        """Greedy is the comparison, confidence is derived from VHS: neither may
        be a component, or the comparison and the confidence become circular."""
        self.assertNotIn("greedy_score", COMPONENTS)
        self.assertNotIn("confidence_score", COMPONENTS)

    def test_scores_stay_in_range_for_extreme_and_missing_values(self):
        rows = [
            candidate("A", expected_saving=10**12, estimated_cost=0, distance_km=0, travel_time_min=0),
            candidate("B", expected_saving=1, estimated_cost=10**12, distance_km=10**6, travel_time_min=10**6),
            candidate("C", expected_saving=None, estimated_cost=None, distance_km=None, travel_time_min=None),
            candidate("D"),
        ]
        result = apply_auto_vhs(pd.DataFrame(rows))
        scores = pd.to_numeric(result.frame["vhs_score"], errors="coerce")
        self.assertTrue(scores.notna().all())
        self.assertTrue(scores.between(0, 100).all())
        for component in COMPONENTS:
            values = pd.to_numeric(result.frame[component], errors="coerce")
            self.assertTrue(values.notna().all(), component)
            self.assertTrue(values.between(0, 100).all(), component)

    def test_infinite_values_never_become_the_normalization_bound(self):
        rows = [
            candidate("A", expected_saving=float("inf")),
            candidate("B", expected_saving=100),
            candidate("C", expected_saving=200),
        ]
        result = apply_auto_vhs(pd.DataFrame(rows))
        component = pd.to_numeric(result.frame["net_benefit_score"], errors="coerce")
        self.assertTrue(component.notna().all())
        # B and C must still be separated; an inf must not flatten them together.
        by_id = dict(zip(result.frame["route_id"], component))
        self.assertNotEqual(by_id["B"], by_id["C"])

    def test_higher_net_benefit_scores_higher_all_else_equal(self):
        rows = [candidate("A", expected_saving=50000), candidate("B", expected_saving=10000)]
        result = apply_auto_vhs(pd.DataFrame(rows))
        by_id = dict(zip(result.frame["route_id"], result.frame["net_benefit_score"]))
        self.assertGreater(by_id["A"], by_id["B"])

    def test_over_shipping_does_not_raise_the_demand_fit_score(self):
        rows = [
            candidate("A", recommended_qty=10, target_shortfall=10),
            candidate("B", recommended_qty=40, target_shortfall=10),
        ]
        result = apply_auto_vhs(pd.DataFrame(rows))
        by_id = dict(zip(result.frame["route_id"], result.frame["demand_fit_score"]))
        self.assertEqual(by_id["A"], by_id["B"])
        self.assertEqual(by_id["A"], 100.0)

    def test_same_input_produces_the_same_scores(self):
        rows = [candidate(f"R{i}", expected_saving=1000 * i) for i in range(1, 6)]
        first = apply_auto_vhs(pd.DataFrame(rows))
        second = apply_auto_vhs(pd.DataFrame(rows))
        self.assertEqual(list(first.frame["vhs_score"]), list(second.frame["vhs_score"]))
        self.assertEqual(list(first.frame["vhs_rank"]), list(second.frame["vhs_rank"]))

    def test_ranking_does_not_depend_on_row_order(self):
        rows = [candidate(f"R{i}", expected_saving=1000 * (i % 3)) for i in range(1, 9)]
        forward = apply_auto_vhs(pd.DataFrame(rows))
        reverse = apply_auto_vhs(pd.DataFrame(list(reversed(rows))))
        order_a = list(forward.frame.sort_values("vhs_rank")["route_id"])
        order_b = list(reverse.frame.sort_values("vhs_rank")["route_id"])
        self.assertEqual(order_a, order_b)

    def test_tie_break_prefers_net_benefit_then_cost_then_simpler_route(self):
        # Identical component inputs except the documented tie-break keys.
        rows = [
            candidate("VIA", route_type="VIA_DC", dc_id="DC01", dc_name="가상물류센터 1"),
            candidate("CHEAP", estimated_cost=1000),
            candidate("PRICEY", estimated_cost=1000, travel_time_min=99.0),
        ]
        result = apply_auto_vhs(pd.DataFrame(rows))
        order = list(result.frame.sort_values("vhs_rank")["route_id"])
        self.assertLess(order.index("CHEAP"), order.index("PRICEY"))
        self.assertLess(order.index("CHEAP"), order.index("VIA"))

    def test_ranks_are_a_dense_permutation(self):
        rows = [candidate(f"R{i}") for i in range(1, 7)]
        result = apply_auto_vhs(pd.DataFrame(rows))
        self.assertEqual(sorted(result.frame["vhs_rank"]), list(range(1, 7)))

    def test_weights_sum_to_one_and_respect_bounds(self):
        rows = [candidate(f"R{i}", expected_saving=1000 * i) for i in range(1, 5)]
        weights = apply_auto_vhs(pd.DataFrame(rows)).analysis["weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        for component, weight in weights.items():
            low, high = WEIGHT_BOUNDS[component]
            self.assertGreaterEqual(weight, low - 1e-6, component)
            self.assertLessEqual(weight, high + 1e-6, component)


# --------------------------------------------------------------------------- #
# B. Hard constraint vs soft preference
# --------------------------------------------------------------------------- #
class ConstraintBoundaryTests(unittest.TestCase):
    HARD = (
        ("출발지=도착지", {"target_id": "S1"}),
        ("수량 0", {"recommended_qty": 0}),
        ("수량 음수", {"recommended_qty": -5}),
        ("경로 유형 불명", {"route_type": "경유"}),
        ("VIA_DC인데 DC 없음", {"route_type": "VIA_DC", "dc_id": None}),
        ("음수 비용", {"estimated_cost": -100}),
        ("비용 >= 효과", {"estimated_cost": 500, "expected_saving": 500}),
    )

    def test_hard_violations_are_removed_not_penalised(self):
        ctx = context(stock={("S1", "P1"): 100.0})
        for label, override in self.HARD:
            with self.subTest(label):
                result = evaluate_feasibility(candidate("X", **override), ctx)
                self.assertEqual(result.status, STATUS_BLOCKED, label)

    def test_non_positive_net_benefit_is_blocked_with_its_own_reason(self):
        ctx = context(stock={("S1", "P1"): 100.0})
        result = evaluate_feasibility(candidate("X", estimated_cost=9000, expected_saving=8000), ctx)
        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.reason_code, "non_positive_net_benefit")
        self.assertEqual(result.detail["net_benefit"], -1000)

    def test_uncomputable_inputs_are_soft_not_blocked(self):
        """"모르겠다"와 "손해다"는 다르게 다뤄야 한다."""
        ctx = context(stock={("S1", "P1"): 100.0})
        missing_saving = evaluate_feasibility(candidate("X", expected_saving=None), ctx)
        self.assertEqual(missing_saving.status, STATUS_CHECK)
        missing_cost = evaluate_feasibility(
            candidate("X", estimated_cost=None, move_cost=None, distance_km=None), ctx,
        )
        self.assertEqual(missing_cost.status, STATUS_CHECK)

    def test_inventory_floor_is_hard_and_oversupply_stays_soft(self):
        below_safety = evaluate_feasibility(
            candidate("X", recommended_qty=5),
            context(stock={("S1", "P1"): 10.0}, safety={("S1", "P1"): 8.0}),
        )
        self.assertEqual(below_safety.status, STATUS_BLOCKED)
        self.assertEqual(below_safety.reason_code, "inventory_floor_violation")
        oversupply = evaluate_feasibility(
            candidate("X", recommended_qty=50),
            context(stock={("S1", "P1"): 100.0}, demand={("S2", "P1"): 2.0}),
        )
        self.assertEqual(oversupply.status, STATUS_CHECK)

    def test_a_blocked_condition_is_not_also_a_score_penalty(self):
        """Blocked candidates never reach the score, so nothing is counted twice."""
        rows = [
            candidate("GOOD", expected_saving=50000),
            candidate("BAD", expected_saving=100, estimated_cost=9000),
        ]
        outcome = annotate_feasibility(rows, None)
        feasible_ids = {row["route_id"] for row in outcome["feasible"]}
        self.assertEqual(feasible_ids, {"GOOD"})
        self.assertEqual(len(outcome["blocked"]), 1)

    def test_clean_candidate_passes(self):
        ctx = context(
            stock={("S1", "P1"): 100.0}, demand={("S2", "P1"): 40.0},
            safety={("S1", "P1"): 0.0},
        )
        self.assertEqual(evaluate_feasibility(candidate("X"), ctx).status, STATUS_OK)


# --------------------------------------------------------------------------- #
# C. 수요 불확실성
# --------------------------------------------------------------------------- #
class DemandUncertaintyTests(unittest.TestCase):
    def test_scenarios_come_from_the_measured_standard_deviation(self):
        scenarios = demand_scenarios(100.0, 20.0)
        self.assertEqual(scenarios["base"], 100.0)
        self.assertEqual(scenarios["low"], 100.0 - 20.0 * DEMAND_SCENARIO_Z)
        self.assertEqual(scenarios["high"], 100.0 + 20.0 * DEMAND_SCENARIO_Z)

    def test_no_standard_deviation_means_no_invented_scenario(self):
        self.assertIsNone(demand_scenarios(100.0, None))
        self.assertIsNone(demand_scenarios(100.0, 0.0))
        self.assertIsNone(demand_scenarios(None, 20.0))
        self.assertEqual(scenario_status(10.0, None), SCENARIO_UNKNOWN)

    def test_scenario_status_reflects_how_many_scenarios_hold(self):
        scenarios = {"low": 10.0, "base": 20.0, "high": 30.0}
        self.assertEqual(scenario_status(10.0, scenarios), "안정")
        self.assertEqual(scenario_status(60.0, scenarios), "확인 필요")
        self.assertEqual(scenario_status(10_000.0, scenarios), "변동 가능성 큼")

    def test_a_candidate_frame_without_dispersion_gets_unknown_status(self):
        frame = pd.DataFrame([candidate("A")])
        annotated = annotate_decision_metrics(frame, context=context(stock={("S1", "P1"): 50.0}))
        self.assertEqual(annotated.loc[0, "demand_scenario_status"], SCENARIO_UNKNOWN)
        self.assertTrue(pd.isna(annotated.loc[0, "target_demand_std"]))

    def test_demand_risk_score_is_neutral_when_dispersion_is_unknown(self):
        rows = [candidate("A"), candidate("B", expected_saving=99999)]
        result = apply_auto_vhs(pd.DataFrame(rows))
        self.assertTrue((result.frame["demand_risk_score"] == 50.0).all())


# --------------------------------------------------------------------------- #
# D. 강건성 (Top-1 / Top-3 유지율)
# --------------------------------------------------------------------------- #
class RobustnessTests(unittest.TestCase):
    def _scored(self):
        rows = [candidate(f"R{i}", expected_saving=100000 - i * 12000, estimated_cost=500 * i)
                for i in range(1, 9)]
        return apply_auto_vhs(pd.DataFrame(rows))

    def test_every_candidate_gets_the_full_robustness_record(self):
        result = self._scored()
        records = result.analysis["candidate_robustness"]
        self.assertEqual(len(records), len(result.frame))
        for record in records.values():
            for field in ("base_rank", "top1_retention", "top3_retention",
                          "mean_rank", "best_rank", "worst_rank", "rank_shift", "status"):
                self.assertIn(field, record)
            self.assertLessEqual(record["best_rank"], record["worst_rank"])
            self.assertTrue(0.0 <= record["top1_retention"] <= 1.0)
            self.assertTrue(0.0 <= record["top3_retention"] <= 1.0)

    def test_a_clear_winner_is_stable(self):
        rows = [candidate("WIN", expected_saving=10**6)] + [
            candidate(f"R{i}", expected_saving=100) for i in range(1, 6)
        ]
        result = apply_auto_vhs(pd.DataFrame(rows))
        records = result.analysis["candidate_robustness"]
        self.assertEqual(records["WIN"]["top1_retention"], 1.0)
        self.assertEqual(records["WIN"]["status"], ROBUST)

    def test_status_values_are_the_three_user_facing_labels(self):
        result = self._scored()
        allowed = {ROBUST, "검토 필요", ROBUST_FRAGILE}
        for status in result.frame["robustness_status"]:
            self.assertIn(status, allowed)

    def test_robustness_is_reproducible(self):
        rows = [candidate(f"R{i}", expected_saving=50000 - i * 900) for i in range(1, 7)]
        first = candidate_robustness(rows, {c: 1.0 / len(COMPONENTS) for c in COMPONENTS})
        second = candidate_robustness(rows, {c: 1.0 / len(COMPONENTS) for c in COMPONENTS})
        self.assertEqual(first, second)

    def test_robustness_never_reorders_the_recommendation(self):
        result = self._scored()
        by_score = list(result.frame.sort_values("vhs_rank")["route_id"])
        ordered = list(
            result.frame.sort_values(["vhs_score", "route_id"], ascending=[False, True])["route_id"]
        )
        self.assertEqual(by_score[0], ordered[0])


# --------------------------------------------------------------------------- #
# E. 이동 수량 근거
# --------------------------------------------------------------------------- #
class QuantityTests(unittest.TestCase):
    def test_movable_quantity_respects_the_safety_floor(self):
        plan = quantity_plan(
            candidate("A", recommended_qty=10),
            context(stock={("S1", "P1"): 100.0}, safety={("S1", "P1"): 30.0}),
        )
        self.assertEqual(plan["source_movable"], 70.0)

    def test_destination_shortfall_is_demand_minus_its_own_stock(self):
        plan = quantity_plan(
            candidate("A", recommended_qty=10),
            context(stock={("S1", "P1"): 100.0, ("S2", "P1"): 15.0}, demand={("S2", "P1"): 40.0}),
        )
        self.assertEqual(plan["target_shortfall"], 25.0)

    def test_the_smaller_of_the_two_limits_is_reported(self):
        supply_bound = quantity_plan(
            candidate("A", recommended_qty=10),
            context(
                stock={("S1", "P1"): 20.0}, demand={("S2", "P1"): 500.0},
                safety={("S1", "P1"): 0.0},
            ),
        )
        self.assertEqual(supply_bound["qty_limiting_factor"], "출발 점포 이동 가능량")
        demand_bound = quantity_plan(
            candidate("A", recommended_qty=10),
            context(
                stock={("S1", "P1"): 500.0}, demand={("S2", "P1"): 30.0},
                safety={("S1", "P1"): 0.0},
            ),
        )
        self.assertEqual(demand_bound["qty_limiting_factor"], "도착 점포 부족량")

    def test_post_move_risks_are_measured_on_both_sides(self):
        plan = quantity_plan(
            candidate("A", recommended_qty=50),
            context(stock={("S1", "P1"): 60.0, ("S2", "P1"): 10.0},
                    safety={("S1", "P1"): 25.0}, demand={("S2", "P1"): 5.0}),
        )
        self.assertEqual(plan["post_move_source_remaining"], 10.0)
        self.assertEqual(plan["post_move_source_gap"], 15.0)
        self.assertEqual(plan["post_move_target_stock"], 60.0)
        self.assertEqual(plan["post_move_target_excess"], 60.0 - 5.0 * OVERSUPPLY_MULTIPLE)

    def test_unknown_inventory_yields_no_fabricated_limits(self):
        plan = quantity_plan(candidate("A"), context())
        self.assertIsNone(plan["source_movable"])
        self.assertIsNone(plan["target_shortfall"])
        self.assertIsNone(plan["qty_limiting_factor"])


# --------------------------------------------------------------------------- #
# F. 경로·DC 선택
# --------------------------------------------------------------------------- #
class RouteChoiceTests(unittest.TestCase):
    def test_via_dc_can_win_when_its_numbers_are_better(self):
        rows = [
            candidate("DIRECT1", expected_saving=10000, estimated_cost=9000, distance_km=40.0),
            candidate("VIA1", route_type="VIA_DC", dc_id="DC02", dc_name="가상물류센터 2",
                      expected_saving=90000, estimated_cost=2000, distance_km=6.0),
        ]
        result = apply_auto_vhs(pd.DataFrame(rows))
        top = result.frame.sort_values("vhs_rank").iloc[0]
        self.assertEqual(top["route_id"], "VIA1")
        self.assertEqual(top["route_type"], "VIA_DC")

    def test_direct_can_win_when_its_numbers_are_better(self):
        rows = [
            candidate("DIRECT1", expected_saving=90000, estimated_cost=2000, distance_km=4.0),
            candidate("VIA1", route_type="VIA_DC", dc_id="DC01", dc_name="가상물류센터 1",
                      expected_saving=12000, estimated_cost=9000, distance_km=44.0),
        ]
        result = apply_auto_vhs(pd.DataFrame(rows))
        self.assertEqual(result.frame.sort_values("vhs_rank").iloc[0]["route_id"], "DIRECT1")

    def test_dc_choice_follows_the_numbers_not_the_dc_id_order(self):
        for winner, loser in (("DC01", "DC02"), ("DC02", "DC01")):
            with self.subTest(winner=winner):
                rows = [
                    candidate("WIN", route_type="VIA_DC", dc_id=winner, dc_name=winner,
                              expected_saving=90000, estimated_cost=1000, distance_km=3.0),
                    candidate("LOSE", route_type="VIA_DC", dc_id=loser, dc_name=loser,
                              expected_saving=11000, estimated_cost=9000, distance_km=45.0),
                ]
                result = apply_auto_vhs(pd.DataFrame(rows))
                top = result.frame.sort_values("vhs_rank").iloc[0]
                self.assertEqual(top["dc_id"], winner)

    def test_identical_numbers_break_the_tie_towards_the_simpler_route(self):
        rows = [
            candidate("VIA", route_type="VIA_DC", dc_id="DC01", dc_name="가상물류센터 1"),
            candidate("DIRECT", route_type="DIRECT"),
        ]
        result = apply_auto_vhs(pd.DataFrame(rows))
        self.assertEqual(result.frame.sort_values("vhs_rank").iloc[0]["route_id"], "DIRECT")

    def test_the_operational_workbook_keeps_both_dcs_and_both_route_types(self):
        state = operational_result()
        recommendations = state["varo_recommendations"]
        self.assertTrue({"DIRECT", "VIA_DC"} <= {row["route_type"] for row in recommendations})
        self.assertTrue({"DC01", "DC02"} <= {str(row["dc_id"]) for row in recommendations if row.get("dc_id")})


# --------------------------------------------------------------------------- #
# G. 순효과
# --------------------------------------------------------------------------- #
class NetBenefitTests(unittest.TestCase):
    def test_net_benefit_is_saving_minus_cost(self):
        self.assertEqual(net_benefit(candidate("A", expected_saving=5000, estimated_cost=1200)), 3800)

    def test_missing_input_is_none_not_zero(self):
        self.assertIsNone(net_benefit(candidate("A", expected_saving=None)))
        self.assertIsNone(net_benefit(candidate("A", estimated_cost=None, move_cost=None)))

    def test_cost_is_counted_once(self):
        row = candidate("A", expected_saving=5000, estimated_cost=1200, move_cost=1200)
        self.assertEqual(net_benefit(row), 3800)

    def test_via_dc_cost_is_taken_from_the_candidate_not_added_twice(self):
        row = candidate("A", route_type="VIA_DC", dc_id="DC01", estimated_cost=3000,
                        expected_saving=10000, direct_cost=1200, via_dc_cost=3000)
        self.assertEqual(net_benefit(row), 7000)

    def test_kpi_total_only_sums_computable_net_benefits(self):
        kpis = calculate_overview_kpis([
            {"recommended_qty": 1, "expected_saving": 100, "net_benefit": 80},
            {"recommended_qty": 1, "expected_saving": 100, "net_benefit": None},
        ])
        self.assertEqual(kpis["total_net_benefit"], 80)

    def test_kpi_total_is_none_when_nothing_is_computable(self):
        kpis = calculate_overview_kpis([{"recommended_qty": 1, "net_benefit": None}])
        self.assertIsNone(kpis["total_net_benefit"])

    def test_operational_net_benefit_matches_saving_minus_cost_for_every_candidate(self):
        for row in operational_result()["varo_recommendations"]:
            self.assertAlmostEqual(
                float(row["net_benefit"]),
                float(row["expected_saving"]) - float(row["estimated_cost"]),
                places=4, msg=row["route_id"],
            )
            self.assertGreater(float(row["net_benefit"]), 0.0, msg=row["route_id"])


# --------------------------------------------------------------------------- #
# H. Pareto 보조 검증
# --------------------------------------------------------------------------- #
class ParetoTests(unittest.TestCase):
    def test_axis_count_stays_small_enough_to_discriminate(self):
        self.assertGreaterEqual(len(PARETO_OBJECTIVES), 2)
        self.assertLessEqual(len(PARETO_OBJECTIVES), 4)
        for objective in PARETO_OBJECTIVES:
            self.assertIn(objective, COMPONENTS)

    def test_a_dominated_candidate_is_marked(self):
        best = {name: 90.0 for name in PARETO_OBJECTIVES}
        worse = {name: 40.0 for name in PARETO_OBJECTIVES}
        rows = compute_pareto([best, worse])
        self.assertEqual(rows[0]["pareto_status"], "비지배")
        self.assertEqual(rows[1]["pareto_status"], "지배됨")

    def test_front_does_not_swallow_every_candidate_on_real_data(self):
        summary = operational_result()["varo_pipeline_result"]["pareto_analysis"]
        self.assertLess(summary["front_ratio"], 0.5)
        self.assertGreater(summary["front_size"], 0)

    def test_pareto_does_not_override_the_recommendation(self):
        state = operational_result()
        top = sort_recommendations(state["varo_recommendations"])[0]
        self.assertEqual(int(top["vhs_rank"]), 1)


# --------------------------------------------------------------------------- #
# I. Greedy 기준선
# --------------------------------------------------------------------------- #
class GreedyBaselineTests(unittest.TestCase):
    def test_baseline_is_single_objective_and_documented(self):
        rows = [
            candidate("LOW", expected_saving=100),
            candidate("HIGH", expected_saving=90000),
            candidate("MID", expected_saving=5000),
        ]
        order = [row["route_id"] for row in greedy_ranking(rows)]
        self.assertEqual(order, ["HIGH", "MID", "LOW"])
        self.assertIn("절감액", BASELINE_NAME)

    def test_baseline_tie_break_is_deterministic(self):
        rows = [
            candidate("B", expected_saving=1000, estimated_cost=500),
            candidate("A", expected_saving=1000, estimated_cost=500),
            candidate("C", expected_saving=1000, estimated_cost=100),
        ]
        order = [row["route_id"] for row in greedy_ranking(rows)]
        self.assertEqual(order, ["C", "A", "B"])
        self.assertEqual(order, [row["route_id"] for row in greedy_ranking(list(reversed(rows)))])

    def test_comparison_reports_both_sides_on_the_same_candidates(self):
        comparison = operational_result()["varo_pipeline_result"]["validation_report"]["greedy_baseline"]
        self.assertTrue(comparison["comparable"])
        self.assertEqual(comparison["candidate_count"], len(operational_result()["varo_recommendations"]))
        for side in ("vhs", "greedy"):
            for field in ("top1_route_id", "net_benefit_total", "move_cost_total", "shortage_covered"):
                self.assertIn(field, comparison[side])

    def test_empty_candidate_set_is_reported_as_not_comparable(self):
        self.assertFalse(compare_to_vhs([])["comparable"])


# --------------------------------------------------------------------------- #
# J. 신뢰도
# --------------------------------------------------------------------------- #
class ConfidenceTests(unittest.TestCase):
    def test_confidence_is_reported_as_a_grade(self):
        status = operational_result()["varo_pipeline_result"]["confidence_status"]
        self.assertIn(status["status"], {"높음", "보통", "낮음", "계산 불가"})

    def test_confidence_is_not_presented_as_a_success_probability(self):
        """The number is a relative decision indicator; the UI leads with the grade."""
        from services import decision_support

        source = decision_support.recommendation_confidence.__doc__ or ""
        self.assertNotIn("확률", source)
        page = Path("pages/recommendations.py").read_text(encoding="utf-8")
        self.assertIn('metric("추천 신뢰도", str(recommendation.get("confidence_level")', page)

    def test_no_input_means_not_computable_rather_than_a_number(self):
        from services.decision_support import recommendation_confidence

        result = recommendation_confidence([])
        self.assertEqual(result["status"], "계산 불가")
        self.assertIsNone(result["score"])

    def test_stability_tolerance_scales_with_the_candidate_count(self):
        from services.decision_support import recommendation_stability

        small = recommendation_stability({
            "scenarios": 4, "candidate_count": 4, "top1_retention_rate": 1.0, "rank_volatility": 0.4,
        })
        large = recommendation_stability({
            "scenarios": 40, "candidate_count": 60, "top1_retention_rate": 1.0, "rank_volatility": 2.0,
        })
        self.assertEqual(small["status"], "안정")
        self.assertEqual(large["status"], "안정")


# --------------------------------------------------------------------------- #
# K. Stress
# --------------------------------------------------------------------------- #
class StressTests(unittest.TestCase):
    def test_degenerate_inputs_never_raise_or_produce_nan(self):
        cases = {
            "후보 0건": [],
            "후보 1건": [candidate("A")],
            "동일 점수": [candidate(f"S{i}") for i in range(6)],
            "수요 0": [candidate("A", target_shortfall=0), candidate("B")],
            "비용 0": [candidate("A", estimated_cost=0), candidate("B")],
            "시간 0": [candidate("A", travel_time_min=0, distance_km=0), candidate("B")],
            "극단 수치": [candidate("A", expected_saving=10**15, estimated_cost=10**14), candidate("B")],
            "후보 200건": [candidate(f"R{i:03d}", expected_saving=1000 + i) for i in range(200)],
        }
        for label, rows in cases.items():
            with self.subTest(label):
                result = apply_auto_vhs(pd.DataFrame(rows))
                if not rows:
                    self.assertTrue(result.frame.empty)
                    continue
                scores = pd.to_numeric(result.frame["vhs_score"], errors="coerce")
                self.assertTrue(scores.notna().all(), label)
                self.assertTrue(scores.between(0, 100).all(), label)
                self.assertEqual(sorted(result.frame["vhs_rank"]), list(range(1, len(rows) + 1)))

    def test_all_candidates_infeasible_yields_no_recommendation_but_no_crash(self):
        rows = [candidate(f"R{i}", recommended_qty=0) for i in range(4)]
        outcome = annotate_feasibility(rows, None)
        self.assertEqual(outcome["feasible"], [])
        self.assertEqual(len(outcome["blocked"]), 4)

    def test_all_net_benefits_negative_are_all_blocked(self):
        # Distinct (product, source, target) so the duplicate rule does not fire first.
        rows = [
            candidate(f"R{i}", target_id=f"T{i}", expected_saving=10, estimated_cost=1000 + i)
            for i in range(4)
        ]
        outcome = annotate_feasibility(rows, None)
        self.assertEqual(outcome["feasible"], [])
        self.assertTrue(all(
            row["feasibility_reason_code"] == "non_positive_net_benefit" for row in outcome["blocked"]
        ))

    def test_nan_and_inf_never_reach_the_final_recommendations(self):
        for row in operational_result()["varo_recommendations"]:
            for key, value in row.items():
                if isinstance(value, float):
                    self.assertFalse(math.isnan(value), f"{row['route_id']}.{key}")
                    self.assertFalse(math.isinf(value), f"{row['route_id']}.{key}")


# --------------------------------------------------------------------------- #
# L. 화면 노출 · 버전 추적
# --------------------------------------------------------------------------- #
class PresentationTests(unittest.TestCase):
    def test_recommendation_table_shows_the_decision_columns(self):
        from components.tables import build_recommendation_rows

        rows = build_recommendation_rows(
            operational_result()["varo_recommendations"], limit=3,
            include_route_id=False, include_status=False, include_vhs=False,
        )
        self.assertTrue(rows)
        for column in ("순위", "상품", "출발 점포", "도착 점포", "경로 유형", "수량", "예상 순효과", "안정성"):
            self.assertIn(column, rows[0])
        for hidden in ("route_id", "VHS", "net_benefit_score", "weight_summary"):
            self.assertNotIn(hidden, rows[0])

    def test_component_labels_are_plain_korean(self):
        for component, label in COMPONENT_LABELS.items():
            self.assertNotIn("_", label, component)
            self.assertNotIn("score", label.lower(), component)

    def test_validation_page_labels_every_component_without_raw_field_names(self):
        """A component rename must never leave `net_benefit_score` on screen."""
        import pages.validation as validation_page

        for component in COMPONENTS:
            label = validation_page._component_name(component)
            self.assertEqual(label, COMPONENT_LABELS[component], component)
            self.assertNotIn("_score", label, component)
        source = Path("pages/validation.py").read_text(encoding="utf-8")
        self.assertNotIn("영문 필드", source)

    def test_algorithm_version_is_recorded_but_not_a_ui_column(self):
        from components.tables import build_recommendation_rows

        state = operational_result()
        pipeline = state["varo_pipeline_result"]
        self.assertEqual(pipeline["diagnostics"]["algorithm_version"], ALGORITHM_VERSION)
        self.assertEqual(pipeline["validation_report"]["algorithm_version"], ALGORITHM_VERSION)
        self.assertTrue(all(
            row.get("algorithm_version") == ALGORITHM_VERSION for row in state["varo_recommendations"]
        ))
        rows = build_recommendation_rows(state["varo_recommendations"], limit=1)
        self.assertNotIn("algorithm_version", rows[0])

    def test_analysis_records_its_own_provenance(self):
        diagnostics = operational_result()["varo_pipeline_result"]["diagnostics"]
        self.assertIn("analysis_timestamp", diagnostics)
        self.assertIn("data_signature", diagnostics)


# --------------------------------------------------------------------------- #
# M. 회귀 — 작은 fixture에서 전체 파이프라인이 계속 동작
# --------------------------------------------------------------------------- #
class SmallWorkbookRegressionTests(unittest.TestCase):
    def test_small_workbook_still_produces_a_scored_recommendation(self):
        result = run_analysis_pipeline(sample_workbook())
        self.assertEqual(result.status, "success")
        self.assertTrue(result.recommendations)
        top = sort_recommendations(result.recommendations)[0]
        self.assertIsNotNone(top["net_benefit"])
        self.assertIn(top["robustness_status"], {ROBUST, "검토 필요", ROBUST_FRAGILE, "계산 불가"})
        self.assertAlmostEqual(sum(result.vhs_analysis["weights"].values()), 1.0, places=5)

    def test_two_runs_of_the_same_workbook_agree_exactly(self):
        first = run_analysis_pipeline(sample_workbook())
        second = run_analysis_pipeline(sample_workbook())
        self.assertEqual(
            [row["route_id"] for row in sort_recommendations(first.recommendations)],
            [row["route_id"] for row in sort_recommendations(second.recommendations)],
        )
        self.assertEqual(
            [row["vhs_score"] for row in sort_recommendations(first.recommendations)],
            [row["vhs_score"] for row in sort_recommendations(second.recommendations)],
        )


if __name__ == "__main__":
    unittest.main()
