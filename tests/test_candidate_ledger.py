"""Candidate judgment-log tests: stable ids, source-row lineage, status, quantity
basis, reasons, screen-consistency, exceptions, and the review CSV.

All fixtures are built in-package (no external files). Covers spec groups
A(추적)·B(계보)·C(제외 이유)·D(이동 수량)·E(추천 이유)·F(화면 일관성)·
G(상태 예외)·H(UI 안전).
"""
from __future__ import annotations

import unittest

import pandas as pd

from services.candidate_ledger import (
    STATUS_BLOCKED_MOVE,
    STATUS_CHECK_NEEDED,
    STATUS_INSUFFICIENT,
    STATUS_NOT_COMPUTABLE,
    STATUS_RECOMMENDABLE,
    STATUS_RECOMMENDED,
    build_candidate_ledger,
    excluded_candidates,
    ledger_by_route,
    ledger_summary,
    make_candidate_id,
    quantity_basis,
    review_candidates_csv_bytes,
)
from services.candidate_lineage import build_source_references
from services.feasibility import annotate_feasibility, build_inventory_context


# --------------------------------------------------------------------------- #
# Shared in-package data
# --------------------------------------------------------------------------- #
def _inventory() -> pd.DataFrame:
    return pd.DataFrame([
        {"store_id": "S1", "product_id": "P1", "stock_qty": 120, "demand_qty": 20, "demand_std": 2},
        {"store_id": "S2", "product_id": "P1", "stock_qty": 30, "demand_qty": 40, "demand_std": 1},
        {"store_id": "S1", "product_id": "P2", "stock_qty": 10, "demand_qty": 5, "demand_std": 1},
    ])


def _raw_workbook() -> dict[str, pd.DataFrame]:
    return {
        "inventory": _inventory(),
        "products": pd.DataFrame([
            {"product_id": "P1", "product_name": "우유", "unit_price": 3000},
            {"product_id": "P2", "product_name": "만두", "unit_price": 4500},
        ]),
        "routes": pd.DataFrame([
            {"source_id": "S1", "target_id": "S2", "distance_km": 4.0, "estimated_cost": 3000, "travel_time_min": 10},
            {"source_id": "S1", "target_id": "DC01", "distance_km": 3.0, "estimated_cost": 2000, "travel_time_min": 8},
            {"source_id": "DC01", "target_id": "S2", "distance_km": 3.0, "estimated_cost": 2000, "travel_time_min": 8},
        ]),
        "stores": pd.DataFrame([
            {"node_id": "S1", "node_name": "가게1", "node_type": "STORE"},
            {"node_id": "S2", "node_name": "가게2", "node_type": "STORE"},
            {"node_id": "DC01", "node_name": "중앙DC", "node_type": "DC"},
        ]),
    }


def _rec(route_id: str, **overrides) -> dict:
    base = {
        "route_id": route_id, "product_id": "P1", "product_name": "우유",
        "source_id": "S1", "source_name": "가게1", "target_id": "S2", "target_name": "가게2",
        "dc_id": None, "route_type": "DIRECT", "recommended_qty": 20,
        "estimated_cost": 3000, "expected_saving": 5000, "vhs_score": 70,
        "confidence_score": 75,
    }
    base.update(overrides)
    return base


def _ledger(recs, *, signature="sig-abc12345", data=None):
    data = data if data is not None else {"inventory": _inventory()}
    annotated = annotate_feasibility(recs, data)["annotated"]
    return build_candidate_ledger(
        annotated, data=data, raw_data=_raw_workbook(),
        source_metadata={"filename": "inv.xlsx", "sheet_names": {
            "inventory": "재고현황", "products": "상품정보", "routes": "이동경로", "stores": "점포정보"}},
        data_signature=signature,
    )


# --------------------------------------------------------------------------- #
# A. Candidate tracking id
# --------------------------------------------------------------------------- #
class CandidateIdTests(unittest.TestCase):
    def test_id_is_stable_and_sort_independent(self):
        rec = _rec("R1")
        self.assertEqual(make_candidate_id(rec, "sigA"), make_candidate_id(dict(rec), "sigA"))

    def test_direct_and_via_dc_differ(self):
        direct = {"product_id": "P1", "source_id": "S1", "target_id": "S2", "route_type": "DIRECT", "dc_id": None}
        via = {"product_id": "P1", "source_id": "S1", "target_id": "S2", "route_type": "VIA_DC", "dc_id": "DC01"}
        self.assertNotEqual(make_candidate_id(direct, "s"), make_candidate_id(via, "s"))

    def test_dc01_and_dc02_differ(self):
        via1 = {"product_id": "P1", "source_id": "S1", "target_id": "S2", "route_type": "VIA_DC", "dc_id": "DC01"}
        via2 = {"product_id": "P1", "source_id": "S1", "target_id": "S2", "route_type": "VIA_DC", "dc_id": "DC02"}
        self.assertNotEqual(make_candidate_id(via1, "s"), make_candidate_id(via2, "s"))

    def test_different_signature_separates_candidates(self):
        rec = _rec("R1")
        self.assertNotEqual(make_candidate_id(rec, "sigA"), make_candidate_id(rec, "sigB"))

    def test_ledger_ids_survive_reordering(self):
        recs = [_rec("R1"), _rec("R2", product_id="P2", product_name="만두")]
        forward = ledger_by_route(_ledger(recs))
        backward = ledger_by_route(_ledger(list(reversed(recs))))
        self.assertEqual(forward["R1"]["candidate_id"], backward["R1"]["candidate_id"])
        self.assertEqual(forward["R2"]["candidate_id"], backward["R2"]["candidate_id"])


# --------------------------------------------------------------------------- #
# B. Original-row lineage
# --------------------------------------------------------------------------- #
class LineageTests(unittest.TestCase):
    def _refs(self, candidate, **kw):
        return build_source_references(candidate, _raw_workbook(),
                                       {"filename": "inv.xlsx", "sheet_names": {"inventory": "재고현황"}})

    def test_source_stock_row_is_linked(self):
        refs = {r["role"]: r for r in self._refs(_rec("R1"))}
        stock = refs["출발 재고"]
        self.assertTrue(stock["traceable"])
        self.assertEqual(stock["rows"], [2])           # S1/P1 is data row 1 → file row 2
        self.assertEqual(stock["value"], "120")

    def test_target_demand_row_is_linked(self):
        refs = {r["role"]: r for r in self._refs(_rec("R1"))}
        demand = refs["도착 수요"]
        self.assertTrue(demand["traceable"])
        self.assertEqual(demand["rows"], [3])          # S2/P1 is data row 2 → file row 3

    def test_route_row_is_linked(self):
        refs = [r for r in self._refs(_rec("R1")) if r["role"] == "경로 정보"]
        self.assertTrue(refs and refs[0]["traceable"])
        self.assertEqual(refs[0]["rows"], [2])

    def test_via_dc_links_two_route_rows_and_dc(self):
        via = _rec("R2", route_type="VIA_DC", dc_id="DC01", target_id="S2")
        roles = [r["role"] for r in self._refs(via)]
        self.assertEqual(roles.count("경로 정보"), 2)   # S1→DC01 and DC01→S2
        self.assertIn("DC 정보", roles)

    def test_untraceable_is_explicit_not_fabricated(self):
        missing = _rec("R9", source_id="S404", product_id="P404")
        refs = {r["role"]: r for r in self._refs(missing)}
        self.assertFalse(refs["출발 재고"]["traceable"])
        self.assertEqual(refs["출발 재고"]["rows"], [])
        self.assertIn("찾을 수 없", refs["출발 재고"]["note"])

    def test_signature_change_produces_new_candidate_lineage_key(self):
        old = ledger_by_route(_ledger([_rec("R1")], signature="old11111"))
        new = ledger_by_route(_ledger([_rec("R1")], signature="new22222"))
        self.assertNotEqual(old["R1"]["candidate_id"], new["R1"]["candidate_id"])


# --------------------------------------------------------------------------- #
# C. Exclusion reasons
# --------------------------------------------------------------------------- #
class ExclusionReasonTests(unittest.TestCase):
    def test_quantity_exceeds_stock_blocks(self):
        led = ledger_by_route(_ledger([_rec("R1", recommended_qty=99999)]))
        self.assertEqual(led["R1"]["status"], STATUS_BLOCKED_MOVE)
        self.assertTrue(led["R1"]["blocks_recommendation"])
        self.assertTrue(led["R1"]["exclusion_reasons"])

    def test_same_source_target_blocks(self):
        led = ledger_by_route(_ledger([_rec("R1", target_id="S1", target_name="가게1")]))
        self.assertEqual(led["R1"]["status"], STATUS_BLOCKED_MOVE)

    def test_no_route_is_insufficient_data(self):
        led = ledger_by_route(_ledger([_rec("R1", route_type="OTHER")]))
        self.assertEqual(led["R1"]["status"], STATUS_INSUFFICIENT)

    def test_via_dc_missing_dc_is_insufficient_data(self):
        led = ledger_by_route(_ledger([_rec("R1", route_type="VIA_DC", dc_id=None)]))
        self.assertEqual(led["R1"]["status"], STATUS_INSUFFICIENT)

    def test_cost_uncomputable_is_check_needed(self):
        led = ledger_by_route(_ledger([_rec("R1", estimated_cost=None, move_cost=None, distance_km=None)]))
        self.assertEqual(led["R1"]["status"], STATUS_CHECK_NEEDED)

    def test_exclusion_reason_has_no_internal_codes(self):
        led = ledger_by_route(_ledger([_rec("R1", recommended_qty=99999)]))
        blob = " ".join(led["R1"]["exclusion_reasons"])
        for internal in ("reason_code", "feasibility", "quantity_exceeds_stock", "NaN", "coercion"):
            self.assertNotIn(internal, blob)


# --------------------------------------------------------------------------- #
# D. Quantity basis
# --------------------------------------------------------------------------- #
class QuantityBasisTests(unittest.TestCase):
    def setUp(self):
        self.ctx = build_inventory_context({"inventory": _inventory()})

    def test_target_demand_limits(self):
        basis = quantity_basis(_rec("R1", recommended_qty=40), self.ctx)
        self.assertTrue(basis["computable"])
        self.assertEqual(basis["limiting_factor"], "도착 점포 부족량")

    def test_source_movable_limits(self):
        ctx = build_inventory_context({"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 12, "demand_qty": 200},
            {"store_id": "S2", "product_id": "P1", "stock_qty": 0, "demand_qty": 200},
        ])})
        basis = quantity_basis(_rec("R1", recommended_qty=12), ctx)
        self.assertEqual(basis["limiting_factor"], "출발 점포 이동 가능량")

    def test_equal_limits_reported(self):
        ctx = build_inventory_context({"inventory": pd.DataFrame([
            {"store_id": "S1", "product_id": "P1", "stock_qty": 40, "demand_qty": 0},
            {"store_id": "S2", "product_id": "P1", "stock_qty": 0, "demand_qty": 40},
        ])})
        basis = quantity_basis(_rec("R1", recommended_qty=40), ctx)
        self.assertIn("같", basis["limiting_factor"])

    def test_nan_and_inf_quantity_not_computable(self):
        for bad in (float("nan"), float("inf")):
            basis = quantity_basis(_rec("R1", recommended_qty=bad), self.ctx)
            self.assertFalse(basis["computable"])
            self.assertIsNone(basis["recommended_qty"])

    def test_zero_or_negative_quantity_is_blocked_move(self):
        for bad in (0, -5):
            led = ledger_by_route(_ledger([_rec("R1", recommended_qty=bad)]))
            self.assertTrue(led["R1"]["blocks_recommendation"])
            self.assertEqual(led["R1"]["status"], STATUS_BLOCKED_MOVE)

    def test_missing_quantity_is_not_computable(self):
        led = ledger_by_route(_ledger([_rec("R1", recommended_qty=float("nan"))]))
        self.assertEqual(led["R1"]["status"], STATUS_NOT_COMPUTABLE)


# --------------------------------------------------------------------------- #
# E. Recommendation reasons
# --------------------------------------------------------------------------- #
class RecommendationReasonTests(unittest.TestCase):
    def test_reasons_capped_at_three_and_reproducible(self):
        led1 = ledger_by_route(_ledger([_rec("R1"), _rec("R2", product_id="P2")]))
        led2 = ledger_by_route(_ledger([_rec("R1"), _rec("R2", product_id="P2")]))
        self.assertLessEqual(len(led1["R1"]["recommendation_reasons"]), 3)
        self.assertEqual(led1["R1"]["recommendation_reasons"], led2["R1"]["recommendation_reasons"])

    def test_excluded_candidate_has_no_fabricated_recommendation_reason(self):
        led = ledger_by_route(_ledger([_rec("R1", recommended_qty=99999)]))
        self.assertEqual(led["R1"]["recommendation_reasons"], [])

    def test_reasons_have_no_internal_terms(self):
        led = ledger_by_route(_ledger([_rec("R1")]))
        blob = " ".join(led["R1"]["recommendation_reasons"])
        for internal in ("vhs_score", "candidate_id", "data_signature", "NaN", "feasibility_status"):
            self.assertNotIn(internal, blob)


# --------------------------------------------------------------------------- #
# F. Screen consistency
# --------------------------------------------------------------------------- #
class ConsistencyTests(unittest.TestCase):
    def test_summary_buckets_sum_to_generated(self):
        recs = [_rec("R1"), _rec("R2", product_id="P2"),
                _rec("R3", recommended_qty=99999), _rec("R4", target_id="S1", target_name="가게1")]
        summary = ledger_summary(_ledger(recs))
        total = (summary["recommendable_total"] + summary["check_needed"]
                 + summary["blocked_move"] + summary["insufficient_data"] + summary["not_computable"])
        self.assertEqual(total, summary["generated"])

    def test_summary_matches_feasibility_counts(self):
        recs = [_rec("R1"), _rec("R3", recommended_qty=99999)]
        data = {"inventory": _inventory()}
        feas = annotate_feasibility(recs, data)["summary"]
        summary = ledger_summary(_ledger(recs))
        self.assertEqual(summary["recommendable_total"], feas["ok_count"])
        self.assertEqual(summary["check_needed"], feas["check_count"])
        self.assertEqual(summary["excluded_total"], feas["blocked_count"])

    def test_one_status_per_candidate(self):
        led = _ledger([_rec("R1")])
        record = led[0]
        self.assertIn(record["status"], {STATUS_RECOMMENDED, STATUS_RECOMMENDABLE})
        # a recommended candidate never carries exclusion reasons
        self.assertEqual(record["exclusion_reasons"], [])


# --------------------------------------------------------------------------- #
# G. Status exceptions
# --------------------------------------------------------------------------- #
class ExceptionStateTests(unittest.TestCase):
    def test_no_candidates(self):
        led = build_candidate_ledger([], data={"inventory": _inventory()})
        self.assertEqual(led, [])
        self.assertEqual(ledger_summary(led)["generated"], 0)

    def test_all_blocked(self):
        led = _ledger([_rec("R1", recommended_qty=0), _rec("R2", target_id="S1", target_name="가게1")])
        self.assertTrue(all(r["blocks_recommendation"] for r in led))
        self.assertEqual(ledger_summary(led)["recommendable_total"], 0)

    def test_single_recommendation_is_top(self):
        led = _ledger([_rec("R1")])
        self.assertEqual(led[0]["status"], STATUS_RECOMMENDED)
        self.assertTrue(led[0]["is_top"])


# --------------------------------------------------------------------------- #
# H. UI-safety of the review CSV
# --------------------------------------------------------------------------- #
class ReviewCsvTests(unittest.TestCase):
    def test_csv_is_utf8_bom_and_lists_only_excluded(self):
        recs = [_rec("R1"), _rec("R3", recommended_qty=99999)]
        led = _ledger(recs)
        payload = review_candidates_csv_bytes(led)
        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        text = payload.decode("utf-8-sig")
        self.assertIn("상태", text)
        self.assertIn("관련 원본 행", text)
        # the recommended R1 is not part of the review CSV
        self.assertEqual(len(excluded_candidates(led)), 1)

    def test_csv_has_no_internal_leakage(self):
        led = _ledger([_rec("R3", recommended_qty=99999)])
        text = review_candidates_csv_bytes(led).decode("utf-8-sig")
        for banned in ("candidate_id", "C-sig", "Traceback", "session_state", "reason_code", ".py"):
            self.assertNotIn(banned, text)


class StateResetTests(unittest.TestCase):
    def test_applying_new_data_replaces_previous_ledger(self):
        from services.app_state import apply_state_payload

        state: dict = {}
        first = {"candidate_ledger": _ledger([_rec("R1")], signature="old11111"),
                 "connected_algorithms": [], "deferred_algorithms": [],
                 "summary": {}, "excluded_dqn_artifacts": {}}
        apply_state_payload(state, {"varo_pipeline_result": first, "analysis_result": first})
        self.assertTrue(state["varo_pipeline_result"]["candidate_ledger"])
        old_id = state["varo_pipeline_result"]["candidate_ledger"][0]["candidate_id"]

        second = {"candidate_ledger": _ledger([_rec("R1")], signature="new22222"),
                  "connected_algorithms": [], "deferred_algorithms": [],
                  "summary": {}, "excluded_dqn_artifacts": {}}
        apply_state_payload(state, {"varo_pipeline_result": second, "analysis_result": second})
        new_id = state["varo_pipeline_result"]["candidate_ledger"][0]["candidate_id"]
        self.assertNotEqual(old_id, new_id)


if __name__ == "__main__":
    unittest.main()
