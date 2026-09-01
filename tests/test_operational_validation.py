"""Regression contract for the anonymized operational validation workbook.

Everything here starts from a real ``.xlsx`` file produced by
``tools/generate_anonymized_operational_workbook.py`` and goes through the same
services the UI uses (read → inspect into pending → partial exclusion → apply →
user-triggered recommendation run). Nothing builds an internal DataFrame and
calls a single service in isolation.

The workbook is generated once per test session into a temporary directory, and
the intake / analysis results are cached, so the whole module stays fast enough
to live in the normal suite.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd

from services.analysis_pipeline import find_recommendation, sort_recommendations
from services.app_state import has_applied_data
from services.data_application import (
    cancel_pending_data, commit_pending_data, prepare_pending_data, run_applied_analysis,
)
from services.data_issues import collect_data_issues, exclusion_row_refs
from services.data_loader import normalize_loaded_data
from services.data_management_view import build_data_management_view
from services.data_validator import validate_workbook_data
from services.file_reader import read_uploaded_data
from services.home_state import build_home_state
from services.partial_data import build_usable_data
from tools.generate_anonymized_operational_workbook import (
    MANIFEST_NAME, OUTPUT_DIR, WORKBOOK_NAME, generate,
)

SOURCE_TYPE = "업로드 데이터"

_TEMP_DIR: Path | None = None
_CACHE: dict[str, Any] = {}


def setUpModule() -> None:  # noqa: N802 - unittest hook
    global _TEMP_DIR
    _TEMP_DIR = Path(tempfile.mkdtemp(prefix="varo_operational_"))
    generate(_TEMP_DIR)


def tearDownModule() -> None:  # noqa: N802 - unittest hook
    if _TEMP_DIR is not None:
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)


def workbook_path() -> Path:
    assert _TEMP_DIR is not None
    return _TEMP_DIR / WORKBOOK_NAME


def manifest() -> dict[str, Any]:
    assert _TEMP_DIR is not None
    if "manifest" not in _CACHE:
        _CACHE["manifest"] = json.loads(
            (_TEMP_DIR / MANIFEST_NAME).read_text(encoding="utf-8")
        )
    return _CACHE["manifest"]


def new_inspected_state() -> dict[str, Any]:
    """A fresh session state with the workbook inspected (not applied)."""
    state: dict[str, Any] = {}
    state["_status"] = prepare_pending_data(state, str(workbook_path()), WORKBOOK_NAME, SOURCE_TYPE)
    return state


def inspected_state() -> dict[str, Any]:
    if "inspected" not in _CACHE:
        _CACHE["inspected"] = new_inspected_state()
    return _CACHE["inspected"]


def analyzed_state() -> dict[str, Any]:
    """Applied + analysed once; read-only for the tests that share it."""
    if "analyzed" not in _CACHE:
        state = new_inspected_state()
        assert commit_pending_data(state), state.get("data_apply_error")
        assert run_applied_analysis(state), state.get("analysis_run_error")
        _CACHE["analyzed"] = state
    return _CACHE["analyzed"]


def _refs(items: Any) -> set[tuple[str, int]]:
    return {
        (str(item["source_sheet"]), int(item["source_row_number"])) for item in items or []
    }


def _ids(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame.columns:
        return set()
    return {str(value).strip() for value in frame[column].dropna() if str(value).strip()}


# --------------------------------------------------------------------------- #
# A. 익명화 파일 생성
# --------------------------------------------------------------------------- #
class GeneratedWorkbookTests(unittest.TestCase):
    def test_same_seed_produces_identical_workbook_and_manifest(self):
        with tempfile.TemporaryDirectory() as other:
            generate(Path(other))
            first = pd.read_excel(workbook_path(), sheet_name=None)
            second = pd.read_excel(Path(other) / WORKBOOK_NAME, sheet_name=None)
            self.assertEqual(sorted(first), sorted(second))
            for sheet in first:
                self.assertTrue(first[sheet].equals(second[sheet]), msg=f"{sheet} 시트가 달라졌습니다")
            self.assertEqual(
                (Path(other) / MANIFEST_NAME).read_text(encoding="utf-8"),
                (_TEMP_DIR / MANIFEST_NAME).read_text(encoding="utf-8"),
            )

    def test_workbook_matches_the_real_upload_schema(self):
        data, report = read_uploaded_data(workbook_path(), WORKBOOK_NAME, return_report=True)
        self.assertTrue({"stores", "products", "inventory", "routes"}.issubset(report["raw_sheets"]))
        self.assertIn("recommendations", report["raw_sheets"])
        # Before exclusion the file really does contain problems; after the partial
        # exclusion pass it validates cleanly (asserted in LargeIntakeTests).
        self.assertTrue(validate_workbook_data(data).has_errors)
        from services.data_validator import REQUIRED_COLUMNS
        for table, columns in REQUIRED_COLUMNS.items():
            self.assertTrue(
                set(columns).issubset(data[table].columns),
                msg=f"{table}에 필수 컬럼이 없습니다: {set(columns) - set(data[table].columns)}",
            )

    def test_names_are_synthetic_and_carry_no_personal_data(self):
        data, _ = read_uploaded_data(workbook_path(), WORKBOOK_NAME, return_report=True)
        labels = [
            str(value).strip()
            for frame, column in (
                (data["stores"], "node_name"), (data["stores"], "region"),
                (data["products"], "product_name"),
            )
            for value in frame[column].dropna() if str(value).strip()
        ]
        self.assertTrue(labels)
        for label in labels:
            self.assertRegex(label, r"^(가상점포|가상물류센터|가상상품|가상권역)\s\S+$")
            self.assertNotIn("@", label)
        summary = manifest()["anonymization"]
        self.assertFalse(summary["contains_personal_data"])
        self.assertFalse(summary["contains_real_company_data"])

    def test_two_dcs_multiple_stores_products_and_both_route_types(self):
        data, _ = read_uploaded_data(workbook_path(), WORKBOOK_NAME, return_report=True)
        node_type = data["stores"]["node_type"].astype(str).str.upper()
        self.assertGreaterEqual(int((node_type == "DC").sum()), 2)
        self.assertGreater(int((node_type == "STORE").sum()), 5)
        self.assertGreater(len(data["products"]), 5)
        route_type = data["routes"]["route_type"].astype(str).str.upper()
        self.assertGreater(int((route_type == "DIRECT").sum()), 0)
        self.assertGreater(int((route_type == "VIA_DC").sum()), 0)
        rec_type = data["recommendations"]["route_type"].astype(str).str.upper()
        self.assertGreater(int((rec_type == "DIRECT").sum()), 0)
        self.assertGreater(int((rec_type == "VIA_DC").sum()), 0)
        dc_ids = set(data["recommendations"]["dc_id"].dropna().astype(str))
        self.assertTrue({"DC01", "DC02"}.issubset(dc_ids))

    def test_manifest_declares_normal_warning_and_excludable_rows(self):
        issues = manifest()["issues"]
        self.assertEqual(issues["file_blocking_count"], 0)
        self.assertGreater(issues["row_excludable_rows"], 0)
        self.assertGreater(issues["cascade_excluded_rows"], 0)
        self.assertGreater(issues["retained_warning_rows"], 0)
        expected = manifest()["expected"]
        self.assertGreater(expected["applied_rows"], expected["excluded_rows"])
        self.assertTrue(expected["apply_allowed"])

    def test_manifest_counts_match_the_written_file(self):
        data, report = read_uploaded_data(workbook_path(), WORKBOOK_NAME, return_report=True)
        scale = manifest()["scale"]
        node_type = data["stores"]["node_type"].astype(str).str.upper()
        self.assertEqual(int((node_type == "STORE").sum()), scale["store_count"])
        self.assertEqual(int((node_type == "DC").sum()), scale["dc_count"])
        self.assertEqual(len(data["products"]), scale["product_count"])
        raw = report["raw_sheets"]
        for table, expected_rows in scale["source_rows_by_table"].items():
            frame = raw[table]
            blank = frame.apply(lambda row: all(str(v).strip() in ("", "nan", "None") for v in row), axis=1)
            self.assertEqual(int((~blank).sum()), expected_rows, msg=table)


# --------------------------------------------------------------------------- #
# B. 대용량 검사 · 부분 적용
# --------------------------------------------------------------------------- #
class LargeIntakeTests(unittest.TestCase):
    def test_full_intake_chain_runs_without_exception(self):
        data, report = read_uploaded_data(workbook_path(), WORKBOOK_NAME, return_report=True)
        raw = report["raw_sheets"]
        metadata = {
            "filename": WORKBOOK_NAME, "source_type": "excel",
            "sheet_names": dict(report["raw_sheet_names"]),
        }
        normalized = normalize_loaded_data({key: frame.copy() for key, frame in raw.items()})
        self.assertIn("stores", normalized)
        issues = collect_data_issues(data, raw, metadata)
        self.assertGreater(len(issues["issues"]), 0)
        self.assertGreater(len(exclusion_row_refs(issues["issues"])), 0)
        partial = build_usable_data(data, raw, metadata)
        self.assertTrue(partial["apply_allowed"])
        self.assertFalse(partial["validation"].has_errors)

    def test_intake_row_counts_match_the_manifest(self):
        state = inspected_state()
        expected = manifest()["expected"]
        quality = state["pending_quality_summary"]
        self.assertEqual(quality["total_rows"], expected["total_rows"])
        self.assertEqual(quality["applied_rows"], expected["applied_rows"])
        self.assertEqual(quality["excluded_rows"], expected["excluded_rows"])
        self.assertEqual(quality["warning_rows"], expected["warning_rows"])
        self.assertEqual(quality["warning_included_rows"], expected["warning_included_rows"])
        self.assertEqual(dict(quality["excluded_by_table"]), expected["excluded_by_table"])
        self.assertEqual(_refs(state["pending_excluded_row_refs"]), _refs(expected["excluded_row_refs"]))
        usable = state["pending_usable_data"]
        self.assertEqual(
            {name: len(usable[name]) for name in expected["usable_rows_by_table"]},
            expected["usable_rows_by_table"],
        )

    def test_inspect_does_not_apply_and_apply_resets_previous_results(self):
        state = new_inspected_state()
        self.assertFalse(has_applied_data(state.get("varo_data")))
        self.assertEqual(list(state.get("varo_recommendations") or []), [])
        self.assertTrue(state["pending_apply_allowed"])
        self.assertTrue(commit_pending_data(state))
        self.assertTrue(has_applied_data(state["varo_data"]))
        self.assertEqual(state["varo_recommendations"], [])
        self.assertEqual(state["varo_pipeline_result"], {})
        self.assertTrue(state["analysis_run_required"])
        self.assertFalse([key for key in state if key.startswith("pending_") and state.get(key)])

    def test_applied_data_is_kept_while_a_new_file_is_inspected(self):
        state = new_inspected_state()
        self.assertTrue(commit_pending_data(state))
        applied_signature = state["data_signature"]
        applied_rows = len(state["varo_data"]["inventory"])
        prepare_pending_data(state, str(workbook_path()), WORKBOOK_NAME, SOURCE_TYPE)
        self.assertEqual(state["data_signature"], applied_signature)
        self.assertEqual(len(state["varo_data"]["inventory"]), applied_rows)
        cancel_pending_data(state)
        self.assertEqual(state["data_signature"], applied_signature)

    def test_retained_warning_rows_stay_and_excluded_values_are_gone(self):
        state = inspected_state()
        retained = _refs(manifest()["expected"]["retained_warning_row_refs"])
        excluded = _refs(state["pending_excluded_row_refs"])
        self.assertTrue(retained)
        self.assertFalse(retained & excluded)
        usable = state["pending_usable_data"]
        stock = pd.to_numeric(usable["inventory"]["stock_qty"], errors="coerce")
        self.assertFalse(bool((stock < 0).any()))
        self.assertFalse(bool(stock.isna().any()))
        self.assertFalse(bool(usable["inventory"].duplicated(["store_id", "product_id"]).any()))
        self.assertFalse(bool(usable["inventory"]["store_id"].astype(str).str.strip().eq("").any()))
        quantity = pd.to_numeric(usable["recommendations"]["recommended_qty"], errors="coerce")
        self.assertFalse(bool((quantity <= 0).any()))
        self.assertTrue(
            set(usable["recommendations"]["route_type"].astype(str).str.upper()) <= {"DIRECT", "VIA_DC"}
        )


# --------------------------------------------------------------------------- #
# C. 다중 DC
# --------------------------------------------------------------------------- #
class MultiDcTests(unittest.TestCase):
    def _broken_dc_workbook(self, directory: Path, dc_id: str) -> Path:
        _data, report = read_uploaded_data(workbook_path(), WORKBOOK_NAME, return_report=True)
        sheets = {key: frame.copy() for key, frame in report["raw_sheets"].items()}
        stores = sheets["stores"]
        position = stores.index[stores["node_id"].astype(str) == dc_id][0]
        stores.loc[position, ["node_name", "store_name"]] = ""
        path = directory / f"broken_{dc_id}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for key, frame in sheets.items():
                frame.to_excel(
                    writer, sheet_name="v2_recommendations" if key == "recommendations" else key,
                    index=False,
                )
        return path

    def test_both_dcs_survive_and_keep_their_own_routes(self):
        usable = inspected_state()["pending_usable_data"]
        stores, routes = usable["stores"], usable["routes"]
        node_type = stores["node_type"].astype(str).str.upper()
        dc_ids = sorted(str(value) for value in stores.loc[node_type == "DC", "node_id"])
        self.assertEqual(dc_ids, ["DC01", "DC02"])
        names = set(stores.loc[node_type == "DC", "node_name"].astype(str))
        self.assertEqual(len(names), 2, msg="DC 이름이 하나로 합쳐졌습니다")
        for dc_id in dc_ids:
            legs = routes[
                (routes["source_id"].astype(str) == dc_id) | (routes["target_id"].astype(str) == dc_id)
            ]
            self.assertGreater(len(legs), 0, msg=f"{dc_id} 경유 경로가 없습니다")

    def test_via_dc_recommendations_use_their_own_dc_legs(self):
        usable = inspected_state()["pending_usable_data"]
        edges = {
            (str(row["source_id"]).strip(), str(row["target_id"]).strip())
            for _, row in usable["routes"].iterrows()
        }
        recommendations = usable["recommendations"]
        via = recommendations[recommendations["route_type"].astype(str).str.upper() == "VIA_DC"]
        self.assertGreater(len(via), 0)
        for _, row in via.iterrows():
            dc_id = str(row["dc_id"]).strip()
            self.assertIn((str(row["source_id"]).strip(), dc_id), edges)
            self.assertIn((dc_id, str(row["target_id"]).strip()), edges)
        direct = recommendations[recommendations["route_type"].astype(str).str.upper() == "DIRECT"]
        self.assertTrue(bool(direct["dc_id"].isna().all()), msg="DIRECT 추천에 DC가 붙었습니다")

    def test_one_dc_failure_leaves_the_other_dc_and_direct_paths_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            for broken, survivor in (("DC01", "DC02"), ("DC02", "DC01")):
                path = self._broken_dc_workbook(Path(directory), broken)
                state: dict[str, Any] = {}
                prepare_pending_data(state, str(path), path.name, SOURCE_TYPE)
                recommendations = state["pending_usable_data"]["recommendations"]
                remaining = set(recommendations["dc_id"].dropna().astype(str))
                self.assertIn(survivor, remaining, msg=f"{broken} 오류가 {survivor}까지 지웠습니다")
                self.assertNotIn(broken, remaining)
                direct = recommendations[recommendations["route_type"].astype(str).str.upper() == "DIRECT"]
                self.assertGreater(len(direct), 0, msg="DC 오류가 DIRECT 추천까지 제거했습니다")

    def test_candidate_generation_uses_every_dc_not_only_the_first(self):
        """Regression: 두 번째 DC로만 연결되는 구간이 후보에서 사라지지 않아야 한다."""
        from services.candidate_generator import _resolve_route, _route_lookup, _store_ids_by_type

        data, _ = read_uploaded_data(workbook_path(), WORKBOOK_NAME, return_report=True)
        store_ids, dc_ids = _store_ids_by_type(data["stores"])
        self.assertEqual(sorted(dc_ids), ["DC01", "DC02"])
        routes = data["routes"]
        lookup = _route_lookup(routes)
        pairs = []
        for dc_id in dc_ids:
            legs = [
                store for store in store_ids
                if (store, dc_id) in lookup and (dc_id, store) in lookup
            ]
            self.assertGreaterEqual(len(legs), 2, msg=f"{dc_id} 경유 가능한 점포가 부족합니다")
            pairs.append((dc_id, legs[0], legs[1]))
        for dc_id, source, target in pairs:
            without_direct = {
                key: value for key, value in lookup.items() if key != (source, target)
            }
            resolved = _resolve_route(source, target, dc_ids, without_direct)
            self.assertIsNotNone(resolved, msg=f"{dc_id} 경유 후보가 사라졌습니다")
            self.assertEqual(resolved["route_type"], "VIA_DC")
            self.assertEqual(resolved["dc_id"], dc_id, msg="후보 생성에서 DC가 뒤바뀌었습니다")

    def test_route_detail_dc_matches_the_recommendation_table(self):
        state = analyzed_state()
        recommendations = state["varo_recommendations"]
        for item in recommendations:
            detail = find_recommendation(recommendations, str(item["route_id"]))
            self.assertEqual(detail.get("dc_id"), item.get("dc_id"))
            self.assertEqual(detail.get("route_type"), item.get("route_type"))
        final_dcs = {str(item["dc_id"]) for item in recommendations if item.get("dc_id")}
        self.assertTrue({"DC01", "DC02"}.issubset(final_dcs))


# --------------------------------------------------------------------------- #
# D. 참조 관계
# --------------------------------------------------------------------------- #
class ReferenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.usable = inspected_state()["pending_usable_data"]
        self.node_ids = _ids(self.usable["stores"], "node_id")
        self.product_ids = _ids(self.usable["products"], "product_id")

    def test_no_orphan_store_or_product_references_remain(self):
        self.assertLessEqual(_ids(self.usable["inventory"], "store_id"), self.node_ids)
        self.assertLessEqual(_ids(self.usable["inventory"], "product_id"), self.product_ids)
        for column in ("source_id", "target_id"):
            self.assertLessEqual(_ids(self.usable["routes"], column), self.node_ids)
            self.assertLessEqual(_ids(self.usable["recommendations"], column), self.node_ids)
        self.assertLessEqual(_ids(self.usable["recommendations"], "product_id"), self.product_ids)

    def test_every_recommendation_still_has_a_real_route(self):
        edges = {
            (str(row["source_id"]).strip(), str(row["target_id"]).strip())
            for _, row in self.usable["routes"].iterrows()
        }
        node_type = self.usable["stores"]["node_type"].astype(str).str.upper()
        dc_ids = {str(v).strip() for v in self.usable["stores"].loc[node_type == "DC", "node_id"]}
        for _, row in self.usable["recommendations"].iterrows():
            source, target = str(row["source_id"]).strip(), str(row["target_id"]).strip()
            if str(row["route_type"]).upper() == "DIRECT":
                self.assertIn((source, target), edges)
            else:
                dc_id = str(row["dc_id"]).strip()
                self.assertIn(dc_id, dc_ids)
                self.assertIn((source, dc_id), edges)
                self.assertIn((dc_id, target), edges)

    def test_broken_master_row_cascades_exactly_once(self):
        excluded_store = manifest()["expected"]["excluded_store_id"]
        self.assertNotIn(excluded_store, self.node_ids)
        self.assertNotIn(excluded_store, _ids(self.usable["inventory"], "store_id"))
        self.assertNotIn(excluded_store, _ids(self.usable["routes"], "source_id"))
        self.assertNotIn(excluded_store, _ids(self.usable["routes"], "target_id"))
        self.assertNotIn(excluded_store, _ids(self.usable["recommendations"], "source_id"))
        self.assertNotIn(excluded_store, _ids(self.usable["recommendations"], "target_id"))

    def test_healthy_master_rows_are_not_over_removed(self):
        scale = manifest()["scale"]
        self.assertEqual(len(self.node_ids), scale["store_count"] + scale["dc_count"] - 1)
        self.assertEqual(len(self.product_ids), scale["product_count"])
        expected = manifest()["expected"]["usable_rows_by_table"]
        for table, rows in expected.items():
            self.assertEqual(len(self.usable[table]), rows, msg=table)

    def test_cascade_excluded_recommendation_is_the_expected_one(self):
        route_ids = _ids(self.usable["recommendations"], "route_id")
        self.assertNotIn(manifest()["recommendations"]["cascade_excluded_route_id"], route_ids)
        for blocked in manifest()["recommendations"]["feasibility_blocked_route_ids"]:
            self.assertIn(blocked, route_ids, msg="실행 가능성 문제는 검사 단계에서 제외하지 않습니다")


# --------------------------------------------------------------------------- #
# E. 추천 파이프라인
# --------------------------------------------------------------------------- #
class RecommendationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.state = analyzed_state()
        self.pipeline = self.state["varo_pipeline_result"]
        self.recommendations = self.state["varo_recommendations"]

    def test_analysis_only_runs_when_requested_and_then_succeeds(self):
        self.assertFalse(self.state["analysis_run_required"])
        self.assertIn(self.pipeline["status"], ("success", "partial"))
        self.assertEqual(self.pipeline["diagnostics"]["algorithm_errors"], [])
        self.assertGreaterEqual(
            len(self.recommendations), manifest()["expected"]["minimum_recommendation_count"]
        )

    def test_excluded_rows_never_reach_candidate_generation(self):
        route_ids = {str(item["route_id"]) for item in self.pipeline["candidate_ledger"]}
        self.assertNotIn(manifest()["recommendations"]["cascade_excluded_route_id"], route_ids)
        self.assertEqual(
            self.pipeline["ledger_summary"]["generated"],
            len(self.state["varo_data"]["recommendations"]),
        )

    def test_infeasible_candidates_are_blocked_before_ranking(self):
        blocked = {str(item["route_id"]) for item in self.pipeline["excluded_candidates"]}
        expected = set(manifest()["recommendations"]["feasibility_blocked_route_ids"])
        self.assertTrue(expected.issubset(blocked))
        final_ids = {str(item["route_id"]) for item in self.recommendations}
        self.assertFalse(final_ids & expected)

    def test_candidate_status_counts_add_up(self):
        summary = self.pipeline["ledger_summary"]
        self.assertEqual(sum(summary["status_counts"].values()), summary["generated"])
        feasibility = self.pipeline["feasibility_summary"]
        self.assertEqual(
            feasibility["feasible_count"] + feasibility["blocked_count"], feasibility["total"]
        )
        self.assertEqual(feasibility["feasible_count"], len(self.recommendations))

    def test_algorithm_roles_are_unchanged(self):
        sources = self.pipeline["validation_report"]["calculation_sources"]
        self.assertIn("vhs_score_engine", str(sources["vhs"]))
        self.assertIn("heuristic_optimizer", str(sources["greedy"]))
        self.assertTrue(self.pipeline["pareto_analysis"])
        self.assertFalse(self.pipeline["diagnostics"]["dqn_artifacts_read"])
        self.assertTrue(all(str(item["dqn_action"]) == "미연결" for item in self.recommendations))
        self.assertIsNone(self.state.get("dqn_training_result"))

    def test_final_ranking_comes_from_the_single_recommendation_list(self):
        ordered = sort_recommendations(self.recommendations)
        self.assertEqual(
            str(self.state["selected_route_id"]), str(ordered[0]["route_id"]),
        )
        scores = [item.get("vhs_score") for item in ordered]
        self.assertTrue(all(value is not None for value in scores))

    def test_quantities_are_backed_by_real_stock_and_never_faked(self):
        stock = {
            (str(row["store_id"]), str(row["product_id"])): float(row["stock_qty"])
            for _, row in self.state["varo_data"]["inventory"].iterrows()
        }
        for item in self.recommendations:
            quantity = float(item["recommended_qty"])
            self.assertGreater(quantity, 0)
            available = stock.get((str(item["source_id"]), str(item["product_id"])))
            if available is not None:
                self.assertLessEqual(quantity, available)
            self.assertFalse(
                item.get("expected_saving") == 0 and item.get("estimated_cost") == 0,
                msg="계산 불가 값이 0으로 표시되었습니다",
            )

    def test_reasons_are_derived_from_the_computed_result(self):
        reasons = self.pipeline["reason_analysis"]["reasons"]
        top = sort_recommendations(self.recommendations)[0]
        detail = reasons[str(top["route_id"])]
        self.assertTrue(detail["sentences"])
        joined = " ".join(detail["sentences"])
        self.assertIn(f"{int(float(top['recommended_qty']))}", joined.replace(",", ""))


# --------------------------------------------------------------------------- #
# F. 화면 간 일관성
# --------------------------------------------------------------------------- #
class ScreenConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.state = analyzed_state()
        self.home = build_home_state(self.state)
        self.view = build_data_management_view(self.state)
        self.recommendations = self.state["varo_recommendations"]

    def test_home_and_recommendation_page_share_the_top_candidate(self):
        ordered = sort_recommendations(self.recommendations)
        self.assertEqual(
            str(self.home["top_recommendation"]["route_id"]), str(ordered[0]["route_id"])
        )
        self.assertEqual(self.home["recommendation_count"], len(self.recommendations))

    def test_route_detail_matches_the_recommendation_table(self):
        top = self.home["top_recommendation"]
        detail = find_recommendation(self.recommendations, self.state["selected_route_id"])
        for field in (
            "route_id", "recommended_qty", "route_type", "dc_id",
            "estimated_cost", "expected_saving", "vhs_score", "confidence", "reason",
        ):
            self.assertEqual(detail.get(field), top.get(field), msg=field)

    def test_data_management_and_home_agree_on_applied_rows(self):
        current = self.view["current"]
        quality = self.state["data_quality_summary"]
        expected = manifest()["expected"]
        self.assertEqual(current["total_rows"], expected["total_rows"])
        self.assertEqual(current["usable_rows"], expected["applied_rows"])
        self.assertEqual(current["excluded_rows"], expected["excluded_rows"])
        self.assertEqual(current["warning_rows"], quality["warning_rows"])
        self.assertEqual(current["recommendation_count"], self.home["recommendation_count"])

    def test_validation_page_counts_match_the_home_counts(self):
        ledger = self.state["varo_pipeline_result"]["ledger_summary"]
        self.assertEqual(ledger["recommendable_total"], self.home["recommendation_count"])
        self.assertEqual(ledger["excluded_total"], self.home["blocked_count"])

    def test_top_candidate_keeps_its_original_row_lineage(self):
        top = self.home["top_recommendation"]
        record = next(
            item for item in self.state["varo_pipeline_result"]["candidate_ledger"]
            if str(item["route_id"]) == str(top["route_id"])
        )
        self.assertGreater(int(record["traceable_row_count"] or 0), 0)
        self.assertEqual(record["recommended_qty"], top["recommended_qty"])

    def test_internal_identifiers_are_not_part_of_the_user_facing_view(self):
        current = self.view["current"]
        blob = json.dumps(current, ensure_ascii=False, default=str)
        for token in ("candidate_id", "signature", "pending_usable_data"):
            self.assertNotIn(token, blob)


# --------------------------------------------------------------------------- #
# G. 성능·메모리 회귀 (환경에 흔들리지 않는 넉넉한 기준만 사용)
# --------------------------------------------------------------------------- #
class ResourceRegressionTests(unittest.TestCase):
    def test_the_workbook_is_parsed_once_per_intake(self):
        import services.file_reader as file_reader

        original = file_reader.load_excel_data
        calls = {"count": 0}

        def counting(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        with mock.patch.object(file_reader, "load_excel_data", counting):
            state: dict[str, Any] = {}
            prepare_pending_data(state, str(workbook_path()), WORKBOOK_NAME, SOURCE_TYPE)
            self.assertEqual(calls["count"], 1)
            self.assertTrue(commit_pending_data(state))
            self.assertEqual(calls["count"], 1, msg="적용 단계에서 파일을 다시 읽었습니다")
            self.assertTrue(run_applied_analysis(state))
            self.assertEqual(calls["count"], 1, msg="분석 단계에서 파일을 다시 읽었습니다")

    def test_candidate_lineage_does_not_copy_the_whole_source(self):
        state = analyzed_state()
        inventory_rows = len(state["varo_data"]["inventory"])
        ledger = state["varo_pipeline_result"]["candidate_ledger"]
        self.assertTrue(ledger)
        for record in ledger:
            self.assertLess(int(record["traceable_row_count"] or 0), inventory_rows)
            self.assertLess(len(json.dumps(record, ensure_ascii=False, default=str)), 20000)

    def test_pending_intake_is_released_after_apply_and_after_cancel(self):
        applied = new_inspected_state()
        self.assertTrue(commit_pending_data(applied))
        self.assertFalse([k for k in applied if k.startswith("pending_") and applied.get(k)])
        cancelled = new_inspected_state()
        cancel_pending_data(cancelled)
        self.assertFalse([k for k in cancelled if k.startswith("pending_") and cancelled.get(k)])

    def test_whole_flow_finishes_within_a_generous_bound(self):
        import time

        start = time.perf_counter()
        state: dict[str, Any] = {}
        prepare_pending_data(state, str(workbook_path()), WORKBOOK_NAME, SOURCE_TYPE)
        commit_pending_data(state)
        run_applied_analysis(state)
        # Deliberately loose: this guards against a runaway loop, not against
        # normal machine-to-machine variation.
        self.assertLess(time.perf_counter() - start, 300.0)


# --------------------------------------------------------------------------- #
# 저장소에 포함된 검증 파일과 생성기가 어긋나지 않는지
# --------------------------------------------------------------------------- #
class CommittedArtifactTests(unittest.TestCase):
    def test_committed_validation_workbook_matches_the_generator(self):
        committed = OUTPUT_DIR / WORKBOOK_NAME
        if not committed.is_file():
            self.skipTest("validation_data 워크북이 아직 생성되지 않았습니다")
        first = pd.read_excel(committed, sheet_name=None)
        second = pd.read_excel(workbook_path(), sheet_name=None)
        self.assertEqual(sorted(first), sorted(second))
        for sheet in first:
            self.assertTrue(first[sheet].equals(second[sheet]), msg=f"{sheet} 시트가 생성기와 다릅니다")
        self.assertEqual(
            json.loads((OUTPUT_DIR / MANIFEST_NAME).read_text(encoding="utf-8")), manifest(),
        )


if __name__ == "__main__":
    unittest.main()
