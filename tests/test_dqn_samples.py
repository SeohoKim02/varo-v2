"""DQN training sample connection tests.

Loader normalization and catalog discovery are covered deterministically with a
synthetic DQN-style workbook and a temp folder (VARO_DQN_SAMPLES_DIR), so these
never depend on the user's real sample folder. A final graceful check inspects
the real pack when it is present on the machine.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

from components.tables import build_home_top_rows
from services import sample_catalog
from services.analysis_pipeline import build_v2_state
from services.data_application import load_and_apply
from services.data_loader import load_excel_data
from services.data_validator import validate_workbook_data
from services.sample_catalog import discover_dqn_samples, dqn_sample_options, sample_id_from_filename
from services.vhs_score_engine import build_strategy_comparison
from simulation.dynamic_network import build_network_nodes
from tests.fixtures import dqn_style_workbook_sheets, dual_dc_workbook_sheets, write_dqn_style_workbook

REQUIRED_INFO_FIELDS = (
    "sample_id", "file_name", "file_path", "store_count", "dc_count", "product_count",
    "inventory_count", "route_count", "recommendation_count", "has_required_sheets",
    "validation_status", "note", "modified_at", "file_size",
)


class DqnLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.path = write_dqn_style_workbook(Path(cls._dir.name) / "syn_dqn.xlsx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.data = load_excel_data(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_separate_dcs_merged_and_store_type_not_used_as_node_type(self):
        stores = self.data["stores"]
        node_type = stores["node_type"].astype(str).str.upper()
        self.assertEqual(int((node_type == "STORE").sum()), 3)
        self.assertEqual(int((node_type == "DC").sum()), 1)
        # The Korean business tag on store_type must never leak into node_type.
        self.assertEqual(set(node_type) - {"STORE", "DC"}, set())
        self.assertNotIn("dcs", self.data)  # separate sheet folded in and dropped

    def test_recommendations_sheet_recognized_and_duplicate_route_id_promoted(self):
        recs = self.data["recommendations"]
        self.assertEqual(len(recs), 3)
        # RT01 appears twice on the source sheet; recommendation_id makes it unique.
        self.assertFalse(recs["route_id"].duplicated().any())
        self.assertIn("source_id", recs.columns)
        self.assertIn("target_id", recs.columns)
        self.assertEqual(set(recs["source_id"].astype(str)), {"S01"})

    def test_missing_presentation_and_inventory_columns_are_derived(self):
        recs = self.data["recommendations"]
        for column in ("transport_type", "recommendation_grade", "reason"):
            self.assertIn(column, recs.columns)
        inventory = self.data["inventory"]
        self.assertIn("dead_stock_qty", inventory.columns)
        self.assertIn("demand_qty", inventory.columns)

    def test_validation_passes_without_errors(self):
        report = validate_workbook_data(self.data)
        self.assertFalse(report.has_errors)
        self.assertEqual(report.summary["store_count"], 3)
        self.assertEqual(report.summary["dc_count"], 1)

    def test_pipeline_builds_greedy_and_via_dc_keeps_dc(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = build_v2_state(self.data)
        recs = state["recommendations"]
        self.assertFalse(state["validation"].has_errors)
        self.assertEqual(len(recs), 3)
        # Greedy comparison must always be available (item: Greedy always included).
        comparison = build_strategy_comparison(recs)
        self.assertEqual(len(comparison), 3)
        via = [row for row in recs if str(row.get("route_type")).upper() == "VIA_DC"]
        self.assertTrue(via)
        self.assertTrue(all(row.get("dc_id") for row in via))
        # DQN stays disconnected until the user trains it.
        self.assertTrue(all(row.get("dqn_action") in ("미연결", "비교 불가") for row in recs))

    def test_network_nodes_reflect_store_and_dc_counts(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recs = build_v2_state(self.data)["recommendations"]
        nodes = build_network_nodes(self.data, recs)
        self.assertEqual(len(nodes), 4)  # 3 STORE + 1 DC
        self.assertEqual(sum(1 for node in nodes if str(node.get("node_type")).upper() == "DC"), 1)


class DqnHomeTop5Tests(unittest.TestCase):
    def test_home_top5_hides_internal_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_dqn_style_workbook(Path(directory) / "syn_dqn.xlsx")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                recs = build_v2_state(load_excel_data(path))["recommendations"]
        rows = build_home_top_rows(recs)
        self.assertTrue(rows)
        self.assertEqual(set(rows[0]), {"순위", "상품", "출발", "도착", "경로", "수량", "예상 순효과"})
        for internal in ("route_id", "VHS", "vhs_score", "Greedy", "greedy_score", "DQN", "신뢰도", "상태"):
            self.assertNotIn(internal, rows[0])


class DqnReplaceStateTests(unittest.TestCase):
    def test_loading_dqn_sample_resets_transient_state_and_keeps_core(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_dqn_style_workbook(Path(directory) / "syn_dqn.xlsx")
            state = {
                "selected_route_id": "STALE",
                "home_speed_select": "빠름",
                "home_show_all": True,
                "show_all_routes": True,
                "simulation_speed": "빠름",
                "rec_filter_product": "OLD",
                "raw_sheet_select": "OLD",
                "dqn_training_result": {"data_signature": "old", "status": "정상"},
                "kakao_map_state": {"loaded": True},
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ok = load_and_apply(state, path, path.name, "DQN 학습 샘플")
        self.assertTrue(ok)
        # Transient view state cleared.
        self.assertNotEqual(state["selected_route_id"], "STALE")
        for key in ("home_speed_select", "home_show_all", "rec_filter_product", "raw_sheet_select"):
            self.assertNotIn(key, state)
        self.assertFalse(state["show_all_routes"])
        self.assertEqual(state["simulation_speed"], "보통")
        self.assertIsNone(state["dqn_training_result"])
        self.assertIsNone(state["kakao_map_state"])
        # Core applied data preserved.
        self.assertTrue(state["varo_recommendations"])
        self.assertFalse(state["varo_validation"].has_errors)
        self.assertEqual(state["uploaded_filename"], path.name)
        self.assertTrue(state["data_signature"])


class DqnDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        directory = Path(self._dir.name)
        write_dqn_style_workbook(directory / "Varo_DQN_sample_01_3stores_1dc_test.xlsx")
        write_dqn_style_workbook(directory / "Varo_DQN_sample_02_3stores_1dc_test.xlsx")
        # Temp lock file (~$) must be excluded.
        (directory / "~$Varo_DQN_sample_99_lock.xlsx").write_bytes(b"lock")
        # Keyword-matching but non-Varo workbook (missing required sheets) excluded.
        with pd.ExcelWriter(directory / "varo_broken.xlsx", engine="openpyxl") as writer:
            pd.DataFrame([{"a": 1}]).to_excel(writer, sheet_name="Sheet1", index=False)
        self._prev = os.environ.get("VARO_DQN_SAMPLES_DIR")
        os.environ["VARO_DQN_SAMPLES_DIR"] = str(directory)
        sample_catalog._INSPECT_CACHE.clear()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("VARO_DQN_SAMPLES_DIR", None)
        else:
            os.environ["VARO_DQN_SAMPLES_DIR"] = self._prev
        self._dir.cleanup()
        sample_catalog._INSPECT_CACHE.clear()

    def test_discovers_valid_samples_excluding_temp_and_broken(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = discover_dqn_samples()
        names = [sample.file_name for sample in samples]
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(not name.startswith("~$") for name in names))
        self.assertNotIn("varo_broken.xlsx", names)

    def test_sample_metadata_fields_present(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = discover_dqn_samples()
        info = samples[0]
        self.assertEqual(info.store_count, 3)
        self.assertEqual(info.dc_count, 1)
        self.assertTrue(info.has_required_sheets)
        self.assertTrue(info.file_path.endswith(".xlsx"))
        self.assertTrue(info.label.startswith("DQN 샘플"))
        self.assertTrue(info.modified_at)
        self.assertGreater(info.file_size, 0)
        payload = info.to_dict()
        for field_name in REQUIRED_INFO_FIELDS:
            self.assertIn(field_name, payload)

    def test_options_map_uses_labels(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            options = dqn_sample_options()
        self.assertEqual(len(options), 2)
        self.assertTrue(all(label.startswith("DQN 샘플") for label in options))


class HomeAutoRunSafetyTests(unittest.TestCase):
    def test_overview_does_not_import_dqn_or_kakao_services(self):
        source = (Path(__file__).resolve().parents[1] / "pages" / "overview.py").read_text(encoding="utf-8")
        self.assertNotIn("dqn_service", source)
        self.assertNotIn("kakao_service", source)
        self.assertNotIn("build_kakao_map", source)
        self.assertNotIn("train_dqn", source)

    def test_sample_id_from_filename_is_stable(self):
        self.assertEqual(sample_id_from_filename("Varo_DQN_sample_09_8stores_1dc_meal_kit.xlsx"), "09")
        self.assertEqual(sample_id_from_filename("Varo_DQN_sample_10_10stores_2dc_mixed.xlsx"), "10")


class DualDcStructureTests(unittest.TestCase):
    """Environment-independent multi-DC structure check.

    Replaces the environment coverage the graceful real-pack test used to give:
    a Varo-structured workbook with two distinct DCs (DC01/DC02) is discovered and
    reported correctly by the catalog, using an in-package fixture + a temp folder
    so it never depends on the user's external sample pack.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        directory = Path(self._dir.name)
        write_dqn_style_workbook(
            directory / "Varo_DQN_sample_01_3stores_2dc_dual.xlsx",
            dual_dc_workbook_sheets(),
        )
        self._prev = os.environ.get("VARO_DQN_SAMPLES_DIR")
        os.environ["VARO_DQN_SAMPLES_DIR"] = str(directory)
        sample_catalog._INSPECT_CACHE.clear()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("VARO_DQN_SAMPLES_DIR", None)
        else:
            os.environ["VARO_DQN_SAMPLES_DIR"] = self._prev
        self._dir.cleanup()
        sample_catalog._INSPECT_CACHE.clear()

    def test_dual_dc_sample_is_varo_structured_and_distinguishes_dcs(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = discover_dqn_samples()
        self.assertEqual(len(samples), 1)
        info = samples[0]
        self.assertTrue(info.has_required_sheets)
        self.assertTrue(2 <= info.store_count <= 10)
        self.assertGreaterEqual(info.dc_count, 2)

    def test_loaded_workbook_keeps_dc01_and_dc02_separate(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = discover_dqn_samples()
            data = load_excel_data(Path(samples[0].file_path))
        stores = data["stores"]
        node_type = stores["node_type"].astype(str).str.upper()
        dc_ids = set(stores.loc[node_type == "DC", "node_id"].astype(str))
        self.assertEqual(dc_ids, {"DC01", "DC02"})
        # A VIA_DC recommendation routed through DC02 keeps its dc_id (no DC mix-up).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recs = build_v2_state(data)["recommendations"]
        via_dc02 = [row for row in recs if str(row.get("dc_id")) == "DC02"]
        self.assertTrue(via_dc02)
        self.assertTrue(all(str(row.get("route_type")).upper() == "VIA_DC" for row in via_dc02))


class RealDqnPackTests(unittest.TestCase):
    """Integration check for the user's real DQN 10-pack.

    Only runs against the *real* pack (filenames like ``Varo_DQN_sample_NN`` yield
    a numeric sample_id). Unrelated keyword-fallback workbooks are ignored, and if
    no real-pack sample is present on this machine the test skips with a clear
    reason instead of failing — the deterministic structure coverage lives in
    ``DualDcStructureTests``.
    """

    def test_real_pack_when_present_is_varo_structured(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = discover_dqn_samples()
        # Real-pack samples carry a numeric two-digit id; fallbacks carry a slug.
        real_pack = [sample for sample in samples if str(sample.sample_id).isdigit()]
        if not real_pack:
            self.skipTest("실제 DQN 10-pack(Varo_DQN_sample_*)이 이 머신에 없습니다.")
        self.assertTrue(all(sample.has_required_sheets for sample in real_pack))
        self.assertTrue(all(2 <= sample.store_count <= 10 for sample in real_pack))
        self.assertTrue(any(sample.dc_count >= 2 for sample in real_pack))


if __name__ == "__main__":
    unittest.main()
