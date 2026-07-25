"""DQN training-flow tests: filename stem, batch guard, and a real tiny run.

Output paths are patched into a TemporaryDirectory so test runs never pollute
the project's outputs/dqn folder. The real-training test only runs when PyTorch
is installed; without it the flow must degrade to 실행 환경 필요 with no files.
"""
from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

from services import dqn_balanced, dqn_service
from services.dqn_batch import (
    comparison_display_rows,
    compare_samples,
    save_comparison_report,
    train_dqn_on_sample,
    train_dqn_sample_batch,
)
from services.dqn_service import _training_stem, is_torch_available
from services.sample_catalog import DqnSampleInfo
from tests.fixtures import write_dqn_style_workbook


def _info(path: Path) -> DqnSampleInfo:
    return DqnSampleInfo(
        sample_id="77", file_name=path.name, file_path=str(path),
        store_count=3, dc_count=1, product_count=2, inventory_count=6,
        route_count=2, recommendation_count=3, has_required_sheets=True,
        validation_status="통과", note="테스트", label="DQN 샘플 77", category="테스트",
    )


class _PatchedOutputs:
    """Redirect dqn_service save targets into a temp dir for one test."""

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        self._saved = (dqn_service.OUTPUT_DIR, dqn_service.LATEST_JSON, dqn_service.LATEST_MODEL)
        dqn_service.OUTPUT_DIR = root
        dqn_service.LATEST_JSON = root / "latest_dqn_result.json"
        dqn_service.LATEST_MODEL = root / "latest_dqn_model.pt"
        return root

    def __exit__(self, *exc):
        dqn_service.OUTPUT_DIR, dqn_service.LATEST_JSON, dqn_service.LATEST_MODEL = self._saved
        self._dir.cleanup()
        return False


class TrainingStemTests(unittest.TestCase):
    def test_stem_contains_all_required_parts(self):
        stem = _training_stem("07", 6, 1, 300, 0.001, "20260702_120000")
        self.assertEqual(stem, "dqn_07_s6dc1_original_ep300_lr0p001_20260702_120000")

    def test_stem_marks_balanced_variant(self):
        stem = _training_stem("07", 6, 1, 300, 0.001, "t", variant="balanced")
        self.assertIn("_balanced_", stem)

    def test_stem_sanitizes_sample_id(self):
        stem = _training_stem("샘플/01?", 2, 1, 50, 0.01, "t")
        self.assertNotIn("/", stem)
        self.assertNotIn("?", stem)


class BatchWithoutTorchTests(unittest.TestCase):
    def setUp(self):
        self._orig = dqn_service.is_torch_available
        dqn_service.is_torch_available = lambda: False

    def tearDown(self):
        dqn_service.is_torch_available = self._orig

    def test_batch_returns_env_required_rows_and_writes_nothing(self):
        infos = [_info(Path("missing_a.xlsx")), _info(Path("missing_b.xlsx"))]
        with _PatchedOutputs() as out_root:
            rows = train_dqn_sample_batch(infos, episodes=50)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["상태"], "실행 환경 필요")
            self.assertEqual(list(out_root.iterdir()), [])  # no fake artifacts


class RealTrainingFlowTests(unittest.TestCase):
    @unittest.skipUnless(is_torch_available(), "PyTorch 실행 환경이 없어 실제 학습은 건너뜁니다.")
    def test_sample_training_saves_named_result_with_required_fields(self):
        with tempfile.TemporaryDirectory() as directory, _PatchedOutputs() as out_root:
            path = write_dqn_style_workbook(Path(directory) / "syn_train.xlsx")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row, payload = train_dqn_on_sample(_info(path), episodes=25, learning_rate=0.005)
            self.assertIsNotNone(payload)
            for field_name in (
                "data_signature", "sample_id", "sample_name", "store_count", "dc_count",
                "episodes", "learning_rate", "created_at", "action_distribution",
                "reward_history", "loss_history", "final_status", "stability_status",
                "candidate_count", "dqn_action_by_route", "dqn_confidence_by_route",
                "dqn_reference_by_route",
            ):
                self.assertIn(field_name, payload)
            self.assertEqual(payload["sample_id"], "77")
            self.assertEqual(payload["sample_name"], path.name)
            self.assertEqual(payload["stability_status"], payload["status"])
            self.assertEqual(len(payload["loss_history"]), payload["episodes"])
            result_path = Path(payload["result_path"])
            self.assertEqual(result_path.parent, out_root)
            self.assertTrue(result_path.exists())
            for token in ("77", "s3dc1", "ep25", "lr0p005"):
                self.assertIn(token, result_path.name)
            self.assertTrue((out_root / "latest_dqn_result.json").exists())
            self.assertEqual(row["상태"], payload["status"])


_REPORT_FIELDS = (
    "sample_id", "sample_name", "variant", "store_count", "dc_count", "candidate_count",
    "action_distribution", "prediction_distribution", "initial_loss", "final_loss",
    "reward_summary", "stability_status", "dqn_status", "dqn_reflection_available",
    "data_signature_match", "latest_result_path", "model_path",
)


class ComparisonReportTests(unittest.TestCase):
    def test_display_rows_soften_status_and_count_variants(self):
        entries = [
            {"sample_id": "01", "variant": "original", "candidate_count": 4,
             "action_distribution": {"보류": 4}, "prediction_distribution": {"보류": 4},
             "dqn_status": "불안정", "dqn_reflection_available": False},
            {"sample_id": "01", "variant": "balanced", "candidate_count": 4,
             "action_distribution": {"재고 이동": 1, "할인": 1, "폐기": 1, "보류": 1},
             "prediction_distribution": {"재고 이동": 2, "할인": 2},
             "initial_loss": 0.05, "final_loss": 0.01,
             "dqn_status": "정상", "dqn_reflection_available": True},
        ]
        rows = comparison_display_rows(entries)
        self.assertEqual(rows[0]["구분"], "원본")
        self.assertEqual(rows[0]["상태"], "데이터 편향 큼")  # softened
        self.assertEqual(rows[0]["VHS 반영 가능"], "참고만")
        self.assertEqual(rows[1]["구분"], "균형형")
        self.assertEqual(rows[1]["상태"], "비교 가능")
        self.assertEqual(rows[1]["VHS 반영 가능"], "가능")
        self.assertEqual(rows[1]["loss 시작→끝"], "0.05→0.01")

    def test_save_report_writes_json_under_outputs(self):
        with _PatchedOutputs() as out_root:
            path = save_comparison_report([{"sample_id": "01", "variant": "original", "dqn_status": "불안정"}])
            self.assertTrue(Path(path).exists())
            self.assertEqual(Path(path).parent, out_root)
            import json
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(len(payload["entries"]), 1)
            self.assertIn("generated_at", payload)


class CompareWithoutTorchTests(unittest.TestCase):
    def setUp(self):
        self._orig = dqn_service.is_torch_available
        dqn_service.is_torch_available = lambda: False

    def tearDown(self):
        dqn_service.is_torch_available = self._orig

    def test_compare_returns_env_stubs_and_writes_nothing(self):
        info = _info(Path("missing.xlsx"))
        with _PatchedOutputs() as out_root:
            entries = compare_samples([info], episodes=10)
            self.assertEqual(len(entries), 2)  # original + balanced
            self.assertEqual({e["variant"] for e in entries}, {"original", "balanced"})
            for entry in entries:
                self.assertEqual(entry["dqn_status"], "실행 환경 필요")
                self.assertFalse(entry["dqn_reflection_available"])
            self.assertEqual(list(out_root.iterdir()), [])


class RealCompareFlowTests(unittest.TestCase):
    @unittest.skipUnless(is_torch_available(), "PyTorch 실행 환경이 없어 실제 비교 학습은 건너뜁니다.")
    def test_compare_produces_original_and_balanced_entries(self):
        with tempfile.TemporaryDirectory() as directory, _PatchedOutputs():
            saved_balanced = dqn_balanced.BALANCED_DIR
            dqn_balanced.BALANCED_DIR = Path(directory) / "balanced"
            try:
                path = write_dqn_style_workbook(Path(directory) / "syn.xlsx")
                info = _info(path)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    entries = compare_samples([info], episodes=15, learning_rate=0.01)
            finally:
                dqn_balanced.BALANCED_DIR = saved_balanced
            self.assertEqual(len(entries), 2)
            for entry in entries:
                for field_name in _REPORT_FIELDS:
                    self.assertIn(field_name, entry)
            original = next(e for e in entries if e["variant"] == "original")
            balanced = next(e for e in entries if e["variant"] == "balanced")
            self.assertIsInstance(original["initial_loss"], float)
            self.assertIsInstance(original["final_loss"], float)
            # Balancing re-labels target actions, so the balanced variant is at
            # least as diverse as the original (the real 10-sample report shows
            # 4-6 kinds; this synthetic sample only has 3 candidates).
            self.assertGreaterEqual(
                len(balanced["action_distribution"]), len(original["action_distribution"])
            )
            self.assertGreaterEqual(len(balanced["action_distribution"]), 2)


if __name__ == "__main__":
    unittest.main()
