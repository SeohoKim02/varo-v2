"""Tests for DQN training-data quality diagnosis and balanced sample generation.

Deterministic and PyTorch-free: diagnosis and balancing are pure/label-only, and
the balanced writer is redirected into a TemporaryDirectory so the project's
outputs folder is never touched.
"""
from __future__ import annotations

import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

from services import dqn_balanced
from services import dqn_quality
from services.dqn_balanced import balance_actions, generate_balanced_sample, list_balanced_samples
from services.dqn_quality import (
    diagnose_actions,
    diagnose_sample,
    diagnosis_cache_key,
    diagnosis_progress_label,
    diagnosis_rows,
    quality_display_status,
    run_sequential_diagnosis,
)
from services.dqn_service import dqn_display_status, get_torch_runtime_status
from services.sample_catalog import DqnSampleInfo
from tests.fixtures import write_dqn_style_workbook


def _sample_info(path: Path, sample_id: str = "05") -> DqnSampleInfo:
    return DqnSampleInfo(
        sample_id=sample_id, file_name=path.name, file_path=str(path),
        store_count=3, dc_count=1, product_count=2, inventory_count=6,
        route_count=2, recommendation_count=3, has_required_sheets=True,
        validation_status="통과", note="", label=f"DQN 샘플 {sample_id}", category="테스트",
    )


def _skewed_recs(n: int = 10) -> list[dict]:
    return [{"varo_action": "보류", "route_type": "DIRECT"} for _ in range(n)]


def _balanced_recs() -> list[dict]:
    actions = ["재고 이동", "할인", "긴급 할인", "보류", "폐기"]
    return [{"varo_action": actions[i % len(actions)], "route_type": "DIRECT"} for i in range(15)]


class QualityDiagnosisTests(unittest.TestCase):
    def test_single_action_is_unstable(self):
        result = diagnose_actions(_skewed_recs(10))
        self.assertEqual(result["status"], "불안정")
        self.assertEqual(result["action_kinds"], 1)
        self.assertEqual(result["max_action_ratio"], 1.0)

    def test_ninety_percent_skew_is_review(self):
        recs = [{"varo_action": "보류", "route_type": "DIRECT"} for _ in range(9)]
        recs.append({"varo_action": "할인", "route_type": "DIRECT"})
        result = diagnose_actions(recs)
        self.assertEqual(result["status"], "검토 필요")

    def test_too_few_candidates_is_insufficient(self):
        self.assertEqual(diagnose_actions(_skewed_recs(2))["status"], "학습 부족")

    def test_even_distribution_is_normal(self):
        result = diagnose_actions(_balanced_recs())
        self.assertEqual(result["status"], "정상")
        self.assertGreaterEqual(result["action_kinds"], 4)

    def test_basis_marks_data_quality_not_model(self):
        self.assertIn("모델 성능 아님", diagnose_actions(_balanced_recs())["basis"])


class DisplayLabelTests(unittest.TestCase):
    """UI labels soften the raw statuses without changing the stored values."""

    def test_dqn_display_labels(self):
        self.assertEqual(dqn_display_status("정상"), "비교 가능")
        self.assertEqual(dqn_display_status("검토 필요"), "데이터 확인 필요")
        self.assertEqual(dqn_display_status("불안정"), "데이터 편향 큼")
        self.assertEqual(dqn_display_status("학습 필요"), "학습 전")
        self.assertEqual(dqn_display_status("과거 결과"), "이전 데이터 결과")
        self.assertEqual(dqn_display_status("실행 환경 필요"), "실행 환경 필요")
        self.assertEqual(dqn_display_status(None), "-")

    def test_quality_display_labels(self):
        self.assertEqual(quality_display_status("정상"), "균형 양호")
        self.assertEqual(quality_display_status("불안정"), "데이터 편향 큼")
        self.assertEqual(quality_display_status("검토 필요"), "데이터 확인 필요")
        self.assertEqual(quality_display_status("학습 부족"), "후보 수 부족")

    def test_raw_statuses_untouched_by_display_layer(self):
        result = diagnose_actions(_skewed_recs(10))
        self.assertEqual(result["status"], "불안정")  # raw value preserved for logic


def _rich_recs() -> list[dict]:
    """9 candidates with varied feature scores so >= 4 balanced actions are possible."""
    rows = []
    for i in range(9):
        rows.append({
            "recommended_qty": 10 + i,
            "expected_saving": 1000 * i,
            "distance_km": 1.0 + i,
            "route_type": "VIA_DC" if i % 3 == 0 else "DIRECT",
            "savings_score": (i * 11) % 100,
            "feasibility_score": (i * 17) % 100,
            "demand_fit_score": (i * 23) % 100,
            "promotion_score": (i * 29) % 100,
            "route_cost_score": (i * 13) % 100,
            "disposal_risk_score": (i * 37) % 100,
            "varo_action": "보류",
        })
    return rows


class BalanceActionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = _rich_recs()

    def test_balance_produces_at_least_four_actions(self):
        balanced, meta = balance_actions(self.recs)
        self.assertEqual(len(balanced), len(self.recs))
        self.assertGreaterEqual(meta["action_kinds"], 4)

    def test_balance_reduces_dominant_ratio(self):
        balanced, meta = balance_actions(self.recs)
        counts = Counter(row["varo_action"] for row in balanced)
        max_ratio = max(counts.values()) / len(balanced)
        self.assertLess(max_ratio, 0.90)  # no longer skewed enough to be 검토 필요

    def test_core_numeric_fields_preserved(self):
        balanced, _ = balance_actions(self.recs)
        for original, derived in zip(self.recs, balanced):
            for field in ("recommended_qty", "expected_saving", "distance_km", "route_type"):
                self.assertEqual(original.get(field), derived.get(field))

    def test_dispose_only_when_risk_high(self):
        # A low-disposal-risk candidate must never be labelled 폐기.
        rows = [{"disposal_risk_score": 10.0, "savings_score": 80.0, "feasibility_score": 80.0,
                 "route_type": "DIRECT"} for _ in range(6)]
        balanced, _ = balance_actions(rows)
        self.assertNotIn("폐기", {row["varo_action"] for row in balanced})


class BalancedGenerationTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._saved = dqn_balanced.BALANCED_DIR
        dqn_balanced.BALANCED_DIR = Path(self._dir.name)

    def tearDown(self):
        dqn_balanced.BALANCED_DIR = self._saved
        self._dir.cleanup()

    def test_generate_writes_provenance_and_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_dqn_style_workbook(Path(directory) / "syn.xlsx")
            info = DqnSampleInfo(
                sample_id="07", file_name=path.name, file_path=str(path),
                store_count=3, dc_count=1, product_count=2, inventory_count=6,
                route_count=2, recommendation_count=3, has_required_sheets=True,
                validation_status="통과", note="", label="DQN 샘플 07", category="테스트",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = generate_balanced_sample(info)
            self.assertTrue(result["ok"])
            out_path = Path(result["path"])
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.parent, dqn_balanced.BALANCED_DIR)
            payload = dqn_balanced.load_balanced_payload(out_path)
            for key in ("original_sample_id", "derived_from", "generated_at", "balance_policy", "recommendations"):
                self.assertIn(key, payload)
            self.assertEqual(payload["original_sample_id"], "07")
            listed = list_balanced_samples()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["original_sample_id"], "07")


class DiagnosisCacheAndProgressTests(unittest.TestCase):
    def setUp(self):
        dqn_quality._DIAG_CACHE.clear()
        self.addCleanup(dqn_quality._DIAG_CACHE.clear)

    def test_cache_key_stable_for_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_dqn_style_workbook(Path(directory) / "syn.xlsx")
            info = _sample_info(path)
            self.assertEqual(diagnosis_cache_key(info), diagnosis_cache_key(info))

    def test_cache_key_changes_when_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_dqn_style_workbook(Path(directory) / "syn.xlsx")
            info = _sample_info(path)
            before = diagnosis_cache_key(info)
            # rewrite the workbook so size/mtime differ
            write_dqn_style_workbook(path)
            import os
            os.utime(path, (before[1] + 100, before[1] + 100))
            self.assertNotEqual(diagnosis_cache_key(info), before)

    def test_second_diagnosis_is_served_from_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_dqn_style_workbook(Path(directory) / "syn.xlsx")
            info = _sample_info(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                first = diagnose_sample(info)
                self.assertIn(str(path), dqn_quality._DIAG_CACHE)
                second = diagnose_sample(info)  # no re-load
            self.assertIs(first, second)

    def test_progress_label_softens_status_and_has_no_traceback(self):
        label = diagnosis_progress_label(3, 10, "sample_03.xlsx", "불안정")
        self.assertIn("3/10", label)
        self.assertIn("sample_03.xlsx", label)
        self.assertIn("데이터 편향 큼", label)  # softened, not raw 불안정
        self.assertNotIn("불안정", label)
        self.assertEqual(diagnosis_progress_label(1, 5, "a.xlsx"), "1/5 · a.xlsx")

    def test_diagnosis_rows_use_softened_labels(self):
        rows = diagnosis_rows([diagnose_actions(_skewed_recs(10), sample_id="01")])
        self.assertEqual(rows[0]["품질 상태"], "데이터 편향 큼")

    def test_run_sequential_diagnosis_reports_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            infos = []
            for sid in ("01", "02"):
                path = write_dqn_style_workbook(Path(directory) / f"syn_{sid}.xlsx")
                infos.append(_sample_info(path, sid))
            events = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = run_sequential_diagnosis(
                    infos, on_progress=lambda i, total, name, status: events.append((i, total, name, status))
                )
            self.assertEqual(len(results), 2)
            # a before(status=None) and after(status set) event per sample
            self.assertEqual(len(events), 4)
            self.assertEqual(events[0][:3], (1, 2, infos[0].file_name))
            self.assertIsNone(events[0][3])
            self.assertIsNotNone(events[1][3])
            self.assertTrue(all("status" in r for r in results))


class TorchRuntimeStatusTests(unittest.TestCase):
    def test_returns_full_shape(self):
        status = get_torch_runtime_status()
        for key in ("available", "version", "device", "cuda_available", "can_train", "message"):
            self.assertIn(key, status)
        self.assertIsInstance(status["available"], bool)
        self.assertIsInstance(status["can_train"], bool)
        self.assertIsInstance(status["cuda_available"], bool)

    def test_missing_torch_is_env_needed_not_error(self):
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("mocked missing torch")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = _blocked
        try:
            status = get_torch_runtime_status()
        finally:
            builtins.__import__ = real_import
        self.assertFalse(status["available"])
        self.assertFalse(status["can_train"])
        self.assertFalse(status["cuda_available"])
        self.assertEqual(status["message"], "PyTorch 설치 필요")

    def test_display_labels_never_show_raw_review_or_unstable(self):
        self.assertEqual(dqn_display_status("검토 필요"), "데이터 확인 필요")
        self.assertEqual(quality_display_status("검토 필요"), "데이터 확인 필요")


if __name__ == "__main__":
    unittest.main()
