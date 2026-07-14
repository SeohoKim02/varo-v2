"""DQN sample catalog and in-memory balancing contracts."""
from __future__ import annotations

import unittest
from collections import Counter

from services.dqn_samples import (
    BALANCED_OUTPUT_DIR,
    DQN_SAMPLES,
    PROJECT_ROOT,
    balanced_recommendations,
    balanced_sample_metadata,
    build_dqn_training_sets,
    diagnose_dqn_training_sets,
    dqn_sample_path,
    load_sample_recommendations,
)
from services.dqn_service import data_signature_from_recommendations


class DqnSampleTests(unittest.TestCase):
    def test_generated_balanced_samples_stay_under_dqn_output(self):
        self.assertTrue(BALANCED_OUTPUT_DIR.resolve().is_relative_to((PROJECT_ROOT / "outputs").resolve()))
        self.assertEqual(BALANCED_OUTPUT_DIR.name, "dqn_balanced_samples")

    def test_catalog_exposes_samples_01_through_10(self):
        self.assertEqual([sample.label for sample in DQN_SAMPLES], [f"DQN 샘플 {index:02d}" for index in range(1, 11)])
        self.assertEqual(DQN_SAMPLES[0].mode, "original")
        if DQN_SAMPLES[0].source_path is not None:
            self.assertTrue(all(sample.mode == "original" for sample in DQN_SAMPLES))
        else:
            self.assertEqual(DQN_SAMPLES[-1].mode, "balanced")
        self.assertEqual(DQN_SAMPLES[-1].workbook.dc_count, 2)

    def test_sample_01_and_10_load_without_writing_source_files(self):
        for sample in (DQN_SAMPLES[0], DQN_SAMPLES[-1]):
            path = dqn_sample_path(sample)
            before = (path.stat().st_size, path.stat().st_mtime_ns)
            recommendations = load_sample_recommendations(sample)
            after = (path.stat().st_size, path.stat().st_mtime_ns)
            self.assertTrue(recommendations)
            self.assertEqual(before, after)

    def test_balanced_copy_preserves_signature_and_source_rows(self):
        original = load_sample_recommendations(DQN_SAMPLES[2])
        snapshot = [dict(item) for item in original]
        balanced = balanced_recommendations(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(data_signature_from_recommendations(original), data_signature_from_recommendations(balanced))
        counts = Counter(item["target_action"] for item in balanced)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        numeric_keys = ("recommended_qty", "expected_saving", "estimated_cost", "distance_km")
        for source, derived in zip(original, balanced):
            self.assertTrue(all(source.get(key) == derived.get(key) for key in numeric_keys))

    def test_balanced_metadata_records_derivation(self):
        metadata = balanced_sample_metadata("sample_10", "sample10.xlsx", generated_at="2026-07-14T12:00:00")
        self.assertEqual(metadata["original_sample_id"], "sample_10")
        self.assertEqual(metadata["derived_from"], "sample10.xlsx")
        self.assertIn("feature", metadata["balance_policy"])
        self.assertEqual(metadata["generated_at"], "2026-07-14T12:00:00")

    def test_batch_builder_returns_ten_training_sets(self):
        sets = build_dqn_training_sets()
        self.assertEqual(len(sets), 10)
        self.assertEqual(sets[0]["label"], "DQN 샘플 01")
        self.assertEqual(sets[-1]["label"], "DQN 샘플 10")
        self.assertTrue(all(item["recommendations"] for item in sets))

    def test_balanced_batch_has_feature_based_labels_without_severe_skew(self):
        sets = build_dqn_training_sets(mode="balanced")
        diagnostics = diagnose_dqn_training_sets(sets)
        self.assertEqual(len(diagnostics), 10)
        self.assertTrue(all(item["variant"] == "balanced" for item in diagnostics))
        self.assertTrue(all(item["dominant_ratio"] < 0.90 for item in diagnostics))
        self.assertTrue(all(
            "target_action" in row
            for item in sets
            for row in item["recommendations"]
        ))


if __name__ == "__main__":
    unittest.main()
