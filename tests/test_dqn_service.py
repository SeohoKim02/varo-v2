"""V2 DQN service contract tests.

No historical DQN artifact is read.  PyTorch-dependent training is not required
for these tests; the service must remain safe without it.
"""
from __future__ import annotations

import math
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.dqn_service import (
    ACTION_LABELS,
    _artifact_context,
    apply_dqn_reference_to_recommendations,
    apply_dqn_result_to_recommendations,
    build_state_vectors,
    calculate_rewards,
    can_apply_dqn_to_current_data,
    data_signature_from_recommendations,
    dqn_result_summary,
    evaluate_dqn_stability,
    get_dqn_status,
    get_torch_runtime_info,
    get_torch_status,
    get_torch_training_device,
    normalize_action,
    train_dqn_batch,
    train_dqn,
    validate_training_stability,
)


def _recommendations():
    return [
        {
            "route_id": "R001",
            "recommended_qty": 10,
            "distance_km": 2.0,
            "expected_time_min": 12,
            "move_cost": 5000,
            "expected_saving": 20000,
            "vhs_score": 80,
            "confidence_score": 90,
            "varo_action": "transfer",
        },
        {
            "route_id": "R002",
            "recommended_qty": 6,
            "distance_km": 8.0,
            "expected_time_min": 30,
            "move_cost": 11000,
            "expected_saving": 14000,
            "vhs_score": 70,
            "confidence_score": 75,
            "varo_action": "discount",
        },
    ]


def _training_recommendations():
    rows = _recommendations()
    rows.append({
        "route_id": "R003",
        "recommended_qty": 4,
        "distance_km": 5.0,
        "expected_time_min": 20,
        "move_cost": 7000,
        "expected_saving": 17000,
        "vhs_score": 75,
        "confidence_score": 82,
        "varo_action": "hold",
    })
    return rows


class DqnServiceTests(unittest.TestCase):
    def test_artifact_context_contains_training_metadata(self):
        slug = _artifact_context("sample_10", "balanced", 10, 2, 180, 0.001)
        self.assertEqual(slug, "sample_10_balanced_10stores_2dc_ep180_lr0p001")

    def test_training_exposes_fixed_seed_parameter(self):
        parameter = inspect.signature(train_dqn).parameters["seed"]
        self.assertEqual(parameter.default, 17)
    def test_action_mapping(self):
        self.assertEqual(normalize_action("direct_transfer"), "재고 이동")
        self.assertEqual(normalize_action("discount_sale"), "할인")
        self.assertEqual(normalize_action("urgent_discount"), "긴급 할인")
        self.assertEqual(normalize_action("plus_one"), "1+1")
        self.assertEqual(normalize_action("discard"), "폐기")
        self.assertEqual(normalize_action("maintain"), "보류")

    def test_state_vector_handles_missing_values(self):
        vectors = build_state_vectors([{"route_id": "R001"}])
        self.assertEqual(len(vectors), 1)
        self.assertTrue(all(value == 0.5 for value in vectors[0][:18]))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in vectors[0]))

    def test_state_vector_shape_and_bounds(self):
        vectors = build_state_vectors(_recommendations())
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), len(vectors[1]))
        for vector in vectors:
            self.assertTrue(all(0.0 <= value <= 1.0 for value in vector))

    def test_initial_status_is_training_required(self):
        status = get_dqn_status()
        self.assertEqual(status.status, "학습 필요")
        self.assertFalse(status.connected)
        self.assertFalse(status.historical_artifacts_used)

    def test_torch_status_is_safe_tuple(self):
        available, message = get_torch_status()
        self.assertIsInstance(available, bool)
        self.assertIsInstance(message, str)

    def test_missing_torch_has_short_environment_message(self):
        with patch("services.dqn_service.is_torch_available", return_value=False):
            available, message = get_torch_status()
        self.assertFalse(available)
        self.assertEqual(message, "DQN 학습 실행 환경 필요")

    def test_torch_runtime_info_reports_device_and_version(self):
        info = get_torch_runtime_info()
        self.assertIn("available", info)
        self.assertIn("status", info)
        self.assertIn("device", info)
        self.assertIn("version", info)
        if info["available"]:
            self.assertIn(info["device"], {"CPU", "GPU"})
            self.assertNotEqual(info["version"], "-")

    def test_unsupported_cuda_architecture_falls_back_to_cpu(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            get_arch_list=lambda: ["sm_86", "sm_90"],
            get_device_capability=lambda: (12, 0),
        ))
        self.assertEqual(get_torch_training_device(fake_torch), "cpu")

    def test_training_progress_uses_real_service_stages(self):
        if not get_torch_status()[0]:
            self.skipTest("DQN runtime unavailable")
        stages = []
        result = train_dqn(
            _training_recommendations(),
            episodes=20,
            progress_callback=stages.append,
        )
        self.assertEqual(stages, ["데이터 구성 중", "DQN 학습 중", "안정성 검사 중"])
        self.assertEqual(result.episodes, 20)

    def test_result_save_failure_keeps_in_memory_training_result(self):
        if not get_torch_status()[0]:
            self.skipTest("DQN runtime unavailable")
        with patch("services.dqn_service.save_dqn_result", side_effect=OSError("read only")):
            result = train_dqn(_training_recommendations(), episodes=20)
        self.assertIsNone(result.result_path)
        self.assertEqual(result.diagnostics["storage_status"], "session_only")

    def test_apply_mock_training_result_adds_detail_fields_without_scores(self):
        recommendations = _recommendations()
        result = {
            "status": "연결",
            "training_mode": "original",
            "data_signature": data_signature_from_recommendations(recommendations),
            "reflection_mode": "DQN 참고만",
            "candidate_count": 2,
            "dqn_action_by_route": {"R001": "재고 이동", "R002": "할인"},
            "dqn_confidence_by_route": {"R001": 91.5, "R002": 83.0},
            "dqn_reference_by_route": {"R001": 88.0, "R002": 79.0},
            "prediction_distribution": {"재고 이동": 1, "할인": 1},
            "target_distribution": {"재고 이동": 1, "할인": 1},
        }
        updated = apply_dqn_result_to_recommendations(recommendations, result)
        self.assertEqual(updated[0]["dqn_action"], "재고 이동")
        self.assertEqual(updated[0]["dqn_status"], "연결")
        self.assertEqual(updated[0]["dqn_correction"], 0.0)
        self.assertEqual(updated[0]["vhs_score"], recommendations[0]["vhs_score"])

    def test_weak_reflection_is_small_and_only_when_connected(self):
        recommendations = _recommendations()
        result = {
            "status": "연결",
            "training_mode": "original",
            "data_signature": data_signature_from_recommendations(recommendations),
            "reflection_mode": "DQN 약하게 반영",
            "candidate_count": 2,
            "dqn_action_by_route": {"R001": "재고 이동", "R002": "할인"},
            "dqn_confidence_by_route": {"R001": 80.0, "R002": 75.0},
            "dqn_reference_by_route": {"R001": 82.0, "R002": 74.0},
            "prediction_distribution": {"재고 이동": 1, "할인": 1},
            "target_distribution": {"재고 이동": 1, "할인": 1},
        }
        updated = apply_dqn_result_to_recommendations(recommendations, result)
        self.assertGreater(updated[0]["dqn_correction"], 0)
        self.assertLessEqual(updated[0]["dqn_correction"], 2.0)
        self.assertLessEqual(updated[0]["vhs_score"] - recommendations[0]["vhs_score"], 2.0)

    def test_review_required_does_not_change_score(self):
        recommendations = _recommendations()
        result = {
            "status": "검토 필요",
            "reflection_mode": "DQN 약하게 반영",
            "dqn_action_by_route": {"R001": "재고 이동"},
            "dqn_confidence_by_route": {"R001": 99.0},
        }
        updated = apply_dqn_result_to_recommendations(recommendations, result)
        self.assertEqual(updated[0]["dqn_correction"], 0.0)
        self.assertEqual(updated[0]["vhs_score"], recommendations[0]["vhs_score"])

    def test_action_skew_is_review_required(self):
        status, message = validate_training_stability(
            [0.1, 0.05],
            ["재고 이동"] * 9 + ["할인"],
            [0.4] * 10,
        )
        self.assertEqual(status, "검토 필요")
        self.assertIn("치우", message)

    def test_nan_loss_disables_result(self):
        status, _ = validate_training_stability([math.inf], list(ACTION_LABELS), [0.5] * len(ACTION_LABELS))
        self.assertEqual(status, "검토 필요")

    def test_summary_never_uses_historical_artifacts(self):
        summary = dqn_result_summary({"status": "연결", "historical_artifacts_used": True}, _recommendations())
        self.assertFalse(summary["historical_artifacts_used"])

    def test_reward_values_are_bounded(self):
        rewards = calculate_rewards(_recommendations())
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(0.0 <= reward <= 1.0 for reward in rewards))

    def test_data_signature_blocks_past_result(self):
        recommendations = _recommendations()
        signature = data_signature_from_recommendations(recommendations)
        result = {
            "status": "정상",
            "data_signature": "old-signature",
            "dqn_action_by_route": {"R001": "재고 이동"},
            "dqn_confidence_by_route": {"R001": 90.0},
            "dqn_reference_by_route": {"R001": 88.0},
        }
        self.assertFalse(can_apply_dqn_to_current_data(result, signature))
        updated = apply_dqn_reference_to_recommendations(recommendations, result, signature)
        self.assertEqual(updated[0]["dqn_status"], "과거 결과")
        self.assertEqual(updated[0]["dqn_reference_score"], 0.0)

    def test_normal_dqn_result_attaches_reference_score(self):
        recommendations = _recommendations()
        signature = data_signature_from_recommendations(recommendations)
        result = {
            "status": "정상",
            "training_mode": "balanced",
            "candidate_count": 2,
            "data_signature": signature,
            "dqn_action_by_route": {"R001": "재고 이동", "R002": "할인"},
            "dqn_confidence_by_route": {"R001": 90.0, "R002": 82.0},
            "dqn_reference_by_route": {"R001": 88.0, "R002": 80.0},
            "prediction_distribution": {"재고 이동": 1, "할인": 1},
            "target_distribution": {"재고 이동": 1, "할인": 1},
        }
        self.assertTrue(can_apply_dqn_to_current_data(result, signature))
        updated = apply_dqn_reference_to_recommendations(recommendations, result, signature)
        self.assertEqual(updated[0]["dqn_status"], "정상")
        self.assertEqual(updated[0]["dqn_reference_score"], 88.0)

    def test_collapsed_prediction_or_target_distribution_blocks_application(self):
        recommendations = _recommendations()
        signature = data_signature_from_recommendations(recommendations)
        base = {
            "status": "정상",
            "training_mode": "original",
            "candidate_count": 1,
            "data_signature": signature,
            "dqn_action_by_route": {"R001": "재고 이동"},
            "dqn_reference_by_route": {"R001": 80.0},
        }
        for field in ("prediction_distribution", "target_distribution"):
            result = dict(base)
            result[field] = {"보류": 9, "할인": 1}
            self.assertFalse(can_apply_dqn_to_current_data(result, signature))

    def test_missing_current_signature_blocks_application(self):
        result = {
            "status": "정상",
            "data_signature": "stored",
            "dqn_action_by_route": {"R001": "재고 이동"},
        }
        self.assertFalse(can_apply_dqn_to_current_data(result, None))

    def test_evaluate_dqn_stability_detects_past_signature(self):
        status, _ = evaluate_dqn_stability([0.1], ["재고 이동", "할인", "보류"], [0.1, 0.2, 0.3], data_signature="a", current_signature="b")
        self.assertEqual(status, "과거 결과")

    def test_batch_records_one_sample_failure_and_continues(self):
        sets = [
            {"label": "A", "sample_id": "a", "recommendations": _recommendations()},
            {"label": "B", "sample_id": "b", "recommendations": _recommendations()},
        ]
        with patch("services.dqn_service.train_dqn", side_effect=RuntimeError("boom")):
            result = train_dqn_batch(sets, episodes=1)
        self.assertEqual(result["count"], 2)
        self.assertTrue(all(item["result"]["status"] == "검토 필요" for item in result["results"]))


if __name__ == "__main__":
    unittest.main()
