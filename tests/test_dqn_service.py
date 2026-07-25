"""V2 DQN service contract tests.

No historical DQN artifact is read.  PyTorch-dependent training is not required
for these tests; the service must remain safe without it.
"""
from __future__ import annotations

import math
import unittest

from services.dqn_service import (
    ACTION_LABELS,
    UNKNOWN_ACTION,
    action_index_from_label,
    apply_dqn_reference_to_recommendations,
    apply_dqn_result_to_recommendations,
    build_action_mapping,
    build_state_vectors,
    calculate_rewards,
    can_apply_dqn_to_current_data,
    data_signature_from_recommendations,
    dqn_result_summary,
    evaluate_dqn_stability,
    get_dqn_status,
    get_torch_status,
    normalize_action,
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


class DqnServiceTests(unittest.TestCase):
    def test_action_mapping(self):
        self.assertEqual(normalize_action("direct_transfer"), "재고 이동")
        self.assertEqual(normalize_action("discount_sale"), "할인")
        self.assertEqual(normalize_action("urgent_discount"), "긴급 할인")
        self.assertEqual(normalize_action("plus_one"), "1+1")
        self.assertEqual(normalize_action("discard"), "폐기")
        self.assertEqual(normalize_action("maintain"), "보류")

    def test_action_index_is_consistent_across_the_shared_vocabulary(self):
        """The single label<->index mapping must round-trip for every action."""
        mapping = build_action_mapping()
        for index, label in enumerate(ACTION_LABELS):
            self.assertEqual(mapping[label], index)
            self.assertEqual(action_index_from_label(label), index)
            # numeric action code (int and numeric string) resolves to the same label
            self.assertEqual(normalize_action(index), label)
            self.assertEqual(normalize_action(str(index)), label)
        self.assertIsNone(action_index_from_label("존재하지 않는 행동"))

    def test_action_normalization_mixed_inputs(self):
        # Korean, English/alias, and exact-label inputs all normalize identically.
        self.assertEqual(normalize_action("이동"), "재고 이동")
        self.assertEqual(normalize_action("폐기"), "폐기")
        self.assertEqual(normalize_action("hold"), "보류")
        self.assertEqual(normalize_action("dispose"), "폐기")
        # Missing / empty are inferred from route type, not treated as unknown.
        self.assertEqual(normalize_action("", route_type="VIA_DC"), "DC 경유 이동")
        self.assertEqual(normalize_action(None, route_type="DIRECT"), "직접 이동")
        self.assertEqual(normalize_action(None), "재고 이동")
        self.assertEqual(normalize_action(float("nan")), "재고 이동")

    def test_unknown_action_is_flagged_not_fabricated(self):
        # Backward-compatible default keeps a concrete action for training targets…
        self.assertEqual(normalize_action("완전히_모르는_값"), "재고 이동")
        # …but callers can opt in to surface an explicit 확인 필요 state instead.
        self.assertEqual(normalize_action("완전히_모르는_값", allow_unknown=True), UNKNOWN_ACTION)
        self.assertEqual(normalize_action(999, allow_unknown=True), UNKNOWN_ACTION)

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

    def test_apply_mock_training_result_adds_detail_fields_without_scores(self):
        recommendations = _recommendations()
        result = {
            "status": "연결",
            "reflection_mode": "DQN 참고만",
            "dqn_action_by_route": {"R001": "재고 이동", "R002": "할인"},
            "dqn_confidence_by_route": {"R001": 91.5, "R002": 83.0},
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
            "reflection_mode": "DQN 약하게 반영",
            "dqn_action_by_route": {"R001": "재고 이동"},
            "dqn_confidence_by_route": {"R001": 80.0},
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
            "data_signature": signature,
            "dqn_action_by_route": {"R001": "재고 이동"},
            "dqn_confidence_by_route": {"R001": 90.0},
            "dqn_reference_by_route": {"R001": 88.0},
        }
        self.assertTrue(can_apply_dqn_to_current_data(result, signature))
        updated = apply_dqn_reference_to_recommendations(recommendations, result, signature)
        self.assertEqual(updated[0]["dqn_status"], "정상")
        self.assertEqual(updated[0]["dqn_reference_score"], 88.0)

    def test_evaluate_dqn_stability_detects_past_signature(self):
        status, _ = evaluate_dqn_stability([0.1], ["재고 이동", "할인", "보류"], [0.1, 0.2, 0.3], data_signature="a", current_signature="b")
        self.assertEqual(status, "과거 결과")


if __name__ == "__main__":
    unittest.main()
