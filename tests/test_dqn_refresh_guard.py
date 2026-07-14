"""Guard against comparison-only DQN results changing VHS scores."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from pages import validation
from services.dqn_service import data_signature_from_recommendations


class DqnRefreshGuardTests(unittest.TestCase):
    def test_review_result_updates_status_without_recalculating_vhs(self):
        recommendations = [
            {"route_id": "R1", "vhs_score": 81.5, "vhs_rank": 1, "expected_saving": 10000},
            {"route_id": "R2", "vhs_score": 72.0, "vhs_rank": 2, "expected_saving": 8000},
            {"route_id": "R3", "vhs_score": 63.0, "vhs_rank": 3, "expected_saving": 6000},
        ]
        signature = data_signature_from_recommendations(recommendations)
        state = {
            "varo_recommendations": [dict(row) for row in recommendations],
            "data_signature": signature,
            "analysis_result": {"summary": {"average_vhs_score": 72.17}},
            "varo_pipeline_result": {"summary": {"average_vhs_score": 72.17}},
        }
        result = {
            "status": "검토 필요",
            "data_signature": signature,
            "dqn_action_by_route": {"R1": "재고 이동"},
            "dqn_confidence_by_route": {"R1": 99.0},
        }
        with patch.object(validation.st, "session_state", state):
            validation._refresh_recommendations_with_dqn(result)

        self.assertEqual(
            [row["vhs_score"] for row in state["varo_recommendations"]],
            [81.5, 72.0, 63.0],
        )
        self.assertEqual(state["varo_recommendations"][0]["dqn_status"], "검토 필요")
        self.assertEqual(state["analysis_result"]["summary"]["average_vhs_score"], 72.17)

    def test_review_after_normal_result_restores_pre_dqn_baseline(self):
        recommendations = [
            {"route_id": "R1", "vhs_score": 81.5, "vhs_rank": 1, "expected_saving": 10000, "varo_action": "재고 이동"},
            {"route_id": "R2", "vhs_score": 72.0, "vhs_rank": 2, "expected_saving": 8000, "varo_action": "할인"},
            {"route_id": "R3", "vhs_score": 63.0, "vhs_rank": 3, "expected_saving": 6000, "varo_action": "재고 이동"},
        ]
        signature = data_signature_from_recommendations(recommendations)
        pipeline = {
            "summary": {"average_vhs_score": 72.17},
            "vhs_analysis": {"weights": {"dqn_reference_score": 0.0}},
            "connected_algorithms": ["services.vhs_score_engine.apply_auto_vhs"],
        }
        state = {
            "varo_recommendations": [dict(row) for row in recommendations],
            "data_signature": signature,
            "analysis_result": dict(pipeline),
            "varo_pipeline_result": dict(pipeline),
        }
        normal = {
            "status": "정상",
            "training_mode": "balanced",
            "candidate_count": 3,
            "data_signature": signature,
            "dqn_action_by_route": {"R1": "재고 이동", "R2": "할인", "R3": "할인"},
            "dqn_confidence_by_route": {"R1": 90.0, "R2": 85.0, "R3": 80.0},
            "dqn_reference_by_route": {"R1": 88.0, "R2": 82.0, "R3": 75.0},
            "prediction_distribution": {"재고 이동": 1, "할인": 2},
            "target_distribution": {"재고 이동": 2, "할인": 1},
        }
        review = {"status": "검토 필요", "data_signature": signature}
        with patch.object(validation.st, "session_state", state):
            validation._refresh_recommendations_with_dqn(normal)
            self.assertGreater(state["analysis_result"]["vhs_analysis"]["weights"]["dqn_reference_score"], 0)
            validation._refresh_recommendations_with_dqn(review)

        self.assertEqual(
            [row["vhs_score"] for row in state["varo_recommendations"]],
            [81.5, 72.0, 63.0],
        )
        self.assertEqual(state["analysis_result"]["vhs_analysis"]["weights"]["dqn_reference_score"], 0.0)
        self.assertNotIn(
            "services.dqn_service.apply_dqn_reference_to_recommendations",
            state["connected_algorithms"],
        )


if __name__ == "__main__":
    unittest.main()
