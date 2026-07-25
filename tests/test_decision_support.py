"""Stability status and DQN-independent confidence status tests."""
from __future__ import annotations

import unittest

from services.decision_support import (
    CONF_NONE,
    NOT_COMPUTABLE,
    STABLE,
    UNSTABLE,
    recommendation_confidence,
    recommendation_stability,
)


class StabilityTests(unittest.TestCase):
    def test_stable_when_top1_always_retained(self):
        ws = {"scenarios": 6, "top1_retention_rate": 1.0, "rank_volatility": 0.2, "fragile_components": []}
        self.assertEqual(recommendation_stability(ws)["status"], STABLE)

    def test_unstable_when_top1_flips_often(self):
        ws = {"scenarios": 6, "top1_retention_rate": 0.4, "rank_volatility": 1.5, "fragile_components": ["절감액"]}
        self.assertEqual(recommendation_stability(ws)["status"], UNSTABLE)

    def test_not_computable_without_scenarios(self):
        self.assertEqual(recommendation_stability({"scenarios": 0})["status"], NOT_COMPUTABLE)
        self.assertEqual(recommendation_stability(None)["status"], NOT_COMPUTABLE)


def _recs(feasible=True, scores=(90, 60, 40)):
    rows = []
    for i, score in enumerate(scores):
        rows.append({
            "route_id": f"R{i}", "recommended_qty": 10, "expected_saving": 1000,
            "estimated_cost": 100, "route_type": "DIRECT", "vhs_score": score,
            "vhs_rank": i + 1, "feasibility_status": "추천 가능" if feasible else "데이터 확인 필요",
            "strategy_match": True, "pareto_status": "비지배" if i == 0 else "지배됨",
        })
    return rows


class ConfidenceTests(unittest.TestCase):
    def test_confidence_computable_without_dqn(self):
        result = recommendation_confidence(_recs(), pipeline={}, stability={"status": STABLE})
        self.assertIn(result["status"], ("높음", "보통", "낮음"))
        self.assertIsInstance(result["score"], float)
        # DQN factor absent -> still a real score, never 계산 불가 on good data
        self.assertNotEqual(result["status"], CONF_NONE)

    def test_missing_dqn_does_not_lower_confidence(self):
        recs = _recs()
        without = recommendation_confidence(recs, pipeline={}, stability={"status": STABLE})
        with_normal = recommendation_confidence(
            recs,
            pipeline={"dqn_training_result": {"status": "정상"}},
            stability={"status": STABLE},
        )
        # normal DQN may only raise (or equal); it must never be lower than no-DQN.
        self.assertGreaterEqual(with_normal["score"], without["score"])

    def test_no_recommendations_is_not_computable(self):
        result = recommendation_confidence([], pipeline={})
        self.assertEqual(result["status"], CONF_NONE)
        self.assertIsNone(result["score"])

    def test_infeasible_candidates_lower_confidence_reason(self):
        result = recommendation_confidence(_recs(feasible=False), pipeline={}, stability={"status": STABLE})
        self.assertTrue(any("데이터 확인" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
