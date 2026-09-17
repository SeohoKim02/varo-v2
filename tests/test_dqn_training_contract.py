"""Proof that the DQN in Varo V2 really trains, saves, loads, and infers.

Production DQN code is never mocked here.  Only the local artifact directory is
redirected to a temporary folder and the episode count is kept small.
"""
from __future__ import annotations

import contextlib
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import dqn_service
from services.dqn_service import (
    ACTION_CONCENTRATION_LIMIT,
    ACTION_LABELS,
    DQN_STATE_COLUMNS,
    FEATURE_COLUMNS,
    action_concentration,
    action_reward_matrix,
    build_training_states,
    chronological_split,
    dqn_inference_view,
    infer_dqn_actions,
    model_payload_is_compatible,
    read_training_comparison_rows,
    reward_optimal_actions,
    state_schema_fingerprint,
    train_dqn,
)

TORCH_AVAILABLE = dqn_service.get_torch_status()[0]


def _candidates(count: int = 14, seed: int = 11, action: str = "재고 이동") -> list[dict]:
    """Synthetic but structurally real V2 candidates, all with one heuristic label."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        rows.append({
            "route_id": f"R{index:03d}",
            "product_id": f"P{index % 4:03d}",
            "source_id": f"S{index % 3:03d}",
            "target_id": f"S{(index + 1) % 3:03d}",
            "route_type": "DIRECT" if index % 3 else "VIA_DC",
            "recommended_qty": rng.randint(2, 40),
            "expected_saving": rng.randint(1000, 50000),
            "savings_score": rng.uniform(0, 100),
            "move_cost": rng.randint(1000, 30000),
            "estimated_cost": rng.randint(1000, 30000),
            "distance_km": rng.uniform(1, 40),
            "expected_time_min": rng.uniform(5, 90),
            "travel_time_min": rng.uniform(5, 90),
            "disposal_risk_score": rng.uniform(0, 100),
            "days_to_expiry": rng.randint(1, 30),
            "expiry_days": rng.randint(1, 30),
            "demand_fit_score": rng.uniform(0, 100),
            "inventory_balance_score": rng.uniform(0, 100),
            "route_cost_score": rng.uniform(0, 100),
            "feasibility_score": rng.uniform(20, 100),
            "promotion_score": rng.uniform(0, 100),
            "vhs_score": rng.uniform(40, 95),
            "greedy_rank": index + 1,
            "confidence_score": rng.uniform(50, 100),
            "varo_action": action,
            "greedy_action": action,
            "greedy_strategy": action,
        })
    return rows


@contextlib.contextmanager
def _artifact_directory():
    """Redirect every DQN artifact path to a temporary folder."""
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        with patch.multiple(
            dqn_service,
            OUTPUT_DIR=directory,
            LATEST_JSON=directory / "latest_dqn_result.json",
            LATEST_MODEL=directory / "latest_dqn_model.pt",
            LATEST_BATCH_JSON=directory / "latest_dqn_batch.json",
            LATEST_COMPARISON_JSON=directory / "latest_dqn_comparison.json",
            TRAINING_COMPARISON_CSV=directory / "dqn_training_comparison.csv",
            LATEST_RESULT_BY_VARIANT={
                "original": directory / "latest_dqn_result_original.json",
                "balanced": directory / "latest_dqn_result_balanced.json",
            },
            LATEST_MODEL_BY_VARIANT={
                "original": directory / "latest_dqn_model_original.pt",
                "balanced": directory / "latest_dqn_model_balanced.pt",
            },
        ):
            directory.mkdir(parents=True, exist_ok=True)
            yield directory


class DqnRewardAndStateContractTests(unittest.TestCase):
    """Checks that need no PyTorch runtime."""

    def test_state_excludes_other_strategy_outputs(self):
        self.assertNotIn("vhs_score", DQN_STATE_COLUMNS)
        self.assertNotIn("greedy_rank", DQN_STATE_COLUMNS)
        for column in DQN_STATE_COLUMNS:
            self.assertIn(column, FEATURE_COLUMNS)

    def test_state_vectors_ignore_vhs_and_greedy_values(self):
        base = _candidates()
        changed = [dict(row, vhs_score=1.0, greedy_rank=999) for row in base]
        self.assertEqual(build_training_states(base), build_training_states(changed))

    def test_state_vectors_stay_bounded_with_missing_values(self):
        vectors = build_training_states([{"route_id": "R001"}, {"route_id": "R002"}])
        for vector in vectors:
            self.assertTrue(all(0.0 <= value <= 1.0 for value in vector))
            self.assertTrue(all(math.isfinite(value) for value in vector))

    def test_action_rewards_cover_every_action_and_stay_bounded(self):
        matrix = action_reward_matrix(_candidates())
        self.assertEqual(len(matrix), 14)
        for row in matrix:
            self.assertEqual(len(row), len(ACTION_LABELS))
            self.assertTrue(all(0.0 <= value <= 1.0 for value in row))

    def test_rewards_ignore_vhs_and_greedy_values(self):
        base = _candidates()
        changed = [dict(row, vhs_score=1.0, greedy_rank=999, varo_action="폐기") for row in base]
        self.assertEqual(action_reward_matrix(base), action_reward_matrix(changed))

    def test_infeasible_transfer_is_penalised_against_holding(self):
        rows = [
            dict(_candidates(3)[0], feasibility_score=5, route_type="DIRECT"),
            dict(_candidates(3)[1], feasibility_score=95, route_type="DIRECT"),
            dict(_candidates(3)[2], feasibility_score=50, route_type="DIRECT"),
        ]
        matrix = action_reward_matrix(rows)
        transfer = ACTION_LABELS.index("재고 이동")
        hold = ACTION_LABELS.index("보류")
        self.assertLess(matrix[0][transfer], matrix[0][hold])

    def test_chronological_split_keeps_time_order(self):
        rows = [dict(row, snapshot_date=f"2026-09-{index + 1:02d}") for index, row in enumerate(_candidates(12))]
        shuffled = list(reversed(rows))
        train, holdout = chronological_split(shuffled)
        self.assertTrue(train and holdout)
        self.assertEqual(len(train) + len(holdout), len(rows))
        latest_train = max(str(row["snapshot_date"]) for row in train)
        earliest_holdout = min(str(row["snapshot_date"]) for row in holdout)
        self.assertLess(latest_train, earliest_holdout)

    def test_small_candidate_sets_keep_every_row_for_training(self):
        train, holdout = chronological_split(_candidates(4))
        self.assertEqual(len(train), 4)
        self.assertEqual(holdout, [])

    def test_action_concentration_flags_a_dominant_action(self):
        skewed = action_concentration(["재고 이동"] * 19 + ["할인"])
        self.assertTrue(skewed["concentrated"])
        self.assertGreaterEqual(skewed["dominant_ratio"], ACTION_CONCENTRATION_LIMIT)
        mixed = action_concentration(["재고 이동"] * 5 + ["할인"] * 5)
        self.assertFalse(mixed["concentrated"])
        self.assertEqual(mixed["distinct_actions"], 2)

    def test_reward_optimal_reference_is_not_the_heuristic_label(self):
        rows = _candidates()
        reference = reward_optimal_actions(rows)
        self.assertEqual(len(reference), len(rows))
        self.assertGreater(len(set(reference)), 1)


@unittest.skipUnless(TORCH_AVAILABLE, "DQN runtime unavailable")
class DqnRealTrainingTests(unittest.TestCase):
    def test_short_training_runs_and_changes_parameters(self):
        with _artifact_directory():
            result = train_dqn(_candidates(), episodes=12, learning_rate=0.002)
        diagnostics = result.diagnostics
        self.assertEqual(result.episodes, 12)
        self.assertTrue(diagnostics["parameters_changed"])
        self.assertGreater(diagnostics["optimizer_steps"], 0)
        self.assertEqual(len(result.loss_history), 12)
        self.assertEqual(len(result.reward_history), 12)
        self.assertTrue(any(value is not None for value in result.loss_history))

    def test_training_uses_replay_target_network_and_exploration(self):
        with _artifact_directory():
            result = train_dqn(_candidates(), episodes=12, learning_rate=0.002)
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics["replay_capacity"], dqn_service.REPLAY_CAPACITY)
        self.assertEqual(diagnostics["replay_batch_size"], dqn_service.REPLAY_BATCH_SIZE)
        self.assertEqual(diagnostics["target_sync_episodes"], dqn_service.TARGET_SYNC_EPISODES)
        self.assertEqual(diagnostics["epsilon_range"], [dqn_service.EPSILON_START, dqn_service.EPSILON_END])
        self.assertEqual(diagnostics["discount_factor"], dqn_service.DISCOUNT_FACTOR)
        self.assertGreater(len(diagnostics["exploration_distribution"]), 1)

    def test_model_file_is_written_and_reloadable(self):
        with _artifact_directory() as directory:
            result = train_dqn(_candidates(), episodes=20, learning_rate=0.002)
            self.assertEqual(result.status, dqn_service.NORMAL_STATUS)
            self.assertTrue(result.model_path)
            model_file = Path(result.model_path)
            self.assertTrue(model_file.exists())
            self.assertEqual(model_file.suffix, ".pt")
            self.assertTrue(list(directory.glob("dqn_model_*.pt")))

            import torch

            payload = torch.load(model_file, map_location="cpu", weights_only=False)
            self.assertEqual(payload["feature_columns"], list(DQN_STATE_COLUMNS))
            self.assertEqual(payload["state_schema"], state_schema_fingerprint())
            compatible, _ = model_payload_is_compatible(payload, result.data_signature, "original")
            self.assertTrue(compatible)

    def test_saved_model_infers_the_same_actions(self):
        rows = _candidates()
        with _artifact_directory():
            trained = train_dqn(rows, episodes=20, learning_rate=0.002)
            loaded = infer_dqn_actions(rows, trained.data_signature, model_path=trained.model_path)
        self.assertEqual(loaded.model_status, "loaded")
        self.assertEqual(loaded.dqn_action_by_route, trained.dqn_action_by_route)
        self.assertEqual(loaded.candidate_count, len(rows))

    def test_actions_come_from_a_real_model_forward_pass(self):
        rows = _candidates()
        with _artifact_directory():
            trained = train_dqn(rows, episodes=20, learning_rate=0.002)
            inferred = infer_dqn_actions(rows, trained.data_signature, model_path=trained.model_path)

            import torch

            payload = torch.load(trained.model_path, map_location="cpu", weights_only=False)
            model = dqn_service._model(payload["input_size"], len(ACTION_LABELS), seed=payload["seed"])
            model.load_state_dict(payload["state_dict"])
            model.eval()
            base = build_training_states(rows)
            expected, _ = dqn_service._greedy_rollout(model, base, dqn_service._cost_shares(rows))
        expected_actions = [ACTION_LABELS[index] for index in expected]
        self.assertEqual(list(inferred.dqn_action_by_route.values()), expected_actions)

    def test_different_weights_produce_different_actions(self):
        rows = _candidates()
        with _artifact_directory() as directory:
            trained = train_dqn(rows, episodes=20, learning_rate=0.002)
            baseline = infer_dqn_actions(rows, trained.data_signature, model_path=trained.model_path)

            import torch

            payload = dict(torch.load(trained.model_path, map_location="cpu", weights_only=False))
            state_dict = {key: value.clone() for key, value in payload["state_dict"].items()}
            last_bias = [key for key in state_dict if key.endswith("bias")][-1]
            bumped = torch.zeros_like(state_dict[last_bias])
            bumped[ACTION_LABELS.index("폐기")] = 500.0
            state_dict[last_bias] = state_dict[last_bias] + bumped
            payload["state_dict"] = state_dict
            altered = directory / "altered_model.pt"
            torch.save(payload, altered)
            changed = infer_dqn_actions(rows, trained.data_signature, model_path=str(altered))
        self.assertNotEqual(changed.dqn_action_by_route, baseline.dqn_action_by_route)
        self.assertEqual(set(changed.dqn_action_by_route.values()), {"폐기"})

    def test_missing_model_reports_training_needed_without_borrowing_values(self):
        rows = _candidates()
        with _artifact_directory():
            view = dqn_inference_view(rows, "signature-without-model")
        self.assertFalse(view["available"])
        self.assertEqual(view["status"], dqn_service.NEEDS_TRAINING_STATUS)
        for item in view["recommendations"]:
            self.assertEqual(item["dqn_action"], "비교 불가")
            self.assertIsNone(item["dqn_confidence"])

    def test_incompatible_feature_schema_is_refused(self):
        rows = _candidates()
        with _artifact_directory() as directory:
            trained = train_dqn(rows, episodes=20, learning_rate=0.002)

            import torch

            payload = dict(torch.load(trained.model_path, map_location="cpu", weights_only=False))
            payload["state_schema"] = "legacy-schema"
            payload["feature_columns"] = list(FEATURE_COLUMNS)
            legacy = directory / "legacy_model.pt"
            torch.save(payload, legacy)
            compatible, message = model_payload_is_compatible(payload, trained.data_signature, "original")
            refused = infer_dqn_actions(rows, trained.data_signature, model_path=str(legacy))
        self.assertFalse(compatible)
        self.assertIn("schema", message)
        self.assertNotEqual(refused.model_status, "loaded")
        self.assertIn(refused.status, {dqn_service.NEEDS_TRAINING_STATUS, dqn_service.PAST_RESULT_STATUS})

    def test_dqn_does_not_copy_greedy_or_vhs_decisions(self):
        rows = _candidates(action="재고 이동")
        with _artifact_directory():
            result = train_dqn(rows, episodes=20, learning_rate=0.002)
        heuristic = result.diagnostics["heuristic_action_distribution"]
        self.assertEqual(heuristic, {"재고 이동": len(rows)})
        self.assertGreater(len(result.prediction_distribution), 1)
        self.assertNotEqual(result.prediction_distribution, heuristic)

    def test_changing_vhs_and_greedy_values_does_not_change_the_policy(self):
        rows = _candidates()
        rewritten = [dict(row, vhs_score=10.0, greedy_rank=500, varo_action="폐기") for row in rows]
        with _artifact_directory():
            first = train_dqn(rows, episodes=16, learning_rate=0.002)
        with _artifact_directory():
            second = train_dqn(rewritten, episodes=16, learning_rate=0.002)
        self.assertEqual(
            list(first.dqn_action_by_route.values()),
            list(second.dqn_action_by_route.values()),
        )

    def test_action_concentration_is_measured_from_real_inference(self):
        rows = _candidates()
        with _artifact_directory():
            result = train_dqn(rows, episodes=20, learning_rate=0.002)
        concentration = result.diagnostics["action_concentration"]
        self.assertEqual(concentration["total"], len(rows))
        self.assertEqual(
            concentration["distinct_actions"],
            len([value for value in result.prediction_distribution.values() if value > 0]),
        )
        self.assertEqual(
            concentration["dominant_action"],
            max(result.prediction_distribution, key=result.prediction_distribution.get),
        )

    def test_a_collapsed_policy_is_marked_for_review(self):
        rows = _candidates()
        def collapsed(model, base_states, cost_shares):
            size = len(base_states)
            return [0] * size, [[1.0, *([0.0] * (len(ACTION_LABELS) - 1))] for _ in range(size)]

        with _artifact_directory():
            with patch.object(dqn_service, "_greedy_rollout", side_effect=collapsed):
                result = train_dqn(rows, episodes=12, learning_rate=0.002)
        self.assertEqual(result.status, dqn_service.NEEDS_REVIEW_STATUS)
        self.assertIsNone(result.model_path)

    def test_training_never_learns_from_holdout_candidates(self):
        rows = [
            dict(row, snapshot_date=f"2026-09-{index + 1:02d}")
            for index, row in enumerate(_candidates(20))
        ]
        with _artifact_directory():
            result = train_dqn(rows, episodes=12, learning_rate=0.002)
        diagnostics = result.diagnostics
        train_count = diagnostics["train_candidate_count"]
        holdout_count = diagnostics["holdout_candidate_count"]
        self.assertEqual(diagnostics["time_order_column"], "snapshot_date")
        self.assertGreater(holdout_count, 0)
        self.assertEqual(train_count + holdout_count, len(rows))
        train, holdout = chronological_split(rows)
        self.assertEqual(len(train), train_count)
        self.assertTrue(
            max(str(row["snapshot_date"]) for row in train)
            < min(str(row["snapshot_date"]) for row in holdout)
        )
        self.assertIsNotNone(diagnostics["holdout_mean_reward"])

    def test_cumulative_comparison_file_records_every_run(self):
        with _artifact_directory() as directory:
            train_dqn(_candidates(), episodes=12, learning_rate=0.002)
            train_dqn(_candidates(seed=29), episodes=12, learning_rate=0.003)
            path = directory / "dqn_training_comparison.csv"
            rows = read_training_comparison_rows(path)
            self.assertTrue(path.exists())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["state_schema"], state_schema_fingerprint())
        self.assertEqual(rows[1]["learning_rate"], "0.003")
        for row in rows:
            self.assertTrue(row["timestamp"])
            self.assertTrue(row["dominant_action"])

    def test_model_filename_keeps_training_context(self):
        with _artifact_directory():
            result = train_dqn(
                _candidates(), episodes=20, learning_rate=0.001, candidate_count=12,
                sample_id="sample_03", training_mode="balanced", store_count=10, dc_count=2,
            )
        name = Path(result.model_path).name
        self.assertIn("sample_03", name)
        self.assertIn("balanced", name)
        self.assertIn("10stores", name)
        self.assertIn("2dc", name)
        self.assertIn("ep20", name)
        self.assertIn("lr0p001", name)


if __name__ == "__main__":
    unittest.main()
