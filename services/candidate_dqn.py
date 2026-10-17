"""Small candidate DQN over the shared finite-allocation problem.

Never loads or modifies production DQN/latest. Hard feasibility is independent
of Q values. Learned policy quality is empirical, never an optimality claim.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn

from services.candidate_selection import SelectionProblem, evaluate, validate, _service
from services.operating_snapshot import signature

VERSION = "masked-selection-dqn-0.1"
FEATURES = 12


def decision_fingerprint(p: SelectionProblem) -> str:
    """Ignore run labels/seeds: relabelling a training instance is not holdout data."""
    return signature({"capacities": p.capacities, "needs": p.needs, "max_routes": p.max_routes,
                      "service_metric": p.service_metric,
                      "candidates": sorted([{"source": c.source, "target": c.target, "product": c.product,
                          "state": c.state, "lot": c.lot, "unit": c.unit, "resources": c.resources,
                          "need_key": c.need_key, "options": [(o.quantity, o.cost) for o in c.options],
                          "vhs": c.vhs} for c in p.candidates], key=lambda x: signature(x))})


def reward_scale(p: SelectionProblem) -> dict:
    """Finite service lattice gives a proven bound, not an arbitrary weight.

    Every possible plan's service difference is a multiple of the lattice step.
    Total normalized cost penalty is < half a step, so service dominates cost.
    This proof applies to exact rewards, not approximate neural predictions.
    """
    fractions = []
    for c in p.candidates:
        for o in c.options:
            q = Fraction(str(o.quantity))
            fractions.append(q if p.service_metric == "LEGACY_RAW_QTY" else
                             q / Fraction(str(p.needs[c.need_key])) / max(1, sum(v > 0 for v in p.needs.values())))
    if not fractions:
        return {"service_quantum": 1.0, "cost_bound": 0.0, "cost_weight": 0.5}
    denominator = reduce(lcm, (x.denominator for x in fractions), 1)
    numerator = reduce(gcd, (x.numerator * (denominator // x.denominator) for x in fractions))
    quantum = float(Fraction(numerator, denominator))
    if quantum < 1e-8:
        raise ValueError("service lattice too fine for reliable candidate training reward")
    bound = p.max_routes * max(o.cost for c in p.candidates for o in c.options)
    return {"service_quantum": quantum, "cost_bound": bound, "cost_weight": quantum / (2 * (bound + 1))}


class SelectionEnvironment:
    def __init__(self, problem: SelectionProblem):
        validate(problem)
        self.problem = problem
        self.scale = reward_scale(problem)
        self.plan = {}
        self.done = False
        self.actions = [(c, o) for c in sorted(problem.candidates, key=lambda c: c.candidate_id)
                        for o in sorted(c.options, key=lambda o: o.quantity)] + [(None, None)]

    def valid_actions(self) -> list[int]:
        if self.done:
            return []
        valid = []
        for i, (c, o) in enumerate(self.actions):
            if c is None:
                valid.append(i)
            elif c.candidate_id not in self.plan and not evaluate(self.problem, {**self.plan, c.candidate_id: o.quantity})["violations"]:
                valid.append(i)
        return valid

    def features(self, action: int) -> np.ndarray:
        c, o = self.actions[action]
        ev = evaluate(self.problem, self.plan)
        remaining_routes = max(0, self.problem.max_routes - len(self.plan)) / max(1, self.problem.max_routes)
        if c is None:
            return np.array([1, 0, 0, 0, 0, 0, 0, remaining_routes, ev["service"], 0, 0, 1], np.float32)
        ratios = [ev["remaining"][k] / max(self.problem.capacities[k], 1e-9) for k in c.resources]
        need = self.problem.needs[c.need_key]
        return np.array([0, _service(self.problem, c, o), o.cost / max(1, self.scale["cost_bound"]),
                         o.quantity / need, min(ratios), np.mean(ratios), len(c.resources) / 10,
                         remaining_routes, ev["service"], ev["remaining"][c.need_key] / need,
                         self.scale["service_quantum"], 1], np.float32)

    def step(self, action: int) -> tuple[float, bool]:
        if action not in self.valid_actions():
            raise ValueError("invalid action: feasibility mask rejects it")
        c, o = self.actions[action]
        if c is None:
            self.done = True
            return 0.0, True
        self.plan[c.candidate_id] = o.quantity
        reward = _service(self.problem, c, o) - self.scale["cost_weight"] * o.cost
        self.done = len(self.plan) >= self.problem.max_routes
        return reward, self.done


def _network():
    return nn.Sequential(nn.Linear(FEATURES, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))


@dataclass
class TrainedCandidate:
    network: nn.Module
    metadata: dict


def train(problems: list[SelectionProblem], seed=17, episodes=250, learning_rate=0.002) -> TrainedCandidate:
    if not problems or episodes <= 0:
        raise ValueError("training problems and positive episodes required")
    for p in problems:
        validate(p)
    rng = random.Random(seed)
    started = time.perf_counter()
    history, replay, diagnostics = [], [], []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network, target = _network(), _network()
            target.load_state_dict(network.state_dict())
            optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
            for episode in range(episodes):
                env = SelectionEnvironment(problems[rng.randrange(len(problems))])
                total = 0.0
                epsilon = max(0.05, 1 - episode / max(1, episodes * 0.85))
                episode_losses, mask_counts, chosen_actions = [], [], []
                while not env.done:
                    actions = env.valid_actions()
                    mask_counts.append(len(actions))
                    features = np.array([env.features(a) for a in actions])
                    with torch.no_grad():
                        qs = network(torch.from_numpy(features)).flatten().numpy()
                    idx = rng.randrange(len(actions)) if rng.random() < epsilon else int(np.argmax(qs))
                    chosen_actions.append(actions[idx])
                    feature = features[idx].copy()
                    reward, done = env.step(actions[idx])
                    next_features = np.array([env.features(a) for a in env.valid_actions()], dtype=np.float32).reshape(-1, FEATURES)
                    replay.append((feature, reward, next_features, done))
                    replay = replay[-5000:]
                    total += reward
                    if len(replay) >= 16:
                        batch = rng.sample(replay, min(32, len(replay)))
                        x = torch.from_numpy(np.array([b[0] for b in batch]))
                        with torch.no_grad():
                            y = torch.tensor([r + (float(target(torch.from_numpy(n)).max()) if len(n) and not d else 0)
                                              for _, r, n, d in batch], dtype=torch.float32)
                        loss = nn.functional.smooth_l1_loss(network(x).flatten(), y)
                        episode_losses.append(float(loss.detach()))
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(), 2.0)
                        optimizer.step()
                history.append(total)
                diagnostics.append({"episode": episode + 1, "return": total,
                    "mean_td_loss": float(np.mean(episode_losses)) if episode_losses else None,
                    "gradient_updates": len(episode_losses), "valid_action_counts": mask_counts,
                    "actions": chosen_actions, "invalid_action_attempts": 0})
                if episode % 20 == 0:
                    target.load_state_dict(network.state_dict())
            network.eval()
    finally:
        torch.set_num_threads(previous_threads)
    return TrainedCandidate(network, {"version": VERSION, "seed": seed, "episodes": episodes,
             "learning_rate": learning_rate, "training_problem_signatures": [p.base_problem_signature for p in problems],
             "training_candidate_signatures": [p.candidate_set_signature for p in problems],
             "training_decision_fingerprints": [decision_fingerprint(p) for p in problems],
             "reward_history": history, "gamma": 1.0, "device": "cpu", "seconds": time.perf_counter() - started,
             "training_diagnostics": diagnostics,
             "production_approved": False, "legacy_latest_accessed": False})


def infer(model: TrainedCandidate, problem: SelectionProblem) -> dict:
    env = SelectionEnvironment(problem)
    trace = []
    started = time.perf_counter()
    with torch.no_grad():
        while not env.done:
            actions = env.valid_actions()
            features = torch.from_numpy(np.array([env.features(a) for a in actions]))
            q = model.network(features).flatten().numpy()
            selected = actions[int(np.argmax(q))]
            before_plan = dict(env.plan)
            selected_feature = env.features(selected).tolist()
            reward, done = env.step(selected)
            trace.append({"action": selected, "reward": reward, "valid_action_count": len(actions),
                          "valid_actions": actions, "state_plan": before_plan,
                          "selected_features": selected_feature, "invalid_action_attempts": 0})
    return {"plan": env.plan, "evaluation": evaluate(problem, env.plan), "trace": trace,
            "seconds": time.perf_counter() - started, "seed": model.metadata["seed"],
            "base_problem_signature": problem.base_problem_signature, "candidate_set_signature": problem.candidate_set_signature,
            "training_overlap": decision_fingerprint(problem) in model.metadata["training_decision_fingerprints"],
            "optimality_proven": False, "production_approved": False}


def save_candidate(model: TrainedCandidate, path: Path) -> None:
    path = Path(path)
    if path.exists() or any("latest" in part.lower() or part.lower() == "dqn" for part in path.parts):
        raise ValueError("new explicitly named candidate artifact required; latest/overwrite forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a race from overwriting an existing candidate.
    with path.open("xb") as handle:
        torch.save({"state_dict": model.network.state_dict(), "metadata": model.metadata}, handle)
