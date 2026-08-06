"""ST-1: Pure terminal penalty.

Reward: 0 for every alive step, -1 on fall (imbalance termination).
No per-step survival bonus, no shaping.

Tests whether self-correcting tasks (standing balance) can learn
purely from the environment's natural failure signal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


class ST1TerminalConfig(CombatExperimentBase):

    name = "st1_terminal"
    reward_keys = ("r_fall",)
    gammas = {"r_fall": 0.99}

    BLUEPRINT = "basic_balance_phi_env.yaml"

    _survival_rate: float = 0.0

    def video_env_blueprint(self):
        return self._make_video_blueprint(self._env_pb())

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def build_rollout_jobs(self, policy_bp, base_seed):
        return self._build_selfplay_jobs(self._env_pb(), policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp, base_seed):
        return self._build_selfplay_jobs(self._env_pb(), policy_bp, base_seed, self.eval_episodes)

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("survived", 0.0) > best_esum.get("survived", 0.0)

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0,)

    def next_weights(self, eval_metrics, current_weights) -> Tuple[float, ...]:
        self._survival_rate = float(eval_metrics.get("survived", 0.0))
        return (1.0,)

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        fell = all(r.startswith("imbalance") for r in episode.agent_termination_reason.values())
        r_fall = np.zeros(T, dtype=np.float32)
        if fell:
            r_fall[-1] = -1.0
        return {"r_fall": r_fall}

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        fell = all(r.startswith("imbalance") for r in episode.agent_termination_reason.values())
        return {"survived": 0.0 if fell else 1.0}

    def scheduler_info(self) -> Dict[str, Any]:
        return {"survival_rate": round(self._survival_rate, 3)}

    def scheduler_state(self) -> dict:
        return {"survival_rate": self._survival_rate}

    def load_scheduler_state(self, state: dict) -> None:
        self._survival_rate = float(state.get("survival_rate", 0.0))


EXPERIMENT = ST1TerminalConfig()
