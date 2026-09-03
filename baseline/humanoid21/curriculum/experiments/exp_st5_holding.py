"""ST-5: Delta + holding bonus (γ_s = 1.05).

Reward: r_t = 1.05·φ(t) - φ(t-1), where φ = uprightness * (height / 1.28).
r_0 = 1.05·φ(0) - initial_phi.
Episode terminates on fall (imbalance).

The holding bonus (γ_s - 1)·φ(t) provides a continuous per-step
incentive to stay in high-φ states. Tests whether holding helps
on unstable targets (standing balance) despite SNR dilution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.rollout import extract_per_step_field
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


class ST5HoldingConfig(CombatExperimentBase):

    name = "st5_holding"
    reward_keys = ("r_fall",)
    gammas = {"r_fall": 0.99}

    BLUEPRINT = "basic_balance_phi_env.yaml"
    shaping_gamma: float = 1.05

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
        phi = extract_per_step_field(episode.observer_outputs, "height_phi", "phi", T)
        if phi is None:
            phi = np.zeros(T, dtype=np.float32)
        initial_phi_arr = extract_per_step_field(episode.observer_outputs, "height_phi", "initial_phi", T)
        initial_phi = float(initial_phi_arr[0]) if initial_phi_arr is not None else 0.0

        gs = self.shaping_gamma
        r_fall = np.zeros(T, dtype=np.float32)
        r_fall[0] = gs * phi[0] - initial_phi
        if T > 1:
            r_fall[1:] = gs * phi[1:] - phi[:-1]
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


EXPERIMENT = ST5HoldingConfig()
