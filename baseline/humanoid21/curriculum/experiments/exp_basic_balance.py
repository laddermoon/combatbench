
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.ppo_trainer import _extract_per_step_scalar
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


class BasicBalanceConfig(ExperimentConfig):

    name = "basic_balance"
    reward_keys = ("r_fall", "r_cross")
    gammas = {"r_fall": 0.99, "r_cross": 0.99}

    # Stage blueprints
    BLUEPRINT = "basic_balance_env.yaml"  # Basic: fall detection only

    # Stateful scheduler
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
        return (3.0, 1.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Stage transition: 1 -> 2 when eval survival rate reaches 100%.

        Both stages use identical weights (1.0,); only the blueprint differs
        (stage 2 adds InitialStatePerturbationPlugin). The active blueprint is
        resolved by current_env_blueprint() from self._stage, so simply
        advancing the stage switches the env on the next rollout.
        """
        # survived = fraction of eval episodes that did NOT terminate (= did not
        # fall) within the horizon, aggregated by compute_episode_metrics.
        survival_rate = float(eval_metrics.get("survived", 0.0))
        self._survival_rate = survival_rate

        # Weights always (1.0, 1.0) regardless of stage.
        return (3.0, 1.0)

    # Small per-step survival bonus (each alive step is worth this much).
    per_step_survival_reward: float = 0.01

    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """r_fall: per-step survival bonus + terminal signal.
        r_cross: cross-support balance reward from CrossSupportBalanceRewarder.
        """
        fell = "imbalance" in termination_proposals
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        r_cross = _extract_per_step_scalar(observer_outputs, "cross_support", T)

        return {"r_fall": r_fall, "r_cross": r_cross}

    def compute_episode_metrics(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, float]:
        """Per-episode metrics. ``survived`` = 0 only if the robot fell.

        ``"imbalance"`` in termination_proposals means ImbalanceTerminationPlugin
        triggered (robot fell). ``"timeout"`` means the robot stood the full
        horizon — that counts as survived.
        """
        fell = "imbalance" in termination_proposals
        return {"survived": 0.0 if fell else 1.0}

    def scheduler_info(self) -> Dict[str, Any]:
        """Return current scheduler state for logging."""
        return {
            "survival_rate": round(self._survival_rate, 3),
        }

    def scheduler_state(self) -> dict:
        """Serialize scheduler state for checkpoint."""
        return {
            "survival_rate": self._survival_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._survival_rate = float(state.get("survival_rate", 0.0))


# Singleton instance for the registry
EXPERIMENT = BasicBalanceConfig()
