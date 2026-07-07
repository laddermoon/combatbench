
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.ppo_trainer import _extract_per_step_scalar, _extract_per_step_field
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


class BasicBalanceV2Config(CombatExperimentBase):

    name = "basic_balance_v2"
    reward_keys = ("r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_joint": 0.99,
        "r_vel": 0.99,
        "r_tilt": 0.99,
        "r_foot": 0.99,
    }

    # Stage blueprints
    BLUEPRINT = "basic_balance_v2_env.yaml"  # Basic: fall detection only

    # SAC: auto-alpha is REQUIRED for the PPO-aligned per-component Q
    # normalization.  Normalizing Q changes its scale relative to the
    # entropy term (alpha * logpi); a fixed alpha would let entropy dominate
    # and collapse the policy.  Auto-alpha holds the policy entropy at
    # target_entropy regardless of the (normalized) Q scale.
    sac_auto_alpha = True

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
        return (3.0, 1.0, 0.2, 0.2, 0.2, 0.2)

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

        # Weights always regardless of stage.
        return (3.0, 1.0, 0.2, 0.2, 0.2, 0.2)

    # Small per-step survival bonus (each alive step is worth this much).
    per_step_survival_reward: float = 0.01

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """r_fall: per-step survival bonus + terminal signal.
        r_cross: cross-support balance reward from CrossSupportBalanceRewarder.
        r_joint: joint deviation penalty/bonus.
        r_vel: joint angular velocity penalty/bonus.
        r_tilt: torso tilt penalty/bonus.
        r_foot: foot clearance penalty/bonus.
        """
        T = episode.num_frames
        fell = "imbalance" in episode.termination_proposals
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        r_cross = _extract_per_step_scalar(episode.observer_outputs, "cross_support", T)

        # Extract fields from the 'posture' observer
        joint_dev_arr = _extract_per_step_field(episode.observer_outputs, "posture", "joint_deviation", T)
        joint_vel_arr = _extract_per_step_field(episode.observer_outputs, "posture", "joint_vel", T)
        torso_tilt_arr = _extract_per_step_field(episode.observer_outputs, "posture", "torso_tilt", T)
        foot_height_arr = _extract_per_step_field(episode.observer_outputs, "posture", "foot_height", T)

        # Fallback if observer fields are missing
        if joint_dev_arr is None:
            joint_dev_arr = np.zeros(T, dtype=np.float32)
        if joint_vel_arr is None:
            joint_vel_arr = np.zeros(T, dtype=np.float32)
        if torso_tilt_arr is None:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)
        if foot_height_arr is None:
            foot_height_arr = np.zeros(T, dtype=np.float32)

        # Vectorized calculation: small bonus when within normal range, linear penalty outside
        # 1. Joint deviation: normal <= 0.12, penalty factor 5.0
        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint)

        # 2. Joint velocity: normal <= 0.5, penalty factor 1.0
        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel)

        # 3. Torso tilt: normal <= 0.26 rad (~15 deg), penalty factor 3.0
        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)

        # 4. Foot height: normal <= 0.10m, penalty factor 5.0
        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot)

        return {
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Per-episode metrics. ``survived`` = 0 only if the robot fell.

        ``"imbalance"`` in termination_proposals means ImbalanceTerminationPlugin
        triggered (robot fell). ``"timeout"`` means the robot stood the full
        horizon — that counts as survived.
        """
        fell = "imbalance" in episode.termination_proposals
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
EXPERIMENT = BasicBalanceV2Config()
