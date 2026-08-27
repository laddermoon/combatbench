"""Curriculum training experiment for humanoid21 Standup task.

This experiment uses RandomFallenStatePlugin to initialize the robot in a
random fallen state at the start of each episode. The learning policy then
trains to stand up from that state. No mixed policy or episode segmentation
is needed — every step is a trainable stand-up step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.common.rollout import (
    extract_per_step_field,
    extract_per_step_scalar,
)
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class StandupRandomFallConfig(CombatExperimentBase):
    """Standup curriculum experiment with random fall initialization."""

    name = "standup_random_fall"
    reward_keys = ("r_potential", "r_cross")
    gammas = {
        "r_potential": 0.99,
        "r_cross": 0.99,
    }

    BLUEPRINT = "standup_random_fall_env.yaml"

    max_updates: int = 15000

    # --- PPO tuning ---
    log_std_min: float = -1.8
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 2

    # --- Video recording ---
    video_eval_interval: int = 2

    # --- Stateful metrics ---
    _success_rate: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        return self._env_pb().materialize(
            agent_id=agent_id,
        )

    def video_env_blueprint(self):
        return self._materialize_env("robot_a")

    # ---- Job construction -------------------------------------------------

    def _build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        env_bps: Dict[str, EnvBlueprint] = {
            aid: self._materialize_env(aid)
            for aid in ("robot_a", "robot_b")
        }

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = "robot_a"

            jobs.append((
                policy_bp, policy_bp,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": 2.0},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # ---- Eval comparison --------------------------------------------------

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("success", 0.0) > best_esum.get("success", 0.0)

    # ---- Scheduler --------------------------------------------------------

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0, 0.1)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        self._success_rate = float(eval_metrics.get("success", 0.0))
        return (1.0, 0.1)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Extract potential-difference reward and cross-support balance reward."""
        T = episode.num_frames
        oo = episode.observer_outputs

        # Extract potential values from the StandupPotentialRewarder observer plugin
        potentials = extract_per_step_field(oo, "standup", "potential", T)
        r_potential = np.zeros(T, dtype=np.float32)
        if potentials is not None:
            # r_potential[t] = potentials[t] - potentials[t-1] (Potential Difference)
            r_potential[1:] = potentials[1:] - potentials[:-1]
            r_potential[0] = potentials[0] - 0.0
            
            # Scale potential difference so the total possible reward sum is 10.0
            scale = float(self.custom_config.get("potential_reward_scale", 10.0))
            r_potential *= scale

        # Extract cross support balance reward
        r_cross = extract_per_step_scalar(oo, "cross_support", T)
        if r_cross is None:
            r_cross = np.zeros(T, dtype=np.float32)

        return {
            "r_potential": r_potential,
            "r_cross": r_cross,
        }

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Compute metrics for success monitoring and curriculum progression."""
        T = episode.num_frames
        oo = episode.observer_outputs

        stages = extract_per_step_field(oo, "standup", "stage", T)
        potentials = extract_per_step_field(oo, "standup", "potential", T)

        if stages is not None and len(stages) > 0:
            max_stage = float(np.max(stages))
            success = 1.0 if max_stage >= 5.0 else 0.0
            avg_stage = float(np.mean(stages))
        else:
            max_stage = 0.0
            success = 0.0
            avg_stage = 0.0

        max_potential = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0

        return {
            "success": success,
            "max_stage": max_stage,
            "avg_stage": avg_stage,
            "max_potential": max_potential,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "success_rate": round(self._success_rate, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "success_rate": self._success_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))


# Register singleton config for the registry
EXPERIMENT = StandupRandomFallConfig()