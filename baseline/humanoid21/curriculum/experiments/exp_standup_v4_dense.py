"""Standup v4 Dense — smooth potential with Dense reward mode.

Uses the same StandupPotentialRewarderV3 smooth potential as V4,
but replaces V4's complex reward assembly (PBRS + transition bonus +
per-step bonuses + terminal bonus) with pure Dense:

    r_t = (1-γ)·φ_smooth(t)

This isolates the effect of the smooth potential function itself,
without any engineered bonuses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.ppo_trainer import _extract_per_step_field
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class StandupV4DenseConfig(CombatExperimentBase):
    """Standup from random fall using V4 smooth potential + Dense reward."""

    name = "standup_v4_dense"
    reward_keys = ("r_potential",)
    gammas = {"r_potential": 0.99}

    BLUEPRINT = "standup_v4_env.yaml"

    # --- Training schedule ---
    max_updates: int = 5000
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5

    # --- PPO tuning (aligned with V4) ---
    log_std_min: float = -4.0
    learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 0.0

    # --- Video ---
    video_eval_interval: int = 5

    # --- Experiment-specific ---
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 400,
        "potential_reward_scale": 1.0,
        "height_threshold": 0.5,
    }

    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # --- Stateful ---
    _last_avg_stage: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        return self._env_pb().materialize(
            agent_id=agent_id,
            max_steps=self.custom_config["max_steps"],
            height_threshold=self.custom_config["height_threshold"],
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
        env_bp = self._materialize_env("robot_a")
        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            jobs.append((
                policy_bp, policy_bp,
                env_bp, seed,
                {"agent_id": "robot_a", "initial_distance": 2.0},
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
        return (1.0,)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        self._last_avg_stage = float(eval_metrics.get("avg_stage", 0.0))
        return (1.0,)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Dense reward: r_t = (1-γ)·φ_smooth(t)."""
        T = episode.num_frames
        oo = episode.observer_outputs

        potentials = _extract_per_step_field(oo, "standup", "potential_smooth", T)
        r = np.zeros(T, dtype=np.float32)

        if potentials is not None:
            pot_scale = float(self.custom_config.get("potential_reward_scale", 1.0))
            gamma = self.gammas["r_potential"]
            r[:] = pot_scale * (1.0 - gamma) * potentials[:]

        return {"r_potential": r}

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        T = episode.num_frames
        oo = episode.observer_outputs

        stages = _extract_per_step_field(oo, "standup", "stage", T)
        potentials = _extract_per_step_field(oo, "standup", "potential_smooth", T)

        if stages is not None and len(stages) > 0:
            max_stage = float(np.max(stages))
            final_stage = float(stages[-1])
            avg_stage = float(np.mean(stages))
        else:
            max_stage = 0.0
            final_stage = 0.0
            avg_stage = 0.0

        max_potential = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0
        final_potential = float(potentials[-1]) if potentials is not None and len(potentials) > 0 else 0.0

        success = 1.0 if max_stage >= 5.0 else 0.0

        term_reasons = getattr(episode, "termination_proposals", [])
        early_success = 1.0 if any("success" in str(r) for r in term_reasons) else 0.0

        return {
            "success": success,
            "early_success": early_success,
            "max_stage": max_stage,
            "final_stage": final_stage,
            "avg_stage": avg_stage,
            "max_potential": max_potential,
            "final_potential": final_potential,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "avg_stage": round(self._last_avg_stage, 3),
            "height_threshold": self.custom_config["height_threshold"],
        }

    def scheduler_state(self) -> dict:
        return {
            "last_avg_stage": self._last_avg_stage,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._last_avg_stage = float(state.get("last_avg_stage", 0.0))


# Register singleton config for the registry
EXPERIMENT = StandupV4DenseConfig()
