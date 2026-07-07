"""Standup v4 — from-scratch standup training with smooth potential + transition bonuses.

Single experiment using only the smooth (gapless) potential with upward
transition bonuses to bridge risky stage crossings.

Key design:
  - Smooth potential (no gaps, continuous 0→1) throughout
  - Upward transition bonus: one-time reward for crossing stage boundaries upward
  - Per-step stage3+ bonus to incentivize hands-off balancing
  - pot_scale = 8.0, terminal_bonus = 50.0
  - height_threshold = 0.5 (easier starts, more time to practice standing)
  - No phase switching — gapped potential removed (caused KL explosion)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.ppo_trainer import (
    _extract_per_step_field,
    _extract_per_step_scalar,
)
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class StandupV4Config(CombatExperimentBase):
    """Phased standup training — from zero to success in one experiment."""

    name = "standup_v4"
    reward_keys = ("r_standup",)
    gammas = {
        "r_standup": 0.99,
    }

    BLUEPRINT = "standup_v4_env.yaml"

    # --- Training schedule ---
    max_updates: int = 5000
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5

    # --- PPO tuning ---
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
        "pot_scale": 8.0,
        "terminal_bonus": 50.0,
        "height_threshold": 0.5,
        "stage3_per_step_bonus": 0.1,
        "stage5_per_step_bonus": 0.2,
        "transition_bonus": 2.0,
        "wall_penalty": -0.05,
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
        T = episode.num_frames
        oo = episode.observer_outputs

        potentials = _extract_per_step_field(oo, "standup", "potential_smooth", T)
        pot_scale = float(self.custom_config["pot_scale"])
        terminal_bonus = float(self.custom_config["terminal_bonus"])
        wall_penalty = float(self.custom_config["wall_penalty"])
        stage3_bonus = float(self.custom_config["stage3_per_step_bonus"])
        stage5_bonus = float(self.custom_config["stage5_per_step_bonus"])
        trans_bonus = float(self.custom_config["transition_bonus"])

        heights = _extract_per_step_field(oo, "height", "height", T)
        wall_contacts = _extract_per_step_field(oo, "standup", "has_wall_contact", T)
        stages = _extract_per_step_field(oo, "standup", "stage", T)

        r = np.zeros(T, dtype=np.float32)

        # 1. PBRS potential difference
        if potentials is not None:
            r[1:] += pot_scale * (potentials[1:] - potentials[:-1])
            r[0] += pot_scale * (potentials[0] - 0.0)

        # 2. Wall penalty (only at standing height)
        if wall_contacts is not None and wall_penalty != 0.0 and heights is not None:
            standing_mask = (heights > 0.45).astype(np.float32)
            r[:] += wall_penalty * wall_contacts * standing_mask

        # 3. Per-step stage bonuses
        if stages is not None:
            if stage3_bonus > 0:
                r[:] += stage3_bonus * (stages >= 3.0).astype(np.float32)
            if stage5_bonus > 0:
                r[:] += stage5_bonus * (stages >= 5.0).astype(np.float32)

        # 4. Upward transition bonus — one-time reward for crossing stage boundaries upward
        if stages is not None and trans_bonus > 0 and T > 1:
            stage_arr = np.asarray(stages, dtype=np.float32)
            upward = stage_arr[1:] > stage_arr[:-1]
            r[1:][upward] += trans_bonus

        # 5. Terminal success bonus
        term_reasons = getattr(episode, "termination_proposals", [])
        if any("success" in str(r_) for r_ in term_reasons):
            r[-1] += terminal_bonus

        return {
            "r_standup": r,
        }

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        T = episode.num_frames
        oo = episode.observer_outputs

        stages = _extract_per_step_field(oo, "standup", "stage", T)
        potentials = _extract_per_step_field(oo, "standup", "potential_gapped", T)

        if stages is not None and len(stages) > 0:
            max_stage = float(np.max(stages))
            final_stage = float(stages[-1])
            success = 1.0 if max_stage >= 5.0 else 0.0
            avg_stage = float(np.mean(stages))
        else:
            max_stage = 0.0
            final_stage = 0.0
            success = 0.0
            avg_stage = 0.0

        max_potential = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0
        final_potential = float(potentials[-1]) if potentials is not None and len(potentials) > 0 else 0.0

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
EXPERIMENT = StandupV4Config()
