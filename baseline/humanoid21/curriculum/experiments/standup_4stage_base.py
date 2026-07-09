"""Base class for 4-stage standup experiments.

Uses Standup4StageRewarder with a natural prone-to-stand process:
  Stage 0→1: roll over to prone (f_down guided)
  Stage 1→2: find hand+foot support (exploration, allowed jump)
  Stage 2→3: push up to standing (height+uprightness guided)
  Stage 3→4: narrow feet (foot distance guided)

Training: 2-phase chain (exploration → precise), same as original S1→S3 pattern.
Thresholds from original S8/S9 final working config.
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


class Standup4StageBase(CombatExperimentBase):
    """Base class for 4-stage standup experiments."""

    reward_keys = ("r_standup",)
    gammas = {"r_standup": 0.99}

    BLUEPRINT = "standup_4stage_env.yaml"
    max_updates: int = 4000
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5
    video_eval_interval: int = 5

    # PPO params — minibatch_size=4096 matching original
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    # Reward config (Phase A matches S1: pure potential, no extras)
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 200,
        "potential_reward_scale": 1.0,
        "terminal_success_bonus": 0.0,
        "time_penalty": 0.0,
        "stage4_per_step_bonus": 0.0,
    }
    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    _success_rate: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        return self._env_pb().materialize(
            agent_id=agent_id,
            max_steps=self.custom_config["max_steps"],
        )

    def video_env_blueprint(self):
        return self._materialize_env("robot_a")

    # ---- Job construction -------------------------------------------------

    def _build_jobs(self, policy_bp, base_seed, n_episodes):
        env_bp = self._materialize_env("robot_a")
        jobs = []
        for i in range(n_episodes):
            jobs.append((
                policy_bp, policy_bp, env_bp, int(base_seed + i),
                {"agent_id": "robot_a", "initial_distance": 2.0},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp, base_seed):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp, base_seed):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # ---- Eval comparison --------------------------------------------------

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("success", 0.0) > best_esum.get("success", 0.0)

    # ---- Scheduler --------------------------------------------------------

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0,)

    def next_weights(self, eval_metrics, current_weights):
        self._success_rate = float(eval_metrics.get("success", 0.0))
        return (1.0,)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        potentials = _extract_per_step_field(oo, "standup", "potential", T)
        pot_scale = float(self.custom_config.get("potential_reward_scale", 1.0))
        terminal_bonus = float(self.custom_config.get("terminal_success_bonus", 0.0))
        time_penalty = float(self.custom_config.get("time_penalty", 0.0))
        stage4_bonus = float(self.custom_config.get("stage4_per_step_bonus", 0.0))

        stages = _extract_per_step_field(oo, "standup", "stage", T)

        r = np.zeros(T, dtype=np.float32)

        if potentials is not None:
            delta_pot = pot_scale * (potentials[1:] - potentials[:-1])
            neg_mask = delta_pot < 0
            delta_pot[neg_mask] *= 1.2
            r[1:] += delta_pot
            r[0] += pot_scale * (potentials[0] - 0.0)

        r[:] += time_penalty

        if stages is not None and stage4_bonus > 0:
            r[:] += stage4_bonus * (stages >= 4.0).astype(np.float32)

        if stages is not None and terminal_bonus > 0:
            if float(np.max(stages)) >= 4.0:
                r[-1] += terminal_bonus

        return {"r_standup": r}

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        T = episode.num_frames
        oo = episode.observer_outputs
        stages = _extract_per_step_field(oo, "standup", "stage", T)
        potentials = _extract_per_step_field(oo, "standup", "potential", T)
        foot_dists = _extract_per_step_field(oo, "standup", "foot_distance", T)

        if stages is not None and len(stages) > 0:
            max_stage = float(np.max(stages))
            final_stage = float(stages[-1])
            success = 1.0 if max_stage >= 4.0 else 0.0
            avg_stage = float(np.mean(stages))
        else:
            max_stage = final_stage = success = avg_stage = 0.0

        max_pot = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0
        final_pot = float(potentials[-1]) if potentials is not None and len(potentials) > 0 else 0.0

        min_foot_dist = float(np.min(foot_dists)) if foot_dists is not None and len(foot_dists) > 0 else 0.0
        final_foot_dist = float(foot_dists[-1]) if foot_dists is not None and len(foot_dists) > 0 else 0.0

        term_reasons = getattr(episode, "termination_proposals", [])
        early_success = 1.0 if any("success" in str(r) for r in term_reasons) else 0.0

        return {
            "success": success, "early_success": early_success,
            "max_stage": max_stage, "final_stage": final_stage,
            "avg_stage": avg_stage, "max_potential": max_pot,
            "final_potential": final_pot,
            "min_foot_dist": min_foot_dist,
            "final_foot_dist": final_foot_dist,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {"success_rate": round(self._success_rate, 3)}

    def scheduler_state(self) -> dict:
        return {"success_rate": self._success_rate}

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))
