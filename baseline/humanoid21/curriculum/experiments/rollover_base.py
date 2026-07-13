"""Base class for rollover ablation: Delta vs PBRS.

Pure orientation signal (rollover_score = clip((f_down+1)/2, 0, 1)).
No base reward, no terminal bonus, no termination plugin.

Two reward modes:
  - "delta":  r_t = φ(t) - φ(t-1)
  - "pbrs":   r_t = γ·φ(t) - φ(t-1)

All other settings identical to standup_4stage_a.
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


class RolloverBase(CombatExperimentBase):
    """Base class for rollover Delta-vs-PBRS ablation."""

    reward_keys = ("r_standup",)
    gammas = {"r_standup": 0.99}

    BLUEPRINT = "rollover_env.yaml"
    reward_mode: str = "delta"  # overridden by subclasses

    max_updates: int = 300
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5
    video_eval_interval: int = 5

    # PPO params — identical to standup_4stage_a
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 200,
        "potential_reward_scale": 1.0,
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
        return esum.get("max_potential", 0.0) > best_esum.get("max_potential", 0.0)

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
        gamma = self.gammas["r_standup"]

        r = np.zeros(T, dtype=np.float32)

        if potentials is not None:
            if self.reward_mode == "delta":
                # r_t = φ(t) - φ(t-1)
                r[1:] = pot_scale * (potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (potentials[0] - 0.0)
            elif self.reward_mode == "pbrs":
                # r_t = γ·φ(t) - φ(t-1)
                r[1:] = pot_scale * (gamma * potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (gamma * potentials[0] - 0.0)
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        return {"r_standup": r}

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        T = episode.num_frames
        oo = episode.observer_outputs
        potentials = _extract_per_step_field(oo, "standup", "potential", T)

        max_pot = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0
        final_pot = float(potentials[-1]) if potentials is not None and len(potentials) > 0 else 0.0
        avg_pot = float(np.mean(potentials)) if potentials is not None and len(potentials) > 0 else 0.0

        success = 1.0 if max_pot >= 0.9 else 0.0

        return {
            "success": success,
            "max_potential": max_pot,
            "final_potential": final_pot,
            "avg_potential": avg_pot,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {"success_rate": round(self._success_rate, 3)}

    def scheduler_state(self) -> dict:
        return {"success_rate": self._success_rate}

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))
