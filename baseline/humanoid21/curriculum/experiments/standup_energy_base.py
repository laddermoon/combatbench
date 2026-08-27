"""Base class for energy-based standup experiments.

Single continuous orbital-energy potential (StandupEnergyRewarder).
No stage machinery.  Success = no hands/other body parts touching ground
for 40 consecutive steps (~2 seconds), detected by StandupEnergyTerminationPlugin.

Training: single phase, PPO with entropy exploration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.common.rollout import extract_per_step_field
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class StandupEnergyBase(CombatExperimentBase):
    """Base class for energy-based standup experiments."""

    reward_keys = ("r_standup",)
    gammas = {"r_standup": 0.99}

    BLUEPRINT = "standup_energy_env.yaml"
    max_updates: int = 5000
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5
    video_eval_interval: int = 5

    # PPO params
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    # Reward config
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 200,
        "potential_reward_scale": 1.0,
        "terminal_success_bonus": 100.0,
        "time_penalty": 0.0,
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

        potentials = extract_per_step_field(oo, "standup", "potential", T)
        pot_scale = float(self.custom_config.get("potential_reward_scale", 1.0))
        terminal_bonus = float(self.custom_config.get("terminal_success_bonus", 0.0))
        time_penalty = float(self.custom_config.get("time_penalty", 0.0))

        r = np.zeros(T, dtype=np.float32)

        if potentials is not None:
            delta_pot = pot_scale * (potentials[1:] - potentials[:-1])
            r[1:] += delta_pot
            r[0] += pot_scale * (potentials[0] - 0.0)

        r[:] += time_penalty

        # Terminal success bonus: check termination reason
        if terminal_bonus > 0:
            term_reasons = episode.agent_termination_reason.values()
            if any("success" in str(reason) for reason in term_reasons):
                r[-1] += terminal_bonus

        return {"r_standup": r}

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        T = episode.num_frames
        oo = episode.observer_outputs
        potentials = extract_per_step_field(oo, "standup", "potential", T)
        e_scores = extract_per_step_field(oo, "standup", "e_score", T)
        is_balanced = extract_per_step_field(oo, "standup", "is_balanced", T)

        max_pot = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0
        final_pot = float(potentials[-1]) if potentials is not None and len(potentials) > 0 else 0.0
        avg_e_score = float(np.mean(e_scores)) if e_scores is not None and len(e_scores) > 0 else 0.0

        balanced_steps = 0
        if is_balanced is not None and len(is_balanced) > 0:
            balanced_steps = int(np.sum(is_balanced > 0.5))

        term_reasons = episode.agent_termination_reason.values()
        success = 1.0 if any("success" in str(r) for r in term_reasons) else 0.0
        stagnation = 1.0 if any("stagnation" in str(r) for r in term_reasons) else 0.0

        return {
            "success": success,
            "stagnation": stagnation,
            "max_potential": max_pot,
            "final_potential": final_pot,
            "avg_e_score": avg_e_score,
            "balanced_steps": balanced_steps,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {"success_rate": round(self._success_rate, 3)}

    def scheduler_state(self) -> dict:
        return {"success_rate": self._success_rate}

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))
