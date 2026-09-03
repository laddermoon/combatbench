"""Base class for original-code standup reproduction experiments.

Uses the ACTUAL original rewarder classes (not rewritten versions).
height_threshold is always 0.3 (hardcoded in blueprints, never parameterized).
minibatch_size = 4096 (matching original, not 8192 from CombatExperimentBase).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.rollout import extract_per_step_field
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class StandupOrigBase(CombatExperimentBase):
    """Base class for original-code standup reproduction."""

    reward_keys = ("r_standup",)
    gammas = {"r_standup": 0.99}

    # Subclass overrides:
    BLUEPRINT = "standup_orig_v1_env.yaml"
    max_updates: int = 20000
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5
    video_eval_interval: int = 5

    # PPO params — minibatch_size=4096 matching original (NOT 8192 from base)
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    # Reward config (subclass overrides)
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 400,
        "potential_reward_scale": 1.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 0.0,
        "time_penalty": 0.0,
        "wall_penalty": 0.0,
        "stage5_per_step_bonus": 0.0,
    }
    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # Termination params (passed to blueprint)
    TERMINATION_PARAMS: Dict[str, Any] = {
        "success_height": 0.75,
        "success_uprightness": 0.85,
        "success_hold_steps": 10,
        "stagnation_height": 0.25,
        "stagnation_steps": 150,
    }

    # Stateful
    _success_rate: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        params = dict(
            agent_id=agent_id,
            max_steps=self.custom_config["max_steps"],
        )
        params.update(self.TERMINATION_PARAMS)
        return self._env_pb().materialize(**params)

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
        heights = extract_per_step_field(oo, "height", "height", T)
        pot_scale = float(self.custom_config.get("potential_reward_scale", 1.0))
        h_scale = float(self.custom_config.get("height_reward_scale", 0.0))
        terminal_bonus = float(self.custom_config.get("terminal_success_bonus", 0.0))
        time_penalty = float(self.custom_config.get("time_penalty", 0.0))
        wall_penalty = float(self.custom_config.get("wall_penalty", 0.0))
        stage5_bonus = float(self.custom_config.get("stage5_per_step_bonus", 0.0))

        wall_contacts = extract_per_step_field(oo, "standup", "has_wall_contact", T)
        stages = extract_per_step_field(oo, "standup", "stage", T)

        r = np.zeros(T, dtype=np.float32)

        if potentials is not None:
            r[1:] += pot_scale * (potentials[1:] - potentials[:-1])
            r[0] += pot_scale * (potentials[0] - 0.0)

        if heights is not None and h_scale > 0:
            r[1:] += h_scale * (heights[1:] - heights[:-1])

        r[:] += time_penalty

        if wall_contacts is not None and wall_penalty != 0.0 and heights is not None:
            standing_mask = (heights > 0.45).astype(np.float32)
            r[:] += wall_penalty * wall_contacts * standing_mask

        if stages is not None and stage5_bonus > 0:
            r[:] += stage5_bonus * (stages >= 5.0).astype(np.float32)

        term_reasons = episode.agent_termination_reason.values()
        if any("success" in str(r_) for r_ in term_reasons):
            r[-1] += terminal_bonus

        return {"r_standup": r}

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        T = episode.num_frames
        oo = episode.observer_outputs
        stages = extract_per_step_field(oo, "standup", "stage", T)
        potentials = extract_per_step_field(oo, "standup", "potential", T)

        if stages is not None and len(stages) > 0:
            max_stage = float(np.max(stages))
            final_stage = float(stages[-1])
            success = 1.0 if max_stage >= 5.0 else 0.0
            avg_stage = float(np.mean(stages))
        else:
            max_stage = final_stage = success = avg_stage = 0.0

        max_pot = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0
        final_pot = float(potentials[-1]) if potentials is not None and len(potentials) > 0 else 0.0

        term_reasons = episode.agent_termination_reason.values()
        early_success = 1.0 if any("success" in str(r) for r in term_reasons) else 0.0

        return {
            "success": success, "early_success": early_success,
            "max_stage": max_stage, "final_stage": final_stage,
            "avg_stage": avg_stage, "max_potential": max_pot,
            "final_potential": final_pot,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {"success_rate": round(self._success_rate, 3)}

    def scheduler_state(self) -> dict:
        return {"success_rate": self._success_rate}

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))
