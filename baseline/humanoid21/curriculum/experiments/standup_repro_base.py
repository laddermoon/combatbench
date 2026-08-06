"""Shared base class for staged standup reproduction experiments.

Each stage (S1-S9) subclasses this, setting BLUEPRINT, PPO params,
and DEFAULT_CUSTOM_CONFIG to match the training history.
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


class StandupReproBase(CombatExperimentBase):
    """Base class for 9-stage standup reproduction."""

    reward_keys = ("r_standup",)
    gammas = {"r_standup": 0.99}

    BLUEPRINT = "standup_repro_v1_env.yaml"

    # Subclass overrides:
    max_updates: int = 20000
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5
    video_eval_interval: int = 5

    # PPO params (subclass overrides)
    log_std_min: float = -4.0
    learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    entropy_coef: float = 0.0

    # Reward config (subclass overrides)
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 400,
        "potential_reward_scale": 5.0,
        "height_reward_scale": 0.0,
        "terminal_success_bonus": 50.0,
        "time_penalty": 0.0,
        "wall_penalty": 0.0,
        "stage5_per_step_bonus": 0.0,
        "curriculum_phase": 0,
        "height_thresholds": [0.5, 0.3, 0.15],
        "phase_transition_success_rate": 0.5,
        "phase_transition_eval_count": 5,
    }
    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # Termination params (subclass overrides)
    TERMINATION_PARAMS: Dict[str, Any] = {
        "success_height": 0.75,
        "success_uprightness": 0.85,
        "success_hold_steps": 10,
        "stagnation_height": 0.25,
        "stagnation_steps": 150,
    }

    # V2 rewarder params (subclass overrides for V2 stages)
    REWARDER_PARAMS: Dict[str, Any] = {}

    # Stateful
    _success_rate: float = 0.0
    _curriculum_phase: int = 0
    _eval_count: int = 0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        thresholds = self.custom_config["height_thresholds"]
        ht = thresholds[self._curriculum_phase]
        params = dict(
            agent_id=agent_id,
            max_steps=self.custom_config["max_steps"],
            height_threshold=ht,
        )
        params.update(self.TERMINATION_PARAMS)
        params.update(self.REWARDER_PARAMS)
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
        self._eval_count += 1
        thresholds = self.custom_config["height_thresholds"]
        tr = self.custom_config["phase_transition_success_rate"]
        tc = self.custom_config["phase_transition_eval_count"]
        if (self._curriculum_phase < len(thresholds) - 1
                and self._success_rate >= tr and self._eval_count >= tc):
            self._curriculum_phase += 1
            self._eval_count = 0
            print(f"[curriculum] Phase {self._curriculum_phase} "
                  f"(ht={thresholds[self._curriculum_phase]}m)", flush=True)
        return (1.0,)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        potentials = _extract_per_step_field(oo, "standup", "potential", T)
        heights = _extract_per_step_field(oo, "height", "height", T)
        pot_scale = float(self.custom_config.get("potential_reward_scale", 5.0))
        h_scale = float(self.custom_config.get("height_reward_scale", 0.0))
        terminal_bonus = float(self.custom_config.get("terminal_success_bonus", 50.0))
        time_penalty = float(self.custom_config.get("time_penalty", 0.0))
        wall_penalty = float(self.custom_config.get("wall_penalty", 0.0))
        stage5_bonus = float(self.custom_config.get("stage5_per_step_bonus", 0.0))

        wall_contacts = _extract_per_step_field(oo, "standup", "has_wall_contact", T)
        stages = _extract_per_step_field(oo, "standup", "stage", T)

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
        stages = _extract_per_step_field(oo, "standup", "stage", T)
        potentials = _extract_per_step_field(oo, "standup", "potential", T)

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
        thresholds = self.custom_config["height_thresholds"]
        return {
            "success_rate": round(self._success_rate, 3),
            "curriculum_phase": self._curriculum_phase,
            "height_threshold": thresholds[self._curriculum_phase],
        }

    def scheduler_state(self) -> dict:
        return {
            "success_rate": self._success_rate,
            "curriculum_phase": self._curriculum_phase,
            "eval_count": self._eval_count,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))
        self._curriculum_phase = int(state.get("curriculum_phase", 0))
        self._eval_count = int(state.get("eval_count", 0))
