"""Standup v2 — train robot to stand up from random fallen state.

Single-critic design using potential-based reward shaping (PBRS).
No cross-support balance reward (that's for after standing, not during get-up).

Curriculum: gradually lower the RandomFallenStatePlugin height_threshold
so the robot starts from progressively harder fallen states.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.humanoid21.curriculum.framework.ppo_trainer import (
    _extract_per_step_field,
    _extract_per_step_scalar,
)
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class StandupV2Config(CombatExperimentBase):
    """Standup from random fall — v2 with simplified reward and curriculum."""

    name = "standup_v2"
    reward_keys = ("r_potential",)
    gammas = {
        "r_potential": 0.99,
    }

    BLUEPRINT = "standup_v2_env.yaml"

    # --- Training schedule ---
    max_updates: int = 20000
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5

    # --- PPO tuning ---
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    # --- Video ---
    video_eval_interval: int = 5

    # --- Experiment-specific ---
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 400,
        "potential_reward_scale": 1.0,
        # Curriculum: height_threshold for RandomFallenStatePlugin
        # Phase 0: fall from half-squat (0.5m) — easy
        # Phase 1: fall from lower (0.3m) — medium
        # Phase 2: fall from ground (0.15m) — hard
        "curriculum_phase": 0,
        "height_thresholds": [0.5, 0.3, 0.15],
        "phase_transition_success_rate": 0.5,
        "phase_transition_eval_count": 5,
    }

    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # --- Stateful ---
    _success_rate: float = 0.0
    _curriculum_phase: int = 0
    _eval_count: int = 0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
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
        self._success_rate = float(eval_metrics.get("success", 0.0))
        self._eval_count += 1

        # Curriculum progression
        thresholds = self.custom_config["height_thresholds"]
        transition_rate = self.custom_config["phase_transition_success_rate"]
        transition_count = self.custom_config["phase_transition_eval_count"]

        if (
            self._curriculum_phase < len(thresholds) - 1
            and self._success_rate >= transition_rate
            and self._eval_count >= transition_count
        ):
            self._curriculum_phase += 1
            self._eval_count = 0
            print(
                f"[curriculum] Advancing to phase {self._curriculum_phase} "
                f"(height_threshold={thresholds[self._curriculum_phase]}m)",
                flush=True,
            )

        return (1.0,)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        potentials = _extract_per_step_field(oo, "standup", "potential", T)
        r_potential = np.zeros(T, dtype=np.float32)
        if potentials is not None:
            r_potential[1:] = potentials[1:] - potentials[:-1]
            r_potential[0] = potentials[0] - 0.0
            scale = float(self.custom_config.get("potential_reward_scale", 1.0))
            r_potential *= scale

        return {
            "r_potential": r_potential,
        }

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
            max_stage = 0.0
            final_stage = 0.0
            success = 0.0
            avg_stage = 0.0

        max_potential = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0
        final_potential = float(potentials[-1]) if potentials is not None and len(potentials) > 0 else 0.0

        # Check termination reason
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


# Register singleton config for the registry
EXPERIMENT = StandupV2Config()
