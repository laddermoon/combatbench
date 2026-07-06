"""Standup v4 — phased from-scratch standup training with dual potential.

Single experiment that trains from zero to success in two phases:

Phase A (foundation, ~500 updates):
  - Smooth potential (no gaps, continuous 0→1)
  - pot_scale = 5.0, terminal_bonus = 50.0
  - height_threshold = 0.5 (easy: fall from half-squat)
  - No wall penalty, no stage5 bonus (keep reward simple)
  - Target: avg_stage > 2.5

Phase B (transition, ~2000 updates):
  - Gapped potential (0.10 gap at Stage 3→4, velocity gate, wall detection)
  - pot_scale = 10.0, terminal_bonus = 100.0
  - height_threshold = 0.3 (harder: lower fall)
  - Wall penalty + stage5 bonus active
  - Target: success > 0.3

Auto-switch: when avg_stage > 2.5 for 3 consecutive evals, advance to Phase B.
No environment restart needed — the potential switch happens in extract_rewards.
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
        # Phase A (smooth potential)
        "phase_a_pot_scale": 8.0,
        "phase_a_terminal_bonus": 50.0,
        "phase_a_height_threshold": 0.5,
        "phase_a_stage3_bonus": 0.05,
        # Phase B (gapped potential)
        "phase_b_pot_scale": 10.0,
        "phase_b_terminal_bonus": 100.0,
        "phase_b_height_threshold": 0.3,
        "phase_b_wall_penalty": -0.05,
        "phase_b_stage5_bonus": 0.1,
        # Phase transition
        "phase_transition_avg_stage": 1.8,
        "phase_transition_consecutive_evals": 3,
    }

    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # --- Stateful ---
    _training_phase: int = 0  # 0 = Phase A, 1 = Phase B
    _consecutive_good_evals: int = 0
    _last_avg_stage: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        if self._training_phase == 0:
            h_thresh = self.custom_config["phase_a_height_threshold"]
        else:
            h_thresh = self.custom_config["phase_b_height_threshold"]
        return self._env_pb().materialize(
            agent_id=agent_id,
            max_steps=self.custom_config["max_steps"],
            height_threshold=h_thresh,
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
        avg_stage = float(eval_metrics.get("avg_stage", 0.0))
        self._last_avg_stage = avg_stage

        # Phase transition: A → B
        if self._training_phase == 0:
            threshold = self.custom_config["phase_transition_avg_stage"]
            required = self.custom_config["phase_transition_consecutive_evals"]

            if avg_stage >= threshold:
                self._consecutive_good_evals += 1
            else:
                self._consecutive_good_evals = 0

            if self._consecutive_good_evals >= required:
                self._training_phase = 1
                self._consecutive_good_evals = 0
                print(
                    f"[standup_v4] Phase A → Phase B: avg_stage={avg_stage:.2f} "
                    f"(threshold={threshold}, consecutive={required}). "
                    f"Switching to gapped potential, pot_scale=10, height_threshold=0.3",
                    flush=True,
                )

        return (1.0,)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        # Select potential field based on training phase
        if self._training_phase == 0:
            potentials = _extract_per_step_field(oo, "standup", "potential_smooth", T)
            pot_scale = float(self.custom_config["phase_a_pot_scale"])
            terminal_bonus = float(self.custom_config["phase_a_terminal_bonus"])
            wall_penalty = 0.0
            stage5_bonus = 0.0
            stage3_bonus = float(self.custom_config.get("phase_a_stage3_bonus", 0.0))
        else:
            potentials = _extract_per_step_field(oo, "standup", "potential_gapped", T)
            pot_scale = float(self.custom_config["phase_b_pot_scale"])
            terminal_bonus = float(self.custom_config["phase_b_terminal_bonus"])
            wall_penalty = float(self.custom_config["phase_b_wall_penalty"])
            stage5_bonus = float(self.custom_config["phase_b_stage5_bonus"])
            stage3_bonus = 0.0

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

        # 3. Per-step Stage 5 bonus
        if stages is not None and stage5_bonus > 0:
            r[:] += stage5_bonus * (stages >= 5.0).astype(np.float32)

        # 3b. Per-step Stage 3+ bonus (Phase A only — encourage hands-off)
        if stages is not None and stage3_bonus > 0:
            r[:] += stage3_bonus * (stages >= 3.0).astype(np.float32)

        # 4. Terminal success bonus
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
        phase_name = "A_smooth" if self._training_phase == 0 else "B_gapped"
        if self._training_phase == 0:
            h_thresh = self.custom_config["phase_a_height_threshold"]
        else:
            h_thresh = self.custom_config["phase_b_height_threshold"]
        return {
            "training_phase": phase_name,
            "avg_stage": round(self._last_avg_stage, 3),
            "consecutive_good_evals": self._consecutive_good_evals,
            "height_threshold": h_thresh,
        }

    def scheduler_state(self) -> dict:
        return {
            "training_phase": self._training_phase,
            "consecutive_good_evals": self._consecutive_good_evals,
            "last_avg_stage": self._last_avg_stage,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._training_phase = int(state.get("training_phase", 0))
        self._consecutive_good_evals = int(state.get("consecutive_good_evals", 0))
        self._last_avg_stage = float(state.get("last_avg_stage", 0.0))


# Register singleton config for the registry
EXPERIMENT = StandupV4Config()
