"""Standup v3 — single policy: stand up from fall + dynamic stepping balance.

Combines potential-based reward shaping (PBRS) for the get-up phase with
balance rewards (cross-support, posture, survival) for the standing phase.

Reward structure:
  - r_potential: PBRS potential difference (guides robot through get-up stages)
  - r_fall: per-step survival bonus + terminal fall penalty
  - r_cross: cross-support balance (alternating foot support = stepping)
  - r_joint: joint deviation penalty (posture quality)
  - r_vel: joint velocity penalty (smooth motion)
  - r_tilt: torso tilt penalty (upright posture)
  - r_foot: foot height penalty (encourages stepping)

PBRS naturally fades when the robot stops transitioning between stages (ΔV→0).
Balance rewards are per-step and become dominant once standing, teaching
the robot to maintain dynamic balance through stepping.

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


class StandupV3Config(CombatExperimentBase):
    """Standup from random fall + dynamic balance — single policy."""

    name = "standup_v3"
    reward_keys = ("r_potential", "r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    gammas = {
        "r_potential": 0.99,
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_joint": 0.99,
        "r_vel": 0.99,
        "r_tilt": 0.99,
        "r_foot": 0.99,
    }

    BLUEPRINT = "standup_v3_env.yaml"

    # --- Training schedule ---
    max_updates: int = 20000
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5

    # --- PPO tuning ---
    log_std_min: float = -4.0
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 0.0

    # --- Video ---
    video_eval_interval: int = 5

    # --- Experiment-specific ---
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 600,
        "potential_reward_scale": 10.0,
        "wall_penalty": -0.05,
        "stage5_per_step_bonus": 0.1,
        "per_step_survival_reward": 0.02,
        "terminal_fall_penalty": 3.0,
        # Curriculum: height_threshold for RandomFallenStatePlugin
        "curriculum_phase": 0,
        "height_thresholds": [0.5, 0.3, 0.15],
        "phase_transition_success_rate": 0.5,
        "phase_transition_eval_count": 5,
    }

    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # --- Stateful ---
    _survival_rate: float = 0.0
    _curriculum_phase: int = 0
    _eval_count: int = 0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        thresholds = self.custom_config["height_thresholds"]
        h_thresh = thresholds[self._curriculum_phase]
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
        return esum.get("survived", 0.0) > best_esum.get("survived", 0.0)

    # ---- Scheduler --------------------------------------------------------

    def initial_weights(self) -> Tuple[float, ...]:
        return (3.0, 6.0, 2.0, 0.2, 0.2, 0.2, 0.2)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        self._survival_rate = float(eval_metrics.get("survived", 0.0))
        self._eval_count += 1

        # Curriculum progression
        thresholds = self.custom_config["height_thresholds"]
        transition_rate = self.custom_config["phase_transition_success_rate"]
        transition_count = self.custom_config["phase_transition_eval_count"]

        if (
            self._curriculum_phase < len(thresholds) - 1
            and self._survival_rate >= transition_rate
            and self._eval_count >= transition_count
        ):
            self._curriculum_phase += 1
            self._eval_count = 0
            print(
                f"[curriculum] Advancing to phase {self._curriculum_phase} "
                f"(height_threshold={thresholds[self._curriculum_phase]}m)",
                flush=True,
            )

        return (3.0, 6.0, 2.0, 0.2, 0.2, 0.2, 0.2)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        # --- PBRS potential ---
        potentials = _extract_per_step_field(oo, "standup", "potential", T)
        heights = _extract_per_step_field(oo, "height", "height", T)
        stages = _extract_per_step_field(oo, "standup", "stage", T)
        wall_contacts = _extract_per_step_field(oo, "standup", "has_wall_contact", T)

        pot_scale = float(self.custom_config.get("potential_reward_scale", 10.0))
        wall_penalty = float(self.custom_config.get("wall_penalty", 0.0))
        stage5_bonus = float(self.custom_config.get("stage5_per_step_bonus", 0.0))

        r_potential = np.zeros(T, dtype=np.float32)
        if potentials is not None:
            r_potential[1:] += pot_scale * (potentials[1:] - potentials[:-1])
            r_potential[0] += pot_scale * (potentials[0] - 0.0)

        # Wall penalty (only at standing height)
        if wall_contacts is not None and wall_penalty != 0.0 and heights is not None:
            standing_mask = (heights > 0.45).astype(np.float32)
            r_potential[:] += wall_penalty * wall_contacts * standing_mask

        # Stage 5 per-step bonus
        if stages is not None and stage5_bonus > 0:
            r_potential[:] += stage5_bonus * (stages >= 5.0).astype(np.float32)

        # --- Survival reward ---
        fell = "imbalance" in episode.termination_proposals
        per_step_survival = float(self.custom_config.get("per_step_survival_reward", 0.01))
        terminal_penalty = float(self.custom_config.get("terminal_fall_penalty", 1.0))
        r_fall = np.full(T, per_step_survival, dtype=np.float32)
        if fell:
            r_fall[-1] = -terminal_penalty
        else:
            r_fall[-1] = terminal_penalty

        # --- Cross-support balance ---
        r_cross = _extract_per_step_scalar(oo, "cross_support", T)
        if r_cross is None:
            r_cross = np.zeros(T, dtype=np.float32)

        # --- Posture rewards ---
        joint_dev_arr = _extract_per_step_field(oo, "posture", "joint_deviation", T)
        joint_vel_arr = _extract_per_step_field(oo, "posture", "joint_vel", T)
        torso_tilt_arr = _extract_per_step_field(oo, "posture", "torso_tilt", T)
        foot_height_arr = _extract_per_step_field(oo, "posture", "foot_height", T)

        if joint_dev_arr is None:
            joint_dev_arr = np.zeros(T, dtype=np.float32)
        if joint_vel_arr is None:
            joint_vel_arr = np.zeros(T, dtype=np.float32)
        if torso_tilt_arr is None:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)
        if foot_height_arr is None:
            foot_height_arr = np.zeros(T, dtype=np.float32)

        # Gate posture rewards by height — only apply when robot is standing
        # (height > 0.45m). During get-up from fallen state, posture values
        # (tilt, vel, joint deviation) are naturally huge and would dominate
        # the advantage signal, drowning out PBRS and survival rewards.
        if heights is not None:
            standing_mask = (heights > 0.45).astype(np.float32)
        else:
            standing_mask = np.ones(T, dtype=np.float32)

        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint)
        r_joint = (r_joint * standing_mask).astype(np.float32)

        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel)
        r_vel = (r_vel * standing_mask).astype(np.float32)

        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)
        r_tilt = (r_tilt * standing_mask).astype(np.float32)

        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot)
        r_foot = (r_foot * standing_mask).astype(np.float32)

        return {
            "r_potential": r_potential,
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        T = episode.num_frames
        oo = episode.observer_outputs

        stages = _extract_per_step_field(oo, "standup", "stage", T)
        potentials = _extract_per_step_field(oo, "standup", "potential", T)

        fell = "imbalance" in episode.termination_proposals
        survived = 0.0 if fell else 1.0

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

        return {
            "survived": survived,
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
            "survival_rate": round(self._survival_rate, 3),
            "curriculum_phase": self._curriculum_phase,
            "height_threshold": thresholds[self._curriculum_phase],
        }

    def scheduler_state(self) -> dict:
        return {
            "survival_rate": self._survival_rate,
            "curriculum_phase": self._curriculum_phase,
            "eval_count": self._eval_count,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._curriculum_phase = int(state.get("curriculum_phase", 0))
        self._eval_count = int(state.get("eval_count", 0))


# Register singleton config for the registry
EXPERIMENT = StandupV3Config()
