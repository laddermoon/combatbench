"""basic_balance_v2_stage_seg: staged reward with segment-based phase control.

Based on basic_balance_v2, adds two-phase reward scheme:
  - **Struggle phase**: goal is to recover balance. Reward = per-step penalty
    (-0.01) + terminal: +1 if recovers to stability, -1 if falls.
  - **Stability phase**: same as basic_balance_v2 (r_cross, r_joint, r_vel,
    r_tilt, r_foot) but r_fall is replaced by r_struggle which penalizes
    transition from stability to struggle.

Uses PhaseObserver (uprightness + height with hysteresis) to determine phase.
Uses prepare_segments (v2 API) to split episodes into per-phase segments with
per-key critic control via Segment.key_weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.experiment import Segment
from baseline.framework.ppo_trainer import _extract_per_step_scalar, _extract_per_step_field
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint


class BasicBalanceV2StageSegConfig(CombatExperimentBase):

    name = "basic_balance_v2_stage_seg"
    # r_struggle replaces r_fall: struggle-phase survival + phase transition rewards
    # Stability-phase keys remain the same as basic_balance_v2
    reward_keys = ("r_struggle", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    gammas = {
        "r_struggle": 0.99,
        "r_cross": 0.99,
        "r_joint": 0.99,
        "r_vel": 0.99,
        "r_tilt": 0.99,
        "r_foot": 0.99,
    }

    BLUEPRINT = "basic_balance_v2_stage_seg_env.yaml"

    sac_auto_alpha = True

    _survival_rate: float = 0.0

    # Phase reward constants
    struggle_per_step_penalty: float = -0.01
    struggle_recover_bonus: float = 1.0
    struggle_fall_penalty: float = -1.0
    stability_to_struggle_penalty: float = -1.0
    per_step_stability_bonus: float = 0.01

    def video_env_blueprint(self):
        return self._make_video_blueprint(self._env_pb())

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def build_rollout_jobs(self, policy_bp, base_seed):
        return self._build_selfplay_jobs(self._env_pb(), policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp, base_seed):
        return self._build_selfplay_jobs(self._env_pb(), policy_bp, base_seed, self.eval_episodes)

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("survived", 0.0) > best_esum.get("survived", 0.0)

    def initial_weights(self) -> Tuple[float, ...]:
        return (3.0, 1.0, 0.2, 0.2, 0.2, 0.2)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        survival_rate = float(eval_metrics.get("survived", 0.0))
        self._survival_rate = survival_rate
        return (3.0, 1.0, 0.2, 0.2, 0.2, 0.2)

    def _extract_phase_info(self, episode) -> Tuple[np.ndarray, np.ndarray]:
        """Extract per-step phase and transition arrays from PhaseObserver output.

        Returns:
            is_struggle: (T,) bool array — True if step is in struggle phase.
            transition: (T,) object array — transition type per step.
        """
        T = episode.num_frames
        phase_node = episode.observer_outputs.get("phase")
        if phase_node is None:
            return np.zeros(T, dtype=bool), np.array(["none"] * T, dtype=object)

        is_struggle = np.zeros(T, dtype=bool)
        transitions = np.array(["none"] * T, dtype=object)

        if isinstance(phase_node, dict):
            # PhaseObserver outputs a dict with per-step fields
            phase_arr = phase_node.get("is_struggle")
            trans_arr = phase_node.get("transition")
            if phase_arr is not None:
                is_struggle = np.asarray(phase_arr, dtype=bool).reshape(-1)
                if is_struggle.shape[0] != T:
                    is_struggle = np.zeros(T, dtype=bool)
            if trans_arr is not None:
                transitions = np.asarray(trans_arr, dtype=object).reshape(-1)
                if transitions.shape[0] != T:
                    transitions = np.array(["none"] * T, dtype=object)
        else:
            # If observer outputs a scalar/list of scalars
            try:
                raw = np.asarray(phase_node, dtype=object).reshape(-1)
                for t in range(min(len(raw), T)):
                    val = raw[t]
                    if isinstance(val, dict):
                        is_struggle[t] = val.get("is_struggle", False)
                        transitions[t] = val.get("transition", "none")
                    elif isinstance(val, str):
                        is_struggle[t] = val == "struggle"
            except Exception:
                pass

        return is_struggle, transitions

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Phase-dependent reward extraction.

        Struggle phase:
          - r_struggle: per-step -0.01, +1 on recover transition, -1 on fall
          - r_cross, r_joint, r_vel, r_tilt, r_foot: all zeros (not relevant)

        Stability phase:
          - r_struggle: only -1 on stability_to_struggle transition (no per-step bonus)
          - r_cross, r_joint, r_vel, r_tilt, r_foot: same as basic_balance_v2
        """
        T = episode.num_frames
        fell = "imbalance" in episode.termination_proposals

        is_struggle, transitions = self._extract_phase_info(episode)

        # --- r_struggle ---
        r_struggle = np.zeros(T, dtype=np.float32)

        for t in range(T):
            if is_struggle[t]:
                r_struggle[t] = self.struggle_per_step_penalty
                # Check for recovery transition at this step
                if transitions[t] == "struggle_to_stability":
                    r_struggle[t] += self.struggle_recover_bonus
            else:
                # Stability phase: only penalize transition back to struggle
                if transitions[t] == "stability_to_struggle":
                    r_struggle[t] = self.stability_to_struggle_penalty

        # Terminal: if fell during struggle phase
        if fell:
            r_struggle[-1] += self.struggle_fall_penalty

        # --- Stability-phase rewards (same as basic_balance_v2) ---
        r_cross = _extract_per_step_scalar(episode.observer_outputs, "cross_support", T)

        joint_dev_arr = _extract_per_step_field(episode.observer_outputs, "posture", "joint_deviation", T)
        joint_vel_arr = _extract_per_step_field(episode.observer_outputs, "posture", "joint_vel", T)
        torso_tilt_arr = _extract_per_step_field(episode.observer_outputs, "posture", "torso_tilt", T)
        foot_height_arr = _extract_per_step_field(episode.observer_outputs, "posture", "foot_height", T)

        if joint_dev_arr is None:
            joint_dev_arr = np.zeros(T, dtype=np.float32)
        if joint_vel_arr is None:
            joint_vel_arr = np.zeros(T, dtype=np.float32)
        if torso_tilt_arr is None:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)
        if foot_height_arr is None:
            foot_height_arr = np.zeros(T, dtype=np.float32)

        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint)

        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel)

        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)

        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot)

        return {
            "r_struggle": r_struggle,
            "r_cross": r_cross,
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }

    def prepare_segments(self, episode) -> Optional[List[Segment]]:
        """Split episode into struggle and stability segments.

        Each contiguous run of same-phase steps becomes a segment.
        Struggle segments only train r_struggle critic.
        Stability segments train all stability-phase critics (r_struggle + 5 others).

        Termination at phase boundaries:
          - Struggle → Stability: "truncated" (bootstrap — the robot is still
            alive, V(s_next) is meaningful for r_struggle)
          - Stability → Struggle: "terminated" for stability keys (the MDP
            for posture maintenance ends), "truncated" for r_struggle (it
            continues into the struggle segment)
        """
        T = episode.num_frames
        is_struggle, _ = self._extract_phase_info(episode)

        if T == 0:
            return []

        # Find contiguous phase runs
        segments: List[Segment] = []
        seg_start = 0
        current_is_struggle = bool(is_struggle[0])

        for t in range(1, T):
            if bool(is_struggle[t]) != current_is_struggle:
                # Phase boundary at t
                if current_is_struggle:
                    # Struggle segment: only r_struggle active
                    segments.append(Segment(
                        start=seg_start,
                        end=t,
                        weight=1.0,
                        key_weights={"r_struggle": 1.0},
                        termination="truncated",
                    ))
                else:
                    # Stability segment: all keys active
                    segments.append(Segment(
                        start=seg_start,
                        end=t,
                        weight=1.0,
                        key_weights=None,  # all keys
                        termination="terminated",
                    ))
                seg_start = t
                current_is_struggle = bool(is_struggle[t])

        # Last segment
        if seg_start < T:
            fell = "imbalance" in episode.termination_proposals
            if current_is_struggle:
                segments.append(Segment(
                    start=seg_start,
                    end=T,
                    weight=1.0,
                    key_weights={"r_struggle": 1.0},
                    termination="terminated" if fell else "truncated",
                ))
            else:
                segments.append(Segment(
                    start=seg_start,
                    end=T,
                    weight=1.0,
                    key_weights=None,
                    termination="terminated" if fell else "truncated",
                ))

        return segments

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        fell = "imbalance" in episode.termination_proposals
        is_struggle, _ = self._extract_phase_info(episode)
        struggle_steps = int(np.sum(is_struggle))
        total_steps = episode.num_frames
        return {
            "survived": 0.0 if fell else 1.0,
            "struggle_ratio": float(struggle_steps / max(total_steps, 1)),
            "struggle_steps": struggle_steps,
        }

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "survival_rate": round(self._survival_rate, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "survival_rate": self._survival_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._survival_rate = float(state.get("survival_rate", 0.0))


EXPERIMENT = BasicBalanceV2StageSegConfig()
