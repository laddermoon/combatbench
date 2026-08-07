"""V2 dual-perspective staged balance — phase-based trajectories per agent.

Each episode produces multiple trajectories per agent: one per contiguous
phase run (struggle or stability). Both agents are observed and terminated
independently via DualImbalanceTerminationPlugin.

Reward channels (8 total):
  - r_struggle_strug: terminal-only rewards during struggle phase
  - r_struggle_stab: terminal-only rewards during stability phase
  - r_height: dense height shaping (struggle phase only)
  - r_cross, r_joint, r_vel, r_tilt, r_foot: dense state rewards (stability phase only)

Key design:
  - r_struggle split into r_struggle_strug / r_struggle_stab to use
    separate critics per phase.
  - State rewards (r_height, r_cross, etc.) are always truncated
    (is_terminated=False) — they have no terminal value.
  - Only struggle channels can be terminated (they carry terminal
    bonuses/penalties at phase transitions and fall).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.ppo_trainer import _extract_per_step_scalar, _extract_per_step_field

from .base import CombatExperimentV2Base


class BasicBalanceV2StageSegDual(CombatExperimentV2Base):

    name = "v2_basic_balance_v2_stage_seg_dual"

    _channel_names = (
        "r_struggle_strug", "r_struggle_stab",
        "r_height", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot",
    )
    _gamma = 0.99
    _gae_lambda = 0.95

    env_blueprint = "basic_balance_v2_stage_seg_dual_env.yaml"
    agent_used = "both"

    episodes_per_update: int = 256 * 4

    _actor_weights: Tuple[float, ...] = (3.0, 3.0, 0.3, 1.0, 0.2, 0.2, 0.2, 0.2)

    struggle_recover_bonus: float = 1.0
    struggle_fall_penalty: float = -1.0
    stability_to_struggle_penalty: float = -1.0

    _survival_rate: float = 0.0
    _best_survived: float = -1.0

    _AGENT_IDS = ("robot_a", "robot_b")

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(name=k, gamma=self._gamma, gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    def _extract_phase_info(self, episode, phase_key: str, T: int) -> np.ndarray:
        """Extract per-step is_struggle bool array from PhaseObserver output.

        Returns (T,) bool array.
        """
        phase_node = episode.observer_outputs.get(phase_key)
        if phase_node is None:
            return np.zeros(T, dtype=bool)

        if isinstance(phase_node, dict):
            phase_arr = phase_node.get("is_struggle")
            if phase_arr is not None:
                is_struggle = np.asarray(phase_arr, dtype=bool).reshape(-1)
                if is_struggle.shape[0] >= T:
                    return is_struggle[:T]
            return np.zeros(T, dtype=bool)

        # Fallback: list of dicts
        is_struggle = np.zeros(T, dtype=bool)
        try:
            raw = np.asarray(phase_node, dtype=object).reshape(-1)
            for t in range(min(len(raw), T)):
                val = raw[t]
                if isinstance(val, dict):
                    is_struggle[t] = val.get("is_struggle", False)
                elif isinstance(val, str):
                    is_struggle[t] = val == "struggle"
        except Exception:
            pass
        return is_struggle

    def _phase_runs(
        self, is_struggle: np.ndarray, T: int,
    ) -> List[Tuple[int, int, bool]]:
        """Decompose [0, T) into contiguous same-phase runs.

        Returns list of (start, end, is_struggle) with end exclusive.
        """
        if T == 0:
            return []
        runs: List[Tuple[int, int, bool]] = []
        seg_start = 0
        current = bool(is_struggle[0])
        for t in range(1, T):
            if bool(is_struggle[t]) != current:
                runs.append((seg_start, t, current))
                seg_start = t
                current = bool(is_struggle[t])
        runs.append((seg_start, T, current))
        return runs

    # ------------------------------------------------------------------
    # Trajectory building
    # ------------------------------------------------------------------

    def _count_phase_frames(
        self,
        episode,
        agent_id: str,
        phase_key: str,
    ) -> Tuple[int, int]:
        """Count struggle and stability frames for one agent (truncated)."""
        T_full = episode.num_frames
        if T_full == 0:
            return (0, 0)

        records = episode.agent_termination_proposal_records.get(agent_id, ())
        if records:
            first_reason, term_step = records[0]
            fell = first_reason.startswith("imbalance")
            T = term_step if fell else T_full
        else:
            T = T_full

        if T == 0:
            return (0, 0)

        is_struggle = self._extract_phase_info(episode, phase_key, T)
        n_struggle = int(is_struggle.sum())
        n_stability = T - n_struggle
        return (n_struggle, n_stability)

    def _build_agent_trajectories(
        self,
        episode,
        agent_id: str,
        cross_key: str,
        posture_key: str,
        phase_key: str,
        stability_aw_scale: float = 1.0,
    ) -> List[Trajectory]:
        """Build one Trajectory per phase run for a single agent.

        ``stability_aw_scale`` multiplies the actor_weight of all active
        channels on stability-phase trajectories.
        """
        T_full = episode.num_frames
        if T_full == 0:
            return []

        # --- Truncate at agent's termination step ---
        records = episode.agent_termination_proposal_records.get(agent_id, ())
        if records:
            first_reason, term_step = records[0]
            fell = first_reason.startswith("imbalance")
            if fell:
                T = term_step
            else:
                T = T_full
        else:
            fell = False
            T = T_full

        if T == 0:
            return []

        obs_all = episode.observations.get(agent_id)
        acts_all = episode.actions.get(agent_id)
        fin_obs = episode.final_observation.get(agent_id)

        if obs_all is None or acts_all is None or fin_obs is None:
            return []

        obs_all = np.asarray(obs_all, dtype=np.float32)
        acts_all = np.asarray(acts_all, dtype=np.float32)

        # --- Phase runs on truncated range ---
        is_struggle = self._extract_phase_info(episode, phase_key, T)
        runs = self._phase_runs(is_struggle, T)

        # --- Compute full reward arrays (length T) ---
        r_struggle_strug = np.zeros(T, dtype=np.float32)
        r_struggle_stab = np.zeros(T, dtype=np.float32)

        for idx, (start, end, is_str) in enumerate(runs):
            is_last = (idx == len(runs) - 1)
            if is_str:
                # Struggle phase run
                if not is_last:
                    # Transition: struggle -> stability (recovered)
                    r_struggle_strug[end - 1] += self.struggle_recover_bonus
                elif fell:
                    # Last run and fell
                    r_struggle_strug[end - 1] += self.struggle_fall_penalty
            else:
                # Stability phase run
                if not is_last:
                    # Transition: stability -> struggle (degraded)
                    r_struggle_stab[end - 1] += self.stability_to_struggle_penalty
                elif fell:
                    # Last run and fell
                    r_struggle_stab[end - 1] += self.struggle_fall_penalty

        # r_height from PhaseObserver height field
        height_arr = np.zeros(T, dtype=np.float32)
        phase_node = episode.observer_outputs.get(phase_key)
        if phase_node is not None and isinstance(phase_node, dict):
            h_raw = phase_node.get("height")
            if h_raw is not None:
                h_arr = np.asarray(h_raw, dtype=np.float32).reshape(-1)
                if h_arr.shape[0] >= T:
                    height_arr = h_arr[:T]
        r_height = (height_arr * 0.01).astype(np.float32)

        # State rewards from observer outputs
        r_cross = _extract_per_step_scalar(episode.observer_outputs, cross_key, T_full)
        if r_cross is not None:
            r_cross = r_cross[:T]
        else:
            r_cross = np.zeros(T, dtype=np.float32)

        joint_dev_arr = _extract_per_step_field(episode.observer_outputs, posture_key, "joint_deviation", T_full)
        joint_vel_arr = _extract_per_step_field(episode.observer_outputs, posture_key, "joint_vel", T_full)
        torso_tilt_arr = _extract_per_step_field(episode.observer_outputs, posture_key, "torso_tilt", T_full)
        foot_height_arr = _extract_per_step_field(episode.observer_outputs, posture_key, "foot_height", T_full)

        if joint_dev_arr is not None:
            joint_dev_arr = joint_dev_arr[:T]
        else:
            joint_dev_arr = np.zeros(T, dtype=np.float32)
        if joint_vel_arr is not None:
            joint_vel_arr = joint_vel_arr[:T]
        else:
            joint_vel_arr = np.zeros(T, dtype=np.float32)
        if torso_tilt_arr is not None:
            torso_tilt_arr = torso_tilt_arr[:T]
        else:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)
        if foot_height_arr is not None:
            foot_height_arr = foot_height_arr[:T]
        else:
            foot_height_arr = np.zeros(T, dtype=np.float32)

        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint).astype(np.float32)

        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel).astype(np.float32)

        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt).astype(np.float32)

        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot).astype(np.float32)

        all_rewards = {
            "r_struggle_strug": r_struggle_strug,
            "r_struggle_stab": r_struggle_stab,
            "r_height": r_height,
            "r_cross": r_cross.astype(np.float32),
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }

        # --- Build one Trajectory per phase run ---
        trajectories: List[Trajectory] = []
        for idx, (start, end, is_str) in enumerate(runs):
            T_run = end - start
            if T_run == 0:
                continue

            is_last = (idx == len(runs) - 1)

            # last_obs: next step's obs if mid-episode, else final obs
            if is_last:
                last_obs = np.asarray(fin_obs, dtype=np.float32)
            else:
                last_obs = np.asarray(obs_all[end], dtype=np.float32)

            # Active channels per phase
            if is_str:
                active_keys = {"r_struggle_strug", "r_height"}
            else:
                active_keys = {"r_struggle_stab", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot"}

            # Per-channel is_terminated:
            #   - struggle channels: terminated if mid-episode or fell (terminal reward present)
            #     truncated if last run + timeout (no terminal reward)
            #   - state rewards: always truncated (no terminal value)
            struggle_terminated = (not is_last) or fell

            # Include ALL channels in every trajectory so the buffer can
            # infer the full reward_keys set from any trajectory.
            # Inactive channels get reward=0, is_terminated=True, actor_weight=0.
            channels: Dict[str, ChannelData] = {}
            for ch_idx, key in enumerate(self._channel_names):
                aw = float(self._actor_weights[ch_idx]) if ch_idx < len(self._actor_weights) else 1.0

                # Scale stability-phase active channels
                if not is_str and key in active_keys:
                    aw *= stability_aw_scale

                if key not in active_keys:
                    channels[key] = ChannelData(
                        reward=np.zeros(T_run, dtype=np.float32),
                        is_terminated=True,
                        actor_weight=0.0,
                    )
                    continue

                if key in ("r_struggle_strug", "r_struggle_stab"):
                    is_term = struggle_terminated
                else:
                    is_term = False  # state rewards always truncated

                channels[key] = ChannelData(
                    reward=all_rewards[key][start:end].astype(np.float32),
                    is_terminated=is_term,
                    actor_weight=aw,
                )

            trajectories.append(Trajectory(
                obs=np.asarray(obs_all[start:end], dtype=np.float32),
                actions=np.asarray(acts_all[start:end], dtype=np.float32),
                last_obs=last_obs,
                channels=channels,
                importance=1.0,
                mode=None,
                log_prob=None,
            ))

        return trajectories

    def build_trajectories(self, episodes) -> List[Trajectory]:
        """Build trajectories — multiple per agent, one per phase run.

        Two-pass: first count struggle/stability frames across all episodes
        to compute a stability actor_weight scale, then build trajectories
        with the adjusted weights so neither phase dominates the gradient.
        """
        agent_specs = [
            ("robot_a", "cross_support_a", "posture_a", "phase_a"),
            ("robot_b", "cross_support_b", "posture_b", "phase_b"),
        ]

        # --- Pass 1: count phase frames ---
        n_struggle = 0
        n_stability = 0
        for episode in episodes:
            for agent_id, _, _, phase_key in agent_specs:
                ns, nb = self._count_phase_frames(episode, agent_id, phase_key)
                n_struggle += ns
                n_stability += nb

        # --- Compute stability aw scale (total-weight balance, one-sided) ---
        # Per-frame base total weights:
        #   struggle: r_struggle_strug(3.0) + r_height(0.3) = 3.3
        #   stability: r_struggle_stab(3.0) + r_cross(1.0) + 4×0.2 = 4.8
        W_STRUGGLE = 3.0 + 0.3
        W_STABILITY = 3.0 + 1.0 + 0.2 * 4
        s_struggle = W_STRUGGLE * n_struggle
        s_stability = W_STABILITY * n_stability
        if s_stability < s_struggle and s_stability > 0:
            stability_aw_scale = s_struggle / s_stability
        else:
            stability_aw_scale = 1.0

        # --- Pass 2: build trajectories with scaled weights ---
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, cross_key, posture_key, phase_key in agent_specs:
                agent_trajs = self._build_agent_trajectories(
                    episode, agent_id, cross_key, posture_key, phase_key,
                    stability_aw_scale=stability_aw_scale,
                )
                all_trajs.extend(agent_trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        survived_count = 0
        total_agents = 0
        for ep in episodes:
            for aid in self._AGENT_IDS:
                total_agents += 1
                term_reason = ep.agent_termination_reason.get(aid, "")
                if not term_reason.startswith("imbalance"):
                    survived_count += 1

        survival_rate = float(survived_count / max(total_agents, 1))
        self._survival_rate = survival_rate

        survived_metric = float(survived_count)
        is_new_best = survived_metric > self._best_survived
        if is_new_best:
            self._best_survived = survived_metric

        self._actor_weights = (3.0, 3.0, 0.3, 1.0, 0.2, 0.2, 0.2, 0.2)

        return {
            "is_new_best": is_new_best,
            "info": {
                "survived": survived_metric,
                "survival_rate": round(survival_rate, 3),
            },
        }

    def state(self) -> dict:
        return {
            "survival_rate": self._survival_rate,
            "best_survived": self._best_survived,
            "actor_weights": list(self._actor_weights),
        }

    def load_state(self, state: dict) -> None:
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._best_survived = float(state.get("best_survived", -1.0))
        aw = state.get("actor_weights")
        if aw is not None:
            self._actor_weights = tuple(float(w) for w in aw)


EXPERIMENT = BasicBalanceV2StageSegDual()
