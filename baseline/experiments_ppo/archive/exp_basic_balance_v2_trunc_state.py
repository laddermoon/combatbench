"""V2 basic_balance_v2 with per-channel termination — only r_fall terminated.

Based on v2_basic_balance_v2. The only change: r_fall keeps the original
termination logic (terminated if fell, truncated if timeout), while all
state rewards (r_cross, r_joint, r_vel, r_tilt, r_foot) are always
truncated (is_terminated=False, bootstrap from V(s_end)).

Reward channels: r_fall, r_cross, r_joint, r_vel, r_tilt, r_foot
All channels use gamma=0.99, gae_lambda=0.95.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_scalar, extract_per_step_field

from .base import CombatExperimentPPOBase


class BasicBalanceV2TruncState(CombatExperimentPPOBase):

    name = "v2_basic_balance_v2_trunc_state"

    # --- Reward channels ---
    _channel_names = ("r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    _gamma = 0.99
    _gae_lambda = 0.95

    # --- Env / rollout config ---
    env_blueprint = "basic_balance_v2_env.yaml"

    # --- Reward constants ---
    per_step_survival_reward: float = 0.01
    terminal_fall_penalty: float = 1.0

    # --- Actor weights (matching V1 initial_weights / next_weights) ---
    _actor_weights: Tuple[float, ...] = (3.0, 1.0, 0.2, 0.2, 0.2, 0.2)

    # --- Stateful scheduler ---
    _survival_rate: float = 0.0
    _best_survived: float = -1.0

    # ------------------------------------------------------------------
    # ExperimentPPO abstract methods
    # ------------------------------------------------------------------

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(name=k, gamma=self._gamma, gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    def build_trajectories(self, episodes) -> List[Trajectory]:
        """Convert episodes into trajectories — one per episode, all 6 channels.

        Only r_fall uses the fell-based termination flag. All state rewards
        (r_cross, r_joint, r_vel, r_tilt, r_foot) are always truncated.
        """
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            trajs = self._build_one(episode)
            all_trajs.extend(trajs)
        return all_trajs

    def _build_one(self, episode) -> List[Trajectory]:
        T = episode.num_frames
        if T == 0:
            return []

        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        obs_all = episode.observations.get(ep_target)
        acts_all = episode.actions.get(ep_target)
        fin_obs = episode.final_observation.get(ep_target)

        if obs_all is None or acts_all is None or fin_obs is None:
            return []

        fell = all(
            r.startswith("imbalance")
            for r in episode.agent_termination_reason.values()
        )

        # --- r_fall: per-step survival bonus + terminal signal ---
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = self.terminal_fall_penalty
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        # --- r_cross ---
        r_cross = extract_per_step_scalar(episode.observer_outputs, "cross_support", T)

        # --- posture-based channels ---
        joint_dev_arr = extract_per_step_field(episode.observer_outputs, "posture", "joint_deviation", T)
        joint_vel_arr = extract_per_step_field(episode.observer_outputs, "posture", "joint_vel", T)
        torso_tilt_arr = extract_per_step_field(episode.observer_outputs, "posture", "torso_tilt", T)
        foot_height_arr = extract_per_step_field(episode.observer_outputs, "posture", "foot_height", T)

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

        # --- Per-channel termination ---
        # r_fall: terminated if fell (V=0), truncated if timeout (bootstrap)
        # state rewards: always truncated (bootstrap from V(s_end))
        is_terminated_fall = fell

        # --- Build channels ---
        all_rewards = {
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_joint": r_joint.astype(np.float32),
            "r_vel": r_vel.astype(np.float32),
            "r_tilt": r_tilt.astype(np.float32),
            "r_foot": r_foot.astype(np.float32),
        }

        channels: Dict[str, ChannelData] = {}
        for idx, key in enumerate(self._channel_names):
            aw = float(self._actor_weights[idx]) if idx < len(self._actor_weights) else 1.0
            is_term = is_terminated_fall if key == "r_fall" else False
            channels[key] = ChannelData(
                reward=all_rewards[key].astype(np.float32),
                is_terminated=is_term,
                actor_weight=aw,
            )

        return [Trajectory(
            obs=np.asarray(obs_all, dtype=np.float32),
            actions=np.asarray(acts_all, dtype=np.float32),
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            mode=None,
        )]

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        """Process eval episodes — matches V1 compute_episode_metrics + compare_eval + next_weights."""
        survived_count = 0
        for ep in episodes:
            fell = all(
                r.startswith("imbalance")
                for r in ep.agent_termination_reason.values()
            )
            if not fell:
                survived_count += 1

        survival_rate = float(survived_count / max(len(episodes), 1))
        self._survival_rate = survival_rate

        survived_metric = float(survived_count)

        is_new_best = survived_metric > self._best_survived
        if is_new_best:
            self._best_survived = survived_metric

        # V1 next_weights always returns the same weights
        self._actor_weights = (3.0, 1.0, 0.2, 0.2, 0.2, 0.2)

        return {
            "is_new_best": is_new_best,
            "info": {
                "survived": survived_metric,
                "survival_rate": round(survival_rate, 3),
            },
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

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


EXPERIMENT_CLASS = BasicBalanceV2TruncState
