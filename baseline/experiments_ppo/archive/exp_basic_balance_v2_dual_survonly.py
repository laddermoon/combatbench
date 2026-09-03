"""V2 dual-perspective basic_balance_v2 — variant 2: survival reward only.

r_fall = 0.01/step, NO fall penalty, NO timeout bonus.
Everything else identical to exp_basic_balance_v2_dual.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_scalar, extract_per_step_field

from .base import CombatExperimentPPOBase


class BasicBalanceV2DualSurvOnly(CombatExperimentPPOBase):

    name = "v2_basic_balance_v2_dual_survonly"

    _channel_names = ("r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    _gamma = 0.99
    _gae_lambda = 0.95

    env_blueprint = "basic_balance_v2_dual_env.yaml"
    agent_used = "both"

    per_step_survival_reward: float = 0.01

    episodes_per_update: int = 256 * 4

    _actor_weights: Tuple[float, ...] = (3.0, 1.0, 0.2, 0.2, 0.2, 0.2)

    _survival_rate: float = 0.0
    _best_survived: float = -1.0

    _AGENT_IDS = ("robot_a", "robot_b")

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(name=k, gamma=self._gamma, gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    def _build_single_trajectory(
        self,
        episode,
        agent_id: str,
        cross_key: str,
        posture_key: str,
    ) -> Trajectory:
        T_full = episode.num_frames
        if T_full == 0:
            return None

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
            return None

        obs_all = episode.observations.get(agent_id)
        acts_all = episode.actions.get(agent_id)
        fin_obs = episode.final_observation.get(agent_id)

        if obs_all is None or acts_all is None or fin_obs is None:
            return None

        obs_all = np.asarray(obs_all[:T], dtype=np.float32)
        acts_all = np.asarray(acts_all[:T], dtype=np.float32)

        # --- r_fall: 0.01/step only, NO terminal signals ---
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)

        r_cross = extract_per_step_scalar(episode.observer_outputs, cross_key, T_full)[:T]

        joint_dev_arr = extract_per_step_field(episode.observer_outputs, posture_key, "joint_deviation", T_full)
        joint_vel_arr = extract_per_step_field(episode.observer_outputs, posture_key, "joint_vel", T_full)
        torso_tilt_arr = extract_per_step_field(episode.observer_outputs, posture_key, "torso_tilt", T_full)
        foot_height_arr = extract_per_step_field(episode.observer_outputs, posture_key, "foot_height", T_full)

        if joint_dev_arr is not None:
            joint_dev_arr = joint_dev_arr[:T]
        if joint_vel_arr is not None:
            joint_vel_arr = joint_vel_arr[:T]
        if torso_tilt_arr is not None:
            torso_tilt_arr = torso_tilt_arr[:T]
        if foot_height_arr is not None:
            foot_height_arr = foot_height_arr[:T]

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

        is_terminated = fell

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
            channels[key] = ChannelData(
                reward=all_rewards[key].astype(np.float32),
                is_terminated=is_terminated,
                actor_weight=aw,
            )

        return Trajectory(
            obs=obs_all,
            actions=acts_all,
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            mode=None,
        )

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, cross_key, posture_key in [
                ("robot_a", "cross_support_a", "posture_a"),
                ("robot_b", "cross_support_b", "posture_b"),
            ]:
                traj = self._build_single_trajectory(episode, agent_id, cross_key, posture_key)
                if traj is not None:
                    all_trajs.append(traj)
        return all_trajs

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
        }

    def load_state(self, state: dict) -> None:
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._best_survived = float(state.get("best_survived", -1.0))


EXPERIMENT_CLASS = BasicBalanceV2DualSurvOnly
