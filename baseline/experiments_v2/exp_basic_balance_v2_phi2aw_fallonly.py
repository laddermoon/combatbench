"""V2 experiment: NO survive reward + fall penalty + NO timeout + φ² shaping.

Exp I in the ablation plan: (Survive无, Fall有, TB无, φ²).
r_fall = 0/step + (-1 on fall), NO survival reward, NO timeout bonus.
Shaping channels scaled by φ².
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.ppo_trainer import _extract_per_step_scalar, _extract_per_step_field

from .base import CombatExperimentV2Base


class BasicBalanceV2Phi2AWFallOnly(CombatExperimentV2Base):

    name = "v2_basic_balance_v2_phi2aw_fallonly"

    _channel_names = ("r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    _gamma = 0.99
    _gae_lambda = 0.95

    env_blueprint = "basic_balance_v2_phi_dual_env.yaml"
    agent_used = "both"

    episodes_per_update: int = 256 * 4

    # --- Reward constants ---
    terminal_fall_penalty: float = 1.0

    # --- Actor weight base coefficients ---
    _fall_aw_base: float = 3.0
    _shaping_aw_bases: Tuple[float, ...] = (1.0, 0.2, 0.2, 0.2, 0.2)

    _AGENT_IDS = ("robot_a", "robot_b")

    _survival_rate: float = 0.0
    _best_survived: float = -1.0

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(name=k, gamma=self._gamma, gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        cross_key: str,
        posture_key: str,
        phi_key: str,
    ) -> List[Trajectory]:
        T_full = episode.num_frames
        if T_full == 0:
            return []

        records = episode.agent_termination_proposal_records.get(agent_id, ())
        if records:
            first_reason, term_step = records[0]
            fell = first_reason.startswith("imbalance")
            T = term_step if fell else T_full
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

        # --- Extract φ per step ---
        phi_arr = _extract_per_step_field(episode.observer_outputs, phi_key, "phi", T_full)
        if phi_arr is not None:
            phi_arr = phi_arr[:T]
        else:
            phi_arr = np.ones(T, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # --- r_fall: 0/step + (-1 on fall), NO survival, NO timeout ---
        r_fall = np.zeros(T, dtype=np.float32)
        if fell:
            r_fall[-1] = -self.terminal_fall_penalty

        # --- r_cross ---
        r_cross = _extract_per_step_scalar(episode.observer_outputs, cross_key, T_full)
        if r_cross is not None:
            r_cross = r_cross[:T]
        else:
            r_cross = np.zeros(T, dtype=np.float32)

        # --- posture-based channels ---
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

        # --- Per-step actor weights ---
        aw_fall = self._fall_aw_base
        phi_sq = (phi_arr * phi_arr).astype(np.float32)
        aw_cross = (self._shaping_aw_bases[0] * phi_sq).astype(np.float32)
        aw_joint = (self._shaping_aw_bases[1] * phi_sq).astype(np.float32)
        aw_vel = (self._shaping_aw_bases[2] * phi_sq).astype(np.float32)
        aw_tilt = (self._shaping_aw_bases[3] * phi_sq).astype(np.float32)
        aw_foot = (self._shaping_aw_bases[4] * phi_sq).astype(np.float32)

        is_terminated = fell

        all_rewards = {
            "r_fall": r_fall,
            "r_cross": r_cross.astype(np.float32),
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }
        all_aws = {
            "r_fall": aw_fall,
            "r_cross": aw_cross,
            "r_joint": aw_joint,
            "r_vel": aw_vel,
            "r_tilt": aw_tilt,
            "r_foot": aw_foot,
        }

        channels: Dict[str, ChannelData] = {}
        for key in self._channel_names:
            channels[key] = ChannelData(
                reward=all_rewards[key].astype(np.float32),
                is_terminated=is_terminated,
                actor_weight=all_aws[key],
            )

        return [Trajectory(
            obs=np.asarray(obs_all[:T], dtype=np.float32),
            actions=np.asarray(acts_all[:T], dtype=np.float32),
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            mode=None,
            log_prob=None,
        )]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        agent_specs = [
            ("robot_a", "cross_support_a", "posture_a", "height_phi_a"),
            ("robot_b", "cross_support_b", "posture_b", "height_phi_b"),
        ]

        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, cross_key, posture_key, phi_key in agent_specs:
                agent_trajs = self._build_agent_trajectory(
                    episode, agent_id, cross_key, posture_key, phi_key,
                )
                all_trajs.extend(agent_trajs)
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


EXPERIMENT = BasicBalanceV2Phi2AWFallOnly()
