"""V2 dual-agent experiment: φ-scaled survonly r_fall + r_cross only, fixed actor weights.

Same as exp_basic_balance_v2_phi_dual_fixaw_survonly.py but removes r_joint, r_vel,
r_tilt, r_foot — only r_fall and r_cross remain.

This isolates whether the "原地平衡" (static balance) local optimum in fixaw_survonly
is caused by the posture shaping channels (r_joint/r_vel/r_tilt/r_foot) penalizing
movement at fixed full weight, or by the φ-scaled r_fall itself.

  - r_fall: 0.01 × φ(t) per step, no fall penalty, no timeout bonus
  - r_cross: alternating step reward/penalty
  - actor weights: (3.0, 1.0) — fixed, no φ scaling

Compare with:
  - exp_basic_balance_v2_phi_dual_fixaw_survonly.py (0.01×φ + r_cross + 4 posture channels)
  - exp_basic_balance_v2_dual_survonly.py (fixed 0.01 + r_cross + 4 posture channels)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.common.rollout import extract_per_step_scalar, extract_per_step_field

from .base import CombatExperimentPPOBase


class BasicBalanceV2PhiDualFixAWSurvOnlyCrossOnly(CombatExperimentPPOBase):

    name = "v2_basic_balance_v2_phi_dual_fixaw_survonly_crossonly"

    _channel_names = ("r_fall", "r_cross")
    _gamma = 0.99
    _gae_lambda = 0.95

    env_blueprint = "basic_balance_v2_phi_dual_env.yaml"
    agent_used = "both"

    episodes_per_update: int = 256 * 4

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Fixed actor weights (no φ scaling) ---
    _actor_weights: Tuple[float, ...] = (3.0, 1.0)

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

        # --- Truncate at agent's termination step ---
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

        # --- Extract φ per step (used in reward only, not in actor weights) ---
        phi_arr = extract_per_step_field(episode.observer_outputs, phi_key, "phi", T_full)
        if phi_arr is not None:
            phi_arr = phi_arr[:T]
        else:
            phi_arr = np.ones(T, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # --- r_fall: 0.01 × φ(t) per step only — no terminal signal ---
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- r_cross ---
        r_cross = extract_per_step_scalar(episode.observer_outputs, cross_key, T_full)
        if r_cross is not None:
            r_cross = r_cross[:T]
        else:
            r_cross = np.zeros(T, dtype=np.float32)

        # --- Fixed actor weights (no φ scaling) ---
        is_terminated = fell

        # --- Build channels ---
        all_rewards = {
            "r_fall": r_fall,
            "r_cross": r_cross.astype(np.float32),
        }

        channels: Dict[str, ChannelData] = {}
        for idx, key in enumerate(self._channel_names):
            aw = float(self._actor_weights[idx]) if idx < len(self._actor_weights) else 1.0
            channels[key] = ChannelData(
                reward=all_rewards[key].astype(np.float32),
                is_terminated=is_terminated,
                actor_weight=aw,
            )

        return [Trajectory(
            obs=np.asarray(obs_all[:T], dtype=np.float32),
            actions=np.asarray(acts_all[:T], dtype=np.float32),
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            mode=None,
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


EXPERIMENT_CLASS = BasicBalanceV2PhiDualFixAWSurvOnlyCrossOnly
