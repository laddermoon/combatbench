"""V2 experiment: 4-stage standup with dead-zone-free dense potential.

Dead-zone-free stage-dependent potential (4 stages):
  Stage 1 [0.00, 0.10):  φ = 0.10 × f_score        (rollover)
  Stage 2 [0.10, 0.20):  φ = 0.10 + 0.10 × contact  (establish support)
  Stage 3 [0.20, 0.30):  φ = 0.20 + 0.10 × d_score  (close hand-foot distance)
  Stage 4 [0.30, 1.00]:  φ = 0.30 + 0.70 × p4       (stand up on two feet)

Reward: r_t = (1-γ) × φ(t) = 0.01 × φ(t)  (dense mode, γ=0.99)

Dual-agent training: both robots get RandomFallenStatePlugin and
separate StandingBalance4StageRewarder observers, doubling data
collection per episode. Every step is trainable (no mixed policy, no
episode segmentation). The dense reward ensures sustained signal at
the standing plateau, avoiding the delta-mode signal collapse at
Stage 4 top.

Blueprint: standup_4stage_dense_v2_env.yaml
Observer:  StandingBalance4StageRewarder (provides "potential" per step)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.common.rollout import extract_per_step_field

from .base import CombatExperimentV2Base


class Standup(CombatExperimentV2Base):
    """4-stage standup training with dense potential reward.

    Single reward channel r_potential:
      r_t = (1-γ) × φ(t),  γ = 0.99  →  r_t = 0.01 × φ(t)

    Eval metric: max_potential across eval episodes.
    """

    name = "standup"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channel ---
    _channel_name = "r_potential"
    _gamma: float = 0.99
    _gae_lambda: float = 0.95

    # --- PPO tuning (aligned with original 4-stage ablation) ---
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    max_updates: int = 5000
    eval_interval: int = 5
    eval_episodes: int = 64

    # --- Video recording ---
    video_eval_interval: int = 2

    # --- Env blueprint ---
    env_blueprint = "standup_4stage_dense_v2_env.yaml"
    agent_used = "both"
    max_steps: int = 200

    _AGENT_OBS = (
        ("robot_a", "standing_balance_a"),
        ("robot_b", "standing_balance_b"),
    )

    # --- Stateful metrics ---
    _best_potential: float = -1.0
    _success_rate: float = 0.0

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return (
            RewardChannel(
                name=self._channel_name,
                gamma=self._gamma,
                gae_lambda=self._gae_lambda,
            ),
        )

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, obs_key in self._AGENT_OBS:
                trajs = self._build_agent_trajectory(episode, agent_id, obs_key)
                all_trajs.extend(trajs)
        return all_trajs

    def _build_agent_trajectory(self, episode, agent_id: str, obs_key: str) -> List[Trajectory]:
        T_full = episode.num_frames
        if T_full == 0:
            return []

        obs_all = episode.observations.get(agent_id)
        acts_all = episode.actions.get(agent_id)
        fin_obs = episode.final_observation.get(agent_id)

        if obs_all is None or acts_all is None or fin_obs is None:
            return []

        obs_all = np.asarray(obs_all, dtype=np.float32)
        acts_all = np.asarray(acts_all, dtype=np.float32)

        # --- Extract potential φ(t) from observer ---
        phi_arr = extract_per_step_field(
            episode.observer_outputs, obs_key, "potential", T_full,
        )
        if phi_arr is not None:
            phi_arr = phi_arr[:T_full]
        else:
            phi_arr = np.zeros(T_full, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # --- Dense reward: r_t = (1-γ) × φ(t) = 0.01 × φ(t) ---
        r_potential = ((1.0 - self._gamma) * phi_arr).astype(np.float32)

        # --- No early termination: episode runs to timeout ---
        # The robot can fall and stand up repeatedly; every step is trainable.
        is_terminated = False

        channels: Dict[str, ChannelData] = {
            self._channel_name: ChannelData(
                reward=r_potential,
                is_terminated=is_terminated,
                actor_weight=np.ones(T_full, dtype=np.float32),
            ),
        }

        return [Trajectory(
            obs=obs_all,
            actions=acts_all,
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            mode=None,
            log_prob=None,
        )]

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        final_pots = []
        max_stages = []
        max_h_torsos = []
        success_count = 0

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue

            for agent_id, obs_key in self._AGENT_OBS:
                phi = extract_per_step_field(
                    ep.observer_outputs, obs_key, "potential", T,
                )
                stages = extract_per_step_field(
                    ep.observer_outputs, obs_key, "stage", T,
                )
                h_torso = extract_per_step_field(
                    ep.observer_outputs, obs_key, "h_torso", T,
                )

                if phi is not None and len(phi) > 0:
                    mx = float(np.max(phi))
                    fn = float(phi[-1])
                else:
                    mx = 0.0
                    fn = 0.0
                max_pots.append(mx)
                final_pots.append(fn)

                if stages is not None and len(stages) > 0:
                    max_stages.append(float(np.max(stages)))
                else:
                    max_stages.append(0.0)

                if h_torso is not None and len(h_torso) > 0:
                    max_h_torsos.append(float(np.max(h_torso)))
                else:
                    max_h_torsos.append(0.0)

                if mx >= 0.9:
                    success_count += 1

        n = max(len(max_pots), 1)
        mean_max_pot = sum(max_pots) / n if max_pots else 0.0
        mean_final_pot = sum(final_pots) / n if final_pots else 0.0
        mean_max_stage = sum(max_stages) / n if max_stages else 0.0
        mean_max_h = sum(max_h_torsos) / n if max_h_torsos else 0.0
        success_rate = success_count / n

        self._success_rate = success_rate
        is_new_best = mean_max_pot > self._best_potential
        if is_new_best:
            self._best_potential = mean_max_pot

        return {
            "is_new_best": is_new_best,
            "info": {
                "max_pot": round(mean_max_pot, 3),
                "final_pot": round(mean_final_pot, 3),
                "max_stage": round(mean_max_stage, 2),
                "max_h": round(mean_max_h, 3),
                "success": round(success_rate, 3),
            },
        }

    def state(self) -> dict:
        return {
            "best_potential": self._best_potential,
            "success_rate": self._success_rate,
        }

    def load_state(self, state: dict) -> None:
        self._best_potential = float(state.get("best_potential", -1.0))
        self._success_rate = float(state.get("success_rate", 0.0))


EXPERIMENT_CLASS = Standup
