"""Minimal PPO experiment — the smallest correct ExperimentPPO subclass.

This file is the **canonical minimal example** referenced by GUIDE.md §4.
It is a real, auto-discovered experiment (``--experiment minimal``) that
demonstrates the minimum viable implementation of every abstract method.

Unlike the combat experiments (standup, standup_step_v3), this one:
- Uses a single reward channel with a simple dense potential.
- Uses the same standup environment blueprint (so it actually runs).
- Has conservative, well-documented hyperparameters.

Every code block in GUIDE.md §4 should be traceable to this file.  If
you change this file, update GUIDE.md; if you change GUIDE.md, update
this file.  The test ``test_minimal_example_imports`` enforces that the
file is importable and the class is registered.

P1-5: This file exists so that GUIDE.md's examples are tested code,
not untested documentation that silently rots.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field

from .base import CombatExperimentPPOBase


class MinimalExperiment(CombatExperimentPPOBase):
    """Minimal single-channel PPO experiment on the standup environment.

    This is a teaching example — it trains on the same 4-stage standup
    task as ``standup``, but with fewer episodes and a simpler reward
    structure.  It is not meant to produce a good policy; it is meant
    to be the shortest correct example of an ``ExperimentPPO`` subclass.

    Reward: single channel ``r_potential`` = (1-γ) × φ(t).
    Eval metric: max potential across eval episodes.
    """

    name = "minimal"
    actor_blueprint = "init_policy_truncated_normal.yaml"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channel ---
    _channel_name = "r_potential"
    _gamma: float = 0.99
    _gae_lambda: float = 0.95

    # --- Exploration ---
    # 0.0 = neutral.  See GUIDE.md §3.4 and DESIGN_unified_exploration_control.md
    # for the distinction between explore_factor (rollout) and
    # uncertainty_floor (training).
    explore_factor: float = 0.0

    # --- PPO tuning ---
    # These values are for demonstration.  Real experiments should tune
    # these for their specific task.
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096

    # --- Rollout schedule (small for fast smoke tests) ---
    episodes_per_update: int = 16
    max_updates: int = 100
    eval_interval: int = 10
    eval_episodes: int = 8

    # --- Video recording ---
    video_eval_interval: int = 0  # disable video in minimal example

    # --- Env blueprint (same as standup so it actually runs) ---
    env_blueprint = "standup_4stage_dense_v2_env.yaml"
    agent_used = "both"
    max_steps: int = 200

    _AGENT_OBS = (
        ("robot_a", "standing_balance_a"),
        ("robot_b", "standing_balance_b"),
    )

    # --- Stateful metrics ---
    _best_potential: float = -1.0

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

    def _build_agent_trajectory(
        self, episode, agent_id: str, obs_key: str,
    ) -> List[Trajectory]:
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

        # Extract potential φ(t) from observer
        phi_arr = extract_per_step_field(
            episode.observer_outputs, obs_key, "potential", T_full,
        )
        if phi_arr is not None:
            phi_arr = phi_arr[:T_full]
        else:
            phi_arr = np.zeros(T_full, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # Dense reward: r_t = (1-γ) × φ(t)
        r_potential = ((1.0 - self._gamma) * phi_arr).astype(np.float32)

        channels: Dict[str, ChannelData] = {
            self._channel_name: ChannelData(
                reward=r_potential,
                is_terminated=False,
                actor_weight=np.ones(T_full, dtype=np.float32),
            ),
        }

        return [Trajectory(
            obs=obs_all,
            actions=acts_all,
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            explore_factor=self.extract_explore_factor(episode, agent_id, T_full),
        )]

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for agent_id, obs_key in self._AGENT_OBS:
                phi = extract_per_step_field(
                    ep.observer_outputs, obs_key, "potential", T,
                )
                if phi is not None and len(phi) > 0:
                    max_pots.append(float(np.max(phi)))

        mean_max_pot = sum(max_pots) / len(max_pots) if max_pots else 0.0
        is_new_best = mean_max_pot > self._best_potential
        if is_new_best:
            self._best_potential = mean_max_pot

        return {
            "is_new_best": is_new_best,
            "info": {
                "max_pot": round(mean_max_pot, 3),
            },
        }

    def state(self) -> dict:
        return {"best_potential": self._best_potential}

    def load_state(self, state: dict) -> None:
        self._best_potential = float(state.get("best_potential", -1.0))


EXPERIMENT_CLASS = MinimalExperiment
