"""SAC standup experiment — 4-stage dense potential.

Semantically mirrors ``baseline/experiments_ppo/exp_standup.py`` while using
``sac_transition_v1`` and the SAC-owned collection/data contracts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.sac.experiment import SACRewardChannel
from baseline.framework.sac.observer_utils import extract_per_step_field
from baseline.framework.sac.transition import (
    SACTransitionSlice,
    build_agent_transition_slice,
)

from .base import CombatExperimentSACBase


class SacStandup(CombatExperimentSACBase):
    name = "sac_standup"

    env_blueprint = "standup_4stage_dense_v2_env.yaml"
    agent_used = "both"

    obs_dim = 96
    action_dim = 21
    episodes_per_update = 64
    max_env_steps = 2_000_000
    eval_interval = 20_000
    eval_episodes = 16
    max_steps = 200

    warmup_steps = 10_000
    utd_ratio = 0.5
    batch_size = 256
    replay_buffer_size = 500_000
    init_alpha = 0.2
    target_entropy = -21.0
    reward_scale = 1.0

    _gamma = 0.99
    _channel_name = "r_potential"
    _AGENT_OBS = (
        ("robot_a", "standing_balance_a"),
        ("robot_b", "standing_balance_b"),
    )
    _best_potential: float = -1.0
    _success_rate: float = 0.0
    _ep_final_pots: List[float]

    def reward_channels(self) -> Tuple[SACRewardChannel, ...]:
        return (
            SACRewardChannel(
                name=self._channel_name,
                gamma=self._gamma,
                n_step=1,
                n_critics=2,
                trunk_group="shared",
            ),
        )

    def build_slices(self, episodes: List[Any]) -> List[SACTransitionSlice]:
        self._ep_final_pots = []
        all_slices: List[SACTransitionSlice] = []
        for episode in episodes:
            for agent_id, obs_key in self._AGENT_OBS:
                T_full = episode.num_frames
                if T_full == 0:
                    continue
                T = episode.agent_frame_boundary.get(agent_id, T_full)
                if T == 0:
                    continue
                potential = extract_per_step_field(
                    episode.observer_outputs, obs_key, "potential", T_full,
                )
                if potential is None:
                    raise KeyError(
                        f"Missing required observer field {obs_key}.potential"
                    )
                potential = np.clip(
                    np.asarray(potential, dtype=np.float32), 0.0, 1.0,
                )
                self._ep_final_pots.append(float(potential[T - 1]))
                reward = ((1.0 - self._gamma) * potential[:T]).astype(np.float32)
                sl = build_agent_transition_slice(
                    episode,
                    agent_id,
                    channel_names=(self._channel_name,),
                    rewards={self._channel_name: reward},
                    actor_gate={
                        self._channel_name: np.ones(T, dtype=np.float32),
                    },
                    task_facts={"potential": potential[:T]},
                    reward_features={
                        "potential": potential[:T],
                    },
                    versions={
                        "reward_semantics": "standup_4stage_dense_v2",
                        "objective_mode": "shannon",
                        "regularizer_mode": "entropy",
                        "policy_arch": "s01_shared_sigma",
                    },
                )
                if sl is not None:
                    all_slices.append(sl)
        return all_slices

    def on_eval(self, episodes: List[Any], env_step: int) -> Dict[str, Any]:
        max_pots: List[float] = []
        final_pots: List[float] = []
        max_stages: List[float] = []
        success_count = 0
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            for _agent_id, obs_key in self._AGENT_OBS:
                phi = extract_per_step_field(
                    ep.observer_outputs, obs_key, "potential", ep.num_frames,
                )
                stages = extract_per_step_field(
                    ep.observer_outputs, obs_key, "stage", ep.num_frames,
                )
                if phi is None or len(phi) == 0:
                    raise KeyError(
                        f"Missing required eval observer field {obs_key}.potential"
                    )
                mx = float(np.max(phi))
                max_pots.append(mx)
                final_pots.append(float(phi[-1]))
                max_stages.append(float(np.max(stages)) if stages is not None else 0.0)
                if mx >= 0.9:
                    success_count += 1

        n = max(len(max_pots), 1)
        success_rate = float(success_count / n)
        mean_max_pot = float(sum(max_pots) / n) if max_pots else 0.0
        mean_final_pot = float(sum(final_pots) / n) if final_pots else 0.0
        mean_max_stage = float(sum(max_stages) / n) if max_stages else 0.0
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


EXPERIMENT_CLASS = SacStandup
