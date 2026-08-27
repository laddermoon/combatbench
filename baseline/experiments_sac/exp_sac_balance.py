"""SAC balance experiment — 2-channel validation.

Mirrors the PPO ``basic_balance`` experiment but uses the SAC
interface (TrajectorySlice with per-step dones, per-channel n-step).

Two reward channels:
  - r_fall: 0.01 × φ(t) per step (survival reward, dense)
  - r_cross: alternating step reward/penalty (balance signal)

Actor weights:
  - r_fall: fixed 3.0
  - r_cross: 1.0 × φ² (gated by height proxy)

This is the simplest SAC experiment, designed to validate:
  - Action-gradient normalization produces correct gradient shares.
  - UTD ratio can be pushed without divergence.
  - Per-channel n-step TD works correctly.
  - The full training loop runs end-to-end.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from baseline.framework.sac.experiment import (
    DataSource,
    SACParams,
    SACRewardChannel,
    TrajectorySlice,
)
from baseline.common.rollout import extract_per_step_field, extract_per_step_scalar

from .base import CombatExperimentSACBase


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class SacBalance(CombatExperimentSACBase):

    name = "sac_balance"

    _channel_names = ("r_fall", "r_cross")
    _gamma = 0.99

    env_blueprint = "basic_balance_v2_phi_dual_env.yaml"
    agent_used = "both"

    # Match PPO's parallelism (192 CPUs available, PPO uses 96)
    rollout_workers: int = 96

    # SAC collection: 256 episodes per round (PPO uses 1024, but SAC
    # reuses data via replay so fewer new episodes are needed)
    episodes_per_update: int = 256
    max_env_steps: int = 10_000_000
    eval_interval: int = 100_000
    eval_episodes: int = 32

    # SAC knobs — tuned for balance task
    warmup_steps: int = 10_000
    # UTD ratio: 0.25 gives 1600 grad steps per round (256 eps × 25 steps
    # × 0.25). Capped at 2000 to keep round time ~52s. SAC's replay
    # buffer provides additional data reuse beyond the raw UTD.
    utd_ratio: float = 0.25
    max_grad_steps_per_round: int = 2000
    batch_size: int = 256
    replay_buffer_size: int = 1_000_000
    init_alpha: float = 0.2
    # Conservative target entropy: -10 instead of -21 (=-action_dim).
    # The aggressive -21 caused alpha to collapse to ~0.003 in <20 rounds,
    # leading to policy collapse (ep_len crashed 36→8). -10 allows the
    # policy to become deterministic enough to exploit while keeping
    # enough exploration to avoid collapse.
    target_entropy: float = -10.0
    # Lower alpha LR to slow down alpha convergence (3e-4 was too fast
    # with 2000 grad steps per round).
    alpha_lr: float = 1e-4
    # Clamp alpha to prevent total collapse: log_alpha_min=-5 → alpha≈0.007
    log_alpha_min: float = -5.0
    use_grad_norm: bool = True
    q_hidden_dim: int = 256
    # LayerNorm in Q trunk for stability (prevents Q overestimation crash)
    q_layer_norm: bool = True
    # Reward scale: amplify small per-step rewards (~0.005) so they're
    # visible relative to the entropy bonus (alpha × |target_entropy| ≈
    # 1.8/step). With scale=200, reward ≈ 1.0/step, comparable to the
    # entropy bonus. PPO doesn't need this because GAE naturally amplifies
    # credit assignment; SAC with 1-step TD does.
    reward_scale: float = 200.0

    # Reward constants
    per_step_phi_coef: float = 0.01

    # Actor weights
    _base_actor_weights: Tuple[float, ...] = (3.0, 1.0)

    _AGENT_IDS = ("robot_a", "robot_b")

    _survival_rate: float = 0.0
    _best_survived: float = -1.0

    def reward_channels(self) -> Tuple[SACRewardChannel, ...]:
        return (
            SACRewardChannel(
                name="r_fall", gamma=self._gamma, n_step=1,
                n_critics=2, trunk_group="shared",
            ),
            SACRewardChannel(
                name="r_cross", gamma=self._gamma, n_step=1,
                n_critics=2, trunk_group="shared",
            ),
        )

    def _build_agent_slices(
        self,
        episode,
        agent_id: str,
        cross_key: str,
        phi_key: str,
    ) -> List[TrajectorySlice]:
        T_full = episode.num_frames
        if T_full == 0:
            return []

        # Truncate at agent's termination step
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
        fin_obs = np.asarray(fin_obs, dtype=np.float32)

        # Extract φ per step
        phi_arr = extract_per_step_field(
            episode.observer_outputs, phi_key, "phi", T_full,
        )
        phi_arr = np.clip(phi_arr[:T], 0.0, 1.0).astype(np.float32)

        # r_fall: 0.01 × φ(t) per step
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # r_cross
        r_cross = extract_per_step_scalar(
            episode.observer_outputs, cross_key, T_full,
        )[:T].astype(np.float32)

        # Per-step dones: True only at the last step if the agent fell
        # (true termination → no bootstrap). If truncated (timeout),
        # done=False at all steps → bootstrap from next_obs.
        dones_fall = np.zeros(T, dtype=bool)
        dones_cross = np.zeros(T, dtype=bool)
        if fell and T > 0:
            dones_fall[-1] = True
            dones_cross[-1] = True

        # Actor weights
        aw_fall = np.full(T, self._base_actor_weights[0], dtype=np.float32)
        aw_cross = (self._base_actor_weights[1] * phi_arr ** 2).astype(np.float32)

        return [TrajectorySlice(
            obs=obs_all[:T],
            actions=acts_all[:T],
            last_obs=fin_obs,
            rewards={
                "r_fall": r_fall,
                "r_cross": r_cross,
            },
            dones={
                "r_fall": dones_fall,
                "r_cross": dones_cross,
            },
            actor_weights={
                "r_fall": aw_fall,
                "r_cross": aw_cross,
            },
            tags={
                "phi": phi_arr,
                "fell": np.array([float(fell)] * T, dtype=np.float32),
            },
            importance=1.0,
        )]

    def build_slices(self, episodes: List[Any]) -> List[TrajectorySlice]:
        agent_specs = [
            ("robot_a", "cross_support_a", "height_phi_a"),
            ("robot_b", "cross_support_b", "height_phi_b"),
        ]

        all_slices: List[TrajectorySlice] = []
        for episode in episodes:
            for agent_id, cross_key, phi_key in agent_specs:
                agent_slices = self._build_agent_slices(
                    episode, agent_id, cross_key, phi_key,
                )
                all_slices.extend(agent_slices)
        return all_slices

    def on_eval(self, episodes: List[Any], env_step: int) -> Dict[str, Any]:
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


EXPERIMENT_CLASS = SacBalance
