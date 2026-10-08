"""SAC balance experiment — 2-channel validation.

Mirrors the PPO ``basic_balance`` task semantics while emitting
``sac_transition_v2`` agent-transition slices.

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

from baseline.framework.sac.collection import SACFactSpec
from baseline.framework.sac.experiment import SACRewardChannel
from baseline.framework.sac.observer_utils import (
    extract_per_step_field,
    extract_per_step_scalar,
)
from baseline.framework.sac.transition import (
    SACTransitionSlice,
    build_agent_transition_slice,
)

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
    use_grad_norm: bool = False
    q_hidden_dim: int = 256
    # LayerNorm in Q trunk for stability (prevents Q overestimation crash)
    q_layer_norm: bool = True
    # Reward scale: amplify small per-step rewards (~0.005) so they're
    # visible relative to the entropy bonus. Scale=200 caused Q
    # overestimation and divergence at 1.5M env steps. Scale=50 with
    # lower critic LR (1e-4) should be more stable while still
    # providing enough signal.
    reward_scale: float = 50.0

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

    def pre_action_fact_specs(self) -> Tuple[SACFactSpec, ...]:
        provider = "baseline.experiments_sac.fact_providers:HeightPhiPreActionProvider"
        return tuple(
            SACFactSpec(
                name="phi_pre",
                agent_id=agent_id,
                provider=provider,
                config={"standing_height": 1.28},
            )
            for agent_id in self._AGENT_IDS
        )

    def _pre_action_phi(self, episode, agent_id: str) -> np.ndarray:
        facts = episode.pre_action_facts.get(agent_id)
        if not isinstance(facts, dict) or "phi_pre" not in facts:
            raise KeyError(
                f"Missing required pre-action fact 'phi_pre' for {agent_id!r}"
            )
        return np.asarray(facts["phi_pre"], dtype=np.float32).reshape(-1)

    def _build_agent_slices(
        self,
        episode,
        agent_id: str,
        cross_key: str,
        phi_key: str,
    ) -> List[SACTransitionSlice]:
        T_full = episode.num_frames
        if T_full == 0:
            return []
        T = episode.agent_frame_boundary.get(agent_id, T_full)
        if T == 0:
            return []

        phi_post = extract_per_step_field(
            episode.observer_outputs, phi_key, "phi", T_full,
        )
        if phi_post is None:
            raise KeyError(f"Missing required observer field {phi_key}.phi")
        phi_post = np.asarray(phi_post, dtype=np.float32)
        phi_post_clipped = np.clip(phi_post[:T], 0.0, 1.0)

        phi_pre = self._pre_action_phi(episode, agent_id)
        if phi_pre.shape[0] < T:
            raise ValueError(
                f"phi_pre length {phi_pre.shape[0]} < transition boundary {T}"
            )
        phi_pre_clipped = np.clip(phi_pre[:T], 0.0, 1.0)

        r_fall = (self.per_step_phi_coef * phi_post_clipped).astype(np.float32)
        r_cross = extract_per_step_scalar(
            episode.observer_outputs, cross_key, T_full,
        )[:T].astype(np.float32)

        actor_gate = {
            "r_fall": np.full(T, self._base_actor_weights[0], dtype=np.float32),
            "r_cross": (
                self._base_actor_weights[1] * phi_pre_clipped ** 2
            ).astype(np.float32),
        }
        actor_gate_next = {
            "r_fall": np.full(T, self._base_actor_weights[0], dtype=np.float32),
            "r_cross": (
                self._base_actor_weights[1] * phi_post_clipped ** 2
            ).astype(np.float32),
        }
        task_facts = {
            "phi_pre": phi_pre[:T],
            "phi_post_reference": phi_post[:T],
        }
        reward_features = {
            "cross_support": extract_per_step_scalar(
                episode.observer_outputs, cross_key, T_full,
            )[:T],
        }
        for field in ("height", "uprightness", "initial_phi"):
            value = extract_per_step_field(
                episode.observer_outputs, phi_key, field, T_full,
            )
            if value is None:
                raise KeyError(f"Missing required observer field {phi_key}.{field}")
            reward_features[field] = np.asarray(value, dtype=np.float32)[:T]

        sl = build_agent_transition_slice(
            episode,
            agent_id,
            channel_names=self._channel_names,
            rewards={"r_fall": r_fall, "r_cross": r_cross},
            actor_gate=actor_gate,
            actor_gate_next=actor_gate_next,
            task_facts=task_facts,
            reward_features=reward_features,
            versions={
                "reward_semantics": "basic_balance_phi_cross_v1",
                "objective_mode": "shannon",
                "regularizer_mode": "entropy",
                "policy_arch": "s01_shared_sigma",
            },
        )
        return [] if sl is None else [sl]

    def build_slices(self, episodes: List[Any]) -> List[SACTransitionSlice]:
        agent_specs = [
            ("robot_a", "cross_support_a", "height_phi_a"),
            ("robot_b", "cross_support_b", "height_phi_b"),
        ]

        all_slices: List[SACTransitionSlice] = []
        for episode in episodes:
            for agent_id, cross_key, phi_key in agent_specs:
                all_slices.extend(
                    self._build_agent_slices(episode, agent_id, cross_key, phi_key)
                )
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
