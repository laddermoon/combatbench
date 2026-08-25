"""V2 dual-agent experiment: r_fall + per-foot height channels with a
direction-scheduling actor_weight state machine.

Based on ``exp_basic_balance.py``, replacing the single ``r_cross`` channel
(CrossSupportBalanceRewarder's sparse penalty scalar) with two dense,
physically-grounded channels — one per foot::

    r_fall       = 0.01 × φ_height          γ=0.99, aw = 3.0 (fixed)
    r_left_foot  = clip(h_left,  -0.1, 0.1) γ=0.90, aw = state machine
    r_right_foot = clip(h_right, -0.1, 0.1) γ=0.90, aw = state machine

The reward carries only *physical fact* (foot height); the *intent* (which
foot should rise / descend right now) is carried by ``actor_weight``.

The stepping state machine and its rule table live in
``baseline/humanoid21/end2end/stepping_state_machine.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.ppo_trainer import _extract_per_step_field

from baseline.humanoid21.end2end.stepping_state_machine import (
    compute_foot_weights,
    FOOT_WEIGHT,
    FOOT_HEIGHT_CLIP,
    PHASE_A_STEPS,
    PHASE_B_END,
    DOUBLE_GRACE_STEPS,
    STATE_DOUBLE,
    STATE_SUPPORT_L,
    STATE_SUPPORT_R,
    STATE_FLIGHT,
)

from .base import CombatExperimentV2Base


class BasicBalanceStep(CombatExperimentV2Base):

    name = "basic_balance_step"

    _channel_names = ("r_fall", "r_left_foot", "r_right_foot")

    # Per-channel discount: r_fall is long-horizon (survival), the foot
    # channels are local/reactive (did this action lift the foot now?).
    _channel_gammas = {
        "r_fall": 0.99,
        "r_left_foot": 0.9,
        "r_right_foot": 0.9,
    }
    _gae_lambda = 0.95

    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"

    episodes_per_update: int = 256 * 4

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- r_fall actor weight (fixed, same as exp_basic_balance) ---
    r_fall_actor_weight: float = 3.0

    _AGENT_IDS = ("robot_a", "robot_b")

    _survival_rate: float = 0.0
    _best_survived: float = -1.0

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "basic_balance_step_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(
                name=k,
                gamma=self._channel_gammas[k],
                gae_lambda=self._gae_lambda,
            )
            for k in self._channel_names
        )

    # ------------------------------------------------------------------
    # Stepping state machine — delegated to stepping_state_machine.compute_foot_weights
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        foot_key: str,
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

        # --- r_fall: 0.01 × φ(t) per step ---
        phi_arr = _extract_per_step_field(
            episode.observer_outputs, phi_key, "phi", T_full,
        )
        if phi_arr is not None:
            phi_arr = phi_arr[:T]
        else:
            phi_arr = np.ones(T, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- Foot heights (saturated) ---
        h_left = self._extract_foot_field(episode, foot_key, "h_left_foot", T_full, T)
        h_right = self._extract_foot_field(episode, foot_key, "h_right_foot", T_full, T)
        r_left = np.clip(h_left, -FOOT_HEIGHT_CLIP, FOOT_HEIGHT_CLIP).astype(np.float32)
        r_right = np.clip(h_right, -FOOT_HEIGHT_CLIP, FOOT_HEIGHT_CLIP).astype(np.float32)

        # --- Contacts → stepping state machine → foot actor weights ---
        contact_l = self._extract_foot_field(
            episode, foot_key, "left_foot_contact", T_full, T,
        )
        contact_r = self._extract_foot_field(
            episode, foot_key, "right_foot_contact", T_full, T,
        )
        w_left, w_right = compute_foot_weights(
            contact_l.astype(bool), contact_r.astype(bool), T,
        )

        is_terminated = fell

        all_rewards = {
            "r_fall": r_fall,
            "r_left_foot": r_left,
            "r_right_foot": r_right,
        }
        actor_weights = {
            "r_fall": np.full(T, self.r_fall_actor_weight, dtype=np.float32),
            "r_left_foot": w_left,
            "r_right_foot": w_right,
        }

        channels: Dict[str, ChannelData] = {}
        for key in self._channel_names:
            channels[key] = ChannelData(
                reward=all_rewards[key].astype(np.float32),
                is_terminated=is_terminated,
                actor_weight=actor_weights[key],
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

    @staticmethod
    def _extract_foot_field(
        episode, foot_key: str, field: str, T_full: int, T: int,
    ) -> np.ndarray:
        """Extract a FootStateObserver field, truncated to ``T``.

        Raises if the observer or field is missing — a silent zero fallback
        would make the stepping signal vanish without any error.
        """
        arr = _extract_per_step_field(
            episode.observer_outputs, foot_key, field, T_full,
        )
        if arr is None:
            raise KeyError(
                f"_extract_foot_field: observer '{foot_key}' field '{field}' "
                f"missing from episode.observer_outputs "
                f"(available observers={list(episode.observer_outputs.keys())})"
            )
        return arr[:T]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        agent_specs = [
            ("robot_a", "foot_state_a", "height_phi_a"),
            ("robot_b", "foot_state_b", "height_phi_b"),
        ]

        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, foot_key, phi_key in agent_specs:
                all_trajs.extend(
                    self._build_agent_trajectory(
                        episode, agent_id, foot_key, phi_key,
                    )
                )
        return all_trajs

    # ------------------------------------------------------------------
    # Evaluation
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


EXPERIMENT_CLASS = BasicBalanceStep
