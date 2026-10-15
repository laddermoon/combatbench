"""Hierarchy Stage 0 — Standup + push recovery + guard posture (65-dim).

Warm-start from ``standup_mbs_ef05_s42`` (u350, 96-dim MBS) via
``init_hier_ckpt.py`` — actor/critic first layers sliced (H,96)→(H,65),
8 command columns zero-padded.

The CommandSimulator strips obs[57:96] (opponent block) and appends
an 8-dim all-zero STAND command; the policy sees 65 dims.  Goal: lock
in get-up + push-survive + hands-to-guard under the new contract before
any strike/move commands are introduced.

Channels (4):
    r_potential = 0.01 × φ(t),         aw = 3.0 (fixed)
    r_fall      = fall + wall-lean + push-clean,  aw = 2.0 (fixed)
    r_cross     = alternation timing,  aw = 1.0 × g_safe(φ)
    r_guard     = guard posture score, aw = 1.5 × g_safe(φ)

Safety gate: g_safe(φ) = I(φ ≥ φ_safe) × φ², φ_safe = 0.75.
When φ drops below the threshold all task-channel gradients vanish;
only r_potential and r_fall keep flowing.

Blueprint: baseline/humanoid21/hierarchy/blueprints/hier_s0_stand_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import (
    extract_per_step_field,
    extract_per_step_scalar,
)
from baseline.humanoid21.end2end.stepping_state_machine import (
    fall_onset_mask,
)

from .exp_balance_pure import BalancePure


# Safety gate threshold — below this φ all task-channel aw → 0
PHI_SAFE: float = 0.75


def g_safe(phi: np.ndarray) -> np.ndarray:
    """Safety-gated actor weight multiplier."""
    return np.where(phi >= PHI_SAFE, phi ** 2, 0.0).astype(np.float32)


class HierS0StandMBS(BalancePure):
    """65-dim standup consolidation + guard + push recovery."""

    name = "hier_s0_stand_mbs"

    # --- Network ---
    obs_dim: int = 65
    action_dim: int = 21

    actor_blueprint = "init_policy_mbs_65d.yaml"

    # --- Reward channels ---
    _channel_names = ("r_potential", "r_fall", "r_cross", "r_guard")
    _channel_gammas = {
        "r_potential": 0.99, "r_fall": 0.99,
        "r_cross": 0.99, "r_guard": 0.99,
    }

    # --- Guard channel ---
    r_guard_actor_weight: float = 1.5
    _GUARD_OBS = {
        "robot_a": "guard_posture_a",
        "robot_b": "guard_posture_b",
    }

    # --- Agent-observer mapping (extended with guard) ---
    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a"),
        ("robot_b", "foot_state_b", "standing_balance_b"),
    )
    _AGENT_IDS = ("robot_a", "robot_b")
    agent_used = "both"
    max_steps: int = 400

    # ------------------------------------------------------------------
    # Blueprint — hier_s0_stand_env.yaml
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s0_stand_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    # ------------------------------------------------------------------
    # Trajectory building — BalancePure recipe + r_guard + g_safe gate
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
        T = episode.agent_frame_boundary.get(agent_id, T_full)
        if T == 0:
            return []

        obs_all = episode.observations.get(agent_id)
        acts_all = episode.actions.get(agent_id)
        fin_obs = episode.final_observation.get(agent_id)
        if obs_all is None or acts_all is None or fin_obs is None:
            return []

        obs_all = np.asarray(obs_all, dtype=np.float32)
        acts_all = np.asarray(acts_all, dtype=np.float32)

        phi_arr = extract_per_step_field(
            episode.observer_outputs, phi_key, "potential", T_full,
        )
        if phi_arr is not None:
            phi_arr = np.clip(
                np.asarray(phi_arr[:T], dtype=np.float32), 0.0, 1.0)
        else:
            phi_arr = np.zeros(T, dtype=np.float32)

        r_potential = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- r_fall: fall-onset + wall-lean + clean-push bonus ---
        r_fall = (
            -self.fall_penalty
            * fall_onset_mask(phi_arr).astype(np.float32)
        )

        wall_key = self._WALL_OBS.get(agent_id)
        if wall_key is not None and self.wall_penalty > 0:
            from baseline.humanoid21.end2end.wall_contact_observer import (
                sustained_wall_mask,
            )
            wc = extract_per_step_field(
                episode.observer_outputs, wall_key, "wall_contact", T_full,
            )
            if wc is not None:
                standing = phi_arr >= 0.5
                lean = sustained_wall_mask(
                    wc, window=self.wall_window,
                    count_thresh=self.wall_count_thresh,
                )
                r_fall = r_fall - (
                    self.wall_penalty
                    * lean[:T].astype(np.float32)
                    * standing[:T].astype(np.float32)
                )

        push_key = self._PUSH_OBS.get(agent_id)
        if push_key is not None and self.push_clean_bonus > 0:
            clean = extract_per_step_field(
                episode.observer_outputs, push_key,
                "push_window_clean", T_full,
            )
            if clean is not None:
                r_fall = r_fall + (
                    self.push_clean_bonus
                    * np.asarray(clean[:T], dtype=bool).astype(np.float32)
                )

        # --- r_cross: alternation timing, g_safe(φ) gate ---
        cross_key = self._CROSS_OBS.get(agent_id)
        r_cross = extract_per_step_scalar(
            episode.observer_outputs, cross_key, T_full,
        ) if cross_key is not None else None
        if r_cross is None:
            r_cross = np.zeros(T, dtype=np.float32)
        else:
            r_cross = np.asarray(r_cross[:T], dtype=np.float32)

        # --- r_guard: guard posture score, g_safe(φ) gate ---
        guard_key = self._GUARD_OBS.get(agent_id)
        guard_score = extract_per_step_field(
            episode.observer_outputs, guard_key, "guard_score", T_full,
        )
        if guard_score is not None:
            r_guard = np.asarray(
                guard_score[:T], dtype=np.float32)
        else:
            r_guard = np.zeros(T, dtype=np.float32)

        gs = g_safe(phi_arr)

        channels = {
            "r_potential": ChannelData(
                reward=r_potential,
                is_terminated=False,
                actor_weight=np.full(
                    T, self.r_potential_actor_weight, dtype=np.float32),
            ),
            "r_fall": ChannelData(
                reward=r_fall.astype(np.float32),
                is_terminated=False,
                actor_weight=np.full(
                    T, self.r_fall_actor_weight, dtype=np.float32),
            ),
            "r_cross": ChannelData(
                reward=r_cross,
                is_terminated=False,
                actor_weight=(
                    self.r_cross_actor_weight * gs
                ).astype(np.float32),
            ),
            "r_guard": ChannelData(
                reward=r_guard,
                is_terminated=False,
                actor_weight=(
                    self.r_guard_actor_weight * gs
                ).astype(np.float32),
            ),
        }
        return [Trajectory(
            obs=obs_all[:T],
            actions=acts_all[:T],
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(episode, agent_id, T),
        )]

    # ------------------------------------------------------------------
    # Eval — BalancePure metrics + guard score
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        # Augment with guard score
        all_guards = []
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for agent_id in self._AGENT_IDS:
                gkey = self._GUARD_OBS.get(agent_id)
                if gkey is None:
                    continue
                g = extract_per_step_field(
                    ep.observer_outputs, gkey, "guard_score", T,
                )
                if g is not None and len(g) > 0:
                    all_guards.append(
                        float(np.mean(np.asarray(g))))
        guard_mean = sum(all_guards) / max(len(all_guards), 1)
        result["info"]["guard"] = round(guard_mean, 3)
        return result


EXPERIMENT_CLASS = HierS0StandMBS
