"""Hierarchy Step 2 — push recovery -> standard stepping (65-dim).

Warm-start from the Step-1 final checkpoint (65-dim MBS).  The robot
already stands up and steps cyclically; this stage adds the 12-level
StandingTriggeredForce curriculum (40N -> 100N) and teaches it to
survive pushes and re-settle into the standard stepping gait.

Channels (3):
    r_potential = 0.01 × φ(t),   aw = 3.0 (fixed)
    r_fall      = fall + wall-lean + push-clean,  aw = 2.0 (fixed)
    r_std_clip  = clip + 0.3·pose + 0.5·cad,
                                 aw = φ² · w_clip · (push_phase != PUSHING)

Clip gate: while a push is physically applied the gait reward is cut
(phase==2) so the policy may break stance and take large recovery
steps; in WAIT/DELAY/OBSERVE it is back on so the robot returns to the
standard stepping gait.  φ² keeps it standing-only.

Blueprint: baseline/humanoid21/hierarchy/blueprints/hier_s2_balance_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.humanoid21.end2end.stepping_state_machine import (
    fall_onset_mask,
)

from .exp_balance_pure import BalancePure


_PUSH_PHASE_PUSHING = 2


class HierS2BalanceMBS(BalancePure):
    """Step-2: standup+stepping policy under the push curriculum."""

    name = "hier_s2_balance_mbs"

    # --- Network (65-dim CommandSimulator contract) ---
    obs_dim: int = 65
    action_dim: int = 21
    actor_blueprint = "init_policy_mbs_65d.yaml"

    # --- Channels: replace r_cross with r_std_clip ---
    _channel_names = ("r_potential", "r_fall", "r_std_clip")
    _channel_gammas = {
        "r_potential": 0.99, "r_fall": 0.99, "r_std_clip": 0.99,
    }

    _CLIP_OBS = {"robot_a": "standard_clip_a", "robot_b": "standard_clip_b"}
    _clip_aw_scale: float = 2.0

    # ------------------------------------------------------------------
    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s2_balance_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

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
        phi_arr = (np.clip(np.asarray(phi_arr[:T], dtype=np.float32),
                           0.0, 1.0) if phi_arr is not None
                   else np.zeros(T, dtype=np.float32))

        r_potential = (self.per_step_phi_coef * phi_arr).astype(np.float32)

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
        push_phase = extract_per_step_field(
            episode.observer_outputs, push_key, "push_phase", T_full,
        ) if push_key is not None else None
        not_pushing = (np.asarray(push_phase[:T]) != _PUSH_PHASE_PUSHING) \
            if push_phase is not None else np.ones(T, dtype=bool)
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

        # --- r_std_clip: gait reward, φ²-gated and off during pushes ---
        clip_key = self._CLIP_OBS.get(agent_id)
        r_clip = np.zeros(T, dtype=np.float32)
        if clip_key is not None:
            for field, w in (("std_clip", 1.0), ("std_pose", 0.3),
                             ("std_cad", 0.5)):
                arr = extract_per_step_field(
                    episode.observer_outputs, clip_key, field, T_full)
                if arr is not None:
                    r_clip = r_clip + w * np.clip(
                        np.asarray(arr[:T], dtype=np.float32), -1.0, 1.0)

        aw_clip = (self._clip_aw_scale * phi_arr * phi_arr
                   * not_pushing.astype(np.float32)).astype(np.float32)

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
            "r_std_clip": ChannelData(
                reward=r_clip,
                is_terminated=False,
                actor_weight=aw_clip,
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
    # Eval — BalancePure metrics + clip score on standing frames
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        clips: List[float] = []
        cads: List[float] = []
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for agent_id, foot_key, phi_key in self._AGENT_OBS:
                phi = extract_per_step_field(
                    ep.observer_outputs, phi_key, "potential", T)
                c = extract_per_step_field(
                    ep.observer_outputs, self._CLIP_OBS.get(agent_id),
                    "std_clip", T)
                cad = extract_per_step_field(
                    ep.observer_outputs, self._CLIP_OBS.get(agent_id),
                    "std_cad", T)
                if phi is None or c is None:
                    continue
                stand = np.asarray(phi[:T]) >= 0.9
                if stand.any():
                    clips.append(float(np.asarray(c[:T])[stand].mean()))
                    if cad is not None:
                        cads.append(float(np.asarray(cad[:T])[stand].mean()))
        result["info"]["clip"] = round(
            sum(clips) / max(len(clips), 1), 3)
        result["info"]["cad"] = round(
            sum(cads) / max(len(cads), 1), 3)
        return result


EXPERIMENT_CLASS = HierS2BalanceMBS
