"""Guard-96 Step 3 — follow & face an arc-wandering opponent (96-dim).

96-dim AttackPrep stage 3: warm-start from the ``hier_s2_guard_96d``
balance policy and learn to hold the [0.6, 1.0] m distance band around
an ``ArcMovePlugin`` opponent while staying in the jab_l dynamic guard.

Motivation vs follow_pure / follow_s42b: the old waypoint mover fled
outward at a fixed per-episode speed, so the policy only learned
forward chasing — no quick turns, no retreat, no band holding.  The new
mover gives random headings (bounded-curvature arcs inside a segment,
arbitrary jumps at boundaries), a 1-3 s time-segmented speed profile
with rate-limited ramps (v* ~ U[0,1] m/s), 50/50 facing modes, and it
may move TOWARD the learner — the band reward is symmetric.

Channels (7):
    r_potential = 0.01 × φ(t) × I(h_torso ≤ 1.24),  aw = 3.0
    r_fall      = fall-onset + wall-lean,           aw = 2.0
    r_cross     = alternation guidance,             aw = 1.0·φ²
    r_band      = sgn-gated radial velocity:
                  dist>1.0 → +v_rad, dist<0.6 → −v_rad, else 0,
                                                   aw = 3.0·out_band·φ²
    r_tangential= −|v_tan| lateral drift,           aw = 1.0·out_band·φ²
    r_face      = cos(forward, to_opp)⁺,            aw = 1.0·dist_gate·φ²
    r_std_clip  = clip + pose + 0.5·cad + 0.5·h_term(h→1.21),
                          aw = φ²·(1.0 out-band, 3.0 in-band)

Blueprint: blueprints/hier_s3_follow_guard_96d_env.yaml
Launch:
    --experiment hier_s3_follow_guard_96d \
    --resume-from <s2 ckpt> --reset-update
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.humanoid21.rewards.follow_opponent import (
    compute_radial_tangential_rewards,
)

from .exp_balance_pure import BalancePure
from .exp_balance_step import _balance_explore_factor
from .exp_follow_pure import FollowPure


class HierS3FollowGuard96D(FollowPure):
    """Follow + face + jab_l guard under the raw 96-dim contract."""

    name = "hier_s3_follow_guard_96d"

    actor_blueprint = "init_policy_mbs_96d.yaml"

    max_updates: int = 6000

    # --- Channels: swap r_radial for band-aware r_band, add r_std_clip ---
    _channel_names = (
        "r_potential", "r_fall", "r_cross",
        "r_band", "r_tangential", "r_face", "r_std_clip",
    )
    _channel_gammas = {
        "r_potential": 0.99, "r_fall": 0.99, "r_cross": 0.99,
        "r_band": 0.99, "r_tangential": 0.99, "r_face": 0.99,
        "r_std_clip": 0.99,
    }

    # --- Distance band (attack-prep range) ---
    D_MIN: float = 0.6
    D_MAX: float = 1.0
    HOLD_DIST: float = 1.0          # eval hold = within reach band
    D_FACE: float = 1.5
    D_STRIKE: float = 0.5

    r_band_actor_weight: float = 3.0
    r_tangential_actor_weight: float = 1.0
    r_face_actor_weight: float = 1.0

    # --- Guard clip channel (stronger inside the band) ---
    _CLIP_OBS = {
        "robot_a": "standard_clip_a", "robot_b": "standard_clip_b",
    }
    _clip_aw_out: float = 1.0       # aw scale outside the band
    _clip_aw_in: float = 3.0        # aw scale inside the band
    _H_TARGET: float = 1.21
    _H_SIGMA: float = 0.08
    _pot_h_cap = 1.24

    # --- Speed curriculum: 9 levels 0.2 -> 1.0 m/s ---
    # Promotes when eval hold (dist <= HOLD_DIST on standing frames)
    # >= PROMOTE_HOLD_RATIO for FOLLOW_PROMOTE_PATIENCE evals running.
    # v* ~ U[0, level_max] inside the ArcMovePlugin's speed segments.
    LEVEL_SPEEDS: Tuple[float, ...] = (
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    )

    # ------------------------------------------------------------------
    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s3_follow_guard_96d_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    # ------------------------------------------------------------------
    def _build_agent_trajectory(
        self, episode, agent_id: str, foot_key: str, phi_key: str,
    ) -> List[Trajectory]:
        # Base channels (r_potential + r_fall + r_cross) straight from
        # BalancePure — bypass FollowPure's radial/follow additions.
        trajs = BalancePure._build_agent_trajectory(
            self, episode, agent_id, foot_key, phi_key,
        )
        if not trajs:
            return trajs

        T_full = episode.num_frames
        oo = episode.observer_outputs
        app_key = self._APPROACH_OBS.get(agent_id)
        face_key = self._FACE_OBS.get(agent_id)
        clip_key = self._CLIP_OBS.get(agent_id)

        self_x = extract_per_step_field(oo, app_key, "self_x", T_full)
        self_y = extract_per_step_field(oo, app_key, "self_y", T_full)
        opp_x = extract_per_step_field(oo, app_key, "opp_x", T_full)
        opp_y = extract_per_step_field(oo, app_key, "opp_y", T_full)
        fwd_x = extract_per_step_field(oo, face_key, "forward_x", T_full)
        fwd_y = extract_per_step_field(oo, face_key, "forward_y", T_full)

        phi = extract_per_step_field(oo, phi_key, "potential", T_full)
        stand2 = (
            np.clip(np.asarray(phi, dtype=np.float32), 0.0, 1.0) ** 2
            if phi is not None else np.zeros(T_full, dtype=np.float32)
        )

        h_arr = extract_per_step_field(oo, phi_key, "h_torso", T_full)
        h_arr = (np.asarray(h_arr[:T_full], dtype=np.float32)
                 if h_arr is not None else np.zeros(T_full, dtype=np.float32))

        if self_x is None or opp_x is None:
            for t in trajs:
                T = t.obs.shape[0]
                z = np.zeros(T, dtype=np.float32)
                for key in ("r_band", "r_tangential", "r_face",
                            "r_std_clip"):
                    t.channels[key] = ChannelData(
                        reward=z.copy(),
                        is_terminated=t.channels["r_fall"].is_terminated,
                        actor_weight=z.copy(),
                    )
            return trajs

        self_xy = np.stack([self_x, self_y], axis=1).astype(np.float64)
        opp_xy = np.stack([opp_x, opp_y], axis=1).astype(np.float64)
        fwd = (
            np.stack([fwd_x, fwd_y], axis=1).astype(np.float64)
            if fwd_x is not None and fwd_y is not None
            else np.tile([1.0, 0.0], (T_full, 1))
        )
        dist = np.linalg.norm(opp_xy - self_xy, axis=1)
        far = dist > self.D_MAX
        near = dist < self.D_MIN
        out_band = (far | near).astype(np.float32)
        in_band = (1.0 - out_band).astype(np.float32)

        # --- r_band: sign-gated radial velocity ---
        r_radial, r_tan = compute_radial_tangential_rewards(
            self_xy, opp_xy, gate=False,
        )
        r_band = np.where(
            far, r_radial,
            np.where(near, -r_radial, 0.0),
        ).astype(np.float32)

        # --- r_face ---
        to_opp_hat = (opp_xy - self_xy) / np.maximum(dist, 1e-6)[:, None]
        cos_angle = np.sum(fwd * to_opp_hat, axis=1)
        r_face = np.maximum(0.0, cos_angle).astype(np.float32)
        dist_gate = np.clip(
            (self.D_FACE - dist) / (self.D_FACE - self.D_STRIKE),
            0.0, 1.0,
        ).astype(np.float32)

        # --- r_std_clip: guard posture, emphasised inside the band ---
        r_clip = np.zeros(T_full, dtype=np.float32)
        if clip_key is not None:
            for field, w in (("std_clip", 1.0), ("std_pose", 1.0),
                             ("std_cad", 0.5)):
                arr = extract_per_step_field(
                    oo, clip_key, field, T_full)
                if arr is not None:
                    r_clip = r_clip + w * np.clip(
                        np.asarray(arr[:T_full], dtype=np.float32),
                        -1.0, 1.0)
        r_clip = r_clip + 0.5 * np.exp(
            -((h_arr - self._H_TARGET) / self._H_SIGMA) ** 2)
        clip_w = (
            self._clip_aw_out
            + (self._clip_aw_in - self._clip_aw_out) * in_band
        ).astype(np.float32)

        for t in trajs:
            T = t.obs.shape[0]
            ref = t.channels["r_fall"]
            # Apply the potential cap post-hoc on the base channel.
            if self._pot_h_cap is not None:
                pot = t.channels["r_potential"]
                t.channels["r_potential"] = ChannelData(
                    reward=(pot.reward
                            * (h_arr[:T] <= self._pot_h_cap)
                            ).astype(np.float32),
                    is_terminated=pot.is_terminated,
                    actor_weight=pot.actor_weight,
                )
            t.channels["r_band"] = ChannelData(
                reward=r_band[:T],
                is_terminated=ref.is_terminated,
                actor_weight=(
                    self.r_band_actor_weight
                    * out_band[:T] * stand2[:T]
                ).astype(np.float32),
            )
            t.channels["r_tangential"] = ChannelData(
                reward=r_tan[:T].astype(np.float32),
                is_terminated=ref.is_terminated,
                actor_weight=(
                    self.r_tangential_actor_weight
                    * out_band[:T] * stand2[:T]
                ).astype(np.float32),
            )
            t.channels["r_face"] = ChannelData(
                reward=r_face[:T],
                is_terminated=ref.is_terminated,
                actor_weight=(
                    self.r_face_actor_weight
                    * dist_gate[:T] * stand2[:T]
                ).astype(np.float32),
            )
            t.channels["r_std_clip"] = ChannelData(
                reward=r_clip[:T],
                is_terminated=ref.is_terminated,
                actor_weight=(
                    clip_w[:T] * stand2[:T]
                ).astype(np.float32),
            )
        return trajs

    # ------------------------------------------------------------------
    # Jobs — learner side + ArcMove env (no speed curriculum params)
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        smax = self.current_speed
        env_bps: Dict[str, Any] = {
            aid: self._env_pb().materialize(
                max_steps=self.max_steps,
                agent_id=aid,
                oppo_agent_id=(
                    "robot_b" if aid == "robot_a" else "robot_a"
                ),
                arc_v_max=smax,
            )
            for aid in self._AGENT_IDS
        }
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=_balance_explore_factor)
        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=policy_bp,
                env_bp=env_bps[agent_id],
                seed=seed,
                episode_options={
                    "agent_id": agent_id,
                    "initial_distance": initial_distance,
                },
                sampling_a=sampling,
                sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Eval — FollowPure metrics + band ratio + guard clip metrics
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)

        band_ratios: List[float] = []
        clips: List[float] = []
        cads: List[float] = []
        poses: List[float] = []
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            agent_id = str(
                ep.episode_options.get("agent_id", "robot_a"))
            _, phi_key = self._AGENT_OBS_MAP[agent_id]
            app_key = self._APPROACH_OBS[agent_id]
            clip_key = self._CLIP_OBS[agent_id]

            phi = extract_per_step_field(
                ep.observer_outputs, phi_key, "potential", T)
            if phi is None:
                continue
            standing = np.asarray(phi, dtype=np.float32) >= 0.5
            if not standing.any():
                continue

            self_x = extract_per_step_field(
                ep.observer_outputs, app_key, "self_x", T)
            self_y = extract_per_step_field(
                ep.observer_outputs, app_key, "self_y", T)
            opp_x = extract_per_step_field(
                ep.observer_outputs, app_key, "opp_x", T)
            opp_y = extract_per_step_field(
                ep.observer_outputs, app_key, "opp_y", T)
            if self_x is not None and opp_x is not None:
                dist = np.linalg.norm(
                    np.stack([opp_x, opp_y], axis=1)
                    - np.stack([self_x, self_y], axis=1), axis=1)
                band_ratios.append(float(np.mean(
                    (dist[standing] >= self.D_MIN)
                    & (dist[standing] <= self.D_MAX))))

            c = extract_per_step_field(
                ep.observer_outputs, clip_key, "std_clip", T)
            p = extract_per_step_field(
                ep.observer_outputs, clip_key, "std_pose", T)
            cd = extract_per_step_field(
                ep.observer_outputs, clip_key, "std_cad", T)
            if c is not None:
                clips.append(float(np.asarray(c[:T])[standing].mean()))
            if p is not None:
                poses.append(float(np.asarray(p[:T])[standing].mean()))
            if cd is not None:
                cads.append(float(np.asarray(cd[:T])[standing].mean()))

        result["info"]["band"] = round(
            sum(band_ratios) / max(len(band_ratios), 1), 3)
        result["info"]["clip"] = round(
            sum(clips) / max(len(clips), 1), 3)
        result["info"]["pose"] = round(
            sum(poses) / max(len(poses), 1), 3)
        result["info"]["cad"] = round(
            sum(cads) / max(len(cads), 1), 3)
        return result


EXPERIMENT_CLASS = HierS3FollowGuard96D
