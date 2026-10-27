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
    r_band_step = stride-settled radial verdict — learner dxy over the
                  gait cycle decomposed on d_hat = to_opp(t_b):
                  dist>1.0 → +2·rad, dist<0.6 → −2·rad, else 0;
                  sparse at landing, aw = 3.0·out_band·φ² (dense)
    r_tan_step  = −|tan| per stride, same settlement, aw = 1.0·out_band·φ²
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
from .exp_balance_pure import BalancePure
from .exp_balance_step import _balance_explore_factor
from .exp_follow_pure import FollowPure


class HierS3FollowGuard96D(FollowPure):
    """Follow + face + jab_l guard under the raw 96-dim contract."""

    name = "hier_s3_follow_guard_96d"

    actor_blueprint = "init_policy_mbs_96d.yaml"

    max_updates: int = 6000

    # --- Channels: stride-settled band/tangent + dense face + clip ---
    _channel_names = (
        "r_potential", "r_fall", "r_cross",
        "r_band_step", "r_tan_step", "r_face", "r_std_clip",
    )
    _channel_gammas = {
        "r_potential": 0.99, "r_fall": 0.99, "r_cross": 0.99,
        "r_band_step": 0.99, "r_tan_step": 0.99, "r_face": 0.99,
        "r_std_clip": 0.99,
    }

    # --- Distance band (attack-prep range) ---
    D_MIN: float = 0.6
    D_MAX: float = 1.0
    HOLD_DIST: float = 1.0          # eval hold = within reach band
    D_FACE: float = 1.5
    D_STRIKE: float = 0.5

    # --- Stride settlement (skill_step_mbs recipe) ---
    _CONTACT_TAU: float = 0.10      # feet contact threshold
    _MIN_STRIKE_GAP: int = 3        # debounce frames between strikes
    _RAD_COEF: float = 2.0          # radial displacement coef / stride
    _TAN_COEF: float = 1.0          # tangential displacement coef

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
        0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
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
                for key in ("r_band_step", "r_tan_step", "r_face",
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
        out_band = (
            (dist > self.D_MAX) | (dist < self.D_MIN)
        ).astype(np.float32)
        in_band = (1.0 - out_band).astype(np.float32)

        # --- Stride-settled verdicts (skill_step_mbs recipe) ---
        # Each gait cycle [t_a, t_b] is scored ONCE at the landing step:
        # the LEARNER's net displacement dxy is decomposed against
        # d_hat — the direction to the opponent AT t_b — so the
        # opponent's own motion never feeds the reward (unlike Δdist).
        lc = extract_per_step_field(
            oo, foot_key, "left_foot_contact", T_full)
        rc = extract_per_step_field(
            oo, foot_key, "right_foot_contact", T_full)
        r_band_step = np.zeros(T_full, dtype=np.float32)
        r_tan_step = np.zeros(T_full, dtype=np.float32)
        if lc is not None and rc is not None:
            contact = np.stack(
                [np.asarray(lc[:T_full]) > 0.5,
                 np.asarray(rc[:T_full]) > 0.5], axis=1)
            for t_a, t_b in self._stride_windows(
                    self._strike_bounds(contact), T_full):
                if t_b <= t_a:
                    continue
                dxy = self_xy[t_b] - self_xy[t_a]
                to_opp = opp_xy[t_b] - self_xy[t_b]
                dist_b = float(np.linalg.norm(to_opp))
                if dist_b < 1e-6:
                    continue
                d_hat = to_opp / dist_b
                rad = float(dxy @ d_hat)
                tan = dxy - rad * d_hat
                if dist_b > self.D_MAX:
                    r_band_step[t_b - 1] += self._RAD_COEF * rad
                elif dist_b < self.D_MIN:
                    r_band_step[t_b - 1] += -self._RAD_COEF * rad
                r_tan_step[t_b - 1] += (
                    -self._TAN_COEF * float(np.linalg.norm(tan)))

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
            t.channels["r_band_step"] = ChannelData(
                reward=r_band_step[:T],
                is_terminated=ref.is_terminated,
                actor_weight=(
                    self.r_band_actor_weight
                    * out_band[:T] * stand2[:T]
                ).astype(np.float32),
            )
            t.channels["r_tan_step"] = ChannelData(
                reward=r_tan_step[:T],
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
    # Stride segmentation (from foot_state observer contacts)
    # ------------------------------------------------------------------

    @classmethod
    def _strike_bounds(cls, contact: np.ndarray) -> Dict[int, List[int]]:
        """Per-foot foot-strike frames {foot: [t, ...]} — rising edges
        of ground contact, debounced by _MIN_STRIKE_GAP."""
        prev = np.zeros_like(contact)
        prev[1:] = contact[:-1]
        rise = contact & ~prev
        rise[0] = False
        bounds: Dict[int, List[int]] = {}
        for foot in range(2):
            strikes: List[int] = []
            for t in np.flatnonzero(rise[:, foot]):
                if strikes and t - strikes[-1] < cls._MIN_STRIKE_GAP:
                    continue
                strikes.append(int(t))
            bounds[foot] = strikes
        return bounds

    @staticmethod
    def _stride_windows(bounds: Dict[int, List[int]],
                        T: int) -> List[tuple]:
        """Sliding per-foot windows (prev same-foot strike, strike].
        A virtual closing strike on the last-landed foot settles the
        in-flight cycle at T−1; no complete cycle → whole-episode win."""
        wins: List[tuple] = []
        last_foot, last_t = -1, -1
        for foot, strikes in bounds.items():
            for i in range(1, len(strikes)):
                wins.append((strikes[i - 1], strikes[i]))
            if strikes and strikes[-1] > last_t:
                last_foot, last_t = foot, strikes[-1]
        if last_foot >= 0 and last_t < T - 1 and \
                len(bounds[last_foot]) >= 2:
            wins.append((bounds[last_foot][-2], T - 1))
        wins.sort(key=lambda w: w[1])
        if not wins:
            wins = [(0, T - 1)]
        return wins

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
