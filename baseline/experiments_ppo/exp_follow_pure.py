"""New-route follow — directly on top of standup, NO perturbation.

Stage-2-skip variant (per user direction): the standup checkpoint goes
straight into chasing a scripted moving target.  Balance and stepping
are NOT pretrained or scaffolded — no gait clock (96-dim obs), no foot
reward channels, and **no push perturbation at all** (the balance_pure
line is the pushed control arm).

Target motion satisfies the new-route stage-3 requirements:
  * random waypoints          → direction randomness
  * per-episode speed
    U[SPEED_MIN_FRAC·smax, smax] → speed randomness, smax set by the
    13-level speed curriculum (promote at eval hold_ratio ≥ 0.5 ×2)
  * min_avoid_distance        → never moves toward the learner

Channels (5):
    r_potential = 0.01 × φ(t),              aw = 3.0
    r_fall      = fall-onset + wall-lean,   aw = 2.0
    r_radial    — velocity toward opponent, aw = 3.0 × out_zone × φ²
    r_tangential— lateral drift penalty,    aw = 1.0 × out_zone × φ²
    r_face      — cos(forward, to_opp)⁺,    aw = 1.0 × dist_gate × φ²

Foot-state observations are eval-only stepping diagnostics (does
stepping emerge under pure task pressure?).

quality = 10·speed_level + hold_ratio + 0.2·facing_ratio

Blueprint: baseline/humanoid21/end2end/follow_pure_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.humanoid21.rewards.follow_opponent import (
    FOLLOW_DIST_MAX,
    compute_radial_tangential_rewards,
)
from baseline.humanoid21.end2end.stepping_state_machine import (
    detect_step_cycles,
)

from .exp_balance_pure import BalancePure
from .exp_balance_step import _balance_explore_factor


class FollowPure(BalancePure):
    """Standup → follow/face with zero perturbation and no gait scaffold."""

    name = "follow_pure"
    agent_used = "random"

    max_steps: int = 600
    max_updates: int = 12000

    # --- Channels ---
    _channel_names = BalancePure._channel_names + (
        "r_radial", "r_tangential", "r_face",
    )
    _channel_gammas = {
        **BalancePure._channel_gammas,
        "r_radial": 0.99,
        "r_tangential": 0.99,
        "r_face": 0.99,
    }

    # --- Follow/face actor weights (follow_v2 recipe) ---
    r_radial_actor_weight: float = 3.0
    r_tangential_actor_weight: float = 1.0
    r_face_actor_weight: float = 1.0
    D_FACE: float = 1.5
    D_STRIKE: float = 0.7
    HOLD_DIST: float = 1.1

    # --- Opponent speed curriculum (follow_v2's 13 levels) ---
    LEVEL_SPEEDS: Tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 1.0, 1.2, 1.5,
    )
    # Per-episode target speed ~ U[frac·smax, smax] — speed randomness
    # on top of the curriculum upper bound.
    SPEED_MIN_FRAC: float = 0.3
    PROMOTE_HOLD_RATIO: float = 0.5
    FOLLOW_PROMOTE_PATIENCE: int = 2

    _APPROACH_OBS = {
        "robot_a": "approach_velocity_a",
        "robot_b": "approach_velocity_b",
    }
    _FACE_OBS = {
        "robot_a": "face_opponent_a",
        "robot_b": "face_opponent_b",
    }
    _AGENT_OBS_MAP = {
        "robot_a": ("foot_state_a", "standing_balance_a"),
        "robot_b": ("foot_state_b", "standing_balance_b"),
    }

    # --- Follow curriculum / tracking state ---
    _speed_level: int = 0
    _f_consecutive_pass: int = 0
    _hold_ratio: float = 0.0
    _facing_ratio: float = 0.0
    _f_best_quality: float = -1.0
    _f_quality_ema: float | None = None
    _f_last_best_update: int = -1

    # ------------------------------------------------------------------
    # Blueprint
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "follow_pure_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    @property
    def current_speed(self) -> float:
        idx = max(0, min(self._speed_level, len(self.LEVEL_SPEEDS) - 1))
        return float(self.LEVEL_SPEEDS[idx])

    # ------------------------------------------------------------------
    # Trajectories — base recipe + follow/face channels, learner only
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self, episode, agent_id: str, foot_key: str, phi_key: str,
    ) -> List[Trajectory]:
        trajs = super()._build_agent_trajectory(
            episode, agent_id, foot_key, phi_key,
        )
        if not trajs:
            return trajs

        T_full = episode.num_frames
        oo = episode.observer_outputs
        app_key = self._APPROACH_OBS.get(agent_id)
        face_key = self._FACE_OBS.get(agent_id)

        self_x = extract_per_step_field(oo, app_key, "self_x", T_full)
        self_y = extract_per_step_field(oo, app_key, "self_y", T_full)
        opp_x = extract_per_step_field(oo, app_key, "opp_x", T_full)
        opp_y = extract_per_step_field(oo, app_key, "opp_y", T_full)
        fwd_x = extract_per_step_field(oo, face_key, "forward_x", T_full)
        fwd_y = extract_per_step_field(oo, face_key, "forward_y", T_full)

        if self_x is None or opp_x is None:
            for t in trajs:
                T = t.obs.shape[0]
                z = np.zeros(T, dtype=np.float32)
                ref = t.channels["r_fall"]
                for key, aw in (
                    ("r_radial", self.r_radial_actor_weight),
                    ("r_tangential", self.r_tangential_actor_weight),
                    ("r_face", self.r_face_actor_weight),
                ):
                    t.channels[key] = ChannelData(
                        reward=z.copy(),
                        is_terminated=ref.is_terminated,
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

        r_radial, r_tangential = compute_radial_tangential_rewards(
            self_xy, opp_xy, gate=False,
        )
        out_zone = (dist > FOLLOW_DIST_MAX).astype(np.float32)

        to_opp_norm = np.maximum(dist, 1e-6)
        to_opp_hat = (opp_xy - self_xy) / to_opp_norm[:, None]
        cos_angle = np.sum(fwd * to_opp_hat, axis=1)
        r_face = np.maximum(0.0, cos_angle).astype(np.float32)
        dist_gate = np.clip(
            (self.D_FACE - dist) / (self.D_FACE - self.D_STRIKE),
            0.0, 1.0,
        ).astype(np.float32)

        phi = extract_per_step_field(oo, phi_key, "potential", T_full)
        stand2 = (
            np.clip(np.asarray(phi, dtype=np.float32), 0.0, 1.0) ** 2
            if phi is not None
            else np.zeros(T_full, dtype=np.float32)
        )

        for t in trajs:
            T = t.obs.shape[0]
            ref = t.channels["r_fall"]
            t.channels["r_radial"] = ChannelData(
                reward=r_radial[:T].astype(np.float32),
                is_terminated=ref.is_terminated,
                actor_weight=(
                    self.r_radial_actor_weight
                    * out_zone[:T] * stand2[:T]
                ).astype(np.float32),
            )
            t.channels["r_tangential"] = ChannelData(
                reward=r_tangential[:T].astype(np.float32),
                is_terminated=ref.is_terminated,
                actor_weight=(
                    self.r_tangential_actor_weight
                    * out_zone[:T] * stand2[:T]
                ).astype(np.float32),
            )
            t.channels["r_face"] = ChannelData(
                reward=r_face[:T].astype(np.float32),
                is_terminated=ref.is_terminated,
                actor_weight=(
                    self.r_face_actor_weight
                    * dist_gate[:T] * stand2[:T]
                ).astype(np.float32),
            )
        return trajs

    def build_trajectories(self, episodes) -> List[Trajectory]:
        """Learner-only trajectories — the opponent is script-driven."""
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            agent_id = str(
                episode.episode_options.get("agent_id", "robot_a")
            )
            foot_key, phi_key = self._AGENT_OBS_MAP[agent_id]
            all_trajs.extend(self._build_agent_trajectory(
                episode, agent_id, foot_key, phi_key,
            ))
        if not getattr(self, "_ef_verified", False):
            self._ef_verified = True
            self._verify_explore_factor_flow(all_trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Jobs — per-seed learner side + opponent speed band (NO impulse)
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        smax = self.current_speed
        smin = self.SPEED_MIN_FRAC * smax
        env_bps: Dict[str, Any] = {
            aid: self._env_pb().materialize(
                max_steps=self.max_steps,
                agent_id=aid,
                oppo_agent_id=(
                    "robot_b" if aid == "robot_a" else "robot_a"
                ),
                random_move_speed=smax,
                random_move_speed_min=smin,
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
    # Eval — standing/stepping diagnostics + hold/facing + speed level
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        success_count = 0
        all_falls = []
        all_cycles = []
        all_swings = []
        all_alt = []
        wall_frames = 0
        lean_frames = 0
        wall_total = 0
        hold_ratios: List[float] = []
        facing_ratios: List[float] = []
        dist_means: List[float] = []

        from baseline.humanoid21.end2end.stepping_state_machine import (
            count_falls,
        )
        from baseline.humanoid21.end2end.wall_contact_observer import (
            sustained_wall_mask,
        )

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            agent_id = str(
                ep.episode_options.get("agent_id", "robot_a")
            )
            foot_key, phi_key = self._AGENT_OBS_MAP[agent_id]
            phi = extract_per_step_field(
                ep.observer_outputs, phi_key, "potential", T,
            )
            if phi is not None and len(phi) > 0:
                mx = float(np.max(phi))
                all_falls.append(count_falls(phi))
            else:
                mx = 0.0
            max_pots.append(mx)
            if mx >= 0.9:
                success_count += 1

            # Passive stepping diagnostics
            contact_l = extract_per_step_field(
                ep.observer_outputs, foot_key, "left_foot_contact", T)
            contact_r = extract_per_step_field(
                ep.observer_outputs, foot_key, "right_foot_contact", T)
            h_left = extract_per_step_field(
                ep.observer_outputs, foot_key, "h_left_foot", T)
            h_right = extract_per_step_field(
                ep.observer_outputs, foot_key, "h_right_foot", T)
            if (
                contact_l is not None and contact_r is not None
                and h_left is not None and h_right is not None
                and phi is not None
            ):
                det = detect_step_cycles(
                    np.asarray(contact_l, dtype=bool),
                    np.asarray(contact_r, dtype=bool),
                    np.asarray(h_left, dtype=np.float32),
                    np.asarray(h_right, dtype=np.float32),
                    standing=np.asarray(phi) >= self.step_phi_gate,
                    min_air_steps=self.step_min_air_frames,
                    h_thresh=self.step_lift_threshold,
                )
                all_cycles.append(
                    det["n_cycles_left"] + det["n_cycles_right"])
                all_swings.append(det["n_swings"])
                if det["alt_ratio"] is not None:
                    all_alt.append(det["alt_ratio"])

            # Wall-lean accounting
            wk = self._WALL_OBS.get(agent_id)
            if wk is not None:
                wc = extract_per_step_field(
                    ep.observer_outputs, wk, "wall_contact", T,
                )
                if wc is not None:
                    wall_frames += int(np.asarray(wc, dtype=bool).sum())
                    lean_frames += int(sustained_wall_mask(
                        wc, window=self.wall_window,
                        count_thresh=self.wall_count_thresh,
                    ).sum())
                    wall_total += len(wc)

            # Follow metrics on the learner's standing frames
            app_key = self._APPROACH_OBS[agent_id]
            face_key = self._FACE_OBS[agent_id]
            self_x = extract_per_step_field(
                ep.observer_outputs, app_key, "self_x", T)
            self_y = extract_per_step_field(
                ep.observer_outputs, app_key, "self_y", T)
            opp_x = extract_per_step_field(
                ep.observer_outputs, app_key, "opp_x", T)
            opp_y = extract_per_step_field(
                ep.observer_outputs, app_key, "opp_y", T)
            if self_x is None or opp_x is None or phi is None:
                continue
            standing = np.asarray(phi, dtype=np.float32) >= 0.5
            if not standing.any():
                continue
            dist = np.linalg.norm(
                np.stack([opp_x, opp_y], axis=1)
                - np.stack([self_x, self_y], axis=1),
                axis=1,
            )
            dist_s = dist[standing]
            hold_ratios.append(float(np.mean(dist_s <= self.HOLD_DIST)))
            dist_means.append(float(np.mean(dist_s)))

            fwd_x = extract_per_step_field(
                ep.observer_outputs, face_key, "forward_x", T)
            fwd_y = extract_per_step_field(
                ep.observer_outputs, face_key, "forward_y", T)
            in_face = standing & (dist < self.D_FACE)
            if fwd_x is not None and fwd_y is not None and in_face.any():
                to_opp = (
                    np.stack([opp_x, opp_y], axis=1)
                    - np.stack([self_x, self_y], axis=1)
                )
                to_opp_hat = to_opp / np.maximum(
                    np.linalg.norm(to_opp, axis=1), 1e-6,
                )[:, None]
                cos = np.sum(
                    np.stack([fwd_x, fwd_y], axis=1) * to_opp_hat, axis=1,
                )
                facing_ratios.append(float(np.mean(cos[in_face] > 0.5)))

        n = max(len(max_pots), 1)
        success_rate = success_count / n
        falls_mean = sum(all_falls) / max(len(all_falls), 1) if all_falls else 0.0
        cycles_mean = sum(all_cycles) / max(len(all_cycles), 1) if all_cycles else 0.0
        swings_mean = sum(all_swings) / max(len(all_swings), 1) if all_swings else 0.0
        alt_mean = sum(all_alt) / max(len(all_alt), 1) if all_alt else 0.0
        wall_rate = wall_frames / max(wall_total, 1)
        lean_rate = lean_frames / max(wall_total, 1)
        hold = float(np.mean(hold_ratios)) if hold_ratios else 0.0
        facing = float(np.mean(facing_ratios)) if facing_ratios else 0.0
        mean_dist = float(np.mean(dist_means)) if dist_means else 0.0
        self._success_rate = success_rate
        self._hold_ratio = hold
        self._facing_ratio = facing

        # --- Speed-level promotion (hold ≥ 0.5 × patience) ---
        prev_level = self._speed_level
        if self._speed_level < len(self.LEVEL_SPEEDS) - 1:
            if hold >= self.PROMOTE_HOLD_RATIO:
                self._f_consecutive_pass += 1
                if self._f_consecutive_pass >= self.FOLLOW_PROMOTE_PATIENCE:
                    self._speed_level += 1
                    self._f_consecutive_pass = 0
            else:
                self._f_consecutive_pass = 0
        promoted = self._speed_level > prev_level

        quality = 10.0 * self._speed_level + hold + 0.2 * facing

        if self._f_quality_ema is None:
            self._f_quality_ema = quality
        else:
            self._f_quality_ema = (
                0.85 * self._f_quality_ema + 0.15 * quality
            )
        improved = (
            success_rate >= 0.9 and quality > self._f_best_quality
        )
        if improved:
            self._f_best_quality = quality
        trend_improved = (
            success_rate >= 0.9
            and self._f_quality_ema > self._f_best_quality
        )
        if trend_improved:
            self._f_best_quality = self._f_quality_ema
        if improved or trend_improved or promoted:
            self._f_last_best_update = update
        if self._f_last_best_update < 0:
            self._f_last_best_update = update
        improved_pot = sum(max_pots) / n > self._best_potential
        if improved_pot:
            self._best_potential = sum(max_pots) / n

        no_improvement = update - self._f_last_best_update
        stop_training = (
            no_improvement >= self._no_improvement_limit
            and update >= self._min_updates
        )
        is_new_best = improved or improved_pot or promoted

        return {
            "is_new_best": is_new_best,
            "stop_training": stop_training,
            "info": {
                "max_pot": round(sum(max_pots) / n, 3),
                "success": round(success_rate, 3),
                "falls": round(falls_mean, 2),
                "wallc": round(wall_rate, 4),
                "wallln": round(lean_rate, 4),
                "cycles": round(cycles_mean, 2),
                "swings": round(swings_mean, 2),
                "alt": round(alt_mean, 3) if all_alt else None,
                "speed": round(self.current_speed, 2),
                "hold": round(hold, 3),
                "facing": round(facing, 3),
                "dist": round(mean_dist, 3),
                "slevel": float(self._speed_level),
            },
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            **super().state(),
            "speed_level": self._speed_level,
            "f_consecutive_pass": self._f_consecutive_pass,
            "hold_ratio": self._hold_ratio,
            "facing_ratio": self._facing_ratio,
            "f_best_quality": self._f_best_quality,
            "f_quality_ema": (
                self._f_quality_ema
                if self._f_quality_ema is not None else -1.0
            ),
        }

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._f_best_quality = -1.0
        self._f_quality_ema = None
        self._f_last_best_update = -1


EXPERIMENT_CLASS = FollowPure
