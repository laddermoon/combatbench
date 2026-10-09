"""Follow + face stage on top of the converged balance-step policy.

Warm-start from ``balance_step`` (s42i, level 11/11, recovery 0.95).
The learner keeps the full standup→step→balance stack — random fallen
init, all five reward channels, wall-lean penalty, clean push-window
bonus — while a scripted target (``RandomMovePlugin``) roams the
arena at a curriculum-controlled speed.  Three new reward channels
(adapted from the archived ``exp_follow_v2`` recipe, follow+face
merged per the end2end README):

  r_radial      — smoothed velocity toward the opponent (+approach /
                  −retreat), aw = 3.0 × out_zone × φ²
  r_tangential  — lateral drift penalty, aw = 1.0 × out_zone × φ²
  r_face        — max(0, cos(forward, to_opp)), aw = 1.0 × dist_gate × φ²

All gates live in actor_weight (critics always see the raw reward):
follow rewards only fire when upright (φ²) and outside the hold
radius; the face reward ramps in between D_FACE=1.5m and D_STRIKE=0.7m.

Push maintenance: the force plugin stays active on the learner at a
mid-curriculum level (100N × 11-15 steps).  The parent's recovery
promotion remains armed, so push difficulty re-ramps toward level 11
as the policy regains robustness while walking — preventing balance
forgetting without drowning the new follow objective.

Speed curriculum (same 13 levels as follow_v2): eval hold_ratio —
fraction of standing frames with dist ≤ 1.1m — ≥ 0.5 for
PROMOTE_PATIENCE consecutive evals promotes one level.

Best-of-run / early stop use a follow composite —
    quality = 10·speed_level + hold_ratio + 0.2·facing_ratio + 0.05·gait_q
recorded under the *measured* level (pre-promotion) to avoid the
cross-level misattribution bug family.

Blueprint: baseline/humanoid21/end2end/follow_step_env.yaml
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

from .exp_balance_step import BalanceStep, _balance_explore_factor


class FollowStep(BalanceStep):
    """Standup + step + balance maintenance + follow/face curriculum."""

    name = "follow_step"
    agent_used = "random"

    # Follow needs longer episodes: standup (~40) + transit + hold.
    max_steps: int = 600
    # Resumed at ~u5000 — leave headroom for the 13-level speed
    # curriculum; early stop still bounds the tail.
    max_updates: int = 12000

    # --- Follow/face channels (γ=0.99, same horizon as r_fall) ---
    _channel_names = BalanceStep._channel_names + (
        "r_radial", "r_tangential", "r_face",
    )
    _channel_gammas = {
        **BalanceStep._channel_gammas,
        "r_radial": 0.99,
        "r_tangential": 0.99,
        "r_face": 0.99,
    }

    # --- Follow/face actor weights (follow_v2 recipe) ---
    r_radial_actor_weight: float = 3.0
    r_tangential_actor_weight: float = 1.0
    r_face_actor_weight: float = 1.0
    D_FACE: float = 1.5          # face aw ramps in below this distance
    D_STRIKE: float = 0.7        # face aw fully active inside this
    HOLD_DIST: float = 1.1       # eval: dist <= this counts as "holding"

    # --- Opponent speed curriculum (follow_v2's 13 levels) ---
    LEVEL_SPEEDS: Tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 1.0, 1.2, 1.5,
    )
    PROMOTE_HOLD_RATIO: float = 0.5
    FOLLOW_PROMOTE_PATIENCE: int = 2

    # Warm-start push level: drop from the balance-final 11 to the
    # mid-curriculum maintenance setting; the parent promotion ramps
    # it back up as robustness is regained while moving.
    PUSH_LEVEL_START: int = 6

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
    # Blueprint — balance env + RandomMovePlugin + follow observers
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "follow_step_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    @property
    def current_speed(self) -> float:
        idx = max(0, min(self._speed_level, len(self.LEVEL_SPEEDS) - 1))
        return float(self.LEVEL_SPEEDS[idx])

    # ------------------------------------------------------------------
    # Trajectories — balance recipe + follow/face channels, learner only
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
            # Observers absent — keep channels zero so critics still
            # exist (channel set must match reward_channels()).
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

        # Facing score: cos(forward_xy, to_opp_xy) clipped at 0.
        to_opp_norm = np.maximum(dist, 1e-6)
        to_opp_hat = opp_xy - self_xy
        to_opp_hat = to_opp_hat / to_opp_norm[:, None]
        cos_angle = np.sum(fwd * to_opp_hat, axis=1)
        r_face = np.maximum(0.0, cos_angle).astype(np.float32)
        dist_gate = np.clip(
            (self.D_FACE - dist) / (self.D_FACE - self.D_STRIKE),
            0.0, 1.0,
        ).astype(np.float32)

        # Upright gate: φ² — follow/face gradients only fire when the
        # robot is ~fully standing; standup/recovery stays primary.
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
    # Jobs — per-seed learner side + opponent speed + learner push
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        speed = self.current_speed
        env_bps: Dict[str, Any] = {
            aid: self._env_pb().materialize(
                max_steps=self.max_steps,
                agent_id=aid,
                oppo_agent_id=(
                    "robot_b" if aid == "robot_a" else "robot_a"
                ),
                random_move_speed=speed,
            )
            for aid in self._AGENT_IDS
        }
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=_balance_explore_factor)
        force = self.current_force
        dur_min, dur_max = self.current_duration_range
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
                    "impulse_params": {
                        agent_id: {
                            "force": force,
                            "duration_min": dur_min,
                            "duration_max": dur_max,
                            "body": "torso",
                            "seed": seed,
                        },
                    },
                },
                sampling_a=sampling,
                sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Eval — gait + recovery/wall (parent) + hold/facing + speed level
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        info = result["info"]

        # --- Follow metrics on the learner's standing frames ---
        hold_ratios: List[float] = []
        facing_ratios: List[float] = []
        dist_means: List[float] = []
        for ep in episodes:
            if ep.num_frames == 0:
                continue
            agent_id = str(
                ep.episode_options.get("agent_id", "robot_a")
            )
            foot_key, phi_key = self._AGENT_OBS_MAP[agent_id]
            app_key = self._APPROACH_OBS[agent_id]
            face_key = self._FACE_OBS[agent_id]
            T_full = ep.num_frames

            self_x = extract_per_step_field(
                ep.observer_outputs, app_key, "self_x", T_full)
            self_y = extract_per_step_field(
                ep.observer_outputs, app_key, "self_y", T_full)
            opp_x = extract_per_step_field(
                ep.observer_outputs, app_key, "opp_x", T_full)
            opp_y = extract_per_step_field(
                ep.observer_outputs, app_key, "opp_y", T_full)
            phi = extract_per_step_field(
                ep.observer_outputs, phi_key, "potential", T_full)
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
                ep.observer_outputs, face_key, "forward_x", T_full)
            fwd_y = extract_per_step_field(
                ep.observer_outputs, face_key, "forward_y", T_full)
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

        hold = float(np.mean(hold_ratios)) if hold_ratios else 0.0
        facing = float(np.mean(facing_ratios)) if facing_ratios else 0.0
        mean_dist = float(np.mean(dist_means)) if dist_means else 0.0
        self._hold_ratio = hold
        self._facing_ratio = facing

        # --- Speed-level promotion (hold_ratio ≥ 0.5 × patience).
        # Quality recorded under the MEASURED level — promotion after.
        gait_q = (
            info.get("step", 0.0)
            - 0.1 * (info.get("falls") or 0.0)
            + 0.1 * (info.get("alt") or 0.0)
            * min(1.0, info.get("cycles", 0.0) / 10.0)
            + min(info.get("solepk") or 0.0, 0.10)
            + 0.005 * min(info.get("cycles", 0.0), 20.0)
        )
        quality = (
            10.0 * self._speed_level
            + hold
            + 0.2 * facing
            + 0.05 * gait_q
        )

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

        if self._f_quality_ema is None:
            self._f_quality_ema = quality
        else:
            self._f_quality_ema = (
                0.85 * self._f_quality_ema + 0.15 * quality
            )
        improved = (
            info.get("success", 0.0) >= 0.9
            and quality > self._f_best_quality
        )
        if improved:
            self._f_best_quality = quality
        trend_improved = (
            info.get("success", 0.0) >= 0.9
            and self._f_quality_ema > self._f_best_quality
        )
        if trend_improved:
            self._f_best_quality = self._f_quality_ema
        if improved or trend_improved or promoted:
            self._f_last_best_update = update
        if self._f_last_best_update < 0:
            self._f_last_best_update = update

        no_improvement = update - self._f_last_best_update
        stop_training = (
            no_improvement >= self._no_improvement_limit
            and update >= self._min_updates
        )
        is_new_best = improved or promoted

        info.update({
            "speed": round(self.current_speed, 2),
            "hold": round(hold, 3),
            "facing": round(facing, 3),
            "dist": round(mean_dist, 3),
            "slevel": float(self._speed_level),
        })
        return {
            "is_new_best": is_new_best,
            "stop_training": stop_training,
            "info": info,
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
        if "speed_level" in state:
            # Own-run resume: restore follow curriculum, re-anchor the
            # follow best/EMA (same polluted-anchor rationale as the
            # parent's _b_* reset).
            self._speed_level = int(state["speed_level"])
            self._f_consecutive_pass = int(
                state.get("f_consecutive_pass", 0))
            self._hold_ratio = float(state.get("hold_ratio", 0.0))
            self._facing_ratio = float(state.get("facing_ratio", 0.0))
        else:
            # Warm-start from a balance ckpt: drop the push curriculum
            # from the mastered level 11 to mid-curriculum maintenance;
            # the parent promotion re-ramps it as robustness returns.
            self._level = self.PUSH_LEVEL_START
            self._consecutive_pass = 0
        self._f_best_quality = -1.0
        self._f_quality_ema = None
        self._f_last_best_update = -1


EXPERIMENT_CLASS = FollowStep
