"""Pure balance recovery — no stepping machinery (new baseline line).

New-route stage 2 (per bootstrip.md "新思路"): warm-start from the
standup checkpoint and learn to survive curriculum-scheduled torso
pushes WITHOUT any gait scaffolding:

  * no GaitClockSimulator — plain Humanoid21Simulator, obs is the
    canonical 96 dims (native standup-ckpt + platform contract fit);
  * no foot reward channels, no clock gating, no alternation token,
    no step-cycle bonus, no swing exemption, no lift_phi_gate;
  * no σ×2.0 standing exploration boost (that existed to *discover*
    stepping — BalanceStep's σ×1.0 standing / σ×0.5 low stays).

Hypothesis: under strong pushes a static-bracing policy falls — the
only way to keep collecting r_potential and avoid r_fall is to step.
If stepping emerges, it arrives as a *means* of balance recovery,
driven purely by the task; the step/balance_step machinery stays as
the alternate historical route.

Channels (3):
    r_potential = 0.01 × φ(t),        aw = 3.0 (fixed)
    r_fall      = −0.2 × fall_onset
                  − wall_penalty × sustained-lean(standing)
                  + push_clean_bonus × clean PUSHING/OBSERVE frame,
                                      aw = 2.0 (fixed)
    r_cross     = CrossSupportBalanceRewarder alternation timing
                  (guidance per user direction — penalty-only
                  alternation constraint, not gait shaping),
                                      aw = 1.0 × φ² (standing-only)

Foot-state observations are still collected (eval only) so cycles /
swings / alt report whether stepping emerged — they feed nothing.

Curriculum: same 12 push levels as balance_step (40N→100N, promotion
at eval recovery_rate ≥ 0.8 for PROMOTE_PATIENCE consecutive evals).
quality = 10·level + recovery_rate (+ tiny gait diagnostics kept out
of the quality so best-tracking tracks balance, not gait style).

Blueprint: baseline/humanoid21/end2end/balance_pure_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import (
    extract_per_step_field,
    extract_per_step_scalar,
)
from baseline.humanoid21.end2end.stepping_state_machine import (
    count_falls,
    detect_step_cycles,
    fall_onset_mask,
)

from .base import CombatExperimentPPOBase
from .exp_balance_step import _balance_explore_factor, _EF_LOW, _H_EF_SWITCH


class BalancePure(CombatExperimentPPOBase):
    """Push-recovery curriculum on a step-free reward recipe."""

    name = "balance_pure"

    # --- Network: canonical 96-dim obs (no gait clock) ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = ("r_potential", "r_fall", "r_cross")
    _channel_gammas = {
        "r_potential": 0.99, "r_fall": 0.99, "r_cross": 0.99,
    }
    _gae_lambda: float = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01
    fall_penalty: float = 0.2
    r_potential_actor_weight: float = 3.0
    r_fall_actor_weight: float = 2.0
    # Cross-support alternation guidance (CrossSupportBalanceRewarder):
    # aw = coef × φ² — inactive while low, full pressure once standing.
    r_cross_actor_weight: float = 1.0
    _CROSS_OBS = {
        "robot_a": "cross_support_a", "robot_b": "cross_support_b",
    }

    # --- Wall-lean penalty (folded into r_fall, upright frames only) ---
    wall_penalty: float = 0.03
    wall_window: int = 40
    wall_count_thresh: int = 10
    _WALL_OBS = {"robot_a": "wall_contact_a", "robot_b": "wall_contact_b"}

    # --- Push clean-window survival bonus (folded into r_fall) ---
    push_clean_bonus: float = 0.02
    _PUSH_OBS = {"robot_a": "push_state_a", "robot_b": "push_state_b"}

    # --- Curriculum: 12 levels (same as balance_step) ---
    LEVEL_FORCES: Tuple[float, ...] = (
        40.0, 40.0, 40.0, 40.0,
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    )
    LEVEL_DURATION_RANGES: Tuple[Tuple[int, int], ...] = (
        (1, 10), (11, 20), (21, 30), (31, 40),
        (1, 5), (6, 10), (11, 15), (16, 20),
        (21, 25), (26, 30), (31, 35), (36, 40),
    )
    PROMOTE_RECOVERY_RATE: float = 0.8
    PROMOTE_PATIENCE: int = 2

    # --- Eval stepping diagnostics (passive — measure emergence only) ---
    step_lift_threshold: float = 0.05
    step_phi_gate: float = 0.9
    step_min_air_frames: int = 3

    # --- Env ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"
    max_steps: int = 400

    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a"),
        ("robot_b", "foot_state_b", "standing_balance_b"),
    )
    _AGENT_IDS = ("robot_a", "robot_b")

    # --- PPO tuning (same conservative warm-start recipe as Step) ---
    learning_rate: float = 3e-5
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.03
    update_epochs: int = 4
    minibatch_size: int = 4096
    uncertainty_floor: float = 0.4
    uncertainty_coef: float = 1.0

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    max_updates: int = 5000
    eval_interval: int = 5
    eval_episodes: int = 64
    video_eval_interval: int = 5

    # --- Early stop: finite curriculum (level 11 is the goal) so the
    # mechanism stays, but generous — intervene manually if it stalls.
    _no_improvement_limit: int = 400
    _min_updates: int = 800

    # --- Stateful metrics ---
    _best_potential: float = -1.0
    _b_best_quality: float = -1.0
    _b_quality_ema: float | None = None
    _b_last_best_update: int = -1
    _level: int = 0
    _consecutive_pass: int = 0
    _recovery_rate: float = 1.0
    _success_rate: float = 0.0

    # ------------------------------------------------------------------
    # Blueprint / models
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "balance_pure_env.yaml"
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

    @property
    def current_force(self) -> float:
        idx = max(0, min(self._level, len(self.LEVEL_FORCES) - 1))
        return float(self.LEVEL_FORCES[idx])

    @property
    def current_duration_range(self) -> Tuple[int, int]:
        idx = max(0, min(self._level, len(self.LEVEL_DURATION_RANGES) - 1))
        return self.LEVEL_DURATION_RANGES[idx]

    # ------------------------------------------------------------------
    # Trajectory building — minimal recipe
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

        # --- r_fall: fall-onset penalty + wall-lean + clean-push bonus ---
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

        # --- r_cross: alternation-timing guidance, φ²-gated aw ---
        cross_key = self._CROSS_OBS.get(agent_id)
        r_cross = extract_per_step_scalar(
            episode.observer_outputs, cross_key, T_full,
        ) if cross_key is not None else None
        if r_cross is None:
            r_cross = np.zeros(T, dtype=np.float32)
        else:
            r_cross = np.asarray(r_cross[:T], dtype=np.float32)

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
                    self.r_cross_actor_weight * phi_arr ** 2
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

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, foot_key, phi_key in self._AGENT_OBS:
                all_trajs.extend(
                    self._build_agent_trajectory(
                        episode, agent_id, foot_key, phi_key,
                    )
                )
        if not getattr(self, "_ef_verified", False):
            self._ef_verified = True
            self._verify_explore_factor_flow(all_trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Jobs — initial_distance + phase ef + impulse_params curriculum
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        env_bp = self._env_pb().materialize(max_steps=self.max_steps)
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=_balance_explore_factor)
        force = self.current_force
        dur_min, dur_max = self.current_duration_range
        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            impulse_params = {
                rid: {
                    "force": force,
                    "duration_min": dur_min,
                    "duration_max": dur_max,
                    "body": "torso",
                    "seed": seed,
                }
                for rid in self._AGENT_IDS
            }
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=policy_bp,
                env_bp=env_bp,
                seed=seed,
                episode_options={
                    "initial_distance": initial_distance,
                    "impulse_params": impulse_params,
                },
                sampling_a=sampling,
                sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    def _verify_explore_factor_flow(self, trajs) -> None:
        if not trajs:
            return
        t0 = trajs[0]
        ef = (t0.sampling_ctx or {}).get("explore_factor")
        if ef is None:
            print("  [ef-verify] FAIL: sampling_ctx has no explore_factor",
                  flush=True)
            return
        ef = np.asarray(ef, dtype=np.float32)
        h_obs = np.asarray(t0.obs, dtype=np.float32)[:, 45]
        standing = h_obs >= _H_EF_SWITCH
        ok_low = bool(np.all(ef[~standing] < 0)) if (~standing).any() else True
        ok_stand = bool(np.all(ef[standing] == 0.0)) if standing.any() else True
        print(
            f"  [ef-verify-balance] T={len(ef)} ef_min={ef.min():.3f} "
            f"ef_max={ef.max():.3f} | low={int((~standing).sum())} ef<0: "
            f"{'OK' if ok_low else 'FAIL'} | stand={int(standing.sum())} "
            f"ef==0: {'OK' if ok_stand else 'FAIL'}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Eval — standing/recovery + passive stepping diagnostics + curriculum
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        success_count = 0
        all_falls = []
        all_cycles = []
        all_swings = []
        all_alt = []
        total_push = 0
        total_fall = 0
        wall_frames = 0
        lean_frames = 0
        wall_total = 0

        from baseline.humanoid21.end2end.wall_contact_observer import (
            sustained_wall_mask,
        )

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for agent_id, foot_key, phi_key in self._AGENT_OBS:
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

                # Passive stepping diagnostics — did stepping emerge?
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

            em = dict(getattr(ep, "episode_metrics", None) or {})
            for rid in self._AGENT_IDS:
                total_push += int(em.get(f"{rid}_push_count", 0))
                total_fall += int(em.get(f"{rid}_fall_count", 0))
                wk = self._WALL_OBS.get(rid)
                if wk is not None and ep.num_frames > 0:
                    wc = extract_per_step_field(
                        ep.observer_outputs, wk, "wall_contact",
                        ep.num_frames,
                    )
                    if wc is not None:
                        wall_frames += int(np.asarray(wc, dtype=bool).sum())
                        lean_frames += int(sustained_wall_mask(
                            wc, window=self.wall_window,
                            count_thresh=self.wall_count_thresh,
                        ).sum())
                        wall_total += len(wc)

        n = max(len(max_pots), 1)
        success_rate = success_count / n
        falls_mean = sum(all_falls) / max(len(all_falls), 1) if all_falls else 0.0
        cycles_mean = sum(all_cycles) / max(len(all_cycles), 1) if all_cycles else 0.0
        swings_mean = sum(all_swings) / max(len(all_swings), 1) if all_swings else 0.0
        alt_mean = sum(all_alt) / max(len(all_alt), 1) if all_alt else 0.0
        wall_rate = wall_frames / max(wall_total, 1)
        lean_rate = lean_frames / max(wall_total, 1)
        recovery_rate = (
            float(1.0 - total_fall / total_push) if total_push > 0 else 1.0
        )
        self._recovery_rate = recovery_rate
        self._success_rate = success_rate

        # --- Curriculum promotion ---
        prev_level = self._level
        if self._level < len(self.LEVEL_FORCES) - 1:
            if recovery_rate >= self.PROMOTE_RECOVERY_RATE and total_push > 0:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0
        promoted = self._level > prev_level

        # Composite quality: level dominates, recovery breaks ties.
        quality = 10.0 * self._level + recovery_rate

        if self._b_quality_ema is None:
            self._b_quality_ema = quality
        else:
            self._b_quality_ema = 0.85 * self._b_quality_ema + 0.15 * quality
        improved = (
            success_rate >= 0.9 and quality > self._b_best_quality)
        if improved:
            self._b_best_quality = quality
        trend_improved = (
            success_rate >= 0.9
            and self._b_quality_ema > self._b_best_quality
        )
        if trend_improved:
            self._b_best_quality = self._b_quality_ema
        if improved or trend_improved or promoted:
            self._b_last_best_update = update
        if self._b_last_best_update < 0:
            self._b_last_best_update = update
        improved_pot = sum(max_pots) / n > self._best_potential
        if improved_pot:
            self._best_potential = sum(max_pots) / n

        no_improvement = update - self._b_last_best_update
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
                "recovery": round(recovery_rate, 3),
                "pushes": total_push,
                "push_falls": total_fall,
                "level": float(self._level),
                "force": round(self.current_force, 1),
                "wallc": round(wall_rate, 4),
                "wallln": round(lean_rate, 4),
                # Passive stepping-emergence diagnostics (not rewarded)
                "cycles": round(cycles_mean, 2),
                "swings": round(swings_mean, 2),
                "alt": round(alt_mean, 3) if all_alt else None,
            },
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            **super().state(),
            "level": self._level,
            "consecutive_pass": self._consecutive_pass,
            "recovery_rate": self._recovery_rate,
            "b_best_quality": self._b_best_quality,
            "b_quality_ema": (
                self._b_quality_ema if self._b_quality_ema is not None
                else -1.0
            ),
            "best_potential": self._best_potential,
        }

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._level = int(state.get("level", 0))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._recovery_rate = float(state.get("recovery_rate", 1.0))
        self._best_potential = float(state.get("best_potential", -1.0))
        self._b_best_quality = -1.0
        self._b_quality_ema = None
        self._b_last_best_update = -1


EXPERIMENT_CLASS = BalancePure
