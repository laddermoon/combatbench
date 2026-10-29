"""P5 punch skill — 66-dim, warm-start from hier_s2_balance_s42i_b.

Per-episode flow (see hier_punch_p5_env.yaml):
    PolicyInitStatePlugin rolls the S2-guard policy in an internal sim
    -> injects a fresh guard stance -> StrikeTargetPlugin resolves the
    job's (hand, dist, lat, z) params into a world-fixed target and
    issues the v2 effector command -> the rollout records measured P5.

Offline per episode this class plans the punch (plan_punch) and scores
it DeepMimic-style, split into per-part channels:

    r_strike_hand  punch-hand tracking          aw 3.0
    r_guard_hand   guard-hand tracking          aw 1.0
    r_head         head / COM proxy             aw 1.0
    r_feet         both feet (anchor = hard)    aw 1.5
    r_fall         fall + wall-lean             aw 2.0 (task window only)

All strike weights live ONLY inside the task window [0, T_task]; after
the plan finishes the channels are silent — post-completion state
(including falls) is free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from point5.contracts import Point5State, StaticStartResolver
from point5.punch_planner import (
    PunchPlannerConfig, plan_punch)
from point5.trajectory import build_reference

from .exp_hier_s2_balance_mbs import HierS2BalanceMBS

_ROOT = Path(__file__).resolve().parents[2]
_DT = 0.05      # seconds per action step

_PARTS = ("strike_hand", "guard_hand", "head", "feet")


class PunchP5MBS(HierS2BalanceMBS):
    """Stage: single P5-guided punch, 66-dim, warm-start s42i_b."""

    name = "punch_p5_mbs"
    obs_dim = 66
    # Eight 50-ms actions cover the 0.40-s plan; no unscored tail.
    max_steps = 8

    _P5_OBS = {"robot_a": "point5_a", "robot_b": "point5_b"}

    _channel_names = (
        "r_strike_hand", "r_guard_hand", "r_head", "r_feet", "r_fall",
    )
    _channel_gammas = {k: 0.99 for k in _channel_names}

    # Keep the guard while loading; prioritize the fist only once the
    # advancing foot should have landed.
    _AW_LOAD = {"strike_hand": 0.75, "guard_hand": 1.0, "head": 1.0,
                "feet": 3.0, "fall": 2.0}
    _AW_STRIKE = {"strike_hand": 6.0, "guard_hand": 0.75, "head": 0.75,
                  "feet": 1.5, "fall": 2.0}
    _SIG = {"strike_hand": 0.05, "guard_hand": 0.06, "head": 0.08,
            "feet": 0.05}
    _PLAN = PunchPlannerConfig()
    _best_strike_success = -1.0
    _best_strike_error = float("inf")

    # --- env / actor --------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint)
        return ParameterizedEnvBlueprint.load(
            _ROOT / "baseline" / "humanoid21" / "hierarchy"
            / "blueprints" / "hier_punch_p5_env.yaml")

    # --- jobs ----------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        jobs = super().build_jobs(
            policy_bp, base_seed, n_episodes, update=update,
            stochastic=stochastic)
        rng = np.random.default_rng(base_seed + 777)
        for job in jobs:
            for rid in self._AGENT_IDS:
                job.episode_options[f"{rid}_strike_hand"] = (
                    "left" if rng.random() < 0.5 else "right")
                job.episode_options[f"{rid}_strike_dist"] = float(
                    rng.uniform(0.35, 0.60))
                job.episode_options[f"{rid}_strike_lat"] = float(
                    rng.uniform(-0.25, 0.25))
                job.episode_options[f"{rid}_strike_z"] = float(
                    rng.uniform(0.85, 1.45))
        return jobs

    # --- trajectories ---------------------------------------------------

    def _build_agent_trajectory(self, episode, agent_id, foot_key,
                                phi_key):
        trajs = super()._build_agent_trajectory(
            episode, agent_id, foot_key, phi_key)
        em = dict(getattr(episode, "episode_metrics", None) or {})
        p5_key = self._P5_OBS.get(agent_id)
        oo = getattr(episode, "observer_outputs", None) or {}
        pos = None
        if p5_key and trajs:
            Tep = trajs[0].obs.shape[0]
            cols = [extract_per_step_field(oo, p5_key, f"p5_{i}", Tep)
                    for i in range(15)]
            if all(c is not None for c in cols):
                pos = np.stack([np.asarray(c) for c in cols], axis=1)
        txyz = [em.get(f"{agent_id}_strike_t{ax}") for ax in "xyz"]
        target = (np.asarray(txyz, dtype=np.float64)
                  if all(v is not None for v in txyz) else None)
        hand = ("left" if em.get(f"{agent_id}_strike_hand_i", 0.0) < 0.5
                else "right")
        start_xyz = [em.get(f"{agent_id}_strike_p5_{i}") for i in range(15)]
        if trajs and (pos is None or target is None
                      or any(v is None for v in start_xyz)):
            raise ValueError(
                f"Missing P5 observations or punch start/target for {agent_id}")
        for tr in trajs:
            T = tr.obs.shape[0]
            scores = {p: np.zeros(T, np.float32) for p in _PARTS}
            t_task = 0.0
            active_steps = 0
            if len(pos) >= 2:
                pos = np.asarray(pos).reshape(len(pos), 5, 3)
                st0 = Point5State(
                    pos=np.asarray(start_xyz, dtype=np.float64).reshape(5, 3))
                plan = plan_punch(st0, target, hand, self._PLAN)
                if plan.reachable:
                    ref = build_reference(plan.task, StaticStartResolver(
                        st0, np.zeros(3), np.eye(3)))
                    t_task = ref.duration
                    active_steps = min(T, int(np.ceil((t_task - 1e-9) / _DT)))
                    pn = {p: i for i, p in
                          enumerate(("head", "hand_right", "hand_left",
                                     "foot_right", "foot_left"))}
                    hp_idx = pn[f"hand_{hand}"]
                    hg_idx = pn["hand_right" if hand == "left"
                                else "hand_left"]
                    mp = {
                        "strike_hand": [hp_idx],
                        "guard_hand": [hg_idx],
                        "head": [pn["head"]],
                        "feet": [pn["foot_left"], pn["foot_right"]],
                    }
                    for t in range(min(active_steps, len(pos))):
                        # Observer frame t is measured AFTER action t.
                        # The final action crosses the impact time: score
                        # that post-action pose against the impact keyframe.
                        t_s = min((t + 1) * _DT, t_task)
                        ref_pos = ref.sample_time(t_s).pos
                        for part, idxs in mp.items():
                            d2 = sum(
                                np.sum((pos[t][i] - ref_pos[i]) ** 2)
                                for i in idxs)
                            sig2 = self._SIG[part] ** 2
                            # Narrow precision peak plus a broad tail:
                            # the old pure Gaussian was ~0 at 15cm foot
                            # or 30cm fist error, giving no recovery signal.
                            scores[part][t] = (
                                0.5 * np.exp(-d2 / sig2)
                                + 0.5 / (1.0 + d2 / (9.0 * sig2)))
            win = np.arange(T) < active_steps
            ref_ch = tr.channels.get("r_fall")
            # A single punch is a finite task, not a truncated continuing
            # balance episode: there is no value to bootstrap after impact.
            term = True
            fall_rew = (np.asarray(ref_ch.reward, dtype=np.float32)
                        if ref_ch is not None else np.zeros(T, np.float32))
            loading = (np.arange(T) + 1) * _DT <= self._PLAN.dur_load
            for part in _PARTS:
                key = f"r_{part}"
                aw = np.where(loading, self._AW_LOAD[part],
                              self._AW_STRIKE[part])
                tr.channels[key] = ChannelData(
                    reward=scores[part], is_terminated=term,
                    actor_weight=(aw * win).astype(np.float32))
            fall_aw = np.where(loading, self._AW_LOAD["fall"],
                               self._AW_STRIKE["fall"])
            tr.channels["r_fall"] = ChannelData(
                reward=fall_rew, is_terminated=term,
                actor_weight=(fall_aw * win).astype(np.float32))
        return trajs

    def on_eval(self, episodes, update):
        """Evaluate the punch, not the inherited push-recovery curriculum."""
        attempts = successes = 0
        errors = []
        for ep in episodes:
            metrics = ep.episode_metrics
            for rid, obs_name in self._P5_OBS.items():
                xyz = [metrics.get(f"{rid}_strike_t{ax}") for ax in "xyz"]
                if any(v is None for v in xyz):
                    continue
                hand_idx = (2 if metrics.get(f"{rid}_strike_hand_i", 0)
                            < 0.5 else 1)
                t_full = ep.num_frames
                duration = (self._PLAN.dur_load + self._PLAN.dur_reach
                            + self._PLAN.dur_snap)
                hit_idx = min(round(duration / _DT) - 1,
                              ep.agent_frame_boundary.get(rid, t_full) - 1)
                if hit_idx < 0:
                    continue
                cols = [extract_per_step_field(
                    ep.observer_outputs, obs_name,
                    f"p5_{3 * hand_idx + i}", t_full) for i in range(3)]
                if any(c is None for c in cols):
                    continue
                phi = extract_per_step_field(
                    ep.observer_outputs,
                    "standing_balance_a" if rid == "robot_a"
                    else "standing_balance_b", "potential", t_full)
                fell = (phi is not None and
                        np.any(np.asarray(phi[:hit_idx + 1]) < 0.5))
                err = float(np.linalg.norm(
                    [c[hit_idx] - v for c, v in zip(cols, xyz)]))
                attempts += 1
                successes += err < 0.10 and not fell
                errors.append(err)
        rate = successes / attempts if attempts else 0.0
        mean_err = float(np.mean(errors)) if errors else float("inf")
        improved = attempts > 0 and (
            rate > self._best_strike_success
            or (rate == self._best_strike_success
                and mean_err < self._best_strike_error))
        if improved:
            self._best_strike_success = rate
            self._best_strike_error = mean_err
        return {
            "is_new_best": improved,
            "stop_training": False,
            "info": {
                "strike_success": round(rate, 3),
                "impact_hand_error": round(mean_err, 3) if errors else None,
                "strike_attempts": attempts,
            },
        }


EXPERIMENT_CLASS = PunchP5MBS
