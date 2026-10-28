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
from mocap.point5.contracts import Point5State
from mocap.point5.punch_planner import (
    PunchPlannerConfig, plan_punch)
from mocap.point5.contracts import StaticStartResolver
from mocap.point5.trajectory import build_reference

from .exp_hier_s2_balance_mbs import HierS2BalanceMBS

_ROOT = Path(__file__).resolve().parents[2]
_DT = 0.05      # seconds per action step

_PARTS = ("strike_hand", "guard_hand", "head", "feet")


class PunchP5MBS(HierS2BalanceMBS):
    """Stage: single P5-guided punch, 66-dim, warm-start s42i_b."""

    name = "punch_p5_mbs"
    obs_dim = 66
    max_steps = 12

    _P5_OBS = {"robot_a": "point5_a", "robot_b": "point5_b"}

    _channel_names = (
        "r_strike_hand", "r_guard_hand", "r_head", "r_feet", "r_fall",
    )
    _channel_gammas = {k: 0.99 for k in _channel_names}

    _AW = {"strike_hand": 3.0, "guard_hand": 1.0, "head": 1.0,
           "feet": 1.5, "fall": 2.0}
    _SIG = {"strike_hand": 0.05, "guard_hand": 0.06, "head": 0.08,
            "feet": 0.05}
    _PLAN = PunchPlannerConfig()

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
                    rng.uniform(0.35, 0.85))
                job.episode_options[f"{rid}_strike_lat"] = float(
                    rng.uniform(-0.25, 0.25))
                job.episode_options[f"{rid}_strike_z"] = float(
                    rng.uniform(0.6, 1.5))
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
        for tr in trajs:
            T = tr.obs.shape[0]
            scores = {p: np.zeros(T, np.float32) for p in _PARTS}
            t_task = 0.0
            if pos is not None and target is not None and len(pos) >= 2:
                pos = np.asarray(pos).reshape(len(pos), 5, 3)
                try:
                    st0 = Point5State(pos=pos[0].astype(np.float64))
                    plan = plan_punch(st0, np.asarray(target), hand,
                                      self._PLAN)
                except Exception:
                    plan = None
                if plan is not None and plan.reachable:
                    ref = build_reference(plan.task, StaticStartResolver(
                        st0, np.zeros(3), np.eye(3)))
                    t_task = ref.duration
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
                    for t in range(min(T, len(pos))):
                        t_s = t * _DT
                        if t_s > t_task:
                            break
                        ref_pos = ref.sample_time(t_s).pos
                        for part, idxs in mp.items():
                            d2 = sum(
                                np.sum((pos[t][i] - ref_pos[i]) ** 2)
                                for i in idxs)
                            scores[part][t] = np.exp(
                                -d2 / (self._SIG[part] ** 2))
            win = np.arange(T) * _DT <= t_task + 1e-9
            ref_ch = tr.channels.get("r_fall")
            term = bool(ref_ch.is_terminated) if ref_ch is not None else False
            fall_rew = (np.asarray(ref_ch.reward, dtype=np.float32)
                        if ref_ch is not None else np.zeros(T, np.float32))
            for part in _PARTS:
                key = f"r_{part}"
                tr.channels[key] = ChannelData(
                    reward=scores[part], is_terminated=term,
                    actor_weight=(self._AW[part] * win).astype(np.float32))
            tr.channels["r_fall"] = ChannelData(
                reward=fall_rew, is_terminated=term,
                actor_weight=(self._AW["fall"] * win).astype(np.float32))
        return trajs


EXPERIMENT_CLASS = PunchP5MBS
