"""Stand-punch skill — 66-dim, warm-start from hier_s2_balance_s42i_b.

Self-contained experiment (inherits CombatExperimentPPOBase only — no
cross-experiment inheritance).  Per-episode flow:

    build_jobs samples a reachable (hand, local target) pair and plans
    ONCE — the serialized P5Clip travels inside episode_options ->
    StandPunchTargetPlugin retimes/resolves it into a world-fixed
    strike-hand guidance track -> the rollout records measured P5
    (world positions + torso yaw) -> this class scores the clip
    DeepMimic-style, frame-anchored at 50 ms per action.

Reward channels (all inside the 0.65 s task window, 13 actions):

    r_strike_hand  punch-hand position          aw 0.75 / 6.0
    r_guard_hand   guard-hand position          aw 1.0 / 0.75
    r_head         head / COM proxy             aw 1.0 / 0.75
    r_feet         both feet                    aw 3.0 / 1.5
    r_torso_yaw    torso facing vs frame yaw    aw 2.0 / 2.0
    r_fall         fall penalty                 aw 2.0 / 2.0

Weights split at _load_seconds (0.50 s): feet/torso lead while
loading, strike_hand dominates the impact segment.  Every channel is
terminated=True — a finite skill, no bootstrapped future.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory)
from baseline.framework.rollout import extract_per_step_field
from point5.contracts import P5Clip, Point5State
from point5.humanoid21.stand_punch import (
    StandPunchConfig, plan_stand_punch, stand_p5)

from baseline.humanoid21.hierarchy.p5_world import clip_frame_samples
from baseline.humanoid21.plugins.stand_punch_target import (
    SEGMENT_DURATIONS, timed_stand_clip)

from .base import CombatExperimentPPOBase

_ROOT = Path(__file__).resolve().parents[2]
_DT = 0.05      # seconds per action step

_PARTS = ("strike_hand", "guard_hand", "head", "feet", "torso_yaw")

# Tight reachable box from STAND_PUNCH.md (validated vs stand_p5()).
_TGT_X = (0.95, 1.30)
_TGT_Y = (-0.30, 0.30)
_TGT_Z = (0.75, 1.35)


def _fall_onset_mask(phi: np.ndarray) -> np.ndarray:
    """1.0 from the first frame the balance potential drops below 0.5."""
    phi = np.asarray(phi)
    idx = np.nonzero(phi < 0.5)[0]
    m = np.zeros(len(phi), dtype=np.float32)
    if len(idx):
        m[int(idx[0]):] = 1.0
    return m


class StandPunchMBS(CombatExperimentPPOBase):
    """Stage: standing-pose P5-guided jab, 66-dim, warm-start s42i_b."""

    name = "stand_punch_mbs"
    obs_dim = 66
    agent_used = "both"   # self-play env; both robots get a task
    actor_blueprint = "init_policy_mbs_66d.yaml"
    # 0.65 s task: stand->coil .10 / coil->swing .20 / swing->land .20 /
    # land->strike .15, thirteen 50-ms actions, no unscored tail.
    max_steps = 13

    _AGENT_IDS = ("robot_a", "robot_b")
    _P5_OBS = {"robot_a": "point5_a", "robot_b": "point5_b"}
    _BAL_OBS = {"robot_a": "standing_balance_a",
                "robot_b": "standing_balance_b"}
    # (agent_id, foot_key, phi_key) triples for build_trajectories.
    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a"),
        ("robot_b", "foot_state_b", "standing_balance_b"),
    )

    _channel_names = (
        "r_strike_hand", "r_guard_hand", "r_head", "r_feet",
        "r_torso_yaw", "r_fall",
    )
    _channel_gammas = {k: 0.99 for k in _channel_names}
    _gae_lambda = 0.95

    def reward_channels(self):
        return tuple(
            RewardChannel(name=k, gamma=self._channel_gammas[k],
                          gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    fall_penalty = 1.0
    _STAND_PLAN = StandPunchConfig()

    # Keep the guard while loading; prioritize the fist only once the
    # advancing foot should have landed.
    _AW_LOAD = {"strike_hand": 0.75, "guard_hand": 1.0, "head": 1.0,
                "feet": 3.0, "fall": 2.0, "torso_yaw": 2.0}
    _AW_STRIKE = {"strike_hand": 6.0, "guard_hand": 0.75, "head": 0.75,
                  "feet": 1.5, "fall": 2.0, "torso_yaw": 2.0}
    _SIG = {"strike_hand": 0.05, "guard_hand": 0.06, "head": 0.08,
            "feet": 0.05, "torso_yaw": 0.30}  # rad ~17 deg
    _best_strike_success = -1.0
    _best_strike_error = float("inf")

    # --- env -----------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint)
        return ParameterizedEnvBlueprint.load(
            _ROOT / "baseline" / "humanoid21" / "hierarchy"
            / "blueprints" / "hier_stand_punch_env.yaml")

    # --- jobs ----------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        jobs = super().build_jobs(
            policy_bp, base_seed, n_episodes, update=update,
            stochastic=stochastic)
        rng = np.random.default_rng(base_seed + 555)
        for job in jobs:
            for rid in self._AGENT_IDS:
                hand, tgt, clip_d = self._sample_target(rng)
                job.episode_options[f"{rid}_sp_hand"] = hand
                job.episode_options[f"{rid}_sp_clip"] = clip_d
                for a, v in zip("xyz", tgt):
                    job.episode_options[f"{rid}_sp_t{a}"] = float(v)
        return jobs

    @staticmethod
    def _sample_target(rng, max_tries: int = 64):
        """Rejection-sample a reachable (hand, target, clip) triple.

        The plan runs ONCE here — the serialized clip travels inside
        episode_options to both the command plugin and the scorer, so
        nothing downstream re-plans or silently falls back.
        """
        st0 = Point5State(pos=stand_p5())
        for _ in range(max_tries):
            hand = "left" if rng.random() < 0.5 else "right"
            tgt = np.array([rng.uniform(*_TGT_X),
                            rng.uniform(*_TGT_Y),
                            rng.uniform(*_TGT_Z)])
            plan = plan_stand_punch(tgt, hand, initial=st0)
            if plan.reachable:
                return hand, tgt, plan.task.to_dict()
        # Fallback to a known-good central target.
        tgt = np.array([1.05, 0.0, 1.10])
        return "left", tgt, plan_stand_punch(
            tgt, "left", initial=st0).task.to_dict()

    # --- reference -----------------------------------------------------

    def _plan_reference(self, st0, target_local, hand, f1,
                        clip_dict=None):
        """(world_pos (T,5,3), frame_yaws (T,)) for scoring, or None."""
        if clip_dict is None:
            return None
        clip = timed_stand_clip(P5Clip.from_dict(clip_dict), st0.pos)
        return clip_frame_samples(clip, f1, _DT)

    @property
    def _load_seconds(self) -> float:
        # Everything before the strike segment is "loading".
        return float(sum(SEGMENT_DURATIONS[:-1]))

    @property
    def _task_seconds(self) -> float:
        return float(sum(SEGMENT_DURATIONS))

    # --- trajectories --------------------------------------------------

    def build_trajectories(self, episodes) -> List[Trajectory]:
        trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, foot_key, phi_key in self._AGENT_OBS:
                trajs.extend(self._build_agent_trajectory(
                    episode, agent_id, foot_key, phi_key))
        return trajs

    def _build_agent_trajectory(self, episode, agent_id, foot_key,
                                phi_key):
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

        # --- r_fall from the balance observer's potential --------------
        phi_key = phi_key or self._BAL_OBS.get(agent_id)
        phi_arr = extract_per_step_field(
            episode.observer_outputs, phi_key, "potential", T_full)
        phi_arr = (np.clip(np.asarray(phi_arr[:T], dtype=np.float32),
                           0.0, 1.0) if phi_arr is not None
                   else np.zeros(T, dtype=np.float32))
        fall_rew = (-self.fall_penalty
                    * _fall_onset_mask(phi_arr)).astype(np.float32)

        # --- per-step P5 + yaw observation columns ---------------------
        em = dict(getattr(episode, "episode_metrics", None) or {})
        oo = getattr(episode, "observer_outputs", None) or {}
        p5_key = self._P5_OBS.get(agent_id)
        pos = yaw_col = None
        if p5_key:
            cols = [extract_per_step_field(oo, p5_key, f"p5_{i}", T_full)
                    for i in range(15)]
            if all(c is not None for c in cols):
                pos = np.stack([np.asarray(c) for c in cols],
                               axis=1).reshape(-1, 5, 3)
            yaw_col = extract_per_step_field(oo, p5_key, "p5_yaw",
                                             T_full)

        # v2: stored start P5 + strike_lt* target are LOCAL; strike_f1_*
        # is the world pose of that frame; the clip resolves to world.
        lxyz = [em.get(f"{agent_id}_strike_lt{ax}") for ax in "xyz"]
        target = (np.asarray(lxyz, dtype=np.float64)
                  if all(v is not None for v in lxyz) else None)
        f1v = [em.get(f"{agent_id}_strike_f1_{k}")
               for k in ("ox", "oy", "yaw")]
        f1 = ((np.asarray(f1v[:2], dtype=np.float64), float(f1v[2]))
              if all(v is not None for v in f1v) else None)
        hand = ("left" if em.get(f"{agent_id}_strike_hand_i", 0.0) < 0.5
                else "right")
        start_xyz = [em.get(f"{agent_id}_strike_p5_{i}")
                     for i in range(15)]
        clip_d = (getattr(episode, "episode_options", None) or {}).get(
            f"{agent_id}_sp_clip")
        if (pos is None or target is None or f1 is None or clip_d is None
                or any(v is None for v in start_xyz)):
            raise ValueError(
                f"Missing P5 observations or punch start/target for "
                f"{agent_id}")

        scores = {p: np.zeros(T, np.float32) for p in _PARTS}
        active_steps = 0
        if len(pos) >= 2:
            st0 = Point5State(
                pos=np.asarray(start_xyz, dtype=np.float64).reshape(5, 3))
            # ref_world[t] / ref_yaws[t] = state the robot should reach
            # after action t (time min((t+1)*dt, duration)).
            ref = self._plan_reference(st0, target, hand, f1,
                                       clip_dict=clip_d)
            if ref is not None:
                ref_world, ref_yaws = ref
                active_steps = min(T, len(ref_world))
                pn = {
                    "strike_hand": [f"hand_{hand}"],
                    "guard_hand": ["hand_right" if hand == "left"
                                   else "hand_left"],
                    "head": ["head"],
                    "feet": ["foot_left", "foot_right"],
                }
                from point5.contracts import POINT_INDEX
                mp = {k: [POINT_INDEX[n] for n in v]
                      for k, v in pn.items()}
                for t in range(min(active_steps, len(pos))):
                    ref_pos = ref_world[t]
                    if yaw_col is not None:
                        err = np.deg2rad(float(
                            (yaw_col[t] - ref_yaws[t] + 180.0)
                            % 360.0 - 180.0))
                        sig2 = self._SIG["torso_yaw"] ** 2
                        scores["torso_yaw"][t] = (
                            0.5 * np.exp(-(err * err) / sig2)
                            + 0.5 / (1.0 + (err * err)
                                     / (9.0 * sig2)))
                    for part, idxs in mp.items():
                        d2 = sum(
                            np.sum((pos[t][i] - ref_pos[i]) ** 2)
                            for i in idxs)
                        sig2 = self._SIG[part] ** 2
                        # Narrow precision peak plus a broad tail so
                        # far-off poses still get a recovery gradient.
                        scores[part][t] = (
                            0.5 * np.exp(-d2 / sig2)
                            + 0.5 / (1.0 + d2 / (9.0 * sig2)))

        win = np.arange(T) < active_steps
        # A single punch is a finite task, not a truncated continuing
        # balance episode: there is no value to bootstrap after impact.
        term = True
        loading = (np.arange(T) + 1) * _DT <= self._load_seconds
        channels: Dict[str, ChannelData] = {}
        for part in _PARTS:
            aw = np.where(loading, self._AW_LOAD[part],
                          self._AW_STRIKE[part])
            channels[f"r_{part}"] = ChannelData(
                reward=scores[part], is_terminated=term,
                actor_weight=(aw * win).astype(np.float32))
        fall_aw = np.where(loading, self._AW_LOAD["fall"],
                           self._AW_STRIKE["fall"])
        channels["r_fall"] = ChannelData(
            reward=fall_rew, is_terminated=term,
            actor_weight=(fall_aw * win).astype(np.float32))

        return [Trajectory(
            obs=obs_all[:T],
            actions=acts_all[:T],
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(episode, agent_id, T),
        )]

    # --- eval ----------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        """Impact-frame fist error + success rate, not balance metrics."""
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
                hit_idx = min(round(self._task_seconds / _DT) - 1,
                              ep.agent_frame_boundary.get(rid,
                                                          t_full) - 1)
                if hit_idx < 0:
                    continue
                cols = [extract_per_step_field(
                    ep.observer_outputs, obs_name,
                    f"p5_{3 * hand_idx + i}", t_full) for i in range(3)]
                if any(c is None for c in cols):
                    continue
                phi = extract_per_step_field(
                    ep.observer_outputs, self._BAL_OBS[rid],
                    "potential", t_full)
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
                "impact_hand_error": (round(mean_err, 3)
                                      if errors else None),
                "strike_attempts": attempts,
            },
        }


EXPERIMENT_CLASS = StandPunchMBS
