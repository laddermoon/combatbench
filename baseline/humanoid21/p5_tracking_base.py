"""Configurable P5-trajectory tracking experiment base (library, not an
experiment registry file).

Implements the full stand-punch pipeline — reachable sampling, clip
serialization, frame-anchored P5 scoring, strike eval — with the reward
formulation factored into class-attribute knobs so each ablation
variant is a thin ``exp_*.py`` config:

    _DIST_MODE    "linear" r=-d | "gauss" exp(-(d/s)^2) |
                  "tail"  peak+broad-tail | "axial" attack-axis split
    _ANCHOR_MODE  "time" hard lock | "keyframe" 5 sparse nodes |
                  "progress" self-paced reference index
    _AW           per-channel actor weight (uniform 1.0 in the base)
    _AW_PHASE     optional (load_aw, strike_aw) overrides per channel
    _SIG          kernel scale for gauss/tail
    _EXTRAS       extra channels: "torso_yaw", "geom"

Every variant keeps the same episode structure (13 x 50 ms actions,
0.65 s retimed clip, world-frame P5 observations) — only the reward
changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory)
from baseline.framework.rollout import extract_per_step_field
from point5.contracts import P5Clip, POINT_INDEX, Point5State
from point5.humanoid21.stand_punch import (
    StandPunchConfig, plan_stand_punch, stand_p5)

from baseline.humanoid21.hierarchy.p5_world import clip_frame_samples
from baseline.humanoid21.plugins.stand_punch_target import (
    SEGMENT_DURATIONS, timed_stand_clip)

from baseline.experiments_ppo.base import CombatExperimentPPOBase

_ROOT = Path(__file__).resolve().parents[2]
_DT = 0.05      # seconds per action step

# Per-POINT channels (one per P5 landmark) — role weights are applied
# at scoring time via _AW_ROLE since the striking hand varies.
POINT_CHANNELS = ("head", "hand_left", "hand_right",
                  "foot_left", "foot_right")

# Relative-geometry pairs for the optional "geom" channel.
_GEOM_PAIRS = (("head", "foot_left"), ("head", "foot_right"),
               ("hand_left", "head"), ("hand_right", "head"),
               ("foot_left", "foot_right"))

# Keyframe boundary times -> action steps that complete each node
# (stand .10 / coil .30 / land .50 / strike .65 -> steps 2,6,10,13).
_KEYFRAME_STEPS = tuple(
    int(round(sum(SEGMENT_DURATIONS[:i + 1]) / _DT)) - 1
    for i in range(len(SEGMENT_DURATIONS)))

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


class P5TrackingBase(CombatExperimentPPOBase):
    """Stand-punch P5 tracking with a configurable reward formulation."""

    name = "p5_tracking_base"   # subclasses MUST override
    obs_dim = 66
    agent_used = "both"
    actor_blueprint = "init_policy_mbs_66d.yaml"
    max_steps = 13

    _AGENT_IDS = ("robot_a", "robot_b")
    _P5_OBS = {"robot_a": "point5_a", "robot_b": "point5_b"}
    _BAL_OBS = {"robot_a": "standing_balance_a",
                "robot_b": "standing_balance_b"}
    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a"),
        ("robot_b", "foot_state_b", "standing_balance_b"),
    )

    # ---- ablation knobs ----------------------------------------------
    _DIST_MODE = "linear"        # linear | gauss | tail | axial
    _ANCHOR_MODE = "time"        # time | keyframe | progress
    _SIG = 0.05                  # gauss/tail kernel scale (m / rad)
    _SIG_YAW = 0.30              # torso-yaw kernel scale (rad)
    _AXIAL_LAT_W = 2.0           # lateral-vs-axial error weight
    _AW = 1.0                    # uniform per-channel actor weight
    _AW_STRIKE_HAND = None       # role override for the punch hand
    _AW_PHASE = None             # {ch: (load_aw, strike_aw)} overrides
    _EXTRAS: Tuple[str, ...] = ()  # "torso_yaw" | "geom"
    _FALL_AW = 2.0
    _FALL_PEN = 1.0
    _STAND_PLAN = StandPunchConfig()
    _best_strike_success = -1.0
    _best_strike_error = float("inf")

    # ---- channels ------------------------------------------------------

    @property
    def _channel_names(self) -> Tuple[str, ...]:
        names = [f"r_{p}" for p in POINT_CHANNELS]
        names += [f"r_{e}" for e in self._EXTRAS]
        names.append("r_fall")
        return tuple(names)

    _gae_lambda = 0.95

    def reward_channels(self):
        return tuple(
            RewardChannel(name=k, gamma=0.99,
                          gae_lambda=self._gae_lambda)
            for k in self._channel_names)

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
        st0 = Point5State(pos=stand_p5())
        for _ in range(max_tries):
            hand = "left" if rng.random() < 0.5 else "right"
            tgt = np.array([rng.uniform(*_TGT_X),
                            rng.uniform(*_TGT_Y),
                            rng.uniform(*_TGT_Z)])
            plan = plan_stand_punch(tgt, hand, initial=st0)
            if plan.reachable:
                return hand, tgt, plan.task.to_dict()
        tgt = np.array([1.05, 0.0, 1.10])
        return "left", tgt, plan_stand_punch(
            tgt, "left", initial=st0).task.to_dict()

    @property
    def _load_seconds(self) -> float:
        return float(sum(SEGMENT_DURATIONS[:-1]))

    @property
    def _task_seconds(self) -> float:
        return float(sum(SEGMENT_DURATIONS))

    # --- scoring --------------------------------------------------------

    def _dist_score(self, d2: float) -> float:
        """Per-point reward from squared distance."""
        if self._DIST_MODE == "linear":
            return -float(np.sqrt(d2))
        if self._DIST_MODE == "gauss":
            return float(np.exp(-d2 / (self._SIG ** 2)))
        # "tail": narrow precision peak plus a broad recovery tail.
        sig2 = self._SIG ** 2
        return float(0.5 * np.exp(-d2 / sig2)
                     + 0.5 / (1.0 + d2 / (9.0 * sig2)))

    def _point_scores(self, pos_t: np.ndarray, ref_t: np.ndarray,
                      attack_axis: np.ndarray) -> Dict[str, float]:
        """Per-channel scores for one (measured, reference) frame."""
        out: Dict[str, float] = {}
        for name in POINT_CHANNELS:
            i = POINT_INDEX[name]
            diff = pos_t[i] - ref_t[i]
            if self._DIST_MODE == "axial":
                ax = float(diff @ attack_axis)
                lat = float(np.linalg.norm(diff - ax * attack_axis))
                out[name] = -(abs(ax) + self._AXIAL_LAT_W * lat)
            else:
                out[name] = self._dist_score(float(diff @ diff))
        if "geom" in self._EXTRAS:
            g = 0.0
            for a, b in _GEOM_PAIRS:
                ia, ib = POINT_INDEX[a], POINT_INDEX[b]
                dm = float(np.linalg.norm(pos_t[ia] - pos_t[ib]))
                dr = float(np.linalg.norm(ref_t[ia] - ref_t[ib]))
                g += abs(dm - dr)
            out["geom"] = -g
        return out

    def _anchor_index(self, t: int, actual_pos: np.ndarray,
                      ref_world: np.ndarray, attack_axis: np.ndarray,
                      n_ref: int) -> int:
        """Reference sample index for action step t."""
        if self._ANCHOR_MODE == "progress":
            # Self-paced: the furthest ref sample whose planned axial
            # progress (moving foot + strike hand) is already covered
            # by the actual motion — never ahead of schedule t.
            def prog(p):
                return float(
                    (p[POINT_INDEX["foot_left"]]
                     + p[POINT_INDEX["foot_right"]]
                     + p[POINT_INDEX["hand_left"]]
                     + p[POINT_INDEX["hand_right"]]) @ attack_axis)
            p_act = prog(actual_pos)
            best = 0
            for j in range(min(t + 1, n_ref)):
                if prog(ref_world[j]) <= p_act + 0.02:
                    best = j
            return best
        return min(t, n_ref - 1)

    # --- trajectories ---------------------------------------------------

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

        phi_arr = extract_per_step_field(
            episode.observer_outputs, phi_key, "potential", T_full)
        phi_arr = (np.clip(np.asarray(phi_arr[:T], dtype=np.float32),
                           0.0, 1.0) if phi_arr is not None
                   else np.zeros(T, dtype=np.float32))
        fall_rew = (-self._FALL_PEN
                    * _fall_onset_mask(phi_arr)).astype(np.float32)

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

        st0 = Point5State(
            pos=np.asarray(start_xyz, dtype=np.float64).reshape(5, 3))
        clip = timed_stand_clip(P5Clip.from_dict(clip_d), st0.pos)
        ref_world, ref_yaws = clip_frame_samples(clip, f1, _DT)
        n_ref = len(ref_world)
        active_steps = min(T, n_ref)
        ax2 = np.asarray(clip.metadata.get("attack_axis", [1.0, 0.0]),
                         dtype=np.float64)
        attack_axis = np.array([ax2[0], ax2[1], 0.0])

        ch_names = list(POINT_CHANNELS) + list(self._EXTRAS)
        scores = {c: np.zeros(T, np.float32) for c in ch_names}
        for t in range(min(active_steps, len(pos))):
            j = self._anchor_index(t, pos[t], ref_world, attack_axis,
                                   n_ref)
            ps = self._point_scores(pos[t], ref_world[j], attack_axis)
            for c, v in ps.items():
                scores[c][t] = v
            if "torso_yaw" in self._EXTRAS and yaw_col is not None:
                err = np.deg2rad(float(
                    (yaw_col[t] - ref_yaws[j] + 180.0) % 360.0 - 180.0))
                sig2 = self._SIG_YAW ** 2
                scores["torso_yaw"][t] = (
                    0.5 * np.exp(-(err * err) / sig2)
                    + 0.5 / (1.0 + (err * err) / (9.0 * sig2)))

        # --- actor weights ----------------------------------------------
        if self._ANCHOR_MODE == "keyframe":
            win = np.zeros(T, dtype=bool)
            for k in _KEYFRAME_STEPS:
                if k < active_steps:
                    win[k] = True
        else:
            win = np.arange(T) < active_steps
        loading = (np.arange(T) + 1) * _DT <= self._load_seconds

        def _aw(ch: str) -> np.ndarray:
            if self._AW_PHASE and ch in self._AW_PHASE:
                lo, hi = self._AW_PHASE[ch]
                base = np.where(loading, lo, hi)
            else:
                base = np.full(T, self._AW, dtype=np.float32)
            if (self._AW_STRIKE_HAND is not None
                    and ch == f"hand_{hand}"):
                aw = self._AW_STRIKE_HAND
                base = (np.where(loading, aw[0], aw[1])
                        if isinstance(aw, (tuple, list))
                        else np.full(T, aw, dtype=np.float32))
            return (base * win).astype(np.float32)

        channels: Dict[str, ChannelData] = {}
        for c in ch_names:
            channels[f"r_{c}"] = ChannelData(
                reward=scores[c], is_terminated=True,
                actor_weight=_aw(c))
        channels["r_fall"] = ChannelData(
            reward=fall_rew, is_terminated=True,
            actor_weight=(np.full(T, self._FALL_AW, np.float32)
                          * win).astype(np.float32))

        return [Trajectory(
            obs=obs_all[:T],
            actions=acts_all[:T],
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(episode, agent_id, T),
        )]

    # --- eval -----------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        """Impact-frame fist error + success rate."""
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
