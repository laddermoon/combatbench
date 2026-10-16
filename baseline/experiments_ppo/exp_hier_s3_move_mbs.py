"""Hierarchy Step 3 — command-conditioned movement (66-dim, v2).

Warm-start from the Step-2 final checkpoint.  Each episode carries a
per-robot ``move_schedule`` in episode_options — a list of
``[n_steps, vx, vy, w]`` segments (m/s, rad/s) compiled by
CommandSimulator into per-step 9-dim commands.  The first segment is
always STOP so the robot can stand up before motion commands start.

Offline segment-level rewards (per user spec — observers only record
raw kinematics, nothing is scored per-step online):

  For each constant-command segment [a, b), scored on [a+SETTLE, b):
    v̂ = mean ego-frame velocity  (per-step Δp rotated by -yaw(t))
    ŵ = mean yaw rate
    s_move = exp(-||v̂ - v*||² / σ_v²)
    s_turn = exp(-(ŵ - w*)² / σ_w²)
  The segment score is broadcast to ALL steps of the segment.

Channels (4):
    r_potential = 0.01 × φ(t),                       aw = 1.0
    r_move      = s_move broadcast,                aw = 3·φ²
    r_turn      = s_turn broadcast,                aw = 3·φ²
    r_std_clip  = clip + pose + 0.5·cad + h_term,  aw = 3·φ²·I(STOP seg)

Curriculum — only the command-sampling space changes, never the reward:
    L0  v∈[0.2,0.5]  all-direction, w=0,          seg 60-80 st
    L1  v∈[0.2,0.9]
    L2  v∈[0.2,1.5]  full speed range
    L3  + turn-only segs w∈±[0.5,1.5], + arcs (v,w both nonzero)
    L4  seg 30-60 st, w≤2π, denser stop/reverse
Promotion: median v_err < 0.12 m/s (and w_err < 0.3 rad/s once turns
exist), fall rate < 5%, for 2 consecutive evals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field

from baseline.humanoid21.hierarchy.command_simulator import (
    BODY_DIM, V_MAX, VX, VY, WZ, W_MAX,
)
from .exp_hier_s1_standup_step_mbs import HierS1StandupStepMBS

_DT = 0.05          # s per action step (20 Hz)
_SETTLE = 10        # steps exempt from scoring after a command switch


class HierS3MoveMBS(HierS1StandupStepMBS):
    """Step-3: (vx, vy, w) command following, segment-level rewards."""

    name = "hier_s3_move_mbs"
    obs_dim: int = 66
    max_steps: int = 400

    # --- Reward ---
    _SIGMA_V = 0.25      # m/s
    _SIGMA_W = 0.4       # rad/s
    _MOVE_AW = 3.0
    _EMA_SPAN = 16       # steps; ~1.5 gait cycles — smooths stepping
                       # oscillation while staying responsive (~8-step lag)

    _MOVE_OBS = {"robot_a": "move_track_a", "robot_b": "move_track_b"}

    # --- Curriculum: command-sampling space only ---
    _level: int = 0
    _promote_streak: int = 0
    _LEVELS = (
        # v_hi, seg_len, p_stop, p_turn, p_arc, w_hi, dir (heading cone)
        dict(v_hi=0.5, seg=(60, 80),  p_stop=0.35, p_turn=0.0,  p_arc=0.0,  w_hi=0.0,  dir=np.pi / 4),
        dict(v_hi=0.9, seg=(60, 80),  p_stop=0.35, p_turn=0.0,  p_arc=0.0,  w_hi=0.0,  dir=np.pi / 2),
        dict(v_hi=1.5, seg=(60, 80),  p_stop=0.35, p_turn=0.0,  p_arc=0.0,  w_hi=0.0,  dir=np.pi),
        dict(v_hi=1.5, seg=(60, 80),  p_stop=0.30, p_turn=0.25, p_arc=0.20, w_hi=1.5,  dir=np.pi),
        dict(v_hi=1.5, seg=(30, 60),  p_stop=0.30, p_turn=0.25, p_arc=0.20, w_hi=6.28, dir=np.pi),
    )
    _V_LO = 0.2
    _LEAD_STOP = (60, 100)   # first segment: STOP so the robot can stand

    # ------------------------------------------------------------------
    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s3_move_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    def reward_channels(self) -> tuple:
        names = ("r_potential", "r_move", "r_turn", "r_std_clip")
        return tuple(
            RewardChannel(name=n, gamma=self._gamma,
                          gae_lambda=self._gae_lambda)
            for n in names
        )

    # ------------------------------------------------------------------
    # Command schedule sampling (curriculum = intent space only)
    # ------------------------------------------------------------------

    def _sample_schedule(self, rng: np.random.Generator) -> List[list]:
        lv = self._LEVELS[self._level]
        segs: List[list] = []
        lead = int(rng.integers(*self._LEAD_STOP))
        segs.append([lead, 0.0, 0.0, 0.0])
        t = lead
        horizon = self.max_steps + 40
        while t < horizon:
            n = int(rng.integers(lv["seg"][0], lv["seg"][1] + 1))
            u = rng.uniform()
            if u < lv["p_stop"]:
                segs.append([n, 0.0, 0.0, 0.0])
            elif u < lv["p_stop"] + lv["p_turn"]:
                w = float(rng.uniform(0.5, lv["w_hi"]) * rng.choice([-1, 1]))
                segs.append([n, 0.0, 0.0, w])
            else:
                # Heading cone narrows with level: the s42i stepping gait
                # is fore-aft, so early levels train forward-cone walking
                # before opening up to lateral/backward commands.
                th = float(rng.uniform(-lv["dir"], lv["dir"]))
                v = float(rng.uniform(self._V_LO, lv["v_hi"]))
                w = 0.0
                if u > 1.0 - lv["p_arc"]:
                    w = float(rng.uniform(0.3, min(lv["w_hi"], 1.0))
                              * rng.choice([-1, 1]))
                segs.append([n, v * np.cos(th), v * np.sin(th), w])
            t += n
        return segs

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        from baseline.framework.rollout import Job, SamplingSpec
        env_bp = self._env_pb().materialize(max_steps=self.max_steps)
        rng = np.random.default_rng(base_seed)
        sampling = SamplingSpec(explore_factor=self.explore_factor)
        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max))
            move_schedule = {
                rid: self._sample_schedule(rng)
                for rid in self._AGENT_IDS
            }
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=policy_bp,
                env_bp=env_bp,
                seed=seed,
                episode_options={
                    "initial_distance": initial_distance,
                    "move_schedule": move_schedule,
                },
                sampling_a=sampling,
                sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # Offline segment analysis
    # ------------------------------------------------------------------

    def _segment_stats(self, obs_all, root_xy, yaw):
        """Per-segment (a, b, v*, w*, is_stop, s_move, s_turn, v_err, w_err)."""
        T = obs_all.shape[0]
        cmd = obs_all[:, BODY_DIM:]
        vx_s = cmd[:, VX] * V_MAX
        vy_s = cmd[:, VY] * V_MAX
        w_s = cmd[:, WZ] * W_MAX

        # constant-cmd segment boundaries
        key = np.round(cmd[:, VX:WZ + 1], decimals=3)
        change = np.any(np.diff(key, axis=0) != 0.0, axis=1)
        bounds = [0] + list(np.nonzero(change)[0] + 1) + [T]

        # per-step ego-frame velocity: dp[t] rotated by -yaw[t]
        dp = np.diff(root_xy, axis=0) / _DT                     # (T-1, 2)
        yw = yaw[1:]
        c, s = np.cos(-yw), np.sin(-yw)
        vx_ego = c * dp[:, 0] - s * dp[:, 1]
        vy_ego = s * dp[:, 0] + c * dp[:, 1]
        dyaw = np.diff(np.unwrap(yaw)) / _DT                    # (T-1,)

        segs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            lo = min(a + _SETTLE, max(a, b - 5))   # keep ≥5 scored steps
            v_err = float(np.linalg.norm(
                [vx_ego[lo:b].mean() - vx_s[a],
                 vy_ego[lo:b].mean() - vy_s[a]])) if b - lo >= 2 else np.nan
            w_err = float(abs(dyaw[lo:b].mean() - w_s[a])) \
                if b - lo >= 2 else np.nan
            s_move = float(np.exp(-v_err ** 2 / self._SIGMA_V ** 2)) \
                if np.isfinite(v_err) else 0.0
            s_turn = float(np.exp(-w_err ** 2 / self._SIGMA_W ** 2)) \
                if np.isfinite(w_err) else 0.0
            is_stop = abs(vx_s[a]) < 1e-3 and abs(vy_s[a]) < 1e-3 \
                and abs(w_s[a]) < 1e-3
            segs.append(dict(a=a, b=b, is_stop=is_stop, s_move=s_move,
                             s_turn=s_turn, v_err=v_err, w_err=w_err,
                             w_cmd=float(w_s[a])))
        return segs

    def _dense_move_rewards(self, obs_all, root_xy, yaw_arr, T):
        """Per-step (r_move, r_turn) from EMA-smoothed ego velocity.

        Broadcasting one segment score over a 60-80-step window gives no
        credit for *when* the robot moves — the only gradient is "do
        better somewhere in this window".  Here every step is scored on
        how well the EMA of realized ego velocity / yaw-rate matches the
        command issued at that step, so early acceleration is rewarded
        immediately.  The EMA (~1.5 gait cycles) smooths stepping
        oscillation, preserving the segment-mean semantics: sustained
        mean velocity is what counts.
        """
        cmd_t = obs_all[:, BODY_DIM:]
        vx_c = cmd_t[:, VX].astype(np.float64) * V_MAX
        vy_c = cmd_t[:, VY].astype(np.float64) * V_MAX
        w_c = cmd_t[:, WZ].astype(np.float64) * W_MAX
        dp = np.diff(root_xy, axis=0) / _DT
        yw = yaw_arr[1:]
        c, s = np.cos(-yw), np.sin(-yw)
        vel = np.zeros((T, 3))
        vel[1:, 0] = c * dp[:, 0] - s * dp[:, 1]
        vel[1:, 1] = s * dp[:, 0] + c * dp[:, 1]
        vel[1:, 2] = np.diff(np.unwrap(yaw_arr)) / _DT
        alpha = 2.0 / (self._EMA_SPAN + 1)
        ema = np.zeros(3)
        vel_s = np.empty_like(vel)
        for t in range(T):
            ema += alpha * (vel[t] - ema)
            vel_s[t] = ema
        v_err_t = np.hypot(vel_s[:, 0] - vx_c, vel_s[:, 1] - vy_c)
        r_move = np.exp(-v_err_t ** 2 / self._SIGMA_V ** 2) \
            .astype(np.float32)
        w_err_t = np.abs(vel_s[:, 2] - w_c)
        r_turn = np.exp(-w_err_t ** 2 / self._SIGMA_W ** 2) \
            .astype(np.float32)
        return r_move, r_turn

    # ------------------------------------------------------------------
    def _build_agent_trajectory(self, episode, agent_id: str, obs_key: str):
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
            episode.observer_outputs, obs_key, "potential", T_full)
        if phi_arr is not None:
            phi_arr = np.clip(
                np.asarray(phi_arr[:T], dtype=np.float32), 0.0, 1.0)
            if phi_arr.size and self._ep_final_pots is not None:
                self._ep_final_pots.append(float(phi_arr[-1]))
        else:
            phi_arr = np.zeros(T, dtype=np.float32)
        phi2 = phi_arr * phi_arr

        # --- raw kinematics for segment analysis ---
        mk = self._MOVE_OBS[agent_id]
        rx = extract_per_step_field(
            episode.observer_outputs, mk, "root_x", T_full)
        ry = extract_per_step_field(
            episode.observer_outputs, mk, "root_y", T_full)
        yaw = extract_per_step_field(
            episode.observer_outputs, mk, "yaw", T_full)
        kin_ok = rx is not None and ry is not None and yaw is not None
        if kin_ok:
            root_xy = np.stack([rx[:T], ry[:T]], axis=1).astype(np.float64)
            yaw_arr = np.asarray(yaw[:T], dtype=np.float64)
            segs = self._segment_stats(obs_all[:T], root_xy, yaw_arr)
        else:
            segs = []

        # Dense per-step tracking reward (see _dense_move_rewards).  The
        # segment-mean score is kept for eval metrics only.
        r_move, r_turn = self._dense_move_rewards(obs_all[:T], root_xy,
                                                  yaw_arr, T) \
            if kin_ok else (np.zeros(T, np.float32),
                            np.zeros(T, np.float32))
        stop_mask = np.zeros(T, dtype=bool)
        for sg in segs:
            if sg["is_stop"]:
                stop_mask[sg["a"]:sg["b"]] = True

        # --- std clip: full reward only on STOP segments ---
        ck = self._CLIP_OBS[agent_id]
        def _f(name, lo, hi):
            a = extract_per_step_field(
                episode.observer_outputs, ck, name, T_full)
            return (np.clip(np.asarray(a[:T], dtype=np.float32), lo, hi)
                    if a is not None else np.zeros(T, dtype=np.float32))
        clip_arr = _f("std_clip", 0.0, 1.0)
        pose_arr = _f("std_pose", 0.0, 1.0)
        cad_arr = _f("std_cad", -1.0, 1.0)
        h_arr = extract_per_step_field(
            episode.observer_outputs, obs_key, "h_torso", T_full)
        h_arr = (np.asarray(h_arr[:T], dtype=np.float32)
                 if h_arr is not None else np.zeros(T, dtype=np.float32))
        h_term = np.exp(-((h_arr - self._H_TARGET) / self._H_SIGMA) ** 2)
        r_clip = clip_arr + pose_arr + 0.5 * cad_arr + 0.5 * h_term

        # --- one trajectory per constant-command segment ---------------
        # Aligns GAE boundaries with task boundaries (same machinery as
        # s4's strike spans): each critic gets a per-segment baseline, so
        # e.g. the "free" move reward on STOP segments is absorbed by
        # V(s) instead of inflating advantages elsewhere.  All channels
        # ride every segment — STOP segments still need the move/turn
        # penalty against drift.  Mid-episode spans bootstrap from the
        # next segment's first obs (which already carries its command).
        spans = [(sg["a"], sg["b"]) for sg in segs] if segs else [(0, T)]
        sctx = self.extract_sampling_ctx(episode, agent_id, T)
        fin = np.asarray(fin_obs, dtype=np.float32)
        trajs = []
        for a, b in spans:
            if b - a < 4:
                continue
            last = fin if b >= T else obs_all[b]
            ctx_slice = ({k: v[a:b] for k, v in sctx.items()}
                         if sctx is not None else None)
            trajs.append(Trajectory(
                obs=obs_all[a:b],
                actions=acts_all[a:b],
                last_obs=last,
                channels={
                    "r_potential": ChannelData(
                        reward=(0.01 * phi_arr[a:b]).astype(np.float32),
                        is_terminated=False,
                        actor_weight=np.ones(b - a, dtype=np.float32)),
                    "r_move": ChannelData(
                        reward=r_move[a:b], is_terminated=False,
                        actor_weight=(self._MOVE_AW * phi2[a:b])
                            .astype(np.float32)),
                    "r_turn": ChannelData(
                        reward=r_turn[a:b], is_terminated=False,
                        actor_weight=(self._MOVE_AW * phi2[a:b])
                            .astype(np.float32)),
                    "r_std_clip": ChannelData(
                        reward=r_clip[a:b].astype(np.float32),
                        is_terminated=False,
                        actor_weight=(self._clip_aw_scale * phi2[a:b]
                                      * stop_mask[a:b].astype(np.float32)
                                      ).astype(np.float32)),
                },
                importance=1.0,
                sampling_ctx=ctx_slice,
            ))
        return trajs

    # ------------------------------------------------------------------
    # Eval & curriculum promotion
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        v_errs, w_errs, stops, clips = [], [], [], []
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for agent_id in self._AGENT_IDS:
                obs_all = ep.observations.get(agent_id)
                if obs_all is None:
                    continue
                mk = self._MOVE_OBS[agent_id]
                rx = extract_per_step_field(
                    ep.observer_outputs, mk, "root_x", T)
                ry = extract_per_step_field(
                    ep.observer_outputs, mk, "root_y", T)
                yaw = extract_per_step_field(
                    ep.observer_outputs, mk, "yaw", T)
                if rx is None or ry is None or yaw is None:
                    continue
                obs_arr = np.asarray(obs_all[:T], dtype=np.float32)
                root_xy = np.stack([rx[:T], ry[:T]], 1).astype(np.float64)
                segs = self._segment_stats(
                    obs_arr, root_xy, np.asarray(yaw[:T], np.float64))
                for sg in segs:
                    if sg["is_stop"]:
                        stops.append(sg["s_move"])
                    else:
                        if np.isfinite(sg["v_err"]):
                            v_errs.append(sg["v_err"])
                        if np.isfinite(sg["w_err"]) \
                                and abs(sg["w_cmd"]) > 0.1:
                            w_errs.append(sg["w_err"])
                ck = self._CLIP_OBS[agent_id]
                c = extract_per_step_field(
                    ep.observer_outputs, ck, "std_clip", T)
                obs_key = dict(self._AGENT_OBS)[agent_id]
                phi = extract_per_step_field(
                    ep.observer_outputs, obs_key, "potential", T)
                if c is not None and phi is not None:
                    stand = np.asarray(phi[:T]) >= 0.9
                    # clip only on STOP segment steps
                    cmd = np.asarray(obs_all[:T], np.float32)[:, BODY_DIM:]
                    is_stop = (np.abs(cmd[:, VX]) < 1e-3) & \
                              (np.abs(cmd[:, VY]) < 1e-3) & \
                              (np.abs(cmd[:, WZ]) < 1e-3)
                    m = stand & is_stop
                    if m.any():
                        clips.append(float(np.asarray(c[:T])[m].mean()))

        v_med = float(np.median(v_errs)) if v_errs else float("nan")
        w_med = float(np.median(w_errs)) if w_errs else float("nan")
        stop_resid = 1.0 - float(np.mean(stops)) if stops else float("nan")
        success = result["info"].get("success", 0.0)

        # promotion: 2 consecutive compliant evals
        ok = (np.isfinite(v_med) and v_med < 0.12
              and success >= 0.95
              and (self._level < 3 or
                   (np.isfinite(w_med) and w_med < 0.3)))
        self._promote_streak = self._promote_streak + 1 if ok else 0
        promoted = False
        if self._promote_streak >= 2 and self._level < len(self._LEVELS) - 1:
            self._level += 1
            self._promote_streak = 0
            promoted = True

        result["info"]["level"] = float(self._level)
        result["info"]["v_err"] = round(v_med, 3)
        result["info"]["w_err"] = round(w_med, 3)
        result["info"]["stop_resid"] = round(stop_resid, 3)
        result["info"]["clip_stop"] = round(
            sum(clips) / max(len(clips), 1), 3)
        if promoted:
            result["info"]["promoted"] = True
        return result

    def state(self) -> dict:
        return {**super().state(), "s3_level": self._level,
                "s3_streak": self._promote_streak}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._level = int(state.get("s3_level", 0))
        self._promote_streak = int(state.get("s3_streak", 0))


EXPERIMENT_CLASS = HierS3MoveMBS
