"""Hierarchy Step 4 — command-conditioned punches (66-dim, v2).

Warm-start from the Step-2/3 lineage.  Schedules (``move_schedule``) mix
STOP segments with strike segments written as full 10-field rows:

    [n_steps, 0, 0, 0, 0, +1, side, tx, ty, tz]

The target is pelvis-ego at segment onset.  Style follows commanded
distance (user spec): near (<0.52 m horizontal) → jab clip, far →
lunge clip; side picks the hand via mirrored variants.

  side=+1 L : near → hdm05_3146_88_2895_127    far → cmu_144_20_m_seg01
  side=-1 R : near → hdm05_3146_88_2895_127_m  far → cmu_144_20_seg01

Reward channels (5):
    r_potential = 0.01 × φ(t),                      aw = 1.0
    r_move/r_turn = seg intent matching             aw = 3·φ² (dead at L0)
    r_std_clip  = clip+pose+0.5·cad+h_term,        aw = 3·φ²·I(STOP non-strike)
    r_strike    = StrikeTrackObserver reward,       aw = 4·φ²·I(in_strike)

Curriculum — only intent-sampling changes, never reward formulas:
    L0  targets = exact natural landing point of the picked clip
    L1  target jitter ±0.05 m (staying within its near/far class)
    L2  target jitter ±0.15 m
Promotion: median strike kp_err_rel < 0.10 m and success ≥ 0.95,
2 consecutive evals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field

from baseline.humanoid21.hierarchy.command_simulator import (
    BODY_DIM, EFF_TYPE,
)
from baseline.humanoid21.hierarchy.strike_track_observer import (
    _CLIP_MAP, _clip_meta,
)

from .exp_hier_s1_standup_step_mbs import HierS1StandupStepMBS
from .exp_hier_s3_move_mbs import HierS3MoveMBS

_STRIKE_OBS = {"robot_a": "strike_track_a", "robot_b": "strike_track_b"}
_PAD_STEPS = 15            # post-clip recovery steps inside a strike seg


class HierS4PunchMBS(HierS3MoveMBS):
    """Step-4-A: strike units on a standing/stepping base."""

    name = "hier_s4_punch_mbs"

    _STRIKE_AW = 4.0

    # --- Curriculum: intent-sampling only ---
    # (s4-prefixed attrs: s3's on_eval also touches self._level/_LEVELS —
    # we bypass it, but keep names distinct to avoid silent coupling.)
    _s4_level: int = 0
    _s4_streak: int = 0
    _S4_LEVELS = (
        # tgt_jitter, stop_gap
        dict(jit=0.00, gap=(40, 70)),
        dict(jit=0.05, gap=(40, 70)),
        dict(jit=0.15, gap=(30, 60)),
    )

    # ------------------------------------------------------------------
    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s4_punch_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    def reward_channels(self) -> tuple:
        names = ("r_potential", "r_move", "r_turn", "r_std_clip",
                 "r_strike")
        return tuple(
            RewardChannel(name=n, gamma=self._gamma,
                          gae_lambda=self._gae_lambda)
            for n in names
        )

    # ------------------------------------------------------------------
    # Schedule sampling: STOP lead → alternating STOP / strike
    # ------------------------------------------------------------------

    def _sample_strike_row(self, rng: np.random.Generator) -> List[float]:
        side = int(rng.choice([+1, -1]))
        near = bool(rng.uniform() < 0.5)
        clip = _CLIP_MAP[(side, near)]
        land, n_ctrl = _clip_meta(clip)
        jit = self._S4_LEVELS[self._s4_level]["jit"]
        tgt = land.copy()
        if jit > 0:
            alt_land, _ = _clip_meta(_CLIP_MAP[(side, not near)])
            for _ in range(16):
                tgt = land + np.concatenate(
                    [rng.uniform(-jit, jit, 2), [0.0]])
                # jittered target must still resolve to this clip
                if np.linalg.norm(tgt - land) \
                        < 0.9 * np.linalg.norm(tgt - alt_land):
                    break
                tgt = land.copy()
        return [n_ctrl + _PAD_STEPS, 0, 0, 0, 0, +1, side,
                float(tgt[0]), float(tgt[1]), float(tgt[2])]

    def _sample_schedule(self, rng: np.random.Generator) -> List[list]:
        lv = self._S4_LEVELS[self._s4_level]
        segs: List[list] = []
        lead = int(rng.integers(*self._LEAD_STOP))
        segs.append([lead, 0.0, 0.0, 0.0])
        t = lead
        horizon = self.max_steps + 40
        while t < horizon:
            gap = int(rng.integers(lv["gap"][0], lv["gap"][1] + 1))
            segs.append([gap, 0.0, 0.0, 0.0])
            t += gap
            if t >= horizon:
                break
            row = self._sample_strike_row(rng)
            segs.append(row)
            t += int(row[0])
        return segs

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

        # --- segment analysis (movement channels unchanged) ------------
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

        r_move = np.zeros(T, dtype=np.float32)
        r_turn = np.zeros(T, dtype=np.float32)
        stop_mask = np.zeros(T, dtype=bool)
        strike_cmd = np.abs(obs_all[:T, BODY_DIM + EFF_TYPE]) > 0.5
        for sg in segs:
            r_move[sg["a"]:sg["b"]] = sg["s_move"]
            r_turn[sg["a"]:sg["b"]] = sg["s_turn"]
            if sg["is_stop"]:
                stop_mask[sg["a"]:sg["b"]] = True
        stop_mask &= ~strike_cmd          # strikes are not STOP segments

        # --- std clip: full reward only on non-strike STOP segments -----
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

        # --- strike tracking reward (online, per-step from observer) ----
        sk = _STRIKE_OBS[agent_id]
        s_rew = extract_per_step_field(
            episode.observer_outputs, sk, "reward", T_full)
        s_in = extract_per_step_field(
            episode.observer_outputs, sk, "in_strike", T_full)
        r_strike = (np.asarray(s_rew[:T], dtype=np.float32)
                    if s_rew is not None else np.zeros(T, np.float32))
        strike_mask = (np.asarray(s_in[:T], dtype=np.float32)
                       if s_in is not None else np.zeros(T, np.float32))

        channels = {
            "r_potential": ChannelData(
                reward=(0.01 * phi_arr).astype(np.float32),
                is_terminated=False,
                actor_weight=np.ones(T, dtype=np.float32)),
            "r_move": ChannelData(
                reward=r_move, is_terminated=False,
                actor_weight=(self._MOVE_AW * phi2).astype(np.float32)),
            "r_turn": ChannelData(
                reward=r_turn, is_terminated=False,
                actor_weight=(self._MOVE_AW * phi2).astype(np.float32)),
            "r_std_clip": ChannelData(
                reward=r_clip.astype(np.float32), is_terminated=False,
                actor_weight=(self._clip_aw_scale * phi2
                              * stop_mask.astype(np.float32)
                              ).astype(np.float32)),
            "r_strike": ChannelData(
                reward=r_strike, is_terminated=False,
                actor_weight=(self._STRIKE_AW * phi2
                              * strike_mask).astype(np.float32)),
        }
        return [Trajectory(
            obs=obs_all[:T],
            actions=acts_all[:T],
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            sampling_ctx=self.extract_sampling_ctx(episode, agent_id, T),
        )]

    # ------------------------------------------------------------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        # bypass s3's on_eval — its move-curriculum promotion mutates
        # self._level; s4 tracks its own level (_s4_level).
        result = HierS1StandupStepMBS.on_eval(self, episodes, update)
        kp_rel, q_rmse, srew, n_strike_frames = [], [], [], 0
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for agent_id in self._AGENT_IDS:
                sk = _STRIKE_OBS[agent_id]
                s_in = extract_per_step_field(
                    ep.observer_outputs, sk, "in_strike", T)
                if s_in is None:
                    continue
                s_in = np.asarray(s_in[:T]) > 0.5
                if not s_in.any():
                    continue
                n_strike_frames += int(s_in.sum())
                for name, acc in (("kp_err_rel_m", kp_rel),
                                  ("q_rmse_deg", q_rmse),
                                  ("reward", srew)):
                    a = extract_per_step_field(
                        ep.observer_outputs, sk, name, T)
                    if a is not None:
                        acc.append(float(np.asarray(a[:T])[s_in].mean()))

        kp_med = float(np.median(kp_rel)) if kp_rel else float("nan")
        success = result["info"].get("success", 0.0)
        ok = (np.isfinite(kp_med) and kp_med < 0.10
              and success >= 0.95)
        self._s4_streak = self._s4_streak + 1 if ok else 0
        promoted = False
        if self._s4_streak >= 2 \
                and self._s4_level < len(self._S4_LEVELS) - 1:
            self._s4_level += 1
            self._s4_streak = 0
            promoted = True

        result["info"]["level"] = float(self._s4_level)
        result["info"]["strike_kp"] = round(kp_med, 3)
        result["info"]["strike_q"] = round(
            float(np.median(q_rmse)), 1) if q_rmse else float("nan")
        result["info"]["strike_r"] = round(
            float(np.mean(srew)), 3) if srew else float("nan")
        result["info"]["strike_frames"] = float(n_strike_frames)
        if promoted:
            result["info"]["promoted"] = True
        return result

    def state(self) -> dict:
        return {**super().state(), "s4_level": self._s4_level,
                "s4_streak": self._s4_streak}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._s4_level = int(state.get("s4_level", 0))
        self._s4_streak = int(state.get("s4_streak", 0))


EXPERIMENT_CLASS = HierS4PunchMBS
