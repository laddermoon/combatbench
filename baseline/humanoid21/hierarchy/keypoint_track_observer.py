"""KeypointTrackObserver — two-phase keypoint-position strike reward.

Successor of StrikeTrackObserver for the fixed-66-dim keypoint
experiment (STRIKE_KEYPOINT_2PHASE_PLAN.md).  Instead of per-joint
angle MSE it scores **keypoint positions** against frozen per-phase
targets built from the resolved clip:

  phase 1 (ready)   — T1: clip frame-0 keypoints, ground-aligned and
      anchored to the robot's xy/yaw at the strike's first step.
      Success: all 15 keypoints within their tolerances
      (hands/feet 3 cm, rest 5 cm).  On that frame T2 is built.
  phase 2 (landing) — T2: clip landing-frame keypoints under the same
      anchor, rigidly translated so the power-foot keypoint equals the
      robot's actual power-foot position on the success frame.  The
      fixed target IS the rear-foot constraint — no separate channel.

Per-step output fields: ``reward`` (current phase −L_kp), ``phase``
(1/2, the phase the frame was scored under), ``p1_done``/``p2_done``
(1.0 on the achieving frame), ``kp_rms``, ``kp_max_err_m``,
``foot_disp_m`` (power foot vs its anchored target, diagnostic),
``hand_err_m`` (punching hand), ``in_strike``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from envs.framework.observer_plugin import BaseObserverPlugin
from mocap.humanoid21.tracking.ref_data import RefClip

from .kp_kin import (KP_IDX, KP_NAMES, N_KP, POWER_FOOT, TOL_M,
                     anchored, build_p2_target, ground_align,
                     kp_errors, kp_loss, kp_pass, live_keypoints,
                     ref_keypoints, yaw_of)
from .strike_track_observer import (
    SKILL_CLIPS_DIR, clip_land_frame, punch_hand_name, resolve_clip)


class KeypointTrackObserver(BaseObserverPlugin):
    """Two-phase keypoint-position tracking for one agent."""

    def __init__(self, agent_id: str = 'robot_a',
                 clips_dir: str = str(SKILL_CLIPS_DIR),
                 tol_ee_m: float = 0.03,
                 tol_mid_m: float = 0.05) -> None:
        self.agent_id = str(agent_id)
        self._clips_dir = clips_dir
        self._tol = np.array(
            [tol_ee_m if n.startswith(("hand", "foot")) else tol_mid_m
             for n in KP_NAMES], dtype=np.float64)
        self._clips: Dict[str, RefClip] = {}
        self._segments: List[Dict[str, Any]] = []
        self._t = -1
        self._output: Dict[str, float] = self._zero()

    # ------------------------------------------------------------------
    def _clip(self, clip_id: str) -> RefClip:
        if clip_id not in self._clips:
            self._clips[clip_id] = RefClip(clip_id, self._clips_dir)
        return self._clips[clip_id]

    @staticmethod
    def _zero() -> Dict[str, float]:
        return {'reward': 0.0, 'phase': 0.0,
                'p1_done': 0.0, 'p2_done': 0.0,
                'kp_rms': 0.0, 'kp_max_err_m': 0.0,
                'foot_disp_m': 0.0, 'hand_err_m': 0.0,
                'in_strike': 0.0}

    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx) -> None:
        self._t = -1
        self._segments = []
        self._output = self._zero()
        sched = (ctx.episode_options.get('move_schedule') or {}).get(
            self.agent_id) or []
        clip_ov = (ctx.episode_options.get('strike_clip') or {}).get(
            self.agent_id)
        t = 0
        for seg in sched:
            n = int(seg[0])
            row: Dict[str, Any] = dict(a=t, b=t + n, ref=None,
                                       phase=1, T1=None, T2=None)
            if len(seg) > 6 and int(round(seg[5])) == +1:
                side = int(round(seg[6]))
                cid = clip_ov or resolve_clip(
                    side, seg[7:10], self._clips_dir)
                ref = self._clip(cid)
                row['clip'] = cid
                row['ref'] = ref
                row['k_land'] = clip_land_frame(cid, self._clips_dir)
                # Frozen reference keypoints: frame 0 (ground-aligned)
                # and the landing frame, plus the frame-0 root anchor.
                row['pts0'] = ground_align(ref_keypoints(ref, 0))
                row['ptsL'] = ref_keypoints(ref, row['k_land'])
                row['ref_xy0'] = ref.root_pos[0][:2].copy()
                row['ref_yaw0'] = yaw_of(ref.root_quat[0])
                row['rear_idx'] = KP_IDX[POWER_FOOT[cid]]
                row['hand_idx'] = KP_IDX[punch_hand_name(ref)]
            self._segments.append(row)
            t += n

    # ------------------------------------------------------------------
    def on_post_action_step(self, ctx) -> None:
        self._t += 1
        self._output = self._compute(ctx)

    def get_output(self) -> Dict[str, float]:
        return self._output

    # ------------------------------------------------------------------
    def _compute(self, ctx) -> Dict[str, float]:
        seg = next((s for s in self._segments
                    if s['a'] <= self._t < s['b']), None)
        if seg is None or seg.get('ref') is None:
            return self._zero()
        out = self._zero()
        out['in_strike'] = 1.0

        core = ctx.accessor.get_core_state()[self.agent_id]
        live = live_keypoints(core)
        root_pos = np.asarray(core['root_pos'], dtype=np.float64)
        root_rot = np.asarray(core['root_rot'], dtype=np.float64)

        # Lazy anchor on the first step inside the segment (same
        # convention as StrikeTrackObserver): T1 is frozen here.
        if seg['T1'] is None:
            yaw = yaw_of(root_rot)
            seg['anchor'] = dict(xy=root_pos[:2].copy(), yaw=yaw)
            seg['T1'] = anchored(seg['pts0'], seg['ref_xy0'],
                                 seg['ref_yaw0'], root_pos[:2], yaw)

        if seg['phase'] == 1:
            errs = kp_errors(live, seg['T1'])
            out['phase'] = 1.0
            if np.all(errs <= self._tol):
                out['p1_done'] = 1.0
                seg['phase'] = 2
                # T2: landing pose rigidly translated so the power-foot
                # target sits exactly on the actual foot position.
                seg['T2'] = build_p2_target(
                    seg['ptsL'], seg['ref_xy0'], seg['ref_yaw0'],
                    seg['anchor']['xy'], seg['anchor']['yaw'],
                    seg['rear_idx'], live[seg['rear_idx']])
                seg['rear_anchor'] = live[seg['rear_idx']].copy()
        else:
            errs = kp_errors(live, seg['T2'])
            out['phase'] = 2.0
            out['foot_disp_m'] = float(
                np.linalg.norm(live[seg['rear_idx']]
                               - seg['rear_anchor']))
            if np.all(errs <= self._tol):
                out['p2_done'] = 1.0

        out['reward'] = -kp_loss(errs)
        out['kp_rms'] = float(np.sqrt(np.mean(errs * errs)))
        out['kp_max_err_m'] = float(errs.max())
        out['hand_err_m'] = float(errs[seg['hand_idx']])
        return out
