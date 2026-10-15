"""StandardClipObserver — cyclic-stepping reward from the Route-B
``policy_capture`` standard clip library.

Per action step the observer builds the same 53-dim feature vector that
``extract_standard_clips.py`` produced offline::

    f_t = [ joint_pos_norm (21)      # core_state, already (q-MID)/HALF
          , joint_vel_norm (21)      # core_state, qv/HALF
          , projected_gravity (3)    # root frame
          , foot_L rel pelvis (3)    # torso frame
          , foot_R rel pelvis (3)    # torso frame
          , contact_R (1)            # feet_forces[0] > thresh  ([R,L] order)
          , contact_L (1) ]

Three complementary scores are emitted (each roughly in [0,1]):

``std_clip`` — strict transition match: whitened z_t=[f_{t-4}, f_t] on
the *dynamic subspace* (leg joints q/qv + gravity + feet + contacts,
35 dims -> 70-dim pairs) vs the library's transition pairs.
~0.05 for a frozen pose, ~0.5 for real stepping.

``std_pose`` — far-field lure: single-frame pose dims (q+grav+feet_rel,
30 dims) in the raw hand-weighted space vs all clip frames.  ~0.2 for
near-standing poses -> nonzero gradient pulling toward the manifold.

``std_cad`` — gait cadence: -s_t * s_{t-8} where s_t is the sign of the
left-minus-right foot load (dead-zone 0.15 body-weight).  The lag is the
half-period measured off the clips (~8 action steps).  Cyclic stepping
earns ~+0.6; a two-foot planted stance earns 0; parking on one leg is
actively penalised (-1).  This is the discriminant that keeps the
robot *moving* once it stands near the manifold.

Config:
    agent_id      robot to observe (default 'robot_a')
    clip_path     npz produced by extract_standard_clips.py
    sigma         transition bandwidth, whitened space (default 4.0)
    sigma_pose    pose bandwidth, raw weighted space (default 1.5)
    cadence_lag   half-cycle lag in action steps (default 8)
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np

from envs.framework.context import ReadOnlySimContext
from envs.framework.observer_plugin import BaseObserverPlugin
from envs.humanoid21.batch_binding import quat_to_rot_np

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CLIPS = (_REPO / 'baseline/humanoid21/hierarchy/assets'
                  / 'standard_stepping_clips.npz')
_CONTACT_THRESH = 0.15
_POSE_IDX = np.array(list(range(0, 21)) + list(range(42, 51)))  # q+grav+feet
# dynamic subspace: leg q (idx 3..14), leg qv (21+3..21+14), grav, feet, contact
_DYN_IDX = np.array(
    list(range(3, 15)) + list(range(24, 36)) + list(range(42, 53)))


class StandardClipObserver(BaseObserverPlugin):
    name = 'standard_clip'

    def __init__(self, agent_id: str = 'robot_a',
                 clip_path: str = str(_DEFAULT_CLIPS),
                 sigma: float = 4.0, sigma_pose: float = 1.5,
                 cadence_lag: int = 8) -> None:
        self.agent_id = agent_id
        self._sfx = '_a' if agent_id.endswith('_a') else '_b'
        self._sigma2 = 2.0 * sigma * sigma
        self._sigma_pose2 = 2.0 * sigma_pose * sigma_pose
        self._cad_lag = int(cadence_lag)

        d = np.load(clip_path, allow_pickle=False)
        feats = np.asarray(d['clip_features'], dtype=np.float64)  # (M,L,53)
        self._delta = int(d['delta']) if 'delta' in d.files else 4
        if 'feat_mu' in d.files:
            self._mu = np.asarray(d['feat_mu'], dtype=np.float64)
            self._sd = np.asarray(d['feat_sd_eff'], dtype=np.float64)
        else:
            flat = feats.reshape(-1, feats.shape[-1])
            self._mu = flat.mean(0)
            self._sd = flat.std(0) + 1e-3
        w = (np.asarray(d['feature_weights'], dtype=np.float64)
             if 'feature_weights' in d.files else np.ones(53))
        self._w_pose = w[_POSE_IDX]

        fw = (feats - self._mu) / self._sd
        fd = fw[..., _DYN_IDX]
        self._z_ref = np.concatenate(
            [fd[:, :-self._delta], fd[:, self._delta:]], axis=-1
        ).reshape(-1, 2 * fd.shape[-1])                          # (M*(L-d), 70)
        self._p_ref = feats.reshape(-1, feats.shape[-1])[:, _POSE_IDX]

        self._hist: deque[np.ndarray] = deque(maxlen=self._delta + 1)
        self._s_hist: deque[float] = deque(maxlen=self._cad_lag + 1)
        self._score = 0.0
        self._pose_score = 0.0
        self._cad = 0.0

    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._hist.clear()
        self._s_hist.clear()
        self._score = 0.0
        self._pose_score = 0.0
        self._cad = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        f, s = self._frame_features(ctx)

        p = f[_POSE_IDX]
        dp2 = (((self._p_ref - p) ** 2) * self._w_pose).sum(axis=1)
        self._pose_score = float(np.exp(-dp2.min() / self._sigma_pose2))

        self._s_hist.append(s)
        if len(self._s_hist) > self._cad_lag:
            self._cad = -self._s_hist[0] * self._s_hist[-1]
        else:
            self._cad = 0.0

        self._hist.append((f - self._mu) / self._sd)
        if len(self._hist) <= self._delta:
            self._score = 0.0
            return
        z = np.concatenate([self._hist[0][_DYN_IDX], self._hist[-1][_DYN_IDX]])
        d2 = ((self._z_ref - z) ** 2).sum(axis=1)
        self._score = float(np.exp(-d2.min() / self._sigma2))

    def _frame_features(self, ctx: ReadOnlySimContext) -> np.ndarray:
        core = ctx.accessor.get_core_state()[self.agent_id]
        ds = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        sfx = self._sfx

        q_norm = np.asarray(core['joint_pos_norm'], dtype=np.float64)
        qv_norm = np.asarray(core['joint_vel_norm'], dtype=np.float64)
        grav = np.asarray(
            ds['root_state']['projected_gravity'], dtype=np.float64)

        R = quat_to_rot_np(np.asarray(core['root_rot'], dtype=np.float64))
        RT = R.T
        bx = ds['body_xpos']
        pelvis = np.asarray(bx[f'pelvis{sfx}'], dtype=np.float64)
        fl = RT @ (np.asarray(bx[f'foot_left{sfx}']) - pelvis)
        fr = RT @ (np.asarray(bx[f'foot_right{sfx}']) - pelvis)

        ff = np.asarray(ds['feet_forces'], dtype=np.float64)      # [R, L]
        contact = (ff > _CONTACT_THRESH).astype(np.float64)
        # cadence signal: signed load difference, dead-zone 0.15 body-weight
        ld = ff[1] - ff[0]                                        # L - R
        s = float(np.sign(ld)) if abs(ld) > _CONTACT_THRESH else 0.0

        f = np.concatenate(
            [q_norm, qv_norm, grav, fl, fr, contact])             # 53
        return f, s

    # ------------------------------------------------------------------
    def get_output(self) -> Dict[str, Any]:
        return {
            'std_clip': self._score,
            'std_pose': self._pose_score,
            'std_cad': self._cad,
        }
