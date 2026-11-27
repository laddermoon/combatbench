"""PHC bundle (mocap/humanoid28/gmr pipeline, DATASPEC v2) -> CoreClip28.

Convention notes vs the humanoid21 converter:

- joint_pos / joint_vel are RAW RADIANS / rad/s — the h28 env has no
  normalization layer (MimicKit semantics), so dof passes through
  unchanged and ref_action = joint_pos (pd_target_rad).
- keypoint_pos = env KEYPOINT_BODIES world positions FK'd from the
  resampled states on the mocap model (same kinematic tree as the
  battle model's `_a` subtree).
- Velocities are re-derived on the 20 Hz grid by centered finite
  differences so every stored frame is a self-consistent CoreState
  usable directly as an RSI seed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .core_clip import (CONTROL_DT, N_DOF, CoreClip, SPEC_VERSION,
                        record_lineage)

_MOCAP_XML = (Path(__file__).resolve().parents[2] / 'gmr'
              / 'assets' / 'humanoid28' / 'humanoid28_mocap.xml')
_KP_CACHE: Dict[str, Any] = {}


def _kp_model():
    """Mocap-XML model + body ids for the env's semantic keypoint set
    (KEYPOINT_BODIES, unsuffixed names — same tree as battle `_a`)."""
    if 'm' not in _KP_CACHE:
        import mujoco
        from envs.humanoid28.meta import Humanoid28Meta
        mm = mujoco.MjModel.from_xml_path(str(_MOCAP_XML))
        names = list(Humanoid28Meta.KEYPOINT_BODIES)
        bids = np.array([mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in names])
        _KP_CACHE.update(m=mm, d=mujoco.MjData(mm), names=names, bids=bids)
    return _KP_CACHE


def fk_keypoints(states: Dict[str, np.ndarray], dof_names
                 ) -> "tuple[list, np.ndarray]":
    """World-frame keypoint positions FK'd from CoreClip states.

    qpos = [root_pos, root_rot(wxyz), joint_pos] written verbatim —
    joint_pos is already radians so no denormalization. Returns
    (keypoint_names, pos (T,K,3) float32)."""
    import mujoco
    c = _kp_model()
    T = len(states['root_pos'])
    out = np.empty((T, len(c['bids']), 3), np.float32)
    qpos = np.zeros(c['m'].nq)
    for t in range(T):
        qpos[0:3] = states['root_pos'][t]
        qpos[3:7] = states['root_rot'][t]
        qpos[7:] = states['joint_pos'][t]
        c['d'].qpos[:] = qpos
        mujoco.mj_forward(c['m'], c['d'])
        out[t] = c['d'].xpos[c['bids']]
    return c['names'], out


def _slerp_wxyz(q0, q1, a):
    q0, q1 = np.asarray(q0, np.float64), np.asarray(q1, np.float64)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        out = q0 + a * (q1 - q0)
    else:
        th = np.arccos(np.clip(d, -1.0, 1.0))
        out = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    return out / np.linalg.norm(out)


def phc_to_coreclip(clip: Dict[str, Any], clip_id: Optional[str] = None,
                    bundle: Optional[str] = None,
                    with_ref_action: bool = True,
                    meta_extra: Optional[Dict[str, Any]] = None,
                    edge_skip: int = 2, jump_bad_rad: float = 0.6
                    ) -> CoreClip:
    """PHC clip_dict (30 fps positions) -> CoreClip28 (20 Hz CoreStates).

    Positions/dof are resampled onto the 20 Hz grid (lerp pos/dof, slerp
    root_rot); joint_pos stays in RAW RADIANS (h28 has no norm layer).
    Velocities are then derived on that grid by centered finite
    differences (rederive_velocities) so each stored frame is a
    self-consistent CoreState usable directly as an RSI seed.
    """
    from .rederive import rederive_velocities

    rp = np.asarray(clip["root_trans_offset"], np.float64)
    rq_xyzw = np.asarray(clip["root_rot"], np.float64)
    dof = np.asarray(clip["dof"], np.float64)
    dof_names = [str(x) for x in clip["dof_names"]]
    T30 = len(rp)
    fps = int(clip["fps"])
    if fps != 30:
        raise ValueError(f"phc_to_coreclip expects 30fps, got {fps}")
    if dof.shape[1] != N_DOF:
        raise ValueError(f"expected {N_DOF} dof, got {dof.shape[1]}")

    # resample onto the 20Hz grid: 20Hz time k*0.05 = 1.5k in 30Hz frames
    n20 = int(np.floor((T30 - 1) / (fps * CONTROL_DT))) + 1
    rpos, rquat, jp = (np.empty((n20, 3)), np.empty((n20, 4)),
                     np.empty((n20, N_DOF)))
    for k in range(n20):
        f = k * fps * CONTROL_DT          # 1.5*k
        i0 = min(int(f), T30 - 1)
        i1 = min(i0 + 1, T30 - 1)
        a = f - i0
        rpos[k] = (1 - a) * rp[i0] + a * rp[i1]
        rquat[k] = _slerp_wxyz(np.roll(rq_xyzw[i0], 1),
                               np.roll(rq_xyzw[i1], 1), a)
        jp[k] = (1 - a) * dof[i0] + a * dof[i1]

    # frames where RSI would seed garbage: per-frame dof jumps that are
    # already flagged at PHC level plus clip edges (velocity endpoints
    # are forward/backward diffs)
    jumps = np.abs(np.diff(jp, axis=0)).max(axis=1)
    bad = sorted(set((np.nonzero(jumps > jump_bad_rad)[0] + 1).tolist()
                     + list(range(edge_skip))
                     + list(range(max(0, n20 - edge_skip), n20))))

    phc_meta = clip.get("meta", {})
    cid = clip_id or phc_meta.get("clip_id", "<unnamed>")
    states = {
        "root_pos": rpos.astype(np.float32),
        "root_rot": rquat.astype(np.float32),
        "root_vel_local": np.zeros((n20, 3), np.float32),
        "root_angular_vel_local": np.zeros((n20, 3), np.float32),
        "joint_pos": jp.astype(np.float32),
        "joint_vel": np.zeros((n20, N_DOF), np.float32),
    }
    meta: Dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "clip_id": cid,
        "robot": "humanoid28",
        "joint_space": "humanoid28",
        "dof_names": dof_names,
        "source": {"type": "phc", "bundle": bundle, "phc_clip_id": cid,
                   "resample": "30fps lerp/slerp -> 20Hz"},
        "phc_issues": list(phc_meta.get("issues", [])),
        "phc_quality_level": phc_meta.get("quality_level"),
        "labels": list(phc_meta.get("labels", [])),
        "split": phc_meta.get("split"),
        "rsi_bad_frames": bad,
        "notes": "converted from PHC kinematic retarget; velocities "
                 "rederived by centered finite difference — physical "
                 "executability NOT verified",
    }
    if meta_extra:
        meta.update(meta_extra)
    meta = record_lineage(meta, "convert", parent=cid,
                          fmt="phc->coreclip", src_fps=fps)
    ra = jp.astype(np.float32) if with_ref_action else None
    if ra is not None:
        meta["ref_action_semantics"] = "pd_target_rad"
    kp_names, kp = fk_keypoints(states, dof_names)
    meta["keypoint_names"] = kp_names

    cc = CoreClip(states, meta, ra, kp)
    cc = rederive_velocities(cc)          # fills the 3 velocity fields
    cc.meta["clip_id"] = cid              # rederive appends '_vel'
    cc.meta["source"] = meta["source"]
    return cc
