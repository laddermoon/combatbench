"""Converters between the retarget mocap format and CoreClip.

Convention mapping (see CORECLIP_SPEC.md §4):

  RefClip (retarget npz)            CoreClip (CoreState)
  --------------------------      ----------------------------
  joint_angles (rad)          ->    joint_pos_norm = (q - mid)/half
  joint_velocities (rad/s)    ->    joint_vel_norm = qv/half
  root_linvel (WORLD)         ->    root_vel_local = R^T @ v_world
  root_angvel (WORLD)         ->    root_angular_vel_local = R^T @ w_world
  root_quat (wxyz)            ->    root_rot (wxyz, unchanged)
  120Hz frames                ->    RefClip already downsamples to 20Hz

Normalization basis is Humanoid21 joint limits (robot_spec /
Humanoid21Meta) — the same mid/half used by RefClip and the
simulator, verified by test_norm_params_match_sim.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as _R

from .core_clip import (CONTROL_DT, CoreClip, SPEC_VERSION,
                        record_lineage)


def _w_to_local(q_wxyz: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """World-frame vector -> body-local frame via R^T."""
    rot = _R.from_quat(np.roll(q_wxyz, -1))        # wxyz -> xyzw
    return rot.inv().apply(v_world)


# ----------------------------------------------------------------------
# PHC bundle (mocap/humanoid21/gmr pipeline, DATASPEC v2) -> CoreClip
# ----------------------------------------------------------------------

_MOCAP_XML = (Path(__file__).resolve().parents[1] / 'gmr'
              / 'humanoid21_mocap.xml')
_NORM_CACHE: Dict[str, Any] = {}


def _phc_norm(dof_names):
    """(mid, half) normalization basis per PHC dof name, from the mocap
    XML jnt_range — identical to Humanoid21Meta norm params (ranges
    verified equal to the battle model)."""
    key = tuple(dof_names)
    if key not in _NORM_CACHE:
        import mujoco
        mm = mujoco.MjModel.from_xml_path(str(_MOCAP_XML))
        lo = np.array([mm.jnt_range[mm.joint(f'{n}_a').id, 0]
                       for n in dof_names])
        hi = np.array([mm.jnt_range[mm.joint(f'{n}_a').id, 1]
                       for n in dof_names])
        _NORM_CACHE[key] = ((lo + hi) / 2.0, (hi - lo) / 2.0)
    return _NORM_CACHE[key]


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
    """PHC clip_dict (30 fps positions) -> CoreClip (20 Hz CoreStates).

    Positions/dof are resampled onto the 20 Hz grid (lerp pos/dof, slerp
    root_rot); velocities are then derived on that grid by centered
    finite differences (rederive_velocities) so each stored frame is a
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
    mid, half = _phc_norm(dof_names)

    # resample onto the 20Hz grid: 20Hz time k*0.05 = 1.5k in 30Hz frames
    n20 = int(np.floor((T30 - 1) / (fps * CONTROL_DT))) + 1
    rpos, rquat, jpn = (np.empty((n20, 3)), np.empty((n20, 4)),
                      np.empty((n20, 21)))
    for k in range(n20):
        f = k * fps * CONTROL_DT          # 1.5*k
        i0 = min(int(f), T30 - 1)
        i1 = min(i0 + 1, T30 - 1)
        a = f - i0
        rpos[k] = (1 - a) * rp[i0] + a * rp[i1]
        rquat[k] = _slerp_wxyz(np.roll(rq_xyzw[i0], 1),
                               np.roll(rq_xyzw[i1], 1), a)
        jpn[k] = ((1 - a) * dof[i0] + a * dof[i1] - mid) / half

    # frames where RSI would seed garbage: per-frame dof jumps that are
    # already flagged at PHC level plus clip edges (velocity endpoints
    # are forward/backward diffs)
    jumps = np.abs(np.diff(jpn * half[None], axis=0)).max(axis=1)
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
        "joint_pos_norm": jpn.astype(np.float32),
        "joint_vel_norm": np.zeros((n20, 21), np.float32),
    }
    meta: Dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "clip_id": cid,
        "robot": "humanoid21",
        "joint_space": "humanoid21",
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
    ra = jpn.astype(np.float32) if with_ref_action else None
    if ra is not None:
        meta["ref_action_semantics"] = "pd_target_norm"

    cc = CoreClip(states, meta, ra)
    cc = rederive_velocities(cc)          # fills the 3 velocity fields
    cc.meta["clip_id"] = cid              # rederive appends '_vel'
    cc.meta["source"] = meta["source"]
    return cc


def refclip_to_coreclip(ref, clip_id: Optional[str] = None,
                        start: int = 0, end: Optional[int] = None,
                        with_ref_action: bool = True,
                        meta_extra: Optional[Dict[str, Any]] = None
                        ) -> CoreClip:
    """Convert a loaded RefClip (20Hz) into a CoreClip.

    ``ref`` is a ``mocap.humanoid21.tracking.ref_data.RefClip``.
    ``ref_action[k]`` is the normalized PD target = the reference
    joint angles at frame k (``pd_target_norm``, causal).
    """
    k0, k1 = int(start), (ref.n_ctrl if end is None else int(end))
    if not (0 <= k0 < k1 <= ref.n_ctrl):
        raise ValueError(f"bad frame range [{k0},{k1}) "
                         f"for clip of {ref.n_ctrl} ctrl frames")

    qn = ref.q_norm[k0:k1]
    qvn = ref.qv_norm[k0:k1]
    rpos = ref.root_pos[k0:k1]
    rquat = ref.root_quat[k0:k1]

    T = k1 - k0
    vl = np.empty((T, 3), np.float64)
    av = np.empty((T, 3), np.float64)
    for i in range(T):
        vl[i] = _w_to_local(rquat[i], ref.root_linvel[k0 + i])
        av[i] = _w_to_local(rquat[i], ref.root_angvel[k0 + i])

    states = {
        "root_pos": rpos.astype(np.float32),
        "root_rot": rquat.astype(np.float32),
        "root_vel_local": vl.astype(np.float32),
        "root_angular_vel_local": av.astype(np.float32),
        "joint_pos_norm": qn.astype(np.float32),
        "joint_vel_norm": qvn.astype(np.float32),
    }
    meta: Dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "clip_id": clip_id or f"{ref.clip_id}_core",
        "robot": "humanoid21",
        "joint_space": "humanoid21",
        "source": {"type": "refclip", "ref_clip_id": ref.clip_id,
                   "ctrl_range": [k0, k1]},
        "notes": "converted from retarget clip; kinematic reference "
                 "only — physical executability NOT verified",
    }
    if meta_extra:
        meta.update(meta_extra)
    meta = record_lineage(meta, "convert", parent=ref.clip_id,
                          ctrl_range=[k0, k1])
    ra = qn.astype(np.float32) if with_ref_action else None
    if ra is not None:
        meta["ref_action_semantics"] = "pd_target_norm"
    return CoreClip(states, meta, ra)


def model_sha256(sim) -> str:
    """sha256 of the arena XML the simulator compiled — for meta.model."""
    return hashlib.sha256(Path(sim.ARENA_XML).read_bytes()).hexdigest()
