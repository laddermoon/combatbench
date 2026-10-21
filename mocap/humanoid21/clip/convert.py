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

from .core_clip import CoreClip, SPEC_VERSION


def _w_to_local(q_wxyz: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """World-frame vector -> body-local frame via R^T."""
    rot = _R.from_quat(np.roll(q_wxyz, -1))        # wxyz -> xyzw
    return rot.inv().apply(v_world)


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
    ra = qn.astype(np.float32) if with_ref_action else None
    if ra is not None:
        meta["ref_action_semantics"] = "pd_target_norm"
    return CoreClip(states, meta, ra)


def model_sha256(sim) -> str:
    """sha256 of the arena XML the simulator compiled — for meta.model."""
    return hashlib.sha256(Path(sim.ARENA_XML).read_bytes()).hexdigest()
