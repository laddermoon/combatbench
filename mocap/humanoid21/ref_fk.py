"""Reference-compatible ASF/AMC forward kinematics.

Faithful reimplementation of CMU's official ASF/AMC viewer
(mocap.cs.cmu.edu `asf_amc_viewer`: skeleton.cxx / display.cxx / transform.cxx),
which differs from `baseline/humanoid21/mocap/fk.py` in two ways:

1. ASF `direction` is a GLOBAL-frame vector (see skeleton.cxx comment
   "the bone's direction vector in global coordinate"); it is mapped to the
   bone local frame via dir_local = C(b)^T @ dir_global before FK.
2. A child's joint position equals the PARENT bone's distal endpoint:
   joint(child) = joint(parent) + F(parent) @ dir_local(parent) * len(parent),
   where F(b) is the bone's post-DOF world frame. Legacy fk.py instead used
   the child's own direction/length in the parent frame — producing
   disconnected/displaced segments (visible as "bizarre" preview renders).

Frame recursion (per official drawBone / ComputeRotationToParentCoordSystem):
    rot_pc(b) = C(parent)^T @ C(b)
    F(b)      = F(parent) @ rot_pc(b) @ R_dof(b)
    joint(b)  = joint(parent) + F(parent) @ dir_local(parent) * len(parent)
    distal(b) = joint(b) + F(b) @ dir_local(b) * len(b)   # children attach here

DOF rotation composition follows the viewer's hardcoded glRotate order
(post-multiply Rz, then Ry, then Rx => R_dof = Rz @ Ry @ Rx, vector sees the
listed 'rx ry rz' order applied x-first). All CMU bones use 'rx ry rz' or
subsets, matching this convention.

Root is special-cased like the viewer: orientation = Rx(rx) @ Ry(ry) @ Rz(rz)
(scipy intrinsic `XYZ`), root axis/orientation fields are ignored (all-zero in
CMU files), and root translations use AMC units (scaled by units_length).

Rest-pose identity (zero AMC DOFs): F(b) = C(b), so each bone's world
direction equals its ASF global `direction`. This is used as a numerical
self-check in `mocap/humanoid21/fk_check.py`.

Earlier code mistakenly swapped SciPy's intrinsic uppercase and extrinsic
lowercase Euler conventions. The extra radius-rx sign reversal added to
compensate for backward bending was also wrong; with the proper axis/DOF
order, positive radius rx folds the forearm forward without a sign change.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from baseline.humanoid21.mocap.asf_parser import Skeleton, parse_asf
from baseline.humanoid21.mocap.amc_parser import parse_amc


def _Rz_Ry_Rx(rx: float, ry: float, rz: float) -> np.ndarray:
    """Matrix Rz(rz) @ Ry(ry) @ Rx(rx) — extrinsic 'xyz' (x applied first)."""
    return Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()


def _Rx_Ry_Rz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Matrix Rx(rx) @ Ry(ry) @ Rz(rz) — intrinsic 'XYZ' (used for root)."""
    return Rotation.from_euler("XYZ", [rx, ry, rz], degrees=True).as_matrix()


@dataclass
class RefSkeleton:
    """ASF skeleton prepared for official-convention FK."""

    bone_names: List[str]               # DFS order from root
    parent: Dict[str, Optional[str]]
    C: Dict[str, np.ndarray]            # (3,3) axis matrix Rz@Ry@Rx
    dir_local: Dict[str, np.ndarray]    # C^T @ direction
    length: Dict[str, float]
    dof: Dict[str, List[str]]
    units_length: float

    @classmethod
    def from_asf(cls, asf: Skeleton) -> "RefSkeleton":
        names = list(asf.bones.keys())
        parent: Dict[str, Optional[str]] = {n: b.parent for n, b in asf.bones.items()}
        C, dir_local, length, dof = {}, {}, {}, {}
        for name, b in asf.bones.items():
            Cb = _Rz_Ry_Rx(*b.axis)
            C[name] = Cb
            dir_local[name] = Cb.T @ np.asarray(b.direction, dtype=np.float64)
            length[name] = float(b.length)
            dof[name] = list(b.dof)
        # DFS order from root (parents always precede children).
        order = ["root"]
        stack = ["root"]
        while stack:
            p = stack.pop()
            for ch in asf.bones[p].children:
                order.append(ch)
                stack.append(ch)
        assert len(order) == len(names), (
            f"hierarchy DFS reached {len(order)} of {len(names)} bones"
        )
        return cls(order, parent, C, dir_local, length, dof, asf.units_length)


def _dof_values(frame: Dict[str, np.ndarray], bone: str, dofs: List[str]) -> Tuple[float, float, float]:
    """Map a bone's AMC values to (drx, dry, drz) degrees via its dof list."""
    rx = ry = rz = 0.0
    vals = frame.get(bone)
    if vals is None:
        return rx, ry, rz
    if len(vals) != len(dofs):
        raise ValueError(
            f"bone {bone}: AMC gives {len(vals)} values but ASF dof has {len(dofs)}: {dofs}"
        )
    for name, v in zip(dofs, vals):
        if name == "rx":
            rx = v
        elif name == "ry":
            ry = v
        elif name == "rz":
            rz = v
        elif name in ("tx", "ty", "tz", "l"):
            continue  # translations not used by CMU bones (root handled separately)
        else:
            raise ValueError(f"bone {bone}: unknown dof token {name!r}")
    return rx, ry, rz


def fk_frame(
    skel: RefSkeleton,
    frame: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute one frame. Returns (joint_pos, distal_pos, F) dicts by bone name.

    joint_pos[b]: proximal joint of bone b (where its DOF rotation is applied).
    distal_pos[b]: distal end of bone b — where its children attach.
    F[b]: post-DOF world frame of bone b.
    Positions are in ASF units scaled by units_length (inches for CMU: x0.45).
    """
    root_vals = np.asarray(frame["root"], dtype=np.float64)
    if len(root_vals) != 6:
        raise ValueError(f"root must have 6 values, got {len(root_vals)}")
    u = skel.units_length

    joint: Dict[str, np.ndarray] = {}
    distal: Dict[str, np.ndarray] = {}
    F: Dict[str, np.ndarray] = {}

    joint["root"] = root_vals[0:3] * u
    F["root"] = _Rx_Ry_Rz(root_vals[3], root_vals[4], root_vals[5])
    distal["root"] = joint["root"]  # root dir = (0,0,0): children sit at root joint

    for b in skel.bone_names[1:]:
        p = skel.parent[b]
        rot_pc = skel.C[p].T @ skel.C[b]          # child-local -> parent-local
        Rm = F[p] @ rot_pc
        joint[b] = distal[p]
        rx, ry, rz = _dof_values(frame, b, skel.dof[b])
        F[b] = Rm @ _Rz_Ry_Rx(rx, ry, rz)
        distal[b] = joint[b] + F[b] @ (skel.dir_local[b] * skel.length[b] * u)
    return joint, distal, F


def fk_batch(
    skel: RefSkeleton,
    frames: List[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """All frames -> (joint_positions (T,N,3), distal_positions (T,N,3), names)."""
    names = skel.bone_names
    T = len(frames)
    J = np.zeros((T, len(names), 3))
    D = np.zeros((T, len(names), 3))
    for t, fr in enumerate(frames):
        j, d, _ = fk_frame(skel, fr)
        for i, n in enumerate(names):
            J[t, i] = j[n]
            D[t, i] = d[n]
    return J, D, names


def load_clip(asf_path: str, amc_path: str):
    """Parse ASF + AMC, return (RefSkeleton, frames)."""
    return RefSkeleton.from_asf(parse_asf(asf_path)), parse_amc(amc_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("asf")
    ap.add_argument("amc")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    skel, frames = load_clip(args.asf, args.amc)
    joint, distal, F = fk_frame(skel, frames[args.frame])
    for n in skel.bone_names:
        wdir = F[n] @ skel.dir_local[n]
        print(f"{n:12s} joint={np.round(joint[n],3)} distal={np.round(distal[n],3)} wdir={np.round(wdir,3)}")
