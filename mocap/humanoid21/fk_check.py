"""Numerical FK validation against the CMU-official convention.

Checks:
  1. REST POSE: with all-zero AMC DOFs, every bone's world direction must equal
     its ASF `direction` (declared in global coordinates). |err| ~ 1e-15.
  2. CONNECTIVITY: every non-root bone's joint coincides with parent's distal.
  3. REAL FRAME: run a real AMC frame through ref_fk and legacy fk.py, print
     per-bone position deltas and skeleton stats (height, foot height spread).

Usage:
    PYTHONPATH=. python3 -m mocap.humanoid21.fk_check \
        --asf baseline/humanoid21/mocap/data/raw/13.asf \
        --amc baseline/humanoid21/mocap/data/raw/13_17.amc [--frame 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from baseline.humanoid21.mocap.asf_parser import parse_asf
from baseline.humanoid21.mocap.amc_parser import parse_amc
from baseline.humanoid21.mocap.fk import compute_fk

from mocap.humanoid21.ref_fk import RefSkeleton, _Rz_Ry_Rx, fk_frame
from mocap.humanoid21.retarget import _abdomen_angles


def check_rotation_order() -> None:
    x, y, z = 23.0, -34.0, 57.0
    rx, ry, rz = (Rotation.from_euler(axis, angle, degrees=True).as_matrix()
                  for axis, angle in zip("xyz", (x, y, z)))
    np.testing.assert_allclose(_Rz_Ry_Rx(x, y, z), rz @ ry @ rx, atol=1e-14)
    angles = _abdomen_angles(np.eye(3), rz @ ry @ rx)
    np.testing.assert_allclose(angles, np.deg2rad([x, y, z]), atol=1e-14)


def check_elbow_flexion(skel: RefSkeleton) -> None:
    frame = {"root": np.zeros(6), "lradius": np.array([45.0]),
             "rradius": np.array([45.0])}
    joint, distal, _ = fk_frame(skel, frame)
    for side in ("l", "r"):
        forearm = distal[f"{side}radius"] - joint[f"{side}radius"]
        assert forearm[2] > 0.5 * np.linalg.norm(forearm), (side, forearm)


def check_reference_arms(skel: RefSkeleton, frames, clip: str) -> None:
    if clip == "07_01":
        joint, distal, _ = fk_frame(skel, frames[0])
        for side in ("l", "r"):
            upper = distal[f"{side}humerus"] - joint[f"{side}humerus"]
            assert upper[1] / np.linalg.norm(upper) < -0.85, (side, upper)
    elif clip == "13_17":
        joint, distal, _ = fk_frame(skel, frames[236])
        across = joint["rhumerus"] - joint["lhumerus"]
        facing = np.cross([0.0, 1.0, 0.0], across)
        facing /= np.linalg.norm(facing)
        reach = distal["rradius"] - joint["rhumerus"]
        assert reach @ facing / np.linalg.norm(reach) > 0.6, reach

        torso = []
        for fr in frames[335:371]:
            joints, ends, rotations = fk_frame(skel, fr)
            assert ends["head"][1] > joints["root"][1]
            torso.append(rotations["thorax"])
        relative = Rotation.from_matrix(torso[1:]) * Rotation.from_matrix(torso[:-1]).inv()
        assert np.rad2deg(relative.magnitude()).max() < 15.0


def check_rest_pose(skel: RefSkeleton, asf) -> None:
    zero_frame = {"root": np.zeros(6)}
    for name, dofs in skel.dof.items():
        if name == "root" or not dofs:
            continue
        zero_frame[name] = np.zeros(len(dofs))
    _, distal, F = fk_frame(skel, zero_frame)
    worst, worst_bone = 0.0, None
    for n in skel.bone_names:
        if n == "root":
            continue
        wdir = F[n] @ skel.dir_local[n]
        want = np.asarray(asf.bones[n].direction, dtype=np.float64)
        err = float(np.max(np.abs(wdir - want)))
        if err > worst:
            worst, worst_bone = err, n
    print(f"[rest-pose direction] max |world_dir - ASF direction| = {worst:.3e} (bone {worst_bone})")
    assert worst < 1e-9, "rest-pose direction mismatch — FK convention wrong"


def check_connectivity(skel: RefSkeleton, frame) -> float:
    joint, distal, _ = fk_frame(skel, frame)
    worst = 0.0
    for n in skel.bone_names:
        if n == "root":
            continue
        err = float(np.max(np.abs(joint[n] - distal[skel.parent[n]])))
        worst = max(worst, err)
    print(f"[connectivity] max |joint(child) - distal(parent)| = {worst:.3e}")
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asf", required=True)
    ap.add_argument("--amc", required=True)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    asf = parse_asf(args.asf)
    skel = RefSkeleton.from_asf(asf)
    frames = parse_amc(args.amc)
    frame = frames[args.frame]

    check_rotation_order()
    check_rest_pose(skel, asf)
    check_elbow_flexion(skel)
    check_reference_arms(skel, frames, Path(args.amc).stem)
    check_connectivity(skel, frame)

    joint_r, distal_r, F_r = fk_frame(skel, frame)
    joints_old = compute_fk(asf, frame)  # legacy implementation

    print("\n=== ref vs legacy FK joint positions (ASF units x 0.45) ===")
    for n in skel.bone_names:
        old = joints_old[n][1]
        new = joint_r[n]
        d = float(np.linalg.norm(new - old))
        print(f"{n:12s} ref={np.round(new,2)}  legacy={np.round(old,2)}  |d|={d:.2f}")

    feet = np.array([distal_r["lfoot"], distal_r["rfoot"], distal_r["ltoes"], distal_r["rtoes"]])
    head = distal_r["head"]
    print(f"\nhead distal y={head[1]:.2f}  foot/toe distal y range "
          f"[{feet[:,1].min():.2f}, {feet[:,1].max():.2f}]  root y={joint_r['root'][1]:.2f}")


if __name__ == "__main__":
    main()
