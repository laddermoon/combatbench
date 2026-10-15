"""M1-W1: coordinate & scale calibration CMU-ASF -> humanoid21 MuJoCo world.

Findings driving this calibration:

- CMU world: y-up; the ASF rest pose (all AMC DOFs = 0) is a T-pose with the
  subject facing +z (toes point +z), arms along +/-x, standing legs along -y.
- humanoid21 world: z-up, robot rest faces +x.
- Rigid conversion v_mj = C @ v_cmu with C = cyclic permutation
  (x,y,z)_cmu -> (z,x,y)_mj maps y_cmu->z_mj (up) and z_cmu->y... wait, see
  below; the exact C is chosen so human forward +z_cmu lands on robot +x.

  v_mj = (z_cmu, x_cmu, y_cmu), i.e. C = [[0,0,1],[1,0,0],[0,1,0]]
  (up: +y_cmu -> +z_mj; forward: +z_cmu -> +x_mj; right: -x_cmu -> +y_mj;
  det(C)=+1, proper rotation).
  Rotations conjugate: R_mj = C @ R_cmu @ C.T; quaternions wxyz transform the
  same way via matrix conversion.

- Scale: robot standing hip-joint height 0.817 m / human hip-joint height
  ~6.9 CMU units  =>  ~0.118 m per CMU unit (provisional, revisit in M2 if
  stride/foot placement shows systematic mismatch).

Run as a script to emit calibration.json with numeric self-checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mocap.humanoid21.parsers.asf_parser import parse_asf
from mocap.humanoid21.ref_fk import RefSkeleton, fk_frame

# (x,y,z)_cmu -> (z,x,y)_mj : up->up, forward(+z_cmu)->forward(+x_mj)
C_CMU2MJ = np.array([[0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]])

ROBOT_XML = PROJECT_ROOT / "envs/humanoid21/battle_circular_v2.xml"
DEFAULT_ASF = Path("/data1/dev/combatbench_mocap/raw/13.asf")
OUT = Path(__file__).parent / "calibration.json"


def cmu_pos_to_mj(p: np.ndarray) -> np.ndarray:
    """(…,3) CMU positions -> MuJoCo world positions (still CMU units)."""
    return np.asarray(p) @ C_CMU2MJ.T


def cmu_rot_to_mj(R: np.ndarray) -> np.ndarray:
    """(3,3) or (…,3,3) CMU rotation matrix -> MuJoCo-world rotation."""
    return C_CMU2MJ @ np.asarray(R) @ C_CMU2MJ.T


def robot_geometry() -> dict:
    """Standing geometry of robot A subtree at qpos0."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    g = lambda n: d.xpos[bid(n)].copy()
    return {
        "torso_z": float(g("torso_a")[2]),
        "head_z": float(g("head_a")[2]),
        "pelvis_z": float(g("pelvis_a")[2]),
        "hip_joint_z": float((g("thigh_right_a")[2] + g("thigh_left_a")[2]) / 2),
        "knee_z": float(g("shin_right_a")[2]),
        "ankle_z": float(g("foot_right_a")[2]),
        "hip_width": float(abs(g("thigh_left_a")[1] - g("thigh_right_a")[1])),
        "shoulder_z": float(g("upper_arm_right_a")[2]),
    }


def human_geometry(asf_path: str) -> dict:
    """Rest-pose (all-zero DOF) geometry of the ASF skeleton, CMU units."""
    skel = RefSkeleton.from_asf(parse_asf(asf_path))
    zero = {"root": np.zeros(6)}
    for n, dofs in skel.dof.items():
        if n != "root" and dofs:
            zero[n] = np.zeros(len(dofs))
    j, d, _ = fk_frame(skel, zero)
    sole = min(d["ltoes"][1], d["rtoes"][1], d["lfoot"][1], d["rfoot"][1])
    hip = (j["lfemur"] + j["rfemur"]) / 2
    ankle = (j["lfoot"] + j["rfoot"]) / 2
    return {
        "hip_height_units": float(hip[1] - sole),
        "ankle_height_units": float(ankle[1] - sole),
        "leg_length_units": float(hip[1] - ankle[1]),
        "total_height_units": float(d["head"][1] - sole),
        "shoulder_width_units": float(abs(j["lhumerus"][0] - j["rhumerus"][0])),
        "rest_hip_y": float(hip[1]),
        "rest_sole_y": float(sole),
    }


def main() -> None:
    rob = robot_geometry()
    hum = human_geometry(str(DEFAULT_ASF))
    scale = rob["hip_joint_z"] / hum["hip_height_units"]
    cal = {
        "cmu_to_mujoco": "v_mj = (z_cmu, x_cmu, y_cmu); R_mj = C R_cmu C^T",
        "C": C_CMU2MJ.tolist(),
        "meters_per_cmu_unit": scale,
        "robot": rob,
        "human_rest": hum,
        "derived": {
            "human_height_m": hum["total_height_units"] * scale,
            "human_leg_m": hum["leg_length_units"] * scale,
            "robot_leg_m": rob["hip_joint_z"] - rob["ankle_z"],
        },
        "note": "scale = robot hip height / human rest hip height; provisional, "
                "recalibrate in M2 if foot placement is systematically off.",
    }
    OUT.write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal["derived"], indent=2))
    print(f"scale = {scale:.4f} m/unit; human height = {cal['derived']['human_height_m']:.2f} m")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
