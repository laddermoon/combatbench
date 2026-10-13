"""Robot kinematic model for humanoid21 retargeting (robot A subtree).

Orientation-only chain compositor built from robot_spec.json (compiled-model
facts) + body local offsets read from the compiled MuJoCo model. Used by
`retarget.py` for per-chain direction matching; MuJoCo mj_forward is used for
final position/keypoint extraction and validation.

MuJoCo body-joint semantics: a joint defined inside body B rotates B relative
to its parent about a fixed axis in B's local frame; multiple joints in one
body compose in XML order: R_B = R_parent · R_axis1(q1) · R_axis2(q2) ...
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SPEC = json.loads((Path(__file__).parent / "robot_spec.json").read_text())
XML = PROJECT_ROOT / "envs/humanoid21/battle_circular_v2.xml"

CONTROLLED_JOINTS: List[str] = SPEC["controlled_joints_order"]

# joint name -> (body_name_suffix, axis_local, (lo, hi) rad)
_JSPEC = {
    j["name"]: (j["body"].removesuffix("_a"), np.asarray(j["axis_local"], float),
                tuple(j["range_rad"]))
    for j in SPEC["joints"]
}

# kinematic tree: body -> (parent_body, [joints in XML order])
# joint names here are the suffix-less CONTROLLED_JOINTS names
BODY_TREE: Dict[str, Tuple[Optional[str], List[str]]] = {
    "torso":       (None,          []),                      # freejoint (root quat)
    "waist_lower": ("torso",       ["abdomen_z", "abdomen_y"]),
    "pelvis":      ("waist_lower", ["abdomen_x"]),
    "thigh_right": ("pelvis",      ["hip_x_right", "hip_z_right", "hip_y_right"]),
    "shin_right":  ("thigh_right", ["knee_right"]),
    "foot_right":  ("shin_right",  ["ankle_y_right", "ankle_x_right"]),
    "thigh_left":  ("pelvis",      ["hip_x_left", "hip_z_left", "hip_y_left"]),
    "shin_left":   ("thigh_left",  ["knee_left"]),
    "foot_left":   ("shin_left",   ["ankle_y_left", "ankle_x_left"]),
    "upper_arm_right": ("torso",   ["shoulder1_right", "shoulder2_right"]),
    "lower_arm_right": ("upper_arm_right", ["elbow_right"]),
    "hand_right":  ("lower_arm_right", []),
    "upper_arm_left":  ("torso",   ["shoulder1_left", "shoulder2_left"]),
    "lower_arm_left":  ("upper_arm_left", ["elbow_left"]),
    "hand_left":   ("lower_arm_left", []),
    "head":        ("torso",       []),
}

# body-local direction of the "bone" (toward the distal joint) — used as the
# matching direction. Derived from model body child offsets; listed explicitly
# for transparency (verified against the compiled model at runtime in
# `check_bone_dirs`).
BONE_DIR_LOCAL: Dict[str, np.ndarray] = {
    "waist_lower": np.array([0, 0, -1.0]),          # torso -> pelvis (down)
    "pelvis":      np.array([0, 0, -1.0]),          # pelvis -> thighs (down)
    "thigh_right": np.array([0, 0.02497, -0.99969]),  # -> shin at (0,.01,-.4)
    "shin_right":  np.array([0, 0, -1.0]),            # -> foot at (0,0,-.39)
    "foot_right":  np.array([1.0, 0, 0]),             # toes forward +x
    "thigh_left":  np.array([0, -0.02497, -0.99969]), # shin pos (0,-.01,-.4)
    "shin_left":   np.array([0, 0, -1.0]),
    "foot_left":   np.array([1.0, 0, 0]),
    "upper_arm_right": np.array([1, -1, -1]) / np.sqrt(3),   # -> lower_arm (.18,-.18,-.18)
    "lower_arm_right": np.array([1, 1, 1]) / np.sqrt(3),     # -> hand (.18,.18,.18)
    "upper_arm_left":  np.array([1, 1, -1]) / np.sqrt(3),
    "lower_arm_left":  np.array([1, -1, 1]) / np.sqrt(3),
}


def axis_rot(axis: np.ndarray, q: float) -> np.ndarray:
    """Rodrigues rotation of angle q about unit axis (pure numpy, no scipy)."""
    c, s = np.cos(q), np.sin(q)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K


def _quat_mat(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])


# ---------- fast analytic position FK (verified vs mj_forward) ----------
# Per MuJoCo semantics each joint rotates its body about the joint anchor
# (jnt.pos, expressed in the body's rest frame). Composing joints in order:
#   p += R @ (I - R_axis(q)) @ a_j      R @= R_axis(q)
# where R is the cumulative body-frame rotation and a_j the constant anchor.


def _load_geom_consts():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(XML))
    body_pos, body_quat, jnt_pos = {}, {}, {}
    for b in BODY_TREE:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{b}_a")
        body_pos[b] = m.body_pos[bid].astype(float).copy()
        body_quat[b] = m.body_quat[bid].astype(float).copy()  # wxyz, rest frame
    for j in SPEC["joints"]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j["full_name"])
        jnt_pos[j["name"]] = m.jnt_pos[jid].astype(float).copy()
    return body_pos, body_quat, jnt_pos


_BODY_POS, _BODY_QUAT, _JNT_POS = _load_geom_consts()


def _chain(body: str) -> List[str]:
    chain = []
    b = body
    while b is not None:
        chain.append(b)
        b = BODY_TREE[b][0]
    return chain[::-1]


def fk_positions(q: Dict[str, float], root_pos: np.ndarray,
                 root_quat_wxyz: np.ndarray,
                 needed: Optional[set] = None
                 ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """World position + orientation for each body (suffix-less names).

    Identical math to mj_kinematics for this model (verified against
    mj_forward at 1e-15). If `needed` is given, only the ancestor chains of
    those bodies are computed — much cheaper inside solver inner loops.
    """
    if needed is None:
        order = [b for b in BODY_TREE if b != "torso"]
    else:
        req = set()
        for b in needed:
            req.update(_chain(b))
        order = [b for b in BODY_TREE if b in req and b != "torso"]
    pos: Dict[str, np.ndarray] = {}
    rot: Dict[str, np.ndarray] = {}
    pos["torso"] = np.asarray(root_pos, float)
    rot["torso"] = _quat_mat(*root_quat_wxyz)
    for b in order:
        parent, joints = BODY_TREE[b]
        p = pos[parent] + rot[parent] @ _BODY_POS[b]
        R = rot[parent] @ _quat_mat(*_BODY_QUAT[b])
        for jn in joints:
            _, ax, _ = _JSPEC[jn]
            Q = axis_rot(ax, q.get(jn, 0.0))
            a = _JNT_POS[jn]
            p = p + R @ (a - Q @ a)   # rotate about anchor
            R = R @ Q
        pos[b] = p
        rot[b] = R
    return pos, rot


def body_world_rot(body: str, q: Dict[str, float],
                   R_torso: np.ndarray) -> np.ndarray:
    """World orientation of `body` given joint angles (rad) and torso rotation.

    Walks BODY_TREE upward composing parent · Π R_axis_local(q). Only
    orientation — joint anchor positions do not affect it.
    """
    chain = []
    b = body
    while b is not None:
        chain.append(b)
        b = BODY_TREE[b][0]
    chain.reverse()
    R = R_torso.copy()
    for bname in chain[1:]:
        for jn in BODY_TREE[bname][1]:
            _, ax, _ = _JSPEC[jn]
            R = R @ axis_rot(ax, q.get(jn, 0.0))
    return R


def world_dir(body: str, q: Dict[str, float], R_torso: np.ndarray) -> np.ndarray:
    return body_world_rot(body, q, R_torso) @ BONE_DIR_LOCAL[body]


def joint_limits(name: str) -> Tuple[float, float]:
    return _JSPEC[name][2]


def knee_axis_world(side: str, q: Dict[str, float], R_torso: np.ndarray) -> np.ndarray:
    """World direction of the knee flexion axis for the hip twist constraint."""
    return body_world_rot(f"thigh_{side}", q, R_torso) @ np.array([0, -1.0, 0])


class RobotFK:
    """MuJoCo-backed FK for positions/verification (robot A only)."""

    def __init__(self):
        import mujoco
        self.mujoco = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        self.q0 = self.m.qpos0[:28].copy()  # robot A: freejoint(7)+21
        # qpos address of each controlled joint within robot A's 28-dof slice
        self.qadr = {}
        spec_by_full = {j["full_name"]: j for j in SPEC["joints"]}
        for name in CONTROLLED_JOINTS:
            self.qadr[name] = spec_by_full[f"{name}_a"]["qposadr"]

    def forward(self, qvec28: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """qvec28: robot-A qpos slice [root pos3, quat wxyz4, 21 joints].
        Returns (body_pos, body_rot_mat) dicts keyed by suffix-less name."""
        self.d.qpos[:28] = qvec28
        self.d.qpos[28:] = self.m.qpos0[28:]  # robot B frozen at default
        self.mujoco.mj_forward(self.m, self.d)
        pos, rot = {}, {}
        for b in BODY_TREE:
            bid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_BODY, f"{b}_a")
            pos[b] = self.d.xpos[bid].copy()
            rot[b] = self.d.xmat[bid].reshape(3, 3).copy()
        return pos, rot

    def qvec(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray,
             q: Dict[str, float]) -> np.ndarray:
        v = self.q0.copy()
        v[0:3] = root_pos
        v[3:7] = root_quat_wxyz
        for name, val in q.items():
            v[self.qadr[name]] = val
        return v
