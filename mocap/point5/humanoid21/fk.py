"""Forward kinematics for Humanoid21 landmark and body positions.

Loads the arena model once (read-only, never steps physics). Joint values
are raw qpos radians in ``CONTROLLED_JOINTS`` order; root is (pos, quat_wxyz).
"""
from __future__ import annotations

from typing import Dict, Optional

import mujoco
import numpy as np

from envs.humanoid21.meta import Humanoid21Meta
from envs.humanoid21.simulator import Humanoid21Simulator

from .mechanics import LANDMARKS

_ROOT_DOFS = 6
_ROOT_QPOS = 7


class Humanoid21FK:
    """Read-only kinematics model for one or both arena robots."""

    def __init__(self) -> None:
        self.model = mujoco.MjSpec.from_file(
            Humanoid21Simulator.ARENA_XML).compile()
        self.data = mujoco.MjData(self.model)
        self._tables = Humanoid21Meta.build_runtime_tables(self.model)
        self._robots = self._tables["robots"]

    def controlled_joints(self, robot_id: str) -> list:
        return list(self._robots[robot_id]["controlled_joint_names"])

    def joint_ranges(self, robot_id: str) -> np.ndarray:
        return np.asarray(self._robots[robot_id]["jnt_ranges"],
                          dtype=np.float64)

    def set_pose(self, robot_id: str, root_pos, root_quat_wxyz,
                 joint_pos: np.ndarray) -> None:
        """Write ``qpos`` for one robot and refresh kinematics."""
        r = self._robots[robot_id]
        joints = np.asarray(joint_pos, dtype=np.float64)
        if joints.shape != (len(r["qpos_indices"]),):
            raise ValueError(
                f"joint_pos must be ({len(r['qpos_indices'])},), "
                f"got {joints.shape}")
        adr = int(r["root_qpos_adr"])
        self.data.qpos[adr:adr + 3] = np.asarray(root_pos, dtype=np.float64)
        self.data.qpos[adr + 3:adr + _ROOT_QPOS] = np.asarray(
            root_quat_wxyz, dtype=np.float64)
        self.data.qpos[np.asarray(r["qpos_indices"])] = joints
        mujoco.mj_kinematics(self.model, self.data)

    def landmark_positions(self, robot_id: str) -> np.ndarray:
        """(5,3) world positions of the business points, POINT_NAMES order."""
        r = self._robots[robot_id]
        out = np.zeros((5, 3), dtype=np.float64)
        for i, name in enumerate(_lm_order()):
            body, off = LANDMARKS[name]
            bid = r["keypoint_body_ids"].get(body)
            if bid is None:
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY,
                    f"{body}{r['suffix']}")
            out[i] = (np.asarray(self.data.xpos[bid])
                      + np.asarray(self.data.xmat[bid]).reshape(3, 3) @ off)
        return out

    def body_pose(self, robot_id: str) -> Dict[str, np.ndarray]:
        """All body world poses for rendering: name -> (xpos, xmat)."""
        r = self._robots[robot_id]
        result = {}
        for bid, name in zip(r["body_ids_sorted"], r["body_names"]):
            result[name] = (np.asarray(self.data.xpos[bid]).copy(),
                            np.asarray(self.data.xmat[bid]).reshape(3, 3)
                            .copy())
        return result


def _lm_order():
    from mocap.point5.contracts import POINT_NAMES
    return POINT_NAMES
