"""起点状态工具：point5 姿态 → 仿真器 core_state 快照。

core_state 字段与 ``Humanoid21Simulator.set_core_state`` 对齐：
root_pos(3) / root_rot(4, wxyz) / root_vel_local(3) /
root_angular_vel_local(3) / joint_pos_norm(21) / joint_vel_norm(21)。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation

from point5.humanoid21.robot_state import RobotState, joint_ranges


def _pry_to_quat_wxyz(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw)·Ry(pitch)·Rx(roll) → quat [w,x,y,z]。"""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    # q = qz ⊗ qy ⊗ qx
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return np.array([w, x, y, z], np.float64)


def robot_state_to_core(rs: RobotState) -> Dict[str, np.ndarray]:
    """RobotState → core_state dict。

    注意：RobotState 处于 P5 标准帧（yaw≡0、踝中点原点），写入仿真
    即以该帧为世界系姿态/位置——对技能起点而言足够（与场地无关）。
    速度字段从 root_linvel/root_angvel 转机体系（qvel 语义：
    线速度世界系乘 R^T、角速度本就在机体系——与 simulator 注释一致，
    但这里输入已是 P5 帧量，按同规则变换）。"""
    rng = joint_ranges()
    ref = (rng[:, 0] + rng[:, 1]) / 2.0
    scale = (rng[:, 1] - rng[:, 0]) / 2.0
    jpos_norm = (np.asarray(rs.qpos, np.float64) - ref) / scale
    jvel_norm = np.asarray(rs.qvel, np.float64) / scale
    pitch, roll, yaw = (float(v) for v in rs.root_orientation)
    root_rot = _pry_to_quat_wxyz(pitch, roll, yaw)
    Rm = Rotation.from_quat(
        [root_rot[1], root_rot[2], root_rot[3], root_rot[0]])
    vel_local = Rm.inv().apply(np.asarray(rs.root_linvel, np.float64))
    return {
        "root_pos": np.asarray(rs.root_position, np.float64),
        "root_rot": root_rot,
        "root_vel_local": vel_local,
        "root_angular_vel_local": np.asarray(rs.root_angvel, np.float64),
        "joint_pos_norm": jpos_norm,
        "joint_vel_norm": jvel_norm,
    }


def ready_core_state() -> Dict[str, np.ndarray]:
    """全部技能的初始候选起点：point5 READY 准备姿（零速度）。"""
    return robot_state_to_core(RobotState.standard("ready"))


def to_jsonable(state: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {k: np.asarray(v).tolist() for k, v in state.items()}
