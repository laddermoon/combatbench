"""V1 clip 观测器：每步世界系运动学特征 + core_state 快照。

输出字段（全部世界系，供训练侧组装 ``skills.Clip``）：

    com(3)        全身质心（body_xipos 质量加权）
    ankle_l/r(3)  踝心（ankle_y 关节锚点，P5 语义）
    hand_l/r(3)   手
    root_pos(3)   torso
    root_pry(3)   pitch/roll/yaw（R=Rz(yaw)·Ry(pitch)·Rx(roll)）
    root_vel(3)   torso 世界线速度
    contact(2)    踝心 z < 0.10m 的 (left,right) 接地标志（v0 近似）

    snap_*        ``get_core_state()[agent_id]`` 平铺字段——该 agent
                  每帧的完整可重置快照，末帧即终止状态候选（起点池）。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from envs.framework.context import SimContext
from envs.framework.observer_plugin import BaseObserverPlugin

_SNAP_KEYS = ("root_pos", "root_rot", "root_vel_local",
              "root_angular_vel_local", "joint_pos_norm", "joint_vel_norm")
_CONTACT_ANKLE_Z = 0.10       # 踝心 z 低于此值视为接地（v0）


def _quat_to_pry(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quat → (pitch, roll, yaw)，R=Rz(yaw)·Ry(pitch)·Rx(roll)。"""
    Rm = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3],
                             q_wxyz[0]]).as_matrix()
    pitch = float(np.arcsin(np.clip(-Rm[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(Rm[2, 1], Rm[2, 2]))
    yaw = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
    return np.array([pitch, roll, yaw], np.float64)


class V1ClipObserver(BaseObserverPlugin):
    """逐帧世界系特征 + core_state 快照（每 agent 一个实例）。"""

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = str(agent_id)
        self._n: Dict[str, str] = {}
        self._out: Dict[str, Any] = {}

    def on_pre_episode(self, ctx: SimContext) -> None:
        sd = ctx.accessor.get_static_data()[self.agent_id]
        kb, kj = sd["keypoint_body_names"], sd["keypoint_joint_names"]
        self._n = dict(
            torso=kb["torso"], hand_l=kb["hand_left"],
            hand_r=kb["hand_right"], foot_l=kb["foot_left"],
            foot_r=kb["foot_right"],
            ank_l=kj["ankle_y_left"], ank_r=kj["ankle_y_right"])
        self._masses = sd["body_masses_by_name"]

    def on_post_action_step(self, ctx: SimContext) -> None:
        acc = ctx.accessor
        rd = acc.get_derived_state([self.agent_id])[self.agent_id]
        n = self._n
        names = list(rd["body_xipos"].keys())
        mvec = np.array([self._masses[bn] for bn in names], np.float64)
        pos = np.stack([rd["body_xipos"][bn] for bn in names])
        com = (mvec[:, None] * pos).sum(0) / mvec.sum()
        ank_l = np.asarray(rd["joint_world_anchor"][n["ank_l"]])
        ank_r = np.asarray(rd["joint_world_anchor"][n["ank_r"]])
        core = acc.get_core_state()[self.agent_id]
        out: Dict[str, Any] = {
            "com": com.tolist(),
            "ankle_l": ank_l.tolist(),
            "ankle_r": ank_r.tolist(),
            "hand_l": np.asarray(rd["body_xpos"][n["hand_l"]]).tolist(),
            "hand_r": np.asarray(rd["body_xpos"][n["hand_r"]]).tolist(),
            "root_pos": np.asarray(rd["body_xpos"][n["torso"]]).tolist(),
            "root_pry": _quat_to_pry(
                np.asarray(rd["body_xquat"][n["torso"]])).tolist(),
            "root_vel": np.asarray(
                rd["body_linvel_world"][n["torso"]]).tolist(),
            "contact": [float(ank_l[2] < _CONTACT_ANKLE_Z),
                        float(ank_r[2] < _CONTACT_ANKLE_Z)],
        }
        for k in _SNAP_KEYS:
            out["snap_" + k] = np.asarray(core[k]).tolist()
        self._out = out

    def get_output(self) -> Dict[str, Any]:
        return self._out
