"""PoseAnchor — 精确的无位置机器人姿态表达（锚点用）。

    qpos   (21,)  关节角，弧度（真实单位，非归一化）
    root_z (1,)   躯干高度，世界系 z
    g_body (3,)   机体系重力向量 — 编码俯仰/侧倾，不含 yaw

与 obs 模块一/二精确对应：obs[0:21] 归一化关节角 ×range，
obs[42:45] 重力向量，obs[45] 高度。yaw 与水平位置不在
此表达——锚点的位置语义由脚位/支撑态给。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_JOINTS = (21,)
_G = (3,)


@dataclass
class PoseAnchor:
    """关节角 + 高度 + 俯仰——机器人姿态的精确表达。"""
    qpos: np.ndarray               # (21,) rad
    root_z: float                  # 世界系高度
    g_body: np.ndarray             # (3,) 机体系重力向量

    def __post_init__(self):
        self.qpos = np.asarray(self.qpos, np.float64)
        assert self.qpos.shape == _JOINTS, \
            f"qpos expected {_JOINTS}, got {self.qpos.shape}"
        self.g_body = np.asarray(self.g_body, np.float64)
        assert self.g_body.shape == _G, \
            f"g_body expected {_G}, got {self.g_body.shape}"
        self.root_z = float(self.root_z)

    @classmethod
    def from_obs(cls, obs: np.ndarray, joint_ranges: np.ndarray):
        """obs (96,) -> PoseAnchor（关节角反归一化）。

        joint_ranges: (21,2) 各关节 (lo,hi) 弧度范围
        （Humanoid21FK.joint_ranges）。
        """
        rng = np.asarray(joint_ranges, np.float64)
        ref, sc = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2
        return cls(qpos=np.asarray(obs[0:21], np.float64) * sc + ref,
                   root_z=float(obs[45]),
                   g_body=np.asarray(obs[42:45], np.float64))

    def tilt_angles(self) -> np.ndarray:
        """(2,) [roll, pitch] rad — 由 g_body 分解的倾斜角。"""
        gx, gy, gz = self.g_body / (np.linalg.norm(self.g_body) + 1e-12)
        return np.array([np.arctan2(-gy, -gz), np.arctan2(gx, -gz)])
