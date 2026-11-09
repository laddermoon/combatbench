"""Pose / RobotState — 机器人姿态的静态/动态两级表达。

    Pose        静态姿态：qpos + root_z + g_body（无位置——
                水平位置与 yaw 由 anchor() 解析）
    RobotState  动态快照 = Pose + 速度（全量必带）

与 SupportGeometry/SupportState 同构：静态层管几何，动态层
叠加测量。obs 是本质动态快照，只经 RobotState.from_obs 进。

    a = pose.anchor('left', xy=(0.3, 0.0), yaw=np.pi/4)
    a.p5()  a.foot_pos()  a.com()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_JOINTS = (21,)
_G = (3,)
_VEL = (3,)


def _vec(v, shape, name):
    v = np.array(v, np.float64)          # copy, never alias
    assert v.shape == shape, f"{name}: expected {shape}, got {v.shape}"
    return v


def tilt_angles(g_body: np.ndarray) -> np.ndarray:
    """(2,) [roll, pitch] rad — g_body 分解出的躯干倾斜角。"""
    g = np.asarray(g_body, np.float64)
    g = g / (np.linalg.norm(g) + 1e-12)
    return np.array([np.arctan2(-g[1], -g[2]),
                     np.arctan2(g[0], -g[2])])


def joint_ranges() -> np.ndarray:
    """(21,2) 关节弧度范围 — 机器人形态元数据（懒加载）。"""
    raise NotImplementedError


@dataclass
class AnchoredPose:
    """Pose 解析进坐标系后的视图——所有帧内输出住这里。"""
    _p5: np.ndarray                  # (5,3) anchored local
    _feet: dict                      # side -> (2,) anchored local
    _com: np.ndarray                 # (3,) anchored local
    R: np.ndarray = field(           # world -> anchored rotation
        default_factory=lambda: np.eye(3))

    def to_p5(self):
        """锚定局部帧 -> P5。"""
        raise NotImplementedError

    def foot_pos(self) -> dict:
        return self._feet

    def com(self) -> np.ndarray:
        return self._com


@dataclass
class Pose:
    """关节角 + 高度 + 俯仰——机器人姿态的静态表达。"""
    qpos: np.ndarray                 # (21,) rad
    root_z: float                    # 世界系高度
    g_body: np.ndarray               # (3,) 机体系重力向量

    def __post_init__(self):
        self.qpos = _vec(self.qpos, _JOINTS, "qpos")
        self.g_body = _vec(self.g_body, _G, "g_body")
        self.root_z = float(self.root_z)

    # ---------- constructors ----------

    @classmethod
    def infer_from_p5(cls, p5, tolerance: float = 0.02) -> "Pose":
        """P5 局部姿态 -> Pose（solve_equilibrium 反解关节角）。

        infer_* = 近似推断（源里没有 qpos/root_z，靠 solver 重建），
        区别于 from_* 的无损解码。
        """
        raise NotImplementedError

    # ---------- unanchored ----------

    def to_p5(self):
        """自身局部帧 -> P5（踝中点原点、躯干 +x）。"""
        return self.anchor('ankle_mid').to_p5()

    def to_support_geometry(self) -> "SupportGeometry":
        """-> SupportGeometry：委托 SupportGeometry.from_pose(self)。

        实现住 target 层（X.from_y 惯例），此方法是源侧语法糖——
        同 Pose.p5() 先例。
        """
        from .support_state import SupportGeometry
        return SupportGeometry.from_pose(self)

    # ---------- anchored outputs ----------

    def anchor(self, point: str = "ankle_mid",
               xy=(0.0, 0.0), yaw: float = 0.0) -> AnchoredPose:
        """把无位置姿态解析进坐标系。

        point: 'ankle_mid' | 'left' | 'right' | 'com'
        """
        raise NotImplementedError

    def render(self, out_path, **kw) -> list:
        """渲染为 PNG（需 OpenGL platform）。"""
        raise NotImplementedError


@dataclass
class RobotState(Pose):
    """动态快照 = 姿态 + 速度——全部必带，from_obs 唯一入口。

    无动态的场合用 Pose（from_p5），不构造残缺的 RobotState。
    """
    qvel: np.ndarray                           # (21,) rad/s
    root_linvel: np.ndarray                    # (3,) 机体系线速度
    root_angvel: np.ndarray                    # (3,) 机体系角速度

    def __post_init__(self):
        super().__post_init__()
        self.qvel = _vec(self.qvel, _JOINTS, "qvel")
        self.root_linvel = _vec(self.root_linvel, _VEL, "root_linvel")
        self.root_angvel = _vec(self.root_angvel, _VEL, "root_angvel")

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "RobotState":
        """obs (96,) -> RobotState：关节/速度无损快照。

        生成态位于 **P5 坐标系**：无绝对位置、无绝对 yaw——
        经 anchor()/to_p5() 解析出踝中点原点、躯干 +x 的
        标准帧量；g_body/linvel/angvel 为机体系分量。
        """
        raise NotImplementedError

    def pose(self) -> Pose:
        """剥动态 → Pose（对应 SupportState.geometry）。"""
        return Pose(qpos=self.qpos.copy(), root_z=self.root_z,
                    g_body=self.g_body.copy())

    def to_support_state(self):
        """-> SupportState：委托 SupportState.from_pose(self)。

        实现住 target 层（X.from_y 惯例），此方法是源侧语法糖——
        同 Pose.to_p5() 先例。
        """
        from .support_state import SupportState
        return SupportState.from_pose(self)
