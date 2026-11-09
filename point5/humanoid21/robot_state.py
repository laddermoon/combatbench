"""Pose / RobotState — 机器人姿态的静态/动态两级表达。

    Pose        静态姿态：qpos + root_z + g_body（无位置——
                水平位置与 yaw 由 anchor() 解析）
    RobotState  动态快照 = Pose + 速度 + 实测接触力

与 SupportGeometry/SupportState 同构：静态层管几何，动态层
叠加测量。obs 是本质动态快照，只经 RobotState.from_obs 进。

    a = pose.anchor('left', xy=(0.3, 0.0), yaw=np.pi/4)
    a.p5()  a.foot_pos()  a.com()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_JOINTS = (21,)
_G = (3,)
_VEL = (3,)
_FORCE = (2,)


def _vec(v, shape, name):
    v = np.array(v, np.float64)          # copy, never alias
    assert v.shape == shape, f"{name}: expected {shape}, got {v.shape}"
    return v


@dataclass
class AnchoredPose:
    """Pose 解析进坐标系后的视图——所有帧内输出住这里。"""
    _p5: np.ndarray                  # (5,3) anchored local
    _feet: dict                      # side -> (2,) anchored local
    _com: np.ndarray                 # (3,) anchored local
    R: np.ndarray = field(           # world -> anchored rotation
        default_factory=lambda: np.eye(3))

    def p5(self):
        """锚定局部帧 P5。"""
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
    robot_id: str = "robot_a"

    def __post_init__(self):
        self.qpos = _vec(self.qpos, _JOINTS, "qpos")
        self.g_body = _vec(self.g_body, _G, "g_body")
        self.root_z = float(self.root_z)

    # ---------- constructors ----------

    @classmethod
    def from_p5(cls, p5, robot_id: str = "robot_a",
                tolerance: float = 0.02) -> "Pose":
        """P5 局部姿态 -> Pose（solve_equilibrium 反解，近似）。"""
        raise NotImplementedError

    @classmethod
    def joint_ranges(cls, robot_id: str = "robot_a") -> np.ndarray:
        """(21,2) 关节弧度范围（obs 反归一化用）。"""
        raise NotImplementedError

    # ---------- unanchored ----------

    def p5(self):
        """自身局部帧 P5（踝中点原点、躯干 +x）。"""
        return self.anchor('ankle_mid').p5()

    def com_height(self) -> float:
        """精确质量加权质心高度——绝对量，无需锚定。"""
        raise NotImplementedError

    def tilt_angles(self) -> np.ndarray:
        """(2,) [roll, pitch] rad — 由 g_body 分解。"""
        raise NotImplementedError

    def support_geometry(self):
        """-> SupportGeometry（静态支撑几何：com + 脚位 + yaw）。"""
        raise NotImplementedError

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
    """动态快照 = 姿态 + 速度 + 实测接触力。

    qvel/root_linvel/root_angvel/foot_force_ratio 均可缺省——
    from_p5 的规划态没有动态语义；from_obs 态必带。
    """
    qvel: Optional[np.ndarray] = None            # (21,) rad/s
    root_linvel: Optional[np.ndarray] = None     # (3,) 机体系线速度
    root_angvel: Optional[np.ndarray] = None     # (3,) 机体系角速度
    foot_force_ratio: Optional[np.ndarray] = None  # (2,) 右/左脚实测力度比

    def __post_init__(self):
        super().__post_init__()
        for n, shp in (("qvel", _JOINTS), ("root_linvel", _VEL),
                       ("root_angvel", _VEL),
                       ("foot_force_ratio", _FORCE)):
            v = getattr(self, n)
            if v is not None:
                setattr(self, n, _vec(v, shp, n))

    @classmethod
    def from_obs(cls, obs: np.ndarray,
                 robot_id: str = "robot_a") -> "RobotState":
        """obs (96,) -> RobotState：关节/速度/接触力无损快照。"""
        raise NotImplementedError

    def to_pose(self) -> Pose:
        """剥动态 → Pose（对应 SupportState.to_geometry）。"""
        return Pose(qpos=self.qpos.copy(), root_z=self.root_z,
                    g_body=self.g_body.copy(), robot_id=self.robot_id)

    def support_state(self):
        """-> SupportState（几何 + 足高 + 动量，obs 链无损）。"""
        raise NotImplementedError

    def com_vel(self, **anchor_kw) -> np.ndarray:
        """子树质心速度 (3,) — 锚定帧，与 com() 同系。"""
        raise NotImplementedError

    def ang_mom(self, **anchor_kw) -> np.ndarray:
        """子树角动量 (3,) — 锚定帧。"""
        raise NotImplementedError
