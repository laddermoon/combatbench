"""SupportGeometry / SupportState — 支撑语义类型。

两种粒度，均**帧无关**——与坐标系有关的量可表达于任何
**z=0 为地面、+z 竖直向上的右手系**（leveled 帧族），
所在帧由持有方约定：

    SupportGeometry  静态几何：com + 双脚位/足高 + 躯干朝向
    SupportState     动态支撑态 = 几何 + 动量

来源决定内容：obs 解码态带实测动量；规划态给设计值；
P5 反解无动量——只能构造 SupportGeometry。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

_COM = (3,)
_FOOT = (2,)
_VEL = (3,)
_MOM = (3,)

# 平脚贴地时的基准踝高（模型形态常数，stand_punch 断言 0.087±0.01）
ANKLE_Z_PLANTED = 0.087


def _wrap(a: float) -> float:
    """角度归一到 (-pi, pi]。"""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _vec(v, shape, name):
    v = np.array(v, np.float64)          # copy, never alias
    assert v.shape == shape, f"{name}: expected {shape}, got {v.shape}"
    return v


@dataclass
class SupportGeometry:
    """静态几何锚点 — leveled 帧族，所在帧由持有方约定
    （SupportPlan 统一首帧 P5 系）。足高记发力潜力——贴地才有
    推力权威，不记实测分力。

    foot_height 定义 = 脚踝实际高度 − 平脚贴地时的基准踝高，
    ≈0 表示落地。
    """
    com: np.ndarray                    # (3,) 质心
    left_foot: np.ndarray              # (2,) 脚踝 xy
    right_foot: np.ndarray             # (2,) 脚踝 xy
    torso_yaw: float                   # 躯干朝向 rad
    left_foot_height: float = 0.0      # 左脚踝抬升高度（≈0 落地）
    right_foot_height: float = 0.0     # 右脚踝抬升高度

    def __post_init__(self):
        self.com = _vec(self.com, _COM, "geometry.com")
        self.left_foot = _vec(self.left_foot, _FOOT,
                              "geometry.left_foot")
        self.right_foot = _vec(self.right_foot, _FOOT,
                               "geometry.right_foot")
        self.torso_yaw = float(self.torso_yaw)
        self.left_foot_height = float(self.left_foot_height)
        self.right_foot_height = float(self.right_foot_height)

    @classmethod
    def from_pose(cls, pose) -> "SupportGeometry":
        """Pose -> 几何锚点（薄委托：提取实现住 Pose.to_support_geometry）。"""
        return pose.to_support_geometry()

    def _se2(self, dyaw: float, dxy) -> "SupportGeometry":
        """SE(2) 平面变换：点乘 Rz+平移，yaw += dyaw，足高不变。"""
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R2 = np.array([[c, -s], [s, c]])
        return SupportGeometry(
            com=R @ self.com + np.array([dxy[0], dxy[1], 0.0]),
            left_foot=R2 @ self.left_foot + np.asarray(dxy, float),
            right_foot=R2 @ self.right_foot + np.asarray(dxy, float),
            torso_yaw=_wrap(self.torso_yaw + dyaw),
            left_foot_height=self.left_foot_height,
            right_foot_height=self.right_foot_height)

    def standardize(self) -> "SupportGeometry":
        """转成自身 P5 系（踝中点→原点、torso_yaw→0）。"""
        mid = (self.left_foot + self.right_foot) / 2.0
        yaw = float(self.torso_yaw)
        c, s = np.cos(-yaw), np.sin(-yaw)
        t = -np.array([[c, -s], [s, c]]) @ mid
        return self._se2(-yaw, t)

    def reanchor(self, xy=(0.0, 0.0),
                 yaw: float = 0.0) -> "SupportGeometry":
        """重新锚定到指定位置/朝向（standardize 的逆操作）。

        输出保证踝中点在 xy、躯干朝向 yaw，几何形状不变——
        内部先 standardize 再做 SE(2) 平面变换。
        """
        return self.standardize()._se2(float(yaw), np.asarray(xy, float))

    def com_acc_ankle(self) -> np.ndarray:
        """此几何下脚踝力矩可产生的质心加速度（3,)。"""
        raise NotImplementedError

    def com_acc_gravity(self) -> np.ndarray:
        """重力引起的质心加速度分量（3,)。"""
        raise NotImplementedError

    def reference_p5(self):
        """由几何合成参考 P5 意图——非精确还原。

        几何只含 com/脚/yaw，头与手按标称站姿生成；返回的是
        可用参照点阵，不做信息保真承诺（故名 reference 非 to_）。
        """
        raise NotImplementedError


@dataclass
class SupportState(SupportGeometry):
    """动态支撑态 = 几何 + 动量 — 帧无关。

    com_vel/ang_mom 必带——动量缺席的场合用 SupportGeometry，
    不构造残缺的 SupportState。
    """
    com_vel: np.ndarray = field(kw_only=True)  # (3,) 质心速度
    ang_mom: np.ndarray = field(kw_only=True)  # (3,) 全身角动量，关于质心——换帧平移不变，仅随 yaw 旋转分量

    def __post_init__(self):
        super().__post_init__()
        self.com_vel = _vec(self.com_vel, _VEL, "com_vel")
        self.ang_mom = _vec(self.ang_mom, _MOM, "ang_mom")

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "SupportState":
        """obs (96,) -> SupportState，产 **P5 坐标系**的态。

        RobotState.from_obs(obs) -> from_pose 两段式——
        机体系输入解码时转 P5 帧（同 RobotState 约定）。
        """
        raise NotImplementedError

    @classmethod
    def from_robot_state(cls, robot_state) -> "SupportState":
        """RobotState -> SupportState（薄委托：提取实现住
        RobotState.to_support_state）。"""
        return robot_state.to_support_state()

    def _se2(self, dyaw: float, dxy) -> "SupportState":
        """SE(2) 变换：几何走父类，com_vel/ang_mom 乘 Rz(dyaw)。"""
        g = super()._se2(dyaw, dxy)
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return SupportState(
            com=g.com, left_foot=g.left_foot, right_foot=g.right_foot,
            torso_yaw=g.torso_yaw,
            left_foot_height=g.left_foot_height,
            right_foot_height=g.right_foot_height,
            com_vel=R @ self.com_vel, ang_mom=R @ self.ang_mom)

    def standardize(self) -> "SupportState":
        """几何归一 + 动量随帧旋转——保持动态字段。"""
        return super().standardize()

    def reanchor(self, xy=(0.0, 0.0),
                 yaw: float = 0.0) -> "SupportState":
        """同 standardize——帧变换时 com_vel/ang_mom 随 yaw 旋转。"""
        return super().reanchor(xy, yaw)

    def geometry(self) -> SupportGeometry:
        """获取静态几何锚点（剥掉动量）。"""
        return SupportGeometry(
            com=self.com.copy(),
            left_foot=self.left_foot.copy(),
            right_foot=self.right_foot.copy(),
            left_foot_height=self.left_foot_height,
            right_foot_height=self.right_foot_height,
            torso_yaw=self.torso_yaw)

    def capture_point(self) -> np.ndarray:
        """捕获点（3,) — com + com_vel·√(z/g)。"""
        raise NotImplementedError

    def transition_to(self, anchor: SupportGeometry
                      ) -> Tuple[SupportGeometry, "SupportState"]:
        """向目标锚点迈一步。

        返回 (next_geometry, next_state)：转移语义待 F4 落地。
        """
        raise NotImplementedError
