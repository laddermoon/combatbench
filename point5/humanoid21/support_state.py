"""SupportGeometry / SupportState — 支撑语义类型。

两种粒度，均**帧无关**（不规定坐标系，任何坐标系下都合法）：

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


def _vec(v, shape, name):
    v = np.array(v, np.float64)          # copy, never alias
    assert v.shape == shape, f"{name}: expected {shape}, got {v.shape}"
    return v


@dataclass
class SupportGeometry:
    """静态几何锚点 — 帧无关。

    com/双脚位/朝向为绝对量（在哪个系由持有方约定——
    SupportPlan 统一首帧 P5 系）。足高记发力潜力——贴地才有
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
        """Pose -> 几何锚点。

        P5 侧走 Pose.infer_from_p5 -> from_pose 两段式——不在此
        重复，避免遮蔽 Geometry 与 Pose 的依赖关系。
        """
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
    ang_mom: np.ndarray = field(kw_only=True)  # (3,) 角动量

    def __post_init__(self):
        super().__post_init__()
        self.com_vel = _vec(self.com_vel, _VEL, "com_vel")
        self.ang_mom = _vec(self.ang_mom, _MOM, "ang_mom")

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "SupportState":
        """obs (96,) -> SupportState：Pose.from_obs 转换链（无损）。"""
        raise NotImplementedError

    @classmethod
    def from_pose(cls, pose) -> "SupportState":
        """Pose -> SupportState：Pose 自带的支撑态输出。"""
        raise NotImplementedError

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

    def com_acc_ankle(self) -> np.ndarray:
        """此状态下脚踝力矩可产生的质心加速度（3,)。"""
        raise NotImplementedError

    def com_acc_gravity(self) -> np.ndarray:
        """重力引起的质心加速度分量（3,)。"""
        raise NotImplementedError

    def transition_to(self, anchor: SupportGeometry
                      ) -> Tuple[SupportGeometry, "SupportState"]:
        """向目标锚点迈一步。

        返回 (next_geometry, next_state)：转移语义待 F4 落地。
        """
        raise NotImplementedError
