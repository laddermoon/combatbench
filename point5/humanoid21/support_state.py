"""SupportGeometry / SupportState — 支撑语义类型。

两种粒度，均**帧无关**（不规定坐标系，任何坐标系下都合法）：

    SupportGeometry  静态几何：com + 双脚位 + 躯干朝向
    SupportState     动态支撑态 = 几何 + 力度分配 + 动量

来源决定内容：obs 解码来的带实测力/动量；规划来的是设计值；
P5 反解来的没有实测力（给规划默认分配）也没有动量。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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

    com/双脚位为绝对坐标（在哪个系由持有方约定——
    SupportPlan 统一首帧 P5 系）。torso_yaw v1 不处理。
    """
    com: np.ndarray                    # (3,)
    left_foot: np.ndarray              # (2,) xy
    right_foot: np.ndarray             # (2,) xy
    torso_yaw: Optional[float] = None

    def __post_init__(self):
        self.com = _vec(self.com, _COM, "geometry.com")
        self.left_foot = _vec(self.left_foot, _FOOT,
                              "geometry.left_foot")
        self.right_foot = _vec(self.right_foot, _FOOT,
                               "geometry.right_foot")

    @classmethod
    def from_p5(cls, p5, tolerance: float = 0.02) -> "SupportGeometry":
        """P5 局部姿态 -> 几何锚点（经 Pose 反解，近似）。"""
        raise NotImplementedError

    def to_p5(self):
        """生成参考 P5 姿态（几何 -> 意图点阵）。"""
        raise NotImplementedError


@dataclass
class SupportState(SupportGeometry):
    """动态支撑态 = 几何 + 力度分配 + 动量 — 帧无关。

    left/right_foot_support_ratio: 左/右脚承重比例（[0,1]，
    实测或设计值）。com_vel/ang_mom 可缺省——from_p5 的规划
    态没有动量语义；obs 解码态必带。
    """
    left_foot_support_ratio: float = 0.0
    right_foot_support_ratio: float = 0.0
    com_vel: Optional[np.ndarray] = None     # (3,) 质心速度
    ang_mom: Optional[np.ndarray] = None     # (3,) 角动量
    for_frame: int = 0                       # clip 帧索引（元数据）

    def __post_init__(self):
        super().__post_init__()
        self.left_foot_support_ratio = float(
            self.left_foot_support_ratio)
        self.right_foot_support_ratio = float(
            self.right_foot_support_ratio)
        for n, shp in (("com_vel", _VEL), ("ang_mom", _MOM)):
            v = getattr(self, n)
            if v is not None:
                setattr(self, n, _vec(v, shp, n))

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "SupportState":
        """obs (96,) -> SupportState：Pose.from_obs 转换链（无损）。"""
        raise NotImplementedError

    @classmethod
    def from_pose(cls, pose) -> "SupportState":
        """Pose -> SupportState：Pose 自带的支撑态输出。"""
        raise NotImplementedError

    def to_geometry(self) -> SupportGeometry:
        """获取静态几何锚点（剥掉力度/动量）。"""
        return SupportGeometry(com=self.com.copy(),
                               left_foot=self.left_foot.copy(),
                               right_foot=self.right_foot.copy(),
                               torso_yaw=self.torso_yaw)

    def capture_point(self) -> np.ndarray:
        """捕获点（3,) — 需 com_vel，缺席则拒（同 Pose.com_vel 规则）。"""
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
