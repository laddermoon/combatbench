"""算法1 信息项 —— alg1_infoflow.md 的代码落地。

    输入: Observation(96,), P5Clip
    输出: R = 支撑状态序列（List[SupportState]，≥3，首项=当前态）

支撑状态（SupportState）：两脚 XY 位置 + 支撑力度分配（体重归一）
+ 质心 (3,)；不含质心速度/角速度、不带时间。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .contracts import SupportState  # noqa: F401  (re-exported types)

_COM = (3,)


def _chk(v: np.ndarray, shape, name: str) -> np.ndarray:
    v = np.asarray(v, np.float64)
    assert v.shape == shape, f"{name}: expected {shape}, got {v.shape}"
    return v


@dataclass
class I1:
    """当前支撑状态 — F1(Observation)。

    当前两脚位置、支撑分配、质心——形态上就是当前时刻的 SupportState。
    """
    lm: SupportState


@dataclass
class I2:
    """当前质心动量 — F2(Observation)。

    com_vel (3,) + ang_mom (3,)：动量项，决定支撑转移的可行方向
    （离地锚点要求 capture point 指向落脚区）。
    """
    com_vel: np.ndarray
    ang_mom: np.ndarray

    def __post_init__(self):
        self.com_vel = _chk(self.com_vel, _COM, "I2.com_vel")
        self.ang_mom = _chk(self.ang_mom, _COM, "I2.ang_mom")


@dataclass
class I3:
    """Clip 逐帧目标支撑状态 — F3(Clip)。

    逐帧变换：每帧 P5 → 对应支撑状态（脚位取 P5 踝点、
    质心经工具1 推荐并投影支撑域）。与 clip 帧一一对齐。
    """
    targets: List[SupportState]        # len = clip 帧数


@dataclass
class R:
    """支撑状态序列 — F4(I1, I2, I3)。

    锚点 = 支撑事件锚点 + 姿态锚点；不带时间、允许重复；
    长度 ≥3（clip 无换脚需求时以维持平衡的锚点补齐）。
    """
    landmarks: List[SupportState]

    def __post_init__(self):
        assert len(self.landmarks) >= 3, "R: sequence must have >=3 anchors"


__all__ = ["I1", "I2", "I3", "R"]
