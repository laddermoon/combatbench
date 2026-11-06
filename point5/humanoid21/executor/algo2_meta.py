"""算法2 信息项 —— alg2_infoflow.md 的代码落地。

每个 I 是一个命名信息项（dataclass），承载载荷 + 来源语义；
功能点 F1-F5 只出现在来源签名里，不在此定义。

    输入: Observation(96,), P5Clip, List[SupportState]
    输出: R (Action 21,)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contracts import Action, SupportState  # noqa: F401  (re-exported types)
from point5.humanoid21.pose import Pose

# ---------- payload shapes ----------
_P5 = (5, 3)      # 头/左右手/左右脚，重力对齐局部系
_JOINTS = 21      # humanoid21 关节数
_COM = (3,)       # 质心位置


def _chk(v: np.ndarray, shape, name: str) -> np.ndarray:
    v = np.asarray(v, np.float64)
    assert v.shape == shape, f"{name}: expected {shape}, got {v.shape}"
    return v


# ---------- P5 空间 ----------

@dataclass
class I1:
    """当前 P5 — 取 clip.Base（观测侧参考起点）。"""
    p5: np.ndarray

    def __post_init__(self):
        self.p5 = _chk(self.p5, _P5, "I1.p5")


@dataclass
class I2:
    """下一锚点的 P5 — F1(plan, clip)。

    经 SupportState.for_frame 从 clip.p5[for_frame] 取；
    接触锚点缺失时由锚点支撑态 + 先验姿态 P5 补全。
    """
    p5: np.ndarray
    completed: bool = False      # True = 由 F1 补全（接触锚点）

    def __post_init__(self):
        self.p5 = _chk(self.p5, _P5, "I2.p5")


@dataclass
class I3:
    """下一帧目标 P5 — F2(I1, I2, I8)。"""
    p5: np.ndarray

    def __post_init__(self):
        self.p5 = _chk(self.p5, _P5, "I3.p5")


# ---------- 姿态空间 ----------

@dataclass
class I4:
    """当前姿态 — Pose（qpos + root_z + g_body + 速度）。"""
    pose: Pose

    def __post_init__(self):
        assert isinstance(self.pose, Pose), "I4.pose must be Pose"


@dataclass
class I5:
    """目标姿态 — Pose（I3 反解的关节空间参考）。"""
    pose: Pose

    def __post_init__(self):
        assert isinstance(self.pose, Pose), "I5.pose must be Pose"


# ---------- 支撑态空间（统一首帧 P5 系） ----------

def _st(v, name):
    assert isinstance(v, SupportState), f"{name} must be SupportState"
    return v


@dataclass
class I6:
    """当前支撑态 — obs 解码（f12）为主，plan[0] 备选。"""
    state: SupportState

    def __post_init__(self):
        self.state = _st(self.state, "I6.state")


@dataclass
class I7:
    """锚点支撑态 — 下一 SupportState。"""
    state: SupportState

    def __post_init__(self):
        self.state = _st(self.state, "I7.state")


@dataclass
class I8:
    """下一帧支撑态目标 — F4(I6, I7, I9)。"""
    state: SupportState

    def __post_init__(self):
        self.state = _st(self.state, "I8.state")


@dataclass
class I9:
    """下下锚点支撑态 — F4 前瞻输入。"""
    state: SupportState

    def __post_init__(self):
        self.state = _st(self.state, "I9.state")


# ---------- 动作空间 ----------

@dataclass
class R:
    """结果动作 — F5(I8, I5)，归一化 PD 目标。"""
    action: Action

    def __post_init__(self):
        self.action = _chk(self.action, (_JOINTS,), "R.action")


__all__ = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "R"]
