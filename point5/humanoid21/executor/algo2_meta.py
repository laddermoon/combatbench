"""算法2 信息项 —— alg2_infoflow.md 的代码落地。

每个 I 是一个命名信息项（dataclass），承载载荷 + 来源语义；
功能点 F1-F5 只出现在来源签名里，不在此定义。

    输入: Observation(96,), P5Clip, List[LandMark]
    输出: R (Action 21,)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contracts import Action, LandMark  # noqa: F401  (re-exported types)

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
    """当前 P5 — Observation 解码（clip kf0 同一份）。"""
    p5: np.ndarray

    def __post_init__(self):
        self.p5 = _chk(self.p5, _P5, "I1.p5")


@dataclass
class I2:
    """下一 landmark 的 P5 — F1(landmarks, clip)。

    姿态锚点：经 lm.frame 从 clip 关键帧直接取；
    接触锚点：缺失，由锚点质心态 + 先验/后续姿态 P5 补全。
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
    """当前姿态 — obs[0:21] 关节位置。"""
    pose: np.ndarray

    def __post_init__(self):
        self.pose = _chk(self.pose, (_JOINTS,), "I4.pose")


@dataclass
class I5:
    """目标姿态 — F3(I3, I4, I8?)。"""
    pose: np.ndarray

    def __post_init__(self):
        self.pose = _chk(self.pose, (_JOINTS,), "I5.pose")


# ---------- 质心空间（位置 only） ----------

@dataclass
class I6:
    """当前质心位置 — obs 解码（工具1/8）。"""
    com: np.ndarray

    def __post_init__(self):
        self.com = _chk(self.com, _COM, "I6.com")


@dataclass
class I7:
    """锚点质心位置 — 下一 landmark 的 com。"""
    com: np.ndarray

    def __post_init__(self):
        self.com = _chk(self.com, _COM, "I7.com")


@dataclass
class I8:
    """下一帧质心目标 — F4(I6, I7, I9)。"""
    com: np.ndarray

    def __post_init__(self):
        self.com = _chk(self.com, _COM, "I8.com")


@dataclass
class I9:
    """下下个 landmark 的质心位置 — F4 前瞻输入。"""
    com: np.ndarray

    def __post_init__(self):
        self.com = _chk(self.com, _COM, "I9.com")


# ---------- 动作空间 ----------

@dataclass
class R:
    """结果动作 — F5(I8, I5)，归一化 PD 目标。"""
    action: Action

    def __post_init__(self):
        self.action = _chk(self.action, (_JOINTS,), "R.action")


__all__ = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "R"]
