"""P5 / P5Clip — 意图协议。

P5 双身份：
  1. 数据协议定义——5 个具名 landmark + 躯干朝向
  2. 标准化坐标系——standardize() 归一到标准 P5 帧
     （踝中点原点、躯干 +x、z 向上、yaw≡0）

P5Clip 序列：所有 P5 共用同一个坐标系（坐标统一），
帧语义由持有方约定。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_PT = (3,)


def _vec(v, name):
    v = np.array(v, np.float64)          # copy, never alias
    assert v.shape == _PT, f"{name}: expected {_PT}, got {v.shape}"
    return v


@dataclass(frozen=True)
class P5:
    """P5 姿态 — 5 landmark 点 + 躯干朝向（帧由持有方约定）。

    torso_yaw: 躯干朝向与所在帧 +X 轴夹角 rad（±π）；
    标准 P5 帧里 yaw≡0。
    """
    head: np.ndarray                   # (3,)
    left_hand: np.ndarray              # (3,)
    right_hand: np.ndarray             # (3,)
    left_ankle: np.ndarray             # (3,)
    right_ankle: np.ndarray            # (3,)
    torso_yaw: float                   # rad

    def __post_init__(self):
        for n in ("head", "left_hand", "right_hand",
                  "left_ankle", "right_ankle"):
            object.__setattr__(self, n, _vec(getattr(self, n), n))
        object.__setattr__(self, "torso_yaw", float(self.torso_yaw))

    @property
    def pos(self) -> np.ndarray:
        """(5,3) 点阵 — POINT_NAMES 序兼容口。"""
        return np.array([self.head, self.right_hand, self.left_hand,
                         self.right_ankle, self.left_ankle])

    def standardize(self) -> "P5":
        """归一到标准 P5 帧（踝中点原点、躯干 +x、yaw→0）。"""
        raise NotImplementedError

    def to_image(self, path: Optional[str] = None):
        """渲染参考图——给 path 存 PNG，不给返回 PIL Image。"""
        raise NotImplementedError


@dataclass(frozen=True)
class P5Clip:
    """P5 序列 — 全部帧共用同一坐标系。可索引/可迭代。"""
    frames: Tuple[P5, ...]

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, i):
        return self.frames[i]

    def to_video(self, path: str):
        """序列渲染为 MP4。"""
        raise NotImplementedError
