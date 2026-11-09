"""Point5 意图协议（v3: 具名 landmark + 标准化坐标系）。

This package must not import mujoco / envs.* / baseline.*.
"""
from .p5 import P5, P5Clip, POINT_INDEX, POINT_NAMES

__all__ = ["P5", "P5Clip", "POINT_INDEX", "POINT_NAMES"]
