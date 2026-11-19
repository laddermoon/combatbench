"""battle_circular_v2 场地 24 边形墙几何（共享常数+判定）。

墙 = 24 边凸多边形：wall_00..23 平面段，内切半径 ~3.41m、外接
~3.44m（取自 XML wall_00 中点坐标）。供调度器 collision_fn、
平面动画、评分判定共用。
"""
from __future__ import annotations

import math

import numpy as np

N_SEG = 24
APOTHEM = float(np.hypot(3.3814, 0.4452))            # 内切圆半径
CIRC = APOTHEM / math.cos(math.pi / N_SEG)           # 外接半径 ≈3.44
_VERT_ANGLES = np.linspace(0, 2 * math.pi, N_SEG, endpoint=False)
VERTS = CIRC * np.stack([np.cos(_VERT_ANGLES),
                         np.sin(_VERT_ANGLES)], axis=1)
# 各边中点外法线（顶点角 + π/24）
_EDGE = _VERT_ANGLES + math.pi / N_SEG
NORMALS = np.stack([np.cos(_EDGE), np.sin(_EDGE)], axis=1)

WALL_MARGIN = 0.30        # 脚点距墙安全边距 m


def inside_arena(p: np.ndarray, margin: float = WALL_MARGIN) -> bool:
    """24 边形凸域判定：p·n_i ≤ apothem−margin ∀i（xy 平面）。"""
    return bool(np.all(NORMALS @ p[:2] <= APOTHEM - margin))


__all__ = [
    "N_SEG", "APOTHEM", "CIRC", "VERTS", "NORMALS",
    "WALL_MARGIN", "inside_arena",
]
