"""Algorithm 2 — f1-f5 链（alg2_infoflow.md / algo2_meta.py）。

    obs/landmarks ──► I1, I4, I6, I7
    f1(landmarks, clip) -> I2
    f4(i6, i7)          -> I8
    f2(i1, i2, i8)      -> I3
    f3(i3)              -> I5   （工具2 pose_solver）
    f5(i8, i5)          -> R    （求解器本体）

旧实现（landmark 走查 CEM）已清空，待 f1-f5 落地后重接。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .algo2_functions import f1, f2, f3, f4, f5
from .algo2_meta import I1, I4, I6, I7
from .contracts import Context, PlanError


class ActionComputer:
    """Algorithm 2 — stateless per-call f1-f5 chain."""

    def __init__(self, robot_id: str = "robot_a"):
        self.robot_id = robot_id
        self._decoder = None

    def _dec(self):
        if self._decoder is None:
            from .centroidal_planner import ObsDecoder
            self._decoder = ObsDecoder(self.robot_id)
        return self._decoder

    def compute(self, ctx: Context, landmarks: List, wsim) -> np.ndarray:
        """LandMark seq + obs -> next (21,) normalized action."""
        obs = np.asarray(ctx.observation, np.float64)
        if not landmarks or len(landmarks) < 2:
            raise PlanError("algorithm 2 needs >=2 landmarks")

        i1 = I1(self._dec().decode_p5_local(obs)[0])
        i4 = I4(obs[0:21].copy())
        i6 = I6(np.asarray(landmarks[0].com, np.float64))
        i7 = I7(np.asarray(landmarks[1].com, np.float64))

        i2 = f1(landmarks, ctx.clip)
        i8 = f4(i6, i7)
        i3 = f2(i1, i2, i8, i6, i7)
        i5 = f3(i3, i4, i8)
        return f5(i8, i5).action.astype(np.float32)
