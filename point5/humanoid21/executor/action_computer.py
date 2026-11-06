"""Algorithm 2 — f1-f5 链（alg2_infoflow.md / algo2_meta.py）。

    obs/landmarks ──► I1, I4, I6, I7
    f1(landmarks, clip) -> I2
    f4(i6, i7)          -> I8
    f2(i1, i2, i8)      -> I3
    f3(i3)              -> I5   （solve_equilibrium）
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
            from point5.humanoid21.pose import Pose
            self._decoder = Pose
        return self._decoder

    def compute(self, ctx: Context, landmarks: List, wsim) -> np.ndarray:
        """SupportState seq + obs -> next (21,) normalized action."""
        from point5.humanoid21.pose import Pose
        from .algo1_functions import f12

        obs = np.asarray(ctx.observation, np.float64)
        if not landmarks or len(landmarks) < 2:
            raise PlanError("algorithm 2 needs >=2 landmarks")

        pose = Pose.from_obs(obs)
        i1v, _ = f12(obs)
        i1 = I1(pose.anchor('ankle_mid').p5().pos)
        i4 = I4(pose)                       # Pose（关节+高度+倾斜+速度）
        i6 = I6(i1v.lm)                     # 当前支撑态（f12 解码）
        i7 = I7(landmarks[1])               # 下一锚点支撑态

        i2 = f1(landmarks, ctx.clip)
        i8 = f4(landmarks)
        i3 = f2(i1, i2, i8, i6, i7)
        i5 = f3(i3, i4, i8)
        return f5(i8, i5).action.astype(np.float32)
