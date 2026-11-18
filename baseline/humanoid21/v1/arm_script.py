"""随机手臂脚本调度器（V2-S3 扰动源）。

每臂独立倒计时：倒计时到点 → 重新采样 3 关节目标（norm 空间均匀
分布，无语义动作）+ 新保持时长 U(hold_range)。两臂/双机各自独立
rng，避免相位同步。

用法：
    sch = ArmScript(rng, hold_range=(2, 20))
    tgt6 = sch.step()        # 每个 action step 调一次 → (6,) norm 目标
    tgt6 = sch.current       # 当前目标（不推进）
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class ArmScript:
    """双臂随机目标调度器。norm 空间（=action 空间）输出。"""

    def __init__(self, rng: np.random.Generator,
                 hold_range: Tuple[int, int] = (2, 20),
                 tgt_range: Tuple[float, float] = (-1.0, 1.0)) -> None:
        self._rng = rng
        self._hold_range = (int(hold_range[0]), int(hold_range[1]))
        self._tgt_range = (float(tgt_range[0]), float(tgt_range[1]))
        # 每臂: [s1, s2, e] 目标 + 剩余保持步数（0=下步即重采样）
        self._tgt = np.zeros((2, 3), np.float32)
        self._hold = np.zeros(2, np.int64)
        self.current = self._tgt.reshape(6)

    def _resample(self, arm: int) -> None:
        lo, hi = self._tgt_range
        self._tgt[arm] = self._rng.uniform(lo, hi, 3).astype(np.float32)
        h_lo, h_hi = self._hold_range
        self._hold[arm] = int(self._rng.integers(h_lo, h_hi + 1))

    def step(self) -> np.ndarray:
        """推进一个控制步并返回当前双臂目标 (6,) = [右3|左3]。"""
        for arm in (0, 1):
            if self._hold[arm] <= 0:
                self._resample(arm)
            self._hold[arm] -= 1
        self.current = self._tgt.reshape(6)
        return self.current


__all__ = ["ArmScript"]
