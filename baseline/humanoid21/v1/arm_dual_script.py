"""双臂独立劫持指令流（ARMDUAL_V1_PLAN §2）：同步同相位。

两臂共享同一周期时钟——每个周期到期统一重采，不存在一臂
中途换令另一臂延续：

    每周期：
      P(dual_prob)  → 双臂同劫（各采独立随机目标）
      否则          → 单臂：P(左)=P(右)=0.5，选中臂采目标
      时长 ~ U(hold_range) 帧（沿用 arm U(2,20)）
      被劫臂 tgt ~ U(tgt_range)³；自由臂 flag=0、tgt=0

时间占比 ≈ dual/仅右/仅左 = 0.1/0.45/0.45；恒有 ≥1 臂在令
（无双臂全释放期）。"软"语义下 flag=1 的臂 err 计分，
flag=0 臂自由（tgt 全零 = "不存在目标"）。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class ArmDualScript:
    """双机双臂独立劫持调度器（每机一实例）。"""

    def __init__(self, rng: np.random.Generator,
                 hold_range: Tuple[int, int] = (2, 20),
                 tgt_range: Tuple[float, float] = (-1.0, 1.0),
                 dual_prob: float = 0.1) -> None:
        self._rng = rng
        self._lo, self._hi = hold_range
        self._t_lo, self._t_hi = tgt_range
        self._dual_p = float(dual_prob)
        self._flags = np.zeros(2, bool)        # [r, l]
        self._cur = np.zeros(6, np.float32)    # tgt_r(3)|tgt_l(3)
        self._rem = 0                          # 当前周期剩余帧

    # -------------------------------------------------- 查询接口 --
    @property
    def current(self) -> np.ndarray:
        """(6,) 当期目标；自由臂段恒零。"""
        return self._cur

    @property
    def flags(self) -> np.ndarray:
        """(2,) bool：各臂本周期是否被劫持。"""
        return self._flags

    # -------------------------------------------------- 推进 ------
    def step(self) -> None:
        """每 action step 调一次；到期统一重采（同步边界）。"""
        self._rem -= 1
        if self._rem <= 0:
            self._draw()

    def _draw(self) -> None:
        self._rem = int(self._rng.integers(self._lo, self._hi + 1))
        self._flags[:] = False
        self._cur[:] = 0.0
        if self._rng.random() < self._dual_p:
            self._flags[:] = True
            self._cur[:] = self._rng.uniform(
                self._t_lo, self._t_hi, 6).astype(np.float32)
        else:
            arm = int(self._rng.integers(2))     # 0=r / 1=l
            self._flags[arm] = True
            self._cur[arm * 3:arm * 3 + 3] = self._rng.uniform(
                self._t_lo, self._t_hi, 3).astype(np.float32)


__all__ = ["ArmDualScript"]
