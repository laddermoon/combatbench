"""STEPTVAR V1 仿真器：随机时长步令（STEPTVAR_V1_PLAN）。

继承 V2StepCmdSimulator，唯一核心差异：指令周期从固定
_period_f 改为逐令随机 dur_f ~ U(period_range)，在 _issue
（晋升/首令）时刻采样并存入 sst——**时长不写 obs**：策略
无法得知本令还剩几帧，唯一鲁棒策略=永远尽快到位（ASAP
语义的压力测试）。

短令（1-5f）注定不可达 → 晋升常态化发生在动脚/轴心脚悬空的
中间脚型上；长令（~20f）到位后需驻留。发令时刻记录诊断：
issue_res（初始残差=preview 预摆位效果）、issue_st_z（轴心脚
z，>0.20 ≈ 半空发令）、issue_dyaw，随 _settle 报告上报。

其余全部不变：锚定/残差/preview(P̂)/撞墙/保持令/结算时序。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .v2_simulator import V2Simulator
from .v2_stepcmd_simulator import (
    V2StepCmdSimulator, _ROBOTS, _wrap_pi)


class V2StepTvarSimulator(V2StepCmdSimulator):
    """V2StepCmdSimulator + 逐令随机时长 U(period_range)（obs 不可见）。"""

    def __init__(self, *args: Any,
                 period_range: Tuple[int, int] = (1, 20),
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._dur_lo, self._dur_hi = (int(period_range[0]),
                                      int(period_range[1]))

    # -------------------------------------------------------- 帧结算 -------
    def physical_step(self) -> None:
        """物理子步推进；末子步帧结算，到期判 st["dur_f"]。"""
        # 跳过父类固定周期结算，直接调 V2Simulator 的物理推进
        V2Simulator.physical_step(self)
        if not self._step_on:
            return
        self._phys_sub += 1
        if self._phys_sub < self._spa:
            return
        self._phys_sub = 0
        for rid in _ROBOTS:
            st = self._sst.get(rid)
            if st is None:
                continue
            st["report"] = None
            st["age_f"] += 1
            self._check_arrive(rid)
            if st["age_f"] >= st["dur_f"]:
                self._settle(rid)
                self._rotate(rid)

    # -------------------------------------------------------- 发令 ---------
    def _issue(self, rid: str) -> None:
        """父类锚定后：采本令时长 + 记发令时刻诊断。"""
        super()._issue(rid)
        st = self._sst[rid]
        st["dur_f"] = int(st["rng"].integers(
            self._dur_lo, self._dur_hi + 1))
        # 发令时刻诊断：初始残差 / 轴心脚 z / 初始 yaw 残差
        fl, fr, yaw = self._feet_yaw(rid)
        c = st["cur"]
        kp = self._robot(rid)["keypoint_body_ids"]
        zl = float(self.data.xpos[kp["foot_left"]][2])
        zr = float(self.data.xpos[kp["foot_right"]][2])
        if c.foot < 0:
            st["issue_res"] = float(np.linalg.norm(fl - st["tgt_mv"]))
            st["issue_dyaw"] = float(abs(_wrap_pi(st["yaw_t"] - yaw)))
            st["issue_st_z"] = zr
        elif c.foot > 0:
            st["issue_res"] = float(np.linalg.norm(fr - st["tgt_mv"]))
            st["issue_dyaw"] = float(abs(_wrap_pi(st["yaw_t"] - yaw)))
            st["issue_st_z"] = zl
        else:
            st["issue_res"] = 0.0
            st["issue_dyaw"] = 0.0
            st["issue_st_z"] = max(zl, zr)

    # -------------------------------------------------------- 结算 ---------
    def _settle(self, rid: str) -> None:
        """父类报告 + dur/res0/st_z（eval 分桶与半空诊断）。"""
        super()._settle(rid)
        st = self._sst[rid]
        st["report"].update({
            "dur": float(st["dur_f"]),
            "res0": float(st["issue_res"]),
            "st_z": float(st["issue_st_z"]),
        })


__all__ = ["V2StepTvarSimulator"]
