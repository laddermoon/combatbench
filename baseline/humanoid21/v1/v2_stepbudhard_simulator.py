"""STEPBUDHARD V1 联合仿真器：方向+预算步令 × 双臂独立硬劫持。

继承 V2StepBudgetSimulator（方向+预算步令流，STEPBUDGET_V1_PLAN），
叠加 ARMDUAL 逐臂独立**硬劫持**：被劫臂 act 维度在物理下发前
覆写为指令 tgt（ARM_TAKEOVER 执行层语义）。父类臂机制全程关闭
（arm_script=False → V2Simulator 的 flag/覆写/旧 ArmScript 休眠）。

指令流（与软劫持版同分布）：两臂共享同步周期时钟，
P(0.1) 双臂同劫 / P(0.9) 单臂（左右 50/50），被劫臂
tgt~U(-1,1)³，周期 dur~U(2,20)f。

obs 指令段（与 steparm/stepbudarm 同布局）：
    seg[0:8]   = [flag_r,tgt_r(3),flag_l,tgt_l(3)] → obs[57:65]
    seg[8:10]  = waist 预留零                      → obs[65:67]
    seg[10:22] = step_cur|step_next 方向+预算      → obs[67:79]

观测承诺（硬劫持契约核心）：
    eff_flag_arm = script_flag_arm AND standing_flag
    obs 先验告知"本步该臂将被接管、接管成什么值"；覆写执行严格
    兑现承诺。eff_flag=0 的臂 tgt 写 0（"无指令"标记）。

站立门控（硬劫持无奖励仲裁，失衡强动臂有害）——复用 S3 去抖
状态机：h>flag_h 持续 flag_settle 帧 → standing_flag=1；
低于阈值宽限 flag_grace 帧 → 清零。

set_action 顺序：兑现承诺（记录 intent_err=‖策略输出−tgt‖ →
覆写 act[15:18]/[18:21]）→ super().set_action → 推进 script +
门控。死臂维轨迹存的是策略输出（从未执行，有界漂移无害——
但部署必须配覆写执行器）。

报告经 get_derived_state(["arm_dual"]) 供 observer 读取：
    err_r/l  = intent_err（策略输出与被执行 tgt 的距离）
    hij_r/l  = 本帧实际覆写的臂（= eff_flag 快照）
    flag_r/l = script 调度位（信息用；覆写与否看 hij）
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .v2_simulator import BODY_DIM
from .v2_stepbudget_simulator import V2StepBudgetSimulator
from .arm_dual_script import ArmDualScript

_ROBOTS = ("robot_a", "robot_b")
_ARM_SEG = slice(0, 8)
_ARM_ACT_R = slice(15, 18)
_ARM_ACT_L = slice(18, 21)


class V2StepBudgetArmHardSimulator(V2StepBudgetSimulator):
    """V2StepBudgetSimulator + 每机双臂独立硬劫持（覆写执行器）。"""

    def __init__(self, *args: Any,
                 arm_hold_range: Tuple[int, int] = (2, 20),
                 arm_tgt_range: Tuple[float, float] = (-1.0, 1.0),
                 dual_prob: float = 0.1,
                 flag_height: float = 1.10,
                 flag_grace: int = 5,
                 flag_settle: int = 20,
                 **kw: Any) -> None:
        kw["arm_script"] = False
        super().__init__(*args, **kw)
        self._hold_range = tuple(arm_hold_range)
        self._tgt_range = tuple(arm_tgt_range)
        self._dual_p = float(dual_prob)
        self._flag_h = float(flag_height)
        self._flag_grace = int(flag_grace)
        self._flag_settle = int(flag_settle)
        self._dst: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------- reset --
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(seed=seed, options=options)
        base = 0 if seed is None else int(seed) & 0xFFFFFFFF
        for i, rid in enumerate(_ROBOTS):
            rng = np.random.default_rng([base, 0xE7 + i])
            st = {"script": ArmDualScript(
                      rng, self._hold_range, self._tgt_range,
                      self._dual_p),
                  "timer": 0, "below": 0, "sflag": 0.0,
                  "err": None, "hij": (False, False)}
            st["script"].step()          # obs[0] 即显示首个周期承诺
            self._dst[rid] = st

    # ---------------------------------------------------------- 指令段 -----
    def _eff_flags(self, st: Dict[str, Any]) -> Tuple[bool, bool]:
        fl = st["script"].flags
        ok = st["sflag"] >= 0.5
        return (bool(fl[0]) and ok, bool(fl[1]) and ok)

    def get_observation(self) -> Dict[str, Any]:
        base = super().get_observation()   # 已写步令段 seg[10:22]
        out = self._data_cache.get("_obs_v2budhard")
        if out is None:
            out = {}
            for rid, o in base.items():
                arr = np.asarray(o, np.float32).copy()
                st = self._dst.get(rid)
                if st is not None:
                    seg = np.zeros(8, np.float32)
                    ef = self._eff_flags(st)
                    tg = st["script"].current
                    if ef[0]:
                        seg[0] = 1.0
                        seg[1:4] = tg[0:3]
                    if ef[1]:
                        seg[4] = 1.0
                        seg[5:8] = tg[3:6]
                    arr[BODY_DIM + _ARM_SEG.start:
                        BODY_DIM + _ARM_SEG.stop] = seg
                out[rid] = arr
            self._data_cache["_obs_v2budhard"] = out
        return out

    # -------------------------------------------------------- action 覆写 --
    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        """兑现 obs 承诺（记录 intent_err → 覆写），再推进状态。"""
        patched: Dict[str, Optional[Any]] = dict(action)
        for rid in _ROBOTS:
            st = self._dst.get(rid)
            if st is None:
                continue
            raw = patched.get(rid)
            if raw is None:
                st["err"], st["hij"] = None, (False, False)
                continue
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().numpy()
            act = np.asarray(raw, np.float32).copy()
            ef = self._eff_flags(st)
            tg = st["script"].current
            st["hij"] = ef
            st["err"] = (
                float(np.linalg.norm(act[_ARM_ACT_R] - tg[0:3]))
                if ef[0] else 0.0,
                float(np.linalg.norm(act[_ARM_ACT_L] - tg[3:6]))
                if ef[1] else 0.0,
            )
            if ef[0]:
                act[_ARM_ACT_R] = tg[0:3]
            if ef[1]:
                act[_ARM_ACT_L] = tg[3:6]
            patched[rid] = act
        super().set_action(patched)
        # ---- 推进：站立门控 + 调度器（每 action step 恰好一次） ----
        core = self.get_core_state()
        for rid in _ROBOTS:
            st = self._dst.get(rid)
            if st is None:
                continue
            h = float(np.asarray(core[rid]["root_pos"], np.float64)[2])
            if h > self._flag_h:
                st["timer"] += 1
                st["below"] = 0
            else:
                st["below"] += 1
                if st["below"] > self._flag_grace:
                    st["timer"] = 0
            st["sflag"] = 1.0 if st["timer"] >= self._flag_settle \
                else 0.0
            st["script"].step()

    # ------------------------------------------------- observer 接口 ------
    def arm_dual(self, rid: str) -> Optional[Dict[str, float]]:
        st = self._dst.get(rid)
        if st is None or st["err"] is None:
            return None
        err, hij = st["err"], st["hij"]
        fl = st["script"].flags
        return {"err_r": float(err[0]), "err_l": float(err[1]),
                "hij_r": float(hij[0]), "hij_l": float(hij[1]),
                "flag_r": float(fl[0]), "flag_l": float(fl[1])}

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        """扩 DATASPEC：stepbudget（父类）+ arm_dual（本类）。"""
        _EXTRA = "arm_dual"
        if fields is None:
            out = super().get_derived_state()
            out[_EXTRA] = {r: self.arm_dual(r) for r in _ROBOTS}
            return out
        fields = list(fields)
        base = [f for f in fields if f != _EXTRA]
        out = (super().get_derived_state(base) if base else {})
        if _EXTRA in fields:
            out[_EXTRA] = {r: self.arm_dual(r) for r in _ROBOTS}
        return out


__all__ = ["V2StepBudgetArmHardSimulator"]
