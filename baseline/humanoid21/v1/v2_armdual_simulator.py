"""双臂独立软劫持仿真器（ARMDUAL_V1_PLAN）：逐臂 flag+目标，只进观测。

继承 V2Simulator（79 维 obs），但**父类臂机制全程关闭**
（arm_script=False → flag 状态机/接管覆写/旧 ArmScript 均不起）：
指令流由本类 ArmDualScript 独立驱动——两臂共享同步周期时钟，
P(0.1) 双臂同劫 / P(0.9) 单臂（左右 50/50），被劫臂 tgt~U(-1,1)³，
自由臂 flag=0、tgt=0（"不存在目标"）。

obs 指令段新布局（腰段让 1 位，总量 22 维不变）：
    seg[0:8] = [flag_r, tgt_r(3), flag_l, tgt_l(3)]  → obs[57:65]
    seg[8:10] = waist 预留零                         → obs[65:67]
    seg[10:22] = step_*（step_enabled=False 时全零） → obs[67:79]

软劫持语义（D17）：act[15:21] 不覆写、归策略全权；被劫臂
err = ‖act−tgt‖ 逐帧结算上报，自由臂 err=0 不计分。
报告经 get_derived_state(["arm_dual"]) 供 observer 读取。
flag_r/l 为 live 当前值；hij_r/l 为 err 结算时刻快照（边界帧
不错位——同 armcmd 的 rel 快照惯例）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .v2_simulator import BODY_DIM, V2Simulator
from .arm_dual_script import ArmDualScript

_ROBOTS = ("robot_a", "robot_b")
_ARM_SEG = slice(0, 8)     # seg 内偏移：双臂独立令 → obs[57:65]
_ARM_ACT_R = slice(15, 18)   # 右臂关节段
_ARM_ACT_L = slice(18, 21)   # 左臂关节段


class V2ArmDualSimulator(V2Simulator):
    """V2Simulator + 每机双臂独立劫持指令流（软，不覆写动作）。"""

    def __init__(self, *args: Any,
                 arm_hold_range: Tuple[int, int] = (2, 20),
                 arm_tgt_range: Tuple[float, float] = (-1.0, 1.0),
                 dual_prob: float = 0.1,
                 **kw: Any) -> None:
        # 父类臂机制关闭：arm_script=False → _arm_on=False，
        # flag 状态机/覆写/旧 ArmScript 全部休眠（seg[0:8] 本类自写）
        kw["arm_script"] = False
        super().__init__(*args, **kw)
        self._hold_range = tuple(arm_hold_range)
        self._tgt_range = tuple(arm_tgt_range)
        self._dual_p = float(dual_prob)
        # 每机 {script, err, hij}（reset 时初始化）
        self._dst: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------- reset --
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(seed=seed, options=options)
        base = 0 if seed is None else int(seed) & 0xFFFFFFFF
        for i, rid in enumerate(_ROBOTS):
            rng = np.random.default_rng([base, 0xE2 + i])
            st = {"script": ArmDualScript(
                      rng, self._hold_range, self._tgt_range,
                      self._dual_p),
                  "err": None, "hij": (False, False)}
            st["script"].step()          # obs[0] 即显示首个周期承诺
            self._dst[rid] = st

    # ---------------------------------------------------------- 指令段 -----
    def get_observation(self) -> Dict[str, Any]:
        base = super().get_observation()      # seg[0:8] 父类保持零
        ck = "_obs_v2dual"
        out = self._data_cache.get(ck)
        if out is None:
            out = {}
            for rid, o in base.items():
                arr = np.asarray(o, np.float32).copy()
                st = self._dst.get(rid)
                if st is not None:
                    seg = np.zeros(8, np.float32)
                    fl = st["script"].flags
                    tg = st["script"].current
                    seg[0] = float(fl[0])
                    seg[1:4] = tg[0:3]
                    seg[4] = float(fl[1])
                    seg[5:8] = tg[3:6]
                    arr[BODY_DIM + _ARM_SEG.start:
                        BODY_DIM + _ARM_SEG.stop] = seg
                out[rid] = arr
            self._data_cache[ck] = out
        return out

    # -------------------------------------------------------- action 结算 --
    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        """先按 obs 承诺的 flag/tgt 结算被劫臂 err，再推进脚本。"""
        for rid in _ROBOTS:
            st = self._dst.get(rid)
            if st is None:
                continue
            raw = action.get(rid)
            if raw is None:
                st["err"], st["hij"] = None, (False, False)
                continue
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().numpy()
            act = np.asarray(raw, np.float32)
            fl = st["script"].flags
            tg = st["script"].current
            st["hij"] = (bool(fl[0]), bool(fl[1]))
            st["err"] = (
                float(np.linalg.norm(act[_ARM_ACT_R] - tg[0:3]))
                if fl[0] else 0.0,
                float(np.linalg.norm(act[_ARM_ACT_L] - tg[3:6]))
                if fl[1] else 0.0,
            )
        super().set_action(action)
        for rid in _ROBOTS:
            st = self._dst.get(rid)
            if st is not None:
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
        """扩 DATASPEC：arm_dual = {rid: {err,hij,flag}|None}。"""
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


__all__ = ["V2ArmDualSimulator"]
