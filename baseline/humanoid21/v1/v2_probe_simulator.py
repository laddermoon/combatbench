"""CMD_PROBE 仿真器：96 维 obs 状态注入 + 22 维指令注入。

配套 cmd_probe.py（Lib）：obs→RobotState(from_obs) 写回本 sim，
22 维指令段反解为锚定目标（残差→StepCmd 闭式逆），rollout
期间 obs 指令段由真值自然演化，结算走父类标准管线。

probe_apply(rid, spec)（env.reset 后调用）：
    spec = {
        "step_vec":  (6,) | None   # [foot,Δx,Δy,Δψ,0,0] 当前令残差
        "dur":       int           # 当前令时长（到期→晋升 nxt→hold）
        "step_nxt_vec": (6,)|None  # 预览令残差（对 P̂ 反解）
        "nxt_dur":   int           # 预览令时长
        "arm":       (flags(2,), tgt(6,)) | None
                     # None→双臂自由（flag=0,tgt=0）
    }
注入态语义：RobotState 是 P5 系（踝中点原点/躯干+x）——指令
均相对量，世界系位置平移不影响完成度度量；robot_b 停置
(10,10) 不干扰（V2 obs 已剥对手段）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

from .v2_simulator import V2Simulator
from .v2_steparm_simulator import V2StepArmSimulator
from .v2_stepcmd_simulator import _ROBOTS
from .step_cmd_gen import StepCmd, _rot


class _FrozenArmScript:
    """冻结臂令：接口同 ArmDualScript（flags/current/step）。"""

    def __init__(self, flags: np.ndarray, tgt: np.ndarray) -> None:
        self._f = np.asarray(flags, bool).copy()
        self._t = np.asarray(tgt, np.float32).copy()

    @property
    def flags(self) -> np.ndarray:
        return self._f

    @property
    def current(self) -> np.ndarray:
        return self._t

    def step(self) -> None:
        pass


def res_to_stepcmd(fl: np.ndarray, fr: np.ndarray, yaw: float,
                   vec: np.ndarray) -> StepCmd:
    """残差 → StepCmd 闭式逆（cmd_target/residual 之逆）。

    vec = [foot, Δx, Δy, Δψ,…]（本体系残差）；foot=0 时 Δψ 不可
    表达（cmd_target 对保持令吞 yaw——已知语义边界，调用方警告）。
    """
    foot = int(round(float(vec[0])))
    if foot == 0:
        return StepCmd()
    ankle = np.asarray(fl if foot < 0 else fr, float)
    stance = np.asarray(fr if foot < 0 else fl, float)
    yaw_t = float(yaw) + float(vec[3])
    tgt = ankle + _rot(float(yaw)) @ np.array([vec[1], vec[2]], float)
    v = _rot(yaw_t).T @ (tgt - stance)
    lat = float(v[1]) if foot < 0 else -float(v[1])
    return StepCmd(foot=foot, sag=float(v[0]), lat=lat,
                   dyaw=float(vec[3]))


class V2CmdProbeSimulator(V2StepArmSimulator):
    """V2StepArmSimulator + 状态注入 + 注入指令序列轮转。"""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._probe: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------ 状态注入 ----------
    def inject_state(self, rid: str, rs: Any,
                     park_other: bool = True) -> None:
        """RobotState → sim 真值（take_action 同款写回配方）。"""
        from point5.humanoid21.robot_state import _zyx_to_quat_wxyz
        c = self._robot(rid)
        adr, vadr = int(c["root_qpos_adr"]), int(c["root_qvel_adr"])
        quat = _zyx_to_quat_wxyz(rs.root_orientation)
        self.data.qpos[adr:adr + 3] = rs.root_position
        self.data.qpos[adr + 3:adr + 7] = quat
        self.data.qpos[np.asarray(c["qpos_indices"])] = rs.qpos
        self.data.qvel[vadr:vadr + 3] = rs.root_linvel
        Rm = np.empty(9)
        mujoco.mju_quat2Mat(Rm, quat)
        self.data.qvel[vadr + 3:vadr + 6] = (
            Rm.reshape(3, 3).T @ rs.root_angvel)
        self.data.qvel[np.asarray(c["qvel_indices"])] = rs.qvel
        if park_other:
            other = "robot_b" if rid == "robot_a" else "robot_a"
            oc = self._robot(other)
            badr = int(oc["root_qpos_adr"])
            bvadr = int(oc["root_qvel_adr"])
            self.data.qpos[badr:badr + 3] = (10.0, 10.0, 0.5)
            self.data.qvel[bvadr:bvadr + 6] = 0.0
        self._data_cache.clear()
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------ 指令注入 ----------
    def probe_apply(self, rid: str, spec: Dict[str, Any]) -> None:
        """注入 22 维指令（env.reset + inject_state 之后）。"""
        st, dst = self._sst[rid], self._dst[rid]
        arm = spec.get("arm")
        if arm is None:
            dst["script"] = _FrozenArmScript(
                np.zeros(2, bool), np.zeros(6))
        else:
            dst["script"] = _FrozenArmScript(*arm)

        fl, fr, yaw = self._feet_yaw(rid)
        sv = spec.get("step_vec")
        st["cur"] = (res_to_stepcmd(fl, fr, yaw, sv)
                     if sv is not None else StepCmd())
        self._issue(rid)
        st["dur_f"] = max(1, int(spec.get("dur", 20)))

        pfl, pfr, pyaw = self._post_state(st)
        nv = spec.get("step_nxt_vec")
        nxt = (res_to_stepcmd(pfl, pfr, pyaw, nv)
               if nv is not None else StepCmd())
        st["nxt"] = nxt
        self._write_nxt_vec(rid)
        self._probe[rid] = {
            "nxt_pending": (nxt if nxt.foot != 0 else None),
            "nxt_dur": int(spec.get("nxt_dur", 15)),
        }
        self._data_cache.clear()          # obs 立刻反映注入指令

    # ------------------------------------------------ 帧结算+轮转 -------
    def physical_step(self) -> None:
        V2Simulator.physical_step(self)
        if not self._step_on or not self._probe:
            return
        self._phys_sub += 1
        if self._phys_sub < self._spa:
            return
        self._phys_sub = 0
        for rid in _ROBOTS:
            st = self._sst.get(rid)
            if st is None or rid not in self._probe:
                continue
            st["report"] = None
            st["age_f"] += 1
            self._check_arrive(rid)
            if st["age_f"] >= st["dur_f"]:
                self._settle(rid)
                self._rotate_probe(rid)

    def _rotate_probe(self, rid: str) -> None:
        spec = self._probe[rid]
        st = self._sst[rid]
        pend = spec.get("nxt_pending")
        if pend is not None:
            cmd, dur = pend, spec["nxt_dur"]
            spec["nxt_pending"] = None
        else:
            cmd, dur = StepCmd(), 15
        st["cur"] = cmd
        self._issue(rid)
        st["dur_f"] = dur
        st["nxt"] = StepCmd()
        self._write_nxt_vec(rid)


__all__ = ["V2CmdProbeSimulator", "res_to_stepcmd"]
