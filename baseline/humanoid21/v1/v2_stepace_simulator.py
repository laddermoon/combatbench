"""STEPACE V1 验收仿真器：脚本令源 + 事件门控发令（STEPACE_V1_PLAN）。

继承 V2StepTvarSimulator，把随机采样令源换成 per-robot
TaskScript：每 obs 帧结算后查询脚本门控——门控命中（落地/
首达）或到期，都走 _settle+晋升；门控晋升标 trunc=1。

脚本接口（step_task_scripts.py）：
    reset(ctx)        任务系锚点
    should_issue(ctx) 门控判定（fixed 门控恒 False→等 dur 到期）
    produce(ctx)      → (StepCmd, dur_cap) 含 P 纠偏
    peek(ctx)         → 下一条名义令（预览）

脚本令照走撞墙检查（出界→保持令兜底）。obs/残差/结算语义
全部继承不变。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import mujoco
import numpy as np

from .v2_simulator import V2Simulator
from .v2_steptvar_simulator import V2StepTvarSimulator
from .v2_stepcmd_simulator import _ROBOTS
from .step_cmd_gen import StepCmd
from .step_task_scripts import make_script

_GROUND_GEOM = "ground"


class V2StepAceSimulator(V2StepTvarSimulator):
    """V2StepTvarSimulator + 脚本令源 + 事件门控晋升。"""

    def __init__(self, *args: Any,
                 step_script: Optional[Dict[str, Any]] = None,
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._script_spec: Dict[str, Any] = dict(
            step_script or {"name": "walk_fwd"})
        self._script: Dict[str, Any] = {}

    # ------------------------------------------------ 外部配置 ----------
    def configure_script(self, name: str, **params: Any) -> None:
        """harness 入口：换任务/门控/纠偏，下次 reset 生效。"""
        self._script_spec = {"name": name, **params}

    # ------------------------------------------------------------- reset --
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(seed=seed, options=options)
        if not self._step_on:
            return
        base = 0 if seed is None else int(seed) & 0xFFFFFFFF
        for i, rid in enumerate(_ROBOTS):
            spec = dict(self._script_spec)
            name = spec.pop("name")
            rng = np.random.default_rng([base, 0xA7 + i])
            sc = make_script(name, rng, **spec)
            ctx = self._script_ctx(rid)
            sc.reset(ctx)
            self._script[rid] = sc
            self._rotate_scripted(rid, ctx)

    # -------------------------------------------------------- 帧结算 -------
    def physical_step(self) -> None:
        """末子步：结算 + 门控/到期晋升。"""
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
            ctx = self._script_ctx(rid)
            expired = st["age_f"] >= st["dur_f"]
            gated = (not expired
                     and self._script[rid].should_issue(ctx))
            if expired or gated:
                self._settle(rid)
                st["report"]["trunc"] = 1.0 if gated else 0.0
                self._rotate_scripted(rid, ctx)

    # -------------------------------------------------------- 内部 --------
    def _foot_contacts(self, rid: str) -> tuple:
        """(左触地, 右触地)：脚体↔ground geom 任一 contact。"""
        kp = self._robot(rid)["keypoint_body_ids"]
        fl_b, fr_b = kp["foot_left"], kp["foot_right"]
        cl = cr = False
        m = self.model
        for c in self.data.contact:
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            if b1 == fl_b or b2 == fl_b:
                g = c.geom2 if b1 == fl_b else c.geom1
                if mujoco.mj_id2name(
                        m, mujoco.mjtObj.mjOBJ_GEOM, g) == _GROUND_GEOM:
                    cl = True
            if b1 == fr_b or b2 == fr_b:
                g = c.geom2 if b1 == fr_b else c.geom1
                if mujoco.mj_id2name(
                        m, mujoco.mjtObj.mjOBJ_GEOM, g) == _GROUND_GEOM:
                    cr = True
        return cl, cr

    def _script_ctx(self, rid: str) -> Dict[str, Any]:
        st = self._sst[rid]
        fl, fr, yaw = self._feet_yaw(rid)
        kp = self._robot(rid)["keypoint_body_ids"]
        cl, cr = self._foot_contacts(rid)
        return {
            "fl": fl, "fr": fr, "yaw": yaw,
            "z_l": float(self.data.xpos[kp["foot_left"]][2]),
            "z_r": float(self.data.xpos[kp["foot_right"]][2]),
            "c_l": cl, "c_r": cr,
            "age": int(st["age_f"]), "dur": int(st["dur_f"]),
            "t_arrive": int(st["t_arrive"]),
            "cur": st["cur"], "cmd_i": self._script.get(rid) and
            self._script[rid]._i or 0,
        }

    def _rotate_scripted(self, rid: str, ctx: Dict[str, Any]) -> None:
        """脚本晋升：produce→撞墙检查→issue→peek→写预览。"""
        st = self._sst[rid]
        sc = self._script[rid]
        cmd, dur = sc.produce(ctx)
        if not isinstance(cmd, StepCmd):
            cmd = StepCmd()
        if self._collide(rid, cmd, self._feet_yaw(rid)):
            cmd = StepCmd()                      # 出界→保持兜底
        st["cur"] = cmd
        st["pending_dur"] = max(1, int(dur))
        self._issue(rid)                          # 锚定+诊断+采样dur
        st["dur_f"] = st["pending_dur"]           # 脚本时长覆盖
        pk = sc.peek(ctx)
        st["nxt"] = pk if isinstance(pk, StepCmd) else StepCmd()
        self._write_nxt_vec(rid)


__all__ = ["V2StepAceSimulator"]
