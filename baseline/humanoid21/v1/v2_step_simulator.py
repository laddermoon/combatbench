"""V2 步令跟随仿真器（S4）：指令流状态机 + 步边界结算（S4_PLAN §2）。

继承 V2Simulator（79 维 obs + 臂脚本接管）。新增每机独立指令流：

    reset:  随机姿态起链 → cur / next 两条采样令（age=0）
    每 action step（set_action 推进点，D6 同点位）:
        bd.update(feet_forces)  —— obs 同源力信号驱相位机
        ├─ 边界/超时 → 量测(脚位+yaw) vs 发令锚的 apply_order 目标
        │            → step_report 落袋（observer 取）
        │            → cur←next, next=sched.sample(墙 collision_fn)
        └─ 否则 age += 0.1 继续
    get_observation: seg[10:16]=step_cur(5+age), seg[16:22]=step_next

step_report 字段（边界帧一次性产出，observer 每步读取）：
    event/foot_ok/timed_out/executed/cmd_foot/d_mv/d_st/d_yaw/order

部署 parity：step_boundary.StepBoundary 与 feet_forces 皆是 obs
流纯函数——指令源在部署侧拿同一 obs 驱动同码，轮转时机一致。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .v2_simulator import BODY_DIM, CMD_DIM, OBS_DIM, V2Simulator
from .arena_geom import inside_arena
from .step_boundary import StepBoundary
from .step_gait import (
    POSTURES, StepOrder, apply_order,
)
from .step_scheduler import SchedState, StepScheduler

_ROBOTS = ("robot_a", "robot_b")
_CUR = slice(10, 16)     # seg 内偏移：step_cur
_NXT = slice(16, 22)     # seg 内偏移：step_next
_FOOT_BODY = {-1: "foot_left", +1: "foot_right"}


def _quat_yaw(q_wxyz: np.ndarray) -> float:
    """wxyz quat → 偏航角（躯干 heading，+x 前）。"""
    w, x, y, z = (float(v) for v in q_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class V2StepSimulator(V2Simulator):
    """V2Simulator + 双机步令调度/结算。"""

    def __init__(self, *args: Any,
                 step_enabled: bool = True,
                 lift_thr: float = 0.1,
                 land_thr: float = 0.4,
                 lift_deb: int = 2,
                 timeout_age: float = 6.0,
                 age_inc: float = 0.1,
                 wall_margin: float = 0.30,
                 steps_per_action: int = 25,
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._step_on = bool(step_enabled)
        self._bd_args = dict(lift_thr=lift_thr, land_thr=land_thr,
                             lift_deb=lift_deb, timeout_age=timeout_age,
                             age_inc=age_inc)
        self._wall_margin = float(wall_margin)
        # 必须与 yaml runtime.phy_steps_per_action 一致——检测挂在
        # 每 action step 最后一个物理子步，使该帧 obs 力=检测输入
        # （部署侧指令源见到这帧 obs 即轮转 ⇒ 训/部时序严格对齐）。
        self._spa = int(steps_per_action)
        self._phys_sub = 0
        # 每机 {st, cur, nxt, bd, anchor, tgt, report, sched}
        self._sst: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------- reset --
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(seed=seed, options=options)
        self._phys_sub = 0
        if not self._step_on:
            return
        base = 0 if seed is None else int(seed) & 0xFFFFFFFF
        for i, rid in enumerate(_ROBOTS):
            rng = np.random.default_rng([base, 0xC7 + i])
            sched = StepScheduler(seed=int(rng.integers(1 << 31)))
            init_p = POSTURES[int(rng.integers(len(POSTURES)))]
            st = {"sched": sched,
                  "st": SchedState.initial(init_p),
                  "bd": StepBoundary(**self._bd_args),
                  "report": None}
            self._sst[rid] = st
            st["cur"] = sched.sample(st["st"])
            st["st"] = StepScheduler.advance(st["st"], st["cur"])
            st["nxt"] = self._sample_next(rid)
            self._issue(rid)

    # ---------------------------------------------------------- 指令段 -----
    def get_observation(self) -> Dict[str, Any]:
        base = super().get_observation()
        if not self._step_on:
            return base
        ck = "_obs_v2step"
        out = self._data_cache.get(ck)
        if out is None:
            out = {}
            for rid, o in base.items():
                arr = np.asarray(o, np.float32).copy()
                st = self._sst.get(rid)
                if st is not None:
                    cur_v = st["cur"].to_vec()
                    cur_v[5] = np.float32(st["bd"].age)
                    arr[BODY_DIM + _CUR.start:BODY_DIM + _CUR.stop] = cur_v
                    arr[BODY_DIM + _NXT.start:BODY_DIM + _NXT.stop] = \
                        st["nxt"].to_vec()
                out[rid] = arr
            self._data_cache[ck] = out
        return out

    # -------------------------------------------------------- 结算/轮转 ---
    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        """兑现 obs 承诺（父类臂覆写+状态推进）；新 action step 起算子步。"""
        super().set_action(action)
        self._phys_sub = 0

    def physical_step(self) -> None:
        """物理子步推进；末子步做步令结算（与部署轮转同帧对齐）。"""
        super().physical_step()
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
            ev = st["bd"].update(self._get_feet_forces(rid))
            if ev is not None:
                self._settle(rid, ev)
                self._rotate(rid)

    # ----------------------------------------------------------- 内部 ----
    def _feet_yaw(self, rid: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """当前双脚世界 xy + 躯干 yaw。"""
        kp = self._robot(rid)["keypoint_body_ids"]
        core = self.get_core_state()
        yaw = _quat_yaw(np.asarray(core[rid]["root_rot"], np.float64))
        fl = np.asarray(self.data.xpos[kp["foot_left"]][:2], np.float64)
        fr = np.asarray(self.data.xpos[kp["foot_right"]][:2], np.float64)
        return fl, fr, yaw

    def _issue(self, rid: str) -> None:
        """cur 令生效：存发令锚 + 预算目标 + 相位机开新周期。"""
        st = self._sst[rid]
        fl, fr, yaw = self._feet_yaw(rid)
        tfl, tfr, tyaw = apply_order(fl, fr, yaw, st["cur"])
        st["anchor"] = (fl, fr, yaw)
        st["tgt"] = (tfl, tfr, tyaw)
        st["bd"].begin_order(st["cur"].foot)

    def _sample_next(self, rid: str):
        st = self._sst[rid]

        def collides(o: StepOrder, _st: SchedState) -> bool:
            fl, fr, yaw = self._feet_yaw(rid)   # 采样时刻实测位姿
            tfl, tfr, _ = apply_order(fl, fr, yaw, o)
            mv = tfl if o.foot < 0 else tfr
            return not inside_arena(mv, self._wall_margin)

        return st["sched"].sample(st["st"], collision_fn=collides)

    def _rotate(self, rid: str) -> None:
        st = self._sst[rid]
        st["cur"] = st["nxt"]
        st["nxt"] = self._sample_next(rid)
        self._issue(rid)

    def _settle(self, rid: str, ev) -> None:
        """边界帧量测结算：executed/超时 → 报告。"""
        st = self._sst[rid]
        fl, fr, yaw = self._feet_yaw(rid)
        tfl, tfr, tyaw = st["tgt"]
        afl, afr, _ = st["anchor"]
        o = st["cur"]

        if ev.timed_out:
            executed = None
            meas_foot = o.foot                     # 该动的脚没动成
            ex_pos = fl if o.foot < 0 else fr      # 按当前位量测
        else:
            executed = int(ev.executed)
            ex_pos = fl if executed < 0 else fr

        tgt_mv = tfl if o.foot < 0 else tfr        # 目标动脚落点
        # 另一脚（stance）：超时=非指令脚；落地=非执行脚
        other_side = -o.foot if executed is None else -executed
        other_pos = fl if other_side < 0 else fr
        other_anchor = afl if other_side < 0 else afr

        st["report"] = {
            "event": 1.0,
            "foot_ok": float(executed == o.foot) if executed is not None else 0.0,
            "timed_out": float(ev.timed_out),
            "executed": float(executed) if executed is not None else 0.0,
            "cmd_foot": float(o.foot),
            "d_mv": float(np.linalg.norm(ex_pos - tgt_mv)),
            "d_st": float(np.linalg.norm(other_pos - other_anchor)),
            "d_yaw": float(abs(_wrap_pi(yaw - tyaw))),
            "age": float(ev.age),
            "order": o.to_vec(),
        }

    # ------------------------------------------------- observer 接口 ------
    def step_report(self, rid: str) -> Optional[Dict[str, Any]]:
        """本步边界报告（内部/测试用）；observer 走 get_derived_state。"""
        st = self._sst.get(rid)
        return None if st is None else st["report"]

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        """扩 DATASPEC：step_reports = {rid: 边界报告dict|None}。"""
        _EXTRA = "step_reports"
        if fields is None:
            out = super().get_derived_state()
            out[_EXTRA] = {r: self.step_report(r) for r in _ROBOTS}
            return out
        fields = list(fields)
        base = [f for f in fields if f != _EXTRA]
        out = (super().get_derived_state(base) if base else {})
        if _EXTRA in fields:
            out[_EXTRA] = {r: self.step_report(r) for r in _ROBOTS}
        return out


__all__ = ["V2StepSimulator"]
