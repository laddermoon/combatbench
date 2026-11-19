"""V2 步令跟随仿真器（S4）：指令流状态机 + 时限预算结算（D16）。

继承 V2Simulator（79 维 obs + 臂脚本接管）。新增每机独立指令流：

    reset:  随机姿态起链 → cur / next 两条采样令
    每 action step（physical_step 末子步，obs 帧对齐点）:
        age_f += 1
        └─ age_f ≥ budget → 到期结算：量测(脚位+yaw) vs 发令锚的
           apply_order 目标 → step_report 落袋（observer 取）
           → cur←next, next=sched.sample(墙 collision_fn)
    get_observation: seg[10:16]=step_cur(5+age/budget), seg[16:22]=step_next

时限预算（D16）：令到点必须处于目标状态，中间过程不测量。
发令时刻按动脚行程线性插值：H = h_min + (h_max-h_min)·s/s_max，
h_min=0.15s, h_max=0.50s（20Hz → 3~10 帧），s_max≈0.53m。

step_report 字段（到期帧一次性产出，observer 每步读取）：
    event/foot_ok(d_mv<tol)/executed/cmd_foot/d_mv/d_st/d_yaw/
    age/budget/order

部署 parity：指令源在部署侧只需按同样公式数帧——发令时拿
动脚位+目标位算同一 budget，轮转时机天然一致（无观测侧检测）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .v2_simulator import BODY_DIM, CMD_DIM, OBS_DIM, V2Simulator
from .arena_geom import inside_arena
from .step_gait import (
    POSTURES, StepOrder, apply_order,
)
from .step_scheduler import SchedState, StepScheduler

_ROBOTS = ("robot_a", "robot_b")
_CUR = slice(10, 16)     # seg 内偏移：step_cur
_NXT = slice(16, 22)     # seg 内偏移：step_next
_FOOT_BODY = {-1: "foot_left", +1: "foot_right"}
_EXEC_EPS = 0.01         # 判定"哪只脚动了"的最小位移 m


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
                 budget_min_s: float = 0.15,
                 budget_max_s: float = 0.50,
                 budget_travel_m: float = 0.53,
                 foot_tol: float = 0.10,
                 wall_margin: float = 0.30,
                 steps_per_action: int = 25,
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._step_on = bool(step_enabled)
        self._travel_max = float(budget_travel_m)
        self._foot_tol = float(foot_tol)
        self._wall_margin = float(wall_margin)
        # 必须与 yaml runtime.phy_steps_per_action 一致——结算挂在
        # 每 action step 最后一个物理子步，使到期帧 obs=结算量测点
        # （部署侧指令源数同一帧计数 ⇒ 训/部时序严格对齐）。
        self._spa = int(steps_per_action)
        dt_action = float(self.model.opt.timestep) * self._spa
        self._h_min = max(1.0, budget_min_s / dt_action)
        self._h_max = max(self._h_min, budget_max_s / dt_action)
        self._phys_sub = 0
        # 每机 {st, cur, nxt, anchor, tgt, budget, age_f, report, sched}
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
                    cur_v[5] = np.float32(
                        min(st["age_f"] / max(st["budget"], 1e-9), 1.0))
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
            st["age_f"] += 1
            if st["age_f"] >= st["budget"]:
                self._settle(rid)
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
        """cur 令生效：存发令锚 + 目标位 + 按行程算时限预算。"""
        st = self._sst[rid]
        fl, fr, yaw = self._feet_yaw(rid)
        tfl, tfr, tyaw = apply_order(fl, fr, yaw, st["cur"])
        st["anchor"] = (fl, fr, yaw)
        st["tgt"] = (tfl, tfr, tyaw)
        # 时限预算：动脚行程 s 线性插值 [h_min, h_max]（D16）
        mv_pos = fl if st["cur"].foot < 0 else fr
        mv_tgt = tfl if st["cur"].foot < 0 else tfr
        s = float(np.linalg.norm(mv_pos - mv_tgt))
        st["budget"] = (self._h_min + (self._h_max - self._h_min)
                        * min(s / self._travel_max, 1.0))
        st["age_f"] = 0

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

    def _settle(self, rid: str) -> None:
        """到期帧量测结算：到位判定 → 报告。"""
        st = self._sst[rid]
        fl, fr, yaw = self._feet_yaw(rid)
        tfl, tfr, tyaw = st["tgt"]
        afl, afr, _ = st["anchor"]
        o = st["cur"]

        mv_pos = fl if o.foot < 0 else fr
        tgt_mv = tfl if o.foot < 0 else tfr        # 目标动脚落点
        other_pos = fr if o.foot < 0 else fl       # stance 脚=非指令脚
        other_anchor = afr if o.foot < 0 else afl
        d_mv = float(np.linalg.norm(mv_pos - tgt_mv))

        # 诊断：哪只脚实际动了（净位移大者；都≈0 → 0 未动）
        dl = float(np.linalg.norm(fl - afl))
        dr = float(np.linalg.norm(fr - afr))
        executed = (0.0 if max(dl, dr) < _EXEC_EPS
                    else (-1.0 if dl > dr else 1.0))

        st["report"] = {
            "event": 1.0,
            "foot_ok": float(d_mv < self._foot_tol),
            "executed": executed,
            "cmd_foot": float(o.foot),
            "d_mv": d_mv,
            "d_st": float(np.linalg.norm(other_pos - other_anchor)),
            "d_yaw": float(abs(_wrap_pi(yaw - tyaw))),
            "age": float(st["age_f"]),
            "budget": float(st["budget"]),
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
