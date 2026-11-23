"""STEPBUDGET V1 仿真器：方向 + 时间预算步令（STEPBUDGET_V1_PLAN）。

继承 V2StepCmdSimulator 只为复用 _feet_yaw / 观测缓存 / derived
接口骨架；指令语义全部重写：

    obs 指令段（cur=obs[67:73], nxt=obs[73:79]）：
        [foot, ux, uy, Δψ, budget/20, 0]
        u   = 方向单位向量（机体系，每帧按当前 yaw 换算）；原地/保持=(0,0)
        Δψ  = wrap(ψ* − yaw)：到本令**绝对目标朝向**的残差
        cur 预算 = 剩余帧 /20；nxt 预算 = 总预算 /20

    预览即承诺：令在**成为 nxt 时**即钉死世界系（ψ*、方向角），
    晋升 cur←nxt 原样，不重采、不重锚方向。
    ψ* 累积：ψ*_new = ψ*_prev + dpsi，再夹到 |ψ*−yaw_now| ≤ π/4
    （防执行失败时目标跑飞）。

    预算 = 唯一时钟：age_f ≥ budget → 结算 → 晋升。

逐帧 live 字段（本帧活跃令，在晋升**之前**量测，奖励对齐）：
    prog   = ΔΦ，Φ = clip(动脚沿方向投影位移, 0, s_max)
             s_max = 发令时刻沿方向射出站姿包络的最远距离
    viol   = 动脚到站姿包络的距离（目标朝向系，原点=当前轴心脚）——诊断
    viol_end = 结算帧的 viol（包络判的是**终态**站姿；中途 yaw 未到位时
             目标朝向系下站姿看似旋转，逐帧罚会误伤，故只在终态计罚；
             中途过冲已由 Φ 的 s_max 封顶消除激励）
    lat    = 动脚到方向线的垂距（原地踏步=到原位距离）
    pv_dr  = 轴心脚漂移（保持令=双脚之大者）；悬空发令的轴心脚
             在其首次落地时才锚定，落地前计 0
    pv_tol = 漂移死区：令龄 < 5 → 0.10m，否则 0.04m
    yaw_e  = |Δψ|；yaw_w = 预算消耗比 age/budget（保持令=1）
    land   = 结算帧动脚着地（1/0；保持令 0）
    mv     = 本令有动脚（1/0）；zdir = 原地踏步（1/0）

结算报告（结算帧）：budget/stride/s_max/dir_err/lat/landed/viol_max/
pv_dr/yaw_e/rel/clip/rand/hold/zdir。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .arena_geom import inside_arena
from .step_budget_gen import (
    BUDGET_MAX, BudgetCmd, BudgetCmdGen, env_local, env_violation,
    ray_exit,
)
from .v2_simulator import BODY_DIM, V2Simulator
from .v2_stepcmd_simulator import V2StepCmdSimulator, _ROBOTS, _wrap_pi

_CUR = slice(10, 16)
_NXT = slice(16, 22)
_FOOT_Z0 = 0.027            # 站立足心 xipos z（foot_state_observer）
_LAND_DZ = 0.02             # 足心高出站立 < 此值 → 着地
_EARLY_F = 5                # 令龄 < 此值 → 轴心脚宽死区
_TOL_EARLY, _TOL_LATE = 0.10, 0.04
_PSI_CLAMP = math.pi / 4      # |ψ*−yaw| 上限：±90° 实测使未训策略
                              # 下 yaw_e≈1.65、包络因系旋转全违规


class V2StepBudgetSimulator(V2StepCmdSimulator):
    """V2Simulator + 方向/预算步令流（Clip 组合 + 20% 随机）。"""

    def __init__(self, *args: Any, p_random: float = 0.2,
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._p_rand = float(p_random)

    # ------------------------------------------------------------- reset --
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        V2Simulator.reset(self, seed=seed, options=options)
        self._phys_sub = 0
        if not self._step_on:
            return
        base = 0 if seed is None else int(seed) & 0xFFFFFFFF
        for i, rid in enumerate(_ROBOTS):
            rng = np.random.default_rng([base, 0xB6 + i])
            _, _, yaw = self._feet_yaw(rid)
            st: Dict[str, Any] = {
                "gen": BudgetCmdGen(rng, p_random=self._p_rand),
                "psi": yaw, "report": None, "live": None,
            }
            self._sst[rid] = st
            st["cur"] = self._make(rid)
            self._issue(rid)
            st["nxt"] = self._make(rid)

    # ---------------------------------------------------------- 令构造 -----
    def _make(self, rid: str) -> Dict[str, Any]:
        """从生成器取一条令，钉死世界系 ψ* 与方向角（预览即承诺）。"""
        st = self._sst[rid]
        c: BudgetCmd = st["gen"].next()
        _, _, yaw = self._feet_yaw(rid)
        psi = st["psi"] + c.dpsi
        psi = yaw + float(np.clip(_wrap_pi(psi - yaw),
                                  -_PSI_CLAMP, _PSI_CLAMP))
        st["psi"] = psi
        dir_w = None if (c.foot == 0 or c.rel is None) else psi + c.rel
        if dir_w is not None:
            dir_w = self._wall_fix(rid, c.foot, dir_w)
        return {"cmd": c, "foot": c.foot, "psi": psi, "dir": dir_w,
                "budget": int(c.budget)}

    def _wall_fix(self, rid: str, foot: int, dir_w: float) -> float:
        """方向朝墙（动脚 +0.6m 出界）→ 改朝场地中心。"""
        fl, fr, _ = self._feet_yaw(rid)
        p = fl if foot < 0 else fr
        probe = p + 0.6 * np.array([math.cos(dir_w), math.sin(dir_w)])
        if inside_arena(probe, self._wall_margin):
            return dir_w
        return math.atan2(-p[1], -p[0])

    # ----------------------------------------------------------- 发令 -----
    def _foot_z(self, rid: str) -> Tuple[float, float]:
        kp = self._robot(rid)["keypoint_body_ids"]
        return (float(self.data.xipos[kp["foot_left"]][2]),
                float(self.data.xipos[kp["foot_right"]][2]))

    def _grounded(self, z: float) -> bool:
        return z - _FOOT_Z0 < _LAND_DZ

    def _issue(self, rid: str) -> None:
        st = self._sst[rid]
        c = st["cur"]
        fl, fr, yaw = self._feet_yaw(rid)
        zl, zr = self._foot_z(rid)
        st["age_f"] = 0
        st["a_fl"], st["a_fr"] = fl.copy(), fr.copy()
        st["viol_max"] = 0.0
        st["phi"] = 0.0
        if c["foot"] == 0:
            st["a_pv"] = None
            return
        mv, pv = (fl, fr) if c["foot"] < 0 else (fr, fl)
        z_pv = zr if c["foot"] < 0 else zl
        st["a_mv"] = mv.copy()
        st["prev_mv"] = mv.copy()
        st["a_pv"] = pv.copy() if self._grounded(z_pv) else None
        st["pv_air"] = st["a_pv"] is None
        st["s_max"] = 0.0
        if c["dir"] is not None:
            rel_t = c["dir"] - c["psi"]
            s0, l0 = env_local(mv, pv, c["psi"], c["foot"])
            side = 1.0 if c["foot"] < 0 else -1.0
            st["s_max"] = ray_exit(s0, l0, math.cos(rel_t),
                                   side * math.sin(rel_t))

    # ------------------------------------------------------------ obs ----
    def _vec(self, c: Dict[str, Any], yaw: float,
             bud: float) -> np.ndarray:
        v = np.zeros(6, np.float32)
        v[0] = float(c["foot"])
        if c["dir"] is not None:
            a = c["dir"] - yaw
            v[1], v[2] = math.cos(a), math.sin(a)
        v[3] = _wrap_pi(c["psi"] - yaw)
        v[4] = bud / float(BUDGET_MAX)
        return v

    def get_observation(self) -> Dict[str, Any]:
        base = V2Simulator.get_observation(self)
        if not self._step_on:
            return base
        ck = "_obs_v2bud"
        out = self._data_cache.get(ck)
        if out is None:
            out = {}
            for rid, o in base.items():
                arr = np.asarray(o, np.float32).copy()
                st = self._sst.get(rid)
                if st is not None:
                    _, _, yaw = self._feet_yaw(rid)
                    cur, nxt = st["cur"], st["nxt"]
                    arr[BODY_DIM + _CUR.start:BODY_DIM + _CUR.stop] = \
                        self._vec(cur, yaw, cur["budget"] - st["age_f"])
                    arr[BODY_DIM + _NXT.start:BODY_DIM + _NXT.stop] = \
                        self._vec(nxt, yaw, nxt["budget"])
                out[rid] = arr
            self._data_cache[ck] = out
        return out

    # ------------------------------------------------------- 帧结算 -------
    def physical_step(self) -> None:
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
            done = st["age_f"] >= st["cur"]["budget"]
            st["live"] = self._measure(rid, final=done)
            if done:
                self._settle(rid)
                st["cur"] = st["nxt"]
                self._issue(rid)
                st["nxt"] = self._make(rid)

    def _measure(self, rid: str, final: bool) -> Dict[str, float]:
        st = self._sst[rid]
        c = st["cur"]
        fl, fr, yaw = self._feet_yaw(rid)
        zl, zr = self._foot_z(rid)
        yaw_e = abs(_wrap_pi(c["psi"] - yaw))
        L: Dict[str, float] = {
            "prog": 0.0, "viol": 0.0, "viol_end": 0.0, "lat": 0.0, "pv_dr": 0.0,
            "pv_tol": _TOL_EARLY if st["age_f"] < _EARLY_F else _TOL_LATE,
            "yaw_e": yaw_e, "yaw_w": 1.0, "land": 0.0,
            "mv": float(c["foot"] != 0),
            "zdir": float(c["foot"] != 0 and c["dir"] is None),
        }
        if c["foot"] == 0:
            L["pv_dr"] = float(max(np.linalg.norm(fl - st["a_fl"]),
                                   np.linalg.norm(fr - st["a_fr"])))
            L["pv_tol"] = _TOL_LATE
            return L
        L["yaw_w"] = min(1.0, st["age_f"] / float(c["budget"]))
        mv, pv = (fl, fr) if c["foot"] < 0 else (fr, fl)
        z_mv, z_pv = (zl, zr) if c["foot"] < 0 else (zr, zl)
        # 轴心脚：悬空发令 → 首次落地锚定
        if st["a_pv"] is None and self._grounded(z_pv):
            st["a_pv"] = pv.copy()
        if st["a_pv"] is not None:
            L["pv_dr"] = float(np.linalg.norm(pv - st["a_pv"]))
        # 包络（原点 = 当前轴心脚，目标朝向系）
        s, l = env_local(mv, pv, c["psi"], c["foot"])
        L["viol"] = env_violation(s, l)
        st["viol_max"] = max(st["viol_max"], L["viol"])
        d = mv - st["a_mv"]
        if c["dir"] is not None:
            u = np.array([math.cos(c["dir"]), math.sin(c["dir"])])
            proj = float(d @ u)
            phi = float(np.clip(proj, 0.0, st["s_max"]))
            L["prog"] = phi - st["phi"]
            st["phi"] = phi
            L["lat"] = abs(float(d[0] * u[1] - d[1] * u[0]))
        else:
            L["lat"] = float(np.linalg.norm(d))
        if final:
            L["land"] = float(self._grounded(z_mv))
            L["viol_end"] = L["viol"]
        st["prev_mv"] = mv.copy()
        return L

    def _settle(self, rid: str) -> None:
        st = self._sst[rid]
        c, L = st["cur"], st["live"]
        cmd: BudgetCmd = c["cmd"]
        fl, fr, _ = self._feet_yaw(rid)
        rep: Dict[str, Any] = {
            "event": 1.0, "hold": float(c["foot"] == 0),
            "zdir": L["zdir"], "budget": float(c["budget"]),
            "clip": float(cmd.clip), "rand": float(cmd.rand),
            "rel": float(cmd.rel) if cmd.rel is not None else 0.0,
            "stride": 0.0, "s_max": 0.0, "dir_err": 0.0,
            "lat": L["lat"], "landed": L["land"],
            "viol_max": float(st["viol_max"]), "viol_end": L["viol_end"],
            "pv_dr": L["pv_dr"],
            "yaw_e": L["yaw_e"], "pv_air": 0.0,
        }
        if c["foot"] != 0:
            mv = fl if c["foot"] < 0 else fr
            d = mv - st["a_mv"]
            rep["pv_air"] = float(st["pv_air"])
            if c["dir"] is not None:
                u = np.array([math.cos(c["dir"]), math.sin(c["dir"])])
                rep["stride"] = float(d @ u)
                rep["s_max"] = float(st["s_max"])
                if np.linalg.norm(d) > 0.05:
                    rep["dir_err"] = abs(_wrap_pi(
                        math.atan2(d[1], d[0]) - c["dir"]))
        st["report"] = rep

    # ------------------------------------------------- observer 接口 ------
    def _stepcmd_state(self, rid: str) -> Dict[str, Any]:
        st = self._sst.get(rid)
        if st is None:
            return {"live": None, "report": None}
        return {"live": st["live"], "report": st["report"]}

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        _EXTRA = "stepbudget"
        if fields is None:
            out = V2Simulator.get_derived_state(self)
            out[_EXTRA] = {r: self._stepcmd_state(r) for r in _ROBOTS}
            return out
        fields = list(fields)
        base = [f for f in fields if f != _EXTRA]
        out = (V2Simulator.get_derived_state(self, base) if base else {})
        if _EXTRA in fields:
            out[_EXTRA] = {r: self._stepcmd_state(r) for r in _ROBOTS}
        return out


__all__ = ["V2StepBudgetSimulator"]
