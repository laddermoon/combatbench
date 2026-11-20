"""V1 步令实验仿真器：0.5s 固定周期连续指令流 + D20 残差编码。

继承 V2Simulator（79 维 obs；arm_script=False → cmd[0:7] 全零，
双臂自控）。每机独立指令流——无站立门控：reset 即施令，失衡
期间令照跑（策略自学"先保命再跟令"，STEPCMD_V1_PLAN §2）。

    reset:  采 cur + nxt 两条令，cur 立即按当前位姿锚定世界系目标
    每 action step 末物理子步（obs 帧对齐点）:
        age_f += 1
        首达判定：‖mv‖<arrive_tol_m 且 |Δψ|<arrive_tol_rad → t_arrive
        └─ age_f ≥ period → 到期结算 → cur←nxt（按实际位姿重锚，
           D20 晋升语义）→ nxt=sample_cmd（撞墙检查基于预测后态 P̂）

obs 指令段（残差每帧由真值重算，STEP_CMD.md）：
    step_cur  = [foot, Δx, Δy, Δψ, 0, 0]    obs[67:73]
    step_next = [foot, Δx, Δy, Δψ, 0, 0]    obs[73:79]
                next 残差在 cur 的预测完成态 P̂ 系内表达（静态）
    保持令(foot=0)：Δx/Δy=0，Δψ=当前 yaw 对发令朝向的漂移
                （"回到发令朝向"仍是有效伺服残差）

stepcmd 结算报告（到期帧一次性产出）+ live 字段（每帧）：
    live : res_st = stance 脚世界系漂移 m（保持令=双脚之大者）
    event: hold/cmd_foot/d_mv/d_st/d_yaw/t_arrive/age/order

observer 经 get_derived_state(["stepcmd"]) 取 {rid: {...}}。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .v2_simulator import BODY_DIM, V2Simulator
from .arena_geom import inside_arena
from .step_cmd_gen import StepCmd, cmd_target, residual, sample_cmd

_ROBOTS = ("robot_a", "robot_b")
_CUR = slice(10, 16)     # seg 内偏移：step_cur → obs[67:73]
_NXT = slice(16, 22)     # seg 内偏移：step_next → obs[73:79]
_FOOT_BODY = {-1: "foot_left", +1: "foot_right"}


def _quat_yaw(q_wxyz: np.ndarray) -> float:
    """wxyz quat → 偏航角（躯干 heading，+x 前）。"""
    w, x, y, z = (float(v) for v in q_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class V2StepCmdSimulator(V2Simulator):
    """V2Simulator + 每机 0.5s 周期连续步令流（D20 残差编码）。"""

    def __init__(self, *args: Any,
                 step_enabled: bool = True,
                 order_period_s: float = 0.5,
                 arrive_tol_m: float = 0.10,
                 arrive_tol_rad: float = 0.10,
                 wall_margin: float = 0.30,
                 steps_per_action: int = 25,
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._step_on = bool(step_enabled)
        self._wall_margin = float(wall_margin)
        self._tol_m = float(arrive_tol_m)
        self._tol_r = float(arrive_tol_rad)
        # 与 yaml runtime.phy_steps_per_action 一致——结算挂在
        # 每 action step 最后一个物理子步，到期帧即 obs 量测点
        self._spa = int(steps_per_action)
        dt_action = float(self.model.opt.timestep) * self._spa
        self._period_f = max(1, int(round(order_period_s / dt_action)))
        self._phys_sub = 0
        # 每机 {rng,last_foot,last_dyaw, cur,nxt, tgt_mv,yaw_t,
        #       a_fl,a_fr,a_yaw, nxt_vec, age_f,t_arrive, report}
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
            rng = np.random.default_rng([base, 0xD4 + i])
            st = {"rng": rng, "last_foot": 0, "last_dyaw": 0.0,
                  "report": None}
            self._sst[rid] = st
            st["cur"] = self._sample(rid, use_pred=False)
            self._issue(rid)
            st["nxt"] = self._sample(rid, use_pred=True)
            self._write_nxt_vec(rid)

    # ---------------------------------------------------------- 指令段 -----
    def get_observation(self) -> Dict[str, Any]:
        base = super().get_observation()
        if not self._step_on:
            return base
        ck = "_obs_v2cmd"
        out = self._data_cache.get(ck)
        if out is None:
            out = {}
            for rid, o in base.items():
                arr = np.asarray(o, np.float32).copy()
                st = self._sst.get(rid)
                if st is not None:
                    arr[BODY_DIM + _CUR.start:BODY_DIM + _CUR.stop] = \
                        self._cur_vec(rid)
                    arr[BODY_DIM + _NXT.start:BODY_DIM + _NXT.stop] = \
                        st["nxt_vec"]
                out[rid] = arr
            self._data_cache[ck] = out
        return out

    # -------------------------------------------------------- 结算/轮转 ---
    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        super().set_action(action)
        self._phys_sub = 0

    def physical_step(self) -> None:
        """物理子步推进；末子步做帧结算（与部署轮转同帧对齐）。"""
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
            self._check_arrive(rid)
            if st["age_f"] >= self._period_f:
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

    def _mv_pos(self, st: Dict[str, Any],
                fl: np.ndarray, fr: np.ndarray) -> np.ndarray:
        """动脚踝当前世界位（保持令返回双脚中点——无动脚）。"""
        c = st["cur"]
        if c.foot < 0:
            return fl
        if c.foot > 0:
            return fr
        return 0.5 * (fl + fr)

    def _post_state(self, st: Dict[str, Any]) -> Tuple[
            np.ndarray, np.ndarray, float]:
        """cur 的预测完成态 P̂：动脚→目标位，stance/无令脚→锚位，yaw→yaw_t。"""
        c = st["cur"]
        if c.foot < 0:
            pfl, pfr = st["tgt_mv"], st["a_fr"]
        elif c.foot > 0:
            pfl, pfr = st["a_fl"], st["tgt_mv"]
        else:
            pfl, pfr = st["a_fl"], st["a_fr"]
        return pfl, pfr, st["yaw_t"]

    def _issue(self, rid: str) -> None:
        """cur 令生效：按实际位姿锚定世界目标。"""
        st = self._sst[rid]
        fl, fr, yaw = self._feet_yaw(rid)
        tgt_mv, yaw_t = cmd_target(fl, fr, yaw, st["cur"])
        st["a_fl"], st["a_fr"], st["a_yaw"] = fl, fr, yaw
        st["tgt_mv"], st["yaw_t"] = tgt_mv, yaw_t
        st["age_f"] = 0
        st["t_arrive"] = -1
        self._check_arrive(rid)          # 帧 0 已到位（近零步）也记

    def _write_nxt_vec(self, rid: str) -> None:
        """nxt 预览：残差表达在 P̂ 系内（cur 存活期静态，D20）。

        必须在新 nxt 采样之后调用（_rotate 中顺序：晋升→issue→
        采样新 nxt→写预览）。"""
        st = self._sst[rid]
        pfl, pfr, pyaw = self._post_state(st)
        nxt = st["nxt"]
        vec = np.zeros(6, np.float32)
        vec[0] = float(nxt.foot)
        if nxt.foot != 0:
            n_tgt, n_yt = cmd_target(pfl, pfr, pyaw, nxt)
            n_ankle = pfl if nxt.foot < 0 else pfr
            vec[1], vec[2], vec[3] = residual(n_ankle, pyaw, n_tgt, n_yt)
        st["nxt_vec"] = vec

    def _cur_vec(self, rid: str) -> np.ndarray:
        """obs[67:73]：本帧残差 [foot, Δx, Δy, Δψ, 0, 0]。"""
        st = self._sst[rid]
        fl, fr, yaw = self._feet_yaw(rid)
        c = st["cur"]
        vec = np.zeros(6, np.float32)
        vec[0] = float(c.foot)
        if c.foot != 0:
            ankle = fl if c.foot < 0 else fr
            vec[1], vec[2], vec[3] = residual(
                ankle, yaw, st["tgt_mv"], st["yaw_t"])
        else:
            # 保持令：Δψ = 对发令朝向的漂移（xy 无动脚目标）
            vec[3] = np.float32(_wrap_pi(st["a_yaw"] - yaw))
        return vec

    def _collide(self, rid: str, cmd: StepCmd,
                 base: Tuple[np.ndarray, np.ndarray, float]) -> bool:
        """cmd 落点出界判定；base = 参考系（cur 采样用实测，nxt 用 P̂）。"""
        if cmd.foot == 0:
            return False
        fl, fr, yaw = base
        tgt, _ = cmd_target(fl, fr, yaw, cmd)
        return not inside_arena(tgt, self._wall_margin)

    def _sample(self, rid: str, use_pred: bool) -> StepCmd:
        """采一条令；use_pred=True → 撞墙检查基于 cur 预测后态 P̂。"""
        st = self._sst[rid]
        if use_pred and "tgt_mv" in st:
            base = self._post_state(st)
        else:
            base = self._feet_yaw(rid)
        cmd = sample_cmd(
            st["rng"], st["last_foot"], st["last_dyaw"],
            collide_fn=lambda c: self._collide(rid, c, base))
        if cmd.foot != 0:
            st["last_foot"] = cmd.foot
        st["last_dyaw"] = cmd.dyaw
        return cmd

    def _check_arrive(self, rid: str) -> None:
        st = self._sst[rid]
        if st["t_arrive"] >= 0:
            return
        c = st["cur"]
        if c.foot == 0:
            st["t_arrive"] = 0          # 保持令无行程，恒已到位
            return
        fl, fr, yaw = self._feet_yaw(rid)
        ankle = fl if c.foot < 0 else fr
        d_mv = float(np.linalg.norm(ankle - st["tgt_mv"]))
        d_yaw = abs(_wrap_pi(st["yaw_t"] - yaw))
        if d_mv < self._tol_m and d_yaw < self._tol_r:
            st["t_arrive"] = st["age_f"]

    def _rotate(self, rid: str) -> None:
        st = self._sst[rid]
        st["cur"] = st["nxt"]
        self._issue(rid)                # 晋升按实际位姿重锚（D20）
        st["nxt"] = self._sample(rid, use_pred=True)
        self._write_nxt_vec(rid)

    def _settle(self, rid: str) -> None:
        """到期帧量测：动脚偏差 / stance 漂移 / yaw 偏差 → 报告。"""
        st = self._sst[rid]
        fl, fr, yaw = self._feet_yaw(rid)
        c = st["cur"]
        dl = float(np.linalg.norm(fl - st["a_fl"]))
        dr = float(np.linalg.norm(fr - st["a_fr"]))
        if c.foot != 0:
            mv_pos = fl if c.foot < 0 else fr
            d_mv = float(np.linalg.norm(mv_pos - st["tgt_mv"]))
            d_st = dr if c.foot < 0 else dl        # stance=非指令脚
        else:
            d_mv = 0.0
            d_st = max(dl, dr)                     # 保持令双脚皆 stance
        st["report"] = {
            "event": 1.0,
            "hold": float(c.foot == 0),
            "cmd_foot": float(c.foot),
            "d_mv": d_mv,
            "d_st": d_st,
            "d_yaw": float(abs(_wrap_pi(yaw - st["yaw_t"]))),
            "t_arrive": float(st["t_arrive"]),
            "age": float(st["age_f"]),
            "order": np.array([c.foot, c.sag, c.lat, c.dyaw],
                              np.float32),
        }

    # ------------------------------------------------- observer 接口 ------
    def _stepcmd_state(self, rid: str) -> Dict[str, Any]:
        """live 字段（每帧）+ 边界报告（到期帧）。"""
        st = self._sst.get(rid)
        out: Dict[str, Any] = {"res_st": 0.0, "report": None}
        if st is None:
            return out
        fl, fr, _ = self._feet_yaw(rid)
        c = st["cur"]
        if c.foot < 0:
            out["res_st"] = float(np.linalg.norm(fr - st["a_fr"]))
        elif c.foot > 0:
            out["res_st"] = float(np.linalg.norm(fl - st["a_fl"]))
        else:
            out["res_st"] = float(max(
                np.linalg.norm(fl - st["a_fl"]),
                np.linalg.norm(fr - st["a_fr"])))
        out["report"] = st["report"]
        return out

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        """扩 DATASPEC：stepcmd = {rid: {res_st, report}}。"""
        _EXTRA = "stepcmd"
        if fields is None:
            out = super().get_derived_state()
            out[_EXTRA] = {r: self._stepcmd_state(r) for r in _ROBOTS}
            return out
        fields = list(fields)
        base = [f for f in fields if f != _EXTRA]
        out = (super().get_derived_state(base) if base else {})
        if _EXTRA in fields:
            out[_EXTRA] = {r: self._stepcmd_state(r) for r in _ROBOTS}
        return out


__all__ = ["V2StepCmdSimulator"]
