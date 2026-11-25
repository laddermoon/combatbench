"""STEPBUDHARD V1 — 方向+预算步令 × 双臂独立硬劫持：79 维。

= v2_stepbudget 任务框架 + ARMDUAL 逐臂**硬**劫持
（V2StepBudgetArmHardSimulator）：obs[57:65]=[flag_r,tgt_r,
flag_l,tgt_l]（站立门控下 flag=0 → tgt 全零），被劫臂
act[15:18]/[18:21] 在物理下发前覆写为 tgt——臂输出是死通道，
策略学"看到承诺→用自由维补偿已知扰动"。

通道（7，同 stepbudget，无臂跟随奖励——执行有物理保证）：
    r_potential/r_fall/r_progress/r_lateral/r_land/r_yaw/r_pivot。

eval = stepbudget 全字段 + 硬劫持字段：
    intent_err_r/l  被劫帧 ‖策略输出−tgt‖（策略还想不想管臂）
    occ_dual/r/l    各臂覆写占比（含站立门控）
    err_stand/err_down  φ 分解
    land_1arm/land_2arm / stride_1arm/stride_2arm（令周期平均被劫
        臂数分桶——硬劫持版"双臂同劫是否更拖累迈步/着地"）

warm-start: v2_stepbudget_r2 最新 ckpt --reset-update（与软劫持
stepbudarm 同基座 → 软/硬干净对比）。臂维 obs 由死维变活跃，
步指标预期瞬态回落。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.rollout import extract_per_step_field
from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

from .exp_v2_stepbudget import V2StepBudget

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2StepBudgetArmHard(V2StepBudget):
    name = "v2_stepbudhard"

    _ARM_OBS = {"robot_a": "arm_dual_a", "robot_b": "arm_dual_b"}

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_stepbudhard_env.yaml")

    # ---------------- eval ----------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        info = res["info"]

        sr = sl = nr = nl = 0.0
        n_both = n_r = n_l = n_all = 0.0
        es = ed = ns = nd = 0.0
        l_1 = l_2 = st_1 = st_2 = n_1 = n_2 = 0
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for rid, _, phi_key in self._AGENT_OBS:
                af_key = self._ARM_OBS.get(rid)
                er = extract_per_step_field(
                    ep.observer_outputs, af_key, "err_r", T)
                el = extract_per_step_field(
                    ep.observer_outputs, af_key, "err_l", T)
                hr = extract_per_step_field(
                    ep.observer_outputs, af_key, "hij_r", T)
                hl = extract_per_step_field(
                    ep.observer_outputs, af_key, "hij_l", T)
                phi = extract_per_step_field(
                    ep.observer_outputs, phi_key, "potential", T)
                if (er is None or el is None or hr is None
                        or hl is None):
                    continue
                er = np.asarray(er)
                el = np.asarray(el)
                hr = np.asarray(hr) > 0.5
                hl = np.asarray(hl) > 0.5
                sr += float(er[hr].sum())
                nr += int(hr.sum())
                sl += float(el[hl].sum())
                nl += int(hl.sum())
                n_both += int((hr & hl).sum())
                n_r += int((hr & ~hl).sum())
                n_l += int((hl & ~hr).sum())
                n_all += T
                if phi is not None:
                    phi = np.asarray(phi)
                    hij = hr | hl
                    err = np.where(hr, er, 0.0) + np.where(
                        hl, el, 0.0)
                    m_s = hij & (phi >= 0.85)
                    m_d = hij & (phi < 0.5)
                    es += float(err[m_s].sum())
                    ns += int(m_s.sum())
                    ed += float(err[m_d].sum())
                    nd += int(m_d.sum())

                sr_key = self._STEP_OBS.get(rid)
                ev = extract_per_step_field(
                    ep.observer_outputs, sr_key, "event", T)
                if ev is None:
                    continue
                ev = np.asarray(ev) > 0.5
                idx = np.flatnonzero(ev)
                bud = np.asarray(extract_per_step_field(
                    ep.observer_outputs, sr_key, "budget", T))
                hd = np.asarray(extract_per_step_field(
                    ep.observer_outputs, sr_key, "hold", T)) > 0.5
                ld = np.asarray(extract_per_step_field(
                    ep.observer_outputs, sr_key, "landed", T))
                st_ = np.asarray(extract_per_step_field(
                    ep.observer_outputs, sr_key, "stride", T))
                n_hij = (hr.astype(np.float32)
                         + hl.astype(np.float32))
                for i in idx:
                    if hd[i]:
                        continue
                    lo = max(0, i - int(bud[i]) + 1)
                    nh = float(n_hij[lo:i + 1].mean())
                    if nh >= 1.5:
                        n_2 += 1
                        l_2 += int(ld[i] > 0.5)
                        st_2 += float(st_[i])
                    else:
                        n_1 += 1
                        l_1 += int(ld[i] > 0.5)
                        st_1 += float(st_[i])

        info["intent_err_r"] = round(sr / max(nr, 1), 3)
        info["intent_err_l"] = round(sl / max(nl, 1), 3)
        info["occ_dual"] = round(n_both / max(n_all, 1), 3)
        info["occ_r"] = round(n_r / max(n_all, 1), 3)
        info["occ_l"] = round(n_l / max(n_all, 1), 3)
        info["err_stand"] = round(es / max(ns, 1), 3)
        info["err_down"] = round(ed / max(nd, 1), 3)
        info["land_1arm"] = round(l_1 / max(n_1, 1), 3)
        info["land_2arm"] = round(l_2 / max(n_2, 1), 3)
        info["stride_1arm"] = round(st_1 / max(n_1, 1), 3)
        info["stride_2arm"] = round(st_2 / max(n_2, 1), 3)
        return res


EXPERIMENT_CLASS = V2StepBudgetArmHard
