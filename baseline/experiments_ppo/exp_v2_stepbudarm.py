"""STEPBUDARM V1 — 方向+预算步令 × 双臂独立软劫持：79 维。

= v2_stepbudget 任务框架 + ARMDUAL 逐臂劫持机制（V2StepBudgetArmSimulator）：
obs[57:65]=[flag_r,tgt_r,flag_l,tgt_l]（flag=0 臂 tgt 全零），
obs[67:79] 方向+预算步令不变。两套指令流独立计时
（步令 budget 3–20f / 臂令 U(2,20)f）。

通道（9，全 gamma 0.99）：stepbudget 原 7 通道 +
    r_arm_r/l·0.7（被劫帧 −err；臂总权重 1.4，与 steparm 一致）。

eval = stepbudget 全字段 + armdual 臂字段（arm_err_r/l 被劫帧条件
均值、err_stand/err_down φ 分解、occ_dual/r/l）+ 交叉指标
land_1arm/land_2arm / stride_1arm/stride_2arm（按令周期实际被劫
臂数分桶——回答"双臂同劫是否更拖累迈步/着地"；恒≥1臂故无
arm-free 桶）。

warm-start: v2_stepbudget_r2 最新 ckpt --reset-update
（r_arm_r/l 两 critic 冷启；臂令段由死维变活跃 → 步令指标
预期有域迁移瞬态回落）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

from .exp_v2_stepbudget import V2StepBudget

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2StepBudgetArm(V2StepBudget):
    name = "v2_stepbudarm"

    _channel_names = ("r_potential", "r_fall", "r_progress",
                      "r_lateral", "r_land", "r_yaw", "r_pivot",
                      "r_arm_r", "r_arm_l")
    _channel_gammas = {k: 0.99 for k in _channel_names}
    _ARM_OBS = {"robot_a": "arm_dual_a", "robot_b": "arm_dual_b"}

    arm_aw: float = 0.7   # 臂总 1.4（与 steparm 一致）

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_stepbudarm_env.yaml")

    # ---------------- 轨迹 ----------------
    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        foot_key: str,
        phi_key: str,
    ) -> List[Trajectory]:
        trajs = super()._build_agent_trajectory(
            episode, agent_id, foot_key, phi_key)
        if not trajs:
            return trajs
        tr = trajs[0]
        T = tr.obs.shape[0]
        af_key = self._ARM_OBS.get(agent_id)
        er = extract_per_step_field(
            episode.observer_outputs, af_key, "err_r", T)
        el = extract_per_step_field(
            episode.observer_outputs, af_key, "err_l", T)
        r_arm_r = (-np.asarray(er[:T], dtype=np.float32)
                   if er is not None else np.zeros(T, np.float32))
        r_arm_l = (-np.asarray(el[:T], dtype=np.float32)
                   if el is not None else np.zeros(T, np.float32))
        tr.channels["r_arm_r"] = ChannelData(
            reward=r_arm_r.astype(np.float32), is_terminated=False,
            actor_weight=np.full(T, self.arm_aw, np.float32))
        tr.channels["r_arm_l"] = ChannelData(
            reward=r_arm_l.astype(np.float32), is_terminated=False,
            actor_weight=np.full(T, self.arm_aw, np.float32))
        return trajs

    # ---------------- eval ----------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        info = res["info"]

        sr = sl = nr = nl = 0.0
        n_both = n_r = n_l = n_all = 0.0
        es = ed = ns = nd = 0.0
        # 交叉指标：步令结算事件 × 该令实际周期平均被劫臂数
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
                    # h_cap=1.10 → φ_max≈0.889：站稳门取 0.85
                    m_s = hij & (phi >= 0.85)
                    m_d = hij & (phi < 0.5)
                    es += float(err[m_s].sum())
                    ns += int(m_s.sum())
                    ed += float(err[m_d].sum())
                    nd += int(m_d.sum())

                # 交叉：结算帧 event=1 → 该令实际生存窗 = [i-bud+1, i]
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
                    if hd[i]:                      # 保持令不计
                        continue
                    lo = max(0, i - int(bud[i]) + 1)
                    nh = float(n_hij[lo:i + 1].mean())
                    if nh >= 1.5:                  # 双臂主导期
                        n_2 += 1
                        l_2 += int(ld[i] > 0.5)
                        st_2 += float(st_[i])
                    else:                          # 单臂主导期
                        n_1 += 1
                        l_1 += int(ld[i] > 0.5)
                        st_1 += float(st_[i])

        info["arm_err_r"] = round(sr / max(nr, 1), 3)
        info["arm_err_l"] = round(sl / max(nl, 1), 3)
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


EXPERIMENT_CLASS = V2StepBudgetArm
