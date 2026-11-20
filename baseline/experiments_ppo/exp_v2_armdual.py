"""双臂独立软劫持实验（ARMDUAL_V1_PLAN.md）— 逐臂独立令：79 维。

= v2_armsoft 同奖励框架，指令流换 V2ArmDualSimulator：
两臂同步周期边界，P(0.1) 双臂同劫 / P(0.9) 单臂（左右 50/50），
被劫臂 tgt~U(-1,1)³、自由臂 flag=0+tgt=0+err 不计分。
obs[57:65] = [flag_r,tgt_r(3),flag_l,tgt_l(3)]；一臂跟令时
另一臂可作平衡器——本实验验证的正是该自由度红利。

通道（4，全 gamma 0.99）：r_potential·3 + r_fall·2 +
    r_arm_r/l·0.5（被劫帧 −‖act−tgt‖，自由帧 0）。

eval：arm_err_r/l 为**被劫帧条件均值**（hij 掩码归一）；
occ_dual/occ_r/occ_l 占用率诊断（名义 0.1/0.45/0.45）；
err_stand/err_down 同 armsoft 定义但限于被劫帧。

warm-start: --resume-from v2_s2_balance/checkpoints/
checkpoint_u01680.pt --reset-update（r_arm_r/l 两 critic 冷启）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from baseline.framework.rollout import extract_per_step_field
from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

from .exp_v2_armsoft import V2ArmSoft

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2ArmDual(V2ArmSoft):
    name = "v2_armdual"

    # 观测键改指 arm_dual observers（err_r/l 字段同名兼容）
    _ARM_OBS = {"robot_a": "arm_dual_a", "robot_b": "arm_dual_b"}

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_armdual_env.yaml")

    # ---------------- eval ----------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        info = res["info"]
        # hij 条件归一：被劫帧均值 err + 占用率 + 劫持帧 err_stand/down
        sr = sl = nr = nl = 0.0
        n_both = n_r = n_l = n_all = 0.0
        es = ed = ns = nd = 0.0
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
                # 被劫帧 err_stand/down（φ≥0.9 / φ<0.5）
                if phi is not None:
                    phi = np.asarray(phi)
                    hij = hr | hl
                    err = np.where(hr, er, 0.0) + np.where(
                        hl, el, 0.0)
                    m_s = hij & (phi >= 0.9)
                    m_d = hij & (phi < 0.5)
                    es += float(err[m_s].sum())
                    ns += int(m_s.sum())
                    ed += float(err[m_d].sum())
                    nd += int(m_d.sum())

        info["arm_err_r"] = round(sr / max(nr, 1), 3)
        info["arm_err_l"] = round(sl / max(nl, 1), 3)
        info["occ_dual"] = round(n_both / max(n_all, 1), 3)
        info["occ_r"] = round(n_r / max(n_all, 1), 3)
        info["occ_l"] = round(n_l / max(n_all, 1), 3)
        info["err_stand"] = round(es / max(ns, 1), 3)
        info["err_down"] = round(ed / max(nd, 1), 3)
        return res


EXPERIMENT_CLASS = V2ArmDual
