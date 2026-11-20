"""手臂软劫持验证实验（ARMSOFT_V1_PLAN.md）— 最小化臂令跟随：79 维。

= v2_s3_armcmd 最小化变体：无推力、flag 恒 1（令恒激活）、
r_std_clip 移除（std_pose 含臂关节与臂令结构性冲突，连同
standard_clip observers 一并剔除）。验证问题单一：
奖励制软指令能否让策略学会"站稳跟随、失衡忽略"？

通道（4，全 gamma 0.99）：
    r_potential = 0.01·φ(t)                    aw=3.0（无 h_cap）
    r_fall      = 摔倒onset罚 + 靠墙罚          aw=2.0
    r_arm_r     = −w·‖act[15:18]−tgt[0:3]‖     aw=0.5
    r_arm_l     = −w·‖act[18:21]−tgt[3:6]‖     aw=0.5

eval 追加核心证据指标：err_stand(φ≥0.9) vs err_down(φ<0.5)——
失衡帧 err 显著更高 = 学会忽略；同步降 = 无脑跟随。

warm-start: --resume-from v2_s2_balance/checkpoints/
checkpoint_u01680.pt --reset-update（r_std_clip critic 自动丢弃，
r_arm_r/l 两个新 critic 冷启）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.rollout import extract_per_step_field
from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

from .exp_v2_s3_armcmd import V2S3ArmCmd

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2ArmSoft(V2S3ArmCmd):
    name = "v2_armsoft"

    # ---------------- 通道 ----------------
    _channel_names = ("r_potential", "r_fall", "r_arm_r", "r_arm_l")
    _channel_gammas = {
        "r_potential": 0.99, "r_fall": 0.99,
        "r_arm_r": 0.99, "r_arm_l": 0.99,
    }

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_armsoft_env.yaml")

    # ---------------- 轨迹 ----------------
    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        foot_key: str,
        phi_key: str,
    ) -> List:
        trajs = super()._build_agent_trajectory(
            episode, agent_id, foot_key, phi_key)
        for tr in trajs:
            # r_std_clip 移除：std_pose 含臂关节与臂令冲突，整通道剔除
            tr.channels.pop("r_std_clip", None)
        return trajs

    # ---------------- eval ----------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        # 核心证据：err 按帧势位分组（φ≥0.9 站稳 vs φ<0.5 失衡）
        es = ed = ns = nd = 0.0
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for rid, _, phi_key in self._AGENT_OBS:
                af_key = self._ARM_OBS.get(rid)
                phi = extract_per_step_field(
                    ep.observer_outputs, phi_key, "potential", T)
                er = extract_per_step_field(
                    ep.observer_outputs, af_key, "err_r", T)
                el = extract_per_step_field(
                    ep.observer_outputs, af_key, "err_l", T)
                if phi is None or er is None or el is None:
                    continue
                phi = np.asarray(phi)
                err = np.asarray(er) + np.asarray(el)
                m_s = phi >= 0.9
                m_d = phi < 0.5
                es += float(err[m_s].sum())
                ns += int(m_s.sum())
                ed += float(err[m_d].sum())
                nd += int(m_d.sum())
        res["info"]["err_stand"] = round(es / (2 * max(ns, 1)), 3)
        res["info"]["err_down"] = round(ed / (2 * max(nd, 1)), 3)
        return res


EXPERIMENT_CLASS = V2ArmSoft
