"""V2 线 S4 分支 — 臂令跟随 + 步令跟随（79 维，D16/D17）。

任务核心：V2ArmCmdSimulator——臂目标指令流照常进 obs（seg[1:7]），
但 act[15:21] 不覆写、归策略全权；跟随压力由两个新稠密通道表达。
步令部分与 v2_s4_step 完全一致（时限预算到期结算）。

通道（8，全 gamma 0.99）：
    r_potential = 0.01·φ(t)                          aw=3.0
    r_fall      = 摔倒onset罚 + 持续靠墙罚            aw=2.0
    r_step_foot = 到期帧: d_mv<0.10 → +1              aw=1.5
    r_step_mv   = 到期帧: −w_mv·‖指令脚位−目标‖        aw=2.0
    r_step_st   = 到期帧: −w_st·‖stance脚−发令位‖      aw=2.0
    r_step_yaw  = 到期帧: −w_yaw·|yaw−目标yaw|         aw=2.0
    r_arm_r     = −w·‖act[15:18]−tgt[0:3]‖  逐帧      aw=0.5
    r_arm_l     = −w·‖act[18:21]−tgt[3:6]‖  逐帧      aw=0.5

臂通道是"可忽略指令"：站稳跟随划算、失衡时策略自行压权重
（与步令同构）。arm_override=False 消除了接管方案的
off-policy 污染——执行动作=策略输出，臂学习信用分配干净。

warm-start: --resume-from v2_s2_balance/checkpoints/
checkpoint_u01680.pt --reset-update（S2 best，无臂接管史）
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

from .exp_v2_s4_step import V2S4Step

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S4ArmCmd(V2S4Step):
    name = "v2_s4_armcmd"

    _channel_names = V2S4Step._channel_names + ("r_arm_r", "r_arm_l")
    _channel_gammas = dict(V2S4Step._channel_gammas,
                           r_arm_r=0.99, r_arm_l=0.99)
    _ARM_OBS = {"robot_a": "arm_follow_a", "robot_b": "arm_follow_b"}

    arm_aw: float = 0.5    # 两臂共享 actor_weight（次要任务）
    w_arm: float = 1.0     # 通道内系数 /norm-unit

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_step_armcmd_env.yaml")

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
        af_key = self._ARM_OBS.get(agent_id)
        T_full = episode.num_frames
        for tr in trajs:
            T = tr.obs.shape[0]

            def _fld(name: str) -> np.ndarray:
                if af_key is None:
                    return np.zeros(T, dtype=np.float32)
                a = extract_per_step_field(
                    episode.observer_outputs, af_key, name, T_full)
                if a is None:
                    return np.zeros(T, dtype=np.float32)
                return np.asarray(a[:T], dtype=np.float32)

            for ch, fld in (("r_arm_r", "err_r"), ("r_arm_l", "err_l")):
                tr.channels[ch] = ChannelData(
                    reward=(-self.w_arm * _fld(fld)).astype(np.float32),
                    is_terminated=False,
                    actor_weight=np.full(T, self.arm_aw, np.float32))
        return trajs

    # ---------------- eval ----------------
    def on_eval(self, episodes, update) -> Dict[str, Any]:
        res = super().on_eval(episodes, update)
        sum_r = sum_l = n_fr = 0.0
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            for af_key in self._ARM_OBS.values():
                for fld, acc in (("err_r", "r"), ("err_l", "l")):
                    a = extract_per_step_field(
                        ep.observer_outputs, af_key, fld, T)
                    if a is None:
                        continue
                    if acc == "r":
                        sum_r += float(np.sum(np.asarray(a)))
                    else:
                        sum_l += float(np.sum(np.asarray(a)))
                n_fr += T
        res["info"]["arm_err_r"] = round(sum_r / max(n_fr, 1), 3)
        res["info"]["arm_err_l"] = round(sum_l / max(n_fr, 1), 3)
        return res


EXPERIMENT_CLASS = V2S4ArmCmd
