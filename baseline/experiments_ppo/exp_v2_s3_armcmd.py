"""V2 线臂令跟随分支 — 平衡 + 臂令（79 维，D17，S3 血统）。

任务核心：V2ArmCmdSimulator——臂目标指令流照常进 obs（seg[1:7]），
但 act[15:21] 不覆写、归策略全权；跟随压力由两个新稠密通道表达。
本分支是 **S3 的奖励制对照**：任务=推力平衡 + 臂令跟随，
不含 S4 步令任务（step_enabled=False）。

通道（5，全 gamma 0.99）：
    r_potential = 0.01·φ(t)                          aw=3.0
    r_fall      = 摔倒onset罚 + 靠墙罚 + 受推clean赏   aw=2.0
    r_cross     = 支撑轮换计时引导                     aw=2φ²
    r_arm_r     = −w·‖act[15:18]−tgt[0:3]‖  逐帧      aw=0.5
    r_arm_l     = −w·‖act[18:21]−tgt[3:6]‖  逐帧      aw=0.5

臂通道是"可忽略指令"：站稳跟随划算、失衡时策略自行压权重。
arm_override=False 消除了接管方案的 off-policy 污染——执行
动作=策略输出，臂学习信用分配干净（S3 接管是 arm 组步令
学习卡死的首要嫌疑，本实验即其替代验证）。

扰动：躯干推力固定 40N 档最高级——每推 40N × 持续 31-40 帧
（1.55-2.0s）。不接受 100N；目的=检验策略能否在中等扰动下
兼顾平衡与臂令（训练与 eval 同走 build_jobs，eval 指标即
推力条件下的兼顾度）。继承链 BalancePure 的晋级逻辑对单级
表天然失效（_level 恒 0 → 全程同一扰动强度）。

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

from .exp_v2_s3_balance_arm import V2S3BalanceArm

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S3ArmCmd(V2S3BalanceArm):
    name = "v2_s3_armcmd"

    # 扰动封顶：40N 档最高级（原 12 级课程取第 3 级强度固定）。
    LEVEL_FORCES = (40.0,)
    LEVEL_DURATION_RANGES = ((31, 40),)

    _channel_names = V2S3BalanceArm._channel_names + ("r_arm_r", "r_arm_l")
    _channel_gammas = dict(V2S3BalanceArm._channel_gammas,
                           r_arm_r=0.99, r_arm_l=0.99)
    _ARM_OBS = {"robot_a": "arm_follow_a", "robot_b": "arm_follow_b"}

    arm_aw: float = 0.5    # 两臂共享 actor_weight（次要任务）
    w_arm: float = 1.0     # 通道内系数 /norm-unit

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_balance_armcmd_env.yaml")

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


EXPERIMENT_CLASS = V2S3ArmCmd
