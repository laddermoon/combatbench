"""V2 线 Stage 2 — 79 维推力课程平衡恢复（双臂可用，不接脚本）。

参照 `hier_s2_balance_s42i_b`（实验 `hier_s2_balance_mbs`）：从 S1
收敛 ckpt 续训，12 级 StandingTriggeredForce 课程（40N→100N）下
保持/恢复 s42i 标准步态。通道/课程/超参全部继承不变：

    r_potential = 0.01×φ  aw=3.0（固定）
    r_fall      = 摔倒+靠墙+受推净窗  aw=2.0（固定）
    r_std_clip  = clip+pose+0.5cad+0.5h_term  aw=2φ²·(非受推中)

差异仅在输入布局（79 维 = 57 本体 + 22 全零指令段，flag=0
策略自控双臂——臂输出用于抗推平衡，本阶段不接脚本源）。

启动（warm-start，v1_skill 同款 --resume-from + --reset-update）：

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 -B \
      baseline/framework/train.py --experiment v2_s2_balance --algo ppo \
      --resume-from baseline/runs/v2_s1_standup_step/checkpoints/checkpoint_u01500.pt \
      --reset-update --run-name v2_s2_balance --background

配套文件（baseline/humanoid21/v1/）：
  v2_simulator.py · v2_balance_env.yaml · init_policy_mbs_79d.yaml
"""
from __future__ import annotations

from pathlib import Path

import torch

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

from .exp_hier_s2_balance_mbs import HierS2BalanceMBS

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S2Balance(HierS2BalanceMBS):
    name = "v2_s2_balance"
    obs_dim = 79

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_balance_env.yaml")

    def build_actor(self, device: torch.device):
        return PolicyBlueprint.load(
            _V1 / "init_policy_mbs_79d.yaml").build().to(device)


EXPERIMENT_CLASS = V2S2Balance
