"""V2 线 Stage 1 — 79 维（57 本体 + 22 全零指令段）从零起身+标准踱步。

参照 `hier_s1_s42i_scratch`（实验 `hier_s1_standup_step_mbs`）：倒地
→起身→s42i 标准动态踱步单阶段收敛配方，奖励完全不变
（r_potential + r_std_clip·φ² 门控）。差异仅在输入布局：

- 指令槽 9 维 → 22 维（D8：flag|arm_tgt|waist_tgt|step_cur|step_next）
- S1 全段恒零：flag=0=策略自控双臂（起身需要），无脚本/脚步指令
- 仿真器换 V2Simulator（剥对手 39 维的极简实现，无 CommandSimulator
  的 cmd_version/ee_obs 等未用能力）

配套文件（均在 baseline/humanoid21/v1/）：
  v2_simulator.py · v2_standup_step_env.yaml · init_policy_mbs_79d.yaml
"""
from __future__ import annotations

from pathlib import Path

import torch

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

from .exp_hier_s1_standup_step_mbs import HierS1StandupStepMBS

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S1StandupStep(HierS1StandupStepMBS):
    name = "v2_s1_standup_step"
    obs_dim = 79

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_standup_step_env.yaml")

    def build_actor(self, device: torch.device):
        return PolicyBlueprint.load(
            _V1 / "init_policy_mbs_79d.yaml").build().to(device)


EXPERIMENT_CLASS = V2S1StandupStep
