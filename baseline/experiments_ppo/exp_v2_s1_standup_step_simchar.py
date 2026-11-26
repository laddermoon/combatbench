"""V2 线 Stage 1 — SimChar 对齐执行器版本（79 维，从零起身+标准踱步）。

与 `v2_s1_standup_step` 的唯一差异：环境 yaml 换用
`v2_standup_step_env_simchar.yaml`，仿真器以
`actuation_profile="simchar"` 构建（扭矩/KP/KD 全系统对齐 DeepMimic
humanoid3d，PD 目标范围按饱和角公式扩展，设计依据见
envs/humanoid21/ACTUATION_DESIGN.md）。

奖励、网络、观测、插件配置完全不变——本实验是与 legacy 执行器
配方的严格对照组：唯一自变量是执行器配置。

注意：simchar profile 下 alpha∈[1.14, 2.04]，action=±1 对应的 PD 目标
超出机械限位 e_sat = gear/KP，关节到限位时仍可输出满力矩；
观测归一化不变（仍按机械限位）。
"""
from __future__ import annotations

from pathlib import Path

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint

from .exp_v2_s1_standup_step import V2S1StandupStep

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S1StandupStepSimchar(V2S1StandupStep):
    name = "v2_s1_standup_step_simchar"

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_standup_step_env_simchar.yaml")


EXPERIMENT_CLASS = V2S1StandupStepSimchar
