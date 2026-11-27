"""V2 线 Stage 1 [target_beta=1.0 变体] — 79 维从零起身+标准踱步。

与 `v2_s1_standup_step` 的唯一自变量: 仿真器构造参数 `target_beta=1.0`
(PD 目标范围扩展, 关节到机械限位时仍能输出满力矩; 动作接口仍为
[-1,1], 内部按 alpha=1+beta*gear/(KP*scale) 逐关节扩展目标范围)。
奖励/网络/观测/插件全部继承不变, 构成干净对照。

动机: legacy 映射下 action=±1 对应的目标位置就是机械限位本身,
关节一旦顶到限位 PD 误差为零, ctrl 输出 0.0000 完全失去发力能力;
起身过程中膝/踝/髋频繁顶限位, 可能是 simchar/legacy 系实验
站不起来的因素之一。
"""
from __future__ import annotations

from pathlib import Path

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint

from .exp_v2_s1_standup_step import V2S1StandupStep

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S1StandupStepBeta1(V2S1StandupStep):
    name = "v2_s1_standup_step_beta1"

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_standup_step_env_beta1.yaml")


EXPERIMENT_CLASS = V2S1StandupStepBeta1
