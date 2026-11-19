"""V2 线 Stage 4 ablation — 步令跟随 · 无臂脚本接管（79 维）。

与 v2_s4_step 唯一差异：env 换 v2_step_env_noarm.yaml
（arm_script: false）。通道/aw/eval 全部继承——纯消融，
臂接管是否干扰步令学习。

warm-start: --resume-from v2_s2_balance/checkpoints/
checkpoint_u01680.pt --reset-update
（S2 best：level 11 + recovery 0.901，未经臂扰动训练；
r_std_clip critic 不在本通道集自动冷启丢弃。）
"""
from __future__ import annotations

from pathlib import Path

from envs.framework.parameterized_blueprint import (
    ParameterizedEnvBlueprint,
)

from .exp_v2_s4_step import V2S4Step

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S4StepNoArm(V2S4Step):
    name = "v2_s4_step_noarm"

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_step_env_noarm.yaml")


EXPERIMENT_CLASS = V2S4StepNoArm
