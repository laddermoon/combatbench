"""双臂独立劫持 observer：透传 V2ArmDualSimulator 的 arm_dual 状态。

字段（每帧都有值——稠密）：
    err_r/l = 被劫臂 ‖act−tgt‖（自由臂恒 0，不计分）
    hij_r/l = err 结算时刻的劫持快照（eval 归一化分母）
    flag_r/l = 当前帧劫持标记（live，诊断用）

实验侧装配 r_arm_r/l = −err×flag（自由帧 err=0 即不计分）；
eval 用 hij 掩码对被劫帧做条件归一。
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

_ZERO: Dict[str, Any] = {
    "err_r": 0.0, "err_l": 0.0,
    "hij_r": 0.0, "hij_l": 0.0,
    "flag_r": 0.0, "flag_l": 0.0,
}


class ArmDualObserver(BaseObserverPlugin):
    name = "arm_dual"
    priority = 0

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = agent_id
        self._out: Dict[str, Any] = dict(_ZERO)

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._out = dict(_ZERO)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        ds = ctx.accessor.get_derived_state(["arm_dual"])
        rep = (ds.get("arm_dual") or {}).get(self.agent_id)
        self._out = dict(_ZERO) if rep is None else dict(rep)

    def get_output(self) -> Dict[str, Any]:
        return self._out
