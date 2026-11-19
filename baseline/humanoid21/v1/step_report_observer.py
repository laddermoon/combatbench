"""步令结算 observer：把 V2StepSimulator 的 step_report 透传成 per-step 字段。

非边界帧全零；边界帧字段：
    event      = 1
    foot_ok    = 迈对脚 (executed==cmd_foot)
    timed_out  = 超时判负
    executed   = 实际落地脚 (-1/0/+1；0=无落地)
    cmd_foot   = 指令脚
    d_mv       = 执行脚位↔目标脚位 距离 m
    d_st       = 另一脚↔发令原位 距离 m
    d_yaw      = yaw↔目标yaw 差 rad

实验侧把这些字段直接装配成 r_step_foot / r_step_mv / r_step_st /
r_step_yaw 四个稀疏通道。报告经 get_derived_state(["step_reports"])
读取（accessor 白名单合规通道）。
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

_ZERO: Dict[str, Any] = {
    "event": 0.0, "foot_ok": 0.0, "timed_out": 0.0,
    "executed": 0.0, "cmd_foot": 0.0,
    "d_mv": 0.0, "d_st": 0.0, "d_yaw": 0.0, "age": 0.0,
}


class StepReportObserver(BaseObserverPlugin):
    name = "step_report"
    priority = 0

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = agent_id
        self._out: Dict[str, Any] = dict(_ZERO)

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._out = dict(_ZERO)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        ds = ctx.accessor.get_derived_state(["step_reports"])
        rep = (ds.get("step_reports") or {}).get(self.agent_id)
        self._out = dict(_ZERO) if rep is None else dict(rep)
        self._out.pop("order", None)

    def get_output(self) -> Dict[str, Any]:
        return self._out
