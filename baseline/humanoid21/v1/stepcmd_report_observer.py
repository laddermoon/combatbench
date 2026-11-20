"""V1 步令结算 observer：透传 V2StepCmdSimulator 的 stepcmd 状态。

live 字段（每帧都有值）：
    res_st   = stance 脚世界系漂移 m（保持令=双脚之大者，无死区——
               死区在实验通道侧施加，便于调阈值不动 sim）

边界字段（非到期帧全零；到期帧一次性产出）：
    event    = 1
    hold     = 本令为保持令
    cmd_foot = 指令脚 (−1/0/+1)
    d_mv     = 动脚位↔目标位 距离 m（保持令恒 0）
    d_st     = stance脚↔发令锚 距离 m（保持令=双脚之大者）
    d_yaw    = |yaw−目标yaw| rad（保持令目标=发令朝向）
    t_arrive = 首达帧号（mv<0.10m 且 |Δψ|<0.10rad；−1=未达；
               保持令恒 0）
    age      = 令龄（帧，恒=period）

实验侧：res_st → r_step_st（死区积分）；mv/yaw 残差直接读 obs 列
（obs[68:71] = Δx,Δy,Δψ——obs 显示量=被结算量）；边界字段进 eval。
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

_ZERO: Dict[str, Any] = {
    "res_st": 0.0,
    "event": 0.0, "hold": 0.0, "cmd_foot": 0.0,
    "d_mv": 0.0, "d_st": 0.0, "d_yaw": 0.0,
    "t_arrive": 0.0, "age": 0.0,
}


class StepCmdReportObserver(BaseObserverPlugin):
    name = "stepcmd_report"
    priority = 0

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = agent_id
        self._out: Dict[str, Any] = dict(_ZERO)

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._out = dict(_ZERO)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        ds = ctx.accessor.get_derived_state(["stepcmd"])
        s = (ds.get("stepcmd") or {}).get(self.agent_id) or {}
        out = dict(_ZERO)
        out["res_st"] = float(s.get("res_st", 0.0))
        rep = s.get("report")
        if rep is not None:
            out.update({k: float(v) for k, v in rep.items()
                        if k in out})
        self._out = out

    def get_output(self) -> Dict[str, Any]:
        return self._out
