"""STEPTVAR V1 结算 observer：stepcmd_report 同构 + 随机时长诊断。

live 字段：res_st（同 stepcmd_report）。
边界字段：stepcmd 全部字段 + STEPTVAR 新增：
    dur   = 本令时长（帧，U(1,20)）
    res0  = 发令时刻动脚初始残差 m（保持令=0）
    st_z  = 发令时刻轴心脚 z 高度 m（保持令=双脚较大者；
            >0.20 ≈ 半空发令）
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

_ZERO: Dict[str, Any] = {
    "res_st": 0.0,
    "event": 0.0, "hold": 0.0, "cmd_foot": 0.0,
    "d_mv": 0.0, "d_st": 0.0, "d_yaw": 0.0,
    "t_arrive": 0.0, "age": 0.0,
    "dur": 0.0, "res0": 0.0, "st_z": 0.0,
}


class StepTvarReportObserver(BaseObserverPlugin):
    name = "steptvar_report"
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
