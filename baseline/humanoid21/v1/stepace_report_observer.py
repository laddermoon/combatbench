"""STEPACE 验收结算 observer：steptvar 字段 + trunc。

trunc=1 表示该令被门控（落地/首达）提前晋升而非到期——
区分"被成功衔接"与"被超时踢出"。
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

_ZERO: Dict[str, Any] = {
    "res_st": 0.0,
    "event": 0.0, "hold": 0.0, "cmd_foot": 0.0,
    "d_mv": 0.0, "d_st": 0.0, "d_yaw": 0.0,
    "t_arrive": 0.0, "age": 0.0,
    "dur": 0.0, "res0": 0.0, "st_z": 0.0, "trunc": 0.0,
}


class StepAceReportObserver(BaseObserverPlugin):
    name = "stepace_report"
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
