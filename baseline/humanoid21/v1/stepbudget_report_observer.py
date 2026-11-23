"""STEPBUDGET 观测器：透传 V2StepBudgetSimulator 的 live + 结算报告。

live 字段每帧都有值（本帧活跃令，晋升前量测——奖励对齐 action t）；
报告字段非结算帧全零。字段语义见 v2_stepbudget_simulator 模块注释。
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

LIVE_KEYS = ("prog", "viol", "viol_end", "lat", "pv_dr", "pv_tol", "yaw_e",
             "yaw_w", "land", "mv", "zdir")
REPORT_KEYS = ("event", "hold", "zdir_r", "budget", "clip", "rand",
               "rel", "stride", "s_max", "dir_err", "lat_r", "landed",
               "viol_max", "viol_end_r", "pv_dr_r", "yaw_e_r", "pv_air")
# 报告中与 live 同名的字段加 _r 后缀
_REP_RENAME = {"zdir": "zdir_r", "viol_end": "viol_end_r", "lat": "lat_r", "pv_dr": "pv_dr_r",
               "yaw_e": "yaw_e_r"}


def _zero() -> Dict[str, float]:
    out = {k: 0.0 for k in LIVE_KEYS}
    out.update({k: 0.0 for k in REPORT_KEYS})
    out["clip"] = -2.0
    return out


class StepBudgetReportObserver(BaseObserverPlugin):
    name = "stepbudget_report"
    priority = 0

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = agent_id
        self._out: Dict[str, Any] = _zero()

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._out = _zero()

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        ds = ctx.accessor.get_derived_state(["stepbudget"])
        s = (ds.get("stepbudget") or {}).get(self.agent_id) or {}
        out = _zero()
        live = s.get("live")
        if live:
            out.update({k: float(live[k]) for k in LIVE_KEYS})
        rep = s.get("report")
        if rep:
            for k, v in rep.items():
                kk = _REP_RENAME.get(k, k)
                if kk in out:
                    out[kk] = float(v)
        self._out = out

    def get_output(self) -> Dict[str, Any]:
        return self._out
