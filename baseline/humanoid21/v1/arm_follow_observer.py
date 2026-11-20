"""臂令跟随 observer：把 V2ArmCmdSimulator 的 arm_follow 透传成 per-step
字段（D17，稠密信号——每帧都有值）。

字段：
    err_r = ‖右臂 3 关节输出 − tgt[0:3]‖  (norm 空间 L2)
    err_l = ‖左臂 3 关节输出 − tgt[3:6]‖
    rel_r/l = 各臂当前是否处于释放（不劫持）周期
    flag  = 近期站稳标记（信息位，门控已废）

实验侧装配成 r_arm_r / r_arm_l 两个稠密通道（-w·err 逐帧）。
报告经 get_derived_state(["arm_follow"]) 读取（accessor 白名单
合规通道）。
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

_ZERO: Dict[str, Any] = {"err_r": 0.0, "err_l": 0.0,
                         "rel_r": 0.0, "rel_l": 0.0, "flag": 0.0}


class ArmFollowObserver(BaseObserverPlugin):
    name = "arm_follow"
    priority = 0

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = agent_id
        self._out: Dict[str, Any] = dict(_ZERO)

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._out = dict(_ZERO)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        ds = ctx.accessor.get_derived_state(["arm_follow"])
        rep = (ds.get("arm_follow") or {}).get(self.agent_id)
        self._out = dict(_ZERO) if rep is None else dict(rep)

    def get_output(self) -> Dict[str, Any]:
        return self._out
