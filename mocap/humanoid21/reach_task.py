"""M4 受控任务：走到场地中心（target-reaching locomotion）。

设计：目标选世界原点——``arena_center_local``（96 维观测 [54:57]）
已提供 ego 系目标向量，**无需扩展观测**。robot_b 远处站立仅作场景。

- ``ReachGuardPlugin``：|xy| < goal_radius → 'reach_done'；
  根高度 < fall_height 或头低于骨盆 → 'reach_fall'。
- ``ReachRewarder`` observer：r_task = 距离缩短（稠密，m/步 × 20）
  + 到达一次性奖励 + 每步小生存项；诊断输出 dist/dt。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import (
    BaseObserverPlugin, BasePlugin, ReadOnlySimContext, SimContext)


class ReachGuardPlugin(BasePlugin):
    def __init__(self, track_robot: str = 'robot_a',
                 goal_radius: float = 0.8,
                 fall_height: float = 0.55) -> None:
        self.track_robot = track_robot
        self.goal_radius = float(goal_radius)
        self.fall_height = float(fall_height)

    @property
    def name(self) -> str:
        return 'reach_guard'

    def to_blueprint(self) -> Dict[str, Any]:
        return {'track_robot': self.track_robot,
                'goal_radius': self.goal_radius,
                'fall_height': self.fall_height}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'ReachGuardPlugin':
        return cls(**config)

    def on_post_action_step(self, ctx: SimContext) -> None:
        core = ctx.accessor.get_core_state()[self.track_robot]
        pos = np.asarray(core['root_pos'])
        if float(pos[2]) < self.fall_height:
            ctx.request_termination('reach_fall')
            return
        if float(np.linalg.norm(pos[:2])) < self.goal_radius:
            ctx.request_termination('reach_done')


class ReachRewarder(BaseObserverPlugin):
    """r_task = w_prog·(dist_prev−dist_now)·20 + w_goal·done + w_alive。

    progress 项按 20Hz 控制帧折算成 "m/s 等效"，量级 ~1；
    done 一次性 +5；alive 每步 +0.02 鼓励存活但不喧宾夺主。
    """

    def __init__(self, agent_id: str = 'robot_a',
                 goal_radius: float = 0.8,
                 w_prog: float = 1.0, w_goal: float = 5.0,
                 w_alive: float = 0.02) -> None:
        self.agent_id = str(agent_id)
        self.goal_radius = float(goal_radius)
        self.w_prog, self.w_goal, self.w_alive = \
            float(w_prog), float(w_goal), float(w_alive)
        self._prev_dist: Optional[float] = None
        self._output: Dict[str, float] = {'reward': 0.0}

    def _dist(self, ctx: ReadOnlySimContext) -> float:
        pos = np.asarray(
            ctx.accessor.get_core_state()[self.agent_id]['root_pos'])
        return float(np.linalg.norm(pos[:2]))

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._prev_dist = self._dist(ctx)
        self._output = {'reward': 0.0, 'dist': self._prev_dist}

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        d = self._dist(ctx)
        prog = (self._prev_dist - d) * 20.0 if self._prev_dist is not None else 0.0
        done = 1.0 if d < self.goal_radius else 0.0
        r = self.w_prog * prog + self.w_goal * done + self.w_alive
        self._output = {'reward': float(r), 'dist': d,
                        'progress': float(prog), 'reached': done}
        self._prev_dist = d

    def get_output(self) -> Dict[str, float]:
        return dict(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {'agent_id': self.agent_id, 'goal_radius': self.goal_radius,
                'w_prog': self.w_prog, 'w_goal': self.w_goal,
                'w_alive': self.w_alive}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'ReachRewarder':
        return cls(**config)
