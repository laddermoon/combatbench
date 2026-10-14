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
from scipy.spatial.transform import Rotation as R

from envs.framework import (
    BaseObserverPlugin, BasePlugin, ReadOnlySimContext, SimContext)


class RotateSpawnPlugin(BasePlugin):
    """绕场地中心把 robot_a 的出生点旋转到随机角度（面朝中心不变）。

    原生出生构型 A 固定在 -x 侧面向 +x；对 pos 与 yaw 同时施加
    R_z(θ) 后仍在同一半径上且恰好面朝中心——任务 = "走到中心"，
    方向随机化防止策略把 "x 轴正方向" 背下来。held-out 角度即天然
    泛化测试条件。
    """

    def __init__(self, target_robot: str = 'robot_a',
                 random_seed: Optional[int] = None) -> None:
        self.target_robot = target_robot
        self._rng = np.random.RandomState(random_seed)
        self.angle_deg = 0.0

    @property
    def name(self) -> str:
        return 'rotate_spawn'

    @property
    def require_mutator(self) -> bool:
        return True

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    def to_blueprint(self) -> Dict[str, Any]:
        return {'target_robot': self.target_robot}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'RotateSpawnPlugin':
        return cls(**config)

    def on_pre_episode(self, ctx: SimContext) -> None:
        theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self.angle_deg = float(np.degrees(theta))
        core = ctx.accessor.get_core_state()[self.target_robot]
        pos = np.asarray(core['root_pos'], dtype=np.float32).copy()
        rot = np.asarray(core['root_rot'], dtype=np.float32).copy()

        c, s = np.cos(theta), np.sin(theta)
        new_x = c * pos[0] - s * pos[1]
        new_y = s * pos[0] + c * pos[1]
        pos[0], pos[1] = np.float32(new_x), np.float32(new_y)

        Rz = R.from_euler('z', theta)
        q_old = R.from_quat([rot[1], rot[2], rot[3], rot[0]])
        q_new = (Rz * q_old).as_quat()
        rot[:] = np.float32([q_new[3], q_new[0], q_new[1], q_new[2]])

        ctx.mutator.set_core_state({self.target_robot: {
            'root_pos': pos, 'root_rot': rot,
            'root_vel_local': np.asarray(
                core['root_vel_local'], dtype=np.float32),
            'root_angular_vel_local': np.asarray(
                core['root_angular_vel_local'], dtype=np.float32),
            'joint_pos_norm': np.asarray(
                core['joint_pos_norm'], dtype=np.float32),
            'joint_vel_norm': np.asarray(
                core['joint_vel_norm'], dtype=np.float32),
        }})


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
