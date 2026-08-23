"""站立触发扰动插件。

基于 ``ConstantForcePlugin`` 的施力机制（``apply_external_force``，
相对 heading 方向，持续 N 个 action step），加入站立触发状态机：

    WAIT_STAND → DELAY → PUSHING → WAIT_STAND → ...（循环）

机器人从倒地站起来（torso height > threshold）后，等待一段随机时间，
然后施加外力扰动。施力结束后回到 WAIT_STAND，等重新站立再扰动。
整个 episode 内循环，不终止 episode。

施力参数从 ``episode_options["impulse_params"]`` 读取，格式与
``ConstantForcePlugin`` / ``RelativeImpulsePlugin`` 一致::

    {"robot_a": {"direction_angle": 90.0, "force": 200.0,
                  "duration_action_steps": 4, "body": "torso"}}

方向定义（同 ConstantForcePlugin）：
    direction_angle 是力指向的方向（机器人倒下的方向），
    相对机器人 heading：0°=前, 90°=右, 180°=后, 270°=左。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BasePlugin
from envs.framework.context import SimContext


# --- 状态机常量 ---
_WAIT_STAND = 0
_DELAY = 1
_PUSHING = 2


class _RobotPushState:
    """单个机器人的扰动状态机。"""

    __slots__ = (
        "robot_id",
        "state",
        "delay_remaining",
        "push_remaining",
        "direction_vec",
        "force",
        "direction_angle",
        "duration_action_steps",
        "body_name",
        "push_count",
        "fall_count",
        "fell_during_push",
    )

    def __init__(self, robot_id: str, body_name: str):
        self.robot_id = robot_id
        self.state = _WAIT_STAND
        self.delay_remaining = 0
        self.push_remaining = 0
        self.direction_vec: Optional[np.ndarray] = None
        self.force: float = 0.0
        self.direction_angle: float = 0.0
        self.duration_action_steps: int = 0
        self.body_name = body_name
        self.push_count: int = 0
        self.fall_count: int = 0
        self.fell_during_push: bool = False


class StandingTriggeredForcePlugin(BasePlugin):
    """站立触发外力扰动插件。

    机器人站起来（height > threshold）后，等待 base_delay + random_delay
    个 action step，然后施加外力持续 duration_action_steps 个 action step。
    施力结束后回到等待状态，等重新站立再扰动。

    支持双机器人：每个机器人独立状态机。
    """

    def __init__(
        self,
        target_robots: Union[str, Sequence[str]] = ("robot_a", "robot_b"),
        standing_height_threshold: float = 1.15,
        base_delay_steps: int = 10,
        random_delay_max_steps: int = 20,
        impulse_body: str = "torso",
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robots: 目标机器人 ID 或 ID 列表。
            standing_height_threshold: 站立判断的 torso 高度阈值 (m)。
            base_delay_steps: 站立后基础等待步数（action steps）。
            random_delay_max_steps: 站立后额外随机等待步数上限（action steps）。
            impulse_body: 默认施力部位。
            random_seed: 随机种子（实际由 set_episode_seed 覆盖）。
        """
        if isinstance(target_robots, str):
            self._target_robots = [target_robots]
        else:
            self._target_robots = list(target_robots)

        self.standing_height_threshold = float(standing_height_threshold)
        self.base_delay_steps = int(base_delay_steps)
        self.random_delay_max_steps = int(random_delay_max_steps)
        self.impulse_body = impulse_body
        self._rng = np.random.RandomState(random_seed)

        # 每个机器人的状态机
        self._states: Dict[str, _RobotPushState] = {
            rid: _RobotPushState(rid, impulse_body)
            for rid in self._target_robots
        }

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return "standing_triggered_force"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robots": self._target_robots,
            "standing_height_threshold": self.standing_height_threshold,
            "base_delay_steps": self.base_delay_steps,
            "random_delay_max_steps": self.random_delay_max_steps,
            "impulse_body": self.impulse_body,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandingTriggeredForcePlugin":
        return cls(**config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_heading(root_rot: np.ndarray) -> float:
        """从 root_rot 四元数 [w,x,y,z] 提取 heading（yaw, 弧度）。"""
        rot = R.from_quat([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])
        forward = rot.apply(np.array([1.0, 0.0, 0.0]))
        return float(np.arctan2(forward[1], forward[0]))

    def _get_height(self, ctx: SimContext, robot_id: str) -> float:
        core_state = ctx.accessor.get_core_state()
        return float(np.asarray(core_state[robot_id]["root_pos"], dtype=np.float64)[2])

    def _load_params(self, ctx: SimContext) -> None:
        """从 episode_options["impulse_params"] 读取每个机器人的施力参数。"""
        params_all = ctx.episode_options.get("impulse_params", {})
        for rid in self._target_robots:
            st = self._states[rid]
            p = params_all.get(rid, {})
            if p:
                st.force = float(p.get("force", 0.0))
                st.direction_angle = float(p.get("direction_angle", 0.0))
                st.duration_action_steps = int(p.get("duration_action_steps", 0))
                body = p.get("body", p.get("body_name"))
                if body is not None:
                    st.body_name = str(body)
            else:
                st.force = 0.0
                st.duration_action_steps = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._load_params(ctx)
        for rid in self._target_robots:
            st = self._states[rid]
            st.state = _WAIT_STAND
            st.delay_remaining = 0
            st.push_remaining = 0
            st.direction_vec = None
            st.push_count = 0
            st.fall_count = 0
            st.fell_during_push = False
            ctx.metrics[f"{rid}_push_count"] = 0
            ctx.metrics[f"{rid}_fall_count"] = 0
            ctx.metrics[f"{rid}_push_active"] = False

    def on_pre_action_step(self, ctx: SimContext) -> None:
        for rid in self._target_robots:
            st = self._states[rid]
            height = self._get_height(ctx, rid)

            if st.state == _WAIT_STAND:
                if height > self.standing_height_threshold and st.duration_action_steps > 0:
                    # 站起来了，进入延迟
                    st.delay_remaining = (
                        self.base_delay_steps
                        + int(self._rng.randint(0, self.random_delay_max_steps + 1))
                    )
                    st.state = _DELAY
                    ctx.metrics[f"{rid}_push_active"] = False

            elif st.state == _DELAY:
                if height < self.standing_height_threshold:
                    # 机器人倒下，站立步数清零，回到等待
                    st.state = _WAIT_STAND
                    st.delay_remaining = 0
                    ctx.metrics[f"{rid}_push_active"] = False
                elif st.delay_remaining > 0:
                    st.delay_remaining -= 1
                else:
                    # 延迟结束，进入施力
                    # 此时机器人已站立，heading 稳定
                    core_state = ctx.accessor.get_core_state()
                    root_rot = np.asarray(
                        core_state[rid]["root_rot"], dtype=np.float64
                    )
                    heading = self._extract_heading(root_rot)
                    abs_angle = heading - np.radians(st.direction_angle)
                    st.direction_vec = np.array(
                        [np.cos(abs_angle), np.sin(abs_angle), 0.0],
                        dtype=np.float64,
                    )
                    st.push_remaining = st.duration_action_steps
                    st.state = _PUSHING
                    st.push_count += 1
                    st.fell_during_push = False

                    ctx.metrics[f"{rid}_impulse_body"] = st.body_name
                    ctx.metrics[f"{rid}_impulse_force"] = st.force
                    ctx.metrics[f"{rid}_impulse_direction_angle"] = st.direction_angle
                    ctx.metrics[f"{rid}_impulse_duration_action_steps"] = st.duration_action_steps
                    ctx.metrics[f"{rid}_impulse_direction"] = st.direction_vec.tolist()
                    ctx.metrics[f"{rid}_impulse_heading"] = heading
                    ctx.metrics[f"{rid}_push_count"] = st.push_count
                    ctx.metrics[f"{rid}_push_active"] = True

            elif st.state == _PUSHING:
                # 施力期间检测机器人是否摔倒
                if not st.fell_during_push and height < self.standing_height_threshold:
                    st.fell_during_push = True
                    st.fall_count += 1
                    ctx.metrics[f"{rid}_fall_count"] = st.fall_count

                if st.push_remaining <= 0:
                    # 施力结束，回到等待
                    st.state = _WAIT_STAND
                    st.direction_vec = None
                    ctx.metrics[f"{rid}_push_active"] = False

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        for rid in self._target_robots:
            st = self._states[rid]
            if st.state == _PUSHING and st.direction_vec is not None and st.push_remaining > 0:
                ctx.mutator.apply_external_force(
                    body_name=st.body_name,
                    force=st.direction_vec * st.force,
                    robot_id=rid,
                )

    def on_post_action_step(self, ctx: SimContext) -> None:
        for rid in self._target_robots:
            st = self._states[rid]
            if st.state == _PUSHING and st.push_remaining > 0:
                st.push_remaining -= 1
                if st.push_remaining <= 0:
                    st.state = _WAIT_STAND
                    st.direction_vec = None
                    ctx.metrics[f"{rid}_push_active"] = False
