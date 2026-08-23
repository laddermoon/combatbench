"""站立触发扰动插件。

基于 ``ConstantForcePlugin`` 的施力机制（``apply_external_force``，
相对 heading 方向，持续 N 个 action step），加入站立触发状态机：

    WAIT_STAND → DELAY → PUSHING → OBSERVE → (检查 standing_timer) → DELAY 或 WAIT_STAND

机器人从倒地站起来（torso height > threshold）后，持续站立满
``standing_settle_steps``（默认 1 秒 = 20 action steps），再等待
0~0.5 秒随机延迟，然后施加外力扰动。施力结束后进入观察期
（默认 1 秒），在 PUSHING + OBSERVE 整个窗口内检测机器人是否
有非脚部身体部位接触地面（第三点触地），以此判断是否摔倒。

摔倒检测借鉴 ``ImbalanceTerminationPlugin._is_non_foot_grounded()``，
通过 MuJoCo 接触数据筛选机器人↔地面接触，排除双脚。

每次扰动的 direction_angle 和 duration 在插件内部独立随机采样，
确保同一 episode 内多次扰动的参数各不相同。force 由课程 level 决定，
duration 在 level 指定的区间内随机。

施力参数从 ``episode_options["impulse_params"]`` 读取::

    {"robot_a": {"force": 100.0, "duration_min": 11, "duration_max": 20,
                 "body": "torso", "seed": 42}}

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
_OBSERVE = 3

# 脚部 body 名称（排除在第三点触地检测之外）
_FOOT_BODY_NAMES = {'foot_left', 'foot_right'}


class _RobotPushState:
    """单个机器人的扰动状态机。"""

    __slots__ = (
        "robot_id",
        "state",
        "standing_timer",
        "delay_remaining",
        "push_remaining",
        "observe_remaining",
        "direction_vec",
        "force",
        "duration_min",
        "duration_max",
        "body_name",
        "push_count",
        "fall_count",
        "fell_during_push",
    )

    def __init__(self, robot_id: str, body_name: str):
        self.robot_id = robot_id
        self.state = _WAIT_STAND
        self.standing_timer: int = 0
        self.delay_remaining: int = 0
        self.push_remaining: int = 0
        self.observe_remaining: int = 0
        self.direction_vec: Optional[np.ndarray] = None
        self.force: float = 0.0
        self.duration_min: int = 0
        self.duration_max: int = 0
        self.body_name = body_name
        self.push_count: int = 0
        self.fall_count: int = 0
        self.fell_during_push: bool = False


class StandingTriggeredForcePlugin(BasePlugin):
    """站立触发外力扰动插件。

    机器人站起来（height > threshold）并持续站立满 ``standing_settle_steps``
    后，等待 0~0.5 秒随机延迟，然后施加外力持续 ``duration`` 个 action step。
    施力结束后进入观察期 ``observe_steps``，在 PUSHING + OBSERVE 整个窗口内
    检测第三点触地（非脚部身体部位接触地面）来判断是否摔倒。

    支持双机器人：每个机器人独立状态机。
    """

    def __init__(
        self,
        target_robots: Union[str, Sequence[str]] = ("robot_a", "robot_b"),
        standing_height_threshold: float = 1.15,
        standing_settle_steps: int = 20,
        random_delay_max_steps: int = 10,
        observe_steps: int = 20,
        impulse_body: str = "torso",
        force_threshold: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robots: 目标机器人 ID 或 ID 列表。
            standing_height_threshold: 站立判断的 torso 高度阈值 (m)。
            standing_settle_steps: 站稳所需持续站立步数（action steps，默认 20=1 秒）。
            random_delay_max_steps: 站稳后随机等待步数上限（action steps，默认 10=0.5 秒）。
            observe_steps: 施力后观察窗口步数（action steps，默认 20=1 秒）。
            impulse_body: 默认施力部位。
            force_threshold: 接触力阈值（牛顿），低于此值的接触不计为触地。
            random_seed: 随机种子（实际由 episode_options 中的 seed 覆盖）。
        """
        if isinstance(target_robots, str):
            self._target_robots = [target_robots]
        else:
            self._target_robots = list(target_robots)

        self.standing_height_threshold = float(standing_height_threshold)
        self.standing_settle_steps = int(standing_settle_steps)
        self.random_delay_max_steps = int(random_delay_max_steps)
        self.observe_steps = int(observe_steps)
        self.impulse_body = impulse_body
        self.force_threshold = float(force_threshold)
        self._rng = np.random.RandomState(random_seed)
        self._ground_geom_name: Optional[str] = None

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
            "standing_settle_steps": self.standing_settle_steps,
            "random_delay_max_steps": self.random_delay_max_steps,
            "observe_steps": self.observe_steps,
            "impulse_body": self.impulse_body,
            "force_threshold": self.force_threshold,
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

    def _is_non_foot_grounded(self, ctx: SimContext, robot_id: str) -> bool:
        """检查指定机器人是否有非脚部部位与地面接触。

        借鉴 ImbalanceTerminationPlugin._is_non_foot_grounded() 的实现，
       使用 contacts_vec 通过 aff 筛选机器人↔环境接触。
        """
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')
        if cv is None or cv['ncon'] == 0:
            return False

        static_data = ctx.accessor.get_static_data()
        body_id_to_name = static_data.get('body_id_to_name', {})
        geom_id_to_name = static_data.get('geom_id_to_name', {})
        ground_geom = self._ground_geom_name or 'ground'

        robot_aff = 1 if robot_id == 'robot_a' else 2

        aff1 = cv['aff1']
        aff2 = cv['aff2']
        geom1 = cv['geom1']
        geom2 = cv['geom2']
        body1 = cv['body1']
        body2 = cv['body2']
        force_mag = cv['force_mag']

        for i in range(cv['ncon']):
            # 一侧是环境 (aff=0)，另一侧是目标机器人
            if aff1[i] == 0 and aff2[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom1[i]), '')
                body_robot = body_id_to_name.get(int(body2[i]), '')
            elif aff2[i] == 0 and aff1[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom2[i]), '')
                body_robot = body_id_to_name.get(int(body1[i]), '')
            else:
                continue

            if geom_env != ground_geom:
                continue
            if float(force_mag[i]) < self.force_threshold:
                continue
            if not any(foot in body_robot for foot in _FOOT_BODY_NAMES):
                return True

        return False

    def _load_params(self, ctx: SimContext) -> None:
        """从 episode_options["impulse_params"] 读取每个机器人的施力参数。

        新格式:
            {"force": 100.0, "duration_min": 11, "duration_max": 20,
             "body": "torso", "seed": 42}

        兼容旧格式:
            {"direction_angle": 90.0, "force": 200.0,
             "duration_action_steps": 4, "body": "torso"}
        """
        params_all = ctx.episode_options.get("impulse_params", {})
        for rid in self._target_robots:
            st = self._states[rid]
            p = params_all.get(rid, {})
            if p:
                st.force = float(p.get("force", 0.0))
                if "duration_min" in p and "duration_max" in p:
                    st.duration_min = int(p["duration_min"])
                    st.duration_max = int(p["duration_max"])
                elif "duration_action_steps" in p:
                    # 兼容旧格式
                    st.duration_max = int(p["duration_action_steps"])
                    st.duration_min = max(1, st.duration_max)
                else:
                    st.duration_min = 0
                    st.duration_max = 0
                body = p.get("body", p.get("body_name"))
                if body is not None:
                    st.body_name = str(body)
            else:
                st.force = 0.0
                st.duration_min = 0
                st.duration_max = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._load_params(ctx)

        # 从 static_data 获取地面 geom 名称
        static_data = ctx.accessor.get_static_data()
        self._ground_geom_name = static_data.get('ground_geom_name', 'ground')

        # 从 impulse_params 中读取 seed 初始化 RNG
        params_all = ctx.episode_options.get("impulse_params", {})
        seed = None
        for rid in self._target_robots:
            p = params_all.get(rid, {})
            if "seed" in p:
                seed = int(p["seed"])
                break
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        for rid in self._target_robots:
            st = self._states[rid]
            st.state = _WAIT_STAND
            st.standing_timer = 0
            st.delay_remaining = 0
            st.push_remaining = 0
            st.observe_remaining = 0
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

            # --- 站立计时器：始终运行，跨状态 ---
            if height > self.standing_height_threshold:
                st.standing_timer += 1
            else:
                st.standing_timer = 0

            if st.state == _WAIT_STAND:
                if st.standing_timer >= self.standing_settle_steps and st.force > 0:
                    # 站稳满 1 秒，进入延迟
                    st.delay_remaining = int(
                        self._rng.randint(0, self.random_delay_max_steps + 1)
                    )
                    st.state = _DELAY
                    ctx.metrics[f"{rid}_push_active"] = False

            elif st.state == _DELAY:
                if st.delay_remaining > 0:
                    st.delay_remaining -= 1
                else:
                    # 延迟结束，独立采样本次扰动参数，进入施力
                    direction_angle = float(self._rng.uniform(0, 360))
                    duration = int(self._rng.randint(
                        st.duration_min, st.duration_max + 1
                    ))

                    core_state = ctx.accessor.get_core_state()
                    root_rot = np.asarray(
                        core_state[rid]["root_rot"], dtype=np.float64
                    )
                    heading = self._extract_heading(root_rot)
                    abs_angle = heading - np.radians(direction_angle)
                    st.direction_vec = np.array(
                        [np.cos(abs_angle), np.sin(abs_angle), 0.0],
                        dtype=np.float64,
                    )
                    st.push_remaining = duration
                    st.state = _PUSHING
                    st.push_count += 1
                    st.fell_during_push = False

                    ctx.metrics[f"{rid}_impulse_body"] = st.body_name
                    ctx.metrics[f"{rid}_impulse_force"] = st.force
                    ctx.metrics[f"{rid}_impulse_direction_angle"] = direction_angle
                    ctx.metrics[f"{rid}_impulse_duration_action_steps"] = duration
                    ctx.metrics[f"{rid}_impulse_direction"] = st.direction_vec.tolist()
                    ctx.metrics[f"{rid}_impulse_heading"] = heading
                    ctx.metrics[f"{rid}_push_count"] = st.push_count
                    ctx.metrics[f"{rid}_push_active"] = True

            elif st.state == _PUSHING:
                # 摔倒检测（第三点触地）
                if not st.fell_during_push and self._is_non_foot_grounded(ctx, rid):
                    st.fell_during_push = True
                    st.fall_count += 1
                    ctx.metrics[f"{rid}_fall_count"] = st.fall_count

                if st.push_remaining <= 0:
                    # 施力结束，进入观察
                    st.observe_remaining = self.observe_steps
                    st.state = _OBSERVE
                    st.direction_vec = None
                    ctx.metrics[f"{rid}_push_active"] = False

            elif st.state == _OBSERVE:
                # 继续摔倒检测
                if not st.fell_during_push and self._is_non_foot_grounded(ctx, rid):
                    st.fell_during_push = True
                    st.fall_count += 1
                    ctx.metrics[f"{rid}_fall_count"] = st.fall_count

                if st.observe_remaining > 0:
                    st.observe_remaining -= 1
                else:
                    # 观察结束，检查 standing_timer 决定下一步
                    if st.standing_timer >= self.standing_settle_steps:
                        # 一直站着，直接进入延迟（只需等随机时间）
                        st.delay_remaining = int(
                            self._rng.randint(0, self.random_delay_max_steps + 1)
                        )
                        st.state = _DELAY
                    else:
                        # 倒过或还没站稳，回到等待
                        st.state = _WAIT_STAND

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
