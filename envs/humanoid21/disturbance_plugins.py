"""
Humanoid21 外部扰动插件

提供多种外部扰动模式，用于测试机器人的鲁棒性和平衡能力。
"""

import os

import numpy as np
from typing import Optional

from framework import BasePlugin, SimContext


_TURB_DEBUG = os.environ.get("COMBATBENCH_TURB_DEBUG", "0") == "1"
_TURB_DEBUG_MAX_PHYS_STEPS = max(0, int(os.environ.get("COMBATBENCH_TURB_DEBUG_MAX_PHYS_STEPS", "400")))


class RandomPushPlugin(BasePlugin):
    """
    随机推力插件

    在随机动作步间隔后，对指定机器人施加持续若干动作步的随机方向推力。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        target_body: str = "torso",
        force_magnitude: float = 200.0,  # 牛顿
        min_interval: int = 50,  # 最小间隔（动作步数）
        max_interval: int = 150,  # 最大间隔（动作步数）
        push_duration_steps: int = 1,  # 每次推力持续动作步数
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            target_body: 目标 body 名称
            force_magnitude: 力的大小（牛顿）
            min_interval: 两次推力之间的最小等待动作步数
            max_interval: 两次推力之间的最大等待动作步数
            push_duration_steps: 每次推力持续的动作步数
            random_seed: 随机种子
        """
        if min_interval < 0:
            raise ValueError(f"min_interval must be >= 0, got {min_interval}")
        if max_interval < min_interval:
            raise ValueError(f"max_interval must be >= min_interval, got min={min_interval}, max={max_interval}")
        if push_duration_steps <= 0:
            raise ValueError(f"push_duration_steps must be > 0, got {push_duration_steps}")

        self.target_robot = target_robot
        self.target_body = target_body
        self.force_magnitude = force_magnitude
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.push_duration_steps = push_duration_steps

        self._rng = np.random.RandomState(random_seed)
        self._action_step_count = 0
        self._wait_remaining_action_steps = 0
        self._current_force = None  # 当前持续施加的力
        self._push_remaining_action_steps = 0  # 当前推力剩余动作步数
        self._push_active_this_action = False

    def _sample_interval_action_steps(self) -> int:
        if self.min_interval == self.max_interval:
            return self.min_interval
        return int(self._rng.randint(self.min_interval, self.max_interval + 1))

    def _sample_force(self) -> np.ndarray:
        angle = self._rng.uniform(0, 2 * np.pi)
        return np.array([
            np.cos(angle) * self.force_magnitude,
            np.sin(angle) * self.force_magnitude,
            self._rng.uniform(-0.2, 0.2) * self.force_magnitude
        ])

    def _debug_log(self, ctx: SimContext, stage: str, force: Optional[np.ndarray]) -> None:
        if not _TURB_DEBUG:
            return
        if _TURB_DEBUG_MAX_PHYS_STEPS > 0 and ctx.physics_step > _TURB_DEBUG_MAX_PHYS_STEPS:
            return
        core_state = ctx.accessor.get_core_state()[self.target_robot]
        derived_state = ctx.accessor.get_derived_state()[self.target_robot]
        root_pos = np.asarray(core_state["root_pos"], dtype=np.float64)
        root_vel_local = np.asarray(core_state["root_vel_local"], dtype=np.float64)
        linear_vel = np.asarray(derived_state["root_state"]["linear_vel"], dtype=np.float64)
        uprightness = float(np.asarray(derived_state["uprightness"], dtype=np.float64).reshape(-1)[0])
        force_array = np.zeros(3, dtype=np.float64) if force is None else np.asarray(force, dtype=np.float64)
        print(
            f"turb_debug[{self.target_robot}] stage={stage} phy={ctx.physics_step} epi={ctx.episode_step} "
            f"action={self._action_step_count} wait_remaining={self._wait_remaining_action_steps} "
            f"push_remaining={self._push_remaining_action_steps} active={self._push_active_this_action} "
            f"force=({force_array[0]:.6f},{force_array[1]:.6f},{force_array[2]:.6f}) |F|={float(np.linalg.norm(force_array)):.6f} "
            f"root_pos=({root_pos[0]:.6f},{root_pos[1]:.6f},{root_pos[2]:.6f}) "
            f"root_vel_local=({root_vel_local[0]:.6f},{root_vel_local[1]:.6f},{root_vel_local[2]:.6f}) "
            f"linear_vel=({linear_vel[0]:.6f},{linear_vel[1]:.6f},{linear_vel[2]:.6f}) upright={uprightness:.6f}",
            flush=True,
        )

    @property
    def name(self) -> str:
        return "random_push"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Episode 开始时重置"""
        self._action_step_count = 0
        self._wait_remaining_action_steps = self._sample_interval_action_steps()
        self._current_force = None
        self._push_remaining_action_steps = 0
        self._push_active_this_action = False
        ctx.metrics[f'{self.target_robot}_push_count'] = 0
        ctx.metrics[f'{self.target_robot}_push_active'] = False
        ctx.metrics[f'{self.target_robot}_next_push_wait_action_steps'] = self._wait_remaining_action_steps

    def on_pre_action_step(self, ctx: SimContext) -> None:
        """在动作步边界上调度扰动状态机"""
        self._action_step_count += 1
        self._push_active_this_action = False

        if self._push_remaining_action_steps > 0 and self._current_force is not None:
            self._push_active_this_action = True
            ctx.metrics[f'{self.target_robot}_push_active'] = True
            ctx.metrics[f'{self.target_robot}_next_push_wait_action_steps'] = 0
            self._debug_log(ctx, "action_continue", self._current_force)
            return

        if self._wait_remaining_action_steps > 0:
            self._wait_remaining_action_steps -= 1
            ctx.metrics[f'{self.target_robot}_push_active'] = False
            ctx.metrics[f'{self.target_robot}_next_push_wait_action_steps'] = self._wait_remaining_action_steps
            self._debug_log(ctx, "action_wait", None)
            return

        self._current_force = self._sample_force()
        self._push_remaining_action_steps = self.push_duration_steps
        self._push_active_this_action = True

        ctx.metrics[f'{self.target_robot}_push_count'] += 1
        ctx.metrics[f'{self.target_robot}_last_push_force'] = float(np.linalg.norm(self._current_force))
        ctx.metrics[f'{self.target_robot}_push_duration_action_steps'] = self.push_duration_steps
        ctx.metrics[f'{self.target_robot}_push_active'] = True
        ctx.metrics[f'{self.target_robot}_next_push_wait_action_steps'] = 0
        self._debug_log(ctx, "action_start", self._current_force)

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """在推力激活的动作步内，对每个物理步持续施力"""
        if not self._push_active_this_action or self._current_force is None:
            return

        self._debug_log(ctx, "phy_before_apply", self._current_force)
        ctx.mutator.apply_external_force(
            body_name=self.target_body,
            force=self._current_force,
            robot_id=self.target_robot
        )
        self._debug_log(ctx, "phy_after_apply", self._current_force)

    def on_post_action_step(self, ctx: SimContext) -> None:
        """在动作步结束后推进等待/持续计数器"""
        if self._push_active_this_action and self._push_remaining_action_steps > 0:
            self._push_remaining_action_steps -= 1
            if self._push_remaining_action_steps == 0:
                self._current_force = None
                self._wait_remaining_action_steps = self._sample_interval_action_steps()
        ctx.metrics[f'{self.target_robot}_push_active'] = self._push_active_this_action
        ctx.metrics[f'{self.target_robot}_next_push_wait_action_steps'] = self._wait_remaining_action_steps
        self._debug_log(ctx, "action_end", self._current_force)


class PeriodicUpwardForcePlugin(BasePlugin):
    """
    周期性向上推力插件

    按固定动作步间隔施加向上的力，可用于测试机器人的抗干扰能力。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        target_body: str = "torso",
        force_magnitude: float = 300.0,  # 牛顿
        interval: int = 100,  # 间隔（动作步数）
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            target_body: 目标 body 名称
            force_magnitude: 力的大小（牛顿）
            interval: 扰动间隔（动作步数）
        """
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")

        self.target_robot = target_robot
        self.target_body = target_body
        self.force_magnitude = force_magnitude
        self.interval = interval
        self._action_step_count = 0
        self._apply_this_action = False

    @property
    def name(self) -> str:
        return "periodic_upward_force"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Episode 开始时重置"""
        self._action_step_count = 0
        self._apply_this_action = False
        ctx.metrics[f'{self.target_robot}_upward_force_count'] = 0

    def on_pre_action_step(self, ctx: SimContext) -> None:
        """在动作步边界上调度周期性向上推力"""
        self._action_step_count += 1
        self._apply_this_action = (self._action_step_count % self.interval == 0)

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """在激活的动作步内，对每个物理步施加向上的力"""
        if self._apply_this_action:
            force = np.array([0, 0, self.force_magnitude])

            ctx.mutator.apply_external_force(
                body_name=self.target_body,
                force=force,
                robot_id=self.target_robot
            )

    def on_post_action_step(self, ctx: SimContext) -> None:
        """在动作步结束后记录本步是否施加了向上推力"""
        if self._apply_this_action:
            ctx.metrics[f'{self.target_robot}_upward_force_count'] += 1


class ConditionalHeightLimitPlugin(BasePlugin):
    """
    条件高度限制插件

    当机器人高度超过阈值时施加向下的力。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        max_height: float = 1.4,  # 最大高度（米）
        force_magnitude: float = 500.0,  # 向下的力（牛顿）
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            max_height: 最大允许高度
            force_magnitude: 施加的向下力大小
        """
        self.target_robot = target_robot
        self.max_height = max_height
        self.force_magnitude = force_magnitude

    @property
    def name(self) -> str:
        return "conditional_height_limit"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """检查高度并施加力"""
        derived_state = ctx.accessor.get_derived_state()
        height = derived_state[self.target_robot]['root_state']['height'][0]

        if height > self.max_height:
            # 施加向下的力
            force = np.array([0, 0, -self.force_magnitude])

            # 力的大小与超出高度成正比
            excess = height - self.max_height
            force *= min(excess * 2, 3.0)  # 最多放大3倍

            ctx.mutator.apply_external_force(
                body_name="torso",
                force=force,
                robot_id=self.target_robot
            )

            ctx.metrics[f'{self.target_robot}_height_limit_active'] = True


class ContinuousWindPlugin(BasePlugin):
    """
    持续风力插件

    持续施加侧向力，模拟风的效果。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        wind_direction: np.ndarray = None,  # 风向向量
        wind_strength: float = 50.0,  # 风力（牛顿）
        gust_probability: float = 0.01,  # 阵风概率
        gust_multiplier: float = 3.0,  # 阵风倍数
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            wind_direction: 风向向量（默认向右 [1, 0, 0]）
            wind_strength: 基础风力（牛顿）
            gust_probability: 每步出现阵风的概率
            gust_multiplier: 阵风强度倍数
        """
        self.target_robot = target_robot
        self.wind_direction = wind_direction if wind_direction is not None else np.array([1.0, 0.0, 0.0])
        self.wind_strength = wind_strength
        self.gust_probability = gust_probability
        self.gust_multiplier = gust_multiplier

        self._rng = np.random.RandomState()

    @property
    def name(self) -> str:
        return "continuous_wind"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """持续施加风力"""
        # 基础风力
        force = self.wind_direction * self.wind_strength

        # 随机阵风
        if self._rng.rand() < self.gust_probability:
            force *= self.gust_multiplier
            ctx.metrics[f'{self.target_robot}_gust_active'] = True

        # 施加到 torso
        ctx.mutator.apply_external_force(
            body_name="torso",
            force=force,
            robot_id=self.target_robot
        )


class HeadStrikePlugin(BasePlugin):
    """
    头部打击插件

    模拟对头部的突然打击，用于测试抗打击能力。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        strike_force: float = 400.0,  # 打击力（牛顿）
        strike_interval: int = 200,  # 打击间隔（物理步数）
        random_direction: bool = True,
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            strike_force: 打击力大小（牛顿）
            strike_interval: 打击间隔
            random_direction: 是否使用随机方向
        """
        self.target_robot = target_robot
        self.strike_force = strike_force
        self.strike_interval = strike_interval
        self.random_direction = random_direction

        self._step_count = 0
        self._rng = np.random.RandomState()

    @property
    def name(self) -> str:
        return "head_strike"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Episode 开始时重置"""
        self._step_count = 0
        ctx.metrics[f'{self.target_robot}_head_strike_count'] = 0

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """检查是否需要施加打击"""
        self._step_count += 1

        if self._step_count % self.strike_interval == 0:
            # 生成打击方向
            if self.random_direction:
                # 随机水平方向，略微向下
                angle = self._rng.uniform(0, 2 * np.pi)
                direction = np.array([
                    np.cos(angle),
                    np.sin(angle),
                    -0.3  # 略微向下
                ])
                direction = direction / np.linalg.norm(direction)
            else:
                # 固定方向：从前方打击
                direction = np.array([1.0, 0.0, -0.2])
                direction = direction / np.linalg.norm(direction)

            force = direction * self.strike_force

            # 施加到头部
            ctx.mutator.apply_external_force(
                body_name="head",
                force=force,
                robot_id=self.target_robot
            )

            ctx.metrics[f'{self.target_robot}_head_strike_count'] += 1


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("扰动插件使用示例")
    print("=" * 60)

    print("\n1. 随机推力插件:")
    print("   plugin = RandomPushPlugin(")
    print("       target_robot='robot_a',")
    print("       force_magnitude=200.0,")
    print("       min_interval=50,")
    print("       max_interval=150,")
    print("       push_duration_steps=1  # 每次推力持续1个动作步（默认）")
    print("   )")
    print("")
    print("   # 持续3个动作步的推力（更有冲击力）")
    print("   plugin = RandomPushPlugin(")
    print("       target_robot='robot_a',")
    print("       force_magnitude=200.0,")
    print("       min_interval=50,")
    print("       max_interval=150,")
    print("       push_duration_steps=3")
    print("   )")

    print("\n2. 周期性向上推力:")
    print("   plugin = PeriodicUpwardForcePlugin(")
    print("       target_robot='robot_b',")
    print("       force_magnitude=300.0,")
    print("       interval=100  # 每100个动作步施加一次")
    print("   )")

    print("\n3. 高度限制插件:")
    print("   plugin = ConditionalHeightLimitPlugin(")
    print("       target_robot='robot_a',")
    print("       max_height=1.4,")
    print("       force_magnitude=500.0")
    print("   )")

    print("\n4. 持续风力:")
    print("   plugin = ContinuousWindPlugin(")
    print("       target_robot='robot_a',")
    print("       wind_direction=[1, 0, 0],  # 向右吹")
    print("       wind_strength=50.0")
    print("   )")

    print("\n5. 头部打击:")
    print("   plugin = HeadStrikePlugin(")
    print("       target_robot='robot_a',")
    print("       strike_force=400.0,")
    print("       strike_interval=200")
    print("   )")
