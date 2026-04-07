"""
Humanoid21 外部扰动插件

提供多种外部扰动模式，用于测试机器人的鲁棒性和平衡能力。
"""

import numpy as np
from typing import Optional

from framework import BasePlugin, SimContext


class RandomPushPlugin(BasePlugin):
    """
    随机推力插件

    在随机时间间隔对指定机器人施加随机方向的推力。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        target_body: str = "torso",
        force_magnitude: float = 200.0,  # 牛顿
        min_interval: int = 50,  # 最小间隔（物理步数）
        max_interval: int = 150,  # 最大间隔（物理步数）
        push_duration_steps: int = 1,  # 每次推力持续步数
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            target_body: 目标 body 名称
            force_magnitude: 力的大小（牛顿）
            min_interval: 最小扰动间隔
            max_interval: 最大扰动间隔
            push_duration_steps: 每次推力持续的物理步数（默认为1，即单帧推力）
            random_seed: 随机种子
        """
        self.target_robot = target_robot
        self.target_body = target_body
        self.force_magnitude = force_magnitude
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.push_duration_steps = push_duration_steps

        self._rng = np.random.RandomState(random_seed)
        self._step_count = 0
        self._next_disturbance = self._rng.randint(min_interval, max_interval)
        self._current_force = None  # 当前持续施加的力
        self._push_remaining_steps = 0  # 当前推力剩余步数

    @property
    def name(self) -> str:
        return "random_push"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Episode 开始时重置"""
        self._step_count = 0
        self._next_disturbance = self._rng.randint(self.min_interval, self.max_interval)
        self._current_force = None
        self._push_remaining_steps = 0
        ctx.metrics[f'{self.target_robot}_push_count'] = 0

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """在每个物理步前检查是否需要施加扰动"""
        self._step_count += 1

        # 如果正在持续推力，继续施加
        if self._push_remaining_steps > 0:
            if self._current_force is not None:
                ctx.mutator.apply_external_force(
                    body_name=self.target_body,
                    force=self._current_force,
                    robot_id=self.target_robot
                )
                ctx.metrics[f'{self.target_robot}_push_active'] = True
            self._push_remaining_steps -= 1
            return

        # 检查是否需要开始新的推力
        if self._step_count >= self._next_disturbance:
            # 生成随机方向的力（水平面，避免直接推倒）
            angle = self._rng.uniform(0, 2 * np.pi)
            force = np.array([
                np.cos(angle) * self.force_magnitude,
                np.sin(angle) * self.force_magnitude,
                self._rng.uniform(-0.2, 0.2) * self.force_magnitude  # 轻微垂直分量
            ])

            # 保存当前力和持续步数
            self._current_force = force
            self._push_remaining_steps = self.push_duration_steps - 1  # 第一帧已施加

            # 施加力
            ctx.mutator.apply_external_force(
                body_name=self.target_body,
                force=force,
                robot_id=self.target_robot
            )

            # 记录
            ctx.metrics[f'{self.target_robot}_push_count'] += 1
            ctx.metrics[f'{self.target_robot}_last_push_force'] = float(np.linalg.norm(force))
            ctx.metrics[f'{self.target_robot}_push_duration_steps'] = self.push_duration_steps

            # 设置下一次扰动时间
            self._next_disturbance = self._step_count + self._rng.randint(
                self.min_interval, self.max_interval
            )


class PeriodicUpwardForcePlugin(BasePlugin):
    """
    周期性向上推力插件

    按固定间隔施加向上的力，可用于测试机器人的抗干扰能力。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        target_body: str = "torso",
        force_magnitude: float = 300.0,  # 牛顿
        interval: int = 100,  # 间隔（物理步数）
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            target_body: 目标 body 名称
            force_magnitude: 力的大小（牛顿）
            interval: 扰动间隔（物理步数）
        """
        self.target_robot = target_robot
        self.target_body = target_body
        self.force_magnitude = force_magnitude
        self.interval = interval
        self._step_count = 0

    @property
    def name(self) -> str:
        return "periodic_upward_force"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Episode 开始时重置"""
        self._step_count = 0
        ctx.metrics[f'{self.target_robot}_upward_force_count'] = 0

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """在每个物理步前检查是否需要施加扰动"""
        self._step_count += 1

        if self._step_count % self.interval == 0:
            # 施加向上的力
            force = np.array([0, 0, self.force_magnitude])

            ctx.mutator.apply_external_force(
                body_name=self.target_body,
                force=force,
                robot_id=self.target_robot
            )

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
    print("       push_duration_steps=1  # 每次推力持续1帧（默认）")
    print("   )")
    print("")
    print("   # 持续3帧的推力（更有冲击力）")
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
    print("       interval=100")
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
