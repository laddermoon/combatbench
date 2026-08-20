"""
Humanoid21 外部扰动插件

提供多种外部扰动模式，用于测试机器人的鲁棒性和平衡能力。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BasePlugin
from envs.framework.context import ReadOnlySimContext, SimContext
from envs.framework.observer_plugin import BaseRuntimeUnit


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
        # RNG is rebuilt on every set_episode_seed() call; the ctor value is
        # only used until the first runner-issued seed arrives.
        self._rng = np.random.RandomState(random_seed)
        self._action_step_count = 0
        self._wait_remaining_action_steps = 0
        self._current_force = None  # 当前持续施加的力
        self._push_remaining_action_steps = 0  # 当前推力剩余动作步数
        self._push_active_this_action = False

    def set_episode_seed(self, seed: int) -> None:
        """Rebuild the per-plugin RNG immediately (see framework/SEED.md)."""
        self._rng = np.random.RandomState(int(seed))

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

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "target_body": self.target_body,
            "force_magnitude": self.force_magnitude,
            "min_interval": self.min_interval,
            "max_interval": self.max_interval,
            "push_duration_steps": self.push_duration_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "RandomPushPlugin":
        return cls(**config)

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Episode 开始时重置（RNG 已由 set_episode_seed 重建）"""
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


class InitialStatePerturbationPlugin(BasePlugin):
    def __init__(
        self,
        target_robot: str = "robot_a",
        # 单位说明（参见 DATASPEC.md §3 核心状态）：
        # 作用于 joint_pos_norm（归一化关节角度 [-1, 1]），无量纲，1.0 = 整段关节行程的半量程。
        joint_pos_delta_max: float = 0.1,
        # 作用于 joint_vel_norm（归一化关节角速度），无量纲，单位为“半量程/秒”。
        joint_vel_delta_max: float = 0.1,
        # 作用于 root_pos[:2]（Torso 世界 xy 坐标），单位：米 (m)。
        root_xy_offset_max: float = 0.0,
        # 绕自身 x、y 轴的倾斜扰动（欧拉角，'xy' 顺序），单位：度 (deg)。
        root_tilt_deg_max: float | Sequence[float] = 0.0,
        # 作用于 root_vel_local（Torso 局部线速度 (x, y, z)），单位：米/秒 (m/s)。
        root_linear_velocity_delta_max: float | Sequence[float] = 0.0,
        # 作用于 root_angular_vel_local（Torso 局部角速度 (x, y, z)），单位：弧度/秒 (rad/s)。
        root_angular_velocity_delta_max: float | Sequence[float] = 0.0,
        # 扰动姿态后是否把机器人重新"放回地面"。
        #
        # 关节扰动与躯干倾斜都会改变足底相对 Torso 的高度，而本插件不修改
        # ``root_pos[2]``。由于标准站姿近乎直腿，任意随机关节偏移只会"缩短"
        # 有效腿长，因此扰动总是让双脚离地（单边偏置，永远不会压入地面）。
        # 结果是 ``joint_pos_delta_max`` 隐式变成了一个"跌落高度"参数：在
        # scale=0.9 下约 46% 的 episode 开局时双脚离地 >5cm，最高 29cm。
        #
        # 打开本选项后，姿态扰动完成后会平移 ``root_pos[2]``，使最低的脚
        # 回到扰动前的标称高度，从而把"跌落"与"姿态扰动"解耦。
        #
        # 默认 False 以保持既有实验的行为不变。
        reground: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.target_robot = target_robot
        self.reground = bool(reground)
        self.joint_pos_delta_max = float(joint_pos_delta_max)
        self.joint_vel_delta_max = float(joint_vel_delta_max)
        self.root_xy_offset_max = float(root_xy_offset_max)
        self.root_tilt_deg_max = self._as_max_vector(root_tilt_deg_max, 2, "root_tilt_deg_max")
        self.root_linear_velocity_delta_max = self._as_max_vector(
            root_linear_velocity_delta_max,
            3,
            "root_linear_velocity_delta_max",
        )
        self.root_angular_velocity_delta_max = self._as_max_vector(
            root_angular_velocity_delta_max,
            3,
            "root_angular_velocity_delta_max",
        )
        # RNG is rebuilt on every set_episode_seed() call; the ctor value is
        # only used until the first runner-issued seed arrives.
        self._rng = np.random.RandomState(random_seed)

        if self.joint_pos_delta_max < 0.0:
            raise ValueError(f"joint_pos_delta_max must be >= 0, got {joint_pos_delta_max}")
        if self.joint_vel_delta_max < 0.0:
            raise ValueError(f"joint_vel_delta_max must be >= 0, got {joint_vel_delta_max}")
        if self.root_xy_offset_max < 0.0:
            raise ValueError(f"root_xy_offset_max must be >= 0, got {root_xy_offset_max}")

    @staticmethod
    def _as_max_vector(value: float | Sequence[float], length: int, name: str) -> np.ndarray:
        if np.isscalar(value):
            scalar_value = float(value)
            if scalar_value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
            return np.full((length,), scalar_value, dtype=np.float32)
        array_value = np.asarray(value, dtype=np.float32).reshape(-1)
        if array_value.shape[0] != length:
            raise ValueError(f"{name} must have length {length}, got shape {array_value.shape}")
        if np.any(array_value < 0.0):
            raise ValueError(f"{name} entries must be >= 0, got {value}")
        return array_value

    def _sample_signed(self, max_value: float, shape: tuple[int, ...]) -> np.ndarray:
        if max_value <= 0.0:
            return np.zeros(shape, dtype=np.float32)
        return self._rng.uniform(-max_value, max_value, size=shape).astype(np.float32)

    def _sample_signed_vector(self, max_values: np.ndarray) -> np.ndarray:
        if np.all(max_values <= 0.0):
            return np.zeros_like(max_values, dtype=np.float32)
        return self._rng.uniform(-max_values, max_values).astype(np.float32)

    def set_episode_seed(self, seed: int) -> None:
        """Rebuild the per-plugin RNG immediately (see framework/SEED.md)."""
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return f"initial_state_perturbation_{self.target_robot}"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "joint_pos_delta_max": self.joint_pos_delta_max,
            "joint_vel_delta_max": self.joint_vel_delta_max,
            "root_xy_offset_max": self.root_xy_offset_max,
            "root_tilt_deg_max": self.root_tilt_deg_max.tolist(),
            "root_linear_velocity_delta_max": self.root_linear_velocity_delta_max.tolist(),
            "root_angular_velocity_delta_max": self.root_angular_velocity_delta_max.tolist(),
            "reground": self.reground,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "InitialStatePerturbationPlugin":
        return cls(**config)

    def _lowest_foot_z(self, ctx: SimContext) -> Optional[float]:
        """当前状态下该机器人两只脚 body 原点的最低世界系 z (m)。"""
        static = ctx.accessor.get_static_data()
        keypoints = static.get(self.target_robot, {}).get("keypoint_body_names", {})
        body_xpos = ctx.accessor.get_derived_state([self.target_robot]).get(
            self.target_robot, {}
        ).get("body_xpos", {})

        zs = [
            float(body_xpos[keypoints[name]][2])
            for name in ("foot_left", "foot_right")
            if name in keypoints and keypoints[name] in body_xpos
        ]
        return min(zs) if zs else None

    def on_pre_episode(self, ctx: SimContext) -> None:
        # RNG 已由 set_episode_seed 重建；on_pre_episode 只负责业务逻辑。
        # 标称足底高度必须在任何 mutate 之前采样。
        nominal_foot_z = self._lowest_foot_z(ctx) if self.reground else None

        core_state = ctx.accessor.get_core_state()
        if self.target_robot not in core_state:
            raise ValueError(f"Unknown target_robot: {self.target_robot}")

        target_state = core_state[self.target_robot]
        new_state = {
            self.target_robot: {
                'root_pos': np.asarray(target_state['root_pos'], dtype=np.float32).copy(),
                'root_rot': np.asarray(target_state['root_rot'], dtype=np.float32).copy(),
                'root_vel_local': np.asarray(target_state['root_vel_local'], dtype=np.float32).copy(),
                'root_angular_vel_local': np.asarray(target_state['root_angular_vel_local'], dtype=np.float32).copy(),
                'joint_pos_norm': np.asarray(target_state['joint_pos_norm'], dtype=np.float32).copy(),
                'joint_vel_norm': np.asarray(target_state['joint_vel_norm'], dtype=np.float32).copy(),
            }
        }

        joint_pos_delta = self._sample_signed(
            self.joint_pos_delta_max,
            new_state[self.target_robot]['joint_pos_norm'].shape,
        )
        joint_vel_delta = self._sample_signed(
            self.joint_vel_delta_max,
            new_state[self.target_robot]['joint_vel_norm'].shape,
        )
        root_xy_delta = self._sample_signed(self.root_xy_offset_max, (2,))
        root_linear_velocity_delta = self._sample_signed_vector(self.root_linear_velocity_delta_max)
        root_angular_velocity_delta = self._sample_signed_vector(self.root_angular_velocity_delta_max)
        root_tilt_delta_deg = self._sample_signed_vector(self.root_tilt_deg_max)

        new_state[self.target_robot]['joint_pos_norm'] = np.clip(
            new_state[self.target_robot]['joint_pos_norm'] + joint_pos_delta,
            -1.0,
            1.0,
        )
        new_state[self.target_robot]['joint_vel_norm'] = (
            new_state[self.target_robot]['joint_vel_norm'] + joint_vel_delta
        ).astype(np.float32)
        new_state[self.target_robot]['root_pos'][:2] = (
            new_state[self.target_robot]['root_pos'][:2] + root_xy_delta
        ).astype(np.float32)
        new_state[self.target_robot]['root_vel_local'] = (
            new_state[self.target_robot]['root_vel_local'] + root_linear_velocity_delta
        ).astype(np.float32)
        new_state[self.target_robot]['root_angular_vel_local'] = (
            new_state[self.target_robot]['root_angular_vel_local'] + root_angular_velocity_delta
        ).astype(np.float32)

        current_root_rot = new_state[self.target_robot]['root_rot']
        current_rotation = R.from_quat([
            float(current_root_rot[1]),
            float(current_root_rot[2]),
            float(current_root_rot[3]),
            float(current_root_rot[0]),
        ])
        tilt_rotation = R.from_euler('xy', root_tilt_delta_deg.astype(np.float64), degrees=True)
        perturbed_rotation = current_rotation * tilt_rotation
        perturbed_quat_xyzw = perturbed_rotation.as_quat().astype(np.float32)
        new_state[self.target_robot]['root_rot'] = np.array([
            perturbed_quat_xyzw[3],
            perturbed_quat_xyzw[0],
            perturbed_quat_xyzw[1],
            perturbed_quat_xyzw[2],
        ], dtype=np.float32)

        ctx.mutator.set_core_state(new_state)

        # 姿态扰动会抬高足底（见 ``reground`` 说明）。平移 root 高度把最低的
        # 脚放回标称高度，使扰动只改变"姿态"而不附带一次自由落体。
        reground_dz = 0.0
        if self.reground and nominal_foot_z is not None:
            perturbed_foot_z = self._lowest_foot_z(ctx)
            if perturbed_foot_z is not None:
                reground_dz = float(nominal_foot_z - perturbed_foot_z)
                new_state[self.target_robot]['root_pos'][2] = np.float32(
                    new_state[self.target_robot]['root_pos'][2] + reground_dz
                )
                ctx.mutator.set_core_state(new_state)
        ctx.metrics[f'{self.target_robot}_initial_perturbation_reground_dz'] = reground_dz

        ctx.metrics[f'{self.target_robot}_initial_perturbation_joint_pos_linf'] = float(np.max(np.abs(joint_pos_delta)))
        ctx.metrics[f'{self.target_robot}_initial_perturbation_joint_vel_linf'] = float(np.max(np.abs(joint_vel_delta)))
        ctx.metrics[f'{self.target_robot}_initial_perturbation_root_xy_offset'] = float(np.linalg.norm(root_xy_delta))
        ctx.metrics[f'{self.target_robot}_initial_perturbation_root_linear_velocity'] = float(np.linalg.norm(root_linear_velocity_delta))
        ctx.metrics[f'{self.target_robot}_initial_perturbation_root_angular_velocity'] = float(np.linalg.norm(root_angular_velocity_delta))
        ctx.metrics[f'{self.target_robot}_initial_perturbation_root_tilt_deg'] = float(np.linalg.norm(root_tilt_delta_deg))


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

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "target_body": self.target_body,
            "force_magnitude": self.force_magnitude,
            "interval": self.interval,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "PeriodicUpwardForcePlugin":
        return cls(**config)

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

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "max_height": self.max_height,
            "force_magnitude": self.force_magnitude,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ConditionalHeightLimitPlugin":
        return cls(**config)

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
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            wind_direction: 风向向量（默认向右 [1, 0, 0]）
            wind_strength: 基础风力（牛顿）
            gust_probability: 每步出现阵风的概率
            gust_multiplier: 阵风强度倍数
            random_seed: 随机种子（阵风采样用）
        """
        self.target_robot = target_robot
        self.wind_direction = wind_direction if wind_direction is not None else np.array([1.0, 0.0, 0.0])
        self.wind_strength = wind_strength
        self.gust_probability = gust_probability
        self.gust_multiplier = gust_multiplier

        # RNG is rebuilt on every set_episode_seed() call.
        self._rng = np.random.RandomState(random_seed)

    def to_blueprint(self) -> Dict[str, Any]:
        wd = self.wind_direction
        return {
            "target_robot": self.target_robot,
            "wind_direction": wd.tolist() if isinstance(wd, np.ndarray) else wd,
            "wind_strength": self.wind_strength,
            "gust_probability": self.gust_probability,
            "gust_multiplier": self.gust_multiplier,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ContinuousWindPlugin":
        cfg = dict(config)
        wd = cfg.get("wind_direction")
        if isinstance(wd, list):
            cfg["wind_direction"] = np.array(wd, dtype=np.float64)
        return cls(**cfg)

    def set_episode_seed(self, seed: int) -> None:
        """Rebuild the per-plugin RNG immediately (see framework/SEED.md)."""
        self._rng = np.random.RandomState(int(seed))

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
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robot: 目标机器人 ID
            strike_force: 打击力大小（牛顿）
            strike_interval: 打击间隔
            random_direction: 是否使用随机方向
            random_seed: 随机种子（随机方向采样用）
        """
        self.target_robot = target_robot
        self.strike_force = strike_force
        self.strike_interval = strike_interval
        self.random_direction = random_direction

        self._step_count = 0
        # RNG is rebuilt on every set_episode_seed() call.
        self._rng = np.random.RandomState(random_seed)

    def set_episode_seed(self, seed: int) -> None:
        """Rebuild the per-plugin RNG immediately (see framework/SEED.md)."""
        self._rng = np.random.RandomState(int(seed))

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "strike_force": self.strike_force,
            "strike_interval": self.strike_interval,
            "random_direction": self.random_direction,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "HeadStrikePlugin":
        return cls(**config)

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


class RandomFallenStatePlugin(BasePlugin):
    """随机摔倒状态初始化插件。

    在 ``on_pre_episode`` 时，通过内部仿真实例让指定机器人从当前姿态
    随机倒下并静止，然后将摔倒后的核心状态写回真实环境。

    工作流程：
    1. 读取真实环境的当前 core state 作为内部 sim 的初始状态。
    2. 给目标机器人设置随机 action（uniform[-1, 1]）。
    3. 非目标机器人每隔 ``reset_interval`` 物理步重置回初始状态，
       防止其倒下后干扰目标机器人的摔倒轨迹。
    4. 循环执行物理步，直到目标机器人高度低于阈值或达到 ``max_phy_steps``。
    5. 取摔倒后的 core state 写回真实环境。
    """

    def __init__(
        self,
        target_robots: str | Sequence[str] = "robot_a",
        max_phy_steps: int = 1000,
        height_threshold: float = 0.3,
        reset_interval: int = 50,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robots: 要初始化的机器人，``"robot_a"``、``"robot_b"`` 或 ``"both"``。
            max_phy_steps: 内部仿真最多跑多少物理步。
            height_threshold: 目标机器人 root 高度低于此值时提前终止 (m)。
            reset_interval: 每隔多少物理步重置非目标机器人回初始状态。
            random_seed: 随机种子。
        """
        if isinstance(target_robots, str):
            if target_robots == "both":
                self._target_set = {"robot_a", "robot_b"}
            else:
                self._target_set = {target_robots}
        else:
            self._target_set = set(target_robots)

        for rid in self._target_set:
            if rid not in ("robot_a", "robot_b"):
                raise ValueError(f"Invalid target_robot: {rid}")

        self.max_phy_steps = int(max_phy_steps)
        self.height_threshold = float(height_threshold)
        self.reset_interval = int(reset_interval)
        self._rng = np.random.RandomState(random_seed)

        self._internal_sim: Optional[Any] = None

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return "random_fallen_state"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robots": sorted(self._target_set) if len(self._target_set) > 1 else list(self._target_set)[0],
            "max_phy_steps": self.max_phy_steps,
            "height_threshold": self.height_threshold,
            "reset_interval": self.reset_interval,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "RandomFallenStatePlugin":
        return cls(**config)

    def _ensure_internal_sim(self) -> Any:
        if self._internal_sim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            self._internal_sim = Humanoid21Simulator()
        return self._internal_sim

    def on_pre_episode(self, ctx: SimContext) -> None:
        sim = self._ensure_internal_sim()

        _debug = os.environ.get("COMBATBENCH_FALL_DEBUG", "0") == "1"
        _debug_dir = Path(os.environ.get("COMBATBENCH_FALL_DEBUG_DIR", "/tmp/fall_debug"))

        def _debug_img(tag: str, step_num: int = -1):
            if not _debug:
                return
            _debug_dir.mkdir(parents=True, exist_ok=True)
            try:
                img = sim.get_broadcastview_image()
                from PIL import Image
                core = sim.get_core_state()
                h = float(core["robot_a"]["root_pos"][2])
                fname = f"{tag}_s{step_num:04d}_h{h:.3f}.png" if step_num >= 0 else f"{tag}.png"
                Image.fromarray(img).save(str(_debug_dir / fname))
                print(f"[fall_debug] saved {fname} (height={h:.4f})", flush=True)
            except Exception as e:
                print(f"[fall_debug] render failed: {e}", flush=True)

        # 1. 读取真实环境的当前 core state
        real_state = ctx.accessor.get_core_state()

        # 2. 初始化内部 sim（reset 到站姿，然后写入真实状态）
        sim.reset()
        sim.set_core_state(real_state)
        _debug_img("00_initial")

        # 3. 保存非目标机器人的初始状态（用于定期重置）
        non_target_state = {
            rid: {k: v.copy() for k, v in state.items()}
            for rid, state in real_state.items()
            if rid not in self._target_set
        }

        # 4. 给目标机器人设置随机 action
        random_action = {}
        for rid in ("robot_a", "robot_b"):
            if rid in self._target_set:
                random_action[rid] = self._rng.uniform(
                    -1.0, 1.0, size=(21,)
                ).astype(np.float32)
            else:
                random_action[rid] = real_state[rid].get(
                    "joint_pos_norm",
                    np.zeros(21, dtype=np.float32),
                )
        sim.set_action(random_action)
        _debug_img("01_after_set_action")

        # 5. 循环物理步
        _debug_milestones = {1, 5, 10, 25, 50, 100, 200, 300, 500, 1000, 1500, 2000, 2500}
        step = -1
        min_height = float("inf")
        for step in range(self.max_phy_steps):
            sim.physical_step()

            # 定期重置非目标机器人
            if non_target_state and (step + 1) % self.reset_interval == 0:
                sim.set_core_state(non_target_state)

            # 检查目标机器人高度是否低于阈值
            core = sim.get_core_state()
            min_height = min(
                (float(core[rid]["root_pos"][2]) for rid in self._target_set if rid in core),
                default=float("inf"),
            )

            if _debug and (step + 1) in _debug_milestones:
                _debug_img("step", step + 1)

            if min_height < self.height_threshold:
                break

        if _debug:
            _debug_img("99_final", step + 1)
            print(f"[fall_debug] total_steps={step+1} final_height={min_height:.4f}", flush=True)

        # 6. 取最终 core state 写回真实环境
        fallen_state = sim.get_core_state()
        result_state = {}
        for rid in self._target_set:
            if rid in fallen_state:
                result_state[rid] = fallen_state[rid]

        ctx.mutator.set_core_state(result_state)

        # 记录 metrics
        for rid in self._target_set:
            if rid in fallen_state:
                ctx.metrics[f"{rid}_fallen_init_steps"] = step + 1
                ctx.metrics[f"{rid}_fallen_init_height"] = float(fallen_state[rid]["root_pos"][2])
                ctx.metrics[f"{rid}_fallen_init_height_threshold"] = min_height < self.height_threshold


class ImpulsePerturbationPlugin(BasePlugin):
    """冲量扰动插件 — 用内部仿真器 + 策略生成物理一致的扰动后状态。

    工作流程（参照 ``RandomFallenStatePlugin`` 的内部 sim 模式）：
    1. ``on_pre_episode`` 时读取真实环境的 core state，写入内部 sim。
    2. 在内部 sim 中，每物理步：策略出 action → ``set_action`` →
       ``apply_external_force`` → ``physical_step``。
    3. 持续 ``duration_action_steps × phy_steps_per_action`` 个物理步后，
       取 core state 写回真实环境。

    两种参数来源（优先级从高到低）：
    - **固定模式**：``ctx.episode_options["impulse_params"]`` 指定确切参数。
    - **随机模式**：用构造器传入的范围由 RNG 采样。

    策略通过 ``PolicyBlueprint`` 文件路径惰性加载。
    """

    def __init__(
        self,
        target_robot: str = "robot_a",
        policy_blueprint_path: Optional[str] = None,
        impulse_body: str = "torso",
        force_magnitude: float | Sequence[float] = (100, 500),
        duration_action_steps: int | Sequence[int] = (1, 8),
        direction_mode: str = "random_horizontal",
        fixed_direction: Optional[Sequence[float]] = None,
        phy_steps_per_action: int = 25,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            target_robot: 目标机器人 ID。
            policy_blueprint_path: 策略 blueprint YAML 文件路径。
                用于在内部 sim 中控制机器人。如果为 None，则用零 action
                （PD 控制器拉回默认站姿）。
            impulse_body: 冲量作用的 body 名称（如 'torso', 'head'）。
            force_magnitude: 力的大小 (N)。标量=固定值，(min, max)=随机采样范围。
            duration_action_steps: 持续时间 (动作步)。标量=固定值，(min, max)=随机采样范围。
            direction_mode: 'random_horizontal'（水平面随机方向）或
                'fixed'（使用 fixed_direction）。
            fixed_direction: direction_mode='fixed' 时使用的方向向量 [x, y, z]。
            phy_steps_per_action: 每动作步的物理步数（与蓝图 runtime 配置一致）。
            random_seed: 随机种子。
        """
        self.target_robot = target_robot
        self.policy_blueprint_path = policy_blueprint_path
        self.impulse_body = impulse_body
        if isinstance(force_magnitude, (int, float)):
            self.force_magnitude_range = (float(force_magnitude), float(force_magnitude))
        else:
            self.force_magnitude_range = (float(force_magnitude[0]),
                                           float(force_magnitude[1]))
        if isinstance(duration_action_steps, int):
            self.duration_action_steps_range = (duration_action_steps, duration_action_steps)
        else:
            self.duration_action_steps_range = (int(duration_action_steps[0]),
                                                 int(duration_action_steps[1]))
        self.direction_mode = direction_mode
        self.fixed_direction = (np.asarray(fixed_direction, dtype=np.float64)
                                if fixed_direction is not None else None)
        self.phy_steps_per_action = int(phy_steps_per_action)
        self._rng = np.random.RandomState(random_seed)
        self._internal_sim: Optional[Any] = None
        self._policy: Optional[Any] = None

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return "impulse_perturbation"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "policy_blueprint_path": self.policy_blueprint_path,
            "impulse_body": self.impulse_body,
            "force_magnitude": list(self.force_magnitude_range),
            "duration_action_steps": list(self.duration_action_steps_range),
            "direction_mode": self.direction_mode,
            "fixed_direction": (self.fixed_direction.tolist()
                                if self.fixed_direction is not None else None),
            "phy_steps_per_action": self.phy_steps_per_action,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ImpulsePerturbationPlugin":
        return cls(**config)

    def _ensure_internal_sim(self) -> Any:
        if self._internal_sim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            self._internal_sim = Humanoid21Simulator()
        return self._internal_sim

    def _ensure_policy(self) -> Any:
        if self._policy is None and self.policy_blueprint_path is not None:
            from envs.framework.policy import PolicyBlueprint
            bp = PolicyBlueprint.load(Path(self.policy_blueprint_path))
            self._policy = bp.build()
        return self._policy

    def _sample_direction(self) -> np.ndarray:
        if self.direction_mode == "fixed" and self.fixed_direction is not None:
            d = self.fixed_direction.copy()
        else:
            angle = self._rng.uniform(0, 2 * np.pi)
            d = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        return d

    def _resolve_params(self, ctx: SimContext) -> Dict[str, Any]:
        params = ctx.episode_options.get("impulse_params", None)
        if params is not None:
            return {
                "body": params["impulse_body"],
                "direction": np.asarray(params["impulse_direction"], dtype=np.float64),
                "force": float(params["impulse_force"]),
                "duration_action_steps": int(params["impulse_duration_steps"]),
            }
        return {
            "body": self.impulse_body,
            "direction": self._sample_direction(),
            "force": float(self._rng.uniform(*self.force_magnitude_range)),
            "duration_action_steps": int(self._rng.randint(
                self.duration_action_steps_range[0],
                self.duration_action_steps_range[1] + 1,
            )),
        }

    def on_pre_episode(self, ctx: SimContext) -> None:
        sim = self._ensure_internal_sim()
        policy = self._ensure_policy()

        other_robot = "robot_b" if self.target_robot == "robot_a" else "robot_a"

        params = self._resolve_params(ctx)
        body = params["body"]
        direction = params["direction"]
        force = params["force"]
        duration_action_steps = params["duration_action_steps"]
        duration_phy_steps = duration_action_steps * self.phy_steps_per_action

        # 1. 读取真实环境当前 core state，初始化内部 sim
        real_state = ctx.accessor.get_core_state()
        sim.reset()
        sim.set_core_state(real_state)

        # 2. 策略 reset
        if policy is not None:
            policy.reset(seed=int(self._rng.randint(0, 2**31 - 1)))

        # 保存非目标机器人的初始状态（定期重置，防止干扰）
        non_target_state = {
            rid: {k: v.copy() for k, v in state.items()}
            for rid, state in real_state.items()
            if rid != self.target_robot
        }

        # 3. 施力 + 策略控制 + 物理步
        for i in range(duration_phy_steps):
            # 策略出 action
            if policy is not None:
                obs = sim.get_observation()
                action, _ = policy.act(obs.get(self.target_robot))
                sim.set_action({
                    self.target_robot: action,
                    other_robot: np.zeros(21, dtype=np.float32),
                })
            else:
                sim.set_action({
                    rid: np.zeros(21, dtype=np.float32)
                    for rid in ("robot_a", "robot_b")
                })

            # 施加外力（每步重新施加，因为 physical_step 会清零 xfrc_applied）
            sim.apply_external_force(
                body_name=body,
                force=direction * force,
                robot_id=self.target_robot,
            )
            sim.physical_step()

            # 定期重置非目标机器人
            if non_target_state and (i + 1) % self.phy_steps_per_action == 0:
                sim.set_core_state(non_target_state)

        # 4. 取扰动后的 core state 写回真实环境
        perturbed_state = sim.get_core_state()
        ctx.mutator.set_core_state({
            self.target_robot: perturbed_state[self.target_robot],
        })

        # 5. 记录元数据到 metrics
        ctx.metrics[f"{self.target_robot}_impulse_body"] = body
        ctx.metrics[f"{self.target_robot}_impulse_force"] = force
        ctx.metrics[f"{self.target_robot}_impulse_duration_action_steps"] = duration_action_steps
        ctx.metrics[f"{self.target_robot}_impulse_duration_phy_steps"] = duration_phy_steps
        ctx.metrics[f"{self.target_robot}_impulse_direction"] = direction.tolist()


class ConstantForcePlugin(BasePlugin):
    """恒定外力插件 — 在 episode 开始后持续施力指定步数。

    通过 ``on_pre_phy_step`` 钩子在每个物理步施加恒定外力，
    持续 ``duration_action_steps`` 个 action 步后自动停止。
    与 ``EnvRuntime`` 配合使用，由 ``EnvRuntime`` 管理 action 步/物理步节奏。

    方向处理（二维水平面）：
        direction 参数为相对机器人朝向的角度（度）：
        0°=向前, 90°=向右, 180°=向后, 270°=向左。

        heading 提取方式：将局部 forward [1,0,0] 通过四元数旋转到世界坐标，
        取 atan2(forward_y, forward_x)，丢弃 z 分量。
        方向向量为 [cos(abs_angle), sin(abs_angle), 0]，z 分量恒为 0。

        前提假设：机器人在施力时基本直立（pitch/roll ≈ 0）。
        当机器人趴在地上（pitch ≈ 90°）时，forward 在水平面投影接近零向量，
        heading 变得不稳定/无意义。但方向向量在 on_pre_episode 时已计算好，
        后续不变，因此不影响扰动过程。
        如需在非直立状态下使用（如连续扰动中再次施力），需要考虑三维方向处理。
    """

    def __init__(
        self,
        agent_id: str = "robot_a",
        force: float = 100.0,
        direction: float = 0.0,
        duration_action_steps: int = 4,
        body_name: str = "torso",
    ):
        """
        Args:
            agent_id: 目标机器人 ID（'robot_a' 或 'robot_b'）。
            force: 力的大小（牛顿）。
            direction: 相对角度（度），0°=向前, 90°=向右, 180°=向后, 270°=向左。
            duration_action_steps: 持续时间（action 步数）。
            body_name: 受力部位名称，必须在 ROBOT_BODY_TREE 中定义。
        """
        from envs.humanoid21.meta import Humanoid21Meta

        valid_bodies = list(Humanoid21Meta.ROBOT_BODY_TREE.keys())
        if body_name not in valid_bodies:
            raise ValueError(
                f"body_name must be one of {valid_bodies}, got {body_name!r}"
            )

        self.agent_id = agent_id
        self.force = float(force)
        self.direction = float(direction)
        self.duration_action_steps = int(duration_action_steps)
        self.body_name = body_name

        self._direction_vec: Optional[np.ndarray] = None
        self._remaining_action_steps = 0
        self._active = False

    @property
    def name(self) -> str:
        return "constant_force"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "force": self.force,
            "direction": self.direction,
            "duration_action_steps": self.duration_action_steps,
            "body_name": self.body_name,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ConstantForcePlugin":
        return cls(**config)

    @staticmethod
    def _extract_heading(root_rot: np.ndarray) -> float:
        """从 root_rot 四元数 [w,x,y,z] 提取 heading（yaw, 弧度）。

        heading = atan2(forward_y, forward_x)，其中 forward = R @ [1,0,0]。
        """
        rot = R.from_quat([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])
        forward = rot.apply(np.array([1.0, 0.0, 0.0]))
        return float(np.arctan2(forward[1], forward[0]))

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Episode 开始时初始化施力状态。

        支持通过 ``episode_options["impulse_params"]`` 按 episode 覆盖参数。
        格式::

            {"robot_a": {"force": 100, "direction_angle": 45,
                          "duration_action_steps": 4, "body": "torso"}}

        其中 ``direction_angle`` 为相对角度（度），会赋给 ``self.direction``。
        """
        params = ctx.episode_options.get("impulse_params", {})
        per_agent = params.get(self.agent_id, {})
        if per_agent:
            self.force = float(per_agent.get("force", self.force))
            self.direction = float(per_agent.get("direction_angle", per_agent.get("direction", self.direction)))
            self.duration_action_steps = int(per_agent.get("duration_action_steps", self.duration_action_steps))
            body = per_agent.get("body", per_agent.get("body_name"))
            if body is not None:
                self.body_name = str(body)

        self._direction_vec = None  # 延迟到 on_pre_action_step 计算
        self._remaining_action_steps = self.duration_action_steps
        self._active = True

    def on_pre_action_step(self, ctx: SimContext) -> None:
        if self._direction_vec is None:
            core_state = ctx.accessor.get_core_state()
            root_rot = np.asarray(core_state[self.agent_id]["root_rot"], dtype=np.float64)
            heading = self._extract_heading(root_rot)
            abs_angle = heading - np.radians(self.direction)
            self._direction_vec = np.array(
                [np.cos(abs_angle), np.sin(abs_angle), 0.0], dtype=np.float64
            )

            ctx.metrics[f"{self.agent_id}_impulse_body"] = self.body_name
            ctx.metrics[f"{self.agent_id}_impulse_force"] = self.force
            ctx.metrics[f"{self.agent_id}_impulse_direction_angle"] = self.direction
            ctx.metrics[f"{self.agent_id}_impulse_duration_action_steps"] = self.duration_action_steps
            ctx.metrics[f"{self.agent_id}_impulse_direction"] = self._direction_vec.tolist()
            ctx.metrics[f"{self.agent_id}_impulse_heading"] = heading

        if self._remaining_action_steps <= 0:
            self._active = False

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        if not self._active or self._direction_vec is None:
            return
        ctx.mutator.apply_external_force(
            body_name=self.body_name,
            force=self._direction_vec * self.force,
            robot_id=self.agent_id,
        )

    def on_post_action_step(self, ctx: SimContext) -> None:
        if self._remaining_action_steps > 0:
            self._remaining_action_steps -= 1


# ============================================================
# 状态捕获插件 + 观察器（用于状态池生成）
# ============================================================


class EpisodeEndCaptureObserver(BaseRuntimeUnit):
    """在每个 action step 的 ``on_post_action_step`` 中直接从
    ``ctx.accessor`` 读取 core_state + observation，并覆盖 ``self._output``。

    ``EpisodeBufferRecorder`` 只在 ``on_post_action_step`` 时调用
    ``get_output()`` 缓存帧，因此必须在此 hook 中写入（不能用
    ``on_post_episode``，因为 recorder 不会在该 hook 中抓 observer
    outputs）。每步覆盖确保最后一帧保存的是 episode-end 状态。

    同时从 ``ctx.metrics`` 读取 impulse 元数据（由 ``ConstantForcePlugin``
    等写入），无需额外 plugin 中转。
    """

    def __init__(self, target_robot: str = "robot_a") -> None:
        self.target_robot = target_robot
        self._output: Dict[str, Any] = {}

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = {}

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        robot_cs = core_state[self.target_robot]

        obs = ctx.accessor.get_observation()
        robot_obs = obs.get(self.target_robot)

        metrics = ctx.metrics
        self._output = {
            "core_state": {k: np.asarray(v).copy() for k, v in robot_cs.items()},
            "observation": np.asarray(robot_obs, dtype=np.float32).copy()
            if robot_obs is not None else None,
            "impulse_force": metrics.get(f"{self.target_robot}_impulse_force"),
            "impulse_duration": metrics.get(
                f"{self.target_robot}_impulse_duration_action_steps"
            ),
            "impulse_direction": metrics.get(f"{self.target_robot}_impulse_direction"),
            "impulse_direction_angle": metrics.get(
                f"{self.target_robot}_impulse_direction_angle"
            ),
        }

    def get_output(self) -> Any:
        return self._output if self._output else {}


class StateCapturePlugin(BasePlugin):
    """在第一个 action step 的 on_pre_action_step 中捕获扰动后的
    core_state + observation，写入 ctx.metrics。

    此时 ImpulsePerturbationPlugin 已经在 on_pre_episode 中完成扰动，
    物理步尚未执行，状态即为扰动后初始状态。

    需要配合 StateCaptureObserver 使用：observer 在 on_post_action_step
    中从 ctx.metrics 读取数据并通过 get_output() 暴露给 EpisodeRecorder。
    """

    def __init__(self, target_robot: str = "robot_a") -> None:
        self.target_robot = target_robot
        self._captured = False

    def to_blueprint(self) -> Dict[str, Any]:
        return {"target_robot": self.target_robot}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StateCapturePlugin":
        return cls(**config)

    @property
    def name(self) -> str:
        return f"{self.target_robot}_state_capture"

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._captured = False

    def on_pre_action_step(self, ctx: SimContext) -> None:
        if self._captured:
            return
        self._captured = True

        core_state = ctx.accessor.get_core_state()
        robot_cs = core_state[self.target_robot]
        ctx.metrics[f"{self.target_robot}_captured_core_state"] = {
            k: v.copy() for k, v in robot_cs.items()
        }

        obs = ctx.accessor.get_observation()
        robot_obs = obs.get(self.target_robot)
        if robot_obs is not None:
            ctx.metrics[f"{self.target_robot}_captured_observation"] = np.asarray(
                robot_obs, dtype=np.float32
            ).copy()


class StateCaptureObserver(BaseRuntimeUnit):
    """将 StateCapturePlugin 写入 ctx.metrics 的捕获数据通过 observer_outputs
    暴露出来，使其出现在 Episode.observer_outputs["state_capture"] 中。

    on_post_action_step 从 ctx.metrics 读取一次，缓存并在 get_output() 返回。
    """

    def __init__(self, target_robot: str = "robot_a") -> None:
        self.target_robot = target_robot
        self._output: Dict[str, Any] = {}

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = {}

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        if self._output:
            return
        metrics = ctx.metrics
        prefix = f"{self.target_robot}_captured_"
        cs = metrics.get(f"{prefix}core_state")
        obs = metrics.get(f"{prefix}observation")
        if cs is not None:
            self._output = {
                "core_state": {k: np.asarray(v).copy() for k, v in cs.items()},
                "observation": np.asarray(obs).copy() if obs is not None else None,
                "impulse_force": metrics.get(f"{self.target_robot}_impulse_force"),
                "impulse_duration": metrics.get(
                    f"{self.target_robot}_impulse_duration_action_steps"
                ),
                "impulse_direction": metrics.get(f"{self.target_robot}_impulse_direction"),
            }

    def get_output(self) -> Any:
        return self._output if self._output else {}


class StateBankInitPlugin(BasePlugin):
    """从 .npz 状态池加载状态，在 on_pre_episode 中注入到 sim。

    通用化设计：只负责加载 ``states`` 数组并按 ``state_bank_index``
    注入 core_state，不读取任何 metadata 字段。

    工作流：
    1. 加载 .npz 文件中的 ``states`` 数组
    2. on_pre_episode 时，按 episode_options 中的 state_bank_index
       指定索引（或随机采样）
    3. 用 ctx.mutator.set_core_state 注入到 sim
    """

    CORE_STATE_FIELDS = [
        "root_pos", "root_rot", "root_vel_local",
        "root_angular_vel_local", "joint_pos_norm", "joint_vel_norm",
    ]
    CORE_STATE_DIMS = [3, 4, 3, 3, 21, 21]

    def __init__(
        self,
        state_bank_path: str,
        target_robot: str = "robot_a",
        seed: int = 42,
    ) -> None:
        self.state_bank_path = str(state_bank_path)
        self.target_robot = target_robot
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._bank: Optional[Dict[str, np.ndarray]] = None
        self._current_index: int = -1

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "state_bank_path": self.state_bank_path,
            "target_robot": self.target_robot,
            "seed": self._seed,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StateBankInitPlugin":
        return cls(**config)

    @property
    def name(self) -> str:
        return f"{self.target_robot}_state_bank_init"

    @property
    def require_mutator(self) -> bool:
        return True

    def _load_bank(self) -> None:
        if self._bank is not None:
            return
        data = np.load(self.state_bank_path, allow_pickle=True)
        self._bank = {
            "states": data["states"].astype(np.float32),
        }

    def _unflatten_state(self, vec: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        offset = 0
        for name, dim in zip(self.CORE_STATE_FIELDS, self.CORE_STATE_DIMS):
            out[name] = vec[offset:offset + dim].astype(np.float32)
            offset += dim
        return out

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._load_bank()

        idx = ctx.episode_options.get("state_bank_index")
        if idx is not None:
            idx = int(idx)
        else:
            idx = int(self._rng.randint(0, len(self._bank["states"])))
        self._current_index = idx

        state_vec = self._bank["states"][idx]
        robot_state = self._unflatten_state(state_vec)

        full_state = {self.target_robot: robot_state}
        ctx.mutator.set_core_state(full_state)

        ctx.metrics[f"{self.target_robot}_state_bank_index"] = idx


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

    print("\n5. 随机摔倒状态初始化:")
    print("   plugin = RandomFallenStatePlugin(")
    print("       target_robots='robot_a',  # 或 'robot_b' 或 'both'")
    print("       # 随机 action 从 uniform[-1, 1] 采样")
    print("       max_phy_steps=500,       # 最多跑500物理步")
    print("       height_threshold=0.3,    # 高度低于此值时提前终止 (m)")
    print("       reset_interval=50,       # 每50步重置非目标机器人")
    print("   )")

    print("\n6. 头部打击:")
    print("   plugin = HeadStrikePlugin(")
    print("       target_robot='robot_a',")
    print("       strike_force=400.0,")
    print("       strike_interval=200")
    print("   )")
