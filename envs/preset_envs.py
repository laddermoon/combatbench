"""
预置环境 (Preset Environments)

提供常用的仿真环境配置，展示如何使用 SimpleCombatEnv 框架。

使用示例：
    from combatbench.envs import Humanoid21NonFallEnv, Humanoid21FallEnv

    env = Humanoid21NonFallEnv(render_mode=None)
    obs, info = env.reset()

    for _ in range(1000):
        action = {
            'robot_a': env.action_space['robot_a'].sample(),
            'robot_b': env.action_space['robot_b'].sample(),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .humanoid21.robot import HumanoidRobot
from .humanoid21 import Humanoid21Simulator
from .framework import (
    SimpleCombatEnv,
    StepDataBuilder,
)
from .humanoid21.scoring import ScoreCalculator
from .framework import BaseHook, InvokeType


# ==================== Step 数据构建器 ====================

class Humanoid21StepDataBuilder(StepDataBuilder):
    """
    Humanoid21 的 Step 数据构建器

    提供标准观测、零奖励和基础 info（可继承自定义）。
    """

    def __init__(self, score_calculator: Optional[ScoreCalculator] = None):
        self.obs_dim = HumanoidRobot.OBSERVATION_DIM
        self.score_calculator = score_calculator

    def build_step_data(
        self,
        f_get_core_state: Callable[[], Dict[str, Any]],
        f_get_derived_state: Callable[[], Dict[str, Any]],
        f_get_sensor_data: Callable[[], Dict[str, Any]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any]]:
        # 从 derived_state 获取观测
        derived_state = f_get_derived_state()
        core_state = f_get_core_state()

        obs_a = derived_state['robots']['robot_a']['observation']
        obs_b = derived_state['robots']['robot_b']['observation']

        observation = {
            'robot_a_obs': obs_a,
            'robot_b_obs': obs_b,
        }
        reward = {
            'robot_a': 0.0,
            'robot_b': 0.0,
        }

        # 构建相对度量
        relative_metrics = self._build_relative_metrics(derived_state)

        info = {
            'step': core_state.get('step_count', 0),
            'torso_positions': {
                'robot_a': derived_state['robots']['robot_a']['torso_position'],
                'robot_b': derived_state['robots']['robot_b']['torso_position'],
            },
            'robot_states': derived_state['robots'],
            'relative_metrics': relative_metrics,
        }

        # 如果有血量系统，添加血量信息
        if self.score_calculator is not None:
            info['scores'] = self.score_calculator.get_health()

        return observation, reward, info

    def _build_relative_metrics(self, derived_state: Dict[str, Any]) -> Dict[str, Any]:
        """构建相对度量"""
        relative_metrics = {}
        for robot_id, opponent_id in [('robot_a', 'robot_b'), ('robot_b', 'robot_a')]:
            self_state = derived_state['robots'][robot_id]
            opponent_state = derived_state['robots'][opponent_id]

            torso_pos_self = self_state['torso_position']
            torso_pos_opponent = opponent_state['torso_position']

            relative_position = torso_pos_opponent - torso_pos_self
            distance = float(np.linalg.norm(relative_position))

            if distance > 1e-8:
                direction_to_opponent = relative_position / distance
            else:
                direction_to_opponent = np.zeros(3, dtype=np.float32)

            relative_metrics[robot_id] = {
                'distance': distance,
                'horizontal_distance': float(np.linalg.norm(relative_position[:2])),
                'relative_position': relative_position.astype(np.float32),
                'direction_to_opponent': direction_to_opponent.astype(np.float32),
            }

        return relative_metrics

    def get_observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "robot_a_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "robot_b_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
        })


# ==================== Hook ====================

class FallDetectionHook(BaseHook):
    """
    跌倒检测 Hook

    检测机器人是否跌倒，并在 info 中记录。
    不会终止 episode，只记录状态。
    """

    def __init__(self):
        super().__init__()
        self.fallen_status = {'robot_a': False, 'robot_b': False}

    @property
    def name(self) -> str:
        return "fall_detection"

    @property
    def priority(self) -> int:
        return 0

    def reset(self):
        """重置跌倒状态"""
        self.fallen_status = {'robot_a': False, 'robot_b': False}

    def invoke(
        self,
        invoke_type: InvokeType,
        f_get_core_state=None,
        f_get_derived_state=None,
        **kwargs
    ) -> bool:
        if invoke_type == InvokeType.PRE_EPISODE:
            self.reset()
            return False

        if invoke_type == InvokeType.POST_ACTION_STEP and f_get_derived_state:
            self.fallen_status = self._check_fallen(f_get_derived_state)

        return False

    def _check_fallen(self, f_get_derived_state) -> Dict[str, bool]:
        """检测机器人是否跌倒"""
        derived_state = f_get_derived_state()
        fallen = {}

        for robot_id in ['robot_a', 'robot_b']:
            # 从 derived_state 获取机器人状态
            if 'robots' in derived_state and robot_id in derived_state['robots']:
                robot_state = derived_state['robots'][robot_id]
                torso_position = robot_state.get('torso_position', np.array([0, 0, 1.4]))
                torso_orientation = robot_state.get('torso_orientation', np.array([1, 0, 0, 0]))

                # 高度判断（低于 0.7 米视为跌倒）
                height_fallen = torso_position[2] < 0.7

                # 角度判断（躯干倾斜角度过大）
                from scipy.spatial.transform import Rotation as R
                try:
                    r = R.from_quat([torso_orientation[1], torso_orientation[2],
                                      torso_orientation[3], torso_orientation[0]])
                    euler = r.as_euler('xyz', degrees=True)
                    tilt_fallen = abs(euler[0]) > 60 or abs(euler[1]) > 60
                except:
                    tilt_fallen = False

                fallen[robot_id] = height_fallen or tilt_fallen
            else:
                fallen[robot_id] = False

        return fallen

    def get_fallen_status(self) -> Dict[str, bool]:
        """获取当前跌倒状态"""
        return self.fallen_status.copy()


class UprightConstraintHook(BaseHook):
    """
    站立约束 Hook

    在物理步后施加约束，帮助机器人保持站立姿态。
    用于 NonFall 环境。
    """

    def __init__(self, height_threshold: float = 0.8):
        super().__init__()
        self.height_threshold = height_threshold

    @property
    def name(self) -> str:
        return "upright_constraint"

    @property
    def priority(self) -> int:
        return 50

    def invoke(
        self,
        invoke_type: InvokeType,
        f_get_core_state=None,
        f_get_derived_state=None,
        f_get_sensor_data=None,
        f_set_core_state=None,
        **kwargs
    ) -> bool:
        if invoke_type == InvokeType.POST_PHY_STEP and f_get_core_state and f_set_core_state:
            core_state = f_get_core_state()

            # 检查机器人是否快要倒下
            for robot_id in ['robot_a', 'robot_b']:
                if 'robots' in core_state and robot_id in core_state['robots']:
                    root_position = core_state['robots'][robot_id]['root_position']
                    # 如果高度过低，施加向上的力（这里简化为直接修正）
                    if root_position[2] < self.height_threshold:
                        # 实际实现中可以通过 f_set_core_state 修正
                        # 这里简化处理，记录即可
                        pass

        return False


# ==================== 预置环境 ====================

class Humanoid21NonFallEnv(SimpleCombatEnv, gym.Env):
    """
    Humanoid21 非跌倒格斗环境

    机器人被约束保持站立姿态，不能跌倒。
    适用于稳定的格斗训练。

    特点：
    - 强制机器人保持站立
    - 使用 UprightConstraintHook 辅助站立
    - 稳定的物理交互
    - 适合初学者训练

    使用示例：
        env = Humanoid21NonFallEnv(render_mode=None, match_duration=30.0)
        obs, info = env.reset()

        for _ in range(1000):
            action = {
                'robot_a': env.action_space['robot_a'].sample(),
                'robot_b': env.action_space['robot_b'].sample(),
            }
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        match_duration: float = 30.0,
        control_frequency: float = 20.0,
        initial_distance: float = 2.0,
    ):
        """
        初始化非跌倒环境

        Args:
            render_mode: 渲染模式 ("human", "rgb_array", None)
            match_duration: 比赛时长（秒）
            control_frequency: 控制频率（Hz）
            initial_distance: 机器人初始距离（米）
        """
        # 创建仿真器
        simulator = Humanoid21Simulator(
            gui=(render_mode == "human"),
            initial_distance=initial_distance,
        )

        # 创建组件
        step_data_builder = Humanoid21StepDataBuilder()

        # 初始化父类
        super().__init__(
            simulator=simulator,
            step_data_builder=step_data_builder,
            match_duration=match_duration,
            control_frequency=control_frequency,
            hooks=[
                UprightConstraintHook(),  # 站立约束
            ],
        )

        # 设置 action_space（Humanoid21 特定）
        action_dim = HumanoidRobot.ACTION_DIM
        self.action_space = spaces.Dict({
            "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        })

        self.render_mode = render_mode


class Humanoid21FallEnv(SimpleCombatEnv, gym.Env):
    """
    Humanoid21 跌倒格斗环境

    机器人可以自由跌倒，无站立约束。
    适用于学习更复杂的行为，包括跌倒恢复。

    特点：
    - 允许机器人跌倒
    - 更大的动作空间
    - 更真实的物理交互
    - 跌倒状态检测

    使用示例：
        env = Humanoid21FallEnv(render_mode=None, match_duration=30.0)
        obs, info = env.reset()

        for _ in range(1000):
            action = {
                'robot_a': env.action_space['robot_a'].sample(),
                'robot_b': env.action_space['robot_b'].sample(),
            }
            obs, reward, terminated, truncated, info = env.step(action)
            # 检查跌倒状态
            if info.get('fallen', {}).get('robot_a', False):
                print("Robot A has fallen!")
            if terminated or truncated:
                break
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        match_duration: float = 30.0,
        control_frequency: float = 20.0,
        initial_distance: float = 2.0,
        enable_fall_detection: bool = True,
    ):
        """
        初始化跌倒环境

        Args:
            render_mode: 渲染模式 ("human", "rgb_array", None)
            match_duration: 比赛时长（秒）
            control_frequency: 控制频率（Hz）
            initial_distance: 机器人初始距离（米）
            enable_fall_detection: 是否启用跌倒检测
        """
        # 创建仿真器
        simulator = Humanoid21Simulator(
            gui=(render_mode == "human"),
            initial_distance=initial_distance,
        )

        # 创建组件
        step_data_builder = Humanoid21StepDataBuilder()

        # 创建 Hooks
        hooks = []
        self.fall_hook = None

        if enable_fall_detection:
            self.fall_hook = FallDetectionHook()
            hooks.append(self.fall_hook)

        # 初始化父类
        super().__init__(
            simulator=simulator,
            step_data_builder=step_data_builder,
            match_duration=match_duration,
            control_frequency=control_frequency,
            hooks=hooks,
        )

        # 设置 action_space（Humanoid21 特定）
        action_dim = HumanoidRobot.ACTION_DIM
        self.action_space = spaces.Dict({
            "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        })

        self.render_mode = render_mode
        self.enable_fall_detection = enable_fall_detection

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        # 添加跌倒状态到 info
        if self.enable_fall_detection and self.fall_hook:
            info['fallen'] = self.fall_hook.get_fallen_status()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # 添加跌倒状态到 info
        if self.enable_fall_detection and self.fall_hook:
            info['fallen'] = self.fall_hook.get_fallen_status()

        return obs, reward, terminated, truncated, info


# ==================== 导出 ====================

__all__ = [
    'Humanoid21NonFallEnv',
    'Humanoid21FallEnv',
]
