"""
Humanoid21 格斗环境

提供多种配置的 Gym 环境，支持单智能体和双智能体模式。
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .robot import HumanoidRobot
from . import Humanoid21Simulator
from ..framework import (
    SimpleCombatEnv,
    StepDataBuilder,
    BaseHook,
    InvokeType,
)
from .scoring import ScoreCalculator

# 导入策略基类
from ...policy.base import BaseCombatPolicy


# ==================== 单智能体数据构建器 ====================

class SingleAgentStepDataBuilder(StepDataBuilder):
    """
    单智能体 Step 数据构建器

    只返回机器人 A 的观测和奖励，用于单智能体 RL 训练。
    """

    def __init__(self, score_calculator: Optional[ScoreCalculator] = None):
        self.obs_dim = HumanoidRobot.OBSERVATION_DIM
        self.score_calculator = score_calculator

    def build_step_data(
        self,
        f_get_core_state: Callable[[], Dict[str, Any]],
        f_get_derived_state: Callable[[], Dict[str, Any]],
        f_get_sensor_data: Callable[[], Dict[str, Any]],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """构建单智能体观测、奖励和 info"""
        derived_state = f_get_derived_state()
        core_state = f_get_core_state()

        # 只返回机器人 A 的观测
        obs_a = derived_state['robots']['robot_a']['observation']

        # 零奖励（可继承覆盖）
        reward = 0.0

        # 构建相对度量
        relative_metrics = self._build_relative_metrics(derived_state)

        info = {
            'step': core_state.get('step_count', 0),
            'torso_position': derived_state['robots']['robot_a']['torso_position'],
            'opponent_position': derived_state['robots']['robot_b']['torso_position'],
            'robot_states': derived_state['robots'],
            'relative_metrics': relative_metrics['robot_a'],
        }

        # 如果有血量系统
        if self.score_calculator is not None:
            info['scores'] = self.score_calculator.get_health()

        return obs_a, reward, info

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
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)


# ==================== Hook ====================

class FallDetectionHook(BaseHook):
    """跌倒检测 Hook"""

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
        derived_state = f_get_derived_state()
        fallen = {}

        for robot_id in ['robot_a', 'robot_b']:
            if 'robots' in derived_state and robot_id in derived_state['robots']:
                robot_state = derived_state['robots'][robot_id]
                torso_position = robot_state.get('torso_position', np.array([0, 0, 1.4]))
                torso_orientation = robot_state.get('torso_orientation', np.array([1, 0, 0, 0]))

                height_fallen = torso_position[2] < 0.7

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
        return self.fallen_status.copy()


class UprightConstraintHook(BaseHook):
    """站立约束 Hook - 防止机器人跌倒"""

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

            for robot_id in ['robot_a', 'robot_b']:
                if 'robots' in core_state and robot_id in core_state['robots']:
                    root_position = core_state['robots'][robot_id]['root_position']
                    if root_position[2] < self.height_threshold:
                        # 这里可以施加修正力，简化实现暂时记录
                        pass

        return False


class FreezeRobotHook(BaseHook):
    """
    冻结机器人 Hook

    在每个物理步后重置机器人 B 的状态到初始位置，使其像雕塑一样不动。
    """

    def __init__(self, robot_id: str = 'robot_b'):
        super().__init__()
        self.robot_id = robot_id
        self._initial_qpos = None
        self._initial_qvel = None

    @property
    def name(self) -> str:
        return f"freeze_{self.robot_id}"

    @property
    def priority(self) -> int:
        return 100  # 高优先级，最后执行

    def invoke(
        self,
        invoke_type: InvokeType,
        f_get_core_state=None,
        f_get_derived_state=None,
        f_get_sensor_data=None,
        f_set_core_state=None,
        **kwargs
    ) -> bool:
        if invoke_type == InvokeType.PRE_EPISODE and f_get_core_state:
            # 保存初始状态
            core_state = f_get_core_state()
            if 'robots' in core_state and self.robot_id in core_state['robots']:
                self._initial_qpos = core_state['robots'][self.robot_id]['qpos'].copy()
                self._initial_qvel = core_state['robots'][self.robot_id]['qvel'].copy()

        elif invoke_type == InvokeType.POST_PHY_STEP and f_set_core_state:
            # 重置到初始状态
            if self._initial_qpos is not None and self._initial_qvel is not None:
                def set_state():
                    return {
                        'robots': {
                            self.robot_id: {
                                'qpos': self._initial_qpos.copy(),
                                'qvel': self._initial_qvel.copy(),
                            }
                        }
                    }
                f_set_core_state(set_state())

        return False


class OpponentPolicyHook(BaseHook):
    """
    对手策略 Hook

    在每个动作步自动应用对手策略，使环境变为单智能体模式。
    """

    def __init__(self, opponent_policy: BaseCombatPolicy, opponent_id: str = 'robot_b'):
        super().__init__()
        self.opponent_policy = opponent_policy
        self.opponent_id = opponent_id
        self._last_obs_b = None

    @property
    def name(self) -> str:
        return f"opponent_policy_{self.opponent_id}"

    @property
    def priority(self) -> int:
        return -100  # 低优先级，最先执行

    def invoke(
        self,
        invoke_type: InvokeType,
        f_get_core_state=None,
        f_get_derived_state=None,
        f_get_sensor_data=None,
        f_set_core_state=None,
        **kwargs
    ) -> bool:
        if invoke_type == InvokeType.PRE_ACTION_STEP and f_get_derived_state:
            derived_state = f_get_derived_state()
            if 'robots' in derived_state and self.opponent_id in derived_state['robots']:
                # 获取对手观测
                obs_dict = derived_state['robots'][self.opponent_id]['observation']
                self._last_obs_b = obs_dict

        return False

    def get_opponent_action(self) -> np.ndarray:
        """获取对手动作"""
        if self._last_obs_b is not None:
            return self.opponent_policy.act(self._last_obs_b)
        return np.zeros(HumanoidRobot.ACTION_DIM, dtype=np.float32)

    def reset(self):
        self._last_obs_b = None
        if hasattr(self.opponent_policy, 'reset'):
            self.opponent_policy.reset()


# ==================== 双智能体环境 ====================

class Humanoid21DualAgentEnv(SimpleCombatEnv, gym.Env):
    """
    Humanoid21 双智能体格斗环境

    两个机器人都受控，用于比赛或 Self-play。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        match_duration: float = 30.0,
        control_frequency: float = 20.0,
        initial_distance: float = 2.0,
        opponent_policy_b: Optional[BaseCombatPolicy] = None,
        enable_nonfall: bool = False,
        enable_fall_detection: bool = False,
    ):
        """
        初始化双智能体环境

        Args:
            render_mode: 渲染模式
            match_duration: 比赛时长（秒）
            control_frequency: 控制频率（Hz）
            initial_distance: 机器人初始距离（米）
            opponent_policy_b: 机器人B的策略（None表示双智能体，提供表示单智能体）
            enable_nonfall: 是否启用站立约束
            enable_fall_detection: 是否启用跌倒检测
        """
        simulator = Humanoid21Simulator(
            gui=(render_mode == "human"),
            initial_distance=initial_distance,
        )

        # 使用双智能体数据构建器
        step_data_builder = DualAgentStepDataBuilder()

        # 构建 Hooks
        hooks = []
        self.opponent_hook = None
        self.fall_hook = None

        # 对手策略 Hook
        if opponent_policy_b is not None:
            self.opponent_hook = OpponentPolicyHook(opponent_policy_b, 'robot_b')
            hooks.append(self.opponent_hook)

        # Nonfall Hook
        if enable_nonfall:
            hooks.append(UprightConstraintHook())

        # 跌倒检测 Hook
        if enable_fall_detection:
            self.fall_hook = FallDetectionHook()
            hooks.append(self.fall_hook)

        super().__init__(
            simulator=simulator,
            step_data_builder=step_data_builder,
            match_duration=match_duration,
            control_frequency=control_frequency,
            hooks=hooks,
        )

        # 设置空间
        action_dim = HumanoidRobot.ACTION_DIM

        if opponent_policy_b is None:
            # 双智能体模式
            self.action_space = spaces.Dict({
                "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
                "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            })
        else:
            # 单智能体模式（只控制 robot_a）
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self.render_mode = render_mode
        self.opponent_policy_b = opponent_policy_b
        self.enable_fall_detection = enable_fall_detection

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        if self.opponent_hook:
            self.opponent_hook.reset()

        if self.enable_fall_detection and self.fall_hook:
            info['fallen'] = self.fall_hook.get_fallen_status()

        return obs, info

    def step(self, action):
        # 单智能体模式：自动填充对手动作
        if self.opponent_policy_b is not None:
            opponent_action = self.opponent_hook.get_opponent_action()
            action = {
                'robot_a': np.asarray(action, dtype=np.float32),
                'robot_b': opponent_action,
            }

        obs, reward, terminated, truncated, info = super().step(action)

        if self.enable_fall_detection and self.fall_hook:
            info['fallen'] = self.fall_hook.get_fallen_status()

        return obs, reward, terminated, truncated, info


# ==================== 单智能体环境 ====================

class Humanoid21SingleAgentEnv(SimpleCombatEnv, gym.Env):
    """
    Humanoid21 单智能体格斗环境

    只控制机器人 A，机器人 B 使用固定策略。
    用于单智能体强化学习训练。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        match_duration: float = 30.0,
        control_frequency: float = 20.0,
        initial_distance: float = 2.0,
        opponent_policy: BaseCombatPolicy = None,
        opponent_type: str = 'standing',  # 'frozen', 'standing', 'policy'
        enable_nonfall: bool = False,
        enable_fall_detection: bool = False,
    ):
        """
        初始化单智能体环境

        Args:
            render_mode: 渲染模式
            match_duration: 比赛时长（秒）
            control_frequency: 控制频率（Hz）
            initial_distance: 机器人初始距离（米）
            opponent_policy: 对手策略（当 opponent_type='policy' 时使用）
            opponent_type: 对手类型
                - 'frozen': 机器人B完全冻结（雕塑模式）
                - 'standing': 机器人B站立不动（可被打倒）
                - 'policy': 使用指定策略
            enable_nonfall: 是否启用站立约束
            enable_fall_detection: 是否启用跌倒检测
        """
        simulator = Humanoid21Simulator(
            gui=(render_mode == "human"),
            initial_distance=initial_distance,
        )

        # 使用单智能体数据构建器
        step_data_builder = SingleAgentStepDataBuilder()

        # 构建 Hooks
        hooks = []
        self.fall_hook = None

        # 对手处理
        if opponent_type == 'frozen':
            hooks.append(FreezeRobotHook('robot_b'))
        elif opponent_type == 'standing' or opponent_type == 'policy':
            if opponent_policy is None:
                from ...policy.standing import StandingCombatPolicy
                opponent_policy = StandingCombatPolicy()
            self.opponent_hook = OpponentPolicyHook(opponent_policy, 'robot_b')
            hooks.append(self.opponent_hook)

        # Nonfall Hook
        if enable_nonfall:
            hooks.append(UprightConstraintHook())

        # 跌倒检测 Hook
        if enable_fall_detection:
            self.fall_hook = FallDetectionHook()
            hooks.append(self.fall_hook)

        super().__init__(
            simulator=simulator,
            step_data_builder=step_data_builder,
            match_duration=match_duration,
            control_frequency=control_frequency,
            hooks=hooks,
        )

        # 单智能体动作空间
        action_dim = HumanoidRobot.ACTION_DIM
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self.render_mode = render_mode
        self.opponent_type = opponent_type
        self.enable_fall_detection = enable_fall_detection

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.runner.reset()

        # 重置对手策略
        if self.opponent_type in ['standing', 'policy'] and self.opponent_hook:
            self.opponent_hook.reset()

        # 手动调用一次 step_data_builder 获取初始观测
        obs, reward, info = self.step_data_builder.get_last_data()
        if obs is None:
            # 使用单智能体格式的初始数据
            obs = self.simulator.robot_a.get_observation(opponent_robot=self.simulator.robot_b)
            reward = 0.0
            info = {'step': 0}

        # 添加跌倒状态
        if self.enable_fall_detection and self.fall_hook:
            info['fallen'] = self.fall_hook.get_fallen_status()

        return obs, info

    def step(self, action):
        self.current_step += 1

        # 时间到，终止
        if self.current_step > self.max_steps:
            obs, reward, info = self.step_data_builder.get_last_data()
            if obs is None:
                obs = self.simulator.robot_a.get_observation(opponent_robot=self.simulator.robot_b)
                reward = 0.0
                info = {'step': self.current_step}
            return obs, float(reward), True, False, info

        # Hook 已终止
        if not self.runner.is_episode_active:
            obs, reward, info = self.step_data_builder.get_last_data()
            if obs is None:
                obs = self.simulator.robot_a.get_observation(opponent_robot=self.simulator.robot_b)
                reward = 0.0
                info = {'step': self.current_step}
            return obs, float(reward), True, False, info

        # 自动填充对手动作
        if self.opponent_type in ['standing', 'policy']:
            opponent_action = self.opponent_hook.get_opponent_action()
        else:
            opponent_action = np.zeros(HumanoidRobot.ACTION_DIM, dtype=np.float32)

        full_action = {
            'robot_a': np.asarray(action, dtype=np.float32),
            'robot_b': opponent_action,
        }

        self.runner.step(full_action)

        # 获取数据（从 Hook 缓存）
        obs, reward, info = self.step_data_builder.get_last_data()

        if obs is None:
            obs = self.simulator.robot_a.get_observation(opponent_robot=self.simulator.robot_b)
            reward = 0.0
            info = {'step': self.current_step}

        # 添加跌倒状态
        if self.enable_fall_detection and self.fall_hook:
            info['fallen'] = self.fall_hook.get_fallen_status()

        # 检查是否由 Hook 终止
        terminated = not self.runner.is_episode_active

        return obs, float(reward), terminated, False, info


# ==================== 便捷类 ====================

class Humanoid21VsFrozenEnv(Humanoid21SingleAgentEnv):
    """对抗冻结对手的单智能体环境"""

    def __init__(self, **kwargs):
        kwargs.setdefault('opponent_type', 'frozen')
        super().__init__(**kwargs)


class Humanoid21VsStandingEnv(Humanoid21SingleAgentEnv):
    """对抗站立对手的单智能体环境"""

    def __init__(self, **kwargs):
        kwargs.setdefault('opponent_type', 'standing')
        super().__init__(**kwargs)


class Humanoid21VsPolicyEnv(Humanoid21SingleAgentEnv):
    """对抗策略对手的单智能体环境"""

    def __init__(self, opponent_policy: BaseCombatPolicy, **kwargs):
        kwargs['opponent_policy'] = opponent_policy
        kwargs.setdefault('opponent_type', 'policy')
        super().__init__(**kwargs)


class Humanoid21NonFallEnv(Humanoid21SingleAgentEnv):
    """带站立约束的单智能体环境"""

    def __init__(self, **kwargs):
        kwargs.setdefault('opponent_type', 'standing')
        kwargs.setdefault('enable_nonfall', True)
        super().__init__(**kwargs)


class Humanoid21FallEnv(Humanoid21SingleAgentEnv):
    """可跌倒的单智能体环境"""

    def __init__(self, **kwargs):
        kwargs.setdefault('opponent_type', 'standing')
        kwargs.setdefault('enable_fall_detection', True)
        super().__init__(**kwargs)


class Humanoid21MatchEnv(Humanoid21DualAgentEnv):
    """双智能体比赛环境"""

    def __init__(self, **kwargs):
        kwargs.setdefault('opponent_policy_b', None)
        super().__init__(**kwargs)


# ==================== 双智能体数据构建器 ====================

class DualAgentStepDataBuilder(StepDataBuilder):
    """双智能体数据构建器"""

    def __init__(self, score_calculator: Optional[ScoreCalculator] = None):
        self.obs_dim = HumanoidRobot.OBSERVATION_DIM
        self.score_calculator = score_calculator

    def build_step_data(
        self,
        f_get_core_state: Callable[[], Dict[str, Any]],
        f_get_derived_state: Callable[[], Dict[str, Any]],
        f_get_sensor_data: Callable[[], Dict[str, Any]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any]]:
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

        info = {
            'step': core_state.get('step_count', 0),
            'robot_states': derived_state['robots'],
        }

        if self.score_calculator is not None:
            info['scores'] = self.score_calculator.get_health()

        return observation, reward, info

    def get_observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "robot_a_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "robot_b_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
        })


# ==================== 导出 ====================

__all__ = [
    # 单智能体环境
    'Humanoid21SingleAgentEnv',
    'Humanoid21VsFrozenEnv',
    'Humanoid21VsStandingEnv',
    'Humanoid21VsPolicyEnv',
    'Humanoid21NonFallEnv',
    'Humanoid21FallEnv',

    # 双智能体环境
    'Humanoid21DualAgentEnv',
    'Humanoid21MatchEnv',

    # Hooks
    'FallDetectionHook',
    'UprightConstraintHook',
    'FreezeRobotHook',
    'OpponentPolicyHook',

    # 数据构建器
    'SingleAgentStepDataBuilder',
    'DualAgentStepDataBuilder',
]
