"""
Humanoid21 专用 Hook 实现

提供用于 21 自由度人形机器人格斗的各种 Hook 实现。
这些 Hook 可以被独立使用，也可以通过 CombatEnv 框架使用。
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
from gymnasium import spaces

from ..hook.base_hook import BaseHook, InvokeType
from ..rl_env import StepDataBuilder
from ...core.humanoid_robot import HumanoidRobot


class GymEnvironmentAdapter:
    """
    Gym 环境适配器

    将 CombatGymEnv 的接口适配为 Hook 可用的函数接口。
    这样 combat_gym.py 可以在不修改的情况下使用 Hook 框架。
    """

    def __init__(self, gym_env):
        """初始化适配器"""
        self.gym_env = gym_env

    def make_get_action(self) -> Callable[[], Dict[str, Any]]:
        """创建获取动作的函数"""
        def get_action() -> Dict[str, Any]:
            action = {}
            if hasattr(self.gym_env, 'actions'):
                action['robot_a'] = self.gym_env.actions.get('robot_a')
                action['robot_b'] = self.gym_env.actions.get('robot_b')
            return action
        return get_action

    def make_get_static_data(self) -> Callable[[], Dict[str, Any]]:
        """创建获取静态数据的函数"""
        def get_static_data() -> Dict[str, Any]:
            return {
                'physics': {
                    'dt': self.gym_env.dt,
                    'sim_frequency': self.gym_env.sim_frequency,
                },
                'robots': {
                    'robot_a': {
                        'dof': 21,
                        'joint_names': self.gym_env._joint_names,
                    },
                    'robot_b': {
                        'dof': 21,
                        'joint_names': self.gym_env._joint_names,
                    },
                },
            }
        return get_static_data

    def make_get_sensor_data(self) -> Callable[[], Dict[str, Any]]:
        """创建获取传感器数据的函数"""
        def get_sensor_data() -> Dict[str, Any]:
            sensor_data = {
                'touch': {},
                'force': {},
                'imu': {},
            }

            for robot_id, robot in [('robot_a', self.gym_env.robot_a), ('robot_b', self.gym_env.robot_b)]:
                # 脚部接触
                if hasattr(robot, 'get_feet_contact'):
                    sensor_data['touch'][robot_id] = robot.get_feet_contact()
                else:
                    sensor_data['touch'][robot_id] = {}

                # 外部受力
                if hasattr(robot, 'get_external_forces'):
                    sensor_data['force'][robot_id] = robot.get_external_forces()
                else:
                    sensor_data['force'][robot_id] = np.zeros(6)

                # IMU 数据
                if hasattr(robot, 'get_torso_state'):
                    torso_state = robot.get_torso_state()
                    sensor_data['imu'][robot_id] = {
                        'linear_velocity': torso_state.get('linear_velocity', np.zeros(3)),
                        'angular_velocity': torso_state.get('angular_velocity', np.zeros(3)),
                    }
                else:
                    sensor_data['imu'][robot_id] = {
                        'linear_velocity': np.zeros(3),
                        'angular_velocity': np.zeros(3),
                    }

            return sensor_data
        return get_sensor_data

    def make_get_core_state(self) -> Callable[[], Dict[str, Any]]:
        """创建获取核心状态的函数"""
        def get_core_state() -> Dict[str, Any]:
            core_state = {
                'time': self.gym_env.physics.data.time,
                'robots': {},
            }

            for robot_id, robot in [('robot_a', self.gym_env.robot_a), ('robot_b', self.gym_env.robot_b)]:
                joint_states = robot.get_joint_states()
                torso_state = robot.get_torso_state()

                core_state['robots'][robot_id] = {
                    'root_position': torso_state['position'],
                    'root_orientation': torso_state['orientation'],
                    'root_linear_velocity': torso_state['linear_velocity'],
                    'root_angular_velocity': torso_state['angular_velocity'],
                    'joint_positions': joint_states['positions'],
                    'joint_velocities': joint_states['velocities'],
                }

            return core_state
        return get_core_state

    def make_get_derived_state(self) -> Callable[[], Dict[str, Any]]:
        """创建获取衍生状态的函数"""
        def get_derived_state() -> Dict[str, Any]:
            derived_state = {
                'contacts': [],
                'robots': {},
            }

            # 获取碰撞信息（如果有）
            if hasattr(self.gym_env, 'collision_detector'):
                # 这里可以添加碰撞检测逻辑
                pass

            for robot_id, robot in [('robot_a', self.gym_env.robot_a), ('robot_b', self.gym_env.robot_b)]:
                # 关键点位置
                keypoint_positions = robot.get_keypoint_positions()
                keypoint_velocities = robot.get_keypoint_velocities()

                derived_state['robots'][robot_id] = {
                    'keypoint_positions': keypoint_positions,
                    'keypoint_velocities': keypoint_velocities,
                    'torso_position': robot.get_position(),
                    'torso_orientation': robot.get_orientation(),
                }

            return derived_state
        return get_derived_state

    def make_set_core_state(self) -> Optional[Callable[[Dict[str, Any]], None]]:
        """创建设置核心状态的函数"""
        def set_core_state(state: Dict[str, Any]) -> None:
            if 'time' in state:
                self.gym_env.physics.data.time = float(state['time'])

            for robot_id, robot_state in state.get('robots', {}).items():
                robot = getattr(self.gym_env, robot_id, None)
                if robot is None:
                    continue

                # 设置关节状态
                if 'joint_positions' in robot_state:
                    for i, joint_name in enumerate(self.gym_env._joint_names):
                        joint_id = robot._joint_indices.get(joint_name)
                        if joint_id is not None:
                            qpos_idx = self.gym_env.physics.model.jnt_qposadr[joint_id]
                            self.gym_env.physics.data.qpos[qpos_idx] = float(robot_state['joint_positions'][i])

                if 'joint_velocities' in robot_state:
                    for i, joint_name in enumerate(self.gym_env._joint_names):
                        joint_id = robot._joint_indices.get(joint_name)
                        if joint_id is not None:
                            qvel_idx = self.gym_env.physics.model.jnt_dofadr[joint_id]
                            self.gym_env.physics.data.qvel[qvel_idx] = float(robot_state['joint_velocities'][i])

            # 更新物理缓存
            mujoco = __import__('mujoco')
            mujoco.mj_kinematics(self.gym_env.physics.model, self.gym_env.physics.data)
            mujoco.mj_comPos(self.gym_env.physics.model, self.gym_env.physics.data)
            mujoco.mj_camLight(self.gym_env.physics.model, self.gym_env.physics.data)
            mujoco.mj_collision(self.gym_env.physics.model, self.gym_env.physics.data)

        return set_core_state

    def make_set_action(self) -> Callable[[Dict[str, Any]], None]:
        """创建设置动作的函数"""
        def set_action(action: Dict[str, Any]) -> None:
            if 'robot_a' in action and action['robot_a'] is not None:
                self.gym_env.actions['robot_a'] = np.asarray(action['robot_a'], dtype=np.float32).reshape(21)
            if 'robot_b' in action and action['robot_b'] is not None:
                self.gym_env.actions['robot_b'] = np.asarray(action['robot_b'], dtype=np.float32).reshape(21)
        return set_action


class GymHookWrapper(BaseHook):
    """
    Gym Hook 包装器

    将基于 Hook 框架的组件包装为 combat_gym.py 可用的形式。
    """

    def __init__(self, hook: BaseHook, gym_env):
        """
        初始化包装器

        Args:
            hook: 要包装的 Hook
            gym_env: CombatGymEnv 实例
        """
        self.hook = hook
        self.gym_env = gym_env
        self.adapter = GymEnvironmentAdapter(gym_env)

    @property
    def name(self) -> str:
        return self.hook.name

    def invoke(
        self,
        invoke_type: InvokeType,
        f_get_action: Callable[[], Dict[str, Any]],
        f_get_static_data: Callable[[], Dict[str, Any]],
        f_get_sensor_data: Callable[[], Dict[str, Any]],
        f_get_core_state: Callable[[], Dict[str, Any]],
        f_get_derived_state: Callable[[], Dict[str, Any]],
        f_set_core_state: Optional[Callable[[Dict[str, Any]], None]],
        f_set_action: Optional[Callable[[Dict[str, Any]], None]],
    ) -> bool:
        """调用 Hook"""
        return self.hook.invoke(
            invoke_type,
            f_get_action,
            f_get_static_data,
            f_get_sensor_data,
            f_get_core_state,
            f_get_derived_state,
            f_set_core_state,
            f_set_action,
        )


# ==================== 默认实现 ====================

class DefaultStepDataBuilder(StepDataBuilder):
    """默认的 step 数据构建器：零奖励 + 机器人观测 + 基础 info"""

    def __init__(self):
        self.obs_dim = HumanoidRobot.OBSERVATION_DIM

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
        reward = {'robot_a': 0.0, 'robot_b': 0.0}

        # 构建 info
        info = {
            'step': core_state.get('step_count', 0),
        }

        return observation, reward, info

    def get_observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "robot_a_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "robot_b_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
        })


# ==================== 常用 Hook ====================

class HealthTerminationHook(BaseHook):
    """
    血量终止 Hook

    当任一机器人血量归零时终止。
    """

    def __init__(self, score_calculator):
        super().__init__()
        self.score_calculator = score_calculator

    @property
    def name(self) -> str:
        return "health_termination_hook"

    @property
    def priority(self) -> int:
        return -90

    def invoke(self, invoke_type: InvokeType, *args, **kwargs) -> bool:
        if invoke_type == InvokeType.POST_ACTION_STEP:
            is_over, winner, reason = self.score_calculator.check_match_over()
            if is_over:
                return True  # 终止
        return False
