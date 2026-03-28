"""
Humanoid21Simulator: 基于 MuJoCo 的21自由度人形机器人格斗仿真器

实现 OpenSimulator 接口，提供双机器人格斗仿真的完整功能。

核心状态 (Core State) - 可读可写:
- 广义坐标 q: 关节位置、浮动基座位置和朝向
- 广义速度 q̇: 关节速度、浮动基座线速度和角速度

衍生状态 (Derived State) - 只读:
- 接触点、接触力、摩擦力
- 末端执行器位置、关键点位置和速度
- 质心位置、雅可比矩阵
- 传感器数据（力、IMU、触摸）
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import mujoco

from ..core.physics import PhysicsEngine
from .robot import HumanoidRobot
from ..framework import OpenSimulator


class Humanoid21Simulator(OpenSimulator):
    """
    21自由度人形机器人格斗仿真器

    实现 OpenSimulator 接口，支持双机器人格斗仿真。
    """

    # 默认配置
    DEFAULT_DT = 0.002  # 物理时间步 (秒)
    DEFAULT_ROOT_HEIGHT = 1.282  # 默认根节点高度
    DEFAULT_INITIAL_DISTANCE = 2.0  # 默认初始距离

    # 关键点定义
    KEYPOINT_BODIES = {
        'head': 'head',
        'right_hand': 'hand_right',
        'left_hand': 'hand_left',
        'right_elbow': 'lower_arm_right',
        'left_elbow': 'lower_arm_left',
        'right_knee': 'shin_right',
        'left_knee': 'shin_left',
        'right_foot': 'foot_right',
        'left_foot': 'foot_left'
    }

    def __init__(
        self,
        arena_xml: Optional[str] = None,
        dt: float = DEFAULT_DT,
        gui: bool = False,
        initial_distance: float = DEFAULT_INITIAL_DISTANCE,
        root_height: float = DEFAULT_ROOT_HEIGHT,
        # 新增参数
        control_mode: str = 'torque',  # 'torque' 或 'residual_pd'
        non_fall_mode: bool = False,
        non_fall_pitch_limit_deg: float = 5.0,
        non_fall_roll_limit_deg: float = 5.0,
        default_kp: float = 4.0,
        default_kd: float = 0.4,
    ):
        """
        初始化仿真器

        Args:
            arena_xml: 场景XML文件路径，默认使用 battle_v1.xml
            dt: 物理时间步长 (秒)
            gui: 是否启用GUI
            initial_distance: 两个机器人的初始距离 (米)
            root_height: 机器人根节点高度 (米)
            control_mode: 控制模式 ('torque' 直接扭矩控制, 'residual_pd' 残差PD控制)
            non_fall_mode: 是否启用防跌倒模式
            non_fall_pitch_limit_deg: 防跌倒俯仰角限制 (度)
            non_fall_roll_limit_deg: 防跌倒横滚角限制 (度)
            default_kp: 默认比例增益
            default_kd: 默认微分增益
        """
        self.dt = dt
        self.gui = gui
        self.initial_distance = initial_distance
        self.root_height = root_height

        # 控制模式
        self.control_mode = control_mode
        self.non_fall_mode = non_fall_mode
        self.non_fall_pitch_limit_deg = float(non_fall_pitch_limit_deg)
        self.non_fall_roll_limit_deg = float(non_fall_roll_limit_deg)

        # PD控制器参数
        self._default_kp = float(default_kp)
        self._default_kd = float(default_kd)
        self._controller_kp = np.full(HumanoidRobot.ACTION_DIM, self._default_kp, dtype=np.float32)
        self._controller_kd = np.full(HumanoidRobot.ACTION_DIM, self._default_kd, dtype=np.float32)

        # 参考位置 (用于残差PD控制)
        self._reference_positions = {
            'robot_a': np.zeros(HumanoidRobot.ACTION_DIM, dtype=np.float32),
            'robot_b': np.zeros(HumanoidRobot.ACTION_DIM, dtype=np.float32),
        }

        # 动作缩放 (用于残差PD控制)
        self._action_scale = {
            'robot_a': np.ones(HumanoidRobot.ACTION_DIM, dtype=np.float32) * 0.25,
            'robot_b': np.ones(HumanoidRobot.ACTION_DIM, dtype=np.float32) * 0.25,
        }

        # 非跌倒模式统计
        self._clamp_counts = {
            'current_step': {'robot_a': 0, 'robot_b': 0},
            'episode': {'robot_a': 0, 'robot_b': 0},
        }

        # 加载场景XML
        if arena_xml is None:
            arena_xml = os.path.join(
                os.path.dirname(__file__),
                '../../assets/battle_v1.xml'
            )

        # 初始化物理引擎
        self.physics = PhysicsEngine(
            gui=gui,
            dt=dt,
            arena_xml=arena_xml
        )
        self.model = self.physics.model
        self.data = self.physics.data

        # 初始化机器人
        self._init_robots()

        # 设置初始位置和姿态
        self._set_initial_poses()

        # 缓存静态数据
        self._static_data_cache = None

    def _init_robots(self):
        """初始化机器人实例"""
        # Robot A: 面向 +X 方向
        pos_a = [-self.initial_distance / 2, 0, self.root_height]
        orn_a = [1, 0, 0, 0]  # 单位四元数

        self.robot_a = HumanoidRobot(
            self.physics, pos_a, orn_a,
            robot_id="robot_a",
            color=(0.8, 0.2, 0.2)
        )

        # Robot B: 面向 -X 方向 (绕Z轴旋转180度)
        pos_b = [self.initial_distance / 2, 0, self.root_height]
        orn_b = [0, 0, 0, 1]  # 180度旋转

        self.robot_b = HumanoidRobot(
            self.physics, pos_b, orn_b,
            robot_id="robot_b",
            color=(0.2, 0.2, 0.8)
        )

        # 机器人查找表
        self._robots = {
            'robot_a': self.robot_a,
            'robot_b': self.robot_b,
        }

        # 根节点缓存
        self._root_joint_cache = self._build_root_joint_cache()

    def _build_root_joint_cache(self) -> Dict[str, Dict]:
        """
        构建根关节缓存

        Returns:
            根关节信息字典
        """
        cache = {}
        root_poses = self._get_default_root_poses()

        for robot_id, root_pose in root_poses.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, root_pose['joint_name']
            )
            if joint_id < 0:
                continue

            cache[robot_id] = {
                'joint_id': joint_id,
                'qpos_adr': int(self.model.jnt_qposadr[joint_id]),
                'qvel_adr': int(self.model.jnt_dofadr[joint_id]),
            }

        return cache

    def _get_default_root_poses(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        获取默认的根节点姿态

        Returns:
            根节点姿态字典
        """
        # 将角度转换为四元数
        from scipy.spatial.transform import Rotation as R

        quat_a_xyzw = R.from_euler('z', 0, degrees=True).as_quat()
        quat_a = np.array([quat_a_xyzw[3], quat_a_xyzw[0], quat_a_xyzw[1], quat_a_xyzw[2]])

        quat_b_xyzw = R.from_euler('z', 180, degrees=True).as_quat()
        quat_b = np.array([quat_b_xyzw[3], quat_b_xyzw[0], quat_b_xyzw[1], quat_b_xyzw[2]])

        return {
            'robot_a': {
                'joint_name': 'root_red',
                'position': np.array([-self.initial_distance / 2.0, 0.0, self.root_height], dtype=np.float64),
                'orientation': quat_a,
            },
            'robot_b': {
                'joint_name': 'root_blue',
                'position': np.array([self.initial_distance / 2.0, 0.0, self.root_height], dtype=np.float64),
                'orientation': quat_b,
            },
        }

    def _set_initial_poses(self) -> None:
        """
        设置初始位置和姿态到 MuJoCo 数据

        这会设置 qpos 并调用 mj_forward() 来计算所有衍生状态（包括有效的四元数）。
        """
        root_poses = self._get_default_root_poses()

        for robot_id, root_pose in root_poses.items():
            cache = self._root_joint_cache.get(robot_id)
            if cache is None:
                continue

            # 设置位置和方向到 qpos
            qpos_adr = cache['qpos_adr']
            self.data.qpos[qpos_adr:qpos_adr + 3] = root_pose['position']
            self.data.qpos[qpos_adr + 3:qpos_adr + 7] = root_pose['orientation']

        # 重置速度
        for cache in self._root_joint_cache.values():
            qvel_adr = cache['qvel_adr']
            self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

        # 调用 mj_forward 计算所有衍生状态（包括有效的四元数）
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """
        重置仿真状态

        将机器人重置到初始位置和姿态。
        """
        # 重置速度和控制器
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # 重新设置初始位置和姿态
        self._set_initial_poses()

        # 重置非跌倒模式统计
        self._clamp_counts = {
            'current_step': {'robot_a': 0, 'robot_b': 0},
            'episode': {'robot_a': 0, 'robot_b': 0},
        }

        # 重置参考位置为当前关节位置 (用于残差PD控制)
        for robot_id in ['robot_a', 'robot_b']:
            robot = self._robots[robot_id]
            joint_states = robot.get_joint_states()
            self._reference_positions[robot_id] = joint_states['positions'].copy()

    # ==================== OpenSimulator 必需接口 ====================

    def get_physical_frequency(self) -> float:
        """
        获取物理仿真频率

        Returns:
            物理仿真的频率 (Hz)
        """
        return 1.0 / self.dt

    def set_action(self, action: Dict[str, Any]) -> None:
        """
        接收动作指令并设置到仿真器

        Args:
            action: 动作指令字典，格式：
                {
                    'robot_a': np.ndarray,  # shape=(21,), 机器人A的动作
                    'robot_b': np.ndarray,  # shape=(21,), 机器人B的动作
                }

        注意：
        - 动作值应该在有效范围内 [-1, 1]
        - 动作会在下一个 physical_step 时生效
        - 根据控制模式 (control_mode) 决定如何应用动作：
          * 'torque': 直接扭矩控制
          * 'residual_pd': 残差PD控制
        """
        for robot_id in ['robot_a', 'robot_b']:
            if robot_id in action and action[robot_id] is not None:
                act = np.asarray(action[robot_id], dtype=np.float32).reshape(21)
                act = np.clip(act, -1.0, 1.0)

                if self.control_mode == 'residual_pd':
                    # 使用残差PD控制
                    self.apply_action_residual_pd(robot_id, act)
                else:
                    # 使用直接扭矩控制
                    robot = self.robot_a if robot_id == 'robot_a' else self.robot_b
                    robot.apply_action(act)

    def physical_step(self) -> None:
        """
        执行一次物理仿真步进

        功能：
        1. 根据当前控制指令计算力矩/力
        2. 执行碰撞检测
        3. 数值积分更新位置和速度
        4. 更新所有内部缓存
        5. 如果启用非跌倒模式，限制根节点朝向
        """
        self.physics.step()

        # 非跌倒模式：限制根节点朝向
        if self.non_fall_mode:
            self.enforce_non_fall_mode()

    def get_sensor_data(self) -> Dict[str, Any]:
        """
        获取传感器数据（属于衍生状态，只读）

        Returns:
            传感器数据字典：
            {
                'touch': {
                    'robot_a': {'left_foot': bool, 'right_foot': bool, ...},
                    'robot_b': {...}
                },
                'force': {
                    'robot_a': np.ndarray,  # 外部受力 (6,)
                    'robot_b': np.ndarray,
                },
                'imu': {
                    'robot_a': {
                        'linear_acceleration': np.ndarray,
                        'angular_velocity': np.ndarray,
                    },
                    'robot_b': {...}
                }
            }
        """
        sensors = {
            'touch': {},
            'force': {},
            'imu': {},
        }

        for robot_id, robot in self._robots.items():
            # 触摸传感器（脚部接触）
            feet_contact = robot.get_feet_contact()
            sensors['touch'][robot_id] = feet_contact

            # 外部受力
            external_forces = robot.get_external_forces()
            sensors['force'][robot_id] = external_forces

            # IMU 数据（从 torso 状态获取）
            torso_state = robot.get_torso_state()
            sensors['imu'][robot_id] = {
                'linear_velocity': torso_state['linear_velocity'],
                'angular_velocity': torso_state['angular_velocity'],
            }

        return sensors

    def get_static_data(self) -> Dict[str, Any]:
        """
        获取静态数据

        Returns:
            静态数据字典，包含机器人、场景、物理、相机配置等
        """
        if self._static_data_cache is not None:
            return self._static_data_cache

        # 构建机器人静态数据
        robots_data = {}
        for robot_id, robot in self._robots.items():
            joint_limits = robot.get_joint_position_limits()
            ctrl_limits = robot.get_actuator_ctrl_limits()

            robots_data[robot_id] = {
                'model_type': 'HumanoidRobot',
                'dof': HumanoidRobot.ACTION_DIM,
                'joint_names': list(HumanoidRobot.CONTROLLED_JOINTS),
                'actuator_names': list(HumanoidRobot.CONTROLLED_JOINTS),
                'body_names': self._get_all_body_names(robot),
                'geom_names': self._get_all_geom_names(robot),
                'keypoint_bodies': self.KEYPOINT_BODIES.copy(),
                'initial_position': robot.get_position(),
                'joint_position_limits': joint_limits,
                'actuator_ctrl_limits': ctrl_limits,
            }

        # 场景数据
        scene_data = {
            'arena_type': 'battle_v1',
            'arena_size': (6.0, 6.0, 3.0),
            'floor_height': 0.0,
            'gravity': self.model.opt.gravity.copy(),
            'timestep': self.dt,
        }

        # 物理数据
        physics_data = {
            'solver': 'PGS',
            'iterations': int(self.model.opt.iterations),
            'integrator': 'RK4',
            'timestep': self.dt,
        }

        # 相机数据（广播视角）
        cameras_data = {
            'broadcast': {
                'name': 'broadcast',
                'resolution': (1280, 720),
                'fovy': 45.0,
            }
        }

        self._static_data_cache = {
            'robots': robots_data,
            'scene': scene_data,
            'physics': physics_data,
            'cameras': cameras_data,
        }

        return self._static_data_cache

    def _get_all_body_names(self, robot: HumanoidRobot) -> List[str]:
        """获取机器人所有body名称"""
        names = []
        suffix = robot.suffix
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.endswith(suffix):
                names.append(name)
        return names

    def _get_all_geom_names(self, robot: HumanoidRobot) -> List[str]:
        """获取机器人所有geom名称"""
        names = []
        suffix = robot.suffix
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.endswith(suffix):
                names.append(name)
        return names

    def get_core_state(self) -> Dict[str, Any]:
        """
        获取核心状态（可读可写）

        Returns:
            核心状态字典：
            {
                'time': float,
                'robots': {
                    'robot_a': {
                        'root_position': np.ndarray,  # (3,)
                        'root_orientation': np.ndarray,  # (4,)
                        'root_linear_velocity': np.ndarray,  # (3,)
                        'root_angular_velocity': np.ndarray,  # (3,)
                        'joint_positions': np.ndarray,  # (21,)
                        'joint_velocities': np.ndarray,  # (21,)
                    },
                    'robot_b': {...}
                }
            }
        """
        core_state = {
            'time': float(self.data.time),
            'robots': {},
        }

        for robot_id, robot in self._robots.items():
            # 获取根节点状态
            root_cache = self._root_joint_cache[robot_id]
            qpos_adr = root_cache['qpos_adr']
            qvel_adr = root_cache['qvel_adr']

            root_position = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
            root_orientation = self.data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
            root_linear_velocity = self.data.qvel[qvel_adr:qvel_adr + 3].copy()
            root_angular_velocity = self.data.qvel[qvel_adr + 3:qvel_adr + 6].copy()

            # 获取关节状态
            joint_states = robot.get_joint_states()

            core_state['robots'][robot_id] = {
                'root_position': root_position.astype(np.float32),
                'root_orientation': root_orientation.astype(np.float32),
                'root_linear_velocity': root_linear_velocity.astype(np.float32),
                'root_angular_velocity': root_angular_velocity.astype(np.float32),
                'joint_positions': joint_states['positions'].astype(np.float32),
                'joint_velocities': joint_states['velocities'].astype(np.float32),
            }

        return core_state

    def set_core_state(self, state: Dict[str, Any]) -> None:
        """
        设置核心状态（可读可写）

        Args:
            state: 核心状态字典，格式与 get_core_state 返回值相同

        注意：
        - 修改后会更新运动学、碰撞、动力学缓存
        """
        # 设置仿真时间
        if 'time' in state:
            self.data.time = float(state['time'])

        # 设置每个机器人的核心状态
        for robot_id, robot_state in state.get('robots', {}).items():
            if robot_id not in self._root_joint_cache:
                continue

            root_cache = self._root_joint_cache[robot_id]
            qpos_adr = root_cache['qpos_adr']
            qvel_adr = root_cache['qvel_adr']

            # 设置根节点位置和朝向
            if 'root_position' in robot_state:
                self.data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(
                    robot_state['root_position'], dtype=np.float64
                )
            if 'root_orientation' in robot_state:
                self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(
                    robot_state['root_orientation'], dtype=np.float64
                )

            # 设置根节点速度
            if 'root_linear_velocity' in robot_state:
                self.data.qvel[qvel_adr:qvel_adr + 3] = np.asarray(
                    robot_state['root_linear_velocity'], dtype=np.float64
                )
            if 'root_angular_velocity' in robot_state:
                self.data.qvel[qvel_adr + 3:qvel_adr + 6] = np.asarray(
                    robot_state['root_angular_velocity'], dtype=np.float64
                )

            # 设置关节状态
            robot = self._robots[robot_id]
            if 'joint_positions' in robot_state:
                for i, joint in enumerate(HumanoidRobot.CONTROLLED_JOINTS):
                    if joint in robot._joint_indices:
                        j_idx = robot._joint_indices[joint]
                        qpos_idx = self.model.jnt_qposadr[j_idx]
                        self.data.qpos[qpos_idx] = float(robot_state['joint_positions'][i])

            if 'joint_velocities' in robot_state:
                for i, joint in enumerate(HumanoidRobot.CONTROLLED_JOINTS):
                    if joint in robot._joint_indices:
                        j_idx = robot._joint_indices[joint]
                        qvel_idx = self.model.jnt_dofadr[j_idx]
                        self.data.qvel[qvel_idx] = float(robot_state['joint_velocities'][i])

        # 更新物理引擎缓存
        self._update_physics_cache()

    def _update_physics_cache(self) -> None:
        """
        更新物理引擎内部缓存

        顺序：
        1. 运动学缓存（正向运动学结果）
        2. 碰撞检测缓存
        3. 动力学缓存（如果需要）
        """
        # 1. 运动学缓存
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camLight(self.model, self.data)
        mujoco.mj_tendon(self.model, self.data)
        mujoco.mj_transmission(self.model, self.data)

        # 2. 碰撞检测缓存
        mujoco.mj_collision(self.model, self.data)

        # 3. 动力学缓存（如果需要）
        mujoco.mj_factorM(self.model, self.data)

    def get_derived_state(self) -> Dict[str, Any]:
        """
        获取衍生状态（只读）

        Returns:
            衍生状态字典：
            {
                'contacts': [...],
                'robots': {
                    'robot_a': {
                        'observation': np.ndarray,  # 观测 (obs_dim,)
                        'keypoint_positions': {...},
                        'keypoint_velocities': {...},
                        'torso_position': np.ndarray,
                        'torso_orientation': np.ndarray,
                    },
                    'robot_b': {...}
                }
            }
        """
        derived_state = {
            'contacts': self._get_all_contacts(),
            'robots': {},
        }

        for robot_id, robot in self._robots.items():
            # 获取观测
            opponent_robot = self.robot_b if robot_id == 'robot_a' else self.robot_a
            observation = robot.get_observation(opponent_robot=opponent_robot)

            # 获取关键点位置和速度
            keypoint_positions = robot.get_keypoint_positions()
            keypoint_velocities = robot.get_keypoint_velocities()

            # 计算质心位置（简化版，使用 torso 位置）
            torso_state = robot.get_torso_state()

            derived_state['robots'][robot_id] = {
                'observation': observation.astype(np.float32),
                'keypoint_positions': {k: v.astype(np.float32) for k, v in keypoint_positions.items()},
                'keypoint_velocities': {k: v.astype(np.float32) for k, v in keypoint_velocities.items()},
                'torso_position': torso_state['position'].astype(np.float32),
                'torso_orientation': torso_state['orientation'].astype(np.float32),
            }

        return derived_state

    def _get_all_contacts(self) -> List[Dict[str, Any]]:
        """
        获取所有接触点信息

        Returns:
            接触点列表
        """
        contacts = []

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1 = con.geom1
            geom2 = con.geom2

            # 获取 geom 名称
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)

            # 获取对应的 body
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            body_name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body_name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

            contact = {
                'geom_a': name1 or '',
                'geom_b': name2 or '',
                'body_a': body_name1 or '',
                'body_b': body_name2 or '',
                'position': con.pos.copy().astype(np.float32),
                'normal': con.frame[:3].copy().astype(np.float32),  # 法向 (前3个元素)
                'distance': float(con.dist),
            }
            contacts.append(contact)

        return contacts

    def get_broadcastview_image(self) -> np.ndarray:
        """
        获取当前状态下广播视角的观测图片

        Returns:
            图像数组，shape: (720, 1280, 3), dtype: np.uint8
        """
        try:
            # 计算相机位置
            pos_a = self.robot_a.get_position()
            pos_b = self.robot_b.get_position()
            center = (pos_a + pos_b) / 2.0

            # 观察目标
            lookat = center.copy()
            lookat[2] = 1.0

            # 计算方向和角度
            direction = pos_b - pos_a
            dist_ab = np.linalg.norm(direction)
            if dist_ab > 1e-6:
                direction = direction / dist_ab
            else:
                direction = np.array([1.0, 0.0, 0.0])

            # 相机方位角
            dir_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            azi = dir_angle + 90.0
            ele = -20.0
            dist = max(2.5, min(4.0, dist_ab * 1.5))

            # 创建相机
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.distance = dist
            cam.elevation = ele
            cam.azimuth = azi

            # 渲染
            renderer = mujoco.Renderer(self.model, height=720, width=1280)
            renderer.update_scene(self.data, camera=cam)
            image = renderer.render()
            del renderer

            return image
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to render broadcast view: {e}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    # ==================== 可选的高级接口 ====================

    def get_contacts(
        self,
        body_a: Optional[str] = None,
        body_b: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取碰撞接触信息

        Args:
            body_a: 第一个body名称，如果为None则返回所有碰撞
            body_b: 第二个body名称，如果为None则不限制

        Returns:
            碰撞列表
        """
        all_contacts = self._get_all_contacts()

        if body_a is None and body_b is None:
            return all_contacts

        filtered = []
        for contact in all_contacts:
            if body_a is not None and contact['body_a'] != body_a and contact['body_b'] != body_a:
                continue
            if body_b is not None and contact['body_a'] != body_b and contact['body_b'] != body_b:
                continue
            filtered.append(contact)

        return filtered

    def apply_external_force(
        self,
        robot_id: str,
        body_name: str,
        force: np.ndarray,
        position: Optional[np.ndarray] = None
    ) -> None:
        """
        施加外部力

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            body_name: 目标body名称
            force: 力向量 (3,)
            position: 施力位置（世界坐标系），如果为None则施加到质心
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return

        # 解析 body 名称
        full_body_name = f"{body_name}{robot.suffix}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)
        if body_id < 0:
            return

        # 施力位置
        if position is None:
            point = self.data.xpos[body_id].copy()
        else:
            point = np.asarray(position, dtype=np.float64)

        # 施加力
        mujoco.mj_applyFT(
            self.model,
            self.data,
            np.asarray(force, dtype=np.float64).reshape(3),
            np.zeros(3),  # 无力矩
            point,
            body_id,
            self.data.qfrc_applied,
        )

    def apply_external_torque(
        self,
        robot_id: str,
        body_name: str,
        torque: np.ndarray
    ) -> None:
        """
        施加外部力矩

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            body_name: 目标body名称
            torque: 力矩向量 (3,)
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return

        full_body_name = f"{body_name}{robot.suffix}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)
        if body_id < 0:
            return

        point = self.data.xpos[body_id].copy()

        mujoco.mj_applyFT(
            self.model,
            self.data,
            np.zeros(3),  # 无力
            np.asarray(torque, dtype=np.float64).reshape(3),
            point,
            body_id,
            self.data.qfrc_applied,
        )

    def get_body_position(self, robot_id: str, body_name: str) -> np.ndarray:
        """
        获取body位置（运动学衍生状态，只读）

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            body_name: body名称

        Returns:
            位置向量 (3,)
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return np.zeros(3)

        full_body_name = f"{body_name}{robot.suffix}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)
        if body_id < 0:
            return np.zeros(3)

        return self.data.xpos[body_id].copy().astype(np.float32)

    def get_body_orientation(self, robot_id: str, body_name: str) -> np.ndarray:
        """
        获取body朝向（运动学衍生状态，只读）

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            body_name: body名称

        Returns:
            四元数 [w, x, y, z]
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        full_body_name = f"{body_name}{robot.suffix}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)
        if body_id < 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return self.data.xquat[body_id].copy().astype(np.float32)

    def get_joint_position(self, robot_id: str, joint_name: str) -> float:
        """
        获取关节位置（核心状态，可读可写）

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            joint_name: 关节名称

        Returns:
            关节角度/位置
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return 0.0

        joint_idx = robot._joint_indices.get(joint_name)
        if joint_idx is None:
            return 0.0

        qpos_idx = self.model.jnt_qposadr[joint_idx]
        return float(self.data.qpos[qpos_idx])

    def get_joint_velocity(self, robot_id: str, joint_name: str) -> float:
        """
        获取关节速度（核心状态，可读可写）

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            joint_name: 关节名称

        Returns:
            关节速度
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return 0.0

        joint_idx = robot._joint_indices.get(joint_name)
        if joint_idx is None:
            return 0.0

        qvel_idx = self.model.jnt_dofadr[joint_idx]
        return float(self.data.qvel[qvel_idx])

    def set_joint_position(self, robot_id: str, joint_name: str, position: float) -> None:
        """
        设置关节位置（核心状态，可读可写）

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            joint_name: 关节名称
            position: 目标位置
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return

        joint_idx = robot._joint_indices.get(joint_name)
        if joint_idx is None:
            return

        qpos_idx = self.model.jnt_qposadr[joint_idx]
        self.data.qpos[qpos_idx] = float(position)

        # 更新缓存
        self._update_physics_cache()

    def set_joint_velocity(self, robot_id: str, joint_name: str, velocity: float) -> None:
        """
        设置关节速度（核心状态，可读可写）

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            joint_name: 关节名称
            velocity: 目标速度
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return

        joint_idx = robot._joint_indices.get(joint_name)
        if joint_idx is None:
            return

        qvel_idx = self.model.jnt_dofadr[joint_idx]
        self.data.qvel[qvel_idx] = float(velocity)

    # ==================== 残差PD控制方法 ====================

    def set_reference_positions(self, robot_id: str, positions: np.ndarray) -> None:
        """
        设置参考位置 (用于残差PD控制)

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            positions: 参考位置数组, shape: (21,)
        """
        if robot_id not in self._reference_positions:
            return
        positions = np.asarray(positions, dtype=np.float32).reshape(HumanoidRobot.ACTION_DIM)
        self._reference_positions[robot_id] = positions.copy()

    def set_action_scale(self, robot_id: str, scale: float) -> None:
        """
        设置动作缩放 (用于残差PD控制)

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            scale: 缩放因子
        """
        if robot_id not in self._action_scale:
            return
        self._action_scale[robot_id] = np.full(HumanoidRobot.ACTION_DIM, float(scale), dtype=np.float32)

    def set_pd_gains(self, kp: float = None, kd: float = None) -> None:
        """
        设置PD控制器增益

        Args:
            kp: 比例增益 (None表示不修改)
            kd: 微分增益 (None表示不修改)
        """
        if kp is not None:
            self._controller_kp = np.full(HumanoidRobot.ACTION_DIM, float(kp), dtype=np.float32)
        if kd is not None:
            self._controller_kd = np.full(HumanoidRobot.ACTION_DIM, float(kd), dtype=np.float32)

    def _compute_target_positions(self, robot_id: str, residual_action: np.ndarray) -> np.ndarray:
        """
        从残差动作计算目标位置

        Args:
            robot_id: 机器人ID
            residual_action: 残差动作, shape: (21,), 范围 [-1, 1]

        Returns:
            目标位置, shape: (21,)
        """
        robot = self._robots[robot_id]
        joint_limits = robot.get_joint_position_limits()

        # 计算目标位置: 参考位置 + 动作缩放 * 残差动作
        target_positions = self._reference_positions[robot_id] + self._action_scale[robot_id] * residual_action

        # 限制在关节位置范围内
        target_positions = np.clip(
            target_positions,
            joint_limits['lower'],
            joint_limits['upper'],
        ).astype(np.float32)

        return target_positions

    def _compute_pd_torque(self, robot_id: str, target_positions: np.ndarray) -> np.ndarray:
        """
        从目标位置计算PD控制扭矩

        Args:
            robot_id: 机器人ID
            target_positions: 目标位置, shape: (21,)

        Returns:
            控制扭矩, shape: (21,)
        """
        robot = self._robots[robot_id]
        joint_states = robot.get_joint_states()

        current_positions = joint_states['positions']
        current_velocities = joint_states['velocities']

        # PD控制律: torque = kp * (target - current) - kd * velocity
        torque = self._controller_kp * (target_positions - current_positions) - self._controller_kd * current_velocities

        # 限制在执行器扭矩范围内
        ctrl_limits = robot.get_actuator_ctrl_limits()
        torque = np.clip(
            torque,
            ctrl_limits['lower'],
            ctrl_limits['upper'],
        ).astype(np.float32)

        return torque

    def apply_action_residual_pd(self, robot_id: str, action: np.ndarray) -> None:
        """
        应用残差PD控制动作

        Args:
            robot_id: 机器人ID ('robot_a' 或 'robot_b')
            action: 残差动作, shape: (21,), 范围 [-1, 1]
        """
        action = np.asarray(action, dtype=np.float32).reshape(HumanoidRobot.ACTION_DIM)
        action = np.clip(action, -1.0, 1.0)

        # 计算目标位置
        target_positions = self._compute_target_positions(robot_id, action)

        # 计算PD扭矩
        torque = self._compute_pd_torque(robot_id, target_positions)

        # 应用扭矩
        robot = self._robots[robot_id]
        robot.apply_action(torque)

    # ==================== 非跌倒模式方法 ====================

    def _clamp_root_orientation(self, robot_id: str) -> bool:
        """
        限制根节点朝向 (防止跌倒)

        Args:
            robot_id: 机器人ID

        Returns:
            是否发生了限制
        """
        if not self.non_fall_mode:
            return False

        if robot_id not in self._root_joint_cache:
            return False

        root_cache = self._root_joint_cache[robot_id]
        qpos_adr = root_cache['qpos_adr']

        # 获取当前朝向 (wxyz)
        orientation_wxyz = np.asarray(
            self.data.qpos[qpos_adr + 3:qpos_adr + 7],
            dtype=np.float64
        )

        if np.linalg.norm(orientation_wxyz) < 1e-8:
            return False

        # 转换为 xyzw 格式 (scipy 格式)
        orientation_xyzw = np.array([
            orientation_wxyz[1], orientation_wxyz[2],
            orientation_wxyz[3], orientation_wxyz[0],
        ], dtype=np.float64)

        # 转换为欧拉角
        from scipy.spatial.transform import Rotation as R
        try:
            rotation = R.from_quat(orientation_xyzw)
            roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)
        except:
            return False

        # 限制roll和pitch
        clamped_roll = float(np.clip(roll, -self.non_fall_roll_limit_deg, self.non_fall_roll_limit_deg))
        clamped_pitch = float(np.clip(pitch, -self.non_fall_pitch_limit_deg, self.non_fall_pitch_limit_deg))

        if np.isclose(roll, clamped_roll) and np.isclose(pitch, clamped_pitch):
            return False

        # 如果发生了限制，更新朝向
        clamped_rotation = R.from_euler('xyz', [clamped_roll, clamped_pitch, yaw], degrees=True)
        clamped_xyzw = clamped_rotation.as_quat()
        clamped_wxyz = np.array([
            clamped_xyzw[3], clamped_xyzw[0],
            clamped_xyzw[1], clamped_xyzw[2],
        ], dtype=np.float64)

        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = clamped_wxyz

        # 清零角速度
        qvel_adr = root_cache['qvel_adr']
        self.data.qvel[qvel_adr:qvel_adr + 3] = 0.0

        # 统计限制次数
        self._clamp_counts['current_step'][robot_id] += 1
        self._clamp_counts['episode'][robot_id] += 1

        return True

    def enforce_non_fall_mode(self) -> bool:
        """
        强制执行非跌倒模式 (限制根节点朝向)

        Returns:
            是否发生了任何限制
        """
        if not self.non_fall_mode:
            return False

        changed = False
        for robot_id in ('robot_a', 'robot_b'):
            changed = self._clamp_root_orientation(robot_id) or changed

        if changed:
            # 更新物理引擎缓存
            mujoco.mj_forward(self.model, self.data)

        return changed

    def get_clamp_counts(self) -> Dict[str, Dict[str, int]]:
        """
        获取限制次数统计

        Returns:
            限制次数字典: {'current_step': {'robot_a': int, 'robot_b': int}, 'episode': {...}}
        """
        return {
            'current_step': self._clamp_counts['current_step'].copy(),
            'episode': self._clamp_counts['episode'].copy(),
        }

    def reset_clamp_counts(self) -> None:
        """重置当前步骤的限制次数"""
        self._clamp_counts['current_step'] = {'robot_a': 0, 'robot_b': 0}

    def close(self) -> None:
        """关闭仿真器"""
        try:
            self.physics.close()
        except:
            pass

    def __del__(self):
        """析构函数"""
        self.close()
