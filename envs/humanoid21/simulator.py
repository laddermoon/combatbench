import os
# Set EGL backend BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import mujoco
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from envs.framework.backend import BaseSimulator
from envs.humanoid21.meta import Humanoid21Meta

_TURB_DEBUG = os.environ.get("COMBATBENCH_TURB_DEBUG", "0") == "1"
_TURB_DEBUG_MAX_PHYS_STEPS = max(0, int(os.environ.get("COMBATBENCH_TURB_DEBUG_MAX_PHYS_STEPS", "400")))


class MujocoCombatSimulator(BaseSimulator):
    """
    Humanoid21 双机器人对抗仿真器
    
    严格按照 CONTROLSPEC.md 和 DATASPEC.md 实现：
    - 归一化位置控制 (action ∈ [-1, 1])
    - 固化的 KP/KD 参数
    - 局部坐标系优先
    - 按机器人隔离的数据接口

    静态元数据 (DT, KP, KD, CONTROLLED_JOINTS, INITIAL_POSES, geom 分类等)
    全部由 Humanoid21Meta 统一管理。
    """

    # 静态参数 — 从 Humanoid21Meta 引用，保持单一数据源
    DT = Humanoid21Meta.DT
    ACTION_DIM = Humanoid21Meta.ACTION_DIM
    ARENA_XML = str(Path(__file__).parent / 'battle_v1.xml')
    KP = Humanoid21Meta.KP
    KD = Humanoid21Meta.KD
    CONTROLLED_JOINTS = Humanoid21Meta.CONTROLLED_JOINTS
    INITIAL_POSES = Humanoid21Meta.INITIAL_POSES
    
    def __init__(self, initial_distance: float = 2.0, debug_torque: bool = False,
                 initial_pose_a: str = 'standing', initial_pose_b: str = 'standing'):
        """
        初始化模拟器

        Args:
            initial_distance: 两个机器人之间的初始距离
            debug_torque: 是否打印力矩调试信息
            initial_pose_a: 机器人A的初始姿态 ('standing', 'squat', 'stand_on_left_leg', 'prone', 'supine')
            initial_pose_b: 机器人B的初始姿态
        """
        self.dt = self.DT
        self.initial_distance = initial_distance
        self.action_dim = self.ACTION_DIM
        self._debug_torque = debug_torque  # 是否打印力矩调试信息

        # 验证初始姿态参数
        valid_poses = list(self.INITIAL_POSES.keys())
        if initial_pose_a not in valid_poses:
            raise ValueError(f"initial_pose_a must be one of {valid_poses}, got {initial_pose_a}")
        if initial_pose_b not in valid_poses:
            raise ValueError(f"initial_pose_b must be one of {valid_poses}, got {initial_pose_b}")

        self._initial_pose_a = initial_pose_a
        self._initial_pose_b = initial_pose_b

        # 加载 MuJoCo 模型
        self.model = mujoco.MjSpec.from_file(self.ARENA_XML).compile()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.DT
        mujoco.mj_forward(self.model, self.data)

        # 校验模型与 Humanoid21Meta 静态元数据一致
        meta_errors = Humanoid21Meta.validate(self.model)
        if meta_errors:
            raise ValueError(f"Model validation failed:\n" + "\n".join(meta_errors))

        # 构建运行时查找表 + per-robot 结构化数据
        self._meta = Humanoid21Meta.build_runtime_tables(self.model)
        self._robots = self._meta['robots']  # {'robot_a': {...}, 'robot_b': {...}}
        self._env_geom_ids = self._meta['env_geom_ids']
        self._ground_geom_id = self._meta['ground_geom_id']
        self._body_to_robot = self._meta['body_to_robot']
        self._geom_id_to_name = self._meta['geom_id_to_name']
        self._body_id_to_name = self._meta['body_id_to_name']
        self._body_id_to_aff = self._meta['body_id_to_aff']
        self._geom_id_to_aff = self._meta['geom_id_to_aff']

        self._compute_normalization_params()

        # 初始化控制目标
        self._target_pos_norm = {
            'robot_a': self.INITIAL_POSES[initial_pose_a]['action'].copy(),
            'robot_b': self.INITIAL_POSES[initial_pose_b]['action'].copy()
        }

        # 调试用计数器
        self._step_count = 0

        # 广播镜头状态缓存（用于平滑运动）
        self._prev_cam_pos = None
        self._prev_lookat = None
        self._prev_azi = None
        self._prev_ele = None
        self._prev_dist = None

        # 数据输出缓存（在 physical_step / reset / set_action 后自动失效）
        self._data_cache: Dict[str, Any] = {}

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "initial_distance": self.initial_distance,
            "debug_torque": self._debug_torque,
            "initial_pose_a": self._initial_pose_a,
            "initial_pose_b": self._initial_pose_b,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "MujocoCombatSimulator":
        return cls(**config)

    def _robot(self, robot_id: str) -> Dict[str, Any]:
        """获取 per-robot 结构化数据"""
        return self._robots[robot_id]

    def _compute_normalization_params(self):
        """计算归一化参数 (reference 和 scale)"""
        self._norm_params = {}
        
        for robot_id in ['robot_a', 'robot_b']:
            jnt_ranges = self._robot(robot_id)['jnt_ranges']  # (21, 2)
            
            lower = jnt_ranges[:, 0]
            upper = jnt_ranges[:, 1]
            
            # Reference = (Down + Up) / 2
            reference = (lower + upper) / 2.0
            
            # Scale = (Up - Down) / 2
            scale = (upper - lower) / 2.0
            
            self._norm_params[robot_id] = {
                'reference': reference.astype(np.float32),
                'scale': scale.astype(np.float32),
                'lower': lower.astype(np.float32),
                'upper': upper.astype(np.float32)
            }
    
    def get_static_data(self) -> Dict[str, Any]:
        """
        获取静态属性 (按 DATASPEC.md §2)

        返回按 robot_a 和 robot_b 分离的字典 + 若干全局键。

        Per-agent fields
        ----------------
        - ``dof_names`` (List[str], len=21): 受控自由度短名（不含后缀）
        - ``body_names`` (List[str]): 该机器人子树内**所有** body 的全名（带后缀，按 body id 稳定排序）
        - ``body_masses_by_name`` (Dict[str, float]): 与 ``body_names`` 对齐的每 body 质量 (kg)
        - ``joint_names`` (List[str]): 子树内全部 joint 的全名
        - ``controlled_joint_names`` (List[str]): 21 个受控 joint 的全名（带后缀）
        - ``root_joint_name`` (str): freejoint 的全名
        - ``keypoint_body_names`` (Dict[str, str]): 语义 → 全名映射
          (``torso``/``head``/``pelvis``/``foot_left``/``foot_right``/``hand_left``/``hand_right``)
        - ``keypoint_joint_names`` (Dict[str, str]): 语义 → 全名映射
          (``ankle_x_left``/``ankle_x_right``/``ankle_y_left``/``ankle_y_right``)
        - ``joint_limits`` (ndarray, shape=(21, 2)): 受控关节物理限位 [min, max] (rad)

        Global fields
        -------------
        - ``dt`` (float): 单个物理子步的仿真时长 (s)
        - ``ground_geom_name`` (str): 地面 geom 名称（用于接触过滤）
        """
        if '_static_data' in self._data_cache:
            return self._data_cache['_static_data']

        result: Dict[str, Any] = {}

        for robot_id in ['robot_a', 'robot_b']:
            cache = self._robot(robot_id)
            body_names = list(cache['body_names'])
            body_masses = np.asarray(cache['body_masses'], dtype=np.float32)
            result[robot_id] = {
                'dof_names': self.CONTROLLED_JOINTS.copy(),
                'body_names': body_names,
                'body_masses_by_name': {
                    name: float(mass) for name, mass in zip(body_names, body_masses)
                },
                'joint_names': list(cache['joint_names']),
                'controlled_joint_names': list(cache['controlled_joint_names']),
                'root_joint_name': cache['root_joint_name'],
                'keypoint_body_names': dict(cache['keypoint_body_names']),
                'keypoint_joint_names': dict(cache['keypoint_joint_names']),
                'joint_limits': cache['jnt_ranges'].copy(),
            }

        # Global
        result['dt'] = float(self.DT)
        result['ground_geom_name'] = 'ground'
        result['ground_geom_id'] = self._ground_geom_id
        result['geom_id_to_name'] = dict(self._geom_id_to_name)
        result['body_id_to_name'] = dict(self._body_id_to_name)
        result['body_id_to_aff'] = dict(self._body_id_to_aff)
        result['geom_id_to_aff'] = dict(self._geom_id_to_aff)

        self._data_cache['_static_data'] = result
        return result
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据 (暂时返回空字典，未来可扩展)"""
        return {}
    
    def get_action(self) -> Dict[str, Any]:
        """获取当前动作目标"""
        if '_action' in self._data_cache:
            return self._data_cache['_action']
        result = {
            'robot_a': self._target_pos_norm['robot_a'].copy(),
            'robot_b': self._target_pos_norm['robot_b'].copy()
        }
        self._data_cache['_action'] = result
        return result
    
    def get_core_state(self) -> Dict[str, Any]:
        """
        获取核心状态 (按 DATASPEC.md 3)

        返回按 robot_a 和 robot_b 分离的字典，每个包含：
        - root_pos (3,): Torso 绝对位置
        - root_rot (4,): Torso 绝对姿态四元数 [w,x,y,z]
        - root_vel_local (3,): Torso 局部线速度
        - root_angular_vel_local (3,): Torso 局部角速度
        - joint_pos_norm (21,): 归一化关节位置 [-1, 1]
        - joint_vel_norm (21,): 归一化关节速度
        """
        if '_core_state' in self._data_cache:
            return self._data_cache['_core_state']

        result = {}

        for robot_id in ['robot_a', 'robot_b']:
            cache = self._robot(robot_id)
            norm_params = self._norm_params[robot_id]

            # Root 位置和姿态
            root_qpos_adr = cache['root_qpos_adr']
            root_pos = self.data.qpos[root_qpos_adr:root_qpos_adr+3].copy()
            root_rot = self.data.qpos[root_qpos_adr+3:root_qpos_adr+7].copy()  # [w,x,y,z]

            # Root 速度 (全局坐标系)
            root_qvel_adr = cache['root_qvel_adr']
            root_vel_global = self.data.qvel[root_qvel_adr:root_qvel_adr+3].copy()
            root_angular_vel_global = self.data.qvel[root_qvel_adr+3:root_qvel_adr+6].copy()

            # 转换到局部坐标系
            rot = R.from_quat([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])  # scipy 用 xyzw
            root_vel_local = rot.inv().apply(root_vel_global)
            root_angular_vel_local = rot.inv().apply(root_angular_vel_global)

            # 关节位置和速度 (原始值)
            qpos_indices = cache['qpos_indices']
            qvel_indices = cache['qvel_indices']
            joint_pos = self.data.qpos[qpos_indices].copy()
            joint_vel = self.data.qvel[qvel_indices].copy()

            # 归一化
            joint_pos_norm = (joint_pos - norm_params['reference']) / norm_params['scale']
            joint_vel_norm = joint_vel / norm_params['scale']

            result[robot_id] = {
                'root_pos': root_pos.astype(np.float32),
                'root_rot': root_rot.astype(np.float32),
                'root_vel_local': root_vel_local.astype(np.float32),
                'root_angular_vel_local': root_angular_vel_local.astype(np.float32),
                'joint_pos_norm': joint_pos_norm.astype(np.float32),
                'joint_vel_norm': joint_vel_norm.astype(np.float32)
            }

        self._data_cache['_core_state'] = result
        return result
    
    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        """获取派生数据 (按 DATASPEC.md §4) — 向量化版本

        Args:
            fields: 需要的字段列表。None 表示返回全部。
                支持的字段: ``torso_distance``, ``contacts``, ``robot_a``, ``robot_b``。
                返回 dict 的 key 与 fields 一一对应。

        全局 (shared)
        -------------
        - ``torso_distance`` (ndarray shape=(1,))
        - ``contacts`` (Dict): SoA 向量化接触数据 (见 DATASPEC §4.1)

        单边视角 (per-agent, in ``robot_a`` / ``robot_b`` keys)
        -----------------------------------------------------
        保留原有: ``root_state`` / ``feet_forces`` / ``opponent_basic_pose`` /
        ``opponent_keypoint_pos`` / ``opponent_keypoint_vel`` / ``observation`` /
        ``uprightness`` / ``opponent_in_local`` + per-body/joint arrays。
        """
        if fields is None:
            fields = ['torso_distance', 'contacts', 'robot_a', 'robot_b']
        else:
            fields = list(fields)
            unknown = set(fields) - {'torso_distance', 'contacts', 'robot_a', 'robot_b'}
            if unknown:
                raise KeyError(f"get_derived_state: unknown fields {unknown}")

        cache_key = ('_derived_state', tuple(fields))
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        result: Dict[str, Any] = {}

        if 'torso_distance' in fields:
            torso_a_id = self._robot('robot_a')['root_body_id']
            torso_b_id = self._robot('robot_b')['root_body_id']
            pos_a = self.data.xpos[torso_a_id]
            pos_b = self.data.xpos[torso_b_id]
            result['torso_distance'] = np.array([np.linalg.norm(pos_b - pos_a)], dtype=np.float32)

        if 'contacts' in fields:
            contacts_vec = self._extract_contacts()
            self._cached_contacts_vec = contacts_vec
            result['contacts'] = contacts_vec

        for rid in ('robot_a', 'robot_b'):
            if rid in fields:
                opp_id = 'robot_b' if rid == 'robot_a' else 'robot_a'
                view = self._get_robot_view(rid, opp_id)
                view.update(self._collect_body_joint_arrays(rid))
                result[rid] = view

        self._data_cache[cache_key] = result
        return result

    def _collect_body_joint_arrays(self, robot_id: str) -> Dict[str, Any]:
        """Extract per-body and per-joint world-frame quantities for a robot.

        Returns a dict of five body-keyed sub-dicts + one joint-keyed dict.
        Arrays are copied so callers cannot mutate the MuJoCo buffers. All
        arrays are ``float32`` per DATASPEC.
        """
        cache = self._robot(robot_id)
        body_ids: np.ndarray = cache['body_ids_sorted']
        body_names: List[str] = cache['body_names']

        xpos = np.asarray(self.data.xpos[body_ids], dtype=np.float32)
        xipos = np.asarray(self.data.xipos[body_ids], dtype=np.float32)
        xquat = np.asarray(self.data.xquat[body_ids], dtype=np.float32)
        cvel = np.asarray(self.data.cvel[body_ids], dtype=np.float32)
        angvel = cvel[:, 0:3]
        linvel = cvel[:, 3:6]

        body_xpos = {name: xpos[i].copy() for i, name in enumerate(body_names)}
        body_xipos = {name: xipos[i].copy() for i, name in enumerate(body_names)}
        body_xquat = {name: xquat[i].copy() for i, name in enumerate(body_names)}
        body_linvel_world = {name: linvel[i].copy() for i, name in enumerate(body_names)}
        body_angvel_world = {name: angvel[i].copy() for i, name in enumerate(body_names)}

        joint_ids_by_name: Dict[str, int] = cache['joint_ids_by_name']
        joint_world_anchor: Dict[str, np.ndarray] = {
            name: np.asarray(self.data.xanchor[jid], dtype=np.float32).copy()
            for name, jid in joint_ids_by_name.items()
        }

        return {
            'body_xpos': body_xpos,
            'body_xipos': body_xipos,
            'body_xquat': body_xquat,
            'body_linvel_world': body_linvel_world,
            'body_angvel_world': body_angvel_world,
            'joint_world_anchor': joint_world_anchor,
        }
    
    def _extract_contacts(self) -> Dict[str, Any]:
        """提取所有接触信息，生成向量化 contacts_vec (SoA)。

        返回:
            contacts_vec: Dict[str, Any] — SoA 向量化接触数据 (见 DATASPEC §4.1)
        """
        ncon = int(self.data.ncon)
        geom_id_to_aff = self._geom_id_to_aff

        # --- 预分配 SoA arrays ---
        geom1_arr = np.empty(ncon, dtype=np.int32)
        geom2_arr = np.empty(ncon, dtype=np.int32)
        body1_arr = np.empty(ncon, dtype=np.int32)
        body2_arr = np.empty(ncon, dtype=np.int32)
        aff1_arr = np.empty(ncon, dtype=np.int8)
        aff2_arr = np.empty(ncon, dtype=np.int8)
        force_mag_arr = np.empty(ncon, dtype=np.float32)
        force_world_arr = np.empty((ncon, 3), dtype=np.float32)
        position_arr = np.empty((ncon, 3), dtype=np.float32)
        normal_arr = np.empty((ncon, 3), dtype=np.float32)
        frame_arr = np.empty((ncon, 3, 3), dtype=np.float32)

        for i in range(ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)

            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])

            c_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_wrench)
            force_contact_on_geom2 = c_wrench[:3]

            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            force_world_on_b = frame.T @ force_contact_on_geom2
            force_magnitude = float(np.linalg.norm(force_contact_on_geom2))
            normal_world = frame[0]
            position_world = np.asarray(contact.pos, dtype=np.float64)

            geom1_arr[i] = g1
            geom2_arr[i] = g2
            body1_arr[i] = b1
            body2_arr[i] = b2
            aff1_arr[i] = geom_id_to_aff.get(g1, 0)
            aff2_arr[i] = geom_id_to_aff.get(g2, 0)
            force_mag_arr[i] = force_magnitude
            force_world_arr[i] = force_world_on_b
            position_arr[i] = position_world
            normal_arr[i] = normal_world
            frame_arr[i] = frame

        return {
            'ncon': ncon,
            'geom1': geom1_arr,
            'geom2': geom2_arr,
            'body1': body1_arr,
            'body2': body2_arr,
            'aff1': aff1_arr,
            'aff2': aff2_arr,
            'force_mag': force_mag_arr,
            'force_world': force_world_arr,
            'position': position_arr,
            'normal': normal_arr,
            'frame': frame_arr,
        }
    
    def _get_robot_view(self, robot_id: str, opponent_id: str) -> Dict[str, Any]:
        """
        获取单个机器人的视角信息

        按照 OBSERVATION_zh.md 返回完整的观测空间:
        - 模块二：全局状态 (13维) - root_state
        - 模块三：触觉力反馈 (2维) - feet_forces
        - 模块四：对手观测 (39维) - opponent_*
        """
        cache = self._robot(robot_id)
        opp_cache = self._robot(opponent_id)

        torso_id = cache['root_body_id']
        opp_torso_id = opp_cache['root_body_id']

        # 自身 Torso 的位置和姿态
        self_pos = self.data.xpos[torso_id]
        self_quat = self.data.xquat[torso_id]  # [w,x,y,z]
        self_rot = R.from_quat([self_quat[1], self_quat[2], self_quat[3], self_quat[0]])

        # 模块二：全局状态 (13维)
        # 1. 高度 (Z轴) - 1维
        height = self_pos[2]

        # 2. 局部朝向 (6维) - 世界坐标四元数 → 局部旋转矩阵（取前两列）
        # 获取自身在世界坐标系中的旋转矩阵
        world_rot_mat = self_rot.as_matrix()  # shape: (3, 3)
        # 提取前两列（局部坐标系的 x 和 y 轴在世界坐标系中的表示）
        # 这样模型可以知道"我的前方朝向"和"我的左侧朝向"
        local_orientation = world_rot_mat[:, :2].T.flatten()  # (6,) - 按列展平

        # 3. 运动速度 (6维)
        root_qvel_adr = cache['root_qvel_adr']
        linear_vel = self.data.qvel[root_qvel_adr:root_qvel_adr+3].copy()  # 全局线速度
        angular_vel = self.data.qvel[root_qvel_adr+3:root_qvel_adr+6].copy()  # 全局角速度

        # 模块三：触觉力反馈 (2维)
        feet_forces = self._get_feet_forces(robot_id)

        # 模块四：对手观测 (39维)
        # 4.1 对手基础位姿 (9维)
        opponent_basic = self._get_opponent_basic_pose(self_pos, self_rot, opp_torso_id)

        # 4.2 对手关键点位置 (15维)
        opponent_keypoint_pos = self._get_opponent_keypoints_pos(
            self_pos, self_rot, opponent_id
        )

        # 4.3 对手关键点速度 (15维)
        opponent_keypoint_vel = self._get_opponent_keypoints_vel(
            self_pos, self_rot, opponent_id
        )

        # 计算模块一本体感知（需要从 get_core_state 获取）
        # 这里先构建占位符，实际使用时需要传入 core_state
        # 为了简化，我们在返回字典中不包含模块一，因为它是从 get_core_state 获取的
        # 完整的平铺观测需要结合 get_core_state 和 get_derived_state

        # 计算模块一本体感知 (42维)
        cache = self._robot(robot_id)
        norm_params = self._norm_params[robot_id]

        # 获取关节位置和速度
        qpos_indices = cache['qpos_indices']
        qvel_indices = cache['qvel_indices']

        joint_pos = self.data.qpos[qpos_indices]
        joint_vel = self.data.qvel[qvel_indices]

        # 归一化
        joint_pos_norm = (joint_pos - norm_params['reference']) / norm_params['scale']
        joint_vel_norm = joint_vel / norm_params['scale']

        # 模块一本体感知 (42维)
        proprioception = np.concatenate([
            joint_pos_norm,  # 21维
            joint_vel_norm,  # 21维
        ]).astype(np.float32)

        return {
            # 模块二：全局状态 (13维)
            'root_state': {
                'height': np.array([height], dtype=np.float32),  # 1维
                'local_orientation': local_orientation.astype(np.float32),  # 6维
                'linear_vel': linear_vel.astype(np.float32),  # 3维
                'angular_vel': angular_vel.astype(np.float32),  # 3维
            },

            # 模块三：触觉力反馈 (2维)
            'feet_forces': feet_forces,  # 2维

            # 模块四：对手观测
            # 4.1 对手基础位姿 (9维)
            'opponent_basic_pose': opponent_basic,

            # 4.2 对手关键点位置 (15维)
            'opponent_keypoint_pos': opponent_keypoint_pos,

            # 4.3 对手关键点速度 (15维)
            'opponent_keypoint_vel': opponent_keypoint_vel,

            # 完整平铺观测 (96维) - 模块一+二+三+四
            'observation': np.concatenate([
                proprioception,        # 42维 - 模块一本体感知
                local_orientation,       # 6维 - 局部朝向
                [height],                  # 1维 - 高度
                linear_vel,               # 3维 - 线速度(全局)
                angular_vel,              # 3维 - 角速度(全局)
                feet_forces,              # 2维 - 足底受力
                opponent_basic['relative_pos'],     # 3维
                opponent_basic['relative_vel'],     # 3维
                opponent_basic['face_vector'],      # 3维
                opponent_keypoint_pos['head'],       # 3维
                opponent_keypoint_pos['hand_right'], # 3维
                opponent_keypoint_pos['hand_left'],  # 3维
                opponent_keypoint_pos['foot_right'], # 3维
                opponent_keypoint_pos['foot_left'],  # 3维
                opponent_keypoint_vel['head'],       # 3维
                opponent_keypoint_vel['hand_right'], # 3维
                opponent_keypoint_vel['hand_left'],  # 3维
                opponent_keypoint_vel['foot_right'], # 3维
                opponent_keypoint_vel['foot_left'],  # 3维
            ]).astype(np.float32),  # 总共 96 维

            # 兼容旧版本
            'uprightness': np.array([world_rot_mat[2, 2]], dtype=np.float32),
            'opponent_in_local': {
                'pos': opponent_basic['relative_pos'],
                'vel': opponent_basic['relative_vel'],
                'rot': opponent_basic['face_vector'],  # FaceVector 替代四元数
            }
        }

    def get_observation(self) -> Dict[str, Any]:
        """Return per-agent flat observation vectors (96-dim).

        通过 get_derived_state(fields=['robot_a', 'robot_b']) 获取 observation。
        """
        if '_observation' in self._data_cache:
            return self._data_cache['_observation']
        derived = self.get_derived_state(['robot_a', 'robot_b'])
        result = {
            "robot_a": derived["robot_a"]["observation"],
            "robot_b": derived["robot_b"]["observation"],
        }
        self._data_cache['_observation'] = result
        return result

    def _get_opponent_basic_pose(
        self,
        self_pos: np.ndarray,
        self_rot: R,
        opp_torso_id: int
    ) -> Dict[str, np.ndarray]:
        """
        获取对手基础位姿 (9维)
        - 相对位置 (3维)
        - 相对速度 (3维)
        - FaceVector (3维) - 对手朝向的单位向量在Ego坐标系中的值
        """
        # 对手 Torso 的位置和姿态
        opp_pos = self.data.xpos[opp_torso_id]
        opp_quat = self.data.xquat[opp_torso_id]  # [w,x,y,z]
        opp_rot = R.from_quat([opp_quat[1], opp_quat[2], opp_quat[3], opp_quat[0]])

        # 对手速度 (全局坐标系)
        opp_vel_global = self.data.cvel[opp_torso_id, 3:6]  # 线速度

        # 1. 相对位置 (3维) - 对手根关节 - 自身根关节
        relative_pos = opp_pos - self_pos
        relative_pos_local = self_rot.inv().apply(relative_pos)

        # 2. 相对速度 (3维)
        relative_vel_local = self_rot.inv().apply(opp_vel_global)

        # 3. FaceVector (3维) - 对手朝向的单位向量在Ego坐标系中的值
        # 对手的"前方"在自身坐标系中表示
        opp_forward = opp_rot.apply([1, 0, 0])  # 对手的局部x轴（前方）
        face_vector = self_rot.inv().apply(opp_forward)  # 转换到自身局部坐标系

        return {
            'relative_pos': relative_pos_local.astype(np.float32),  # 3维
            'relative_vel': relative_vel_local.astype(np.float32),  # 3维
            'face_vector': face_vector.astype(np.float32),  # 3维
        }

    def _get_opponent_keypoints_pos(
        self,
        self_pos: np.ndarray,
        self_rot: R,
        opponent_id: str
    ) -> Dict[str, np.ndarray]:
        """
        获取对手关键点位置 (15维)
        - 头部中心点 (3维)
        - 左右手中心点 (6维)
        - 左右脚中心点 (6维)
        """
        opp_cache = self._robot(opponent_id)

        # 获取各个关键点的位置
        kp = opp_cache['keypoint_body_ids']
        head_pos = self.data.xpos[kp['head']]
        hand_right_pos = self.data.xpos[kp['hand_right']]
        hand_left_pos = self.data.xpos[kp['hand_left']]
        foot_right_pos = self.data.xpos[kp['foot_right']]
        foot_left_pos = self.data.xpos[kp['foot_left']]

        # 转换到自身局部坐标系
        head_local = self_rot.inv().apply(head_pos - self_pos)
        hand_right_local = self_rot.inv().apply(hand_right_pos - self_pos)
        hand_left_local = self_rot.inv().apply(hand_left_pos - self_pos)
        foot_right_local = self_rot.inv().apply(foot_right_pos - self_pos)
        foot_left_local = self_rot.inv().apply(foot_left_pos - self_pos)

        return {
            'head': head_local.astype(np.float32),  # 3维
            'hand_right': hand_right_local.astype(np.float32),  # 3维
            'hand_left': hand_left_local.astype(np.float32),  # 3维
            'foot_right': foot_right_local.astype(np.float32),  # 3维
            'foot_left': foot_left_local.astype(np.float32),  # 3维
        }

    def _get_opponent_keypoints_vel(
        self,
        self_pos: np.ndarray,
        self_rot: R,
        opponent_id: str
    ) -> Dict[str, np.ndarray]:
        """
        获取对手关键点速度 (15维)
        - 头部中心点速度 (3维)
        - 左右手中心点速度 (6维)
        - 左右脚中心点速度 (6维)
        """
        opp_cache = self._robot(opponent_id)

        # 获取各个关键点的速度
        # cvel[frame, 0:3] 是角速度, [3:6] 是线速度
        kp = opp_cache['keypoint_body_ids']
        head_vel = self.data.cvel[kp['head'], 3:6]
        hand_right_vel = self.data.cvel[kp['hand_right'], 3:6]
        hand_left_vel = self.data.cvel[kp['hand_left'], 3:6]
        foot_right_vel = self.data.cvel[kp['foot_right'], 3:6]
        foot_left_vel = self.data.cvel[kp['foot_left'], 3:6]

        # 转换到自身局部坐标系
        head_vel_local = self_rot.inv().apply(head_vel)
        hand_right_vel_local = self_rot.inv().apply(hand_right_vel)
        hand_left_vel_local = self_rot.inv().apply(hand_left_vel)
        foot_right_vel_local = self_rot.inv().apply(foot_right_vel)
        foot_left_vel_local = self_rot.inv().apply(foot_left_vel)

        return {
            'head': head_vel_local.astype(np.float32),  # 3维
            'hand_right': hand_right_vel_local.astype(np.float32),  # 3维
            'hand_left': hand_left_vel_local.astype(np.float32),  # 3维
            'foot_right': foot_right_vel_local.astype(np.float32),  # 3维
            'foot_left': foot_left_vel_local.astype(np.float32),  # 3维
        }
    
    def _get_feet_forces(self, robot_id: str) -> np.ndarray:
        """获取双脚与地面的接触受力 — 向量化版本，使用 contacts_vec"""
        cache = self._robot(robot_id)
        kp = cache['keypoint_body_ids']
        foot_right_id = kp['foot_right']
        foot_left_id = kp['foot_left']

        ground_geom_id = self._ground_geom_id

        cv = getattr(self, '_cached_contacts_vec', None)
        if cv is None:
            cv = self._extract_contacts()
        ncon = cv['ncon']
        if ncon == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        geom1 = cv['geom1']
        geom2 = cv['geom2']
        body1 = cv['body1']
        body2 = cv['body2']
        force_mag = cv['force_mag']

        g1_ground = geom1 == ground_geom_id
        g2_ground = geom2 == ground_geom_id
        ground_mask = g1_ground | g2_ground
        if not np.any(ground_mask):
            return np.array([0.0, 0.0], dtype=np.float32)

        other_body = np.where(g1_ground, body2, body1)
        forces = force_mag[ground_mask]
        bodies = other_body[ground_mask]

        right_force = float(np.sum(forces[bodies == foot_right_id]))
        left_force = float(np.sum(forces[bodies == foot_left_id]))

        return np.array([right_force, left_force], dtype=np.float32)

    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        """
        设置动作 (按 CONTROLSPEC.md)

        输入:
            action: {'robot_a': <action>, 'robot_b': <action>}
            <action> 支持以下类型:
              - np.ndarray  (shape=(21,), dtype=float32)
              - torch.Tensor (CPU / GPU 均可; 含 .detach().cpu().numpy())
              - Python list / tuple (长度 21)
              - None (跳过该机器人)
            每个 action 的值域最终会被裁剪到 [-1, 1]
        """
        self._data_cache.pop('_action', None)

        def _to_numpy(raw: Any) -> np.ndarray:
            """将支持的多类型输入统一转为 np.ndarray(float32, shape=(ACTION_DIM,))."""
            # 1) torch.Tensor: detach -> cpu -> numpy
            if hasattr(raw, 'detach'):
                raw = raw.detach().cpu().numpy()
            # 2) 其他可迭代 / 数组: np.asarray
            arr = np.asarray(raw, dtype=np.float32)
            return arr

        for robot_id in ['robot_a', 'robot_b']:
            if robot_id in action and action[robot_id] is not None:
                act = _to_numpy(action[robot_id])
                if act.shape != (self.ACTION_DIM,):
                    raise ValueError(
                        f"Action for {robot_id} must have shape ({self.ACTION_DIM},), "
                        f"got {act.shape}"
                    )
                # 裁剪到 [-1, 1]
                act = np.clip(act, -1.0, 1.0)
                self._target_pos_norm[robot_id] = act
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """
        重置环境到指定的初始姿态

        Args:
            seed: 随机种子
            options: 可选参数，包括:
                - initial_distance: 初始距离
                - initial_pose_a: 机器人A的初始姿态
                - initial_pose_b: 机器人B的初始姿态
        """
        self._data_cache.clear()
        mujoco.mj_resetData(self.model, self.data)

        # 设置初始距离
        dist = self.initial_distance
        if options and 'initial_distance' in options:
            dist = float(options['initial_distance'])

        # 获取初始姿态配置
        pose_a_name = options.get('initial_pose_a', self._initial_pose_a) if options else self._initial_pose_a
        pose_b_name = options.get('initial_pose_b', self._initial_pose_b) if options else self._initial_pose_b

        pose_a = self.INITIAL_POSES[pose_a_name]
        pose_b = self.INITIAL_POSES[pose_b_name]

        # 设置双方初始位置和姿态
        for robot_id, pose_config, x_offset in [
            ('robot_a', pose_a, -dist/2.0),
            ('robot_b', pose_b, dist/2.0)
        ]:
            cache = self._robot(robot_id)
            norm_params = self._norm_params[robot_id]
            root_qpos_adr = cache['root_qpos_adr']
            qpos_indices = cache['qpos_indices']

            # 设置根部位置 (加上 x_offset 以保持面对面)
            root_pos = pose_config['root_pos'].copy()
            root_pos[0] = x_offset  # x 方向偏移
            self.data.qpos[root_qpos_adr:root_qpos_adr+3] = root_pos

            # 设置根部姿态
            # robot_b 需要旋转 180° 绕 z 轴，使其面向 robot_a
            root_quat = pose_config['root_quat'].copy()
            if robot_id == 'robot_b':
                # 使用 scipy.Rotation 进行四元数旋转
                # MuJoCo 使用 [w,x,y,z] 格式，scipy 使用 [x,y,z,w] 格式
                q_mujoco = root_quat  # [w,x,y,z]
                q_scipy = np.array([q_mujoco[1], q_mujoco[2], q_mujoco[3], q_mujoco[0]])  # [x,y,z,w]

                # 创建原始旋转
                rot_original = R.from_quat(q_scipy)

                # 创建 180° 绕 z 轴的旋转 (使用弧度: π)
                rot_z = R.from_euler('z', np.pi, degrees=False)

                # 应用旋转: q_new = rot_z * q_original
                rot_new = rot_z * rot_original

                # 转换回 MuJoCo 格式 [w,x,y,z]
                q_new_scipy = rot_new.as_quat()  # [x,y,z,w]
                root_quat = np.array([q_new_scipy[3], q_new_scipy[0], q_new_scipy[1], q_new_scipy[2]], dtype=np.float32)

            self.data.qpos[root_qpos_adr+3:root_qpos_adr+7] = root_quat

            # 设置关节位置
            self.data.qpos[qpos_indices] = pose_config['joint_pos']

        # 速度清零
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # 计算控制目标：基于实际设置的关节位置计算对应的 action 值
        # action = (joint_pos - reference) / scale
        for robot_id, pose_config, x_offset in [
            ('robot_a', pose_a, -dist/2.0),
            ('robot_b', pose_b, dist/2.0)
        ]:
            cache = self._robot(robot_id)
            norm_params = self._norm_params[robot_id]
            qpos_indices = cache['qpos_indices']

            # 获取实际设置的关节位置
            actual_joint_pos = self.data.qpos[qpos_indices]

            # 计算对应的 action 值
            action = (actual_joint_pos - norm_params['reference']) / norm_params['scale']
            self._target_pos_norm[robot_id] = action.astype(np.float32)

        # 重置调试计数器
        self._step_count = 0

        # 重置广播镜头状态缓存
        self._prev_cam_pos = None
        self._prev_lookat = None
        self._prev_azi = None
        self._prev_ele = None
        self._prev_dist = None

        mujoco.mj_forward(self.model, self.data)
    
    def physical_step(self) -> None:
        """执行一步物理仿真"""
        self._data_cache.clear()
        if _TURB_DEBUG and (_TURB_DEBUG_MAX_PHYS_STEPS <= 0 or self._step_count <= _TURB_DEBUG_MAX_PHYS_STEPS):
            torso_rows = []
            for robot_id in ['robot_a', 'robot_b']:
                torso_body_id = self._robot(robot_id)['root_body_id']
                applied_force = self.data.xfrc_applied[torso_body_id, :3].copy()
                applied_torque = self.data.xfrc_applied[torso_body_id, 3:6].copy()
                root_qvel_adr = self._robot(robot_id)['root_qvel_adr']
                root_vel = self.data.qvel[root_qvel_adr:root_qvel_adr+3].copy()
                torso_pos = self.data.xpos[torso_body_id].copy()
                torso_rows.append(
                    f"{robot_id}:F=({applied_force[0]:.6f},{applied_force[1]:.6f},{applied_force[2]:.6f}) "
                    f"T=({applied_torque[0]:.6f},{applied_torque[1]:.6f},{applied_torque[2]:.6f}) "
                    f"pos=({torso_pos[0]:.6f},{torso_pos[1]:.6f},{torso_pos[2]:.6f}) "
                    f"vel=({root_vel[0]:.6f},{root_vel[1]:.6f},{root_vel[2]:.6f})"
                )
            print(f"turb_phys_pre step={self._step_count} | " + " | ".join(torso_rows), flush=True)
        # 应用 PD 控制
        self._apply_pd_control()
        
        # 执行物理步
        mujoco.mj_step(self.model, self.data)
        if _TURB_DEBUG and (_TURB_DEBUG_MAX_PHYS_STEPS <= 0 or self._step_count <= _TURB_DEBUG_MAX_PHYS_STEPS):
            torso_rows = []
            for robot_id in ['robot_a', 'robot_b']:
                torso_body_id = self._robot(robot_id)['root_body_id']
                applied_force = self.data.xfrc_applied[torso_body_id, :3].copy()
                root_qvel_adr = self._robot(robot_id)['root_qvel_adr']
                root_vel = self.data.qvel[root_qvel_adr:root_qvel_adr+3].copy()
                torso_pos = self.data.xpos[torso_body_id].copy()
                torso_rows.append(
                    f"{robot_id}:F_after=({applied_force[0]:.6f},{applied_force[1]:.6f},{applied_force[2]:.6f}) "
                    f"pos=({torso_pos[0]:.6f},{torso_pos[1]:.6f},{torso_pos[2]:.6f}) "
                    f"vel=({root_vel[0]:.6f},{root_vel[1]:.6f},{root_vel[2]:.6f})"
                )
            print(f"turb_phys_post step={self._step_count} | " + " | ".join(torso_rows), flush=True)
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

    def _apply_pd_control(self) -> None:
        """应用 PD 控制力矩 (按 CONTROLSPEC.md)"""
        for robot_id in ['robot_a', 'robot_b']:
            cache = self._robot(robot_id)
            norm_params = self._norm_params[robot_id]

            # 获取归一化目标位置
            target_pos_norm = self._target_pos_norm[robot_id]

            # 反归一化: Target_rad = action * scale + reference
            target_pos_rad = target_pos_norm * norm_params['scale'] + norm_params['reference']

            # 获取当前关节状态
            qpos_indices = cache['qpos_indices']
            qvel_indices = cache['qvel_indices']
            current_pos = self.data.qpos[qpos_indices]
            current_vel = self.data.qvel[qvel_indices]

            # PD 控制: Torque = KP * (Target - Current) - KD * Vel
            torque = self.KP * (target_pos_rad - current_pos) - self.KD * current_vel

            # 应用到执行器
            actuator_ids = cache['actuator_ids']
            for i, act_id in enumerate(actuator_ids):
                # 获取 gear 和 ctrlrange
                gear = self.model.actuator_gear[act_id, 0]
                if gear == 0:
                    gear = 1.0

                # Ctrl = Torque / Gear
                ctrl_value = torque[i] / gear

                # 限幅前记录原始力矩（用于调试）
                ctrl_range = self.model.actuator_ctrlrange[act_id]
                max_torque = max(abs(ctrl_range[0]), abs(ctrl_range[1])) * abs(gear)

                # 限幅
                ctrl_value_clipped = np.clip(ctrl_value, ctrl_range[0], ctrl_range[1])
                saturated = abs(ctrl_value) > abs(ctrl_value_clipped)

                self.data.ctrl[act_id] = ctrl_value_clipped

                # 调试打印（每100步打印一次，或者力矩饱和时打印）
                if self._debug_torque:
                    if saturated or (self._step_count % 100 == 0 and self._step_count < 1000):
                        torque_pct = abs(torque[i]) / max_torque * 100 if max_torque > 0 else 0
                        ctrl_pct = abs(ctrl_value_clipped) / max(abs(ctrl_range[0]), abs(ctrl_range[1])) * 100
                        print(f"Step {self._step_count:5d} {robot_id} {self.CONTROLLED_JOINTS[i]:<20}: "
                              f"torque={torque[i]:>8.2f} Nm ({torque_pct:>5.1f}%) "
                              f"ctrl={ctrl_value_clipped:>7.4f} ({ctrl_pct:>5.1f}%) "
                              f"{'SAT!' if saturated else ''}")

        self._step_count += 1
    
    def get_physical_frequency(self) -> float:
        """获取物理仿真频率"""
        return 1.0 / self.dt
    
    def set_core_state(self, state: Dict[str, Any]) -> None:
        """
        设置核心状态

        按照 DATASPEC.md 规范，接受按 robot_id 分离的结构化数据：
        {
            'robot_a': {
                'root_pos': (3,),      # 根节点位置
                'root_rot': (4,),      # 根节点四元数 [w,x,y,z]
                'root_vel_local': (3,),      # 根节点局部线速度
                'root_angular_vel_local': (3,), # 根节点局部角速度
                'joint_pos_norm': (21,), # 归一化关节位置
                'joint_vel_norm': (21,), # 归一化关节速度
            },
            'robot_b': { ... }
        }
        """
        self._data_cache.clear()
        for robot_id in ['robot_a', 'robot_b']:
            if robot_id not in state:
                continue

            robot_state = state[robot_id]
            cache = self._robot(robot_id)
            norm_params = self._norm_params[robot_id]

            root_qpos_adr = cache['root_qpos_adr']
            root_qvel_adr = cache['root_qvel_adr']
            qpos_indices = cache['qpos_indices']
            qvel_indices = cache['qvel_indices']

            # 设置根节点位置和姿态
            if 'root_pos' in robot_state:
                self.data.qpos[root_qpos_adr:root_qpos_adr+3] = robot_state['root_pos']

            if 'root_rot' in robot_state:
                self.data.qpos[root_qpos_adr+3:root_qpos_adr+7] = robot_state['root_rot']

            # 设置根节点速度（需要从局部速度转换到全局速度）
            if 'root_vel_local' in robot_state or 'root_angular_vel_local' in robot_state:
                # 获取当前姿态
                quat = self.data.qpos[root_qpos_adr+3:root_qpos_adr+7]  # [w,x,y,z]
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                rot_mat = rot.as_matrix()

                # 局部速度转全局速度
                if 'root_vel_local' in robot_state:
                    local_vel = robot_state['root_vel_local']
                    global_vel = rot_mat @ local_vel
                    self.data.qvel[root_qvel_adr:root_qvel_adr+3] = global_vel

                if 'root_angular_vel_local' in robot_state:
                    local_angular_vel = robot_state['root_angular_vel_local']
                    global_angular_vel = rot_mat @ local_angular_vel
                    self.data.qvel[root_qvel_adr+3:root_qvel_adr+6] = global_angular_vel

            # 设置关节位置和速度（从归一化值转换回实际值）
            if 'joint_pos_norm' in robot_state:
                joint_pos_norm = robot_state['joint_pos_norm']
                joint_pos = joint_pos_norm * norm_params['scale'] + norm_params['reference']
                self.data.qpos[qpos_indices] = joint_pos

            if 'joint_vel_norm' in robot_state:
                joint_vel_norm = robot_state['joint_vel_norm']
                joint_vel = joint_vel_norm * norm_params['scale']
                self.data.qvel[qvel_indices] = joint_vel

        # 刷新物理引擎状态
        mujoco.mj_forward(self.model, self.data)

    def apply_external_force(
        self,
        body_name: str,
        force: np.ndarray,
        torque: Optional[np.ndarray] = None,
        robot_id: str = "robot_a"
    ) -> None:
        """
        对指定 body 施加外力和/或外力矩

        Args:
            body_name: body 名称（如 'head', 'torso', 'hand_right'）
            force: 3D 力向量 [fx, fy, fz] (牛顿)
            torque: 可选的 3D 力矩向量 [tx, ty, tz] (牛顿·米)
            robot_id: 机器人 ID ('robot_a' 或 'robot_b')

        实现：
            使用 MuJoCo 的 xfrc_applied 字段施加外力
            xfrc_applied shape: (nbody, 6) -> [fx, fy, fz, tx, ty, tz]
            注意：施加的力会在下一个物理步生效，之后自动清零
        """
        if robot_id not in self._robots:
            raise ValueError(f"Unknown robot_id: {robot_id}")

        cache = self._robot(robot_id)
        suffix = cache['suffix']
        full_body_name = f"{body_name}{suffix}"

        # 获取 body ID
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)

        if body_id < 0:
            raise ValueError(f"Body not found: {full_body_name}")

        # 施加外力和力矩到 xfrc_applied
        # xfrc_applied 在每个物理步后自动清零，所以这里直接累加即可
        force = np.asarray(force, dtype=np.float64)
        self.data.xfrc_applied[body_id, :3] += force
        if _TURB_DEBUG and (_TURB_DEBUG_MAX_PHYS_STEPS <= 0 or self._step_count <= _TURB_DEBUG_MAX_PHYS_STEPS):
            current_force = self.data.xfrc_applied[body_id, :3].copy()
            current_torque = self.data.xfrc_applied[body_id, 3:6].copy()
            print(
                f"turb_apply robot={robot_id} body={full_body_name} sim_step={self._step_count} "
                f"input_force=({force[0]:.6f},{force[1]:.6f},{force[2]:.6f}) "
                f"stored_force=({current_force[0]:.6f},{current_force[1]:.6f},{current_force[2]:.6f}) "
                f"stored_torque=({current_torque[0]:.6f},{current_torque[1]:.6f},{current_torque[2]:.6f})",
                flush=True,
            )

        if torque is not None:
            torque = np.asarray(torque, dtype=np.float64)
            self.data.xfrc_applied[body_id, 3:6] += torque
    
    def get_broadcastview_image(self) -> np.ndarray:
        """Broadcast-view camera that always keeps both robots fully in frame.

        Camera design:
        - lookat = bounding-box center of both robots (including fallen height)
        - distance = derived from bounding-box diagonal so both robots fit
        - azimuth = perpendicular to robot-robot axis, auto-flips to whichever
          side keeps the camera inside the arena boundary
        - elevation = -20 deg (fixed downward tilt)
        - EMA smoothing on all parameters to reduce jitter
        """
        try:
            torso_a_id = self._robot('robot_a')['root_body_id']
            torso_b_id = self._robot('robot_b')['root_body_id']

            pos_a = self.data.xpos[torso_a_id]
            pos_b = self.data.xpos[torso_b_id]

            # --- lookat: bounding-box center of both robots (XYZ) ---
            # Use actual heights so fallen robots are included vertically.
            bbox_min = np.minimum(pos_a, pos_b)
            bbox_max = np.maximum(pos_a, pos_b)
            target_lookat = (bbox_min + bbox_max) / 2.0
            # Clamp lookat height to [0.2, 1.2] to avoid camera going underground
            # or overshooting upward when both robots are still standing.
            target_lookat[2] = np.clip(target_lookat[2], 0.2, 0.9)

            # --- azimuth: perpendicular to the robot-robot axis ---
            direction = pos_b - pos_a
            dist_ab = np.linalg.norm(direction[:2])  # horizontal only
            if dist_ab > 1e-6:
                dir_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            else:
                dir_angle = 0.0

            # --- distance: based on robot separation (same as original) ---
            dist_ab = np.linalg.norm(direction[:2])
            want_dist = float(np.clip(dist_ab * 1.5, 2.5, 4.0))

            # --- side selection & arena clamping ---
            # Arena walls at ±3.05 m; keep camera inside ±3.0 m.
            # At ele=-20°, cam horizontal offset = dist * cos(20°) ≈ 0.94*dist.
            # Max safe dist from center = 3.0 / cos(20°) ≈ 3.19 m, so dist<=4.0
            # can push the camera out when lookat is not at center.
            arena_limit = 3.0

            def _max_dist_for_side(azi_deg: float, ele_deg: float,
                                   lookat: np.ndarray) -> float:
                """Max dist along azi_deg that keeps camera inside arena."""
                a = np.radians(azi_deg)
                c = np.cos(np.radians(ele_deg))
                cx = -np.cos(a) * c
                cy = -np.sin(a) * c
                limits = []
                if abs(cx) > 1e-6:
                    limits.append((arena_limit - abs(lookat[0])) / abs(cx))
                if abs(cy) > 1e-6:
                    limits.append((arena_limit - abs(lookat[1])) / abs(cy))
                return float(min(limits)) if limits else 99.0

            azi_option_a = dir_angle + 90.0
            azi_option_b = dir_angle - 90.0

            # Hysteresis side selection: stay on current side unless it can no longer
            # accommodate want_dist; then try the other side.
            prev_side = getattr(self, '_prev_azi_side', None)
            if prev_side is None:
                md_a = _max_dist_for_side(azi_option_a, -20.0, target_lookat)
                md_b = _max_dist_for_side(azi_option_b, -20.0, target_lookat)
                chosen_side = 1 if md_a >= md_b else -1
            else:
                cur_md = _max_dist_for_side(
                    azi_option_a if prev_side == 1 else azi_option_b,
                    -20.0, target_lookat)
                alt_md = _max_dist_for_side(
                    azi_option_b if prev_side == 1 else azi_option_a,
                    -20.0, target_lookat)
                if cur_md >= want_dist:
                    chosen_side = prev_side       # current side fits – stay
                elif alt_md >= want_dist:
                    chosen_side = -prev_side      # flip to other side
                else:
                    chosen_side = prev_side       # both tight – stay

            self._prev_azi_side = chosen_side
            target_azi = azi_option_a if chosen_side == 1 else azi_option_b

            # Resolve elevation: -20° normally; steepen only if want_dist doesn't
            # fit inside the arena at the current elevation.
            target_ele = -20.0
            target_dist = want_dist
            for ele_candidate in np.arange(-20.0, -41.0, -5.0):
                max_d = _max_dist_for_side(target_azi, ele_candidate, target_lookat)
                if want_dist <= max_d or ele_candidate <= -40.0:
                    target_ele = ele_candidate
                    target_dist = float(np.clip(want_dist, 2.0, max_d))
                    break

            # --- EMA smoothing ---
            alpha_pos = 0.15   # faster tracking for dist/azi/ele
            alpha_look = 0.25  # faster lookat tracking

            if self._prev_azi is None:
                azi = target_azi
                ele = target_ele
                dist = target_dist
                lookat = target_lookat.copy()
            else:
                diff = (target_azi - self._prev_azi + 180) % 360 - 180
                azi = self._prev_azi + diff * alpha_pos
                ele = self._prev_ele * (1.0 - alpha_pos) + target_ele * alpha_pos
                dist = self._prev_dist * (1.0 - alpha_pos) + target_dist * alpha_pos
                lookat = self._prev_lookat * (1.0 - alpha_look) + target_lookat * alpha_look

            self._prev_azi = azi
            self._prev_ele = ele
            self._prev_dist = dist
            self._prev_lookat = lookat.copy()

            # Hard clamp: after EMA smoothing the dist may transiently exceed the
            # arena limit. Always enforce that the camera stays inside the arena.
            hard_max = _max_dist_for_side(azi, ele, lookat)
            dist = float(np.clip(dist, 0.5, max(0.5, hard_max)))

            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.distance = dist
            cam.elevation = ele
            cam.azimuth = azi

            renderer = mujoco.Renderer(self.model, height=720, width=1280)
            renderer.update_scene(self.data, camera=cam)
            return renderer.render()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to render broadcast view: {e}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)
