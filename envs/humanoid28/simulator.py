import os
# Set EGL backend BEFORE importing mujoco
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import mujoco
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from envs.framework.backend import BaseSimulator
from envs.humanoid28.meta import Humanoid28Meta


class Humanoid28Simulator(BaseSimulator):
    """
    Humanoid28 双机器人对抗仿真器

    严格按照 CONTROLSPEC.md 和 DATASPEC.md 实现：
    - 弧度制 PD 位置目标控制（MimicKit 控制格式，无归一化层）
    - PD 增益/扭矩上限来自 MimicKit（kp=XML stiffness, kd=XML damping,
      torque cap = motor gear；被动刚度阻尼已剥离）
    - 局部坐标系优先
    - 按机器人隔离的数据接口

    静态元数据 (DT, CONTROLLED_JOINTS, INITIAL_POSES, geom 分类等)
    全部由 Humanoid28Meta 统一管理。
    """

    # 静态参数 — 从 Humanoid28Meta 引用，保持单一数据源
    DT = Humanoid28Meta.DT
    ACTION_DIM = Humanoid28Meta.ACTION_DIM
    ARENA_XML = str(Path(__file__).parent / 'battle_humanoid28.xml')
    CONTROLLED_JOINTS = Humanoid28Meta.CONTROLLED_JOINTS
    INITIAL_POSES = Humanoid28Meta.INITIAL_POSES

    def __init__(self, initial_distance: float = 2.0, debug_torque: bool = False,
                 initial_pose_a: str = 'standing', initial_pose_b: str = 'standing'):
        """
        初始化模拟器

        Args:
            initial_distance: 两个机器人之间的初始距离
            debug_torque: 是否打印力矩调试信息
            initial_pose_a: 机器人A的初始姿态 (当前仅 'standing')
            initial_pose_b: 机器人B的初始姿态
        """
        self.dt = self.DT
        self.initial_distance = initial_distance
        self.action_dim = self.ACTION_DIM
        self._debug_torque = debug_torque

        # PD 增益 / 扭矩上限 / action bounds（来自 MimicKit，经 robot_actuation.json）
        actuation = Humanoid28Meta.load_actuation()
        self._kp = actuation['kp']          # (28,) float32
        self._kd = actuation['kd']          # (28,) float32
        self._gear = actuation['gear']      # (28,) float32
        bounds = Humanoid28Meta.compute_action_bounds()
        self._action_low = bounds['low']    # (28,) float32
        self._action_high = bounds['high']  # (28,) float32

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

        # 校验模型与 Humanoid28Meta 静态元数据一致
        meta_errors = Humanoid28Meta.validate(self.model)
        if meta_errors:
            raise ValueError("Model validation failed:\n" + "\n".join(meta_errors))

        # 构建运行时查找表 + per-robot 结构化数据
        self._meta = Humanoid28Meta.build_runtime_tables(self.model)
        self._robots = self._meta['robots']
        self._env_geom_ids = self._meta['env_geom_ids']
        self._ground_geom_id = self._meta['ground_geom_id']
        self._body_to_robot = self._meta['body_to_robot']
        self._geom_id_to_name = self._meta['geom_id_to_name']
        self._body_id_to_name = self._meta['body_id_to_name']
        self._body_id_to_aff = self._meta['body_id_to_aff']
        self._geom_id_to_aff = self._meta['geom_id_to_aff']

        # 广播相机防出界的场地半径：从 ground geom 的 size[0] 读取
        if self._ground_geom_id >= 0:
            self._arena_radius = float(self.model.geom_size[self._ground_geom_id][0])
        else:
            self._arena_radius = 3.44

        self._build_pd_tables()

        # 初始化控制目标：弧度制位置目标，初值 = 初始姿态关节位置
        # （reset() 会基于实际关节位置重新写入，此处仅为 reset 前的占位）
        self._target_pos = {
            'robot_a': self.INITIAL_POSES[initial_pose_a]['joint_pos'].astype(np.float32).copy(),
            'robot_b': self.INITIAL_POSES[initial_pose_b]['joint_pos'].astype(np.float32).copy(),
        }

        # 调试用计数器
        self._step_count = 0

        # reset() 传入的种子。当前物理仿真为确定性实现（mj_resetData +
        # 无 RNG 源），seed 暂不被消费；保存以备未来随机初始化/RNG 消费
        # 者使用，并满足 BaseSimulator 的种子契约。
        self._seed: Optional[int] = None

        # 广播镜头状态缓存（用于平滑运动）
        self._prev_cam_pos = None
        self._prev_lookat = None
        self._prev_azi = None
        self._prev_ele = None
        self._prev_dist = None
        self._prev_azi_side = None

        # 数据输出缓存（在 physical_step / reset / set_action 后自动失效）
        self._data_cache: Dict[str, Any] = {}

        # 广播渲染器缓存：首次调用 get_broadcastview_image 时惰性创建
        self._renderer = None

    def close(self) -> None:
        """释放资源，特别是缓存的 Renderer 持有的 EGL 上下文 / GPU framebuffer。"""
        if self._renderer is not None:
            self._renderer = None

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "initial_distance": self.initial_distance,
            "debug_torque": self._debug_torque,
            "initial_pose_a": self._initial_pose_a,
            "initial_pose_b": self._initial_pose_b,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "Humanoid28Simulator":
        return cls(**config)

    def _robot(self, robot_id: str) -> Dict[str, Any]:
        """获取 per-robot 结构化数据"""
        return self._robots[robot_id]

    def _build_pd_tables(self):
        """预计算 PD 控制所需的静态执行器参数 (gear / ctrlrange)。"""
        self._pd_tables = {}
        for robot_id in ['robot_a', 'robot_b']:
            actuator_ids = np.asarray(self._robot(robot_id)['actuator_ids'])
            gear = np.array(self.model.actuator_gear[actuator_ids, 0], dtype=np.float64)
            gear[gear == 0] = 1.0
            ctrl_lo = np.array(self.model.actuator_ctrlrange[actuator_ids, 0], dtype=np.float64)
            ctrl_hi = np.array(self.model.actuator_ctrlrange[actuator_ids, 1], dtype=np.float64)
            self._pd_tables[robot_id] = {
                'actuator_ids': actuator_ids,
                'gear': gear,
                'ctrl_lo': ctrl_lo,
                'ctrl_hi': ctrl_hi,
            }

    def get_static_data(self) -> Dict[str, Any]:
        """
        获取静态属性 (按 DATASPEC.md §2)

        Per-agent fields 含 ``joint_limits``（受控关节物理限位 rad）与
        ``action_bounds``（MimicKit 语义的 PD 目标裁剪范围 rad）。
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
                'action_bounds': {
                    'low': self._action_low.copy(),
                    'high': self._action_high.copy(),
                },
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
        """获取传感器数据。当前 arena XML 未定义任何 sensor，恒返回 ``{}``。"""
        return {}

    def get_action(self) -> Dict[str, Any]:
        """获取当前动作目标（弧度制 PD 位置目标）"""
        if '_action' in self._data_cache:
            return self._data_cache['_action']
        result = {
            'robot_a': self._target_pos['robot_a'].copy(),
            'robot_b': self._target_pos['robot_b'].copy()
        }
        self._data_cache['_action'] = result
        return result

    def get_core_state(self) -> Dict[str, Any]:
        """
        获取核心状态 (按 DATASPEC.md §3)

        返回按 robot_a 和 robot_b 分离的字典，每个包含：
        - root_pos (3,): pelvis 绝对位置
        - root_rot (4,): pelvis 绝对姿态四元数 [w,x,y,z]
        - root_vel_local (3,): pelvis 机体系线速度
        - root_angular_vel_local (3,): pelvis 机体系角速度（等效陀螺仪）
        - joint_pos (28,): 关节位置原始弧度值
        - joint_vel (28,): 关节速度原始弧度/秒

        与 humanoid21 的差异：本环境无归一化层，关节量均为原始物理值。
        """
        if '_core_state' in self._data_cache:
            return self._data_cache['_core_state']

        result = {}

        for robot_id in ['robot_a', 'robot_b']:
            cache = self._robot(robot_id)

            # Root 位置和姿态
            root_qpos_adr = cache['root_qpos_adr']
            root_pos = self.data.qpos[root_qpos_adr:root_qpos_adr+3].copy()
            root_rot = self.data.qpos[root_qpos_adr+3:root_qpos_adr+7].copy()  # [w,x,y,z]

            # MuJoCo free joint 的 qvel 两半部分坐标系不一致：
            #   qvel[0:3] 线速度 → 世界系，需乘 R^T 转到机体系
            #   qvel[3:6] 角速度 → 已经是机体系，直接取用
            root_qvel_adr = cache['root_qvel_adr']
            root_vel_global = self.data.qvel[root_qvel_adr:root_qvel_adr+3].copy()

            rot = R.from_quat([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])  # scipy 用 xyzw
            root_vel_local = rot.inv().apply(root_vel_global)
            root_angular_vel_local = self.data.qvel[root_qvel_adr+3:root_qvel_adr+6].copy()

            # 关节位置和速度 (原始弧度值，无归一化)
            qpos_indices = cache['qpos_indices']
            qvel_indices = cache['qvel_indices']
            joint_pos = self.data.qpos[qpos_indices].copy()
            joint_vel = self.data.qvel[qvel_indices].copy()

            result[robot_id] = {
                'root_pos': root_pos.astype(np.float32),
                'root_rot': root_rot.astype(np.float32),
                'root_vel_local': root_vel_local.astype(np.float32),
                'root_angular_vel_local': root_angular_vel_local.astype(np.float32),
                'joint_pos': joint_pos.astype(np.float32),
                'joint_vel': joint_vel.astype(np.float32)
            }

        self._data_cache['_core_state'] = result
        return result

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        """获取派生数据 (按 DATASPEC.md §4)

        Args:
            fields: 需要的字段列表。None 表示返回全部。
                支持的字段: ``root_distance``, ``contacts``, ``robot_a``, ``robot_b``。
        """
        if fields is None:
            fields = ['root_distance', 'contacts', 'robot_a', 'robot_b']
        else:
            fields = list(fields)
            unknown = set(fields) - {'root_distance', 'contacts', 'robot_a', 'robot_b'}
            if unknown:
                raise KeyError(f"get_derived_state: unknown fields {unknown}")

        result: Dict[str, Any] = {}

        if 'root_distance' in fields:
            ck = '_derived_root_distance'
            cached = self._data_cache.get(ck)
            if cached is None:
                root_a_id = self._robot('robot_a')['root_body_id']
                root_b_id = self._robot('robot_b')['root_body_id']
                pos_a = self.data.xpos[root_a_id]
                pos_b = self.data.xpos[root_b_id]
                cached = np.array([np.linalg.norm(pos_b - pos_a)], dtype=np.float32)
                self._data_cache[ck] = cached
            result['root_distance'] = cached

        if 'contacts' in fields:
            ck = '_derived_contacts'
            cached = self._data_cache.get(ck)
            if cached is None:
                cached = self._extract_contacts()
                self._data_cache[ck] = cached
            result['contacts'] = cached

        for rid in ('robot_a', 'robot_b'):
            if rid in fields:
                ck = f'_derived_{rid}'
                cached = self._data_cache.get(ck)
                if cached is None:
                    opp_id = 'robot_b' if rid == 'robot_a' else 'robot_a'
                    view = self._get_robot_view(rid, opp_id)
                    view.update(self._collect_body_joint_arrays(rid))
                    cached = view
                    self._data_cache[ck] = cached
                result[rid] = cached

        return result

    def _collect_body_joint_arrays(self, robot_id: str) -> Dict[str, Any]:
        """Extract per-body and per-joint world-frame quantities for a robot."""
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
        """提取所有接触信息，生成向量化 contacts (SoA)。"""
        ncon = int(self.data.ncon)
        geom_id_to_aff = self._geom_id_to_aff

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
        - 模块一：本体感知 (56维，原始弧度值无压缩)
        - 模块二：全局状态 (13维) - root_state
        - 模块三：触觉力反馈 (2维) - feet_forces
        - 模块四：对手观测 (39维) - opponent_*
        """
        cache = self._robot(robot_id)
        opp_cache = self._robot(opponent_id)

        root_id = cache['root_body_id']
        opp_root_id = opp_cache['root_body_id']

        # 自身 pelvis 的位置和姿态
        self_pos = self.data.xpos[root_id]
        self_quat = self.data.xquat[root_id]  # [w,x,y,z]
        self_rot = R.from_quat([self_quat[1], self_quat[2], self_quat[3], self_quat[0]])
        self_rot_inv = self_rot.inv()

        # 模块二：全局状态
        height = self_pos[2]

        # 重力投影 (3维)：站直时为 (0, 0, -1)；绕重力轴 (yaw) 不变
        world_rot_mat = self_rot.as_matrix()
        projected_gravity = -world_rot_mat[2, :]  # (3,)

        # 运动速度 (6维) —— 均在机体系下表示
        root_qvel_adr = cache['root_qvel_adr']
        linear_vel_world = self.data.qvel[root_qvel_adr:root_qvel_adr+3]
        linear_vel = self_rot_inv.apply(linear_vel_world)  # 机体系线速度
        angular_vel = self.data.qvel[root_qvel_adr+3:root_qvel_adr+6].copy()  # 机体系角速度

        # 模块三：触觉力反馈 (2维)
        feet_forces = self._get_feet_forces(robot_id)

        # 场地中心在机体系中的位置 (3维)
        arena_center_local = self_rot_inv.apply(-self_pos)

        # 模块四：对手观测 (39维)
        opponent_basic = self._get_opponent_basic_pose(
            self_pos, self_rot_inv, opp_root_id,
            self_vel_world=linear_vel_world, opp_cache=opp_cache,
        )
        opponent_keypoint_pos = self._get_opponent_keypoints_pos(
            self_pos, self_rot_inv, opponent_id
        )
        opponent_keypoint_vel = self._get_opponent_keypoints_vel(
            self_pos, self_rot_inv, opponent_id
        )

        # 模块一本体感知 (56维)：关节位置/速度原始弧度值，无归一化无压缩
        # （用户决策：与 MimicKit 的 dof_pos/dof_vel 语义一致）
        qpos_indices = cache['qpos_indices']
        qvel_indices = cache['qvel_indices']
        joint_pos = self.data.qpos[qpos_indices].astype(np.float32)
        joint_vel = self.data.qvel[qvel_indices].astype(np.float32)

        # --- 观测空间非线性变换（仅 root/对手 模块内 humanoid21 惯例的少数维度） ---
        # 关节速度不做任何变换（原始弧度/秒）。
        ang_vel_obs = (np.sign(angular_vel) * np.sqrt(np.abs(angular_vel / 2))).astype(np.float32)
        opp_kp_vel_head = opponent_keypoint_vel['head']  # 头部不做变换
        opp_kp_vel_hands_feet = np.concatenate([
            opponent_keypoint_vel['right_hand'],
            opponent_keypoint_vel['left_hand'],
            opponent_keypoint_vel['right_foot'],
            opponent_keypoint_vel['left_foot'],
        ])
        opp_kp_vel_hands_feet_obs = (
            np.sign(opp_kp_vel_hands_feet) * np.sqrt(np.abs(opp_kp_vel_hands_feet)) / 2
        ).astype(np.float32)

        return {
            # 模块二：全局状态 (13维)
            'root_state': {
                'height': np.array([height], dtype=np.float32),  # 1维
                'projected_gravity': projected_gravity.astype(np.float32),  # 3维
                'linear_vel': linear_vel.astype(np.float32),  # 3维 (机体系)
                'angular_vel': angular_vel.astype(np.float32),  # 3维 (机体系)
                'arena_center_local': arena_center_local.astype(np.float32),  # 3维
            },

            # 模块三：触觉力反馈 (2维)
            'feet_forces': feet_forces,  # 2维

            # 模块四：对手观测
            'opponent_basic_pose': opponent_basic,
            'opponent_keypoint_pos': opponent_keypoint_pos,
            'opponent_keypoint_vel': opponent_keypoint_vel,

            # 完整平铺观测 (110维) - 模块一+二+三+四
            # 关节位置/速度为原始弧度值（无压缩）；
            # 角速度与对手四肢速度按 humanoid21 惯例做 sign*sqrt 变换
            'observation': np.concatenate([
                joint_pos,                             # 28维 - 关节位置 (原始弧度)
                joint_vel,                             # 28维 - 关节速度 (原始弧度/秒)
                projected_gravity,                     # 3维 - 重力投影(机体系)
                [height],                              # 1维 - pelvis 高度
                linear_vel,                            # 3维 - 线速度(机体系)
                ang_vel_obs,                           # 3维 - 角速度(机体系, sqrt 变换)
                feet_forces,                           # 2维 - 足底受力(体重归一化)
                arena_center_local,                    # 3维 - 场地中心在机体系中的位置
                opponent_basic['relative_pos'],        # 3维
                opponent_basic['relative_vel'],        # 3维
                opponent_basic['face_vector'],         # 3维
                opponent_keypoint_pos['head'],         # 3维
                opponent_keypoint_pos['right_hand'],   # 3维
                opponent_keypoint_pos['left_hand'],    # 3维
                opponent_keypoint_pos['right_foot'],   # 3维
                opponent_keypoint_pos['left_foot'],    # 3维
                opp_kp_vel_head,                       # 3维 - 对手头部速度 (不变换)
                opp_kp_vel_hands_feet_obs[:3],         # 3维 - 对手右手速度 (sqrt 变换)
                opp_kp_vel_hands_feet_obs[3:6],        # 3维 - 对手左手速度 (sqrt 变换)
                opp_kp_vel_hands_feet_obs[6:9],        # 3维 - 对手右脚速度 (sqrt 变换)
                opp_kp_vel_hands_feet_obs[9:12],       # 3维 - 对手左脚速度 (sqrt 变换)
            ]).astype(np.float32),  # 总共 110 维

            # 供 observer / analysis 使用的原始量
            'uprightness': np.array([world_rot_mat[2, 2]], dtype=np.float32),
            'opponent_in_local': {
                'pos': opponent_basic['relative_pos'],
                'vel': opponent_basic['relative_vel'],
                'rot': opponent_basic['face_vector'],
            }
        }

    def get_observation(self) -> Dict[str, Any]:
        """Return per-agent flat observation vectors (110-dim)."""
        if '_observation' in self._data_cache:
            return self._data_cache['_observation']
        result = {}
        for rid in ('robot_a', 'robot_b'):
            opp_id = 'robot_b' if rid == 'robot_a' else 'robot_a'
            result[rid] = self._get_robot_view(rid, opp_id)['observation']
        self._data_cache['_observation'] = result
        return result

    def _get_opponent_basic_pose(
        self,
        self_pos: np.ndarray,
        self_rot_inv: R,
        opp_root_id: int,
        self_vel_world: np.ndarray,
        opp_cache: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        获取对手基础位姿 (9维)
        - 相对位置 (3维)
        - 相对速度 (3维)
        - FaceVector (3维) - 对手朝向的单位向量在Ego坐标系中的值

        relative_vel 契约（决策 #14：明确、正确）：
            真正的相对速度 v_rel = v_opp_world − v_self_world（双方均为 root
            freejoint 的世界系线速度），旋转到自身机体系。
            注意这不是 relative_pos 在机系下的严格时间导数（缺少 ω×r 修正项），
            而是"相对速度向量的机系表示"；与 humanoid21 的错误实现
            （仅旋转对手绝对速度、不减自身速度）不同。
        """
        # 对手 pelvis 的位置和姿态
        opp_pos = self.data.xpos[opp_root_id]
        opp_quat = self.data.xquat[opp_root_id]  # [w,x,y,z]
        opp_rot = R.from_quat([opp_quat[1], opp_quat[2], opp_quat[3], opp_quat[0]])

        # 对手 root 速度（freejoint qvel 世界系线速度，与自身同一信源，
        # 保证相对速度两端定义一致）
        opp_qvel_adr = opp_cache['root_qvel_adr']
        opp_vel_global = self.data.qvel[opp_qvel_adr:opp_qvel_adr+3]

        # 1. 相对位置 (3维) - 对手根关节 - 自身根关节
        relative_pos = opp_pos - self_pos
        relative_pos_local = self_rot_inv.apply(relative_pos)

        # 2. 相对速度 (3维) - 真正的相对速度旋转到机系
        relative_vel_world = opp_vel_global - self_vel_world
        relative_vel_local = self_rot_inv.apply(relative_vel_world)

        # 3. FaceVector (3维) - 对手"前方"(局部x轴)在自身坐标系中的表示
        opp_forward = opp_rot.apply([1, 0, 0])
        face_vector = self_rot_inv.apply(opp_forward)

        return {
            'relative_pos': relative_pos_local.astype(np.float32),  # 3维
            'relative_vel': relative_vel_local.astype(np.float32),  # 3维
            'face_vector': face_vector.astype(np.float32),  # 3维
        }

    def _get_opponent_keypoints_pos(
        self,
        self_pos: np.ndarray,
        self_rot_inv: R,
        opponent_id: str
    ) -> Dict[str, np.ndarray]:
        """获取对手关键点位置 (15维)：头/双手/双脚在自身机系下的位置。"""
        opp_cache = self._robot(opponent_id)

        kp = opp_cache['keypoint_body_ids']
        head_pos = self.data.xpos[kp['head']]
        hand_right_pos = self.data.xpos[kp['right_hand']]
        hand_left_pos = self.data.xpos[kp['left_hand']]
        foot_right_pos = self.data.xpos[kp['right_foot']]
        foot_left_pos = self.data.xpos[kp['left_foot']]

        deltas = np.stack([
            head_pos - self_pos,
            hand_right_pos - self_pos,
            hand_left_pos - self_pos,
            foot_right_pos - self_pos,
            foot_left_pos - self_pos,
        ])
        local = self_rot_inv.apply(deltas)

        return {
            'head': local[0].astype(np.float32),
            'right_hand': local[1].astype(np.float32),
            'left_hand': local[2].astype(np.float32),
            'right_foot': local[3].astype(np.float32),
            'left_foot': local[4].astype(np.float32),
        }

    def _get_opponent_keypoints_vel(
        self,
        self_pos: np.ndarray,
        self_rot_inv: R,
        opponent_id: str
    ) -> Dict[str, np.ndarray]:
        """获取对手关键点速度 (15维)：cvel 线速度（世界系）旋转到自身机系。"""
        opp_cache = self._robot(opponent_id)

        kp = opp_cache['keypoint_body_ids']
        head_vel = self.data.cvel[kp['head'], 3:6]
        hand_right_vel = self.data.cvel[kp['right_hand'], 3:6]
        hand_left_vel = self.data.cvel[kp['left_hand'], 3:6]
        foot_right_vel = self.data.cvel[kp['right_foot'], 3:6]
        foot_left_vel = self.data.cvel[kp['left_foot'], 3:6]

        vels = np.stack([
            head_vel,
            hand_right_vel,
            hand_left_vel,
            foot_right_vel,
            foot_left_vel,
        ])
        local = self_rot_inv.apply(vels)

        return {
            'head': local[0].astype(np.float32),
            'right_hand': local[1].astype(np.float32),
            'left_hand': local[2].astype(np.float32),
            'right_foot': local[3].astype(np.float32),
            'left_foot': local[4].astype(np.float32),
        }

    def _get_feet_forces(self, robot_id: str) -> np.ndarray:
        """获取双脚与地面的接触受力，以自身体重 m*g 归一化（无因次）。

        返回 ``[right, left]``，单位为"体重的倍数"。设计动机同 humanoid21：
        原始牛顿值会霸占策略网络第一层输入能量，必须归一化。
        """
        cache = self._robot(robot_id)
        kp = cache['keypoint_body_ids']
        foot_right_id = kp['right_foot']
        foot_left_id = kp['left_foot']
        body_weight = cache['body_weight']

        ground_geom_id = self._ground_geom_id

        cv = self._data_cache.get('_derived_contacts')
        if cv is None:
            cv = self._extract_contacts()
            self._data_cache['_derived_contacts'] = cv
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

        return np.array(
            [right_force / body_weight, left_force / body_weight],
            dtype=np.float32,
        )

    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        """
        设置动作 (按 CONTROLSPEC.md)

        输入:
            action: {'robot_a': <action>, 'robot_b': <action>}
            <action> 支持以下类型:
              - np.ndarray  (shape=(28,), dtype=float32)
              - torch.Tensor (CPU / GPU 均可; 含 .detach().cpu().numpy())
              - Python list / tuple (长度 28)
              - None (跳过该机器人)
            action 语义：弧度制 PD 位置目标（MimicKit 控制格式），
            逐关节裁剪到 action bounds（MimicKit 扩展限位，见
            Humanoid28Meta.compute_action_bounds）。
        """
        self._data_cache.pop('_action', None)

        def _to_numpy(raw: Any) -> np.ndarray:
            if hasattr(raw, 'detach'):
                raw = raw.detach().cpu().numpy()
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
                # 裁剪到 MimicKit action bounds（弧度）
                act = np.clip(act, self._action_low, self._action_high)
                self._target_pos[robot_id] = act

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """
        重置环境到指定的初始姿态

        Args:
            seed: 随机种子（当前确定性实现不消费，仅保存溯源）
            options: 可选参数 initial_distance / initial_pose_a / initial_pose_b
        """
        self._data_cache.clear()
        self._seed = seed
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
            root_qpos_adr = cache['root_qpos_adr']
            qpos_indices = cache['qpos_indices']

            # 设置根部位置 (加上 x_offset 以保持面对面)
            root_pos = pose_config['root_pos'].copy()
            root_pos[0] = x_offset
            self.data.qpos[root_qpos_adr:root_qpos_adr+3] = root_pos

            # 设置根部姿态
            # robot_b 需要旋转 180° 绕 z 轴，使其面向 robot_a
            root_quat = pose_config['root_quat'].copy()
            if robot_id == 'robot_b':
                q_mujoco = root_quat  # [w,x,y,z]
                q_scipy = np.array([q_mujoco[1], q_mujoco[2], q_mujoco[3], q_mujoco[0]])

                rot_original = R.from_quat(q_scipy)
                rot_z = R.from_euler('z', np.pi, degrees=False)
                rot_new = rot_z * rot_original

                q_new_scipy = rot_new.as_quat()  # [x,y,z,w]
                root_quat = np.array([q_new_scipy[3], q_new_scipy[0], q_new_scipy[1], q_new_scipy[2]], dtype=np.float32)

            self.data.qpos[root_qpos_adr+3:root_qpos_adr+7] = root_quat

            # 设置关节位置（原始弧度值）
            self.data.qpos[qpos_indices] = pose_config['joint_pos']

        # 速度清零
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # 控制目标 = 当前关节位置（MimicKit 位置目标语义下，这意味着
        # "保持当前姿态"，无需任何归一化换算）
        for robot_id in ['robot_a', 'robot_b']:
            cache = self._robot(robot_id)
            qpos_indices = cache['qpos_indices']
            actual_joint_pos = self.data.qpos[qpos_indices]
            self._target_pos[robot_id] = actual_joint_pos.astype(np.float32).copy()

        # 重置调试计数器
        self._step_count = 0

        # 重置广播镜头状态缓存（含 side 选择 —— humanoid21 review 指出
        # 漏清 _prev_azi_side 会破坏录像严格可复现性，本环境修正）
        self._prev_cam_pos = None
        self._prev_lookat = None
        self._prev_azi = None
        self._prev_ele = None
        self._prev_dist = None
        self._prev_azi_side = None

        mujoco.mj_forward(self.model, self.data)

    def physical_step(self) -> None:
        """执行一步物理仿真"""
        self._data_cache.clear()
        # 应用 PD 控制
        self._apply_pd_control()

        # 执行物理步
        mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

    def _apply_pd_control(self) -> None:
        """应用 PD 控制力矩 (MimicKit 语义) — 向量化实现。

        与 MimicKit Newton pd_explicit 相同的核心公式：
            tau = kp * (target_pos - q) - kd * qd
            tau = clamp(tau, -torque_lim, +torque_lim)
        其中 torque_lim = motor gear（ctrlrange [-1,1] 下两者数值相等）。
        写入 ctrl = clip(torque / gear, ctrl_lo, ctrl_hi)。
        """
        for robot_id in ['robot_a', 'robot_b']:
            cache = self._robot(robot_id)

            # 弧度制位置目标（无需反归一化）
            target_pos_rad = self._target_pos[robot_id]

            # 获取当前关节状态
            qpos_indices = cache['qpos_indices']
            qvel_indices = cache['qvel_indices']
            current_pos = self.data.qpos[qpos_indices]
            current_vel = self.data.qvel[qvel_indices]

            # PD 控制: Torque = KP * (Target - Current) - KD * Vel
            torque = self._kp * (target_pos_rad - current_pos) - self._kd * current_vel

            # 向量化映射到执行器: Ctrl = clip(Torque / Gear, lo, hi)
            pd = self._pd_tables[robot_id]
            ctrl_value = torque / pd['gear']
            ctrl_value_clipped = np.clip(ctrl_value, pd['ctrl_lo'], pd['ctrl_hi'])
            self.data.ctrl[pd['actuator_ids']] = ctrl_value_clipped

            # 调试打印（每100步打印一次，或者力矩饱和时打印）
            if self._debug_torque:
                gear = pd['gear']
                ctrl_lo = pd['ctrl_lo']
                ctrl_hi = pd['ctrl_hi']
                for i in range(len(torque)):
                    saturated = abs(ctrl_value[i]) > abs(ctrl_value_clipped[i])
                    if saturated or (self._step_count % 100 == 0 and self._step_count < 1000):
                        max_torque = max(abs(ctrl_lo[i]), abs(ctrl_hi[i])) * abs(gear[i])
                        torque_pct = abs(torque[i]) / max_torque * 100 if max_torque > 0 else 0
                        ctrl_pct = abs(ctrl_value_clipped[i]) / max(abs(ctrl_lo[i]), abs(ctrl_hi[i])) * 100
                        print(f"Step {self._step_count:5d} {robot_id} {self.CONTROLLED_JOINTS[i]:<20}: "
                              f"torque={torque[i]:>8.2f} Nm ({torque_pct:>5.1f}%) "
                              f"ctrl={ctrl_value_clipped[i]:>7.4f} ({ctrl_pct:>5.1f}%) "
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
                'root_vel_local': (3,),      # 根节点机体系线速度
                'root_angular_vel_local': (3,), # 根节点机体系角速度
                'joint_pos': (28,), # 关节位置原始弧度值
                'joint_vel': (28,), # 关节速度原始弧度/秒
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

            root_qpos_adr = cache['root_qpos_adr']
            root_qvel_adr = cache['root_qvel_adr']
            qpos_indices = cache['qpos_indices']
            qvel_indices = cache['qvel_indices']

            if 'root_pos' in robot_state:
                self.data.qpos[root_qpos_adr:root_qpos_adr+3] = robot_state['root_pos']

            if 'root_rot' in robot_state:
                self.data.qpos[root_qpos_adr+3:root_qpos_adr+7] = robot_state['root_rot']

            # 与 get_core_state 对称：线速度需从机体系转回世界系再写入 qvel[0:3]，
            # 而角速度本就以机体系存放于 qvel[3:6]，直接写入。
            if 'root_vel_local' in robot_state or 'root_angular_vel_local' in robot_state:
                if 'root_vel_local' in robot_state:
                    quat = self.data.qpos[root_qpos_adr+3:root_qpos_adr+7]  # [w,x,y,z]
                    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                    local_vel = robot_state['root_vel_local']
                    global_vel = rot.as_matrix() @ local_vel
                    self.data.qvel[root_qvel_adr:root_qvel_adr+3] = global_vel

                if 'root_angular_vel_local' in robot_state:
                    self.data.qvel[root_qvel_adr+3:root_qvel_adr+6] = \
                        robot_state['root_angular_vel_local']

            # 关节位置和速度（原始弧度值，直接写入）
            if 'joint_pos' in robot_state:
                self.data.qpos[qpos_indices] = robot_state['joint_pos']

            if 'joint_vel' in robot_state:
                self.data.qvel[qvel_indices] = robot_state['joint_vel']

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
            body_name: body 名称（如 'head', 'torso', 'right_hand'）
            force: 3D 力向量 [fx, fy, fz] (牛顿)
            torque: 可选的 3D 力矩向量 [tx, ty, tz] (牛顿·米)
            robot_id: 机器人 ID ('robot_a' 或 'robot_b')
        """
        if robot_id not in self._robots:
            raise ValueError(f"Unknown robot_id: {robot_id}")

        cache = self._robot(robot_id)
        suffix = cache['suffix']
        full_body_name = f"{body_name}{suffix}"

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)

        if body_id < 0:
            raise ValueError(f"Body not found: {full_body_name}")

        force = np.asarray(force, dtype=np.float64)
        self.data.xfrc_applied[body_id, :3] += force

        if torque is not None:
            torque = np.asarray(torque, dtype=np.float64)
            self.data.xfrc_applied[body_id, 3:6] += torque

    def get_broadcastview_image(self) -> Optional[np.ndarray]:
        """Broadcast-view camera that always keeps both robots fully in frame.

        与 humanoid21 相同的相机逻辑：lookat=双机 bounding-box 中心、
        方位角垂直于双机连线、高度 -20°、EMA 平滑、场地内硬约束。
        """
        try:
            root_a_id = self._robot('robot_a')['root_body_id']
            root_b_id = self._robot('robot_b')['root_body_id']

            pos_a = self.data.xpos[root_a_id]
            pos_b = self.data.xpos[root_b_id]

            # --- lookat: bounding-box center of both robots (XYZ) ---
            bbox_min = np.minimum(pos_a, pos_b)
            bbox_max = np.maximum(pos_a, pos_b)
            target_lookat = (bbox_min + bbox_max) / 2.0
            target_lookat[2] = np.clip(target_lookat[2], 0.2, 0.9)

            # --- azimuth: perpendicular to the robot-robot axis ---
            direction = pos_b - pos_a
            dist_ab = np.linalg.norm(direction[:2])  # horizontal only
            if dist_ab > 1e-6:
                dir_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            else:
                dir_angle = 0.0

            # --- distance: based on robot separation ---
            dist_ab = np.linalg.norm(direction[:2])
            want_dist = float(np.clip(dist_ab * 1.5, 2.5, 4.0))

            # --- side selection & arena clamping ---
            arena_radius = self._arena_radius

            def _max_dist_for_side(azi_deg: float, ele_deg: float,
                                   lookat: np.ndarray) -> float:
                """Max dist along azi_deg that keeps camera inside circular arena."""
                a = np.radians(azi_deg)
                c = np.cos(np.radians(ele_deg))
                dx = -np.cos(a) * c
                dy = -np.sin(a) * c
                A = dx * dx + dy * dy
                B = 2.0 * (lookat[0] * dx + lookat[1] * dy)
                C = lookat[0] ** 2 + lookat[1] ** 2 - arena_radius ** 2
                if A < 1e-12:
                    return 99.0
                disc = B * B - 4.0 * A * C
                if disc < 0:
                    return 0.0
                sqrt_disc = np.sqrt(disc)
                d1 = (-B - sqrt_disc) / (2.0 * A)
                d2 = (-B + sqrt_disc) / (2.0 * A)
                return float(max(0.0, d2))

            azi_option_a = dir_angle + 90.0
            azi_option_b = dir_angle - 90.0

            # Hysteresis side selection
            prev_side = self._prev_azi_side
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
                    chosen_side = prev_side
                elif alt_md >= want_dist:
                    chosen_side = -prev_side
                else:
                    chosen_side = prev_side

            self._prev_azi_side = chosen_side
            target_azi = azi_option_a if chosen_side == 1 else azi_option_b

            # Resolve elevation
            target_ele = -20.0
            target_dist = want_dist
            for ele_candidate in np.arange(-20.0, -41.0, -5.0):
                max_d = _max_dist_for_side(target_azi, ele_candidate, target_lookat)
                if want_dist <= max_d or ele_candidate <= -40.0:
                    target_ele = ele_candidate
                    target_dist = float(np.clip(want_dist, 2.0, max_d))
                    break

            # --- EMA smoothing ---
            alpha_pos = 0.15
            alpha_look = 0.25

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

            # Hard clamp
            hard_max = _max_dist_for_side(azi, ele, lookat)
            dist = float(np.clip(dist, 0.5, max(0.5, hard_max)))

            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.distance = dist
            cam.elevation = ele
            cam.azimuth = azi

            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
            renderer = self._renderer
            renderer.update_scene(self.data, camera=cam)
            return renderer.render()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to render broadcast view: {e}")
            return None
