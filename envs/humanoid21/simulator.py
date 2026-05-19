import os
# Set EGL backend BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import mujoco
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseSimulator


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
    """
    
    # 固定参数
    DT = 0.002
    ACTION_DIM = 21
    ARENA_XML = str(Path(__file__).parent / 'battle_v1.xml')
    
    # 固化的 PD 控制参数 (不可配置)
    # 这些参数通过 ACCEPTANCE_CRITERIA.md 中的测试验证和调优
    # 调优目标: 跟踪误差 <0.05rad, 响应延迟 <0.2s, 控制努力 <30%, 系统稳定
    #
    # 使用较高的 PD 增益以改善跟踪和响应，同时保持控制努力可接受
    # 参数参考 DeepMimic 和 MuJoCo Menagerie 主流 PD 设定的上限值
    KP = np.array([
        # 腹部 (abdomen_z, abdomen_y, abdomen_x) - 战斗中需维持上半身直立
        1000.0, 1000.0, 1000.0,
        # 右腿 (hip_x=roll, hip_z=yaw, hip_y=pitch, knee, ankle_y, ankle_x)
        150.0, 200.0, 200.0, 200.0, 100.0, 100.0,
        # 左腿
        150.0, 200.0, 200.0, 200.0, 100.0, 100.0,
        # 右臂 (shoulder1, shoulder2, elbow)
        150.0, 150.0, 100.0,
        # 左臂
        150.0, 150.0, 100.0
    ], dtype=np.float32)

    KD = np.array([
        # 腹部 - 高阻尼以减少过冲
        100.0, 100.0, 100.0,
        # 右腿 - 踝部较低增益以保持柔顺性
        15.0, 20.0, 20.0, 20.0, 10.0, 10.0,
        # 左腿
        15.0, 20.0, 20.0, 20.0, 10.0, 10.0,
        # 右臂
        15.0, 15.0, 10.0,
        # 左臂
        15.0, 15.0, 10.0
    ], dtype=np.float32)
    
    # 受控关节名称 (固定顺序)
    CONTROLLED_JOINTS = [
        'abdomen_z', 'abdomen_y', 'abdomen_x',
        'hip_x_right', 'hip_z_right', 'hip_y_right', 'knee_right', 'ankle_y_right', 'ankle_x_right',
        'hip_x_left', 'hip_z_left', 'hip_y_left', 'knee_left', 'ankle_y_left', 'ankle_x_left',
        'shoulder1_right', 'shoulder2_right', 'elbow_right',
        'shoulder1_left', 'shoulder2_left', 'elbow_left'
    ]

    # 初始姿态配置（来自 humanoid.xml 的 keyframes）
    # 每个姿态包含 root_pos, root_quat, joint_pos, action
    INITIAL_POSES = {
        'standing': {
            'root_pos': np.array([0, 0, 1.282], dtype=np.float32),
            'root_quat': np.array([1, 0, 0, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0, 0,  # abdomen
                0, 0, 0, 0, 0, 0,  # right leg
                0, 0, 0, 0, 0, 0,  # left leg
                0, 0, 0,  # right arm
                0, 0, 0  # left arm
            ], dtype=np.float32),
            'action': np.array([
                -0.0000, 0.4286, -0.0000,
                0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
                0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
                0.1724, 0.1724, 0.3333, 0.1724, 0.1724, 0.3333,
            ], dtype=np.float32)
        },
        'squat': {
            'root_pos': np.array([0, 0, 0.596], dtype=np.float32),
            'root_quat': np.array([0.988015, 0, 0.154359, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0.4, 0,
                -0.25, -0.5, -2.5, -2.65, -0.8, 0.56,
                -0.25, -0.5, -2.5, -2.65, -0.8, 0.56,
                0, 0, 0,
                0, 0, 0
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4287, 0.0000,
                0.4998, 0.2630, 0.7642, 0.9747, -0.0003, 0.0002,
                0.4998, 0.2630, 0.7642, 0.9747, -0.0003, 0.0002,
                0.1724, 0.1724, 0.3333, 0.1724, 0.1724, 0.3333,
            ], dtype=np.float32)
        },
        'stand_on_left_leg': {
            'root_pos': np.array([0, 0, 1.21948], dtype=np.float32),
            'root_quat': np.array([0.971588, -0.179973, 0.135318, -0.0729076], dtype=np.float32),
            'joint_pos': np.array([
                -0.0516, -0.202, 0.23,
                -0.24, -0.007, -0.34, -1.76, -0.466, -0.0415,
                -0.08, -0.01, -0.37, -0.685, -0.35, -0.09,
                0.109, -0.067, -0.7,
                -0.05, 0.12, 0.16
            ], dtype=np.float32),
            'action': np.array([
                -0.0000, 0.4285, 0.0001,
                0.4998, 0.2632, 0.7646, 0.9749, -0.0002, -0.0000,
                0.4999, 0.2632, 0.7646, 0.9752, -0.0001, -0.0000,
                0.1724, 0.1724, 0.3332, 0.1724, 0.1724, 0.3334,
            ], dtype=np.float32)
        },
        'prone': {
            'root_pos': np.array([0.4, 0, 0.0757706], dtype=np.float32),
            'root_quat': np.array([0.7325, 0, 0.680767, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0.0729, 0,
                0.0077, 0.0019, -0.026, -0.351, -0.27, 0,
                0.0077, 0.0019, -0.026, -0.351, -0.27, 0,
                0.56, -0.62, -1.752,
                0.186, -0.73, -1.73
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4286, 0.0000,
                0.5000, 0.2632, 0.7647, 0.9752, -0.0001, 0.0000,
                0.5000, 0.2632, 0.7647, 0.9752, -0.0001, 0.0000,
                0.1725, 0.1723, 0.3329, 0.1725, 0.1722, 0.3329,
            ], dtype=np.float32)
        },
        'supine': {
            'root_pos': np.array([-0.4, 0, 0.08122], dtype=np.float32),
            'root_quat': np.array([0.722788, 0, -0.69107, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, -0.25, 0,
                0.0182, 0.0142, 0.3, 0.042, -0.44, -0.02,
                0.0182, 0.0142, 0.3, 0.042, -0.44, -0.02,
                0.186, -0.73, -1.73,
                0.186, -0.73, -1.73
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4285, 0.0000,
                0.5000, 0.2632, 0.7648, 0.9753, -0.0002, -0.0000,
                0.5000, 0.2632, 0.7648, 0.9753, -0.0002, -0.0000,
                0.1725, 0.1722, 0.3329, 0.1725, 0.1722, 0.3329,
            ], dtype=np.float32)
        }
    }
    
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

        # 缓存索引和静态参数
        self._cache_robot_indices()
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

    def _cache_robot_indices(self):
        """缓存机器人的关节和body索引"""
        self._robot_cache = {}

        for robot_id, suffix in [('robot_a', '_red'), ('robot_b', '_blue')]:
            cache = {}

            # Root joint (freejoint)
            root_jnt_name = f"root{suffix}"
            root_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, root_jnt_name)
            cache['root_qpos_adr'] = self.model.jnt_qposadr[root_jnt_id]
            cache['root_qvel_adr'] = self.model.jnt_dofadr[root_jnt_id]

            # Torso body (根节点，带 freejoint)
            torso_name = f"torso{suffix}"
            cache['torso_body_id'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, torso_name)
            cache['body_ids'] = self._collect_subtree_body_ids(cache['torso_body_id'])

            # 受控关节索引
            qpos_indices = []
            qvel_indices = []
            actuator_ids = []
            jnt_ranges = []

            for jnt_name in self.CONTROLLED_JOINTS:
                full_name = f"{jnt_name}{suffix}"
                jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)

                if jnt_id < 0:
                    raise ValueError(f"Joint {full_name} not found in model")
                if act_id < 0:
                    raise ValueError(f"Actuator {full_name} not found in model")

                qpos_indices.append(self.model.jnt_qposadr[jnt_id])
                qvel_indices.append(self.model.jnt_dofadr[jnt_id])
                actuator_ids.append(act_id)

                # 检查关节限位
                if not self.model.jnt_limited[jnt_id]:
                    raise ValueError(
                        f"Joint {full_name} has no limits. "
                        f"All joints must have finite limits for normalized control."
                    )
                jnt_ranges.append(self.model.jnt_range[jnt_id].copy())

            cache['qpos_indices'] = np.array(qpos_indices, dtype=np.int32)
            cache['qvel_indices'] = np.array(qvel_indices, dtype=np.int32)
            cache['actuator_ids'] = np.array(actuator_ids, dtype=np.int32)
            cache['jnt_ranges'] = np.array(jnt_ranges, dtype=np.float32)  # shape: (21, 2)
            cache['suffix'] = suffix

            # 脚部 body (用于接触检测)
            foot_right_name = f"foot_right{suffix}"
            foot_left_name = f"foot_left{suffix}"
            cache['foot_right_body_id'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_right_name)
            cache['foot_left_body_id'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_left_name)

            # 关键点 body (用于对手观测)
            cache['head_body_id'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"head{suffix}")
            cache['hand_right_body_id'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hand_right{suffix}")
            cache['hand_left_body_id'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hand_left{suffix}")

            # --- Extended metadata for public IDataAccessor (DATASPEC §2 / §4) ---
            # Deterministic ordering of this robot's body subtree (sorted by body id).
            body_ids_sorted: List[int] = sorted(int(b) for b in cache['body_ids'])
            body_names_subtree: List[str] = []
            body_masses_subtree: List[float] = []
            for bid in body_ids_sorted:
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if not bname:
                    raise ValueError(f"Body id {bid} in {robot_id} subtree has no name")
                body_names_subtree.append(bname)
                body_masses_subtree.append(float(self.model.body_mass[bid]))
            cache['body_ids_sorted'] = np.asarray(body_ids_sorted, dtype=np.int32)
            cache['body_names_subtree'] = body_names_subtree
            cache['body_masses_subtree'] = np.asarray(body_masses_subtree, dtype=np.float32)

            # All joints attached to bodies in this subtree (includes root + controlled
            # + 2-DoF ankle joints etc.). Keyed by name for public access.
            joint_names_subtree: List[str] = []
            joint_ids_by_name: Dict[str, int] = {}
            for bid in body_ids_sorted:
                nj = int(self.model.body_jntnum[bid])
                j_start = int(self.model.body_jntadr[bid])
                for jid in range(j_start, j_start + nj):
                    jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    if jname and jname not in joint_ids_by_name:
                        joint_names_subtree.append(jname)
                        joint_ids_by_name[jname] = int(jid)
            cache['joint_names_subtree'] = joint_names_subtree
            cache['joint_ids_by_name'] = joint_ids_by_name

            # Keypoint body-name map. Observers can look up semantic roles
            # ("torso", "foot_left", ...) without knowing the "_red"/"_blue" suffix.
            cache['keypoint_body_names'] = {
                'torso': f'torso{suffix}',
                'head': f'head{suffix}',
                'pelvis': f'pelvis{suffix}',
                'foot_left': f'foot_left{suffix}',
                'foot_right': f'foot_right{suffix}',
                'hand_left': f'hand_left{suffix}',
                'hand_right': f'hand_right{suffix}',
            }
            # Keypoint joint-name map. Currently exposes the 2-DoF ankle joints
            # (needed by balance observers); expand here when new semantic
            # joint roles are introduced.
            cache['keypoint_joint_names'] = {
                f'ankle_{axis}_{side}': f'ankle_{axis}_{side}{suffix}'
                for axis in ('x', 'y') for side in ('left', 'right')
            }
            cache['root_joint_name'] = root_jnt_name
            cache['controlled_joint_names'] = [
                f"{jn}{suffix}" for jn in self.CONTROLLED_JOINTS
            ]

            self._robot_cache[robot_id] = cache
    
    def _collect_subtree_body_ids(self, root_body_id: int) -> set[int]:
        body_ids = set()
        stack = [root_body_id]

        while stack:
            body_id = stack.pop()
            if body_id in body_ids:
                continue
            body_ids.add(body_id)
            for child_body_id in range(self.model.nbody):
                if child_body_id == body_id:
                    continue
                if int(self.model.body_parentid[child_body_id]) == body_id:
                    stack.append(child_body_id)

        return body_ids
    
    def _compute_normalization_params(self):
        """计算归一化参数 (reference 和 scale)"""
        self._norm_params = {}
        
        for robot_id in ['robot_a', 'robot_b']:
            jnt_ranges = self._robot_cache[robot_id]['jnt_ranges']  # (21, 2)
            
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
        result: Dict[str, Any] = {}

        for robot_id in ['robot_a', 'robot_b']:
            cache = self._robot_cache[robot_id]
            body_names = list(cache['body_names_subtree'])
            body_masses = np.asarray(cache['body_masses_subtree'], dtype=np.float32)
            result[robot_id] = {
                'dof_names': self.CONTROLLED_JOINTS.copy(),
                'body_names': body_names,
                'body_masses_by_name': {
                    name: float(mass) for name, mass in zip(body_names, body_masses)
                },
                'joint_names': list(cache['joint_names_subtree']),
                'controlled_joint_names': list(cache['controlled_joint_names']),
                'root_joint_name': cache['root_joint_name'],
                'keypoint_body_names': dict(cache['keypoint_body_names']),
                'keypoint_joint_names': dict(cache['keypoint_joint_names']),
                'joint_limits': cache['jnt_ranges'].copy(),
            }

        # Global
        result['dt'] = float(self.DT)
        result['ground_geom_name'] = 'ground'

        return result
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据 (暂时返回空字典，未来可扩展)"""
        return {}
    
    def get_action(self) -> Dict[str, Any]:
        """获取当前动作目标"""
        return {
            'robot_a': self._target_pos_norm['robot_a'].copy(),
            'robot_b': self._target_pos_norm['robot_b'].copy()
        }
    
    def _get_body_names(self, robot_id: str) -> List[str]:
        """获取机器人的body名称列表"""
        suffix = self._robot_cache[robot_id]['suffix']
        # 简化版本，只返回主要部位
        return [
            f'torso{suffix}',
            f'head{suffix}',
            f'pelvis{suffix}',
            f'thigh_right{suffix}',
            f'shin_right{suffix}',
            f'foot_right{suffix}',
            f'thigh_left{suffix}',
            f'shin_left{suffix}',
            f'foot_left{suffix}',
            f'upper_arm_right{suffix}',
            f'lower_arm_right{suffix}',
            f'hand_right{suffix}',
            f'upper_arm_left{suffix}',
            f'lower_arm_left{suffix}',
            f'hand_left{suffix}'
        ]
    
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
        result = {}
        
        for robot_id in ['robot_a', 'robot_b']:
            cache = self._robot_cache[robot_id]
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
        
        return result
    
    def get_derived_state(self) -> Dict[str, Any]:
        """
        获取派生数据 (按 DATASPEC.md §4)

        全局 (shared)
        -------------
        - ``torso_distance`` (ndarray shape=(1,))
        - ``robot_robot_contacts`` (List[Dict]): 旧 schema，保留以兼容现有消费者
        - ``robot_environment_contacts`` (List[Dict]): 旧 schema
        - ``contacts`` (List[Dict]): **新** 结构化接触列表，每条含：
            * ``geom_a_name`` / ``geom_b_name`` / ``body_a_name`` / ``body_b_name`` (str)
            * ``position_world`` (ndarray(3,)): 接触点世界坐标
            * ``normal_world`` (ndarray(3,)): 接触法向量 (单位向量, 指向 geom_b)
            * ``frame_world`` (ndarray(3,3)): 接触坐标系 [n; t1; t2] (行存储)
            * ``force_on_body_b_world`` (ndarray(3,)): 世界坐标系下 geom_a 对 geom_b 施加的 3D 力
            * ``force_magnitude`` (float): 上面向量的模

        单边视角 (per-agent, in ``robot_a`` / ``robot_b`` keys)
        -----------------------------------------------------
        保留原有: ``root_state`` / ``feet_forces`` / ``opponent_basic_pose`` /
        ``opponent_keypoint_pos`` / ``opponent_keypoint_vel`` / ``observation`` /
        ``uprightness`` / ``opponent_in_local``.

        **新增** per-body 字段（均 keyed by body 全名，与
        ``get_static_data()[robot_id]['body_names']`` 对齐）:
        - ``body_xpos`` (Dict[str, ndarray(3,)]):  body 坐标系原点世界位置
        - ``body_xipos`` (Dict[str, ndarray(3,)]): body **惯性中心** 世界位置
        - ``body_xquat`` (Dict[str, ndarray(4,)]): body 姿态四元数 [w,x,y,z]
        - ``body_linvel_world`` (Dict[str, ndarray(3,)]): body 线速度 (世界系)
        - ``body_angvel_world`` (Dict[str, ndarray(3,)]): body 角速度 (世界系)

        **新增** per-joint 字段:
        - ``joint_world_anchor`` (Dict[str, ndarray(3,)]): 每个关节铰链锚点的世界坐标。
          对 freejoint 无几何意义，值取 ``data.xanchor[jid]``。
        """
        # 全局对抗信息
        torso_a_id = self._robot_cache['robot_a']['torso_body_id']
        torso_b_id = self._robot_cache['robot_b']['torso_body_id']

        pos_a = self.data.xpos[torso_a_id]
        pos_b = self.data.xpos[torso_b_id]
        torso_distance = np.linalg.norm(pos_b - pos_a)

        # Structured + legacy contacts produced in a single pass
        robot_robot_contacts, robot_environment_contacts, contacts = self._extract_contacts()

        # 单边视角（高层观测）+ per-body/joint 详细物理量
        robot_a_view = self._get_robot_view('robot_a', 'robot_b')
        robot_b_view = self._get_robot_view('robot_b', 'robot_a')
        robot_a_view.update(self._collect_body_joint_arrays('robot_a'))
        robot_b_view.update(self._collect_body_joint_arrays('robot_b'))

        return {
            'torso_distance': np.array([torso_distance], dtype=np.float32),
            'robot_robot_contacts': robot_robot_contacts,
            'robot_environment_contacts': robot_environment_contacts,
            'contacts': contacts,
            'robot_a': robot_a_view,
            'robot_b': robot_b_view
        }

    def _collect_body_joint_arrays(self, robot_id: str) -> Dict[str, Any]:
        """Extract per-body and per-joint world-frame quantities for a robot.

        Returns a dict of five body-keyed sub-dicts + one joint-keyed dict.
        Arrays are copied so callers cannot mutate the MuJoCo buffers. All
        arrays are ``float32`` per DATASPEC.
        """
        cache = self._robot_cache[robot_id]
        body_ids: np.ndarray = cache['body_ids_sorted']
        body_names: List[str] = cache['body_names_subtree']

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
    
    def _extract_contacts(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """提取所有接触信息，同时生成三种视图：

        1. ``robot_robot_contacts`` (legacy): 仅 {body_a, body_b, force 标量}
        2. ``robot_environment_contacts`` (legacy): 机器人 ↔ 环境接触，仅模
        3. ``contacts`` (new, DATASPEC §4.1): 全部接触 + 方向/位置/力向量/帧矩阵

        新增的 ``contacts`` 用于替代观察者直接读取 ``data.contact`` 的需求。
        """
        body_ids_a = self._robot_cache['robot_a']['body_ids']
        body_ids_b = self._robot_cache['robot_b']['body_ids']

        robot_robot_contacts: List[Dict[str, Any]] = []
        robot_environment_contacts: List[Dict[str, Any]] = []
        contacts: List[Dict[str, Any]] = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            body1_id = int(self.model.geom_bodyid[geom1])
            body2_id = int(self.model.geom_bodyid[geom2])
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ''
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ''
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or ''
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or ''

            c_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_wrench)
            force_contact_on_geom2 = c_wrench[:3]

            # contact.frame is row-major [n; t1; t2] in world coords → 3x3
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            # Rotate contact-frame linear force into world frame. MuJoCo's
            # convention: mj_contactForce returns force *on geom2 by geom1* in
            # the contact frame.
            force_world_on_b = frame.T @ force_contact_on_geom2
            force_magnitude = float(np.linalg.norm(force_contact_on_geom2))
            normal_world = frame[0]  # first row is the contact normal in world
            position_world = np.asarray(contact.pos, dtype=np.float64)

            contacts.append({
                'geom_a_name': geom1_name,
                'geom_b_name': geom2_name,
                'body_a_name': body1_name,
                'body_b_name': body2_name,
                'position_world': position_world.astype(np.float32),
                'normal_world': normal_world.astype(np.float32),
                'frame_world': frame.astype(np.float32),
                'force_on_body_b_world': force_world_on_b.astype(np.float32),
                'force_magnitude': force_magnitude,
            })

            is_a1 = body1_id in body_ids_a
            is_b1 = body1_id in body_ids_b
            is_a2 = body2_id in body_ids_a
            is_b2 = body2_id in body_ids_b

            if (is_a1 and is_b2) or (is_b1 and is_a2):
                robot_robot_contacts.append({
                    'body_a': body1_name if is_a1 else body2_name,
                    'body_b': body2_name if is_b2 else body1_name,
                    'force': force_magnitude,
                })
            elif (is_a1 or is_b1) != (is_a2 or is_b2):
                if is_a1 or is_b1:
                    robot_environment_contacts.append({
                        'robot': 'robot_a' if is_a1 else 'robot_b',
                        'body': body1_name,
                        'environment_geom': geom2_name,
                        'environment_body': body2_name,
                        'force': force_magnitude,
                    })
                else:
                    robot_environment_contacts.append({
                        'robot': 'robot_a' if is_a2 else 'robot_b',
                        'body': body2_name,
                        'environment_geom': geom1_name,
                        'environment_body': body1_name,
                        'force': force_magnitude,
                    })

        return robot_robot_contacts, robot_environment_contacts, contacts
    
    def _get_robot_view(self, robot_id: str, opponent_id: str) -> Dict[str, Any]:
        """
        获取单个机器人的视角信息

        按照 OBSERVATION_zh.md 返回完整的观测空间:
        - 模块二：全局状态 (13维) - root_state
        - 模块三：触觉力反馈 (2维) - feet_forces
        - 模块四：对手观测 (39维) - opponent_*
        """
        cache = self._robot_cache[robot_id]
        opp_cache = self._robot_cache[opponent_id]

        torso_id = cache['torso_body_id']
        opp_torso_id = opp_cache['torso_body_id']

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
        cache = self._robot_cache[robot_id]
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

        Mirrors :class:`Humanoid21Observer` — extracts the ``observation``
        field that ``get_derived_state`` already computes for each robot.
        """
        derived = self.get_derived_state()
        return {
            "robot_a": derived["robot_a"]["observation"],
            "robot_b": derived["robot_b"]["observation"],
        }

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
        opp_cache = self._robot_cache[opponent_id]

        # 获取各个关键点的位置
        head_pos = self.data.xpos[opp_cache['head_body_id']]
        hand_right_pos = self.data.xpos[opp_cache['hand_right_body_id']]
        hand_left_pos = self.data.xpos[opp_cache['hand_left_body_id']]
        foot_right_pos = self.data.xpos[opp_cache['foot_right_body_id']]
        foot_left_pos = self.data.xpos[opp_cache['foot_left_body_id']]

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
        opp_cache = self._robot_cache[opponent_id]

        # 获取各个关键点的速度
        # cvel[frame, 0:3] 是角速度, [3:6] 是线速度
        head_vel = self.data.cvel[opp_cache['head_body_id'], 3:6]
        hand_right_vel = self.data.cvel[opp_cache['hand_right_body_id'], 3:6]
        hand_left_vel = self.data.cvel[opp_cache['hand_left_body_id'], 3:6]
        foot_right_vel = self.data.cvel[opp_cache['foot_right_body_id'], 3:6]
        foot_left_vel = self.data.cvel[opp_cache['foot_left_body_id'], 3:6]

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
        """获取双脚与地面的接触受力"""
        cache = self._robot_cache[robot_id]
        foot_right_id = cache['foot_right_body_id']
        foot_left_id = cache['foot_left_body_id']
        
        right_force = 0.0
        left_force = 0.0
        
        # 地面 geom id (通常是 0)
        ground_geom_id = 0
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # 检查是否涉及地面
            if geom1 != ground_geom_id and geom2 != ground_geom_id:
                continue
            
            # 获取非地面的 geom 对应的 body
            other_geom = geom2 if geom1 == ground_geom_id else geom1
            body_id = self.model.geom_bodyid[other_geom]
            
            # 计算接触力
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            force = float(np.linalg.norm(c_array[:3]))
            
            # 累加到对应的脚
            if body_id == foot_right_id:
                right_force += force
            elif body_id == foot_left_id:
                left_force += force
        
        return np.array([right_force, left_force], dtype=np.float32)

    def set_action(self, action: Dict[str, Optional[np.ndarray]]) -> None:
        """
        设置动作 (按 CONTROLSPEC.md)
        
        输入:
            action: {'robot_a': ndarray(21,), 'robot_b': ndarray(21,)}
            每个 action 的值域为 [-1, 1]
        """
        for robot_id in ['robot_a', 'robot_b']:
            if robot_id in action and action[robot_id] is not None:
                act = np.asarray(action[robot_id], dtype=np.float32)
                if act.shape != (self.ACTION_DIM,):
                    raise ValueError(f"Action for {robot_id} must have shape ({self.ACTION_DIM},), got {act.shape}")
                
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
            cache = self._robot_cache[robot_id]
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
            cache = self._robot_cache[robot_id]
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
        if _TURB_DEBUG and (_TURB_DEBUG_MAX_PHYS_STEPS <= 0 or self._step_count <= _TURB_DEBUG_MAX_PHYS_STEPS):
            torso_rows = []
            for robot_id in ['robot_a', 'robot_b']:
                torso_body_id = self._robot_cache[robot_id]['torso_body_id']
                applied_force = self.data.xfrc_applied[torso_body_id, :3].copy()
                applied_torque = self.data.xfrc_applied[torso_body_id, 3:6].copy()
                root_qvel_adr = self._robot_cache[robot_id]['root_qvel_adr']
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
                torso_body_id = self._robot_cache[robot_id]['torso_body_id']
                applied_force = self.data.xfrc_applied[torso_body_id, :3].copy()
                root_qvel_adr = self._robot_cache[robot_id]['root_qvel_adr']
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
            cache = self._robot_cache[robot_id]
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
        for robot_id in ['robot_a', 'robot_b']:
            if robot_id not in state:
                continue

            robot_state = state[robot_id]
            cache = self._robot_cache[robot_id]
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
        if robot_id not in self._robot_cache:
            raise ValueError(f"Unknown robot_id: {robot_id}")

        cache = self._robot_cache[robot_id]
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
        """
        获取广播视角图像（智能动态跟踪版本）

        相机特性：
        - 方位角随机器人移动自动调整（始终从侧面观看）
        - 距离根据机器人间距动态缩放
        - 有防撞墙边界检测
        - 使用EMA平滑减少镜头抖动
        """
        try:
            torso_a_id = self._robot_cache['robot_a']['torso_body_id']
            torso_b_id = self._robot_cache['robot_b']['torso_body_id']

            pos_a = self.data.xpos[torso_a_id]
            pos_b = self.data.xpos[torso_b_id]
            center = (pos_a + pos_b) / 2.0

            # 基础视角：两个机器人的中心，高度略降低（腰部高度）
            target_lookat = center.copy()
            target_lookat[2] = 1.0  # 固定观察高度为腰部

            # 计算两个机器人之间的方向向量
            direction = pos_b - pos_a
            dist_ab = np.linalg.norm(direction)
            if dist_ab > 1e-6:
                direction = direction / dist_ab
            else:
                direction = np.array([1.0, 0.0, 0.0])

            # 期望从侧面观看两个机器人（方位角对应direction的法向量）
            # arctan2(y, x) 获取向量在XY平面上的角度
            dir_angle = np.degrees(np.arctan2(direction[1], direction[0]))

            # 相机在侧面，所以方位角 + 90度
            target_azi = dir_angle + 90.0
            target_ele = -20.0  # 俯视20度

            # 相机距离：基础距离为间距 * 1.5，限制在 2.5 到 4.0 之间
            target_dist = max(2.5, min(4.0, dist_ab * 1.5))

            # --- 边界限制（防止相机移出墙外）---
            # 房间边界约为 x,y ∈ [-3.05, 3.05]
            # 预留 0.5 安全距离 -> 墙边界限制在 2.55
            limit = 2.55

            # 在MuJoCo中，给定azimuth、elevation和distance，相机的世界坐标系水平偏移约为：
            # dx = -dist * cos(azi) * cos(ele)
            # dy = -dist * sin(azi) * cos(ele)
            azi_rad = np.radians(target_azi)
            ele_rad = np.radians(target_ele)

            dx = -target_dist * np.cos(azi_rad) * np.cos(ele_rad)
            dy = -target_dist * np.sin(azi_rad) * np.cos(ele_rad)

            cam_x = target_lookat[0] + dx
            cam_y = target_lookat[1] + dy

            # 如果期望X超出房间，缩短距离以接近墙壁
            if abs(cam_x) > limit:
                max_dx = limit - target_lookat[0] if cam_x > 0 else -limit - target_lookat[0]
                factor = -np.cos(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dx / factor))

            # 如果期望Y超出房间
            if abs(cam_y) > limit:
                max_dy = limit - target_lookat[1] if cam_y > 0 else -limit - target_lookat[1]
                factor = -np.sin(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dy / factor))

            # --- 平滑滤波（EMA）---
            alpha_pos = 0.05  # 极坐标和距离的平滑系数
            alpha_look = 0.1  # 观察焦点的平滑系数

            if self._prev_azi is None:
                # 首次渲染，直接使用目标值
                azi = target_azi
                ele = target_ele
                dist = target_dist
                lookat = target_lookat.copy()
            else:
                # 角度平滑需要处理360度循环跳变
                diff = (target_azi - self._prev_azi + 180) % 360 - 180
                azi = self._prev_azi + diff * alpha_pos
                ele = self._prev_ele * (1.0 - alpha_pos) + target_ele * alpha_pos
                dist = self._prev_dist * (1.0 - alpha_pos) + target_dist * alpha_pos
                lookat = self._prev_lookat * (1.0 - alpha_look) + target_lookat * alpha_look

            # 更新缓存
            self._prev_azi = azi
            self._prev_ele = ele
            self._prev_dist = dist
            self._prev_lookat = lookat.copy()

            # 设置相机参数
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.distance = dist
            cam.elevation = ele
            cam.azimuth = azi

            # 渲染
            renderer = mujoco.Renderer(self.model, height=720, width=1280)
            renderer.update_scene(self.data, camera=cam)
            return renderer.render()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to render broadcast view: {e}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)
