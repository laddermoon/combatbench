import os
# Set EGL backend BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import mujoco
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseSimulator


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
    KP = np.array([
        # 腰部 (abdomen) - 需要较高刚度支撑躯干
        200.0, 200.0, 200.0,
        # 右腿 (hip, knee, ankle) - 承重关节需要高刚度
        250.0, 250.0, 250.0, 300.0, 200.0, 200.0,
        # 左腿
        250.0, 250.0, 250.0, 300.0, 200.0, 200.0,
        # 右臂 (shoulder, elbow) - 末端关节可以较低
        150.0, 150.0, 120.0,
        # 左臂
        150.0, 150.0, 120.0
    ], dtype=np.float32)
    
    KD = np.array([
        # 腰部 - 增加阻尼以减少过冲
        20.0, 20.0, 20.0,
        # 右腿
        25.0, 25.0, 25.0, 30.0, 20.0, 20.0,
        # 左腿
        25.0, 25.0, 25.0, 30.0, 20.0, 20.0,
        # 右臂
        15.0, 15.0, 12.0,
        # 左臂
        15.0, 15.0, 12.0
    ], dtype=np.float32)
    
    # 受控关节名称 (固定顺序)
    CONTROLLED_JOINTS = [
        'abdomen_z', 'abdomen_y', 'abdomen_x',
        'hip_x_right', 'hip_z_right', 'hip_y_right', 'knee_right', 'ankle_y_right', 'ankle_x_right',
        'hip_x_left', 'hip_z_left', 'hip_y_left', 'knee_left', 'ankle_y_left', 'ankle_x_left',
        'shoulder1_right', 'shoulder2_right', 'elbow_right',
        'shoulder1_left', 'shoulder2_left', 'elbow_left'
    ]
    
    def __init__(self, initial_distance: float = 2.0):
        self.dt = self.DT
        self.initial_distance = initial_distance
        self.action_dim = self.ACTION_DIM
        
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
            'robot_a': np.zeros(self.ACTION_DIM, dtype=np.float32),
            'robot_b': np.zeros(self.ACTION_DIM, dtype=np.float32)
        }
    
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
            
            # 脚部 geom (用于接触检测)
            foot_right_name = f"foot_right{suffix}"
            foot_left_name = f"foot_left{suffix}"
            cache['foot_right_body_id'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_right_name)
            cache['foot_left_body_id'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_left_name)
            
            self._robot_cache[robot_id] = cache
    
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
        获取静态属性 (按 DATASPEC.md 2.1)
        
        返回按 robot_a 和 robot_b 分离的字典
        """
        result = {}
        
        for robot_id in ['robot_a', 'robot_b']:
            result[robot_id] = {
                'dof_names': self.CONTROLLED_JOINTS.copy(),
                'body_names': self._get_body_names(robot_id),
                'joint_limits': self._robot_cache[robot_id]['jnt_ranges'].copy()
            }
        
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
        获取派生数据 (按 DATASPEC.md 4)
        
        返回：
        - 全局对抗信息 (torso_distance, combat_contacts)
        - 单边视角信息 (robot_a, robot_b)
        """
        # 全局对抗信息
        torso_a_id = self._robot_cache['robot_a']['torso_body_id']
        torso_b_id = self._robot_cache['robot_b']['torso_body_id']
        
        pos_a = self.data.xpos[torso_a_id]
        pos_b = self.data.xpos[torso_b_id]
        torso_distance = np.linalg.norm(pos_b - pos_a)
        
        # 双方接触
        combat_contacts = self._extract_combat_contacts()
        
        # 单边视角
        robot_a_view = self._get_robot_view('robot_a', 'robot_b')
        robot_b_view = self._get_robot_view('robot_b', 'robot_a')
        
        return {
            'torso_distance': np.array([torso_distance], dtype=np.float32),
            'combat_contacts': combat_contacts,
            'robot_a': robot_a_view,
            'robot_b': robot_b_view
        }
    
    def _extract_combat_contacts(self) -> List[Dict]:
        """提取双方机器人之间的接触"""
        suffix_a = self._robot_cache['robot_a']['suffix']
        suffix_b = self._robot_cache['robot_b']['suffix']
        
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
            
            if not geom1_name or not geom2_name:
                continue
            
            # 判断归属
            is_a1 = geom1_name.endswith(suffix_a)
            is_b1 = geom1_name.endswith(suffix_b)
            is_a2 = geom2_name.endswith(suffix_a)
            is_b2 = geom2_name.endswith(suffix_b)
            
            # 只保留双方之间的碰撞
            if (is_a1 and is_b2) or (is_b1 and is_a2):
                # 计算接触力
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                force = float(np.linalg.norm(c_array[:3]))
                
                # 提取body名称
                body1_id = self.model.geom_bodyid[geom1]
                body2_id = self.model.geom_bodyid[geom2]
                body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                
                contacts.append({
                    'body_a': body1_name if is_a1 else body2_name,
                    'body_b': body2_name if is_b2 else body1_name,
                    'force': force
                })
        
        return contacts
    
    def _get_robot_view(self, robot_id: str, opponent_id: str) -> Dict[str, Any]:
        """获取单个机器人的视角信息"""
        cache = self._robot_cache[robot_id]
        opp_cache = self._robot_cache[opponent_id]
        
        torso_id = cache['torso_body_id']
        opp_torso_id = opp_cache['torso_body_id']
        
        # 自身 Torso 的位置和姿态
        self_pos = self.data.xpos[torso_id]
        self_quat = self.data.xquat[torso_id]  # [w,x,y,z]
        self_rot = R.from_quat([self_quat[1], self_quat[2], self_quat[3], self_quat[0]])
        
        # 直立度 (Torso 局部 z 轴与世界 z 轴的内积)
        local_z = self_rot.apply([0, 0, 1])
        uprightness = float(local_z[2])  # 世界 z 轴是 [0, 0, 1]
        
        # 双脚受力
        feet_forces = self._get_feet_forces(robot_id)
        
        # 对手在局部坐标系下的状态
        opponent_in_local = self._get_opponent_in_local(
            self_pos, self_rot, opp_torso_id
        )
        
        return {
            'uprightness': np.array([uprightness], dtype=np.float32),
            'feet_forces': feet_forces,
            'opponent_in_local': opponent_in_local
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
    
    def _get_opponent_in_local(
        self, 
        self_pos: np.ndarray, 
        self_rot: R, 
        opp_torso_id: int
    ) -> Dict[str, np.ndarray]:
        """获取对手在当前机器人局部坐标系下的完整状态"""
        # 对手 Torso 的位置和姿态
        opp_pos = self.data.xpos[opp_torso_id]
        opp_quat = self.data.xquat[opp_torso_id]  # [w,x,y,z]
        opp_rot = R.from_quat([opp_quat[1], opp_quat[2], opp_quat[3], opp_quat[0]])
        
        # 对手速度 (全局坐标系)
        opp_vel_global = self.data.cvel[opp_torso_id, 3:6]  # 线速度
        opp_angular_vel_global = self.data.cvel[opp_torso_id, 0:3]  # 角速度
        
        # 转换到自身局部坐标系
        relative_pos = opp_pos - self_pos
        pos_local = self_rot.inv().apply(relative_pos)
        vel_local = self_rot.inv().apply(opp_vel_global)
        angular_vel_local = self_rot.inv().apply(opp_angular_vel_global)
        
        # 相对姿态 (对手相对于自身的旋转)
        relative_rot = self_rot.inv() * opp_rot
        rot_quat_xyzw = relative_rot.as_quat()  # [x,y,z,w]
        rot_local = np.array([rot_quat_xyzw[3], rot_quat_xyzw[0], rot_quat_xyzw[1], rot_quat_xyzw[2]], dtype=np.float32)  # [w,x,y,z]
        
        return {
            'pos': pos_local.astype(np.float32),
            'rot': rot_local,
            'vel': vel_local.astype(np.float32),
            'angular_vel': angular_vel_local.astype(np.float32)
        }
    
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
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始距离
        dist = self.initial_distance
        if options and 'initial_distance' in options:
            dist = float(options['initial_distance'])
        
        # 设置双方初始位置 (面对面站立)
        for robot_id, x_offset, quat in [
            ('robot_a', -dist/2.0, [1.0, 0.0, 0.0, 0.0]),  # 面向 +x
            ('robot_b', dist/2.0, [0.0, 0.0, 0.0, 1.0])   # 面向 -x (旋转180度)
        ]:
            cache = self._robot_cache[robot_id]
            root_qpos_adr = cache['root_qpos_adr']
            
            # 位置: [x, y, z]
            self.data.qpos[root_qpos_adr:root_qpos_adr+3] = [x_offset, 0.0, 1.282]
            # 姿态: [w, x, y, z]
            self.data.qpos[root_qpos_adr+3:root_qpos_adr+7] = quat
        
        # 速度清零
        self.data.qvel[:] = 0.0
        
        # 重置控制目标为零位
        self._target_pos_norm['robot_a'][:] = 0.0
        self._target_pos_norm['robot_b'][:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
    
    def physical_step(self) -> None:
        """执行一步物理仿真"""
        # 应用 PD 控制
        self._apply_pd_control()
        
        # 执行物理步
        mujoco.mj_step(self.model, self.data)
    
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
                
                # 限幅
                ctrl_range = self.model.actuator_ctrlrange[act_id]
                ctrl_value = np.clip(ctrl_value, ctrl_range[0], ctrl_range[1])
                
                self.data.ctrl[act_id] = ctrl_value
    
    def get_physical_frequency(self) -> float:
        """获取物理仿真频率"""
        return 1.0 / self.dt
    
    def set_core_state(self, state: Dict[str, Any]) -> None:
        """
        设置核心状态 (用于测试和状态恢复)
        
        注意: 这个方法暂时保留旧的全局 qpos/qvel 接口用于兼容
        """
        if 'qpos' in state:
            self.data.qpos[:] = state['qpos']
        if 'qvel' in state:
            self.data.qvel[:] = state['qvel']
        
        mujoco.mj_forward(self.model, self.data)
    
    def get_broadcastview_image(self) -> np.ndarray:
        """获取广播视角图像 (保留原实现用于可视化)"""
        try:
            torso_a_id = self._robot_cache['robot_a']['torso_body_id']
            torso_b_id = self._robot_cache['robot_b']['torso_body_id']
            
            pos_a = self.data.xpos[torso_a_id]
            pos_b = self.data.xpos[torso_b_id]
            center = (pos_a + pos_b) / 2.0
            
            # 简化的固定视角
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = [center[0], center[1], 1.0]
            cam.distance = 4.0
            cam.elevation = -20.0
            cam.azimuth = 90.0
            
            renderer = mujoco.Renderer(self.model, height=720, width=1280)
            renderer.update_scene(self.data, camera=cam)
            return renderer.render()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to render: {e}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)
