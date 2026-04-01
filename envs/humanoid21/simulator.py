import os
# Set EGL backend BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import mujoco
import numpy as np
from typing import Any, Dict, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseSimulator

class MujocoCombatSimulator(BaseSimulator):
    # 固定参数
    DT = 0.002
    ACTION_DIM = 21
    KP = 50.0
    KD = 5.0
    ARENA_XML = str(Path(__file__).parent / 'battle_v1.xml')

    def __init__(self, initial_distance: float = 2.0):
        self.dt = self.DT
        self.initial_distance = initial_distance
        self.action_dim = self.ACTION_DIM

        self.model = mujoco.MjSpec.from_file(self.ARENA_XML).compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.model.opt.timestep = self.DT

        self._action = {'robot_a': np.zeros(self.ACTION_DIM), 'robot_b': np.zeros(self.ACTION_DIM)}

        # 缓存关节点和索引
        self._cache_indices()

        # PD 控制器参数
        self.kp = np.full(self.ACTION_DIM, self.KP, dtype=np.float32)
        self.kd = np.full(self.ACTION_DIM, self.KD, dtype=np.float32)
        self.target_positions = {
            'robot_a': np.zeros(self.ACTION_DIM, dtype=np.float32),
            'robot_b': np.zeros(self.ACTION_DIM, dtype=np.float32)
        }
        self._pd_initialized = False
        
        # 用于摄像机平滑
        self._prev_azi = None
        self._prev_ele = None
        self._prev_dist = None
        self._prev_lookat = None
        
    def _cache_indices(self):
        self.robot_info = {'robot_a': {}, 'robot_b': {}}
        for robot_id, suffix in [('robot_a', '_red'), ('robot_b', '_blue')]:
            # Root joint for humanoid is a free joint
            root_jnt_name = f"root{suffix}"
            root_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, root_jnt_name)
            qpos_adr = self.model.jnt_qposadr[root_jnt_id]
            qvel_adr = self.model.jnt_dofadr[root_jnt_id]
            
            body_name = f"pelvis{suffix}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            
            self.robot_info[robot_id] = {
                'body_id': body_id,
                'root_jnt_id': root_jnt_id,
                'qpos_adr': qpos_adr,
                'qvel_adr': qvel_adr,
                'suffix': suffix
            }
            
            # actuator indices
            # humanoid21 has 21 DOFs per robot
            # In old code: CONTROLLED_JOINTS
            controlled_joints = [
                'abdomen_z', 'abdomen_y', 'abdomen_x',
                'hip_x_right', 'hip_z_right', 'hip_y_right', 'knee_right', 'ankle_y_right', 'ankle_x_right',
                'hip_x_left', 'hip_z_left', 'hip_y_left', 'knee_left', 'ankle_y_left', 'ankle_x_left',
                'shoulder1_right', 'shoulder2_right', 'elbow_right',
                'shoulder1_left', 'shoulder2_left', 'elbow_left'
            ]
            actuators = []
            qpos_indices = []
            qvel_indices = []
            jnt_ranges = []
            ctrl_ranges = []
            qpos0_list = []
            
            for j in controlled_joints:
                full_name = f"{j}{suffix}"
                j_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
                act_name = f"{j}{suffix}"
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                
                actuators.append(act_id)
                if j_idx >= 0:
                    qpos_adr = self.model.jnt_qposadr[j_idx]
                    qpos_indices.append(qpos_adr)
                    qvel_indices.append(self.model.jnt_dofadr[j_idx])
                    
                    if self.model.jnt_limited[j_idx]:
                        jnt_ranges.append(self.model.jnt_range[j_idx].copy())
                    else:
                        import numpy as np
                        jnt_ranges.append(np.array([-np.pi, np.pi]))
                    
                    qpos0_list.append(self.model.qpos0[qpos_adr])
                else:
                    qpos_indices.append(-1)
                    qvel_indices.append(-1)
                    import numpy as np
                    jnt_ranges.append(np.array([-np.pi, np.pi]))
                    qpos0_list.append(0.0)
                
                if act_id >= 0:
                    ctrl_ranges.append(self.model.actuator_ctrlrange[act_id].copy())
                else:
                    import numpy as np
                    ctrl_ranges.append(np.array([-1.0, 1.0]))
                    
            self.robot_info[robot_id]['actuators'] = actuators
            self.robot_info[robot_id]['qpos_indices'] = qpos_indices
            self.robot_info[robot_id]['qvel_indices'] = qvel_indices
            self.robot_info[robot_id]['jnt_ranges'] = jnt_ranges
            self.robot_info[robot_id]['ctrl_ranges'] = ctrl_ranges
            self.robot_info[robot_id]['qpos0'] = qpos0_list


    def get_static_data(self) -> Dict[str, Any]:
        return {
            'dt': self.dt,
            'robot_info': self.robot_info
        }

    def get_core_state(self) -> Dict[str, Any]:
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'time': self.data.time,
        }

    def set_core_state(self, state: Dict[str, Any]) -> None:
        """
        设置核心状态（qpos, qvel）。

        注意：time 字段被忽略。MuJoCo 内部时间由仿真引擎自动管理，
        不应手动修改，否则可能导致物理仿真结果异常。
        """
        if 'time' in state and state['time'] != self.data.time:
            import warnings
            warnings.warn(
                "Ignoring 'time' in set_core_state: simulation time is managed "
                "internally by MuJoCo and should not be modified manually.",
                UserWarning
            )

        self.data.qpos[:] = state['qpos']
        self.data.qvel[:] = state['qvel']
        # time 保持不变，由仿真引擎自动管理

        mujoco.mj_forward(self.model, self.data)

    def get_derived_state(self) -> Dict[str, Any]:
        """
        获取派生状态（碰撞、位置等）。

        返回 robot_a 和 robot_b 之间的碰撞，以及两个机器人的位置数据。
        """
        # 获取机器人后缀用于识别归属
        suffix_a = self.robot_info['robot_a']['suffix']  # '_red'
        suffix_b = self.robot_info['robot_b']['suffix']  # '_blue'

        # 只收集 robot_a vs robot_b 的碰撞
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)

            # 判断 geom 归属
            is_a1 = geom1_name.endswith(suffix_a) if geom1_name else False
            is_b1 = geom1_name.endswith(suffix_b) if geom1_name else False
            is_a2 = geom2_name.endswith(suffix_a) if geom2_name else False
            is_b2 = geom2_name.endswith(suffix_b) if geom2_name else False

            # 只保留 robot_a vs robot_b 的碰撞
            if ((is_a1 and is_b2) or (is_b1 and is_a2)):
                body1 = self.model.geom_bodyid[geom1]
                body2 = self.model.geom_bodyid[geom2]

                # 获取接触力（牛顿）
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                force = np.linalg.norm(c_array[:3])

                contacts.append({
                    'geom_a': geom1,
                    'geom_b': geom2,
                    'body_a': body1,
                    'body_b': body2,
                    'position': contact.pos.copy(),
                    'normal': contact.frame[:3].copy(),
                    'force': force
                })

        # 提取 robot_a 和 robot_b 的 body 数据
        # robot_a: bodies 1-16, robot_b: bodies 17-32
        robot_a_data = self._extract_robot_data('robot_a')
        robot_b_data = self._extract_robot_data('robot_b')

        return {
            'contacts': contacts,
            'robot_a': robot_a_data,
            'robot_b': robot_b_data
        }

    def _extract_robot_data(self, robot_id: str) -> Dict[str, np.ndarray]:
        """
        提取指定机器人的位置和速度数据

        返回:
            xpos: (16, 3) - body 位置
            xvelp: (16, 3) - 线速度
            xquat: (16, 4) - 四元数
            opponent_pos: (3,) - 对手在自身坐标系中的位置
            opponent_facing: (3,) - 对手朝向在自身坐标系中的前向量
            foot_contact: dict - 两脚与地面的接触信息 {'right': bool, 'left': bool}
            foot_force: dict - 双脚受力 {'right': float, 'left': float, 'contact_points': list}
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        # 找到属于该机器人的 body 索引范围
        if robot_id == 'robot_a':
            # bodies 1-16 (torso_red to hand_left_red)
            start_body = 1
            end_body = 17
            root_body = self.robot_info['robot_a']['body_id']
            opponent_id = 'robot_b'
            opponent_root = self.robot_info['robot_b']['body_id']
            # 脚 geom: 13, 14 (right), 17, 18 (left)
            foot_geoms = [13, 14, 17, 18]
            right_foot_geoms = [13, 14]
            left_foot_geoms = [17, 18]
        else:  # robot_b
            # bodies 17-32 (torso_blue to hand_left_blue)
            start_body = 17
            end_body = 33
            root_body = self.robot_info['robot_b']['body_id']
            opponent_id = 'robot_a'
            opponent_root = self.robot_info['robot_a']['body_id']
            # 脚 geom: 32, 33 (right), 36, 37 (left)
            foot_geoms = [32, 33, 36, 37]
            right_foot_geoms = [32, 33]
            left_foot_geoms = [36, 37]

        # 提取 body 数据
        data = {
            'xpos': self.data.xpos[start_body:end_body].copy(),
            'xvelp': self.data.cvel[start_body:end_body, 3:].copy(),
            'xquat': self.data.xquat[start_body:end_body].copy()
        }

        # 1. 计算对手在自身坐标系中的位置和朝向
        self_quat = self.data.xquat[root_body]  # [w, x, y, z]
        self_rotation = R.from_quat([self_quat[1], self_quat[2], self_quat[3], self_quat[0]])  # xyzw

        # 对手位置（世界坐标系）
        opponent_pos_world = self.data.xpos[opponent_root].copy()
        self_pos_world = self.data.xpos[root_body]

        # 相对位置
        relative_pos = opponent_pos_world - self_pos_world
        opponent_pos_local = self_rotation.inv().apply(relative_pos)
        data['opponent_pos'] = opponent_pos_local

        # 对手朝向（前向量）
        opponent_quat = self.data.xquat[opponent_root]  # [w, x, y, z]
        opponent_rotation = R.from_quat([opponent_quat[1], opponent_quat[2], opponent_quat[3], opponent_quat[0]])
        # 对手的前向量在世界坐标系中是 [1, 0, 0]
        forward_world = opponent_rotation.apply([1, 0, 0])
        # 转换到自身坐标系
        opponent_facing = self_rotation.inv().apply(forward_world)
        data['opponent_facing'] = opponent_facing

        # 2. 检测脚与地面的接触和受力
        ground_geom = 0  # 地面 geom id
        foot_contact = {
            'right': False,
            'left': False
        }
        foot_force = {
            'right': 0.0,
            'left': 0.0,
            'contact_points': []  # 接触点位置列表
        }

        # 检查碰撞中的脚与地面接触
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            # 检查是否是脚与地面的碰撞
            is_foot = g1 in foot_geoms or g2 in foot_geoms
            is_ground = g1 == ground_geom or g2 == ground_geom

            if is_foot and is_ground:
                # 获取接触力
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                force = np.linalg.norm(c_array[:3])

                # 确定是哪只脚
                foot_geom = g1 if g1 in foot_geoms else g2

                if robot_id == 'robot_a':
                    if foot_geom in right_foot_geoms:
                        foot_contact['right'] = True
                        foot_force['right'] += force
                        foot_force['contact_points'].append({
                            'foot': 'right',
                            'position': contact.pos.copy(),
                            'force': force
                        })
                    else:  # left
                        foot_contact['left'] = True
                        foot_force['left'] += force
                        foot_force['contact_points'].append({
                            'foot': 'left',
                            'position': contact.pos.copy(),
                            'force': force
                        })
                else:  # robot_b
                    if foot_geom in right_foot_geoms:
                        foot_contact['right'] = True
                        foot_force['right'] += force
                        foot_force['contact_points'].append({
                            'foot': 'right',
                            'position': contact.pos.copy(),
                            'force': force
                        })
                    else:  # left
                        foot_contact['left'] = True
                        foot_force['left'] += force
                        foot_force['contact_points'].append({
                            'foot': 'left',
                            'position': contact.pos.copy(),
                            'force': force
                        })

        data['foot_contact'] = foot_contact
        data['foot_force'] = foot_force

        return data

    def get_sensor_data(self) -> Dict[str, Any]:
        return {'sensordata': self.data.sensordata.copy()}

    def get_action(self) -> Dict[str, Any]:
        return self._action

    def set_action(self, action: Dict[str, Any]) -> None:
        self._action = action
        
        # 将 action 转换为目标关节位置
        if not self._pd_initialized:
            return
            
        for r_id in ['robot_a', 'robot_b']:
            if r_id in action and action[r_id] is not None:
                residual_action = np.clip(np.asarray(action[r_id], dtype=np.float32), -1.0, 1.0)
                
                joint_limits = self.joint_limits[r_id]
                target_pos = self.reference_pos[r_id] + self.action_scale[r_id] * residual_action
                target_pos = np.clip(target_pos, joint_limits['lower'], joint_limits['upper'])
                
                self.target_positions[r_id] = target_pos.astype(np.float32)

    def _init_pd_limits(self) -> None:
        """
        初始化 PD 控制器的限制和参考位置

        控制方式：
        - Reference = (Down + Up) / 2  (关节限位的中间值)
        - Scale = (Up - Down) / 2      (关节范围的一半)
        - Target = Reference + Action * Scale
        - Action 范围: [-1, 1]
        """
        if self._pd_initialized:
            return

        self.joint_limits = {
            'robot_a': {'lower': np.zeros(self.action_dim), 'upper': np.zeros(self.action_dim)},
            'robot_b': {'lower': np.zeros(self.action_dim), 'upper': np.zeros(self.action_dim)}
        }
        self.ctrl_limits = {
            'robot_a': {'lower': np.zeros(self.action_dim), 'upper': np.zeros(self.action_dim)},
            'robot_b': {'lower': np.zeros(self.action_dim), 'upper': np.zeros(self.action_dim)}
        }
        self.action_scale = {
            'robot_a': np.zeros(self.action_dim),
            'robot_b': np.zeros(self.action_dim)
        }
        self.reference_pos = {
            'robot_a': np.zeros(self.action_dim),
            'robot_b': np.zeros(self.action_dim)
        }

        for r_id in ['robot_a', 'robot_b']:
            if r_id in self.robot_info:
                info = self.robot_info[r_id]

                if 'jnt_ranges' in info:
                    for i, r in enumerate(info['jnt_ranges']):
                        self.joint_limits[r_id]['lower'][i] = r[0]
                        self.joint_limits[r_id]['upper'][i] = r[1]

                if 'ctrl_ranges' in info:
                    for i, r in enumerate(info['ctrl_ranges']):
                        self.ctrl_limits[r_id]['lower'][i] = r[0]
                        self.ctrl_limits[r_id]['upper'][i] = r[1]

            # 计算 reference_pos 和 action_scale
            # Reference = (Down + Up) / 2 (关节中位)
            # Scale = (Up - Down) / 2 (关节范围的一半)
            lower = self.joint_limits[r_id]['lower']
            upper = self.joint_limits[r_id]['upper']

            # 计算参考位置和 scale
            for i in range(self.action_dim):
                lo = lower[i]
                hi = upper[i]

                if not np.isfinite(lo) or not np.isfinite(hi):
                    # 不支持无限范围
                    raise ValueError(
                        f"Joint {i} of {r_id} has infinite range "
                        f"(lower={lo}, upper={hi}). "
                        f"All joints must have finite limits for this control method."
                    )

                # 有限范围：使用中位和范围的一半
                self.reference_pos[r_id][i] = (lo + hi) / 2.0
                self.action_scale[r_id][i] = (hi - lo) / 2.0

            # 初始化目标位置为参考位置
            self.target_positions[r_id] = self.reference_pos[r_id].copy()

        self._pd_initialized = True

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        
        dist = self.initial_distance
        if options and 'initial_distance' in options:
            dist = float(options['initial_distance'])

        # 获取 qpos 的索引并不是连续的 0:7 和 7:14，必须通过缓存的 qpos_adr 获取
        qpos_adr_a = self.robot_info['robot_a']['qpos_adr']
        self.data.qpos[qpos_adr_a:qpos_adr_a+3] = [-dist / 2.0, 0.0, 1.282]
        self.data.qpos[qpos_adr_a+3:qpos_adr_a+7] = [1.0, 0.0, 0.0, 0.0]
        
        qpos_adr_b = self.robot_info['robot_b']['qpos_adr']
        self.data.qpos[qpos_adr_b:qpos_adr_b+3] = [dist / 2.0, 0.0, 1.282]
        self.data.qpos[qpos_adr_b+3:qpos_adr_b+7] = [0.0, 0.0, 0.0, 1.0]

        # 确保重置时速度被清零，避免残留动量
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        
        # 初始化 PD 控制器
        self._init_pd_limits()

    def physical_step(self) -> None:
        # 在物理步之前应用 PD 控制
        if self._pd_initialized:
            self._apply_pd_control()
        
        mujoco.mj_step(self.model, self.data)
    
    def _apply_pd_control(self) -> None:
        """计算并应用 PD 控制力矩"""
        for r_id in ['robot_a', 'robot_b']:
            target_pos = self.target_positions[r_id]

            current_pos = np.zeros(self.action_dim, dtype=np.float32)
            current_vel = np.zeros(self.action_dim, dtype=np.float32)

            if r_id in self.robot_info and 'qpos_indices' in self.robot_info[r_id]:
                qpos_idx_list = self.robot_info[r_id]['qpos_indices']
                qvel_idx_list = self.robot_info[r_id]['qvel_indices']

                for i in range(len(qpos_idx_list)):
                    if qpos_idx_list[i] >= 0:
                        current_pos[i] = self.data.qpos[qpos_idx_list[i]]
                        current_vel[i] = self.data.qvel[qvel_idx_list[i]]

            # 计算 PD 控制力矩
            torque_action = self.kp * (target_pos - current_pos) - self.kd * current_vel

            # 应用到执行器（考虑 gear 放大因子）
            if 'actuators' in self.robot_info[r_id]:
                act_indices = self.robot_info[r_id]['actuators']
                for i, act_idx in enumerate(act_indices):
                    if act_idx >= 0:
                        # 获取 gear 值用于限幅
                        gear = self.model.actuator_gear[act_idx, 0] if self.model.actuator_gear[act_idx, 0] != 0 else 1.0

                        # 计算 ctrl 值（ctrl * gear = desired_torque）
                        ctrl_value = torque_action[i] / gear if gear != 0 else torque_action[i]

                        # 根据 actuator 的 ctrlrange 进行限幅
                        ctrl_range = self.model.actuator_ctrlrange[act_idx]
                        ctrl_value = np.clip(ctrl_value, ctrl_range[0], ctrl_range[1])

                        self.data.ctrl[act_idx] = ctrl_value

    def get_physical_frequency(self) -> float:
        return 1.0 / self.dt

    def get_broadcastview_image(self) -> Any:
        try:
            pos_a = self.data.xpos[self.robot_info['robot_a']['body_id']]
            pos_b = self.data.xpos[self.robot_info['robot_b']['body_id']]
            center = (pos_a + pos_b) / 2.0
            
            target_lookat = center.copy()
            target_lookat[2] = 1.0  
            
            direction = pos_b - pos_a
            dist_ab = np.linalg.norm(direction)
            if dist_ab > 1e-6:
                direction = direction / dist_ab
            else:
                direction = np.array([1.0, 0.0, 0.0])

            dir_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            target_azi = dir_angle + 90.0
            target_ele = -20.0  
            target_dist = max(2.5, min(4.0, dist_ab * 1.5))
            
            limit = 2.55
            azi_rad = np.radians(target_azi)
            ele_rad = np.radians(target_ele)
            
            dx = -target_dist * np.cos(azi_rad) * np.cos(ele_rad)
            dy = -target_dist * np.sin(azi_rad) * np.cos(ele_rad)
            
            cam_x = target_lookat[0] + dx
            cam_y = target_lookat[1] + dy
            
            if abs(cam_x) > limit:
                max_dx = limit - target_lookat[0] if cam_x > 0 else -limit - target_lookat[0]
                factor = -np.cos(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dx / factor))
                    
            if abs(cam_y) > limit:
                max_dy = limit - target_lookat[1] if cam_y > 0 else -limit - target_lookat[1]
                factor = -np.sin(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dy / factor))

            alpha_pos = 0.05  
            alpha_look = 0.1  
            
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

            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.distance = dist
            cam.elevation = ele
            cam.azimuth = azi

            renderer = mujoco.Renderer(self.model, height=720, width=1280)
            renderer.update_scene(self.data, camera=cam)
            image = renderer.render()
            return image
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to render broadcast view: {e}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)
