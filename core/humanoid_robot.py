"""
MuJoCo Humanoid 机器人包装 (21 DOF)
"""

import numpy as np
import mujoco


class HumanoidRobot:
    """
    基于 MuJoCo 的 humanoid 模型，21 个可控 DOF。
    """

    ACTION_DIM = 21

    CONTROLLED_JOINTS = [
        'abdomen_z', 'abdomen_y', 'abdomen_x',
        'hip_x_right', 'hip_z_right', 'hip_y_right', 'knee_right', 'ankle_y_right', 'ankle_x_right',
        'hip_x_left', 'hip_z_left', 'hip_y_left', 'knee_left', 'ankle_y_left', 'ankle_x_left',
        'shoulder1_right', 'shoulder2_right', 'elbow_right',
        'shoulder1_left', 'shoulder2_left', 'elbow_left'
    ]

    # 初始姿态（直立，全零）
    INITIAL_POSITION = {joint: [0.0] for joint in CONTROLLED_JOINTS}

    def __init__(self, physics_engine, position, orientation, robot_id="robot", color=(0.8, 0.2, 0.2)):
        self.physics = physics_engine
        self.model = physics_engine.model
        self.data = physics_engine.data
        self.robot_id = robot_id
        self.color = color
        
        self.suffix = '_a' if self.robot_id == 'robot_a' else '_b'
        self.suffix = '_red' if self.robot_id == 'robot_a' else '_blue' # battle_v1.xmluses _red and _blue

        self._joint_indices = self._get_joint_indices()

    def _get_joint_indices(self):
        indices = {}
        for joint in self.CONTROLLED_JOINTS:
            full_name = f"{joint}{self.suffix}"
            try:
                idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
                if idx >= 0:
                    indices[joint] = idx
                else:
                    print(f"Warning: Joint {full_name} not found")
            except Exception as e:
                print(f"Warning: Exception getting joint {full_name}: {e}")
        return indices

    def set_joint_targets(self, action):
        action = np.clip(action, -1.0, 1.0)
        action_idx = 0
        for joint in self.CONTROLLED_JOINTS:
            if joint in self._joint_indices:
                joint_idx = self._joint_indices[joint]
                motor_name = f"{joint}{self.suffix}"
                motor_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
                if motor_idx >= 0:
                    ctrl_range = self.model.actuator_ctrlrange[motor_idx]
                    target = action[action_idx] * ctrl_range[1]
                    self.data.ctrl[motor_idx] = target
            action_idx += 1

    def get_position(self):
        body_name = f"pelvis{self.suffix}"
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if idx >= 0:
            return self.data.xpos[idx].copy()
        return np.zeros(3)

    def get_torso_state(self):
        body_name = f"pelvis{self.suffix}"
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        if idx >= 0:
            pos = self.data.xpos[idx].copy()
            quat = self.data.xquat[idx].copy()
            cvel = self.data.cvel[idx].copy()
            return {
                'position': pos,
                'orientation': quat,
                'linear_velocity': cvel[3:6], # mujoco cvel is [rot, lin]
                'angular_velocity': cvel[0:3]
            }
            
        return {
            'position': np.zeros(3),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0]),
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3)
        }

    def get_joint_states(self):
        pos = np.zeros(self.ACTION_DIM)
        vel = np.zeros(self.ACTION_DIM)
        
        for i, joint in enumerate(self.CONTROLLED_JOINTS):
            if joint in self._joint_indices:
                j_idx = self._joint_indices[joint]
                qpos_idx = self.model.jnt_qposadr[j_idx]
                qvel_idx = self.model.jnt_dofadr[j_idx]
                pos[i] = self.data.qpos[qpos_idx]
                vel[i] = self.data.qvel[qvel_idx]
                
        return {'positions': pos, 'velocities': vel}

    def get_feet_contact(self):
        # 通过检测脚部 geom 与 ground 的接触
        contact = {'left_foot': False, 'right_foot': False}
        floor_geom_ids = set()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and 'floor' in name.lower() or name == '地面':
                floor_geom_ids.add(i)
                
        left_foot_names = [f"foot_left{self.suffix}", f"foot1_left{self.suffix}", f"foot2_left{self.suffix}"]
        right_foot_names = [f"foot_right{self.suffix}", f"foot1_right{self.suffix}", f"foot2_right{self.suffix}"]
        
        left_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in left_foot_names}
        right_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in right_foot_names}
        
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if (g1 in floor_geom_ids and g2 in left_ids) or (g2 in floor_geom_ids and g1 in left_ids):
                contact['left_foot'] = True
            if (g1 in floor_geom_ids and g2 in right_ids) or (g2 in floor_geom_ids and g1 in right_ids):
                contact['right_foot'] = True
                
        return contact

    def get_external_forces(self):
        # 简化版：仅获取主要body的外部受力
        forces = np.zeros(6)
        
        torso_name = f"torso{self.suffix}"
        torso_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, torso_name)
        if torso_idx >= 0:
            forces[:3] = self.data.cfrc_ext[torso_idx, :3]
            
        head_name = f"head{self.suffix}"
        head_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, head_name)
        if head_idx >= 0:
            forces[:3] += self.data.cfrc_ext[head_idx, :3]
            
        left_hand_name = f"hand_left{self.suffix}"
        right_hand_name = f"hand_right{self.suffix}"
        lh_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, left_hand_name)
        rh_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, right_hand_name)
        if lh_idx >= 0: forces[3:6] += self.data.cfrc_ext[lh_idx, :3]
        if rh_idx >= 0: forces[3:6] += self.data.cfrc_ext[rh_idx, :3]
            
        return forces

    def get_keypoint_positions(self):
        parts = {
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
        
        positions = {}
        for key, base_name in parts.items():
            full_name = f"{base_name}{self.suffix}"
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_name)
            if idx >= 0:
                positions[key] = self.data.xpos[idx].copy()
            else:
                positions[key] = self.get_position()
        return positions

    def get_keypoint_velocities(self):
        parts = {
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
        
        velocities = {}
        for key, base_name in parts.items():
            full_name = f"{base_name}{self.suffix}"
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full_name)
            if idx >= 0:
                velocities[key] = self.data.cvel[idx, 3:6].copy() # lin vel
            else:
                velocities[key] = np.zeros(3)
        return velocities

    def get_observation(self, opponent_robot=None, opponent_health=None, self_health=None):
        obs_list = []
        
        # ========== Module 1: Proprioception (42 dims) ==========
        joint_states = self.get_joint_states()
        obs_list.append(joint_states['positions'])   # 21 dims
        obs_list.append(joint_states['velocities'])  # 21 dims

        # ========== Module 2: Root State (13 dims) ==========
        torso_state = self.get_torso_state()
        position = torso_state['position']
        orientation_quat = torso_state['orientation']  # [w, x, y, z]

        obs_list.append([position[2]])  # 1dims: Z axis height

        # 6dims: Local orientation
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])
        rot_matrix = rotation.as_matrix()
        local_orientation = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]])
        obs_list.append(local_orientation)

        # 6dims: Movement velocity
        obs_list.append(torso_state['linear_velocity'])
        obs_list.append(torso_state['angular_velocity'])

        # ========== Module 3: Tactile & Force Feedback (8 dims) ==========
        feet_contact = self.get_feet_contact()
        obs_list.append([float(feet_contact['left_foot']), float(feet_contact['right_foot'])]) # 2dims
        obs_list.append(self.get_external_forces())  # 6dims

        # ========== Module 4: Opponent Observation (64 dims) ==========
        if opponent_robot is not None:
            opponent_torso = opponent_robot.get_torso_state()

            # 10dims: Base pose
            obs_list.append(opponent_torso['position'] - torso_state['position'])
            obs_list.append(opponent_torso['linear_velocity'] - torso_state['linear_velocity'])
            obs_list.append(opponent_torso['orientation'])

            # 27dims: Keypoint positions
            opponent_keypoints = opponent_robot.get_keypoint_positions()
            keypoint_pos = []
            for key in ['head', 'right_hand', 'left_hand', 'right_elbow', 'left_elbow', 'right_knee', 'left_knee', 'right_foot', 'left_foot']:
                rel_pos = opponent_keypoints[key] - torso_state['position']
                local_pos = rot_matrix.T @ rel_pos
                keypoint_pos.append(local_pos)
            obs_list.append(np.concatenate(keypoint_pos))

            # 27dims: Keypoint velocities
            opponent_keyvels = opponent_robot.get_keypoint_velocities()
            keypoint_vel = []
            for key in ['head', 'right_hand', 'left_hand', 'right_elbow', 'left_elbow', 'right_knee', 'left_knee', 'right_foot', 'left_foot']:
                rel_vel = opponent_keyvels[key] - torso_state['linear_velocity']
                local_vel = rot_matrix.T @ rel_vel
                keypoint_vel.append(local_vel)
            obs_list.append(np.concatenate(keypoint_vel))
        else:
            obs_list.append(np.zeros(64))

        return np.concatenate(obs_list).astype(np.float32)
