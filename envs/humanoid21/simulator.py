import mujoco
import numpy as np
import os
from typing import Any, Dict, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseSimulator

class MujocoCombatSimulator(BaseSimulator):
    def __init__(self, arena_xml: str, dt: float = 0.002):
        self.dt = dt
        self.model = mujoco.MjSpec.from_file(arena_xml).compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        
        self.model.opt.timestep = dt
        
        self._action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        
        # 缓存关节点和索引
        self._cache_indices()
        
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
            for j in controlled_joints:
                act_name = f"{j}{suffix}"
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                actuators.append(act_id)
            self.robot_info[robot_id]['actuators'] = actuators

    def get_static_data(self) -> Dict[str, Any]:
        return {
            'dt': self.dt,
            'robot_info': self.robot_info
        }

    def get_core_state(self) -> Dict[str, Any]:
        state = {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'time': self.data.time,
            'robot_a': {},
            'robot_b': {}
        }
        for r_id in ['robot_a', 'robot_b']:
            qpos_adr = self.robot_info[r_id]['qpos_adr']
            qvel_adr = self.robot_info[r_id]['qvel_adr']
            # qpos: [x,y,z, qw,qx,qy,qz]
            state[r_id]['root_position'] = self.data.qpos[qpos_adr:qpos_adr+3].copy()
            state[r_id]['root_orientation'] = self.data.qpos[qpos_adr+3:qpos_adr+7].copy()
            state[r_id]['root_linear_velocity'] = self.data.qvel[qvel_adr:qvel_adr+3].copy()
            state[r_id]['root_angular_velocity'] = self.data.qvel[qvel_adr+3:qvel_adr+6].copy()
            
        return state

    def set_core_state(self, state: Dict[str, Any]) -> None:
        self.data.qpos[:] = state['qpos']
        self.data.qvel[:] = state['qvel']
        self.data.time = state.get('time', self.data.time)
        
        # 如果上层修改了 structured data，这里同步回 qpos/qvel
        for r_id in ['robot_a', 'robot_b']:
            if r_id in state:
                r_state = state[r_id]
                qpos_adr = self.robot_info[r_id]['qpos_adr']
                qvel_adr = self.robot_info[r_id]['qvel_adr']
                if 'root_position' in r_state:
                    self.data.qpos[qpos_adr:qpos_adr+3] = r_state['root_position']
                if 'root_orientation' in r_state:
                    self.data.qpos[qpos_adr+3:qpos_adr+7] = r_state['root_orientation']
                if 'root_linear_velocity' in r_state:
                    self.data.qvel[qvel_adr:qvel_adr+3] = r_state['root_linear_velocity']
                if 'root_angular_velocity' in r_state:
                    self.data.qvel[qvel_adr+3:qvel_adr+6] = r_state['root_angular_velocity']

        mujoco.mj_forward(self.model, self.data)

    def get_derived_state(self) -> Dict[str, Any]:
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            # Simple force calculation mapping
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            impulse = np.linalg.norm(c_array[:3]) # Approximate force/impulse

            contacts.append({
                'geom_a': geom1,
                'geom_b': geom2,
                'body_a': body1,
                'body_b': body2,
                'position': contact.pos.copy(),
                'normal': contact.frame[:3].copy(),
                'impulse': impulse
            })
            
        return {
            'contacts': contacts,
            'robot_a': {'xpos': self.data.xpos.copy(), 'xvelp': self.data.cvel[:, 3:].copy(), 'xquat': self.data.xquat.copy()}, # basic derived
            'robot_b': {}
        }

    def get_sensor_data(self) -> Dict[str, Any]:
        return {'sensordata': self.data.sensordata.copy()}

    def get_action(self) -> Dict[str, Any]:
        return self._action

    def set_action(self, action: Dict[str, Any]) -> None:
        self._action = action
        for r_id in ['robot_a', 'robot_b']:
            if r_id in action:
                acts = action[r_id]
                act_ids = self.robot_info[r_id]['actuators']
                for i, act_id in enumerate(act_ids):
                    if act_id >= 0:
                        ctrl_range = self.model.actuator_ctrlrange[act_id]
                        self.data.ctrl[act_id] = np.clip(acts[i], ctrl_range[0], ctrl_range[1])

    def physical_step(self) -> None:
        mujoco.mj_step(self.model, self.data)

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
