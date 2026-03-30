import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List
import mujoco

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseRLAdapter, SimContext, TerminationReason

class Humanoid21RLAdapter(BaseRLAdapter):
    """
    针对 21DOF Humanoid 双人对战的 RL 数据适配器。
    负责定义动作/观测空间，以及从 SimContext 中提取和构建 RL 所需的 obs, reward, info。
    """
    ACTION_DIM = 21
    OBS_DIM = 127
    
    def __init__(self):
        super().__init__()

    def get_observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "robot_a_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32),
            "robot_b_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32),
        })

    def get_action_space(self) -> spaces.Dict:
        return spaces.Dict({
            "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(self.ACTION_DIM,), dtype=np.float32),
            "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(self.ACTION_DIM,), dtype=np.float32),
        })

    def build_observation(self, ctx: SimContext) -> Dict[str, np.ndarray]:
        sim = ctx.accessor # type: MujocoCombatSimulator
        
        # 获取静态数据和状态
        static_data = sim.get_static_data()
        robot_info = static_data['robot_info']
        core_state = sim.get_core_state()
        
        qpos_array = core_state['qpos']
        qvel_array = core_state['qvel']
        
        # 暂时直接访问底层数据用于构建详细观测
        # TODO: 未来可以扩展 get_derived_state 接口来提供这些数据
        data = sim.data
        model = sim.model

        obs_dict = {}
        for self_id, opponent_id in [('robot_a', 'robot_b'), ('robot_b', 'robot_a')]:
            obs_list = []
            
            # ========== Module 1: Proprioception (42 dims) ==========
            pos = np.zeros(self.ACTION_DIM)
            vel = np.zeros(self.ACTION_DIM)
            
            if self_id in robot_info and 'qpos_indices' in robot_info[self_id]:
                qpos_idx_list = robot_info[self_id]['qpos_indices']
                qvel_idx_list = robot_info[self_id]['qvel_indices']
                for i in range(len(qpos_idx_list)):
                    if qpos_idx_list[i] >= 0:
                        pos[i] = qpos_array[qpos_idx_list[i]]
                        vel[i] = qvel_array[qvel_idx_list[i]]
            
            obs_list.append(pos)
            obs_list.append(vel)

            # ========== Module 2: Root State (13 dims) ==========
            body_id = robot_info[self_id]['body_id']
            pos_root = data.xpos[body_id].copy()
            quat_root = data.xquat[body_id].copy() # [w, x, y, z]
            cvel_root = data.cvel[body_id].copy()  # [rot, lin]
            
            obs_list.append([pos_root[2]]) # 1 dim z height
            
            # 6dims local orientation (from rotation matrix)
            from scipy.spatial.transform import Rotation as R
            rot = R.from_quat([quat_root[1], quat_root[2], quat_root[3], quat_root[0]])
            rot_mat = rot.as_matrix()
            obs_list.append(np.concatenate([rot_mat[:, 0], rot_mat[:, 1]]))
            
            obs_list.append(cvel_root[3:6]) # lin vel
            obs_list.append(cvel_root[0:3]) # ang vel
            
            # ========== Module 3: Tactile & Force Feedback (8 dims) ==========
            # simplified feet contact
            left_foot_name = f"foot_left{robot_info[self_id]['suffix']}"
            right_foot_name = f"foot_right{robot_info[self_id]['suffix']}"
            
            left_contact = 0.0
            right_contact = 0.0
            # floor geom is assumed to be 0 or check name
            for idx in range(data.ncon):
                c = data.contact[idx]
                g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
                g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
                is_floor = 'floor' in g1_name.lower() or 'floor' in g2_name.lower() or g1_name == '地面' or g2_name == '地面'
                if not is_floor:
                    # check if geom is part of the default plane which might not have a name
                    if model.geom_type[c.geom1] == mujoco.mjtGeom.mjGEOM_PLANE or model.geom_type[c.geom2] == mujoco.mjtGeom.mjGEOM_PLANE:
                        is_floor = True
                        
                if is_floor:
                    b1 = model.geom_bodyid[c.geom1]
                    b2 = model.geom_bodyid[c.geom2]
                    b1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
                    b2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
                    if left_foot_name in b1_name or left_foot_name in b2_name:
                        left_contact = 1.0
                    if right_foot_name in b1_name or right_foot_name in b2_name:
                        right_contact = 1.0
            obs_list.append([left_contact, right_contact])
            
            # simplified external forces (6 dims)
            forces = np.zeros(6)
            for part, f_slice in [
                (f"pelvis{robot_info[self_id]['suffix']}", slice(0,3)),
                (f"head{robot_info[self_id]['suffix']}", slice(0,3)),
            ]:
                b_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
                if b_idx >= 0:
                    forces[f_slice] += data.cfrc_ext[b_idx, :3]
            for part in [f"hand_left{robot_info[self_id]['suffix']}", f"hand_right{robot_info[self_id]['suffix']}"]:
                b_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
                if b_idx >= 0:
                    forces[3:6] += data.cfrc_ext[b_idx, :3]
            obs_list.append(forces)
            
            # ========== Module 4: Opponent Observation (64 dims) ==========
            opp_body_id = robot_info[opponent_id]['body_id']
            opp_pos = data.xpos[opp_body_id].copy()
            opp_quat = data.xquat[opp_body_id].copy()
            opp_cvel = data.cvel[opp_body_id].copy()
            
            obs_list.append(opp_pos - pos_root)
            obs_list.append(opp_cvel[3:6] - cvel_root[3:6])
            obs_list.append(opp_quat)
            
            keypoint_parts = [
                'head', 'hand_right', 'hand_left', 'lower_arm_right', 'lower_arm_left',
                'shin_right', 'shin_left', 'foot_right', 'foot_left'
            ]
            
            kp_pos = []
            kp_vel = []
            for bp in keypoint_parts:
                b_name = f"{bp}{robot_info[opponent_id]['suffix']}"
                b_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b_name)
                if b_idx >= 0:
                    b_xpos = data.xpos[b_idx]
                    b_xvel = data.cvel[b_idx, 3:6] # linear vel
                else:
                    b_xpos = opp_pos
                    b_xvel = opp_cvel[3:6]
                
                rel_pos = b_xpos - pos_root
                local_pos = rot_mat.T @ rel_pos
                kp_pos.append(local_pos)
                
                rel_vel = b_xvel - cvel_root[3:6]
                local_vel = rot_mat.T @ rel_vel
                kp_vel.append(local_vel)
                
            obs_list.append(np.concatenate(kp_pos))
            obs_list.append(np.concatenate(kp_vel))
            
            final_obs = np.concatenate([np.asarray(o).flatten() for o in obs_list]).astype(np.float32)
            obs_dict[f"{self_id}_obs"] = final_obs

        return obs_dict

    def build_reward(self, ctx: SimContext) -> Dict[str, float]:
        # Reward function: 
        # based on damage taken (negative reward) and damage dealt (positive reward)
        # However, combat_gym.py originally returns 0.0 for both.
        return {'robot_a': 0.0, 'robot_b': 0.0}

    def build_info(self, ctx: SimContext) -> Dict[str, Any]:
        info = {
            'step': ctx.episode_step,
            'health': {
                'robot_a': ctx.metrics.get('health_a', 100.0),
                'robot_b': ctx.metrics.get('health_b', 100.0),
            },
            'damage_taken': {
                'robot_a': ctx.metrics.get('damage_taken_a', 0.0),
                'robot_b': ctx.metrics.get('damage_taken_b', 0.0),
            },
            'events': list(ctx.events)
        }
        
        # 判定 winner
        if ctx.is_terminated:
            proposals = ctx.termination_proposals
            winner = None
            if TerminationReason.KO in proposals:
                ha = ctx.metrics.get('health_a', 0)
                hb = ctx.metrics.get('health_b', 0)
                if ha <= 0 and hb > 0: winner = 'robot_b'
                elif hb <= 0 and ha > 0: winner = 'robot_a'
                else: winner = 'draw'
            elif TerminationReason.TIMEOUT in proposals:
                ha = ctx.metrics.get('health_a', 0)
                hb = ctx.metrics.get('health_b', 0)
                if ha > hb: winner = 'robot_a'
                elif hb > ha: winner = 'robot_b'
                else: winner = 'draw'
            
            info['winner'] = winner
            
        return info
