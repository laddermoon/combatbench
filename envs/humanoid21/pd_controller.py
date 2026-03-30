import numpy as np
from typing import Any, Dict
import mujoco

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BasePlugin, SimContext

class PDControllerPlugin(BasePlugin):
    """
    基于目标位置的 PD 控制器插件。
    在控制步开始前（on_pre_action_step），将输入的标准化 action（[-1, 1]）映射为目标关节位置，
    在每次物理步前（on_pre_phy_step），根据当前关节位置和速度计算力矩并应用到底层。
    """
    def __init__(self, action_dim: int = 21, kp: float = 4.0, kd: float = 0.4):
        self.action_dim = action_dim
        self.kp = np.full(action_dim, kp, dtype=np.float32)
        self.kd = np.full(action_dim, kd, dtype=np.float32)
        
        # 缓存目标位置
        self.target_positions = {
            'robot_a': np.zeros(action_dim, dtype=np.float32),
            'robot_b': np.zeros(action_dim, dtype=np.float32)
        }
        
        self._is_initialized = False

    @property
    def name(self) -> str:
        return "pd_controller"

    @property
    def require_mutator(self) -> bool:
        return True

    def _init_limits(self, sim: Any) -> None:
        if self._is_initialized:
            return
            
        self.joint_limits = {'robot_a': {'lower': np.zeros(self.action_dim), 'upper': np.zeros(self.action_dim)},
                             'robot_b': {'lower': np.zeros(self.action_dim), 'upper': np.zeros(self.action_dim)}}
        self.ctrl_limits = {'robot_a': {'lower': np.zeros(self.action_dim), 'upper': np.zeros(self.action_dim)},
                            'robot_b': {'lower': np.zeros(self.action_dim), 'upper': np.zeros(self.action_dim)}}
        self.action_scale = {'robot_a': np.zeros(self.action_dim), 'robot_b': np.zeros(self.action_dim)}
        self.reference_pos = {'robot_a': np.zeros(self.action_dim), 'robot_b': np.zeros(self.action_dim)}
        
        controlled_joints = [
            'abdomen_z', 'abdomen_y', 'abdomen_x',
            'hip_x_right', 'hip_z_right', 'hip_y_right', 'knee_right', 'ankle_y_right', 'ankle_x_right',
            'hip_x_left', 'hip_z_left', 'hip_y_left', 'knee_left', 'ankle_y_left', 'ankle_x_left',
            'shoulder1_right', 'shoulder2_right', 'elbow_right',
            'shoulder1_left', 'shoulder2_left', 'elbow_left'
        ]

        model = sim.model
        for r_id in ['robot_a', 'robot_b']:
            suffix = sim.robot_info[r_id]['suffix']
            
            for i, jname in enumerate(controlled_joints):
                full_name = f"{jname}{suffix}"
                j_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
                act_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
                
                if j_idx >= 0 and act_idx >= 0:
                    if model.jnt_limited[j_idx]:
                        lower, upper = model.jnt_range[j_idx]
                    else:
                        lower, upper = -np.pi, np.pi
                        
                    self.joint_limits[r_id]['lower'][i] = lower
                    self.joint_limits[r_id]['upper'][i] = upper
                    
                    c_lower, c_upper = model.actuator_ctrlrange[act_idx]
                    self.ctrl_limits[r_id]['lower'][i] = c_lower
                    self.ctrl_limits[r_id]['upper'][i] = c_upper
            
            # Default action scale from joint limits
            lower = self.joint_limits[r_id]['lower']
            upper = self.joint_limits[r_id]['upper']
            default_scale = np.full(self.action_dim, 0.25, dtype=np.float32)
            finite_mask = np.isfinite(lower) & np.isfinite(upper)
            default_scale[finite_mask] = 0.25 * (upper[finite_mask] - lower[finite_mask])
            self.action_scale[r_id] = np.maximum(default_scale, 1e-3).astype(np.float32)

        self._is_initialized = True

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._init_limits(ctx.accessor)
        self.target_positions = {
            'robot_a': np.zeros(self.action_dim, dtype=np.float32),
            'robot_b': np.zeros(self.action_dim, dtype=np.float32)
        }

    def on_pre_action_step(self, ctx: SimContext) -> None:
        action_dict = ctx.accessor.get_action()
        for r_id in ['robot_a', 'robot_b']:
            if r_id in action_dict and action_dict[r_id] is not None:
                residual_action = np.clip(np.asarray(action_dict[r_id], dtype=np.float32), -1.0, 1.0)
                
                joint_limits = self.joint_limits[r_id]
                target_pos = self.reference_pos[r_id] + self.action_scale[r_id] * residual_action
                target_pos = np.clip(target_pos, joint_limits['lower'], joint_limits['upper'])
                
                self.target_positions[r_id] = target_pos.astype(np.float32)

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        sim = ctx.accessor
        model = sim.model
        data = sim.data
        
        controlled_joints = [
            'abdomen_z', 'abdomen_y', 'abdomen_x',
            'hip_x_right', 'hip_z_right', 'hip_y_right', 'knee_right', 'ankle_y_right', 'ankle_x_right',
            'hip_x_left', 'hip_z_left', 'hip_y_left', 'knee_left', 'ankle_y_left', 'ankle_x_left',
            'shoulder1_right', 'shoulder2_right', 'elbow_right',
            'shoulder1_left', 'shoulder2_left', 'elbow_left'
        ]

        # Calculate and apply torque
        for r_id in ['robot_a', 'robot_b']:
            suffix = sim.robot_info[r_id]['suffix']
            target_pos = self.target_positions[r_id]
            
            current_pos = np.zeros(self.action_dim, dtype=np.float32)
            current_vel = np.zeros(self.action_dim, dtype=np.float32)
            
            act_indices = []
            
            for i, jname in enumerate(controlled_joints):
                full_name = f"{jname}{suffix}"
                j_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
                act_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
                
                act_indices.append(act_idx)
                if j_idx >= 0:
                    qpos_idx = model.jnt_qposadr[j_idx]
                    qvel_idx = model.jnt_dofadr[j_idx]
                    current_pos[i] = data.qpos[qpos_idx]
                    current_vel[i] = data.qvel[qvel_idx]
                    
            torque_action = self.kp * (target_pos - current_pos) - self.kd * current_vel
            
            c_lower = self.ctrl_limits[r_id]['lower']
            c_upper = self.ctrl_limits[r_id]['upper']
            torque_action = np.clip(torque_action, c_lower, c_upper)
            
            # Apply to data.ctrl
            for i, act_idx in enumerate(act_indices):
                if act_idx >= 0:
                    data.ctrl[act_idx] = torque_action[i]
