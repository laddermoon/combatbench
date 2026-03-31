import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import BasePlugin, SimContext, TerminationReason

class NonFallConstraintPlugin(BasePlugin):
    """
    防摔倒约束插件。
    在物理步之后，如果机器人的 pitch 或 roll 超过阈值，
    会强行将其拉回阈值范围内，并重置角速度。
    """
    def __init__(self, pitch_limit_deg: float = 5.0, roll_limit_deg: float = 5.0):
        self.pitch_limit_deg = pitch_limit_deg
        self.roll_limit_deg = roll_limit_deg

    @property
    def name(self) -> str:
        return "non_fall_constraint"

    @property
    def require_mutator(self) -> bool:
        return True  # 声明写入权限

    def on_post_phy_step(self, ctx: SimContext) -> None:
        state = ctx.accessor.get_core_state()
        changed = False

        for robot_id in ['robot_a', 'robot_b']:
            orientation_wxyz = state[robot_id]['root_orientation']
            if np.linalg.norm(orientation_wxyz) < 1e-8:
                continue

            orientation_xyzw = np.array([
                orientation_wxyz[1], orientation_wxyz[2], 
                orientation_wxyz[3], orientation_wxyz[0]
            ])
            
            try:
                rot = R.from_quat(orientation_xyzw)
                euler = rot.as_euler('xyz', degrees=True)
            except ValueError:
                continue

            roll, pitch, yaw = euler
            clamped_roll = float(np.clip(roll, -self.roll_limit_deg, self.roll_limit_deg))
            clamped_pitch = float(np.clip(pitch, -self.pitch_limit_deg, self.pitch_limit_deg))

            if not (np.isclose(roll, clamped_roll) and np.isclose(pitch, clamped_pitch)):
                clamped_rotation = R.from_euler('xyz', [clamped_roll, clamped_pitch, yaw], degrees=True)
                clamped_xyzw = clamped_rotation.as_quat()
                clamped_wxyz = np.array([
                    clamped_xyzw[3], clamped_xyzw[0], 
                    clamped_xyzw[1], clamped_xyzw[2]
                ], dtype=np.float64)

                state[robot_id]['root_orientation'] = clamped_wxyz
                # 清零水平线性速度（x, y），而不是角速度
                state[robot_id]['root_linear_velocity'][:2] = 0.0
                
                # 记录拉回次数
                ctx.metrics[f'{robot_id}_clamp_count'] = ctx.metrics.get(f'{robot_id}_clamp_count', 0) + 1
                changed = True

        if changed:
            ctx.mutator.set_core_state(state)


class CombatScoringPlugin(BasePlugin):
    """
    战斗计分与KO判断插件。
    在每个控制步结束时，通过检查物理引擎的碰撞点来计算伤害。
    """
    ATTACK_PARTS = {'hand', 'larm', 'uarm', 'thigh', 'shin', 'foot'}
    DAMAGE_TARGET_PARTS = {'head', 'torso', 'waist_upper', 'waist_lower'}
    
    DAMAGE_RULES = {
        'head': -3.0,
        'torso': -1.0,
    }

    def __init__(
        self,
        initial_health: float = 100.0,
        initial_health_a: Optional[float] = None,
        initial_health_b: Optional[float] = None,
        damage_scale: float = 100.0
    ):
        # 支持分别设置双方初始血量，如果只设置 initial_health 则双方相同
        self.initial_health_a = initial_health_a if initial_health_a is not None else initial_health
        self.initial_health_b = initial_health_b if initial_health_b is not None else initial_health
        self.damage_scale = damage_scale

    @property
    def name(self) -> str:
        return "combat_scoring"

    def on_pre_episode(self, ctx: SimContext) -> None:
        ctx.metrics['health_a'] = self.initial_health_a
        ctx.metrics['health_b'] = self.initial_health_b
        ctx.metrics['damage_taken_a'] = 0.0
        ctx.metrics['damage_taken_b'] = 0.0
        # reset events list explicitly
        while len(ctx.events) > 0:
            ctx.events.pop()

    def _get_part_category(self, geom_name: str) -> str:
        if not geom_name: return None
        name_lower = geom_name.lower()
        
        base_name = name_lower
        for suffix in ['_red', '_blue', '_a', '_b']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        if 'head' in base_name: return 'head'
        if any(p in base_name for p in ['torso', 'waist', 'pelvis', 'butt']): return 'torso'
        if 'hand' in base_name: return 'hand'
        if 'lower_arm' in base_name: return 'larm'
        if 'upper_arm' in base_name: return 'uarm'
        if 'thigh' in base_name: return 'thigh'
        if 'shin' in base_name: return 'shin'
        if 'foot' in base_name: return 'foot'
        return None

    def on_post_action_step(self, ctx: SimContext) -> None:
        derived_state = ctx.accessor.get_derived_state()
        contacts = derived_state.get('contacts', [])
        
        for contact in contacts:
            geom1_name = contact.get('geom1_name', '')
            geom2_name = contact.get('geom2_name', '')
            
            is_geom1_a = geom1_name.endswith('_red')
            is_geom1_b = geom1_name.endswith('_blue')
            is_geom2_a = geom2_name.endswith('_red')
            is_geom2_b = geom2_name.endswith('_blue')

            if (is_geom1_a and is_geom2_b) or (is_geom1_b and is_geom2_a):
                cat1 = self._get_part_category(geom1_name)
                cat2 = self._get_part_category(geom2_name)
                
                impulse = contact.get('impulse', 0.0)

                attacker, defender, hit_part = None, None, None

                if is_geom1_a and is_geom2_b:
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_a', 'robot_b', cat2
                    elif cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_b', 'robot_a', cat1
                elif is_geom1_b and is_geom2_a:
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_b', 'robot_a', cat2
                    elif cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_a', 'robot_b', cat1
                        
                if attacker and defender and hit_part:
                    # Target part mapping
                    if hit_part in ['waist_upper', 'waist_lower', 'torso']:
                        damage_part = 'torso'
                    elif hit_part == 'head':
                        damage_part = 'head'
                    else:
                        damage_part = None

                    if damage_part:
                        weight = -self.DAMAGE_RULES.get(damage_part, 0.0)
                        if weight > 0:
                            damage = (weight * impulse) / self.damage_scale
                            # Record events and metrics
                            ctx.events.append({
                                'type': 'hit',
                                'attacker': attacker,
                                'defender': defender,
                                'part': damage_part,
                                'damage': damage
                            })
                            health_key = 'health_a' if defender == 'robot_a' else 'health_b'
                            damage_key = 'damage_taken_a' if defender == 'robot_a' else 'damage_taken_b'
                            ctx.metrics[health_key] = max(0.0, ctx.metrics[health_key] - damage)
                            ctx.metrics[damage_key] += damage

        # Check KO
        if ctx.metrics['health_a'] <= 0 or ctx.metrics['health_b'] <= 0:
            ctx.request_termination(TerminationReason.KO)


class FrozenRobotPlugin(BasePlugin):
    """
    冻结机器人插件。
    让指定的机器人保持初始姿态完全静止，像雕塑一样。
    在每个物理步后强制重置该机器人的位置、姿态和速度到初始状态。
    """
    def __init__(self, frozen_robot_id: str = 'robot_b'):
        """
        Args:
            frozen_robot_id: 要冻结的机器人ID，默认为 'robot_b'
        """
        self.frozen_robot_id = frozen_robot_id
        self.initial_state = None
        
    @property
    def name(self) -> str:
        return "frozen_robot"
    
    @property
    def require_mutator(self) -> bool:
        return True
    
    def on_pre_episode(self, ctx: SimContext) -> None:
        """在episode开始时记录初始状态"""
        state = ctx.accessor.get_core_state()
        if self.frozen_robot_id in state:
            # 保存初始状态
            self.initial_state = {
                'root_position': state[self.frozen_robot_id]['root_position'].copy(),
                'root_orientation': state[self.frozen_robot_id]['root_orientation'].copy(),
                'root_linear_velocity': np.zeros(3, dtype=np.float64),
                'root_angular_velocity': np.zeros(3, dtype=np.float64),
            }
            
            # 同时保存关节状态（如果有的话）
            static_data = ctx.accessor.get_static_data()
            robot_info = static_data.get('robot_info', {})
            if self.frozen_robot_id in robot_info:
                info = robot_info[self.frozen_robot_id]
                if 'qpos_indices' in info:
                    qpos_indices = info['qpos_indices']
                    qvel_indices = info['qvel_indices']
                    
                    full_state = ctx.accessor.get_core_state()
                    qpos_array = full_state['qpos']
                    
                    # 保存关节位置
                    joint_positions = np.zeros(len(qpos_indices), dtype=np.float64)
                    for i, idx in enumerate(qpos_indices):
                        if idx >= 0:
                            joint_positions[i] = qpos_array[idx]
                    
                    self.initial_state['joint_positions'] = joint_positions
                    self.initial_state['qpos_indices'] = qpos_indices
                    self.initial_state['qvel_indices'] = qvel_indices
    
    def on_post_phy_step(self, ctx: SimContext) -> None:
        """在每个物理步后强制重置机器人状态"""
        if self.initial_state is None:
            return
        
        state = ctx.accessor.get_core_state()
        
        # 重置根部状态
        state[self.frozen_robot_id]['root_position'] = self.initial_state['root_position'].copy()
        state[self.frozen_robot_id]['root_orientation'] = self.initial_state['root_orientation'].copy()
        state[self.frozen_robot_id]['root_linear_velocity'] = np.zeros(3, dtype=np.float64)
        state[self.frozen_robot_id]['root_angular_velocity'] = np.zeros(3, dtype=np.float64)
        
        # 重置关节状态
        if 'joint_positions' in self.initial_state:
            qpos_indices = self.initial_state['qpos_indices']
            qvel_indices = self.initial_state['qvel_indices']
            joint_positions = self.initial_state['joint_positions']
            
            qpos_array = state['qpos']
            qvel_array = state['qvel']
            
            for i, (qpos_idx, qvel_idx) in enumerate(zip(qpos_indices, qvel_indices)):
                if qpos_idx >= 0:
                    qpos_array[qpos_idx] = joint_positions[i]
                if qvel_idx >= 0:
                    qvel_array[qvel_idx] = 0.0
            
            state['qpos'] = qpos_array
            state['qvel'] = qvel_array
        
        # 应用状态
        ctx.mutator.set_core_state(state)
