"""
Humanoid21 战斗仿真插件

包含以下功能插件：
1. NonFallConstraintPlugin - 防摔倒约束
2. CombatScoringPlugin - 战斗计分与KO判断（使用新的 combat_contacts 接口）
3. FrozenRobotPlugin - 冻结机器人

按照 DATASPEC.md 规范使用数据接口。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import BasePlugin, SimContext, TerminationReason

class NonFallConstraintPlugin(BasePlugin):
    """
    防摔倒约束插件

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
        core_state = ctx.accessor.get_core_state()
        static_data = ctx.accessor.get_static_data()
        robot_info = static_data.get('robot_info', {})

        changed = False

        for robot_id in ['robot_a', 'robot_b']:
            if robot_id not in robot_info:
                continue

            info = robot_info[robot_id]
            root_qpos_adr = info['root_qpos_adr']
            root_qvel_adr = info['root_qvel_adr']
            norm_params = static_data[robot_id]['norm_params']

            # 获取当前朝向 (四元数 [w,x,y,z])
            root_rot = core_state[robot_id]['root_rot']
            if np.linalg.norm(root_rot) < 1e-8:
                continue

            # 转换为欧拉角
            try:
                rot = R.from_quat([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])
                euler = rot.as_euler('xyz', degrees=True)
            except ValueError:
                continue

            roll, pitch, yaw = euler
            clamped_roll = float(np.clip(roll, -self.roll_limit_deg, self.roll_limit_deg))
            clamped_pitch = float(np.clip(pitch, -self.pitch_limit_deg, self.pitch_limit_deg))

            if not (np.isclose(roll, clamped_roll) and np.isclose(pitch, clamped_pitch)):
                # 需要拉回：构建新的状态
                new_state = {
                    robot_id: {
                        'root_pos': core_state[robot_id]['root_pos'].copy(),
                        'root_rot': None,  # 下面设置
                        'joint_pos_norm': core_state[robot_id]['joint_pos_norm'].copy(),
                        'joint_vel_norm': core_state[robot_id]['joint_vel_norm'].copy(),
                        'root_vel_local': core_state[robot_id]['root_vel_local'].copy(),
                        'root_angular_vel_local': core_state[robot_id]['root_angular_vel_local'].copy(),
                    }
                }

                # 计算新的四元数
                clamped_rotation = R.from_euler('xyz', [clamped_roll, clamped_pitch, yaw], degrees=True)
                clamped_xyzw = clamped_rotation.as_quat()
                new_state[robot_id]['root_rot'] = np.array([
                    clamped_xyzw[3], clamped_xyzw[0], clamped_xyzw[1], clamped_xyzw[2]
                ], dtype=np.float32)

                # 清零水平线性速度
                new_state[robot_id]['root_vel_local'] = np.zeros(3, dtype=np.float32)

                ctx.mutator.set_core_state(new_state)

                # 记录拉回次数
                ctx.metrics[f'{robot_id}_clamp_count'] = ctx.metrics.get(f'{robot_id}_clamp_count', 0) + 1
                changed = True


class CombatScoringPlugin(BasePlugin):
    """
    战斗计分与KO判断插件

    按照 DATASPEC.md 规范使用 combat_contacts 接口：
    - combat_contacts: List[Dict] - 双方实体之间的物理接触及受力列表
      格式: [{'body_a': 'head', 'body_b': 'torso', 'force': 150.0}, ...]
      规则: 仅记录双方机器人之间的碰撞，排除机器人与自身的接触
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
        # Per-episode HP carry-over flows through ``ctx.episode_options``
        # (see envs/framework/RESET.md §4). Constructor values are used as
        # defaults whenever the option is missing — this matches the
        # standalone "single round at full HP" use case.
        opts = ctx.episode_options
        ctx.metrics['health_a'] = float(
            opts.get('initial_health_a', self.initial_health_a)
        )
        ctx.metrics['health_b'] = float(
            opts.get('initial_health_b', self.initial_health_b)
        )
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
        # 使用新的 robot_robot_contacts 接口（按照 DATASPEC.md 规范）
        # 格式: [{'body_a': 'head', 'body_b': 'torso', 'force': 150.0}, ...]
        contacts = derived_state.get('robot_robot_contacts', [])

        for contact in contacts:
            # 新格式使用 body_a 和 body_b
            body_a_name = contact.get('body_a', '')
            body_b_name = contact.get('body_b', '')

            # 判断 body 属于哪个机器人
            is_body_a_a = body_a_name.endswith('_a') or '_red' in body_a_name
            is_body_a_b = body_a_name.endswith('_b') or '_blue' in body_a_name
            is_body_b_a = body_b_name.endswith('_a') or '_red' in body_b_name
            is_body_b_b = body_b_name.endswith('_b') or '_blue' in body_b_name

            if (is_body_a_a and is_body_b_b) or (is_body_a_b and is_body_b_a):
                cat1 = self._get_part_category(body_a_name)
                cat2 = self._get_part_category(body_b_name)

                force = contact.get('force', 0.0)

                attacker, defender, hit_part = None, None, None

                if is_body_a_a and is_body_b_b:
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_a', 'robot_b', cat2
                    elif cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_b', 'robot_a', cat1
                elif is_body_a_b and is_body_b_a:
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
                            damage = (weight * force) / self.damage_scale
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
    冻结机器人插件

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
        core_state = ctx.accessor.get_core_state()

        if self.frozen_robot_id not in core_state:
            return

        # 保存初始状态（按照新格式）
        self.initial_state = {
            'root_pos': core_state[self.frozen_robot_id]['root_pos'].copy(),
            'root_rot': core_state[self.frozen_robot_id]['root_rot'].copy(),
            'joint_pos_norm': core_state[self.frozen_robot_id]['joint_pos_norm'].copy(),
        }

        # 保存另一个机器人的ID
        self.other_robot_id = 'robot_b' if self.frozen_robot_id == 'robot_a' else 'robot_a'

    def on_post_phy_step(self, ctx: SimContext) -> None:
        """在每个物理步后强制重置机器人状态"""
        if self.initial_state is None:
            return

        core_state = ctx.accessor.get_core_state()

        # 构建冻结状态 - 需要包含两个机器人
        frozen_state = {
            self.frozen_robot_id: {
                'root_pos': self.initial_state['root_pos'].copy(),
                'root_rot': self.initial_state['root_rot'].copy(),
                'joint_pos_norm': self.initial_state['joint_pos_norm'].copy(),
                'joint_vel_norm': np.zeros(21, dtype=np.float32),
                'root_vel_local': np.zeros(3, dtype=np.float32),
                'root_angular_vel_local': np.zeros(3, dtype=np.float32),
            },
            self.other_robot_id: {
                'root_pos': core_state[self.other_robot_id]['root_pos'].copy(),
                'root_rot': core_state[self.other_robot_id]['root_rot'].copy(),
                'joint_pos_norm': core_state[self.other_robot_id]['joint_pos_norm'].copy(),
                'joint_vel_norm': core_state[self.other_robot_id]['joint_vel_norm'].copy(),
                'root_vel_local': core_state[self.other_robot_id]['root_vel_local'].copy(),
                'root_angular_vel_local': core_state[self.other_robot_id]['root_angular_vel_local'].copy(),
            }
        }

        ctx.mutator.set_core_state(frozen_state)
