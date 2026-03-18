"""
碰撞检测模块 - 21DOF 版本

负责检测机器人之间的碰撞和身体部位击中判定。

V1.0 规则：
- 攻击部位：手、前臂、肘、上臂、脚、小腿、膝、大腿
- 受击部位：头部、躯干（只有这两个部位受击才掉血）
- 物理条件：相对速度 > 1.0 m/s（过滤慢速接触）
"""

import numpy as np
import mujoco

class CollisionDetector:
    """
    碰撞检测器
    """

    # 攻击部位
    ATTACK_PARTS = {
        'hand', 'larm', 'uarm', 'thigh', 'shin', 'foot'
    }

    # 受击部位
    DAMAGE_TARGET_PARTS = {
        'head', 'torso', 'waist_upper', 'waist_lower', 'pelvis', 'butt'
    }

    def __init__(self, velocity_threshold=1.0):
        self.velocity_threshold = velocity_threshold

    def get_part_category(self, geom_name):
        if not geom_name:
            return None
        
        name_lower = geom_name.lower()
        
        # 移除 _red / _blue 等后缀，获取基础名称
        base_name = name_lower
        for suffix in ['_red', '_blue', '_a', '_b']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        if 'head' in base_name:
            return 'head'
        elif any(part in base_name for part in ['torso', 'waist', 'pelvis', 'butt']):
            return 'torso'
        elif 'hand' in base_name:
            return 'hand'
        elif 'lower_arm' in base_name:
            return 'larm'
        elif 'upper_arm' in base_name:
            return 'uarm'
        elif 'thigh' in base_name:
            return 'thigh'
        elif 'shin' in base_name:
            return 'shin'
        elif 'foot' in base_name:
            return 'foot'
        
        return None

    def get_damage_part(self, part_category):
        if part_category == 'head':
            return 'head'
        elif part_category == 'torso':
            return 'torso'
        return None

    def check_collisions(self, robot_a, robot_b, physics):
        collisions = []
        data = physics.data
        model = physics.model

        # 遍历所有接触点
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # 获取接触的两个 geom
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            if not geom1_name or not geom2_name:
                continue

            robot_a_suffix = '_a' if robot_a.robot_id == 'robot_a' else '_red'
            robot_b_suffix = '_b' if robot_b.robot_id == 'robot_b' else '_blue'

            # 判定属于哪个机器人
            is_geom1_a = geom1_name.endswith(robot_a_suffix)
            is_geom1_b = geom1_name.endswith(robot_b_suffix)
            is_geom2_a = geom2_name.endswith(robot_a_suffix)
            is_geom2_b = geom2_name.endswith(robot_b_suffix)

            # 只关心两个机器人之间的碰撞
            if (is_geom1_a and is_geom2_b) or (is_geom1_b and is_geom2_a):
                cat1 = self.get_part_category(geom1_name)
                cat2 = self.get_part_category(geom2_name)

                # 获取接触点相对速度
                cvel = np.zeros(6)
                mujoco.mj_contactVelocity(model, data, i, cvel)
                rel_speed = np.linalg.norm(cvel[:3]) # 平移相对速度

                if rel_speed < self.velocity_threshold:
                    continue

                # 判定谁打谁
                if is_geom1_a and is_geom2_b:
                    # A 碰 B
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        collisions.append({'attacker': 'robot_a', 'defender': 'robot_b', 'hit_part': cat2, 'velocity': rel_speed})
                    if cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        collisions.append({'attacker': 'robot_b', 'defender': 'robot_a', 'hit_part': cat1, 'velocity': rel_speed})
                elif is_geom1_b and is_geom2_a:
                    # B 碰 A
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        collisions.append({'attacker': 'robot_b', 'defender': 'robot_a', 'hit_part': cat2, 'velocity': rel_speed})
                    if cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        collisions.append({'attacker': 'robot_a', 'defender': 'robot_b', 'hit_part': cat1, 'velocity': rel_speed})

        # 去重（同一次物理step中，同一个部位可能产生多个接触点）
        unique_collisions = []
        seen = set()
        for c in collisions:
            key = (c['attacker'], c['defender'], c['hit_part'])
            if key not in seen:
                seen.add(key)
                unique_collisions.append(c)

        return unique_collisions
