"""
Collision Detection Module - 21DOF Version

Responsible for detecting collisions between robots and body part hit judgment.

V1.0 Rules:
- Attacking parts: hand, forearm, elbow, upper arm, foot, shin, knee, thigh
- Target parts: head, torso (only these parts deduct HP when hit)
- Physical condition: relative velocity > 1.0 m/s (filters slow contact)
"""

import numpy as np
import mujoco

class CollisionDetector:
    """
    Collision Detector
    """

    # Attacking parts
    ATTACK_PARTS = {
        'hand', 'larm', 'uarm', 'thigh', 'shin', 'foot'
    }

    # Target parts
    DAMAGE_TARGET_PARTS = {
        'head', 'torso', 'waist_upper', 'waist_lower', 'pelvis', 'butt'
    }

    def __init__(self, velocity_threshold=1.0):
        self.velocity_threshold = velocity_threshold

    def get_part_category(self, geom_name):
        if not geom_name:
            return None
        
        name_lower = geom_name.lower()
        
        # Remove suffixes like _red / _blue to get base name
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

        robot_a_suffixes = tuple(
            suffix
            for suffix in dict.fromkeys([
                getattr(robot_a, 'suffix', None),
                '_a' if robot_a.robot_id == 'robot_a' else '_b',
                '_red' if robot_a.robot_id == 'robot_a' else '_blue',
            ])
            if suffix
        )
        robot_b_suffixes = tuple(
            suffix
            for suffix in dict.fromkeys([
                getattr(robot_b, 'suffix', None),
                '_a' if robot_b.robot_id == 'robot_a' else '_b',
                '_red' if robot_b.robot_id == 'robot_a' else '_blue',
            ])
            if suffix
        )

        # Iterate through all contact points
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # Get the two contacted geoms
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            if not geom1_name or not geom2_name:
                continue

            # Determine which robot it belongs to
            is_geom1_a = geom1_name.endswith(robot_a_suffixes)
            is_geom1_b = geom1_name.endswith(robot_b_suffixes)
            is_geom2_a = geom2_name.endswith(robot_a_suffixes)
            is_geom2_b = geom2_name.endswith(robot_b_suffixes)

            # Only care about collisions between the two robots
            if (is_geom1_a and is_geom2_b) or (is_geom1_b and is_geom2_a):
                cat1 = self.get_part_category(geom1_name)
                cat2 = self.get_part_category(geom2_name)

                # Get relative velocity at contact point
                body1_id = model.geom_bodyid[geom1_id]
                body2_id = model.geom_bodyid[geom2_id]
                vel1 = data.cvel[body1_id, 3:6] if body1_id >= 0 else np.zeros(3)
                vel2 = data.cvel[body2_id, 3:6] if body2_id >= 0 else np.zeros(3)
                rel_speed = np.linalg.norm(vel1 - vel2)

                if rel_speed < self.velocity_threshold:
                    continue

                # Determine who hit whom
                if is_geom1_a and is_geom2_b:
                    # A hits B
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        collisions.append({'attacker': 'robot_a', 'defender': 'robot_b', 'hit_part': cat2, 'velocity': rel_speed})
                    if cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        collisions.append({'attacker': 'robot_b', 'defender': 'robot_a', 'hit_part': cat1, 'velocity': rel_speed})
                elif is_geom1_b and is_geom2_a:
                    # B hits A
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        collisions.append({'attacker': 'robot_b', 'defender': 'robot_a', 'hit_part': cat2, 'velocity': rel_speed})
                    if cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        collisions.append({'attacker': 'robot_a', 'defender': 'robot_b', 'hit_part': cat1, 'velocity': rel_speed})

        # Deduplicate (multiple contact points may occur on the same part in one physics step)
        unique_collisions = []
        seen = set()
        for c in collisions:
            key = (c['attacker'], c['defender'], c['hit_part'])
            if key not in seen:
                seen.add(key)
                unique_collisions.append(c)

        return unique_collisions
