"""
T800 战斗仿真插件

参照 humanoid21/plugins.py 的 CombatScoringPlugin 实现，适配 T800 的 25 DOF、body name 命名规则（LINK_*、_red/_blue 后缀）和 DATASPEC.md 中的 robot_robot_contacts 接口。
"""

import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import BasePlugin, SimContext, TerminationReason


class T800CombatScoringPlugin(BasePlugin):
    """
    T800 版战斗计分与KO判断插件

    按照 DATASPEC.md 规范使用 `robot_robot_contacts` 接口：
    - robot_robot_contacts: List[Dict] - 格式 [{'body_a': 'LINK_HEAD_red', 'body_b': 'LINK_TORSO_blue', 'force': 150.0}, ...]
    - 仅记录双方机器人之间的碰撞
    - 伤害规则主要针对 head 和 torso
    """

    ATTACK_PARTS = {'hand', 'larm', 'uarm', 'shoulder', 'elbow', 'thigh', 'shin', 'foot', 'wrist'}
    DAMAGE_TARGET_PARTS = {'head', 'torso', 'waist', 'chest', 'pelvis'}

    DAMAGE_RULES = {
        'head': -3.0,
        'torso': -1.0,
        'waist': -1.0,
    }

    def __init__(
        self,
        initial_health: float = 100.0,
        initial_health_a: Optional[float] = None,
        initial_health_b: Optional[float] = None,
        damage_scale: float = 100.0
    ):
        # 支持分别设置双方初始血量
        self.initial_health_a = initial_health_a if initial_health_a is not None else initial_health
        self.initial_health_b = initial_health_b if initial_health_b is not None else initial_health
        self.damage_scale = damage_scale

    @property
    def name(self) -> str:
        return "t800_combat_scoring"

    def on_pre_episode(self, ctx: SimContext) -> None:
        """Per-episode HP 初始化，从 episode_options 读取（支持 carry-over）"""
        opts = ctx.episode_options
        ctx.metrics['health_a'] = float(
            opts.get('initial_health_a', self.initial_health_a)
        )
        ctx.metrics['health_b'] = float(
            opts.get('initial_health_b', self.initial_health_b)
        )
        ctx.metrics['damage_taken_a'] = 0.0
        ctx.metrics['damage_taken_b'] = 0.0
        # 清空 events
        while len(ctx.events) > 0:
            ctx.events.pop()

    def _get_part_category(self, geom_or_body_name: str) -> Optional[str]:
        """T800 专用 body/geom 名称分类，处理 LINK_ 前缀和颜色后缀"""
        if not geom_or_body_name:
            return None
        name_lower = geom_or_body_name.lower()

        # 移除常见前缀和后缀
        base_name = name_lower
        for prefix in ['link_', 'link']:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        for suffix in ['_red', '_blue', '_a', '_b', '_left', '_right']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        if 'head' in base_name:
            return 'head'
        if any(p in base_name for p in ['torso', 'waist', 'chest', 'pelvis', 'butt']):
            return 'torso'
        if any(p in base_name for p in ['hand', 'wrist', 'palm']):
            return 'hand'
        if any(p in base_name for p in ['lower_arm', 'elbow', 'forearm']):
            return 'larm'
        if any(p in base_name for p in ['upper_arm', 'shoulder']):
            return 'uarm'
        if 'thigh' in base_name or 'hip' in base_name:
            return 'thigh'
        if 'shin' in base_name or 'knee' in base_name:
            return 'shin'
        if 'foot' in base_name or 'ankle' in base_name:
            return 'foot'
        return None

    def on_post_action_step(self, ctx: SimContext) -> None:
        """使用 simulator 提供的 robot_robot_contacts 计算伤害和 KO"""
        derived_state = ctx.accessor.get_derived_state()
        contacts = derived_state.get('robot_robot_contacts', [])

        for contact in contacts:
            body_a_name = contact.get('body_a', '') or contact.get('body_a_name', '')
            body_b_name = contact.get('body_b', '') or contact.get('body_b_name', '')

            # 判断归属（兼容 _red/_blue 和 _a/_b）
            is_a_a = '_red' in body_a_name or body_a_name.endswith(('_a', '_red'))
            is_a_b = '_blue' in body_a_name or body_a_name.endswith(('_b', '_blue'))
            is_b_a = '_red' in body_b_name or body_b_name.endswith(('_a', '_red'))
            is_b_b = '_blue' in body_b_name or body_b_name.endswith(('_b', '_blue'))

            if (is_a_a and is_b_b) or (is_a_b and is_b_a):
                cat1 = self._get_part_category(body_a_name)
                cat2 = self._get_part_category(body_b_name)
                force = contact.get('force', contact.get('force_magnitude', 0.0))

                attacker = defender = hit_part = None

                if is_a_a and is_b_b:
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_a', 'robot_b', cat2
                    elif cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_b', 'robot_a', cat1
                else:  # is_a_b and is_b_a
                    if cat1 in self.ATTACK_PARTS and cat2 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_b', 'robot_a', cat2
                    elif cat2 in self.ATTACK_PARTS and cat1 in self.DAMAGE_TARGET_PARTS:
                        attacker, defender, hit_part = 'robot_a', 'robot_b', cat1

                if attacker and defender and hit_part:
                    # 映射到 DAMAGE_RULES 支持的 key
                    damage_part = 'head' if hit_part == 'head' else 'torso'

                    weight = -self.DAMAGE_RULES.get(damage_part, 0.0)
                    if weight > 0:
                        damage = (weight * force) / self.damage_scale
                        ctx.events.append({
                            'type': 'hit',
                            'attacker': attacker,
                            'defender': defender,
                            'part': damage_part,
                            'damage': damage,
                            'force': force,
                        })
                        health_key = 'health_a' if defender == 'robot_a' else 'health_b'
                        damage_key = 'damage_taken_a' if defender == 'robot_a' else 'damage_taken_b'
                        ctx.metrics[health_key] = max(0.0, ctx.metrics[health_key] - damage)
                        ctx.metrics[damage_key] += damage

        # KO 判断
        if ctx.metrics.get('health_a', 100.0) <= 0 or ctx.metrics.get('health_b', 100.0) <= 0:
            ctx.request_termination(TerminationReason.KO)


class FrozenRobotPlugin(BasePlugin):
    """
    冻结机器人插件（T800 适配版）

    在每个物理步后把指定机器人恢复到 episode 初始状态。
    """

    def __init__(self, frozen_robot_id: str = "robot_b"):
        self.frozen_robot_id = frozen_robot_id
        self.other_robot_id = "robot_a" if frozen_robot_id == "robot_b" else "robot_b"
        self.initial_state = None

    @property
    def name(self) -> str:
        return "frozen_robot"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_episode(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        if self.frozen_robot_id not in core_state:
            return

        rs = core_state[self.frozen_robot_id]
        self.initial_state = {
            "root_pos": rs["root_pos"].copy(),
            "root_quat_wxyz": rs["root_quat_wxyz"].copy(),
            "joint_pos": rs["joint_pos"].copy(),
        }

    def on_post_phy_step(self, ctx: SimContext) -> None:
        if self.initial_state is None:
            return

        core_state = ctx.accessor.get_core_state()
        other = core_state[self.other_robot_id]

        frozen_state = {
            self.frozen_robot_id: {
                "root_pos": self.initial_state["root_pos"].copy(),
                "root_quat_wxyz": self.initial_state["root_quat_wxyz"].copy(),
                "joint_pos": self.initial_state["joint_pos"].copy(),
                "joint_vel": np.zeros(25, dtype=np.float32),
                "root_vel": np.zeros(3, dtype=np.float32),
                "root_ang_vel": np.zeros(3, dtype=np.float32),
            },
            self.other_robot_id: {
                "root_pos": other["root_pos"].copy(),
                "root_quat_wxyz": other["root_quat_wxyz"].copy(),
                "joint_pos": other["joint_pos"].copy(),
                "joint_vel": other["joint_vel"].copy(),
                "root_vel": other["root_vel"].copy(),
                "root_ang_vel": other["root_ang_vel"].copy(),
            },
        }
        ctx.mutator.set_core_state(frozen_state)


__all__ = ["T800CombatScoringPlugin", "FrozenRobotPlugin"]
