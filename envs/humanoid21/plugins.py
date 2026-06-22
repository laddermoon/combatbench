"""
Humanoid21 战斗仿真插件

包含以下功能插件：
1. NonFallConstraintPlugin - 防摔倒约束
2. CombatScoringPlugin - 战斗计分与KO判断（使用新的 combat_contacts 接口）
3. FrozenRobotPlugin - 冻结机器人

按照 DATASPEC.md 规范使用数据接口。
"""

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from envs.framework import BasePlugin, OBSERVER_DISPATCHER_PRIORITY
from envs.framework.context import SimContext, TerminationReason

class NonFallConstraintPlugin(BasePlugin):
    """
    防摔倒约束插件

    在物理步之后，如果机器人的 pitch 或 roll 超过阈值，
    会强行将其拉回阈值范围内，并重置角速度。
    """
    def __init__(self, pitch_limit_deg: float = 5.0, roll_limit_deg: float = 5.0):
        self.pitch_limit_deg = pitch_limit_deg
        self.roll_limit_deg = roll_limit_deg

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "pitch_limit_deg": self.pitch_limit_deg,
            "roll_limit_deg": self.roll_limit_deg,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "NonFallConstraintPlugin":
        return cls(**config)

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
    """Combat scoring and KO determination plugin.

    Damage is computed **per physics substep** (on_post_phy_step), not per
    action step, so transient contacts that appear and disappear within the
    50 ms action window are never missed.

    Damage formula (per substep)::

        damage = part_weight × (force / force_scale)² × dt

    where ``dt`` is the physics timestep in seconds (default 0.002 s).

    The ``(force / force_scale)²`` term creates a quadratic threshold:
    forces below ``force_scale`` are suppressed, forces above it are amplified.
    For example, with ``force_scale = 100 N``:

        50 N  → (0.5)² = 0.25   (suppressed)
        100 N → (1.0)² = 1.0    (threshold)
        200 N → (2.0)² = 4.0    (amplified)
        400 N → (4.0)² = 16.0   (heavily amplified)

    Options consumed from ``ctx.episode_options``:
      - ``initial_health_a`` (float): Starting HP for robot_a.
      - ``initial_health_b`` (float): Starting HP for robot_b.

    Constructor-only option:
      - ``score_log_file`` (str | None): If set, append one concise line per
        physics substep to this file for audit/review. ``None`` (default)
        disables logging. Deliberately a constructor parameter rather than
        an environment variable so concurrent processes (e.g. parallel
        matches) can log to separate files without collision.
    """
    ATTACK_PARTS = {'hand', 'larm', 'uarm', 'thigh', 'shin', 'foot'}
    DAMAGE_TARGET_PARTS = {'head', 'torso', 'waist_upper', 'waist_lower'}

    # Part weight × (force / force_scale)² × dt.
    # Head is 3× more vulnerable than torso.  The quadratic threshold means
    # forces near/below force_scale are suppressed, forces well above it are
    # amplified.  With force_scale=100, dt=0.002, a 1000 N hit sustained for
    # one action step (25 substeps, 50 ms) deals 15 HP to head, 5 HP to torso.
    DAMAGE_RULES = {
        'head': 3.0,
        'torso': 1.0,
    }

    def __init__(
        self,
        initial_health: float = 100.0,
        initial_health_a: Optional[float] = None,
        initial_health_b: Optional[float] = None,
        force_scale: float = 100.0,
        phy_step_dt: float = 0.002,
        request_termination_on_ko: bool = True,
        score_log_file: Optional[str] = None,
    ):
        self.initial_health_a = initial_health_a if initial_health_a is not None else initial_health
        self.initial_health_b = initial_health_b if initial_health_b is not None else initial_health
        self.force_scale = force_scale
        self.phy_step_dt = phy_step_dt
        self.request_termination_on_ko = bool(request_termination_on_ko)
        self._action_damage_a = 0.0
        self._action_damage_b = 0.0

        # --- concise per-substep score audit log (constructor-driven) ---
        # One append line per physics substep.  Default None = no logging.
        # Passed via constructor (not env var) so parallel processes in the
        # same environment can log to distinct files.
        self.score_log_file = score_log_file
        self._score_log_handle = None
        self._score_log_total_step = 0
        # per-substep damage broken down by defender × part (head/torso)
        self._step_dmg_a_head = 0.0
        self._step_dmg_a_torso = 0.0
        self._step_dmg_b_head = 0.0
        self._step_dmg_b_torso = 0.0
        if self.score_log_file:
            parent = os.path.dirname(self.score_log_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # line-buffered: each '\n' triggers a flush so the audit trail
            # survives even without explicit close.
            self._score_log_handle = open(
                self.score_log_file, 'a', buffering=1, encoding='utf-8',
            )
            self._score_log_handle.write(
                '# combat_score_log columns: '
                'step episode_step physics_step health_a health_b '
                'dmg_a_head dmg_a_torso dmg_b_head dmg_b_torso\n'
            )

        # --- debug logging (enabled via COMBAT_SCORE_DEBUG_FILE env var) ---
        self._debug_file: Optional[str] = os.environ.get('COMBAT_SCORE_DEBUG_FILE')
        if self._debug_file:
            parent = os.path.dirname(self._debug_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Create the file if it doesn't exist; append mode preserves content.
            with open(self._debug_file, 'a'):
                pass
            self._debug_episode = 0
            self._debug_log('plugin_init',
                            force_scale=self.force_scale,
                            phy_step_dt=self.phy_step_dt,
                            damage_rules=self.DAMAGE_RULES)

    def _debug_log(self, event: str, **data) -> None:
        """Append a JSON line to the debug file.  No-op if debug is off."""
        if not self._debug_file:
            return
        record = {
            'ts': time.time(),
            'event': event,
            **data,
        }
        try:
            with open(self._debug_file, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except OSError:
            pass

    def __del__(self):
        # Best-effort close of the score-log handle.  Line-buffered mode
        # already flushes per line, so this is mainly about releasing the
        # descriptor promptly in long-running daemons.
        h = getattr(self, '_score_log_handle', None)
        if h is not None:
            try:
                h.close()
            except Exception:
                pass

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "initial_health": self.initial_health_a,
            "initial_health_a": self.initial_health_a,
            "initial_health_b": self.initial_health_b,
            "force_scale": self.force_scale,
            "phy_step_dt": self.phy_step_dt,
            "request_termination_on_ko": self.request_termination_on_ko,
            "score_log_file": self.score_log_file,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "CombatScoringPlugin":
        return cls(**config)

    @property
    def name(self) -> str:
        return "combat_scoring"

    @property
    def priority(self) -> int:
        return OBSERVER_DISPATCHER_PRIORITY + 1

    def on_pre_episode(self, ctx: SimContext) -> None:
        opts = ctx.episode_options
        ctx.metrics['health_a'] = float(
            opts.get('initial_health_a', self.initial_health_a)
        )
        ctx.metrics['health_b'] = float(
            opts.get('initial_health_b', self.initial_health_b)
        )
        ctx.metrics['damage_taken_a'] = 0.0
        ctx.metrics['damage_taken_b'] = 0.0
        while len(ctx.events) > 0:
            ctx.events.pop()
        self._score_log_total_step = 0

        if self._debug_file:
            self._debug_episode += 1
            self._debug_log('pre_episode',
                            episode=self._debug_episode,
                            health_a=ctx.metrics['health_a'],
                            health_b=ctx.metrics['health_b'],
                            episode_options=dict(opts))

    def on_pre_action_step(self, ctx: SimContext) -> None:
        self._action_damage_a = 0.0
        self._action_damage_b = 0.0
        self._debug_log('pre_action_step',
                        episode_step=ctx.episode_step,
                        health_a=ctx.metrics.get('health_a', 0),
                        health_b=ctx.metrics.get('health_b', 0))

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

    def _resolve_hit(self, body_a_name, body_b_name):
        """Classify a contact into (damage_part, defender) or None.

        Returns the damage rule key ('head' / 'torso') and which robot is
        the defender, based on which body is the attack part vs target part
        and the team suffix of the target body.
        """
        # Must be cross-team (robot_a vs robot_b)
        a_is_a = body_a_name.endswith('_a') or '_red' in body_a_name
        a_is_b = body_a_name.endswith('_b') or '_blue' in body_a_name
        b_is_a = body_b_name.endswith('_a') or '_red' in body_b_name
        b_is_b = body_b_name.endswith('_b') or '_blue' in body_b_name
        if not ((a_is_a and b_is_b) or (a_is_b and b_is_a)):
            return None

        cat_a = self._get_part_category(body_a_name)
        cat_b = self._get_part_category(body_b_name)

        a_attacks = cat_a in self.ATTACK_PARTS
        b_attacks = cat_b in self.ATTACK_PARTS
        a_target = cat_a in self.DAMAGE_TARGET_PARTS
        b_target = cat_b in self.DAMAGE_TARGET_PARTS

        # Identify which body is the target (defender's part being hit)
        if a_attacks and b_target:
            target_name = body_b_name
            hit_cat = cat_b
        elif b_attacks and a_target:
            target_name = body_a_name
            hit_cat = cat_a
        else:
            return None

        # Map hit category to damage rule key
        if hit_cat == 'head':
            damage_part = 'head'
        elif hit_cat in ('torso', 'waist_upper', 'waist_lower'):
            damage_part = 'torso'
        else:
            return None

        # Defender robot from the target body's team suffix
        if target_name.endswith('_a') or '_red' in target_name:
            defender = 'robot_a'
        elif target_name.endswith('_b') or '_blue' in target_name:
            defender = 'robot_b'
        else:
            return None

        return damage_part, defender

    def on_post_phy_step(self, ctx: SimContext) -> None:
        """Per-substep damage: (force / force_scale)² × dt × part_weight."""
        self._step_dmg_a_head = 0.0
        self._step_dmg_a_torso = 0.0
        self._step_dmg_b_head = 0.0
        self._step_dmg_b_torso = 0.0
        contacts = ctx.accessor.get_derived_state().get('robot_robot_contacts', [])
        debug_contacts = [] if self._debug_file else None

        for contact in contacts:
            force = contact.get('force', 0.0)
            if force <= 0:
                continue

            body_a = contact.get('body_a', '')
            body_b = contact.get('body_b', '')
            result = self._resolve_hit(body_a, body_b)

            if result is None:
                if debug_contacts is not None:
                    debug_contacts.append({
                        'body_a': body_a, 'body_b': body_b,
                        'force': force, 'resolved': False,
                    })
                continue

            damage_part, defender = result
            part_weight = self.DAMAGE_RULES.get(damage_part, 0.0)
            if part_weight <= 0:
                continue

            damage = part_weight * (force / self.force_scale) ** 2 * self.phy_step_dt
            if damage <= 0:
                continue

            health_key = 'health_a' if defender == 'robot_a' else 'health_b'
            damage_key = 'damage_taken_a' if defender == 'robot_a' else 'damage_taken_b'

            ctx.metrics[health_key] = max(0.0, ctx.metrics[health_key] - damage)
            ctx.metrics[damage_key] += damage

            if defender == 'robot_a':
                self._action_damage_a += damage
                if damage_part == 'head':
                    self._step_dmg_a_head += damage
                else:
                    self._step_dmg_a_torso += damage
            else:
                self._action_damage_b += damage
                if damage_part == 'head':
                    self._step_dmg_b_head += damage
                else:
                    self._step_dmg_b_torso += damage

            if debug_contacts is not None:
                debug_contacts.append({
                    'body_a': body_a, 'body_b': body_b,
                    'force': force, 'resolved': True,
                    'damage_part': damage_part, 'defender': defender,
                    'damage': damage,
                    'health_a': ctx.metrics['health_a'],
                    'health_b': ctx.metrics['health_b'],
                })

        if debug_contacts is not None:
            self._debug_log('post_phy_step',
                            episode_step=ctx.episode_step,
                            physics_step=ctx.physics_step,
                            num_contacts=len(contacts),
                            contacts=debug_contacts,
                            health_a=ctx.metrics.get('health_a', 0),
                            health_b=ctx.metrics.get('health_b', 0),
                            action_damage_a=self._action_damage_a,
                            action_damage_b=self._action_damage_b)

        # Concise per-substep audit log: one line, space-separated.
        if self._score_log_handle is not None:
            self._score_log_total_step += 1
            self._score_log_handle.write(
                f'{self._score_log_total_step} {ctx.episode_step} '
                f'{ctx.physics_step} '
                f'{ctx.metrics.get("health_a", 0):.4g} '
                f'{ctx.metrics.get("health_b", 0):.4g} '
                f'{self._step_dmg_a_head:.4g} {self._step_dmg_a_torso:.4g} '
                f'{self._step_dmg_b_head:.4g} {self._step_dmg_b_torso:.4g}\n'
            )

    def on_post_action_step(self, ctx: SimContext) -> None:
        """Record hit events and check KO (once per action step)."""
        if self._action_damage_a > 0.001:
            ctx.events.append({
                'type': 'hit',
                'defender': 'robot_a',
                'damage': round(self._action_damage_a, 2),
            })
        if self._action_damage_b > 0.001:
            ctx.events.append({
                'type': 'hit',
                'defender': 'robot_b',
                'damage': round(self._action_damage_b, 2),
            })

        ko = False
        if self.request_termination_on_ko:
            if ctx.metrics['health_a'] <= 0 or ctx.metrics['health_b'] <= 0:
                ko = True
                ctx.request_termination(TerminationReason.KO)

        self._debug_log('post_action_step',
                        episode_step=ctx.episode_step,
                        action_damage_a=self._action_damage_a,
                        action_damage_b=self._action_damage_b,
                        health_a=ctx.metrics.get('health_a', 0),
                        health_b=ctx.metrics.get('health_b', 0),
                        damage_taken_a=ctx.metrics.get('damage_taken_a', 0),
                        damage_taken_b=ctx.metrics.get('damage_taken_b', 0),
                        events=list(ctx.events),
                        ko=ko)


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

    def to_blueprint(self) -> Dict[str, Any]:
        return {"frozen_robot_id": self.frozen_robot_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "FrozenRobotPlugin":
        return cls(**config)

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
