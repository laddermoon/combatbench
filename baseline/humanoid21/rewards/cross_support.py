"""Cross-support balance reward plugin for humanoid21.

Provides:
  * :class:`CrossSupportBalanceRewarder` — Reward based on alternating
    single-foot support balance.

Hook conventions
----------------
Observers use the framework's canonical dispatch hooks:
``on_pre_episode`` / ``on_post_action_step``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


# Cross-support balance (交替支撑平衡) 训练参数
# 足底接触：从 derived_state["robot_environment_contacts"] 读取，与 ground geom
# 有接触即视为着地（无力阈值）。
#
# 以下默认步数按本模块 ``CONTROL_FREQUENCY``（当前 20 Hz，约 50 ms/步）设计；
# 若改控制频率，建议按比例缩放各 *_STEPS 环境变量。
#
# CROSS_SUPPORT_INITIAL_GRACE_STEPS（默认 30）
#   复位后允许「尚未出现第一次单脚支撑」的等待步数上限（可双脚着地/双脚离地），
#   再长则按 initial 惩罚。约 1.5 s：给接触与姿态稳定留余量。
# CROSS_SUPPORT_INITIAL_PENALTY_COEF（默认 0.25）
#   第一次单脚支撑前等待过长时的惩罚系数；按超时比例线性增加，封顶 1 倍系数。
# CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS（默认 4）
#   每次单脚支撑中，支撑脚着地时长最小值（约 0.2 s）。
# CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF（默认 0.45）
#   支撑脚着地时长过短时的惩罚强度。
# CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS（默认 18）
#   换支撑脚间隔（A->B）的最大步数（约 0.9 s）。
# CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF（默认 0.25）
#   换支撑脚间隔不在区间内时的惩罚强度。
CROSS_SUPPORT_INITIAL_GRACE_STEPS = 30
CROSS_SUPPORT_INITIAL_PENALTY_COEF = 0.25
CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS = 4
CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF = 0.45
CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS = 18
CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF = 0.25


# Double-support penalty defaults (0 = disabled, backward compatible)
CROSS_SUPPORT_DOUBLE_SUPPORT_MAX_STEPS = 0
CROSS_SUPPORT_DOUBLE_SUPPORT_PENALTY_COEF = 0.0

# Single-support bonus default (0 = disabled, backward compatible)
CROSS_SUPPORT_SINGLE_SUPPORT_BONUS = 0.0
CROSS_SUPPORT_MIN_HEIGHT = 0.0

# Foot lift height threshold: non-support foot's ankle_x joint height
# must exceed STANDING_FOOT_HEIGHT + this value (m) to count as a real
# single-support step.  Prevents micro-lift shuffling.
# Standing ankle_x z ≈ 0.067 m, so threshold = 0.067 + 0.06 = 0.127 m.
#
# STANDING_FOOT_HEIGHT was obtained by measuring the ankle_x joint world
# z-coordinate when the robot is in the default standing pose:
#   sim = Humanoid21Simulator(initial_distance=2.0)
#   sim.reset(seed=42, options={'initial_pose_a': 'standing',
#                                'initial_pose_b': 'standing'})
#   ds = sim.get_derived_state(['robot_a'])
#   jwa = ds['robot_a']['joint_world_anchor']
#   jwa['ankle_x_left_a'][2]  # → 0.067
#   jwa['ankle_x_right_a'][2] # → 0.067
CROSS_SUPPORT_FOOT_LIFT_MIN_HEIGHT = 0.06
CROSS_SUPPORT_STANDING_FOOT_HEIGHT = 0.067


class CrossSupportBalanceRewarder(BaseObserverPlugin):
    """交叉支撑平衡奖励插件（语义归约版）。

    保留初始逻辑：开局到第一次单脚支撑前，超过 ``initial_grace_steps`` 开始惩罚。

    进入单脚支撑后，仅关注两项原子指标：
      1) 单脚支撑时长（单次段落）：只惩罚过短，不惩罚过长
      2) 换支撑脚间隔（A -> B）：超过 ``switch_interval_max_steps`` 则惩罚

    其中 A -> B 间隔从 A 脚本轮第一次单脚支撑开始计时，中间允许出现 A 脚再次单脚支撑，
    直到第一次出现 B 脚单脚支撑。
    """

    STATE_WAIT_FIRST_SINGLE_SUPPORT = "wait_first_single_support"
    STATE_TRACKING = "tracking"

    def __init__(
        self,
        agent_id: str,
        initial_grace_steps: int = CROSS_SUPPORT_INITIAL_GRACE_STEPS,
        initial_penalty_coef: float = CROSS_SUPPORT_INITIAL_PENALTY_COEF,
        foot_lift_min_steps: int = CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS,
        foot_lift_penalty_coef: float = CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF,
        switch_interval_max_steps: int = CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS,
        switch_interval_penalty_coef: float = CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF,
        double_support_max_steps: int = CROSS_SUPPORT_DOUBLE_SUPPORT_MAX_STEPS,
        double_support_penalty_coef: float = CROSS_SUPPORT_DOUBLE_SUPPORT_PENALTY_COEF,
        single_support_bonus: float = CROSS_SUPPORT_SINGLE_SUPPORT_BONUS,
        min_height: float = CROSS_SUPPORT_MIN_HEIGHT,
        foot_lift_min_height: float = CROSS_SUPPORT_FOOT_LIFT_MIN_HEIGHT,
    ) -> None:
        self.agent_id = str(agent_id)
        self.initial_grace_steps = int(initial_grace_steps)
        self.initial_penalty_coef = float(initial_penalty_coef)
        self.foot_lift_min_steps = int(foot_lift_min_steps)
        self.foot_lift_penalty_coef = float(foot_lift_penalty_coef)
        self.switch_interval_max_steps = max(0, int(switch_interval_max_steps))
        self.switch_interval_penalty_coef = float(switch_interval_penalty_coef)
        self.double_support_max_steps = max(0, int(double_support_max_steps))
        self.double_support_penalty_coef = float(double_support_penalty_coef)
        self.single_support_bonus = float(single_support_bonus)
        self.min_height = float(min_height)
        self.foot_lift_min_height = float(foot_lift_min_height)

        # 状态变量
        self._state: str = self.STATE_WAIT_FIRST_SINGLE_SUPPORT
        self._state_timer: int = 0
        self._current_support_foot: Optional[str] = None  # 'left' or 'right'
        self._current_support_steps: int = 0
        self._switch_anchor_foot: Optional[str] = None  # 'left' or 'right'
        self._switch_interval_steps: int = 0
        self._double_support_steps: int = 0
        self._output: float = 0.0
        self._ground_geom_name: Optional[str] = None

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        """重置状态"""
        self._state = self.STATE_WAIT_FIRST_SINGLE_SUPPORT
        self._state_timer = 0
        self._current_support_foot = None
        self._current_support_steps = 0
        self._switch_anchor_foot = None
        self._switch_interval_steps = 0
        self._double_support_steps = 0
        self._output = 0.0
        # 获取地面 geom 名称
        static_data = ctx.accessor.get_static_data()
        if 'ground_geom_name' not in static_data:
            raise KeyError(
                f"on_pre_episode: 'ground_geom_name' not in static_data "
                f"(available={list(static_data.keys())})"
            )
        self._ground_geom_name = static_data['ground_geom_name']

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        """计算每步奖励"""
        reward = 0.0

        # 检测双脚接触状态
        left_foot_contact, right_foot_contact = self._get_foot_contact_state(ctx)

        # Get robot height for gating single-support bonus
        if self.min_height > 0.0:
            core_state = ctx.accessor.get_core_state()
            if self.agent_id not in core_state:
                raise KeyError(
                    f"on_post_action_step: '{self.agent_id}' not in "
                    f"core_state (available={list(core_state.keys())})"
                )
            cs = core_state[self.agent_id]
            if 'root_pos' not in cs:
                raise KeyError(
                    f"on_post_action_step: 'root_pos' not in "
                    f"core_state['{self.agent_id}'] "
                    f"(available={list(cs.keys())})"
                )
            root_height = float(cs['root_pos'][2])
        else:
            root_height = 1.0  # no gate

        # Double-support penalty: penalize prolonged static double-support
        # to force the robot to step alternately.
        # Gated by min_height: don't penalize during get-up phase.
        if self.double_support_max_steps > 0 and root_height >= self.min_height:
            if left_foot_contact and right_foot_contact:
                self._double_support_steps += 1
                if self._double_support_steps > self.double_support_max_steps:
                    excess = self._double_support_steps - self.double_support_max_steps
                    denom = max(1, self.double_support_max_steps)
                    reward -= self.double_support_penalty_coef * min(excess / denom, 1.0)
            else:
                self._double_support_steps = 0

        if self._state == self.STATE_WAIT_FIRST_SINGLE_SUPPORT:
            reward += self._handle_wait_first_single_support(left_foot_contact, right_foot_contact)
        elif self._state == self.STATE_TRACKING:
            reward += self._handle_tracking(left_foot_contact, right_foot_contact, root_height)

        self._output = reward

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "initial_grace_steps": self.initial_grace_steps,
            "initial_penalty_coef": self.initial_penalty_coef,
            "foot_lift_min_steps": self.foot_lift_min_steps,
            "foot_lift_penalty_coef": self.foot_lift_penalty_coef,
            "switch_interval_max_steps": self.switch_interval_max_steps,
            "switch_interval_penalty_coef": self.switch_interval_penalty_coef,
            "double_support_max_steps": self.double_support_max_steps,
            "double_support_penalty_coef": self.double_support_penalty_coef,
            "single_support_bonus": self.single_support_bonus,
            "min_height": self.min_height,
            "foot_lift_min_height": self.foot_lift_min_height,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "CrossSupportBalanceRewarder":
        return cls(**config)

    def _get_foot_contact_state(self, ctx: ReadOnlySimContext) -> tuple[bool, bool]:
        """检测双脚是否与地面接触（无力阈值：有接触条目即算）— 向量化版本。

        使用 ``contacts_vec``，通过 aff 筛选机器人↔环境接触，
        通过 geom_id_to_name 确认地面 geom，通过 body_id_to_name 获取 body 名。

        Returns:
            (left_foot_contact, right_foot_contact)
        """
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get("contacts")

        robot_suffix = '_a' if self.agent_id == 'robot_a' else '_b'
        left_foot_body = f"foot_left{robot_suffix}"
        right_foot_body = f"foot_right{robot_suffix}"
        ground_geom = self._ground_geom_name
        if ground_geom is None:
            raise RuntimeError(
                f"_get_foot_contact_state: _ground_geom_name is None "
                f"(on_pre_episode not called?)"
            )

        left_foot_contact = False
        right_foot_contact = False

        if cv is not None and cv['ncon'] > 0:
            static_data = ctx.accessor.get_static_data()
            if 'body_id_to_name' not in static_data:
                raise KeyError(
                    f"_get_foot_contact_state: 'body_id_to_name' not in "
                    f"static_data (available={list(static_data.keys())})"
                )
            if 'geom_id_to_name' not in static_data:
                raise KeyError(
                    f"_get_foot_contact_state: 'geom_id_to_name' not in "
                    f"static_data (available={list(static_data.keys())})"
                )
            body_id_to_name = static_data['body_id_to_name']
            geom_id_to_name = static_data['geom_id_to_name']

            robot_aff = 1 if self.agent_id == 'robot_a' else 2

            aff1 = cv['aff1']
            aff2 = cv['aff2']
            geom1 = cv['geom1']
            geom2 = cv['geom2']
            body1 = cv['body1']
            body2 = cv['body2']

            for i in range(cv['ncon']):
                if aff1[i] == 0 and aff2[i] == robot_aff:
                    geom_env = geom_id_to_name.get(int(geom1[i]), '')
                    body_robot = body_id_to_name.get(int(body2[i]), '')
                elif aff2[i] == 0 and aff1[i] == robot_aff:
                    geom_env = geom_id_to_name.get(int(geom2[i]), '')
                    body_robot = body_id_to_name.get(int(body1[i]), '')
                else:
                    continue

                if geom_env != ground_geom:
                    continue
                if body_robot == left_foot_body:
                    left_foot_contact = True
                elif body_robot == right_foot_body:
                    right_foot_contact = True

        # --- Foot lift height check ---
        # If a foot is not in contact but its ankle joint height is below
        # STANDING_FOOT_HEIGHT + foot_lift_min_height, treat it as still in
        # contact (micro-lift does not count as a real step).
        if self.foot_lift_min_height > 0.0:
            threshold = CROSS_SUPPORT_STANDING_FOOT_HEIGHT + self.foot_lift_min_height
            left_h, right_h = self._get_foot_min_heights(ctx)
            if not left_foot_contact and left_h < threshold:
                left_foot_contact = True
            if not right_foot_contact and right_h < threshold:
                right_foot_contact = True

        return left_foot_contact, right_foot_contact

    def _get_foot_min_heights(self, ctx: ReadOnlySimContext) -> tuple[float, float]:
        """获取左右脚踝关节（ankle_x）距地面的高度 (z)。

        使用 derived_state["joint_world_anchor"] 获取踝关节世界坐标，
        取 ankle_x 关节的 z 值作为脚的高度。
        """
        robot_suffix = '_a' if self.agent_id == 'robot_a' else '_b'
        left_ankle = f"ankle_x_left{robot_suffix}"
        right_ankle = f"ankle_x_right{robot_suffix}"

        derived_state = ctx.accessor.get_derived_state([self.agent_id])
        robot_state = derived_state.get(self.agent_id, {})
        jwa = robot_state.get('joint_world_anchor', {})

        if left_ankle not in jwa:
            raise KeyError(
                f"_get_foot_min_heights: '{left_ankle}' not in "
                f"joint_world_anchor (agent={self.agent_id}, "
                f"available={list(jwa.keys())})"
            )
        if right_ankle not in jwa:
            raise KeyError(
                f"_get_foot_min_heights: '{right_ankle}' not in "
                f"joint_world_anchor (agent={self.agent_id}, "
                f"available={list(jwa.keys())})"
            )
        left_z = float(jwa[left_ankle][2])
        right_z = float(jwa[right_ankle][2])

        return left_z, right_z

    def _single_support_foot(self, left_foot_contact: bool, right_foot_contact: bool) -> Optional[str]:
        """返回当前是否为单脚支撑：'left' / 'right' / None。"""
        if left_foot_contact and not right_foot_contact:
            return "left"
        if right_foot_contact and not left_foot_contact:
            return "right"
        return None

    def _begin_tracking(self, support_foot: str) -> None:
        """第一次进入单脚支撑后，初始化追踪器。"""
        self._state = self.STATE_TRACKING
        self._current_support_foot = support_foot
        self._current_support_steps = 1
        self._switch_anchor_foot = support_foot
        self._switch_interval_steps = 0

    def _handle_wait_first_single_support(
        self, left_foot_contact: bool, right_foot_contact: bool
    ) -> float:
        """从任意初始接触状态，等待第一次单脚支撑。"""
        reward = 0.0
        support_foot = self._single_support_foot(left_foot_contact, right_foot_contact)
        if support_foot is not None:
            self._begin_tracking(support_foot)
            return reward

        self._state_timer += 1
        if self._state_timer > self.initial_grace_steps:
            excess = self._state_timer - self.initial_grace_steps
            denom = max(1, self.initial_grace_steps)
            reward -= self.initial_penalty_coef * min(excess / denom, 1.0)
        return reward

    def _handle_tracking(
        self, left_foot_contact: bool, right_foot_contact: bool, root_height: float = 1.0
    ) -> float:
        """追踪单脚支撑短时惩罚与换脚间隔区间惩罚。"""
        reward = 0.0
        current_single_support = self._single_support_foot(left_foot_contact, right_foot_contact)

        # Positive bonus for being in single support — creates gradient
        # toward stepping vs static double support.
        # Gated by min_height: only reward stepping when robot is tall enough
        # to prevent squatting + foot-shuffling hack.
        if (
            current_single_support is not None
            and self.single_support_bonus > 0.0
            and root_height >= self.min_height
        ):
            reward += self.single_support_bonus

        # A -> B 换脚间隔从 A 脚本轮第一次单脚开始计时，期间允许 A 再次单脚。
        self._switch_interval_steps += 1

        # 1) 单脚支撑时长：仅惩罚过短（段落结束时结算）
        if self._current_support_foot is None:
            if current_single_support is not None:
                self._current_support_foot = current_single_support
                self._current_support_steps = 1
        elif current_single_support == self._current_support_foot:
            self._current_support_steps += 1
        else:
            if self._current_support_steps < self.foot_lift_min_steps:
                deficit = self.foot_lift_min_steps - self._current_support_steps
                reward -= self.foot_lift_penalty_coef * (deficit / max(1, self.foot_lift_min_steps))
            if current_single_support is None:
                self._current_support_foot = None
                self._current_support_steps = 0
            else:
                self._current_support_foot = current_single_support
                self._current_support_steps = 1

        # 2) 换支撑脚间隔：当首次出现 opposite single support 时结算并重置锚点
        if (
            current_single_support is not None
            and self._switch_anchor_foot is not None
            and current_single_support != self._switch_anchor_foot
        ):
            if self._switch_interval_steps > self.switch_interval_max_steps:
                excess = self._switch_interval_steps - self.switch_interval_max_steps
                denom = max(1, self.switch_interval_max_steps)
                reward -= self.switch_interval_penalty_coef * min(excess / denom, 1.0)
            self._switch_anchor_foot = current_single_support
            self._switch_interval_steps = 0

        return reward

