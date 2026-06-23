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
    ) -> None:
        self.agent_id = str(agent_id)
        self.initial_grace_steps = int(initial_grace_steps)
        self.initial_penalty_coef = float(initial_penalty_coef)
        self.foot_lift_min_steps = int(foot_lift_min_steps)
        self.foot_lift_penalty_coef = float(foot_lift_penalty_coef)
        self.switch_interval_max_steps = max(0, int(switch_interval_max_steps))
        self.switch_interval_penalty_coef = float(switch_interval_penalty_coef)

        # 状态变量
        self._state: str = self.STATE_WAIT_FIRST_SINGLE_SUPPORT
        self._state_timer: int = 0
        self._current_support_foot: Optional[str] = None  # 'left' or 'right'
        self._current_support_steps: int = 0
        self._switch_anchor_foot: Optional[str] = None  # 'left' or 'right'
        self._switch_interval_steps: int = 0
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
        self._output = 0.0
        # 获取地面 geom 名称
        static_data = ctx.accessor.get_static_data()
        self._ground_geom_name = static_data.get('ground_geom_name', 'ground')

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        """计算每步奖励"""
        reward = 0.0

        # 检测双脚接触状态
        left_foot_contact, right_foot_contact = self._get_foot_contact_state(ctx)

        if self._state == self.STATE_WAIT_FIRST_SINGLE_SUPPORT:
            reward = self._handle_wait_first_single_support(left_foot_contact, right_foot_contact)
        elif self._state == self.STATE_TRACKING:
            reward = self._handle_tracking(left_foot_contact, right_foot_contact)

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
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "CrossSupportBalanceRewarder":
        return cls(**config)

    def _get_foot_contact_state(self, ctx: ReadOnlySimContext) -> tuple[bool, bool]:
        """检测双脚是否与地面接触（无力阈值：有接触条目即算）。

        仅使用 ``derived_state['robot_environment_contacts']``，与
        ``get_static_data()['ground_geom_name']`` 中的地面 geom 名匹配。

        Returns:
            (left_foot_contact, right_foot_contact)
        """
        derived_state = ctx.accessor.get_derived_state()
        env_contacts = derived_state.get("robot_environment_contacts", [])

        robot_suffix = '_a' if self.agent_id == 'robot_a' else '_b'
        left_foot_body = f"foot_left{robot_suffix}"
        right_foot_body = f"foot_right{robot_suffix}"
        ground_geom = self._ground_geom_name or "ground"

        left_foot_contact = False
        right_foot_contact = False

        for contact in env_contacts:
            if contact.get("robot") != self.agent_id:
                continue
            env_geom = contact.get("environment_geom", "") or ""
            if env_geom != ground_geom:
                continue
            body = contact.get("body", "") or ""
            if body == left_foot_body:
                left_foot_contact = True
            elif body == right_foot_body:
                right_foot_contact = True

        return left_foot_contact, right_foot_contact

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
        self, left_foot_contact: bool, right_foot_contact: bool
    ) -> float:
        """追踪单脚支撑短时惩罚与换脚间隔区间惩罚。"""
        reward = 0.0
        current_single_support = self._single_support_foot(left_foot_contact, right_foot_contact)

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

