"""Punch-motion alternating reward plugin for humanoid21.

Provides:
  * :class:`PunchMotionRewarder` — State-machine reward that encourages
    alternating left/right punches toward the opponent.

Hook conventions
----------------
Observers use the framework's canonical dispatch hooks:
``on_pre_episode`` / ``on_post_action_step``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


# Punch-motion (交替出拳) 训练参数
#
# 距离阈值
# PUNCH_REACH (默认 0.5 m)
#   手到对手躯干距离 < 此值 → "靠近对手"（拳头前出）
# RETRACT_DISTANCE (默认 0.3 m)
#   手到自己躯干距离 < 此值 → "收回身旁"
#
# 步数阈值 (控制频率 20 Hz, 50 ms/步)
# INITIAL_GRACE_STEPS (默认 30, 约 1.5 s)
#   复位后允许「尚未出现第一次前出」的等待步数上限，再长则按 initial 惩罚。
# FORWARD_MIN_STEPS (默认 3, 约 150 ms)
#   每次拳头前出至少停留的步数，过短则惩罚（防 flicking / 高频抽动）。
# SWITCH_INTERVAL_MAX_STEPS (默认 15, 约 0.75 s)
#   交替换拳间隔的最大步数，超过则惩罚（防出拳后不换手）。
#
# 惩罚系数
# INITIAL_PENALTY_COEF (默认 0.25)
# FORWARD_MIN_PENALTY_COEF (默认 0.3)
# SWITCH_INTERVAL_PENALTY_COEF (默认 0.25)
PUNCH_REACH = 0.5
RETRACT_DISTANCE = 0.3
INITIAL_GRACE_STEPS = 30
INITIAL_PENALTY_COEF = 0.25
FORWARD_MIN_STEPS = 3
FORWARD_MIN_PENALTY_COEF = 0.3
SWITCH_INTERVAL_MAX_STEPS = 15
SWITCH_INTERVAL_PENALTY_COEF = 0.25


class PunchMotionRewarder(BaseObserverPlugin):
    """交替出拳奖励插件（状态机版）。

    状态机完全参照 :class:`CrossSupportBalanceRewarder` 的设计：

    ``WAIT_FIRST_PUNCH`` — 等待第一次拳头前出（左拳或右拳靠近对手 + 另一只收回）。
      超过 ``initial_grace_steps`` 开始线性惩罚。

    ``TRACKING`` — 进入交替出拳循环，追踪两项原子指标：
      1) 前出时长（单次段落）：只惩罚过短（防 flicking），不惩罚过长。
      2) 换拳间隔（A → B）：超过 ``switch_interval_max_steps`` 则惩罚。

    其中 A → B 间隔从 A 拳本轮第一次前出开始计时，中间允许 A 拳再次前出，
    直到第一次出现 B 拳前出。
    """

    STATE_WAIT_FIRST_PUNCH = "wait_first_punch"
    STATE_TRACKING = "tracking"

    def __init__(
        self,
        agent_id: str,
        punch_reach: float = PUNCH_REACH,
        retract_distance: float = RETRACT_DISTANCE,
        initial_grace_steps: int = INITIAL_GRACE_STEPS,
        initial_penalty_coef: float = INITIAL_PENALTY_COEF,
        forward_min_steps: int = FORWARD_MIN_STEPS,
        forward_min_penalty_coef: float = FORWARD_MIN_PENALTY_COEF,
        switch_interval_max_steps: int = SWITCH_INTERVAL_MAX_STEPS,
        switch_interval_penalty_coef: float = SWITCH_INTERVAL_PENALTY_COEF,
    ) -> None:
        self.agent_id = str(agent_id)
        self.opp_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self.punch_reach = float(punch_reach)
        self.retract_distance = float(retract_distance)
        self.initial_grace_steps = int(initial_grace_steps)
        self.initial_penalty_coef = float(initial_penalty_coef)
        self.forward_min_steps = int(forward_min_steps)
        self.forward_min_penalty_coef = float(forward_min_penalty_coef)
        self.switch_interval_max_steps = max(0, int(switch_interval_max_steps))
        self.switch_interval_penalty_coef = float(switch_interval_penalty_coef)

        # 状态变量
        self._state: str = self.STATE_WAIT_FIRST_PUNCH
        self._state_timer: int = 0
        self._current_forward_fist: Optional[str] = None  # 'left' or 'right'
        self._current_forward_steps: int = 0
        self._switch_anchor_fist: Optional[str] = None  # 'left' or 'right'
        self._switch_interval_steps: int = 0
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        """重置状态"""
        self._state = self.STATE_WAIT_FIRST_PUNCH
        self._state_timer = 0
        self._current_forward_fist = None
        self._current_forward_steps = 0
        self._switch_anchor_fist = None
        self._switch_interval_steps = 0
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        """计算每步奖励"""
        reward = 0.0

        left_forward, right_forward = self._get_fist_state(ctx)

        if self._state == self.STATE_WAIT_FIRST_PUNCH:
            reward = self._handle_wait_first_punch(left_forward, right_forward)
        elif self._state == self.STATE_TRACKING:
            reward = self._handle_tracking(left_forward, right_forward)

        self._output = reward

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "punch_reach": self.punch_reach,
            "retract_distance": self.retract_distance,
            "initial_grace_steps": self.initial_grace_steps,
            "initial_penalty_coef": self.initial_penalty_coef,
            "forward_min_steps": self.forward_min_steps,
            "forward_min_penalty_coef": self.forward_min_penalty_coef,
            "switch_interval_max_steps": self.switch_interval_max_steps,
            "switch_interval_penalty_coef": self.switch_interval_penalty_coef,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "PunchMotionRewarder":
        return cls(**config)

    # ------------------------------------------------------------------
    # 几何检测
    # ------------------------------------------------------------------

    def _get_fist_state(self, ctx: ReadOnlySimContext) -> tuple[bool, bool]:
        """检测双拳前出状态。

        Returns:
            (left_forward, right_forward)
            left_forward  = 左拳靠近对手 and 右拳收回身旁
            right_forward = 右拳靠近对手 and 左拳收回身旁
        """
        derived = ctx.accessor.get_derived_state([self.agent_id, self.opp_id])
        self_view = derived.get(self.agent_id, {})
        opp_view = derived.get(self.opp_id, {})

        body_xpos = self_view.get("body_xpos", {})
        opp_body_xpos = opp_view.get("body_xpos", {})

        self_torso = body_xpos.get("torso")
        left_hand = body_xpos.get("hand_left")
        right_hand = body_xpos.get("hand_right")
        opp_torso = opp_body_xpos.get("torso")

        if self_torso is None or left_hand is None or right_hand is None or opp_torso is None:
            return False, False

        d_left_opp = float(np.linalg.norm(left_hand - opp_torso))
        d_right_opp = float(np.linalg.norm(right_hand - opp_torso))
        d_left_self = float(np.linalg.norm(left_hand - self_torso))
        d_right_self = float(np.linalg.norm(right_hand - self_torso))

        left_near_opp = d_left_opp < self.punch_reach
        right_near_opp = d_right_opp < self.punch_reach
        left_retracted = d_left_self < self.retract_distance
        right_retracted = d_right_self < self.retract_distance

        left_forward = left_near_opp and right_retracted
        right_forward = right_near_opp and left_retracted

        return left_forward, right_forward

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------

    def _begin_tracking(self, fist: str) -> None:
        """第一次进入前出后，初始化追踪器。"""
        self._state = self.STATE_TRACKING
        self._current_forward_fist = fist
        self._current_forward_steps = 1
        self._switch_anchor_fist = fist
        self._switch_interval_steps = 0

    def _handle_wait_first_punch(
        self, left_forward: bool, right_forward: bool
    ) -> float:
        """等待第一次拳头前出。"""
        reward = 0.0
        if left_forward:
            self._begin_tracking("left")
            return reward
        if right_forward:
            self._begin_tracking("right")
            return reward

        self._state_timer += 1
        if self._state_timer > self.initial_grace_steps:
            excess = self._state_timer - self.initial_grace_steps
            denom = max(1, self.initial_grace_steps)
            reward -= self.initial_penalty_coef * min(excess / denom, 1.0)
        return reward

    def _handle_tracking(
        self, left_forward: bool, right_forward: bool
    ) -> float:
        """追踪前出时长惩罚与换拳间隔惩罚。"""
        reward = 0.0
        current_fist = None
        if left_forward:
            current_fist = "left"
        elif right_forward:
            current_fist = "right"

        # A → B 换拳间隔从 A 拳本轮第一次前出开始计时
        self._switch_interval_steps += 1

        # 1) 前出时长：仅惩罚过短（段落结束时结算）
        if self._current_forward_fist is None:
            if current_fist is not None:
                self._current_forward_fist = current_fist
                self._current_forward_steps = 1
        elif current_fist == self._current_forward_fist:
            self._current_forward_steps += 1
        else:
            if self._current_forward_steps < self.forward_min_steps:
                deficit = self.forward_min_steps - self._current_forward_steps
                reward -= self.forward_min_penalty_coef * (
                    deficit / max(1, self.forward_min_steps)
                )
            if current_fist is None:
                self._current_forward_fist = None
                self._current_forward_steps = 0
            else:
                self._current_forward_fist = current_fist
                self._current_forward_steps = 1

        # 2) 换拳间隔：当首次出现 opposite fist forward 时结算并重置锚点
        if (
            current_fist is not None
            and self._switch_anchor_fist is not None
            and current_fist != self._switch_anchor_fist
        ):
            if self._switch_interval_steps > self.switch_interval_max_steps:
                excess = self._switch_interval_steps - self.switch_interval_max_steps
                denom = max(1, self.switch_interval_max_steps)
                reward -= self.switch_interval_penalty_coef * min(excess / denom, 1.0)
            self._switch_anchor_fist = current_fist
            self._switch_interval_steps = 0

        return reward
