"""Action-limit (joint-pose) reward plugin for humanoid21.

Provides:
  * :class:`ActionLimitRewarder` — potential-based shaping reward that
    keeps the robot's joint configuration close to its initial standing
    pose, discouraging contorted / unnatural postures while still
    allowing normal locomotion within a tolerance band.

Motivation
----------
The robot may learn to alternate foot support (good) but with grossly
twisted joint angles (bad). We anchor on the per-episode **initial
pose** (a natural standing posture) and penalize how far the current
joint configuration drifts from it.

Reward shape (mirrors :class:`OpponentRelationRewarder`)
--------------------------------------------------------
Define a potential over the mean absolute joint deviation::

    dev   = mean(|joint_pos_norm(s) - joint_pos_norm(s_0)|)
    Phi(s) = -max(0, dev - dev_max) * penalty_coef     # <= 0, 0 inside band

The per-step reward is a potential *difference* plus a small persistent
*level* term::

    r_t = shaping_gamma * Phi(s_t) - Phi(s_{t-1})  +  level_coef * Phi(s_t)

The difference term credits returning toward the neutral pose and
penalizes drifting away (telescoping: only net drift matters). The
level term does NOT telescope: while the pose stays contorted it is a
persistent small penalty, giving continuous pressure to stay natural —
fixing the "hold a twisted pose forever for ~0 net reward" blind spot.

Hook conventions
----------------
Observers use the framework's canonical dispatch hooks:
``on_pre_episode`` / ``on_post_action_step``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Deviation tolerance band: mean absolute per-joint deviation (in the
# normalized joint-position space) within this value => no penalty.
# Wide enough to permit natural locomotion, tight enough to punish
# extreme contortion. TUNE against observed walking deviations.
ACTION_LIMIT_DEV_MAX = 0.5
# Linear penalty slope applied to the deviation excess beyond dev_max.
ACTION_LIMIT_PENALTY_COEF = 1.0
# Potential-difference discount. 1.0 = pure geometric difference
# Phi(s_t) - Phi(s_{t-1}); set equal to this reward's critic gamma to
# match PBRS theory.
ACTION_LIMIT_SHAPING_GAMMA = 1.0
# Persistent level-term coefficient alpha (see module docstring).
# 0.0 => pure potential difference.
ACTION_LIMIT_LEVEL_COEF = 0.05


class ActionLimitRewarder(BaseObserverPlugin):
    """关节角度限制奖励（基于初始姿态的势函数 / potential-based shaping）。

    以每个 episode 的**初始姿态**（自然站姿）为基准，惩罚当前关节配置
    相对基准的偏移，抑制扭曲/非常规动作，同时在容差带内允许正常运动。

    势函数（势能越高越理想，<=0）::

        dev    = mean(|joint_pos_norm(s) - joint_pos_norm(s_0)|)
        Phi(s) = -max(0, dev - dev_max) * penalty_coef

    即：平均关节偏移在 ``dev_max`` 内 Phi=0（自然区，无惩罚），超出则线性变负。

    每步输出 = **势差** + 小的**持续水平项**::

        r_t = shaping_gamma * Phi(s_t) - Phi(s_{t-1})  +  level_coef * Phi(s_t)

    势差项奖励“回归自然姿态”、惩罚“偏离”，但会 telescoping（只看净偏移）；
    水平项不 telescope，对“持续保持扭曲姿态”施加持续小惩罚，弥补盲区。
    ``level_coef=0`` 退回纯势差。

    暴露 ``.within_limit`` 布尔属性（当前是否在容差带内）。
    """

    def __init__(
        self,
        agent_id: str,
        dev_max: float = ACTION_LIMIT_DEV_MAX,
        penalty_coef: float = ACTION_LIMIT_PENALTY_COEF,
        shaping_gamma: float = ACTION_LIMIT_SHAPING_GAMMA,
        level_coef: float = ACTION_LIMIT_LEVEL_COEF,
    ) -> None:
        self.agent_id = str(agent_id)
        self.dev_max = float(dev_max)
        self.penalty_coef = float(penalty_coef)
        self.shaping_gamma = float(shaping_gamma)
        self.level_coef = float(level_coef)
        self.within_limit: bool = True
        self._reference_joint_pos: Optional[np.ndarray] = None
        self._output: float = 0.0
        self._prev_phi: float = 0.0

    def _read_joint_pos(self, ctx: ReadOnlySimContext) -> np.ndarray:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        return np.asarray(core_state["joint_pos_norm"], dtype=np.float64).reshape(-1)

    def _compute_phi(self, ctx: ReadOnlySimContext) -> float:
        """当前状态的势能 Phi(s) = -max(0, dev - dev_max) * coef，并刷新 within_limit。"""
        joint_pos = self._read_joint_pos(ctx)
        if self._reference_joint_pos is None:
            # Guard: if pre-episode never ran, anchor on the first observed pose.
            self._reference_joint_pos = joint_pos.copy()
        dev = float(np.mean(np.abs(joint_pos - self._reference_joint_pos)))
        dev_excess = max(0.0, dev - self.dev_max)
        self.within_limit = dev_excess == 0.0
        return float(-(dev_excess * self.penalty_coef))

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self.within_limit = True
        self._output = 0.0
        self._reference_joint_pos = None
        # 以初始姿态为基准，并初始化 Phi_prev（基准处 dev=0 -> Phi=0）。
        try:
            self._reference_joint_pos = self._read_joint_pos(ctx).copy()
            self._prev_phi = self._compute_phi(ctx)
        except Exception:
            self._prev_phi = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        phi = self._compute_phi(ctx)
        # r_t = gamma * Phi(s_t) - Phi(s_{t-1})  +  level_coef * Phi(s_t)
        self._output = float(
            self.shaping_gamma * phi - self._prev_phi + self.level_coef * phi
        )
        self._prev_phi = phi

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "dev_max": self.dev_max,
            "penalty_coef": self.penalty_coef,
            "shaping_gamma": self.shaping_gamma,
            "level_coef": self.level_coef,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ActionLimitRewarder":
        return cls(**config)
