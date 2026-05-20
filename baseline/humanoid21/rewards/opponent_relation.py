"""Opponent relation reward plugin for humanoid21.

Provides:
  * :class:`OpponentRelationRewarder` — Reward based on proximity and
    heading toward opponent, with attribution-safe design.

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
CONTROL_FREQUENCY = 20

# Opponent-relation reward (课程二：接近并朝向对手)
# 两个损失项都采用"容忍区间内无损失，超出后线性增长"：
# 1) 距离损失：距离在 [min, max] 内为 0，超出则线性惩罚
# 2) 朝向损失：朝向误差角 <= max_angle 时为 0，超出则线性惩罚
OPP_REL_DIST_MIN = 1.0
OPP_REL_DIST_MAX = 2.2
OPP_REL_DIST_LINEAR_RANGE = 1.0
OPP_REL_HEADING_MAX_ANGLE_DEG = 25.0
OPP_REL_HEADING_LINEAR_RANGE_DEG = 45.0
OPP_REL_DIST_PENALTY_COEF = 1.0
OPP_REL_HEADING_PENALTY_COEF = 1.0


class OpponentRelationRewarder(BaseObserverPlugin):
    """对手相对关系奖励 —— **归因安全版（2026-05-08 重写）**。

    设计目标：课程二里鼓励"主动接近并朝向对手"，同时在已经进入
    合适距离后保持在该区间。

    **旧版的归因漏洞**：旧实现直接读 ``distance = ||opp - self||``，
    当对手朝我走过来时 ``distance`` 也会减小，即使我站着不动也"白得"
    奖励。在两 agent 共享同一 policy 的自博弈训练里，这会产生"互等
    对方"的退化均衡。

    **新版本**：距离信号只依赖智能体自身的速度在"对手方向"上的投影
    （closing velocity）：

      - ``distance > dist_max`` （太远）:
          ``dist_signal = clip(closing_vel / typical_closing_speed, -1, 1)``
          主动接近得正奖励，被动远离（被对手拉开）不扣分，主动远离扣分。
      - ``distance ∈ [dist_min, dist_max]`` （在区间内）:
          ``dist_signal = +1``  恒定正奖励，鼓励"留在格斗区"。
      - ``distance < dist_min`` （太近）:
          ``dist_signal = clip(-closing_vel / typical_closing_speed, -1, 1)``
          主动拉开得正奖励。

    速度项天然奖励"尽快进入格斗区"：出界时间越长，错过的 +1 越多。

    朝向惩罚保持原样 —— 朝向永远由智能体自身控制，没有归因问题。

    每步输出范围粗略为 ``[-dist_coef - heading_coef, +dist_coef]``。
    暴露 ``.in_range`` 布尔属性给上游观察者读取（用于课程门控的
    ``in_range_steps`` 统计）。
    """

    def __init__(
        self,
        agent_id: str,
        dist_min: float = OPP_REL_DIST_MIN,
        dist_max: float = OPP_REL_DIST_MAX,
        dist_linear_range: float = OPP_REL_DIST_LINEAR_RANGE,
        heading_max_angle_deg: float = OPP_REL_HEADING_MAX_ANGLE_DEG,
        heading_linear_range_deg: float = OPP_REL_HEADING_LINEAR_RANGE_DEG,
        dist_penalty_coef: float = OPP_REL_DIST_PENALTY_COEF,
        heading_penalty_coef: float = OPP_REL_HEADING_PENALTY_COEF,
        typical_closing_speed: float = 1.0,
    ) -> None:
        self.agent_id = str(agent_id)
        self.opponent_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self.dist_min = float(dist_min)
        self.dist_max = float(dist_max)
        # ``dist_linear_range`` is no longer used for the (deprecated)
        # distance-excess penalty but kept as a constructor arg for
        # backward compatibility with callers that pass it positionally.
        self.dist_linear_range = max(1e-6, float(dist_linear_range))
        self.heading_max_angle_deg = float(heading_max_angle_deg)
        self.heading_linear_range_deg = max(1e-6, float(heading_linear_range_deg))
        self.dist_penalty_coef = float(dist_penalty_coef)
        self.heading_penalty_coef = float(heading_penalty_coef)
        self.typical_closing_speed = max(1e-3, float(typical_closing_speed))
        self._dt = 1.0 / CONTROL_FREQUENCY
        self._prev_self_xy: Optional[np.ndarray] = None
        # ``in_range`` -> distance ∈ [dist_min, dist_max] (geometric only).
        # ``in_non_penalty_zone`` -> in_range AND heading angle within
        #   ``heading_max_angle_deg`` of the opponent direction.
        # The latter is what the curriculum gate uses to decide whether
        # "final stance" qualifies for stage 3.
        self.in_range: bool = False
        self.in_non_penalty_zone: bool = False
        self._output: float = 0.0

    @staticmethod
    def _robot_forward_xy_from_root_rot(root_rot_wxyz: np.ndarray) -> np.ndarray:
        """从四元数 [w, x, y, z] 计算机体前向在地平面的单位向量。"""
        q = np.asarray(root_rot_wxyz, dtype=np.float64).reshape(-1)
        if q.shape[0] != 4:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 1e-8:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        w, x, y, z = (q / norm).tolist()
        # 旋转矩阵第一列（本地 x 轴在世界系中的方向）
        fx = 1.0 - 2.0 * (y * y + z * z)
        fy = 2.0 * (x * y + w * z)
        fxy = np.asarray([fx, fy], dtype=np.float64)
        f_norm = float(np.linalg.norm(fxy))
        if f_norm < 1e-8:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        return fxy / f_norm

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._prev_self_xy = None
        self.in_range = False
        self.in_non_penalty_zone = False
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        self_state = core_state[self.agent_id]
        opp_state = core_state[self.opponent_id]
        # Forward direction comes from the *torso* body via the balance
        # plugin's derived state. The pelvis/root-body quaternion (which
        # an earlier version of this code used directly) does NOT share
        # axes with the torso on the humanoid21 model — its local +X is
        # roughly sideways, which caused the policy to converge on a
        # "stand sideways to the opponent" optimum that satisfied the
        # 25° heading band on the wrong axis. We now read the same
        # ``robot_forward_ground_direction`` (length-2, ground-plane,
        # unit norm) that ``BalanceAnalysisPlugin`` exposes.
        derived_state = ctx.accessor.get_derived_state()
        self_derived = derived_state.get(self.agent_id, {})
        forward_xy_derived = self_derived.get("robot_forward_ground_direction")

        self_xy = np.asarray(self_state["root_pos"][:2], dtype=np.float64)
        opp_xy = np.asarray(opp_state["root_pos"][:2], dtype=np.float64)
        delta_xy = opp_xy - self_xy
        distance = float(np.linalg.norm(delta_xy))
        self.in_range = self.dist_min <= distance <= self.dist_max

        # ---- 1) Attribution-safe distance signal ---------------------
        # Self-velocity (finite-difference on own root position only —
        # opponent motion does NOT enter this term).
        if self._prev_self_xy is None:
            v_self = np.zeros(2, dtype=np.float64)
        else:
            v_self = (self_xy - self._prev_self_xy) / self._dt
        self._prev_self_xy = self_xy.copy()

        if distance < 1e-6:
            closing_vel = 0.0
        else:
            to_opp_unit = delta_xy / distance
            closing_vel = float(np.dot(v_self, to_opp_unit))

        if self.in_range:
            # Reward being in the fight zone. Constant +1 means "each
            # in-range step collects the full bonus"; this is what makes
            # "arrive fast" optimal (every step wasted out-of-range is
            # one step of +1 left on the table).
            dist_signal = 1.0
        elif distance > self.dist_max:
            # Too far: reward positive closing velocity; penalize retreat.
            dist_signal = float(np.clip(
                closing_vel / self.typical_closing_speed, -1.0, 1.0,
            ))
        else:  # distance < dist_min
            # Too close: reward retreat (negative closing vel).
            dist_signal = float(np.clip(
                -closing_vel / self.typical_closing_speed, -1.0, 1.0,
            ))

        # ---- 2) Heading penalty (unchanged; always self-attributed) --
        if distance < 1e-6:
            heading_penalty = 0.0
            heading_in_zone = True
        else:
            to_opp_unit = delta_xy / distance
            if forward_xy_derived is not None:
                # Preferred: torso-derived ground-plane forward (same as
                # the rest of the codebase). Already unit-normalized,
                # but we re-normalize defensively in case of nan/inf.
                fxy = np.asarray(forward_xy_derived, dtype=np.float64).reshape(-1)[:2]
                f_norm = float(np.linalg.norm(fxy))
                if f_norm > 1e-8:
                    forward_unit = fxy / f_norm
                else:
                    forward_unit = self._robot_forward_xy_from_root_rot(
                        np.asarray(self_state["root_rot"], dtype=np.float64)
                    )
            else:
                # Fallback for envs without BalanceAnalysisPlugin in the
                # derived-state pipeline (e.g. minimal unit-test stubs).
                forward_unit = self._robot_forward_xy_from_root_rot(
                    np.asarray(self_state["root_rot"], dtype=np.float64)
                )
            cosang = float(np.clip(np.dot(forward_unit, to_opp_unit), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cosang)))
            angle_excess = max(0.0, angle_deg - self.heading_max_angle_deg)
            heading_penalty = min(angle_excess / self.heading_linear_range_deg, 1.0)
            heading_in_zone = angle_deg <= self.heading_max_angle_deg

        # "non-penalty zone" = both distance AND heading are in the
        # tolerance band (i.e. r2 has zero penalty contribution from
        # both terms; the dist_signal is still +1 because in_range).
        self.in_non_penalty_zone = bool(self.in_range and heading_in_zone)

        self._output = float(
            self.dist_penalty_coef * dist_signal
            - self.heading_penalty_coef * heading_penalty
        )

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "OpponentRelationRewarder":
        return cls(**config)

